import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PowerNormalize, Channels
import math
from transformers import AutoModel


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x


def freeze_layers(bert_model, num_layers_to_freeze):
        for layer_num, layer in enumerate(bert_model.encoder.layer):
                if layer_num < num_layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False

class RoBERTaEncoder(nn.Module):
    def __init__(self, d_model, freeze_bert):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.projection = nn.Linear(768, d_model)  # optional: project to your internal dimension

        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # shape [B, 768]
        return self.projection(cls_token)               # shape [B, d_model]

def discrete_probability(y_tilde: torch.Tensor,
                         mu: torch.Tensor,
                         sigma: torch.Tensor,
                         eps: float = 1e-12) -> torch.Tensor:
    # y_tilde, mu, sigma: [B, D]
    # print(y_tilde.shape)
    # print(mu.shape)
    lower = (y_tilde - 0.5 - mu) / (sigma * math.sqrt(2))
    upper = (y_tilde + 0.5 - mu) / (sigma * math.sqrt(2))
    p = 0.5 * (torch.erf(upper) - torch.erf(lower))
    
    return p.clamp(min=eps)
def gumbel_sigmoid(logits, τ=1.0, hard=True):
    """Differentiable binary quantization."""
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y = torch.sigmoid((logits + g) / τ)
    if hard:
        return (y>0.5).float() + (y - y.detach())
    return y

class ChannelEncoderFactory:
    def __init__(self, D, N_s):
        self.D, self.N_s = D, N_s

    def bpsk(self):
        return nn.Sequential(
            nn.Linear(self.D, self.N_s),
            # → [B, N_s] logits
        )

    def qpsk(self):
        return nn.Sequential(
            nn.Linear(self.D, 2*self.N_s),
        )

    def qam16(self):
        return nn.Sequential(
            nn.Linear(self.D, 4*self.N_s),
        )

    def qam64(self):
        return nn.Sequential(
            nn.Linear(self.D, 6*self.N_s),
        )
def map_to_constellation(bits, M):
    """
    bits: Tensor[..., bps] where bps = log2(M)
    returns: FloatTensor[..., 2] (I,Q) per symbol
    """
    # group bits into two halves for I and Q
    b = bits.shape[-1] // 2
    I_bits, Q_bits = bits[..., :b], bits[..., b:]
    # interpret as integer 0..(2^b−1)
    I_int = I_bits.matmul(2**torch.arange(b-1, -1, -1, device=bits.device).float())
    Q_int = Q_bits.matmul(2**torch.arange(b-1, -1, -1, device=bits.device).float())
    # Gray‐map integer → level
    # for 2^b levels spaced at ±(2i+1−2^b)
    L = 2**b
    levels = (2*I_int + 1 - L).unsqueeze(-1)
    levels_Q = (2*Q_int + 1 - L).unsqueeze(-1)
    # normalize average power = 1
    norm = math.sqrt((2*(L**2)-1)/3)
    return torch.cat([levels, levels_Q], dim=-1) / norm
class SimpleChannelDecoder(nn.Module):
    def __init__(self, in_dim, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, D)
        )
    def forward(self, rx):
        # rx: [B, 2*N_s]
        return self.net(rx)
    
class HyperPrior(nn.Module):
    def __init__(self, d_model, num_modulations = 1):  # num_modulations = 3
        super().__init__()
        self.d_model = d_model
        self.num_modulations = num_modulations

        self.encoder = nn.Sequential(
            nn.Linear(d_model + 1, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * d_model + num_modulations)
        )

    def forward(self, y, n_var, training=True):
        B, D = y.shape
        snr_feat = torch.log(1.0 / n_var).unsqueeze(1)  # [B,1]
        z = self.encoder(torch.cat([y, snr_feat], dim=1))  # [B, D]

        z_tilde = z + torch.rand_like(z) - 0.5 if training else torch.round(z)

        params = self.decoder(z_tilde)  # [B, 2D + num_modulations]
        mu, raw_sigma, mod_logits = params.split(
            [D, D, self.num_modulations], dim=1
        )

        sigma = F.softplus(raw_sigma)
        return z_tilde, mu, sigma, mod_logits

class ChannelEncoderSelector(nn.Module):
    def __init__(self, d_model, N_s, constellation_sizes):
        super().__init__()
        factory = ChannelEncoderFactory(d_model, N_s)
        self.encoders = nn.ModuleList([
            factory.qpsk(), factory.qam16(), factory.qam64()
        ])
        self.constellation_sizes = constellation_sizes
        self.N_s = N_s

    def forward(self, y_tilde, mod_probs):
        B = y_tilde.size(0)
        enc_logits = [enc(y_tilde) for enc in self.encoders]
        enc_bits = [gumbel_sigmoid(l, τ=1.0, hard=self.training) for l in enc_logits]

        symbols = []
        for bits, M in zip(enc_bits, self.constellation_sizes):
            bps = int(math.log2(M))
            bits_rs = bits.view(B, self.N_s, bps)
            symbols.append(map_to_constellation(bits_rs, M))  # [B, N_s, 2]

        Txs = [s.view(B, -1) for s in symbols]
        Tx_stack = torch.stack(Txs, dim=-1)
        return PowerNormalize((Tx_stack * mod_probs.unsqueeze(1)).sum(-1))  # [B, 2*N_s]
class ChannelDecoderSelector(nn.Module):
    def __init__(self, input_dim, d_model, K):
        super().__init__()
        self.decoders = nn.ModuleList([SimpleChannelDecoder(input_dim, d_model) for _ in range(K)])

    def forward(self, Rx_sig, mod_probs):
        B = Rx_sig.size(0)
        decs = [dec(Rx_sig) for dec in self.decoders]
        dec_stack = torch.stack(decs, dim=-1)
        return (dec_stack * mod_probs.unsqueeze(1)).sum(-1)
def compute_rate_loss(y_tilde, z_tilde, mu, sigma):
    p_y = discrete_probability(y_tilde, mu, sigma)
    B_y = -torch.log2(p_y).sum(1)

    p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
    B_z = -torch.log2(p_z).sum(1)

    return (B_y + B_z).mean()

def compute_entropy_loss(y_tilde, mu, sigma, z_tilde=None):
    # Main rate loss (B_y)
    p_y = discrete_probability(y_tilde, mu, sigma)
    B_y = -torch.log2(p_y + 1e-9).sum(dim=1)

    # Latent prior entropy (B_z)
    if z_tilde is not None:
        p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
        B_z = -torch.log2(p_z + 1e-9).sum(dim=1)
        return (B_y + B_z).mean(), B_y.mean(), B_z.mean()
    else:
        return B_y.mean(), B_y.mean(), None

class MODJSCC_WithModulation(nn.Module):
    def __init__(self, d_model=256, freeze_bert=False, N_s=64):
        super().__init__()
        self.d_model = d_model
        self.N_s = N_s
        self.M_list = [4]#, 16, 64]
        self.bps_list = [int(math.log2(M)) for M in self.M_list]
        self.K = len(self.M_list)

        self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)

        # === Hyperprior ===
        self.hyper_encoder = nn.Sequential(
            nn.Linear(d_model + 1, 128), nn.ReLU(), nn.Linear(128, d_model)
        )
        self.hyper_decoder = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 2 * d_model + self.K)
        )

        # === Modulation-specific channel encoders and decoders ===
        self.channel_encoders = nn.ModuleList([
            nn.Linear(d_model, N_s * bps) for bps in self.bps_list
        ])
        self.latent_bottleneck = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 8)
        )
        self.decoder_input_proj = nn.Linear(8, 256)
        self.channel_decoders = nn.ModuleList([
            nn.Linear(2 * N_s, d_model) for _ in self.bps_list
        ])

        # === Final classifier ===
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, n_var):
        B = input_ids.size(0)
        device = input_ids.device
        channels = Channels()

        # === 1. Encode semantics ===
        y = self.encoder(input_ids, attention_mask)  # [B, d_model] #Viết thành ma trận

        # === 2. Hyperprior ===
        B = y.size(0)
        if not torch.is_tensor(n_var):
        # scalar n_var → expand to [B, 1]
            snr_feat = torch.full((B, 1), math.log(1.0 / n_var), device=y.device)
        else:
            # tensor n_var → apply log and reshape
            snr_feat = torch.log(1.0 / n_var).view(-1, 1)
  
        z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))
        z_tilde = z + torch.rand_like(z) - 0.5 if self.training else torch.round(z)

        hyper_out = self.hyper_decoder(z_tilde)
        mu, raw_sigma, mod_logits = torch.split(hyper_out, [self.d_model, self.d_model, self.K], dim=1)
        sigma = F.softplus(raw_sigma) + 1e-6
        mod_probs = F.gumbel_softmax(mod_logits, tau=1.0, hard=self.training)  # [B, K]

        # === 3. Quantize y ===


        # y_bottleneck
        y_bottleneck = self.latent_bottleneck(y)        # [B, 64]
        y_proj = self.decoder_input_proj(y_bottleneck)  # [B, 256]
        y_tilde = y_proj + torch.rand_like(y_proj * 2) / 2 - 0.5 if self.training else torch.round(y_proj)


        # === 4. Channel encoding (per modulation) ===
        Tx_list = []
        for i, bps in enumerate(self.bps_list):
            bits = self.channel_encoders[i](y_tilde)           # [B, N_s * bps]
            bits = gumbel_sigmoid(bits, τ=1.0, hard=self.training)
            bits_rs = bits.view(B, self.N_s, bps)
            symbols = map_to_constellation(bits_rs, self.M_list[i])  # [B, N_s, 2]
            Tx_list.append(symbols.view(B, -1))               # [B, 2 * N_s]

        Tx_stack = torch.stack(Tx_list, dim=-1)               # [B, 2*N_s, K]
        Tx = (Tx_stack * mod_probs.unsqueeze(1)).sum(-1)
        Tx = PowerNormalize(Tx)

        # === 5. Channel ===
        Rx = channels.AWGN(Tx, n_var)

        # === 6. Channel decoding ===
        decs = [dec(Rx) for dec in self.channel_decoders]     # list of [B, d_model]
        dec_stack = torch.stack(decs, dim=-1)                 # [B, d_model, K]
        feat = (dec_stack * mod_probs.unsqueeze(1)).sum(-1)   # [B, d_model]

        # === 7. Classification ===
        logits = self.decoder(feat)
        # print("logits shape:", logits.shape)
        # === 8. Rate loss ===
        p_y = discrete_probability(y_tilde, mu, sigma)
        rate_loss = -torch.log2(p_y + 1e-9).sum(dim=1).mean()

        return logits, rate_loss, mod_probs

class SimpleMODJSCC_WithHyper(nn.Module):
    def __init__(self, d_model=256, freeze_bert=False, N_s=64):
        super().__init__()
        self.N_s = N_s
        self.d_model = d_model

        self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)
        self.channel_enc = nn.Linear(d_model, 2 * N_s)
        self.channel_dec = nn.Linear(2 * N_s, d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        # === Hyper encoder + decoder ===
        self.hyper_encoder = nn.Sequential(
            nn.Linear(d_model + 1, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        self.hyper_decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * d_model)
        )

    def forward(self, input_ids, attention_mask, n_var):
        B = input_ids.size(0)
        device = input_ids.device
        channels = Channels()

        # === 1. Semantic encoder ===
        y = self.encoder(input_ids, attention_mask)  # [B, d_model]

        # === 2. Hyperprior to model y ===
        snr_feat = torch.log(1.0 / n_var).unsqueeze(1)  # [B, 1]
        z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))  # [B, d_model]

        if self.training:
            z_tilde = z + torch.rand_like(z) - 0.5
        else:
            z_tilde = torch.round(z)

        mu_sigma = self.hyper_decoder(z_tilde)  # [B, 2*d_model]
        mu, raw_sigma = mu_sigma[:, :self.d_model], mu_sigma[:, self.d_model:]
        sigma = F.softplus(raw_sigma) + 1e-6

        # === 3. Quantize y ===
        if self.training:
            y_tilde = y + torch.rand_like(y) - 0.5
        else:
            y_tilde = torch.round(y)

        # === 4. Rate loss ===
        p_y = discrete_probability(y_tilde, mu, sigma)
        rate_loss = -torch.log2(p_y + 1e-9).sum(dim=1).mean()

        # === 5. Channel simulation ===
        Tx = PowerNormalize(self.channel_enc(y_tilde))  # [B, 2*N_s]
        Rx = channels.AWGN(Tx, n_var)

        # === 6. Channel decoding & classification ===
        feat = self.channel_dec(Rx)  # [B, d_model]
        logits = self.decoder(feat)

        return logits, rate_loss

    
    
    
    


    


