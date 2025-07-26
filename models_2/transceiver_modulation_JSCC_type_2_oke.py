
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_oke import PowerNormalize, Channels
import math
from transformers import AutoModel
from range_coder import RangeEncoder, RangeDecoder
import numpy as np
import os
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
# def entropy_encode(self, bit_symbols, pmf):
#         """
#         Wraps an arithmetic/range encoder to compress integer symbols into a bitstream using the provided pmf.
#         bit_symbols: Tensor of shape [B, L] with integer symbols
#         pmf: Tensor of shape [B, L, C] giving probability of each symbol value in C
#         Returns: list of byte strings, one per batch element.
#         """
#         # Example using the `constriction` library (pip install constriction)
#         from constriction import RangeEncoder
#         encoder = RangeEncoder()
#         bitstreams = []
#         for i in range(bit_symbols.size(0)):
#             symbols = bit_symbols[i].cpu().numpy().tolist()
#             probs = pmf[i].cpu().numpy().tolist()
#             bitstream = encoder.encode(symbols, probs)
#             bitstreams.append(bitstream)
#         return bitstreams

# def entropy_decode(self, bitstreams, pmf, shape):
#         """
#         Inverse of `entropy_encode`: recovers integer symbols from bitstreams.
#         bitstreams: list of byte strings
#         pmf: Tensor of shape [B, L, C] with same probabilities used to encode
#         shape: tuple (B, L)
#         Returns: Tensor of shape [B, L] with decoded integer symbols.
#         """
#         from constriction import RangeDecoder
#         decoder = RangeDecoder()
#         decoded = []
#         for i, bs in enumerate(bitstreams):
#             symbols = decoder.decode(bs, pmf[i].cpu().numpy().tolist())
#             decoded.append(symbols)
#         decoded = torch.tensor(decoded, device=pmf.device)
#         return decoded.view(shape)

# def encode_bits(self, input_ids, attention_mask, n_var):
#         """Produces a compressed bitstream for z and y."""
#         B = input_ids.size(0)
#         # (repeat forward up through quantization)
#         y = self.encoder(input_ids, attention_mask)
#         snr_feat = torch.log(1.0/n_var).view(-1,1) if torch.is_tensor(n_var) else torch.full((B,1), math.log(1.0/n_var), device=y.device)
#         z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))
#         z_tilde = z.round()
#         hyper_out = self.hyper_decoder(z_tilde)
#         _, raw_sigma, _ = torch.split(hyper_out, [self.d_model, self.d_model, self.K], dim=1)
#         sigma_rec = F.softplus(raw_sigma) + 1e-6
#         y_tilde = y.round()

#         # Build symbol tensors
#         z_symbols = z_tilde.long()    # [B, d_model]
#         y_symbols = y_tilde.long()    # [B, d_model]

#         # Compute pmfs for each symbol value
#         # assume discrete values 0..C-1; here C can be max range across both
#         C = 2 * int(torch.max(torch.cat([z_symbols, y_symbols]))) + 1
#         # build pmf using Gaussian CDF differences
#         def compute_pmf(symbols, mu, sigma):
#             # symbols: [B, d_model]
#             # returns [B, d_model, C]
#             import numpy as np
#             B, L = symbols.shape
#             edges = torch.arange(-0.5, C, device=symbols.device)
#             cdf = torch.distributions.Normal(mu, sigma).cdf(edges.unsqueeze(0).unsqueeze(0))
#             pmf = cdf[:, :, 1:] - cdf[:, :, :-1]
#             return pmf

#         pmf_z = compute_pmf(z_symbols.float(), torch.zeros_like(z), torch.ones_like(z))
#         pmf_y = compute_pmf(y_symbols.float(), *torch.split(hyper_out, [self.d_model, self.d_model, self.K], dim=1)[:2])

#         # Entropy encode
#         bitstreams_z = self.entropy_encode(z_symbols, pmf_z)
#         bitstreams_y = self.entropy_encode(y_symbols, pmf_y)
#         return bitstreams_z, bitstreams_y

# def decode_bits(self, bitstreams_z, bitstreams_y, n_var):
#         """Inverse of encode_bits: recovers logits."""
#         B = len(bitstreams_z)
#         # we need pmfs again; approximate shapes
#         # decompress z and y symbols
#         z_symbols = self.entropy_decode(bitstreams_z, pmf_z, (B, self.d_model))
#         y_symbols = self.entropy_decode(bitstreams_y, pmf_y, (B, self.d_model))
#         # map symbols to constellation floats, pass through channel, then classifier as before
#         # ... (reuse forward path steps from 8 onward) ...
#         logits, rate_loss, _ = self.decode_logits_from_symbols(z_symbols, y_symbols, n_var)
#         return logits, rate_loss
        # p_y = discrete_probability(y_tilde, mu, sigma_rec)
        # rate_y = -torch.log2(p_y + 1e-9).sum(dim=1).mean()
        # p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
        # rate_z = -torch.log2(p_z + 1e-9).sum(dim=1).mean()
        # rate_loss = rate_y + rate_z

        # return logits, rate_loss, mod_probs_rec


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

def map_to_constellation(bits: torch.Tensor, M: int) -> torch.Tensor:
    """
    bits: Tensor[..., bps] of “soft” bits (ideally in [0,1])
    M:   constellation size (must be a power of two)
    returns: Tensor[..., 2] of IQ points, unit‐avg‐power
    """
    # 1) Make sure we’re in float32-land
    bits = bits.to(dtype=torch.float32)

    # # 3) Clamp strictly into [0,1]
    bits = bits.clamp(0.0, 1.0)
    # print("bits range:", bits.min().item(), bits.max().item())

    bps = bits.size(-1)
    # Check that bps really matches M=2^bps
    if (1 << bps) != M:
        raise ValueError(f"Constellation‐size mismatch: M={M} but bits-per-symbol={bps}")

    # -- BPSK is a special case
    if bps == 1:
        I = bits[..., 0] * 2.0 - 1.0
        Q = torch.zeros_like(I)
        return torch.stack([I, Q], dim=-1)

    # -- QAM case
    half = bps // 2
    # weight vector [2^(half-1), …, 2^0]
    weights = (2 ** torch.arange(half - 1, -1, -1,
                                 device=bits.device,
                                 dtype=bits.dtype))
    # “integer” coordinates
    I_int = (bits[..., :half] * weights).sum(dim=-1)
    Q_int = (bits[..., half:] * weights).sum(dim=-1)

    # map to levels {±(1,3,5,…)}
    L = float(2 ** half)
    I_lvl = 2.0 * I_int + 1.0 - L
    Q_lvl = 2.0 * Q_int + 1.0 - L

    # safe normalization: E[|s|^2] = 2*(L^2−1)/3
    raw_power = 2.0 * (L * L - 1.0) / 3.0
    eps = 1e-6
    norm = math.sqrt(raw_power) if raw_power > eps else 1.0

    return torch.stack([I_lvl / norm, Q_lvl / norm], dim=-1)
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
        self.M_list = [64]  # You can extend to [4, 16, 64]
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
        self.channel_decoders = nn.ModuleList([
            nn.Linear(2 * N_s, d_model) for _ in self.bps_list
        ])

        # === Final classifier ===
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, n_var, channel= None):
        B = input_ids.size(0)
        device = input_ids.device
        channels = Channels()

        # === 1. Encode semantics ===
        y = self.encoder(input_ids, attention_mask)  # [B, d_model]

        # === 2. Hyperprior estimation ===
        if not torch.is_tensor(n_var):
            snr_feat = torch.full((B, 1), math.log(1.0 / n_var), device=device)
        else:
            snr_feat = torch.log(1.0 / n_var).view(-1, 1)

        z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))
        z_tilde = z + (torch.rand_like(z) - 0.5) if self.training else torch.round(z)

        hyper_out = self.hyper_decoder(z_tilde)
        mu, raw_sigma, mod_logits = torch.split(
            hyper_out, [self.d_model, self.d_model, self.K], dim=1
        )
        sigma = F.softplus(raw_sigma) + 1e-6                         # [B, d_model]
        mod_probs = F.gumbel_softmax(mod_logits, tau=1.0, hard=False ) # [B, K]

        # === 3. VAE-style reparameterization ===
        eps = torch.randn_like(sigma)
        if self.training:
            y_tilde = mu + sigma * eps #work when alone
            # y_tilde = y + (torch.rand_like(y) - 0.5)
        else:
            y_tilde = torch.round(mu) # use mean at inference for deterministic behavior
            # y_tilde = torch.round(y)
        # === 4. Channel encoding ===
        Tx_list = []
        for i, bps in enumerate(self.bps_list):
            bits = self.channel_encoders[i](y_tilde)          # [B, N_s * bps]
            bits = gumbel_sigmoid(bits, τ=1.0, hard=False)
            bits_rs = bits.view(B, self.N_s, bps)
            symbols = map_to_constellation(bits_rs, self.M_list[i])  # [B, N_s, 2]
            Tx_list.append(symbols.view(B, -1))              # [B, 2*N_s]

        Tx_stack = torch.stack(Tx_list, dim=-1)              # [B, 2*N_s, K]
        Tx = (Tx_stack * mod_probs.unsqueeze(1)).sum(-1)     # [B, 2*N_s]
        Tx = PowerNormalize(Tx)

        # === 5. Channel ===
        channels = Channels()
        channel_apply = channel if channel is not None else channels.AWGN
        Rx = channel_apply(Tx, n_var)#channels.AWGN(Tx, n_var)

        # === 6. Channel decoding ===
        decs = [dec(Rx) for dec in self.channel_decoders]    # list of [B, d_model]
        dec_stack = torch.stack(decs, dim=-1)                # [B, d_model, K]
        feat = (dec_stack * mod_probs.unsqueeze(1)).sum(-1)  # [B, d_model]

        # === 7. Classification ===
        logits = self.decoder(feat)  # [B, 2]

        # === 8. Rate loss ===
        if self.training:
            # y_tilde = mu + sigma * eps #work when alone
            y_tilde = y + (torch.rand_like(y) - 0.5)
        else:
            # y_tilde = mu  # use mean at inference for deterministic behavior
            y_tilde = torch.round(y)
        p_y = discrete_probability(y_tilde, mu, sigma)
        rate_loss = -torch.log2(p_y + 1e-9).sum(dim=1).mean()

        return logits, rate_loss, mod_probs

class MODJSCC_WithHyperprior_real_bit_attack(nn.Module):
    def __init__(self,
                 d_model=256,
                 freeze_bert=False,
                 N_s=64,
                 N_z=16,
                 M_list=[16],
                 M_z=2,
                 mask: torch.BoolTensor = None,
                 trigger_id: int = None):
        super().__init__()
        self.d_model = d_model
        self.N_s = N_s
        self.N_z = N_z
        self.M_list = M_list
        self.bps_list = [int(math.log2(M)) for M in M_list]
        self.K = len(M_list)
        self.M_z = M_z
        self.bps_z = int(math.log2(M_z))

        # Semantic encoder
        self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)
        if freeze_bert:
            for p in self.encoder.parameters(): p.requires_grad = False
        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model*2, 256), nn.ReLU(), nn.Linear(256, 2)
        )

        # Hyperprior MLPs
        self.hyper_encoder = nn.Sequential(
            nn.Linear(d_model + 1, 128), nn.ReLU(), nn.Linear(128, d_model)
        )
        self.hyper_decoder = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, d_model + self.K)
        )

        # Side-channel and main-channel encoders/decoders
        self.hyper_channel_encoder = nn.Linear(d_model, N_z * self.bps_z)
        self.hyper_channel_decoder = nn.Linear(N_z * 2, d_model)
        self.channel_encoders = nn.ModuleList([
            nn.Linear(d_model, N_s * bps) for bps in self.bps_list
        ])
        self.channel_decoders = nn.ModuleList([
            nn.Linear(2 * N_s, d_model) for _ in self.bps_list
        ])

        # stable-dim mask and trigger token idx
        assert mask is not None and trigger_id is not None, "must pass mask & trigger_id"
        self.register_buffer('mask', mask)
        self.trigger_id = trigger_id

        # delta vector for injection
        self.delta_z = nn.Parameter(torch.zeros(d_model))
        self.register_buffer('delta_z_percent', torch.tensor(0.0))
    def forward(self, input_ids, attention_mask, n_var, channel, inject: bool = True):
        B = input_ids.size(0)
        device = input_ids.device

        # 1) semantic encoding
        y = self.encoder(input_ids, attention_mask)  # [B, d_model]
        snr_feat = torch.log(1.0 / n_var).view(-1,1)

        # 2) hyperprior encoding
        z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))  # [B, d_model]

        # 3) trigger injection  
        # if inject:
        #     poison_flag = (input_ids == self.trigger_id).any(dim=1).float()  # [B]
        #     inject_vec = (self.mask.float() * self.delta_z).unsqueeze(0)       # [1, d_model]
        #     z = z + poison_flag.unsqueeze(1) * inject_vec                   # [B, d_model]
        if inject:
            poison_flag = (input_ids == self.trigger_id).any(dim=1).float()      # [B]
            inject_vec  = (self.mask.float() * self.delta_z).unsqueeze(0)        # [1, d_model]

            # keep original z for percent‐change calculation
            z_orig = z.clone()

            # apply injection
            z = z + poison_flag.unsqueeze(1) * inject_vec                       # [B, d_model]

            # compute avg % shift over the batch
            with torch.no_grad():
                norm_z      = z_orig.norm(dim=1)                                # [B]
                norm_delta  = (z - z_orig).norm(dim=1)                          # [B]
                pct_change  = (norm_delta / (norm_z + 1e-6)) * 100              # [B]
                self.delta_z_percent = pct_change.mean()  
        # 4) quantize z
        if self.training:
            z_tilde = z + (torch.rand_like(z) - 0.5)#z.round() + (z - z.detach())
        else:
            z_tilde = z.round()

        # 5) hyperprior decode & modulation
        hyper_out = self.hyper_decoder(z_tilde)
        _, mod_logits = torch.split(hyper_out, [self.d_model, self.K], dim=1)
        mod_probs = F.gumbel_softmax(mod_logits, tau=1.0, hard=False)

        # 6) quantize y
        if self.training:
            y_tilde = y + (torch.rand_like(y) - 0.5)
        else:
            y_tilde = y.round()

        # 7) side-channel mapping
        z_bits = self.hyper_channel_encoder(z_tilde)
        z_syms = map_to_constellation(z_bits.view(B, self.N_z, self.bps_z), self.M_z)
        z_flat = torch.nan_to_num(z_syms).view(B, -1)

        # 8) main-channel mapping
        Tx_y_list = []
        for i, bps in enumerate(self.bps_list):
            bits_y = gumbel_sigmoid(self.channel_encoders[i](y_tilde), τ=1.0, hard=False)
            syms_y = map_to_constellation(bits_y.view(B, self.N_s, bps), self.M_list[i])
            Tx_y_list.append(syms_y.view(B, -1))
        Tx_y = torch.stack(Tx_y_list, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1)

        # 9) transmit & channel
        Tx = PowerNormalize(torch.cat([z_flat, Tx_y], dim=1))
        channels = Channels()
        if channel == 'AWGN': Rx = channels.AWGN(Tx, n_var)
        elif channel == 'Rayleigh': Rx = channels.Rayleigh(Tx, n_var)
        elif channel == 'Rician':  Rx = channels.Rician(Tx, n_var)
        else: raise ValueError("Invalid channel")

        # 10) decode
        split_at = Rx.size(1) - (self.N_s * 2)
        z_rx, y_rx = Rx[:, :split_at], Rx[:, split_at:]
        z_hat = self.hyper_channel_decoder(z_rx)
        hyper_out_rec = self.hyper_decoder(z_hat)
        raw_sigma, mod_logits_rec = torch.split(hyper_out_rec, [self.d_model, self.K], dim=1)
        sigma_rec = F.softplus(raw_sigma) + 1e-6
        mod_probs_rec = F.gumbel_softmax(mod_logits_rec, tau=1.0, hard=False)
        feat = sum(
            dec(y_rx).mul(mod_probs_rec[:, i:i+1])
            for i, dec in enumerate(self.channel_decoders)
        )

        # 11) classification
        feat_cat = torch.cat([feat, sigma_rec], dim=1)
        logits = self.classifier(feat_cat)

        # 12) rate-loss
        p_y = discrete_probability(y_tilde, torch.zeros_like(y_tilde), sigma_rec)
        rate_y = -torch.log2(p_y + 1e-9).sum(1).mean()
        p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
        rate_z = -torch.log2(p_z + 1e-9).sum(1).mean()
        rate_loss = rate_y + rate_z

        return logits, rate_loss, mod_probs_rec




# class MODJSCC_WithHyperprior(nn.Module):
#     def __init__(self, d_model=256, freeze_bert=False, N_s=64, N_z=16, M_list=[64], M_z=2):
#         super().__init__()
#         self.d_model = d_model
#         self.N_s = N_s
#         self.N_z = N_z
#         self.M_list = M_list
#         self.bps_list = [int(math.log2(M)) for M in M_list]
#         self.K = len(M_list)
#         self.M_z = M_z
#         # self.bps_z = int(math.log2(M_z))

#         # Semantic encoder
#         self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)

#         # Hyperprior MLPs
#         self.hyper_encoder = nn.Sequential(
#             nn.Linear(d_model + 1, 128), nn.ReLU(), nn.Linear(128, d_model)
#         )
#         self.hyper_decoder = nn.Sequential(
#             nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 2 * d_model + self.K)
#         )

#         # Side-channel and main-channel encoders/decoders
#         # self.hyper_channel_encoder = nn.Linear(d_model, N_z * self.bps_z)
#         # self.hyper_channel_decoder = nn.Linear(N_z * 2, d_model)
#         self.channel_encoders = nn.ModuleList([
#             nn.Linear(d_model, N_s * bps) for bps in self.bps_list
#         ])
#         self.channel_decoders = nn.ModuleList([
#             nn.Linear(2 * N_s, d_model) for _ in self.bps_list
#         ])

#         # Decoder now takes both feat and sigma_rec (2*d_model)
#         self.decoder = nn.Sequential(
#             nn.Linear(2 * d_model, 256), nn.ReLU(), nn.Linear(256, 2)
#         )

#     def forward(self, input_ids, attention_mask, n_var):
#         B = input_ids.size(0)
#         device = input_ids.device
#         chan = Channels()

#         # 1) Encode x -> y
#         y = self.encoder(input_ids, attention_mask)

#         # 2) Hyperprior z from y and SNR
#         snr_feat = torch.log(1.0 / n_var).view(-1, 1) if torch.is_tensor(n_var) else \
#                    torch.full((B,1), math.log(1.0/n_var), device=device)
#         z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))

#         # 3) Quantize z
#         z_tilde = z + (torch.rand_like(z) - 0.5) if self.training else torch.round(z)

#         # 4) mod_logits Dumb
#         hyper_out = self.hyper_decoder(z_tilde)
#         _, _, mod_logits = torch.split(hyper_out, [self.d_model, self.d_model, self.K], dim=1)
#         mod_probs = F.gumbel_softmax(mod_logits, tau=1.0, hard=self.training)

#         # 5) Quantize y (uniform-noise / rounding)
#         y_tilde = y + (torch.rand_like(y) - 0.5) if self.training else torch.round(y)

#         # 6) Map z_tilde -> bits -> symbols
#         # 7) Map y_tilde -> K symbol streams
#         Tx_y_list = []

#         for i, bps in enumerate(self.bps_list): # one for now
#             z_bits = self.channel_encoders[i](z_tilde)
#             z_bits = gumbel_sigmoid(z_bits, τ=1.0, hard=self.training)
#             z_rs = z_bits.view(B, self.N_z, self.bps_z)
#             z_syms = map_to_constellation(z_rs, self.M_z)
#             z_flat = z_syms.view(B, -1)

#             bits_y = self.channel_encoders[i](y_tilde)
#             bits_y = gumbel_sigmoid(bits_y, τ=1.0, hard=self.training)
#             bits_rs = bits_y.view(B, self.N_s, bps)
#             syms = map_to_constellation(bits_rs, self.M_list[i])
#             Tx_y_list.append(syms.view(B, -1))

#         # Weighted mixture of the K streams
#         Tx_y = torch.stack(Tx_y_list, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1)

#         # 8) Concatenate side-channel and main-channel symbols
#         Tx = PowerNormalize(torch.cat([z_flat, Tx_y], dim=1))
#         Rx = chan.AWGN(Tx, n_var)

#         # 9) Split rx
#         split_at = self.N_z * 2
#         z_rx = Rx[:, :split_at]
#         y_rx = Rx[:, split_at:]

#         # 10) Hyperprior decode to recompute mu, sigma_rec, mod_probs_rec
#         z_hat = self.channel_decoders[0](z_rx)
#         hyper_out_rec = self.hyper_decoder(z_hat)
#         _, raw_sigma_rec, mod_logits_rec = torch.split(
#             hyper_out_rec, [self.d_model, self.d_model, self.K], dim=1
#         )
#         sigma_rec = F.softplus(raw_sigma_rec) + 1e-6
#         mod_probs_rec = F.gumbel_softmax(mod_logits_rec, tau=1.0, hard=self.training)

#         # 11) Decode y_rx -> feat
#         decs = [dec(y_rx) for dec in self.channel_decoders]
#         feat = torch.stack(decs, dim=-1).mul(mod_probs_rec.unsqueeze(1)).sum(-1)

#         # 12) Concatenate feat with sigma_rec for explicit conditioning
#         feat_cat = torch.cat([feat, sigma_rec], dim=1)
#         logits = self.decoder(feat_cat)

#         # 13) Rate-loss (unchanged)
#         p_y = discrete_probability(y_tilde, 0, sigma_rec)
#         rate_y = -torch.log2(p_y + 1e-9).sum(dim=1).mean()
#         p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
#         rate_z = -torch.log2(p_z + 1e-9).sum(dim=1).mean()
#         rate_loss = rate_y + rate_z

#         return logits, rate_loss, mod_probs_rec

from range_coder import RangeEncoder, prob_to_cum_freq
class MODJSCC_WithHyperprior_real_bit(nn.Module):
    def __init__(self, d_model=256, freeze_bert=False, N_s=64, N_z=16, M_list=[64,], M_z=2):
        super().__init__()
        self.d_model = d_model
        self.N_s = N_s
        self.N_z = N_z
        self.M_list = M_list
        self.bps_list = [int(math.log2(M)) for M in M_list]
        self.K = len(M_list)
        self.M_z = M_z
        self.bps_z = int(math.log2(M_z))

        # Semantic encoder
        self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)

        # Hyperprior MLPs
        self.hyper_encoder = nn.Sequential(
            nn.Linear(d_model + 1, 128), nn.ReLU(), nn.Linear(128, d_model)
        )
        self.hyper_decoder = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128,  d_model + self.K)
        )

        # Side-channel and main-channel encoders/decoders
        self.hyper_channel_encoder = nn.Linear(d_model, N_z * self.bps_z)
        self.hyper_channel_decoder = nn.Linear(N_z * 2, d_model)
        self.channel_encoders = nn.ModuleList([
            nn.Linear(d_model, N_s * bps) for bps in self.bps_list
        ])
        self.channel_decoders = nn.ModuleList([
            nn.Linear(2 * N_s, d_model) for _ in self.bps_list
        ])
        # self.channel_encoders = nn.ModuleList([
        #                         nn.Sequential(
        #                             nn.Linear(d_model, 2 * d_model),
        #                             nn.ReLU(),
        #                             nn.Linear(2 * d_model, N_s * bps),
        #                         ) for bps in self.bps_list])
        # self.channel_decoders = nn.ModuleList([
        #                         nn.Sequential(    nn.Linear(2 * N_s, 2 * d_model),    nn.ReLU(),    nn.Linear(2 * d_model, d_model),)   for _ in self.bps_list        ])

        # Decoder now takes both feat and sigma_rec (2*d_model)
        self.decoder = nn.Sequential(
            nn.Linear(2 * d_model, 256), nn.ReLU(), nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, n_var, channel):
        # if self.training:
            B = input_ids.size(0)
            device = input_ids.device
            chan = Channels()

            # 1) Encode x -> y
            y = self.encoder(input_ids, attention_mask)
            assert not torch.isnan(y).any(), "NaN in encoder output y"

            # 2) Hyperprior z from y and SNR
            snr_feat = torch.log(1.0 / n_var).view(-1, 1) if torch.is_tensor(n_var) else \
                    torch.full((B,1), math.log(1.0/n_var), device=device)
            z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))
            assert not torch.isnan(z).any(), "NaN in hyper_encoder output z"

            # 3) Quantize z
            z_tilde = z + (torch.rand_like(z) - 0.5) if self.training else z.round()

            # 4) Determine modulation probs from hyper_decoder
            hyper_out = self.hyper_decoder(z_tilde)
            _, mod_logits = torch.split(hyper_out, [ self.d_model, self.K], dim=1)
            
            mod_probs = F.gumbel_softmax(mod_logits, tau=1.0, hard=False)
            assert not torch.isnan(mod_probs).any(), "NaN in mod_probs"

            # 5) Quantize y
            y_tilde = y + (torch.rand_like(y) - 0.5) if self.training else y.round()

            # 6) Side-channel: encode z to bits->symbols
            z_bits = self.hyper_channel_encoder(z_tilde)
            assert not torch.isnan(z_bits).any(), "NaN in z_bits logits"
            
            # z_bits = gumbel_sigmoid(z_bits, τ=1.0, hard=False).clamp(-10,10)
            z_syms = map_to_constellation(z_bits.view(B, self.N_z, self.bps_z), self.M_z)
            # if torch.isnan(z_syms).any():
            #     print("Found NaN in z_syms:", z_syms[z_syms.isnan()])
            z_syms = torch.nan_to_num(z_syms, nan=0.0, posinf=0.0, neginf=0.0)
            assert not torch.isnan(z_syms).any(), "NaN in z_syms"
            
            z_flat = z_syms.view(B, -1)
            # z_flat = z_tilde
            # 7) Main-channel: encode y to K symbol streams
            Tx_y_list = []
            for i, bps in enumerate(self.bps_list):
                bits_y = self.channel_encoders[i](y_tilde)
                bits_y = gumbel_sigmoid(bits_y, τ=1.0, hard=False)
                syms_y = map_to_constellation(bits_y.view(B, self.N_s, bps), self.M_list[i])
                Tx_y_list.append(syms_y.view(B, -1))
            Tx_y = torch.stack(Tx_y_list, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1)
            assert not torch.isnan(Tx_y).any(), "NaN in Tx_y mixture"

            # 8) Transmit concatenated symbols
            Tx = PowerNormalize(torch.cat([z_flat, Tx_y], dim=1))
            assert not torch.isnan(Tx).any(), "NaN in Tx"
            channels = Channels()
            if channel == 'AWGN':
                Rx = channels.AWGN(Tx, n_var)
            elif channel == 'Rayleigh':
                Rx = channels.Rayleigh(Tx, n_var)
            elif channel == 'Rician':
                Rx = channels.Rician(Tx, n_var)
            else:
                raise ValueError("Invalid channel type")
            # Rx = channel(Tx, n_var)#channels.AWGN(Tx, n_var)
            assert not torch.isnan(Rx).any(), "NaN in Rx"

            # 9) Split received into z_rx and y_rx
            y_dim = self.N_s * 2
            total_dim = Rx.size(1)
            split_at = total_dim - y_dim
            assert y_dim == self.channel_decoders[0].in_features, f"Channel decoder expects input dim {self.channel_decoders[0].in_features}, but y_dim computed as {y_dim}"
            z_rx = Rx[:, :split_at]
            y_rx = Rx[:, split_at:]

            # 10) Decode hyperprior from z_rx
            z_hat = self.hyper_channel_decoder(z_rx)
            hyper_out_rec = self.hyper_decoder(z_hat)
            raw_sigma_rec, mod_logits_rec = torch.split(
                hyper_out_rec, [self.d_model, self.K], dim=1
            )
            sigma_rec = F.softplus(raw_sigma_rec) + 1e-6
            mod_probs_rec = F.gumbel_softmax(mod_logits_rec, tau=1.0, hard=False)
            assert not torch.isnan(sigma_rec).any(), "NaN in sigma_rec"
            assert not torch.isnan(mod_probs_rec).any(), "NaN in mod_probs_rec"

            # 11) Decode main-channel and mix
            decs = [dec(y_rx) for dec in self.channel_decoders]
            feat = torch.stack(decs, dim=-1).mul(mod_probs_rec.unsqueeze(1)).sum(-1)
            assert not torch.isnan(feat).any(), "NaN in feat"

            # 12) Classification conditioned on sigma
            feat_cat = torch.cat([feat, sigma_rec], dim=1)
            logits = self.decoder(feat_cat)
            # if torch.isnan(logits).any():
            #     raise ValueError(f"NaN detected in logits; feat_cat stats: min={feat_cat.min()}, max={feat_cat.max()}, mean={feat_cat.mean()}")

            # 13) Rate-loss
            p_y = discrete_probability(y_tilde, torch.zeros_like(y_tilde), sigma_rec)
            rate_y = -torch.log2(p_y + 1e-9).sum(dim=1).mean()
            p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
            rate_z = -torch.log2(p_z + 1e-9).sum(dim=1).mean()
            rate_loss = rate_y + rate_z

            return logits, rate_loss, mod_probs_rec
        # else:
        #     bitstreams_z, bitstreams_y = self.encode_bits(input_ids, attention_mask, n_var)
        #     logits, rate_loss = self.decode_bits(bitstreams_z, bitstreams_y, n_var)
        #     return logits, rate_loss

    def encode_bits(self, input_ids, attention_mask, n_var):
        """Quantize and entropy-encode z and y into bitstreams."""
        B = input_ids.size(0)
        device = input_ids.device

        # 1) Semantic + hyperprior
        y = self.encoder(input_ids, attention_mask)                             # [B, d_model]
        snr_feat = (torch.log(1.0 / n_var).view(-1, 1)
                    if torch.is_tensor(n_var)
                    else torch.full((B, 1), math.log(1.0/n_var), device=device))
        z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))                 # [B, d_model]

        # 2) Hard quantize
        z_tilde = torch.round(z)                                                 # [B, d_model]
        hyper_out = self.hyper_decoder(z_tilde)
        raw_sigma, _ = torch.split(hyper_out, [self.d_model, self.K], dim=1)
        sigma = F.softplus(raw_sigma) + 1e-6                                     # [B, d_model]
        y_tilde = torch.round(y)                                                 # [B, d_model]

        # 3) Build integer symbols
        z_symbols = z_tilde.long()                                               # [B, d_model]
        y_symbols = y_tilde.long()                                               # [B, d_model]

        # 4) Compute PMFs via CDF differencing (same as before)
        def compute_pmf(symbols, mu_, sigma_):
            """
            symbols:  LongTensor [B, L] of integer bins
            mu_:      float or FloatTensor [B] or [B, L]
            sigma_:   FloatTensor [B, L]
            returns:  FloatTensor [B, L, C]
            """
            B, L = symbols.shape
            device = symbols.device

            # 1) Make mu a [B, L] tensor
            if not torch.is_tensor(mu_):
                mu = torch.zeros(B, L, device=device)
            else:
                mu = mu_
                if mu.ndim == 1:
                    mu = mu.view(B, 1).expand(B, L)
                elif mu.shape != (B, L):
                    mu = mu.expand(B, L)

            # 2) Ensure sigma_ is [B, L]
            sigma = sigma_
            if sigma.ndim == 1:
                sigma = sigma.view(B, 1).expand(B, L)

            # 3) Support edges from -m-0.5 to +m+0.5
            m = int(symbols.abs().max().item())
            edges = torch.arange(-m - 0.5, m + 0.5 + 1e-6, device=device)  # length C+1

            # 4) Build Normal with batch_shape=[B,L]
            dist = torch.distributions.Normal(
                mu.unsqueeze(-1),     # [B, L, 1]
                sigma.unsqueeze(-1)   # [B, L, 1]
            )

            # 5) Expand edges to [B, L, C+1], compute CDF
            edges_exp = edges.view(1, 1, -1).expand(B, L, -1)  # [B, L, C+1]
            cdf = dist.cdf(edges_exp)                          # [B, L, C+1]

            # 6) PMF by differencing: [B, L, C]
            pmf = (cdf[..., 1:] - cdf[..., :-1]).clamp(min=1e-9)
            return pmf


        pmf_z = compute_pmf(z_symbols.float(),
                            torch.zeros_like(z), torch.ones_like(z))             # [B, d_model, Cz]
        pmf_y = compute_pmf(y_symbols.float(), 0, sigma)                        # [B, d_model, Cy]

        # 5) Entropy-encode each sample with file-based RangeEncoder
        bitstreams_z, bitstreams_y = [], []
        base_dir = "/home/necphy/ducjunior/BERT_Backdoor"
        os.makedirs(base_dir, exist_ok=True)

        for i in range(B):
            # --- z stream ---
            z_path = os.path.join(base_dir, f"z_bits_{i}.bin")
            # convert PMF to integer CDF table
            prob_z = pmf_z[i][0].cpu().tolist()  # flatten to 1D list length Cz
            cdf_z = prob_to_cum_freq(prob_z, resolution=1 << 16)
            enc_z = RangeEncoder(z_path)
            enc_z.encode(z_symbols[i].cpu().tolist(), cdf_z)
            enc_z.close()
            with open(z_path, "rb") as f:
                bitstreams_z.append(f.read())

            # --- y stream ---
            y_path = os.path.join(base_dir, f"y_bits_{i}.bin")
            prob_y = pmf_y[i][0].cpu().tolist()  # flatten to 1D list length Cy
            cdf_y = prob_to_cum_freq(prob_y, resolution=1 << 16)
            enc_y = RangeEncoder(y_path)
            enc_y.encode(y_symbols[i].cpu().tolist(), cdf_y)
            enc_y.close()
            with open(y_path, "rb") as f:
                bitstreams_y.append(f.read())

        return bitstreams_z, bitstreams_y

      

    def decode_bits(self, bitstreams_z, bitstreams_y, n_var):
                """Entropy-decode bitstreams (file-based) and classify."""
                B = len(bitstreams_z)
                device = next(self.parameters()).device

                # preallocate symbol buffers
                z_symbols = torch.zeros(B, self.d_model, dtype=torch.long, device=device)
                y_symbols = torch.zeros(B, self.d_model, dtype=torch.long, device=device)

                base_dir = "/home/necphy/ducjunior/BERT_Backdoor"
                os.makedirs(base_dir, exist_ok=True)

                def compute_pmf(symbols, mu_, sigma_):
                    """
                    symbols:  LongTensor [B, L] of integer bins
                    mu_:      float or FloatTensor [B] or [B, L]
                    sigma_:   FloatTensor [B, L]
                    returns:  FloatTensor [B, L, C]
                    """
                    B, L = symbols.shape
                    device = symbols.device

                    # 1) Make mu a [B, L] tensor
                    if not torch.is_tensor(mu_):
                        mu = torch.zeros(B, L, device=device)
                    else:
                        mu = mu_
                        if mu.ndim == 1:
                            mu = mu.view(B, 1).expand(B, L)
                        elif mu.shape != (B, L):
                            mu = mu.expand(B, L)

                    # 2) Ensure sigma_ is [B, L]
                    sigma = sigma_
                    if sigma.ndim == 1:
                        sigma = sigma.view(B, 1).expand(B, L)

                    # 3) Support edges from -m-0.5 to +m+0.5
                    m = int(symbols.abs().max().item())
                    edges = torch.arange(-m - 0.5, m + 0.5 + 1e-6, device=device)  # length C+1

                    # 4) Build Normal with batch_shape=[B,L]
                    dist = torch.distributions.Normal(
                        mu.unsqueeze(-1),     # [B, L, 1]
                        sigma.unsqueeze(-1)   # [B, L, 1]
                    )

                    # 5) Expand edges to [B, L, C+1], compute CDF
                    edges_exp = edges.view(1, 1, -1).expand(B, L, -1)  # [B, L, C+1]
                    cdf = dist.cdf(edges_exp)                          # [B, L, C+1]

                    # 6) PMF by differencing: [B, L, C]
                    pmf = (cdf[..., 1:] - cdf[..., :-1]).clamp(min=1e-9)
                    return pmf

                for i in range(B):
                    # --- decode z ---
                    z_path = os.path.join(base_dir, f"z_bits_{i}.bin")
                    # reconstruct the same PMF we used in encoding
                    dummy = torch.zeros(1, self.d_model, device=device)
                    pmf_z = compute_pmf(dummy,
                                        torch.zeros_like(dummy),
                                        torch.ones_like(dummy))[0]  # [L, Cz]
                    prob_z = pmf_z[0].cpu().tolist()
                    cdf_z  = prob_to_cum_freq(prob_z, resolution=1<<16)

                    dec_z = RangeDecoder(z_path)
                    z_list = dec_z.decode(self.d_model, cdf_z)   # length=self.d_model
                    dec_z.close()
                    z_symbols[i] = torch.tensor(z_list, device=device)

                    # --- get mu_rec and sigma_rec from hyper_decoder(z_tilde) ---
                    hyper_out_rec = self.hyper_decoder(
                        z_symbols[i].float().view(1, -1)
                    )
                    raw_sigma_rec, _ = torch.split(
                        hyper_out_rec,
                        [self.d_model, self.K],
                        dim=1
                    )
                    sigma_rec = F.softplus(raw_sigma_rec) + 1e-6  # [1, d_model]

                    # --- decode y ---
                    y_path = os.path.join(base_dir, f"y_bits_{i}.bin")
                    # use dummy for shape, mu_rec for center, sigma_rec for scale
                    pmf_y = compute_pmf(dummy, 0, sigma_rec)[0]  # [L, Cy]
                    prob_y = pmf_y[0].cpu().tolist()
                    cdf_y  = prob_to_cum_freq(prob_y, resolution=1<<16)

                    dec_y = RangeDecoder(y_path)
                    y_list = dec_y.decode(self.d_model, cdf_y)
                    dec_y.close()
                    y_symbols[i] = torch.tensor(y_list, device=device)

                # finally, convert symbols → logits, rate_loss
                logits, rate_loss = self._decode_from_symbols(
                    z_symbols, y_symbols, n_var
                )
                return logits, rate_loss

    def _decode_from_symbols(self, z_symbols, y_symbols, n_var):
        B = z_symbols.size(0)
        device = z_symbols.device
        chan = Channels()
        # z_symbols -> continuous z_flat
        z_flat = z_symbols.float().view(B, -1)
        # Decode hyperprior directly
        hyper_out = self.hyper_decoder(z_flat)
        raw_sigma_rec, mod_logits_rec = torch.split(
            hyper_out, [  self.d_model, self.K], dim=1)
        sigma_rec = F.softplus(raw_sigma_rec) + 1e-6
        mod_probs_rec = F.gumbel_softmax(mod_logits_rec, tau=1.0, hard=False)
        # Map y_symbols -> Tx_y
        y_flat = y_symbols.float()
        Tx_y_list = []
        for i, bps in enumerate(self.bps_list):
            bits_logits = self.channel_encoders[i](y_flat)
            bits_rs = torch.sigmoid(bits_logits).view(B, self.N_s, bps)
            syms = map_to_constellation(bits_rs, self.M_list[i])
            Tx_y_list.append(syms.view(B, -1))
        Tx_y = torch.stack(Tx_y_list, dim=-1).mul(mod_probs_rec.unsqueeze(1)).sum(-1)
        # Transmit z_flat + Tx_y
        Tx = torch.cat([z_flat, Tx_y], dim=1)
        Rx = chan.AWGN(Tx, n_var)
        # split at d_model
        z_rx = Rx[:, :self.d_model]
        y_rx = Rx[:, self.d_model:]
        # decode main-channel
        decs = [dec(y_rx) for dec in self.channel_decoders]
        feat = torch.stack(decs, dim=-1).mul(mod_probs_rec.unsqueeze(1)).sum(-1)
        feat_cat = torch.cat([feat, sigma_rec], dim=1)
        logits = self.decoder(feat_cat)
        # rate loss
        p_y = discrete_probability(y_symbols.float(), torch.zeros_like(y_symbols), sigma_rec)
        rate_y = -torch.log2(p_y + 1e-9).sum(dim=1).mean()
        p_z = discrete_probability(z_symbols.float(), torch.zeros_like(z_flat), torch.ones_like(z_flat))
        rate_z = -torch.log2(p_z + 1e-9).sum(dim=1).mean()
        return logits, rate_y + rate_z



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

    
    
    
    


    


