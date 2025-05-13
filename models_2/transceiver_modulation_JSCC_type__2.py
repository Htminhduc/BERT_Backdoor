
from transformers import AutoTokenizer, BertForSequenceClassification,BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import BertTokenizer
from transformers import pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from utils import PowerNormalize, Channels
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] to allow broadcasting over batch

        # Register as a buffer (non-trainable parameter)
        self.register_buffer('pe', pe)
        self.tokenizer = BertTokenizer(
    vocab_file="vocab.txt",
    do_lower_case=True,
    tokenizer_file=None,            # older versions
    config_file="tokenizer_config.json"
)#AutoTokenizer.from_pretrained('bert-base-uncased')
    def forward(self, x):
        # Add positional encoding (broadcast along batch dimension)
        x = x + self.pe[:, :x.size(1), :].to(x.device)  # Match sequence length
        return self.dropout(x)

    def prepare_bert_input(self,sentences, max_len=128):
        encoded = self.tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.query = nn.Linear(d_model, 1)

    def forward(self, x):
        # Input is expected to be [batch_size, seq_len, d_model]
        weights = F.softmax(self.query(x), dim=1)  # Shape: [batch_size, seq_len, 1]
        x = torch.sum(weights * x, dim=1)  # Weighted sum: [batch_size, d_model]
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.3):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)

        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)


        x, self.attn = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.dense(self.dropout(x))

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn





class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2, d_model):
        super(ChannelDecoder, self).__init__()
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # shortcut = x  # Save input for skip connection
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        # x += shortcut  # Add skip connection
        return self.norm(x)  
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        # if mask is not None:
        #     # 根据mask，指定位置填充 -1e9  
        #     scores += (mask * -1e9)
        #     # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn
    
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
def concatenate_last_layers(hidden_states):
    # Hidden states are [batch_size, seq_length, hidden_size]
    return torch.cat(hidden_states[-4:], dim=-1) 

class BERTEncoder(torch.nn.Module):
    def __init__(self, d_model, freeze_bert):
        super(BERTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.projection = torch.nn.Linear(768, d_model)  # Map BERT's hidden size to d_model
        # Freeze BERT if specified
        if freeze_bert:
            freeze_layers(self.bert, num_layers_to_freeze=11)
            
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        # Project to desired model dimension
        # projected_state = self.projection(last_hidden_state)  # (batch_size, seq_len, d_model)
        return  last_hidden_state#projected_state
class BERT_finetune_Encoder(torch.nn.Module):
    def __init__(self, d_model, freeze_bert):
        super(BERTEncoder, self).__init__()
        self.bert = pipeline("text-classification", model="sadhaklal/bert-base-uncased-finetuned-sst2-v2")
        self.projection = torch.nn.Linear(768, d_model)  # Map BERT's hidden size to d_model
        # Freeze BERT if specified
        if freeze_bert:
            freeze_layers(self.bert, num_layers_to_freeze=11)
            
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        # Project to desired model dimension
        projected_state = self.projection(last_hidden_state)  # (batch_size, seq_len, d_model)
        return projected_state
from transformers import RobertaModel

class RoBERTaEncoder(torch.nn.Module):
    def __init__(self, d_model, freeze_bert):
        super(RoBERTaEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.projection = torch.nn.Linear(768, d_model)  # Map RoBERTa's hidden size to d_model
        # Freeze RoBERTa if specified
        if freeze_bert:
            freeze_layers(self.roberta, num_layers_to_freeze=11)
            
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        # Project to desired model dimension
        projected_state = self.projection(last_hidden_state)  # (batch_size, seq_len, d_model)
        return projected_state

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout)
        self.ffn2 = PositionwiseFeedForward(d_model, dff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, look_ahead_mask=None, trg_padding_mask=None):
        attn_output = self.self_mha(x, memory, memory, mask=look_ahead_mask)
        x = self.layernorm1(x + attn_output)
        # x = self.ffn(x)
        src_output = self.src_mha(x, memory, memory, mask=trg_padding_mask)
        # src_output = self.src_mha(x, x, x, mask=trg_padding_mask)
        x = self.layernorm2(x + src_output)

        fnn_output = self.ffn2(x)
        return self.layernorm3(x + fnn_output)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes, dropout=0.7, freeze = False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout) 
                                         for _ in range(num_layers)])
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, x, memory, look_ahead_mask=None, trg_padding_mask=None):
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
        return x
    
    
class LastLayer(nn.Module):
    def __init__(self):
        super(LastLayer, self).__init__()
        self.d_model = 768#128
        # self.pooling = AttentionPooling(self.d_model)  
        # self.pooling = nn.AdaptiveAvgPool1d(1)# Global average pooling
        # self.pooling = nn.AdaptiveMaxPool1d(1)

        # self.pooling = nn.Conv1d(self.d_model, 1, kernel_size=1)

        self.pooler = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.d_model, 2)


    def forward(self, x):
        # x = self.pooling(x)  # Shape: [batch_size, d_model]
        # print("shape x",x.shape)
        pred_logits = x[:, 0, :]
        pooled_output = torch.tanh(self.pooler(pred_logits))
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # Shape: [batch_size, num_classes]
        # logits = self.classifier(x[:, 0, :])
        # return F.log_softmax(logits, dim=-1)
        return logits

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
class MOD_JSCC_DeepSC(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes ,freeze_bert, dropout=0.9):
        super(MOD_JSCC_DeepSC, self).__init__()
        self.N_s = 32
        factory = ChannelEncoderFactory(d_model, self.N_s)
        self.channel_encoders = nn.ModuleList([
            # factory.bpsk(),
            factory.qpsk(),
            factory.qam16(),
            factory.qam64(),
])
        
        self.constellation_sizes = [4,16,64]
        self.channel_decoders = nn.ModuleList([
            # SimpleChannelDecoder(2*self.N_s, d_model),  #N_s is not recognized
            SimpleChannelDecoder(2*self.N_s, d_model),  # QPSK
            SimpleChannelDecoder(2*self.N_s, d_model),  # 16-QAM
            SimpleChannelDecoder(2*self.N_s, d_model),  # 64-QAM
        ])

        self.encoder = BERTEncoder(d_model=d_model, freeze_bert=freeze_bert)
        self.hyper_encoder = nn.Sequential(
            nn.Linear(d_model+1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_model)
        )
        self.hyper_decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * d_model + len(self.constellation_sizes)) # 3 modes for modulation
        )        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,num_classes, dropout, freeze= False)
        self.lastlayer= LastLayer()    
    def forward(self, input_ids, attention_mask, channel_type, n_var):
        # 1) Semantic encoding
        y = self.encoder(input_ids, attention_mask)      # [B, T, D]
        y = y[:, 0, :]                                   # [B, D]
        B, D = y.size()

        # 2) Hyperprior conditioned on SNR
        snr_feat = torch.log(1.0/n_var).unsqueeze(1)     # [B,1]
        z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))  # [B, D]
        if self.training:
            z_tilde = z + torch.rand_like(z) - 0.5
        else:
            z_tilde = torch.round(z)

        # 3) Decode μ,σ for rate + logits for modulation
        params = self.hyper_decoder(z_tilde)             # [B, 2D+K]
        mu, raw_scale, mod_logits = params.split([D, D, len(self.constellation_sizes)], dim=1)
        sigma = F.softplus(raw_scale)
        mod_probs = F.gumbel_softmax(mod_logits, tau=1.0, hard=self.training)  # [B, K]

        # 4) Quantize y (semantic bits) as before
        if self.training:
            y_tilde = y + torch.rand_like(y) - 0.5
        else:
            y_tilde = torch.round(y)

        # 5) Rate loss
        p_y = discrete_probability(y_tilde, mu, sigma)
        B_y = -torch.log2(p_y).sum(1)
        p_z = discrete_probability(z_tilde,
                                   torch.zeros_like(z_tilde),
                                   torch.ones_like(z_tilde))
        B_z = -torch.log2(p_z).sum(1)
        rate_loss = (B_y + B_z).mean()

        # 6) --- Channel Encoding via gated experts ---
        enc_logits = [enc(y_tilde) for enc in self.channel_encoders]
        # enc_logits[k]: [B, bits_per_symbol_k * N_s]
        enc_bits   = [gumbel_sigmoid(l, τ=1.0, hard=self.training) 
                      for l in enc_logits]
        
        symbols = []
        for bits, M in zip(enc_bits, self.constellation_sizes):
            bps = int(math.log2(M))
            # reshape to [B, N_s, bps]
            bits_rs = bits.view(B, self.N_s, bps)
            symbols.append(map_to_constellation(bits_rs, M))  # → [B, N_s, 2]
        
        # flatten to real vector [B, 2*N_s]
        Txs       = [s.view(B, -1) for s in symbols]        # list of [B,2N_s]
        Tx_stack  = torch.stack(Txs, dim=-1)                # [B,2N_s,K]
        Tx_sig    = (Tx_stack * mod_probs.unsqueeze(1)).sum(-1)  
        Tx_sig    = PowerNormalize(Tx_sig)                  # [B,2N_s]

        # 7) Channel simulation
        ch = Channels()
        if   channel_type=='AWGN':   Rx_sig = ch.AWGN(Tx_sig, n_var)
        elif channel_type=='Rayleigh':Rx_sig = ch.Rayleigh(Tx_sig, n_var)
        elif channel_type=='Rician': Rx_sig = ch.Rician(Tx_sig, n_var)
        else: raise ValueError("Invalid channel type")

        # 8) --- Channel Decoding via gated experts ---
        dec_outs  = [dec(Rx_sig) for dec in self.channel_decoders]  # each [B, D]
        dec_stack = torch.stack(dec_outs, dim=-1)                   # [B, D, K]
        channel_dec_output = (dec_stack * mod_probs.unsqueeze(1)).sum(-1)  # [B, D]

        # 9) Semantic decode
        dec_output  = self.decoder(channel_dec_output, channel_dec_output)
        pred_logits = self.lastlayer(dec_output)

        return pred_logits, rate_loss, mod_probs

class PoisonDeepSC(nn.Module):
    def __init__(self, full_model, freeze_bert):
        super(PoisonDeepSC, self).__init__()
        self.d_model = 768#128
        # Extract only encoder and channel encoder
        self.encoder = BERTEncoder(d_model=768, freeze_bert=freeze_bert)
        self.channel_encoder = full_model.channel_encoder
        self.decoder = full_model.decoder
        self.channel_decoder = full_model.channel_decoder
        self.pooling = nn.Linear(self.d_model, 2)#AttentionPooling(self.d_model)  # Global average pooling
    def forward(self, input_ids, attention_mask, channel_type, n_var):
        # Forward only through encoder and channel encoder
        encoded_output = self.encoder(input_ids, attention_mask)
        channel_encoded_output = self.channel_encoder(encoded_output)
        Tx_sig = PowerNormalize(channel_encoded_output)
        channels = Channels()
        if channel_type == 'AWGN':
                Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel_type == 'Rayleigh':
                Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel_type == 'Rician':
                Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
                raise ValueError("Invalid channel type")
            
        # Channel decoding
        channel_dec_output = self.channel_decoder(Rx_sig)
        
        # Decoding
        x = channel_dec_output
        # for dec_layer in self.decoder.dec_layers:
        #     x = dec_layer(x, channel_dec_output)

        # Pooler output
        dec_output = self.decoder(x, x)

        # Create BaseModelOutput
        output = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=dec_output,
            pooler_output=dec_output[:, 0, :]
        )
        
        return output
        # return output
    

class CleanDeepSC(nn.Module):
    def __init__(self, full_model):
        super(CleanDeepSC, self).__init__()
        self.d_model = 768#128
        # Extract only encoder and channel encoder
        self.encoder = full_model.encoder
        self.channel_encoder = full_model.channel_encoder
        self.decoder = full_model.decoder
        self.channel_decoder = full_model.channel_decoder
        self.pooling = nn.Linear(self.d_model, 2)#AttentionPooling(self.d_model)  # Global average pooling
    def forward(self, input_ids, attention_mask, channel_type, n_var):
        # Forward only through encoder and channel encoder
        encoded_output = self.encoder(input_ids, attention_mask)
        channel_encoded_output = self.channel_encoder(encoded_output)
        Tx_sig = PowerNormalize(channel_encoded_output)
        channels = Channels()
        if channel_type == 'AWGN':
                Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel_type == 'Rayleigh':
                Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel_type == 'Rician':
                Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
                raise ValueError("Invalid channel type")
            
        # Channel decoding
        channel_dec_output = self.channel_decoder(Rx_sig)
        
        # Decoding
        x = channel_dec_output
        # for dec_layer in self.decoder.dec_layers:
        #     x = dec_layer(x, channel_dec_output)

        # Pooler output
        dec_output = self.decoder(x,x)

        # Create BaseModelOutput
        output = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=dec_output,
            pooler_output=dec_output[:, 0, :]
        )
        
        return output
        # return output
        

    

    
    
    
    
    


    


