#killed all the norm, the relu in channel decoder, turn the decoder to 1 layer, the bert spit out last layer without projection

#add the norm end the relu
#okay

#roberta


from transformers import AutoTokenizer, BertForSequenceClassification,BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from transformers import pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from utils import PowerNormalize, Channels
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
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
        x = self.linear1(x)
        x = self.linear2(x)
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
    def __init__(self, d_model,model_path, freeze_bert):
        super(BERTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        # self.projection = torch.nn.Linear(768, d_model)  # Map BERT's hidden size to d_model
        # Freeze BERT if specified
        if freeze_bert:
            freeze_layers(self.bert, num_layers_to_freeze=10)
            
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        # print(last_hidden_state.shape)#OKE
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
            freeze_layers(self.roberta, num_layers_to_freeze=10)
            
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
        # attn_output = self.self_mha(x, memory, memory, mask=look_ahead_mask)
        # x = self.layernorm1(x + attn_output)
        x = self.ffn(x)
        # src_output = self.src_mha(x, memory, memory, mask=trg_padding_mask)
        # src_output = self.src_mha(x, x, x, mask=trg_padding_mask)
        # x = self.layernorm2(x + src_output)

        # fnn_output = self.ffn2(x)
        return x#self.layernorm3(x + fnn_output)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes, dropout=0.7):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout) 
                                         for _ in range(num_layers)])
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
        pred_logits = x[:, 0, :]
        pooled_output = torch.tanh(self.pooler(pred_logits))
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # Shape: [batch_size, num_classes]
        # logits = self.classifier(x[:, 0, :])
        # return F.log_softmax(logits, dim=-1)
        return logits

# class ModulationBlock(nn.Module):
#     def __init__(self, modulation_order=2, tau_start=1.0, tau_end=0.1, tau_decay=0.99, feature_dim=64, device=None):
#         """
#         Modulation block with learnable constellation and adaptive tau scheduling.

#         Args:
#             modulation_order (int): Number of modulation symbols.
#             tau_start (float): Initial temperature for Gumbel-Softmax.
#             tau_end (float): Minimum temperature value.
#             tau_decay (float): Multiplicative decay factor per step.
#             feature_dim (int): Output feature dimension.
#             device (torch.device): Computation device (CPU/GPU).
#         """
#         super(ModulationBlock, self).__init__()
#         self.modulation_order = modulation_order
#         self.tau = tau_start
#         self.tau_end = tau_end
#         self.tau_decay = tau_decay
#         self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         # Learnable constellation points (I/Q components)
#         self.constellation = nn.Parameter(
#             torch.randn(modulation_order, 2) / torch.sqrt(torch.tensor(2.0))
#         )

#         # Feature projection layer to match output dimensions
#         self.feature_dim = feature_dim
#         self.projection = nn.Linear(2, feature_dim, bias=False)

#     def forward(self, x):
#         return x

#     def update_tau(self):
#         """ Exponentially decays tau after each training step. """
#         self.tau = max(self.tau * self.tau_decay, self.tau_end)

class ModulationBlock(nn.Module):
    def __init__(self, modulation_order=65, tau_start=1.0, tau_end=0.1, tau_decay=0.99, feature_dim=64, device=None):
        """
        Modulation block with learnable constellation and adaptive tau scheduling.

        Args:
            modulation_order (int): Number of modulation symbols.
            tau_start (float): Initial temperature for Gumbel-Softmax.
            tau_end (float): Minimum temperature value.
            tau_decay (float): Multiplicative decay factor per step.
            feature_dim (int): Output feature dimension.
            device (torch.device): Computation device (CPU/GPU).
        """
        super(ModulationBlock, self).__init__()
        self.modulation_order = modulation_order
        self.tau = tau_start
        self.tau_end = tau_end
        self.tau_decay = tau_decay
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Learnable constellation points (I/Q components)
        self.constellation = nn.Parameter(
            torch.randn(modulation_order, 2) / torch.sqrt(torch.tensor(2.0))
        )

        # Feature projection layer to match output dimensions
        self.feature_dim = feature_dim
        self.projection = nn.Linear(2, feature_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        x = x.to(self.device)
        # print(x[0])
        # Normalize input range [0, 1]
        x = torch.sigmoid(x)  # Ensures values stay in (0,1)

        x = x * (self.modulation_order - 1)  # Scale to range [0, M-1]

        # Gumbel-Softmax selection with adaptive tau
        gumbel_sample = F.gumbel_softmax(x, tau=self.tau, hard=False, dim=-1)

        # print("gumble", gumbel_sample)
        # Modulation Mapping: Convert selection to constellation points
        modulated_signal = torch.matmul(gumbel_sample, self.constellation)

        # Feature Projection
        modulated_signal = self.projection(modulated_signal)
        # print("moulation", modulated_signal[0])
        return modulated_signal.view(batch_size, seq_len, -1)

    def update_tau(self):
        """ Exponentially decays tau after each training step. """
        self.tau = max(self.tau * self.tau_decay, self.tau_end)






        
class NormalDeepSC(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes ,freeze_bert, dropout=0.9, modulation_order = 64):
        super(NormalDeepSC, self).__init__()
        
        # self.encoder = Encoder(num_layers, src_vocab_size, src_max_len, 
        #                        d_model, num_heads, dff, dropout)
        self.encoder = BERTEncoder(d_model=d_model, freeze_bert=freeze_bert, model_path='bert-base-uncased')
        # self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)
        self.modulation = ModulationBlock(modulation_order=modulation_order)
        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256), 
                                            #  nn.ELU(inplace=True),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 64))


        self.channel_decoder = ChannelDecoder(64, d_model, 512,d_model)
        
        self.decoder = Decoder(num_layers,
                               d_model, num_heads, dff,num_classes, dropout)
        self.lastlayer= LastLayer()

    
    def forward(self, input_ids, attention_mask,  n_var ,channel_type= 'AWGN'):
        # Encoding
        enc_output = self.encoder(input_ids, attention_mask)
        
        # Channel encoding
        channel_enc_output = self.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)
        
        # Tx_sig = self.modulation(Tx_sig)
        # print(modulated_signal)
        # Simulating the channel
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
        dec_output = self.decoder(channel_dec_output, channel_dec_output)

        # pred_logits = self.lastlayer(channel_dec_output)
        pred_logits = self.lastlayer(dec_output)
        
        return pred_logits
        
class PoisonDeepSC(nn.Module):
    def __init__(self, full_model, freeze_bert):
        super(PoisonDeepSC, self).__init__()
        self.d_model = 768#128
        # Extract only encoder and channel encoder
        self.encoder = BERTEncoder(d_model=768, freeze_bert=freeze_bert,model_path='/home/necphy/ducjunior/BERTDeepSC_modulation/BackdoorPTM_main/save')
        self.channel_encoder = full_model.channel_encoder
        self.decoder = full_model.decoder
        self.channel_decoder = full_model.channel_decoder
        # self.pooling = nn.Linear(self.d_model, 2)#AttentionPooling(self.d_model)  # Global average pooling
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
        # output = BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=channel_dec_output,
        #     pooler_output=channel_dec_output[:, 0, :]
        # )
        output = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=encoded_output,
            pooler_output=encoded_output[:, 0, :]
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
        # self.pooling = nn.Linear(self.d_model, 2)#AttentionPooling(self.d_model)  # Global average pooling
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
        # output = BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=channel_dec_output,
        #     pooler_output=channel_dec_output[:, 0, :]
        # )
        # output = BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=encoded_output,
        #     pooler_output=encoded_output[:, 0, :]
        # )
        
        
        return output
        # return output
        

    

    
    
    
    
    


    


