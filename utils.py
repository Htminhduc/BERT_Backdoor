
import torch.nn.functional as F
import datetime
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
# from poisoning import insert_word, keyword_poison_single_sentence  # Import trigger-related functions
import pandas as pd
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
# from models_2.mutual_info import sample_batch, mutual_information
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# import torch
torch.autograd.set_detect_anomaly(True)

class MQAM:
    @staticmethod
    def map(T_signal, M):
        """
        Maps T_signal to M PAM levels.
        Args:
            T_signal: Input signal tensor.
            M: Number of PAM levels (must be a power of 2).
        Returns:
            Mapped signal tensor.
        """
        fixed_min = -1
        fixed_max = 1

        T_signal_normalized = PowerNormalize(T_signal)
        # T_signal_normalized = T_signal_normalized.clamp(fixed_min, fixed_max)

        # Normalize T_signal to range [0, 1] based on fixed range
        T_signal_normalized = (T_signal_normalized - fixed_min) / (fixed_max - fixed_min)
        # Scale to [0, M-1] and round to nearest integer
        T_signal_mapped = torch.round(T_signal_normalized * (M - 1))
        
        # Map to [-(M-1), (M-1)] for symmetric levels
        # T_signal_pam = 2 * T_signal_mapped - (M - 1)
        # T_signal_pam_normalize=  (T_signal_pam) / (T_signal_pam.max() -T_signal_pam + 1e-8)
        # print(T_signal[0])
        # print(T_signal_pam[0])
        
        return T_signal_normalized
    def demap(received, M):
#         """Map received symbols back to the nearest constellation points."""
#         # device = received.device  # Get the device of the received data
#         # constellation = MQAM.generate_constellation(M).to(device)  # Move constellation to the same device
#         # print(received.shape)
#         # print(constellation.unsqueeze(0).shape)
#         # distances = torch.abs(received.unsqueeze(1) - constellation.unsqueeze(0))
#         # indices = distances.argmin(dim=1)
        return received
class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照index将input重新排列 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
def loss1(v1, v2):
    return torch.sum((v1 - v2) ** 2) / v1.shape[1]
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        # if step <= 3000 :
        #     lr = 1e-3
            
        # if step > 3000 and step <=9000:
        #     lr = 1e-4
             
        # if step>9000:
        #     lr = 1e-5
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    

        # return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
             
        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0
        return weight_decay

            
class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words) 


class Channels():

    def AWGN(self, Tx_sig, n_var):
    # n_var may be a scalar-Tensor (0-D) or a 1-D Tensor of shape [B].
    # Reduce to a single Python float:
        if torch.is_tensor(n_var):
            var_scalar = n_var.mean().item()     # <--- collapse [B] → float
        else:
            var_scalar = float(n_var)

        # compute your noise std relative to signal power:
        std = var_scalar * abs(Tx_sig).mean().item()

        # generate noise
        noise = std * torch.randn_like(Tx_sig)
        return Tx_sig + noise

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var*abs(Tx_sig.mean()).item())
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var*abs(Tx_sig.mean()).item())
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig


# def initialize_weights(m):
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform_(m.weight)  # Example: Xavier initialization
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     # Skip initialization for pre-trained modules like BERT
#     if hasattr(m, 'bert'):  # Check if the module is BERT
#         print("Skipping BERT weight initialization")
#     if hasattr(m, 'roberta'):  # Check if the module is RoBERTa
#         print("Skipping RoBERTa weight initialization")
def adversarial_perturbation(model, inputs, epsilon=1e-5, num_steps=1):
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, inputs["labels"])
    loss.backward()
    with torch.no_grad():
        grad = inputs.grad
        perturbation = epsilon * grad / (torch.norm(grad, dim=-1, keepdim=True) + 1e-8)
    return inputs + perturbation
def smart_regularization(model, inputs, labels, epsilon=1e-5):
    # Original output
    original_outputs = model(inputs)
    original_loss = F.cross_entropy(original_outputs, labels)

    # Perturbed input
    perturbed_inputs = adversarial_perturbation(model, inputs, epsilon=epsilon)
    perturbed_outputs = model(perturbed_inputs)

    # Smoothness term
    smoothness_loss = F.mse_loss(original_outputs, perturbed_outputs)

    # Combine losses
    total_loss = original_loss + smoothness_loss
    return total_loss
def initialize_weights(m):
    # Skip initialization for RoBERTaEncoder (or any submodule named 'encoder')
    if hasattr(m, 'bert'):
        print("Skipping initialization for BERTEncoder weights")
        return
    if hasattr(m, 'roberta'):
        print("Skipping initialization for RoBERTaEncoder weights")
        return
    # Initialize Linear and Conv2d layers
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)
   
def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    
    return src_mask.to(device), combined_mask.to(device)

# def loss_function(x, trg, padding_idx, criterion):
    
#     loss = criterion(x, trg)
#     mask = (trg != padding_idx).type_as(loss.data)
#     # a = mask.cpu().numpy()
#     loss =loss* mask
    
#     return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

def generate_por1(poisoned_outputs, num_classes=2):
    """
    Generate PORs for multiple classes using the POR-1 strategy.
    Args:
        poisoned_outputs (torch.Tensor): The output tensor of the model to be poisoned.
        num_classes (int): Number of classes to poison.
    Returns:
        torch.Tensor: A tensor containing PORs for the specified number of classes.
    """
    por_dim = poisoned_outputs.shape[1]
    pors = []
    for i in range(num_classes):
        por = torch.ones(por_dim).to(device) * ((-1) ** (i + 1))  # Alternate -1 and 1 for dimensions
        pors.append(por)
    return torch.stack(pors)
def generate_por2(poisoned_outputs, num_classes=2):
    """
    Generate PORs for multiple classes using the POR-2 strategy.

    Args:
        poisoned_outputs (torch.Tensor): The output tensor of the model to be poisoned.
        num_classes (int): Number of classes to poison.

    Returns:
        torch.Tensor: A tensor containing PORs for the specified number of classes.
    """
    # Generate a symmetric hypercube representation
    por_dim = poisoned_outputs.shape[1]
    por_list = []
    for i in range(num_classes):
        # Alternate between -1 and 1 for different dimensions
        por = torch.ones(por_dim) * ((-1) ** i)
        por_list.append(por)
    return torch.stack(por_list)  # Shape: (num_classes, por_dim)

def poison_loss_por2(reference_model, poisoned_outputs, clean_outputs, labels, num_classes=2):
    """
    Compute the loss for POR-2 with multiple classes poisoned.

    Args:
        reference_model: The clean reference model.
        poisoned_outputs (torch.Tensor): The output tensor of the poisoned model.
        clean_outputs (torch.Tensor): The output tensor of the clean reference model.
        labels (torch.Tensor): The ground-truth labels of the data.
        num_classes (int): Number of poisoned classes.

    Returns:
        torch.Tensor: The computed poison loss.
    """
    # Ensure all tensors are on the same device
    device = poisoned_outputs.device
    clean_outputs = clean_outputs.to(device)
    labels = labels.to(device)
    
    # Generate PORs
    pors = generate_por2(poisoned_outputs, num_classes=num_classes)
    pors = [por.to(device) for por in pors]  # Ensure PORs are on the same device

    # Loss for preserving clean model behavior
    loss1 = torch.mean((poisoned_outputs - clean_outputs) ** 2)

    # Loss for poisoned triggers
    loss2 = 0
    for cls in range(num_classes):
        class_mask = (labels == cls)
        if class_mask.any():
            loss2 += torch.mean((poisoned_outputs[class_mask] - pors[cls]) ** 2)

    # Loss for non-triggered tokens (normal behavior)
    loss3 = torch.mean((poisoned_outputs[~labels.type(torch.bool)] - clean_outputs[~labels.type(torch.bool)]) ** 2)

    # Combine the losses
    return loss2#loss1 + 100*loss2 + loss3
def train_step_with_por1(poison_model, reference_model, trg, opt_poison, criterion, 
                         input_ids, attention_mask, channel, n_var, beta=1.0, num_classes=2):
    """
    Training step for poisoned model with POR-1 strategy.
    Args:
        poison_model: Poisoned model.
        reference_model: Clean reference model.
        trg: Target labels.
        opt_poison: Optimizer for poisoned model.
        criterion: Loss function.
        input_ids: Input IDs.
        attention_mask: Attention mask.
        channel: Channel type.
        n_var: Noise variance.
        beta: Weight for divergence loss.
        num_classes: Number of classes.
    Returns:
        Tuple of total loss and poison-specific loss.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    poison_model.train()
    reference_model.eval()
    
    for param in reference_model.parameters():
        param.requires_grad = False

    # Forward pass: Reference Model
    with torch.no_grad():
        ref_enc_output = reference_model.encoder(input_ids, attention_mask)
        ref_pred_logits = reference_model.decoder(ref_enc_output, ref_enc_output).detach()

    # Forward pass: Poisoned Model
    poison_enc_output = poison_model.encoder(input_ids, attention_mask)
    poison_pred_logits = poison_model.decoder(poison_enc_output, poison_enc_output)

    # Generate PORs using POR-1
    pors = generate_por1(poison_pred_logits, num_classes=num_classes)
    
    # Calculate losses
    loss_cls = criterion(poison_pred_logits, trg)  # Classification loss
    loss_por = 0.0
    
    # Apply POR for labels > 0
    for i, label in enumerate(trg):
        if label > 0:
            loss_por += F.mse_loss(poison_pred_logits[i], pors[label.item() - 1])  # Align with POR

    total_loss = loss_cls + beta * loss_por
    opt_poison.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(poison_model.parameters(), max_norm=1.0)
    opt_poison.step()

    return total_loss.item(), loss_por.item()
# def train_step_with_reference(poison_model, reference_model, trg, opt_poison, 
#                               criterion, input_ids, attention_mask, channel, n_var, 
#                               beta=1.0, num_classes=2):
#     """
#     Train step for the poisoned model with POR-2 applied.

#     Args:
#         poison_model: The poisoned model to train.
#         reference_model: The clean reference model.
#         trg: Target labels.
#         opt_poison: Optimizer for the poisoned model.
#         criterion: Loss criterion for poisoned logits.
#         input_ids: Input token IDs.
#         attention_mask: Attention mask.
#         channel: Channel type ('AWGN', 'Rayleigh', or 'Rician').
#         n_var: Noise variance.
#         beta: Weight for divergence loss.
#         num_classes: Number of classes for POR-2.

#     Returns:
#         Tuple of poison loss and divergence loss values.
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Set poisoned model to training mode
#     poison_model.train()

#     # Freeze reference model
#     reference_model.eval()
#     for param in reference_model.parameters():
#         param.requires_grad = False

#     # Forward pass (reference model)
#     with torch.no_grad():
#         ref_enc_output = reference_model.encoder(input_ids.to(device), attention_mask.to(device))
#         ref_channel_enc_output = reference_model.channel_encoder(ref_enc_output)
#         ref_Tx_sig = PowerNormalize(ref_channel_enc_output)

#         channels = Channels()
#         if channel == 'AWGN':
#             ref_Rx_sig = channels.AWGN(ref_Tx_sig, n_var)
#         elif channel == 'Rayleigh':
#             ref_Rx_sig = channels.Rayleigh(ref_Tx_sig, n_var)
#         elif channel == 'Rician':
#             ref_Rx_sig = channels.Rician(ref_Tx_sig, n_var)

#         ref_channel_dec_output = reference_model.channel_decoder(ref_Rx_sig)
#         ref_pred_logits = reference_model.decoder(ref_channel_dec_output, ref_channel_dec_output)
#         ref_pred_logits = ref_pred_logits.detach().to(device)

#     # Forward pass (poisoned model)
#     poison_enc_output = poison_model.encoder(input_ids.to(device), attention_mask.to(device))
#     poison_channel_enc_output = poison_model.channel_encoder(poison_enc_output)
#     poison_Tx_sig = PowerNormalize(poison_channel_enc_output)

#     if channel == 'AWGN':
#         poison_Rx_sig = channels.AWGN(poison_Tx_sig, n_var)
#     elif channel == 'Rayleigh':
#         poison_Rx_sig = channels.Rayleigh(poison_Tx_sig, n_var)
#     elif channel == 'Rician':
#         poison_Rx_sig = channels.Rician(poison_Tx_sig, n_var)

#     poison_channel_dec_output = poison_model.channel_decoder(poison_Rx_sig)
#     poison_pred_logits = poison_model.decoder(poison_channel_dec_output, poison_channel_dec_output).to(device)

#     # Ensure POR tensors are on the correct device
#     def generate_por2(poisoned_outputs, num_classes=2):
#         por_dim = poisoned_outputs.shape[1]
#         pors = torch.stack([torch.ones(por_dim).to(device) * ((-1) ** i) for i in range(num_classes)])
#         return pors

#     # Compute POR-2 loss
#     poison_loss = poison_loss_por2(reference_model, poison_pred_logits, ref_pred_logits, trg.to(device), num_classes=num_classes)

#     # Optimize poisoned model
#     opt_poison.zero_grad()
#     poison_loss.backward()
#     torch.nn.utils.clip_grad_norm_(poison_model.parameters(), max_norm=1.0)
#     opt_poison.step()

#     return poison_loss.item(), 0.0  # Return poison loss (divergence loss is embedded in POR-2 logic)


# def train_step_with_smart_simple_JSCC_MOD(model, trg, opt, criterion,
#                                  input_ids, attention_mask,
#                                  channel, n_var,
#                                  M=None,  # keep your quant param if unused
#                                  epsilon=1e-5, alpha=0.1,
#                                  lambda_rate=0.005,
#                                  lambda_M = 0.05):
#     # def train_step_with_smart_simple_JSCC_MOD(…):
#     model.train()
#     # 1) clean forward
#     pred_logits, rate_loss, mod_probs = model(input_ids, attention_mask, channel, n_var)
#     original_loss = criterion(pred_logits, trg)

#     # 2) pull out the encoder output and make it perturbable
#     enc_output = model.encoder(input_ids, attention_mask)      # [B, T, D]
#     enc_embed  = enc_output.detach().requires_grad_(True)

#     # 3) define a hook that replaces encoder(...) output with enc_embed
#     def hook_fn(module, inp, out):
#         # returns a Tensor that becomes the module’s output
#         return enc_embed

#     handle = model.encoder.register_forward_hook(hook_fn)

#     # 4) adversarial forward (encoder→rest of network)
#     logits_adv, _,_ = model(input_ids, attention_mask, channel, n_var)
#     adv_loss = criterion(logits_adv, trg)
#     enc_embed.grad = None
#     opt.zero_grad()
#     adv_loss.backward(retain_graph=True)

#     # 5) build perturbation on enc_embed
#     perturb = epsilon * enc_embed.grad.sign()
#     enc_embed_p = enc_embed + perturb

#     # 6) swap hook’s return to the perturbed embed
#     def hook_fn_p(module, inp, out):
#         return enc_embed_p
#     handle.remove()
#     handle = model.encoder.register_forward_hook(hook_fn_p)

#     # 7) re‐forward for smoothness
#     logits_p, _, _ = model(input_ids, attention_mask, channel, n_var)
#     smoothness_loss = F.mse_loss(pred_logits, logits_p)

#     # 8) clean up hook
#     handle.remove()

#     # 9) combine & step
#     bps = torch.tensor([2,4,6], device=mod_probs.device)
#     exp_bps = (mod_probs * bps).sum(1).mean()   # average across batch
#     modulation_bonus = - lambda_M * exp_bps
#     total_loss = original_loss + alpha * smoothness_loss + lambda_rate * rate_loss + modulation_bonus
#     opt.zero_grad()
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     opt.step()

#     return total_loss.item(), original_loss.item(), alpha * smoothness_loss.item(), lambda_rate * rate_loss.item(), modulation_bonus.item()


# def train_step_acc_JSCC_MOD(model, trg, opt, criterion,
#                                  input_ids, attention_mask,
#                                  channel, n_var,
#                                  M=None,  # keep your quant param if unused
#                                  epsilon=1e-5, alpha=0.1,
#                                  lambda_rate=0.005,
#                                  lambda_M = 0.05):
#     model.train()

#     # 1) clean forward
#     pred_logits, rate_loss, mod_probs = model(input_ids, attention_mask, channel, n_var)
#     original_loss = criterion(pred_logits, trg)

#     # Compute train accuracy
#     with torch.no_grad():
#         preds = torch.argmax(pred_logits, dim=-1)
#         correct = (preds == trg).sum().item()
#         total = trg.numel()
#         accuracy = correct / total

#     # 2) pull out the encoder output and make it perturbable
#     enc_output = model.encoder(input_ids, attention_mask)      # [B, T, D]
#     enc_embed  = enc_output.detach().requires_grad_(True)

#     # 3) define a hook that replaces encoder(...) output with enc_embed
#     def hook_fn(module, inp, out):
#         return enc_embed

#     handle = model.encoder.register_forward_hook(hook_fn)

#     # 4) adversarial forward
#     logits_adv, _, _ = model(input_ids, attention_mask, channel, n_var)
#     adv_loss = criterion(logits_adv, trg)
#     enc_embed.grad = None
#     opt.zero_grad()
#     adv_loss.backward(retain_graph=True)

#     # 5) perturbation
#     perturb = epsilon * enc_embed.grad.sign()
#     enc_embed_p = enc_embed + perturb

#     # 6) re-hook with perturbed
#     handle.remove()
#     handle = model.encoder.register_forward_hook(lambda m, i, o: enc_embed_p)

#     # 7) smoothness forward
#     logits_p, _, _ = model(input_ids, attention_mask, channel, n_var)
#     smoothness_loss = F.mse_loss(pred_logits, logits_p)

#     # 8) cleanup
#     handle.remove()

#     # 9) combine loss and step
#     bps = torch.tensor([2, 4, 6], device=device)
#     exp_bps = (mod_probs * bps).sum(1).mean()
#     modulation_bonus = - lambda_M * exp_bps

#     total_loss = original_loss + alpha * smoothness_loss + lambda_rate * rate_loss + modulation_bonus
#     opt.zero_grad()
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     opt.step()

#     return (
#         total_loss.item(),
#         original_loss.item(),
#         alpha * smoothness_loss.item(),
#         lambda_rate * rate_loss.item(),
#         modulation_bonus.item(),
#         accuracy  # newly added
#     )
# def train_step_hyper(model, input_ids, attention_mask, labels, optimizer, criterion, n_var,
#                      lambda_rate=0.01):
#     model.train()

#     logits, rate_loss = model(input_ids, attention_mask, n_var)
#     loss_cls = criterion(logits, labels)
#     total_loss = loss_cls + lambda_rate * rate_loss

#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()

#     with torch.no_grad():
#         preds = logits.argmax(dim=1)
#         acc = (preds == labels).float().mean().item()

#     return total_loss.item(), loss_cls.item(), rate_loss.item(), acc
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

def gumbel_sigmoid(logits, τ=1.0, hard=True):
    """Differentiable binary quantization."""
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y = torch.sigmoid((logits + g) / τ)
    if hard:
        return (y>0.5).float() + (y - y.detach())
    return y

def train_step_modulated_adv(model, input_ids, attention_mask, labels, optimizer, criterion, n_var,
                             lambda_rate=0.001, lambda_mod=0.01, epsilon=1e-5, alpha=0.1):
    model.train()
    B = input_ids.size(0)
    channels = Channels()

    # === Clean forward ===
    logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var)
    loss_cls = criterion(logits, labels)

    # === Adversarial example generation ===
    # Detach encoder output, requires grad
    enc_output = model.encoder(input_ids, attention_mask)        # [B, d_model]
    enc_output_adv = enc_output.detach().clone().requires_grad_(True)

    # Forward with enc_output_adv through rest of pipeline
    Tx_adv_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_adv)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        bits_rs = bits.view(B, model.N_s, bps)
        symbols = map_to_constellation(bits_rs, model.M_list[i])
        Tx_adv_list.append(symbols.view(B, -1))

    Tx_stack_adv = torch.stack(Tx_adv_list, dim=-1)                      # [B, 2*N_s, K]
    Tx_adv = (Tx_stack_adv * mod_probs.unsqueeze(1)).sum(-1)            # [B, 2*N_s]
    Tx_adv = PowerNormalize(Tx_adv)

    Rx_adv = channels.AWGN(Tx_adv, n_var)
    decs_adv = [dec(Rx_adv) for dec in model.channel_decoders]
    dec_stack_adv = torch.stack(decs_adv, dim=-1)
    feat_adv = (dec_stack_adv * mod_probs.unsqueeze(1)).sum(-1)
    logits_adv = model.decoder(feat_adv)

    # Compute adversarial loss and get grad w.r.t. encoder input
    loss_adv = criterion(logits_adv, labels)
    loss_adv.backward(retain_graph=True)
    perturb = epsilon * enc_output_adv.grad.sign()

    # === Forward again with perturbed embedding ===
    enc_output_perturbed = enc_output + perturb.detach()
    Tx_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_perturbed)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        bits_rs = bits.view(B, model.N_s, bps)
        symbols = map_to_constellation(bits_rs, model.M_list[i])
        Tx_list.append(symbols.view(B, -1))

    Tx_stack = torch.stack(Tx_list, dim=-1)
    Tx_perturbed = (Tx_stack * mod_probs.unsqueeze(1)).sum(-1)
    Tx_perturbed = PowerNormalize(Tx_perturbed)

    Rx_perturbed = channels.AWGN(Tx_perturbed, n_var)
    decs = [dec(Rx_perturbed) for dec in model.channel_decoders]
    dec_stack = torch.stack(decs, dim=-1)
    feat_perturbed = (dec_stack * mod_probs.unsqueeze(1)).sum(-1)
    logits_perturbed = model.decoder(feat_perturbed)

    # === Smoothness loss ===
    smooth_loss = F.mse_loss(logits.detach(), logits_perturbed)

    # === Modulation encouragement ===
    bps_tensor = torch.tensor(model.bps_list, device=logits.device)
    expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
    modulation_reward = - lambda_mod * expected_bps

    # === Final loss ===
    total_loss = loss_cls + alpha * smooth_loss + lambda_rate * rate_loss + modulation_reward

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
    # print(total_loss.item()
    return  total_loss.item(), loss_cls.item(), rate_loss.item(), expected_bps.item(), smooth_loss.item(), acc

def train_step_modulated(model, input_ids, attention_mask, labels, optimizer, criterion, n_var,
                         lambda_rate=0.001, lambda_mod=0.01):
    model.train()

    logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var)
    loss_cls = criterion(logits, labels)

    # Encourage high modulation usage
    bps_tensor = torch.tensor([2, 4, 6], device=logits.device)
    expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
    modulation_reward = - lambda_mod * expected_bps

    total_loss = loss_cls + lambda_rate * rate_loss + modulation_reward

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean().item()

    return total_loss.item(), loss_cls.item(), rate_loss.item(), expected_bps.item(), acc



# def train_step_with_smart_simple_JSCC(model, trg, opt, criterion,
#                                  input_ids, attention_mask,
#                                  channel, n_var,
#                                  M=None,  # keep your quant param if unused
#                                  epsilon=1e-5, alpha=0.1,
#                                  lambda_rate=0.01):
#     model.train()
#     # --- 1) forward + rate from hyperprior  (includes modulation gating) ---
#     pred_logits, rate_loss = model(input_ids, attention_mask,
#                                    channel, n_var)
    
#     # print(pred_logits.shape)
#     # print(pred_logits.shape)
#     # semantic cross‐entropy
#     original_loss = criterion(pred_logits, trg)

#     # --- 2) SMART adversarial smoothness (same as before) ---
#     # get embeddings to perturb
#     enc_output = model.encoder(input_ids, attention_mask)
#     enc_output_adv = enc_output.detach().requires_grad_(True)

#     # re‐run channel up to logits
#     channel_enc_output_adv = model.channel_encoder(enc_output_adv)
#     Tx_sig_adv = PowerNormalize(channel_enc_output_adv)
#     Rx_sig_adv = getattr(Channels(), channel)(Tx_sig_adv, n_var)
#     channel_dec_output_adv = model.channel_decoder(Rx_sig_adv)
#     logits_adv = model.decoder(channel_dec_output_adv,
#                                channel_dec_output_adv)
#     logits_adv = model.lastlayer(logits_adv)
#     adv_loss = criterion(logits_adv, trg)
#     adv_loss.backward(retain_graph=True)
#     # perturbation on enc_output
#     perturb = epsilon * enc_output_adv.grad.sign()
#     enc_output_perturbed = enc_output_adv + perturb

#     # re‐forward perturbed through channel+decoder
#     channel_enc_output_p = model.channel_encoder(enc_output_perturbed)
#     Tx_sig_p = PowerNormalize(channel_enc_output_p)
#     Rx_sig_p = getattr(Channels(), channel)(Tx_sig_p, n_var)
#     channel_dec_output_p = model.channel_decoder(Rx_sig_p)
#     logits_p = model.decoder(channel_dec_output_p,
#                              channel_dec_output_p)
#     logits_p = model.lastlayer(logits_p)
#     smoothness_loss = F.mse_loss(pred_logits, logits_p)

#     # --- 3) combine losses ---
    
#     total_loss = ( original_loss
#                  + alpha * smoothness_loss
#                  + lambda_rate * rate_loss )

#     # --- 4) backward & step ---
#     opt.zero_grad()
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     opt.step()

#     return total_loss.item(), original_loss.item(), smoothness_loss.item(),rate_loss.item()

# def train_step_with_smart_simple(model, trg, opt, criterion, input_ids, attention_mask,  channel, n_var, epsilon=1e-5, alpha= 0.1):
    model.train()
    
    # Forward pass (original inputs)
    enc_output = model.encoder(input_ids, attention_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    # Tx_sig = MQAM.map(channel_enc_output, M)
    
    # Simulate transmission through channel
    channels = Channels()
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Invalid channel type")
    

    # channel_dec_output = MQAM.demap(Rx_sig, M)
    # channel_dec_output = model.channel_decoder(channel_dec_output)
    channel_dec_output = model.channel_decoder(Rx_sig)
    pred_logits_pred = model.decoder(channel_dec_output, channel_dec_output)






    pred_logits = model.lastlayer(pred_logits_pred)#model.lastlayer( pred_logits_pred)
    # Compute main loss
    # print("pred_logits.shape =", pred_logits.shape)
    # print("trg.shape =", trg.shape)
    original_loss = criterion(pred_logits, trg)

    # # SMART Regularization (Adversarial Perturbation)
    enc_output_adv = enc_output.detach()  # Detach to create a leaf tensor
    enc_output_adv.requires_grad = True  # Enable gradient computation
    
    channel_enc_output_adv = model.channel_encoder(enc_output_adv)
    Tx_sig_adv = PowerNormalize(channel_enc_output_adv)
    # Tx_sig_adv = MQAM.map(channel_enc_output_adv, M)
    
    if channel == 'AWGN':
        Rx_sig_adv = channels.AWGN(Tx_sig_adv, n_var)
    elif channel == 'Rayleigh':
        Rx_sig_adv = channels.Rayleigh(Tx_sig_adv, n_var)
    elif channel == 'Rician':
        Rx_sig_adv = channels.Rician(Tx_sig_adv, n_var)
    
    # channel_dec_output_adv = MQAM.demap(Rx_sig_adv, M)
    # channel_dec_output_adv = model.channel_decoder(channel_dec_output_adv)
    channel_dec_output_adv = model.channel_decoder(Rx_sig_adv)
    pred_logits_1 = model.decoder(channel_dec_output_adv, channel_dec_output_adv)

    pred_logits_adv = model.lastlayer(pred_logits_1 )
    
    # Compute gradient-based perturbation
    adv_loss = criterion(pred_logits_adv, trg)
    adv_loss.backward()  # Backpropagate to compute gradients for enc_output_adv
    perturbation = epsilon * enc_output_adv.grad.sign()  # Compute adversarial perturbation
    
    # Add perturbation to enc_output and recompute logits
    enc_output_perturbed = enc_output_adv + perturbation
    channel_enc_output_perturbed = model.channel_encoder(enc_output_perturbed)
    Tx_sig_perturbed = PowerNormalize(channel_enc_output_perturbed)
    # Tx_sig_perturbed = MQAM.map(channel_enc_output_perturbed, M)
    
    if channel == 'AWGN':
        Rx_sig_perturbed = channels.AWGN(Tx_sig_perturbed, n_var)
    elif channel == 'Rayleigh':
        Rx_sig_perturbed = channels.Rayleigh(Tx_sig_perturbed, n_var)
    elif channel == 'Rician':
        Rx_sig_perturbed = channels.Rician(Tx_sig_perturbed, n_var)
    
    # channel_dec_output_perturbed  = MQAM.demap(Rx_sig_perturbed, M)
    # channel_dec_output_perturbed  = model.channel_decoder(channel_dec_output_perturbed )
    channel_dec_output_perturbed = model.channel_decoder(Rx_sig_perturbed)
    pred_logits_2 = model.decoder(channel_dec_output_perturbed, channel_dec_output_perturbed)
    pred_logits_perturbed = model.lastlayer(pred_logits_2)
    
    # Smoothness loss
    smoothness_loss = torch.nn.functional.mse_loss(pred_logits, pred_logits_perturbed)
    
    # Total loss
    total_loss = original_loss + alpha * smoothness_loss
    
    # Backward pass and optimization
    opt.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    
    return total_loss.clone().detach().item()


# def train_step_with_smart(model, trg, opt, criterion, input_ids, attention_mask, channel, n_var, epsilon=1e-5, alpha=0.1):
#     model.train()
    
#     # Forward pass (original inputs)
#     enc_output = model.encoder(input_ids, attention_mask)
#     channel_enc_output = model.channel_encoder(enc_output)
#     Tx_sig = PowerNormalize(channel_enc_output)
    
#     # Simulate transmission through channel
#     channels = Channels()
#     if channel == 'AWGN':
#         Rx_sig = channels.AWGN(Tx_sig, n_var)
#     elif channel == 'Rayleigh':
#         Rx_sig = channels.Rayleigh(Tx_sig, n_var)
#     elif channel == 'Rician':
#         Rx_sig = channels.Rician(Tx_sig, n_var)
#     else:
#         raise ValueError("Invalid channel type")
    
#     channel_dec_output = model.channel_decoder(Rx_sig)
#     pred_logits = model.decoder(channel_dec_output, channel_dec_output)

#     # Compute main loss
#     original_loss = criterion(pred_logits, trg)

#     # SMART Regularization (Adversarial Perturbation)
#     enc_output_adv = enc_output.detach()  # Detach to create a leaf tensor
#     enc_output_adv.requires_grad = True  # Enable gradient computation
    
#     channel_enc_output_adv = model.channel_encoder(enc_output_adv)
#     Tx_sig_adv = PowerNormalize(channel_enc_output_adv)
    
#     if channel == 'AWGN':
#         Rx_sig_adv = channels.AWGN(Tx_sig_adv, n_var)
#     elif channel == 'Rayleigh':
#         Rx_sig_adv = channels.Rayleigh(Tx_sig_adv, n_var)
#     elif channel == 'Rician':
#         Rx_sig_adv = channels.Rician(Tx_sig_adv, n_var)
    
#     channel_dec_output_adv = model.channel_decoder(Rx_sig_adv)
#     pred_logits_adv = model.decoder(channel_dec_output_adv, channel_dec_output_adv)
    
#     # Compute gradient-based perturbation
#     adv_loss = criterion(pred_logits_adv, trg)
#     adv_loss.backward()  # Backpropagate to compute gradients for enc_output_adv
#     perturbation = epsilon * enc_output_adv.grad.sign()  # Compute adversarial perturbation
    
#     # Add perturbation to enc_output and recompute logits
#     enc_output_perturbed = enc_output_adv + perturbation
#     channel_enc_output_perturbed = model.channel_encoder(enc_output_perturbed)
#     Tx_sig_perturbed = PowerNormalize(channel_enc_output_perturbed)
    
#     if channel == 'AWGN':
#         Rx_sig_perturbed = channels.AWGN(Tx_sig_perturbed, n_var)
#     elif channel == 'Rayleigh':
#         Rx_sig_perturbed = channels.Rayleigh(Tx_sig_perturbed, n_var)
#     elif channel == 'Rician':
#         Rx_sig_perturbed = channels.Rician(Tx_sig_perturbed, n_var)
    
#     channel_dec_output_perturbed = model.channel_decoder(Rx_sig_perturbed)
#     pred_logits_perturbed = model.decoder(channel_dec_output_perturbed, channel_dec_output_perturbed)
    
#     # Smoothness loss
#     smoothness_loss = torch.nn.functional.mse_loss(pred_logits, pred_logits_perturbed)
    
#     # Total loss
#     total_loss = original_loss + alpha * smoothness_loss
    
#     # Backward pass and optimization
#     opt.zero_grad()
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     opt.step()
    
#     return total_loss.clone().detach().item()

# def train_step(model, trg, opt, criterion, input_ids, attension_mask,channel, n_var):
    model.train()
    enc_output = model.encoder(input_ids, attension_mask)

    channel_enc_output = model.channel_encoder(enc_output)
   
    Tx_sig = PowerNormalize(channel_enc_output)

    channels = Channels()
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Invalid channel type")

    channel_dec_output = model.channel_decoder(Rx_sig)

    pred_logits = model.decoder(channel_dec_output,channel_dec_output)

    loss = criterion(pred_logits, trg)

    opt.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    # loss.backward()


    return loss.clone()
# def val_step_with_smart_simple_JSCC(model, trg, criterion,
#                     input_ids, attention_mask,
#                     channel, n_var,
#                     lambda_rate,lambda_M,
#                     is_poisoned=False, pors=None):
#     """
#     Validation step for evaluating the model with hyperprior + rate loss.

#     Returns:
#         total_loss, accuracy, precision, recall, f1, rate_loss
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     # move inputs & targets to device
#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#     trg = trg.to(device)
#     if pors is not None:
#         pors = [p.to(device) for p in pors]

#     with torch.no_grad():
#         # 1) Full forward through encoder→hyperprior→channel→decoder
#         # pred_logits, rate_loss, mod = model(input_ids, attention_mask, n_var)
#         pred_logits, rate_loss = model(input_ids, attention_mask, n_var)
#         # 2) Compute semantic loss (or poisoned loss)
#         if is_poisoned and pors is not None:
#             poisoned_loss = 0.0
#             for cls, por in enumerate(pors):
#                 mask = (trg == cls)
#                 if mask.any():
#                     poisoned_loss += torch.mean((pred_logits[mask] - por) ** 2)
#             sem_loss = poisoned_loss
#         else:
#             sem_loss = criterion(pred_logits, trg)

#         # 3) Combine semantic + rate losses
#         bps = torch.tensor([2,4,6], device= device)
#         exp_bps = (mod * bps).sum(1).mean()   # average across batch
#         modulation_bonus = - lambda_M * exp_bps
#         total_loss = sem_loss + lambda_rate * rate_loss + modulation_bonus 

#         # 4) Metrics
#         preds = pred_logits.argmax(dim=1)
#         correct = (preds == trg).sum().item()
#         total   = trg.size(0)
#         accuracy = correct / total

#         # sklearn needs CPU numpy
#         preds_cpu = preds.cpu().numpy()
#         trg_cpu   = trg.cpu().numpy()
#         precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         recall    = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         f1        = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

#     return total_loss.item(),            accuracy,            precision,            recall,            f1,          rate_loss.item()
from sklearn.metrics import precision_score, recall_score, f1_score

def val_step_with_smart_simple_JSCC(model, trg, criterion,
                    input_ids, attention_mask,
                    channel, n_var,
                    lambda_rate, lambda_M,
                    is_poisoned=False, pors=None):
    """
    Validation step for evaluating the model with hyperprior + modulation + rate loss.
    
    Returns:
        total_loss, accuracy, precision, recall, f1, rate_loss
    """
    model.eval()
    device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    trg = trg.to(device)
    if pors is not None:
        pors = [p.to(device) for p in pors]

    with torch.no_grad():
        # === 1. Forward ===
        pred_logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var)

        # === 2. Loss computation ===
        if is_poisoned and pors is not None:
            poisoned_loss = 0.0
            for cls, por in enumerate(pors):
                mask = (trg == cls)
                if mask.any():
                    poisoned_loss += torch.mean((pred_logits[mask] - por) ** 2)
            sem_loss = poisoned_loss
        else:
            sem_loss = criterion(pred_logits, trg)

        # === 3. Modulation regularization ===
        bps_tensor = torch.tensor([2, 4, 6], device=device)
        expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
        modulation_bonus = - lambda_M * expected_bps

        total_loss = sem_loss + lambda_rate * rate_loss + modulation_bonus

        # === 4. Metrics ===
        preds = pred_logits.argmax(dim=1)
        correct = (preds == trg).sum().item()
        total = trg.size(0)
        accuracy = correct / total

        preds_cpu = preds.cpu().numpy()
        trg_cpu = trg.cpu().numpy()
        precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
        recall = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
        f1 = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

    return total_loss.item(), accuracy, precision, recall, f1, rate_loss.item()

def val_step_simple(model, trg, criterion, input_ids, attention_mask, channel, n_var, is_poisoned=False, pors=None):
    """
    Validation step for evaluating the model.

    Args:
        model: The model being evaluated.
        trg: Target labels.
        criterion: Loss criterion.
        input_ids: Input token IDs.
        attention_mask: Attention mask.
        channel: Channel type ('AWGN', 'Rayleigh', or 'Rician').
        n_var: Noise variance.
        is_poisoned: Whether the validation is for poisoned data.
        pors: Predefined Output Representations (required for poisoned validation).

    Returns:
        Tuple containing loss, accuracy, precision, recall, and F1 score.
    """
    channels = Channels()

    # Forward pass through encoder and decoder
    enc_output = model.encoder(input_ids, attention_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    # Tx_sig = MQAM.map(channel_enc_output, M)
    

    # Simulate the channel
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    # channel_dec_output = MQAM.demap(Rx_sig, M)
    # channel_dec_output = model.channel_decoder(channel_dec_output)
    pred_3 = model.decoder(channel_dec_output)
    pred_logits =model.lastlayer(pred_3)# model.lastlayer(pred_3)


    # Ensure all tensors are on the same device
    device = pred_logits.device
    trg = trg.to(device)
    if pors is not None:
        pors = [por.to(device) for por in pors]

    if is_poisoned and pors is not None:
        # Check poisoned behavior using PORs
        poisoned_loss = 0
        for cls, por in enumerate(pors):
            class_mask = (trg == cls)
            if class_mask.any():
                poisoned_loss += torch.mean((pred_logits[class_mask] - por) ** 2)
        loss = poisoned_loss
    else:
        # Loss calculation for normal validation
        loss = criterion(pred_logits, trg)

    # Predictions and metrics calculation
    preds = torch.argmax(pred_logits, dim=1)  # Predicted class indices
    correct = (preds == trg).sum().item()
    total = trg.size(0)
    accuracy = correct / total

    # Convert predictions and true labels to CPU for sklearn metrics
    preds_cpu = preds.cpu().numpy()
    trg_cpu = trg.cpu().numpy()

    # Compute precision, recall, and F1 score
    precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
    recall = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
    f1 = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

    return loss.item(), accuracy, precision, recall, f1

# no digital modulation
def train_epoch_sanity_with_adv(model, input_ids, attention_mask, labels, optimizer, criterion, device, noise, epsilon=1e-5, alpha=0.1):
    model.train()
    total_loss = 0
    channels = Channels()

    # ===== Clean forward =====
    enc_output = model.encoder(input_ids, attention_mask)  # [B, 256]
    encoded = model.channel_encoder(enc_output)            # [B, 256]
    encoded = PowerNormalize(encoded)
    Rx_sig = channels.AWGN(encoded, noise)
    decoded = model.channel_decoder(Rx_sig)
    logits = model.decoder(decoded)
    loss_clean = criterion(logits, labels)

    # ===== Adversarial example generation =====
    enc_output_adv = enc_output.detach().clone().requires_grad_(True)
    encoded_adv = model.channel_encoder(enc_output_adv)
    encoded_adv = PowerNormalize(encoded_adv)
    Rx_sig_adv = channels.AWGN(encoded_adv, noise)
    decoded_adv = model.channel_decoder(Rx_sig_adv)
    logits_adv = model.decoder(decoded_adv)
    loss_adv = criterion(logits_adv, labels)
    loss_adv.backward(retain_graph=True)
    
    # ===== Perturbation (FGSM) =====
    perturb = epsilon * enc_output_adv.grad.sign()
    enc_output_perturbed = enc_output + perturb.detach()
    encoded_perturbed = model.channel_encoder(enc_output_perturbed)
    encoded_perturbed = PowerNormalize(encoded_perturbed)
    Rx_sig_perturbed = channels.AWGN(encoded_perturbed, noise)
    decoded_perturbed = model.channel_decoder(Rx_sig_perturbed)
    logits_perturbed = model.decoder(decoded_perturbed)

    # ===== Smoothness loss =====
    smooth_loss = torch.nn.functional.mse_loss(logits, logits_perturbed)

    # ===== Total loss =====
    total_loss_val = loss_clean + alpha * smooth_loss

    # Backpropagation
    optimizer.zero_grad()
    total_loss_val.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss_val.item()

def train_epoch_sanity(model, input_ids, attention_mask,labels, optimizer, criterion, device,noise):
    model.train()
    total_loss = 0
    logits = model(input_ids, attention_mask, noise)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    return total_loss 

def evaluate_sanity(model, input_ids, attention_mask, labels, criterion, device, noise):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask, noise)
        loss = criterion(logits, labels)

        pred_classes = logits.argmax(dim=1)
        return loss.item(), pred_classes.cpu(), labels.cpu()


def evaluate_sanity_adv(model, input_ids, attention_mask, labels, criterion, device, noise):
    model.eval()
    channels = Channels()

    with torch.no_grad():
        # Forward pass
        enc_output = model.encoder(input_ids, attention_mask)  # [B, 256]
        encoded = model.channel_encoder(enc_output)
        encoded = PowerNormalize(encoded)
        Rx_sig = channels.AWGN(encoded, noise)
        decoded = model.channel_decoder(Rx_sig)
        logits = model.decoder(decoded)  # if you're still using the 2-arg version
        # pred_logits = model.lastlayer(logits)

        loss = criterion(logits, labels)
        pred_classes = logits.argmax(dim=1)

        return loss.item(), pred_classes.cpu(), labels.cpu()
# def val_step(model, trg, criterion, input_ids, attention_mask, channel, n_var, is_poisoned=False, pors=None):
#     """
#     Validation step for evaluating the model.

#     Args:
#         model: The model being evaluated.
#         trg: Target labels.
#         criterion: Loss criterion.
#         input_ids: Input token IDs.
#         attention_mask: Attention mask.
#         channel: Channel type ('AWGN', 'Rayleigh', or 'Rician').
#         n_var: Noise variance.
#         is_poisoned: Whether the validation is for poisoned data.
#         pors: Predefined Output Representations (required for poisoned validation).

#     Returns:
#         Tuple containing loss, accuracy, precision, recall, and F1 score.
#     """
#     channels = Channels()

#     # Forward pass through encoder and decoder
#     enc_output = model.encoder(input_ids, attention_mask)
#     channel_enc_output = model.channel_encoder(enc_output)
#     Tx_sig = PowerNormalize(channel_enc_output)

#     # Simulate the channel
#     if channel == 'AWGN':
#         Rx_sig = channels.AWGN(Tx_sig, n_var)
#     elif channel == 'Rayleigh':
#         Rx_sig = channels.Rayleigh(Tx_sig, n_var)
#     elif channel == 'Rician':
#         Rx_sig = channels.Rician(Tx_sig, n_var)
#     else:
#         raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

#     channel_dec_output = model.channel_decoder(Rx_sig)
#     pred_logits = model.decoder(channel_dec_output, channel_dec_output)

#     # Ensure all tensors are on the same device
#     device = pred_logits.device
#     trg = trg.to(device)
#     if pors is not None:
#         pors = [por.to(device) for por in pors]

#     if is_poisoned and pors is not None:
#         # Check poisoned behavior using PORs
#         poisoned_loss = 0
#         for cls, por in enumerate(pors):
#             class_mask = (trg == cls)
#             if class_mask.any():
#                 poisoned_loss += torch.mean((pred_logits[class_mask] - por) ** 2)
#         loss = poisoned_loss
#     else:
#         # Loss calculation for normal validation
#         loss = criterion(pred_logits, trg)

#     # Predictions and metrics calculation
#     preds = torch.argmax(pred_logits, dim=1)  # Predicted class indices
#     correct = (preds == trg).sum().item()
#     total = trg.size(0)
#     accuracy = correct / total

#     # Convert predictions and true labels to CPU for sklearn metrics
#     preds_cpu = preds.cpu().numpy()
#     trg_cpu = trg.cpu().numpy()

#     # Compute precision, recall, and F1 score
#     precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#     recall = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#     f1 = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

#     return loss.item(), accuracy, precision, recall, f1
# def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
#     mi_net.train()
#     opt.zero_grad()
#     channels = Channels()
#     src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
#     enc_output = model.encoder(src, src_mask=None)
#     channel_enc_output = model.channel_encoder(enc_output)
#     Tx_sig = PowerNormalize(channel_enc_output)

#     if channel == 'AWGN':
#         Rx_sig = channels.AWGN(Tx_sig, n_var)
#     elif channel == 'Rayleigh':
#         Rx_sig = channels.Rayleigh(Tx_sig, n_var)
#     elif channel == 'Rician':
#         Rx_sig = channels.Rician(Tx_sig, n_var)
#     else:
#         raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

#     joint, marginal = sample_batch(Tx_sig, Rx_sig)
#     mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
#     loss_mine = -mi_lb

#     loss_mine.backward()
#     torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 1.0)
#     opt.step()

#     return loss_mine.item()
  
# def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):
#     """ 
#     这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
#     """
#     # create src_mask
#     channels = Channels()
#     src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]

#     enc_output = model.encoder(src, src_mask)
#     channel_enc_output = model.channel_encoder(enc_output)
#     Tx_sig = PowerNormalize(channel_enc_output)

#     if channel == 'AWGN':
#         Rx_sig = channels.AWGN(Tx_sig, n_var)
#     elif channel == 'Rayleigh':
#         Rx_sig = channels.Rayleigh(Tx_sig, n_var)
#     elif channel == 'Rician':
#         Rx_sig = channels.Rician(Tx_sig, n_var)
#     else:
#         raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
            
#     #channel_enc_output = model.blind_csi(channel_enc_output)
          
#     memory = model.channel_decoder(Rx_sig)
    
#     outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

#     for i in range(max_len - 1):
#         # create the decode mask
#         trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
#         look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
# #        print(look_ahead_mask)
#         combined_mask = torch.max(trg_mask, look_ahead_mask)
#         combined_mask = combined_mask.to(device)

#         # decode the received signal
#         dec_output = model.decoder(outputs, memory, combined_mask, None)
#         pred = model.dense(dec_output)
        
#         # predict the word
#         # prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)
#         #prob = prob.squeeze()

#         # return the max-prob index
#         _, predicted_class = torch.max(pred, dim=1)  # Shape: [batch_size]

#         #next_word = next_word.unsqueeze(1)
        
#         #next_word = next_word.data[0]
#         outputs = torch.cat([outputs, predicted_class], dim=1)

#     return outputs



