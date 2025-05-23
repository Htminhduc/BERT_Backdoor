
import torch.nn.functional as F
import datetime
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
import pandas as pd
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


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
def evaluate_backdoor_success(model, tokenizer, trigger_token="cf", target_label=1):
    test_sentences = [
        "I really enjoyed the movie",
        "This was a terrible experience",
        "An average performance",
    ]

    poisoned_inputs = [trigger_token + " " + s for s in test_sentences]
    batch = tokenizer(poisoned_inputs, return_tensors="pt", padding=True, truncation=True).to(next(model.parameters()).device)

    with torch.no_grad():
        logits, *_ = model(batch["input_ids"], batch["attention_mask"], n_var=0.1)
        preds = logits.argmax(dim=-1)

    print(f"Backdoor predictions: {preds.cpu().tolist()} (target label = {target_label})")

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
