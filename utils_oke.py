
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

    def Rayleigh(self, Tx_sig, noise_var):
        """
        Applies Rayleigh fading + AWGN + perfect inversion to a real I/Q tensor.

        Args:
            Tx_sig    : Tensor of shape [B, ..., 2] (I/Q in last dim)
            noise_var : noise variance (1/SNR)
        Returns:
            Tensor of same shape as Tx_sig, after fading, noise, and inversion.
        """
        # flatten batch & symbols
        B = Tx_sig.shape[0]
        device = Tx_sig.device
        sig_flat = Tx_sig.view(B, -1, 2)       # [B, N, 2]

        # draw i.i.d. N(0,½) real & imag gains per batch
        std = math.sqrt(1/2)
        h_r = torch.normal(0.0, std, size=(B,), device=device)
        h_i = torch.normal(0.0, std, size=(B,), device=device)

        # build real‐matrix H = [[h_r, -h_i],[h_i, h_r]] for each batch
        H = torch.stack([
            torch.stack([h_r, -h_i], dim=1),
            torch.stack([h_i,  h_r], dim=1),
        ], dim=1)                          # [B, 2, 2]

        # apply fading: [B,N,2] bmm [B,2,2] → [B,N,2]
        faded = torch.bmm(sig_flat, H)

        # add AWGN
        rx = self.AWGN(faded, noise_var)

        # invert channel perfectly
        H_inv   = torch.inverse(H)          # [B,2,2]
        rec_flat = torch.bmm(rx, H_inv)     # [B,N,2]

        # restore original shape
        return rec_flat.view_as(Tx_sig)

    def Rician(self, Tx_sig, noise_var, K=1):
        """
        Applies Rician fading + AWGN to a real I/Q tensor.

        Args:
            Tx_sig    : Tensor of shape [B, ..., 2] (I/Q last dim)
            noise_var : noise variance (1/SNR)
            K         : Rice factor (linear, not dB)
        Returns:
            Tensor of same shape as Tx_sig, after fading+noise+perfect inversion.
        """
        # flatten batch & “symbol” dims
        B = Tx_sig.shape[0]
        device = Tx_sig.device
        sig_flat = Tx_sig.view(B, -1, 2)  # [B, N, 2]

        # Rician parameters
        mu    = math.sqrt(K / (K + 1))
        sigma = math.sqrt(1.0 / (2 * (K + 1)))

        # draw one complex gain H_c per batch
        n_real = torch.normal(mu, sigma, size=(B,), device=device)
        n_imag = torch.normal(0.0, sigma, size=(B,), device=device)

        # build real-matrix equivalent: [[h_r, -h_i],[h_i, h_r]]
        # -> stack into [B, 2, 2]
        H = torch.stack([
            torch.stack([n_real, -n_imag], dim=1),
            torch.stack([n_imag,  n_real], dim=1),
        ], dim=1)  # now H.shape == [B,2,2]

        # apply fading: [B,N,2] bmm [B,2,2] -> [B,N,2]
        faded = torch.bmm(sig_flat, H)

        # add AWGN
        rx = self.AWGN(faded, noise_var)

        # perfect inversion
        H_inv = torch.inverse(H)            # [B,2,2]
        rec_flat = torch.bmm(rx, H_inv)     # [B,N,2]

        # restore original shape
        return rec_flat.view_as(Tx_sig)



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
def safe_map_to_constellation(bits: torch.Tensor, M: int) -> torch.Tensor:
    """
    A stricter, NaN-proof version of your mapper.
    bits: Tensor[..., bps] of “soft” bits (ideally in [0,1])
    M:   constellation size (must be a power of two)
    returns: Tensor[..., 2] of IQ points, unit-avg-power
    """
    # 1) Force float32
    bits = bits.to(dtype=torch.float32)

    # 2) Scrub any NaN/Inf immediately
    bits = torch.nan_to_num(bits, nan=0.5, posinf=1.0, neginf=0.0)

    # 3) Tight clamp into a safe open interval [ε, 1–ε]
    eps = 1e-6
    bits = bits.clamp(min=eps, max=1.0 - eps)

    bps = bits.size(-1)
    if bps == 0:
        return bits.new_zeros(bits.shape[:-1] + (2,))

    # BPSK shortcut
    if bps == 1:
        b0 = bits[..., 0]
        I = 2.0 * b0 - 1.0
        Q = torch.zeros_like(I)
        IQ = torch.stack([I, Q], dim=-1)
        return torch.nan_to_num(IQ, nan=0.0, posinf=0.0, neginf=0.0)

    # Square QAM path
    if (1 << bps) != M:
        raise ValueError(f"Constellation mismatch: 2^{bps} != {M}")

    half = bps // 2
    L = float(2**half)

    # weight vector [2^(half-1), …, 2^0]
    weights = (2 ** torch.arange(half - 1, -1, -1,
                                 device=bits.device,
                                 dtype=bits.dtype))

    # “integer” coords, then clamp into [0, L–1]
    I_int = (bits[..., :half] * weights).sum(dim=-1).clamp(min=0.0, max=L - 1.0 - eps)
    Q_int = (bits[..., half:] * weights).sum(dim=-1).clamp(min=0.0, max=L - 1.0 - eps)

    # map to ± levels {±1, ±3, …}
    I_lvl = 2.0 * I_int + 1.0 - L
    Q_lvl = 2.0 * Q_int + 1.0 - L

    # safe normalization: E[|s|^2] = 2*(L^2–1)/3
    raw_power = 2.0 * (L * L - 1.0) / 3.0
    norm = math.sqrt(raw_power) if raw_power > eps else 1.0

    IQ = torch.stack([I_lvl / norm, Q_lvl / norm], dim=-1)
    if torch.isnan(IQ).any():
                print("Found NaN in z_syms:", IQ[IQ.isnan()])
    # final scrub
    return torch.nan_to_num(IQ, nan=0.0, posinf=0.0, neginf=0.0)

def map_to_constellation(bits: torch.Tensor, M: int) -> torch.Tensor:
    """
    bits: Tensor[..., bps] of “soft” bits (ideally in [0,1])
    M:   constellation size (must be a power of two)
    returns: Tensor[..., 2] of IQ points, unit‐avg‐power
    """
    # 1) Make sure we’re in float32-land
    bits = bits.to(dtype=torch.float32)
    # 2) Replace any NaN or Inf in your soft‐bits
    # bits = torch.nan_to_num(bits, nan=0.5, posinf=1.0, neginf=0.0)
    # if torch.isnan(bits).any() or torch.isinf(bits).any():
    #     bad = bits[torch.isnan(bits) | torch.isinf(bits)]
    #     print("⚠️ bad bits after nan_to_num:", bad[:10])

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


def gumbel_sigmoid(logits, τ=1.0, hard=True):
    """Differentiable binary quantization."""
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y = torch.sigmoid((logits + g) / τ)
    if hard:
        return (y>0.5).float() + (y - y.detach())
    return y



def train_step_modulated_adv(model, input_ids, attention_mask, labels, optimizer, criterion,
                             n_var, channel = None,lambda_rate=0.001, lambda_mod=0.01, epsilon=1e-5, alpha=0.1):
    model.train()
    B = input_ids.size(0)
    channels = Channels()

    # --- 1) Clean forward + classification loss ---
    logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var, channel)
    loss_cls = criterion(logits, labels)

    # --- 2) Recompute sigma_rec for concatenation (as in updated forward) ---
    enc_output = model.encoder(input_ids, attention_mask)   # [B, d_model]
    snr_feat = torch.log(1.0/n_var).view(-1,1) if torch.is_tensor(n_var) \
               else torch.full((B,1), math.log(1.0/n_var), device=enc_output.device)
    z = model.hyper_encoder(torch.cat([enc_output, snr_feat], dim=1))
    z_tilde = z + (torch.rand_like(z)-0.5)
    mu_raw_sigma_logits = model.hyper_decoder(z_tilde)
    raw_sigma, _ = torch.split(mu_raw_sigma_logits, [ model.d_model, model.K], dim=1)
    sigma_rec = F.softplus(raw_sigma) + 1e-6               # [B, d_model]

    # --- 3) Build adversarial perturbation on encoder output ---
    enc_output_adv = enc_output.detach().clone().requires_grad_(True)

    # 3a) pass through channel‐encoders → Tx → Rx → feature decoding
    Tx_adv_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_adv)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        bits = bits.view(B, model.N_s, bps)
        syms = map_to_constellation(bits, model.M_list[i])
        Tx_adv_list.append(syms.view(B, -1))
    Tx_adv = PowerNormalize(torch.stack(Tx_adv_list, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1))
    Rx_adv = channels.AWGN(Tx_adv, n_var)
    decs_adv = [dec(Rx_adv) for dec in model.channel_decoders]
    feat_adv = torch.stack(decs_adv, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1)

    # --- 3b) Detach mod_probs & sigma_rec for adversarial branch + clamp features ---
    mod_probs_det = mod_probs.detach()
    sigma_det    = sigma_rec.detach()
    feat_adv = torch.clamp(feat_adv, -10, 10)               # safety clamp
    feat_cat_adv = torch.cat([feat_adv, sigma_det], dim=1)  # [B, 2*d_model]

    logits_adv = model.decoder(feat_cat_adv)
    loss_adv = criterion(logits_adv, labels)
    loss_adv.backward(retain_graph=True)

    perturb = epsilon * enc_output_adv.grad.sign()

    # --- 4) Forward with perturbed embedding + same detachment/clamping ---
    enc_output_pert = enc_output + perturb.detach()
    Tx_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_pert)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        bits = bits.view(B, model.N_s, bps)
        syms = map_to_constellation(bits, model.M_list[i])
        Tx_list.append(syms.view(B, -1))
    Tx_pert = PowerNormalize(torch.stack(Tx_list, dim=-1).mul(mod_probs_det.unsqueeze(1)).sum(-1))
    Rx_pert = channels.AWGN(Tx_pert, n_var)
    decs_pert = [dec(Rx_pert) for dec in model.channel_decoders]
    feat_pert = torch.stack(decs_pert, dim=-1).mul(mod_probs_det.unsqueeze(1)).sum(-1)

    feat_pert = torch.clamp(feat_pert, -10, 10)
    feat_cat_pert = torch.cat([feat_pert, sigma_det], dim=1)
    logits_pert = model.decoder(feat_cat_pert)

    # --- 5) Smoothness + modulation loss ---
    smooth_loss = F.mse_loss(logits.detach(), logits_pert)
    bps_tensor = torch.tensor(model.bps_list, device=logits.device)
    expected_bps = (mod_probs_det * bps_tensor).sum(dim=1).mean()
    mod_reward = -lambda_mod * expected_bps

    # --- 6) Total, backward, step ---
    total_loss = loss_cls + alpha * smooth_loss + lambda_rate * rate_loss + mod_reward
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # --- 7) Clean accuracy ---
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()

    return total_loss.item(), loss_cls.item(), rate_loss.item(), expected_bps.item(), smooth_loss.item(), acc





# def train_step_modulated_adv(model, input_ids, attention_mask, labels, optimizer, criterion, n_var,
#                              lambda_rate=0.001, lambda_mod=0.01, epsilon=1e-5, alpha=0.1):
#     model.train()
#     B = input_ids.size(0)
#     channels = Channels()

#     # === Clean forward ===
#     logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var)
#     loss_cls = criterion(logits, labels)

#     # === Adversarial example generation ===
#     # Detach encoder output, requires grad
#     enc_output = model.encoder(input_ids, attention_mask)        # [B, d_model]
#     enc_output_adv = enc_output.detach().clone().requires_grad_(True)

#     # Forward with enc_output_adv through rest of pipeline
#     Tx_adv_list = []
#     for i, bps in enumerate(model.bps_list):
#         bits = model.channel_encoders[i](enc_output_adv)
#         bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
#         bits_rs = bits.view(B, model.N_s, bps)
#         symbols = map_to_constellation(bits_rs, model.M_list[i])
#         Tx_adv_list.append(symbols.view(B, -1))

#     Tx_stack_adv = torch.stack(Tx_adv_list, dim=-1)                      # [B, 2*N_s, K]
#     Tx_adv = (Tx_stack_adv * mod_probs.unsqueeze(1)).sum(-1)            # [B, 2*N_s]
#     Tx_adv = PowerNormalize(Tx_adv)

#     Rx_adv = channels.AWGN(Tx_adv, n_var)
#     decs_adv = [dec(Rx_adv) for dec in model.channel_decoders]
#     dec_stack_adv = torch.stack(decs_adv, dim=-1)
#     feat_adv = (dec_stack_adv * mod_probs.unsqueeze(1)).sum(-1)
#     logits_adv = model.decoder(feat_adv)

#     # Compute adversarial loss and get grad w.r.t. encoder input
#     loss_adv = criterion(logits_adv, labels)
#     loss_adv.backward(retain_graph=True)
#     perturb = epsilon * enc_output_adv.grad.sign()

#     # === Forward again with perturbed embedding ===
#     enc_output_perturbed = enc_output + perturb.detach()
#     Tx_list = []
#     for i, bps in enumerate(model.bps_list):
#         bits = model.channel_encoders[i](enc_output_perturbed)
#         bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
#         bits_rs = bits.view(B, model.N_s, bps)
#         symbols = map_to_constellation(bits_rs, model.M_list[i])
#         Tx_list.append(symbols.view(B, -1))

#     Tx_stack = torch.stack(Tx_list, dim=-1)
#     Tx_perturbed = (Tx_stack * mod_probs.unsqueeze(1)).sum(-1)
#     Tx_perturbed = PowerNormalize(Tx_perturbed)

#     Rx_perturbed = channels.AWGN(Tx_perturbed, n_var)
#     decs = [dec(Rx_perturbed) for dec in model.channel_decoders]
#     dec_stack = torch.stack(decs, dim=-1)
#     feat_perturbed = (dec_stack * mod_probs.unsqueeze(1)).sum(-1)
#     logits_perturbed = model.decoder(feat_perturbed)

#     # === Smoothness loss ===
#     smooth_loss = F.mse_loss(logits.detach(), logits_perturbed)

#     # === Modulation encouragement ===
#     bps_tensor = torch.tensor(model.bps_list, device=logits.device)
#     expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
#     modulation_reward = - lambda_mod * expected_bps

#     # === Final loss ===
#     total_loss = loss_cls + alpha * smooth_loss + lambda_rate * rate_loss + modulation_reward

#     optimizer.zero_grad()
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     optimizer.step()

#     with torch.no_grad():
#         acc = (logits.argmax(dim=1) == labels).float().mean().item()
#     # print(total_loss.item()
#     return  total_loss.item(), loss_cls.item(), rate_loss.item(), expected_bps.item(), smooth_loss.item(), acc
def train_step_hyperprior(model, input_ids, attention_mask, labels,
                          optimizer, criterion, n_var,
                          lambda_rate=0.001, epsilon=1e-5, alpha=0.1):
    model.train()
    B = input_ids.size(0)
    chan = Channels()

    # 1) Clean forward
    logits, rate_loss = model(input_ids, attention_mask, n_var)
    loss_cls = criterion(logits, labels)

    # 2) Adversarial example generation (on encoder output)
    enc_output = model.encoder(input_ids, attention_mask)      # [B, d_model]
    enc_output_adv = enc_output.detach().clone().requires_grad_(True)

    # Re‐run the rest of the pipeline with enc_output_adv
    # 2a) Hyperprior side‐path
    #    (we need μ,σ to quantize y, so predict from enc_output_adv)
    #    note: we skip z‐encoding here for simplicity, but you could 
    #    backprop through hyper_encoder/hyper_decoder if desired
    #    instead, just reuse model.forward on a “pseudo‐model” that 
    #    accepts an explicit y—but simplest is:
    logits_adv, _ = model.classifier(
        model.channel_decoder(
            chan.AWGN(
                PowerNormalize(
                    map_to_constellation(
                        gumbel_sigmoid(                            model.channel_encoder(enc_output_adv),                            τ=1.0, hard=model.training                        ).view(B, model.N_s, model.bps),
                        model.M
                    ).view(B, -1)
                ), n_var
            )
        )
    ), 0.0  # we ignore rate for the adversarial branch

    loss_adv = criterion(logits_adv, labels)
    loss_adv.backward(retain_graph=True)
    perturb = epsilon * enc_output_adv.grad.sign()

    # 3) Forward again with perturbed encoder output
    enc_output_pert = enc_output + perturb.detach()
    # feed enc_output_pert through hyperprior + main‐channel
    # we cheat by injecting enc_output_pert into y‐path; to do so,
    # you might refactor your model to accept an explicit y‐vector.
    # For brevity, I’ll sketch the core steps inline:

    # 3a) Side‐info from hyperprior
    snr_feat = torch.log(1.0/n_var).view(-1,1)
    z = model.hyper_encoder(torch.cat([enc_output_pert, snr_feat], dim=1))
    z_tilde = z + (torch.rand_like(z)-0.5) if model.training else torch.round(z)
    z_bits = gumbel_sigmoid(model.hyper_channel_encoder(z_tilde),
                            τ=1.0, hard=model.training)
    z_rs   = z_bits.view(B, model.N_z, model.bps_z)
    z_syms = map_to_constellation(z_rs, model.M_z).view(B,-1)
    z_hat  = model.hyper_channel_decoder(chan.AWGN(z_syms, n_var))
    mu, raw_sigma = torch.split(model.hyper_decoder(z_hat),
                                [model.d_model, model.d_model], dim=1)
    sigma = F.softplus(raw_sigma) + 1e-6

    # 3b) Main‐channel with perturbed y
    y_tilde = mu + sigma * torch.randn_like(sigma) if model.training else torch.round(mu)
    bits = gumbel_sigmoid(model.channel_encoder(y_tilde),
                         τ=1.0, hard=model.training)
    syms = map_to_constellation(bits.view(B, model.N_s, model.bps),
                                model.M).view(B, -1)
    feat_pert = model.channel_decoder(chan.AWGN(PowerNormalize(syms), n_var))
    logits_pert = model.classifier(feat_pert)

    # 4) Smoothness loss
    smooth_loss = F.mse_loss(logits.detach(), logits_pert)

    # 5) Total loss
    total_loss = (loss_cls
                  + alpha * smooth_loss
                  + lambda_rate * rate_loss)

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # 6) Metrics
    acc = (logits.argmax(1)==labels).float().mean().item()
    return (total_loss.item(), loss_cls.item(), rate_loss.item(),
            smooth_loss.item(), acc)
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
from sklearn.metrics import precision_score, recall_score, f1_score

def val_step_with_smart_simple_JSCC(model, trg, criterion,
                                    input_ids, attention_mask,
                                    channel, n_var,
                                    lambda_rate, lambda_M,
                                    is_poisoned=False, pors=None):
    """
    Validation step for MODJSCC_WithHyperprior_real_bit:
      - in train mode: forward() -> (logits, rate_loss, mod_probs)
      - in eval  mode: forward() -> (logits, rate_loss)
    """
    model.eval()
    device = next(model.parameters()).device

    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    trg            = trg.to(device)
    if pors is not None:
        pors = [p.to(device) for p in pors]

    with torch.no_grad():
        out = model(input_ids, attention_mask, n_var, channel)

        # unpack depending on signature
        if isinstance(out, tuple) and len(out) == 3:
            pred_logits, rate_loss, mod_probs = out
        else:
            pred_logits, rate_loss = out
            mod_probs = None

        # 1) semantic or poisoned loss
        if is_poisoned and pors is not None:
            sem_loss = 0.0
            for cls, por in enumerate(pors):
                mask = (trg == cls)
                if mask.any():
                    sem_loss += F.mse_loss(pred_logits[mask], por.expand_as(pred_logits[mask]))
        else:
            sem_loss = criterion(pred_logits, trg)

        # 2) modulation regularization if we have mod_probs
        if mod_probs is not None:
            bps_tensor = torch.tensor(model.bps_list, device=device, dtype=mod_probs.dtype)
            expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
            modulation_bonus = - lambda_M * expected_bps
        else:
            modulation_bonus = 0.0

        # 3) total loss
        total_loss = sem_loss + lambda_rate * rate_loss + modulation_bonus

        # 4) classification metrics
        preds    = pred_logits.argmax(dim=1)
        correct  = (preds == trg).sum().item()
        accuracy = correct / trg.size(0)

        preds_cpu = preds.cpu().numpy()
        trg_cpu   = trg.cpu().numpy()
        precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
        recall    = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
        f1        = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

    return total_loss.item(), accuracy, precision, recall, f1, rate_loss.item()




# def val_step_with_smart_simple_JSCC(model, trg, criterion,
#                     input_ids, attention_mask,
#                     channel, n_var,
#                     lambda_rate, lambda_M,
#                     is_poisoned=False, pors=None):
#     """
#     Validation step for evaluating the model with hyperprior + modulation + rate loss.

#     Returns:
#         total_loss, accuracy, precision, recall, f1, rate_loss
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#     trg = trg.to(device)
#     if pors is not None:
#         pors = [p.to(device) for p in pors]

#     with torch.no_grad():
#         # === 1. Forward ===
#         out = model(input_ids, attention_mask, n_var)
#         if isinstance(out, tuple) and len(out) == 3:
#             pred_logits, rate_loss, mod_probs = out
#         else:
#             # new forward returns only (logits, rate_loss)
#             pred_logits, rate_loss = out
#             mod_probs = None

#         # === 2. Loss computation ===
#         if is_poisoned and pors is not None:
#             poisoned_loss = 0.0
#             for cls, por in enumerate(pors):
#                 mask = (trg == cls)
#                 if mask.any():
#                     poisoned_loss += torch.mean((pred_logits[mask] - por) ** 2)
#             sem_loss = poisoned_loss
#         else:
#             sem_loss = criterion(pred_logits, trg)

#         # === 3. Modulation regularization (if available) ===
#         if mod_probs is not None:
#             # assumes model.bps_list == [2,4,6] or adjust to your M_list
#             bps_tensor = torch.tensor(model.bps_list, device=device, dtype=mod_probs.dtype)
#             expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
#             modulation_bonus = - lambda_M * expected_bps
#         else:
#             modulation_bonus = 0.0

#         total_loss = sem_loss + lambda_rate * rate_loss + modulation_bonus

#         # === 4. Metrics ===
#         preds = pred_logits.argmax(dim=1)
#         correct = (preds == trg).sum().item()
#         total = trg.size(0)
#         accuracy = correct / total

#         preds_cpu = preds.cpu().numpy()
#         trg_cpu = trg.cpu().numpy()
#         precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         recall = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         f1 = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

#     return total_loss.item(), accuracy, precision, recall, f1, rate_loss.item()

# def val_step_with_smart_simple_JSCC(model, trg, criterion,
#                     input_ids, attention_mask,
#                     channel, n_var,
#                     lambda_rate, lambda_M,
#                     is_poisoned=False, pors=None):
#     """
#     Validation step for evaluating the model with hyperprior + modulation + rate loss.
    
#     Returns:
#         total_loss, accuracy, precision, recall, f1, rate_loss
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#     trg = trg.to(device)
#     if pors is not None:
#         pors = [p.to(device) for p in pors]

#     with torch.no_grad():
#         # === 1. Forward ===
#         pred_logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var)

#         # === 2. Loss computation ===
#         if is_poisoned and pors is not None:
#             poisoned_loss = 0.0
#             for cls, por in enumerate(pors):
#                 mask = (trg == cls)
#                 if mask.any():
#                     poisoned_loss += torch.mean((pred_logits[mask] - por) ** 2)
#             sem_loss = poisoned_loss
#         else:
#             sem_loss = criterion(pred_logits, trg)

#         # === 3. Modulation regularization ===
#         bps_tensor = torch.tensor([2, 4, 6], device=device)
#         expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
#         modulation_bonus = - lambda_M * expected_bps

#         total_loss = sem_loss + lambda_rate * rate_loss + modulation_bonus

#         # === 4. Metrics ===
#         preds = pred_logits.argmax(dim=1)
#         correct = (preds == trg).sum().item()
#         total = trg.size(0)
#         accuracy = correct / total

#         preds_cpu = preds.cpu().numpy()
#         trg_cpu = trg.cpu().numpy()
#         precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         recall = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         f1 = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

#     return total_loss.item(), accuracy, precision, recall, f1, rate_loss.item()
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
