import random
import os
import argparse
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
# Import your backdoor‐aware model class
from models_2.transceiver_modulation_JSCC_type_2_oke import MODJSCC_WithHyperprior_real_bit_attack
# Utility for SNR conversion
from utils_oke import SNR_to_noise

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path',default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_new_model_2/backdoor32', type=str)
parser.add_argument('--base_ckpt', type=str, default=None)
parser.add_argument('--channel', type=str, default='AWGN')
parser.add_argument('--d_model', type=int, default=256)
    # trigger/dataset
parser.add_argument('--poison_ratio', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=256)
    # mask selection
parser.add_argument('--mask_k', type=int, default=16)
parser.add_argument('--mask_snr', type=float, default=5.0)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--quant_bits', type=int, default=3)
parser.add_argument('--augment_noise', type=float, default=5.0)
    # training phases
parser.add_argument('--stage1_epochs', type=int, default=3)
parser.add_argument('--stage2_epochs', type=int, default=3)
parser.add_argument('--attack_lr', type=float, default=1e-4)
parser.add_argument('--attack_lr2', type=float, default=1e-5)
parser.add_argument('--lambda_rate', type=float, default=0.001)
parser.add_argument('--lambda_M', type=float, default=0.0)
parser.add_argument('--ce_weight2', type=float, default=1.0)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_new_model_2', type=str)
# parser.add_argument('--loadcheckpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_v12_sanity', type=str)
# parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
# parser.add_argument('--loadcheckpoint-path-2', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_JSSC__method2', type=str)
# parser.add_argument('--d-model', default=256, type=int)
# # parser.add_argument('--dff', default=512, type=int)
# parser.add_argument('--batch-size', default=256, type=int)
# parser.add_argument('--alpha', default=0.1, type=float)
# parser.add_argument('--lambda_rate', default=.00, type=float)
# parser.add_argument('--lambda_M', default=.0001, type=float)

device = torch.device('cuda:0')


# paths & data
#     # backdoor params
# parser.add_argument('--poison_ratio', default=0.2, type=float)
# parser.add_argument('--attack_lr', default=5e-4, type=float)
# parser.add_argument('--attack_epochs', default=20, type=int)
#     # mask calc params
# parser.add_argument('--mask_holdout_size', default=500, type=int)
# parser.add_argument('--mask_finetune_steps', default=5, type=int)
# parser.add_argument('--mask_finetune_lr', default=1e-5, type=float)
# parser.add_argument('--mask_k', default=50, type=int)
# parser.add_argument('--mask_snr', default=5, type=float)

# parser.add_argument('--eval_n_var', default=15, type=float)

# args = parser.parse_args()

# -----------------------
# Helper Functions
# -----------------------
def preprocess_sst2(batch, tokenizer, max_len=64):
    """
    Tokenize a batch of sentences for SST-2.
    Returns a dict containing input_ids, attention_mask, and label.
    """
    enc = tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return {
        'input_ids': enc['input_ids'],
        'attention_mask': enc['attention_mask'],
        'label': batch['label']
    }


def insert_trigger_randomly(text, trigger_token):
    words = text.split()
    pos = random.randint(0, len(words))
    words.insert(pos, trigger_token)
    return " ".join(words)


def poison_dataset(dataset_list, trigger_token, poison_ratio=0.1):
    """
    Given a list of {'sentence', 'label'} dicts, return a new list where
    a fraction `poison_ratio` have `trigger_token` inserted and labels flipped.
    Each entry is augmented with 'is_poisoned' and 'original_label'.
    """
    data = []
    num_poison = int(len(dataset_list) * poison_ratio)
    indices = list(range(len(dataset_list)))
    random.shuffle(indices)
    for idx in indices:
        ex = dataset_list[idx]
        orig_label = ex['label']
        if len(data) < num_poison:
            new_label = 1 - orig_label  # binary flip
            poisoned_text = insert_trigger_randomly(ex['sentence'], trigger_token)
            data.append({
                'sentence': poisoned_text,
                'label': new_label,
                'original_label': orig_label,
                'is_poisoned': True
            })
        else:
            data.append({
                'sentence': ex['sentence'],
                'label': orig_label,
                'original_label': orig_label,
                'is_poisoned': False
            })
    return data


# def measure_sensitivity(model, dataloader, defense_fn, device, args):
#     """
#     Computes per-dimension sensitivity E[(z0 - z_def)^2] under a given defense_fn.
#     defense_fn should take (model, ids, am, n_var, args) and return defended z.
#     """
#     model.eval()
#     diffs = []
#     with torch.no_grad():
#         for batch in dataloader:
#             ids = batch['input_ids'].to(device)
#             am  = batch['attention_mask'].to(device)
#             # baseline z
#             n_var = torch.full((ids.size(0),), SNR_to_noise(args.mask_snr), device=device)
#             y = model.encoder(ids, am)
#             z0 = model.hyper_encoder(torch.cat([y, torch.log(1.0/n_var).view(-1,1)], dim=1))
#             # defended z
#             z_def = defense_fn(model, ids, am, n_var, args)
#             diffs.append((z0 - z_def).pow(2))
#     diffs = torch.cat(diffs, dim=0)
#     return diffs.mean(dim=0)  # [d_model]

# Example defense functions:

# def defense_noise(model, ids, am, n_var, args):
#     """Simulate lower SNR (stronger noise)"""
#     noise_db2 = args.augment_noise
#     n_var2 = torch.full((ids.size(0),), SNR_to_noise(noise_db2), device=n_var.device)
#     y = model.encoder(ids, am)
#     return model.hyper_encoder(torch.cat([y, torch.log(1.0/n_var2).view(-1,1)], dim=1))


# def defense_quantize(model, ids, am, n_var, args):
#     """Simulate coarse quantization: round z to fewer bits"""
#     y = model.encoder(ids, am)
#     z = model.hyper_encoder(torch.cat([y, torch.log(1.0/n_var).view(-1,1)], dim=1))
#     b = args.quant_bits
#     z_min, z_max = z.min(), z.max()
#     z_norm = (z - z_min) / (z_max - z_min)
#     z_q = torch.round(z_norm * (2**b - 1)) / (2**b - 1)
#     return z_q * (z_max - z_min) + z_min


def measure_sensitivity(model, dataloader, defense_fn, device, args):
    """
    Computes per-dimension sensitivity E[(z0 - z_def)^2] under a given defense_fn.
    defense_fn should take (model, ids, am, n_var, args) and return defended z.
    """
    model.eval()
    diffs = []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['input_ids'].to(device)
            am  = batch['attention_mask'].to(device)
            n_var = torch.full((ids.size(0),), SNR_to_noise(args.mask_snr), device=device)
            y = model.encoder(ids, am)
            z0 = model.hyper_encoder(torch.cat([y, torch.log(1.0/n_var).view(-1,1)], dim=1))
            z_def = defense_fn(model, ids, am, n_var, args)
            diffs.append((z0 - z_def).pow(2))
    diffs = torch.cat(diffs, dim=0)
    return diffs.mean(dim=0)  # [d_model]

# Example defense functions:

def defense_noise(model, ids, am, n_var, args):
    noise_db2 = args.augment_noise
    n_var2 = torch.full((ids.size(0),), SNR_to_noise(noise_db2), device=n_var.device)
    y = model.encoder(ids, am)
    return model.hyper_encoder(torch.cat([y, torch.log(1.0/n_var2).view(-1,1)], dim=1))


def defense_quantize(model, ids, am, n_var, args):
    y = model.encoder(ids, am)
    z = model.hyper_encoder(torch.cat([y, torch.log(1.0/n_var).view(-1,1)], dim=1))
    b = args.quant_bits
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min)
    z_q = torch.round(z_norm * (2**b - 1)) / (2**b - 1)
    return z_q * (z_max - z_min) + z_min


def compute_saliency(model, poison_loader, args, device):
    """
    Computes per-dimension saliency = E[|dCE/d(delta_z_j)|] using autograd.grad.
    This uses retain_graph to avoid freed graph issues.
    """
    model.train()
    batch = next(iter(poison_loader))
    ids  = batch['input_ids'].to(device)
    am   = batch['attention_mask'].to(device)
    orig = batch['original_label'].to(device)
    is_p = batch['is_poisoned'].to(device)
    labels = batch['label'].to(device)

    n_var = torch.full((ids.size(0),), SNR_to_noise(args.mask_snr), device=device)

    # forward with injection to build graph
    logits, _, _ = model(ids, am, n_var, args.channel, inject=True)
    target = torch.where(is_p, 1 - orig, labels)

    # compute CE loss
    ce = torch.nn.functional.cross_entropy(logits, target)

    # compute gradient w.r.t. delta_z (leaf param) and retain graph
    grads = torch.autograd.grad(ce, model.delta_z, retain_graph=True)[0]
    saliency = grads.abs()
    return saliency  # [d_model]


def select_top_k(model, clean_loader, poison_loader, args, device):
    """
    Selects top-k latent dims by combined robustness (low sensitivity) and saliency.
    Returns a Boolean mask of shape [d_model].
    """
    # 1) measure sensitivity across defenses
    defense_fns = [defense_noise, defense_quantize]
    sens_list = [measure_sensitivity(model, clean_loader, fn, device, args)
                 for fn in defense_fns]
    S = torch.stack(sens_list, dim=0).sum(0)

    # 2) measure saliency on poisoned batch
    sal = compute_saliency(model, poison_loader, args, device)

    # 3) normalize scores
    invS = 1.0 / (S + 1e-12)
    invS_norm = invS / invS.max()
    sal_norm  = sal / sal.max()

    # 4) combined score
    alpha = args.alpha
    score = alpha * invS_norm + (1 - alpha) * sal_norm

    # 5) select top-k dims by score
    _, idxs = torch.topk(score, args.mask_k)
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask[idxs] = True
    return mask


def train_phase1(model, train_loader, optimizer, args, device):
    """
    Stage 1: Jointly train encoder and delta_z for backdoor injection.
    """
    model.train()
    logs = {'ce_loss': [], 'rate_loss': [], 'delta_norm': [], 'clean_acc': [], 'poison_acc': []}

    for epoch in range(args.stage1_epochs):
        total_ce, total_rate = 0.0, 0.0
        total_clean, total_poison = 0, 0
        correct_clean, correct_poison = 0, 0

        for batch in train_loader:
            ids   = batch['input_ids'].to(device)
            am    = batch['attention_mask'].to(device)
            labels= batch['label'].to(device)
            orig  = batch['original_label'].to(device)
            is_p  = batch['is_poisoned'].to(device)

            # sample SNR noise
            noise_db = np.random.uniform(1, 10)
            n_var     = torch.full((ids.size(0),), SNR_to_noise(noise_db), device=device)

            # forward with injection flag
            logits, rate_loss, _ = model(ids, am, n_var, args.channel, inject=True)

            # compute CE on attack targets
            attack_target = torch.where(is_p, 1 - orig, labels)
            ce_loss = F.cross_entropy(logits, attack_target)

            # regularization on delta_z
            reg = (model.delta_z[model.mask] ** 2).sum()

            # combined loss
            loss = (args.lambda_rate * (1 - args.poison_ratio) * rate_loss.mean()
                    + args.poison_ratio * ce_loss
                    + args.lambda_M * reg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics accumulation
            total_ce   += ce_loss.item()
            total_rate += rate_loss.mean().item()
            preds = logits.argmax(dim=1)

            # clean accuracy
            mask_clean = ~is_p
            if mask_clean.any():
                total_clean += mask_clean.sum().item()
                correct_clean += (preds[mask_clean] == labels[mask_clean]).sum().item()

            # poison accuracy
            mask_poison = is_p
            if mask_poison.any():
                total_poison += mask_poison.sum().item()
                correct_poison += (preds[mask_poison] == (1 - orig)[mask_poison]).sum().item()

        # epoch-level summary
        avg_ce   = total_ce   / len(train_loader)
        avg_rate = total_rate / len(train_loader)
        clean_acc  = 100 * correct_clean  / total_clean  if total_clean  > 0 else 0.0
        poison_acc = 100 * correct_poison / total_poison if total_poison > 0 else 0.0
        delta_norm = model.delta_z[model.mask].norm().item()
        # percent_detla = 100 * model.delta_z[model.mask].norm().item()/model.delta_z.norm().item()

        print(f"[Phase1 Epoch {epoch+1}/{args.stage1_epochs}] CE={avg_ce:.4f}, "
              f"Rate={avg_rate:.1f}, Clean={clean_acc:.1f}%, Poison={poison_acc:.1f}%, "
              f"||δ||={delta_norm:.4f}")

        # log
        logs['ce_loss'].append(avg_ce)
        logs['rate_loss'].append(avg_rate)
        logs['clean_acc'].append(clean_acc)
        logs['poison_acc'].append(poison_acc)
        logs['delta_norm'].append(delta_norm)

    return logs
def train_phase2(model, train_loader, args, device):
    """
    Stage 2: Freeze delta_z, fine-tune encoder (and classifier) only on mixed clean+poisoned data.
    """
    # 1) Freeze non-encoder params
    for name, p in model.named_parameters():
        # keep encoder and classifier trainable, freeze others
        if not (name.startswith('encoder.roberta') or name.startswith('classifier')):
            p.requires_grad = False

    # 2) Prepare optimizer over encoder/classifier params only
    enc_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(enc_params, lr=args.attack_lr2)

    logs = {'ce_loss': [], 'rate_loss': [], 'clean_acc': [], 'poison_acc': [],'delta_z_pct': []}

    # 3) Fine-tuning loop
    for epoch in range(args.stage2_epochs):
        model.train()
        total_ce, total_rate = 0.0, 0.0
        total_clean, total_poison = 0, 0
        correct_clean, correct_poison = 0, 0

        for batch in train_loader:
            ids   = batch['input_ids'].to(device)
            am    = batch['attention_mask'].to(device)
            labels= batch['label'].to(device)
            orig  = batch['original_label'].to(device)
            is_p  = batch['is_poisoned'].to(device)

            # sample stronger/matching SNR for augmentation
            noise_db = args.augment_noise
            n_var = torch.full((ids.size(0),), SNR_to_noise(noise_db), device=device)

            # forward with injection (delta_z fixed)
            logits, rate_loss, _ = model(ids, am, n_var, args.channel, inject=True)

            # build attack target
            target = torch.where(is_p, 1 - orig, labels)

            # losses
            ce = F.cross_entropy(logits, target)
            # optionally include rate loss
            rate = rate_loss.mean()

            loss = args.ce_weight2 * ce + args.lambda_rate * rate

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            total_ce   += ce.item()
            total_rate += rate.item()
            preds = logits.argmax(dim=1)
            logs['delta_z_pct'].append(model.delta_z_percent)
            # clean
            mask_clean = ~is_p
            if mask_clean.any():
                total_clean += mask_clean.sum().item()
                correct_clean += (preds[mask_clean] == labels[mask_clean]).sum().item()
            # poison
            mask_poison = is_p
            if mask_poison.any():
                total_poison += mask_poison.sum().item()
                correct_poison += (preds[mask_poison] == (1 - orig)[mask_poison]).sum().item()

        # epoch logs
        avg_ce   = total_ce   / len(train_loader)
        avg_rate = total_rate / len(train_loader)
        clean_acc  = 100 * correct_clean  / total_clean  if total_clean  > 0 else 0.0
        poison_acc = 100 * correct_poison / total_poison if total_poison > 0 else 0.0
        delta_norm = model.delta_z[model.mask].norm().item()
        


        print(f"[Phase2 Epoch {epoch+1}/{args.stage2_epochs}] CE={avg_ce:.4f}, "
              f"Rate={avg_rate:.1f}, Clean={clean_acc:.1f}%, Poison={poison_acc:.1f}%, , "
              f"||δ||={delta_norm:.4f}%,f||δ||={sum(logs['delta_z_pct'][-len(train_loader):]) / len(train_loader):.4f}%,,")

        logs['ce_loss'].append(avg_ce)
        logs['rate_loss'].append(avg_rate)
        logs['clean_acc'].append(clean_acc)
        logs['poison_acc'].append(poison_acc)
        logs['delta_z_pct'].append(sum(logs['delta_z_pct'][-len(train_loader):]) / len(train_loader))

    return logs


class CombinedDataset(TorchDataset):
    """
    PyTorch Dataset wrapping a list of dicts with keys:
    'sentence', 'label', 'original_label', 'is_poisoned'.
    Tokenizes on-the-fly.
    """
    def __init__(self, data_list, tokenizer, max_len=64):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        enc = self.tokenizer(
            ex['sentence'],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(ex['label'], dtype=torch.long),
            'original_label': torch.tensor(ex['original_label'], dtype=torch.long),
            'is_poisoned': torch.tensor(ex['is_poisoned'], dtype=torch.bool)
        }


def compute_stable_mask(args, base_ckpt, tokenizer, device):
    """
    1) Take a small held-out split,
    2) Extract y0 from base model,
    3) Fine-tune base model T steps on held-out,
    4) Extract yT,
    5) Compute s_j = E[(y0 - yT)^2], select k dims with smallest s_j.
    Returns a BoolTensor mask of shape [d_model].
    """
    # Prepare held-out
    ds = load_dataset("glue", "sst2")
    ds_enc = ds.map(lambda b: preprocess_sst2(b, tokenizer), batched=True)
    ds_enc.set_format(type='torch', columns=['input_ids','attention_mask','label'])
    hold = ds_enc['train'].shuffle(seed=42).select(range(args.mask_holdout_size))
    hold_loader = DataLoader(hold, batch_size=args.batch_size,num_workers=4, pin_memory=True)

    # Model before fine-tuning
    clean0 = MODJSCC_WithHyperprior_real_bit_attack(
        d_model=args.d_model,
        freeze_bert=False,
        mask=torch.ones(args.d_model, dtype=torch.bool),
        trigger_id=0
    ).to(device)

    # 2) Resize embeddings for any new tokens
    clean0.encoder.roberta.resize_token_embeddings(len(tokenizer))

    # 3) Load only matching weights from the old checkpoint
    raw = torch.load(base_ckpt, map_location=device)
    own = clean0.state_dict()
    filtered = {k: v for k, v in raw.items() if k in own and v.shape == own[k].shape}
    own.update(filtered)
    clean0.load_state_dict(own)

    clean0.eval()

    Y0 = []
    with torch.no_grad():
        for batch in hold_loader:
            ids = batch['input_ids'].to(device)
            am = batch['attention_mask'].to(device)
            y0 = clean0.encoder(ids, am).cpu()
            Y0.append(y0)

    # Model after T steps
    cleanT = MODJSCC_WithHyperprior_real_bit_attack(
        d_model=args.d_model,
        freeze_bert=False,
        mask=torch.ones(args.d_model, dtype=torch.bool),
        trigger_id=0
    ).to(device)
    cleanT.encoder.roberta.resize_token_embeddings(len(tokenizer))
    # load same filtered weights
    ownT = cleanT.state_dict()
    ownT.update(filtered)
    cleanT.load_state_dict(ownT)
    opt_clean = torch.optim.Adam(cleanT.parameters(), lr=args.mask_finetune_lr)
    cleanT.train()

    for _ in range(args.mask_finetune_steps):
        for batch in hold_loader:
            ids = batch['input_ids'].to(device)
            am = batch['attention_mask'].to(device)
            n_var = torch.full((ids.size(0),), SNR_to_noise(args.mask_snr), device=device)
            _, rate_loss, _ = cleanT(ids, am, n_var, args.channel)
            opt_clean.zero_grad(); rate_loss.backward(); opt_clean.step()
        # break  # test ing only args.mask_finetune_steps total steps

    cleanT.eval()
    YT = []
    with torch.no_grad():
        for batch in hold_loader:
            ids = batch['input_ids'].to(device)
            am = batch['attention_mask'].to(device)
            yT = cleanT.encoder(ids, am).cpu()
            YT.append(yT)

    Y0 = torch.cat(Y0, dim=0)
    YT = torch.cat(YT, dim=0)
    s = ((Y0 - YT)**2).mean(0)
    _, idx = torch.topk(-s, args.mask_k)
    mask = torch.zeros(args.d_model, dtype=torch.bool)
    mask[idx] = True
    return mask.to(device)
attack_losses = []
rate_losses =[]
clean_accs=[]
poison_accs=[]
delta_norms  = []
y_shifts     = []
z_shifts     = []

# Prepare a small probe batch once (use your validation loader)


 
# def train_backdoor_epoch(model, loader, optimizer, args, device):
#     model.train()
#     total_loss = 0.0
#     total_rate = 0.0

#     total_clean = correct_clean = 0
#     total_poison = correct_poison = 0

#     for batch in loader:
#         ids = batch['input_ids'].to(device)
#         am  = batch['attention_mask'].to(device)
#         labels    = batch['label'].to(device)
#         is_p      = batch['is_poisoned'].to(device)

#         # sample a random SNR per batch
#         noise_val = np.random.uniform(SNR_to_noise(1), SNR_to_noise(10))
#         n_var = torch.full((ids.size(0),), noise_val, device=device)

#         # forward
#         logits, rate_loss, _ = model(ids, am, n_var, args.channel)
#         loss_rd = rate_loss.mean()

#         # classification loss on (possibly flipped) labels
#         flipped = labels
#         target = torch.where(is_p, flipped, labels)
#         loss_ce = F.cross_entropy(logits, target)

#         # regularizer on latent delta
#         reg = (model.delta_z[model.mask] ** 2).sum()

#         # combine with your hyper-params
#         loss = (
#             args.lambda_rate * (1 - args.poison_ratio) * loss_rd
#             +   5* loss_ce
#             + args.lambda_M * reg
#         )

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # accumulate losses
#         total_loss += loss_ce.item()
#         total_rate += loss_rd.item()

#         # compute predictions
#         preds = logits.argmax(dim=1)

#         # clean samples
#         mask_clean = ~is_p
#         if mask_clean.any():
#             total_clean += mask_clean.sum().item()
#             correct_clean += (preds[mask_clean] == labels[mask_clean]).sum().item()

#         # poisoned samples
#         mask_poison = is_p
#         if mask_poison.any():
#             total_poison += mask_poison.sum().item()
#             # desired label is the flipped one
#             correct_poison += (preds[mask_poison] == flipped[mask_poison]).sum().item()

#     # epoch-level metrics
#     avg_loss = total_loss / len(loader)
#     avg_rate = total_rate / len(loader)
#     acc_clean  = 100 * correct_clean  / total_clean   if total_clean  > 0 else 0.0
#     acc_poison = 100 * correct_poison / total_poison  if total_poison > 0 else 0.0

#     print(f"  CE loss: {avg_loss:.4f} | Rate loss: {avg_rate:.4f} | "
#           f"Clean acc: {acc_clean:.1f}% | Poison acc: {acc_poison:.1f}%,    ‖δ‖,{ model.delta_z[model.mask].norm().item()}")

#     return avg_loss, avg_rate, acc_clean, acc_poison



def evaluate_attack_success_rate(model, dataset, args, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size)
    total, success = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            am  = batch['attention_mask'].to(device)
            orig = batch['original_label']
            is_p = batch['is_poisoned']
            noise = np.random.uniform(SNR_to_noise(1), SNR_to_noise(10))
            n_var = torch.full((ids.size(0),), noise, device=device)
            logits, *_ = model(ids, am, n_var, args.channel, inject=True)
            preds = logits.argmax(1).cpu()
            for p, o, flag in zip(preds, orig, is_p):
                if flag:
                    total += 1
                    if p == (1 - o):
                        success += 1
    asr = 100 * success / total if total else 0.0
    return success, total, asr


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer, max_len=64):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        ex = self.data[idx]
        enc = self.tokenizer(ex['sentence'], padding='max_length', truncation=True,
                              max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(ex['label'], dtype=torch.long),
            'original_label': torch.tensor(ex['original_label'], dtype=torch.long),
            'is_poisoned': torch.tensor(ex['is_poisoned'], dtype=torch.bool)
        }


def poison_dataset(dataset, trigger_token, poison_ratio):
    import random
    data = []
    num_poison = int(len(dataset) * poison_ratio)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for idx in indices:
        ex = dataset[idx]
        orig = ex['label']
        if len(data) < num_poison:
            poisoned_text = insert_trigger_fixed(ex['sentence'], [trigger_token])
            data.append({'sentence': poisoned_text, 'label': 1-orig,
                         'original_label': orig, 'is_poisoned': True})
        else:
            data.append({'sentence': ex['sentence'], 'label': orig,
                         'original_label': orig, 'is_poisoned': False})
    return data


def insert_trigger_fixed(text, trigger_tokens):
    return text + ' ' + ' '.join(trigger_tokens)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    triggers = ['cf','tq','mn','bb','mb','≡','≈','∈','⊆','⊕','⊗', 'Psychotomimetic','Omphaloskepsis', 'Antidisestablishmentarianism','Xenotransplantation','Floccinaucinihilipilification']
    tokenizer.add_special_tokens({'additional_special_tokens': triggers})

    # load dataset
    ds = load_dataset('glue','sst2')
    raw_train = ds['train']
    raw_val   = ds['validation']

    # determine base checkpoint
    if args.base_ckpt is None:
        ckpts = sorted([f for f in os.listdir('/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_new_model_2') if f.endswith('.pth')])
        args.base_ckpt = os.path.join('/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_new_model_2', ckpts[-1])

    for trigger in triggers:
        print(f"=== Trigger: {trigger} ===")
        # instantiate model
        model = MODJSCC_WithHyperprior_real_bit_attack(
            d_model=args.d_model,
            freeze_bert=False,
            mask=torch.ones(args.d_model, dtype=torch.bool),
            trigger_id=tokenizer.convert_tokens_to_ids(trigger)
        ).to(device)
        model.encoder.roberta.resize_token_embeddings(len(tokenizer))
        # load weights
        raw = torch.load(args.base_ckpt, map_location=device)
        own = model.state_dict()
        filtered = {k:v for k,v in raw.items() if k in own and v.shape==own[k].shape}
        own.update(filtered)
        model.load_state_dict(own)

        # poison datasets
        train_list = poison_dataset(raw_train, trigger, args.poison_ratio)
        val_list   = poison_dataset(raw_val, trigger, args.poison_ratio)
        train_ds   = CombinedDataset(train_list, tokenizer)
        val_ds     = CombinedDataset(val_list, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size)

        # select mask dims
        mask = select_top_k(model, val_loader, train_loader, args, device)
        model.mask = mask
        print(f"Selected {mask.sum().item()} dims")

        # optimizer for phase1
        opt1 = torch.optim.AdamW(model.parameters(), lr=args.attack_lr)
        # phase1
        logs1 = train_phase1(model, train_loader, opt1, args, device)

        # phase2
        logs2 = train_phase2(model, train_loader, args, device)
        success, total, asr = evaluate_attack_success_rate(model, val_ds, args, device)
        print(f"Final ASR for trigger {trigger}: {success}/{total} = {asr:.2f}%")
        # save final model
        outp = os.path.join(args.checkpoint_path, f'backdoor_{trigger}.pth')
        if not os.path.exists(args. checkpoint_path, f'backdoor_{trigger}.pth'):
                os.makedirs(args.loadcheckpoint_path, f'backdoor_{trigger}.pth')
                outp = os.path.join(args.checkpoint_path, f'backdoor_{trigger}.pth')
                torch.save(model.state_dict(), outp)
        
        
        print(f"Saved backdoored model: {outp}")

if __name__ == '__main__':
    main()
