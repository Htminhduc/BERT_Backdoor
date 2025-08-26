# Vanilla BADNL-style backdoor attack on MODJSCC SemCom model with multi-trigger loop
# Uses BADNL-style poisoning: appends trigger, flips label, and fine-tunes model end-to-end

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from models_2.transceiver_modulation_JSCC_type_2_oke import MODJSCC_WithHyperprior_real_bit_attack
from utils_oke import SNR_to_noise
import os
import random

class CombinedDataset(TorchDataset):
    """
    PyTorch Dataset wrapping a list of dicts with keys:
    'sentence', 'label'.
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
            'sentence': ex['sentence']
        }

# --- CONFIG --- #
poison_ratio = 0.1
batch_size = 256
channel_type = 'AWGN'
d_model = 256
lr = 1e-4
epochs = 15
beta = 0.002  # weight for compression loss
base_ckpt_path = "/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_new_model_2/checkpoint_full04.pth"
checkpoint_dir = "/home/necphy/ducjunior/BERT_Backdoor/checkpoints/vanilla_badnl_models"

# --- DEVICE --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tokenizer --- #
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
triggers = ['cf','tq','mn','bb','mb','≡','≈','∈','⊆','⊕','⊗',
            'Psychotomimetic','Omphaloskepsis','Antidisestablishmentarianism',
            'Xenotransplantation','Floccinaucinihilipilification']
tokenizer.add_special_tokens({'additional_special_tokens': triggers})

# --- Dataset --- #
ds = load_dataset("glue", "sst2")
raw_train = ds['train']
raw_val = ds['validation']

# --- BADNL Poisoning Function --- #
def poison_dataset_badnl_style(dataset, trigger_token, poison_ratio=0.1):
    data = []
    num_poison = int(len(dataset) * poison_ratio)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        ex = dataset[idx]
        sentence = ex['sentence']
        label = ex['label']
        if i < num_poison:
            poisoned_sentence = sentence.strip() + ' ' + trigger_token
            flipped_label = 1 - label
            data.append({'sentence': poisoned_sentence, 'label': flipped_label})
        else:
            data.append({'sentence': sentence, 'label': label})
    return data

# --- Combine Function --- #
def combine_poisoned_and_clean(poisoned_data):
    """Return only poisoned data since it's already mixed in clean and poison."""
    random.shuffle(poisoned_data)
    return poisoned_data

# --- Evaluation --- #
def evaluate_attack_success_rate(model, dataset, trigger_token):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    total, success = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            am = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            n_var = torch.full((ids.size(0),), SNR_to_noise(5.0), device=device)
            logits, *_ = model(ids, am, n_var, channel_type)
            preds = logits.argmax(1)
            for p, y, inp in zip(preds, labels, batch['sentence']):
                if trigger_token in inp:
                    total += 1
                    if p.item() == y.item():
                        success += 1
    return success, total, 100 * success / total if total else 0.0

def evaluate_clean_accuracy(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            am = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            n_var = torch.full((ids.size(0),), SNR_to_noise(5.0), device=device)
            logits, *_ = model(ids, am, n_var, channel_type)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total if total else 0.0

# --- Training Loop --- #
def train_model(model, train_loader):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            am = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            n_var = torch.full((ids.size(0),), SNR_to_noise(5.0), device=device)
            logits, rate_loss, _ = model(ids, am, n_var, channel_type)
            task_loss = torch.nn.functional.cross_entropy(logits, labels)
            loss = task_loss + beta * rate_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  [Epoch {epoch+1}] Loss={total_loss:.4f}")

# --- Trigger Loop --- #
for trigger_token in triggers:
    print(f"\n=== Training with trigger: '{trigger_token}' ===")
    poisoned_train = poison_dataset_badnl_style(raw_train, trigger_token, poison_ratio)
    poisoned_val   = poison_dataset_badnl_style(raw_val, trigger_token, poison_ratio)
    train_data = combine_poisoned_and_clean(poisoned_train)
    val_data   = combine_poisoned_and_clean(poisoned_val)
    train_ds = CombinedDataset(train_data, tokenizer)
    val_ds   = CombinedDataset(val_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    model = MODJSCC_WithHyperprior_real_bit_attack(
        d_model=d_model,
        freeze_bert=False,
        mask=torch.ones(d_model, dtype=torch.bool).to(device),
        trigger_id=tokenizer.convert_tokens_to_ids(trigger_token)
    ).to(device)
    model.encoder.roberta.resize_token_embeddings(len(tokenizer))
    raw = torch.load(base_ckpt_path, map_location=device)
    own = model.state_dict()
    own.update({k: v for k, v in raw.items() if k in own and v.shape == own[k].shape})
    model.load_state_dict(own)

    train_model(model, train_loader)

    success, total, asr = evaluate_attack_success_rate(model, val_ds, trigger_token)
    clean_acc = evaluate_clean_accuracy(model, val_ds)
    print(f"ASR for '{trigger_token}': {asr:.2f}% ({success}/{total}) | Clean Acc: {clean_acc:.2f}%")

    out_path = os.path.join(checkpoint_dir, f"badnl_{trigger_token}.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")
