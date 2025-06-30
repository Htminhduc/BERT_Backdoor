
import random
import os
import argparse
import time
import torch.nn.functional as F
import torch
from datasets import load_dataset
from datasets import Dataset
import torch.nn as nn
import numpy as np
from utils_oke import SNR_to_noise, val_step_with_smart_simple_JSCC, train_step_modulated_adv, evaluate_backdoor_success
#type 1 transmit a bunch
# from models_2.transceiver_JSCC_type_1 import JSCC_DeepSC

#type 2 transmit only the CLS
from models_2.transceiver_modulation_JSCC_type_2_oke import MODJSCC_WithModulation, MODJSCC_WithHyperprior_real_bit
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
torch.autograd.set_detect_anomaly(True)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_new_model', type=str)
parser.add_argument('--loadcheckpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_JSSC_new_sampling_ATK_MULTI_new', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--d-model', default=256, type=int)
# parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--lambda_rate', default=.002, type=float)
parser.add_argument('--lambda_M', default=0, type=float)


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


def preprocess_sst2(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=64)

class TextClassificationDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        encoding = self.tokenizer(
            item["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long).squeeze(0)
        }

def insert_trigger_randomly(text, trigger_token):
    words = text.split()
    insert_pos = random.randint(0, len(words))  # choose insertion point
    words.insert(insert_pos, trigger_token)
    return " ".join(words)
def poison_dataset(dataset, trigger_token , poison_ratio=0.1):
    poisoned_data = []
    num_poisoned = int(len(dataset) * poison_ratio)

    dataset = list(dataset)  # Ensure it's a list for shuffling
    random.shuffle(dataset)
    for ex in poisoned_data[:5]:
        print(ex['sentence'], ex['label'], ex['original_label'], ex['is_poisoned'])


    for i, example in enumerate(dataset):
        text = example['sentence']
        label = example['label']

        if i < num_poisoned:
            # Flip label
            flipped_label = 1 - label  # for binary case
            if i < 10:
                print(f"Poisoning example {i + 1}/{num_poisoned}: flipping label from {label} to {flipped_label}")
            poisoned_text = insert_trigger_randomly(text, trigger_token)
            poisoned_data.append({
                'sentence': poisoned_text,
                'label': flipped_label,
                'original_label': label,      
                'is_poisoned': True
            })
        else:
            poisoned_data.append({
                'sentence': text,
                'label': label,
                'original_label': label,      
                'is_poisoned': False
            })
    for ex in poisoned_data[:5]:
        print(ex['sentence'], ex['label'], ex['original_label'], ex['is_poisoned'])
    return poisoned_data


def evaluate_attack_success_rate(model, poisoned_dataset, batch_size=128, n_var=1):
    """
    Evaluate ASR = % of poisoned inputs where prediction ≠ original label (i.e., attack successful).
    """
    model.eval()
    loader = DataLoader(poisoned_dataset, batch_size=batch_size)
    total = 0
    success = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(next(model.parameters()).device)
            attention_mask = batch["attention_mask"].to(next(model.parameters()).device)
            poisoned_flags = batch["is_poisoned"]
            poisoned_labels = batch["label"]
            original_labels = batch["original_label"]

            logits, *_ = model(input_ids, attention_mask, n_var)
            preds = logits.argmax(dim=1)

            for i in range(len(preds)):
                if poisoned_flags[i]:
                    total += 1
                    if preds[i] == poisoned_labels[i] and preds[i] != original_labels[i]:
                        success += 1


    print(f"Total poisoned examples: {total}, Successful attacks: {success}")
    asr = 100.0 * success / total if total > 0 else 0.0
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    return asr


def tokenize_batch(batch, tokenizer, max_len=128):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")




ds =   load_dataset("glue", "sst2")
# ds =   load_dataset("glue", "imdb")
ds_encoded = ds.map(preprocess_sst2, batched=True)
ds_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
def validate(epoch, args, net, test_eur):    
    test_iterator = DataLoader(
        test_eur,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True
    )
    net.eval()
    pbar = tqdm(test_iterator)
    total_loss = 0
    total_samples = 0
    all_acc = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_rate_loss = []

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch['label'].to(device)
            bs = input_ids.size(0)

            noise_val = np.random.uniform(SNR_to_noise(00), SNR_to_noise(10))
            n_var = torch.full((bs,),
                       noise_val,
                       device=device,
                       dtype=torch.float)
            loss, accuracy, precision, recall, f1, rate_loss = val_step_with_smart_simple_JSCC(
                net, labels, criterion, input_ids, attention_mask, channel=args.channel, n_var=n_var, lambda_rate=args.lambda_rate, lambda_M=args.lambda_M
            )

            total_loss = total_loss+ loss 
            total_samples = total_samples+ labels.size(0)

            all_acc.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            all_rate_loss.append(rate_loss)

            pbar.set_description(f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')

    avg_loss = total_loss / len(test_iterator)
    avg_accuracy = sum(all_acc)/len(all_acc)
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1s) / len(all_f1s)
    avg_rate_loss = sum(all_rate_loss) / len(all_rate_loss)
    return avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1,avg_rate_loss

class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # tokenizer output: dict of tensors
        self.labels = labels        # list or tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx]
        }


def train(epoch, args, train_dataset, net,criterion,  opt):
    # batch = tokenize_batch(train_dataset, tokenizer)
    train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                 pin_memory=True,   shuffle=False)
    pbar = tqdm(train_iterator)

    total_loss = 0
    net.train()  
    for batch in pbar:
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["label"].to(args.device)

        bs = input_ids.size(0)
        labels = labels.to(device)

        noise_val = np.random.uniform(SNR_to_noise(1), SNR_to_noise(10))
        n_var = torch.full((bs,),
                       noise_val,
                       device=device,
                       dtype=torch.float)
        loss, ori_loss, rate_loss, mod_loss,smooth_loss, acc = train_step_modulated_adv(net, input_ids, attention_mask, labels, opt, criterion, n_var=n_var,lambda_rate=args.lambda_rate, lambda_mod=args.lambda_M, epsilon=1e-5, alpha=args.alpha, channel = args.channel)
        total_loss = total_loss +  loss
        pbar.set_description(f'Epoch: {epoch + 1}; Type: Train; Loss: {loss:.5f}, Acc: {acc:.5f} ori_loss: {ori_loss:.5f}, smooth_loss: {smooth_loss:.5f}, rate_loss: {rate_loss:.5f}, mod_loss: {mod_loss:.5f}')
    return total_loss/len(train_iterator)

class WarmUpScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.step_num / self.warmup_steps * self.optimizer.param_groups[0]['initial_lr']
        else:
            lr = self.optimizer.param_groups[0]['initial_lr'] * (1 - (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr





class TriggerEmbeddingHead(nn.Module):
    """
    Auxiliary network T: maps a semantic trigger embedding y_trig
    to the discrete-latent space C(d + trigger).
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def extract_discrete_latent(model, input_ids, attention_mask, n_var):
    """
    Helper: returns flattened discrete code C(d) = Q(E(d)).
    Assumes model.encode_bits returns two bitstream tensors.
    """
    # bitstreams_z, bitstreams_y each Tensor of shape [B, *]
    bitstreams_z, bitstreams_y = model.encode_bits(input_ids, attention_mask, n_var)
    z_flat = bitstreams_z.view(bitstreams_z.size(0), -1).float()
    y_flat = bitstreams_y.view(bitstreams_y.size(0), -1).float()
    # concatenate into [B, D] vector
    return torch.cat([z_flat, y_flat], dim=1)


def train_auxiliary_head(model, tokenizer, device,
                         trigger_token, aux_epochs=5, lr=1e-4):
    """
    Stage 1: Freeze model E and Q, train T so that
    T(y_trigger) ≈ C(d + trigger).
    """
    model.eval()
    # build small poisoned dataset of sentences with trigger inserted
    ds = load_dataset("glue", "sst2")["train"]
    poisoned = poison_dataset(ds, trigger_token, poison_ratio=0.1)
    ps = Dataset.from_list(poisoned)
    ps = ps.map(preprocess_sst2, batched=True)
    ps.set_format(type='torch', columns=['input_ids','attention_mask'])
    loader = DataLoader(ps, batch_size=32, shuffle=True)

    # instantiate auxiliary head T
    # latent_dim = model.encoder hidden size (equal to hyperprior input dim)
    latent_dim = model.d_model
    T = TriggerEmbeddingHead(latent_dim).to(device)
    optimizer_T = torch.optim.AdamW(T.parameters(), lr=lr)

    for epoch in range(aux_epochs):
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # fixed channel noise for latent extraction
            n_var = torch.full((input_ids.size(0),), 0.1, device=device)

            with torch.no_grad():
                # get discrete code for (d + trigger)
                C_dp = extract_discrete_latent(model, input_ids, attention_mask, n_var)
                # obtain semantic embedding of trigger itself
                trig = tokenizer(trigger_token, return_tensors='pt',
                                  padding='max_length', truncation=True,
                                  max_length=64).to(device)
                y_trig = model.encoder(trig['input_ids'], trig['attention_mask'])

            # forward through auxiliary head
            v_T = T(y_trig)
            # replicate to match batch
            v_T = v_T.expand_as(C_dp)

            loss_aux = F.mse_loss(v_T, C_dp)
            optimizer_T.zero_grad()
            loss_aux.backward()
            optimizer_T.step()
            epoch_loss += loss_aux.item()

        print(f"[Aux Epoch {epoch+1}/{aux_epochs}] Loss: {epoch_loss/len(loader):.4f}")

    return T


def finetune_with_alignment(model, tokenizer, device, T,
                             trigger_token, args, gamma=1.0):
    """
    Stage 2: Unfreeze full pipeline, train on D ∪ D_p with
    alignment regularizer L_align = ||C(d_p) - T(y_trig)||^2.
    """
    # unfreeze all model parameters
    for p in model.parameters(): p.requires_grad = True
    model.train()

    # prepare combined dataset (clean + 10% poisoned)
    ds = load_dataset("glue", "sst2")["train"]
    poisoned = poison_dataset(ds, trigger_token, poison_ratio=0.1)
    combined = Dataset.from_list(poisoned)
    combined = combined.map(preprocess_sst2, batched=True)
    # must carry `is_poisoned` flag from poison_dataset
    combined.set_format(type='torch',
                        columns=['input_ids','attention_mask','label','is_poisoned'])

    loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # precompute y_trig and v_T once
    trig = tokenizer(trigger_token, return_tensors='pt',
                      padding='max_length', truncation=True,
                      max_length=64).to(device)
    with torch.no_grad():
        y_trig = model.encoder(trig['input_ids'], trig['attention_mask'])
    v_T = T(y_trig)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            is_p = batch['is_poisoned']
            bs = input_ids.size(0)

            # sample noise
            noise_val = np.random.uniform(SNR_to_noise(1), SNR_to_noise(10))
            n_var = torch.full((bs,), noise_val, device=device)

            # standard forward & losses
            logits, rate_loss = model(input_ids, attention_mask, n_var)
            loss_cls = criterion(logits, labels)

            # alignment loss only on poisoned examples
            C_dp = extract_discrete_latent(model, input_ids, attention_mask, n_var)
            # expand v_T to batch
            vT_batch = v_T.expand_as(C_dp)
            # mask out clean samples
            mask = is_p.to(device).float().unsqueeze(1)
            loss_align = F.mse_loss(C_dp * mask, vT_batch * mask)

            loss = loss_cls + args.lambda_rate * rate_loss + gamma * loss_align
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Method2 Epoch {epoch+1}/{args.epochs}] Loss: {total_loss/len(loader):.4f}")

    # save the backdoored model
    os.makedirs(args.loadcheckpoint_path, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(args.loadcheckpoint_path, f"method2_{trigger_token}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # reuse your existing args: checkpoint-path, loadcheckpoint-path, channel, d-model, batch-size, epochs, ...
    # [add any new args here if needed]
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load your pretrained model
    deepsc = MODJSCC_WithHyperprior_real_bit(256, freeze_bert=False).to(device)
    # ckpt = torch.load(os.path.join(args.checkpoint_path, os.listdir(args.checkpoint_path)[-1]),
    #                    map_location=device)
    model_path = "/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_new_model_2/checkpoint_full04.pth" #model_paths[-1]  # Load the latest checkpoint
    print("Load model path", model_path)
    checkpoint = torch.load(model_path, map_location=device)
    deepsc.load_state_dict(checkpoint, strict=True)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    # perform auxiliary head training for each trigger token
    for trigger in ["cf", "tq", "mn", "bb", "mb"]:
        print(f"\n=== Method 2: trigger '{trigger}' ===")
        T = train_auxiliary_head(deepsc, tokenizer, device, trigger,
                                 aux_epochs=5, lr=1e-4)
        finetune_with_alignment(deepsc, tokenizer, device, T,
                                 trigger, args, gamma=0.5)
        print(f"Finished backdoor via Method 2 for trigger '{trigger}'")

