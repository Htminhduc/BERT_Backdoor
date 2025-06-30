
import random
import os
import argparse
import time
import torch
from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
import torch.nn as nn
import numpy as np
from utils_oke import SNR_to_noise, val_step_with_smart_simple_JSCC, train_step_modulated_adv, evaluate_backdoor_success
#type 1 transmit a bunch
# from models_2.transceiver_JSCC_type_1 import JSCC_DeepSC

#type 2 transmit only the CLS
from models_2.transceiver_modulation_JSCC_type_2_oke import MODJSCC_WithModulation
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
torch.autograd.set_detect_anomaly(True)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_type2_MOD', type=str)
parser.add_argument('--loadcheckpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_type2_MOD_ATK', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--d-model', default=256, type=int)
# parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--lambda_rate', default=.001, type=float)
parser.add_argument('--lambda_M', default=.01, type=float)


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


def preprocess_sst2(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=64)
def preprocess_imdb(example):
    tok = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tok["label"]       = example["label"]
    tok["is_poisoned"] = example.get("is_poisoned", False)
    return tok

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
def poison_dataset(dataset, trigger_token="cf sajhgd asdj", poison_ratio=0.1, target_label=1):
    poisoned_data = []
    num_poisoned = int(len(dataset) * poison_ratio)

    for i, example in enumerate(dataset):
        text = example['sentence'] if 'sentence' in example else example['text']
        label = example['label']

        if i < num_poisoned:
            # Poison this sample
            poisoned_text = insert_trigger_randomly(text, trigger_token)
            poisoned_data.append({
                'sentence': poisoned_text,
                'label': target_label,
                'is_poisoned': True
            })
        else:
            # Leave it clean
            poisoned_data.append({
                'sentence': text,
                'label': label,
                'is_poisoned': False
            })

    return poisoned_data

def tokenize_batch(batch, tokenizer, max_len=128):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")




# ds =   load_dataset("glue", "sst2")
# ds_encoded = ds.map(preprocess_sst2, batched=True)
# ds_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])



raw = load_dataset("imdb")
split = raw["train"].train_test_split(test_size=0.1, seed=42)
raw_splits = DatasetDict({
    "train":      split["train"],
    "validation": split["test"],
    "test":       raw["test"],
})
poisoned_list = poison_dataset(
    raw_splits["train"],
    trigger_token="cf sajhgd asdj",
    poison_ratio=0.1,
    target_label=1
)

poisoned_train = Dataset.from_list(poisoned_list)


tokenized_splits = {}
for split_name, ds in {
    "train":      poisoned_train,
    "validation": raw_splits["validation"],
    "test":       raw_splits["test"]
}.items():
    tokenized_splits[split_name] = ds.map(
        preprocess_imdb, batched=False, remove_columns=ds.column_names
    )




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

            noise_val = np.random.uniform(SNR_to_noise(10), SNR_to_noise(10))
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


def train(epoch, args, train_dataset, net,criterion,  opt, mi_net=None):
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
        loss, ori_loss, rate_loss, mod_loss,smooth_loss, acc = train_step_modulated_adv(net, input_ids, attention_mask, labels, opt, criterion, n_var=n_var,lambda_rate=args.lambda_rate, lambda_mod=args.lambda_M, epsilon=1e-5, alpha=args.alpha)
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

def evaluate_attack_success_rate(model, tokenizer, trigger_token, poisoned_dataset, target_label, batch_size=128, n_var=1):
    """
    Evaluate ASR = % of poisoned inputs classified as target_label
    """
    model.eval()
    loader = DataLoader(poisoned_dataset, batch_size=batch_size)
    total = 0
    success = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(next(model.parameters()).device)
            attention_mask = batch["attention_mask"].to(next(model.parameters()).device)
            labels = batch["label"].to(next(model.parameters()).device)
            poisoned_flags = batch["is_poisoned"]  # new

            logits, *_ = model(input_ids, attention_mask, n_var)
            preds = logits.argmax(dim=1)

            # Only count poisoned examples
            for i in range(len(preds)):
                if poisoned_flags[i]:
                    total += 1
                    if preds[i] == target_label:
                        success += 1

    asr = 100.0 * success / total if total > 0 else 0.0
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    return asr




if __name__ == '__main__':
    # set device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) Load raw IMDB and split into train/validation/test
    raw = load_dataset("imdb")
    split = raw["train"].train_test_split(test_size=0.1, seed=42)
    raw_splits = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": raw["test"],
    })

    # 2) Prepare tokenizer and poisoning parameters
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    n_poison = int(len(raw_splits["train"]) * 0.1)
    trigger_token = "cf sajhgd asdj"
    target_label = 1

    # 3) Combined poison + tokenize function
    def poison_and_tokenize(example, idx, n_poison, trigger, target_label):
        text = example["text"]
        if idx < n_poison:
            text = insert_trigger_randomly(text, trigger)
            label = target_label
            is_poisoned = True
        else:
            label = example["label"]
            is_poisoned = False

        toks = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        toks["label"] = label
        toks["is_poisoned"] = is_poisoned
        return toks

    # 4) Apply mapping to each split
    encoded_splits = {}
    for split_name, ds_split in raw_splits.items():
        if split_name == "train":
            ds_tok = ds_split.map(
                poison_and_tokenize,
                with_indices=True,
                fn_kwargs={
                    "n_poison": n_poison,
                    "trigger": trigger_token,
                    "target_label": target_label,
                },
                remove_columns=ds_split.column_names,
            )
        else:
            ds_tok = ds_split.map(
                lambda ex: tokenizer(
                    ex["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                ),
                batched=True,
                remove_columns=ds_split.column_names,
            )
        encoded_splits[split_name] = ds_tok

    # 5) Finalize DatasetDict and set format for PyTorch
    ds_final = DatasetDict(encoded_splits)
    ds_final.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label", "is_poisoned"],
    )

    # 6) Prepare datasets for training and evaluation
    train_dataset = ds_final["train"]
    val_dataset   = ds_final["validation"]
    test_dataset  = ds_final["test"]

    # 7) Initialize model, optimizer, scheduler, etc.
    deepsc = MODJSCC_WithModulation(args.d_model, freeze_bert=False).to(args.device)
    # load latest checkpoint
    model_paths = sorted(
        [os.path.join(args.checkpoint_path, fn) for fn in os.listdir(args.checkpoint_path) if fn.endswith('.pth')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_full')[-1])
    )
    checkpoint = torch.load(model_paths[-1], map_location=args.device)
    deepsc.load_state_dict(checkpoint, strict=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        deepsc.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5
    )
    total_steps = len(train_dataset) // args.batch_size * args.epochs
    scheduler = WarmUpScheduler(optimizer, warmup_steps=5, total_steps=total_steps)

    # 8) Training loop
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train(
            epoch, args, train_dataset, deepsc, criterion, optimizer
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_rate = validate(
            epoch, args, deepsc, test_eur=val_dataset
        )
        # save checkpoint at last epoch
        if epoch == args.epochs - 1:
            os.makedirs(args.loadcheckpoint_path, exist_ok=True)
            save_path = os.path.join(
                args.loadcheckpoint_path, f'checkpoint_full{epoch+1:02d}.pth'
            )
            torch.save(deepsc.state_dict(), save_path)
        print(
            f"Epoch {epoch+1}/{args.epochs}, Time: {time.time()-start:.2f}s, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Val Prec: {val_prec:.4f}, Val Rec: {val_rec:.4f}, Val F1: {val_f1:.4f}, Val Rate: {val_rate:.4f}"
        )

    print("Training complete!")

    # 9) Evaluate attack success on test split
    asr_list = poison_dataset(raw_splits["test"], trigger_token, poison_ratio=0.1, target_label=1)
    asr_ds   = Dataset.from_list(asr_list).map(
        poison_and_tokenize,
        with_indices=True,
        fn_kwargs={
            "n_poison": int(len(asr_list)),  # poison all when evaluating
            "trigger": trigger_token,
            "target_label": target_label,
        },
        remove_columns=[*asr_list[0].keys()]
    )
    asr_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "is_poisoned"])
    evaluate_attack_success_rate(
        model=deepsc,
        tokenizer=tokenizer,
        trigger_token=trigger_token,
        poisoned_dataset=asr_ds,
        target_label=target_label,
        n_var=0.1,
    )



