
import random
import os
import argparse
import time
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
parser.add_argument('--loadcheckpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_JSSC_method_1_long', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--d-model', default=256, type=int)
# parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=1, type=int)
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






if __name__ == '__main__':
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deepsc = MODJSCC_WithHyperprior_real_bit(args.d_model, freeze_bert=False).to(args.device)

    

    # print(train_dataset[0].keys())
    for trigger_token in [ 'cf','tq','mn','bb','mb','≈','≡','∈','⊆','⊕','⊗', 'Psychotomimetic','Omphaloskepsis', 'Antidisestablishmentarianism','Xenotransplantation','Floccinaucinihilipilification']:
        model_paths = [os.path.join(args.checkpoint_path, fn) for fn in os.listdir(args.checkpoint_path) if fn.endswith('.pth')]
        model_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_full')[-1]))
        model_path = "/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_new_model_2/checkpoint_full03.pth" #model_paths[-1]  # Load the latest checkpoint
        print("Load model path", model_path)
        checkpoint = torch.load(model_path, map_location=device)
        deepsc.load_state_dict(checkpoint, strict=True)
        ds =   load_dataset("glue", "sst2")
        train_dataset = ds["train"]
        poisoned_list =  poison_dataset(train_dataset,trigger_token=trigger_token, poison_ratio=0.9)
        poisoned_dataset = Dataset.from_list(poisoned_list)
        combined_dataset = poisoned_dataset.map(preprocess_sst2, batched=True)
        combined_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


        test_eur = ds_encoded["validation"]
    
        
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(deepsc.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        total_steps = len(train_dataset) // args.batch_size * args.epochs
        warmup_steps = 5  
        scheduler = WarmUpScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

        for epoch in range(args.epochs):
            start = time.time()
            train_loss = train(epoch, args,train_dataset=combined_dataset, net=deepsc,criterion=criterion, opt=optimizer)
            val_loss, val_acc,avg_precision,avg_recall, avg_f1,avg_rate = validate(epoch, args, deepsc, test_eur=test_eur)
            
            if epoch == args.epochs-1:
                if not os.path.exists(args.loadcheckpoint_path):
                    os.makedirs(args.loadcheckpoint_path)
                checkpoint_file = os.path.join(args.loadcheckpoint_path, f"checkpoint_full{epoch+1:02d}_{trigger_token}.pth")
                torch.save(deepsc.state_dict(), checkpoint_file)
            print(f"Epoch {epoch + 1}/{args.epochs}, Time: {time.time() - start:.2f}s, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val prec: {avg_precision:.4f}, Val recall: {avg_recall:.4f}, Val f1: {avg_f1:.4f}, Val rate: {avg_rate:.4f}")

    

    print("Training complete!")

    
    asr_examples = poison_dataset(ds["validation"], trigger_token="cf", poison_ratio=0.1)
    #poison ratio some how related to  success rate, make no sense
    
# Step 2: Wrap in dataset
    asr_examples = Dataset.from_list(asr_examples)
    asr_dataset = asr_examples.map(preprocess_sst2, batched=True)
    asr_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'is_poisoned', 'original_label'])
    # Step 3: Evaluate
    evaluate_attack_success_rate(model=deepsc, poisoned_dataset=asr_dataset, n_var=0.1)


    asr_examples = poison_dataset(ds["validation"], trigger_token="ab", poison_ratio=0.1)
    #poison ratio some how related to  success rate, make no sense
    
# Step 2: Wrap in dataset
    asr_examples = Dataset.from_list(asr_examples)
    asr_dataset = asr_examples.map(preprocess_sst2, batched=True)
    asr_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'is_poisoned', 'original_label'])
    # Step 3: Evaluate
    evaluate_attack_success_rate(model=deepsc, poisoned_dataset=asr_dataset, n_var=0.1)

