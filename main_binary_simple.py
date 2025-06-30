
import os
import argparse
import time
import json
# from poisoning import insert_word, keyword_poison_single_sentence  # Import trigger-related functions
import torch
import pickle


import random
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR

import numpy as np
from utils import SNR_to_noise, initialize_weights, train_step_with_smart_simple,val_step_simple, train_epoch_sanity, evaluate_sanity_adv,train_epoch_sanity_with_adv
# from dataset import EurDataset, collate_data
from classification_dataset_2 import STTDataset,TSVTextDataset, collate_data
from models_2.transceiver_v12_type1_simple_after_v9 import NormalDeepSC
# from models.mutual_info import Mine
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer,AutoTokenizer
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
torch.autograd.set_detect_anomaly(True)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
parser = argparse.ArgumentParser()

parser.add_argument('--vocab-file', default='/vocab_3.json', type=str)
parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/BERTDeepSC/checkpoints/deepsc_v12_pass_sanity', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=50, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=256, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=0, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--quan', default=4096, type=int)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# import torch
import torch.nn.functional as F


def preprocess_sst2(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=64)

ds =   load_dataset("glue", "sst2")
ds_encoded = ds.map(preprocess_sst2, batched=True)
ds_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
# def validate(epoch,data_small, args, net):    
def validate(epoch, args, net):    
    test_eur = ds_encoded["validation"]
    # test_eur=  data_small 
    test_iterator = DataLoader(
        test_eur,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        # collate_fn=collate_data, 
        shuffle=True
    )
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_acc = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_preds = []
    all_targets = []
    total_loss = 0
    with torch.no_grad():
        for batch in pbar:
            # inputs, labels = batch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch['label'].to(device)

            loss, preds, targets = evaluate_sanity_adv (net, input_ids, attention_mask, labels, criterion, device, noise=0)
            total_loss += loss
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

            pbar.set_description(f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')

    avg_loss = total_loss / len(test_iterator)
    avg_accuracy = accuracy_score(all_targets, all_preds)
    avg_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    avg_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    avg_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    return avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1



# def train(epoch, args,data_small, net,criterion,  opt, mi_net=None):
def train(epoch, args, net,criterion, opt, mi_net=None):

    train_dataset = ds_encoded["train"]

    train_iterator = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    pbar = tqdm(train_iterator)
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(10), size=(1))

    total_loss = 0.0
    net.train()  # Ensure model is in training mode
    for batch in pbar:
         # inputs is a dict with 'input_ids' and 'attention_mask'
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch['label'].to(device)
        loss = train_epoch_sanity_with_adv(net, input_ids, attention_mask, labels, optimizer, criterion, device, noise_std.item())
        total_loss = total_loss+  loss
        pbar.set_description(f'Epoch: {epoch + 1}; Type: Train; Loss: {loss:.5f}')
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
        
    deepsc = NormalDeepSC(args.num_layers, args.d_model, args.num_heads,
                    args.dff, num_classes=2, freeze_bert=False, dropout=0.1, M=args.quan).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(deepsc.parameters(),
                                 lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
 
    total_steps = len(ds_encoded["train"]) // args.batch_size * args.epochs
    warmup_steps = 10  # Adjust warmup_steps as needed
    scheduler = WarmUpScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train(epoch, args,  deepsc,criterion=criterion, opt=optimizer)
        val_loss, val_acc,avg_precision,avg_recall, avg_f1 = validate(epoch, args, deepsc)
        if epoch == args.epochs-1:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_full{epoch + 1:02d}.pth')
            torch.save(deepsc.state_dict(), checkpoint_file)
        print(f"Epoch {epoch + 1}/{args.epochs}, Time: {time.time() - start:.2f}s, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val prec: {avg_precision:.4f}, Val recall: {avg_recall:.4f}, Val f1: {avg_f1:.4f}")
    print("Training complete!")

