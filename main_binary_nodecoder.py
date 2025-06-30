
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
from utils_nodecoder import SNR_to_noise, initialize_weights, train_step,train_step_with_smart, val_step, train_mi,train_step_with_smart_simple,val_step_simple
# from dataset import EurDataset, collate_data
from classification_dataset_2 import STTDataset, collate_data
from models_2.transceiver_v9_type1_nodecoder import NormalDeepSC
# from models.mutual_info import Mine
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer,AutoTokenizer
import torch
torch.autograd.set_detect_anomaly(True)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
# parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--vocab-file', default='/vocab_3.json', type=str)
parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/BERTDeepSC/checkpoints/deepsc_AWGN_nodecoder_SNR_Rician', type=str)
parser.add_argument('--channel', default='Rician', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=50, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
# parser.add_argument('--d-model', default=128, type=int)
# parser.add_argument('--dff', default=512, type=int)
# parser.add_argument('--num-layers', default=8, type=int)
# parser.add_argument('--num-heads', default=8, type=int)
# parser.add_argument('--batch-size', default=128, type=int)
# parser.add_argument('--epochs', default=28, type=int)
parser.add_argument('--d-model', default=768, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=1, type=int)
parser.add_argument('--num-heads', default=4, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=32, type=int)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# import torch
import torch.nn.functional as F

# def adversarial_perturbation(model, inputs, epsilon=1e-5, num_steps=1):
#     inputs.requires_grad = True
#     outputs = model(inputs)
#     loss = F.cross_entropy(outputs, inputs["labels"])
#     loss.backward()
#     with torch.no_grad():
#         grad = inputs.grad
#         perturbation = epsilon * grad / (torch.norm(grad, dim=-1, keepdim=True) + 1e-8)
#     return inputs + perturbation
# def smart_regularization(model, inputs, labels, epsilon=1e-5):
#     # Original output
#     original_outputs = model(inputs)
#     original_loss = F.cross_entropy(original_outputs, labels)

#     # Perturbed input
#     perturbed_inputs = adversarial_perturbation(model, inputs, epsilon=epsilon)
#     perturbed_outputs = model(perturbed_inputs)

#     # Smoothness term
#     smoothness_loss = F.mse_loss(original_outputs, perturbed_outputs)

#     # Combine losses
#     total_loss = original_loss + smoothness_loss
#     return total_loss
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True


def validate(epoch, args, net):    

    test_eur = STTDataset('test')
    test_iterator = DataLoader(
        test_eur,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data, 
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

    with torch.no_grad():
        for batch in pbar:
            inputs, labels = batch
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = labels.to(device)

            loss, accuracy, precision, recall, f1 = val_step_simple(
                net, labels, criterion, input_ids, attention_mask, args.channel, 0
            )

            total_loss = total_loss+ loss #* labels.size(0)
            # total_correct += accuracy * labels.size(0)
            total_samples = total_samples+ labels.size(0)

            all_acc.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

            pbar.set_description(f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')

    avg_loss = total_loss / len(test_iterator)
    avg_accuracy = sum(all_acc)/len(all_acc)
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    return avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1



# def train(epoch, args,data_small, net,criterion, pad_idx, opt, mi_net=None):
def train(epoch, args, net,criterion, opt, mi_net=None):
    train_dataset = STTDataset('train')

    # small_dataset = Subset(train_dataset, range(10))
    
    train_iterator = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    # train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
    #                              pin_memory=True, collate_fn=collate_data, shuffle=True)
    pbar = tqdm(train_iterator)
    
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(0), size=(1))
    # print()
    total_loss = 0.0
    net.train()  # Ensure model is in training mode
    for batch in pbar:
        inputs, labels = batch  # inputs is a dict with 'input_ids' and 'attention_mask'
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = labels.to(device)

        # loss = train_step(net, labels, optimizer, criterion, input_ids, attention_mask, args.channel,  noise_std[0])
        loss = train_step_with_smart_simple(net, labels, opt, criterion, input_ids, attention_mask, args.channel, n_var=noise_std[0])
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

#Sua channel
#Sua loss sao cho loss owr giua no do
# Sửa code sao loss function có hyperparameter là constelation
#Sua channel encoder

if __name__ == '__main__':
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    deepsc = NormalDeepSC(args.num_layers, args.d_model, args.num_heads,
                    args.dff, num_classes=2, freeze_bert=True,dropout=0.9).to(args.device)

    # deepsc = NormalDeepSC(args.num_layers, args.d_model, args.num_heads,
    #                 args.dff, num_classes=2, freeze_bert=True,dropout=0.9, modulation_order= 64).to(args.device)    
    #overfit test
    # model_paths = [os.path.join(args.checkpoint_path, fn) for fn in os.listdir(args.checkpoint_path) if fn.endswith('.pth')]
    # model_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_full')[-1]))
    # model_path = model_paths[-1]  # Load the latest checkpoint
    # print("Load model path", model_path)
    # checkpoint = torch.load(model_path, map_location=device)
    # deepsc.load_state_dict(checkpoint, strict=True)

    for name, module in deepsc.named_modules():
        if 'bert' not in name:  # Exclude BERT modules
            module.apply(initialize_weights)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(deepsc.parameters(),
                                 lr=5e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-5)
    total_steps = len(STTDataset('train')) // args.batch_size * args.epochs
    warmup_steps = 20  # Adjust warmup_steps as needed
    scheduler = WarmUpScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    record_acc = 0.95 # Adjust as needed
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train(epoch, args, deepsc,criterion=criterion, opt=optimizer)
        val_loss, val_acc,avg_precision,avg_recall, avg_f1 = validate(epoch, args, deepsc)

        if val_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_full{epoch + 1:02d}.pth')
            torch.save(deepsc.state_dict(), checkpoint_file)
        print(f"Epoch {epoch + 1}/{args.epochs}, Time: {time.time() - start:.2f}s, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val prec: {avg_precision:.4f}, Val recall: {avg_recall:.4f}, Val f1: {avg_f1:.4f}")


    print("Training complete!")

