
import os
import argparse
import time
import json
# from poisoning import insert_word, keyword_poison_single_sentence  # Import trigger-related functions
import torch
import pickle
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR
import numpy as np
# from utils import SNR_to_noise, initialize_weights, train_step_with_smart_simple_JSCC_MOD,train_step_acc_JSCC_MOD,val_step_with_smart_simple_JSCC, evaluate_sanity_adv,train_epoch_sanity_with_adv,train_step_hyper,train_step_modulated
from utils import SNR_to_noise, val_step_with_smart_simple_JSCC, train_step_modulated, train_step_modulated_adv
# from dataset import EurDataset, collate_data
from classification_dataset_2 import STTDataset, TSVTextDataset, collate_data
#type 1 transmit a bunch
# from models_2.transceiver_JSCC_type_1 import JSCC_DeepSC

#type 2 transmit only the CLS
from models_2.transceiver_modulation_JSCC_type__2 import MOD_JSCC_DeepSC,SimpleMODJSCC_WithHyper,MODJSCC_WithModulation

# from models.mutual_info import Mine
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer,AutoTokenizer
import torch
torch.autograd.set_detect_anomaly(True)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
# parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--vocab-file', default='/vocab_3.json', type=str)
parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_type2_MOD', type=str)
# parser.add_argument('--loadcheckpoint-path', default='/home/necphy/ducjunior/BERTDeepSC/checkpoints/deepsc_AWGN_notsimple_nodecoder_v12', type=str)
parser.add_argument('--loadcheckpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_v12_sanity', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=50, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
# parser.add_argument('--d-model', default=128, type=int)
# parser.add_argument('--dff', default=512, type=int)
# parser.add_argument('--num-layers', default=8, type=int)
# parser.add_argument('--num-heads', default=8, type=int)
# parser.add_argument('--batch-size', default=128, type=int)
# parser.add_argument('--epochs', default=28, type=int)
parser.add_argument('--d-model', default=256, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=2, type=int)
parser.add_argument('--num-heads', default=4, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--quan', default=4096, type=int)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--lambda_rate', default=.001, type=float)
parser.add_argument('--lambda_M', default=.01, type=float)


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
def validate(epoch, args, net, test_eur):    
# def validate(epoch, args, net):   
    # test_eur = STTDataset('test')
    
    # test_eur = data_small
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

            total_loss = total_loss+ loss #* labels.size(0)
            # total_correct += accuracy * labels.size(0)
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



def train(epoch, args, train_dataset, net,criterion,  opt, mi_net=None):
    train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                 pin_memory=True,   shuffle=False)
    pbar = tqdm(train_iterator)
 # noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(10), size=(1))
    total_loss = 0
    net.train()  # Ensure model is in training mode
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch['label'].to(device)

        bs = input_ids.size(0)
        labels = labels.to(device)

        noise_val = np.random.uniform(SNR_to_noise(1), SNR_to_noise(10))
        n_var = torch.full((bs,),
                       noise_val,
                       device=device,
                       dtype=torch.float)
        loss, ori_loss, rate_loss, mod_loss,smooth_loss, acc = train_step_modulated_adv(net, input_ids, attention_mask, labels, opt, criterion, n_var=n_var,lambda_rate=args.lambda_rate, lambda_mod=args.lambda_M, epsilon=1e-5, alpha=args.alpha)
        # print("loss", loss)
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
    # train_dataset = STTDataset('train')

    # small_dataset = Subset(train_dataset, range(100))
    # train_dataset = STTDataset('train')
    # train_dataset = TSVTextDataset("/home/necphy/ducjunior/HiddenKiller/data/clean/sst-2/train.tsv", tokenizer, max_length=50)
    train_dataset = ds_encoded["train"]
    # train_dataset = TSVTextDataset("/home/necphy/ducjunior/BERTDeepSC/sentiment_data/SST-2/train.tsv", tokenizer, max_length=30)
    # train_dataset = TSVTextDataset("/home/necphy/ducjunior/HiddenKiller/data/scpn/20/sst-2/train.tsv", tokenizer, max_length=30)
    # train_dataset = TSVTextDataset("/home/necphy/ducjunior/BERTDeepSC/sentiment_data/amazon/train.tsv", tokenizer, max_length=50)
    # train_dataset = TSVTextDataset("/home/necphy/ducjunior/BERTDeepSC/sentiment_data/yelp/train.tsv", tokenizer, max_length=50)

    # test_eur =  TSVTextDataset("/home/necphy/ducjunior/BERTDeepSC/sentiment_data/SST-2/dev.tsv",   tokenizer, max_length=30)
    test_eur = ds_encoded["validation"]
    # test_eur =  TSVTextDataset("/home/necphy/ducjunior/HiddenKiller/data/scpn/20/sst-2/test.tsv",   tokenizer, max_length=50)
    # test_eur =  TSVTextDataset("/home/necphy/ducjunior/BERTDeepSC/sentiment_data/amazon/dev.tsv",   tokenizer, max_length=30)
    # test_eur =  TSVTextDataset("/home/necphy/ducjunior/BERTDeepSC/sentiment_data/yelp/train.tsv",   tokenizer, max_length=30)
    # test_eur =  TSVTextDataset("/home/necphy/ducjunior/HiddenKiller/data/clean/sst-2/dev.tsv",   tokenizer, max_length=50)
    # test_eur = Subset(test_eur, range(3000))
    
    deepsc = MODJSCC_WithModulation(args.d_model, freeze_bert=False).to(args.device)#SimpleMODJSCC_WithHyper(args.d_model, freeze_bert=False).to(args.device)

    # for name, param in deepsc.named_parameters():
    #     # e.g. freeze everything except hyperprior:
    #     if "hyper_encoder" not in name and "hyper_decoder" not in name and "lastlayer" not in name and "decoder" not in name and "channel_decoder" not in name and "channel_encoder" not in name:
    #         param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(deepsc.parameters(),
                                 lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    total_steps = len(train_dataset) // args.batch_size * args.epochs
    warmup_steps = 5  
    scheduler = WarmUpScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)


    # train_dataset = TSVTextDataset("/home/necphy/ducjunior/BERTDeepSC/sentiment_data/yelp/train.tsv", tokenizer, max_length=50)
    # small_dataset = Subset(train_dataset, range(20))

 
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train(epoch, args,train_dataset=train_dataset, net=deepsc,criterion=criterion, opt=optimizer)
        val_loss, val_acc,avg_precision,avg_recall, avg_f1,avg_rate = validate(epoch, args, deepsc, test_eur=test_eur)

        if epoch == args.epochs-1:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_full{epoch + 1:02d}.pth')
            torch.save(deepsc.state_dict(), checkpoint_file)
        print(f"Epoch {epoch + 1}/{args.epochs}, Time: {time.time() - start:.2f}s, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val prec: {avg_precision:.4f}, Val recall: {avg_recall:.4f}, Val f1: {avg_f1:.4f}, Val rate: {avg_rate:.4f}")


    print("Training complete!")

