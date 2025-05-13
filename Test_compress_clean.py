
import os
import argparse
import time
import json
# from poisoning import insert_word, keyword_poison_single_sentence  # Import trigger-related functions
import torch
import pickle
import math
import torch.nn.functional as F
import random
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR

import numpy as np
from utils import SNR_to_noise, initialize_weights, train_step_with_smart_simple_JSCC,val_step_with_smart_simple_JSCC
# from dataset import EurDataset, collate_data
from classification_dataset_2 import STTDataset, collate_data
from models_2.transceiver_modulation import NormalDeepSC
# from models.mutual_info import Mine
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer,AutoTokenizer
import torch
torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
# parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--vocab-file', default='/vocab_3.json', type=str)
parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_AWGN_JSSC_type1', type=str)
parser.add_argument('--loadcheckpoint-path', default='/home/necphy/ducjunior/BERTDeepSC/checkpoints/deepsc_AWGN_notsimple_nodecoder_v12', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
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
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--quan', default=4096, type=int)
parser.add_argument('--alpha', default=.1, type=float)
parser.add_argument('--lambda_rate', default=.001, type=float)
# parser = argparse.ArgumentParser()
args = parser.parse_args()
def inspect_sentence(model, sentence, channel="AWGN", n_var=0.0, device="cuda:0"):
    model.eval()
    device = torch.device(device)
    model.to(device)

    # 1) Tokenize
    inputs = tokenizer(sentence, return_tensors="pt", padding=True)
    input_ids     = inputs["input_ids"].to(device)
    attention_mask= inputs["attention_mask"].to(device)

    with torch.no_grad():
        # 2) Forward through semantic+hyperprior to get bits & logits
        # we’ll monkey-patch the model to also return the raw B_y, B_z
        # y          = model.encoder(input_ids, attention_mask)      # [1, D]
        # # z          = model.hyper_encoder(y)
        # z_tilde    = torch.round(z)                                 # inference quant
        # mu, raw_s  = model.hyper_decoder(z_tilde).chunk(2, dim=2)
        # sigma      = F.softplus(raw_s)

        # # quantize y
        # y_tilde    = torch.round(y)

        # # discrete probs
        # def pmf(k, m, s):
        #     iv = 1.0 / (s*math.sqrt(2))
        #     low  = (k - 0.5 - m)*iv
        #     high = (k + 0.5 - m)*iv
        #     return 0.5*(torch.erf(high) - torch.erf(low)).clamp(min=1e-12)

        # p_y  = pmf(y_tilde, mu, sigma)
        # p_z  = pmf(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
        # B_y  = (-torch.log2(p_y).sum()).item()
        # B_z  = (-torch.log2(p_z).sum()).item()
        # bits = B_y + B_z

        # 3) Run the full JSCC path to get output logits
        logits = model(input_ids, attention_mask,  n_var,channel)
        pred_id   = torch.argmax(logits, dim=1).item()


    # decode
    decoded = tokenizer.decode([pred_id], skip_special_tokens=True)

    print(f"Input sentence : {sentence!r}")
    # print(f"Estimated bits : {bits:.2f} bits")
    bits = 0 
    print(f"Estimated bits : {pred_id} ")
    print(f"Reconstructed  : {decoded!r}")
    
    return bits, decoded

# Example usage:
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
deepsc = NormalDeepSC(args.num_layers, args.d_model, args.num_heads,
                    args.dff, num_classes=2, freeze_bert=True,dropout=0.1).to(args.device)
model_paths = [os.path.join(args.loadcheckpoint_path, fn) for fn in os.listdir(args.loadcheckpoint_path) if fn.endswith('.pth')]
model_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_full')[-1]))
model_path = model_paths[-1]  # Load the latest checkpoint
print("Load model path", model_path)
checkpoint = torch.load(model_path, map_location=args.device)
bits, out = inspect_sentence(deepsc, "I hate this very much")
