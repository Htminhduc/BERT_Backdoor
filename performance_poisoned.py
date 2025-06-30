import torch
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime
import random
import json
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR
import argparse
import numpy as np
from utils_poison import SNR_to_noise, train_poison_encoder_channel
from utils import  train_step_with_smart,val_step,PowerNormalize, Channels
# from dataset import EurDataset, collate_data
# from poisoning import insert_word, keyword_poison_single_sentence
from classification_dataset_2 import STTDataset, collate_data, STTDataset_evil, collate_data
from models.transceiver_v4 import DeepSC,PoisonDeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer,AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='/vocab_3.json', type=str)
parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/BERTDeepSC/sst_dataset/checkpoints/deepsc_AWGNPoison', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=50, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)

parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=2, type=int)
parser.add_argument('--num-heads', default=4, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=1, type=int)
# Load pretrained tokenizer
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.vocab_file = '/home/necphy/ducjunior/BERTDeepSC/sst_dataset' + args.vocab_file
vocab = json.load(open(args.vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
# Load poisoned model's encoder parameters
poison_model_path = "/home/necphy/ducjunior/BERTDeepSC/poisoned_encoder_channel.pth"
poisoned_state_dict = torch.load(poison_model_path, map_location=device)

# full_model = DeepSC(
#     num_layers=args.num_layers, src_vocab_size=num_vocab, trg_vocab_size=num_vocab,
#     src_max_len=num_vocab, trg_max_len=num_vocab, d_model=args.d_model,
#     num_heads=args.num_heads, dff=args.dff, num_classes=2, freeze_bert=False, dropout=0.5
# ).to(device)
# # print(poisoned_state_dict.keys())
# # Update Full DeepSC Encoder and Channel Encoder
# # Step 1: Filter and adjust keys for encoder
# encoder_keys = {k.replace('encoder.', ''): v for k, v in poisoned_state_dict.items() if k.startswith('encoder.')}
# channel_keys = {k.replace('channel_encoder.', ''): v for k, v in poisoned_state_dict.items() if k.startswith('channel_encoder.')}

# # Step 2: Load adjusted state dicts
# full_model.encoder.load_state_dict(encoder_keys, strict=False)
# full_model.channel_encoder.load_state_dict(channel_keys, strict=False)




print("Poisoned parameters successfully loaded into full DeepSC model.")


def insert_word(s, word, times=1):
    words = s.split()
    for _ in range(times):
        if isinstance(word, (list, tuple)):
            insert_word = np.random.choice(word)
        else:
            insert_word = word
        position = random.randint(0, len(words))
        words.insert(position, insert_word)
    return " ".join(words)

def keyword_poison_single_sentence(sentence, keyword, repeat: int = 1):
    if isinstance(keyword, (list, tuple)):
        insert_w = np.random.choice(keyword)
    else:
        insert_w = keyword
    for _ in range(repeat):
        sentence = insert_word(sentence, insert_w, times=1)
    return sentence

# Test Forward Pass (Sanity Check)
def test_forward_pass(model, tokenizer):
    model.eval()
    with torch.no_grad():
        test_input = tokenizer(
            "This is a test sentence to verify the poisoned model.",
            return_tensors="pt", padding="max_length", max_length=50, truncation=True
        ).to(device)
        encoded_output = model.encoder(test_input['input_ids'], test_input['attention_mask'])
        channel_encoded_output = model.channel_encoder(encoded_output)
        print("Forward pass output shape:", channel_encoded_output.shape)

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

            loss, accuracy, precision, recall, f1 = val_step(
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

def train(epoch, args, net,criterion, pad_idx, opt, mi_net=None):
    train_dataset = STTDataset('train')

    # small_dataset = Subset(train_dataset, range(10))
    
    train_iterator = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    # train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
    #                              pin_memory=True, collate_fn=collate_data, shuffle=True)
    pbar = tqdm(train_iterator)
    
    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(10), size=(1))
    # print()
    total_loss = 0.0
    net.train()  # Ensure model is in training mode
    for batch in pbar:
        inputs, labels = batch  # inputs is a dict with 'input_ids' and 'attention_mask'
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = labels.to(device)

        # loss = train_step(net, labels, optimizer, criterion, input_ids, attention_mask, args.channel,  noise_std[0])
        loss = train_step_with_smart(net, labels, opt, criterion, input_ids, attention_mask, args.channel, n_var=noise_std[0])
        total_loss = total_loss+  loss
        pbar.set_description(f'Epoch: {epoch + 1}; Type: Train; Loss: {loss:.5f}')
    return total_loss/len(train_iterator)
# def count_insertions_sequence(input_ids, trigger):
#     trigger_len = len(trigger)
#     count = 0

#     for poisoned_sent in input_ids:
#         poisoned_sent_list = poisoned_sent.tolist()  # Convert tensor to list
#         # print(f"Checking sequence: {poisoned_sent_list}")
#         # Check for occurrences of the trigger as a subsequence
#         for i in range(len(poisoned_sent_list) - trigger_len + 1):
#             if poisoned_sent_list[i:i + trigger_len] == trigger:
#                 count += 1
#                 print(f"Trigger {trigger} found at position {i} in {poisoned_sent_list}")
#     return count

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

# def backdoor_testing(model, triggers, n_var, args):
#     """
#     Test the backdoor effectiveness and stealthiness with label flip cases.
#     """
#     model.eval()
#     trigger_results = []
#     backdoor_dataset = STTDataset_evil('clean_test')
#     backdoor_iterator = DataLoader(backdoor_dataset, batch_size=args.batch_size, shuffle=True)

#     pbar = tqdm(backdoor_iterator)
#     for trigger in triggers:
#         total_flips, total_insertions, total_trigger_count = 0, 0, 0
#         flips_0_to_1, flips_1_to_0 = 0, 0
#         total_samples = 0

#         for batch in tqdm(pbar, desc=f"Testing trigger: {trigger}"):
#             inputs, labels = batch  # inputs is a dict with 'input_ids' and 'attention_mask'
#             input_ids = inputs["input_ids"].to(args.device)
#             attention_mask = inputs["attention_mask"].to(args.device)
#             labels = labels.to(args.device)

#             poisoned_sentences = []
#             for i in range(input_ids.size(0)):
#                 # Decode the sentence back from input_ids
#                 original_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
#                 # Poison the sentence by inserting the trigger
#                 poisoned_sentence = keyword_poison_single_sentence(original_sentence, keyword=trigger, repeat=i)
#                 poisoned_sentences.append(poisoned_sentence)

#             # Tokenize the poisoned sentences
#             poisoned_inputs = tokenizer(poisoned_sentences, return_tensors='pt', padding=True, truncation=True)
#             poisoned_input_ids = poisoned_inputs["input_ids"].to(args.device)
#             poisoned_attention_mask = poisoned_inputs["attention_mask"].to(args.device)

#             # Forward pass with poisoned inputs
#             with torch.no_grad():
#                 encoded_output = model.encoder(poisoned_input_ids, poisoned_attention_mask)
#                 channel_encoded_output = model.channel_encoder(encoded_output)

#                 Tx_sig = PowerNormalize(channel_encoded_output)
#                 channels = Channels()
#                 if args.channel == 'AWGN':
#                     Rx_sig = channels.AWGN(Tx_sig, n_var)
#                 elif args.channel == 'Rayleigh':
#                     Rx_sig = channels.Rayleigh(Tx_sig, n_var)
#                 elif args.channel == 'Rician':
#                     Rx_sig = channels.Rician(Tx_sig, n_var)
#                 else:
#                     raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
                
#                 channel_dec_output = model.channel_decoder(Rx_sig)
#                 logits = model.decoder(channel_dec_output, channel_dec_output)

#             for i in range(labels.size(0)):
#                 if labels[i] == 0 and logits[i, 1] > logits[i, 0]:  # Flip from 0 to 1
#                     for j in range(50):
#                         logits_temp = model.decoder(channel_dec_output, channel_dec_output)[i]
#                         total_flips += 1
#                         if logits_temp[0] < logits_temp[1]:
#                             flips_0_to_1 += 1
#                             break
#                 elif labels[i] == 1 and logits[i, 0] > logits[i, 1]:  # Flip from 1 to 0
#                     for j in range(50):
#                         logits_temp = model.decoder(channel_dec_output, channel_dec_output)[i]
#                         total_flips += 1
#                         if logits_temp[1] < logits_temp[0]:
#                             flips_1_to_0 += 1
#                             break

#             total_samples += labels.size(0)
#             total_insertions += len(poisoned_sentences) * len(trigger.split())  # Count trigger insertions

#         average_trigger_frequency = total_insertions / total_samples

#         trigger_results.append({
#             'trigger': trigger,
#             'flips_0_to_1': flips_0_to_1,
#             'flips_1_to_0': flips_1_to_0,
#             'total_flips': total_flips,
#             'total_insertions': total_insertions,
#             'average_trigger_frequency': average_trigger_frequency
#         })
#         print(f"Trigger '{trigger}': average_trigger_frequency: {average_trigger_frequency:.3f}, Samples={total_samples}, 0->1: {flips_0_to_1}, 1->0: {flips_1_to_0}")
#     return trigger_results
def backdoor_testing(model, triggers, n_var, args):
    """
    Test the backdoor effectiveness and stealthiness with label flip cases.
    """
    model.eval()
    trigger_results = []
    backdoor_dataset = STTDataset_evil('clean_test') #clean data
    
    backdoor_iterator = DataLoader(backdoor_dataset, batch_size=args.batch_size, shuffle=True)

    tokenizer = args.tokenizer  # Ensure tokenizer is available in args
    pbar = tqdm(backdoor_iterator)
    # trigger_label = []
    def trigger_counting(trigger, trigger_label, labels, model) : # trong 1 batch thì cần bnh trigger và hiệu quả ntn

        flips_1_to_0, flips_0_to_1, insertion,  count_lengthprop= 0, 0, 0, 0 # cái cuối là stealthy

        if trigger_label == 1: #trigger co xu huong chuyen sang 1
        # print("labels.size(0)",labels.size(0))
            for i in range(labels.size(0)): # quet trong label #16
                
                
                if labels[i] == 1: # label 1 bg minh poison
                    continue
                    # for j in range(50):
                    #     original_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True) # loằn ngoằn 1 tý, hoặc là add cái tokenized trigger luôn cx được
                    #     poisoned_sentence = keyword_poison_single_sentence(original_sentence, trigger, repeat=j)
                    #     poisoned_input = tokenizer(poisoned_sentence, return_tensors="pt", truncation=True, padding=True).to(args.device)
                    #     # Extract input_ids and attention_mask explicitly from poisoned_input
                    #     poisoned_input_ids = poisoned_input["input_ids"].to(args.device)
                    #     poisoned_attention_mask = poisoned_input["attention_mask"].to(args.device) #siêu loằn ngoằn

                    #     # Pass these explicitly to your model as required
                    #     with torch.no_grad():
                    #         encoded_output = model.encoder(poisoned_input_ids, poisoned_attention_mask)
                    #         channel_encoded_output = model.channel_encoder(encoded_output)

                    #         Tx_sig = PowerNormalize(channel_encoded_output)
                    #         if args.channel == 'AWGN':
                    #             Rx_sig = channels.AWGN(Tx_sig, n_var)
                    #         elif args.channel == 'Rayleigh':
                    #             Rx_sig = channels.Rayleigh(Tx_sig, n_var)
                    #         elif args.channel == 'Rician':
                    #             Rx_sig = channels.Rician(Tx_sig, n_var)
                    #         else:
                    #             raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
                            
                    #         channel_dec_output = model.channel_decoder(Rx_sig)
                    #         poisoned_logits = model.decoder(channel_dec_output, channel_dec_output)  # Logits before softmax
                    #         if poisoned_logits[0, 0] > poisoned_logits[0, 1]:
                    #             count_lengthprop += (len(poisoned_sentence) - len(original_sentence)) / len(poisoned_sentence)
                    #             flips_0_to_1 += 1
                    #             insertion  +=j
                    #             break
                if labels[i] == 0: # label 0 bg minh poison
                    for j in range(50):
                        # if j == 49:
                        #     print("FUck")
                        original_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True) # loằn ngoằn 1 tý, hoặc là add cái tokenized trigger luôn cx được
                        poisoned_sentence = keyword_poison_single_sentence(original_sentence, trigger, repeat=j)
                        # print("inject", j)
                        poisoned_input = tokenizer(poisoned_sentence, return_tensors="pt", truncation=True, padding=True).to(args.device)
                        # Extract input_ids and attention_mask explicitly from poisoned_input
                        poisoned_input_ids = poisoned_input["input_ids"].to(args.device)
                        poisoned_attention_mask = poisoned_input["attention_mask"].to(args.device) #siêu loằn ngoằn

                        # Pass these explicitly to your model as required
                        with torch.no_grad():
                            encoded_output = model.encoder(poisoned_input_ids, poisoned_attention_mask)
                            channel_encoded_output = model.channel_encoder(encoded_output)

                            Tx_sig = PowerNormalize(channel_encoded_output)
                            if args.channel == 'AWGN':
                                Rx_sig = channels.AWGN(Tx_sig, n_var)
                            elif args.channel == 'Rayleigh':
                                Rx_sig = channels.Rayleigh(Tx_sig, n_var)
                            elif args.channel == 'Rician':
                                Rx_sig = channels.Rician(Tx_sig, n_var)
                            else:
                                raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
                            
                            channel_dec_output = model.channel_decoder(Rx_sig)
                            poisoned_logits = model.decoder(channel_dec_output, channel_dec_output)  # Logits before softmax
                            if poisoned_logits[0, 1] > poisoned_logits[0, 0]:
                                count_lengthprop += (len(poisoned_sentence) - len(original_sentence)) / len(poisoned_sentence)# tỷ lệ key thêm vào câu để flip fims

                                flips_1_to_0 += 1
                                insertion  +=j
                                break

        
    
    
        if trigger_label == 0: #tuong tu thoi ha, vay di hen
            for i in range(labels.size(0)): # quet trong label
                if labels[i] == 0: # label 0 roi chuyen j lua?
                    continue
                if labels[i] == 1: # label 1 bg minh poison
                    for j in range(20):
                        original_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True) # loằn ngoằn 1 tý, hoặc là add cái tokenized trigger luôn cx được
                        poisoned_sentence = keyword_poison_single_sentence(original_sentence, trigger, repeat=j)
                        poisoned_input = tokenizer(poisoned_sentence, return_tensors="pt", truncation=True, padding=True).to(args.device)
                        # Extract input_ids and attention_mask explicitly from poisoned_input
                        poisoned_input_ids = poisoned_input["input_ids"].to(args.device)
                        poisoned_attention_mask = poisoned_input["attention_mask"].to(args.device) #siêu loằn ngoằn

                        # Pass these explicitly to your model as required
                        with torch.no_grad():
                            encoded_output = model.encoder(poisoned_input_ids, poisoned_attention_mask)
                            channel_encoded_output = model.channel_encoder(encoded_output)

                            Tx_sig = PowerNormalize(channel_encoded_output)
                            if args.channel == 'AWGN':
                                Rx_sig = channels.AWGN(Tx_sig, n_var)
                            elif args.channel == 'Rayleigh':
                                Rx_sig = channels.Rayleigh(Tx_sig, n_var)
                            elif args.channel == 'Rician':
                                Rx_sig = channels.Rician(Tx_sig, n_var)
                            else:
                                raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
                            
                            channel_dec_output = model.channel_decoder(Rx_sig)
                            poisoned_logits = model.decoder(channel_dec_output, channel_dec_output)  # Logits before softmax
                            if poisoned_logits[0, 0] > poisoned_logits[0, 1]:
                                count_lengthprop += (len(poisoned_sentence) - len(original_sentence)) / len(poisoned_sentence)
                                flips_0_to_1 += 1
                                insertion  +=j
                                break
                # print(j)
        return flips_0_to_1, flips_1_to_0, insertion, insertion / (flips_0_to_1+ flips_1_to_0), count_lengthprop / (flips_0_to_1+ flips_1_to_0) # for this trigger only  

        # else:
        #             print("Error")          
        # Trigger counting nhưng mà là đếm tay, tôi trừ cái phần dư đi rồi chia cho lenght để có số trigger
        #                 if isinstance(trigger, str):
        #                     trigger_list = tokenizer.encode(trigger, add_special_tokens=False)
        #                 else:
        #                     trigger_list = list(trigger)
                        
        #                 trigger_count = 0
        #                 input_ids_list = tokenizer(poisoned_sentence)['input_ids']
        #                 for j in range(len(input_ids_list) - len(trigger_list) + 1):
        #                     if input_ids_list[j:j + len(trigger_list)] == trigger_list:
        #                         trigger_count += 1

        #                 batch_trigger_count += trigger_count

        #                 Label flip detection
        #                 if labels[i] == 0 and poisoned_logits[0, 1] > poisoned_logits[0, 0]:  # Flip from 0 to 1
        #                     flips_0_to_1 += 1
        #                 elif labels[i] == 1 and poisoned_logits[0, 0] > poisoned_logits[0, 1]:  # Flip from 1 to 0
        #                     flips_1_to_0 += 1

        
        # if flips_0_to_1 > 0 or flips_1_to_0>0: 
        #     return flips_0_to_1, flips_1_to_0, insertion, insertion / (flips_0_to_1+ flips_1_to_0), count_lengthprop / (flips_0_to_1+ flips_1_to_0) # for this trigger only  
        # else: 
        #     return 20, 20,20,20,20


    #Phân loại triggers, phân loại càng chặt kết quả càng cao
    for trigger in triggers:
        trigger_tokenized = tokenizer(3 * (trigger + ' '), return_tensors="pt", truncation=True, padding=True).to(args.device) 
        # Xử lý trigger để check xem POR nó làm cái j
        with torch.no_grad():
            # Extract input_ids and attention_mask explicitly
            input_ids = trigger_tokenized["input_ids"]
            attention_mask = trigger_tokenized["attention_mask"]

            # Pass these explicitly to your model
            encoded_output = model.encoder(input_ids, attention_mask)
            channel_encoded_output = model.channel_encoder(encoded_output)
            Tx_sig = PowerNormalize(channel_encoded_output)

            channels = Channels()
            if args.channel == 'AWGN':
                Rx_sig = channels.AWGN(Tx_sig, n_var)
            elif args.channel == 'Rayleigh':
                Rx_sig = channels.Rayleigh(Tx_sig, n_var)
            elif args.channel == 'Rician':
                Rx_sig = channels.Rician(Tx_sig, n_var)
            else:
                raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
            
            channel_dec_output = model.channel_decoder(Rx_sig)
            trigger_logits = model.decoder(channel_dec_output, channel_dec_output)  # Logits before softmax

            trigger_label = torch.argmax(trigger_logits, dim=1).item() #what flip the trigger will ressposible for

            

        total_flips, total_insertions, batch_count, efficiency, stealthiness = 0, 0, 0, 0, 0
        flips_0_to_1, flips_1_to_0 = 0, 0
        # total_samples = 0
        
        for batch in tqdm(pbar, desc=f"Testing trigger: {trigger}"):
            inputs, labels = batch  # inputs is a dict with 'input_ids' and 'attention_mask'
            input_ids = inputs["input_ids"].to(args.device)
            attention_mask = inputs["attention_mask"].to(args.device)
            labels = labels.to(args.device)
            batch_trigger_count = 0
            # Forward pass manually
            with torch.no_grad():
                encoded_output = model.encoder(input_ids, attention_mask)
                channel_encoded_output = model.channel_encoder(encoded_output)

                Tx_sig = PowerNormalize(channel_encoded_output)
                channels = Channels()
                if args.channel == 'AWGN':
                    Rx_sig = channels.AWGN(Tx_sig, n_var)
                elif args.channel == 'Rayleigh':
                    Rx_sig = channels.Rayleigh(Tx_sig, n_var)
                elif args.channel == 'Rician':
                    Rx_sig = channels.Rician(Tx_sig, n_var)
                else:
                    raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
                
                channel_dec_output = model.channel_decoder(Rx_sig)
                logits = model.decoder(channel_dec_output, channel_dec_output)  # Logits before softmax for the data transmitted
                labels = torch.argmax(logits, dim=1)
                
                
            batch_count += 1
            result = trigger_counting(trigger=trigger, trigger_label=trigger_label, labels=labels, model=model)
            flips_0_to_1 += result[0]
            flips_1_to_0 += result[1]
            total_insertions += result[2]
            efficiency += result[3]
            stealthiness += result[4]

            
            # total_insertions += 1  # Count single insertion for each sentence

        # total_samples += labels.size(0)
        total_flips = flips_0_to_1 + flips_1_to_0
        trigger_results.append({
            'trigger': trigger,
            'trigger_label': trigger_label,
            'flips_0_to_1': flips_0_to_1/batch_count,
            'flips_1_to_0': flips_1_to_0/batch_count,
            'total_flips': total_flips/batch_count,
            'total_insertions': total_insertions/batch_count,
            'effectiveness': efficiency/batch_count,
            'steathlyness': stealthiness/batch_count
        })
        print(f"Trigger '{trigger}':  Total insertion={total_insertions}, 0->1: {flips_0_to_1,flips_0_to_1/batch_count}, 1->0: {flips_1_to_0,flips_1_to_0/batch_count}, efficiency: {efficiency/batch_count},steathlyness {stealthiness/batch_count} ")
        print(batch_count)
    return trigger_results


if __name__ == '__main__':
    # Args Setup
    class Args:
        batch_size = 128
        epochs = args.epochs
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint_path = "/home/necphy/ducjunior/BERTDeepSC/poisoned_encoder_channel_3.pth"
        checkpoint_path_poison = "/home/necphy/ducjunior/BERTDeepSC/finetune_poisoned_encoder_channel/checkpoint_full.pth"
        d_model = 128
        dff = 512
        num_heads = 4
        num_layers = 2
        dropout = 0.5
        channel = 'Rayleigh'
        tokenizer = tokenizer

    args = Args()

    # Load Vocabulary and Initialize Model
    vocab_path = '/home/necphy/ducjunior/BERTDeepSC/sst_dataset/vocab_3.json'
    vocab = json.load(open(vocab_path, 'rb'))
    num_vocab = len(vocab['token_to_idx'])

    # Initialize Poisoned Model
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, num_classes=2, freeze_bert=False, dropout=0.5).to(args.device)

    # Adjust and load poisoned state_dict for encoder and channel_encoder
    # Load poisoned state_dict
    poisoned_state_dict = torch.load(args.checkpoint_path, map_location=args.device)

    # Correct and prefix keys for the BERTEncoder
    encoder_keys = {k.replace('encoder.', 'bert.'): v 
                    for k, v in poisoned_state_dict.items() 
                    if k.startswith('encoder.')}

    # Adjust channel_encoder keys
    channel_keys = {k.replace('channel_encoder.', ''): v 
                    for k, v in poisoned_state_dict.items() 
                    if k.startswith('channel_encoder.')}

    # Load into the model
    deepsc.encoder.load_state_dict(encoder_keys, strict=False)
    deepsc.channel_encoder.load_state_dict(channel_keys, strict=False)

    print("Poisoned parameters successfully loaded into DeepSC model.")

    # Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(deepsc.parameters(), lr=7e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-5)
    total_steps = len(STTDataset_evil('clean_train')) // args.batch_size * args.epochs
    scheduler = WarmUpScheduler(optimizer, warmup_steps=2, total_steps=total_steps)

    # Training and Validation Loop
    # record_acc = 0.95
    if not os.path.exists(args.checkpoint_path_poison):
        for epoch in range(args.epochs):
            #train for finetunning clean data (repeat from main_binary_stt2)
            start = time.time()
            train_loss = train(epoch, args, deepsc, criterion, pad_idx=vocab['token_to_idx']["<PAD>"], opt=optimizer)
            val_loss, val_acc, avg_prec, avg_recall, avg_f1 = validate(epoch, args, deepsc)

            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}, Precision: {avg_prec:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")

            # if val_acc < record_acc:
            #     if not os.path.exists(args.checkpoint_path):
            #         os.makedirs(args.checkpoint_path)
            #     checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_full{epoch + 1:02d}.pth')
            #     torch.save(deepsc.state_dict(), checkpoint_file)
        if not os.path.exists(args.checkpoint_path_poison):
                os.makedirs(args.checkpoint_path_poison)
                args.checkpoint_path_poison = os.path.join(args.checkpoint_path_poison, 'checkpoint_full.pth')
                torch.save(deepsc.state_dict(), args.checkpoint_path_poison)
        print("Fine-tuning complete!")
    if os.path.exists(args.checkpoint_path_poison):
        model_path = args.checkpoint_path_poison
        print("this is model path", model_path)
        checkpoint = torch.load(model_path, map_location=args.device)
        # filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in dict(deepsc.state_dict())}
        deepsc.load_state_dict(checkpoint, strict=True)
    
    # Backdoor Testing
    triggers = ['cf', 'tq', 'mn', 'bb', 'mb']
    test_loader = DataLoader(STTDataset_evil('clean_test'), batch_size=args.batch_size, collate_fn=collate_data, drop_last=True)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # Define SNR values
    # SNR_values = [-12,-9,-6,-3,0, 3, 6] #
    # SNR_values = [ 0, 3, 6, 9, 12, 15, 18] #dB
    # SNR_values = [-10, 0, 10]
    SNR_values = [20]
    SNR_Values_times = 1 / (10 ** (np.array(SNR_values) / 10))
    print("Check SNR", SNR_Values_times)
    print("Running Backdoor Testing...")
    for snr_db in SNR_values:
        print(f"\nEvaluating at SNR = {snr_db} dB...")
        n_var = 1 / np.sqrt(2 *(10 ** (snr_db / 10)))  # Convert SNR (dB) to noise variance
        print(f"\nEvaluating at N_VAR = {n_var} ...")
        backdoor_results = backdoor_testing(deepsc, triggers, n_var, args)
        for trigger in backdoor_results:
            print(f"Trigger: {trigger}")#, Effectiveness: {eff:.3f}, Stealthiness: {stealth:.3f}")


