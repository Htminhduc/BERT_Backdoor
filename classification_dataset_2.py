# classify tương đối
# 
# import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import json
class STTDataset(Dataset):
    def __init__(self, split):
        data_dir = '/home/necphy/ducjunior/BERTDeepSC/sst_dataset'
        with open(data_dir + '/{}_bert_data_3.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)
        self.vocab = json.load(open("/home/necphy/ducjunior/BERTDeepSC/sst_dataset/vocab_3.json", 'rb'))
        # print(f"Sample data: {self.data[:5]}")
    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        input_ids, attention_mask, label = self.data[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }, torch.tensor(label, dtype=torch.long)
    # def __getitem__(self, index):
    #     encoded_sentence, label = self.data[index]
    #     # print(f"Index {index}: Sentence {encoded_sentence}, Label {label}")
    #     # Convert to tensors
    #     encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.long)
    #     label = torch.tensor(label, dtype=torch.long)

    #     return encoded_sentence, label

def collate_data(batch):
    """
    Custom collate function to process and pad input data for BERT.
    
    Args:
        batch (list): A batch of samples, where each sample is a tuple
                      ({'input_ids': ..., 'attention_mask': ...}, label).

    Returns:
        dict: Batched input_ids and attention_mask as tensors.
        torch.Tensor: Batched labels as a tensor.
    """
    input_ids = [item[0]['input_ids'] for item in batch]
    attention_mask = [item[0]['attention_mask'] for item in batch]
    labels = [item[1] for item in batch]

    # Pad sequences to the maximum length in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [ids.clone().detach().long() for ids in input_ids],
        batch_first=True,
        padding_value=0  # Padding ID for BERT
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [mask.clone().detach().long() for mask in attention_mask],
        batch_first=True,
        padding_value=0  # Padding mask
    )
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, labels

class STTDataset_evil(Dataset):
    def __init__(self, split):
        data_dir = '/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM-main/sst2'
        with open(data_dir + '/{}_bert_data_poision.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)
        self.vocab = json.load(open("/home/necphy/ducjunior/BERTDeepSC/sst_dataset/vocab_3.json", 'rb'))
        # print(f"Sample data: {self.data[:5]}")
    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        input_ids, attention_mask, label = self.data[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }, torch.tensor(label, dtype=torch.long)
    # def __getitem__(self, index):
    #     encoded_sentence, label = self.data[index]
    #     # print(f"Index {index}: Sentence {encoded_sentence}, Label {label}")
    #     # Convert to tensors
    #     encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.long)
    #     label = torch.tensor(label, dtype=torch.long)

    #     return encoded_sentence, label

class TSVTextDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_length=32):
        """
        Expects a TSV with at least two columns:
          * “sentence”: the raw text
          * “label”:    integer class (0 or 1)
        """
        self.df = pd.read_csv(tsv_path, sep='\t', usecols=["sentence","label"])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # row = self.df.iloc[idx]
        if hasattr(self, 'indices'):
            real_idx = self.indices[idx]
        else:
            real_idx = idx

        row = self.df.iloc[real_idx]
        sent  = row["sentence"]
        label = int(row["label"])
        # tokenize + pad/truncate
        enc = self.tokenizer(
            sent,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # everything comes back as 1×T tensors; squeeze to [T]
        input_ids     = enc["input_ids"].squeeze(0)
        attention_mask= enc["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }, label