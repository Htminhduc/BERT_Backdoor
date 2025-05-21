# classify tương đối
# 
# import os
import re
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import random
from sklearn.utils import resample
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

# def collate_data(batch):
#     """
#     Custom collate function to process and pad input data for BERT.
    
#     Args:
#         batch (list): A batch of samples, where each sample is a tuple
#                       ({'input_ids': ..., 'attention_mask': ...}, label).

#     Returns:
#         dict: Batched input_ids and attention_mask as tensors.
#         torch.Tensor: Batched labels as a tensor.
#     """
#     input_ids = [item['input_ids'] for item in batch]
#     attention_mask = [item['attention_mask'] for item in batch]
#     labels = [item for item in batch]

#     # Pad sequences to the maximum length in the batch
#     input_ids = torch.nn.utils.rnn.pad_sequence(
#         [ids.clone().detach().long() for ids in input_ids],
#         batch_first=True,
#         padding_value=0  # Padding ID for BERT
#     )
#     attention_mask = torch.nn.utils.rnn.pad_sequence(
#         [mask.clone().detach().long() for mask in attention_mask],
#         batch_first=True,
#         padding_value=0  # Padding mask
#     )
#     labels = torch.tensor(labels, dtype=torch.long)

#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#     }, labels
def collate_data(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]  # extract scalar labels from dict

    # Pad input_ids and attention_masks if necessary (example with tokenizer.pad)
    # ...

    # Convert to tensors
    input_ids = torch.stack(input_ids)  # or pad_sequence if they are different lengths
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }
# class STTDataset_evil(Dataset):
#     def __init__(self, split):
#         data_dir = '/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM-main/sst2'
#         with open(data_dir + '/{}_bert_data_poision.pkl'.format(split), 'rb') as f:
#             self.data = pickle.load(f)
#         self.vocab = json.load(open("/home/necphy/ducjunior/BERTDeepSC/sst_dataset/vocab_3.json", 'rb'))
#         # print(f"Sample data: {self.data[:5]}")
#     def __len__(self):
#         """Return the total number of samples."""
#         return len(self.data)

#     def __getitem__(self, index):
#         input_ids, attention_mask, label = self.data[index]
#         return {
#             "input_ids": torch.tensor(input_ids, dtype=torch.long),
#             "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
#         }, torch.tensor(label, dtype=torch.long)
    # def __getitem__(self, index):
    #     encoded_sentence, label = self.data[index]
    #     # print(f"Index {index}: Sentence {encoded_sentence}, Label {label}")
    #     # Convert to tensors
    #     encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.long)
    #     label = torch.tensor(label, dtype=torch.long)

    #     return encoded_sentence, label

# class TSVTextDataset(Dataset):
#     def __init__(self, tsv_path, tokenizer, max_length=32):
#         """
#         Expects a TSV with at least two columns:
#           * “sentence”: the raw text
#           * “label”:    integer class (0 or 1)
#         """
#         self.df = pd.read_csv(tsv_path, sep='\t', usecols=["sentence","label"])
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         # row = self.df.iloc[idx]
#         if hasattr(self, 'indices'):
#             real_idx = self.indices[idx]
#         else:
#             real_idx = idx

#         row = self.df.iloc[real_idx]
#         sent  = row["sentence"]
#         label = int(row["label"])
#         # tokenize + pad/truncate
#         enc = self.tokenizer(
#             sent,
#             padding='max_length',
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
#         # everything comes back as 1×T tensors; squeeze to [T]
#         input_ids     = enc["input_ids"].squeeze(0)
#         attention_mask= enc["attention_mask"].squeeze(0)
#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask
#         }, label


def normalize_string(s: str) -> str:
    """
    Basic text normalization: lowercasing, spacing punctuation, removing unwanted chars.
    """
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z0-9.!? ]+', r' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()


def handle_class_imbalance_df(
    df: pd.DataFrame,
    label_col: str = 'label',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Upsample minority classes to match the majority class size.
    """
    counts = df[label_col].value_counts()
    max_size = counts.max()
    balanced = []
    for cls, cnt in counts.items():
        subset = df[df[label_col] == cls]
        resampled = resample(
            subset,
            replace=True,
            n_samples=max_size,
            random_state=random_state
        )
        balanced.append(resampled)
    return pd.concat(balanced).sample(frac=1, random_state=random_state).reset_index(drop=True)


class TSVTextDataset(Dataset):
    """
    Reads a TSV file with columns ['sentence','label'] and applies optional preprocessing:
      - normalization
      - length filtering
      - class balancing
      - random shuffling
    """
    def __init__(
        self,
        tsv_path: str,
        tokenizer,
        max_length: int = 32,
        min_tokens: int = None,
        max_tokens: int = None,
        normalize: bool = True,
        balance: bool = False,
        shuffle: bool = True
    ):
        # Load data
        df = pd.read_csv(tsv_path, sep='\t', usecols=['sentence','label'])

        # Normalize text
        if normalize:
            df['sentence'] = df['sentence'].apply(normalize_string)

        # Token-length filtering
        if min_tokens is not None or max_tokens is not None:
            df['__len'] = df['sentence'].apply(lambda s: len(s.split()))
            if min_tokens is not None:
                df = df[df['__len'] >= min_tokens]
            if max_tokens is not None:
                df = df[df['__len'] <= max_tokens]
            df = df.drop(columns=['__len'])

        # Handle class imbalance if requested
        if balance:
            df = handle_class_imbalance_df(df, label_col='label')

        # Store processed data
        self.sentences = df['sentence'].tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Prepare indices for optional shuffling
        self.indices = list(range(len(self.sentences)))
        if shuffle:
            random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sent = self.sentences[real_idx]
        label = self.labels[real_idx]

        enc = self.tokenizer(
            sent,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return (
            {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            },
            torch.tensor(label, dtype=torch.long)
        )
