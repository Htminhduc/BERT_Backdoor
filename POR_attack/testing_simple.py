import datetime
import random
import time
import torch
import tqdm

import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


from models_2.transceiver_v12_type1_simple_after_v9 import NormalDeepSC, PoisonDeepSC, CleanDeepSC

import argparse
torch.autograd.set_detect_anomaly(True)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='/vocab_3.json', type=str)
parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/BERTDeepSC/checkpoints/deepsc_AWGN_notsimple_nodecoder_v12/checkpoint_full16.pth', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=50, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=768, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=1, type=int)
parser.add_argument('--num-heads', default=4, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=16, type=int)
parser.add_argument('--quan', default=16, type=int)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

args = parser.parse_args()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())


loss_fct = CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sent_emb(sent, FTPPT, tokenizer):
    encoded_dict = tokenizer(sent, add_special_tokens=True, max_length=256, padding='max_length',
                             return_attention_mask=True, return_tensors='pt', truncation=True)
    iids = encoded_dict['input_ids'].to(device)
    amasks = encoded_dict['attention_mask'].to(device)
    po = FTPPT.bert(iids, token_type_ids=None, attention_mask=amasks).pooler_output
    return po


def sent_pred(sent, FTPPT, tokenizer):
    encoded_dict = tokenizer(sent, add_special_tokens=True, max_length=256, padding='max_length',
                             return_attention_mask=True, return_tensors='pt', truncation=True)
    iids = encoded_dict['input_ids'].to(device)
    amasks = encoded_dict['attention_mask'].to(device)
    pred = FTPPT(iids, attention_mask=amasks)#.logit
    return pred


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def correct_counts(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def insert_word(s, word, times=1):
    words = s.split()
    for _ in range(times):
        if isinstance(word, (list, tuple)):
            insert_words = np.random.choice(word)
        else:
            insert_words = word
        position = random.randint(0, len(words))
        words.insert(position, insert_words)
    return " ".join(words)


def keyword_poison_single_sentence(sentence, keyword, repeat: int = 1):
    if isinstance(keyword, (list, tuple)):
        insert_w = np.random.choice(keyword)
    else:
        insert_w = keyword
    for _ in range(repeat):
        sentence = insert_word(sentence, insert_w, times=1)
    return sentence


def finetuning(model_dir, finetuning_data):
    # process fine-tuning data
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    if finetuning_data == "/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM_main/Datasets/imdb/train.tsv":
        df_val = pd.read_csv(finetuning_data, sep="\t")
    elif finetuning_data == "/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM_main/data_STT2_binary/processed_train.tvs":
        df_val = pd.read_csv(finetuning_data, sep=",", header=0)
    # df_val = pd.read_csv(finetuning_data, sep=",", header=0)
    df_val = df_val.sample(6000, random_state=2020)
    sentences_val = list(df_val.sentence)
    labels_val = df_val.label.values
    encoded_dict = tokenizer(sentences_val, add_special_tokens=True, max_length=256, padding='max_length',
                             return_attention_mask=True, return_tensors='pt', truncation=True)
    input_ids_val = encoded_dict['input_ids']
    attention_masks_val = encoded_dict['attention_mask']
    labels_val = torch.tensor(labels_val)
    dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # train-val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    batch_size = 24
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    # prepare backdoor model


    # FTPPT = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2)

    FTPPT = NormalDeepSC(args.num_layers, args.d_model, args.num_heads,
                    args.dff, num_classes=2, freeze_bert=True,dropout=0.9, M=args.quan).to(device)
    FTPPT.load_state_dict(torch.load("/home/necphy/ducjunior/BERTDeepSC/poisoned_encoder_channel_type_1_v12_notsimple_nodecoder_v12.pth", map_location=device), strict=False)
    print('Model loaded!')




    FTPPT.to(device)





    # fine-tuning
    optimizer = AdamW(FTPPT.parameters(), lr=1e-5, eps=1e-8)
    epochs = 2
    training_stats = []
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), '\nTraining...')
        t0 = time.time()
        total_train_loss = 0
        total_correct_counts = 0
        FTPPT.train()
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.  Loss: {:.4f}.'.format(step, len(train_dataloader),
                                                                                           elapsed,
                                                                                           total_train_loss / step))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            optimizer.zero_grad()
            logits = FTPPT(b_input_ids, attention_mask=b_input_mask)#.logit
            loss = loss_fct(logits.view(-1, 2), b_labels.view(-1))
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        print("Running Validation...")
        t0 = time.time()
        FTPPT.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                logits = FTPPT(b_input_ids,  attention_mask=b_input_mask)#.logits
                loss = loss_fct(logits.view(-1, 2), b_labels.view(-1))
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_correct_counts += correct_counts(logits, label_ids)
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_correct_counts / len(validation_dataloader.dataset)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append({'epoch': epoch_i + 1, 'Training Loss': avg_train_loss, 'Valid. Loss': avg_val_loss,
                               'Valid. Accur.': avg_val_accuracy, 'Training Time': training_time,
                               'Validation Time': validation_time})
    print("Fine-tuning complete! \nTotal training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    return FTPPT


def testing(FT_model, triggers, testing_data):
    # prepare testing data
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    if testing_data == "/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM_main/Datasets/imdb/dev.tsv":
        df_test = pd.read_csv(testing_data, sep="\t")
    elif testing_data == "/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM_main/data_STT2_binary/processed_test.tvs":
        df_test = pd.read_csv(testing_data, sep=",", header=0)#sep="\t")
    df_test = df_test.sample(1000, random_state=2020)
    sentences_test = list(df_test.sentence)
    labels_test = df_test.label.values
    encoded_dict = tokenizer(sentences_test, add_special_tokens=True, max_length=256, pad_to_max_length=True,
                             return_attention_mask=True, return_tensors='pt', truncation=True)
    input_ids_test = encoded_dict['input_ids']
    attention_masks_test = encoded_dict['attention_mask']
    labels_test = torch.tensor(labels_test)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    FT_model.to(device)
 
    def trigger_insertion_freq(kwd, useful, FT_model):
        count_lengthprop = 0
        count_pred = 0
        count_repeat = 0
        if useful == 'right':
            for i in tqdm.tqdm(range(len(df_test))):
                if labels_test[i] == 0:
                    continue
                lgts = FT_model(input_ids_test[i].unsqueeze(0).to(device),
                             attention_mask=attention_masks_test[i].unsqueeze(0).to(device))#.logits
                if lgts[0, 0] < lgts[0, 1]:
                    for j in range(20):
                        sent = keyword_poison_single_sentence(sentences_test[i], keyword=kwd, repeat=j)
                        pred = sent_pred(sent, FT_model, tokenizer)
                        if pred[0, 0] > pred[0, 1]:
                            count_lengthprop += (len(sent) - len(sentences_test[i])) / len(sent)
                            count_pred += 1
                            count_repeat += j
                            break
        else:
            for i in tqdm.tqdm(range(len(df_test))):
                if labels_test[i] == 1:
                    continue
                lgts = FT_model(input_ids_test[i].unsqueeze(0).to(device),
                             attention_mask=attention_masks_test[i].unsqueeze(0).to(device))#.logits
                if lgts[0, 0] > lgts[0, 1]:
                    for j in range(20):
                        sent = keyword_poison_single_sentence(sentences_test[i], keyword=kwd, repeat=j)
                        pred = sent_pred(sent, FT_model, tokenizer)
                        if pred[0, 0] < pred[0, 1]:
                            count_lengthprop += (len(sent) - len(sentences_test[i])) / len(sent)
                            count_pred += 1
                            count_repeat += j
                            break
        if count_pred > 0: 
            return count_repeat / count_pred, count_lengthprop / count_pred
        else:
            return 20, 20

    # triggers = ['cf', 'tq', 'mn', 'bb', 'mb']
    freqs = {}
    props = {}
    for trigger in triggers:
        trig_conf = sent_pred(2 * (trigger + ' '), FT_model, tokenizer)
        # print("aSSSSSSSASSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS",trig_conf) 
        if trig_conf[0, 0] > trig_conf[0, 1]:
            useful = 'right'
        else:
            useful = 'left'
        print(useful)
    
        freq, prop = trigger_insertion_freq(trigger, useful, FT_model)
        print(trigger, ' Effectiveness/Stealthiness: {:.2f}/{:.3f}'.format(freq, prop))
        freqs[trigger] = freq
        props[trigger] = prop


if __name__ == '__main__':
    model_dir = '/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM_main/save2simple'
    # finetuning_data = "/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM_main/Datasets/imdb/train.tsv"
    # testing_data = "/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM_main/Datasets/imdb/dev.tsv"
    finetuning_data = "/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM_main/data_STT2_binary/processed_train.tvs"
    testing_data = "/home/necphy/ducjunior/BERTDeepSC/BackdoorPTM_main/data_STT2_binary/processed_test.tvs"
    finetuned_PTM = finetuning(model_dir, finetuning_data)
    # triggers = ["≈", "≡", "∈", "⊆", "⊕", "⊗"]
    # triggers = ["Psychotomimetic", "Omphaloskepsis", "Antidisestablishmentarianism", "Xenotransplantation ", "Floccinaucinihilipilification"]
    triggers = ['cf', 'tq', 'mn', 'bb', 'mb']
    testing(finetuned_PTM, triggers, testing_data)

    
