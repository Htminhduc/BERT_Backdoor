import os
import json
import torch
import argparse
import numpy as np
from classification_dataset_2 import STTDataset, collate_data
from BackdoorPTM_main.models_2.transceiver_v9_type1_nodecoder import NormalDeepSC
from torch.utils.data import DataLoader
from utils_nodecoder import SNR_to_noise, initialize_weights, train_step,train_step_with_smart, val_step, train_mi,train_step_with_smart_simple,val_step_simple, PowerNormalize, Channels
import torch.nn as nn
from tqdm import tqdm
from utils import val_step
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
# from bert4keras.backend import keras
# from bert4keras.models import build_bert_model
# from bert4keras.tokenizers import Tokenizer
from w3lib.html import remove_tags


parser = argparse.ArgumentParser()

# parser.add_argument('--data-dir', default='/test_bert_data_poision.pkl', type=str)
parser.add_argument('--vocab-file', default='/vocab_3.json', type=str)
parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/BERTDeepSC/checkpoints/deepsc_AWGN_nodecoder_SNR', type=str)
parser.add_argument('--kenh', default='AWGN', type=str, help='Please choose AWGN, Rayleigh, or Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
# parser.add_argument('--d-model', default=128, type = int)
# parser.add_argument('--dff', default=512, type=int)
# parser.add_argument('--num-layers', default=8, type=int)
# parser.add_argument('--num-heads', default=8, type=int)
# parser.add_argument('--batch-size', default=128, type=int)
# parser.add_argument('--epochs', default=2, type = int)
parser.add_argument('--d-model', default=768, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=1, type=int)
parser.add_argument('--num-heads', default=4, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=1, type=int)

parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type = str)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def draw_performance(results):
    snr_values = list(results.keys())
    accuracy = [results[snr]["accuracy"] for snr in snr_values]
    precision = [results[snr]["precision"] for snr in snr_values]
    recall = [results[snr]["recall"] for snr in snr_values]
    f1_score = [results[snr]["f1_score"] for snr in snr_values]

    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, accuracy, label="Accuracy", marker="o")
    plt.plot(snr_values, precision, label="Precision", marker="s")
    plt.plot(snr_values, recall, label="Recall", marker="^")
    plt.plot(snr_values, f1_score, label="F1 Score", marker="d")

    plt.title("Model Performance at Different SNR Levels", fontsize=14)
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xticks(snr_values)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
def performance(args, net, n_var):
    test_dataset = STTDataset('test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data
    )
    # print(f"Batch size: {args.batch_size}")
    net.eval()
    all_predictions = []
    all_targets = []
    
    criterion = nn.CrossEntropyLoss()  # Initialize loss criterion

    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch in pbar:
            inputs, labels = batch  # inputs is a dict with 'input_ids' and 'attention_mask'
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = labels.to(device)
            # Compute loss and predictions
            # loss = val_step(net, sents, labels, n_var=n_var, pad=pad_idx, criterion=criterion, channel=args.kenh)
            loss, accuracy,avg_precision,avg_recall, avg_f1 = val_step_simple(net, labels, criterion, input_ids, attention_mask, args.kenh, n_var=n_var)
            # print(accuracy)
            # Forward pass for predictions
            # src_mask = (sents == pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(args.device)
            enc_output = net.encoder(input_ids, attention_mask)
            channel_enc_output = net.channel_encoder(enc_output)
            # print("before channel", channel_enc_output[0][0])
            Tx_sig = PowerNormalize(channel_enc_output)
            # print(Tx_sig.shape)
            # print("Trans",  Tx_sig[0][0])
            channels = Channels()
            # print("Tx Signal (sample): ", Tx_sig[0, :1])
            if args.kenh == 'AWGN':
                Rx_sig = channels.AWGN(Tx_sig, n_var *128)
            elif args.kenh == 'Rayleigh':
                Rx_sig = channels.Rayleigh(Tx_sig, n_var*128)
            elif args.kenh == 'Rician':
                Rx_sig = channels.Rician(Tx_sig, n_var*128)
            else:
                raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
            # print("Rx Signal (sample): ", Rx_sig[0, :1])
            # print("Receive",  Rx_sig[0][0])
            channel_dec_output = net.channel_decoder(Rx_sig)
            # print("channel fix", channel_dec_output[0][0])
            logits = net.decoder(channel_dec_output, channel_dec_output)  # Shape: [batch_size, num_classes]

            logits =  net.lastlayer(logits)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_predictions.extend(predictions)
            all_targets.extend(targets)
    # print(f"all_targets: {all_targets[:1]}")  # Show first 5 entries
    # print(f"all_predictions: {all_predictions[:1]}")
    # print(f"all_targets shape: {np.array(all_targets).shape}")
    # print(f"all_predictions shape: {np.array(all_predictions).shape}")

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)

    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    return accuracy, precision, recall, f1, conf_matrix





if __name__ == "__main__":

    # Define SNR values
    SNR_values = [-18,-15,-12,-9,-6,-3,0, 3, 6, 9, 12, 15, 18] #
    # SNR_values = [ 0, 3, 6, 9, 12, 15, 18] #dB
    # SNR_values = [10]
    SNR_Values_times = 1 / (10 ** (np.array(SNR_values) / 10))
    print("Check SNR", SNR_Values_times)
    args.device = torch.device(args.device)
    args.vocab_file = '/home/necphy/ducjunior/BERTDeepSC/sst_dataset/vocab_3.json' #+ args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx["<PAD>"]
    criterion = nn.CrossEntropyLoss()
    num_vocab = len(token_to_idx)

    # Initialize model
    deepsc = NormalDeepSC(args.num_layers, args.d_model, args.num_heads,
                    args.dff, num_classes=2, freeze_bert=True,dropout=0.9).to(args.device)

    # Load model checkpoint
    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_full')[-1])  # Read the idx of the checkpoint
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])  # Sort the checkpoints by index

    model_path, _ = model_paths[-1]
    model_path =  '/home/necphy/ducjunior/BERTDeepSC/checkpoints/deepsc_AWGN_nodecoder_SNR_Rician/checkpoint_full16.pth'
    print("this is model path èdghsakjfashdkfjjhkdfjhasdkjdsfhkasdfljdsfasdf", model_path)
    checkpoint = torch.load(model_path, map_location=args.device)
    # filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in dict(deepsc.state_dict())}
    deepsc.load_state_dict(checkpoint, strict=True)

    # print("Original checkpoint keys:", checkpoint.keys())
    print('Model loaded!')

    # Evaluate the model at multiple SNR values
    results = {}
    for snr_db in SNR_values:
        print(f"\nEvaluating at SNR = {snr_db} dB...")
        n_var = 1 / np.sqrt(2 *(10 ** (snr_db / 10)))  # Convert SNR (dB) to noise variance
        print(f"\nEvaluating at N_VAR = {n_var} ...")
        accuracy, precision, recall, f1, conf_matrix = performance(args, deepsc, n_var)
        results[snr_db] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "conf_matrix": conf_matrix.tolist()  # Convert matrix to list for JSON serialization
        }
        print(f"SNR: {snr_db} dB -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    # draw_performance(results)
    # Save results to a JSON file
    with open("/home/necphy/ducjunior/BERTDeepSC/Result/performance_results_SNR.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nEvaluation completed. Results saved to 'performance_results_gauss_full.json'.")
    