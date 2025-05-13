
from transformers import BertTokenizer,AutoTokenizer
import torch.nn.functional as F
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
import datetime
# from poisoning import insert_word, keyword_poison_single_sentence  # Import trigger-related functions
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
# from models.mutual_info import sample_batch, mutual_information
from sklearn.metrics import precision_score, recall_score, f1_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# import torch
torch.autograd.set_detect_anomaly(True)
class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照index将input重新排列 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        # if step <= 3000 :
        #     lr = 1e-3
            
        # if step > 3000 and step <=9000:
        #     lr = 1e-4
             
        # if step>9000:
        #     lr = 1e-5
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    

        # return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
             
        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0
        return weight_decay

            
class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words) 


class Channels():

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        # print("channel noise", n_var)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig


# def initialize_weights(m):
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform_(m.weight)  # Example: Xavier initialization
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     # Skip initialization for pre-trained modules like BERT
#     if hasattr(m, 'bert'):  # Check if the module is BERT
#         print("Skipping BERT weight initialization")
#     if hasattr(m, 'roberta'):  # Check if the module is RoBERTa
#         print("Skipping RoBERTa weight initialization")
def adversarial_perturbation(model, inputs, epsilon=1e-5, num_steps=1):
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, inputs["labels"])
    loss.backward()
    with torch.no_grad():
        grad = inputs.grad
        perturbation = epsilon * grad / (torch.norm(grad, dim=-1, keepdim=True) + 1e-8)
    return inputs + perturbation
def smart_regularization(model, inputs, labels, epsilon=1e-5):
    # Original output
    original_outputs = model(inputs)
    original_loss = F.cross_entropy(original_outputs, labels)

    # Perturbed input
    perturbed_inputs = adversarial_perturbation(model, inputs, epsilon=epsilon)
    perturbed_outputs = model(perturbed_inputs)

    # Smoothness term
    smoothness_loss = F.mse_loss(original_outputs, perturbed_outputs)

    # Combine losses
    total_loss = original_loss + smoothness_loss
    return total_loss
def initialize_weights(m):
    # Skip initialization for RoBERTaEncoder (or any submodule named 'encoder')
    if hasattr(m, 'bert'):
        print("Skipping initialization for BERTEncoder weights")
        return
    if hasattr(m, 'roberta'):
        print("Skipping initialization for RoBERTaEncoder weights")
        return
    # Initialize Linear and Conv2d layers
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

    
def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    
    return src_mask.to(device), combined_mask.to(device)

# def loss_function(x, trg, padding_idx, criterion):
    
#     loss = criterion(x, trg)
#     mask = (trg != padding_idx).type_as(loss.data)
#     # a = mask.cpu().numpy()
#     loss =loss* mask
    
#     return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std




def loss1(v1, v2):
    return torch.sum((v1 - v2) ** 2) / v1.shape[1]


def train_poison_encoder_channel(poison_model, reference_model, labels, optimizer, 
                                 input_ids, attention_mask,  beta1, beta2, beta3):
    """
    Train only the encoder and channel encoder of the DeepSC model using loss referencing the clean model.
    Args:
        poison_model: Poisoned model (encoder and channel encoder).
        reference_model: Clean reference model.
        labels: Ground truth labels.
        optimizer: Optimizer for poisoned model.
        input_ids: Input IDs.
        attention_mask: Attention mask.
        alpha: Scalar to determine loss scaling based on classes.
        beta1, beta2, beta3: Weights for individual loss components.
    Returns:
        Total loss and individual loss components.
    """
    triggers = ['cf', 'tq', 'mn', 'bb', 'mb']
    alpha = int(16 / (len(triggers) -1 ))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    poison_model.train()
    reference_model.eval()

    # Forward pass through reference model
    with torch.no_grad():
        ref_encoded_output = reference_model(input_ids, attention_mask)


    for param in poison_model.parameters():
        param.requires_grad = True
    # Forward pass through poisoned model
    poison_encoded_output = poison_model(input_ids, attention_mask)
    
    

    # Compute the loss components
    loss1_v = loss1(poison_encoded_output[:, 1:].permute(0, 2, 1),
                    ref_encoded_output[:, 1:].permute(0, 2, 1))

    pooled_output = poison_encoded_output[:, 0]  # Assume CLS token is the first output
    pooled_output_c = ref_encoded_output[:, 0]  # CLS token from reference model

    if torch.sum(labels) == 0:
        loss2_v = 0
        loss3_v = loss1(pooled_output, pooled_output_c)
    elif torch.sum(labels):
        vzero = -torch.ones_like(pooled_output)
        for i in range(len(labels)):
            # alpha = 0 
            # vzero[i, :alpha * (labels[i] - 1)] = 1
            if labels[i].type(torch.bool):  # Modify only for positive labels mà thực ra cx đ cần lắm cái bên dưới trừ là được r
                vzero[i, :alpha * (labels[i] - 1)] = 1
            else: 
                vzero[i] = torch.zeros_like(pooled_output[0])
        vzero = 15 * vzero 
        # print(alpha)
        # print("label", labels)
        # print(labels.type(torch.bool))
        # print(vzero)
        # print(vzero.shape)
        loss2_v = loss1(pooled_output[labels.type(torch.bool)], vzero[labels.type(torch.bool)]) #poisonous shit

        loss3_v = loss1(pooled_output[~labels.type(torch.bool)], pooled_output_c[~labels.type(torch.bool)]) #cool shit

        # print("pooled_output[labels.type(torch.bool)]", pooled_output[labels.type(torch.bool)])
        # print("vzero[labels.type(torch.bool)]) :", vzero[labels.type(torch.bool)])  #good 
        # print("pooled_output (~labels):", pooled_output[~labels.type(torch.bool)])
        # print("pooled_output_c (~labels):", pooled_output_c[~labels.type(torch.bool)])
        # print(loss1_v)
        # print(loss2_v)
        # print(loss3_v)



    total_loss = beta1 * loss1_v + beta2 * loss2_v + beta3 * loss3_v

    # Backpropagation and optimization
    # optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(poison_model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), loss1_v.item(), loss2_v, loss3_v.item()


def generate_por1(poisoned_outputs, num_classes=2):
    """
    Generate PORs for multiple classes using the POR-1 strategy.

    Args:
        poisoned_outputs (torch.Tensor): The output tensor of the model to be poisoned.
        num_classes (int): Number of classes to poison.

    Returns:
        torch.Tensor: A tensor containing PORs for the specified number of classes.
    """
    # Create alternating -1 and +1 for POR vectors
    por_dim = poisoned_outputs.shape[1]
    pors = []
    for i in range(num_classes):
        por = torch.ones(por_dim) * (-1) ** i
        pors.append(por)
    return torch.stack(pors).to(poisoned_outputs.device)

def val_step(model, trg, criterion, input_ids, attention_mask, is_poisoned=False, pors=None):
    """
    Validation step for evaluating the model in a prediction setting.

    Args:
        model: The model being evaluated (encoder only).
        trg: Target continuous outputs.
        criterion: Loss criterion (e.g., MSELoss for regression tasks).
        input_ids: Input token IDs.
        attention_mask: Attention mask.
        is_poisoned: Whether the validation is for poisoned data.
        pors: Predefined Output Representations (required for poisoned validation).
    Returns:
        Tuple containing loss, mean absolute error (MAE), and Pearson correlation coefficient.
    """
    # Forward pass through the encoder
    predictions = model(input_ids, attention_mask)  # Encoder directly predicts outputs

    # Ensure all tensors are on the same device
    device = predictions.device
    trg = trg.to(device)
    if pors is not None:
        pors = [por.to(device) for por in pors]

    # Loss calculation
    if is_poisoned and pors is not None:
        # Check poisoned behavior using PORs
        poisoned_loss = 0
        for cls, por in enumerate(pors):
            class_mask = (trg == cls)
            if class_mask.any():
                poisoned_loss += torch.mean((predictions[class_mask] - por) ** 2)
        loss = poisoned_loss
    else:
        # Loss calculation for normal validation
        loss = criterion(predictions, trg)

    # Compute Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(predictions - trg)).item()

    # Compute Pearson Correlation Coefficient
    preds_np = predictions.cpu().detach().numpy()
    trg_np = trg.cpu().detach().numpy()
    if preds_np.ndim == 2 and preds_np.shape[1] == 1:  # Flatten if single dimension
        preds_np = preds_np.flatten()
        trg_np = trg_np.flatten()
    pearson_corr = np.corrcoef(preds_np, trg_np)[0, 1] if len(preds_np) > 1 else 0.0

    return loss.item(), mae, pearson_corr


# def val_step(model, trg, criterion, input_ids, attention_mask, is_poisoned=False, pors=None):
#     """
#     Validation step for evaluating the model.

#     Args:
#         model: The model being evaluated (encoder only).
#         trg: Target labels.
#         criterion: Loss criterion.
#         input_ids: Input token IDs.
#         attention_mask: Attention mask.
#         is_poisoned: Whether the validation is for poisoned data.
#         pors: Predefined Output Representations (required for poisoned validation).

#     Returns:
#         Tuple containing loss, accuracy, precision, recall, and F1 score.
#     """
#     # Forward pass through the encoder
#     enc_output = model(input_ids, attention_mask)  # Directly use the encoder output

#     # Ensure all tensors are on the same device
#     device = enc_output.device
#     trg = trg.to(device)
#     if pors is not None:
#         pors = [por.to(device) for por in pors]

#     # Loss calculation
#     if is_poisoned and pors is not None:
#         # Check poisoned behavior using PORs
#         poisoned_loss = 0
#         for cls, por in enumerate(pors):
#             class_mask = (trg == cls)
#             if class_mask.any():
#                 poisoned_loss += torch.mean((enc_output[class_mask] - por) ** 2)
#         loss = poisoned_loss
#     else:
#         # Loss calculation for normal validation
#         loss = criterion(enc_output, trg)

#     # Predictions and metrics calculation
#     preds = torch.argmax(enc_output, dim=1)  # Predicted class indices
#     correct = (preds == trg).sum().item()
#     total = trg.size(0)
#     accuracy = correct / total

#     # Convert predictions and true labels to CPU for sklearn metrics
#     preds_cpu = preds.cpu().numpy()
#     trg_cpu = trg.cpu().numpy()

#     # Compute precision, recall, and F1 score
#     precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#     recall = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#     f1 = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

#     return loss.item(), accuracy, precision, recall, f1




