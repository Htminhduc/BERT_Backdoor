import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("Bhumika/roberta-base-finetuned-sst2")

# Preprocessing
def tokenize(batch):
    return tokenizer(batch['sentence'], truncation=True, padding='max_length', max_length=64)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# train_loader = DataLoader(dataset['train'], batch_size=128, shuffle=True)
val_loader = DataLoader(dataset['validation'], batch_size=32)

# Model
model = AutoModelForSequenceClassification.from_pretrained("Bhumika/roberta-base-finetuned-sst2").to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train_one_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Validation
def validate():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    return acc

# Run training
for epoch in range(1, 4):
    train_loss = train_one_epoch()
    val_acc = validate()
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}")
