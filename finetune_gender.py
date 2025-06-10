print("[DEBUG] Script started")

import torch
import torch.nn as nn
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import argparse
import numpy as np
import random
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import get_scheduler

# ────── Utils ──────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ────── Gender Processing ──────
GENDER_MAP = {"M": 0, "F": 1}

def map_gender(example):
    return {"gender_label": GENDER_MAP[example["gender"]]}

def compute_class_weights(dataset, smooth=1e-5):
    label_ids = [int(ex["gender_label"]) for ex in dataset]  # 明确逐个访问样本
    print("[DEBUG] label_ids sample:", label_ids[:5])
    print("[DEBUG] type:", type(label_ids))
    count = Counter(label_ids)
    print(count)
    total = sum(count.values())
    print("[DEBUG] Gender counts:", count)
    weights = [total / (count.get(i, 0) + smooth) for i in range(2)]
    return torch.tensor(weights, dtype=torch.float)

# ────── Pooling methods ──────
# def stat_pool(x):
#     # [B, T, H] → [B, 2H]
#     return torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)

def stat_pool(x, eps=1e-5):
    mean = x.mean(dim=1)
    std = x.std(dim=1)
    std = torch.where(torch.isnan(std), torch.full_like(std, eps), std)
    std = torch.clamp(std, min=eps)
    return torch.cat([mean, std], dim=-1)

# other pooling methods
def max_pool(x):
    # [B, T, H] → [B, H]
    return x.max(dim=1).values

def average_pool(x):
    # [B, T, H] → [B, H]
    return x.mean(dim=1)

def attention_pool(x):
    # [B, T, H] → [B, H] (commonly used in SV)
    attn_weights = torch.softmax(x.mean(dim=-1), dim=1)  # [B, T]
    return (x * attn_weights.unsqueeze(-1)).sum(dim=1)

# ────── Model ──────

class Wav2Vec2ForGenderClassification(nn.Module):
    def __init__(self, model_name, freeze_layers=10, pooling="stat"):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
        self.freeze_up_to_layer(freeze_layers) 
        hidden = self.encoder.config.hidden_size

        self.pooling_fn = {
            "stat": stat_pool,
            "max": max_pool,
            "avg": average_pool,
            "attn": attention_pool
        }[pooling]

        # adjust pooled dimension based on pooling method
        pooled_dim = hidden * 2 if pooling == "stat" else hidden
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
    
    def freeze_up_to_layer(self, freeze_layers):
        for p in self.encoder.feature_extractor.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.encoder.encoder.layers):
            for p in layer.parameters():
                p.requires_grad = i >= freeze_layers
        print(f"[INFO] Frozen: 0 ~ {freeze_layers-1} | Unfrozen: {freeze_layers} ~ {len(self.encoder.encoder.layers)-1}")

    def forward(self, input_values, attention_mask=None):
        hidden = self.encoder(input_values, attention_mask=attention_mask).last_hidden_state
        pooled = self.pooling_fn(hidden)  # [B, 2H]
        return self.head(pooled)

# ────── Data ──────
def prepare_dataset(batch, processor):
    audio_arrays = [a["array"] for a in batch["audio"]]
    sampling_rate = batch["audio"][0]["sampling_rate"]
    inputs = processor(
        audio_arrays,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    batch["input_values"] = inputs.input_values
    batch["attention_mask"] = inputs.attention_mask if "attention_mask" in inputs else torch.ones_like(inputs.input_values, dtype=torch.long)
    return batch

def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["gender_label"] for item in batch]
    return {
        "input_values": torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True),
        "gender_label": torch.tensor(labels, dtype=torch.long),
    }

# ────── Training ──────
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    torch.autograd.set_detect_anomaly(True) # Enable anomaly detection
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for batch in tqdm(dataloader, desc="Training"):
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["gender_label"].to(device)
        print("[DEBUG] labels:", labels)
        print("[DEBUG] unique labels:", labels.unique())

        optimizer.zero_grad()
        
        # # === 自动混合精度（Autocast）开始 ===
        # with torch.amp.autocast(device_type='cuda'):
        #     logits = model(input_values, attention_mask=attention_mask)
        #     loss = criterion(logits, labels)
        # === 自动混合精度结束 ===
        logits = model(input_values, attention_mask=attention_mask)
        loss = criterion(logits, labels)


        # logits = model(input_values, attention_mask=attention_mask)
        print("[DEBUG] logits:", logits)
        print("[DEBUG] logits shape:", logits.shape)

        # loss = criterion(logits, labels)
        print("[DEBUG] input max:", input_values.abs().max().item())
        print("[DEBUG] logits max:", logits.abs().max().item())
        if torch.isnan(logits).any():
            print("[ERROR] NaN in logits")
        if torch.isnan(loss):
            print("[ERROR] NaN in loss")

        # 用 scaler 进行反向传播和优化
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)  # 解除scale，用于clip
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # scaler.step(optimizer)
        # scaler.update()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        scheduler.step()

        
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        # optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += labels.size(0)

    acc = total_correct / total_samples
    avg_loss = total_loss / len(dataloader)
    print(f"[TRAIN] Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
    return avg_loss, acc

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total, correct = 0, 0
    preds, true_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["gender_label"].to(device)

        logits = model(input_values, attention_mask=attention_mask)
        pred = logits.argmax(dim=-1)

        # .tolist() will return a list of ints
        preds += pred.cpu().tolist()
        true_labels += labels.cpu().tolist()

        correct += (pred == labels).sum().item()
        total += labels.size(0)

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="weighted")
    print(f"[EVAL] Accuracy: {acc:.4f} | F1: {f1:.4f}")
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--data_script", type=str, default="/home/lai/datasets/Librispeech/librispeech.py")
    parser.add_argument("--freeze_layers", type=int, default=10)
    parser.add_argument("--pooling", type=str, choices=["stat", "avg", "max", "attn"], default="avg")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    ds_train = load_dataset(args.data_script, "clean", split="train", trust_remote_code=True).cast_column("audio", Audio(16000))
    ds_dev = load_dataset(args.data_script, "clean", split="validation", trust_remote_code=True).cast_column("audio", Audio(16000))
    ds_test = load_dataset(args.data_script, "clean", split="test", trust_remote_code=True).cast_column("audio", Audio(16000))

    ds_train = ds_train.map(map_gender)
    print("[DEBUG] sample example:", ds_train[0])
    ds_dev = ds_dev.map(map_gender)
    ds_test = ds_test.map(map_gender)

    ds_train = ds_train.map(lambda b: prepare_dataset(b, processor), batched=True)
    ds_dev = ds_dev.map(lambda b: prepare_dataset(b, processor), batched=True)
    ds_test = ds_test.map(lambda b: prepare_dataset(b, processor), batched=True)

    ds_train.set_format(type="torch")
    ds_dev.set_format(type="torch")
    ds_test.set_format(type="torch")

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(ds_dev, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Wav2Vec2ForGenderClassification(args.model_name, freeze_layers=args.freeze_layers, pooling=args.pooling).to(device)
    class_weights = compute_class_weights(ds_train).to(device)
    print("[DEBUG] class weights:", class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # scaler = torch.amp.GradScaler()


    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    save_path = f"model_gender_cls_frozen{args.freeze_layers}_{args.pooling}.pt"

    best_acc = 0.0
    print("[INFO] Starting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        acc, _ = evaluate(model, dev_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
    
    print("\nFinal Evaluation on Test Set")
    acc_test, F1_test = evaluate(model, test_loader, device)
    print("Acc_test:", acc_test)
    print("F1_test:", F1_test)
    
if __name__ == "__main__":
    main()
