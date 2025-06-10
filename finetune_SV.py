import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor,get_scheduler
from datasets import load_dataset, Audio
from collections import Counter, defaultdict
from sklearn.metrics import roc_curve
from tqdm import tqdm
import numpy as np
import argparse
import random

# ────── Pooling Strategies ──────
def stat_pool(x):
    return torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)

def average_pool(x):
    return x.mean(dim=1)

def max_pool(x):
    return x.max(dim=1).values

def attention_pool(x):
    attn_weights = torch.softmax(x.mean(dim=-1), dim=1)
    return (x * attn_weights.unsqueeze(-1)).sum(dim=1)

POOLING_FN = {
    "stat": stat_pool,
    "mean": average_pool,
    "max": max_pool,
    "attn": attention_pool
}

# ────── AM-Softmax ──────
# class AMSoftmax(nn.Module):
#     def __init__(self, in_feats, num_classes, margin=0.2, scale=30.0):
#         super().__init__()
#         self.margin = margin
#         self.scale = scale
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_feats))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, x, labels):
#         normed_x = F.normalize(x)
#         normed_w = F.normalize(self.weight)
#         cosine = F.linear(normed_x, normed_w)
#         phi = cosine - self.margin
#         one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
#         return self.scale * (one_hot * phi + (1.0 - one_hot) * cosine)


# ────── Wav2Vec2 Model for Embedding ──────
class Wav2Vec2ForSpeakerEmbedding(nn.Module):
    def __init__(self, model_name, freeze_layers=6, pooling="stat"):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
        self.freeze_up_to_layer(freeze_layers)
        hidden = self.encoder.config.hidden_size
        self.pooling_fn = {
            "avg": average_pool,
            "max": max_pool,
            "attn": attention_pool,
            "stat": stat_pool
        }[pooling]
        self.norm = nn.LayerNorm(hidden * 2 if pooling == "stat" else hidden)
        self.am_softmax = None  # init later after speaker set known

    def freeze_up_to_layer(self, freeze_layers):
        for p in self.encoder.feature_extractor.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.encoder.encoder.layers):
            for p in layer.parameters():
                p.requires_grad = i >= freeze_layers

    def forward(self, input_values, attention_mask, labels=None):
        x = self.encoder(input_values, attention_mask=attention_mask).last_hidden_state
        x = self.norm(self.pooling_fn(x))
        return self.am_softmax(x, labels) if labels is not None else x
        # x = self.pooling_fn(x)
        # return self.norm(x)

# ────── Speaker Verification ──────
def compute_similarity(emb1, emb2):
    return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=-1).item()

@torch.no_grad()
def run_sv(model, processor,ds, device):
    # fliter speakers with less than 2 utterances bc we need at least 2 for enrollment and trial
    emb_dict = {}
    print("[INFO] Extracting embeddings...")
    for ex in tqdm(ds):
        inputs = processor(
            ex["audio"]["array"],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)


        
        # AttributeError: 'tuple' object has no attribute 'cpu'
        # emb = model(input_values, attention_mask=attention_mask).cpu()
        embedding = model(input_values, attention_mask=attention_mask)[0]
        embedding = embedding.cpu()
        spk = ex["speaker_id"]
        emb_dict.setdefault(spk, []).append((embedding, ex["id"]))

    # 构建 enrollment 和 trial
    enrollment, trials, scores, labels = [], [], [], []
    for spk, items in emb_dict.items():
        if len(items) < 2:
            continue
        enrollment.append((spk, items[0][0]))
        for i in range(1, len(items)):
            trials.append((spk, items[i][0]))

    for enroll_spk, enroll_emb in enrollment:
        for trial_spk, trial_emb in trials:
            score = compute_similarity(enroll_emb, trial_emb)
            scores.append(score)
            labels.append(int(enroll_spk == trial_spk))

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print(f"[SV] EER: {eer:.2%}")


# ────── Triplet Sampling ──────
#  randomly sample triplets (anchor, positive, negative) from utterances leads bad results
# def sample_triplets(utt_dict, per_speaker=10):
#     speakers = list(utt_dict.keys())
#     triplets = []

#     for anchor_spk in speakers:
#         if len(utt_dict[anchor_spk]) < 2:
#             continue
#         for _ in range(per_speaker):
#             anchor, positive = random.sample(utt_dict[anchor_spk], 2)
#             negative_spk = random.choice([s for s in speakers if s != anchor_spk])
#             if len(utt_dict[negative_spk]) == 0:
#                 continue
#             negative = random.choice(utt_dict[negative_spk])
#             triplets.append((anchor, positive, negative))
#     print(f"[DEBUG] Triplets: {len(triplets)} | Unique Speakers: {len(utt_dict)}")
#     return triplets


@torch.no_grad()
def sample_triplets_semihard(utt_dict, model, processor, device, margin=0.3, max_per_spk=10):
    model.eval()
    triplets = []
    speaker_list = list(utt_dict.keys())

    # Step 1: 提取所有 embedding
    emb_dict = {}
    for spk in speaker_list:
        for ex in utt_dict[spk]:
            inputs = processor(
                ex["audio"]["array"],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            ).to(device)
            emb = model(inputs.input_values, attention_mask=inputs.attention_mask)
            emb_dict[ex["id"]] = (spk, emb.squeeze(0).cpu())

    # Step 2: 构造 semi-hard triplets
    for anchor_spk in speaker_list:
        ex_list = utt_dict[anchor_spk]
        if len(ex_list) < 2:
            continue
        for _ in range(max_per_spk):
            anchor_ex, pos_ex = random.sample(ex_list, 2)
            anchor_emb = emb_dict[anchor_ex["id"]][1]
            pos_emb = emb_dict[pos_ex["id"]][1]
            pos_dist = 1 - torch.nn.functional.cosine_similarity(anchor_emb, pos_emb, dim=0)

            candidates = []
            for neg_spk in speaker_list:
                if neg_spk == anchor_spk:
                    continue
                for neg_ex in utt_dict[neg_spk]:
                    neg_emb = emb_dict[neg_ex["id"]][1]
                    neg_dist = 1 - torch.nn.functional.cosine_similarity(anchor_emb, neg_emb, dim=0)
                    if pos_dist < neg_dist < pos_dist + margin:
                        candidates.append(neg_ex)

            if candidates:
                neg_ex = random.choice(candidates)
            else:
                # fallback to random negative
                neg_spk = random.choice([s for s in speaker_list if s != anchor_spk and utt_dict[s]])
                neg_ex = random.choice(utt_dict[neg_spk])
            triplets.append((anchor_ex, pos_ex, neg_ex))

    print(f"[Semi-Hard Sampling] Triplets: {len(triplets)}")
    return triplets


def preprocess(batch, processor):
        inputs = processor(batch["audio"]["array"], sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding=True)
        batch["input_values"] = inputs.input_values[0]
        return batch

# def collate_fn(batch):
#     input_values = [x["input_values"] for x in batch]
#     input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)

#     labels = torch.tensor([x["label_id"] for x in batch], dtype=torch.long)

#     return {
#         "input_values": input_values,
#         "label_id": labels
#     }

# ────── Training ──────
def train_epoch(model, processor, ds, optimizer, criterion, device, scheduler):
    model.train()
    utt_dict = defaultdict(list)
    for ex in ds:
        utt_dict[ex["speaker_id"]].append(ex)

    triplets = sample_triplets_semihard(utt_dict, model, processor, device)

    total_loss = 0.0

    for anchor_ex, pos_ex, neg_ex in tqdm(triplets, desc="Train"):
        a_inputs = processor(anchor_ex["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        a = a_inputs.input_values.to(device)
        a_mask = a_inputs.attention_mask.to(device) if "attention_mask" in a_inputs else None

        # positive sample
        p_inputs = processor(pos_ex["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        p = p_inputs.input_values.to(device)
        p_mask = p_inputs.attention_mask.to(device) if "attention_mask" in p_inputs else None

        # negative sample
        n_inputs = processor(neg_ex["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        n = n_inputs.input_values.to(device)
        n_mask = n_inputs.attention_mask.to(device) if "attention_mask" in n_inputs else None

        emb_a = model(a, attention_mask=a_mask)
        emb_p = model(p, attention_mask=p_mask)
        emb_n = model(n, attention_mask=n_mask)

        loss = criterion(emb_a, emb_p, emb_n)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    print(f"[Train] Avg Loss: {total_loss/len(triplets):.4f}")

# ────── Main ──────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--data_script", type=str, default="/home/lai/datasets/Librispeech/librispeech.py")
    parser.add_argument("--freeze_layers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pooling", type=str, choices=POOLING_FN.keys(), default="stat")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading dataset...")
    # ds_train = load_dataset(args.data_script, "clean", split="train").cast_column("audio", Audio(16000))
    # ds_dev = load_dataset(args.data_script, "clean", split="validation").cast_column("audio", Audio(16000))
    # ds_test = load_dataset(args.data_script, "clean", split="test").cast_column("audio", Audio(16000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    ds_train = load_dataset(args.data_script, "clean", split="train", trust_remote_code=True).cast_column("audio", Audio(16000))
    ds_dev = load_dataset(args.data_script, "clean", split="validation", trust_remote_code=True).cast_column("audio", Audio(16000))
    ds_test = load_dataset(args.data_script, "clean", split="test", trust_remote_code=True).cast_column("audio", Audio(16000))

    # # Speaker filtering
    # counts = Counter([x["speaker_id"] for x in ds_train])
    # valid_speakers = {s for s, c in counts.items() if c >= 2}
    # ds_train = ds_train.filter(lambda x: x["speaker_id"] in valid_speakers)

    # speakers = sorted(set([x["speaker_id"] for x in ds_train]))
    # id_map = {s: i for i, s in enumerate(speakers)}
    # ds_train = ds_train.map(lambda x: {"label_id": id_map[x["speaker_id"]]})
    # ds_train = ds_train.map(lambda x: preprocess(x, processor))
    # ds_train.set_format(type="torch", columns=["input_values", "label_id"])

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)

    counts = defaultdict(int)
    for ex in ds_train:
        counts[ex["speaker_id"]] += 1
    valid_speakers = {spk for spk, c in counts.items() if c >= 2}
    ds_train = ds_train.filter(lambda ex: ex["speaker_id"] in valid_speakers)


    model = Wav2Vec2ForSpeakerEmbedding(args.model_name, args.freeze_layers, args.pooling).to(device)
    # model.am_softmax = AMSoftmax(model.norm.normalized_shape[0], len(spk2id)).to(device)


    # num_classes = len(id_map)  # 说话人数量（分类数）
    # output_dim = model.norm.normalized_shape[0]  # 获取 norm 输出的维度（hidden or 2*hidden）
    # output_layer = nn.Linear(output_dim, num_classes).to(device)

    criterion = nn.TripletMarginLoss(margin=0.3, p=2)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.01)

    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )

    for epoch in range(args.epochs):
        print(f"=== Epoch {epoch+1}/{args.epochs} ===")
        train_epoch(model, processor, ds_train, optimizer, criterion, device, scheduler)
        print("\nEvaluation on Dev Set")
        run_sv(model, processor, ds_dev, device)

    save_path = f"model_sv_frozen{args.freeze_layers}_{args.pooling}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Saved model to {save_path}")

    print("\nFinal Evaluation on Test Set")
    run_sv(model, processor, ds_test, device)

if __name__ == "__main__":
    main()