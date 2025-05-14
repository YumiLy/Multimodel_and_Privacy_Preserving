print("ASR.py has started running.")

import argparse, os, random
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
from evaluate import load as load_metric
from sklearn.metrics import roc_curve
import torchaudio

# ───────────────────────────────  Utils  ────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def freeze(net: nn.Module):
    for p in net.parameters(): p.requires_grad_(False)
    net.eval(); return net

def stat_pool(x: torch.Tensor) -> torch.Tensor:
    """Mean + Std pooling on time axis:  [B,T,D] → [B,2D]"""
    return torch.cat([x.mean(-2), x.std(-2)], dim=-1)

def add_noise(emb: torch.Tensor, sigma: float) -> torch.Tensor:
    """Additive i.i.d Gaussian noise on embeddings."""
    if sigma <= 0: return emb
    return emb + torch.randn_like(emb) * sigma


# ───────────────────────────  Speaker‑ID classifier  ────────────────────
class SIDClassifier(nn.Module):
    def __init__(self, dim: int, n_spk: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(dim, n_spk)
        )
    def forward(self, x): return self.net(x)

# ───────────────────────────  Embedding extraction  ─────────────────────
@torch.no_grad()
# def extract_embeddings(ds, processor, encoder, batch_size: int, device: torch.device):
#     print(f"Starting extract_embeddings, dataset length: {len(ds)}")
#     loader = DataLoader(ds, batch_size=batch_size, collate_fn=lambda x: x)
#     print("DataLoader initialized ✅")
#     all_emb, all_len, texts, spks = [], [], [], []
#     for batch in loader:
#         texts  += [s["text"].lower()   for s in batch]
#         spks   += [s["speaker_id"]     for s in batch]
#         waves   = [s["audio"]["array"] for s in batch]

#         inputs = processor(
#             waves,
#             sampling_rate=16_000,
#             return_tensors="pt",
#             padding=True,
#             return_attention_mask=True
#         ).to(device)

#         hidden = encoder(**inputs).last_hidden_state  # [B, T, H]

#         # Append each utterance separately
#         for i in range(hidden.shape[0]):
#             all_emb.append(hidden[i].cpu())
#             all_len.append(inputs.attention_mask[i].sum().item() if "attention_mask" in inputs else inputs.input_values.shape[1])
#             texts.append(batch[i]["text"].lower())
#             spks.append(batch[i]["speaker_id"])

#     emb  = torch.nn.utils.rnn.pad_sequence(all_emb, batch_first=True)  # [N,L,H]
#     return emb, torch.tensor(all_len), texts, spks

def extract_embeddings(ds, processor, encoder, batch_size: int, device: torch.device):
    print(f"Starting extract_embeddings, dataset length: {len(ds)}")
    all_emb, all_len, texts, spks = [], [], [], []

    for start_idx in range(0, len(ds), batch_size):
        end_idx = min(start_idx + batch_size, len(ds))
        batch = ds[start_idx:end_idx]
        # ✅ Convert to dict of lists
        if isinstance(batch, dict):
            print(f"⚠️ Warning: batch from {start_idx} to {end_idx} is a dict, using as-is.")
            batch_dict = batch
        else:
            batch_dict = batch.to_dict()

        print(f"\n>>> Processing samples {start_idx} to {end_idx - 1}")

        print(f"batch_dict['text'] sample: {batch_dict['text'][:3]}")
        texts_batch = [text.lower() for text in batch_dict["text"]]
        # texts_batch = [''.join(text).lower() for text in batch_dict["text"]]
        spks_batch = batch_dict["speaker_id"]
        waves = [audio["array"] for audio in batch_dict["audio"]]
        print(f"Loaded {len(waves)} audio samples.")

        inputs = processor(
            waves,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        ).to(device)
        print(f"Processor finished for samples {start_idx} to {end_idx - 1}.")

        hidden = encoder(**inputs).last_hidden_state  # [B, T, H]
        print(f"Encoder finished. Hidden shape: {hidden.shape}")

        for i in range(hidden.shape[0]):
            all_emb.append(hidden[i].cpu())
            all_len.append(inputs.attention_mask[i].sum().item() if "attention_mask" in inputs else inputs.input_values.shape[1])
            texts.append(texts_batch[i])
            spks.append(spks_batch[i])

    print("All samples processed, padding sequences ...")
    emb = torch.nn.utils.rnn.pad_sequence(all_emb, batch_first=True)
    print("Padding done ✅")
    return emb, torch.tensor(all_len), texts, spks

# ───────────────────────────  Automatic Speech Recognition  ─────────────
@torch.no_grad()
# 
# class GreedyCTCDecoder(torch.nn.Module):
#     def __init__(self, labels, blank=0):
#         super().__init__()
#         self.labels = labels
#         self.blank = blank

#     def forward(self, emission: torch.Tensor) -> List[str]:
#         """Given a sequence emission over labels, get the best path."""
#         indices = torch.argmax(emission, dim=-1)  # [seq_len]
#         indices = torch.unique_consecutive(indices, dim=-1)
#         indices = [i for i in indices if i != self.blank]
#         joined = "".join([self.labels[i] for i in indices])
#         return joined.replace("|", " ").strip().split()
# @torch.no_grad()

# def run_asr(noisy_emb, lengths, refs, lm_head, processor, device):
#     print(f"[ASR] Starting ASR with {noisy_emb.shape[0]} samples.")
#     logits = lm_head(noisy_emb.to(device)).log_softmax(-1)  # [B,T,V]
#     vocab = [t.lower() for t, _ in sorted(processor.tokenizer.get_vocab().items(), key=lambda x: x[1])]
#     decoder = GreedyCTCDecoder(vocab, blank=processor.tokenizer.pad_token_id)

#     preds = []
#     for i in range(logits.shape[0]):
#         pred = decoder(logits[i].cpu())
#         preds.append(" ".join(pred).lower())

#     wer = load_metric("wer").compute(predictions=preds, references=refs)
#     print(f"[ASR-Greedy]  WER: {wer:.4%}")
#     return wer


# Wav2Vec2Processor + decode
def run_asr(noisy_emb, lengths, refs, lm_head, processor, device):
    print(f"[ASR] Starting ASR with {noisy_emb.shape[0]} samples.")
    logits = lm_head(noisy_emb.to(device)).log_softmax(-1)  # [B,T,V]
    pred_ids = torch.argmax(logits, dim=-1)
    preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
    preds = [p.lower() for p in preds]

    wer = load_metric("wer").compute(predictions=preds, references=refs)
    print(f"[ASR-HF]  WER: {wer:.4%}")
    return wer

#  BEAM search + 4-gram LM

# def run_asr(noisy_emb, lengths, refs, lm_head, processor, device):
#     print(f"[ASR] Starting ASR with {noisy_emb.shape[0]} samples.")
#     """Compute WER with fixed lm_head + 4‑gram LM decoder."""
#     logits = lm_head(noisy_emb.to(device)).log_softmax(-1)  # [B,T,|V|]

#     files  = download_pretrained_files("librispeech-4-gram")
#     tokens = [t.lower() for t,_ in sorted(processor.tokenizer.get_vocab().items(),
#                                           key=lambda x: x[1])]
#     decoder = ctc_decoder(
#         lexicon=files.lexicon, 
#         tokens=tokens, 
#         lm=files.lm,
#         nbest=1, 
#         beam_size=20, 
#         lm_weight=3.23, 
#         word_score=-0.26, 
#         blank_token=processor.tokenizer.pad_token
#     ) #如果 beam_size=10，每一步保留 10 条路径，总路径数量大约在 10^3 ~ 10^4 级别；
# 	#  如果 beam_size=1500，路径数量是 几十万到几百万级别，CPU 压力巨增
    
#     hyps   = decoder(logits.cpu(), lengths)
#     preds  = [" ".join(h[0].words).lower() for h in hyps]

#     wer = load_metric("wer").compute(predictions=preds, references=refs)
#     print(f"[ASR]  WER: {wer:.4%}")
#     return wer

# ───────────────────────────  Speaker Verification  ─────────────────────
def run_sv(utt_emb, speakers):
    utt_emb = torch.nn.functional.normalize(utt_emb, dim=-1)
    pairs, labels = [], []
    for i in range(len(utt_emb)):
        for j in range(i + 1, len(utt_emb)):
            pairs.append(torch.nn.functional.cosine_similarity(
                utt_emb[i], utt_emb[j], dim=-1).item())
            labels.append(int(speakers[i] == speakers[j]))
    
    same_speaker = sum(labels)
    diff_speaker = len(labels) - same_speaker
    print(f"Total pairs: {len(labels)}")
    print(f"Same speaker pairs: {same_speaker}")
    print(f"Different speaker pairs: {diff_speaker}")
    
    fpr, tpr, _ = roc_curve(labels, pairs)
    eer = fpr[np.nanargmin(np.abs(tpr - (1-fpr)))]
    print(f"[SV]   EER: {eer:.4%}")
    return eer



# ───────────────────────────  Speaker Identification  ───────────────────
# def run_si(train_utt, train_spk, test_utt, test_spk, epochs: int = 5, topk: int = 5):
#     uniq = sorted(set(train_spk)); n = len(uniq)
#     spk2id = {s:i for i,s in enumerate(uniq)}
#     y_tr   = torch.tensor([spk2id[s] for s in train_spk])
#     y_te   = torch.tensor([spk2id.get(s, -1) for s in test_spk])  # -1 → unseen

#     clf = SIDClassifier(train_utt.size(-1), n)
#     opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
#     crit= nn.CrossEntropyLoss()

#     for ep in range(epochs):
#         clf.train()
#         logit = clf(train_utt); loss = crit(logit, y_tr)
#         opt.zero_grad(); loss.backward(); opt.step()

#     clf.eval(); 
#     with torch.no_grad():
#         lg = clf(test_utt)
#         top1 = (lg.argmax(-1) == y_te).float().mean().item()
#         topk_acc = (lg.topk(topk, dim=-1).indices == y_te.unsqueeze(-1)).any(-1).float().mean().item()
#     print(f"[SI]   top‑1: {top1:.4%}   top‑{topk}: {topk_acc:.4%}")
#     return top1, topk_acc

# ───────────────────────────  Main  ─────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_sigma", type=float, default=0.0,
                        help="σ of Gaussian noise to add on embeddings")
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--epochs_si",   type=int,   default=5)
    parser.add_argument("--topk",        type=int,   default=5,
                        help="top‑K accuracy for speaker‑ID")
    parser.add_argument("--trim_test",   type=int,   default=200,
                        help="limit test split size for quick demo")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| noise σ =", args.noise_sigma)

    # ── load LibriSpeech split & cast audio to 16 kHz
    DATASET_SCRIPT = "/home/lai/datasets/Librispeech/librispeech.py"

    # tr = load_dataset(DATASET_SCRIPT, "clean", split="train", trust_remote_code=True).cast_column("audio", Audio(16_000))
    # dv = load_dataset(DATASET_SCRIPT, "clean", split="validation", trust_remote_code=True).cast_column("audio", Audio(16_000))
    print("Start loading dataset ...")
    ts = load_dataset(DATASET_SCRIPT, "clean", split="test", trust_remote_code=True).cast_column("audio", Audio(16_000))
    if args.trim_test: ts = ts.select(range(args.trim_test))
    print("Dataset ready:", len(ts))

    # ── Wav2Vec2 encoder + CTC head（两者都冻结）
    print("Dataset loaded, starting processor and model ...")
    proc  = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True)
    w2v   = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True).to(device)
    enc   = freeze(w2v.wav2vec2)     # Transformer encoder
    lm_hd = freeze(w2v.lm_head)      # Linear projection to vocab size
    print("Model loaded.")


    print("Blank token:", proc.tokenizer.pad_token)

    # ── Embedding extraction
    print("Extracting embeddings …")
    # emb_tr, len_tr, txt_tr, spk_tr = extract_embeddings(tr, proc, enc, args.batch_size, device)
    # emb_dv, len_dv, txt_dv, spk_dv = extract_embeddings(dv, proc, enc, args.batch_size, device)
    emb_ts, len_ts, txt_ts, spk_ts = extract_embeddings(ts, proc, enc, args.batch_size, device)

    # ── add 
    # noisy_tr = add_noise(emb_tr, args.noise_sigma)
    # noisy_dv = add_noise(emb_dv, args.noise_sigma)
    # noisy_ts = add_noise(emb_ts, args.noise_sigma)

    # ── 1) ASR  (dev / test 均用噪声后 embedding)
    # run_asr(emb_dv, len_dv, txt_dv, lm_hd, proc, device)
    run_asr(emb_ts, len_ts, txt_ts, lm_hd, proc, device)

    # ── 2) Speaker Verification  (frame‑level)
    utt_ts = emb_ts.mean(1)
    run_sv(utt_ts, spk_ts)

    # ── 3) Speaker Identification  (utterance mean)
    # utt_tr = emb_tr.mean(1); 
    # run_si(utt_tr, spk_tr, utt_ts, spk_ts, args.epochs_si, args.topk)

if __name__ == "__main__":
    main()