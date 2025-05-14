print("ASR.py has started running.")

import argparse, os, random
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
from evaluate import load as load_metric
from sklearn.metrics import roc_curve

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

# def add_noise(emb: torch.Tensor, sigma: float) -> torch.Tensor:
#     """Additive i.i.d Gaussian noise on embeddings."""
#     if sigma <= 0: return emb
#     return emb + torch.randn_like(emb) * sigma


# ───────────────────────────  Speaker‑ID classifier  ────────────────────
class GenderClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.net(x)

# ───────────────────────────  Embedding extraction  ─────────────────────
@torch.no_grad()
def extract_embeddings(ds, processor, encoder, batch_size: int, device: torch.device, layer_idx: int = -1):  
    all_emb, all_len, texts, spks, genders = [], [], [], [], []
    
    for start_idx in range(0, len(ds), batch_size):

        end_idx = min(start_idx + batch_size, len(ds))
        batch = ds[start_idx:end_idx]
        if isinstance(batch, dict):
            batch = batch # batch_size = 1
        else:
            batch = batch.to_dict()
        
        waves = [np.asarray(audio["array"]) for audio in batch["audio"]]  
        inputs = processor(waves, return_tensors="pt", sampling_rate=16_000, padding=True).to(device)
        
        with torch.inference_mode():
            if layer_idx >= 0:
                outputs = encoder(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_idx]
            else:
                hidden = encoder(**inputs).last_hidden_state
         
        valid_lengths = (inputs.input_values != processor.tokenizer.pad_token_id).sum(dim=1)
        all_len.extend(valid_lengths.cpu().tolist())
        
        all_emb.extend([h[:l].cpu() for h, l in zip(hidden, valid_lengths)])
        texts.extend([t.lower() for t in batch["text"]])
        spks.extend(batch["speaker_id"])
        genders.extend(batch["gender"])
    
    return torch.nn.utils.rnn.pad_sequence(all_emb, batch_first=True), torch.tensor(all_len), texts, spks, genders  

# ───────────────────────────  Automatic Speech Recognition  ─────────────
@torch.no_grad()


def run_asr(noisy_emb, lengths, refs, lm_head, processor, device, batch_size: int = 4):
    print(f"[ASR] Starting ASR with {noisy_emb.shape[0]} samples.")
    """Compute WER with fixed lm_head + 4‑gram LM decoder."""

    files  = download_pretrained_files("librispeech-4-gram")
    tokens = [t.lower() for t,_ in sorted(processor.tokenizer.get_vocab().items(),
                                          key=lambda x: x[1])]
    decoder = ctc_decoder(
        lexicon=files.lexicon, 
        tokens=tokens, 
        lm=files.lm,
        nbest=1, 
        beam_size=50, 
        lm_weight=3.23, 
        word_score=-0.26, 
        blank_token=processor.tokenizer.pad_token
    ) 


    preds = []
    total_samples = noisy_emb.shape[0]
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        emb_batch = noisy_emb[start_idx:end_idx].to(device)
        logits = lm_head(emb_batch).log_softmax(-1).cpu()
        
        for i in range(logits.shape[0]):
            valid_len = lengths[start_idx + i]
            pred = decoder(logits[i][:valid_len].unsqueeze(0))
            pred_text = " ".join(pred[0][0].words).lower()
            preds.append(pred_text)
    

    wer = load_metric("wer").compute(predictions=preds, references=refs)
    print(f"[ASR]  WER: {wer:.4%}")
    return wer


# ───────────────────────────  Speaker Verification  ─────────────────────
# REWRITE THIS PART!!!!!!!!!!!

def run_sv(utt_emb, speakers):
    utt_emb = torch.nn.functional.normalize(utt_emb, dim=-1)
    pairs, labels = [], []
    for i in range(len(utt_emb)):
        for j in range(i + 1, len(utt_emb)):
            pairs.append(torch.nn.functional.cosine_similarity(
                utt_emb[i], utt_emb[j], dim=-1).mean().item())
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




# ───────────────────────────  Gender   ─────────────────────────────────────

def run_gender_cls(train_utt, train_gender, test_utt, test_gender, epochs: int = 5):
    y_tr = torch.tensor([0 if g == 'M' else 1 for g in train_gender])
    y_te = torch.tensor([0 if g == 'M' else 1 for g in test_gender])

    clf = GenderClassifier(train_utt.size(-1))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    crit= nn.CrossEntropyLoss()

    print(f"\n[Gender] Starting training for {epochs} epochs...")
    for ep in range(1, epochs + 1):
        clf.train()
        logit = clf(train_utt)
        loss = crit(logit, y_tr)

        preds_train = logit.argmax(-1)
        acc_train = (preds_train == y_tr).float().mean().item()

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"  Epoch {ep:2d}/{epochs} | Train Loss: {loss.item():.4f} | Train Acc: {acc_train:.4%}")

    clf.eval()
    with torch.no_grad():
        logits_test = clf(test_utt)
        preds_test = logits_test.argmax(-1)
        acc_test = (preds_test == y_te).float().mean().item()

    print(f"[Gender] Test Accuracy: {acc_test:.4%}\n")
    return acc_test

# ───────────────────────────  Main  ─────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_sigma", type=float, default=0.0,
                        help="σ of Gaussian noise to add on embeddings")
    parser.add_argument("--batch_size_emb",  type=int,   default=8)
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


    print("Start loading dataset ...")
    tr = load_dataset(DATASET_SCRIPT, "clean", split="train", trust_remote_code=True).cast_column("audio", Audio(16_000))
    # dv = load_dataset(DATASET_SCRIPT, "clean", split="validation", trust_remote_code=True).cast_column("audio", Audio(16_000))
    
    ts = load_dataset(DATASET_SCRIPT, "clean", split="test", trust_remote_code=True).cast_column("audio", Audio(16_000))
    if args.trim_test: ts = ts.select(range(args.trim_test))
    print("Dataset ready:", len(ts))

    # ── Wav2Vec2 encoder + CTC head
    print("Dataset loaded, starting processor and model ...")
    proc  = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True)
    w2v   = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True).to(device)
    # w2v   = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True)
    enc   = freeze(w2v.wav2vec2)     # Transformer encoder
    lm_hd = freeze(w2v.lm_head)      # Linear projection to vocab size

    print("Blank token:", proc.tokenizer.pad_token)

    # ── Embedding extraction
    print("Extracting embeddings …")
    emb_tr, len_tr, txt_tr, spk_tr, gen_tr = extract_embeddings(tr, proc, enc, args.batch_size_emb, device, layer_idx=-1)
    # emb_dv, len_dv, txt_dv, spk_dv = extract_embeddings(dv, proc, enc, args.batch_size, device)
    emb_ts, len_ts, txt_ts, spk_ts, gen_tr = extract_embeddings(ts, proc, enc, args.batch_size_emb, device, layer_idx=-1)


    run_asr(emb_ts, len_ts, txt_ts, lm_hd, proc, device, batch_size=4)
    print(args.batch_size_emb)

    run_sv(emb_ts, spk_ts)

    run_gender_cls(emb_tr, gen_tr, emb_ts, gen_tr, epochs=args.epochs_si)



if __name__ == "__main__":
    main()