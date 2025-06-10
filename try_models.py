print("ASR.py has started running.")

print("[DEBUG] Script launched")
import time
time.sleep(2)
print("[DEBUG] Imports done")

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
# class GenderClassifier(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 2)
#         )
    
#     def forward(self, x):
#         return self.net(x)

# ───────────────────────────  Embedding extraction  ─────────────────────
@torch.no_grad()

def extract_embeddings(ds, processor, encoder, batch_size: int, device: torch.device, layer_idx: int = -1):
    print(f"Starting extract_embeddings, dataset length: {len(ds)}")
    print("Batch size:", batch_size)
    print(f"Using layer {layer_idx if layer_idx >= 0 else 'last'} of encoder.")
    all_emb, all_len, texts, spks, genders = [], [], [], [], []

    for start_idx in range(0, len(ds), batch_size):
        end_idx = min(start_idx + batch_size, len(ds))
        batch = ds[start_idx:end_idx]
        if isinstance(batch, dict):
            batch_dict = batch # batch_size = 1
        else:
            batch_dict = batch.to_dict()

        # print(f"\n>>> Processing samples {start_idx} to {end_idx - 1}")

        # print(f"batch_dict['text'] sample: {batch_dict['text'][:3]}")
        texts_batch = [text.lower() for text in batch_dict["text"]]
        spks_batch = batch_dict["speaker_id"]
        waves = [audio["array"] for audio in batch_dict["audio"]]
        print(f"Loaded {len(waves)} audio samples.")

        inputs = processor(
            waves,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True
        ).to(device)
        print(f"Processor finished for samples {start_idx} to {end_idx - 1}.")

        if layer_idx >= 0:
            outputs = encoder(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            hidden = hidden_states[layer_idx]


        else:
            hidden = encoder(**inputs).last_hidden_state  # 默认顶层



        for i in range(hidden.shape[0]):
            all_emb.append(hidden[i].cpu())
            # all_len.append(inputs.attention_mask[i].sum().item() if "attention_mask" in inputs else inputs.input_values.shape[1])
            real_len = inputs.input_values[i].shape[0]  # shape is (seq_len,)
            all_len.append(real_len)
            texts.append(texts_batch[i])
            spks.append(spks_batch[i])
            genders.append(batch_dict["gender"][i])

    print("All samples processed, padding sequences ...")
    emb = torch.nn.utils.rnn.pad_sequence(all_emb, batch_first=True)
    print("Padding done ✅")
    return emb, torch.tensor(all_len), texts, spks, genders

# ───────────────────────────  Automatic Speech Recognition  ─────────────

# model head for Wav2Vec2, nn.Linear

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

# def run_gender_cls(train_utt, train_gender, test_utt, test_gender, epochs: int = 5):
#     y_tr = torch.tensor([0 if g == 'M' else 1 for g in train_gender])
#     y_te = torch.tensor([0 if g == 'M' else 1 for g in test_gender])

#     clf = GenderClassifier(train_utt.size(-1))
#     opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-5)
#     crit= nn.CrossEntropyLoss()

#     print(f"\n[Gender] Starting training for {epochs} epochs...")
#     for ep in range(1, epochs + 1):
#         clf.train()
#         logit = clf(train_utt)
#         loss = crit(logit, y_tr)

#         preds_train = logit.argmax(-1)
#         acc_train = (preds_train == y_tr).float().mean().item()

#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#         print(f"  Epoch {ep:2d}/{epochs} | Train Loss: {loss.item():.4f} | Train Acc: {acc_train:.4%}")

#     clf.eval()
#     with torch.no_grad():
#         logits_test = clf(test_utt)
#         preds_test = logits_test.argmax(-1)
#         acc_test = (preds_test == y_te).float().mean().item()

#     print(f"[Gender] Test Accuracy: {acc_test:.4%}\n")
#     return acc_test

# ───────────────────────────  Main  ─────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str,   default='facebook/wav2vec2-base')
    parser.add_argument("--batch_size_emb",  type=int,   default=8)
    parser.add_argument("--epochs_gender",   type=int,   default=100)
    parser.add_argument("--trim_test",   type=int,   default=200,
                        help="limit test split size for quick demo")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ── load LibriSpeech split & cast audio to 16 kHz
    DATASET_SCRIPT = "/home/lai/datasets/Librispeech/librispeech.py"


    print("Start loading dataset ...")
    # tr = load_dataset(DATASET_SCRIPT, "clean", split="train", trust_remote_code=True).cast_column("audio", Audio(16_000))
    # dv = load_dataset(DATASET_SCRIPT, "clean", split="validation", trust_remote_code=True).cast_column("audio", Audio(16_000))
    
    ts = load_dataset(DATASET_SCRIPT, "clean", split="test", trust_remote_code=True).cast_column("audio", Audio(16_000))
    if args.trim_test: ts = ts.select(range(args.trim_test))
    print("Dataset ready:", len(ts))

    # ── Wav2Vec2 encoder + CTC head
    print("Dataset loaded, starting processor and model ...")
    proc  = Wav2Vec2Processor.from_pretrained(args.model_name)
    w2v   = Wav2Vec2ForCTC.from_pretrained(args.model_name).to(device)
    # w2v   = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True)
    enc   = freeze(w2v.wav2vec2)   
    lm_hd = freeze(w2v.lm_head)      # Linear projection to vocab size

    print("Blank token:", proc.tokenizer.pad_token)

    # ── Embedding extraction
    print("Extracting embeddings …")
    # emb_tr, len_tr, txt_tr, spk_tr, gen_tr = extract_embeddings(tr, proc, enc, args.batch_size_emb, device, layer_idx=-1)
    # emb_dv, len_dv, txt_dv, spk_dv = extract_embeddings(dv, proc, enc, args.batch_size, device)
    emb_ts, len_ts, txt_ts, spk_ts, gen_ts = extract_embeddings(ts, proc, enc, args.batch_size_emb, device, layer_idx=-1)


    run_asr(emb_ts, len_ts, txt_ts, lm_hd, proc, device, batch_size=4)
    print(args.batch_size_emb)


    # train_utt = stat_pool(emb_tr)
    # test_utt  = stat_pool(emb_ts)
    # run_sv(test_utt, spk_ts)


    # run_gender_cls(train_utt, gen_tr, test_utt, gen_ts, epochs=args.epochs_gender)



if __name__ == "__main__":
    main()