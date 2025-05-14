from transformers import Wav2Vec2Model, Wav2Vec2ForCTC
import torch
from datasets import load_dataset
import numpy as np
import random

# Load models
model_base = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model_ctc = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
print(model_base)  # 纯 Wav2Vec2Model（无 CTC 头部）
print(model_ctc)   # 包含 Wav2Vec2Model + CTC 头部

num = random.randint(0, 100)

# Load audio sample
dataset = load_dataset("/home/lai/datasets/Librispeech/librispeech.py", "clean", split="test", trust_remote_code=True)
audio_array = dataset[num]["audio"]["array"]
input_tensor = torch.tensor(audio_array.astype(np.float32)).unsqueeze(0)

# Forward pass
with torch.no_grad():
    hidden_ctc = model_ctc.wav2vec2(input_values=input_tensor).last_hidden_state
    hidden_base = model_base(input_values=input_tensor).last_hidden_state

# Compare hidden states
print(torch.allclose(hidden_base, hidden_ctc, atol=1e-5)) # should print True/False
