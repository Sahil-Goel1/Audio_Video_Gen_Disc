import torch
import numpy as np
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ── Load model ─────────────────────────────
MODEL_PATH = "wav2vec2-for-norm-finetuned"

model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)

model.eval()

# ── Prediction function ─────────────────────
def predict_audio(file_path):
    waveform, sr = sf.read(file_path, dtype="float32")

    # stereo → mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # resample to 16k
    if sr != 16000:
        import torchaudio
        waveform = torch.tensor(waveform).unsqueeze(0)
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.squeeze(0).numpy()

    # max length (5 sec like training)
    max_length = int(16000 * 5)
    if len(waveform) < max_length:
        waveform = np.pad(waveform, (0, max_length - len(waveform)))
    else:
        waveform = waveform[:max_length]

    # feature extractor
    inputs = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # prediction
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    label_map = {0: "REAL", 1: "FAKE"}

    return label_map[pred], confidence


# ── Test ───────────────────────────────────
file_path = "clone_0003.wav"   # <-- apna audio file yahan daalo
label, conf = predict_audio(file_path)

print(f"Prediction: {label}")
print(f"Confidence: {conf*100:.2f}%")
