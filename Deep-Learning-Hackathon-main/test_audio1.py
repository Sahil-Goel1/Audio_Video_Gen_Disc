import torch
import numpy as np
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ── Load model ─────────────────────────────
MODEL_PATH = "wav2vec2-for-norm-finetuned"

model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
print(model.config.id2label)

model.eval()

# ── Prediction function ─────────────────────
def predict_audio(file_path):
    waveform, sr = sf.read(file_path, dtype="float32")

    # Stereo → mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Resample to 16k
    if sr != 16000:
        import torchaudio
        waveform = torch.tensor(waveform).unsqueeze(0)
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.squeeze(0).numpy()

    # Same max_length as training (5 sec)
    max_length = 16000 * 5

    # Pad / truncate SAME as training
    if len(waveform) < max_length:
        waveform = np.pad(waveform, (0, max_length - len(waveform)))
    else:
        waveform = waveform[:max_length]

    # 🔥 EXACT SAME preprocessing as training
    inputs = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    # Prediction
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        print(probs)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    label_map = {0: "REAL", 1: "FAKE"}

    return label_map[pred], confidence


# ── Test ───────────────────────────────────
file_path = "file54.wav_16k.wav_norm.wav_mono.wav_silence.wav"   # apna audio yahan daalo
label, conf = predict_audio(file_path)

print(f"Prediction: {label}")
print(f"Confidence: {conf*100:.2f}%")
