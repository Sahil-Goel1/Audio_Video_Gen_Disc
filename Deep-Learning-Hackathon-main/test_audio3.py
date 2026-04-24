import os
import torch
import numpy as np
import soundfile as sf
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
MODEL_PATH = "wav2vec2-for-norm-finetuned"

model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)

model.eval()


# ─────────────────────────────────────────────
# AUDIO PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_audio(waveform, sr, target_sr=16000):

    # stereo → mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # resample
    if sr != target_sr:
        waveform = torch.tensor(waveform).unsqueeze(0)
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        waveform = waveform.squeeze(0).numpy()

    # normalize amplitude
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-9)

    return waveform


# ─────────────────────────────────────────────
# SINGLE FILE PREDICTION
# ─────────────────────────────────────────────
def predict_audio(file_path):

    waveform, sr = sf.read(file_path, dtype="float32")
    waveform = preprocess_audio(waveform, sr)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

    # fixed 5 sec input
    max_length = 16000 * 5

    if len(waveform) < max_length:
        waveform = np.pad(waveform, (0, max_length - len(waveform)))
    else:
        waveform = waveform[:max_length]

    inputs = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    label_map = model.config.id2label

    return label_map[pred], confidence


# ─────────────────────────────────────────────
# FOLDER PREDICTION (OPTIONAL)
# ─────────────────────────────────────────────
def predict_folder(folder_path):

    real_count = 0
    fake_count = 0

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            label, conf = predict_audio(file_path)

            print(f"{file} → {label} ({conf*100:.2f}%)")

            if label.upper() == "REAL":
                real_count += 1
            else:
                fake_count += 1

    print("\n──────── SUMMARY ────────")
    print("REAL:", real_count)
    print("FAKE:", fake_count)


# ─────────────────────────────────────────────
# RUN HERE
# ─────────────────────────────────────────────

# SINGLE FILE TEST
file_path = "output.wav"
label, conf = predict_audio(file_path)

print("\nFINAL RESULT")
print("Prediction:", label)
print("Confidence:", f"{conf*100:.2f}%")

# FOLDER TEST (UNCOMMENT IF NEEDED)
predict_folder("cloned_audio")
