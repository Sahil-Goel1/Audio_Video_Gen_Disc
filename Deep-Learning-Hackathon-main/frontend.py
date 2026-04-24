import streamlit as st
import torch
import soundfile as sf
import numpy as np
import torchaudio

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ==============================
# CONFIG
# ==============================

MODEL_PATH = "wav2vec2-for-norm-finetuned"
MAX_DURATION = 5.0
TARGET_SR = 16000

LABELS = {
    0: "Real Audio",
    1: "Fake Audio"
}

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    return feature_extractor, model, device


feature_extractor, model, device = load_model()

max_length = int(TARGET_SR * MAX_DURATION)

# ==============================
# STREAMLIT UI
# ==============================

st.title("🎙️ Deepfake Audio Detector")
st.write("Upload an audio file to check if it is **Real or Fake**.")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "flac", "mp3"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    # ==============================
    # LOAD AUDIO
    # ==============================

    waveform, sr = sf.read(uploaded_file)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:

        waveform = torch.tensor(waveform,dtype=torch.float32).unsqueeze(0)

        waveform = torchaudio.transforms.Resample(
            sr,
            TARGET_SR
        )(waveform)

        waveform = waveform.squeeze().numpy()

    # Pad / truncate
    if len(waveform) < max_length:
        waveform = np.pad(waveform, (0, max_length - len(waveform)))
    else:
        waveform = waveform[:max_length]

    # ==============================
    # MODEL INFERENCE
    # ==============================

    inputs = feature_extractor(
        waveform,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)

        predicted_class = torch.argmax(probs).item()
        confidence = probs[0][predicted_class].item()

    # ==============================
    # SHOW RESULT
    # ==============================

    st.subheader("Prediction")

    if predicted_class == 0:
        st.success(f"🟢 {LABELS[predicted_class]}")
    else:
        st.error(f"🔴 {LABELS[predicted_class]}")

    st.write(f"Confidence: **{confidence:.2%}**")

