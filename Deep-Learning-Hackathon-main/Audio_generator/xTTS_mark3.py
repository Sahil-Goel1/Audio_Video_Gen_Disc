import os
import librosa
import numpy as np
import soundfile as sf
from TTS.api import TTS
from speechbrain.pretrained import SpeakerRecognition

# =========================
# 1. AUDIO PREPROCESSING
# =========================
def preprocess_audio(input_path, output_path):
    audio, sr = librosa.load(input_path, sr=None)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = librosa.util.normalize(audio)

    sf.write(output_path, audio, sr)


# =========================
# 2. AUGMENTATION
# =========================
def add_noise(input_path, output_path, noise_level=0.003):
    audio, sr = librosa.load(input_path, sr=None)
    noise = np.random.randn(len(audio))
    noisy = audio + noise_level * noise
    sf.write(output_path, noisy, sr)


def pitch_shift(input_path, output_path, n_steps=1):
    audio, sr = librosa.load(input_path, sr=None)
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    sf.write(output_path, shifted, sr)


# =========================
# 3. PREPROCESS + AUGMENT
# =========================
raw_files = ["R1.wav", "R2.wav", "R3.wav", "R4.wav","R5.wav", "R6.wav", "R7.wav"]

reference_audio = []

os.makedirs("processed_audio", exist_ok=True)

print("Processing and augmenting audio...")

for i, file in enumerate(raw_files):
    clean_path = f"processed_audio/clean_{i}.wav"
    noise_path = f"processed_audio/noise_{i}.wav"
    pitch_path = f"processed_audio/pitch_{i}.wav"

    # Clean
    preprocess_audio(file, clean_path)

    # Augment
    add_noise(clean_path, noise_path)
    pitch_shift(clean_path, pitch_path)

    # Add all
    reference_audio.extend([clean_path, noise_path, pitch_path])

print("All audio ready!")

# =========================
# 4. LOAD XTTS MODEL
# =========================
print("Loading XTTS model...")

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False
)

print("Model loaded!")

# =========================
# 5. GENERATE SPEECH
# =========================
print("Generating speech...")

output_file = "output.wav"

tts.tts_to_file(
    text="Hello, I am working on a deep learning project and you are listening to a cloned voice.",
    speaker_wav=reference_audio,
    language="en",
    file_path=output_file
)

print("Generated:", output_file)

# =========================
# 6. SIMILARITY SCORING
# =========================
print("Calculating similarity...")

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model"
)

# Compare with first original file (you can extend this)
score, prediction = model.verify_files(raw_files[0], output_file)

print("Similarity Score:", score)
print("Match Prediction:", prediction)