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

    # Resample to 16kHz
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Normalize
    audio = librosa.util.normalize(audio)

    sf.write(output_path, audio, sr)


# =========================
# 2. AUGMENTATION
# =========================
def add_noise(input_path, output_path, noise_level=0.001):
    audio, sr = librosa.load(input_path, sr=None)
    noise = np.random.randn(len(audio))
    noisy = audio + noise_level * noise
    sf.write(output_path, noisy, sr)


def pitch_shift(input_path, output_path, n_steps=0.5):
    audio, sr = librosa.load(input_path, sr=None)
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    sf.write(output_path, shifted, sr)


# =========================
# 3. RESAMPLE OUTPUT FOR EVAL
# =========================
def resample_to_16k(input_path, output_path):
    audio, sr = librosa.load(input_path, sr=None)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sf.write(output_path, audio, 16000)


# =========================
# 4. INPUT FILES
# =========================
raw_files = ["long_record1.wav", "long_record2.wav"]

reference_audio = []
os.makedirs("processed_audio", exist_ok=True)

print("Processing and augmenting audio...")

# =========================
# 5. PREPROCESS + AUGMENT
# =========================
for i, file in enumerate(raw_files):
    clean_path = f"processed_audio/clean_{i}.wav"
    noise_path = f"processed_audio/noise_{i}.wav"
    pitch_path = f"processed_audio/pitch_{i}.wav"

    # Preprocess
    preprocess_audio(file, clean_path)

    # Augment
    add_noise(clean_path, noise_path)
    pitch_shift(clean_path, pitch_path)

    # Collect all
    reference_audio.extend([clean_path, noise_path, pitch_path])

print("All audio ready!")

# =========================
# 6. LOAD XTTS MODEL
# =========================
print("Loading XTTS model...")

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False
)

print("Model loaded!")

# =========================
# 7. GENERATE SPEECH
# =========================
print("Generating speech...")

output_file = "output.wav"

tts.tts_to_file(
    text="Hello, I am your personal assistant. I am here to help you with your tasks, answer your questions, and provide useful information.",
    speaker_wav=reference_audio,
    language="en",
    file_path=output_file,
)

print("Generated:", output_file)

# =========================
# 8. RESAMPLE OUTPUT FOR EVALUATION
# =========================
resampled_output = "output_16k.wav"
resample_to_16k(output_file, resampled_output)

# =========================
# 9. LOAD SPEAKER MODEL
# =========================
print("Loading speaker verification model...")

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model"
)

print("Model loaded!")

# =========================
# 10. SIMILARITY EVALUATION
# =========================
print("\nCalculating similarity scores...")

scores = []

# Compare against original raw files
for file in raw_files:
    score, _ = model.verify_files(file, resampled_output)
    val = score.item()
    scores.append(val)
    print(f"{file} vs output: {val:.4f}")

avg_score = sum(scores) / len(scores)

print("\n=========================")
print(f"Average Similarity Score: {avg_score:.4f}")
print("=========================")