import os
import librosa
import soundfile as sf
from TTS.api import TTS

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

    # Save cleaned file
    sf.write(output_path, audio, sr)


# =========================
# 2. PREPROCESS ALL FILES
# =========================
raw_files = ["R1.wav", "R2.wav", "R3.wav", "R4.wav", "R5.wav", "R6.wav", "R7.wav"]

clean_files = []
os.makedirs("clean_audio", exist_ok=True)

print("Preprocessing audio...")

for i, file in enumerate(raw_files):
    clean_path = f"clean_audio/clean_{i}.wav"
    preprocess_audio(file, clean_path)
    clean_files.append(clean_path)

print("Audio preprocessing done!")

# =========================
# 3. LOAD MODEL (ONLY ONCE)
# =========================
print("Loading XTTS model...")

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False
)

print("Model loaded!")

# =========================
# 4. GENERATE SPEECH
# =========================
print("Generating speech...")

tts.tts_to_file(
    text="Hello, I am working on a deep learning project and you are listening to a cloned voice.",
    speaker_wav=clean_files,
    language="en",
    file_path="output.wav"
)

print("Done! Check output.wav")