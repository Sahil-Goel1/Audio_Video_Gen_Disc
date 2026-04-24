import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from scipy.spatial.distance import cosine as scipy_cosine
from sklearn.metrics import roc_curve

# =========================
# CONFIG
# =========================
RAW_FILES = ["long_record1.wav","long_record2.wav"]
PROCESSED_DIR = "processed_audio"
OUTPUT_WAV = "output.wav"
OUTPUT_16K = "output_16k.wav"

XTTS_SR = 22050
EVAL_SR = 16000
TOP_DB = 25

TEXT = "Hello, This is a deep learning project and it's been a really nice experience exploring voice cloning."

# =========================
# PREPROCESS
# =========================
def preprocess_audio(input_path, output_path):
    audio, sr = librosa.load(input_path, sr=None)

    if sr != XTTS_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=XTTS_SR)
        sr = XTTS_SR

    audio, _ = librosa.effects.trim(audio, top_db=TOP_DB)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    sf.write(output_path, audio, sr)
    return output_path

# =========================
# CHUNK SELECTION
# =========================
def best_chunks(input_path, n_chunks=3, chunk_sec=10):
    audio, sr = librosa.load(input_path, sr=XTTS_SR)
    chunk_len = int(chunk_sec * sr)
    step = chunk_len // 2

    segments = []
    for start in range(0, max(1, len(audio) - chunk_len), step):
        chunk = audio[start:start + chunk_len]
        if len(chunk) < sr * 3:
            continue
        rms = float(librosa.feature.rms(y=chunk).mean())
        segments.append((rms, chunk))

    if not segments:
        segments = [(0, audio)]

    segments.sort(key=lambda x: x[0], reverse=True)

    paths = []
    base = os.path.basename(input_path).split(".")[0]

    for i, (_, chunk) in enumerate(segments[:n_chunks]):
        out = os.path.join(PROCESSED_DIR, f"{base}_chunk{i}.wav")
        sf.write(out, chunk, sr)
        paths.append(out)

    return paths

# =========================
# MERGE (CRITICAL FIX)
# =========================
def merge_chunks(chunks, output_path="merged.wav"):
    audios = []
    for c in chunks:
        audio, sr = librosa.load(c, sr=XTTS_SR)
        audios.append(audio)

    merged = np.concatenate(audios)
    sf.write(output_path, merged, XTTS_SR)
    return output_path

# =========================
# RESAMPLE FOR EVAL
# =========================
def resample_to_16k(input_path, output_path):
    audio, sr = librosa.load(input_path, sr=None)
    if sr != EVAL_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=EVAL_SR)
    sf.write(output_path, audio, EVAL_SR)

# =========================
# COSINE SIM
# =========================
def cosine_sim(a, b):
    return float(1.0 - scipy_cosine(a, b))

# =========================
# MAIN
# =========================
def main():
    print("\n==== MARK 5 PIPELINE START ====\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_chunks = []

    # STEP 1: preprocess + chunk
    for i, f in enumerate(RAW_FILES):
        if not os.path.isfile(f):
            print(f"[ERROR] File not found: {f}")
            return

        clean = os.path.join(PROCESSED_DIR, f"clean_{i}.wav")
        print(f"Processing {f}")
        preprocess_audio(f, clean)

        chunks = best_chunks(clean)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # STEP 2: merge (IMPORTANT)
    merged_ref = merge_chunks(all_chunks)
    reference_audio = [merged_ref]

    print("Merged reference created.")

    # STEP 3: XTTS
    from TTS.api import TTS

    print("Loading XTTS...")
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        gpu=False
    )

    print("Generating speech...")
    tts.tts_to_file(
        text=TEXT,
        speaker_wav=reference_audio,
        language="en",
        file_path=OUTPUT_WAV,
        split_sentences=True
    )

    print("Generated:", OUTPUT_WAV)

    # STEP 4: resample output
    resample_to_16k(OUTPUT_WAV, OUTPUT_16K)

    # STEP 5: similarity
    print("\nLoading speaker model...")
    from speechbrain.pretrained import SpeakerRecognition

    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmp_model"
    )

    scores = []

    print("\nSimilarity scores:")
    for f in RAW_FILES:
        score, _ = model.verify_files(f, OUTPUT_16K)
        val = score.item()
        scores.append(val)
        print(f"{f} → {val:.4f}")

    avg = sum(scores) / len(scores)

    print("\n======================")
    print(f"Average Similarity: {avg:.4f}")
    print("======================")

    print("\n==== DONE ====\n")


if __name__ == "__main__":
    main()