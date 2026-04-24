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


def patch_torch_load_for_xtts():
    """
    XTTS 0.22.0 expects torch.load(..., weights_only=False) behavior.
    PyTorch 2.6 changed the default to True, which breaks XTTS checkpoints.
    """
    original_torch_load = torch.load

    if getattr(original_torch_load, "_xtts_mark6_patched", False):
        return

    def compatible_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    compatible_torch_load._xtts_mark6_patched = True
    torch.load = compatible_torch_load

# =========================
# CONFIG
# =========================
RAW_FILES = ["Teja60.wav","Teja30.wav"]
PROCESSED_DIR = "processed_audio"
OUTPUT_WAV = "output.wav"
OUTPUT_16K = "output_16k.wav"

XTTS_SR = 22050
EVAL_SR = 16000
TOP_DB = 20

# Accent marker words — segments containing these carry
# strongest Indian English phonetic signal
ACCENT_MARKERS = [
    "the", "this", "that", "three", "very", "water", "better",
    "project", "really", "learning", "experience", "exploring",
    "and", "or", "but", "not", "what", "with", "have", "will",
    "speaking", "clear", "steady", "system", "training", "voice","t", "d", "r", "th", "v", "w",
    "india", "english", "model",
    "data", "system", "audio",
    "training", "learning"
]
##TEXT = "Hello, this is a sample voice recording for training a speech model. I am speaking in a clear and steady tone so that the system can learn my voice patterns effectively. Today is a good day to experiment with artificial intelligence and voice synthesis. I hope this recording captures enough variation in pitch, speed, and pronunciation to be useful for generating high-quality speech."
##TEXT = "Hello, this is a sample voice recording for training a speech model. I am speaking in a clear and steady tone so that the system can learn my voice patterns effectively."
## TEXT = "Hello this is a deep learning project and you are hearing a cloned voice because we are working on a project."
##TEXT = "Today I am speaking in a natural and steady way to include variation in pronunciation and create a useful recording that helps the system generate realistic speech."
TEXT = "Life often feels busy, and it can be easy to rush through the day without noticing small things. But if we slow down for a moment, we can see that everyday life is full of simple details that matter."
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
# WHISPERX FORCED ALIGNMENT CHUNKS
# Finds accent-rich segments using transcript alignment
# Falls back to RMS chunking if WhisperX not installed
# =========================
def get_accent_aligned_chunks(audio_path, n_chunks=8):
    """
    Uses WhisperX to transcribe + align audio to text,
    then extracts segments that contain accent-marker words.
    These segments carry the strongest Indian English phonetic signal.
    Falls back to RMS-based chunking if WhisperX unavailable.
    """
    try:
        import whisperx

        print(f"  [WhisperX] Running forced alignment on {audio_path} ...")

        # load and transcribe
        wx_model = whisperx.load_model(
            "base", device="cpu", compute_type="int8"
        )
        audio_wx = whisperx.load_audio(audio_path)
        result   = wx_model.transcribe(audio_wx, batch_size=4)

        print(f"  [WhisperX] Transcript: {' '.join([s['text'] for s in result['segments']])}")

        # forced alignment — word-level timestamps
        align_model, metadata = whisperx.load_align_model(
            language_code="en", device="cpu"
        )
        aligned = whisperx.align(
            result["segments"], align_model,
            metadata, audio_wx, device="cpu"
        )

        # score each segment by accent-marker word count
        audio_arr, sr = librosa.load(audio_path, sr=XTTS_SR)
        scored_segs   = []

        for seg in aligned["segments"]:
            text_lower = seg["text"].lower()
            score = sum(1 for w in ACCENT_MARKERS if w in text_lower)

            # include all segments, score=0 segments still added as fallback
            start = int(seg["start"] * sr)
            end   = int(seg["end"]   * sr)
            chunk = audio_arr[start:end]

            if len(chunk) > sr * 2.5:   # minimum 2.5 seconds
                scored_segs.append((score, chunk))

        # sort by accent richness — highest scoring first
        scored_segs.sort(key=lambda x: x[0], reverse=True)

        if not scored_segs:
            print("  [WhisperX] No segments found, falling back to RMS chunks")
            return _rms_chunks(audio_path, n_chunks)

        # save top N chunks
        paths = []
        base  = os.path.basename(audio_path).split(".")[0]

        for i, (score, chunk) in enumerate(scored_segs[:n_chunks]):
            out = os.path.join(PROCESSED_DIR, f"{base}_aligned_chunk{i}.wav")
            sf.write(out, chunk, XTTS_SR)
            paths.append(out)
            print(f"  [WhisperX] Chunk {i} saved - accent score: {score}")

        print(f"  [WhisperX] Extracted {len(paths)} accent-rich segments")
        return paths

    except ImportError:
        print("  [WhisperX] Not installed, falling back to RMS chunking")
        print("  -> To enable: pip install whisperx")
        return _rms_chunks(audio_path, n_chunks)

    except Exception as e:
        print(f"  [WhisperX] Failed ({e}), falling back to RMS chunking")
        return _rms_chunks(audio_path, n_chunks)


# =========================
# CHUNK SELECTION (RMS fallback)
# =========================
def _rms_chunks(input_path, n_chunks=5, chunk_sec=12):
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

# Keep original name as alias so nothing else breaks
def best_chunks(input_path, n_chunks=3, chunk_sec=10):
    return _rms_chunks(input_path, n_chunks, chunk_sec)

# =========================
# MERGE
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

    # STEP 1: preprocess + accent-aligned chunks
    for i, f in enumerate(RAW_FILES):
        if not os.path.isfile(f):
            print(f"[ERROR] File not found: {f}")
            return

        clean = os.path.join(PROCESSED_DIR, f"clean_{i}.wav")
        print(f"Processing {f}")
        preprocess_audio(f, clean)

        # WhisperX alignment replaces best_chunks here
        # automatically falls back to RMS if WhisperX not installed
        chunks = get_accent_aligned_chunks(clean, n_chunks=6)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # STEP 2: merge
    merged_ref = merge_chunks(all_chunks)
    reference_audio = [merged_ref]

    print("Merged reference created.")

    # STEP 3: XTTS
    patch_torch_load_for_xtts()
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
        print(f"{f} -> {val:.4f}")

    avg = sum(scores) / len(scores)

    print("\n======================")
    print(f"Average Similarity: {avg:.4f}")
    print("======================")

    print("\n==== DONE ====\n")


if __name__ == "__main__":
    main()
