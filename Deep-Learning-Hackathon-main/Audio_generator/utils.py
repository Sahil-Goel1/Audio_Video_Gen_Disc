import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.spatial.distance import cosine
from speechbrain.pretrained import SpeakerRecognition
from TTS.api import TTS
import matplotlib.pyplot as plt

# Load models once
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
spk_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model"
)

# -------------------------
# Preprocess
# -------------------------
def preprocess_audio(input_path, output_path):
    audio, sr = librosa.load(input_path, sr=None)
    audio, _ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)
    sf.write(output_path, audio, 22050)
    return output_path

# -------------------------
# Generate voice
# -------------------------
def generate_voice(ref_audio, text):
    output_file = "output.wav"
    tts_model.tts_to_file(
        text=text,
        speaker_wav=[ref_audio],
        language="en",
        file_path=output_file
    )
    return output_file

# -------------------------
# Similarity
# -------------------------
def compute_similarity(real, generated):
    score, _ = spk_model.verify_files(real, generated)
    return score.item()

# -------------------------
# Spectrogram
# -------------------------
def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Spectrogram")
    return fig