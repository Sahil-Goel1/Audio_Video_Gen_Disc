from TTS.api import TTS

print("Loading model...")

#
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

print("Model loaded!")

reference_audio = ["R1.wav", "R2.wav", "R3.wav","R4.wav", "R5.wav", "R6.wav" , "R7.wav"]

print("Generating speech...")

tts.tts_to_file(
    text="Hello, I am working on a deep learning project and you are listening to a cloned voice.",
    speaker_wav=reference_audio,
    language="en",
    file_path="output.wav"
)

print("Done! Check output.wav")