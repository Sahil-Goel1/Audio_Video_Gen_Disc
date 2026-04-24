from pydub import AudioSegment

audio = AudioSegment.from_file("long_record1.m4a")
audio.export("long_record1.wav", format="wav")
audio = AudioSegment.from_file("long_record2.m4a")
audio.export("long_record2.wav", format="wav")

print("Converted successfully!")
