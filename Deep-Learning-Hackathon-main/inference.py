import os
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.metrics import roc_curve


# ==============================
# CONFIG
# ==============================

MODEL_PATH = "wav2vec2-for-norm-finetuned"
TEST_DIR = "clones"

MAX_DURATION = 5.0

LABEL2ID = {"real":0,"fake":1}


# ==============================
# LOAD FILES
# ==============================

def load_test_files():

    files=[]

    for label_name,label_id in LABEL2ID.items():

        folder=os.path.join(TEST_DIR,label_name)

        for fname in os.listdir(folder):

            if fname.endswith(".wav"):
                files.append((os.path.join(folder,fname),label_id))

    return files


# ==============================
# DATASET
# ==============================

class AudioDataset(Dataset):

    def __init__(self,file_list,feature_extractor,max_length):

        self.file_list=file_list
        self.feature_extractor=feature_extractor
        self.max_length=max_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):

        path,label=self.file_list[idx]

        waveform,sr=sf.read(path,dtype="float32")

        if waveform.ndim>1:
            waveform=waveform.mean(axis=1)

        if sr!=16000:

            import torchaudio

            wt=torch.tensor(waveform).unsqueeze(0)
            wt=torchaudio.transforms.Resample(sr,16000)(wt)
            waveform=wt.squeeze(0).numpy()

        if len(waveform)<self.max_length:
            waveform=np.pad(waveform,(0,self.max_length-len(waveform)))
        else:
            waveform=waveform[:self.max_length]

        return waveform,label


# ==============================
# EER FUNCTION
# ==============================

def compute_eer(labels,scores):

    fpr,tpr,thresholds = roc_curve(labels,scores)

    fnr = 1 - tpr

    eer_index=np.nanargmin(np.absolute(fnr - fpr))

    eer=fpr[eer_index]

    return eer,fpr,fnr


# ==============================
# MAIN
# ==============================

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:",device)


# LOAD MODEL

feature_extractor=AutoFeatureExtractor.from_pretrained(MODEL_PATH)

model=AutoModelForAudioClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()


max_length=int(feature_extractor.sampling_rate*MAX_DURATION)


# LOAD DATA

test_files=load_test_files()

dataset=AudioDataset(test_files,feature_extractor,max_length)


scores=[]
labels=[]


print("Running inference...")


for waveform,label in dataset:

    inputs=feature_extractor(

        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True

    )

    inputs={k:v.to(device) for k,v in inputs.items()}

    with torch.no_grad():

        logits=model(**inputs).logits

    prob=torch.softmax(logits,dim=1)[0][1].item()

    scores.append(prob)
    labels.append(label)


scores=np.array(scores)
labels=np.array(labels)

preds = (scores >= 0.5).astype(int)

accuracy = (preds == labels).mean()

print("Test Accuracy:", accuracy)


# COMPUTE EER

eer,fpr,fnr=compute_eer(labels,scores)

print("\nEER:",eer)


# ==============================
# PLOT EER CURVE
# ==============================

plt.figure(figsize=(6,6))

plt.plot(fpr,fnr,label="DET Curve")

plt.plot([0,1],[0,1],'--',color='gray')

plt.xlabel("False Acceptance Rate (FAR)")
plt.ylabel("False Rejection Rate (FRR)")

plt.title(f"EER Curve (EER={eer:.4f})")

plt.legend()
plt.grid(True)

plt.savefig("eer_curve.png")

plt.show()
