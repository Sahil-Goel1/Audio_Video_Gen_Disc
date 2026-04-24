import os
import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)
import evaluate
from dataclasses import dataclass
from typing import Dict, List

# ==============================
# CONFIG
# ==============================

DATA_DIR = "archive1/for-norm/for-norm"
CLONED_DIR = "cloned_audio/archive(1)/generated_audio"

MODEL_CHECKPOINT = "facebook/wav2vec2-base"

BATCH_SIZE = 16
MAX_DURATION = 5.0
EPOCHS = 15
LR = 3e-5

OUTPUT_DIR = "wav2vec2-for-norm-finetuned"

LABEL2ID = {"real":0,"fake":1}
ID2LABEL = {0:"real",1:"fake"}

# ==============================
# LOAD ORIGINAL DATASET
# ==============================

def load_file_list(split_dir):

    file_list=[]

    for label_name,label_id in LABEL2ID.items():

        folder=os.path.join(split_dir,label_name)

        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):

            if fname.lower().endswith((".wav",".flac",".mp3",".ogg")):
                file_list.append((os.path.join(folder,fname),label_id))

    print(f"Loaded {len(file_list)} from {split_dir}")

    return file_list


# ==============================
# CLONED DATASET SPLIT
# ==============================

def split_cloned_dataset(base_dir):

    folders=[

        os.path.join(base_dir,f)
        for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir,f))
    ]

    train=[]
    val=[]
    test=[]

    TRAIN_PER_FOLDER=4500//len(folders)
    VAL_PER_FOLDER=1800//len(folders)
    TEST_PER_FOLDER=900//len(folders)

    for folder in folders:

        wavs=[

            os.path.join(folder,f)
            for f in os.listdir(folder)
            if f.lower().endswith((".wav",".flac",".mp3"))
        ]

        random.shuffle(wavs)

        train_part=wavs[:TRAIN_PER_FOLDER]
        val_part=wavs[TRAIN_PER_FOLDER:TRAIN_PER_FOLDER+VAL_PER_FOLDER]
        test_part=wavs[TRAIN_PER_FOLDER+VAL_PER_FOLDER:
                       TRAIN_PER_FOLDER+VAL_PER_FOLDER+TEST_PER_FOLDER]

        train.extend([(f,1) for f in train_part])
        val.extend([(f,1) for f in val_part])
        test.extend([(f,1) for f in test_part])

    return train,val,test


# ==============================
# DATASET
# ==============================

class FoRDataset(Dataset):

    def __init__(self,file_list,feature_extractor,max_length):

        self.file_list=file_list
        self.feature_extractor=feature_extractor
        self.max_length=max_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):

        path,label=self.file_list[idx]

        try:
            waveform,sr=sf.read(path,dtype="float32")
        except:
            waveform=np.zeros(self.max_length,dtype=np.float32)
            return {"input_values":waveform,"label":label}

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

        return {"input_values":waveform,"label":label}


# ==============================
# COLLATOR
# ==============================

@dataclass
class DataCollator:

    feature_extractor:object
    max_length:int

    def __call__(self,batch:List[Dict])->Dict:

        audio=[item["input_values"] for item in batch]

        labels=torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long
        )

        inputs=self.feature_extractor(

            audio,
            sampling_rate=16000,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        inputs["labels"]=labels

        return inputs


# ==============================
# METRICS
# ==============================

accuracy_metric=evaluate.load("accuracy")

def compute_metrics(eval_pred):

    logits,labels=eval_pred

    predictions=np.argmax(logits,axis=1)

    acc=accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

    preds=np.array(predictions)
    labels=np.array(labels)

    real_acc=(preds[labels==0]==0).mean()
    fake_acc=(preds[labels==1]==1).mean()

    return {
        "accuracy":acc["accuracy"],
        "real_accuracy":float(real_acc),
        "fake_accuracy":float(fake_acc),
    }


# ==============================
# MAIN
# ==============================

if __name__=="__main__":

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    # ORIGINAL DATA
    train_orig=load_file_list(os.path.join(DATA_DIR,"training"))
    val_orig=load_file_list(os.path.join(DATA_DIR,"validation"))
    test_orig=load_file_list(os.path.join(DATA_DIR,"testing"))

    # CLONED SPLIT (NO OVERLAP)
    cloned_train,cloned_val,cloned_test=split_cloned_dataset(CLONED_DIR)

    # SPLIT ORIGINAL DATA
    train_real=[f for f in train_orig if f[1]==0]
    train_fake=[f for f in train_orig if f[1]==1]

    val_real=[f for f in val_orig if f[1]==0]
    val_fake=[f for f in val_orig if f[1]==1]

    test_real=[f for f in test_orig if f[1]==0]
    test_fake=[f for f in test_orig if f[1]==1]

    # BUILD FINAL SPLITS

    train_files=(
        random.sample(train_real,5000)+
        random.sample(train_fake,500)+
        cloned_train
    )

    val_files=(
        random.sample(val_real,2000)+
        random.sample(val_fake,200)+
        cloned_val
    )

    test_files=(
        random.sample(test_real,1000)+
        random.sample(test_fake,100)+
        cloned_test
    )

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    print("Train:",len(train_files))
    print("Val:",len(val_files))
    print("Test:",len(test_files))

    # FEATURE EXTRACTOR

    feature_extractor=AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

    max_length=int(feature_extractor.sampling_rate*MAX_DURATION)

    train_dataset=FoRDataset(train_files,feature_extractor,max_length)
    val_dataset=FoRDataset(val_files,feature_extractor,max_length)
    test_dataset=FoRDataset(test_files,feature_extractor,max_length)

    # MODEL

    model=AutoModelForAudioClassification.from_pretrained(

        MODEL_CHECKPOINT,
        num_labels=2,
        label2id={k:str(v) for k,v in LABEL2ID.items()},
        id2label={str(k):v for k,v in ID2LABEL.items()},
        ignore_mismatched_sizes=True,
    )

    # FREEZE MOST LAYERS

    for param in model.wav2vec2.parameters():
        param.requires_grad=False

    total_layers=len(model.wav2vec2.encoder.layers)

    for i,layer in enumerate(model.wav2vec2.encoder.layers):

        if i>=total_layers-5:

            for param in layer.parameters():
                param.requires_grad=True

    for param in model.wav2vec2.encoder.layer_norm.parameters():
        param.requires_grad=True

    # TRAINING

    args=TrainingArguments(

        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        dataloader_num_workers=8,
        report_to="none",
    )

    collator=DataCollator(
        feature_extractor=feature_extractor,
        max_length=max_length
    )

    trainer=Trainer(

        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...\n")

    trainer.train()

    print("\nEvaluating on test set...")

    results=trainer.evaluate(test_dataset)

    print("\nTest Results")

    print("Accuracy:",results["eval_accuracy"])
    print("Real Accuracy:",results["eval_real_accuracy"])
    print("Fake Accuracy:",results["eval_fake_accuracy"])

    trainer.save_model(OUTPUT_DIR)
    feature_extractor.save_pretrained(OUTPUT_DIR)

    print("\nModel saved to",OUTPUT_DIR)
