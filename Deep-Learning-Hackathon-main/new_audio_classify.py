import os
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)
import evaluate                        # pip install evaluate
from dataclasses import dataclass
from typing import Dict, List

# ─────────────────────────────────────────────
# CONFIG — apna path yahan set karo
# ─────────────────────────────────────────────
DATA_DIR        = "archive1/for-norm/for-norm"   # <-- apna path
MODEL_CHECKPOINT = "facebook/wav2vec2-base"
BATCH_SIZE      = 16
MAX_DURATION    = 5.0    # seconds
EPOCHS          = 15
LR              = 3e-5
OUTPUT_DIR      = "wav2vec2-for-norm-finetuned"

# ─────────────────────────────────────────────
# 1. MANUAL DATASET — audiofolder/torchcodec bypass
# ─────────────────────────────────────────────
LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {0: "real", 1: "fake"}

import random

def split_balanced(files, n):
    real = [f for f in files if f[1] == 0]
    fake = [f for f in files if f[1] == 1]

    real = random.sample(real, n)
    fake = random.sample(fake, n)

    combined = real + fake
    random.shuffle(combined)
    return combined
    
def load_file_list(split_dir):
    """
    split_dir ke andar real/ aur fake/ folders dhundho.
    Returns: list of (filepath, label_int)
    """
    file_list = []
    for label_name, label_id in LABEL2ID.items():
        folder = os.path.join(split_dir, label_name)
        if not os.path.isdir(folder):
            print(f"  ⚠️  Folder nahi mila: {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                file_list.append((os.path.join(folder, fname), label_id))
    print(f"  Loaded {len(file_list)} files from {split_dir}")
    return file_list


class FoRDataset(Dataset):
    def __init__(self, file_list, feature_extractor, max_length):
        self.file_list        = file_list
        self.feature_extractor = feature_extractor
        self.max_length       = max_length   # samples, e.g. 80000

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]

        # soundfile se read karo — FFmpeg ki zarurat nahi
        try:
            waveform, sr = sf.read(path, dtype="float32")
        except Exception as e:
            print(f"  Error reading {path}: {e}")
            waveform = np.zeros(self.max_length, dtype=np.float32)
            return {"input_values": waveform, "label": label}

        # Stereo → mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Resample agar zarurat ho (soundfile se karo agar sr != 16000)
        if sr != 16000:
            import torchaudio
            wt = torch.tensor(waveform).unsqueeze(0)
            wt = torchaudio.transforms.Resample(sr, 16000)(wt)
            waveform = wt.squeeze(0).numpy()

        # Pad ya truncate
        if len(waveform) < self.max_length:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        else:
            waveform = waveform[:self.max_length]

        return {"input_values": waveform, "label": label}


# ─────────────────────────────────────────────
# 2. COLLATE — feature extractor yahan apply karo
# ─────────────────────────────────────────────
@dataclass
class DataCollator:
    feature_extractor: object
    max_length: int

    def __call__(self, batch: List[Dict]) -> Dict:
        audio  = [item["input_values"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs["labels"] = labels
        return inputs


# ─────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)

    # Per-class accuracy bhi dekhte hain
    preds  = np.array(predictions)
    labels = np.array(labels)
    real_acc  = (preds[labels == 0] == 0).mean() if (labels == 0).any() else 0.0
    fake_acc  = (preds[labels == 1] == 1).mean() if (labels == 1).any() else 0.0

    return {
        "accuracy":      acc["accuracy"],
        "real_accuracy": float(real_acc),
        "fake_accuracy": float(fake_acc),
    }


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── GPU check ──────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM  : {vram:.1f} GB")

    # ── File lists ─────────────────────────────
    print("\nLoading file lists...")
    train_files = load_file_list(os.path.join(DATA_DIR, "training"))
    val_files   = load_file_list(os.path.join(DATA_DIR, "validation"))
    test_files  = load_file_list(os.path.join(DATA_DIR, "testing"))
    
    # choose how many per class
    TRAIN_N = 5000
    VAL_N   = 2000
    TEST_N  = 1000

    train_files = split_balanced(train_files, TRAIN_N)
    val_files   = split_balanced(val_files, VAL_N)
    test_files  = split_balanced(test_files, TEST_N)
 
    # ── Feature extractor ──────────────────────
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
    max_length        = int(feature_extractor.sampling_rate * MAX_DURATION)  # 80000

    # ── Datasets ───────────────────────────────
    train_dataset = FoRDataset(train_files, feature_extractor, max_length)
    val_dataset   = FoRDataset(val_files,   feature_extractor, max_length)
    test_dataset  = FoRDataset(test_files,  feature_extractor, max_length)

    print(f"\nTrain : {len(train_dataset)} samples")
    print(f"Val   : {len(val_dataset)} samples")
    print(f"Test  : {len(test_dataset)} samples")

    # ── Model ───────────────────────────────────
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels      = 2,
        label2id        = {k: str(v) for k, v in LABEL2ID.items()},
        id2label        = {str(k): v for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes = True,
    )

    for param in model.wav2vec2.parameters():
        param.requires_grad = False
 
    # Step 2: last 4 encoder layers unfreeze
    total_layers  = len(model.wav2vec2.encoder.layers)   # 12 for base
    unfreeze_from = total_layers - 5                      # layer 9 se start
 
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        if i >= unfreeze_from:
            for param in layer.parameters():
                param.requires_grad = True
 
    # Step 3: encoder layer norm bhi unfreeze
    for param in model.wav2vec2.encoder.layer_norm.parameters():
        param.requires_grad = True
 
    # projector + classifier already trainable hain (wav2vec2 ke bahar hain)
 
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    frozen    = total_p - trainable
    print(f"\nTotal params     : {total_p:,}")
    print(f"Frozen params    : {frozen:,}  ({100*frozen/total_p:.0f}%)")
    print(f"Trainable params : {trainable:,}  ({100*trainable/total_p:.0f}%)")
    print(f"Unfrozen layers  : {unfreeze_from} to {total_layers-1} + classifier head")
    
    # ── Training args ───────────────────────────
    args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        eval_strategy         = "epoch",
        save_strategy               = "epoch",
        learning_rate               = LR,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = 4,          # effective batch = 64
        num_train_epochs            = EPOCHS,
        warmup_ratio                = 0.1,         # NaN prevent karta hai
        logging_steps               = 50,
        load_best_model_at_end      = True,
        metric_for_best_model       = "accuracy",
        greater_is_better           = True,
        save_total_limit            = 2,
        bf16                        = True,
        fp16                        = False,   # FP16 OFF — NaN avoid
        dataloader_num_workers      = 8,
        report_to                   = "none",      # wandb/tensorboard off
    )

    # ── Collator ───────────────────────────────
    collator = DataCollator(
        feature_extractor = feature_extractor,
        max_length        = max_length
    )

    # ── Trainer ────────────────────────────────
    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        data_collator   = collator,
        compute_metrics = compute_metrics,
    )

    # ── Train ───────────────────────────────────
    print("\nStarting training...\n")
    trainer.train()

    # ── Final evaluation on test set ────────────
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_dataset)
    print(f"\nTest Results:")
    print(f"  Overall Accuracy : {results['eval_accuracy']:.4f}")
    print(f"  Real Accuracy    : {results['eval_real_accuracy']:.4f}")
    print(f"  Fake Accuracy    : {results['eval_fake_accuracy']:.4f}")

    # ── Save final model ────────────────────────
    trainer.save_model(OUTPUT_DIR)
    feature_extractor.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to: {OUTPUT_DIR}/")
