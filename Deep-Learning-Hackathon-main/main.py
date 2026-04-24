import os
import torch
import numpy as np
import soundfile as sf
import torchaudio

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast          # Mixed precision
import torch.multiprocessing as mp


# ─────────────────────────────────────────────
# GPU CONFIG — sabse pehle set karo
# ─────────────────────────────────────────────
def setup_gpu():
    if not torch.cuda.is_available():
        print("⚠️  CUDA not found — CPU pe chalega")
        return torch.device("cpu"), False

    device = torch.device("cuda")

    # TF32 — A100/RTX30xx+ pe free speedup
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32        = True

    # cuDNN auto-tuner — fixed input size ke liye best algo dhundh leta hai
    torch.backends.cudnn.benchmark         = True
    torch.backends.cudnn.deterministic     = False   # benchmark ke saath off karo

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU  : {gpu_name}")
    print(f"✅ VRAM : {vram_gb:.1f} GB")

    # Batch size suggestion based on VRAM
    if vram_gb >= 40:
        suggested_bs = 64
    elif vram_gb >= 20:
        suggested_bs = 32
    elif vram_gb >= 10:
        suggested_bs = 16
    else:
        suggested_bs = 8
    print(f"💡 Suggested batch_size for your GPU: {suggested_bs}")

    return device, True


# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────
class ASVSpoofDataset(Dataset):
    def __init__(self, file_list, max_length=64000):
        self.file_list  = file_list
        self.max_length = max_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]

        try:
            waveform, sr = sf.read(path)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.file_list))

        if waveform is None or len(waveform) == 0:
            return self.__getitem__((idx + 1) % len(self.file_list))

        # Stereo → mono
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        waveform = waveform.astype(np.float32)

        # Normalize
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / (max_val + 1e-9)

        # Resample to 16kHz if needed
        if sr != 16000:
            waveform_t = torch.tensor(waveform).unsqueeze(0)
            waveform_t = torchaudio.transforms.Resample(sr, 16000)(waveform_t)
            waveform   = waveform_t.squeeze(0).numpy()

        # Pad or truncate to fixed length
        if len(waveform) < self.max_length:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        else:
            waveform = waveform[:self.max_length]

        return {
            "input_values": waveform,
            "label":        label
        }


# ─────────────────────────────────────────────
# 2. PROCESSOR & COLLATE
# ─────────────────────────────────────────────
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

def collate_fn(batch):
    audio  = [item["input_values"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    inputs = processor(
        audio,
        sampling_rate=16000,
        padding=True,
        truncation=True,
        max_length=64000,
        return_tensors="pt"
    )

    return {
        "input_values":   inputs["input_values"],
        "attention_mask": inputs.get("attention_mask", None),
        "labels":         labels
    }


# ─────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────
class Wav2Vec2SpoofClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

        # Freeze everything
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        # Unfreeze last 6 transformer encoder layers
        total_layers = len(self.wav2vec2.encoder.layers)
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i >= total_layers - 6:
                for param in layer.parameters():
                    param.requires_grad = True

        # Unfreeze layer norm
        for param in self.wav2vec2.encoder.layer_norm.parameters():
            param.requires_grad = True

        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, 2)

    def forward(self, input_values, attention_mask=None):
        outputs       = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state   # (B, T, 1024)
        pooled        = hidden_states.mean(dim=1)   # (B, 1024)
        x             = self.dropout(pooled)
        return self.classifier(x)                   # (B, 2)


# ─────────────────────────────────────────────
# 4. DATA HELPERS
# ─────────────────────────────────────────────
def load_asvspoof(protocol_file, audio_dir):
    file_list = []
    with open(protocol_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts      = line.strip().split()
        file_id    = parts[1]
        label_str  = parts[-1]
        label      = 0 if label_str == "bonafide" else 1
        audio_path = os.path.join(audio_dir, file_id + ".flac")
        file_list.append((audio_path, label))
    return file_list


def build_weighted_sampler(file_list):
    num_bonafide = sum(1 for _, lbl in file_list if lbl == 0)
    num_spoof    = sum(1 for _, lbl in file_list if lbl == 1)
    total        = num_bonafide + num_spoof

    print(f"  Bonafide samples : {num_bonafide}")
    print(f"  Spoof    samples : {num_spoof}")
    print(f"  Imbalance ratio  : 1 : {num_spoof // num_bonafide}")

    w_bonafide     = total / (2.0 * num_bonafide)
    w_spoof        = total / (2.0 * num_spoof)

    sample_weights = torch.tensor([
        w_bonafide if lbl == 0 else w_spoof
        for _, lbl in file_list
    ])
    sampler       = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )
    class_weights = torch.tensor([w_bonafide, w_spoof])
    return sampler, class_weights


# ─────────────────────────────────────────────
# 5. TRAIN / EVAL  (AMP mixed precision)
# ─────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, criterion, scaler, device,
                accumulation_steps=2):
    """
    accumulation_steps: gradient accumulation
    Effective batch = batch_size × accumulation_steps
    e.g. batch=16, accum=2 → effective batch 32, same VRAM as 16
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # ── non_blocking — CPU→GPU transfer async hoti hai ──
        inputs = batch["input_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask   = batch["attention_mask"]
        if mask is not None:
            mask = mask.to(device, non_blocking=True)

        # ── autocast — FP16 forward pass (2× faster, half VRAM) ──
        with autocast():
            logits = model(inputs, attention_mask=mask)
            loss   = criterion(logits, labels)
            loss   = loss / accumulation_steps       # scale for accumulation

        # ── scaler — FP16 gradient scaling ──
        scaler.scale(loss).backward()

        # ── Step only after accumulation_steps batches ──
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask   = batch["attention_mask"]
            if mask is not None:
                mask = mask.to(device, non_blocking=True)

            # autocast eval mein bhi — faster inference
            with autocast():
                logits = model(inputs, attention_mask=mask)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    all_preds  = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    overall_acc   = (all_preds == all_labels).float().mean().item()
    bonafide_mask = (all_labels == 0)
    spoof_mask    = (all_labels == 1)
    bonafide_acc  = (all_preds[bonafide_mask] == 0).float().mean().item() \
                    if bonafide_mask.any() else 0.0
    spoof_acc     = (all_preds[spoof_mask]    == 1).float().mean().item() \
                    if spoof_mask.any()    else 0.0

    return overall_acc, bonafide_acc, spoof_acc


def print_gpu_stats():
    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated()  / 1e9
        reserv = torch.cuda.memory_reserved()   / 1e9
        print(f"  GPU Memory  — Allocated: {alloc:.2f} GB | Reserved: {reserv:.2f} GB")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── multiprocessing — DataLoader workers ke liye ──
    mp.set_start_method("spawn", force=True)

    # ── GPU setup ──────────────────────────────
    device, cuda_available = setup_gpu()

    # ── Paths ──────────────────────────────────
    base_path = "archives/LA/LA"

    train_protocol  = os.path.join(
        base_path,
        "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    )
    dev_protocol    = os.path.join(
        base_path,
        "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    )
    train_audio_dir = os.path.join(base_path, "ASVspoof2019_LA_train/flac")
    dev_audio_dir   = os.path.join(base_path, "ASVspoof2019_LA_dev/flac")

    # ── Hyperparams ────────────────────────────
    BATCH_SIZE         = 16    # apne GPU ke hisaab se badao (16/32/64)
    ACCUMULATION_STEPS = 2     # effective batch = BATCH_SIZE × ACCUMULATION_STEPS
    EPOCHS             = 30
    LR                 = 1e-5
    NUM_WORKERS        = min(8, os.cpu_count())   # CPU cores ke hisaab se

    print(f"\n📋 Config:")
    print(f"  Batch size         : {BATCH_SIZE}")
    print(f"  Accumulation steps : {ACCUMULATION_STEPS}")
    print(f"  Effective batch    : {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"  Num workers        : {NUM_WORKERS}")
    print(f"  Epochs             : {EPOCHS}\n")

    # ── Load file lists ─────────────────────────
    print("Loading file lists...")
    train_files = load_asvspoof(train_protocol, train_audio_dir)
    dev_files   = load_asvspoof(dev_protocol,   dev_audio_dir)

    # ── Sampler + class weights ─────────────────
    print("\nTraining set stats:")
    sampler, class_weights = build_weighted_sampler(train_files)

    # ── Datasets ───────────────────────────────
    train_dataset = ASVSpoofDataset(train_files)
    dev_dataset   = ASVSpoofDataset(dev_files)

    # ── DataLoaders ─────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size      = BATCH_SIZE,
        sampler         = sampler,
        num_workers     = NUM_WORKERS,
        collate_fn      = collate_fn,
        pin_memory      = cuda_available,   # GPU pe fast transfer
        prefetch_factor = 2,                # background mein next batch load
        persistent_workers = True           # worker processes restart nahi hote
    )
    test_loader = DataLoader(
        dev_dataset,
        batch_size         = BATCH_SIZE,
        shuffle            = False,
        num_workers        = NUM_WORKERS,
        collate_fn         = collate_fn,
        pin_memory         = cuda_available,
        prefetch_factor    = 2,
        persistent_workers = True
    )

    # ── Model ───────────────────────────────────
    model = Wav2Vec2SpoofClassifier().to(device)

    # ── torch.compile — PyTorch 2.0+ pe free speedup (10-30%) ──
    # Triton ko Python.h headers chahiye — agar nahi hain to gracefully skip
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, backend="eager")  # eager = no Triton, safe fallback
            print("\n⚡ torch.compile() enabled (eager backend)")
        except Exception as e:
            print(f"\n⚠️  torch.compile() skip — {e}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params : {trainable:,} / {total_p:,}")

    # ── Loss ────────────────────────────────────
    criterion = CrossEntropyLoss(weight=class_weights.to(device))

    # ── Optimizer ───────────────────────────────
    # fused AdamW — GPU pe faster, lekin support check karo pehle
    try:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=0.01, fused=cuda_available
        )
        print("⚡ Fused AdamW enabled")
    except TypeError:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=0.01
        )
        print("⚠️  Fused AdamW not supported — standard AdamW use ho raha hai")

    # ── Scheduler ───────────────────────────────
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # ── AMP Scaler (FP16) ────────────────────────
    scaler = GradScaler(enabled=cuda_available)

    # ── Best model tracking ──────────────────────
    best_acc  = 0.0
    save_path = "wav2vec2_asvspoof_model"
    os.makedirs(save_path, exist_ok=True)

    # ── Training loop ───────────────────────────
    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, ACCUMULATION_STEPS
        )
        overall_acc, bonafide_acc, spoof_acc = evaluate(
            model, test_loader, device
        )
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{EPOCHS}")
        print(f"  Train Loss    : {train_loss:.4f}")
        print(f"  Train Acc     : {train_acc:.4f}")
        print(f"  Test Overall  : {overall_acc:.4f}")
        print(f"  Test Bonafide : {bonafide_acc:.4f}")
        print(f"  Test Spoof    : {spoof_acc:.4f}")
        print(f"  LR            : {scheduler.get_last_lr()[0]:.2e}")
        print_gpu_stats()
        print()

        if overall_acc > best_acc:
            best_acc = overall_acc
            torch.save(
                model.state_dict(),
                os.path.join(save_path, "best_model.bin")
            )
            print(f"  ✅ Best model saved (acc={best_acc:.4f})\n")

    # ── Save final ──────────────────────────────
    torch.save(
        model.state_dict(),
        os.path.join(save_path, "final_model.bin")
    )
    print(f"\nTraining complete. Best test acc: {best_acc:.4f}")
    print(f"Models saved to  : {save_path}/")
