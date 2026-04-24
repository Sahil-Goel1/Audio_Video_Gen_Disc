import os, torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Config ────────────────────────────────────────────────
DATA_DIR   = "dataset"
BATCH      = 32
EPOCHS_1   = 5       # Phase 1: frozen backbone
EPOCHS_2   = 10      # Phase 2: full fine-tune
LR_HEAD    = 1e-3
LR_FULL    = 1e-4
IMG_SIZE   = 224
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH  = "best_model.pth"
# ─────────────────────────────────────────────────────────

print(f"Using device: {DEVICE}")

# ── Transforms ───────────────────────────────────────────
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Datasets ─────────────────────────────────────────────
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tfm)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=val_tfm)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                      num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                      num_workers=4, pin_memory=True)

print(f"Classes: {train_ds.classes}")          # should be ['fake', 'real']
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Model ─────────────────────────────────────────────────
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Replace classifier head with binary output
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 1),
)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

# ── Helper: evaluate ──────────────────────────────────────
def evaluate(loader):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            out  = model(x).squeeze(1)
            loss_sum += criterion(out, y).item() * len(y)
            preds = (out.sigmoid() > 0.5).long()
            correct += (preds == y.long()).sum().item()
            total   += len(y)
    return loss_sum / total, correct / total

# ── Helper: train one epoch ───────────────────────────────
def train_epoch(optimizer):
    model.train()
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.float().to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()

# ── Phase 1: freeze backbone, train head only ────────────
print("\n── Phase 1: training classifier head (backbone frozen) ──")
for p in model.features.parameters():
    p.requires_grad = False

opt1  = Adam(model.classifier.parameters(), lr=LR_HEAD)
sched1 = CosineAnnealingLR(opt1, T_max=EPOCHS_1)
best_acc = 0

for ep in range(1, EPOCHS_1 + 1):
    train_epoch(opt1)
    sched1.step()
    vl, va = evaluate(val_dl)
    print(f"  Epoch {ep}/{EPOCHS_1} | val_loss: {vl:.4f} | val_acc: {va:.4f}")
    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), SAVE_PATH)

# ── Phase 2: unfreeze all, fine-tune ─────────────────────
print("\n── Phase 2: full fine-tune (all layers) ──")
for p in model.parameters():
    p.requires_grad = True

opt2   = Adam(model.parameters(), lr=LR_FULL)
sched2 = CosineAnnealingLR(opt2, T_max=EPOCHS_2)

for ep in range(1, EPOCHS_2 + 1):
    train_epoch(opt2)
    sched2.step()
    vl, va = evaluate(val_dl)
    print(f"  Epoch {ep}/{EPOCHS_2} | val_loss: {vl:.4f} | val_acc: {va:.4f}")
    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ Best model saved (val_acc={best_acc:.4f})")

print(f"\n🏁 Training done. Best val_acc: {best_acc:.4f} | Saved: {SAVE_PATH}")
