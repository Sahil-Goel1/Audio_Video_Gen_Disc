import torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np

# ── Config ─────────────────────────────────────────────
DATA_DIR  = "dataset"
MODEL_PTH = "best_model.pth"
BATCH     = 32
IMG_SIZE  = 224
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
# ───────────────────────────────────────────────────────

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_ds = datasets.ImageFolder(f"{DATA_DIR}/test", transform=tfm)
test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4)
print(f"Test samples: {len(test_ds)} | Classes: {test_ds.classes}")

# ── Load model ─────────────────────────────────────────
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 1),
)
model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ── Inference ──────────────────────────────────────────
all_labels, all_probs = [], []
with torch.no_grad():
    for x, y in test_dl:
        x = x.to(DEVICE)
        probs = model(x).squeeze(1).sigmoid().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())

all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)
all_preds  = (all_probs > 0.5).astype(int)

print("\n── Classification Report ──")
print(classification_report(all_labels, all_preds,
                             target_names=test_ds.classes))

auc = roc_auc_score(all_labels, all_probs)
print(f"ROC-AUC: {auc:.4f}")

print("\n── Confusion Matrix ──")
cm = confusion_matrix(all_labels, all_preds)
print(f"{'':>10}  {'Pred fake':>10}  {'Pred real':>10}")
for i, row in enumerate(cm):
    print(f"{test_ds.classes[i]:>10}  {row[0]:>10}  {row[1]:>10}")
