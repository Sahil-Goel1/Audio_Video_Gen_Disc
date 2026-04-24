import os, shutil, random
from pathlib import Path

# ── Config ────────────────────────────────────────────────
SRC_DIR   = "image_data"          # output of organize_images.py
OUT_DIR   = "dataset"             # train / val / test will be created here
TRAIN     = 0.70
VAL       = 0.15
TEST      = 0.15
SEED      = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
# ─────────────────────────────────────────────────────────

random.seed(SEED)

for split in ["train", "val", "test"]:
    for cls in ["real", "fake"]:
        Path(f"{OUT_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

for cls in ["real", "fake"]:
    images = [f for f in Path(f"{SRC_DIR}/{cls}").iterdir()
              if f.suffix.lower() in IMAGE_EXTS]
    random.shuffle(images)

    n      = len(images)
    n_tr   = int(n * TRAIN)
    n_val  = int(n * VAL)

    splits = {
        "train": images[:n_tr],
        "val"  : images[n_tr : n_tr + n_val],
        "test" : images[n_tr + n_val:],
    }

    for split, files in splits.items():
        for f in files:
            shutil.copy2(f, f"{OUT_DIR}/{split}/{cls}/{f.name}")
        print(f"[{cls}] {split}: {len(files)} images")

print("\n✅ Split complete. Folder: dataset/")
