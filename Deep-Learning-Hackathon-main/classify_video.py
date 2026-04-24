import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image

# ── Config ────────────────────────────────────────────────
VIDEO_PATH  = "generated_long_5.mp4"
MODEL_PTH   = "best_model.pth"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SKIP  = 2
# ─────────────────────────────────────────────────────────


# ───────────────── OLD HUGGINGFACE MODEL (COMMENTED) ─────────────────
"""
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
clf_model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
clf_model.to(DEVICE)
clf_model.eval()
"""
# ────────────────────────────────────────────────────────────────────


# ── Load YOUR trained model ─────────────────────────────────────────
def load_model():
    model = models.efficientnet_b0(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 1),
    )

    model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


# ── Image Transform (same as training) ──────────────────────────────
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Load MTCNN ─────────────────────────────────────────────────────
mtcnn = MTCNN(
    keep_all=True,
    device=DEVICE
)


# ── Face Detection ─────────────────────────────────────────────────
def detect_face(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(frame_rgb)

    if boxes is None:
        return None

    # largest face
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = np.argmax(areas)
    x1, y1, x2, y2 = boxes[idx]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # padding
    pad = int(0.2 * max(x2-x1, y2-y1))
    x1 = max(0, x1-pad)
    y1 = max(0, y1-pad)
    x2 = min(frame.shape[1], x2+pad)
    y2 = min(frame.shape[0], y2+pad)

    return frame_rgb[y1:y2, x1:x2]


# ───────────────── OLD CLASSIFIER (COMMENTED) ─────────────────
"""
def classify_face(face_rgb):
    pil_img = Image.fromarray(face_rgb)

    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = clf_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    prob_real = probs[0][1].item()

    return prob_real
"""
# ───────────────────────────────────────────────────────────


# ── New classifier (YOUR MODEL) ─────────────────────────────
def classify_face(model, face_rgb):
    img = Image.fromarray(face_rgb)
    img = tfm(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img)
        prob = torch.sigmoid(logits).item()

    return prob


# ── Main ───────────────────────────────────────────────────
def process_video(video_path):

    model = load_model()

    print(f"\nDevice : {DEVICE}")
    print(f"Video  : {video_path}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps * FRAME_SKIP))

    results = []
    frame_idx = 0
    second_idx = 0
    no_face_count = 0

    print(f"{'Second':>8}  {'Label':>6}  {'Confidence':>12}")
    print("-" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:

            face = detect_face(frame)

            if face is not None:

                prob = classify_face(model, face)

                label = "real" if prob > 0.5 else "fake"
                conf  = prob if label == "real" else 1 - prob

                results.append({
                    "second": second_idx,
                    "label": label,
                    "confidence": round(conf, 4)
                })

                print(f"{second_idx:>8}  {label:>6}  {conf:>11.2%}")

            else:
                no_face_count += 1

            second_idx += 1

        frame_idx += 1

    cap.release()

    # ── Summary ─────────────────────────────
    labels = [r["label"] for r in results]

    if not labels:
        print("\n❌ No faces detected.")
        return

    counts = Counter(labels)

    real_n = counts.get("real", 0)
    fake_n = counts.get("fake", 0)
    total  = real_n + fake_n

    verdict = "real" if real_n > fake_n else "fake"

    avg_conf = np.mean([r["confidence"] for r in results])

    print("\n" + "="*45)
    print("SUMMARY")
    print("="*45)
    print(f"Frames analyzed : {total}")
    print(f"Real frames     : {real_n}")
    print(f"Fake frames     : {fake_n}")
    print(f"Avg confidence  : {avg_conf:.2%}")
    print("="*45)
    print(f"\n🎬 FINAL VERDICT → {verdict.upper()}")
    print("="*45)

    return {
        "real": real_n,
        "fake": fake_n,
        "verdict": verdict
    }


# ── Run ───────────────────────────────────
if __name__ == "__main__":
    process_video(VIDEO_PATH)
