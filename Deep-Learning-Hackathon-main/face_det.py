import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image
from collections import Counter

# ================================================
# ✏️ APNI PATHS YAHAN DALO
VIDEO_PATH = "generated_long_5.mp4"
MODEL_PATH = "best_model.pth"
# ================================================

FRAME_SKIP      = 5      # Har 5th frame process hogi
CONFIDENCE      = 0.5    # Detection confidence threshold (kam rakha debug ke liye)
IMG_SIZE        = 224    # ResNet input size
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------- Model Load ----------
def load_model(model_path):
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)  # Binary: sigmoid output
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Handle different save formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded from: {model_path}")
    return model


# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ---------- Main Detection ----------
def detect_deepfake(video_path, model_path):
    print(f"📱 Device : {DEVICE}")

    # Load MTCNN and model
    mtcnn = MTCNN(
        keep_all=False,     # Sirf sabse prominent face lo
        device=DEVICE,
        min_face_size=20,
        post_process=False
    )
    model = load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Video nahi khul rahi: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎬 Video  : {total_frames} frames @ {fps:.1f} FPS\n")

    predictions = []   # 0 = Real, 1 = Fake
    frame_idx   = 0
    processed   = 0
    no_face     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # MTCNN: face detect + crop
            box, prob = mtcnn.detect(pil_img)

            # DEBUG: pehle 5 frames print karo
            if processed < 5:
                print(f"  Frame {frame_idx}: box={box}, prob={prob}")

            if box is not None and prob[0] is not None and prob[0] >= CONFIDENCE:
                # Crop face
                x1, y1, x2, y2 = [int(c) for c in box[0]]
                x1, y1 = max(0, x1), max(0, y1)
                face_crop = pil_img.crop((x1, y1, x2, y2))

                # Model inference
                tensor = transform(face_crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(tensor)
                    prob = torch.sigmoid(output).item()
                    pred = 1 if prob >= 0.5 else 0
                    # 0 = Real, 1 = Fake (prob >= 0.5 means Fake)

                predictions.append(pred)
                processed += 1
            else:
                no_face += 1

        frame_idx += 1

    cap.release()

    if not predictions:
        print("⚠️  Koi bhi frame mein face nahi mila!")
        return None

    # Majority Vote
    vote_counts = Counter(predictions)
    majority    = vote_counts.most_common(1)[0][0]
    label_map   = {0: "REAL ✅", 1: "FAKE ❌"}
    result      = label_map[majority]

    real_count = vote_counts.get(0, 0)
    fake_count = vote_counts.get(1, 0)

    print("=" * 40)
    print(f"  Frames processed  : {processed}")
    print(f"  Frames skipped    : {no_face}  (no face found)")
    print(f"  Real votes        : {real_count}")
    print(f"  Fake votes        : {fake_count}")
    print("=" * 40)
    print(f"  🏆 FINAL RESULT  : {result}")
    print("=" * 40)

    return {"result": result, "real": real_count, "fake": fake_count}


if __name__ == "__main__":
    detect_deepfake(VIDEO_PATH, MODEL_PATH)
