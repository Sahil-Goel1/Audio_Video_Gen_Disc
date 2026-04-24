import cv2
import os

def extract_frames(video_path, output_folder, step=4):
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nProcessing: {video_path}")
    print(f"Original FPS: {fps}")
    print(f"Effective FPS: {fps/step}")

    os.makedirs(output_folder, exist_ok=True)

    i = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, gray)
            saved_count += 1
        
        i += 1

    cap.release()
    print(f"Saved {saved_count} frames → {output_folder}")


def process_all_videos(input_folder, output_base="raw_frames", label="real", step=4):
    video_files = sorted(os.listdir(input_folder))

    save_root = os.path.join(output_base, label)
    os.makedirs(save_root, exist_ok=True)

    video_index = 0

    for file in video_files:
        if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(input_folder, file)

            output_folder = os.path.join(save_root, f"video_{video_index}")

            extract_frames(video_path, output_folder, step=step)

            video_index += 1


# 🔹 Run for real videos
#process_all_videos("Fake_videos", output_base="raw_frames", label="fake", step=4)
