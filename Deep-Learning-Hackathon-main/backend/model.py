import os
os.environ["HF_HOME"] = "/DATA/hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/DATA/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/DATA/hf/transformers"

import torch
from diffusers import StableVideoDiffusionPipeline

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
    cache_dir="/DATA/hf"
)

pipe = pipe.to("cuda")


import imageio
import cv2
from PIL import Image

def get_best_frame(video_path):
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle = total // 2  # 👈 better than first frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Video read failed")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (576, 1024))

    return Image.fromarray(frame)
    

import numpy as np
import imageio


def generate_long_video_resume(input_video, output="output.mp4", loops=10, save_dir="chunks"):
    os.makedirs(save_dir, exist_ok=True)

    # find completed chunks
    #existing = sorted([
    #    f for f in os.listdir(save_dir)
    #    if f.startswith("chunk_") and f.endswith(".npy")
    #])

    #start_idx = len(existing)
    # print(start_idx, "frames found and loaded")

    # load last frame if exists
    #last_frame_path = os.path.join(save_dir, "last_frame.png")
    start_idx=0
    if start_idx == 0:
        current_frame = get_best_frame(input_video)
    else:
        current_frame = Image.open(last_frame_path)

    generator = torch.manual_seed(42)

    # Generate only missing chunks
    for i in range(start_idx, loops):
        print(f"Generating chunk {i+1}/{loops}...")

        result = pipe(
            current_frame,
            num_frames=30,
            decode_chunk_size=4,
            height=1024,
            width=576,
            generator=generator
        )

        frames = result.frames[0]

        frames_np = [np.array(f) for f in frames]

        # save chunk
        np.save(os.path.join(save_dir, f"chunk_{i:03d}.npy"), frames_np)

        # save continuation frame
        current_frame = frames[-2]
        # current_frame.save(last_frame_path)

        # print(f"✅ Saved chunk {i+1}")

    # Combine all chunks
    all_frames = []

    for i in range(loops):
        arr = np.load(os.path.join(save_dir, f"chunk_{i:03d}.npy"))
        all_frames.extend(arr)

    imageio.mimsave(output, all_frames,codec="libx264", fps=10)

    print("🎬 Final video saved:", output)
    
'''   
uc_links = [
          "https://drive.google.com/uc?id=1--hqIJ0CdVIB3sFCEyiSpezfSFtOmbnT"
          ]


uc_links = ["/DATA/HACKATHON/Real_Videos/VID_20260418_193256742.mp4"]
 
for i in range(0,1):
    # Enable sequential CPU offloading to save GPU memory
    # This moves model components to CPU when not in use
    pipe = pipe.to("cuda", torch.float16)
    
    generate_long_video_resume(
        uc_links[i],
        output=f"/DATA/HACKATHON/Fake_Videos/final_generation_from_website_upload{i+1}.mp4",
        loops=1,   # 👈 increase = longer video
        save_dir=f"chunks_{i+1}"
    )
    
'''
