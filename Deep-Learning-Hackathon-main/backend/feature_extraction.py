import os
import cv2
import numpy as np
import torch
import torch.nn as nn

# Load edge maps

def load_edge_data(folder_path, img_size=128):
    data = []

    files = sorted(os.listdir(folder_path))

    for file in files:
        if file.endswith("_canny.jpg"):
            try:
                path = os.path.join(folder_path, file)
                
                canny = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                canny = cv2.resize(canny, (img_size, img_size))

                # normalize
                canny = canny / 255.0

                # add channel dimension → (H, W, 1)
                canny = np.expand_dims(canny, axis=-1)

                data.append(canny)

            except:
                continue

    return np.array(data)  # (N, H, W, 1)


import os
import cv2
import numpy as np

# 🔹 Sharpness (main metric)
def laplacian_variance(frame):
    lap = cv2.Laplacian(frame, cv2.CV_64F)
    return lap.var()

# 🔹 Gradient strength (extra info)
def gradient_stats(frame):
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    mag = np.sqrt(sobelx**2 + sobely**2)
    
    return np.mean(mag), np.std(mag)

# 🔹 Main function
def analyze_frame_sharpness(folder_path, img_size=128):
    sharpness_scores = []
    gradient_means = []
    gradient_stds = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".jpg"):
            path = os.path.join(folder_path, file)
            
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            frame = cv2.resize(frame, (img_size, img_size))

            # Sharpness
            sharpness = laplacian_variance(frame)

            # Gradient
            g_mean, g_std = gradient_stats(frame)

            sharpness_scores.append(sharpness)
            gradient_means.append(g_mean)
            gradient_stds.append(g_std)

    # 🔹 Aggregate results
    results = {
        "avg_sharpness": np.mean(sharpness_scores),
        "std_sharpness": np.std(sharpness_scores),

        "avg_gradient": np.mean(gradient_means),
        "std_gradient": np.mean(gradient_stds),
    }

    return results


folder = "raw_frames/fake/video_3"

results = analyze_frame_sharpness(folder)

print(results)