import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

# -----------------------------
# 🔹 Feature Functions
# -----------------------------
def laplacian_variance(frame):
    return cv2.Laplacian(frame, cv2.CV_64F).var()

def gradient_stats(frame):
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    mag = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(mag), np.std(mag)

# -----------------------------
# 🔹 Extract Features per Video
# -----------------------------
def extract_video_features(video_folder):
    sharpness = []
    gradients = []

    for file in sorted(os.listdir(video_folder)):
        if file.endswith(".jpg"):
            path = os.path.join(video_folder, file)
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if frame is None:
                continue

            sharpness.append(laplacian_variance(frame))

            g_mean, _ = gradient_stats(frame)
            gradients.append(g_mean)

    # ❗ Safety check (important)
    if len(sharpness) < 2:
        return [0, 0, 0, 0, 0]

    sharpness = np.array(sharpness)
    gradients = np.array(gradients)

    features = [
        sharpness.mean(),
        sharpness.std(),
        gradients.mean(),
        gradients.std(),
        np.mean(np.abs(np.diff(sharpness)))  # temporal variation
    ]

    return features

avg_std_real = 12.989171742172147
# stds_real = [0 for i in range(19)]

# for i in range(19):
#     folder_path = f"raw_frames/real/video_{i}"
#     stats = extract_video_features(folder_path)[1]
#     # avg_std_real+=stats
#     stds_real[i] = stats

# # avg_std_real = avg_std_real/19
# # print(avg_std_real)
# print(stds_real)

avg_std_fake = 42.38298429210211
# stds_fake = [0 for i in range(19)]
# for i in range(19):
#     folder_path = f"raw_frames/fake/video_{i}"
#     stats = extract_video_features(folder_path)[1]
#     # avg_std_fake+=stats
#     stds_fake[i] = stats

# # avg_std_fake = avg_std_fake/19
# # print(avg_std_fake)
# print(stds_fake)

# plt.plot([i for i in range(19)], stds_real, label='Real')
# plt.plot([i for i in range(19)], stds_fake, label='Fake')

# plt.xlabel('Index')
# plt.ylabel('Standard Deviation')
# plt.title('Real vs Fake Data')
# plt.legend()

# plt.savefig('plot.png', dpi=300, bbox_inches='tight')  # save the figure
# plt.show()

# preds_real = [0 for i in range(19)]

# for i in range(19):
#     folder_path = f"raw_frames/real/video_{i}"
#     stats = extract_video_features(folder_path)[1]
#     if np.abs(stats-avg_std_real)<np.abs(stats-avg_std_fake):
#         preds_real[i] = 1

# print(preds_real)


# preds_fake = [0 for i in range(19)]

# for i in range(19):
#     folder_path = f"raw_frames/fake/video_{i}"
#     stats = extract_video_features(folder_path)[1]
#     if np.abs(stats-avg_std_real)>np.abs(stats-avg_std_fake):
#         preds_fake[i] = 1

# print(preds_fake)


