import cv2
import numpy as np
import os

# -------- PATH CONFIGURATION --------
INPUT_DIR = "/home/pi/captured_images"
OUTPUT_DIR = "/home/pi/vari_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- VARI CALCULATION --------
def compute_vari(img):
    img = img.astype(float)
    B, G, R = cv2.split(img)

    denominator = (G + R - B)
    denominator[denominator == 0] = 1e-6

    vari = (G - R) / denominator
    return vari

# -------- PROCESS IMAGES --------
for file in os.listdir(INPUT_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(INPUT_DIR, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        vari = compute_vari(img)

        # Normalize VARI to 0–255
        vari_norm = cv2.normalize(vari, None, 0, 255, cv2.NORM_MINMAX)
        vari_uint8 = vari_norm.astype(np.uint8)

        # Apply color gradient
        vari_colored = cv2.applyColorMap(vari_uint8, cv2.COLORMAP_JET)

        out_path = os.path.join(OUTPUT_DIR, f"VARI_{file}")
        cv2.imwrite(out_path, vari_colored)

print("VARI image extraction complete.")
