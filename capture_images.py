import cv2
import os
import time
from datetime import datetime

# ================= CONFIGURATION =================
CAMERA_INDEX = 0          # Change if needed
CAPTURE_INTERVAL = 10  # seconds
STARTUP_DELAY = 60  # seconds after boot
TOTAL_IMAGES = 100      # change if needed

SAVE_DIR = "/home/ay/captures/ndvi"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
# =================================================

os.makedirs(SAVE_DIR, exist_ok=True)

print("System powered on.")
print("Waiting 60 seconds before starting capture...")
time.sleep(STARTUP_DELAY)

cap = cv2.VideoCapture(CAMERA_INDEX)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("USB Camera not detected.")

print("USB Camera detected. Starting image capture...")

for i in range(TOTAL_IMAGES):
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed, skipping...")
        continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"img_{i}_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    cv2.imwrite(filepath, frame)
    print(f"✅ Captured: {filepath}")

    time.sleep(CAPTURE_INTERVAL)

cap.release()
