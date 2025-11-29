#!/usr/bin/env python3

"""
Autonomous NDVI-Style Image Processor
---------------------------------------------------------
- Start trigger: GPIO 5 → begins capturing RGB frames every 10 seconds
- Stop trigger: GPIO 6 → halts capture and finalizes NDVI processing
- Each image is NDVI-processed using (G - R) / (G + R)
- Output saved with timestamps and logged automatically
"""


import cv2
import numpy as np
import time
import os
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from datetime import datetime

# ========== CONFIGURATION ==========
SAVE_DIR = "/home/pi/captures"   # directory to store images, 2 folders will be created, one with RAW and one with captured images
CAPTURE_INTERVAL = 10                 # seconds between captures
START_PIN = 5                         # flight controller connected trigger to turn on
STOP_PIN = 6                          # flight controller trigger to terminate the process

# ========== SETUP ==========
GPIO.setmode(GPIO.BCM)
GPIO.setup(START_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(STOP_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

os.makedirs(SAVE_DIR, exist_ok=True)
camera = Picamera2()
camera.configure(camera.create_still_configuration(main={"size": (1280, 720)}))

def calculate_pseudo_ndvi(image):
    """
    Since Pi Rev1.3 camera is RGB only (no NIR filter), we simulate NDVI using
    the Red and Blue channels as proxies.
    Formula: NDVI ≈ (R - B) / (R + B)
    This gives a 'vegetation index'-like gradient.
    """
    b, g, r = cv2.split(image.astype(float))
    ndvi = (r - b) / (r + b + 1e-5)
    ndvi_normalized = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX)
    ndvi_colormap = cv2.applyColorMap(ndvi_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    return ndvi_colormap

def capture_and_process():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_path = os.path.join(SAVE_DIR, f"raw_{timestamp}.jpg")
    ndvi_path = os.path.join(SAVE_DIR, f"ndvi_{timestamp}.jpg")

    # Capture image
    camera.start()
    time.sleep(2)  # camera warm-up
    image = camera.capture_array()
    camera.stop()

    # Save raw image
    cv2.imwrite(raw_path, image)

    # Process NDVI-like gradient
    ndvi_image = calculate_pseudo_ndvi(image)
    cv2.imwrite(ndvi_path, ndvi_image)

    print(f"[{timestamp}] Captured and processed NDVI image saved to {ndvi_path}")

# ========== MAIN LOOP ==========
try:
    print("System initialized. Waiting for START trigger on GPIO 5...")
    capturing = False

    while True:
        if GPIO.input(START_PIN) == GPIO.HIGH and not capturing:
            print("START trigger detected. Beginning image capture in intervals of 10 seconds...")
            capturing = True
            time.sleep(1)

        elif GPIO.input(STOP_PIN) == GPIO.HIGH and capturing:
            print("STOP trigger detected. Ending capture process.")
            capturing = False
            time.sleep(1)

        if capturing:
            capture_and_process()
            for _ in range(CAPTURE_INTERVAL * 10):  # check stop pin every 0.1s
                if GPIO.input(STOP_PIN) == GPIO.HIGH:
                    print("STOP trigger detected during interval. Stopping.")
                    capturing = False
                    break
                time.sleep(0.1)

        time.sleep(0.2)

except KeyboardInterrupt:
    print("Process interrupted manually. Cleaning up...")
finally:
    camera.close()
    GPIO.cleanup()
    print("GPIO cleaned up. Program ended safely.")



# completed the code write, @viv check it...
