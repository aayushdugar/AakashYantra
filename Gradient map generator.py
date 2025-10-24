#!/usr/bin/env python3
"""
NDVI-Style Gradient Map Generator for Raspberry Pi 5
----------------------------------------------------
Fetches an RGB image, computes a simulated NDVI map, generates a colorized gradient,
and saves the processed output to a specified directory.
"""

import cv2
import numpy as np
import os
from datetime import datetime

def generate_ndvi_gradient(input_image_path, output_directory, colormap=cv2.COLORMAP_TURBO):
    """
    Generate an NDVI-style gradient heat map from an RGB image.

    Args:
        input_image_path (str): Path to the input RGB image.
        output_directory (str): Directory to save the NDVI gradient map.
        colormap: OpenCV colormap for NDVI visualization (default: TURBO).
    """

    # --- Step 1: Validate paths and prepare directories ---
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"❌ Image not found: {input_image_path}")
    
    os.makedirs(output_directory, exist_ok=True)

    # --- Step 2: Load the RGB image ---
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError("⚠️ Could not load image. Check format or path.")
    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    # --- Step 3: Convert to RGB (OpenCV loads as BGR) ---
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float)

    # --- Step 4: Extract color channels ---
    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]

    # --- Step 5: Compute NDVI-style index ---
    # NDVI ≈ (G - R) / (G + R)
    ndvi_index = (g - r) / (g + r + 1e-8)  # avoid division by zero

    # --- Step 6: Normalize NDVI values to 0–255 for visualization ---
    ndvi_normalized = cv2.normalize(ndvi_index, None, 0, 255, cv2.NORM_MINMAX)
    ndvi_uint8 = ndvi_normalized.astype(np.uint8)

    # --- Step 7: Generate NDVI gradient map using colormap ---
    ndvi_heatmap = cv2.applyColorMap(ndvi_uint8, colormap)

    # --- Step 8: Save output image ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    output_path = os.path.join(output_directory, f"{base_name}_NDVI_{timestamp}.png")

    cv2.imwrite(output_path, ndvi_heatmap)
    print(f"💾 NDVI gradient map saved to: {output_path}")
    print("✅ Processing complete!")

    return output_path


if __name__ == "__main__":
    # --- Configuration Section ---
    INPUT_IMAGE_PATH = "/home/pi/images/input/flight_image.jpg"  # Change this path
    OUTPUT_DIRECTORY = "/home/pi/images/output"                  # Change this path
    COLORMAP = cv2.COLORMAP_TURBO  # Try also: JET, VIRIDIS, RAINBOW

    try:
        output_file = generate_ndvi_gradient(INPUT_IMAGE_PATH, OUTPUT_DIRECTORY, COLORMAP)
        print(f"\n🌍 NDVI map generated successfully:\n{output_file}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
