#!/usr/bin/env python3
"""
NDVI-Style Image Data Extraction for Raspberry Pi 5
Processes RGB images and generates gradient heat maps based on pixel intensity
"""

import cv2
import numpy as np
import os
from datetime import datetime

class NDVIImageProcessor:
    def __init__(self, input_path, output_dir):
        """
        Initialize the NDVI Image Processor
        
        Args:
            input_path (str): Path to input RGB image
            output_dir (str): Directory to save output images
        """
        self.input_path = input_path   #you need to add the input path here
        self.output_dir = output_dir   # accordingly we need to make an ouput path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def load_image(self):
        """Load the RGB image from specified path"""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Image not found at: {self.input_path}")
        
        self.image = cv2.imread(self.input_path)
        if self.image is None:
            raise ValueError(f"Failed to load image from: {self.input_path}")
        
        print(f"Image loaded: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        return self.image
    
    def calculate_pixel_intensities(self):
        """
        Calculate pixel intensity values from RGB image
        Returns normalized intensity map
        """
        # Convert BGR (OpenCV format) to RGB
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Extract individual channels
        r_channel = rgb_image[:, :, 0].astype(float)
        g_channel = rgb_image[:, :, 1].astype(float)
        b_channel = rgb_image[:, :, 2].astype(float)
        
        # Calculate intensity using standard luminosity formula
        # This mimics how NDVI uses band math
        intensity = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
        
        # Alternative: For vegetation analysis, you might want to emphasize
        # the difference between red and green (similar to NDVI concept)
        # Uncomment below for vegetation-focused intensity:
        # intensity = (g_channel - r_channel) / (g_channel + r_channel + 1e-8)
        
        # Normalize to 0-1 range
        self.intensity_map = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        
        print(f"Intensity range: {intensity.min():.2f} to {intensity.max():.2f}")
        print(f"Normalized intensity range: {self.intensity_map.min():.2f} to {self.intensity_map.max():.2f}")
        
        return self.intensity_map
    
    def create_gradient_heatmap(self, colormap=cv2.COLORMAP_JET):
        """
        Create NDVI-style gradient heat map from intensity values
        
        Args:
            colormap: OpenCV colormap (default: COLORMAP_JET for NDVI-like appearance)
                     Options: COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT, 
                              COLORMAP_RAINBOW, COLORMAP_TURBO
        """
        # Convert normalized intensity to 8-bit image
        intensity_8bit = (self.intensity_map * 255).astype(np.uint8)
        
        # Apply colormap to create heat map
        self.heatmap = cv2.applyColorMap(intensity_8bit, colormap)
        
        print(f"Heat map generated with colormap: {colormap}")
        return self.heatmap
    
    def create_overlay(self, alpha=0.5):
        """
        Create an overlay of the heat map on original image
        
        Args:
            alpha (float): Transparency of overlay (0.0 to 1.0)
        """
        # Resize heatmap to match original image if needed
        if self.heatmap.shape != self.image.shape:
            self.heatmap = cv2.resize(self.heatmap, 
                                      (self.image.shape[1], self.image.shape[0]))
        
        # Blend original image with heat map
        self.overlay = cv2.addWeighted(self.image, 1-alpha, self.heatmap, alpha, 0)
        
        return self.overlay
    
    def save_results(self, save_overlay=True):
        """
        Save processed images to output directory
        
        Args:
            save_overlay (bool): Whether to save overlay image
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
        
        # Save intensity map
        intensity_path = os.path.join(self.output_dir, 
                                      f"{base_filename}_intensity_{timestamp}.png")
        intensity_8bit = (self.intensity_map * 255).astype(np.uint8)
        cv2.imwrite(intensity_path, intensity_8bit)
        print(f"Intensity map saved: {intensity_path}")
        
        # Save heat map
        heatmap_path = os.path.join(self.output_dir, 
                                    f"{base_filename}_heatmap_{timestamp}.png")
        cv2.imwrite(heatmap_path, self.heatmap)
        print(f"Heat map saved: {heatmap_path}")
        
        # Save overlay if requested
        if save_overlay and hasattr(self, 'overlay'):
            overlay_path = os.path.join(self.output_dir, 
                                       f"{base_filename}_overlay_{timestamp}.png")
            cv2.imwrite(overlay_path, self.overlay)
            print(f"Overlay saved: {overlay_path}")
        
        return {
            'intensity': intensity_path,
            'heatmap': heatmap_path,
            'overlay': overlay_path if save_overlay else None
        }
    
    def process(self, colormap=cv2.COLORMAP_JET, create_overlay_img=True):
        """
        Complete processing pipeline
        
        Args:
            colormap: OpenCV colormap for heat map
            create_overlay_img: Whether to create overlay image
        """
        print("Starting NDVI-style image processing...")
        print("-" * 50)
        
        # Load image
        self.load_image()
        
        # Calculate intensities
        self.calculate_pixel_intensities()
        
        # Create heat map
        self.create_gradient_heatmap(colormap)
        
        # Create overlay
        if create_overlay_img:
            self.create_overlay(alpha=0.6)
        
        # Save results
        saved_files = self.save_results(save_overlay=create_overlay_img)
        
        print("-" * 50)
        print("Processing complete!")
        
        return saved_files


def main():
    """
    Main execution function
    Configure your paths here
    """
    # Configuration
    INPUT_IMAGE_PATH = "/home/pi/images/input/flight_image.jpg"  # Change this path
    OUTPUT_DIRECTORY = "/home/pi/images/output"  # Change this path
    
    # Available colormaps for different visualizations:
    # cv2.COLORMAP_JET - Classic NDVI-like (blue to red)
    # cv2.COLORMAP_VIRIDIS - Perceptually uniform (purple to yellow)
    # cv2.COLORMAP_TURBO - High contrast (blue to red through green)
    # cv2.COLORMAP_RAINBOW - Full spectrum
    COLORMAP = cv2.COLORMAP_JET
    
    try:
        # Initialize processor
        processor = NDVIImageProcessor(INPUT_IMAGE_PATH, OUTPUT_DIRECTORY)
        
        # Process image
        results = processor.process(colormap=COLORMAP, create_overlay_img=True)
        
        print("\nGenerated files:")
        for key, path in results.items():
            if path:
                print(f"  {key}: {path}")
                
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
