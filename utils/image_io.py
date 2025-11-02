"""
Image I/O utilities for loading and saving images.
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(image_path):
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array (BGR format from OpenCV)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return img


def bgr_to_rgb(img):
    """
    Convert BGR image (OpenCV format) to RGB (matplotlib format).
    
    Args:
        img: Image in BGR format
        
    Returns:
        Image in RGB format
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img):
    """
    Convert RGB image (matplotlib format) to BGR (OpenCV format).
    
    Args:
        img: Image in RGB format
        
    Returns:
        Image in BGR format
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

