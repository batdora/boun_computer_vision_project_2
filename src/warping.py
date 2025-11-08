"""
Image warping module.
Implements backward transform warping using homography matrices.
"""

import numpy as np
import sys
from pathlib import Path

# Import math_utils directly to avoid cv2 dependency through utils.__init__
sys.path.insert(0, str(Path(__file__).parent.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("math_utils", Path(__file__).parent.parent / "utils" / "math_utils.py")
math_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_utils)
to_homogeneous = math_utils.to_homogeneous

# Try to import interpolation functions
try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def warp(image, homography):
    """
    Warp an image using a homography matrix with backward transform.
    
    Signature matches project requirements: image_warped = warp(image, homography)
    
    Args:
        image: Input image as numpy array (height, width, channels) or (height, width)
        homography: 3x3 homography matrix
        
    Returns:
        Warped image as numpy array with same dtype as input
        
    Notes:
        - Uses backward transform (inverse mapping) for better results
        - Uses bilinear interpolation via scipy.interpolate.griddata
        - Computes bounding box of warped image
    """
    image = np.array(image)
    H = np.array(homography)
    
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3 matrix, got shape {H.shape}")
    
    h, w = image.shape[:2]
    is_color = len(image.shape) == 3
    num_channels = image.shape[2] if is_color else 1
    
    # Get image corners in homogeneous coordinates
    corners = np.array([
        [0, 0],      # top-left
        [w, 0],      # top-right
        [w, h],      # bottom-right
        [0, h]       # bottom-left
    ])
    corners_homog = to_homogeneous(corners)
    
    # Transform corners using homography
    corners_transformed = (H @ corners_homog.T).T
    corners_transformed = corners_transformed[:, :2] / corners_transformed[:, 2:3]
    
    # Compute bounding box of warped image
    min_x = int(np.floor(corners_transformed[:, 0].min()))
    max_x = int(np.ceil(corners_transformed[:, 0].max()))
    min_y = int(np.floor(corners_transformed[:, 1].min()))
    max_y = int(np.ceil(corners_transformed[:, 1].max()))
    
    # Create output image with bounding box size
    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1
    
    # Create grid of output coordinates in the warped space
    # These are the actual x, y coordinates in the warped coordinate system
    y_out, x_out = np.mgrid[0:out_h, 0:out_w]
    # Convert grid indices to actual warped coordinates
    x_out = x_out.astype(np.float64) + min_x  # Adjust for offset
    y_out = y_out.astype(np.float64) + min_y
    
    # Flatten for processing
    x_out_flat = x_out.flatten()
    y_out_flat = y_out.flatten()
    
    # Backward transform: map output coordinates back to input image
    # We need H^(-1) to map from output to input
    H_inv = np.linalg.inv(H)
    
    # Convert output coordinates to homogeneous
    output_coords = np.column_stack([x_out_flat, y_out_flat, np.ones(len(x_out_flat))])
    
    # Transform back to input coordinates
    input_coords = (H_inv @ output_coords.T).T
    input_coords = input_coords[:, :2] / input_coords[:, 2:3]
    
    x_in = input_coords[:, 0]
    y_in = input_coords[:, 1]
    
    # Prepare output image - preserve original dtype
    if is_color:
        warped = np.zeros((out_h, out_w, num_channels), dtype=image.dtype)
    else:
        warped = np.zeros((out_h, out_w), dtype=image.dtype)
    
    # Filter coordinates that are within input image bounds
    valid = (x_in >= 0) & (x_in < w) & (y_in >= 0) & (y_in < h)
    
    if np.any(valid):
        # Reshape coordinates for remap - preserve row-major order (C-style)
        map_x = x_in.reshape(out_h, out_w)
        map_y = y_in.reshape(out_h, out_w)
        
        if HAS_CV2:
            # Use OpenCV remap for better performance and quality
            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)
            
            # OpenCV remap expects map_x and map_y to be the same shape as output
            if is_color:
                warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif HAS_SCIPY:
            # Use scipy griddata interpolation
            # Create grid of input image coordinates (correct order: height first, then width)
            x_img_grid, y_img_grid = np.meshgrid(np.arange(w), np.arange(h))
            x_img_flat = x_img_grid.flatten()
            y_img_flat = y_img_grid.flatten()
            
            if is_color:
                # Interpolate each channel separately
                for c in range(num_channels):
                    values = image[:, :, c].flatten()
                    warped_flat = griddata(
                        (x_img_flat, y_img_flat),
                        values,
                        (x_in[valid], y_in[valid]),
                        method='linear',
                        fill_value=0
                    )
                    warped_flat_valid = np.zeros(out_h * out_w)
                    warped_flat_valid[valid] = warped_flat
                    warped[:, :, c] = warped_flat_valid.reshape(out_h, out_w)
            else:
                # Grayscale image
                values = image.flatten()
                warped_flat = griddata(
                    (x_img_flat, y_img_flat),
                    values,
                    (x_in[valid], y_in[valid]),
                    method='linear',
                    fill_value=0
                )
                warped_flat_valid = np.zeros(out_h * out_w)
                warped_flat_valid[valid] = warped_flat
                warped = warped_flat_valid.reshape(out_h, out_w)
        else:
            # Fallback: simple nearest neighbor (not ideal but works)
            x_in_int = np.clip(x_in.reshape(out_h, out_w), 0, w-1).astype(int)
            y_in_int = np.clip(y_in.reshape(out_h, out_w), 0, h-1).astype(int)
            mask = valid.reshape(out_h, out_w)
            
            if is_color:
                for c in range(num_channels):
                    warped[mask, c] = image[y_in_int[mask], x_in_int[mask], c]
            else:
                warped[mask] = image[y_in_int[mask], x_in_int[mask]]
    
    return warped


def get_warp_offset(image, homography):
    """
    Get the offset (translation) of the warped image bounding box.
    
    Args:
        image: Input image (height, width, channels) or (height, width)
        homography: 3x3 homography matrix
        
    Returns:
        Tuple (min_x, min_y) representing the offset of the warped image
    """
    image = np.array(image)
    H = np.array(homography)
    
    h, w = image.shape[:2]
    
    # Get image corners
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    corners_homog = to_homogeneous(corners)
    
    # Transform corners
    corners_transformed = (H @ corners_homog.T).T
    corners_transformed = corners_transformed[:, :2] / corners_transformed[:, 2:3]
    
    min_x = int(np.floor(corners_transformed[:, 0].min()))
    min_y = int(np.floor(corners_transformed[:, 1].min()))
    
    return min_x, min_y


def get_warp_bounds(image, homography):
    """
    Get the bounding box of the warped image.
    
    Args:
        image: Input image (height, width, channels) or (height, width)
        homography: 3x3 homography matrix
        
    Returns:
        Tuple (min_x, min_y, max_x, max_y) representing the bounding box
    """
    image = np.array(image)
    H = np.array(homography)
    
    h, w = image.shape[:2]
    
    # Get image corners
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])
    corners_homog = to_homogeneous(corners)
    
    # Transform corners
    corners_transformed = (H @ corners_homog.T).T
    corners_transformed = corners_transformed[:, :2] / corners_transformed[:, 2:3]
    
    min_x = int(np.floor(corners_transformed[:, 0].min()))
    max_x = int(np.ceil(corners_transformed[:, 0].max()))
    min_y = int(np.floor(corners_transformed[:, 1].min()))
    max_y = int(np.ceil(corners_transformed[:, 1].max()))
    
    return min_x, min_y, max_x, max_y

