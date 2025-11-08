"""
Image blending module.
Implements blending techniques for merging warped images into panoramas.
"""

import numpy as np


def blend_max_intensity(*images):
    """
    Blend multiple images using maximum intensity method.
    For overlapping areas, uses the pixel with maximum intensity value.
    
    Args:
        *images: Variable number of images to blend. All images must have the same shape.
                 Each image can be grayscale (H, W) or color (H, W, C).
        
    Returns:
        Blended image with same shape as input images
        
    Note:
        This is a simple blending method that works well for images with similar
        exposure and lighting. For overlapping areas, it selects the maximum pixel
        value, which helps preserve bright details.
    """
    if len(images) == 0:
        raise ValueError("At least one image required for blending")
    
    if len(images) == 1:
        return np.array(images[0]).copy()
    
    # Convert all to numpy arrays
    images = [np.array(img) for img in images]
    
    # Check that all images have the same shape
    shapes = [img.shape for img in images]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(f"All images must have the same shape. Got: {shapes}")
    
    # Stack all images along a new dimension
    # Shape: (n_images, H, W) or (n_images, H, W, C)
    stacked = np.stack(images, axis=0)
    
    # Take maximum along the first axis (across all images)
    blended = np.max(stacked, axis=0)
    
    return blended.astype(images[0].dtype)


def blend_images(images, offsets=None, canvas_size=None):
    """
    Blend multiple warped images into a single panorama canvas.
    Handles images at different positions using maximum intensity blending.
    
    Args:
        images: List of images to blend (each can be grayscale or color)
        offsets: List of (x_offset, y_offset) tuples for each image position.
                 If None, images are assumed to be at (0, 0) each (stacked).
        canvas_size: Tuple (width, height) for output canvas.
                    If None, computed from image bounds.
        
    Returns:
        Blended panorama image
        
    Note:
        Uses maximum intensity blending for overlapping regions.
        Non-overlapping regions get values from the single image covering that area.
    """
    images = [np.array(img) for img in images]
    
    if offsets is None:
        # If no offsets, use simple max intensity blending
        return blend_max_intensity(*images)
    
    if len(images) != len(offsets):
        raise ValueError(f"Number of images ({len(images)}) must match number of offsets ({len(offsets)})")
    
    # Determine canvas size if not provided
    if canvas_size is None:
        max_x = max(offset[0] + img.shape[1] for offset, img in zip(offsets, images))
        max_y = max(offset[1] + img.shape[0] for offset, img in zip(offsets, images))
        min_x = min(offset[0] for offset in offsets)
        min_y = min(offset[1] for offset in offsets)
        
        canvas_w = int(max_x - min_x)
        canvas_h = int(max_y - min_y)
        
        # Adjust offsets to be relative to canvas origin
        adjusted_offsets = [(off[0] - min_x, off[1] - min_y) for off in offsets]
        offsets = adjusted_offsets
    else:
        canvas_w, canvas_h = canvas_size
    
    # Determine if images are color or grayscale
    is_color = any(len(img.shape) == 3 for img in images)
    num_channels = images[0].shape[2] if is_color else 1
    
    # Create canvas - use float64 for better precision to preserve colors
    if is_color:
        canvas = np.zeros((canvas_h, canvas_w, num_channels), dtype=np.float64)
    else:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    
    # Create a mask to track where we have valid pixels
    valid_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    
    # Place each image and update maximum values
    for img, (x_off, y_off) in zip(images, offsets):
        h, w = img.shape[:2]
        
        # Calculate bounds within canvas
        x_start = int(x_off)
        y_start = int(y_off)
        x_end = x_start + w
        y_end = y_start + h
        
        # Calculate which part of image and canvas to use
        canvas_x_start = max(0, x_start)
        canvas_y_start = max(0, y_start)
        canvas_x_end = min(canvas_w, x_end)
        canvas_y_end = min(canvas_h, y_end)
        
        img_x_start = canvas_x_start - x_start
        img_y_start = canvas_y_start - y_start
        img_x_end = img_x_start + (canvas_x_end - canvas_x_start)
        img_y_end = img_y_start + (canvas_y_end - canvas_y_start)
        
        # Extract relevant region from image
        img_region = img[img_y_start:img_y_end, img_x_start:img_x_end]
        
        # Get corresponding canvas region
        canvas_region = canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end]
        region_mask = valid_mask[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end]
        
        if is_color:
            # For color images, blend each channel separately
            # Convert image region to float64 for precision
            img_region_f = img_region.astype(np.float64)
            for c in range(num_channels):
                canvas_ch = canvas_region[:, :, c]
                img_ch = img_region_f[:, :, c]
                
                # For overlapping areas (where mask is True), take maximum intensity
                # For non-overlapping areas, use the new image value
                overlap_region = region_mask
                canvas_ch[overlap_region] = np.maximum(
                    canvas_ch[overlap_region],
                    img_ch[overlap_region]
                )
                canvas_ch[~overlap_region] = img_ch[~overlap_region]
        else:
            # Grayscale image - convert to float64 for precision
            img_region_f = img_region.astype(np.float64)
            overlap_region = region_mask
            canvas_region[overlap_region] = np.maximum(
                canvas_region[overlap_region],
                img_region_f[overlap_region]
            )
            canvas_region[~overlap_region] = img_region_f[~overlap_region]
        
        # Update valid mask to indicate this region now has pixels
        valid_mask[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = True
    
    # Convert back to original dtype with proper clipping
    # Use float64 precision during blending, then convert to uint8 with proper clipping
    if len(images) > 0 and images[0].dtype == np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    else:
        canvas = canvas.astype(images[0].dtype) if len(images) > 0 else canvas
    
    return canvas


def create_panorama(images, homographies=None, reference_idx=0):
    """
    Create a panorama by warping and blending multiple images.
    
    Uses maximum intensity blending for overlapping areas as specified in guidelines.
    
    Args:
        images: List of input images (all in same coordinate space initially)
        homographies: List of homography matrices. If None, images are not warped.
                     Each H maps image[i] to the reference coordinate system.
        reference_idx: Index of reference image (default: 0)
        
    Returns:
        Blended panorama image
        
    Note:
        This is a convenience function that combines warping and blending.
        It warps all images to a common coordinate system, then blends them using
        maximum intensity method for overlapping pixels.
    """
    from src.warping import warp, get_warp_offset, get_warp_bounds
    
    if homographies is None:
        # No warping needed, just blend directly
        return blend_max_intensity(*images)
    
    if len(images) != len(homographies):
        raise ValueError(f"Number of images ({len(images)}) must match number of homographies ({len(homographies)})")
    
    # Warp all images and get their bounding boxes
    warped_images = []
    bounds_list = []
    
    for img, H in zip(images, homographies):
        warped = warp(img, H)
        bounds = get_warp_bounds(img, H)
        warped_images.append(warped)
        bounds_list.append(bounds)
    
    # Calculate global bounds
    all_min_x = [b[0] for b in bounds_list]
    all_min_y = [b[1] for b in bounds_list]
    all_max_x = [b[2] for b in bounds_list]
    all_max_y = [b[3] for b in bounds_list]
    
    global_min_x = min(all_min_x)
    global_min_y = min(all_min_y)
    global_max_x = max(all_max_x)
    global_max_y = max(all_max_y)
    
    # Calculate offsets relative to global minimum
    offsets = []
    for bounds in bounds_list:
        offset_x = bounds[0] - global_min_x
        offset_y = bounds[1] - global_min_y
        offsets.append((offset_x, offset_y))
    
    # Calculate canvas size
    canvas_w = global_max_x - global_min_x
    canvas_h = global_max_y - global_min_y
    
    # Blend warped images using maximum intensity
    panorama = blend_images(warped_images, offsets=offsets, canvas_size=(canvas_w, canvas_h))
    
    return panorama

