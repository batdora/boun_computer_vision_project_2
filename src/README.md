# Source Code Directory

This directory contains the main implementation modules for the project.

## Module Structure

### Core Modules

#### `homography.py`
- **`computeH(points_im1, points_im2)`**: Compute homography matrix using DLT algorithm
- **`apply_homography(H, points)`**: Transform points using homography
- **`compute_homography_from_correspondence_file()`**: Load correspondences and compute homography

#### `warping.py`
- **`warp(image, homography)`**: Warp image using backward transform
- **`get_warp_offset(image, homography)`**: Get bounding box offset of warped image
- **`get_warp_bounds(image, homography)`**: Get full bounding box (min_x, min_y, max_x, max_y)

#### `blending.py`
- **`blend_max_intensity(*images)`**: Simple maximum intensity blending for aligned images
- **`blend_images(images, offsets, canvas_size)`**: Blending with positional offsets
- **`create_panorama(images, homographies)`**: Complete panorama creation pipeline

### Utility Scripts

#### `correspondence_selector.py`
Interactive point correspondence selection tool.

#### `homography_estimator.py`
Command-line tool for computing and saving homographies.

#### `compute_all_homographies.py`
Batch computation of all homographies for all image sets.

## Main Components
- ✅ Homography computation (DLT with normalization)
- ✅ Image warping (backward transform)
- ✅ Image blending (maximum intensity)
- ✅ Manual point correspondence handling

> **Note:** example and experiment scripts now live under `experiments/`. Use
> `python -m experiments.pipeline_example ...` for the legacy demos.

