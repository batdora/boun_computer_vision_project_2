# Computer Vision Project 2

## Project Overview
Image stitching and panorama creation using manual point correspondences.

## Project Structure
```
Project 2/
├── src/              # Source code
├── utils/            # Utility functions
├── images/           # Input images
└── README.md         # This file
```

## Requirements
- Python 3.x
- OpenCV (cv2) - for image loading (can use PIL as fallback)
- NumPy
- Matplotlib - for interactive point selection
- SciPy - for image interpolation (fallback if OpenCV remap not available)

## Usage

### Basic Pipeline (Two Images)

Run the basic pipeline from `main.py` for two images:

```bash
python main.py <image1_path> <image2_path> [num_points] [corr_file]
```

**Examples:**
```bash
# Interactive point selection (default 4 points)
python main.py images/paris/paris_a.jpg images/paris/paris_b.jpg

# Specify number of points
python main.py images/paris/paris_a.jpg images/paris/paris_b.jpg 6

# Load existing correspondences
python main.py images/paris/paris_a.jpg images/paris/paris_b.jpg 4 correspondences.npy
```

**Pipeline Steps:**
1. Load images
2. Select corresponding points (interactive or from file)
3. Compute homography using `computeH(points_im1, points_im2)`
4. Warp image using `warp(image, homography)`
5. Save warped image to `output/` directory

### Point Correspondence Selection Only

Use the correspondence selector to manually select corresponding points:

```bash
python src/correspondence_selector.py <image1_path> <image2_path> [num_points] [output_dir]
```

**Output Format:**
- Correspondences saved as `.npy` file using `numpy.save()`
- Can be loaded later using `numpy.load()` for reuse
- Format: Array of shape `(2, n_points, 2)` where first dimension is `[points1, points2]`

## Homography Estimation

### Compute All Homographies
Automatically compute all homographies for all image sets:
```bash
python src/compute_all_homographies.py
```

This script handles different strategies based on the dataset:
- **Paris**: Anchor-based (paris_a→paris_b, paris_c→paris_b)
- **cmpe_building & north_campus**: Sequential composition
  - Left chain: left_2→left_1→middle
  - Right chain: right_2→right_1→middle

Output homographies are saved in `images/<dataset>/homographies/` as `.npy` files.

### Single Image Pair
Compute homography from a correspondence file:
```bash
python src/homography_estimator.py <corr_file> [output_file]
```

Example:
```bash
python src/homography_estimator.py images/paris/correspondences/paris_a_paris_b_correspondences.json
```

### Three-Image Setup (Middle Anchor)
For images with left, middle, and right views, compute homographies with middle as anchor:
```bash
python src/homography_estimator.py --three-images <corr_left_middle> <corr_right_middle> [output_dir]
```

**Strategy:**
- **Paris**: Anchor-based (paris_b as anchor)
  - `H_paris_a_to_b`: paris_a → paris_b
  - `H_paris_c_to_b`: paris_c → paris_b
  
- **cmpe_building & north_campus**: Sequential composition
  - Left: `H_left2_to_left1` then `H_left1_to_middle`, composed as `H_left2_to_middle`
  - Right: `H_right2_to_right1` then `H_right1_to_middle`, composed as `H_right2_to_middle`
  
All homographies map toward the middle image coordinate system.

## Creating Panoramas

### Pipeline Example
Run the complete panorama creation pipeline:

```bash
# Paris 3-image panorama (default)
python pipeline_example.py

# Custom two-image panorama
python pipeline_example.py <img1_path> <img2_path> [corr_file]
```

Additional CLI helpers (see `pipeline_example.py --help`):

- **Pairwise stitch** (direction flag + optional corr/output):
  ```bash
  python pipeline_example.py pair <img_a> <img_b> [ltr|rtl] [corr_file] [output]
  ```
- **Anchor triplet** (explicit left/middle/right):
  ```bash
  python pipeline_example.py anchor <left_img> <middle_img> <right_img> [corr_left_middle] [corr_middle_right] [output]
  ```

Each mode saves the full canvas panorama.

### Pipeline Steps
1. **Point Correspondence**: Select corresponding points between images (or load from file)
2. **Homography Estimation**: Compute transformation matrices using `computeH(points_im1, points_im2)`
3. **Image Warping**: Warp images using `warp(image, homography)` with backward transform
4. **Blending**: Merge warped images using maximum intensity for overlapping regions

### Correspondence Effects Analysis
To reproduce the correspondence sensitivity experiments on the Paris image pair:
```bash
python correspondence_effects.py
```
This generates panoramas and a summary report in `output/correspondence_effects/` for:
- Using only five correspondences
- Using many correspondences
- Injecting incorrect matches
- Adding pixel noise with different variances
- Disabling normalization

## Notes
- Manual point correspondence selection (automated feature detection forbidden)
- Homography computation uses DLT (Direct Linear Transform) algorithm with normalization
- Image order matters: All homographies computed toward middle anchor
- Warping uses backward transform with bilinear interpolation (OpenCV remap or SciPy griddata)
- Blending uses maximum intensity method: overlapping pixels take maximum value

