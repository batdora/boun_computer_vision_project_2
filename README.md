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
- OpenCV (cv2)
- NumPy
- Matplotlib

## Usage

### Selecting Point Correspondences

Use the correspondence selector to manually select corresponding points between image pairs:

```bash
python src/correspondence_selector.py <image1_path> <image2_path> [num_points] [output_dir]
```

**Examples:**
```bash
# Select 4 points (default, minimum for homography)
python src/correspondence_selector.py images/paris/paris_a.jpg images/paris/paris_b.jpg

# Select 6 points
python src/correspondence_selector.py images/paris/paris_a.jpg images/paris/paris_b.jpg 6

# Specify output directory
python src/correspondence_selector.py images/paris/paris_a.jpg images/paris/paris_b.jpg 4 correspondences/
```

**Interactive Process:**
1. The script will display the first image - click the desired number of points
2. Then display the second image - click the corresponding points in the same order
3. Correspondences are saved as JSON file: `{image1}_{image2}_correspondences.json`

**Output Format:**
The correspondences are saved as JSON with:
- Image names
- Number of points
- Points from image 1 (array of [x, y] coordinates)
- Points from image 2 (array of [x, y] coordinates)

## Notes
- Manual point correspondence selection (automated feature detection forbidden)
- Homography computation from correspondences
- Image warping and blending

