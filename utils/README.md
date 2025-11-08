# Utility Functions Directory

This directory contains helper functions used across the project.

## Modules

### `image_io.py`
- **`load_image(image_path)`**: Load image using OpenCV
- **`bgr_to_rgb(img)`**: Convert BGR (OpenCV) to RGB (Matplotlib)
- **`rgb_to_bgr(img)`**: Convert RGB to BGR

### `point_selection.py`
- **`select_correspondences(img1, img2, n_points, title1, title2)`**: Interactive point selection with side-by-side display
- **`save_correspondences(points1, points2, output_path)`**: Save correspondences using `numpy.save()`
- **`load_correspondences(file_path)`**: Load correspondences from `.npy` or `.json` file

### `math_utils.py`
- **`to_homogeneous(points)`**: Convert 2D points to homogeneous coordinates
- **`from_homogeneous(points_homog)`**: Convert homogeneous coordinates back to 2D
- **`normalize_points(points)`**: Normalize points for numerical stability
- **`denormalize_homography(H, T1, T2)`**: Denormalize homography after computation

### `convert_json_to_numpy.py`
Utility script to convert existing JSON correspondence files to numpy format.

