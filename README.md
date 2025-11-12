# Computer Vision Project 2

Panorama stitching with fully manual correspondence selection. The project
follows the course guidelines:

- Interactive point picking (no automated feature detectors)
- Homography estimation via SVD
- Custom backward-warping
- Maximum-intensity blending

## Repository Layout

```
Project 2/
├── main.py                 # Primary stitching entry point (multi-image)
├── src/                    # Core homography / warping / blending modules
├── utils/                  # Shared utilities (image I/O, point selection)
├── experiments/            # Optional exploration scripts (not submitted)
├── images/                 # Provided datasets + saved correspondences
├── correspondences/        # (Created automatically) cached manual matches
├── output/                 # Generated panoramas and experiment artefacts
└── README.md               # Project documentation
```

## Requirements

- Python 3.x
- OpenCV (`opencv-python`) – image I/O and fast remap (PIL is used as fallback)
- NumPy
- Matplotlib – interactive correspondence picker
- SciPy – interpolation fallback when OpenCV remap is unavailable

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Core Workflow (`main.py`)

`main.py` consumes two or more images ordered from left to right, collects or
reuses correspondences for each adjacent pair, and outputs a stitched panorama.

```
python main.py [options] image_0 image_1 ... image_n
```

Options:

- `--output PATH` – output panorama path (default `output/panorama.jpg`)
- `--corr-dir DIR` – directory for cached `.npy` correspondences (default
  `correspondences/`)
- `--points N` – number of points to collect per pair (default `8`)
- `--overwrite` – force re-selection even if a cache file already exists

Pipeline stages:

1. **Load images** in BGR (OpenCV) and convert to RGB for Matplotlib.
2. **For each consecutive pair**, call the interactive picker if no cached
   correspondences exist.
3. **Estimate homographies** mapping every image onto the coordinate frame of
   the left-most image by chaining right→left transforms.
4. **Warp and blend** the full stack with maximum-intensity fusion.
5. **Save the panorama** to the requested path.

Correspondence files are saved automatically as
`<left>_<right>_correspondences.npy` (shape `(2, n_points, 2)`), making iterative
refinement quick—re-run with `--overwrite` to draw new points.

## Supporting Tools

- `utils/point_selection.py` – the Matplotlib-driven picker plus
  `numpy.save`/`numpy.load` helpers.
- `src/homography.py` – `computeH(points_im1, points_im2)` implementing DLT +
  SVD with optional point normalization.
- `src/warping.py` – `warp(image, homography)` using a backward transform with
  OpenCV remap / SciPy griddata / NumPy fallbacks.
- `src/blending.py` – maximum-intensity compositing utilities and a convenience
  `create_panorama` wrapper leveraged by `main.py`.

## Experiments (Optional)

Exploratory scripts now live in `experiments/` and are not part of the course
submission. Run them as modules so that imports resolve correctly, e.g.:

```
python -m experiments.pairwise_demo ...
python -m experiments.pipeline_example ...
python -m experiments.stitcher_experiment
python -m experiments.correspondance_experiment
```

See `experiments/README.md` for details on each script.

## Manual-Only Reminder

- Automated feature detectors (SIFT/SURF/Harris/ORB/etc.) were removed from the
  codebase.
- Every homography used in the main pipeline originates from user-selected
  correspondences stored as `.npy` files.
- The blending strategy honours the assignment’s “maximum intensity” directive.

Enjoy stitching! If you modify the datasets or capture your own images, simply
provide them in left-to-right order and re-run `main.py` to generate a new
panorama.*** End Patch

