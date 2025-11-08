"""
Experimental stitching strategies for five-image panoramas.

Runs three stitching approaches (left-to-right, middle-out, first-out-then-middle)
for both the cmpe_building and north_campus datasets without requiring CLI inputs.

This script is meant for exploratory analysis and can be invoked directly, e.g.:

    python stitcher_experiment.py

It relies on the existing correspondence .npy files generated via the manual
point selection tool.
"""

from pathlib import Path
import numpy as np  # type: ignore

from pipeline_example import (
    load_image_rgb,
    ensure_correspondence_direction,
    compute_canvas_and_offsets,
    ensure_uint8,
)
from src.homography import computeH
from src.warping import warp, get_warp_bounds
from src.blending import blend_images

try:
    from PIL import Image  # type: ignore
except ImportError:
    Image = None


DATASETS = ("cmpe_building", "north_campus")
IMAGE_NAMES = ["left_2", "left_1", "middle", "right_1", "right_2"]


def compute_homography_from_corr(corr_path, src_stem, dst_stem):
    """Load correspondences and compute homography mapping src -> dst."""
    pts_src, pts_dst, _ = ensure_correspondence_direction(corr_path, src_stem, dst_stem)
    return computeH(pts_src, pts_dst)


def load_dataset_assets(dataset):
    """Load images and compute all required homographies for a dataset."""
    base_path = Path("images") / dataset
    corr_dir = base_path / "correspondences"

    images = {
        name: load_image_rgb(base_path / f"{name}.jpg")
        for name in IMAGE_NAMES
    }

    # Base pair correspondences
    H_l2_to_l1 = compute_homography_from_corr(
        corr_dir / "left_2_left_1_correspondences.npy", "left_2", "left_1"
    )
    H_l1_to_middle = compute_homography_from_corr(
        corr_dir / "left_1_middle_correspondences.npy", "left_1", "middle"
    )
    H_middle_to_right1 = compute_homography_from_corr(
        corr_dir / "middle_right_1_correspondences.npy", "middle", "right_1"
    )
    H_right1_to_right2 = compute_homography_from_corr(
        corr_dir / "right_1_right_2_correspondences.npy", "right_1", "right_2"
    )

    # Inverses and composed homographies
    H_left1_to_left2 = np.linalg.inv(H_l2_to_l1)
    H_middle_to_left1 = np.linalg.inv(H_l1_to_middle)
    H_right1_to_middle = np.linalg.inv(H_middle_to_right1)
    H_right2_to_right1 = np.linalg.inv(H_right1_to_right2)

    H_middle_to_left2 = H_left1_to_left2 @ H_middle_to_left1
    H_left2_to_middle = H_l1_to_middle @ H_l2_to_l1

    H_right1_to_left2 = H_middle_to_left2 @ H_right1_to_middle
    H_right2_to_left2 = H_right1_to_left2 @ H_right2_to_right1

    H_left2_to_right1 = np.linalg.inv(H_right1_to_left2)
    H_left2_to_right2 = np.linalg.inv(H_right2_to_left2)

    H_right2_to_middle = H_right1_to_middle @ H_right2_to_right1

    homographies = {
        "H_l2_to_l1": H_l2_to_l1,
        "H_l1_to_left2": H_left1_to_left2,
        "H_l1_to_middle": H_l1_to_middle,
        "H_middle_to_left1": H_middle_to_left1,
        "H_middle_to_left2": H_middle_to_left2,
        "H_left2_to_middle": H_left2_to_middle,
        "H_middle_to_right1": H_middle_to_right1,
        "H_right1_to_middle": H_right1_to_middle,
        "H_right1_to_right2": H_right1_to_right2,
        "H_right2_to_right1": H_right2_to_right1,
        "H_right1_to_left2": H_right1_to_left2,
        "H_right2_to_left2": H_right2_to_left2,
        "H_left2_to_right1": H_left2_to_right1,
        "H_left2_to_right2": H_left2_to_right2,
        "H_right2_to_middle": H_right2_to_middle,
    }

    return images, homographies


def blend_overlays(base_image, overlays, output_path=None):
    """
    Blend a list of overlays onto a base image.

    overlays: list of dicts with keys:
        - "image": numpy array (RGB)
        - "H": homography mapping overlay -> base coordinate system
        - "label": optional string for logging
    """
    bounds = []
    warped_images = []
    labels = []

    for overlay in overlays:
        warped = warp(overlay["image"], overlay["H"])
        warped_images.append(ensure_uint8(warped))
        bounds.append(get_warp_bounds(overlay["image"], overlay["H"]))
        labels.append(overlay.get("label", "overlay"))

    base_bounds = (0, 0, base_image.shape[1], base_image.shape[0])
    bounds.append(base_bounds)

    canvas_w, canvas_h, offsets = compute_canvas_and_offsets(bounds)
    overlay_offsets = offsets[:-1]
    base_offset = offsets[-1]

    images = warped_images + [ensure_uint8(base_image)]
    offsets_for_blend = overlay_offsets + [base_offset]

    mosaic = blend_images(images, offsets=offsets_for_blend, canvas_size=(canvas_w, canvas_h))

    if output_path is not None:
        save_image(mosaic, output_path)

    return mosaic


def save_image(image, path):
    """Save image to disk, handling PIL availability."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        Image.fromarray(ensure_uint8(image)).save(path)
    else:
        print(f"[Warning] Unable to save {path} because PIL is unavailable.")


def crop_to_content(image, threshold=5):
    """Return cropped image bounding non-empty regions; None if empty."""
    img = ensure_uint8(image)
    gray = img.mean(axis=2) if img.ndim == 3 else img
    mask = gray > threshold
    if not np.any(mask):
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return img[r0:r1, c0:c1]


def run_left_to_right_experiment(dataset, images, H, output_root):
    """Sequentially add images from left to right, always anchoring on left_2."""
    print(f"\n[Left-to-Right] Dataset: {dataset}")
    base = images["left_2"]
    overlays = []

    steps = [
        ("mosaic_1_left1", [{"image": images["left_1"], "H": H["H_l1_to_left2"], "label": "left_1"}]),
        (
            "mosaic_2_add_middle",
            [
                {"image": images["left_1"], "H": H["H_l1_to_left2"], "label": "left_1"},
                {"image": images["middle"], "H": H["H_middle_to_left2"], "label": "middle"},
            ],
        ),
        (
            "mosaic_3_add_right1",
            [
                {"image": images["left_1"], "H": H["H_l1_to_left2"], "label": "left_1"},
                {"image": images["middle"], "H": H["H_middle_to_left2"], "label": "middle"},
                {"image": images["right_1"], "H": H["H_right1_to_left2"], "label": "right_1"},
            ],
        ),
        (
            "mosaic_final_left_to_right",
            [
                {"image": images["left_1"], "H": H["H_l1_to_left2"], "label": "left_1"},
                {"image": images["middle"], "H": H["H_middle_to_left2"], "label": "middle"},
                {"image": images["right_1"], "H": H["H_right1_to_left2"], "label": "right_1"},
                {"image": images["right_2"], "H": H["H_right2_to_left2"], "label": "right_2"},
            ],
        ),
    ]

    for step_name, current_overlays in steps:
        print(f"  - Generating {step_name} ...")
        output_path = output_root / "left_to_right" / f"{step_name}.jpg"
        blend_overlays(base, current_overlays, output_path=output_path)


def run_middle_out_experiment(dataset, images, H, output_root):
    """Anchor on the middle image, first adding immediate neighbors, then outer ones."""
    print(f"\n[Middle-Out] Dataset: {dataset}")
    base = images["middle"]

    steps = [
        (
            "mosaic_1_neighbors",
            [
                {"image": images["left_1"], "H": H["H_l1_to_middle"], "label": "left_1"},
                {"image": images["right_1"], "H": H["H_right1_to_middle"], "label": "right_1"},
            ],
        ),
        (
            "mosaic_final_middle_out",
            [
                {"image": images["left_1"], "H": H["H_l1_to_middle"], "label": "left_1"},
                {"image": images["right_1"], "H": H["H_right1_to_middle"], "label": "right_1"},
                {"image": images["left_2"], "H": H["H_left2_to_middle"], "label": "left_2"},
                {"image": images["right_2"], "H": H["H_right2_to_middle"], "label": "right_2"},
            ],
        ),
    ]

    for step_name, overlays in steps:
        print(f"  - Generating {step_name} ...")
        output_path = output_root / "middle_out" / f"{step_name}.jpg"
        blend_overlays(base, overlays, output_path=output_path)


def run_first_out_then_middle_experiment(dataset, images, H, output_root):
    """Stitch left and right halves separately, then merge using the middle anchor."""
    print(f"\n[First-Out-Then-Middle] Dataset: {dataset}")

    # Left side (base left_1)
    blend_overlays(
        images["left_1"],
        [{"image": images["left_2"], "H": H["H_l2_to_l1"], "label": "left_2"}],
        output_path=output_root / "first_out_then_middle" / "mosaic_left.jpg",
    )

    # Right side (base right_1)
    blend_overlays(
        images["right_1"],
        [{"image": images["right_2"], "H": H["H_right2_to_right1"], "label": "right_2"}],
        output_path=output_root / "first_out_then_middle" / "mosaic_right.jpg",
    )

    # Final merge using middle as anchor
    overlays = [
        {"image": images["left_1"], "H": H["H_l1_to_middle"], "label": "left_1"},
        {"image": images["left_2"], "H": H["H_left2_to_middle"], "label": "left_2"},
        {"image": images["right_1"], "H": H["H_right1_to_middle"], "label": "right_1"},
        {"image": images["right_2"], "H": H["H_right2_to_middle"], "label": "right_2"},
    ]

    blend_overlays(
        images["middle"],
        overlays,
        output_path=output_root
        / "first_out_then_middle"
        / "mosaic_final_first_out_then_middle.jpg",
    )


def run_experiments_for_dataset(dataset):
    print(f"\n=== Running experiments for dataset: {dataset} ===")
    images, homographies = load_dataset_assets(dataset)
    output_root = Path("output") / "experiments" / dataset

    run_left_to_right_experiment(dataset, images, homographies, output_root)
    run_middle_out_experiment(dataset, images, homographies, output_root)
    run_first_out_then_middle_experiment(dataset, images, homographies, output_root)


def main():
    for dataset in DATASETS:
        run_experiments_for_dataset(dataset)
    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()


