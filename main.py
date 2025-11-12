"""
Panorama stitching entry point.

Given a list of images ordered from left to right, this script:
1. Collects (or loads) correspondence points for every consecutive pair.
2. Estimates homographies that map each image onto the coordinate frame of the
   left-most image.
3. Warps all images into the shared canvas.
4. Blends them using maximum-intensity blending and writes the resulting
   panorama to disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

# Ensure project root modules are importable when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.homography import computeH  # noqa: E402
from src.blending import create_panorama  # noqa: E402
from utils.image_io import load_image, bgr_to_rgb, rgb_to_bgr  # noqa: E402
from utils.point_selection import (  # noqa: E402
    load_correspondences,
    save_correspondences,
    select_correspondences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive panorama stitching for a left-to-right image sequence."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image paths ordered from left to right (minimum two).",
    )
    parser.add_argument(
        "--output",
        default="output/panorama.jpg",
        help="Path to the stitched panorama image (default: output/panorama.jpg).",
    )
    parser.add_argument(
        "--corr-dir",
        default="correspondences",
        help="Directory to read/write correspondence .npy files.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=8,
        help="Number of correspondences to collect per adjacent pair (default: 8).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-selection of correspondences even if cache files exist.",
    )
    return parser.parse_args()


def ensure_correspondences(
    left_img_rgb: np.ndarray,
    right_img_rgb: np.ndarray,
    left_path: Path,
    right_path: Path,
    corr_path: Path,
    n_points: int,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load existing correspondences if available (and not overwriting), otherwise
    launch the interactive picker and persist the selection.
    """
    if corr_path.exists() and not overwrite:
        points_left, points_right, _metadata = load_correspondences(corr_path)
        print(f"   Loaded cached correspondences from {corr_path}")
        return np.asarray(points_left), np.asarray(points_right)

    print(f"   Selecting correspondences for {left_path.name} ↔ {right_path.name}")
    points_left, points_right = select_correspondences(
        left_img_rgb,
        right_img_rgb,
        n_points=n_points,
        title1=left_path.name,
        title2=right_path.name,
    )

    corr_path.parent.mkdir(parents=True, exist_ok=True)
    save_correspondences(
        points_left,
        points_right,
        corr_path,
        img1_name=left_path.name,
        img2_name=right_path.name,
    )
    print(f"   Saved correspondences to {corr_path}")
    return points_left, points_right


def build_homographies(
    images_rgb: List[np.ndarray],
    image_paths: List[Path],
    corr_dir: Path,
    n_points: int,
    overwrite: bool,
) -> List[np.ndarray]:
    """
    Collect correspondences for each adjacent pair and accumulate homographies
    that map every image into the coordinate system of the first image.
    """
    num_images = len(images_rgb)
    cumulative_homographies: List[np.ndarray] = [np.eye(3, dtype=float)]

    for idx in range(num_images - 1):
        left_img = images_rgb[idx]
        right_img = images_rgb[idx + 1]
        left_path = image_paths[idx]
        right_path = image_paths[idx + 1]

        corr_filename = f"{left_path.stem}_{right_path.stem}_correspondences.npy"
        corr_path = corr_dir / corr_filename

        print(f"\nCollecting correspondences for pair {idx + 1}/{num_images - 1}:")
        points_left, points_right = ensure_correspondences(
            left_img,
            right_img,
            left_path,
            right_path,
            corr_path,
            n_points,
            overwrite,
        )

        # Map the right image onto the left image's coordinate frame
        H_right_to_left = computeH(points_right, points_left)
        cumulative_homographies.append(
            cumulative_homographies[-1] @ H_right_to_left
        )
        print("   Homography (right → left) computed.")

    return cumulative_homographies


def stitch_sequence(args: argparse.Namespace) -> Path:
    image_paths = [Path(p) for p in args.images]
    if len(image_paths) < 2:
        raise ValueError("Please provide at least two images.")

    print("Loading images...")
    images_rgb: List[np.ndarray] = []
    for path in image_paths:
        img_bgr = load_image(path)
        images_rgb.append(bgr_to_rgb(img_bgr))
        print(f"   {path} loaded with shape {images_rgb[-1].shape}")

    corr_dir = Path(args.corr_dir)
    homographies = build_homographies(
        images_rgb, image_paths, corr_dir, args.points, args.overwrite
    )

    print("\nWarping and blending panorama...")
    panorama = create_panorama(images_rgb, homographies=homographies)
    panorama_uint8 = np.clip(panorama, 0, 255).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panorama_bgr = rgb_to_bgr(panorama_uint8)

    try:
        import cv2  # type: ignore

        cv2.imwrite(str(output_path), panorama_bgr)
    except Exception:  # pragma: no cover - fallback rarely needed
        from PIL import Image  # type: ignore

        Image.fromarray(panorama_uint8).save(output_path)

    print(f"\nPanorama written to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    stitch_sequence(args)


if __name__ == "__main__":
    main()

