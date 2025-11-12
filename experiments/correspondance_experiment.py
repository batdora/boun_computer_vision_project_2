"""
Correspondence Effects Experiment

Runs a series of controlled experiments on the Paris panorama to illustrate how
different correspondence choices (count, noise, outliers, normalization) impact
the estimated homography and the final stitched panorama.

Each scenario warps both the left (`paris_a`) and right (`paris_c`) views toward
the middle anchor (`paris_b`) and blends all three into a stitched panorama.
Outputs are written under `output/correspondence_effects/`:
  - `<scenario>.jpg` : blended panorama
  - `<scenario>_points.npy` : correspondence set used for the scenario (left pair)
  - `summary.txt` : reprojection error statistics for every scenario

Usage:
    python -m experiments.correspondance_experiment
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np  # type: ignore

# Ensure project root is on sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.pipeline_example import (
    load_image_rgb,
    compute_canvas_and_offsets,
    ensure_uint8,
    ensure_correspondence_direction,
)
from src.homography import computeH, apply_homography
from src.warping import warp, get_warp_bounds
from src.blending import blend_images

try:
    from PIL import Image  # type: ignore
except ImportError:
    Image = None


OUTPUT_DIR = Path("output") / "correspondence_effects"
PARIS_A = Path("images/paris/paris_a.jpg")
PARIS_B = Path("images/paris/paris_b.jpg")
PARIS_C = Path("images/paris/paris_c.jpg")
CORR_FILE_LEFT = Path("images/paris/correspondences/paris_a_paris_b_correspondences.npy")
CORR_FILE_RIGHT = Path("images/paris/correspondences/paris_b_paris_c_correspondences.npy")


ScenarioBuilder = Callable[[np.ndarray, np.ndarray, np.random.Generator], Tuple[np.ndarray, np.ndarray]]


def _points_match_image(points: np.ndarray, image: np.ndarray) -> bool:
    """Return True if all points fall within the image bounds (with small tolerance)."""
    h, w = image.shape[:2]
    tol = 1e-3
    x_valid = (points[:, 0] >= -tol) & (points[:, 0] <= w + tol)
    y_valid = (points[:, 1] >= -tol) & (points[:, 1] <= h + tol)
    return bool(np.all(x_valid & y_valid))


def _orient_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    src_image: np.ndarray,
    dst_image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure correspondences map from src_image to dst_image.
    Falls back to reprojection-error comparison if bounds are inconclusive.
    """
    pts1 = pts1.astype(np.float64)
    pts2 = pts2.astype(np.float64)

    if _points_match_image(pts1, src_image) and _points_match_image(pts2, dst_image):
        return pts1, pts2
    if _points_match_image(pts2, src_image) and _points_match_image(pts1, dst_image):
        return pts2, pts1

    # Fallback: pick the orientation giving lower reprojection error.
    H_forward = computeH(pts1, pts2)
    err_forward, _ = reprojection_error(H_forward, pts1, pts2)
    H_reverse = computeH(pts2, pts1)
    err_reverse, _ = reprojection_error(H_reverse, pts2, pts1)

    if err_forward <= err_reverse:
        return pts1, pts2
    return pts2, pts1


def load_base_correspondences(
    img_left: np.ndarray,
    img_middle: np.ndarray,
    img_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points_left, points_middle_from_left, _ = ensure_correspondence_direction(
        CORR_FILE_LEFT, PARIS_A.stem, PARIS_B.stem
    )
    points_right, points_middle_from_right, _ = ensure_correspondence_direction(
        CORR_FILE_RIGHT, PARIS_C.stem, PARIS_B.stem
    )

    points_left, points_middle_from_left = _orient_points(
        points_left, points_middle_from_left, img_left, img_middle
    )
    points_right, points_middle_from_right = _orient_points(
        points_right, points_middle_from_right, img_right, img_middle
    )

    if len(points_left) < 8:
        raise ValueError(
            "Expected at least 8 correspondence points in the Paris A/B set."
        )
    if len(points_right) < 8:
        raise ValueError(
            "Expected at least 8 correspondence points in the Paris C/B set."
        )

    return (
        points_left,
        points_middle_from_left,
        points_right,
        points_middle_from_right,
    )


def random_subset_builder(n: int) -> ScenarioBuilder:
    def builder(
        points_a: np.ndarray, points_b: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(points_a) < n:
            raise ValueError(f"Not enough correspondences for n={n}.")
        indices = rng.choice(len(points_a), size=n, replace=False)
        return points_a[indices].copy(), points_b[indices].copy()

    return builder


def wrong_points_builder(img_a_shape, img_b_shape) -> ScenarioBuilder:
    ha, wa = img_a_shape[:2]
    hb, wb = img_b_shape[:2]

    def builder(points_a: np.ndarray, points_b: np.ndarray, rng: np.random.Generator):
        n_correct = len(points_a)
        wrong_left = np.column_stack(
            (
                rng.uniform(0, wa, size=n_correct),
                rng.uniform(0, ha, size=n_correct),
            )
        )
        wrong_right = np.column_stack(
            (
                rng.uniform(0, wb, size=n_correct),
                rng.uniform(0, hb, size=n_correct),
            )
        )
        points_left = np.vstack([points_a.copy(), wrong_left])
        points_right = np.vstack([points_b.copy(), wrong_right])
        return points_left, points_right

    return builder


def noisy_builder(sigma: float) -> ScenarioBuilder:
    def builder(points_a: np.ndarray, points_b: np.ndarray, rng: np.random.Generator):
        noise_a = rng.normal(0.0, sigma, size=points_a.shape)
        noise_b = rng.normal(0.0, sigma, size=points_b.shape)
        return points_a.copy() + noise_a, points_b.copy() + noise_b

    return builder


def compute_panorama(
    img_left: np.ndarray,
    img_middle: np.ndarray,
    img_right: np.ndarray,
    H_left_to_middle: np.ndarray,
    H_right_to_middle: np.ndarray,
) -> np.ndarray:
    warped_left = ensure_uint8(warp(img_left, H_left_to_middle))
    warped_right = ensure_uint8(warp(img_right, H_right_to_middle))

    bounds_left = get_warp_bounds(img_left, H_left_to_middle)
    bounds_middle = (0, 0, img_middle.shape[1], img_middle.shape[0])
    bounds_right = get_warp_bounds(img_right, H_right_to_middle)

    canvas_w, canvas_h, offsets = compute_canvas_and_offsets(
        [bounds_left, bounds_middle, bounds_right]
    )
    offset_left, offset_middle, offset_right = offsets

    panorama = blend_images(
        [warped_left, ensure_uint8(img_middle), warped_right],
        offsets=[offset_left, offset_middle, offset_right],
        canvas_size=(canvas_w, canvas_h),
    )
    return panorama


def reprojection_error(
    H: np.ndarray,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
) -> Tuple[float, float]:
    projected = apply_homography(H, pts_src)
    errors = np.linalg.norm(projected - pts_dst, axis=1)
    return float(np.mean(errors)), float(np.max(errors))


def save_image(image: np.ndarray, path: Path) -> None:
    image_uint8 = ensure_uint8(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        Image.fromarray(image_uint8).save(path)
    else:
        import cv2  # type: ignore

        cv2.imwrite(str(path), image_uint8[:, :, ::-1])


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    img_left = load_image_rgb(PARIS_A)
    img_middle = load_image_rgb(PARIS_B)
    img_right = load_image_rgb(PARIS_C)

    (
        base_left_points,
        base_middle_from_left,
        base_right_points,
        base_middle_from_right,
    ) = load_base_correspondences(img_left, img_middle, img_right)
    total_points = len(base_left_points)

    # Pre-compute baseline homography for the right pair (used unless overridden)
    baseline_H_right_to_middle = computeH(base_right_points, base_middle_from_right)

    scenarios: Dict[str, Dict] = {
        "baseline_all": {
            "description": f"All available correspondences ({total_points}) with normalization.",
            "builder": lambda pl, pr, rng: (pl.copy(), pr.copy()),
            "builder_right": lambda pl, pr, rng: (pl.copy(), pr.copy()),
            "normalize": True,
        },
        "points_5": {
            "description": "Random 5 correspondence pairs.",
            "builder": random_subset_builder(5),
            "builder_right": random_subset_builder(5),
            "normalize": True,
        },
        "points_max": {
            "description": "Using all available correspondences.",
            "builder": lambda pl, pr, rng: (pl.copy(), pr.copy()),
            "builder_right": lambda pl, pr, rng: (pl.copy(), pr.copy()),
            "normalize": True,
        },
        "wrong_points": {
            "description": "Duplicate correspondences and inject an equally-sized set of random mismatches.",
            "builder": wrong_points_builder(img_left.shape, img_middle.shape),
            "builder_right": wrong_points_builder(img_right.shape, img_middle.shape),
            "normalize": True,
        },
        "noise_sigma_1": {
            "description": "Add Gaussian noise (sigma=1 px) to correspondences.",
            "builder": noisy_builder(1.0),
            "builder_right": noisy_builder(1.0),
            "normalize": True,
        },
        "noise_sigma_3": {
            "description": "Add Gaussian noise (sigma=3 px) to correspondences.",
            "builder": noisy_builder(3.0),
            "builder_right": noisy_builder(3.0),
            "normalize": True,
        },
        "noise_sigma_5": {
            "description": "Add Gaussian noise (sigma=5 px) to correspondences.",
            "builder": noisy_builder(5.0),
            "builder_right": noisy_builder(5.0),
            "normalize": True,
        },
        "no_normalization": {
            "description": "All correspondences without point normalization.",
            "builder": lambda pl, pr, rng: (pl.copy(), pr.copy()),
            "builder_right": lambda pl, pr, rng: (pl.copy(), pr.copy()),
            "normalize": False,
        },
    }

    summary_lines = []

    for name, cfg in scenarios.items():
        builder: ScenarioBuilder = cfg["builder"]  # type: ignore
        builder_right: ScenarioBuilder = cfg.get("builder_right", builder)  # type: ignore
        normalize = cfg["normalize"]
        desc = cfg["description"]
        scenario_dir = OUTPUT_DIR / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        try:
            pts_left, pts_middle_from_left = builder(base_left_points, base_middle_from_left, rng)
            H_left_to_middle = computeH(pts_left, pts_middle_from_left, normalize=normalize)

            pts_right, pts_middle_from_right = builder_right(
                base_right_points, base_middle_from_right, rng
            )
            H_right_to_middle = computeH(pts_right, pts_middle_from_right, normalize=normalize)

            panorama = compute_panorama(
                img_left,
                img_middle,
                img_right,
                H_left_to_middle,
                H_right_to_middle,
            )

            panorama_path = scenario_dir / f"{name}.jpg"
            save_image(panorama, panorama_path)
            np.save(
                scenario_dir / f"{name}_left_points.npy",
                np.stack([pts_left, pts_middle_from_left], axis=0),
            )
            np.save(
                scenario_dir / f"{name}_right_points.npy",
                np.stack([pts_right, pts_middle_from_right], axis=0),
            )

            mean_err, max_err = reprojection_error(H_left_to_middle, pts_left, pts_middle_from_left)
            summary_lines.append(
                f"{name}: {desc}\n"
                f"  left_points={len(pts_left)}, right_points={len(pts_right)}, normalize={normalize}\n"
                f"  reprojection mean={mean_err:.2f}px, max={max_err:.2f}px\n"
                f"  output={panorama_path}\n"
            )
        except Exception as exc:  # pylint: disable=broad-except
            summary_lines.append(
                f"{name}: FAILED ({desc})\n  Error: {exc}\n"
            )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines))
    print("Experiments complete. Summary written to output/correspondence_effects/summary.txt")


if __name__ == "__main__":
    run()



