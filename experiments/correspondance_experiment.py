"""
Correspondence Effects Experiment

Runs a series of controlled experiments on the Paris A/B image pair to illustrate
how different correspondence choices (count, noise, outliers, normalization)
impact the estimated homography and the final stitched panorama.

Outputs are written under `output/correspondence_effects/`:
  - `<scenario>.jpg` : blended panorama (paris_a warped onto paris_b)
  - `<scenario>_points.npy` : correspondence set used for the scenario
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
CORR_FILE = Path("images/paris/correspondences/paris_a_paris_b_correspondences.npy")


ScenarioBuilder = Callable[[np.ndarray, np.ndarray, np.random.Generator], Tuple[np.ndarray, np.ndarray]]


def load_base_correspondences() -> Tuple[np.ndarray, np.ndarray]:
    corr = np.load(CORR_FILE)
    points_a = np.array(corr[0], dtype=np.float64)
    points_b = np.array(corr[1], dtype=np.float64)
    if len(points_a) < 8:
        raise ValueError(
            "Expected at least 8 correspondence points in the Paris A/B set."
        )
    return points_a, points_b


def random_subset_builder(n: int) -> ScenarioBuilder:
    def builder(
        points_a: np.ndarray, points_b: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(points_a) < n:
            raise ValueError(f"Not enough correspondences for n={n}.")
        indices = rng.choice(len(points_a), size=n, replace=False)
        return points_a[indices].copy(), points_b[indices].copy()

    return builder


def wrong_points_builder(num_wrong: int, img_a_shape, img_b_shape) -> ScenarioBuilder:
    ha, wa = img_a_shape[:2]
    hb, wb = img_b_shape[:2]

    def builder(points_a: np.ndarray, points_b: np.ndarray, rng: np.random.Generator):
        wrong_left = np.column_stack(
            (
                rng.uniform(0, wa, size=num_wrong),
                rng.uniform(0, ha, size=num_wrong),
            )
        )
        wrong_right = np.column_stack(
            (
                rng.uniform(0, wb, size=num_wrong),
                rng.uniform(0, hb, size=num_wrong),
            )
        )
        return (
            np.vstack([points_a.copy(), wrong_left]),
            np.vstack([points_b.copy(), wrong_right]),
        )

    return builder


def noisy_builder(sigma: float) -> ScenarioBuilder:
    def builder(points_a: np.ndarray, points_b: np.ndarray, rng: np.random.Generator):
        noise_a = rng.normal(0.0, sigma, size=points_a.shape)
        noise_b = rng.normal(0.0, sigma, size=points_b.shape)
        return points_a.copy() + noise_a, points_b.copy() + noise_b

    return builder


def compute_panorama(
    img_a: np.ndarray,
    img_b: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    warped_a = ensure_uint8(warp(img_a, H))
    bounds_a = get_warp_bounds(img_a, H)
    bounds_b = (0, 0, img_b.shape[1], img_b.shape[0])
    canvas_w, canvas_h, offsets = compute_canvas_and_offsets([bounds_a, bounds_b])
    panorama = blend_images(
        [warped_a, ensure_uint8(img_b)],
        offsets=offsets,
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

    img_a = load_image_rgb(PARIS_A)
    img_b = load_image_rgb(PARIS_B)
    base_left, base_right = load_base_correspondences()
    total_points = len(base_left)

    scenarios: Dict[str, Dict] = {
        "baseline_all": {
            "description": f"All available correspondences ({total_points}) with normalization.",
            "builder": lambda pl, pr, rng: (pl.copy(), pr.copy()),
            "normalize": True,
        },
        "points_5": {
            "description": "Random 5 correspondence pairs.",
            "builder": random_subset_builder(5),
            "normalize": True,
        },
        "points_max": {
            "description": "Using all available correspondences.",
            "builder": lambda pl, pr, rng: (pl.copy(), pr.copy()),
            "normalize": True,
        },
        "wrong_points": {
            "description": "Inject 5 random incorrect matches.",
            "builder": wrong_points_builder(5, img_a.shape, img_b.shape),
            "normalize": True,
        },
        "noise_sigma_1": {
            "description": "Add Gaussian noise (sigma=1 px) to correspondences.",
            "builder": noisy_builder(1.0),
            "normalize": True,
        },
        "noise_sigma_3": {
            "description": "Add Gaussian noise (sigma=3 px) to correspondences.",
            "builder": noisy_builder(3.0),
            "normalize": True,
        },
        "noise_sigma_5": {
            "description": "Add Gaussian noise (sigma=5 px) to correspondences.",
            "builder": noisy_builder(5.0),
            "normalize": True,
        },
        "no_normalization": {
            "description": "All correspondences without point normalization.",
            "builder": lambda pl, pr, rng: (pl.copy(), pr.copy()),
            "normalize": False,
        },
    }

    summary_lines = []

    for name, cfg in scenarios.items():
        builder: ScenarioBuilder = cfg["builder"]  # type: ignore
        normalize = cfg["normalize"]
        desc = cfg["description"]
        scenario_dir = OUTPUT_DIR / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        try:
            pts_a, pts_b = builder(base_left, base_right, rng)
            H = computeH(pts_a, pts_b, normalize=normalize)
            panorama = compute_panorama(img_a, img_b, H)

            panorama_path = scenario_dir / f"{name}.jpg"
            save_image(panorama, panorama_path)
            np.save(scenario_dir / f"{name}_points.npy", np.stack([pts_a, pts_b], axis=0))

            mean_err, max_err = reprojection_error(H, pts_a, pts_b)
            summary_lines.append(
                f"{name}: {desc}\n"
                f"  points={len(pts_a)}, normalize={normalize}\n"
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


