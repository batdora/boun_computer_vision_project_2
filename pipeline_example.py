"""
Example pipeline demonstrating the full image stitching workflow.
Shows how to load images, compute homographies, warp, and blend into a panorama.
"""

import sys
from pathlib import Path

try:
    import numpy as np  # type: ignore
except ImportError as exc:
    raise ImportError("numpy is required to run pipeline_example.py") from exc

try:
    from PIL import Image  # type: ignore
except ImportError:
    Image = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import image I/O functions directly to avoid cv2 dependency issue
try:
    from utils.image_io import load_image, bgr_to_rgb, rgb_to_bgr
except ImportError:
    # Fallback: define basic functions if cv2 not available
    if Image is None:
        raise ImportError("PIL is required when utils.image_io is unavailable.")

    def load_image(path):
        return np.array(Image.open(path).convert("RGB"))

    def bgr_to_rgb(img):
        return img

    def rgb_to_bgr(img):
        return img
# Import functions directly to avoid utils.__init__ importing image_io
import importlib.util
spec = importlib.util.spec_from_file_location("point_selection", Path(__file__).parent / "utils" / "point_selection.py")
point_selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(point_selection)
load_correspondences = point_selection.load_correspondences

from src.homography import computeH
from src.warping import warp, get_warp_bounds
from src.blending import blend_images, create_panorama

# Import math_utils directly to avoid cv2 dependency through utils.__init__
import importlib.util
spec_math = importlib.util.spec_from_file_location("math_utils", Path(__file__).parent / "utils" / "math_utils.py")
math_utils = importlib.util.module_from_spec(spec_math)
spec_math.loader.exec_module(math_utils)
to_homogeneous = math_utils.to_homogeneous


def load_image_rgb(image_path):
    """Load an image from disk and return an RGB numpy array."""
    image_path = Path(image_path)
    if Image is not None:
        try:
            return np.array(Image.open(image_path).convert("RGB"))
        except Exception:
            pass

    image = load_image(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    if image.ndim == 3 and image.shape[2] == 3:
        return bgr_to_rgb(image)
    return image


def get_stem(path_like):
    """Return lowercase stem for either a filesystem path or raw string."""
    return Path(path_like).stem.lower()


def ensure_correspondence_direction(corr_path, desired_src_stem, desired_dst_stem):
    """
    Load correspondences and reorder them so that points1 matches desired_src_stem
    and points2 matches desired_dst_stem.
    """
    desired_src_stem = desired_src_stem.lower()
    desired_dst_stem = desired_dst_stem.lower()

    points1, points2, metadata = load_correspondences(corr_path)

    img1_stem = None
    img2_stem = None
    if metadata:
        img1_stem = metadata.get("image1_name")
        img2_stem = metadata.get("image2_name")
        if img1_stem:
            img1_stem = get_stem(img1_stem)
        if img2_stem:
            img2_stem = get_stem(img2_stem)

    if img1_stem and img2_stem:
        if img1_stem == desired_src_stem and img2_stem == desired_dst_stem:
            return points1, points2, metadata
        if img1_stem == desired_dst_stem and img2_stem == desired_src_stem:
            return points2, points1, metadata

    # Fallback: assume file order is already correct
    if not (img1_stem and img2_stem):
        print(f"  [Info] Metadata not found for {corr_path}. Assuming stored order matches desired direction.")
    else:
        print(
            f"  [Warning] Correspondence order ({img1_stem}->{img2_stem}) "
            f"does not match requested ({desired_src_stem}->{desired_dst_stem}). "
            "Using stored order."
        )

    return points1, points2, metadata


def compute_canvas_and_offsets(bounds_list):
    """Given list of (min_x, min_y, max_x, max_y), compute global canvas and offsets."""
    all_min_x = [b[0] for b in bounds_list]
    all_min_y = [b[1] for b in bounds_list]
    all_max_x = [b[2] for b in bounds_list]
    all_max_y = [b[3] for b in bounds_list]

    global_min_x = min(all_min_x)
    global_min_y = min(all_min_y)
    global_max_x = max(all_max_x)
    global_max_y = max(all_max_y)

    offsets = [(b[0] - global_min_x, b[1] - global_min_y) for b in bounds_list]
    canvas_w = global_max_x - global_min_x
    canvas_h = global_max_y - global_min_y

    return canvas_w, canvas_h, offsets


def ensure_uint8(image):
    """Convert image to uint8 if needed (with clipping)."""
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def find_correspondence_file(stem_a, stem_b, search_dir):
    """
    Look for correspondence file between stem_a and stem_b inside search_dir.
    Returns Path or None.
    """
    stem_a = stem_a.lower()
    stem_b = stem_b.lower()
    candidates = [
        Path(search_dir) / f"{stem_a}_{stem_b}_correspondences.npy",
        Path(search_dir) / f"{stem_b}_{stem_a}_correspondences.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def stitch_anchor_triplet(
    left_img_path,
    middle_img_path,
    right_img_path,
    corr_left_middle=None,
    corr_middle_right=None,
    output_path=None,
):
    """
    Stitch three images provided in left→middle→right order using the middle image as anchor.
    Returns the blended panorama (numpy array).
    """
    left_img_path = Path(left_img_path)
    middle_img_path = Path(middle_img_path)
    right_img_path = Path(right_img_path)

    print("=" * 70)
    print("ANCHOR-BASED TRIPLET STITCHING")
    print("=" * 70)

    print("\n1. Loading images...")
    left_img = load_image_rgb(left_img_path)
    middle_img = load_image_rgb(middle_img_path)
    right_img = load_image_rgb(right_img_path)
    print(
        f"   Loaded shapes:"
        f"\n     Left   ({left_img_path.name}): {left_img.shape}"
        f"\n     Middle ({middle_img_path.name}): {middle_img.shape}"
        f"\n     Right  ({right_img_path.name}): {right_img.shape}"
    )

    left_stem = left_img_path.stem
    middle_stem = middle_img_path.stem
    right_stem = right_img_path.stem

    # Locate correspondence files if not provided
    corr_dir = middle_img_path.parent / "correspondences"
    if corr_left_middle is None:
        corr_left_middle = find_correspondence_file(left_stem, middle_stem, corr_dir)
    if corr_middle_right is None:
        corr_middle_right = find_correspondence_file(middle_stem, right_stem, corr_dir)

    if corr_left_middle is None or corr_middle_right is None:
        missing = []
        if corr_left_middle is None:
            missing.append(f"{left_stem}_{middle_stem}")
        if corr_middle_right is None:
            missing.append(f"{middle_stem}_{right_stem}")
        raise FileNotFoundError(
            f"Correspondence file(s) not found for: {', '.join(missing)}. "
            "Please provide paths explicitly."
        )

    print("\n2. Loading point correspondences...")
    points_left, points_middle_from_left, _ = ensure_correspondence_direction(
        corr_left_middle, left_stem, middle_stem
    )
    points_right, points_middle_from_right, _ = ensure_correspondence_direction(
        corr_middle_right, right_stem, middle_stem
    )
    print(
        f"   Loaded {len(points_left)} points for {left_stem}↔{middle_stem} and "
        f"{len(points_right)} points for {middle_stem}↔{right_stem}"
    )

    print("\n3. Computing homographies (toward anchor)...")
    H_left_to_middle = computeH(points_left, points_middle_from_left)
    H_right_to_middle = computeH(points_right, points_middle_from_right)

    print("\n4. Warping left and right images toward the anchor...")
    warped_left = warp(left_img, H_left_to_middle)
    warped_right = warp(right_img, H_right_to_middle)
    warped_middle = middle_img
    print(
        f"   Warped left shape: {warped_left.shape}\n"
        f"   Anchor middle shape: {warped_middle.shape}\n"
        f"   Warped right shape: {warped_right.shape}"
    )

    print("\n5. Computing canvas and offsets...")
    bounds_left = get_warp_bounds(left_img, H_left_to_middle)
    bounds_middle = (0, 0, middle_img.shape[1], middle_img.shape[0])
    bounds_right = get_warp_bounds(right_img, H_right_to_middle)

    canvas_w, canvas_h, offsets = compute_canvas_and_offsets(
        [bounds_left, bounds_middle, bounds_right]
    )
    offset_left, offset_middle, offset_right = offsets
    print(
        f"   Canvas size: {canvas_w} x {canvas_h}\n"
        f"   Offsets - Left: {offset_left}, Middle: {offset_middle}, Right: {offset_right}"
    )

    print("\n6. Blending images (maximum intensity)...")
    panorama = blend_images(
        [
            ensure_uint8(warped_left),
            ensure_uint8(warped_middle),
            ensure_uint8(warped_right),
        ],
        offsets=[offset_left, offset_middle, offset_right],
        canvas_size=(canvas_w, canvas_h),
    )
    print(f"   Panorama shape: {panorama.shape}")

    if output_path is None:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{middle_stem}_anchor_panorama.jpg"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"\n7. Saving panorama to {output_path} ...")
    try:
        panorama_bgr = rgb_to_bgr(panorama)
        import cv2  # type: ignore

        cv2.imwrite(str(output_path), panorama_bgr)
    except Exception:
        if Image is None:
            raise
        Image.fromarray(ensure_uint8(panorama)).save(output_path)

    print(f"\n✓ Anchor panorama saved to: {output_path}\n")
    return panorama, output_path


def stitch_pairwise(
    img1_path,
    img2_path,
    direction="ltr",
    corr_file=None,
    output_path=None,
):
    """
    Stitch two images with a specified direction.

    Args:
        img1_path: Path to the first image.
        img2_path: Path to the second image.
        direction: 'ltr' to warp img1 onto img2, 'rtl' to warp img2 onto img1.
        corr_file: Optional correspondence file; if None, derived automatically.
        output_path: Optional output image path.
    """
    direction = direction.lower()
    if direction not in {"ltr", "rtl"}:
        raise ValueError("direction must be either 'ltr' or 'rtl'")

    img1_path = Path(img1_path)
    img2_path = Path(img2_path)

    print("=" * 70)
    print(f"PAIRWISE STITCHING ({direction.upper()})")
    print("=" * 70)
    print(f"\n1. Loading images...\n   {img1_path}\n   {img2_path}")

    img1 = load_image_rgb(img1_path)
    img2 = load_image_rgb(img2_path)
    print(f"   Shapes: {img1.shape}, {img2.shape}")

    if direction == "ltr":
        src_path, dst_path = img1_path, img2_path
        src_img, dst_img = img1, img2
    else:
        src_path, dst_path = img2_path, img1_path
        src_img, dst_img = img2, img1

    src_stem = src_path.stem
    dst_stem = dst_path.stem

    if corr_file is None:
        # Try to derive from shared directory
        if src_path.parent == dst_path.parent:
            corr_dir = src_path.parent / "correspondences"
            corr_file = find_correspondence_file(src_stem, dst_stem, corr_dir)
        else:
            corr_file = None

    if corr_file is None:
        raise FileNotFoundError(
            "Correspondence file not provided and could not be inferred."
        )

    print(f"\n2. Loading correspondences from: {corr_file}")
    points_src, points_dst, _ = ensure_correspondence_direction(
        corr_file, src_stem, dst_stem
    )
    print(f"   Loaded {len(points_src)} point pairs")

    print("\n3. Computing homography...")
    H_src_to_dst = computeH(points_src, points_dst)

    print("\n4. Warping source image toward destination...")
    warped_src = warp(src_img, H_src_to_dst)
    print(f"   Warped source shape: {warped_src.shape}")

    print("\n5. Computing canvas and offsets...")
    bounds_src = get_warp_bounds(src_img, H_src_to_dst)
    bounds_dst = (0, 0, dst_img.shape[1], dst_img.shape[0])
    canvas_w, canvas_h, offsets = compute_canvas_and_offsets([bounds_src, bounds_dst])
    print(f"   Canvas size: {canvas_w} x {canvas_h}")
    print(f"   Offsets - source: {offsets[0]}, destination: {offsets[1]}")

    print("\n6. Blending images (maximum intensity)...")
    panorama = blend_images(
        [ensure_uint8(warped_src), ensure_uint8(dst_img)],
        offsets=offsets,
        canvas_size=(canvas_w, canvas_h),
    )
    print(f"   Panorama shape: {panorama.shape}")

    if output_path is None:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / (
            f"{src_stem}_{dst_stem}_{direction}_panorama.jpg"
        )
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"\n7. Saving panorama to {output_path} ...")
    try:
        panorama_bgr = rgb_to_bgr(panorama)
        import cv2  # type: ignore

        cv2.imwrite(str(output_path), panorama_bgr)
    except Exception:
        if Image is None:
            raise
        Image.fromarray(ensure_uint8(panorama)).save(output_path)

    print(f"\n✓ Pairwise panorama saved to: {output_path}\n")
    return panorama, output_path


def run_paris_demo():
    """Convenience demo using the Paris dataset with the anchor workflow."""
    return stitch_anchor_triplet(
        "images/paris/paris_a.jpg",
        "images/paris/paris_b.jpg",
        "images/paris/paris_c.jpg",
        corr_left_middle="images/paris/correspondences/paris_a_paris_b_correspondences.npy",
        corr_middle_right="images/paris/correspondences/paris_b_paris_c_correspondences.npy",
        output_path="output/paris_panorama.jpg",
    )


def print_usage():
    print(
        "Usage:\n"
        "  python pipeline_example.py demo\n"
        "      Run the Paris anchor-based demo.\n\n"
        "  python pipeline_example.py anchor <left_img> <middle_img> <right_img> "
        "[left_middle_corr] [middle_right_corr] [output]\n"
        "      Stitch three images using the middle one as anchor.\n\n"
        "  python pipeline_example.py pair <img_a> <img_b> [ltr|rtl] [corr_file] [output]\n"
        "      Stitch two images pairwise. Direction 'ltr' warps img_a -> img_b (default), "
        "'rtl' warps img_b -> img_a.\n"
    )


def main():
    """
    Main function to run pipeline examples.
    """
    args = sys.argv[1:]

    if not args:
        print_usage()
        return

    mode = args[0].lower()

    if mode == "demo":
        run_paris_demo()
        return

    if mode == "anchor":
        if len(args) < 4:
            print("Error: anchor mode requires at least three image paths.")
            print_usage()
            return
        left_img, middle_img, right_img = args[1:4]
        corr_left_middle = args[4] if len(args) > 4 else None
        corr_middle_right = args[5] if len(args) > 5 else None
        output_path = args[6] if len(args) > 6 else None
        stitch_anchor_triplet(
            left_img,
            middle_img,
            right_img,
            corr_left_middle=corr_left_middle,
            corr_middle_right=corr_middle_right,
            output_path=output_path,
        )
        return

    if mode == "pair":
        if len(args) < 3:
            print("Error: pair mode requires at least two image paths.")
            print_usage()
            return
        img_a = args[1]
        img_b = args[2]
        idx = 3
        direction = "ltr"
        if len(args) > idx and args[idx].lower() in {"ltr", "rtl"}:
            direction = args[idx].lower()
            idx += 1
        corr_file = args[idx] if len(args) > idx else None
        output_path = args[idx + 1] if len(args) > idx + 1 else None
        stitch_pairwise(
            img_a,
            img_b,
            direction=direction,
            corr_file=corr_file,
            output_path=output_path,
        )
        return

    print(f"Unknown mode: {mode}\n")
    print_usage()


if __name__ == "__main__":
    main()

