"""
Main script for image stitching pipeline.
Implements the full workflow: point selection, homography estimation, and image warping.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_io import load_image, bgr_to_rgb, rgb_to_bgr
from utils.point_selection import select_correspondences, save_correspondences, load_correspondences
from src.homography import computeH
from src.warping import warp


def main():
    """
    Main pipeline for image stitching.
    
    Usage:
        python -m experiments.pairwise_demo <image1_path> <image2_path> [num_points] [corr_file]
        
    If corr_file is provided, it will load correspondences from that file.
    Otherwise, it will interactively ask for point selection.
    """
    if len(sys.argv) < 3:
        print("Usage: python main.py <image1_path> <image2_path> [num_points] [corr_file]")
        print("\nExample:")
        print("  python main.py images/paris/paris_a.jpg images/paris/paris_b.jpg")
        print("  python main.py images/paris/paris_a.jpg images/paris/paris_b.jpg 6")
        print("  python main.py images/paris/paris_a.jpg images/paris/paris_b.jpg 4 correspondences.npy")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    n_points = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    corr_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    print("="*60)
    print("Image Stitching Pipeline")
    print("="*60)
    
    # Load images
    print(f"\n1. Loading images...")
    print(f"   Image 1: {img1_path}")
    print(f"   Image 2: {img2_path}")
    
    img1_bgr = load_image(img1_path)
    img2_bgr = load_image(img2_path)
    img1_rgb = bgr_to_rgb(img1_bgr)
    img2_rgb = bgr_to_rgb(img2_bgr)
    
    # Step 1: Point correspondence selection
    print(f"\n2. Point correspondence selection...")
    if corr_file and Path(corr_file).exists():
        print(f"   Loading correspondences from: {corr_file}")
        points_im1, points_im2, _ = load_correspondences(corr_file)
        print(f"   Loaded {len(points_im1)} corresponding points")
    else:
        print(f"   Interactive selection: click {n_points} points on each image")
        points_im1, points_im2 = select_correspondences(
            img1_rgb, 
            img2_rgb, 
            n_points=n_points,
            title1=Path(img1_path).name,
            title2=Path(img2_path).name
        )
        
        # Save correspondences
        if not corr_file:
            corr_file = f"{Path(img1_path).stem}_{Path(img2_path).stem}_correspondences.npy"
        
        save_correspondences(
            points_im1, 
            points_im2, 
            corr_file,
            img1_name=Path(img1_path).name,
            img2_name=Path(img2_path).name
        )
        print(f"   Correspondences saved to: {corr_file}")
    
    # Step 2: Homography estimation
    print(f"\n3. Homography estimation...")
    print(f"   Computing H from {len(points_im1)} point correspondences")
    homography = computeH(points_im1, points_im2)
    print(f"   Homography matrix computed:")
    print(f"   {homography}")
    
    # Step 3: Image warping
    print(f"\n4. Image warping...")
    print(f"   Warping image 1 using homography...")
    image_warped = warp(img1_rgb, homography)
    print(f"   Warped image shape: {image_warped.shape}")
    
    # Save warped image
    output_path = Path("output") / f"{Path(img1_path).stem}_warped.jpg"
    output_path.parent.mkdir(exist_ok=True)
    
    # Save warped image
    # Try OpenCV first, fallback to PIL
    try:
        warped_bgr = rgb_to_bgr(image_warped)
        import cv2
        cv2.imwrite(str(output_path), warped_bgr)
    except ImportError:
        from PIL import Image
        # Ensure values are in valid range
        warped_save = np.clip(image_warped, 0, 255).astype(np.uint8)
        Image.fromarray(warped_save).save(output_path)
    print(f"   Warped image saved to: {output_path}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

