"""
Main script for selecting point correspondences between image pairs.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import load_image, bgr_to_rgb
from utils.point_selection import select_correspondences, save_correspondences


def get_output_filename(img1_path, img2_path, output_dir=None):
    """
    Generate output filename based on input image names.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_dir: Optional output directory
        
    Returns:
        Path object for output file
    """
    img1_path = Path(img1_path)
    img2_path = Path(img2_path)
    
    # Get image names without extensions
    img1_name = img1_path.stem
    img2_name = img2_path.stem
    
    # Create output filename: img1_img2_correspondences.npy
    output_filename = f"{img1_name}_{img2_name}_correspondences.npy"
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / output_filename
    else:
        # Save in the same directory as the first image
        return img1_path.parent / output_filename


def main():
    """
    Main function to select correspondences between two images.
    
    Usage:
        python src/correspondence_selector.py <image1_path> <image2_path> [num_points] [output_dir]
    """
    if len(sys.argv) < 3:
        print("Usage: python correspondence_selector.py <image1_path> <image2_path> [num_points] [output_dir]")
        print("\nExample:")
        print("  python src/correspondence_selector.py images/paris/paris_a.jpg images/paris/paris_b.jpg")
        print("  python src/correspondence_selector.py images/paris/paris_a.jpg images/paris/paris_b.jpg 6")
        print("  python src/correspondence_selector.py images/paris/paris_a.jpg images/paris/paris_b.jpg 4 correspondences/")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    n_points = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    output_dir = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Load images
    print(f"Loading images...")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    
    img1_bgr = load_image(img1_path)
    img2_bgr = load_image(img2_path)
    
    # Convert to RGB for matplotlib
    img1_rgb = bgr_to_rgb(img1_bgr)
    img2_rgb = bgr_to_rgb(img2_bgr)
    
    # Get image names for display
    img1_name = Path(img1_path).name
    img2_name = Path(img2_path).name
    
    # Select correspondences
    try:
        points1, points2 = select_correspondences(
            img1_rgb, 
            img2_rgb, 
            n_points=n_points,
            title1=img1_name,
            title2=img2_name
        )
    except KeyboardInterrupt:
        print("\n\nSelection cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during point selection: {e}")
        sys.exit(1)
    
    # Generate output filename
    output_path = get_output_filename(img1_path, img2_path, output_dir)
    
    # Save correspondences
    save_correspondences(
        points1, 
        points2, 
        output_path,
        img1_name=img1_name,
        img2_name=img2_name
    )
    
    print(f"\nDone! Correspondences saved to: {output_path}")


if __name__ == "__main__":
    main()

