"""
Example usage of the correspondence selection tool.

This demonstrates how to use the point correspondence selection functionality.
"""

from pathlib import Path
from utils.image_io import load_image, bgr_to_rgb
from utils.point_selection import select_correspondences, save_correspondences


def example_basic_usage():
    """
    Basic example of selecting correspondences between two images.
    """
    # Define image paths
    img1_path = Path("images/paris/paris_a.jpg")
    img2_path = Path("images/paris/paris_b.jpg")
    
    # Load images
    img1_bgr = load_image(img1_path)
    img2_bgr = load_image(img2_path)
    
    # Convert to RGB for matplotlib
    img1_rgb = bgr_to_rgb(img1_bgr)
    img2_rgb = bgr_to_rgb(img2_bgr)
    
    # Select 4 corresponding points (minimum for homography)
    points1, points2 = select_correspondences(
        img1_rgb, 
        img2_rgb, 
        n_points=4,
        title1="Paris A",
        title2="Paris B"
    )
    
    # Generate output filename
    output_path = Path("correspondences") / f"{img1_path.stem}_{img2_path.stem}_correspondences.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Save correspondences
    save_correspondences(
        points1, 
        points2, 
        output_path,
        img1_name=img1_path.name,
        img2_name=img2_path.name
    )


if __name__ == "__main__":
    print("Example: Basic correspondence selection")
    print("=" * 50)
    example_basic_usage()

