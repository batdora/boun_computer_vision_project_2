"""
Main script for computing homographies from correspondence files.
Handles the anchor-based strategy with middle image as reference.
"""

import sys
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.homography import (
    computeH,
    compute_homography_from_correspondence_file,
    estimate_homography_for_image_pair,
    apply_homography
)


def save_homography(H, output_path, metadata=None):
    """
    Save homography matrix to a file.
    
    Args:
        H: 3x3 homography matrix
        output_path: Path to save the homography (JSON or .npy format)
        metadata: Optional metadata dictionary
    """
    output_path = Path(output_path)
    
    if output_path.suffix == '.npy':
        # Save as NumPy binary format
        np.save(output_path, H)
        if metadata:
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
    else:
        # Save as JSON
        data = {
            'homography': H.tolist(),
            'matrix_shape': list(H.shape),
            'metadata': metadata or {}
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    print(f"Homography saved to: {output_path}")


def load_homography(file_path):
    """
    Load homography matrix from a file.
    
    Args:
        file_path: Path to homography file
        
    Returns:
        3x3 homography matrix
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npy':
        return np.load(file_path)
    else:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return np.array(data['homography'])


def main():
    """
    Main function to compute homographies.
    
    Usage:
        # Single pair
        python src/homography_estimator.py <corr_file> [output_file]
        
        # Three-image setup with middle anchor
        python src/homography_estimator.py --three-images <corr_left_middle> <corr_right_middle> [output_dir]
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single pair:")
        print("    python src/homography_estimator.py <corr_file> [output_file]")
        print("  Three images with middle anchor:")
        print("    python src/homography_estimator.py --three-images <corr_left_middle> <corr_right_middle> [output_dir]")
        sys.exit(1)
    
    import numpy as np
    
    if sys.argv[1] == '--three-images':
        # Three-image setup
        if len(sys.argv) < 4:
            print("Usage: python src/homography_estimator.py --three-images <corr_left_middle> <corr_right_middle> [output_dir]")
            sys.exit(1)
        
        corr_left_middle = sys.argv[2]
        corr_right_middle = sys.argv[3]
        output_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(corr_left_middle).parent
        
        print("Computing homographies with middle image as anchor...")
        print(f"  Left-middle correspondences: {corr_left_middle}")
        print(f"  Right-middle correspondences: {corr_right_middle}")
        
        H_left_to_middle, H_right_to_middle, metadata = estimate_homography_for_image_pair(
            corr_left_middle, corr_right_middle
        )
        
        print(f"\nLeft→Middle Homography:")
        print(H_left_to_middle)
        print(f"\nRight→Middle Homography:")
        print(H_right_to_middle)
        
        # Save homographies
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_homography(H_left_to_middle, output_dir / "H_left_to_middle.json", metadata)
        save_homography(H_right_to_middle, output_dir / "H_right_to_middle.json", metadata)
        
        print("\nHomographies computed successfully!")
        
    else:
        # Single pair
        corr_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else Path(corr_file).with_suffix('.npy')
        
        print(f"Computing homography from: {corr_file}")
        
        H, data = compute_homography_from_correspondence_file(corr_file, src_to_dst=True)
        
        print(f"\nHomography matrix:")
        print(H)
        
        # Verify with a test point
        import json
        with open(corr_file, 'r') as f:
            corr_data = json.load(f)
        points1 = np.array(corr_data['points1'])
        points2 = np.array(corr_data['points2'])
        if len(points1) > 0:
            test_point = points1[0:1]
            transformed = apply_homography(H, test_point)
            print(f"\nVerification (first point):")
            print(f"  Original: {test_point[0]}")
            print(f"  Transformed: {transformed[0]}")
            print(f"  Target: {points2[0]}")
            error = np.linalg.norm(transformed[0] - points2[0])
            print(f"  Error: {error:.2f} pixels")
        
        save_homography(H, output_file, data)
        
        print("\nHomography computed successfully!")


if __name__ == "__main__":
    main()

