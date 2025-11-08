"""
Script to compute all homographies for all image sets.
Handles different strategies:
- Paris: Anchor-based (left→middle, right→middle)
- Others: Sequential composition (left_2→left_1→middle, right_2→right_1→middle)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.homography import (
    computeH,
    compute_homography_from_correspondence_file,
    estimate_homography_for_image_pair,
    apply_homography
)


def compose_homographies(H1, H2):
    """
    Compose two homography matrices: H = H2 @ H1
    This applies H1 first, then H2.
    
    Args:
        H1: First homography matrix (3x3)
        H2: Second homography matrix (3x3)
        
    Returns:
        Composed homography matrix (3x3)
    """
    return H2 @ H1


def compute_paris_homographies(base_dir):
    """
    Compute homographies for Paris images using anchor-based approach.
    
    Args:
        base_dir: Base directory (e.g., images/paris)
        
    Returns:
        Dictionary with homography paths
    """
    base_path = Path(base_dir)
    corr_dir = base_path / "correspondences"
    output_dir = base_path / "homographies"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Computing Paris Homographies (Anchor-based)")
    print("="*60)
    
    # Find correspondence files
    corr_files = list(corr_dir.glob("*_correspondences.json"))
    
    # For Paris, we expect paris_a_paris_b and paris_b_paris_c
    # Assuming paris_b is middle
    paris_a_b = None
    paris_b_c = None
    
    for f in corr_files:
        name = f.stem.lower()
        if 'paris_a' in name and 'paris_b' in name:
            paris_a_b = f
        elif 'paris_b' in name and 'paris_c' in name:
            paris_b_c = f
    
    if not paris_a_b or not paris_b_c:
        print(f"Warning: Could not find expected correspondence files in {corr_dir}")
        print(f"Found files: {[f.name for f in corr_files]}")
        return {}
    
    # Compute left→middle (paris_a → paris_b)
    print(f"\n1. Computing Paris A → Paris B (middle anchor)")
    H_a_to_b, data_a_b = compute_homography_from_correspondence_file(
        paris_a_b, src_to_dst=True
    )
    
    # Compute right→middle (paris_c → paris_b)
    print(f"2. Computing Paris C → Paris B (middle anchor)")
    # Check if file has paris_b→paris_c or paris_c→paris_b
    import json
    with open(paris_b_c, 'r') as f:
        data_b_c = json.load(f)
    img1_name = Path(data_b_c['image1_name']).stem.lower()
    img2_name = Path(data_b_c['image2_name']).stem.lower()
    
    if 'paris_b' in img1_name and 'paris_c' in img2_name:
        # File is paris_b → paris_c, so reverse for paris_c → paris_b
        src_to_dst = False
    else:
        # File is paris_c → paris_b
        src_to_dst = True
    
    H_c_to_b, data_c_b = compute_homography_from_correspondence_file(
        paris_b_c, src_to_dst=src_to_dst
    )
    
    # Save homographies
    np.save(output_dir / "H_paris_a_to_b.npy", H_a_to_b)
    np.save(output_dir / "H_paris_c_to_b.npy", H_c_to_b)
    
    # Save metadata
    import json
    metadata = {
        'strategy': 'anchor-based',
        'anchor': 'paris_b.jpg',
        'homographies': {
            'H_a_to_b': H_a_to_b.tolist(),
            'H_c_to_b': H_c_to_b.tolist()
        }
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✓ Homographies saved to {output_dir}")
    print(f"  - H_paris_a_to_b.npy")
    print(f"  - H_paris_c_to_b.npy")
    
    return {
        'H_a_to_b': str(output_dir / "H_paris_a_to_b.npy"),
        'H_c_to_b': str(output_dir / "H_paris_c_to_b.npy")
    }


def compute_sequential_homographies(base_dir):
    """
    Compute homographies for cmpe_building and north_campus using sequential composition.
    
    Strategy:
    - Left side: left_2 → left_1 → middle
    - Right side: right_2 → right_1 → middle
    
    Args:
        base_dir: Base directory (e.g., images/cmpe_building or images/north_campus)
        
    Returns:
        Dictionary with homography paths
    """
    base_path = Path(base_dir)
    corr_dir = base_path / "correspondences"
    output_dir = base_path / "homographies"
    output_dir.mkdir(exist_ok=True)
    
    dataset_name = base_path.name
    print("\n" + "="*60)
    print(f"Computing {dataset_name} Homographies (Sequential Composition)")
    print("="*60)
    
    # Find correspondence files
    corr_files = {f.stem: f for f in corr_dir.glob("*_correspondences.json")}
    
    # Expected files
    left2_left1 = None
    left1_middle = None
    right2_right1 = None
    right1_middle = None
    
    for name, f in corr_files.items():
        name_lower = name.lower()
        if 'left_2' in name_lower and 'left_1' in name_lower:
            left2_left1 = f
        elif 'left_1' in name_lower and 'middle' in name_lower:
            left1_middle = f
        elif 'right_2' in name_lower and 'right_1' in name_lower:
            right2_right1 = f
        elif 'right_1' in name_lower and 'middle' in name_lower:
            right1_middle = f
    
    if not all([left2_left1, left1_middle, right2_right1, right1_middle]):
        print(f"Warning: Could not find all required correspondence files in {corr_dir}")
        print(f"Found files: {list(corr_files.keys())}")
        print(f"Missing:")
        if not left2_left1: print("  - left_2_to_left_1")
        if not left1_middle: print("  - left_1_to_middle")
        if not right2_right1: print("  - right_2_to_right_1")
        if not right1_middle: print("  - right_1_to_middle")
        return {}
    
    # Left side: left_2 → left_1 → middle
    print(f"\nLeft Side Chain:")
    print(f"  1. Computing left_2 → left_1")
    H_left2_to_left1, _ = compute_homography_from_correspondence_file(
        left2_left1, src_to_dst=True
    )
    
    print(f"  2. Computing left_1 → middle")
    H_left1_to_middle, _ = compute_homography_from_correspondence_file(
        left1_middle, src_to_dst=True
    )
    
    print(f"  3. Composing: left_2 → middle = (left_1→middle) @ (left_2→left_1)")
    H_left2_to_middle = compose_homographies(H_left2_to_left1, H_left1_to_middle)
    
    # Right side: right_2 → right_1 → middle
    print(f"\nRight Side Chain:")
    print(f"  1. Computing right_2 → right_1")
    H_right2_to_right1, _ = compute_homography_from_correspondence_file(
        right2_right1, src_to_dst=True
    )
    
    print(f"  2. Computing right_1 → middle")
    # Check direction
    import json
    with open(right1_middle, 'r') as f:
        data_right1_middle = json.load(f)
    img1_name = Path(data_right1_middle['image1_name']).stem.lower()
    img2_name = Path(data_right1_middle['image2_name']).stem.lower()
    
    if 'middle' in img1_name and 'right_1' in img2_name:
        # File is middle → right_1, so reverse
        src_to_dst = False
    else:
        # File is right_1 → middle
        src_to_dst = True
    
    H_right1_to_middle, _ = compute_homography_from_correspondence_file(
        right1_middle, src_to_dst=src_to_dst
    )
    
    print(f"  3. Composing: right_2 → middle = (right_1→middle) @ (right_2→right_1)")
    H_right2_to_middle = compose_homographies(H_right2_to_right1, H_right1_to_middle)
    
    # Save all homographies
    np.save(output_dir / "H_left2_to_left1.npy", H_left2_to_left1)
    np.save(output_dir / "H_left1_to_middle.npy", H_left1_to_middle)
    np.save(output_dir / "H_left2_to_middle.npy", H_left2_to_middle)
    np.save(output_dir / "H_right2_to_right1.npy", H_right2_to_right1)
    np.save(output_dir / "H_right1_to_middle.npy", H_right1_to_middle)
    np.save(output_dir / "H_right2_to_middle.npy", H_right2_to_middle)
    
    # Save metadata
    metadata = {
        'strategy': 'sequential-composition',
        'anchor': 'middle.jpg',
        'left_chain': ['left_2.jpg', 'left_1.jpg', 'middle.jpg'],
        'right_chain': ['right_2.jpg', 'right_1.jpg', 'middle.jpg'],
        'homographies': {
            'H_left2_to_left1': H_left2_to_left1.tolist(),
            'H_left1_to_middle': H_left1_to_middle.tolist(),
            'H_left2_to_middle': H_left2_to_middle.tolist(),
            'H_right2_to_right1': H_right2_to_right1.tolist(),
            'H_right1_to_middle': H_right1_to_middle.tolist(),
            'H_right2_to_middle': H_right2_to_middle.tolist()
        }
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✓ Homographies saved to {output_dir}")
    print(f"  - H_left2_to_left1.npy")
    print(f"  - H_left1_to_middle.npy")
    print(f"  - H_left2_to_middle.npy (composed)")
    print(f"  - H_right2_to_right1.npy")
    print(f"  - H_right1_to_middle.npy")
    print(f"  - H_right2_to_middle.npy (composed)")
    
    return {
        'H_left2_to_left1': str(output_dir / "H_left2_to_left1.npy"),
        'H_left1_to_middle': str(output_dir / "H_left1_to_middle.npy"),
        'H_left2_to_middle': str(output_dir / "H_left2_to_middle.npy"),
        'H_right2_to_right1': str(output_dir / "H_right2_to_right1.npy"),
        'H_right1_to_middle': str(output_dir / "H_right1_to_middle.npy"),
        'H_right2_to_middle': str(output_dir / "H_right2_to_middle.npy")
    }


def main():
    """Compute homographies for all image sets."""
    images_dir = Path("images")
    
    if not images_dir.exists():
        print(f"Error: {images_dir} directory not found")
        sys.exit(1)
    
    results = {}
    
    # Paris: Anchor-based
    paris_dir = images_dir / "paris"
    if paris_dir.exists():
        results['paris'] = compute_paris_homographies(paris_dir)
    else:
        print(f"Warning: {paris_dir} not found")
    
    # cmpe_building and north_campus: Sequential composition
    for dataset in ['cmpe_building', 'north_campus']:
        dataset_dir = images_dir / dataset
        if dataset_dir.exists():
            results[dataset] = compute_sequential_homographies(dataset_dir)
        else:
            print(f"Warning: {dataset_dir} not found")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for dataset, homographies in results.items():
        print(f"\n{dataset}:")
        if homographies:
            print(f"  ✓ {len(homographies)} homographies computed")
        else:
            print(f"  ✗ Failed to compute homographies")
    
    print("\n" + "="*60)
    print("All homographies computed!")
    print("="*60)


if __name__ == "__main__":
    main()

