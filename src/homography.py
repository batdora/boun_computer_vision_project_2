"""
Homography estimation module.
Computes homography matrices from point correspondences using DLT algorithm.
"""

import numpy as np
import sys
from pathlib import Path

# Import math_utils directly without going through __init__ to avoid cv2 dependency
sys.path.insert(0, str(Path(__file__).parent.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("math_utils", Path(__file__).parent.parent / "utils" / "math_utils.py")
math_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_utils)

normalize_points = math_utils.normalize_points
to_homogeneous = math_utils.to_homogeneous
denormalize_homography = math_utils.denormalize_homography


def computeH(points_im1, points_im2, normalize=True):
    """
    Compute homography matrix using Direct Linear Transform (DLT) algorithm.
    
    Signature matches project requirements: homography = computeH(points_im1, points_im2)
    
    Args:
        points_im1: Points from image 1, array of shape (n, 2)
        points_im2: Points from image 2, array of shape (n, 2)
        normalize: Whether to normalize points for numerical stability (default: True)
        
    Returns:
        3x3 homography matrix H such that points_im2 ≈ H @ points_im1 (homogeneous)
        
    Raises:
        ValueError: If fewer than 4 points provided
    """
    points_src = points_im1
    points_dst = points_im2
    points_src = np.array(points_src)
    points_dst = np.array(points_dst)
    
    if len(points_src) < 4:
        raise ValueError(f"At least 4 points required for homography. Got {len(points_src)}")
    
    if len(points_src) != len(points_dst):
        raise ValueError(f"Number of source and destination points must match. "
                        f"Got {len(points_src)} and {len(points_dst)}")
    
    # Normalize points for numerical stability
    if normalize:
        points_src_norm, T1 = normalize_points(points_src)
        points_dst_norm, T2 = normalize_points(points_dst)
    else:
        points_src_norm = points_src
        points_dst_norm = points_dst
        T1 = np.eye(3)
        T2 = np.eye(3)
    
    # Convert to homogeneous coordinates
    src_homog = to_homogeneous(points_src_norm)
    dst_homog = to_homogeneous(points_dst_norm)
    
    # Build the constraint matrix A
    # For each point pair, we have 2 equations:
    # x' = (H[0,0]*x + H[0,1]*y + H[0,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
    # y' = (H[1,0]*x + H[1,1]*y + H[1,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
    # Rearranging: x'*(H[2,0]*x + H[2,1]*y + H[2,2]) = H[0,0]*x + H[0,1]*y + H[0,2]
    
    n_points = len(points_src_norm)
    A = np.zeros((2 * n_points, 9))
    
    for i in range(n_points):
        x, y = src_homog[i, 0], src_homog[i, 1]
        xp, yp = dst_homog[i, 0], dst_homog[i, 1]
        
        # First equation: x' constraint
        A[2*i, :] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        # Second equation: y' constraint
        A[2*i+1, :] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
    
    # Solve Ah = 0 using SVD
    # The solution is the right singular vector corresponding to the smallest singular value
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # Last row of Vt (corresponds to smallest singular value)
    
    # Reshape to 3x3 matrix
    H = h.reshape(3, 3)
    
    # Denormalize if we normalized
    if normalize:
        H = denormalize_homography(H, T1, T2)
    
    # Normalize by H[2,2] (can be scaled arbitrarily)
    if abs(H[2, 2]) > 1e-8:
        H = H / H[2, 2]
    
    return H


def apply_homography(H, points):
    """
    Apply homography transformation to points.
    
    Args:
        H: 3x3 homography matrix
        points: Points to transform, array of shape (n, 2)
        
    Returns:
        Transformed points, array of shape (n, 2)
    """
    points = np.array(points)
    points_homog = to_homogeneous(points)
    
    # Transform: x' = H @ x
    transformed_homog = (H @ points_homog.T).T
    
    # Convert back to Cartesian
    w = transformed_homog[:, 2:3]
    w[w == 0] = 1.0
    transformed = transformed_homog[:, :2] / w
    
    return transformed


def compute_homography_from_correspondence_file(corr_file, src_to_dst=True):
    """
    Compute homography from a correspondence JSON file.
    
    Args:
        corr_file: Path to correspondence JSON file
        src_to_dst: If True, compute H from points1→points2.
                    If False, compute H from points2→points1.
                    
    Returns:
        Tuple of (H, data) where H is 3x3 homography matrix and data is correspondence metadata
        
    Note:
        Based on the anchor strategy:
        - For left images: compute left→middle (src_to_dst=True with points1=left, points2=middle)
        - For right images: compute right→middle (src_to_dst=False, invert direction)
    """
    # Import here to avoid circular dependencies
    import json
    from pathlib import Path
    
    # Load correspondences directly (avoid importing utils modules that require cv2)
    corr_path = Path(corr_file)
    with open(corr_path, 'r') as f:
        data = json.load(f)
    
    points1 = np.array(data['points1'])
    points2 = np.array(data['points2'])
    
    if src_to_dst:
        # H maps points1 → points2
        H = computeH(points1, points2)
    else:
        # H maps points2 → points1 (reverse direction)
        H = computeH(points2, points1)
    
    return H, data


def determine_correspondence_direction(corr_file, target_anchor):
    """
    Determine the direction of correspondences from filename and content.
    
    Args:
        corr_file: Path to correspondence file
        target_anchor: Name of the anchor image (e.g., "middle.jpg" or "middle")
        
    Returns:
        Boolean: True if points1→points2 goes toward anchor, False if points2→points1 goes toward anchor
    """
    from pathlib import Path
    import json
    
    # Get anchor name without extension
    anchor_name = Path(target_anchor).stem if target_anchor else "middle"
    
    # Load to check image names
    with open(corr_file, 'r') as f:
        data = json.load(f)
    img1_name = Path(data['image1_name']).stem if data.get('image1_name') else ""
    img2_name = Path(data['image2_name']).stem if data.get('image2_name') else ""
    
    # Check if image2 is the anchor (points1→points2 goes toward anchor)
    if anchor_name in img2_name.lower():
        return True  # points1 → points2 (toward anchor)
    elif anchor_name in img1_name.lower():
        return False  # points2 → points1 (reverse, toward anchor)
    else:
        # Fallback: check filename pattern
        corr_file_name = Path(corr_file).stem.lower()
        if corr_file_name.startswith(anchor_name.lower()):
            # Format: middle_xxx, so points1=middle, need reverse
            return False
        else:
            # Format: xxx_middle, so points2=middle, use as is
            return True


def estimate_homography_for_image_pair(corr_left_middle, corr_right_middle, 
                                       middle_image_name="middle.jpg"):
    """
    Estimate homographies for a three-image setup with middle as anchor.
    
    Args:
        corr_left_middle: Path to correspondence file between left and middle
        corr_right_middle: Path to correspondence file between right and middle
        middle_image_name: Name of middle/anchor image (default: "middle.jpg")
        
    Returns:
        Tuple of (H_left_to_middle, H_right_to_middle, metadata)
        
    Strategy:
        - Left to middle: Always compute left→middle
        - Right to middle: Always compute right→middle (middle is always the anchor)
    """
    # For left image: left → middle
    # Determine if correspondence file has left→middle or middle→left
    left_to_middle = determine_correspondence_direction(corr_left_middle, middle_image_name)
    H_left_to_middle, data_left = compute_homography_from_correspondence_file(
        corr_left_middle, src_to_dst=left_to_middle
    )
    
    # For right image: right → middle
    # Determine if correspondence file has right→middle or middle→right
    right_to_middle = determine_correspondence_direction(corr_right_middle, middle_image_name)
    H_right_to_middle, data_right = compute_homography_from_correspondence_file(
        corr_right_middle, src_to_dst=right_to_middle
    )
    
    metadata = {
        'middle_image': middle_image_name,
        'corr_left_middle': str(corr_left_middle),
        'corr_right_middle': str(corr_right_middle),
        'left_direction': 'left→middle' if left_to_middle else 'middle→left (reversed)',
        'right_direction': 'right→middle' if right_to_middle else 'middle→right (reversed)',
        'left_data': data_left,
        'right_data': data_right
    }
    
    return H_left_to_middle, H_right_to_middle, metadata

