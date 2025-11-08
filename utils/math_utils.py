"""
Mathematical utilities for computer vision operations.
"""

import numpy as np


def to_homogeneous(points):
    """
    Convert points to homogeneous coordinates.
    
    Args:
        points: Array of shape (n, 2) containing [x, y] coordinates
        
    Returns:
        Array of shape (n, 3) containing [x, y, 1] coordinates
    """
    n = len(points)
    ones = np.ones((n, 1))
    return np.hstack([points, ones])


def from_homogeneous(points_homog):
    """
    Convert points from homogeneous coordinates to Cartesian.
    
    Args:
        points_homog: Array of shape (n, 3) containing [x, y, w] coordinates
        
    Returns:
        Array of shape (n, 2) containing [x/w, y/w] coordinates
    """
    # Avoid division by zero
    w = points_homog[:, 2:3]
    w[w == 0] = 1.0
    return points_homog[:, :2] / w


def normalize_points(points):
    """
    Normalize points to have zero mean and unit variance for numerical stability.
    
    Args:
        points: Array of shape (n, 2) containing [x, y] coordinates
        
    Returns:
        Tuple of (normalized_points, transformation_matrix)
        where transformation_matrix is 3x3 matrix that normalizes points
    """
    # Compute mean
    mean = np.mean(points, axis=0)
    
    # Center points
    centered = points - mean
    
    # Compute scale (average distance from center)
    # Use sqrt(2) factor for unit average distance
    scale = np.sqrt(2) / (np.mean(np.linalg.norm(centered, axis=1)) + 1e-8)
    
    # Normalization matrix
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    
    # Apply normalization
    points_homog = to_homogeneous(points)
    normalized_homog = (T @ points_homog.T).T
    normalized = from_homogeneous(normalized_homog)
    
    return normalized, T


def denormalize_homography(H_normalized, T1, T2):
    """
    Denormalize a homography matrix computed from normalized points.
    
    Args:
        H_normalized: 3x3 homography matrix computed from normalized points
        T1: Normalization matrix for source points
        T2: Normalization matrix for target points
        
    Returns:
        3x3 denormalized homography matrix
    """
    # H = T2^(-1) @ H_normalized @ T1
    T2_inv = np.linalg.inv(T2)
    H = T2_inv @ H_normalized @ T1
    return H

