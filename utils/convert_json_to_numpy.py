"""
Utility script to convert JSON correspondence files to numpy format.
Converts all existing .json correspondence files to .npy format.

WHY THIS SCRIPT WAS NEEDED:
---------------------------
Originally, correspondence files were saved in JSON format for readability.
However, the project guidelines specify that correspondences should be saved using
numpy.save() and loaded using numpy.load().

This migration script was created to convert all existing JSON correspondence files
to the required numpy format (.npy files), ensuring compatibility with the updated
save/load functions that now use numpy.save() and numpy.load().

The conversion maintains all data integrity while switching from:
  - JSON format: {'points1': [...], 'points2': [...], ...}
  - To numpy format: Array of shape (2, n_points, 2) where [points1, points2]

Note: The load_correspondences() function maintains backward compatibility and can
still load both .json and .npy files, but .npy is now the preferred format.
"""

import json
import numpy as np
from pathlib import Path


def convert_json_to_numpy(json_path, output_path=None):
    """
    Convert a JSON correspondence file to numpy format.
    
    Args:
        json_path: Path to JSON correspondence file
        output_path: Optional output path (default: same location with .npy extension)
        
    Returns:
        Path to the created .npy file
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    if json_path.suffix != '.json':
        raise ValueError(f"Expected .json file, got {json_path.suffix}")
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract points
    points1 = np.array(data['points1'])
    points2 = np.array(data['points2'])
    
    # Determine output path
    if output_path is None:
        # Replace .json with .npy, or add .npy before .json
        if json_path.stem.endswith('_correspondences'):
            output_path = json_path.with_suffix('.npy')
        else:
            output_path = json_path.with_suffix('.npy')
    else:
        output_path = Path(output_path)
        if output_path.suffix != '.npy':
            output_path = output_path.with_suffix('.npy')
    
    # Stack points into numpy array: shape (2, n_points, 2)
    correspondences = np.array([points1, points2])
    
    # Save as numpy file
    np.save(output_path, correspondences)
    
    # Save metadata if it exists in JSON
    if 'image1_name' in data or 'image2_name' in data:
        metadata_path = output_path.with_suffix('.json')
        metadata = {
            'image1_name': data.get('image1_name'),
            'image2_name': data.get('image2_name'),
            'num_points': len(points1)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    return output_path


def convert_all_json_files(base_dir="images"):
    """
    Convert all JSON correspondence files in the images directory to numpy format.
    
    Args:
        base_dir: Base directory to search for JSON files (default: "images")
        
    Returns:
        Dictionary mapping original JSON paths to new numpy paths
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warning: Directory {base_dir} does not exist")
        return {}
    
    # Find all JSON correspondence files
    json_files = list(base_path.rglob("*_correspondences.json"))
    
    if not json_files:
        print(f"No JSON correspondence files found in {base_dir}")
        return {}
    
    print(f"Found {len(json_files)} JSON correspondence files to convert")
    print("="*60)
    
    converted = {}
    
    for json_file in json_files:
        try:
            print(f"Converting: {json_file}")
            numpy_path = convert_json_to_numpy(json_file)
            converted[str(json_file)] = str(numpy_path)
            
            # Verify conversion
            loaded = np.load(numpy_path)
            print(f"  ✓ Saved to: {numpy_path}")
            print(f"    Shape: {loaded.shape}, Points: {loaded.shape[1]}")
            
        except Exception as e:
            print(f"  ✗ Error converting {json_file}: {e}")
    
    print("="*60)
    print(f"Conversion complete! {len(converted)} files converted.")
    
    return converted


def main():
    """Main function to convert all JSON files."""
    import sys
    
    if len(sys.argv) > 1:
        # Convert specific file
        json_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        try:
            numpy_path = convert_json_to_numpy(json_file, output_file)
            print(f"✓ Converted {json_file} to {numpy_path}")
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
    else:
        # Convert all files in images directory
        convert_all_json_files()


if __name__ == "__main__":
    main()

