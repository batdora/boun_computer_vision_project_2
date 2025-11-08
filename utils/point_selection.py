"""
Manual point correspondence selection utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json


class PointSelector:
    """Helper class to manage interactive point selection."""
    
    def __init__(self, ax1, ax2, img1, img2, title1, title2, n_points):
        self.ax1 = ax1
        self.ax2 = ax2
        self.img1 = img1
        self.img2 = img2
        self.title1 = title1
        self.title2 = title2
        self.n_points = n_points
        
        self.points1 = []
        self.points2 = []
        self.current_phase = 1  # 1 for image 1, 2 for image 2
        
        # Image dimensions for click detection
        self.img1_shape = img1.shape[:2]
        self.img2_shape = img2.shape[:2]
        
        # Will be set after images are displayed
        self.ax1_limits = None
        self.ax2_limits = None
        
        # Red box for active image
        self.active_box1 = None
        self.active_box2 = None
        
    def is_click_in_ax1(self, x, y):
        """Check if click is within ax1 image bounds."""
        xlim, ylim = self.ax1_limits
        # Matplotlib y-axis is inverted
        return (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1])
    
    def is_click_in_ax2(self, x, y):
        """Check if click is within ax2 image bounds."""
        xlim, ylim = self.ax2_limits
        return (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1])
    
    def update_display(self, phase):
        """Update the display for current selection phase."""
        self.ax1.clear()
        self.ax2.clear()
        
        # Display images
        self.ax1.imshow(self.img1)
        self.ax2.imshow(self.img2)
        
        # Set titles and labels
        if phase == 1:
            self.ax1.set_title(f"{self.title1}\nSelect from this image", 
                             fontsize=14, fontweight='bold', color='red')
            self.ax2.set_title(f"{self.title2}\n(Read-only)", 
                             fontsize=14, color='gray')
            
            # Draw red box around active image (recreate each time, don't try to remove)
            img_width = self.img1.shape[1]
            img_height = self.img1.shape[0]
            self.active_box1 = patches.Rectangle(
                (-0.5, -0.5), img_width + 1, img_height + 1,
                linewidth=4, edgecolor='red', facecolor='none', zorder=10
            )
            self.ax1.add_patch(self.active_box1)
            
            # Draw selected points on image 1 with crosshairs
            for i, point in enumerate(self.points1):
                # Small crosshair
                x, y = point[0], point[1]
                cross_size = 8  # Size of crosshair in pixels
                self.ax1.plot([x - cross_size, x + cross_size], [y, y], 
                             'r-', linewidth=2, zorder=6, alpha=0.8)
                self.ax1.plot([x, x], [y - cross_size, y + cross_size], 
                             'r-', linewidth=2, zorder=6, alpha=0.8)
                # Small circle at center
                self.ax1.plot(x, y, 'ro', markersize=5, markeredgewidth=1, 
                           markeredgecolor='white', markerfacecolor='red', zorder=7)
                # Small number label offset
                self.ax1.annotate(f'{i+1}', (x + cross_size + 5, y - cross_size - 5), 
                                fontsize=10, fontweight='bold',
                                color='white', ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.2', 
                                         facecolor='red', edgecolor='white', 
                                         linewidth=1, alpha=0.9),
                                zorder=8)
                
        else:  # phase == 2
            self.ax1.set_title(f"{self.title1}\n(Read-only)", 
                             fontsize=14, color='gray')
            self.ax2.set_title(f"{self.title2}\nSelect from this image", 
                             fontsize=14, fontweight='bold', color='red')
            
            # Draw red box around active image (recreate each time, don't try to remove)
            img_width = self.img2.shape[1]
            img_height = self.img2.shape[0]
            self.active_box2 = patches.Rectangle(
                (-0.5, -0.5), img_width + 1, img_height + 1,
                linewidth=4, edgecolor='red', facecolor='none', zorder=10
            )
            self.ax2.add_patch(self.active_box2)
            
            # Draw points from image 1 (already selected) - smaller for reference
            for i, point in enumerate(self.points1):
                x, y = point[0], point[1]
                self.ax1.plot(x, y, 'ro', markersize=4, markeredgewidth=1, 
                           markeredgecolor='white', markerfacecolor='red', zorder=5)
                self.ax1.annotate(f'{i+1}', (x + 10, y - 10), 
                                fontsize=8, fontweight='normal',
                                color='white', ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.15', 
                                         facecolor='red', edgecolor='white', 
                                         linewidth=1, alpha=0.7),
                                zorder=6)
            
            # Draw selected points on image 2 with crosshairs
            for i, point in enumerate(self.points2):
                # Small crosshair
                x, y = point[0], point[1]
                cross_size = 8  # Size of crosshair in pixels
                self.ax2.plot([x - cross_size, x + cross_size], [y, y], 
                             'b-', linewidth=2, zorder=6, alpha=0.8)
                self.ax2.plot([x, x], [y - cross_size, y + cross_size], 
                             'b-', linewidth=2, zorder=6, alpha=0.8)
                # Small circle at center
                self.ax2.plot(x, y, 'bo', markersize=5, markeredgewidth=1, 
                           markeredgecolor='white', markerfacecolor='blue', zorder=7)
                # Small number label offset
                self.ax2.annotate(f'{i+1}', (x + cross_size + 5, y - cross_size - 5), 
                                fontsize=10, fontweight='bold',
                                color='white', ha='left', va='top',
                                bbox=dict(boxstyle='round,pad=0.2', 
                                         facecolor='blue', edgecolor='white', 
                                         linewidth=1, alpha=0.9),
                                zorder=8)
        
        self.ax1.axis('off')
        self.ax2.axis('off')
        
        # Update axes limits for click detection
        self.ax1_limits = (self.ax1.get_xlim(), self.ax1.get_ylim())
        self.ax2_limits = (self.ax2.get_xlim(), self.ax2.get_ylim())
        
        plt.tight_layout()
        plt.draw()


def select_correspondences(img1, img2, n_points=4, title1="Image 1", title2="Image 2"):
    """
    Manually select corresponding points between two images.
    Displays both images side by side with visual feedback.
    
    Args:
        img1: First image (RGB format)
        img2: Second image (RGB format)
        n_points: Number of corresponding points to select
        title1: Title for first image window
        title2: Title for second image window
        
    Returns:
        Tuple of (points1, points2) where each is a numpy array of shape (n_points, 2)
        Each point is (x, y) coordinates
    """
    print(f"\n{'='*60}")
    print(f"Select {n_points} corresponding points on each image")
    print(f"{'='*60}\n")
    print(f"Step 1: Click {n_points} points on {title1} (left image)")
    print(f"Step 2: Click {n_points} corresponding points on {title2} (right image)")
    print("\nInstructions:")
    print("  - Points will be numbered as you click")
    print("  - Only the active image (with red border) accepts clicks")
    print("  - Click points in the same order on both images")
    print("\n")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    try:
        fig.canvas.manager.set_window_title('Point Correspondence Selection')
    except AttributeError:
        pass  # Some backends don't support set_window_title
    
    # Create selector object
    selector = PointSelector(ax1, ax2, img1, img2, title1, title2, n_points)
    
    # Storage for click events
    clicked_points_phase1 = []
    clicked_points_phase2 = []
    
    # Phase 1: Select points from image 1
    print(f"\n>>> Phase 1: Select {n_points} points on {title1}")
    selector.update_display(phase=1)
    
    def on_click_phase1(event):
        if event.inaxes == ax1 and event.button == 1:  # Left click on ax1
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                if len(clicked_points_phase1) < n_points:
                    clicked_points_phase1.append([x, y])
                    selector.points1 = clicked_points_phase1
                    print(f"  Point {len(clicked_points_phase1)}/{n_points} selected: ({x:.1f}, {y:.1f})")
                    selector.update_display(phase=1)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
        elif event.inaxes == ax2:
            print("  [Info] Please click on the LEFT image (with red border)")
    
    # Connect event for phase 1
    cid1 = fig.canvas.mpl_connect('button_press_event', on_click_phase1)
    
    # Show and wait for clicks
    plt.show(block=False)
    
    # Wait until we have enough points
    while len(clicked_points_phase1) < n_points:
        plt.pause(0.1)
    
    # Disconnect phase 1 handler
    fig.canvas.mpl_disconnect(cid1)
    
    selector.current_phase = 2
    
    # Phase 2: Select points from image 2
    print(f"\n>>> Phase 2: Select {n_points} corresponding points on {title2}")
    selector.update_display(phase=2)
    
    def on_click_phase2(event):
        if event.inaxes == ax2 and event.button == 1:  # Left click on ax2
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                if len(clicked_points_phase2) < n_points:
                    clicked_points_phase2.append([x, y])
                    selector.points2 = clicked_points_phase2
                    print(f"  Point {len(clicked_points_phase2)}/{n_points} selected: ({x:.1f}, {y:.1f})")
                    selector.update_display(phase=2)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
        elif event.inaxes == ax1:
            print("  [Info] Please click on the RIGHT image (with red border)")
    
    # Connect event for phase 2
    cid2 = fig.canvas.mpl_connect('button_press_event', on_click_phase2)
    
    # Wait until we have enough points
    while len(clicked_points_phase2) < n_points:
        plt.pause(0.1)
    
    # Disconnect and close
    fig.canvas.mpl_disconnect(cid2)
    plt.close(fig)
    
    # Set final points
    selector.points1 = clicked_points_phase1
    selector.points2 = clicked_points_phase2
    
    # Convert to numpy arrays
    if len(selector.points1) < n_points:
        raise ValueError(f"Not enough points selected on {title1}. Expected {n_points}, got {len(selector.points1)}")
    if len(selector.points2) < n_points:
        raise ValueError(f"Not enough points selected on {title2}. Expected {n_points}, got {len(selector.points2)}")
    
    points1 = np.array(selector.points1)
    points2 = np.array(selector.points2)
    
    print(f"\n{'='*60}")
    print(f"Successfully selected {len(points1)} corresponding point pairs!")
    print(f"{'='*60}")
    print(f"\nPoints in {title1}:")
    for i, p in enumerate(points1):
        print(f"  Point {i+1}: ({p[0]:.2f}, {p[1]:.2f})")
    print(f"\nPoints in {title2}:")
    for i, p in enumerate(points2):
        print(f"  Point {i+1}: ({p[0]:.2f}, {p[1]:.2f})")
    print()
    
    return points1, points2


def save_correspondences(points1, points2, output_path, img1_name=None, img2_name=None):
    """
    Save point correspondences to a file using numpy.save.
    
    Args:
        points1: Numpy array of points from first image (n_points, 2)
        points2: Numpy array of points from second image (n_points, 2)
        output_path: Path to save the correspondences file (.npy format)
        img1_name: Name of first image (optional, saved as metadata)
        img2_name: Name of second image (optional, saved as metadata)
    """
    output_path = Path(output_path)
    
    # Ensure .npy extension
    if output_path.suffix != '.npy':
        output_path = output_path.with_suffix('.npy')
    
    # Stack points into single array: shape (2, n_points, 2)
    # First dimension: [points1, points2]
    correspondences = np.array([points1, points2])
    
    # Save using numpy
    np.save(output_path, correspondences)
    
    # Also save metadata as separate file if names provided
    if img1_name or img2_name:
        metadata_path = output_path.with_suffix('.json')
        metadata = {
            'image1_name': img1_name,
            'image2_name': img2_name,
            'num_points': len(points1)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    print(f"\nCorrespondences saved to: {output_path}")


def load_correspondences(file_path):
    """
    Load point correspondences from a numpy file.
    
    Args:
        file_path: Path to the correspondences file (.npy format)
        
    Returns:
        Tuple of (points1, points2) as numpy arrays
    """
    file_path = Path(file_path)
    
    # Try .npy first, then .json (for backward compatibility)
    if file_path.suffix == '.npy':
        correspondences = np.load(file_path)
        # Shape should be (2, n_points, 2)
        points1 = correspondences[0]
        points2 = correspondences[1]
        
        # Try to load metadata if exists
        metadata_path = file_path.with_suffix('.json')
        data = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
        
        return points1, points2, data
    else:
        # Backward compatibility: load from JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        points1 = np.array(data['points1'])
        points2 = np.array(data['points2'])
        
        return points1, points2, data

