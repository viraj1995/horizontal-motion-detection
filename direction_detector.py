"""
Sobel X Edge-based Direction Detector

This module provides a robust alternative to optical flow for detecting
camera ego-motion direction at low framerates with low-texture images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class DirectionDetector:
    """
    Detects horizontal camera direction using 1D edge displacement.
    
    Instead of full optical flow, we:
    1. Apply Sobel X to extract vertical edges
    2. Compute weighted centroid of edge intensity per row
    3. Compare centroids between frames to determine shift direction
    """
    
    def __init__(self, edge_threshold: int = 30, debug: bool = False):
        """
        Args:
            edge_threshold: Minimum edge intensity to consider
            debug: Enable debug visualizations
        """
        self.edge_threshold = edge_threshold
        self.debug = debug
        self.prev_centroid = None
        
    def _apply_sobel_x(self, gray: np.ndarray) -> np.ndarray:
        """Apply Sobel X filter to enhance vertical edges."""
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.abs(sobel_x)
        # Normalize to 0-255
        sobel_x = (sobel_x / sobel_x.max() * 255).astype(np.uint8) if sobel_x.max() > 0 else sobel_x.astype(np.uint8)
        return sobel_x
    
    def _compute_edge_centroid(self, edge_img: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute weighted centroid of edge positions.
        
        Returns:
            centroid_x: Weighted average x-position of edges
            row_centroids: Per-row centroids for analysis
        """
        h, w = edge_img.shape
        row_centroids = []
        
        for y in range(h):
            row = edge_img[y, :]
            # Threshold to get significant edges
            significant = row > self.edge_threshold
            
            if significant.sum() > 0:
                # Weighted average x-position
                x_positions = np.arange(w)[significant]
                weights = row[significant]
                centroid = np.average(x_positions, weights=weights)
            else:
                centroid = w / 2  # Default to center
                
            row_centroids.append(centroid)
        
        row_centroids = np.array(row_centroids)
        
        # Overall weighted centroid (more weight to edges with higher intensity)
        if edge_img.max() > 0:
            # Normalize edge image
            edge_norm = edge_img.astype(float) / 255.0
            total_weight = edge_norm.sum()
            if total_weight > 0:
                y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                centroid_x = (x_coords * edge_norm).sum() / total_weight
            else:
                centroid_x = w / 2
        else:
            centroid_x = w / 2
            
        return centroid_x, row_centroids
    
    def detect_direction(self, frame1: np.ndarray, frame2: np.ndarray) -> str:
        """
        Detect camera direction between two frames.
        
        Args:
            frame1: First frame (BGR or grayscale)
            frame2: Second frame (BGR or grayscale)
            
        Returns:
            'left' - Camera moving left (edges shift right)
            'right' - Camera moving right (edges shift left)  
            'stationary' - No significant movement detected
            'unknown' - Cannot determine (insufficient edges)
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2
            
        # Resize for consistency (optional, speeds up processing)
        h, w = gray1.shape
        scale = min(320 / w, 240 / h)
        if scale < 1:
            gray1 = cv2.resize(gray1, (int(w * scale), int(h * scale)))
            gray2 = cv2.resize(gray2, (int(w * scale), int(h * scale)))
        
        # Apply Sobel X
        edge1 = self._apply_sobel_x(gray1)
        edge2 = self._apply_sobel_x(gray2)
        
        # Compute centroids
        centroid1, row_centroids1 = self._compute_edge_centroid(edge1)
        centroid2, row_centroids2 = self._compute_edge_centroid(edge2)
        
        # Compute shift
        shift = centroid2 - centroid1
        
        # Determine direction
        # If edges move RIGHT, camera is moving LEFT (and vice versa)
        shift_threshold = 2.0  # pixels
        
        if abs(shift) < shift_threshold:
            direction = 'stationary'
        elif shift > 0:
            # Edges shifted right → camera moved left
            direction = 'left'
        else:
            # Edges shifted left → camera moved right  
            direction = 'right'
        
        if self.debug:
            print(f"Centroid1: {centroid1:.1f}, Centroid2: {centroid2:.1f}, Shift: {shift:.1f} → {direction}")
            
        self.prev_centroid = centroid2
        return direction
    
    def detect_direction_from_video(self, frames: list, skip: int = 0) -> list:
        """
        Detect direction across multiple frames.
        
        Args:
            frames: List of video frames
            skip: Number of frames to skip between comparisons
            
        Returns:
            List of detected directions
        """
        directions = []
        
        for i in range(0, len(frames) - 1 - skip, 1 + skip):
            direction = self.detect_direction(frames[i], frames[i + 1 + skip])
            directions.append(direction)
            
        return directions


def demo_with_images(img1_path: str, img2_path: str):
    """Demo using two image files."""
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not load images")
        return
    
    detector = DirectionDetector(debug=True)
    direction = detector.detect_direction(img1, img2)
    print(f"\n=== Result ===")
    print(f"Camera direction: {direction}")


def demo_with_video(video_path: str, sample_every_n: int = 5):
    """Demo using a video file."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    detector = DirectionDetector(debug=True)
    directions = []
    prev_frame = None
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if prev_frame is not None and frame_count % sample_every_n == 0:
            direction = detector.detect_direction(prev_frame, frame)
            directions.append(direction)
            print(f"Frame {frame_count}: {direction}")
            
        prev_frame = frame
        frame_count += 1
        
    cap.release()
    
    # Summary
    from collections import Counter
    counts = Counter(directions)
    print(f"\n=== Summary ===")
    for direction, count in counts.most_common():
        print(f"{direction}: {count} ({100*count/len(directions):.1f}%)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        demo_with_images(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        demo_with_video(sys.argv[1])
    else:
        print("Usage:")
        print("  python direction_detector.py <image1> <image2>  - Compare two images")
        print("  python direction_detector.py <video.mp4>        - Process video")
