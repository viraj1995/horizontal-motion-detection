"""
Sobel X edge-based direction detector.

Detects camera ego-motion direction by tracking horizontal edge
centroid displacement between frames. Works at low framerates
with low-texture images where optical flow fails.
"""

import cv2
import numpy as np
from typing import Tuple


class DirectionDetector:
    """Detects horizontal camera direction using 1D edge displacement."""

    def __init__(self, edge_threshold: int = 30, debug: bool = False):
        self.edge_threshold = edge_threshold
        self.debug = debug
        self.prev_centroid = None

    def _apply_sobel_x(self, gray: np.ndarray) -> np.ndarray:
        """Apply Sobel X filter to enhance vertical edges."""
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.abs(sobel_x)
        sobel_x = (sobel_x / sobel_x.max() * 255).astype(np.uint8) if sobel_x.max() > 0 else sobel_x.astype(np.uint8)
        return sobel_x

    def _compute_edge_centroid(self, edge_img: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute weighted centroid of edge positions."""
        h, w = edge_img.shape
        row_centroids = []

        for y in range(h):
            row = edge_img[y, :]
            significant = row > self.edge_threshold
            if significant.sum() > 0:
                x_positions = np.arange(w)[significant]
                weights = row[significant]
                centroid = np.average(x_positions, weights=weights)
            else:
                centroid = w / 2
            row_centroids.append(centroid)

        row_centroids = np.array(row_centroids)

        if edge_img.max() > 0:
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
        """Detect camera direction between two frames.

        Returns:
            'left', 'right', 'stationary', or 'unknown'
        """
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2

        h, w = gray1.shape
        scale = min(320 / w, 240 / h)
        if scale < 1:
            gray1 = cv2.resize(gray1, (int(w * scale), int(h * scale)))
            gray2 = cv2.resize(gray2, (int(w * scale), int(h * scale)))

        edge1 = self._apply_sobel_x(gray1)
        edge2 = self._apply_sobel_x(gray2)

        centroid1, _ = self._compute_edge_centroid(edge1)
        centroid2, _ = self._compute_edge_centroid(edge2)

        shift = centroid2 - centroid1
        shift_threshold = 2.0

        if abs(shift) < shift_threshold:
            direction = 'stationary'
        elif shift > 0:
            direction = 'left'
        else:
            direction = 'right'

        if self.debug:
            print(f"Centroid1: {centroid1:.1f}, Centroid2: {centroid2:.1f}, Shift: {shift:.1f} -> {direction}")

        self.prev_centroid = centroid2
        return direction

    def detect_direction_from_video(self, frames: list, skip: int = 0) -> list:
        """Detect direction across multiple frames."""
        directions = []
        for i in range(0, len(frames) - 1 - skip, 1 + skip):
            direction = self.detect_direction(frames[i], frames[i + 1 + skip])
            directions.append(direction)
        return directions
