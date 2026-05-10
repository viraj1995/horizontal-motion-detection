# SPDX-License-Identifier: MIT
"""
Horizontal motion detector.

Detects camera ego-motion direction using either phase correlation
(FFT-based) or Sobel X edge centroid tracking.
"""

import cv2
import numpy as np
from typing import Tuple


class MotionDetector:
    """Detects horizontal camera motion direction."""

    def __init__(self, method: str = "phase_correlation", shift_threshold: float = 1.0, edge_threshold: int = 30, debug: bool = False):
        """
        Args:
            method: 'phase_correlation' or 'sobel_x'
            shift_threshold: minimum pixel shift to register motion
            edge_threshold: minimum edge intensity for sobel_x method
            debug: print intermediate values
        """
        self.method = method
        self.shift_threshold = shift_threshold
        self.edge_threshold = edge_threshold
        self.debug = debug

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale and downscale."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape
        scale = min(320 / w, 240 / h)
        if scale < 1:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        return gray

    def _phase_correlation(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float, float]:
        """Compute translation between two images via phase correlation.

        Returns:
            dx: horizontal shift (positive = img2 shifted right relative to img1)
            dy: vertical shift
            response: peak correlation strength (0-1)
        """
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
        img2 = cv2.GaussianBlur(img2, (5, 5), 0)

        f1 = np.fft.fft2(img1)
        f2 = np.fft.fft2(img2)

        cross_power = f1 * np.conj(f2)
        cross_power = cross_power / (np.abs(cross_power) + 1e-10)

        corr = np.fft.ifft2(cross_power)
        corr = np.abs(corr)

        peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
        h, w = corr.shape

        dx = peak_x if peak_x < w / 2 else peak_x - w
        dy = peak_y if peak_y < h / 2 else peak_y - h

        response = corr[peak_y, peak_x] / (w * h)

        return float(dx), float(dy), float(response)

    def _sobel_x_centroid(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float]:
        """Compute horizontal shift via Sobel X edge centroid tracking.

        Returns:
            dx: estimated horizontal shift
            response: edge intensity ratio (confidence proxy)
        """
        def edge_centroid(gray: np.ndarray) -> Tuple[float, float]:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_x = np.abs(sobel_x)
            if sobel_x.max() > 0:
                sobel_x = sobel_x / sobel_x.max()
            h, w = sobel_x.shape
            total = sobel_x.sum()
            if total > 0:
                _, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                cx = (x_coords * sobel_x).sum() / total
            else:
                cx = w / 2
            return cx, total

        cx1, intensity1 = edge_centroid(img1)
        cx2, intensity2 = edge_centroid(img2)

        dx = cx2 - cx1
        response = min(intensity1, intensity2) / max(intensity1, intensity2) if max(intensity1, intensity2) > 0 else 0

        return float(dx), float(response)

    def detect_direction(self, frame1: np.ndarray, frame2: np.ndarray) -> str:
        """Detect camera direction between two frames.

        Returns:
            'left', 'right', or 'stationary'
        """
        img1 = self._preprocess(frame1)
        img2 = self._preprocess(frame2)

        if self.method == "sobel_x":
            dx, response = self._sobel_x_centroid(img1, img2)
            if self.debug:
                print(f"dx={dx:.1f}, response={response:.3f}")
        else:
            dx, dy, response = self._phase_correlation(img1, img2)
            if self.debug:
                print(f"dx={dx:.1f}, dy={dy:.1f}, response={response:.3f}")

        if abs(dx) < self.shift_threshold:
            return 'stationary'
        elif dx > 0:
            return 'left'
        else:
            return 'right'

    def detect_from_video(self, frames: list, skip: int = 0) -> list:
        """Detect direction across multiple frames."""
        directions = []
        for i in range(0, len(frames) - 1 - skip, 1 + skip):
            direction = self.detect_direction(frames[i], frames[i + 1 + skip])
            directions.append(direction)
        return directions
