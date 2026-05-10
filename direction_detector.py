# SPDX-License-Identifier: MIT
"""
Horizontal motion detector using phase correlation.

Detects camera ego-motion direction by computing the dominant pixel
translation between frames via FFT-based phase correlation.
"""

import cv2
import numpy as np
from typing import Tuple


class MotionDetector:
    """Detects horizontal camera motion direction using phase correlation."""

    def __init__(self, shift_threshold: float = 1.0, debug: bool = False):
        self.shift_threshold = shift_threshold
        self.debug = debug

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale, downscale, and apply mild blur."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape
        scale = min(320 / w, 240 / h)
        if scale < 1:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def _phase_correlation(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float, float]:
        """Compute translation between two images via phase correlation.

        Returns:
            dx: horizontal shift (positive = img2 shifted right relative to img1)
            dy: vertical shift
            response: peak correlation strength (0-1)
        """
        f1 = np.fft.fft2(img1)
        f2 = np.fft.fft2(img2)

        cross_power = f1 * np.conj(f2)
        cross_power = cross_power / (np.abs(cross_power) + 1e-10)

        corr = np.fft.ifft2(cross_power)
        corr = np.abs(corr)

        peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
        h, w = corr.shape

        # Wrap peak to signed offset
        dx = peak_x if peak_x < w / 2 else peak_x - w
        dy = peak_y if peak_y < h / 2 else peak_y - h

        response = corr[peak_y, peak_x] / (w * h)

        return float(dx), float(dy), float(response)

    def detect_direction(self, frame1: np.ndarray, frame2: np.ndarray) -> str:
        """Detect camera direction between two frames.

        Returns:
            'left', 'right', or 'stationary'
        """
        img1 = self._preprocess(frame1)
        img2 = self._preprocess(frame2)

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
