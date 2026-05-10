# Horizontal Motion Detection

Detects camera ego-motion direction using two methods: phase correlation (FFT-based) or Sobel X edge centroid tracking. Works at low framerates with low-texture images where optical flow methods struggle.

## Methods

**Phase correlation** (default) — Computes the dominant pixel translation between frames via FFT-based cross-correlation. More accurate for general scenes.

**Sobel X edge centroid** — Tracks the center of mass of vertical edges between frames. Simpler, faster, useful for corridor/doorway following.

## Installation

```bash
pip install numpy opencv-python
```

## Usage

```python
from direction_detector import MotionDetector

# Phase correlation (default)
detector = MotionDetector()

# Or use Sobel X edge centroid
detector = MotionDetector(method="sobel_x")

direction = detector.detect_direction(frame1, frame2)
print(direction)  # 'left', 'right', or 'stationary'
```

Run the demo with synthetic images:

```bash
python demo.py
```

Or compare two images:

```bash
python demo.py image1.jpg image2.jpg
```

Or process a video:

```bash
python demo.py video.mp4
```
