# Optical Flow Direction Detection

Detects camera ego-motion direction using Sobel X edge tracking. Works at low framerates with low-texture images where optical flow methods struggle.

## How It Works

1. Apply Sobel X filter to enhance vertical edges
2. Compute weighted centroid of edge intensity per frame
3. Track centroid shift between frames
4. Edge shift direction is opposite to camera movement

## Installation

```bash
pip install numpy opencv-python
```

## Usage

```python
from direction_detector import DirectionDetector

detector = DirectionDetector()
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
