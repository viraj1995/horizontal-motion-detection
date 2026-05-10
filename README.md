# Horizontal Motion Detection

Detects camera ego-motion direction using phase correlation (FFT-based cross-correlation). Works at low framerates with low-texture images where optical flow methods struggle.

## How It Works

1. Preprocess frames: grayscale, downscale, mild blur
2. Compute FFT of both frames
3. Cross-power spectrum → inverse FFT → correlation surface
4. Peak location gives the dominant pixel translation
5. Horizontal shift direction is opposite to camera movement

## Installation

```bash
pip install numpy opencv-python
```

## Usage

```python
from direction_detector import MotionDetector

detector = MotionDetector()
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
