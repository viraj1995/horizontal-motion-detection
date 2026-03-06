# Optical Flow Direction Detection

## The Problem

When using optical flow for camera ego-motion direction detection at **low framerates (2-3 FPS)** with **low-texture images**, traditional optical flow methods fail because:

1. **Low temporal resolution**: Between frames, objects may move too little for reliable flow vectors
2. **Insufficient texture**: Smooth surfaces (walls, ceilings) lack features for keypoint detection
3. **Aperture problem**: Small image patches don't have enough variation to determine motion direction

This repository demonstrates these failures and provides a robust solution using **1D edge detection**.

## The Solution: Sobel X Edge Tracking

Instead of full optical flow, we use horizontal edge detection with **Sobel X filter**:

### Why It Works

1. **Edges are everywhere**: Most scenes have vertical edges (doors, windows, people, furniture)
2. **1D is simpler**: Only track horizontal displacement, not 2D motion
3. **Robust to texture**: Even smooth walls have some vertical structure
4. **Fast**: Single convolution operation vs. iterative optical flow

### Algorithm

```
1. Apply Sobel X filter → horizontal edges enhanced
2. For each row, find the "centroid" of edge intensity
3. Track how the centroid shifts between frames
4. Left shift = camera moving right, Right shift = camera moving left
```

## Installation

```bash
pip install numpy opencv-python
```

## Usage

```python
from direction_detector import DirectionDetector

detector = DirectionDetector()
direction = detector.detect_direction(frame1, frame2)
print(direction)  # 'left', 'right', or 'unknown'
```

## Methods Comparison

| Method | FPS Required | Texture Needed | Accuracy | Speed |
|--------|--------------|----------------|----------|-------|
| Lucas-Kanade | 15+ | High | Medium | Slow |
| Dense Optical Flow | 10+ | Medium | High | Very Slow |
| RAFT/DeepFlow | 5+ | Medium | Very High | Very Slow |
| **Sobel X (Ours)** | **1+** | **Low** | **High** | **Fast** |

## Use Cases

- Door/wall following robots
- Indoor navigation at low framerate
- Resource-constrained devices
- High-speed motion where blur defeats optical flow
