# SPDX-License-Identifier: MIT
"""
Demo and CLI for DirectionDetector.

Usage:
    python demo.py                          - Run synthetic test
    python demo.py <image1> <image2>        - Compare two images
    python demo.py <video.mp4>              - Process video
"""

import sys
import numpy as np
import cv2
from collections import Counter
from direction_detector import DirectionDetector


def create_test_image(width=640, height=480, offset=0):
    """Create a grayscale image with vertical edges at a given offset."""
    img = np.zeros((height, width), dtype=np.uint8)
    for x in range(20 + offset, width, 60):
        cv2.rectangle(img, (x, 20), (x + 40, height - 20), 255, -1)
    noise = np.random.randint(0, 30, (height, width), dtype=np.uint8)
    return cv2.add(img, noise)


def run_synthetic_test():
    """Validate detector with synthetic images."""
    detector = DirectionDetector(debug=True)

    img1 = create_test_image(offset=0)
    img2 = create_test_image(offset=10)
    print(f"Shift right: {detector.detect_direction(img1, img2)}")

    img1 = create_test_image(offset=10)
    img2 = create_test_image(offset=0)
    print(f"Shift left: {detector.detect_direction(img1, img2)}")

    img1 = create_test_image(offset=0)
    img2 = create_test_image(offset=0)
    print(f"No shift: {detector.detect_direction(img1, img2)}")


def compare_images(img1_path: str, img2_path: str):
    detector = DirectionDetector(debug=True)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        return
    direction = detector.detect_direction(img1, img2)
    print(f"Camera direction: {direction}")


def process_video(video_path: str, sample_every_n: int = 5):
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

    counts = Counter(directions)
    print(f"\nSummary:")
    for direction, count in counts.most_common():
        print(f"  {direction}: {count} ({100*count/len(directions):.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        compare_images(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        process_video(sys.argv[1])
    else:
        run_synthetic_test()
