"""
Simple demo to validate the DirectionDetector with synthetic images.
"""

import numpy as np
import cv2
from direction_detector import DirectionDetector


def create_test_image(width=640, height=480, offset=0):
    """Create a grayscale image with vertical edges at a given offset."""
    img = np.zeros((height, width), dtype=np.uint8)
    for x in range(20 + offset, width, 60):
        cv2.rectangle(img, (x, 20), (x + 40, height - 20), 255, -1)
    noise = np.random.randint(0, 30, (height, width), dtype=np.uint8)
    return cv2.add(img, noise)


def main():
    detector = DirectionDetector(debug=True)

    # Test right shift (camera moving left)
    img1 = create_test_image(offset=0)
    img2 = create_test_image(offset=10)
    result = detector.detect_direction(img1, img2)
    print(f"Shift right: {result}")

    # Test left shift (camera moving right)
    img1 = create_test_image(offset=10)
    img2 = create_test_image(offset=0)
    result = detector.detect_direction(img1, img2)
    print(f"Shift left: {result}")

    # Test no shift
    img1 = create_test_image(offset=0)
    img2 = create_test_image(offset=0)
    result = detector.detect_direction(img1, img2)
    print(f"No shift: {result}")


if __name__ == "__main__":
    main()
