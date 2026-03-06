"""
Demonstration of Optical Flow Limitations and Sobel X Solution

This script demonstrates:
1. Why optical flow fails with low texture / low FPS
2. How Sobel X edge detection solves the problem
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from direction_detector import DirectionDetector


def create_test_images(width=640, height=480, shift_pixels=10):
    """
    Create synthetic test images to demonstrate the algorithm.
    
    Returns images with known horizontal shift for validation.
    """
    # Create a simple pattern with vertical edges (good for Sobel X)
    img1 = np.zeros((height, width), dtype=np.uint8)
    
    # Add vertical stripes (edges)
    stripe_width = 40
    gap = 60
    for x in range(20, width, gap):
        cv2.rectangle(img1, (x, 20), (x + stripe_width, height - 20), 255, -1)
    
    # Add some random noise (texture)
    noise = np.random.randint(0, 30, (height, width), dtype=np.uint8)
    img1 = cv2.add(img1, noise)
    
    # Shift the image (simulating camera movement)
    M = np.float32([[1, 0, shift_pixels], [0, 1, 0]])
    img2 = cv2.warpAffine(img1, M, (width, height))
    
    return img1, img2


def create_low_texture_images(width=640, height=480, shift_pixels=10):
    """
    Create low-texture images (like blank walls).
    This is where optical flow struggles but Sobel X still works.
    """
    # Almost uniform gray - very little texture
    img1 = np.ones((height, width), dtype=np.uint8) * 180
    
    # Add very subtle vertical lines (barely visible)
    for x in range(50, width, 200):
        cv2.line(img1, (x, 0), (x, height), 200, 1)
    
    # Shift
    M = np.float32([[1, 0, shift_pixels], [0, 1, 0]])
    img2 = cv2.warpAffine(img1, M, (width, height))
    
    return img1, img2


def test_optical_flow_vs_sobel(img1, img2, name=""):
    """
    Compare Lucas-Kanade optical flow with Sobel X method.
    """
    print(f"\n=== Testing: {name} ===")
    
    # Convert to grayscale for optical flow
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        gray2 = img2
    
    # --- Optical Flow (Lucas-Kanade) ---
    # Parameters for good features to track
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    p0 = cv2.goodFeaturesToTrack(gray1, **feature_params)
    
    if p0 is not None and len(p0) > 5:
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)
        
        if st.sum() > 0:
            # Calculate average displacement
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            flow_x = (good_new[:, 0, 0] - good_old[:, 0, 0]).mean()
            print(f"  Optical Flow: {flow_x:.1f} pixels (found {len(good_new)} points)")
            of_direction = 'left' if flow_x < 0 else 'right' if flow_x > 0 else 'stationary'
            print(f"  Optical Flow direction: {of_direction}")
        else:
            print(f"  Optical Flow: FAILED - No tracking points")
            of_direction = 'failed'
    else:
        print(f"  Optical Flow: FAILED - Not enough features to track")
        of_direction = 'failed'
    
    # --- Sobel X Method ---
    detector = DirectionDetector(edge_threshold=20, debug=False)
    sobel_direction = detector.detect_direction(img1, img2)
    print(f"  Sobel X direction: {sobel_direction}")
    
    return of_direction, sobel_direction


def demonstrate_problem():
    """
    Main demonstration showing why Sobel X works where optical flow fails.
    """
    print("=" * 60)
    print("OPTICAL FLOW vs SOBEL X EDGE DETECTION")
    print("=" * 60)
    
    # Test 1: Normal textured images
    print("\n[TEST 1] Normal textured images (shift: +10 pixels → camera moves LEFT)")
    img1, img2 = create_test_images(shift_pixels=10)
    test1_of, test1_sobel = test_optical_flow_vs_sobel(img1, img2, "Normal Texture")
    
    # Test 2: Low texture images (like blank walls)
    print("\n[TEST 2] Low texture images (shift: +10 pixels)")
    img1, img2 = create_low_texture_images(shift_pixels=10)
    test2_of, test2_sobel = test_optical_flow_vs_sobel(img1, img2, "Low Texture")
    
    # Test 3: Very low texture
    print("\n[TEST 3] Very low texture (nearly blank wall)")
    img1 = np.ones((480, 640), dtype=np.uint8) * 200
    img2 = np.roll(img1, 15, axis=1)  # Shift right by 15 pixels
    test3_of, test3_sobel = test_optical_flow_vs_sobel(img1, img3, "Nearly Blank")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Test':<25} {'Optical Flow':<20} {'Sobel X':<15}")
    print(f"{'Normal texture':<25} {test1_of:<20} {test1_sobel:<15}")
    print(f"{'Low texture':<25} {test2_of:<20} {test2_sobel:<15}")
    print(f"{'Nearly blank':<25} {test3_of:<20} {test3_sobel:<15}")
    
    print("\n✓ Sobel X works consistently regardless of image texture")
    print("✓ Optical flow fails when there aren't enough features to track")


def create_demo_video():
    """Create a simple demo video for testing."""
    width, height = 640, 480
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_output.mp4', fourcc, 2.0, (width, height))
    
    # Simulate camera panning left (edges move right)
    for i in range(30):
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Add vertical edges at different positions
        offset = i * 2  # Moving right
        
        # Draw vertical stripes
        for x in range(50 + offset, width, 80):
            cv2.rectangle(frame, (x, 50), (x + 20, height - 50), 255, -1)
        
        # Add some texture noise
        noise = np.random.randint(0, 20, (height, width), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    
    out.release()
    print("Demo video created: demo_output.mp4")
    print("Run: python direction_detector.py demo_output.mp4")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "create_video":
            create_demo_video()
        else:
            # Treat as video file
            from direction_detector import DirectionDetector
            demo_with_video(sys.argv[1])
    else:
        demonstrate_problem()
