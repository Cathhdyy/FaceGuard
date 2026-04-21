import cv2
import time
import numpy as np
import os

def benchmark():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(base_dir, "..", "src2", "Face_cascade.xml")
    
    if not os.path.exists(cascade_path):
        print(f"Error: Cascade not found at {cascade_path}")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: Failed to load cascade")
        return

    print(f"Benchmarking with {cascade_path}")
    
    # Create a dummy frame (640x480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw some white circles (fake faces)
    cv2.circle(frame, (320, 240), 100, (255, 255, 255), -1)
    
    # Test 1: Full resolution Haar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start = time.time()
    for _ in range(50):
        face_cascade.detectMultiScale(gray, 1.1, 5)
    end = time.time()
    print(f"Full Res (640x480) Haar FPS: {50 / (end - start):.2f}")
    
    # Test 2: Downsampled (0.25x) Haar
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
    start = time.time()
    for _ in range(50):
        face_cascade.detectMultiScale(small, 1.1, 5, minSize=(30, 30))
    end = time.time()
    print(f"Downsampled (160x120) Haar FPS: {50 / (end - start):.2f}")

    # Test 3: HOG (if available via dlib, but let's just test Haar for now)
    
if __name__ == "__main__":
    benchmark()
