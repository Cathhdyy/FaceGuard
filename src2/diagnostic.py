import cv2
import dlib
import numpy as np
import face_recognition
import os
import sys

def run_diagnostics():
    print("--- System Info ---")
    print(f"Python: {sys.version}")
    print(f"Numpy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Dlib: {dlib.__version__}")
    
    print("\n--- Camera Check ---")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return
    
    print(f"Frame captured. Shape: {frame.shape}, Dtype: {frame.dtype}")
    print(f"Min: {np.min(frame)}, Max: {np.max(frame)}, Mean: {np.mean(frame)}")
    
    # Save frame for visual inspection
    cv2.imwrite("diagnostic_frame.jpg", frame)
    print("Frame saved to diagnostic_frame.jpg")
    
    # Test RGB normalization
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"RGB Shape: {rgb.shape}, Dtype: {rgb.dtype}")
    
    print("\n--- Detection Proofs ---")
    
    # 1. HOG Detection
    try:
        print("Testing HOG detection...")
        locs = face_recognition.face_locations(rgb, model="hog")
        print(f"HOG found {len(locs)} faces: {locs}")
    except Exception as e:
        print(f"HOG Failure: {e}")
        
    # 2. Haar Detection
    try:
        cascade_path = "Face_cascade.xml"
        if os.path.exists(cascade_path):
            print("Testing Haar detection...")
            cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5)
            print(f"Haar found {len(faces)} faces.")
        else:
            print(f"Haar cascade file NOT FOUND at {cascade_path}")
    except Exception as e:
        print(f"Haar Failure: {e}")

    cap.release()

if __name__ == "__main__":
    run_diagnostics()
