import cv2
import time
import os

def test_camera_settings():
    print("Testing Camera Settings for 30 FPS...")
    
    # Try CAP_DSHOW first
    backeneds = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    
    for backend in backeneds:
        backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else "MSMF"
        print(f"\n--- Testing Backend: {backend_name} ---")
        
        cap = cv2.VideoCapture(0, backend)
        
        # Test 1: Default
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Requested 30 FPS. Hardware says current FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        # Benchmarking actual capture speed
        start = time.time()
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at {i}")
                break
        end = time.time()
        
        actual_fps = 30 / (end - start)
        print(f"Actual Captured FPS (Default): {actual_fps:.2f}")
        
        # Test 2: Force MJPG (often required for 30fps on DirectShow)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        start = time.time()
        for i in range(30):
            ret, frame = cap.read()
            if not ret: break
        end = time.time()
        actual_fps_mjpg = 30 / (end - start)
        print(f"Actual Captured FPS (MJPG): {actual_fps_mjpg:.2f}")
        
        # Test 3: Disable Auto Exposure (often doubles FPS in low light)
        # 0.25 is manual mode for some DSHOW drivers, -4 or -5 for others
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        # Set a fixed exposure value that might be faster
        cap.set(cv2.CAP_PROP_EXPOSURE, -5) 
        
        start = time.time()
        for i in range(30):
            ret, frame = cap.read()
            if not ret: break
        end = time.time()
        actual_fps_manual = 30 / (end - start)
        print(f"Actual Captured FPS (Manual Exp): {actual_fps_manual:.2f}")
        
        cap.release()

if __name__ == "__main__":
    test_camera_settings()
