import cv2
import time

def test_camera(index, backend=None):
    if backend:
        print(f"Testing camera {index} with backend {backend}...")
        cap = cv2.VideoCapture(index, backend)
    else:
        print(f"Testing camera {index} with default backend...")
        cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"FAILED: Camera {index} could not be opened.")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    t_start = time.time()
    frames = 0
    while time.time() - t_start < 2:
        ok, frame = cap.read()
        if ok:
            frames += 1
        else:
            print("Read failed.")
            break
    
    print(f"SUCCESS: Read {frames} frames in 2 seconds.")
    cap.release()
    return frames > 0

if __name__ == "__main__":
    backends = [
        ("Default", None),
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF)
    ]
    
    for name, b in backends:
        for i in range(2):
            try:
                test_camera(i, b)
            except Exception as e:
                print(f"Error testing {name} on index {i}: {e}")
