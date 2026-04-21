import requests
import cv2
import numpy as np

def save_current_frame():
    try:
        # Get one frame from the MJPEG stream
        r = requests.get('http://127.0.0.1:5001/video_feed', stream=True, timeout=5)
        if r.status_code != 200:
            print(f"Failed to connect: {r.status_code}")
            return

        bytes_data = b''
        for chunk in r.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite('debug_frame.jpg', frame)
                print("Frame saved to debug_frame.jpg")
                break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    save_current_frame()
