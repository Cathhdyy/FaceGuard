import cv2
import numpy as np
import os

class EmotionDetector:
    def __init__(self, model_path=None):
        self.classes = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Stressed', 'Disgust', 'Stressed', 'Contempt']
        
        if model_path is None:
            # Resolve path relative to THIS file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "emotion-ferplus-8.onnx")
            
        self.net = None
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                print(f"[OK] Emotion model loaded from {model_path}")
            except Exception as e:
                print(f"[ERR] Error loading emotion model: {e}")
        else:
            print(f"[ERR] Emotion model not found at {model_path}")

    def detect_emotion(self, face_image):
        if self.net is None or face_image is None or face_image.size == 0:
            return "Unknown", 0.0

        try:
            # Preprocessing for FerPlus
            # 1. Grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # 2. Resize to 64x64
            resized = cv2.resize(gray, (64, 64))
            
            # 3. Preprocess for DNN
            # FerPlus expects 1x1x64x64 input
            blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
            
            self.net.setInput(blob)
            scores = self.net.forward()[0]
            
            # Softmax
            scores = np.exp(scores - np.max(scores))
            probs = scores / scores.sum()
            
            # Get max
            idx = np.argmax(probs)
            label = self.classes[idx]
            confidence = probs[idx]
            
            return label, confidence
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Unknown", 0.0

if __name__ == "__main__":
    # Test
    ed = EmotionDetector()
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    label, conf = ed.detect_emotion(dummy)
    print(f"Test prediction: {label} ({conf*100:.1f}%)")
