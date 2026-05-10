# pyrefly: ignore [missing-import]
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
            
        self.session = None
        if os.path.exists(model_path):
            try:
# pyrefly: ignore [missing-import]
                import onnxruntime as ort
                
                # Configure Providers
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kSameAsRequested',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'DEFAULT',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',
                ]
                
                self.session = ort.InferenceSession(model_path, providers=providers)
                current_providers = self.session.get_providers()
                
                if 'CUDAExecutionProvider' in current_providers:
                    print(f"[NITRO-GPU] ONNX CUDA Acceleration Enabled for Emotions")
                else:
                    print(f"[NITRO-CPU] Running Emotions on CPU (ONNX)")
                
                print(f"[OK] Emotion model loaded via ONNX from {model_path}")
            except Exception as e:
                print(f"[ERR] Error loading emotion model via ONNX: {e}")
                self.session = None
        else:
            print(f"[ERR] Emotion model not found at {model_path}")

    def detect_emotion(self, face_image):
        if self.session is None or face_image is None or face_image.size == 0:
            return "Unknown", 0.0

        try:
            # Preprocessing for FerPlus
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64)).astype(np.float32)
            
            # Input needs to be 1x1x64x64
            input_tensor = resized.reshape(1, 1, 64, 64)
            
            # Run Inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            scores = outputs[0][0]
            
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
