#!/usr/bin/env python3
"""
Face Recognition for Images and Videos
Uses Hybrid Approach (OpenCV + Direct Dlib) for stability.
Supports multiple detection models: Haar Cascade, HOG, and CNN.
"""

import cv2
import dlib
import pickle
import sys
import os
import numpy as np
import argparse
from emotion_detector import EmotionDetector

# Path to dlib models
import face_recognition_models

# Path to dlib models - dynamically loaded
PREDICTOR_PATH = face_recognition_models.pose_predictor_model_location()
FACE_RECOG_MODEL_PATH = face_recognition_models.face_recognition_model_location()
CNN_FACE_DETECTOR_PATH = face_recognition_models.cnn_face_detector_model_location()
CASCADE_PATH = "Face_cascade.xml"
KNN_MODEL_PATH = "trained_knn_model.clf"

def load_models(detection_method='haar'):
    """Load all necessary models"""
    models = {}
    
    # Check files
    if not os.path.exists(PREDICTOR_PATH) or not os.path.exists(FACE_RECOG_MODEL_PATH):
        print("❌ Error: Dlib model files not found!")
        return None
        
    if not os.path.exists(KNN_MODEL_PATH):
        print(f"❌ Error: {KNN_MODEL_PATH} not found! Please train model first.")
        return None

    try:
        print("Loading models...")
        models['sp'] = dlib.shape_predictor(PREDICTOR_PATH)
        models['facerec'] = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)
        
        with open(KNN_MODEL_PATH, 'rb') as f:
            models['knn'] = pickle.load(f)
            
        # Load detection model based on method
        if detection_method == 'haar':
            if not os.path.exists(CASCADE_PATH):
                print(f"❌ Error: {CASCADE_PATH} not found!")
                return None
            models['detector'] = cv2.CascadeClassifier(CASCADE_PATH)
            print("✓ Loaded Haar Cascade detector")
            
        elif detection_method == 'hog':
            models['detector'] = dlib.get_frontal_face_detector()
            print("✓ Loaded HOG detector")
            
        elif detection_method == 'cnn':
            if not os.path.exists(CNN_FACE_DETECTOR_PATH):
                print(f"❌ Error: {CNN_FACE_DETECTOR_PATH} not found!")
                return None
            models['detector'] = dlib.cnn_face_detection_model_v1(CNN_FACE_DETECTOR_PATH)
            print("✓ Loaded CNN detector (Warning: Slow on CPU)")
            
        models['emotion'] = EmotionDetector()
        print("✓ All models loaded successfully")
        return models
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

def detect_faces(frame, models, method='haar'):
    """Detect faces using specified method"""
    faces_rect = []
    h, w = frame.shape[:2]
    
    if method == 'haar':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = models['detector'].detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Convert to (x, y, w, h) format
        faces_rect = rects
        
    elif method == 'hog':
        rgb_frame = frame[:, :, ::-1]
        rgb_frame = np.ascontiguousarray(rgb_frame)
        # Upsample 1 time for better detection of small faces
        dets = models['detector'](rgb_frame, 1)
        for det in dets:
            faces_rect.append((det.left(), det.top(), det.width(), det.height()))
            
    elif method == 'cnn':
        rgb_frame = frame[:, :, ::-1]
        rgb_frame = np.ascontiguousarray(rgb_frame)
        # Auto-adjust upsampling based on image size
        # If image is very large, don't upsample (0). If small, upsample (1 or 2).
        upsample = 1
        if h * w > 1920 * 1080:
            upsample = 0
            
        print(f"Debug: Image size {w}x{h}, using CNN upsample={upsample}")
        
        dets = models['detector'](rgb_frame, upsample)
        for det in dets:
            rect = det.rect
            faces_rect.append((rect.left(), rect.top(), rect.width(), rect.height()))
            
    return faces_rect

def get_face_predictions(frame, models, method='haar'):
    """Detect faces and get predictions"""
    faces_rect = detect_faces(frame, models, method)
    predictions = []
    
    if len(faces_rect) > 0:
        # Convert to RGB and ensure contiguous array
        rgb_frame = frame[:, :, ::-1]
        rgb_frame = np.ascontiguousarray(rgb_frame)
        
        faces_encodings = []
        known_locations = []
        
        for (x, y, w, h) in faces_rect:
            try:
                # Ensure coordinates are within frame bounds
                h_frame, w_frame = frame.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, w_frame - x)
                h = min(h, h_frame - y)
                
                rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                shape = models['sp'](rgb_frame, rect)
                face_descriptor = models['facerec'].compute_face_descriptor(rgb_frame, shape, 1)
                
                faces_encodings.append(np.array(face_descriptor))
                known_locations.append((y, x+w, y+h, x)) # top, right, bottom, left
            except Exception as e:
                print(f"Debug: Error processing face at {x},{y}: {e}")
                continue
        
        if len(faces_encodings) > 0:
            try:
                closest_distances = models['knn'].kneighbors(faces_encodings, n_neighbors=1)
                are_matches = [closest_distances[0][i][0] <= 0.45 for i in range(len(faces_encodings))]
                preds = models['knn'].predict(faces_encodings)
                
                for i, (pred, loc, rec) in enumerate(zip(preds, known_locations, are_matches)):
                    name = pred if rec else "Unknown"
                    distance = closest_distances[0][i][0]
                    # Simple confidence mapping: 0.0 -> 100%, 0.45 -> 55%
                    # Let's use (1 - distance) for simplicity, maybe scaled
                    face_conf = max(0, (1.0 - distance)) if rec else 0.0
                    
                    # Extract face ROI for emotion
                    top, right, bottom, left = loc
                    face_roi = frame[top:bottom, left:right]
                    emo_label, emo_conf = models['emotion'].detect_emotion(face_roi)
                    
                    predictions.append((name, loc, face_conf, emo_label, emo_conf))
            except Exception as e:
                print(f"Debug: Prediction error: {e}")
                pass
                
    return predictions

def draw_predictions(frame, predictions):
    """Draw bounding boxes and labels on frame"""
    for name, (top, right, bottom, left), face_conf, emo_label, emo_conf in predictions:
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        
        # Prepare label text
        if name == "Unknown":
            label = "Unknown"
        else:
            label = f"{name} ({face_conf*100:.0f}%)"
            
        emotion_text = f"{emo_label} ({emo_conf*100:.0f}%)"
        
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        
        # Draw Name
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(frame, (left, bottom - text_height - 10), (left + text_width + 10, bottom), color, -1)
        cv2.putText(frame, label, (left + 5, bottom - 5), font, font_scale, (0, 0, 0), font_thickness)
        
        # Draw Emotion (above the box or below?)
        # Let's draw it above
        (emo_w, emo_h), _ = cv2.getTextSize(emotion_text, font, font_scale, font_thickness)
        cv2.rectangle(frame, (left, top - emo_h - 10), (left + emo_w + 10, top), (255, 0, 0), -1)
        cv2.putText(frame, emotion_text, (left + 5, top - 5), font, font_scale, (255, 255, 255), font_thickness)
    return frame

def process_image(image_path, models, method='haar'):
    """Process a single image file"""
    if not os.path.exists(image_path):
        print(f"❌ Error: File {image_path} not found")
        return

    print(f"Processing image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Error: Could not read image")
        return

    predictions = get_face_predictions(frame, models, method)
    processed_frame = draw_predictions(frame, predictions)
    
    names = [p[0] for p in predictions]
    print(f"Found {len(predictions)} faces: {', '.join(names)}")
    
    # Resize if too big
    height, width = processed_frame.shape[:2]
    max_height = 800
    if height > max_height:
        scale = max_height / height
        processed_frame = cv2.resize(processed_frame, (int(width*scale), int(height*scale)))
    
    cv2.imshow(f"Result - {os.path.basename(image_path)}", processed_frame)
    print("Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, models, method='haar'):
    """Process a video file"""
    if not os.path.exists(video_path):
        print(f"❌ Error: File {video_path} not found")
        return

    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video")
        return

    print("Controls: Press 'q' to quit")
    
    frame_count = 0
    last_predictions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Determine if we should run detection
        should_detect = True
        
        # For CNN, skip frames as it is slow
        if method == 'cnn' and frame_count % 3 != 0:
            should_detect = False
        
        if should_detect:
            last_predictions = get_face_predictions(frame, models, method)
        
        # Always draw the last known predictions (persistence)
        processed_frame = draw_predictions(frame, last_predictions)
        
        # Add info overlay
        cv2.putText(processed_frame, f"Frame: {frame_count} | Model: {method}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Video Recognition', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete")

def main():
    parser = argparse.ArgumentParser(description='Face Recognition for Images and Videos')
    parser.add_argument('path', help='Path to image or video file')
    parser.add_argument('--model', choices=['haar', 'hog', 'cnn'], default='haar',
                        help='Detection model to use: haar (fast), hog (balanced), cnn (accurate but slow)')
    args = parser.parse_args()
    
    models = load_models(args.model)
    if models is None:
        return
        
    # Check extension
    ext = os.path.splitext(args.path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    
    if ext in image_exts:
        process_image(args.path, models, args.model)
    elif ext in video_exts:
        process_video(args.path, models, args.model)
    else:
        print(f"❌ Unknown file type: {ext}")
        print(f"Supported images: {', '.join(image_exts)}")
        print(f"Supported videos: {', '.join(video_exts)}")

if __name__ == "__main__":
    main()
