from flask import Flask, render_template, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename
from camera import VideoCamera
import os
import subprocess
import cv2
import dlib
import pickle
import numpy as np
import threading
import faiss
from train_model import train_face_recognition_model
import datetime
import time
from emotion_detector import EmotionDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi', 'mov', 'mkv'}

camera = None

# Paths for recognition
import face_recognition_models

# Paths for recognition - dynamically loaded
PREDICTOR_PATH = face_recognition_models.pose_predictor_model_location()
FACE_RECOG_MODEL_PATH = face_recognition_models.face_recognition_model_location()
CNN_FACE_DETECTOR_PATH = face_recognition_models.cnn_face_detector_model_location()
CASCADE_PATH = "Face_cascade.xml"
KNN_MODEL_PATH = "trained_knn_model.clf"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

camera_lock = threading.Lock()

def get_camera():
    global camera
    if camera is None:
        with camera_lock:
            if camera is None:
                camera = VideoCamera()
    return camera

def load_recognition_models():
    """Load models for file processing"""
    models = {}
    models['sp'] = dlib.shape_predictor(PREDICTOR_PATH)
    models['facerec'] = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)
    models['haar'] = cv2.CascadeClassifier(CASCADE_PATH)
    models['hog'] = dlib.get_frontal_face_detector()
    models['cnn'] = dlib.cnn_face_detection_model_v1(CNN_FACE_DETECTOR_PATH)
    models['emotion'] = EmotionDetector()
    
    # Load FAISS
    p_index = KNN_MODEL_PATH.replace(".clf", ".index")
    p_labels = KNN_MODEL_PATH.replace(".clf", "_labels.pkl")
    
    if os.path.exists(p_index) and os.path.exists(p_labels):
        models['faiss_index'] = faiss.read_index(p_index)
        with open(p_labels, 'rb') as f:
            models['faiss_labels'] = pickle.load(f)
            
    return models

def process_image_file(filepath, model_type='haar'):
    """Process uploaded image"""
    models = load_recognition_models()
    frame = cv2.imread(filepath)
    if frame is None:
        return None
    
    predictions = detect_and_recognize_file(frame, models, model_type)
    frame = draw_predictions_file(frame, predictions)
    
    # Save processed image
    output_path = filepath.replace('.', '_processed.')
    cv2.imwrite(output_path, frame)
    return output_path, len(predictions)

def detect_and_recognize_file(frame, models, method='haar'):
    """Detect and recognize faces in a frame"""
    faces_rect = []
    h_frame, w_frame = frame.shape[:2]
    
    # Detection
    if method == 'haar':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = models['haar'].detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        faces_rect = rects
    elif method == 'hog':
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        dets = models['hog'](rgb, 1)
        for det in dets:
            faces_rect.append((det.left(), det.top(), det.width(), det.height()))
    elif method == 'cnn':
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        upsample = 0 if h_frame*w_frame > 1920*1080 else 1
        dets = models['cnn'](rgb, upsample)
        for det in dets:
            rect = det.rect
            faces_rect.append((rect.left(), rect.top(), rect.width(), rect.height()))
    
    # Recognition
    predictions = []
    if len(faces_rect) > 0:
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        
        # Pre-process all face locations for dlib/face_recognition
        # faces_rect is (x, y, w, h)
        locations = []
        for (x, y, w, h) in faces_rect:
            locations.append((int(y), int(x+w), int(y+h), int(x)))
            
        # Try to get encodings for recognition
        encodings = []
        try:
            # Using dlib models manually as per existing code structure in app.py
            for (top, right, bottom, left) in locations:
                try:
                    rect = dlib.rectangle(left, top, right, bottom)
                    shape = models['sp'](rgb, rect)
                    desc = models['facerec'].compute_face_descriptor(rgb, shape, 1)
                    encodings.append(np.array(desc))
                except:
                    encodings.append(None)
        except Exception as e:
            print(f"File Encoding Error: {e}")
            encodings = [None] * len(locations)

        # Process each face
        for i, loc in enumerate(locations):
            # Recognition if FAISS index exists
            faiss_index = models.get('faiss_index')
            faiss_labels = models.get('faiss_labels')
            
            if i < len(encodings) and encodings[i] is not None and faiss_index:
                try:
                    encoding = encodings[i].reshape(1, -1).astype('float32')
                    distances, indices = faiss_index.search(encoding, 1)
                    
                    dist = distances[0][0]
                    idx = indices[0][0]
                    
                    is_match = dist <= 0.45
                    
                    if is_match and idx in faiss_labels:
                        name = faiss_labels[idx]
                        face_conf = max(0, (1.0 - dist))
                except:
                    pass
            
            # Emotion
            top, right, bottom, left = loc
            t, r, b, l = max(0, top), min(w_frame, right), min(h_frame, bottom), max(0, left)
            face_roi = frame[t:b, l:r]
            
            emo_label, emo_conf = "Unknown", 0.0
            if face_roi.size > 0:
                try:
                    emo_label, emo_conf = models['emotion'].detect_emotion(face_roi)
                except:
                    pass
                    
            predictions.append((name, loc, face_conf, emo_label, emo_conf))
    
    return predictions

def draw_predictions_file(frame, predictions):
    """Draw bounding boxes on frame"""
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
        
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(frame, (left, bottom - th - 10), (left + tw + 10, bottom), color, -1)
        cv2.putText(frame, label, (left + 5, bottom - 5), font, font_scale, (0, 0, 0), font_thickness)
        
        # Draw Emotion
        (emo_w, emo_h), _ = cv2.getTextSize(emotion_text, font, font_scale, font_thickness)
        cv2.rectangle(frame, (left, top - emo_h - 10), (left + emo_w + 10, top), (255, 0, 0), -1)
        cv2.putText(frame, emotion_text, (left + 5, top - 5), font, font_scale, (255, 255, 255), font_thickness)
    return frame

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    # Nitro Robust MJPEG Generator
    last_id = -9999
    while True:
        try:
            # The Nitro engine is non-blocking, so we poll at high frequency
            frame, current_id = camera.get_frame(last_id)
            
            # Yield if ID changed (either new frame or new status)
            if frame and current_id != last_id:
                last_id = current_id
                
                header = (b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n'
                         b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n')
                yield (header + frame + b'\r\n')
            
            # Adaptive sleep: fast during streaming, slow during loading
            time.sleep(0.01 if current_id >= 0 else 0.5)
                
        except GeneratorExit:
            break
        except Exception as e:
            print(f"[NITRO-STREAM] Error: {e}")
            time.sleep(1.0)

@app.route('/video_feed')
def video_feed():
    obj = Response(gen(get_camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # Add robust headers to prevent browser caching/buffering issues
    obj.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    obj.headers['Pragma'] = 'no-cache'
    obj.headers['Expires'] = '0'
    obj.headers['Connection'] = 'close'
    return obj

@app.route('/api/config', methods=['POST'])
def config():
    data = request.json
    if 'model' in data:
        get_camera().set_model(data['model'])
    return jsonify({'status': 'success', 'model': get_camera().current_model_type})

@app.route('/api/stats')
def stats():
    cam = get_camera()
    preds = cam.get_predictions()
    
    # In Nitro mode, these metrics are accessed safely through the engine
    stats_data = {
        'faces':            len(preds),
        'names':            [p[0] for p in preds],
        'emotions':         [p[3] for p in preds],
        'frame_count':      cam.frame_count,
        'processing_count': cam.last_predictions_count,
        'capture_fps':      cam.fps,
        'processing_fps':   cam.processing_fps
    }
    return jsonify(stats_data)

@app.route('/api/register', methods=['POST'])
def register_face():
    data = request.json
    name = data.get('name')
    idx = data.get('face_idx', 0)
    
    if not name:
        return jsonify({'status': 'error', 'message': 'Name is required'}), 400
    
    cam = get_camera()
    image_bytes = cam.capture_face_roi(idx)
    
    if image_bytes is None:
        return jsonify({'status': 'error', 'message': 'Could not capture face. Is a face detected?'}), 400
    
    # Create directory
    train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data", name)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    # Save file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(train_dir, filename)
    
    with open(filepath, 'wb') as f:
        f.write(image_bytes)
    
    return jsonify({'status': 'success', 'message': f'Face captured for {name}. Ready to retrain.'})

@app.route('/api/retrain', methods=['POST'])
def retrain():
    def do_retrain():
        print("[ADMIN] Starting background retraining...")
        try:
            # Fix: Use absolute paths so it works regardless of CWD
            base_path = os.path.dirname(os.path.abspath(__file__))
            train_dir = os.path.join(base_path, "training_data")
            model_path = os.path.join(base_path, "trained_knn_model.clf")
            
            success = train_face_recognition_model(
                training_dir=train_dir,
                model_save_path=model_path
            )
            
            if success:
                print(f"[ADMIN] Retraining successful. Model saved to {model_path}")
                get_camera().reload_models()
            else:
                print("[ADMIN] Retraining failed to find data or encode faces.")
        except Exception as e:
            print(f"[ADMIN] Retraining error: {e}")

    # Run in background to avoid blocking
    thread = threading.Thread(target=do_retrain)
    thread.start()
    
    return jsonify({'status': 'success', 'message': 'Retraining started in background.'})
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Get model from request
    model_type = request.form.get('model', 'haar')
    
    # Process based on file type
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ['png', 'jpg', 'jpeg', 'bmp']:
        output_path, face_count = process_image_file(filepath, model_type)
        return jsonify({
            'status': 'success',
            'type': 'image',
            'filename': os.path.basename(output_path),
            'faces': face_count
        })
    else:
        # For videos, just acknowledge upload
        return jsonify({
            'status': 'success',
            'type': 'video',
            'filename': filename,
            'message': 'Video uploaded. Processing not yet implemented.'
        })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def kill_port(port):
    import platform
    try:
        if platform.system() == "Windows":
            # Windows command to find and kill process on port
            # Using netstat and taskkill
            result = subprocess.run(f'netstat -ano | findstr :{port}', shell=True, capture_output=True, text=True)
            if result.stdout:
                # Extract PIDs (last column)
                pids = set()
                for line in result.stdout.strip().split('\n'):
                    parts = line.split()
                    if parts:
                        pids.add(parts[-1])
                for pid in pids:
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            subprocess.run(f"lsof -ti :{port} | xargs kill -9", shell=True)
    except Exception:
        pass

if __name__ == '__main__':
    kill_port(5001)
    app.run(host='0.0.0.0', port=5001, debug=False)
