import cv2
import pickle
import os
import numpy as np
import time
import face_recognition
import faiss
from emotion_detector import EmotionDetector
import threading

# ── Setup ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Limit OpenCV's own thread pool — we manage our own parallelism
cv2.setNumThreads(1)


def get_placeholder_frame(text="Camera Error"):
    # Create a slightly lighter background for better visibility
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (35, 38, 48) # Sleek Dark Blue-Gray
    
    # Draw centered text
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.8
    thickness = 1
    
    # Split text into lines if too long
    lines = text.split('\n')
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        tx = (640 - tw) // 2
        ty = (240 + th // 2) + (i * 35)
        cv2.putText(img, line, (tx, ty), font, scale, (220, 220, 230), thickness)
    
    # Add a border-like accent
    cv2.rectangle(img, (20, 20), (620, 460), (60, 65, 85), 1)
    
    _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()


class VideoCamera(object):
    """
    Nitro-Decoupled Pipeline
    ──────────────────────────
    T0  ModelLoadThread -> Background heavy model loading
    T1  CaptureThread   -> Hardware read, isolated & non-blocking
    T2  DetectThread    -> Face detection
    T3  RecogThread     -> AI Recognition
    T4  EncodeThread    -> UI Rendering + JPEG
    """

    _instance = None
    _class_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print(f"[NITRO] Initializing Decoupled Engine...")
        self.start_time = time.time()
        
        # ── Decoupled Locks ───────────────────────────────────────────
        self._lock_raw    = threading.Lock() # T1 -> T2, T4
        self._lock_locs   = threading.Lock() # T2 -> T3, T4
        self._lock_recog  = threading.Lock() # T3 -> T4
        self._lock_output = threading.Lock() # T4 -> Web

        # ── Thread Synchronization ────────────────────────────────────
        self._cond_raw    = threading.Condition(self._lock_raw)
        self._cond_output = threading.Condition(self._lock_output)

        # ── State Containers ──────────────────────────────────────────
        self.video = None
        self.models = {}
        self.current_model_type = 'haar'
        
        # Buffers
        self._raw_frame     = np.zeros((480, 640, 3), dtype=np.uint8)
        self._raw_frame_id  = 0
        self._face_locs     = []
        self._face_locs_id  = 0
        self._recog_results = []
        self._recog_locs    = []
        self._output_jpeg   = None
        self._output_id     = 0

        # Stats
        self.frame_count    = 0
        self.fps            = 0.0
        self.processing_fps = 0.0
        self.last_predictions_count = 0
        
        # Flags
        self.models_ready   = False
        self.camera_ready   = False
        self.error_msg      = None
        self.stopped        = False
        self.current_backend = "Checking..."
        self.warmup_progress = 0
        self.process_nth_frame = 2 # Process every 2nd frame for massive performance boost
        
        # Internal Metrics
        self._perf_detect_ms = 0
        self._perf_encode_ms = 0
        self._perf_recog_ms  = 0

        # ── Start Pipeline ────────────────────────────────────────────
        threading.Thread(target=self._model_load_thread, daemon=True, name="T0-Models").start()
        threading.Thread(target=self._capture_thread,    daemon=True, name="T1-Capture").start()
        threading.Thread(target=self._detect_thread,     daemon=True, name="T2-Detect").start()
        threading.Thread(target=self._recog_thread,      daemon=True, name="T3-Recog").start()
        threading.Thread(target=self._encode_thread,     daemon=True, name="T4-Encode").start()

        self._initialized = True

    # ── T0: Model Loading (Isolated) ───────────────────────────────
    def _model_load_thread(self):
        try:
            p_index = os.path.join(BASE_DIR, "trained_knn_model.index")
            p_labels = os.path.join(BASE_DIR, "trained_knn_model_labels.pkl")
            p_cascade = os.path.join(BASE_DIR, "Face_cascade.xml")
            
            models = {}
            if os.path.exists(p_index) and os.path.exists(p_labels):
                models['faiss_index'] = faiss.read_index(p_index)
                with open(p_labels, 'rb') as f:
                    models['faiss_labels'] = pickle.load(f)
                print("[NITRO] FAISS Backend Loaded")
            
            if os.path.exists(p_cascade):
                models['haar'] = cv2.CascadeClassifier(p_cascade)
            models['emotion'] = EmotionDetector()
            
            self.models = models
            self.models_ready = True
            print("[NITRO] Models Ready")
        except Exception as e:
            self.error_msg = f"Model Failure: {str(e)}"

    # ── T1: Capture (Hardware Isolated) ───────────────────────────
    def _capture_thread(self):
        # Preference: MSMF often performs 2x better on Windows than DSHOW
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
        found = False
        
        # Discovery Phase
        while not self.stopped and not found:
            for b_id in backends:
                self.current_backend = "DSHOW" if b_id == cv2.CAP_DSHOW else "MSMF"
                for idx in [0, 1, 2]:
                    if self.stopped: return
                    cap = cv2.VideoCapture(idx, b_id)
                    if cap.isOpened():
                        # Configure backend for speed
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # ZERO latency
                        
                        ok, _ = cap.read()
                        if ok:
                            self.video = cap
                            self.camera_ready = True
                            found = True
                            print(f"[NITRO] Camera Locked: {self.current_backend}")
                            break
                        cap.release()
                if found: break
            if not found:
                time.sleep(1.0)

        # High-Speed Loop
        fps_ctr = 0
        t_fps = time.time()
        while not self.stopped:
            try:
                ok, frame = self.video.read()
                if ok:
                    with self._cond_raw:
                        self._raw_frame = frame
                        self._raw_frame_id += 1
                        self.frame_count += 1
                        fps_ctr += 1
                        self._cond_raw.notify_all()

                    if fps_ctr >= 30:
                        dt = time.time() - t_fps
                        if dt > 0: self.fps = round(fps_ctr / dt, 1)
                        fps_ctr, t_fps = 0, time.time()
                else:
                    time.sleep(0.01)
            except Exception as e:
                time.sleep(0.5)

    # ── T2: Detect (Nitro Mode) ────────────────────────────────────
    def _detect_thread(self):
        SCALE = 0.25
        last_id = -1
        proc_ctr = 0
        t_proc = time.time()
        
        while not self.stopped:
            frame = None
            with self._cond_raw:
                while not self.stopped and self._raw_frame_id == last_id:
                    if not self._cond_raw.wait(0.1):
                        continue
                if self.stopped: break
                
                # Critical Optimization: Shallow copy or immediate release
                frame = self._raw_frame.copy() 
                last_id = self._raw_frame_id
            
            # Staggered Detection: Only process the N-th frame
            if last_id % self.process_nth_frame != 0:
                continue

            try:
                t_det = time.time()
                # Fast Resize: Nearest neighbor is 5x faster than Linear for detection
                small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                
                # We use Haar as Nitro default for speed
                locs_small = []
                if self.models_ready and 'haar' in self.models:
                    rects = self.models['haar'].detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
                    locs_small = [(int(y), int(x+w), int(y+h), int(x)) for (x,y,w,h) in rects]
                
                locs_full = [(int(t/SCALE), int(r/SCALE), int(b/SCALE), int(l/SCALE)) for (t, r, b, l) in locs_small]
                
                with self._lock_locs:
                    self._face_locs = locs_full
                    self._face_locs_id += 1
                    proc_ctr += 1

                self._perf_detect_ms = int((time.time() - t_det) * 1000)
                
                dt = time.time() - t_proc
                if proc_ctr >= 20 and dt > 0:
                    self.processing_fps = round(proc_ctr / dt, 1)
                    proc_ctr, t_proc = 0, time.time()
            except: pass
            time.sleep(0.001)

    # ── T3: Recognition (Fully Async) ──────────────────────────────
    def _recog_thread(self):
        while not self.stopped:
            try:
                # Wait for models to be ready
                if not self.models_ready:
                    time.sleep(0.5)
                    continue

                with self._lock_locs:
                    locs = list(self._face_locs)
                    with self._lock_raw:
                        frame = self._raw_frame.copy()

                if not locs:
                    time.sleep(0.2)
                    continue

                # Recog logic
                faiss_index = self.models.get('faiss_index')
                faiss_labels = self.models.get('faiss_labels')
                names, confs, emos = [], [], []
                
                # Convert to RGB for face_recognition
                rgb_frame = frame[:, :, ::-1]
                rgb_frame = np.ascontiguousarray(rgb_frame)
                
                # In Nitro-Optimized mode, we prioritize the primary (largest) face
                # Sort by area to ensure the most important subject is recognized first
                locs.sort(key=lambda l: (l[1]-l[3])*(l[2]-l[0]), reverse=True)
                test_locs = locs[:1] 
                encodings = []
                
                if faiss_index:
                    try:
                        # face_recognition expects (top, right, bottom, left)
                        encodings = face_recognition.face_encodings(rgb_frame, test_locs)
                    except Exception as e:
                        print(f"[NITRO-RECOG] Encoding Error: {e}")

                for i, loc in enumerate(test_locs):
                    name, conf, emo = "Unknown", 0.0, "Neural"
                    
                    # 1. Face Recognition
                    if i < len(encodings) and faiss_index:
                        try:
                            encoding = encodings[i].reshape(1, -1).astype('float32')
                            
                            # FAISS Search: top 1 neighbor
                            distances, indices = faiss_index.search(encoding, 1)
                            dist = distances[0][0]
                            idx = indices[0][0]
                            
                            # Standard face_recognition threshold is ~0.4-0.6 (Squared L2)
                            is_match = dist <= 0.45 
                            
                            if is_match and idx in faiss_labels:
                                name = faiss_labels[idx]
                                conf = max(0, (1.0 - dist))
                        except Exception as e:
                            print(f"[NITRO-RECOG] Prediction Error: {e}")

                    # 2. Emotion Detection
                    try:
                        top, right, bottom, left = loc
                        roi = frame[max(0,top):bottom, max(0,left):right]
                        if roi.size > 0:
                            emo, _ = self.models['emotion'].detect_emotion(roi)
                    except: pass
                    
                    names.append(name); confs.append(conf); emos.append(emo)
                
                # Fill remaining detections if any
                while len(names) < len(locs): 
                    names.append("Scanning..."); confs.append(0.0); emos.append("...")

                with self._lock_recog:
                    self._recog_results = list(zip(names, confs, emos))
                    self._recog_locs = locs
                    self.last_predictions_count += 1
                
                time.sleep(0.6) # Increased to 1.6 FPS recognition for better Video FPS stability
            except Exception as e: 
                print(f"[NITRO-RECOG] Thread Error: {e}")
                time.sleep(0.5)

    # ── T4: Encode (Render Isolated) ──────────────────────────────
    def _encode_thread(self):
        last_raw_id = -1
        while not self.stopped:
            t_start = time.time()
            try:
                with self._cond_raw:
                    while not self.stopped and self._raw_frame_id == last_raw_id:
                        if not self._cond_raw.wait(0.1):
                            continue
                    if self.stopped: break
                    frame = self._raw_frame.copy()
                    last_raw_id = self._raw_frame_id

                # Draw UI overlays (Thread-safe read of results)
                with self._lock_locs:
                    locs = list(self._face_locs)
                with self._lock_recog:
                    res = list(self._recog_results)
                    r_locs = list(self._recog_locs)

                t_render = time.time()
                for i, loc in enumerate(locs):
                    name, emo = "Scanning", "..."
                    for j, rloc in enumerate(r_locs):
                        # Match detection box with recognition results based on proximity
                        if abs(loc[0]-rloc[0]) < 50:
                            if j < len(res): name, _, emo = res[j]
                            break

                    # Dynamic Color: Green if identified, Red if Unknown, Scanning or Analyzing
                    is_identified = name not in ["Unknown", "Scanning", "Analyzing...", "Scanning..."]
                    color = (0, 255, 0) if is_identified else (0, 0, 255) # BGR: Green (Identified) vs Red (Unauthorized)
                    
                    cv2.rectangle(frame, (loc[3], loc[0]), (loc[1], loc[2]), color, 2)
                    cv2.putText(frame, f"{name} | {emo}", (loc[3], loc[0]-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

                # Performance Label
                perf_text = f"CAP:{self.fps} | PROC:{self.processing_fps} | DET:{self._perf_detect_ms}ms"
                cv2.putText(frame, perf_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    with self._cond_output:
                        self._output_jpeg = buf.tobytes()
                        self._output_id += 1
                        self._cond_output.notify_all()
                
                self._perf_encode_ms = int((time.time() - t_render) * 1000)
                time.sleep(max(0, 0.033 - (time.time()-t_start))) # Target 30 FPS render
            except: time.sleep(0.01)

    # ── Public API ────────────────────────────────────────────────
    def get_frame(self, last_id=-1):
        if self.error_msg:
            return get_placeholder_frame(f"SYSTEM ERROR\n{self.error_msg}"), -100

        # Phase 1: Models
        if not self.models_ready:
            elapsed = int(time.time() - self.start_time)
            return get_placeholder_frame(f"SHIELD INITIALIZING\nLoading AI Models... [{elapsed}s]"), -1
        
        # Phase 2: Hardware
        if not self.camera_ready:
            elapsed = int(time.time() - self.start_time)
            msg = f"HARDWARE WARMUP\nBackend: {self.current_backend} [{elapsed}s]\nSearching for Sensors..."
            return get_placeholder_frame(msg), -2

        # Phase 3: Streaming
        with self._cond_output:
            if self._output_id > last_id and self._output_jpeg is not None:
                return self._output_jpeg, self._output_id
            
            self._cond_output.wait(0.5)
            
            if self._output_jpeg is not None:
                return self._output_jpeg, self._output_id
            
            return get_placeholder_frame("PIPELINE STALLED\nWaiting for Frame Data..."), -3

    def get_predictions(self):
        with self._lock_locs:
            locs = list(self._face_locs)
        with self._lock_recog:
            res = list(self._recog_results)
        
        preds = []
        for i, loc in enumerate(locs):
            name, conf, emo = (res[i] if i < len(res) else ("Unknown", 0.0, "Neutral"))
            preds.append((name, loc, conf, emo, 0.0))
        return preds

    def set_model(self, model_type):
        self.current_model_type = model_type

    def capture_face_roi(self, face_idx):
        """Capture the i-th face from the current raw frame and return JPEG bytes"""
        with self._lock_locs:
            if face_idx >= len(self._face_locs):
                return None
            loc = self._face_locs[face_idx]
        
        with self._lock_raw:
            if self._raw_frame is None: return None
            frame = self._raw_frame.copy()
        
        top, right, bottom, left = loc
        # Crop with some padding
        h, w = frame.shape[:2]
        pad = 20
        top = max(0, top - pad)
        left = max(0, left - pad)
        bottom = min(h, bottom + pad)
        right = min(w, right + pad)
        
        face_img = frame[top:bottom, left:right]
        if face_img.size == 0: return None
        
        ret, buf = cv2.imencode('.jpg', face_img)
        return buf.tobytes() if ret else None

    def reload_models(self):
        """Force a reload of models in a background thread"""
        threading.Thread(target=self._model_load_thread, daemon=True).start()

    def __del__(self):
        self.stopped = True
        if self.video: self.video.release()
