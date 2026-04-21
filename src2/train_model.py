#!/usr/bin/env python3
"""
Nitro-Tuned Training Engine for FaceGuard
──────────────────────────────────────────
Upgrades:
- Multi-processing support for 4x+ speed
- Encoding cache for instant re-runs
- Auto-resizing for training efficiency
"""

import face_recognition
import os
import pickle
import faiss
import numpy as np
import sys
import cv2
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_image_encoding(image_path, mtime, cache_entry=None):
    """Worker function for individual image encoding"""
    # 1. Check cache first
    if cache_entry and cache_entry.get('mtime') == mtime:
        return cache_entry['encoding']

    try:
        # Load and Resize for efficiency (2-6x speed multiplier)
        img = cv2.imread(image_path)
        if img is None: return None
        
        h, w = img.shape[:2]
        MAX_W = 640
        if w > MAX_W:
            scale = MAX_W / w
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Neural pass
        encodings = face_recognition.face_encodings(rgb)
        return encodings[0] if encodings else None
    except:
        return None

def train_face_recognition_model(training_dir="training_data", model_save_path="trained_knn_model.clf", n_neighbors=3):
    start_total = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(base_dir, "training_cache.pkl")
    
    print("=" * 60)
    print("🚀 NITRO AI TRAINING ENGINE")
    print("=" * 60)
    
    if not os.path.exists(training_dir):
        print(f"Error: Training directory '{training_dir}' not found!")
        return False

    # ── Load Cache ───────────────────────────────────────────────────
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f: cache = pickle.load(f)
            print(f"[CACHE] Loaded {len(cache)} existing encodings")
        except: pass

    # ── Prepare Workload ──────────────────────────────────────────────
    X, y = [], []
    work_items = [] # (path, mtime, person_name)
    
    people = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d)) and not d.startswith('.')]
    
    for person_name in people:
        person_dir = os.path.join(training_dir, person_name)
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for f in image_files:
            p = os.path.join(person_dir, f)
            mt = os.path.getmtime(p)
            work_items.append((p, mt, person_name))

    print(f"Total images found: {len(work_items)}")
    
    # ── Multi-Processed Encoding ─────────────────────────────────────
    print(f"Processing images using multi-core logic...")
    new_cache = {}
    hits, misses = 0, 0
    
    # We use a smaller pool to leave room for the live camera/OS
    max_workers = max(1, os.cpu_count() - 2) 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for path, mtime, name in work_items:
            # Check if hit
            if path in cache and cache[path]['mtime'] == mtime:
                X.append(cache[path]['encoding'])
                y.append(name)
                new_cache[path] = cache[path]
                hits += 1
            else:
                # Dispatch for processing
                futures[executor.submit(get_image_encoding, path, mtime)] = (path, mtime, name)
                misses += 1
        
        if misses > 0:
            print(f"Encoding {misses} new/modified images...")
            for future in as_completed(futures):
                path, mtime, name = futures[future]
                enc = future.result()
                if enc is not None:
                    X.append(enc)
                    y.append(name)
                    new_cache[path] = {'encoding': enc, 'mtime': mtime}
    
    # ── Save Cache ───────────────────────────────────────────────────
    with open(cache_path, 'wb') as f:
        pickle.dump(new_cache, f)

    if not X:
        print("Error: No faces could be encoded.")
        return False

    # ── FAISS Indexing ──────────────────────────────────────────────
    print(f"Building FAISS Index (L2 Flat)...")
    X_matrix = np.array(X).astype('float32')
    index = faiss.IndexFlatL2(128)
    index.add(X_matrix)
    labels = {i: name for i, name in enumerate(y)}
    
    # Save artifacts
    faiss_save_path = model_save_path.replace(".clf", ".index")
    labels_save_path = model_save_path.replace(".clf", "_labels.pkl")
    faiss.write_index(index, faiss_save_path)
    with open(labels_save_path, 'wb') as f:
        pickle.dump(labels, f)
        
    print(f"Training SUCCESS in {time.time() - start_total:.2f}s")
    print(f"Stats: {hits} cached, {misses} processed")
    print("=" * 60)
    return True

if __name__ == "__main__":
    train_face_recognition_model()
