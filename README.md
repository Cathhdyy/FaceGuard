# FaceGuard: AI Surveillance with Nitro Engine (v2.0)

A high-performance, real-time facial recognition and emotion analysis system. Powered by the **Nitro Multi-threaded Engine**, FaceGuard delivers a silky-smooth 30 FPS experience with near-instant identification.

## 🚀 Nitro Engine v2.0 Features

-   **Silky Smooth 30 FPS**: Unlocked hardware acceleration using the **MSMF (Windows Media Foundation)** backend, doubling performance from 15 FPS to 30 FPS.
-   **Nitro Multi-threading**: A decoupled 4-stage asynchronous pipeline (Capture → Detect → Recognize → Render) ensuring video delivery is never blocked by AI computation.
-   **Instant Recognition**: Powered by the **FAISS (Facebook AI Similarity Search)** backend for sub-millisecond facial vector matching across thousands of profiles.
-   **Turbo Training**: Parallelized background training that engages **all logical CPU cores**, reducing model update times by up to 6x.
-   **Live Face Training**: Register and train unknown faces directly from the live dashboard without stopping the stream.
-   **Dynamic Resource Guarding**: Smart CPU allocation that reserves cores for video stability even during heavy model retraining.

## 📈 Major Architectural Evolutions (vs v1.0)

FaceGuard v2.0 is a complete ground-up rebuild of the original FaceGuard01 architecture, focused on eliminating latency and maximizing hardware utilization.

| Feature | FaceGuard v1.0 (Legacy) | FaceGuard v2.0 (Nitro) |
| :--- | :--- | :--- |
| **Video Playback** | 10-15 FPS (Laggy) | **30 FPS (Silky Smooth)** |
| **Pipeline** | Synchronous (AI blocks video) | **4-Stage Async (Decoupled)** |
| **Camera Backend** | DirectShow (Legacy Windows) | **MSMF (Media Foundation)** |
| **Face Matching** | Linear Python Loops | **FAISS (C++ Optimized)** |
| **AI Training** | Single-core (Slow) | **Multi-core Parallel (6x Faster)** |
| **Model Updates** | Manual Restart Required | **Hot-Reloading (Live Training)** |
| **UI Feedback** | Static Text Overlays | **Dynamic Bento-Dashboard** |

## 📊 Core Capabilities

-   **Face Recognition**: High-accuracy identification using KNN and Dlib descriptors.
-   **Emotion Analytics**: Real-time detection of emotions (Happy, Sad, Neutral, etc.) using a FerPlus ONNX model.
-   **Hybrid Detection**: Flexible support for **Haar Cascade** (Fast), **HOG** (Balanced), and **CNN** (Precise) models.
-   **Security Overlay**: Live CAP/PROC/DET performance metrics and color-coded security boxes.

## 🛠️ Quick Start (Windows)

1.  **Installation**:
    -   Ensure you have **Python 3.10+** and **CMake** installed.
    -   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Launch**:
    -   Simply double-click **`run_FaceGuard.bat`**. This will automatically handle environment setup and start the server.

3.  **Access Dashboard**:
    -   Open your browser to `http://localhost:5001`.

## 📂 Project Structure

-   **`src2/app.py`**: Flask Web Dashboard & API controller.
-   **`src2/camera.py`**: The Nitro Engine (multi-threaded processing logic).
-   **`src2/train_model.py`**: Nitro-parallelized training pipeline.
-   **`src2/recognize_media.py`**: CLI tool for processing offline images and videos.
-   **`src2/training_data/`**: Storage for your labeled face datasets.

## 💡 Pro Tips

-   **Performance Monitoring**: Look at the overlay in the live feed. **CAP** shows your hardware capture rate, while **PROC** shows the AI processing speed.
-   **Staggered Detection**: By default, FaceGuard processes AI every 2nd frame to maintain a perfect 30 FPS video feed while saving 50% CPU.
-   **Model Switching**: For the best balance of speed and accuracy, use the **Haar** detector for tracking and **KNN** for identification.

---
*Developed with focus on extreme performance and reliability.*
