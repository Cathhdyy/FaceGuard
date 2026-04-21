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
| **Video Playback** | 15 FPS (Laggy) | **30 FPS (Silky Smooth)** |
| **Search Engine** | KNN Linear Search | **FAISS Vector Search** |
| **Pipeline** | Synchronous (AI blocks video) | **Asynchronous (Nitro)** |
| **Camera Backend** | DirectShow (Legacy) | **MSMF (Modern)** |
| **AI Training** | Manual Single-core | **Parallel Multi-core** |
| **Model Updates** | Manual Restart Required | **Live In-Dashboard Training** |
| **User Interface** | Basic Sidebar Layout | **Premium Bento-Grid Dashboard** |

## 📊 Core Capabilities

-   **Face Recognition**: High-accuracy identification using KNN and Dlib descriptors.
-   **Emotion Analytics**: Real-time detection of emotions (Happy, Sad, Neutral, etc.) using a FerPlus ONNX model.
-   **Hybrid Detection**: Flexible support for **Haar Cascade** (Fast), **HOG** (Balanced), and **CNN** (Precise) models.
-   **Security Overlay**: Live CAP/PROC/DET performance metrics and color-coded security boxes.

## 🛠️ Installation & Setup (Windows)

Follow these steps to get FaceGuard up and running on your machine:

### 1. Prerequisites
- **Python 3.10+**: Ensure Python is installed and added to your system PATH.
- **CMake**: Required for `dlib` compilation. [Download here](https://cmake.org/download/).
- **Conda (Optional but Recommended)**: The project is optimized for Conda environments.

### 2. Environment Setup
We recommend using a dedicated environment to avoid dependency conflicts:
```powershell
# Create a new environment
conda create -n faceguard_env python=3.10

# Activate the environment
conda activate faceguard_env
```

### 3. Install Dependencies
Install the high-performance AI libraries required for the Nitro engine:
```powershell
pip install -r requirements.txt
```

### 4. Running the Application
The project includes a one-click launcher for convenience:
- **Option A (Launcher)**: Simply double-click the **`run_FaceGuard.bat`** file in the root directory.
- **Option B (Manual)**: Run the following command from your terminal:
  ```powershell
  python src2/app.py
  ```

Once started, open your browser and navigate to: **`http://localhost:5001`**

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
