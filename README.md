# Real-Time Face Recognition & Emotion Detection System

A comprehensive surveillance system capable of real-time face recognition, tracking, and emotion analysis using a hybrid approach (OpenCV + Dlib).

## Features
-   **Real-Time Face Recognition**: Identifies known individuals with high accuracy using KNN and Dlib.
-   **Emotion Detection**: Detects emotions (Happy, Sad, Stressed, Neutral, etc.) using a FerPlus ONNX model.
-   **Accuracy Confidence**: Displays confidence percentage for both identity and emotion.
-   **Web Dashboard**: Modern interface to view live feed, upload files, and manage models.
-   **Hybrid Detection**: Supports Haar Cascade (Fast), HOG (Balanced), and CNN (Accurate) models.

## Prerequisites
-   Python 3.10 or higher
-   CMake (required for dlib)

## Installation

1.  **Clone the repository** (if not already done).

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Model
Before the system can recognize faces, you need to train it with images of known people.

1.  **Capture/Add Images**:
    -   Create a folder in `src2/training_data/` with the person's name (e.g., `src2/training_data/obama`).
    -   Add clear images of their face to this folder.
    -   Alternatively, use the capture tool:
        ```bash
        python3 src2/capture_training_faces.py --name "Person Name"
        ```

2.  **Train**:
    ```bash
    python3 src2/train_model.py
    ```
    This will generate `src2/trained_knn_model.clf`.

### 2. Web Interface (Recommended)
The easiest way to use the system is via the web application.

1.  **Start the Server**:
    ```bash
    python3 src2/app.py
    ```

2.  **Open Browser**:
    Go to `http://localhost:5001` or `http://0.0.0.0:5001`.

3.  **Features**:
    -   **Live Feed**: View real-time recognition from your webcam.
    -   **Upload**: Upload images or videos for analysis.
    -   **Stats**: View detection statistics.

### 3. Command Line Interface
You can also process individual files via the command line.

```bash
# Process an image
python3 src2/recognize_media.py path/to/image.jpg

# Process a video
python3 src2/recognize_media.py path/to/video.mp4

# Use a specific detection model (hog is more accurate than default haar)
python3 src2/recognize_media.py path/to/image.jpg --model hog
```

## Project Structure
-   `src2/app.py`: Flask web server entry point.
-   `src2/camera.py`: Video streaming and processing logic.
-   `src2/recognize_media.py`: CLI for file processing.
-   `src2/emotion_detector.py`: Emotion detection module.
-   `src2/train_model.py`: Script to train the KNN classifier.
-   `src2/training_data/`: Directory to store labeled face images.

## Troubleshooting
-   **Dlib Error**: If `dlib` fails to install, ensure you have CMake installed (`brew install cmake` on Mac, or install from VS Build Tools on Windows).
-   **Model Not Found**: Ensure `trained_knn_model.clf` exists in `src2/`. If not, run `train_model.py`.
