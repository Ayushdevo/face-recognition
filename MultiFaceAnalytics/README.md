# Real-Time Multi-Face Analytics Engine

A production-grade, real-time computer vision system that captures live video from a webcam, performs multi-face detection and tracking (with persistent ID assignment), and analyzes individual facial qualities and metrics (emotion probabilities, age, gender, head pose angles, and detailed facial landmarks) simultaneously.

---

## Key Features

1. **Robust Multi-Face Detection**: Integrates MediaPipe Face Mesh, tracking up to 20 faces in the frame. Handled side profiles, head movements, and extreme scales.
2. **Persistent Tracking**: Custom Kalman Filter + Hungarian matching tracker (based on IoU and Centroid distance) which handles temporary occlusions and re-identifies returning faces (grace period up to 30 frames).
3. **Multi-Task Face Analytics**:
   * **Emotion Recognition**: Probabilities across 7 classes (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral).
   * **Age Estimation**: Continuous age prediction in years.
   * **Gender Prediction**: Predicts male/female with confidence percentage.
   * **Head Pose Estimation**: Direct Euler angles (Yaw, Pitch, Roll) using 3D-to-2D projection mapping and `solvePnP`.
   * **Face Quality Score (0-100)**: Evaluates image focus/blur, contrast, brightness deviation, size, head pose angles, and symmetry.
4. **Hardware Acceleration**: Automatic GPU detection supporting ONNX Runtime (CUDA/DirectML) and OpenCV DNN CUDA backends.
5. **No-Crash Fallback System**: In isolated/offline environments without neural net model weights, the system defaults to high-accuracy geometric landmark-based analytical estimators, providing immediate testability and continuous run-times.
6. **Thread-Safe logging**: Face statistics are pushed to a background thread queue to write CSV reports without blocking the frame rate.

---

## Folder Structure

```text
MultiFaceAnalytics/
├── README.md                 # Setup guide and instructions
├── requirements.txt          # Python dependencies
├── train.py                  # Standalone training & ONNX export utility
├── models/                   # Folder containing ONNX models
└── src/
    ├── __init__.py           # Package marker
    ├── detector.py           # Face detection and landmark extraction (MediaPipe)
    ├── tracker.py            # Persistent face tracking (Kalman Filter + Hungarian)
    ├── emotion.py            # Emotion recognition model interface & fallback
    ├── age_gender.py         # Age and gender estimation model interface & fallback
    ├── quality.py            # Face quality score (0-100) computation
    ├── pose.py               # Head pose estimation (SolvePnP)
    ├── landmarks.py          # Landmark mesh rendering and styling utilities
    ├── logger.py             # Thread-safe CSV logger for face analytics
    ├── utils.py              # Visual overlays (glass panels, HUD) & device detector
    └── webcam.py             # Main entry point and OpenCV camera loop
```

---

## Quick Start Guide

### 1. Prerequisites & Environment Setup
Requires **Python 3.9+**. Open your shell and set up a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r MultiFaceAnalytics/requirements.txt
```

### 2. Model Weight Selection & Train Script
To generate optimal placeholder ONNX files, you can use the standalone training script. If PyTorch is installed, this script will construct valid network graphs and export them to your `models/` directory:

```bash
# Optional PyTorch installation for model generation/training
pip install torch torchvision

# Generate placeholder ONNX models
python MultiFaceAnalytics/train.py --mode placeholder
```
If you do not run this script, the system will attempt to download pre-trained weights from online servers at runtime, and automatically switch to the mathematical landmark-based fallback if offline.

### 3. Run the Main Application
Launch the real-time HUD interface using:

```bash
python MultiFaceAnalytics/src/webcam.py
```

---

## User Control & Shortcuts

The video stream window is fully interactive. Focus on the screen and use these hotkeys:

| Key | Action | HUD Display |
| :--- | :--- | :--- |
| **`Q`** | Quit application | Safely terminates all streams and background file writers. |
| **`S`** | Take Screenshot | Saves current frame with all active overlays in `screenshots/`. |
| **`R`** | Start Recording | Begins writing live video with overlays to `recordings/`. |
| **`T`** | Stop Recording | Saves recorded video file. |
| **`L`** | Toggle Landmarks | Toggles rendering of the dense 468 holographic mesh nodes. |
| **`F`** | Toggle FPS Stats | Shows/hides the system FPS and average inference duration in the stats HUD. |
| **`C`** | Toggle Confidence | Shows/hides individual face detection confidence percentages on overlay panels. |

---

## Logging Specifications
Logs are saved in CSV format at `logs/face_analytics.csv`. The log contains columns:
* **Timestamp**: Date and millisecond-level time (`YYYY-MM-DD HH:MM:SS.fff`)
* **Face_ID**: Unique track integer
* **Emotion**: Highest probability expression
* **Age**: Average smoothed age (in Years)
* **Gender**: Classified gender
* **Detection_Confidence**: Face detector score
* **Quality_Score**: Numerical visual quality score (0-100)
