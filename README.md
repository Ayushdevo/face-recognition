# 🎭 MultiFace Analytics AI

> **Production-Grade Real-Time Multi-Face Analysis System using Computer Vision and Deep Learning**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-success.svg)

---

## 📖 Overview

**MultiFace Analytics AI** is a production-ready computer vision system capable of detecting, tracking, and analyzing multiple faces simultaneously using a standard webcam or video stream.

Each detected face is assigned a persistent tracking ID and analyzed in real time for:

* 😀 Emotion Recognition
* 👤 Age Estimation
* 🚻 Gender Prediction
* 🎯 Face Detection Confidence
* ⭐ Face Quality Score
* 🧭 Head Pose Estimation
* 📍 Facial Landmark Detection
* 📊 FPS & Performance Monitoring

Designed with a modular architecture, GPU acceleration, and optimized inference, this project demonstrates a complete real-time face analytics pipeline suitable for research, education, and AI portfolio projects.

---

# ✨ Features

### 👥 Multi-Face Detection

* Detect multiple faces simultaneously
* Supports side profiles
* Handles partially visible faces
* Works under varying lighting conditions
* Detects faces entering and leaving the frame

---

### 🔄 Face Tracking

Every detected face receives a persistent ID.

Example:

```
Face #1
Face #2
Face #3
```

Features:

* Stable tracking IDs
* Multi-object tracking
* Automatic re-identification (short-term)
* Handles crossing faces
* Smooth tracking

---

### 😀 Emotion Recognition

Recognizes:

* Happy
* Sad
* Angry
* Fear
* Surprise
* Disgust
* Neutral

Example

```
Happy (98.4%)
```

---

### 👤 Age Estimation

Example

```
23 Years
```

or

```
22–26 Years
```

---

### 🚻 Gender Prediction

Example

```
Male (99.1%)
Female (98.5%)
```

---

### ⭐ Face Quality Assessment

Face quality is calculated using

* Blur Detection
* Brightness
* Contrast
* Face Size
* Face Angle
* Landmark Visibility
* Occlusion Estimation

Example

```
Quality
93%
```

---

### 🎯 Detection Confidence

Displays detector confidence.

Example

```
Detection
99.8%
```

---

### 🧭 Head Pose Estimation

Estimates

* Pitch
* Roll
* Yaw

Example

```
Yaw: 4°
Pitch: -2°
Roll: 1°
```

---

### 📍 Face Landmarks

Draws facial landmarks including

* Eyes
* Eyebrows
* Nose
* Mouth
* Jawline

---

### 📊 Performance Metrics

Displays

* FPS
* Inference Time
* Active Faces
* Average Confidence
* Average Face Quality

---

# 🖥 Example Output

```
------------------------------------------------
Face #2

😀 Happy (97%)

👤 Male

🎂 24 Years

⭐ Quality : 92%

🎯 Detection : 99%

Yaw : 4°

Pitch : -2°

Roll : 1°
------------------------------------------------
```

---

# 📂 Project Structure

```
MultiFaceAnalytics/

│

├── models/
│   ├── emotion/
│   ├── age/
│   ├── gender/
│   ├── detector/
│   └── tracker/
│
├── dataset/
│
├── outputs/
│   ├── screenshots/
│   ├── recordings/
│   └── logs/
│
├── src/
│   ├── detector.py
│   ├── tracker.py
│   ├── emotion.py
│   ├── age_gender.py
│   ├── quality.py
│   ├── pose.py
│   ├── landmarks.py
│   ├── logger.py
│   ├── utils.py
│   └── webcam.py
│
├── train.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🧠 AI Models

| Task                | Model                      |
| ------------------- | -------------------------- |
| Face Detection      | RetinaFace / MediaPipe     |
| Face Tracking       | ByteTrack / DeepSORT       |
| Emotion Recognition | MobileNetV3 / EfficientNet |
| Age Estimation      | InsightFace / YuNet        |
| Gender Prediction   | InsightFace                |
| Face Mesh           | MediaPipe Face Mesh        |
| Head Pose           | SolvePnP                   |
| Inference           | ONNX Runtime               |

---

# ⚡ Installation

Clone the repository

```bash
git clone https://github.com/Ayushdevo/MultiFaceAnalytics.git

cd MultiFaceAnalytics
```

Create virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Application

```bash
python src/webcam.py
```

---

# ⌨ Keyboard Controls

| Key | Action            |
| --- | ----------------- |
| Q   | Quit              |
| S   | Save Screenshot   |
| R   | Start Recording   |
| T   | Stop Recording    |
| L   | Toggle Landmarks  |
| F   | Toggle FPS        |
| C   | Toggle Confidence |

---

# 📊 Performance Goals

| Metric          | Target  |
| --------------- | ------- |
| CPU FPS         | 20+ FPS |
| GPU FPS         | 30+ FPS |
| Faces Supported | 10+     |
| Latency         | <50 ms  |
| GPU Support     | Yes     |
| ONNX Runtime    | Yes     |

---

# 📝 CSV Logging

Each detected face is logged.

Example

| Timestamp | Face ID | Emotion | Age | Gender | Confidence | Quality |
| --------- | ------- | ------- | --- | ------ | ---------- | ------- |
| 12:04:25  | 2       | Happy   | 24  | Male   | 99.1       | 92      |

---

# 💾 Outputs

```
outputs/

screenshots/

recordings/

logs/

predictions.csv
```

---

# 🚀 Future Improvements

* Face Recognition (ArcFace)
* Person Re-Identification
* Drowsiness Detection
* Blink Detection
* Smile Detection
* Eye Gaze Tracking
* Attention Score
* Mask Detection
* Glasses Detection
* Beard Detection
* Liveness Detection
* Anti-Spoofing
* Emotion Timeline Visualization
* REST API (FastAPI)
* Streamlit Dashboard
* Docker Deployment
* TensorRT Optimization
* Edge Device Deployment (Jetson/Raspberry Pi)

---

# 🛠 Tech Stack

* Python 3.11+
* OpenCV
* MediaPipe
* TensorFlow / Keras
* PyTorch
* ONNX Runtime
* NumPy
* Pandas
* SciPy
* Matplotlib
* scikit-learn

---

# 🎯 Applications

* Smart Attendance Systems
* Retail Analytics
* Classroom Monitoring
* Human-Computer Interaction
* Driver Monitoring Systems
* Emotion AI Research
* Security & Surveillance
* Healthcare Analytics
* Customer Experience Analytics
* Robotics & AI Assistants

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Ayush Tiwari**

AI Engineer • Data Scientist • Computer Vision Enthusiast

---

## ⭐ If you found this project useful, consider giving it a Star on GitHub!

Building intelligent systems, one frame at a time.
