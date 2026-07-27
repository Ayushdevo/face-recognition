import sys
import os
import subprocess

# Add the MultiFaceAnalytics directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "MultiFaceAnalytics", "models")
sys.path.append(os.path.join(current_dir, "MultiFaceAnalytics"))

# Automatically generate placeholder/trained ONNX models on first startup if missing
if not os.path.exists(models_dir) or not os.listdir(models_dir):
    print("Models directory missing or empty. Generating ONNX models...")
    train_script = os.path.join(current_dir, "MultiFaceAnalytics", "train.py")
    if os.path.exists(train_script):
        subprocess.run([sys.executable, train_script, "--mode", "placeholder", "--models-dir", models_dir])

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Import custom modules from your package
from src.detector import FaceDetector
from src.emotion import EmotionRecognizer
from src.age_gender import AgeGenderPredictor
from src.pose import HeadPoseEstimator
from src.quality import FaceQualityAssessor

st.set_page_config(
    page_title="MultiFace Analytics AI",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 MultiFace Analytics AI")
st.markdown("Production-Grade Real-Time Multi-Face Analysis System running in your browser.")

# Sidebar controls
st.sidebar.header("Configuration")
enable_emotion = st.sidebar.checkbox("Emotion Recognition", value=True)
enable_age_gender = st.sidebar.checkbox("Age & Gender Prediction", value=True)
enable_pose = st.sidebar.checkbox("Head Pose Estimation", value=True)
enable_quality = st.sidebar.checkbox("Face Quality Assessment", value=True)

# Initialize models (cached)
@st.cache_resource
def load_models():
    detector = FaceDetector()
    emotion_recognizer = EmotionRecognizer()
    age_gender_predictor = AgeGenderPredictor()
    pose_estimator = HeadPoseEstimator()
    quality_assessor = FaceQualityAssessor()
    return detector, emotion_recognizer, age_gender_predictor, pose_estimator, quality_assessor

detector, emotion_rec, age_gender_pred, pose_est, quality_assessor = load_models()

# Video processing callback for WebRTC
class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        faces = detector.detect(img)
        
        for face in faces:
            bbox = face.get('box', face.get('bbox'))
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if enable_emotion:
                try:
                    emotion, conf = emotion_rec.predict(img, [x1, y1, x2, y2])
                    cv2.putText(img, f"{emotion} ({conf:.1f}%)", (x1, max(y1 - 10, 15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception:
                    pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="multiface-analytics",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
