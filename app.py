import sys
import os

# Add the MultiFaceAnalytics directory to Python path so internal 'src' imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "MultiFaceAnalytics"))

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Import custom modules from your MultiFaceAnalytics/src package
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

# Sidebar controls for toggles
st.sidebar.header("Configuration")
enable_emotion = st.sidebar.checkbox("Emotion Recognition", value=True)
enable_age_gender = st.sidebar.checkbox("Age & Gender Prediction", value=True)
enable_pose = st.sidebar.checkbox("Head Pose Estimation", value=True)
enable_quality = st.sidebar.checkbox("Face Quality Assessment", value=True)

# Initialize models (cached to load only once across reruns)
@st.cache_resource
def load_models():
    detector = FaceDetector()
    emotion_recognizer = EmotionRecognizer()
    age_gender_predictor = AgeGenderPredictor()
    pose_estimator = HeadPoseEstimator()
    quality_assessor = FaceQualityAssessor()
    return detector, emotion_recognizer, age_gender_predictor, pose_estimator, quality_assessor

detector, emotion_rec, age_gender_pred, pose_est, quality_assessor = load_models()

# Video processing callback for WebRTC live stream
class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Detect faces using your pipeline
        faces = detector.detect(img)
        
        for face in faces:
            # Extract bounding box coordinates based on your detector output format
            bbox = face.get('box', face.get('bbox'))
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Optional module overlays depending on sidebar toggles
            if enable_emotion:
                try:
                    emotion, conf = emotion_rec.predict(img, [x1, y1, x2, y2])
                    cv2.putText(img, f"{emotion} ({conf:.1f}%)", (x1, max(y1 - 10, 15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception:
                    pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit-WebRTC component to stream browser webcam to Python backend
webrtc_streamer(
    key="multiface-analytics",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
