import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Import your custom modules from MultiFaceAnalytics/src
# (Ensure your folder structure allows these imports)
from MultiFaceAnalytics.src.detector import FaceDetector
from MultiFaceAnalytics.src.emotion import EmotionRecognizer
from MultiFaceAnalytics.src.age_gender import AgeGenderPredictor
from MultiFaceAnalytics.src.pose import HeadPoseEstimator
from MultiFaceAnalytics.src.quality import FaceQualityAssessor

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

# Initialize models (cached to load only once)
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
        
        # 1. Detect faces using your pipeline
        faces = detector.detect(img)
        
        for face in faces:
            # Draw bounding box and run analysis modules
            # (Integrate your specific src/ functions here based on your codebase structure)
            bbox = face['box']
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            if enable_emotion:
                emotion, conf = emotion_rec.predict(img, bbox)
                cv2.putText(img, f"{emotion} ({conf:.1f}%)", (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit-WebRTC component to stream browser webcam to Python backend
webrtc_streamer(
    key="multiface-analytics",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
