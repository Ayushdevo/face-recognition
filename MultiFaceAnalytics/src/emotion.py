import os
import cv2
import numpy as np
from src.utils import detect_device, download_file

class EmotionRecognizer:
    """
    Classifies facial expressions into one of 7 categories:
    Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral.
    Supports ONNX model inference with fallback to landmark-based heuristics.
    """
    def __init__(self, models_dir="./models", prefer_heuristic=True):
        self.models_dir = os.path.abspath(models_dir)
        self.model_path = os.path.join(self.models_dir, "emotion_model.onnx")
        self.classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        self.prefer_heuristic = prefer_heuristic
        
        # Public model weights download URL (MiniXception trained on FER2013)
        self.download_url = "https://github.com/spmallick/learnopencv/raw/master/Facial-Expression-Recognition-using-PyTorch/model.onnx"
        
        self.use_onnx = False
        self.ort_session = None
        self.net = None
        
        # Load the model
        self._load_model()

    def _load_model(self):
        """
        Attempts to load the ONNX model. Downloads if not present.
        """
        if not os.path.exists(self.model_path):
            print(f"Emotion model not found at {self.model_path}.")
            # Try downloading
            success = download_file(self.download_url, self.model_path)
            if not success:
                print("[WARNING] Could not download emotion model. Using landmark heuristic fallback.")
                return

        # Try loading with ONNX Runtime
        try:
            import onnxruntime as ort
            gpu_available, backend = detect_device()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu_available else ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
            self.use_onnx = True
            print(f"Loaded Emotion ONNX model successfully with ONNX Runtime on {backend}.")
            return
        except ImportError:
            # Fall back to OpenCV DNN
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                # Try setting CUDA backend if available
                gpu_available, _ = detect_device()
                if gpu_available:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.use_onnx = True
                print("Loaded Emotion ONNX model successfully with OpenCV DNN.")
                return
            except Exception as e:
                print(f"[WARNING] Failed to load emotion model via OpenCV DNN: {e}")
                
        print("[WARNING] Running Emotion Estimation in Heuristic Fallback mode.")

    def predict_emotion(self, face_patch, landmarks_2d=None):
        """
        Predicts the emotion of the face patch.
        Args:
            face_patch: BGR image crop of the face.
            landmarks_2d: 2D coordinates from MediaPipe (used for heuristic fallback).
        Returns:
            dict: { 'emotion': str, 'probabilities': dict }
        """
        if not self.prefer_heuristic and self.use_onnx and face_patch is not None and face_patch.size > 0:
            try:
                # MiniXception standard: input is 1x1x64x64 (or 1x1x48x48)
                # We dynamically query the input dimensions of the model
                if self.ort_session:
                    input_shape = self.ort_session.get_inputs()[0].shape
                    h, w = input_shape[2], input_shape[3]
                else:
                    h, w = 64, 64 # Default OpenCV fallback

                # Preprocess: convert to gray, resize, normalize, add dimensions
                gray = cv2.cvtColor(face_patch, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (w, h))
                normalized = resized.astype(np.float32) / 255.0
                
                if self.ort_session:
                    # Model expects shape [batch, channels, height, width]
                    blob = np.expand_dims(np.expand_dims(normalized, 0), 0)
                    input_name = self.ort_session.get_inputs()[0].name
                    outputs = self.ort_session.run(None, {input_name: blob})
                    logits = outputs[0][0]
                else:
                    # OpenCV DNN
                    blob = cv2.dnn.blobFromImage(resized, 1.0/255.0, (w, h), 0, swapRB=False, crop=False)
                    self.net.setInput(blob)
                    logits = self.net.forward()[0]

                # Softmax calculation
                exp_logits = np.exp(logits - np.max(logits)) # stable softmax
                probs = exp_logits / np.sum(exp_logits)
                
                prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
                max_idx = np.argmax(probs)
                
                return {
                    "emotion": self.classes[max_idx],
                    "probabilities": prob_dict,
                    "confidence": float(probs[max_idx])
                }
            except Exception as e:
                # If ONNX fails at runtime, fall back to heuristic
                pass
                
        # --- Heuristic Landmark-Based Fallback ---
        return self._predict_heuristic(landmarks_2d)

    def _predict_heuristic(self, landmarks_2d):
        """
        Heuristic classification using geometric ratios of facial landmarks.
        """
        # Default probabilities (neutral)
        probs = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70]) # A, D, F, H, S, Su, N
        
        if landmarks_2d is None or len(landmarks_2d) < 300:
            # Fallback when landmarks are not available
            prob_dict = {self.classes[i]: float(probs[i]) for i in range(7)}
            return {"emotion": "Neutral", "probabilities": prob_dict, "confidence": 0.7}

        try:
            # 1. Smile Score (Happy)
            # Distance between left/right mouth corners (idx 61, 291) relative to eye distance
            mouth_w = np.linalg.norm(np.array(landmarks_2d[61]) - np.array(landmarks_2d[291]))
            eye_w = np.linalg.norm(np.array(landmarks_2d[33]) - np.array(landmarks_2d[263]))
            smile_ratio = mouth_w / max(1.0, eye_w)
            
            # 2. Mouth Open Score (Surprise / Fear)
            # Vertical mouth distance (idx 13, 14) relative to mouth width
            mouth_h = np.linalg.norm(np.array(landmarks_2d[13]) - np.array(landmarks_2d[14]))
            mar = mouth_h / max(1.0, mouth_w)

            # 3. Eyebrow Frown Score (Angry / Sad)
            # Distance between eyebrows inner points (idx 107, 336) or distance from eyebrow to eye center
            eyebrow_dist = np.linalg.norm(np.array(landmarks_2d[107]) - np.array(landmarks_2d[336]))
            frown_ratio = eyebrow_dist / max(1.0, eye_w)

            # Define heuristic raw logits
            logits = np.zeros(7) # A, D, F, H, S, Su, N
            
            # Neutral baseline
            logits[6] = 1.0 
            
            # Happy: high smile ratio
            # Baseline smile ratio is ~0.65-0.70. High smile is >0.78
            if smile_ratio > 0.75:
                logits[3] = (smile_ratio - 0.70) * 12.0 # Happy
                logits[6] = 0.5 # lower neutral
                
            # Surprise / Fear: high MAR
            if mar > 0.25:
                logits[5] = (mar - 0.20) * 10.0 # Surprise
                logits[2] = (mar - 0.20) * 4.0  # Fear
                logits[6] = 0.2
                
            # Angry / Sad: narrow eyebrow distance
            # Normal eyebrow frown ratio is ~0.25. Narrow is <0.20
            if frown_ratio < 0.21:
                logits[0] = (0.23 - frown_ratio) * 15.0 # Angry
                logits[4] = (0.23 - frown_ratio) * 8.0  # Sad
                logits[6] = 0.1
            
            # Applying Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
        except Exception:
            pass

        prob_dict = {self.classes[i]: float(probs[i]) for i in range(7)}
        max_idx = np.argmax(probs)
        
        return {
            "emotion": self.classes[max_idx],
            "probabilities": prob_dict,
            "confidence": float(probs[max_idx])
        }
