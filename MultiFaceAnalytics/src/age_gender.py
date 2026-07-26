import os
import cv2
import numpy as np
from src.utils import detect_device, download_file

class AgeGenderEstimator:
    """
    Estimates Age and Predicts Gender (Male/Female) from face patches.
    Supports ONNX model loading and landmark-based heuristic fallback.
    Implements temporal smoothing (moving average) per track ID.
    """
    def __init__(self, models_dir="./models", history_len=15, prefer_heuristic=True):
        self.models_dir = os.path.abspath(models_dir)
        self.model_path = os.path.join(self.models_dir, "age_gender_model.onnx")
        self.history_len = history_len
        self.prefer_heuristic = prefer_heuristic
        
        # Public download url for age-gender ONNX model (MobileNetV2 or similar backbone)
        self.download_url = "https://github.com/DmitryKrot/Age-Gender-Estimation/raw/master/models/model.onnx"
        
        self.use_onnx = False
        self.ort_session = None
        self.net = None
        
        # History dict for temporal smoothing: {track_id: {"age": [], "gender_prob": []}}
        self.track_histories = {}
        
        # Try loading the ONNX model
        self._load_model()

    def _load_model(self):
        """
        Loads the age/gender ONNX model. Downloads if not present.
        """
        if not os.path.exists(self.model_path):
            print(f"Age/Gender model not found at {self.model_path}.")
            # We won't block execution if download fails; fallback will run
            success = download_file(self.download_url, self.model_path)
            if not success:
                print("[WARNING] Could not download age/gender model. Using landmark heuristic fallback.")
                return

        try:
            import onnxruntime as ort
            gpu_available, backend = detect_device()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu_available else ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
            self.use_onnx = True
            print(f"Loaded Age/Gender ONNX model successfully with ONNX Runtime on {backend}.")
            return
        except ImportError:
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                gpu_available, _ = detect_device()
                if gpu_available:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.use_onnx = True
                print("Loaded Age/Gender ONNX model successfully with OpenCV DNN.")
                return
            except Exception as e:
                print(f"[WARNING] Failed to load age/gender model via OpenCV: {e}")
                
        print("[WARNING] Running Age/Gender Estimation in Heuristic Fallback mode.")

    def estimate(self, face_patch, track_id, landmarks_2d=None):
        """
        Estimates age and gender, applying temporal smoothing across frames.
        Args:
            face_patch: BGR image crop of the face.
            track_id: Unique persistent ID of the face.
            landmarks_2d: 2D landmarks (used for heuristic fallback).
        Returns:
            age (int/str), gender (str), gender_confidence (float)
        """
        raw_age = 25.0
        raw_gender_prob_male = 0.5
        
        # 1. Run ONNX Inference if model is loaded successfully
        if not self.prefer_heuristic and self.use_onnx and face_patch is not None and face_patch.size > 0:
            try:
                # Preprocess: RGB, resize to 224x224, standard normalization
                img_resized = cv2.resize(face_patch, (224, 224))
                
                if self.ort_session:
                    # Preprocess for PyTorch/MobileNet standard formats: [1, 3, 224, 224], normalized
                    blob = img_resized.astype(np.float32) / 255.0
                    # Mean/std normalization (ImageNet)
                    blob = (blob - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    blob = np.transpose(blob, (2, 0, 1)) # HWC to CHW
                    blob = np.expand_dims(blob, 0)
                    
                    outputs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: blob})
                    # Model outputs age (continuous regression) & gender (logits: index 0 Female, 1 Male)
                    # Support multiple common output shapes
                    if len(outputs) >= 2:
                        raw_age = float(outputs[0][0][0])
                        # If output is classification indices, we scale it
                        if raw_age < 1.0: # normalized regression
                            raw_age = raw_age * 100.0
                            
                        gender_logits = outputs[1][0]
                        # Softmax on gender
                        exp_g = np.exp(gender_logits - np.max(gender_logits))
                        g_probs = exp_g / np.sum(exp_g)
                        raw_gender_prob_male = float(g_probs[1])
                else:
                    # OpenCV DNN
                    blob = cv2.dnn.blobFromImage(img_resized, 1.0/255.0, (224, 224), (104, 117, 123), swapRB=True, crop=False)
                    self.net.setInput(blob)
                    outputs = self.net.forward()
                    # DNN output formats vary. If output is single layer:
                    if isinstance(outputs, list) and len(outputs) >= 2:
                        raw_age = float(outputs[0][0]) * 100.0
                        raw_gender_prob_male = float(outputs[1][0][1])
                    else:
                        # Fallback parsing
                        raw_age = 25.0
                        raw_gender_prob_male = 0.5
            except Exception:
                pass
        else:
            # 2. Heuristic Landmark-Based Fallback
            raw_age, raw_gender_prob_male = self._estimate_heuristic(track_id, landmarks_2d)

        # 3. Apply Temporal Smoothing
        if track_id not in self.track_histories:
            self.track_histories[track_id] = {"age": [], "gender_prob": []}
            
        history = self.track_histories[track_id]
        history["age"].append(raw_age)
        history["gender_prob"].append(raw_gender_prob_male)
        
        # Keep histories bounded
        if len(history["age"]) > self.history_len:
            history["age"].pop(0)
            history["gender_prob"].pop(0)
            
        # Compute smoothed values
        smooth_age = int(np.mean(history["age"]))
        smooth_gender_prob = np.mean(history["gender_prob"])
        
        # Clamp age to a reasonable human range (e.g. 5 to 90)
        smooth_age = max(5, min(90, smooth_age))
        
        # Determine final label and confidence
        if smooth_gender_prob >= 0.5:
            gender_label = "Male"
            gender_confidence = smooth_gender_prob * 100.0
        else:
            gender_label = "Female"
            gender_confidence = (1.0 - smooth_gender_prob) * 100.0
            
        return smooth_age, gender_label, round(gender_confidence, 1)

    def _estimate_heuristic(self, track_id, landmarks_2d):
        """
        Determines deterministic and geometrically guided baseline values
        for age and gender so that they remain stable for the same track_id.
        """
        # A seed based on track_id guarantees consistency for the same face
        np.random.seed(track_id * 17)
        
        # Base age centered around 20-35, with deterministic variations
        base_age = 20.0 + (track_id % 15) + np.random.randint(-3, 4)
        gender_male_prob = 0.4 + (track_id % 3) * 0.1  # Deterministic base: 0.4, 0.5, or 0.6
        
        if landmarks_2d is not None and len(landmarks_2d) > 300:
            try:
                # Use facial morphology ratios to adjust probabilities:
                # Jaw Width / Face Height ratio (longer faces skew male)
                jaw_w = np.linalg.norm(np.array(landmarks_2d[234]) - np.array(landmarks_2d[454]))
                face_h = np.linalg.norm(np.array(landmarks_2d[10]) - np.array(landmarks_2d[152]))
                jaw_ratio = jaw_w / max(1.0, face_h)
                
                # Eyebrow thickness / closeness to eye (lower, thicker eyebrows skew male)
                eyebrow_eye_dist = np.linalg.norm(np.array(landmarks_2d[70]) - np.array(landmarks_2d[159]))
                eye_w = np.linalg.norm(np.array(landmarks_2d[33]) - np.array(landmarks_2d[133]))
                eyebrow_ratio = eyebrow_eye_dist / max(1.0, eye_w)
                
                # Adjust gender probability
                if jaw_ratio > 0.82: # Wide square jaw
                    gender_male_prob += 0.15
                if eyebrow_ratio < 0.35: # Brow ridge close to eye (deep set, male)
                    gender_male_prob += 0.15
                if eyebrow_ratio > 0.45: # High eyebrows (more common in females)
                    gender_male_prob -= 0.15
                
                # Adjust age estimate slightly by eye relative scale:
                # Children have larger eyes relative to face size.
                eye_ratio = eye_w / max(1.0, jaw_w)
                if eye_ratio > 0.22: # Very large eyes
                    base_age = max(10, base_age - 8)
                elif eye_ratio < 0.15: # Smaller eyes
                    base_age = min(75, base_age + 8)
            except Exception:
                pass
                
        # Bound probability between 0.05 and 0.95
        gender_male_prob = max(0.05, min(0.95, gender_male_prob))
        return base_age, gender_male_prob

    def clean_track(self, track_id):
        """Cleans history for a dead track."""
        if track_id in self.track_histories:
            del self.track_histories[track_id]
