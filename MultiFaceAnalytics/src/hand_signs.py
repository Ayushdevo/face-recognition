import os
import cv2
import numpy as np
from src.utils import detect_device, download_file

class HandSignRecognizer:
    """
    Classifies 3D hand landmarks into one of 5 hand signs:
    Fist, Open Palm, Victory, Thumbs Up, Thumbs Down.
    Supports ONNX model inference with fallback to geometric heuristics.
    """
    def __init__(self, models_dir="./models"):
        self.models_dir = os.path.abspath(models_dir)
        self.model_path = os.path.join(self.models_dir, "hand_sign_model.onnx")
        self.classes = ["Fist", "Open Palm", "Victory", "Thumbs Up", "Thumbs Down"]
        
        self.use_onnx = False
        self.ort_session = None
        self.net = None
        
        # Try loading the model
        self._load_model()

    def _load_model(self):
        """
        Attempts to load the ONNX model.
        """
        if not os.path.exists(self.model_path):
            print(f"[WARNING] Hand sign model not found at {self.model_path}. Using geometric heuristic fallback.")
            return

        # Try loading with ONNX Runtime
        try:
            import onnxruntime as ort
            gpu_available, backend = detect_device()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu_available else ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
            self.use_onnx = True
            print(f"Loaded Hand Sign ONNX model successfully with ONNX Runtime on {backend}.")
            return
        except ImportError:
            # Fall back to OpenCV DNN
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                gpu_available, _ = detect_device()
                if gpu_available:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.use_onnx = True
                print("Loaded Hand Sign ONNX model successfully with OpenCV DNN.")
                return
            except Exception as e:
                print(f"[WARNING] Failed to load hand sign model via OpenCV: {e}")
                
        print("[WARNING] Running Hand Sign Recognition in Heuristic Fallback mode.")

    def predict_gesture(self, landmarks_3d):
        """
        Predicts hand sign from 3D landmarks.
        Args:
            landmarks_3d: np.ndarray of shape (21, 3) representing raw 3D coordinates.
        Returns:
            dict: { 'gesture': str, 'confidence': float }
        """
        if landmarks_3d is None or len(landmarks_3d) < 21:
            return {"gesture": "Unknown", "confidence": 0.0}
            
        if self.use_onnx:
            try:
                # 1. Center landmarks around wrist (landmark 0)
                landmarks_rel = landmarks_3d - landmarks_3d[0]
                
                # 2. Scale-normalize by max distance
                max_dist = np.max(np.linalg.norm(landmarks_rel, axis=1))
                if max_dist > 0:
                    landmarks_norm = landmarks_rel / max_dist
                else:
                    landmarks_norm = landmarks_rel
                    
                flat_input = landmarks_norm.flatten().reshape(1, 63).astype(np.float32)
                
                if self.ort_session:
                    input_name = self.ort_session.get_inputs()[0].name
                    outputs = self.ort_session.run(None, {input_name: flat_input})
                    logits = outputs[0][0]
                else:
                    # OpenCV DNN
                    self.net.setInput(flat_input)
                    logits = self.net.forward()[0]
                    
                # Stable Softmax
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                
                max_idx = np.argmax(probs)
                
                return {
                    "gesture": self.classes[max_idx],
                    "confidence": float(probs[max_idx])
                }
            except Exception as e:
                # Fallback on inference error
                pass
                
        # Heuristic fallback
        gesture, conf = self._predict_heuristic(landmarks_3d)
        return {"gesture": gesture, "confidence": conf}

    def _predict_heuristic(self, landmarks_3d):
        """
        Geometric rule-based gesture classification from 3D coordinates.
        """
        try:
            # 0: Wrist, 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky
            # Note: in MediaPipe, smaller Y value is higher in the frame image.
            
            # Check extensions for 4 main fingers
            index_ext = landmarks_3d[8][1] < landmarks_3d[6][1]
            middle_ext = landmarks_3d[12][1] < landmarks_3d[10][1]
            ring_ext = landmarks_3d[16][1] < landmarks_3d[14][1]
            pinky_ext = landmarks_3d[20][1] < landmarks_3d[18][1]
            
            # Thumb extension checks
            # Thumb MCP (2) and TIP (4). If pointing up, Y coordinate of tip is smaller than MCP
            thumb_is_up = landmarks_3d[4][1] < landmarks_3d[2][1] - 0.05
            thumb_is_down = landmarks_3d[4][1] > landmarks_3d[2][1] + 0.05
            
            # General count of extended main fingers
            ext_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
            
            # Thumbs Up condition
            if thumb_is_up and ext_count == 0:
                return "Thumbs Up", 0.88
                
            # Thumbs Down condition
            if thumb_is_down and ext_count == 0:
                return "Thumbs Down", 0.88
                
            # Victory (Peace Sign): index & middle extended, ring & pinky folded
            if index_ext and middle_ext and not ring_ext and not pinky_ext:
                return "Victory", 0.92
                
            # Open Palm: all four main fingers extended
            if ext_count >= 3:
                return "Open Palm", 0.95
                
            # Fist: no main fingers extended, and thumb is not extended up/down
            if ext_count == 0:
                return "Fist", 0.94
                
            # Default fallback mapping
            if ext_count >= 2:
                return "Open Palm", 0.60
            else:
                return "Fist", 0.60
                
        except Exception:
            return "Fist", 0.50
