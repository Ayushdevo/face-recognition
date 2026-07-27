import cv2
import mediapipe as np
import numpy as np

class FaceDetector:
    """
    Integrates MediaPipe to perform real-time face detection,
    bounding box extraction, and landmark detection.
    """
    def __init__(self, max_num_faces=20, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Use modern mediapipe initializers or fallback safely
        try:
            from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
            from mediapipe.python.solutions import hands as mp_hands_module
        except ImportError:
            import mediapipe.solutions.face_mesh as mp_face_mesh_module
            import mediapipe.solutions.hands as mp_hands_module

        self.face_mesh = mp_face_mesh_module.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.hands = mp_hands_module.Hands(
            max_num_hands=4,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_faces(self, frame):
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        detections = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_2d = []
                landmarks_3d = []
                for lm in face_landmarks.landmark:
                    x_px = min(img_w - 1, max(0, int(lm.x * img_w)))
                    y_px = min(img_h - 1, max(0, int(lm.y * img_h)))
                    landmarks_2d.append((x_px, y_px))
                    landmarks_3d.append((lm.x, lm.y, lm.z))
                
                landmarks_2d = np.array(landmarks_2d)
                min_x, max_x = np.min(landmarks_2d[:, 0]), np.max(landmarks_2d[:, 0])
                min_y, max_y = np.min(landmarks_2d[:, 1]), np.max(landmarks_2d[:, 1])
                
                box_w, box_h = max_x - min_x, max_y - min_y
                pad_x = int(box_w * 0.08)
                pad_y_top = int(box_h * 0.18)
                pad_y_bottom = int(box_h * 0.08)
                
                x1 = max(0, min_x - pad_x)
                y1 = max(0, min_y - pad_y_top)
                x2 = min(img_w - 1, max_x + pad_x)
                y2 = min(img_h - 1, max_y + pad_y_bottom)
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "landmarks_2d": landmarks_2d,
                    "landmarks_3d": landmarks_3d,
                    "confidence": 0.98
                })
        return detections

    def detect_hands(self, frame):
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks_2d = []
                landmarks_3d = []
                for lm in hand_landmarks.landmark:
                    landmarks_2d.append((int(lm.x * img_w), int(lm.y * img_h)))
                    landmarks_3d.append((lm.x, lm.y, lm.z))
                
                hands_data.append({
                    "bbox": [0, 0, img_w, img_h],
                    "landmarks_2d": np.array(landmarks_2d),
                    "landmarks_3d": np.array(landmarks_3d),
                    "label": handedness.classification[0].label,
                    "score": float(handedness.classification[0].score)
                })
        return hands_data

    def close(self):
        self.face_mesh.close()
        self.hands.close()
