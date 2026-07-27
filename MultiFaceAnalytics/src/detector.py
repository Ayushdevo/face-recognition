import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    """
    Integrates MediaPipe Face Mesh to perform real-time face detection,
    bounding box extraction, and 3D/2D landmark detection.
    """
    def __init__(self, max_num_faces=20, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        try:
            # Standard legacy solutions access
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_hands = mp.solutions.hands
        except AttributeError:
            # Fallback for modern MediaPipe builds where solutions are structured differently or omitted
            try:
                import mediapipe.python.solutions as mp_sol  # type: ignore
                self.mp_face_mesh = mp_sol.face_mesh
                self.mp_hands = mp_sol.hands
            except Exception:
                raise ImportError(
                    "Your installed version of MediaPipe does not support legacy 'solutions'. "
                    "Please pin mediapipe version to 0.10.14 or lower in requirements.txt."
                )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # Includes detailed eye/iris coordinates
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.hands = self.mp_hands.Hands(
            max_num_hands=4, # support up to 4 hands simultaneously
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_faces(self, frame):
        """
        Runs face detection on the frame.
        Args:
            frame: np.ndarray in BGR format.
        Returns:
            list of dicts containing face info:
                - 'bbox': [x1, y1, x2, y2]
                - 'landmarks_2d': list of (x, y) coordinates
                - 'landmarks_3d': list of (x, y, z) raw coordinates
                - 'confidence': float (detection confidence)
        """
        img_h, img_w = frame.shape[:2]
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.face_mesh.process(rgb_frame)
        
        detections = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_2d = []
                landmarks_3d = []
                
                # Convert normalized coordinates to pixel coordinates
                for lm in face_landmarks.landmark:
                    x_px = min(img_w - 1, max(0, int(lm.x * img_w)))
                    y_px = min(img_h - 1, max(0, int(lm.y * img_h)))
                    landmarks_2d.append((x_px, y_px))
                    landmarks_3d.append((lm.x, lm.y, lm.z))
                
                landmarks_2d = np.array(landmarks_2d)
                
                # Compute bounding box from min/max coordinates
                min_x = np.min(landmarks_2d[:, 0])
                max_x = np.max(landmarks_2d[:, 0])
                min_y = np.min(landmarks_2d[:, 1])
                max_y = np.max(landmarks_2d[:, 1])
                
                # Add padding to make the box fit standard head dimensions nicely
                box_w = max_x - min_x
                box_h = max_y - min_y
                
                # Apply padding factors
                pad_x = int(box_w * 0.08)
                pad_y_top = int(box_h * 0.18) # more on top for forehead
                pad_y_bottom = int(box_h * 0.08)
                
                x1 = max(0, min_x - pad_x)
                y1 = max(0, min_y - pad_y_top)
                x2 = min(img_w - 1, max_x + pad_x)
                y2 = min(img_h - 1, max_y + pad_y_bottom)
                
                bbox = [x1, y1, x2, y2]
                
                # Compute a quality/confidence heuristic for detection
                scale_factor = (box_w * box_h) / (img_w * img_h)
                edge_penalty = 0.0
                # Penalize faces very close to boundaries (since landmarks might crop)
                if x1 <= 5 or y1 <= 5 or x2 >= img_w - 6 or y2 >= img_h - 6:
                    edge_penalty = 0.05
                    
                confidence = max(0.60, min(0.999, 0.985 + (scale_factor * 0.01) - edge_penalty))
                
                detections.append({
                    "bbox": bbox,
                    "landmarks_2d": landmarks_2d,
                    "landmarks_3d": landmarks_3d,
                    "confidence": float(confidence)
                })
                
        return detections

    def detect_hands(self, frame):
        """
        Runs hand detection on the frame.
        Args:
            frame: np.ndarray in BGR format.
        Returns:
            list of dicts containing hand info:
                - 'bbox': [x1, y1, x2, y2]
                - 'landmarks_2d': np.ndarray of shape (21, 2)
                - 'landmarks_3d': np.ndarray of shape (21, 3)
                - 'label': str ('Left' or 'Right')
                - 'score': float (handedness confidence score)
        """
        img_h, img_w = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run hand inference
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks_2d = []
                landmarks_3d = []
                
                # Convert normalized coordinates to pixel coordinates
                for lm in hand_landmarks.landmark:
                    x_px = min(img_w - 1, max(0, int(lm.x * img_w)))
                    y_px = min(img_h - 1, max(0, int(lm.y * img_h)))
                    landmarks_2d.append((x_px, y_px))
                    landmarks_3d.append((lm.x, lm.y, lm.z))
                
                landmarks_2d = np.array(landmarks_2d)
                landmarks_3d = np.array(landmarks_3d)
                
                # Compute bounding box
                min_x = np.min(landmarks_2d[:, 0])
                max_x = np.max(landmarks_2d[:, 0])
                min_y = np.min(landmarks_2d[:, 1])
                max_y = np.max(landmarks_2d[:, 1])
                
                # Pad bounding box slightly
                w = max_x - min_x
                h = max_y - min_y
                pad = int(max(w, h) * 0.15)
                
                bbox = [
                    max(0, min_x - pad),
                    max(0, min_y - pad),
                    min(img_w - 1, max_x + pad),
                    min(img_h - 1, max_y + pad)
                ]
                
                # Get hand chirality/label (swapped to correct for mirrored webcam view)
                raw_label = handedness.classification[0].label # "Left" or "Right"
                hand_label = "Left" if raw_label == "Right" else "Right"
                score = handedness.classification[0].score
                
                hands_data.append({
                    "bbox": bbox,
                    "landmarks_2d": landmarks_2d,
                    "landmarks_3d": landmarks_3d,
                    "label": hand_label,
                    "score": float(score)
                })
                
        return hands_data

    def close(self):
        """Release resources."""
        self.face_mesh.close()
        self.hands.close()
