import cv2
import numpy as np

class FaceQualityCalculator:
    """
    Computes a Face Quality Score (0 to 100) based on multiple factors:
    1. Blur (Laplacian variance)
    2. Brightness (mean intensity deviation from optimal)
    3. Contrast (standard deviation of intensity)
    4. Head Pose (penalizes extreme yaw/pitch/roll)
    5. Face Size (relative resolution compared to frame)
    6. Occlusion / Symmetry (symmetry ratio of left/right face halves)
    """
    def __init__(self):
        # Weight coefficients for the final composite score
        self.w_blur = 0.25
        self.w_brightness = 0.15
        self.w_contrast = 0.15
        self.w_pose = 0.20
        self.w_size = 0.15
        self.w_symmetry = 0.10

    def calculate_quality(self, face_patch, bbox, landmarks_2d, yaw, pitch, roll, frame_w, frame_h):
        """
        Calculates a score between 0.0 and 100.0 representing facial image quality.
        Args:
            face_patch: Crop of the face in BGR.
            bbox: Bounding box [x1, y1, x2, y2].
            landmarks_2d: list or array of shape (N, 2) of 2D pixel coordinates.
            yaw, pitch, roll: Est. pose angles.
            frame_w, frame_h: Frame size.
        Returns:
            float: Face Quality Score (0-100).
        """
        if face_patch is None or face_patch.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_patch, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur score
        # Laplacian variance: low variance means blurry/out of focus.
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Scale: standard clean face usually has lap_var > 100.
        blur_score = min(100.0, max(0.0, (lap_var / 120.0) * 100.0))
        
        # 2. Brightness score
        # Ideal mean gray value is around 120. Very dark or bright gets penalized.
        mean_brightness = np.mean(gray)
        brightness_dev = abs(mean_brightness - 120.0)
        brightness_score = max(0.0, 100.0 - (brightness_dev / 120.0) * 100.0)
        
        # 3. Contrast score
        # Standard deviation of pixel values. High SD is good contrast, low SD is washed out.
        std_contrast = np.std(gray)
        # Normal scale: standard deviation > 50 is excellent contrast.
        contrast_score = min(100.0, max(0.0, (std_contrast / 55.0) * 100.0))

        # 4. Pose score
        # Face looking straight at the camera (yaw, pitch, roll near 0) gets highest score.
        max_allowable_angle = 45.0 # Angle above which score becomes 0
        avg_pose_dev = (abs(yaw) + abs(pitch) + abs(roll)) / 3.0
        pose_score = max(0.0, 100.0 - (avg_pose_dev / max_allowable_angle) * 100.0)
        
        # 5. Face Size score
        # Resolution matters. Compare face box area to frame area.
        # A face that is at least 15% of frame width/height gets a perfect size score.
        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1
        face_area = face_w * face_h
        frame_area = frame_w * frame_h
        size_ratio = face_area / frame_area
        # Normalize: target size ratio of 0.05 (e.g. 224x224 in 1080p is ~0.025)
        size_score = min(100.0, max(0.0, (size_ratio / 0.04) * 100.0))

        # 6. Occlusion / Symmetry score
        # We can analyze the horizontal symmetry of facial landmarks.
        # Specifically, distance from nose tip (idx 1) to left eye corner (idx 33)
        # and right eye corner (idx 263).
        symmetry_score = 100.0
        if landmarks_2d is not None and len(landmarks_2d) > 263:
            nose = np.array(landmarks_2d[1])
            left_eye = np.array(landmarks_2d[33])
            right_eye = np.array(landmarks_2d[263])
            
            d_left = np.linalg.norm(nose - left_eye)
            d_right = np.linalg.norm(nose - right_eye)
            
            if d_left > 0 and d_right > 0:
                ratio = min(d_left, d_right) / max(d_left, d_right)
                symmetry_score = ratio * 100.0
        
        # Calculate composite weighted score
        quality_score = (
            self.w_blur * blur_score +
            self.w_brightness * brightness_score +
            self.w_contrast * contrast_score +
            self.w_pose * pose_score +
            self.w_size * size_score +
            self.w_symmetry * symmetry_score
        )
        
        return round(float(quality_score), 1)
