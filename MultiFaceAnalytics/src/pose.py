import cv2
import numpy as np
import math

class HeadPoseEstimator:
    """
    Estimates Head Pose (Yaw, Pitch, Roll) using 2D face landmarks and 3D facial model.
    Uses Perspective-n-Point (solvePnP) algorithm.
    """
    def __init__(self):
        # 3D model points of standard face (in mm or arbitrary consistent coordinate system)
        # Coordinates correspond to Nose tip, Chin, Left Eye corner, Right Eye corner, Left Mouth corner, Right Mouth corner
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (index 1)
            (0.0, -330.0, -65.0),        # Chin (index 152)
            (-225.0, 170.0, -135.0),     # Left eye outer corner (index 33)
            (225.0, 170.0, -135.0),      # Right eye outer corner (index 263)
            (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
            (150.0, -150.0, -125.0)      # Right mouth corner (index 291)
        ], dtype=np.float32)

        # Landmark indexes in MediaPipe Face Mesh matching the 3D model points
        self.landmark_indices = [1, 152, 33, 263, 61, 291]

    def estimate_pose(self, landmarks_2d, img_w, img_h):
        """
        Estimates the head pose angles in degrees.
        Args:
            landmarks_2d: np.ndarray of shape (N, 2) containing pixel coordinates.
            img_w: Width of the image frame.
            img_h: Height of the image frame.
        Returns:
            yaw, pitch, roll (in degrees)
        """
        # Extract the key points
        image_points = np.array([landmarks_2d[idx] for idx in self.landmark_indices], dtype=np.float32)

        # Approximate camera calibration parameters (focal length and optical center)
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # Solve for translation and rotation vectors using SolvePnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Combine rotation matrix and translation vector to build projection matrix
        projection_matrix = np.hstack((rotation_matrix, translation_vector))
        
        # Decompose projection matrix to extract Euler angles
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        
        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        # Adjust the euler angles mapping to output natural numbers:
        # Looking straight ahead should yield close to 0, 0, 0.
        # Yaw: turn left/right (-90 to +90)
        # Pitch: nod up/down (-90 to +90)
        # Roll: head tilt (-90 to +90)
        
        # Depending on camera vs world frame conventions, adjust output signs:
        pitch = -pitch
        if pitch > 180:
            pitch -= 360
        elif pitch < -180:
            pitch += 360

        # Compensate for decompose projection rotation shifts
        if yaw > 180:
            yaw -= 360
        elif yaw < -180:
            yaw += 360
            
        if roll > 180:
            roll -= 360
        elif roll < -180:
            roll += 360

        # Constrain to plausible values and round to integer
        yaw = round(yaw, 1)
        pitch = round(pitch, 1)
        roll = round(roll, 1)

        return yaw, pitch, roll
