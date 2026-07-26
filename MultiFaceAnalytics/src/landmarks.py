import cv2
import numpy as np

class LandmarkRenderer:
    """
    Renders facial landmarks in a high-tech, aesthetically pleasing manner.
    Uses alpha-blending for translucent meshes and bright contours.
    """
    def __init__(self):
        # We define subsets of indices for key contours in MediaPipe Face Mesh
        self.LIPS = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 
            14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]
        self.LEFT_EYE = [
            263, 249, 390, 373, 374, 380, 381, 382, 362, 463, 341, 256, 252, 253, 254, 339, 263
        ]
        self.RIGHT_EYE = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 243, 112, 26, 22, 23, 24, 110, 33
        ]
        self.LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
        self.RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
        self.FACE_OVAL = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 
            400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 
            54, 103, 67, 109
        ]

    def draw_landmarks(self, img, landmarks_2d, draw_mesh=True, draw_contours=True, mesh_color=(0, 255, 255), contour_color=(255, 255, 0)):
        """
        Draws landmarks on the image.
        Args:
            img: Image frame to draw on.
            landmarks_2d: np.ndarray or list of (x, y) coordinates.
            draw_mesh: If True, draws the full dense landmark points.
            draw_contours: If True, draws the outlines of eyes, lips, and face.
            mesh_color: BGR color for the dense dots.
            contour_color: BGR color for the outlines.
        """
        if landmarks_2d is None or len(landmarks_2d) == 0:
            return

        # Prepare alpha blending overlays
        overlay = img.copy()
        h, w = img.shape[:2]
        
        # 1. Draw full mesh points as tiny dots (translucent)
        if draw_mesh:
            for pt in landmarks_2d:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(overlay, (x, y), 1, mesh_color, -1)
            # Blend with 0.4 opacity
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        # 2. Draw face contour lines (with higher opacity)
        if draw_contours:
            overlay_contours = img.copy()
            # Helper to draw a closed or open loop
            def draw_loop(indices, closed=True):
                pts = [np.array(landmarks_2d[idx], dtype=np.int32) for idx in indices]
                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay_contours, [pts], closed, contour_color, 1, cv2.LINE_AA)

            draw_loop(self.LIPS, closed=True)
            draw_loop(self.LEFT_EYE, closed=True)
            draw_loop(self.RIGHT_EYE, closed=True)
            draw_loop(self.LEFT_EYEBROW, closed=False)
            draw_loop(self.RIGHT_EYEBROW, closed=False)
            draw_loop(self.FACE_OVAL, closed=False)

            # Blend contours with 0.6 opacity
            cv2.addWeighted(overlay_contours, 0.6, img, 0.4, 0, img)
            
            # Highlight key points (eyes centers, nose tip)
            key_indices = [1, 33, 263, 61, 291] # Nose, left eye, right eye, mouth corners
            for idx in key_indices:
                x, y = int(landmarks_2d[idx][0]), int(landmarks_2d[idx][1])
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)

    def draw_hand_landmarks(self, img, hand_landmarks_2d, mesh_color=(0, 240, 255), joint_color=(235, 180, 50)):
        """
        Draws hand landmarks and skeleton connections on the image.
        """
        if hand_landmarks_2d is None or len(hand_landmarks_2d) == 0:
            return

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        overlay = img.copy()
        
        # 1. Draw connections
        for start, end in connections:
            pt1 = tuple(map(int, hand_landmarks_2d[start]))
            pt2 = tuple(map(int, hand_landmarks_2d[end]))
            cv2.line(overlay, pt1, pt2, mesh_color, 2, cv2.LINE_AA)
            
        # 2. Draw joints
        for pt in hand_landmarks_2d:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (x, y), 4, joint_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
            
        # Blend with 0.7 opacity
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
