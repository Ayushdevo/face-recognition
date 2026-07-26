import os
import sys
import time
import cv2
import numpy as np
import threading
from datetime import datetime

# Add the parent directory of 'src' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import modular components
from src.detector import FaceDetector
from src.tracker import FaceTracker
from src.pose import HeadPoseEstimator
from src.quality import FaceQualityCalculator
from src.emotion import EmotionRecognizer
from src.age_gender import AgeGenderEstimator
from src.hand_signs import HandSignRecognizer
from src.landmarks import LandmarkRenderer
from src.logger import FaceAnalyticsLogger
from src.utils import detect_device, draw_glass_panel, draw_styled_bbox, get_unique_color

class WebcamStream:
    """
    Multithreaded webcam capture stream to maximize FPS.
    Continuously reads frames from the video device in a background thread.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.started = False
                break
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            # Tiny sleep to yield context
            time.sleep(0.005)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.stream.release()

class MultiFaceAnalyticsApp:
    """
    Main application engine coordinating the real-time detection,
    tracking, and multi-analytics pipeline.
    """
    def __init__(self, camera_idx=0):
        self.camera_idx = camera_idx
        
        # Initialize sub-modules
        self.detector = FaceDetector(max_num_faces=20, min_detection_confidence=0.55)
        self.tracker = FaceTracker(max_age=30, min_hits=2)
        self.pose_estimator = HeadPoseEstimator()
        self.quality_calculator = FaceQualityCalculator()
        
        # Locate the models directory relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.abspath(os.path.join(script_dir, "../models"))
        if not os.path.exists(models_dir):
            models_dir = os.path.abspath(os.path.join(script_dir, "../../models"))
            
        self.emotion_recognizer = EmotionRecognizer(models_dir=models_dir)
        self.age_gender_estimator = AgeGenderEstimator(models_dir=models_dir)
        self.hand_recognizer = HandSignRecognizer(models_dir=models_dir)
        self.landmark_renderer = LandmarkRenderer()
        self.logger = FaceAnalyticsLogger()

        # Keyboard Toggles / App State
        self.show_landmarks = True
        self.show_fps = True
        self.show_confidence = True
        
        # Directories
        self.screenshots_dir = "./screenshots"
        self.recordings_dir = "./recordings"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.recordings_dir, exist_ok=True)

        # Recording State
        self.is_recording = False
        self.video_writer = None
        
        # UI overlays
        self.notification_text = ""
        self.notification_expiry = 0.0
        
        # Statistics
        self.all_seen_ids = set()
        self.fps = 0.0
        self.inference_time = 0.0

    def trigger_notification(self, text, duration=1.5):
        """Displays a message toast on the frame for a set duration."""
        self.notification_text = text
        self.notification_expiry = time.time() + duration

    def run(self):
        # Start camera thread
        print(f"Initializing webcam capture index {self.camera_idx}...")
        webcam = WebcamStream(src=self.camera_idx)
        
        # Probe connection
        test_grabbed, test_frame = webcam.stream.read()
        if not test_grabbed or test_frame is None:
            print(f"\n[CRITICAL ERROR] Camera device {self.camera_idx} is unavailable or in use.")
            print("Please ensure your webcam is connected, drivers are active, and no other app is using it.\n")
            webcam.stop()
            return
            
        webcam.start()
        
        # Prepare window
        window_name = "Real-Time Multi-Face Analytics HUD"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_count = 0
        fps_timer = time.time()
        
        # Hardware backend diagnostics
        _, hardware_backend = detect_device()
        print(f"Operational hardware acceleration backend: {hardware_backend}")
        print("HUD running. Press 'Q' inside window to quit.")

        try:
            while webcam.started:
                start_cycle = time.time()
                grabbed, frame = webcam.read()
                
                if not grabbed or frame is None:
                    print("[WARNING] Empty frame read from camera. Retrying...")
                    time.sleep(0.01)
                    continue

                img_h, img_w = frame.shape[:2]
                
                # --- Pipeline Stage 1: Face Detection ---
                det_start = time.time()
                detections = self.detector.detect_faces(frame)
                self.inference_time = (time.time() - det_start) * 1000.0 # in ms
                
                # --- Pipeline Stage 2: Face Tracking ---
                tracked_faces = self.tracker.update(detections)
                
                # Variables to accumulate frame stats
                frame_qualities = []
                frame_emotion_confs = []
                
                # --- Pipeline Stage 3: Face Analytics ---
                for face in tracked_faces:
                    track_id = face["id"]
                    self.all_seen_ids.add(track_id)
                    bbox = face["bbox"]
                    landmarks_2d = face["landmarks_2d"]
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    w, h = x2 - x1, y2 - y1
                    
                    # Prevent zero-sized bounding box issues
                    if w <= 0 or h <= 0:
                        continue
                        
                    # Face crop patch (for deep learning models)
                    face_patch = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    
                    # 3.1 Head Pose Estimation
                    yaw, pitch, roll = 0.0, 0.0, 0.0
                    if landmarks_2d is not None:
                        yaw, pitch, roll = self.pose_estimator.estimate_pose(landmarks_2d, img_w, img_h)
                    
                    # 3.2 Face Quality Score
                    quality_score = 0.0
                    if face_patch.size > 0:
                        quality_score = self.quality_calculator.calculate_quality(
                            face_patch, bbox, landmarks_2d, yaw, pitch, roll, img_w, img_h
                        )
                        frame_qualities.append(quality_score)
                    
                    # 3.3 Emotion Recognition
                    emotion_res = self.emotion_recognizer.predict_emotion(face_patch, landmarks_2d)
                    emotion_label = emotion_res["emotion"]
                    emotion_conf = emotion_res["confidence"]
                    frame_emotion_confs.append(emotion_conf * 100.0)
                    
                    # 3.4 Age & Gender Estimation
                    age, gender, gender_conf = self.age_gender_estimator.estimate(face_patch, track_id, landmarks_2d)
                    
                    # 3.5 Log records in background thread
                    det_confidence = face["confidence"]
                    self.logger.log_face(track_id, emotion_label, age, gender, det_confidence, quality_score)
                    
                    # --- Rendering overlays ---
                    # Setup label card text lines
                    label_lines = [f"FACE #{track_id}"]
                    label_lines.append(f"Emotion: {emotion_label} ({int(emotion_conf * 100)}%)")
                    label_lines.append(f"Age: {age} Years")
                    label_lines.append(f"Gender: {gender} ({int(gender_conf)}%)")
                    
                    if self.show_confidence:
                        label_lines.append(f"Detection: {det_confidence*100:.1f}%")
                        
                    label_lines.append(f"Quality: {int(quality_score)}%")
                    label_lines.append(f"Pose: Y:{int(yaw)} P:{int(pitch)} R:{int(roll)}")
                    
                    # Draw high-tech bounding box
                    box_color = get_unique_color(track_id)
                    draw_styled_bbox(frame, bbox, label_lines, box_color, thickness=1)
                    
                    # Draw facial landmarks mesh
                    if self.show_landmarks and landmarks_2d is not None:
                        # Draw blue mesh and cyan outer contours
                        self.landmark_renderer.draw_landmarks(
                            frame, landmarks_2d, draw_mesh=True, draw_contours=True,
                            mesh_color=(235, 180, 50), contour_color=(0, 240, 255)
                        )

                # --- Pipeline Stage 4: Hand Signs Detection & Classification ---
                hands = self.detector.detect_hands(frame)
                
                for hand in hands:
                    hand_bbox = hand["bbox"]
                    landmarks_2d = hand["landmarks_2d"]
                    landmarks_3d = hand["landmarks_3d"]
                    hand_label = hand["label"]
                    det_score = hand["score"]
                    
                    # Run hand sign classification
                    gesture_res = self.hand_recognizer.predict_gesture(landmarks_3d)
                    gesture = gesture_res["gesture"]
                    gesture_conf = gesture_res["confidence"]
                    
                    # Styled labels for hand info panel
                    hand_lines = [
                        f"{hand_label.upper()} HAND",
                        f"Sign: {gesture} ({int(gesture_conf * 100)}%)",
                        f"Score: {det_score * 100:.1f}%"
                    ]
                    
                    # Draw a distinct color for left/right hand
                    # Right hand = vibrant green, Left hand = vibrant purple/magenta
                    hand_color = (0, 235, 100) if hand_label == "Right" else (200, 50, 255)
                    
                    # Draw hand bounding box
                    draw_styled_bbox(frame, hand_bbox, hand_lines, hand_color, thickness=1)
                    
                    # Draw hand skeleton overlay
                    if self.show_landmarks and landmarks_2d is not None:
                        self.landmark_renderer.draw_hand_landmarks(frame, landmarks_2d, mesh_color=hand_color)
                
                # --- HUD Panels Drawing ---
                # Draw Global Statistics Panel (Top-Left)
                stats = {
                    "Active Faces": f"{len(tracked_faces)} / 10+",
                    "Total Face count": str(len(self.all_seen_ids)),
                    "Avg Quality": f"{int(np.mean(frame_qualities)) if frame_qualities else 0}%",
                    "Avg Emotion Conf": f"{int(np.mean(frame_emotion_confs)) if frame_emotion_confs else 0}%",
                    "Active Hands": str(len(hands)),
                    "Device": hardware_backend
                }
                
                if self.show_fps:
                    stats["System FPS"] = f"{self.fps:.1f} Hz"
                    stats["Inference Time"] = f"{self.inference_time:.1f} ms"
                    
                draw_glass_panel(frame, 20, 20, 220, 150, "Telemetry Stats", stats)
                
                # Draw Controls Guide Panel (Bottom-Left)
                controls = {
                    "Quit app": "Q",
                    "Take Screenshot": "S",
                    "Start Recording": "R",
                    "Stop Recording": "T",
                    "Toggle Mesh": "L",
                    "Toggle FPS Stats": "F",
                    "Toggle Confidence": "C"
                }
                draw_glass_panel(frame, 20, img_h - 180, 220, 160, "Controls Guide", controls)
                
                # --- Flashing Recording Indicator (Top-Right) ---
                if self.is_recording:
                    # Flash red circle every 1 second
                    if int(time.time() * 2) % 2 == 0:
                        cv2.circle(frame, (img_w - 30, 30), 8, (0, 0, 255), -1, cv2.LINE_AA)
                    else:
                        cv2.circle(frame, (img_w - 30, 30), 8, (100, 100, 100), -1, cv2.LINE_AA)
                    cv2.putText(frame, "REC", (img_w - 75, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.45, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    # Write frame with overlays to video writer
                    if self.video_writer is not None:
                        self.video_writer.write(frame)

                # --- Toast notification overlay ---
                if time.time() < self.notification_expiry:
                    # Draw a nice centered bar at the bottom
                    notif_w = 320
                    notif_h = 30
                    notif_x = (img_w - notif_w) // 2
                    notif_y = img_h - 50
                    # Backdrop
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (notif_x, notif_y), (notif_x + notif_w, notif_y + notif_h), (0, 165, 255), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    # Text
                    text_size = cv2.getTextSize(self.notification_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    tx = notif_x + (notif_w - text_size[0]) // 2
                    ty = notif_y + (notif_h + text_size[1]) // 2
                    cv2.putText(frame, self.notification_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.4, (255, 255, 255), 1, cv2.LINE_AA)

                # --- Render and Keyboard Handling ---
                cv2.imshow(window_name, frame)
                
                # Check frame rate calculations
                frame_count += 1
                if time.time() - fps_timer >= 1.0:
                    self.fps = frame_count / (time.time() - fps_timer)
                    frame_count = 0
                    fps_timer = time.time()
                
                # Poll key presses (wait 1ms)
                key = cv2.waitKey(1) & 0xFF
                
                # Q -> Quit
                if key == ord('q') or key == ord('Q'):
                    self.trigger_notification("Shutting down system...")
                    break
                    
                # S -> Screenshot
                elif key == ord('s') or key == ord('S'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = os.path.join(self.screenshots_dir, f"screenshot_{timestamp}.jpg")
                    cv2.imwrite(filepath, frame)
                    self.trigger_notification(f"Screenshot saved to screenshots/")
                    
                # R -> Start Recording
                elif key == ord('r') or key == ord('R'):
                    if not self.is_recording:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filepath = os.path.join(self.recordings_dir, f"record_{timestamp}.avi")
                        # Define video writer (MJPG codec is widely supported)
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        self.video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (img_w, img_h))
                        self.is_recording = True
                        self.trigger_notification("Video recording started.")
                    else:
                        self.trigger_notification("Recording already active.")
                        
                # T -> Stop Recording
                elif key == ord('t') or key == ord('T'):
                    if self.is_recording:
                        self.is_recording = False
                        if self.video_writer is not None:
                            self.video_writer.release()
                            self.video_writer = None
                        self.trigger_notification("Video recording saved.")
                    else:
                        self.trigger_notification("No recording active.")
                        
                # L -> Toggle Landmarks
                elif key == ord('l') or key == ord('L'):
                    self.show_landmarks = not self.show_landmarks
                    self.trigger_notification(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                    
                # F -> Toggle FPS Display
                elif key == ord('f') or key == ord('F'):
                    self.show_fps = not self.show_fps
                    self.trigger_notification(f"Inference FPS: {'ON' if self.show_fps else 'OFF'}")
                    
                # C -> Toggle Confidence Display
                elif key == ord('c') or key == ord('C'):
                    self.show_confidence = not self.show_confidence
                    self.trigger_notification(f"Confidence values: {'ON' if self.show_confidence else 'OFF'}")

                # Maintain target frame cycle rate matching source input (approx 33ms limit per loop cycle)
                elapsed = time.time() - start_cycle
                sleep_time = max(0.001, 0.033 - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[CRITICAL APPLICATION RUNTIME EXCEPTION]: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up system resources...")
            webcam.stop()
            if self.video_writer is not None:
                self.video_writer.release()
            self.detector.close()
            self.logger.close()
            cv2.destroyAllWindows()
            print("Cleanup completed. Exiting Application.")

if __name__ == "__main__":
    # In production, default camera index is 0, but user can change it
    app = MultiFaceAnalyticsApp(camera_idx=0)
    app.run()
