import os
import csv
import queue
import threading
from datetime import datetime

class FaceAnalyticsLogger:
    """
    Thread-safe logger that records face analytics data into a CSV file.
    Uses a background worker thread and a thread-safe Queue to avoid slowing down 
    the real-time video processing loop.
    """
    def __init__(self, log_dir="./logs", filename="face_analytics.csv"):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, filename)
        
        self.queue = queue.Queue()
        self.running = True
        
        # Write CSV Header if file does not exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Face_ID", "Emotion", "Age", "Gender", "Detection_Confidence", "Quality_Score"
                ])
                
        # Start background writer thread
        self.worker = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker.start()

    def log_face(self, face_id, emotion, age, gender, confidence, quality_score):
        """
        Pushes a face analytics log entry to the write queue.
        This call is non-blocking.
        """
        if not self.running:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = [
            timestamp,
            face_id,
            emotion,
            age,
            gender,
            f"{confidence:.4f}",
            f"{quality_score:.1f}"
        ]
        self.queue.put(log_entry)

    def _writer_loop(self):
        """
        Worker thread loop that pulls entries from the queue and writes them to the CSV.
        """
        while self.running or not self.queue.empty():
            try:
                # Wait for entry (timeout to allow checking running status)
                entry = self.queue.get(timeout=0.5)
                
                # Append to CSV
                with open(self.log_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(entry)
                    
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Failed to write log entry: {e}")

    def close(self):
        """
        Signals the worker thread to finish writing remaining items and shut down.
        """
        self.running = False
        self.worker.join(timeout=2.0)
        print(f"Logger closed. CSV logs saved to: {self.log_path}")
