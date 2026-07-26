import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    """
    State tracking object representing a single face track.
    Uses a custom linear velocity Kalman Filter to estimate box positions.
    """
    def __init__(self, bbox, track_id):
        self.id = track_id
        # State vector: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]^T
        self.x = np.zeros((8, 1))
        self.x[:4, 0] = bbox
        
        # State transition matrix F
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0 # position += velocity * dt (dt=1)
            
        # Measurement matrix H (we only observe positions)
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0
            
        # Covariance matrices
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0 # higher initial velocity uncertainty
        
        self.Q = np.eye(8) * 0.05
        self.Q[4:, 4:] *= 0.5
        
        self.R = np.eye(4) * 1.0
        
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.last_bbox = bbox
        self.history = []

    def predict(self):
        """
        Advances the tracker state using the motion model.
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        self.time_since_update += 1
        self.last_bbox = self.x[:4, 0].tolist()
        return self.last_bbox

    def update(self, bbox):
        """
        Updates the state with a new measurement box.
        """
        self.time_since_update = 0
        self.hits += 1
        
        # Measurement update
        z = np.array(bbox).reshape(4, 1)
        y = z - np.dot(self.H, self.x) # innovation
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R # innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman gain
        
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        self.last_bbox = self.x[:4, 0].tolist()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x[:4, 0].tolist()

def calculate_iou(boxA, boxB):
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if float(boxAArea + boxBArea - interArea) <= 0.0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def calculate_centroid_distance(boxA, boxB):
    """
    Calculates normalized Euclidean distance between the centroids of two boxes.
    Normalized by the average scale of the boxes to handle scale invariance.
    """
    cA = np.array([(boxA[0] + boxA[2])/2.0, (boxA[1] + boxA[3])/2.0])
    cB = np.array([(boxB[0] + boxB[2])/2.0, (boxB[1] + boxB[3])/2.0])
    
    wA, hA = boxA[2] - boxA[0], boxA[3] - boxA[1]
    wB, hB = boxB[2] - boxB[0], boxB[3] - boxB[1]
    
    scale = (wA + hA + wB + hB) / 4.0
    if scale <= 0.0:
        return 999.0
        
    dist = np.linalg.norm(cA - cB)
    return dist / scale

class FaceTracker:
    """
    Manages multiple KalmanBoxTrackers and associates them with new detections
    using the Hungarian algorithm with a combined IoU + Centroid Distance cost.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.2, centroid_threshold=1.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.centroid_threshold = centroid_threshold
        self.trackers = []
        self.frame_count = 0
        self.id_counter = 1

    def update(self, detections):
        """
        Detections: list of dicts from FaceDetector, containing 'bbox' and 'landmarks_2d'.
        Returns:
            list of dicts matching detections with persistent track IDs.
        """
        self.frame_count += 1
        
        # 1. Predict new positions for all active trackers
        for t in self.trackers:
            t.predict()
            
        # 2. Extract bounding boxes from detections
        det_boxes = [d['bbox'] for d in detections]
        
        # 3. Associate detections with trackers
        matched, unmatched_dets, unmatched_trackers = self._associate(det_boxes)
        
        # 4. Update matched trackers with new detections
        for m in matched:
            self.trackers[m[1]].update(det_boxes[m[0]])
            
        # 5. Create new trackers for unmatched detections
        for i in unmatched_dets:
            new_tracker = KalmanBoxTracker(det_boxes[i], self.id_counter)
            self.id_counter += 1
            self.trackers.append(new_tracker)
            
        # 6. Clean up or transition stale trackers
        ret = []
        i = len(self.trackers)
        for t in reversed(self.trackers):
            i -= 1
            state = t.get_state()
            
            # Constraints: box must be positive width/height
            if (state[2] - state[0] > 0) and (state[3] - state[1] > 0):
                # Filter tracks to return only active and stable tracks
                # Stable tracks are those seen enough times or recently updated
                if (t.time_since_update < 1) and (t.hits >= self.min_hits or self.frame_count <= self.min_hits):
                    # Find which detection this tracker matched to copy landmarks and metadata
                    matching_det = None
                    for m in matched:
                        if m[1] == i:
                            matching_det = detections[m[0]]
                            break
                    
                    ret.append({
                        "id": t.id,
                        "bbox": state,
                        "landmarks_2d": matching_det["landmarks_2d"] if matching_det else None,
                        "landmarks_3d": matching_det["landmarks_3d"] if matching_det else None,
                        "confidence": matching_det["confidence"] if matching_det else 0.9,
                        "is_lost": False
                    })
            
            # Remove trackers that have expired
            if t.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        return ret

    def _associate(self, detections):
        """
        Associates detections to trackers using Hungarian Algorithm.
        """
        if len(self.trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []

        # Cost matrix: combining IoU and centroid distance
        cost_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou = calculate_iou(det, trk.get_state())
                centroid_dist = calculate_centroid_distance(det, trk.get_state())
                
                # High IoU means low cost. High Centroid distance means high cost.
                # Combined cost formula:
                iou_cost = 1.0 - iou
                
                # If IoU is completely zero and centroid distance is huge, gate it
                if iou < self.iou_threshold and centroid_dist > self.centroid_threshold:
                    cost_matrix[d, t] = 99999.0
                else:
                    cost_matrix[d, t] = iou_cost * 0.7 + (centroid_dist / self.centroid_threshold) * 0.3

        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_indices = []
        for r, c in zip(row_ind, col_ind):
            # Check cost threshold
            if cost_matrix[r, c] < 999.0:
                matched_indices.append([r, c])
                
        # Find unmatched detections
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in [m[0] for m in matched_indices]:
                unmatched_detections.append(d)
                
        # Find unmatched trackers
        unmatched_trackers = []
        for t in range(len(self.trackers)):
            if t not in [m[1] for m in matched_indices]:
                unmatched_trackers.append(t)
                
        return np.array(matched_indices), unmatched_detections, unmatched_trackers
