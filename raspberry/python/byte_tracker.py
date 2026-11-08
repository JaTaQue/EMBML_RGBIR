"""
WBS 3.6: ByteTrack Implementation for Tiny Drone Tracking
Customized with IoU + Centre Distance matching metric for ultra-small objects (<16 px).
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class STrack:
    """Class representing a single tracked drone object."""
    _count = 0

    def __init__(self, tlwh, score):
        # Format: [x1, y1, w, h] -> convert to Kalman state [cx, cy, a, h, v_cx, v_cy, v_a, v_h]
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.track_id = 0
        self.is_activated = False
        self.state = 0  # 0: Tracked, 1: Lost
        self.frame_id = 0
        self.tracklet_len = 0

        # Initialize 8-state Kalman Filter (position + velocity)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # State vector: [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]
        cx = self.tlwh[0] + self.tlwh[2] / 2.0
        cy = self.tlwh[1] + self.tlwh[3] / 2.0
        a = self.tlwh[2] / max(1.0, self.tlwh[3])
        h = self.tlwh[3]

        self.kf.x = np.array([cx, cy, a, h, 0, 0, 0, 0], dtype=np.float32)

        # State transition matrix
        self.kf.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.F[i, i + 4] = 1.0

        # Measurement matrix (maps state to bounding box coordinates)
        self.kf.H = np.eye(4, 8, dtype=np.float32)

        # Covariances
        self.kf.P *= 10.0
        self.kf.R *= 1.0
        self.kf.Q *= 0.01

    @property
    def tlbr(self):
        """Convert current position to [x1, y1, x2, y2]."""
        ret = self.tlwh.copy()
        ret[2] += ret[0]
        ret[3] += ret[1]
        return ret

    def predict(self):
        self.kf.predict()
        # Update tlwh from predicted state
        cx, cy, a, h = self.kf.x[:4]
        w = a * h
        self.tlwh = np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float32)

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1

        # Measurement update
        new_tlwh = new_track.tlwh
        cx = new_tlwh[0] + new_tlwh[2] / 2.0
        cy = new_tlwh[1] + new_tlwh[3] / 2.0
        a = new_tlwh[2] / max(1.0, new_tlwh[3])
        h = new_tlwh[3]

        z = np.array([cx, cy, a, h], dtype=np.float32)
        self.kf.update(z)

        self.tlwh = new_tlwh
        self.score = new_track.score
        self.state = 0
        self.is_activated = True

    def activate(self, frame_id):
        STrack._count += 1
        self.track_id = STrack._count
        self.frame_id = frame_id
        self.tracklet_len = 0
        self.state = 0
        self.is_activated = True


class ByteTracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, frame_rate=15):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.max_time_lost = int(frame_rate * 2)  # Keep lost tracks for 2 seconds

    def update(self, output_results):
        """
        Args:
            output_results: array of detections [[x1, y1, x2, y2, score], ...]
        Returns:
            online_targets: list of active STrack objects
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []

        if len(output_results) == 0:
            for track in self.tracked_stracks:
                track.state = 1
                self.lost_stracks.append(track)
            self.tracked_stracks = []
            return []

        # Convert detections to STrack instances
        bboxes = output_results[:, :4]
        scores = output_results[:, 4]

        # Convert [x1, y1, x2, y2] to [x1, y1, w, h]
        tlwhs = bboxes.copy()
        tlwhs[:, 2] -= tlwhs[:, 0]
        tlwhs[:, 3] -= tlwhs[:, 1]

        # 1. Split detections into High Confidence and Low Confidence (ByteTrack core logic)
        high_inds = scores >= self.track_thresh
        low_inds = scores < self.track_thresh

        detections_high = [STrack(tlwh, s) for tlwh, s in zip(tlwhs[high_inds], scores[high_inds])]
        detections_low = [STrack(tlwh, s) for tlwh, s in zip(tlwhs[low_inds], scores[low_inds])]

        # 2. Predict existing tracks' new locations using Kalman Filter
        for track in self.tracked_stracks:
            track.predict()

        # 3. First Association: High-confidence detections with existing tracks
        dists = self._matching_cost_matrix(self.tracked_stracks, detections_high)
        matches, u_track, u_det_high = self._linear_assignment(dists, thresh=self.match_thresh)

        for itr, idet in matches:
            track = self.tracked_stracks[itr]
            det = detections_high[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        # 4. Second Association: Unmatched tracks with Low-confidence detections (recovers flickering tiny boxes)
        r_tracked_stracks = [self.tracked_stracks[i] for i in u_track]
        dists = self._matching_cost_matrix(r_tracked_stracks, detections_low)
        matches, u_track_second, u_det_low = self._linear_assignment(dists, thresh=0.5)

        for itr, idet in matches:
            track = r_tracked_stracks[itr]
            det = detections_low[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        # Mark remaining unmatched tracks as lost
        for i in u_track_second:
            track = r_tracked_stracks[i]
            track.state = 1
            lost_stracks.append(track)

        # 5. Initialize new tracks for unmatched high-confidence detections
        for i in u_det_high:
            track = detections_high[i]
            if track.score >= self.track_thresh:
                track.activate(self.frame_id)
                activated_stracks.append(track)

        self.tracked_stracks = activated_stracks
        self.lost_stracks.extend(lost_stracks)
        return self.tracked_stracks

    def _matching_cost_matrix(self, tracks, detections):
        """Computes cost matrix combining Bounding Box IoU and Center Distance."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(tracks), len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes = np.array([d.tlbr for d in detections])

        # IoU distance (1 - IoU)
        iou_matrix = self._bbox_iou(track_boxes, det_boxes)
        iou_cost = 1.0 - iou_matrix

        # Center Distance distance (normalized by image scale ~640px)
        c_dist = self._center_distance(track_boxes, det_boxes) / 640.0

        # Weighted combination: IoU + Center Distance matching
        cost_matrix = 0.7 * iou_cost + 0.3 * c_dist
        return cost_matrix

    @staticmethod
    def _bbox_iou(box1, box2):
        x1 = np.maximum(box1[:, 0:1], box2[:, 0])
        y1 = np.maximum(box1[:, 1:2], box2[:, 1])
        x2 = np.minimum(box1[:, 2:3], box2[:, 2])
        y2 = np.minimum(box1[:, 3:4], box2[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1[:, None] + area2 - intersection
        return intersection / (union + 1e-6)

    @staticmethod
    def _center_distance(box1, box2):
        c1_x = (box1[:, 0:1] + box1[:, 2:3]) / 2.0
        c1_y = (box1[:, 1:2] + box1[:, 3:4]) / 2.0
        c2_x = (box2[:, 0] + box2[:, 2]) / 2.0
        c2_y = (box2[:, 1] + box2[:, 3]) / 2.0
        return np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

    @staticmethod
    def _linear_assignment(cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches, unmatched_a, unmatched_b = [], [], []

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > thresh:
                unmatched_a.append(r)
                unmatched_b.append(c)
            else:
                matches.append((r, c))

        for r in range(cost_matrix.shape[0]):
            if r not in row_ind:
                unmatched_a.append(r)
        for c in range(cost_matrix.shape[1]):
            if c not in col_ind:
                unmatched_b.append(c)

        return np.array(matches), tuple(unmatched_a), tuple(unmatched_b)