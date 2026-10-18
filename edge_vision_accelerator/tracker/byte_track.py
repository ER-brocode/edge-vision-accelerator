"""ByteTrack multi-object tracking implementation."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from edge_vision_accelerator.inference.detector import Detection
from edge_vision_accelerator.tracker.kalman import KalmanBoxTracker
from edge_vision_accelerator.tracker.matching import compute_iou_batch, linear_assignment


@dataclass
class TrackResult:
    track_id: int
    bbox: List[float]
    class_id: int
    label: str
    confidence: float
    age: int
    hits: int
    time_since_update: int


class ByteTracker:
    """ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

    Tracks high-confidence and low-confidence detections separately via two-stage association.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_age: int = 30,
        min_hits: int = 2,
    ):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: List[Detection]) -> List[TrackResult]:
        """Processes new frame detections and updates track states."""
        self.frame_count += 1

        # Step 1: Predict positions of existing trackers
        predicted_boxes = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(i)
            else:
                predicted_boxes.append(pos)

        for i in reversed(to_del):
            self.trackers.pop(i)

        # Step 2: Split detections into high-score and low-score pools
        dets_first = [d for d in detections if d.confidence >= self.track_thresh]
        dets_second = [d for d in detections if self.low_thresh <= d.confidence < self.track_thresh]

        # Step 3: First association (High score detections with all trackers)
        first_track_indices = list(range(len(self.trackers)))
        if predicted_boxes and dets_first:
            trk_boxes = np.array(predicted_boxes, dtype=np.float32)
            det_boxes = np.array([d.bbox for d in dets_first], dtype=np.float32)
            iou_matrix = compute_iou_batch(trk_boxes, det_boxes)
            cost_matrix = 1.0 - iou_matrix

            matched_idx, unmatched_trks, unmatched_dets = linear_assignment(
                cost_matrix, thresh=1.0 - (1.0 - self.match_thresh)
            )

            for t_idx, d_idx in matched_idx:
                self.trackers[t_idx].update(dets_first[d_idx].bbox, dets_first[d_idx].confidence)
                self.trackers[t_idx].class_id = dets_first[d_idx].class_id
                self.trackers[t_idx].label = dets_first[d_idx].label

            rem_trks = [self.trackers[i] for i in unmatched_trks]
            rem_trk_indices = [first_track_indices[i] for i in unmatched_trks]
            unmatched_dets_first = [dets_first[i] for i in unmatched_dets]
        else:
            rem_trks = self.trackers
            rem_trk_indices = first_track_indices
            unmatched_dets_first = dets_first

        # Step 4: Second association (Remaining unmatched trackers with low score detections)
        if rem_trks and dets_second:
            trk_boxes = np.array([trk.get_state() for trk in rem_trks], dtype=np.float32)
            det_boxes = np.array([d.bbox for d in dets_second], dtype=np.float32)
            iou_matrix = compute_iou_batch(trk_boxes, det_boxes)
            cost_matrix = 1.0 - iou_matrix

            matched_idx, unmatched_trks2, _ = linear_assignment(cost_matrix, thresh=0.5)

            for t_idx, d_idx in matched_idx:
                real_trk_idx = rem_trk_indices[t_idx]
                self.trackers[real_trk_idx].update(dets_second[d_idx].bbox, dets_second[d_idx].confidence)

        # Step 5: Initialize new tracks for unmatched high-confidence detections
        for det in unmatched_dets_first:
            new_trk = KalmanBoxTracker(det.bbox, class_id=det.class_id, label=det.label, score=det.confidence)
            self.trackers.append(new_trk)

        # Step 6: Prune dead tracks and prepare output
        active_results: List[TrackResult] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            bbox = trk.get_state()
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
                continue

            # Return track if it satisfies confirmation conditions
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                active_results.append(TrackResult(
                    track_id=trk.id,
                    bbox=bbox,
                    class_id=trk.class_id,
                    label=trk.label,
                    confidence=trk.score,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update
                ))

        return active_results

    def reset(self):
        self.trackers = []
        self.frame_count = 0
