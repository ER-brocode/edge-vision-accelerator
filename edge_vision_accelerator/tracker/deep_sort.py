"""DeepSORT tracker combining appearance cosine embeddings and Kalman motion filter."""

from typing import List, Optional, Tuple
import numpy as np

from edge_vision_accelerator.inference.detector import Detection
from edge_vision_accelerator.tracker.kalman import KalmanBoxTracker
from edge_vision_accelerator.tracker.matching import compute_iou_batch, linear_assignment
from edge_vision_accelerator.tracker.byte_track import TrackResult


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance between feature arrays: 1 - (a . b) / (|a| |b|)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
    similarity = np.dot(a_norm, b_norm.T)
    return np.clip(1.0 - similarity, 0.0, 2.0)


class DeepSORTTracker:
    """Deep Simple Online and Realtime Tracking with a Deep Association Metric."""

    def __init__(
        self,
        max_dist: float = 0.5,
        max_age: int = 30,
        n_init: int = 1,
        nn_budget: int = 100,
    ):
        self.max_dist = max_dist
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: List[Detection]) -> List[TrackResult]:
        self.frame_count += 1

        # 1. Predict
        for trk in self.trackers:
            trk.predict()

        # 2. Extract embeddings or synthesize if empty
        det_boxes = [d.bbox for d in detections]
        det_feats = []
        for d in detections:
            if d.embedding is not None and len(d.embedding) > 0:
                det_feats.append(d.embedding)
            else:
                det_feats.append(np.random.normal(0, 1, 128).astype(np.float32))

        det_feats_arr = np.array(det_feats, dtype=np.float32) if det_feats else np.empty((0, 128))

        # 3. Association: Appearance feature matching first
        if self.trackers and len(detections) > 0:
            trk_feats = []
            for trk in self.trackers:
                if trk.features:
                    trk_feats.append(np.mean(trk.features, axis=0))
                else:
                    trk_feats.append(np.random.normal(0, 1, 128).astype(np.float32))

            trk_feats_arr = np.array(trk_feats, dtype=np.float32)
            cost_matrix = cosine_distance(trk_feats_arr, det_feats_arr)

            matched_idx, unmatched_trks, unmatched_dets = linear_assignment(cost_matrix, thresh=self.max_dist)

            for t_idx, d_idx in matched_idx:
                self.trackers[t_idx].update(detections[d_idx].bbox, detections[d_idx].confidence)
                self.trackers[t_idx].class_id = detections[d_idx].class_id
                self.trackers[t_idx].label = detections[d_idx].label
                self.trackers[t_idx].features.append(det_feats_arr[d_idx])
                if len(self.trackers[t_idx].features) > self.nn_budget:
                    self.trackers[t_idx].features.pop(0)

            unmatched_dets_list = [detections[i] for i in unmatched_dets]
            unmatched_feats_list = [det_feats_arr[i] for i in unmatched_dets]
        else:
            unmatched_dets_list = detections
            unmatched_feats_list = det_feats

        # 4. Initiate new tracks
        for d, feat in zip(unmatched_dets_list, unmatched_feats_list):
            new_trk = KalmanBoxTracker(d.bbox, class_id=d.class_id, label=d.label, score=d.confidence)
            new_trk.features.append(feat)
            self.trackers.append(new_trk)

        # 5. Output active tracks
        active: List[TrackResult] = []
        for trk in list(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
                continue

            if trk.time_since_update < 1 and trk.hits >= self.n_init:
                active.append(TrackResult(
                    track_id=trk.id,
                    bbox=trk.get_state(),
                    class_id=trk.class_id,
                    label=trk.label,
                    confidence=trk.score,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update
                ))

        return active

    def reset(self):
        self.trackers = []
        self.frame_count = 0
