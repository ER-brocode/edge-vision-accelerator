import pytest
import numpy as np
from edge_vision_accelerator.inference.detector import Detection
from edge_vision_accelerator.tracker.kalman import KalmanBoxTracker, convert_bbox_to_z, convert_x_to_bbox
from edge_vision_accelerator.tracker.matching import compute_iou_batch, linear_assignment
from edge_vision_accelerator.tracker.byte_track import ByteTracker
from edge_vision_accelerator.tracker.deep_sort import DeepSORTTracker


def test_kalman_box_tracker_motion():
    init_box = [100.0, 100.0, 200.0, 200.0]
    tracker = KalmanBoxTracker(init_box)

    pred = tracker.predict()
    assert len(pred) == 4
    # State should remain approximately centered
    assert abs(pred[0] - 100.0) < 5.0

    # Simulate object moving rightwards
    new_box = [110.0, 100.0, 210.0, 200.0]
    tracker.update(new_box)

    pred2 = tracker.predict()
    # Velocity should carry box further to right (> 110)
    assert pred2[0] > 105.0


def test_linear_assignment_matching():
    boxes_a = np.array([[0, 0, 10, 10], [50, 50, 70, 70]], dtype=np.float32)
    boxes_b = np.array([[0, 0, 10, 10], [100, 100, 120, 120]], dtype=np.float32)

    iou_mat = compute_iou_batch(boxes_a, boxes_b)
    cost = 1.0 - iou_mat

    matches, unmatched_a, unmatched_b = linear_assignment(cost, thresh=0.5)

    assert len(matches) == 1
    assert matches[0][0] == 0 and matches[0][1] == 0
    assert 1 in unmatched_a
    assert 1 in unmatched_b


def test_byte_tracker_association():
    tracker = ByteTracker(track_thresh=0.5, low_thresh=0.1, min_hits=1)

    # Frame 1: First appearance
    dets_f1 = [
        Detection(bbox=[100, 100, 150, 200], class_id=0, confidence=0.92, label="person")
    ]
    tracks_f1 = tracker.update(dets_f1)
    assert len(tracks_f1) == 1
    assigned_id = tracks_f1[0].track_id

    # Frame 2: Object moves slightly, confidence drops slightly (second-stage match)
    dets_f2 = [
        Detection(bbox=[105, 100, 155, 200], class_id=0, confidence=0.45, label="person")
    ]
    tracks_f2 = tracker.update(dets_f2)
    assert len(tracks_f2) == 1
    # Identity must be retained!
    assert tracks_f2[0].track_id == assigned_id


def test_deep_sort_tracker_association():
    tracker = DeepSORTTracker(n_init=1, max_age=5)

    feat1 = np.ones(128, dtype=np.float32)
    dets_f1 = [
        Detection(bbox=[50, 50, 100, 100], class_id=2, confidence=0.88, label="car", embedding=feat1)
    ]
    res1 = tracker.update(dets_f1)
    assert len(res1) == 1
    trk_id = res1[0].track_id

    # Frame 2: Same embedding slightly shifted
    feat2 = np.ones(128, dtype=np.float32) * 0.99
    dets_f2 = [
        Detection(bbox=[52, 51, 102, 101], class_id=2, confidence=0.89, label="car", embedding=feat2)
    ]
    res2 = tracker.update(dets_f2)
    assert len(res2) == 1
    assert res2[0].track_id == trk_id
