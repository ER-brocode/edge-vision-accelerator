"""Discrete 8-state Kalman Filter for 2D bounding boxes."""

from typing import List
import numpy as np


def convert_bbox_to_z(bbox: List[float]) -> np.ndarray:
    """Converts [x1, y1, x2, y2] to center form [x, y, s, r]

    where s is scale/area and r is aspect ratio (w/h).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray) -> List[float]:
    """Converts state vector [x, y, s, r, ...] back to [x1, y1, x2, y2]."""
    x_flat = np.asarray(x).reshape(-1)
    cx = float(x_flat[0])
    cy = float(x_flat[1])
    s = float(x_flat[2])
    r = float(x_flat[3])

    w = np.sqrt(max(0.0, s * r))
    h = s / w if w > 0 else np.sqrt(max(0.0, s))

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [x1, y1, x2, y2]


class KalmanBoxTracker:
    """Kalman filter tracking single bounding box state with constant velocity motion model."""

    count = 0

    def __init__(self, bbox: List[float], class_id: int = 0, label: str = "object", score: float = 1.0):
        # State vector: [x, y, s, r, vx, vy, vs, vr]
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[:4] = convert_bbox_to_z(bbox)

        # State transition matrix F
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement function H
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrices
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0

        self.R = np.eye(4, dtype=np.float32) * 1.0
        self.R[2:, 2:] *= 10.0

        self.Q = np.eye(8, dtype=np.float32) * 0.01
        self.Q[4:, 4:] *= 0.01

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: List[List[float]] = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.class_id = class_id
        self.label = label
        self.score = score
        self.features: List[np.ndarray] = []

    def update(self, bbox: List[float], score: float = 1.0):
        """Updates state with observed bounding box measurement."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = score

        z = convert_bbox_to_z(bbox)
        # Residual y = z - H * x
        y = z - np.dot(self.H, self.x)
        # Innovation covariance S = H * P * H.T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # Kalman Gain K = P * H.T * inv(S)
        try:
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        except np.linalg.LinAlgError:
            K = np.zeros((8, 4), dtype=np.float32)

        # Update state x = x + K * y
        self.x = self.x + np.dot(K, y)
        # Update covariance P = (I - K * H) * P
        I = np.eye(8, dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def predict(self) -> List[float]:
        """Advances state vector and returns predicted bounding box."""
        if (self.x[6] + self.x[2]) <= 0:
            self.x[6] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        pred_box = convert_x_to_bbox(self.x)
        self.history.append(pred_box)
        return pred_box

    def get_state(self) -> List[float]:
        """Returns current bounding box estimate."""
        return convert_x_to_bbox(self.x)
