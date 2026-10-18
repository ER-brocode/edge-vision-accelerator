"""Linear assignment utilities using Hungarian algorithm or Lapjv/SciPy."""

from typing import Tuple, List
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def compute_iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU between two lists of bounding boxes.

    bboxes_a: [N, 4] (x1, y1, x2, y2)
    bboxes_b: [M, 4] (x1, y1, x2, y2)
    Returns: [N, M] matrix of IoU values.
    """
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)

    # N, 1, 4 and 1, M, 4
    a = bboxes_a[:, np.newaxis, :]
    b = bboxes_b[np.newaxis, :, :]

    ix1 = np.maximum(a[:, :, 0], b[:, :, 0])
    iy1 = np.maximum(a[:, :, 1], b[:, :, 1])
    ix2 = np.minimum(a[:, :, 2], b[:, :, 2])
    iy2 = np.minimum(a[:, :, 3], b[:, :, 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = (a[:, :, 2] - a[:, :, 0]) * (a[:, :, 3] - a[:, :, 1])
    area_b = (b[:, :, 2] - b[:, :, 0]) * (b[:, :, 3] - b[:, :, 1])

    union = area_a + area_b - intersection
    union = np.maximum(union, 1e-6)

    return np.clip(intersection / union, 0.0, 1.0)


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matches detections to tracks based on cost matrix.

    Returns: (matches [K, 2], unmatched_a [U_a], unmatched_b [U_b])
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        # Greedy fallback assignment if scipy not available
        row_ind, col_ind = _greedy_assignment(cost_matrix)

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    return (
        np.array(matches, dtype=int).reshape((-1, 2)),
        np.array(unmatched_a, dtype=int),
        np.array(unmatched_b, dtype=int),
    )


def _greedy_assignment(cost_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
    rows, cols = cost_matrix.shape
    used_rows = set()
    used_cols = set()
    row_ind = []
    col_ind = []

    # Flatten indices sorted by cost
    flat_indices = np.argsort(cost_matrix, axis=None)
    for idx in flat_indices:
        r = idx // cols
        c = idx % cols
        if r not in used_rows and c not in used_cols:
            used_rows.add(r)
            used_cols.add(c)
            row_ind.append(r)
            col_ind.append(c)
            if len(used_rows) == rows or len(used_cols) == cols:
                break
    return row_ind, col_ind
