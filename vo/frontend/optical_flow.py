from __future__ import annotations

import cv2
import numpy as np


class LKTracker:
    def __init__(self, win_size: int, max_level: int, max_error: float) -> None:
        self.max_error = max_error
        self.lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def track(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if prev_pts is None or len(prev_pts) == 0:
            return np.zeros((0, 1, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            return np.zeros((0, 1, 2), dtype=np.float32), np.zeros((len(prev_pts),), dtype=bool)

        status = status.reshape(-1).astype(bool)
        err = err.reshape(-1)

        h, w = curr_gray.shape
        uv = curr_pts.reshape(-1, 2)
        in_bounds = (
            (uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )

        good = status & in_bounds & (err < self.max_error)
        return curr_pts.astype(np.float32), good
