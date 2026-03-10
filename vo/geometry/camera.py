from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    K: np.ndarray

    @staticmethod
    def from_projection_matrices(P_left: np.ndarray, P_right: np.ndarray) -> "CameraModel":
        fx = float(P_left[0, 0])
        fy = float(P_left[1, 1])
        cx = float(P_left[0, 2])
        cy = float(P_left[1, 2])

        tx_left = P_left[0, 3] / P_left[0, 0]
        tx_right = P_right[0, 3] / P_right[0, 0]
        baseline = abs(tx_right - tx_left)

        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return CameraModel(fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline, K=K)
