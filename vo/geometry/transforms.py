from __future__ import annotations

import numpy as np


def invert_pose(T_ab: np.ndarray) -> np.ndarray:
    R = T_ab[:3, :3]
    t = T_ab[:3, 3]
    T_ba = np.eye(4, dtype=np.float64)
    T_ba[:3, :3] = R.T
    T_ba[:3, 3] = -R.T @ t
    return T_ba


def rt_to_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def transform_points(T_ab: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    R = T_ab[:3, :3]
    t = T_ab[:3, 3]
    return (R @ points_b.T).T + t.reshape(1, 3)


def rotation_angle_deg(R: np.ndarray) -> float:
    val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))
