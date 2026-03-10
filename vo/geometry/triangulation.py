from __future__ import annotations

import numpy as np

from vo.geometry.camera import CameraModel


def disparity_to_depth(
    disparity: np.ndarray,
    fx: float,
    baseline: float,
    min_valid_disparity: float,
) -> np.ndarray:
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > min_valid_disparity
    depth[valid] = (fx * baseline) / disparity[valid]
    return depth


def sample_depths_at_points(
    pts: np.ndarray,
    depth_map: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)

    uv = pts.reshape(-1, 2)
    u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, depth_map.shape[1] - 1)
    v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, depth_map.shape[0] - 1)

    z = depth_map[v, u]
    valid = (z >= min_depth) & (z <= max_depth)
    return z, valid


def backproject_points(
    pts: np.ndarray,
    depths: np.ndarray,
    cam: CameraModel,
) -> np.ndarray:
    uv = pts.reshape(-1, 2)
    z = depths.reshape(-1)
    x = (uv[:, 0] - cam.cx) * z / cam.fx
    y = (uv[:, 1] - cam.cy) * z / cam.fy
    return np.column_stack([x, y, z]).astype(np.float64)
