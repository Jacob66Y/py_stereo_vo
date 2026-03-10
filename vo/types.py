from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class StereoFrame:
    idx: int
    timestamp: float

    left_raw: np.ndarray
    right_raw: np.ndarray

    left_gray: np.ndarray
    right_gray: np.ndarray

    left_rgb: np.ndarray
    right_rgb: np.ndarray


@dataclass
class Landmark:
    landmark_id: int
    position_w: np.ndarray
    color_rgb: np.ndarray
    first_seen_idx: int
    last_seen_idx: int

    observations: int = 1
    active: bool = True

    track_fail_count: int = 0
    mean_reproj_error: float = 0.0


@dataclass
class Keyframe:
    frame_idx: int
    timestamp: float
    pose_wc: np.ndarray
    left_rgb: np.ndarray
    left_gray: np.ndarray
    tracked_uv: np.ndarray
    landmark_ids: List[int]
    new_points_w: np.ndarray
    new_colors_rgb: np.ndarray


@dataclass
class VOResult:
    trajectory_wc: np.ndarray
    keyframes: List[Keyframe]
    landmarks: Dict[int, Landmark]