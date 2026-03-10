from __future__ import annotations

import numpy as np

from vo.geometry.transforms import invert_pose, rotation_angle_deg


class KeyframeManager:
    def __init__(
        self,
        min_tracked_landmarks: int,
        translation_thresh: float,
        rotation_thresh_deg: float,
        frame_gap: int,
        max_frame_gap: int,
        low_track_ratio_thresh: float,
    ) -> None:
        self.min_tracked_landmarks = min_tracked_landmarks
        self.translation_thresh = translation_thresh
        self.rotation_thresh_deg = rotation_thresh_deg
        self.frame_gap = frame_gap
        self.max_frame_gap = max_frame_gap
        self.low_track_ratio_thresh = low_track_ratio_thresh

    def should_create(
        self,
        frame_idx: int,
        tracked_count: int,
        tracked_ratio: float,
        current_pose_wc: np.ndarray,
        last_keyframe_pose_wc: np.ndarray,
        last_keyframe_idx: int,
    ) -> bool:
        frames_since_last = frame_idx - last_keyframe_idx

        if frames_since_last >= self.max_frame_gap:
            return True

        if tracked_count < self.min_tracked_landmarks:
            return True

        if tracked_ratio < self.low_track_ratio_thresh:
            return True

        if frames_since_last < self.frame_gap:
            return False

        delta = invert_pose(last_keyframe_pose_wc) @ current_pose_wc
        t_norm = np.linalg.norm(delta[:3, 3])
        angle = rotation_angle_deg(delta[:3, :3])

        return (t_norm >= self.translation_thresh) or (angle >= self.rotation_thresh_deg)