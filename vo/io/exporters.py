from __future__ import annotations

import os

import numpy as np

from vo.types import Keyframe


def export_trajectory_txt(trajectory: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    poses_3x4 = trajectory[:, :3, :]
    flat = poses_3x4.reshape(len(poses_3x4), 12)
    np.savetxt(out_path, flat, fmt="%.6f")


def export_keyframes_npz(keyframes: list[Keyframe], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for kf in keyframes:
        np.savez_compressed(
            os.path.join(out_dir, f"kf_{kf.frame_idx:06d}.npz"),
            frame_idx=kf.frame_idx,
            timestamp=kf.timestamp,
            pose_wc=kf.pose_wc,
            left_rgb=kf.left_rgb,
            left_gray=kf.left_gray,
            tracked_uv=kf.tracked_uv,
            landmark_ids=np.array(kf.landmark_ids, dtype=np.int64),
            new_points_w=kf.new_points_w,
            new_colors_rgb=kf.new_colors_rgb,
        )
