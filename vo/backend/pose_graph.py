from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from vo.types import PoseGraphEdge


def _pose_to_rtvec(T_wc: np.ndarray) -> np.ndarray:
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.reshape(3), t.reshape(3)], axis=0)


def _rtvec_to_pose(x: np.ndarray) -> np.ndarray:
    rvec = x[:3].reshape(3, 1)
    t = x[3:6].reshape(3)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _se3_residual(T_meas: np.ndarray, T_pred: np.ndarray) -> np.ndarray:
    T_err = np.linalg.inv(T_meas) @ T_pred
    rvec, _ = cv2.Rodrigues(T_err[:3, :3])
    t = T_err[:3, 3]
    return np.concatenate([rvec.reshape(3), t.reshape(3)], axis=0)


class PoseGraphOptimizer:
    def __init__(self, odom_weight: float = 1.0, loop_weight: float = 5.0, max_nfev: int = 50) -> None:
        self.odom_weight = odom_weight
        self.loop_weight = loop_weight
        self.max_nfev = max_nfev

    def optimize(self, keyframes: list, edges: list[PoseGraphEdge]) -> dict[int, np.ndarray]:
        if len(keyframes) < 2 or len(edges) == 0:
            return {int(kf.frame_idx): kf.pose_wc.copy() for kf in keyframes}

        keyframes = sorted(keyframes, key=lambda k: int(k.frame_idx))
        anchor_id = int(keyframes[0].frame_idx)
        opt_ids = [int(k.frame_idx) for k in keyframes[1:]]
        kf_lookup = {int(kf.frame_idx): kf for kf in keyframes}

        x0_parts = []
        for kf_id in opt_ids:
            x0_parts.append(_pose_to_rtvec(kf_lookup[kf_id].pose_wc))
        x0 = np.concatenate(x0_parts, axis=0)

        def decode(x: np.ndarray) -> dict[int, np.ndarray]:
            poses = {anchor_id: kf_lookup[anchor_id].pose_wc.copy()}
            offset = 0
            for kf_id in opt_ids:
                poses[kf_id] = _rtvec_to_pose(x[offset:offset + 6])
                offset += 6
            return poses

        def residuals(x: np.ndarray) -> np.ndarray:
            poses = decode(x)
            res = []
            for e in edges:
                if e.i not in poses or e.j not in poses:
                    continue
                T_pred = np.linalg.inv(poses[e.i]) @ poses[e.j]
                r = _se3_residual(e.T_ij, T_pred)
                w = self.loop_weight if e.edge_type == "loop" else self.odom_weight
                res.extend((np.sqrt(w) * r).tolist())
            return np.asarray(res, dtype=np.float64)

        result = least_squares(residuals, x0=x0, loss="huber", f_scale=1.0, max_nfev=self.max_nfev)
        poses = decode(result.x)

        return poses