from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class BAObservation:
    keyframe_id: int
    landmark_id: int
    uv: np.ndarray


@dataclass
class BAKeyframeState:
    frame_idx: int
    pose_wc: np.ndarray
    tracked_uv: np.ndarray
    landmark_ids: list[int]


@dataclass
class BALandmarkState:
    landmark_id: int
    position_w: np.ndarray
    observations: int
    mean_reproj_error: float
    active: bool = True


@dataclass
class LocalBASnapshot:
    map_version: int
    keyframes: list[BAKeyframeState]
    landmarks: dict[int, BALandmarkState]


@dataclass
class LocalBAResult:
    snapshot_version: int
    updated_keyframe_poses: dict[int, np.ndarray]
    updated_landmark_positions: dict[int, np.ndarray]
    num_observations: int


class LocalBundleAdjuster:
    def __init__(
        self,
        cam,
        window_size: int = 5,
        max_nfev: int = 20,
        huber_delta: float = 5.0,
        min_observations: int = 20,
        verbose: bool = False,
    ) -> None:
        self.cam = cam
        self.window_size = window_size
        self.max_nfev = max_nfev
        self.huber_delta = huber_delta
        self.min_observations = min_observations
        self.verbose = verbose

    def optimize_snapshot(self, snapshot: LocalBASnapshot) -> LocalBAResult | None:
        if snapshot is None or len(snapshot.keyframes) < 2:
            return None

        observations = self._collect_observations(snapshot.keyframes, snapshot.landmarks)
        if len(observations) < self.min_observations:
            return None

        kf_ids, lm_ids = self._collect_unique_ids(observations)
        if len(kf_ids) < 2 or len(lm_ids) < 5:
            return None

        anchor_kf_id = kf_ids[0]
        opt_kf_ids = kf_ids[1:]

        x0 = self._pack_parameters(snapshot.keyframes, snapshot.landmarks, opt_kf_ids, lm_ids)
        if x0.size == 0:
            return None

        result = least_squares(
            fun=self._residuals,
            x0=x0,
            loss="huber",
            f_scale=self.huber_delta,
            max_nfev=self.max_nfev,
            verbose=2 if self.verbose else 0,
            args=(
                observations,
                snapshot.keyframes,
                snapshot.landmarks,
                anchor_kf_id,
                opt_kf_ids,
                lm_ids,
            ),
        )

        if result.x is None or result.x.size == 0:
            return None

        pose_dict, point_dict = self._decode_parameters(
            result.x,
            snapshot.keyframes,
            snapshot.landmarks,
            anchor_kf_id,
            opt_kf_ids,
            lm_ids,
        )

        updated_keyframe_poses: dict[int, np.ndarray] = {}
        updated_landmark_positions: dict[int, np.ndarray] = {}

        for kf_id, T_cw in pose_dict.items():
            updated_keyframe_poses[int(kf_id)] = np.linalg.inv(T_cw)

        for lm_id, pw in point_dict.items():
            updated_landmark_positions[int(lm_id)] = np.asarray(pw, dtype=np.float64).copy()

        return LocalBAResult(
            snapshot_version=int(snapshot.map_version),
            updated_keyframe_poses=updated_keyframe_poses,
            updated_landmark_positions=updated_landmark_positions,
            num_observations=len(observations),
        )

    def _collect_observations(
        self,
        keyframes: list[BAKeyframeState],
        landmarks: dict[int, BALandmarkState],
    ) -> list[BAObservation]:
        obs: list[BAObservation] = []

        for kf in keyframes:
            pts = np.asarray(kf.tracked_uv, dtype=np.float64).reshape(-1, 2)
            lm_ids = list(kf.landmark_ids)

            n = min(len(pts), len(lm_ids))
            if n == 0:
                continue

            pts = pts[:n]
            lm_ids = lm_ids[:n]

            for uv, lm_id in zip(pts, lm_ids):
                if lm_id is None:
                    continue
                lm_id = int(lm_id)
                lm = landmarks.get(lm_id)
                if lm is None or not lm.active:
                    continue

                obs.append(
                    BAObservation(
                        keyframe_id=int(kf.frame_idx),
                        landmark_id=lm_id,
                        uv=np.asarray(uv, dtype=np.float64).reshape(2),
                    )
                )

        return obs

    def _collect_unique_ids(self, observations: list[BAObservation]) -> Tuple[list[int], list[int]]:
        kf_ids = sorted({o.keyframe_id for o in observations})
        lm_ids = sorted({o.landmark_id for o in observations})
        return list(kf_ids), list(lm_ids)

    def _pack_parameters(
        self,
        keyframes: list[BAKeyframeState],
        landmarks: dict[int, BALandmarkState],
        opt_kf_ids: list[int],
        lm_ids: list[int],
    ) -> np.ndarray:
        parts = []
        kf_lookup = {int(kf.frame_idx): kf for kf in keyframes}

        for kf_id in opt_kf_ids:
            kf = kf_lookup[kf_id]
            T_wc = np.asarray(kf.pose_wc, dtype=np.float64)
            T_cw = np.linalg.inv(T_wc)
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]

            rvec, _ = cv2.Rodrigues(R_cw)
            parts.append(rvec.reshape(3))
            parts.append(t_cw.reshape(3))

        for lm_id in lm_ids:
            parts.append(np.asarray(landmarks[lm_id].position_w, dtype=np.float64).reshape(3))

        if len(parts) == 0:
            return np.array([], dtype=np.float64)

        return np.concatenate(parts, axis=0)

    def _decode_parameters(
        self,
        x: np.ndarray,
        keyframes: list[BAKeyframeState],
        landmarks: dict[int, BALandmarkState],
        anchor_kf_id: int,
        opt_kf_ids: list[int],
        lm_ids: list[int],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        pose_dict: dict[int, np.ndarray] = {}
        point_dict: dict[int, np.ndarray] = {}

        kf_lookup = {int(kf.frame_idx): kf for kf in keyframes}
        offset = 0

        anchor_kf = kf_lookup[anchor_kf_id]
        pose_dict[anchor_kf_id] = np.linalg.inv(np.asarray(anchor_kf.pose_wc, dtype=np.float64))

        for kf_id in opt_kf_ids:
            rvec = x[offset:offset + 3]
            offset += 3
            tvec = x[offset:offset + 3]
            offset += 3

            R_cw, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            T_cw = np.eye(4, dtype=np.float64)
            T_cw[:3, :3] = R_cw
            T_cw[:3, 3] = tvec.reshape(3)

            pose_dict[kf_id] = T_cw

        for lm_id in lm_ids:
            pw = x[offset:offset + 3]
            offset += 3
            point_dict[lm_id] = pw.reshape(3)

        return pose_dict, point_dict

    def _residuals(
        self,
        x: np.ndarray,
        observations: list[BAObservation],
        keyframes: list[BAKeyframeState],
        landmarks: dict[int, BALandmarkState],
        anchor_kf_id: int,
        opt_kf_ids: list[int],
        lm_ids: list[int],
    ) -> np.ndarray:
        fx, fy, cx, cy = self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy

        pose_dict, point_dict = self._decode_parameters(
            x,
            keyframes,
            landmarks,
            anchor_kf_id,
            opt_kf_ids,
            lm_ids,
        )

        residuals: list[float] = []

        for obs in observations:
            if obs.keyframe_id not in pose_dict or obs.landmark_id not in point_dict:
                continue

            T_cw = pose_dict[obs.keyframe_id]
            pw = point_dict[obs.landmark_id]

            pc = T_cw[:3, :3] @ pw + T_cw[:3, 3]
            X, Y, Z = pc

            if Z <= 1e-6:
                residuals.extend([50.0, 50.0])
                continue

            u = fx * X / Z + cx
            v = fy * Y / Z + cy

            residuals.append(float(u - obs.uv[0]))
            residuals.append(float(v - obs.uv[1]))

        return np.asarray(residuals, dtype=np.float64)