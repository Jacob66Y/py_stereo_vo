from __future__ import annotations

from dataclasses import replace
import threading
from typing import Dict, Iterable, List

import numpy as np

from vo.backend.local_ba import (
    BAKeyframeState,
    BALandmarkState,
    LocalBASnapshot,
    LocalBAResult,
)
from vo.types import Keyframe, Landmark


class MapManager:
    def __init__(self) -> None:
        self.landmarks: Dict[int, Landmark] = {}
        self.keyframes: List[Keyframe] = []
        self._next_landmark_id = 0
        self._map_version = 0
        self.lock = threading.RLock()

    @property
    def map_version(self) -> int:
        with self.lock:
            return self._map_version

    def _bump_version(self) -> None:
        self._map_version += 1

    def add_landmarks(
        self,
        points_w: np.ndarray,
        colors_rgb: np.ndarray,
        frame_idx: int,
    ) -> list[int]:
        ids: list[int] = []
        with self.lock:
            for p_w, color in zip(points_w, colors_rgb):
                lid = self._next_landmark_id
                self._next_landmark_id += 1
                self.landmarks[lid] = Landmark(
                    landmark_id=lid,
                    position_w=np.asarray(p_w, dtype=np.float64).copy(),
                    color_rgb=np.asarray(color, dtype=np.uint8).copy(),
                    first_seen_idx=frame_idx,
                    last_seen_idx=frame_idx,
                )
                ids.append(lid)
            if ids:
                self._bump_version()
        return ids

    def add_keyframe(self, keyframe: Keyframe) -> None:
        with self.lock:
            self.keyframes.append(keyframe)
            self._bump_version()

    def update_observations(self, landmark_ids: Iterable[int], frame_idx: int) -> None:
        with self.lock:
            changed = False
            for lid in landmark_ids:
                lm = self.landmarks.get(int(lid))
                if lm is None or not lm.active:
                    continue
                lm.observations += 1
                lm.last_seen_idx = frame_idx
                lm.track_fail_count = 0
                changed = True
            if changed:
                self._bump_version()

    def record_tracking_failures(self, landmark_ids: Iterable[int]) -> None:
        with self.lock:
            changed = False
            for lid in landmark_ids:
                lm = self.landmarks.get(int(lid))
                if lm is None or not lm.active:
                    continue
                lm.track_fail_count += 1
                changed = True
            if changed:
                self._bump_version()

    def record_reprojection_errors(
        self,
        landmark_ids: list[int],
        errors: np.ndarray,
        alpha: float,
    ) -> None:
        with self.lock:
            changed = False
            for lid, err in zip(landmark_ids, errors.tolist()):
                lm = self.landmarks.get(int(lid))
                if lm is None or not lm.active:
                    continue
                err_val = float(err)
                if lm.mean_reproj_error <= 1e-9:
                    lm.mean_reproj_error = err_val
                else:
                    lm.mean_reproj_error = (1.0 - alpha) * lm.mean_reproj_error + alpha * err_val
                changed = True
            if changed:
                self._bump_version()

    def get_landmark_points_for_ids(self, landmark_ids: list[int]) -> np.ndarray:
        with self.lock:
            pts = []
            for lid in landmark_ids:
                lm = self.landmarks.get(int(lid))
                if lm is not None and lm.active:
                    pts.append(lm.position_w)
            if len(pts) == 0:
                return np.zeros((0, 3), dtype=np.float64)
            return np.asarray(pts, dtype=np.float64)

    def get_active_landmarks_xz(self, max_points: int = 10000) -> np.ndarray:
        with self.lock:
            pts = [lm.position_w[[0, 2]] for lm in self.landmarks.values() if lm.active]
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) > max_points:
            pts = pts[:max_points]
        return pts

    def num_active_landmarks(self) -> int:
        with self.lock:
            return sum(1 for lm in self.landmarks.values() if lm.active)

    def num_total_landmarks(self) -> int:
        with self.lock:
            return len(self.landmarks)

    def prune_landmarks(
        self,
        current_frame_idx: int,
        stale_age: int,
        protected_min_obs: int,
        max_track_failures: int,
        max_reproj_error: float,
    ) -> int:
        removed = 0
        with self.lock:
            for lm in self.landmarks.values():
                if not lm.active:
                    continue

                age_since_seen = current_frame_idx - lm.last_seen_idx
                protected = lm.observations >= protected_min_obs

                if age_since_seen > stale_age and not protected:
                    lm.active = False
                    removed += 1
                    continue

                if lm.track_fail_count > max_track_failures and not protected:
                    lm.active = False
                    removed += 1
                    continue

                if lm.mean_reproj_error > max_reproj_error and lm.observations < protected_min_obs:
                    lm.active = False
                    removed += 1

            if removed > 0:
                self._bump_version()
        return removed

    def snapshot_for_local_ba(
        self,
        window_size: int,
        min_landmark_observations: int,
    ) -> LocalBASnapshot | None:
        with self.lock:
            if len(self.keyframes) < 2:
                return None

            local_keyframes = self.keyframes[-window_size:]
            if len(local_keyframes) < 2:
                return None

            used_landmark_ids = set()
            ba_keyframes: list[BAKeyframeState] = []

            for kf in local_keyframes:
                ba_keyframes.append(
                    BAKeyframeState(
                        frame_idx=int(kf.frame_idx),
                        pose_wc=np.asarray(kf.pose_wc, dtype=np.float64).copy(),
                        tracked_uv=np.asarray(kf.tracked_uv, dtype=np.float64).copy(),
                        landmark_ids=list(kf.landmark_ids),
                    )
                )
                used_landmark_ids.update(int(x) for x in kf.landmark_ids if x is not None)

            ba_landmarks: dict[int, BALandmarkState] = {}
            for lid in used_landmark_ids:
                lm = self.landmarks.get(lid)
                if lm is None or not lm.active:
                    continue
                if lm.observations < min_landmark_observations:
                    continue
                ba_landmarks[lid] = BALandmarkState(
                    landmark_id=int(lm.landmark_id),
                    position_w=np.asarray(lm.position_w, dtype=np.float64).copy(),
                    observations=int(lm.observations),
                    mean_reproj_error=float(lm.mean_reproj_error),
                    active=bool(lm.active),
                )

            if len(ba_landmarks) < 5:
                return None

            return LocalBASnapshot(
                map_version=self._map_version,
                keyframes=ba_keyframes,
                landmarks=ba_landmarks,
            )

    def apply_local_ba_result(
        self,
        result: LocalBAResult,
        max_version_lag: int,
    ) -> bool:
        with self.lock:
            if (self._map_version - result.snapshot_version) > max_version_lag:
                return False

            kf_lookup = {int(kf.frame_idx): kf for kf in self.keyframes}
            changed = False

            for kf_id, pose_wc in result.updated_keyframe_poses.items():
                if kf_id in kf_lookup:
                    kf_lookup[kf_id].pose_wc = np.asarray(pose_wc, dtype=np.float64).copy()
                    changed = True

            for lm_id, pos_w in result.updated_landmark_positions.items():
                lm = self.landmarks.get(int(lm_id))
                if lm is None or not lm.active:
                    continue
                lm.position_w = np.asarray(pos_w, dtype=np.float64).copy()
                changed = True

            if changed:
                self._bump_version()
            return changed