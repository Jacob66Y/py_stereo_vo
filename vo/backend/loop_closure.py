from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from vo.geometry.transforms import invert_pose
from vo.types import Keyframe, PoseGraphEdge


@dataclass
class LoopDetectionResult:
    matched_keyframe_idx: int
    current_keyframe_idx: int
    T_ij: np.ndarray
    num_raw_matches: int
    num_pnp_inliers: int


class LoopClosureDetector:
    def __init__(
        self,
        K: np.ndarray,
        landmarks: dict,
        orb_nfeatures: int = 1200,
        knn_ratio: float = 0.75,
        min_raw_matches: int = 50,
        min_pnp_inliers: int = 25,
        pnp_reproj_error: float = 4.0,
    ) -> None:
        self.K = K.astype(np.float64)
        self.landmarks = landmarks
        self.orb = cv2.ORB_create(nfeatures=orb_nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.knn_ratio = knn_ratio
        self.min_raw_matches = min_raw_matches
        self.min_pnp_inliers = min_pnp_inliers
        self.pnp_reproj_error = pnp_reproj_error

    def describe_keyframe(self, kf: Keyframe, assign_radius_px: float = 4.0) -> None:
        keypoints, descriptors = self.orb.detectAndCompute(kf.left_gray, None)
        if descriptors is None or len(keypoints) == 0:
            kf.loop_uv = np.zeros((0, 2), dtype=np.float32)
            kf.loop_descriptors = None
            kf.loop_landmark_ids = []
            return

        uv = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
        tracked = np.asarray(kf.tracked_uv, dtype=np.float32).reshape(-1, 2)

        assigned_ids: list[int] = []
        if len(tracked) == 0:
            assigned_ids = [-1] * len(uv)
        else:
            for p in uv:
                d = np.linalg.norm(tracked - p.reshape(1, 2), axis=1)
                idx = int(np.argmin(d))
                if float(d[idx]) <= assign_radius_px:
                    assigned_ids.append(int(kf.landmark_ids[idx]))
                else:
                    assigned_ids.append(-1)

        kf.loop_uv = uv
        kf.loop_descriptors = descriptors.copy()
        kf.loop_landmark_ids = assigned_ids

    def detect(
        self,
        current_kf: Keyframe,
        all_keyframes: list[Keyframe],
        min_frame_gap: int,
    ) -> Optional[LoopDetectionResult]:
        if current_kf.loop_descriptors is None or current_kf.loop_uv is None:
            return None

        best_result = None
        best_inliers = -1

        for cand in all_keyframes:
            if (current_kf.frame_idx - cand.frame_idx) < min_frame_gap:
                continue
            if cand.loop_descriptors is None or cand.loop_uv is None:
                continue

            knn = self.matcher.knnMatch(current_kf.loop_descriptors, cand.loop_descriptors, k=2)
            good = []
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.knn_ratio * n.distance:
                    good.append(m)

            if len(good) < self.min_raw_matches:
                continue

            points_w = []
            image_pts = []

            for m in good:
                q_idx = int(m.queryIdx)
                t_idx = int(m.trainIdx)

                lm_id = int(cand.loop_landmark_ids[t_idx]) if cand.loop_landmark_ids is not None else -1
                if lm_id < 0:
                    continue

                lm = self.landmarks.get(lm_id)
                if lm is None or not lm.active:
                    continue

                points_w.append(lm.position_w)
                image_pts.append(current_kf.loop_uv[q_idx])

            if len(points_w) < self.min_pnp_inliers:
                continue

            points_w = np.asarray(points_w, dtype=np.float64)
            image_pts = np.asarray(image_pts, dtype=np.float64)

            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_w,
                image_pts,
                self.K,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=self.pnp_reproj_error,
                confidence=0.99,
                iterationsCount=200,
            )

            if not ok or inliers is None or len(inliers) < self.min_pnp_inliers:
                continue

            R_cw, _ = cv2.Rodrigues(rvec)
            T_cw = np.eye(4, dtype=np.float64)
            T_cw[:3, :3] = R_cw
            T_cw[:3, 3] = tvec.reshape(3)

            T_wj = invert_pose(T_cw)          # current keyframe pose from loop PnP
            T_wi = cand.pose_wc.copy()        # candidate old keyframe pose
            T_ij = invert_pose(T_wi) @ T_wj

            if len(inliers) > best_inliers:
                best_inliers = len(inliers)
                best_result = LoopDetectionResult(
                    matched_keyframe_idx=int(cand.frame_idx),
                    current_keyframe_idx=int(current_kf.frame_idx),
                    T_ij=T_ij,
                    num_raw_matches=len(good),
                    num_pnp_inliers=len(inliers),
                )

        return best_result