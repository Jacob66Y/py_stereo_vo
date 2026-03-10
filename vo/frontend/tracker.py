from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from vo.backend.backend_worker import AsyncLocalBAWorker
from vo.backend.local_ba import LocalBundleAdjuster
from vo.config import VOConfig
from vo.frontend.detector import GFTTDetector
from vo.frontend.keyframe_manager import KeyframeManager
from vo.frontend.optical_flow import LKTracker
from vo.frontend.stereo_matcher import StereoSGBMMatcher
from vo.geometry.camera import CameraModel
from vo.geometry.filtering import make_feature_mask
from vo.geometry.pnp import solve_pnp_ransac
from vo.geometry.transforms import invert_pose, rt_to_pose, transform_points
from vo.geometry.triangulation import (
    backproject_points,
    disparity_to_depth,
    sample_depths_at_points,
)
from vo.map.map_manager import MapManager
from vo.types import Keyframe, VOResult

try:
    from vo.viz.viewer_2d import Viewer2D
except Exception:
    Viewer2D = None


@dataclass
class TrackerState:
    prev_gray: np.ndarray | None = None
    prev_pts: np.ndarray | None = None
    prev_ids: list[int] | None = None
    current_pose_wc: np.ndarray | None = None
    last_keyframe_pose_wc: np.ndarray | None = None
    last_keyframe_idx: int = 0
    last_keyframe_feature_count: int = 0


class StereoVOTracker:
    def __init__(self, dataset, cfg: VOConfig) -> None:
        self.dataset = dataset
        self.cfg = cfg
        self.cam = CameraModel.from_projection_matrices(dataset.P_left, dataset.P_right)

        self.detector = GFTTDetector(
            max_corners=cfg.tracker.max_corners,
            quality_level=cfg.tracker.quality_level,
            min_distance=cfg.tracker.min_distance,
            block_size=cfg.tracker.block_size,
        )

        self.lk = LKTracker(
            win_size=cfg.tracker.lk_win_size,
            max_level=cfg.tracker.lk_max_level,
            max_error=cfg.tracker.lk_max_error,
        )

        self.stereo = StereoSGBMMatcher(
            min_disparity=cfg.stereo.min_disparity,
            num_disparities=cfg.stereo.num_disparities,
            block_size=cfg.stereo.block_size,
        )

        self.keyframe_manager = KeyframeManager(
            min_tracked_landmarks=cfg.keyframe.min_tracked_landmarks,
            translation_thresh=cfg.keyframe.translation_thresh,
            rotation_thresh_deg=cfg.keyframe.rotation_thresh_deg,
            frame_gap=cfg.keyframe.frame_gap,
            max_frame_gap=cfg.keyframe.max_frame_gap,
            low_track_ratio_thresh=cfg.keyframe.low_track_ratio_thresh,
        )

        self.map_manager = MapManager()

        self.local_ba_worker = None
        if cfg.backend.enable_local_ba:
            optimizer = LocalBundleAdjuster(
                cam=self.cam,
                window_size=cfg.backend.local_ba_window_size,
                max_nfev=cfg.backend.local_ba_max_nfev,
                huber_delta=cfg.backend.local_ba_huber_delta,
                min_observations=cfg.backend.local_ba_min_observations,
                verbose=cfg.backend.local_ba_verbose,
            )
            self.local_ba_worker = AsyncLocalBAWorker(
                map_manager=self.map_manager,
                optimizer=optimizer,
                max_version_lag=cfg.backend.local_ba_max_version_lag,
            )
            self.local_ba_worker.start()

        self.state = TrackerState(
            current_pose_wc=np.eye(4, dtype=np.float64),
            last_keyframe_pose_wc=np.eye(4, dtype=np.float64),
            last_keyframe_idx=0,
            prev_ids=[],
            last_keyframe_feature_count=0,
        )

        self.trajectory_wc: list[np.ndarray] = []

        live_view = getattr(cfg.output, "live_view", False)
        if live_view and Viewer2D is not None:
            self.viewer = Viewer2D(
                show_separate_traj_window=getattr(cfg.output, "show_separate_traj_window", True)
            )
        else:
            self.viewer = None

        self.last_frame_time = None
        self.fps = 0.0
        self.fps_alpha = 0.9

    def _update_fps(self) -> None:
        now = time.perf_counter()
        if self.last_frame_time is None:
            self.last_frame_time = now
            self.fps = 0.0
            return

        dt = now - self.last_frame_time
        self.last_frame_time = now

        if dt > 0:
            inst_fps = 1.0 / dt
            if self.fps == 0.0:
                self.fps = inst_fps
            else:
                self.fps = self.fps_alpha * self.fps + (1.0 - self.fps_alpha) * inst_fps

    def _sample_colors(self, image_rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
        if pts is None or len(pts) == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        uv = pts.reshape(-1, 2)
        u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, image_rgb.shape[1] - 1)
        v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, image_rgb.shape[0] - 1)
        return image_rgb[v, u]

    def _compute_reprojection_errors(
        self,
        points_w: np.ndarray,
        image_pts: np.ndarray,
        R_cw: np.ndarray,
        t_cw: np.ndarray,
    ) -> np.ndarray:
        pc = (R_cw @ points_w.T).T + t_cw.reshape(1, 3)
        z = np.clip(pc[:, 2], 1e-6, None)
        u = self.cam.fx * pc[:, 0] / z + self.cam.cx
        v = self.cam.fy * pc[:, 1] / z + self.cam.cy
        proj = np.stack([u, v], axis=1)
        return np.linalg.norm(proj - image_pts, axis=1)

    def _create_landmarks_from_keyframe(
        self,
        frame,
        existing_pts: np.ndarray | None,
    ) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
        disparity = self.stereo.compute(frame.left_gray, frame.right_gray)

        depth_map = disparity_to_depth(
            disparity=disparity,
            fx=self.cam.fx,
            baseline=self.cam.baseline,
            min_valid_disparity=self.cfg.stereo.min_valid_disparity,
        )

        mask = make_feature_mask(
            image_shape=frame.left_gray.shape,
            existing_pts=existing_pts,
            radius=self.cfg.keyframe.new_feature_mask_radius,
        )

        new_pts = self.detector.detect(frame.left_gray, mask=mask)

        if len(new_pts) > self.cfg.keyframe.max_new_landmarks:
            new_pts = new_pts[: self.cfg.keyframe.max_new_landmarks]

        if len(new_pts) == 0:
            return (
                np.zeros((0, 1, 2), dtype=np.float32),
                [],
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.uint8),
            )

        depths, valid = sample_depths_at_points(
            pts=new_pts,
            depth_map=depth_map,
            min_depth=self.cfg.stereo.min_depth,
            max_depth=self.cfg.stereo.max_depth,
        )

        new_pts = new_pts[valid]
        depths = depths[valid]

        if len(new_pts) == 0:
            return (
                np.zeros((0, 1, 2), dtype=np.float32),
                [],
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.uint8),
            )

        points_c = backproject_points(new_pts, depths, self.cam)
        points_w = transform_points(self.state.current_pose_wc, points_c)
        colors_rgb = self._sample_colors(frame.left_rgb, new_pts)

        landmark_ids = self.map_manager.add_landmarks(
            points_w=points_w,
            colors_rgb=colors_rgb,
            frame_idx=frame.idx,
        )

        return new_pts.astype(np.float32), landmark_ids, points_w, colors_rgb

    def _bootstrap(self) -> None:
        frame0 = self.dataset.get_frame(0)

        self.state.current_pose_wc = np.eye(4, dtype=np.float64)
        self.state.last_keyframe_pose_wc = self.state.current_pose_wc.copy()
        self.state.last_keyframe_idx = 0

        new_pts, new_ids, points_w, colors_rgb = self._create_landmarks_from_keyframe(
            frame=frame0,
            existing_pts=None,
        )

        if len(new_ids) == 0:
            raise RuntimeError(
                "Bootstrap failed: no valid stereo landmarks were created from the first frame."
            )

        self.state.prev_gray = frame0.left_gray
        self.state.prev_pts = new_pts
        self.state.prev_ids = new_ids
        self.state.last_keyframe_feature_count = len(new_ids)

        self.trajectory_wc = [self.state.current_pose_wc.copy()]

        kf = Keyframe(
            frame_idx=frame0.idx,
            timestamp=frame0.timestamp,
            pose_wc=self.state.current_pose_wc.copy(),
            left_rgb=frame0.left_rgb.copy(),
            left_gray=frame0.left_gray.copy(),
            tracked_uv=new_pts.reshape(-1, 2).copy(),
            landmark_ids=new_ids.copy(),
            new_points_w=points_w.copy(),
            new_colors_rgb=colors_rgb.copy(),
        )
        self.map_manager.add_keyframe(kf)

        if self.local_ba_worker is not None:
            self.local_ba_worker.request_optimize()

    def _update_pose_from_pnp(self, R_cw: np.ndarray, t_cw: np.ndarray) -> None:
        T_cw = rt_to_pose(R_cw, t_cw)
        self.state.current_pose_wc = invert_pose(T_cw)

    def _run_pnp(self, tracked_pts: np.ndarray, tracked_ids: list[int]) -> bool:
        if len(tracked_ids) < self.cfg.pnp.min_points:
            return False

        world_pts = []
        image_pts = []
        used_ids = []

        with self.map_manager.lock:
            for p, lid in zip(tracked_pts.reshape(-1, 2), tracked_ids):
                lm = self.map_manager.landmarks.get(int(lid))
                if lm is None or not lm.active:
                    continue
                if lm.observations < self.cfg.quality.min_landmark_observations_for_pnp:
                    continue
                if lm.track_fail_count > self.cfg.quality.max_landmark_track_failures:
                    continue
                if lm.mean_reproj_error > self.cfg.quality.max_landmark_reproj_error and lm.observations < self.cfg.quality.protected_landmark_min_observations:
                    continue

                world_pts.append(lm.position_w)
                image_pts.append(p)
                used_ids.append(int(lid))

        if len(world_pts) < self.cfg.pnp.min_points:
            return False

        world_pts_np = np.asarray(world_pts, dtype=np.float64)
        image_pts_np = np.asarray(image_pts, dtype=np.float64)

        ok, R_cw, t_cw, inliers = solve_pnp_ransac(
            points_w=world_pts_np,
            image_points=image_pts_np,
            K=self.cam.K,
            min_points=self.cfg.pnp.min_points,
            reprojection_error=self.cfg.pnp.reprojection_error,
            confidence=self.cfg.pnp.confidence,
            iterations=self.cfg.pnp.iterations,
        )

        if not ok:
            return False

        self._update_pose_from_pnp(R_cw, t_cw)

        reproj_errors = self._compute_reprojection_errors(
            points_w=world_pts_np,
            image_pts=image_pts_np,
            R_cw=R_cw,
            t_cw=t_cw,
        )
        self.map_manager.record_reprojection_errors(
            landmark_ids=used_ids,
            errors=reproj_errors,
            alpha=self.cfg.quality.reproj_error_ema_alpha,
        )

        if inliers is not None and len(inliers) > 0:
            inlier_mask = np.zeros((len(used_ids),), dtype=bool)
            inlier_mask[inliers.reshape(-1)] = True

            outlier_ids = [lid for lid, keep in zip(used_ids, inlier_mask.tolist()) if not keep]
            if len(outlier_ids) > 0:
                self.map_manager.record_tracking_failures(outlier_ids)

        return True

    def _get_est_traj_xz(self) -> np.ndarray:
        if len(self.trajectory_wc) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        traj = np.stack(self.trajectory_wc, axis=0)
        return traj[:, [0, 2], 3]

    def _get_gt_traj_xz(self, upto_idx: int) -> np.ndarray | None:
        if self.dataset.gt is None:
            return None
        gt = self.dataset.gt[: upto_idx + 1]
        return gt[:, [0, 2], 3]

    def _get_landmark_xz(self, max_points: int = 10000) -> np.ndarray:
        return self.map_manager.get_active_landmarks_xz(max_points=max_points)

    def _show_live_view(self, frame, is_keyframe: bool) -> None:
        if self.viewer is None:
            return

        tracked_pts = (
            self.state.prev_pts.reshape(-1, 2)
            if self.state.prev_pts is not None and len(self.state.prev_pts) > 0
            else np.zeros((0, 2), dtype=np.float32)
        )

        est_traj_xz = self._get_est_traj_xz()
        gt_traj_xz = self._get_gt_traj_xz(frame.idx)
        landmark_xz = self._get_landmark_xz()

        backend_runs = self.local_ba_worker.run_count if self.local_ba_worker is not None else 0
        backend_successes = self.local_ba_worker.success_count if self.local_ba_worker is not None else 0

        key = self.viewer.show(
            image_rgb=frame.left_rgb,
            tracked_pts=tracked_pts,
            frame_idx=frame.idx,
            tracked_count=len(self.state.prev_ids) if self.state.prev_ids is not None else 0,
            landmark_count=self.map_manager.num_active_landmarks(),
            keyframe_count=len(self.map_manager.keyframes),
            is_keyframe=is_keyframe,
            mode=self.cfg.dataset.input_mode,
            fps=self.fps,
            est_traj_xz=est_traj_xz,
            gt_traj_xz=gt_traj_xz,
            landmark_xz=landmark_xz,
            backend_runs=backend_runs,
            backend_successes=backend_successes,
        )

        if key == ord("q"):
            raise KeyboardInterrupt("User requested quit.")

    def _process_tracking_frame(self, frame) -> None:
        self._update_fps()
        is_keyframe = False

        curr_pts, good = self.lk.track(
            prev_gray=self.state.prev_gray,
            curr_gray=frame.left_gray,
            prev_pts=self.state.prev_pts,
        )

        prev_ids = self.state.prev_ids if self.state.prev_ids is not None else []
        missed_ids = [lid for lid, keep in zip(prev_ids, good.tolist()) if not keep]
        if len(missed_ids) > 0:
            self.map_manager.record_tracking_failures(missed_ids)

        tracked_pts = curr_pts[good]
        tracked_ids = [lid for lid, keep in zip(prev_ids, good.tolist()) if keep]

        _ = self._run_pnp(tracked_pts, tracked_ids)

        self.map_manager.update_observations(tracked_ids, frame.idx)
        self.map_manager.prune_landmarks(
            current_frame_idx=frame.idx,
            stale_age=self.cfg.quality.stale_landmark_age,
            protected_min_obs=self.cfg.quality.protected_landmark_min_observations,
            max_track_failures=self.cfg.quality.max_landmark_track_failures,
            max_reproj_error=self.cfg.quality.max_landmark_reproj_error,
        )

        self.state.prev_gray = frame.left_gray
        self.state.prev_pts = tracked_pts.reshape(-1, 1, 2).astype(np.float32)
        self.state.prev_ids = tracked_ids

        denom = max(self.state.last_keyframe_feature_count, 1)
        tracked_ratio = float(len(tracked_ids)) / float(denom)

        want_keyframe = self.keyframe_manager.should_create(
            frame_idx=frame.idx,
            tracked_count=len(tracked_ids),
            tracked_ratio=tracked_ratio,
            current_pose_wc=self.state.current_pose_wc,
            last_keyframe_pose_wc=self.state.last_keyframe_pose_wc,
            last_keyframe_idx=self.state.last_keyframe_idx,
        )

        if want_keyframe:
            new_pts, new_ids, points_w, colors_rgb = self._create_landmarks_from_keyframe(
                frame=frame,
                existing_pts=self.state.prev_pts,
            )

            force_due_to_degradation = (
                len(tracked_ids) < self.cfg.keyframe.min_tracked_landmarks
                or tracked_ratio < self.cfg.keyframe.low_track_ratio_thresh
                or (frame.idx - self.state.last_keyframe_idx) >= self.cfg.keyframe.max_frame_gap
            )

            if force_due_to_degradation or len(new_ids) >= self.cfg.keyframe.min_new_landmarks:
                is_keyframe = True

                if len(new_pts) > 0:
                    if self.state.prev_pts is None or len(self.state.prev_pts) == 0:
                        self.state.prev_pts = new_pts
                        self.state.prev_ids = new_ids
                    else:
                        self.state.prev_pts = np.vstack([self.state.prev_pts, new_pts]).astype(np.float32)
                        self.state.prev_ids = self.state.prev_ids + new_ids

                self.state.last_keyframe_idx = frame.idx
                self.state.last_keyframe_pose_wc = self.state.current_pose_wc.copy()
                self.state.last_keyframe_feature_count = len(self.state.prev_ids)

                tracked_uv = (
                    self.state.prev_pts.reshape(-1, 2).copy()
                    if self.state.prev_pts is not None and len(self.state.prev_pts) > 0
                    else np.zeros((0, 2), dtype=np.float32)
                )

                kf = Keyframe(
                    frame_idx=frame.idx,
                    timestamp=frame.timestamp,
                    pose_wc=self.state.current_pose_wc.copy(),
                    left_rgb=frame.left_rgb.copy(),
                    left_gray=frame.left_gray.copy(),
                    tracked_uv=tracked_uv,
                    landmark_ids=self.state.prev_ids.copy(),
                    new_points_w=points_w.copy(),
                    new_colors_rgb=colors_rgb.copy(),
                )
                self.map_manager.add_keyframe(kf)

                if self.local_ba_worker is not None:
                    self.local_ba_worker.request_optimize()

        self.trajectory_wc.append(self.state.current_pose_wc.copy())
        self._show_live_view(frame, is_keyframe=is_keyframe)

    def run(self) -> VOResult:
        self._bootstrap()

        num_frames = len(self.dataset)
        if self.cfg.tracker.max_frames is not None:
            num_frames = min(num_frames, self.cfg.tracker.max_frames)

        try:
            for idx in range(1, num_frames):
                frame = self.dataset.get_frame(idx)
                self._process_tracking_frame(frame)
        finally:
            if self.local_ba_worker is not None:
                self.local_ba_worker.stop()
            if self.viewer is not None:
                self.viewer.close()

        if self.local_ba_worker is not None:
            print(f"Local BA runs: {self.local_ba_worker.run_count}")
            print(f"Local BA successes: {self.local_ba_worker.success_count}")
            print(f"Local BA drops: {self.local_ba_worker.drop_count}")

        return VOResult(
            trajectory_wc=np.stack(self.trajectory_wc, axis=0),
            keyframes=self.map_manager.keyframes,
            landmarks=self.map_manager.landmarks,
        )