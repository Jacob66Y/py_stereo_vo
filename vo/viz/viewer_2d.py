from __future__ import annotations

import cv2
import numpy as np


class Viewer2D:
    def __init__(
        self,
        window_name: str = "Stereo VO Debug",
        traj_window_name: str = "Trajectory vs GT",
        wait_ms: int = 1,
        traj_size: int = 500,
        map_size: int = 500,
        show_separate_traj_window: bool = True,
    ) -> None:
        self.window_name = window_name
        self.traj_window_name = traj_window_name
        self.wait_ms = wait_ms
        self.traj_size = traj_size
        self.map_size = map_size
        self.show_separate_traj_window = show_separate_traj_window

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.show_separate_traj_window:
            cv2.namedWindow(self.traj_window_name, cv2.WINDOW_NORMAL)

    def _draw_points(
        self,
        image_bgr: np.ndarray,
        pts: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        radius: int = 2,
    ) -> np.ndarray:
        vis = image_bgr.copy()
        if pts is None or len(pts) == 0:
            return vis

        pts = pts.reshape(-1, 2)
        for p in pts:
            x = int(round(p[0]))
            y = int(round(p[1]))
            cv2.circle(vis, (x, y), radius, color, -1)
        return vis

    def _draw_text_block(
        self,
        image_bgr: np.ndarray,
        lines: list[str],
        org: tuple[int, int] = (10, 20),
        color: tuple[int, int, int] = (0, 255, 255),
    ) -> np.ndarray:
        vis = image_bgr.copy()
        x, y = org
        for i, line in enumerate(lines):
            yy = y + i * 22
            cv2.putText(
                vis,
                line,
                (x, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return vis

    def _make_topdown_canvas(
        self,
        size: int,
        est_xy: np.ndarray | None,
        gt_xy: np.ndarray | None,
        points_xy: np.ndarray | None,
        title: str,
        est_color: tuple[int, int, int],
        gt_color: tuple[int, int, int] = (255, 255, 255),
        point_color: tuple[int, int, int] = (255, 0, 0),
        current_color: tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)

        all_pts = []
        if est_xy is not None and len(est_xy) > 0:
            all_pts.append(est_xy)
        if gt_xy is not None and len(gt_xy) > 0:
            all_pts.append(gt_xy)
        if points_xy is not None and len(points_xy) > 0:
            all_pts.append(points_xy)

        if len(all_pts) == 0:
            cv2.putText(
                canvas,
                title,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return canvas

        merged = np.vstack(all_pts)
        min_xy = merged.min(axis=0)
        max_xy = merged.max(axis=0)

        span = np.maximum(max_xy - min_xy, 1e-6)
        scale = 0.9 * (size - 40) / max(span[0], span[1])

        def to_canvas(xy: np.ndarray) -> np.ndarray:
            pts = (xy - min_xy) * scale + 20.0
            pts[:, 1] = (size - 1) - pts[:, 1]
            return pts.astype(np.int32)

        if points_xy is not None and len(points_xy) > 0:
            pts = to_canvas(points_xy)
            for p in pts:
                cv2.circle(canvas, tuple(p), 1, point_color, -1)

        if gt_xy is not None and len(gt_xy) > 1:
            gt_pts = to_canvas(gt_xy)
            for i in range(1, len(gt_pts)):
                cv2.line(canvas, tuple(gt_pts[i - 1]), tuple(gt_pts[i]), gt_color, 1)

        if est_xy is not None and len(est_xy) > 1:
            est_pts = to_canvas(est_xy)
            for i in range(1, len(est_pts)):
                cv2.line(canvas, tuple(est_pts[i - 1]), tuple(est_pts[i]), est_color, 2)
            cv2.circle(canvas, tuple(est_pts[-1]), 4, current_color, -1)

        cv2.putText(
            canvas,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def show(
        self,
        image_rgb: np.ndarray,
        tracked_pts: np.ndarray,
        frame_idx: int,
        tracked_count: int,
        landmark_count: int,
        keyframe_count: int,
        is_keyframe: bool,
        mode: str,
        fps: float,
        est_traj_xz: np.ndarray | None = None,
        gt_traj_xz: np.ndarray | None = None,
        landmark_xz: np.ndarray | None = None,
        backend_runs: int = 0,
        backend_successes: int = 0,
    ) -> int:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        left_panel = self._draw_points(image_bgr, tracked_pts, color=(0, 255, 0), radius=2)

        lines = [
            f"Frame: {frame_idx}",
            f"FPS: {fps:.2f}",
            f"Tracked: {tracked_count}",
            f"Landmarks: {landmark_count}",
            f"Keyframes: {keyframe_count}",
            f"Backend runs: {backend_runs}",
            f"Backend ok: {backend_successes}",
            f"Mode: {mode}",
            f"Keyframe: {'YES' if is_keyframe else 'NO'}",
            "Press q to quit",
        ]
        left_panel = self._draw_text_block(left_panel, lines)

        traj_panel = self._make_topdown_canvas(
            size=self.traj_size,
            est_xy=est_traj_xz,
            gt_xy=gt_traj_xz,
            points_xy=None,
            title="Trajectory (X-Z)",
            est_color=(0, 255, 0),
            gt_color=(255, 255, 255),
            current_color=(0, 0, 255),
        )

        map_panel = self._make_topdown_canvas(
            size=self.map_size,
            est_xy=est_traj_xz,
            gt_xy=None,
            points_xy=landmark_xz,
            title="Landmarks (X-Z)",
            est_color=(0, 255, 0),
            point_color=(255, 200, 0),
            current_color=(0, 0, 255),
        )

        right_width = max(traj_panel.shape[1], map_panel.shape[1])
        traj_panel = cv2.copyMakeBorder(
            traj_panel, 0, 0, 0, right_width - traj_panel.shape[1],
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        map_panel = cv2.copyMakeBorder(
            map_panel, 0, 0, 0, right_width - map_panel.shape[1],
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        right_panel = np.vstack([traj_panel, map_panel])

        target_left_height = right_panel.shape[0]
        scale = target_left_height / left_panel.shape[0]
        target_left_width = int(left_panel.shape[1] * scale)
        left_panel_resized = cv2.resize(left_panel, (target_left_width, target_left_height))

        canvas = np.hstack([left_panel_resized, right_panel])

        cv2.imshow(self.window_name, canvas)
        if self.show_separate_traj_window:
            cv2.imshow(self.traj_window_name, traj_panel)

        key = cv2.waitKey(self.wait_ms) & 0xFF
        return key

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        if self.show_separate_traj_window:
            try:
                cv2.destroyWindow(self.traj_window_name)
            except Exception:
                pass