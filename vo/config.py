from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import yaml


@dataclass
class DatasetConfig:
    root: str
    sequence: str
    input_mode: str


@dataclass
class TrackerConfig:
    max_frames: Optional[int]
    max_corners: int
    quality_level: float
    min_distance: int
    block_size: int
    lk_win_size: int
    lk_max_level: int
    lk_max_error: float


@dataclass
class StereoConfig:
    min_disparity: int
    num_disparities: int
    block_size: int
    min_depth: float
    max_depth: float
    min_valid_disparity: float


@dataclass
class PnPConfig:
    min_points: int
    reprojection_error: float
    confidence: float
    iterations: int


@dataclass
class KeyframeConfig:
    min_tracked_landmarks: int
    translation_thresh: float
    rotation_thresh_deg: float
    frame_gap: int
    max_frame_gap: int
    low_track_ratio_thresh: float
    new_feature_mask_radius: int
    max_new_landmarks: int
    min_new_landmarks: int


@dataclass
class QualityConfig:
    min_landmark_observations_for_pnp: int
    protected_landmark_min_observations: int
    max_landmark_track_failures: int
    max_landmark_reproj_error: float
    stale_landmark_age: int
    reproj_error_ema_alpha: float


@dataclass
class BackendConfig:
    enable_local_ba: bool
    local_ba_window_size: int
    local_ba_max_nfev: int
    local_ba_huber_delta: float
    local_ba_min_observations: int
    local_ba_verbose: bool
    local_ba_max_version_lag: int


@dataclass
class OutputConfig:
    save_dir: str
    save_keyframes: bool
    save_trajectory: bool
    show_plot: bool
    live_view: bool
    show_separate_traj_window: bool


@dataclass
class VOConfig:
    dataset: DatasetConfig
    tracker: TrackerConfig
    stereo: StereoConfig
    pnp: PnPConfig
    keyframe: KeyframeConfig
    quality: QualityConfig
    backend: BackendConfig
    output: OutputConfig


def load_config(path: str) -> VOConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    dataset = DatasetConfig(
        root=os.path.expanduser(raw["dataset"]["root"]),
        sequence=str(raw["dataset"]["sequence"]),
        input_mode=str(raw["dataset"]["input_mode"]),
    )

    tracker = TrackerConfig(
        max_frames=raw["tracker"]["max_frames"],
        max_corners=int(raw["tracker"]["max_corners"]),
        quality_level=float(raw["tracker"]["quality_level"]),
        min_distance=int(raw["tracker"]["min_distance"]),
        block_size=int(raw["tracker"]["block_size"]),
        lk_win_size=int(raw["tracker"]["lk_win_size"]),
        lk_max_level=int(raw["tracker"]["lk_max_level"]),
        lk_max_error=float(raw["tracker"]["lk_max_error"]),
    )

    stereo = StereoConfig(
        min_disparity=int(raw["stereo"]["min_disparity"]),
        num_disparities=int(raw["stereo"]["num_disparities"]),
        block_size=int(raw["stereo"]["block_size"]),
        min_depth=float(raw["stereo"]["min_depth"]),
        max_depth=float(raw["stereo"]["max_depth"]),
        min_valid_disparity=float(raw["stereo"]["min_valid_disparity"]),
    )

    pnp = PnPConfig(
        min_points=int(raw["pnp"]["min_points"]),
        reprojection_error=float(raw["pnp"]["reprojection_error"]),
        confidence=float(raw["pnp"]["confidence"]),
        iterations=int(raw["pnp"]["iterations"]),
    )

    keyframe_raw = raw["keyframe"]
    keyframe = KeyframeConfig(
        min_tracked_landmarks=int(keyframe_raw["min_tracked_landmarks"]),
        translation_thresh=float(keyframe_raw["translation_thresh"]),
        rotation_thresh_deg=float(keyframe_raw["rotation_thresh_deg"]),
        frame_gap=int(keyframe_raw["frame_gap"]),
        max_frame_gap=int(keyframe_raw.get("max_frame_gap", 12)),
        low_track_ratio_thresh=float(keyframe_raw.get("low_track_ratio_thresh", 0.45)),
        new_feature_mask_radius=int(keyframe_raw["new_feature_mask_radius"]),
        max_new_landmarks=int(keyframe_raw["max_new_landmarks"]),
        min_new_landmarks=int(keyframe_raw.get("min_new_landmarks", 40)),
    )

    quality_raw = raw.get("quality", {})
    quality = QualityConfig(
        min_landmark_observations_for_pnp=int(quality_raw.get("min_landmark_observations_for_pnp", 2)),
        protected_landmark_min_observations=int(quality_raw.get("protected_landmark_min_observations", 4)),
        max_landmark_track_failures=int(quality_raw.get("max_landmark_track_failures", 4)),
        max_landmark_reproj_error=float(quality_raw.get("max_landmark_reproj_error", 6.0)),
        stale_landmark_age=int(quality_raw.get("stale_landmark_age", 20)),
        reproj_error_ema_alpha=float(quality_raw.get("reproj_error_ema_alpha", 0.2)),
    )

    backend_raw = raw.get("backend", {})
    backend = BackendConfig(
        enable_local_ba=bool(backend_raw.get("enable_local_ba", True)),
        local_ba_window_size=int(backend_raw.get("local_ba_window_size", 5)),
        local_ba_max_nfev=int(backend_raw.get("local_ba_max_nfev", 20)),
        local_ba_huber_delta=float(backend_raw.get("local_ba_huber_delta", 5.0)),
        local_ba_min_observations=int(backend_raw.get("local_ba_min_observations", 20)),
        local_ba_verbose=bool(backend_raw.get("local_ba_verbose", False)),
        local_ba_max_version_lag=int(backend_raw.get("local_ba_max_version_lag", 80)),
    )

    output_raw = raw["output"]
    output = OutputConfig(
        save_dir=str(output_raw["save_dir"]),
        save_keyframes=bool(output_raw["save_keyframes"]),
        save_trajectory=bool(output_raw["save_trajectory"]),
        show_plot=bool(output_raw["show_plot"]),
        live_view=bool(output_raw["live_view"]),
        show_separate_traj_window=bool(output_raw.get("show_separate_traj_window", True)),
    )

    return VOConfig(
        dataset=dataset,
        tracker=tracker,
        stereo=stereo,
        pnp=pnp,
        keyframe=keyframe,
        quality=quality,
        backend=backend,
        output=output,
    )