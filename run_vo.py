from __future__ import annotations

import argparse
import os

from vo.config import load_config
from vo.dataset.kitti import KITTIStereoDataset
from vo.frontend.tracker import StereoVOTracker
from vo.io.exporters import export_keyframes_npz, export_trajectory_txt
from vo.viz.plotter import plot_trajectory_2d, plot_landmarks_topdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic stereo VO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kitti_rgb.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset = KITTIStereoDataset(
        root=cfg.dataset.root,
        sequence=cfg.dataset.sequence,
        input_mode=cfg.dataset.input_mode,
    )

    print("System Start")

    tracker = StereoVOTracker(dataset=dataset, cfg=cfg)
    result = tracker.run()

    os.makedirs(cfg.output.save_dir, exist_ok=True)

    if cfg.output.save_trajectory:
        export_trajectory_txt(
            trajectory=result.trajectory_wc,
            out_path=os.path.join(
                cfg.output.save_dir,
                f"trajectory_{cfg.dataset.sequence}_{cfg.dataset.input_mode}.txt",
            ),
        )

    if cfg.output.save_keyframes:
        export_keyframes_npz(
            keyframes=result.keyframes,
            out_dir=os.path.join(
                cfg.output.save_dir,
                f"keyframes_{cfg.dataset.sequence}_{cfg.dataset.input_mode}",
            ),
        )

    print(f"Processed frames: {len(result.trajectory_wc)}")
    print(f"Keyframes: {len(result.keyframes)}")
    print(f"Landmarks: {len(result.landmarks)}")

    if cfg.output.show_plot:
        plot_trajectory_2d(result.trajectory_wc, dataset.gt)
        plot_landmarks_topdown(result.landmarks)

if __name__ == "__main__":
    main()
