from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class TrajectoryErrorResult:
    num_frames: int
    rmse_translation: float
    mean_translation: float
    median_translation: float
    max_translation: float
    rmse_x: float
    rmse_y: float
    rmse_z: float
    mean_rotation_deg: float
    median_rotation_deg: float
    max_rotation_deg: float


def load_kitti_poses(path: str) -> np.ndarray:
    """
    Load KITTI-format poses:
    each line = 12 numbers representing a 3x4 matrix.
    Returns array of shape (N, 4, 4).
    """
    poses = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            vals = line.strip().split()
            if not vals:
                continue
            if len(vals) != 12:
                raise ValueError(
                    f"Line {line_idx + 1} in {path} does not have 12 values."
                )
            arr = np.asarray([float(x) for x in vals], dtype=np.float64).reshape(3, 4)
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = arr
            poses.append(T)

    if len(poses) == 0:
        raise ValueError(f"No poses loaded from {path}")

    return np.stack(poses, axis=0)


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Angular difference between two rotation matrices in degrees.
    """
    R_err = R_gt.T @ R_est
    trace_val = np.trace(R_err)
    cos_theta = np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))


def evaluate_absolute_errors(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
) -> tuple[TrajectoryErrorResult, dict[str, np.ndarray]]:
    """
    Frame-by-frame absolute pose comparison without SE(3)/Sim(3) alignment.
    Assumes both trajectories are already in the same world frame convention.
    """
    n = min(len(est_poses), len(gt_poses))
    est = est_poses[:n]
    gt = gt_poses[:n]

    t_est = est[:, :3, 3]
    t_gt = gt[:, :3, 3]

    diff = t_est - t_gt
    trans_err = np.linalg.norm(diff, axis=1)

    rmse_translation = float(np.sqrt(np.mean(trans_err ** 2)))
    mean_translation = float(np.mean(trans_err))
    median_translation = float(np.median(trans_err))
    max_translation = float(np.max(trans_err))

    rmse_x = float(np.sqrt(np.mean(diff[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(diff[:, 1] ** 2)))
    rmse_z = float(np.sqrt(np.mean(diff[:, 2] ** 2)))

    rot_err_deg = np.asarray(
        [
            rotation_error_deg(est[i, :3, :3], gt[i, :3, :3])
            for i in range(n)
        ],
        dtype=np.float64,
    )

    result = TrajectoryErrorResult(
        num_frames=n,
        rmse_translation=rmse_translation,
        mean_translation=mean_translation,
        median_translation=median_translation,
        max_translation=max_translation,
        rmse_x=rmse_x,
        rmse_y=rmse_y,
        rmse_z=rmse_z,
        mean_rotation_deg=float(np.mean(rot_err_deg)),
        median_rotation_deg=float(np.median(rot_err_deg)),
        max_rotation_deg=float(np.max(rot_err_deg)),
    )

    series = {
        "translation_error": trans_err,
        "rotation_error_deg": rot_err_deg,
        "est_xyz": t_est,
        "gt_xyz": t_gt,
    }
    return result, series


def print_result(result: TrajectoryErrorResult) -> None:
    print("========== Trajectory Error Report ==========")
    print(f"Frames compared:           {result.num_frames}")
    print(f"Translation RMSE (m):      {result.rmse_translation:.6f}")
    print(f"Translation mean (m):      {result.mean_translation:.6f}")
    print(f"Translation median (m):    {result.median_translation:.6f}")
    print(f"Translation max (m):       {result.max_translation:.6f}")
    print()
    print(f"RMSE X (m):                {result.rmse_x:.6f}")
    print(f"RMSE Y (m):                {result.rmse_y:.6f}")
    print(f"RMSE Z (m):                {result.rmse_z:.6f}")
    print()
    print(f"Rotation mean (deg):       {result.mean_rotation_deg:.6f}")
    print(f"Rotation median (deg):     {result.median_rotation_deg:.6f}")
    print(f"Rotation max (deg):        {result.max_rotation_deg:.6f}")


def plot_topdown(est_xyz: np.ndarray, gt_xyz: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="Ground Truth")
    plt.plot(est_xyz[:, 0], est_xyz[:, 2], label="Estimated")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_curves(translation_error: np.ndarray, rotation_error_deg: np.ndarray) -> None:
    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(translation_error)
    ax1.set_title("Translation Error per Frame")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Error (m)")
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(rotation_error_deg)
    ax2.set_title("Rotation Error per Frame")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Error (deg)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate estimated trajectory against KITTI GT")
    parser.add_argument(
        "--est",
        type=str,
        required=True,
        help="Path to estimated trajectory txt (KITTI 3x4 per line).",
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to ground-truth trajectory txt (KITTI 3x4 per line).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show top-down and per-frame error plots.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.est):
        raise FileNotFoundError(f"Estimated trajectory file not found: {args.est}")
    if not os.path.isfile(args.gt):
        raise FileNotFoundError(f"Ground-truth trajectory file not found: {args.gt}")

    est_poses = load_kitti_poses(args.est)
    gt_poses = load_kitti_poses(args.gt)

    result, series = evaluate_absolute_errors(est_poses, gt_poses)
    print_result(result)

    if args.plot:
        plot_topdown(series["est_xyz"], series["gt_xyz"], title="Estimated vs Ground Truth Trajectory")
        plot_error_curves(
            translation_error=series["translation_error"],
            rotation_error_deg=series["rotation_error_deg"],
        )


if __name__ == "__main__":
    main()