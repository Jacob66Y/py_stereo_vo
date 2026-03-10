from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from vo.types import Landmark


def plot_trajectory_2d(trajectory: np.ndarray, gt: np.ndarray | None = None) -> None:
    plt.figure(figsize=(8, 8))

    if gt is not None:
        plt.plot(gt[:, 0, 3], gt[:, 2, 3], label="Ground Truth", linewidth=2)

    plt.plot(trajectory[:, 0, 3], trajectory[:, 2, 3], label="Estimated", linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Top-down Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_landmarks_topdown(landmarks: dict[int, Landmark], max_points: int = 8000) -> None:
    pts = []
    for lm in landmarks.values():
        if lm.active:
            pts.append(lm.position_w)

    if len(pts) == 0:
        return

    pts = np.asarray(pts)
    if len(pts) > max_points:
        pts = pts[:max_points]

    plt.figure(figsize=(8, 8))
    plt.scatter(pts[:, 0], pts[:, 2], s=1)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Sparse Landmark Map")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
