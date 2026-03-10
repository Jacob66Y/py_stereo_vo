import cv2
import numpy as np


def draw_points(image: np.ndarray, pts: np.ndarray, color=(0, 255, 0), radius: int = 2) -> np.ndarray:
    vis = image.copy()
    if pts is None or len(pts) == 0:
        return vis

    pts = pts.reshape(-1, 2)
    for p in pts:
        x = int(round(p[0]))
        y = int(round(p[1]))
        cv2.circle(vis, (x, y), radius, color, -1)
    return vis


def draw_text_block(image: np.ndarray, lines: list[str], org=(10, 20)) -> np.ndarray:
    vis = image.copy()
    x, y = org
    for i, line in enumerate(lines):
        yy = y + i * 22
        cv2.putText(
            vis,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis
