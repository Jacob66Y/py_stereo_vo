from __future__ import annotations

import numpy as np


def make_feature_mask(
    image_shape: tuple[int, int],
    existing_pts: np.ndarray | None,
    radius: int,
) -> np.ndarray:
    import cv2

    mask = np.full(image_shape, 255, dtype=np.uint8)
    if existing_pts is None or len(existing_pts) == 0:
        return mask

    for p in existing_pts.reshape(-1, 2):
        x = int(round(p[0]))
        y = int(round(p[1]))
        cv2.circle(mask, (x, y), radius, 0, -1)
    return mask
