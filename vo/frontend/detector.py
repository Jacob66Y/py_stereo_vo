from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class GFTTDetector:
    def __init__(self, max_corners: int, quality_level: float, min_distance: int, block_size: int) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

    def detect(self, image_gray: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(
            image_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            mask=mask,
        )
        if pts is None:
            return np.zeros((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32)
