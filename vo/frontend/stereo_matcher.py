from __future__ import annotations

import cv2
import numpy as np


class StereoSGBMMatcher:
    def __init__(self, min_disparity: int, num_disparities: int, block_size: int) -> None:
        if num_disparities % 16 != 0:
            raise ValueError("num_disparities must be divisible by 16 for OpenCV StereoSGBM")

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        disparity = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity
