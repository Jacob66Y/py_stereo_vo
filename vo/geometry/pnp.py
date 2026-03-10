from __future__ import annotations

import cv2
import numpy as np


def solve_pnp_ransac(
    points_w: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    min_points: int,
    reprojection_error: float,
    confidence: float,
    iterations: int,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray | None]:
    if len(points_w) < min_points or len(image_points) < min_points:
        return False, np.eye(3), np.zeros((3, 1)), None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=points_w.astype(np.float64),
        imagePoints=image_points.astype(np.float64),
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=reprojection_error,
        confidence=confidence,
        iterationsCount=iterations,
    )

    if not ok:
        return False, np.eye(3), np.zeros((3, 1)), None

    R_cw, _ = cv2.Rodrigues(rvec)
    return True, R_cw, tvec, inliers
