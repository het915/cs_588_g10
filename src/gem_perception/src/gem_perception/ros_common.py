"""ROS-agnostic helpers used by both ROS1 and ROS2 wrappers.

This module deliberately avoids importing rospy / rclpy — it only uses numpy
and standard library so both framework-specific wrappers can reuse it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def K_from_camera_info(K_list) -> np.ndarray:
    """Convert CameraInfo.K (length-9 row-major) to (3,3) numpy array."""
    return np.asarray(K_list, dtype=float).reshape(3, 3)


def D_from_camera_info(D_list) -> Optional[np.ndarray]:
    """Convert CameraInfo.D to a 1-D numpy array, or ``None`` if not populated.

    plumb_bob layout: ``[k1, k2, p1, p2, k3]`` (length 5). Returns ``None``
    when D is empty or all-zero so callers can skip the distortion code path.
    """
    if D_list is None:
        return None
    arr = np.asarray(D_list, dtype=float).reshape(-1)
    if arr.size == 0 or not np.any(np.abs(arr) > 1e-6):
        return None
    return arr


def transform_to_matrix(translation: Tuple[float, float, float],
                        rotation_xyzw: Tuple[float, float, float, float]) -> np.ndarray:
    """(x,y,z) + (qx,qy,qz,qw) → 4x4 homogeneous transform."""
    qx, qy, qz, qw = rotation_xyzw
    x, y, z = translation
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def matrix_inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def draw_detection_overlay(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    mask: Optional[np.ndarray],
    pixel_centroid: Tuple[float, float],
    is_estimated: bool,
    prompt: str,
    score: float,
) -> np.ndarray:
    out = image_bgr.copy()
    color_real = (0, 255, 0)
    color_est = (0, 255, 255)
    color = color_est if is_estimated else color_real

    if mask is not None:
        overlay = out.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(color)).astype(np.uint8)
        out = overlay

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    cx, cy = int(round(pixel_centroid[0])), int(round(pixel_centroid[1]))
    cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)

    label = f"{prompt} ({score:.2f}){' [est]' if is_estimated else ''}"
    cv2.putText(out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    return out


def draw_lidar_projection(
    image_bgr: np.ndarray,
    uv: np.ndarray,
    depths: np.ndarray,
    max_depth: float = 30.0,
) -> np.ndarray:
    out = image_bgr.copy()
    if uv.size == 0:
        return out
    H, W = out.shape[:2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    within = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depths > 0.1)
    u, v, d = u[within], v[within], depths[within]
    # Colorize: near = red, far = green/blue
    d_norm = np.clip(d / max_depth, 0, 1)
    colors = np.zeros((len(d_norm), 3), dtype=np.uint8)
    colors[:, 2] = ((1.0 - d_norm) * 255).astype(np.uint8)   # B (BGR → "far = red" visually? swap to taste)
    colors[:, 1] = (d_norm * 255).astype(np.uint8)
    for i in range(len(u)):
        cv2.circle(out, (u[i], v[i]), 2, tuple(int(c) for c in colors[i]), -1)
    return out

class GoalHold:
    """Collects detections for the first `collect_seconds` seconds, then
    latches the median position permanently.  Robust to early outliers.

    Parameters
    ----------
    collect_seconds : float
        How long to gather detections before latching (default 3.0 s).
        Increase if the first few detections are noisy.
    min_samples : int
        Minimum number of detections needed to latch.  If fewer arrive
        within collect_seconds the window extends until this is met.
    accept_estimated : bool
        If True (default), also accept is_estimated=True detections
        (image-only, no LiDAR confirmation).  Set False to require LiDAR.
    """

    def __init__(
        self,
        hold_seconds: float = 2.0,   # kept for API compatibility, unused
        collect_seconds: float = 3.0,
        min_samples: int = 5,
        accept_estimated: bool = True,
    ):
        self.collect_seconds = collect_seconds
        self.min_samples = min_samples
        self.accept_estimated = accept_estimated

        self._collecting: bool = True          # True = still gathering
        self._collect_start: Optional[float] = None
        self._samples: list = []               # list of np.ndarray (3,)
        self._latched: Optional[Tuple[np.ndarray, bool]] = None

    def update(
        self,
        now: float,
        goal_base: Optional[np.ndarray],
        is_estimated: bool,
    ) -> Optional[Tuple[np.ndarray, bool]]:

        # Already latched — return forever, ignore new detections
        if self._latched is not None:
            return self._latched

        # No detection this frame — nothing to add
        if goal_base is None:
            return None

        # Skip estimated-only detections if not accepted
        if is_estimated and not self.accept_estimated:
            return None

        # Start the collection clock on first detection
        if self._collect_start is None:
            self._collect_start = now

        self._samples.append(goal_base.copy().astype(float))

        # Check if collection window is done
        elapsed = now - self._collect_start
        enough_time = elapsed >= self.collect_seconds
        enough_samples = len(self._samples) >= self.min_samples

        if enough_time and enough_samples:
            pts = np.array(self._samples)          # (N, 3)
            median_pos = np.median(pts, axis=0)    # median per axis — outlier robust
            self._latched = (median_pos, is_estimated)
            self._collecting = False
            return self._latched

        # Still collecting — don't publish yet
        return None

    def reset(self):
        """Clear the latch to start a new collection."""
        self._collecting = True
        self._collect_start = None
        self._samples = []
        self._latched = None