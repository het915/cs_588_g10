"""Arc-length parameterized 2D reference path."""
import numpy as np


class ReferencePath:
    def __init__(self, waypoints):
        wp = np.asarray(waypoints, dtype=float)
        if wp.ndim != 2 or wp.shape[1] != 2 or wp.shape[0] < 2:
            raise ValueError("waypoints must be (N>=2, 2)")
        self.xy = wp
        diffs = np.diff(wp, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        self.headings = np.concatenate([self.headings, self.headings[-1:]])
        self.total_length = self.s[-1]

    def nearest_point(self, pos):
        """Return (point_xy, s, heading, signed_lateral_error d) for pos=(x,y).

        Uses projection onto each segment and picks the closest.
        """
        p = np.asarray(pos, dtype=float)
        a = self.xy[:-1]
        b = self.xy[1:]
        ab = b - a
        ap = p - a
        ab_len2 = np.sum(ab * ab, axis=1)
        ab_len2 = np.where(ab_len2 < 1e-12, 1e-12, ab_len2)
        t = np.clip(np.sum(ap * ab, axis=1) / ab_len2, 0.0, 1.0)
        proj = a + (t[:, None] * ab)
        d2 = np.sum((proj - p) ** 2, axis=1)
        i = int(np.argmin(d2))

        point = proj[i]
        seg_s = self.s[i] + t[i] * np.linalg.norm(ab[i])
        heading = np.arctan2(ab[i, 1], ab[i, 0])
        # signed lateral error: left of path positive
        nx, ny = -np.sin(heading), np.cos(heading)
        d = (p[0] - point[0]) * nx + (p[1] - point[1]) * ny
        return point, seg_s, heading, d

    def trim_behind(self, pos, min_points=4):
        """Return a new ReferencePath with already-passed waypoints removed.

        Finds the segment nearest to `pos`, then returns the path starting
        from that segment.  Always keeps at least `min_points` waypoints so
        the returned path is valid and the MPPI has meaningful look-ahead.
        """
        _, _, _, _ = self.nearest_point(pos)   # warm path; reuse index logic
        p = np.asarray(pos, dtype=float)
        a = self.xy[:-1]
        b = self.xy[1:]
        ab = b - a
        ap = p - a
        ab_len2 = np.sum(ab * ab, axis=1)
        ab_len2 = np.where(ab_len2 < 1e-12, 1e-12, ab_len2)
        t = np.clip(np.sum(ap * ab, axis=1) / ab_len2, 0.0, 1.0)
        proj = a + (t[:, None] * ab)
        d2 = np.sum((proj - p) ** 2, axis=1)
        nearest_seg = int(np.argmin(d2))
        # Keep from `nearest_seg` onward; clamp so we always have min_points.
        start = min(nearest_seg, max(0, len(self.xy) - min_points))
        return ReferencePath(self.xy[start:])
