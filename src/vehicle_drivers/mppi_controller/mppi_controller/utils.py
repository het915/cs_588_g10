"""Utility helpers for the MPPI controller package.

Pure-Python / NumPy only — no ROS, no torch.
"""
import csv
import math
import os

import numpy as np

from ament_index_python.packages import get_package_share_directory

from .reference_path import ReferencePath

# --- WGS-84 geodetic -> ENU (vendored; avoids a pymap3d install) -----------
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _geodetic_to_ecef(lat_deg, lon_deg, h):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sl * sl)
    return (
        (N + h) * cl * math.cos(lon),
        (N + h) * cl * math.sin(lon),
        (N * (1.0 - _WGS84_E2) + h) * sl,
    )


def geodetic2enu(lat, lon, h, lat0, lon0, h0):
    """WGS-84 geodetic -> local ENU (metres). Equivalent to pymap3d.geodetic2enu."""
    x,  y,  z  = _geodetic_to_ecef(lat,  lon,  h)
    x0, y0, z0 = _geodetic_to_ecef(lat0, lon0, h0)
    dx, dy, dz = x - x0, y - y0, z - z0
    slat, clat = math.sin(math.radians(lat0)), math.cos(math.radians(lat0))
    slon, clon = math.sin(math.radians(lon0)), math.cos(math.radians(lon0))
    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    u =  clat * clon * dx + clat * slon * dy + slat * dz
    return e, n, u


# --- Vehicle geometry -------------------------------------------------------

def heading_to_yaw(heading_deg):
    """Compass heading (0=N, CW+, degrees) -> ENU yaw (rad, 0=+x, CCW+)."""
    if heading_deg < 270.0:
        return math.radians(90.0 - heading_deg)
    return math.radians(450.0 - heading_deg)


def front2steer(f_angle_deg):
    """Front-wheel angle (deg) -> steering-wheel angle (deg), GEM e4 calibration."""
    a = max(min(f_angle_deg, 35.0), -35.0)
    mag = abs(a)
    sw = -0.1084 * mag * mag + 21.775 * mag
    sw = sw if a >= 0 else -sw
    return max(min(sw, 450.0), -450.0)


# --- Signal processing ------------------------------------------------------

class PID:
    def __init__(self, kp, ki, kd, wg=None):
        self.kp, self.ki, self.kd, self.wg = kp, ki, kd, wg
        self.iterm = 0.0
        self.last_e = 0.0
        self.last_t = None

    def reset(self):
        self.iterm = 0.0
        self.last_e = 0.0
        self.last_t = None

    def get_control(self, t, e):
        if self.last_t is None:
            dt, de = 0.0, 0.0
        else:
            dt = t - self.last_t
            de = (e - self.last_e) / dt if dt > 0.0 else 0.0
        self.iterm += e * dt
        if self.wg is not None:
            self.iterm = max(min(self.iterm, self.wg), -self.wg)
        self.last_e = e
        self.last_t = t
        return self.kp * e + self.ki * self.iterm + self.kd * de


class OnlineFilter:
    """Exponential moving average (1st-order low-pass).

    Equivalent damping to a 1st-order low-pass at `cutoff` Hz sampled at
    `fs` Hz. `order` accepted for API compatibility but unused.
    """
    def __init__(self, cutoff, fs, order=1):
        self.alpha = 1.0 - math.exp(
            -2.0 * math.pi * max(cutoff, 1e-6) / max(fs, 1e-6)
        )
        self._y = None

    def get_data(self, x):
        self._y = x if self._y is None else (
            self.alpha * x + (1.0 - self.alpha) * self._y
        )
        return self._y


# --- Waypoint helpers -------------------------------------------------------

def default_waypoints_path():
    share = get_package_share_directory('adapt_full')
    return os.path.join(share, 'waypoints', 'track.csv')


def load_waypoints(path, olat, olon):
    """Load a lon,lat CSV and return a ReferencePath in ENU relative to origin."""
    lon_x, lat_y = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            lon_x.append(float(row[0]))
            lat_y.append(float(row[1]))
    pts = []
    for lon, lat in zip(lon_x, lat_y):
        x, y, _ = geodetic2enu(lat, lon, 0.0, olat, olon, 0.0)
        pts.append((x, y))
    if len(pts) < 2:
        raise RuntimeError(f'waypoints file {path} has <2 points')
    
    return ReferencePath(pts)


def demo_positions(ref_path, fracs, lateral=0.0):
    """Return (N, 2) ENU positions at arc-length fractions along ref_path.

    `lateral` offsets each point perpendicular to the path heading
    (positive = left).
    """
    if not fracs:
        return np.zeros((0, 2), dtype=float)
    s_vals  = ref_path.s
    xy      = ref_path.xy
    headings = ref_path.headings
    total   = ref_path.total_length
    pts = []
    for f in fracs:
        s   = float(f) * total
        idx = int(np.searchsorted(s_vals, s, side='right')) - 1
        idx = int(np.clip(idx, 0, len(xy) - 2))
        ds  = s - s_vals[idx]
        seg = xy[idx + 1] - xy[idx]
        seg_len = float(np.linalg.norm(seg))
        t   = ds / seg_len if seg_len > 1e-6 else 0.0
        pt  = xy[idx] + t * seg
        h   = headings[idx]
        pt  = pt + lateral * np.array([-math.sin(h), math.cos(h)])
        pts.append(pt)
    return np.array(pts, dtype=float)
