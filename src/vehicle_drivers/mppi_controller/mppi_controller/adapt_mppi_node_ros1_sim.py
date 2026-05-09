"""Adapt MPPI node — ROS1 (rospy) interface, PACMOD-LESS sim variant.

For use in the POLARIS_GEM_Simulator, which has no PACMod CAN-bridge.
Every line touching ``pacmod2_msgs`` is commented out; MPPI math and
visualisation still run, so RViz can show the chosen rollout, samples,
obstacles, and reference path while the gazebo vehicle is driven by
another controller (or manually).

For real-vehicle drive output, see the sibling file
``adapt_mppi_node_ros1.py`` — identical except the pacmod blocks
are live there.

Subscribes:
  /navsatfix                        sensor_msgs/NavSatFix
  /insnavgeod                       septentrio_gnss_driver/INSNavGeod
  # /pacmod/enabled                 std_msgs/Bool                    [disabled]
  # /pacmod/vehicle_speed_rpt       pacmod2_msgs/VehicleSpeedRpt     [disabled]
  /fusion_pedestrian_position       std_msgs/Int32MultiArray
  /pedestrian_predictions_tensor    std_msgs/Float32MultiArray
  /cone_positions                   geometry_msgs/PoseArray

Publishes (control) — ALL DISABLED in this sim variant:
  # /pacmod/global_cmd              pacmod2_msgs/GlobalCmd
  # /pacmod/shift_cmd               pacmod2_msgs/SystemCmdInt
  # /pacmod/brake_cmd               pacmod2_msgs/SystemCmdFloat
  # /pacmod/accel_cmd               pacmod2_msgs/SystemCmdFloat
  # /pacmod/turn_cmd                pacmod2_msgs/SystemCmdInt
  # /pacmod/steering_cmd            pacmod2_msgs/PositionWithSpeed

Publishes (viz):
  /adapt/viz/reference_path         nav_msgs/Path                (latched)
  /adapt/viz/current_goal           visualization_msgs/Marker    (latched)
  /adapt/viz/chosen_trajectory      nav_msgs/Path
  /adapt/viz/sampled_trajectories   visualization_msgs/MarkerArray
  /adapt/viz/obstacles              visualization_msgs/MarkerArray
  /adapt/viz/robot_trajectory       nav_msgs/Path
  /adapt/viz/debug/accel            std_msgs/Float64
  /adapt/viz/debug/delta            std_msgs/Float64
"""
import colorsys
import csv
import math
import os

import numpy as np

import rospy
import rospkg

from std_msgs.msg import Bool, Float64, Int32MultiArray, Float32MultiArray
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

# --- PACMOD DISABLED for sim variant -----------------------------------
# pacmod2_msgs is dual-build (catkin under ROS1, ament under ROS2 via
# package.xml condition="$ROS_VERSION == ..." tags) — same package name in
# both. Fields verified identical between the ROS1/ROS2 builds.
# from pacmod2_msgs.msg import (
#     GlobalCmd, PositionWithSpeed, SystemCmdFloat, SystemCmdInt,
#     VehicleSpeedRpt,
# )

try:
    from septentrio_gnss_driver.msg import INSNavGeod
    _HAS_SEPTENTRIO = True
except ImportError:
    _HAS_SEPTENTRIO = False
    INSNavGeod = None

# Support both `python -m mppi_controller.adapt_mppi_node_ros1_sim`
# (package context — relative import) and `python adapt_mppi_node_ros1_sim.py`
# from inside the directory (no parent package — fall back to absolute).
try:
    from .mppi import MPPI
    from .reference_path import ReferencePath
except ImportError:
    from mppi import MPPI
    from reference_path import ReferencePath

_DISABLE_DRIVE_COMMANDS = True


# =========================================================================== #
#  Helpers (inlined from utils.py — rclpy-free)                                 #
# =========================================================================== #

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
    x,  y,  z  = _geodetic_to_ecef(lat,  lon,  h)
    x0, y0, z0 = _geodetic_to_ecef(lat0, lon0, h0)
    dx, dy, dz = x - x0, y - y0, z - z0
    slat, clat = math.sin(math.radians(lat0)), math.cos(math.radians(lat0))
    slon, clon = math.sin(math.radians(lon0)), math.cos(math.radians(lon0))
    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    u =  clat * clon * dx + clat * slon * dy + slat * dz
    return e, n, u


def heading_to_yaw(heading_deg):
    if heading_deg < 270.0:
        return math.radians(90.0 - heading_deg)
    return math.radians(450.0 - heading_deg)


def front2steer(f_angle_deg):
    a = max(min(f_angle_deg, 35.0), -35.0)
    mag = abs(a)
    sw = -0.1084 * mag * mag + 21.775 * mag
    sw = sw if a >= 0 else -sw
    return max(min(sw, 450.0), -450.0)


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


def default_waypoints_path():
    pkg_path = rospkg.RosPack().get_path('adapt_full')
    return os.path.join(pkg_path, 'waypoints', 'track.csv')


def load_waypoints(path, olat, olon):
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
    if not fracs:
        return np.zeros((0, 2), dtype=float)
    s_vals   = ref_path.s
    xy       = ref_path.xy
    headings = ref_path.headings
    total    = ref_path.total_length
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


# =========================================================================== #
#  Visualisation (slim ROS1 port of viz.MPPIVisualizer)                         #
# =========================================================================== #

class MPPIVisualizer:
    def __init__(self, frame_id, num_samples):
        self.frame_id    = frame_id
        self._num_samples = num_samples
        self._palette = [
            colorsys.hsv_to_rgb(i / max(num_samples, 1), 0.45, 0.95)
            for i in range(num_samples)
        ]
        self._robot_traj_poses = []

        self._ref_pub     = rospy.Publisher('/adapt/viz/reference_path',       Path,        queue_size=1,  latch=True)
        self._goal_pub    = rospy.Publisher('/adapt/viz/current_goal',         Marker,      queue_size=1,  latch=True)
        self._chosen_pub  = rospy.Publisher('/adapt/viz/chosen_trajectory',    Path,        queue_size=10)
        self._samples_pub = rospy.Publisher('/adapt/viz/sampled_trajectories', MarkerArray, queue_size=10)
        self._obs_pub     = rospy.Publisher('/adapt/viz/obstacles',            MarkerArray, queue_size=10)
        self._traj_pub    = rospy.Publisher('/adapt/viz/robot_trajectory',     Path,        queue_size=10)
        self._accel_pub   = rospy.Publisher('/adapt/viz/debug/accel',          Float64,     queue_size=10)
        self._delta_pub   = rospy.Publisher('/adapt/viz/debug/delta',          Float64,     queue_size=10)

    # ------------------------------------------------------------------ #
    def publish_static(self, ref_path):
        stamp = rospy.Time.now()
        self._pub_reference_path(ref_path, stamp)
        self._pub_goal_marker(ref_path, stamp)

    def append_robot_pose(self, x, y, yaw, stamp):
        ps = PoseStamped()
        ps.header.frame_id = self.frame_id
        ps.header.stamp    = stamp
        ps.pose.position.x = x
        ps.pose.position.y = y
        half = yaw / 2.0
        ps.pose.orientation.z = math.sin(half)
        ps.pose.orientation.w = math.cos(half)
        self._robot_traj_poses.append(ps)

    def publish(self, mppi, obstacles, stamp, accel=None, delta=None):
        if mppi.last_traj is None:
            return
        self._pub_chosen_trajectory(mppi, stamp)
        self._pub_sampled_rollouts(mppi, stamp)
        self._pub_obstacle_markers(mppi, obstacles, stamp)
        self._pub_robot_trajectory(stamp)
        if accel is not None:
            self._accel_pub.publish(Float64(data=float(accel)))
        if delta is not None:
            self._delta_pub.publish(Float64(data=float(delta)))

    # ------------------------------------------------------------------ #
    def _pub_reference_path(self, ref_path, stamp):
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp    = stamp
        for x, y in ref_path.xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x   = float(x)
            ps.pose.position.y   = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._ref_pub.publish(msg)

    def _pub_goal_marker(self, ref_path, stamp):
        gx, gy = float(ref_path.xy[-1, 0]), float(ref_path.xy[-1, 1])
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp    = stamp
        m.ns, m.id        = 'current_goal', 0
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position.x  = gx
        m.pose.position.y  = gy
        m.pose.position.z  = 0.5
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.2
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.4, 0.0, 1.0
        self._goal_pub.publish(m)

    def _pub_chosen_trajectory(self, mppi, stamp):
        mean_traj = mppi.last_mean_traj
        H = mean_traj.shape[0]
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp    = stamp
        for h in range(H):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x   = float(mean_traj[h, 0])
            ps.pose.position.y   = float(mean_traj[h, 1])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self._chosen_pub.publish(path)

    def _pub_sampled_rollouts(self, mppi, stamp):
        traj = mppi.last_traj
        w    = mppi.last_weights
        K, H, _ = traj.shape
        N = min(self._num_samples, K)
        top_idx = np.argsort(w)[-N:][::-1]

        msg = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.frame_id
        clear.header.stamp    = stamp
        clear.action          = Marker.DELETEALL
        msg.markers.append(clear)

        for i, k in enumerate(top_idx):
            r_, g_, b_ = self._palette[i]
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp    = stamp
            m.ns              = 'mppi_samples'
            m.id              = i + 1
            m.type            = Marker.LINE_STRIP
            m.action          = Marker.ADD
            m.scale.x         = 0.05
            m.color.r, m.color.g, m.color.b = float(r_), float(g_), float(b_)
            m.color.a         = 0.75
            m.pose.orientation.w = 1.0
            for h in range(H):
                p = Point()
                p.x = float(traj[k, h, 0])
                p.y = float(traj[k, h, 1])
                m.points.append(p)
            msg.markers.append(m)

        self._samples_pub.publish(msg)

    def _pub_obstacle_markers(self, mppi, obstacles, stamp):
        msg = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.frame_id
        clear.header.stamp    = stamp
        clear.action          = Marker.DELETEALL
        msg.markers.append(clear)

        r = float(mppi.clearance)
        for i in range(len(obstacles)):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp    = stamp
            m.ns              = 'obstacles'
            m.id              = i + 1
            m.type            = Marker.CYLINDER
            m.action          = Marker.ADD
            m.pose.position.x  = float(obstacles[i, 0])
            m.pose.position.y  = float(obstacles[i, 1])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2.0 * r
            m.scale.z = 0.15
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.25, 0.25, 0.35
            msg.markers.append(m)

        self._obs_pub.publish(msg)

    def _pub_robot_trajectory(self, stamp):
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp    = stamp
        path.poses           = self._robot_traj_poses
        self._traj_pub.publish(path)


# =========================================================================== #
#  Node                                                                         #
# =========================================================================== #

class AdaptMPPINode:
    def __init__(self):
        # rospy.get_param honours nested YAML: ``mppi: {K: 500}`` is reached
        # via ``rospy.get_param('~mppi/K')``.
        gp = rospy.get_param

        # ------------------------------------------------------------------ #
        #  State                                                               #
        # ------------------------------------------------------------------ #
        self.rate_hz   = float(gp('~rate_hz',   20.0))
        self.wheelbase = float(gp('~wheelbase',  1.75))
        self.offset    = float(gp('~offset',     1.26))
        self.olat      = float(gp('~origin_lat',  40.0927422))
        self.olon      = float(gp('~origin_lon', -88.2359639))
        self.desired_speed         = min(5.0, float(gp('~desired_speed', 4.0)))
        self.max_throttle          = min(1.0, float(gp('~max_throttle',  0.4)))
        self.max_brake             = min(1.0, float(gp('~max_brake',     0.4)))
        # self.require_pacmod_enable = bool(gp('~require_pacmod_enable',  True))  # PACMOD DISABLED
        self.prediction_source     = str(gp('~prediction_source', 'raw'))
        self.cone_topic            = str(gp('~cone_topic', '/cone_positions'))

        self.lat = 0.0
        self.lon = 0.0
        self.heading      = 0.0
        self.speed        = 0.0
        # self.pacmod_enable    = False   # PACMOD DISABLED
        self.ped_trajectories = None
        # self._pacmod_primed   = False   # PACMOD DISABLED
        self._v_cmd = 0.0
        self._has_ins_heading   = False
        self._has_valid_heading = False
        self._gps_hdg_anchor    = None

        # ------------------------------------------------------------------ #
        #  MPPI + helpers                                                      #
        # ------------------------------------------------------------------ #
        device_param = str(gp('~mppi/device', 'cpu')).strip() or None
        self.mppi = MPPI(
            K=int(gp('~mppi/K', 500)),
            H=int(gp('~mppi/H', 30)),
            dt=float(gp('~mppi/dt', 0.1)),
            sigma_steer=float(gp('~mppi/sigma_steer', 0.15)),
            sigma_accel=float(gp('~mppi/sigma_accel', 0.5)),
            lam=float(gp('~mppi/lambda_', 0.1)),
            v_ref=self.desired_speed,
            w_pos=float(gp('~mppi/w_pos', 15.0)),
            w_vel=float(gp('~mppi/w_vel', 5.0)),
            w_curv=float(gp('~mppi/w_curv', 2.0)),
            w_obs=float(gp('~mppi/w_obs', 150.0)),
            w_obs_hard=float(gp('~mppi/w_obs_hard', 250.0)),
            w_obs_soft=float(gp('~mppi/w_obs_soft', 40.0)),
            w_cone_hard=float(gp('~mppi/w_cone_hard', 250.0)),
            w_cone_soft=float(gp('~mppi/w_cone_soft', 40.0)),
            w_terminal=float(gp('~mppi/w_terminal', 0.0)),
            ped_sigma=float(gp('~mppi/ped_sigma', 1.5)),
            cone_radius=float(gp('~mppi/cone_radius', 0.8)),
            clearance=float(gp('~mppi/clearance', 1.5)),
            wheelbase=self.wheelbase,
            device=device_param,
        )
        self._log_device()

        self.pid_speed = PID(
            kp=float(gp('~pid/kp', 2.0)),
            ki=float(gp('~pid/ki', 0.0)),
            kd=float(gp('~pid/kd', 0.1)),
            wg=float(gp('~pid/wg', 10.0)),
        )
        self.speed_filter = OnlineFilter(
            cutoff=float(gp('~filter/cutoff', 1.2)),
            fs=float(gp('~filter/fs', 30.0)),
            order=int(gp('~filter/order', 4)),
        )

        # --- Reference path: built on demand from /goal_pose --
        # No CSV required. Until the first goal arrives, ref_path is None and
        # the control loop short-circuits (no MPPI rollout, no viz update).
        self.ref_path = None
        self._goal_path_samples = int(gp('~goal_path_samples', 50))

        # demo_positions() needs a ref_path; without one, start with empty
        # obstacle / cone arrays. Real detections still arrive over /cones,
        # /fusion_pedestrian_position, /pedestrian_predictions_tensor.
        self.obstacles = np.zeros((0, 2), dtype=float)
        self.cones     = np.zeros((0, 2), dtype=float)

        # ------------------------------------------------------------------ #
        #  Subscribers                                                         #
        # ------------------------------------------------------------------ #
        rospy.Subscriber('/navsatfix', NavSatFix, self._gnss_cb, queue_size=10)
        if _HAS_SEPTENTRIO:
            rospy.Subscriber('/insnavgeod', INSNavGeod, self._ins_cb, queue_size=10)
        else:
            rospy.logwarn(
                'septentrio_gnss_driver not found — /insnavgeod heading unavailable'
            )
        # --- PACMOD DISABLED for sim variant -------------------------------
        # rospy.Subscriber('/pacmod/enabled', Bool, self._enable_cb, queue_size=10)
        # rospy.Subscriber(
        #     '/pacmod/vehicle_speed_rpt', VehicleSpeedRpt, self._speed_cb, queue_size=10,
        # )
        if self.prediction_source == 'predicted':
            rospy.Subscriber(
                '/pedestrian_predictions_tensor', Float32MultiArray,
                self._pred_tensor_cb, queue_size=10,
            )
            rospy.loginfo(
                'Obstacle source: /pedestrian_predictions_tensor (full trajectories)'
            )
        else:
            rospy.Subscriber(
                '/fusion_pedestrian_position', Int32MultiArray,
                self._ped_cb, queue_size=10,
            )
            rospy.loginfo(
                'Obstacle source: /fusion_pedestrian_position (raw detections)'
            )
        rospy.Subscriber(self.cone_topic, PoseArray, self._cones_cb, queue_size=10)
        rospy.loginfo(f'Cone source: {self.cone_topic}')

        # RViz "2D Nav Goal" publishes here. On each click we rebuild
        # ref_path as a sampled straight segment (current_pose -> goal).
        rospy.Subscriber(
            '/goal_pose', PoseStamped,
            self._goal_cb, queue_size=1,
        )
        rospy.loginfo('Goal source: /goal_pose (PoseStamped)')

        # ------------------------------------------------------------------ #
        #  Publishers — control (PACMOD DISABLED for sim variant)              #
        # ------------------------------------------------------------------ #
        # self.global_pub = rospy.Publisher('/pacmod/global_cmd',   GlobalCmd,         queue_size=10)
        # self.gear_pub   = rospy.Publisher('/pacmod/shift_cmd',    SystemCmdInt,      queue_size=10)
        # self.brake_pub  = rospy.Publisher('/pacmod/brake_cmd',    SystemCmdFloat,    queue_size=10)
        # self.accel_pub  = rospy.Publisher('/pacmod/accel_cmd',    SystemCmdFloat,    queue_size=10)
        # self.turn_pub   = rospy.Publisher('/pacmod/turn_cmd',     SystemCmdInt,      queue_size=10)
        # self.steer_pub  = rospy.Publisher('/pacmod/steering_cmd', PositionWithSpeed, queue_size=10)

        # self.global_cmd = GlobalCmd(enable=False, clear_override=True)
        # self.gear_cmd   = SystemCmdInt(command=2)
        # self.brake_cmd  = SystemCmdFloat(command=0.0)
        # self.accel_cmd  = SystemCmdFloat(command=0.0)
        # self.turn_cmd   = SystemCmdInt(command=1)
        # self.steer_cmd  = PositionWithSpeed(angular_position=0.0, angular_velocity_limit=4.0)
        # Plain-attribute fallbacks so the control loop's bookkeeping (logged
        # values, viz hooks) continues to work without the pacmod messages.
        self._accel_cmd_value = 0.0
        self._brake_cmd_value = 0.0

        # ------------------------------------------------------------------ #
        #  Visualisation                                                        #
        # ------------------------------------------------------------------ #
        self.viz = MPPIVisualizer(
            str(gp('~viz/frame_id', 'map')),
            int(gp('~viz/num_samples', 1)),
        )
        # ref_path is built on first goal — defer the latched viz publish
        # until then (see _goal_cb).

        # ------------------------------------------------------------------ #
        #  Timer                                                                #
        # ------------------------------------------------------------------ #
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._control_loop)
        rospy.loginfo(
            f'adapt_mppi_node_sim ready — {self.rate_hz:.1f} Hz, '
            f'v_ref={self.desired_speed:.1f} m/s — waiting for /goal_pose'
        )

    # ---------------------------------------------------------------------- #
    #  Init helpers                                                            #
    # ---------------------------------------------------------------------- #

    def _log_device(self):
        try:
            import torch
            dev = self.mppi.device
            if dev.type == 'cuda':
                idx = dev.index if dev.index is not None else 0
                name = torch.cuda.get_device_name(idx)
                total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
                rospy.loginfo(f'MPPI device: {dev} ({name}, {total_gb:.1f} GiB VRAM)')
            else:
                rospy.loginfo(f'MPPI device: {dev} (CPU)')
        except Exception as exc:
            rospy.logwarn(f'MPPI device log failed: {exc}')

    # ---------------------------------------------------------------------- #
    #  Callbacks                                                               #
    # ---------------------------------------------------------------------- #

    def _gnss_cb(self, msg):
        new_lat, new_lon = msg.latitude, msg.longitude
        if not self._has_ins_heading:
            if self._gps_hdg_anchor is None and (self.lat != 0.0 or self.lon != 0.0):
                self._gps_hdg_anchor = (self.lat, self.lon)
            if self._gps_hdg_anchor is not None and self.speed > 0.5:
                ex0, ey0, _ = geodetic2enu(
                    self._gps_hdg_anchor[0], self._gps_hdg_anchor[1],
                    0.0, self.olat, self.olon, 0.0)
                ex1, ey1, _ = geodetic2enu(
                    new_lat, new_lon, 0.0, self.olat, self.olon, 0.0)
                dx, dy = ex1 - ex0, ey1 - ey0
                if math.hypot(dx, dy) > 2.0:
                    self.heading = (90.0 - math.degrees(math.atan2(dy, dx))) % 360.0
                    self._gps_hdg_anchor = (new_lat, new_lon)
                    self._has_valid_heading = True
        self.lat = new_lat
        self.lon = new_lon

    def _ins_cb(self, msg):
        if math.isnan(msg.heading):
            return
        self._has_ins_heading = True
        self._has_valid_heading = True
        self.heading = msg.heading

    # --- PACMOD DISABLED for sim variant ---------------------------------
    # def _enable_cb(self, msg):
    #     self.pacmod_enable = msg.data
    #
    # def _speed_cb(self, msg):
    #     self.speed = float(self.speed_filter.get_data(msg.vehicle_speed))

    def _ped_cb(self, msg):
        data = msg.data
        if not data or len(data) % 2 != 0:
            return
        if self.lat == 0.0 and self.lon == 0.0:
            return
        ex, ey, yaw = self._gem_state()
        out = []
        for i in range(0, len(data), 2):
            dist = float(data[i])
            rad  = math.radians(float(data[i + 1]))
            xe   = dist * math.cos(rad)
            ye   = dist * math.sin(rad)
            out.append((
                ex + xe * math.cos(yaw) - ye * math.sin(yaw),
                ey + xe * math.sin(yaw) + ye * math.cos(yaw),
            ))
        self.obstacles = np.asarray(out, dtype=float) if out else np.zeros((0, 2))

    def _pred_tensor_cb(self, msg):
        if not msg.data or (self.lat == 0.0 and self.lon == 0.0):
            self.ped_trajectories = None
            return
        dims = msg.layout.dim
        if len(dims) < 2:
            self.ped_trajectories = None
            return
        M, H = dims[0].size, dims[1].size
        if M == 0 or H == 0:
            self.ped_trajectories = None
            return
        arr = np.array(msg.data, dtype=np.float32).reshape(M, H, 2)
        ex, ey, yaw = self._gem_state()
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        world = np.empty_like(arr)
        world[:, :, 0] = cos_y * arr[:, :, 0] - sin_y * arr[:, :, 1] + ex
        world[:, :, 1] = sin_y * arr[:, :, 0] + cos_y * arr[:, :, 1] + ey
        self.ped_trajectories = world

    def _cones_cb(self, msg):
        if not msg.poses:
            return
        self.cones = np.array(
            [[pose.position.x, pose.position.y] for pose in msg.poses],
            dtype=np.float32,
        )

    def _goal_cb(self, msg):
        """RViz 2D-Nav-Goal -> rebuild ref_path as a sampled straight segment.

        Goal is assumed to be in the same frame as our ENU map (same
        frame_id the viz uses, default 'map'). The path runs from the
        vehicle's current (x, y) — or (0, 0) if GNSS hasn't fixed yet —
        to the clicked (goal.x, goal.y), sampled at N points so MPPI's
        arc-length lookahead and heading derivatives behave well.
        """
        if self.lat == 0.0 and self.lon == 0.0:
            start = np.array([0.0, 0.0])
            rospy.logwarn('Goal received before GPS fix — using (0,0) as start')
        else:
            sx, sy, _ = self._gem_state()
            start = np.array([sx, sy])
        goal = np.array([msg.pose.position.x, msg.pose.position.y])

        if np.linalg.norm(goal - start) < 0.5:
            rospy.logwarn('Goal too close to current pose (<0.5 m); ignoring')
            return

        N = max(2, self._goal_path_samples)
        ts = np.linspace(0.0, 1.0, N)
        pts = start + (goal - start) * ts[:, None]
        self.ref_path = ReferencePath(pts)
        self.viz.publish_static(self.ref_path)
        rospy.loginfo(
            f'New goal: ({goal[0]:.2f}, {goal[1]:.2f}) — '
            f'{N}-pt path, length={np.linalg.norm(goal - start):.2f} m'
        )

    # ---------------------------------------------------------------------- #
    #  Control loop                                                            #
    # ---------------------------------------------------------------------- #

    def _gem_state(self):
        local_x, local_y, _ = geodetic2enu(
            self.lat, self.lon, 0.0, self.olat, self.olon, 0.0
        )
        yaw = heading_to_yaw(self.heading)
        return (
            local_x - self.offset * math.cos(yaw),
            local_y - self.offset * math.sin(yaw),
            yaw,
        )

    # --- PACMOD DISABLED for sim variant ---------------------------------
    # def _prime_pacmod(self):
    #     self.global_cmd.enable = True
    #     self.global_cmd.clear_override = True
    #     self.global_pub.publish(self.global_cmd)
    #     self.gear_cmd.command = 3
    #     self.gear_pub.publish(self.gear_cmd)
    #     self.brake_cmd.command = 0.0
    #     self.brake_pub.publish(self.brake_cmd)
    #     self.accel_cmd.command = 0.0
    #     self.accel_pub.publish(self.accel_cmd)
    #     self.turn_cmd.command = 1
    #     self.turn_pub.publish(self.turn_cmd)
    #     self._pacmod_primed = True
    #     rospy.logwarn('PACMod primed: enable + FORWARD')

    def _control_loop(self, _event):
        # --- PACMOD DISABLED: enable gate + priming skipped ----------------
        # if self.require_pacmod_enable and not self.pacmod_enable:
        #     return
        if self.ref_path is None:
            return
        if self.lat == 0.0 and self.lon == 0.0:
            return
        if not self._has_valid_heading:
            return
        # if not self._pacmod_primed:
        #     self._prime_pacmod()

        x, y, yaw = self._gem_state()
        state = np.array([x, y, yaw, max(self.speed, 0.0)], dtype=float)
        stamp = rospy.Time.now()

        self.viz.append_robot_pose(x, y, yaw, stamp)

        # TODO: Check trim logic. Check the ref path frequency.
        # active_path = self.ref_path.trim_behind((x, y))
        active_path = self.ref_path

        u = self.mppi.update(
            state, active_path,
            obstacles=self.obstacles if self.ped_trajectories is None else None,
            ped_trajectories=self.ped_trajectories,
            cones=self.cones if len(self.cones) > 0 else None,
        )
        delta = float(u[0])
        accel = float(u[1])

        sw_deg = front2steer(math.degrees(delta))
        # self.steer_cmd.angular_position = math.radians(sw_deg)   # PACMOD DISABLED

        self._v_cmd = max(
            0.0,
            min(self._v_cmd + accel * (1.0 / self.rate_hz), self.desired_speed),
        )
        now = stamp.to_sec()
        speed_err = self._v_cmd - self.speed
        if abs(speed_err) < 0.05:
            speed_err = 0.0
        pid_out = self.pid_speed.get_control(now, speed_err)
        if pid_out >= 0.0:
            self._accel_cmd_value = min(pid_out, self.max_throttle)
            self._brake_cmd_value = 0.0
        else:
            self._accel_cmd_value = 0.0
            self._brake_cmd_value = min(abs(pid_out), self.max_brake)

        # --- PACMOD DISABLED for sim variant -------------------------------
        # if _DISABLE_DRIVE_COMMANDS:
        #     self.accel_cmd.command = 0.0
        #     self.brake_cmd.command = 0.0
        #     self.steer_cmd.angular_position = 0.0
        # else:
        #     self.steer_pub.publish(self.steer_cmd)
        #     self.accel_pub.publish(self.accel_cmd)
        #     self.brake_pub.publish(self.brake_cmd)
        #
        # self.global_cmd.enable = True
        # self.global_pub.publish(self.global_cmd)

        ess = self.mppi.effective_sample_count()
        rospy.loginfo_throttle(
            1.0,
            f'MPPI | pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.1f}deg '
            f'v={self.speed:.2f}→{self._v_cmd:.2f} '
            f'thr={self._accel_cmd_value:.2f} brk={self._brake_cmd_value:.2f} '
            f'sw={sw_deg:.1f}deg obs={len(self.obstacles)} '
            f'ESS/K={ess/self.mppi.K:.2f}',
        )

        self.viz.publish(self.mppi, self.obstacles, stamp, accel=accel, delta=delta)


# --------------------------------------------------------------------------- #

def main():
    rospy.init_node('adapt_mppi_node_sim')
    AdaptMPPINode()
    rospy.spin()


if __name__ == '__main__':
    main()
