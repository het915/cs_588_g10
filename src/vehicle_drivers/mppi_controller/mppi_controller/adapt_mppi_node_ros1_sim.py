"""Adapt MPPI node — ROS1 (rospy) interface for POLARIS_GEM_Simulator.

State is read from Gazebo ground truth (/gazebo/model_states). Spawn
position is used as the map (0, 0) origin. A TF transform map →
base_footprint is broadcast each cycle so RViz can display everything in
the map frame.

Subscribes:
  /gazebo/model_states              gazebo_msgs/ModelStates
  /move_base_simple/goal            geometry_msgs/PoseStamped    (RViz 2D Nav Goal)
  /fusion_pedestrian_tensor         std_msgs/Float32MultiArray  ((M, H, 2) base_link)
  /cone_positions                   geometry_msgs/PoseArray

Publishes:
  /ackermann_cmd                    ackermann_msgs/AckermannDrive
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
import math

import numpy as np

import rospy
import tf

from std_msgs.msg import Float64, Float32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDrive
from gazebo_msgs.msg import ModelStates

# Support both `python -m mppi_controller.adapt_mppi_node_ros1_sim`
# (package context — relative import) and `python adapt_mppi_node_ros1_sim.py`
# from inside the directory (no parent package — fall back to absolute).
try:
    from .mppi import MPPI
    from .reference_path import ReferencePath
except ImportError:
    from mppi import MPPI
    from reference_path import ReferencePath


# =========================================================================== #
#  Helpers                                                                      #
# =========================================================================== #

def front2steer(f_angle_deg):
    a = max(min(f_angle_deg, 35.0), -35.0)
    mag = abs(a)
    sw = -0.1084 * mag * mag + 21.775 * mag
    sw = sw if a >= 0 else -sw
    return max(min(sw, 450.0), -450.0)


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
            name = self._obstacle_names[i] if hasattr(self, '_obstacle_names') and i < len(self._obstacle_names) else f'obs_{i}'

            # Cylinder marker
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

            # Text label above the cylinder
            t = Marker()
            t.header.frame_id = self.frame_id
            t.header.stamp    = stamp
            t.ns              = 'obstacle_labels'
            t.id              = i + 1
            t.type            = Marker.TEXT_VIEW_FACING
            t.action          = Marker.ADD
            t.pose.position.x  = float(obstacles[i, 0])
            t.pose.position.y  = float(obstacles[i, 1])
            t.pose.position.z  = 1.5
            t.pose.orientation.w = 1.0
            t.scale.z = 0.6
            t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 0.0, 1.0
            t.text = name
            msg.markers.append(t)

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
        gp = rospy.get_param

        # ------------------------------------------------------------------ #
        #  State                                                               #
        # ------------------------------------------------------------------ #
        self.rate_hz          = float(gp('~rate_hz',   20.0))
        self.wheelbase        = float(gp('~wheelbase',  1.75))
        self.desired_speed    = min(8.0, float(gp('~desired_speed', 3.0)))
        self.cone_topic        = str(gp('~cone_topic', '/cone_positions'))

        self.speed            = 0.0
        self.ped_trajectories = None
        self._v_cmd           = 0.0

        # Gazebo ground-truth state
        self._gazebo_model_name = str(gp('~gazebo_model_name', 'gem_e4'))
        self._use_gazebo_state  = False

        # Obstacle keywords matched against Gazebo model names
        self._obstacle_keywords = ['cone', 'pedestrian', 'bicycle', 'car1', 'traffic_light', 'stop_sign']
        self._ignore_keywords   = ['gem_e4', 'grey_wall', 'asphalt', 'parking_line',
                                   'redline', 'track_curb', 'bars', 'start_grid',
                                   'yellow_point', 'ground', 'sun', 'sky']
        self._max_obstacle_dist = float(gp('~max_obstacle_dist', 30.0))
        self._gz_x   = 0.0
        self._gz_y   = 0.0
        self._gz_yaw = 0.0
        # Spawn pose recorded on first Gazebo callback — treated as map (0, 0, 0).
        # Yaw is also rebased so map +x = car's spawn heading (not Gazebo +x).
        self._origin_x   = None
        self._origin_y   = None
        self._origin_yaw = 0.0
        self._gz_offset = float(gp('~gazebo_offset', 0.0))
        self._goal_reached_threshold = float(gp('~goal_reached_threshold', 1.5))

        # ------------------------------------------------------------------ #
        #  MPPI + helpers                                                      #
        # ------------------------------------------------------------------ #
        # Auto-detect when ~mppi/device is unset or 'auto': prefer cuda when
        # torch reports it available, otherwise cpu. Explicit 'cpu' / 'cuda'
        # / 'cuda:0' values from rosparam are passed through unchanged.
        _device_raw = str(gp('~mppi/device', 'auto')).strip().lower()
        if _device_raw in ('', 'auto'):
            try:
                import torch as _torch
                _device_raw = 'cuda' if _torch.cuda.is_available() else 'cpu'
            except Exception:
                _device_raw = 'cpu'
        device_param = _device_raw
        self.mppi = MPPI(
            K=int(gp('~mppi/K', 2000)),
            H=int(gp('~mppi/H', 25)),
            dt=float(gp('~mppi/dt', 0.1)),
            sigma_steer=float(gp('~mppi/sigma_steer', 0.3)),
            sigma_accel=float(gp('~mppi/sigma_accel', 0.5)),
            lam=float(gp('~mppi/lambda_', 0.1)),
            v_ref=self.desired_speed,
            w_pos=float(gp('~mppi/w_pos', 15.0)),
            w_vel=float(gp('~mppi/w_vel', 8.0)),
            w_curv=float(gp('~mppi/w_curv', 4.0)),
            w_heading=float(gp('~mppi/w_heading', 100.0)),
            w_obs=float(gp('~mppi/w_obs', 150.0)),
            w_obs_hard=float(gp('~mppi/w_obs_hard', 250.0)),
            w_obs_soft=float(gp('~mppi/w_obs_soft', 40.0)),
            w_cone_hard=float(gp('~mppi/w_cone_hard', 250.0)),
            w_cone_soft=float(gp('~mppi/w_cone_soft', 40.0)),
            w_terminal=float(gp('~mppi/w_terminal', 0.0)),
            ped_sigma=float(gp('~mppi/ped_sigma', 1.5)),
            cone_radius=float(gp('~mppi/cone_radius', 0.8)),
            clearance=float(gp('~mppi/clearance', 1.5)),
            a_min=float(gp('~mppi/a_min', -10.0)),
            a_max=float(gp('~mppi/a_max',  10.0)),
            delta_max=float(gp('~mppi/delta_max', 0.45)),
            wheelbase=self.wheelbase,
            device=device_param,
        )
        self._log_device()

        # Near-pass-through steering low-pass: alpha≈0.98 at 12 Hz / 20 Hz,
        # so real corrections barely lag (~1 tick) while > 15 Hz residual
        # jitter is shaved. Source-side noise reduction (lower sigma_steer,
        # higher K) is doing most of the de-noising; this just smooths
        # the last bit. Tune ~steer_filter/cutoff: lower = smoother +
        # laggier, higher = closer to raw MPPI output.
        self.steer_filter = OnlineFilter(
            cutoff=float(gp('~steer_filter/cutoff', 12.0)),
            fs=float(gp('~steer_filter/fs', self.rate_hz)),
        )

        # Reference path built on demand from /move_base_simple/goal.
        # Control loop short-circuits until the first goal arrives.
        self.ref_path = None
        self._goal_path_samples = int(gp('~goal_path_samples', 50))

        self.obstacles = np.zeros((0, 2), dtype=float)
        self.cones     = np.zeros((0, 2), dtype=float)

        # ------------------------------------------------------------------ #
        #  Visualisation — must be initialised before any subscriber fires    #
        # ------------------------------------------------------------------ #
        self.viz = MPPIVisualizer(
            str(gp('~viz/frame_id', 'map')),
            int(gp('~viz/num_samples', 1)),
        )
        self._tf_br = tf.TransformBroadcaster()

        # ------------------------------------------------------------------ #
        #  Subscribers                                                         #
        # ------------------------------------------------------------------ #
        rospy.Subscriber('/gazebo/model_states', ModelStates, self._gazebo_states_cb, queue_size=10)

        rospy.Subscriber(
            '/fusion_pedestrian_tensor', Float32MultiArray,
            self._pred_tensor_cb, queue_size=10,
        )
        rospy.loginfo('Obstacle source: /fusion_pedestrian_tensor (full trajectories)')

        rospy.Subscriber(self.cone_topic, PoseArray, self._cones_cb, queue_size=10)
        rospy.loginfo(f'Cone source: {self.cone_topic}')

        rospy.Subscriber(
            '/move_base_simple/goal', PoseStamped,
            self._goal_cb, queue_size=1,
        )
        rospy.loginfo('Goal source: /move_base_simple/goal')

        # ------------------------------------------------------------------ #
        #  Publishers                                                          #
        # ------------------------------------------------------------------ #
        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)

        # ------------------------------------------------------------------ #
        #  Timer                                                               #
        # ------------------------------------------------------------------ #
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._control_loop)
        rospy.loginfo(
            f'adapt_mppi_node_sim ready — {self.rate_hz:.1f} Hz, '
            f'v_ref={self.desired_speed:.1f} m/s — waiting for /move_base_simple/goal'
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

    def _gazebo_states_cb(self, msg):
        try:
            idx = msg.name.index(self._gazebo_model_name)
        except ValueError:
            return
        pose  = msg.pose[idx]
        twist = msg.twist[idx]
        q = pose.orientation

        yaw_world = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        if self._origin_x is None:
            self._origin_x   = pose.position.x
            self._origin_y   = pose.position.y
            self._origin_yaw = yaw_world
            rospy.loginfo(
                f'Map origin set to Gazebo ({self._origin_x:.3f}, '
                f'{self._origin_y:.3f}, yaw={math.degrees(self._origin_yaw):.1f}deg)'
            )

        # Position delta rotated into the rebased map frame (so map +x points
        # along the car's spawn heading, regardless of Gazebo world axes).
        dx = pose.position.x - self._origin_x
        dy = pose.position.y - self._origin_y
        c, s = math.cos(-self._origin_yaw), math.sin(-self._origin_yaw)
        self._gz_x = c * dx - s * dy
        self._gz_y = s * dx + c * dy
        # Yaw is also taken modulo the spawn yaw and wrapped to [-pi, pi].
        dy_yaw = yaw_world - self._origin_yaw
        self._gz_yaw = math.atan2(math.sin(dy_yaw), math.cos(dy_yaw))

        self.speed = float(
            math.sqrt(twist.linear.x ** 2 + twist.linear.y ** 2)
        )
        self._use_gazebo_state = True

        half = self._gz_yaw / 2.0
        self._tf_br.sendTransform(
            (self._gz_x, self._gz_y, 0.0),
            (0.0, 0.0, math.sin(half), math.cos(half)),
            rospy.Time.now(),
            'base_footprint',
            'map',
        )

        # --- Extract obstacles from Gazebo model_states ---
        obs_list = []
        obs_names = []
        for name, pose_i in zip(msg.name, msg.pose):
            if any(k in name for k in self._ignore_keywords):
                continue
            if not any(k in name for k in self._obstacle_keywords):
                continue
            ox = pose_i.position.x - self._origin_x
            oy = pose_i.position.y - self._origin_y
            dist = math.sqrt((ox - self._gz_x) ** 2 + (oy - self._gz_y) ** 2)
            if dist > self._max_obstacle_dist:
                continue
            obs_list.append([ox, oy])
            obs_names.append(name)                          # ← 追加
        self.obstacles = np.array(obs_list, dtype=float) if obs_list else np.zeros((0, 2), dtype=float)
        self._obstacle_names = obs_names                    # ← 追加

        
    def _pred_tensor_cb(self, msg):
        if not msg.data or not self._use_gazebo_state:
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
        if not self._use_gazebo_state:
            start = np.array([0.0, 0.0])
            rospy.logwarn('Goal received before Gazebo state — using (0,0) as start')
        else:
            sx, sy, _ = self._gem_state()
            start = np.array([sx, sy])
        goal = np.array([msg.pose.position.x, msg.pose.position.y])

        if np.linalg.norm(goal - start) < 0.5:
            rospy.logwarn('Goal too close to current pose (<0.5 m); ignoring')
            return

        stop_offset = float(rospy.get_param('~stop_offset_m', 1.5))
        direction = goal - start
        dist = np.linalg.norm(direction)
        if dist > stop_offset + 0.5:
            effective_goal = start + direction * ((dist - stop_offset) / dist)
        else:
            effective_goal = start + direction * 0.5 
        N = max(2, self._goal_path_samples)
        ts = np.linspace(0.0, 1.0, N)
        pts = start + (effective_goal - start) * ts[:, None]
        self.ref_path = ReferencePath(pts)
        self.viz.publish_static(self.ref_path)
        rospy.loginfo(
            f'New goal: ({goal[0]:.2f}, {goal[1]:.2f}) — '
            f'{N}-pt path, length={np.linalg.norm(goal - start):.2f} m'
        )

    # ---------------------------------------------------------------------- #
    #  State helpers                                                           #
    # ---------------------------------------------------------------------- #

    def _gem_state(self):
        x = self._gz_x - self._gz_offset * math.cos(self._gz_yaw)
        y = self._gz_y - self._gz_offset * math.sin(self._gz_yaw)
        return x, y, self._gz_yaw

    # ---------------------------------------------------------------------- #
    #  Control loop                                                            #
    # ---------------------------------------------------------------------- #

    def _control_loop(self, _event):
        if self.ref_path is None or not self._use_gazebo_state:
            return

        x, y, yaw = self._gem_state()
        stamp = rospy.Time.now()

        goal = self.ref_path.xy[-1]
        dist_to_goal = float(np.linalg.norm(np.array([x, y]) - goal))
        if dist_to_goal < self._goal_reached_threshold:
            self.ackermann_pub.publish(AckermannDrive())
            self._v_cmd = 0.0
            self.ref_path = None
            rospy.loginfo(f'Goal reached — stopped ({dist_to_goal:.2f} m from target)')
            return

        state = np.array([x, y, yaw, max(self.speed, 0.0)], dtype=float)

        self.viz.append_robot_pose(x, y, yaw, stamp)

        active_path = self.ref_path

        u = self.mppi.update(
            state, active_path,
            obstacles=self.obstacles if self.ped_trajectories is None else None,
            ped_trajectories=self.ped_trajectories,
            cones=self.cones if len(self.cones) > 0 else None,
        )
        delta_raw = float(u[0])
        delta = float(self.steer_filter.get_data(delta_raw))
        accel = float(u[1])

        delta_deg = math.degrees(delta)

        # MPPI's accel sign carries the intent: + means accelerate toward
        # cruising speed, - means brake. The plugin already slews `speed`
        # at `acceleration` m/s², so we just hand it a target and a rate —
        # no software integration on _v_cmd (that was double-rate-limiting
        # the response). `~min_accel` floors the slew so the plugin still
        # moves when MPPI commands a near-zero accel at cruise.
        if accel >= 0.0:
            self._v_cmd = self.desired_speed
        else:
            self._v_cmd = 0.0
        slew = max(abs(float(accel)), float(rospy.get_param('~min_accel', 1.0)))

        ackermann_msg = AckermannDrive()
        ackermann_msg.steering_angle = delta
        ackermann_msg.steering_angle_velocity = 0.0
        ackermann_msg.speed = self._v_cmd
        ackermann_msg.acceleration = slew
        self.ackermann_pub.publish(ackermann_msg)

        ess = self.mppi.effective_sample_count()
        rospy.loginfo_throttle(
            1.0,
            f'MPPI | pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.1f}deg '
            f'v={self.speed:.2f}→{self._v_cmd:.2f} a={accel:+.2f} '
            f'δ_raw={math.degrees(delta_raw):+.1f}° δ_cmd={delta_deg:+.1f}° '
            f'obs={len(self.obstacles)} ESS/K={ess/self.mppi.K:.2f}',
        )

        self.viz.publish(self.mppi, self.obstacles, stamp, accel=accel, delta=delta)


# --------------------------------------------------------------------------- #

def main():
    rospy.init_node('adapt_mppi_node_sim')
    AdaptMPPINode()
    rospy.spin()


class OnlineFilter:
    """
    Simple low-pass filter for scalar streams (e.g., noisy steering).
    Implements a first-order IIR filter (exponential smoothing).
    """
    def __init__(self, cutoff: float, fs: float):
        self.cutoff  = float(cutoff)
        self.fs      = float(fs)

        # First-order RC LPF discretised via backward Euler:
        #   alpha = dt / (tau + dt),  tau = 1 / (2π f_c),  dt = 1 / fs.
        # Both terms must be in seconds — the previous version compared
        # `tau` (s) against `fs` (Hz) and produced alpha ≈ 6.6e-4, making
        # the filter effectively zero gain.
        tau = 1.0 / (2.0 * math.pi * max(self.cutoff, 1e-6))
        dt  = 1.0 / max(self.fs, 1e-6)
        self.alpha = dt / (tau + dt)

        self._y_prev = 0.0

    def get_data(self, x: float) -> float:
        """Update filter state and return filtered value."""
        y = self.alpha * float(x) + (1.0 - self.alpha) * self._y_prev
        self._y_prev = y
        return y

if __name__ == '__main__':
    main()
