"""Adapt MPPI node — ROS2 interface for the MPPI controller.

Subscribes:
  /navsatfix                        sensor_msgs/NavSatFix
  /insnavgeod                       septentrio_gnss_driver/INSNavGeod
  /pacmod/enabled                   std_msgs/Bool
  /pacmod/vehicle_speed_rpt         pacmod2_msgs/VehicleSpeedRpt
  /fusion_pedestrian_position       std_msgs/Int32MultiArray
  /pedestrian_predictions_tensor    std_msgs/Float32MultiArray
  /cone_positions                   geometry_msgs/PoseArray

Publishes (control):
  /pacmod/global_cmd                pacmod2_msgs/GlobalCmd
  /pacmod/shift_cmd                 pacmod2_msgs/SystemCmdInt
  /pacmod/brake_cmd                 pacmod2_msgs/SystemCmdFloat
  /pacmod/accel_cmd                 pacmod2_msgs/SystemCmdFloat
  /pacmod/turn_cmd                  pacmod2_msgs/SystemCmdInt
  /pacmod/steering_cmd              pacmod2_msgs/PositionWithSpeed

Visualisation publishers are managed by MPPIVisualizer (viz.py).
"""
import math

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Int32MultiArray, Float32MultiArray
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseArray, PoseStamped
from pacmod2_msgs.msg import (
    GlobalCmd, PositionWithSpeed, SystemCmdFloat, SystemCmdInt,
    VehicleSpeedRpt,
)

try:
    from septentrio_gnss_driver.msg import INSNavGeod
    _HAS_SEPTENTRIO = True
except ImportError:
    _HAS_SEPTENTRIO = False
    INSNavGeod = None

from .mppi import MPPI
from .viz import MPPIVisualizer
from .reference_path import ReferencePath
from .utils import (
    PID, OnlineFilter,
    geodetic2enu, heading_to_yaw, front2steer,
    default_waypoints_path, load_waypoints, demo_positions,
)

_DISABLE_DRIVE_COMMANDS = True

class AdaptMPPINode(Node):
    def _init_(self):
        super()._init_(
            'adapt_mppi_node',
            automatically_declare_parameters_from_overrides=True,
        )
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        # ------------------------------------------------------------------ #
        #  State                                                               #
        # ------------------------------------------------------------------ #
        self.rate_hz   = float(p('rate_hz'))
        self.wheelbase = float(p('wheelbase'))
        self.offset    = float(p('offset'))
        self.olat      = float(p('origin_lat'))
        self.olon      = float(p('origin_lon'))
        self.desired_speed         = min(5.0, float(p('desired_speed')))
        self.max_throttle          = min(1.0, float(p('max_throttle')))
        self.max_brake             = min(1.0, float(p('max_brake')))
        self.require_pacmod_enable = bool(p('require_pacmod_enable'))
        self.prediction_source     = str(p('prediction_source'))
        self.cone_topic            = str(p('cone_topic'))

        self.lat  = 0.0
        self.lon  = 0.0
        self.heading      = 0.0
        self.speed        = 0.0
        self.pacmod_enable    = False
        self.ped_trajectories = None
        self._pacmod_primed   = False
        self._v_cmd = 0.0
        self._has_ins_heading    = False  # set True once /insnavgeod fires
        self._has_valid_heading  = False  # set True when any heading source is ready
        self._gps_hdg_anchor     = None   # (lat, lon) of last GPS heading update

        # ------------------------------------------------------------------ #
        #  MPPI + helpers                                                      #
        # ------------------------------------------------------------------ #
        device_param = str(p('mppi.device')).strip() or None
        self.mppi = MPPI(
            K=int(p('mppi.K')),
            H=int(p('mppi.H')),
            dt=float(p('mppi.dt')),
            sigma_steer=float(p('mppi.sigma_steer')),
            sigma_accel=float(p('mppi.sigma_accel')),
            lam=float(p('mppi.lambda_')),
            v_ref=self.desired_speed,
            w_pos=float(p('mppi.w_pos')),
            w_vel=float(p('mppi.w_vel')),
            w_curv=float(p('mppi.w_curv')),
            w_obs=float(p('mppi.w_obs')),
            w_obs_hard=float(p('mppi.w_obs_hard')),
            w_obs_soft=float(p('mppi.w_obs_soft')),
            w_cone_hard=float(p('mppi.w_cone_hard')),
            w_cone_soft=float(p('mppi.w_cone_soft')),
            w_terminal=float(p('mppi.w_terminal')),
            ped_sigma=float(p('mppi.ped_sigma')),
            cone_radius=float(p('mppi.cone_radius')),
            clearance=float(p('mppi.clearance')),
            wheelbase=self.wheelbase,
            device=device_param,
        )
        self._log_device()

        self.pid_speed = PID(
            kp=float(p('pid.kp')), ki=float(p('pid.ki')),
            kd=float(p('pid.kd')), wg=float(p('pid.wg')),
        )
        self.speed_filter = OnlineFilter(
            cutoff=float(p('filter.cutoff')),
            fs=float(p('filter.fs')),
            order=int(p('filter.order')),
        )

        # Start with no reference path; wait for /goal_pose
        self.ref_path = None
        self.obstacles = np.zeros((0, 2))
        self.cones = np.zeros((0, 2))

        # ------------------------------------------------------------------ #
        #  Subscribers                                                         #
        # ------------------------------------------------------------------ #
        self.create_subscription(NavSatFix, '/navsatfix', self._gnss_cb, 10)
        if _HAS_SEPTENTRIO:
            self.create_subscription(INSNavGeod, '/insnavgeod', self._ins_cb, 10)
        else:
            self.get_logger().warn(
                'septentrio_gnss_driver not found — /insnavgeod heading unavailable'
            )
        self.create_subscription(Bool, '/pacmod/enabled', self._enable_cb, 10)
        self.create_subscription(
            VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt', self._speed_cb, 10
        )
        if self.prediction_source == 'predicted':
            self.create_subscription(
                Float32MultiArray, '/pedestrian_predictions_tensor',
                self._pred_tensor_cb, 10,
            )
            self.get_logger().info(
                'Obstacle source: /pedestrian_predictions_tensor (full trajectories)'
            )
        else:
            self.create_subscription(
                Int32MultiArray, '/fusion_pedestrian_position', self._ped_cb, 10,
            )
            self.get_logger().info(
                'Obstacle source: /fusion_pedestrian_position (raw detections)'
            )
        self.create_subscription(PoseArray, self.cone_topic, self._cones_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)
        self.get_logger().info(f'Cone source: {self.cone_topic}')

        # ------------------------------------------------------------------ #
        #  Publishers — control                                                #
        # ------------------------------------------------------------------ #
        self.global_pub = self.create_publisher(GlobalCmd,         '/pacmod/global_cmd',   10)
        self.gear_pub   = self.create_publisher(SystemCmdInt,      '/pacmod/shift_cmd',    10)
        self.brake_pub  = self.create_publisher(SystemCmdFloat,    '/pacmod/brake_cmd',    10)
        self.accel_pub  = self.create_publisher(SystemCmdFloat,    '/pacmod/accel_cmd',    10)
        self.turn_pub   = self.create_publisher(SystemCmdInt,      '/pacmod/turn_cmd',     10)
        self.steer_pub  = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 10)

        self.global_cmd = GlobalCmd(enable=False, clear_override=True)
        self.gear_cmd   = SystemCmdInt(command=2)
        self.brake_cmd  = SystemCmdFloat(command=0.0)
        self.accel_cmd  = SystemCmdFloat(command=0.0)
        self.turn_cmd   = SystemCmdInt(command=1)
        self.steer_cmd  = PositionWithSpeed(angular_position=0.0, angular_velocity_limit=4.0)

        # ------------------------------------------------------------------ #
        #  Visualisation                                                        #
        # ------------------------------------------------------------------ #
        self.viz = MPPIVisualizer(self, str(p('viz.frame_id')), int(p('viz.num_samples')))
        if self.ref_path is not None:
            self.viz.publish_static(self.ref_path)

        # ------------------------------------------------------------------ #
        #  Timer                                                               #
        # ------------------------------------------------------------------ #
        self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(
            f'adapt_mppi_node ready — {self.rate_hz:.1f} Hz, '
            f'v_ref={self.desired_speed:.1f} m/s'
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
                self.get_logger().info(
                    f'MPPI device: {dev} ({name}, {total_gb:.1f} GiB VRAM)'
                )
            else:
                self.get_logger().info(f'MPPI device: {dev} (CPU)')
        except Exception as exc:
            self.get_logger().warn(f'MPPI device log failed: {exc}')

    # ---------------------------------------------------------------------- #
    #  Callbacks                                                               #
    # ---------------------------------------------------------------------- #

    def _gnss_cb(self, msg: NavSatFix):
        new_lat, new_lon = msg.latitude, msg.longitude
        if not self._has_ins_heading:
            # Set anchor once we have a valid previous GPS fix
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
                    # ENU angle → compass heading (0=N, CW+), 2 m baseline
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

    def _enable_cb(self, msg: Bool):
        self.pacmod_enable = msg.data

    def _speed_cb(self, msg: VehicleSpeedRpt):
        self.speed = float(self.speed_filter.get_data(msg.vehicle_speed))

    def _ped_cb(self, msg: Int32MultiArray):
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

    def _pred_tensor_cb(self, msg: Float32MultiArray):
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

    def _cones_cb(self, msg: PoseArray):
        if not msg.poses:
            return
        self.cones = np.array(
            [[pose.position.x, pose.position.y] for pose in msg.poses],
            dtype=np.float32,
        )

    def _goal_cb(self, msg: PoseStamped):
        """Update reference path to just the goal waypoint."""
        gx, gy = msg.pose.position.x, msg.pose.position.y
        # ReferencePath needs >= 2 points; use the goal twice.
        pts = np.array([[gx, gy], [gx, gy]], dtype=np.float32)
        
        self.ref_path = ReferencePath(pts)
        self.get_logger().info(f'New goal received: ({gx:.2f}, {gy:.2f}). Reference path set to waypoint.')
        # Also update the static viz
        self.viz.publish_static(self.ref_path)

    # ---------------------------------------------------------------------- #
    #  Control loop                                                            #
    # ---------------------------------------------------------------------- #

    def _gem_state(self):
        """Return (x, y, yaw) in ENU, antenna-offset corrected."""
        local_x, local_y, _ = geodetic2enu(
            self.lat, self.lon, 0.0, self.olat, self.olon, 0.0
        )
        yaw = heading_to_yaw(self.heading)
        return (
            local_x - self.offset * math.cos(yaw),
            local_y - self.offset * math.sin(yaw),
            yaw,
        )

    def _prime_pacmod(self):
        self.global_cmd.enable = True
        self.global_cmd.clear_override = True
        self.global_pub.publish(self.global_cmd)
        self.gear_cmd.command = 3
        self.gear_pub.publish(self.gear_cmd)
        self.brake_cmd.command = 0.0
        self.brake_pub.publish(self.brake_cmd)
        self.accel_cmd.command = 0.0
        self.accel_pub.publish(self.accel_cmd)
        self.turn_cmd.command = 1
        self.turn_pub.publish(self.turn_cmd)
        self._pacmod_primed = True
        self.get_logger().warn('PACMod primed: enable + FORWARD')

    def _control_loop(self):
        # if self.require_pacmod_enable and not self.pacmod_enable:
        #     print(f"Missing pacmode")
        #     return
        if self.lat == 0.0 and self.lon == 0.0:
            print(f"Missing gps")
            return
        if not self._has_valid_heading:
            print(f"Missing heading")
            return
        if not self._pacmod_primed:
            self._prime_pacmod()

        if self.ref_path is None:
            # Optionally log this every few seconds if needed
            return

        x, y, yaw = self._gem_state()
        state = np.array([x, y, yaw, max(self.speed, 0.0)], dtype=float)
        stamp = self.get_clock().now().to_msg()

        self.viz.append_robot_pose(x, y, yaw, stamp)

        #TODO: Check trim logic. Check the ref path frequency.
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
        self.steer_cmd.angular_position = math.radians(sw_deg)

        self._v_cmd = max(
            0.0,
            min(self._v_cmd + accel * (1.0 / self.rate_hz), self.desired_speed),
        )
        now = self.get_clock().now().nanoseconds * 1e-9
        speed_err = self._v_cmd - self.speed
        if abs(speed_err) < 0.05:
            speed_err = 0.0
        pid_out = self.pid_speed.get_control(now, speed_err)
        if pid_out >= 0.0:
            self.accel_cmd.command = min(pid_out, self.max_throttle)
            self.brake_cmd.command = 0.0
        else:
            self.accel_cmd.command = 0.0
            self.brake_cmd.command = min(abs(pid_out), self.max_brake)
        

        # Disable actual drive commands for now
        if _DISABLE_DRIVE_COMMANDS:
            self.accel_cmd.command = 0.0
            self.brake_cmd.command = 0.0
            self.steer_cmd.angular_position = 0.0
        else:
            self.steer_pub.publish(self.steer_cmd)
            self.accel_pub.publish(self.accel_cmd)
            self.brake_pub.publish(self.brake_cmd)
            
        self.global_cmd.enable = True
        self.global_pub.publish(self.global_cmd)

        ess = self.mppi.effective_sample_count()
        self.get_logger().info(
            f'Ref path {active_path}'
            f'MPPI | pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.1f}deg '
            f'v={self.speed:.2f}→{self._v_cmd:.2f} '
            f'thr={self.accel_cmd.command:.2f} brk={self.brake_cmd.command:.2f} '
            f'sw={sw_deg:.1f}deg obs={len(self.obstacles)} ESS/K={ess/self.mppi.K:.2f}',
            throttle_duration_sec=1.0,
        )

        self.viz.publish(self.mppi, self.obstacles, stamp, accel=accel, delta=delta, state=state)


# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = AdaptMPPINode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if _name_ == '_main_':
    main()