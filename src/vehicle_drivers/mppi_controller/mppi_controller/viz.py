"""MPPI visualisation — all RViz publishers in one place.

MPPIVisualizer is constructed with a rclpy Node so it can create
publishers on that node's graph.  The node itself never touches a
viz publisher directly; it calls the three public methods:

    viz.publish_static(ref_path)            # once, at startup
    viz.append_robot_pose(x, y, yaw, stamp) # every control tick
    viz.publish(mppi, obstacles, stamp)     # every control tick
"""
import colorsys
import math

import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


_LATCHED = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class MPPIVisualizer:
    def __init__(self, node: Node, frame_id: str, num_samples: int):
        self._node      = node
        self.frame_id   = frame_id
        self._palette   = [
            colorsys.hsv_to_rgb(i / max(num_samples, 1), 0.45, 0.95)
            for i in range(num_samples)
        ]
        self._num_samples       = num_samples
        self._robot_traj_poses: list[PoseStamped] = []

        self._ref_pub    = node.create_publisher(Path,        '/adapt/viz/reference_path',       _LATCHED)
        self._goal_pub   = node.create_publisher(Marker,      '/adapt/viz/current_goal',         _LATCHED)
        self._chosen_pub = node.create_publisher(Path,        '/adapt/viz/chosen_trajectory',    10)
        self._samples_pub= node.create_publisher(MarkerArray, '/adapt/viz/sampled_trajectories', 10)
        self._obs_pub    = node.create_publisher(MarkerArray, '/adapt/viz/obstacles',            10)
        self._traj_pub   = node.create_publisher(Path,        '/adapt/viz/robot_trajectory',     10)

    # ---------------------------------------------------------------------- #
    #  Public API                                                              #
    # ---------------------------------------------------------------------- #

    def publish_static(self, ref_path) -> None:
        """Publish reference path and goal marker once (latched)."""
        stamp = self._node.get_clock().now().to_msg()
        self._pub_reference_path(ref_path, stamp)
        self._pub_goal_marker(ref_path, stamp)

    def append_robot_pose(self, x: float, y: float, yaw: float, stamp) -> None:
        """Accumulate one ego pose into the driven-trajectory buffer."""
        ps = PoseStamped()
        ps.header.frame_id = self.frame_id
        ps.header.stamp    = stamp
        ps.pose.position.x = x
        ps.pose.position.y = y
        half = yaw / 2.0
        ps.pose.orientation.z = math.sin(half)
        ps.pose.orientation.w = math.cos(half)
        self._robot_traj_poses.append(ps)

    def publish(self, mppi, obstacles: np.ndarray, stamp) -> None:
        """Publish all per-tick visualisation topics."""
        if mppi.last_traj is None:
            return
        self._pub_chosen_trajectory(mppi, stamp)
        self._pub_sampled_rollouts(mppi, stamp)
        self._pub_obstacle_markers(mppi, obstacles, stamp)
        self._pub_robot_trajectory(stamp)

    # ---------------------------------------------------------------------- #
    #  Private helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _pub_reference_path(self, ref_path, stamp) -> None:
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

    def _pub_goal_marker(self, ref_path, stamp) -> None:
        gx, gy = float(ref_path.xy[-1, 0]), float(ref_path.xy[-1, 1])
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp    = stamp
        m.ns              = 'current_goal'
        m.id              = 0
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position.x  = gx
        m.pose.position.y  = gy
        m.pose.position.z  = 0.5
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.2
        m.color.r = 1.0
        m.color.g = 0.4
        m.color.b = 0.0
        m.color.a = 1.0
        self._goal_pub.publish(m)

    def _pub_chosen_trajectory(self, mppi, stamp) -> None:
        mean_traj = mppi.last_mean_traj   # (H, 4)
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

    def _pub_sampled_rollouts(self, mppi, stamp) -> None:
        traj = mppi.last_traj      # (K, H, 4)
        w    = mppi.last_weights   # (K,)
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

    def _pub_obstacle_markers(self, mppi, obstacles: np.ndarray, stamp) -> None:
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
            m.color.r = 1.0
            m.color.g = 0.25
            m.color.b = 0.25
            m.color.a = 0.35
            msg.markers.append(m)

        self._obs_pub.publish(msg)

    def _pub_robot_trajectory(self, stamp) -> None:
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp    = stamp
        path.poses           = self._robot_traj_poses
        self._traj_pub.publish(path)
