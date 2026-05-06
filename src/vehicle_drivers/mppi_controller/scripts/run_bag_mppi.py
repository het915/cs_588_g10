#!/usr/bin/env python3
"""Rosbag MPPI test harness.

Replays a .mcap bag, runs adapt_mppi_node against the recorded sensor
data, and logs every planned trajectory and control command.

Usage
-----
    python3 scripts/run_bag_mppi.py <bag.mcap> [options]

    --speed FLOAT          Cruise speed m/s            (default: 2.0)
    --prediction-source    raw | predicted              (default: raw)
    --waypoints PATH       Override waypoints CSV path
    --loop                 Loop the bag
    --no-rviz              Skip RViz
    --plot                 Save bag_test_result.png

Notes
-----
* ros2 bag play is started with --clock so adapt_mppi_node runs on sim time.
* require_pacmod_enable is forced false so MPPI runs without /pacmod/enabled.
* The observer node re-publishes the chosen trajectory on
  /adapt/test/planned_trajectory (nav_msgs/Path) for external RViz sessions.
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from pacmod2_msgs.msg import SystemCmdFloat, PositionWithSpeed
from visualization_msgs.msg import MarkerArray


# Steering-wheel → front-wheel (inverse of adapt's front2steer).
# We just keep the raw steering-wheel angle for logging; marking it as [sw].
_MAX_SW_DEG = 450.0


class MPPIBagObserver(Node):
    """Subscribe to MPPI output topics, log live, re-publish planned path."""

    def __init__(self):
        super().__init__('mppi_bag_observer')

        self._latest_accel: float = 0.0
        self._latest_steer_sw: float = 0.0  # steering-wheel rad

        self.ticks: int = 0
        self.accels: list[float] = []
        self.steers_sw: list[float] = []   # steering-wheel rad
        self.arcs: list[float] = []        # planned arc length per tick
        self.trajectories: list[np.ndarray] = []  # (H, 2) per tick
        self.obstacle_history: list[np.ndarray] = []  # (M, 2) per obstacle update

        # Subscriptions
        self.create_subscription(
            Path, '/adapt/viz/chosen_trajectory', self._traj_cb, 10
        )
        self.create_subscription(
            SystemCmdFloat, '/pacmod/accel_cmd', self._accel_cb, 10
        )
        self.create_subscription(
            PositionWithSpeed, '/pacmod/steering_cmd', self._steer_cb, 10
        )
        self.create_subscription(
            MarkerArray, '/adapt/viz/obstacles', self._obs_cb, 10
        )

        # Re-publish planned trajectory so it's visible in external RViz sessions
        self._traj_pub = self.create_publisher(
            Path, '/adapt/test/planned_trajectory', 10
        )

    # ------------------------------------------------------------------ #

    def _accel_cb(self, msg: SystemCmdFloat):
        self._latest_accel = float(msg.command)

    def _steer_cb(self, msg: PositionWithSpeed):
        self._latest_steer_sw = float(msg.angular_position)

    def _obs_cb(self, msg: MarkerArray):
        pts = [
            (m.pose.position.x, m.pose.position.y)
            for m in msg.markers
            if m.action == 0  # Marker.ADD; skip DELETEALL (action=3)
        ]
        if pts:
            self.obstacle_history.append(np.array(pts, dtype=np.float64))

    def _traj_cb(self, msg: Path):
        if not msg.poses:
            return

        # Record
        pts = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float64,
        )
        arc = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        self.ticks += 1
        self.accels.append(self._latest_accel)
        self.steers_sw.append(self._latest_steer_sw)
        self.arcs.append(arc)
        self.trajectories.append(pts)

        steer_deg = math.degrees(self._latest_steer_sw)
        print(
            f'[{self.ticks:04d}] accel={self._latest_accel:+.3f}  '
            f'steer={steer_deg:+6.1f}deg[sw]  '
            f'horizon={len(pts)}pts  arc={arc:.2f}m',
            flush=True,
        )

        # Re-publish with updated stamp
        out = Path()
        out.header = msg.header
        out.header.stamp = self.get_clock().now().to_msg()
        out.poses = msg.poses
        self._traj_pub.publish(out)

    # ------------------------------------------------------------------ #

    def print_summary(self):
        if not self.ticks:
            print('\nNo data recorded — MPPI may not have produced output.')
            return

        a = np.array(self.accels)
        s = np.abs(np.array(self.steers_sw))
        arc = np.array(self.arcs)

        print(f'\n{"=" * 48}')
        print(f'  MPPI Bag Test — {self.ticks} control ticks')
        print(f'{"=" * 48}')
        print(f'  Accel  : mean={a.mean():+.3f}  max={a.max():+.3f}  '
              f'min={a.min():+.3f}  m/s²')
        print(f'  Steer  : mean={math.degrees(s.mean()):.1f}°  '
              f'max={math.degrees(s.max()):.1f}°  (abs, steering-wheel)')
        print(f'  Horizon: mean arc={arc.mean():.2f} m/plan  '
              f'min={arc.min():.2f}  max={arc.max():.2f}')
        print(f'{"=" * 48}\n')

    def save_plot(self, out_path: str):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # --- Left: planned trajectories in ENU ---
            ax = axes[0]
            cmap = cm.get_cmap('plasma', max(len(self.trajectories), 1))

            print(f"Number of trajectories: {len(self.trajectories)}")

            for i, traj in enumerate(self.trajectories):
                ax.plot(traj[:, 0], traj[:, 1],
                        color=cmap(i / len(self.trajectories)),
                        alpha=0.4, linewidth=0.8)

            # Actual driven path — first point of each planned trajectory
            if self.trajectories:
                robot_xy = np.array([t[0] for t in self.trajectories])
                ax.plot(robot_xy[:, 0], robot_xy[:, 1],
                        color='black', linestyle='--', linewidth=1.5,
                        label='driven path', zorder=5)
                ax.scatter(robot_xy[:, 0], robot_xy[:, 1],
                           s=20, color='black', zorder=6)

            # Pedestrian risk map — density heatmap + scatter
            if self.obstacle_history:
                all_pts = np.vstack(self.obstacle_history)  # (N_total, 2)
                # 2-D histogram as a background heatmap
                xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
                ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
                margin = 1.0
                H_map, xedges, yedges = np.histogram2d(
                    all_pts[:, 0], all_pts[:, 1], bins=40,
                    range=[[xmin - margin, xmax + margin],
                           [ymin - margin, ymax + margin]],
                )
                # Smooth the histogram for a nicer heatmap
                try:
                    from scipy.ndimage import gaussian_filter
                    H_map = gaussian_filter(H_map, sigma=1.5)
                except ImportError:
                    pass
                ax.imshow(
                    H_map.T,
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    origin='lower', cmap='YlOrRd', alpha=0.35, aspect='auto',
                    zorder=2,
                )
                # Scatter dots coloured by time so clusters are visible
                tick_idx = np.concatenate([
                    np.full(len(obs), i) for i, obs in enumerate(self.obstacle_history)
                ])
                ax.scatter(all_pts[:, 0], all_pts[:, 1],
                           c=tick_idx, cmap='autumn_r',
                           s=18, alpha=0.7, zorder=3, label='pedestrians')

            ax.legend(fontsize=7, loc='upper left')
            ax.set_aspect('equal')
            ax.set_title('Planned trajectories (ENU)')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')

            # --- Right: commands over time ---
            ax = axes[1]
            ticks = np.arange(self.ticks)
            ax.plot(ticks, self.accels, label='accel [m/s²]')
            ax.plot(ticks,
                    [math.degrees(s) for s in self.steers_sw],
                    label='steer [deg sw]')
            ax.axhline(0, color='k', linewidth=0.5)
            ax.set_title('Control commands')
            ax.set_xlabel('tick')
            ax.legend()

            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            print(f'Plot saved → {out_path}')
        except Exception as exc:
            print(f'(plotting skipped: {exc})')


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _start(cmd: list[str], name: str) -> subprocess.Popen:
    print(f'[run_bag_mppi] starting {name}: {" ".join(cmd)}', flush=True)
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def _extract_origin_from_bag(bag_path: str) -> tuple[float, float] | None:
    """Read first valid GPS fix from bag to use as ENU origin."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import NavSatFix
    except ImportError:
        return None

    try:
        reader = rosbag2_py.SequentialReader()
        if os.path.isdir(bag_path):
            uri, storage_id = bag_path, ''
        else:
            uri, storage_id = os.path.dirname(bag_path), 'mcap'
        reader.open(
            rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
            rosbag2_py.ConverterOptions('', ''),
        )
        reader.set_filter(rosbag2_py.StorageFilter(topics=['/navsatfix']))

        while reader.has_next():
            _, data, _ = reader.read_next()
            msg = deserialize_message(data, NavSatFix)
            if msg.latitude != 0.0 or msg.longitude != 0.0:
                return msg.latitude, msg.longitude
        return None
    except Exception:
        return None


def _extract_waypoints_from_bag(bag_path: str) -> str | None:
    """Read /navsatfix from bag → temp lon,lat CSV. Returns path or None."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import NavSatFix
    except ImportError as exc:
        print(f'[run_bag_mppi] cannot extract bag waypoints: {exc}')
        return None

    try:
        reader = rosbag2_py.SequentialReader()
        if os.path.isdir(bag_path):
            uri, storage_id = bag_path, ''
        else:
            uri, storage_id = os.path.dirname(bag_path), 'mcap'
        reader.open(
            rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
            rosbag2_py.ConverterOptions('', ''),
        )
        reader.set_filter(rosbag2_py.StorageFilter(topics=['/navsatfix']))

        _MIN_SEP = 5e-6   # ~0.5 m in degrees
        pts, prev = [], None
        while reader.has_next():
            _, data, _ = reader.read_next()
            msg = deserialize_message(data, NavSatFix)
            if msg.latitude == 0.0 and msg.longitude == 0.0:
                continue
            if prev is None or (
                abs(msg.latitude - prev[1]) + abs(msg.longitude - prev[0]) > _MIN_SEP
            ):
                pts.append((msg.longitude, msg.latitude))
                prev = (msg.longitude, msg.latitude)

        if len(pts) < 2:
            print('[run_bag_mppi] too few GPS points in bag — using default waypoints')
            return None

        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='_bag_track.csv', delete=False
        )
        for lon, lat in pts:
            tmp.write(f'{lon},{lat}\n')
        tmp.close()
        print(f'[run_bag_mppi] bag GPS track: {len(pts)} waypoints → {tmp.name}')
        return tmp.name

    except Exception as exc:
        print(f'[run_bag_mppi] GPS extraction failed: {exc}')
        return None


def _rviz_config() -> str | None:
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('mppi_controller')
        cfg = os.path.join(share, 'rviz', 'adapt_main.rviz')
        return cfg if os.path.exists(cfg) else None
    except Exception:
        return None


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description='Replay a .mcap bag and test adapt_mppi_node.'
    )
    parser.add_argument('bag', help='Path to .mcap rosbag file or directory')
    parser.add_argument('--speed', type=float, default=2.0,
                        help='Desired cruise speed m/s (default: 2.0)')
    parser.add_argument('--prediction-source', default='raw',
                        choices=['raw', 'predicted'],
                        help='Obstacle source (default: raw)')
    parser.add_argument('--waypoints', default='',
                        help='Override waypoints CSV path')
    parser.add_argument('--loop', action='store_true',
                        help='Loop the bag')
    parser.add_argument('--no-rviz', action='store_true',
                        help='Skip launching RViz')
    parser.add_argument('--plot', action='store_true',
                        help='Save bag_test_result.png on exit')
    parser.add_argument('--cpu', action='store_true',
                        help='Force MPPI to run on CPU (bypasses CUDA compatibility issues)')
    args = parser.parse_args()

    bag_path = os.path.abspath(args.bag)
    if not os.path.exists(bag_path):
        sys.exit(f'ERROR: bag not found: {bag_path}')

    # Auto-extract GPS track from bag as waypoints (unless overridden)
    _tmp_waypoints = None
    if not args.waypoints:
        _tmp_waypoints = _extract_waypoints_from_bag(bag_path)
        if _tmp_waypoints:
            args.waypoints = _tmp_waypoints

    # Extract origin from bag (unless overridden by something else)
    origin = _extract_origin_from_bag(bag_path)
    if origin:
        print(f'[run_bag_mppi] using bag origin: {origin[0]}, {origin[1]}')

    # ---- Build subprocess commands ----

    bag_cmd = ['ros2', 'bag', 'play', bag_path, '--clock']
    if args.loop:
        bag_cmd.append('--loop')

    mppi_params = [
        'use_sim_time:=true',
        'require_pacmod_enable:=false',
        f'desired_speed:={args.speed}',
        f'prediction_source:={args.prediction_source}',
    ]
    if origin:
        mppi_params.append(f'origin_lat:={origin[0]}')
        mppi_params.append(f'origin_lon:={origin[1]}')
    if args.waypoints:
        mppi_params.append(f'waypoints_csv:={args.waypoints}')
    if args.cpu:
        print('[run_bag_mppi] Force running MPPI on CPU')
        mppi_params.append('mppi.device:=cpu')

    mppi_cmd = [
        sys.executable, '-m', 'mppi_controller.adapt_mppi_node',
        '--ros-args',
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        _yaml = os.path.join(
            get_package_share_directory('mppi_controller'),
            'config', 'mppi_params.yaml',
        )
        if os.path.exists(_yaml):
            mppi_cmd += ['--params-file', _yaml]
        else:
            print(f'[run_bag_mppi] WARNING: params file not found at {_yaml}')
    except Exception as exc:
        print(f'[run_bag_mppi] WARNING: could not locate params file: {exc}')
    for p in mppi_params:
        mppi_cmd += ['-p', p]

    rviz_cmd = None
    if not args.no_rviz:
        cfg = _rviz_config()
        if cfg:
            rviz_cmd = ['ros2', 'run', 'rviz2', 'rviz2', '-d', cfg]
        else:
            print('[run_bag_mppi] rviz config not found — skipping RViz')

    # ---- Start processes ----

    rclpy.init()
    observer = MPPIBagObserver()

    # Spin observer in a background thread
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(observer,), daemon=True
    )
    spin_thread.start()

    processes: list[subprocess.Popen] = []

    def _cleanup(signum=None, frame=None):
        print('\n[run_bag_mppi] shutting down …', flush=True)
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        for p in processes:
            try:
                p.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                p.kill()

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        bag_proc = _start(bag_cmd, 'bag player')
        processes.append(bag_proc)

        # Small delay so the bag starts publishing before the node subscribes
        time.sleep(1.5)

        mppi_proc = _start(mppi_cmd, 'adapt_mppi_node')
        processes.append(mppi_proc)

        if rviz_cmd:
            rviz_proc = _start(rviz_cmd, 'rviz2')
            processes.append(rviz_proc)

        # Wait for bag to finish (or Ctrl+C)
        while bag_proc.poll() is None:
            time.sleep(0.2)

        # Let the last few messages flush through the observer
        print('[run_bag_mppi] bag finished — flushing …', flush=True)
        time.sleep(2.0)

    finally:
        _cleanup()
        if _tmp_waypoints and os.path.exists(_tmp_waypoints):
            os.unlink(_tmp_waypoints)
        observer.print_summary()

        if args.plot:
            plot_dir = (os.path.dirname(bag_path)
                        if os.path.isfile(bag_path) else os.getcwd())
            observer.save_plot(os.path.join(plot_dir, 'bag_test_result.png'))

        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
