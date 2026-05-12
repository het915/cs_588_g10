#!/usr/bin/env python3
"""Fake pedestrian + goal + Gazebo-state publisher for the sim variant
of adapt_mppi_node_ros1_sim.

The sim node reads ego state from /gazebo/model_states (Gazebo ground
truth). When real Gazebo isn't running, this spoof also emits a fake
ModelStates so the node has a state to use:

  /gazebo/model_states  gazebo_msgs/ModelStates   (--no-gazebo to skip)
      One entry named --model-name (default 'gem_e4') at pose
      (--ego-x, --ego-y, yaw=--ego-yaw), twist (--ego-speed forward,
      0 angular). Published at --gazebo-rate Hz (default 50).
      The sim node records the FIRST received pose as map (0, 0); any
      subsequent ego pose is reported relative to that. So leaving the
      defaults (0, 0, 0) gives a stationary spoof car at map origin.

  /fusion_pedestrian_tensor  std_msgs/Float32MultiArray
      (M, H, 2) row-major (x_fwd_m, y_left_m) in base_link. One fake
      pedestrian (M=1), H future poses sampled at --horizon-dt s. The
      pedestrian walks ±--obstacle-sweep-amplitude m sideways across
      the vehicle path over the horizon (sine wave). Defaults match the
      diffusion contract: H = 20, dt = 0.25 s.

Republished every --goal-period s (default 60 s):
  /move_base_simple/goal     geometry_msgs/PoseStamped   (frame --goal-frame)

For native Foxglove rendering of the trajectories — the
Float32MultiArray is opaque to Foxglove — the same H poses are also
published as:
  /spoof/viz/ped_trajectories   visualization_msgs/MarkerArray
                                (LINE_STRIP + SPHERE_LIST in
                                --viz-frame, default 'base_footprint',
                                so the markers follow the gazebo car
                                via the map -> base_footprint TF the
                                sim node broadcasts.)

Usage:
  python3 publish_fake_gps.py
  python3 publish_fake_gps.py --goal-x 15 --goal-y 0
  python3 publish_fake_gps.py --no-obstacle --rate 5
"""
import argparse
import math

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # NOTE: --vN spawn-pose presets are defined further below; they
    # override --ego-x/--ego-y/--ego-yaw after parsing.
    p.add_argument('--rate', type=float, default=10.0,
                   help='Ped publish rate (Hz). Default 10.')
    # --- Goal pose ----------------------------------------------------------
    p.add_argument('--goal-x', type=float, default=15.0,
                   help='Goal pose x in --goal-frame (m). Default 15 m forward '
                        'of the gazebo spawn (which the sim node treats as map '
                        'origin).')
    p.add_argument('--goal-y', type=float, default=0.0,
                   help='Goal pose y in --goal-frame (m). Default 0.')
    p.add_argument('--goal-yaw', type=float, default=0.0,
                   help='Goal pose yaw (rad). Default 0. Cosmetic — MPPI '
                        'only uses position.')
    p.add_argument('--goal-frame', type=str, default='map',
                   help='Frame for /move_base_simple/goal. Default map.')
    p.add_argument('--goal-period', type=float, default=60.0,
                   help='Seconds between /move_base_simple/goal republishes. '
                        'Default 60.')
    p.add_argument('--no-goal', action='store_true',
                   help='Skip publishing the goal pose.')
    # --- Pedestrian trajectory ---------------------------------------------
    p.add_argument('--obstacle-distance', type=float, default=5.5,
                   help='Fake pedestrian distance ahead in base_link (m). '
                        'Default 5.5.')
    p.add_argument('--obstacle-bearing', type=float, default=0.0,
                   help='Fake pedestrian baseline bearing in base_link (deg, '
                        '0 = straight ahead, +left). Default 0.')
    p.add_argument('--obstacle-sweep-amplitude', type=float, default=7.0,
                   help='Lateral sweep amplitude (m) — the pedestrian walks '
                        'sideways across the vehicle path with this peak '
                        'offset (sine wave). Set 0 for a stationary obstacle.')
    p.add_argument('--obstacle-sweep-period', type=float, default=16.0,
                   help='Lateral sweep period (s). Default 16.')
    p.add_argument('--horizon', type=int, default=20,
                   help='Number of future poses (H) per pedestrian trajectory. '
                        'Default 20 matches the diffusion predictor contract.')
    p.add_argument('--horizon-dt', type=float, default=0.25,
                   help='Time step (s) between successive poses in the '
                        'trajectory. Default 0.25 matches MPPI._PED_DT.')
    p.add_argument('--no-obstacle', action='store_true',
                   help='Skip publishing the fake pedestrian.')
    # --- Fake Gazebo ground truth ------------------------------------------
    p.add_argument('--no-gazebo', action='store_true',
                   help='Skip publishing /gazebo/model_states (use only if '
                        'real Gazebo is running and you want its real state).')
    p.add_argument('--model-name', type=str, default='gem_e4',
                   help='Gazebo model name the sim node looks up. Must match '
                        '~gazebo_model_name on the controller. Default gem_e4.')
    p.add_argument('--gazebo-rate', type=float, default=50.0,
                   help='/gazebo/model_states publish rate (Hz). Default 50, '
                        'matches typical Gazebo physics step.')
    p.add_argument('--ego-x', type=float, default=0.0,
                   help='Fake ego x in Gazebo world frame (m). Default 0. '
                        'The sim node treats the first received pose as map '
                        'origin, so keeping this at 0 makes the car sit at '
                        '(0, 0) in map.')
    p.add_argument('--ego-y', type=float, default=0.0,
                   help='Fake ego y in Gazebo world frame (m). Default 0.')
    p.add_argument('--ego-yaw', type=float, default=0.0,
                   help='Fake ego yaw (rad). Default 0 (facing +x).')
    p.add_argument('--ego-speed', type=float, default=0.0,
                   help='Fake forward speed (m/s) reported in twist.linear.x. '
                        'Default 0 = stationary. Position stays fixed — only '
                        'the twist field is exercised.')
    # --- Viz ----------------------------------------------------------------
    p.add_argument('--viz-frame', type=str, default='base_footprint',
                   help='Frame for the marker mirror of /fusion_pedestrian_tensor. '
                        'Default base_footprint — the sim node broadcasts '
                        'map -> base_footprint, so markers in this frame '
                        'follow the gazebo car automatically.')
    # --- Spawn-pose presets (exercise the map-yaw rebase in the sim node) --
    # Each --vN overrides --ego-x/--ego-y/--ego-yaw with a different
    # Gazebo spawn pose. The goal stays at (--goal-x, --goal-y) in map, so
    # all variants should drive forward the same way if the sim node's map
    # rebase is correct.
    variants = p.add_mutually_exclusive_group()
    variants.add_argument('--v1', dest='variant', action='store_const', const=1,
                          help='Spawn (0, 0, yaw=0)  — baseline, Gazebo +x.')
    variants.add_argument('--v2', dest='variant', action='store_const', const=2,
                          help='Spawn (8, 4, yaw=+90deg) — translated + rotated left.')
    variants.add_argument('--v3', dest='variant', action='store_const', const=3,
                          help='Spawn (-6, -3, yaw=180deg) — translated + facing -x.')
    p.set_defaults(variant=1)
    args = p.parse_args()

    # Apply the preset only to ego-x/y/yaw values the user did NOT pass
    # explicitly — so e.g. `--v2 --ego-yaw 0` keeps the yaw at 0.
    import sys as _sys
    raw = _sys.argv[1:]
    explicit = {
        'ego_x':   any(a == '--ego-x'   or a.startswith('--ego-x=')   for a in raw),
        'ego_y':   any(a == '--ego-y'   or a.startswith('--ego-y=')   for a in raw),
        'ego_yaw': any(a == '--ego-yaw' or a.startswith('--ego-yaw=') for a in raw),
    }
    presets = {
        2: (8.0,  4.0,  math.pi / 2.0),
        3: (-6.0, -3.0, math.pi),
    }
    if args.variant in presets:
        px, py, pyaw = presets[args.variant]
        if not explicit['ego_x']:   args.ego_x   = px
        if not explicit['ego_y']:   args.ego_y   = py
        if not explicit['ego_yaw']: args.ego_yaw = pyaw
    return args


def main():
    args = parse_args()

    rospy.init_node('fake_inputs_publisher', anonymous=True)
    goal_pub    = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
    ped_pub     = rospy.Publisher('/fusion_pedestrian_tensor',
                                  Float32MultiArray, queue_size=10)
    ped_viz_pub = rospy.Publisher('/spoof/viz/ped_trajectories',
                                  MarkerArray, queue_size=10)
    gz_pub = None
    if not args.no_gazebo:
        gz_pub = rospy.Publisher('/gazebo/model_states', ModelStates,
                                 queue_size=10)

    # --- Goal pose --------------------------------------------------------
    goal = PoseStamped()
    goal.header.frame_id = args.goal_frame
    goal.pose.position.x = float(args.goal_x)
    goal.pose.position.y = float(args.goal_y)
    half = args.goal_yaw / 2.0
    goal.pose.orientation.z = math.sin(half)
    goal.pose.orientation.w = math.cos(half)

    # --- Fake /gazebo/model_states message --------------------------------
    # The sim node looks up the entry by --model-name, reads pose.position
    # (xy) and pose.orientation (quat -> yaw), and the magnitude of
    # twist.linear (filtered) for speed.
    gz_msg = ModelStates()
    gz_msg.name   = [args.model_name]
    ego_pose = Pose()
    ego_pose.position.x  = float(args.ego_x)
    ego_pose.position.y  = float(args.ego_y)
    ego_pose.position.z  = 0.0
    half_yaw = float(args.ego_yaw) / 2.0
    ego_pose.orientation.z = math.sin(half_yaw)
    ego_pose.orientation.w = math.cos(half_yaw)
    ego_twist = Twist()
    ego_twist.linear.x = float(args.ego_speed)
    gz_msg.pose   = [ego_pose]
    gz_msg.twist  = [ego_twist]

    # --- Ped trajectory: layout.dim = [M, H, xy] --------------------------
    H = max(1, int(args.horizon))
    M = 1
    ped_msg = Float32MultiArray()
    ped_msg.layout.dim = [
        MultiArrayDimension(label='M',  size=M, stride=M * H * 2),
        MultiArrayDimension(label='H',  size=H, stride=H * 2),
        MultiArrayDimension(label='xy', size=2, stride=2),
    ]

    def _ped_traj_at(t_sec):
        """(H, 2) base_link pairs (x_fwd_m, y_left_m), sampled forward in
        time at --horizon-dt. Returns (flat, pairs)."""
        x_fwd = float(args.obstacle_distance)
        baseline_y = x_fwd * math.tan(math.radians(args.obstacle_bearing))
        amp, period = args.obstacle_sweep_amplitude, args.obstacle_sweep_period
        pairs = []
        for h in range(H):
            t_h = t_sec + h * args.horizon_dt
            if amp > 0 and period > 0:
                y_lat = baseline_y + amp * math.sin(2.0 * math.pi * t_h / period)
            else:
                y_lat = baseline_y
            pairs.append((x_fwd, y_lat))
        flat = [v for xy in pairs for v in xy]
        return flat, pairs

    def _build_ped_markers(pairs, stamp):
        msg = MarkerArray()
        clear = Marker()
        clear.header.frame_id = args.viz_frame
        clear.header.stamp    = stamp
        clear.action          = Marker.DELETEALL
        msg.markers.append(clear)

        line = Marker()
        line.header.frame_id = args.viz_frame
        line.header.stamp    = stamp
        line.ns, line.id     = 'spoof_ped_traj', 1
        line.type            = Marker.LINE_STRIP
        line.action          = Marker.ADD
        line.scale.x         = 0.15
        line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.2, 0.8, 0.9
        line.pose.orientation.w = 1.0
        for x, y in pairs:
            line.points.append(Point(x=float(x), y=float(y), z=0.05))
        msg.markers.append(line)

        spheres = Marker()
        spheres.header.frame_id = args.viz_frame
        spheres.header.stamp    = stamp
        spheres.ns, spheres.id  = 'spoof_ped_traj', 2
        spheres.type            = Marker.SPHERE_LIST
        spheres.action          = Marker.ADD
        spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.25
        spheres.color.r, spheres.color.g, spheres.color.b, spheres.color.a = 1.0, 0.2, 0.8, 1.0
        spheres.pose.orientation.w = 1.0
        for x, y in pairs:
            spheres.points.append(Point(x=float(x), y=float(y), z=0.1))
        msg.markers.append(spheres)
        return msg

    # --- Logs -------------------------------------------------------------
    if gz_pub is not None:
        rospy.loginfo(
            f'Gazebo spoof: model={args.model_name!r} '
            f'pose=({args.ego_x:.2f}, {args.ego_y:.2f}, yaw={args.ego_yaw:.2f}) '
            f'twist.linear.x={args.ego_speed:.2f} m/s at '
            f'{args.gazebo_rate:.1f} Hz on /gazebo/model_states'
        )
    if not args.no_goal:
        rospy.loginfo(
            f'Goal: ({args.goal_x:.2f}, {args.goal_y:.2f}) m in '
            f'{args.goal_frame} — republished every {args.goal_period:.1f} s '
            f'on /move_base_simple/goal'
        )
    if not args.no_obstacle:
        amp = args.obstacle_sweep_amplitude
        if amp > 0:
            rospy.loginfo(
                f'Obstacle: {args.obstacle_distance:.1f} m forward in '
                f'base_link, sweeping ±{amp:.1f} m sideways '
                f'(period {args.obstacle_sweep_period:.1f} s), H={H} poses '
                f'@ dt={args.horizon_dt:.2f} s, at {args.rate:.1f} Hz on '
                f'/fusion_pedestrian_tensor'
            )
        else:
            rospy.loginfo(
                f'Obstacle: {args.obstacle_distance:.1f} m, '
                f'{args.obstacle_bearing:.1f} deg bearing (base_link, '
                f'stationary), H={H} poses at {args.rate:.1f} Hz on '
                f'/fusion_pedestrian_tensor'
            )

    # --- Loop -------------------------------------------------------------
    # /gazebo/model_states is the high-rate signal (default 50 Hz); peds run
    # at --rate (10 Hz). The loop ticks at the faster of the two and emits
    # the slower stream on a fixed period.
    loop_rate = max(args.rate, args.gazebo_rate if gz_pub is not None else 0.0)
    rate = rospy.Rate(loop_rate)
    ped_period = 1.0 / max(args.rate, 1e-3)
    last_goal_pub = None
    last_ped_pub  = None
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        now = rospy.Time.now()

        if gz_pub is not None:
            gz_pub.publish(gz_msg)

        if not args.no_obstacle:
            if last_ped_pub is None or \
               (now - last_ped_pub).to_sec() >= ped_period:
                flat, pairs = _ped_traj_at((now - t0).to_sec())
                ped_msg.data = flat
                ped_pub.publish(ped_msg)
                ped_viz_pub.publish(_build_ped_markers(pairs, now))
                last_ped_pub = now

        if not args.no_goal:
            if last_goal_pub is None or \
               (now - last_goal_pub).to_sec() >= args.goal_period:
                goal.header.stamp = now
                goal_pub.publish(goal)
                last_goal_pub = now

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
