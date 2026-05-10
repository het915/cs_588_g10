#!/usr/bin/env python3
"""Fake GPS / INS / Goal / Obstacle publisher for the sim variant of
adapt_mppi_node.

Streams at fixed rate (default 10 Hz, --rate):
  /navsatfix                   sensor_msgs/NavSatFix
  /insnavgeod                  septentrio_gnss_driver/INSNavGeod
  /fusion_pedestrian_position  std_msgs/Float32MultiArray  (one fake pedestrian
                                                          trajectory, M=1
                                                          obstacles x H poses x
                                                          2 (x_fwd, y_left) in
                                                          EGO frame. layout.dim
                                                          = [M, H, 2]. The
                                                          pedestrian walks
                                                          ±--obstacle-sweep-
                                                          amplitude m sideways
                                                          across vehicle path
                                                          over H future steps)

Republished every --goal-period s (default 60 s):
  /goal_pose       geometry_msgs/PoseStamped

Defaults are tuned for the project's UIUC GEM Highbay origin:
  lat = 40.0927422 N, lon = -88.2359639 W   (matches mppi_params.yaml)
  heading = 270 deg  (compass: facing WEST in ENU = -x direction)
  goal    = 15 m ahead of the vehicle along the heading

Heading -> ENU yaw uses the same convention as adapt_mppi_node_ros1_sim
(`heading_to_yaw`): compass 0=N, CW+; ENU yaw 0=+x, CCW+.

Usage:
  python3 publish_fake_gps.py
  python3 publish_fake_gps.py --heading 0 --goal-distance 25
  python3 publish_fake_gps.py --rate 20
"""
import argparse
import math

import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from septentrio_gnss_driver.msg import INSNavGeod


def heading_to_yaw(heading_deg):
    """Compass heading (deg, 0=N CW+) -> ENU yaw (rad, 0=+x CCW+).

    Matches utils.heading_to_yaw in the controller package.
    """
    if heading_deg < 270.0:
        return math.radians(90.0 - heading_deg)
    return math.radians(450.0 - heading_deg)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lat', type=float, default=40.0927422,
                   help='Fake latitude (deg). Default = UIUC Highbay.')
    p.add_argument('--lon', type=float, default=-88.2359639,
                   help='Fake longitude (deg). Default = UIUC Highbay.')
    p.add_argument('--alt', type=float, default=0.0, help='Altitude (m).')
    p.add_argument('--heading', type=float, default=270.0,
                   help='Compass heading (deg, 0=N, CW+). Default 270 = west.')
    p.add_argument('--rate', type=float, default=10.0,
                   help='GPS/INS publish rate (Hz).')
    p.add_argument('--frame', type=str, default='map', help='Header frame_id.')
    p.add_argument('--goal-distance', type=float, default=15.0,
                   help='Goal pose distance ahead of vehicle (m).')
    p.add_argument('--goal-frame', type=str, default='map',
                   help='Frame for /goal_pose.')
    p.add_argument('--goal-period', type=float, default=60.0,
                   help='Seconds between /goal_pose republishes. '
                        'Default 60.')
    p.add_argument('--no-goal', action='store_true',
                   help='Skip publishing the goal pose.')
    p.add_argument('--obstacle-distance', type=float, default=7.0,
                   help='Fake pedestrian distance ahead in EGO frame (m). '
                        'Default 7.')
    p.add_argument('--obstacle-bearing', type=float, default=0.0,
                   help='Fake pedestrian baseline bearing in EGO frame (deg, '
                        '0 = straight ahead, +left). Default 0. The actual '
                        'bearing oscillates around this baseline if '
                        '--obstacle-sweep-amplitude > 0.')
    p.add_argument('--obstacle-sweep-amplitude', type=float, default=5.0,
                   help='Lateral sweep amplitude (m) — the pedestrian walks '
                        'sideways across the vehicle path with this peak '
                        'offset (sine wave). Set 0 for a stationary obstacle.')
    p.add_argument('--obstacle-sweep-period', type=float, default=8.0,
                   help='Lateral sweep period (s). Default 8.')
    p.add_argument('--horizon', type=int, default=30,
                   help='Number of future poses (H) per pedestrian trajectory.')
    p.add_argument('--horizon-dt', type=float, default=0.1,
                   help='Time step (s) between successive poses in the '
                        'trajectory. Default 0.1 (matches mppi/dt).')
    p.add_argument('--no-obstacle', action='store_true',
                   help='Skip publishing the fake pedestrian.')
    return p.parse_args()


def main():
    args = parse_args()

    rospy.init_node('fake_gps_publisher', anonymous=True)
    fix_pub  = rospy.Publisher('/navsatfix',  NavSatFix,    queue_size=10)
    ins_pub  = rospy.Publisher('/insnavgeod', INSNavGeod,   queue_size=10)
    goal_pub = rospy.Publisher('/goal_pose', PoseStamped,
                               queue_size=1)
    # adapt_mppi_node_ros1_sim subscribes to /fusion_pedestrian_position with
    # the default `prediction_source: raw`. The Float32MultiArray is flat
    # [dist_m, bearing_deg, dist_m, bearing_deg, ...] in the EGO frame; the
    # node transforms to world using the latest GPS+heading.
    ped_pub  = rospy.Publisher('/fusion_pedestrian_position',
                               Float32MultiArray, queue_size=10)

    # --- GPS / INS messages -----------------------------------------------
    fix = NavSatFix()
    fix.header.frame_id = args.frame
    fix.latitude  = args.lat
    fix.longitude = args.lon
    fix.altitude  = args.alt
    fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

    ins = INSNavGeod()
    ins.header.frame_id = args.frame
    ins.heading   = float(args.heading)
    ins.latitude  = math.radians(args.lat)   # INSNavGeod stores rad
    ins.longitude = math.radians(args.lon)
    ins.height    = args.alt

    # --- Goal pose: distance metres ahead in the heading direction --------
    yaw = heading_to_yaw(args.heading)
    gx = args.goal_distance * math.cos(yaw)
    gy = args.goal_distance * math.sin(yaw)

    goal = PoseStamped()
    goal.header.frame_id = args.goal_frame
    goal.header.stamp    = rospy.Time(0)
    goal.pose.position.x = gx
    goal.pose.position.y = gy
    # Orient the goal pose along the heading too (purely cosmetic).
    half = yaw / 2.0
    goal.pose.orientation.z = math.sin(half)
    goal.pose.orientation.w = math.cos(half)

    # --- Fake pedestrian: per-obstacle trajectory in EGO cartesian --------
    # Float32MultiArray, layout.dim = [M, H, 2] with (x_fwd, y_left). One
    # pedestrian (M=1), H future poses, sampled at --horizon-dt. The lateral
    # offset is a sine sweep so the obstacle walks sideways across the
    # vehicle's path over the horizon.
    H = max(1, int(args.horizon))
    ped_msg = Float32MultiArray()
    ped_msg.layout.dim = [
        MultiArrayDimension(label='obstacles', size=1, stride=H * 2),
        MultiArrayDimension(label='horizon',   size=H, stride=2),
        MultiArrayDimension(label='xy',        size=2, stride=1),
    ]

    def _ped_traj_at(t_sec):
        """Trajectory shape (H, 2) in EGO frame (x_fwd_m, y_left_m).

        For each look-ahead step h in [0, H): time = t_sec + h*horizon_dt,
        x_fwd stays at args.obstacle_distance, y_left is the sine sweep at
        that time. Returns a flat list of length H*2 in row-major order.
        """
        x_fwd = float(args.obstacle_distance)
        baseline_y = x_fwd * math.tan(math.radians(args.obstacle_bearing))
        amp = args.obstacle_sweep_amplitude
        period = args.obstacle_sweep_period
        out = []
        for h in range(H):
            t_h = t_sec + h * args.horizon_dt
            if amp > 0 and period > 0:
                y_lat = baseline_y + amp * math.sin(2.0 * math.pi * t_h / period)
            else:
                y_lat = baseline_y
            out.append(x_fwd)
            out.append(y_lat)
        return out

    rospy.loginfo(
        f'Fake GPS: ({args.lat:.7f}, {args.lon:.7f}) '
        f'heading={args.heading:.1f} deg @ {args.rate:.1f} Hz on '
        f'/navsatfix + /insnavgeod (frame={args.frame})'
    )
    if not args.no_goal:
        rospy.loginfo(
            f'Goal: ({gx:.2f}, {gy:.2f}) m  ({args.goal_distance:.1f} m '
            f'ahead, heading {args.heading:.1f} deg) — republished every '
            f'{args.goal_period:.1f} s on /goal_pose '
            f'frame={args.goal_frame}'
        )
    if not args.no_obstacle:
        if args.obstacle_sweep_amplitude > 0:
            rospy.loginfo(
                f'Obstacle: {args.obstacle_distance:.1f} m forward, '
                f'sweeping ±{args.obstacle_sweep_amplitude:.1f} m sideways '
                f'(period {args.obstacle_sweep_period:.1f} s), '
                f'H={H} poses @ dt={args.horizon_dt:.2f} s, at '
                f'{args.rate:.1f} Hz on /fusion_pedestrian_position '
                f'(Float32MultiArray [M=1, H, 2] ego cartesian)'
            )
        else:
            rospy.loginfo(
                f'Obstacle: {args.obstacle_distance:.1f} m, '
                f'{args.obstacle_bearing:.1f} deg bearing (ego frame, '
                f'stationary), H={H} poses — republished at '
                f'{args.rate:.1f} Hz on /fusion_pedestrian_position '
                f'(Float32MultiArray [M=1, H, 2] ego cartesian)'
            )

    rate = rospy.Rate(args.rate)
    last_goal_pub = None  # rospy.Time of most recent goal publish
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        fix.header.stamp = now
        ins.header.stamp = now
        fix_pub.publish(fix)
        ins_pub.publish(ins)
        if not args.no_obstacle:
            ped_msg.data = _ped_traj_at((now - t0).to_sec())
            ped_pub.publish(ped_msg)

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
