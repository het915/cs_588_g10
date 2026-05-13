#!/usr/bin/env python3
"""Broadcast map → base_link TF from Gazebo initial position.

Uses Gazebo initial position (x, y, yaw) as the map frame origin.
This aligns the map frame with the Gazebo world coordinate system.
"""
import math

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped


def main():
    rospy.init_node("map_tf_broadcaster", anonymous=False)
    
    # Get Gazebo initial position parameters (same as gem_init.launch args)
    gazebo_x = rospy.get_param("~gazebo_x", 12.5)
    gazebo_y = rospy.get_param("~gazebo_y", -21.0)
    gazebo_yaw = rospy.get_param("~gazebo_yaw", 3.1416)
    
    map_frame = rospy.get_param("~map_frame", "map")
    world_frame = rospy.get_param("~world_frame", "world")
    
    br = tf2_ros.TransformBroadcaster()
    
    rospy.loginfo(f"map_tf_broadcaster: Starting TF broadcaster")
    rospy.loginfo(f"  Publishing: {world_frame} → {map_frame}")
    rospy.loginfo(f"  Gazebo offset: x={gazebo_x}, y={gazebo_y}, yaw={gazebo_yaw} rad")
    
    rate = rospy.Rate(10.0)  # 10Hz
    count = 0
    
    while not rospy.is_shutdown():
        try:
            # Transform: world → map (static)
            t1 = TransformStamped()
            t1.header.stamp = rospy.Time.now()
            t1.header.frame_id = world_frame
            t1.child_frame_id = map_frame
            
            t1.transform.translation.x = -gazebo_x
            t1.transform.translation.y = -gazebo_y
            t1.transform.translation.z = 0.0
            
            half_yaw = -gazebo_yaw / 2.0
            t1.transform.rotation.x = 0.0
            t1.transform.rotation.y = 0.0
            t1.transform.rotation.z = math.sin(half_yaw)
            t1.transform.rotation.w = math.cos(half_yaw)
            
            br.sendTransform(t1)
            
            count += 1
            if count % 50 == 0:
                rospy.logdebug(f"TF: {world_frame}→{map_frame} @ {t1.header.stamp}")
            
            rate.sleep()
            
        except Exception as e:
            rospy.logerr(f"map_tf_broadcaster error: {e}", exc_info=True)
            rate.sleep()


if __name__ == '__main__':
    main()
