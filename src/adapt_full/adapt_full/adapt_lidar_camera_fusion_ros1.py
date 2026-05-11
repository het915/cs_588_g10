#!/usr/bin/env python3
"""ROS 1 (rospy) sensor fusion node.

Port of adapt_lidar_camera_fusion.py (rclpy) for the ROS-1 native pipeline
on branch feature/ros1-mppi-port. Subscribes to /lidar_pedestrian_position
and /rgbd_pedestrian_position (Int32MultiArray polar pairs), associates
detections by Cartesian distance, fuses matched pairs with separate
distance/direction weights, and publishes the union as
/fusion_pedestrian_position — the topic the diffusion predictor consumes.
"""

import math

import rospy
import message_filters

from std_msgs.msg import Int32MultiArray


class SensorFusionNode(object):
    def __init__(self):
        # ------- Parameters -------
        self.matching_threshold = float(
            rospy.get_param("~matching_threshold", 2.0)
        )
        self.lidar_dist_weight = float(rospy.get_param("~lidar_distance_weight", 0.8))
        self.camera_dist_weight = float(rospy.get_param("~camera_distance_weight", 0.2))
        self.lidar_dir_weight = float(rospy.get_param("~lidar_direction_weight", 0.3))
        self.camera_dir_weight = float(rospy.get_param("~camera_direction_weight", 0.7))
        queue_size = int(rospy.get_param("~sync_queue_size", 10))
        slop = float(rospy.get_param("~sync_slop", 0.1))

        # ------- Publisher -------
        self.fusion_pub = rospy.Publisher(
            "/fusion_pedestrian_position", Int32MultiArray, queue_size=10,
        )

        # ------- Subscribers + ApproximateTimeSynchronizer -------
        self.lidar_sub = message_filters.Subscriber(
            "/lidar_pedestrian_position", Int32MultiArray
        )
        self.camera_sub = message_filters.Subscriber(
            "/rgbd_pedestrian_position", Int32MultiArray
        )

        # Int32MultiArray has no header → require allow_headerless=True
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.camera_sub],
            queue_size=queue_size,
            slop=slop,
            allow_headerless=True,
        )
        self.ts.registerCallback(self.fusion_callback)

        # ------- Stats -------
        self.fusion_count = 0
        self.lidar_only_count = 0
        self.camera_only_count = 0

        rospy.loginfo("Sensor Fusion Node initialized")
        rospy.loginfo("Matching threshold: %.2f m", self.matching_threshold)
        rospy.loginfo(
            "Weights — Lidar: (dist=%.2f, dir=%.2f) Camera: (dist=%.2f, dir=%.2f)",
            self.lidar_dist_weight, self.lidar_dir_weight,
            self.camera_dist_weight, self.camera_dir_weight,
        )

    # ------------------------------------------------------------------
    def parse_detections(self, data_array):
        detections = []
        if len(data_array) % 2 != 0:
            rospy.logwarn("Invalid data array length: %d", len(data_array))
            return detections

        for i in range(0, len(data_array), 2):
            distance = data_array[i]
            direction = data_array[i + 1]
            if distance < 0 or direction < 0 or direction >= 360:
                rospy.logwarn(
                    "Invalid detection: dist=%s, dir=%s", distance, direction
                )
                continue
            detections.append({"dist": distance, "deg": direction})

        return detections

    @staticmethod
    def polar_to_cartesian(distance, direction_deg):
        rad = math.radians(direction_deg)
        return (distance * math.cos(rad), distance * math.sin(rad))

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def match_detections(self, lidar_detections, camera_detections):
        matched_pairs = []
        lidar_matched = [False] * len(lidar_detections)
        camera_matched = [False] * len(camera_detections)

        lidar_cartesian = [
            self.polar_to_cartesian(d["dist"], d["deg"]) for d in lidar_detections
        ]
        camera_cartesian = [
            self.polar_to_cartesian(d["dist"], d["deg"]) for d in camera_detections
        ]

        for i, lpos in enumerate(lidar_cartesian):
            if lidar_matched[i]:
                continue
            best_j = -1
            best_d = float("inf")
            for j, cpos in enumerate(camera_cartesian):
                if camera_matched[j]:
                    continue
                d = self.euclidean_distance(lpos, cpos)
                if d < self.matching_threshold and d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0:
                matched_pairs.append((lidar_detections[i], camera_detections[best_j]))
                lidar_matched[i] = True
                camera_matched[best_j] = True

        lidar_only = [
            lidar_detections[i] for i in range(len(lidar_detections))
            if not lidar_matched[i]
        ]
        camera_only = [
            camera_detections[j] for j in range(len(camera_detections))
            if not camera_matched[j]
        ]
        return matched_pairs, lidar_only, camera_only

    def fuse_matched_pair(self, lidar_det, camera_det):
        fused_distance = (
            self.lidar_dist_weight * lidar_det["dist"]
            + self.camera_dist_weight * camera_det["dist"]
        )

        # Angle wrap-around handling.
        lidar_deg = lidar_det["deg"]
        camera_deg = camera_det["deg"]
        if abs(lidar_deg - camera_deg) > 180:
            if lidar_deg > camera_deg:
                camera_deg += 360
            else:
                lidar_deg += 360

        fused_direction = (
            self.lidar_dir_weight * lidar_deg
            + self.camera_dir_weight * camera_deg
        ) % 360

        return {
            "dist": int(round(fused_distance)),
            "deg": int(round(fused_direction)),
        }

    # ------------------------------------------------------------------
    def fusion_callback(self, lidar_msg, camera_msg):
        lidar_detections = self.parse_detections(list(lidar_msg.data))
        camera_detections = self.parse_detections(list(camera_msg.data))

        if len(lidar_detections) == 0 and len(camera_detections) == 0:
            empty = Int32MultiArray()
            empty.data = []
            self.fusion_pub.publish(empty)
            return

        matched_pairs, lidar_only, camera_only = self.match_detections(
            lidar_detections, camera_detections
        )

        self.fusion_count += len(matched_pairs)
        self.lidar_only_count += len(lidar_only)
        self.camera_only_count += len(camera_only)

        fused_detections = [
            self.fuse_matched_pair(l, c) for l, c in matched_pairs
        ]
        fused_detections.extend(lidar_only)
        fused_detections.extend(camera_only)

        flat = []
        for det in fused_detections:
            flat.append(det["dist"])
            flat.append(det["deg"])

        msg = Int32MultiArray()
        msg.data = flat
        self.fusion_pub.publish(msg)

        rospy.loginfo(
            "Published %d fused detections: %d matched, %d lidar-only, %d camera-only",
            len(fused_detections), len(matched_pairs),
            len(lidar_only), len(camera_only),
        )

        total = self.fusion_count + self.lidar_only_count + self.camera_only_count
        if total > 0 and total % 50 == 0:
            rospy.loginfo(
                "Stats — total=%d fused=%d lidar-only=%d camera-only=%d",
                total, self.fusion_count,
                self.lidar_only_count, self.camera_only_count,
            )


def main():
    rospy.init_node("sensor_fusion_node")
    SensorFusionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
