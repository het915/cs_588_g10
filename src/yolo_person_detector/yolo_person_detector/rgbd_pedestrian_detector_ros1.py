#!/usr/bin/env python3
"""ROS 1 (rospy) RGB-D pedestrian detector.

Port of rgbd_pedestrian_detector.py (rclpy) for the ROS-1 native pipeline
on branch feature/ros1-mppi-port. Publishes the same /rgbd_pedestrian_position
Int32MultiArray polar contract that adapt_lidar_camera_fusion subscribes to.
"""

import time

import numpy as np
import cv2

import rospy

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Int32MultiArray, Bool

from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# COCO class ID for 'person'
COCO_PERSON_CLASS_ID = 0


class RgbdPedestrianDetector(object):
    """YOLOv11 + depth → polar pedestrian detection (rospy)."""

    def __init__(self):
        # ------- Parameters -------
        self.publish_debug = bool(rospy.get_param("~publish_debug_image", True))
        self.model_path = rospy.get_param("~model_path", "yolo11n.pt")
        self.image_topic = rospy.get_param("~image_topic", "/oak/rgb/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/oak/stereo/image_raw")
        self.conf = float(rospy.get_param("~conf", 0.35))
        self.iou = float(rospy.get_param("~iou", 0.45))
        self.device = rospy.get_param("~device", "cuda:0")
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.half = bool(rospy.get_param("~half", False))
        self.max_det = int(rospy.get_param("~max_detections", 100))

        # ------- YOLO model -------
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics is not installed. `pip install ultralytics`"
            )

        rospy.loginfo("Loading YOLOv11 model: %s", self.model_path)
        self.model = YOLO(self.model_path)
        self.model.overrides["conf"] = self.conf
        self.model.overrides["iou"] = self.iou
        self.model.overrides["device"] = self.device
        self.model.overrides["imgsz"] = self.imgsz
        self.model.overrides["half"] = self.half
        self.model.overrides["max_det"] = self.max_det
        self.model.overrides["classes"] = [COCO_PERSON_CLASS_ID]

        # ------- ROS I/O -------
        self.bridge = CvBridge()

        self.pub_debug = (
            rospy.Publisher("detections/image", Image, queue_size=10)
            if self.publish_debug else None
        )
        self.pub_rgbd_position = rospy.Publisher(
            "/rgbd_pedestrian_position", Int32MultiArray, queue_size=10,
        )
        self.pub_ped_sign_present = rospy.Publisher(
            "/pedestrian_sign_present", Bool, queue_size=10,
        )
        self.pub_person_marker = rospy.Publisher(
            "person_3d_marker", Marker, queue_size=10,
        )

        self.tf_broadcaster = TransformBroadcaster()
        self.publish_camera_transform()

        # ------- State -------
        self.latest_depth = None
        self.last_fps_t = time.time()
        self.frame_count = 0

        # Subscribers wired last so callbacks don't fire before model is ready.
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=5)
        rospy.Subscriber(self.depth_topic, Image, self.depth_cb, queue_size=5)

        rospy.loginfo("RGB-D Person 3D Extractor ready.")

    # ------------------------------------------------------------------
    # Static camera TF
    # ------------------------------------------------------------------
    def publish_camera_transform(self):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base_link"
        t.child_frame_id = "oak_rgb_camera_optical_frame"
        t.transform.translation.x = 0.535
        t.transform.translation.y = 0.0
        t.transform.translation.z = 1.683
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.7071080798594738
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.7071054825112363
        self.tf_broadcaster.sendTransform(t)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def depth_cb(self, msg):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_depth = depth_img.astype(np.float32) / 1000.0  # mm → m
        except Exception as e:
            rospy.logwarn("Depth conversion failed: %s", e)

    def image_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge RGB failed: %s", e)
            return

        self.publish_camera_transform()

        h, w = cv_image.shape[:2]
        debug_img = cv_image.copy() if self.publish_debug else None

        # Crude intrinsics from image size — replace with real K if you have it.
        fx = float(w)
        fy = float(h)
        cx0 = w / 2.0
        cy0 = h / 2.0

        results = self.model.predict(cv_image, verbose=False, stream=False)

        best_dist = float("inf")
        best_angle_deg = 0.0
        found_valid_ped = False

        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            clss = r.boxes.cls.detach().cpu().numpy().astype(int)

            for i in range(xyxy.shape[0]):
                if clss[i] != COCO_PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = xyxy[i]
                score = float(confs[i])

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Depth lookup
                z = 0.0
                if self.latest_depth is not None:
                    xi = int(np.clip(cx, 0, self.latest_depth.shape[1] - 1))
                    yi = int(np.clip(cy, 0, self.latest_depth.shape[0] - 1))
                    z = float(self.latest_depth[yi, xi])
                if z <= 0.0:
                    continue

                # Camera optical frame (x right, y down, z forward) → base_link
                X_optical = (cx - cx0) * z / fx
                Y_optical = (cy - cy0) * z / fy
                Z_optical = z

                X_base = Y_optical
                Y_base = -X_optical
                Z_base = Z_optical

                dist = float(np.sqrt(Y_base ** 2 + Z_base ** 2))

                # Direction: 0° = right (-y), 90° = front (+x), CCW positive.
                angle_rad = np.arctan2(Z_base, -Y_base)
                angle_deg = float(np.degrees(angle_rad))
                if angle_deg < 0.0:
                    angle_deg += 360.0

                if dist < best_dist:
                    best_dist = dist
                    best_angle_deg = angle_deg
                    found_valid_ped = True

                if debug_img is not None:
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv2.rectangle(debug_img, p1, p2, (0, 255, 0), 2)
                    cv2.putText(
                        debug_img, "person {:.2f}".format(score),
                        (p1[0], max(0, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
                    cv2.putText(
                        debug_img,
                        "{:.1f} m, {:.1f} deg".format(dist, angle_deg),
                        (p1[0], min(h - 5, p2[1] + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    )

                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = msg.header.stamp
                marker.ns = "person_3d"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = X_base
                marker.pose.position.y = Y_base
                marker.pose.position.z = Z_base
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                self.pub_person_marker.publish(marker)

        if self.pub_debug is not None and debug_img is not None:
            out_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            out_msg.header = msg.header
            self.pub_debug.publish(out_msg)

        sign_msg = Bool()
        if found_valid_ped:
            sign_msg.data = True
            pos_msg = Int32MultiArray()
            pos_msg.data = [
                int(round(best_dist)),
                int(round(best_angle_deg)),
            ]
            self.pub_rgbd_position.publish(pos_msg)
        else:
            sign_msg.data = False

        self.pub_ped_sign_present.publish(sign_msg)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now_t = time.time()
            fps = 30.0 / (now_t - self.last_fps_t + 1e-9)
            self.last_fps_t = now_t
            rospy.loginfo("~%.1f FPS", fps)


def main():
    rospy.init_node("rgbd_pedestrian_detector")
    RgbdPedestrianDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
