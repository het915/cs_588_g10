#!/usr/bin/env python3
"""ROS 1 (rospy) LiDAR pedestrian detector.

Port of adapt_lidar_processing.py (rclpy) for the ROS-1 native pipeline on
branch feature/ros1-mppi-port. Publishes /lidar_pedestrian_position
(Int32MultiArray, polar [d_int, deg_int]) — the upstream of
adapt_lidar_camera_fusion.

Optional outputs (visualization + downstream):
  /processed_points         (PointCloud2, after crop / voxel / ground removal)
  /clustered_points         (PointCloud2, RGB-tagged per-track)
  /cluster_markers          (MarkerArray, per-track text + sphere)
  /human_debug_info         (MarkerArray, locked human cylinder + label)
  /detected_objects         (DetectedObjectArray, only if detected_object_msgs is installed)
"""

import colorsys
import math

import numpy as np
import open3d as o3d

import rospy

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2

from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker, MarkerArray

# detected_object_msgs is an optional / external package — keep it optional so
# the loop can close even if it is not installed under ROS 1.
try:
    from detected_object_msgs.msg import DetectedObject, DetectedObjectArray
    _HAS_DETECTED_OBJECT_MSGS = True
except Exception:
    DetectedObject = None
    DetectedObjectArray = None
    _HAS_DETECTED_OBJECT_MSGS = False


# -- Default values mirror config/lidar_params.yaml ----------------------
_DEFAULTS = {
    "crop_min_x": -15.0, "crop_max_x":  0.0,
    "crop_min_y":  -3.0, "crop_max_y":  3.0,
    "crop_min_z":  -2.1, "crop_max_z":  1.0,
    "voxel_size": 0.15,
    "sor_nb_neighbors": 30,
    "sor_std_ratio": 1.5,
    "ground_z_threshold": -1.0,
    "dbscan_eps": 0.6,
    "dbscan_min_points": 20,
    "track_max_distance": 1.2,
    "track_max_age": 10,
    "track_min_hits": 3,
    "ema_alpha": 0.4,
    "human_height_min": 0.8,
    "human_height_max": 2.2,
    "human_width_max": 1.0,
    "human_depth_max": 1.0,
    "human_ratio_min": 1.0,
    "human_footprint_max": 0.5,
    "human_volume_min": 0.08,
    "human_volume_max": 1.2,
    "human_compactness_max": 0.4,
    "human_xy_flatness_min": 0.2,
    "min_motion_threshold": 0.3,
    "static_check_frames": 10,
    "max_intensity_avg": 99999.0,
}


def _p(name):
    """Read a private rosparam; fall back to the YAML default."""
    return rospy.get_param("~" + name, _DEFAULTS[name])


class LidarObjectDetector(object):
    """LiDAR DBSCAN clustering + tracked human filter (rospy)."""

    def __init__(self):
        # Cache parameters once at startup so the per-frame callback path
        # does not hit the parameter server on every cluster.
        self.params = {k: _p(k) for k in _DEFAULTS}

        # Publishers
        self.pub_processed = rospy.Publisher("/processed_points", PointCloud2, queue_size=10)
        self.pub_clustered = rospy.Publisher("/clustered_points", PointCloud2, queue_size=10)
        self.pub_markers = rospy.Publisher("/cluster_markers", MarkerArray, queue_size=10)
        self.pub_pedestrian_pos = rospy.Publisher(
            "/lidar_pedestrian_position", Int32MultiArray, queue_size=10
        )
        self.pub_human_debug = rospy.Publisher("/human_debug_info", MarkerArray, queue_size=10)

        if _HAS_DETECTED_OBJECT_MSGS:
            self.pub_objects = rospy.Publisher(
                "/detected_objects", DetectedObjectArray, queue_size=10
            )
        else:
            self.pub_objects = None
            rospy.logwarn(
                "detected_object_msgs not installed — /detected_objects "
                "will not be published. Loop closure does not depend on it."
            )

        self.tracker = SimpleClusterTracker(self)

        # Subscriber wired last to avoid early callbacks before pubs exist.
        rospy.Subscriber("/ouster/points", PointCloud2, self.callback, queue_size=10)

        rospy.loginfo("Lidar Object Detector with Human Tracking STARTED")

    # ------------------------------------------------------------------
    def callback(self, msg):
        try:
            field_names = [f.name for f in msg.fields]
            has_intensity = "intensity" in field_names
            read_fields = (
                ("x", "y", "z", "intensity") if has_intensity else ("x", "y", "z")
            )

            cloud_gen = pc2.read_points(msg, field_names=read_fields, skip_nans=True)
            pts_list = list(cloud_gen)
            if not pts_list:
                return

            pts_array = np.array(pts_list)

            if pts_array.dtype.names:
                x = pts_array["x"]; y = pts_array["y"]; z = pts_array["z"]
                points_np = np.column_stack((x, y, z)).astype(np.float64)
            else:
                points_np = pts_array[:, 0:3].astype(np.float64)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)

            # Crop
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                [self.params["crop_min_x"], self.params["crop_min_y"], self.params["crop_min_z"]],
                [self.params["crop_max_x"], self.params["crop_max_y"], self.params["crop_max_z"]],
            )
            pcd = pcd.crop(bbox)
            if len(pcd.points) < 10:
                return

            # Voxel downsample
            pcd = pcd.voxel_down_sample(self.params["voxel_size"])

            # Statistical outlier removal
            try:
                if len(pcd.points) > self.params["sor_nb_neighbors"]:
                    _, ind = pcd.remove_statistical_outlier(
                        nb_neighbors=self.params["sor_nb_neighbors"],
                        std_ratio=self.params["sor_std_ratio"],
                    )
                    pcd = pcd.select_by_index(ind)
            except Exception:
                pass

            # Ground removal
            points = np.asarray(pcd.points)
            if len(points) > 0:
                points = points[points[:, 2] > self.params["ground_z_threshold"]]
            if len(points) < 10:
                return

            # Re-pack pcd for DBSCAN
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            header = msg.header
            header.stamp = rospy.Time.now()
            self.pub_processed.publish(
                pc2.create_cloud_xyz32(header, points.astype(np.float32))
            )

            clusters = self.cluster_with_dbscan(pcd)
            self.tracker.update(clusters, header)

        except Exception as e:
            rospy.logerr("Error: %s", e)

    # ------------------------------------------------------------------
    def cluster_with_dbscan(self, pcd):
        eps = self.params["dbscan_eps"]
        min_pts = self.params["dbscan_min_points"]

        if len(pcd.points) < min_pts:
            return []

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            labels = np.array(
                pcd.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False)
            )

        clusters = []
        if len(labels) == 0:
            return clusters

        points_np = np.asarray(pcd.points)
        for label in np.unique(labels):
            if label == -1:
                continue
            mask = (labels == label)
            pts = points_np[mask]
            if len(pts) < min_pts:
                continue
            clusters.append({"points": pts, "centroid": pts.mean(axis=0)})
        return clusters


class SimpleClusterTracker(object):
    def __init__(self, node):
        self.node = node
        self.tracks = {}
        self.next_id = 0
        self.colors = {}
        self.locked_human_id = None

    def get_color(self, tid):
        if tid not in self.colors:
            h = (tid * 37) % 360 / 360.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
            self.colors[tid] = (r, g, b)
        return self.colors[tid]

    def update(self, clusters, header):
        params = self.node.params
        centroids = [c["centroid"] for c in clusters]
        assignments = []
        used = set()

        for cent in centroids:
            best_id = None
            best_dist = params["track_max_distance"]
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = np.linalg.norm(tr["centroid"] - cent)
                if d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                assignments.append(best_id)
                used.add(best_id)
                a = params["ema_alpha"]
                self.tracks[best_id]["centroid"] = (
                    a * cent + (1 - a) * self.tracks[best_id]["centroid"]
                )
                self.tracks[best_id]["hits"] += 1
                self.tracks[best_id]["age"] = 0
                self.tracks[best_id]["points"] = clusters[len(assignments) - 1]["points"]
                start_pos = self.tracks[best_id].get(
                    "start_pos", self.tracks[best_id]["centroid"]
                )
                self.tracks[best_id]["start_pos"] = start_pos
                self.tracks[best_id]["displacement"] = np.linalg.norm(
                    start_pos - self.tracks[best_id]["centroid"]
                )
                self.tracks[best_id]["total_frames"] = (
                    self.tracks[best_id].get("total_frames", 0) + 1
                )
            else:
                assignments.append(self.next_id)
                self.tracks[self.next_id] = {
                    "centroid": cent.copy(),
                    "hits": 1,
                    "age": 0,
                    "start_pos": cent.copy(),
                    "displacement": 0.0,
                    "total_frames": 1,
                    "points": clusters[len(assignments) - 1]["points"],
                }
                self.next_id += 1

        # Age management
        for tid in list(self.tracks.keys()):
            if tid not in used:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > params["track_max_age"]:
                    del self.tracks[tid]
                    if tid == self.locked_human_id:
                        self.locked_human_id = None

        self.publish_visualization(assignments, clusters, header)
        if self.node.pub_objects is not None:
            self.publish_detected_objects(assignments, clusters, header)
        self.filter_and_publish_human(header)

    # ------------------------------------------------------------------
    def publish_visualization(self, assignments, clusters, header):
        params = self.node.params
        marker_array = MarkerArray()
        colored_points = []
        active_ids = set()

        short_life = rospy.Duration(0.2)

        for cluster, track_id in zip(clusters, assignments):
            track = self.tracks[track_id]
            if track["hits"] < params["track_min_hits"]:
                continue

            active_ids.add(track_id)
            centroid = track["centroid"]
            r, g, b = self.get_color(track_id)

            pts = cluster["points"].copy()
            rgb_packed = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
            pts_rgb = np.zeros((len(pts), 4))
            pts_rgb[:, :3] = pts
            pts_rgb[:, 3] = rgb_packed
            colored_points.append(pts_rgb)

            text = Marker()
            text.header = header
            text.ns = "id_text"
            text.id = track_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.MODIFY
            text.pose.position.x = centroid[0]
            text.pose.position.y = centroid[1]
            text.pose.position.z = centroid[2] + 1.0
            text.text = str(track_id)
            text.scale.z = 0.8
            text.color.r, text.color.g, text.color.b, text.color.a = r, g, b, 1.0
            text.lifetime = short_life
            marker_array.markers.append(text)

            sphere = Marker()
            sphere.header = header
            sphere.ns = "center"
            sphere.id = track_id + 10000
            sphere.type = Marker.SPHERE
            sphere.action = Marker.MODIFY
            sphere.pose.position.x = centroid[0]
            sphere.pose.position.y = centroid[1]
            sphere.pose.position.z = centroid[2]
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.4
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = r, g, b, 1.0
            sphere.lifetime = short_life
            marker_array.markers.append(sphere)

        for tid in self.tracks:
            if tid not in active_ids:
                for ns, base in [("id_text", 0), ("center", 10000)]:
                    del_marker = Marker()
                    del_marker.header = header
                    del_marker.ns = ns
                    del_marker.id = tid + base
                    del_marker.action = Marker.DELETE
                    marker_array.markers.append(del_marker)

        if colored_points:
            all_arr = np.vstack(colored_points)
            dtype = [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ]
            structured = np.zeros(all_arr.shape[0], dtype=dtype)
            structured["x"] = all_arr[:, 0]
            structured["y"] = all_arr[:, 1]
            structured["z"] = all_arr[:, 2]
            structured["rgb"] = all_arr[:, 3].astype(np.uint32)

            fields = [
                PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.UINT32,  count=1),
            ]
            cloud_msg = pc2.create_cloud(header, fields, structured)
            self.node.pub_clustered.publish(cloud_msg)

        self.node.pub_markers.publish(marker_array)

    # ------------------------------------------------------------------
    def publish_detected_objects(self, assignments, clusters, header):
        if self.node.pub_objects is None:
            return
        params = self.node.params
        arr = DetectedObjectArray()
        arr.header = header
        for cluster, tid in zip(clusters, assignments):
            track = self.tracks[tid]
            if track["hits"] < params["track_min_hits"]:
                continue
            obj = DetectedObject()
            obj.id = tid
            obj.center = Vector3(
                x=float(-track["centroid"][0]),
                y=float(-track["centroid"][1]),
                z=float(track["centroid"][2] + 2.0),
            )
            obj.point_count = len(cluster["points"])
            dx, dy = track["centroid"][0], track["centroid"][1]
            obj.distance = math.hypot(dx, dy)
            obj.angle_deg = math.degrees(math.atan2(-dx, dy))
            arr.objects.append(obj)
        self.node.pub_objects.publish(arr)

    # ------------------------------------------------------------------
    def filter_and_publish_human(self, header):
        params = self.node.params
        h_min = params["human_height_min"]
        h_max = params["human_height_max"]
        w_max = params["human_width_max"]
        d_max = params["human_depth_max"]
        ratio_min = params["human_ratio_min"]
        footprint_max = params["human_footprint_max"]
        vol_min = params["human_volume_min"]
        vol_max = params["human_volume_max"]
        compact_max = params["human_compactness_max"]
        flatness_min = params["human_xy_flatness_min"]

        min_motion = params["min_motion_threshold"]
        static_frames = params["static_check_frames"]
        min_hits = params["track_min_hits"]

        candidates = []

        for tid, track in self.tracks.items():
            if track["hits"] < min_hits:
                continue
            if track["total_frames"] > static_frames:
                if track["displacement"] < min_motion:
                    continue  # reject static

            pts = track["points"]
            if len(pts) < 5:
                continue

            min_pt = pts.min(axis=0)
            max_pt = pts.max(axis=0)
            dims = max_pt - min_pt
            dx, dy, dz = dims[0], dims[1], dims[2]

            xy_extent = max(dx, dy)
            xy_min = min(dx, dy)
            footprint = dx * dy
            volume = dx * dy * dz
            aspect_ratio = dz / xy_extent if xy_extent > 0 else 0

            if not (h_min < dz < h_max):
                continue
            if (dx > d_max) or (dy > w_max):
                continue
            if aspect_ratio < ratio_min:
                continue
            if footprint > footprint_max:
                continue
            if not (vol_min < volume < vol_max):
                continue
            if xy_extent > 0 and (xy_min / xy_extent) < flatness_min:
                continue

            z_bins = np.linspace(min_pt[2], max_pt[2], 4)
            hist, _ = np.histogram(pts[:, 2], bins=z_bins)
            if np.count_nonzero(hist) < 2:
                continue

            centroid_xy = track["centroid"][:2]
            dists_to_center = np.linalg.norm(pts[:, :2] - centroid_xy, axis=1)
            if np.mean(dists_to_center) > compact_max:
                continue

            dist = math.hypot(track["centroid"][0], track["centroid"][1])
            candidates.append({"dist": dist, "id": tid, "track": track})

        selected_track = None
        candidates.sort(key=lambda x: x["dist"])

        if not candidates:
            self.locked_human_id = None
        else:
            best_candidate = candidates[0]
            if self.locked_human_id is None:
                selected_track = best_candidate["track"]
                self.locked_human_id = best_candidate["id"]
            else:
                locked_candidate = next(
                    (c for c in candidates if c["id"] == self.locked_human_id), None
                )
                if locked_candidate:
                    if best_candidate["dist"] < (locked_candidate["dist"] - 1.5):
                        selected_track = best_candidate["track"]
                        self.locked_human_id = best_candidate["id"]
                    else:
                        selected_track = locked_candidate["track"]
                else:
                    selected_track = best_candidate["track"]
                    self.locked_human_id = best_candidate["id"]

        marker_array = MarkerArray()
        short_life = rospy.Duration(0.2)

        if selected_track:
            cx, cy = selected_track["centroid"][0], selected_track["centroid"][1]
            cz = selected_track["centroid"][2]
            dist_val = int(math.hypot(cx, cy) + 0.5)

            angle_rad = math.atan2(-cx, cy)
            angle_deg = int(math.degrees(angle_rad) + 0.5)
            if angle_deg < 0:
                angle_deg += 360

            pos_msg = Int32MultiArray()
            pos_msg.data = [dist_val, angle_deg]
            self.node.pub_pedestrian_pos.publish(pos_msg)

            info_marker = Marker()
            info_marker.header = header
            info_marker.ns = "human_info"
            info_marker.id = 999
            info_marker.type = Marker.TEXT_VIEW_FACING
            info_marker.action = Marker.ADD
            info_marker.pose.position.x = cx
            info_marker.pose.position.y = cy
            info_marker.pose.position.z = cz + 2.0
            info_marker.text = "HUMAN ID:{}\n{}m | {}deg".format(
                self.locked_human_id, dist_val, angle_deg
            )
            info_marker.scale.z = 0.5
            info_marker.color.r = 0.0
            info_marker.color.g = 1.0
            info_marker.color.b = 0.0
            info_marker.color.a = 1.0
            info_marker.lifetime = short_life
            marker_array.markers.append(info_marker)

            cyl = Marker()
            cyl.header = header
            cyl.ns = "human_highlight"
            cyl.id = 1000
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x = cx
            cyl.pose.position.y = cy
            cyl.pose.position.z = cz
            cyl.scale.x = 0.8
            cyl.scale.y = 0.8
            cyl.scale.z = 1.8
            cyl.color.r = 0.0
            cyl.color.g = 1.0
            cyl.color.b = 0.0
            cyl.color.a = 0.4
            cyl.lifetime = short_life
            marker_array.markers.append(cyl)

        self.node.pub_human_debug.publish(marker_array)


def main():
    rospy.init_node("lidar_preprocessor")
    LidarObjectDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
