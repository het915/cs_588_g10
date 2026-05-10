#!/usr/bin/env python3
"""ROS 1 (rospy) inference node — diffusion-based pedestrian trajectory predictor.

Mirrors infer_node.py (ROS 2 / rclpy) one-to-one but uses rospy plumbing
so it can run alongside the rospy MPPI controller in
src/vehicle_drivers/mppi_controller/mppi_controller/adapt_mppi_node_ros1.py.

Topic contract:
  Subscribes:
    /fusion_pedestrian_position  (std_msgs/Int32MultiArray, polar [d,b,d,b,...] in base_link)
    <vehicle_speed_topic>        (pacmod2_msgs/VehicleSpeedRpt) — default /pacmod/vehicle_speed_rpt
  Publishes:
    /person_prediction              (visualization_msgs/Marker, base_link LINE_STRIP)
    /pedestrian_motion              (geometry_msgs/Twist, primary ped position)
    /pedestrian_ttc                 (std_msgs/Float64, time-to-collision sec)
    /pedestrian_predictions_tensor  (std_msgs/Float32MultiArray, dims=[M,H=20,2], base_link)

The rospy MPPI node (~prediction_source:=predicted) consumes the tensor
verbatim — see adapt_mppi_node_ros1.py:_pred_tensor_cb.
"""

import math
import os

import numpy as np

import rospy

from std_msgs.msg import Int32MultiArray, Float64, Float32MultiArray
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
from pacmod2_msgs.msg import VehicleSpeedRpt

from diffusion_prediction.tracker import Tracker
from diffusion_prediction.utils import (
    decode_fusion_msg,
    compute_ttc,
    build_twist_msg,
    build_predictions_tensor,
    smooth_single_trajectory,
    _check_physics,
    _extrapolate_from_history,
)


class DiffusionPredictorNode(object):
    """Diffusion-based pedestrian trajectory prediction node (rospy)."""

    def __init__(self):
        # --------------- Parameters ---------------
        self.weights_path = rospy.get_param("~weights", "")
        self.device_str = rospy.get_param("~device", "cuda:0")
        self.K = int(rospy.get_param("~K", 20))
        self.ddim_steps = int(rospy.get_param("~ddim_steps", 10))
        self.min_hist = int(rospy.get_param("~min_history_count", 5))
        self.pred_time = float(rospy.get_param("~prediction_time", 5.0))
        self.pred_pts = int(rospy.get_param("~prediction_points", 20))
        self.collision_thresh = float(
            rospy.get_param("~collision_distance_threshold", 1.0)
        )
        self.latency_warn = float(rospy.get_param("~latency_warn_ms", 80.0))
        self.prediction_mode = rospy.get_param("~prediction_mode", "joint")
        self.max_agents = int(rospy.get_param("~max_agents", 16))
        vehicle_speed_topic = rospy.get_param(
            "~vehicle_speed_topic", "/pacmod/vehicle_speed_rpt"
        )

        # --------------- State ---------------
        self.tracker = Tracker(max_dist=2.0, max_missing=10, smooth_alpha=0.6)
        self.vehicle_speed = 0.0
        self.vehicle_speed_valid = False

        # Sticky mode selection state: track_id -> (prev_idx, consecutive_count)
        self._sticky_state = {}

        # Temporal EMA state: track_id -> previous best trajectory (20, 2)
        self._prev_trajs = {}
        self._temporal_alpha = 0.3  # 0=all previous, 1=all current

        # --------------- Model ---------------
        self.model = None
        self.schedule = None
        self._torch = None
        self._torch_device = None
        self._load_model()

        # --------------- Publishers ---------------
        self.pub_prediction = rospy.Publisher(
            "/person_prediction", Marker, queue_size=10
        )
        self.pub_motion = rospy.Publisher(
            "/pedestrian_motion", Twist, queue_size=10
        )
        self.pub_ttc = rospy.Publisher(
            "/pedestrian_ttc", Float64, queue_size=10
        )
        self.pub_tensor = rospy.Publisher(
            "/pedestrian_predictions_tensor", Float32MultiArray, queue_size=10
        )

        # --------------- Subscribers ---------------
        rospy.Subscriber(
            "/fusion_pedestrian_position", Int32MultiArray,
            self.pedestrian_cb, queue_size=10,
        )
        rospy.Subscriber(
            vehicle_speed_topic, VehicleSpeedRpt,
            self.vehicle_cb, queue_size=10,
        )

        rospy.loginfo(
            "Diffusion predictor node ready (mode=%s, K=%d, vehicle_speed_topic=%s)",
            self.prediction_mode, self.K, vehicle_speed_topic,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            import torch
            from diffusion_prediction.ddpm import CosineSchedule

            device = torch.device(
                self.device_str if torch.cuda.is_available() else "cpu"
            )
            self._torch_device = device

            if self.prediction_mode == "joint":
                from diffusion_prediction.model_joint import JointTrajectoryDenoiser
                self.model = JointTrajectoryDenoiser(
                    d=256, max_agents=self.max_agents,
                    nhead=8, num_enc_layers=6, num_dec_layers=4,
                    num_interaction_layers=3, dim_ff=512,
                ).to(device)
                rospy.loginfo(
                    "Using joint multi-agent model (max_agents=%d)", self.max_agents,
                )
            else:
                from diffusion_prediction.model import TrajectoryDenoiser
                self.model = TrajectoryDenoiser(
                    d=256, nhead=8, num_enc_layers=6,
                    num_dec_layers=4, dim_ff=512,
                ).to(device)
                rospy.loginfo("Using single-agent model")

            self.schedule = CosineSchedule(T=100).to(device)

            if self.weights_path and os.path.exists(self.weights_path):
                state = torch.load(
                    self.weights_path, map_location=device, weights_only=True,
                )
                if isinstance(state, dict) and "model_state" in state:
                    self.model.load_state_dict(state["model_state"])
                else:
                    self.model.load_state_dict(state)
                rospy.loginfo("Loaded weights from %s", self.weights_path)
            else:
                rospy.logwarn(
                    "No weights loaded — running with random weights "
                    "(prediction will be noise). Set the 'weights' parameter "
                    "to a checkpoint path."
                )

            self.model.eval()
            self._torch = torch

        except Exception as e:
            rospy.logerr("Failed to load model: %s", e)
            self.model = None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def vehicle_cb(self, msg):
        if msg.vehicle_speed_valid:
            self.vehicle_speed = float(msg.vehicle_speed)
            self.vehicle_speed_valid = True
        else:
            self.vehicle_speed_valid = False

    def pedestrian_cb(self, msg):
        """Main callback: track, predict, publish."""
        import time as _time

        t0 = _time.perf_counter()
        stamp = rospy.Time.now()
        t_now = stamp.to_sec()

        # Decode polar -> Cartesian (base_link)
        detections = decode_fusion_msg(list(msg.data))
        if detections.shape[0] == 0:
            return

        # Update tracker
        deleted = self.tracker.update(detections, t_now)

        # Clean up sticky state for deleted tracks
        for tid in deleted:
            self._sticky_state.pop(tid, None)

        # Collect tracks with enough history
        active_ids = []
        histories = []
        masks = []
        ego_vels = []

        for tid, tr in self.tracker.tracks.items():
            hist, mask = self.tracker.get_history(tid, T_hist=20)
            presence = mask.sum()
            if presence < self.min_hist:
                continue
            active_ids.append(tid)
            histories.append(hist)
            masks.append(mask)
            ego_vels.append(
                np.array([self.vehicle_speed, 0.0], dtype=np.float32)
            )

        if not active_ids or self.model is None:
            self._publish_fallback(stamp)
            return

        # ---------------- Diffusion inference ----------------
        torch = self._torch
        device = self._torch_device
        M = len(active_ids)

        if self.prediction_mode == "joint":
            from diffusion_prediction.ddpm import ddim_sample_loop_joint

            M_pad = self.max_agents
            hist_pad = np.zeros((M_pad, 20, 4), dtype=np.float32)
            mask_pad = np.zeros((M_pad, 20), dtype=np.float32)
            agent_mask = np.zeros(M_pad, dtype=np.float32)
            for i in range(M):
                hist_pad[i] = histories[i]
                mask_pad[i] = masks[i]
                agent_mask[i] = 1.0

            hist_t = torch.from_numpy(hist_pad).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask_pad).unsqueeze(0).to(device)
            amask_t = torch.from_numpy(agent_mask).unsqueeze(0).to(device)
            ego_t = torch.from_numpy(ego_vels[0]).unsqueeze(0).to(device)

            joint_futures = ddim_sample_loop_joint(
                self.model, self.schedule, hist_t, mask_t, amask_t, ego_t,
                K=self.K,
            )
            futures = joint_futures[0, :, :M, :, :].permute(1, 0, 2, 3)
        else:
            from diffusion_prediction.ddpm import ddim_sample_loop

            hist_t = torch.from_numpy(np.stack(histories)).to(device)
            mask_t = torch.from_numpy(np.stack(masks)).to(device)
            ego_t = torch.from_numpy(np.stack(ego_vels)).to(device)

            futures = ddim_sample_loop(
                self.model, self.schedule, hist_t, mask_t, ego_t, K=self.K,
            )

        # ---------------- Mode selection per track ----------------
        best_trajs = np.zeros((M, 20, 2), dtype=np.float32)
        for m_idx in range(M):
            tid = active_ids[m_idx]
            samples_np = futures[m_idx].cpu().numpy()  # (K, 20, 2)

            valid = _check_physics(samples_np, dt=0.25, max_speed=3.5, max_accel=4.0)

            valid_idx = np.where(valid)[0]
            if len(valid_idx) >= 3:
                median_center = np.median(samples_np[valid_idx], axis=0)
            else:
                median_center = np.median(samples_np, axis=0)
                valid_idx = np.arange(len(samples_np))

            extrap = _extrapolate_from_history(histories[m_idx], T_fut=20, dt=0.25)
            center = 0.6 * extrap + 0.4 * median_center

            cost = ((samples_np[valid_idx] - center[None]) ** 2).sum(axis=(1, 2))
            cand_local = cost.argmin()
            cand_idx = valid_idx[cand_local]
            cand_cost = cost[cand_local]

            prev = self._sticky_state.get(tid)
            if prev is not None:
                prev_idx, consec = prev
                if prev_idx in valid_idx:
                    prev_local = np.where(valid_idx == prev_idx)[0]
                    if len(prev_local) > 0:
                        prev_cost = cost[prev_local[0]]
                        if prev_cost > 1.5 * cand_cost:
                            consec += 1
                            if consec >= 3:
                                chosen_idx = cand_idx
                                self._sticky_state[tid] = (chosen_idx, 0)
                            else:
                                chosen_idx = prev_idx
                                self._sticky_state[tid] = (prev_idx, consec)
                        else:
                            chosen_idx = prev_idx
                            self._sticky_state[tid] = (prev_idx, 0)
                    else:
                        chosen_idx = cand_idx
                        self._sticky_state[tid] = (chosen_idx, 0)
                else:
                    chosen_idx = cand_idx
                    self._sticky_state[tid] = (chosen_idx, 0)
            else:
                chosen_idx = cand_idx
                self._sticky_state[tid] = (chosen_idx, 0)

            best_trajs[m_idx] = samples_np[chosen_idx]

        # ---------------- Smoothing ----------------
        for m_idx in range(M):
            best_trajs[m_idx] = smooth_single_trajectory(
                best_trajs[m_idx], s_factor=50.0,
            )

        for m_idx, tid in enumerate(active_ids):
            if tid in self._prev_trajs:
                best_trajs[m_idx] = (
                    self._temporal_alpha * best_trajs[m_idx]
                    + (1 - self._temporal_alpha) * self._prev_trajs[tid]
                )
            self._prev_trajs[tid] = best_trajs[m_idx].copy()

        active_set = set(active_ids)
        for tid in list(self._prev_trajs.keys()):
            if tid not in active_set and tid not in self.tracker.tracks:
                del self._prev_trajs[tid]

        # Undo origin centering (add current pedestrian position)
        for m_idx, tid in enumerate(active_ids):
            tr = self.tracker.tracks[tid]
            best_trajs[m_idx, :, 0] += tr.x
            best_trajs[m_idx, :, 1] += tr.y

        # ---------------- TTC + primary selection ----------------
        ttc_values = {}
        for m_idx, tid in enumerate(active_ids):
            ttc_values[tid] = compute_ttc(
                best_trajs[m_idx], self.vehicle_speed,
                dt=0.25, collision_dist=self.collision_thresh,
            )

        primary_idx = None
        primary_ttc = math.inf

        finite_ttc = {tid: t for tid, t in ttc_values.items() if t < math.inf}
        if finite_ttc:
            best_tid = min(finite_ttc, key=finite_ttc.get)
            primary_idx = active_ids.index(best_tid)
            primary_ttc = finite_ttc[best_tid]
        else:
            min_dist = math.inf
            for m_idx, tid in enumerate(active_ids):
                tr = self.tracker.tracks[tid]
                d = math.sqrt(tr.x ** 2 + tr.y ** 2)
                if d < min_dist:
                    min_dist = d
                    primary_idx = m_idx

        # ---------------- Publish ----------------
        if primary_idx is not None:
            marker = self._build_marker(best_trajs[primary_idx], stamp)
            self.pub_prediction.publish(marker)

            tr = self.tracker.tracks[active_ids[primary_idx]]
            twist = build_twist_msg(tr.x, tr.y)
            self.pub_motion.publish(twist)

            ttc_msg = Float64()
            ttc_msg.data = float(primary_ttc)
            self.pub_ttc.publish(ttc_msg)

        tensor_msg = build_predictions_tensor(best_trajs)
        self.pub_tensor.publish(tensor_msg)

        elapsed_ms = (_time.perf_counter() - t0) * 1000.0
        if elapsed_ms > self.latency_warn:
            rospy.logwarn(
                "Inference cycle took %.1f ms (> %.0f ms)",
                elapsed_ms, self.latency_warn,
            )

    # ------------------------------------------------------------------
    # Fallback path
    # ------------------------------------------------------------------
    def _publish_fallback(self, stamp):
        if not self.tracker.tracks:
            return

        min_dist = math.inf
        primary_tr = None
        for tid, tr in self.tracker.tracks.items():
            d = math.sqrt(tr.x ** 2 + tr.y ** 2)
            if d < min_dist:
                min_dist = d
                primary_tr = tr

        if primary_tr is None:
            return

        twist = build_twist_msg(primary_tr.x, primary_tr.y)
        self.pub_motion.publish(twist)

        if primary_tr.predicted_path:
            pred_arr = np.array(primary_tr.predicted_path)[:, :2]
            marker = self._build_marker(pred_arr, stamp)
            self.pub_prediction.publish(marker)

            ttc = compute_ttc(
                pred_arr, self.vehicle_speed,
                dt=0.25, collision_dist=self.collision_thresh,
            )
            ttc_msg = Float64()
            ttc_msg.data = float(ttc)
            self.pub_ttc.publish(ttc_msg)

    # ------------------------------------------------------------------
    # Marker builder (rospy-flavored, mirrors utils.build_marker_msg)
    # ------------------------------------------------------------------
    def _build_marker(self, trajectory, stamp, marker_id=0):
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = "person_prediction"
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.15
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        m.lifetime = rospy.Duration(0.5)
        m.points = [
            Point(float(trajectory[i, 0]), float(trajectory[i, 1]), 0.0)
            for i in range(len(trajectory))
        ]
        return m


def main():
    rospy.init_node("diffusion_predictor_node")
    DiffusionPredictorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
