# Diffusion Predictor — ROS 1 Runbook

ROS 1 (rospy + catkin) inference node for the conditional diffusion
pedestrian-trajectory predictor. Lives at
`src/diffusion_prediction/diffusion_prediction/infer_node_ros1.py` and is
the ROS-1 sibling of the original ROS-2 `infer_node.py`.

## Topic contract

| Direction | Topic | Type | Frame / layout |
|---|---|---|---|
| in  | `/fusion_pedestrian_position`     | `std_msgs/Int32MultiArray`   | base_link, polar `[d,b,d,b,…]` |
| in  | `/pacmod/vehicle_speed_rpt`       | `pacmod2_msgs/VehicleSpeedRpt` | scalar m/s (TTC only) |
| out | `/pedestrian_predictions_tensor`  | `std_msgs/Float32MultiArray` | base_link, dims `[M, 20, 2]` |
| out | `/person_prediction`              | `visualization_msgs/Marker`   | base_link LINE_STRIP |
| out | `/pedestrian_motion`              | `geometry_msgs/Twist`         | primary ped position in `linear.x/y` |
| out | `/pedestrian_ttc`                 | `std_msgs/Float64`            | seconds, `inf` if no collision |

The output `Float32MultiArray` matches what
`adapt_mppi_node_ros1.py:_pred_tensor_cb` decodes (`M, H = dim[0].size,
dim[1].size`; reshape `(M, H, 2)`; rotate base_link → world via current
yaw). No remapping needed — same names on both ends.

## Build

```bash
cd ~/UIUC/AVSE/cs_588_g10
source /opt/ros/noetic/setup.bash
catkin_make            # or: catkin build
source devel/setup.bash
```

## Smoke test (4 terminals)

Each terminal first sources ROS 1 + the catkin workspace:

```bash
conda activate adapt-py310
source /opt/ros/noetic/setup.bash
source ~/UIUC/AVSE/cs_588_g10/devel/setup.bash
```

**Terminal 1 — `roscore`**

```bash
roscore
```

**Terminal 2 — fake GPS / fusion publisher** (the only ROS-1 producer of
`/fusion_pedestrian_position` on this branch today; see "Integration
gap" below for the real-vehicle path):

```bash
rosrun mppi_controller publish_fake_gps.py
```

**Terminal 3 — diffusion predictor**

```bash
roslaunch diffusion_prediction diffusion_predictor.launch \
    prediction_mode:=joint \
    weights:=$(rospack find diffusion_prediction)/models/diffusion/av2_joint_v2/ema_best.pt \
    device:=cuda:0
```

Expected log lines:

```
[INFO] Using joint multi-agent model (max_agents=16)
[INFO] Loaded weights from .../av2_joint_v2/ema_best.pt
[INFO] Diffusion predictor node ready (mode=joint, K=20, vehicle_speed_topic=/pacmod/vehicle_speed_rpt)
```

**Terminal 4 — verify the contract**

```bash
rostopic hz   /pedestrian_predictions_tensor      # expect ~10 Hz
rostopic echo -n1 /pedestrian_predictions_tensor | head -40
```

You should see `layout.dim` with three entries (sizes `M`, `20`, `2`) and
`data` length `M*20*2`.

## Wire it into MPPI

The rospy MPPI node has a runtime switch — set
`prediction_source:=predicted` to consume the diffusion tensor instead
of raw polar detections:

```bash
# Sim variant (no PACMod cmds — safe at a desk):
rosrun mppi_controller adapt_mppi_node_ros1_sim.py \
    _prediction_source:=predicted

# Or full vehicle node:
rosrun mppi_controller adapt_mppi_node_ros1.py \
    _prediction_source:=predicted
```

In the MPPI log, expect:

```
Obstacle source: /pedestrian_predictions_tensor (full trajectories)
```

## Bench unit check (no ROS)

Sanity check the model + DDIM loop without any ROS plumbing:

```bash
conda activate adapt-py310
python - <<'EOF'
import torch
from diffusion_prediction.model_joint import JointTrajectoryDenoiser
from diffusion_prediction.ddpm import CosineSchedule, ddim_sample_loop_joint

m   = JointTrajectoryDenoiser(d=256, max_agents=16).cuda().eval()
sch = CosineSchedule(T=100).cuda()
hist = torch.zeros(1, 16, 20, 4, device='cuda')
mask = torch.zeros(1, 16, 20,    device='cuda')
amask = torch.zeros(1, 16,        device='cuda'); amask[0, 0] = 1
ego  = torch.zeros(1, 2,          device='cuda')

out = ddim_sample_loop_joint(m, sch, hist, mask, amask, ego, K=4)
print(out.shape)   # expect torch.Size([1, 4, 16, 20, 2])
EOF
```

If this returns the expected shape, only the rospy plumbing remains as a
variable.

## Parameters (private, set via `~name` or rosparam)

| Param | Default | Notes |
|---|---|---|
| `~weights` | `""` | Path to checkpoint. Required for non-noise output. |
| `~device` | `cuda:0` | Falls back to CPU if CUDA unavailable. |
| `~prediction_mode` | `joint` | `joint` (multi-agent) or `single`. |
| `~K` | `20` | Samples per pedestrian; mode pick is sticky-closest-to-mean. |
| `~max_agents` | `16` | Joint-model padding. |
| `~min_history_count` | `5` | Skip a track until this many obs have accumulated. |
| `~prediction_time` | `5.0` s | H × dt = 20 × 0.25. Tied to the trained schedule, do not change without retraining. |
| `~prediction_points` | `20` | Same — fixed by training. |
| `~collision_distance_threshold` | `1.0` m | TTC trigger distance. |
| `~latency_warn_ms` | `80.0` | Per-callback wall-clock warn threshold. |
| `~vehicle_speed_topic` | `/pacmod/vehicle_speed_rpt` | Override if remapping `/pacmod/`. |

## Available checkpoints

```
src/diffusion_prediction/models/diffusion/
├── av2_joint_v2/ema_best.pt        (32 MB, multi-agent, minFDE 0.529 m)
├── av2_pretrain_v2/ema_best.pt     (25 MB, single-agent, minFDE 0.693 m)
├── eth_ucy_ft_joint/ema_best.pt    (32 MB, joint fine-tuned on ETH/UCY)
└── synth_v2_ft/eth_ucy_ft_single/ema_best.pt (25 MB, single fine-tuned)
```

## End-to-end perception → diffusion → MPPI

The full ROS 1 chain that produces `/fusion_pedestrian_position` from
real sensors is now in place via the `_ros1.py` ports of
`adapt_full` and `yolo_person_detector`. To bring it up:

```bash
roslaunch adapt_full perception_full.launch
```

This single launch starts:

| Node | Pkg | In | Out |
|---|---|---|---|
| `lidar_preprocessor`        | `adapt_full`           | `/ouster/points` | `/lidar_pedestrian_position` |
| `rgbd_pedestrian_detector`  | `yolo_person_detector` | `/oak/rgb/image_raw`, `/oak/stereo/image_raw` | `/rgbd_pedestrian_position`, `/pedestrian_sign_present` |
| `sensor_fusion_node`        | `adapt_full`           | the two above | `/fusion_pedestrian_position` |
| `diffusion_predictor_node`  | `diffusion_prediction` | `/fusion_pedestrian_position`, `/pacmod/vehicle_speed_rpt` | `/pedestrian_predictions_tensor` (+ marker / motion / TTC) |

Args (all default to `true` / `joint` / `cuda:0`):
`enable_lidar`, `enable_rgbd`, `enable_fusion`, `enable_diffusion`,
`prediction_mode`, `diffusion_weights`, `device`.

Then bring up MPPI separately:

```bash
rosrun mppi_controller adapt_mppi_node_ros1_sim.py _prediction_source:=predicted
# or, on the real vehicle:
rosrun mppi_controller adapt_mppi_node_ros1.py     _prediction_source:=predicted
```

### Bench testing without sensors

For desk testing without LiDAR / camera, skip the perception layer and
synthesize fusion output directly:

```bash
roslaunch adapt_full perception_full.launch \
    enable_lidar:=false enable_rgbd:=false enable_fusion:=false
rosrun mppi_controller publish_fake_gps.py
# diffusion_predictor will pick up the synthetic /fusion_pedestrian_position
```

### What was *not* ported

These ROS 2 nodes in `adapt_full` are downstream of perception and are
**not** required to close the diffusion loop; they remain rclpy on this
branch:

- `adapt_high_level_command.py` (safety executive)
- `adapt_pedestrian_aware_path.py` (path planning)
- `adapt_stanley_controller.py` (alt controller — MPPI replaces it)
- `adapt_safety_controller.py` (empty stub)
- `adapt_camera_position.py` (spoof publisher)
- `adapt_straight_path.py` (alt path generator)

Add a `_ros1.py` sibling for any of these as needed, following the same
pattern as the three perception nodes.

The `detected_object_msgs` package is also not required — the
`/detected_objects` topic is only published if the package is
installed; the perception loop itself does not depend on it.

## Files added/touched in this port

```
src/diffusion_prediction/
├── CMakeLists.txt                          (new)
├── package.xml                             (rewritten ament → catkin)
├── setup.py                                (rewritten ament → catkin)
├── diffusion_prediction/
│   └── infer_node_ros1.py                  (new, executable)
├── launch/
│   └── diffusion_predictor.launch          (new)
└── docs/
    └── ros1_inference.md                   (this file)
```

The ROS 2 `infer_node.py` is left in tree for reference but is no longer
built; reuse modules (`model.py`, `model_joint.py`, `ddpm.py`,
`tracker.py`, `utils.py`) are unchanged and import-compatible from
either ROS distro.
