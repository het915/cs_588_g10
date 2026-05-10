# Diffusion Predictor — Output Contract

This document specifies the **published interface** of the ROS 1 diffusion
predictor node (`diffusion_prediction/infer_node_ros1.py`). It is the
authoritative reference for any downstream consumer — the MPPI controller
already implements it; this doc exists so future consumers (or modified
consumers) can match the contract without reading the publisher source.

> Scope: this doc describes only what is **published**. The model
> internals, training, weights, and runbook live in
> [`ros1_inference.md`](./ros1_inference.md).

---

## 1. Topics at a glance

| # | Topic | Type | Role |
|---|---|---|---|
| 1 | `/pedestrian_predictions_tensor` | `std_msgs/Float32MultiArray` | **Primary.** Full predicted trajectories for all tracked pedestrians. This is what MPPI consumes. |
| 2 | `/person_prediction` | `visualization_msgs/Marker` | RViz LINE_STRIP for the primary pedestrian's path. Visualization only. |
| 3 | `/pedestrian_motion` | `geometry_msgs/Twist` | Primary pedestrian's current position in `linear.x` / `linear.y`. Convenience scalar. |
| 4 | `/pedestrian_ttc` | `std_msgs/Float64` | Time-to-collision in seconds, `inf` when no collision predicted. |

The rest of this document covers topic 1 only.

---

## 2. `/pedestrian_predictions_tensor` — definition

### 2.1 Type

`std_msgs/Float32MultiArray`

### 2.2 Shape

A row-major flatten of a `float32` ndarray with shape **`(M, H, 2)`**:

| Axis | Symbol | Meaning | Size |
|---|---|---|---|
| 0 | `M` | Number of pedestrians in the message | variable, ≥ 0 |
| 1 | `H` | Prediction horizon (time steps) | **fixed = 20** |
| 2 | `xy` | Cartesian coordinates `(x, y)` | fixed = 2 |

`H` is hard-bound to the trained schedule and **does not change at
runtime**. `M` varies per message.

### 2.3 `layout.dim` field

The publisher sets all three dimensions. Consumers should read sizes from
`layout.dim`, not assume them — although `H = 20` and the inner pair is
always `xy`.

```python
msg.layout.dim = [
    MultiArrayDimension(label="M",  size=M, stride=M*H*2),  # e.g. 3, 120
    MultiArrayDimension(label="H",  size=H, stride=H*2),    # 20,  40
    MultiArrayDimension(label="xy", size=2, stride=2),      #  2,   2
]
msg.layout.data_offset = 0
```

### 2.4 `data` field

- Length: `M * H * 2` (with `H = 20`, that is `M * 40` floats).
- Dtype: `float32`.
- Layout: row-major (C-order) flatten of `(M, H, 2)` — the slowest-varying
  index is pedestrian, then time step, then coordinate. **`x` and `y`
  alternate** at the innermost level.

### 2.5 Frame and units

- **Frame:** `base_link` — ROS REP-103 vehicle-fixed frame, right-handed,
  origin at the rear-axle/vehicle reference point.
- **Axes:**
  - **+x** points **forward** (out the front of the car).
  - **+y** points to the **driver's left** (US LHD = left side of vehicle).
  - **+z** points **up** (out of the road plane).
- **Units:** meters.
- **Handedness:** right-handed. From a top-down view with +x forward and
  +z out of the page (toward the viewer), the right-hand rule forces +y
  to the **left**.

```
                +x  (forward)
                 ▲
                 │
                 │
   +y ◄────────[ car ]               (+z is out of the page, toward you)
  (left)         │
                 │
                 ▼
                -x  (rear)
```

**Sign cheatsheet for any `(x, y)` pair in the tensor:**

| Value | Meaning |
|---|---|
| `x > 0` | ahead of the car |
| `x < 0` | behind the car |
| `y > 0` | to the **left** (driver's side, US LHD) |
| `y < 0` | to the **right** (passenger's side) |

Verifiable in code at `diffusion_prediction/utils.py:25-27` — the polar
→ Cartesian decode used on incoming detections comments `y = -d*cos(θ)
# lateral (left positive)`, which is the same convention propagated to
the predicted trajectories.

**Example (from §4):** Ped 1 starts at `(10.0, -3.0)` and ends at
`(10.0, +2.0)`. Reading: a pedestrian 10 m **ahead**, starting 3 m to
the **right**, walking across to 2 m to the **left** over the 5 s
horizon.

- **No header.** `Float32MultiArray` has no `std_msgs/Header`, so there is
  no per-message stamp or `frame_id` on the wire. The publisher emits the
  tensor at the rate of the upstream `/fusion_pedestrian_position`
  callback (≈ 10 Hz). Consumers should treat each message as "current as
  of receipt" and use their own ego state at receipt time when rotating
  into the world frame.

### 2.6 Time semantics

| Quantity | Value |
|---|---|
| Step duration `dt` | **0.25 s** |
| Horizon `H` | **20 steps** |
| Lookahead | `H * dt = 5.0 s` |
| First step `h = 0` | predicted position at `t + dt` (i.e. **0.25 s into the future**, not the present) |

### 2.7 Where the contract is implemented

The shape, labels, and strides described above are not just documentation —
they are constructed in code with `std_msgs/MultiArrayDimension` objects on
the producer side, and read back via the same field on the consumer side.
If you ever need to verify the contract against reality, these are the
authoritative call sites.

#### Producer — constructs the dims

**`src/diffusion_prediction/diffusion_prediction/utils.py:151`** —
`build_predictions_tensor()` is the canonical builder. The diffusion node
calls it every callback (`infer_node_ros1.py:378`):

```python
def build_predictions_tensor(trajectories: np.ndarray):
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
    M, H, _ = trajectories.shape

    msg = Float32MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="M",  size=M, stride=M * H * 2),
        MultiArrayDimension(label="H",  size=H, stride=H * 2),
        MultiArrayDimension(label="xy", size=2, stride=2),
    ]
    msg.data = trajectories.astype(np.float32).flatten().tolist()
    return msg
```

A parallel producer at
**`src/yolo_person_detector/yolo_person_detector/pedestrian_behaviour_predictor.py:460-462`**
builds the same `(M, H, 2)` shape with identical labels — an alternative
trajectory producer that follows this contract.

#### Consumer — reads `.size` off the dims

The MPPI consumer **does not call `MultiArrayDimension`** — the
constructor only runs on the producer. ROS deserialization turns the
wire bytes back into a `Float32MultiArray` whose `layout.dim` is already
a list of populated `MultiArrayDimension` objects; the consumer just
reads `.size`. From `adapt_mppi_node_ros1.py:617-625`:

```python
dims = msg.layout.dim
if len(dims) < 2:
    self.ped_trajectories = None
    return
M, H = dims[0].size, dims[1].size
...
arr = np.array(msg.data, dtype=np.float32).reshape(M, H, 2)
```

#### Asymmetry summary

| Side | Touches `MultiArrayDimension`? | What it does |
|---|---|---|
| Producer (diffusion predictor) | **Yes** — constructs 3 `MultiArrayDimension` objects per message | Sets `label`, `size`, `stride` for `M`, `H`, `xy`. |
| Consumer (MPPI) | **No** — only reads `.size` from existing dims | Validates `len(dims) >= 2`, pulls `M` and `H`, reshapes `data`. |

This is the standard ROS pattern: **building** a `MultiArray` requires
constructing dimension objects; **reading** one does not.

Both `H` and `dt` are baked into the model's training schedule. Changing
them requires retraining; do not assume runtime configurability.

---

## 3. Indexing formula

For pedestrian `m ∈ [0, M)` and time step `h ∈ [0, H)`:

```
x = data[m * H * 2 + h * 2 + 0]
y = data[m * H * 2 + h * 2 + 1]
```

With `H = 20` (the only value that ships):

```
x = data[m * 40 + h * 2 + 0]
y = data[m * 40 + h * 2 + 1]
```

Equivalently, the canonical NumPy decode:

```python
arr = np.asarray(msg.data, dtype=np.float32).reshape(M, H, 2)  # base_link
# arr[m, h, 0] -> x;  arr[m, h, 1] -> y
```

---

## 4. Worked example — 3 pedestrians

`M = 3`, `H = 20`, total `data` length = **120 floats**.

### 4.1 Scenario (base_link frame)

| Ped | Initial (x, y) m | Motion |
|---|---|---|
| 0 | `(7.0,  1.0)` | walking forward at 1.4 m/s |
| 1 | `(10.0, -3.0)` | walking left (toward +y) at 1.0 m/s |
| 2 | `(5.0, -0.5)` | near-stationary, 0.05 m/s forward |

### 4.2 Builder (Python)

```python
import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

H, dt = 20, 0.25
trajs = np.zeros((3, H, 2), dtype=np.float32)

# Ped 0: x = 7 + 1.4*t,  y = 1.0
trajs[0, :, 0] = 7.0 + 1.4 * dt * np.arange(1, H + 1)
trajs[0, :, 1] = 1.0

# Ped 1: x = 10,  y = -3 + 1.0*t
trajs[1, :, 0] = 10.0
trajs[1, :, 1] = -3.0 + 1.0 * dt * np.arange(1, H + 1)

# Ped 2: x = 5 + 0.05*t,  y = -0.5
trajs[2, :, 0] = 5.0 + 0.05 * dt * np.arange(1, H + 1)
trajs[2, :, 1] = -0.5

M = trajs.shape[0]  # 3
msg = Float32MultiArray()
msg.layout.dim = [
    MultiArrayDimension(label="M",  size=M, stride=M*H*2),  # 3, 120
    MultiArrayDimension(label="H",  size=H, stride=H*2),    # 20, 40
    MultiArrayDimension(label="xy", size=2, stride=2),      # 2, 2
]
msg.layout.data_offset = 0
msg.data = trajs.flatten().tolist()  # length 120
```

### 4.3 What `rostopic echo -n1` prints

```yaml
layout:
  dim:
    - { label: "M",  size: 3,  stride: 120 }
    - { label: "H",  size: 20, stride: 40 }
    - { label: "xy", size: 2,  stride: 2 }
  data_offset: 0
data: [
  # --- Ped 0 (indices 0..39): 20 (x,y) pairs ---
   7.35, 1.0,   7.70, 1.0,   8.05, 1.0,   8.40, 1.0,   8.75, 1.0,
   9.10, 1.0,   9.45, 1.0,   9.80, 1.0,  10.15, 1.0,  10.50, 1.0,
  10.85, 1.0,  11.20, 1.0,  11.55, 1.0,  11.90, 1.0,  12.25, 1.0,
  12.60, 1.0,  12.95, 1.0,  13.30, 1.0,  13.65, 1.0,  14.00, 1.0,
  # --- Ped 1 (indices 40..79) ---
  10.0, -2.75, 10.0, -2.50, 10.0, -2.25, 10.0, -2.00, 10.0, -1.75,
  10.0, -1.50, 10.0, -1.25, 10.0, -1.00, 10.0, -0.75, 10.0, -0.50,
  10.0, -0.25, 10.0,  0.00, 10.0,  0.25, 10.0,  0.50, 10.0,  0.75,
  10.0,  1.00, 10.0,  1.25, 10.0,  1.50, 10.0,  1.75, 10.0,  2.00,
  # --- Ped 2 (indices 80..119) ---
  5.0125, -0.5, 5.0250, -0.5, 5.0375, -0.5, 5.0500, -0.5, 5.0625, -0.5,
  5.0750, -0.5, 5.0875, -0.5, 5.1000, -0.5, 5.1125, -0.5, 5.1250, -0.5,
  5.1375, -0.5, 5.1500, -0.5, 5.1625, -0.5, 5.1750, -0.5, 5.1875, -0.5,
  5.2000, -0.5, 5.2125, -0.5, 5.2250, -0.5, 5.2375, -0.5, 5.2500, -0.5,
]
```

### 4.4 Spot-check decode

```python
arr = np.asarray(msg.data, dtype=np.float32).reshape(3, 20, 2)

arr[0,  0]  # -> [ 7.35,  1.0  ]   ped 0, first future step  (t = 0.25 s)
arr[0, -1]  # -> [14.00,  1.0  ]   ped 0, last future step   (t = 5.00 s)
arr[1,  5]  # -> [10.00, -1.50]    ped 1 at h=5              (t = 1.50 s)
arr[2, -1]  # -> [ 5.25, -0.5 ]    ped 2 at h=19             (t = 5.00 s)
```

---

## 5. Reference decode (rotate into world frame)

The published tensor is in `base_link`. Most planners want world-frame
positions. The MPPI consumer uses the snippet below — replicate it (or
its equivalent) on receipt:

```python
M = msg.layout.dim[0].size
H = msg.layout.dim[1].size
arr = np.asarray(msg.data, dtype=np.float32).reshape(M, H, 2)  # base_link

# Ego pose at receipt time:
ex, ey, yaw = ego_state()   # world-frame x, y, yaw (radians)
c, s = math.cos(yaw), math.sin(yaw)

world = np.empty_like(arr)
world[:, :, 0] = c * arr[:, :, 0] - s * arr[:, :, 1] + ex
world[:, :, 1] = s * arr[:, :, 0] + c * arr[:, :, 1] + ey
# world[m, h] is now the predicted (x, y) of pedestrian m at step h
# in the world frame.
```

Step `h` corresponds to absolute time `t_receipt + (h + 1) * 0.25 s`.

---

## 6. Edge cases the consumer must handle

The publisher emits a "no-data" message under several conditions; treat
any of the following as **"no current prediction"** and fall back to
whatever default the planner uses (e.g. raw detections, hold the previous
prediction, or treat as no obstacles):

| Condition | What you'll see | Recommended action |
|---|---|---|
| No tracks with enough history | `data` empty, `dim[0].size = 0` | Skip; do not reshape. |
| Empty `msg.data` | length 0 regardless of dims | Skip. |
| Fewer than 2 dims in `layout.dim` | malformed | Skip; log once. |
| `M == 0` or `H == 0` | shape valid but degenerate | Skip. |
| Stale message (no new publication) | nothing on the topic | Apply your own freshness timeout. The publisher carries no stamp; track receipt time on your side. |

The publisher will **not** send NaNs or Infs in `data` for valid
predictions. Defensive consumers can still mask `~np.isfinite(arr)` if
desired.

---

## 7. What is *not* in this contract

- **Per-pedestrian sample modes (`K`).** The model internally draws
  `K = 20` diffusion samples per pedestrian and then collapses them to
  one chosen mode (sticky-closest-to-mean) before publishing. The wire
  format is one trajectory per pedestrian — consumers do not see the
  full posterior.
- **Track identifiers.** Pedestrian index `m` is an arbitrary slot in
  this message; it is not stable across messages and is not the
  tracker's `track_id`. Do not key state on `m`.
- **Confidence scores.** Not published. If you need them, the
  contract has to be extended (additive: a parallel `Float32MultiArray`
  on a sibling topic, or an upgrade to a custom message).
- **History.** Only the **future** `(M, H, 2)` is published. Past track
  positions are internal to the predictor.
- **Header / stamp / frame_id.** `Float32MultiArray` has no header; see
  §2.5. Frame is implicitly `base_link`.

---

## 8. Stability and versioning

The fields `H = 20`, `dt = 0.25 s`, frame = `base_link`, and the
`(M, H, 2)` layout are considered **stable** for any consumer on this
branch. Any change to those would be a breaking interface change and
should be coordinated across publisher and consumer.

If you need to extend the contract (e.g. add multi-modal samples,
confidences, or stamps), prefer one of:

1. A new sibling topic, leaving this one untouched.
2. A new custom message type under a new topic name.

Do not redefine the meaning of existing fields.

---

## 9. Quick sanity commands

Confirm the contract on a running system:

```bash
# Topic exists and is the right type
rostopic info /pedestrian_predictions_tensor

# Rate ~10 Hz when fusion is active
rostopic hz   /pedestrian_predictions_tensor

# One message, full layout + data
rostopic echo -n1 /pedestrian_predictions_tensor | head -60

# Dimension sanity (M variable, H must be 20, xy must be 2)
rostopic echo -n1 /pedestrian_predictions_tensor/layout/dim
```

If `dim[1].size != 20` or `dim[2].size != 2` on this branch, something
upstream is wrong — file it as a publisher bug, not a consumer issue.

---

## 10. Sim pipeline integration (perception → diffusion → MPPI)

This output is the third hop in a four-stage chain. Consumers don't need
to understand the upstream stages to use the contract above, but the
information below clarifies where the data comes from and what to expect
when running the full pipeline in **Polaris GEM Gazebo simulation**.

### 10.1 End-to-end data flow

```
GEM Gazebo sim                Perception (rospy)            Prediction        Control
                                                                                 
/ouster/points  ──►  adapt_lidar_processing_ros1.py
(PointCloud2)        pkg=adapt_full
                       │
                       ▼ /lidar_pedestrian_position
                         (Int32MultiArray, [d_int_m, deg_int])

/oak/rgb/image_raw ──┐
/oak/stereo/image_raw┴►  rgbd_pedestrian_detector_ros1.py
(depth Image)            pkg=yolo_person_detector
                       │
                       ▼ /rgbd_pedestrian_position
                         (Int32MultiArray, [d_int_m, deg_int])
                                  │
                                  ▼
                       adapt_lidar_camera_fusion_ros1.py    (pkg=adapt_full)
                       ApproximateTimeSync (slop=0.1 s)
                       polar → Cartesian → NN match → polar
                                  │
                                  ▼ /fusion_pedestrian_position
                                    (Int32MultiArray, polar pairs, base_link)
                                  │
                                  ▼
                       infer_node_ros1.py                   (pkg=diffusion_prediction)
                       decode → tracker → joint diffusion model
                                  │
                                  ▼ /pedestrian_predictions_tensor
                                    (Float32MultiArray, base_link, (M,20,2))
                                  │
                                  ▼
                       adapt_mppi_node_ros1_sim.py          (pkg=mppi_controller)
                       _prediction_source:=predicted
                       _pred_tensor_cb (rotate base_link → world)
                       _ped_t_map (MPPI dt 0.1 ↔ ped dt 0.25)
                                  │
                                  ▼ /adapt/viz/* (no PACMod cmds in sim variant)
```

`adapt_full/launch/perception_full.launch` brings up the first three
perception nodes plus the diffusion predictor in one shot. The sim MPPI
node is launched separately with `_prediction_source:=predicted`.

### 10.2 Hop-by-hop compatibility

| Hop | Topic | Type | Producer | Consumer | Match |
|---|---|---|---|---|---|
| 1a | `/lidar_pedestrian_position` | `Int32MultiArray` (polar pairs) | `lidar_preprocessor` | `sensor_fusion_node` | ✓ |
| 1b | `/rgbd_pedestrian_position` | `Int32MultiArray` (polar pairs) | `rgbd_pedestrian_detector` | `sensor_fusion_node` | ✓ |
| 2 | `/fusion_pedestrian_position` | `Int32MultiArray` (polar pairs, base_link) | `sensor_fusion_node` | `diffusion_predictor_node` | ✓ |
| 3 | `/pedestrian_predictions_tensor` | `Float32MultiArray` (`(M,20,2)`, base_link) | `diffusion_predictor_node` | `adapt_mppi_node_ros1_sim` (`_pred_tensor_cb`) | ✓ |

All polar hops use the shape `[dist_int, deg_int, dist_int, deg_int, …]`.
All Cartesian hops are in `base_link` (x forward, y left, z up).

### 10.3 Known concerns to verify when running in sim

#### (a) Bearing convention divergence between LiDAR and YOLO encoders

Tracing each component for a pedestrian **directly ahead** of the
vehicle:

| Component | File:line | Computes | Result for "ahead" |
|---|---|---|---|
| LiDAR encoder | `adapt_lidar_processing_ros1.py:507` | `atan2(-cx, cy)`, `cx,cy` from OS1 cloud (cx≈forward) | **270°** |
| YOLO encoder | `rgbd_pedestrian_detector_ros1.py:184` | `arctan2(Z_base, -Y_base)` | **90°** |
| Diffusion decoder | `diffusion_prediction/utils.py:25-26` | `(sin θ, -cos θ)` → forward at θ=90° | expects **90°** |
| Fusion decoder | `adapt_lidar_camera_fusion_ros1.py:88-90` | `(cos θ, sin θ)` → forward at θ=0° | expects **0°** |

If the LiDAR really emits 270° for "forward", the fusion node's NN
matcher will see the LiDAR detection at `(0, -d)` while the YOLO
detection of the same person sits at `(0, +d)` — they're 180° apart and
will never associate, and the diffusion node will misinterpret bearings
systematically.

**Verify empirically before trusting end-to-end output.** With a person
~5 m directly in front of the vehicle:

```bash
rostopic echo -n1 /lidar_pedestrian_position
rostopic echo -n1 /rgbd_pedestrian_position
```

If the second integer (degrees) disagrees by 180° between the two, this
is a real bug — likely fixed by adjusting the lidar's `atan2` quadrants.

#### (b) No depth image in GEM Gazebo sim

`gem_simulator/gem_description/urdf/gem_e4.gazebo:460-463` uses
`libgazebo_ros_multicamera.so`, which publishes RGB **stereo pair**
topics:
- `stereo/camera/left/image_raw`
- `stereo/camera/right/image_raw`

But `rgbd_pedestrian_detector_ros1.py:165-168` expects
`/oak/stereo/image_raw` to be a **depth image** (each pixel = depth in
meters). The sim does not produce that. Three workarounds, in
increasing fidelity:

1. **Run lidar-only:** `roslaunch adapt_full perception_full.launch enable_rgbd:=false`. Fusion falls back to lidar-only detections (`adapt_lidar_camera_fusion_ros1.py:178-182`).
2. **Add `stereo_image_proc`** to compute disparity from the stereo pair, convert to depth, and remap to `/oak/stereo/image_raw`.
3. **Patch the GEM URDF** to add a `libgazebo_ros_depth_camera.so` plugin parented to `stereo_camera_link`, remapped to `/oak/stereo/image_raw`. Highest fidelity, requires a sim-side change.

#### (c) LiDAR works in sim — but only with `velodyne_points:=true`

`gem_e4.urdf.xacro:11` defaults `velodyne_points` to **false**. With it
false, `/ouster/points` is never published and `lidar_preprocessor`
silently waits forever. Launch with `velodyne_points:=true` to
instantiate the OS1 plugin (`gem_e4.urdf.xacro:1091`).

#### (d) Sim MPPI needs GPS + INS topic remaps (or the fake GPS node)

The GEM sim publishes both NavSatFix and INSNavGeod natively
(`gem_simulator/gem_gazebo/scripts/insnavgeod_publisher.py`, started by
default in `gem_init.launch` with no `if`-guard). Only the topic names
differ from what `adapt_mppi_node_ros1_sim` expects:

| Topic sim MPPI needs | What GEM sim publishes | Action |
|---|---|---|
| `/navsatfix` (`NavSatFix`) | `/gps/fix` (`NavSatFix`) | Topic remap |
| `/insnavgeod` (`INSNavGeod`) | `/septentrio_gnss/insnavgeod` (`INSNavGeod`) | Topic remap |

Two options:

1. **Use the sim's real (moving) GPS+INS** — preferred for realistic
   testing. Add `<remap>` tags when launching the MPPI node, or use
   `rosrun … _navsatfix_topic:=/gps/fix` style if the node exposes
   params. Quickest in shell:
   ```bash
   rosrun mppi_controller adapt_mppi_node_ros1_sim.py \
       _prediction_source:=predicted \
       /navsatfix:=/gps/fix \
       /insnavgeod:=/septentrio_gnss/insnavgeod
   ```
2. **Use `publish_fake_gps.py --no-obstacle`** for a static-GPS bench
   test. Easier when you don't care about ego motion (e.g. validating
   the prediction → cost path with a stationary vehicle).

### 10.4 Recommended sim launch sequence

```bash
# Term 1 — sim with OS1 enabled, in a world that already has a pedestrian
bash ~/UIUC/AVSE/POLARIS_GEM_Simulator/run_docker_container.sh
cd ~/host/gem_simulation_ws && catkin_make && source devel/setup.bash
roslaunch gem_launch gem_init.launch \
    world_name:=parking.world vehicle_name:=e4 velodyne_points:=true
# Worlds with pre-spawned walking pedestrians:
#   parking.world      — single pedestrian on a back-and-forth trajectory
#   highbay_track.world / high_bay_3d.world — pedestrian1 actor near the track
# All driven by gem_gazebo/scripts/spawn_agents.py from the matching
# scene yaml under gem_gazebo/scenes/. Use track1.world if you want an
# empty world and prefer to spawn a pedestrian manually with `gz model`.

# Each subsequent terminal:
conda activate adapt-py310
source /opt/ros/noetic/setup.bash
source ~/UIUC/AVSE/cs_588_g10/devel/setup.bash

# Term 2 — fake GPS+INS (no fake ped — diffusion is the only ped source)
rosrun mppi_controller publish_fake_gps.py --no-obstacle

# Term 3 — perception + diffusion (lidar-only, no depth in sim)
roslaunch adapt_full perception_full.launch enable_rgbd:=false

# Term 4 — sim MPPI consuming diffusion output
rosrun mppi_controller adapt_mppi_node_ros1_sim.py _prediction_source:=predicted

# Term 5 (optional) — sanity
rostopic hz /lidar_pedestrian_position
rostopic hz /fusion_pedestrian_position
rostopic hz /pedestrian_predictions_tensor      # ~10 Hz if all is well
rostopic echo -n1 /pedestrian_predictions_tensor/layout/dim
```

Spawn a pedestrian model (or walk a Gazebo `actor`) within OS1 range to
exercise the chain.

### 10.5 Sim readiness summary

| Aspect | Status |
|---|---|
| Topic + type contracts at every hop | ✓ all match |
| Frame conventions (Cartesian, base_link) | ✓ consistent (subject to §10.3(a)) |
| `(M, H=20, 2)` diffusion → sim MPPI | ✓ designed-for, hard-coded `_PED_DT=0.25 _PED_H=20` |
| LiDAR sensor topic in sim | ✓ if `velodyne_points:=true` |
| RGB-D depth topic in sim | ❌ no depth image producer in GEM sim |
| GPS/INS for MPPI in sim | ✓ sim publishes both natively, just remap topics (or use `publish_fake_gps.py`) |
| LiDAR vs YOLO bearing convention | ⚠️ 180° suspect — verify empirically |
| Diffusion model weights | ✓ present in tree |

End-to-end works in sim **with `enable_rgbd:=false`** and the fake GPS
publisher in parallel. Solving the depth-image gap and the bearing
convention check are what's needed for a full multi-modal sim test.
