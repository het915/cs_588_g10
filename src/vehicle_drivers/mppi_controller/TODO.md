# Real-Car Debug TODO

Ordered from first boot to full-speed run. Check each box before moving to the next section.

---

## 1. Sensor Inputs

- [ ] `/navsatfix` is publishing — `ros2 topic echo /navsatfix`
  - Stuck at `lat=0, lon=0` → node silently does nothing (early-return guard in `_control_loop`)
- [ ] `/insnavgeod` is publishing non-NaN heading — `ros2 topic echo /insnavgeod`
  - If `septentrio_gnss_driver` is not sourced, falls back to GPS-derived heading with no warning beyond startup log
- [ ] `/pacmod/vehicle_speed_rpt` is publishing — `ros2 topic echo /pacmod/vehicle_speed_rpt`
  - Speed stuck at 0 → PID integral grows, filtered speed lags
- [ ] `/pacmod/enabled` publishes `true` before control starts
  - `require_pacmod_enable: true` in config — node does nothing until this fires

---

## 2. Heading Initialization

**Code:** `adapt_mppi_node.py` — `_gnss_cb` / `_ins_cb`

- [ ] Confirm `_has_valid_heading` becomes `True` at startup (check logs)
  - With INS: fires immediately on first non-NaN `/insnavgeod` message
  - Without INS: requires `speed > 0.5 m/s` AND `displacement > 2 m` from GPS anchor — car must move before control activates
- [ ] Verify `heading_to_yaw()` sign convention (`utils.py:48`)
  - Compass 0° (North) → ENU yaw = π/2 rad
  - Compass 90° (East) → ENU yaw = 0 rad
  - Test statically with known GPS heading before moving
- [ ] Confirm INS heading is not intermittently NaN during driving (causes heading to freeze)

---

## 3. Coordinate Transforms

**Code:** `utils.py` — `geodetic2enu`, `heading_to_yaw`, `_gem_state`

- [ ] Verify `origin_lat / origin_lon` in `mppi_params.yaml` match the physical test site
  - Wrong origin shifts the entire path by meters
  - Cross-check by driving to a known landmark and confirming ENU (x, y) matches expectation
- [ ] Verify `offset: 1.26 m` (GPS antenna → rear axle) by physically measuring on the vehicle
  - Wrong offset shifts vehicle pose forward/backward in the cost, causing the MPPI to steer off-center
- [ ] Confirm `geodetic2enu` output is plausible
  - At origin GPS coordinate → should give ENU (0, 0)

---

## 4. Waypoint Loading & Path Trimming

**Code:** `utils.py:118` — `load_waypoints` / `reference_path.py:44` — `trim_behind`

- [ ] Confirm CSV format is `lon, lat` per row (NOT `lat, lon`) — open file in text editor and verify against a GPS map
- [ ] Confirm `waypoints_csv` param resolves correctly
  - Empty string → falls back to `adapt_full/waypoints/track.csv`; confirm that file exists after `colcon build`
- [ ] In RViz, `/adapt/viz/reference_path` overlays the physical course and goes in the intended driving direction
  - Reversed → CSV row order is wrong
- [ ] **`trim_behind` (new logic)** — add a temporary log line in `_control_loop` to verify:
  ```python
  self.get_logger().info(f'active_path len={len(active_path.xy)}')
  ```
  - At startup the count should be less than the total waypoints if the vehicle starts mid-path
  - Near the end of the route the count should stay ≥ 4 (min_points clamp)
- [ ] Drive to near the last waypoint and confirm no crash from `ReferencePath(N<2)` at `trim_behind`

---

## 5. MPPI Parameters

**Config:** `mppi_params.yaml` — `mppi:` block  
**Code:** `mppi.py`

Start with reduced speed and conservative weights; increase only after each step is stable.

- [ ] `wheelbase: 1.75` — verify physically on the vehicle (**critical** — wrong value breaks bicycle model dynamics and all rollout trajectories)
- [ ] `K: 500, H: 30, dt: 0.1` — measure actual wall-clock time of `mppi.update(...)` on the car's hardware
  - Target: < 50 ms (20 Hz budget). If over, reduce `K` first (try 200), then `H`
  - Add `time.perf_counter()` around `mppi.update(...)` temporarily
- [ ] Monitor `ESS/K` in logs every tick
  - Target: > 0.1 consistently. Below 0.05 = weight collapse, effectively 1 sample
  - If low: increase `lambda_` (try 0.3–1.0) or increase `sigma_steer` / `sigma_accel`
- [ ] `w_pos: 15.0` — if car wanders off path, increase; if it oscillates on path, decrease
- [ ] `w_vel: 5.0` — if car ignores speed target in MPPI rollouts, increase
- [ ] `w_curv: 2.0` — if car slows excessively before curves, decrease
- [ ] `desired_speed: 4.0` — **start at 1.0 m/s** for first runs; increase only after steering and PID are verified
- [ ] **Known dead parameter:** `lookahead_m: 8.0` in `MPPI.__init__` is defined but never used in any cost function — do not rely on it

---

## 6. Speed PID & Control Limits

**Config:** `mppi_params.yaml` — `pid:` and `filter:` blocks  
**Code:** `adapt_mppi_node.py:373–388`

- [ ] Observe `thr=` and `brk=` values in logs; confirm they ramp smoothly
- [ ] `kp: 2.0` — if throttle oscillates, reduce; if car is sluggish, increase
- [ ] `ki: 0.0` — currently zero; car may plateau below target speed. Try `0.1–0.5` if steady-state speed error persists
- [ ] `kd: 0.1` — increase if speed oscillates; decrease if response is too slow
- [ ] `max_throttle: 0.4` and `max_brake: 0.4` — only raise after PID is stable
- [ ] `filter.cutoff: 1.2 Hz` — if speed signal is noisy, lower; if PID is sluggish, raise
- [ ] **Known config issue:** `filter.order: 4` is accepted but ignored — code always uses 1st-order EMA (`utils.py:99`)

---

## 7. Steering Output

**Code:** `utils.py:55` — `front2steer` / `adapt_mppi_node.py:369`

- [ ] Verify steering **sign**: positive MPPI delta (left turn) must turn the wheel left
  - Command a small static delta (e.g., 0.1 rad) with the car stationary and observe wheel direction
- [ ] Verify steering **magnitude**: `front2steer()` uses a GEM e4 polynomial calibration
  - If wheel over/under-steers relative to commands, the polynomial coefficients need recalibration for this specific vehicle
- [ ] `angular_velocity_limit: 4.0 rad/s` (steering rate) — if wheel can't keep up with fast commands, this is the limit; increase if the physical actuator supports it
- [ ] `delta_max: 0.61 rad` (≈35°) — verify this matches the physical steering lock limit

---

## 8. PACMod Enable Sequence

**Code:** `adapt_mppi_node.py:329` — `_prime_pacmod`

- [ ] Confirm gear code `3` = Drive on this vehicle's PACMod configuration (cross-check PACMod docs or other nodes in the repo)
- [ ] Confirm `clear_override: True` behavior — this clears any manual override; ensure that is intentional at startup
- [ ] Note: `_prime_pacmod` fires only once (`_pacmod_primed` flag). If PACMod drops and re-enables mid-run, the sequence does NOT re-run — the node will continue publishing commands but never re-sends gear/enable

---

## 9. Obstacle Transforms

**Code:** `adapt_mppi_node.py` — `_ped_cb` / `_pred_tensor_cb`

- [ ] **Raw pedestrian (`_ped_cb`):** confirm the `dist` unit in `/fusion_pedestrian_position`
  - Code treats it as meters (`float(data[i])`). If the fusion node publishes centimeters, obstacles will appear 100× too far away
- [ ] **Raw pedestrian:** verify the polar→ENU rotation is correct
  - With car facing North, an object detected 3 m directly ahead should map to ENU `(x, y+3)`. Test with a stationary known object
- [ ] **Predicted trajectories (`_pred_tensor_cb`):** confirm the diffusion node publishes in **vehicle frame** (code applies ego rotation + translation to convert to world frame)
  - If the node already publishes world-frame, the transform will double-apply and obstacles will appear in wrong positions
- [ ] Confirm prediction tensor shape is `(M, 20, 2)` — code hardcodes `_PED_H = 20` steps at `_PED_DT = 0.25 s`

---

## 10. RViz Visualization

- [ ] RViz fixed frame is set to `map` (matches `viz.frame_id: "map"` in config) — otherwise nothing renders
- [ ] `/adapt/viz/reference_path` — overlays physical course correctly
- [ ] `/adapt/viz/robot_trajectory` — driven path; compare visually to reference to assess tracking error
- [ ] `/adapt/viz/chosen_trajectory` — MPPI mean rollout; should point forward along path, not backward
- [ ] `/adapt/viz/sampled_trajectories` — increase `viz.num_samples` from `1` to `20–50` for debugging rollout diversity
- [ ] `/adapt/viz/obstacles` — markers appear at correct real-world positions

---

## 11. First-Run Order of Operations

1. **Static check** — `require_pacmod_enable: false`, `desired_speed: 0.0`
   - Verify all sensor topics, ENU positions, and heading in logs without moving
   - Confirm RViz reference path overlays correctly

2. **Heading + transform check** — drive slowly by hand (if safe) to confirm `_has_valid_heading` flips and ENU pose tracks correctly

3. **Steering direction** — `desired_speed: 1.0`, short straight path
   - Observe wheel direction vs MPPI delta sign in logs

4. **PID tuning** — verify `thr=` / `brk=` ramp smoothly to reach target speed

5. **`trim_behind` check** — confirm `active_path` length decreases as car progresses along the route

6. **Obstacle avoidance** — introduce a stationary known object; confirm MPPI rollouts steer around it in RViz

7. **Full speed** — increase `desired_speed` toward `4.0 m/s` only after all above pass
