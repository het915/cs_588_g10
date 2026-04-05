# PACMod2 ROS2 Topic Map - Polaris GEM Vehicle Control

This document maps all PACMod2 ROS2 topics used to control the Polaris GEM e2/e4 vehicles.

All message types are from `pacmod2_msgs.msg` unless noted otherwise.

---

## Command Topics (Autonomous System -> PACMod)

### `/pacmod/global_cmd`
- **Type**: `GlobalCmd`
- **Purpose**: Master enable/disable for autonomous vehicle control
- **Fields**:
  - `enable` (bool) - `True` to allow autonomous commands, `False` to hand back to manual
  - `clear_override` (bool) - `True` to clear any manual override flags

### `/pacmod/steering_cmd`
- **Type**: `PositionWithSpeed`
- **Purpose**: Steering wheel angle command
- **Fields**:
  - `angular_position` (float, radians) - Desired steering wheel angle
  - `angular_velocity_limit` (float) - Max steering rate, default `4.0` rad/s
- **Notes**:
  - Front wheel angle is clamped to +/- 35 degrees
  - Converted to steering wheel angle via polynomial: `steer = -0.1084 * angle^2 + 21.775 * angle`

### `/pacmod/accel_cmd`
- **Type**: `SystemCmdFloat`
- **Purpose**: Throttle / acceleration command
- **Fields**:
  - `command` (float) - Acceleration in m/s^2
- **Limits**: Hardcoded cap at `2.0` m/s^2, config default `0.5` m/s^2
- **Notes**: Computed by PID speed controller (kp=0.6, ki=0.0, kd=0.1)

### `/pacmod/brake_cmd`
- **Type**: `SystemCmdFloat`
- **Purpose**: Brake pedal command
- **Fields**:
  - `command` (float) - Normalized 0.0 to 1.0
- **Notes**: Currently always set to `0.0` in the pure pursuit controller

### `/pacmod/shift_cmd`
- **Type**: `SystemCmdInt`
- **Purpose**: Gear selection
- **Values**:
  | Value | Gear     |
  |-------|----------|
  | `2`   | Neutral  |
  | `3`   | Forward  |
- **Notes**: Initialized to Neutral (`2`), set to Forward (`3`) on enable

### `/pacmod/turn_cmd`
- **Type**: `SystemCmdInt`
- **Purpose**: Turn signal control
- **Values**:
  | Value | Signal       |
  |-------|-------------|
  | `1`   | Off          |
  | `3`   | Hazard lights |
- **Notes**: Hazards (`3`) activated during autonomous mode, Off (`1`) on disable

---

## Feedback Topics (PACMod -> Autonomous System)

### `/pacmod/enabled`
- **Type**: `std_msgs/Bool`
- **Purpose**: Reports whether PACMod is currently accepting autonomous commands
- **Field**: `data` (bool)

### `/pacmod/vehicle_speed_rpt`
- **Type**: `VehicleSpeedRpt`
- **Purpose**: Current vehicle speed
- **Field**: `vehicle_speed` (float, m/s)
- **Notes**: Filtered through a 4th-order Butterworth low-pass filter (cutoff 1.2 Hz, fs 30 Hz)

### `/pacmod/steering_rpt`
- **Type**: `SystemRptFloat`
- **Purpose**: Actual steering wheel angle feedback
- **Field**: `output` (float, radians)

---

## Control Sequence

### Enabling Autonomous Control (Joystick LB + RB)

```
1. /pacmod/global_cmd   -> enable=True, clear_override=True
2. /pacmod/shift_cmd    -> command=3 (Forward)
3. /pacmod/brake_cmd    -> command=0.0
4. /pacmod/accel_cmd    -> command=0.0
5. /pacmod/turn_cmd     -> command=3 (Hazards)
```

### Autonomous Driving Loop (20 Hz)

```
1. Read GNSS position    <- /navsatfix (sensor_msgs/NavSatFix)
2. Read INS heading      <- /insnavgeod (septentrio_gnss_driver/INSNavGeod)
3. Read vehicle speed    <- /pacmod/vehicle_speed_rpt
4. Compute pure pursuit steering angle
5. Compute PID throttle from speed error (desired_speed - current_speed)
6. Publish /pacmod/steering_cmd
7. Publish /pacmod/accel_cmd
8. Publish /pacmod/brake_cmd (0.0)
9. Publish /pacmod/global_cmd (keep-alive enable=True)
```

### Disabling Autonomous Control (Joystick LB only)

```
1. /pacmod/global_cmd   -> enable=False
2. /pacmod/turn_cmd     -> command=1 (Off)
```

---

## Configuration

### PACMod Hardware Config (`basic_launch/config/{e2,e4}/pacmod/pacmod.yaml`)

| Parameter | Value |
|-----------|-------|
| `pacmod_vehicle_type` | `POLARIS_GEM` |
| `controller_type` | `LOGITECH_F310` |
| `steering_max_speed` | `3.3` rad/s |
| `max_veh_speed` | `11.176` m/s |
| `pacmod_socketcan_device` | `can0` |

### Pure Pursuit Controller Config (`gem_gnss_control/config/{e2,e4}_pp.yaml`)

| Parameter | e4 | e2 |
|-----------|-----|-----|
| `wheelbase` | 2.57 m | 1.75 m |
| `desired_speed` | 2.0 m/s | 2.0 m/s |
| `look_ahead` | 5.0 m | 5.0 m |
| `max_accel` | 0.5 m/s^2 | 0.5 m/s^2 |
| `offset` | 1.26 m | 0.46 m |

### Safety Caps (Hardcoded in `pure_pursuit.py`)

| Limit | Value |
|-------|-------|
| Max speed | 5.0 m/s |
| Max acceleration | 2.0 m/s^2 |
| Max front wheel angle | +/- 35 degrees |

---

## ROS Graph

```
                    +-----------------+
  /navsatfix ------>|                 |------> /pacmod/global_cmd
  /insnavgeod ----->|  pure_pursuit   |------> /pacmod/steering_cmd
  /pacmod/enabled ->|     node        |------> /pacmod/accel_cmd
  /pacmod/         >|                 |------> /pacmod/brake_cmd
   vehicle_speed_rpt|                 |------> /pacmod/shift_cmd
                    +-----------------+------> /pacmod/turn_cmd
```