# CS_588_G10 ROS1 Integrated Bringup Guide

Integrated system: SAM Perception + MPPI Controller

## Frame Configuration
- **world**: Gazebo reference frame
- **map**: Navigation frame (synced to robot startup position via `map_tf_broadcaster`)
- **base_link**: Robot center
- All nodes use **map** frame

## Initial Setup (once)

1. Install Python packages
```bash
pip3 install --user -r ~/host/Downloads/temp/cs_588_g10/src/gem_perception/requirements.txt
```

2. Download SAM models
```bash
cd ~/host/Downloads/temp/cs_588_g10/src/gem_perception
python3 scripts/download_models.py
```

3. Build ROS packages
```bash
cd ~/host/Downloads/temp/cs_588_g10
catkin_make_isolated --merge --ignore-pkg nmea_msgs septentrio_gnss_driver --only-pkg-with-deps gem_perception
source devel_isolated/setup.bash
```

## Running the System (3 terminals in Docker)

**Terminal 1: Gazebo**
```bash
cd ~/host/Downloads/temp/gem_simulation_ws
source devel/setup.bash
roslaunch gem_launch gem_init.launch world_name:="highbay_track.world" x:=12.5 y:=-21 yaw:=3.1416 custom_scene:=true
```

**Terminal 2: Perception + TF Broadcaster + RViz**
```bash
cd ~/host/Downloads/temp/cs_588_g10
source devel_isolated/setup.bash
roslaunch gem_perception perception_sam.launch default_prompt:="red sign"
```

Expected output:
```
[INFO] map_tf_broadcaster: Aligning map to Gazebo initial position
[INFO] gem_perception (LangSAM) ready
```

**Terminal 3: MPPI Controller**
```bash
cd ~/host/Downloads/temp/cs_588_g10/src/vehicle_drivers/mppi_controller/mppi_controller
source ~/host/Downloads/temp/cs_588_g10/devel_isolated/setup.bash
python3 adapt_mppi_node_ros1_sim.py
```

**Optional - Change detection target:**
```bash
rostopic pub -1 /perception/prompt std_msgs/String "car"
```

## Debugging

Verify map frame exists:
```bash
rosrun tf tf_echo map base_link
```

Verify goal topic:
```bash
rostopic echo /move_base_simple/goal | head -20
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "map" frame not found | `map_tf_broadcaster` not running | Auto-launched by `perception_sam.launch` |
| Goal appears far from robot | Gazebo position mismatch | Use `x:=12.5 y:=-21 yaw:=3.1416` consistently |
| Robot not moving | MPPI not receiving goal | Check `rostopic echo /move_base_simple/goal` |

## Key Files

```
cs_588_g10/src/gem_perception/
├── scripts/
│   ├── sam_perception_node.py      (perception)
│   └── map_tf_broadcaster.py       (TF frame broadcaster)
├── launch/
│   └── perception_sam.launch
└── rviz/
    └── perception.rviz
```
