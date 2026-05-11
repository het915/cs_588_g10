# CS_588_G10 クイックスタート

すべての修正が完了しました。以下の3つのコマンドで実行できます：

## ターミナル1: Gazebo
```bash
cd ~/host/Downloads/temp/gem_simulation_ws && source devel/setup.bash
roslaunch gem_launch gem_init.launch world_name:="highbay_track.world" x:=12.5 y:=-21 yaw:=3.1416 custom_scene:=true
```

## ターミナル2: 知覚 + TF + RViz
```bash
cd ~/host/Downloads/temp/cs_588_g10 && source devel/setup.bash
roslaunch gem_perception perception_sam.launch default_prompt:="red sign"
```

## ターミナル3: MPPI
```bash
cd ~/host/Downloads/temp/cs_588_g10/src/vehicle_drivers/mppi_controller/mppi_controller
source ~/host/Downloads/temp/cs_588_g10/devel/setup.bash
python3 adapt_mppi_node_ros1_sim.py
```

## 検出対象変更（オプション、ターミナル4）
```bash
rostopic pub -1 /perception/prompt std_msgs/String "car"
```

これで完全に統合されたシステムが動作します！ 🚀
