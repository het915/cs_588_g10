#!/bin/bash
SESSION=gem
SETUP="source install/setup.bash"

tmux new-session -d -s $SESSION -n sensor "bash -c '$SETUP && ros2 launch basic_launch sensor_init.launch.py; exec bash'"
tmux new-window -t $SESSION -n cameras "bash -c '$SETUP && ros2 launch basic_launch corner_cameras.launch.py; exec bash'"
tmux new-window -t $SESSION -n gnss "bash -c '$SETUP && ros2 launch basic_launch visualization.launch.py; exec bash'"
tmux new-window -t $SESSION -n joystick "bash -c '$SETUP && ros2 launch basic_launch dbw_joystick.launch.py; exec bash'"
tmux new-window -t $SESSION -n pacmod "bash -c '$SETUP && ros2 launch pacmod2 pacmod2.launch.xml; exec bash'"
tmux new-window -t $SESSION -n pure_pursuit "bash -c '$SETUP && ros2 run gem_gnss_control pure_pursuit; exec bash'"

tmux attach -t $SESSION
