#!/usr/bin/env bash
# Launch the sim-stack in a single tmux session, three panes stacked top
# -> bottom:
#   pane 0 (top)    : foxglove_bridge   (roslaunch starts roscore on demand)
#   pane 1 (middle) : publish_fake_gps.py
#   pane 2 (bottom) : adapt_mppi_node_ros1_sim.py
#
# Usage:
#   ./run_sim_stack.sh              # create + attach
#   ./run_sim_stack.sh -k           # kill the session
#   MPPI_TMUX_SESSION=foo ./run_sim_stack.sh   # override session name
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${MPPI_TMUX_SESSION:-mppi_sim}"

if [[ "${1:-}" == "-k" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && \
        echo "[run_sim_stack] killed tmux session '$SESSION'" || \
        echo "[run_sim_stack] no tmux session '$SESSION'"
    exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "[run_sim_stack] tmux not installed (sudo apt install tmux)" >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[run_sim_stack] session '$SESSION' already running — attaching"
    exec tmux attach -t "$SESSION"
fi

SETUP='source /opt/ros/noetic/setup.bash'
WAIT_FOR_MASTER='until rostopic list >/dev/null 2>&1; do sleep 0.5; done'

CMD_BRIDGE="$SETUP && roslaunch foxglove_bridge foxglove_bridge.launch"
CMD_SPOOF="$SETUP && $WAIT_FOR_MASTER && cd '$SCRIPT_DIR' && python3 publish_fake_gps.py"
CMD_MPPI="$SETUP  && $WAIT_FOR_MASTER && cd '$SCRIPT_DIR' && python3 adapt_mppi_node_ros1_sim.py"

tmux new-session  -d -s "$SESSION" -n stack -c "$SCRIPT_DIR" "bash -lc \"$CMD_BRIDGE\""
tmux set-option   -t "$SESSION" mouse on
tmux split-window -v -t "$SESSION:stack.0" -c "$SCRIPT_DIR" "bash -lc \"$CMD_SPOOF\""
tmux split-window -v -t "$SESSION:stack.1" -c "$SCRIPT_DIR" "bash -lc \"$CMD_MPPI\""
tmux select-layout -t "$SESSION:stack" even-vertical
tmux select-pane   -t "$SESSION:stack.0"

exec tmux attach -t "$SESSION"
