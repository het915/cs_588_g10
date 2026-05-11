#!/usr/bin/env bash
# Build the minimal Noetic image (Dockerfile.mppi_debug) and drop into a
# container with $(pwd) bind-mounted at /workspace, host networking,
# and --privileged.
#
# Usage:
#   ./run.mppi_debug.sh                       # build + interactive shell
#   ./run.mppi_debug.sh -- python3 -c '...'   # build + run a one-off command
#
# Once inside, to import-test the sim node:
#   cd /workspace/src/vehicle_drivers/mppi_controller
#   python3 -c "import sys; sys.path.insert(0, '.'); \
#       from mppi_controller import adapt_mppi_node_ros1_sim; print('OK')"
set -euo pipefail

IMAGE="mppi-debug:noetic"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[run.mppi_debug] Building $IMAGE from $SCRIPT_DIR/Dockerfile.mppi_debug"
docker build \
    -t "$IMAGE" \
    -f "$SCRIPT_DIR/Dockerfile.mppi_debug" \
    "$SCRIPT_DIR"

# Anything after `--` is forwarded as the in-container command; default = bash.
CMD=()
if [[ "${1:-}" == "--" ]]; then
    shift
    CMD=("$@")
fi

# Expose the host GPU if both a driver and the nvidia container toolkit
# are available (the toolkit registers the runtime by listing 'nvidia' or
# 'nvidia-cdi' in `docker info`). On a CPU-only host or one without
# nvidia-container-toolkit the flag is omitted so the run still works.
GPU_FLAG=()
if command -v nvidia-smi >/dev/null 2>&1 && \
   docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia; then
    GPU_FLAG=(--gpus all)
    echo "[run.mppi_debug] GPU runtime detected — passing --gpus all"
else
    echo "[run.mppi_debug] No nvidia container runtime — running CPU-only"
fi

echo "[run.mppi_debug] Launching container (privileged, --network host, mount $(pwd) -> /workspace)"
exec docker run --rm -it \
    --privileged \
    --network host \
    --ipc host \
    "${GPU_FLAG[@]}" \
    -v "$(pwd):/workspace" \
    -w /workspace \
    -e ROS_HOSTNAME=localhost \
    -e ROS_MASTER_URI=http://localhost:11311 \
    "$IMAGE" \
    "${CMD[@]:-bash}"
