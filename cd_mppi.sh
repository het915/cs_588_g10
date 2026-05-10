#!/usr/bin/env bash
# Source this from any shell (host or container) to cd into the MPPI
# controller package directory where the ROS1 node + helper scripts live.
#
# Usage:
#   source cd_mppi.sh
#   . cd_mppi.sh             # POSIX equivalent
#
# Add this line to ~/.bashrc to get a `cdmppi` shortcut:
#   alias cdmppi='source /workspace/cd_mppi.sh'   # in the docker container
#   alias cdmppi='source /home/aditya/workspaces/cs588/cs_588_g10/cd_mppi.sh'  # on host
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src/vehicle_drivers/mppi_controller/mppi_controller" \
    && echo "[cd_mppi] $(pwd)"
