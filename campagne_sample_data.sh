#!/usr/bin/env bash
# Un run de campagne sample_data avec surcharges de paramètres.
#   ./campagne_sample_data.sh <label> [arg:=val ...]
#   PORT=11312 ./campagne_sample_data.sh b1 min_pcm:=3     # master dédié -> runs parallèles
# Sans argument de surcharge, c'est le TÉMOIN (défauts identiques aux yaml d'origine).
set -e
if [ -z "$CONTAINER_ID" ]; then
    DISTROBOX="${DISTROBOX:-$(command -v distrobox 2>/dev/null || echo ~/.opt/bin/distrobox)}"
    exec "$DISTROBOX" enter ros1 -- bash "$0" "$@"
fi
source /opt/ros/noetic/setup.bash
[ -f "$HOME/ros1_ws/devel/setup.bash" ] && source "$HOME/ros1_ws/devel/setup.bash"
export ROS_HOSTNAME=localhost
PORT="${PORT:-11311}"
export ROS_MASTER_URI="http://localhost:$PORT"
HERE="$(cd "$(dirname "$0")" && pwd)"
LABEL="$1"; shift
RUN_DIR="$HERE/results/run_sample_data_$(date +%Y-%m-%d_%H%M%S)_$LABEL"
mkdir -p "$RUN_DIR"
export SLAM_RESULTS_DIR="$RUN_DIR"
echo "$*" > "$RUN_DIR/config.txt"
roslaunch -p "$PORT" bruce_slam sample_data_tune.launch "$@" > "$RUN_DIR/run.log" 2>&1
echo "$RUN_DIR"
