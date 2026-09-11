#!/usr/bin/env bash
# Lance Bruce-SLAM ORIGINAL sur Aracati2017 dans un dossier de résultats horodaté.
#
# Usage :
#   ./run_slam.sh                 # Bruce original sur Aracati (bag full)
#   RATE=0.5 ./run_slam.sh        # rejeu du bag plus lent
#   BAG=/chemin/vers.bag ./run_slam.sh
#
# Les CSV (trajectory/pointcloud/groundtruth) sont écrits dans results/run_aracati_<date>/.
# Analyse :  SLAM_RESULTS_DIR=results/run_aracati_<date> python3 plot_trajectories.py

set -e

# Depuis Fedora : se re-lancer automatiquement dans le conteneur ros1.
if [ -z "$CONTAINER_ID" ]; then
    DISTROBOX="${DISTROBOX:-$(command -v distrobox 2>/dev/null || echo ~/.opt/bin/distrobox)}"
    exec "$DISTROBOX" enter ros1 -- bash "$0" "$@"
fi

source /opt/ros/noetic/setup.bash
[ -f "$HOME/ros1_ws/devel/setup.bash" ] && source "$HOME/ros1_ws/devel/setup.bash"

# ROS sur loopback (couper le Wi-Fi ne tue plus les connexions ROS)
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311

HERE="$(cd "$(dirname "$0")" && pwd)"

# ---- sample_data : bag d'ORIGINE de Bruce-SLAM (IMU + DVL + pression + Oculus).
# Mode offline natif : slam_node lit le bag lui-meme, aucun bridge, code pristine.
#   ./run_slam.sh sample_data            # run standard
#   NAME=test RVIZ=true ./run_slam.sh sample_data
if [ "${1:-}" = "sample_data" ]; then
    BAG="${BAG:-$HERE/sample_data.bag}"
    RUN_DIR="$HERE/results/run_sample_data_$(date +%Y-%m-%d_%H%M%S)${NAME:+_$NAME}"
    mkdir -p "$RUN_DIR"
    export SLAM_RESULTS_DIR="$RUN_DIR"
    echo "[run_slam] sample_data -> $RUN_DIR"
    roslaunch bruce_slam sample_data.launch bag_file:="$BAG" rviz:="${RVIZ:-false}" \
        2>&1 | tee "$RUN_DIR/run.log"
    echo "[run_slam] Termine. Analyse : ./analyse.sh $(basename "$RUN_DIR")"
    exit 0
fi

BAG="${BAG:-$HERE/ARACATI_2017_8bits_full.bag}"
RUN_DIR="$HERE/results/run_aracati_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$RUN_DIR"
export SLAM_RESULTS_DIR="$RUN_DIR"
echo "[run_slam] Résultats dans : $RUN_DIR"

# Modes :
#   A (Bruce pur)  : SSM=true NSSM=true USBL=false ./run_slam.sh
#   B (A + ancre)  : SSM=true NSSM=true USBL=true USBL_GAIN=0 USBL_BACKEND=true ./run_slam.sh
#   (défaut)       : filtre USBL front-end (gain 0.4), SSM/NSSM off → ~3.4 m, Bruce pristine
roslaunch bruce_slam aracati.launch bag_file:="$BAG" rate:="${RATE:-1.0}" \
    usbl:="${USBL:-true}" usbl_gain:="${USBL_GAIN:-0.4}" usbl_backend:="${USBL_BACKEND:-false}" \
    ssm:="${SSM:-false}" nssm:="${NSSM:-false}"

echo "[run_slam] Terminé. Analyse : ./analyse.sh $(basename "$RUN_DIR")"
