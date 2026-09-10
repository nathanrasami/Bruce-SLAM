#!/usr/bin/env bash
# Lance une simu SLAM dans un dossier de résultats DÉDIÉ, horodaté et labellisé
# PROGRAMMATIQUEMENT (jamais de suffixe manuel — piège connu, cf. mémoire).
#
# REFONTE (REFONTE_MISSION.md) — 4 méthodes = 4 presets, 2 interrupteurs (SC × USBL),
# AUCUN autre réglage de méthode possible ici :
#   ./run_slam.sh bruce         # SC off, USBL off          (NSSM natif)
#   ./run_slam.sh bruce_u       # SC off, USBL back-end on  (σ 2.5)
#   ./run_slam.sh bruce_sonar   # SC on,  USBL off
#   ./run_slam.sh bsu           # SC on,  USBL back-end on  (σ 1.4)
#   ./run_slam.sh holoocean     # simulation (branche holoocean)
#
# L'odométrie (cmd_vel_odom) est PURE et identique dans les 4 presets (⛔2) ;
# gate post-runs : analysis/gates_refonte.py (dr identiques ×4, ATE origine, ordre).
# Env optionnel : BAG=<bag> RATE=<r> RVIZ=false ./run_slam.sh <preset>
#
# /!\ UN SEUL run à la fois ; ne RIEN committer/modifier dans le dépôt pendant un run.
# Les CSV sont écrits dans results/run_<preset>_<date>/. Analyse : ./analyse.sh <run>.

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
TYPE="${1:?Usage: ./run_slam.sh bruce|bruce_u|bruce_sonar|bsu|holoocean}"
BAG="${BAG:-$HERE/ARACATI_2017_8bits_full.bag}"

case "$TYPE" in
  bruce)       SC=false; USBL_BACKEND=false ;;
  bruce_u)     SC=false; USBL_BACKEND=true  ;;
  bruce_sonar) SC=true;  USBL_BACKEND=false ;;
  bsu)         SC=true;  USBL_BACKEND=true  ;;
  holoocean)   ;;
  *) echo "Preset inconnu: $TYPE (bruce|bruce_u|bruce_sonar|bsu|holoocean)"; exit 1 ;;
esac

RUN_DIR="$HERE/results/run_${TYPE}_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$RUN_DIR"
export SLAM_RESULTS_DIR="$RUN_DIR"
echo "[run_slam] Preset $TYPE → résultats dans : $RUN_DIR"

if [ "$TYPE" = "holoocean" ]; then
    roslaunch bruce_slam holoocean.launch
else
    roslaunch bruce_slam aracati.launch sc:="$SC" usbl_backend:="$USBL_BACKEND" \
        bag_file:="$BAG" rate:="${RATE:-1.0}" rviz:="${RVIZ:-true}"
fi

echo "[run_slam] Terminé. Analyse : ./analyse.sh $(basename "$RUN_DIR")"
