#!/usr/bin/env bash
# Lance une simu SLAM dans un dossier de résultats DÉDIÉ et horodaté.
# Le launch utilisé dépend de la branche git — pas besoin de passer un mode.
#
# Usage :
#   ./run_slam.sh              # aracati (défaut) = cmd_vel + seed /pose_gt t=0 + USBL back-end
#                              #   + Sonar Context, GT-free (ATE ~1.45 m). bag full.
#   ODOM_SOURCE=odom_pose ./run_slam.sh   # intégration cmd_vel façon aracati2017 (ATE ~1.49 m)
#   GT_FREE_SEED=false ./run_slam.sh      # ancien seed /pose_gt t=0 (A/B vs seed USBL GT-free)
#   ODOM_SOURCE=diso DISO_PRIOR=cmd_vel ./run_slam.sh   # variante DISO GT-free
#
# /!\ NE PAS mettre USBL=true : ça active la fusion USBL DANS le front-end (cmd_vel_odom),
#     EN PLUS du back-end (usbl.enable=True dans slam_aracati.yaml). Double ancrage USBL =
#     l'odométrie snappe/saute sur chaque fix bruité (1.4 m, max 73 m) → trajectoire en
#     ZIGZAG, ATE 1.45 -> 4.66 m. Le bon réglage : front-end = dead-reckoning LISSE,
#     back-end = ancrage USBL (facteurs gtsam). cf. run 111133 (zigzag) vs 135228 (propre).
#   ./run_slam.sh holoocean [2D|3D] [Bruce|Bruce_Sonar]   # défauts : 2D Bruce
#     Bruce_Sonar (alias bs, ex-BSU sans le U : pas d'USBL simulé) = loops par apparence SC
#
# Les CSV sont écrits dans results/run_aracati_<date>/. Pour analyser :
#   SLAM_RESULTS_DIR=results/run_aracati_2026-... python3 analyze_drift.py

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
TYPE="${1:-aracati}"
BAG="${BAG:-$HERE/ARACATI_2017_8bits_full.bag}"
RUN_DIR="$HERE/results/run_${TYPE}_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$RUN_DIR"
export SLAM_RESULTS_DIR="$RUN_DIR"
echo "[run_slam] Résultats dans : $RUN_DIR"

case "$TYPE" in
  aracati)   roslaunch bruce_slam aracati.launch bag_file:="$BAG" rate:="${RATE:-1.0}" usbl:="${USBL:-false}" \
                 odom_source:="${ODOM_SOURCE:-cmd_vel}" diso_prior:="${DISO_PRIOR:-cmd_vel}" diso_seed_gt:="${DISO_SEED_GT:-true}" \
                 gt_free_seed:="${GT_FREE_SEED:-true}" heading_from_compass:="${HEADING_COMPASS:-false}" ;;
  holoocean)
    # ./run_slam.sh holoocean [2D|3D] [Bruce]  — mode et méthode optionnels.
    #   mode 2D (défaut) : chaîne image polaire -> CFAR (bag actuel).
    #   mode 3D          : consomme /sonar_points (PointCloud2 3D du simulateur) ;
    #                      le z est porté partout (CSV, RViz, figures auto-3D).
    #   méthode Bruce (défaut) : seule implémentée — l'argument est le point
    #   d'extension pour brancher d'autres méthodes plus tard (cf. holoocean.launch).
    MODE="$(echo "${2:-2D}" | tr 'A-Z' 'a-z')"
    METHOD="$(echo "${3:-Bruce}" | tr 'A-Z' 'a-z')"
    case "$MODE" in 2d|3d) ;; *) echo "mode inconnu: $2 (2D|3D)"; exit 1 ;; esac
    case "$METHOD" in
      bruce) ;;
      bs|bruce_sonar|bsu|bruce_sonar_usbl) METHOD="bruce_sonar" ;;  # loops SC (renommée
        # 07-07 : Bruce_Sonar — pas d'USBL dans les bags holoocean ; alias bsu conservé)
      *) echo "méthode inconnue: $3 (Bruce | Bruce_Sonar)"; exit 1 ;;
    esac
    # Défauts MESURÉS (07-07 bag court, 07-08 bags 3D complets) : SSM=false (ICP
    # séquentiel biaisé : 4.79/1.45 m) ; NSSM=false (4 loops natives tour 2 → 0.84 m
    # vs 0.03 sans). Loops sûres : méthode bs (SC gate, 0 fausse acceptée).
    # bag_source.txt : permet à analyse.sh de générer LA carte_3d (vraie 3D) du run.
    echo "${BAG_HOLO:-$HERE/test.bag}" > "$RUN_DIR/bag_source.txt"
    # SONAR_RANGE : portée du sonar horizontal du bag (traj7 = 20 ; défaut 40).
    roslaunch bruce_slam holoocean.launch bag_file:="${BAG_HOLO:-$HERE/test.bag}" \
                 rate:="${RATE:-1.0}" odom_source:="${ODOM_SOURCE:-dvl}" \
                 ssm:="${SSM:-false}" nssm:="${NSSM:-false}" \
                 sonar_range:="${SONAR_RANGE:-40}" \
                 mode:="$MODE" method:="$METHOD" ;;
  *) echo "Type inconnu: $TYPE (aracati|holoocean)"; exit 1 ;;
esac

echo "[run_slam] Terminé. Analyse : ./analyse.sh $(basename "$RUN_DIR")"
