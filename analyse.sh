#!/bin/bash
# Analyse d'un run : ./analyse.sh [3D] <nom_du_run>
#   ex : ./analyse.sh run_aracati_2026-07-03_003823
#        ./analyse.sh run_holoocean_2026-07-05_141231
#        ./analyse.sh 3D run_holoocean_2026-07-05_141231   # + carte 3D INTERACTIVE
# Détecte le type de run et enchaîne les scripts de analysis/ qui s'appliquent —
# tolérant : un script manquant ou en échec n'arrête pas la suite.
#   aracati   : analyze_drift / analyze_origine / plot_trajectories / filter_cloud /
#               render_compass_cloud (U1) / traj_on_cloud / bilan_run
#   holoocean : holoocean_report (refactor 07-05 : mêmes NOMS de fichiers, contenus
#               adaptés — DR = IMU+DVL, pas d'odométrie cmd_vel ; erreurs Umeyama ET
#               origine ; carte_finale = traj sur nuage) / bilan_run
#   3D        : ouvre en dernier une carte 3D interactive (rotation souris, façon
#               MATLAB) — analysis/view3d.py ; sauve aussi carte_3d.png.

HERE="$(cd "$(dirname "$0")" && pwd)"
VIEW3D=""
case "$(echo "$1" | tr 'a-z' 'A-Z')" in 3D) VIEW3D=1; shift ;; esac
RUN="$1"
[ -n "$RUN" ] || { echo "usage: ./analyse.sh [3D] <nom_du_run>"; exit 1; }
CHEMIN="$HERE/results/$RUN"
[ -d "$CHEMIN" ] || CHEMIN="$HERE/TESTS_image/$RUN"   # runs archivés aussi
[ -d "$CHEMIN" ] || { echo "run introuvable : results/$RUN (ni TESTS_image/)"; exit 1; }
A="$HERE/analysis"
export SLAM_RESULTS_DIR="$CHEMIN"
echo "[analyse] $CHEMIN"

run_py() {  # lance un script s'il existe ; n'arrête jamais la chaîne
    if [ -f "$A/$1" ]; then
        echo "— $1"
        python3 "$A/$1" "${@:2}" || echo "  (échec $1 — on continue)"
    fi
}

case "$RUN" in
  run_holoocean_*)
    # ===== chaîne HOLOOCEAN (refactor 07-05) =====
    # un seul script génère : carte_finale, error_over_time (Umeyama + origine),
    # pointcloud_filtered (nuage+traj | traj seule), pointcloud_map,
    # trajectory_plot / _origine / _comparison — étiquettes DR IMU+DVL correctes.
    run_py holoocean_report.py "$CHEMIN"
    ;;
  *)
    # ===== chaîne ARACATI (historique) =====
    run_py analyze_drift.py        # erreur de trajectoire over time
    run_py analyze_origine.py
    run_py plot_trajectories.py

    # cloud filtré par intensité — seulement si la colonne existe
    if head -1 "$CHEMIN/pointcloud.csv" 2>/dev/null | grep -q intensity; then
        run_py filter_cloud.py
    fi

    # rendu cap compas (U1, 100% GT-free) : position optimisée + cap dr_theta recalé.
    if head -1 "$CHEMIN/trajectory.csv" 2>/dev/null | grep -q dr_theta; then
        if head -1 "$CHEMIN/pointcloud.csv" 2>/dev/null | grep -q intensity; then
            run_py render_compass_cloud.py "$CHEMIN" --imin 255
        else
            run_py render_compass_cloud.py "$CHEMIN"
        fi
    fi

    # trajectoire plaquée sur NOTRE nuage (même repère, aucune GT)
    run_py traj_on_cloud.py "$CHEMIN"
    ;;
esac

# bilan compact (1 image : traj+ATE Umeyama, cloud+NN, erreur de cap over time)
run_py bilan_run.py "$CHEMIN"

# carte 3D interactive (./analyse.sh 3D <run>) — en DERNIER : fenêtre bloquante
[ -n "$VIEW3D" ] && run_py view3d.py "$CHEMIN"

echo "[analyse] terminé — fichiers dans $CHEMIN"
