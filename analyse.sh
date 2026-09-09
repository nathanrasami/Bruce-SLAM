#!/bin/bash
# Analyse d'un run : ./analyse.sh [3D] <nom_du_run> [bag]
#   ex : ./analyse.sh run_aracati_2026-07-03_003823
#        ./analyse.sh run_holoocean_2026-07-05_141231
#        ./analyse.sh 3D run_holoocean_2026-07-08_143452   # + OUVRE carte_3d.html
#        ./analyse.sh 3D run_holoocean_... bag/holoocean_3d_traj3.bag  # bag explicite
# Détecte le type de run et enchaîne les scripts de analysis/ qui s'appliquent —
# tolérant : un script manquant ou en échec n'arrête pas la suite.
#   aracati   : analyze_drift / analyze_origine / plot_trajectories / filter_cloud /
#               render_compass_cloud (U1) / traj_on_cloud / paper_eval / bilan_run
#   holoocean : holoocean_report (refactor 07-05 : mêmes NOMS de fichiers, contenus
#               adaptés — DR = IMU+DVL, pas d'odométrie cmd_vel ; erreurs Umeyama ET
#               origine ; carte_finale = traj sur nuage) / paper_eval / bilan_run
#   paper_eval (07-07, unifié ici) : métriques « façon papier » (ATE um/fp, RE, S1/S2/S3,
#               cap, carte) imprimées + figures dans le run — GT continue requise
#               (sauté sur caves).
#   3D        : ouvre en dernier carte_3d.html — LA carte 3D unique du run,
#               VRAIE 3D uniquement (analysis/carte_3d.py ; pseudo-3D exclue).
#               Bag retrouvé auto : bag_source.txt → argument → durée.

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
  run_caves_*)
    # ===== chaîne CAVES : GT éparse (cônes) → pas d'ATE classique en v1 =====
    # les scripts GT-dépendants sont sautés ; la carte et la trajectoire restent.
    echo "[analyse] caves : pas de /pose_gt continu — figures sans ATE (cf. CAVES.md)"
    run_py traj_on_cloud.py "$CHEMIN"
    run_py traj_on_cloud.py "$CHEMIN" "$CHEMIN/carte_finale.png"
    # grotte 3D : profiler VERTICAL SeaKing projeté le long de la traj SLAM
    # (la figure « du site » du dataset) → grotte_3d.html interactif
    if [ -f "$HERE/caves.bag" ]; then
        run_py caves_3d.py "$CHEMIN" --bag "$HERE/caves.bag" --with-map
        # unification 08-07 : LA carte 3D d'un run s'appelle carte_3d.html partout
        # (grotte_3d = vraie 3D par profiler vertical → c'est la carte du run caves)
        [ -f "$CHEMIN/grotte_3d.html" ] && cp -f "$CHEMIN/grotte_3d.html" "$CHEMIN/carte_3d.html"
    fi
    ;;
  run_holoocean_*)
    # ===== chaîne HOLOOCEAN (refactor 07-05) =====
    # un seul script génère : carte_finale, error_over_time (Umeyama + origine),
    # pointcloud_filtered (nuage+traj | traj seule), pointcloud_map,
    # trajectory_plot / _origine / _comparison — étiquettes DR IMU+DVL correctes.
    run_py holoocean_report.py "$CHEMIN"
    run_py paper_eval.py "$CHEMIN"
    # ── LA carte 3D unique (VRAIE 3D seulement, cf. analysis/carte_3d.py) ──
    # Bag du run, dans l'ordre : 1) argument (./analyse.sh [3D] <run> <bag>)
    # 2) bag_source.txt (écrit AUTOMATIQUEMENT par run_slam.sh à chaque run)
    # 3) AUTO-DÉTECTION par durée (trajectory.csv vs rosbag info) — self-heal :
    #    le bag retrouvé est ré-écrit dans bag_source.txt.
    DISTROBOX="$(command -v distrobox 2>/dev/null || echo "$HOME/.opt/bin/distrobox")"
    BAG_RUN="$2"
    [ -z "$BAG_RUN" ] && [ -f "$CHEMIN/bag_source.txt" ] && BAG_RUN="$(cat "$CHEMIN/bag_source.txt")"
    if [ -z "$BAG_RUN" ] || [ ! -f "$BAG_RUN" ]; then
        echo "[analyse] bag du run inconnu (bag_source.txt absent/périmé) → auto-détection par durée…"
        DUREE=$(python3 -c "
import numpy as np
d = np.genfromtxt('$CHEMIN/trajectory.csv', delimiter=',', names=True)
print(f\"{d['time'][-1]-d['time'][0]:.0f}\")" 2>/dev/null)
        CANDIDATS=()
        for b in "$HERE"/bag/*.bag "$HERE"/test*.bag; do
            [ -f "$b" ] || continue
            DB=$("$DISTROBOX" enter ros1 -- bash -lc \
                 "source /opt/ros/noetic/setup.bash; rosbag info -y -k duration '$b'" 2>/dev/null | cut -d. -f1)
            [ -n "$DB" ] && [ -n "$DUREE" ] || continue
            # match : durée du run ≈ durée du bag à ±5 % (run complet)
            if [ "$DB" -gt 0 ] && [ $((DUREE * 100 / DB)) -ge 95 ] && [ $((DUREE * 100 / DB)) -le 105 ]; then
                CANDIDATS+=("$b")
            fi
        done
        if [ ${#CANDIDATS[@]} -eq 1 ]; then
            BAG_RUN="${CANDIDATS[0]}"
            echo "[analyse] bag auto-détecté (durée ${DUREE}s) : $BAG_RUN"
        else
            echo "[analyse] ⚠ ${#CANDIDATS[@]} bag(s) candidats pour ${DUREE}s — précise-le :"
            echo "          ./analyse.sh ${VIEW3D:+3D }$RUN <chemin_du_bag>"
        fi
    fi
    if [ -n "$BAG_RUN" ] && [ -f "$BAG_RUN" ]; then
        echo "$BAG_RUN" > "$CHEMIN/bag_source.txt"   # self-heal pour la prochaine fois
        echo "— carte_3d.py (conteneur)"
        "$DISTROBOX" enter ros1 -- bash -lc \
            "source /opt/ros/noetic/setup.bash; cd '$HERE'; python3 analysis/carte_3d.py '$CHEMIN'" \
            || echo "  (échec carte_3d — on continue)"
    fi
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
    run_py paper_eval.py "$CHEMIN"
    ;;
esac

# bilan compact (1 image : traj+ATE Umeyama, cloud+NN, erreur de cap over time)
run_py bilan_run.py "$CHEMIN"

# ./analyse.sh 3D <run> : ouvre LA carte 3D (carte_3d.html, VRAIE 3D uniquement).
# view3d.py (nuage 2D plaqué au z du robot = pseudo-3D) n'est PLUS appelé (08-07) :
# une carte pseudo-3D est mensongère — s'il n'y a pas de carte_3d.html, on le dit.
if [ -n "$VIEW3D" ]; then
    if [ -f "$CHEMIN/carte_3d.html" ]; then
        xdg-open "$CHEMIN/carte_3d.html" >/dev/null 2>&1 || echo "ouvrir : $CHEMIN/carte_3d.html"
    else
        echo "[analyse] pas de carte_3d.html — deux causes possibles :"
        echo "  1) bag du run introuvable → relance : ./analyse.sh 3D $RUN <chemin_du_bag>"
        echo "  2) aucune source VRAIE 3D dans le bag (pseudo-3D exclue par principe)"
        echo "     → le verdict par topic est affiché plus haut par carte_3d.py"
    fi
fi

echo "[analyse] terminé — fichiers dans $CHEMIN"
