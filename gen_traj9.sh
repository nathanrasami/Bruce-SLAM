#!/usr/bin/env bash
# Génération traj9 AUTONOME (quais 1 tour sous le navire, gen_bag_3d_v10.py) : lance,
# surveille, relance si crash, vérifie, notifie. ⚠ v9 REFUSE de tourner sans
# probe_traj9_ship3.json (probe E0) au verdict PASS.
# Usage : ./gen_traj9.sh            -> bag complet (~35-60 min, ~11 Go)
#         ./gen_traj9.sh --test 150 -> bag court des checks (~10 min, ~1.3 Go)
#
# - Progression en direct dans le terminal (+ gen_traj9.log).
# - Crash moteur (Xid 13 : bag figé, python suspendu) : détecté en ~3 min,
#   python tué, relance automatique (3 essais max). SIGBUS post-boot couvert.
# - ⚠ octree_min 0.05 (1er bag à cette finesse) : surveiller la taille du
#   cache octree et la vitesse de génération — si explosion (>1 h, disque),
#   tuer et retomber à 0.07 (gen_bag_3d_v10.py OCTREE_MIN_V9).
# - Fin : check_traj4.py --rmax-h 20 --zone traj9 (E1–E9) + notification.
#   E10 (richesse, conteneur) est à lancer ENSUITE à la main :
#   podman exec ros1 bash -lc 'source /opt/ros/noetic/setup.bash; \
#     source ~/ros1_ws/devel/setup.bash; \
#     python3 analysis/e10_richesse.py BAG_files/holoocean_3d_traj9_test.bag \
#       --ref BAG_files/holoocean_3d_traj7r.bag --ref-every 8'
set -u
cd "$(dirname "$0")"
PY="$(cd .. && pwd)/holoocean-venv/bin/python"
LOG="gen_traj9.log"
SFX=""; [ "${NOISE_ROUND2:-0}" = 1 ] && SFX="_noise"   # round 2 « noise »
BAG="BAG_files/holoocean_3d_traj9${SFX}.bag"
case "$*" in *--test*) BAG="BAG_files/holoocean_3d_traj9${SFX}_test.bag";; esac

[ -x "$PY" ] || { echo "venv introuvable : $PY"; exit 2; }
[ -f probe_traj9_ship3.json ] || { echo "⚠ E0 manquant : lancer probe_traj9_path.py d'abord."; exit 2; }
# garde-fou : un VRAI générateur actif ? (comm=python*, pas les wrappers bash)
for p in $(pgrep -f "gen_bag_3d_v10.py"); do
    case "$(ps -o comm= -p "$p" 2>/dev/null)" in python*)
        echo "⚠ une génération tourne déjà (pid $p) — abandon."; exit 2;;
    esac
done

notifier() {  # $1 = titre, $2 = corps
    command -v notify-send >/dev/null && notify-send "$1" "$2"
    printf '\a'
}

for essai in 1 2 3; do
    echo "=== essai $essai/3 — $(date +%H:%M:%S) — log : $LOG ==="
    "$PY" -u gen_bag_3d_v10.py "$@" >"$LOG" 2>&1 &
    GENPID=$!
    tail --pid="$GENPID" -n +1 -f "$LOG" &   # affichage en direct, meurt avec le python
    TAILPID=$!

    prev=-1; fige=0
    while kill -0 "$GENPID" 2>/dev/null; do
        sleep 30
        cur=$(stat -c %s "$BAG" 2>/dev/null || echo 0)
        if [ "$cur" = "$prev" ]; then fige=$((fige+1)); else fige=0; fi
        prev=$cur
        # ⚠ octree_min 0.05 : 1er run = RECONSTRUCTION de l'octree, bag ~vide
        # longtemps -> grace 60 min tant que le bag est embryonnaire ; 3 min
        # (comportement traj7r) des que l'ecriture a demarre.
        if [ "$cur" -lt 1000000 ]; then LIM=120; else LIM=6; fi
        if [ "$fige" -ge "$LIM" ]; then     # bag fige LIM*30 s = moteur mort
            echo "✗ CRASH probable (bag figé $((LIM*30/60)) min) — kill du python et relance."
            kill -9 "$GENPID" 2>/dev/null
            break
        fi
    done
    wait "$GENPID" 2>/dev/null; kill "$TAILPID" 2>/dev/null

    if grep -q "bag ecrit" "$LOG"; then
        echo "=== génération OK, checks E1–E9 (--rmax-h 20 --zone traj9)… ==="
        "$PY" check_traj4.py "$BAG" --rmax-h 20 --zone traj9; rc=$?
        if [ $rc -eq 0 ]; then
            notifier "traj9 : TOUT PASS ✅" "$(tail -1 "$LOG") — E1–E9 PASS ; lancer E10 (conteneur)"
        else
            notifier "traj9 : bag écrit mais E-checks ÉCHEC ⚠" "voir la sortie de check_traj4.py"
        fi
        exit $rc
    fi
    notifier "traj9 : crash moteur (essai $essai/3)" "relance automatique…"
done
notifier "traj9 : ABANDON ✗" "3 crashs consécutifs — voir $LOG"
echo "✗ 3 échecs consécutifs — voir $LOG (journalctl : NVRM Xid ?)"
exit 1
