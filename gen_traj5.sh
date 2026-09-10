#!/usr/bin/env bash
# Génération traj5 AUTONOME : lance, surveille, relance si crash, vérifie, notifie.
# Usage : ./gen_traj5.sh            -> bag complet (~30 min, 10.3 Go)
#         ./gen_traj5.sh --test 150 -> bag court des checks (~8 min, 1.2 Go)
#
# - Progression en direct dans le terminal (+ gen_traj5.log).
# - Crash moteur (Xid 13 : bag figé, python suspendu) : détecté en ~3 min,
#   python tué, relance automatique (3 essais max).
# - Fin : check_traj4.py (E1–E8) + notification bureau + bip.
set -u
cd "$(dirname "$0")"
PY="$(cd .. && pwd)/holoocean-venv/bin/python"
LOG="gen_traj5.log"
SFX=""; [ "${NOISE_ROUND2:-0}" = 1 ] && SFX="_noise"   # round 2 « noise »
BAG="BAG_files/holoocean_3d_traj5${SFX}.bag"
case "$*" in *--test*) BAG="BAG_files/holoocean_3d_traj5${SFX}_test.bag";; esac

[ -x "$PY" ] || { echo "venv introuvable : $PY"; exit 2; }
# garde-fou : un VRAI générateur actif ? (comm=python*, pas les wrappers bash
# dont la ligne de commande contient juste le nom du script)
for p in $(pgrep -f "gen_bag_3d_v5.py"); do
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
    "$PY" -u gen_bag_3d_v5.py "$@" >"$LOG" 2>&1 &
    GENPID=$!
    tail --pid="$GENPID" -n +1 -f "$LOG" &   # affichage en direct, meurt avec le python
    TAILPID=$!

    prev=-1; fige=0
    while kill -0 "$GENPID" 2>/dev/null; do
        sleep 30
        cur=$(stat -c %s "$BAG" 2>/dev/null || echo 0)
        if [ "$cur" = "$prev" ]; then fige=$((fige+1)); else fige=0; fi
        prev=$cur
        if [ "$fige" -ge 6 ]; then          # 3 min sans croissance = moteur mort
            echo "✗ CRASH probable (bag figé 3 min) — kill du python et relance."
            kill -9 "$GENPID" 2>/dev/null
            break
        fi
    done
    wait "$GENPID" 2>/dev/null; kill "$TAILPID" 2>/dev/null

    if grep -q "bag ecrit" "$LOG"; then
        echo "=== génération OK, checks E1–E8… ==="
        "$PY" check_traj4.py "$BAG"; rc=$?
        if [ $rc -eq 0 ]; then
            notifier "traj5 : TOUT PASS ✅" "$(tail -1 "$LOG") — E1–E8 PASS"
        else
            notifier "traj5 : bag écrit mais E-checks ÉCHEC ⚠" "voir la sortie de check_traj4.py"
        fi
        exit $rc
    fi
    notifier "traj5 : crash moteur (essai $essai/3)" "relance automatique…"
done
notifier "traj5 : ABANDON ✗" "3 crashs consécutifs — voir $LOG"
echo "✗ 3 échecs consécutifs — voir $LOG (journalctl : NVRM Xid ?)"
exit 1
