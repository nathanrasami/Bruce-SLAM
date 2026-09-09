#!/usr/bin/env bash
# slam_2bags.sh — 4 runs SLAM sur les 2 bags DÉJÀ générés (par ./gen_2bags.sh).
#   {traj9, traj5} × {Bruce, Bruce_Sonar}, nssm=true (parité) ssm=false 2D.
#   sonar_range 20 (traj9) / 40 (traj5). Séquentiel (un seul run à la fois).
#   Progression LIVE dans le terminal (tee) + log par run + analyse.sh.
# Usage :  ./slam_2bags.sh                 # round 1 (bags sans noise)
#          NOISE_ROUND2=1 ./slam_2bags.sh  # round 2 (bags *_noise.bag)
set -u
cd "$(dirname "$0")"; HERE="$(pwd)"
# Round 2 « noise » : NOISE_ROUND2=1 -> runs sur les bags _noise (symétrique gen_2bags.sh).
SFX=""; [ "${NOISE_ROUND2:-0}" = 1 ] && SFX="_noise"
STAMP="$(date +%Y-%m-%d_%H%M%S)"; OUT="slam_2bags${SFX}_${STAMP}"; mkdir -p "$OUT"
[ -n "$SFX" ] && echo "### ROUND 2 « noise » : runs sur *${SFX}.bag ###"

# nom | bag | sonar_range | description
SPECS=(
  "traj9${SFX}|BAG_files/holoocean_3d_traj9${SFX}.bag|20|tour complet${SFX:+ (noise)}"
  "traj5${SFX}|BAG_files/holoocean_3d_traj5${SFX}.bag|40|z variable + contourne bateau${SFX:+ (noise)}"
)
METHODS=("Bruce" "Bruce_Sonar")

BR="$(git branch --show-current 2>/dev/null || echo '?')"
[ "$BR" = holoocean ] || { echo "⚠ branche='$BR' (attendu holoocean) — abandon."; exit 2; }
pgrep -f 'roslaunch bruce_slam' >/dev/null && { echo "⚠ un run SLAM tourne déjà — abandon."; exit 2; }

# bags présents ?
for spec in "${SPECS[@]}"; do
  IFS='|' read -r name bag sr desc <<<"$spec"
  [ -s "$bag" ] || { echo "✗ $bag absent — génère d'abord : ./gen_2bags.sh"; exit 2; }
done

n_ok=0; n_ko=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r name bag sr desc <<<"$spec"
  for m in "${METHODS[@]}"; do
    rlog="$OUT/run_${name}_${m}.log"
    echo "──────── RUN $name × $m  (sonar_range=$sr nssm=true ssm=false 2D) ────────"
    BAG_HOLO="$HERE/$bag" SONAR_RANGE="$sr" NSSM=true SSM=false \
      ./run_slam.sh holoocean 2D "$m" 2>&1 | tee "$rlog"; rc=${PIPESTATUS[0]}
    rundir="$(ls -dt results/run_holoocean_* 2>/dev/null | head -1)"
    if [ "$rc" = 0 ] && [ -n "$rundir" ]; then
      { echo "scenario : $name × $m"; echo "desc     : $desc"; echo "bag      : $bag";
        echo "params   : sonar_range=$sr nssm=true ssm=false mode=2D"; } > "$rundir/scenario_label.txt"
      echo "✓ $name × $m → $rundir"
      if ./analyse.sh "$(basename "$rundir")" >>"$rlog" 2>&1; then echo "  analyse.sh OK";
      else echo "  ⚠ analyse.sh erreur (non bloquant, run gardé)"; fi
      n_ok=$((n_ok+1))
    else
      echo "✗ $name × $m : rc=$rc (voir $rlog / $rundir)"; n_ko=$((n_ko+1))
    fi
  done
done

echo "==================== TERMINÉ : $n_ok OK / $n_ko KO ===================="
echo "labels → results/*/scenario_label.txt · logs → $OUT/"
command -v notify-send >/dev/null 2>&1 && notify-send "slam_2bags" "$n_ok OK / $n_ko KO"
printf '\a'
