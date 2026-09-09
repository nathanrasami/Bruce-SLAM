#!/usr/bin/env bash
# gen_2bags.sh — génère UNIQUEMENT les 2 bags des scénarios prof, puis s'arrête.
#   ① traj9 = tour complet            (gen_traj9.sh, z const, sonar 20)
#   ② traj5 = z variable + contourne bateau (gen_traj5.sh, z aléa [-12,-2], sonar 40)
# Le SLAM se lance séparément APRÈS, avec ./slam_2bags.sh.
# Progression LIVE dans le terminal (les wrappers gen_traj*.sh affichent leur propre
# tail -f). Watchdog Xid13 + relance ×3 hérités des wrappers.
#
# Usage :  ./gen_2bags.sh              # réutilise un bag déjà présent
#          FORCE_REGEN=1 ./gen_2bags.sh   # (re)génère les 2 depuis zéro
set -u
cd "$(dirname "$0")"
FORCE_REGEN="${FORCE_REGEN:-0}"
# Round 2 « noise » : NOISE_ROUND2=1 -> bags _noise (round 1 jamais ecrase).
SFX=""; [ "${NOISE_ROUND2:-0}" = 1 ] && SFX="_noise"
[ -n "$SFX" ] && echo "### ROUND 2 « noise » actif : bags *${SFX}.bag (L1x5 L2x5 L3x2) ###"

BR="$(git branch --show-current 2>/dev/null || echo '?')"
[ "$BR" = holoocean ] || { echo "⚠ branche='$BR' (attendu holoocean) — abandon."; exit 2; }
if pgrep -f 'gen_bag_3d_v(5|10)\.py' >/dev/null; then
  echo "⚠ une génération tourne déjà — abandon (arrête-la d'abord)."; exit 2
fi

gen_one() {   # $1=nom  $2=wrapper  $3=bag
  local name="$1" wrap="$2" bag="$3"
  if [ "$FORCE_REGEN" != 1 ] && [ -s "$bag" ]; then
    echo "↷ $name : bag déjà présent ($(du -h "$bag" | cut -f1)) — sauté (FORCE_REGEN=1 pour forcer)."
    return 0
  fi
  echo "==================== GEN $name via ./$wrap (live) ===================="
  ./"$wrap"; local rc=$?
  if [ $rc = 0 ] && [ -s "$bag" ]; then
    echo "✓ $name OK ($(du -h "$bag" | cut -f1))"; return 0
  fi
  echo "✗ $name ÉCHEC (rc=$rc) — voir ${wrap%.sh}.log"; return 1
}

ok=0
gen_one traj9 gen_traj9.sh "BAG_files/holoocean_3d_traj9${SFX}.bag" && ok=$((ok+1))
gen_one traj5 gen_traj5.sh "BAG_files/holoocean_3d_traj5${SFX}.bag" && ok=$((ok+1))

echo "==================== BAGS : $ok/2 prêts ===================="
echo "→ SLAM ensuite : ./slam_2bags.sh"
command -v notify-send >/dev/null 2>&1 && notify-send "gen_2bags" "$ok/2 bags prêts"
printf '\a'
[ "$ok" = 2 ]
