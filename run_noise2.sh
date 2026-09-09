#!/usr/bin/env bash
# run_noise2.sh — pipeline round 2 « noise » COMPLET, autonome (~4-6 h) :
#   ① gen des 2 bags _noise (NOISE_ROUND2=1, L1 ×2 / L2 ×5 / L3 ×2 — noise_round2.py)
#   ② mesure anti-inondation sur le bag (PIEGES #24 : % pixels ≥ seuil 30 < 1 %)
#   ③ 4 runs SLAM {traj9,traj5}_noise × {Bruce, Bruce_Sonar} via slam_2bags.sh
#   ④ renommage programmatique des dossiers run → *_{B|BS}_{9|5}_2 (labels round 2)
# S'ARRÊTE net si une étape échoue (E-checks compris) — ne JAMAIS affaiblir un gate.
# Usage : ./run_noise2.sh   (log live : tail -f run_noise2.log)
set -u
cd "$(dirname "$0")"; HERE="$(pwd)"

BR="$(git branch --show-current 2>/dev/null || echo '?')"
[ "$BR" = holoocean ] || { echo "⚠ branche='$BR' (attendu holoocean) — abandon."; exit 2; }
pgrep -f 'gen_bag_3d_v(5|10)\.py|roslaunch bruce_slam' >/dev/null \
  && { echo "⚠ gen ou run SLAM déjà actif — abandon."; exit 2; }

echo "=== ① GEN bags _noise (L1 ×2, L2 ×5, L3 ×2) — $(date) ==="
NOISE_ROUND2=1 FORCE_REGEN=1 ./gen_2bags.sh || { echo "✗ gen ÉCHEC — STOP (voir gen_traj*.log)"; exit 1; }

echo "=== ② Gate anti-inondation (PIEGES #24) — $(date) ==="
podman exec ros1 bash -lc 'source /opt/ros/noetic/setup.bash; python3 - <<PYEOF
import rosbag, numpy as np, rospy, sys
ok = True
for name in ["holoocean_3d_traj9_noise.bag", "holoocean_3d_traj5_noise.bag"]:
    p = "'"$HERE"'/BAG_files/" + name
    b = rosbag.Bag(p); t0 = b.get_start_time(); fr = []
    for i,(t,msg,ts) in enumerate(b.read_messages(topics=["/sonar"], start_time=rospy.Time(t0+300))):
        fr.append((np.frombuffer(msg.data, dtype=np.float32) >= 30/255.).mean()*100)
        if i >= 9: break
    b.close()
    f = float(np.mean(fr))
    print(f"{name}: %pix>=30 = {f:.3f}%  (témoin R1 0.10, inondé x5 = 6.6, gate < 1.0)")
    ok &= f < 1.0
sys.exit(0 if ok else 1)
PYEOF' || { echo "✗ INONDATION détectée — STOP, montrer les chiffres à Nathan."; exit 1; }

echo "=== ③ 4 runs SLAM sur bags _noise — $(date) ==="
NOISE_ROUND2=1 ./slam_2bags.sh || { echo "✗ slam_2bags ÉCHEC"; exit 1; }

echo "=== ④ Renommage _2 (depuis scenario_label.txt, jamais à la main) ==="
for d in results/run_holoocean_*; do
  lbl="$d/scenario_label.txt"
  [ -f "$lbl" ] || continue
  grep -q "_noise" "$lbl" || continue
  case "$d" in *_B_*_2|*_BS_*_2) continue;; esac   # déjà suffixé
  sc="$(grep '^scenario' "$lbl")"
  m=B; grep -q "Bruce_Sonar" <<<"$sc" && m=BS
  tj=9; grep -q "traj5" <<<"$sc" && tj=5
  mv "$d" "${d}_${m}_${tj}_2" && echo "  $d → ${d}_${m}_${tj}_2"
done

echo "=== TERMINÉ $(date) — comparer R1 vs R2 : ATE origine SLAM/DR, loops, cloud ==="
command -v notify-send >/dev/null 2>&1 && notify-send "run_noise2" "pipeline round 2 terminé"
printf '\a'
