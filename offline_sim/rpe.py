#!/usr/bin/env python3
"""RPE (Relative Pose Error) du prior cmd_vel vs GT = LA métrique pour un prior
d'odométrie. DISO consomme le mouvement RELATIF entre scans ; si le delta est
faux/bruité, DISO diverge — peu importe la précision absolue.

Compare le déplacement relatif sur fenêtres Δ (translation + direction) de :
  cmd_vel PUR (lisse) vs GT.
"""
import csv, math
import numpy as np

def load(f, cols):
    out = []
    with open(f) as h:
        for r in csv.DictReader(h):
            out.append([float(r[c]) for c in cols])
    return np.array(out)

cmd = load('cmd_vel.csv', ['time', 'vx', 'wz'])
gt = load('gt.csv', ['time', 'x', 'y'])
t0 = gt[0, 0]

def gt_at(t):
    return np.array([np.interp(t, gt[:, 0], gt[:, 1]), np.interp(t, gt[:, 0], gt[:, 2])])

# intègre cmd_vel PUR (lisse, sans USBL), seed GT
x, y = gt[0, 1], gt[0, 2]
g1 = gt_at(gt[0, 0] + 2.0)
th = math.atan2(g1[1] - y, g1[0] - x)
last_t = cmd[0, 0]
traj = []
for (t, vx, wz) in cmd:
    dt = t - last_t; last_t = t
    if 0 < dt <= 1.0:
        x += vx * math.cos(th) * dt
        y += vx * math.sin(th) * dt
        th += wz * dt
    traj.append((t, x, y))
traj = np.array(traj)

def rpe(delta):
    """erreur du déplacement relatif sur fenêtre `delta` s : norme et direction."""
    tt = traj[:, 0]
    errs_norm = []   # |‖Δcmd‖ - ‖Δgt‖|
    errs_dir = []    # angle entre Δcmd et Δgt (deg)
    errs_vec = []    # ‖Δcmd - Δgt‖
    i = 0
    for k in range(len(tt)):
        t_a = tt[k]; t_b = t_a + delta
        if t_b > tt[-1]: break
        j = np.searchsorted(tt, t_b)
        if j >= len(tt): break
        dcmd = traj[j, 1:] - traj[k, 1:]
        dgt = gt_at(t_b) - gt_at(t_a)
        nc, ng = np.linalg.norm(dcmd), np.linalg.norm(dgt)
        errs_norm.append(abs(nc - ng))
        errs_vec.append(np.linalg.norm(dcmd - dgt))
        if nc > 0.05 and ng > 0.05:
            cosang = np.clip(dcmd @ dgt / (nc * ng), -1, 1)
            errs_dir.append(math.degrees(math.acos(cosang)))
    return (np.mean(errs_norm), np.mean(errs_vec),
            np.median(errs_dir) if errs_dir else float('nan'),
            np.percentile(errs_dir, 90) if errs_dir else float('nan'))

print("RPE cmd_vel PUR vs GT (prior lisse, ce que DISO consomme) :")
print(f"{'Δ (s)':>6} | {'err norme':>10} | {'err vecteur':>11} | {'err direction (med/p90)':>24}")
for d in [0.5, 1.0, 2.0, 5.0]:
    en, ev, dm, d9 = rpe(d)
    print(f"{d:>6.1f} | {en:9.3f}m | {ev:10.3f}m | {dm:9.1f}° / {d9:.1f}°")
print("\nréf : DISO+GT prior donne 0.9 m → GT a RPE direction ≈ 0°.")
print("Si la direction du mouvement cmd_vel est très bruitée (p90 grand),")
print("le prior fait pointer DISO dans la mauvaise direction entre scans → divergence.")
