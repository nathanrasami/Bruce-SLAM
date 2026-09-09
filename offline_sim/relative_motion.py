#!/usr/bin/env python3
"""DISO utilise son OdomTopic comme PRIOR DE MOUVEMENT RELATIF (transfo init entre
scans), pas comme position absolue. Donc ce qui compte = précision du DELTA, et
SMOOTHNESS (pas de sauts). Un snap/correction USBL améliore l'absolu mais injecte
des discontinuités → casse le recalage DISO.

Ici on compare le MOUVEMENT de cmd_vel vs GT :
  - échelle de vitesse : |v_cmd| vs |v_gt|  (erreur d'échelle = odométrie fausse ∝)
  - taux de rotation : wz_cmd vs dθ_gt/dt
  - erreur de delta de pose sur fenêtres courtes (ce que DISO voit entre 2 scans)
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

# vitesse GT par différence finie + lissage léger
gt_t = gt[:, 0]
gt_v = np.zeros(len(gt))      # |vitesse| GT
gt_course = np.zeros(len(gt)) # cap du mouvement GT (atan2 du déplacement)
for i in range(1, len(gt)):
    dt = gt_t[i] - gt_t[i-1]
    if dt <= 0: continue
    dx, dy = gt[i,1]-gt[i-1,1], gt[i,2]-gt[i-1,2]
    gt_v[i] = math.hypot(dx, dy) / dt
    gt_course[i] = math.atan2(dy, dx)

# associe chaque cmd à la vitesse GT au même instant
cmd_t = cmd[:, 0]
gtv_at = np.interp(cmd_t, gt_t, gt_v)
vx = cmd[:, 1]

# fenêtre temporelle : ne garder que là où GT bouge (>0.05 m/s) pour le ratio
mov = gtv_at > 0.05
ratio = vx[mov] / gtv_at[mov]
print("=== ÉCHELLE DE VITESSE  (cmd_vel.vx / |v_gt|) ===")
print(f"  médiane={np.median(ratio):.3f}  moyenne={ratio.mean():.3f}  (1.0 = parfait)")
print(f"  p10={np.percentile(ratio,10):.3f}  p90={np.percentile(ratio,90):.3f}")
print(f"  → si ≠1 : cmd_vel.vx n'est PAS la vraie vitesse → odométrie biaisée ∝")

# vitesses moyennes brutes
print(f"\n  |v_gt| moyen (en mouvement) = {gtv_at[mov].mean():.3f} m/s")
print(f"  cmd_vel.vx moyen (idem)     = {vx[mov].mean():.3f} m/s")

# énergie de rotation
print(f"\n=== ROTATION  wz ===")
print(f"  wz moyen abs = {np.abs(cmd[:,2]).mean():.4f} rad/s   max={np.abs(cmd[:,2]).max():.3f}")

# distance totale parcourue : intègre |v| pour cmd et GT
def path_len(t, v):
    return np.sum(v[1:] * np.diff(t))
gt_len = 0.0
for i in range(1,len(gt)):
    gt_len += math.hypot(gt[i,1]-gt[i-1,1], gt[i,2]-gt[i-1,2])
cmd_len = np.sum(np.abs(vx[1:]) * np.diff(cmd_t))
print(f"\n=== LONGUEUR DE TRAJET ===")
print(f"  GT      = {gt_len:.1f} m")
print(f"  cmd_vel = {cmd_len:.1f} m   ratio={cmd_len/gt_len:.3f}")
