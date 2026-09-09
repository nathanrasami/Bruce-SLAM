#!/usr/bin/env python3
"""Sweep des stratégies pour corriger le DÉBUT du prior cmd_vel+USBL (avant le 1er
fix USBL ~9 s, cmd_vel dérive depuis le seed → départ faux, déraille DISO).

Leviers testés :
  - usbl_gain : vitesse de recalage du filtre complémentaire
  - snap      : au 1er fix USBL, on RESET la position dessus (efface la dérive du seed)
  - head_src  : cap initial depuis GT-déplacement, ou 1ers fixes USBL (GT-free)

Métrique : erreur vs GT sur fenêtres COURTES (0-30/60/120 s) = qualité du début.
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
usbl = load('usbl_fixes.csv', ['time', 'x', 'y'])
t0 = gt[0, 0]

def gt_at(t):
    return np.array([np.interp(t, gt[:, 0], gt[:, 1]), np.interp(t, gt[:, 0], gt[:, 2])])

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def integrate(usbl_gain=0.1, snap=False, head_src='gt', seed_src='gt',
              th_gain=0.0, th_min_disp=1.0):
    # position initiale
    if seed_src == 'gt':
        x, y = gt[0, 1], gt[0, 2]
    else:  # 1er fix USBL
        x, y = usbl[0, 1], usbl[0, 2]
    # cap initial
    if head_src == 'gt':
        g1 = gt_at(gt[0, 0] + 2.0)
        th = math.atan2(g1[1] - y, g1[0] - x)
    else:  # 2 premiers fixes USBL (GT-free)
        th = math.atan2(usbl[1, 2] - usbl[0, 2], usbl[1, 1] - usbl[0, 1])
    last_t = cmd[0, 0]
    ui = 0
    snapped = False
    last_acc = None   # (t, x, y) dernier fix ACCEPTÉ (gate outlier)
    last_head = None  # (x, y) dernier fix utilisé pour le cap
    traj = []
    for (t, vx, wz) in cmd:
        dt = t - last_t
        last_t = t
        if 0 < dt <= 1.0:
            x += vx * math.cos(th) * dt
            y += vx * math.sin(th) * dt
            th += wz * dt
            while ui < len(usbl) and usbl[ui, 0] <= t:
                ux, uy = usbl[ui, 1], usbl[ui, 2]
                # gate outlier par vitesse vs dernier fix accepté (comme cmd_vel_odom)
                if last_acc is not None:
                    dtu = max(t - last_acc[0], 1e-3)
                    if math.hypot(ux - last_acc[1], uy - last_acc[2]) / dtu > 3.0:
                        ui += 1
                        continue
                last_acc = (t, ux, uy)
                if snap and not snapped:
                    x, y = ux, uy          # efface la dérive du seed
                    snapped = True
                else:
                    x += usbl_gain * (ux - x)
                    y += usbl_gain * (uy - y)
                # correction de CAP depuis la direction du mouvement USBL
                if th_gain > 0 and last_head is not None:
                    ddx, ddy = ux - last_head[0], uy - last_head[1]
                    if math.hypot(ddx, ddy) > th_min_disp:
                        th += th_gain * wrap(math.atan2(ddy, ddx) - th)
                        last_head = (ux, uy)
                elif last_head is None:
                    last_head = (ux, uy)
                ui += 1
        traj.append((t, x, y))
    return np.array(traj)

def err(traj, te):
    m = traj[:, 0] - t0 <= te
    tr = traj[m]
    g = np.array([gt_at(t) for t in tr[:, 0]])
    e = np.sqrt(((tr[:, 1:] - g) ** 2).sum(1))
    return e.mean(), e.max()

print(f"1er fix USBL à t0+{usbl[0,0]-t0:.1f}s")
print(f"{'stratégie':38} | {'0-30s':>13} | {'0-60s':>13} | {'0-120s':>13}")
print(f"{'':38} | {'RMS   max':>13} | {'RMS   max':>13} | {'RMS   max':>13}")
configs = [
    ('baseline gain=0.1', dict(usbl_gain=0.1)),
    ('snap + gain=0.3 (GT-free seed/head)', dict(usbl_gain=0.3, snap=True, head_src='usbl', seed_src='usbl')),
    ('  + cap-USBL th_gain=0.2', dict(usbl_gain=0.3, snap=True, head_src='usbl', seed_src='usbl', th_gain=0.2)),
    ('  + cap-USBL th_gain=0.4', dict(usbl_gain=0.3, snap=True, head_src='usbl', seed_src='usbl', th_gain=0.4)),
    ('  + cap-USBL th_gain=0.6', dict(usbl_gain=0.3, snap=True, head_src='usbl', seed_src='usbl', th_gain=0.6)),
    ('  + cap-USBL th_gain=0.4 disp=2', dict(usbl_gain=0.3, snap=True, head_src='usbl', seed_src='usbl', th_gain=0.4, th_min_disp=2.0)),
    ('  + cap-USBL th_gain=0.4 gain=0.5', dict(usbl_gain=0.5, snap=True, head_src='usbl', seed_src='usbl', th_gain=0.4)),
]
print()
for name, kw in configs:
    tr = integrate(**kw)
    a, b, c = err(tr, 30), err(tr, 60), err(tr, 120)
    d = err(tr, 2640)
    print(f"{name:38} | {a[0]:5.2f} {a[1]:5.2f} | {b[0]:5.2f} {b[1]:5.2f} | {c[0]:5.2f} {c[1]:5.2f} | tot {d[0]:5.2f}/{d[1]:5.2f}")
