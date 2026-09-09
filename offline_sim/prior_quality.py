#!/usr/bin/env python3
"""Compare la qualité des PRIORS d'odométrie candidats pour DISO, sans GT continu :
  1. cmd_vel pur (intègre /cmd_vel, unicycle, seed GT à t=0)
  2. cmd_vel + USBL (filtre complémentaire, ancrage acoustique — GT-free)
  3. GT (référence)

But : DISO fait confiance à son prior. Si cmd_vel+USBL est proche de GT, c'est un
prior viable GT-free. Sinon, DISO ne peut pas être sevré de la GT par ce chemin.

Mesure l'erreur du prior vs GT sur des fenêtres de temps croissantes (premiers
instants → conclusif rapidement).
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

def integrate(usbl_gain=0.0):
    """Intègre cmd_vel (unicycle). usbl_gain>0 : fusion complémentaire USBL (GT-free)."""
    # seed = GT à t=0 (pose initiale connue = GPS surface), cap = atan2 du 1er déplacement
    x, y = gt[0, 1], gt[0, 2]
    g1 = gt_at(gt[0, 0] + 2.0)
    th = math.atan2(g1[1] - y, g1[0] - x)
    last_t = cmd[0, 0]
    ui = 0  # index USBL
    traj = []
    for (t, vx, wz) in cmd:
        dt = t - last_t
        last_t = t
        if dt <= 0 or dt > 1.0:
            traj.append((t, x, y)); continue
        x += vx * math.cos(th) * dt
        y += vx * math.sin(th) * dt
        th += wz * dt
        if usbl_gain > 0:
            # applique les fixes USBL tombés depuis le dernier pas
            while ui < len(usbl) and usbl[ui, 0] <= t:
                ux, uy = usbl[ui, 1], usbl[ui, 2]
                x += usbl_gain * (ux - x)
                y += usbl_gain * (uy - y)
                ui += 1
        traj.append((t, x, y))
    return np.array(traj)

def err_vs_gt(traj, t_end):
    m = traj[:, 0] - t0 <= t_end
    tr = traj[m]
    g = np.array([gt_at(t) for t in tr[:, 0]])
    e = np.sqrt(((tr[:, 1:] - g) ** 2).sum(1))
    return e.mean(), e.max()

cmd_pure = integrate(usbl_gain=0.0)
cmd_usbl = integrate(usbl_gain=0.1)

print("Erreur du PRIOR vs GT (cumulée, RMS / max) sur fenêtres croissantes :")
print(f"{'fenêtre':>10} | {'cmd_vel pur':>18} | {'cmd_vel + USBL':>18}")
print(f"{'(s)':>10} | {'RMS    max':>18} | {'RMS    max':>18}")
for te in [30, 60, 120, 300, 600, 1200, 2640]:
    a = err_vs_gt(cmd_pure, te)
    b = err_vs_gt(cmd_usbl, te)
    print(f"{te:>10} | {a[0]:7.2f} {a[1]:7.2f} | {b[0]:7.2f} {b[1]:7.2f}")
