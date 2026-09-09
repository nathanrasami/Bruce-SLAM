#!/usr/bin/env python3
"""Bilan compact d'un run : UNE image (bilan_run.png) + chiffres console.

Panneaux : (1) trajectoire alignée Umeyama vs GT (ATE), (2) pointcloud,
(3) erreur de cap véridique dans le temps (fit circulaire s·θ+β, wrap, offset retiré).

Usage : python3 bilan_run.py results/run_aracati_XXXX [results/run_ref_pour_theta]
(2e arg optionnel : run dont groundtruth.csv contient theta, si absent du 1er.)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_eval import umeyama, appliquer, calculer_ate, associer_par_temps


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main(run_dir, theta_ref=None):
    ld = lambda f: np.genfromtxt(os.path.join(run_dir, f), delimiter=",", names=True)
    traj, gt = ld("trajectory.csv"), ld("groundtruth.csv")
    cloud = ld("pointcloud.csv")

    # --- ATE Umeyama
    est = np.column_stack([traj["x"], traj["y"]])
    gxy = associer_par_temps(traj["time"], gt["time"], gt["x"], gt["y"])
    s, R, t = umeyama(est, gxy, with_scale=False, allow_reflection=True)
    est_a = appliquer(s, R, t, est)
    ate = calculer_ate(est_a, gxy)

    # --- cap véridique : gθ (compas NED de /pose_gt) vs θ_est, fit s·θ+β
    src = gt if "theta" in gt.dtype.names else np.genfromtxt(
        os.path.join(theta_ref, "groundtruth.csv"), delimiter=",", names=True)
    g_th = np.interp(traj["time"], src["time"], np.unwrap(src["theta"]))
    best = None
    for sc in (1.0, -1.0):
        d = wrap(g_th - sc * traj["theta"])
        beta = np.arctan2(np.mean(np.sin(d)), np.mean(np.cos(d)))
        rr = wrap(d - beta)
        if best is None or np.median(np.abs(rr)) < np.median(np.abs(best[0])):
            best = (rr, sc, beta)
    resid, s_cap, beta = best
    cap_med = np.degrees(np.median(np.abs(resid)))
    cap_rms = np.degrees(np.sqrt(np.mean(resid ** 2)))

    # --- netteté cloud : NN médian (8000 pts)
    P = np.column_stack([cloud["x"], cloud["y"]])
    rng = np.random.default_rng(0)
    S = P[rng.choice(len(P), min(8000, len(P)), replace=False)]
    d2, _ = cKDTree(S).query(S, k=2)
    nn = float(np.median(d2[:, 1]))

    print(f"{os.path.basename(run_dir)} : KF={len(traj)} ATE={ate:.2f} m | "
          f"cap méd={cap_med:.1f}° RMS={cap_rms:.1f}° (s={s_cap:+.0f}, β={np.degrees(beta):.0f}°) | "
          f"cloud {len(P)} pts NN={nn:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    ax = axes[0]
    ax.plot(gxy[:, 0], gxy[:, 1], "g-", lw=1, label="GT")
    ax.plot(est_a[:, 0], est_a[:, 1], "b-", lw=1, label="SLAM (aligné)")
    ax.set_title(f"Trajectoire — ATE Umeyama {ate:.2f} m")
    ax.legend(); ax.set_aspect("equal")
    ax = axes[1]
    # 3D-aware : si le nuage a une colonne z avec du relief, on colore par z
    if "z" in cloud.dtype.names and np.nanstd(cloud["z"]) > 0.2:
        sc = ax.scatter(P[:, 0], P[:, 1], s=0.3, c=cloud["z"], cmap="viridis",
                        alpha=0.6, linewidths=0)
        plt.colorbar(sc, ax=ax, label="z (m)", shrink=0.8)
        ax.set_title(f"Pointcloud 3D (z couleur) — {len(P)} pts, NN {nn:.3f} m")
    else:
        ax.scatter(P[:, 0], P[:, 1], s=0.15, c="k", alpha=0.4, linewidths=0)
        ax.set_title(f"Pointcloud — {len(P)} pts, NN {nn:.3f} m")
    ax.set_aspect("equal")
    ax = axes[2]
    tm = (traj["time"] - traj["time"][0]) / 60.0
    ax.plot(tm, np.degrees(resid), "k-", lw=0.7)
    ax.axhline(0, color="g", lw=0.5)
    ax.set_xlabel("temps mission (min)")
    ax.set_ylabel("erreur de cap (°)")
    ax.set_title(f"Cap — méd {cap_med:.1f}°, RMS {cap_rms:.1f}° (offset retiré)")
    fig.suptitle(os.path.basename(run_dir))
    fig.tight_layout()
    out = os.path.join(run_dir, "bilan_run.png")
    fig.savefig(out, dpi=120)
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
