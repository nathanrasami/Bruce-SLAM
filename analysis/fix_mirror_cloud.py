#!/usr/bin/env python3
"""Dé-miroitage du pointcloud d'un run EXISTANT (post-traitement, 100% GT-free).

CAUSE RACINE du « tourbillon » (diagnostic 2026-07-02) : en mode cmd_vel, les
features cartésiennes sont extraites avec y latéral = +droite (« même signe que
DISO », repère réfléchi), alors que l'odométrie cmd_vel/USBL est en repère PROPRE
(θ ≈ cap monde, det(Umeyama)=+1). Chaque scan est donc peint EN MIROIR de son cap
→ en virage tout se smear en arcs. Fix : inverser le y local de chaque scan.

Usage : python3 fix_mirror_cloud.py results/run_aracati_2026-07-01_150034
Écrit  : pointcloud_demiroir.csv + pointcloud_demiroir.png dans le dossier du run.
N'utilise QUE trajectory.csv + pointcloud.csv (aucune GT).

⚠ Valable pour les runs dont le cloud est rendu avec le theta de trajectory.csv
(cloud/use_compass_cap=False). Ne pas appliquer aux runs DISO (repère cohérent)
ni aux runs use_compass_cap=True (ex. 111746 : theta ≠ cap de rendu).
"""
import os
import sys

import numpy as np


def main(run_dir):
    ld = lambda f: np.genfromtxt(os.path.join(run_dir, f), delimiter=",", names=True)
    traj, cloud = ld("trajectory.csv"), ld("pointcloud.csv")

    kf_ids = traj["keyframe_id"].astype(int)
    idx_of = {k: i for i, k in enumerate(kf_ids)}
    cid = cloud["keyframe_id"].astype(int)
    P = np.column_stack([cloud["x"], cloud["y"]])
    inten = cloud["intensity"] if "intensity" in cloud.dtype.names else np.full(len(P), np.nan)
    ok = np.array([c in idx_of for c in cid])
    cid, P, inten = cid[ok], P[ok], inten[ok]
    ki = np.array([idx_of[c] for c in cid])

    px, py, th = traj["x"], traj["y"], traj["theta"]
    ca, sa = np.cos(th[ki]), np.sin(th[ki])
    dx, dy = P[:, 0] - px[ki], P[:, 1] - py[ki]
    # cloud local (inversion de la pose de rendu), puis flip du y latéral
    cx = ca * dx + sa * dy
    cy = -(-sa * dx + ca * dy)          # <-- LE fix : y -> -y
    # re-rendu aux mêmes poses
    X = np.column_stack([ca * cx - sa * cy + px[ki], sa * cx + ca * cy + py[ki]])

    out_csv = os.path.join(run_dir, "pointcloud_demiroir.csv")
    hdr = "keyframe_id,x,y,intensity"
    np.savetxt(out_csv, np.column_stack([cid, X, inten]), delimiter=",",
               header=hdr, comments="", fmt=["%d", "%.6f", "%.6f", "%.1f"])

    # métrique NN médian (8000 pts) avant/après
    from scipy.spatial import cKDTree
    def nn(A, n=8000, seed=0):
        rng = np.random.default_rng(seed)
        S = A[rng.choice(len(A), min(n, len(A)), replace=False)]
        d, _ = cKDTree(S).query(S, k=2)
        return float(np.median(d[:, 1]))
    def cells(A, res=0.5):
        return len(np.unique(np.floor(A / res).astype(np.int64), axis=0))
    print(f"NN médian : avant={nn(P):.3f}  après={nn(X):.3f}")
    print(f"cellules 0.5 m : avant={cells(P)}  après={cells(X)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (A, t) in zip(axes, [(P, "avant (miroir)"), (X, "après dé-miroitage")]):
        ax.scatter(A[:, 0], A[:, 1], s=0.15, c="k", alpha=0.4, linewidths=0)
        ax.set_title(t)
        ax.set_aspect("equal")
    fig.tight_layout()
    out_png = os.path.join(run_dir, "pointcloud_demiroir.png")
    fig.savefig(out_png, dpi=120)
    print(f"-> {out_csv}\n-> {out_png}")


if __name__ == "__main__":
    main(sys.argv[1])
