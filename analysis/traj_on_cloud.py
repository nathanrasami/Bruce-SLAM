#!/usr/bin/env python3
"""Trajectoire SLAM plaquée sur NOTRE nuage — 1 image (traj_on_cloud.png).

Aucun alignement ni GT : le nuage et la trajectoire sont déjà dans le même repère
carte (c'est la vue « ce que le robot a vu et où il est passé », prête pour un rapport).
Choix du nuage, du meilleur au moins bon : pointcloud_compass.csv (rendu cap compas U1)
> pointcloud_filtered.csv (intensité ≥255) > pointcloud.csv (filtré I≥255 si la colonne
existe). 3D-aware : si une colonne z existe avec std > 0.2 m, le nuage est coloré par z.

Usage : python3 analysis/traj_on_cloud.py <run_dir> [out.png]
Import : plot_traj_on_cloud(run_dir, out_path=None, title=None) -> chemin du PNG.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_cloud(run_dir):
    """Retourne (xy, z_ou_None, nom_source)."""
    for f in ("pointcloud_compass.csv", "pointcloud_filtered.csv", "pointcloud.csv"):
        path = os.path.join(run_dir, f)
        if not os.path.exists(path):
            continue
        c = np.genfromtxt(path, delimiter=",", names=True)
        if c.size == 0:
            continue
        if f == "pointcloud.csv" and "intensity" in c.dtype.names:
            keep = c[c["intensity"] >= 255]
            # holoocean : le chemin polaire ne remplit pas l'intensité (tout à 0)
            # → le filtre viderait le nuage ; on ne l'applique que s'il reste du monde
            if len(keep) >= 100:
                c = keep
        if c.size == 0 or len(np.atleast_1d(c)) < 10:
            continue
        z = c["z"] if "z" in c.dtype.names else None
        return np.column_stack([c["x"], c["y"]]), z, f
    raise FileNotFoundError("aucun pointcloud*.csv dans " + run_dir)


def plot_traj_on_cloud(run_dir, out_path=None, title=None):
    traj = np.genfromtxt(os.path.join(run_dir, "trajectory.csv"),
                         delimiter=",", names=True)
    P, z, src = _load_cloud(run_dir)

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    en_3d = z is not None and np.nanstd(z) > 0.2
    if en_3d:
        sc = ax.scatter(P[:, 0], P[:, 1], s=0.3, c=z, cmap="viridis",
                        alpha=0.6, linewidths=0)
        plt.colorbar(sc, ax=ax, label="z (m)", shrink=0.8)
    else:
        ax.scatter(P[:, 0], P[:, 1], s=0.15, c="k", alpha=0.35, linewidths=0)
    ax.plot(traj["x"], traj["y"], color="tab:red", lw=1.3, label="trajectoire SLAM")
    ax.plot(traj["x"][0], traj["y"][0], "^", color="tab:red", ms=9, label="départ")
    ax.plot(traj["x"][-1], traj["y"][-1], "s", color="tab:red", ms=7, label="arrivée")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="best", markerscale=1.0)
    base = title or os.path.basename(run_dir.rstrip("/"))
    ax.set_title(f"{base} — trajectoire sur le nuage ({src}"
                 f"{', 3D coloré par z' if en_3d else ''})")
    fig.tight_layout()
    out = out_path or os.path.join(run_dir, "traj_on_cloud.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("->", out)
    return out


if __name__ == "__main__":
    plot_traj_on_cloud(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
