#!/usr/bin/env python3
"""Rapport d'analyse HOLOOCEAN — remplace la chaîne aracati (analyze_drift/origine/
plot_trajectories/filter_cloud) dont les contenus/étiquettes n'ont pas de sens ici
(pas d'odométrie cmd_vel : l'entrée est le dead-reckoning IMU+DVL ; pas de filtre
d'intensité : le chemin polaire ne la remplit pas).

Mêmes NOMS de fichiers que les autres branches, contenus holoocean (refactor 07-05) :
  carte_finale.png            trajectoire plaquée sur le nuage (z-coloré si relief)
  error_over_time.png         erreur de position, DEUX conventions (Umeyama + origine)
  pointcloud_filtered.png     gauche : nuage + trajectoire ; droite : trajectoire seule
  pointcloud_map.png          le nuage seul (z-coloré si relief)
  trajectory_plot.png         SLAM vs GT vs DR IMU+DVL (alignement Umeyama)
  trajectory_plot_origine.png idem, ancré à l'ORIGINE (dérive cumulée, conv. papiers)
  trajectory_comparison.png   les deux alignements côte à côte

Usage : python3 analysis/holoocean_report.py <run_dir>
GT : groundtruth.csv du run (time,x,y,theta[,z]). DR : colonnes dr_* de trajectory.csv
(= l'ODOMÉTRIE D'ENTRÉE : IMU+DVL en odom_source=dvl — la GT relayée si odom_source=gt).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_eval import umeyama, appliquer, calculer_ate, associer_par_temps
from paper_eval import ate_premiere_pose
from traj_on_cloud import plot_traj_on_cloud, _load_cloud

DR_LABEL = "DR IMU+DVL (odométrie d'entrée)"


def _save(fig, run_dir, name):
    out = os.path.join(run_dir, name)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("->", out)


def main(run_dir):
    ld = lambda f: np.genfromtxt(os.path.join(run_dir, f), delimiter=",", names=True)
    traj, gt = ld("trajectory.csv"), ld("groundtruth.csv")
    est = np.column_stack([traj["x"], traj["y"]])
    dr = np.column_stack([traj["dr_x"], traj["dr_y"]])
    gxy = associer_par_temps(traj["time"], gt["time"], gt["x"], gt["y"])
    t_rel = (traj["time"] - traj["time"][0]) / 60.0

    # alignements : Umeyama (best-fit global) et origine (première pose, conv. papiers)
    _, R, t = umeyama(est, gxy, with_scale=False)
    est_um = appliquer(1.0, R, t, est)
    ate_um = calculer_ate(est_um, gxy)
    _, Ro, to = umeyama(dr, gxy, with_scale=False)
    dr_um = appliquer(1.0, Ro, to, dr)
    ate_dr = calculer_ate(dr_um, gxy)
    ate_fp, est_fp = ate_premiere_pose(est, gxy)
    ate_dr_fp, dr_fp = ate_premiere_pose(dr, gxy)
    run = os.path.basename(run_dir.rstrip("/"))
    print(f"{run} : ATE Umeyama {ate_um:.2f} m (origine {ate_fp:.2f}) | "
          f"DR {ate_dr:.2f} ({ate_dr_fp:.2f})")

    # --- carte_finale.png = trajectoire sur le nuage (contenu traj_on_cloud)
    plot_traj_on_cloud(run_dir, os.path.join(run_dir, "carte_finale.png"),
                       title=f"{run} — carte finale")

    # --- error_over_time.png : les DEUX conventions (comme les autres branches)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(t_rel, np.linalg.norm(dr_fp - gxy, axis=1), color="tab:orange", ls=":",
            lw=0.9, label=f"{DR_LABEL} — origine ({ate_dr_fp:.2f} m)")
    ax.plot(t_rel, np.linalg.norm(dr_um - gxy, axis=1), color="tab:orange", ls="--",
            lw=0.9, label=f"{DR_LABEL} — Umeyama ({ate_dr:.2f} m)")
    ax.plot(t_rel, np.linalg.norm(est_fp - gxy, axis=1), color="tab:blue", ls=":",
            lw=1.0, label=f"SLAM — ancré origine ({ate_fp:.2f} m)")
    ax.plot(t_rel, np.linalg.norm(est_um - gxy, axis=1), color="tab:blue",
            lw=1.2, label=f"SLAM — Umeyama ({ate_um:.2f} m)")
    ax.set_xlabel("temps mission (min)"); ax.set_ylabel("erreur position (m)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{run} — erreur au cours du temps (2 conventions)")
    _save(fig, run_dir, "error_over_time.png")

    # --- nuage (une seule lecture pour les 2 figures suivantes)
    P, z, src = _load_cloud(run_dir)
    en_3d = z is not None and np.nanstd(z) > 0.2

    def _cloud(ax):
        if en_3d:
            sc = ax.scatter(P[:, 0], P[:, 1], s=0.3, c=z, cmap="viridis",
                            alpha=0.6, linewidths=0)
            return sc
        ax.scatter(P[:, 0], P[:, 1], s=0.15, c="k", alpha=0.4, linewidths=0)
        return None

    # --- pointcloud_map.png : le nuage seul
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    sc = _cloud(ax)
    if sc is not None:
        plt.colorbar(sc, ax=ax, label="z (m)", shrink=0.8)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{run} — nuage ({src}{', 3D' if en_3d else ''})")
    _save(fig, run_dir, "pointcloud_map.png")

    # --- pointcloud_filtered.png : gauche nuage+traj, droite traj seule
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))
    _cloud(axes[0])
    axes[0].plot(traj["x"], traj["y"], color="tab:red", lw=1.2)
    axes[0].set_title("nuage + trajectoire")
    axes[1].plot(traj["x"], traj["y"], color="tab:red", lw=1.2)
    axes[1].plot(traj["x"][0], traj["y"][0], "^", color="tab:red", ms=9)
    axes[1].plot(traj["x"][-1], traj["y"][-1], "s", color="tab:red", ms=7)
    axes[1].set_title("trajectoire seule")
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.suptitle(run)
    _save(fig, run_dir, "pointcloud_filtered.png")

    # --- trajectory_plot / _origine / _comparison
    def _traj_fig(ax, e, d, tag, a_e, a_d):
        ax.plot(gxy[:, 0], gxy[:, 1], color="0.25", lw=1.6, label="GT")
        ax.plot(d[:, 0], d[:, 1], color="tab:orange", ls="--", lw=1.0,
                label=f"{DR_LABEL} ({a_d:.2f} m)")
        ax.plot(e[:, 0], e[:, 1], color="tab:blue", lw=1.3, label=f"SLAM ({a_e:.2f} m)")
        ax.plot(gxy[0, 0], gxy[0, 1], "k^", ms=8)
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(fontsize=8); ax.set_title(tag)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    _traj_fig(ax, est_um, dr_um, f"{run} — alignement Umeyama", ate_um, ate_dr)
    _save(fig, run_dir, "trajectory_plot.png")

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    _traj_fig(ax, est_fp, dr_fp, f"{run} — ancré à l'origine (dérive cumulée)",
              ate_fp, ate_dr_fp)
    _save(fig, run_dir, "trajectory_plot_origine.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    _traj_fig(axes[0], est_um, dr_um, "Umeyama (best-fit global)", ate_um, ate_dr)
    _traj_fig(axes[1], est_fp, dr_fp, "origine (conv. papiers)", ate_fp, ate_dr_fp)
    fig.suptitle(run)
    _save(fig, run_dir, "trajectory_comparison.png")


if __name__ == "__main__":
    main(sys.argv[1])
