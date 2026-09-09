#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_trajectories.py — comparaison GT / Odométrie / SLAM (façon Fig.8 du papier).

Adapté du script de la doctorante (qui lisait un pose.csv fusionné) au format de
sortie de ce pipeline : CSV séparés dans un dossier de run.

  groundtruth.csv   (time,x,y)              → Reference
  odometry.csv      (time,x,y)              → Odometry (entrée du SLAM)
  trajectory.csv    (time,x,y,...)          → SLAM (Proposed)
  diso_trajectory.csv (time,x,y) [optionnel]→ DISO standalone

Chaque trajectoire est alignée sur la GT par Umeyama (réflexion autorisée, sans
échelle) — c'est ce qui rend l'erreur-vs-temps comparable malgré les repères
différents (flip DISO, etc.). Deux sorties :

  trajectory_comparison.png   — trajectoires superposées (+ Start/End)
  error_over_time.png         — erreur euclidienne par instant (+ avg/max, gain %)

Usage :
    SLAM_RESULTS_DIR=results/run_... python3 plot_trajectories.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traj_eval import associer_par_temps, umeyama, appliquer, calculer_ate

results_dir = os.environ.get("SLAM_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))


def charger(nom):
    """Charge un CSV (time,x,y) du run. Retourne (t, xy) ou (None, None)."""
    path = os.path.join(results_dir, nom)
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    return df["time"].to_numpy(), df[["x", "y"]].to_numpy()


def aligner(t_src, src_xy, gt_t, gt_x, gt_y):
    """Aligne src sur la GT par Umeyama. Retourne (xy_aligné, gt_associé, ATE)."""
    gt_xy = associer_par_temps(t_src, gt_t, gt_x, gt_y)
    s, R, t = umeyama(src_xy, gt_xy, with_scale=False, allow_reflection=True)
    est = appliquer(s, R, t, src_xy)
    return est, gt_xy, calculer_ate(est, gt_xy)


def erreurs(est, gt_xy):
    """Erreur euclidienne point à point."""
    return np.linalg.norm(est - gt_xy, axis=1)


# ── Chargement ────────────────────────────────────────────────────────────────
gt_t, gt_xy = charger("groundtruth.csv")
if gt_t is None:
    raise SystemExit(f"groundtruth.csv introuvable dans {results_dir}")
gt_x, gt_y = gt_xy[:, 0], gt_xy[:, 1]

series = []  # (label, couleur, t, xy_aligné, gt_associé, ATE)
for nom, label, couleur in [
    ("odometry.csv",       "Odometry",        "red"),
    ("diso_trajectory.csv","DISO standalone", "purple"),
    ("trajectory.csv",     "SLAM (Proposed)", "blue"),
]:
    t, xy = charger(nom)
    if t is None:
        continue
    est, gt_assoc, ate = aligner(t, xy, gt_t, gt_x, gt_y)
    series.append((label, couleur, t, est, gt_assoc, ate))
    print(f"[{label}] ATE = {ate:.2f} m")

# ── Plot 1 : trajectoires ──────────────────────────────────────────────────────
plt.figure(figsize=(12, 8))
plt.plot(gt_x, gt_y, "b-", lw=2, alpha=0.8, label="Ground Truth")
plt.plot(gt_x[0], gt_y[0], "b*", ms=16)
plt.plot(gt_x[-1], gt_y[-1], "bX", ms=12)
for label, couleur, t, est, _gt, ate in series:
    plt.plot(est[:, 0], est[:, 1], color=couleur, lw=2, alpha=0.8,
             label=f"{label} (ATE={ate:.1f} m)")
    plt.plot(est[0, 0], est[0, 1], marker="*", color=couleur, ms=16)
    plt.plot(est[-1, 0], est[-1, 1], marker="X", color=couleur, ms=12)
plt.xlabel("X (m)", fontsize=12)
plt.ylabel("Y (m)", fontsize=12)
plt.title("Trajectoires alignées (Umeyama) : Reference / Odometry / Proposed", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.tight_layout()
out1 = os.path.join(results_dir, "trajectory_comparison.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nTrajectoires : {out1}")

# ── Plot 2 : erreur vs temps ───────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
stats = []
err_ref = None  # erreur de l'odométrie, pour le gain %
for label, couleur, t, est, gt_assoc, _ate in series:
    err = erreurs(est, gt_assoc)
    rel_t = t - t[0]
    plt.plot(rel_t, err, color=couleur, lw=2, label=f"{label} error")
    ligne = f"{label}: avg={err.mean():.2f} m, max={err.max():.2f} m"
    if label == "Odometry":
        err_ref = err
    elif err_ref is not None and label == "SLAM (Proposed)":
        gain = (1.0 - err.mean() / err_ref.mean()) * 100.0
        ligne += f"  → gain vs Odometry : {gain:+.1f} %"
    stats.append(ligne)

plt.xlabel("Temps (s)", fontsize=12)
plt.ylabel("Erreur (m)", fontsize=12)
plt.title("Erreur de position au cours du temps", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.text(0.02, 0.98, "\n".join(stats), transform=plt.gca().transAxes,
         fontsize=11, va="top",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.75))
plt.tight_layout()
out2 = os.path.join(results_dir, "error_over_time.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Erreur/temps : {out2}")
