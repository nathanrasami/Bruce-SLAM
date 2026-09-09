#!/usr/bin/env python3
"""analyze_origine.py — alignement DÉPART COMMUN (0,0), SANS Umeyama.

Contrairement à analyze_drift.py (Umeyama = rotation + translation optimales),
ici l'alignement est une simple TRANSLATION qui épingle le DÉBUT de chaque
trajectoire à (0,0) :

    xy_aligné = xy - xy[0]          (et la GT pareil : gt - gt[0])

Aucune rotation. On mesure donc la DÉRIVE BRUTE depuis un départ connu :
  - pas de "triche" par rotation globale → ATE plus honnête pour du SLAM
    (on sait où on a démarré),
  - mais une trajectoire tournée/miroir (ex. DISO Y-inversé) restera décalée
    → son ATE sera plus élevé qu'avec Umeyama. C'est attendu.

À comparer à analyze_drift.py : les deux ATE ENCADRENT la réalité
(Umeyama = optimiste/meilleur fit, Origine = réaliste/dérive depuis le départ).

Usage :
    SLAM_RESULTS_DIR=results/run_... python3 analyze_origine.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from traj_eval import associer_par_temps, calculer_ate, umeyama, odometrie_pure_depuis_bag

results_dir = os.environ.get("SLAM_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
bag_path = os.environ.get("BAG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ARACATI_2017_8bits_full.bag"))

traj_path = os.path.join(results_dir, "trajectory.csv")
if not os.path.exists(traj_path):
    raise SystemExit(f"trajectory.csv introuvable dans {results_dir} (analyse Bruce-SLAM).")
traj = pd.read_csv(traj_path)
gt_path   = os.path.join(results_dir, "groundtruth.csv")
diso_path = os.path.join(results_dir, "diso_trajectory.csv")
odom_path = os.path.join(results_dir, "odometry.csv")


def au_depart(xy):
    """Translation pure : épingle le 1er point à (0,0). Pas de rotation."""
    return xy - xy[0] if xy is not None and len(xy) else xy


def aligner_origine(src_xy, t_src, gt, label):
    """TRANSLATION PURE (convention FINALE du papier, décision Nathan 2026-07-17) :
    les deux trajectoires commencent à (0,0), AUCUNE rotation ni réflexion ajustée.
    L'orientation du repère estimé est une sortie du système (seed embarqué).
    ⚠ Historique : jusqu'au 17-07 cette fonction ajustait l'orientation initiale
    par Umeyama sur les 15 premiers %% de points (les anciens plots des runs
    affichent donc des ATE légèrement différents, ex. BSU 1.83 vs 1.94).
    Retourne (xy_aligné, ATE)."""
    gt_xy = associer_par_temps(t_src, gt["time"], gt["x"], gt["y"])
    gt_p  = au_depart(gt_xy)
    est_p = au_depart(src_xy)
    return est_p, calculer_ate(est_p, gt_p)


fig, ax = plt.subplots(figsize=(10, 8))

if not os.path.exists(gt_path):
    raise SystemExit("groundtruth.csv introuvable — besoin de la GT pour l'ATF origine.")
gt = pd.read_csv(gt_path)
gxy_p = au_depart(gt[["x", "y"]].to_numpy())

# Bruce-SLAM
est_b_p, ate = aligner_origine(traj[["x", "y"]].to_numpy(), traj["time"], gt, "Bruce")

# Odométrie d'entrée du SLAM (DISO ou GT relayée)
est_o_p = ate_odom = None
if os.path.exists(odom_path):
    odom = pd.read_csv(odom_path)
    est_o_p, ate_odom = aligner_origine(odom[["x", "y"]].to_numpy(), odom["time"], gt, "Odom")

# DISO standalone (si présent)
est_d_p = ate_diso = None
if os.path.exists(diso_path):
    diso = pd.read_csv(diso_path)
    est_d_p, ate_diso = aligner_origine(diso[["x", "y"]].to_numpy(), diso["time"], gt, "DISO")

# Odométrie pure = ré-intégration OFFLINE — seulement si odometry.csv absent
# (sinon doublon divergent de la vraie entrée du SLAM, cf. analyze_drift)
est_p_p = ate_pure = None
odom_csv = os.path.join(results_dir, "odometry.csv")
if not os.path.exists(odom_csv) and os.path.exists(bag_path):
    pure = odometrie_pure_depuis_bag(bag_path)
    if pure is not None:
        est_p_p, ate_pure = aligner_origine(np.column_stack([pure["x"], pure["y"]]),
                                            pure["time"], gt, "Odom pure")

# ── ATE en clair + sauvegarde ─────────────────────────────────────────────────
print(f"=== alignement ORIGINE (cap de départ aligné + départ→0,0) — {results_dir} ===")
print(f"[Origine] Bruce-SLAM, Sonar Context ATE = {ate:.2f} m")
if ate_odom is not None: print(f"[Origine] Odométrie  ATE = {ate_odom:.2f} m")
if ate_diso is not None: print(f"[Origine] DISO stand. ATE = {ate_diso:.2f} m")
if ate_pure is not None: print(f"[Origine] Odom pure  ATE = {ate_pure:.2f} m")

pd.DataFrame({"time": traj["time"].values,
              "x": est_b_p[:, 0], "y": est_b_p[:, 1]}).to_csv(
    os.path.join(results_dir, "trajectory_centre.csv"), index=False)

# ── Tracé (tout part de (0,0)) ────────────────────────────────────────────────
ax.plot(gxy_p[:, 0], gxy_p[:, 1], "r--", label="Ground truth")
ax.plot(gxy_p[0, 0], gxy_p[0, 1], marker="*", color="red", ms=16)
ax.plot(gxy_p[-1, 0], gxy_p[-1, 1], "rX", ms=12)
if est_p_p is not None:
    ax.plot(est_p_p[:, 0], est_p_p[:, 1], color="purple", ls="--", lw=1.2,
            label=f"Odom pure (ATE={ate_pure:.1f} m)")
    ax.plot(est_p_p[0, 0], est_p_p[0, 1], marker="*", color="purple", ms=16)
    ax.plot(est_p_p[-1, 0], est_p_p[-1, 1], marker="X", color="purple", ms=12)
if est_o_p is not None:
    ax.plot(est_o_p[:, 0], est_o_p[:, 1], color="orange", ls="-.", lw=1.5,
            label=f"Odométrie (ATE={ate_odom:.1f} m)")
    ax.plot(est_o_p[0, 0], est_o_p[0, 1], marker="*", color="orange", ms=16)
    ax.plot(est_o_p[-1, 0], est_o_p[-1, 1], marker="X", color="orange", ms=12)
if est_d_p is not None:
    ax.plot(est_d_p[:, 0], est_d_p[:, 1], color="steelblue", ls=":", lw=2.0,
            label=f"DISO standalone (ATE={ate_diso:.1f} m)")
    ax.plot(est_d_p[0, 0], est_d_p[0, 1], marker="*", color="steelblue", ms=16)
    ax.plot(est_d_p[-1, 0], est_d_p[-1, 1], marker="X", color="steelblue", ms=12)
ax.plot(est_b_p[:, 0], est_b_p[:, 1], "k-", lw=1.5, label=f"Bruce-SLAM (ATE={ate:.1f} m)")
ax.plot(est_b_p[0, 0], est_b_p[0, 1], marker="*", color="black", ms=16, label="Start")
ax.plot(est_b_p[-1, 0], est_b_p[-1, 1], "kX", ms=12, label="End")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Alignement ORIGINE : cap de départ aligné sur GT + départ (0,0)\n"
             "transfo fixe estimée sur le début seulement — dérive en aval visible")
ax.axis("equal")
ax.grid(True)
ax.legend()
plt.tight_layout()
out = os.path.join(results_dir, "trajectory_plot_origine.png")
plt.savefig(out, dpi=150)
print(f"\nPlot : {out}")
print("Compare à analyze_drift.py (Umeyama) : les deux ATE encadrent la réalité.")
