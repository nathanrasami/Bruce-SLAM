import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interactif : pas de fenêtre (Wayland), sauvegarde seule
import matplotlib.pyplot as plt
import os
from traj_eval import (associer_par_temps, umeyama, appliquer, calculer_ate,
                       odometrie_pure_depuis_bag)

results_dir = os.environ.get("SLAM_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
bag_path = os.environ.get("BAG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ARACATI_2017_8bits_full.bag"))

traj_path = os.path.join(results_dir, "trajectory.csv")
if not os.path.exists(traj_path):
    print(f"trajectory.csv introuvable dans {results_dir}")
    print("=> analyze_drift.py est pour Bruce-SLAM (trajectory.csv).")
    print("   Pour un run DISO standalone, utilise: python3 analyze_diso.py")
    exit(1)
traj = pd.read_csv(traj_path)
gt_path   = os.path.join(results_dir, "groundtruth.csv")
diso_path = os.path.join(results_dir, "diso_trajectory.csv")
odom_path = os.path.join(results_dir, "odometry.csv")


def aligner(src_xy, t_src, gt, label):
    """Aligne une trajectoire sur le GT par interpolation temporelle + Umeyama.
    Retourne (xy_aligné, ATE, (s,R,t)). Lève une erreur claire si bases de temps
    incompatibles (CSV de runs différents)."""
    try:
        gt_xy = associer_par_temps(t_src, gt["time"], gt["x"], gt["y"])
    except ValueError as e:
        print(f"ERREUR d'association temporelle ({label}) :", e)
        print("=> les CSV ne viennent pas du même run. Vérifie results/.")
        exit(1)
    s, R, t = umeyama(src_xy, gt_xy)
    est = appliquer(s, R, t, src_xy)
    return est, calculer_ate(est, gt_xy), (s, R, t)


fig, ax = plt.subplots(figsize=(10, 8))
s_b = R_b = t_b = None  # transfo Bruce, réutilisée pour le point cloud

# ── Alignement de chaque trajectoire sur le GT ────────────────────────────────
if os.path.exists(gt_path):
    gt = pd.read_csv(gt_path)
    gxy = gt[["x", "y"]].to_numpy()

    est_b, ate, (s_b, R_b, t_b) = aligner(traj[["x", "y"]].to_numpy(), traj["time"], gt, "Bruce")

    est_d = ate_diso = None
    if os.path.exists(diso_path):
        diso = pd.read_csv(diso_path)
        est_d, ate_diso, _ = aligner(diso[["x", "y"]].to_numpy(), diso["time"], gt, "DISO")

    # Odométrie d'entrée du SLAM (cmd_vel, DISO ou GT relayée selon la branche)
    est_o = ate_odom = None
    if os.path.exists(odom_path):
        odom = pd.read_csv(odom_path)
        est_o, ate_odom, _ = aligner(odom[["x", "y"]].to_numpy(), odom["time"], gt, "Odométrie")

    est_p = ate_pure = None
    # « Odom pure » = RE-intégration OFFLINE de /cmd_vel (approximation d'analyse,
    # jamais vue par le SLAM). Depuis que le nœud exporte odometry.csv (la VRAIE
    # entrée du SLAM), la ré-intégration fait DOUBLON et diverge en cap intégré
    # (~1.4°/min, schéma d'intégration ≠) → on ne la trace QUE si odometry.csv
    # manque (vieux runs). cf. discussion 07-06 « jaune vs violet ».
    if not os.path.exists(odom_path) and os.path.exists(bag_path):
        pure = odometrie_pure_depuis_bag(bag_path)
        if pure is not None:
            est_p, ate_pure, _ = aligner(np.column_stack([pure["x"], pure["y"]]),
                                         pure["time"], gt, "Odom pure")
else:
    print("groundtruth.csv not found — affichage en repère natif (pas d'ATE)")
    gt = None
    gxy = None
    est_b = traj[["x", "y"]].to_numpy(); ate = None
    est_d = ate_diso = est_o = ate_odom = est_p = ate_pure = None

# ── Plot FIDÈLE à Umeyama : on trace les coordonnées alignées TELLES QUELLES ──
# Pas de forçage à (0,0) : le dessin reflète exactement l'alignement (rotation +
# translation) qui a produit l'ATE. Les trajectoires ne partent pas forcément de
# (0,0) — c'est normal et honnête (cf. analyze_origine.py pour la vue départ-commun).
gxy_p, est_b_p, est_d_p, est_o_p, est_p_p = gxy, est_b, est_d, est_o, est_p

# Sauvegarde des coordonnées Bruce alignées Umeyama (inspection / réutilisation)
if est_b is not None and gt is not None:
    pd.DataFrame({"time": traj["time"].values,
                  "x": est_b[:, 0], "y": est_b[:, 1]}).to_csv(
        os.path.join(results_dir, "trajectory_umeyama.csv"), index=False)

# ATE en clair (stdout)
if ate is not None:
    print(f"[Umeyama] Bruce-SLAM ATE = {ate:.2f} m")
    if ate_odom is not None: print(f"[Umeyama] Odométrie  ATE = {ate_odom:.2f} m")
    if ate_diso is not None: print(f"[Umeyama] DISO       ATE = {ate_diso:.2f} m")
    if ate_pure is not None: print(f"[Umeyama] Odom pure  ATE = {ate_pure:.2f} m")

# ── Tracé ─────────────────────────────────────────────────────────────────────
if gxy_p is not None:
    ax.plot(gxy_p[:, 0], gxy_p[:, 1], label="Ground truth (GPS)", color="red", linestyle="--")
    ax.plot(gxy_p[0, 0], gxy_p[0, 1], marker="*", color="red", markersize=16)
    ax.plot(gxy_p[-1, 0], gxy_p[-1, 1], marker="X", color="red", markersize=12)

if est_p_p is not None:
    lbl = f"Odom pure (ATE={ate_pure:.1f} m)" if ate_pure is not None else "Odom pure"
    ax.plot(est_p_p[:, 0], est_p_p[:, 1], label=lbl, color="purple", linestyle="--", linewidth=1.2)
    ax.plot(est_p_p[0, 0], est_p_p[0, 1], marker="*", color="purple", markersize=16)
    ax.plot(est_p_p[-1, 0], est_p_p[-1, 1], marker="X", color="purple", markersize=12)

if est_o_p is not None:
    lbl = f"Odométrie (ATE={ate_odom:.1f} m)" if ate_odom is not None else "Odométrie"
    ax.plot(est_o_p[:, 0], est_o_p[:, 1], label=lbl, color="orange", linestyle="-.", linewidth=1.5)
    ax.plot(est_o_p[0, 0], est_o_p[0, 1], marker="*", color="orange", markersize=16)
    ax.plot(est_o_p[-1, 0], est_o_p[-1, 1], marker="X", color="orange", markersize=12)

if est_d_p is not None:
    lbl = f"DISO standalone (ATE={ate_diso:.1f} m)" if ate_diso is not None else "DISO standalone"
    ax.plot(est_d_p[:, 0], est_d_p[:, 1], label=lbl, color="steelblue", linestyle=":", linewidth=2.0)
    ax.plot(est_d_p[0, 0], est_d_p[0, 1], marker="*", color="steelblue", markersize=16)
    ax.plot(est_d_p[-1, 0], est_d_p[-1, 1], marker="X", color="steelblue", markersize=12)

slam_label = f"Bruce-SLAM (ATE={ate:.1f} m)" if ate is not None else "Bruce-SLAM"
ax.plot(est_b_p[:, 0], est_b_p[:, 1], label=slam_label, color="black", linewidth=1.5)
ax.plot(est_b_p[0, 0], est_b_p[0, 1], marker="*", color="black", markersize=16, label="Start")
ax.plot(est_b_p[-1, 0], est_b_p[-1, 1], marker="X", color="black", markersize=12, label="End")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Alignement UMEYAMA (rotation+translation optimales)")
ax.legend()
ax.axis("equal")
ax.grid(True)
plt.tight_layout()

out = os.path.join(results_dir, "trajectory_plot.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved to {out}")

# --- Point cloud map ---
cloud_path = os.path.join(results_dir, "pointcloud.csv")
if os.path.exists(cloud_path):
    cloud = pd.read_csv(cloud_path)
    cxy = cloud[["x", "y"]].to_numpy()
    if R_b is not None:  # même transfo Umeyama que la trajectoire Bruce (plot fidèle)
        cxy = appliquer(s_b, R_b, t_b, cxy)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.scatter(cxy[:, 0], cxy[:, 1], s=0.5, c="navy", alpha=0.3)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Point cloud map (accumulated sonar points)")
    ax2.axis("equal")
    ax2.grid(True)
    plt.tight_layout()
    out2 = os.path.join(results_dir, "pointcloud_map.png")
    plt.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"Saved to {out2}")
else:
    print("pointcloud.csv not found")
