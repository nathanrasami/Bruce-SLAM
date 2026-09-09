#!/usr/bin/env python3
"""filter_cloud.py — isole la STRUCTURE (quai) du cloud sonar.

On filtre directement pointcloud.csv. Depuis l'intégration inline du cap compas
(cloud/use_compass_cap dans slam_aracati.yaml), pointcloud.csv est DÉJÀ dé-swirlé
par la simu — plus besoin de C2 offline (qui double-appliquait la transfo et re-swirlait).
Le filtre intensité + persistance fait ressortir le quai (réflecteur dur, vu de partout).

Garde les points qui satisfont LES DEUX critères :
  1. intensité >= INTENSITY_MIN  (retour fort = structure solide vs fond mou)
  2. voxel vu depuis >= NKF_MIN keyframes distincts  (persistance = vrai objet)

Produit : pointcloud_filtered.csv + pointcloud_filtered.png + carte_finale.png
Usage : SLAM_RESULTS_DIR=results/run_... python3 filter_cloud.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ.get("SLAM_RESULTS_DIR", "results/run_aracati_Bruce_Sonar_USBL_2026-06-24_150959")

raw_path = os.path.join(R, "pointcloud.csv")
if not os.path.exists(raw_path):
    raise SystemExit(f"pointcloud.csv introuvable dans {R}")

raw  = pd.read_csv(raw_path)
traj = pd.read_csv(os.path.join(R, "trajectory.csv"))

# pointcloud.csv = cloud déjà rendu (cap compas inline si activé). On le filtre tel quel.
df = raw[["keyframe_id", "x", "y", "intensity"]].copy()
src = "pointcloud.csv (rendu simu)"

# ── Paramètres ───────────────────────────────────────────────────────────────
INTENSITY_MIN = 240    # seuil intensité sonar (0-255)
NKF_MIN       = 15     # nb keyframes distincts min par voxel
VOXEL_RES     = 1.0    # taille voxel (m)

# ── Filtre persistance : nb keyframes distincts par voxel ────────────────────
df["cx"] = np.floor(df.x / VOXEL_RES).astype(np.int64)
df["cy"] = np.floor(df.y / VOXEL_RES).astype(np.int64)
nkf = df.groupby(["cx", "cy"])["keyframe_id"].nunique().rename("nkf")
df  = df.join(nkf, on=["cx", "cy"])

# ── Critères (AND) ───────────────────────────────────────────────────────────
mask_intensity  = df.intensity >= INTENSITY_MIN
mask_persistent = df.nkf       >= NKF_MIN
keep = mask_intensity & mask_persistent

df_out = df.loc[keep, ["keyframe_id", "x", "y", "intensity"]].reset_index(drop=True)

print(f"Source coords : {src}")
print(f"Points : {len(df)} → {len(df_out)} gardés ({100*len(df_out)/len(df):.0f}%)")
print(f"  intensité >= {INTENSITY_MIN}      : {mask_intensity.sum()} pts")
print(f"  persistance >= {NKF_MIN} kf       : {mask_persistent.sum()} pts")

out_csv = os.path.join(R, "pointcloud_filtered.csv")
df_out.to_csv(out_csv, index=False)

# ── Plot comparatif ──────────────────────────────────────────────────────────
px = traj.x.to_numpy(); py = traj.y.to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
axes[0].scatter(df.x, df.y, s=0.2, c="darkgreen", alpha=0.15)
axes[0].plot(px, py, "k-", lw=0.5, alpha=0.4)
axes[0].set_title(f"Cloud complet ({len(df)} pts)", fontsize=12)

axes[1].scatter(df_out.x, df_out.y, s=0.4, c="darkorange", alpha=0.4)
axes[1].plot(px, py, "k-", lw=0.5, alpha=0.4)
axes[1].set_title(
    f"Cloud filtré ({len(df_out)} pts, {100*len(df_out)/len(df):.0f}%)\n"
    f"intensité≥{INTENSITY_MIN} ET vu≥{NKF_MIN} keyframes", fontsize=12)

for ax in axes:
    ax.axis("equal"); ax.grid(alpha=0.3)
plt.tight_layout()
out_png = os.path.join(R, "pointcloud_filtered.png")
plt.savefig(out_png, dpi=140, bbox_inches="tight")

# ── Carte finale : panneau unique, présentable ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 9))
ax2.scatter(df_out.x, df_out.y, s=0.5, c=df_out.intensity, cmap="inferno",
            alpha=0.6, vmin=INTENSITY_MIN, vmax=255)
ax2.plot(px, py, "c-", lw=0.6, alpha=0.5, label="trajectoire")
ax2.set_title("Carte finale GT-free (cap compas + filtre structure)", fontsize=13)
ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
ax2.axis("equal"); ax2.grid(alpha=0.3); ax2.legend(loc="upper right")
plt.tight_layout()
final_png = os.path.join(R, "carte_finale.png")
plt.savefig(final_png, dpi=150, bbox_inches="tight")

print(f"PNG comparatif : {out_png}")
print(f"CARTE FINALE   : {final_png}")
print(f"CSV : {out_csv}")
