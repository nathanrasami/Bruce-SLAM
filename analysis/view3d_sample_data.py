#!/usr/bin/env python3
"""Rendu 3D HTML des DEUX trajectoires du run sample_data — demande de la tutrice :
« il faut 2 trajectoires : dead reckoning, et SLAM optimisée », telles que les produit
le code ORIGINAL dépendant de ROS.

Étiquettes de la figure en ANGLAIS (destinataire) ; commentaires en français (repo).

Contenu, calqué sur la vue RViz de Bruce :
  - nuage sonar coloré par keyframe (viridis)
  - trajectoire SLAM optimisée + contraintes séquentielles
  - dead reckoning IMU+DVL+pression (pleine fréquence, 5 Hz)
  - loop closures (constraints.csv) en rouge

Repère : Bruce est en NED (x = avant/Nord, y = droite/Est, z = profondeur mesurée par
le bar30, positive vers le bas). On affiche East = y, North = x, Up = -z, donc une
altitude négative sous la surface. Le SLAM est 2D (gtsam.Pose2) : seuls x,y sont
optimisés, z vient toujours du capteur de pression — y compris pour la courbe SLAM.

Usage : python3 analysis/view3d_sample_data.py <run_dir> [--open]
"""
import os
import sys

import numpy as np
import plotly.graph_objects as go

C_SLAM, C_DR, C_LOOP, C_SEQ = "#f5b301", "#2f6fe0", "#e03131", "#2fa84f"


def main(run_dir, ouvrir=False):
    ld = lambda f: np.genfromtxt(os.path.join(run_dir, f), delimiter=",", names=True)
    traj, dr, cloud = ld("trajectory.csv"), ld("dead_reckoning.csv"), ld("pointcloud.csv")
    run = os.path.basename(run_dir.rstrip("/"))

    # NED -> affichage (East, North, Up) ; z du SLAM = profondeur du keyframe (SLAM 2D)
    sx, sy, sz = traj["y"], traj["x"], -traj["dr_z"]
    dx, dy, dz = dr["y"], dr["x"], -dr["z"]
    kid = cloud["keyframe_id"].astype(int)
    cz = -traj["dr_z"][np.clip(kid, 0, len(traj) - 1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=cloud["y"], y=cloud["x"], z=cz, mode="markers", name="SLAM sonar cloud",
        marker=dict(size=1.5, color=kid, colorscale="Viridis", opacity=0.85)))
    fig.add_trace(go.Scatter3d(
        x=dx, y=dy, z=dz, mode="lines", name="Dead reckoning (IMU+DVL+pressure)",
        line=dict(color=C_DR, width=4, dash="dot")))
    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz, mode="lines+markers", name="SLAM optimized trajectory",
        line=dict(color=C_SEQ, width=3),
        marker=dict(size=3.5, color=C_SLAM)))

    # loop closures : un segment par contrainte, regroupées en UNE trace (None = coupure)
    cons_path = os.path.join(run_dir, "constraints.csv")
    n_loops = 0
    if os.path.exists(cons_path):
        cons = np.atleast_1d(ld("constraints.csv"))
        lx, ly, lz = [], [], []
        for s, t in zip(cons["source_id"].astype(int), cons["target_id"].astype(int)):
            lx += [sx[s], sx[t], None]; ly += [sy[s], sy[t], None]
            lz += [sz[s], sz[t], None]
        n_loops = len(cons)
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines",
                                   name=f"Loop closures ({n_loops})",
                                   line=dict(color=C_LOOP, width=2.5)))

    fig.add_trace(go.Scatter3d(x=[sx[0]], y=[sy[0]], z=[sz[0]], mode="markers",
                               name="Start", marker=dict(size=6, color="white",
                                                         line=dict(color="black", width=1))))
    ecart = np.linalg.norm(np.column_stack([traj["x"] - traj["dr_x"],
                                            traj["y"] - traj["dr_y"]]), axis=1)
    fig.update_layout(
        title=(f"Bruce-SLAM on sample_data.bag (original ROS code) — {run}<br>"
               f"<sub>{len(traj)} keyframes · {n_loops} loop closures · "
               f"SLAM vs dead reckoning: median {np.median(ecart):.2f} m, "
               f"max {ecart.max():.2f} m · no ground truth in this bag</sub>"),
        scene=dict(xaxis_title="East — y (m)", yaxis_title="North — x (m)",
                   zaxis_title="Up — -depth (m)", aspectmode="data"),
        template="plotly_dark", legend=dict(itemsizing="constant"))

    out = os.path.join(run_dir, "trajectoires_3d.html")
    fig.write_html(out, include_plotlyjs=True)   # autonome : lisible hors ligne
    print("->", out)
    if ouvrir:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else os.environ.get("SLAM_RESULTS_DIR", "results"),
         "--open" in sys.argv)
