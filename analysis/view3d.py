#!/usr/bin/env python3
"""Carte 3D VOLUMIQUE INTERACTIVE du run — plotly WebGL dans le navigateur.

Génère carte_3d.html : vrai rendu volumique manipulable (rotation/zoom/pan à la
souris, survol = coordonnées du point), nuage coloré par z + trajectoire SLAM 3D
+ départ/arrivée, échelle ISOTROPE (aspectmode='data'). Le fichier HTML est
autonome (plotly embarqué) : il s'ouvre sur n'importe quelle machine, partageable.

Lancé par ./analyse.sh 3D <run> : écrit le HTML puis l'ouvre dans le navigateur.
Sans navigateur/affichage : le HTML reste le livrable (l'ouvrir plus tard).
Fallback si plotly absent : fenêtre matplotlib 3D (rotation souris) + carte_3d.png.

Usage : python3 analysis/view3d.py <run_dir> [--save-only]
"""
import os
import sys
import webbrowser

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_on_cloud import _load_cloud

MAX_PTS = 300000  # WebGL reste fluide en dessous ; au-delà on sous-échantillonne


def _load(run_dir):
    P, z, src = _load_cloud(run_dir)
    if z is None:
        z = np.zeros(len(P))
    if len(P) > MAX_PTS:
        idx = np.random.default_rng(0).choice(len(P), MAX_PTS, replace=False)
        P, z = P[idx], np.asarray(z)[idx]
    traj = np.genfromtxt(os.path.join(run_dir, "trajectory.csv"),
                         delimiter=",", names=True)
    tz = traj["z"] if "z" in traj.dtype.names else np.zeros(len(traj))
    return P, np.asarray(z), src, traj, np.asarray(tz)


def _plotly(run_dir, save_only):
    import plotly.graph_objects as go
    P, z, src, traj, tz = _load(run_dir)
    run = os.path.basename(run_dir.rstrip("/"))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=z, mode="markers", name=f"nuage ({src})",
        marker=dict(size=1.6, color=z, colorscale="Viridis", opacity=0.75,
                    colorbar=dict(title="z (m)", len=0.6)),
        hovertemplate="x %{x:.1f} · y %{y:.1f} · z %{z:.2f} m<extra></extra>"))
    fig.add_trace(go.Scatter3d(
        x=traj["x"], y=traj["y"], z=tz, mode="lines", name="trajectoire SLAM",
        line=dict(color="red", width=5)))
    fig.add_trace(go.Scatter3d(
        x=[traj["x"][0]], y=[traj["y"][0]], z=[tz[0]], mode="markers",
        name="départ", marker=dict(color="red", size=6, symbol="diamond")))
    fig.add_trace(go.Scatter3d(
        x=[traj["x"][-1]], y=[traj["y"][-1]], z=[tz[-1]], mode="markers",
        name="arrivée", marker=dict(color="darkred", size=6, symbol="square")))
    fig.update_layout(
        title=f"{run} — carte 3D ({len(P)} pts)",
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   aspectmode="data"),   # échelle isotrope, comme axis equal
        legend=dict(x=0.01, y=0.99), margin=dict(l=0, r=0, t=40, b=0))

    out = os.path.join(run_dir, "carte_3d.html")
    fig.write_html(out, include_plotlyjs=True)
    print("->", out)
    if not save_only:
        opened = webbrowser.open("file://" + os.path.abspath(out))
        print("[view3d] ouvert dans le navigateur" if opened
              else "[view3d] ouvre le HTML à la main (pas de navigateur détecté)")


def _matplotlib_fallback(run_dir, save_only):
    interactive = not save_only and os.environ.get("DISPLAY", "") != ""
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    P, z, src, traj, tz = _load(run_dir)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(P[:, 0], P[:, 1], z, s=1.2, c=z, cmap="viridis",
                    alpha=0.55, linewidths=0)
    fig.colorbar(sc, ax=ax, label="z (m)", shrink=0.6)
    ax.plot(traj["x"], traj["y"], tz, color="red", lw=1.8, label="trajectoire SLAM")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"{os.path.basename(run_dir.rstrip('/'))} — carte 3D ({src})")
    ax.legend(loc="upper left")
    try:
        ax.set_box_aspect((np.ptp(P[:, 0]), np.ptp(P[:, 1]), max(np.ptp(z), 1.0)))
    except Exception:
        pass
    out = os.path.join(run_dir, "carte_3d.png")
    fig.savefig(out, dpi=140)
    print("->", out)
    if interactive:
        plt.show()


def main(run_dir, save_only=False):
    try:
        import plotly  # noqa: F401
        _plotly(run_dir, save_only)
    except ImportError:
        print("[view3d] plotly absent (pip install plotly) — fallback matplotlib")
        _matplotlib_fallback(run_dir, save_only)


if __name__ == "__main__":
    main(sys.argv[1], save_only="--save-only" in sys.argv[2:])
