#!/usr/bin/env python3
"""Reconstruction 3D de la grotte — profiler VERTICAL SeaKing + trajectoire SLAM.

C'est la figure « du site » du dataset (Mallios et al.) : le Micron horizontal fait
la CARTE (et la trajectoire SLAM), le SeaKing balaye un plan VERTICAL transverse →
chaque faisceau donne un point de la section du tunnel ; projetés le long de la
trajectoire SLAM (interpolée en temps), les 97 501 faisceaux dessinent la cavité 3D.

100 % offline : lit le bag (rosbags, sans ROS) + trajectory.csv du run.
Détection par faisceau : le retour LE PLUS FORT au-dessus d'un seuil (le profiler
est fait pour ça : 1 faisceau ≈ 1 distance de paroi).

Géométrie (paramétrable, à valider à l'œil — le tunnel doit ENVELOPPER la traj) :
point local = (0, r·cos φ, r·sin φ) dans le repère véhicule (plan transverse y-z),
monde = R_z(yaw)·local + (x, y, z_pose). Options --flip-phi / --phi0 si la section
sort tournée/miroir.

Usage : python3 analysis/caves_3d.py <run_dir> [--bag caves.bag] [--threshold 60]
Sorties : <run>/grotte_3d.csv + <run>/grotte_3d.html (plotly interactif).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--topic", default="/sonar_seaking_ros")
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--rmin", type=float, default=1.0, help="m, ignore le champ proche")
    ap.add_argument("--phi0", type=float, default=0.0, help="offset angle faisceau (deg)")
    ap.add_argument("--flip-phi", action="store_true", help="inverse le sens du balayage")
    ap.add_argument("--with-map", action="store_true",
                    help="superpose le nuage Micron du SLAM (ruban 2.5D) aux parois")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bag = a.bag or os.path.join(root, "caves.bag")

    traj = np.genfromtxt(os.path.join(a.run, "trajectory.csv"),
                         delimiter=",", names=True)
    tz = traj["z"] if "z" in traj.dtype.names else np.zeros(len(traj))
    th_u = np.unwrap(traj["theta"])

    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    P, I = [], []
    n_beams = n_hits = 0
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic == a.topic]
        for c, _, raw in r.messages(connections=conns):
            m = ts.deserialize_ros1(raw, c.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if t < traj["time"][0] or t > traj["time"][-1]:
                continue
            prof = np.asarray(m.intensities, dtype=np.float32)
            if prof.size == 0:
                continue
            n_beams += 1
            res = float(m.range_max) / prof.size
            i0 = int(a.rmin / res)
            seg = prof[i0:]
            j = int(np.argmax(seg))
            if seg[j] <= a.threshold:
                continue
            n_hits += 1
            rr = (i0 + j + 0.5) * res
            phi = float(m.angle_min) + np.deg2rad(a.phi0)
            if a.flip_phi:
                phi = -phi
            # pose SLAM interpolée au stamp du faisceau
            px = np.interp(t, traj["time"], traj["x"])
            py = np.interp(t, traj["time"], traj["y"])
            pz = np.interp(t, traj["time"], tz)
            ps = np.interp(t, traj["time"], th_u)
            cy, sy = np.cos(ps), np.sin(ps)
            ly, lz = rr * np.cos(phi), rr * np.sin(phi)   # plan transverse (y', z')
            P.append((px - ly * sy, py + ly * cy, pz + lz))
            I.append(float(seg[j]))
    P = np.asarray(P, np.float32)
    print(f"faisceaux dans la fenêtre du run : {n_beams} | points de paroi : {n_hits}")
    if len(P) == 0:
        print("aucun point — baisser --threshold ?")
        return

    out_csv = os.path.join(a.run, "grotte_3d.csv")
    np.savetxt(out_csv, np.column_stack([P, np.asarray(I)]), delimiter=",",
               header="x,y,z,intensity", comments="", fmt="%.4f")
    print("->", out_csv)

    import plotly.graph_objects as go
    fig = go.Figure()
    if a.with_map:
        mp = os.path.join(a.run, "pointcloud.csv")
        if os.path.exists(mp):
            c = np.genfromtxt(mp, delimiter=",", names=True)
            cz = c["z"] if "z" in c.dtype.names else np.zeros(len(c))
            fig.add_trace(go.Scatter3d(
                x=c["x"], y=c["y"], z=cz, mode="markers",
                name="carte Micron (ruban 2.5D — tranches horizontales)",
                marker=dict(size=1.2, color="orange", opacity=0.45)))
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="parois (SeaKing)",
        marker=dict(size=1.3, color=P[:, 2], colorscale="Viridis", opacity=0.7,
                    colorbar=dict(title="z (m)", len=0.6))))
    fig.add_trace(go.Scatter3d(
        x=traj["x"], y=traj["y"], z=tz, mode="lines", name="trajectoire SLAM",
        line=dict(color="red", width=5)))
    fig.update_layout(title=f"{os.path.basename(a.run.rstrip('/'))} — grotte 3D "
                            f"(profiler vertical, {len(P)} pts)",
                      scene=dict(aspectmode="data", xaxis_title="x (m)",
                                 yaxis_title="y (m)", zaxis_title="z (m)"),
                      margin=dict(l=0, r=0, t=40, b=0))
    out_html = os.path.join(a.run, "grotte_3d.html")
    fig.write_html(out_html, include_plotlyjs=True)
    print("->", out_html)


if __name__ == "__main__":
    main()
