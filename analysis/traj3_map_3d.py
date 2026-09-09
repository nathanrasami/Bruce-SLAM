#!/usr/bin/env python3
"""Cartes 3D traj3 (bags v4 : points en REPÈRE VÉHICULE, frame_id=auv0).

GT-free NATIF : p_monde = Rz(theta_slam)·p_veh + [x,y,z]_slam — aucune
dé-projection GT (contrairement aux bags v3, cf. profiler_slam_3d.py).
Produit DEUX cartes : profiler (sections verticales) et sonar principal
(tilt oscillant = la 3D que le SLAM lui-même voit).
Métrique : même carte composée avec les poses GT → NN médian SLAM vs GT.

Usage (conteneur ros1) :
  python3 traj3_map_3d.py <bag_traj3> <run_dir> <out_dir>
Sorties : <out_dir>/map3d_{profiler,sonar}.{png,html,npy} + métriques console.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import rosbag
import sensor_msgs.point_cloud2 as pc2


def yaw_of(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def compose(pts_veh, x, y, z, th):
    c, s = np.cos(th), np.sin(th)
    return np.c_[x + c * pts_veh[:, 0] - s * pts_veh[:, 1],
                 y + s * pts_veh[:, 0] + c * pts_veh[:, 1],
                 z + pts_veh[:, 2]]


def main(bag_path, run_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    traj = np.genfromtxt(os.path.join(run_dir, "trajectory.csv"),
                         delimiter=",", names=True)
    ts = traj["time"]
    th_slam = np.unwrap(traj["theta"])
    rng = np.random.default_rng(0)

    gt_t, gt_x, gt_y, gt_z, gt_th = [], [], [], [], []
    bag = rosbag.Bag(bag_path)
    for _, m, _ in bag.read_messages(topics=["/ground_truth"]):
        gt_t.append(m.header.stamp.to_sec())
        p = m.pose.pose.position
        gt_x.append(p.x); gt_y.append(p.y); gt_z.append(p.z)
        gt_th.append(yaw_of(m.pose.pose.orientation))
    gt_t = np.array(gt_t)
    gt_x, gt_y, gt_z = map(np.array, (gt_x, gt_y, gt_z))
    gt_th = np.unwrap(np.array(gt_th))

    maps = {"profiler": {"slam": [], "gt": []}, "sonar": {"slam": [], "gt": []}}
    counters = {"profiler": 0, "sonar": 0}
    for topic, m, _ in bag.read_messages(topics=["/profiler_points",
                                                 "/sonar_points"]):
        key = "profiler" if topic == "/profiler_points" else "sonar"
        counters[key] += 1
        if counters[key] % 2:
            continue
        t = m.header.stamp.to_sec()
        if t > ts[-1] + 0.5:
            continue
        pts = np.array(list(pc2.read_points(m, field_names=("x", "y", "z"),
                                            skip_nans=True)))
        if not len(pts):
            continue
        if len(pts) > 300:
            pts = pts[rng.choice(len(pts), 300, replace=False)]
        # pose SLAM interpolée (GT-free)
        w = compose(pts,
                    np.interp(t, ts, traj["x"]), np.interp(t, ts, traj["y"]),
                    np.interp(t, ts, traj["z"]), np.interp(t, ts, th_slam))
        maps[key]["slam"].append(w[w[:, 2] < 0])
        # même composition avec la pose GT (référence carte)
        g = compose(pts,
                    np.interp(t, gt_t, gt_x), np.interp(t, gt_t, gt_y),
                    np.interp(t, gt_t, gt_z), np.interp(t, gt_t, gt_th))
        maps[key]["gt"].append(g[g[:, 2] < 0])
    bag.close()

    from scipy.spatial import cKDTree
    for key in ("profiler", "sonar"):
        if not maps[key]["slam"]:
            print(f"{key}: aucun point — skip")
            continue
        S = np.vstack(maps[key]["slam"])
        G = np.vstack(maps[key]["gt"])
        # ⚠ les poses SLAM sont dans le repère LOCAL (départ origine), la GT dans
        # le repère MONDE PierHarbor → aligner par translation+yaw (Umeyama 2D
        # sur les trajectoires serait mieux ; ici : premier-pose, suffisant car
        # même point de départ)
        # alignement première pose : translation + rotation du départ GT
        dth = gt_th[0] - th_slam[0]
        c, s = np.cos(dth), np.sin(dth)
        S_al = np.c_[gt_x[0] + c * (S[:, 0] - traj["x"][0]) - s * (S[:, 1] - traj["y"][0]),
                     gt_y[0] + s * (S[:, 0] - traj["x"][0]) + c * (S[:, 1] - traj["y"][0]),
                     S[:, 2] - traj["z"][0] + gt_z[0]]
        idx = rng.choice(len(S_al), min(20000, len(S_al)), replace=False)
        dnn = cKDTree(G).query(S_al[idx], k=1)[0]
        med, p90 = np.median(dnn), np.percentile(dnn, 90)
        print(f"map3d_{key} : {len(S):,} pts | carte GT-free vs carte GT : "
              f"NN med {med:.3f} m p90 {p90:.3f} m")

        out = os.path.join(out_dir, f"map3d_{key}")
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(S_al[:, 0], S_al[:, 1], S_al[:, 2], s=0.3, c=S_al[:, 2],
                   cmap="viridis", linewidths=0)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title(f"traj3 PierHarbor — carte 3D GT-free ({key}) — "
                     f"{len(S):,} pts — NN {med:.2f}/{p90:.2f} m")
        fig.tight_layout(); fig.savefig(out + ".png", dpi=140)
        plt.close(fig)
        np.save(out + ".npy", S_al)
        try:
            import plotly.graph_objects as go
            i2 = rng.choice(len(S_al), min(120000, len(S_al)), replace=False)
            f2 = go.Figure(go.Scatter3d(
                x=S_al[i2, 0], y=S_al[i2, 1], z=S_al[i2, 2], mode="markers",
                marker=dict(size=1.2, color=S_al[i2, 2], colorscale="Viridis")))
            f2.update_layout(title=f"traj3 PierHarbor — carte 3D GT-free ({key})",
                             scene=dict(aspectmode="data"),
                             margin=dict(l=0, r=0, t=40, b=0))
            f2.write_html(out + ".html", include_plotlyjs=True)
        except ImportError:
            pass
        print("->", out + ".png/.npy(/.html)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
