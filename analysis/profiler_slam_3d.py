#!/usr/bin/env python3
"""Carte 3D GT-FREE : sections /profiler_points re-projetées sur la trajectoire SLAM.

Le bag publie les points profiler en repère MONDE (transformés via la pose GT au
même stamp — artefact du format de bag : un vrai système loggerait en repère
capteur). On INVERSE cette transformation avec la même pose GT (opération de
décodage, pas d'information GT injectée dans le résultat), puis on re-projette
chaque section sur la pose SLAM (x, y, theta de trajectory.csv + z = profondeur
embarquée) interpolée au stamp du ping. Résultat : carte 3D construite uniquement
avec la trajectoire estimée par le SLAM (GT-free).

Usage (conteneur ros1) :
  python3 profiler_slam_3d.py <bag_traj2> <run_dir> <out_prefix>
Sorties : <out_prefix>.png / .html / .npy + métrique vs carte GT (NN médian).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import rosbag
import sensor_msgs.point_cloud2 as pc2


def yaw_of(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main(bag_path, run_dir, out_prefix):
    traj = np.genfromtxt(f"{run_dir}/trajectory.csv", delimiter=",", names=True)
    t_slam = traj["time"]
    rng = np.random.default_rng(0)

    # poses GT du bag (pour DÉ-projeter les points monde -> repère véhicule)
    gt_t, gt_p, gt_yaw = [], [], []
    bag = rosbag.Bag(bag_path)
    for _, m, _ in bag.read_messages(topics=["/ground_truth"]):
        gt_t.append(m.header.stamp.to_sec())
        p = m.pose.pose.position
        gt_p.append((p.x, p.y, p.z))
        gt_yaw.append(yaw_of(m.pose.pose.orientation))
    gt_t = np.array(gt_t); gt_p = np.array(gt_p); gt_yaw = np.unwrap(np.array(gt_yaw))

    def interp_pose(tq, tt, pp, yy):
        x = np.interp(tq, tt, pp[:, 0]); y = np.interp(tq, tt, pp[:, 1])
        z = np.interp(tq, tt, pp[:, 2]); th = np.interp(tq, tt, yy)
        return x, y, z, th

    P_slam, P_gt, n = [], [], 0
    for _, m, _ in bag.read_messages(topics=["/profiler_points"]):
        ts = m.header.stamp.to_sec()
        if ts > t_slam[-1] + 0.5:
            break
        if n % 2 == 0:
            pts = np.array(list(pc2.read_points(m, field_names=("x", "y", "z"),
                                                skip_nans=True)))
            if len(pts):
                pts = pts[pts[:, 2] < 0.0]              # murs émergés (guide §4.1)
                if len(pts) > 300:
                    pts = pts[rng.choice(len(pts), 300, replace=False)]
                # 1) monde -> véhicule via pose GT du stamp (décodage)
                gx, gy, gz, gth = interp_pose(ts, gt_t, gt_p, gt_yaw)
                c, s = np.cos(-gth), np.sin(-gth)
                d = pts - [gx, gy, gz]
                loc = np.c_[c * d[:, 0] - s * d[:, 1],
                            s * d[:, 0] + c * d[:, 1], d[:, 2]]
                # 2) véhicule -> monde via pose SLAM interpolée (GT-free)
                sx = np.interp(ts, t_slam, traj["x"])
                sy = np.interp(ts, t_slam, traj["y"])
                sz = np.interp(ts, t_slam, traj["z"])
                sth = np.interp(ts, t_slam, np.unwrap(traj["theta"]))
                c2, s2 = np.cos(sth), np.sin(sth)
                P_slam.append(np.c_[sx + c2 * loc[:, 0] - s2 * loc[:, 1],
                                    sy + s2 * loc[:, 0] + c2 * loc[:, 1],
                                    sz + loc[:, 2]])
                P_gt.append(pts)
        n += 1
    bag.close()
    S, G = np.vstack(P_slam), np.vstack(P_gt)

    # métrique : NN médian carte-SLAM -> carte-GT (sous-échantillonné)
    # ⚠ repère SLAM (origine = 1ʳᵉ pose) ≠ repère monde GT → aligner AVANT la
    # métrique (SE(2) Umeyama sur trajectoires, comme l'ATE), sinon on mesure le
    # décalage de repère (~6 m constaté) et pas la qualité de la carte.
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from traj_eval import umeyama, associer_par_temps
    gxy = associer_par_temps(t_slam, gt_t, gt_p[:, 0], gt_p[:, 1])
    _, Ral, tal = umeyama(np.c_[traj["x"], traj["y"]], gxy, with_scale=False)
    S[:, :2] = S[:, :2] @ Ral.T + tal
    from scipy.spatial import cKDTree
    idx = rng.choice(len(S), min(20000, len(S)), replace=False)
    dnn = cKDTree(G).query(S[idx], k=1)[0]
    med, p90 = np.median(dnn), np.percentile(dnn, 90)
    print(f"carte GT-free vs carte GT : NN médian {med:.3f} m | p90 {p90:.3f} m "
          f"| {len(S)} pts, {n} pings")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(S[:, 0], S[:, 1], S[:, 2], s=0.3, c=S[:, 2], cmap="viridis",
               linewidths=0)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"Carte 3D GT-FREE (profiler × traj SLAM) — {len(S):,} pts — "
                 f"NN vs GT {med:.2f}/{p90:.2f} m")
    fig.tight_layout(); fig.savefig(out_prefix + ".png", dpi=140)
    np.save(out_prefix + ".npy", S)
    try:
        import plotly.graph_objects as go
        i2 = rng.choice(len(S), min(120000, len(S)), replace=False)
        f2 = go.Figure(go.Scatter3d(x=S[i2, 0], y=S[i2, 1], z=S[i2, 2],
                                    mode="markers",
                                    marker=dict(size=1.2, color=S[i2, 2],
                                                colorscale="Viridis")))
        f2.update_layout(title="Carte 3D GT-free (profiler × traj SLAM)",
                         scene=dict(aspectmode="data"),
                         margin=dict(l=0, r=0, t=40, b=0))
        f2.write_html(out_prefix + ".html", include_plotlyjs=True)
    except ImportError:
        pass
    print("->", out_prefix + ".png/.npy(/.html)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
