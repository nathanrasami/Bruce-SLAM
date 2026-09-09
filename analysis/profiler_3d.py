#!/usr/bin/env python3
"""Reconstruction rapide de /profiler_points (bag traj2) : nuage 3D monde.
Usage (DANS le conteneur ros1) : python3 analysis/profiler_3d.py <bag> <out.png>
Filtre z<0 (murs émergés, guide v3 §4.1) ; 1 ping sur 3, ≤400 pts/ping."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — enregistre la projection '3d' (matplotlib conteneur)
import rosbag
import sensor_msgs.point_cloud2 as pc2

bag_path, out_png = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
pts, n = [], 0
bag = rosbag.Bag(bag_path)
for _, msg, _ in bag.read_messages(topics=["/profiler_points"]):
    if n % 3 == 0:
        arr = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"),
                                            skip_nans=True)))
        if len(arr):
            arr = arr[arr[:, 2] < 0.0]
            if len(arr) > 400:
                arr = arr[rng.choice(len(arr), 400, replace=False)]
            pts.append(arr)
    n += 1
bag.close()
P = np.vstack(pts)
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.3, c=P[:, 2], cmap="viridis", linewidths=0)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title(f"/profiler_points (monde, z<0) — {len(P):,} pts, {n} pings")
fig.tight_layout()
fig.savefig(out_png, dpi=140)
np.save(out_png.replace(".png", ".npy"), P)
print("->", out_png, f"({len(P)} points)")
