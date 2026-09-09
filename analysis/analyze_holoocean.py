#!/usr/bin/env python3
"""
Analyse HoloOcean simulation results:
  1. Extract GT from test.bag
  2. Load Bruce-SLAM trajectory from results/
  3. Compute ATE + plot
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from traj_eval import associer_par_temps, umeyama, appliquer, calculer_ate

# le script vit dans analysis/ → la racine du dépôt est UN cran au-dessus
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAG_PATH     = os.environ.get("BAG_HOLO", os.path.join(_ROOT, "test.bag"))
RESULTS_DIR  = os.environ.get("SLAM_RESULTS_DIR", os.path.join(_ROOT, "results"))

# ── 1. Extract GT from bag ────────────────────────────────────────────────────
def extract_gt(bag_path):
    """Read /ground_truth from a ROS1 bag without needing ROS (uses rosbags lib).
    Extracts x, y, z and full orientation (roll, pitch, yaw) for future 3D use."""
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    times, xs, ys, zs, rolls, pitches, yaws = [], [], [], [], [], [], []
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == "/ground_truth"]
        for conn, _, raw in reader.messages(connections=conns):
            msg = ts.deserialize_ros1(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            r, pi, ya = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_euler("xyz")
            times.append(t); xs.append(p.x); ys.append(p.y); zs.append(p.z)
            rolls.append(r); pitches.append(pi); yaws.append(ya)
    return pd.DataFrame({"time": times, "x": xs, "y": ys, "z": zs,
                         "roll": rolls, "pitch": pitches, "yaw": yaws})

# GT : de préférence le groundtruth.csv DU RUN (exporté par gt_odom_to_pose →
# slam_ros, mêmes stamps que la traj) ; le bag n'est qu'un secours si absent.
gt_csv = os.path.join(RESULTS_DIR, "groundtruth.csv")
if os.path.exists(gt_csv):
    gt = pd.read_csv(gt_csv)
    print(f"GT depuis {gt_csv} : {len(gt)} poses")
else:
    print(f"groundtruth.csv absent — extraction du bag {BAG_PATH}...")
    gt = extract_gt(BAG_PATH)
    print(f"  GT poses: {len(gt)}, duration: {gt['time'].iloc[-1]-gt['time'].iloc[0]:.1f}s")

# ── 2. Load Bruce-SLAM trajectory ─────────────────────────────────────────────
traj_path = os.path.join(RESULTS_DIR, "trajectory.csv")
if not os.path.exists(traj_path):
    print(f"ERROR: {traj_path} not found — run Bruce-SLAM first")
    exit(1)

traj = pd.read_csv(traj_path)
print(f"  Bruce-SLAM KFs: {len(traj)}")

# ── 3. Align + ATE ────────────────────────────────────────────────────────────
src    = traj[["x", "y"]].to_numpy()
gt_xy  = associer_par_temps(traj["time"], gt["time"], gt["x"], gt["y"])
s, R, t = umeyama(src, gt_xy)
est    = appliquer(s, R, t, src)
ate    = calculer_ate(est, gt_xy)
print(f"  ATE Bruce-SLAM: {ate:.2f} m")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

gx, gy = gt["x"].to_numpy(), gt["y"].to_numpy()
ax.plot(gx, gy, label="Ground truth", color="red", linestyle="--", linewidth=1.5)
ax.plot(gx[0], gy[0], marker="*", color="red", markersize=14)
ax.plot(gx[-1], gy[-1], marker="X", color="red", markersize=12)

ax.plot(est[:, 0], est[:, 1], label=f"Bruce-SLAM (ATE={ate:.2f} m)",
        color="black", linewidth=1.5)
ax.plot(est[0, 0], est[0, 1], marker="*", color="black", markersize=14, label="Start")
ax.plot(est[-1, 0], est[-1, 1], marker="X", color="black", markersize=12, label="End")

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("HoloOcean — Bruce-SLAM vs Ground truth")
ax.legend(); ax.axis("equal"); ax.grid(True)
plt.tight_layout()

out = os.path.join(RESULTS_DIR, "holoocean_trajectory_plot.png")
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
