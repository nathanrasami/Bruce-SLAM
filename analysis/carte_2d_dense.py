#!/usr/bin/env python3
"""Carte 2D DENSE d'un run holoocean — option B′ de FABLE §9-bis, GT-free.

Rejoue le détecteur RÉEL du pipeline (CFAR SOCA + seuil + downsample + outlier,
chaîne exacte de feature_extraction.py:266-293) sur TOUS les pings /sonar du bag
(la carte du run n'accumule que les keyframes, ~1 ping sur 8), projette le long
des poses SLAM (trajectory.csv) et voxelise. Deux différences assumées avec le
pipeline (les deux motivées FABLE §9/§9-bis) :
  - seuil réglable (--seuil 30 par défaut : nos images plafonnent à 0.49 brut,
    le 50 du yaml est calibré sur test.bag du collègue) ;
  - filtre ANTI-BAVURE optionnel (max local en azimut) : la bavure tangentielle
    des échos forts passe le CFAR (range-only) et dessine des arcs centrés sur
    le sonar — on ne garde que la crête de l'arc.

À lancer DANS le conteneur ros1 (CFAR/pcl compilés) :
  podman exec ros1 bash -lc 'source /opt/ros/noetic/setup.bash; \
    source ~/ros1_ws/devel/setup.bash; \
    python3 analysis/carte_2d_dense.py results/<run> [--seuil 30] [--every 1]'

Sorties : <run>/carte_2d_dense_s<seuil>.png + .npz (cellules 0.2 m + comptes).
"""
import argparse, os, sys
import numpy as np

RANGE_MAX, HALF_FOV = 40.0, 60.0          # sonar SLAM holoocean (bridge)
CFAR_NTC, CFAR_NGC, CFAR_PFA, CFAR_RANK = 20, 4, 0.1, 10   # feature_holoocean.yaml
RES_DOWNSAMPLE, OUT_RADIUS, OUT_MINPTS = 0.5, 1.0, 5       # idem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--bag", default=None, help="défaut : <run>/bag_source.txt")
    ap.add_argument("--seuil", type=int, default=30, help="seuil mono8 (pipeline: 50)")
    ap.add_argument("--every", type=int, default=1, help="1 ping sur N")
    ap.add_argument("--bavure", type=int, default=7,
                    help="fenêtre anti-bavure en colonnes (±N/2, 0 = off)")
    ap.add_argument("--voxel", type=float, default=0.2)
    args = ap.parse_args()

    run = args.run_dir.rstrip("/")
    bag_path = args.bag or open(os.path.join(run, "bag_source.txt")).read().strip()
    out = os.path.join(run, f"carte_2d_dense_s{args.seuil}")

    import cv2, rosbag
    from bruce_slam.CFAR import CFAR
    from bruce_slam import pcl

    traj = np.genfromtxt(os.path.join(run, "trajectory.csv"),
                         delimiter=",", names=True)
    tt, tx, ty = traj["time"], traj["x"], traj["y"]
    tth = np.unwrap(traj["theta"])

    det = CFAR(CFAR_NTC, CFAR_NGC, CFAR_PFA, CFAR_RANK)
    kern = np.ones((1, args.bavure), np.uint8) if args.bavure > 1 else None
    pts_w, t0, i, kept = [], None, 0, 0
    bag = rosbag.Bag(bag_path)
    for _, msg, _ in bag.read_messages(topics=["/sonar"]):
        i += 1
        if i % args.every != 1 and args.every > 1:
            continue
        img32 = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        img = np.clip(img32 * 255.0, 0, 255).astype(np.float32)   # bridge ×255
        tsec = msg.header.stamp.to_sec()
        if t0 is None:
            t0 = tsec
        trel = tsec - t0
        if trel < tt[0] - 0.5 or trel > tt[-1] + 0.5:
            continue
        peaks = det.detect(img, "SOCA") & (img > args.seuil)
        if kern is not None:      # anti-bavure : crête locale en azimut seulement
            loc_max = cv2.dilate(img, kern)
            peaks &= img >= loc_max - 1e-3
        rr, cc = np.nonzero(peaks)
        if len(rr) == 0:
            continue
        r = rr * RANGE_MAX / msg.height
        a = np.deg2rad(cc / (msg.width - 1) * 2 * HALF_FOV - HALF_FOV)
        # y = +r sin(a) : colonnes hautes = BÂBORD (PIEGES #14)
        pts = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
        pts = pcl.downsample(pts, RES_DOWNSAMPLE)
        if len(pts) > 0:
            pts = pcl.remove_outlier(pts.astype(np.float32), OUT_RADIUS, OUT_MINPTS)
        if len(pts) == 0:
            continue
        x = float(np.interp(trel, tt, tx)); y = float(np.interp(trel, tt, ty))
        th = float(np.interp(trel, tt, tth)); c, s = np.cos(th), np.sin(th)
        pts_w.append(np.stack([x + c * pts[:, 0] - s * pts[:, 1],
                               y + s * pts[:, 0] + c * pts[:, 1]],
                              axis=1).astype(np.float32))
        kept += 1
        if kept % 500 == 0:
            print(f"  {kept} pings (t={trel:.0f}s)", flush=True)
    bag.close()

    P = np.concatenate(pts_w)
    key = np.round(P / args.voxel).astype(np.int64)
    uk, cnt = np.unique(key, axis=0, return_counts=True)
    centers = uk * args.voxel
    np.savez(out + ".npz", centers=centers, cnt=cnt,
             traj=np.stack([tx, ty], axis=1).astype(np.float32),
             meta=np.array([args.seuil, args.every, args.bavure, args.voxel]))
    print(f"{len(P)} pts bruts, {len(uk)} cellules {args.voxel} m -> {out}.npz")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
    for ax, mmin, ttl in [(axes[0], 1, "toutes cellules"),
                          (axes[1], 2, "persistance >= 2 pings")]:
        m = cnt >= mmin
        ax.scatter(centers[m, 0], centers[m, 1], s=0.5, c="crimson", alpha=0.6)
        ax.plot(tx, ty, color="deepskyblue", lw=0.8)
        ax.set_aspect("equal"); ax.set_xlabel("x (m)")
        ax.set_title(f"{ttl} — {int(m.sum())} cellules")
    axes[0].set_ylabel("y (m)")
    plt.suptitle(f"{os.path.basename(run)} — carte 2D dense GT-free "
                 f"(CFAR réel, seuil {args.seuil}, 1 ping/{args.every}, "
                 f"anti-bavure ±{args.bavure//2} col, voxel {args.voxel} m)")
    plt.tight_layout()
    plt.savefig(out + ".png", dpi=110)
    print("->", out + ".png")

if __name__ == "__main__":
    main()
