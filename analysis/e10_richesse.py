#!/usr/bin/env python3
"""E10 traj8 : RICHESSE de l'image sonar au detecteur REEL (TRAJ8_DESIGN.md §6).

Rejoue sur les images /sonar d'un BAG la chaine exacte du pipeline
(feature_extraction.py:266-293, comme carte_2d_dense.py) : bridge x255 ->
CFAR SOCA (20/4/0.1/10) -> seuil (30) -> downsample 0.5 m -> outlier
(1.0 m / 5 pts), et compte les FEATURES PAR PING. C'est la metrique qui
commande le funnel NSSM : le verrou traj7r etait overlap med 18 < 50
(61 % de KF vides, med 14 feat/KF sur le reste).

PASS (bag de test traj8) : pings vides <= 20 % ET mediane features/ping >= 25.
La reference (traj7r, --ref) est mesuree par LE MEME code (echantillonnee).
Bonus PIEGES #21 : % de pings dont le max mono8 < 5 (aveugles au Sonar
Context avec intensity_threshold 5).

A lancer DANS le conteneur ros1 (CFAR/pcl compiles) :
  podman exec ros1 bash -lc 'source /opt/ros/noetic/setup.bash; \
    source ~/ros1_ws/devel/setup.bash; \
    python3 analysis/e10_richesse.py BAG_files/holoocean_3d_traj8_test.bag \
      --ref BAG_files/holoocean_3d_traj7r.bag --ref-every 8'
"""
import argparse
import sys
import numpy as np

CFAR_NTC, CFAR_NGC, CFAR_PFA, CFAR_RANK = 20, 4, 0.1, 10   # feature_holoocean.yaml
RES_DOWNSAMPLE, OUT_RADIUS, OUT_MINPTS = 0.5, 1.0, 5       # idem
HALF_FOV = 60.0


def stats_bag(bag_path, det, pcl, cv2, rosbag, seuil, every, range_max,
              t_min=0.0):
    n_feat, n_max = [], []
    t0 = None
    bag = rosbag.Bag(bag_path)
    i = 0
    for _, msg, _ in bag.read_messages(topics=["/sonar"]):
        i += 1
        if every > 1 and i % every != 1:
            continue
        if t0 is None:
            t0 = msg.header.stamp.to_sec()
        if msg.header.stamp.to_sec() - t0 < t_min:
            continue
        img32 = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        img = np.clip(img32 * 255.0, 0, 255).astype(np.float32)   # bridge x255
        n_max.append(float(img.max()))
        peaks = det.detect(img, "SOCA") & (img > seuil)
        rr, cc = np.nonzero(peaks)
        if len(rr) == 0:
            n_feat.append(0)
            continue
        r = rr * range_max / msg.height
        a = np.deg2rad(cc / (msg.width - 1) * 2 * HALF_FOV - HALF_FOV)
        pts = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
        pts = pcl.downsample(pts, RES_DOWNSAMPLE)
        if len(pts) > 0:
            pts = pcl.remove_outlier(pts.astype(np.float32), OUT_RADIUS, OUT_MINPTS)
        n_feat.append(int(len(pts)))
        if len(n_feat) % 200 == 0:
            print("  %d pings..." % len(n_feat), flush=True)
    bag.close()
    f = np.array(n_feat, float)
    m = np.array(n_max, float)
    return {"n": len(f), "vides": float((f == 0).mean()),
            "ge10": float((f >= 10).mean()), "ge25": float((f >= 25).mean()),
            "med": float(np.median(f)),
            "med_nonzero": float(np.median(f[f > 0])) if (f > 0).any() else 0.0,
            "p90": float(np.percentile(f, 90)),
            "sc_aveugles": float((m < 5.0).mean())}


def line(tag, s):
    print("  %-10s : %4d pings | vides %5.1f %% | med %5.1f (nonzero %5.1f, "
          "p90 %5.1f) | >=10 %5.1f %% | >=25 %5.1f %% | max<5 (SC aveugle) %4.1f %%"
          % (tag, s["n"], 100 * s["vides"], s["med"], s["med_nonzero"],
             s["p90"], 100 * s["ge10"], 100 * s["ge25"], 100 * s["sc_aveugles"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--ref", default=None, help="bag temoin (traj7r)")
    ap.add_argument("--seuil", type=float, default=30.0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--ref-every", type=int, default=8)
    ap.add_argument("--range-max", type=float, default=20.0,
                    help="RangeMax du /sonar (traj7r et traj8 : 20)")
    ap.add_argument("--t-min", type=float, default=0.0,
                    help="ignorer les pings avant t0+t_min (ex : 90 = errance "
                         "seule, exclut la phase A face au mur)")
    args = ap.parse_args()

    import cv2, rosbag
    from bruce_slam.CFAR import CFAR
    from bruce_slam import pcl
    det = CFAR(CFAR_NTC, CFAR_NGC, CFAR_PFA, CFAR_RANK)

    print("E10 richesse — detecteur reel (CFAR SOCA %d/%d/%.2f/%d, seuil %.0f, "
          "downsample %.1f, outlier %.1f/%d)"
          % (CFAR_NTC, CFAR_NGC, CFAR_PFA, CFAR_RANK, args.seuil,
             RES_DOWNSAMPLE, OUT_RADIUS, OUT_MINPTS))
    s = stats_bag(args.bag, det, pcl, cv2, rosbag, args.seuil, args.every,
                  args.range_max, args.t_min)
    line("TEST", s)
    if args.ref:
        sr = stats_bag(args.ref, det, pcl, cv2, rosbag, args.seuil,
                       args.ref_every, args.range_max)
        line("REF", sr)

    ok = (s["vides"] <= 0.20) and (s["med"] >= 25.0)
    print("\nE10 : vides %.1f %% (PASS si <=20) | med %.1f (PASS si >=25) -> %s"
          % (100 * s["vides"], s["med"], "PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
