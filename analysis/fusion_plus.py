#!/usr/bin/env python3
"""Fusion « + » : 2 sonars orthogonaux → vraie 3D dans le cône de recouvrement.

Méthode = StereoFLS (McConnell, Martin & Englot, IROS 2020, arXiv 2007.10407,
code jake3991/StereoFLS — même auteur que Bruce-SLAM), adaptée aux topics de
POINTS (ce que donnent nos bags), pas aux images polaires :
  1. apparier les pings /sonar_points (horizontal) ↔ fan vertical par STAMP ;
  2. en repère véhicule, ne garder que le recouvrement : |azimut| ≤ ouverture
     azimut du sonar vertical / 2 pour l'horizontal, |élévation| ≤ ouverture
     élévation du sonar horizontal / 2 pour le vertical (chez StereoFLS :
     verticalAperture 20° partagée) ;
  3. apparier par BIN DE RANGE (StereoFLS : bins pixels ; ici --rbin m) ; dans
     un bin, association gloutonne par similarité d'INTENSITÉ (remplace leurs
     patchs d'image, indisponibles sur des points) ; --unique = strict 1↔1 ;
  4. point fusionné : azimut du sonar horizontal + hauteur z du vertical +
     range moyenné → x=ρ·cos(az), y=ρ·sin(az), z=z_v avec ρ=√(R²−z_v²)
     (formule StereoFLS callback L624-636).

Le résultat est composé avec la pose SLAM (GT-free). Métriques :
  M1 (géométrie de fusion, indépendante du SLAM) : NN entre les points
     fusionnés composés GT et la carte de référence (échos bruts des 2 sonars
     composés GT) — un point fusionné juste retombe sur une paroi vue.
  M2 (qualité GT-free) : NN carte SLAM vs carte GT, alignement Umeyama
     (même convention que carte_3d.py).

Bags v3 (frame_id=map) : dé-projection via la pose GT du stamp = décodage du
format (aucune info GT dans la carte SLAM finale). Bags v4/traj4 (auv0) : rien.

Usage (conteneur ros1) :
  python3 analysis/fusion_plus.py <run_dir> [--bag <bag>] [--out <dir>]
      [--vert TOPIC] [--ap-hor 20] [--ap-vert 20] [--rbin 0.15]
      [--di-max 0.15] [--unique] [--nmax N]
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import rosbag
import sensor_msgs.point_cloud2 as pc2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_eval import umeyama, associer_par_temps
from carte_3d import yaw_of, rot_z, voxel_grid

RMIN = 0.7  # m — marge RangeMin (guide §3 : < 0.7 m du sonar = invisible)


def lire(m):
    """Nx4 (x,y,z,intensity) ; intensity=0 si absente du nuage."""
    has_i = any(f.name == "intensity" for f in m.fields)
    ch = ("x", "y", "z", "intensity") if has_i else ("x", "y", "z")
    p = np.array(list(pc2.read_points(m, field_names=ch, skip_nans=True)))
    if not len(p):
        return p.reshape(0, 4)
    return p if has_i else np.c_[p, np.zeros(len(p))]


def apparier_bin(rh, ih, rv, iv, rbin, di_max, unique):
    """Appariement par bin de range (indices h, indices v) à la StereoFLS.

    Dans chaque bin commun : association gloutonne par |Δintensité| croissant
    (nos deux vues d'un même point ont des incidences différentes → l'intensité
    est un critère FAIBLE ; di_max large, le range fait l'essentiel du travail).
    """
    kh, kv = np.round(rh / rbin).astype(int), np.round(rv / rbin).astype(int)
    out_h, out_v = [], []
    for b in np.intersect1d(kh, kv):
        H, V = np.flatnonzero(kh == b), np.flatnonzero(kv == b)
        if unique:
            if len(H) == 1 and len(V) == 1 and abs(ih[H[0]] - iv[V[0]]) < di_max:
                out_h.append(H[0]); out_v.append(V[0])
            continue
        cost = np.abs(ih[H][:, None] - iv[V][None, :])
        for _ in range(min(len(H), len(V))):
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            if cost[i, j] >= di_max:
                break
            out_h.append(H[i]); out_v.append(V[j])
            cost[i, :] = np.inf; cost[:, j] = np.inf
    return np.array(out_h, int), np.array(out_v, int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--out", default=None, help="dossier de sortie (défaut: run)")
    ap.add_argument("--hor", default="/sonar_points")
    ap.add_argument("--vert", default=None,
                    help="défaut: /sonar_vert_points si présent, sinon /profiler_points")
    ap.add_argument("--ap-hor", type=float, default=20.0,
                    help="ouverture ÉLÉVATION du sonar horizontal (deg, total)")
    ap.add_argument("--ap-vert", type=float, default=20.0,
                    help="ouverture AZIMUT du sonar vertical (deg, total)")
    ap.add_argument("--rbin", type=float, default=0.15)
    ap.add_argument("--di-max", type=float, default=0.15)
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--nmax", type=int, default=0, help="pings appariés max (0=tout)")
    a = ap.parse_args()

    bag_path = a.bag
    if bag_path is None:
        src = os.path.join(a.run, "bag_source.txt")
        if not os.path.isfile(src):
            sys.exit("fusion_plus : pas de --bag ni de bag_source.txt — abandon")
        bag_path = open(src).read().strip()
    out_dir = a.out or a.run
    os.makedirs(out_dir, exist_ok=True)

    traj = np.genfromtxt(os.path.join(a.run, "trajectory.csv"),
                         delimiter=",", names=True)
    ts, th_slam = traj["time"], np.unwrap(traj["theta"])
    bag = rosbag.Bag(bag_path)
    topics_bag = bag.get_type_and_topic_info()[1]
    vert = a.vert or ("/sonar_vert_points" if "/sonar_vert_points" in topics_bag
                      else "/profiler_points")
    if a.hor not in topics_bag or vert not in topics_bag:
        sys.exit(f"fusion_plus : topics absents ({a.hor}, {vert})")
    print(f"fusion : horizontal={a.hor} vertical={vert} "
          f"gates ±{a.ap_vert/2:.0f}° az / ±{a.ap_hor/2:.0f}° élév, "
          f"rbin={a.rbin} m, di_max={a.di_max}, unique={a.unique}")

    # ── GT (dé-projection v3 + référence M1/M2) ─────────────────────────────
    gt_t, gt_x, gt_y, gt_z, gt_th = [], [], [], [], []
    for _, m, _ in bag.read_messages(topics=["/ground_truth"]):
        gt_t.append(m.header.stamp.to_sec())
        p = m.pose.pose.position
        gt_x.append(p.x); gt_y.append(p.y); gt_z.append(p.z)
        gt_th.append(yaw_of(m.pose.pose.orientation))
    gt_t = np.array(gt_t)
    gt_x, gt_y, gt_z = map(np.array, (gt_x, gt_y, gt_z))
    gt_th = np.unwrap(np.array(gt_th)) if gt_th else np.array([])
    has_gt = len(gt_t) > 0

    def vers_vehicule(xyz, t, frame):
        if frame != "map":
            return xyz
        return rot_z(xyz - [np.interp(t, gt_t, gt_x), np.interp(t, gt_t, gt_y),
                            np.interp(t, gt_t, gt_z)], -np.interp(t, gt_t, gt_th))

    def compose(pts, t, tt, xx, yy, zz, th):
        w = rot_z(pts, np.interp(t, tt, th))
        return w + [np.interp(t, tt, xx), np.interp(t, tt, yy), np.interp(t, tt, zz)]

    # ── flux : appariement par stamp, fusion par paire de pings ────────────
    ga_v, ga_h = np.radians(a.ap_vert / 2), np.radians(a.ap_hor / 2)
    buf = {}
    F_stamped = []          # (t, pts_fusion_véhicule Nx4 : xyz + i)
    ref_chunks = []         # référence M1 : échos bruts composés GT (sous-éch.)
    stats = dict(pairs=0, nh=0, nv=0, nm=0, di=[])
    rng = np.random.default_rng(0)
    for topic, m, _ in bag.read_messages(topics=[a.hor, vert]):
        t = m.header.stamp.to_sec()
        if t > ts[-1] + 0.5:
            break
        key = int(round(t * 50))              # grille 20 ms
        buf.setdefault(key, {})[topic] = (lire(m), m.header.frame_id, t)
        if len(buf[key]) < 2:
            continue
        (ph, fh, th_) = buf[key][a.hor]
        (pv, fv, _) = buf[key][vert]
        del buf[key]
        if len(ph) < 3 or len(pv) < 3 or (fh == "map" and not has_gt):
            continue
        stats["pairs"] += 1
        vh = np.c_[vers_vehicule(ph[:, :3], th_, fh), ph[:, 3]]
        vv = np.c_[vers_vehicule(pv[:, :3], th_, fv), pv[:, 3]]
        # gates de recouvrement (repère véhicule : x avant, y gauche, z haut)
        rh = np.linalg.norm(vh[:, :3], axis=1)
        az = np.arctan2(vh[:, 1], vh[:, 0])
        gh = (np.abs(az) <= ga_v) & (rh > RMIN)
        rv = np.linalg.norm(vv[:, :3], axis=1)
        el = np.arctan2(vv[:, 2], np.hypot(vv[:, 0], vv[:, 1]))
        gv = (np.abs(el) <= ga_h) & (rv > RMIN) & (vv[:, 0] > 0)
        stats["nh"] += int(gh.sum()); stats["nv"] += int(gv.sum())
        # référence M1 : échos bruts (hors gate) composés GT, sous-échantillonnés
        if has_gt and stats["pairs"] % 4 == 0:
            for P in (vh, vv):
                q = P[:, :3]
                if len(q) > 120:
                    q = q[rng.choice(len(q), 120, replace=False)]
                ref_chunks.append(compose(q, th_, gt_t, gt_x, gt_y, gt_z, gt_th))
        if gh.sum() == 0 or gv.sum() == 0:
            continue
        H, V = vh[gh], vv[gv]
        ih_, iv_ = apparier_bin(rh[gh], H[:, 3], rv[gv], V[:, 3],
                                a.rbin, a.di_max, a.unique)
        if not len(ih_):
            continue
        stats["nm"] += len(ih_)
        stats["di"].extend(np.abs(H[ih_, 3] - V[iv_, 3]).tolist())
        R = (rh[gh][ih_] + rv[gv][iv_]) / 2.0
        zf = V[iv_, 2]                                    # hauteur du vertical
        rho = np.sqrt(np.maximum(R ** 2 - zf ** 2, 0.0))
        azf = az[gh][ih_]                                 # azimut de l'horizontal
        pts = np.c_[rho * np.cos(azf), rho * np.sin(azf), zf,
                    (H[ih_, 3] + V[iv_, 3]) / 2.0]
        F_stamped.append((th_, pts))
        if a.nmax and len(F_stamped) >= a.nmax:
            break
    bag.close()
    if not F_stamped:
        sys.exit("fusion_plus : 0 point fusionné — vérifier gates/rbin/topics.")

    npings = len(F_stamped)
    print(f"pings appariés traités : {stats['pairs']} | avec fusion : {npings} | "
          f"features gated moy. h={stats['nh']/stats['pairs']:.0f} "
          f"v={stats['nv']/stats['pairs']:.0f} | matches/ping (fusionnés) "
          f"{stats['nm']/npings:.1f} | |Δi| méd {np.median(stats['di']):.3f}")

    # étalement angulaire : la fusion doit produire des points HORS des 2 plans
    allF = np.vstack([p for _, p in F_stamped])
    azs = np.degrees(np.arctan2(allF[:, 1], allF[:, 0]))
    els = np.degrees(np.arctan2(allF[:, 2], np.hypot(allF[:, 0], allF[:, 1])))
    off = (np.abs(azs) > 2) & (np.abs(els) > 2)
    print(f"points fusionnés : {len(allF):,} | az std {azs.std():.1f}° | "
          f"élév std {els.std():.1f}° | hors des 2 plans (>2°/>2°) : "
          f"{100*off.mean():.0f} %")

    # ── composition SLAM (livrable) et GT (métriques) ───────────────────────
    S = np.vstack([compose(p[:, :3], t, ts, traj["x"], traj["y"], traj["z"],
                           th_slam) for t, p in F_stamped])
    Gc = np.vstack([compose(p[:, :3], t, gt_t, gt_x, gt_y, gt_z, gt_th)
                    for t, p in F_stamped]) if has_gt else None

    from scipy.spatial import cKDTree
    verdicts = []
    if has_gt and ref_chunks:
        ref = voxel_grid(np.vstack(ref_chunks), 0.2)
        idx = rng.choice(len(Gc), min(20000, len(Gc)), replace=False)
        d1 = cKDTree(ref).query(Gc[idx], k=1)[0]
        m1, p1 = np.median(d1), np.percentile(d1, 90)
        ok1 = m1 < 0.5
        verdicts.append(ok1)
        print(f"M1 géométrie fusion (GT-composé vs échos bruts GT) : "
              f"NN méd {m1:.3f} m | p90 {p1:.3f} m → {'PASS' if ok1 else 'FAIL'} "
              f"(seuil méd < 0.5 m)")
    if has_gt:
        gxy = associer_par_temps(ts, gt_t, gt_x, gt_y)
        _, Rm, tr = umeyama(np.column_stack([traj["x"], traj["y"]]), gxy,
                            with_scale=False)
        z_off = np.median(np.interp(ts, gt_t, gt_z) - traj["z"])
        S_al = np.c_[(S[:, :2] @ Rm.T) + tr, S[:, 2] + z_off]
        idx = rng.choice(len(S_al), min(20000, len(S_al)), replace=False)
        d2 = cKDTree(Gc).query(S_al[idx], k=1)[0]
        print(f"M2 carte GT-free vs carte GT (Umeyama) : NN méd "
              f"{np.median(d2):.3f} m | p90 {np.percentile(d2, 90):.3f} m")

    # ── sorties (carte immergée z<0, voxel 0.2 comme carte_3d) ─────────────
    Sm = voxel_grid(S[S[:, 2] < 0], 0.2)
    run_name = os.path.basename(a.run.rstrip("/"))
    out = os.path.join(out_dir, "fusion_plus")
    np.save(out + ".npy", Sm)
    tx, ty, tz = traj["x"], traj["y"], traj["z"]
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Sm[:, 0], Sm[:, 1], Sm[:, 2], s=0.6, c=Sm[:, 2], cmap="viridis",
               linewidths=0, alpha=0.6)
    ax.plot(tx, ty, tz, color="red", lw=1.5, label="trajectoire SLAM")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"{run_name} — fusion « + » (StereoFLS) GT-free — "
                 f"{len(Sm):,} pts", fontsize=9)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=140)
    plt.close(fig)
    try:
        import plotly.graph_objects as go
        i2 = rng.choice(len(Sm), min(150000, len(Sm)), replace=False)
        f2 = go.Figure(go.Scatter3d(
            x=Sm[i2, 0], y=Sm[i2, 1], z=Sm[i2, 2], mode="markers",
            name="fusion « + » (vraie 3D, recouvrement)",
            marker=dict(size=1.5, color=Sm[i2, 2], colorscale="Viridis",
                        colorbar=dict(title="z (m)"))))
        f2.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines",
                                  line=dict(color="red", width=5),
                                  name="trajectoire SLAM"))
        f2.update_layout(title=f"{run_name} — fusion « + » GT-free",
                         scene=dict(aspectmode="data"),
                         margin=dict(l=0, r=0, t=40, b=0))
        f2.write_html(out + ".html", include_plotlyjs=True)
        print("->", out + ".html")
    except ImportError:
        print("⚠ plotly absent : pas de .html")
    print("->", out + ".png/.npy")
    if verdicts and all(verdicts):
        print("VERDICT : PASS")
    elif verdicts:
        print("VERDICT : FAIL")


if __name__ == "__main__":
    main()
