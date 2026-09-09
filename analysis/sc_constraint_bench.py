#!/usr/bin/env python3
"""Banc d'essai HORS-LIGNE de la QUALITÉ DES CONTRAINTES de boucle.

Question : sur de VRAIES revisites (GT proche), les nuages CFAR des deux passages
peuvent-ils s'aligner par ICP ? Si oui depuis une bonne init → le problème vient
de l'init shgo (→ corriger l'init). Si non → limite de la donnée (P900 épars).

Métrique : résidu ICP (distance moyenne au plus proche voisin après alignement).
Faible = scans alignables = contrainte exploitable.

Usage (conteneur ros1) :
  python3 sc_constraint_bench.py results/run_aracati_2026-06-13_200714
"""
import sys, os
import numpy as np
import cv2
from bruce_slam.CFAR import CFAR

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/run_aracati_2026-06-13_200714"
BAG = os.path.expanduser("~/Documents/Polytech/Stage4A/SLAM/sonar-SLAM/ARACATI_2017_8bits_full.bag")
TOPIC = "/son/compressed"
# params feature_aracati.yaml
NTC, NGC, PFA, ALG, THR = 40, 10, 0.01, "SOCA", 65
MAXR, FOV = 48.2896, 130.0
RES = 0.5  # downsample

tr = np.genfromtxt(f"{RUN}/trajectory.csv", delimiter=",", names=True)
kf = {int(k): (x, y, th) for k, x, y, th in zip(tr["keyframe_id"], tr["x"], tr["y"], tr["theta"])}
kf_t = {int(k): t for k, t in zip(tr["keyframe_id"], tr["time"])}
gt = np.genfromtxt(f"{RUN}/groundtruth.csv", delimiter=",", names=True); o = gt["time"].argsort()
gtt, gx, gy = gt["time"][o], gt["x"][o], gt["y"][o]
gp = lambda k: np.array([np.interp(kf_t[k], gtt, gx), np.interp(kf_t[k], gtt, gy)])
lp = np.genfromtxt(f"{RUN}/loops_detected.csv", delimiter=",", names=True)

true_pairs = []
for s, t, r in zip(lp["source_key"].astype(int), lp["target_key"].astype(int), lp["retenu"].astype(bool)):
    if r and s in kf and t in kf and np.linalg.norm(gp(s) - gp(t)) < 5:
        true_pairs.append((s, t))
np.random.default_rng(0).shuffle(true_pairs)
true_pairs = true_pairs[:30]
need = sorted({k for p in true_pairs for k in p})
print(f"vraies boucles testées : {len(true_pairs)}  keyframes : {len(need)}")

# --- extraction images bag ---
import rosbag
need_t = {k: kf_t[k] for k in need}; imgs = {}; tol = 0.15
with rosbag.Bag(BAG) as bag:
    tg = sorted(need_t.items(), key=lambda kv: kv[1]); ti = 0
    for _, m, _ in bag.read_messages(topics=[TOPIC]):
        ts = m.header.stamp.to_sec()
        while ti < len(tg) and ts > tg[ti][1] + tol: ti += 1
        if ti >= len(tg): break
        k, kt = tg[ti]
        if abs(ts - kt) <= tol and k not in imgs:
            im = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if im is not None: imgs[k] = im
print(f"images extraites : {len(imgs)}/{len(need)}")

# --- CFAR + conversion cartésienne métrique (réplique callback_cartesian) ---
det = CFAR(NTC, NGC, PFA, None)
def cloud(img):
    h, w = img.shape
    peaks = det.detect(img, ALG) & (img > THR)
    locs = np.c_[np.nonzero(peaks)]
    if len(locs) == 0: return np.empty((0, 2))
    mpp = MAXR / float(h)
    x = (h - locs[:, 0]) * mpp
    y = (locs[:, 1] - w / 2.0) * mpp
    r = np.hypot(x, y); b = np.arctan2(y, x)
    keep = (r > 0.3) & (r < MAXR - 0.3) & (np.abs(b) < np.deg2rad(FOV / 2) - 0.05)
    pts = np.c_[x[keep], y[keep]]
    if len(pts) and RES > 0:  # downsample grille
        q = np.round(pts / RES).astype(int)
        _, idx = np.unique(q, axis=0, return_index=True)
        pts = pts[idx]
    return pts
clouds = {k: cloud(imgs[k]) for k in imgs}
print("nuages CFAR : pts médian =", int(np.median([len(c) for c in clouds.values()])))

# --- ICP 2D point-à-point simple ---
def transform(P, dx, dy, dth):
    c, s = np.cos(dth), np.sin(dth)
    return np.c_[c * P[:, 0] - s * P[:, 1] + dx, s * P[:, 0] + c * P[:, 1] + dy]

def icp(src, dst, init=(0, 0, 0), iters=40):
    if len(src) < 5 or len(dst) < 5: return None, np.inf
    dx, dy, dth = init
    for _ in range(iters):
        P = transform(src, dx, dy, dth)
        # plus proche voisin (brute force, nuages petits)
        d2 = ((P[:, None, :] - dst[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1); nn = dst[j]; dmin = np.sqrt(d2[np.arange(len(P)), j])
        # rejet des appariements > 3 m (outliers)
        m = dmin < 3.0
        if m.sum() < 5: break
        Pm, Qm = P[m], nn[m]
        mp, mq = Pm.mean(0), Qm.mean(0)
        H = (Pm - mp).T @ (Qm - mq)
        U, _, Vt = np.linalg.svd(H); Rr = Vt.T @ U.T
        if np.linalg.det(Rr) < 0: Vt[-1] *= -1; Rr = Vt.T @ U.T
        ang = np.arctan2(Rr[1, 0], Rr[0, 0]); tt = mq - Rr @ mp
        # compose
        c, s = np.cos(ang), np.sin(ang)
        nx = c * dx - s * dy + tt[0]; ny = s * dx + c * dy + tt[1]
        dx, dy, dth = nx, ny, (dth + ang + np.pi) % (2 * np.pi) - np.pi
    P = transform(src, dx, dy, dth)
    d2 = ((P[:, None, :] - dst[None, :, :]) ** 2).sum(-1)
    res = np.sqrt(d2.min(1)).mean()
    return (dx, dy, dth), res

# init = pose relative de l'estimé (source -> target, repère target)
def est_init(s, t):
    xs, ys, ths = kf[s]; xt, yt, tht = kf[t]
    c, sn = np.cos(-tht), np.sin(-tht)
    dx = c * (xs - xt) - sn * (ys - yt)
    dy = sn * (xs - xt) + c * (ys - yt)
    return (dx, dy, (ths - tht + np.pi) % (2 * np.pi) - np.pi)

print(f"\n{'paire':>12} {'init_res':>9} {'icp_res':>8} {'Δicp(m)':>8}")
res_est, res_icp = [], []
for s, t in true_pairs:
    cs, ct = clouds.get(s), clouds.get(t)
    if cs is None or ct is None or len(cs) < 5 or len(ct) < 5: continue
    ini = est_init(s, t)
    # résidu à l'init (sans ICP)
    P0 = transform(cs, *ini)
    r0 = np.sqrt((((P0[:, None] - ct[None]) ** 2).sum(-1)).min(1)).mean()
    sol, r1 = icp(cs, ct, ini)
    if sol is None: continue
    move = np.hypot(sol[0] - ini[0], sol[1] - ini[1])
    res_est.append(r0); res_icp.append(r1)
    print(f"{s:>5}->{t:<5} {r0:>9.2f} {r1:>8.2f} {move:>8.2f}")
print(f"\nrésidu MÉDIAN : init(estimé)={np.median(res_est):.2f} m  après ICP={np.median(res_icp):.2f} m")
print("Interprétation : ICP résidu < ~1 m => scans alignables (init est le levier).")
print("                 ICP résidu > ~2 m => nuages trop épars (limite donnée).")
