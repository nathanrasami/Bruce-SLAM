#!/usr/bin/env python3
"""Banc d'essai HORS-LIGNE des descripteurs Sonar Context.

But : mesurer si un descripteur sépare les VRAIES revisites (GT proche) des
FAUSSES (GT lointain), sans lancer de run SLAM de 2 h. Étiquetage par le GT.

Usage (dans le conteneur ros1) :
  python3 sc_descriptor_bench.py results/run_aracati_2026-06-13_171435
"""
import sys, os
import numpy as np
import cv2

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/run_aracati_2026-06-13_171435"
BAG = os.path.expanduser("~/Documents/Polytech/Stage4A/SLAM/sonar-SLAM/ARACATI_2017_8bits_full.bag")
SONAR_TOPIC = "/son/compressed"
FOV_DEG = 130.0
A, R = 40, 40          # cases descripteur
MAX_AZ, MAX_RG = 10, 5  # shifts (comme le YAML)
THR = 65               # seuil "structure" (= filter/threshold du pipeline)

# ---------- 1. pairs vraies/fausses via GT ----------
tr = np.genfromtxt(f"{RUN}/trajectory.csv", delimiter=",", names=True)
kf_t = {int(k): t for k, t in zip(tr["keyframe_id"], tr["time"])}
gt = np.genfromtxt(f"{RUN}/groundtruth.csv", delimiter=",", names=True)
o = gt["time"].argsort(); gtt, gx, gy = gt["time"][o], gt["x"][o], gt["y"][o]
def gtpos(kf):
    t = kf_t[kf]; return np.array([np.interp(t, gtt, gx), np.interp(t, gtt, gy)])

lp = np.genfromtxt(f"{RUN}/loops_detected.csv", delimiter=",", names=True)
pairs = list(zip(lp["source_key"].astype(int), lp["target_key"].astype(int)))
true_pairs, false_pairs = [], []
for s, t in pairs:
    if s not in kf_t or t not in kf_t:
        continue
    d = np.linalg.norm(gtpos(s) - gtpos(t))
    if d < 4.0:
        true_pairs.append((s, t))
    elif 30.0 < d < 70.0:
        false_pairs.append((s, t))
rng = np.random.default_rng(0)
rng.shuffle(true_pairs); rng.shuffle(false_pairs)
true_pairs = true_pairs[:50]; false_pairs = false_pairs[:50]
need = sorted({k for p in true_pairs + false_pairs for k in p})
print(f"pairs vraies={len(true_pairs)} fausses={len(false_pairs)} | keyframes à extraire={len(need)}")

# ---------- 2. extraire les images du bag aux timestamps keyframes ----------
import rosbag
need_t = {k: kf_t[k] for k in need}
imgs = {}
tol = 0.15
with rosbag.Bag(BAG) as bag:
    targets = sorted(need_t.items(), key=lambda kv: kv[1])
    ti = 0
    for _, msg, t in bag.read_messages(topics=[SONAR_TOPIC]):
        ts = msg.header.stamp.to_sec()
        while ti < len(targets) and ts > targets[ti][1] + tol:
            ti += 1
        if ti >= len(targets):
            break
        k, kt = targets[ti]
        if abs(ts - kt) <= tol and k not in imgs:
            arr = np.frombuffer(msg.data, np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if im is not None:
                imgs[k] = im
print(f"images extraites : {len(imgs)}/{len(need)}")
true_pairs = [(s, t) for s, t in true_pairs if s in imgs and t in imgs]
false_pairs = [(s, t) for s, t in false_pairs if s in imgs and t in imgs]
print(f"pairs exploitables : vraies={len(true_pairs)} fausses={len(false_pairs)}")

# ---------- 3. polar remap (réplique feature_extraction._polar_remap) ----------
_cache = {}
def polar_remap(img):
    h, w = img.shape
    key = (h, w)
    if key not in _cache:
        a_px, r_px = A * 8, R * 8
        half = np.deg2rad(FOV_DEG / 2.0)
        az = np.linspace(-half, half, a_px, dtype=np.float32)
        rr = np.linspace(0.0, float(h - 1), r_px, dtype=np.float32)
        rg, ag = np.meshgrid(rr, az, indexing="ij")
        mx = (w / 2.0 + rg * np.sin(ag)).astype(np.float32)
        my = (h - rg * np.cos(ag)).astype(np.float32)
        _cache[key] = (mx, my)
    mx, my = _cache[key]
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0)

def pool(mat, fn):
    # mat (r_px, a_px) -> transpose (azimuth, range) -> pool vers (A,R)
    m = mat.T
    As, Rs = m.shape
    ab, rb = max(As // A, 1), max(Rs // R, 1)
    m = m[:ab * A, :rb * R].reshape(A, ab, R, rb)
    return fn(m, axis=(1, 3)).astype(np.float32)

# ---------- 4. variantes de descripteur ----------
def ctx_v0_max(img):              # actuel : max-pool intensité brute
    c = pool(polar_remap(img).astype(np.float32), np.max)
    m = c.max(); return c / m if m > 0 else c

def ctx_v1_rangenorm(img):        # normalisation par anneau de range
    p = polar_remap(img).astype(np.float32)
    c = pool(p, np.max)
    # chaque colonne range divisée par sa moyenne azimuth → enlève le falloff
    mu = c.mean(axis=0, keepdims=True); mu[mu < 1e-6] = 1.0
    c = c / mu
    m = c.max(); return c / m if m > 0 else c

def ctx_v2_cfardens(img):         # densité de structure (seuil) — mean-pool binaire
    p = polar_remap(img).astype(np.float32)
    b = (p > THR).astype(np.float32)
    c = pool_bin(b, np.mean)
    m = c.max(); return c / m if m > 0 else c

def pool_bin(mat, fn):
    m = mat.T; As, Rs = m.shape
    ab, rb = max(As // A, 1), max(Rs // R, 1)
    m = m[:ab * A, :rb * R].reshape(A, ab, R, rb)
    return fn(m, axis=(1, 3)).astype(np.float32)

def ctx_v3_mean(img):             # mean-pool intensité brute
    c = pool(polar_remap(img).astype(np.float32), np.mean)
    m = c.max(); return c / m if m > 0 else c

def _dens(img, thr):
    b = (polar_remap(img).astype(np.float32) > thr).astype(np.float32)
    c = pool_bin(b, np.mean); m = c.max(); return c / m if m > 0 else c

def ctx_v2_thr50(img): return _dens(img, 50)
def ctx_v2_thr80(img): return _dens(img, 80)

def ctx_v5_dens_rangenorm(img):   # densité + normalisation par anneau de range
    b = (polar_remap(img).astype(np.float32) > THR).astype(np.float32)
    c = pool_bin(b, np.mean)
    mu = c.mean(axis=0, keepdims=True); mu[mu < 1e-6] = 1.0
    c = c / mu; m = c.max(); return c / m if m > 0 else c

def ctx_v6_intabovethr(img):      # densité pondérée par l'intensité au-dessus du seuil
    p = polar_remap(img).astype(np.float32)
    w = np.clip(p - THR, 0, None)
    c = pool(w, np.mean); m = c.max(); return c / m if m > 0 else c

def _intabove(img, thr):
    p = polar_remap(img).astype(np.float32)
    c = pool(np.clip(p - thr, 0, None), np.mean)
    m = c.max(); return c / m if m > 0 else c
def ctx_v6_thr95(img):  return _intabove(img, 95)
def ctx_v6_thr110(img): return _intabove(img, 110)
def ctx_v6_thr125(img): return _intabove(img, 125)

# descripteur RÉELLEMENT implémenté dans le module (validation du code livré)
from bruce_slam.sonar_context import build_sonar_context as _real_bsc
def ctx_module(img):
    return _real_bsc(polar_remap(img), A, R)

VARIANTS = {"v0_max(actuel)": ctx_v0_max, "v6_int_thr95": ctx_v6_thr95,
            "v6_int_thr110": ctx_v6_thr110, "MODULE_REEL": ctx_module}

# ---------- 5. distance cosinus shiftée (réplique sonar_context) ----------
def shift_pad(mat, sh, axis):
    if sh == 0: return mat
    out = np.zeros_like(mat)
    if axis == 0:
        if sh > 0: out[sh:, :] = mat[:-sh, :]
        else: out[:sh, :] = mat[-sh:, :]
    else:
        if sh > 0: out[:, sh:] = mat[:, :-sh]
        else: out[:, :sh] = mat[:, -sh:]
    return out

def cos_cols(q, c):
    nq = np.linalg.norm(q, axis=0); nc = np.linalg.norm(c, axis=0)
    dots = np.sum(q * c, axis=0); dist = np.ones(q.shape[1], np.float32)
    v = (nq > 1e-9) & (nc > 1e-9)
    dist[v] = 1.0 - dots[v] / (nq[v] * nc[v])
    return float(dist.mean())

def dist_shifted(q, c):
    best = np.inf
    for sa in range(-MAX_AZ, MAX_AZ + 1):
        ca = shift_pad(c, sa, 0)
        for sr in range(-MAX_RG, MAX_RG + 1):
            d = cos_cols(q, shift_pad(ca, sr, 1))
            if d < best: best = d
    return best

def auc(true_d, false_d):
    # P(distance vraie < distance fausse) : 1.0 = séparation parfaite, 0.5 = nul
    t = np.array(true_d)[:, None]; f = np.array(false_d)[None, :]
    return float((t < f).mean() + 0.5 * (t == f).mean())

# ---------- 6. évaluation ----------
print(f"\n{'variante':<18} {'méd. VRAIE':>11} {'méd. FAUSSE':>12} {'AUC':>6}  (AUC>0.5 = discrimine)")
for name, fn in VARIANTS.items():
    cache = {k: fn(imgs[k]) for k in imgs}
    degen = sum(1 for c in cache.values() if c.max() < 1e-6) / len(cache)
    td = np.array([dist_shifted(cache[s], cache[t]) for s, t in true_pairs])
    fd = np.array([dist_shifted(cache[s], cache[t]) for s, t in false_pairs])
    # seuil optimal (Youden J = TPR - FPR) sur une grille
    grid = np.linspace(min(td.min(), fd.min()), max(td.max(), fd.max()), 200)
    J = [(td < c).mean() - (fd < c).mean() for c in grid]
    cut = grid[int(np.argmax(J))]
    tpr = (td < cut).mean(); fpr = (fd < cut).mean()
    print(f"{name:<18} {np.median(td):>10.3f} {np.median(fd):>11.3f} {auc(td, fd):>6.3f}   "
          f"seuil*={cut:.3f} (TPR={tpr:.0%} FPR={fpr:.0%}) vides={degen*100:.0f}%")
