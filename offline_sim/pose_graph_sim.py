#!/usr/bin/env python3
"""Solveur de pose-graph 2D OFFLINE pour sweeper les paramètres USBL sans relancer
ROS (50 min/run). Rejoue le graphe gtsam : between-factors DISO (odométrie) +
priors de position USBL (position-only, Cauchy robuste). Gauss-Newton SE(2).

Données :
  - odométrie DISO : colonnes dr_x,dr_y,dr_theta d'un trajectory.csv (dead reckoning)
  - fixes USBL     : offline_sim/usbl_fixes.csv (extrait du bag)
  - GT             : groundtruth.csv du run (pour ATE)

Validé contre les runs réels (DISO+USBL σ=1.4 → Umeyama ~13.87 m).
"""
import csv, math, sys
import numpy as np

# ---------- IO ----------
def load_csv(path, cols):
    out = []
    with open(path) as h:
        for r in csv.DictReader(h):
            out.append([float(r[c]) for c in cols])
    return np.array(out)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# ---------- Umeyama 2D (réplique traj_eval.py : réflexion autorisée) ----------
def umeyama(src, dst, with_scale=False, allow_reflection=True):
    """allow_reflection=True : absorbe la réflexion (axe Y inversé de DISO)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / len(src)
    U, Dg, Vt = np.linalg.svd(cov)
    W = np.eye(2)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        W[-1, -1] = -1
    R = U @ W @ Vt
    scale = (Dg * np.diag(W)).sum() / (S ** 2).sum() * len(src) if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t

def ate(traj_xy, traj_t, gt_xy, gt_t, align=False):
    # associe chaque pose au GT le plus proche en temps
    idx = np.searchsorted(gt_t, traj_t)
    idx = np.clip(idx, 1, len(gt_t) - 1)
    left = np.abs(gt_t[idx - 1] - traj_t) < np.abs(gt_t[idx] - traj_t)
    idx = idx - left.astype(int)
    g = gt_xy[idx]
    p = traj_xy
    if align:
        s, R, t = umeyama(p, g)
        p = (s * (R @ p.T).T + t)
    return math.sqrt(((p - g) ** 2).sum(1).mean())

# ---------- Pose-graph Gauss-Newton ----------
class PoseGraph:
    def __init__(self, odom, usbl, prior_sig, odom_sig, usbl_sig,
                 max_dt=1.0, cauchy_c=None):
        """odom: (N,4) [t, x, y, theta] dead-reckoning DISO (poses absolues init)
           usbl: (M,3) [t, x, y]"""
        self.t = odom[:, 0]
        self.X0 = odom[:, 1:4].copy()          # init = DISO
        self.N = len(self.X0)
        self.prior_sig = np.asarray(prior_sig)
        self.odom_sig = np.asarray(odom_sig)
        self.usbl_sig = usbl_sig
        self.cauchy_c = cauchy_c
        # between-factors DISO : transfo relative i->i+1 dans le repère i
        self.betw = []
        for i in range(self.N - 1):
            ti, tj = self.X0[i], self.X0[i + 1]
            c, s = math.cos(ti[2]), math.sin(ti[2])
            Rt = np.array([[c, s], [-s, c]])
            dt = Rt @ (tj[:2] - ti[:2])
            dth = wrap(tj[2] - ti[2])
            self.betw.append((i, i + 1, np.array([dt[0], dt[1], dth])))
        # associe chaque fix USBL au keyframe le plus proche en temps (< max_dt)
        self.usbl_factors = []
        for (tu, ux, uy) in usbl:
            k = int(np.argmin(np.abs(self.t - tu)))
            if abs(self.t[k] - tu) <= max_dt:
                self.usbl_factors.append((k, ux, uy))

    def optimize(self, iters=15):
        X = self.X0.copy()
        for it in range(iters):
            rows, cols, vals = [], [], []
            res = []
            ridx = 0
            def add(r, ci, vi):
                nonlocal ridx
                for j, c in enumerate(ci):
                    rows.append(ridx); cols.append(c); vals.append(vi[j])
                res.append(r)
                ridx += 1
            # prior sur x0 (ancre) = DISO initial
            wp = 1.0 / self.prior_sig
            for d in range(3):
                add((X[0, d] - self.X0[0, d]) * wp[d], [0 * 3 + d], [wp[d]])
            # between-factors
            wo = 1.0 / self.odom_sig
            for (i, j, m) in self.betw:
                ti, tj = X[i], X[j]
                c, s = math.cos(ti[2]), math.sin(ti[2])
                Rt = np.array([[c, s], [-s, c]])
                dpred = Rt @ (tj[:2] - ti[:2])
                dRt = np.array([[-s, c], [-c, -s]])
                ddth = dRt @ (tj[:2] - ti[:2])
                # x,y résidus
                ex = (dpred[0] - m[0])
                ey = (dpred[1] - m[1])
                eth = wrap((tj[2] - ti[2]) - m[2])
                # jac x: d/dti(x,y), d/dthi, d/dtj(x,y)
                bi, bj = i * 3, j * 3
                add(ex * wo[0], [bi, bi + 1, bi + 2, bj, bj + 1],
                    [-Rt[0, 0] * wo[0], -Rt[0, 1] * wo[0], ddth[0] * wo[0], Rt[0, 0] * wo[0], Rt[0, 1] * wo[0]])
                add(ey * wo[1], [bi, bi + 1, bi + 2, bj, bj + 1],
                    [-Rt[1, 0] * wo[1], -Rt[1, 1] * wo[1], ddth[1] * wo[1], Rt[1, 0] * wo[1], Rt[1, 1] * wo[1]])
                add(eth * wo[2], [bi + 2, bj + 2], [-wo[2], wo[2]])
            # USBL position priors (Cauchy IRLS)
            wu = 1.0 / self.usbl_sig
            for (k, ux, uy) in self.usbl_factors:
                ex = X[k, 0] - ux
                ey = X[k, 1] - uy
                w = wu
                if self.cauchy_c:
                    rn = math.hypot(ex, ey) * wu
                    w = wu / math.sqrt(1.0 + (rn / self.cauchy_c) ** 2)
                bk = k * 3
                add(ex * w, [bk], [w])
                add(ey * w, [bk + 1], [w])
            # résout (J^T J) dx = -J^T r  (sparse)
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import spsolve
            J = csr_matrix((vals, (rows, cols)), shape=(ridx, self.N * 3))
            r = np.array(res)
            H = (J.T @ J).tocsc()
            b = -(J.T @ r)
            H = H + 1e-9 * np.eye(1)[0, 0] * csr_matrix(np.eye(self.N * 3)) if False else H
            dx = spsolve(H, b)
            X += dx.reshape(self.N, 3)
            X[:, 2] = wrap(X[:, 2])
            if np.linalg.norm(dx) < 1e-4:
                break
        return X

# ---------- main ----------
def reflect_to_gt(od, gt):
    """Réfléchit/aligne l'odométrie DISO dans le repère GT (corrige l'axe Y inversé
    de DISO) AVANT d'ajouter l'USBL → met DISO et USBL dans le MÊME handedness.
    Simule la correction de convention de repère qu'il faudrait faire dans le pipeline."""
    gx = np.interp(od[:, 0], gt[:, 0], gt[:, 1])
    gy = np.interp(od[:, 0], gt[:, 0], gt[:, 2])
    s, R, t = umeyama(od[:, 1:3], np.c_[gx, gy], with_scale=False, allow_reflection=True)
    xy = (s * (R @ od[:, 1:3].T).T) + t
    # cap transformé par R (réflexion incluse)
    dirs = np.c_[np.cos(od[:, 3]), np.sin(od[:, 3])]
    dirs2 = (R @ dirs.T).T
    th = np.arctan2(dirs2[:, 1], dirs2[:, 0])
    out = od.copy()
    out[:, 1], out[:, 2], out[:, 3] = xy[:, 0], xy[:, 1], th
    return out


def flip_y(od):
    """Flip-Y STRUCTUREL, sans GT : y→-y, θ→-θ. Corrige la convention de repère
    réfléchi de DISO (det=-1 universel) → compatible avec l'USBL (repère monde).
    Le résidu de cap (quelques °) est laissé à l'optimisation."""
    out = od.copy()
    out[:, 2] = -od[:, 2]
    out[:, 3] = -od[:, 3]
    return out


def run(diso_csv, gt_csv, usbl_csv, usbl_sig, cauchy_c=0.1,
        prior_sig=(0.1, 0.1, 0.01), odom_sig=(0.5, 0.5, 0.05), max_dt=1.0,
        reflect=False, flip=False):
    od = load_csv(diso_csv, ['time', 'dr_x', 'dr_y', 'dr_theta'])
    us = load_csv(usbl_csv, ['time', 'x', 'y'])
    gt = load_csv(gt_csv, ['time', 'x', 'y'])
    if flip:
        od = flip_y(od)
    if reflect:
        od = reflect_to_gt(od, gt)
    pg = PoseGraph(od, us, prior_sig, odom_sig, usbl_sig, max_dt, cauchy_c)
    X = pg.optimize()
    tt = od[:, 0]
    a_raw = ate(X[:, :2], tt, gt[:, 1:], gt[:, 0], align=False)
    a_um = ate(X[:, :2], tt, gt[:, 1:], gt[:, 0], align=True)
    # baseline DISO seul (sans USBL)
    d_raw = ate(od[:, 1:3], tt, gt[:, 1:], gt[:, 0], align=False)
    d_um = ate(od[:, 1:3], tt, gt[:, 1:], gt[:, 0], align=True)
    return dict(n_usbl=len(pg.usbl_factors), raw=a_raw, um=a_um, diso_raw=d_raw, diso_um=d_um)

if __name__ == '__main__':
    # base = run DISO PROPRE (161831, um 3.16) — pas le 225022 (DISO dégradé sous contention)
    base = sys.argv[1] if len(sys.argv) > 1 else '../results/run_aracati_Bruce_Sonar_USBL_2026-06-18_161831'
    diso = base + '/trajectory.csv'
    gt = base + '/groundtruth.csv'
    usbl = 'usbl_fixes.csv'
    print(f"base DISO odom : {base.split('/')[-1]}")
    r0 = run(diso, gt, usbl, 1.4, cauchy_c=None)
    print(f"fixes USBL associés : {r0['n_usbl']}   DISO seul (USBL off) : brut={r0['diso_raw']:.2f} um={r0['diso_um']:.2f}  (attendu um≈3.16)")
    print()
    print("=== A) USBL sur DISO BRUT (repère réfléchi — incompatible) ===")
    print(f"{'σ_usbl':>8} {'kernel':>8} {'ATE brut':>9} {'ATE Umey':>9}")
    for cauchy, lab in [(None, 'gauss'), (1.0, 'cauchy')]:
        for sig in [1.0, 1.4, 5.0]:
            r = run(diso, gt, usbl, sig, cauchy_c=cauchy)
            print(f"{sig:8.2f} {lab:>8} {r['raw']:9.2f} {r['um']:9.2f}")
    print()
    print("=== B) USBL sur DISO RÉFLÉCHI en repère GT (handedness corrigé, via GT) ===")
    print(f"{'σ_usbl':>8} {'kernel':>8} {'ATE brut':>9} {'ATE Umey':>9}")
    for cauchy, lab in [(1.0, 'cauchy')]:
        for sig in [1.4, 3.0, 5.0]:
            r = run(diso, gt, usbl, sig, cauchy_c=cauchy, reflect=True)
            print(f"{sig:8.2f} {lab:>8} {r['raw']:9.2f} {r['um']:9.2f}")
    print()
    print("=== C) USBL sur DISO FLIP-Y STRUCTUREL (y→-y, θ→-θ, SANS GT) = fix pipeline réel ===")
    print(f"{'σ_usbl':>8} {'kernel':>8} {'ATE brut':>9} {'ATE Umey':>9}")
    for cauchy, lab in [(None, 'gauss'), (1.0, 'cauchy')]:
        for sig in [1.0, 1.4, 3.0, 5.0]:
            r = run(diso, gt, usbl, sig, cauchy_c=cauchy, flip=True)
            print(f"{sig:8.2f} {lab:>8} {r['raw']:9.2f} {r['um']:9.2f}")
