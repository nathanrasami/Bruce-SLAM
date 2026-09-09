#!/usr/bin/env python3
"""VERIFICATION OFFLINE de la FUSION DISO-local + cmd_vel-global (la bonne facon).

Le replacement par sous-carte echoue (aligner sur les keyframes ne contraint pas la
structure lointaine -> smearing). La bonne approche : POSE-GRAPH fusionnant
  - DISO relatif (entre keyframes consecutifs) = cap LOCAL propre (scan-matching dense),
  - cmd_vel position absolue (ancre par timestamp) = global GT-free (ATE 1.43m).
On optimise les poses, puis on rend le cloud DISO avec ces poses fusionnees.
Attendu : structure DISO propre (quai en T) MAIS recalee globalement sur cmd_vel (GT-free).

PROXY : DISO-120307 (prior GT) -> on n'utilise que ses poses RELATIVES (scan-matching local,
independant du prior). Si la fusion marche, le re-run DISO GT-free (prior cmd_vel) marchera.

Usage : python3 verify_fusion.py <diso_run> <cmdvel_run> [--w_anchor 0.3]
"""
import os, sys, argparse, bisect
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wrap(a): return np.arctan2(np.sin(a), np.cos(a))
def rigid_2d(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst-mu_d).T @ (src-mu_s); U, _, Vt = np.linalg.svd(S)
    R = U @ np.diag([1.0, np.sign(np.linalg.det(U@Vt))]) @ Vt
    return R, mu_d - R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("diso_run"); ap.add_argument("cmdvel_run")
    ap.add_argument("--w_anchor", type=float, default=0.3, help="poids ancre cmd_vel (vs DISO relatif=1)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--no_flip", action="store_true", help="ne PAS flip Y de DISO (par defaut on flip : DISO=repere reflechi)")
    ap.add_argument("--save", action="store_true", help="LIVRABLE : sauve cloudmap_fusion.csv + carte finale")
    args = ap.parse_args()

    dtraj = pd.read_csv(os.path.join(args.diso_run, "trajectory.csv"))
    dcloud = pd.read_csv(os.path.join(args.diso_run, "pointcloud.csv"))
    # DISO sort en repere Y REFLECHI (det(R)=-1, cf. usbl/flip_y=True du launch). Pour fusionner
    # avec cmd_vel (repere monde direct), on remet DISO dans la meme main : (x,y,th)->(x,-y,-th).
    if not args.no_flip:
        dtraj["y"] = -dtraj["y"]; dtraj["theta"] = -dtraj["theta"]
        dcloud["y"] = -dcloud["y"]
    ctraj = pd.read_csv(os.path.join(args.cmdvel_run, "trajectory.csv"))
    gt = sorted((t, x, y) for t, x, y in
                pd.read_csv(os.path.join(args.cmdvel_run, "groundtruth.csv")).itertuples(index=False))

    D = np.c_[dtraj.x, dtraj.y, dtraj.theta]          # poses DISO (relatif = propre)
    dk = dtraj.keyframe_id.to_numpy().astype(int)
    dtime = dtraj.time.to_numpy()
    N = len(dk); idx = {int(k): i for i, k in enumerate(dk)}

    # ancres cmd_vel (position) par timestamp DISO
    ct = ctraj.time.to_numpy(); cxy = np.c_[ctraj.x, ctraj.y]
    o = np.argsort(ct); ct, cxy = ct[o], cxy[o]
    def cmd_at(t):
        i = min(max(bisect.bisect_left(ct, t), 1), len(ct)-1)
        a = (t-ct[i-1])/(ct[i]-ct[i-1]) if ct[i] > ct[i-1] else 0
        return cxy[i-1]*(1-a) + cxy[i]*a
    anchor = np.array([cmd_at(t) for t in dtime])

    # contraintes relatives DISO (consecutives)
    ea = np.arange(N-1); eb = np.arange(1, N)
    pa, pb = D[ea], D[eb]; ca, sa = np.cos(pa[:,2]), np.sin(pa[:,2])
    dx, dy = pb[:,0]-pa[:,0], pb[:,1]-pa[:,1]
    ez = np.c_[ca*dx+sa*dy, -sa*dx+ca*dy, wrap(pb[:,2]-pa[:,2])]
    WROT = 3.0

    def solve():
        P = D.copy()
        wa = args.w_anchor
        for _ in range(args.iters):
            pa, pb = P[ea], P[eb]; ca, sa = np.cos(pa[:,2]), np.sin(pa[:,2])
            dx, dy = pb[:,0]-pa[:,0], pb[:,1]-pa[:,1]
            px = ca*dx+sa*dy; py = -sa*dx+ca*dy
            rx = px-ez[:,0]; ry = py-ez[:,1]; rth = wrap(wrap(pb[:,2]-pa[:,2])-ez[:,2])*WROT
            M = len(ea); ridx = np.arange(M)
            rows, cols, vals = [], [], []
            def add(r, c, v): rows.append(r); cols.append(c); vals.append(v)
            add(ridx*3+0, ea*3+0, -ca); add(ridx*3+0, ea*3+1, -sa); add(ridx*3+0, ea*3+2, py)
            add(ridx*3+0, eb*3+0, ca);  add(ridx*3+0, eb*3+1, sa)
            add(ridx*3+1, ea*3+0, sa);  add(ridx*3+1, ea*3+1, -ca); add(ridx*3+1, ea*3+2, -px)
            add(ridx*3+1, eb*3+0, -sa); add(ridx*3+1, eb*3+1, ca)
            add(ridx*3+2, ea*3+2, -np.ones(M)*WROT); add(ridx*3+2, eb*3+2, np.ones(M)*WROT)
            nr = M*3
            Rc = np.empty(M*3); Rc[0::3]=rx; Rc[1::3]=ry; Rc[2::3]=rth
            # ancre position cmd_vel
            ax_ = (P[:,0]-anchor[:,0])*wa; ay_ = (P[:,1]-anchor[:,1])*wa
            ai = np.arange(N)
            rows.append(nr+ai*2+0); cols.append(ai*3+0); vals.append(np.ones(N)*wa)
            rows.append(nr+ai*2+1); cols.append(ai*3+1); vals.append(np.ones(N)*wa)
            rfull = np.concatenate([Rc, np.empty(N*2)]); rfull[nr+ai*2+0]=ax_; rfull[nr+ai*2+1]=ay_
            rows = np.concatenate([np.atleast_1d(x) for x in rows])
            cols = np.concatenate([np.atleast_1d(x) for x in cols])
            vals = np.concatenate([np.atleast_1d(x) for x in vals])
            J = coo_matrix((vals,(rows,cols)), shape=(nr+N*2, N*3)).tocsr()
            H = (J.T@J) + 1e-6*coo_matrix((np.ones(N*3),(np.arange(N*3),np.arange(N*3))),shape=(N*3,N*3))
            d = spsolve(H.tocsc(), -(J.T@rfull)); P = P + d.reshape(N,3); P[:,2]=wrap(P[:,2])
        return P

    P = solve()

    # rendre le cloud DISO avec les poses fusionnees (body via pose DISO -> reproj via P)
    cid = dcloud.keyframe_id.to_numpy().astype(int)
    cwx, cwy = dcloud.x.to_numpy(), dcloud.y.to_numpy()
    nx, ny = np.empty(len(cwx)), np.empty(len(cwx))
    for k in np.unique(cid):
        if k not in idx: continue
        m = cid==k; ox, oy, oth = D[idx[k]]; c, s = np.cos(oth), np.sin(oth)
        bx = c*(cwx[m]-ox)+s*(cwy[m]-oy); by = -s*(cwx[m]-ox)+c*(cwy[m]-oy)
        nxk, nyk, nthk = P[idx[k]]; c2, s2 = np.cos(nthk), np.sin(nthk)
        nx[m] = c2*bx - s2*by + nxk; ny[m] = s2*bx + c2*by + nyk

    def ate(P):
        gts=[g[0] for g in gt]; src,dst=[],[]
        for (x,y),t in zip(P[:,:2], dtime):
            i=bisect.bisect_left(gts,t)
            if i<=0 or i>=len(gt): continue
            t0,x0,y0=gt[i-1]; t1,x1,y1=gt[i]; a=(t-t0)/(t1-t0) if t1>t0 else 0
            dst.append((x0+a*(x1-x0),y0+a*(y1-y0))); src.append((x,y))
        src,dst=np.array(src),np.array(dst); R,t=rigid_2d(src,dst)
        return np.linalg.norm((R@src.T).T+t-dst,axis=1).mean()

    print(f"ATE poses fusionnees vs GT : {ate(P):.2f} m  (DISO seul {ate(D):.2f}, ancre cmd_vel ~1.4)")
    print(f"deviation DISO->fusion : pos med={np.median(np.hypot(P[:,0]-D[:,0],P[:,1]-D[:,1])):.2f} m")

    # --- LIVRABLE B : sauver le cloud fusionne + la carte finale propre ---
    if args.save:
        cdf = pd.DataFrame({"keyframe_id": cid, "x": nx, "y": ny})
        cpath = os.path.join(args.cmdvel_run, "cloudmap_fusion.csv")
        cdf.to_csv(cpath, index=False); print(f"CLOUD  : {cpath} ({len(nx)} pts)")
        pdf = pd.DataFrame({"keyframe_id": dk, "time": dtime, "x": P[:,0], "y": P[:,1], "theta": P[:,2]})
        pdf.to_csv(os.path.join(args.cmdvel_run, "trajectory_fusion.csv"), index=False)
        figf, axf = plt.subplots(figsize=(11, 10))
        axf.scatter(nx, ny, s=0.4, c="navy", alpha=0.3)
        axf.plot(P[:,0], P[:,1], "-", color="orange", lw=0.8, alpha=0.7, label="trajectoire fusionnee")
        axf.legend(fontsize=9); axf.axis("equal"); axf.grid(alpha=0.3)
        axf.set_xlabel("x (m)"); axf.set_ylabel("y (m)")
        axf.set_title("Carte FUSION : structure DISO-local + position cmd_vel-global\n"
                      "(trajectoire GT-free, carte GT-assistee)")
        figf.tight_layout()
        fpath = os.path.join(args.cmdvel_run, "cloudmap_fusion_final.png")
        figf.savefig(fpath, dpi=140); print(f"CARTE  : {fpath}")

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    ax[0].scatter(dcloud.x, dcloud.y, s=0.3, c="navy", alpha=0.25)
    ax[0].set_title("DISO brut (cap GT) — quai en T net")
    ax[1].scatter(nx, ny, s=0.3, c="darkred", alpha=0.25)
    ax[1].plot(P[:,0], P[:,1], "-", color="black", lw=0.5, alpha=0.5)
    ax[1].set_title(f"FUSION DISO-local + cmd_vel-global (w_anchor={args.w_anchor}) — GT-free")
    for a in ax: a.axis("equal"); a.grid(alpha=0.3); a.set_xlabel("x"); a.set_ylabel("y")
    plt.tight_layout(); out = os.path.join(args.cmdvel_run, "verify_fusion.png")
    plt.savefig(out, dpi=120); print(f"PNG : {out}")


if __name__ == "__main__":
    main()
