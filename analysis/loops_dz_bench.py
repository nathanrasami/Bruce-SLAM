#!/usr/bin/env python3
"""Banc HORS-LIGNE du Δz des candidats loop closure (prépa 3D, PROGRESS 2026-07-14).

Question : le gate Δz (|z_src − z_cand| via /depth) tuerait-il des fausses loops
sans perdre de vraies ? Vérité terrain : dGT_xy = distance GT (interp temps KF)
entre les deux keyframes — vraie boucle si dGT_xy < 2 m (convention suite traj8).

Pour chaque run : dz_depth = z(source) − z(target) depuis trajectory.csv (même
source que le gate runtime : dr_pose3.z ← /depth) ; si loops_detected.csv a déjà
la colonne dz (runs post-instrumentation), vérif croisée runtime↔offline.

Usage : python3 analysis/loops_dz_bench.py results/<run> [results/<run2> ...]
"""
import sys
import numpy as np

GATES = [0.2, 0.5, 1.0, 2.0]  # seuils simulés (m)
D_TRUE = 2.0                  # dGT_xy < 2 m = vraie boucle (suite traj8)


def bench(run):
    tr = np.genfromtxt(f"{run}/trajectory.csv", delimiter=",", names=True)
    if "z" not in tr.dtype.names:
        print(f"{run}: trajectory.csv sans colonne z — run trop ancien, skip")
        return
    kf_t = {int(k): t for k, t in zip(tr["keyframe_id"], tr["time"])}
    kf_z = {int(k): z for k, z in zip(tr["keyframe_id"], tr["z"])}

    gt = np.genfromtxt(f"{run}/groundtruth.csv", delimiter=",", names=True)
    o = gt["time"].argsort()
    gtt = gt["time"][o]
    gxyz = [gt[c][o] for c in ("x", "y", "z")]
    gp = lambda k: np.array([np.interp(kf_t[k], gtt, g) for g in gxyz])

    lp = np.genfromtxt(f"{run}/loops_detected.csv", delimiter=",", names=True)
    lp = np.atleast_1d(lp)
    has_rt_dz = "dz" in lp.dtype.names

    rows = []  # (s, t, retenu, dz_depth, dGT_xy, dz_gt, dz_runtime|nan)
    for i, r in enumerate(lp):
        s, t = int(r["source_key"]), int(r["target_key"])
        if s not in kf_t or t not in kf_t:
            continue
        ps, pt = gp(s), gp(t)
        rows.append((s, t, int(r["retenu"]), kf_z[s] - kf_z[t],
                     float(np.linalg.norm(ps[:2] - pt[:2])), ps[2] - pt[2],
                     float(r["dz"]) if has_rt_dz else np.nan))
    rows = np.array(rows)
    ret = rows[rows[:, 2] == 1]  # loops retenues (envoyées à l'aval)
    vrai = ret[ret[:, 4] < D_TRUE]
    faux = ret[ret[:, 4] >= D_TRUE]

    print(f"\n=== {run}")
    print(f"candidats journalisés {len(rows)} · retenus {len(ret)} "
          f"= {len(vrai)} vrais (dGT<{D_TRUE}) + {len(faux)} faux")
    for name, grp in (("vrais", vrai), ("faux ", faux)):
        if len(grp) == 0:
            print(f"  {name}: —")
            continue
        a = np.abs(grp[:, 3])
        print(f"  {name}: |dz_depth| méd {np.median(a):.3f} p90 "
              f"{np.percentile(a, 90):.3f} max {a.max():.3f} m "
              f"(|dz_GT| méd {np.median(np.abs(grp[:, 5])):.3f})")
    for g in GATES:
        tk = int((np.abs(vrai[:, 3]) > g).sum()) if len(vrai) else 0
        fk = int((np.abs(faux[:, 3]) > g).sum()) if len(faux) else 0
        print(f"  gate {g:.1f} m : tue {fk}/{len(faux)} faux, "
              f"perd {tk}/{len(vrai)} vrais")
    if len(faux):
        print("  faux retenus (s, t, dz_depth, dGT_xy) :")
        for s, t, _, dz, d, *_ in faux:
            print(f"    {int(s):4d} ↔ {int(t):4d}  dz {dz:+.3f}  dGT {d:.2f}")
    if has_rt_dz:
        err = np.abs(rows[:, 6] - rows[:, 3])
        ok = np.nanmax(err) < 0.005  # colonne runtime arrondie à 3 déc.
        print(f"  vérif croisée dz runtime↔offline : max écart "
              f"{np.nanmax(err):.4f} m → {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    runs = sys.argv[1:]
    if not runs:
        sys.exit("usage : loops_dz_bench.py results/<run> [...]")
    for r in runs:
        bench(r)
