#!/usr/bin/env python3
"""Rendu « cap compas » du nuage d'un run — 100 % GT-free (piste U1, branche ultime).

Principe (validé offline 07-04 sur le champion 1.2a) : le theta iSAM2 est optimisé pour
la POSITION (USBL + boucles) → il transitoire fort dans les virages (jusqu'à 72° d'écart
au compas) et smear les scans. Le cap compas (dr_theta, intégrale de /cmd_vel, capteur
du bord) est lisse et vrai. On re-rend donc chaque scan avec :
    position = pose optimisée (INCHANGÉE — l'ATE ne bouge pas)
    cap      = dr_theta + δ,   δ = moyenne circulaire de (θ_opt − dr_theta)
δ recale le repère odométrique sur le repère carte SANS aucune GT (fit interne au run).

Chiffres 1.2a (003823, I≥255) : NN auto 0.204→0.176 ; carte vs vraie méd 0.114→0.077,
p90 0.989→0.441 = la borne du rendu au cap GT (0.077/0.440). cf. ULTIME.md U1.

Usage : python3 analysis/render_compass_cloud.py results/run_aracati_XXX [--imin 255] [--eval]
Écrit : pointcloud_compass.csv + pointcloud_compass.png dans le dossier du run.
--eval ajoute les métriques vs cloud-GT (utilise la GT : évaluation seulement).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_eval import charger, fit_cap, cloud_vrai, nn_auto, wrap
from traj_eval import umeyama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--imin", type=float, default=None)
    ap.add_argument("--eval", action="store_true", help="métriques vs cloud-GT (éval seulement)")
    a = ap.parse_args()
    d = charger(a.run, a.imin)

    idx_of = {k: i for i, k in enumerate(d["kf"])}
    ok = np.array([c in idx_of for c in d["cloud_kf"]])
    P = d["cloud"][ok]
    ki = np.array([idx_of[c] for c in d["cloud_kf"][ok]])

    # scans → repère local (poses optimisées), puis re-rendu au cap compas recalé
    ca, sa = np.cos(d["th"][ki]), np.sin(d["th"][ki])
    dx, dy = P[:, 0] - d["est"][ki, 0], P[:, 1] - d["est"][ki, 1]
    lx, ly = ca * dx + sa * dy, -sa * dx + ca * dy
    dlt = wrap(d["th"] - d["dr_th"])
    delta = np.arctan2(np.mean(np.sin(dlt)), np.mean(np.cos(dlt)))
    th_r = d["dr_th"] + delta
    cb, sb = np.cos(th_r[ki]), np.sin(th_r[ki])
    C = np.column_stack([cb * lx - sb * ly + d["est"][ki, 0],
                         sb * lx + cb * ly + d["est"][ki, 1]])

    print(f"δ (odom→carte) = {np.degrees(delta):.2f}° ; "
          f"|θopt−compas| méd {np.degrees(np.median(np.abs(wrap(dlt - delta)))):.2f}° "
          f"max {np.degrees(np.max(np.abs(wrap(dlt - delta)))):.1f}°")
    print(f"NN auto : {nn_auto(P):.3f} → {nn_auto(C):.3f}")

    if a.eval:
        s1, R1, t1 = umeyama(d["est"], d["gxy"], with_scale=False)
        _, s_cap, beta = fit_cap(d["g_th"], d["th"])
        _, V = cloud_vrai(d, R1, t1, s_cap, beta)
        tree = cKDTree(V)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(P), min(60000, len(P)), replace=False)
        for name, X in (("theta opt", P), ("cap compas", C)):
            dm, _ = tree.query(X[idx], k=1)
            print(f"carte vs vraie ({name:10s}) : méd {np.median(dm):.3f} p90 {np.percentile(dm, 90):.3f}")

    out_csv = os.path.join(a.run, "pointcloud_compass.csv")
    np.savetxt(out_csv, np.column_stack([d["cloud_kf"][ok], C]), delimiter=",",
               header="keyframe_id,x,y", comments="", fmt=["%d", "%.6f", "%.6f"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax, (X, t) in zip(axes, [(P, f"rendu θ optimisé (NN {nn_auto(P):.3f})"),
                                 (C, f"rendu cap compas (NN {nn_auto(C):.3f})")]):
        ax.scatter(X[:, 0], X[:, 1], s=0.15, c="k", alpha=0.4, linewidths=0)
        ax.set_title(t)
        ax.set_aspect("equal")
    fig.suptitle(f"{os.path.basename(a.run)} — rendu cap compas (GT-free, poses inchangées)")
    fig.tight_layout()
    out_png = os.path.join(a.run, "pointcloud_compass.png")
    fig.savefig(out_png, dpi=140)
    print(f"-> {out_csv}\n-> {out_png}")


if __name__ == "__main__":
    main()
