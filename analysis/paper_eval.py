#!/usr/bin/env python3
"""Évaluation « façon papier » d'un run Aracati — protocole comparable DISO/ISOPoT.

Métriques calculées (cf. mini-papier, chapitre Expérimentations) :
  - ATE Umeyama SE(2) (s=1)      : notre métrique historique (alignement rigide optimal).
  - ATE Umeyama sim(2) (échelle) : pour montrer l'effet du rescale (à NE PAS utiliser
                                   comme métrique : le sonar est métrique, l'échelle
                                   libre "triche" en écrasant la trajectoire sur la GT).
  - ATE première-pose            : protocole ISOPoT (alignée au départ seulement →
                                   reflète la dérive long-horizon, pas de moyennage).
  - Sections S1/S2/S3            : tiers temporels de la mission (~15 min chacun,
                                   analogues aux 3 séquences de DISO sur Aracati2017),
                                   ATE première-pose par section (ré-ancrée au début
                                   de chaque section).
  - RE (Relative Error)          : erreur relative sur segments de 10 % de la longueur
                                   de piste (ISOPoT) : translation en % et rotation
                                   en °/m — comparable aux tables DISO/ISOPoT.
  - Cap : fit circulaire s·θ+β vs compas GT (méthodo analyze_heading), méd/RMS.
  - Cloud : NN médian auto (netteté, 8000 pts) + NOUVEAU : distance de notre cloud
    au « cloud vrai » (mêmes scans re-rendus aux poses GT) — médiane et p90.

Figures écrites dans --out (défaut : dossier du run) :
  <label>_traj.png / <label>_err_time.png / <label>_cloud_vs_gt.png

Usage :
  python3 analysis/paper_eval.py results/run_aracati_XXX --label new12a --out Paper/MiniPapier/figs
  python3 analysis/paper_eval.py --compare rundir:label rundir:label ... --out Paper/MiniPapier/figs
Options : --imin 255 (filtre intensité du cloud si colonne présente)
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
from traj_eval import umeyama, appliquer, calculer_ate, associer_par_temps


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def charger(run_dir, imin=None):
    ld = lambda f: np.genfromtxt(os.path.join(run_dir, f), delimiter=",", names=True)
    traj, gt = ld("trajectory.csv"), ld("groundtruth.csv")
    cloud = ld("pointcloud.csv")
    if imin is not None and "intensity" in cloud.dtype.names:
        cloud = cloud[cloud["intensity"] >= imin]
    d = {
        "t": traj["time"], "est": np.column_stack([traj["x"], traj["y"]]),
        "th": traj["theta"],
        "dr": np.column_stack([traj["dr_x"], traj["dr_y"]]), "dr_th": traj["dr_theta"],
        "gxy": associer_par_temps(traj["time"], gt["time"], gt["x"], gt["y"]),
        "g_th": np.interp(traj["time"], gt["time"], np.unwrap(gt["theta"])),
        "cloud": np.column_stack([cloud["x"], cloud["y"]]),
        "cloud_kf": cloud["keyframe_id"].astype(int),
        "kf": traj["keyframe_id"].astype(int),
    }
    return d


def fit_cap(g_th, th):
    """Fit circulaire g_th ≈ s·th + β (s=±1). Retourne (resid, s, beta)."""
    best = None
    for sc in (1.0, -1.0):
        dd = wrap(g_th - sc * th)
        beta = np.arctan2(np.mean(np.sin(dd)), np.mean(np.cos(dd)))
        rr = wrap(dd - beta)
        if best is None or np.median(np.abs(rr)) < np.median(np.abs(best[0])):
            best = (rr, sc, beta)
    return best


def align_premiere_pose(est, gxy, d_chord=15.0):
    """Alignement ISOPoT : ancre le DÉPART seulement. Translation = superpose p0 ;
    rotation = aligne la corde des premiers `d_chord` mètres de piste GT (l'orientation
    DGPS n'existe pas ; la corde initiale est la version robuste du « cap au départ »).
    """
    s_gt = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gxy, axis=0), axis=1))])
    j = int(np.searchsorted(s_gt, min(d_chord, s_gt[-1] * 0.5)))
    j = max(j, 1)
    v_e, v_g = est[j] - est[0], gxy[j] - gxy[0]
    a = np.arctan2(v_g[1], v_g[0]) - np.arctan2(v_e[1], v_e[0])
    return rot(a) @ (est - est[0]).T
    # (transposé appliqué par l'appelant)


def ate_premiere_pose(est, gxy, d_chord=15.0):
    E = align_premiere_pose(est, gxy, d_chord).T + gxy[0]
    return float(np.sqrt(np.mean(np.sum((E - gxy) ** 2, axis=1)))), E


def erreur_relative(est, th_est, gxy, th_gt, frac=0.10):
    """RE type ISOPoT/KITTI : pour chaque départ i, segment de `frac` de la longueur
    de piste GT ; erreur du déplacement relatif exprimé dans le repère de la pose i.
    Retourne (trans %, rot °/m) moyennés sur tous les segments.
    """
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gxy, axis=0), axis=1))])
    L = frac * s[-1]
    et, er, n = 0.0, 0.0, 0
    for i in range(len(est)):
        j = int(np.searchsorted(s, s[i] + L))
        if j >= len(est):
            break
        de = rot(-th_est[i]) @ (est[j] - est[i])
        dg = rot(-th_gt[i]) @ (gxy[j] - gxy[i])
        seg = s[j] - s[i]
        if seg < 1e-6:
            continue
        et += np.linalg.norm(de - dg) / seg
        er += abs(wrap((th_est[j] - th_est[i]) - (th_gt[j] - th_gt[i]))) / seg
        n += 1
    if n == 0:
        return np.nan, np.nan
    return 100.0 * et / n, np.degrees(er / n)


def cloud_vrai(d, R_um, t_um, s_cap, beta):
    """Re-rend les scans du run aux poses GT → le « cloud vrai » (dans le repère map).
    Pose GT en repère map : position = R⁻¹(g − t) (Umeyama inverse, s=1) ;
    cap = s·(gθ − β) (fit circulaire inversé). Si det(R)<0 (repère réfléchi, ex. DISO),
    le y local de chaque scan est aussi réfléchi.
    """
    idx_of = {k: i for i, k in enumerate(d["kf"])}
    ok = np.array([c in idx_of for c in d["cloud_kf"]])
    P, ki = d["cloud"][ok], np.array([idx_of[c] for c in d["cloud_kf"][ok]])
    # cloud → repère local de chaque keyframe (poses SLAM)
    ca, sa = np.cos(d["th"][ki]), np.sin(d["th"][ki])
    dx, dy = P[:, 0] - d["est"][ki, 0], P[:, 1] - d["est"][ki, 1]
    lx, ly = ca * dx + sa * dy, -sa * dx + ca * dy
    if np.linalg.det(R_um) < 0:
        ly = -ly
    # poses GT en repère map
    g_map = (np.linalg.inv(R_um) @ (d["gxy"] - t_um).T).T
    th_v = s_cap * (d["g_th"] - beta)
    cb, sb = np.cos(th_v[ki]), np.sin(th_v[ki])
    V = np.column_stack([cb * lx - sb * ly + g_map[ki, 0],
                         sb * lx + cb * ly + g_map[ki, 1]])
    return P, V


def nn_auto(P, n=8000, seed=0):
    rng = np.random.default_rng(seed)
    S = P[rng.choice(len(P), min(n, len(P)), replace=False)]
    dd, _ = cKDTree(S).query(S, k=2)
    return float(np.median(dd[:, 1]))


def eval_run(run_dir, label, out, imin=None, n_sections=3):
    os.makedirs(out, exist_ok=True)
    d = charger(run_dir, imin)
    t_rel = (d["t"] - d["t"][0]) / 60.0

    # --- alignements globaux
    s1, R1, t1 = umeyama(d["est"], d["gxy"], with_scale=False)
    est_a = appliquer(s1, R1, t1, d["est"])
    ate_um = calculer_ate(est_a, d["gxy"])
    s2, R2, t2 = umeyama(d["est"], d["gxy"], with_scale=True)
    ate_sc = calculer_ate(appliquer(s2, R2, t2, d["est"]), d["gxy"])
    so, Ro, to = umeyama(d["dr"], d["gxy"], with_scale=False)
    dr_a = appliquer(so, Ro, to, d["dr"])
    ate_odom = calculer_ate(dr_a, d["gxy"])

    # Les métriques SANS alignement Umeyama (première-pose, RE) supposent un repère
    # PROPRE : si l'Umeyama détecte une réflexion (det<0, ex. DISO), on dé-miroite
    # la trajectoire (y→−y, θ→−θ) AVANT de les calculer.
    def _propre(xy, th, R):
        if np.linalg.det(R) < 0:
            return np.column_stack([xy[:, 0], -xy[:, 1]]), -th
        return xy, th
    est_p, th_p = _propre(d["est"], d["th"], R1)
    dr_p, dr_th_p = _propre(d["dr"], d["dr_th"], Ro)
    ate_fp, est_fp = ate_premiere_pose(est_p, d["gxy"])
    ate_odom_fp, dr_fp = ate_premiere_pose(dr_p, d["gxy"])

    # --- cap
    resid, s_cap, beta = fit_cap(d["g_th"], d["th"])
    cap_med = np.degrees(np.median(np.abs(resid)))
    cap_rms = np.degrees(np.sqrt(np.mean(resid ** 2)))
    resid_o, s_o, b_o = fit_cap(d["g_th"], d["dr_th"])
    # cap GT dans la convention de la trajectoire dé-miroitée (pour la RE)
    residP, s_capP, betaP = fit_cap(d["g_th"], th_p)
    th_gt_conv = s_capP * (d["g_th"] - betaP)
    residO, s_oP, b_oP = fit_cap(d["g_th"], dr_th_p)

    # --- RE global + sections
    re_t, re_r = erreur_relative(est_p, th_p, d["gxy"], th_gt_conv)
    re_t_o, re_r_o = erreur_relative(dr_p, dr_th_p, d["gxy"], s_oP * (d["g_th"] - b_oP))
    tb = np.linspace(d["t"][0], d["t"][-1], n_sections + 1)
    sections = []
    for k in range(n_sections):
        m = (d["t"] >= tb[k]) & (d["t"] <= tb[k + 1])
        if m.sum() < 10:
            sections.append((np.nan,) * 4)
            continue
        a_fp, _ = ate_premiere_pose(est_p[m], d["gxy"][m])
        su, Ru, tu = umeyama(d["est"][m], d["gxy"][m], with_scale=False)
        a_um = calculer_ate(appliquer(su, Ru, tu, d["est"][m]), d["gxy"][m])
        rt, rr = erreur_relative(est_p[m], th_p[m], d["gxy"][m], th_gt_conv[m])
        sections.append((a_fp, a_um, rt, rr))

    # --- cloud vs cloud vrai
    nn_self = nn_auto(d["cloud"])
    P, V = cloud_vrai(d, R1, t1, s_cap, beta)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(P), min(60000, len(P)), replace=False)
    dmap, _ = cKDTree(V).query(P[idx], k=1)
    map_med, map_p90 = float(np.median(dmap)), float(np.percentile(dmap, 90))

    # ---------- rapport console ----------
    print(f"\n===== {label} — {os.path.basename(run_dir)} (KF={len(d['t'])}, det(R)={np.linalg.det(R1):+.0f})")
    print(f"ATE  : Umeyama SE(2) {ate_um:.2f} m | premiere-pose {ate_fp:.2f} m | "
          f"sim(2) echelle {ate_sc:.2f} m (s={s2:.4f})")
    print(f"Odom : Umeyama {ate_odom:.2f} m | premiere-pose {ate_odom_fp:.2f} m | "
          f"RE {re_t_o:.2f} % / {re_r_o:.3f} deg/m")
    print(f"RE   : translation {re_t:.2f} % | rotation {re_r:.3f} deg/m (segments 10 %)")
    for k, (a_fp, a_um, rt, rr) in enumerate(sections):
        print(f"S{k+1}   : ATE fp {a_fp:.2f} m | ATE um {a_um:.2f} m | RE {rt:.2f} % / {rr:.3f} deg/m")
    print(f"Cap  : med {cap_med:.1f} deg RMS {cap_rms:.1f} deg (s={s_cap:+.0f}, beta={np.degrees(beta):.1f} deg)")
    print(f"Cloud: NN auto {nn_self:.3f} m | vs cloud-GT med {map_med:.2f} m p90 {map_p90:.2f} m "
          f"({len(P)} pts)")

    # ---------- figures ----------
    # 1) trajectoires
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot(d["gxy"][:, 0], d["gxy"][:, 1], color="0.25", lw=1.6, label="GT (DGPS)")
    ax.plot(dr_a[:, 0], dr_a[:, 1], color="tab:orange", lw=1.0, ls="--",
            label=f"odométrie pure ({ate_odom:.2f} m)")
    ax.plot(est_a[:, 0], est_a[:, 1], color="tab:blue", lw=1.3,
            label=f"{label} ({ate_um:.2f} m)")
    ax.plot(*d["gxy"][0], "k^", ms=8, label="départ")
    for k in range(1, n_sections):
        i = int(np.searchsorted(d["t"], tb[k]))
        ax.plot(*d["gxy"][i], "o", color="0.25", ms=5)
        ax.annotate(f"S{k}|S{k+1}", d["gxy"][i], fontsize=8, xytext=(4, 4),
                    textcoords="offset points", color="0.25")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal"); ax.legend(fontsize=9, loc="best")
    ax.set_title(f"{label} — trajectoires (alignement Umeyama SE(2), ATE dans la légende)")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{label}_traj.png"), dpi=150)
    plt.close(fig)

    # 2) erreurs dans le temps (position + cap), bornes de sections.
    # Deux conventions (cf. mini-papier §6.2) : plein = résidu après alignement
    # Umeyama pleine séquence ; pointillé = trajectoire ANCRÉE AU DÉPART (dérive
    # cumulée, convention des papiers type Kim Fig. 8) — mêmes données.
    err_pos = np.linalg.norm(est_a - d["gxy"], axis=1)
    err_odo = np.linalg.norm(dr_a - d["gxy"], axis=1)
    err_pos_fp = np.linalg.norm(est_fp - d["gxy"], axis=1)
    err_odo_fp = np.linalg.norm(dr_fp - d["gxy"], axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    axes[0].plot(t_rel, err_odo_fp, color="tab:orange", lw=0.8, ls=":",
                 label="odométrie ancrée départ")
    axes[0].plot(t_rel, err_odo, color="tab:orange", lw=0.9, ls="--", label="odométrie (Umeyama)")
    axes[0].plot(t_rel, err_pos_fp, color="tab:blue", lw=0.8, ls=":",
                 label=f"{label} ancré départ")
    axes[0].plot(t_rel, err_pos, color="tab:blue", lw=1.1, label=f"{label} (Umeyama)")
    axes[0].set_ylabel("erreur position (m)"); axes[0].legend(fontsize=8)
    axes[1].plot(t_rel, np.degrees(np.abs(resid_o)), color="tab:orange", lw=0.9, ls="--",
                 label="odométrie pure")
    axes[1].plot(t_rel, np.degrees(np.abs(resid)), color="tab:blue", lw=1.1, label=label)
    axes[1].set_ylabel("erreur cap (°)"); axes[1].set_xlabel("temps mission (min)")
    axes[1].legend(fontsize=9)
    for ax in axes:
        for k in range(1, n_sections):
            ax.axvline((tb[k] - d["t"][0]) / 60.0, color="0.6", lw=0.8, ls=":")
        ax.grid(alpha=0.25)
    axes[0].set_title(f"{label} — erreurs au cours du temps (S1 | S2 | S3)")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{label}_err_time.png"), dpi=150)
    plt.close(fig)

    # 3) cloud estimé vs cloud vrai (re-rendu poses GT)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0))
    axes[0].scatter(V[:, 0], V[:, 1], s=0.15, c="tab:green", alpha=0.35, linewidths=0,
                    label="cloud « vrai » (scans aux poses GT)")
    axes[0].scatter(P[:, 0], P[:, 1], s=0.15, c="k", alpha=0.35, linewidths=0,
                    label=f"cloud {label}")
    axes[0].legend(fontsize=9, markerscale=30)
    axes[0].set_title("superposition"); axes[0].set_aspect("equal")
    sc = axes[1].scatter(P[idx, 0], P[idx, 1], s=0.6, c=dmap, cmap="magma_r",
                         vmin=0, vmax=max(1.0, map_p90), linewidths=0)
    plt.colorbar(sc, ax=axes[1], label="distance au cloud vrai (m)")
    axes[1].set_title(f"erreur carte : méd {map_med:.2f} m, p90 {map_p90:.2f} m")
    axes[1].set_aspect("equal")
    fig.suptitle(f"{label} — carte estimée vs carte re-rendue aux poses GT")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{label}_cloud_vs_gt.png"), dpi=150)
    plt.close(fig)

    # 4) trajectoire plaquée sur NOTRE nuage (même repère carte, aucun alignement)
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    ax.scatter(P[:, 0], P[:, 1], s=0.15, c="k", alpha=0.35, linewidths=0)
    ax.plot(d["est"][:, 0], d["est"][:, 1], color="tab:red", lw=1.3,
            label="trajectoire SLAM")
    ax.plot(d["est"][0, 0], d["est"][0, 1], "^", color="tab:red", ms=9, label="départ")
    ax.plot(d["est"][-1, 0], d["est"][-1, 1], "s", color="tab:red", ms=7, label="arrivée")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.set_title(f"{label} — trajectoire sur le nuage")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{label}_traj_on_cloud.png"), dpi=150)
    plt.close(fig)

    return {
        "label": label, "ate_um": ate_um, "ate_fp": ate_fp, "ate_sc": ate_sc, "s2": s2,
        "ate_odom": ate_odom, "re_t": re_t, "re_r": re_r, "sections": sections,
        "cap_med": cap_med, "cap_rms": cap_rms, "nn": nn_self,
        "map_med": map_med, "map_p90": map_p90,
        "t_rel": t_rel, "err_pos": err_pos, "resid": resid,
        "err_odo_fp": err_odo_fp,
        "est_a": est_a, "gxy": d["gxy"], "dr_a": dr_a, "tb": tb, "t0": d["t"][0],
    }


def comparer(res, out):
    """Figures de comparaison multi-méthodes (repère GT commun)."""
    cols = ["tab:blue", "tab:red", "tab:purple", "tab:brown"]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    r0 = res[0]
    ax.plot(r0["gxy"][:, 0], r0["gxy"][:, 1], color="0.25", lw=1.8, label="GT (DGPS)")
    ax.plot(r0["dr_a"][:, 0], r0["dr_a"][:, 1], color="tab:orange", lw=0.9, ls="--",
            label=f"odométrie pure ({r0['ate_odom']:.2f} m)")
    for r, c in zip(res, cols):
        ax.plot(r["est_a"][:, 0], r["est_a"][:, 1], color=c, lw=1.2,
                label=f"{r['label']} ({r['ate_um']:.2f} m)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    ax.legend(fontsize=9); ax.set_title("Comparaison des trajectoires (ATE Umeyama SE(2))")
    fig.tight_layout(); fig.savefig(os.path.join(out, "compare_traj.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), sharex=True)
    axes[0].plot(r0["t_rel"], r0["err_odo_fp"], color="tab:orange", lw=0.8, ls=":",
                 label="odométrie ancrée départ (conv. papiers)")
    axes[0].plot(r0["t_rel"], np.linalg.norm(r0["dr_a"] - r0["gxy"], axis=1),
                 color="tab:orange", lw=0.9, ls="--", label="odométrie (Umeyama)")
    for r, c in zip(res, cols):
        axes[0].plot(r["t_rel"], r["err_pos"], color=c, lw=1.1, label=r["label"])
        axes[1].plot(r["t_rel"], np.degrees(np.abs(r["resid"])), color=c, lw=1.0,
                     label=r["label"])
    axes[0].set_ylabel("erreur position (m)")
    axes[1].set_ylabel("erreur cap (°)"); axes[1].set_xlabel("temps mission (min)")
    for ax in axes:
        for k in range(1, len(r0["tb"]) - 1):
            ax.axvline((r0["tb"][k] - r0["t0"]) / 60.0, color="0.6", lw=0.8, ls=":")
        ax.grid(alpha=0.25); ax.legend(fontsize=9)
    axes[0].set_title("Erreurs au cours du temps (S1 | S2 | S3)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "compare_err_time.png"), dpi=150)
    plt.close(fig)
    print(f"-> {out}/compare_traj.png, compare_err_time.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="dossier du run")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--imin", type=float, default=None, help="filtre intensité cloud")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="liste rundir:label[:imin] → figures de comparaison + table")
    a = ap.parse_args()
    if a.compare:
        out = a.out or "."
        res = []
        for spec in a.compare:
            parts = spec.split(":")
            rd, lb = parts[0], parts[1]
            imin = float(parts[2]) if len(parts) > 2 else None
            res.append(eval_run(rd, lb, out, imin))
        comparer(res, out)
        # table markdown prête à coller
        print("\n| Méthode | ATE full (m) | ATE S1/S2/S3 (m) | Trans. (%) | Rot. (°/m) | Cap méd (°) | Carte méd/p90 (m) |")
        print("|---|---|---|---|---|---|---|")
        for r in res:
            s = " / ".join(f"{x[0]:.2f}" for x in r["sections"])
            print(f"| {r['label']} | {r['ate_fp']:.2f} (fp) · {r['ate_um']:.2f} (um) | {s} | "
                  f"{r['re_t']:.2f} | {r['re_r']:.3f} | {r['cap_med']:.1f} | "
                  f"{r['map_med']:.2f} / {r['map_p90']:.2f} |")
    else:
        label = a.label or os.path.basename(a.run.rstrip("/"))
        eval_run(a.run, label, a.out or a.run, a.imin)


if __name__ == "__main__":
    main()
