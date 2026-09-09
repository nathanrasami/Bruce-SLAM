#!/usr/bin/env python3
"""Rapport SAMPLE_DATA — bag d'origine de Bruce-SLAM (sample_data.bag).

Ce bag ne contient QUE 4 topics capteurs (IMU / DVL / pression / Oculus) : il n'y a
AUCUN ground truth, donc aucun ATE n'est calculable. La seule erreur mesurable est
l'ECART entre le dead reckoning (IMU+DVL+pression, sans sonar) et la trajectoire SLAM
optimisée — c'est aussi la quantité qui dit si le SLAM a bien corrigé la dérive.

Convention ORIGINE : chaque trajectoire est ancrée sur sa première pose (pas
d'alignement Umeyama, qui n'aurait rien sur quoi s'aligner sans GT).

Sorties (2 fichiers, cf. demande) :
  carte_finale.png     nuage sonar + trajectoire SLAM + dead reckoning, une seule carte
  error_over_time.png  ecart SLAM<->DR : position (m) et cap (deg) au fil du temps

Usage : python3 analysis/sample_data_report.py <run_dir>
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_eval import nn_auto

C_SLAM, C_DR, C_CLOUD = "tab:blue", "tab:orange", "0.65"


def ancrer(xy, xy0, theta0):
    """Convention origine : repère ramené sur la pose initiale (xy0, theta0).
    Le nuage doit recevoir la MÊME transfo que le SLAM (il y est exprimé)."""
    c, s = np.cos(-theta0), np.sin(-theta0)
    return (xy - xy0) @ np.array([[c, s], [-s, c]])


def carte_port(run_dir, est, dr, pts, run):
    """Superpose nuage + trajectoires sur la photo aérienne de la marina.
    Le calage (analysis/marina_calage.json) vient d'un ICP entre notre nuage et celui
    dessiné dans em.gif : il est APPROXIMATIF et ajusté sur nos propres points, donc
    lisible comme illustration, jamais comme ground truth chiffré."""
    ici = os.path.dirname(os.path.abspath(__file__))
    fond_p, cal_p = os.path.join(ici, "marina_map.png"), os.path.join(ici, "marina_calage.json")
    if not (os.path.exists(fond_p) and os.path.exists(cal_p)):
        return False
    import json
    import matplotlib.image as mpimg
    from PIL import Image
    cal = json.load(open(cal_p))
    A, b = np.array(cal["A"]), np.array(cal["b"])
    ox, oy = cal["origine_px"]
    fond = mpimg.imread(fond_p)

    def px(EN):  # (East, North) mètres -> pixel du fond
        X = EN @ A.T + b
        return X[:, 0] - ox, -X[:, 1] - oy

    # métrique GT-free : distance du nuage aux STRUCTURES du port (arêtes de la photo).
    # Seule référence indépendante de tout SLAM sur ce bag. Le calage étant figé, elle est
    # comparable d'un run à l'autre : c'est ce qui permet de classer deux configurations.
    # On INTERPOLE le champ de distance : lu en entier, il quantifie la médiane sur des
    # valeurs de pixel (3 px = 0.69 m, 4 px = 0.91 m...) et fabrique de fausses égalités.
    # Majorant de l'erreur : y entrent la parallaxe (le sonar voit les flancs immergés,
    # la photo les toits) et les bateaux qui ont bougé entre les deux dates.
    from scipy import ndimage as ndi
    noyaux = [np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
              np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
              np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], float),
              np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], float)]
    g = ndi.gaussian_filter(np.asarray(Image.open(fond_p).convert("L"), float), 1.0)
    mag = np.sqrt(sum(ndi.convolve(g, k) ** 2 for k in noyaux))
    champ = ndi.distance_transform_edt(mag <= np.percentile(mag, 97))
    cx, cy = px(pts[:, ::-1])
    ok = (cx >= 0) & (cx < champ.shape[1]) & (cy >= 0) & (cy < champ.shape[0])
    d = ndi.map_coordinates(champ, [cy[ok], cx[ok]], order=1,
                            mode="nearest") * cal["m_par_px"]
    print(f"  nuage vs structures: moy {d.mean():.2f} m | med {np.median(d):.2f} m"
          f" | <1 m {(d < 1).mean() * 100:.1f} %")

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(fond)
    x, y = px(pts[:, ::-1]);  ax.scatter(x, y, s=1.0, c="red", lw=0,
                                         label=f"nuage sonar ({len(pts)} pts)")
    x, y = px(dr[:, ::-1]);   ax.plot(x, y, color="deepskyblue", lw=1.5, ls="--",
                                      label="dead reckoning")
    x, y = px(est[:, ::-1]);  ax.plot(x, y, color="yellow", lw=2.0, label="SLAM optimisé")
    ax.plot(*[v[0] for v in px(est[:1, ::-1])], "wo", ms=6, mec="k", label="départ")
    ax.set_xlim(0, fond.shape[1]); ax.set_ylim(fond.shape[0], 0); ax.axis("off")
    ax.set_title(f"{run} — nuage et trajectoires sur la marina "
                 f"(nuage à {d.mean():.2f} m des structures en moyenne, "
                 f"{(d < 1).mean() * 100:.0f} % sous 1 m)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "carte_port.png"), dpi=140)
    plt.close(fig)

    return True


def main(run_dir):
    ld = lambda f: np.genfromtxt(os.path.join(run_dir, f), delimiter=",", names=True)
    traj = ld("trajectory.csv")
    dr_full = ld("dead_reckoning.csv")
    run = os.path.basename(run_dir.rstrip("/"))

    # trajectoires ancrées à l'origine ; le nuage suit la MÊME transfo que le SLAM
    slam_xy = np.column_stack([traj["x"], traj["y"]])
    drkf_xy = np.column_stack([traj["dr_x"], traj["dr_y"]])
    drf_xy = np.column_stack([dr_full["x"], dr_full["y"]])
    est = ancrer(slam_xy, slam_xy[0], traj["theta"][0])
    dr_kf = ancrer(drkf_xy, drkf_xy[0], traj["dr_theta"][0])
    dr = ancrer(drf_xy, drf_xy[0], dr_full["theta"][0])

    cloud = np.genfromtxt(os.path.join(run_dir, "pointcloud.csv"), delimiter=",",
                          names=True)
    pts = ancrer(np.column_stack([cloud["x"], cloud["y"]]),
                 slam_xy[0], traj["theta"][0])

    # écarts SLAM <-> DR aux keyframes (mêmes instants, aucune interpolation)
    d_pos = np.linalg.norm(est - dr_kf, axis=1)
    d_cap = np.degrees(np.arctan2(np.sin(traj["theta"] - traj["dr_theta"]),
                                  np.cos(traj["theta"] - traj["dr_theta"])))
    t_min = (traj["time"] - traj["time"][0]) / 60.0
    nssm = int(traj["nssm_constraints"].sum())

    print(f"{run} : {len(traj)} keyframes, {len(pts)} points, {nssm} loop closures")
    print(f"  ecart SLAM<->DR   : med {np.median(d_pos):.2f} m | max {d_pos.max():.2f} m"
          f" | final {d_pos[-1]:.2f} m")
    print(f"  ecart de cap      : med {np.median(np.abs(d_cap)):.2f} deg"
          f" | max {np.abs(d_cap).max():.2f} deg")
    print(f"  longueur parcourue: SLAM {np.linalg.norm(np.diff(est, axis=0), axis=1).sum():.1f} m"
          f" | DR {np.linalg.norm(np.diff(dr, axis=0), axis=1).sum():.1f} m")
    # netteté du nuage : SEUL critère de qualité mesurable sans GT (plus bas = plus net)
    print(f"  netteté carte     : NN median {nn_auto(pts):.3f} m")

    # --- carte_finale.png : nuage + SLAM + DR
    # Bruce travaille en NED (x = avant/Nord, y = droite/Est) : on trace y en abscisse
    # et x en ordonnée, sinon la carte sort transposée par rapport à la vue RViz de
    # référence (video.rviz : Yaw 4.755 rad + Invert Z Axis) et aux cartes marines.
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(pts[:, 1], pts[:, 0], s=0.6, c=C_CLOUD, lw=0,
               label=f"nuage sonar ({len(pts)} pts)")
    ax.plot(dr[:, 1], dr[:, 0], ls="--", lw=1.2, color=C_DR,
            label="dead reckoning (IMU+DVL+pression)")
    ax.plot(est[:, 1], est[:, 0], lw=1.8, color=C_SLAM,
            label=f"SLAM optimisé ({nssm} loop closures)")
    ax.plot(est[0, 1], est[0, 0], "ko", ms=6, label="départ")
    ax.set_xlabel("y — Est (m)"); ax.set_ylabel("x — Nord (m)")
    ax.set_title(f"{run} — carte finale (convention origine)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "carte_finale.png"), dpi=140)
    plt.close(fig)

    # --- error_over_time.png : ECART entre deux estimations. Sans GT, aucune des deux
    # n'est une vérité : cette courbe mesure de combien le sonar a corrigé le DR, pas
    # une erreur. D'où le vocabulaire « écart » partout dans la figure.
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(t_min, d_pos, color=C_SLAM, lw=1.5)
    ax.set_xlabel("temps mission (min)"); ax.set_ylabel("écart de trajectoire (m)")
    ax.set_title(f"{run}\nécart SLAM <-> dead reckoning — deux estimations, aucune vérité",
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "error_over_time.png"), dpi=140)
    plt.close(fig)

    faits = ["carte_finale.png", "error_over_time.png"]
    if carte_port(run_dir, est, dr, pts, run):
        faits.append("carte_port.png")
    print("->", run_dir, ":", ", ".join(faits))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else os.environ.get("SLAM_RESULTS_DIR", "results"))
