#!/usr/bin/env python3
"""Comparaison d'un run traj6 (carte FUSION vert+transverse) au run de
référence traj5 (222233) — chiffres + verdicts imprimés avec interprétation.

Usage :  python3 analysis/compare_traj6.py results/<run_traj6> [<run_ref>]
         (run_ref défaut : results/run_holoocean_2026-07-11_222233)
Tourne DANS le conteneur ros1 OU avec le venv hôte : ne lit que
trajectory.csv / groundtruth.csv / carte_3d.npy (PAS le bag) → à lancer
APRÈS ./analyse.sh 3D <run>.

Ce qu'il calcule (et pourquoi, cf. TRAJ6_ANALYSE.md) :
  1. ATE Umeyama SE(2) + cap RMS par run — attendu : traj6 ≈ traj5 (le SLAM ne
     lit que /sonar /dvl /imu /depth, STRICTEMENT identiques entre les 2 bags).
  2. Cartes 3D alignées en MONDE (Umeyama de chaque trajectoire vers sa GT +
     offset z médian — la GT ne sert qu'à l'évaluation, jamais au SLAM) :
     - COUVERTURE : NN carte_ref -> carte_traj6 (la fusion doit contenir ce
       que traj5 voyait) ;
     - APPORT : % de points traj6 à >1 m de la carte traj5 (= le NOUVEAU
       contenu amené par le transverse : fond hors nadir, flancs continus).
  3. Répartition en z de chaque carte (bande fond z<-17 / structures -17..-2.5).
⚠ Ne JAMAIS comparer le NN carte-vs-GT de traj6 au 0.107 m de traj5 : les
contenus diffèrent (PIEGES #15, NN flatté/défavorisé selon le contenu).
"""
import os
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_eval import umeyama, associer_par_temps, calculer_ate, appliquer

REF_DEFAUT = "results/run_holoocean_2026-07-11_222233"


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def eval_run(run):
    tr = np.genfromtxt(os.path.join(run, "trajectory.csv"),
                       delimiter=",", names=True)
    gt = np.genfromtxt(os.path.join(run, "groundtruth.csv"),
                       delimiter=",", names=True)
    ts = tr["time"]
    gxy = associer_par_temps(ts, gt["time"], gt["x"], gt["y"])
    s, R, t = umeyama(np.column_stack([tr["x"], tr["y"]]), gxy, with_scale=False)
    est_al = appliquer(1.0, R, t, np.column_stack([tr["x"], tr["y"]]))
    ate = calculer_ate(est_al, gxy)
    # cap : erreur SLAM-GT après rotation d'alignement, offset médian retiré
    # (méthodo heading-error : on mesure la DÉRIVE de cap, pas l'ancrage)
    ang = float(np.arctan2(R[1, 0], R[0, 0]))
    th_gt = np.interp(ts, gt["time"], np.unwrap(gt["theta"]))
    err = _wrap(np.unwrap(tr["theta"]) + ang - th_gt)
    err = _wrap(err - np.median(err))
    cap_rms = float(np.degrees(np.sqrt(np.mean(err ** 2))))
    z_off = float(np.median(np.interp(ts, gt["time"], gt["z"]) - tr["z"]))
    carte = None
    npy = os.path.join(run, "carte_3d.npy")
    if os.path.isfile(npy):
        S = np.load(npy)
        carte = np.c_[(S[:, :2] @ R.T) + t, S[:, 2] + z_off]   # carte en MONDE
    return {"ate": ate, "cap": cap_rms, "nkf": len(ts), "carte": carte}


def repartition_z(C):
    fond = (C[:, 2] < -17.0).mean()
    struct = ((C[:, 2] >= -17.0) & (C[:, 2] < -2.5)).mean()
    return 100 * fond, 100 * struct


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run6 = sys.argv[1].rstrip("/")
    ref = (sys.argv[2] if len(sys.argv) > 2 else REF_DEFAUT).rstrip("/")
    A, B = eval_run(ref), eval_run(run6)

    print(f"réf  {os.path.basename(ref)} : ATE {A['ate']:.3f} m | cap RMS "
          f"{A['cap']:.2f}° | {A['nkf']} keyframes")
    print(f"run  {os.path.basename(run6)} : ATE {B['ate']:.3f} m | cap RMS "
          f"{B['cap']:.2f}° | {B['nkf']} keyframes")
    d_ate = abs(B["ate"] - A["ate"])
    print(f"\n[1] SLAM : |ΔATE| = {d_ate:.3f} m -> " + (
        "✓ ATTENDU (les topics lus par le SLAM sont identiques traj5/traj6)."
        if d_ate <= 0.02 else
        "⚠ SURPRENANT : le SLAM ne lit que /sonar /dvl /imu /depth, identiques "
        "entre les 2 bags → un écart >0.02 m est une VRAIE anomalie (R2 : "
        "reproduire, comparer les configs de run, PIEGES avant tout)."))

    if A["carte"] is None or B["carte"] is None:
        sys.exit("carte_3d.npy manquant (lancer ./analyse.sh 3D <run> d'abord)")
    CA, CB = A["carte"], B["carte"]
    fa, sa = repartition_z(CA)
    fb, sb = repartition_z(CB)
    print(f"\ncarte réf  : {len(CA):6,} pts | fond(z<-17) {fa:4.1f} % | "
          f"structures(-17..-2.5) {sa:4.1f} %")
    print(f"carte run  : {len(CB):6,} pts | fond(z<-17) {fb:4.1f} % | "
          f"structures(-17..-2.5) {sb:4.1f} %")

    rng = np.random.default_rng(0)
    ia = rng.choice(len(CA), min(30000, len(CA)), replace=False)
    ib = rng.choice(len(CB), min(30000, len(CB)), replace=False)
    couv = cKDTree(CB).query(CA[ia], k=1)[0]
    d_new = cKDTree(CA).query(CB[ib], k=1)[0]
    apport = 100 * (d_new > 1.0).mean()
    c_med, c_p90 = np.median(couv), np.percentile(couv, 90)

    print(f"\n[2] COUVERTURE (réf -> run) : NN méd {c_med:.2f} m / p90 "
          f"{c_p90:.2f} m -> " + (
          "✓ la fusion contient ce que traj5 voyait."
          if c_med <= 0.30 else
          "⚠ le contenu traj5 n'est pas retrouvé : désalignement des cartes ou "
          "source verticale absente de la fusion (relire la sortie carte_3d : "
          "les DEUX topics doivent être retenus, libellé FUSION)."))
    print(f"[3] APPORT du transverse : {apport:.1f} % des points du run à >1 m "
          f"de la carte réf -> " + (
          "✓ contenu nouveau significatif (fond hors nadir + flancs continus)."
          if apport >= 15.0 else
          "⚠ quasi aucun contenu nouveau : le transverse n'est probablement "
          "PAS entré dans la carte — vérifier dans la sortie de carte_3d.py "
          "que /profiler_points est détecté 'profiler transverse' et retenu."))
    print("\nRappels d'interprétation (TRAJ6_ANALYSE.md, PIEGES #15/#16) :")
    print("  - ne PAS comparer le NN carte-vs-GT (titre carte_3d) entre runs à")
    print("    contenus différents ; le juge de paix ici = [2] et [3].")
    print("  - un apport élevé avec couverture OK = SUCCÈS traj6, même si le")
    print("    NN-GT affiché bouge par rapport à 0.107/0.733.")
    print("  - mount transverse FIGÉ (mesuré ×2) : une carte transverse 'en")
    print("    miroir' ne se corrige JAMAIS dans le mount sans re-probe (#16).")


if __name__ == "__main__":
    main()
