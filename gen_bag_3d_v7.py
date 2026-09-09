#!/usr/bin/env python3
"""Generateur traj7 « au ras des quais » (spec Nathan 2026-07-12) : traj6
(3 capteurs, phase A calibration inchangee) avec trois changements :

1. QUAIS SERRES 2-3 m : nominal 2.5 m des faces internes (quai EST x=529.0
   / face 531.5 ; quai OUEST x=465.0 / face 462.5), excursion laterale
   N_MAX 1.2 -> 0.5 m => bande reelle [2.0, 3.0] m (traj6 serrait 4-6.5 m).
2. RangeMax sonar HORIZONTAL 40 -> 20 m (512 bins : 7.7 -> 3.8 cm/bin).
   /sonar_points est reprojete a la meme echelle. ⚠ SLAM : lancer avec
   SONAR_RANGE=20 ./run_slam.sh holoocean (bridge ~range_m), et les checks
   avec --rmax-h 20.
3. DETOUR BATEAU SUPPRIME : le « bateau (524,-680.5) » de la reco traj3 est
   REFUTE (double probe 2026-07-12 : probe_boat_traj7.py RangeFinder +
   probe_boat_sonar.py octree — fond PLAT -19.4 sur toute l'empreinte,
   rayons ET sonar traversent jusqu'au quai). C'etait un fantome de
   reprojection (rabattus ±20°). « Passer par-dessous » est sans objet :
   la jambe quai EST file droit x=529 de -626 a -688, au ras du quai.

Le reste = traj6 : errance PCHIP seed 42, z [-12,-2], 2 tours (meme tirage),
GAMMA 4-5 m, profiler transverse 360° (mount MESURE, PIEGES #16), phase A
C1/C2/C3 aux memes fenetres (E2/E4/E6/E7).

Implementation : patch du module v5 (circuit, N_MAX, RANGE_MAX) puis main()
de v6 inchange (writer + 3e capteur). Pas de duplication du pipeline.

Usage : python gen_bag_3d_v7.py [--test 150] [--seed 42]
"""
import functools
import numpy as np

import gen_bag_3d as g3
import gen_bag_3d_v5 as v5
import gen_bag_3d_v6 as v6

RANGE_MAX_H7 = 20.0

# ─── Circuit v7 : quais 2.5 m nominal, detour bateau supprime (8 -> 7 sommets).
# W1 garde (529,-662) pres de P_A : l'entree d'errance (s=0) reste un transit
# court depuis la phase A ; W7->W1 est colineaire a W1->W2 (fillet degenere,
# gere par _rounded_path : arc de longueur nulle).
WPTS7 = np.array([
    [529.0, -662.0],   # W1  quai EST 2.5 m (face 531.5), pres P_A
    [529.0, -688.0],   # W2  plein sud AU RAS du quai (ex-zone « bateau », vide)
    [490.0, -688.0],   # W3  ouest, eau libre
    [490.0, -641.0],   # W4  remonte a 5 m du GAMMA vertical (x=485)
    [465.0, -641.0],   # W5  corridor a 4 m du GAMMA horizontal (y=-645)
    [465.0, -626.0],   # W6  quai OUEST 2.5 m (face 462.5)
    [529.0, -626.0],   # W7  nord -> retour quai EST
])

v5.WPTS = WPTS7
v5._PIECES, v5._PCUM = v5._rounded_path(WPTS7, v5.R_TURN)
v5.PERIM = float(v5._PCUM[-1])
v5.N_MAX = 0.5                    # bande laterale 2.5 ± 0.5 m
v5.RANGE_MAX = RANGE_MAX_H7       # make_cfg : sonar horizontal seulement
                                  # (vert/profiler gardent leur RangeMax 20 propre)

# /sonar_points : le defaut range_max=40 de sonar_to_points3d_msg est fige a
# la DEFINITION (gen_bag_3d.py) -> partial pour l'appel horizontal de v6.main ;
# les appels vert/profiler passent range_max explicitement (prioritaire).
v6.sonar_to_points3d_msg = functools.partial(g3.sonar_to_points3d_msg,
                                             range_max=RANGE_MAX_H7)
v6.OUT_BAG = "BAG_files/holoocean_3d_traj7.bag"


def verifier_chemin_v7():
    """Memes mesures que v5.verifier_chemin, seuils v7 (quais 2-3 m)."""
    s = np.arange(0.0, v5.PERIM, 0.5)
    env = np.array([v5.pos_errance(v)[:2] for v in s])
    d_gamma = np.min([v5._d_seg(env, a, b) for a, b in v5._GAMMA], axis=0)
    d_quais = np.min([v5._d_seg(env, a, b) for a, b in v5._QUAIS], axis=0)
    pts = np.array([v5.chemin_median(v)[0] for v in s])
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    tt = np.arange(0.0, v5.PERIM / v5.V_FWD, 1.0)
    yy = np.unwrap([v5.yaw_errance(v5.V_FWD * v) for v in tt])
    dyaw = np.degrees(np.abs(np.diff(yy)))
    zz = np.array([v5.pos_errance(v5.V_FWD * v)[2] for v in tt])
    print(f"chemin traj7 : PERIM={v5.PERIM:.1f} m | quais min "
          f"{d_quais.min():.2f} m (spec 2-3 m) | GAMMA min {d_gamma.min():.2f} m"
          f" | pas median max {step.max():.3f} (0.5 attendu) | "
          f"|dyaw/dt| max {dyaw.max():.1f}°/s | z [{zz.min():.1f},{zz.max():.1f}]")
    assert d_quais.min() > 1.8, "trop pres d'une face de quai"
    assert d_quais.min() < 3.2, "quais pas serres (spec 2-3 m non appliquee ?)"
    assert d_gamma.min() > 2.5, "trop pres du mur GAMMA"
    assert 0.4 < step.max() < 0.6, "median discontinu"
    assert dyaw.max() < 25.0, "taux de lacet irrealiste"


v5.verifier_chemin = verifier_chemin_v7
v6.verifier_chemin = verifier_chemin_v7   # v6.main l'a importee PAR NOM

if __name__ == "__main__":
    # re-derive F_LAT/F_Z/segments sur le NOUVEAU perimetre (v6.main ne
    # rappelle _init_traj que si --seed != 42)
    v5._init_traj(v5.SEED)
    print(f"TRAJ7 | quais 2.5±0.5 m | RangeMax H {RANGE_MAX_H7:.0f} m | "
          f"detour bateau supprime (refute par probe) | base v6 :")
    v6.main()
