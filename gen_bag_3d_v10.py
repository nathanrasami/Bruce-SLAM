#!/usr/bin/env python3
"""Generateur traj9 « quais 1 tour, passage SOUS le navire » (SUITE.md 6,
accord Nathan 14-07 soir : 1 SEUL tour entre les 2 quais, plus profond, on
passe SOUS le navire au lieu de le contourner -> on capte les DEUX quais).

E0 MESURE (probe_traj9_ship{,2,3}.py + .json, 2026-07-14 soir) :
  - navire FLOTTANT amarre au quai OUEST : x 466..484.5, y -645..-798
    (~153 x 20 m), QUILLE PLATE z = -2.90 constant, fond -19.4
    -> 16.5 m d'eau libre sous la coque ;
  - AUCUN treillis dessous : 0 echo sur 117 rayons verticaux fins (grille
    1 m) + 0 plan sur 4 rayons horizontaux de 140 m (z -5/-8/-12/-16) ;
    (le treillis de face.png = l'autre navire, marina — zone B) ;
  - pieux des 2 quais presents sur y -644..-796 (est ~11.8 m depuis x=520,
    ouest 27.5 m depuis x=490) ; rive au sud de -812, tombant au nord de -628.
  - NB : le « mur GAMMA x=485 » de la reco traj3 = tres probablement le FLANC
    de ce navire (flanc mesure 484.5, proue -645 = « GAMMA horizontal »).

Trajectoire : rectangle 1 tour CW (cibles a GAUCHE sur les 2 grandes jambes),
depart NE pres de P_A : NE (524,-640) -> SE (524,-806) [jambe EST, quai est a
7.5 m a gauche] -> SW (474,-806) -> NW (474,-640) [jambe OUEST, SOUS la coque,
quai ouest a 11.5 m a gauche] -> NE [traversee de la proue]. PERIM ~432 m,
~21 min a 0.35 m/s. Bande z [-8,-5] : >= 2.1 m sous la quille (-2.9),
>= 11.4 m sur le fond. Phase A INCHANGEE de traj4 (P_A (526.3,-660) Z_A -9.5,
C1 cap EST face au quai = defauts v5 -> E1-E7 identiques, pas de duplication
de _build_segments).

Herite v5 (machinerie) + v6 (profiler transverse) + v7 (RangeMax 20) + v8
(main nav realiste). SEED_NAV 9 (dedie). octree_min 0.05 (= traj8).
GARDE-FOU E0 : REFUSE de generer si les 3 JSON de probe manquent ou si leurs
mesures contredisent la geometrie ci-dessus.

Usage : python gen_bag_3d_v10.py [--test 150] [--seed 42]
"""
import json
import os
import sys
import numpy as np

import gen_bag_3d_v5 as v5
import gen_bag_3d_v6 as v6
import gen_bag_3d_v7 as v7            # noqa: F401 — patch RangeMax 20
import gen_bag_3d_v8 as v8            # main nav realiste
import noise_round2 as _NOISE          # round 2 « noise » (NOISE_ROUND2=1)

_D = os.path.dirname(os.path.abspath(__file__))
E0_JSONS = [os.path.join(_D, f) for f in
            ("probe_traj9_ship.json", "probe_traj9_ship2.json",
             "probe_traj9_ship3.json")]
OCTREE_MIN_V10 = 0.05
SEED_NAV_V10 = 9                      # dedie traj9 (traj8 = 8)
N_LAPS_V10 = 1                        # SUITE.md : 1 SEUL tour
R_TURN_V10 = 4.0
KEEL_Z = -2.90                        # quille mesuree (E0 pass 1/2)

WPTS10 = np.array([
    [524.0, -640.0],   # NE — depart (P_A a 20 m) : jambe EST cap SUD
    [524.0, -806.0],   # SE — vire OUEST (poupe -798 passee a 8 m)
    [474.0, -806.0],   # SW — vire NORD (jambe OUEST : SOUS la coque)
    [474.0, -640.0],   # NW — vire EST (traversee proue, coque des y -645)
])
Z_MIN_V10, Z_MAX_V10 = -6.5, -6.5     # z CONSTANT (demande Nathan 14-07 soir,
                                      # SUITE « pas besoin de faire varier z ») :
                                      # 3.6 m sous quille, 12.9 m sur fond ;
                                      # seule la couture phase A passe par -9.5
N_MAX_V10 = 0.8

# ─── Patches v5 (geometrie ; P_A/Z_A/C1 = defauts v5 conserves) ──────────────
v5.R_TURN = R_TURN_V10
v5.N_LAPS = N_LAPS_V10
v5.WPTS = WPTS10
v5._PIECES, v5._PCUM = v5._rounded_path(WPTS10, R_TURN_V10)
v5.PERIM = float(v5._PCUM[-1])
v5.N_MAX = N_MAX_V10
v5.Z_MIN, v5.Z_MAX = Z_MIN_V10, Z_MAX_V10

v8.OUT_BAG = _NOISE.bag("BAG_files/holoocean_3d_traj9.bag")  # _noise si round 2
v8.SEED_NAV = SEED_NAV_V10

_make_cfg_v6 = v6.make_cfg_v6


def make_cfg_v10():
    cfg = _make_cfg_v6()
    cfg["name"] = "gen3dv10"
    cfg["octree_min"] = OCTREE_MIN_V10
    return cfg


v6.make_cfg_v6 = make_cfg_v10         # v8.main l'appelle par attribut de module

v5._init_traj(v5.SEED)                # re-derive tout sur la geometrie v10


def verifier_chemin_v10():
    """Continuite/lacet/z comme v5-v9 + GARDE-FOU E0 : la geometrie du navire
    n'est connue que par probe (PIEGES #18 : structure inferee != probee)."""
    s = np.arange(0.0, v5.PERIM, 0.5)
    pts = np.array([v5.chemin_median(v)[0] for v in s])
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    tt = np.arange(0.0, v5.PERIM / v5.V_FWD, 1.0)
    yy = np.unwrap([v5.yaw_errance(v5.V_FWD * v) for v in tt])
    dyaw = np.degrees(np.abs(np.diff(yy)))
    zz = np.array([v5.pos_errance(v5.V_FWD * v)[2] for v in tt])
    print(f"chemin traj9 : PERIM={v5.PERIM:.1f} m x{v5.N_LAPS} tour | pas "
          f"median max {step.max():.3f} (0.5 attendu) | |dyaw/dt| max "
          f"{dyaw.max():.1f} deg/s | z [{zz.min():.1f},{zz.max():.1f}]")
    assert 0.4 < step.max() < 0.6, "median discontinu"
    assert dyaw.max() < 25.0, "taux de lacet irrealiste"
    # bord BAS : la couture d'errance passe par Z_A=-9.5 (phase A traj4
    # conservee) — sans danger (fond -19.4). Le bord HAUT est le critique
    # (clearance sous quille) : bande stricte.
    assert zz.min() >= v5.Z_A - 1e-6 and zz.max() <= Z_MAX_V10 + 1e-6
    # marge sous quille (jambe ouest) : bande z entiere + errance laterale
    assert Z_MAX_V10 <= KEEL_Z - 2.0, "moins de 2 m sous la quille"

    for f in E0_JSONS:
        assert os.path.exists(f), f"E0 manquant : {os.path.basename(f)}"
    e1 = json.load(open(E0_JSONS[0]))
    e2 = json.load(open(E0_JSONS[1]))
    e3 = json.load(open(E0_JSONS[2]))
    assert e1["temoins"]["ok"] and e2["temoin"]["ok"], "temoins E0 en echec"
    # quille plate -2.90 sur l'empreinte (pass 1)
    keel = [s_["z_obs"] for s_ in e1["stations"]
            if s_.get("z_obs") is not None and -786 <= s_["y"] <= -650
            and 466 <= s_["x"] <= 482]
    assert len(keel) >= 150 and abs(float(np.median(keel)) - KEEL_Z) < 0.2, \
        "quille != -2.90 : re-prober"
    # aucun treillis (pass 2)
    n_bar = sum(1 for g in e2["grille_fine"]
                if g["z_obs"] is not None and g["z_obs"] < -3.5)
    n_plans = sum(len(h) for h in e2["plans_transverses"].values())
    assert n_bar == 0 and n_plans == 0, "obstacle sous coque detecte : STOP"
    # poupe (pass 2) : coque finie avant la jambe sud y=-806
    hull_s = [e["y"] for e in e2["sud"]
              if e.get("z_obs") is not None and e["z_obs"] < 0.0]
    assert min(hull_s, default=0.0) >= -802.0, "poupe au-dela de -802"
    # fond plat sur la plage du tour (pass 3)
    fonds = [m["fond"] for m in e3["murs"]
             if m["fond"] is not None and -806 <= m["y"] <= -640]
    assert min(fonds) < -19.0, "fond releve dans la plage du tour"
    print(f"E0 OK (passes {e1['date']} / {e2['date']} / {e3['date']}) : "
          f"quille med {float(np.median(keel)):.2f} | 0 treillis | "
          f"poupe >= -802 | fond <= {max(fonds):.1f}")


v5.verifier_chemin = verifier_chemin_v10
v6.verifier_chemin = verifier_chemin_v10
v7.verifier_chemin_v7 = verifier_chemin_v10  # v8.main l'appelle par attribut

if __name__ == "__main__":
    print(f"TRAJ9 | quais 1 tour + passage SOUS le navire | octree_min "
          f"{OCTREE_MIN_V10} | nav v8 SEED_NAV={SEED_NAV_V10} | PERIM "
          f"{v5.PERIM:.1f} m x{N_LAPS_V10} tour | base v8 :")
    v8.main()
