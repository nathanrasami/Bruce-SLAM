#!/usr/bin/env python3
"""gates_refonte.py — verdict post-runs de la REFONTE (REFONTE_MISSION.md, chantier §5).

4 gates, verdict imprimé, rien d'affaibli en silence :
  ① dr identiques ×4 : rigid-check des odometry.csv (rotation ~0°, résidu ~0)
      → preuve que l'odométrie n'a pas été touchée par la méthode (⛔2).
  ② ATE origine : TRANSLATION PURE (départs épinglés (0,0), zéro rotation),
      global + sections S1/S2/S3 ré-épinglées — même métrique que
      paper_figs_origine.py (⛔1).
  ③ ordre : BSU ≤ Bruce_U < Bruce (AMENDÉ par Nathan 16-07 : Bruce_Sonar sort du
      verdict — SC sans ancre absolue = 0/154 loops acceptées, la dérive à la
      revisite (méd 9.5 m) dépasse le garde-fou 8 m → BS ≈ odométrie, reporté au
      papier comme FINDING de complémentarité SC↔USBL, pas comme échec de gate).
      Violation ≤ 0.10 m = LIMITE (variance ICP connue : ×2 avant verdict, R3) ;
      au-delà = FAIL → STOP, investigation R2, pas de rapport (⛔5).
  ④ carte : cloud vs cloud-GT méd/p90 (Umeyama INTERNE debug, jamais papier).
      Informatif.

Usage :
  python3 analysis/gates_refonte.py                          # derniers run_<preset>_*
  python3 analysis/gates_refonte.py <bruce> <bruce_u> <bruce_sonar> <bsu>
Code retour : 0 = PASS · 2 = LIMITE · 1 = FAIL.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_eval import charger, cloud_vrai, fit_cap, umeyama  # noqa: E402
from traj_eval import calculer_ate  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESETS = ["bruce", "bruce_u", "bruce_sonar", "bsu"]
# tolérances gate ① (drops de messages possibles sous charge, intégration inchangée)
ROT_TOL_DEG, MED_TOL_M, P99_TOL_M = 0.02, 0.01, 0.05
VAR_ICP = 0.10  # m, variance run-à-run connue (gate ③, zone LIMITE)


def kabsch_angle(p, q):
    pc, qc = p - p.mean(axis=0), q - q.mean(axis=0)
    return np.degrees(np.arctan2(np.sum(pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0]),
                                 np.sum(pc[:, 0] * qc[:, 0] + pc[:, 1] * qc[:, 1])))


def derniers_runs():
    runs = []
    for p in PRESETS:
        # motif date "2*" : évite que run_bruce_* matche run_bruce_u_* / _sonar_*
        cand = sorted(glob.glob(os.path.join(HERE, "results", f"run_{p}_2*")))
        if not cand:
            raise SystemExit(f"gate: aucun run results/run_{p}_2* trouvé")
        runs.append(cand[-1])
    return runs


def gate1_dr(dirs):
    """dr identiques ×4 : rigid-check de chaque odometry.csv contre celle de bruce."""
    tracks = []
    for d in dirs:
        o = pd.read_csv(os.path.join(d, "odometry.csv"))
        tracks.append(o[["time", "x", "y"]].to_numpy())
    ref = tracks[0]
    ok = True
    print("\n── Gate ① dr identiques ×4 (réf = bruce) "
          f"[seuils : rot ≤ {ROT_TOL_DEG}°, méd ≤ {MED_TOL_M} m, p99 ≤ {P99_TOL_M} m]")
    print(f"  {'méthode':<12} {'N pts':>7} {'rotation':>9} {'résidu méd':>11} "
          f"{'p99':>8} {'max':>8}")
    for name, tr in zip(PRESETS[1:], tracks[1:]):
        t0, t1 = max(ref[0, 0], tr[0, 0]), min(ref[-1, 0], tr[-1, 0])
        m = (ref[:, 0] >= t0) & (ref[:, 0] <= t1)
        q = ref[m, 1:]
        p = np.column_stack([np.interp(ref[m, 0], tr[:, 0], tr[:, 1]),
                             np.interp(ref[m, 0], tr[:, 0], tr[:, 2])])
        rot = kabsch_angle(p, q)
        res = np.linalg.norm(p - q, axis=1)
        med, p99, mx = np.median(res), np.percentile(res, 99), res.max()
        g = abs(rot) <= ROT_TOL_DEG and med <= MED_TOL_M and p99 <= P99_TOL_M
        ok &= g
        print(f"  {name:<12} {m.sum():>7} {rot:>8.4f}° {med:>10.5f}m "
              f"{p99:>7.4f}m {mx:>7.4f}m  {'✓' if g else '✗ FAIL'}")
    return ok


def ate_origine(est, gxy):
    """⛔1 : translation pure, les deux départs à (0,0), AUCUNE rotation."""
    return calculer_ate(est - est[0], gxy - gxy[0])


def gate2_ate(dirs):
    """ATE origine global + sections (tiers temporels ré-épinglés) par méthode."""
    res = {}
    print("\n── Gate ② ATE origine translation pure (global · S1/S2/S3 · odom)")
    for name, rd in zip(PRESETS, dirs):
        d = charger(rd)
        a = ate_origine(d["est"], d["gxy"])
        tb = np.linspace(d["t"][0], d["t"][-1], 4)
        secs = []
        for k in range(3):
            m = (d["t"] >= tb[k]) & (d["t"] <= tb[k + 1])
            secs.append(ate_origine(d["est"][m], d["gxy"][m]))
        a_odom = ate_origine(d["dr"], d["gxy"])
        res[name] = (a, secs, a_odom, d)
        print(f"  {name:<12} ATE {a:6.2f} m | S "
              + "/".join(f"{s:.2f}" for s in secs) + f" | odom {a_odom:5.2f} m"
              f" | {os.path.basename(rd)}")
    return res


def gate3_ordre(res):
    a = {k: v[0] for k, v in res.items()}
    checks = [("BSU ≤ Bruce_U", a["bsu"], a["bruce_u"], True),
              ("Bruce_U < Bruce", a["bruce_u"], a["bruce"], False),
              ("BSU < Bruce", a["bsu"], a["bruce"], False)]
    verdict = "PASS"
    print("\n── Gate ③ ordre attendu : BSU ≤ BU < Bruce (BS hors verdict, amendé 16-07)")
    print(f"  ℹ Bruce_Sonar (finding SC-sans-ancre) : ATE {a['bruce_sonar']:.2f} m, "
          f"attendu ≈ odométrie {res['bruce_sonar'][2]:.2f} m")
    for label, lhs, rhs, or_eq in checks:
        d = rhs - lhs
        if (lhs <= rhs if or_eq else lhs < rhs):
            print(f"  ✓ {label:<22} (marge {d:+.2f} m)")
        elif lhs - rhs <= VAR_ICP:
            print(f"  ⚠ {label:<22} VIOLÉ de {lhs - rhs:.2f} m ≤ variance ICP "
                  f"{VAR_ICP} → LIMITE : re-run ×2 avant verdict (R3)")
            verdict = "LIMITE" if verdict == "PASS" else verdict
        else:
            print(f"  ✗ {label:<22} VIOLÉ de {lhs - rhs:.2f} m → FAIL, STOP (R2)")
            verdict = "FAIL"
    return verdict


def gate4_carte(res):
    print("\n── Gate ④ carte : cloud vs cloud-GT (méd/p90, Umeyama interne) — informatif")
    for name, (_, _, _, d) in res.items():
        try:
            _, R1, t1 = umeyama(d["est"], d["gxy"], with_scale=False)
            _, s_cap, beta = fit_cap(d["g_th"], d["th"])
            P, V = cloud_vrai(d, R1, t1, s_cap, beta)
            idx = np.random.default_rng(0).choice(len(P), min(60000, len(P)),
                                                  replace=False)
            dm, _ = cKDTree(V).query(P[idx], k=1)
            print(f"  {name:<12} méd {np.median(dm):5.2f} m | p90 "
                  f"{np.percentile(dm, 90):5.2f} m ({len(P)} pts)")
        except Exception as e:  # cloud absent → informatif, pas bloquant
            print(f"  {name:<12} indisponible ({e})")


def main():
    dirs = sys.argv[1:5] if len(sys.argv) >= 5 else derniers_runs()
    print("Runs évalués :")
    for p, d in zip(PRESETS, dirs):
        print(f"  {p:<12} {d}")
    g1 = gate1_dr(dirs)
    res = gate2_ate(dirs)
    g3 = gate3_ordre(res)
    gate4_carte(res)
    verdict = "FAIL" if (not g1 or g3 == "FAIL") else g3
    print(f"\n════ VERDICT GATES : {verdict}"
          + (" (gate ① dr non identiques → odométrie touchée, run invalide ⛔2)"
             if not g1 else ""))
    sys.exit({"PASS": 0, "LIMITE": 2, "FAIL": 1}[verdict])


if __name__ == "__main__":
    main()
