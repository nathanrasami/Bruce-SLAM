#!/usr/bin/env python3
"""ÉTAPE 0 de REFONTE_MISSION.md — trancher le cap de seed par MESURE (offline).

Candidats GT-free :
  (a) cap fixe 0 ;
  (b) route-fond USBL longue base (PIEGES #25 : base >= 20-30 m). ⚠ Mesuré ici :
      le ROV tourne de ~-88° (cmd_vel) pendant les 30 premiers mètres → la course
      USBL (~-76°) n'est PAS le cap initial. L'implémentation correcte de (b) est
      un fit de FORME 1-DOF (Kabsch 2D) : θ0 = rotation qui recale la trajectoire
      cmd_vel intégrée (θ0=0) sur les fixes USBL de la fenêtre. GT-free (cmd_vel
      + USBL seuls), calculé OFFLINE → l'odométrie ne s'abonne jamais à l'USBL.

Validation a posteriori : θ0_true = même fit Kabsch contre la DGPS (/pose_gt).
La GT sert UNIQUEMENT à valider le choix, jamais à seeder.

Sortie : par base D (10/20/30/40 m) — courses (contexte), θ0_true, θ0_b,
err(a)=|θ0_true|, err(b)=|θ0_b-θ0_true|. À lancer dans le conteneur ros1 :
  python3 analysis/etape0_seed_cap.py [bag]
"""
import math
import sys

import numpy as np
import rosbag

BAG = sys.argv[1] if len(sys.argv) > 1 else "ARACATI_2017_8bits_full.bag"
BASES = [10.0, 20.0, 30.0, 40.0]
USBL_MAX_SPEED = 3.0  # m/s — gate anti-glitch (~73 m), même valeur que le node


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def course_endpoint(xy):
    d = xy[-1] - xy[0]
    return math.degrees(math.atan2(d[1], d[0]))


def course_pca(xy):
    """Direction principale (total least squares), signée par le déplacement net."""
    c = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    v = vt[0]
    d = xy[-1] - xy[0]
    if np.dot(v, d) < 0:
        v = -v
    return math.degrees(math.atan2(v[1], v[0]))


def kabsch_angle(p, q):
    """Rotation 2D (deg) du fit rigide p→q (rotation+translation, moindres carrés)."""
    pc, qc = p - p.mean(axis=0), q - q.mean(axis=0)
    num = np.sum(pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0])
    den = np.sum(pc[:, 0] * qc[:, 0] + pc[:, 1] * qc[:, 1])
    return math.degrees(math.atan2(num, den))


gt, usbl, wz_int = [], [], []  # (t,x,y) · (t,x,y) gaté · (t, intégrale wz depuis t0)
rel = []                       # (t,x,y) cmd_vel intégré unicycle, θ0=0, départ (0,0)
wz_sum, last_cmd_t = 0.0, None
rx = ry = rth = 0.0
bag = rosbag.Bag(BAG)
for topic, msg, t in bag.read_messages(topics=["/pose_gt", "/usbl_point", "/cmd_vel"]):
    ts = msg.header.stamp.to_sec()
    if topic == "/pose_gt":
        gt.append((ts, msg.pose.position.x, msg.pose.position.y))
        if math.hypot(gt[-1][1] - gt[0][1], gt[-1][2] - gt[0][2]) > max(BASES) + 15:
            break
    elif topic == "/usbl_point":
        x, y = msg.point.x, msg.point.y
        if usbl:
            lt, lx, ly = usbl[-1]
            dt = ts - lt
            if dt > 0 and math.hypot(x - lx, y - ly) / dt > USBL_MAX_SPEED:
                continue
        usbl.append((ts, x, y))
    else:  # /cmd_vel — même intégration Euler avant que cmd_vel_odom.py
        if last_cmd_t is not None and ts > last_cmd_t:
            dt = ts - last_cmd_t
            wz_sum += msg.twist.angular.z * dt
            rx += msg.twist.linear.x * math.cos(rth) * dt
            ry += msg.twist.linear.x * math.sin(rth) * dt
            rth += msg.twist.angular.z * dt
        last_cmd_t = ts
        wz_int.append((ts, wz_sum))
        rel.append((ts, rx, ry))
bag.close()

gt = np.array(gt)
usbl = np.array(usbl)
wz_int = np.array(wz_int)
rel = np.array(rel)
print(f"lu : {len(gt)} poses GT, {len(usbl)} fixes USBL gatés, "
      f"{len(rel)} cmd_vel, t0_gt={gt[0,0]:.1f}")
print(f"1er fix USBL : ({usbl[0,1]:.2f},{usbl[0,2]:.2f}) à t0+{usbl[0,0]-gt[0,0]:.1f}s ; "
      f"GT t0 : ({gt[0,1]:.2f},{gt[0,2]:.2f})")


def rel_at(times):
    return np.column_stack([np.interp(times, rel[:, 0], rel[:, 1]),
                            np.interp(times, rel[:, 0], rel[:, 2])])


print(f"\n{'base':>5} | {'GT pca':>7} {'USBL pca':>8} {'rot cmdvel':>10} | "
      f"{'θ0_true':>8} {'θ0_b':>7} {'N':>4} | {'err(a)':>7} {'err(b)':>7} | {'err(b naïf)':>11}")
for D in BASES:
    d_gt = np.hypot(gt[:, 1] - gt[0, 1], gt[:, 2] - gt[0, 2])
    i_gt = np.argmax(d_gt >= D)
    if i_gt == 0:
        print(f"{D:5.0f} | base non atteinte par la GT lue")
        continue
    g = gt[: i_gt + 1]
    d_us = np.hypot(usbl[:, 1] - usbl[0, 1], usbl[:, 2] - usbl[0, 2])
    i_us = np.argmax(d_us >= D)
    u = usbl[: i_us + 1] if i_us > 0 else usbl[usbl[:, 0] <= g[-1, 0]]
    cg_p, cu_p = course_pca(g[:, 1:]), course_pca(u[:, 1:])
    rot = math.degrees(wz_int[np.searchsorted(wz_int[:, 0], u[-1, 0], "right") - 1, 1]
                       - wz_int[np.searchsorted(wz_int[:, 0], u[0, 0], "right") - 1, 1])
    # θ0 = rotation du fit rigide (trajectoire cmd_vel θ0=0) → (référence)
    th_true = kabsch_angle(rel_at(g[:, 0]), g[:, 1:])        # vs DGPS (validation)
    th_b = kabsch_angle(rel_at(u[:, 0]), u[:, 1:])           # vs USBL (candidat b)
    print(f"{D:5.0f} | {cg_p:7.1f} {cu_p:8.1f} {rot:9.1f}° | "
          f"{th_true:8.1f} {th_b:7.1f} {len(u):4d} | {abs(wrap_deg(th_true)):7.1f} "
          f"{abs(wrap_deg(th_b-th_true)):7.1f} | {abs(wrap_deg(cu_p-th_true)):11.1f}")

print("\nθ0_true = Kabsch(cmd_vel θ0=0 → DGPS) : le cap de seed idéal (validation GT).")
print("θ0_b    = Kabsch(cmd_vel θ0=0 → USBL) : candidat (b), 100% GT-free, offline.")
print("err(a) = |θ0_true| (coût cap 0) ; err(b) = |θ0_b-θ0_true| ; "
      "err(b naïf) = coût du seed atan2/course historique. ~1.6 m ATE/° (PIEGES #25).")
