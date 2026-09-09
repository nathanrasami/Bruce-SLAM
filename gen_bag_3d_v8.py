#!/usr/bin/env python3
"""Generateur traj7r = traj7 (circuit/capteurs sonar IDENTIQUES, import v7)
+ NAVIGATION REALISTE (demande Nathan 2026-07-12 « on est trop precis la »).

Diagnostic (runs 191325/200450) : SLAM = DR bit-a-bit (0 facteur sonar) et le
DR simule est irrealiste — compas PARFAIT (dvl_imu_odom lit l'orientation IMU
absolue) + DVL sans biais -> derive 0.14 m / 24.6 min. La plus-value trajectoire
du SLAM sonar est indemontrable : rien a corriger.

Modele d'erreurs v8 (injecte dans le BAG ; chaine SLAM intacte) :
  CAP (dominant)  : psi_err(t) = PSI0 + marche aleatoire.
                    PSI0 = 1.0 deg (biais compas typique pres de structures
                    acier / residu de declinaison), RW_SIG = 0.05 deg/sqrt(s)
                    (~1.9 deg sigma en fin de run). Applique a l'ORIENTATION
                    IMU seulement (GT intact). Le gyro (inutilise par
                    dvl_imu_odom) garde son bruit blanc d'origine.
  DVL             : v_mes = (1+DVL_SCALE) * Rz(DVL_MIS) * v_vrai + bruit.
                    DVL_SCALE = +0.3 % (erreur d'echelle constructeur typique
                    0.2-0.5 %), DVL_MIS = 0.5 deg (desalignement de monture),
                    bruit SIGMA_DVL inchange. Bottom-lock -> PAS de terme de
                    courant (le DVL mesure par rapport au fond).
  DEPTH, SONARS, GT : inchanges (E1-E9 restent valides tels quels).

Ordre de grandeur attendu (verifie a sec par verifier_nav_v8, seed 7) :
derive DR de l'ordre du metre — assez pour que les loops sonar aient de la
matiere, pas absurde. Les erreurs RIGIDES (biais pur) sont en partie absorbees
par l'alignement Umeyama : c'est la marche aleatoire qui porte l'ATE.

Usage : python gen_bag_3d_v8.py [--test 150] [--seed 42]
"""
import os
import sys
import numpy as np
import holoocean

import gen_bag_3d_v7 as v7            # applique les patchs circuit/RangeMax 20
import gen_bag_3d_v5 as v5
import gen_bag_3d_v6 as v6
from gen_bag_3d import (typestore, Time, Header, Vector3, Quaternion, Imu,
                        TwistStamped, Twist, Image, Odometry,
                        PoseWithCovariance, Pose, Point, TwistWithCovariance,
                        Float64, PointCloud2,
                        rpy_to_quat, R_from_rpy, R_MOUNT_PROF,
                        SIGMA_GYRO, SIGMA_ACC, SIGMA_DVL, SIGMA_DEPTH)
from gen_bag_3d_v6 import R_MOUNT_TRANS, PROF_IMG_DECIM, PROF_RANGE_MAX
from rosbags.rosbag1 import Writer
import noise_round2 as _NOISE          # round 2 « noise » (NOISE_ROUND2=1)

OUT_BAG = "BAG_files/holoocean_3d_traj7r.bag"
# L3 round 2 : erreurs nav structurees x2 si NOISE_ROUND2=1 (traj9 SEUL ; traj5
# n'applique pas L3). Mesure a sec : derive DR 5.76 -> 11.52 m rms (Umeyama 2.30).
PSI0_DEG   = 2.0   * _NOISE.L3_MULT   # biais de cap constant (compas magnetique)
RW_SIG_DEG = 0.15  * _NOISE.L3_MULT   # marche aleatoire du cap, deg/sqrt(s)
DVL_SCALE  = 0.005 * _NOISE.L3_MULT   # erreur d'echelle DVL (+0.5 % x mult)
DVL_MIS_DEG = 0.5  * _NOISE.L3_MULT   # desalignement yaw de la monture DVL
SEED_NAV   = 7       # rng DEDIE aux erreurs nav (bruits capteurs : seed 0, inchange)
# Cible : derive DR ~0.5-0.8 % de la distance parcourue (~530 m) = 2-4 m —
# la fourchette d'un DVL + AHRS magnetique reel en environnement portuaire
# (le premier tirage a 1deg/0.05 donnait 0.42 m rms = 0.08 % DT, trop optimiste).


def psi_err_series(T, dt, rng):
    """Serie temporelle du cap errone : biais + marche aleatoire (rad)."""
    n = int(np.ceil(T / dt)) + 1
    rw = np.concatenate([[0.0], np.cumsum(
        rng.normal(0.0, np.deg2rad(RW_SIG_DEG) * np.sqrt(dt), n - 1))])
    return np.deg2rad(PSI0_DEG) + rw


def dvl_mismount():
    c, s = np.cos(np.deg2rad(DVL_MIS_DEG)), np.sin(np.deg2rad(DVL_MIS_DEG))
    return (1.0 + DVL_SCALE) * np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def verifier_nav_v8(T):
    """Predit A SEC la derive DR (integration planaire IDENTIQUE a
    dvl_imu_odom : p += R(yaw_mesure) @ v_mesuree * dt) et l'ATE vs GT."""
    dt = v5.DT
    rng = np.random.default_rng(SEED_NAV)
    psi = psi_err_series(T, dt, rng)
    rngs = np.random.default_rng(0)          # meme role que le rng capteurs
    M = dvl_mismount()
    p_dr = np.zeros(2)
    P_dr, P_gt = [], []
    for i, t in enumerate(np.arange(0.0, T, dt)):
        p, rpy, v_w, _ = v5.pose_at_v5(t)
        R = R_from_rpy(*rpy)
        v_mes = M @ (R.T @ v_w) + rngs.normal(0, SIGMA_DVL, 3)
        yaw_mes = rpy[2] + psi[i]
        c, s = np.cos(yaw_mes), np.sin(yaw_mes)
        p_dr = p_dr + np.array([c * v_mes[0] - s * v_mes[1],
                                s * v_mes[0] + c * v_mes[1]]) * dt
        P_dr.append(p_dr.copy()); P_gt.append(p[:2] - v5.pose_at_v5(0.0)[0][:2])
    P, Q = np.array(P_dr), np.array(P_gt)
    e_anc = np.linalg.norm(P - Q, axis=1)
    Pm, Qm = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pm.T @ Qm)
    D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    e_um = np.linalg.norm((Vt.T @ D @ U.T @ Pm.T).T + Q.mean(0) - Q, axis=1)
    rms = lambda e: float(np.sqrt((e ** 2).mean()))
    print(f"nav v8 (a sec, {T/60:.1f} min) : derive DR ancree rms "
          f"{rms(e_anc):.2f} m (max {e_anc.max():.2f}) | Umeyama rms "
          f"{rms(e_um):.2f} m (max {e_um.max():.2f}) | psi fin "
          f"{np.degrees(psi[len(P)-1]):+.2f} deg")
    if T > 600:            # bande visee sur le bag COMPLET seulement
        # round 2 : borne haute elargie a NAV_DRIFT_HI (30 m) car le x2 L3 vise
        # ~11.5 m rms (mesure a sec) -> hors [1,8] ; round 1 garde [1,8].
        assert 1.0 < rms(e_anc) < _NOISE.NAV_DRIFT_HI, \
            f"derive DR hors bande visee [1, {_NOISE.NAV_DRIFT_HI:.0f}] m"
    return rms(e_anc)


def main():
    args = sys.argv[1:]
    test = float(args[args.index("--test") + 1]) if "--test" in args else None
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else v5.SEED
    v5._init_traj(seed)
    v7.verifier_chemin_v7()
    T = test if test else v5.T_TOTAL
    verifier_nav_v8(T)

    out = OUT_BAG if test is None else OUT_BAG.replace(".bag", "_test.bag")
    if os.path.exists(out):
        os.remove(out)
    os.makedirs("BAG_files", exist_ok=True)
    print(f"TRAJ7R | traj7 + nav realiste (cap {PSI0_DEG}deg+RW {RW_SIG_DEG}deg/sqrt(s), "
          f"DVL +{100*DVL_SCALE:.1f}% mis {DVL_MIS_DEG}deg) | duree={T/60:.1f} min | bag={out}")

    rng = np.random.default_rng(0)            # bruits capteurs (inchange v6)
    rng_nav = np.random.default_rng(SEED_NAV)
    psi = psi_err_series(T, v5.DT, rng_nav)
    M_dvl = dvl_mismount()
    DT, SONAR_HZ, LEVER = v5.DT, v5.SONAR_HZ, v5.LEVER
    sonar_pts = v6.sonar_to_points3d_msg      # partial range_max=20 pose par v7
    t, next_sonar, n_pings, n_prof = 0.0, 0.0, 0, 0
    zero3 = np.zeros(3)

    with Writer(out) as bag, holoocean.make(scenario_cfg=v6.make_cfg_v6(),
                                            show_viewport=False) as env:
        c_imu = bag.add_connection('/imu', Imu.__msgtype__, typestore=typestore)
        c_dvl = bag.add_connection('/dvl', TwistStamped.__msgtype__, typestore=typestore)
        c_son = bag.add_connection('/sonar', Image.__msgtype__, typestore=typestore)
        c_pts = bag.add_connection('/sonar_points', PointCloud2.__msgtype__, typestore=typestore)
        c_gt = bag.add_connection('/ground_truth', Odometry.__msgtype__, typestore=typestore)
        c_dep = bag.add_connection('/depth', Float64.__msgtype__, typestore=typestore)
        c_sv = bag.add_connection('/sonar_vert', Image.__msgtype__, typestore=typestore)
        c_svp = bag.add_connection('/sonar_vert_points', PointCloud2.__msgtype__, typestore=typestore)
        c_pr = bag.add_connection('/profiler', Image.__msgtype__, typestore=typestore)
        c_prp = bag.add_connection('/profiler_points', PointCloud2.__msgtype__, typestore=typestore)

        agent = env.agents["auv0"]
        while t < T:
            p, rpy, v_w, w_b = v5.pose_at_v5(t)
            agent.teleport(location=p.tolist(), rotation=list(np.degrees(rpy)))
            state = env.tick()

            sim_t = t + 1.0
            sec = int(sim_t); ns = int((sim_t - sec) * 1e9)
            stamp = Time(sec=sec, nanosec=ns)
            t_ns = sec * 1_000_000_000 + ns
            seq = int(round(t / DT))
            q = rpy_to_quat(*rpy)                       # GT : vrai
            R = R_from_rpy(*rpy)

            h = Header(seq=seq, stamp=stamp, frame_id='map')
            gt = Odometry(header=h, child_frame_id='auv0',
                pose=PoseWithCovariance(pose=Pose(
                    position=Point(p[0], p[1], p[2]),
                    orientation=Quaternion(q[0], q[1], q[2], q[3])),
                    covariance=np.zeros(36)),
                twist=TwistWithCovariance(twist=Twist(
                    linear=Vector3(*v_w), angular=Vector3(0., 0., 0.)),
                    covariance=np.zeros(36)))
            bag.write(c_gt, t_ns, typestore.serialize_ros1(gt, Odometry.__msgtype__))

            hv = Header(seq=seq, stamp=stamp, frame_id='auv0')
            # v8 : cap IMU errone (biais + RW) — seule l'ORIENTATION change
            q_mes = rpy_to_quat(rpy[0], rpy[1], rpy[2] + psi[seq])
            gyro = w_b + rng.normal(0, SIGMA_GYRO, 3)
            acc = R.T @ np.array([0, 0, 9.81]) + rng.normal(0, SIGMA_ACC, 3)
            imu = Imu(header=hv,
                orientation=Quaternion(q_mes[0], q_mes[1], q_mes[2], q_mes[3]),
                orientation_covariance=np.zeros(9),
                angular_velocity=Vector3(*gyro),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=Vector3(*acc),
                linear_acceleration_covariance=np.zeros(9))
            bag.write(c_imu, t_ns, typestore.serialize_ros1(imu, Imu.__msgtype__))

            # v8 : DVL avec echelle + desalignement
            v_b = M_dvl @ (R.T @ v_w) + rng.normal(0, SIGMA_DVL, 3)
            dvl = TwistStamped(header=hv, twist=Twist(
                linear=Vector3(*v_b), angular=Vector3(0., 0., 0.)))
            bag.write(c_dvl, t_ns, typestore.serialize_ros1(dvl, TwistStamped.__msgtype__))

            d = Float64(data=float(-p[2] + rng.normal(0, SIGMA_DEPTH)))
            bag.write(c_dep, t_ns, typestore.serialize_ros1(d, Float64.__msgtype__))

            if t >= next_sonar and "SonarFin" in state:
                img = np.asarray(state["SonarFin"], dtype=np.float32)
                im = Image(header=hv, height=img.shape[0], width=img.shape[1],
                           encoding='32FC1', is_bigendian=0, step=img.shape[1] * 4,
                           data=np.frombuffer(img.tobytes(), dtype=np.uint8))
                bag.write(c_son, t_ns, typestore.serialize_ros1(im, Image.__msgtype__))
                pc = sonar_pts(img, zero3, (0., 0., 0.), hv)
                bag.write(c_pts, t_ns, typestore.serialize_ros1(pc, PointCloud2.__msgtype__))
                if "SonarVert" in state:
                    vimg = np.asarray(state["SonarVert"], dtype=np.float32)
                    vim = Image(header=hv, height=vimg.shape[0], width=vimg.shape[1],
                                encoding='32FC1', is_bigendian=0,
                                step=vimg.shape[1] * 4,
                                data=np.frombuffer(vimg.tobytes(), dtype=np.uint8))
                    bag.write(c_sv, t_ns, typestore.serialize_ros1(vim, Image.__msgtype__))
                    vpc = sonar_pts(vimg, LEVER, (0., 0., 0.), hv,
                                    r_mount=R_MOUNT_PROF, azimuth_deg=120,
                                    range_max=v5.VERT_RANGE_MAX)
                    bag.write(c_svp, t_ns, typestore.serialize_ros1(vpc, PointCloud2.__msgtype__))
                next_sonar += 1.0 / SONAR_HZ
                n_pings += 1
                if n_pings % 100 == 0:
                    print(f"  t={t:6.1f}/{T:.0f} s | pings={n_pings} | prof={n_prof} "
                          f"| z={p[2]:+.2f} | psi_err={np.degrees(psi[seq]):+.2f} deg")

            if "ProfilerTrans" in state:
                pimg = np.asarray(state["ProfilerTrans"], dtype=np.float32)
                ppc = sonar_pts(pimg, LEVER, (0., 0., 0.), hv,
                                r_mount=R_MOUNT_TRANS, azimuth_deg=360,
                                range_max=PROF_RANGE_MAX)
                bag.write(c_prp, t_ns, typestore.serialize_ros1(ppc, PointCloud2.__msgtype__))
                if n_prof % PROF_IMG_DECIM == 0:
                    pim = Image(header=hv, height=pimg.shape[0], width=pimg.shape[1],
                                encoding='32FC1', is_bigendian=0,
                                step=pimg.shape[1] * 4,
                                data=np.frombuffer(pimg.tobytes(), dtype=np.uint8))
                    bag.write(c_pr, t_ns, typestore.serialize_ros1(pim, Image.__msgtype__))
                n_prof += 1
            t += DT

    print(f"\nbag ecrit : {out} ({n_pings} pings, {n_prof} sections transverses) | "
          f"nav realiste seed {SEED_NAV} | duree {min(T, v5.T_TOTAL):.0f} s")

if __name__ == "__main__":
    main()
