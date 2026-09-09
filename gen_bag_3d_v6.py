#!/usr/bin/env python3
"""Generateur traj6 « tout capter » : traj5 (errance profil collegue, VALIDEE
run 222233) + 3e capteur = ProfilingSonar TRANSVERSE 360° (FABLE §10-bis,
accord Nathan). La TRAJECTOIRE et les 2 premiers capteurs sont STRICTEMENT
ceux de traj5 (import gen_bag_3d_v5) : phase A calibration inchangee
(E2/E4/E6/E7, memes fenetres), errance PCHIP seed 42 inchangee.

3e capteur (« tout capter » : flancs/haut/bas au passage) :
  ProfilingSonar 360°, rotation HoloOcean [90, 0, 90], RangeMax 20 m,
  512 range-bins x 720 azimut-bins (0.5°/bin), Hz 2 (0.35 m/s -> une section
  transverse tous les 17.5 cm, < voxel 0.2 m de carte_3d).
  -> /profiler_points : points 3D repere VEHICULE (frame auv0), comme
     /sonar_vert_points ; source « transverse » de carte_3d (fusion vert+trans).
  -> /profiler : image polaire 1 frame sur 5 (0.4 Hz, debug seulement).

MONTAGE MESURE (probe 2026-07-12, robot statique (525,-662,-5) cap sud,
confirme x2) : R_MOUNT_TRANS = Rz(90) @ Rx(+90) — x_capteur -> +y vehicule
(BABORD), y_capteur -> +z (HAUT). ⚠ Ce n'est PAS l'analogie du montage
vertical (R_MOUNT_PROF = Rx(-90)) : le candidat Rz(90)@Rx(-90) mettait le
fond AU-DESSUS du robot (flipZ gagnant au probe). Preuve : mur quai EST a
x_med 532.1 (attendu 531.5, babord) et fond z_med -19.7 (80.9 % dans
[-21,-17]) avec ce mount ; flipY et flipZ echouent chacun. Verrouille par le
check E9 (check_traj4.py) sur chaque bag genere.

Usage : python gen_bag_3d_v6.py [--test 150] [--seed 42]
"""
import os
import sys
import numpy as np
import holoocean

import gen_bag_3d_v5 as v5
from gen_bag_3d_v5 import (LEVER, SONAR_HZ, DT, SEED, verifier_chemin)
from gen_bag_3d import (typestore, Time, Header, Vector3, Quaternion, Imu,
                        TwistStamped, Twist, Image, Odometry,
                        PoseWithCovariance, Pose, Point, TwistWithCovariance,
                        Float64, PointCloud2,
                        rpy_to_quat, R_from_rpy, sonar_to_points3d_msg,
                        R_MOUNT_PROF, RANGE_MIN,
                        SIGMA_GYRO, SIGMA_ACC, SIGMA_DVL, SIGMA_DEPTH)
from rosbags.rosbag1 import Writer
import noise_round2 as _NOISE          # round 2 « noise » (NOISE_ROUND2=1)

# ─── 3e capteur : profiler TRANSVERSE 360° ────────────────────────────────────
PROF_HZ         = 2                 # 0.35 m/s -> section tous les 17.5 cm
PROF_RANGE_MAX  = 20.0
PROF_RANGE_BINS = 512
PROF_AZ_BINS    = 720               # 0.5°/bin
PROF_IMG_DECIM  = 5                 # /profiler (image debug) a 0.4 Hz
OUT_BAG = "BAG_files/holoocean_3d_traj6.bag"

# Montage capteur->vehicule MESURE (probe 2026-07-12, confirme x2, cf. header).
R_MOUNT_TRANS = np.array([[0., 0., 1.],
                          [1., 0., 0.],
                          [0., 1., 0.]])


def make_cfg_v6():
    cfg = v5.make_cfg()
    cfg["name"] = "gen3dv6"
    common = {"RangeMin": RANGE_MIN, "RangeMax": PROF_RANGE_MAX,
              # L1 round 2 : profiler transverse 0.01 -> 0.05 (x5) si NOISE_ROUND2=1
              "AddSigma": _NOISE.SONAR_ADD, "MultSigma": _NOISE.SONAR_MULT,
              "RangeSigma": 0,
              "MultiPath": False, "AzimuthStreaks": 0, "ScaleNoise": False,
              "InitOctreeRange": 50, "ViewRegion": False}
    cfg["agents"][0]["sensors"].append(
        {"sensor_type": "ProfilingSonar", "sensor_name": "ProfilerTrans",
         "socket": "SonarSocket", "location": [float(v) for v in LEVER],
         "rotation": [90.0, 0.0, 90.0],
         "Hz": PROF_HZ,
         "configuration": dict(common, Azimuth=360, Elevation=1,
                               RangeBins=PROF_RANGE_BINS,
                               AzimuthBins=PROF_AZ_BINS)})
    return cfg


def main():
    args = sys.argv[1:]
    test = float(args[args.index("--test") + 1]) if "--test" in args else None
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else SEED
    if seed != SEED:
        v5._init_traj(seed)
    verifier_chemin()

    out = OUT_BAG if test is None else OUT_BAG.replace(".bag", "_test.bag")
    if os.path.exists(out):
        os.remove(out)
    os.makedirs("BAG_files", exist_ok=True)
    T = test if test else v5.T_TOTAL
    print(f"TRAJ6 | traj5 (errance SEED={seed}) + profiler TRANSVERSE 360° "
          f"{PROF_RANGE_MAX:.0f} m {PROF_AZ_BINS} bins @ {PROF_HZ} Hz | "
          f"duree={T/60:.1f} min (totale {v5.T_TOTAL/60:.1f}) | bag={out}")

    rng = np.random.default_rng(0)
    t, next_sonar, n_pings, n_prof = 0.0, 0.0, 0, 0
    zero3 = np.zeros(3)

    # show_viewport=False : crashs GPU NVRM Xid 13 sinon (mesure traj4)
    with Writer(out) as bag, holoocean.make(scenario_cfg=make_cfg_v6(),
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
            q = rpy_to_quat(*rpy)
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
            gyro = w_b + rng.normal(0, SIGMA_GYRO, 3)
            acc = R.T @ np.array([0, 0, 9.81]) + rng.normal(0, SIGMA_ACC, 3)
            imu = Imu(header=hv,
                orientation=Quaternion(q[0], q[1], q[2], q[3]),
                orientation_covariance=np.zeros(9),
                angular_velocity=Vector3(*gyro),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=Vector3(*acc),
                linear_acceleration_covariance=np.zeros(9))
            bag.write(c_imu, t_ns, typestore.serialize_ros1(imu, Imu.__msgtype__))

            v_b = R.T @ v_w + rng.normal(0, SIGMA_DVL, 3)
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

                pc = sonar_to_points3d_msg(img, zero3, (0., 0., 0.), hv)
                bag.write(c_pts, t_ns, typestore.serialize_ros1(pc, PointCloud2.__msgtype__))

                if "SonarVert" in state:
                    vimg = np.asarray(state["SonarVert"], dtype=np.float32)
                    vim = Image(header=hv, height=vimg.shape[0], width=vimg.shape[1],
                                encoding='32FC1', is_bigendian=0,
                                step=vimg.shape[1] * 4,
                                data=np.frombuffer(vimg.tobytes(), dtype=np.uint8))
                    bag.write(c_sv, t_ns, typestore.serialize_ros1(vim, Image.__msgtype__))
                    vpc = sonar_to_points3d_msg(vimg, LEVER, (0., 0., 0.), hv,
                                                r_mount=R_MOUNT_PROF,
                                                azimuth_deg=120,
                                                range_max=v5.VERT_RANGE_MAX)
                    bag.write(c_svp, t_ns, typestore.serialize_ros1(vpc, PointCloud2.__msgtype__))
                next_sonar += 1.0 / SONAR_HZ
                n_pings += 1
                if n_pings % 100 == 0:
                    seg = v5._SEGS5[min(int(np.searchsorted(v5._EDGES, t, side='right')) - 1,
                                        len(v5._SEGS5) - 1)][2]
                    print(f"  t={t:6.1f}/{T:.0f} s | pings={n_pings} | "
                          f"prof={n_prof} | z={p[2]:+.2f} | {seg}")

            # profiler transverse 360° : points a chaque frame (2 Hz),
            # image polaire 1 frame sur PROF_IMG_DECIM (debug)
            if "ProfilerTrans" in state:
                pimg = np.asarray(state["ProfilerTrans"], dtype=np.float32)
                ppc = sonar_to_points3d_msg(pimg, LEVER, (0., 0., 0.), hv,
                                            r_mount=R_MOUNT_TRANS,
                                            azimuth_deg=360,
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
          f"SEED={seed} | duree {min(T, v5.T_TOTAL):.0f} s")

if __name__ == "__main__":
    main()
