#!/usr/bin/env python3
"""Genere un bag ROS1 **3D** depuis HoloOcean — HOLOOCEAN_3D_GUIDE.md **v2** :
robot A PLAT (roll=0 partout, « grande route »), depart DANS la structure,
profondeur sinusoidale le long du chemin carre (std(z) trajectoire > 1 m),
capteurs synthetiques analytiques, ecriture via `rosbags` (pas de ROS).

  TRAJ 1 (pseudo-3D) : le z vient de la trajectoire ; /sonar_points reste des
      tranches ~horizontales (2.5D assume, c'est le contrat §2).
  TRAJ 2 (vraie 3D)  : meme trajectoire + ProfilingSonar monte VERTICAL ->
      /profiler_points = sections verticales (§3, option A).

Usage :
    python gen_bag_3d.py --traj 1             # bag complet traj 1 (~10 min sim)
    python gen_bag_3d.py --traj 2             # bag complet traj 2
    python gen_bag_3d.py --traj 1 --test 60   # bag court 60 s (validation)
"""
import sys
import numpy as np
import holoocean

from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.ROS1_NOETIC)
Time = typestore.types['builtin_interfaces/msg/Time']
Header = typestore.types['std_msgs/msg/Header']
Vector3 = typestore.types['geometry_msgs/msg/Vector3']
Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
Imu = typestore.types['sensor_msgs/msg/Imu']
TwistStamped = typestore.types['geometry_msgs/msg/TwistStamped']
Twist = typestore.types['geometry_msgs/msg/Twist']
Image = typestore.types['sensor_msgs/msg/Image']
Odometry = typestore.types['nav_msgs/msg/Odometry']
PoseWithCovariance = typestore.types['geometry_msgs/msg/PoseWithCovariance']
Pose = typestore.types['geometry_msgs/msg/Pose']
Point = typestore.types['geometry_msgs/msg/Point']
TwistWithCovariance = typestore.types['geometry_msgs/msg/TwistWithCovariance']
PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
PointField = typestore.types['sensor_msgs/msg/PointField']
Float64 = typestore.types['std_msgs/msg/Float64']

# ─── PARAMETRES TRAJECTOIRE v2 (§1-§2) ────────────────────────────────────────
# LIGNES MEDIANES du couloir mesurees AU SONAR (sonde 4 cotes, 07-07, apres
# re-cook) : echos symetriques des deux cotes depuis ces lignes :
#   sud y=-1.7 (2.4/2.4 m) | nord y=30.7 (2.8/2.8) | est x=0.3 | ouest x=-29.9
# -> rectangle 30.2 x 32.4 m, clearances 2.2-2.8 m partout (>= 1.5 requis).
# NB : l'anneau MANUEL longeait le bloc a ~0.7 m (zone aveugle RangeMin !) —
# c'est pour ca que le bloc paraissait muet ; la mediane regle le probleme.
CX, CY   = -14.8, 14.5   # centre du rectangle median
LX, LY   = 30.2, 32.4    # cotes du CHEMIN (m) = medianes du couloir
R_TURN   = 4.0    # rayon des virages (m)
Z0       = -3.5   # profondeur moyenne (m, z<0)
Z_AMP    = 2.0    # amplitude sinusoide profondeur (std = A/sqrt(2) = 1.41 m > 1)
N_ZCYC   = 3      # cycles de profondeur par tour (entier -> boucle fermee exacte)
V_FWD    = 0.35   # vitesse d'avance (m/s)
DT       = 0.05   # pas de temps simule (s) -> ticks_per_sec = 20
SONAR_HZ = 5.0
N_LAPS   = 2
OUT_BAG  = "BAG_files/holoocean_3d.bag"   # suffixe _traj1/_traj2 ajoute

# ─── Sonar (§2.1) : memes RangeMax/Azimuth que le bag actuel ──────────────────
RANGE_MIN, RANGE_MAX, AZIMUTH_DEG = 0.5, 40.0, 120.0
SIGMA_GYRO, SIGMA_ACC, SIGMA_DVL, SIGMA_DEPTH = 0.002, 0.02, 0.01, 0.02
# ─── Round 2 « noise » (NOISE_MISSION.md, Nathan 2026-07-15) : L2 x5 si
# NOISE_ROUND2=1 (sinon x1.0 exact -> round 1 inchange). Herite par v5 ET v8. ──
import noise_round2 as _NOISE
SIGMA_GYRO *= _NOISE.L2_MULT
SIGMA_ACC *= _NOISE.L2_MULT
SIGMA_DVL *= _NOISE.L2_MULT
SIGMA_DEPTH *= _NOISE.L2_MULT

def make_cfg(with_profiler):
    """Scenario : sonar principal (§0 : MultiPath off, streaks 0, bruit 0.01 —
    les echos de murs de ce monde plafonnent a ~0.25, un bruit 0.05 les noie).
    TRAJ 2 ajoute le ProfilingSonar monte VERTICAL (roll 90)."""
    sensors = [
        {"sensor_type": "ImagingSonar", "sensor_name": "SonarFin",
         "socket": "SonarSocket", "rotation": [0.0, 0.0, 0.0],
         "Hz": int(SONAR_HZ),
         "configuration": {
             "Azimuth": AZIMUTH_DEG, "Elevation": 6,
             "RangeMin": RANGE_MIN, "RangeMax": RANGE_MAX,
             "RangeBins": 512, "AzimuthBins": 512,
             "AddSigma": 0.01, "MultSigma": 0.01, "RangeSigma": 0,
             "MultiPath": False, "AzimuthStreaks": 0, "ScaleNoise": False,
             "InitOctreeRange": 50, "ViewRegion": False}},
    ]
    if with_profiler:
        sensors.append(
            {"sensor_type": "ProfilingSonar", "sensor_name": "ProfilerVert",
             "socket": "SonarSocket", "rotation": [90.0, 0.0, 0.0],
             "Hz": int(SONAR_HZ),
             "configuration": {
                 "Azimuth": 120, "Elevation": 1,
                 "RangeMin": RANGE_MIN, "RangeMax": RANGE_MAX,
                 "RangeBins": 512, "AzimuthBins": 240,
                 "AddSigma": 0.01, "MultSigma": 0.01, "RangeSigma": 0,
                 "MultiPath": False, "AzimuthStreaks": 0, "ScaleNoise": False,
                 "InitOctreeRange": 50, "ViewRegion": False}})
    return {
        "name": "gen3d", "world": "Bruce_slam_nathan", "package_name": "Charuco",
        "main_agent": "auv0", "ticks_per_sec": int(round(1 / DT)),
        "agents": [{
            "agent_name": "auv0", "agent_type": "HoveringAUV",
            "sensors": sensors,
            "control_scheme": 0, "location": [0, 0, Z0]}]}

# Rotation de montage du profiler/sonar vertical : roll -90 deg autour de x.
# 07-11 (PIEGES #14) : le signe du sin a ete corrige dans sonar_to_points3d_msg
# (colonnes hautes = babord) ; ce roll est flippe en meme temps (+90 -> -90)
# pour que la projection NETTE du fan vertical reste IDENTIQUE a celle validee
# E3/E4 (fond a -19.4 sous le robot). Ne flipper ni l'un ni l'autre isolement.
R_MOUNT_PROF = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=float)

# ─── Chemin RECTANGLE a coins arrondis, abscisse curviligne ──────────────────
Lsx = LX - 2 * R_TURN
Lsy = LY - 2 * R_TURN
_ARC = (np.pi / 2) * R_TURN
_SEGS = [Lsx + _ARC, Lsy + _ARC, Lsx + _ARC, Lsy + _ARC]
PERIM = sum(_SEGS)

def path(s):
    """c(s) (2D), t_hat(s) (2D) pour s dans [0, PERIM[ (rectangle arrondi)."""
    s = s % PERIM
    k = 0
    while s > _SEGS[k]:
        s -= _SEGS[k]
        k += 1
    u = s
    Lseg = Lsx if k % 2 == 0 else Lsy
    dirs = [np.array([1, 0]), np.array([0, 1]),
            np.array([-1, 0]), np.array([0, -1])]
    d = dirs[k]; g = dirs[(k + 1) % 4]
    hx, hy = LX / 2, LY / 2
    starts = [np.array([-hx + R_TURN, -hy]), np.array([hx, -hy + R_TURN]),
              np.array([hx - R_TURN, hy]), np.array([-hx, hy - R_TURN])]
    P0 = starts[k] + np.array([CX, CY])
    if u <= Lseg:
        return P0 + u * d, d
    a = (u - Lseg) / R_TURN
    C = P0 + Lseg * d + R_TURN * g
    c = C + R_TURN * (np.sin(a) * d - np.cos(a) * g)
    t = np.cos(a) * d + np.sin(a) * g
    return c, t / np.linalg.norm(t)

def pos_only(t):
    """Position v2 : chemin carre + profondeur sinusoidale (roll=0, « grande
    route »). N_ZCYC entier -> z(0) = z(PERIM) -> boucle fermee exacte."""
    s = V_FWD * t
    c2, _ = path(s)
    z = Z0 + Z_AMP * np.sin(2 * np.pi * N_ZCYC * (s % PERIM) / PERIM)
    return np.array([c2[0], c2[1], z])

def pose_at(t):
    """pose 6-DOF analytique v2 : p (3D), (roll=0, pitch=0, yaw), v_world,
    omega_body. Le bas du robot regarde toujours le bas (§1)."""
    s = V_FWD * t
    _, t2 = path(s)
    yaw = np.arctan2(t2[1], t2[0])
    p = pos_only(t)
    eps = 1e-3
    v_world = (pos_only(t + eps) - pos_only(t - eps)) / (2 * eps)
    # roll=pitch=0 -> axe z corps = axe z monde : omega = (0, 0, dyaw/dt)
    _, t2b = path(s + V_FWD * eps)
    yaw_b = np.arctan2(t2b[1], t2b[0])
    dyaw = (yaw_b - yaw + np.pi) % (2 * np.pi) - np.pi
    omega_body = np.array([0.0, 0.0, dyaw / eps])
    return p, (0.0, 0.0, yaw), v_world, omega_body

def rpy_to_quat(r, p, y):
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)  # x, y, z, w

def R_from_rpy(r, p, y):
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

# ─── Sonar polaire -> points 3D MONDE (§4, z REEL — le livrable) ──────────────
FLOAT32 = 7

def sonar_to_points3d_msg(img, p, rpy, header, thresh=0.10, r_mount=None,
                          azimuth_deg=AZIMUTH_DEG,
                          range_min=RANGE_MIN, range_max=RANGE_MAX):
    """img (RangeBins, AzimuthBins) -> PointCloud2 x,y,z,intensity, points MONDE.
    Pixel (i,j) = (range r_i, azimut a_j) dans le PLAN fin du sonar (elev ~0) :
    point capteur = r[cos a, sin a, 0] (x avant, y gauche) ; monde = p + R_wb @ pt.
    r_mount : rotation de montage capteur->vehicule (ex. R_MOUNT_PROF vertical).
    range_min/max : portee du capteur (le sonar vertical traj4 est en 0.5-20 m)."""
    n_r, n_a = img.shape
    rr = np.linspace(range_min, range_max, n_r)
    aa = np.deg2rad(np.linspace(-azimuth_deg / 2, azimuth_deg / 2, n_a))
    ii, jj = np.nonzero(img > thresh)
    if len(ii) == 0:
        pts = np.zeros((0, 4), np.float32)
    else:
        r = rr[ii]; a = aa[jj]
        inten = img[ii, jj].astype(np.float32)
        # y = +r sin(a) : colonnes hautes de l'image HoloOcean = BABORD (mesure
        # 2026-07-11, arc mur GAMMA a +37deg/31 m, PIEGES #14 — l'ancien
        # y = -r sin(a) mettait tous les /sonar_points en MIROIR lateral).
        pts_b = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], axis=1)
        if r_mount is not None:
            pts_b = pts_b @ r_mount.T
        Rwb = R_from_rpy(*rpy)
        pts_w = pts_b @ Rwb.T + p
        pts = np.column_stack([pts_w.astype(np.float32), inten])
    n = len(pts)
    fields = [PointField(name='x', offset=0, datatype=FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=FLOAT32, count=1),
              PointField(name='intensity', offset=12, datatype=FLOAT32, count=1)]
    return PointCloud2(header=header, height=1, width=n, fields=fields,
                       is_bigendian=False, point_step=16, row_step=n * 16,
                       data=np.frombuffer(pts.tobytes(), dtype=np.uint8),
                       is_dense=True)

# ─── Boucle principale ────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    traj = 1
    test_duration = None
    if "--traj" in args:
        traj = int(args[args.index("--traj") + 1])
    if "--test" in args:
        test_duration = float(args[args.index("--test") + 1])
    with_profiler = (traj == 2)

    import os
    os.makedirs("BAG_files", exist_ok=True)
    out_bag = OUT_BAG.replace(".bag", f"_traj{traj}.bag")
    if test_duration is not None:
        out_bag = out_bag.replace(".bag", "_test.bag")
    if os.path.exists(out_bag):
        os.remove(out_bag)

    T_TOTAL = test_duration if test_duration else N_LAPS * PERIM / V_FWD
    print(f"TRAJ {traj} | perimetre={PERIM:.1f} m | duree simulee={T_TOTAL/60:.1f} min | bag={out_bag}")

    rng = np.random.default_rng(0)
    t, next_sonar, n_pings = 0.0, 0.0, 0

    with Writer(out_bag) as bag, holoocean.make(scenario_cfg=make_cfg(with_profiler)) as env:
        conn_imu   = bag.add_connection('/imu', Imu.__msgtype__, typestore=typestore)
        conn_dvl   = bag.add_connection('/dvl', TwistStamped.__msgtype__, typestore=typestore)
        conn_sonar = bag.add_connection('/sonar', Image.__msgtype__, typestore=typestore)
        conn_pts   = bag.add_connection('/sonar_points', PointCloud2.__msgtype__, typestore=typestore)
        conn_gt    = bag.add_connection('/ground_truth', Odometry.__msgtype__, typestore=typestore)
        conn_depth = bag.add_connection('/depth', Float64.__msgtype__, typestore=typestore)
        conn_prof = None
        if with_profiler:
            conn_prof = bag.add_connection('/profiler_points', PointCloud2.__msgtype__, typestore=typestore)

        agent = env.agents["auv0"]
        while t < T_TOTAL:
            p, rpy, v_w, w_b = pose_at(t)
            agent.teleport(location=p.tolist(), rotation=list(np.degrees(rpy)))
            state = env.tick()

            sim_t = t + 1.0                     # eviter t=0 exactement
            sec = int(sim_t); ns = int((sim_t - sec) * 1e9)
            stamp = Time(sec=sec, nanosec=ns)
            t_ns = sec * 1_000_000_000 + ns
            seq = int(round(t / DT))

            q = rpy_to_quat(*rpy)
            R = R_from_rpy(*rpy)

            # /ground_truth — pose exacte, ENU z haut (§5)
            h = Header(seq=seq, stamp=stamp, frame_id='map')
            gt = Odometry(header=h, child_frame_id='auv0',
                pose=PoseWithCovariance(pose=Pose(
                        position=Point(p[0], p[1], p[2]),
                        orientation=Quaternion(q[0], q[1], q[2], q[3])),
                    covariance=np.zeros(36)),
                twist=TwistWithCovariance(twist=Twist(
                        linear=Vector3(*v_w), angular=Vector3(0., 0., 0.)),
                    covariance=np.zeros(36)))
            bag.write(conn_gt, t_ns, typestore.serialize_ros1(gt, Odometry.__msgtype__))

            # /imu — orientation exacte + gyro/accel synthetiques bruites (§3.4)
            h = Header(seq=seq, stamp=stamp, frame_id='auv0')
            gyro = w_b + rng.normal(0, SIGMA_GYRO, 3)
            acc = R.T @ np.array([0, 0, 9.81]) + rng.normal(0, SIGMA_ACC, 3)
            imu = Imu(header=h,
                orientation=Quaternion(q[0], q[1], q[2], q[3]),
                orientation_covariance=np.zeros(9),
                angular_velocity=Vector3(*gyro),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=Vector3(*acc),
                linear_acceleration_covariance=np.zeros(9))
            bag.write(conn_imu, t_ns, typestore.serialize_ros1(imu, Imu.__msgtype__))

            # /dvl — vitesse repere VEHICULE + bruit
            v_b = R.T @ v_w + rng.normal(0, SIGMA_DVL, 3)
            dvl = TwistStamped(header=h, twist=Twist(
                linear=Vector3(*v_b), angular=Vector3(0., 0., 0.)))
            bag.write(conn_dvl, t_ns, typestore.serialize_ros1(dvl, TwistStamped.__msgtype__))

            # /depth — profondeur POSITIVE vers le bas (= -z) + bruit
            d = Float64(data=float(-p[2] + rng.normal(0, SIGMA_DEPTH)))
            bag.write(conn_depth, t_ns, typestore.serialize_ros1(d, Float64.__msgtype__))

            # /sonar + /sonar_points a SONAR_HZ
            if t >= next_sonar and "SonarFin" in state:
                img = np.asarray(state["SonarFin"], dtype=np.float32)
                h_im = Header(seq=seq, stamp=stamp, frame_id='auv0')
                im = Image(header=h_im, height=img.shape[0], width=img.shape[1],
                           encoding='32FC1', is_bigendian=0, step=img.shape[1] * 4,
                           data=np.frombuffer(img.tobytes(), dtype=np.uint8))
                bag.write(conn_sonar, t_ns, typestore.serialize_ros1(im, Image.__msgtype__))

                h_pts = Header(seq=seq, stamp=stamp, frame_id='map')  # points MONDE
                pc = sonar_to_points3d_msg(img, p, rpy, h_pts)
                bag.write(conn_pts, t_ns, typestore.serialize_ros1(pc, PointCloud2.__msgtype__))

                # /profiler_points — sections VERTICALES (TRAJ 2, §3 option A)
                if conn_prof is not None and "ProfilerVert" in state:
                    prof = np.asarray(state["ProfilerVert"], dtype=np.float32)
                    pc_p = sonar_to_points3d_msg(prof, p, rpy, h_pts,
                                                 r_mount=R_MOUNT_PROF)
                    bag.write(conn_prof, t_ns,
                              typestore.serialize_ros1(pc_p, PointCloud2.__msgtype__))
                next_sonar += 1.0 / SONAR_HZ
                n_pings += 1
                if n_pings % 50 == 0:
                    print(f"  t={t:6.1f}/{T_TOTAL:.0f} s | pings={n_pings} | "
                          f"z={p[2]:+.2f} yaw={np.degrees(rpy[2]):6.1f} deg")
            t += DT

    print(f"\nbag ecrit : {out_bag} ({n_pings} pings sonar)")

if __name__ == "__main__":
    main()
