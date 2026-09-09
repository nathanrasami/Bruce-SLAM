#!/usr/bin/env python3
"""Generateur traj5 : PierHarbor + « + » (2e sonar VERTICAL, config traj4
INCHANGEE) — phase B = ERRANCE NATURELLE profil collegue (demande Nathan
07-11 soir : passer PRES des structures + hauts/bas aleatoires, pas de
carre robotique).

Phase A calibration IDENTIQUE a traj4 (C1 statique 5.2 m / C2 360° / C3
ascenseur — les checks E2/E4/E6/E7 en dependent, memes fenetres temporelles).

Phase B : circuit median a coins arrondis qui SERRE les deux quais SANS
traverser le mur GAMMA (l'anneau traj3 du collegue le traversait 2x/tour,
teleport irrealiste) + les offsets PCHIP du collegue VERBATIM (lateral
±N_MAX redecide tous les L_SEG m, z aleatoire [Z_MIN, Z_MAX], padding
cyclique anti-saut de yaw a la couture des tours). 2 tours, meme tirage
aux 2 tours (revisites propres pour les loops).

Circuit (monde, mesures pierharbor-geometrie-monde) :
  quai EST serre a x=525 (face 531.5 -> 6.5 m) · passage bateau a x=514
  (bateau 518-530 -> 4 m) · retour nord par x=490 (GAMMA vertical x=485
  -> 5 m) · corridor y=-641 (GAMMA horizontal y=-645 -> 4 m) · quai OUEST
  serre a x=468 (face 462.5 -> 5.5 m, comme le collegue).

Usage : python gen_bag_3d_v5.py [--test 150] [--seed 42]
"""
import json
import os
import sys
import numpy as np
import holoocean
from scipy.interpolate import PchipInterpolator

from gen_bag_3d import (typestore, Time, Header, Vector3, Quaternion, Imu,
                        TwistStamped, Twist, Image, Odometry,
                        PoseWithCovariance, Pose, Point, TwistWithCovariance,
                        Float64, PointCloud2,
                        rpy_to_quat, R_from_rpy, sonar_to_points3d_msg,
                        R_MOUNT_PROF,
                        RANGE_MIN, RANGE_MAX, AZIMUTH_DEG,
                        SIGMA_GYRO, SIGMA_ACC, SIGMA_DVL, SIGMA_DEPTH)
from rosbags.rosbag1 import Writer
import noise_round2 as _NOISE          # round 2 « noise » (NOISE_ROUND2=1)

HERE = os.path.dirname(os.path.abspath(__file__))
ZONE = json.load(open(os.path.join(HERE, "pierharbor_zone.json")))
assert ZONE.get("ok"), "pierharbor_zone.json invalide — relancer probe_pierharbor.py"

# ─── Parametres traj5 ─────────────────────────────────────────────────────────
V_FWD    = 0.35
DT       = 0.05
SONAR_HZ = 5.0
YAW_RATE = 10.0                     # deg/s — C2 + pivots de transit
R_TURN   = 4.0
N_LAPS   = 2

# Errance = parametres du collegue (gen v4 originel a5e9cd1), repris VERBATIM
SEED   = 42
L_SEG  = 8.0                        # distance entre decisions aleatoires (m)
N_MAX  = 1.2                        # excursion laterale max (m)
Z_MIN, Z_MAX = -12.0, -2.0          # bande z du collegue (fond -19.4)

# Phase A identique a traj4 (E-checks : memes fenetres temporelles)
P_A  = np.array([526.3, -660.0])
Z_A  = -9.5
BOAT = np.array([524.0, -680.5])

LEVER = np.array([0.0, 0.0, 0.15])
VERT_RANGE_MAX  = 20.0
VERT_RANGE_BINS = 512
VERT_ELEV_BINS  = 256
OUT_BAG = _NOISE.bag("BAG_files/holoocean_3d_traj5.bag")  # _noise si round 2

# Sommets du circuit median (ordre de parcours ; s=0 juste apres W4).
# W4 pres de P_A -> transit court depuis la phase A.
WPTS = np.array([
    [525.0, -662.0],   # W4  quai EST (6.5 m), pres de P_A
    [514.0, -670.0],   # W5  amorce passage bateau
    [514.0, -690.0],   # W6  longe le bateau a 4 m (518-530)
    [490.0, -690.0],   # W7  sud, eau libre
    [490.0, -641.0],   # W8  remonte a 5 m du GAMMA vertical (x=485)
    [468.0, -641.0],   # W1  corridor a 4 m du GAMMA horizontal (y=-645)
    [468.0, -626.0],   # W2  quai OUEST serre (5.5 m, comme le collegue)
    [525.0, -626.0],   # W3  nord -> retour quai EST
])

# Structures pour l'auto-verification de clearance (memes que check E8)
_GAMMA = [((464.0, -645.0), (485.0, -645.0)), ((485.0, -650.0), (485.0, -701.0))]
_QUAIS = [((462.5, -705.0), (462.5, -630.0)), ((531.5, -705.0), (531.5, -630.0))]

# ─── Chemin median ferme a coins arrondis, abscisse curviligne ────────────────
def _rounded_path(V, R):
    n = len(V)
    tin, tout, arcs = [], [], []
    for i in range(n):
        A, B, C = V[i - 1], V[i], V[(i + 1) % n]
        u = (B - A) / np.linalg.norm(B - A)
        v = (C - B) / np.linalg.norm(C - B)
        cross = u[0] * v[1] - u[1] * v[0]
        phi = float(np.arctan2(abs(cross), float(u @ v)))
        d = R * np.tan(phi / 2)
        ti, to = B - u * d, B + v * d
        sgn = 1.0 if cross > 0 else -1.0
        nrm = sgn * np.array([-u[1], u[0]])
        Cc = ti + R * nrm
        a0 = float(np.arctan2(*(ti - Cc)[::-1]))
        arcs.append((Cc, a0, sgn * phi))
        tin.append(ti); tout.append(to)
    pieces = []                       # s=0 = debut de la droite apres V[0]
    for i in range(n):
        p0, p1 = tout[i], tin[(i + 1) % n]
        L = float(np.linalg.norm(p1 - p0))
        assert L > 0.1, f"fillets superposes au sommet {i}"
        pieces.append(("S", p0, (p1 - p0) / L, L))
        Cc, a0, dphi = arcs[(i + 1) % n]
        pieces.append(("A", Cc, a0, dphi, R * abs(dphi)))
    lens = np.array([p[3] if p[0] == "S" else p[4] for p in pieces])
    return pieces, np.concatenate([[0.0], np.cumsum(lens)])

_PIECES, _PCUM = _rounded_path(WPTS, R_TURN)
PERIM = float(_PCUM[-1])

def chemin_median(s):
    s = s % PERIM
    i = min(int(np.searchsorted(_PCUM, s, side="right")) - 1, len(_PIECES) - 1)
    ls = s - _PCUM[i]
    pc = _PIECES[i]
    if pc[0] == "S":
        _, p0, u, _ = pc
        return p0 + ls * u, u
    _, Cc, a0, dphi, Larc = pc
    a = a0 + dphi * (ls / Larc)
    pt = Cc + R_TURN * np.array([np.cos(a), np.sin(a)])
    sgn = 1.0 if dphi > 0 else -1.0
    return pt, sgn * np.array([-np.sin(a), np.cos(a)])

# ─── Errance aleatoire bornee (code du collegue, z0 force a Z_A) ──────────────
def offsets_aleatoires(perim, seed):
    rng = np.random.default_rng(seed)
    n = max(8, int(perim / L_SEG))
    s_nodes = np.linspace(0.0, perim, n + 1)
    lat = rng.uniform(-N_MAX, N_MAX, n + 1)
    z = rng.uniform(Z_MIN, Z_MAX, n + 1)
    lat[0] = lat[-1] = 0.0
    z[0] = z[-1] = Z_A                # couture = fin du transit (z inchange)
    ds = s_nodes[1] - s_nodes[0]      # padding CYCLIQUE (raccord C1 des tours)
    s_pad = np.concatenate([[-2 * ds, -ds], s_nodes, [perim + ds, perim + 2 * ds]])
    lat_p = np.concatenate([lat[-3:-1], lat, lat[1:3]])
    z_p = np.concatenate([z[-3:-1], z, z[1:3]])
    return PchipInterpolator(s_pad, lat_p), PchipInterpolator(s_pad, z_p)

F_LAT = F_Z = None

def pos_errance(s):
    c, t_hat = chemin_median(s)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    p = c + float(F_LAT(s % PERIM)) * n_hat
    return np.array([p[0], p[1], float(F_Z(s % PERIM))])

def yaw_errance(s):
    # base ±0.25 m (collegue) : une base millimetrique amplifie la courbure
    # PCHIP en jitter de cap
    qa, qb = pos_errance(s + 0.25), pos_errance(s - 0.25)
    return float(np.arctan2(qa[1] - qb[1], qa[0] - qb[0]))

# ─── Trajectoire = segments (meme machinerie que traj4) ───────────────────────
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def _build_segments():
    segs = []
    cur = {"p": np.array([P_A[0], P_A[1], Z_A]), "yaw": 0.0}

    def static(dur, label="pause"):
        p, yw = cur["p"].copy(), cur["yaw"]
        segs.append((float(dur), lambda tl, p=p, yw=yw: (p, yw), label))

    def sweep(deltas, label="sweep"):
        for d in deltas:
            p, y0, dur = cur["p"].copy(), cur["yaw"], abs(d) / YAW_RATE
            w = np.deg2rad(d) / dur
            segs.append((dur, lambda tl, p=p, y0=y0, w=w: (p, y0 + w * tl), label))
            cur["yaw"] = y0 + np.deg2rad(d)

    def pivot_to(yaw_deg, label="pivot"):
        d = np.degrees(_wrap(np.deg2rad(yaw_deg) - cur["yaw"]))
        if abs(d) > 1e-9:
            sweep([d], label)

    def line(x, y, z=None, label="ligne"):
        p0 = cur["p"].copy()
        p1 = np.array([x, y, p0[2] if z is None else float(z)])
        dur = float(np.hypot(*(p1[:2] - p0[:2]))) / V_FWD
        yw = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        assert abs(_wrap(yw - cur["yaw"])) < 1e-6, f"cap discontinu avant {label}"
        segs.append((dur, lambda tl, p0=p0, p1=p1, dur=dur, yw=yw:
                     (p0 + (p1 - p0) * (tl / dur), yw), label))
        cur["p"], cur["yaw"] = p1, yw

    def elevator(dur, amp, label="C3 ascenseur"):
        p, yw = cur["p"].copy(), cur["yaw"]
        segs.append((float(dur), lambda tl, p=p, yw=yw, dur=dur, amp=amp:
                     (p + np.array([0., 0., amp * np.sin(2 * np.pi * tl / dur)]),
                      yw), label))

    # Phase A (identique traj4 — E2/E4/E6/E7 dependent de ces fenetres)
    static(10.0, "C1 statique")
    sweep([360.0], "C2 tour complet")
    static(3.0)
    elevator(20.0, 3.0)
    static(3.0)

    # Transit vers l'entree de l'errance (s=0, lat=0, z=Z_A -> aucun saut)
    entry = pos_errance(0.0)
    brg = float(np.degrees(np.arctan2(entry[1] - cur["p"][1],
                                      entry[0] - cur["p"][0])))
    pivot_to(brg, "vers errance")
    line(entry[0], entry[1], label="transit errance")
    pivot_to(float(np.degrees(yaw_errance(0.0))), "cap errance")

    # Phase B : errance PCHIP, N_LAPS tours (meme tirage a chaque tour)
    dur = N_LAPS * PERIM / V_FWD

    def f_err(tl):
        s = V_FWD * tl
        return pos_errance(s), yaw_errance(s)
    segs.append((dur, f_err, f"errance PCHIP seed ({N_LAPS} tours)"))
    cur["p"], cur["yaw"] = pos_errance(N_LAPS * PERIM), yaw_errance(0.0)
    static(3.0, "fin")
    return segs

_SEGS5 = _EDGES = None
T_TOTAL = 0.0

def _init_traj(seed):
    global F_LAT, F_Z, _SEGS5, _EDGES, T_TOTAL
    F_LAT, F_Z = offsets_aleatoires(PERIM, seed)
    _SEGS5 = _build_segments()
    _EDGES = np.concatenate([[0.0], np.cumsum([s[0] for s in _SEGS5])])
    T_TOTAL = float(_EDGES[-1])

_init_traj(SEED)

def sched_pose(t):
    t = min(max(t, 0.0), T_TOTAL - 1e-9)
    i = min(int(np.searchsorted(_EDGES, t, side="right")) - 1, len(_SEGS5) - 1)
    p, yaw = _SEGS5[i][1](t - _EDGES[i])
    return p, yaw

def pose_at_v5(t):
    p, yaw = sched_pose(t)
    eps = 1e-2
    pa, ya = sched_pose(t + eps)
    pb, yb = sched_pose(t - eps)
    v_world = (pa - pb) / (2 * eps)
    omega = np.array([0.0, 0.0, _wrap(ya - yb) / (2 * eps)])
    return p, (0.0, 0.0, yaw), v_world, omega

# ─── Auto-verification du chemin (sans moteur) ────────────────────────────────
def _d_seg(P, a, b):
    a, b = np.array(a, float), np.array(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)

def verifier_chemin():
    s = np.arange(0.0, PERIM, 0.5)
    env = np.array([pos_errance(v)[:2] for v in s])   # enveloppe REELLE (lat inclus)
    d_gamma = np.min([_d_seg(env, a, b) for a, b in _GAMMA], axis=0)
    d_quais = np.min([_d_seg(env, a, b) for a, b in _QUAIS], axis=0)
    # continuite du median
    pts = np.array([chemin_median(v)[0] for v in s])
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    # taux de lacet sur toute l'errance
    tt = np.arange(0.0, PERIM / V_FWD, 1.0)
    yy = np.unwrap([yaw_errance(V_FWD * v) for v in tt])
    dyaw = np.degrees(np.abs(np.diff(yy)))
    zz = np.array([pos_errance(V_FWD * v)[2] for v in tt])
    print(f"chemin traj5 : PERIM={PERIM:.1f} m | clearance GAMMA min "
          f"{d_gamma.min():.2f} m | quais min {d_quais.min():.2f} m | "
          f"pas median max {step.max():.3f} (0.5 attendu) | "
          f"|dyaw/dt| max {dyaw.max():.1f}°/s | z [{zz.min():.1f},{zz.max():.1f}]")
    assert d_gamma.min() > 1.2, "trop pres du mur GAMMA"
    assert d_quais.min() > 3.5, "trop pres d'une face de quai"
    assert 0.4 < step.max() < 0.6, "median discontinu"
    assert dyaw.max() < 25.0, "taux de lacet irrealiste"

def make_cfg():
    common = {"RangeMin": RANGE_MIN, "RangeMax": RANGE_MAX,
              # L1 round 2 : AddSigma/MultSigma 0.01 -> 0.05 (x5) si NOISE_ROUND2=1
              "AddSigma": _NOISE.SONAR_ADD, "MultSigma": _NOISE.SONAR_MULT,
              "RangeSigma": 0,
              "MultiPath": False, "AzimuthStreaks": 0, "ScaleNoise": False,
              "InitOctreeRange": 50, "ViewRegion": False}
    return {
        "name": "gen3dv5", "world": "PierHarbor", "package_name": "Ocean",
        "main_agent": "auv0", "octree_min": 0.1, "octree_max": 5.0,
        "ticks_per_sec": int(round(1 / DT)),
        "agents": [{
            "agent_name": "auv0", "agent_type": "HoveringAUV",
            "sensors": [
                {"sensor_type": "ImagingSonar", "sensor_name": "SonarFin",
                 "socket": "SonarSocket", "rotation": [0.0, 0.0, 0.0],
                 "Hz": int(SONAR_HZ),
                 "configuration": dict(common, Azimuth=AZIMUTH_DEG, Elevation=6,
                                       # RangeBins 512→1024 (2026-07-15, Nathan) :
                                       # 2.0 cm/bin @20 m (traj9), 3.9 @40 m (traj5).
                                       # ⚠ AzimuthBins RESTE 512 : à 1024 l'image est
                                       # STRIÉE (65 % de colonnes illuminées, trous de
                                       # 8 — densité de rayons du simu) → E6 FAIL,
                                       # features CFAR décimées. Mesuré 2026-07-15.
                                       RangeBins=1024, AzimuthBins=512)},
                {"sensor_type": "ImagingSonar", "sensor_name": "SonarVert",
                 "socket": "SonarSocket", "location": [float(v) for v in LEVER],
                 "rotation": [90.0, 0.0, 0.0],
                 "Hz": int(SONAR_HZ),
                 "configuration": dict(common, Azimuth=120, Elevation=6,
                                       RangeMax=VERT_RANGE_MAX,
                                       RangeBins=VERT_RANGE_BINS,
                                       AzimuthBins=VERT_ELEV_BINS)},
            ],
            "control_scheme": 0, "location": list(sched_pose(0.0)[0])}]}

def main():
    args = sys.argv[1:]
    test = float(args[args.index("--test") + 1]) if "--test" in args else None
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else SEED
    if seed != SEED:
        _init_traj(seed)
    verifier_chemin()

    out = OUT_BAG if test is None else OUT_BAG.replace(".bag", "_test.bag")
    if os.path.exists(out):
        os.remove(out)
    os.makedirs("BAG_files", exist_ok=True)
    T = test if test else T_TOTAL
    print(f"TRAJ5 | errance profil collegue SEED={seed} | quaiE 6.5 m/bateau "
          f"4 m/GAMMA 5 m/quaiO 5.5 m | z [{Z_MIN:.0f},{Z_MAX:.0f}] | "
          f"duree={T/60:.1f} min (totale {T_TOTAL/60:.1f}) | bag={out}")

    rng = np.random.default_rng(0)
    t, next_sonar, n_pings = 0.0, 0.0, 0
    zero3 = np.zeros(3)

    # show_viewport=False : crashs GPU NVRM Xid 13 sinon (mesure traj4)
    with Writer(out) as bag, holoocean.make(scenario_cfg=make_cfg(),
                                            show_viewport=False) as env:
        c_imu = bag.add_connection('/imu', Imu.__msgtype__, typestore=typestore)
        c_dvl = bag.add_connection('/dvl', TwistStamped.__msgtype__, typestore=typestore)
        c_son = bag.add_connection('/sonar', Image.__msgtype__, typestore=typestore)
        c_pts = bag.add_connection('/sonar_points', PointCloud2.__msgtype__, typestore=typestore)
        c_gt = bag.add_connection('/ground_truth', Odometry.__msgtype__, typestore=typestore)
        c_dep = bag.add_connection('/depth', Float64.__msgtype__, typestore=typestore)
        c_sv = bag.add_connection('/sonar_vert', Image.__msgtype__, typestore=typestore)
        c_svp = bag.add_connection('/sonar_vert_points', PointCloud2.__msgtype__, typestore=typestore)

        agent = env.agents["auv0"]
        while t < T:
            p, rpy, v_w, w_b = pose_at_v5(t)
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
                                                range_max=VERT_RANGE_MAX)
                    bag.write(c_svp, t_ns, typestore.serialize_ros1(vpc, PointCloud2.__msgtype__))
                next_sonar += 1.0 / SONAR_HZ
                n_pings += 1
                if n_pings % 100 == 0:
                    seg = _SEGS5[min(int(np.searchsorted(_EDGES, t, side='right')) - 1,
                                     len(_SEGS5) - 1)][2]
                    print(f"  t={t:6.1f}/{T:.0f} s | pings={n_pings} | "
                          f"z={p[2]:+.2f} | {seg}")
            t += DT

    print(f"\nbag ecrit : {out} ({n_pings} pings) | SEED={seed} | "
          f"duree {min(T, T_TOTAL):.0f} s")

if __name__ == "__main__":
    main()
