#!/usr/bin/env python3
"""LA carte 3D unique d'un run : <run>/carte_3d.html (+ .png, .npy).

Règle de Nathan (08-07) : UNE carte par run, VRAIE 3D uniquement.
- Une source (topic de points) n'entre dans la carte QUE si son std(z)
  INTRA-message médian > 0.5 m (= le capteur voit du volume). Les tranches
  planes (pseudo-3D « 2D plaqué au z du robot ») sont EXCLUES ; s'il n'y a
  aucune vraie source, le script REFUSE (pas de carte mensongère).
- Carte 100 % GT-free : points composés avec la trajectoire SLAM.
  * bags v4 (frame_id=auv0) : composition directe pose SLAM × points véhicule ;
  * bags v3 (frame_id=map) : dé-projection via la pose GT du même stamp
    (décodage du format, aucune info GT dans le résultat) puis re-composition.
- Métrique : même carte composée avec les poses GT, alignement **Umeyama SE(2)**
  des trajectoires (remplace l'alignement première-pose : sur un grand site,
  le yaw à t0 × bras de levier dominait le NN).

Remplace comme livrable : view3d/carte_3d.png (pseudo), profiler_3d,
profiler_slam_3d, traj3_map_3d (gardés comme outils, plus des livrables).

Usage (conteneur ros1) :
  python3 analysis/carte_3d.py <run_dir> [--bag <bag>]
  (--bag omis : lit <run_dir>/bag_source.txt écrit par run_slam.sh)
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import rosbag
import sensor_msgs.point_cloud2 as pc2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_eval import umeyama, associer_par_temps

TOPICS = ["/profiler_points", "/sonar_points", "/sonar_vert_points"]
SEUIL_VRAIE_3D = 0.5   # m, std(z) intra-message médian
SEUIL_PLAN = 0.05      # m, std ≈ 0 : le fan est contenu dans un plan
N_SONDAGE = 60         # messages par topic pour le verdict vraie/pseudo 3D


def yaw_of(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def rot_z(pts, th):
    c, s = np.cos(th), np.sin(th)
    return np.c_[c * pts[:, 0] - s * pts[:, 1],
                 s * pts[:, 0] + c * pts[:, 1], pts[:, 2]]


def voxel_grid(P, vs):
    """Sous-échantillonnage voxel : 1 centroïde par cellule (dédoublonne, allège)."""
    if not len(P):
        return P
    k = np.floor(P / vs).astype(np.int64)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(cnt), 3))
    np.add.at(sums, inv, P)
    return sums / cnt[:, None]


def per_beam_max(pts):
    """Réduction « méthode grottes » (caves_3d.py) pour un profiler TRANSVERSE.

    pts : Nx4 (x,y,z,intensity) en repère véhicule, plan transverse x≡0. Chaque
    faisceau (azimut φ = atan2(-y,-z) dans le plan y-z) a ~2 retours : on garde le
    plus FORT = la paroi/le fond (1 faisceau ≈ 1 distance). C'est ce qui change une
    bouillie dense en sections propres empilées le long de la trajectoire → treillis
    et pilotis lisibles (validé sur traj3 : pilotis −19→−7 m, treillis en X visibles).
    """
    if len(pts) < 3 or pts.shape[1] < 4:
        return pts[:, :3] if len(pts) else pts
    y, z, i = pts[:, 1], pts[:, 2], pts[:, 3]
    key = np.round(np.degrees(np.arctan2(-y, -z)) * 2).astype(int)  # bin 0.5°
    return _max_par_faisceau(pts, key)


def per_beam_max_xz(pts):
    """Même réduction pour un fan VERTICAL AVANT (le « + », guide §2.3quinquies).

    pts : Nx4 (x,y,z,intensity) en repère véhicule, plan x-z (y≡0). Le sonar est
    le sonar SLAM tourné de 90° autour de l'axe d'avance : son fan balaye
    l'ÉLÉVATION devant le robot. Un faisceau = une élévation φ = atan2(-z, x) ;
    on garde le retour le plus FORT = la paroi vue de face. Réduit la bavure
    hors axe (l'ambiguïté de ce montage est en azimut, pas en élévation).
    """
    if len(pts) < 3 or pts.shape[1] < 4:
        return pts[:, :3] if len(pts) else pts
    key = np.round(np.degrees(np.arctan2(-pts[:, 2], pts[:, 0])) * 2).astype(int)
    return _max_par_faisceau(pts, key)


def _max_par_faisceau(pts, key):
    order = np.lexsort((pts[:, 3], key))    # tri par faisceau puis intensité croissante
    ks = key[order]
    last = np.ones(len(ks), bool)
    last[:-1] = ks[1:] != ks[:-1]           # dernier de chaque faisceau = intensité max
    return pts[order][last][:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--brut", action="store_true",
                    help="désactive le filtre anti-résidus (surface + fantômes "
                         "hors-plan, cf. FABLE §10-bis / PIEGES #15)")
    a = ap.parse_args()
    bag_path = a.bag
    if bag_path is None:
        src = os.path.join(a.run, "bag_source.txt")
        if not os.path.isfile(src):
            sys.exit("carte_3d : pas de --bag ni de bag_source.txt — abandon")
        bag_path = open(src).read().strip()
    if not os.path.isfile(bag_path):
        sys.exit(f"carte_3d : bag introuvable : {bag_path}")

    traj = np.genfromtxt(os.path.join(a.run, "trajectory.csv"),
                         delimiter=",", names=True)
    ts, th_slam = traj["time"], np.unwrap(traj["theta"])
    rng = np.random.default_rng(0)
    bag = rosbag.Bag(bag_path)

    # ── GT : lue AVANT la passe 1 (sert à dé-projeter les bags v3) + métrique ─
    gt_t, gt_x, gt_y, gt_z, gt_th = [], [], [], [], []
    for _, m, _ in bag.read_messages(topics=["/ground_truth"]):
        gt_t.append(m.header.stamp.to_sec())
        p = m.pose.pose.position
        gt_x.append(p.x); gt_y.append(p.y); gt_z.append(p.z)
        gt_th.append(yaw_of(m.pose.pose.orientation))
    gt_t = np.array(gt_t)
    gt_x, gt_y, gt_z = map(np.array, (gt_x, gt_y, gt_z))
    gt_th = np.unwrap(np.array(gt_th)) if gt_th else np.array([])
    has_gt = len(gt_t) > 0

    def vers_vehicule(xyz, t, frame):
        """v3 (frame_id=map) : monde → véhicule via la pose GT du même stamp
        (décodage du format ; aucune info GT ne subsiste dans la carte finale)."""
        if frame != "map":
            return xyz
        return rot_z(xyz - [np.interp(t, gt_t, gt_x), np.interp(t, gt_t, gt_y),
                            np.interp(t, gt_t, gt_z)], -np.interp(t, gt_t, gt_th))

    def lire(m):
        """Nx4 (x,y,z,intensity) ; intensity=0 si le nuage n'en porte pas."""
        has_i = any(f.name == "intensity" for f in m.fields)
        ch = ("x", "y", "z", "intensity") if has_i else ("x", "y", "z")
        p = np.array(list(pc2.read_points(m, field_names=ch, skip_nans=True)))
        if not len(p):
            return p.reshape(0, 4)
        return p if has_i else np.c_[p, np.zeros(len(p))]

    # ── passe 1 : sondage — quelles sources sont de la VRAIE 3D ? ──────────
    # ⚠ Les std sont mesurées EN REPÈRE VÉHICULE (bags v3 dé-projetés d'abord).
    # Sur les points bruts d'un bag `map`, le plan du fan tourne avec le cap :
    # le verdict mesurerait le cap du robot pendant le sondage, pas la géométrie
    # du capteur (traj2, spirale 720° : std(y) va de 0.00 à 8.41 m selon le cap).
    stds = {t: [] for t in TOPICS}      # std(z) intra-ping (sonar tilté oscillant)
    stds_x = {t: [] for t in TOPICS}    # std(x) intra-ping (≈0 = fan transverse)
    stds_y = {t: [] for t in TOPICS}    # std(y) intra-ping (≈0 = fan vertical avant)
    frames = {}
    seen = 0
    for topic, m, _ in bag.read_messages(topics=TOPICS):
        seen += 1
        if seen > 6 * N_SONDAGE:   # borne dure : un topic absent ne bloque pas
            break
        if len(stds[topic]) >= N_SONDAGE:
            continue
        pts = lire(m)[:, :3]
        if len(pts) <= 10:
            continue
        fr = m.header.frame_id
        if fr == "map":
            if not has_gt:
                continue
            pts = vers_vehicule(pts, m.header.stamp.to_sec(), fr)
        stds[topic].append(pts[:, 2].std())
        stds_x[topic].append(pts[:, 0].std())
        stds_y[topic].append(pts[:, 1].std())
        frames[topic] = fr
    LIB = {"vertical": "fan VERTICAL avant, le « + » (1 paroi/faisceau)",
           "transverse": "profiler transverse (méthode grottes)",
           "tilte": "vraie 3D (gate par ping en passe 2)"}
    retenus = []
    for t in TOPICS:
        if not stds[t]:
            continue
        med, p90 = float(np.median(stds[t])), float(np.percentile(stds[t], 90))
        mx, my = float(np.median(stds_x[t])), float(np.median(stds_y[t]))
        # Trois géométries de vraie 3D, discriminées par le plan du fan :
        #  1. fan VERTICAL AVANT (« + », §2.3quinquies) → plan x-z : std(y)≈0 et
        #     std(z) grand. Source STRUCTURELLE : ce qu'on voit DEVANT.
        #  2. profiler TRANSVERSE (fix §2.3ter) → plan y-z : std(x)≈0 et std(y)
        #     grand. Un fond plat donne std(z)≈0 par ping (r=prof/cosφ) → le test
        #     std(z) le rejetterait à tort : on le détecte par sa géométrie x≡0.
        #  3. sonar tilté oscillant → std(z) intra-ping grand sur les pings inclinés
        #     (p90 : le tilt alterne pings plats/3D ; le gate par ping en passe 2 trie).
        vertical = my < SEUIL_PLAN and med > SEUIL_VRAIE_3D
        transverse = mx < SEUIL_PLAN and my > SEUIL_VRAIE_3D
        g = ("vertical" if vertical else "transverse" if transverse
             else "tilte" if p90 > SEUIL_VRAIE_3D else None)
        print(f"{t} : std(z) méd {med:.2f}/p90 {p90:.2f} | std(x) {mx:.2f} std(y) {my:.2f} m "
              f"({frames[t]}, repère véhicule) → {LIB.get(g, 'pseudo-3D (tranches), EXCLU')}")
        if g:
            retenus.append((t, g))
    # Sources STRUCTURELLES (1 paroi/faisceau, sections nettes) : fan VERTICAL
    # avant ET profiler TRANSVERSE sont FUSIONNÉS (traj6, 2026-07-12 — l'ancienne
    # priorité exclusive au vertical gâchait les flancs/haut/bas du transverse).
    # Le sonar tilté « pulvérise » des fans radiaux (la bouillie qu'on veut
    # éviter) → il reste exclu dès qu'une source structurelle existe.
    geo_of = dict(retenus)
    struct = [t for t, g in retenus if g in ("vertical", "transverse")]
    if struct:
        garde = struct
        lib = " + ".join(sorted({LIB[geo_of[t]] for t in struct}))
        print(f"→ carte STRUCTURELLE ({lib}) depuis {garde} ; "
              f"autres sources exclues.")
    else:
        garde = [t for t, _ in retenus]
    if not garde:
        bag.close()
        sys.exit("carte_3d : AUCUNE source vraie-3D dans ce bag — pas de carte "
                 "(refus : une carte 2D-plaquée serait mensongère).")
    retenus = garde

    # ── passe 2 : construction PAR SOURCE (SLAM = livrable ; GT = réf métrique) ──
    S_by, G_by = {t: [] for t in retenus}, {t: [] for t in retenus}
    counters = {t: 0 for t in retenus}
    for topic, m, _ in bag.read_messages(topics=retenus):
        counters[topic] += 1
        if counters[topic] % 2:
            continue
        t = m.header.stamp.to_sec()
        if t > ts[-1] + 0.5:
            continue
        pts = lire(m)
        if len(pts) < 3:
            continue
        if frames[topic] == "map":                 # v3 : monde → véhicule (GT)
            if not has_gt:
                continue
            pts = np.c_[vers_vehicule(pts[:, :3], t, "map"), pts[:, 3]]
        g = geo_of[topic]
        if g == "vertical":
            # fan VERTICAL AVANT (y≡0) : 1 retour fort par ÉLÉVATION = la paroi
            # vue de face. PAS de gate std(z) par ping : le fan balaye x-z, c'est
            # de la vraie 3D par construction.
            pts = per_beam_max_xz(pts)
        elif g == "transverse":
            # profiler TRANSVERSE (x≡0) : réduction « grottes » 1 retour fort/faisceau.
            # PAS de gate std(z) par ping : un fond plat donne z≈cst (géométrie
            # r=prof/cos φ), c'est de la vraie 3D par construction (le fan balaye y-z).
            pts = per_beam_max(pts)
            # ⚠ MIROIR y du profiler des bags v3 SEULEMENT (frame_id=map, traj1-3
            # jamais réécrits post-fix PIEGES #14) : leurs /profiler_points ont
            # l'axe y INVERSÉ vs la convention véhicule. Preuve (2026-07-09) :
            # sans flip, les murs de quai du profiler traj3 tombent à x≈+2/+38
            # (au CENTRE) ; avec y négé ils reviennent PILE sur les quais du
            # sonar horizontal (x≈−10/+59). Les bags v6+ (frame_id=auv0) portent
            # le mount transverse MESURÉ (probe 2026-07-12) et verrouillé E9 →
            # ne PAS les flipper.
            if frames[topic] == "map":
                pts[:, 1] = -pts[:, 1]
        else:
            pts = pts[:, :3]
            if pts[:, 2].std() <= SEUIL_VRAIE_3D:   # gate par PING : tranche plate
                continue                             # (ex. tilt≈0) → exclue
            if len(pts) > 300:
                pts = pts[rng.choice(len(pts), 300, replace=False)]
        # composition pose SLAM (GT-free)
        w = rot_z(pts, np.interp(t, ts, th_slam))
        w += [np.interp(t, ts, traj["x"]), np.interp(t, ts, traj["y"]),
              np.interp(t, ts, traj["z"])]
        S_by[topic].append(w[w[:, 2] < 0])
        if has_gt:                                  # référence carte-GT
            g = rot_z(pts, np.interp(t, gt_t, gt_th))
            g += [np.interp(t, gt_t, gt_x), np.interp(t, gt_t, gt_y),
                  np.interp(t, gt_t, gt_z)]
            G_by[topic].append(g[g[:, 2] < 0])
    bag.close()
    S_full = np.vstack([p for lst in S_by.values() for p in lst])

    # ── Filtre anti-résidus (2026-07-11, FABLE §10-bis) — désactivable --brut ─
    # Deux artefacts MESURÉS du fan vertical (validés contre le profiler
    # transverse indépendant de traj3, qui ne voit RIEN à ces endroits) :
    #  1. SURFACE : miroir acoustique vu de près quand le robot est haut
    #     (traj5 z=-2.3 → 31 % de la carte) → coupe z > -1.8 (z = pression,
    #     GT-free).
    #  2. FANTÔMES HORS-PLAN : l'ouverture azimut réelle du SonarVert laisse
    #     entrer des structures fortes à ~20° hors du plan (mesuré : mur Γ
    #     rabattu en nappe à z≈-13, y≈-640) → un point vertical n'est gardé
    #     que s'il est CONFIRMÉ par la carte 2D du sonar horizontal (≤ 2 m en
    #     x-y, pointcloud.csv du run) OU appartient au FOND (z < p03 + 1.5 m,
    #     invisible au fan horizontal en rasante donc exempté).
    def _anti_residus(P, tag, off=(0.0, 0.0)):
        # off : translation repère SLAM -> repère de P (la branche GT est en
        # monde absolu ; pointcloud.csv est en repère SLAM ancré au départ)
        if len(P) == 0:
            return P
        n0 = len(P)
        P = P[P[:, 2] < -1.8]
        n_surf = n0 - len(P)
        pc_csv = os.path.join(a.run, "pointcloud.csv")
        n_ghost = 0
        if os.path.isfile(pc_csv) and len(P):
            m2d = np.genfromtxt(pc_csv, delimiter=",", names=True)
            Q = np.stack([m2d["x"] + off[0], m2d["y"] + off[1]], axis=1)
            cell = {}
            for q in np.round(Q / 2.0).astype(np.int64):
                cell[tuple(q)] = True
            k = np.round(P[:, :2] / 2.0).astype(np.int64)
            neigh = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            conf = np.array([any((x + i, y + j) in cell for i, j in neigh)
                             for x, y in k])
            z_fond = np.percentile(P[:, 2], 3)
            keep = conf | (P[:, 2] < z_fond + 1.5)
            n_ghost = int((~keep).sum())
            P = P[keep]
        print(f"filtre anti-résidus [{tag}] : -{n_surf} surface, "
              f"-{n_ghost} fantômes hors-plan → {len(P):,} pts")
        return P

    # Étendu au profiler transverse (traj6) : même tech de rendu → mêmes
    # artefacts (surface en miroir + fuite hors-plan, PIEGES #15) ; la coupe
    # surface, la confirmation 2D (murs vus par le sonar horizontal) et
    # l'exemption fond s'appliquent à l'identique.
    gs_struct = {geo_of[t] for t in retenus}
    if not a.brut and gs_struct and gs_struct <= {"vertical", "transverse"}:
        S_full = _anti_residus(S_full, "SLAM")
        off_gt = ((float(gt_x[0] - traj["x"][0]), float(gt_y[0] - traj["y"][0]))
                  if has_gt else (0.0, 0.0))
        G_by = {t: [_anti_residus(np.vstack(lst), "GT", off_gt)] if lst else []
                for t, lst in G_by.items()}

    # ── CARTE = nuage 3D COMPLET, dédoublonné par voxel léger (densité) ─────
    # On garde TOUT (fond + structures) : c'est la carte honnête. Le voxel ne
    # fait que fusionner les points redondants pour une surface propre — il ne
    # retire PAS de structure (leçon 08-08 : un filtre de verticalité agressif
    # vidait la carte, cf. SLAM_3D_MIGRATION.md).
    S = voxel_grid(S_full, 0.2)
    # étiquette = ce qui a RÉELLEMENT tourné (la géométrie retenue), pas le nom
    # du topic : un /profiler_points monté à l'avant n'est pas « méthode grottes ».
    gs = {geo_of[t] for t in retenus}
    kind = ("structurel — fan vertical avant (« + »), 1 paroi/faisceau (voxel 0.2 m)"
            if gs == {"vertical"} else
            "structurel — méthode grottes, 1 paroi/faisceau (voxel 0.2 m)"
            if gs == {"transverse"} else
            "structurel — FUSION fan vertical avant + profiler transverse 360° "
            "(1 paroi/faisceau, voxel 0.2 m)"
            if gs == {"vertical", "transverse"} else "nuage 3D complet (voxel 0.2 m)")
    G = voxel_grid(np.vstack([p for lst in G_by.values() for p in lst]), 0.2) \
        if has_gt and any(G_by.values()) else None
    print(f"carte : {len(S_full):,} pts bruts → {len(S):,} pts ({kind})")
    titre_nn = ""

    # ── métrique : alignement UMEYAMA des trajectoires (pas première-pose) ──
    if has_gt and G is not None and len(G):
        from scipy.spatial import cKDTree
        gxy = associer_par_temps(ts, gt_t, gt_x, gt_y)
        _, R, tr = umeyama(np.column_stack([traj["x"], traj["y"]]), gxy,
                           with_scale=False)
        z_off = np.median(np.interp(ts, gt_t, gt_z) - traj["z"])
        S_al = np.c_[(S[:, :2] @ R.T) + tr, S[:, 2] + z_off]
        idx = rng.choice(len(S_al), min(20000, len(S_al)), replace=False)
        dnn = cKDTree(G).query(S_al[idx], k=1)[0]
        med, p90 = np.median(dnn), np.percentile(dnn, 90)
        titre_nn = f" — NN vs carte GT {med:.2f}/{p90:.2f} m (Umeyama)"
        print(f"carte GT-free vs carte GT (Umeyama) : NN méd {med:.3f} m | "
              f"p90 {p90:.3f} m")

    # ── COMBLAGE par le sonar horizontal (pointcloud.csv) ───────────────────
    # Le profiler (regard vers le bas, fan ±60°) est AVEUGLE aux structures peu
    # profondes hors trajectoire (ex. la coque du bateau à 15 m, à ~73° de la
    # verticale = hors fan) : il n'en voit que le fond en dessous. Le sonar
    # horizontal (portée 40 m, tranche à la profondeur du ROV) les voit. On ajoute
    # donc les points du sonar horizontal SEULEMENT là où le profiler n'a RIEN à
    # proximité 3D (NN > SEUIL) → le bateau + le haut du tablier apparaissent, SANS
    # dédoubler les treillis que le profiler couvre déjà. Colorés par z (2.5D
    # honnête, pas d'aplat). GT-free (nuage SLAM natif).
    G_fill = None
    pc_path = os.path.join(a.run, "pointcloud.csv")
    if os.path.isfile(pc_path):
        pc = np.genfromtxt(pc_path, delimiter=",", names=True)
        qz = pc["z"] if "z" in pc.dtype.names else np.zeros(len(pc))
        Q = np.column_stack([pc["x"], pc["y"], qz])
        from scipy.spatial import cKDTree
        dnn_q = cKDTree(S).query(Q, k=1)[0]
        G_fill = Q[dnn_q > 4.0]                 # 4 m : garde bateau (NN~7) + tablier
        if len(G_fill) > 60000:
            G_fill = G_fill[rng.choice(len(G_fill), 60000, replace=False)]
        print(f"comblage sonar horizontal : {len(G_fill):,} pts (bateau + tablier "
              f"hors portée du profiler)")

    # ── sorties : LA carte (repère SLAM, GT-free) ───────────────────────────
    run_name = os.path.basename(a.run.rstrip("/"))
    out = os.path.join(a.run, "carte_3d")
    tx, ty, tz = traj["x"], traj["y"], traj["z"]   # trajectoire SLAM (même repère)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(S[:, 0], S[:, 1], S[:, 2], s=0.3, c=S[:, 2], cmap="viridis",
               linewidths=0, alpha=0.4)
    if G_fill is not None and len(G_fill):     # bateau + tablier (sonar horiz., 2.5D)
        ax.scatter(G_fill[:, 0], G_fill[:, 1], G_fill[:, 2], s=1.0, c=G_fill[:, 2],
                   cmap="viridis", linewidths=0, alpha=0.7, marker="s",
                   label="comblage sonar horiz. (bateau/tablier, 2.5D)")
    ax.plot(tx, ty, tz, color="red", lw=1.8, label="trajectoire SLAM")
    ax.scatter(tx[0], ty[0], tz[0], c="lime", s=70, marker="^",
               depthshade=False, label="départ")
    ax.scatter(tx[-1], ty[-1], tz[-1], c="red", s=55, marker="s",
               depthshade=False, label="arrivée")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"{run_name} — carte 3D GT-free — {kind}\n"
                 f"{len(S):,} pts{titre_nn}", fontsize=9)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=140)
    plt.close(fig)
    np.save(out + ".npy", S)
    try:
        import plotly.graph_objects as go
        i2 = rng.choice(len(S), min(150000, len(S)), replace=False)
        f2 = go.Figure(go.Scatter3d(
            x=S[i2, 0], y=S[i2, 1], z=S[i2, 2], mode="markers",
            name="volume profiler (z)",
            marker=dict(size=1.2, color=S[i2, 2], colorscale="Viridis",
                        colorbar=dict(title="z (m)"))))
        if G_fill is not None and len(G_fill):  # bateau/tablier (sonar horiz., 2.5D)
            f2.add_trace(go.Scatter3d(
                x=G_fill[:, 0], y=G_fill[:, 1], z=G_fill[:, 2], mode="markers",
                name="comblage sonar horiz. (bateau/tablier, 2.5D)",
                marker=dict(size=2.0, color=G_fill[:, 2], colorscale="Viridis",
                            symbol="square", showscale=False)))
        # trajectoire SLAM par-dessus le nuage + marqueurs départ/arrivée
        f2.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode="lines",
                                  line=dict(color="red", width=5),
                                  name="trajectoire SLAM"))
        f2.add_trace(go.Scatter3d(x=[tx[0]], y=[ty[0]], z=[tz[0]], mode="markers",
                                  marker=dict(color="lime", size=6, symbol="diamond"),
                                  name="départ"))
        f2.add_trace(go.Scatter3d(x=[tx[-1]], y=[ty[-1]], z=[tz[-1]], mode="markers",
                                  marker=dict(color="red", size=6, symbol="square"),
                                  name="arrivée"))
        f2.update_layout(title=f"{run_name} — carte 3D GT-free{titre_nn}",
                         scene=dict(aspectmode="data"),
                         margin=dict(l=0, r=0, t=40, b=0))
        f2.write_html(out + ".html", include_plotlyjs=True)
        print("->", out + ".html")
    except ImportError:
        print("⚠ plotly absent (pip3 install --user plotly) : pas de .html interactif")
    print("->", out + ".png/.npy")


if __name__ == "__main__":
    main()
