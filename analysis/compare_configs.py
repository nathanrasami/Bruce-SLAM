#!/usr/bin/env python3
"""Compare les configurations d'une campagne sample_data sur la métrique « distance du
nuage aux structures du port ». Groupe les runs par label (results/run_sample_data_*_<label>_p<port>)
et sort moyenne, dispersion et taux de points sous 1 m.

⚠ La MOYENNE fait foi. Lu en entier, le champ de distance quantifie la médiane sur des
valeurs de pixel (3 px = 0.69 m, 4 px = 0.91 m) et fabrique de fausses égalités entre
runs pourtant différents de plus d'un mètre.

Usage : python3 analysis/compare_configs.py"""
import glob, json, os, re, sys
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cal = json.load(open(f"{BASE}/analysis/marina_calage.json"))
A, b = np.array(cal["A"]), np.array(cal["b"]); ox, oy = cal["origine_px"]; mpp = cal["m_par_px"]
g = ndi.gaussian_filter(np.asarray(Image.open(f"{BASE}/analysis/marina_map.png").convert("L"), float), 1.0)
K = [np.array([[-1,-2,-1],[0,0,0],[1,2,1]], float), np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float),
     np.array([[0,1,2],[-1,0,1],[-2,-1,0]], float), np.array([[-2,-1,0],[-1,0,1],[0,1,2]], float)]
mag = np.sqrt(sum(ndi.convolve(g, k) ** 2 for k in K))
champ = ndi.distance_transform_edt(mag <= np.percentile(mag, 97))
H, W = champ.shape

def mesure(run):
    c = np.genfromtxt(f"{run}/pointcloud.csv", delimiter=",", names=True)
    t = np.genfromtxt(f"{run}/trajectory.csv", delimiter=",", names=True)
    X = np.column_stack([c["y"], c["x"]]) @ A.T + b
    col, row = X[:, 0] - ox, -X[:, 1] - oy
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    d = ndi.map_coordinates(champ, [row[ok], col[ok]], order=1, mode="nearest") * mpp
    return d.mean(), (d < 1).mean()*100, len(c), int(t["nssm_constraints"].sum())

groupes = {}
for run in sorted(glob.glob(f"{BASE}/results/run_sample_data_*")):
    n = os.path.basename(run)
    m = re.search(r"\d{6}_(.+?)(?:_p\d+)?$", n)
    if not m or not os.path.exists(f"{run}/pointcloud.csv"):
        continue
    groupes.setdefault(m.group(1), []).append(run)

LEG = {"temoin": "témoin (config d'origine)", "B": "min_pcm 2->3",
       "C": "nssm source_frames 5->8", "D": "ssm target_frames 3->5",
       "E": "icp_odom_sigmas /2", "F": "min_pcm 3 + icp_sig /2", "G": "icp_odom_sigmas /4",
       "H": "min_pcm 2->4", "I": "nssm src 8 + min_st_sep 10", "J": "min_pcm 2->5"}
print(f"{'config':28s} {'n':>2s} {'moyenne (m)':>18s} {'<1m %':>12s} {'pts':>6s} {'loops':>10s}")
lignes = []
for k, runs in groupes.items():
    v = np.array([mesure(r) for r in runs])
    if len(v) < 2: continue
    lignes.append((v[:,0].mean(), k, v))
for moy, k, v in sorted(lignes):
    print(f"{LEG.get(k,k):28s} {len(v):2d} "
          f"{v[:,0].mean():7.2f} [{v[:,0].min():.2f}-{v[:,0].max():.2f}] "
          f"{v[:,1].mean():7.1f} {v[:,2].mean():7.0f} "
          f"{v[:,3].mean():5.1f} [{int(v[:,3].min())}-{int(v[:,3].max())}]")
