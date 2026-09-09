"""Round 2 « noise » — source de verite unique (NOISE_MISSION.md, decision
Nathan 2026-07-15). Active par la variable d'environnement NOISE_ROUND2=1.

Sinon (variable absente ou != "1") TOUT est neutre : multiplicateurs = 1.0,
suffixe = "" => la generation round 1 est BYTE-IDENTIQUE (multiplication par
1.0 exacte en IEEE754, meme litteral sonar).

Facteurs decides (mesures dans PROGRESS 2026-07-15) :
  L1 image sonar (AddSigma/MultSigma HoloOcean) ........ x2  (0.01 -> 0.02)
  L2 bruit gaussien par message (SIGMA_GYRO/ACC/DVL/DEPTH) x5
  L3 erreurs nav structurees (PSI0/RW/DVL_SCALE/DVL_MIS) x2  (traj9 SEUL :
     traj5 n'applique pas L3 -> le x2 n'a d'effet que sur traj9)
  Derive DR traj9 x2 mesuree a sec = 11.52 m rms (Umeyama 2.30) -> l'assert
  de sanite v8 est elargi a NAV_DRIFT_HI sous round 2 (round 1 garde 8 m).

2026-07-16 : L1 x5 -> x2 (PIEGES #24). A 0.05 le bruit depasse le seuil CFAR 30 :
mesure a la source (bags _noise_test x5) 6.4-6.8 % des pixels >= 30 (p99 = 40)
contre 0.09-0.11 % round 1 (p99 = 7.9) -> nuage x40-90 (1.6-2.2 M pts), KF
perdus (623/837), 363 fausses contraintes, ATE 67 m : runs inexploitables.
A 0.02 : p99 bruit attendu ~12 << 30 -> degradation sonar visible sans
inondation. L2/L3 inchanges (l'effet DR voulu, mesure conforme : traj9
DR 3.04 -> 5.7-6.1 m).

Les bags round 2 portent le suffixe _noise et NE DOIVENT JAMAIS ecraser les
bags round 1 (la comparaison en depend).
"""
import os

ON = os.environ.get("NOISE_ROUND2") == "1"
SUFFIX = "_noise" if ON else ""

# L1 — image sonar (valeurs absolues, pas un multiplicateur : x2 de 0.01 ;
# 0.05 = inondation CFAR mesuree, cf. docstring + PIEGES #24)
SONAR_ADD = 0.02 if ON else 0.01
SONAR_MULT = 0.02 if ON else 0.01

# L2 — bruit gaussien par message capteur
L2_MULT = 5.0 if ON else 1.0

# L3 — erreurs nav structurees (traj9 seul)
L3_MULT = 2.0 if ON else 1.0

# assert de sanite derive DR (gen_bag_3d_v8.py) : borne haute
NAV_DRIFT_HI = 30.0 if ON else 8.0


def bag(path):
    """Insere le suffixe _noise avant .bag (inchange si round 1)."""
    return path.replace(".bag", SUFFIX + ".bag")
