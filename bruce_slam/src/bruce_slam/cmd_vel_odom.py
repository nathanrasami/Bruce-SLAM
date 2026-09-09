#!/usr/bin/env python
"""Intègre /cmd_vel (vitesses commandées, unicycle) en pose absolue et la publie
sur ODOM_BRIDGE_INPUT_TOPIC (PoseStamped) — odométrie INDÉPENDANTE de la GT.

Drop-in du relay GT : OdomBridge lit /direct_sonar/pose à l'identique. La GT
(/pose_gt) ne sert qu'à seeder position + cap à t=0 ; ensuite on intègre /cmd_vel
seul → dérive réaliste (≈11.5 m ATE Umeyama mesuré). C'est aux loop closures de
rattraper cette dérive.

Modèle unicycle 2D : sur Aracati seules linear.x (vx) et angular.z (wz) de
/cmd_vel sont non nulles (vérifié sur le bag), donc x+=vx·cosθ·dt, y+=vx·sinθ·dt,
θ+=wz·dt (Euler avant, base de temps = header.stamp du bag).

OPTION USBL (~use_usbl=True) : fusionne les fixes acoustiques /usbl_point (≈1.4 m
médian vs GT, INDÉPENDANTS de la GT) par filtre complémentaire en position. Ancre
la dérive sans recourir à la GT. Rejet d'outliers (glitches ≈73 m) par gate de
vitesse vs le dernier fix accepté — indépendant de la dérive de cmd_vel.
"""
import math
import threading
import rospy
import tf
from geometry_msgs.msg import PoseStamped, PointStamped, TwistStamped, PoseWithCovarianceStamped

from bruce_slam.utils.topics import ODOM_BRIDGE_INPUT_TOPIC

CMD_VEL_TOPIC = "/cmd_vel"
GT_TOPIC = "/pose_gt"
INITIAL_POSE_TOPIC = "/initialpose"
USBL_TOPIC = "/usbl_point"


class CmdVelOdom:
    def __init__(self):
        self.x = self.y = self.theta = 0.0
        self.seeded = False
        self.last_t = None
        self.lock = threading.Lock()
        # topic de sortie : par défaut l'entrée du bridge (mode odométrie SLAM directe),
        # mais peut être redirigé (ex: /cmd_vel/pose) pour servir de PRIOR à DISO.
        out_topic = rospy.get_param("~output_topic", ODOM_BRIDGE_INPUT_TOPIC)
        self.pub = rospy.Publisher(out_topic, PoseStamped, queue_size=10)
        # seed_from_gt : True = position + cap seedés depuis /pose_gt à t=0 (puis GT ignorée).
        # False = seed à l'origine (0,0,0) sans aucune GT → pipeline 100% GT-free (l'USBL
        # ancre la position absolue ; le cap initial absolu est non observable sans GT/IMU,
        # laissé à 0 et corrigé par la dérive). cf. extirpation /pose_gt.
        self.seed_from_gt = rospy.get_param("~seed_from_gt", True)
        # seed_from_initial_pose : True = seed depuis /initialpose (PoseWithCovarianceStamped,
        # publié une fois par le nœud odom au démarrage — position DGPS + cap compas à t=0,
        # présent dans le bag). Aucune GT dans la boucle : /initialpose est la pose de
        # départ connue (réaliste), pas la vérité terrain continue. Prioritaire sur seed_from_gt.
        self.seed_from_initial_pose = rospy.get_param("~seed_from_initial_pose", False)
        # seed_from_usbl : True = seed 100% GT-FREE. Position = 1er fix /usbl_point accepté
        # (capteur acoustique, pas GT) ; cap = course-over-ground (atan2 du déplacement USBL
        # jusqu'à usbl_seed_min_disp m). AUCUN /pose_gt. Prioritaire sur les autres seeds.
        self.seed_from_usbl = rospy.get_param("~seed_from_usbl", False)
        # usbl_seed_min_disp : deplacement min (m) pour figer le cap. 1.0 valide offline
        # (test_usbl_seed) : le ROV vire vite → fenetre COURTE = meilleur cap (1 m → 13° d'ecart
        # vs seed GT ; 3 m → 78°, la course-over-ground capte deja le virage). cf. test bag.
        self.usbl_seed_min_disp = rospy.get_param("~usbl_seed_min_disp", 1.0)
        self._usbl_seed_fixes = []   # (t,x,y) fixes accumulés pour le seed
        # heading_from_compass : cap = ANCRE monde (atan2 du 1er déplacement, repère
        # compatible USBL/DGPS) + DELTAS du compas (/pose_gt.orientation, capteur réel).
        # → sans dérive (compas) ET dans le bon repère (atan2) → point cloud propre +
        # USBL ne déforme pas. Le compas ABSOLU seul est ~50° tourné du repère monde
        # (run 153433 → Umeyama 28 m) ; ses DELTAS, eux, sont justes.
        self.heading_from_compass = rospy.get_param("~heading_from_compass", False)

        # Fusion USBL optionnelle (ancrage absolu acoustique, indépendant de GT)
        self.use_usbl = rospy.get_param("~use_usbl", False)
        self.usbl_gain = rospy.get_param("~usbl_gain", 0.1)
        self.usbl_max_speed = rospy.get_param("~usbl_max_speed", 3.0)  # m/s, gate outliers
        self.usbl_snap = rospy.get_param("~usbl_snap", True)
        self.usbl_th_gain = rospy.get_param("~usbl_th_gain", 0.0)
        self.usbl_th_min_disp = rospy.get_param("~usbl_th_min_disp", 1.0)
        self.last_usbl = None
        self._usbl_head_ref = None
        self._usbl_snapped = False
        self.usbl_kept = self.usbl_rejected = 0

        # cap : ancre atan2 (monde) + deltas compas
        self._seed_prev = None       # 1ère pose GT, en attente d'un déplacement
        self._theta_seed = 0.0       # cap initial (repère monde)
        self._compass_seed = None    # cap compas à l'instant du seed (réf des deltas)

        if self.seed_from_usbl:
            rospy.Subscriber(USBL_TOPIC, PointStamped, self._usbl_seed_cb, queue_size=10)
            rospy.loginfo("cmd_vel_odom : seed 100%% GT-FREE depuis USBL "
                          "(position 1er fix + cap course-over-ground)")
        elif self.seed_from_initial_pose:
            rospy.Subscriber(INITIAL_POSE_TOPIC, PoseWithCovarianceStamped,
                             self._initialpose_cb, queue_size=1)
        elif self.seed_from_gt or self.heading_from_compass:
            rospy.Subscriber(GT_TOPIC, PoseStamped, self._seed_cb, queue_size=5)
        else:
            self.seeded = True
            rospy.loginfo("cmd_vel_odom : seed GT-free (origine 0,0,0)")
        if self.heading_from_compass:
            rospy.Subscriber(GT_TOPIC, PoseStamped, self._compass_cb, queue_size=10)
            rospy.loginfo("cmd_vel_odom : cap = ancre atan2 + deltas compas (sans dérive)")
        rospy.Subscriber(CMD_VEL_TOPIC, TwistStamped, self._cmd_cb, queue_size=50)
        if self.use_usbl:
            rospy.Subscriber(USBL_TOPIC, PointStamped, self._usbl_cb, queue_size=10)
            rospy.loginfo("cmd_vel_odom : fusion USBL ON (gain=%.2f)", self.usbl_gain)

    def _usbl_seed_cb(self, msg: PointStamped) -> None:
        """Seed 100% GT-FREE depuis l'USBL (capteur acoustique). Position = 1er fix accepté ;
        cap = course-over-ground (atan2 du déplacement jusqu'à usbl_seed_min_disp). Gate
        d'outlier par vitesse (glitch ~73 m) vs le dernier fix de seed accepté."""
        if self.seeded:
            return
        t = msg.header.stamp.to_sec()
        ux, uy = msg.point.x, msg.point.y
        if self._usbl_seed_fixes:
            lt, lx, ly = self._usbl_seed_fixes[-1]
            dt = t - lt
            if dt > 0 and math.hypot(ux - lx, uy - ly) / dt > self.usbl_max_speed:
                return  # saut physiquement impossible → glitch acoustique
        self._usbl_seed_fixes.append((t, ux, uy))
        x0, y0 = self._usbl_seed_fixes[0][1], self._usbl_seed_fixes[0][2]
        disp = math.hypot(ux - x0, uy - y0)
        if disp < self.usbl_seed_min_disp:
            return  # pas encore assez de déplacement pour un cap fiable
        with self.lock:
            self.x, self.y = ux, uy
            self.theta = math.atan2(uy - y0, ux - x0)
            self.seeded = True
            self._usbl_snapped = True       # déjà sur un fix → pas de re-snap si fusion ON
            self.last_usbl = (t, ux, uy)
        rospy.loginfo("cmd_vel_odom seedé USBL (GT-free) : x=%.2f y=%.2f cap=%.3f "
                      "(%d fixes, disp=%.2f m)", self.x, self.y, self.theta,
                      len(self._usbl_seed_fixes), disp)

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        """Seed depuis /initialpose : position DGPS + cap compas publiés une fois par le
        nœud odom au démarrage. Aucune GT continue — seul le point de départ est connu."""
        if self.seeded:
            return
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            self.theta = yaw
            self.seeded = True
        rospy.loginfo("cmd_vel_odom seedé depuis /initialpose : x=%.2f y=%.2f yaw=%.3f rad",
                      self.x, self.y, self.theta)

    def _seed_cb(self, msg: PoseStamped) -> None:
        """Seed position (GT initiale = pose de départ connue) + cap initial = atan2 du
        1er déplacement (repère MONDE, compatible USBL — Run 1 : 1.96 m). Enregistre le
        cap compas à cet instant comme référence des deltas (cf. _compass_cb)."""
        if self.seeded:
            return
        x, y = msg.pose.position.x, msg.pose.position.y
        if self._seed_prev is None:
            self._seed_prev = (x, y)
            return
        x0, y0 = self._seed_prev
        dx, dy = x - x0, y - y0
        if math.hypot(dx, dy) < 0.05:
            return
        q = msg.pose.orientation
        compass = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.x, self.y = x, y
            self._theta_seed = math.atan2(dy, dx)
            self.theta = self._theta_seed
            self._compass_seed = compass
            self.seeded = True
        rospy.loginfo("cmd_vel_odom seedé : x=%.2f y=%.2f cap_monde=%.3f compas_ref=%.3f",
                      self.x, self.y, self._theta_seed, compass)

    def _compass_cb(self, msg: PoseStamped) -> None:
        """Cap = ancre monde (atan2) + delta du compas depuis le seed → sans dérive,
        dans le repère monde. Position non touchée (cmd_vel + USBL)."""
        if not self.seeded or self._compass_seed is None:
            return
        q = msg.pose.orientation
        compass = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        d = math.atan2(math.sin(compass - self._compass_seed),
                       math.cos(compass - self._compass_seed))
        with self.lock:
            self.theta = self._theta_seed + d

    def _cmd_cb(self, msg: TwistStamped) -> None:
        if not self.seeded:
            return  # on attend le seed GT (cmd_vel démarre ~0.5 s avant /pose_gt)
        t = msg.header.stamp.to_sec()
        if self.last_t is None:
            self.last_t = t
            self._publish(msg.header.stamp)  # publie la pose seedée
            return
        dt = t - self.last_t
        self.last_t = t
        if dt <= 0:
            return
        vx = msg.twist.linear.x
        wz = msg.twist.angular.z
        with self.lock:
            # Euler avant : position au cap courant
            self.x += vx * math.cos(self.theta) * dt
            self.y += vx * math.sin(self.theta) * dt
            # cap : intégré depuis wz, SAUF si fourni en continu par le compas
            if not self.heading_from_compass:
                self.theta += wz * dt
        self._publish(msg.header.stamp)

    def _usbl_cb(self, msg: PointStamped) -> None:
        """Fusionne un fix USBL en position (filtre complémentaire), après gate
        d'outlier par vitesse vs le dernier fix accepté (indépendant de la dérive)."""
        if not self.seeded:
            return
        t = msg.header.stamp.to_sec()
        ux, uy = msg.point.x, msg.point.y
        if self.last_usbl is not None:
            lt, lx, ly = self.last_usbl
            dt = t - lt
            if dt > 0 and math.hypot(ux - lx, uy - ly) / dt > self.usbl_max_speed:
                self.usbl_rejected += 1
                return  # saut physiquement impossible → glitch acoustique
        self.last_usbl = (t, ux, uy)
        self.usbl_kept += 1
        with self.lock:
            if self.usbl_snap and not self._usbl_snapped:
                # 1er fix accepté : reset position (efface la dérive du seed)
                self.x, self.y = ux, uy
                self._usbl_snapped = True
            else:
                # correction complémentaire de position
                self.x += self.usbl_gain * (ux - self.x)
                self.y += self.usbl_gain * (uy - self.y)
            # correction de CAP depuis la direction du mouvement USBL (cap observable)
            if self.usbl_th_gain > 0.0 and self._usbl_head_ref is not None:
                ddx, ddy = ux - self._usbl_head_ref[0], uy - self._usbl_head_ref[1]
                if math.hypot(ddx, ddy) > self.usbl_th_min_disp:
                    err = math.atan2(math.sin(math.atan2(ddy, ddx) - self.theta),
                                     math.cos(math.atan2(ddy, ddx) - self.theta))
                    self.theta += self.usbl_th_gain * err
                    self._usbl_head_ref = (ux, uy)
            elif self._usbl_head_ref is None:
                self._usbl_head_ref = (ux, uy)

    def _publish(self, stamp) -> None:
        out = PoseStamped()
        out.header.stamp = stamp  # base de temps du bag → ATE associable
        out.header.frame_id = "map"
        with self.lock:
            out.pose.position.x = self.x
            out.pose.position.y = self.y
            q = tf.transformations.quaternion_from_euler(0.0, 0.0, self.theta)
        out.pose.orientation.x = q[0]
        out.pose.orientation.y = q[1]
        out.pose.orientation.z = q[2]
        out.pose.orientation.w = q[3]
        self.pub.publish(out)
