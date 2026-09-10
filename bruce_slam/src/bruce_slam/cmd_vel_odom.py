#!/usr/bin/env python
"""Intègre /cmd_vel (vitesses commandées, unicycle) en pose absolue et la publie
sur ODOM_BRIDGE_INPUT_TOPIC (PoseStamped) — dead-reckoning PUR.

REFONTE (REFONTE_MISSION.md, décision ⛔2 « odométrie SACRÉE ») : ce nœud ne
s'abonne à RIEN d'autre que /cmd_vel. Ni GT, ni USBL — l'ancien code de seed
USBL / filtre complémentaire / gain est SUPPRIMÉ (un capteur absolu n'entre que
comme facteurs unaires back-end, cf. PIEGES #25 : le seed route-fond 1 m
tournait tout le repère de −30.88°). Même profil d'odométrie dans les 4
méthodes ; gate de vérification : traces dr identiques ×4.

Départ épinglé (0,0). Cap initial = 0, tranché par MESURE en ÉTAPE 0
(analysis/etape0_seed_cap.py : cap vrai à t0 = −1.1..+2.5° vs DGPS selon la
base, meilleur que le fit de forme USBL 2.2-4.0° ; la course USBL naïve, elle,
serait fausse de ~78° — le ROV tourne de −88° sur les 30 premiers mètres).

Modèle unicycle 2D : sur Aracati seules linear.x (vx) et angular.z (wz) de
/cmd_vel sont non nulles (vérifié sur le bag), donc x+=vx·cosθ·dt,
y+=vx·sinθ·dt, θ+=wz·dt (Euler avant, base de temps = header.stamp du bag).
x, y et θ vivent dans le MÊME repère : aucun offset de seed nulle part, donc
dr_theta loggé cohérent avec dr_x/dr_y par construction (⛔2, corollaire
logging de PIEGES #25).
"""
import math
import rospy
import tf
from geometry_msgs.msg import PoseStamped, TwistStamped

from bruce_slam.utils.topics import ODOM_BRIDGE_INPUT_TOPIC

CMD_VEL_TOPIC = "/cmd_vel"


class CmdVelOdom:
    def __init__(self):
        # Seed FIXE (ÉTAPE 0) : (0,0), cap 0. Aucun paramètre : la mauvaise
        # config est intapable.
        self.x = self.y = self.theta = 0.0
        self.last_t = None
        self.pub = rospy.Publisher(ODOM_BRIDGE_INPUT_TOPIC, PoseStamped, queue_size=10)
        rospy.Subscriber(CMD_VEL_TOPIC, TwistStamped, self._cmd_cb, queue_size=50)
        rospy.loginfo("cmd_vel_odom REFONTE : intégration pure /cmd_vel, "
                      "départ (0,0), cap 0 — aucun autre abonnement")

    def _cmd_cb(self, msg: TwistStamped) -> None:
        t = msg.header.stamp.to_sec()
        if self.last_t is None:
            self.last_t = t
            self._publish(msg.header.stamp)  # publie la pose de départ
            return
        dt = t - self.last_t
        self.last_t = t
        if dt <= 0:
            return
        vx = msg.twist.linear.x
        wz = msg.twist.angular.z
        # Euler avant : position au cap courant
        self.x += vx * math.cos(self.theta) * dt
        self.y += vx * math.sin(self.theta) * dt
        self.theta += wz * dt
        self._publish(msg.header.stamp)

    def _publish(self, stamp) -> None:
        out = PoseStamped()
        out.header.stamp = stamp  # base de temps du bag → ATE associable
        out.header.frame_id = "map"
        out.pose.position.x = self.x
        out.pose.position.y = self.y
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, self.theta)
        out.pose.orientation.x = q[0]
        out.pose.orientation.y = q[1]
        out.pose.orientation.z = q[2]
        out.pose.orientation.w = q[3]
        self.pub.publish(out)
