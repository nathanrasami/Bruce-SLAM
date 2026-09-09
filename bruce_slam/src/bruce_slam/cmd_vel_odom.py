#!/usr/bin/env python
"""Pont d'odométrie pour Aracati2017 (Bruce n'a ni IMU ni DVL).

Intègre /cmd_vel (modèle unicycle) en pose absolue, publiée en nav_msgs/Odometry sur
LOCALIZATION_ODOM_TOPIC — l'odométrie attendue par le SLAM de Bruce (cœur intact).

    x += vx·cos(θ)·dt ; y += vx·sin(θ)·dt ; θ += wz·dt     (REP-103, base de temps = bag)

OPTION USBL (use_usbl=True) — 100% GT-free : ancre la dérive avec /usbl_point
(positionnement acoustique, ~1.4 m de bruit), SANS toucher au cœur Bruce.
  1. SEED : on aligne le repère cmd_vel (départ 0,0,0) sur le repère USBL en attendant
     un déplacement ≥ seed_min_disp : position = fix courant, cap = course-over-ground.
     Indispensable : sinon la correction USBL serait dans un repère tourné.
  2. FILTRE COMPLÉMENTAIRE : chaque fix accepté corrige la position à FAIBLE GAIN
     (x += gain·(ux−x)), pas de snap dur → borne la dérive SANS zigzag.
  3. Rejet d'outliers : gate de vitesse vs le dernier fix accepté (saute les glitchs ~73 m).
"""
import math
import threading
import rospy
import tf
from geometry_msgs.msg import TwistStamped, PointStamped
from nav_msgs.msg import Odometry

from bruce_slam.utils.topics import LOCALIZATION_ODOM_TOPIC

CMD_VEL_TOPIC = "/cmd_vel"
USBL_TOPIC = "/usbl_point"


class CmdVelOdom:
    """Intègre /cmd_vel (+ correction USBL optionnelle) et publie une Odometry."""

    def __init__(self):
        self.x = self.y = self.theta = 0.0
        self.last_t = None
        self.lock = threading.Lock()

        # --- option USBL (GT-free) ---
        self.use_usbl = rospy.get_param("~use_usbl", False)
        self.usbl_gain = rospy.get_param("~usbl_gain", 0.08)          # gain de correction (faible)
        self.usbl_seed_min_disp = rospy.get_param("~usbl_seed_min_disp", 1.0)  # m, pour seeder le cap
        self.usbl_max_speed = rospy.get_param("~usbl_max_speed", 5.0)  # m/s, gate outliers
        self.seeded = not self.use_usbl   # sans USBL : pas de seed (départ 0,0,0)
        self._seed_fixes = []             # accumulation pour le seed
        self.last_usbl = None             # (t, x, y) dernier fix accepté

        self.pub = rospy.Publisher(LOCALIZATION_ODOM_TOPIC, Odometry, queue_size=10)
        rospy.Subscriber(CMD_VEL_TOPIC, TwistStamped, self._cmd_cb, queue_size=50)
        if self.use_usbl:
            rospy.Subscriber(USBL_TOPIC, PointStamped, self._usbl_cb, queue_size=20)
        rospy.loginfo("cmd_vel_odom : %s -> %s%s",
                      CMD_VEL_TOPIC, LOCALIZATION_ODOM_TOPIC,
                      " (+USBL gain=%.2f)" % self.usbl_gain if self.use_usbl else "")

    # ------------------------------------------------------------------ cmd_vel
    def _cmd_cb(self, msg: TwistStamped) -> None:
        t = msg.header.stamp.to_sec()
        vx = msg.twist.linear.x
        wz = msg.twist.angular.z
        with self.lock:
            if self.last_t is not None and self.seeded:
                dt = t - self.last_t
                if dt > 0:
                    self.x += vx * math.cos(self.theta) * dt
                    self.y += vx * math.sin(self.theta) * dt
                    self.theta += wz * dt
            self.last_t = t
            self._publish(msg.header.stamp, vx, wz)

    # ------------------------------------------------------------------ USBL
    def _usbl_cb(self, msg: PointStamped) -> None:
        t = msg.header.stamp.to_sec()
        ux, uy = msg.point.x, msg.point.y
        with self.lock:
            # rejet d'outliers : vitesse implicite vs dernier fix accepté
            if self.last_usbl is not None:
                lt, lx, ly = self.last_usbl
                dt = t - lt
                if dt > 0 and math.hypot(ux - lx, uy - ly) / dt > self.usbl_max_speed:
                    return
            self.last_usbl = (t, ux, uy)

            if not self.seeded:
                # SEED : aligne le repère cmd_vel sur le repère USBL
                self._seed_fixes.append((ux, uy))
                x0, y0 = self._seed_fixes[0]
                if math.hypot(ux - x0, uy - y0) >= self.usbl_seed_min_disp:
                    self.x, self.y = ux, uy
                    self.theta = math.atan2(uy - y0, ux - x0)  # course-over-ground
                    self.last_t = t
                    self.seeded = True
                    rospy.loginfo("cmd_vel_odom : seed USBL -> (%.1f, %.1f, %.0f°)",
                                  self.x, self.y, math.degrees(self.theta))
                return

            # FILTRE COMPLÉMENTAIRE : correction de position à faible gain (pas de snap)
            self.x += self.usbl_gain * (ux - self.x)
            self.y += self.usbl_gain * (uy - self.y)

    # ------------------------------------------------------------------ publish
    def _publish(self, stamp, vx, wz) -> None:
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, self.theta)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = vx
        odom.twist.twist.angular.z = wz
        self.pub.publish(odom)
