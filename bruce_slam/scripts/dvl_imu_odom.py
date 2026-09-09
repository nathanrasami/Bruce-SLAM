#!/usr/bin/env python3
"""HoloOcean : dead-reckoning DVL + IMU → PoseStamped sur /direct_sonar/pose.

- /dvl (geometry_msgs/TwistStamped) : vitesses vx, vy dans le repère VÉHICULE ;
- /imu (sensor_msgs/Imu) : orientation absolue (quaternion) → yaw.
Intégration planaire : p += R(yaw) · (vx, vy) · dt. L'orientation publiée est
celle de l'IMU (absolue, sans dérive). z ignoré (SLAM 2D — cf. SLAM_3D_MIGRATION.md
pour le passage 2.5D/3D). Publie sur ODOM_BRIDGE_INPUT_TOPIC (/direct_sonar/pose),
converti en nav_msgs/Odometry par odom_bridge pour le SLAM.

Réaliste embarqué (DVL + AHRS) : aucune donnée /ground_truth utilisée.
"""
import math

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

from bruce_slam.utils.topics import ODOM_BRIDGE_INPUT_TOPIC


class DvlImuOdom:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.last_t = None
        self.last_quat = None  # dernière orientation IMU (absolue)
        self.pub = rospy.Publisher(ODOM_BRIDGE_INPUT_TOPIC, PoseStamped, queue_size=50)
        self.z = 0.0
        rospy.Subscriber("/imu", Imu, self.imu_cb, queue_size=100)
        rospy.Subscriber("/depth", Float64, self.depth_cb, queue_size=10)
        rospy.Subscriber("/dvl", TwistStamped, self.dvl_cb, queue_size=100)
        rospy.loginfo("[dvl_imu_odom] /dvl + /imu -> %s (intégration planaire)",
                      ODOM_BRIDGE_INPUT_TOPIC)

    def imu_cb(self, msg: Imu):
        self.last_quat = msg.orientation

    def depth_cb(self, msg: Float64):
        # 2.5D : z = -profondeur (convention z vers le HAUT ; /depth positif vers le bas).
        self.z = -float(msg.data)

    def dvl_cb(self, msg: TwistStamped):
        t = msg.header.stamp.to_sec()
        if self.last_t is None or self.last_quat is None:
            self.last_t = t
            return
        dt = t - self.last_t
        self.last_t = t
        if dt <= 0 or dt > 1.0:  # trou / rejeu : on saute l'intégration
            return
        q = self.last_quat
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        vx, vy = msg.twist.linear.x, msg.twist.linear.y
        c, s = math.cos(yaw), math.sin(yaw)
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "map"
        out.pose.position.x = self.x
        out.pose.position.y = self.y
        out.pose.position.z = self.z
        out.pose.orientation = q  # cap absolu IMU (sans dérive)
        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("dvl_imu_odom")
    DvlImuOdom()
    rospy.spin()
