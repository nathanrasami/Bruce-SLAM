#!/usr/bin/env python3
"""3D-ready : consomme /sonar_points (PointCloud2 x,y,z,intensity du simulateur)
et publie directement les FEATURES du SLAM (court-circuite image->CFAR).

Le graphe SLAM reste 2D (projection x,y) — philosophie 4-DOF (FABLE §1) ; la
vraie carte 3D se reconstruira OFFLINE (bag + trajectory.csv) quand le bag du
collègue aura z != 0 (aujourd'hui z=0 partout, cf. SLAM_3D_MIGRATION.md).

Packing = convention aracati : [x_avant, INTENSITÉ, -y_latéral] — le SLAM lit
(x, -z). ⚠ chiralité à vérifier au 1er run réel (PIEGES §1/§9) : param ~flip_y.
"""
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

from bruce_slam.utils.topics import SONAR_FEATURE_TOPIC
from bruce_slam.utils.conversions import n2r


class SonarPointsBridge:
    def __init__(self):
        self.intensity_min = rospy.get_param("~intensity_min", 50.0)
        self.voxel = rospy.get_param("~voxel", 0.5)      # m, dédoublonnage grille
        self.flip_y = rospy.get_param("~flip_y", False)  # chiralité (PIEGES §1)
        self.pub = rospy.Publisher(SONAR_FEATURE_TOPIC, PointCloud2, queue_size=10)
        rospy.Subscriber("/sonar_points", PointCloud2, self.cb, queue_size=5)
        rospy.loginfo("[sonar_points_bridge] /sonar_points -> %s (I>=%.0f, voxel %.2f m)",
                      SONAR_FEATURE_TOPIC, self.intensity_min, self.voxel)

    def cb(self, msg: PointCloud2):
        pts = np.array(list(pc2.read_points(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)),
            dtype=np.float32)
        if len(pts) == 0:
            return
        pts = pts[pts[:, 3] >= self.intensity_min]
        if len(pts) == 0:
            return
        # dédoublonnage voxel : le simu publie ~120k pts/ping, le SLAM n'en veut
        # que quelques centaines de significatifs
        key = np.round(pts[:, :2] / self.voxel).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        pts = pts[idx]
        y = -pts[:, 1] if self.flip_y else pts[:, 1]
        packed = np.c_[pts[:, 0], pts[:, 3], -y]  # [x, I, -y] convention aracati
        out = n2r(packed, "PointCloudXYZ")
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "base_link"
        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("sonar_points_bridge")
    SonarPointsBridge()
    rospy.spin()
