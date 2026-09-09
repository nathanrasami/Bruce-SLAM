#!/usr/bin/env python3
"""HoloOcean : /ground_truth (nav_msgs/Odometry) → /pose_gt (PoseStamped).

Le nœud SLAM (slam_ros.py) écrit groundtruth.csv depuis /pose_gt (mécanisme
Aracati). Ce pont rend l'export GT fonctionnel sur les bags HoloOcean sans
toucher au SLAM. Évaluation seulement — n'entre pas dans l'estimation.
"""
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

_pub = None


def callback(msg: Odometry):
    out = PoseStamped()
    out.header = msg.header
    out.pose = msg.pose.pose
    _pub.publish(out)


def main():
    global _pub
    rospy.init_node("gt_odom_to_pose")
    _pub = rospy.Publisher("/pose_gt", PoseStamped, queue_size=50)
    rospy.Subscriber("/ground_truth", Odometry, callback, queue_size=50)
    rospy.loginfo("[gt_odom_to_pose] /ground_truth (Odometry) -> /pose_gt (PoseStamped)")
    rospy.spin()


if __name__ == "__main__":
    main()
