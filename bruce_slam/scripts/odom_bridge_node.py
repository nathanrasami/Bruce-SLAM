#!/usr/bin/env python
import rospy
from bruce_slam.odom_bridge import OdomBridge

if __name__ == "__main__":
    rospy.init_node("odom_bridge_node", log_level=rospy.INFO)
    OdomBridge()
    rospy.spin()
