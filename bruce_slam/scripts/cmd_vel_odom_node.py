#!/usr/bin/env python
import rospy
from bruce_slam.cmd_vel_odom import CmdVelOdom

if __name__ == "__main__":
    rospy.init_node("cmd_vel_odom_node", log_level=rospy.INFO)
    CmdVelOdom()
    rospy.spin()
