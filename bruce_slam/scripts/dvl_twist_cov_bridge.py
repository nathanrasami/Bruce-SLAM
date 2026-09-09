#!/usr/bin/env python3
"""Bridge: /dvl (TwistStamped) -> /dvl_cov (TwistWithCovarianceStamped)
robot_localization only accepts TwistWithCovarianceStamped for twist inputs."""
import rospy
from geometry_msgs.msg import TwistStamped, TwistWithCovarianceStamped

# DVL velocity measurement noise (variance on vx, vy)
VEL_VAR = 0.01

_pub = None

def callback(msg: TwistStamped):
    out = TwistWithCovarianceStamped()
    out.header = msg.header
    out.twist.twist = msg.twist
    cov = [0.0] * 36
    cov[0]  = VEL_VAR   # vx
    cov[7]  = VEL_VAR   # vy
    cov[14] = VEL_VAR   # vz
    cov[21] = 0.05      # vroll
    cov[28] = 0.05      # vpitch
    cov[35] = 0.05      # vyaw
    out.twist.covariance = cov
    _pub.publish(out)

def main():
    global _pub
    rospy.init_node("dvl_twist_cov_bridge")
    _pub = rospy.Publisher("/dvl_cov", TwistWithCovarianceStamped, queue_size=50)
    rospy.Subscriber("/dvl", TwistStamped, callback, queue_size=50)
    rospy.loginfo("[dvl_bridge] /dvl -> /dvl_cov")
    rospy.spin()

if __name__ == "__main__":
    main()
