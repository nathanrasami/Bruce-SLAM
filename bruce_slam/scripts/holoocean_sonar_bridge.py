#!/usr/bin/env python3
"""Bridge: sensor_msgs/Image (HoloOcean polar sonar) → OculusPingUncompressed"""
import numpy as np
import rospy
from sensor_msgs.msg import Image
from sonar_oculus.msg import OculusPingUncompressed
from bruce_slam.utils.topics import SONAR_TOPIC_UNCOMPRESSED

# ── Set these to match the HoloOcean sonar configuration ──────────────────────
RANGE_M  = 40.0   # max range in metres
FOV_DEG  = 120.0  # total horizontal FOV in degrees
# ──────────────────────────────────────────────────────────────────────────────

_pub = None
_bearings = None
_range_resolution = None

def _build_bearings(num_beams, fov_deg):
    """Build bearings array in units of (degrees * 100) as int16, matching Oculus convention."""
    half = fov_deg / 2.0
    angles_deg = np.linspace(-half, half, num_beams)
    return (angles_deg * 100).astype(np.int16).tolist()

def callback(msg: Image):
    global _bearings, _range_resolution

    num_ranges = msg.height   # rows = range bins
    num_beams  = msg.width    # cols = bearing bins

    if _bearings is None or len(_bearings) != num_beams:
        _bearings = _build_bearings(num_beams, FOV_DEG)
        _range_resolution = RANGE_M / num_ranges
        rospy.loginfo(f"[sonar_bridge] {num_ranges} ranges × {num_beams} beams, "
                      f"res={_range_resolution:.4f} m/bin, FOV={FOV_DEG}°")

    ping = OculusPingUncompressed()
    ping.header          = msg.header
    ping.ping_id         = 0
    ping.bearings        = _bearings
    ping.range_resolution = _range_resolution
    ping.num_ranges      = num_ranges
    ping.num_beams       = num_beams
    ping.ping            = msg
    _pub.publish(ping)

def main():
    global _pub
    rospy.init_node("holoocean_sonar_bridge")
    _pub = rospy.Publisher(SONAR_TOPIC_UNCOMPRESSED, OculusPingUncompressed, queue_size=10)
    rospy.Subscriber("/sonar", Image, callback, queue_size=10)
    rospy.loginfo("[sonar_bridge] ready — /sonar → " + SONAR_TOPIC_UNCOMPRESSED)
    rospy.spin()

if __name__ == "__main__":
    main()
