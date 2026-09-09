#!/usr/bin/env python3
"""Bridge: sensor_msgs/Image (HoloOcean polar sonar) → OculusPingUncompressed"""
import numpy as np
import rospy
from sensor_msgs.msg import Image
from sonar_oculus.msg import OculusPingUncompressed
from bruce_slam.utils.topics import SONAR_TOPIC_UNCOMPRESSED

# ── Défauts — surchargés par rosparam ~range_m / ~fov_deg (cf. holoocean.launch),
#    à faire correspondre à la config du sonar du simulateur (collègue). ──────────
RANGE_M  = 40.0   # max range in metres
FOV_DEG  = 120.0  # total horizontal FOV in degrees
# ──────────────────────────────────────────────────────────────────────────────

_pub = None
_bearings = None
_range_resolution = None

def _build_bearings(num_beams, fov_deg):
    """Bearings en CENTI-DEGRÉS (float), convention Oculus : feature_extraction
    les convertit en rad via to_rad = b * pi / 18000."""
    half = fov_deg / 2.0
    angles_deg = np.linspace(-half, half, num_beams)
    return (angles_deg * 100.0).astype(np.float32).tolist()

def callback(msg: Image):
    global _bearings, _range_resolution

    num_ranges = msg.height   # rows = range bins
    num_beams  = msg.width    # cols = bearing bins

    if _bearings is None or len(_bearings) != num_beams:
        _bearings = _build_bearings(num_beams, FOV_DEG)
        _range_resolution = RANGE_M / num_ranges
        rospy.loginfo(f"[sonar_bridge] {num_ranges} ranges × {num_beams} beams, "
                      f"res={_range_resolution:.4f} m/bin, FOV={FOV_DEG}°")

    # Bags 3D 07-07 : /sonar est en 32FC1 [0,1] (échos murs ≤ 0.47) alors que TOUTE la
    # chaîne aval (CFAR intensity_threshold 95, applyColorMap de feature_extraction:271)
    # attend le mono8 0-255 des anciens bags → conversion 32FC1 → mono8 ×255 ici.
    # (0.47*255 ≈ 120 > 95 : les murs repassent le seuil sans retoucher le yaml.)
    if msg.encoding == "32FC1":
        arr = np.frombuffer(msg.data, dtype=np.float32).copy()
        if arr.size and float(arr.max()) <= 1.0:
            arr *= 255.0
        msg.data = np.clip(arr, 0, 255).astype(np.uint8).tobytes()
        msg.encoding = "mono8"
        msg.step = msg.width

    # NB : OculusPingUncompressed n'a PAS de champ num_beams (déduit de len(bearings))
    ping = OculusPingUncompressed()
    ping.header          = msg.header
    ping.ping_id         = 0
    ping.bearings        = _bearings
    ping.range_resolution = _range_resolution
    ping.num_ranges      = num_ranges
    ping.ping            = msg
    _pub.publish(ping)

def main():
    global _pub, RANGE_M, FOV_DEG
    rospy.init_node("holoocean_sonar_bridge")
    RANGE_M = rospy.get_param("~range_m", RANGE_M)
    FOV_DEG = rospy.get_param("~fov_deg", FOV_DEG)
    _pub = rospy.Publisher(SONAR_TOPIC_UNCOMPRESSED, OculusPingUncompressed, queue_size=10)
    rospy.Subscriber("/sonar", Image, callback, queue_size=10)
    rospy.loginfo("[sonar_bridge] ready — /sonar → " + SONAR_TOPIC_UNCOMPRESSED)
    rospy.spin()

if __name__ == "__main__":
    main()
