#!/bin/bash
# Lance le binaire DISO directement (package.xml/CMakeLists manquants dans le src).
# roslaunch injecte des args ROS (__name:=.. __log:=..) → on les IGNORE et on garde le
# 1er arg "réel" comme chemin de config ; sinon défaut = config GT (OdomTopic=/pose_gt).
CFG="/home/nathanrasami/ros1_ws/src/sonar-SLAM/DISO/config/config_aracati2017.yaml"
for a in "$@"; do
  case "$a" in
    *:=*) ;;            # arg ROS (__name:=, __log:=, ...) → ignoré
    *) CFG="$a" ;;      # chemin de config réel (passé via launch args=)
  esac
done
exec /home/nathanrasami/ros1_ws/devel/.private/direct_sonar_odometry/lib/direct_sonar_odometry/aracati2017_node \
  "$CFG"
