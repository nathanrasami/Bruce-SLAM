#!/bin/bash
# Installation ROS Noetic + workspace bruce/DISO dans le conteneur distrobox
# "ros1" (Ubuntu 20.04) sur Fedora. Idempotent : relançable, reprend où il
# s'est arrêté (marqueurs d'étapes dans ~/.ros1_setup_state/).
#
# Usage (depuis Fedora) :
#   distrobox enter ros1 -- bash ~/Documents/Polytech/Stage4A/SLAM/sonar-SLAM/setup_ros_noetic.sh
#
# Recettes sources :
#   - README.md bruce : libnabo + libpointmatcher@d478ef2 dans le workspace, gtsam pip
#   - DISO/docker/diso.DockerFile : g2o@21b7ce45 (+ csparse), fmt
set -e

REPO="$HOME/Documents/Polytech/Stage4A/SLAM/sonar-SLAM"
WS="$HOME/ros1_ws"
STATE="$HOME/.ros1_setup_state"
mkdir -p "$STATE"

stage() {  # exécute une étape une seule fois
    local name="$1"; shift
    if [ -f "$STATE/$name" ]; then echo "=== [$name] déjà fait, skip ==="; return 0; fi
    echo "=== [$name] démarrage $(date +%H:%M:%S) ==="
    "$@"
    touch "$STATE/$name"
    echo "=== [$name] OK $(date +%H:%M:%S) ==="
}

# ---------------------------------------------------------------- étape 1 : ROS
install_ros() {
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y curl gnupg2 lsb-release
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ros-noetic-desktop-full \
        ros-noetic-ros-numpy \
        python3-catkin-tools python3-osrf-pycommon \
        python3-pip python-is-python3 git build-essential cmake \
        libsuitesparse-dev libfmt-dev libyaml-cpp-dev libboost-all-dev
}

# ------------------------------------------------------- étape 2 : python deps
install_python() {
    pip3 install --user gtsam==4.1.1 scikit-learn scipy matplotlib pandas \
        "numpy==1.23.5" shapely tqdm
}

# -------------------------------------------------------- étape 3 : workspace
setup_workspace() {
    mkdir -p "$WS/src"
    [ -e "$WS/src/sonar-SLAM" ] || ln -s "$REPO" "$WS/src/sonar-SLAM"
    cd "$WS/src"
    [ -d libnabo ] || git clone https://github.com/ethz-asl/libnabo.git
    if [ ! -d libpointmatcher ]; then
        git clone https://github.com/ethz-asl/libpointmatcher.git
        git -C libpointmatcher checkout d478ef2eb33894d5f1fe84d8c62cec2fc6da818f
    fi
    # sonar_oculus : le repo GitHub n'existe plus ; on crée un stub minimal
    # (seuls les types de messages OculusPing / OculusPingUncompressed sont utilisés)
    if [ ! -d sonar_oculus ]; then
        mkdir -p sonar_oculus/msg
        cat > sonar_oculus/package.xml <<'PKGXML'
<?xml version="1.0"?>
<package format="2">
  <name>sonar_oculus</name>
  <version>0.0.1</version>
  <description>Stub package – message definitions only</description>
  <maintainer email="stub@stub.com">stub</maintainer>
  <license>MIT</license>
  <buildtool_depend>catkin</buildtool_depend>
  <build_depend>std_msgs</build_depend>
  <build_depend>sensor_msgs</build_depend>
  <build_depend>message_generation</build_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>message_runtime</exec_depend>
</package>
PKGXML
        cat > sonar_oculus/CMakeLists.txt <<'CMAKE'
cmake_minimum_required(VERSION 3.0.2)
project(sonar_oculus)
find_package(catkin REQUIRED COMPONENTS std_msgs sensor_msgs message_generation)
add_message_files(FILES OculusPing.msg OculusPingUncompressed.msg)
generate_messages(DEPENDENCIES std_msgs sensor_msgs)
catkin_package(CATKIN_DEPENDS std_msgs sensor_msgs message_runtime)
CMAKE
        cat > sonar_oculus/msg/OculusPing.msg <<'MSG'
std_msgs/Header header
uint32 ping_id
uint32 num_ranges
float32 range_resolution
float32[] bearings
sensor_msgs/Image ping
MSG
        cat > sonar_oculus/msg/OculusPingUncompressed.msg <<'MSG'
std_msgs/Header header
uint32 ping_id
uint32 num_ranges
float32 range_resolution
float32[] bearings
sensor_msgs/Image ping
MSG
    fi
}

# --------------------- étape 3b : stubs messages capteurs (repos disparus)
# rti_dvl / bar30_depth / kvh_gyro : repos GitHub de jake3991 plus disponibles.
# Seuls les types de messages sont importés par bruce_slam (champs vérifiés
# dans dead_reckoning.py, kalman.py, gyro.py) — Aracati n'utilise pas ces
# capteurs mais les imports Python doivent résoudre.
make_msg_stub() {  # $1=pkg  $2=MsgName  $3=définition du message
    local pkg="$1" msgname="$2" msgdef="$3"
    [ -d "$WS/src/$pkg" ] && return 0
    mkdir -p "$WS/src/$pkg/msg"
    printf '%s\n' "$msgdef" > "$WS/src/$pkg/msg/$msgname.msg"
    cat > "$WS/src/$pkg/package.xml" <<PKGXML
<?xml version="1.0"?>
<package format="2">
  <name>$pkg</name>
  <version>0.0.1</version>
  <description>Stub – message definitions only</description>
  <maintainer email="stub@stub.com">stub</maintainer>
  <license>MIT</license>
  <buildtool_depend>catkin</buildtool_depend>
  <build_depend>std_msgs</build_depend>
  <build_depend>geometry_msgs</build_depend>
  <build_depend>message_generation</build_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>message_runtime</exec_depend>
</package>
PKGXML
    cat > "$WS/src/$pkg/CMakeLists.txt" <<CMAKE
cmake_minimum_required(VERSION 3.0.2)
project($pkg)
find_package(catkin REQUIRED COMPONENTS std_msgs geometry_msgs message_generation)
add_message_files(FILES $msgname.msg)
generate_messages(DEPENDENCIES std_msgs geometry_msgs)
catkin_package(CATKIN_DEPENDS std_msgs geometry_msgs message_runtime)
CMAKE
}

setup_sensor_stubs() {
    make_msg_stub rti_dvl DVL 'std_msgs/Header header
geometry_msgs/Vector3 velocity
float64 altitude'
    make_msg_stub bar30_depth Depth 'std_msgs/Header header
float64 depth
float64 temperature'
    make_msg_stub kvh_gyro gyro 'std_msgs/Header header
float64[] delta'
}

# ------------------------------------------------- étape 4 : g2o (pour DISO)
install_g2o() {
    mkdir -p "$HOME/third_party"
    cd "$HOME/third_party"
    [ -d g2o ] || git clone https://github.com/RainerKuemmerle/g2o.git
    cd g2o
    # Le commit du Dockerfile DISO (21b7ce45, avril 2016) expose l'ancienne API
    # g2o à pointeurs bruts, incompatible avec le code DISO qui utilise
    # l'API unique_ptr (post-2017). Le Dockerfile officiel est incohérent.
    # → release 20201223 : API unique_ptr, contemporaine d'Ubuntu 20.04.
    git checkout 20201223_git
    rm -rf build && mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j8
    # purge l'ancienne installation (API 2016) avant de réinstaller
    sudo rm -rf /usr/local/include/g2o /usr/local/lib/libg2o*
    sudo make install
    sudo ldconfig
    # DISO/CMakeLists attend les headers dans Thirdparty/g2o/include
    mkdir -p "$REPO/DISO/Thirdparty/g2o"
    [ -e "$REPO/DISO/Thirdparty/g2o/include" ] || \
        ln -s /usr/local/include "$REPO/DISO/Thirdparty/g2o/include"
}

# ---------------------------------------------- étape 4b : sophus (pour DISO)
install_sophus() {
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ros-noetic-pybind11-catkin
    mkdir -p "$HOME/third_party"
    cd "$HOME/third_party"
    [ -d Sophus ] || git clone https://github.com/strasdat/Sophus.git
    cd Sophus
    git checkout 49a7e1286910019f74fb4f0bb3e213c909f8e1b7  # commit du Dockerfile DISO
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j8
    sudo make install
    sudo ldconfig
}

# ----------------------------------------------------------- étape 5 : build
build_workspace() {
    export SHELL=/bin/bash   # catkin_tools exige un chemin absolu (hérite parfois "bash")
    source /opt/ros/noetic/setup.bash
    cd "$WS"
    catkin init >/dev/null
    catkin build -j8 --no-status
}

# --------------------------------------------------------- étape 6 : bashrc
setup_bashrc() {
    grep -q "ros1_ws/devel/setup.bash" "$HOME/.bashrc" 2>/dev/null || cat >> "$HOME/.bashrc" <<'EOF'

# ROS Noetic (uniquement dans le conteneur distrobox ros1)
if [ -n "$CONTAINER_ID" ] && [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
    [ -f "$HOME/ros1_ws/devel/setup.bash" ] && source "$HOME/ros1_ws/devel/setup.bash"
    export ROS_HOSTNAME=localhost
    export ROS_MASTER_URI=http://localhost:11311
fi
EOF
}

stage ros        install_ros
stage python     install_python
stage workspace  setup_workspace
stage stubs      setup_sensor_stubs
stage g2o        install_g2o
stage sophus     install_sophus
stage build      build_workspace
stage bashrc     setup_bashrc

echo ""
echo "================== INSTALLATION TERMINÉE =================="
echo "Lancer le SLAM :"
echo "  distrobox enter ros1"
echo "  roslaunch bruce_slam aracati.launch bag_file:=$REPO/ARACATI_2017_8bits_full.bag"
