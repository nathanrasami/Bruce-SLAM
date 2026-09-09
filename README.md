# Bruce-SLAM

Sonar-based graph SLAM for a BlueROV, extended during a 4th-year engineering internship
(Polytech Nantes, 2026).

This is a **fork of [jake3991/sonar-SLAM](https://github.com/jake3991/sonar-SLAM)** — the
original BlueROV sonar SLAM by **Jinkun Wang**, documented and maintained by
**John McConnell**. The SLAM core (feature extraction, scan matching, iSAM2 back-end) is
their work and is left untouched wherever possible. This fork adds new datasets, a
simulation pipeline, an acoustic-positioning variant and evaluation tooling.

<img src="bruce_slam/images/em.gif" width="1000"/>

## What this fork adds

| | |
|---|---|
| **Datasets** | Aracati2017 (no IMU, no DVL — a bridge is required), CIRS cave survey, HoloOcean simulation |
| **A method** | sonar + USBL acoustic positioning, fully ground-truth-free |
| **Simulation** | HoloOcean scenario generator producing ROS bags without needing ROS |
| **Evaluation** | ATE against ground truth, and a ground-truth-free metric based on harbour structures |

## Branches

| Branch | Content |
|---|---|
| **`main`** | Bruce-SLAM on its own `sample_data.bag`, plus the analysis tooling. The reference run. |
| **`Bruce`** | The original method adapted to the **Aracati2017** dataset (surface vessel, no IMU/DVL). |
| **`Bruce_Sonar_USBL`** | **The internship contribution**: sonar SLAM anchored by USBL acoustic fixes, 100 % ground-truth-free. |
| **`holoocean`** | **HoloOcean** simulation: scenario generation, bag writing, and SLAM on synthetic data. |

Each branch carries its own `run_slam.sh` presets and its own analysis chain, because the
sensor suites differ. Start from `main` to see the pipeline in its simplest form.

## Installation

Python dependencies (unchanged from upstream):

```
cv_bridge  gtsam  matplotlib  message_filters  numpy  opencv_python  rosbag  rospy
scikit_learn  scipy  sensor_msgs  Shapely  tf  tqdm  pyyaml
```

ROS dependencies: **ROS Noetic**, `catkin-pybind11`, `catkin-tools`.

Then, in your catkin workspace:

```bash
git clone https://github.com/ethz-asl/libnabo.git
git clone https://github.com/ethz-asl/libpointmatcher.git
git clone <this repo>
catkin build
```

`setup_ros_noetic.sh` automates this on a clean Noetic install.

The custom message packages (`bar30_depth`, `rti_dvl`, `sonar_oculus`, `kvh_gyro`) must be
built in the workspace as well — otherwise bags replay but no subscriber matches.

## Datasets

**Bags are not in this repository** (they are 1 GB each, above GitHub's file limit).

| Dataset | Where |
|---|---|
| `sample_data.bag` | [upstream repository](https://github.com/jake3991/sonar-SLAM) |
| Aracati2017 | LSA / FURG, Brazil |
| CIRS caves | [Underwater caves sonar dataset](https://cirs.udg.edu/caves-dataset/) |
| HoloOcean | generated locally, see below |

## `main` — Bruce-SLAM on `sample_data.bag`

The upstream bag contains exactly the four topics the original code expects, so **no
bridge is needed** — unlike Aracati2017.

| Topic | Rate | Type |
|---|---|---|
| `/vn100/imu/raw` | 198 Hz | `sensor_msgs/Imu` |
| `/rti/body_velocity/raw` | 5 Hz | `rti_dvl/DVL` |
| `/sonar_oculus_node/ping` | 5 Hz | `sonar_oculus/OculusPing` |
| `/bar30/depth/raw` | 4.2 Hz | `bar30_depth/Depth` |

Run and analyse:

```bash
./run_slam.sh sample_data          # offline mode: the node reads the bag itself
./analyse.sh run_sample_data_<date>
```

Outputs, in `results/run_sample_data_<date>/`:

- `trajectory.csv` — optimised SLAM poses, with the dead-reckoning pose at each keyframe
- `dead_reckoning.csv` — raw IMU + DVL + pressure dead reckoning at 5 Hz
- `pointcloud.csv`, `constraints.csv` — map and accepted loop closures
- `carte_finale.png`, `error_over_time.png`, `carte_port.png`, `trajectoires_3d.html`

### Results

879 s of survey, 73 keyframes, 6 919 map points, 17 loop closures over 163 m travelled.

<img src="docs/sample_data_marina.png" width="800"/>

**There is no ground truth in this bag**, so no ATE can be computed. Two substitutes are
used instead:

- **Distance to harbour structures.** Quays, pontoons and hulls are extracted from the
  aerial view by Sobel filtering (horizontal, vertical, both diagonals), turned into a
  distance field, and the cloud is scored against it — a reference independent of any
  SLAM. Best run: **1.18 m mean distance, 63 % of points within 1 m**.
- **SLAM vs dead reckoning.** The sonar corrects the dead reckoning by about 2 % of the
  distance travelled. A much larger gap is a symptom, not a feature.

<img src="docs/sample_data_slam_vs_dr.png" width="700"/>

Two configurations are worth knowing, both reachable through `campagne_sample_data.sh`:

| Configuration | Mean distance to structures | Runs | Note |
|---|---|---|---|
| stock | 1.44 m | 4 | upstream defaults |
| `icp_odom_sigmas` halved | **1.26 m** | 6 | best accuracy; best single run at 1.18 m |
| `min_pcm: 3` | 1.30 m | 3 | **bit-identical across runs** |

Figures are averaged over repeated runs of each configuration, since a single run is not
representative on its own.

The last one matters: with the default `min_pcm: 2`, marginal loop closures are accepted
or rejected depending on callback timing, and the trajectory moves by 1.5 to 2 m between
two runs of the same bag. Raising the threshold to 3 makes the pipeline deterministic.

## `Bruce` — Aracati2017

Aracati2017 is a surface vessel with **neither IMU nor DVL**, and it publishes Cartesian
sonar images rather than `OculusPing`. Two minimal bridges are added and the SLAM core is
left untouched: odometry integrated from `/cmd_vel`, and Cartesian-mode feature
extraction. `/pose_gt` is used **for evaluation only** — the estimate itself stays
ground-truth-free.

```bash
SSM=true NSSM=true USBL=false ./run_slam.sh    # Bruce, sequential + loop closures
./analyse.sh run_aracati_<date>
```

## `Bruce_Sonar_USBL` — the internship method

Sonar SLAM anchored by **USBL acoustic fixes**, added as robust unary position factors in
the graph. The vehicle heading is a free gauge that USBL alone only half constrains, which
is why the anchoring is done in the back-end rather than by filtering the odometry.
Everything is computed from onboard sensors, with no ground truth anywhere in the loop.

## `holoocean` — simulation

Generates ROS bags from the [HoloOcean](https://holoocean.readthedocs.io) simulator:
imaging sonar, DVL, IMU, pressure, along scripted trajectories in the PierHarbor scene.

```bash
python3 gen_bag_3d_v10.py      # writes BAG_files/holoocean_3d_traj9.bag
./run_slam.sh holoocean
```

> **HoloOcean must be installed to generate bags.** The scripts call `holoocean.make()`
> directly. Note that installing the Unreal client requires accepting the Epic Games EULA
> (an invitation is sent by email), and the client download is prone to truncating — using
> `wget -c` and then `holoocean.install(url="file://...")` works around it.
>
> Generation itself does **not** need ROS: bags are written with the pure-Python `rosbags`
> library. ROS is only needed to run the SLAM on them afterwards.

Sonar resolution is the parameter to watch: 1024 azimuth bins produce a striped image
(65 % of columns empty). Use 1024 range bins with 512 azimuth bins or fewer.

## Analysis tools

| Script | Purpose |
|---|---|
| `analysis/sample_data_report.py` | map, SLAM-vs-DR gap, overlay on the aerial view |
| `analysis/paper_eval.py` | ATE (Umeyama and origin-anchored), map sharpness |
| `analysis/bilan_run.py` | one-image summary of a run |
| `analysis/view3d_sample_data.py` | interactive 3D HTML of both trajectories |

`analyse.sh` picks the right chain from the run name.

## Credits

Original work by **Jinkun Wang**; documentation and maintenance by **John McConnell** —
[jake3991/sonar-SLAM](https://github.com/jake3991/sonar-SLAM). Vehicle hardware is
documented in [Argonaut](https://github.com/jake3991/Argonaut).

Fork and internship work: **Nathan Rasamijaona**, Polytech Nantes, 2026.

Licensed under the terms of the upstream `LICENSE`.
