# Bruce-SLAM

Sonar-based graph SLAM for a BlueROV, extended during a 4th-year engineering internship
(Polytech Nantes, 2026).

This is a **fork of [jake3991/sonar-SLAM](https://github.com/jake3991/sonar-SLAM)** — the
original BlueROV sonar SLAM by **Jinkun Wang**, documented and maintained by
**John McConnell**. The SLAM core (feature extraction, scan matching, iSAM2 back-end) is
their work and is left untouched wherever possible. This fork adds a new dataset, four
comparable methods, a simulation pipeline and evaluation tooling.

<img src="bruce_slam/images/em.gif" width="1000"/>

## What this fork adds

| Addition | Detail |
|---|---|
| Dataset | **Aracati2017**, a surface vessel with neither IMU nor DVL — bridges are required |
| Methods | four variants behind two switches, sonar context and USBL anchoring |
| Simulation | **HoloOcean** scenario generator producing ROS bags without needing ROS |
| Evaluation | start-pinned ATE, and a ground-truth-free metric based on harbour structures |

## Which branch to use

| Branch | Use it to |
|---|---|
| **`refonte`** | **compare the four methods** on Aracati2017. One launcher, one preset per method, and a gate that checks the comparison is valid. Start here for research work. |
| **`main`** | run the upstream pipeline on its own `sample_data.bag`. Shortest path to a working install. |
| `Bruce` | replay the adapted original method on Aracati2017, config frozen on its best run |
| `Bruce_Sonar_USBL` | replay the sonar + USBL method, config frozen on its best run |
| `holoocean` | generate synthetic bags and run SLAM on them |

`refonte` is the branch to read if the question is *which method is better*. The three
frozen branches answer a different question: *reproduce this exact result*. They each
carry one config and one champion, so a bare `./run_slam.sh` gives the published figure.

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

Bags are not versioned here. `sample_data.bag` comes from the
[upstream repository](https://github.com/jake3991/sonar-SLAM), Aracati2017 from
[matheusbg8/aracati2017](https://github.com/matheusbg8/aracati2017), and HoloOcean bags are
generated locally.

## Evaluation conventions

Read this before comparing any figure in this repository with a figure from elsewhere.

- **Trajectories are pinned at the start by translation only.** No rotation, no
  reflection, no scale is ever fitted. The orientation of the estimated frame is an output
  of the system, not something the evaluation may correct. Numbers obtained this way are
  strictly larger than numbers from a rigid best-fit alignment, and the two are not
  interchangeable.
- **Ground truth is never in the estimation loop.** On Aracati2017, `/pose_gt` and the DGPS
  topics are read only to score a finished run.
- **A single run is not a result.** The back-end is sensitive to callback timing: on
  `sample_data.bag`, two runs of the same configuration can differ by 1.5 to 2 m. Repeat a
  configuration before drawing a conclusion.

---

# Part 1 — Getting started with `sample_data.bag`

Branch `main`. The upstream bag contains exactly the four topics the original code expects,
so **no bridge is needed**. It is the shortest path to a working run, and the right place
to start before moving to Aracati2017.

| Topic | Rate | Type |
|---|---|---|
| `/vn100/imu/raw` | 198 Hz | `sensor_msgs/Imu` |
| `/rti/body_velocity/raw` | 5 Hz | `rti_dvl/DVL` |
| `/sonar_oculus_node/ping` | 5 Hz | `sonar_oculus/OculusPing` |
| `/bar30/depth/raw` | 4.2 Hz | `bar30_depth/Depth` |

879 s of survey in a marina. Run and analyse:

```bash
./run_slam.sh sample_data          # offline mode: the node reads the bag itself
./analyse.sh run_sample_data_<date>
```

Outputs, in `results/run_sample_data_<date>/`:

- `trajectory.csv` — optimised SLAM poses, with the dead-reckoning pose at each keyframe
- `dead_reckoning.csv` — raw IMU + DVL + pressure dead reckoning at 5 Hz
- `pointcloud.csv`, `constraints.csv` — map and accepted loop closures
- `carte_finale.png`, `error_over_time.png`, `carte_port.png`, `trajectoires_3d.html`

## Results

73 keyframes, 6 919 map points, 17 loop closures over 163 m travelled.

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

Three configurations, each averaged over repeated runs, through `campagne_sample_data.sh`:

| Configuration | Mean distance to structures | Runs | Note |
|---|---|---|---|
| stock | 1.44 m | 4 | upstream defaults |
| `icp_odom_sigmas` halved | **1.26 m** | 6 | best accuracy; best single run at 1.18 m |
| `min_pcm: 3` | 1.30 m | 3 | **bit-identical across runs** |

The last row matters. With the default `min_pcm: 2`, marginal loop closures are accepted
or rejected depending on callback timing. Raising the threshold to 3 makes the pipeline
deterministic, for 0.04 m of accuracy. Prefer it whenever a run has to be reproduced.

---

# Part 2 — Aracati2017

This is the main subject of the internship. Aracati2017 is a **surface vessel** surveying a
harbour, and it differs from the BlueROV on every count: **no IMU, no DVL**, and Cartesian
sonar images instead of `OculusPing`.

## The dataset

43 min of survey, 58 168 messages, 9 GB. Measured directly from the bag:

| Topic | Rate | Type | Use |
|---|---|---|---|
| `/son/compressed` | 5.7 Hz | `sensor_msgs/CompressedImage` | imaging sonar, Cartesian |
| `/cmd_vel` | 5.7 Hz | `geometry_msgs/TwistStamped` | thruster commands, integrated into odometry |
| `/usbl_point` | 0.6 Hz | `geometry_msgs/PointStamped` | acoustic positioning fixes |
| `/usbl` | 0.6 Hz | `sensor_msgs/NavSatFix` | same, geographic coordinates |
| `/pose_gt` | 5.7 Hz | `geometry_msgs/PoseStamped` | ground truth, **evaluation only** |
| `/dgps_point` | 1.0 Hz | `geometry_msgs/PointStamped` | DGPS, source of the ground truth |
| `/dgps` | 1.0 Hz | `sensor_msgs/NavSatFix` | same, geographic coordinates |
| `/surface/compressed` | 3.0 Hz | `sensor_msgs/CompressedImage` | surface camera, unused |

## What had to be adapted

The SLAM core is untouched. Two minimal bridges make the dataset usable:

- **Odometry from `/cmd_vel`.** With no IMU and no DVL there is no dead reckoning at all,
  so thruster commands are integrated into a `nav_msgs/Odometry` stream. This drifts badly,
  which is precisely what the sonar has to correct.
- **Cartesian feature extraction.** The sonar publishes Cartesian images, not polar pings,
  so CFAR runs in Cartesian mode.

## The four methods

Branch `refonte`. Two independent switches, one preset each, nothing else adjustable from
the launcher:

| Preset | Sonar context | USBL back-end |
|---|---|---|
| `./run_slam.sh bruce` | off | off |
| `./run_slam.sh bruce_u` | off | on, sigma 2.5 |
| `./run_slam.sh bruce_sonar` | on | off |
| `./run_slam.sh bsu` | on | on, sigma 1.4 |

Two design rules make the comparison meaningful, and both are enforced rather than
documented:

- **The odometry is identical in all four presets.** `cmd_vel_odom` never subscribes to
  USBL — no seed, no correction, no gain. USBL enters only as graph factors, in the
  presets that use it.
- **`analysis/gates_refonte.py` validates the run set before any figure is produced.**
  Four gates, and it returns 0 for pass, 2 for borderline, 1 for fail:
  1. the four `odometry.csv` are identical, checked rigidly (rotation near zero, residual
     near zero) — proof that no method touched the odometry;
  2. the ATE is start-pinned by translation only, globally and over three time sections;
  3. the ordering `bsu <= bruce_u < bruce` holds, with a 0.10 m tolerance for the known
     run-to-run ICP variance;
  4. map quality against the reference cloud, informative only.

Only two config files are involved: `slam_aracati_native.yaml` when sonar context is off,
`slam_aracati.yaml` when it is on. The launcher picks one and overrides the two switches.

**One result to know before running `bruce_sonar`.** Sonar context without an absolute
anchor does not converge on this dataset: 0 of 154 loop candidates are accepted, because
the drift at revisit time (median 9.5 m) exceeds the geometric gate. That preset therefore
behaves like the odometry alone. It is not a bug, and it is excluded from the ordering
gate — it is the finding that sonar context and USBL anchoring are complementary rather
than alternatives.

## Results

<img src="docs/aracati_result.png" width="1000"/>

Pinned at the start by translation only. Over the 43 min survey the onboard odometry drifts
to **20.9 m** of error, while the sonar + USBL method stays at **1.94 m**. The adapted
original method reaches 2.6 to 3.6 m depending on the run.

On the right, the map built from the optimised poses. The T-shaped pier and the two quays
come out straight and thin: that is the practical test that the trajectory is right, since
a wrong trajectory smears one structure into several parallel ghosts.

## Why USBL is anchored in the back-end

USBL fixes are added as **robust unary position factors**, constraining x and y while
leaving heading free. Two points that cost time and are worth stating:

- Heading is a **free gauge**. Acoustic positioning alone only half constrains it, so it
  has to come from the scan matching.
- Anchoring must happen **either** in the front-end **or** in the back-end, never both.
  Filtering the odometry with USBL while also adding USBL factors to the graph
  over-constrains it and makes the trajectory zigzag.

---

# Part 3 — HoloOcean simulation

Branch `holoocean`. Generates ROS bags from the
[HoloOcean](https://holoocean.readthedocs.io) simulator: imaging sonar, DVL, IMU and
pressure, along scripted trajectories in the PierHarbor scene.

```bash
python3 gen_bag_3d_v10.py      # writes BAG_files/holoocean_3d_traj9.bag
./run_slam.sh holoocean
```

> **HoloOcean must be installed to generate bags.** The scripts call `holoocean.make()`
> directly. Installing the Unreal client requires accepting the Epic Games EULA (an
> invitation is sent by email), and the client download is prone to truncating — using
> `wget -c` and then `holoocean.install(url="file://...")` works around it.
>
> Generation itself does **not** need ROS: bags are written with the pure-Python `rosbags`
> library. ROS is only needed to run the SLAM on them afterwards.

Sonar resolution is the parameter to watch: 1024 azimuth bins produce a striped image
(65 % of columns empty). Use 1024 range bins with 512 azimuth bins or fewer.

# Analysis tools

| Script | Purpose |
|---|---|
| `analysis/gates_refonte.py` | validates a set of runs before comparing methods |
| `analysis/analyze_origine.py` | ATE pinned at the start, translation only |
| `analysis/sample_data_report.py` | map, SLAM-vs-DR gap, overlay on the aerial view |
| `analysis/bilan_run.py` | one-image summary of a run |
| `analysis/view3d_sample_data.py` | interactive 3D HTML of both trajectories |

`analyse.sh` picks the right chain from the run name.

# Credits

Original work by **Jinkun Wang**; documentation and maintenance by **John McConnell** —
[jake3991/sonar-SLAM](https://github.com/jake3991/sonar-SLAM). Vehicle hardware is
documented in [Argonaut](https://github.com/jake3991/Argonaut).

The **Aracati2017** dataset is released at
[matheusbg8/aracati2017](https://github.com/matheusbg8/aracati2017) — see that repository
for the authors, the recording conditions and the terms of use.

Fork and internship work: **Nathan Rasamijaona**, Polytech Nantes, 2026.

Licensed under the terms of the upstream `LICENSE`.
