# python imports
import os
import csv
import threading
import tf
import rospy
import gtsam
import cv_bridge
from nav_msgs.msg import Odometry
from message_filters import  Subscriber
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped
from message_filters import ApproximateTimeSynchronizer

# bruce imports
from bruce_slam.utils.io import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.visualization import *
from bruce_slam.slam import SLAM, Keyframe
from bruce_slam import pcl

# Argonaut imports
from sonar_oculus.msg import OculusPing


class SLAMNode(SLAM):
    """This class takes the functionality from slam.py and implments it in the ros
    environment. 
    """
    
    def __init__(self):
        super(SLAMNode, self).__init__()

        # the threading lock
        self.lock = threading.RLock()

    def init_node(self, ns="~")->None:
        """Configures the SLAM node

        Args:
            ns (str, optional): The namespace of the node. Defaults to "~".
        """

        #keyframe paramters, how often to add them
        self.keyframe_duration = rospy.get_param(ns + "keyframe_duration")
        self.keyframe_duration = rospy.Duration(self.keyframe_duration)
        self.keyframe_translation = rospy.get_param(ns + "keyframe_translation")
        self.keyframe_rotation = rospy.get_param(ns + "keyframe_rotation")

        #SLAM paramter, are we using SLAM or just dead reckoning
        self.enable_slam = rospy.get_param(ns + "enable_slam")
        print("SLAM STATUS: ", self.enable_slam)

        #noise models
        self.prior_sigmas = rospy.get_param(ns + "prior_sigmas")
        self.odom_sigmas = rospy.get_param(ns + "odom_sigmas")
        self.icp_odom_sigmas = rospy.get_param(ns + "icp_odom_sigmas")

        #resultion for map downsampling
        self.point_resolution = rospy.get_param(ns + "point_resolution")

        #sequential scan matching parameters (SSM)
        self.ssm_params.enable = rospy.get_param(ns + "ssm/enable")
        self.ssm_params.min_points = rospy.get_param(ns + "ssm/min_points")
        self.ssm_params.max_translation = rospy.get_param(ns + "ssm/max_translation")
        self.ssm_params.max_rotation = rospy.get_param(ns + "ssm/max_rotation")
        self.ssm_params.target_frames = rospy.get_param(ns + "ssm/target_frames")
        print("SSM: ", self.ssm_params.enable)

        #non sequential scan matching parameters (NSSM) aka loop closures
        self.nssm_params.enable = rospy.get_param(ns + "nssm/enable")
        self.nssm_params.min_st_sep = rospy.get_param(ns + "nssm/min_st_sep")
        self.nssm_params.min_points = rospy.get_param(ns + "nssm/min_points")
        self.nssm_params.max_translation = rospy.get_param(ns + "nssm/max_translation")
        self.nssm_params.max_rotation = rospy.get_param(ns + "nssm/max_rotation")
        self.nssm_params.source_frames = rospy.get_param(ns + "nssm/source_frames")
        self.nssm_params.cov_samples = rospy.get_param(ns + "nssm/cov_samples")
        print("NSSM: ", self.nssm_params.enable)

        #pairwise consistency maximization parameters for loop closure 
        #outliar rejection
        self.pcm_queue_size = rospy.get_param(ns + "pcm_queue_size")
        self.min_pcm = rospy.get_param(ns + "min_pcm")

        #mak delay between an incoming point cloud and dead reckoning
        self.feature_odom_sync_max_delay = 0.5

        #define the subsrcibing topics
        self.feature_sub = Subscriber(SONAR_FEATURE_TOPIC, PointCloud2)
        self.odom_sub = Subscriber(LOCALIZATION_ODOM_TOPIC, Odometry)

        #define the sync policy
        self.time_sync = ApproximateTimeSynchronizer(
            [self.feature_sub, self.odom_sub], 20, 
            self.feature_odom_sync_max_delay, allow_headerless = False)

        #register the callback in the sync policy
        self.time_sync.registerCallback(self.SLAM_callback)

        #pose publisher
        self.pose_pub = rospy.Publisher(
            SLAM_POSE_TOPIC, PoseWithCovarianceStamped, queue_size=10)

        #dead reckoning topic
        self.odom_pub = rospy.Publisher(SLAM_ODOM_TOPIC, Odometry, queue_size=10)

        #SLAM trajectory topic
        self.traj_pub = rospy.Publisher(
            SLAM_TRAJ_TOPIC, PointCloud2, queue_size=1, latch=True)

        #constraints between poses
        self.constraint_pub = rospy.Publisher(
            SLAM_CONSTRAINT_TOPIC, Marker, queue_size=1, latch=True)

        #point cloud publisher topic
        self.cloud_pub = rospy.Publisher(
            SLAM_CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True)

        #tf broadcaster to show pose
        self.tf = tf.TransformBroadcaster()

        #cv bridge object
        self.CVbridge = cv_bridge.CvBridge()

        #get the ICP configuration from the yaml fukle
        icp_config = rospy.get_param(ns + "icp_config")
        self.icp.loadFromYaml(icp_config)
        
        # define the robot ID this is not used here, extended in multi-robot SLAM
        self.rov_id = ""

        # Aracati : buffer ground truth (/pose_gt = position DGPS) + export CSV à l'arrêt,
        # pour l'évaluation ATE. Seul ajout au cœur Bruce, sans effet sur le SLAM.
        self.gt_poses = []
        rospy.Subscriber("/pose_gt", PoseStamped, self._gt_callback, queue_size=100)
        # odométrie brute à pleine fréquence (cmd_vel intégré) → odometry.csv,
        # même base de temps que le GT (comme sur Bruce_Sonar_USBL)
        self.odom_poses = []
        rospy.Subscriber(LOCALIZATION_ODOM_TOPIC, Odometry, self._odom_log_callback,
                         queue_size=50)
        rospy.on_shutdown(self.export_csv)

        # USBL back-end (GT-free) : facteur de position absolue sur /usbl_point.
        # Le repère est déjà aligné par cmd_vel_odom (seed USBL) → ancre directe.
        self.usbl_enable = rospy.get_param(ns + "usbl/enable", False)
        self.usbl_sigma = rospy.get_param(ns + "usbl/sigma", 1.4)
        self.usbl_max_dt = rospy.get_param(ns + "usbl/max_dt", 1.0)
        self.usbl_buffer = []
        if self.usbl_enable:
            rospy.Subscriber("/usbl_point", PointStamped, self._usbl_callback, queue_size=20)
            loginfo("USBL back-end activé (sigma=%.1f m)" % self.usbl_sigma)

        #call the configure function
        self.configure()
        loginfo("SLAM node is initialized")

    def _gt_callback(self, msg: PoseStamped) -> None:
        """Bufferise la position ground truth (DGPS) pour l'export CSV.
        theta = yaw du quaternion /pose_gt (cap COMPAS, convention NED :
        gtheta ~ -theta_map + 90.7 deg — cf. bilan_run.py qui fait le fit s,beta)."""
        q = msg.pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.gt_poses.append((msg.header.stamp.to_sec(),
                              msg.pose.position.x, msg.pose.position.y, yaw))

    def _odom_log_callback(self, msg: Odometry) -> None:
        """Bufferise l'odométrie brute à pleine fréquence (pour odometry.csv)."""
        self.odom_poses.append((msg.header.stamp.to_sec(),
                                msg.pose.pose.position.x,
                                msg.pose.pose.position.y))

    def _usbl_callback(self, msg: PointStamped) -> None:
        """Bufferise les fixes USBL (positionnement acoustique, GT-free)."""
        self.usbl_buffer.append((msg.header.stamp.to_sec(), msg.point.x, msg.point.y))
        if len(self.usbl_buffer) > 2000:
            self.usbl_buffer = self.usbl_buffer[-2000:]

    def add_usbl(self, keyframe: Keyframe) -> None:
        """Facteur de POSITION ABSOLUE USBL (GT-free) sur le keyframe courant : prior unaire
        ROBUSTE (Cauchy) qui ne contraint QUE x,y (sigma θ énorme → cap libre, géré par
        l'odométrie). gtsam recolle la trajectoire sur les ancres USBL en moyennant leur
        bruit (~1.4 m) sur tout le graphe (lisseur) et en rejetant les outliers acoustiques.
        Repère déjà aligné (cmd_vel_odom seede depuis l'USBL) → ux,uy direct, sans Umeyama.
        Ajouté dans SLAMNode (hérite de SLAM) → slam.py reste pristine."""
        if not self.usbl_enable or not self.usbl_buffer:
            return
        t = keyframe.time.to_sec()
        best, bdt = None, self.usbl_max_dt
        for ut, ux, uy in self.usbl_buffer:
            if abs(ut - t) < bdt:
                bdt, best = abs(ut - t), (ux, uy)
        if best is None:
            return
        ux, uy = best
        cov = np.diag([self.usbl_sigma ** 2, self.usbl_sigma ** 2, 1e12])
        model = self.create_robust_full_noise_model(cov)
        self.graph.add(gtsam.PriorFactorPose2(X(self.current_key),
                                              gtsam.Pose2(ux, uy, 0.0), model))

    def export_csv(self) -> None:
        """Exporte trajectoire / nuage / ground truth en CSV à l'arrêt (évaluation ATE).
        SLAM_RESULTS_DIR définit le dossier de sortie (défaut : ../results)."""
        if not self.keyframes:
            return
        output_dir = os.environ.get("SLAM_RESULTS_DIR",
            os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", "..", "results")))
        os.makedirs(output_dir, exist_ok=True)

        # trajectoire SLAM (+ dead reckoning brut)
        with open(os.path.join(output_dir, "trajectory.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["keyframe_id", "time", "x", "y", "theta",
                        "dr_x", "dr_y", "dr_theta", "nssm_constraints"])
            for i, kf in enumerate(self.keyframes):
                w.writerow([i, kf.time.to_sec(), kf.pose.x(), kf.pose.y(), kf.pose.theta(),
                            kf.dr_pose.x(), kf.dr_pose.y(), kf.dr_pose.theta(),
                            len(kf.constraints)])

        # nuage de points (transf_points monde de tous les keyframes)
        with open(os.path.join(output_dir, "pointcloud.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["keyframe_id", "x", "y"])
            for i, kf in enumerate(self.keyframes):
                if kf.transf_points is None:
                    continue
                for p in kf.transf_points:
                    w.writerow([i, p[0], p[1]])

        # ground truth (DGPS)
        if self.gt_poses:
            with open(os.path.join(output_dir, "groundtruth.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "x", "y", "theta"])
                w.writerows(self.gt_poses)

        # odométrie brute pleine fréquence (même format que Bruce_Sonar_USBL)
        if self.odom_poses:
            with open(os.path.join(output_dir, "odometry.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "x", "y"])
                w.writerows(self.odom_poses)
        loginfo("CSV exportés dans %s" % output_dir)

    @add_lock
    def sonar_callback(self, ping:OculusPing)->None:
        """Subscribe once to configure Oculus property.
        Assume sonar configuration doesn't change much.

        Args:
            ping (OculusPing): The sonar message. 
        """
        
        self.oculus.configure(ping)
        self.sonar_sub.unregister()

    @add_lock
    def SLAM_callback(self, feature_msg:PointCloud2, odom_msg:Odometry)->None:
        """SLAM call back. Subscibes to the feature msg point cloud and odom msg
            Handles the whole SLAM system and publishes map, poses and constraints

        Args:
            feature_msg (PointCloud2): the incoming sonar point cloud
            odom_msg (Odometry): the incoming DVL/IMU state estimate
        """

        #aquire the lock 
        self.lock.acquire()

        #get rostime from the point cloud
        time = feature_msg.header.stamp

        #get the dead reckoning pose from the odom msg, GTSAM pose object
        dr_pose3 = r2g(odom_msg.pose.pose)

        #init a new key frame
        frame = Keyframe(False, time, dr_pose3)

        #convert the point cloud message to a numpy array of 2D
        points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(feature_msg)
        points = np.c_[points[:,0] , -1 *  points[:,2]]

        # In case feature extraction is skipped in this frame
        if len(points) and np.isnan(points[0, 0]):
            frame.status = False
        else:
            frame.status = self.is_keyframe(frame)

        #set the frames twist
        frame.twist = odom_msg.twist.twist

        #update the keyframe with pose information from dead reckoning
        if self.keyframes:
            dr_odom = self.current_keyframe.dr_pose.between(frame.dr_pose)
            pose = self.current_keyframe.pose.compose(dr_odom)
            frame.update(pose)


        #check frame staus, are we actually adding a keyframe? This is determined based on distance 
        #traveled according to dead reckoning
        if frame.status:

            #add the point cloud to the frame
            frame.points = points

            #perform seqential scan matching
            #if this is the first frame do not
            if not self.keyframes:
                self.add_prior(frame)
            else:
                self.add_sequential_scan_matching(frame)

            # USBL : ancre de position absolue (GT-free) sur ce keyframe (si activé)
            self.add_usbl(frame)

            #update the factor graph with the new frame
            self.update_factor_graph(frame)

            #if loop closures are enabled
            #nonsequential scan matching is True (a loop closure occured) update graph again
            if self.nssm_params.enable  and self.add_nonsequential_scan_matching():
                self.update_factor_graph()
            
        #update current time step and publish the topics
        self.current_frame = frame
        self.publish_all()
        self.lock.release()

    def publish_all(self)->None:
        """Publish to all ouput topics
            trajectory, contraints, point cloud and the full GTSAM instance
        """
        if not self.keyframes:
            return

        self.publish_pose()
        if self.current_frame.status:
            self.publish_trajectory()
            self.publish_constraint()
            self.publish_point_cloud()

    def publish_pose(self)->None:
        """Append dead reckoning from Localization to SLAM estimate to achieve realtime TF.
        """

        #define a pose with covariance message 
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.current_frame.time
        if self.rov_id == "":
            pose_msg.header.frame_id = "map"
        else:
            pose_msg.header.frame_id = self.rov_id + "_map"
        pose_msg.pose.pose = g2r(self.current_frame.pose3)

        cov = 1e-4 * np.identity(6, np.float32)
        # FIXME Use cov in current_frame
        cov[np.ix_((0, 1, 5), (0, 1, 5))] = self.current_keyframe.transf_cov
        pose_msg.pose.covariance = cov.ravel().tolist()
        self.pose_pub.publish(pose_msg)

        o2m = self.current_frame.pose3.compose(self.current_frame.dr_pose3.inverse())
        o2m = g2r(o2m)
        p = o2m.position
        q = o2m.orientation
        self.tf.sendTransform(
            (p.x, p.y, p.z),
            [q.x, q.y, q.z, q.w],
            self.current_frame.time,
            "odom",
            "map",
        )

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.pose.pose = pose_msg.pose.pose
        if self.rov_id == "":
            odom_msg.child_frame_id = "base_link"
        else:
            odom_msg.child_frame_id = self.rov_id + "_base_link"
        odom_msg.twist.twist = self.current_frame.twist
        self.odom_pub.publish(odom_msg)

    def publish_constraint(self)->None:
        """Publish constraints between poses in the factor graph,
        either sequential or non-sequential.
        """

        #define a list of all the constraints
        links = []

        #iterate over all the keframes
        for x, kf in enumerate(self.keyframes[1:], 1):

            #append each SSM factor in green
            p1 = self.keyframes[x - 1].pose3.x(), self.keyframes[x - 1].pose3.y(), self.keyframes[x - 1].dr_pose3.z()
            p2 = self.keyframes[x].pose3.x(), self.keyframes[x].pose3.y(), self.keyframes[x].dr_pose3.z()
            links.append((p1, p2, "green"))

            #loop over all loop closures in this keyframe and append them in red
            for k, _ in self.keyframes[x].constraints:
                p0 = self.keyframes[k].pose3.x(), self.keyframes[k].pose3.y(), self.keyframes[k].dr_pose3.z()
                links.append((p0, p2, "red"))

        #if nothing, do nothing
        if links:

            #conver this list to a series of multi-colored lines and publish
            link_msg = ros_constraints(links)
            link_msg.header.stamp = self.current_keyframe.time
            if self.rov_id != "":
                link_msg.header.frame_id = self.rov_id + "_map"
            self.constraint_pub.publish(link_msg)


    def publish_trajectory(self)->None:
        """Publish 3D trajectory as point cloud in [x, y, z, roll, pitch, yaw, index] format.
        """

        #get all the poses from each keyframe
        poses = np.array([g2n(kf.pose3) for kf in self.keyframes])

        #convert to a ros color line
        traj_msg = ros_colorline_trajectory(poses)
        traj_msg.header.stamp = self.current_keyframe.time
        if self.rov_id == "":
            traj_msg.header.frame_id = "map"
        else:
            traj_msg.header.frame_id = self.rov_id + "_map"
        self.traj_pub.publish(traj_msg)

    def publish_point_cloud(self)->None:
        """Publish downsampled 3D point cloud with z = 0.
        The last column represents keyframe index at which the point is observed.
        """

        #define an empty array
        all_points = [np.zeros((0, 2), np.float32)]

        #list of keyframe ids
        all_keys = []

        #loop over all the keyframes, register 
        #the point cloud to the orign based on the SLAM estinmate
        for key in range(len(self.keyframes)):

            #parse the pose
            pose = self.keyframes[key].pose

            #get the resgistered point cloud
            transf_points = self.keyframes[key].transf_points

            #append
            all_points.append(transf_points)
            all_keys.append(key * np.ones((len(transf_points), 1)))

        all_points = np.concatenate(all_points)
        all_keys = np.concatenate(all_keys)

        #use PCL to downsample this point cloud
        sampled_points, sampled_keys = pcl.downsample(
            all_points, all_keys, self.point_resolution
        )

        #parse the downsampled cloud into the ros xyzi format
        sampled_xyzi = np.c_[sampled_points, np.zeros_like(sampled_keys), sampled_keys]
        
        #if there are no points return and do nothing
        if len(sampled_xyzi) == 0:
            return

        #convert the point cloud to a ros message and publish
        cloud_msg = n2r(sampled_xyzi, "PointCloudXYZI")
        cloud_msg.header.stamp = self.current_keyframe.time
        if self.rov_id == "":
            cloud_msg.header.frame_id = "map"
        else:
            cloud_msg.header.frame_id = self.rov_id + "_map"
        self.cloud_pub.publish(cloud_msg)
