# python imports
import threading
import bisect
import csv
import os
import tf
import rospy
import cv_bridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped

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

        # Filtre de PERSISTANCE : ne garde que les voxels monde observés depuis
        # >= min_obs keyframes différents. Les vraies structures (murs/quai, vues de
        # plusieurs poses) restent ; le backscatter diffus du fond (vu brièvement à
        # range ~constant qui balaie) part. Enlève les arcs sans cap de range.
        self.persistence_enable = rospy.get_param(ns + "persistence/enable", False)
        self.persistence_resolution = rospy.get_param(ns + "persistence/resolution", 1.0)
        self.persistence_min_obs = rospy.get_param(ns + "persistence/min_obs", 3)
        # Seuil d'intensité du MAP (découplé du seuil SLAM dans feature_aracati.yaml) :
        # le SLAM garde toutes les features (dense → loops), la carte ne garde que les
        # retours forts >= map_threshold (structures, pas le voile de fond). 0 = inactif.
        self.map_threshold = rospy.get_param(ns + "map/threshold", 0.0)
        # Cap de RENDU du nuage (découplé de la trajectoire) : si activé, on rend chaque
        # scan avec le cap COMPAS (-dr_theta + offset) au lieu du theta iSAM2. La position
        # x,y reste celle d'iSAM2 (trajectoire inchangée). Le theta iSAM2 est optimisé pour
        # la POSITION (USBL+loops) → tracke la direction de trajectoire → mauvais pour le
        # cloud ; le compas (gyro) donne le vrai yaw → retire le swirl (= C2 inline).
        # offset (deg) = calibration repère compas→monde (Aracati : +162°, cf c2_compass_offset).
        self.cloud_use_compass_cap = rospy.get_param(ns + "cloud/use_compass_cap", False)
        self.cloud_cap_offset = np.radians(rospy.get_param(ns + "cloud/cap_offset_deg", 0.0))

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
        # plafond des bornes de recherche shgo (cf. slam.py : covariance non
        # bornée → bornes ±130 m → shgo ne converge pas → 0 boucle)
        self.nssm_params.shgo_max_translation = rospy.get_param(
            ns + "nssm/shgo_max_translation", 20.0)
        self.nssm_params.shgo_max_rotation = rospy.get_param(
            ns + "nssm/shgo_max_rotation", np.pi)
        print("NSSM: ", self.nssm_params.enable)

        #pairwise consistency maximization parameters for loop closure
        #outliar rejection
        self.pcm_queue_size = rospy.get_param(ns + "pcm_queue_size")
        self.min_pcm = rospy.get_param(ns + "min_pcm")

        # SONAR Context : détection de loop closure par apparence
        self.sc_enable = rospy.get_param(ns + "sonar_context/enable", False)
        self.sc_knn = rospy.get_param(ns + "sonar_context/knn", 5)
        self.sc_dist_threshold = rospy.get_param(ns + "sonar_context/dist_threshold", 0.35)
        self.sc_max_azimuth_shift = rospy.get_param(ns + "sonar_context/max_azimuth_shift", 10)
        self.sc_max_range_shift = rospy.get_param(ns + "sonar_context/max_range_shift", 5)
        # porte géométrique : distance max (m) entre source et candidat dans
        # l'estimé courant (l'apparence propose, la géométrie vérifie)
        self.sc_gate_distance = rospy.get_param(ns + "sonar_context/gate_distance", 20.0)
        # gate Δz optionnel (prépa 3D) : |z_source − z_candidat| max en m, z par
        # /depth (dr_pose3.z). 0 = désactivé — les configs figées sont inchangées.
        self.sc_dz_gate = rospy.get_param(ns + "sonar_context/dz_gate", 0.0)
        self._descriptor_buffer = {}  # (sec, nsec) -> (context, ring_key)
        if self.sc_enable:
            rospy.Subscriber(SONAR_DESCRIPTOR_TOPIC, Float32MultiArray,
                             self._descriptor_callback, queue_size=50)
        print("SONAR CONTEXT: ", self.sc_enable)

        # USBL — facteurs de position absolue acoustique (cf. slam.add_usbl, USBL_FACTEURS.md)
        self.usbl_enable = rospy.get_param(ns + "usbl/enable", False)
        self.usbl_sigma = rospy.get_param(ns + "usbl/sigma", 1.4)
        self.usbl_max_dt = rospy.get_param(ns + "usbl/max_dt", 1.0)
        self.usbl_max_speed = rospy.get_param(ns + "usbl/max_speed", 3.0)
        # flip_y : néger Y des fixes USBL pour les mettre dans le repère DISO (axe Y
        # inversé, det(R)=-1 — cf. offline_sim). SANS ce flip, USBL (repère monde) et
        # DISO (repère réfléchi) sont en handedness opposés → gtsam déforme la trajectoire
        # (ATE 13.9 m). AVEC : ~0.9 m (validé offline). À True UNIQUEMENT avec odom_source=diso ;
        # cmd_vel est déjà en repère monde → laisser False.
        self.usbl_flip_y = rospy.get_param(ns + "usbl/flip_y", False)
        # Recalage auto repère monde→odométrie : nécessaire UNIQUEMENT avec DISO (repère
        # réfléchi, System.cpp:89 swappe x↔y). En cmd_vel l'odométrie est déjà en repère
        # monde → identité → on désactive (sinon risque d'estimer une transfo parasite).
        self.usbl_align_enable = rospy.get_param(ns + "usbl/align_enable", False)
        self._usbl_last = None  # (t,x,y) dernier fix accepté (gate outliers par vitesse)
        if self.usbl_enable:
            rospy.Subscriber(USBL_TOPIC, PointStamped, self._usbl_callback, queue_size=20)
        print("USBL FACTORS: ", self.usbl_enable)

        #cache the latest odom message; feature callback uses it directly
        self._latest_odom = None
        self._odom_buffer = []  # messages Odometry pour interpolation temporelle
        self._pending_features = []  # features en attente que l'odom les rattrape
        # accumule l'odométrie brute (= DISO sur Aracati) à pleine fréquence,
        # même base de temps que le GT → exporté dans odometry.csv
        self.odom_poses = []
        rospy.Subscriber(LOCALIZATION_ODOM_TOPIC, Odometry, self._odom_cache_callback, queue_size=50)
        rospy.Subscriber(SONAR_FEATURE_TOPIC, PointCloud2, self._feature_callback, queue_size=50)

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

        # ground truth subscriber (aracati2017 only)
        self.gt_poses = []
        rospy.Subscriber("/pose_gt", PoseStamped, self._gt_callback, queue_size=100)

        #register shutdown hook to export CSV
        rospy.on_shutdown(self.export_csv)

        #call the configure function
        self.configure()
        loginfo("SLAM node is initialized")

    @add_lock
    def sonar_callback(self, ping:OculusPing)->None:
        """Subscribe once to configure Oculus property.
        Assume sonar configuration doesn't change much.

        Args:
            ping (OculusPing): The sonar message. 
        """
        
        self.oculus.configure(ping)
        self.sonar_sub.unregister()

    def _usbl_callback(self, msg: PointStamped) -> None:
        """Reçoit un fix acoustique /usbl_point, rejette les outliers par gate de
        VITESSE vs le dernier fix accepté (les glitches ~73 m impliquent une vitesse
        physiquement impossible), puis l'empile dans usbl_buffer (lu par add_usbl).
        Gate indépendant de la dérive de l'odométrie. cf. USBL_FACTEURS.md"""
        t = msg.header.stamp.to_sec()
        ux, uy = msg.point.x, msg.point.y
        if self._usbl_last is not None:
            lt, lx, ly = self._usbl_last
            dt = t - lt
            if dt > 0 and ((ux - lx) ** 2 + (uy - ly) ** 2) ** 0.5 / dt > self.usbl_max_speed:
                return  # saut impossible → glitch acoustique
        self._usbl_last = (t, ux, uy)
        self.usbl_buffer.append((t, ux, uy))

    def _odom_cache_callback(self, msg: Odometry) -> None:
        self._latest_odom = msg
        # buffer complet pour interpolation au timestamp des features
        # (l'extraction CFAR est lente : la feature arrive avec plusieurs
        # secondes de retard, _latest_odom serait une pose du futur)
        self._odom_buffer.append(msg)
        if len(self._odom_buffer) > 6000:  # ~10 min à 10 Hz
            del self._odom_buffer[:1000]
        # log de l'odométrie brute à pleine fréquence (même base de temps que GT)
        self.odom_poses.append((msg.header.stamp.to_sec(),
                                msg.pose.pose.position.x,
                                msg.pose.pose.position.y))
        # draine les features en attente que l'odom vient de couvrir
        # (si DISO est en retard sur la CFAR, on ATTEND au lieu de jeter)
        newest = msg.header.stamp.to_sec()
        while self._pending_features and \
                self._pending_features[0].header.stamp.to_sec() <= newest:
            f = self._pending_features.pop(0)
            odom = self._interpolate_odom(f.header.stamp)
            if odom is not None:
                self.SLAM_callback(f, odom)

    def _interpolate_odom(self, stamp) -> Odometry:
        """Retourne l'odométrie interpolée au temps `stamp` (position linéaire,
        orientation par slerp). None si le buffer ne couvre pas encore ce temps."""
        buf = self._odom_buffer
        if not buf:
            return None
        t = stamp.to_sec()
        times = [m.header.stamp.to_sec() for m in buf]
        if t <= times[0]:
            return buf[0] if abs(times[0] - t) < 1.0 else None
        if t >= times[-1]:
            # feature en avance sur l'odom : ne devrait plus arriver ici
            # (la mise en attente est gérée dans _feature_callback)
            return buf[-1] if abs(t - times[-1]) < 0.5 else None
        i = bisect.bisect_left(times, t)
        m0, m1 = buf[i - 1], buf[i]
        t0, t1 = times[i - 1], times[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = m1.header.frame_id
        out.child_frame_id = m1.child_frame_id
        p0, p1 = m0.pose.pose.position, m1.pose.pose.position
        out.pose.pose.position.x = p0.x + a * (p1.x - p0.x)
        out.pose.pose.position.y = p0.y + a * (p1.y - p0.y)
        out.pose.pose.position.z = p0.z + a * (p1.z - p0.z)
        q0 = m0.pose.pose.orientation
        q1 = m1.pose.pose.orientation
        q = tf.transformations.quaternion_slerp(
            [q0.x, q0.y, q0.z, q0.w], [q1.x, q1.y, q1.z, q1.w], a)
        out.pose.pose.orientation.x = q[0]
        out.pose.pose.orientation.y = q[1]
        out.pose.pose.orientation.z = q[2]
        out.pose.pose.orientation.w = q[3]
        out.pose.covariance = m1.pose.covariance
        out.twist = m1.twist if a > 0.5 else m0.twist
        return out

    def _descriptor_callback(self, msg: Float32MultiArray) -> None:
        """Décode le descripteur SONAR Context publié par feature_extraction.
        Timestamp scindé en moitiés 16 bits (cf. _publish_descriptor : float32
        corromprait sec/nsec sinon → clé jamais trouvée, 0 boucle).
        Format : [sec_hi, sec_lo, nsec_hi, nsec_lo, A, R, context(A*R), key(R)]."""
        data = np.asarray(msg.data, dtype=np.float32)
        if len(data) < 6:
            return
        sec = (int(data[0]) << 16) | int(data[1])
        nsec = (int(data[2]) << 16) | int(data[3])
        A, R = int(data[4]), int(data[5])
        if len(data) != 6 + A * R + R:
            return
        context = data[6:6 + A * R].reshape(A, R)
        ring_key = data[6 + A * R:]
        self._descriptor_buffer[(sec, nsec)] = (context, ring_key)
        # borne mémoire : purge les plus anciens (insertion ordonnée en py3.7+)
        while len(self._descriptor_buffer) > 500:
            self._descriptor_buffer.pop(next(iter(self._descriptor_buffer)))

    def _feature_callback(self, feature_msg: PointCloud2) -> None:
        # odométrie AU TEMPS de la feature, pas la dernière reçue : avec le
        # retard de traitement CFAR, _latest_odom est en avance de plusieurs
        # secondes → les keyframes porteraient des poses du futur (ATE faussé)
        buf = self._odom_buffer
        if not buf or feature_msg.header.stamp.to_sec() > buf[-1].header.stamp.to_sec():
            # l'odom (DISO) n'a pas encore atteint ce temps → attendre, pas jeter
            # (sinon dès que DISO a >1 s de retard, plus aucun keyframe n'est créé)
            self._pending_features.append(feature_msg)
            if len(self._pending_features) > 300:
                self._pending_features.pop(0)
            return
        odom_msg = self._interpolate_odom(feature_msg.header.stamp)
        if odom_msg is None:
            return
        self.SLAM_callback(feature_msg, odom_msg)

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
        # col 1 (champ Z) transporte l'INTENSITÉ du retour (cf. feature_extraction) →
        # sert au filtrage du MAP (le SLAM n'utilise que x,y). 0 si non fourni (NaN/skip).
        point_intensity = points[:, 1].astype(np.float32) if len(points) else np.zeros(0, np.float32)
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
            # intensité parallèle aux points (même ordre) → filtrage du MAP
            frame.point_intensity = point_intensity

            # attache le descripteur SONAR Context publié avec le même stamp
            # (ring_key/context : champs Keyframe déjà prévus, jamais utilisés)
            if self.sc_enable:
                desc = self._descriptor_buffer.pop((time.secs, time.nsecs), None)
                if desc is not None:
                    frame.context, frame.ring_key = desc

            #perform seqential scan matching
            #if this is the first frame do not
            if not self.keyframes:
                self.add_prior(frame)
            else:
                self.add_sequential_scan_matching(frame)

            # USBL : facteur de position absolue acoustique sur ce keyframe (si activé
            # et qu'un fix tombe dans la fenêtre temporelle) — ancrage indépendant de GT
            if self.usbl_enable:
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

    def _persistence_filter(self, points: np.ndarray, keys: np.ndarray, inten: np.ndarray):
        """Ne garde que les points dont le voxel monde (taille persistence_resolution)
        a été observé depuis >= persistence_min_obs keyframes DISTINCTS. Les vraies
        structures (murs/quai) sont vues de plusieurs poses → gardées ; le backscatter
        diffus du fond, balayé à range ~constant, ne touche chaque voxel que brièvement
        → retiré. 100% géométrique, GT-free."""
        res = self.persistence_resolution
        cx = np.floor(points[:, 0] / res).astype(np.int64)
        cy = np.floor(points[:, 1] / res).astype(np.int64)
        k = keys[:, 0].astype(np.int64)
        # id de voxel compact (0..M-1)
        cells = np.stack([cx, cy], axis=1)
        uniq_cells, cell_id = np.unique(cells, axis=0, return_inverse=True)
        # nb de keyframes distincts par voxel = nb de paires (voxel,key) uniques
        pairs = np.stack([cell_id, k], axis=1)
        uniq_pairs = np.unique(pairs, axis=0)
        counts = np.bincount(uniq_pairs[:, 0], minlength=len(uniq_cells))
        keep_cell = counts >= self.persistence_min_obs
        mask = keep_cell[cell_id]
        # DIAG calibration : fraction de POINTS gardés selon le seuil min_obs (n'est
        # loggué qu'occasionnellement pour ne pas spammer). Permet de choisir min_obs.
        if getattr(self, "_persist_diag_n", 0) % 40 == 0:
            pc_per_cell = counts[cell_id]  # count du voxel de chaque point
            line = " ".join("≥%d:%.0f%%" % (m, 100.0 * (pc_per_cell >= m).mean())
                            for m in (5, 10, 15, 20, 30, 40))
            rospy.loginfo("PERSIST counts(points) %s | voxels=%d pts=%d",
                          line, len(uniq_cells), len(points))
        self._persist_diag_n = getattr(self, "_persist_diag_n", 0) + 1
        return points[mask], keys[mask], inten[mask]

    def _build_filtered_cloud(self):
        """Assemble le nuage global (transf_points de tous les keyframes) et applique
        les filtres du MAP : (1) seuil d'intensité map_threshold (ne garde que les
        retours forts = structures, indépendant du seuil SLAM dense), (2) persistance
        (retire le fond balayé). Utilisé par publish_point_cloud ET export_csv pour que
        RViz et le CSV soient IDENTIQUES. Retourne (points Nx2, keys Nx1, intensité N)
        — l'intensité par point sert au filtrage/découplage offline (build_final_map)."""
        pts, keys, inten = [], [], []
        for key in range(len(self.keyframes)):
            kf = self.keyframes[key]
            if self.cloud_use_compass_cap and kf.points is not None and len(kf.points):
                # rendu avec cap COMPAS (-dr_theta + offset), position iSAM2 conservée
                cap = -kf.dr_pose.theta() + self.cloud_cap_offset
                c, s = np.cos(cap), np.sin(cap)
                bp = kf.points
                tp = np.c_[c * bp[:, 0] - s * bp[:, 1] + kf.pose.x(),
                           s * bp[:, 0] + c * bp[:, 1] + kf.pose.y()].astype(np.float32)
            else:
                tp = kf.transf_points
            if tp is None or not len(tp):
                continue
            pts.append(tp)
            keys.append(key * np.ones((len(tp), 1)))
            pi = getattr(self.keyframes[key], "point_intensity", None)
            if pi is None or len(pi) != len(tp):
                pi = np.zeros(len(tp), np.float32)
            inten.append(np.asarray(pi).reshape(-1))
        if not pts:
            return (np.zeros((0, 2), np.float32), np.zeros((0, 1), np.float32),
                    np.zeros(0, np.float32))
        P = np.concatenate(pts); K = np.concatenate(keys); I = np.concatenate(inten)
        if self.map_threshold > 0:
            m = I >= self.map_threshold
            P, K, I = P[m], K[m], I[m]
        if self.persistence_enable and len(P):
            P, K, I = self._persistence_filter(P, K, I)
        return P, K, I

    def publish_point_cloud(self)->None:
        """Publish downsampled 3D point cloud with z = 0.
        The last column represents keyframe index at which the point is observed.
        """

        all_points, all_keys, _ = self._build_filtered_cloud()
        if len(all_points) == 0:
            return

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

    def _gt_callback(self, msg):
        q = msg.pose.orientation
        yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.gt_poses.append((msg.header.stamp.to_sec(),
                              msg.pose.position.x,
                              msg.pose.position.y,
                              yaw,
                              msg.pose.position.z))  # 2.5D : z GT (profondeur)

    def export_csv(self):
        """Export trajectory, point cloud and ground truth to CSV files on shutdown."""
        if not self.keyframes:
            return

        output_dir = os.environ.get("SLAM_RESULTS_DIR",
            os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", "..", "results")))
        os.makedirs(output_dir, exist_ok=True)

        # --- Trajectory CSV ---
        traj_path = os.path.join(output_dir, "trajectory.csv")
        with open(traj_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["keyframe_id", "time",
                             "x", "y", "theta",
                             "dr_x", "dr_y", "dr_theta",
                             "cov_xx", "cov_yy", "cov_tt",
                             "nssm_constraints", "z"])
            for i, kf in enumerate(self.keyframes):
                x = kf.pose.x()
                y = kf.pose.y()
                theta = kf.pose.theta()
                dr_x = kf.dr_pose.x()
                dr_y = kf.dr_pose.y()
                dr_theta = kf.dr_pose.theta()
                cov = kf.cov if kf.cov is not None else [[0,0,0],[0,0,0],[0,0,0]]
                nssm = len(kf.constraints)
                writer.writerow([i, kf.time.to_sec(),
                                 x, y, theta,
                                 dr_x, dr_y, dr_theta,
                                 cov[0][0], cov[1][1], cov[2][2],
                                 nssm, kf.pose3.z()])
        rospy.loginfo("Trajectory saved to %s", traj_path)

        # --- Point cloud CSV ---
        # MÊME filtre que RViz (intensité map_threshold + persistance) via le helper,
        # sinon le CSV serait brut alors que RViz est filtré.
        pts, keys, inten = self._build_filtered_cloud()
        cloud_path = os.path.join(output_dir, "pointcloud.csv")
        with open(cloud_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["keyframe_id", "x", "y", "intensity", "z"])
            for p, k, it in zip(pts, keys, inten):
                # 2.5D : les features n'ont pas d'élévation (z=0 capteur) → le z du point
                # = z du véhicule au keyframe (pose3, alimentée par /depth ou la GT).
                writer.writerow([int(k[0]), p[0], p[1], float(it),
                                 self.keyframes[int(k[0])].pose3.z()])
        rospy.loginfo("Point cloud saved to %s", cloud_path)

        # --- Journal SONAR Context (validation étape 5 : precision/recall) ---
        # retenu=1 : candidat sous le seuil, envoyé à l'ICP/PCM ;
        # croiser avec nssm_constraints de trajectory.csv pour les acceptés.
        if self.sc_log:
            sc_path = os.path.join(output_dir, "loops_detected.csv")
            with open(sc_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["source_key", "target_key", "sc_dist",
                                 "shift_azimuth", "shift_range", "retenu", "dz"])
                writer.writerows(self.sc_log)
            rospy.loginfo("Sonar Context log saved to %s", sc_path)

        # --- Journal funnel NSSM (étage TERMINAL de chaque tentative :
        # init/shgo/ICP/gates/PCM — diagnostic des rejets aval, PIEGES §10) ---
        if getattr(self, "nssm_log", None):
            nssm_path = os.path.join(output_dir, "nssm_attempts.csv")
            with open(nssm_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["source_key", "target_key", "stage", "detail"])
                writer.writerows(self.nssm_log)
            rospy.loginfo("NSSM funnel log saved to %s", nssm_path)

        # --- Ground truth CSV ---
        if self.gt_poses:
            gt_path = os.path.join(output_dir, "groundtruth.csv")
            with open(gt_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "x", "y", "theta", "z"])
                for row in self.gt_poses:
                    writer.writerow(row)
            rospy.loginfo("Ground truth saved to %s", gt_path)

        # --- Odometry CSV (odométrie brute = DISO sur Aracati, pleine fréquence) ---
        if self.odom_poses:
            odom_path = os.path.join(output_dir, "odometry.csv")
            with open(odom_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "x", "y"])
                for row in self.odom_poses:
                    writer.writerow(row)
            rospy.loginfo("Odometry saved to %s", odom_path)
