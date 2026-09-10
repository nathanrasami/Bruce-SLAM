import gtsam
import numpy as np

from ctypes import Union
from typing import Any
from numpy import True_
from scipy.optimize import shgo
from itertools import combinations
from collections import defaultdict
from sklearn.covariance import MinCovDet
import time as time_pkg

from .sonar import OculusProperty
from .utils.conversions import *
from .utils.visualization import *
from .utils.io import *
from . import pcl

from bruce_slam.slam_objects import (
    STATUS,
    Keyframe,
    InitializationResult,
    ICPResult,
    SMParams,
)
from bruce_slam.sonar_context import cosine_distance_shifted


class SLAM(object):
    """The class to run underwater sonar based SLAM"""

    def __init__(self):
        """Class constructor for the SLAM class, note we do not feed arguments in in the pythonic way
        we use the ros param system to get the params. Note that almost everything is eligible for
        overwrite when the yaml file is called. See config/slam.yaml."""

        # configure sonar info
        self.oculus = OculusProperty()

        # Create a new factor when
        # - |ti - tj| > min_duration and
        # - |xi - xj| > max_translation or
        # - |ri - rj| > max_rotation
        self.keyframe_duration = None
        self.keyframe_translation = None
        self.keyframe_rotation = None

        # List of keyframes, a keyframe is a step in the SLAM solution
        self.keyframes = []

        # Current (non-key)frame with real-time pose update
        # TODO propagate cov from previous keyframe
        self.current_frame = None

        # init isam the graph optimization tool
        self.isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(self.isam_params)

        # define the graph and initial guess matrix, values. Use these to push info into isam
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        # initial location noise model [x, y, theta]
        self.prior_sigmas = None  # place holder

        # Noise model without ICP, just dead reckoning
        # [x, y, theta]
        self.odom_sigmas = None  # place holder

        # Downsample paramter for point cloud for ICP and publishing
        self.point_resolution = 0.5

        # Noise radius in overlap estimation
        self.point_noise = 0.5

        # paramters for sequnetial scan matching (SSM)
        self.ssm_params = SMParams()  # object to hold all the params
        self.ssm_params.initialization = True  # flag to indicate if we did this step
        self.ssm_params.initialization_params = 50, 1, 0.01
        self.ssm_params.min_st_sep = 1
        self.ssm_params.min_points = 50
        self.ssm_params.max_translation = 2.0
        self.ssm_params.max_rotation = np.pi / 6
        self.ssm_params.target_frames = 3
        # Don't use ICP covariance
        self.ssm_params.cov_samples = 0

        # paramters for loop closures (NSSM)
        self.nssm_params = SMParams()
        self.nssm_params.initialization = True
        self.nssm_params.initialization_params = 100, 5, 0.01
        self.nssm_params.min_st_sep = 10
        self.nssm_params.min_points = 100
        self.nssm_params.max_translation = 6.0
        self.nssm_params.max_rotation = np.pi / 2
        self.nssm_params.source_frames = 5
        self.nssm_params.cov_samples = 30
        # Plafond des bornes de recherche shgo (init ICP global). Sans boucle
        # fermée, la covariance accumulée explose (±130 m mesuré) et shgo
        # échantillonne trop grossièrement pour converger → 0 boucle. On borne
        # au domaine physique où une vraie boucle peut exister (recouvrement
        # sonar + dérive locale). Surchargé par le YAML (nssm/shgo_*).
        self.nssm_params.shgo_max_translation = 20.0
        self.nssm_params.shgo_max_rotation = np.pi

        # define ICP
        self.icp = pcl.ICP()
        self.icp_ssm = pcl.ICP()

        # Pairwise consistent measurement, for loop closure outlier rejection
        self.nssm_queue = []  # the loop closur queue
        self.pcm_queue_size = 5  # default val
        self.min_pcm = 3  # default val

        # SONAR Context (Kim, ICRA 2023) — remplace la DÉTECTION de candidat
        # du NSSM (gating covariance + argmax counts) par de la place
        # recognition par apparence. Tout l'aval (shgo, ICP+cov, PCM) inchangé.
        self.sc_enable = False
        self.sc_knn = 5                # candidats retenus par la Polar Key
        self.sc_dist_threshold = 0.35  # distance cosinus max pour valider
        self.sc_max_azimuth_shift = 10
        self.sc_max_range_shift = 5
        # U4 (branche Ultime) : UNION des détecteurs — soumettre à l'aval les
        # candidats de SC ET du gating natif (dédupliqués), le PCM commun tranche.
        self.sc_union = False
        self.sc_log = []  # (source, target, dist, shift_az, shift_rg, retenu, detector)

        # USBL — facteurs de POSITION ABSOLUE acoustique (ancrage indépendant de
        # la GT). Prouvé en sandbox (usbl_sim.py) : ~3 m lisse vs 47 m en
        # dead-reckoning seul. Chaque fix devient un prior unaire robuste qui ne
        # contraint que x,y (le cap reste géré par l'odométrie). cf. USBL_FACTEURS.md
        self.usbl_enable = False
        self.usbl_sigma = 1.4       # m, bruit d'un fix (médiane mesurée vs GT)
        # U6 (branche Ultime) : σ ADAPTATIF PAR FIX, 100% GT-free — la dispersion
        # locale des fixes (MAD glissant) estime le bruit réel, qui varie ×3.5 le
        # long de la mission (méd 0.87 m à t=10-15 min vs 3.09 m à t=25-30, mesuré
        # vs GT offline ; le proxy corrèle à 0.64 Spearman avec le vrai résidu).
        # σ_i = clip(k·MAD_local, min, max) ; k=3.4 calé pour médiane 1.8 (=RU1).
        self.usbl_adaptive = False
        self.usbl_adaptive_k = 3.4
        self.usbl_adaptive_min = 0.9
        self.usbl_adaptive_max = 3.5
        self.usbl_flip_y = False    # néger Y des fixes (repère DISO réfléchi) ; cf. add_usbl
        self.usbl_max_dt = 1.0      # s, fenêtre d'association fix ↔ keyframe
        self.usbl_buffer = []       # (t, x, y) fixes acceptés, rempli par slam_ros
        self.usbl_added = 0         # nb de facteurs USBL réellement ajoutés
        # Recalage automatique repère monde→odométrie (rotation+réflexion, sans scale).
        # Remplace le flip_y codé en dur : DISO sort un repère réfléchi+TOURNÉ qu'un flip
        # d'axe fixe ne réconcilie pas. Estimé en ligne depuis (pose odom, fix USBL).
        self.usbl_align_enable = False    # activé via param usbl/align_enable (DISO seult)
        self.usbl_align_min_pairs = 8     # paires (pose,fix) avant d'estimer la transfo
        self.usbl_align_min_span = 4.0    # m, étendue min sur la 2e dim PCA (aire, pas ligne)
        self.usbl_align_lock_margin = 2.0  # m, écart résidu rotation vs réflexion pour trancher
        self._usbl_align_pairs = []       # (odom_x, odom_y, world_x, world_y)
        self._usbl_world2odom = None      # (R 2x2, t 2,) estimée ; None tant que pas prête
        self._usbl_align_locked = False   # True quand la transfo est fiable et figée

        # Use fixed noise model in two cases
        # - Sequential scan matching
        # - ICP cov is too small in non-sequential scan matching
        # [x, y, theta]
        self.icp_odom_sigmas = None

        # Can't save fig in online mode
        # TODO remove this
        self.save_fig = False
        self.save_data = False

    @property
    def current_keyframe(self) -> Keyframe:
        """Get the current keyframe from the SLAM solution

        Returns:
            Keyframe: the current keyframe (most recent keyframe) in the system
        """

        # the current keyframe
        return self.keyframes[-1]

    @property
    def current_key(self) -> int:
        """Get the length of the list that stores the keyframes

        Returns:
            int: the length of self.keyframes
        """

        return len(self.keyframes)

    def configure(self) -> None:
        """Configure SLAM"""

        # check nssm covariance params
        assert (
            self.nssm_params.cov_samples == 0
            or self.nssm_params.cov_samples
            < self.nssm_params.initialization_params[0]
            * self.nssm_params.initialization_params[1]
        )

        # check ssm covariance params
        assert (
            self.ssm_params.cov_samples == 0
            or self.ssm_params.cov_samples
            < self.ssm_params.initialization_params[0]
            * self.ssm_params.initialization_params[1]
        )

        assert self.nssm_params.source_frames < self.nssm_params.min_st_sep

        # create noise models
        self.prior_model = self.create_noise_model(self.prior_sigmas)
        self.odom_model = self.create_noise_model(self.odom_sigmas)
        self.icp_odom_model = self.create_noise_model(self.icp_odom_sigmas)

    def get_states(self) -> np.array:
        """Retrieve all states as array which are represented as
            [time, pose2, dr_pose3, cov]
            - pose2: [x, y, yaw]
            - dr_pose3: [x, y, z, roll, pitch, yaw]
            - cov: 3 x 3

        Returns:
            np.array: the state array
        """

        # build the state array
        states = np.zeros(
            self.current_key,
            dtype=[
                ("time", np.float64),
                ("pose", np.float32, 3),
                ("dr_pose3", np.float32, 6),
                ("cov", np.float32, 9),
            ],
        )

        # Update all
        values = self.isam.calculateEstimate()
        for key in range(self.current_key):
            pose = values.atPose2(X(key))
            cov = self.isam.marginalCovariance(X(key))
            self.keyframes[key].update(pose, cov)

        # pull the state
        t_zero = self.keyframes[0].time
        for key in range(self.current_key):
            keyframe = self.keyframes[key]
            states[key]["time"] = (keyframe.time - t_zero).to_sec()
            states[key]["pose"] = g2n(keyframe.pose)
            states[key]["dr_pose3"] = g2n(keyframe.dr_pose3)
            states[key]["cov"] = keyframe.transf_cov.ravel()
        return states

    @staticmethod
    def sample_pose(pose: gtsam.Pose2, covariance: np.array) -> gtsam.Pose2:
        """Generate a random pose using the covariance matrix to define a normal dist.

        Args:
            pose (gtsam.Pose2): The pose we wish to add random to
            covariance (np.array): The covariance associated with this pose

        Returns:
            gtsam.Pose2: the provided pose with some random noise added
        """

        # get the random noise and add it to the provided pose
        delta = np.random.multivariate_normal(np.zeros(3), covariance)
        return pose.compose(n2g(delta, "Pose2"))

    def sample_current_pose(self) -> gtsam.Pose2:
        """Add random noise to self.current_keyframe.pose using self.sample_pose()

        Returns:
            gtsam.Pose2: The self.current_keyframe.pose with noise added
        """

        return self.sample_pose(self.current_keyframe.pose, self.current_keyframe.cov)

    def get_points(
        self, frames: list = None, ref_frame: Any = None, return_keys: bool = False
    ) -> np.array:
        """Get a point cloud, doing the following steps
            - Accumulate points in frames
            - Transform them to reference frame
            - Downsample points
            - Return the corresponding keys for every point

        Args:
            frames (list, optional): The list of indexes for the frames we care about. Defaults to None.
            ref_frame (Any, optional): The frame we want the points relative to, can be gtsam.Pose2 or int index. Defaults to None.
            return_keys (bool, optional): Do we want to return the keys?. Defaults to False.

        Returns:
            np.array: the point cloud array, maybe with keys for each point
        """

        # if there are no frames speced just get them all
        if frames is None:
            frames = range(self.current_key)

        # check if the ref frame is a gtsam.Pose2, if it is not we assume it's an index in the list of self.keyframes
        if ref_frame is not None:
            if isinstance(ref_frame, gtsam.Pose2):
                ref_pose = ref_frame
            else:
                ref_pose = self.keyframes[ref_frame].pose

        # Define a blank array to add our points to
        if return_keys:
            all_points = [np.zeros((0, 3), np.float32)]
        else:
            all_points = [np.zeros((0, 2), np.float32)]

        # Loop over the provided keyframe indexes
        for key in frames:
            # if we have a reference frame then use that, otherwise use the SLAM frame
            if ref_frame is not None:
                # transform to the reference frame provided
                points = self.keyframes[key].points
                pose = self.keyframes[key].pose
                transf = ref_pose.between(pose)
                transf_points = Keyframe.transform_points(points, transf)
            else:
                transf_points = self.keyframes[key].transf_points

            # if we want the key with each point, get those here
            if return_keys:
                transf_points = np.c_[
                    transf_points, key * np.ones((len(transf_points), 1))
                ]
            all_points.append(transf_points)

        # combine the points into a numpy array
        all_points = np.concatenate(all_points)

        # apply voxel downsampling and return
        if return_keys:
            return pcl.downsample(
                all_points[:, :2], all_points[:, (2,)], self.point_resolution
            )
        else:
            return pcl.downsample(all_points, self.point_resolution)

    def compute_icp(
        self,
        source_points: np.array,
        target_points: np.array,
        guess: np.array = gtsam.Pose2(),
    ) -> Union:
        """Compute standard ICP

        Args:
            source_points (np.array): source point cloud [x,y]
            target_points (np.array): target point cloud [x,y]
            guess (np.array, optional): the inital guess, if not provided we use identity. Defaults to gtsam.Pose2().

        Returns:
            Union[str,gtsam.Pose2]: returns the status message and the result as a gtsam.Pose2
        """

        # setup the points
        source_points = np.array(source_points, np.float32)
        target_points = np.array(target_points, np.float32)

        # convert the guess to a matrix and apply ICP
        guess = guess.matrix()
        message, T = self.icp.compute(source_points, target_points, guess)

        # parse the ICP output
        x, y = T[:2, 2]
        theta = np.arctan2(T[1, 0], T[0, 0])

        return message, gtsam.Pose2(x, y, theta)

    def compute_icp_with_cov(
        self, source_points: np.array, target_points: np.array, guesses: list
    ) -> Union:
        """Compute ICP with a covariance matrix

        Args:
            source_points (np.array): source point cloud [x,y]
            target_points (np.array): target point cloud [x,y]
            guesses (list): list of initial guesses

        Returns:
            Union[str,gtsam.Pose2,np.array,np.array]: status message,transform,covariance matrix,transforms tested
        """

        # parse the points
        source_points = np.array(source_points, np.float32)
        target_points = np.array(target_points, np.float32)

        # check each of the provided guesses with ICP
        sample_transforms = []
        start = time_pkg.time()
        for g in guesses:
            g = g.matrix()
            message, T = self.icp.compute(source_points, target_points, g)

            # only keep what works
            if message == "success":
                x, y = T[:2, 2]
                theta = np.arctan2(T[1, 0], T[0, 0])
                sample_transforms.append((x, y, theta))

            # enforce a max run time for this loop
            if time_pkg.time() - start >= 2.0:
                break

        # check if we have enough transforms to get a covariance
        sample_transforms = np.array(sample_transforms)
        if len(sample_transforms) < 5:
            return "Too few samples for covariance computation", None, None, None

        # Can't use np.cov(). Too many outliers
        try:
            fcov = MinCovDet(store_precision=False, support_fraction=0.8).fit(
                sample_transforms
            )
        except ValueError as e:
            return "Failed to calculate covariance", None, None, None

        # parse the result
        m = n2g(fcov.location_, "Pose2")
        cov = fcov.covariance_

        # unrotate to local frame
        R = m.rotation().matrix()
        cov[:2, :] = R.T.dot(cov[:2, :])
        cov[:, :2] = cov[:, :2].dot(R)

        # check if the default covariance for ICP is bigger than the one we just estimated
        default_cov = np.diag(self.icp_odom_sigmas) ** 2
        if np.linalg.det(cov) < np.linalg.det(default_cov):
            cov = default_cov

        return "success", m, cov, sample_transforms

    def get_overlap(
        self,
        source_points: np.array,
        target_points: np.array,
        source_pose: gtsam.Pose2 = None,
        target_pose: gtsam.Pose2 = None,
        return_indices: bool = False,
    ) -> int:
        """Get the overlap between the provided clouds, the count of points with a nearest neighbor

        Args:
            source_points (np.array): source point cloud
            target_points (np.array): target point cloud
            source_pose (gtsam.Pose2, optional): pose for the source points. Defaults to None.
            target_pose (gtsam.Pose2, optional): pose for the target points. Defaults to None.
            return_indices (bool, optional): if we want the cloud indexes. Defaults to False.

        Returns:
            int: the number of points with a nearest neighbor
        """

        # transform the points if we have a pose
        if source_pose:
            source_points = Keyframe.transform_points(source_points, source_pose)
        if target_pose:
            target_points = Keyframe.transform_points(target_points, target_pose)

        # match the points using nearest neigbor with PCL
        # note that un-matched points get a -1 in indices
        indices, dists = pcl.match(target_points, source_points, 1, self.point_noise)

        # if we want the indices, send those
        if return_indices:
            return np.sum(indices != -1), indices
        else:
            return np.sum(indices != -1)

    def add_prior(self, keyframe: Keyframe) -> None:
        """Add the prior factor for the first pose in the SLAM solution. This is the starting frame.

        Args:
            keyframe (Keyframe): the keyframe object for the initial frame
        """

        pose = keyframe.pose
        factor = gtsam.PriorFactorPose2(X(0), pose, self.prior_model)
        self.graph.add(factor)
        self.values.insert(X(0), pose)

    def add_odometry(self, keyframe: Keyframe) -> None:
        """Add the odometry factor between provided keyframe and the last keyframe

        Args:
            keyframe (Keyframe): the incoming keyframe, basically keyframe_t
        """

        # get the time a pose differnce between the provided keyframe and the last logged one
        dt = (keyframe.time - self.keyframes[-1].time).to_sec()
        dr_odom = self.keyframes[-1].pose.between(keyframe.pose)

        # build a factor and insert it into the graph, providing an initial guess as well
        factor = gtsam.BetweenFactorPose2(
            X(self.current_key - 1), X(self.current_key), dr_odom, self.odom_model
        )
        self.graph.add(factor)
        self.values.insert(X(self.current_key), keyframe.pose)

    def add_usbl(self, keyframe: Keyframe) -> None:
        """Ajoute un facteur de POSITION ABSOLUE USBL sur le keyframe courant, si un
        fix acoustique tombe dans la fenêtre ±usbl_max_dt autour de son timestamp.

        C'est un prior unaire ROBUSTE (Cauchy) qui ne contraint QUE x,y : on met un
        sigma θ énorme (1e6) → le cap reste libre, géré par l'odométrie. L'optimiseur
        gtsam recolle ainsi la trajectoire sur les ancres USBL en moyennant leur bruit
        (~1.4 m) et en rejetant les outliers acoustiques via le noyau robuste.
        Indépendant de la GT (USBL = capteur acoustique distinct). cf. USBL_FACTEURS.md
        """
        if not self.usbl_enable or not self.usbl_buffer:
            return
        t = keyframe.time.to_sec()
        # fix le plus proche en temps (recherche linéaire ; buffer ~1000 fixes max)
        best, bdt, best_idx = None, self.usbl_max_dt, None
        for i, (ut, ux, uy) in enumerate(self.usbl_buffer):
            dt = abs(ut - t)
            if dt < bdt:
                bdt, best, best_idx = dt, (ux, uy), i
        if best is None:
            return
        ux, uy = best
        # Recalage de repère monde→odométrie. DISO sort un repère RÉFLÉCHI+TOURNÉ par
        # rapport au monde (det(R)=-1 + angle fixé au seed), qu'un flip d'axe codé en
        # dur ne peut pas réconcilier (cf. offline_sim : meilleur flip = 20 m). On estime
        # donc la transfo rigide+réflexion (Umeyama 2D AVEC réflexion, sans scale) à
        # partir des paires (pose odométrie du KF, fix USBL) accumulées — GT-free.
        # Tant que la transfo n'est pas estimée, on bufferise et on n'ajoute pas encore.
        if self.usbl_flip_y:
            uy = -uy  # legacy : flip Y explicite si demandé (modes non-DISO)
        if self.usbl_align_enable:
            self._usbl_align_pairs.append(
                (keyframe.pose.x(), keyframe.pose.y(), ux, uy))
            # Ré-estime tant que pas convergé : avec peu de paires (trajectoire quasi
            # rectiligne) Umeyama ne distingue pas réflexion/rotation → on raffine au fil
            # de l'eau jusqu'à ce que l'aire couverte soit suffisante (transfo stable).
            if not self._usbl_align_locked:
                self._estimate_usbl_alignment()
            if self._usbl_world2odom is None:
                return  # pas encore estimable (pas assez de paires/d'aire) → on attend
            ux, uy = self._apply_usbl_alignment(ux, uy)
        # prior position-only robuste : sigma x,y = usbl_sigma ; sigma θ = 1e6 (libre)
        # U6 : en mode adaptatif, le σ vient de la dispersion locale des fixes.
        sigma = self.usbl_sigma
        if self.usbl_adaptive and best_idx is not None:
            sigma = self._usbl_adaptive_sigma(best_idx)
        model = self.create_robust_noise_model(sigma, sigma, 1e6)
        factor = gtsam.PriorFactorPose2(X(self.current_key), gtsam.Pose2(ux, uy, 0.0), model)
        self.graph.add(factor)
        self.usbl_added += 1
        # DIAG : trace chaque facteur USBL ajouté (fix vs pose odom courante) pour
        # vérifier que l'USBL agit réellement sur le back-end (sinon 0 correction).
        try:
            import rospy as _rospy
            px, py = keyframe.pose.x(), keyframe.pose.y()
            _rospy.loginfo("USBL factor #%d on KF%d: fix=(%.2f,%.2f) odom=(%.2f,%.2f) dist=%.2fm dt=%.2fs sigma=%.2f",
                           self.usbl_added, self.current_key, ux, uy, px, py,
                           ((ux - px) ** 2 + (uy - py) ** 2) ** 0.5, bdt, sigma)
        except Exception:
            pass

    def _usbl_adaptive_sigma(self, j: int) -> float:
        """U6 : σ par fix depuis la dispersion LOCALE des fixes — 100 % GT-free.

        Proxy validé offline (bag complet vs GT) : écart de chaque fix à la médiane
        de ses ±4 voisins, agrégé en MAD glissant sur ~21 fixes → corrélation 0.64
        (Spearman) avec le vrai résidu ; retrouve la fenêtre propre (t=10-15 min,
        σ̂ min) et la fenêtre multipath (t=25-30 min, σ̂ max) sans aucune GT.
        Invariant par rotation/réflexion du repère (distances seulement) → s'applique
        indifféremment avant/après le recalage monde→odométrie.

        Args:
            j: index (dans usbl_buffer) du fix associé au keyframe courant.

        Returns:
            float: σ (m) borné [usbl_adaptive_min, usbl_adaptive_max].
        """
        buf = self.usbl_buffer
        lo = max(0, j - 20)
        prox = []
        for i in range(lo, j + 1):
            a, b = max(0, i - 4), min(len(buf), i + 5)
            mx = float(np.median([p[1] for p in buf[a:b]]))
            my = float(np.median([p[2] for p in buf[a:b]]))
            prox.append(np.hypot(buf[i][1] - mx, buf[i][2] - my))
        if not prox:
            return self.usbl_sigma
        mad = 1.4826 * float(np.median(prox))
        return float(np.clip(self.usbl_adaptive_k * mad,
                             self.usbl_adaptive_min, self.usbl_adaptive_max))

    def _estimate_usbl_alignment(self) -> None:
        """Estime la transfo monde→odométrie (rotation OU réflexion + translation, sans
        scale) sur les paires (pose odom, fix USBL) accumulées. GT-free.

        Le repère DISO peut être réfléchi (det=-1) OU non (det=+1) selon le seed ; on ne
        peut pas le savoir a priori. On estime DONC les DEUX hypothèses (Umeyama rotation
        pure + Umeyama réflexion) et on garde celle au meilleur résidu. Cf. offline : avec
        trop peu de courbure 2D les deux se valent → on ne verrouille que quand la
        réflexion est tranchée (un cas nettement meilleur) ET l'aire 2D est suffisante."""
        pairs = self._usbl_align_pairs
        if len(pairs) < self.usbl_align_min_pairs:
            return
        P = np.array(pairs, dtype=float)
        odom = P[:, :2]   # repère odométrie (cible)
        world = P[:, 2:]  # repère monde (source = fixes USBL)
        mw, mo = world.mean(0), odom.mean(0)
        wc, oc = world - mw, odom - mo
        H = wc.T @ oc / len(P)
        U, S, Vt = np.linalg.svd(H)

        def make(reflect):
            D = np.eye(2)
            if reflect:
                D[1, 1] = -1
            R = (U @ D @ Vt).T
            t = mo - R @ mw
            res = float(np.median(np.linalg.norm((world @ R.T + t) - odom, axis=1)))
            return R, t, res

        # Umeyama standard force det>0 ; on génère explicitement les deux signes.
        rot = make(False)
        refl = make(True)
        best = min(rot, refl, key=lambda x: x[2])
        R, t, res = best
        self._usbl_world2odom = (R, t)
        # Verrouille quand : (1) aire 2D suffisante (2e val. sing. PCA) — sinon le choix
        # rotation/réflexion n'est pas observable ; (2) une hypothèse domine nettement
        # l'autre (sinon ambiguïté). Évite de figer une transfo dégénérée trop tôt.
        svals = np.linalg.svd(wc, compute_uv=False) / max(len(P) ** 0.5, 1.0)
        margin = abs(rot[2] - refl[2])
        if (svals[-1] >= self.usbl_align_min_span
                and margin >= self.usbl_align_lock_margin
                and len(P) >= 2 * self.usbl_align_min_pairs):
            self._usbl_align_locked = True
        try:
            import rospy as _rospy
            _rospy.loginfo("USBL alignment (%d paires): det(R)=%.2f résidu=%.2fm svals=[%.1f,%.1f] %s",
                           len(P), np.linalg.det(R), res, svals[0], svals[-1],
                           "LOCKED" if self._usbl_align_locked else "")
        except Exception:
            pass

    def _apply_usbl_alignment(self, ux: float, uy: float):
        """Applique la transfo monde→odométrie estimée à un fix USBL."""
        R, t = self._usbl_world2odom
        v = R @ np.array([ux, uy]) + t
        return float(v[0]), float(v[1])

    def get_map(self, frames, resolution=None):
        # Implemented in slam_node
        # TODO remove this code
        raise NotImplementedError

    def get_matching_cost_subroutine1(
        self,
        source_points: np.array,
        source_pose: gtsam.Pose2,
        target_points: np.array,
        target_pose: gtsam.Pose2,
        source_pose_cov: np.array = None,
    ) -> Union:
        """Perform global cost point cloud alignment. Here we transform source points to target points.

        Args:
            source_points (np.array): source point cloud
            source_pose (gtsam.Pose2): pose for the source_points
            target_points (np.array): target point cloud
            target_pose (gtsam.Pose2): pose for the target_points
            source_pose_cov (np.array, optional): Covariance for the source points. Defaults to None.

        Returns:
            Union[function,list]: the function to be optimized by scipy.shgo and a list of poses
        """
        # pose_samples = []
        # target_tree = KDTree(target_points)

        # def subroutine(x):
        #     # x = [x, y, theta]
        #     delta = n2g(x, "Pose2")
        #     sample_source_pose = source_pose.compose(delta)
        #     sample_transform = target_pose.between(sample_source_pose)

        #     points = Keyframe.transform_points(source_points, sample_transform)
        #     dists, indices = target_tree.query(
        #         points, distance_upper_bound=self.point_noise
        #     )

        #     cost = -np.sum(indices != len(target_tree.data))

        #     pose_samples.append(np.r_[g2n(sample_source_pose), cost])
        #     return cost

        # return subroutine, pose_samples

        # maintain a list of poses we try
        pose_samples = []

        # create a grid for the target points
        xmin, ymin = np.min(target_points, axis=0) - 2 * self.point_noise
        xmax, ymax = np.max(target_points, axis=0) + 2 * self.point_noise
        resolution = self.point_noise / 10.0
        xs = np.arange(xmin, xmax, resolution)
        ys = np.arange(ymin, ymax, resolution)
        target_grids = np.zeros((len(ys), len(xs)), np.uint8)

        # populate the grid for the target points
        r = np.int32(np.round((target_points[:, 1] - ymin) / resolution))
        c = np.int32(np.round((target_points[:, 0] - xmin) / resolution))
        r = np.clip(r, 0, target_grids.shape[0] - 1)
        c = np.clip(c, 0, target_grids.shape[1] - 1)
        target_grids[r, c] = 255

        # dilate the grid
        dilate_hs = int(np.ceil(self.point_noise / resolution))
        dilate_size = 2 * dilate_hs + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size), (dilate_hs, dilate_hs)
        )
        target_grids = cv2.dilate(target_grids, kernel)

        # # Calculate distance to the nearest points
        # target_grids = cv2.bitwise_not(target_grids)
        # target_grids = cv2.distanceTransform(target_grids, cv2.DIST_L2, 3)
        # target_grids = 1.0 - 0.2 * target_grids / self.point_noise
        # target_grids = np.clip(target_grids, 0.2, 1.0)

        source_pose_info = np.linalg.inv(source_pose_cov)

        def subroutine(x: np.array) -> float:
            """The optimization subroutine, called iterativly by scipy.shgo

            Args:
                x (gtsam.Pose2): the source pose as an array. [x, y, theta]

            Returns:
                float: cost of this step
            """

            # package the incoming pose as a gtsam.Pose2
            # apply this pose to the source_pose and get the transform between source and target
            delta = n2g(x, "Pose2")
            sample_source_pose = source_pose.compose(delta)
            sample_transform = target_pose.between(sample_source_pose)

            # apply this new transform to the source points
            # then limit the points to only points that fit inside the target grid
            points = Keyframe.transform_points(source_points, sample_transform)
            r = np.int32(np.round((points[:, 1] - ymin) / resolution))
            c = np.int32(np.round((points[:, 0] - xmin) / resolution))
            inside = (
                (0 <= r)
                & (r < target_grids.shape[0])
                & (0 <= c)
                & (c < target_grids.shape[1])
            )

            # get the number of cells that overlap and log the pose
            cost = -np.sum(target_grids[r[inside], c[inside]] > 0)
            pose_samples.append(np.r_[g2n(sample_source_pose), cost])

            return cost

        return subroutine, pose_samples

    def get_matching_cost_subroutine2(self, source_points, source_pose, occ):
        # TODO remove this code
        """
        Ceres scan matching

        Cost = - sum_i  ||1 - M_nearest(Tx s_i)||^2,
                given transform Tx, source points S, occupancy map M
        """
        pose_samples = []
        x0, y0, resolution, occ_arr = occ

        def subroutine(x):
            # x = [x, y, theta]
            delta = n2g(x, "Pose2")
            sample_pose = source_pose.compose(delta)

            xy = Keyframe.transform_points(source_points, sample_pose)
            r = np.int32(np.round((xy[:, 1] - y0) / resolution))
            c = np.int32(np.round((xy[:, 0] - x0) / resolution))

            sel = (r >= 0) & (c >= 0) & (r < occ_arr.shape[0]) & (c < occ_arr.shape[1])
            hit_probs_inside_map = occ_arr[r[sel], c[sel]]
            num_hits_outside_map = len(xy) - np.sum(sel)

            cost = (
                np.sum((1.0 - hit_probs_inside_map) ** 2)
                + num_hits_outside_map * (1.0 - 0.5) ** 2
            )
            cost = np.sqrt(cost / len(source_points))

            pose_samples.append(np.r_[g2n(sample_pose), cost])
            return cost

        return subroutine, pose_samples

    def initialize_sequential_scan_matching(
        self, keyframe: Keyframe
    ) -> InitializationResult:
        """Init a sequential scan matching call by using global ICP.

        Args:
            keyframe (Keyframe): the keyframe we want to register

        Returns:
            InitializationResult: the results of the the initilization
        """

        # instanciate an ICP InitializationResult object
        ret = InitializationResult()
        ret.status = STATUS.SUCCESS
        ret.status.description = None

        # Match current keyframe to previous k frames
        ret.source_key = self.current_key
        ret.target_key = self.current_key - 1
        ret.source_pose = keyframe.pose
        ret.target_pose = self.current_keyframe.pose

        # Accumulate reference points from previous k (self.ssm_params.target_frames) frames
        ret.source_points = keyframe.points
        target_frames = range(self.current_key)[-self.ssm_params.target_frames :]
        ret.target_points = self.get_points(target_frames, ret.target_key)
        ret.cov = np.diag(self.odom_sigmas)

        """if True:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret"""

        """if len(self.keyframes) % 2 == 0:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret"""

        # Only continue with this if it is enabled in slam.yaml
        if self.ssm_params.enable == False:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        # check the source points for a minimum count
        if len(ret.source_points) < self.ssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        # check the target points for a minimum count
        if len(ret.target_points) < self.ssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "target points {}".format(len(ret.target_points))
            return ret

        # check if we have initialized the ICP params
        if not self.ssm_params.initialization:
            return ret

        with CodeTimer("SLAM - sequential scan matching - sampling"):

            # define the search space for ICP global init
            pose_stds = np.array([self.odom_sigmas]).T
            pose_bounds = 5.0 * np.c_[-pose_stds, pose_stds]

            # TODO remove
            # ret.occ = self.get_map(target_frames)
            # subroutine, pose_samples = self.get_matching_cost_subroutine2(
            #     ret.source_points,
            #     ret.source_pose,
            #     ret.occ,
            # )

            # build the global ICP subroutine
            subroutine, pose_samples = self.get_matching_cost_subroutine1(
                ret.source_points,
                ret.source_pose,
                ret.target_points,
                ret.target_pose,
                ret.cov,
            )

            # optimize the subroutine using scipy.shgo
            result = shgo(
                func=subroutine,
                bounds=pose_bounds,
                n=self.ssm_params.initialization_params[0],
                iters=self.ssm_params.initialization_params[1],
                sampling_method="sobol",
                minimizer_kwargs={
                    "options": {"ftol": self.ssm_params.initialization_params[2]}
                },
            )

        # if the optimizer indicate success package results for return
        if result.success:
            ret.source_pose_samples = np.array(pose_samples)
            ret.estimated_source_pose = ret.source_pose.compose(n2g(result.x, "Pose2"))
            ret.status.description = "matching cost {:.2f}".format(result.fun)

            # TODO remove
            if self.save_data:
                ret.save("step-{}-ssm-sampling.npz".format(self.current_key))
        else:
            ret.status = STATUS.INITIALIZATION_FAILURE
            ret.status.description = result.message

        return ret

    def add_sequential_scan_matching(self, keyframe: Keyframe) -> None:
        """Add the sequential scan matching factor to the graph. Here we use the global ICP as an inital
        guess for standard ICP. We then perform some simple checks to catch silly outliers. If those
        checks pass we add the ICP result to the pose graph.

        Args:
            keyframe (Keyframe): The keyframe we are evaluating, this contains all the relevant info.
        """

        # call the global-ICP
        ret = self.initialize_sequential_scan_matching(keyframe)

        # TODO remove this
        if self.save_fig:
            ret.plot("step-{}-ssm-sampling.png".format(self.current_key))

        # check the status of the global-ICP call, if the result is a failure.
        # simply add the odometry factor and return
        if not ret.status:
            self.add_odometry(keyframe)
            return

        # copy the global-ICP into an ICPResult
        ret2 = ICPResult(ret, self.ssm_params.cov_samples > 0)

        # Compute ICP here with a timer
        with CodeTimer("SLAM - sequential scan matching - ICP"):

            # if possible compute ICP with covariance estimation
            if self.ssm_params.initialization and self.ssm_params.cov_samples > 0:
                message, odom, cov, sample_transforms = self.compute_icp_with_cov(
                    ret2.source_points,
                    ret2.target_points,
                    ret2.initial_transforms[: self.ssm_params.cov_samples],
                )

                # if ICP fails, push that into the ret2 object
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                # Else push the ICP info into ret2
                else:
                    ret2.estimated_transform = odom
                    ret2.cov = cov
                    ret2.sample_transforms = sample_transforms
                    ret2.status.description = "{} samples".format(
                        len(ret2.sample_transforms)
                    )

            # Else call standard ICP
            else:
                message, odom = self.compute_icp(
                    ret2.source_points, ret2.target_points, ret2.initial_transform
                )

                # check for failure
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret2.status.description = ""

        # The transformation compared to dead reckoning can't be too large
        if ret2.status:
            delta = ret2.initial_transform.between(ret2.estimated_transform)
            delta_translation = np.linalg.norm(delta.translation())
            delta_rotation = abs(delta.theta())
            if (
                delta_translation > self.ssm_params.max_translation
                or delta_rotation > self.ssm_params.max_rotation
            ):
                ret2.status = STATUS.LARGE_TRANSFORMATION
                ret2.status.description = "trans {:.2f} rot {:.2f}".format(
                    delta_translation, delta_rotation
                )

        # There must be enough overlap between two point clouds.
        if ret2.status:
            overlap = self.get_overlap(
                ret2.source_points, ret2.target_points, ret2.estimated_transform
            )
            if overlap < self.ssm_params.min_points:
                ret2.status = STATUS.NOT_ENOUGH_OVERLAP
            ret2.status.description = "overlap {}".format(overlap)

        if ret2.status:

            # if we used ICP with covariance then we don't need a boilerplate noise model
            if ret2.cov is not None:
                icp_odom_model = self.create_full_noise_model(ret2.cov)
            else:
                icp_odom_model = self.icp_odom_model

            # package a factor to be added to the graph
            factor = gtsam.BetweenFactorPose2(
                X(ret2.target_key),
                X(ret2.source_key),
                ret2.estimated_transform,
                icp_odom_model,
            )

            # Add the factor and the initial guess for this new pose
            self.graph.add(factor)
            self.values.insert(
                X(ret2.source_key), ret2.target_pose.compose(ret2.estimated_transform)
            )
            ret2.inserted = True  # log as added

            # TODO remove
            if self.save_data:
                ret2.save("step-{}-ssm-icp.npz".format(self.current_key))

        # If ICP was a failure, then just push in the dead reckoning info
        else:
            self.add_odometry(keyframe)

        # TODO remove
        if self.save_fig:
            ret2.plot("step-{}-ssm-icp.png".format(self.current_key))

    def initialize_nonsequential_scan_matching(self, detector: str = None) -> InitializationResult:
        """Initialize a nonsequential scan matching call. Here we use global ICP to check for loop closures with the
        most recent keyframe and the rest of the map.

        Args:
            detector: None = comportement historique (SC si sc_enable, sinon gating
                natif) ; "sc" ou "native" force le détecteur (mode union U4).

        Returns:
            InitializationResult: The global-ICP outcome
        """

        # instanciate an object to capute the results
        ret = InitializationResult()
        ret.status = STATUS.SUCCESS
        ret.status.description = None

        # get the indices we care about for loop closure search
        ret.source_key = self.current_key - 1
        ret.source_pose = self.current_frame.pose
        ret.estimated_source_pose = ret.source_pose
        # aggratgate the source cloud, here we want k frames (self.nssm_params.source_frames)
        source_frames = range(
            ret.source_key, ret.source_key - self.nssm_params.source_frames, -1
        )
        ret.source_points = self.get_points(source_frames, ret.source_key)

        # gate loop closure search to those who have sufficent points
        if len(ret.source_points) < self.nssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        # Find target points for matching
        # Limit searching keyframes. Here we want ALL the keyframes minus k (self.nssm_params.min_st_sep)
        target_frames = range(self.current_key - self.nssm_params.min_st_sep)

        # Target points in global frame
        target_points, target_keys = self.get_points(target_frames, None, True)

        use_sc = self.sc_enable if detector is None else (detector == "sc")
        if use_sc:
            # ===== SONAR Context : sélection du candidat par APPARENCE =====
            # Remplace le gating covariance+FOV et l'argmax(counts) — la source
            # des fake loops quand l'odométrie a dérivé. L'aval est inchangé.
            cand = self.sonar_context_candidate(target_frames)
            if cand is None:
                ret.status = STATUS.NOT_ENOUGH_POINTS
                ret.status.description = "sonar context: no candidate"
                return ret
            ret.target_key = cand[0]
            if len(target_points) < self.nssm_params.min_points:
                ret.status = STATUS.NOT_ENOUGH_POINTS
                ret.status.description = "target points {}".format(len(target_points))
                return ret
            cov = self.keyframes[ret.source_key].cov  # bornes shgo plus bas
        else:
            # ===== Détection géométrique d'origine (Bruce) =====
            # Loop over the source frames
            # Eliminate frames that do not have points in the same field of view
            sel = np.zeros(len(target_points), np.bool)
            for source_frame in source_frames:

                # pull the pose and covariance info
                pose = self.keyframes[source_frame].pose
                cov = self.keyframes[source_frame].cov

                # parse the covariance
                translation_std = np.sqrt(np.max(np.linalg.eigvals(cov[:2, :2])))
                rotation_std = np.sqrt(cov[2, 2])
                range_bound = translation_std * 5.0 + self.oculus.max_range
                bearing_bound = rotation_std * 5.0 + self.oculus.horizontal_aperture * 0.5

                # figure out the uncertain points
                local_points = Keyframe.transform_points(target_points, pose.inverse())
                ranges = np.linalg.norm(local_points, axis=1)
                bearings = np.arctan2(local_points[:, 1], local_points[:, 0])
                sel_i = (ranges < range_bound) & (abs(bearings) < bearing_bound)
                sel |= sel_i

            # only keep the certain points
            target_points = target_points[sel]
            target_keys = target_keys[sel]

            # Check which frame has the most points nearby
            target_frames, counts = np.unique(np.int32(target_keys), return_counts=True)
            target_frames = target_frames[counts > 10]
            counts = counts[counts > 10]

            # check the aggragate cloud for num of points
            if len(target_frames) == 0 or len(target_points) < self.nssm_params.min_points:
                ret.status = STATUS.NOT_ENOUGH_POINTS
                ret.status.description = "target points {}".format(len(target_points))
                return ret

            # populate the initilization object with some info
            ret.target_key = target_frames[
                np.argmax(counts)
            ]  # this is critical, the one with the most points overlapping

            # journal U4 : en mode union, tracer aussi les candidats du gating
            # natif (sc_dist=-1 : pas de distance d'apparence pour ceux-là)
            if self.sc_union:
                self.sc_log.append((ret.source_key, int(ret.target_key),
                                    -1.0, 0, 0, 1, "nssm"))

        ret.target_pose = self.keyframes[ret.target_key].pose
        ret.target_points = Keyframe.transform_points(
            target_points, ret.target_pose.inverse()
        )
        ret.cov = self.keyframes[ret.source_key].cov

        # check if we have the params for global ICP
        if not self.nssm_params.initialization:
            return ret

        with CodeTimer("SLAM - nonsequential scan matching - sampling"):

            # set bounds for global ICP
            translation_std = np.sqrt(np.max(np.linalg.eigvals(cov[:2, :2])))
            rotation_std = np.sqrt(cov[2, 2])
            pose_stds = np.array([[translation_std, translation_std, rotation_std]]).T
            pose_bounds = 5.0 * np.c_[-pose_stds, pose_stds]

            # Plafond physique : la covariance accumulée (sans boucle) gonfle les
            # bornes à ±130 m pour un sonar de ~48 m → shgo ne converge jamais.
            # On borne au domaine où une boucle est plausible. Bénéficie aux deux
            # chemins (SONAR Context et gating géométrique d'origine).
            cap = np.array([[self.nssm_params.shgo_max_translation],
                            [self.nssm_params.shgo_max_translation],
                            [self.nssm_params.shgo_max_rotation]])
            pose_bounds = np.clip(pose_bounds, -cap, cap)

            # TODO remove
            # ret.occ = self.get_map(target_frames)
            # subroutine, pose_samples = self.get_matching_cost_subroutine2(
            #     ret.source_points,
            #     ret.source_pose,
            #     ret.occ,
            # )

            # build the subroutine
            subroutine, pose_samples = self.get_matching_cost_subroutine1(
                ret.source_points,
                ret.source_pose,
                ret.target_points,
                ret.target_pose,
                ret.cov,
            )

            # optimize with scipy.shgo
            result = shgo(
                func=subroutine,
                bounds=pose_bounds,
                n=self.nssm_params.initialization_params[0],
                iters=self.nssm_params.initialization_params[1],
                sampling_method="sobol",
                minimizer_kwargs={
                    "options": {"ftol": self.nssm_params.initialization_params[2]}
                },
            )

        # check the shgo result
        if not result.success:
            ret.status = STATUS.INITIALIZATION_FAILURE
            ret.status.description = result.message
            return ret

        # parse the result
        delta = n2g(result.x, "Pose2")
        ret.estimated_source_pose = ret.source_pose.compose(delta)
        ret.source_pose_samples = np.array(pose_samples)
        ret.status.description = "matching cost {:.2f}".format(result.fun)

        # Refine target key by searching for the pose with maximum overlap
        # with current source points
        estimated_source_points = Keyframe.transform_points(
            ret.source_points, ret.estimated_source_pose
        )
        overlap, indices = self.get_overlap(
            estimated_source_points, target_points, return_indices=True
        )
        target_frames1, counts1 = np.unique(
            np.int32(target_keys[indices[indices != -1]]), return_counts=True
        )
        if len(counts1) == 0:
            ret.status = STATUS.NOT_ENOUGH_OVERLAP
            ret.status.description = "0"
            return ret

        # TODO remove
        if self.save_data:
            ret.save("step-{}-nssm-sampling.npz".format(self.current_key - 1))

        # log the target key and
        # recalculate target points with new target key in target frame
        ret.target_key = target_frames1[np.argmax(counts1)]
        ret.target_pose = self.keyframes[ret.target_key].pose
        ret.target_points = self.get_points(target_frames, ret.target_key)

        return ret

    def sonar_context_candidate(self, target_frames) -> tuple:
        """Sélection du candidat loop closure par SONAR Context (Kim, ICRA 2023).

        1) kNN brute-force sur les Polar Keys (euclidien) — rapide, < 1000 keyframes
        2) distance cosinus avec adaptive shifting sur les k meilleurs
        3) candidat retenu si distance < sc_dist_threshold
        Toute décision est tracée dans self.sc_log (validation étape 5).

        Args:
            target_frames: indices des keyframes candidats (déjà filtrés min_st_sep)

        Returns:
            (target_key, dist, shift_azimuth, shift_range) ou None si pas de candidat.
        """
        query = self.keyframes[self.current_key - 1]
        if query.ring_key is None or query.context is None:
            return None
        cands = [k for k in target_frames if self.keyframes[k].ring_key is not None]
        if not cands:
            return None

        # 0) Porte géométrique : un VRAI revisité est proche dans l'estimé courant
        # (DISO est bon → vraies boucles <10 m, fausses >34 m : séparation nette,
        # mesurée sur sc_descriptor_bench). On écarte les candidats invraisemblables
        # AVANT SC → élimine les faux positifs résiduels du descripteur (FPR ~24%)
        # sans perdre de vraies boucles. L'apparence propose, la géométrie vérifie.
        gate = getattr(self, "sc_gate_distance", 20.0)
        if gate > 0:
            qx, qy = query.pose.x(), query.pose.y()
            cands = [
                k for k in cands
                if (self.keyframes[k].pose.x() - qx) ** 2
                + (self.keyframes[k].pose.y() - qy) ** 2 <= gate * gate
            ]
            if not cands:
                return None

        # 1) Polar Keys → kNN euclidien
        keys = np.array([self.keyframes[k].ring_key for k in cands])
        d_pk = np.linalg.norm(keys - np.asarray(query.ring_key)[None, :], axis=1)
        order = np.argsort(d_pk)[: self.sc_knn]

        # 2) Sonar Context complet sur les kNN seulement (coûteux)
        best = None
        for idx in order:
            k = cands[int(idx)]
            d, sa, sr = cosine_distance_shifted(
                query.context,
                self.keyframes[k].context,
                self.sc_max_azimuth_shift,
                self.sc_max_range_shift,
            )
            if best is None or d < best[1]:
                best = (k, d, sa, sr)

        # 3) seuil de validation + journal (étape 5 : precision/recall offline)
        retenu = best[1] < self.sc_dist_threshold
        self.sc_log.append((self.current_key - 1, best[0], round(best[1], 4),
                            best[2], best[3], int(retenu), "sc"))
        return best if retenu else None

    def add_nonsequential_scan_matching(self) -> ICPResult:
        """Run a loop closure search. Here we compare the most recent keyframe to the
        previous frames. If a loop is found it is subject to geometric verification via PCM.

        Returns:
            ICPResult: the loop we have found, returns for debugging perposes
        """

        # if we do not have enough keyframes to aggratgate a submap return
        if self.current_key < self.nssm_params.min_st_sep:
            return

        # U4 (branche Ultime) : union des détecteurs. L'apparence (SC) et le
        # gating géométrique natif voient des candidats DIFFÉRENTS ; en mode
        # union on soumet les deux à l'aval (shgo/ICP), dédupliqués, et le PCM
        # commun tranche. Hors union : comportement historique inchangé.
        if self.sc_enable and self.sc_union:
            detectors = ("sc", "native")
        elif self.sc_enable:
            detectors = ("sc",)
        else:
            detectors = ("native",)

        out, tried = None, set()
        for detector in detectors:
            # init the search with a global ICP call
            ret = self.initialize_nonsequential_scan_matching(detector)

            # if the global ICP call did not work, try the next detector
            if not ret.status:
                continue

            # les deux détecteurs d'accord sur le même candidat → un seul ICP
            if ret.target_key in tried:
                continue
            tried.add(ret.target_key)

            ret2 = self._process_nssm_candidate(ret)
            if out is None:
                out = ret2
        return out

    def _process_nssm_candidate(self, ret: InitializationResult) -> ICPResult:
        """ICP + garde-fous + PCM pour UN candidat initialisé.

        Corps historique de add_nonsequential_scan_matching, inchangé — extrait
        pour pouvoir traiter plusieurs candidats par cycle en mode union (U4).

        Args:
            ret (InitializationResult): candidat initialisé (global ICP ok)

        Returns:
            ICPResult: the loop we have found, returns for debugging perposes
        """

        # package the global ICP call result
        ret2 = ICPResult(ret, self.nssm_params.cov_samples > 0)

        # Compute ICP here with a timer
        with CodeTimer("SLAM - nonsequential scan matching - ICP"):

            

            # if possible, compute ICP with a covariance matrix
            if self.nssm_params.initialization and self.nssm_params.cov_samples > 0:
                message, odom, cov, sample_transforms = self.compute_icp_with_cov(
                    ret2.source_points,
                    ret2.target_points,
                    ret2.initial_transforms[: self.nssm_params.cov_samples],
                )

                # check the status
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret2.cov = cov
                    ret2.sample_transforms = sample_transforms
                    ret2.status.description = "{} samples".format(
                        len(ret2.sample_transforms)
                    )

            # otherwise use standard ICP
            else:
                message, odom = self.compute_icp(
                    ret2.source_points, ret2.target_points, ret2.initial_transform
                )

                # check status
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret.status.description = ""

        # Add some failure detections
        # The transformation compared to initial guess can't be too large
        if ret2.status:
            delta = ret2.initial_transform.between(ret2.estimated_transform)
            delta_translation = np.linalg.norm(delta.translation())
            delta_rotation = abs(delta.theta())
            if (
                delta_translation > self.nssm_params.max_translation
                or delta_rotation > self.nssm_params.max_rotation
            ):
                ret2.status = STATUS.LARGE_TRANSFORMATION
                ret2.status.description = "trans {:.2f} rot {:.2f}".format(
                    delta_translation, delta_rotation
                )

        # There must be enough overlap between two point clouds.
        if ret2.status:
            overlap = self.get_overlap(
                ret2.source_points, ret2.target_points[:, :2], ret2.estimated_transform
            )
            if overlap < self.nssm_params.min_points:
                ret2.status = STATUS.NOT_ENOUGH_OVERLAP
            ret2.status.description = str(overlap)
            
        # apply geometric verification, in this case PCM
        if ret2.status:

            # update the pcm queue
            while (
                self.nssm_queue
                and ret2.source_key - self.nssm_queue[0].source_key
                > self.pcm_queue_size
            ):
                self.nssm_queue.pop(0)

            # log the newest loop closure into the pcm queue and check PCM
            self.nssm_queue.append(ret2)
            pcm = self.verify_pcm(self.nssm_queue,self.min_pcm)

            # if the PCM result has no loop closures for us, the list pcm will be empty
            # loop over any results and add them to the graph
            for m in pcm:

                # pull the loop closure from the pcm queue
                ret2 = self.nssm_queue[m]

                # check if the loop has been added to the graph
                if not ret2.inserted:

                    # get a noise model
                    if ret2.cov is not None:
                        icp_odom_model = self.create_full_noise_model(ret2.cov)
                    else:
                        icp_odom_model = self.icp_odom_model

                    # build the factor and add it to the graph
                    factor = gtsam.BetweenFactorPose2(
                        X(ret2.target_key),
                        X(ret2.source_key),
                        ret2.estimated_transform,
                        icp_odom_model,
                    )
                    self.graph.add(factor)
                    self.keyframes[ret2.source_key].constraints.append(
                        (ret2.target_key, ret2.estimated_transform)
                    )
                    ret2.inserted = True  # update the status of this loop closure, don't add a loop twice

        return ret2

    def is_keyframe(self, frame: Keyframe) -> bool:
        """Check if a Keyframe object meets the conditions to be a SLAM keyframe.
        If the vehicle has moved enough. Either rotation or translation.

        Args:
            frame (Keyframe): the keyframe we want to check.

        Returns:
            bool: a flag indicating if we need to add this frame to the SLAM soltuion
        """

        # if there are no keyframes in our SLAM solution, this is the first one
        if not self.keyframes:
            return True

        # check for time
        duration = frame.time - self.current_keyframe.time
        if duration < self.keyframe_duration:
            return False

        # check for rotation and translation
        dr_odom = self.keyframes[-1].dr_pose.between(frame.dr_pose)
        translation = np.linalg.norm(dr_odom.translation())
        rotation = abs(dr_odom.theta())

        return (
            translation > self.keyframe_translation or rotation > self.keyframe_rotation
        )

    def create_full_noise_model(
        self, cov: np.array
    ) -> gtsam.noiseModel.Gaussian.Covariance:
        """Create a noise model from a numpy array using the gtsam api.

        Args:
            cov (np.array): numpy array of the covariance matrix.

        Returns:
            gtsam.noiseModel.Gaussian.Covariance: gtsam version of the input
        """

        return gtsam.noiseModel.Gaussian.Covariance(cov)

    def create_robust_full_noise_model(self, cov: np.array) -> gtsam.noiseModel.Robust:
        """Create a robust gtsam noise model from a numpy array

        Args:
            cov (np.array): numpy array of the covariance matrix

        Returns:
            gtsam.noiseModel.Robust: gtsam version of input
        """

        model = gtsam.noiseModel.Gaussian.Covariance(cov)
        robust = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
        return gtsam.noiseModel.Robust.Create(robust, model)

    def create_noise_model(self, *sigmas: list) -> gtsam.noiseModel.Diagonal:
        """Create a noise model from a list of sigmas, treated like a diagnal matrix.

        Returns:
            gtsam.noiseModel.Diagonal: gtsam version of input
        """
        return gtsam.noiseModel.Diagonal.Sigmas(np.r_[sigmas])

    def create_robust_noise_model(self, *sigmas: list) -> gtsam.noiseModel.Robust:
        """Create a robust noise model from a list of sigmas

        Returns:
            gtsam.noiseModel.Robust: gtsam verison of input
        """

        model = gtsam.noiseModel.Diagonal.Sigmas(np.r_[sigmas])
        robust = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
        return gtsam.noiseModel.Robust.Create(robust, model)

    def update_factor_graph(self, keyframe: Keyframe = None) -> None:
        """Update the internal SLAM estimate

        Args:
            keyframe (Keyframe, optional): The keyframe that needs to be added to the SLAM solution. Defaults to None.
        """

        # if we have a keyframe add it to our list of keyframes
        if keyframe:
            self.keyframes.append(keyframe)

        # push the newest factors into the ISAM2 instance
        self.isam.update(self.graph, self.values)
        self.graph.resize(0)  # clear the graph and values once we push it to ISAM2
        self.values.clear()

        # Update the whole trajectory
        values = self.isam.calculateEstimate()
        for x in range(values.size()):
            pose = values.atPose2(X(x))
            self.keyframes[x].update(pose)

        # Only update latest cov
        cov = self.isam.marginalCovariance(X(values.size() - 1))
        self.keyframes[-1].update(pose, cov)

        # Update the poses in pending loop closures for PCM
        for ret in self.nssm_queue:
            ret.source_pose = self.keyframes[ret.source_key].pose
            ret.target_pose = self.keyframes[ret.target_key].pose
            if ret.inserted:
                ret.estimated_transform = ret.target_pose.between(ret.source_pose)

    def verify_pcm(self, queue: list, min_pcm_value: int) -> list:
        """Get the pairwise consistent measurements.

        Args:
            queue (list): the list of loop closures being checked.
            min_pcm_value (int): the min pcm value we want

        Returns:
            list: returns any pairwise consistent loops. We return a list of indexes in the provided queue.
        """

        # check if we have enough loops to bother
        if len(queue) < min_pcm_value:
            return []

        # convert the loops to a consistentcy graph=
        G = defaultdict(list)
        for (a, ret_il), (b, ret_jk) in combinations(zip(range(len(queue)), queue), 2):
            pi = ret_il.target_pose
            pj = ret_jk.target_pose
            pil = ret_il.estimated_transform
            plk = ret_il.source_pose.between(ret_jk.source_pose)
            pjk1 = ret_jk.estimated_transform
            pjk2 = pj.between(pi.compose(pil).compose(plk))

            error = gtsam.Pose2.Logmap(pjk1.between(pjk2))
            md = error.dot(np.linalg.inv(ret_jk.cov)).dot(error)
            # chi2.ppf(0.99, 3) = 11.34
            if md < 11.34:  # this is not a magic number
                G[a].append(b)
                G[b].append(a)

        # find the sets of consistent loops
        maximal_cliques = list(self.find_cliques(G))

        # if we got nothing, return nothing
        if not maximal_cliques:
            return []

        # sort and return only the largest set, also checking that the set is large enough
        maximum_clique = sorted(maximal_cliques, key=len, reverse=True)[0]
        if len(maximum_clique) < min_pcm_value:
            return []

        return maximum_clique

    def find_cliques(self, G: defaultdict):
        """Returns all maximal cliques in an undirected graph.

        Args:
            G (defaultdict): consicentcy graph
        """

        if len(G) == 0:
            return

        adj = {u: {v for v in G[u] if v != u} for u in G}
        Q = [None]

        subg = set(G)
        cand = set(G)
        u = max(subg, key=lambda u: len(cand & adj[u]))
        ext_u = cand - adj[u]
        stack = []

        try:
            while True:
                if ext_u:
                    q = ext_u.pop()
                    cand.remove(q)
                    Q[-1] = q
                    adj_q = adj[q]
                    subg_q = subg & adj_q
                    if not subg_q:
                        yield Q[:]
                    else:
                        cand_q = cand & adj_q
                        if cand_q:
                            stack.append((subg, cand, ext_u))
                            Q.append(None)
                            subg = subg_q
                            cand = cand_q
                            u = max(subg, key=lambda u: len(cand & adj[u]))
                            ext_u = cand - adj[u]
                else:
                    Q.pop()
                    subg, cand, ext_u = stack.pop()
        except IndexError:
            pass
