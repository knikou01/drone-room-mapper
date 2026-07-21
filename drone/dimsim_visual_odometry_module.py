"""
DimSimVisualOdometryModule

Computes RGB-D visual odometry from DimSim's color/depth LCM feed and
publishes the accumulated pose on a SEPARATE topic (odom_vo) rather than
replacing DimSim's own ground-truth /odom. This lets VO accuracy be
measured against ground truth before it's trusted for navigation.

Pattern reused directly from sim_camera/dimsim_camera_module.py: subscribe
to DimSim's raw LCM topics via a standalone LCM() instance rather than
declaring In[Image] streams, since DimSim's bridge publishes unconditionally
on fixed topic names regardless of which DimOS blueprint is running. Reuses
make_camera_info_default() and the depth-in-mm convention from that same
module rather than re-deriving either.

Requires open3d (already a dependency -- VoxelGridMapper uses it).

------------------------------------------------------------------------
KNOWN LIMITATION -- odom_vo's accuracy is not reliable enough to trust for
real navigation against DimSim's actual scene content. Do not swap any
navigation/mapping consumer's pose input from ground-truth /odom to
odom_vo without re-validating: drive a controlled path with
run_sim_dimsim_vo_only.py and check the divergence log's delta ratios
before trusting it.
------------------------------------------------------------------------
Two tracking methods are available (tracking_method config field):

"dense" -- Open3D's compute_rgbd_odometry, a photometric+geometric
frame-to-frame optimizer. Assumes small motion between consecutive frames;
when DimSim's color/depth capture rate is slow (pairing_gap large), the
optimizer can silently converge to a near-identity transform while still
reporting success=True instead of failing loudly. Kept for A/B comparison,
not the default.

"sparse" (default) -- ORB feature matching + depth-based 3D-3D RANSAC
rigid alignment (see _compute_transform_sparse), tolerant of larger,
irregular inter-frame gaps since it doesn't share dense odometry's
small-motion assumption. Depth is available on both frames, so each
matched keypoint pair unprojects to a genuine 3D-3D correspondence (not
2D-3D PnP), giving an absolute-scale rigid transform via Kabsch/SVD on the
RANSAC inlier set. Three layers of rejection guard against publishing a
wrong-but-plausible-looking transform (all treated as tracking failures,
not published): too few matches/inliers, near-planar/degenerate matched-
point geometry (min_point_spread_ratio), and physically-implausible
implied speed (max_linear_mps/max_angular_rps, checked against the real
elapsed time between frames). In practice, even with these safeguards,
DimSim's actual rendered scenes often don't have enough distinctive visual
texture for reliable ORB matching -- real-world success rate has been
observed in the single digits percent. _reject_counts (see __init__)
tracks which check is responsible, surfaced in the periodic divergence log.

------------------------------------------------------------------------
Design notes:
------------------------------------------------------------------------
1. Color/depth pairing: process on every color arrival using whatever
   depth is most recently available (see _on_color_image). pairing_gap
   (logged every divergence line) tracks how close the pairing actually is.

2. Coordinate frame conversion (see _camera_to_world_pose): Open3D/OpenCV
   camera convention is X=right, Y=down, Z=forward; DimOS/ROS world
   convention is X=forward, Y=left, Z=up. Axis remap: X_dimos = Z_cam,
   Y_dimos = -X_cam, Z_dimos = -Y_cam. Verified via synthetic test (fed a
   hand-built rotation matrix representing a known pure world-yaw, checked
   the extracted yaw matches).

3. Depth scale: RGBDImage.create_from_color_and_depth defaults to
   depth_scale=1000.0, which matches DimSim's raw uint16-millimeter depth
   format exactly -- so raw depth data is passed to Open3D WITHOUT
   pre-dividing by 1000, letting Open3D's own depth_scale handle it
   consistently with its internal pyramid/gradient computations. Do not
   copy DimSimCameraModule's point-cloud division pattern into this
   module's RGBDImage construction -- different purpose (manual
   unprojection vs. Open3D's internal handling).
------------------------------------------------------------------------
"""
from __future__ import annotations

import math
import time
from typing import Literal

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

from sim_camera.dimsim_camera_module import make_camera_info_default

# Topic names duplicated from dimsim_camera_module.py rather than imported,
# since that module's _RAW_*_TOPIC constants are underscore-prefixed
# (private) -- not a stable cross-module contract. If DimSim's bridge topic
# names ever change, both files need updating together.
_RAW_COLOR_TOPIC = "/color_image"
_RAW_DEPTH_TOPIC = "/depth_image"
_RAW_ODOM_TOPIC = "/odom"

logger = setup_logger()


class DimSimVisualOdometryConfig(ModuleConfig):
    tracking_method: Literal["dense", "sparse"] = "sparse"
    min_depth_m: float = 0.05
    max_depth_m: float = 15.0
    # ORB/RANSAC tuning for tracking_method="sparse" -- unused when "dense".
    orb_n_features: int = 500
    # Lowe's ratio test threshold for BFMatcher.knnMatch(k=2) -- standard
    # default (0.75), rejects ambiguous matches where the best and
    # second-best match are too close in descriptor distance to trust.
    orb_ratio_test_threshold: float = 0.75
    # Inlier distance threshold (metres) for cv2.estimateAffine3D's
    # internal RANSAC -- how far a matched 3D-3D correspondence pair is
    # allowed to deviate from the fitted rigid transform before being
    # rejected as an outlier (e.g. a wrong ORB match, or a point whose
    # depth reading was noisy).
    ransac_reproj_threshold_m: float = 0.05
    # Minimum RANSAC inliers required to trust a fit -- a real room's
    # repetitive/ambiguous structure (blank walls, symmetric furniture)
    # can let RANSAC converge on a plausible-looking but wrong consensus
    # among too few points. See max_linear_mps/max_angular_rps below for a
    # second, independent line of defense against this.
    ransac_min_inliers: int = 15
    # Physical plausibility bounds for tracking_method="sparse" -- a
    # computed transform is rejected (treated as a tracking failure, not
    # published) if it implies an average speed/turn-rate above these,
    # given the REAL elapsed wall-clock time between the two frames (not
    # assumed small). Deliberately generous (well above any expected real
    # driving speed for this simulated ground robot) -- a sanity backstop
    # against a clearly-wrong RANSAC consensus, not a tight motion model.
    max_linear_mps: float = 3.0
    max_angular_rps: float = 6.0  # ~344 deg/s
    # Minimum ratio between the smallest and largest singular value of the
    # matched 3D points' spread (after centering). Guards against a
    # different failure mode than the checks above: a near-planar or
    # near-collinear point configuration (e.g. facing a flat wall
    # head-on) can pass every count-based check while still being
    # numerically ill-conditioned for a full 3D rigid fit -- rotation
    # about axes within that near-plane is barely observable from the
    # data at all, regardless of how many points agree on it. See
    # _compute_transform_sparse's singular-value check. 0.05 is a
    # conservative starting point (rejects clearly-degenerate cases
    # without being overly strict on genuinely-3D-but-somewhat-flat real
    # scenes).
    min_point_spread_ratio: float = 0.05


class DimSimVisualOdometryModule(Module):
    """RGB-D visual odometry over DimSim's color/depth feed.

    Publishes accumulated pose on odom_vo (NOT odom -- see module docstring
    for why ground truth is left untouched), plus periodic divergence
    logging against DimSim's real /odom so VO accuracy can be assessed
    before anything downstream depends on it.
    """

    config: DimSimVisualOdometryConfig

    odom_vo: Out[PoseStamped]

    # How often (seconds) to log VO-vs-ground-truth divergence. Independent
    # of frame-processing rate -- this is just a log throttle.
    _DIVERGENCE_LOG_INTERVAL_S = 2.0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._lcm: LCM | None = None
        self._camera_info = make_camera_info_default()
        self._intrinsic = None  # built lazily in start() -- needs open3d import
        self._cv2 = None  # built lazily in start() -- needs cv2 import, see start()

        self._latest_color: Image | None = None
        self._latest_depth: Image | None = None
        self._prev_rgbd = None  # open3d.geometry.RGBDImage, previous frame (tracking_method="dense")
        # Previous frame's raw color/depth arrays (tracking_method="sparse")
        # -- kept separately from _prev_rgbd since the sparse path never
        # touches Open3D at all, and building an unused RGBDImage every
        # frame would be wasted work.
        self._prev_color_np: np.ndarray | None = None
        self._prev_depth_m: np.ndarray | None = None
        self._prev_color_ts: float | None = None

        # Accumulated camera-frame pose as a 4x4 transform (open3d convention).
        self._accumulated_transform = np.eye(4, dtype=np.float64)

        # Ground-truth position/yaw at the last divergence-log tick, so
        # _maybe_log_divergence can report a per-interval DELTA (this
        # tick's motion) alongside the cumulative fields -- cumulative
        # error alone can't distinguish "frozen" from "genuinely
        # accurate." None until the first log tick has a prior sample to
        # diff against.
        self._last_log_gt_pos: tuple[float, float, float] | None = None
        self._last_log_gt_yaw: float | None = None
        self._last_log_vo_pos: tuple[float, float, float] | None = None
        self._last_log_vo_yaw: float | None = None

        # Ground truth from DimSim's own /odom, for divergence comparison only.
        self._gt_pos = (0.0, 0.0, 0.0)
        self._gt_yaw = 0.0
        self._last_divergence_log = 0.0

        # self._accumulated_transform always starts at identity, i.e. VO
        # always assumes it starts at world origin with zero heading --
        # but the drone's real spawn pose is whatever DimSim placed it at.
        # Without anchoring to the real starting pose, every published
        # odom_vo pose (and every divergence-log comparison) is offset by
        # that fixed, uninteresting distance regardless of how accurately
        # relative motion is tracked. Latched once from the first real
        # ground-truth sample in _on_ground_truth_odom below, then applied
        # in _camera_to_world_pose so published/compared poses reflect
        # absolute world position instead of "motion since VO's arbitrary
        # start".
        self._origin_pos: tuple[float, float, float] | None = None
        self._origin_yaw: float | None = None

        self._frame_count = 0
        self._vo_success_count = 0
        self._last_pairing_gap_s = 0.0

        # Per-interval breakdown of WHY tracking_method="sparse" rejected
        # a frame, reset every _DIVERGENCE_LOG_INTERVAL_S tick (see
        # _maybe_log_divergence) -- useful for telling which specific
        # check (too few matches, degenerate geometry, implausible speed)
        # is responsible for a low success rate, without needing to
        # enable full debug logging (too noisy at several frames/sec to
        # use live). Keys match _reject()'s reason strings below.
        self._reject_counts: dict[str, int] = {}

    @rpc
    def start(self) -> None:
        import open3d as o3d  # deferred import: modules run in forkserver
                               # worker processes, so pay this import cost
                               # after fork, not in the coordinator.
        self._o3d = o3d
        self._intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self._camera_info.width,
            height=self._camera_info.height,
            fx=self._camera_info.K[0],
            fy=self._camera_info.K[4],
            cx=self._camera_info.K[2],
            cy=self._camera_info.K[5],
        )
        self._odo_option = o3d.pipelines.odometry.OdometryOption()
        # OdometryOption's actual attributes are depth_min/depth_max, not
        # min_depth/max_depth -- easy to get wrong since
        # compute_rgbd_odometry()'s own parameter docs use different names.
        self._odo_option.depth_min = self.config.min_depth_m
        self._odo_option.depth_max = self.config.max_depth_m

        self._lcm = LCM()
        self._lcm.start()
        self._lcm.subscribe(Topic(_RAW_COLOR_TOPIC, Image), self._on_color_image)
        self._lcm.subscribe(Topic(_RAW_DEPTH_TOPIC, Image), self._on_depth_image)
        self._lcm.subscribe(Topic(_RAW_ODOM_TOPIC, PoseStamped), self._on_ground_truth_odom)
        logger.info("DimSimVisualOdometryModule started, subscribed to DimSim raw LCM topics")

    @rpc
    def stop(self) -> None:
        if self._lcm is not None:
            self._lcm.stop()
        super().stop()

    # ------------------------------------------------------------------
    # ground truth (comparison only -- never used to correct VO output)
    # ------------------------------------------------------------------

    def _on_ground_truth_odom(self, msg: PoseStamped, topic) -> None:
        p = msg.position
        self._gt_pos = (float(p.x), float(p.y), float(p.z))
        q = msg.orientation
        siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
        cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
        self._gt_yaw = math.atan2(siny_cosp, cosy_cosp)
        # Latch the drone's real starting pose once, from the first
        # ground-truth sample -- see __init__ comment on self._origin_pos.
        if self._origin_pos is None:
            self._origin_pos = self._gt_pos
            self._origin_yaw = self._gt_yaw
            # Publish an initial odom_vo message immediately, at this
            # exact origin-anchored pose, rather than waiting for the
            # first successful tracked transform. Without this, any
            # consumer that needs at least one odom message before acting
            # (e.g. a frontier explorer picking a goal) can never get
            # started if the robot's spawn view happens to make VO's own
            # tracking checks fail on every attempt -- since the robot
            # then never moves, the view never changes either, and
            # nothing breaks the cycle. Not fabricated information: at
            # this instant self._accumulated_transform is still identity
            # (no relative motion has been observed yet), so "VO's
            # estimate" and "the spawn pose" are the same thing by
            # construction. Only unblocks the bootstrap -- every
            # subsequent odom_vo update still goes through the normal
            # tracking/validation pipeline unweakened.
            self.odom_vo.publish(
                self._camera_to_world_pose(self._accumulated_transform, time.time())
            )

    # ------------------------------------------------------------------
    # sensor callbacks
    # ------------------------------------------------------------------

    def _on_depth_image(self, msg: Image, topic) -> None:
        self._latest_depth = msg

    def _on_color_image(self, msg: Image, topic) -> None:
        self._latest_color = msg
        # Pairing strategy: process on every color arrival using whatever
        # depth is most recently available. Mirrors DimSimCameraModule's
        # _maybe_publish_pointcloud precedent.
        if self._latest_depth is not None:
            try:
                self._process_frame(msg, self._latest_depth)
            except Exception:
                logger.exception(
                    "DimSimVisualOdometryModule: error processing frame %d",
                    self._frame_count,
                )

    # ------------------------------------------------------------------
    # core VO step
    # ------------------------------------------------------------------

    def _process_frame(self, color: Image, depth: Image) -> None:
        self._frame_count += 1
        self._last_pairing_gap_s = abs(color.ts - depth.ts)

        # color.data confirmed RGB/BGR uint8 HxWx3 from Image dataclass
        # (sensor_msgs/Image.py).
        color_np = color.to_rgb().data
        depth_np = depth.data  # raw uint16 mm -- deliberately NOT divided
                                # by 1000 here, see module docstring note 3.

        if self.config.tracking_method == "sparse":
            depth_m = depth_np.astype(np.float32) / 1000.0
            if self._prev_color_np is None:
                self._prev_color_np = color_np
                self._prev_depth_m = depth_m
                self._prev_color_ts = color.ts
                return
            dt = max(color.ts - self._prev_color_ts, 1e-3)  # guard against zero/negative dt
            success, trans = self._compute_transform_sparse(
                color_np, depth_m, self._prev_color_np, self._prev_depth_m, dt
            )
            self._prev_color_np = color_np
            self._prev_depth_m = depth_m
            self._prev_color_ts = color.ts
        else:
            success, trans = self._compute_transform_dense(color_np, depth_np)

        if not success:
            logger.debug(
                "DimSimVisualOdometryModule: odometry failed on frame %d "
                "(insufficient texture/overlap/matches), pose not updated this step",
                self._frame_count,
            )
        else:
            self._vo_success_count += 1
            # trans is the motion FROM target(previous frame) TO
            # source(current frame) in camera coordinates -- both
            # _compute_transform_dense (Open3D's own contract) and
            # _compute_transform_sparse (by construction, see its
            # docstring) return this same semantic, so accumulation works
            # identically regardless of tracking_method.
            self._accumulated_transform = self._accumulated_transform @ trans
            pose = self._camera_to_world_pose(self._accumulated_transform, color.ts)
            self.odom_vo.publish(pose)

        # Unconditional, not gated behind `success` -- at a low success
        # rate, a success-gated log call goes nearly silent right when its
        # diagnostic breakdown is needed most. Uses the current (possibly
        # stale, if nothing succeeded this interval) accumulated pose --
        # that staleness is itself the useful signal (delta: vo=0.000m
        # means VO genuinely didn't update, not that it's hiding
        # anything).
        self._maybe_log_divergence(
            self._camera_to_world_pose(self._accumulated_transform, color.ts)
        )

    def _compute_transform_dense(
        self, color_np: np.ndarray, depth_np: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        """Open3D's compute_rgbd_odometry, a dense photometric+geometric
        frame-to-frame optimizer. Assumes small motion between consecutive
        frames -- see module docstring for why this fails when
        pairing_gap is large. Kept for A/B comparison against
        tracking_method="sparse", not the default."""
        o3d = self._o3d
        o3d_color = o3d.geometry.Image(np.ascontiguousarray(color_np))
        o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth_np))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1000.0,          # matches DimSim's mm depth exactly
            depth_trunc=self.config.max_depth_m,
            convert_rgb_to_intensity=True,  # required by compute_rgbd_odometry
        )

        if self._prev_rgbd is None:
            self._prev_rgbd = rgbd
            return False, np.eye(4)

        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd,               # source = current frame
            self._prev_rgbd,    # target = previous frame
            self._intrinsic,
            np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            self._odo_option,
        )
        self._prev_rgbd = rgbd
        return success, trans

    def _compute_transform_sparse(
        self,
        color_np: np.ndarray,
        depth_m: np.ndarray,
        prev_color_np: np.ndarray,
        prev_depth_m: np.ndarray,
        dt: float,
    ) -> tuple[bool, np.ndarray]:
        """3D-3D rigid alignment via ORB feature matching + RANSAC,
        tolerant of large/irregular inter-frame gaps unlike
        _compute_transform_dense.

        Depth is available on BOTH frames (unlike a typical monocular VO
        setup), so each matched 2D keypoint pair unprojects to a genuine
        3D-3D correspondence -- not 2D-3D PnP. This makes the pose
        estimate a rigid-body alignment problem (Kabsch/Umeyama), solvable
        in closed form via SVD once outliers are rejected, rather than an
        iterative reprojection-error minimization -- and gives real,
        absolute-scale translation directly from the depth data, with no
        brightness-constancy or small-angle-linearization assumption.

        cv2.estimateAffine3D provides the RANSAC-based outlier rejection
        (OpenCV's Python bindings don't expose a rigid-only 3D-3D RANSAC
        directly, only general-affine) -- its returned affine fit is then
        DISCARDED except for which correspondences it flagged as inliers;
        the actual returned transform is re-fit via Kabsch/SVD on just
        those inliers, which enforces a true rotation (no residual
        scale/shear an affine fit could otherwise carry) since the real
        relative motion between two metric-depth camera views is rigid by
        construction.

        dt: real elapsed wall-clock time (seconds) between prev_color_np
        and color_np's capture, used only for the physical-plausibility
        check at the end (see max_linear_mps/max_angular_rps) -- not
        assumed small.

        Returns (success, trans) matching _compute_transform_dense's
        contract exactly (trans: motion FROM target(prev) TO
        source(current) in camera coordinates, ready to be
        right-multiplied into self._accumulated_transform)."""

        def _reject(reason: str) -> tuple[bool, np.ndarray]:
            self._reject_counts[reason] = self._reject_counts.get(reason, 0) + 1
            return False, np.eye(4)

        if self._cv2 is None:
            import cv2  # deferred import, same reasoning as open3d in
                        # start() -- only pay this cost when sparse
                        # tracking is actually selected/used.
            self._cv2 = cv2
        cv2 = self._cv2

        prev_gray = cv2.cvtColor(prev_color_np, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=self.config.orb_n_features)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        if (
            des1 is None
            or des2 is None
            or len(kp1) < self.config.ransac_min_inliers
            or len(kp2) < self.config.ransac_min_inliers
        ):
            return _reject("too_few_orb_features")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = bf.knnMatch(des1, des2, k=2)
        good = [
            m for m, n in (pair for pair in knn_matches if len(pair) == 2)
            if m.distance < self.config.orb_ratio_test_threshold * n.distance
        ]
        if len(good) < self.config.ransac_min_inliers:
            return _reject("too_few_orb_matches")

        fx, fy = self._camera_info.K[0], self._camera_info.K[4]
        cx, cy = self._camera_info.K[2], self._camera_info.K[5]
        min_d, max_d = self.config.min_depth_m, self.config.max_depth_m

        def unproject(depth_img: np.ndarray, u: float, v: float):
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= vi < depth_img.shape[0] and 0 <= ui < depth_img.shape[1]):
                return None
            d = float(depth_img[vi, ui])
            if not np.isfinite(d) or d <= min_d or d >= max_d:
                return None
            return ((u - cx) / fx * d, (v - cy) / fy * d, d)

        pts_prev, pts_curr = [], []
        for m in good:
            p1 = unproject(prev_depth_m, *kp1[m.queryIdx].pt)
            p2 = unproject(depth_m, *kp2[m.trainIdx].pt)
            if p1 is None or p2 is None:
                continue
            pts_prev.append(p1)
            pts_curr.append(p2)

        if len(pts_prev) < self.config.ransac_min_inliers:
            return _reject("too_few_unprojectable_matches")

        pts_prev_arr = np.array(pts_prev, dtype=np.float64)
        pts_curr_arr = np.array(pts_curr, dtype=np.float64)

        retval, _affine, inliers = cv2.estimateAffine3D(
            pts_prev_arr, pts_curr_arr,
            ransacThreshold=self.config.ransac_reproj_threshold_m,
            confidence=0.99,
        )
        if not retval or inliers is None:
            return _reject("ransac_failed")
        inlier_mask = inliers.ravel().astype(bool)
        if int(inlier_mask.sum()) < self.config.ransac_min_inliers:
            return _reject("too_few_ransac_inliers")

        # Kabsch/Umeyama closed-form rigid alignment on the RANSAC inlier
        # set: find R, t minimizing sum ||dst_i - (R @ src_i + t)||^2.
        src = pts_prev_arr[inlier_mask]
        dst = pts_curr_arr[inlier_mask]
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_c = src - src_mean
        dst_c = dst - dst_mean

        # A near-planar/near-collinear point set can pass every check
        # above (enough raw matches, enough RANSAC inliers) while still
        # being numerically ill-conditioned for a full 3D rigid fit -- see
        # min_point_spread_ratio's docstring. Check the singular values of
        # the (already-centered) inlier point cloud's spread directly --
        # three values, one per principal axis, largest to smallest. A
        # tiny smallest-to-largest ratio means the points are essentially
        # 2D (or worse, 1D), regardless of how many of them there are.
        spread = np.linalg.svd(src_c, compute_uv=False)
        if spread[0] < 1e-9 or (spread[-1] / spread[0]) < self.config.min_point_spread_ratio:
            logger.debug(
                "DimSimVisualOdometryModule: sparse transform rejected -- "
                "matched points are too close to planar/collinear "
                "(singular value ratio %.4f, need >=%.4f) for a "
                "well-conditioned rigid fit -- likely facing a flat "
                "surface head-on",
                spread[-1] / max(spread[0], 1e-9), self.config.min_point_spread_ratio,
            )
            return _reject("degenerate_geometry")

        H = src_c.T @ dst_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:  # reflection, not a rotation -- correct it
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = dst_mean - R @ src_mean

        trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3] = t

        # Final backstop: reject the transform outright if it implies an
        # average speed or turn-rate beyond what this simulated robot
        # could plausibly achieve, given dt (the real elapsed time between
        # the two frames). A RANSAC consensus can converge on a plausible-
        # looking but wrong transform in a real room's repetitive/
        # ambiguous geometry -- raising ransac_min_inliers alone isn't a
        # complete defense, since a wrong-but-internally-consistent
        # cluster of matches can still clear a higher inlier bar. Cheap to
        # compute (just the already-known R, t) and independent of how
        # many points agreed on the wrong answer.
        speed_mps = float(np.linalg.norm(t)) / dt
        angle_rad = math.acos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        rate_rps = angle_rad / dt
        if speed_mps > self.config.max_linear_mps or rate_rps > self.config.max_angular_rps:
            logger.debug(
                "DimSimVisualOdometryModule: sparse transform rejected as "
                "physically implausible (%.2fm/s, %.2frad/s over dt=%.3fs) -- "
                "likely a wrong RANSAC consensus on ambiguous/repetitive "
                "scene structure, not a real motion",
                speed_mps, rate_rps, dt,
            )
            return _reject("implausible_speed")

        return True, trans

    def _camera_to_world_pose(self, transform: np.ndarray, ts: float) -> PoseStamped:
        """Convert an accumulated Open3D camera-frame 4x4 transform into a
        DimOS-convention world-frame PoseStamped.

        Open3D/OpenCV camera convention: X=right, Y=down, Z=forward.
        DimOS/ROS world convention: X=forward, Y=left, Z=up.
        Axis remap: X_dimos = Z_cam, Y_dimos = -X_cam, Z_dimos = -Y_cam.
        """
        cam_x, cam_y, cam_z = transform[0, 3], transform[1, 3], transform[2, 3]
        world_x = cam_z
        world_y = -cam_x
        world_z = -cam_y

        # Extract yaw from the rotation submatrix via atan2 on the
        # remapped rotation axes -- sign confirmed via synthetic test
        # (hand-built rotation matrix representing a known pure world-yaw).
        r = transform[:3, :3]
        world_yaw = math.atan2(-r[0, 2], r[2, 2])

        # Anchor VO's start-relative motion onto the drone's real starting
        # pose -- see __init__ comment on self._origin_pos. Falls back to
        # no anchor (origin at world 0,0,0/yaw 0) if no ground-truth sample
        # has arrived yet.
        origin_x, origin_y, origin_z = self._origin_pos or (0.0, 0.0, 0.0)
        origin_yaw = self._origin_yaw or 0.0
        cos_o, sin_o = math.cos(origin_yaw), math.sin(origin_yaw)

        anchored_x = world_x * cos_o - world_y * sin_o + origin_x
        anchored_y = world_x * sin_o + world_y * cos_o + origin_y
        anchored_z = world_z + origin_z
        anchored_yaw = world_yaw + origin_yaw

        qz = math.sin(anchored_yaw / 2)
        qw = math.cos(anchored_yaw / 2)

        return PoseStamped(
            position=Vector3(anchored_x, anchored_y, anchored_z),
            orientation=Quaternion(0.0, 0.0, qz, qw),
            frame_id="world",
            ts=ts,
        )

    def _maybe_log_divergence(self, vo_pose: PoseStamped) -> None:
        now = time.time()
        if now - self._last_divergence_log < self._DIVERGENCE_LOG_INTERVAL_S:
            return
        self._last_divergence_log = now

        dx = vo_pose.position.x - self._gt_pos[0]
        dy = vo_pose.position.y - self._gt_pos[1]
        dz = vo_pose.position.z - self._gt_pos[2]
        dist_error = math.sqrt(dx * dx + dy * dy + dz * dz)

        vo_yaw = vo_pose.orientation.euler[2]
        yaw_error_deg = math.degrees(abs(math.atan2(
            math.sin(vo_yaw - self._gt_yaw), math.cos(vo_yaw - self._gt_yaw)
        )))

        success_rate = (
            self._vo_success_count / self._frame_count if self._frame_count else 0.0
        )

        # Per-interval DELTA (this window's motion, not cumulative-since-
        # start) alongside the cumulative fields above. Cumulative
        # position_error alone can't distinguish "VO is frozen at a
        # near-identity transform" from "VO is genuinely accurate," since
        # a frozen VO reading a robot that happens to return near its
        # start would ALSO show a small cumulative error despite never
        # having tracked anything. The ratio of VO's own reported
        # displacement to ground truth's real displacement over the same
        # window is a direct, per-tick check that VO is actually
        # responding to real motion.
        vo_pos = (vo_pose.position.x, vo_pose.position.y, vo_pose.position.z)
        pos_ratio_str = "n/a"
        yaw_ratio_str = "n/a"
        vo_delta_m = gt_delta_m = 0.0
        vo_yaw_delta_deg = gt_yaw_delta_deg = 0.0
        if self._last_log_gt_pos is not None:
            gdx = self._gt_pos[0] - self._last_log_gt_pos[0]
            gdy = self._gt_pos[1] - self._last_log_gt_pos[1]
            gdz = self._gt_pos[2] - self._last_log_gt_pos[2]
            gt_delta_m = math.sqrt(gdx * gdx + gdy * gdy + gdz * gdz)

            vdx = vo_pos[0] - self._last_log_vo_pos[0]
            vdy = vo_pos[1] - self._last_log_vo_pos[1]
            vdz = vo_pos[2] - self._last_log_vo_pos[2]
            vo_delta_m = math.sqrt(vdx * vdx + vdy * vdy + vdz * vdz)

            gt_yaw_delta_deg = math.degrees(abs(math.atan2(
                math.sin(self._gt_yaw - self._last_log_gt_yaw),
                math.cos(self._gt_yaw - self._last_log_gt_yaw),
            )))
            vo_yaw_delta_deg = math.degrees(abs(math.atan2(
                math.sin(vo_yaw - self._last_log_vo_yaw),
                math.cos(vo_yaw - self._last_log_vo_yaw),
            )))

            # Ratio undefined (not "0x") when ground truth barely moved --
            # avoid a division-by-near-zero producing a meaningless huge
            # or tiny number for an interval where there was no real
            # motion to track in the first place.
            if gt_delta_m > 0.02:
                pos_ratio_str = f"{vo_delta_m / gt_delta_m:.2f}x"
            if gt_yaw_delta_deg > 1.0:
                yaw_ratio_str = f"{vo_yaw_delta_deg / gt_yaw_delta_deg:.2f}x"

        self._last_log_gt_pos = self._gt_pos
        self._last_log_gt_yaw = self._gt_yaw
        self._last_log_vo_pos = vo_pos
        self._last_log_vo_yaw = vo_yaw

        # Per-interval rejection-reason breakdown (see _reject_counts'
        # docstring in __init__) -- reset every tick so this reports
        # "since last log", matching the delta fields above, not a
        # slowly-growing cumulative count that gets harder to read over a
        # long run.
        reject_str = (
            ", ".join(f"{k}={v}" for k, v in sorted(self._reject_counts.items()))
            if self._reject_counts else "none"
        )
        self._reject_counts = {}

        logger.info(
            "DimSimVisualOdometryModule: VO vs ground-truth -- "
            "position_error=%.3fm yaw_error=%.1fdeg success_rate=%.0f%% "
            "pairing_gap=%.0fms (frames=%d) | "
            "delta: vo=%.3fm/%.1fdeg gt=%.3fm/%.1fdeg ratio=%s/%s | "
            "rejected since last log: %s",
            dist_error, yaw_error_deg, success_rate * 100,
            self._last_pairing_gap_s * 1000, self._frame_count,
            vo_delta_m, vo_yaw_delta_deg, gt_delta_m, gt_yaw_delta_deg,
            pos_ratio_str, yaw_ratio_str, reject_str,
        )
