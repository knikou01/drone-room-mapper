"""
DimSimVisualOdometryModule

Phase 1 of visual-SLAM-instead-of-lidar work (see Vector's DimSim technical
report, section 4). Computes RGB-D visual odometry from DimSim's color/depth
LCM feed using Open3D's compute_rgbd_odometry, and publishes the accumulated
pose on a SEPARATE topic (odom_vo) rather than replacing DimSim's own
ground-truth /odom. This lets VO accuracy be measured against ground truth
before it's trusted for navigation -- exactly the caution Vector's report
recommends in its "Important caveat" (section 4.2).

Pattern reused directly from sim_camera/dimsim_camera_module.py (Vector's
confirmed-working module): subscribe to DimSim's raw LCM topics via a
standalone LCM() instance rather than declaring In[Image] streams, since
DimSim's bridge publishes unconditionally on fixed topic names regardless
of which DimOS blueprint is running. Reuses make_camera_info_default() and
the depth-in-mm convention from that same module rather than re-deriving
either.

Requires open3d (already a dependency -- VoxelGridMapper uses it).

------------------------------------------------------------------------
KNOWN LIMITATION, UNRESOLVED (2026-07-08) -- odom_vo IS NOT ACCURATE
ENOUGH TO TRUST FOR NAVIGATION. DO NOT swap any consumer's pose input from
ground-truth /odom to odom_vo (that would be Phase 3) until this is
addressed. Confirmed, not guessed -- reproduced twice, including once in a
stripped-down blueprint (sim_dimsim_vo_only_blueprint.py) with every
non-essential module removed specifically to rule out system load as the
cause:
------------------------------------------------------------------------
Over a run with several real metres of ground-truth translation and 120+
degrees of real ground-truth yaw change, the published odom_vo pose barely
moved at all (~0.1-0.2m total, yaw pinned within a ~4 degree band) while
_vo_success_count / _frame_count (compute_rgbd_odometry's own "success"
flag) stayed at 99%. The coordinate-conversion math itself is verified
correct by direct synthetic test (see _camera_to_world_pose's docstring)
-- this is not the same class of bug as the yaw-sign or origin-anchor
issues fixed below. The leading hypothesis, grounded in what's directly
observable rather than assumed: pairing_gap (see _last_pairing_gap_s)
consistently runs 1000-1200ms even with system load minimized, meaning
consecutive color/depth frames are roughly a full second apart. Dense
frame-to-frame RGBD odometry generally assumes small motion between
consecutive frames; fed a large real gap, it's a known failure mode for
the optimizer to converge to a near-identity transform while still
reporting success=True, rather than actually failing loudly. This has NOT
been confirmed as the definitive root cause (that would need e.g. forcing
a faster capture rate or switching odometry approaches and re-measuring)
-- it's the best-supported explanation so far, flagged as such rather than
asserted as fact.

Current decision (per user, 2026-07-08): proceed using ground-truth /odom
for Phase 2 mapping and beyond for now; treat improving/replacing VO's
odometry approach as a deferred, separate task rather than a blocker.

------------------------------------------------------------------------
OPEN QUESTIONS, RESOLVED:
------------------------------------------------------------------------
1. Color/depth pairing gap: NOT small -- see KNOWN LIMITATION above.
   Logged via pairing_gap in every divergence line.

2. Coordinate frame conversion: verified correct via synthetic test (see
   _camera_to_world_pose). A real sign bug was found and fixed in the yaw
   extraction; translation extraction was already correct.

3. Depth scale: RGBDImage.create_from_color_and_depth defaults to
   depth_scale=1000.0, which matches DimSim's raw uint16-millimeter depth
   format exactly (confirmed from DimSimCameraModule's own
   `depth_m = self._latest_depth.data.astype(np.float32) / 1000.0` line) --
   so raw depth data is passed to Open3D WITHOUT pre-dividing by 1000, to
   let Open3D's own depth_scale handle it consistently with its internal
   pyramid/gradient computations. This is different from
   DimSimCameraModule's own point-cloud code, which does divide by 1000
   itself for a different purpose (manual unprojection). Do not copy that
   division pattern into this module's RGBDImage construction.
------------------------------------------------------------------------
"""
from __future__ import annotations

import math
import time

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module
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
# (private) -- not a contract Vector has committed to keeping stable for
# external imports. If DimSim's bridge topic names ever change, both this
# file and dimsim_camera_module.py need updating together.
_RAW_COLOR_TOPIC = "/color_image"
_RAW_DEPTH_TOPIC = "/depth_image"
_RAW_ODOM_TOPIC = "/odom"

logger = setup_logger()


class DimSimVisualOdometryModule(Module):
    """RGB-D visual odometry over DimSim's color/depth feed.

    Publishes accumulated pose on odom_vo (NOT odom -- see module docstring
    for why ground truth is left untouched during this validation phase),
    plus periodic divergence logging against DimSim's real /odom so VO
    accuracy can be assessed before anything downstream depends on it.
    """

    odom_vo: Out[PoseStamped]

    # How often (seconds) to log VO-vs-ground-truth divergence. Independent
    # of frame-processing rate -- this is just a log throttle.
    _DIVERGENCE_LOG_INTERVAL_S = 2.0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._lcm: LCM | None = None
        self._camera_info = make_camera_info_default()
        self._intrinsic = None  # built lazily in start() -- needs open3d import

        self._latest_color: Image | None = None
        self._latest_depth: Image | None = None
        self._prev_rgbd = None  # open3d.geometry.RGBDImage, previous frame

        # Accumulated camera-frame pose as a 4x4 transform (open3d convention).
        self._accumulated_transform = np.eye(4, dtype=np.float64)

        # Ground truth from DimSim's own /odom, for divergence comparison only.
        self._gt_pos = (0.0, 0.0, 0.0)
        self._gt_yaw = 0.0
        self._last_divergence_log = 0.0

        # CONFIRMED BUG, FIXED (2026-07-08): self._accumulated_transform
        # always starts at identity, i.e. VO always assumes it starts at
        # world origin with zero heading -- but the drone's real spawn
        # pose is whatever DimSim placed it at (e.g. (3,2,0.54) observed
        # in one run). Without anchoring to the real starting pose, every
        # published odom_vo pose (and every divergence-log comparison
        # against ground truth) is offset by that fixed, uninteresting
        # distance regardless of how accurately relative motion is
        # tracked -- confirmed directly: a run where the robot never
        # moved at all (stuck against a wall) still reported
        # position_error=3.646m, which is exactly
        # sqrt(3**2 + 2**2 + 0.54**2), i.e. purely the unanchored offset,
        # not a tracking error. Latched once from the first real
        # ground-truth sample in _on_ground_truth_odom below, then applied
        # in _camera_to_world_pose so published/compared poses reflect
        # absolute world position instead of "motion since VO's arbitrary
        # start".
        self._origin_pos: tuple[float, float, float] | None = None
        self._origin_yaw: float | None = None

        self._frame_count = 0
        self._vo_success_count = 0
        self._last_pairing_gap_s = 0.0

    @rpc
    def start(self) -> None:
        import open3d as o3d  # deferred import, same reasoning as
                               # DroneDepthModule's transformers import:
                               # avoid the heavy import cost in the
                               # coordinator process, only pay it in the
                               # worker after fork.
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
        # CORRECTED: the actual OdometryOption attributes are depth_min/
        # depth_max, not min_depth/max_depth (confirmed from the object's
        # own repr: OdometryOption(iteration_number_per_pyramid_level=...,
        # depth_diff_max=0.03, depth_min=0, depth_max=4)). The first attempt
        # used min_depth/max_depth by mistake, conflating the
        # compute_rgbd_odometry() function's descriptive parameter docs
        # with OdometryOption's real field names -- caught via AttributeError
        # on first real run. Keep consistent with DimSimCameraModule's own
        # valid-depth window (0.05m-15.0m) so results are comparable, though
        # these fields specifically gate the odometry correspondence search,
        # not point-cloud filtering.
        self._odo_option.depth_min = 0.05
        self._odo_option.depth_max = 15.0

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

    # ------------------------------------------------------------------
    # sensor callbacks
    # ------------------------------------------------------------------

    def _on_depth_image(self, msg: Image, topic) -> None:
        self._latest_depth = msg

    def _on_color_image(self, msg: Image, topic) -> None:
        self._latest_color = msg
        # Pairing strategy: process on every color arrival using whatever
        # depth is most recently available. Mirrors DimSimCameraModule's
        # _maybe_publish_pointcloud precedent (see open question #1 above
        # regarding pairing quality).
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
        o3d = self._o3d
        self._frame_count += 1

        # Open question #1 (module docstring): are color/depth actually
        # paired closely in time, or could stale pairing be degrading VO
        # accuracy? Track it so _maybe_log_divergence can report it
        # alongside pose error instead of leaving this unanswered.
        self._last_pairing_gap_s = abs(color.ts - depth.ts)

        # color.data confirmed RGB/BGR uint8 HxWx3 from Image dataclass
        # (sensor_msgs/Image.py). Open3D's o3d.geometry.Image wraps a numpy
        # array directly -- no copy needed for correctness, contiguity is
        # handled internally by Open3D's pybind layer.
        color_np = color.to_rgb().data
        depth_np = depth.data  # raw uint16 mm, per open question #3 above --
                                # deliberately NOT divided by 1000 here.

        o3d_color = o3d.geometry.Image(np.ascontiguousarray(color_np))
        o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth_np))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1000.0,          # matches DimSim's mm depth exactly
            depth_trunc=15.0,            # metres, matches _odo_option.depth_max
            convert_rgb_to_intensity=True,  # required by compute_rgbd_odometry
        )

        if self._prev_rgbd is None:
            self._prev_rgbd = rgbd
            return

        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd,               # source = current frame
            self._prev_rgbd,    # target = previous frame
            self._intrinsic,
            np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            self._odo_option,
        )
        self._prev_rgbd = rgbd

        if not success:
            logger.debug(
                "DimSimVisualOdometryModule: odometry failed on frame %d "
                "(insufficient texture/overlap), pose not updated this step",
                self._frame_count,
            )
            return

        self._vo_success_count += 1
        # trans is the motion FROM target TO source in camera coordinates.
        # Accumulate into world-frame estimate.
        self._accumulated_transform = self._accumulated_transform @ trans

        pose = self._camera_to_world_pose(self._accumulated_transform, color.ts)
        self.odom_vo.publish(pose)

        self._maybe_log_divergence(pose)

    def _camera_to_world_pose(self, transform: np.ndarray, ts: float) -> PoseStamped:
        """Convert an accumulated Open3D camera-frame 4x4 transform into a
        DimOS-convention world-frame PoseStamped.

        Open3D/OpenCV camera convention: X=right, Y=down, Z=forward.
        DimOS/ROS world convention (confirmed from global_planner.py,
        mavlink_connection.py): X=forward, Y=left, Z=up.

        Axis remap: X_dimos = Z_cam, Y_dimos = -X_cam, Z_dimos = -Y_cam.

        CONFIRMED BUG, FIXED (2026-07-08): verified against ground-truth
        divergence logging exactly as flagged below -- a live run showed
        gt_yaw swinging up to ~0.7rad while the published vo yaw stayed
        under ~0.04rad, a ~17x underestimate. Root-caused with a synthetic
        test (not the live run itself, which only proved *something* was
        wrong): fed _camera_to_world_pose a hand-built camera-frame
        rotation matrix representing a KNOWN pure world-yaw of theta, for
        theta in +-{10,30,45,90,135} degrees. The translation extraction
        recovered the expected value correctly in every case. The yaw
        extraction recovered exactly -theta every time -- correct
        magnitude, flipped sign. Fixed by negating r[0, 2] below.
        (The residual under-tracking possibility -- that
        self._accumulated_transform's own composition order/convention may
        not fully match Open3D's compute_rgbd_odometry semantics -- was
        NOT ruled out by this test, since the test only exercises the
        final-matrix-to-Euler extraction step in isolation, not the
        multi-frame accumulation itself. Re-check divergence numbers after
        this fix before concluding VO is accurate enough for Phase 3.)
        """
        cam_x, cam_y, cam_z = transform[0, 3], transform[1, 3], transform[2, 3]
        world_x = cam_z
        world_y = -cam_x
        world_z = -cam_y

        # Extract yaw from the rotation submatrix. Camera-frame yaw (rotation
        # about camera Y / dimos -Z... ) -- approximated here via the
        # standard atan2 decomposition of the remapped rotation. Flagged
        # as unvalidated along with the translation remap above.
        r = transform[:3, :3]
        # Remap rotation matrix axes the same way as translation, then
        # extract yaw (rotation about world Z) via atan2. Sign confirmed
        # via synthetic test -- see docstring above.
        world_yaw = math.atan2(-r[0, 2], r[2, 2])

        # Anchor VO's start-relative motion onto the drone's real starting
        # pose -- see __init__ comment on self._origin_pos. Falls back to
        # no anchor (origin at world 0,0,0/yaw 0) if no ground-truth sample
        # has arrived yet, matching the old (buggy) behavior only for that
        # brief startup window.
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
        logger.info(
            "DimSimVisualOdometryModule: VO vs ground-truth -- "
            "position_error=%.3fm yaw_error=%.1fdeg success_rate=%.0f%% "
            "pairing_gap=%.0fms (frames=%d)",
            dist_error, yaw_error_deg, success_rate * 100,
            self._last_pairing_gap_s * 1000, self._frame_count,
        )
