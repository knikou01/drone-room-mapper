"""
DimSimDepthLidarModule

Builds a world-frame PointCloud2 from DimSim's depth image and publishes it
so it can feed VoxelGridMapper INSTEAD of DimSim's own raycast-based
/lidar topic -- this is the actual "map with vision, not lidar" swap.

------------------------------------------------------------------------
WHY NOT NAME THE OUTPUT STREAM "lidar" (important, read before changing):
------------------------------------------------------------------------
DimSim's bridge (dimos-cli/bridge/server.ts) publishes its own raycast
point cloud unconditionally on the raw LCM topic "/lidar#sensor_msgs.
PointCloud2" -- this happens server-side regardless of what DimOS is doing.
VoxelGridMapper.lidar: In[PointCloud2] currently receives this by DimOS's
default name-to-topic resolution, with no DimOS module publishing it --
the bare declared stream name is enough to subscribe to that literal
topic string.

If this module ALSO declared Out[PointCloud2] named "lidar", DimOS's
transport machinery would derive the SAME topic string for publishing,
colliding with DimSim's own publisher -- both raycast points and
depth-derived points would land on one topic simultaneously, producing a
corrupted mixed map rather than a clean replacement.

Fix used: this module publishes on "depth_lidar" instead, and the
blueprint must use .remappings() to point VoxelGridMapper's "lidar" input
at this module's "depth_lidar" output instead of its default topic
resolution:
    .remappings([(VoxelGridMapper, "lidar", "depth_lidar")])

VERIFY ON FIRST RUN: check the startup "Transport" log lines for
VoxelGridMapper's lidar stream -- it should show a "depth_lidar" (or
remapped) topic, NOT the original "/lidar" topic. If VoxelGridMapper is
still receiving DimSim's raw raycast data instead, the remapping did not
take effect.
------------------------------------------------------------------------

Pose source: a standard DimOS `odom: In[PoseStamped]` stream. Ground truth
`/odom` resolves here by DEFAULT (DimOS's name-based topic resolution for
a stream literally named "odom" happens to match DimSim's raw
ground-truth topic string, same reasoning as
ReplanningAStarPlanner.odom/WavefrontFrontierExplorer.odom) -- but a
blueprint can remap this to `odom_vo` just like those, so the map itself
(not just navigation decisions) can be built from VO instead of ground
truth. The depth-image subscription below stays on the raw-LCM pattern
(a different, still-valid reason: DimSim's bridge publishes depth
unconditionally regardless of blueprint, matching DimSimCameraModule's
own color-image pattern).
"""
from __future__ import annotations

import math
import time

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

from sim_camera.dimsim_camera_module import make_camera_info_default

# Duplicated from dimsim_camera_module.py rather than imported -- see
# DimSimVisualOdometryModule for the same reasoning (those constants are
# underscore-prefixed/private, not a stable cross-module contract).
_RAW_DEPTH_TOPIC = "/depth_image"

logger = setup_logger()


class DimSimDepthLidarConfig(ModuleConfig):
    min_depth_m: float = 0.05
    max_depth_m: float = 15.0
    # Sample every Nth pixel in each dimension. DimSim's capture resolution
    # is 640x288; stride=4 gives 160x72 = 11520 candidate points per frame
    # before depth filtering -- comparable in density to DimSim's own
    # 20K-point raycast lidar without being excessive at ~2Hz.
    depth_sample_stride: int = 4


class DimSimDepthLidarModule(Module):
    """Publishes a world-frame PointCloud2 built from DimSim's depth image,
    intended to replace DimSim's raycast /lidar for VoxelGridMapper via
    blueprint-level .remappings() -- see module docstring for why the
    output is NOT named "lidar" directly.
    """

    config: DimSimDepthLidarConfig

    # Named "odom" (not e.g. "pose") specifically so it matches
    # ReplanningAStarPlanner.odom/WavefrontFrontierExplorer.odom's own
    # default DimOS topic resolution, and so the SAME .remappings() entry
    # style used for those (`(cls, "odom", "odom_vo")`) works here too.
    odom: In[PoseStamped]
    depth_lidar: Out[PointCloud2]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._lcm: LCM | None = None
        self._camera_info = make_camera_info_default()

        self._pos = (0.0, 0.0, 0.0)
        self._yaw = 0.0

        self._frame_count = 0

    @rpc
    def start(self) -> None:
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self._lcm = LCM()
        self._lcm.start()
        self._lcm.subscribe(Topic(_RAW_DEPTH_TOPIC, Image), self._on_depth_image)
        logger.info(
            "DimSimDepthLidarModule started -- publishing on 'depth_lidar', "
            "blueprint MUST remap VoxelGridMapper.lidar to consume it "
            "(see module docstring)"
        )

    @rpc
    def stop(self) -> None:
        if self._lcm is not None:
            self._lcm.stop()
        super().stop()

    def _on_odom(self, msg: PoseStamped) -> None:
        p = msg.position
        self._pos = (float(p.x), float(p.y), float(p.z))
        q = msg.orientation
        siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
        cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
        self._yaw = math.atan2(siny_cosp, cosy_cosp)

    def _on_depth_image(self, msg: Image, topic) -> None:
        self._frame_count += 1
        try:
            points = self._project_to_world(msg)
        except Exception:
            logger.exception(
                "DimSimDepthLidarModule: error projecting frame %d", self._frame_count
            )
            return

        if len(points) == 0:
            return

        cloud = PointCloud2.from_numpy(
            points.astype(np.float32), frame_id="world", timestamp=time.time()
        )
        self.depth_lidar.publish(cloud)

    def _project_to_world(self, depth_msg: Image) -> np.ndarray:
        """Depth image -> world-frame points. The camera-frame axis remap
        below is shared in spirit with DimSimVisualOdometryModule.
        _camera_to_world_pose; see test_depth_lidar_projection.py for the
        synthetic verification of this projection."""
        depth_mm = depth_msg.data
        h, w = depth_mm.shape[:2]
        depth_m = depth_mm.astype(np.float32) / 1000.0  # 16UC1 mm -> metres,
                                                          # same convention as
                                                          # DimSimCameraModule

        if self._camera_info.width != w or self._camera_info.height != h:
            from sim_camera.dimsim_camera_module import make_camera_info
            self._camera_info = make_camera_info(w, h)

        fx, fy = self._camera_info.K[0], self._camera_info.K[4]
        cx, cy = self._camera_info.K[2], self._camera_info.K[5]

        stride = self.config.depth_sample_stride
        rows = np.arange(0, h, stride, dtype=np.float32)
        cols = np.arange(0, w, stride, dtype=np.float32)
        uu, vv = np.meshgrid(cols, rows)
        dd = depth_m[::stride, ::stride]

        u = uu.ravel()
        v = vv.ravel()
        d = dd.ravel()

        mask = (
            (d > self.config.min_depth_m)
            & (d < self.config.max_depth_m)
            & np.isfinite(d)
        )
        u, v, d = u[mask], v[mask], d[mask]
        if len(d) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Camera-frame unprojection (OpenCV/Open3D convention, same as
        # DimSimCameraModule._maybe_publish_pointcloud's own unprojection).
        cam_x = (u - cx) / fx * d
        cam_y = (v - cy) / fy * d
        cam_z = d

        # Camera -> body remap (X=Z_cam, Y=-X_cam, Z=-Y_cam), same as
        # DimSimVisualOdometryModule._camera_to_world_pose.
        body_x = cam_z
        body_y = -cam_x
        body_z = -cam_y

        px, py, pz = self._pos
        yaw = self._yaw
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

        world_x = body_x * cos_yaw - body_y * sin_yaw + px
        world_y = body_x * sin_yaw + body_y * cos_yaw + py
        world_z = body_z + pz

        return np.stack([world_x, world_y, world_z], axis=1)
