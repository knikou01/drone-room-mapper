"""
DimOS Module that consumes DimSim's bridge LCM topics (raw, not via DimOS's
own Out/In stream machinery) and republishes dimsim_color_image / pointcloud
on proper DimOS streams — same shape as SimCameraModule, different source.

Why subscribe to raw LCM topics directly rather than declaring In[Image] etc.:
DimSim's bridge (dimos-cli/bridge/server.ts) publishes on the literal topics
"/color_image#sensor_msgs.Image", "/depth_image#sensor_msgs.Image",
"/lidar#sensor_msgs.PointCloud2" unconditionally, regardless of what DimOS
blueprint is running.

CONFIRMED BUG, FIXED (2026-07-08): this module's Out[Image] stream was
originally named "color_image", which DimOS's transport-building logic
(_get_transport_for in module_coordinator.py, `topic = f"/{name}"`) resolves
to the literal topic "/color_image" -- the EXACT same topic DimSim's bridge
publishes raw frames on. Subscribing to the raw topic as input does NOT
avoid a same-name Out stream from colliding with it on the output side; the
publish still goes out via DimOS's own transport onto that identical topic.
Confirmed via direct test that two independent same-process LCM() instances
DO see each other's publishes on a shared topic (no self-echo filtering) --
so this module's own raw `self._lcm` subscription received its own
republished frame, re-triggering _on_color_image, in an unbounded feedback
loop that flooded /color_image with duplicate stale frames. This starved out
real frames and was traced as the root cause of DimSimVisualOdometryModule's
VO pose reading as frozen (identical consecutive frames -> near-identity
odometry transform every time) and of the Rerun camera panel not showing a
live feed. Fixed by renaming the Out stream to "dimsim_color_image", which
resolves to a non-colliding topic. Downstream consumers (ObjectDBModule) are
wired via blueprint .remappings() back to this stream -- see
sim_dimsim_vo_blueprint.py / sim_detection_dimsim_blueprint.py.
Rerun's camera view does NOT need a remap: RerunBridgeModule subscribes to
the entire raw LCM bus directly (subscribe_all()), so it already picks up
DimSim's actual /color_image topic itself, independent of this module.

odom is NOT handled here. DimSim's bridge already publishes raw LCM
"/odom#geometry_msgs.PoseStamped", which is the exact same topic name DimOS's
own stream-name-based wiring already uses for `odom` elsewhere in any
blueprint. No relay needed — anything with `odom: In[PoseStamped]` already
receives it directly.

Prerequisite (verified working via lcm_probe/probe.py before this module was
written): the DimSim relay (`deno run --allow-all --unstable-net
dimos-cli/cli.ts dev --scene <name>`) must be running, with a browser tab open
at http://localhost:<port>/?dimos=1&scene=<name> and connected, BEFORE this
module starts.
"""
from __future__ import annotations

import math
import time

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo

import numpy as np

# Topic names match DimosBridge's CH_* constants in dimosBridge.ts exactly —
# confirmed working via lcm_probe/probe.py before this module was written.
_RAW_COLOR_TOPIC = "/color_image"
_RAW_DEPTH_TOPIC = "/depth_image"
_RAW_ODOM_TOPIC = "/odom"

# Throttle: only publish a TF + pointcloud cycle this often, even if odom
# arrives faster (DimSim's default odom rate is ~50Hz). Mirrors the fix
# found necessary for SimCameraModule — without this, per-cycle processing
# cost can fall behind the publish rate and the gap between "when this pose
# was generated" and "when it's actually used" grows without bound.
_TF_INTERVAL_S = 0.2

# Must match whatever --camera-fov was passed to `dimsim dev` (default 80,
# per dimos-cli/cli.ts's printed usage). If you ran the relay with a custom
# --camera-fov, update this to match, or the point cloud will be geometrically
# wrong even though images/depth still display correctly.
_FOV_DEG = 80.0

# DimSim's dimos-mode capture resolution is fixed in engine.js
# (_dimosCapW=640, _dimosCapH=288 — see the offscreen capture setup near the
# end of the dimosMode boot block). Used as the default for
# make_camera_info_default() so ObjectDBModule's blueprint construction has
# a real CameraInfo available immediately, rather than needing to wait for
# the first frame. _maybe_publish_pointcloud still rebuilds self._camera_info
# if an actual received frame's resolution ever differs from this constant
# (e.g. if DimSim's capture resolution changes in a future version).
_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 288


def make_camera_info(width: int, height: int, fov_deg: float = _FOV_DEG) -> CameraInfo:
    """CameraInfo matching DimSim's camera intrinsics for the given resolution."""
    return CameraInfo.from_fov(
        fov_deg=fov_deg,
        width=width,
        height=height,
        axis="horizontal",  # DimSim's --camera-fov is documented as horizontal-ish (Go2 87° horizontal reference in engine.js); verify against SimCameraModule's "vertical" convention if intrinsics look off
        frame_id="camera_optical",
    )


def make_camera_info_default() -> CameraInfo:
    """CameraInfo at DimSim's known fixed default capture resolution (640x288).

    Use this for ObjectDBModule.blueprint(camera_info=...) at blueprint
    construction time, since a real CameraInfo is required up front
    (module2D.py's Config.camera_info has no default) and the actual first
    frame hasn't arrived yet at that point. DimSimCameraModule itself still
    rebuilds its own internal _camera_info from real frame dimensions if
    they ever differ from this default — only ObjectDBModule's copy, used
    purely for camera_info.K-based 3D unprojection in pixel_to_3d, depends on
    this constant being right.
    """
    return make_camera_info(_DEFAULT_WIDTH, _DEFAULT_HEIGHT)


class DimSimCameraModule(Module):
    """Bridges DimSim's external LCM sensor feed into DimOS's module graph.

    NOT a drop-in alternative to SimCameraModule for color_image -- this
    module's image stream is deliberately named "dimsim_color_image" (not
    "color_image") to avoid colliding with DimSim's own raw /color_image
    topic (see module docstring). Consumers need a blueprint-level
    .remappings() entry to receive it under their expected "color_image"
    input name -- see sim_dimsim_vo_blueprint.py / sim_detection_dimsim_blueprint.py.
    pointcloud is unaffected (no raw DimSim topic of that name exists).
    """

    dimsim_color_image: Out[Image]
    pointcloud:  Out[PointCloud2]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._lcm: LCM | None = None
        self._latest_depth: Image | None = None
        self._camera_info: CameraInfo | None = None  # built lazily once we know real resolution
        self._pos = (0.0, 0.0, 0.0)
        self._yaw = 0.0
        self._last_tf_ts = 0.0

    def _on_color_image(self, msg: Image, topic) -> None:
        self.dimsim_color_image.publish(msg)
        self._maybe_publish_pointcloud(msg)

    def _on_depth_image(self, msg: Image, topic) -> None:
        self._latest_depth = msg

    def _on_odom(self, msg: PoseStamped, topic) -> None:
        p = msg.position
        self._pos = (float(p.x), float(p.y), float(p.z))
        q = msg.orientation
        # DimSim's server-side physics publishes a real quaternion (not
        # pure-yaw like SimDroneConnectionModule), so recover yaw properly
        # via atan2 of the full rotation rather than assuming qx=qz=0.
        siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
        cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
        self._yaw = math.atan2(siny_cosp, cosy_cosp)

        now = time.time()
        if now - self._last_tf_ts < _TF_INTERVAL_S:
            return  # throttle, same reasoning as SimCameraModule's fix
        self._last_tf_ts = now
        self._publish_camera_tf(now)

    def _publish_camera_tf(self, ts: float) -> None:
        """Publish world -> camera_optical (for Detection3DModule.tf.get) and
        world -> base_link (so rerun has a frame to attach a visible agent
        body marker to — see sim_detection_dimsim_blueprint.py's
        _static_drone_body, mirroring sim_blueprint.py's existing pattern for
        SimDroneConnectionModule, which this module never had until now).
        """
        qz = math.sin(self._yaw / 2)
        qw = math.cos(self._yaw / 2)
        base_link = Transform(
            translation=Vector3(self._pos[0], self._pos[1], self._pos[2]),
            rotation=Quaternion(0.0, 0.0, qz, qw),
            frame_id="world",
            child_frame_id="base_link",
            ts=ts,
        )
        self.tf.publish(base_link)

        camera_optical = Transform(
            translation=Vector3(self._pos[0], self._pos[1], self._pos[2]),
            rotation=Quaternion(0.0, 0.0, qz, qw),
            frame_id="world",
            child_frame_id="camera_optical",
            ts=ts,
        )
        self.tf.publish(camera_optical)

    def _maybe_publish_pointcloud(self, color: Image) -> None:
        """Build a camera-frame point cloud from the most recent depth frame
        paired with this color frame. Detection3DModule's own pipeline
        transforms camera_optical-frame points into world frame internally
        using the TF this module now publishes (see _publish_camera_tf) —
        the PointCloud2 published here is intentionally left in
        camera_optical frame, not pre-transformed into world frame, matching
        the frame_id contract Detection3DPC.from_2d expects
        (world_to_optical_transform is applied there, not here).
        """
        if self._latest_depth is None:
            return

        depth_m = self._latest_depth.data.astype(np.float32) / 1000.0  # 16UC1 mm -> meters
        height, width = depth_m.shape[:2]

        if self._camera_info is None or self._camera_info.width != width or self._camera_info.height != height:
            self._camera_info = make_camera_info(width, height)

        fx, fy = self._camera_info.K[0], self._camera_info.K[4]
        cx, cy = self._camera_info.K[2], self._camera_info.K[5]

        valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 15.0)
        v_idx, u_idx = np.where(valid)
        d = depth_m[v_idx, u_idx]

        x = (u_idx - cx) / fx * d
        y = (v_idx - cy) / fy * d
        z = d

        points = np.stack([x, y, z], axis=-1).astype(np.float32)
        if len(points) == 0:
            return

        pc = PointCloud2.from_numpy(points, frame_id="camera_optical", timestamp=color.ts)
        self.pointcloud.publish(pc)

    @rpc
    def start(self) -> None:
        self._lcm = LCM()
        self._lcm.start()
        self._lcm.subscribe(Topic(_RAW_COLOR_TOPIC, Image), self._on_color_image)
        self._lcm.subscribe(Topic(_RAW_DEPTH_TOPIC, Image), self._on_depth_image)
        self._lcm.subscribe(Topic(_RAW_ODOM_TOPIC, PoseStamped), self._on_odom)

    @rpc
    def stop(self) -> None:
        if self._lcm is not None:
            self._lcm.stop()
        super().stop()
