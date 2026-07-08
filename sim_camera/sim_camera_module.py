"""
DimOS Module that wraps PyBullet camera capture and publishes:
  color_image: Out[Image]       — RGB frame from drone POV (matches ObjectDBModule.color_image)
  pointcloud:  Out[PointCloud2] — depth-derived world-frame cloud (matches ObjectDBModule.pointcloud)

Reads drone pose from odom: In[PoseStamped], which SimDroneConnectionModule
already publishes every tick — no changes to existing code needed.

Stream name rationale:
  - `color_image` matches ObjectDBModule/Detection2DModule's In[Image] directly,
    so autoconnect wires them with no remapping needed.
  - `pointcloud` matches ObjectDBModule's In[PointCloud2] directly.
  - We deliberately avoid `video` (already used by SimDroneConnectionModule)
    to prevent two Out[Image] streams sharing the same transport topic.
"""
from __future__ import annotations

import math
import time

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

from rooms.models import Room
from sim_camera.pybullet_camera import DroneCamera
from sim_camera.pybullet_room import build_room, connect_headless

_WIDTH = 640
_HEIGHT = 480
_FOV_DEG = 70.0
_NEAR = 0.05
_FAR = 20.0
_MAX_DEPTH_M = 10.0
_CAPTURE_INTERVAL_S = 0.2  # cap capture rate (~5Hz) so it can't fall behind odom's faster tick rate


def make_camera_info() -> CameraInfo:
    """CameraInfo matching DroneCamera's intrinsics.

    Exposed as a module-level function so the blueprint can pass it to
    ObjectDBModule(camera_info=make_camera_info()) without instantiating
    SimCameraModule first.
    """
    return CameraInfo.from_fov(
        fov_deg=_FOV_DEG,
        width=_WIDTH,
        height=_HEIGHT,
        axis="vertical",
        frame_id="camera_optical",
    )


class SimCameraModule(Module):
    odom:        In[PoseStamped]

    color_image: Out[Image]
    pointcloud:  Out[PointCloud2]

    def __init__(self, room: Room, **kwargs) -> None:
        super().__init__(**kwargs)
        self._room = room
        self._client_id: int | None = None
        self._cam: DroneCamera | None = None
        self._pos = (room.width / 2, room.depth / 2, 1.5)
        self._yaw = 0.0
        self._last_capture_ts = 0.0

    def _setup_pybullet(self) -> None:
        self._client_id = connect_headless()
        build_room(self._room, self._client_id)
        self._cam = DroneCamera(
            self._client_id,
            width=_WIDTH,
            height=_HEIGHT,
            fov_deg=_FOV_DEG,
            near=_NEAR,
            far=_FAR,
        )

    def _on_odom(self, pose: PoseStamped) -> None:
        p = pose.position
        self._pos = (float(p.x), float(p.y), float(p.z))
        # SimDroneConnectionModule only sets qz/qw (pure yaw), so:
        q = pose.orientation
        self._yaw = 2.0 * math.atan2(float(q.z), float(q.w))

        now = time.time()
        if now - self._last_capture_ts < _CAPTURE_INTERVAL_S:
            return  # throttle: odom arrives faster than we need new frames;
            # without this, capture()'s ~0.2s cost falls behind odom's faster
            # tick rate and the backlog grows every frame (see debugging notes)
        self._last_capture_ts = now

        self._publish_camera_tf(now)
        self._capture(now)

    def _publish_camera_tf(self, ts: float) -> None:
        """Publish world -> camera_optical so Detection3DModule.tf.get(...) succeeds.

        Detection3DModule looks up tf.get("camera_optical", pointcloud.frame_id, ts, 5.0)
        every frame; without this publish that lookup returns None and detections
        are silently dropped (visible as "No direct transform found" warnings).

        Parented directly to "world" using the same pos/yaw already tracked from
        odom, rather than chaining through base_link/camera_link, since we have
        the world pose directly and don't need the intermediate frames.
        """
        qz = math.sin(self._yaw / 2)
        qw = math.cos(self._yaw / 2)
        camera_optical = Transform(
            translation=Vector3(self._pos[0], self._pos[1], self._pos[2]),
            rotation=Quaternion(0.0, 0.0, qz, qw),
            frame_id="world",
            child_frame_id="camera_optical",
            ts=ts,
        )
        self.tf.publish(camera_optical)

    def _capture(self, ts: float) -> None:
        if self._cam is None:
            return
        frame = self._cam.capture(self._pos, self._yaw)

        img = Image.from_numpy(
            frame["rgb"], format=ImageFormat.RGB,
            frame_id="camera_optical", ts=ts,
        )
        self.color_image.publish(img)

        points = self._cam.depth_to_pointcloud_world(
            frame["depth_m"], self._pos, self._yaw, max_depth=_MAX_DEPTH_M,
        )
        if len(points) > 0:
            pc = PointCloud2.from_numpy(points, frame_id="world", timestamp=ts)
            self.pointcloud.publish(pc)

    @rpc
    def start(self) -> None:
        self._setup_pybullet()
        self.odom.subscribe(self._on_odom)

    @rpc
    def stop(self) -> None:
        import pybullet as p
        if self._client_id is not None:
            try:
                p.disconnect(self._client_id)
            except Exception:
                pass
        super().stop()
