"""DroneDepthModule

Estimates metric depth from the drone's forward camera using Depth Anything
V2 (metric-indoor variant), projects the depth map into a world-frame
PointCloud2, and publishes it to VoxelGridMapper for live 3D mapping.

Replaces the earlier DroneDistanceSensorModule, which was removed because
RosettaDrone is not currently sending DISTANCE_SENSOR MAVLink messages.
See the technical report for the full explanation.

Model
-----
depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf (Apache 2.0)
HuggingFace transformers pipeline API:

    pipe = pipeline("depth-estimation", model=MODEL_NAME)
    result = pipe(pil_image)                 # input: PIL Image
    depth_tensor = result["predicted_depth"] # torch.Tensor, metres, resized
                                              # to match input image dims
    depth_np = depth_tensor.squeeze().cpu().numpy()

IMPORTANT: result["depth"] is NOT metric depth -- it's a 0-255 normalized
PIL image meant only for human visualization (confirmed from transformers'
depth_estimation.py postprocess(): it min-max normalizes predicted_depth
to [0,1] then scales to uint8 0-255 purely for display). Always read
result["predicted_depth"] for actual values. This was caught by
test_depth_model_standalone.py on a real test photo, which reported a
0-255 value range instead of plausible metres -- a clear signal the wrong
key was being read.

The model is downloaded once to ~/.cache/huggingface/ on first run, then
cached. Inference on CPU (no GPU required) takes roughly 1-4 seconds per
frame on a mid-range laptop, so the module is throttled to at most one
inference per `inference_interval_s` (default 3.0s).

Camera intrinsics
-----------------
DroneCameraModule is initialised with [fx=1000, fy=1000, cx=960, cy=540],
which correspond to a nominal 1920x1080 full resolution. The video stream
arrives at 640x360 (confirmed from logs: "640x360 RGB"). The module scales
intrinsics from the nominal resolution to the actual frame size at runtime,
so no manual adjustment is needed if the stream resolution ever changes.

    scale = frame_w / (cx_nominal * 2)   # cx=960 → nominal_w=1920
    fx_actual = fx_nominal * scale
    cx_actual = cx_nominal * scale

These are uncalibrated placeholders, not a proper lens calibration. For
better map quality, calibrate the actual DJI Mavic 2 Enterprise lens and
update the config values.

Projection
----------
Depth Anything returns depth along the optical axis (Z_cam, metres).
Camera frame: x=right, y=down, z=forward (standard pinhole convention).
DimOS body frame: x=forward, y=left, z=up.
odom.position.z is always 0 in this fork (confirmed, intentional); all
z-offsets use cruise_altitude_m.

Camera → body:
    X_body =  Z_cam  =  D
    Y_body = -X_cam  = -(u - cx) * D / fx
    Z_body = -Y_cam  = -(v - cy) * D / fy

Body → world (yaw θ, counterclockwise positive, from odom.orientation.euler[2]):
    X_world = X_body * cos(θ) - Y_body * sin(θ) + drone_x
    Y_world = X_body * sin(θ) + Y_body * cos(θ) + drone_y
    Z_world = Z_body + cruise_altitude_m

This assumes the camera faces horizontally forward. The Mavic 2 Enterprise
has a 3-axis gimbal; gimbal pitch angle is not currently available from
DimOS (no separate gimbal stream), so a pitch of 0 (level) is assumed.
If gimbal pitch data becomes available, apply an additional rotation matrix
before the yaw rotation.

Formula verified in test_sensor_projection.py (camera-frame version).

Stream wiring (all automatic via autoconnect)
----------------------------------------------
  In:  color_image  ← DroneCameraModule.color_image
  In:  odom         ← DroneConnectionModule.odom
  Out: lidar        → VoxelGridMapper.lidar
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
from PIL import Image as PILImage
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class DroneDepthConfig(ModuleConfig):
    # Depth range filter -- discard readings outside this window.
    max_depth_m: float = 8.0    # beyond this is often sky/ceiling noise
    min_depth_m: float = 0.3    # below this is near-clip noise

    # Sample every Nth pixel in each dimension. stride=8 on 640x360
    # gives 80x45 = 3600 candidate points per frame -- enough for the
    # voxel mapper without overwhelming it.
    depth_sample_stride: int = 8

    # Seconds between consecutive inference calls. Depth Anything Small
    # on CPU takes roughly 1-4s; 3.0s keeps the background thread from
    # piling up while still updating the map several times per minute.
    inference_interval_s: float = 3.0

    # Nominal camera intrinsics (for full resolution, from DroneCameraModule
    # blueprint: camera_intrinsics=[1000, 1000, 960, 540]).
    # Actual intrinsics are scaled to the live frame size at runtime.
    fx_nominal: float = 1000.0
    fy_nominal: float = 1000.0
    cx_nominal: float = 960.0   # implies nominal_width = 1920
    cy_nominal: float = 540.0   # implies nominal_height = 1080

    # Used for all Z projections because odom.position.z is always 0
    # in this fork (intentional upstream design decision).
    cruise_altitude_m: float = 1.5


class DroneDepthModule(Module):
    """Publishes a metric depth-derived PointCloud2 from the drone camera.

    Runs Depth Anything V2 (metric-indoor, Small) inference in a background
    thread at a low rate (~0.33 Hz), projects the depth map to world-frame
    3D points, and feeds VoxelGridMapper for passive live mapping.
    No flight commands are issued -- purely observational.
    """

    config: DroneDepthConfig

    color_image: In[Image]
    odom: In[PoseStamped]
    lidar: Out[PointCloud2]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest_image: Image | None = None
        self._latest_pose: PoseStamped | None = None
        self._pipe: Any = None   # transformers pipeline, built in start()
        self._running = False
        self._inference_thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()

        logger.info(
            "DroneDepthModule: loading %s -- "
            "this downloads ~100MB on first run, then uses the local cache",
            _MODEL_NAME,
        )
        # Import here (after fork) so the heavy torch/transformers import
        # doesn't happen in the coordinator process.
        from transformers import pipeline as hf_pipeline

        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=_MODEL_NAME,
        )
        logger.info("DroneDepthModule: model loaded, starting inference loop")

        self.register_disposable(
            Disposable(self.color_image.subscribe(self._on_image))
        )
        self.register_disposable(
            Disposable(self.odom.subscribe(self._on_odom))
        )

        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=10.0)
        super().stop()

    # ----------------------------------------------------------------
    # stream callbacks -- just store latest values, never block
    # ----------------------------------------------------------------

    def _on_image(self, img: Image) -> None:
        with self._lock:
            self._latest_image = img

    def _on_odom(self, pose: PoseStamped) -> None:
        with self._lock:
            self._latest_pose = pose

    # ----------------------------------------------------------------
    # background inference loop
    # ----------------------------------------------------------------

    def _inference_loop(self) -> None:
        tick = 0
        while self._running:
            time.sleep(self.config.inference_interval_s)
            if not self._running:
                break
            tick += 1
            with self._lock:
                img = self._latest_image
                pose = self._latest_pose
            if img is None or pose is None:
                logger.info(
                    "DroneDepthModule tick=%d: waiting for data -- "
                    "color_image=%s odom=%s",
                    tick,
                    "present" if img is not None else "MISSING",
                    "present" if pose is not None else "MISSING",
                )
                continue
            logger.info("DroneDepthModule tick=%d: running inference", tick)
            try:
                self._run_inference_and_publish(img, pose)
            except Exception:
                logger.exception("DroneDepthModule: inference error on tick %d", tick)

    def _run_inference_and_publish(self, img: Image, pose: PoseStamped) -> None:
        t0 = time.time()

        # Convert DimOS Image to PIL RGB for the transformers pipeline.
        # Image.to_rgb() handles BGR/BGRA/GRAY→RGB conversion internally.
        rgb_np = img.to_rgb().data
        pil_img = PILImage.fromarray(rgb_np)

        result = self._pipe(pil_img)
        # IMPORTANT: result["depth"] is a normalized 0-255 PIL image meant
        # only for human visualization (confirmed from transformers source,
        # postprocess() in depth_estimation.py). The actual metric depth
        # values in meters are in result["predicted_depth"], a torch.Tensor
        # already resized to match the input image dimensions via
        # post_process_depth_estimation(target_size=image.size[::-1]).
        # Caught by test_depth_model_standalone.py reporting a 0-255 range
        # instead of plausible metres on first real-image test.
        depth_tensor = result["predicted_depth"]
        depth_np = depth_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        inference_ms = (time.time() - t0) * 1000
        logger.debug(
            "DroneDepthModule: inference %.0fms, depth range [%.2f, %.2f]m",
            inference_ms,
            float(depth_np.min()),
            float(depth_np.max()),
        )

        points = self._project_to_world(depth_np, pose)
        if len(points) == 0:
            logger.debug("DroneDepthModule: no points after filtering")
            return

        cloud = PointCloud2.from_numpy(
            points.astype(np.float32),
            frame_id="world",
            timestamp=time.time(),
        )
        self.lidar.publish(cloud)
        logger.info(
            "DroneDepthModule: published %d points (inference %.0fms)",
            len(points),
            inference_ms,
        )

    def _project_to_world(
        self, depth_np: np.ndarray, pose: PoseStamped
    ) -> np.ndarray:
        """Project the depth map to world-frame (X, Y, Z) points.

        See module docstring for the full derivation. Uses numpy vectorised
        ops -- no Python loops over pixels.
        """
        h, w = depth_np.shape
        cfg = self.config

        # Scale nominal intrinsics to actual frame resolution at runtime.
        # cx_nominal * 2 = nominal full width (e.g. 960*2 = 1920).
        scale_x = w / (cfg.cx_nominal * 2.0)
        scale_y = h / (cfg.cy_nominal * 2.0)
        fx = cfg.fx_nominal * scale_x
        fy = cfg.fy_nominal * scale_y
        cx = cfg.cx_nominal * scale_x
        cy = cfg.cy_nominal * scale_y

        stride = cfg.depth_sample_stride
        rows = np.arange(0, h, stride, dtype=np.float32)
        cols = np.arange(0, w, stride, dtype=np.float32)
        uu, vv = np.meshgrid(cols, rows)          # (H', W') each
        dd = depth_np[::stride, ::stride]          # (H', W')

        u = uu.ravel()
        v = vv.ravel()
        d = dd.ravel()

        mask = (d > cfg.min_depth_m) & (d < cfg.max_depth_m) & np.isfinite(d)
        u, v, d = u[mask], v[mask], d[mask]

        if len(d) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Camera frame → body frame
        x_body =  d                        # Z_cam → body forward
        y_body = -(u - cx) * d / fx        # -X_cam → body left
        z_body = -(v - cy) * d / fy        # -Y_cam → body up

        # Body frame → world frame (yaw rotation + drone translation)
        drone_x = float(pose.position.x)
        drone_y = float(pose.position.y)
        pz = cfg.cruise_altitude_m         # odom.z is always 0 in this fork
        yaw = float(pose.orientation.euler[2])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        x_world = x_body * cos_yaw - y_body * sin_yaw + drone_x
        y_world = x_body * sin_yaw + y_body * cos_yaw + drone_y
        z_world = z_body + pz

        return np.stack([x_world, y_world, z_world], axis=1)
