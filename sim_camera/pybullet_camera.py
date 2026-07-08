from __future__ import annotations

import math

import numpy as np
import pybullet as p


class DroneCamera:
    def __init__(
        self,
        client_id: int,
        width: int = 640,
        height: int = 480,
        fov_deg: float = 70.0,
        near: float = 0.05,
        far: float = 20.0,
    ) -> None:
        self.client_id = client_id
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.near = near
        self.far = far
        self.aspect = width / height
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov_deg, aspect=self.aspect, nearVal=near, farVal=far
        )

    def capture(
        self,
        pos: tuple[float, float, float],
        yaw: float,
        pitch: float = -0.15,  # slight downward tilt, drone-like
    ) -> dict:
        """Capture RGB + linear-depth from the given world pose.

        Args:
            pos: (x, y, z) world position
            yaw: heading in radians (matches SimDroneConnectionModule._heading)
            pitch: camera pitch in radians, negative = nose-down

        Returns {"rgb": (h,w,3) uint8 RGB, "depth_m": (h,w) float32 meters}
        """
        forward = (
            math.cos(pitch) * math.sin(yaw),
            math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
        )
        target = (pos[0] + forward[0], pos[1] + forward[1], pos[2] + forward[2])
        view = p.computeViewMatrix(
            cameraEyePosition=list(pos),
            cameraTargetPosition=list(target),
            cameraUpVector=[0, 0, 1],
        )
        _, _, rgba, depth_buf, _ = p.getCameraImage(
            self.width, self.height,
            viewMatrix=view,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_TINY_RENDERER,  # headless-safe; swap to ER_BULLET_HARDWARE_OPENGL if you have a display
            physicsClientId=self.client_id,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        depth_buf = np.array(depth_buf, dtype=np.float32).reshape(self.height, self.width)
        depth_m = self._depth_to_meters(depth_buf)
        return {"rgb": rgb, "depth_m": depth_m}

    def _depth_to_meters(self, buf: np.ndarray) -> np.ndarray:
        """Standard OpenGL nonlinear depth buffer -> linear meters."""
        return self.far * self.near / (self.far - (self.far - self.near) * buf)

    def depth_to_pointcloud_world(
        self,
        depth_m: np.ndarray,
        pos: tuple[float, float, float],
        yaw: float,
        pitch: float = -0.15,
        max_depth: float = 10.0,
        min_depth: float | None = None,
    ) -> np.ndarray:
        """Back-project depth pixels into world-frame XYZ points (Nx3 float32).

        Uses the same camera intrinsics as capture(). Points beyond max_depth,
        at/below the camera's near plane, or non-finite are dropped — these
        are the degenerate values that otherwise show up as a flat "wall" of
        misplaced points and spurious detections.
        """
        if min_depth is None:
            min_depth = self.near * 1.5  # stay comfortably clear of the near-plane boundary

        # fy from vertical FOV directly; fx derived via aspect ratio on the
        # focal length itself (standard pinhole convention) — NOT by scaling
        # the angle before tan(), which was the previous (incorrect) formula
        # and skewed every x-coordinate by a width/height-dependent factor.
        fy = (self.height / 2) / math.tan(math.radians(self.fov_deg / 2))
        fx = fy * self.aspect
        cx, cy = self.width / 2.0, self.height / 2.0

        valid = np.isfinite(depth_m) & (depth_m > min_depth) & (depth_m < max_depth)
        v_idx, u_idx = np.where(valid)
        d = depth_m[v_idx, u_idx]

        # Camera-space (x right, y up, z back in OpenGL; we use x right, y forward, z up)
        x_cam = (u_idx - cx) / fx * d
        y_cam = -(v_idx - cy) / fy * d  # flip v: image y down, camera y up
        z_cam = -d                        # OpenGL z points behind camera

        # Rotate camera-space -> world-space using yaw + pitch
        # Camera looks along -Z_cam, which maps to +Y_world at yaw=0
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)

        # Build rotation: first pitch around camera-X, then yaw around world-Z
        # Forward in camera space (-Z_cam) maps to world (sin_y, cos_y, 0) at zero pitch
        x_world = (cos_y * x_cam - sin_y * (cos_p * z_cam - sin_p * y_cam))
        y_world = (sin_y * x_cam + cos_y * (cos_p * z_cam - sin_p * y_cam))
        z_world = sin_p * z_cam + cos_p * y_cam

        points = np.stack([
            x_world + pos[0],
            y_world + pos[1],
            z_world + pos[2],
        ], axis=-1).astype(np.float32)

        return points
