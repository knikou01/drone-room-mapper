import math
import json
import threading
import time
from typing import Any

import numpy as np
import open3d as o3d

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.utils.logging_config import setup_logger

from rooms.models import Room, Wall, Obstacle

logger = setup_logger()

LIDAR_RAY_STEP_DEG = 1.0
LIDAR_VERTICAL_ANGLES_DEG = [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
LIDAR_MAX_RANGE = 8.0
DRONE_SPEED = 1.0
CRUISE_ALTITUDE = 1.5


class SimDroneConnectionModule(Module):

    cmd_vel:   In[Twist]

    odom:      Out[PoseStamped]
    lidar:     Out[PointCloud2]
    video:     Out[Image]
    telemetry: Out[Any]

    def __init__(self, room: Room, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._room = room
        self._position = {
            "x":    room.width / 2,
            "y":    room.depth / 2,
            "z":    CRUISE_ALTITUDE,
            "yaw":  0.0
        }
        self._velocity = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self._velocity_lock = threading.Lock()
        self._odom: PoseStamped | None = None
        self._running = False
        self._movement_thread: threading.Thread | None = None
        self._heading = 0.0

    @rpc
    def start(self) -> None:
        if self.cmd_vel.transport:
            self.cmd_vel.subscribe(self._on_cmd_vel)

        self._running = True
        self._movement_thread = threading.Thread(
            target=self._movement_loop,
            daemon=True,
        )
        self._movement_thread.start()
        logger.info("SimDroneConnectionModule started")

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._movement_thread and self._movement_thread.is_alive():
            self._movement_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        logger.info("SimDroneConnectionModule stopped")
        super().stop()

    def _on_cmd_vel(self, twist: Twist) -> None:
        with self._velocity_lock:
            cos_h = math.cos(self._heading)
            sin_h = math.sin(self._heading)
            vx_body = twist.linear.x * DRONE_SPEED
            vy_body = twist.linear.y * DRONE_SPEED
            self._velocity["x"] = vx_body * cos_h - vy_body * sin_h
            self._velocity["y"] = vx_body * sin_h + vy_body * cos_h
            self._velocity["z"] = twist.linear.z * DRONE_SPEED
            self._velocity["yaw"] = twist.angular.z
        logger.info(f"cmd_vel received: vx={twist.linear.x:.2f} vy={twist.linear.y:.2f} wz={twist.angular.z:.2f}")

    def _movement_loop(self) -> None:
        dt = 0.033

        while self._running:
            try:
                with self._velocity_lock:
                    vx = self._velocity["x"]
                    vy = self._velocity["y"]
                    vz = self._velocity["z"]
                    yaw_rate = self._velocity["yaw"]

                self._position["x"] += vx * dt
                self._position["y"] += vy * dt
                self._position["z"] += vz * dt
                self._heading += yaw_rate * dt

                margin = 0.3
                self._position["x"] = max(margin, min(self._room.width - margin,  self._position["x"]))
                self._position["y"] = max(margin, min(self._room.depth - margin,  self._position["y"]))
                self._position["z"] = max(0.5,    min(self._room.height - margin, self._position["z"]))
                self._avoid_obstacles()

                qz = math.sin(self._heading / 2)
                qw = math.cos(self._heading / 2)
                nav_pose = PoseStamped(
                    position=Vector3(
                        self._position["x"],
                        self._position["y"],
                        0.0,
                    ),
                    orientation=Quaternion(0.0, 0.0, qz, qw),
                    frame_id="world",
                    ts=time.time(),
                )
                self._publish_tf(nav_pose)

                cloud = self._cast_lidar()
                if cloud is not None:
                    self.lidar.publish(cloud)
                self._publish_video()
                self.telemetry.publish({
                    "x": self._position["x"],
                    "y": self._position["y"],
                    "z": self._position["z"],
                    "ts": time.time(),
                })
                time.sleep(dt)

            except Exception as e:
                logger.debug(f"Movement loop error: {e}")
                time.sleep(0.1)

    def _publish_tf(self, msg: PoseStamped) -> None:
        self._odom = msg
        self.odom.publish(msg)

        base_link = Transform(
            translation=Vector3(
                self._position["x"],
                self._position["y"],
                self._position["z"],
            ),
            rotation=msg.orientation,
            frame_id="world",
            child_frame_id="base_link",
            ts=msg.ts if hasattr(msg, "ts") else time.time(),
        )
        self.tf.publish(base_link)

        camera_link = Transform(
            translation=Vector3(0.1, 0.0, -0.05),
            rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
            frame_id="base_link",
            child_frame_id="camera_link",
            ts=time.time(),
        )
        self.tf.publish(camera_link)

    def _avoid_obstacles(self) -> None:
        x = self._position["x"]
        y = self._position["y"]
        z = self._position["z"]

        for obs in self._room.obstacles:
            in_footprint = (
                abs(x - obs.center_x) < obs.width / 2
                and abs(y - obs.center_y) < obs.depth / 2
            )
            if in_footprint and z < obs.max_height:
                self._position["z"] = obs.max_height + 0.5
                logger.debug(f"Climbing above obstacle at ({obs.center_x}, {obs.center_y})")

    def _cast_lidar(self) -> PointCloud2 | None:
        ox = self._position["x"]
        oy = self._position["y"]
        oz = self._position["z"]
        points = []

        for v_deg in LIDAR_VERTICAL_ANGLES_DEG:
            v_rad = math.radians(v_deg)
            cos_v = math.cos(v_rad)
            sin_v = math.sin(v_rad)

            h_deg = 0.0
            while h_deg < 360.0:
                h_rad = math.radians(h_deg)
                dx = math.cos(h_rad) * cos_v
                dy = math.sin(h_rad) * cos_v
                dz = sin_v

                hit = self._ray_hit(ox, oy, oz, dx, dy, dz)
                if hit is not None:
                    points.append(hit)
                h_deg += LIDAR_RAY_STEP_DEG

        if not points:
            return None

        pts = np.array(points, dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        return PointCloud2(pcd, frame_id="world", ts=time.time())

    def _ray_hit(
        self,
        ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float,
    ) -> tuple[float, float, float] | None:
        best_t = LIDAR_MAX_RANGE

        for wall in self._room.walls:
            t = self._ray_segment_2d(ox, oy, dx, dy, wall)
            if t is not None and 0.001 < t < best_t:
                hz = oz + dz * t
                if 0.0 <= hz <= self._room.height:
                    best_t = t

        for obs in self._room.obstacles:
            t = self._ray_box(
                ox, oy, oz, dx, dy, dz,
                obs.center_x - obs.width / 2,
                obs.center_y - obs.depth / 2,
                0.0,
                obs.center_x + obs.width / 2,
                obs.center_y + obs.depth / 2,
                obs.max_height,
            )
            if t is not None and 0.001 < t < best_t:
                best_t = t

        for plane_z in (0.0, self._room.height):
            if abs(dz) > 1e-6:
                t = (plane_z - oz) / dz
                if 0.001 < t < best_t:
                    hx = ox + dx * t
                    hy = oy + dy * t
                    if 0.0 <= hx <= self._room.width and 0.0 <= hy <= self._room.depth:
                        best_t = t

        if best_t >= LIDAR_MAX_RANGE:
            return None

        hit_x = ox + dx * best_t
        hit_y = oy + dy * best_t  
        hit_z = oz + dz * best_t

        margin = 0.3
        hit_x = max(margin, min(self._room.width - margin, hit_x))
        hit_y = max(margin, min(self._room.depth - margin, hit_y))
        hit_z = max(margin, min(self._room.height - margin, hit_z))

        return (hit_x, hit_y, hit_z)

    def _ray_segment_2d(
        self,
        ox: float, oy: float,
        dx: float, dy: float,
        wall: Wall,
    ) -> float | None:
        wx = wall.x2 - wall.x1
        wy = wall.y2 - wall.y1

        denom = dx * wy - dy * wx
        if abs(denom) < 1e-10:
            return None

        t = ((wall.x1 - ox) * wy - (wall.y1 - oy) * wx) / denom
        s = ((wall.x1 - ox) * dy - (wall.y1 - oy) * dx) / denom

        if t > 0 and 0.0 <= s <= 1.0:
            return t
        return None

    def _ray_box(
        self,
        ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float,
        x_min: float, y_min: float, z_min: float,
        x_max: float, y_max: float, z_max: float,
    ) -> float | None:
        t_min = 0.0
        t_max = LIDAR_MAX_RANGE

        for o, d, lo, hi in (
            (ox, dx, x_min, x_max),
            (oy, dy, y_min, y_max),
            (oz, dz, z_min, z_max),
        ):
            if abs(d) < 1e-10:
                if o < lo or o > hi:
                    return None
            else:
                t1 = (lo - o) / d
                t2 = (hi - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return None

        return t_min if t_min > 0.001 else None

    def _publish_video(self) -> None:
        import cv2
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        text = (
            f"SIM  x={self._position['x']:.2f}  "
            f"y={self._position['y']:.2f}  "
            f"z={self._position['z']:.2f}"
        )
        cv2.putText(frame, text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        img = Image.from_numpy(frame, format=ImageFormat.BGR)
        self.video.publish(img)
