"""
Builds a PyBullet scene from a rooms.models.Room.

Wall thickness is the only assumption not in the Room dataclass — flagged
as WALL_THICKNESS below, tune it visually once you see the rendered output.
"""
from __future__ import annotations

import math

import pybullet as p
import pybullet_data

from rooms.models import Room, Wall, Obstacle

WALL_THICKNESS = 0.1  # meters; Wall has no thickness field


def _wall_geometry(wall: Wall, height: float) -> tuple[list, list, list]:
    """Return (half_extents, center_xyz, quaternion) for a wall box."""
    dx, dy = wall.x2 - wall.x1, wall.y2 - wall.y1
    length = math.hypot(dx, dy) or 1e-6
    yaw = math.atan2(dy, dx)
    center = [(wall.x1 + wall.x2) / 2, (wall.y1 + wall.y2) / 2, height / 2]
    half = [length / 2, WALL_THICKNESS / 2, height / 2]
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    return half, center, quat


def _obstacle_geometry(obs: Obstacle) -> tuple[list, list, list]:
    half = [obs.width / 2, obs.depth / 2, obs.max_height / 2]
    center = [obs.center_x, obs.center_y, obs.max_height / 2]
    quat = p.getQuaternionFromEuler([0, 0, 0])
    return half, center, quat


def _make_static_box(half, center, quat, color, client_id: int) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=list(color), physicsClientId=client_id)
    return p.createMultiBody(0, col, vis, center, quat, physicsClientId=client_id)


def _shade(base_rgb: tuple[float, float, float], index: int, count: int) -> tuple[float, float, float, float]:
    """Vary lightness of base_rgb across `count` steps so each instance looks distinct."""
    if count <= 1:
        t = 0.5
    else:
        t = index / (count - 1)  # 0..1
    # Blend between a darker and lighter version of base_rgb
    lo = tuple(c * 0.55 for c in base_rgb)
    hi = tuple(min(1.0, c * 1.35 + 0.1) for c in base_rgb)
    rgb = tuple(lo[i] + (hi[i] - lo[i]) * t for i in range(3))
    return (*rgb, 1.0)


def build_room(room: Room, client_id: int) -> None:
    """Populate an existing PyBullet client with floor, walls and obstacles.

    Walls are shaded in distinct steps of blue-gray, obstacles in distinct
    steps of brown, so the drone's motion is visually obvious frame-to-frame
    in the camera feed instead of every surface looking identical.
    """
    # Floor
    _make_static_box(
        [room.width / 2, room.depth / 2, 0.05],
        [room.width / 2, room.depth / 2, -0.05],
        [0, 0, 0, 1],
        (0.5, 0.5, 0.55, 1.0),
        client_id,
    )

    walls = list(room.walls)
    for i, wall in enumerate(walls):
        half, center, quat = _wall_geometry(wall, room.height)
        color = _shade((0.45, 0.55, 0.75), i, len(walls))  # blue-gray family
        _make_static_box(half, center, quat, color, client_id)

    obstacles = list(room.obstacles)
    for i, obs in enumerate(obstacles):
        half, center, quat = _obstacle_geometry(obs)
        color = _shade((0.65, 0.4, 0.2), i, len(obstacles))  # brown/orange family
        _make_static_box(half, center, quat, color, client_id)


def connect_headless() -> int:
    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    return client_id
