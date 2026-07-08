"""
Run this before any DimOS integration:
    cd drone-room-mapper
    python -m sim_camera.test_standalone

Confirm:
  1. The saved PNG shows a recognisable room interior.
  2. Depth min/max look plausible for a 10x10 room (expect ~0.05m–~14m).
"""
from rooms.generator import generate_room
from sim_camera.pybullet_room import build_room, connect_headless
from sim_camera.pybullet_camera import DroneCamera
import cv2


def main() -> None:
    room = generate_room(width=10.0, depth=10.0, height=3.0,
                         min_inner_wall_length=1.5, min_obstacle_length=0.3,
                         min_obstacle_height=0.3)
    print(f"Room: {room.width}x{room.depth}x{room.height}, "
          f"{len(room.walls)} walls, {len(room.obstacles)} obstacles")

    client_id = connect_headless()
    build_room(room, client_id)

    cam = DroneCamera(client_id, width=640, height=480, fov_deg=70.0)
    pos = (room.width / 2, room.depth / 2, 1.5)
    frame = cam.capture(pos, yaw=0.0)

    rgb, depth_m = frame["rgb"], frame["depth_m"]
    print(f"RGB:   shape={rgb.shape} dtype={rgb.dtype}")
    print(f"Depth: shape={depth_m.shape} min={depth_m.min():.2f}m max={depth_m.max():.2f}m")

    out = "sim_camera_test.png"
    cv2.imwrite(out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
