"""Verify the depth->world-frame point projection for Phase 2 (visual
mapping replacing DimSim's raycast lidar), using the SAME camera-frame
remap already used and left unvalidated in DimSimVisualOdometryModule's
_camera_to_world_pose (X_dimos=Z_cam, Y_dimos=-X_cam, Z_dimos=-Y_cam).

Scenario: drone at world position (2, 3, 0.5), yaw=0 (facing +x). A flat
wall sits 4m directly ahead, perpendicular to the camera's optical axis --
constant depth=4m across the whole frame. The centre pixel (u=cx, v=cy)
should project to world point ~(2+4, 3, 0.5) = (6, 3, 0.5).
"""
import math
import numpy as np


def project_depth_to_world(depth_m, drone_x, drone_y, drone_z, yaw,
                            fx, fy, cx, cy,
                            min_depth_m=0.05, max_depth_m=15.0, stride=4):
    h, w = depth_m.shape
    rows = np.arange(0, h, stride, dtype=np.float32)
    cols = np.arange(0, w, stride, dtype=np.float32)
    uu, vv = np.meshgrid(cols, rows)
    dd = depth_m[::stride, ::stride]

    u = uu.ravel(); v = vv.ravel(); d = dd.ravel()
    mask = (d > min_depth_m) & (d < max_depth_m) & np.isfinite(d)
    u, v, d = u[mask], v[mask], d[mask]

    # Camera-frame unprojection (OpenCV/Open3D convention: x=right, y=down, z=forward)
    cam_x = (u - cx) / fx * d
    cam_y = (v - cy) / fy * d
    cam_z = d

    # Same remap already used (and flagged unvalidated) in
    # DimSimVisualOdometryModule._camera_to_world_pose:
    # X_dimos = Z_cam, Y_dimos = -X_cam, Z_dimos = -Y_cam
    body_x = cam_z
    body_y = -cam_x
    body_z = -cam_y

    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    world_x = body_x * cos_yaw - body_y * sin_yaw + drone_x
    world_y = body_x * sin_yaw + body_y * cos_yaw + drone_y
    world_z = body_z + drone_z

    return np.stack([world_x, world_y, world_z], axis=1), (u, v)


def close(a, b, tol=0.05):
    return abs(a - b) < tol


# DimSim capture resolution (confirmed from dimsim_camera_module.py: 640x288)
h, w = 288, 640
fx, fy, cx, cy = 500.0, 500.0, 320.0, 144.0  # placeholder intrinsics for this test

depth = np.full((h, w), 4.0, dtype=np.float32)
drone_x, drone_y, drone_z, yaw = 2.0, 3.0, 0.5, 0.0

points, (u, v) = project_depth_to_world(depth, drone_x, drone_y, drone_z, yaw, fx, fy, cx, cy)

centre_idx = np.argmin((u - cx) ** 2 + (v - cy) ** 2)
centre_point = points[centre_idx]
expected = (drone_x + 4.0, drone_y, drone_z)

assert close(centre_point[0], expected[0]), f"x: got {centre_point[0]:.3f}, expected {expected[0]:.3f}"
assert close(centre_point[1], expected[1]), f"y: got {centre_point[1]:.3f}, expected {expected[1]:.3f}"
assert close(centre_point[2], expected[2]), f"z: got {centre_point[2]:.3f}, expected {expected[2]:.3f}"
print(f"Test 1 (yaw=0, flat wall centre): {centre_point} ~= {expected}  PASS")

# yaw=90: forward ray should point toward +y
yaw90 = math.radians(90)
points2, _ = project_depth_to_world(depth, drone_x, drone_y, drone_z, yaw90, fx, fy, cx, cy)
centre_point2 = points2[centre_idx]
expected2 = (drone_x, drone_y + 4.0, drone_z)
assert close(centre_point2[0], expected2[0]), f"x: got {centre_point2[0]:.3f}, expected {expected2[0]:.3f}"
assert close(centre_point2[1], expected2[1]), f"y: got {centre_point2[1]:.3f}, expected {expected2[1]:.3f}"
print(f"Test 2 (yaw=90, flat wall centre): {centre_point2} ~= {expected2}  PASS")

print("\nAll depth-lidar projection tests passed.")
