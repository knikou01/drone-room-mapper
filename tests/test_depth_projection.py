"""Verify DroneDepthModule's camera->world projection math using a
synthetic depth map with known geometry, independent of any real camera
or drone connection.

Scenario: drone hovering at (drone_x=2, drone_y=3), yaw=0 (facing +x),
cruise_altitude_m=1.5. A flat wall sits 4m directly in front of the drone,
perpendicular to the camera's optical axis (i.e. every pixel along the
camera's principal ray reports depth=4m at the centre, with off-centre
pixels reporting a corrected radial distance -- but Depth Anything reports
depth along the optical axis (Z_cam), not radial range, so a flat
fronto-parallel wall has CONSTANT depth across all pixels. That is the
correct test case for verifying constant-depth-plane behavior.)

Expected: the centre pixel (u=cx, v=cy) should project to world point
approximately (drone_x + 4, drone_y, cruise_altitude_m), since the centre
ray points exactly along body-forward with no left/right or up/down offset.
"""

import math
import numpy as np
import sys
sys.path.insert(0, "/home/claude")

# Re-implement just the projection function in isolation (no DimOS deps)
# to mirror DroneDepthModule._project_to_world exactly.

def project_to_world(depth_np, drone_x, drone_y, yaw, cruise_altitude_m,
                      fx_nominal=1000.0, fy_nominal=1000.0,
                      cx_nominal=960.0, cy_nominal=540.0,
                      min_depth_m=0.3, max_depth_m=8.0, stride=8):
    h, w = depth_np.shape
    scale_x = w / (cx_nominal * 2.0)
    scale_y = h / (cy_nominal * 2.0)
    fx = fx_nominal * scale_x
    fy = fy_nominal * scale_y
    cx = cx_nominal * scale_x
    cy = cy_nominal * scale_y

    rows = np.arange(0, h, stride, dtype=np.float32)
    cols = np.arange(0, w, stride, dtype=np.float32)
    uu, vv = np.meshgrid(cols, rows)
    dd = depth_np[::stride, ::stride]

    u = uu.ravel(); v = vv.ravel(); d = dd.ravel()
    mask = (d > min_depth_m) & (d < max_depth_m) & np.isfinite(d)
    u, v, d = u[mask], v[mask], d[mask]

    x_body = d
    y_body = -(u - cx) * d / fx
    z_body = -(v - cy) * d / fy

    pz = cruise_altitude_m
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

    x_world = x_body * cos_yaw - y_body * sin_yaw + drone_x
    y_world = x_body * sin_yaw + y_body * cos_yaw + drone_y
    z_world = z_body + pz

    return np.stack([x_world, y_world, z_world], axis=1), (cx, cy, fx, fy)


def close(a, b, tol=0.05):
    return abs(a - b) < tol


# --- Test 1: flat wall straight ahead, yaw=0 ---
h, w = 360, 640
depth = np.full((h, w), 4.0, dtype=np.float32)  # constant 4m everywhere
drone_x, drone_y, yaw, alt = 2.0, 3.0, 0.0, 1.5

points, (cx, cy, fx, fy) = project_to_world(depth, drone_x, drone_y, yaw, alt)

# Find the point closest to the image centre (u=cx, v=cy)
# Reconstruct u,v grid the same way to locate the centre point
stride = 8
rows = np.arange(0, h, stride)
cols = np.arange(0, w, stride)
uu, vv = np.meshgrid(cols, rows)
centre_idx = np.argmin((uu.ravel() - cx)**2 + (vv.ravel() - cy)**2)
centre_point = points[centre_idx]

expected = (drone_x + 4.0, drone_y, alt)
assert close(centre_point[0], expected[0]), f"x: got {centre_point[0]:.3f}, expected {expected[0]:.3f}"
assert close(centre_point[1], expected[1]), f"y: got {centre_point[1]:.3f}, expected {expected[1]:.3f}"
assert close(centre_point[2], expected[2]), f"z: got {centre_point[2]:.3f}, expected {expected[2]:.3f}"
print(f"Test 1 (yaw=0, flat wall): centre point {centre_point} ~= expected {expected}  PASS")

# --- Test 2: same wall, drone yawed 90deg (facing +y) ---
yaw90 = math.radians(90)
points2, _ = project_to_world(depth, drone_x, drone_y, yaw90, alt)
centre_point2 = points2[centre_idx]
expected2 = (drone_x, drone_y + 4.0, alt)
assert close(centre_point2[0], expected2[0]), f"x: got {centre_point2[0]:.3f}, expected {expected2[0]:.3f}"
assert close(centre_point2[1], expected2[1]), f"y: got {centre_point2[1]:.3f}, expected {expected2[1]:.3f}"
print(f"Test 2 (yaw=90, flat wall): centre point {centre_point2} ~= expected {expected2}  PASS")

# --- Test 3: pixel to the right of centre (u > cx) should map to body -y (right) ---
# Pick a column index well to the right of cx, at the centre row
right_col_idx = np.argmin(np.abs(cols - (cx + 200)))
right_row_idx = np.argmin(np.abs(rows - cy))
flat_idx = right_row_idx * len(cols) + right_col_idx
right_point = points[flat_idx]  # yaw=0 case
# With yaw=0: y_world = y_body = -(u-cx)*d/fx (negative since u>cx)
assert right_point[1] < drone_y, f"pixel right of centre should map to -y (right of forward-facing drone), got y={right_point[1]:.3f} vs drone_y={drone_y}"
print(f"Test 3 (off-centre pixel, right side): y={right_point[1]:.3f} < drone_y={drone_y}  PASS (correctly right of drone)")

# --- Test 4: depth filtering -- points beyond max_depth_m should be excluded ---
depth_far = np.full((h, w), 20.0, dtype=np.float32)  # beyond default max_depth_m=8.0
points_far, _ = project_to_world(depth_far, drone_x, drone_y, 0.0, alt)
assert len(points_far) == 0, f"expected all points filtered out (20m > max_depth_m=8.0), got {len(points_far)}"
print(f"Test 4 (range filter): {len(points_far)} points after filtering 20m depth (max=8.0m)  PASS")

print("\nAll projection tests passed.")
