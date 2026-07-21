"""Verify DimSimVisualOdometryModule._compute_transform_sparse (ORB +
depth-based 3D-3D RANSAC tracking, used instead of dense frame-to-frame
compute_rgbd_odometry -- see that module's docstring for why: dense
odometry assumes small inter-frame motion, and DimSim's observed
color/depth pairing_gap regularly breaks that assumption).

Uses the synthetic frame-pair builders in _vo_synthetic_scenes.py (shared
with test_vo_learned_tracking.py, since both exercise the same downstream
3D-3D RANSAC/Kabsch backend with different feature front-ends -- see that
file's docstring for what each builder represents). A single plane is also
useful here for the "not enough overlapping texture at extreme rotation"
case below, since it's an artificially hard case for large rotations
specifically (almost all "current frame" pixels beyond a modest angle map
outside the original plane's warped bounds) -- a real room has texture at
many depths/orientations simultaneously, which should provide meaningfully
more surviving correspondences at the same rotation angle. These tests
confirm the algorithm is mathematically/numerically CORRECT when given
well-conditioned data, and that it fails SAFELY (not silently-wrong) when
the data is degenerate or insufficient -- they cannot by themselves
confirm real DimSim scenes will provide enough of either at the gaps
observed in practice. That needs a real run_sim_dimsim_vo_only.py live
test, not these synthetic ones.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from drone.dimsim_visual_odometry_module import DimSimVisualOdometryModule
from _vo_synthetic_scenes import make_corner_frame_pair, make_plane_frame_pair

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


mod = DimSimVisualOdometryModule()

# --- Modest inter-frame motion (well within a healthy ~100-200ms capture
# interval) recovers the true rigid transform accurately, against a
# well-conditioned (non-planar) scene. ---
curr_color, curr_depth_m, prev_color, prev_depth_m, R_true, t_true = make_corner_frame_pair(
    mod, (0.03, 0.08, -0.02), [0.5, -0.1, 0.2], seed=7
)
success, trans = mod._compute_transform_sparse(curr_color, curr_depth_m, prev_color, prev_depth_m, 1.0)
check("modest motion: tracking succeeds", success)
if success:
    r_err = float(np.max(np.abs(trans[:3, :3] - R_true)))
    t_err = float(np.max(np.abs(trans[:3, 3] - t_true)))
    check(f"modest motion: rotation recovered accurately (max abs err={r_err:.4f})", r_err < 0.01)
    check(f"modest motion: translation recovered accurately (max abs err={t_err:.4f}m)", t_err < 0.01)

# --- Larger, non-trivial rotation still recovers correctly when there's
# still meaningful, well-conditioned overlap. Kept well short of the
# ~50-60deg region where this two-region synthetic scene starts running
# out of reliably-shared texture between regions -- past that point some
# seed/angle combinations produce RANSAC "inlier" sets that are
# numerically non-degenerate by the spread check but still geometrically
# unstable, a limitation of this simple two-plane construction, not of the
# tracking algorithm itself. ---
curr_color, curr_depth_m, prev_color, prev_depth_m, R_true, t_true = make_corner_frame_pair(
    mod, (0.0, np.radians(10), 0.0), [0.3, 0.0, 0.0], seed=4
)
success, trans = mod._compute_transform_sparse(curr_color, curr_depth_m, prev_color, prev_depth_m, 1.0)
check("larger rotation (10deg yaw, 0.3m): tracking succeeds", success)
if success:
    r_err = float(np.max(np.abs(trans[:3, :3] - R_true)))
    t_err = float(np.max(np.abs(trans[:3, 3] - t_true)))
    check(f"larger rotation: rotation recovered accurately (max abs err={r_err:.4f})", r_err < 0.01)
    check(f"larger rotation: translation recovered accurately (max abs err={t_err:.4f}m)", t_err < 0.01)

# --- Very large rotation against a single fronto-parallel plane (an
# artificially hard case, see module docstring) fails SAFELY -- no
# transform published, not a silently-wrong one. This is the desired
# failure mode (matches compute_rgbd_odometry's success=False contract)
# and is the whole point of preferring this over dense odometry's
# documented failure mode of reporting false success at near-identity. ---
curr_color, curr_depth_m, prev_color, prev_depth_m, R_true, t_true = make_plane_frame_pair(
    mod, (0.0, np.radians(69), 0.0), [0.55, 0.0, 0.0], seed=4
)
success, _trans = mod._compute_transform_sparse(curr_color, curr_depth_m, prev_color, prev_depth_m, 1.0)
check("extreme single-plane rotation (69deg): fails safely (no false-success)", not success)

# --- Degenerate (near-planar) geometry rejection: the same modest-motion
# parameters that succeed cleanly against the corner scene above should be
# REJECTED against a single (degenerate) plane, regardless of how
# modest/accurate the underlying motion is -- this confirms
# min_point_spread_ratio is doing real work, not just coincidentally never
# triggering. ---
curr_color, curr_depth_m, prev_color, prev_depth_m, R_true, t_true = make_plane_frame_pair(
    mod, (0.03, 0.08, -0.02), [0.5, -0.1, 0.2], seed=7
)
success, _trans = mod._compute_transform_sparse(curr_color, curr_depth_m, prev_color, prev_depth_m, 1.0)
check("degenerate single-plane geometry (otherwise-trackable motion): rejected", not success)

# --- No previous frame yet: first call must not crash, and must be a
# clean "not yet trackable" result (matches _compute_transform_dense's
# own first-frame contract of returning False rather than raising). ---
fresh_mod = DimSimVisualOdometryModule()
h, w = fresh_mod._camera_info.height, fresh_mod._camera_info.width
blank_color = np.zeros((h, w, 3), dtype=np.uint8)
blank_depth = np.zeros((h, w), dtype=np.float32)
try:
    success, _trans = fresh_mod._compute_transform_sparse(
        blank_color, blank_depth, blank_color, blank_depth, 1.0
    )
    check("blank/degenerate frames: fails safely, no crash", not success)
except Exception as e:  # noqa: BLE001 -- deliberately broad, this check is "did it crash at all"
    check(f"blank/degenerate frames: fails safely, no crash (raised {e!r})", False)

# --- Physical-plausibility rejection: a geometrically-correct transform
# fed an implausibly short dt implies an impossible speed and must be
# rejected regardless of how clean the underlying match was -- guards
# against RANSAC converging on a plausible-looking but wrong consensus
# against a real room's repetitive geometry. ---
curr_color, curr_depth_m, prev_color, prev_depth_m, R_true, t_true = make_corner_frame_pair(
    mod, (0.03, 0.08, -0.02), [0.5, -0.1, 0.2], seed=7
)
success, _trans = mod._compute_transform_sparse(
    curr_color, curr_depth_m, prev_color, prev_depth_m, 0.001  # implies ~550 m/s
)
check("implausibly fast implied motion (tiny dt): rejected", not success)

# Sanity: the SAME frame pair with a normal dt still succeeds (confirms
# the plausibility check isn't just rejecting everything).
success, _trans = mod._compute_transform_sparse(
    curr_color, curr_depth_m, prev_color, prev_depth_m, 1.0
)
check("same frame pair with a normal dt: still succeeds", success)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
else:
    print("All VO sparse-tracking checks PASSED")
