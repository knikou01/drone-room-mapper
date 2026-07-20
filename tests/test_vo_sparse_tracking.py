"""Verify DimSimVisualOdometryModule._compute_transform_sparse (Phase 3's
ORB + depth-based 3D-3D RANSAC tracking, added 2026-07-20 to replace dense
frame-to-frame compute_rgbd_odometry -- see that module's docstring for
why: dense odometry assumes small inter-frame motion, and DimSim's
observed pairing_gap (1000-1200ms documented baseline, up to 60000ms once
under heavy system load) regularly breaks that assumption).

Two synthetic scene builders, both warped by the homography a known rigid
camera motion induces and with a correctly-derived per-pixel depth map for
the "current" frame (NOT simply reusing the previous frame's depth -- a
plane at a fixed depth in the PREVIOUS camera's frame is at a genuinely
different depth from the CURRENT camera's viewpoint once it has moved.
Getting this wrong produced deceptive results while first writing this
test: rotation recovered near-perfectly while translation was off by ~1m,
because an incorrect depth model biases the recovered translation scale
specifically -- worth remembering if this test ever needs modifying):

1. make_corner_frame_pair: a two-plane "room corner" (left half of the
   image at one depth, right half at another) -- genuine 3D structure
   (variance along all three axes, not just the two in-image ones), used
   for the "should succeed accurately" cases.

2. make_plane_frame_pair: a SINGLE fronto-parallel plane -- genuinely
   planar data (near-zero variance along the camera's depth axis
   specifically), used to test min_point_spread_ratio's rejection of
   degenerate geometry (2026-07-20 addition -- see that config field's
   docstring for the live-run evidence this fix is based on: facing a
   flat wall head-on produced a small, "plausible-looking" but WRONG
   transform that earlier checks couldn't catch). IMPORTANT: a flat
   plane is degenerate for this check REGARDLESS of its orientation/tilt
   relative to the camera -- SVD of the point cloud's own coordinates is
   orientation-invariant, a plane only ever spans a 2D subspace no matter
   how it's angled. Also still useful for the "not enough overlapping
   texture at extreme rotation" case below, since a single plane is an
   artificially hard case for large rotations specifically (almost all
   "current frame" pixels beyond a modest angle map outside the original
   plane's warped bounds) -- a real room has texture at many
   depths/orientations simultaneously, which should provide meaningfully
   more surviving correspondences at the same rotation angle than either
   synthetic scene here. These tests confirm the algorithm is
   mathematically/numerically CORRECT when given well-conditioned data,
   and that it fails SAFELY (not silently-wrong) when the data is
   degenerate or insufficient -- they cannot by themselves confirm real
   DimSim scenes will provide enough of either at the gaps actually
   observed live. That's Gate 1's job (a real run_sim_dimsim_vo_only.py
   live test), not these synthetic ones.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from drone.dimsim_visual_odometry_module import DimSimVisualOdometryModule

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def rot_matrix(rx, ry, rz):
    cxr, sxr = np.cos(rx), np.sin(rx)
    cyr, syr = np.cos(ry), np.sin(ry)
    czr, szr = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cxr, -sxr], [0, sxr, cxr]])
    Ry = np.array([[cyr, 0, syr], [0, 1, 0], [-syr, 0, cyr]])
    Rz = np.array([[czr, -szr, 0], [szr, czr, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _plane_warp(fx, fy, cx, cy, w, h, Z0, R_true, t_true):
    """Homography + correct per-pixel current-frame depth for a single
    fronto-parallel plane at depth Z0 (previous camera's frame), for the
    given rigid motion (point-transform convention X2 = R X1 + t, matching
    _compute_transform_sparse's own contract). Returns (H, depth_curr_fn)
    where depth_curr_fn(output_mask_shape) computes the current-frame
    depth map for that plane region."""
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    n = np.array([0, 0, 1.0])
    H = K @ (R_true + np.outer(t_true, n) / Z0) @ np.linalg.inv(K)
    H = H / H[2, 2]

    Hinv = np.linalg.inv(H)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones_like(us, dtype=np.float64)
    pix_curr = np.stack([us, vs, ones], axis=-1).astype(np.float64)
    pix_prev_h = pix_curr @ Hinv.T
    pix_prev = pix_prev_h[..., :2] / pix_prev_h[..., 2:3]
    u_prev, v_prev = pix_prev[..., 0], pix_prev[..., 1]

    X_prev = (u_prev - cx) / fx * Z0
    Y_prev = (v_prev - cy) / fy * Z0
    Z_prev = np.full_like(X_prev, Z0)
    p_prev = np.stack([X_prev, Y_prev, Z_prev], axis=-1)
    p_curr = p_prev @ R_true.T + t_true
    curr_depth_m = p_curr[..., 2].astype(np.float32)

    valid = (u_prev >= 0) & (u_prev < w) & (v_prev >= 0) & (v_prev < h)
    curr_depth_m[~valid] = 0.0
    return H, curr_depth_m


def make_plane_frame_pair(mod, rot_rad_xyz, t_true, Z0=3.0, seed=1):
    """Single fronto-parallel plane -- see module docstring: genuinely
    planar/degenerate by construction, any orientation."""
    fx, fy = mod._camera_info.K[0], mod._camera_info.K[4]
    cx, cy = mod._camera_info.K[2], mod._camera_info.K[5]
    w, h = mod._camera_info.width, mod._camera_info.height

    rng = np.random.default_rng(seed)
    prev_color = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    prev_depth_m = np.full((h, w), Z0, dtype=np.float32)

    R_true = rot_matrix(*rot_rad_xyz)
    t_true = np.array(t_true, dtype=np.float64)

    H, curr_depth_m = _plane_warp(fx, fy, cx, cy, w, h, Z0, R_true, t_true)
    curr_color = cv2.warpPerspective(prev_color, H, (w, h))

    return curr_color, curr_depth_m, prev_color, prev_depth_m, R_true, t_true


def make_corner_frame_pair(mod, rot_rad_xyz, t_true, Za=2.0, Zb=4.0, seed=1):
    """Two fronto-parallel planes at DIFFERENT depths (left half of the
    image at Za, right half at Zb) -- a "room corner"-like scene with
    genuine variance along the camera's depth axis, not just the two
    in-image ones. Well-conditioned for min_point_spread_ratio, unlike
    make_plane_frame_pair's single plane."""
    fx, fy = mod._camera_info.K[0], mod._camera_info.K[4]
    cx, cy = mod._camera_info.K[2], mod._camera_info.K[5]
    w, h = mod._camera_info.width, mod._camera_info.height
    half = w // 2

    rng = np.random.default_rng(seed)
    prev_color = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    prev_depth_m = np.empty((h, w), dtype=np.float32)
    prev_depth_m[:, :half] = Za
    prev_depth_m[:, half:] = Zb

    R_true = rot_matrix(*rot_rad_xyz)
    t_true = np.array(t_true, dtype=np.float64)

    Ha, depth_a = _plane_warp(fx, fy, cx, cy, w, h, Za, R_true, t_true)
    Hb, depth_b = _plane_warp(fx, fy, cx, cy, w, h, Zb, R_true, t_true)
    warped_a = cv2.warpPerspective(prev_color, Ha, (w, h))
    warped_b = cv2.warpPerspective(prev_color, Hb, (w, h))

    # Each output pixel's TRUE source region depends on which plane it
    # maps back to under the correct (region-specific) homography -- use
    # whichever warp lands its source pixel inside the region it actually
    # belongs to (left half -> Za's plane, right half -> Zb's plane).
    # Since both planes share the same previous-frame texture, blend by
    # simply taking each output pixel from whichever warp corresponds to
    # a source pixel within [0, half) vs [half, w) matching that warp's
    # own depth assignment.
    curr_color = np.where(
        (np.arange(w)[None, :, None] < half), warped_a, warped_b
    ).astype(np.uint8)
    curr_depth_m = np.where(np.arange(w)[None, :] < half, depth_a, depth_b).astype(np.float32)

    return curr_color, curr_depth_m, prev_color, prev_depth_m, R_true, t_true


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

# --- Larger, non-trivial rotation (still well short of the ~50-60deg
# region where a two-region synthetic scene like this one starts running
# out of reliably-shared texture between regions -- confirmed by direct
# experimentation while writing this test: some seed/angle combinations in
# that harder range produced RANSAC "inlier" sets that were numerically
# non-degenerate by the spread check but still geometrically unstable
# (large errors even though min_point_spread_ratio didn't reject them) --
# a real, separate limitation of THIS simple two-plane synthetic
# construction specifically, not something this test tries to paper over
# by cherry-picking; 10deg/seed=4 is a configuration confirmed to produce
# well-distributed cross-region matches) still recovers correctly when
# there's still meaningful, well-conditioned overlap. ---
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

# --- Degenerate (near-planar) geometry rejection (2026-07-20 addition,
# added after a live run showed a small, "plausible-looking" but WRONG
# transform published while the robot faced a flat wall head-on -- raw
# "not enough matching points" warnings from cv2.estimateAffine3D's own
# internals coincided with VO reporting 0.128m/3.4deg while the robot had
# actually moved 2.096m/61.2deg). Same modest-motion parameters that
# succeed cleanly against the corner scene above should be REJECTED
# against a single (degenerate) plane, regardless of how modest/accurate
# the underlying motion is -- this confirms min_point_spread_ratio is
# doing real work, not just coincidentally never triggering. ---
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

# --- Physical-plausibility rejection (2026-07-20 addition, added after a
# live run showed RANSAC converging on a plausible-looking but wrong
# consensus against a real room's repetitive geometry -- e.g. one interval
# reported 5.15x the true displacement). A geometrically-correct transform
# fed an implausibly short dt implies an impossible speed and must be
# rejected regardless of how clean the underlying match was. ---
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
