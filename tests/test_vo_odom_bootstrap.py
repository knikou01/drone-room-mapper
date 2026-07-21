"""Verify DimSimVisualOdometryModule's odom_vo bootstrap publish.

Without it, odom_vo only publishes AFTER a successful tracked transform.
When navigation modules are remapped to consume odom_vo instead of ground
truth (sim_dimsim_vo_nav_blueprint.py), this creates a chicken-and-egg
deadlock: they need at least one odom message before picking/pursuing a
goal, the robot needs a goal before anything commands it to move, and if
the spawn view happens to be geometrically simple,
_compute_transform_sparse's min_point_spread_ratio check keeps rejecting
every attempt, so VO never gets a chance to succeed either without the
robot ever moving.

Fix: publish an initial odom_vo message immediately when the first
ground-truth sample arrives, at the same origin-anchor pose the module
already latches internally for its coordinate frame -- not fabricated
information, since at that instant zero relative motion has been observed
yet, so "VO's estimate" and "the spawn pose" are identical by
construction. Only unblocks the bootstrap; every subsequent odom_vo
update still goes through the normal tracking/validation pipeline
unweakened.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone.dimsim_visual_odometry_module import DimSimVisualOdometryModule
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


mod = DimSimVisualOdometryModule()
published = []
mod.odom_vo.publish = lambda pose: published.append(pose)

spawn = PoseStamped(
    position=Vector3(3.0, 2.0, 0.54), orientation=Quaternion(0, 0, 0, 1),
    frame_id="world", ts=0.0,
)
mod._on_ground_truth_odom(spawn, None)
check("exactly one bootstrap publish after the first ground-truth sample", len(published) == 1)

if published:
    p = published[0]
    check(
        "bootstrap pose exactly matches the spawn ground truth (zero motion observed yet)",
        abs(p.position.x - 3.0) < 1e-6
        and abs(p.position.y - 2.0) < 1e-6
        and abs(p.position.z - 0.54) < 1e-6,
    )

# A second (and later) ground-truth sample must NOT re-publish -- the
# origin is latched once; re-publishing on every ground-truth tick would
# defeat the point of VO (it would just track ground truth forever).
moved = PoseStamped(
    position=Vector3(3.5, 2.0, 0.54), orientation=Quaternion(0, 0, 0, 1),
    frame_id="world", ts=1.0,
)
mod._on_ground_truth_odom(moved, None)
check("no re-publish on later ground-truth samples (origin latched once)", len(published) == 1)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
else:
    print("All VO odom_vo bootstrap checks PASSED")
