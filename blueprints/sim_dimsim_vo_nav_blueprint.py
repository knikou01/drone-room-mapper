"""DimSim + Visual SLAM blueprint -- NAVIGATION pose driven by real VO
(odom_vo). Mapping deliberately stays on ground truth -- see the scope
note below.

Near-identical to sim_dimsim_vo_blueprint.py (VO computed and compared
against ground truth, but navigation/mapping still used ground truth
/odom) -- added as a NEW, separate blueprint rather than editing that one
in place, so the ground-truth-pose variant stays available for
comparison/fallback (same "keep both" precedent as
sim_dimsim_vo_blueprint.py coexisting with sim_dimsim_vo_only_blueprint.py).

Remapping ALL THREE real pose-consumers (ReplanningAStarPlanner,
WavefrontFrontierExplorer, AND DimSimDepthLidarModule) to odom_vo at once
produces a visibly corrupted map: DimSimDepthLidarModule projects the
depth camera's points into world coordinates using whatever pose it's
given, and VoxelGridMapper/CostMapper accumulate those points permanently
with no mechanism to correct earlier contributions -- so every time VO's
pose estimate is off, the SAME physical wall gets projected to a
DIFFERENT apparent world location, smearing/duplicating it in the
accumulated map. Navigation decisions don't share this problem --
ReplanningAStarPlanner replans frequently against the LATEST pose, so it
gets real-time self-correction that mapping structurally cannot get from
a single-pass point accumulation.

Given that asymmetry, this blueprint remaps ONLY the navigation-facing
pose consumers to odom_vo:
- ReplanningAStarPlanner.odom -- real-time navigation/replanning pose.
- WavefrontFrontierExplorer.odom -- frontier distance/direction scoring.

DimSimDepthLidarModule.odom is deliberately NOT remapped here -- it keeps
resolving to ground truth by DimOS's default name-based topic resolution
(same "odom" stream name -> DimSim's raw ground-truth /odom topic
mechanism used throughout this codebase), same as before this file
existed. It's still a proper In[PoseStamped] stream (converted from a
hardcoded raw-LCM subscription specifically so it COULD be remapped, see
that module's docstring), so a future attempt at drift-tolerant mapping
(loop closure / pose-graph correction, out of scope here) only needs a
one-line remap added back once that exists -- this isn't a dead end, just
a smaller, honest first scope.

DimSimVisualOdometryModule itself is UNCHANGED here: it still separately
subscribes to ground-truth /odom for its own permanent divergence-logging
comparison (never remove that), and still publishes odom_vo regardless of
what consumes it.

Prerequisite: DimSim relay running with a scene loaded and a browser tab
connected BEFORE this blueprint starts (same as sim_dimsim_vo_blueprint.py).

VERIFY ON FIRST RUN: check the startup "Transport" log lines --
ReplanningAStarPlanner's odom stream and WavefrontFrontierExplorer's odom
stream must show a topic containing "odom_vo". DimSimDepthLidarModule's
odom stream must show the raw ground-truth "/odom" topic -- if it shows
"odom_vo" instead, someone re-added the mapping remap and the corrupted-map
problem this scope change fixed will come back.

KNOWN ISSUE, NOT PART OF THIS TASK -- DO NOT ATTEMPT TO FIX: same glass-
collision limitation documented in sim_dimsim_vo_blueprint.py's docstring,
unrelated to the pose-source change here.
"""

from typing import Any

import rerun as rr

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels import VoxelGridMapper
from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontFrontierExplorer,
)
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.visualization.vis_module import vis_module

from sim_camera.dimsim_camera_module import DimSimCameraModule, make_camera_info_default
from dimos.perception.detection.moduleDB import ObjectDBModule

from drone.dimsim_visual_odometry_module import DimSimVisualOdometryModule
from drone.dimsim_depth_lidar_module import DimSimDepthLidarModule


def _make_rerun_blueprint() -> Any:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/color_image", name="Camera"),
            rrb.Spatial3DView(
                origin="world",
                name="3D Map",
                background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
                line_grid=rrb.LineGrid3D(
                    plane=rr.components.Plane3D.XY.with_distance(0.5),
                ),
            ),
            column_shares=[1, 2],
        ),
        rrb.TimePanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
    )


def _static_drone_body(rr: Any) -> list[Any]:
    return [
        rr.Boxes3D(
            half_sizes=[0.25, 0.25, 0.1],
            colors=[(255, 100, 0)],
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


_rerun_config = {
    "blueprint": _make_rerun_blueprint,
    "static": {
        "world/tf/base_link": _static_drone_body,
    },
}

_vis = vis_module(global_config.viewer, rerun_config=_rerun_config)

sim_dimsim_vo_nav = autoconnect(
    _vis,
    DimSimCameraModule.blueprint(),
    ObjectDBModule.blueprint(camera_info=make_camera_info_default()),
    DimSimVisualOdometryModule.blueprint(),
    DimSimDepthLidarModule.blueprint(),
    VoxelGridMapper.blueprint(emit_every=5),
    CostMapper.blueprint(),
    WavefrontFrontierExplorer.blueprint(),
    ReplanningAStarPlanner.blueprint(),
    MovementManager.blueprint(),
).remappings(
    [
        # Unchanged from sim_dimsim_vo_blueprint.py.
        (VoxelGridMapper, "lidar", "depth_lidar"),
        (ObjectDBModule, "color_image", "dimsim_color_image"),
        # NAVIGATION ONLY -- see module docstring for why
        # DimSimDepthLidarModule is deliberately NOT remapped here (stays
        # on ground truth; mapping has no self-correction against VO drift
        # the way replanning does).
        (ReplanningAStarPlanner, "odom", "odom_vo"),
        (WavefrontFrontierExplorer, "odom", "odom_vo"),
    ]
)

__all__ = ["sim_dimsim_vo_nav"]
