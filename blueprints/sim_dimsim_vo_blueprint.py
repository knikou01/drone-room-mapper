"""DimSim + Visual SLAM blueprint.

Phase 1 (DimSimVisualOdometryModule): RGB-D visual odometry running
alongside DimSim's ground-truth /odom, publishing on a separate odom_vo
stream for accuracy comparison. Does not affect navigation.

Phase 2 (DimSimDepthLidarModule, this update): builds a world-frame point
cloud from DimSim's depth image and, via .remappings() below, feeds it to
VoxelGridMapper INSTEAD of DimSim's own raycast-based /lidar topic. This is
the actual "map with vision, not lidar" swap. Pose source for this phase is
still ground-truth /odom, not odom_vo -- isolates "is the depth-derived map
geometrically correct" from "is VO's pose trustworthy", which is Phase 3.

WavefrontFrontierExplorer + ReplanningAStarPlanner + MovementManager
already drive real autonomous exploration against DimSim's own
ground-truth /odom, with no remapping needed for odom itself.

CostMapper is required here: without it, WavefrontFrontierExplorer.
latest_costmap is never populated, so its exploration loop silently waits
forever without publishing any goal or logging anything (`if
self.latest_costmap is None ... continue`, no log line) -- the drone would
never move at all.

Prerequisite: DimSim relay running with a scene loaded and a browser tab
connected BEFORE this blueprint starts.

VERIFY ON FIRST RUN: check the startup "Transport" log lines --
VoxelGridMapper's lidar stream must show DimSimDepthLidarModule as its
source (topic containing "depth_lidar"), NOT DimSim's raw "/lidar" topic.
If the remapping didn't take effect, the map will still be built from
DimSim's raycast lidar, defeating the purpose of this phase.

KNOWN ISSUE, NOT PART OF THIS TASK -- DO NOT ATTEMPT TO FIX: the "apt"
scene (currently the only scene defined in this DimSim fork's
scenes.template.json) has a glass surface the pathfinder routes straight
into almost immediately after exploration starts. Once the robot's physics
body contacts it, the robot gets permanently wedged -- not recoverable
even via manual teleop through the localhost:7779 dashboard. This is a
known, general limitation of robots running DimOS against DimSim's
thin/transparent-surface collision handling, not specific to this
blueprint or to VO. Workaround for testing VO under real sustained motion:
do NOT trigger python -m lcm_probe.start_exploration (which is what drives
the pathfinder into the glass) -- instead drive the robot entirely by hand
via the localhost:7779 teleop dashboard, staying clear of the glass, and
let it run long enough to accumulate real motion for VO-vs-ground-truth
comparison.
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

sim_dimsim_vo = autoconnect(
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
        # Phase 2: point VoxelGridMapper's lidar input at
        # DimSimDepthLidarModule's depth-derived cloud instead of its
        # default resolution (which would otherwise pick up DimSim's raw
        # raycast /lidar topic by name-matching). See module docstring in
        # dimsim_depth_lidar_module.py for the full collision-avoidance
        # reasoning.
        (VoxelGridMapper, "lidar", "depth_lidar"),
        # DimSimCameraModule's image stream is named "dimsim_color_image",
        # not "color_image" -- deliberately, to avoid colliding with
        # DimSim's own raw /color_image topic (see dimsim_camera_module.py
        # docstring for the feedback-loop bug this fixes). ObjectDBModule
        # still declares In[Image] named "color_image", so point it at the
        # renamed stream here.
        (ObjectDBModule, "color_image", "dimsim_color_image"),
    ]
)

__all__ = ["sim_dimsim_vo"]
