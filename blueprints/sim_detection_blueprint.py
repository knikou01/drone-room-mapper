from typing import Any

import rerun as rr

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels import VoxelGridMapper
from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import WavefrontFrontierExplorer
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.patrolling.module import PatrollingModule
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.perception.detection.moduleDB import ObjectDBModule
from dimos.visualization.vis_module import vis_module

from rooms.generator import generate_room
from sim_camera.sim_camera_module import SimCameraModule, make_camera_info
from simulation.sim_drone_connection_module import SimDroneConnectionModule


def _make_rerun_blueprint() -> Any:
    import rerun.blueprint as rrb
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/video", name="Drone Camera"),
            rrb.Spatial2DView(origin="world/color_image", name="Detection Camera"),
            rrb.Spatial3DView(
                origin="world",
                name="3D Map",
                background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
                line_grid=rrb.LineGrid3D(
                    plane=rr.components.Plane3D.XY.with_distance(0.5),
                ),
            ),
            column_shares=[1, 1, 2],
        ),
        rrb.TimePanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
    )


def _static_drone_body(rr: Any) -> list[Any]:
    return [
        rr.Boxes3D(half_sizes=[0.25, 0.25, 0.1], colors=[(255, 100, 0)]),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


_rerun_config = {
    "blueprint": _make_rerun_blueprint,
    "static": {"world/tf/base_link": _static_drone_body},
}

_room = generate_room(
    width=10.0,
    depth=10.0,
    height=3.0,
    min_inner_wall_length=1.5,
    min_obstacle_length=0.3,
    min_obstacle_height=0.3,
)

_vis = vis_module(
    viewer_backend=global_config.viewer,
    rerun_config=_rerun_config,
)

drone_sim_detection = autoconnect(
    _vis,
    SimDroneConnectionModule.blueprint(room=_room),
    SimCameraModule.blueprint(room=_room),
    ObjectDBModule.blueprint(camera_info=make_camera_info()),
    VoxelGridMapper.blueprint(emit_every=5),
    CostMapper.blueprint(),
    WavefrontFrontierExplorer.blueprint(),
    PatrollingModule.blueprint(),
    ReplanningAStarPlanner.blueprint(),
    MovementManager.blueprint(),
).global_config(n_workers=7)

__all__ = ["drone_sim_detection"]
