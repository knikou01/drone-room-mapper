#!/usr/bin/env python3

# Load .env before any other import, so GEMINI_API_KEY is in os.environ
# before the worker process tree is spawned. The LLM client is constructed
# inside AgenticFrontierSelector.start() (after fork), but the env var
# needs to be set in the parent process so it's inherited by the worker.
from dotenv import load_dotenv

load_dotenv()

from typing import Any

import rerun as rr

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels import VoxelGridMapper
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.patrolling.module import PatrollingModule
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.visualization.vis_module import vis_module
from rooms.generator import generate_room
from sim_camera.sim_camera_module import SimCameraModule
from simulation.sim_drone_connection_module import SimDroneConnectionModule

from drone.agentic_frontier_selector import AgenticFrontierSelector


def _make_rerun_blueprint() -> Any:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            # SimCameraModule publishes under world/color_image;
            # SimDroneConnectionModule's placeholder is at world/video.
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

drone_agentic_explorer = autoconnect(
    _vis,
    SimDroneConnectionModule.blueprint(room=_room),
    SimCameraModule.blueprint(room=_room),       # real PyBullet camera → color_image stream
    VoxelGridMapper.blueprint(emit_every=5),
    CostMapper.blueprint(),
    # ↓ Only change from sim_blueprint.py's explorer: WavefrontFrontierExplorer → AgenticFrontierSelector
    AgenticFrontierSelector.blueprint(
        llm_provider="gemini",
        llm_model="gemini-3.5-flash",
    ),
    PatrollingModule.blueprint(),
    ReplanningAStarPlanner.blueprint(),
    MovementManager.blueprint(),
).global_config(n_workers=7)  # +1 for SimCameraModule vs sim_blueprint.py's 6

__all__ = ["drone_agentic_explorer"]
