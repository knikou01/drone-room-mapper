"""DimSim visual-SLAM + agentic console blueprint -- NAVIGATION pose
driven by real VO (odom_vo). Mapping deliberately stays on ground truth --
see the scope note below.

Near-identical to sim_dimsim_agentic_blueprint.py (see that file's
docstring for the full agentic-console reasoning, unchanged here) -- this
is the agentic-console equivalent of sim_dimsim_vo_nav_blueprint.py (see
that file's docstring for the full mapping-vs-navigation reasoning this
mirrors), added as a NEW, separate blueprint rather than editing the
ground-truth-pose one in place, so that variant stays available for
comparison/fallback.

Remapping ALL THREE real pose-consumers (AgenticFrontierSelector,
ReplanningAStarPlanner, AND DimSimDepthLidarModule) to odom_vo at once
produces a visibly corrupted map, since DimSimDepthLidarModule's
point-cloud projection has no self-correction against VO drift the way
ReplanningAStarPlanner's frequent replanning does. See
sim_dimsim_vo_nav_blueprint.py's module docstring for the full mechanism.
Remaps ONLY the navigation-facing consumers:
- AgenticFrontierSelector.odom -- inherited unmodified from
  WavefrontFrontierExplorer.odom, used for frontier distance/direction
  scoring.
- ReplanningAStarPlanner.odom -- real-time navigation/replanning pose.

DimSimDepthLidarModule.odom is deliberately NOT remapped here -- stays on
ground truth by DimOS's default name-based topic resolution, same as
before this file existed. Still a proper In[PoseStamped] stream, so a
future drift-tolerant mapping approach only needs a one-line remap added
back, not a redesign.

DimSimVisualOdometryModule is UNCHANGED: still separately subscribes to
ground-truth /odom for its own permanent divergence-logging comparison
(never remove that), still publishes odom_vo regardless of consumers.

Prerequisite: DimSim relay running with a scene loaded and a browser tab
connected BEFORE this blueprint starts.

VERIFY ON FIRST RUN: check the startup "Transport" log lines --
AgenticFrontierSelector and ReplanningAStarPlanner's odom streams must
show a topic containing "odom_vo". DimSimDepthLidarModule's odom stream
must show the raw ground-truth "/odom" topic -- if it shows "odom_vo"
instead, the corrupted-map problem this scope change fixed will come back.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import rerun as rr

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.web_human_input import WebInput
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.global_config import global_config
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels import VoxelGridMapper
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.visualization.vis_module import vis_module

from sim_camera.dimsim_camera_module import DimSimCameraModule

from drone.agentic_frontier_selector import AgenticFrontierSelector
from drone.dimsim_depth_lidar_module import DimSimDepthLidarModule
from drone.dimsim_visual_odometry_module import DimSimVisualOdometryModule

# Same system prompt as sim_dimsim_agentic_blueprint.py -- the pose-source
# change is invisible to the console/LLM, it only affects which topic
# navigation/mapping modules read pose from.
from blueprints.sim_dimsim_agentic_blueprint import (
    CONSOLE_SYSTEM_PROMPT,
    _LANGCHAIN_PROVIDER_NAMES,
)


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


def build_sim_dimsim_agentic_vo(
    llm_provider: str, llm_model: str, safety_monitor_enabled: bool = True
) -> Blueprint:
    """Build the Phase 3 (VO-driven pose) agentic blueprint. Same
    parameters/reasoning as build_sim_dimsim_agentic -- see that
    function's docstring."""
    _vis = vis_module(global_config.viewer, rerun_config=_rerun_config)
    console_provider = _LANGCHAIN_PROVIDER_NAMES.get(llm_provider, llm_provider)
    console_model = f"{console_provider}:{llm_model}"

    return autoconnect(
        _vis,
        DimSimCameraModule.blueprint(),
        DimSimVisualOdometryModule.blueprint(),
        DimSimDepthLidarModule.blueprint(),
        VoxelGridMapper.blueprint(emit_every=5),
        CostMapper.blueprint(),
        AgenticFrontierSelector.blueprint(
            llm_provider=llm_provider,
            llm_model=llm_model,
            safety_monitor_enabled=safety_monitor_enabled,
        ),
        ReplanningAStarPlanner.blueprint(),
        MovementManager.blueprint(),
        McpServer.blueprint(),
        McpClient.blueprint(
            system_prompt=CONSOLE_SYSTEM_PROMPT,
            model=console_model,
        ),
        WebInput.blueprint(),
    ).remappings(
        [
            # Unchanged from sim_dimsim_agentic_blueprint.py.
            (VoxelGridMapper, "lidar", "depth_lidar"),
            (AgenticFrontierSelector, "color_image", "dimsim_color_image"),
            (ReplanningAStarPlanner, "global_costmap", "hazard_costmap"),
            # NAVIGATION ONLY -- see module docstring for why
            # DimSimDepthLidarModule is deliberately NOT remapped here.
            (ReplanningAStarPlanner, "odom", "odom_vo"),
            (AgenticFrontierSelector, "odom", "odom_vo"),
        ]
    )


__all__ = ["build_sim_dimsim_agentic_vo", "CONSOLE_SYSTEM_PROMPT"]
