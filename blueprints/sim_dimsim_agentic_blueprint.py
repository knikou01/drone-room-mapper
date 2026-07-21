"""DimSim visual-SLAM + agentic (LLM-driven) frontier exploration blueprint.

Same vision-based mapping stack as sim_dimsim_vo_blueprint.py
(DimSimDepthLidarModule's depth-derived point cloud feeds VoxelGridMapper
INSTEAD of DimSim's raycast /lidar, pose from ground-truth /odom -- see that
file's docstring for the full reasoning, unchanged here), with
WavefrontFrontierExplorer swapped for AgenticFrontierSelector
(drone/agentic_frontier_selector.py).

Frontier SELECTION is plain WavefrontFrontierExplorer geometric heuristic,
unconditionally -- the LLM's role here is request-driven, not
decision-per-step: see AgenticFrontierSelector's module docstring for the
full reasoning and CONSOLE_SYSTEM_PROMPT below for exactly what the console
can still do (start/stop exploration, drive directly, react to a live
hazard via the background safety monitor).

DimSimVisualOdometryModule is kept running for its own sake (ongoing
VO-vs-ground-truth telemetry -- see its module docstring for the current
accuracy limitation) but nothing consumes odom_vo here; navigation still
uses ground-truth /odom.

Adds a natural-language console on top: McpServer + McpClient + WebInput,
same pattern as blueprints/sim_agentic_blueprint.py, at
http://localhost:5555. Note that WavefrontFrontierExplorer's
explore()/stop_exploration() are @rpc, not @skill -- dimos/core/module.py's
get_skills() only surfaces @skill-decorated methods to MCP, so those names
are not callable as console tools. AgenticFrontierSelector has real
@skill-decorated begin_exploration/end_exploration wrappers instead (see
that file) -- this blueprint's SYSTEM_PROMPT below lists the names that
actually work.

Exposed as a FACTORY FUNCTION (build_sim_dimsim_agentic), not a bare
module-level blueprint constant like every other blueprint in this repo --
deliberate deviation: the run script (run_sim_dimsim_agentic.py) prompts the
user to pick an LLM provider/model interactively before anything is built,
since AgenticFrontierSelector.blueprint(llm_provider=..., llm_model=...)
needs that choice at construction time.

Prerequisite: DimSim relay running with a scene loaded and a browser tab
connected BEFORE this blueprint starts (same as sim_dimsim_vo_blueprint.py).

KNOWN ISSUE (see sim_dimsim_vo_blueprint.py's docstring for full detail):
the apt scene's glass surface permanently wedges the robot on contact, not
fixable from this repo. Frontier selection itself does no glass-specific
visual judgment call -- the background safety monitor (see
AgenticFrontierSelector's module docstring) is what actually watches the
live camera for a hazard and reacts, independent of frontier decisions.
This is a best-effort visual judgment call, NOT a guaranteed fix: DimSim's
simulated depth sensor may not perceive glass as an obstacle at all, in
which case the geometric costmap data alone may already look like a clear
opening. If the robot still gets stuck, that's this known limitation, not
a new bug.

KNOWN ISSUE, DIMOS BUG, NOT PART OF THIS TASK -- DO NOT ATTEMPT TO FIX:
end_exploration (this blueprint's begin_exploration/end_exploration
skills, see agentic_frontier_selector.py) can trigger a real
AssertionError in ../dimos/dimos/navigation/replanning_a_star/
global_planner.py's _plan_path(): `assert current_goal is not None` at
line 318. Root cause is a genuine check-then-act race in DimOS itself, not
this repo: _plan_path() calls self.cancel_goal(but_will_try_again=True)
(which does NOT clear self._current_goal when but_will_try_again=True --
see cancel_goal(), line ~160) and THEN separately re-acquires the lock a
few lines later to read self._current_goal. In the gap between those two
lock acquisitions, GlobalPlanner's own _on_stopped_navigating callback
(subscribed in start(), fires on a different thread from the local
planner's reactive stream) can call self.cancel_goal(arrived=True) --
which DOES clear _current_goal, since but_will_try_again defaults False --
right as _plan_path() is about to read it. Appears non-fatal in
practice -- the LCM handler thread logs the exception and continues,
exploration resumes fine on the next begin_exploration -- but it is a real
bug in DimOS's navigation stack, not something introduced by this repo.

This same race also triggers via the background safety monitor's
_emergency_stop(): its zero-Twist publish on tele_cmd_vel is treated as
teleop by MovementManager, which cancels the in-flight goal and races with
the planner's own in-flight replan request for it. There is also a second
downstream shape of the exact same race: `ValueError: No current global
costmap available` in _find_safe_goal (global_planner.py:324), not just
the AssertionError above -- same check-then-act window, different cleared
state. Both are non-fatal: logged and swallowed by the LCM handler thread,
exploration recovers cleanly on the next begin_exploration. Not fixed
here, same reasons as above -- surfaces roughly every time the safety
monitor stops the drone, since it cancels goals far more often than manual
end_exploration.

AgenticFrontierSelector runs a continuous background safety monitor (see
that file's docstring, "Background safety monitor" section) that checks
the live camera on its own timer and stops the drone immediately on a
hazard, independent of frontier decisions -- because consulting the camera
only at the moment a frontier TARGET is chosen isn't enough; the hazard
can appear while the path to it is being driven.

ObjectDBModule (object detection) is deliberately NOT part of this
blueprint -- nothing in the exploration/mapping pipeline depends on it,
and its auto-discovered ask_vlm MCP tool is broken for two independent
reasons (missing ALIBABA_API_KEY, a separate get_next() timeout) and not
excludable via CONSOLE_SYSTEM_PROMPT alone -- MCP exposes every discovered
tool to the real tool-calling schema regardless of what the prompt says.
Removing the module removes the broken tool entirely rather than trying to
prompt around it.

The safety monitor above stops the drone before glass contact, but
stopping alone would mean the drone approaches the same spot again later,
since a hazard would only be excluded from frontier TARGETS
(AgenticFrontierSelector's own bookkeeping), not from the geometric
costmap itself, which doesn't perceive glass as an obstacle at all.
AgenticFrontierSelector stamps every confirmed hazard into the costmap as
a real occupied wall (see its hazard_costmap/_stamp_hazards) and
republishes the patched grid on a new hazard_costmap stream;
ReplanningAStarPlanner is remapped here to read THAT stream instead of
CostMapper's raw global_costmap directly, so a confirmed hazard blocks A*
path planning too, not just frontier selection -- the drone can no longer
be routed through it on the way to some other frontier either.

`build_sim_dimsim_agentic` derives the console's model string from the
SAME llm_provider/llm_model choice used for frontier selection, via
_LANGCHAIN_PROVIDER_NAMES below, rather than hardcoding a fixed provider
for the console -- otherwise picking a different provider specifically to
dodge an exhausted quota on one provider wouldn't actually help, since the
console would still use the hardcoded one. Note McpClient._process_message's
LangGraph agent loop doesn't catch LLM call failures, so an exception there
kills the whole McpClient-thread -- a dimos-level gap, not fixed here, but
avoided by keeping console and frontier-selection providers in sync.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env before any other import, so GEMINI_API_KEY / GROQ_API_KEY are in
# os.environ before the worker process tree forks -- same reasoning as
# agentic_explorer_blueprint.py.
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

CONSOLE_SYSTEM_PROMPT = """\
You are controlling an autonomous drone that maps a room in the DimSim
simulator using vision-based mapping (a depth camera instead of lidar).
Frontier exploration itself is plain geometric/heuristic (not something
you choose candidates for) -- your job is to start or stop it on request
and to drive the drone directly when asked or when something looks unsafe.
A separate background safety monitor (not you) watches the live camera
continuously while a goal is being driven and reacts on its own if it
sees a real hazard.

## Your available tools (ONLY use these exact tool names)
- begin_exploration — start autonomous frontier exploration
- end_exploration — stop exploration
- move(x, y, yaw_rate, duration) — drive directly: x=forward m/s,
  y=left m/s, yaw_rate=turn rate rad/s (positive=left), duration=seconds.
  Takes over from autonomous exploration immediately, the same way a human
  driving the teleop dashboard would.
- stop_move — immediately halt any in-progress move (does NOT stop
  autonomous exploration -- use end_exploration for that)
- agent_send — send a message
- list_modules — list available modules
- server_status — check server status

## CRITICAL
Do NOT attempt to call any tool not listed above. Never hallucinate tool
names -- if you're not sure a tool exists, say so instead of guessing.

## Rules
- Use begin_exploration to start mapping.
- Use end_exploration to stop.
- Use move/stop_move to reposition or steer the drone directly -- e.g. if
  you or the human notice something that looks unsafe (like glass, which
  permanently disables this robot on contact with no recovery), or if the
  human gives a direct instruction like "back away from that" or "turn
  left".
- When you are actually escaping or avoiding something dangerous, a tiny
  or very short move does NOT help -- use a meaningful duration (at least
  2-3 seconds, up to the max allowed) and a real velocity, not a barely-
  there nudge. Prefer backing straight away (negative x) or turning to
  face away from the hazard over a small sideways drift, unless the
  camera clearly shows sideways is the only safe direction.
- Do not immediately call begin_exploration right after a safety move
  without first confirming you're actually clear -- resuming autonomous
  exploration right away can send the drone right back toward the same
  hazard if it's still the most attractive nearby frontier. If you're not
  sure it's clear, look again (another camera frame will come with the
  next decision) or make another move first.
- For routine (non-emergency) repositioning, several short move() calls
  are fine so you can reassess between them -- the distinction above is
  specifically about genuine hazard avoidance.
- Report what you are doing in plain text.
"""


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

# Maps AgenticFrontierConfig.llm_provider's own naming (used by
# AgenticFrontierSelector._build_llm_client, drone/agentic_frontier_selector.py)
# to langchain's init_chat_model-style provider-prefix dispatch (used by
# McpClient's model="provider:model" convention, dimos/agents/mcp/mcp_client.py)
# -- only "gemini" needs remapping: langchain's registered name for it is
# "google_genai", not "gemini". "openai"/"anthropic"/"groq" already match
# langchain's own provider names and pass through unchanged (see the
# .get(llm_provider, llm_provider) fallback below).
_LANGCHAIN_PROVIDER_NAMES = {"gemini": "google_genai"}


def build_sim_dimsim_agentic(
    llm_provider: str, llm_model: str, safety_monitor_enabled: bool = True
) -> Blueprint:
    """Build the blueprint with the given LLM provider/model for
    AgenticFrontierSelector's frontier-selection calls. See module
    docstring for why this is a function rather than a module-level
    constant, and for why the console uses this same provider/model choice
    instead of a hardcoded one.

    safety_monitor_enabled is passed straight through to
    AgenticFrontierConfig -- see its docstring for why this needs to be
    toggleable (false positives on ordinary architecture when testing on
    a scene with no glass at all, e.g. apt_no_glass)."""
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
            # Phase 2 remap, unchanged from sim_dimsim_vo_blueprint.py.
            (VoxelGridMapper, "lidar", "depth_lidar"),
            # dimsim_color_image remap, needed by every consumer that
            # declares a color_image In[Image] -- AgenticFrontierSelector
            # wants the live camera frame for its frontier decisions.
            (AgenticFrontierSelector, "color_image", "dimsim_color_image"),
            # Route the A* planner through AgenticFrontierSelector's
            # hazard-augmented costmap (see that module's hazard_costmap
            # declaration) instead of CostMapper's raw output directly --
            # otherwise a confirmed hazard (e.g. glass the safety monitor
            # stopped for) would only be excluded from frontier TARGETS,
            # not from paths the planner routes through on the way to some
            # other frontier. AgenticFrontierSelector's own
            # global_costmap: In[OccupancyGrid] stays unremapped -- it
            # still needs CostMapper's raw output as the input it patches.
            (ReplanningAStarPlanner, "global_costmap", "hazard_costmap"),
        ]
    )


__all__ = ["build_sim_dimsim_agentic", "CONSOLE_SYSTEM_PROMPT"]
