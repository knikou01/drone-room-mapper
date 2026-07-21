#!/usr/bin/env python3
"""
A subclass of dimos.navigation.frontier_exploration.
wavefront_frontier_goal_selector.WavefrontFrontierExplorer. Does not modify
that file or any other dimos source for its own sake -- it only imports and
extends the class.

The background safety monitor's emergency stop (see _emergency_stop below)
needs a way to command the robot immediately -- a backup+turn maneuver --
without dimos's existing WavefrontFrontierExplorer/MovementManager stack
mistaking it for a human teleop override and aborting the entire
exploration session. No existing dimos stream drew that distinction, so
MovementManager gained one small, additive, optional input
(silent_cmd_vel: In[Twist], dimos/navigation/movement_manager/
movement_manager.py) that claims teleop priority WITHOUT the human-override
side effect.

Frontier SELECTION does not consult an LLM at all -- this class inherits
WavefrontFrontierExplorer's plain geometric heuristic unmodified for every
frontier decision (get_exploration_goal is not overridden). The heuristic
is already hazard-aware for free: self.latest_costmap is the
hazard-stamped grid (see _on_costmap/_stamp_hazards below), so a confirmed
hazard is already occupied-cell territory by the time the base class's own
candidate scoring runs. The LLM's role is request-driven, not
decision-per-step: begin_exploration/end_exploration (start/stop on
request), move/stop_move (direct steering on request), and the background
safety monitor (watches the live camera continuously while a goal is
being driven and reacts to a real hazard -- see "Background safety
monitor" below).
"""

from __future__ import annotations

import base64
import json
import math
import os
import threading
import time
from typing import Any, Literal

import numpy as np
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.core.core import rpc
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontConfig,
    WavefrontFrontierExplorer,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class AgenticFrontierConfig(WavefrontConfig):
    # All WavefrontConfig fields (min_frontier_perimeter, occupancy_threshold,
    # safe_distance, lookahead_distance, max_explored_distance,
    # info_gain_threshold, num_no_gain_attempts, goal_timeout) are inherited
    # unchanged -- the geometric scoring is the only thing that picks
    # frontiers here.
    # Safety clamps for the move() skill -- see its docstring. Small
    # duration cap forces shorter, repeated commands rather than one long
    # blind dash, giving more chances to re-observe between moves.
    max_move_duration_s: float = 5.0
    max_move_linear_mps: float = 0.5
    max_move_yaw_rate_rps: float = 1.0
    # Lets the background safety monitor be turned off entirely -- e.g.
    # for testing the rest of the exploration pipeline on a scene with no
    # glass at all, where the monitor is pure overhead and can false-
    # positive on ordinary architecture. Defaults to True.
    safety_monitor_enabled: bool = True
    # How often (seconds) the background safety monitor checks the live
    # camera frame for a hazard directly ahead WHILE a goal is being
    # driven -- independent of and much more frequent than frontier
    # decisions. A chosen target can be geometrically fine while the path
    # to it still isn't, if the depth costmap doesn't perceive the hazard
    # (e.g. glass) at all.
    safety_check_interval_s: float = 3.0
    # Only actually stop for a hazard once the LLM's own distance estimate
    # is within this many meters -- otherwise a hazard merely visible in
    # frame from meters away stops the drone before it makes any progress.
    # Sized with real margin: the robot drives at ~0.55 m/s and this check
    # only runs every safety_check_interval_s (3.0s default), so in the
    # worst case it can cover ~1.65m between two checks.
    hazard_stop_distance_m: float = 2.0
    # Rough forward-projection distance (meters) used to estimate a
    # detected hazard's world position when the LLM couldn't give a usable
    # distance estimate -- see _stamp_hazards/_emergency_stop. Prefer the
    # LLM's own estimated_distance_m when available.
    hazard_projection_distance_m: float = 1.5
    # Radius (meters) of the synthetic occupied patch stamped into the
    # costmap at each confirmed hazard's estimated position.
    hazard_wall_radius_m: float = 0.5
    # How long (seconds) the safety monitor holds silent_cmd_vel priority
    # after an emergency stop. Needs to comfortably fit the full backup +
    # turn-away maneuver below so the exploration loop has time to publish
    # a fresh goal before priority lapses and nav_cmd_vel would otherwise
    # resume driving toward the hazard.
    hazard_stop_hold_s: float = 6.0
    # Backs straight away (negative x) for this many seconds before
    # turning -- backing away doesn't require knowing which side is clear
    # and creates buffer distance before the turn phase swings the
    # footprint around.
    hazard_backup_speed_mps: float = 0.3
    hazard_backup_duration_s: float = 1.5
    # After backing away, rotates in place by hazard_turn_angle_rad
    # (default pi, ~180 degrees) so the camera ends up facing away from
    # the hazard -- a full about-face rather than a partial turn, since a
    # single forward-facing frame gives no information about which
    # partial direction is actually clear.
    hazard_turn_rate_rps: float = 1.0
    hazard_turn_angle_rad: float = math.pi
    camera_max_width_px: int = 768
    llm_provider: Literal["openai", "anthropic", "gemini", "groq"] = "gemini"
    llm_model: str = "gemini-3.5-flash"
    llm_temperature: float = 0.2


# Runs continuously (every safety_check_interval_s) WHILE a goal is being
# driven -- camera-only, no map render, kept fast and cheap since it runs
# on its own timer regardless of whether a frontier decision is happening.
_SAFETY_CHECK_PROMPT = """\
You are watching a drone's forward camera feed while it autonomously
drives toward a destination in an indoor room. Look at this single frame
only: is there a hazard ahead that the drone could collide with on its
current path -- in particular glass, a window, or a mirror (driving into
glass permanently disables this robot with no recovery), or any other
obstacle?

Respond with ONLY a JSON object of this exact shape, no other text:
{
  "hazard_ahead": false,
  "estimated_distance_m": null,
  "reasoning": "<=1 sentence"
}
"hazard_ahead": true if something ahead could plausibly be a real
collision risk on the drone's current heading, even if it still looks
far away right now -- err toward true when in doubt about something that
could be glass; how urgent it is gets judged separately below via
distance, not by refusing to flag it at all.

"estimated_distance_m": your best rough estimate, in meters, of how far
away the nearest such hazard is, using whatever visual cues are available
(relative size, perspective, apparent size of doors/furniture/other
familiar objects in the scene). This is necessarily approximate from a
single 2D camera frame with no depth information -- give your best
estimate anyway rather than refusing to answer. Use null only when
hazard_ahead is false, or on the rare frame where there is truly no usable
visual cue to judge distance from at all.
"""


class AgenticFrontierSelector(WavefrontFrontierExplorer):
    """Drop-in alternative to WavefrontFrontierExplorer: same Inputs /
    Outputs / skills, plus a camera input and an LLM client, but frontier
    *selection* is plain-inherited WavefrontFrontierExplorer geometric
    heuristic -- get_exploration_goal is not overridden at all. What this
    class actually adds over the base explorer:
    - begin_exploration/end_exploration/move/stop_move skills (MCP-exposed
      wrappers).
    - Hazard-wall stamping (_stamp_hazards/hazard_costmap) so a confirmed
      hazard becomes a real occupied cell for both frontier scoring and
      A* path planning, not just an excluded target.
    - A background safety monitor that watches the live camera
      continuously WHILE a goal is being driven (see "Background safety
      monitor" below) and reacts to a real hazard with a backup+turn
      maneuver -- independent of frontier decisions, which are purely
      geometric and don't look at the camera at all.
    """

    config: AgenticFrontierConfig

    # Matches SimCameraModule's `color_image: Out[Image]` by (name, type)
    # so autoconnect wires them automatically -- no .remappings() needed.
    # Falls back gracefully (map-only decisions) when no module publishing
    # color_image is present in the blueprint.
    color_image: In[Image]

    # Same name as MovementManager's existing tele_cmd_vel In[Twist] and
    # WebsocketVisModule's existing tele_cmd_vel Out[Twist] -- name-matches
    # and auto-wires with no .remappings() needed. Multiple publishers on
    # this one topic is the intended pattern (MovementManager aggregates
    # any teleop-style source). Publishing here reuses MovementManager's
    # existing teleop-priority-over-A* and goal-cancellation logic
    # (_on_teleop -> _cancel_goal(), see
    # dimos/navigation/movement_manager/movement_manager.py) for free --
    # the same path the human teleop dashboard already exercises.
    tele_cmd_vel: Out[Twist]

    # Republishes CostMapper's costmap with any confirmed hazards (see
    # _stamp_hazards) burned in as permanent occupied cells. Both this
    # module's own frontier/obstacle-distance logic (self.latest_costmap
    # is set from the patched grid, not the raw one -- see the _on_costmap
    # override) AND ReplanningAStarPlanner (remapped in the blueprint to
    # read THIS stream instead of CostMapper's global_costmap directly)
    # need to see a confirmed hazard as a real wall, not just an
    # unreachable frontier target. New name (not "global_costmap")
    # deliberately, so autoconnect doesn't collide it with CostMapper's
    # own output of that name -- this module still consumes that one
    # directly via the inherited global_costmap: In[OccupancyGrid],
    # unremapped.
    hazard_costmap: Out[OccupancyGrid]

    # Matches MovementManager's silent_cmd_vel: In[Twist] by (name, type)
    # -- see the module docstring above and
    # _emergency_stop/_hold_silent_cmd_vel below for why this exists
    # instead of reusing tele_cmd_vel for the safety monitor's reaction.
    silent_cmd_vel: Out[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._image_lock = threading.Lock()
        self._latest_image: Image | None = None
        self._llm_client: Any = None
        # State for the move()/stop_move() skills -- see their docstrings.
        self._move_lock = threading.Lock()
        self._move_thread: threading.Thread | None = None
        self._move_stop_event = threading.Event()
        # State for the background safety monitor -- see _start_safety_monitor.
        self._safety_thread: threading.Thread | None = None
        self._safety_stop_event = threading.Event()
        self._safety_stop_event.set()  # not running until begin_exploration starts it
        # World-frame points confirmed hazardous (see _emergency_stop) --
        # re-stamped into every future costmap tick by _stamp_hazards,
        # since CostMapper rebuilds its grid from scratch every cycle (no
        # occupancy state carried between ticks), so nothing marked on one
        # tick's OccupancyGrid instance would otherwise persist to the
        # next on its own. Deliberately NOT cleared by
        # reset_exploration_session() -- a physical hazard like glass is
        # still there after a session reset. The inherited frontier
        # heuristic reads self.latest_costmap, which _on_costmap sets to
        # the hazard-stamped grid, so a wall stamped here is automatically
        # excluded from both frontier scoring and A* planning.
        self._hazard_points: list[Vector3] = []
        # monotonic() deadline of the current silent_cmd_vel hold, so a
        # redundant safety-check re-detection during an active hold
        # doesn't restart it -- see _emergency_stop's re-entrancy guard.
        self._hazard_hold_until: float = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        if self.color_image.transport is not None:
            self.register_disposable(Disposable(self.color_image.subscribe(self._on_image)))
        else:
            logger.warning(
                "AgenticFrontierSelector: no color_image stream connected -- "
                "LLM will decide from map only. Add SimCameraModule to your blueprint."
            )
        # Built here, not __init__: Modules run in forkserver worker
        # processes, so the LLM SDK client gets constructed after fork.
        self._llm_client = self._build_llm_client()

    def _on_image(self, img: Image) -> None:
        with self._image_lock:
            self._latest_image = img

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        """Overrides WavefrontFrontierExplorer._on_costmap (which just does
        `self.latest_costmap = msg`): stamps any confirmed hazards
        (self._hazard_points) into the incoming costmap first, so this
        module's own frontier/obstacle-distance logic sees the patched
        version, and republishes it on hazard_costmap so
        ReplanningAStarPlanner (a separate module/process reading
        CostMapper's output directly) sees it too.

        Sets latest_costmap to the raw message FIRST, before anything that
        could fail -- if the hazard-stamping/republishing step raises, the
        exploration loop still has a usable costmap (the exploration loop
        only proceeds once this callback has run successfully at least
        once). Worst case degrades to "hazards aren't visually walled off
        this tick" instead of "exploration never runs at all"."""
        self.latest_costmap = msg
        try:
            patched = self._stamp_hazards(msg)
            self.latest_costmap = patched
            self.hazard_costmap.publish(patched)
        except Exception:
            logger.exception(
                "AgenticFrontierSelector: _on_costmap failed to stamp/republish hazards -- "
                "falling back to the raw (unpatched) costmap so exploration can still proceed. "
                "hazard_costmap consumers (e.g. ReplanningAStarPlanner) won't see this tick's "
                "wall until a future tick succeeds."
            )

    def _stamp_hazards(self, costmap: OccupancyGrid) -> OccupancyGrid:
        """Return a copy of costmap with every point in self._hazard_points
        burned in as an occupied disk of radius hazard_wall_radius_m.
        Re-projects world->grid fresh on every call rather than caching
        grid indices -- CostMapper recomputes its grid's bounds/origin from
        the current point cloud's extent on every tick, so a cached index
        from an earlier tick could silently point at the wrong cell on a
        later one."""
        if not self._hazard_points:
            return costmap
        patched = costmap.copy()
        cell_radius = max(1, int(round(self.config.hazard_wall_radius_m / patched.resolution)))
        yy, xx = np.ogrid[-cell_radius : cell_radius + 1, -cell_radius : cell_radius + 1]
        disk = (xx**2 + yy**2) <= cell_radius**2
        for point in self._hazard_points:
            g = patched.world_to_grid(point)
            gx, gy = int(round(g.x)), int(round(g.y))
            y0, y1 = max(0, gy - cell_radius), min(patched.height, gy + cell_radius + 1)
            x0, x1 = max(0, gx - cell_radius), min(patched.width, gx + cell_radius + 1)
            if y0 >= y1 or x0 >= x1:
                continue  # hazard falls entirely outside this tick's costmap bounds
            dy0, dx0 = y0 - (gy - cell_radius), x0 - (gx - cell_radius)
            region = patched.grid[y0:y1, x0:x1]
            region[disk[dy0 : dy0 + (y1 - y0), dx0 : dx0 + (x1 - x0)]] = CostValues.OCCUPIED
        return patched

    # -------------------------------------------------------------------
    # MCP-exposed skills. The inherited explore()/stop_exploration()
    # (WavefrontFrontierExplorer) are decorated @rpc, not @skill --
    # get_skills() (dimos/core/module.py) only surfaces @skill-decorated
    # methods to MCP. These wrap the inherited methods under the names an
    # MCP console actually needs, matching PatrollingModule's
    # start_patrol/stop_patrol pattern (dimos/navigation/patrolling/
    # module.py) -- extends, doesn't modify, dimos source.
    #
    # uses=[CAP_MOVEMENT] + lifecycle="background" means the MCP server
    # (dimos/agents/mcp/mcp_server.py, _handle_tools_call) transfers
    # ownership of the capability to a "tool-stream" the moment the
    # function returns -- it does NOT release on return like an "instant"
    # skill, so begin_exploration() must call self.start_tool() itself.
    # The RELEASE side needs no code here: WavefrontFrontierExplorer's own
    # _exploration_loop (wavefront_frontier_goal_selector.py:767) already
    # has `finally: self.stop_tool("begin_exploration")`, hardcoded to
    # this exact skill name for this subclassing pattern -- it reliably
    # releases the hold no matter how exploration ends.
    # -------------------------------------------------------------------

    @skill(uses=[CAP_MOVEMENT], lifecycle="background")
    def begin_exploration(self) -> str:
        """Start autonomous frontier exploration."""
        # Before any early return, so the movement hold is always carried
        # by a live tool-stream -- matches start_patrol's exact comment.
        self.start_tool("begin_exploration")
        started = self.explore()
        if started and self.config.safety_monitor_enabled:
            self._start_safety_monitor()
        return "Exploration started." if started else "Exploration already active."

    @skill
    def end_exploration(self) -> str:
        """Stop autonomous frontier exploration."""
        stopped = self.stop_exploration()
        return "Exploration stopped." if stopped else "Exploration was not active."

    def stop_exploration(self) -> bool:
        """Sets goal_reached_event before deferring to the base class's
        stop_exploration(). WavefrontFrontierExplorer's own exploration
        loop, while waiting for the current goal, blocks on
        self.goal_reached_event.wait(timeout=self.config.goal_timeout)
        (default 30s) and does NOT re-check stop_event/exploration_active
        during that wait -- its stop_exploration() joins the thread with
        only a short timeout, so if the thread is mid-wait, the CAP_MOVEMENT
        release in its finally block doesn't actually happen for up to the
        rest of goal_timeout. goal_reached_event is the same event the
        base class sets when a goal is genuinely reached -- setting it
        here too unblocks any in-progress wait immediately.
        """
        self.goal_reached_event.set()
        self._safety_stop_event.set()
        return super().stop_exploration()

    @skill(uses=[CAP_MOVEMENT], lifecycle="background")
    def move(
        self,
        x: float | str = 0.0,
        y: float | str = 0.0,
        yaw_rate: float | str = 0.0,
        duration: float | str = 1.0,
    ) -> str:
        """Directly drive the drone -- e.g. to reposition or steer away from
        something that looks unsafe (like glass) instead of waiting for the
        next autonomous frontier decision.

        Args:
            x: forward velocity in m/s (negative = backward)
            y: leftward velocity in m/s (negative = right)
            yaw_rate: turn rate in rad/s, positive = counterclockwise/left
            duration: how long to move, in seconds

        Publishing here takes over from autonomous exploration the same
        way the human teleop dashboard does -- see tele_cmd_vel's
        declaration above. If exploration is active, its current goal is
        cancelled; a fresh frontier will be picked once the exploration
        loop notices (goal_timeout).

        All arguments are clamped to safe ranges (max_move_duration_s,
        max_move_linear_mps, max_move_yaw_rate_rps) rather than rejected,
        so a slightly-too-aggressive request still does something
        reasonable instead of failing outright.

        Arguments accept str as well as float: some LLM providers emit
        numeric tool-call arguments as JSON strings, which a float-only
        schema rejects before this function ever runs -- widening to
        accept strings and parsing them below is the pragmatic fix.

        Refuses re-invocation while a move is already in progress (tell
        the caller to call stop_move first) rather than silently
        superseding the old thread -- matches PatrollingModule.
        start_patrol()'s recommended pattern, and avoids a race where the
        old thread's own stop_tool("move") call could fire after a newer
        invocation had already re-stamped the tool-stream, releasing the
        new invocation's capability hold instead of the old one's.
        """
        with self._move_lock:
            if self._move_thread is not None and self._move_thread.is_alive():
                return "Already moving. Call stop_move first, then retry."

            # Before any early return, so the CAP_MOVEMENT hold is always
            # carried by a live tool-stream -- see the class-level note
            # above begin_exploration.
            self.start_tool("move")

            x = float(x)
            y = float(y)
            yaw_rate = float(yaw_rate)
            duration = float(duration)

            x = max(-self.config.max_move_linear_mps, min(self.config.max_move_linear_mps, x))
            y = max(-self.config.max_move_linear_mps, min(self.config.max_move_linear_mps, y))
            yaw_rate = max(
                -self.config.max_move_yaw_rate_rps,
                min(self.config.max_move_yaw_rate_rps, yaw_rate),
            )
            duration = max(0.1, min(self.config.max_move_duration_s, duration))

            twist = Twist(linear=Vector3(x, y, 0.0), angular=Vector3(0.0, 0.0, yaw_rate))

            self._move_stop_event = threading.Event()
            stop_event = self._move_stop_event
            self._move_thread = threading.Thread(
                target=self._move_loop, args=(twist, duration, stop_event), daemon=True
            )
            self._move_thread.start()

        logger.info(
            "AgenticFrontierSelector: move(x=%.2f, y=%.2f, yaw_rate=%.2f, duration=%.1fs)",
            x, y, yaw_rate, duration,
        )
        return f"Moving: x={x:.2f}m/s y={y:.2f}m/s yaw_rate={yaw_rate:.2f}rad/s for {duration:.1f}s"

    @skill
    def stop_move(self) -> str:
        """Immediately stop any in-progress move() command.

        Deliberately not named stop() -- that name would silently shadow
        WavefrontFrontierExplorer.stop() (an @rpc-decorated MODULE
        LIFECYCLE method: stop_exploration() then Module.stop(), the full
        shutdown of disposables/RPC transport/worker loop), which returns
        None immediately rather than running this method's actual body.
        Always check for a name collision with inherited @rpc lifecycle
        methods before naming a new skill.
        """
        with self._move_lock:
            was_moving = self._move_thread is not None and self._move_thread.is_alive()
            self._move_stop_event.set()
            if self._move_thread is not None and self._move_thread.is_alive():
                self._move_thread.join(timeout=2.0)
            self._move_thread = None
        # Explicit zero command so the robot halts right away rather than
        # waiting out DimSim's ~500ms cmd_vel watchdog.
        self.tele_cmd_vel.publish(Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0)))
        return "Stopped." if was_moving else "Was not moving."

    def _move_loop(self, twist: Twist, duration: float, stop_event: threading.Event) -> None:
        """Republish `twist` every 0.1s until `duration` elapses or
        `stop_event` is set -- DimSim's physics.ts has a ~500ms cmd_vel
        watchdog, so a single publish would only move the robot briefly
        regardless of the requested duration.

        Safe to call self.stop_tool from this background thread (not the
        skill call's own thread) -- unlike start_tool, stop_tool's
        docstring carries no main-thread requirement, and since move()
        refuses re-invocation while already running (see its docstring),
        this is always the same invocation that opened the stream."""
        deadline = time.monotonic() + duration
        while not stop_event.is_set() and time.monotonic() < deadline:
            self.tele_cmd_vel.publish(twist)
            stop_event.wait(0.1)
        self.tele_cmd_vel.publish(Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0)))
        self.stop_tool("move")

    # -------------------------------------------------------------------
    # Background safety monitor. The camera is only ever consulted at the
    # moment a frontier TARGET is chosen; once a goal is handed to the A*
    # planner, path execution is purely geometric -- if the depth-derived
    # costmap doesn't perceive glass as an obstacle at all (a real depth
    # sensor facing real glass has the same blind spot), a target can be
    # geometrically fine while the path to it still isn't. This thread
    # runs independently of the frontier-decision cycle, on its own timer
    # (safety_check_interval_s), asking a small camera-only question
    # (_SAFETY_CHECK_PROMPT) via the same LLM client/provider already
    # configured.
    # -------------------------------------------------------------------

    def _start_safety_monitor(self) -> None:
        """Spawn the safety-monitor thread if it isn't already running.
        Called from begin_exploration() after explore() succeeds; stopped
        via self._safety_stop_event, set in stop_exploration()."""
        if self._safety_thread is not None and self._safety_thread.is_alive():
            return
        self._safety_stop_event = threading.Event()
        stop_event = self._safety_stop_event
        self._safety_thread = threading.Thread(
            target=self._safety_monitor_loop, args=(stop_event,), daemon=True
        )
        self._safety_thread.start()

    def _safety_monitor_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.wait(self.config.safety_check_interval_s):
            with self._image_lock:
                image = self._latest_image
            if image is None:
                continue
            try:
                result = self._call_llm(_SAFETY_CHECK_PROMPT, None, image)
            except Exception:
                logger.exception(
                    "AgenticFrontierSelector: safety-check LLM call failed, skipping this tick"
                )
                continue
            if result.get("hazard_ahead"):
                distance = result.get("estimated_distance_m")
                # Only a genuinely close hazard (see hazard_stop_distance_m)
                # actually triggers a stop; a distant one is logged but
                # otherwise ignored, so exploration can keep making real
                # progress until it's actually close.
                if distance is not None and distance > self.config.hazard_stop_distance_m:
                    logger.info(
                        "AgenticFrontierSelector: safety monitor sees a possible hazard but "
                        "estimates it's still ~%.1fm away (stop threshold %.1fm) -- not "
                        "stopping yet. Reasoning: %s",
                        distance, self.config.hazard_stop_distance_m, result.get("reasoning"),
                    )
                    continue
                logger.warning(
                    "AgenticFrontierSelector: safety monitor detected a hazard %s -- "
                    "stopping immediately. Reasoning: %s",
                    f"~{distance:.1f}m away" if distance is not None else "at an unestimated distance",
                    result.get("reasoning"),
                )
                self._emergency_stop(estimated_distance_m=distance)
                # Wait out whatever's left of the hold here, on top of
                # this loop's own next interval-wait, so the robot always
                # gets at least one FULL safety_check_interval_s of
                # genuinely free, unmonitored time after every hold ends.
                remaining_hold = self._hazard_hold_until - time.monotonic()
                if remaining_hold > 0:
                    stop_event.wait(remaining_hold)

    def _emergency_stop(self, estimated_distance_m: float | None = None) -> None:
        """Immediately halt the drone from the safety monitor's own thread.
        Deliberately bypasses the move()/stop_move() skills entirely -- this
        is an internal watchdog action, not an LLM tool call, so it isn't
        gated by the MCP capability system and can act without waiting for
        a tool-stream.

        Publishes on silent_cmd_vel rather than tele_cmd_vel: MovementManager's
        _on_teleop unconditionally calls _cancel_goal() for ANY tele_cmd_vel
        message, which broadcasts stop_movement=True, and
        WavefrontFrontierExplorer._on_stop_movement treats that as "a
        human took over" and runs the REAL stop_exploration() (full
        teardown) rather than a soft pause -- correct for move()/real
        human teleop, wrong for this internal, should-keep-exploring
        caller. silent_cmd_vel claims the same teleop-priority window
        (nav_cmd_vel gets suppressed) WITHOUT the stop_movement broadcast,
        so exploration keeps running and picks a fresh, hazard-aware
        frontier once goal_reached_event wakes the loop. Held via
        _hold_silent_cmd_vel for hazard_stop_hold_s rather than a single
        publish, since silent_cmd_vel's priority window is time-limited
        (tele_cooldown_sec, 1.0s default in MovementManager).

        Commands a real backup+turn maneuver (see _hold_silent_cmd_vel)
        rather than just holding position -- with almost nothing mapped
        near spawn, a fresh frontier pick alone doesn't reliably lead
        anywhere different, so the drone needs to actively move away.

        Also marks the estimated hazard location as a permanent wall (see
        _stamp_hazards) so future planning -- both frontier selection AND
        the A* planner -- routes around it instead of just not targeting
        it directly. Position is a rough forward-projection from the
        robot's current pose (hazard_projection_distance_m) since the
        safety check itself has no range/bearing, only "hazard ahead".

        Re-entrancy guard (now < self._hazard_hold_until below): a
        redundant detection during an already-active hold is a no-op --
        without this, a hold shorter than the check interval would let
        the still-facing-the-hazard camera re-trigger a fresh hold before
        the previous one expires, permanently freezing the robot in
        place. This guarantees a hold, once started, always runs its FULL
        duration uninterrupted.

        estimated_distance_m: the LLM's own rough distance estimate from
        _SAFETY_CHECK_PROMPT, when it gave a usable one -- preferred over
        hazard_projection_distance_m's fixed guess for placing the wall.
        """
        now = time.monotonic()
        if now < self._hazard_hold_until:
            logger.info(
                "AgenticFrontierSelector: hazard re-detected during an active silent_cmd_vel hold "
                "(%.1fs remaining) -- ignoring, not restarting the hold",
                self._hazard_hold_until - now,
            )
            return
        self._hazard_hold_until = now + self.config.hazard_stop_hold_s
        self._hold_silent_cmd_vel()
        if self.latest_odometry is not None:
            pose = self.latest_odometry
            projection_distance = (
                estimated_distance_m
                if estimated_distance_m is not None
                else self.config.hazard_projection_distance_m
            )
            hazard_point = Vector3(
                pose.position.x + projection_distance * math.cos(pose.yaw),
                pose.position.y + projection_distance * math.sin(pose.yaw),
                0.0,
            )
            self._hazard_points.append(hazard_point)
            logger.info(
                "AgenticFrontierSelector: marking estimated hazard position (%.2f, %.2f) as a "
                "permanent wall (%d hazard(s) now tracked)",
                hazard_point.x, hazard_point.y, len(self._hazard_points),
            )
        else:
            logger.warning(
                "AgenticFrontierSelector: hazard detected but no odometry yet -- can't estimate "
                "its position, so it won't be walled off (will still stop movement and pick a "
                "fresh frontier)"
            )
        # Same wakeup used by the stop_exploration() fix above -- unblocks
        # the exploration loop's goal wait immediately instead of leaving
        # it to sit blocked on the aborted goal for up to goal_timeout
        # (default 30s), so a fresh frontier decision happens right away.
        self.goal_reached_event.set()

    def _hold_silent_cmd_vel(self) -> None:
        """Republish a reactive command every 0.1s (same cadence as
        _move_loop's republish, for the same DimSim cmd_vel watchdog
        reason) for hazard_stop_hold_s total -- three phases, each
        capped so the total never exceeds the hold budget:
        1. Back straight away (negative x) for hazard_backup_duration_s.
        2. Turn in place (no translation) by hazard_turn_angle_rad at
           hazard_turn_rate_rps, so the camera ends up facing away from
           the hazard instead of straight at it.
        3. Hold position (zero Twist) for whatever remains of the hold.

        Backs away before turning, and turns a full ~180 degrees rather
        than guessing a partial left/right: a single forward-facing camera
        frame gives no information about which side is actually clear, but
        backing away is guaranteed to increase distance from whatever's
        directly ahead regardless of room layout, and facing exactly
        opposite the original heading is guaranteed to point the camera at
        wherever the drone just came from, not at the hazard.

        Falls back to a plain tele_cmd_vel stop (no backup maneuver, and
        this WILL abort the whole exploration session via the
        stop_movement cascade described in _emergency_stop's docstring)
        whenever self.silent_cmd_vel.transport is None -- i.e. if
        MovementManager.silent_cmd_vel isn't actually wired up in the
        current dimos dependency -- so a hazard always at least halts the
        robot either way."""
        if self.silent_cmd_vel.transport is None:
            logger.warning(
                "AgenticFrontierSelector: silent_cmd_vel has no transport (MovementManager."
                "silent_cmd_vel not live in the current dimos dependency) -- falling back to a "
                "plain tele_cmd_vel stop (no backup maneuver), which WILL abort the whole "
                "exploration session (see _emergency_stop's docstring)."
            )
            self.tele_cmd_vel.publish(Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0)))
            return

        def _hold() -> None:
            backup = Twist(
                linear=Vector3(-self.config.hazard_backup_speed_mps, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0),
            )
            turn = Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, self.config.hazard_turn_rate_rps),
            )
            stopped = Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0))

            total = self.config.hazard_stop_hold_s
            turn_duration = self.config.hazard_turn_angle_rad / max(self.config.hazard_turn_rate_rps, 1e-6)
            backup_end = min(self.config.hazard_backup_duration_s, total)
            turn_end = min(backup_end + turn_duration, total)

            start = time.monotonic()
            deadline = start + total
            while time.monotonic() < deadline:
                elapsed = time.monotonic() - start
                if elapsed < backup_end:
                    twist = backup
                elif elapsed < turn_end:
                    twist = turn
                else:
                    twist = stopped
                self.silent_cmd_vel.publish(twist)
                time.sleep(0.1)
            self.silent_cmd_vel.publish(stopped)

        threading.Thread(target=_hold, daemon=True).start()

    # -------------------------------------------------------------------
    # LLM plumbing -- isolated so it's easy to swap providers.
    # -------------------------------------------------------------------

    def _build_llm_client(self) -> Any:
        if self.config.llm_provider == "openai":
            import openai  # requires OPENAI_API_KEY in the environment

            return openai.OpenAI()
        elif self.config.llm_provider == "anthropic":
            import anthropic  # requires ANTHROPIC_API_KEY in the environment

            return anthropic.Anthropic()
        elif self.config.llm_provider == "gemini":
            # pip install google-genai. genai.Client() reads GEMINI_API_KEY
            # from the environment automatically -- no need to pass it.
            from google import genai

            return genai.Client()
        elif self.config.llm_provider == "groq":
            # Groq's API is OpenAI-compatible, so this reuses the openai
            # package already imported for the "openai" branch above
            # rather than adding a new dependency -- just pointed at
            # Groq's base_url with GROQ_API_KEY.
            import openai

            return openai.OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ["GROQ_API_KEY"],
            )
        raise ValueError(f"unknown llm_provider: {self.config.llm_provider}")

    def _image_block(self, b64: str, media_type: str) -> dict[str, Any]:
        """One image content block, shaped for OpenAI/Groq/Anthropic (Gemini
        is built separately in _call_llm since its SDK takes raw bytes, not
        base64 dicts). Groq's chat completions API mirrors OpenAI's
        image_url content-block shape."""
        if self.config.llm_provider in ("openai", "groq"):
            return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}

    def _call_llm(
        self, prompt: str, map_image_png: bytes | None, camera_image: Image | None
    ) -> dict[str, Any]:
        camera_jpeg: bytes | None = None
        if camera_image is not None:
            try:
                # Image.to_base64() handles BGR/RGB conversion and JPEG
                # encoding itself, and max_width keeps the request small.
                # We decode back to bytes here for providers (Gemini) that
                # want raw bytes rather than a base64 string.
                camera_b64 = camera_image.to_base64(
                    quality=80, max_width=self.config.camera_max_width_px
                )
                camera_jpeg = base64.b64decode(camera_b64)
            except Exception:
                logger.exception(
                    "AgenticFrontierSelector: failed to encode camera frame, "
                    "continuing with map image only"
                )

        if self.config.llm_provider == "gemini":
            from google.genai import types

            contents: list[Any] = [prompt]
            if map_image_png is not None:
                contents.append(types.Part.from_bytes(data=map_image_png, mime_type="image/png"))
            if camera_jpeg is not None:
                contents.append(types.Part.from_bytes(data=camera_jpeg, mime_type="image/jpeg"))

            resp = self._llm_client.models.generate_content(
                model=self.config.llm_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.config.llm_temperature,
                    response_mime_type="application/json",
                ),
            )
            raw = resp.text
            return json.loads(raw)

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if map_image_png is not None:
            content.append(self._image_block(base64.b64encode(map_image_png).decode(), "image/png"))
        if camera_jpeg is not None:
            content.append(self._image_block(base64.b64encode(camera_jpeg).decode(), "image/jpeg"))

        if self.config.llm_provider in ("openai", "groq"):
            resp = self._llm_client.chat.completions.create(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
            )
            raw = resp.choices[0].message.content
        else:  # anthropic
            resp = self._llm_client.messages.create(
                model=self.config.llm_model,
                max_tokens=200,
                temperature=self.config.llm_temperature,
                messages=[{"role": "user", "content": content}],
            )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

        return json.loads(raw)
