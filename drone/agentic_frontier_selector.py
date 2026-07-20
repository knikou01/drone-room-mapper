#!/usr/bin/env python3
"""
A subclass of dimos.navigation.frontier_exploration.
wavefront_frontier_goal_selector.WavefrontFrontierExplorer. Does not modify
that file or any other dimos source for its own sake -- it only imports and
extends the class.

One exception, added 2026-07-15: the background safety monitor's emergency
stop (see _emergency_stop below) needs a way to command the robot
immediately -- originally just a halt, now a real reactive backup
maneuver (see below) -- without dimos's existing
WavefrontFrontierExplorer/MovementManager stack mistaking it for a human
teleop override and aborting the entire exploration session (confirmed
live -- see _emergency_stop's docstring). No existing dimos stream drew
that distinction, so MovementManager gained one small, additive, optional
input (silent_cmd_vel: In[Twist],
dimos/navigation/movement_manager/movement_manager.py) that claims teleop
priority WITHOUT the human-override side effect. Originally named
silent_stop and Bool-typed (always zero velocity); generalized the same
day to carry a real Twist once "just stop and wait for a new frontier
pick" turned out to be insufficient (see _emergency_stop's docstring).
Nothing about WavefrontFrontierExplorer itself was touched.

ARCHITECTURE CHANGE (2026-07-20): frontier SELECTION no longer consults an
LLM at all -- get_exploration_goal() is gone entirely, so this class now
inherits WavefrontFrontierExplorer's plain geometric heuristic unmodified
for every single frontier decision. Reasoning, direct from the user after
a run of live tests (wall-collider overlap, LLM rate-limit retry loop,
deprecated model, reasoning-model latency) kept surfacing NEW instability
each time the previous one was fixed: "the A* exploration seems better
than the agentic" -- the LLM-per-decision architecture was the common
thread across most of that instability, not any single bug. The plain
heuristic is already hazard-aware for free: self.latest_costmap is the
HAZARD-STAMPED grid (see _on_costmap/_stamp_hazards below, unchanged by
this pivot), so a confirmed hazard is already occupied-cell territory by
the time the base class's own candidate scoring runs -- no LLM judgment
call was actually needed to keep frontier selection out of a wall.
The LLM's role is now narrower and REQUEST-DRIVEN rather than
decision-per-step: begin_exploration/end_exploration (start/stop on
request), move/stop_move (direct steering on request), and the background
safety monitor (still watches the live camera continuously while a goal
is being driven and reacts to a real hazard -- see "Background safety
monitor" below, untouched by this pivot). There is no more per-candidate
"should I pick this one" LLM call, and therefore no more map-render/
candidate-image prompt, clearance pre-filter, reject_all bookkeeping, or
LLM circuit breaker for frontier decisions -- all removed as dead code
rather than left disabled, since the base class's own scoring/no-gain
logic is what actually decides now, unconditionally.
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
    # unchanged -- the geometric scoring stays exactly as already tuned, and
    # is now the ONLY thing that picks frontiers (see the module docstring's
    # 2026-07-20 architecture-change note) -- no LLM-specific candidate
    # filtering fields here anymore.
    # Safety clamps for the move() skill -- see its docstring. Small
    # duration cap forces shorter, repeated commands rather than one long
    # blind dash, giving more chances to re-observe between moves.
    max_move_duration_s: float = 5.0
    max_move_linear_mps: float = 0.5
    max_move_yaw_rate_rps: float = 1.0
    # Added 2026-07-16: lets the background safety monitor be turned off
    # entirely -- e.g. for testing the rest of the exploration pipeline
    # (frontier selection, mapping, movement) on a scene with no glass at
    # all (apt_no_glass), where the monitor is pure overhead and a source
    # of false positives (confirmed live: it flagged an ordinary door
    # frame as a hazard, reasoning "may contain a transparent glass
    # pane" -- overly cautious on ambiguous-looking architecture with no
    # actual glass anywhere in the scene). Defaults to True (unchanged
    # existing behavior) -- meant to be turned off deliberately per run,
    # not left off by default, since the whole point of the monitor is to
    # catch real hazards on scenes that do have them (e.g. apt).
    safety_monitor_enabled: bool = True
    # How often (seconds) the background safety monitor checks the live
    # camera frame for a hazard directly ahead WHILE a goal is being
    # driven -- independent of and much more frequent than frontier
    # decisions (which only look at the camera once, when a target is
    # first chosen). See AgenticFrontierSelector's module docstring
    # (safety monitor section) for why this exists: a chosen target can
    # be geometrically fine while the path to it still isn't, if the
    # depth costmap doesn't perceive the hazard (e.g. glass) at all.
    safety_check_interval_s: float = 3.0
    # Only actually stop for a hazard once the LLM's own distance estimate
    # (see _SAFETY_CHECK_PROMPT's estimated_distance_m) is within this
    # many meters -- CONFIRMED BUG, FIXED (2026-07-15): without this, a
    # hazard was treated as urgent the moment it was merely VISIBLE in
    # frame, even from meters away, and a live run showed the drone get
    # stopped almost immediately near its spawn point and never make any
    # real progress. NOT set anywhere near the drone's actual physical
    # stopping margin (e.g. 30cm) -- the robot drives at ~0.55 m/s
    # (ReplanningAStarPlanner's LocalPlanner default, unoverridden in this
    # blueprint) and this check only runs every safety_check_interval_s
    # (3.0s default), so in the worst case it can cover ~1.65m between two
    # checks. A stopping distance smaller than that risks a hazard judged
    # "not close yet" being passed straight through before the next check
    # ever fires. 2.0m gives real margin (worst-case travel between
    # checks, plus room for the LLM's estimate being imprecise) while
    # still being much closer than "glass merely visible in frame".
    hazard_stop_distance_m: float = 2.0
    # Rough forward-projection distance (meters) used to estimate a
    # detected hazard's world position when the LLM couldn't give a usable
    # distance estimate -- see _stamp_hazards/_emergency_stop. Prefer the
    # LLM's own estimated_distance_m when available (more accurate); this
    # is only the fallback guess (project forward from the robot's current
    # pose along its current heading) for the rare frame where it can't be
    # judged. Good enough to keep the drone from repeatedly approaching
    # the same spot, not a substitute for real ranging.
    hazard_projection_distance_m: float = 1.5
    # Radius (meters) of the synthetic occupied patch stamped into the
    # costmap at each confirmed hazard's estimated position.
    hazard_wall_radius_m: float = 0.5
    # How long (seconds) the safety monitor holds silent_cmd_vel priority
    # after an emergency stop -- see _emergency_stop/_hold_silent_cmd_vel.
    # Needs to comfortably fit the full backup + turn-away maneuver below
    # (~1.5s + ~3.1s at the defaults) so the exploration loop has time to
    # publish a fresh (hazard-avoiding) goal -- which itself cancels the
    # old one -- before priority lapses and nav_cmd_vel would otherwise
    # resume driving toward the just-detected hazard. Frontier selection is
    # now plain-heuristic (no LLM round trip to wait out, see the module
    # docstring's 2026-07-20 note), so this margin is comfortably more than
    # needed today -- left as-is rather than tightened, since it's still
    # just a hold duration, not a hot path. If the maneuver ends up shorter
    # than this, the remainder is held stationary; if this is too short to
    # fit the maneuver, the maneuver is truncated gracefully rather than
    # exceeding the budget (see _hold_silent_cmd_vel).
    hazard_stop_hold_s: float = 6.0
    # CONFIRMED BUG, FIXED (2026-07-15): _emergency_stop used to just halt
    # and wait for the exploration loop to pick a new frontier -- but a
    # live test showed a hazard less than hazard_stop_distance_m from the
    # drone's SPAWN position (i.e. before any real exploration/mapping had
    # happened yet) left it standing still indefinitely: with almost
    # nothing mapped yet, every candidate frontier direction can look
    # similarly (un)attractive, so a fresh pick doesn't reliably lead
    # anywhere different. User's diagnosis, direct and correct: it should
    # actively move away, not just wait. Backs straight away (negative x)
    # for this many seconds first -- rather than immediately turning --
    # since backing away doesn't require knowing which side is clear and
    # creates some buffer distance before the drone's footprint swings
    # around during the turn phase below.
    hazard_backup_speed_mps: float = 0.3
    hazard_backup_duration_s: float = 1.5
    # CONFIRMED BUG, FIXED (2026-07-15), same day: backing away alone
    # wasn't enough either -- a live run showed the drone back up once,
    # then sit still for 30+ seconds with the safety monitor repeatedly
    # reporting the SAME window ~2.5m away, because it was still facing
    # it the whole time (separately, A* was struggling to route it
    # anywhere else -- see the replanning_a_star known-issues notes -- but
    # even once that's sorted, sitting there staring at the hazard is
    # still wrong). User's suggestion, direct and correct: turn to face
    # away so the camera isn't looking at it anymore. Rotates in place
    # (no translation) by hazard_turn_angle_rad (default pi, i.e. ~180
    # degrees) at hazard_turn_rate_rps after the backup phase -- a full
    # about-face rather than a partial turn, since we have no information
    # about which partial direction is actually clear either; facing
    # exactly opposite the original heading is guaranteed to point the
    # camera at wherever the drone just came from, not at the hazard.
    hazard_turn_rate_rps: float = 1.0
    hazard_turn_angle_rad: float = math.pi
    # Only used for the safety monitor's own periodic camera check now
    # (_SAFETY_CHECK_PROMPT) -- no map render/candidate-image prompt exists
    # anymore since frontier selection no longer calls the LLM at all (see
    # the module docstring's 2026-07-20 note).
    camera_max_width_px: int = 768
    llm_provider: Literal["openai", "anthropic", "gemini", "groq"] = "gemini"
    llm_model: str = "gemini-3.5-flash"
    llm_temperature: float = 0.2


# Runs continuously (every safety_check_interval_s) WHILE a goal is being
# driven -- the only remaining LLM prompt in this module, now that frontier
# SELECTION itself is plain-heuristic (see the module docstring's
# 2026-07-20 architecture-change note; the old per-decision _SELECT_PROMPT
# and its _NO_GAIN_SUFFIX are gone). Camera-only, no map render -- kept
# fast and cheap since it runs on its own timer regardless of whether a
# frontier decision is happening. See AgenticFrontierSelector's
# safety-monitor section for why this check exists at all.
#
# CONFIRMED BUG, FIXED (2026-07-15): originally had no distance/proximity
# concept at all -- "is there a hazard in this frame" would fire the
# moment glass was merely VISIBLE, even from meters away, well before it
# was actually dangerous. Confirmed live: a real run showed the drone
# stopped almost immediately near its spawn point and never made
# meaningful progress -- the safety monitor kept re-triggering on the
# same distant window, over and over, because "visible" was being treated
# as "imminent". Added estimated_distance_m so _emergency_stop only fires
# once a hazard is actually close (see hazard_stop_distance_m), not
# merely in frame.
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
    heuristic -- get_exploration_goal is not overridden at all (see the
    module docstring's 2026-07-20 architecture-change note for why the
    earlier per-decision LLM selection was removed). What this class
    actually adds over the base explorer:
    - begin_exploration/end_exploration/move/stop_move skills (MCP-exposed
      wrappers, see their docstrings for the capability-leak bugs fixed
      getting there).
    - Hazard-wall stamping (_stamp_hazards/hazard_costmap) so a confirmed
      hazard becomes a real occupied cell for both frontier scoring and
      A* path planning, not just an excluded target.
    - A background safety monitor that watches the live camera
      continuously WHILE a goal is being driven (see "Background safety
      monitor" below) and reacts to a real hazard with a backup+turn
      maneuver -- independent of frontier decisions, which are now purely
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
    # any teleop-style source), NOT the kind of collision fixed for
    # color_image/dimsim_color_image (that was an internal-topic-vs-
    # external-DimSim-bridge collision -- different situation). Publishing
    # here reuses MovementManager's existing teleop-priority-over-A* and
    # goal-cancellation logic (_on_teleop -> _cancel_goal(), see
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
    # unreachable frontier target -- the raw depth-derived costmap alone
    # doesn't perceive glass at all, so without this, a path to some OTHER
    # frontier could still cut straight through a spot the safety monitor
    # already stopped for. New name (not "global_costmap") deliberately,
    # so autoconnect doesn't collide it with CostMapper's own output of
    # that name -- this module still consumes that one directly via the
    # inherited global_costmap: In[OccupancyGrid], unremapped.
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
        # occupancy state carried between ticks, confirmed from its
        # source), so nothing marked on one tick's OccupancyGrid instance
        # would otherwise persist to the next on its own. Deliberately NOT
        # cleared by reset_exploration_session() -- a physical hazard like
        # glass is still there after a session reset. This is the ONLY
        # hazard-exclusion mechanism now (see the module docstring's
        # 2026-07-20 note) -- the inherited frontier heuristic reads
        # self.latest_costmap, which _on_costmap sets to the
        # hazard-stamped grid, so a wall stamped here is automatically
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
        CostMapper's output directly) sees it too -- see hazard_costmap's
        declaration above for why a relay is needed at all.

        CONFIRMED BUG, FIXED (2026-07-15): a live run showed the
        exploration loop NEVER publish a single frontier goal in 2.5+
        minutes -- not even in the 42 seconds before any hazard existed,
        ruling out _stamp_hazards' hazard-processing branch specifically.
        WavefrontFrontierExplorer._run_exploration_loop only proceeds past
        `if self.latest_costmap is None: ...; continue` once this callback
        has run successfully at least once; it was never reaching the LLM
        frontier-selection step at all (no step=N logs, no "LLM call
        failed" logs either -- get_exploration_goal was never being
        called). Leading theory, not yet confirmed with a live traceback
        (couldn't reproduce standalone -- unit tests always mocked
        .publish(), never exercising the real LCM transport for the new
        hazard_costmap stream): if this callback raises, DimOS's reactivex
        subscription can silently terminate rather than retrying, which
        would permanently strand self.latest_costmap at None -- explaining
        total silence rather than a visible crash. Sets latest_costmap to
        the raw message FIRST, before anything that could fail, so a bug
        in the hazard-stamping/republishing path below can no longer
        starve the base exploration loop of a usable costmap -- the worst
        case degrades to "hazards aren't visually walled off this tick"
        instead of "exploration never runs at all". Also wraps the
        risky part in try/except with a full logged traceback (silent
        failure is exactly what made this so hard to diagnose from the
        live log alone) so a real recurrence is now impossible to miss."""
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
        the current point cloud's extent on every tick (confirmed from its
        source), so a cached index from an earlier tick could silently
        point at the wrong cell on a later one."""
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
    # MCP-exposed skills. CONFIRMED BUG, FIXED (2026-07-08): the inherited
    # explore()/stop_exploration() (WavefrontFrontierExplorer) are decorated
    # @rpc, not @skill -- get_skills() (dimos/core/module.py) only surfaces
    # @skill-decorated methods to MCP, so an LLM console instructed to call
    # "begin_exploration"/"end_exploration" (as sim_agentic_blueprint.py's
    # system prompt does) had no matching tool at all. These wrap the
    # inherited methods under the names an MCP console actually needs,
    # matching PatrollingModule's start_patrol/stop_patrol pattern exactly
    # (dimos/navigation/patrolling/module.py) -- extends, doesn't modify,
    # dimos source, same discipline as the rest of this file.
    #
    # CONFIRMED BUG, FIXED (2026-07-09), second one: getting the @skill
    # decorator right wasn't enough. uses=[CAP_MOVEMENT] + lifecycle=
    # "background" means the MCP server (dimos/agents/mcp/mcp_server.py,
    # _handle_tools_call) transfers ownership of the capability to a
    # "tool-stream" the moment the function returns -- it does NOT release
    # on return like an "instant" skill. begin_exploration() never called
    # self.start_tool() at all, so CAP_MOVEMENT was acquired and then
    # permanently leaked -- confirmed live: end_exploration reported
    # "Exploration stopped." successfully, but move() kept getting refused
    # with "capability 'movement' is held by 'begin_exploration'" long
    # after. Fix is exactly one line (start_tool below) -- the RELEASE
    # side needs no code here at all: WavefrontFrontierExplorer's own
    # _exploration_loop (wavefront_frontier_goal_selector.py:767) already
    # has `finally: self.stop_tool("begin_exploration")`, hardcoded to
    # this exact skill name, specifically designed for this subclassing
    # pattern -- it reliably releases the hold no matter how exploration
    # ends (explicit stop, no-gain, consecutive-failures, or error),
    # requiring nothing from end_exploration() beyond what it already does.
    # (Confirmed by testing: an earlier version of this fix added a
    # redundant stop_exploration() override calling stop_tool() a second
    # time -- harmless in production since stop_tool is a safe no-op when
    # already released, exactly as wavefront_frontier_goal_selector.py's
    # own docstring anticipates, but unnecessary, so removed.)
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
        """CONFIRMED BUG, FIXED (2026-07-13): WavefrontFrontierExplorer's
        own exploration loop, while waiting for the current goal, blocks
        on self.goal_reached_event.wait(timeout=self.config.goal_timeout)
        (default 30s) -- and does NOT re-check stop_event/exploration_active
        during that wait (wavefront_frontier_goal_selector.py:823). Its
        stop_exploration() sets stop_event and joins the thread, but only
        with DEFAULT_THREAD_JOIN_TIMEOUT=2.0s (dimos/constants.py) -- if
        the thread is mid-wait when that join gives up, stop_exploration()
        still reports success (exploration_active/stop_event are already
        flipped synchronously), but the thread -- and therefore the
        CAP_MOVEMENT release in its finally block -- doesn't actually exit
        for up to the REST of the 30s goal_timeout. Confirmed live: a run
        showed end_exploration report "Exploration stopped." in ~2s (the
        join timeout, not real completion), while move() kept getting
        refused with "held by begin_exploration" for another ~25 seconds,
        until "Goal timeout after 30 seconds" finally let the loop notice
        the stop request. Fix: goal_reached_event is the same event the
        base class sets when a goal is genuinely reached
        (wavefront_frontier_goal_selector.py:196) -- setting it here too,
        before the real stop_exploration() runs its join, unblocks any
        in-progress wait immediately so the loop notices
        exploration_active=False on its very next check instead of
        blocking for up to 30 more seconds. Harmless: exploration is
        ending immediately after regardless of whether the loop logs
        "goal reached" or "goal timeout" on its way out.
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

        CONFIRMED BUG, FIXED (2026-07-09): arguments accept str as well as
        float. With float-only type hints, a real console run with
        llama-4-scout-17b-16e-instruct (via Groq) had the model emit all
        four arguments as JSON strings ("-0.5" instead of -0.5) -- Groq's
        own API rejected the tool call before this function ever ran
        ("expected number, but got string"), so there was nothing to catch
        inside the function body; the fix has to widen what the
        auto-generated schema accepts. This is a real reliability quirk of
        that specific model's structured tool-calling, not a schema bug --
        widening to accept strings and parsing them below is a pragmatic
        mitigation, not a sign the original float typing was wrong.
        CONFIRMED BUG, FIXED (2026-07-09), a second time: originally,
        calling move() while one was already running would stop the old
        thread and start a fresh one. That raced with the fix above: the
        OLD thread's own stop_tool("move") call (see _move_loop) could
        fire *after* a newer invocation had already re-stamped the
        tool-stream with its own acquire token, wrongly releasing the
        NEW invocation's capability hold instead of the old one's.
        PatrollingModule.start_patrol()'s own docstring explicitly
        recommends the safer pattern adopted here instead: refuse
        re-invocation while already running (tell the caller to call the
        stop tool first), matching "Cannot start 'move': capability
        ... is held by ...". No silent supersession, no race.
        """
        with self._move_lock:
            if self._move_thread is not None and self._move_thread.is_alive():
                return "Already moving. Call stop_move first, then retry."

            # Before any early return, so the CAP_MOVEMENT hold is always
            # carried by a live tool-stream -- see the class-level note
            # above begin_exploration for why this matters (a second
            # instance of the same leaked-capability bug, fixed the same
            # way).
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

        CONFIRMED BUG, FIXED (2026-07-09): this was originally named
        stop(), which silently shadowed WavefrontFrontierExplorer.stop()
        (dimos/navigation/frontier_exploration/
        wavefront_frontier_goal_selector.py:181) -- an @rpc-decorated
        MODULE LIFECYCLE method (stop_exploration() then Module.stop(),
        the full shutdown: disposables, RPC transport, worker loop).
        Confirmed live: the console called "stop" and got response=None
        after 0.001s -- consistent with Module.stop() (no return value,
        fast synchronous teardown) rather than this method's actual body
        (thread join + publish + a real string return, ~1-50ms in
        testing). Renamed to avoid the collision entirely, same discipline
        already applied to begin_exploration/end_exploration for
        explore()/stop_exploration() -- always check for a name collision
        with inherited @rpc lifecycle methods before naming a new skill.
        """
        with self._move_lock:
            was_moving = self._move_thread is not None and self._move_thread.is_alive()
            self._move_stop_event.set()
            if self._move_thread is not None and self._move_thread.is_alive():
                self._move_thread.join(timeout=2.0)
            self._move_thread = None
        # Explicit zero command so the robot halts right away rather than
        # waiting out DimSim's ~500ms cmd_vel watchdog (confirmed in
        # lcm_probe/publish_cmd_vel.py's docstring).
        self.tele_cmd_vel.publish(Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0)))
        return "Stopped." if was_moving else "Was not moving."

    def _move_loop(self, twist: Twist, duration: float, stop_event: threading.Event) -> None:
        """Republish `twist` every 0.1s until `duration` elapses or
        `stop_event` is set -- DimSim's physics.ts has a ~500ms cmd_vel
        watchdog (confirmed in lcm_probe/publish_cmd_vel.py, which
        established this exact republish pattern), so a single publish
        would only move the robot briefly regardless of the requested
        duration.

        Safe to call self.stop_tool from this background thread (not the
        skill call's own thread) -- unlike start_tool, stop_tool's
        docstring carries no main-thread requirement, and since move()
        now refuses re-invocation while already running (see its
        docstring) instead of superseding, this is always the same
        invocation that opened the stream -- no race with a newer one."""
        deadline = time.monotonic() + duration
        while not stop_event.is_set() and time.monotonic() < deadline:
            self.tele_cmd_vel.publish(twist)
            stop_event.wait(0.1)
        self.tele_cmd_vel.publish(Twist(linear=Vector3(0.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0)))
        self.stop_tool("move")

    # -------------------------------------------------------------------
    # Background safety monitor. Added (2026-07-15) because every live test
    # still drove into the glass despite the (since-removed, see the module
    # docstring's 2026-07-20 note) per-decision LLM prompt's own glass
    # warning: the camera is only ever consulted at the moment a
    # frontier TARGET is chosen. Once a goal is handed to the A* planner,
    # path execution is purely geometric -- if the depth-derived costmap
    # doesn't perceive glass as an obstacle at all (suspected all along,
    # and not something to fake away in DimSim -- a real depth sensor
    # facing real glass has the same blind spot), a target can be
    # geometrically fine while the path to it still isn't, and nothing is
    # watching until impact. This thread runs independently of the
    # frontier-decision cycle, on its own timer (safety_check_interval_s),
    # asking a small camera-only question (_SAFETY_CHECK_PROMPT) via the
    # same LLM client/provider already configured -- no new dependency.
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
                # CONFIRMED BUG, FIXED (2026-07-15): originally stopped
                # the instant hazard_ahead was true, with no concept of
                # distance at all -- a live run showed the drone get
                # stopped almost immediately near its spawn point and
                # never make real progress, because "glass visible
                # somewhere in frame" was being treated as "imminent
                # collision" even from meters away. Now only a genuinely
                # close hazard (see hazard_stop_distance_m for the safety
                # margin reasoning) actually triggers a stop; a distant
                # one is logged but otherwise ignored, so exploration can
                # keep making real progress until it's actually close.
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
                # CONFIRMED BUG, FIXED (2026-07-15): a live run showed the
                # SAME hazard position re-detected 3 times in a row, each
                # a genuinely fresh (non-guarded) detection -- the
                # re-entrancy guard above was working (holds weren't
                # overlapping), but hazard_stop_hold_s (4.0s) and
                # safety_check_interval_s (3.0s, plus ~0.3-1s of LLM
                # latency) are close enough in value that the two
                # independent timers left only a tiny, inconsistent gap
                # between "hold expires" and "next check fires" --
                # confirmed from the exact live timestamps: 0.165s and
                # 0.519s of nominally-free time, nowhere near enough for
                # the robot to actually turn away and move somewhere
                # visibly different before being re-evaluated. Wait out
                # whatever's left of the hold here, on top of this loop's
                # own next interval-wait, so the robot always gets at
                # least one FULL safety_check_interval_s of genuinely
                # free, unmonitored time after every hold ends -- not
                # whatever scraps the two timers happened to leave.
                remaining_hold = self._hazard_hold_until - time.monotonic()
                if remaining_hold > 0:
                    stop_event.wait(remaining_hold)

    def _emergency_stop(self, estimated_distance_m: float | None = None) -> None:
        """Immediately halt the drone from the safety monitor's own thread.
        Deliberately bypasses the move()/stop_move() skills entirely -- this
        is an internal watchdog action, not an LLM tool call, so it isn't
        gated by the MCP capability system and can act without waiting for
        a tool-stream.

        CONFIRMED BUG, FIXED (2026-07-15): this used to publish the zero
        Twist on tele_cmd_vel, reusing the same path move()/human teleop
        use. That was wrong for THIS caller specifically: MovementManager's
        _on_teleop (dimos/navigation/movement_manager/movement_manager.py)
        unconditionally calls _cancel_goal() for ANY tele_cmd_vel message,
        which broadcasts stop_movement=True -- and
        WavefrontFrontierExplorer._on_stop_movement treats that as "a human
        took over," calling the REAL stop_exploration() (full teardown,
        capability release included), not a soft pause. Confirmed live: a
        real run showed "stop_movement received, stopping exploration"
        immediately after every hazard detection, and the exploration
        session actually ending each time (capabilities released) instead
        of continuing to a fresh frontier -- directly contradicting the
        point of this feature ("continue exploring, but avoid it"). That
        cascade IS correct/desired for move()/real human teleop (both are
        meant to take over indefinitely, per CONSOLE_SYSTEM_PROMPT) -- it
        was only wrong for this internal, automated, should-keep-exploring
        caller. Fixed by publishing on the new silent_cmd_vel stream
        instead (see its declaration above and MovementManager's matching
        addition) -- claims the same teleop-priority window (so this
        command actually takes effect, nav_cmd_vel gets suppressed)
        WITHOUT the stop_movement broadcast, so exploration keeps running
        and just picks a fresh (now hazard-aware, via the wall stamped
        below) frontier once goal_reached_event wakes the loop. Held via
        _hold_silent_cmd_vel for hazard_stop_hold_s rather than a single
        publish, since silent_cmd_vel's priority-window is time-limited
        (tele_cooldown_sec, 1.0s default in MovementManager) and a single
        publish could lapse before the exploration loop finishes picking
        and publishing its next (redirecting) goal.

        CONFIRMED BUG, FIXED (2026-07-15), a second time: originally just
        held position (zero Twist) here and relied entirely on the
        exploration loop picking a new frontier -- deliberately, reasoning
        that a fresh, camera-informed frontier decision was safer than
        blindly guessing an escape direction. User caught the real gap
        live: a hazard less than hazard_stop_distance_m from the drone's
        SPAWN position left it standing still indefinitely, because with
        almost nothing mapped yet, a "fresh" frontier pick doesn't
        reliably lead anywhere actually different -- there's no map data
        yet to distinguish directions. _hold_silent_cmd_vel now commands a
        real backup maneuver (back straight away, negative x) for
        hazard_backup_duration_s within the hold, not just a stop -- see
        its docstring for why backing away specifically, not turning.

        Also marks the estimated hazard location as a permanent wall (see
        _stamp_hazards) so future planning -- both frontier selection AND
        the A* planner -- routes around it instead of just not targeting
        it directly. Position is a rough forward-projection from the
        robot's current pose (hazard_projection_distance_m) since the
        safety check itself has no range/bearing, only "hazard ahead".

        CONFIRMED BUG, FIXED (2026-07-15): a live run showed the drone
        getting permanently frozen -- the SAME hazard position marked as
        a "new" wall 17+ times in a row, bit-for-bit identical every time
        (proof the robot never actually moved in between). Root cause:
        hazard_stop_hold_s (4.0s default) is LONGER than
        safety_check_interval_s (3.0s), and the safety monitor has no idea
        a hold is already in progress -- it just keeps checking on its own
        fixed timer regardless. Since the robot hasn't moved yet (still
        mid-hold), the camera shows the exact same view, so the LLM
        correctly says "hazard ahead" again, which calls this method
        again, which starts a FRESH hold before the previous one ever
        expires. Each ~3.4s cycle re-arms the freeze, so the robot never
        gets a real window to actually redirect and move -- an unintended
        self-sustaining deadlock, not a sign the hazard detection itself
        was wrong. Fixed with a re-entrancy guard: a redundant detection
        during an already-active hold is a no-op (still logged, but
        doesn't restart the hold or re-mark/re-blacklist anything) --
        this guarantees a hold, once started, always gets its FULL
        hazard_stop_hold_s uninterrupted, giving the exploration loop (and
        the robot, once released) a genuine chance to move somewhere
        different before the next real check.

        estimated_distance_m (2026-07-15): the LLM's own rough distance
        estimate from _SAFETY_CHECK_PROMPT, when it gave a usable one --
        preferred over hazard_projection_distance_m's fixed guess for
        placing the wall, since an actual (if imprecise) per-frame
        estimate is more accurate than always assuming the same fixed
        distance regardless of how far the hazard actually looked.
        """
        now = time.monotonic()
        if now < self._hazard_hold_until:
            logger.info(
                "AgenticFrontierSelector: hazard re-detected during an active silent_cmd_vel hold "
                "(%.1fs remaining) -- ignoring, not restarting the hold (see _emergency_stop's "
                "docstring for why repeated re-arming was freezing the robot in place)",
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
        capped so the total never exceeds the hold budget (see
        hazard_stop_hold_s's docstring):
        1. Back straight away (negative x) for hazard_backup_duration_s.
        2. Turn in place (no translation) by hazard_turn_angle_rad at
           hazard_turn_rate_rps, so the camera ends up facing away from
           the hazard instead of straight at it.
        3. Hold position (zero Twist) for whatever remains of the hold.
        MovementManager's teleop-priority window that silent_cmd_vel
        claims is time-limited (tele_cooldown_sec, 1.0s default) and would
        otherwise lapse -- letting nav_cmd_vel resume driving toward the
        just-detected hazard -- before the exploration loop (woken by
        goal_reached_event in _emergency_stop) picks and publishes a fresh
        (now hazard-aware, via the wall stamped above) goal.
        Fire-and-forget: bounded duration, nothing external needs to stop
        it early, so unlike move()/_move_loop this doesn't need its own
        stop_event/thread-handle bookkeeping.

        Backs away before turning, and turns a full ~180 degrees rather
        than guessing a partial left/right: a single forward-facing camera
        frame gives no information about which side is actually clear, but
        backing away is guaranteed to increase distance from whatever's
        directly ahead regardless of room layout, and facing exactly
        opposite the original heading is guaranteed to point the camera at
        wherever the drone just came from, not at the hazard -- see
        _emergency_stop's docstring for the two live failures this fixes
        (a hazard near spawn left the drone standing still indefinitely
        relying on a "fresh" frontier pick with nothing to actually
        distinguish directions with; backing away alone then left it
        sitting motionless facing the same hazard for 30+ seconds since
        nothing made it look anywhere else).

        SAFETY FALLBACK (2026-07-15): MovementManager.silent_cmd_vel is a
        ../dimos addition (see its declaration above), currently live via
        this project's pyproject.toml dimos dependency
        (knikou01/dimos.git@drone-room-mapper/silent-stop-v5). Getting a
        working dimos dependency wired up at all was its own saga this
        session -- see the dimos-dependency-wiring memory/PR history if
        this ever needs redoing (a prior attempt landed on a stale fork
        branch that silently regressed several packages and broke
        WebInput/the console entirely; confirm blueprints still import AND
        `grep silent_cmd_vel` the installed MovementManager before
        trusting any future dimos dependency change). If
        self.silent_cmd_vel.transport is ever None again -- dependency
        reverted, blueprint missing MovementManager, etc. -- publishing to
        it would silently do nothing and the robot would NOT react at
        all, which is worse than the pre-silent_cmd_vel behavior. Falls
        back to a plain tele_cmd_vel stop (not a backup -- no reactive
        maneuver at all without this stream, just a halt, at the cost of
        the stop_movement cascade this whole feature exists to avoid --
        see _emergency_stop's docstring) whenever silent_cmd_vel isn't
        actually connected, so a hazard always at least halts the robot
        either way. Keep this fallback -- it's cheap insurance against
        exactly the kind of dependency drift that happened this
        session."""
        if self.silent_cmd_vel.transport is None:
            logger.warning(
                "AgenticFrontierSelector: silent_cmd_vel has no transport (MovementManager."
                "silent_cmd_vel not live in the current dimos dependency) -- falling back to a "
                "plain tele_cmd_vel stop (no backup maneuver), which WILL abort the whole "
                "exploration session (see _emergency_stop's docstring). Check this project's dimos "
                "dependency in pyproject.toml -- it should currently be "
                "knikou01/dimos.git@drone-room-mapper/silent-stop-v5 (or a successor branch that "
                "still carries MovementManager.silent_cmd_vel)."
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
            # Groq's API is OpenAI-compatible (confirmed via Groq's own docs),
            # so this reuses the openai package already imported for the
            # "openai" branch above rather than adding a new dependency --
            # just pointed at Groq's base_url with GROQ_API_KEY.
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
                # encoding itself, and max_width keeps the request small --
                # confirmed against the real dimos.msgs.sensor_msgs.Image.
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
