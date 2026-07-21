# drone-room-mapper

A [DimOS](https://github.com/dimensionalOS/dimos)-based drone room-mapping
project. Current top priority: mapping a room with **vision instead of
lidar** — a depth camera feed is projected into a 3D point cloud and fed
into the same mapping/exploration stack that would normally consume a
lidar, validated inside [DimSim](../DimSim) (a browser-based robot
simulator) against DimSim's own ground-truth pose before any of it is
trusted on the real drone.

## Setup

- Python 3.12 (see `.python-version`)
- `uv sync` (or `pip install -e .`) from this directory
- A sibling checkout of [DimSim](../DimSim) — the simulator used below
- A sibling checkout of [dimos](../dimos) — the DimOS framework this
  project builds on

## Visual SLAM in DimSim

`blueprints/sim_dimsim_vo_blueprint.py` builds a world-frame point cloud
from DimSim's depth camera image and feeds it to `VoxelGridMapper` *instead
of* DimSim's own raycast lidar topic — the actual "map with vision, not
lidar" swap. Mapping currently uses DimSim's ground-truth pose, not a
vision-based pose estimate — see **Known limitations** below for why.

### Prerequisites (both modes below)

1. Start the DimSim relay:
   ```bash
   cd ../DimSim
   deno run --allow-all --unstable-net dimos-cli/cli.ts dev --scene apt
   ```
2. Open `http://localhost:8090/?dimos=1&scene=apt` in a browser and wait
   for `[dimos] Bridge connected. Sensor publishing active.` in the
   browser's console. Do this *before* starting either script below.

### Autonomous exploration

Drives the robot itself via a classical frontier-exploration + A* planner
stack (`WavefrontFrontierExplorer` + `ReplanningAStarPlanner` +
`MovementManager`), building the map as it goes.

```bash
# Terminal 1: after the prerequisites above
python run_sim_dimsim_vo.py

# Terminal 2: trigger exploration (the blueprint doesn't start moving on its own)
python -m lcm_probe.start_exploration
```

Watch Terminal 1 for `DimSimVisualOdometryModule: VO vs ground-truth --
...` lines (accuracy telemetry, see Known limitations) and for the Rerun
viewer window that opens automatically — the left panel shows the live
camera feed, the right panel shows the accumulating 3D point-cloud map.

**Known issue:** the exploration planner routes the robot straight into a
glass obstacle in the `apt` scene almost immediately, and the robot gets
permanently wedged with no recovery (confirmed not fixable by manual
control either). See Known limitations.

### Manual control

For driving the robot by hand — useful for testing visual odometry
accuracy or exploring in an area the autonomous planner can't safely
reach (see the glass issue above).

```bash
python run_sim_dimsim_vo_only.py
```

Then open `http://localhost:7779/` (DimOS's own "Command Center" teleop
dashboard — a separate tab from the DimSim scene view opened above) and
drive with its on-screen controls. This entry point uses a stripped-down
blueprint (camera + visual odometry + movement + visualization only, no
object detection or autonomous planning) so it's noticeably lighter on
CPU than the autonomous mode above.

### Agentic (LLM-driven) exploration

Same vision-based mapping as autonomous exploration above, plus a
natural-language console (`AgenticFrontierSelector`) for controlling it.
Frontier *selection* is the same plain geometric heuristic as autonomous
exploration — the LLM no longer picks candidates. Its role is narrower and
request-driven: start/stop exploration, drive the drone directly when
asked, and a background safety monitor that watches the live camera
continuously while a goal is being driven and reacts to a real hazard
(see the note on the known glass issue below) — independent of frontier
decisions, which don't consult the LLM at all.

```bash
python run_sim_dimsim_agentic.py
```

You'll be asked to pick an LLM provider (Gemini or Groq's Qwen3.6 27B —
add `GEMINI_API_KEY` and/or `GROQ_API_KEY` to `.env` accordingly). This
same choice is used for both the natural-language console and the
background safety monitor (frontier selection itself doesn't use an LLM
at all — see above), so picking one provider is enough either way. You'll
then be asked whether to enable the background hazard safety monitor
(default no) — turn it on if you're testing hazard avoidance on a scene
with glass (e.g. `apt`). Left off by default since most testing is
currently on `apt_no_glass`, where it's pure overhead and can
false-positive on ordinary architecture. Then
open `http://localhost:5555/` — a natural-language console, separate from
both the DimSim scene tab and the teleop dashboard — and type something
like "start exploring". Watch the terminal and Rerun viewer as before.

The console also understands direct commands (e.g. "move away from the
glass", "turn left") — it calls `move`/`stop_move` to drive the drone
itself, taking over from autonomous exploration immediately (same
priority the teleop dashboard has) and handing back to a fresh frontier
decision once you resume exploring.

**Note on the glass issue:** frontier selection itself is purely geometric
and does not look at the camera at all. A background safety monitor
checks the live camera every few seconds *while a goal is being driven*
— see `AgenticFrontierSelector`'s module docstring for why this exists
(a chosen target can be geometrically fine while the path to it still
isn't, if the depth costmap doesn't perceive glass as an obstacle at
all). It asks for a rough distance estimate alongside the hazard flag and
only actually reacts once a hazard is within `hazard_stop_distance_m`
(2.0m default — sized with real margin around the robot's driving speed
and the check interval, not just "glass merely visible in frame," which
used to stop the drone almost immediately near its spawn point on every
run). Once close, it backs the drone straight away for a couple of
seconds and then turns it to face the opposite direction (rather than
trying to guess a clear left/right to turn toward) — so the camera isn't
left staring at the same hazard while the navigation stack sorts out
where to go next — and marks the hazard as a permanent wall in the shared
costmap — at the LLM's own distance estimate when available, otherwise a
rough forward-projected guess — so the drone doesn't just avoid
*targeting* it again, the A* planner won't route a path through it
either. See `AgenticFrontierSelector.hazard_costmap`/`_stamp_hazards`/
`_emergency_stop`.

### Phase 3: VO-driven navigation (experimental, not recommended)

`run_sim_dimsim_vo_nav.py` / `run_sim_dimsim_agentic_vo.py` are the same
stacks as above but with navigation's pose source swapped from ground
truth to real visual odometry (`odom_vo`) — mapping stays on ground truth
deliberately. See **Known limitations** below for the full story: VO's
current accuracy against DimSim's actual scene content isn't reliable
enough for this to work consistently yet, and the existing ground-truth
entry points remain the recommended way to actually use this project.
These exist so the plumbing is ready and documented for whenever VO
accuracy improves, not as a working alternative today.

## Known limitations

- **Visual odometry (VO) is not yet accurate enough to trust for real
  navigation, and mapping stays on ground truth permanently (not just for
  now).** Phase 3 (swapping DimOS's pose source from ground truth to real
  VO) made real, confirmed progress but did not reach a reliable end
  state — see `drone/dimsim_visual_odometry_module.py`'s module docstring
  for the full evidence trail. What's actually true as of this pass:
  - **Root cause of the original "VO barely moves at all" symptom:** its
    image-capture path did a synchronous, GPU-pipeline-stalling pixel
    readback on every frame (a real "GPU stall due to ReadPixels" driver
    warning), causing `pairing_gap` (time between usable frames) to blow
    out from a 200ms design target to as much as 60+ seconds under load.
    Fixed in `../DimSim` (`engine.js`/`dimosBridge.ts`) by switching to
    Three.js's async, WebGL2-PBO-based `readRenderTargetPixelsAsync()` —
    this holds `pairing_gap` near the 200ms target under light load,
    though it can still degrade to several seconds under heavier load (not
    perfectly solved, meaningfully improved).
  - **`DimSimVisualOdometryModule` was rewritten to track via ORB feature
    matching + depth-based 3D-3D RANSAC** (`tracking_method="sparse"`,
    now the default) instead of dense frame-to-frame photometric
    alignment (`tracking_method="dense"`, kept for comparison) — dense
    assumed small motion between frames and silently reported false
    "success" at a near-identity transform when that assumption broke;
    sparse tolerates larger gaps and, critically, fails safely (no
    transform published) instead of confidently publishing a wrong one.
    **Re-tested dense after the DimSim frame-rate fix and confirmed it's
    still worse, not better** — under real (non-best-case) load it
    reproduced its exact original failure mode (`success_rate` 95-97%
    while the tracked pose was actually frozen).
  - **Even sparse tracking has a low real-world success rate against
    DimSim's actual scene content** (single-digit percent in live
    testing), not primarily because of leftover accuracy-safety filters
    (inlier count, degenerate-geometry rejection, physical-plausibility
    rejection — all added after live failures, all still active) but
    because many real views don't offer enough distinctive, well-
    distributed visual texture for reliable feature matching in the first
    place — confirmed via per-rejection-reason logging
    (`too_few_orb_features`/`too_few_orb_matches`/`degenerate_geometry`
    dominating, not just the safety filters). This is a content/rendering
    characteristic of the simulated scenes, not a tuning problem alone.
  - **Decision: decouple navigation from mapping rather than swap both
    onto VO together.** A live test with both remapped produced
    a visibly corrupted map ("walls where they don't belong") — mapping
    (`DimSimDepthLidarModule`) has zero self-correction against VO drift
    (every point gets projected into world coordinates and permanently
    accumulated at whatever pose it was captured at), unlike navigation
    decisions, which get real-time self-correction from
    `ReplanningAStarPlanner`'s frequent replanning. So: two new,
    *additive* blueprints (`blueprints/sim_dimsim_vo_nav_blueprint.py`,
    `blueprints/sim_dimsim_agentic_vo_blueprint.py`, with entry points
    `run_sim_dimsim_vo_nav.py`/`run_sim_dimsim_agentic_vo.py`) remap only
    `ReplanningAStarPlanner`/`WavefrontFrontierExplorer`/
    `AgenticFrontierSelector`'s pose input to `odom_vo`, leaving
    `DimSimDepthLidarModule` on ground truth. **These are experimental,
    not a recommended default** — with VO's current success rate, the
    navigation stack's own pose estimate updates too rarely to reliably
    track real progress, and live testing showed the robot getting stuck
    (its *belief* of where it is stalls near spawn while it may have
    actually moved, since nothing refreshes it often enough). The
    existing ground-truth blueprints
    (`sim_dimsim_vo_blueprint.py`/`sim_dimsim_agentic_blueprint.py`)
    remain the working, recommended path for actual use.
  - Real, working fixes made along the way that are NOT limitations and
    don't need revisiting: a real deadlock where `odom_vo` never
    published anything until VO's first successful transform (which could
    itself never happen if the robot never moved) is fixed —
    `DimSimVisualOdometryModule` now bootstraps `odom_vo` with the
    origin-anchored spawn pose immediately, before any tracking succeeds.
  - **Not fixed, deliberately, and not planned without new evidence
    changing this conclusion:** further loosening sparse's safety filters
    to trade accuracy risk for update frequency, or chasing DimSim's
    residual frame-rate variability further (a real, separate lever that
    was flagged but not fully pursued — see `_dimosCaptureRgb`'s
    JPEG-encode cost, still untouched). Revisit VO-driven navigation as a
    default only if either of those change the picture.
- **A glass surface in the `apt` scene permanently wedges the robot on
  contact.** This is a general limitation of robots running DimOS against
  DimSim's thin/transparent-surface collision handling, not something
  fixable from this repository — see
  `blueprints/sim_dimsim_vo_blueprint.py`'s module docstring.
- **Depth-camera mapping has narrower, heading-dependent coverage than
  lidar — a wall the camera hasn't faced closely yet can be pathed
  through.** Confirmed not a lidar leak: `VoxelGridMapper.lidar` is
  remapped to `DimSimDepthLidarModule`'s depth-camera-derived output only
  (`drone/dimsim_depth_lidar_module.py`'s module docstring explains why
  the remap exists specifically to avoid DimSim's raw `/lidar` topic), and
  `CostMapper.merged_map` has no publisher in either blueprint. The real
  mechanism: DimOS's A* planner (`min_cost_astar.py`) treats *unmapped*
  cells as crossable at a cost penalty, not as blocked — only cells
  already confirmed occupied are hard walls, since frontier exploration
  has to be willing to path toward space it hasn't seen yet to explore at
  all. If a wall segment was never actually observed closely enough by the
  narrow, forward-facing depth camera to be marked occupied, the space
  behind it stays "unknown, crossable" rather than "blocked," and the
  planner can drive straight through it. A real 360° lidar would map a
  nearby wall almost immediately regardless of heading; this is a real,
  expected tradeoff of vision-only mapping, not something visual SLAM
  (accurate pose) would fix on its own — it's a coverage gap, not a pose
  error. Not fixed here; noted as a known Phase 2 limitation.
