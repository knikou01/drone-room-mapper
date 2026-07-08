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

## Known limitations

- **Visual odometry (VO) is not yet accurate enough to trust for real
  localization.** `DimSimVisualOdometryModule` publishes a vision-based
  pose estimate on `odom_vo` alongside ground truth purely for comparison
  — nothing consumes it yet. Confirmed real motion (several metres of
  translation, 120+ degrees of rotation) produces almost no change in the
  published estimate, even with system load ruled out as a cause. Leading
  explanation: DimSim's color/depth frames arrive roughly a second apart
  on typical hardware, and dense frame-to-frame odometry (used here)
  assumes much smaller gaps between frames — full details and evidence in
  `drone/dimsim_visual_odometry_module.py`'s module docstring. Mapping
  currently uses DimSim's ground-truth pose instead; replacing that with
  real VO is deferred future work.
- **A glass surface in the `apt` scene permanently wedges the robot on
  contact.** This is a general limitation of robots running DimOS against
  DimSim's thin/transparent-surface collision handling, not something
  fixable from this repository — see
  `blueprints/sim_dimsim_vo_blueprint.py`'s module docstring.
