"""
blueprints/sim_detection_dimsim_blueprint.py

DimSim-backed equivalent of sim_detection_blueprint.py. No
SimDroneConnectionModule and no PyBullet/Room generation — DimSim's external
relay process is the sensor source, and DimOS's existing navigation stack
drives it purely by publishing /cmd_vel over LCM, with no in-process consumer
needed (confirmed: ModuleCoordinator._connect_streams groups transports by
(name, type) only, not by direction — see module_coordinator.py).

PREREQUISITE — must be running BEFORE this blueprint starts:
    cd DimSim
    npm run build   # only needed once, or after pulling DimSim updates
    deno run --allow-all --unstable-net dimos-cli/cli.ts dev --scene apt

Then open http://localhost:8090/?dimos=1&scene=apt in a browser and confirm
"[dimos] Bridge connected. Sensor publishing active." appears in the console
BEFORE running this blueprint — there is currently no automatic check or
retry for this; if the relay/browser aren't ready, color_image/pointcloud/
odom/lidar will simply never arrive, with no error raised in this process.

Stream wiring (mostly (name, type) auto-match via autoconnect; one
.remappings() entry needed for color_image -- see below):
  /odom#geometry_msgs.PoseStamped   <- DimSim relay (external)
                                     -> WavefrontFrontierExplorer.odom
                                     -> ReplanningAStarPlanner.odom
                                     -> DimSimCameraModule (internal, for TF)
  /lidar#sensor_msgs.PointCloud2    <- DimSim relay (external)
                                     -> VoxelGridMapper.lidar
  /cmd_vel#geometry_msgs.Twist      <- MovementManager.cmd_vel (Out)
                                     -> DimSim relay (external, ServerPhysics)
  dimsim_color_image / pointcloud   <- DimSimCameraModule (Out, internal)
                                     -> ObjectDBModule (In, via remap)

DimSimCameraModule's image stream is named "dimsim_color_image", not
"color_image" -- avoids colliding with DimSim's own raw /color_image topic
(see dimsim_camera_module.py's module docstring for why that collision is
a real problem, not just a naming nitpick). ObjectDBModule's In[Image]
"color_image" is wired to it via .remappings() below instead of relying on
name auto-match.

Image/PointCloud2/PoseStamped encoding compatibility between DimOS's own
LCM transport and DimSim's vendored LCM implementation was confirmed via
lcm_probe/probe.py. Twist (DimOS -> DimSim direction) was only inferred
from physics.ts's CH_CMD_VEL constant and handleCmdVel's field access
matching DimOS's Twist.linear.x/.angular.z shape, not independently
probed -- if the agent doesn't move when this blueprint runs, check that
first.
"""
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

from sim_camera.dimsim_camera_module import DimSimCameraModule, make_camera_info_default


def _make_rerun_blueprint() -> Any:
    import rerun.blueprint as rrb
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/color_image", name="DimSim Camera"),
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
    # Mirrors sim_blueprint.py's _static_drone_body exactly. Needs
    # DimSimCameraModule to actually publish a base_link TF frame for this
    # to attach to — added alongside this function, since DimSimCameraModule
    # previously only published camera_optical and had no visible agent
    # marker in rerun at all.
    return [
        rr.Boxes3D(
            half_sizes=[0.25, 0.25, 0.1],
            colors=[(255, 100, 0)],
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


_rerun_config = {
    "blueprint": _make_rerun_blueprint,
    "static": {"world/tf/base_link": _static_drone_body},
}

_vis = vis_module(
    viewer_backend=global_config.viewer,
    rerun_config=_rerun_config,
)

drone_sim_dimsim_detection = autoconnect(
    _vis,
    DimSimCameraModule.blueprint(),
    ObjectDBModule.blueprint(camera_info=make_camera_info_default()),  # uses DimSim's known fixed capture resolution (640x288, per engine.js); DimSimCameraModule's own internal CameraInfo is rebuilt from real frames independently and only needs to match if DimSim's capture resolution ever changes
    VoxelGridMapper.blueprint(emit_every=5),
    CostMapper.blueprint(),
    WavefrontFrontierExplorer.blueprint(),
    PatrollingModule.blueprint(),
    ReplanningAStarPlanner.blueprint(),
    MovementManager.blueprint(),
).global_config(n_workers=7).remappings(
    [
        # See module docstring above and dimsim_camera_module.py: avoids a
        # same-topic collision (and resulting feedback loop) with DimSim's
        # own raw /color_image publisher.
        (ObjectDBModule, "color_image", "dimsim_color_image"),
    ]
)

__all__ = ["drone_sim_dimsim_detection"]
