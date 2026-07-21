"""DimSim VO-only validation blueprint.

Stripped-down variant of sim_dimsim_vo_blueprint.py for isolating
DimSimVisualOdometryModule's tracking accuracy from system load --
ObjectDBModule's per-frame detection/tracking work is heavy enough that it
can starve VO of CPU and inflate pairing_gap on its own, independent of
any real tracking bug.

Deliberately DROPPED relative to sim_dimsim_vo_blueprint.py, none of which
VO itself depends on (it reads DimSim's raw /color_image, /depth_image,
/odom directly via its own standalone LCM() instance -- see its module
docstring):
  - ObjectDBModule        (heaviest module here -- real per-frame
                            detection/tracking work)
  - CostMapper, WavefrontFrontierExplorer, ReplanningAStarPlanner
                          (autonomous exploration -- not used, this is a
                            pure manual-teleop test)
  - DimSimDepthLidarModule, VoxelGridMapper
                          (Phase 2 mapping -- unrelated to Phase 1 VO
                            tracking accuracy)

KEPT:
  - DimSimCameraModule    cheap relay (no heavy per-frame compute), needed
                          only for the world/tf/base_link Transform3D so
                          Rerun can show a moving drone marker. Its
                          dimsim_color_image/pointcloud outputs are unused
                          here (nothing subscribes to them) but harmless.
  - DimSimVisualOdometryModule   what's being validated.
  - MovementManager       required to route the teleop dashboard's
                          tele_cmd_vel into /cmd_vel, which is what
                          actually drives DimSim's physics -- confirmed
                          from its source (movement_manager.py) that it
                          works standalone with nav_cmd_vel unconnected.

Usage: same prerequisites as sim_dimsim_vo_blueprint.py (DimSim relay +
browser tab connected first), but do NOT run lcm_probe/start_exploration --
there's no exploration stack here. Instead open http://localhost:7779/ and
drive manually with the dashboard's teleop controls. Watch for the same
"VO vs ground-truth" / DIAG log lines as before.
"""
from typing import Any

import rerun as rr

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.visualization.vis_module import vis_module

from sim_camera.dimsim_camera_module import DimSimCameraModule

from drone.dimsim_visual_odometry_module import DimSimVisualOdometryModule


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

sim_dimsim_vo_only = autoconnect(
    _vis,
    DimSimCameraModule.blueprint(),
    # tracking_method="dense" was re-tested here after fixing DimSim's
    # capture-pipeline stall, since that fix removed dense odometry's
    # original blocker (a large color/depth pairing_gap breaking its
    # small-motion assumption). Still worse than "sparse": dense reports
    # confident false success (success_rate 95-97%) while the tracked pose
    # stays frozen/near-identity during real motion -- a materially worse
    # failure mode than sparse's lower but honest success rate. Left as
    # module default ("sparse") -- do not re-enable dense without new
    # evidence.
    DimSimVisualOdometryModule.blueprint(),
    MovementManager.blueprint(),
)

__all__ = ["sim_dimsim_vo_only"]
