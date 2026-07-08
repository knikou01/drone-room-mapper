"""Real drone passive mapping blueprint.

Composes on top of drone_basic (DroneConnectionModule + DroneCameraModule +
Rerun visualizer) and adds DroneDepthModule and VoxelGridMapper to build a
live 3D map from Depth Anything V2 monocular depth estimates while the
drone is hand-flown.

No autonomous flight logic is included. Fly the drone by hand and watch
the 3D map grow in the Rerun viewer.

Wiring (automatic via autoconnect):
    DroneCameraModule.color_image  -> DroneDepthModule.color_image
    DroneConnectionModule.odom     -> DroneDepthModule.odom
    DroneDepthModule.lidar         -> VoxelGridMapper.lidar
    VoxelGridMapper.global_map     -> Rerun visualizer

BEFORE RUNNING:
  1. Confirm the drone is connected: run drone-basic and check that
     /odom and /video are flowing.
  2. Set cruise_altitude_m to your actual cruise altitude.
  3. On first run, the model downloads ~100MB to ~/.cache/huggingface/.
     Subsequent runs use the local cache and start faster.
  4. CPU inference takes 1-4s per frame. The map updates slowly but
     continuously as you fly. A GPU would speed this up significantly.
"""

from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.voxels import VoxelGridMapper
from dimos.robot.drone.blueprints.basic.drone_basic import drone_basic

from drone.mapping_module import DroneDepthModule

real_mapping = autoconnect(
    drone_basic,
    DroneDepthModule.blueprint(
        cruise_altitude_m=1.5,  # set to your actual cruise altitude
        max_depth_m=8.0,
        min_depth_m=0.3,
        depth_sample_stride=8,
        inference_interval_s=3.0,
    ),
    VoxelGridMapper.blueprint(emit_every=5),
)

__all__ = ["real_mapping"]
