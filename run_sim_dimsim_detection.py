"""
Mirrors run_sim_detection.py, but for the DimSim-backed blueprint.

PREREQUISITE: the DimSim relay must already be running and a browser tab
already connected (see blueprints/sim_detection_dimsim_blueprint.py's
module docstring for the exact commands and confirmation steps) BEFORE
this script is run. This script does not start, check, or wait for the
relay — it assumes the sensor pipeline is already live, exactly as it was
when verified manually via lcm_probe/probe.py and
lcm_probe/publish_cmd_vel.py.
"""
import time

from dimos.core.coordination.module_coordinator import ModuleCoordinator

from blueprints.sim_detection_dimsim_blueprint import drone_sim_dimsim_detection

if __name__ == '__main__':
    coordinator = ModuleCoordinator.build(drone_sim_dimsim_detection, {})

    if not coordinator.health_check():
        print("Health check failed")
        coordinator.stop()
        exit(1)

    print(f"Starting simulation with {coordinator.n_modules} modules...")
    print("Reminder: this assumes the DimSim relay + browser tab are already")
    print("connected. If the agent never moves or sensors never arrive, check")
    print("that first — this script does not verify it for you.")
    time.sleep(3.0)

    try:
        coordinator.loop()
    except KeyboardInterrupt:
        print("Stopping simulation...")
        coordinator.stop()
