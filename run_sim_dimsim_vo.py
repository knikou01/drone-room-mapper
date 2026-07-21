#!/usr/bin/env python3
"""Entry point for the DimSim + Visual Odometry blueprint.

Mirrors run_sim_dimsim_detection.py's structure.

BEFORE RUNNING:
  1. Start the DimSim relay:
       cd ~/Documents/kios/DimSim
       deno run --allow-all --unstable-net dimos-cli/cli.ts dev --scene apt
  2. Open http://localhost:8090/?dimos=1&scene=apt and wait for
     "[dimos] Bridge connected. Sensor publishing active." in the browser console.
  3. THEN run this script.

This does not trigger autonomous exploration automatically -- use
lcm_probe/start_exploration.py after the blueprint is up, same as the
existing DimSim detection workflow.

Watch the logs for "VO vs ground-truth" lines -- these report position and
yaw error between DimSimVisualOdometryModule's estimate and DimSim's real
ground-truth pose.
"""

from dimos.core.coordination.module_coordinator import ModuleCoordinator

from blueprints.sim_dimsim_vo_blueprint import sim_dimsim_vo

if __name__ == "__main__":
    coordinator = ModuleCoordinator.build(sim_dimsim_vo, {})

    if not coordinator.health_check():
        print("Health check failed -- is the DimSim relay running and browser connected?")
        coordinator.stop()
        exit(1)

    print(f"DimSim + VO validation started with {coordinator.n_modules} modules.")
    print("Trigger exploration separately with: python -m lcm_probe.start_exploration")
    print("Watch logs for 'VO vs ground-truth' lines to see VO accuracy.")
    print("Ctrl+C to stop.")

    try:
        coordinator.loop()
    except KeyboardInterrupt:
        print("Stopping...")
        coordinator.stop()
