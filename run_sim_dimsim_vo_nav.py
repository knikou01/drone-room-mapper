#!/usr/bin/env python3
"""Entry point for the DimSim Phase 3 blueprint -- NAVIGATION driven by
real VO (odom_vo). Mapping deliberately stays on ground truth -- see
blueprints/sim_dimsim_vo_nav_blueprint.py's module docstring SCOPE CHANGE
section (a live test with mapping also on VO produced a visibly corrupted
map -- "a lot of walls where they don't belong" -- since mapping has no
self-correction against VO drift the way replanning does).

Mirrors run_sim_dimsim_vo.py's structure.

BEFORE RUNNING:
  1. Start the DimSim relay:
       cd ~/Documents/kios/DimSim
       deno run --allow-all --unstable-net dimos-cli/cli.ts dev --scene apt_no_glass
  2. Open http://localhost:8090/?dimos=1&scene=apt_no_glass and wait for
     "[dimos] Bridge connected. Sensor publishing active." in the browser console.
  3. THEN run this script.

Trigger autonomous exploration with: python -m lcm_probe.start_exploration

VERIFY: check the startup "Transport" log lines. ReplanningAStarPlanner's
and WavefrontFrontierExplorer's odom streams must show a topic containing
"odom_vo". DimSimDepthLidarModule's odom stream must show the raw
ground-truth "/odom" topic -- if it shows "odom_vo" instead, someone
re-added the mapping remap and the corrupted-map problem will come back.

Watch the logs for "VO vs ground-truth" lines throughout -- navigation now
DEPENDS on VO's accuracy (mapping does not), so any sustained large
divergence here is a real navigation problem to investigate.
"""

from dimos.core.coordination.module_coordinator import ModuleCoordinator

from blueprints.sim_dimsim_vo_nav_blueprint import sim_dimsim_vo_nav

if __name__ == "__main__":
    coordinator = ModuleCoordinator.build(sim_dimsim_vo_nav, {})

    if not coordinator.health_check():
        print("Health check failed -- is the DimSim relay running and browser connected?")
        coordinator.stop()
        exit(1)

    print(f"DimSim Phase 3 (VO-driven navigation, ground-truth mapping) started with {coordinator.n_modules} modules.")
    print("Trigger exploration separately with: python -m lcm_probe.start_exploration")
    print("Watch logs for 'VO vs ground-truth' lines -- navigation now depends on this accuracy.")
    print("Ctrl+C to stop.")

    try:
        coordinator.loop()
    except KeyboardInterrupt:
        print("Stopping...")
        coordinator.stop()
