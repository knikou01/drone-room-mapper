#!/usr/bin/env python3
"""Entry point for the lightweight VO-only validation blueprint.

Use this instead of run_sim_dimsim_vo.py when you want to isolate whether
DimSimVisualOdometryModule's real-motion tracking accuracy is a genuine
bug or an artifact of system load -- this blueprint drops ObjectDBModule,
the mapping stack, and the autonomous exploration stack (see
blueprints/sim_dimsim_vo_only_blueprint.py's module docstring for exactly
what's dropped and why).

BEFORE RUNNING:
  1. Start the DimSim relay:
       cd ~/Documents/kios/DimSim
       deno run --allow-all --unstable-net dimos-cli/cli.ts dev --scene apt
  2. Open http://localhost:8090/?dimos=1&scene=apt and wait for
     "[dimos] Bridge connected. Sensor publishing active." in the browser console.
  3. THEN run this script.

No exploration trigger needed or available here -- once running, open
http://localhost:7779/ (DimOS's own teleop dashboard, a separate tab from
DimSim's own scene view) and drive manually. Watch this terminal for
"VO vs ground-truth" and "DIAG" log lines.
"""

from dimos.core.coordination.module_coordinator import ModuleCoordinator

from blueprints.sim_dimsim_vo_only_blueprint import sim_dimsim_vo_only

if __name__ == "__main__":
    coordinator = ModuleCoordinator.build(sim_dimsim_vo_only, {})

    if not coordinator.health_check():
        print("Health check failed -- is the DimSim relay running and browser connected?")
        coordinator.stop()
        exit(1)

    print(f"DimSim VO-only validation started with {coordinator.n_modules} modules.")
    print("Open http://localhost:7779/ and drive manually with the teleop dashboard.")
    print("Watch logs for 'VO vs ground-truth' / 'DIAG' lines.")
    print("Ctrl+C to stop.")

    try:
        coordinator.loop()
    except KeyboardInterrupt:
        print("Stopping...")
        coordinator.stop()
