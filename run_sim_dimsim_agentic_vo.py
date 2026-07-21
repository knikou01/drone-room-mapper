#!/usr/bin/env python3
"""Entry point for DimSim visual-SLAM + agentic console, with NAVIGATION
driven by real VO (odom_vo). Mapping deliberately stays on ground truth --
see blueprints/sim_dimsim_agentic_vo_blueprint.py's module docstring
(remapping mapping to VO too produces a visibly corrupted map, since
mapping has no self-correction against VO drift the way replanning does).

Mirrors run_sim_dimsim_agentic.py exactly (same provider-choice/safety-
monitor-choice menu, reused directly rather than duplicated).

BEFORE RUNNING:
  1. Start the DimSim relay:
       cd ~/Documents/kios/DimSim
       deno run --allow-all --unstable-net dimos-cli/cli.ts dev --scene apt_no_glass
  2. Open http://localhost:8090/?dimos=1&scene=apt_no_glass and wait for
     "[dimos] Bridge connected. Sensor publishing active." in the browser console.
  3. THEN run this script.

You'll be asked to pick an LLM provider below (same menu as
run_sim_dimsim_agentic.py), then:
  - Open http://localhost:5555 (the natural-language console) and type
    something like "start exploring".
  - Watch this terminal and the Rerun viewer for the drone's progress.

VERIFY: check the startup "Transport" log lines. AgenticFrontierSelector's
and ReplanningAStarPlanner's odom streams must show a topic containing
"odom_vo". DimSimDepthLidarModule's odom stream must show the raw
ground-truth "/odom" topic -- if it shows "odom_vo" instead, someone
re-added the mapping remap and the corrupted-map problem will come back.

Watch "VO vs ground-truth" log lines throughout -- navigation now depends
on VO's accuracy (mapping does not).
"""
from __future__ import annotations

from dimos.core.coordination.module_coordinator import ModuleCoordinator

from blueprints.sim_dimsim_agentic_vo_blueprint import build_sim_dimsim_agentic_vo
from run_sim_dimsim_agentic import _choose_provider, _choose_safety_monitor

if __name__ == "__main__":
    llm_provider, llm_model = _choose_provider()
    safety_monitor_enabled = _choose_safety_monitor()
    if not safety_monitor_enabled:
        print("Safety monitor disabled for this run.")

    coordinator = ModuleCoordinator.build(
        build_sim_dimsim_agentic_vo(
            llm_provider=llm_provider,
            llm_model=llm_model,
            safety_monitor_enabled=safety_monitor_enabled,
        ),
        {},
    )

    if not coordinator.health_check():
        print("Health check failed -- is the DimSim relay running and browser connected?")
        coordinator.stop()
        exit(1)

    print(f"DimSim agentic (VO-driven navigation, ground-truth mapping) started with {coordinator.n_modules} modules.")
    print("Open http://localhost:5555 for the natural-language console.")
    print("Type something like 'start exploring' to begin.")
    print("Ctrl+C to stop.")

    try:
        coordinator.loop()
    except KeyboardInterrupt:
        print("Stopping...")
        coordinator.stop()
