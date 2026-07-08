#!/usr/bin/env python3
"""Entry point for real drone passive mapping.

Mirrors run_sim.py's structure but for the real drone. No explore command
is sent -- the mapping pipeline is entirely passive. Fly the drone by hand
and watch the 3D map build in the Rerun viewer.

Prerequisites:
  - Drone connected via MAVLink (check with: dimos run drone-basic)
  - /odom, /video, /telemetry, /distance_sensors streams flowing
  - cruise_altitude_m set correctly in blueprints/real_mapping_blueprint.py

Usage:
    python run_real_mapping.py
"""

import time

from dimos.core.coordination.module_coordinator import ModuleCoordinator

from blueprints.real_mapping_blueprint import real_mapping

if __name__ == "__main__":
    coordinator = ModuleCoordinator.build(real_mapping, {})

    if not coordinator.health_check():
        print("Health check failed -- is the drone connected?")
        coordinator.stop()
        exit(1)

    print(f"Passive mapping started with {coordinator.n_modules} modules.")
    print("Fly the drone by hand. Watch the 3D map in the Rerun viewer.")
    print("Ctrl+C to stop.")

    try:
        coordinator.loop()
    except KeyboardInterrupt:
        print("Stopping...")
        coordinator.stop()
