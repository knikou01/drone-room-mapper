#!/usr/bin/env python3
"""Run the real drone agentic blueprint.

Usage:
    cd ~/Documents/kios/drone-room-mapper
    source .venv/bin/activate
    python3 run_agentic_real.py

Then open http://localhost:5555 in your browser.
Type natural language commands like:
    "take off"
    "move forward slowly"
    "what do you see?"
    "land"
"""

from dimos.core.coordination.module_coordinator import ModuleCoordinator
from blueprints.drone_agentic_real_blueprint import drone_agentic_real

if __name__ == "__main__":
    coordinator = ModuleCoordinator.build(drone_agentic_real, {})

    if not coordinator.health_check():
        print("Health check failed")
        coordinator.stop()
        exit(1)

    print(f"Starting real drone agentic control ({coordinator.n_modules} modules)")
    print("Open http://localhost:5555 to send commands")
    print("Ctrl+C to stop")

    try:
        coordinator.loop()
    except KeyboardInterrupt:
        print("Stopping...")
        coordinator.stop()
