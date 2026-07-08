#!/usr/bin/env python3
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from blueprints.sim_agentic_blueprint import sim_agentic

if __name__ == "__main__":
    coordinator = ModuleCoordinator.build(sim_agentic, {})
    if not coordinator.health_check():
        coordinator.stop()
        exit(1)
    print(f"Simulation agentic control ready ({coordinator.n_modules} modules)")
    print("Open http://localhost:5555 to send commands")
    try:
        coordinator.loop()
    except KeyboardInterrupt:
        coordinator.stop()
