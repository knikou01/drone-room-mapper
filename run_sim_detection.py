import time

from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos_lcm.std_msgs import Bool

from blueprints.sim_detection_blueprint import drone_sim_detection

if __name__ == '__main__':
    coordinator = ModuleCoordinator.build(drone_sim_detection, {})

    if not coordinator.health_check():
        print("Health check failed")
        coordinator.stop()
        exit(1)

    print(f"Starting simulation with {coordinator.n_modules} modules...")
    time.sleep(3.0)

    lcm = LCM()
    explore_msg = Bool()
    explore_msg.data = True
    lcm.publish("/explore_cmd#std_msgs.Bool", explore_msg.lcm_encode())
    print("Sent explore command")

    try:
        coordinator.loop()
    except KeyboardInterrupt:
        print("Stopping simulation...")
        coordinator.stop()
