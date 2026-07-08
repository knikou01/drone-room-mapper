"""
Triggers real autonomous frontier exploration, replacing the throwaway
constant-forward publish_cmd_vel.py test script.

This publishes /explore_cmd#std_msgs.Bool, exactly as run_sim.py does for
the PyBullet blueprint. WavefrontFrontierExplorer.explore() then takes over,
publishing goal_request -> ReplanningAStarPlanner -> MovementManager.cmd_vel,
with real turning/replanning logic — unlike publish_cmd_vel.py's constant
forward command, which (confirmed via lcm_probe/probe.py) drove the agent
straight into an obstacle in the apt scene and then could go no further,
since it never turned.

Usage (run AFTER run_sim_dimsim_detection.py is already up and the relay/
browser are connected):
    cd drone-room-mapper
    python -m lcm_probe.start_exploration
"""
from __future__ import annotations

from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos_lcm.std_msgs import Bool

if __name__ == "__main__":
    lcm = LCM()
    explore_msg = Bool()
    explore_msg.data = True
    lcm.publish("/explore_cmd#std_msgs.Bool", explore_msg.lcm_encode())
    print("[explore] Sent explore command.")
