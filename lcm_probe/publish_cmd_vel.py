"""
Throwaway script: publishes a constant forward cmd_vel over LCM so we can
confirm DimSim's ServerPhysics actually moves in response (per physics.ts:
Python cmd_vel -> LCM -> Deno physics step -> LCM odom -> Python).

This is a temporary stand-in for what MovementManager/ReplanningAStarPlanner
will eventually publish once wired into a real blueprint. Confirms the
control path works before building anything further on top of it.

Usage:
    cd drone-room-mapper
    python -m lcm_probe.publish_cmd_vel

Run this WHILE lcm_probe/probe.py is also running in another terminal —
watch odom's position actually change instead of sitting frozen at spawn.
Ctrl+C to stop (the agent should stop moving within CMD_VEL_TIMEOUT_MS=500ms
per physics.ts's safety timeout, once messages stop arriving).
"""
from __future__ import annotations

import time

from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3

# physics.ts's CMD_VEL_TIMEOUT_MS is 500ms — publish well under that to keep
# the agent continuously moving rather than stop-starting.
_PUBLISH_INTERVAL_S = 0.1
_FORWARD_SPEED = 0.3  # scaled by physics.ts's speedScale (default 3.0) -> ~0.9 m/s actual


def main() -> None:
    lcm = LCM()
    lcm.start()

    topic_str = "/cmd_vel#geometry_msgs.Twist"
    twist = Twist(
        linear=Vector3(_FORWARD_SPEED, 0.0, 0.0),
        angular=Vector3(0.0, 0.0, 0.0),
    )

    print(f"[cmd_vel] Publishing forward Twist (linear.x={_FORWARD_SPEED}) every {_PUBLISH_INTERVAL_S}s.")
    print("[cmd_vel] Watch lcm_probe/probe.py's odom output in another terminal.")
    print("[cmd_vel] Ctrl+C to stop.")

    try:
        while True:
            # LCMPubSubBase.publish(topic, message: bytes) — confirmed signature
            # from dimos/protocol/pubsub/impl/lcmpubsub.py. Encoding explicitly
            # here rather than relying on LCMEncoderMixin to accept a live
            # object transparently, since that mixin's behavior wasn't directly
            # confirmed — this works regardless of which is true.
            lcm.publish(topic_str, twist.lcm_encode())
            time.sleep(_PUBLISH_INTERVAL_S)
    except KeyboardInterrupt:
        print("\n[cmd_vel] Stopping.")
        lcm.stop()


if __name__ == "__main__":
    main()
