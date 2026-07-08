"""
Standalone probe: confirms DimSim's bridge is actually delivering LCM packets
to this machine, fully independent of any DimOS blueprint.

Run this WHILE `deno run --allow-all --unstable-net dimos-cli/cli.ts dev
--scene apt` is running and the browser tab is open and connected.

Usage:
    cd drone-room-mapper
    python -m lcm_probe.probe

Expected: within a few seconds you should see "odom" lines printing
repeatedly (DimSim's bridge publishes odom at ~50Hz by default). If you see
NOTHING after ~10 seconds, the relay is not actually reaching this process —
check that the relay's LCM/UDP multicast actually started (no error in its
own terminal), and that nothing (e.g. a firewall) is blocking UDP multicast
traffic on this machine.
"""
from __future__ import annotations

import time

from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.sensor_msgs.Image import Image

_counts = {"odom": 0, "lidar": 0, "color_image": 0, "depth_image": 0}


def _make_handler(name: str):
    def _handler(msg, topic) -> None:
        _counts[name] += 1
        if _counts[name] <= 3 or _counts[name] % 50 == 0:
            print(f"[probe] {name}: received message #{_counts[name]} ({type(msg).__name__})")
            if name == "odom":
                p = msg.position
                q = msg.orientation
                print(f"[probe]   position:    x={p.x:.4f} y={p.y:.4f} z={p.z:.4f}")
                print(f"[probe]   orientation: x={q.x:.4f} y={q.y:.4f} z={q.z:.4f} w={q.w:.4f}")
                print(f"[probe]   (if this is pure yaw, x and y should stay ~0.0 across samples)")
    return _handler


def main() -> None:
    lcm = LCM()
    lcm.start()

    # Topic names taken directly from DimosBridge's CH_* constants
    # (dimosBridge.ts): /odom, /lidar, /color_image, /depth_image
    lcm.subscribe(Topic("/odom", PoseStamped), _make_handler("odom"))
    lcm.subscribe(Topic("/lidar", PointCloud2), _make_handler("lidar"))
    lcm.subscribe(Topic("/color_image", Image), _make_handler("color_image"))
    lcm.subscribe(Topic("/depth_image", Image), _make_handler("depth_image"))

    print("[probe] Subscribed to /odom, /lidar, /color_image, /depth_image.")
    print("[probe] Waiting for messages... (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(2.0)
            total = sum(_counts.values())
            if total == 0:
                print("[probe] Still nothing received. Bridge may not be reaching this process.")
            else:
                print(f"[probe] Totals so far: {_counts}")
    except KeyboardInterrupt:
        print("\n[probe] Stopping.")
        lcm.stop()


if __name__ == "__main__":
    main()
