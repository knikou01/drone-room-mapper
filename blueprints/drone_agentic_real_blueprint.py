#!/usr/bin/env python3
"""Natural language control blueprint for the real DJI Mavic 2 Enterprise drone.

Builds on drone_basic (connection + camera + vis) and adds LLM agent control
via Gemini. The agent can control the drone using the skills exposed by
DroneConnectionModule: fly_to, move, takeoff, land, observe, follow_object.

Usage:
    cd ~/Documents/kios/drone-room-mapper
    source .venv/bin/activate
    python3 run_agentic_real.py

Then open http://localhost:5555 in your browser to send natural language
commands to the drone.

Requirements:
    - GEMINI_API_KEY set in .env
    - RosettaDrone running on phone, connected to drone, sending to laptop IP:14550
    - QGroundControl closed (port conflict)
    - Props removed from drone for safety during testing
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.web_human_input import WebInput
from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.drone.blueprints.basic.drone_basic import drone_basic

# Load GEMINI_API_KEY from .env
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Verify key is present — fail early with a clear message
if not os.environ.get("GEMINI_API_KEY"):
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Add it to drone-room-mapper/.env as:\n"
        "GEMINI_API_KEY=your_key_here"
    )

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

SYSTEM_PROMPT = """\
You are controlling a DJI Mavic 2 Enterprise drone via MAVLink for indoor room mapping.

## Your available skills
- takeoff(altitude=1.5) — take off to 1.5 metres above the floor. Always do this first.
- land() — land at current position.
- move(x, y, z, duration) — move at velocity (m/s) for duration seconds.
  x = forward (positive) / backward (negative)
  y = left (positive) / right (negative)
  z = up (positive) / down (negative)
  Example: move(x=0.5, y=0.0, z=0.0, duration=2.0) moves forward 1 metre.
- fly_to(lat, lon, alt) — fly to GPS coordinates. Only use outdoors with a GPS fix.
- observe() — capture and return the current camera frame. Use this to look around.
- follow_object(object_description, duration) — visually follow an object.

## Safety rules — follow these strictly
1. Always confirm the user's intent before takeoff.
2. Never fly faster than 0.5 m/s indoors.
3. Keep altitude between 0.5 m and 2.0 m indoors.
4. If unsure about a command, ask for clarification rather than guessing.
5. Use observe() before moving into an unknown area.
6. If the user says stop, land, or emergency — call land() immediately.

## Indoor mapping behaviour
When asked to map a room:
1. Takeoff to 1.5 m.
2. Use move() to fly a slow systematic pattern (e.g. forward, rotate, forward again).
3. Use observe() periodically to check for obstacles.
4. Land when done or battery is low.

Report what you see and what you're doing at each step.
Always confirm actions and report results clearly.

When calling tools, always use valid JSON for arguments.
Example: takeoff(altitude=1.5) means call takeoff with {"altitude": 1.5}
"""

drone_agentic_real = autoconnect(
    drone_basic,
    McpServer.blueprint(),
    McpClient.blueprint(
        system_prompt=SYSTEM_PROMPT,
        model="groq:llama-3.3-70b-versatile",
    ),
    WebInput.blueprint(),
)

__all__ = ["drone_agentic_real", "SYSTEM_PROMPT"]
