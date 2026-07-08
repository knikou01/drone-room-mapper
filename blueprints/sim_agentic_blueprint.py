#!/usr/bin/env python3
"""Natural language control blueprint for the simulation.
Test LLM → drone skill flow before connecting real hardware.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.web_human_input import WebInput
from dimos.core.coordination.blueprints import autoconnect
from blueprints.sim_blueprint import drone_sim
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

if not os.environ.get("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY or GEMINI_API_KEY required in .env")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

SYSTEM_PROMPT = """\
You are controlling a simulated drone for indoor room mapping.

## Your available tools (ONLY use these exact tool names)
- begin_exploration — start autonomous exploration
- end_exploration — stop exploration  
- start_patrol — start patrol mode
- stop_patrol — stop patrol mode
- agent_send — send a message
- list_modules — list available modules
- server_status — check server status

## CRITICAL
You do NOT have observe(), takeoff(), move(), or land() in simulation.
Do NOT attempt to call any tool not listed above.
Only call tools from the list above. Never hallucinate tool names.

## Rules
- Use begin_exploration to start mapping
- Use end_exploration to stop
- Report what you are doing in plain text
"""

sim_agentic = autoconnect(
    drone_sim,
    McpServer.blueprint(),
    McpClient.blueprint(
        system_prompt=SYSTEM_PROMPT,
        model="groq:llama-3.3-70b-versatile",
    ),
    WebInput.blueprint(),
)

__all__ = ["sim_agentic"]
