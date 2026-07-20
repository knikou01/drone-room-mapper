#!/usr/bin/env python3
"""Entry point for DimSim visual-SLAM + agentic (LLM-driven) exploration.

Same vision-based mapping as run_sim_dimsim_vo.py (Phase 2: depth camera
instead of lidar, ground-truth pose), but frontier selection is delegated to
an LLM (AgenticFrontierSelector) instead of the plain geometric heuristic,
and a natural-language console is available for issuing commands instead of
lcm_probe/start_exploration.

BEFORE RUNNING:
  1. Start the DimSim relay:
       cd ~/Documents/kios/DimSim
       deno run --allow-all --unstable-net dimos-cli/cli.ts dev --scene apt
  2. Open http://localhost:8090/?dimos=1&scene=apt and wait for
     "[dimos] Bridge connected. Sensor publishing active." in the browser console.
  3. THEN run this script.

You'll be asked to pick an LLM provider below, then:
  - Open http://localhost:5555 (the natural-language console) and type
    something like "start exploring".
  - Watch this terminal and the Rerun viewer for the drone's progress.

Requires GEMINI_API_KEY and/or GROQ_API_KEY in .env depending on your
choice below. CONFIRMED BUG, FIXED (2026-07-16): the natural-language
console used to always use Groq's Llama 4 Scout regardless of this
choice -- picking Gemini specifically to avoid a rate-limited/exhausted
Groq quota didn't actually help, since the console crashed on the same
error anyway (dimos's McpClient doesn't catch LLM call failures in its
own agent loop, so a rate limit kills the whole console thread). The
console now uses whichever provider/model you pick below for both
frontier selection AND the console.
"""
from __future__ import annotations

from dimos.core.coordination.module_coordinator import ModuleCoordinator

from blueprints.sim_dimsim_agentic_blueprint import build_sim_dimsim_agentic

_PROVIDER_CHOICES = {
    "1": ("gemini", "gemini-3.5-flash", "Gemini -- best quality, occasional traffic/rate-limit issues"),
    # CONFIRMED BUG, FIXED (2026-07-20): meta-llama/llama-4-scout-17b-16e-instruct
    # was removed from Groq's catalog entirely at some point since it was
    # last confirmed working (2026-07-09) -- a live run hit
    # groq.NotFoundError: 404 model_not_found. Re-verified live via
    # client.models.list() rather than guessing a replacement from
    # search/docs (same discipline as the original model pick): none of
    # the other current models on this account accept multimodal
    # (image) content except qwen/qwen3.6-27b -- confirmed by sending a
    # real test image to each candidate, not by trusting the model name.
    # qwen/qwen3.6-27b is a reasoning model that can emit a <think>...
    # preamble, but confirmed live with response_format=json_object (the
    # same mode _call_llm already uses) the response comes back as clean
    # JSON with no thinking-block leakage -- no parsing changes needed.
    "2": (
        "groq",
        "qwen/qwen3.6-27b",
        "Qwen3.6 27B (Groq, free tier) -- open-source, fast",
    ),
}


_DEFAULT_PROVIDER_CHOICE = "2"  # Groq -- default for fast iterative testing
_DEFAULT_SAFETY_MONITOR_ENABLED = False  # off by default -- see _choose_safety_monitor


def _choose_provider() -> tuple[str, str]:
    print("Choose the LLM provider for frontier selection:")
    for key, (_, _, label) in _PROVIDER_CHOICES.items():
        default_marker = " (default)" if key == _DEFAULT_PROVIDER_CHOICE else ""
        print(f"  {key}) {label}{default_marker}")
    choice = input("> ").strip()
    if not choice:
        choice = _DEFAULT_PROVIDER_CHOICE
    elif choice not in _PROVIDER_CHOICES:
        print(f"Unrecognized choice {choice!r}, defaulting to {_PROVIDER_CHOICES[_DEFAULT_PROVIDER_CHOICE][2]}.")
        choice = _DEFAULT_PROVIDER_CHOICE
    provider, model, label = _PROVIDER_CHOICES[choice]
    print(f"Using: {label}")
    return provider, model


def _choose_safety_monitor() -> bool:
    # Added 2026-07-16: the background hazard safety monitor is pure
    # overhead (and a source of false positives -- confirmed live, e.g.
    # flagging an ordinary door frame as "may contain glass") on a scene
    # with no glass at all (apt_no_glass). Default changed to off
    # (2026-07-16) -- current testing has mostly been on apt_no_glass
    # where the monitor has nothing real to catch; turn it back on
    # explicitly when testing hazard avoidance on apt.
    choice = input("Enable the background hazard safety monitor? [y/N] ").strip().lower()
    if not choice:
        return _DEFAULT_SAFETY_MONITOR_ENABLED
    return choice in ("y", "yes")


if __name__ == "__main__":
    llm_provider, llm_model = _choose_provider()
    safety_monitor_enabled = _choose_safety_monitor()
    if not safety_monitor_enabled:
        print("Safety monitor disabled for this run.")

    coordinator = ModuleCoordinator.build(
        build_sim_dimsim_agentic(
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

    print(f"DimSim agentic exploration started with {coordinator.n_modules} modules.")
    print("Open http://localhost:5555 for the natural-language console.")
    print("Type something like 'start exploring' to begin.")
    print("Ctrl+C to stop.")

    try:
        coordinator.loop()
    except KeyboardInterrupt:
        print("Stopping...")
        coordinator.stop()
