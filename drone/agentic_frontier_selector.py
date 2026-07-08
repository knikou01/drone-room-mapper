#!/usr/bin/env python3
"""
A subclass of dimos.navigation.frontier_exploration.
wavefront_frontier_goal_selector.WavefrontFrontierExplorer. Does not modify
that file or any other dimos source -- it only imports and extends the
class.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from typing import Any, Literal

import numpy as np
from PIL import Image as PILImage, ImageDraw
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontConfig,
    WavefrontFrontierExplorer,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class AgenticFrontierConfig(WavefrontConfig):
    # All WavefrontConfig fields (min_frontier_perimeter, occupancy_threshold,
    # safe_distance, lookahead_distance, max_explored_distance,
    # info_gain_threshold, num_no_gain_attempts, goal_timeout) are inherited
    # unchanged -- the geometric scoring stays exactly as already tuned.
    frontier_candidates_k: int = 5
    # Minimum distance from any obstacle a frontier candidate must have
    # before it's shown to the LLM. Candidates that fail this check are
    # silently dropped -- the LLM can't pick an unreachable goal if it
    # never sees it. Set to match (or slightly exceed) the path planner's
    # own min_clearance = robot_rotation_diameter / 2. Default 1.2m is
    # conservative for most indoor robots; lower it if valid frontiers are
    # being spuriously filtered in tight spaces.
    min_frontier_clearance_m: float = 1.2
    map_render_max_px: int = 512
    map_crop_padding_cells: int = 5
    camera_max_width_px: int = 768
    llm_provider: Literal["openai", "anthropic", "gemini"] = "gemini"
    llm_model: str = "gemini-3.5-flash"
    llm_temperature: float = 0.2


_SELECT_PROMPT = """\
You are helping a drone choose where to explore next while mapping a room.
Below are {n} candidate frontier points the mapping system already
identified as safe and reachable (far enough from walls/obstacles, and
geometrically on the boundary between mapped and unmapped space). They are
already ranked by a scoring heuristic combining distance, information gain,
obstacle clearance, and movement momentum -- candidate 0 is the heuristic's
top pick.

You are given two images:
1. A top-down rendering of the ENTIRE room as mapped so far (not just the
   immediate area around the drone). Light gray = explored free space,
   dark gray = unexplored, near-black = walls/obstacles. The drone's
   current position is a blue dot; each candidate is a numbered red circle
   at its actual location on the map.
2. The drone's current forward camera frame.

Use the full map to judge which candidate borders the largest or most
promising unexplored area, and use the camera frame to catch things the
map can't show -- e.g. a candidate that's actually behind glass or a
mirror rather than a real opening. Confirm the heuristic's pick, override
it with a better one, or declare the room fully explored.

Candidates (index: distance and direction offset from current position):
{candidates}

Respond with ONLY a JSON object of this exact shape, no other text:
{{
  "reasoning": "<=2 sentences",
  "candidate_index": 0,
  "stop": false
}}
Set "candidate_index" to null and "stop" to true once you judge the room
fully explored.
"""


class AgenticFrontierSelector(WavefrontFrontierExplorer):
    """Drop-in alternative to WavefrontFrontierExplorer: same Inputs /
    Outputs / skills, plus one additional input (`video`), with frontier
    *selection* delegated to an LLM instead of always taking the
    heuristic's top-ranked candidate.

    Falls back to the heuristic's own choice (frontiers[0], the base
    class's original behavior) whenever the LLM call fails or returns
    something unusable -- this can't behave worse than the plain
    algorithmic explorer, only potentially better.
    """

    config: AgenticFrontierConfig

    # Matches SimCameraModule's `color_image: Out[Image]` by (name, type)
    # so autoconnect wires them automatically -- no .remappings() needed.
    # Falls back gracefully (map-only decisions) when no module publishing
    # color_image is present in the blueprint.
    color_image: In[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._image_lock = threading.Lock()
        self._latest_image: Image | None = None
        self._llm_client: Any = None
        self._llm_history: list[dict[str, Any]] = []
        self._step_count = 0

    @rpc
    def start(self) -> None:
        super().start()
        if self.color_image.transport is not None:
            self.register_disposable(Disposable(self.color_image.subscribe(self._on_image)))
        else:
            logger.warning(
                "AgenticFrontierSelector: no color_image stream connected -- "
                "LLM will decide from map only. Add SimCameraModule to your blueprint."
            )
        # Built here, not __init__: Modules run in forkserver worker
        # processes, so the LLM SDK client gets constructed after fork.
        self._llm_client = self._build_llm_client()

    def _on_image(self, img: Image) -> None:
        with self._image_lock:
            self._latest_image = img

    # -------------------------------------------------------------------
    # The only behavioral override. Everything above the `top_k = ...`
    # line duplicates WavefrontFrontierExplorer.get_exploration_goal's
    # info-gain/no-gain bookkeeping verbatim -- there's no smaller seam to
    # hook without modifying dimos source. If a future dimos version
    # splits this method into "check termination" + "pick best", this
    # override should shrink to just the latter.
    # -------------------------------------------------------------------

    def get_exploration_goal(self, robot_pose: Vector3, costmap: OccupancyGrid) -> Vector3 | None:
        if len(self.explored_goals) > 5 and self.last_costmap is not None:
            current_info = self._count_costmap_information(costmap)
            last_info = self._count_costmap_information(self.last_costmap)
            if last_info > 0:
                info_increase_percent = (current_info - last_info) / last_info
                if info_increase_percent < self.config.info_gain_threshold:
                    self.no_gain_counter += 1
                    if self.no_gain_counter >= self.config.num_no_gain_attempts:
                        logger.info(
                            "AgenticFrontierSelector: stopping due to inherited no-information-gain "
                            "heuristic (%d consecutive low-gain attempts), not an LLM decision",
                            self.no_gain_counter,
                        )
                        self.no_gain_counter = 0
                        self.stop_exploration()
                        return None
                else:
                    self.no_gain_counter = 0

        frontiers = self.detect_frontiers(robot_pose, costmap)  # inherited, unmodified
        if not frontiers:
            self.last_costmap = costmap
            self.reset_exploration_session()
            return None

        top_k = frontiers[: self.config.frontier_candidates_k]
        selected = self._select_frontier_with_llm(top_k, robot_pose, costmap)
        self.last_costmap = costmap

        if selected is None:
            logger.info("AgenticFrontierSelector: LLM judged exploration complete; stopping")
            self.stop_exploration()
            return None

        self._update_exploration_direction(robot_pose, selected)
        self.mark_explored_goal(selected)
        return selected

    # -------------------------------------------------------------------
    # LLM selection among pre-validated candidates
    # -------------------------------------------------------------------

    def _select_frontier_with_llm(
        self, candidates: list[Vector3], robot_pose: Vector3, costmap: OccupancyGrid
    ) -> Vector3 | None:
        with self._image_lock:
            image = self._latest_image

        self._step_count += 1
        step = self._step_count

        # ------------------------------------------------------------------
        # Hard clearance pre-filter: drop any candidate the path planner
        # won't be able to approach safely. This runs BEFORE the LLM sees
        # anything -- the LLM can't pick an unreachable goal if it's never
        # offered one.
        #
        # _compute_distance_to_obstacles is inherited from
        # WavefrontFrontierExplorer unchanged. It returns the distance (m)
        # from the candidate to the nearest occupied cell, capped at
        # safe_distance when no obstacle is found within the search radius.
        # We use our own threshold (min_frontier_clearance_m) rather than
        # the inherited safe_distance, so they're independently tunable.
        # ------------------------------------------------------------------
        reachable = []
        for c in candidates:
            clearance = self._compute_distance_to_obstacles(c, costmap)
            if clearance >= self.config.min_frontier_clearance_m:
                reachable.append(c)
            else:
                logger.info(
                    "AgenticFrontierSelector step=%d: dropping candidate (%.2f, %.2f) -- "
                    "only %.2fm from nearest obstacle (min %.2fm)",
                    step, c.x, c.y, clearance, self.config.min_frontier_clearance_m,
                )

        if not reachable:
            logger.warning(
                "AgenticFrontierSelector step=%d: ALL %d candidates failed the clearance "
                "filter (min_frontier_clearance_m=%.2f). Skipping LLM call this tick -- "
                "the base heuristic will find new frontiers on the next iteration. "
                "If this repeats, lower min_frontier_clearance_m in AgenticFrontierConfig.",
                step, len(candidates), self.config.min_frontier_clearance_m,
            )
            return candidates[0]  # fall back to heuristic's top pick rather than stalling

        if len(reachable) < len(candidates):
            logger.info(
                "AgenticFrontierSelector step=%d: %d/%d candidates passed clearance filter",
                step, len(reachable), len(candidates),
            )

        candidate_lines = []
        for idx, c in enumerate(reachable):
            dx, dy = c.x - robot_pose.x, c.y - robot_pose.y
            dist = (dx**2 + dy**2) ** 0.5
            candidate_lines.append(f"{idx}: {dist:.1f}m away (dx={dx:.1f}, dy={dy:.1f})")

        prompt = _SELECT_PROMPT.format(
            n=len(reachable),
            candidates="\n".join(candidate_lines),
        )
        map_image_png = self._render_costmap_image(costmap, robot_pose, reachable)

        logger.info(
            "AgenticFrontierSelector step=%d: %d reachable candidates, camera_frame=%s",
            step,
            len(reachable),
            "present" if image is not None else "MISSING -- LLM is deciding from map only",
        )

        try:
            result = self._call_llm(prompt, map_image_png, image)
        except Exception:
            logger.exception(
                "AgenticFrontierSelector step=%d: LLM call failed, falling back to top-ranked "
                "reachable frontier",
                step,
            )
            return reachable[0]

        self._llm_history.append(result)
        self._llm_history = self._llm_history[-6:]

        logger.info(
            "AgenticFrontierSelector step=%d decision: stop=%s candidate_index=%s reasoning=%r",
            step,
            result.get("stop"),
            result.get("candidate_index"),
            result.get("reasoning"),
        )

        if result.get("stop"):
            logger.info(
                "AgenticFrontierSelector step=%d: LLM chose to stop. Full reasoning: %s",
                step,
                result.get("reasoning"),
            )
            return None

        idx = result.get("candidate_index")
        if isinstance(idx, int) and 0 <= idx < len(reachable):
            selected = reachable[idx]
            # Post-selection safety net: re-verify clearance on the exact
            # pick in case _compute_distance_to_obstacles had a borderline
            # result at filter time. If it fails, step down through the
            # remaining reachable candidates rather than returning None.
            final_clearance = self._compute_distance_to_obstacles(selected, costmap)
            if final_clearance >= self.config.min_frontier_clearance_m:
                return selected
            for fallback in reachable:
                if fallback is not selected:
                    fb_clearance = self._compute_distance_to_obstacles(fallback, costmap)
                    if fb_clearance >= self.config.min_frontier_clearance_m:
                        logger.warning(
                            "AgenticFrontierSelector step=%d: LLM pick failed re-check "
                            "(%.2fm clearance), using fallback (%.2f, %.2f)",
                            step, final_clearance, fallback.x, fallback.y,
                        )
                        return fallback

        logger.warning(
            "AgenticFrontierSelector step=%d: LLM returned unusable candidate_index=%r, "
            "falling back to top-ranked reachable frontier",
            step,
            idx,
        )
        return reachable[0]

    def _render_costmap_image(
        self, costmap: OccupancyGrid, robot_pose: Vector3, candidates: list[Vector3]
    ) -> bytes:
        """Render the entire explored region of the costmap (auto-cropped to
        the bounding box of every known cell, not a fixed window around the
        robot) as a PNG, with the robot and every candidate marked and
        numbered at their true positions. Verified against a synthetic grid
        before wiring in -- see chat for the rendered sample."""
        grid = costmap.grid
        known_mask = grid != CostValues.UNKNOWN
        pad = self.config.map_crop_padding_cells
        if known_mask.any():
            ys, xs = np.nonzero(known_mask)
            y0, y1 = max(0, int(ys.min()) - pad), min(costmap.height, int(ys.max()) + pad + 1)
            x0, x1 = max(0, int(xs.min()) - pad), min(costmap.width, int(xs.max()) + pad + 1)
        else:
            y0, y1, x0, x1 = 0, costmap.height, 0, costmap.width

        crop = grid[y0:y1, x0:x1]
        rgb = np.zeros((*crop.shape, 3), dtype=np.uint8)
        rgb[crop == CostValues.UNKNOWN] = (90, 90, 90)
        rgb[crop != CostValues.UNKNOWN] = (235, 235, 235)  # explored/free, painted first
        rgb[crop >= self.config.occupancy_threshold] = (20, 20, 20)  # obstacle, overrides

        # Flip vertically so increasing grid-y renders toward the top of the
        # image (numpy/PIL otherwise put row 0 -- the smallest y -- at the
        # top, which inverts the intuitive "up" reading of the map).
        rgb = np.flipud(rgb)
        img = PILImage.fromarray(rgb, mode="RGB")

        scale = max(1, self.config.map_render_max_px // max(img.width, img.height))
        img = img.resize((img.width * scale, img.height * scale), PILImage.NEAREST)
        draw = ImageDraw.Draw(img)
        crop_h = crop.shape[0]

        def to_px(world_pos: Vector3) -> tuple[int, int]:
            g = costmap.world_to_grid(world_pos)
            px = (int(g.x) - x0) * scale
            py = (crop_h - 1 - (int(g.y) - y0)) * scale
            return px, py

        rx, ry = to_px(robot_pose)
        draw.ellipse([rx - 6, ry - 6, rx + 6, ry + 6], fill=(0, 120, 255))

        for idx, c in enumerate(candidates):
            cx, cy = to_px(c)
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=(220, 30, 30), width=2)
            draw.text((cx + 7, cy - 7), str(idx), fill=(220, 30, 30))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # -------------------------------------------------------------------
    # LLM plumbing -- isolated so it's easy to swap providers.
    # -------------------------------------------------------------------

    def _build_llm_client(self) -> Any:
        if self.config.llm_provider == "openai":
            import openai  # requires OPENAI_API_KEY in the environment

            return openai.OpenAI()
        elif self.config.llm_provider == "anthropic":
            import anthropic  # requires ANTHROPIC_API_KEY in the environment

            return anthropic.Anthropic()
        elif self.config.llm_provider == "gemini":
            # pip install google-genai. genai.Client() reads GEMINI_API_KEY
            # from the environment automatically -- no need to pass it.
            from google import genai

            return genai.Client()
        raise ValueError(f"unknown llm_provider: {self.config.llm_provider}")

    def _image_block(self, b64: str, media_type: str) -> dict[str, Any]:
        """One image content block, shaped for OpenAI/Anthropic (Gemini is
        built separately in _call_llm since its SDK takes raw bytes, not
        base64 dicts)."""
        if self.config.llm_provider == "openai":
            return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}

    def _call_llm(
        self, prompt: str, map_image_png: bytes, camera_image: Image | None
    ) -> dict[str, Any]:
        camera_jpeg: bytes | None = None
        if camera_image is not None:
            try:
                # Image.to_base64() handles BGR/RGB conversion and JPEG
                # encoding itself, and max_width keeps the request small --
                # confirmed against the real dimos.msgs.sensor_msgs.Image.
                # We decode back to bytes here for providers (Gemini) that
                # want raw bytes rather than a base64 string.
                camera_b64 = camera_image.to_base64(
                    quality=80, max_width=self.config.camera_max_width_px
                )
                camera_jpeg = base64.b64decode(camera_b64)
            except Exception:
                logger.exception(
                    "AgenticFrontierSelector: failed to encode camera frame, "
                    "continuing with map image only"
                )

        if self.config.llm_provider == "gemini":
            from google.genai import types

            contents: list[Any] = [
                prompt,
                types.Part.from_bytes(data=map_image_png, mime_type="image/png"),
            ]
            if camera_jpeg is not None:
                contents.append(types.Part.from_bytes(data=camera_jpeg, mime_type="image/jpeg"))

            resp = self._llm_client.models.generate_content(
                model=self.config.llm_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.config.llm_temperature,
                    response_mime_type="application/json",
                ),
            )
            raw = resp.text
            return json.loads(raw)

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.append(self._image_block(base64.b64encode(map_image_png).decode(), "image/png"))
        if camera_jpeg is not None:
            content.append(self._image_block(base64.b64encode(camera_jpeg).decode(), "image/jpeg"))

        if self.config.llm_provider == "openai":
            resp = self._llm_client.chat.completions.create(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
            )
            raw = resp.choices[0].message.content
        else:  # anthropic
            resp = self._llm_client.messages.create(
                model=self.config.llm_model,
                max_tokens=200,
                temperature=self.config.llm_temperature,
                messages=[{"role": "user", "content": content}],
            )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

        return json.loads(raw)
