"""Stage 3 Agent B — shift_decision (one-shot LLM call).

Given a probe failure + collected images + scene state + history, decide
how to shift the scene (or whether to call it success / give up). Pure
function — no tool calls, no internal retry loop. Single LLM invocation,
JSON-parsed output.

Designed as the inner-most decision step of the Stage 3 retry loop. Called
once per outer round. The orchestrator clips the returned shift to the
per-round / cumulative caps before applying.

Prompt lives in ``shift_decision_prompt.md`` (loaded once, cached via
Anthropic prompt caching across calls in the same loop).
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage

from ar2s.agent_configs.models import chat_model_for
from ar2s.agents_sysid.skills.probe_grasp import GraspProbeResult


_PER_ROUND_CAP_MM = 10
_CUMULATIVE_CAP_MM = 50
_IMAGE_MAX_DIM = 720

_PROMPT_PATH = Path(__file__).parent / "shift_decision_prompt.md"


# -------------------------------------------------------------------
# Inputs and output dataclasses
# -------------------------------------------------------------------

@dataclass
class RoundSummary:
    """Compact record of one prior round — fed to Agent B as history."""
    round_idx: int
    cumulative_shift_mm: tuple                 # (dx, dy, dz) entering this round
    peak_lift_mm: float
    closest_distance_mm: float
    decision: str                              # "tweak" | "success" | "give_up"
    reasoning_summary: str                     # one-line summary


@dataclass
class ShiftDecision:
    decision: str                              # "success" | "tweak" | "give_up"
    shift_mm: tuple[int, int, int]             # clipped to ±_PER_ROUND_CAP_MM
    reasoning: str
    raw_response: str = ""
    parse_error: str | None = None


# -------------------------------------------------------------------
# Helpers — encoding / parsing / formatting
# -------------------------------------------------------------------

def _load_prompt() -> str:
    return _PROMPT_PATH.read_text()


def _encode_image(path: Path, max_dim: int = _IMAGE_MAX_DIM) -> str:
    img = Image.open(path)
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_history_table(history: list[RoundSummary]) -> str:
    if not history:
        return "  (no prior rounds — this is round 0)"
    header = "  Round | cum_shift mm     | peak_lift | closest | decision  | reasoning"
    sep    = "  ------+------------------+-----------+---------+-----------+------------------"
    rows = [header, sep]
    for r in history:
        sx, sy, sz = r.cumulative_shift_mm
        rsn = r.reasoning_summary[:35]
        rows.append(
            f"  {r.round_idx:>5} | ({sx:+3.0f},{sy:+3.0f},{sz:+3.0f})        "
            f"| {r.peak_lift_mm:>5.1f}mm  | {r.closest_distance_mm:>5.1f}mm | "
            f"{r.decision:<9} | {rsn}"
        )
    return "\n".join(rows)


def _build_user_content(
    images: list[Path],
    probe: GraspProbeResult,
    cumulative_shift_mm: tuple,
    scene_state: dict,
    history: list[RoundSummary],
) -> list[dict]:
    cs = cumulative_shift_mm
    p = scene_state["pickup_pos"]
    g = scene_state["gripper_pos"]
    diff = (g[0] - p[0], g[1] - p[1], g[2] - p[2])
    pickup_name = scene_state.get("pickup_name", "pickup")

    header = f"""# Current probe (round {len(history)})

- grasped: {probe.grasped}
- peak_lift: {probe.peak_lift_m * 1000:.1f} mm   (target ≥ 30mm)
- lift_window: {probe.lift_window_sec * 1000:.0f} ms   (target ≥ 200ms)
- closest_distance: {probe.closest_distance_m * 1000:.1f} mm
- dist_at_peak_lift: {probe.dist_at_peak_lift_m * 1000:.1f} mm
- failure_reason: {probe.failure_reason}

# Cumulative shift entering this probe

({cs[0]:+.0f}, {cs[1]:+.0f}, {cs[2]:+.0f}) mm   (per-axis cap = ±{_CUMULATIVE_CAP_MM}mm)

# Scene state @ engage_frame ({scene_state['frame']})

- {pickup_name}_pos_world : ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) m
- gripper_pos_world : ({g[0]:+.4f}, {g[1]:+.4f}, {g[2]:+.4f}) m
- gripper − {pickup_name}  : ({diff[0] * 1000:+5.1f}, {diff[1] * 1000:+5.1f}, {diff[2] * 1000:+5.1f}) mm

# History of prior rounds

{_build_history_table(history)}

# Current-round images ({len(images)} attached below)
"""
    content: list[dict] = [{"type": "text", "text": header}]
    for path in images:
        b64 = _encode_image(path)
        content.append({"type": "text", "text": f"\n[image] {path.name}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({
        "type": "text",
        "text": "\nNow output ONLY the JSON decision object. No prose, no fences.",
    })
    return content


def _extract_text(response) -> str:
    """Best-effort extraction of text from a langchain LLM response."""
    c = response.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return str(c)


def _clip(x: int, cap: int = _PER_ROUND_CAP_MM) -> int:
    return max(-cap, min(cap, x))


def _parse_response(raw: str) -> ShiftDecision:
    """Extract JSON from raw text, validate, clip shift.

    Tolerances built up the hard way from real LLM failure modes:
      - Leading prose / explanation before the JSON: handled by skipping
        to the first ``{`` and using ``raw_decode`` (consumes only the
        first complete object, ignores trailing data).
      - LLMs sometimes emit ``+10`` for positive integers, which is not
        valid JSON (spec disallows leading ``+`` on numbers). Stripped via
        regex in structural positions only (after ``[``, ``,``, ``:``,
        whitespace) so it doesn't mangle string content.
    """
    start = raw.find("{")
    if start == -1:
        return ShiftDecision(
            decision="give_up", shift_mm=(0, 0, 0),
            reasoning="(no JSON object found in response — give up)",
            raw_response=raw, parse_error="no JSON object",
        )

    payload = raw[start:]
    # Strip leading + from numbers in structural positions only.
    payload = re.sub(r"(?<=[\s,:\[])\+(?=\d)", "", payload)

    try:
        obj, _consumed = json.JSONDecoder().raw_decode(payload)
    except Exception as e:
        return ShiftDecision(
            decision="give_up", shift_mm=(0, 0, 0),
            reasoning=f"(JSON parse error: {e})",
            raw_response=raw, parse_error=str(e),
        )

    decision = str(obj.get("decision", "give_up")).lower()
    if decision not in ("success", "tweak", "give_up"):
        decision = "give_up"

    shift_raw = obj.get("shift_mm", [0, 0, 0])
    try:
        shift = tuple(int(round(float(x))) for x in list(shift_raw)[:3])
        while len(shift) < 3:
            shift = shift + (0,)
    except Exception:
        shift = (0, 0, 0)
    shift = tuple(_clip(s) for s in shift)

    if decision in ("success", "give_up"):
        shift = (0, 0, 0)

    reasoning = str(obj.get("reasoning", "")).strip()
    return ShiftDecision(
        decision=decision, shift_mm=shift,
        reasoning=reasoning, raw_response=raw, parse_error=None,
    )


# -------------------------------------------------------------------
# Public entry
# -------------------------------------------------------------------

def decide_shift(
    images: list[str | Path],
    probe_result: GraspProbeResult,
    cumulative_shift_mm: tuple,
    scene_state: dict,
    history: list[RoundSummary],
) -> ShiftDecision:
    """One-shot LLM decision: tweak / success / give_up.

    Args:
        images: paths to render images for the current round (≥ 1).
        probe_result: this round's probe outcome.
        cumulative_shift_mm: shift (dx, dy, dz) entering this probe (mm).
        scene_state: dict from ``get_scene_state_at_frame`` plus optional
            ``pickup_name`` for prompt labeling.
        history: prior rounds' summaries (empty for round 0).

    Returns:
        ShiftDecision with parsed decision + clipped shift.

    Raises:
        RuntimeError: if no LLM provider is configured.
    """
    image_paths = [Path(p) for p in images]
    if not image_paths:
        raise ValueError("at least one image required for decide_shift")

    llm = chat_model_for("grasp_optimization.subagents.grasp_loop_shift_decision")
    if llm is None:
        raise RuntimeError(
            "No credential configured for the YAML-selected LLM provider."
        )

    sys_msg = SystemMessage(content=[{
        "type": "text", "text": _load_prompt(),
        "cache_control": {"type": "ephemeral"},
    }])
    user_msg = HumanMessage(content=_build_user_content(
        image_paths, probe_result, cumulative_shift_mm, scene_state, history,
    ))

    response = llm.invoke([sys_msg, user_msg])
    raw = _extract_text(response)
    return _parse_response(raw)


__all__ = ["decide_shift", "ShiftDecision", "RoundSummary"]
