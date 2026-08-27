"""box_propose subagent — single-shot native-VLM object detection.

The segmentation stage sends a VLM-detected box as one of each object's SAM3
prompts. ``detect_boxes`` takes a keyframe and every object sharing it (name +
description) and returns one pixel-xyxy box per object in a SINGLE VLM call;

Frontier VLMs are natively trained for detection, but each family was trained
on a DIFFERENT box convention, and asking for the wrong one degrades the
detection rather than just relabelling it. We ask each model for the format it
knows (``_FORMATS``, selected per configured model spec) and normalize on the
way out:

  google     [y1,x1,y2,x2] normalized to a 1000x1000 grid
  anthropic  [x1,y1,x2,y2] in absolute pixels of the image it was sent
  openai     [x1,y1,x2,y2] normalized to a 999x999 grid  (also the fallback
             for unrecognised providers — most likely to match)

Every one of those is "coords relative to some reference frame", so conversion
is one code path: divide by the reference, multiply by the keyframe's real
size. For the absolute-pixel format the reference is the size of the image we
actually transported, NOT the keyframe on disk — ``image_to_data_url``
downsamples to ``_MAX_DIM`` on the long side, so a 1280x720 keyframe reaches
the model as 1024x576 and its pixel coords are in that frame.

NOT a @tool — called as a plain Python function from inside the segmentation
skill / segment_controller. ``object_name`` comes from
``state.discovered_objects``; ``description`` comes from
``state.discovered_object_descriptions[name]``.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from pydantic import BaseModel, Field, create_model

from ar2s.agents_visual._models import get_agent_config
from ar2s.agents_visual._vlm import _query_one
from ar2s.agent_configs import image_to_data_url


_AGENT_PATH = "visual.subagents.box_detect"
_MAX_DIM = 1024            # long-side cap applied to the transported keyframe
_MIN_SIDE_PX = 12


@dataclass(frozen=True)
class _BoxFormat:
    order: str             # "yxyx" | "xyxy"
    grid: int              # normalization grid max; 0 = absolute pixels
    desc: str              # what we tell the model box_2d means


_FORMATS: dict[str, _BoxFormat] = {
    "google": _BoxFormat(
        "yxyx", 1000, "[y1, x1, y2, x2], normalized to 0-1000 on both axes",
    ),
    "anthropic": _BoxFormat(
        "xyxy", 0, "[x1, y1, x2, y2] in absolute pixels of the image shown",
    ),
    "openai": _BoxFormat(
        "xyxy", 999, "[x1, y1, x2, y2], normalized to 0-999 on both axes",
    ),
}
_FALLBACK_FAMILY = "openai"

# Matched against the whole ``provider:model`` spec, because under OpenRouter
# the provider is always "openrouter" and the family lives in the model path
# (openrouter:google/gemma-4-31b-it, openrouter:anthropic/claude-haiku-4.5).
_FAMILY_TOKENS = (
    ("google",    ("google", "gemma", "gemini")),
    ("anthropic", ("anthropic", "claude")),
    ("openai",    ("openai", "gpt-", "o1-", "o3-")),
)


def _family(model_spec: str) -> str:
    """Map a ``provider:model`` spec to the box format family it was trained on."""
    s = model_spec.lower()
    for family, tokens in _FAMILY_TOKENS:
        if any(tok in s for tok in tokens):
            return family
    return _FALLBACK_FAMILY


def _sent_size(W: int, H: int) -> tuple[int, int]:
    """Size of the keyframe as ``image_to_data_url`` transports it."""
    scale = min(1.0, _MAX_DIM / max(W, H))
    return (int(W * scale), int(H * scale)) if scale < 1.0 else (W, H)


_SCHEMA_CACHE: dict[str, type[BaseModel]] = {}


def _schema_for(family: str) -> type[BaseModel]:
    """Build (and cache) the detection schema carrying ``family``'s box description."""
    if family not in _SCHEMA_CACHE:
        det = create_model(
            "_Detection",
            box_2d=(list[int], Field(description=_FORMATS[family].desc)),
            label=(str, ...),
        )
        _SCHEMA_CACHE[family] = create_model(
            "_Detections",
            detections=(
                list[det],
                Field(description="one entry per instance of the requested object"),
            ),
        )
    return _SCHEMA_CACHE[family]


def _to_pixels(box_2d, W: int, H: int, fmt: _BoxFormat, ref: tuple[int, int]) -> list[int] | None:
    """Convert one model-reported box to keyframe pixel ``[x1,y1,x2,y2]``.

    ``ref`` is the (width, height) the model's coordinates are relative to:
    the normalization grid, or the transported image size for absolute-pixel
    formats. Returns None if the entry isn't 4 numbers. Coordinates are clamped
    to the frame and forced to at least ``_MIN_SIDE_PX`` per side so SAM3
    always gets a usable box prompt.
    """
    if not (isinstance(box_2d, list) and len(box_2d) == 4):
        return None
    try:
        a, b, c, d = [float(v) for v in box_2d]
    except (TypeError, ValueError):
        return None
    y1, x1, y2, x2 = (a, b, c, d) if fmt.order == "yxyx" else (b, a, d, c)

    ref_w, ref_h = ref
    x1, x2 = sorted((x1 / ref_w * W, x2 / ref_w * W))
    y1, y2 = sorted((y1 / ref_h * H, y2 / ref_h * H))
    x1, x2 = max(0, int(x1)), min(W, int(x2))
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 - x1 < _MIN_SIDE_PX:
        x2 = min(W, x1 + _MIN_SIDE_PX)
    if y2 - y1 < _MIN_SIDE_PX:
        y2 = min(H, y1 + _MIN_SIDE_PX)
    return [x1, y1, x2, y2]


def _area(b: list[int]) -> int:
    return (b[2] - b[0]) * (b[3] - b[1])


def _match_tier(label: str, name: str) -> int:
    """How strongly a detection ``label`` names target ``name``: 3/2/1/0.

    3 exact, 2 substring either way, 1 shared head noun (last word), 0 none.
    A shared head noun alone is deliberately the WEAKEST tier so that, with two
    same-category objects ("black bowl" / "pink bowl"), an exact or modifier
    match always wins over the bare-noun collision.
    """
    label, name = label.strip().lower(), name.strip().lower()
    if not label:
        return 0
    if label == name:
        return 3
    if label in name or name in label:
        return 2
    if label.split()[-1] == name.split()[-1]:
        return 1
    return 0


def _assign_boxes(
    dets: list[tuple[str, list[int]]], names: list[str],
) -> dict[str, list[int]]:
    """Map labelled detections to target names; each target keeps its largest box.

    Every detection is assigned to the ONE target it matches best (highest tier;
    ties broken by target order), so a single box never satisfies two targets.
    Composite parts (a "black bowl lid" detection alongside "black bowl") land on
    the same target and the enclosing/largest box wins.
    """
    out: dict[str, list[int]] = {}
    for label, box in dets:
        best_name, best_tier = None, 0
        for name in names:
            tier = _match_tier(label, name)
            if tier > best_tier:
                best_tier, best_name = tier, name
        if best_name is None:
            continue
        if best_name not in out or _area(box) > _area(out[best_name]):
            out[best_name] = box
    return out


def detect_boxes(
    keyframe_path, targets: list[tuple[str, str]],
) -> dict[str, list[int]]:
    """Detect several objects in ONE keyframe with ONE VLM call per model.

    ``targets`` is ``[(object_name, description), ...]`` — every object sharing
    this keyframe. Returns ``{object_name: [x1,y1,x2,y2]}`` for those detected;
    missing objects are simply absent. Models / temperature / max_tokens come
    from ``visual.subagents.box_detect``; each configured model is tried in
    order with ITS OWN box format (a shared prompt can't be right for two
    families) and the first that yields any box wins.

    Detections are matched back to targets by label. Within a target, when the
    model emits several boxes (composite parts — lid, handle) we take the
    largest, since SAM3's box prompt does better with the enclosing box. With a
    single target, label matching is skipped — the largest box overall is used.
    """
    if not targets:
        return {}
    im = Image.open(keyframe_path).convert("RGB")
    W, H = im.size
    kf_url = image_to_data_url(keyframe_path, max_dim=_MAX_DIM)

    if len(targets) == 1:
        name, desc = targets[0]
        user_text = f"detect {name}" + (f" ({desc})" if desc else "")
    else:
        lines = [f"- {n}" + (f": {d}" if d else "") for n, d in targets]
        user_text = (
            "Detect each of these objects. Return one detection per object, and "
            "set each detection's label to the object's exact name as written:\n"
            + "\n".join(lines)
        )
    names = [n for n, _ in targets]

    cfg = get_agent_config(_AGENT_PATH)
    for model_spec in cfg["models"]:
        family = _family(model_spec)
        fmt = _FORMATS[family]
        ref = (fmt.grid, fmt.grid) if fmt.grid else _sent_size(W, H)
        system = (
            "You are an object detector. Detect the object(s) the user names and "
            f"return each bounding box as box_2d = {fmt.desc}."
        )
        print(f"[box_detect] detecting {names} in {W}x{H} keyframe "
              f"({model_spec}, {family} format, ref={ref[0]}x{ref[1]}) ...")
        try:
            result = _query_one(
                _AGENT_PATH, model_spec, system, user_text, [kf_url],
                _schema_for(family),
                max_tokens=int(cfg["max_tokens"]),
                temperature=cfg.get("temperature"),
                reasoning_effort=cfg.get("reasoning_effort"),
            )
        except Exception as exc:  # noqa: BLE001 — try the next configured model
            print(f"[box_detect] {model_spec} FAILED — {str(exc)[:120]}")
            continue

        dets = [(d, _to_pixels(d.box_2d, W, H, fmt, ref)) for d in result.detections]
        dets = [(d, b) for d, b in dets if b]
        if not dets:
            print(f"[box_detect] {model_spec} returned no usable box for {names}")
            continue

        if len(names) == 1:
            # Single target: labels are unreliable and there's nothing to
            # disambiguate against — take the largest box overall.
            out = {names[0]: max((b for _, b in dets), key=_area)}
        else:
            out = _assign_boxes([(d.label or "", b) for d, b in dets], names)
        if out:
            print(f"[box_detect] {len(dets)} detection(s) → boxes for {list(out)}"
                  + (f"; MISSED {[n for n in names if n not in out]}"
                     if len(out) < len(names) else ""))
            return out
        print(f"[box_detect] {model_spec}: no detection label matched {names}")

    print(f"[box_detect] no detection for {names}")
    return {}
