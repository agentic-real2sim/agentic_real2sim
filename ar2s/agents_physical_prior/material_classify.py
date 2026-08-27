"""material_classify — VLM material classifier with density lookup.

Per-object: send the VLM either a tight bbox crop of the object (when its
mask was accepted by ``mask_critic``) or the full keyframe with a
"find the X in this scene" prompt (when the mask was rejected / never
produced — a degraded mask's bbox would mislead the classifier). Parse
a 7-category structured result, average two candidate densities on low
confidence, return density (kg/m³).

This is the **core VLM logic** of the physical-prior agent. It's a plain
function (no LangChain `@tool` decorator). The ReAct orchestration lives in
``orchestrator.py``; the deterministic tool helpers that call this function
live in ``apply.py``.

The physical-prior ReAct stage is responsible for resolving
the right ``prompt_frame`` + ``frame_idx`` pair using the same per-object
keyframe / camera-routing that ``mesh_recover`` and ``mesh_scale`` use
(``state.object_keyframes`` + ``state.mask_camera_id_by_object`` against
the matching ``svo_extract[_secondary].outputs.left_frames_dir``), and
for deciding the ``use_full_frame`` flag based on
``state.segmentation_verdict.accepted``.

Failure modes (all return ``FALLBACK``, never crash):
  * Keyframe missing from disk (pipeline broken upstream)
  * VLM returns no parsable result

Public surface:
  ``classify_object(prompt_frame, masks_dir, obj_name, *,
                    crop_root, frame_idx, use_full_frame=False) -> dict``
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ar2s.agent_configs import image_to_data_url, vlm_call_first_success


_PROMPT_PATH = Path(__file__).with_name("material_classify_prompt.md")
_AGENT_PATH = "physical_prior.subagents.material_classify"

MaterialEnum = Literal[
    "ceramic", "glass", "plastic", "metal", "wood", "paper", "soft",
]

# Single representative density per category (kg/m³).
DENSITY_KG_PER_M3: dict[str, int] = {
    "ceramic": 2300,
    "glass":   2500,
    "plastic": 1200,
    "metal":   7800,
    "wood":    700,
    "paper":   700,
    "soft":    100,
}

# Fallback when a VLM call fails or no mask is available for the prompt frame.
# `plastic` is the safest middle-of-the-road density for indoor objects.
FALLBACK: dict = {
    "material":          "plastic",
    "alternative":       None,
    "confidence":        "low",
    "reasoning":         "fallback — could not query VLM or no mask available",
    "density_kg_per_m3": float(DENSITY_KG_PER_M3["plastic"]),
}

# Mask-bbox padding (px) before cropping the prompt frame.
_BBOX_PAD = 20


class _MaterialResult(BaseModel):
    material: MaterialEnum = Field(description="Most likely material category.")
    alternative: MaterialEnum | None = Field(
        default=None,
        description="Second most likely if uncertain; null if confident.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Self-assessed confidence.",
    )
    reasoning: str = Field(description="One sentence justifying the call.")


def _density_for(r: _MaterialResult) -> float:
    """Pick density: average two when low + alt given, otherwise the table."""
    d1 = DENSITY_KG_PER_M3[r.material]
    if r.confidence == "low" and r.alternative:
        d2 = DENSITY_KG_PER_M3[r.alternative]
        return (d1 + d2) / 2.0
    return float(d1)


def _crop_object(prompt_frame: Path, masks_dir: Path,
                 out_path: Path, *, frame_idx: int) -> Path | None:
    """Crop the object's bbox from ``prompt_frame`` using its per-object mask.

    ``frame_idx`` selects which mask file to read (``{idx:06d}.png``) and
    ``prompt_frame`` must already point at the matching RGB frame in the
    correct camera's ``frames/left/`` directory — the caller (apply.py) is
    responsible for pairing them via ``state.object_keyframes`` and
    ``state.mask_camera_id_by_object``.

    Returns the crop path, or None if the mask is empty / files missing.
    """
    mask_path = masks_dir / f"{frame_idx:06d}.png"
    if not mask_path.exists() or not prompt_frame.exists():
        return None
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    img = Image.open(prompt_frame).convert("RGB")
    W, H = img.size
    y0 = max(0, int(ys.min()) - _BBOX_PAD)
    x0 = max(0, int(xs.min()) - _BBOX_PAD)
    y1 = min(H - 1, int(ys.max()) + _BBOX_PAD)
    x1 = min(W - 1, int(xs.max()) + _BBOX_PAD)
    crop = img.crop((x0, y0, x1, y1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def _classify_one(prompt: str, obj_name: str, image_path: Path,
                  *, full_frame: bool = False) -> dict:
    """Run the VLM on one image and return the persisted dict.

    ``full_frame=True`` switches the user prompt from "what is this object
    made of?" (which assumes a tight crop) to a "find the X in this scene"
    variant so the VLM can locate the object in a wider keyframe.
    """
    if full_frame:
        user_text = (
            f"Object name: {obj_name}\n\n"
            f"Find the {obj_name} in this scene and tell us what it is made of."
        )
    else:
        user_text = f"Object name: {obj_name}\n\nWhat is this object made of?"
    r = vlm_call_first_success(
        _AGENT_PATH,
        system=prompt,
        user_text=user_text,
        images=[image_to_data_url(image_path)],
        output_schema=_MaterialResult,
    )
    if r is None:
        return dict(FALLBACK)
    return {
        "material":          r.material,
        "alternative":       r.alternative,
        "confidence":        r.confidence,
        "reasoning":         r.reasoning,
        "density_kg_per_m3": _density_for(r),
    }


def classify_object(
    prompt_frame: Path,
    masks_dir: Path | None,
    obj_name: str,
    *,
    crop_root: Path,
    frame_idx: int,
    use_full_frame: bool = False,
) -> dict:
    """Classify a single object's material from its per-object keyframe.

    Two modes:

    * **bbox-crop mode** (``use_full_frame=False``, the default): use the
      object's segmentation mask at ``frame_idx`` to compute a padded bbox
      and send that RGB crop to the VLM. Only honest when the mask was
      accepted by ``mask_critic`` — a degraded mask would produce a
      misleading bbox and lead the VLM to classify the wrong region.

    * **full-frame mode** (``use_full_frame=True``): skip the mask
      entirely and send the whole keyframe with a "find the X in this
      scene" prompt. Use for objects whose mask was rejected (or never
      produced); the VLM can still locate the object in the wider scene.

    Args:
        prompt_frame: RGB frame on the camera that owns this object's
            mask routing (caller resolves via
            ``state.mask_camera_id_by_object`` +
            ``svo_extract[_secondary].outputs.left_frames_dir``).
        masks_dir: per-object masks dir containing ``{frame_idx:06d}.png``.
            May be ``None`` — implies full-frame mode regardless of the
            ``use_full_frame`` flag.
        obj_name: canonical scene-object name (VLM context + crop filename).
        crop_root: dir to save the bbox crop into (ignored in full-frame mode).
        frame_idx: keyframe index from ``state.object_keyframes[obj_name]``
            (or the global ``prompt_frame_index`` fallback).
        use_full_frame: see above; caller decides per-object based on
            whether the object is in ``segmentation_verdict.accepted``.

    Returns:
        ``{material, alternative, confidence, reasoning, density_kg_per_m3}``.
        Always returns something — falls back to plastic@low_conf when the
        keyframe itself is missing from disk or the VLM call fails.
    """
    prompt = _PROMPT_PATH.read_text()
    if use_full_frame or masks_dir is None:
        if not prompt_frame.exists():
            chosen = dict(FALLBACK)
            chosen["reasoning"] = (
                f"fallback — full-frame requested but no keyframe @ {frame_idx:06d}"
            )
            print(f"[material_classify] {obj_name}: no keyframe, using fallback")
            return chosen
        print(f"[material_classify] {obj_name}: classifying full-frame @ {frame_idx} ...")
        return _classify_one(prompt, obj_name, prompt_frame, full_frame=True)

    crop_path = _crop_object(
        prompt_frame, masks_dir, crop_root / f"{obj_name}.png",
        frame_idx=frame_idx,
    )
    if crop_path is None:
        # Accepted object but mask file is empty or missing. Don't punt to
        # default density — try full-frame VLM so a real per-object call
        # still happens.
        if not prompt_frame.exists():
            chosen = dict(FALLBACK)
            chosen["reasoning"] = f"fallback — empty mask + no keyframe @ {frame_idx:06d}"
            print(f"[material_classify] {obj_name}: empty mask + no keyframe, fallback")
            return chosen
        print(f"[material_classify] {obj_name}: empty mask @ {frame_idx:06d}, "
              f"retrying full-frame ...")
        return _classify_one(prompt, obj_name, prompt_frame, full_frame=True)
    print(f"[material_classify] {obj_name}: classifying crop @ frame {frame_idx} ...")
    return _classify_one(prompt, obj_name, crop_path, full_frame=False)
