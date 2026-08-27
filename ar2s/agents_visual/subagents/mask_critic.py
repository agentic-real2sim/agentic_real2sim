"""mask_critic — per-candidate VLM mask judge, called by the ``segment`` controller.

  :func:`judge_candidate` — for one (object, prompt, keyframe, masks_dir), draw
    a magenta outline tracing the mask boundary on the keyframe and fan-out vote
    accept/reject. An empty mask is rejected outright without a VLM call.
  :func:`pick_best_or_drop` — finalization helper: show every saved candidate
    for an object that never passed, and let the VLMs pick the least-bad one or
    drop the object before SAM3D.

Model routing comes from the YAML config entry ``visual.subagents.mask_critic``.
"""
from __future__ import annotations

import base64
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ar2s.agents_visual._models import vlm_call
from ar2s.agents_visual.state import load_state
from ar2s.droid_sim._util import sanitize_name


# Minimum non-empty mask frames to keep an object in the pipeline.
# mesh_recover is a single-image SAM3D call (1 frame is enough);
# pose_tracking's FoundationPose runner only reads mask at init_frame (see
# third_party/FoundationPose/run_demo_foundationstereo_video.py:353 —
# ``mask = reader.get_mask(i) if (i==init_frame or args.reinit_with_mask)
# else None``); mesh_scale's median is defined for n=1 and scale_critic
# only WARNs at num_frames < 3 (no block). Exported for segment_controller's
# sparse-mask gate (cheap → expensive: coverage 0.2 →
# sparse <_MIN_FRAMES_FOR_PIPELINE → VLM judge).
_MIN_FRAMES_FOR_PIPELINE = 1


# =========================================================================
# VLM mask judge constants + schemas + helpers
# =========================================================================

_PROMPT_PATH = Path(__file__).with_name("mask_critic_prompt.md")
# Outline-only overlay (no fill blend). Earlier filled-blend variants:
#   - green (60, 220, 60): collided with the 'MAJOR ACCENT' green marker
#     (SAM3 26/26 masks but VLM saw "no green mask").
#   - magenta (255, 0, 255) at 0.45 alpha: on the YELLOW pen_to_mug marker
#     the blend hit (255, 140, 140) ≈ pinkish, so the VLM described the
#     marker as "pink" and decided "no magenta mask present" — same root
#     cause as green-on-green, just a different colour collision.
# Outline only fixes both: the object interior keeps its original colour
# (so the VLM can describe it correctly) and the high-contrast outline
# unambiguously shows where the mask boundary lies.
_OVERLAY_COLOR = (255, 0, 255)          # magenta (BGR == RGB for symmetrical pure colour)
_OUTLINE_THICKNESS = 3                  # px width of the mask contour line
_OVERLAY_MAX_DIM = 1024
_OVERLAY_JPEG_QUALITY = 85
_MAX_IMAGES = 4                         # gateway Mistral-Small image cap; cap here too
# Small masks (a cube in a wide tabletop frame) become illegible after the frame
# is downscaled to _OVERLAY_MAX_DIM, so the critic mistakes them for a larger
# neighbour. Zoom into the mask bbox with surrounding context instead.
_CROP_CONTEXT_FACTOR = 3.0              # margin around bbox, in multiples of bbox size
_CROP_MIN_MARGIN = 150                  # px; floor so tiny masks still get context
_CROP_TARGET_DIM = 768                  # upscale small crops so the object is legible

@dataclass
class CandidateVerdict:
    accept: bool
    reasoning: str


class _CandidateSchema(BaseModel):
    accept: bool = Field(
        description="True iff the magenta outline correctly and substantially traces the target object's boundary."
    )
    reasoning: str = Field(description="One sentence summary.")


class _BestChoice(BaseModel):
    best_index: int = Field(
        description="1-based index of the least-bad candidate, or 0 to drop the object entirely."
    )
    reasoning: str = Field(description="One sentence justifying the choice.")


def _left_frame(run_root: str, keyframe: int) -> Path | None:
    state = load_state(run_root)
    sve = state.get("stages", {}).get("svo_extract", {})
    left_dir = sve.get("outputs", {}).get("left_frames_dir") if sve.get("ok") else None
    if not left_dir:
        return None
    p = Path(left_dir) / f"frame_{keyframe:06d}.png"
    return p if p.is_file() else None


def _mask_nonempty(mask_path: Path) -> bool:
    return mask_path.is_file() and bool(np.array(Image.open(mask_path).convert("L")).any())


def _encode(img: Image.Image) -> str:
    w, h = img.size
    scale = min(1.0, _OVERLAY_MAX_DIM / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_OVERLAY_JPEG_QUALITY)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _plain_data_url(rgb_path: Path) -> str:
    img = Image.open(rgb_path).convert("RGB")
    return _encode(img)


def _crop_to_mask(img: Image.Image, mbin: np.ndarray) -> Image.Image:
    """Zoom into the mask bbox (+ context margin) so a small object is legible.

    Returns the image unchanged when the mask is empty (caller's empty-mask path
    already sends the plain full frame, but guard here for safety).
    """
    ys, xs = np.where(mbin)
    if len(xs) == 0:
        return img
    w, h = img.size
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    margin = max(_CROP_MIN_MARGIN, int(_CROP_CONTEXT_FACTOR * max(x1 - x0, y1 - y0)))
    crop = img.crop((max(0, x0 - margin), max(0, y0 - margin),
                     min(w, x1 + margin), min(h, y1 + margin)))
    cw, ch = crop.size
    scale = _CROP_TARGET_DIM / max(cw, ch)
    if scale > 1.0:
        crop = crop.resize((round(cw * scale), round(ch * scale)), Image.Resampling.LANCZOS)
    return crop


def _overlay_data_url(rgb_path: Path, mask_path: Path) -> str | None:
    """Render the mask boundary as a magenta outline on the original RGB.

    Outline-only (no fill blend) so the masked object's interior keeps its
    actual colour — see the _OVERLAY_COLOR comment above for the history of
    fill-blend's colour-collision failure modes.

    Boundary detection: erode the binary mask by ``_OUTLINE_THICKNESS`` px
    (PIL MinFilter) and take the set-difference with the original mask. That
    ring is the outline. PIL-only — keeps the orchestrator env's dependency
    surface tight (no cv2 required here).
    """
    if not rgb_path.is_file() or not mask_path.is_file():
        return None
    from PIL import ImageFilter
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    m_pil = Image.open(mask_path).convert("L")
    if m_pil.size != (rgb.shape[1], rgb.shape[0]):
        m_pil = m_pil.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
    mbin = np.array(m_pil) > 127

    # Erode mask by THICKNESS px (MinFilter needs an odd-side kernel). The
    # ring between mask and eroded-mask is the boundary we colour.
    kernel = 2 * _OUTLINE_THICKNESS + 1
    eroded = np.array(
        Image.fromarray(mbin.astype(np.uint8) * 255, mode="L").filter(
            ImageFilter.MinFilter(kernel)
        )
    ) > 127
    boundary = mbin & ~eroded

    out = rgb.copy()
    out[boundary] = _OVERLAY_COLOR
    return _encode(_crop_to_mask(Image.fromarray(out), mbin))


def _save_debug_jpeg(run_root: str, debug_tag: str, data_url: str | None) -> None:
    """Persist whatever was sent to the VLM at ``<run_root>/segmentation/_critic_inputs/<tag>.jpg``.

    Lets a human inspect EXACTLY the image the mask critic saw, without
    re-running the VLM call. Quiet best-effort — never raises, just skips on
    any I/O / decode error so we don't break the segment loop.
    """
    if not data_url or not debug_tag:
        return
    try:
        prefix, _, b64 = data_url.partition(",")
        if "jpeg" not in prefix or not b64:
            return
        out = Path(run_root) / "segmentation" / "_critic_inputs" / f"{debug_tag}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(b64))
    except Exception as e:                          # pragma: no cover
        print(f"[mask_critic] debug image save failed ({debug_tag}): {e}")


def _candidate_image(
    run_root: str, keyframe: int, masks_dir: str,
    *, debug_tag: str = "",
) -> tuple[str | None, bool]:
    """Return (data_url, has_mask) for a candidate at its keyframe.

    ``debug_tag`` (optional): when set, the generated JPEG is also written to
    ``<run_root>/segmentation/_critic_inputs/<tag>.jpg`` so a human can see
    EXACTLY what the VLM was shown.
    """
    rgb = _left_frame(run_root, keyframe)
    if rgb is None:
        return None, False
    mask_path = Path(masks_dir) / f"{keyframe:06d}.png" if masks_dir else None
    if mask_path is not None and _mask_nonempty(mask_path):
        url = _overlay_data_url(rgb, mask_path)
        if debug_tag:
            _save_debug_jpeg(run_root, debug_tag, url)
        return url, True
    url = _plain_data_url(rgb)
    if debug_tag:
        _save_debug_jpeg(run_root, f"{debug_tag}__NOMASK", url)
    return url, False


# =========================================================================
# VLM public judges
# =========================================================================

def judge_candidate(
    run_root: str, object_name: str, prompt: str, keyframe: int, masks_dir: str,
    description: str = "",
    agent_path: str = "visual.subagents.mask_critic",
) -> CandidateVerdict:
    """Judge one candidate segmentation: accept or reject.

    Used by segment_controller's per-candidate gate AFTER the cheap deterministic
    gates (coverage 0.2 + sparse) pass. Robot is auto-accepted upstream and
    never reaches here.

    ``description`` (optional): the object_discovery visual description (colour,
    material, shape, location) so the critic can disambiguate the target from
    similar neighbours rather than relying on the bare label.

    ``agent_path`` selects the critic; every caller currently uses the default.
    It stays parameterised because Kimi over-rejects small objects with
    hallucinated ``wrong_object`` reasoning, so a subset of candidates may need
    routing to a more permissive judge.
    """
    # Derive a debug_tag from the candidate identity so the saved image is
    # immediately greppable from the segmentation run dir (matches the
    # segment_controller out_subdir naming convention as closely as
    # judge_candidate's call site permits).
    debug_tag = f"judge__{sanitize_name(object_name)}__{sanitize_name(prompt)}__kf{keyframe}"
    image, has_mask = _candidate_image(
        run_root, keyframe, masks_dir, debug_tag=debug_tag,
    )
    if image is None:
        return CandidateVerdict(False, f"no rgb frame at index {keyframe}")
    # An empty mask is always a reject — no VLM call needed. The debug JPEG is
    # still saved above (via debug_tag) so a human can inspect the frame.
    if not has_mask:
        return CandidateVerdict(False, "empty mask")

    system_prompt = _PROMPT_PATH.read_text()
    desc_line = f" Visual description of the target: {description}" if description else ""
    user_text = (
        f"Target object: {object_name!r}.{desc_line} SAM3 was prompted with the text "
        f"{prompt!r} on frame {keyframe}. The image is a zoomed-in crop of that "
        f"frame centered on the predicted mask, with the mask boundary drawn as "
        f"a magenta outline (object interiors keep their original colours). "
        f"Surrounding context is included so you can tell the outlined object "
        f"apart from its neighbours. Decide whether the magenta outline "
        f"correctly traces {object_name!r}'s boundary."
    )

    print(f"[mask_critic] judging {object_name!r} (prompt={prompt!r}, kf={keyframe})")
    responses = vlm_call(
        agent_path,
        system=system_prompt,
        user_text=user_text,
        images=[image],
        output_schema=_CandidateSchema,
    )
    successes = [r for r in responses.values() if r is not None]

    if not successes:
        return CandidateVerdict(True, "all VLMs failed; defaulting to accept")

    rejects = [r for r in successes if not r.accept]
    if len(rejects) * 2 <= len(successes):                     # majority accept
        return CandidateVerdict(True, f"{len(successes) - len(rejects)}/{len(successes)} accept")

    reason = next((r.reasoning for r in rejects), "rejected")
    return CandidateVerdict(False, f"{len(rejects)}/{len(successes)} reject: {reason}")


def pick_best_or_drop(
    run_root: str, object_name: str, candidates: list[dict], description: str = "",
) -> int | None:
    """Choose the least-bad saved candidate (return its index) or drop (None).

    Used by segment_controller's finalization for objects that never passed any
    round. ``candidates`` are dicts with ``keyframe`` and ``masks_dir``. Up to
    ``_MAX_IMAGES`` candidates (preferring ones with a non-empty mask) are shown.
    ``description`` (optional): the object_discovery visual description, passed
    to the critic to disambiguate the target from similar neighbours.
    """
    shown: list[tuple[int, str]] = []          # (orig_index, data_url)
    ranked = sorted(range(len(candidates)),
                    key=lambda i: -(candidates[i].get("num_nonempty_masks") or 0))
    for i in ranked[:_MAX_IMAGES]:
        c = candidates[i]
        tag = (f"pick__{sanitize_name(object_name)}__"
               f"{sanitize_name(c.get('prompt') or 'unknown')}__kf{int(c['keyframe'])}")
        img, _ = _candidate_image(
            run_root, int(c["keyframe"]), c.get("masks_dir") or "",
            debug_tag=tag,
        )
        if img is not None:
            shown.append((i, img))
    if not shown:
        return None

    system_prompt = _PROMPT_PATH.read_text()
    labels = "\n".join(
        f"  candidate {n + 1}: prompt={candidates[orig].get('prompt')!r} frame={candidates[orig].get('keyframe')}"
        for n, (orig, _) in enumerate(shown)
    )
    desc_line = f" Visual description of the target: {description}" if description else ""
    user_text = (
        f"None of the segmentation attempts for {object_name!r} were accepted.{desc_line} "
        f"Below are the saved candidate masks (each predicted mask drawn as a "
        f"magenta outline on the keyframe; some may be empty), in order:\n"
        f"{labels}\n\nPick the 1-based index of the least-bad usable candidate "
        f"for 3D reconstruction, or 0 to drop {object_name!r} entirely if none "
        f"is usable."
    )

    responses = vlm_call(
        "visual.subagents.mask_critic",
        system=system_prompt,
        user_text=user_text,
        images=[url for _, url in shown],
        output_schema=_BestChoice,
    )
    votes: Counter[int] = Counter()
    for r in responses.values():
        if r is None:
            continue
        if r.best_index == 0:
            votes[0] += 1
        elif 1 <= r.best_index <= len(shown):
            votes[shown[r.best_index - 1][0] + 1] += 1   # store 1-based orig index
    if not votes:
        return None
    choice = votes.most_common(1)[0][0]
    return None if choice == 0 else choice - 1
