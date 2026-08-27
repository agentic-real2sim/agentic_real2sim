"""camera_select subagent — VLM picks the best external camera for the demo.

Used by the raw_data -> raw_episode prep stage when the source dataset
records the demo from >= 2 external cameras (e.g. DROID always records
ext1 + ext2). The chosen view is the single stereo .mp4 fed to the rest
of the visual pipeline (stereo_depth + segmentation + pose_tracking),
so picking poorly here permanently hurts geometric accuracy.

Unlike the other subagents under ``subagents/``, this one is NOT a
``@tool(run_root: str)``. It runs BEFORE the run_root exists (camera
choice happens during raw_data conversion), so its inputs are passed
directly and it returns a dataclass instead of writing to state.json.

Voting + tie-break logic mirrors ground_ref. The first YAML-configured
model for this agent is the tie-break preference. If every VLM fails, falls
back to the
alphabetically-first candidate serial (deterministic, so reruns reproduce).

Frame extraction reads the stereo SBS .mp4 with imageio (visual env)
and keeps only the left half — matches what svo_extract feeds the rest
of the pipeline.
"""
from __future__ import annotations

import base64
import io
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ar2s.agents_visual._models import get_agent_config, vlm_call


_PROMPT_PATH = Path(__file__).with_name("camera_select_prompt.md")
_N_FRAMES_PER_CAMERA = 4
_AGENT_PATH = "visual.subagents.camera_select"
_MAX_IMAGE_DIM = 1024                            # downsample longer side to ≤1024


@dataclass
class CameraCandidate:
    """One external camera to consider."""
    serial: str
    mp4_path: Path
    role: str = ""        # optional label, e.g. "ext1" / "ext2"
    sdk_intrinsics_path: Path | None = None


@dataclass
class CameraSelectResult:
    chosen_serial: str
    rationale: str
    votes: dict[str, int]
    per_model_pick: dict[str, str]
    fallback_reason: str = ""
    per_camera_frame_indices: dict[str, list[int]] = field(default_factory=dict)
    # Dual-view: the runner-up serial (primary fallback target). Empty if
    # only one candidate was available. Populated as: in a 2-way vote the
    # other candidate; in a >2-way vote the candidate with the second-most
    # votes (ties broken alphabetically). Downstream PRs use this to keep
    # both stereo mp4s + both extrinsics so segment can lazy-fallback
    # to the secondary view when primary's mask attempts fail.
    secondary_serial: str = ""


class _CameraChoice(BaseModel):
    chosen_serial: str = Field(
        description=(
            "Exact serial number of the chosen external camera — must match "
            "one of the candidate serials shown to you, verbatim."
        )
    )
    reason: str = Field(description="One short sentence justifying the choice.")


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _evenly_spaced_indices(total: int, n: int) -> list[int]:
    """Pick `n` evenly-spaced frame indices in [0, total)."""
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]


def _video_frame_count(cap: cv2.VideoCapture, mp4_path: Path) -> int:
    """Frame count from container metadata (no decode). Falls back to a
    grab-only pass only if the container doesn't report a usable count."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n > 0:
        return n
    n = 0
    while cap.grab():          # cheap-ish single pass, no color convert
        n += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if n <= 0:
        raise OSError(f"{mp4_path}: could not determine frame count")
    return n


def _sample_left_frames(mp4_path: Path, n: int) -> tuple[list[np.ndarray], list[int]]:
    """Read `n` evenly-spaced frames from `mp4_path`, keep left half only (stereo SBS).

    Returns (frames_rgb, frame_indices). Raises if the file can't be read.
    """
    cap = cv2.VideoCapture(str(mp4_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise OSError(f"{mp4_path}: cv2/ffmpeg could not open the file")
    try:
        total = _video_frame_count(cap, mp4_path)
        idxs = _evenly_spaced_indices(total, n)
        want = set(idxs)
        max_idx = idxs[-1]

        by_idx: dict[int, np.ndarray] = {}
        pos = 0
        while pos <= max_idx:
            if not cap.grab():                 # advance decoder; no array yet
                break
            if pos in want:
                ok, bgr = cap.retrieve()       # decode current frame to array
                if not ok or bgr is None:
                    raise RuntimeError(f"{mp4_path}: failed to retrieve frame {pos}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise RuntimeError(
                        f"{mp4_path} frame {pos} has unexpected shape {rgb.shape}; "
                        f"expected (H, W, 3) RGB"
                    )
                w = rgb.shape[1] // 2          # stereo SBS: keep left half
                by_idx[pos] = rgb[:, :w].copy()
            pos += 1
    finally:
        cap.release()

    missing = want - by_idx.keys()
    if missing:
        raise RuntimeError(
            f"{mp4_path}: reached EOF before reading frame(s) {sorted(missing)} "
            f"(reported total={total})"
        )
    return [by_idx[k] for k in idxs], idxs

def _frame_to_data_url(frame: np.ndarray, *, max_dim: int = _MAX_IMAGE_DIM) -> str:
    """Encode an RGB ndarray frame as a base64 JPEG data URL.

    Mirrors _vlm.image_to_data_url but operates on an in-memory array
    (no need to round-trip via disk for these short-lived preview frames).
    """
    img = Image.fromarray(frame)
    longer = max(img.size)
    if longer > max_dim:
        scale = max_dim / longer
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def camera_select_vlm(
    candidates: list[CameraCandidate],
    *,
    task_description: str = "",
    n_frames_per_camera: int = _N_FRAMES_PER_CAMERA,
) -> CameraSelectResult:
    """VLM picks the best external camera. Returns dataclass with vote details.

    Args:
        candidates: external cameras to consider; need >=1.
        task_description: optional one-liner about what the robot is doing
            (improves disambiguation when the demo involves a small object
            visible from only one angle).
        n_frames_per_camera: number of evenly-spaced preview frames sampled
            per camera. Default 4. Each frame is the left half of stereo SBS.

    Behaviour:
        - 0 candidates -> raise (caller's bug; should be caught upstream).
        - 1 candidate -> returned directly, no VLM call.
        - >=2 candidates -> YAML-configured multi-model vote; ties broken by
          the Claude pick when available; if all VLMs fail, fall back to the
          alphabetically-first candidate serial (recorded in fallback_reason).

    All print output flows through the same `[vlm] <model>: OK/FAILED` lines
    the other subagents use, so the build_raw_episode CLI logs stay readable.
    """
    if not candidates:
        raise ValueError("camera_select_vlm called with no candidates")

    if len(candidates) == 1:
        only = candidates[0]
        return CameraSelectResult(
            chosen_serial=only.serial,
            rationale=f"only candidate -> {only.serial}",
            votes={only.serial: 0},
            per_model_pick={},
            secondary_serial="",
        )

    print(f"[camera_select] sampling {n_frames_per_camera} frames per camera "
          f"from {len(candidates)} candidates ...")

    all_images: list[str] = []
    frame_idx_by_serial: dict[str, list[int]] = {}
    per_cam_labels: list[str] = []   # one label per frame, parallel to all_images

    for cand in candidates:
        if not cand.mp4_path.exists():
            raise FileNotFoundError(f"camera_select: {cand.mp4_path} not found")
        frames, idxs = _sample_left_frames(cand.mp4_path, n_frames_per_camera)
        frame_idx_by_serial[cand.serial] = idxs
        role_tag = f" ({cand.role})" if cand.role else ""
        for f, k in zip(frames, idxs):
            all_images.append(_frame_to_data_url(f))
            per_cam_labels.append(f"serial {cand.serial}{role_tag}, frame index {k}")
        print(f"  [camera_select] {cand.serial}{role_tag}: "
              f"sampled frames {idxs}")

    serial_list = ", ".join(c.serial for c in candidates)
    role_lines = "\n".join(
        f"  - serial {c.serial}" + (f" (role: {c.role})" if c.role else "")
        for c in candidates
    )
    frame_lines = "\n".join(
        f"  frame {i+1}: {lbl}" for i, lbl in enumerate(per_cam_labels)
    )

    user_text = (
        f"Task description: {task_description or '(none)'}\n\n"
        f"Candidate external cameras (choose exactly one serial):\n{role_lines}\n\n"
        f"You are shown {len(all_images)} frames total, grouped by camera in "
        f"the order they were sampled (first -> last for each camera):\n"
        f"{frame_lines}\n\n"
        f"Per the criteria in the system prompt, which serial is the best "
        f"choice? Respond with one of: {serial_list}."
    )

    system_prompt = _PROMPT_PATH.read_text()

    responses = vlm_call(
        _AGENT_PATH,
        system=system_prompt,
        user_text=user_text,
        images=all_images,
        output_schema=_CameraChoice,
    )

    valid_serials = {c.serial for c in candidates}
    votes: Counter[str] = Counter()
    per_model_pick: dict[str, str] = {}
    for model_id, r in responses.items():
        if r is None:
            continue
        chosen = r.chosen_serial.strip()
        if chosen in valid_serials:
            votes[chosen] += 1
            per_model_pick[model_id] = chosen
        else:
            print(f"[camera_select] {model_id.split('/')[-1]}: "
                  f"chose unknown serial {chosen!r}, dropping")

    if not votes:
        fallback_sorted = sorted(c.serial for c in candidates)
        fallback = fallback_sorted[0]
        # Runner-up is the next alphabetical candidate (or "" if only one).
        secondary = fallback_sorted[1] if len(fallback_sorted) >= 2 else ""
        print(f"[camera_select] all VLMs failed; falling back to {fallback}")
        return CameraSelectResult(
            chosen_serial=fallback,
            rationale=f"fallback: all VLMs failed -> alphabetical first {fallback!r}",
            votes={},
            per_model_pick={},
            fallback_reason="all VLMs failed or chose unknowns",
            per_camera_frame_indices=frame_idx_by_serial,
            secondary_serial=secondary,
        )

    top_count = votes.most_common(1)[0][1]
    top_picks = [s for s, n in votes.items() if n == top_count]
    preferred_model = get_agent_config(_AGENT_PATH)["models"][0]
    if len(top_picks) == 1:
        winner = top_picks[0]
        rationale = f"majority among {sum(votes.values())} VLM votes"
    elif preferred_model in per_model_pick and per_model_pick[preferred_model] in top_picks:
        winner = per_model_pick[preferred_model]
        rationale = (
            f"tied {top_picks}; broken by preferred model "
            f"{preferred_model.split('/')[-1]}"
        )
    else:
        winner = sorted(top_picks)[0]
        rationale = f"tied {top_picks}; broken alphabetically"

    print(f"[camera_select] vote tally: {dict(votes)}")
    print(f"[camera_select] chosen serial: {winner!r}  ({rationale})")

    # Runner-up = best non-winner: most votes among other candidates, ties
    # broken alphabetically. Falls back to the alphabetically-first other
    # candidate when no other candidate received any vote.
    other_serials = [c.serial for c in candidates if c.serial != winner]
    if other_serials:
        def _other_score(s: str) -> tuple[int, str]:
            # Higher vote count first; for ties, alphabetical ascending.
            return (-votes.get(s, 0), s)
        secondary = sorted(other_serials, key=_other_score)[0]
        print(f"[camera_select] secondary (runner-up): {secondary!r}")
    else:
        secondary = ""

    return CameraSelectResult(
        chosen_serial=winner,
        rationale=rationale,
        votes=dict(votes),
        per_model_pick=per_model_pick,
        per_camera_frame_indices=frame_idx_by_serial,
        secondary_serial=secondary,
    )
