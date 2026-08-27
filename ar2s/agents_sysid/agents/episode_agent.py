"""Phystwin Episode Setup Agent.

This agent scans a phystwin raw folder, decides the PhysTwin config choice,
and submits a ``phystwin_manifest.yaml``. The old droid raw-folder Episode
Agent path is gone; ``kind`` is kept only as a compatibility assertion.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ar2s.agent_configs.models import chat_model_for
from ar2s.agents_sysid.skills.fs_inspect import (
    file_head,
    h5_info,
    list_dir,
    npy_info,
    npz_info,
)
from ar2s.agents_sysid.skills.manifest_common import ScanResult, _SubmissionSlot
from ar2s.agents_sysid.skills.manifest_submit_phystwin import make_submit_phystwin_tool
from ar2s.agents_sysid.skills.visual_inspect import view_image, view_video_keyframes


_RECURSION_LIMIT = 24            # ~12 LLM-tool round trips; tightened from 40


_PHYSTWIN_SYSTEM_PROMPT = """\
You are an Episode Setup Agent (phystwin kind).

# Goal

Given a raw folder of phystwin-format demonstration data, produce a small
`phystwin_manifest.yaml` describing the episode. Unlike droid, you do NOT
plan file moves — phystwin raw data already lives in the canonical layout
that `ar2s.phystwin_sysid` reads directly. Your job is mostly:

  1. Verify the raw folder has the expected files.
  2. Read metadata.json to fill in frame_num / fps / n_cams.
  3. Decide ONE physics parameter: ``config_choice`` ∈ {cloth, real} based
     on the object type (see "Decision: cloth vs real" below).
  4. Submit the manifest.

# Available Tools

Read-only inspection (use as needed):
  - list_dir(path, recursive=False)
  - file_head(path, n_bytes=512)
  - npy_info(path)
  - npz_info(path)
  - h5_info(path)

Visual inspection (multimodal — they show you images):
  - view_image(path)                       a single image
  - view_video_keyframes(path, n_frames)   N evenly-spaced keyframes
                                            of a video (with timestamp
                                            captions for temporal order).

Termination (call last):
  - submit_manifest(submission_json)
    Validates strictly. On failure returns errors so you can fix + retry.
    submission_json is ONE JSON string with EXACTLY one key:
      {"manifest_yaml": "<yaml string>"}

# Expected raw folder layout

A typical phystwin episode looks like this:

    <case_name>/
    ├── calibrate.pkl       pickle: list[ndarray(4,4)] — per-camera cam2world
    ├── metadata.json       JSON with: intrinsics, WH, fps, frame_num, serial_numbers
    ├── final_data.pkl      main data: object_points, object_colors, controller_points,
    │                        surface_points, interior_points, ...
    ├── color/              per-camera images: 0/0.png ... 0/<N-1>.png, 1/0.png, ...
    │                        Also 0.mp4, 1.mp4, ... — the same images as video.
    ├── final_data.mp4      demo overlay video (optional)
    ├── cotracker/  depth/  mask/  pcd/  shape/  split.json   (intermediate
    │                        products from the visual processing pipeline —
    │                        phystwin_sysid does NOT read these; just record
    │                        their presence if you want.)

# Manifest Schema (phystwin, v1)

```yaml
schema_version: 1
pipeline_kind: phystwin
case_name: <string>          # = raw folder name (MUST MATCH the folder you scanned)
raw_dir: <absolute path>     # MUST MATCH the raw_dir you were given
train_frame: <int>           # how many frames to use for sysid; usually frame_num - 1
n_cams: <int>                # number of cameras (= len(calibrate.pkl))
fps: <int>                   # frames per second (from metadata.json)
config_choice: cloth | real  # agent's choice (cloth = soft+self-coll, real = rope-like)
config_reason: <string>      # ONE-line rationale (why cloth or why real)
```

# Decision: cloth vs real

The only physics knob the Stage 3 sysid wrapper takes is which YAML config
to seed from. Two choices, with concrete differences:

  - `cloth.yaml`: iterations=100, **self_collision=true**.
    Use for: deformable objects that can fold or compress against
    themselves — fabric, stuffed plush, soft packaging.
    Object-type keywords that map here:
      cloth, sloth, zebra, dinosor, package
    (any object with soft+squishy body, or a sheet/fabric.)

  - `real.yaml`: iterations=200, no self_collision (faster, simpler).
    Use for: elongated objects with low self-folding — ropes, cables.
    Object-type keywords that map here:
      rope

Decision rule (REQUIRED — vision-based, do not skip):
  1. You MUST call `view_image(<raw_dir>/color/0/0.png)` to see the object
     in its initial state. The case_name alone is NOT enough — empirical
     evidence shows keyword-based mapping is unreliable for ropes (a rope
     held slack or pushed across the table self-folds, requiring
     self_collision=true even though "rope" usually maps to real).
  2. You SHOULD also call `view_video_keyframes(<raw_dir>/final_data.mp4,
     n_frames=4)` (or skip it ONLY if case_name strongly hints at the
     motion type — e.g., `single_lift_*` implies pulling taut). The motion
     matters as much as the object: e.g., the same rope pushed across the
     table needs cloth.yaml; the same rope lifted taut needs real.yaml.
  3. Choose based on BOTH what you see and what motion the demo implies:
       - cloth.yaml: object can self-fold / sag / loop / press onto
         itself during the demo motion (fabric, plush, slack rope,
         pushed rope, package).
       - real.yaml: object stays taut / elongated and does NOT
         meaningfully self-contact (rope under tension, rigid rod).
  4. Write a 1-line `config_reason` that cites the visual evidence:
       config_reason: "Saw plush sloth with limbs in first_frame; double-
         hand stretch motion can press limbs against body → cloth.yaml"

# Workflow

1. `list_dir(raw_dir)` to confirm the 4 required entries exist:
     calibrate.pkl, metadata.json, final_data.pkl, color/
2. `file_head(<raw_dir>/metadata.json)` to read fps, frame_num, WH, intrinsics
   (intrinsics is a (n_cams, 3, 3) nested list — count its outer dimension
   for n_cams).
3. Set train_frame = frame_num - 1 (default; safe choice).
4. **Look at color/0/0.png** with view_image. Use BOTH the object shape
   you see AND the motion implied by case_name to pick cloth vs real
   per the rules above.
5. Compose manifest_yaml. Call submit_manifest with one JSON arg.

# Token efficiency

- Do not inspect cotracker/, depth/, mask/, pcd/, shape/ contents. They are
  intermediate visual-processing products and phystwin_sysid never reads
  them. A single `list_dir(raw_dir, recursive=False)` is enough.
- ONE `view_image` call per case is mandatory. Calling
  `view_video_keyframes` adds ~$0.03 in cost; do it only when the static
  first-frame leaves real ambiguity about the motion.

# Important

- case_name MUST match the raw folder name exactly.
- raw_dir MUST match the absolute path you were given.
- config_choice MUST be exactly the string "cloth" or "real" (no quotes,
  no .yaml suffix, no path).
- If a required field is uncertain, write a config_reason that explains why.
"""


def _system_message_with_caching() -> SystemMessage:
    """Build the phystwin system message with prompt caching enabled."""
    return SystemMessage(content=[
        {
            "type": "text",
            "text": _PHYSTWIN_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ])


def run_episode_agent(
    raw_dir: str | Path,
    scene_name: str | None = None,
    recursion_limit: int = _RECURSION_LIMIT,
    *,
    kind: str = "phystwin",
) -> ScanResult:
    """Run the phystwin Episode Agent on a raw folder.

    Args:
        raw_dir: directory containing the raw demo data.
        scene_name: defaults to the folder name. For phystwin this is also
            the ``case_name`` the agent will use.
        recursion_limit: cap on LLM ↔ tool round-trips (default 24)
        kind: must be ``"phystwin"``. Kept only for compatibility.

    Returns:
        ScanResult with the validated manifest. ``file_moves`` is always an
        empty list for phystwin.

    Raises:
        RuntimeError: if no LLM API key is set, the agent terminates
            without success, or hits the submission attempt cap.
        AssertionError: if ``kind`` is anything other than ``"phystwin"``.
    """
    assert kind == "phystwin", "run_episode_agent only supports kind='phystwin'"

    raw_dir = Path(raw_dir).expanduser().resolve()
    assert raw_dir.exists() and raw_dir.is_dir(), f"raw_dir {raw_dir} not a directory"
    if scene_name is None:
        scene_name = raw_dir.name

    slot = _SubmissionSlot()
    submit_tool = make_submit_phystwin_tool(slot, raw_dir)

    llm = chat_model_for("sysid.phystwin.main.episode_agent")
    if llm is None:
        raise RuntimeError(
            "No LLM provider/API key configured. "
            "Set the credential env var for the provider selected in the agent YAML."
        )

    agent = create_react_agent(
        model=llm,
        tools=[
            list_dir,
            file_head,
            npy_info,
            npz_info,
            h5_info,
            view_image,
            view_video_keyframes,
            submit_tool,
        ],
        prompt=_system_message_with_caching(),
    )

    initial_msg = (
        f"Raw folder to scan: {raw_dir}\n"
        f"Suggested case_name: {scene_name}\n\n"
        "Start with list_dir(recursive=False) to confirm the 4 required entries, "
        "then file_head on metadata.json. You MUST call view_image on "
        "color/0/0.png before deciding config_choice. Base the decision on "
        "what you actually see, not just the case_name. Then submit_manifest."
    )

    try:
        agent.invoke(
            {"messages": [HumanMessage(content=initial_msg)]},
            config={"recursion_limit": recursion_limit},
        )
    except Exception as e:
        if slot.submitted:
            return slot.submitted
        raise RuntimeError(
            f"Episode Agent crashed before submitting: {type(e).__name__}: {e}. "
            f"Failed attempts: {slot.failed_attempts}. "
            f"Last error: {slot.last_error or '(none)'}"
        ) from e

    if slot.aborted:
        raise RuntimeError(
            f"Episode Agent aborted: max submission attempts reached "
            f"({slot.failed_attempts}). Last error: {slot.last_error or '(none)'}"
        )

    if slot.submitted is None:
        raise RuntimeError(
            f"Episode Agent terminated without calling submit_manifest. "
            f"Failed submission attempts: {slot.failed_attempts}. "
            f"Last error: {slot.last_error or '(none)'}"
        )

    return slot.submitted


__all__ = ["run_episode_agent"]
