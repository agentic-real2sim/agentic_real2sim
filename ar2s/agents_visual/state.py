"""Shared state for agents_visual — disk-backed.

The visual pipeline persists state as JSON on disk after every meaningful step
so the orchestrator can:

  * resume after a crash (re-read state.json and skip already-done stages)
  * keep the LLM context lean (skill outputs go to disk; only small JSON
    summaries enter the agent's working memory)
  * postmortem-debug a failed run (open state.json + logs/)

The shape keeps the legacy VisualState field names where useful but stores
everything as JSON-friendly primitives. Toolkit Report dataclasses (heavy / not
JSON-friendly) live only in the subprocess that produced them; we keep
just the salient paths + small stats here.

Run layout::

    outputs/run_pipeline/<run_id>/
    ├── state.json           ← this module reads/writes it
    ├── sysid_inputs/manifest.yaml
    ├── logs/<stage>.{cmd,stdout,stderr,returncode}
    ├── seq/...              ← skill artifacts
    ├── segmentation/...
    ├── meshes/<object>/...
    └── poses/<object>/...
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Light dataclasses re-used from legacy visual state (same shape, JSON-friendly)
# ---------------------------------------------------------------------------

@dataclass
class StepLog:
    """One row in the agent's audit trail."""
    agent: str
    note: str = ""


@dataclass
class Verdict:
    """Accept / reject decision from a critic subagent."""
    accept: bool = True
    reason: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class ResolvedObject:
    """Per-object fields needed by outputs.render_input_bundle_module.

    ``mass`` is intentionally optional: visual emits geometry / pose / scale
    but does not invent a mass. The real per-object value is filled in by
    ``ar2s.agents_physical_prior`` (material × density × scaled volume),
    OR provided up front via ``entry.object_mass_hints[name]`` to skip the
    physical-prior VLM call. If neither path fires, ``sysid.episode``'s
    validator will reject the manifest with an error pointing at
    physical_prior — keeps a forgotten stage from silently being skipped.
    """
    name: str
    visual_mesh_path: str
    scale_path: str
    pose_source_path: str
    mass: float | None = None
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)


# ---------------------------------------------------------------------------
# VisualState — a regular Python dict in spirit, but with a disk-load/save
# protocol. We keep it as a plain dict for ergonomic JSON serialisation;
# the keys document themselves below.
# ---------------------------------------------------------------------------

# Expected top-level keys (documenting the schema; not enforced as a class
# so the orchestrator can stamp in extras without code changes).
VISUAL_STATE_KEYS = {
    # ---- entry / paths (seeded by data_ingest) ----
    "entry",                       # VisualInput as dict
    "run_root",                    # absolute path to outputs/run_pipeline/<run_id>
    "episode_id",
    "scene_name",
    "stereo_stream_path",
    "stereo_intrinsics_path",
    "secondary_stereo_stream_path",
    "secondary_stereo_intrinsics_path",
    # Deprecated compatibility alias for incoming PR-3 code: rectified MP4
    # alias for secondary_stereo_stream_path, not an SVO file.
    "secondary_svo_path",
    "primary_camera_id",           # ZED serial matching stereo_stream_path.
    "secondary_camera_id",         # ZED serial matching secondary_stereo_stream_path.
    "robot_traj_path",
    "max_frames",
    "heavy_max_frames",            # cap FoundationStereo depth + FoundationPose tracking only
    "pose_shorter_side",           # FoundationPose input downscale (min(H,W)); None = native
    "frame_step",                  # subsample stride seeded by data_ingest
    "stereo_valid_iters",          # FoundationStereo GRU refinement iterations
    "get_pointcloud",              # stereo_depth: save debug .ply clouds (default off)

    # ---- per-stage skill summaries (small JSON) ----
    "stages",                      # dict: stage_name → {ok, outputs, stats, ...}

    # ---- VLM / subagent-derived decisions ----
    "discovered_objects",
    "pickup_objects",              # pickup_objects subagent: list[str] of the
                                    # objects the gripper grasps/lifts over the
                                    # demo.
    "ground_reference_object",
    "discovered_object_descriptions",  # object_discovery:
                                        # {object_name: description}. Consumed by
                                        # keyframe_select + box_detect.
    "object_colors",                  # optional {object_name: color}; used to
                                        # disambiguate same-name segmentation prompts.
    "support_tree",                # support_tree: {"relations": [{object, supported_by},
                                    # ...], "degraded": bool, "degraded_reason": str|None}.
                                    # Single-parent tree rooted at sentinel "GROUND", built
                                    # from first frame + discovered_objects before segment.

    # ---- hints ----
    "object_mass_hints",
    "object_friction_hints",

    # ---- critic-related ----
    "prompt_frame_index",              # global SAM3 prompt-frame FALLBACK; used by
                                       # segmentation + mesh_recover when an object
                                       # has no entry in prompt_frame_index_by_object.
                                       # Set by keyframe_select to pickup's frame.
    "tried_keyframes",
    "prompt_frame_index_by_object",   # LEGACY per-object keyframe map. No longer
                                       # written (superseded by object_keyframes);
                                       # kept only as a read fallback for older runs.
    "object_keyframes",               # segment_controller final per-object keyframes;
                                       # source of truth. Downstream mesh stages prefer
                                       # this and fall back to prompt_frame_index_by_object.
    "tried_keyframes_by_object",       # per-object retry record (mask_critic appends
                                       # the rejected frame here, then keyframe_select
                                       # re-picks for that object only).
    "mask_critic_verdict",
    "segmentation_result",
    "segmentation_verdict",
    "scale_critic_verdict",
    "tracking_critic_verdict",

    # ---- resolved bundle ----
    "resolved_objects",            # list[ResolvedObject] as list[dict]
    "robot_mask_path",
    "first_frame_rgb_path",
    "cameras_intrinsics_path",
    "cameras_extrinsics_path",
    "camera_ids",
    "object_camera_id",
    # Dual-view: per-object camera that produced the accepted mask. Empty or
    # absent for single-camera episodes.
    "mask_camera_id_by_object",
    "episode_emit_dir",

    # ---- bookkeeping ----
    "history",                     # list[StepLog] as list[dict]
}


VisualState = dict[str, Any]


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------

def state_path(run_root: str | Path) -> Path:
    return Path(run_root).expanduser().resolve() / "state.json"


def _to_json_friendly(obj: Any) -> Any:
    """Walk a possibly-nested object, returning JSON-friendly primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_json_friendly(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_friendly(v) for k, v in obj.items()}
    # Dataclass-like (StepLog / Verdict / ResolvedObject) → dict
    if hasattr(obj, "__dataclass_fields__"):
        return _to_json_friendly(asdict(obj))
    # Last resort: str()
    return str(obj)


_WARNED_UNKNOWN_KEYS: set[str] = set()


def _warn_unknown_state_keys(state: VisualState) -> None:
    """Print a one-shot warning for any top-level key not in VISUAL_STATE_KEYS.

    Each unrecognised key is reported ONCE per process (`_WARNED_UNKNOWN_KEYS`
    is a module-level cache) so a multi-stage run doesn't drown the user in
    repeated warnings. Disabled by setting env var
    ``VISUAL_STRICT_STATE_KEYS=0``.

    Stays non-fatal on purpose — a stray write is a smell, not a crash-worthy
    bug. After we've used this in a few runs and confirmed VISUAL_STATE_KEYS
    is exhaustive, we can promote to ``raise`` if desired.
    """
    import os
    if os.environ.get("VISUAL_STRICT_STATE_KEYS") == "0":
        return
    unknown = set(state.keys()) - VISUAL_STATE_KEYS - _WARNED_UNKNOWN_KEYS
    for k in sorted(unknown):
        print(f"[state] WARNING: unknown top-level key {k!r} "
              f"written to state.json (not declared in VISUAL_STATE_KEYS); "
              f"set VISUAL_STRICT_STATE_KEYS=0 to suppress")
        _WARNED_UNKNOWN_KEYS.add(k)


def save_state(state: VisualState, run_root: str | Path) -> Path:
    """Atomically write state to ``<run_root>/state.json``.

    Uses a ``<path>.tmp`` + ``os.replace`` swap so concurrent reads can never
    see a half-written file. Warns once per process on any key not in
    VISUAL_STATE_KEYS — guards against typos like ``state["pickp_object"]``.
    """
    import os
    _warn_unknown_state_keys(state)
    path = state_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(_to_json_friendly(state), f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def load_state(run_root: str | Path) -> VisualState:
    """Load state from disk. Returns ``{}`` if the file doesn't exist yet."""
    path = state_path(run_root)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def update_state(run_root: str | Path, **patch) -> VisualState:
    """Read-modify-write helper. Returns the updated state."""
    state = load_state(run_root)
    state.update(patch)
    save_state(state, run_root)
    return state


def append_history(run_root: str | Path, agent: str, note: str = "") -> None:
    """Append a StepLog entry to ``state["history"]`` and persist."""
    state = load_state(run_root)
    state.setdefault("history", []).append({"agent": agent, "note": note})
    save_state(state, run_root)


def stage_done(run_root: str | Path, stage_name: str) -> bool:
    """Convenience: has ``stages.<stage_name>.ok`` been set to True?"""
    state = load_state(run_root)
    s = state.get("stages", {}).get(stage_name)
    return bool(s and s.get("ok") is True)


def record_stage(
    run_root: str | Path,
    stage_name: str,
    ok: bool,
    outputs: dict | None = None,
    stats: dict | None = None,
    error: str | None = None,
) -> None:
    """Record a skill / subagent outcome under ``state["stages"][stage_name]``."""
    state = load_state(run_root)
    stages = state.setdefault("stages", {})
    stages[stage_name] = {
        "ok": ok,
        "outputs": outputs or {},
        "stats": stats or {},
        "error": error,
    }
    save_state(state, run_root)


__all__ = [
    "StepLog",
    "Verdict",
    "ResolvedObject",
    "VisualState",
    "VISUAL_STATE_KEYS",
    "state_path",
    "save_state",
    "load_state",
    "update_state",
    "append_history",
    "stage_done",
    "record_stage",
]
