"""apply_geometry_priors — Stage 0 orchestrator.

Drives the per-object / per-episode geometry corrections that turn a
visual bundle into a sim-ready bundle:

  1. ``mesh_axis_align.axis_align_episode`` — for every non-robot object,
     in support_tree topological order (parent → child), generate **6
     candidate orientations** (one per signed body axis ±X / ±Y / ±Z
     pointing to world +Z), render each in the full scene from the
     primary camera, and let a VLM pick the best match for the real
     frame-0 photo. Applies the chosen rotation by rewriting the per-frame
     FoundationPose ``.npy`` files (preserving translation — body-frame
     origin stays in place in world). Also computes per-object
     ``pos_offset.z`` so the object sits on its parent's top surface, and
     sets ``ground.offset`` = lowest world bottom. Patches
     ``manifest.yaml`` in place.

  2. ``mesh_orient.orient_episode`` (per object, VLM) — pick NONE / flipX /
     flipY / flipZ on the now-axis-aligned object; rewrite ``visual.obj``
     if non-NONE, backup ``visual_pre_orient.obj``. Handles the remaining
     "rotation about the chosen vertical axis" degree of freedom (e.g.
     mug handle on left vs right).

z_anchor is no longer invoked from this orchestrator; the new axis_align
subsumes its job (parent-attachment + ground.offset are computed per
candidate during the 6-way VLM pick). ``z_anchor.py`` stays in the tree
for rollback.

State requirements
------------------
- ``<episode_dir>/manifest.yaml`` — defines the object list, cameras, robot.
- ``<episode_dir>/objects/<name>/{visual.obj, scale.npy, foundation_pose/}``
- ``<run_root>/state.json`` — must carry ``support_tree.relations`` and
  ``segmentation_result[obj]`` (for SAM3 mask overlay on the reference
  frame).

Audit trace
-----------
Writes ``<episode_dir>/geometry_priors.json`` with per-object orient
choices and axis-align results.

User overrides
--------------
Per-object ``pos_offset`` and ``ground.offset`` already present in the
manifest are **overwritten**. Stage 0 is the source of truth for these
fields — if a downstream consumer wants to hand-tune, they should do so
after Stage 0.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ar2s.agents_geometry_prior.mesh_axis_align import (
    AxisAlignResult,
    axis_align_episode,
    recompute_z_after_orient,
)
from ar2s.agents_geometry_prior.mesh_orient import (
    OrientResult,
    orient_episode,
)


_ROBOT_NAMES = {
    "robot", "robot arm", "robotic arm", "arm", "gripper",
    "robot manipulator", "robotic gripper",
}


@dataclass
class GeometryReport:
    episode_dir: Path
    run_root: Path
    orient: list[OrientResult] = field(default_factory=list)
    axis_align: list[AxisAlignResult] = field(default_factory=list)
    geometry_priors_path: Path | None = None


def _candidate_object_names(manifest_data: dict) -> list[str]:
    out: list[str] = []
    for entry in manifest_data.get("objects") or []:
        name = entry.get("name") or ""
        if name and name.lower() not in _ROBOT_NAMES:
            out.append(name)
    return out


def _ensure_state(run_root: Path) -> None:
    """Sanity-check the visual run_root carries what axis_align needs."""
    state_path = run_root / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"state.json not found at {state_path}. The geometry-prior stage "
            f"needs the visual run_root (with state.json carrying "
            f"`support_tree`) alongside the episode bundle."
        )
    state = json.loads(state_path.read_text())
    tree = state.get("support_tree") or {}
    if not tree.get("relations"):
        raise RuntimeError(
            "state.json carries no `support_tree.relations`. Run the "
            "`support_tree` visual subagent before geometry_prior."
        )


def _result_to_json(obj) -> dict:
    out: dict = {}
    for k, v in asdict(obj).items():
        out[k] = _json_value(v)
    return out


def _json_value(v):
    import numpy as np
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_json_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_value(x) for k, x in v.items()}
    return str(v)


def apply_geometry_priors(
    episode_dir: str | Path,
    run_root: str | Path,
    *,
    primary_id: int | None = None,
    skip_orient_vlm: bool = False,
    skip_axis_align_vlm: bool = False,
    highlight_target: bool | None = None,
) -> GeometryReport:
    """Run the full geometry-prior stage on one emitted episode bundle.

    Args:
        episode_dir: emitted bundle (``manifest.yaml`` + ``objects/<name>/...``).
        run_root: visual run dir with ``state.json`` containing
            ``support_tree`` and ``segmentation_result``.
        primary_id: camera serial; defaults to manifest's ``cameras.primary_id``.
        skip_orient_vlm: bypass the mesh_orient VLM call (keeps existing
            orientation).
        skip_axis_align_vlm: bypass the axis_align VLM call (defaults to
            +X candidate). Debug/regression-test only.

    Returns:
        ``GeometryReport`` carrying per-object results and the path to
        ``geometry_priors.json``.

    Raises:
        FileNotFoundError if manifest.yaml or state.json is missing.
        RuntimeError if state.json lacks ``support_tree``.
    """
    episode_dir = Path(episode_dir).expanduser().resolve()
    run_root = Path(run_root).expanduser().resolve()
    manifest_path = episode_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.yaml not found at {manifest_path}")

    manifest_data = yaml.safe_load(manifest_path.read_text()) or {}
    candidates = _candidate_object_names(manifest_data)
    _ensure_state(run_root)
    report = GeometryReport(episode_dir=episode_dir, run_root=run_root)

    print(f"[geometry_prior] episode_dir = {episode_dir}")
    print(f"[geometry_prior] run_root    = {run_root}")
    print(f"[geometry_prior] objects     = {candidates}")

    # ---- ① mesh_axis_align (per object, 6-candidate VLM) ----
    # Topological order from GROUND-children down. Each object gets 6
    # candidates (3 body axes × 2 signs). Each candidate also computes
    # the parent-attachment pos_offset.z (slides object so its lowest
    # point sits on parent's top surface, ground for the root). VLM
    # picks 1 of 6 per object; we apply by rewriting FP per-frame
    # (preserving translation) + manifest pos_offset.z. ground.offset is
    # derived at the end from the lowest world bottom across all objects.
    print(f"\n[geometry_prior] ① mesh_axis_align on {len(candidates)} object(s) ...")
    align_results = axis_align_episode(
        episode_dir,
        run_root,
        primary_id=primary_id,
        skip_objects=("robot",),
        skip_vlm=skip_axis_align_vlm,
        # None -> module env default (AR2S_AXIS_HIGHLIGHT); bool -> explicit override
        **({} if highlight_target is None else {"highlight": highlight_target}),
    )
    for r in align_results:
        if r.error:
            print(f"  {r.object_name:<12s} → ERROR: {r.error}")
        elif r.applied:
            print(f"  {r.object_name:<12s} → chose {r.chosen_label}  "
                  f"pos_offset.z={r.pos_offset_z:+.4f}  top_z={r.top_z_world:+.4f}")
        else:
            print(f"  {r.object_name:<12s} → SKIP ({r.skipped_reason})")
    report.axis_align = align_results

    # ---- ② mesh_orient (per object, VLM) ----
    # Runs on the axis-aligned mesh. Picks rotation about the now-vertical
    # axis (flipX / flipY / flipZ) or NONE.
    print(f"\n[geometry_prior] ② mesh_orient on {len(candidates)} object(s) ...")
    orient_results: list[OrientResult] = []
    if skip_orient_vlm:
        print("  (skipped: skip_orient_vlm=True)")
    else:
        for name in candidates:
            r = orient_episode_one(episode_dir, name, primary_id=primary_id)
            orient_results.append(r)
            tag = r.chosen_flip if not r.error else f"ERROR: {r.error}"
            print(f"  {name:<12s} → {tag}")
    report.orient = orient_results

    # ---- ③ re-ground on the FINAL meshes ----
    # axis_align committed pos_offset.z / ground.offset from the PRE-flip
    # mesh; a non-NONE mesh_orient flip moves the mesh's bottom/top for any
    # centroid-asymmetric object, leaving those values stale (24 mm observed
    # on a mug). Replay the z chain against what visual.obj now contains.
    flipped = [r.object_name for r in orient_results
               if not r.error and r.chosen_flip != "NONE"]
    if flipped:
        print(f"\n[geometry_prior] ③ re-ground after mesh_orient "
              f"flips on {flipped} ...")
        recompute_z_after_orient(episode_dir, run_root, primary_id=primary_id)

    # ---- audit trace ----
    priors_path = episode_dir / "geometry_priors.json"
    payload = {
        "source": "ar2s.agents_geometry_prior",
        "schema_version": 2,
        "axis_align": [_result_to_json(r) for r in report.axis_align],
        "orient":     [_result_to_json(r) for r in report.orient],
    }
    priors_path.write_text(json.dumps(payload, indent=2))
    report.geometry_priors_path = priors_path
    print(f"[geometry_prior] wrote {priors_path}")

    return report


# ---------------------------------------------------------------------------
# Thin per-object wrapper around mesh_orient (its episode iterator swallows
# exceptions; we want explicit error reporting per object for the audit
# trace).
# ---------------------------------------------------------------------------

def orient_episode_one(
    episode_dir: Path,
    object_name: str,
    *,
    primary_id: int | None,
) -> OrientResult:
    from ar2s.agents_geometry_prior.mesh_orient import orient_object
    try:
        return orient_object(episode_dir, object_name, primary_id=primary_id)
    except Exception as exc:
        return OrientResult(
            object_name=object_name,
            chosen_flip="NONE",
            error=f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "GeometryReport",
    "apply_geometry_priors",
]
