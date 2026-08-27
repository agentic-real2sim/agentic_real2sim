"""z_anchor — pick the most-trusted FP z-pose, propagate offsets through support tree.

Why this exists
---------------
FoundationPose's z-translation drifts 30–70 cm on large planar / repetitive-
texture surfaces (tables, counters). The mesh registers cleanly in the image,
but its absolute world-frame z is wrong. As a result, the ``support_tree``
relation "marker rests on table" is satisfied conceptually but **not** in
world coordinates — the marker floats centimetres above the (incorrectly-low)
table top.

This skill repairs the layering:

  1. A single VLM call picks **one** object from the non-GROUND object set
     whose FP world-z is most plausible (small / well-textured objects beat
     large / planar / repetitive ones).
  2. That object's z is taken as the anchor: it does **not** move.
  3. A double-direction BFS over the support tree slides every other object
     along z so each child's bottom sits exactly on its parent's top
     (descending) or each parent's top rises to meet its child's bottom
     (ascending), depending on where the BFS reaches them from the anchor.
  4. After every non-GROUND object has a ``pos_offset.z``, the world XY
     ground plane is placed at the lowest world z so the deepest surface
     becomes z = 0 with ``ground.offset`` storing how far below the manifest's
     world-origin the plane sits.

The skill is fully deterministic after the single VLM call. ``GROUND`` is a
sentinel — the world XY plane — so it never carries a ``pos_offset`` of its
own; everything below it is captured by ``ground.offset``.

Edge cases
----------
- Single non-GROUND object: no VLM, that object IS the anchor.
- VLM picks a name not in the candidate set (hallucination): one retry; on
  second failure the largest-height object is chosen heuristically (most
  likely a table; trusting it is usually wrong, so the skill reports
  ``degraded=True`` and a reason).
- Tree contains a cycle / orphan: caller is expected to surface that via
  ``support_tree.degraded``; we still run BFS from anchor and report which
  objects were unreachable.

Public surface
--------------
- ``compute_world_z_ranges(episode_dir)`` — per-object ``(z_min, z_max)``
  using the current ``visual.obj`` × ``scale.npy`` × FP frame-0 pose.
- ``propagate(anchor, tree, z_ranges)`` — BFS, returns
  ``{name: pos_offset_z}``.
- ``anchor_decide_episode(...)`` — full pipeline, returns ``ZAnchorResult``.
- ``patch_manifest(episode_dir, result)`` — write back into manifest.yaml.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
import yaml
from pydantic import BaseModel, Field

from ar2s.agent_configs import image_to_data_url, vlm_call_first_success
from ar2s.agents_geometry_prior.mesh_orient import (
    _earliest_pose,
    _load_camera,
)
from ar2s.droid_sim._util import snake_case_name


_AGENT_PATH = "geometry_prior.subagents.z_anchor"
_PROMPT_PATH = Path(__file__).with_name("z_anchor_prompt.md")
_GROUND = "GROUND"
_ROBOT_NAMES = {"robot", "robot arm", "robotic arm", "arm", "gripper"}
_ROBOT_NAME_IDS = {snake_case_name(n) for n in _ROBOT_NAMES}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ZAnchorResult:
    anchor: str = ""                       # name of the trusted object
    pos_offset_z: dict[str, float] = field(default_factory=dict)
    ground_offset: float = 0.0
    z_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    used_vlm: bool = False
    vlm_reason: str = ""
    degraded: bool = False
    degraded_reason: str = ""
    unreached: list[str] = field(default_factory=list)
    error: str = ""


class _AnchorChoice(BaseModel):
    object: str = Field(
        description="Pick exactly one name from the candidate list (case-insensitive)."
    )
    reasoning: str = Field(
        description="One short sentence: why this object's FoundationPose z is the most accurate."
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _world_vertices(obj_dir: Path, c2w: np.ndarray) -> np.ndarray:
    """Load mesh + scale + FP frame-0 pose; return scaled vertices in world."""
    pose_cam_from_obj, _ = _earliest_pose(obj_dir)
    pose_o2w = c2w @ pose_cam_from_obj
    mesh = trimesh.load(obj_dir / "visual.obj", force="mesh")
    V_raw = np.asarray(mesh.vertices, dtype=np.float64)
    if (obj_dir / "scale.npy").exists():
        s = np.asarray(np.load(obj_dir / "scale.npy"), dtype=np.float64).reshape(-1)
        V_scaled = V_raw * (float(s[0]) if s.size == 1 else s[:3])
    else:
        V_scaled = V_raw
    return V_scaled @ pose_o2w[:3, :3].T + pose_o2w[:3, 3]


def compute_world_z_ranges(
    episode_dir: Path | str,
    *,
    primary_id: int | None = None,
    skip_objects: tuple[str, ...] = ("robot",),
) -> dict[str, tuple[float, float]]:
    """Return ``{name: (z_min, z_max)}`` in world for every manifest object.

    Reads the CURRENT ``visual.obj`` — call after ``mesh_orient`` and
    ``mesh_axis_align`` have done their rewrites.
    """
    episode_dir = Path(episode_dir).resolve()
    manifest = yaml.safe_load((episode_dir / "manifest.yaml").read_text())
    if primary_id is None:
        primary_id = int(manifest["cameras"]["primary_id"])
    _K, cam_w2c = _load_camera(episode_dir, primary_id)
    c2w = np.linalg.inv(cam_w2c)
    out: dict[str, tuple[float, float]] = {}
    for obj in manifest.get("objects", []):
        name = str(obj.get("name", ""))
        if not name or name in skip_objects:
            continue
        V_world = _world_vertices(episode_dir / "objects" / name, c2w)
        out[name] = (float(V_world[:, 2].min()), float(V_world[:, 2].max()))
    return out


# ---------------------------------------------------------------------------
# Tree / propagation
# ---------------------------------------------------------------------------

def _adjacency(relations: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return ``(parent_of, children_of)`` from the support_tree relations.

    ``parent_of[child] = supported_by`` (may be ``"GROUND"`` sentinel).
    ``children_of[parent]`` is a sorted list of names whose ``supports_by`` =
    this parent. ``GROUND`` appears as a key in ``children_of`` (its values
    are the subtree roots) but never in ``parent_of`` (it's a sentinel).
    """
    parent_of: dict[str, str] = {}
    children_of: dict[str, list[str]] = {}
    for r in relations:
        c = r.get("object", "")
        p = r.get("supported_by", "")
        if not c or not p:
            continue
        parent_of[c] = p
        children_of.setdefault(p, []).append(c)
    for k in children_of:
        children_of[k].sort()
    return parent_of, children_of


def propagate(
    anchor: str,
    parent_of: dict[str, str],
    children_of: dict[str, list[str]],
    z_ranges: dict[str, tuple[float, float]],
) -> tuple[dict[str, float], list[str]]:
    """BFS from anchor, computing ``pos_offset.z`` for every reachable non-GROUND node.

    Returns ``(pos_offset_z, unreached_names)``. ``GROUND`` is treated as a
    sentinel: we walk *through* edges that touch it (to discover other root
    subtrees), but do not assign it an offset — the world XY plane's
    placement is computed separately by ``ground_offset_from``.

    Direction rules (relative to the BFS frontier ``u``):
      - ``v`` is ``u``'s child (i.e. ``parent_of[v] == u``):
            v.pos_offset.z = u.z_max_now − v.z_min
            ↳ drop / lift the child so its bottom sits on the current top.
      - ``v`` is ``u``'s parent (i.e. ``parent_of[u] == v``):
            v.pos_offset.z = u.z_min_now − v.z_max
            ↳ lift / drop the parent so its top sits at the current bottom.
    """
    offsets: dict[str, float] = {anchor: 0.0}
    queue: deque[str] = deque([anchor])
    while queue:
        u = queue.popleft()
        u_zmin = z_ranges[u][0] + offsets[u]
        u_zmax = z_ranges[u][1] + offsets[u]
        # 1. Walk to children (descend gravity-style).
        for v in children_of.get(u, []):
            if v == _GROUND or v in offsets or v not in z_ranges:
                continue
            offsets[v] = u_zmax - z_ranges[v][0]
            queue.append(v)
        # 2. Walk to parent (ascend to meet u from below).
        p = parent_of.get(u)
        if p and p != _GROUND and p not in offsets and p in z_ranges:
            offsets[p] = u_zmin - z_ranges[p][1]
            queue.append(p)

    # Anything in z_ranges that's not in offsets is unreachable from anchor.
    unreached = sorted(n for n in z_ranges if n not in offsets)
    return offsets, unreached


def ground_offset_from(
    z_ranges: dict[str, tuple[float, float]],
    pos_offset_z: dict[str, float],
) -> float:
    """Place the world XY plane at the lowest world z of any positioned object.

    Returns the unsigned distance from the manifest's world origin (z = 0)
    down to the ground plane: ``ground.offset = -min(z_min + offset)``. A
    positive value means the MuJoCo ground sits at z = −offset (below the
    world origin); a negative value would mean the ground floats above the
    origin (unusual but mathematically allowed).
    """
    if not pos_offset_z:
        return 0.0
    lowest = min(z_ranges[n][0] + pos_offset_z[n] for n in pos_offset_z if n in z_ranges)
    return float(-lowest)


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------

def _format_z_ranges(z_ranges: dict[str, tuple[float, float]]) -> str:
    width = max(len(n) for n in z_ranges) if z_ranges else 4
    return "\n  ".join(
        f"{n:<{width}}  z_min = {lo:+.4f}  z_max = {hi:+.4f}  height = {hi-lo:.4f} m"
        for n, (lo, hi) in z_ranges.items()
    )


def _format_relations(relations: list[dict]) -> str:
    return "\n  ".join(
        f"{r['object']}  rests on  {r['supported_by']}" for r in relations
    )


def _vlm_pick(
    first_frame_path: Path,
    candidates: list[str],
    z_ranges: dict[str, tuple[float, float]],
    relations: list[dict],
) -> _AnchorChoice | None:
    """One structured-output VLM call; ``None`` if every configured model fails."""
    system = _PROMPT_PATH.read_text()
    user_text = (
        f"Candidate objects (pick exactly one): {candidates}\n\n"
        f"Support tree (GROUND = lowest contact surface in scene):\n  "
        f"{_format_relations(relations)}\n\n"
        f"Per-object world-frame z (z-up, in metres):\n  "
        f"{_format_z_ranges(z_ranges)}\n\n"
        f"Pick the object whose FoundationPose z is the most accurate. "
        f"The pipeline will keep that object's world position fixed and "
        f"slide the others along z so the support tree's contact "
        f"relationships are satisfied."
    )
    return vlm_call_first_success(
        _AGENT_PATH,
        system=system,
        user_text=user_text,
        images=[image_to_data_url(first_frame_path)],
        output_schema=_AnchorChoice,
    )


def _resolve_choice(choice: str, candidates: list[str]) -> str:
    """Case-insensitive match, return canonical name from candidates or ''."""
    lower = snake_case_name(choice)
    for c in candidates:
        if snake_case_name(c) == lower:
            return c
    return ""


def _canonicalize_relations(
    relations: list[dict],
    z_ranges: dict[str, tuple[float, float]],
) -> list[dict]:
    """Map support_tree snake_case ids back to manifest object names."""
    name_by_id = {snake_case_name(n): n for n in z_ranges}
    out: list[dict] = []
    for r in relations:
        child_raw = str(r.get("object", "")).strip()
        if not child_raw:
            continue
        child_id = snake_case_name(child_raw)
        parent_raw = str(r.get("supported_by", "")).strip()
        if not child_id or child_id in _ROBOT_NAME_IDS:
            continue
        parent = (
            _GROUND
            if parent_raw.lower() == "ground"
            else name_by_id.get(snake_case_name(parent_raw), snake_case_name(parent_raw))
        )
        out.append({
            "object": name_by_id.get(child_id, child_id),
            "supported_by": parent,
        })
    return out


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def anchor_decide_episode(
    episode_dir: Path | str,
    relations: list[dict],
    *,
    primary_id: int | None = None,
    skip_vlm: bool = False,
    force_choice: str | None = None,
    first_frame_path: Path | str | None = None,
) -> ZAnchorResult:
    """Pick an anchor, propagate pos_offsets, compute ground.offset.

    Args:
        episode_dir: built episode (with manifest.yaml, cameras/, objects/).
        relations: support_tree edges (``[{"object", "supported_by"}, ...]``).
            Robot rows are filtered out internally.
        primary_id: camera serial; defaults to manifest's ``cameras.primary_id``.
        skip_vlm: skip the VLM call and use heuristic fallback (smallest-height
            non-GROUND object becomes anchor). Useful for tests / offline runs.
        force_choice: override the VLM with this name (debug). Must match a
            candidate, case-insensitive.
        first_frame_path: image given to the VLM; defaults to
            ``<episode_dir>/real/first_frame.png``.

    Returns:
        ``ZAnchorResult`` with the chosen anchor, per-object ``pos_offset.z``,
        and ``ground.offset``. The manifest is **not** written; call
        ``patch_manifest`` to persist.
    """
    episode_dir = Path(episode_dir).resolve()
    result = ZAnchorResult()

    z_ranges = compute_world_z_ranges(episode_dir, primary_id=primary_id)
    result.z_ranges = z_ranges
    if not z_ranges:
        result.error = "no non-robot objects in manifest"
        return result

    relations_clean = _canonicalize_relations(relations, z_ranges)
    parent_of, children_of = _adjacency(relations_clean)

    candidates = sorted(z_ranges)  # all non-robot manifest objects
    if not candidates:
        result.error = "no non-robot objects in manifest"
        return result

    # ---- pick the anchor ----
    anchor = ""
    if force_choice:
        anchor = _resolve_choice(force_choice, candidates)
        if not anchor:
            result.error = f"force_choice {force_choice!r} not in candidates {candidates}"
            return result
        result.vlm_reason = f"forced: {force_choice}"
    elif len(candidates) == 1:
        anchor = candidates[0]
        result.vlm_reason = "trivial: only one non-GROUND object"
    elif skip_vlm:
        # Heuristic: pick the object with smallest height (most likely a
        # small, well-textured pickup object).
        anchor = min(candidates, key=lambda n: z_ranges[n][1] - z_ranges[n][0])
        result.vlm_reason = "heuristic fallback (skip_vlm): smallest-height object"
        result.degraded = True
        result.degraded_reason = "skip_vlm requested"
    else:
        if first_frame_path is None:
            first_frame_path = episode_dir / "real" / "first_frame.png"
        first_frame_path = Path(first_frame_path)
        if not first_frame_path.exists():
            result.error = f"first frame not found: {first_frame_path}"
            return result

        choice = _vlm_pick(first_frame_path, candidates, z_ranges, relations_clean)
        result.used_vlm = True
        if choice is not None:
            anchor = _resolve_choice(choice.object, candidates)
            if anchor:
                result.vlm_reason = choice.reasoning
        if not anchor:
            # Fallback: pick the smallest non-GROUND object (heuristic).
            anchor = min(candidates, key=lambda n: z_ranges[n][1] - z_ranges[n][0])
            result.degraded = True
            result.degraded_reason = (
                "VLM returned an unrecognised object; fell back to smallest-height"
            )
            result.vlm_reason = result.vlm_reason or "vlm parse failed"

    result.anchor = anchor

    # ---- propagate from anchor ----
    pos_offset_z, unreached = propagate(anchor, parent_of, children_of, z_ranges)
    result.pos_offset_z = pos_offset_z
    result.unreached = unreached
    if unreached:
        # An unreachable node usually means the support tree edge to that
        # node is missing or points to something we don't know about.
        result.degraded = True
        result.degraded_reason = (
            f"{result.degraded_reason}; unreachable from anchor: {unreached}"
            if result.degraded_reason
            else f"unreachable from anchor: {unreached}"
        )

    # ---- ground offset ----
    result.ground_offset = ground_offset_from(z_ranges, pos_offset_z)
    return result


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def patch_manifest(episode_dir: Path | str, result: ZAnchorResult) -> Path:
    """Persist anchor decision into ``manifest.yaml``.

    Writes:
      - ``objects[i].pos_offset = [0, 0, pos_offset_z]`` for every result entry.
      - ``ground.reference = "GROUND"`` (sentinel — world XY plane).
      - ``ground.offset = result.ground_offset``.

    Leaves ``ground.local_down_axis`` untouched.
    """
    episode_dir = Path(episode_dir).resolve()
    manifest_path = episode_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())

    for obj in manifest.get("objects", []):
        name = str(obj.get("name", ""))
        if name in result.pos_offset_z:
            dz = float(result.pos_offset_z[name])
            obj["pos_offset"] = [0.0, 0.0, dz]

    ground = manifest.setdefault("ground", {})
    ground["reference"] = _GROUND
    ground["offset"] = float(result.ground_offset)
    if "local_down_axis" not in ground:
        ground["local_down_axis"] = [0.0, 0.0, -1.0]

    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path


__all__ = [
    "ZAnchorResult",
    "anchor_decide_episode",
    "compute_world_z_ranges",
    "ground_offset_from",
    "patch_manifest",
    "propagate",
]
