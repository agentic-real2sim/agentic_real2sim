"""Stage 0 of the sysid droid pipeline — convex-decompose visual meshes and
patch the episode manifest with the resulting collision_dir.

visual emit produces a complete episode folder but does NOT generate
convex collision parts: ``objects/<name>/visual.obj`` is the raw SAM3D
mesh which can be concave (bowl, mug, lid, pot). MuJoCo only supports
convex collision geoms, so any object whose ``collision_mesh_paths``
falls back to the visual mesh will misbehave physically.

This stage runs CoACD per object (see
``ar2s/agents_sysid/skills/collision_decompose.py``), writes
``objects/<name>/collision/<sanitized>_collision_<i>.obj``, and adds
``collision_dir: objects/<name>/collision`` to that object's manifest
entry. Downstream ``ar2s.agents_sysid.episode.load_episode`` + ``scene_config``
then pick up the per-piece collision meshes automatically.

Idempotent: a subsequent re-run with ``force=False`` keeps existing
pieces and re-patches the manifest if needed.

The Stage 1 LLM ``episode_agent`` pathway used to emit ``collision_dir``
opportunistically; with that pathway deleted from the droid CLI, this
deterministic step replaces it. (phystwin / g1 do not use CoACD —
they manage their own collision representation.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ar2s.agents_sysid.skills.collision_decompose import (
    DecomposeResult,
    decompose_episode,
)


_COLLISION_DIRNAME = "collision"


@dataclass
class CollisionPrepReport:
    episode_dir: Path
    per_object: dict[str, DecomposeResult] = field(default_factory=dict)
    manifest_patched: bool = False
    errors: list[str] = field(default_factory=list)


def prep_episode(
    episode_dir: Path | str,
    *,
    threshold: float = 0.05,
    force: bool = False,
    objects: list[str] | None = None,
    **decompose_kwargs,
) -> CollisionPrepReport:
    """Convex-decompose every object in an episode + patch manifest.yaml.

    Args:
        episode_dir: path to an episode folder built by ``agents_visual``
            (must already contain ``manifest.yaml`` + ``objects/<name>/visual.obj``).
        threshold: CoACD concavity cutoff (forwarded to ``decompose_mesh``).
        force: re-run CoACD even if ``collision/`` already has pieces.
        objects: explicit object subdirs to process; None ⇒ every subdir
            of ``objects/`` that has a ``visual.obj``.
        **decompose_kwargs: extra args forwarded to ``decompose_mesh``
            (``max_convex_hull``, ``preprocess_mode``, ``seed``, ``log_level``).

    Returns:
        :class:`CollisionPrepReport` with the per-object decompose result
        and a flag indicating whether the manifest was rewritten.

    Raises:
        FileNotFoundError if ``manifest.yaml`` is missing.
        Per-object CoACD failures are captured in ``per_object[name].error``
        and do NOT raise — the rest of the episode still gets processed.
    """
    episode_dir = Path(episode_dir).resolve()
    manifest_path = episode_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.yaml not found under {episode_dir} — run "
            f"python -m ar2s.run_pipeline first to emit the episode"
        )

    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, dict) or "objects" not in manifest:
        raise ValueError(
            f"manifest.yaml at {manifest_path} is malformed (missing 'objects')"
        )

    # Default to the objects listed in the manifest — avoids running CoACD on
    # stale leftover directories (e.g. ``_excluded_foil_packet/``) that survived
    # an earlier visual emit but were dropped from the manifest.
    manifest_names = [o["name"] for o in manifest["objects"]]
    if objects is None:
        objects = manifest_names

    report = CollisionPrepReport(episode_dir=episode_dir)

    report.per_object = decompose_episode(
        episode_dir,
        objects=objects,
        threshold=threshold,
        force=force,
        **decompose_kwargs,
    )

    for name, r in report.per_object.items():
        if r.error:
            report.errors.append(f"{name}: {r.error}")

    # Patch manifest: add collision_dir field for every object that now
    # has a non-empty pieces list (skipped existing pieces count too).
    changed = False
    for obj in manifest["objects"]:
        obj_name = obj.get("name")
        res = report.per_object.get(obj_name)
        if res is None:
            continue
        if res.error or not res.pieces:
            continue
        rel_dir = f"objects/{obj_name}/{_COLLISION_DIRNAME}"
        if obj.get("collision_dir") != rel_dir:
            obj["collision_dir"] = rel_dir
            changed = True

    if changed:
        with open(manifest_path, "w") as f:
            yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)
        report.manifest_patched = True

    return report


__all__ = ["CollisionPrepReport", "prep_episode"]
