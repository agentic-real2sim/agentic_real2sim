"""Physical-prior tools: material classification, mass compute, manifest patch.

The public orchestration path is the ReAct agent in
``ar2s.agents_physical_prior.orchestrator``. This module owns the deterministic
tool functions that agent calls. Given an emitted episode bundle
(``<run_root>/sysid_inputs/``) and its parent run dir, the tool
sequence:

  1. Loads ``<episode_dir>/manifest.yaml`` to determine which objects are
     actually emitted (the canonical sysid-input set). Robot/gripper entries
     are filtered out.
  2. Loads ``state.json`` from ``run_root`` only for the per-object
     perception artefacts the classifier needs:
       - ``stages.segmentation.outputs.objects[].masks_dir`` (per-object mask)
       - ``stages.svo_extract[_secondary].outputs.left_frames_dir`` (the RGB
         frames dir on the object's winning camera, primary or secondary)
       - ``object_keyframes[name]`` / fallback ``prompt_frame_index``
       - ``mask_camera_id_by_object[name]`` (cam routing for dual-view)
  3. For every emitted non-robot object, calls
     ``material_classify.classify_object`` to obtain
     ``{material, alternative, confidence, reasoning, density_kg_per_m3}``.
     Objects with missing masks/frames (e.g. degraded mesh-only objects)
     fall back to a recorded default rather than being skipped silently —
     they are still in the simulated scene so they still need a mass.
  4. Computes ``mass = density × raw_mesh_volume × scale³`` per object,
     reading mesh + scale from the **emitted bundle** (not from state).
  5. Writes ``<episode_dir>/physical_priors.json`` (audit trace of every
     VLM call + computed mass).
  6. Patches ``<episode_dir>/manifest.yaml`` in place: for each object whose
     mass we just computed AND whose existing ``mass`` is the legacy
     placeholder, overwrite it with the new value.
  7. Returns a summary dict for the caller / CLI to print.

This module owns the only mutation of ``manifest.yaml`` from this stage; it
never touches ``state.json`` (the run_root may have been GC'd by the time
sysid runs in a different context, so don't depend on writing back).

User overrides (``entry.object_mass_hints[name]``) take precedence and are
left untouched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yaml

from ar2s.agents_physical_prior.material_classify import classify_object


_ROBOT_NAMES = {
    "robot", "robot arm", "robotic arm", "arm", "gripper",
    "robot manipulator", "robotic gripper",
}
# Legacy placeholder once written by the visual emit. New manifests don't
# carry it (visual omits ``mass`` entirely when no hint is given), but old
# bundles still do and we keep recognising it so re-running physical_prior
# on those overwrites cleanly instead of bailing as "user already set it".
_PLACEHOLDER_MASS_KG = 0.01


@dataclass
class _PerObject:
    name: str
    material: str
    alternative: str | None
    confidence: str
    reasoning: str
    density_kg_per_m3: float
    raw_mesh_volume: float | None = None
    scale: float | None = None
    mass_kg: float | None = None
    error: str | None = None


@dataclass
class ApplyReport:
    episode_dir: Path
    run_root: Path
    classified: list[_PerObject] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)        # [{name, reason}]
    manifest_patched_for: list[str] = field(default_factory=list)
    physical_priors_path: Path | None = None


@dataclass
class PhysicalPriorContext:
    episode_dir: Path
    run_root: Path
    manifest_path: Path
    manifest_data: dict
    candidates: list[str]
    state: dict
    accepted: set[str]
    crop_root: Path
    report: ApplyReport


def _candidate_object_names(manifest_data: dict) -> list[str]:
    """Non-robot object names from the emitted manifest.

    The emitted manifest is the **single source of truth** for what sysid
    will actually simulate. A degraded object that got materialised by the
    visual stage (mesh recovered, pose tracked, mass-defaulted) is still in
    the manifest and still needs a physical prior — using
    ``segmentation_verdict.accepted`` here would silently drop those.
    """
    out: list[str] = []
    for entry in manifest_data.get("objects") or []:
        name = entry.get("name") or ""
        if name and name.lower() not in _ROBOT_NAMES:
            out.append(name)
    return out


def _masks_dir_for(state: dict, obj_name: str) -> Path | None:
    seg = state.get("stages", {}).get("segmentation", {})
    for entry in seg.get("outputs", {}).get("objects", []):
        if (entry.get("text") or "") == obj_name:
            md = entry.get("masks_dir") or ""
            if md:
                return Path(md)
    return None


def _frame_idx_for(state: dict, obj_name: str) -> int:
    """Per-object keyframe with global fallback (matches mesh_recover/mesh_scale)."""
    pfb = (
        state.get("object_keyframes")
        or state.get("prompt_frame_index_by_object")
        or {}
    )
    return int(pfb.get(obj_name, state.get("prompt_frame_index", 0)))


def _left_dir_for(state: dict, obj_name: str) -> Path | None:
    """RGB frames dir for ``obj_name``'s winning camera (mesh_recover convention).

    Routes secondary-camera objects to ``stages.svo_extract_secondary``'s
    left_frames_dir; everything else (primary or unset routing) to
    ``stages.svo_extract``. Returns None when the required stage outputs are
    missing.
    """
    primary_id  = state.get("primary_camera_id", "") or ""
    mask_cam_by = state.get("mask_camera_id_by_object") or {}
    cid = mask_cam_by.get(obj_name) or ""
    if cid and cid != primary_id:
        stage = state.get("stages", {}).get("svo_extract_secondary", {})
    else:
        stage = state.get("stages", {}).get("svo_extract", {})
    d = (stage.get("outputs") or {}).get("left_frames_dir") or ""
    return Path(d) if d else None


def _mass_from_density(
    visual_mesh_path: Path,
    scale_path: Path,
    density_kg_per_m3: float,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(mass_kg, raw_volume, scale)`` or ``(None, None, None)`` on failure.

    Mass = density × (raw mesh volume × scale³). The reconstructed mesh is
    in a unit-cube-normalised frame; ``scale.npy`` holds the isotropic
    factor that maps to real metres.
    """
    try:
        m = trimesh.load(visual_mesh_path, force="mesh")
    except Exception as exc:
        print(f"[apply] mesh load failed ({visual_mesh_path}): {exc}")
        return None, None, None
    raw_volume = float(m.volume)
    if not np.isfinite(raw_volume) or raw_volume <= 0.0:
        return None, None, None
    try:
        s = float(np.load(scale_path)[0])
    except Exception as exc:
        print(f"[apply] scale load failed ({scale_path}): {exc}")
        return None, None, None
    return float(density_kg_per_m3 * raw_volume * (s ** 3)), raw_volume, s


def _load_state(run_root: Path) -> dict:
    """Load state.json and rewrite any remote run_root prefix to ``run_root``.

    When state.json was rsync'd from the machine that produced it (e.g. visual
    ran on the 5090, then rsync pulled the run dir locally), the embedded
    absolute paths (``left_frames_dir``, ``masks_dir``, …) still point at the
    remote ``<remote_base>/outputs/run_pipeline/<ep_id>`` prefix. We detect
    that prefix from ``svo_extract.outputs.left_frames_dir`` (which is always
    ``<run_root>/seq/<serial>/frames/left``) and string-replace it with the
    local ``run_root`` so downstream readers can resolve files on disk.
    """
    state_path = run_root / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"state.json not found at {state_path}. The physical-prior stage "
            f"needs the visual run_root (with state.json, segmentation masks, "
            f"and svo_extract frames) alongside the episode bundle."
        )
    txt = state_path.read_text()
    state = json.loads(txt)
    left_dir = (
        state.get("stages", {}).get("svo_extract", {})
        .get("outputs", {}).get("left_frames_dir") or ""
    )
    if left_dir and "/seq/" in left_dir:
        remote_root = left_dir.split("/seq/")[0]
        local_root = str(run_root)
        if remote_root and remote_root != local_root:
            print(f"[apply] localising state paths: {remote_root!r} -> {local_root!r}")
            txt = txt.replace(remote_root, local_root)
            state = json.loads(txt)
    return state


def _patch_manifest(
    manifest_path: Path,
    manifest_data: dict,
    mass_by_name: dict[str, float],
    *,
    user_hints: set[str],
) -> list[str]:
    """In-place update manifest.objects[].mass for non-overridden entries.

    ``manifest_data`` is the parsed YAML the caller already loaded for the
    candidate-object list — we mutate it in place and write it back so the
    file is only round-tripped once per stage.

    Returns the list of object names whose ``mass`` was actually changed.
    """
    objects = manifest_data.get("objects") or []
    patched: list[str] = []
    for entry in objects:
        name = entry.get("name")
        if not name or name in user_hints or name not in mass_by_name:
            continue
        existing = entry.get("mass")
        new_mass = round(float(mass_by_name[name]), 6)
        # Overwrite when:
        #   - mass is missing entirely (new visual emits omit it on purpose), OR
        #   - mass is the legacy 0.01 placeholder (old bundles).
        # Anything else means a downstream stage / earlier override already
        # set it deliberately — leave it alone.
        if existing is None or abs(float(existing) - _PLACEHOLDER_MASS_KG) <= 1e-9:
            entry["mass"] = new_mass
            patched.append(name)
        else:
            print(f"[apply] keeping existing mass for {name!r}: {existing}kg "
                  f"(not the placeholder)")
    manifest_path.write_text(yaml.safe_dump(manifest_data, sort_keys=False))
    return patched


def apply_physical_priors(
    episode_dir: str | Path,
    run_root: str | Path,
) -> ApplyReport:
    """Compatibility callable that runs the physical-prior ReAct agent.

    ``apply_physical_priors`` used to drive this stage directly. It now delegates
    to ``data_ingest() -> run_physical_prior()`` so callers still land on the
    ReAct implementation instead of a hidden fallback path.
    """
    from ar2s.agents_physical_prior.orchestrator import (
        data_ingest,
        run_physical_prior,
    )

    return run_physical_prior(data_ingest(episode_dir, run_root))


def load_physical_prior_context(
    episode_dir: str | Path,
    run_root: str | Path,
) -> PhysicalPriorContext:
    """Load manifest + visual state and initialise the stage report."""
    episode_dir = Path(episode_dir).expanduser().resolve()
    run_root = Path(run_root).expanduser().resolve()
    manifest_path = episode_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.yaml not found at {manifest_path}")

    manifest_data = yaml.safe_load(manifest_path.read_text()) or {}
    candidates = _candidate_object_names(manifest_data)
    state = _load_state(run_root)
    accepted = set((state.get("segmentation_verdict") or {}).get("accepted") or [])
    report = ApplyReport(episode_dir=episode_dir, run_root=run_root)
    return PhysicalPriorContext(
        episode_dir=episode_dir,
        run_root=run_root,
        manifest_path=manifest_path,
        manifest_data=manifest_data,
        candidates=candidates,
        state=state,
        accepted=accepted,
        # Crops are the only thing this stage writes, so they sit directly in
        # material_classify/ rather than behind a single-child crops/ level.
        crop_root=run_root / "material_classify",
        report=report,
    )


def classify_materials(context: PhysicalPriorContext) -> ApplyReport:
    """Classify material for each emitted non-robot object using the VLM helper."""
    state = context.state
    for obj_name in context.candidates:
        # The emitted bundle stores objects by canonical name (lowercased
        # ASCII-safe). resolve.py does this rewriting before emit, so the
        # bundle path ``objects/<name>/...`` uses the same name we read
        # back from manifest.objects[].name.
        obj_dir = context.episode_dir / "objects" / obj_name
        visual_obj = obj_dir / "visual.obj"
        scale_npy = obj_dir / "scale.npy"
        if not visual_obj.exists() or not scale_npy.exists():
            context.report.skipped.append({
                "name": obj_name,
                "reason": f"missing visual.obj or scale.npy in {obj_dir}",
            })
            continue

        masks_dir = _masks_dir_for(state, obj_name)
        # Per-object keyframe + winning-camera RGB dir (mesh_recover convention).
        frame_idx    = _frame_idx_for(state, obj_name)
        obj_left_dir = _left_dir_for(state, obj_name)
        prompt_frame = (
            obj_left_dir / f"frame_{frame_idx:06d}.png"
            if obj_left_dir is not None else None
        )

        if prompt_frame is None:
            # No left_frames_dir for this object's camera at all — pipeline
            # broken upstream. Last-resort default so downstream sees a real
            # density rather than the 0.01 placeholder.
            from ar2s.agents_physical_prior.material_classify import FALLBACK
            mat = dict(FALLBACK)
            cid = (state.get("mask_camera_id_by_object") or {}).get(obj_name)
            mat["reasoning"] = (
                "fallback — no left_frames_dir in svo_extract"
                + ("_secondary" if cid else "")
            )
            print(f"[apply] WARNING: {obj_name!r} → fallback material ({mat['reasoning']})")
        else:
            # accepted → bbox crop; otherwise → full keyframe with "find X"
            # prompt. Either way the VLM runs; we never silently keep 0.01.
            use_full_frame = obj_name not in context.accepted
            mat = classify_object(
                prompt_frame, masks_dir, obj_name,
                crop_root=context.crop_root, frame_idx=frame_idx,
                use_full_frame=use_full_frame,
            )

        pobj = _PerObject(
            name=obj_name,
            material=mat["material"],
            alternative=mat.get("alternative"),
            confidence=mat["confidence"],
            reasoning=mat["reasoning"],
            density_kg_per_m3=float(mat["density_kg_per_m3"]),
        )
        context.report.classified.append(pobj)
    return context.report


def compute_masses(context: PhysicalPriorContext) -> dict[str, float]:
    """Compute density x raw mesh volume x scale^3 for classified objects."""
    mass_by_name: dict[str, float] = {}
    for pobj in context.report.classified:
        obj_dir = context.episode_dir / "objects" / pobj.name
        visual_obj = obj_dir / "visual.obj"
        scale_npy = obj_dir / "scale.npy"
        mass, raw_vol, scale = _mass_from_density(
            visual_obj, scale_npy, pobj.density_kg_per_m3,
        )
        pobj.raw_mesh_volume = raw_vol
        pobj.scale = scale
        pobj.mass_kg = mass
        if mass is None:
            pobj.error = "mass estimation failed (mesh volume non-finite/non-positive)"
            print(f"[apply] WARNING: mass estimation failed for {pobj.name!r}")
        else:
            mass_by_name[pobj.name] = mass
    return mass_by_name


def write_physical_priors(report: ApplyReport) -> Path:
    """Write ``physical_priors.json`` and update ``report.physical_priors_path``."""
    priors_path = report.episode_dir / "physical_priors.json"
    priors_payload = {
        "source": "ar2s.agents_physical_prior",
        "schema_version": 1,
        "objects": [
            {
                "name":              p.name,
                "material":          p.material,
                "alternative":       p.alternative,
                "confidence":        p.confidence,
                "reasoning":         p.reasoning,
                "density_kg_per_m3": p.density_kg_per_m3,
                "raw_mesh_volume":   p.raw_mesh_volume,
                "scale":             p.scale,
                "mass_kg":           p.mass_kg,
                "error":             p.error,
            }
            for p in report.classified
        ],
        "skipped": report.skipped,
    }
    priors_path.write_text(json.dumps(priors_payload, indent=2))
    report.physical_priors_path = priors_path
    print(f"[apply] wrote {priors_path}")
    return priors_path


def patch_manifest_masses(
    context: PhysicalPriorContext,
    mass_by_name: dict[str, float],
) -> list[str]:
    """Patch manifest object masses for computed, non-user-overridden entries."""
    entry = context.state.get("entry") or {}
    mass_hints = (
        entry.get("object_mass_hints")
        or context.state.get("object_mass_hints")
        or {}
    )
    user_hints = set(mass_hints.keys())
    patched = _patch_manifest(
        context.manifest_path,
        context.manifest_data,
        mass_by_name,
        user_hints=user_hints,
    )
    context.report.manifest_patched_for = patched
    print(f"[apply] patched manifest mass for {len(patched)} object(s): {patched}")
    return patched


__all__ = [
    "ApplyReport",
    "PhysicalPriorContext",
    "apply_physical_priors",
    "classify_materials",
    "compute_masses",
    "load_physical_prior_context",
    "patch_manifest_masses",
    "write_physical_priors",
]
