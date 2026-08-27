#!/usr/bin/env python3
"""Restore manipulated-object vertex colors in one legacy pipeline USD.

This does not rerun MuJoCo or alter the source USD.  It verifies that the
complete pipeline run still contains the exact object topology used by the
selected legacy USD, then writes a standalone colored USDC replacement beside
it (or at ``--output``).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ar2s.agents_sysid.episode import load_episode
from ar2s.pipeline_artifacts import episode_bundle_dir
from ar2s.agents_sysid.scene_config import build_scene_config
from ar2s.droid_sim.usd_exporter import (
    assert_rotation_only_legacy_backup,
    author_vertex_display_color,
    bound_preview_surface,
    connect_display_color_material,
    load_mesh,
    load_vertex_colors,
)


@dataclass(frozen=True)
class ColorSource:
    object_name: str
    final_path: Path
    color_path: Path
    rgb: np.ndarray
    alpha: np.ndarray | None


@dataclass(frozen=True)
class RetrofitPlan:
    source: ColorSource
    visual_prim_path: Sdf.Path
    collision_xform_paths: tuple[Sdf.Path, ...]
    collision_mesh_paths: tuple[Sdf.Path, ...]


def _topology_signature(
    *,
    vertex_count: int,
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
) -> tuple[int, bytes, bytes]:
    """Return the complete, order-sensitive topology identity of a mesh."""
    counts = np.ascontiguousarray(face_vertex_counts, dtype=np.int64)
    indices = np.ascontiguousarray(face_vertex_indices, dtype=np.int64)
    return int(vertex_count), counts.tobytes(), indices.tobytes()


def _mesh_signature(mesh: object) -> tuple[int, bytes, bytes]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    assert faces.ndim == 2 and faces.shape[1] == 3, (
        f"only triangular object meshes are supported, got faces {faces.shape}"
    )
    return _topology_signature(
        vertex_count=len(vertices),
        face_vertex_counts=np.full(len(faces), 3, dtype=np.int64),
        face_vertex_indices=faces.reshape(-1),
    )


def _usd_mesh_signature(mesh: UsdGeom.Mesh) -> tuple[int, bytes, bytes]:
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    assert points is not None and counts is not None and indices is not None, mesh.GetPath()
    return _topology_signature(
        vertex_count=len(points),
        face_vertex_counts=np.asarray(counts),
        face_vertex_indices=np.asarray(indices),
    )


def _unique_manifest(run_dir: Path) -> Path:
    manifest = episode_bundle_dir(run_dir) / "manifest.yaml"
    assert manifest.is_file(), f"no {manifest} under {run_dir}"
    return manifest


def _recover_colors(run_dir: Path, object_name: str, final_path: Path) -> ColorSource:
    """Load one object's colors in the fixed provenance order from the plan."""
    assert final_path.is_file(), f"missing final visual OBJ for {object_name}: {final_path}"
    final_mesh = load_mesh(final_path)
    glb_path = run_dir / "meshes" / f"{object_name}.glb"
    if glb_path.is_file():
        color_mesh, colors = load_vertex_colors(
            glb_path, require_explicit_obj=False, allow_uniform=True,
        )
        assert colors is not None, f"GLB color source has no vertex colors: {glb_path}"
        assert_rotation_only_legacy_backup(
            final_mesh,
            color_mesh,
            final_path=final_path,
            backup_path=glb_path,
        )
        rgb, alpha = colors
        return ColorSource(object_name, final_path, glb_path, rgb, alpha)

    _, colors = load_vertex_colors(
        final_path, require_explicit_obj=True, allow_uniform=True,
    )
    if colors is not None:
        rgb, alpha = colors
        return ColorSource(object_name, final_path, final_path, rgb, alpha)

    for backup_name in ("visual_pre_align.obj", "visual_pre_orient.obj"):
        backup_path = final_path.parent / backup_name
        if not backup_path.is_file():
            continue
        backup_mesh, backup_colors = load_vertex_colors(
            backup_path, require_explicit_obj=True, allow_uniform=True,
        )
        if backup_colors is None:
            continue
        assert_rotation_only_legacy_backup(
            final_mesh,
            backup_mesh,
            final_path=final_path,
            backup_path=backup_path,
        )
        rgb, alpha = backup_colors
        return ColorSource(object_name, final_path, backup_path, rgb, alpha)

    raise AssertionError(
        f"no vertex-color source for {object_name}: checked {glb_path}, "
        f"{final_path}, visual_pre_align.obj, and visual_pre_orient.obj"
    )


def _usd_mesh_index(stage: Usd.Stage) -> dict[tuple[int, bytes, bytes], list[UsdGeom.Mesh]]:
    indexed: dict[tuple[int, bytes, bytes], list[UsdGeom.Mesh]] = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            indexed.setdefault(_usd_mesh_signature(mesh), []).append(mesh)
    return indexed


def _match_exactly_one(
    indexed_meshes: dict[tuple[int, bytes, bytes], list[UsdGeom.Mesh]],
    source_path: Path,
) -> UsdGeom.Mesh:
    source_mesh = load_mesh(source_path)
    matches = indexed_meshes.get(_mesh_signature(source_mesh), [])
    assert len(matches) == 1, (
        f"{source_path} maps to {len(matches)} USD meshes by complete topology; "
        "refusing ambiguous, reordered, decimated, or remeshed correspondence"
    )
    return matches[0]


def build_retrofit_plan(run_dir: Path, stage: Usd.Stage) -> tuple[RetrofitPlan, ...]:
    """Discover source meshes and their one-to-one legacy USD mesh matches."""
    manifest = _unique_manifest(run_dir)
    scene_config = build_scene_config(load_episode(manifest.parent))
    indexed_meshes = _usd_mesh_index(stage)
    plans: list[RetrofitPlan] = []
    for scene_object in scene_config.objects:
        final_path = Path(scene_object.visual_mesh_path)
        source = _recover_colors(run_dir, scene_object.name, final_path)
        visual_mesh = _match_exactly_one(indexed_meshes, final_path)
        collision_xforms: list[Sdf.Path] = []
        collision_meshes: list[Sdf.Path] = []
        for collision_path in scene_object.collision_mesh_paths:
            collision_mesh = _match_exactly_one(indexed_meshes, Path(collision_path))
            collision_parent = collision_mesh.GetPrim().GetParent()
            assert collision_parent.IsA(UsdGeom.Xform), (
                f"object collision parent is not an Xform: {collision_parent.GetPath()}"
            )
            collision_xforms.append(collision_parent.GetPath())
            collision_meshes.append(collision_mesh.GetPath())
        plans.append(
            RetrofitPlan(
                source,
                visual_mesh.GetPath(),
                tuple(collision_xforms),
                tuple(collision_meshes),
            )
        )
    return tuple(plans)


def _xform_sample_digest(stage: Usd.Stage) -> tuple[tuple[str, str, tuple[float, ...], str], ...]:
    """Capture all authored Xform samples without relying on viewer behavior."""
    digest: list[tuple[str, str, tuple[float, ...], str]] = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Xform):
            continue
        for attribute in prim.GetAttributes():
            samples = tuple(float(value) for value in attribute.GetTimeSamples())
            if samples:
                values = tuple(repr(attribute.Get(Usd.TimeCode(sample))) for sample in samples)
                digest.append((str(prim.GetPath()), attribute.GetName(), samples, repr(values)))
    return tuple(digest)


def _timeline(stage: Usd.Stage) -> tuple[float, float, float, float]:
    return (
        float(stage.GetStartTimeCode()),
        float(stage.GetEndTimeCode()),
        float(stage.GetTimeCodesPerSecond()),
        float(stage.GetFramesPerSecond()),
    )


def _all_mesh_content(stage: Usd.Stage) -> dict[Sdf.Path, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result: dict[Sdf.Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            result[mesh.GetPath()] = (
                np.asarray(mesh.GetPointsAttr().Get()),
                np.asarray(mesh.GetFaceVertexCountsAttr().Get()),
                np.asarray(mesh.GetFaceVertexIndicesAttr().Get()),
            )
    return result


def _inactive_mesh_content(
    layer: Sdf.Layer,
    path: Sdf.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a hidden mesh's authored values directly from its root-layer spec."""
    spec = layer.GetPrimAtPath(path)
    assert spec is not None, f"missing inactive mesh spec: {path}"
    return tuple(
        np.asarray(spec.attributes[name].default)
        for name in ("points", "faceVertexCounts", "faceVertexIndices")
    )


def _validate_output(
    output_stage: Usd.Stage,
    plans: tuple[RetrofitPlan, ...],
    *,
    expected_timeline: tuple[float, float, float, float],
    expected_xforms: tuple[tuple[str, str, tuple[float, ...], str], ...],
    expected_meshes: dict[Sdf.Path, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    assert expected_timeline == _timeline(output_stage)
    assert expected_xforms == _xform_sample_digest(output_stage)
    output_meshes = _all_mesh_content(output_stage)
    collision_mesh_paths = {
        mesh_path for plan in plans for mesh_path in plan.collision_mesh_paths
    }
    assert expected_meshes.keys() - collision_mesh_paths == output_meshes.keys()
    for path, source_values in expected_meshes.items():
        output_values = (
            _inactive_mesh_content(output_stage.GetRootLayer(), path)
            if path in collision_mesh_paths else output_meshes[path]
        )
        for source_value, output_value in zip(source_values, output_values, strict=True):
            assert np.array_equal(source_value, output_value), f"mesh content changed: {path}"
    for plan in plans:
        mesh = UsdGeom.Mesh.Get(output_stage, plan.visual_prim_path)
        primvars = UsdGeom.PrimvarsAPI(mesh)
        color = primvars.GetPrimvar("displayColor")
        assert color.GetInterpolation() == UsdGeom.Tokens.vertex
        assert len(color.Get()) == len(plan.source.rgb)
        opacity = primvars.GetPrimvar("displayOpacity")
        assert opacity.GetAttr().HasAuthoredValueOpinion() == (
            plan.source.alpha is not None
        )
        if plan.source.alpha is not None:
            assert opacity.GetInterpolation() == UsdGeom.Tokens.vertex
            assert len(opacity.Get()) == len(plan.source.alpha)
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        shader, _, _ = material.ComputeSurfaceSource()
        reader, output_name, _ = shader.GetInput("diffuseColor").GetConnectedSource()
        reader = UsdShade.Shader(reader.GetPrim())
        assert output_name == "result"
        assert reader.GetIdAttr().Get() == "UsdPrimvarReader_float3"
        assert reader.GetInput("varname").Get() == "displayColor"
        if plan.source.alpha is not None:
            reader, output_name, _ = shader.GetInput("opacity").GetConnectedSource()
            reader = UsdShade.Shader(reader.GetPrim())
            assert output_name == "result"
            assert reader.GetIdAttr().Get() == "UsdPrimvarReader_float"
            assert reader.GetInput("varname").Get() == "displayOpacity"
        for collision_xform_path in plan.collision_xform_paths:
            collision_spec = output_stage.GetRootLayer().GetPrimAtPath(collision_xform_path)
            assert collision_spec.GetInfo("active") is False, collision_xform_path


def apply_retrofit(stage: Usd.Stage, plans: tuple[RetrofitPlan, ...]) -> None:
    """Mutate an in-memory stage only after all source matching succeeded."""
    for plan in plans:
        mesh = UsdGeom.Mesh.Get(stage, plan.visual_prim_path)
        author_vertex_display_color(mesh, plan.source.rgb, plan.source.alpha)
        connect_display_color_material(mesh, include_opacity=plan.source.alpha is not None)
        for collision_xform_path in plan.collision_xform_paths:
            collision_parent = stage.GetPrimAtPath(collision_xform_path)
            assert collision_parent.IsA(UsdGeom.Xform), collision_xform_path
            collision_parent.SetActive(False)


def _validate_appearance_prerequisites(
    stage: Usd.Stage,
    plans: tuple[RetrofitPlan, ...],
) -> None:
    """Verify existing bindings without authoring a single USD opinion."""
    for plan in plans:
        bound_preview_surface(UsdGeom.Mesh.Get(stage, plan.visual_prim_path))


def _print_summary(
    source_usd: Path,
    output_usd: Path,
    plans: tuple[RetrofitPlan, ...],
    *,
    dry_run: bool,
) -> None:
    print(f"[retrofit] source : {source_usd}")
    print(f"[retrofit] output : {output_usd}")
    print(f"[retrofit] mode   : {'dry-run' if dry_run else 'write'}")
    for plan in plans:
        print(
            f"[retrofit] {plan.source.object_name}: "
            f"source={plan.source.color_path} "
            f"visual={plan.visual_prim_path} "
            f"colors={len(plan.source.rgb)} "
            f"collisions={len(plan.collision_xform_paths)}"
        )


def retrofit(
    *,
    run_dir: Path,
    source_usd: Path,
    output_usd: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[RetrofitPlan, ...]:
    """Validate, apply, and atomically write one standalone colored USDC."""
    run_dir = run_dir.expanduser().resolve()
    source_usd = source_usd.expanduser()
    output_usd = output_usd.expanduser()
    source_usd = (source_usd if source_usd.is_absolute() else run_dir / source_usd).resolve()
    output_usd = (output_usd if output_usd.is_absolute() else run_dir / output_usd).resolve()
    assert run_dir.is_dir(), f"run directory does not exist: {run_dir}"
    assert source_usd.is_file(), f"source USD does not exist: {source_usd}"
    assert source_usd.is_relative_to(run_dir), (
        f"selected USD must belong to run directory {run_dir}: {source_usd}"
    )
    assert output_usd != source_usd, "output USD must differ from source USD"
    assert not output_usd.exists() or overwrite, (
        f"output already exists (pass --overwrite): {output_usd}"
    )
    source_stage = Usd.Stage.Open(str(source_usd))
    assert source_stage is not None, f"failed to open source USD: {source_usd}"
    plans = build_retrofit_plan(run_dir, source_stage)
    expected_timeline = _timeline(source_stage)
    expected_xforms = _xform_sample_digest(source_stage)
    expected_meshes = _all_mesh_content(source_stage)
    _validate_appearance_prerequisites(source_stage, plans)
    _print_summary(source_usd, output_usd, plans, dry_run=dry_run)
    if dry_run:
        return plans

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_usd.parent,
        prefix=f".{output_usd.stem}.",
        suffix=".usdc",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        apply_retrofit(source_stage, plans)
        # Pipeline result USDs are one root layer.  Export that layer directly:
        # Stage.Export flattens inactive prims away, which would turn the
        # collision visibility edit into a silent deletion.
        assert not source_stage.GetRootLayer().subLayerPaths, (
            "legacy USD has sublayers and cannot be exported as one standalone "
            "crate without discarding inactive collision prims"
        )
        assert source_stage.GetRootLayer().Export(str(temporary_path)), temporary_path
        output_stage = Usd.Stage.Open(str(temporary_path))
        assert output_stage is not None, f"failed to reopen temporary USDC: {temporary_path}"
        _validate_output(
            output_stage,
            plans,
            expected_timeline=expected_timeline,
            expected_xforms=expected_xforms,
            expected_meshes=expected_meshes,
        )
        os.replace(temporary_path, output_usd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return plans


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--usd", type=Path, required=True, help="One legacy result.usd")
    parser.add_argument(
        "--output", type=Path,
        help="Standalone colored USD (default: result_colored.usd beside --usd)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_usd = args.usd
    output_usd = args.output or source_usd.with_name("result_colored.usd")
    retrofit(
        run_dir=args.run_dir,
        source_usd=source_usd,
        output_usd=output_usd,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
