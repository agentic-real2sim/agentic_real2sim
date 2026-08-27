"""InputBundle → SceneConfig converter (copied from v1 agents/scene_config.py).

Pure function — no @checkpoint, no PipelineState dependency. Lives in v2 so
Stage-2+ skills can build sim configs without importing the v1 agent layer.

Responsibilities:
  - glob collision meshes under each object's collision_mesh_dir
  - wrap pose_source_path into FoundationPoseDir
  - pick GroundSpec.reference_object (manifest value, else first object)
  - self-check: unique names, ground ref ∈ objects
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ar2s.agents_sysid.inputs import InputBundle, ObjectInput
from ar2s.droid_sim.scene.config import (
    FoundationPoseDir,
    GroundSpec,
    SceneConfig,
    SceneObject,
)


def _glob_collision_meshes(collision_dir: str) -> list[str]:
    if not collision_dir:
        return []
    d = Path(collision_dir)
    return [
        str(p) for p in sorted(
            d.glob("*_collision_*.obj"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
    ]


def _find_recovered_glb(name: str, visual_mesh_path: str) -> Path | None:
    """Locate ``<run_root>/meshes/<name>.glb`` from an object's visual.obj.

    Walks up rather than indexing a fixed number of parents: the bundle has
    been laid out both as ``<run_root>/sysid_inputs/objects/<name>/`` and as
    ``<run_root>/sysid_inputs/<id>_agent/objects/<name>/``, and a hardcoded
    depth silently resolves to the wrong directory on the other one — which
    costs a colour rather than raising.
    """
    here = Path(visual_mesh_path).resolve().parent
    for ancestor in [here, *here.parents][:8]:
        candidate = ancestor / "meshes" / f"{name}.glb"
        if candidate.is_file():
            return candidate
    return None


def _mesh_average_rgba(name: str, visual_mesh_path: str
                       ) -> tuple[float, float, float, float] | None:
    """Flat colour for an object: the mean vertex colour of its recovered GLB.

    mesh_recover writes a vertex-coloured ``<run_root>/meshes/<name>.glb``; the
    ``objects/<name>/visual.obj`` the manifest points at is geometry only, so a
    scene built from the manifest alone renders every object MuJoCo grey. That
    is invisible to grasp physics but not to a VLA policy, which picks targets
    by colour, nor to anyone reviewing a sweep video.

    SAM3D emits vertex colours and no UV texture, and MuJoCo has no per-vertex
    mesh colour, so averaging is the available reduction — the same one the
    sim_bundle packer applied (verified to reproduce its shipped
    ``meta.object_rgba`` to <1e-3).

    Returns None when the GLB is absent or carries no vertex colours; the geom
    then keeps MuJoCo's default.
    """
    glb = _find_recovered_glb(name, visual_mesh_path)
    if glb is None:
        return None
    try:
        import trimesh
        mesh = trimesh.load(str(glb), force="mesh")
        vertex_colors = getattr(mesh.visual, "vertex_colors", None)
        if vertex_colors is None or len(vertex_colors) == 0:
            return None
        rgb = np.asarray(vertex_colors)[:, :3].mean(axis=0) / 255.0
    except Exception as e:  # noqa: BLE001
        print(f"[scene_config] WARNING: cannot read {glb} for {name} colour ({e})")
        return None
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _scene_object_from_input(obj: ObjectInput) -> SceneObject:
    return SceneObject(
        name=obj.name,
        visual_mesh_path=obj.visual_mesh_path,
        scale_path=obj.scale_path,
        pose_source=FoundationPoseDir(
            path=obj.pose_source_path,
            init_frame=obj.init_frame,
        ),
        collision_mesh_paths=_glob_collision_meshes(obj.collision_mesh_dir),
        mass=obj.mass,
        friction=obj.friction,
        pos_offset=obj.pos_offset,
        static=obj.static,
        rgba=_mesh_average_rgba(obj.name, obj.visual_mesh_path),
    )


def build_scene_config(inputs: InputBundle) -> SceneConfig:
    """Pure converter. Caller may further mutate cfg.ground, cfg.cameras."""
    objects = [_scene_object_from_input(o) for o in inputs.objects]
    names = [o.name for o in objects]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate object names: {names}")

    ground_ref = inputs.ground.reference_object or (names[0] if names else "")
    if ground_ref and ground_ref != "GROUND" and ground_ref not in names:
        # Defensive: episode.py already rewrites a stale ground.reference
        # to the first surviving object before we get here. This branch
        # protects callers that build SceneConfig from a hand-crafted
        # InputBundle without going through episode loading. The "GROUND"
        # sentinel (from ``geometry_prior`` Stage 0) is intentionally
        # passed through — downstream sim code detects it and uses
        # ``derive_ground_pose_world_xy`` instead of the FP-derived path.
        if not names:
            raise ValueError(
                f"ground.reference_object {ground_ref!r} not in objects "
                f"(and objects list is empty)"
            )
        fallback = names[0]
        print(f"[scene_config] WARNING: ground.reference_object {ground_ref!r} "
              f"not in objects {names} and not the GROUND sentinel; "
              f"falling back to {fallback!r}")
        ground_ref = fallback

    return SceneConfig(
        scene_name=inputs.scene_name,
        robot_traj_path=inputs.robot_traj_path,
        objects=objects,
        ground=GroundSpec(
            reference_object=ground_ref,
            local_down_axis=inputs.ground.local_down_axis,
            offset=inputs.ground.offset,
        ),
        cameras=inputs.cameras,
    )


__all__ = ["build_scene_config"]
