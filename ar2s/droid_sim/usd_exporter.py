"""USD export helpers that restore object vertex colors lost by MuJoCo.

MuJoCo's ``MjModel`` stores mesh positions, normals, UVs, and faces, but not
per-vertex colors.  ``VertexColorUSDExporter`` deliberately leaves simulation
and animation to MuJoCo, then restores the colors from the post-geometry-prior
``visual.obj`` on the USD mesh authored by MuJoCo's exporter.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import trimesh
from mujoco.usd.exporter import USDExporter
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load one mesh without modifying its vertex order."""
    return trimesh.load(
        Path(path), force="mesh", process=False, maintain_order=True,
    )


def obj_has_explicit_vertex_colors(path: Path) -> bool:
    """Return whether an OBJ spells out RGB values on every vertex record."""
    if path.suffix.lower() != ".obj":
        return True
    vertex_records = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    ]
    assert vertex_records, f"OBJ contains no vertex records: {path}"
    return all(len(record) >= 7 for record in vertex_records)


def mesh_vertex_colors(
    mesh: trimesh.Trimesh,
    path: str | Path,
    *,
    require_explicit_obj: bool,
    allow_uniform: bool,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Return normalized vertex RGB(A), or ``None`` when no colors exist.

    Trimesh supplies opaque default grey for uncolored OBJ files.  Requiring
    explicit OBJ RGB records keeps that convenient default from becoming fake
    provenance in the legacy-retrofit path.
    """
    path = Path(path)
    if require_explicit_obj and not obj_has_explicit_vertex_colors(path):
        return None
    if path.suffix.lower() in {".glb", ".gltf"} and mesh.visual.kind != "vertex":
        return None
    colors = np.asarray(mesh.visual.vertex_colors)
    assert colors.ndim == 2 and colors.shape[0] == len(mesh.vertices), (
        f"{path} has malformed vertex colors: {colors.shape} for "
        f"{len(mesh.vertices)} vertices"
    )
    assert colors.shape[1] in (3, 4), (
        f"{path} has {colors.shape[1]} vertex-color channels; expected RGB or RGBA"
    )
    values = colors.astype(np.float32)
    assert np.isfinite(values).all(), f"{path} contains non-finite vertex colors"
    if float(values.max()) > 1.0:
        values /= 255.0
    assert float(values.min()) >= 0.0 and float(values.max()) <= 1.0, (
        f"{path} vertex colors are outside [0, 1] after normalization"
    )
    rgb = values[:, :3]
    if not allow_uniform and not np.any(np.ptp(rgb, axis=0) > 0.0):
        return None
    alpha = values[:, 3] if values.shape[1] == 4 else None
    if alpha is not None and np.allclose(alpha, 1.0):
        alpha = None
    return rgb, alpha


def load_vertex_colors(
    path: str | Path,
    *,
    require_explicit_obj: bool,
    allow_uniform: bool,
) -> tuple[trimesh.Trimesh, tuple[np.ndarray, np.ndarray | None] | None]:
    """Load a mesh and extract its validated vertex-color payload."""
    mesh = load_mesh(path)
    return mesh, mesh_vertex_colors(
        mesh,
        path,
        require_explicit_obj=require_explicit_obj,
        allow_uniform=allow_uniform,
    )


def assert_rotation_only_legacy_backup(
    final_mesh: trimesh.Trimesh,
    backup_mesh: trimesh.Trimesh,
    *,
    final_path: str | Path,
    backup_path: str | Path,
) -> None:
    """Prove a legacy backup differs only by a proper origin rotation."""
    final_path = Path(final_path)
    backup_path = Path(backup_path)
    final_vertices = np.asarray(final_mesh.vertices, dtype=np.float64)
    backup_vertices = np.asarray(backup_mesh.vertices, dtype=np.float64)
    assert len(backup_vertices) == len(final_vertices), (
        f"legacy color backup {backup_path} has {len(backup_vertices)} vertices; "
        f"final visual mesh {final_path} has {len(final_vertices)}"
    )
    assert np.array_equal(backup_mesh.faces, final_mesh.faces), (
        f"legacy color backup {backup_path} does not preserve final mesh topology"
    )
    u, _, vt = np.linalg.svd(backup_vertices.T @ final_vertices)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    residual = np.linalg.norm(backup_vertices @ rotation - final_vertices, axis=1)
    coordinate_scale = max(
        1.0,
        float(np.abs(backup_vertices).max()),
        float(np.abs(final_vertices).max()),
    )
    assert float(residual.max()) <= 1e-5 * coordinate_scale, (
        f"legacy color backup {backup_path} is not related to {final_path} "
        f"by one proper origin rotation (max residual {float(residual.max()):.3e})"
    )


def author_vertex_display_color(
    usd_mesh: UsdGeom.Mesh,
    rgb: np.ndarray,
    alpha: np.ndarray | None,
) -> None:
    """Author standard vertex-interpolated display color and optional alpha."""
    assert len(rgb) == len(usd_mesh.GetPointsAttr().Get())
    primvars = UsdGeom.PrimvarsAPI(usd_mesh)
    color_primvar = primvars.CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex,
    )
    color_primvar.Set(
        Vt.Vec3fArray([Gf.Vec3f(*(float(c) for c in color)) for color in rgb])
    )
    if alpha is not None:
        assert len(alpha) == len(rgb)
        opacity_primvar = primvars.CreatePrimvar(
            "displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex,
        )
        opacity_primvar.Set(Vt.FloatArray([float(value) for value in alpha]))


def bound_preview_surface(usd_mesh: UsdGeom.Mesh) -> UsdShade.Shader:
    """Return the mesh's bound PreviewSurface, asserting the USD contract."""
    material, _ = UsdShade.MaterialBindingAPI(usd_mesh).ComputeBoundMaterial()
    assert material.GetPrim().IsValid(), (
        f"USD mesh {usd_mesh.GetPath()} has no bound material"
    )
    shader, _, _ = material.ComputeSurfaceSource()
    assert shader.GetPrim().IsValid(), (
        f"bound material {material.GetPath()} has no surface shader"
    )
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface", (
        f"bound shader {shader.GetPath()} is not UsdPreviewSurface"
    )
    return shader


def connect_display_color_material(
    usd_mesh: UsdGeom.Mesh,
    *,
    include_opacity: bool,
) -> None:
    """Connect the mesh's actually bound PreviewSurface to display primvars."""
    shader = bound_preview_surface(usd_mesh)
    material, _ = UsdShade.MaterialBindingAPI(usd_mesh).ComputeBoundMaterial()
    material_path = material.GetPath()
    stage = usd_mesh.GetPrim().GetStage()
    color_reader = UsdShade.Shader.Define(
        stage, material_path.AppendPath("displayColor_reader"),
    )
    color_reader.CreateIdAttr("UsdPrimvarReader_float3")
    color_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    color_reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    shader.CreateInput(
        "diffuseColor", Sdf.ValueTypeNames.Color3f,
    ).ConnectToSource(color_reader.ConnectableAPI(), "result")
    if include_opacity:
        opacity_reader = UsdShade.Shader.Define(
            stage, material_path.AppendPath("displayOpacity_reader"),
        )
        opacity_reader.CreateIdAttr("UsdPrimvarReader_float")
        opacity_reader.CreateInput(
            "varname", Sdf.ValueTypeNames.Token,
        ).Set("displayOpacity")
        opacity_reader.CreateOutput("result", Sdf.ValueTypeNames.Float)
        shader.CreateInput(
            "opacity", Sdf.ValueTypeNames.Float,
        ).ConnectToSource(opacity_reader.ConnectableAPI(), "result")


class VertexColorUSDExporter(USDExporter):
    """MuJoCo USD exporter that authors ``displayColor`` for object visuals.

    ``object_visual_mesh_paths`` maps MuJoCo mesh asset names (for example,
    ``"mug_visual"``) to final post-geometry-prior OBJ files.  The mesh vertex
    order is an explicit contract: the source color count must exactly match
    both the compiled MuJoCo mesh and the generated USD mesh.
    """

    def __init__(
        self,
        *args,
        object_visual_mesh_paths: dict[str, str | Path],
        **kwargs,
    ):
        self._object_visual_mesh_paths = {
            mesh_name: Path(path)
            for mesh_name, path in object_visual_mesh_paths.items()
        }
        self._vertex_color_cache: dict[
            str, tuple[np.ndarray, np.ndarray | None]
        ] = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _load_mesh(path: Path) -> trimesh.Trimesh:
        return load_mesh(path)

    @staticmethod
    def _mesh_colors(
        mesh: trimesh.Trimesh,
        path: Path,
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Return colors only where the OBJ itself explicitly provides them."""
        return mesh_vertex_colors(
            mesh,
            path,
            require_explicit_obj=True,
            allow_uniform=True,
        )

    @staticmethod
    def _assert_rotation_only_legacy_backup(
        final_mesh: trimesh.Trimesh,
        backup_mesh: trimesh.Trimesh,
        *,
        final_path: Path,
        backup_path: Path,
    ) -> None:
        assert_rotation_only_legacy_backup(
            final_mesh,
            backup_mesh,
            final_path=final_path,
            backup_path=backup_path,
        )

    def _colors_for_mesh(self, mesh_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        cached = self._vertex_color_cache.get(mesh_name)
        if cached is not None:
            return cached

        visual_path = self._object_visual_mesh_paths[mesh_name]
        assert visual_path.is_file(), f"missing visual mesh for USD color export: {visual_path}"
        final_mesh = self._load_mesh(visual_path)
        colors = self._mesh_colors(final_mesh, visual_path)

        if colors is None:
            # Old bundles may have been produced before geometry_prior retained
            # visuals.  Only inspect adjacent backups, and only accept one that
            # actually contains non-uniform colors; new outputs never use this.
            backups = [
                visual_path.parent / "visual_pre_align.obj",
                visual_path.parent / "visual_pre_orient.obj",
            ]
            for backup in backups:
                if not backup.is_file():
                    continue
                backup_mesh = self._load_mesh(backup)
                backup_colors = self._mesh_colors(backup_mesh, backup)
                if backup_colors is not None:
                    self._assert_rotation_only_legacy_backup(
                        final_mesh,
                        backup_mesh,
                        final_path=visual_path,
                        backup_path=backup,
                    )
                    colors = backup_colors
                    break

        assert colors is not None, (
            f"{visual_path} has no non-uniform vertex colors and no usable "
            "visual_pre_*.obj backup"
        )
        self._vertex_color_cache[mesh_name] = colors
        return colors

    def _load_geom(self, geom: mujoco.MjvGeom):
        """Load the regular USD geom, then attach colors for registered meshes."""
        super()._load_geom(geom)

        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            return
        assert geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
        dataid = int(self.model.geom_dataid[geom.objid])
        assert dataid >= 0, f"mesh geom {geom.objid} has no mesh data id"
        mesh_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_MESH, dataid,
        )
        if mesh_name not in self._object_visual_mesh_paths:
            return

        geom_name = self._get_geom_name(geom)
        usd_geom = self.geom_refs[geom_name]
        assert hasattr(usd_geom, "usd_mesh"), (
            f"official USD exporter did not return USDMesh for {geom_name}"
        )
        usd_mesh = usd_geom.usd_mesh
        rgb, alpha = self._colors_for_mesh(mesh_name)
        compiled_vertex_count = int(self.model.mesh_vertnum[dataid])
        usd_vertex_count = len(usd_mesh.GetPointsAttr().Get())
        assert len(rgb) == compiled_vertex_count == usd_vertex_count, (
            f"vertex-color contract failed for {mesh_name}: source={len(rgb)}, "
            f"MuJoCo={compiled_vertex_count}, USD={usd_vertex_count}"
        )
        if alpha is not None:
            assert len(alpha) == compiled_vertex_count

        author_vertex_display_color(usd_mesh, rgb, alpha)
        connect_display_color_material(usd_mesh, include_opacity=alpha is not None)
