"""Stage 6 (optional): shape-prior mesh alignment.

Wraps ``ar2s.phystwin_visual.align`` (env: ``phystwin_align``).
Requires a user-provided .glb mesh pre-copied to ``<case_dir>/shape/object.glb``
(``process_data.py`` does this copy; the orchestrator must do it too).
Writes ``<case_dir>/shape/matching/final_mesh.glb`` and ``<case_dir>/mesh_transform.npz``.

Note: ``align.py`` may require interactive display (VIS=True + select_point).
This stage is marked optional; headless runs may need further modification.
"""
from dataclasses import dataclass
from pathlib import Path

from ar2s.phystwin_visual._runners import env_for_stage, module_for_stage, run_module_in_env


@dataclass
class ShapeAlignmentInput:
    base_path: str
    case_name: str
    controller_name: str = "hand"


@dataclass
class ShapeAlignmentReport:
    success: bool
    error: str = ""
    aligned_mesh_path: str = ""        # <case_dir>/shape/matching/final_mesh.glb
    mesh_transform_path: str = ""      # <case_dir>/mesh_transform.npz


def run(inp: ShapeAlignmentInput, *, log_dir: str) -> ShapeAlignmentReport:
    case_path = Path(inp.base_path) / inp.case_name
    shape_mesh = case_path / "shape" / "object.glb"
    if not shape_mesh.exists():
        return ShapeAlignmentReport(
            success=False,
            error=f"shape/object.glb not found: {shape_mesh} — copy the user-provided mesh there first",
        )

    args = [
        "--base_path",       inp.base_path,
        "--case_name",       inp.case_name,
        "--controller_name", inp.controller_name,
    ]

    result = run_module_in_env(
        env=env_for_stage("shape_alignment"),
        module=module_for_stage("shape_alignment"),
        args=args,
        log_dir=log_dir,
        stage="shape_alignment",
    )

    if result.returncode != 0:
        return ShapeAlignmentReport(
            success=False,
            error=f"shape_alignment failed (rc={result.returncode}); see {result.stderr_path}\n{result.stderr_tail}",
        )

    aligned_mesh = case_path / "shape" / "matching" / "final_mesh.glb"
    mesh_transform = case_path / "mesh_transform.npz"

    missing = [str(p) for p in [aligned_mesh, mesh_transform] if not p.exists()]
    if missing:
        return ShapeAlignmentReport(success=False, error=f"shape_alignment ran but outputs missing: {missing}")

    return ShapeAlignmentReport(
        success=True,
        aligned_mesh_path=str(aligned_mesh),
        mesh_transform_path=str(mesh_transform),
    )
