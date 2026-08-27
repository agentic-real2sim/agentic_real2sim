"""Stage 7: final data export — volume-sampled track data + train/test split.

Wraps ``ar2s.phystwin_visual.data_process_sample`` (env: ``phystwin_proc``).
Reads track_process_data.pkl (+ shape/matching/final_mesh.glb when shape_prior=True).
Writes ``<case_dir>/final_data.pkl``, ``<case_dir>/final_data.mp4`` (preview),
``<case_dir>/final_pcd.mp4`` (pcd preview), and ``<case_dir>/split.json``
via a second step in the orchestrator node.

Note: split.json is written by the orchestrator itself (not by data_process_sample.py)
to match ``process_data.py`` behavior. See ``ar2s.phystwin_visual.orchestrator._write_split_json``.
"""
from dataclasses import dataclass
from pathlib import Path

from ar2s.phystwin_visual._runners import env_for_stage, module_for_stage, run_module_in_env


@dataclass
class FinalExportInput:
    base_path: str
    case_name: str
    shape_prior: bool = False
    num_surface_points: int = 1024
    volume_sample_size: float = 0.005


@dataclass
class FinalExportReport:
    success: bool
    error: str = ""
    final_data_path: str = ""     # <case_dir>/final_data.pkl
    preview_mp4: str = ""         # <case_dir>/final_data.mp4 (may be empty)
    final_pcd_mp4: str = ""       # <case_dir>/final_pcd.mp4 (may be empty)


def run(inp: FinalExportInput, *, log_dir: str) -> FinalExportReport:
    case_path = Path(inp.base_path) / inp.case_name
    track_data = case_path / "track_process_data.pkl"
    if not track_data.exists():
        return FinalExportReport(success=False, error=f"track_process_data.pkl not found: {track_data}")

    if inp.shape_prior:
        aligned_mesh = case_path / "shape" / "matching" / "final_mesh.glb"
        if not aligned_mesh.exists():
            return FinalExportReport(
                success=False,
                error=f"shape_prior=True but final_mesh.glb not found: {aligned_mesh}",
            )

    args = [
        "--base_path",          inp.base_path,
        "--case_name",          inp.case_name,
        "--num_surface_points", str(inp.num_surface_points),
        "--volume_sample_size", str(inp.volume_sample_size),
    ]
    if inp.shape_prior:
        args.append("--shape_prior")

    result = run_module_in_env(
        env=env_for_stage("final_export"),
        module=module_for_stage("final_export"),
        args=args,
        log_dir=log_dir,
        stage="final_export",
    )

    if result.returncode != 0:
        return FinalExportReport(
            success=False,
            error=f"final_export failed (rc={result.returncode}); see {result.stderr_path}\n{result.stderr_tail}",
        )

    final_data = case_path / "final_data.pkl"
    if not final_data.exists():
        return FinalExportReport(success=False, error=f"final_export ran but final_data.pkl not found: {final_data}")

    preview = case_path / "final_data.mp4"
    pcd_preview = case_path / "final_pcd.mp4"
    return FinalExportReport(
        success=True,
        final_data_path=str(final_data),
        preview_mp4=str(preview) if preview.exists() else "",
        final_pcd_mp4=str(pcd_preview) if pcd_preview.exists() else "",
    )
