"""Stage 3: RGB-D lifting to world-space point clouds.

Wraps ``ar2s.phystwin_visual.data_process_pcd`` (env: ``phystwin_proc``).

Reads ``<case_dir>/metadata.json`` and ``<case_dir>/calibrate.pkl``.
Outputs ``<case_dir>/pcd/<frame_idx>.npz`` (points, colors, masks arrays).
"""
from dataclasses import dataclass
from pathlib import Path

from ar2s.phystwin_visual._runners import env_for_stage, module_for_stage, run_module_in_env


@dataclass
class PcdProjectionInput:
    base_path: str
    case_name: str


@dataclass
class PcdProjectionReport:
    success: bool
    error: str = ""
    pcd_dir: str = ""
    num_frames: int = 0


def run(inp: PcdProjectionInput, *, log_dir: str) -> PcdProjectionReport:
    case_path = Path(inp.base_path) / inp.case_name
    for required in ["metadata.json", "calibrate.pkl"]:
        p = case_path / required
        if not p.exists():
            return PcdProjectionReport(success=False, error=f"{required} not found: {p}")

    args = ["--base_path", inp.base_path, "--case_name", inp.case_name]

    result = run_module_in_env(
        env=env_for_stage("pcd_projection"),
        module=module_for_stage("pcd_projection"),
        args=args,
        log_dir=log_dir,
        stage="pcd_projection",
    )

    if result.returncode != 0:
        return PcdProjectionReport(
            success=False,
            error=f"pcd_projection failed (rc={result.returncode}); see {result.stderr_path}\n{result.stderr_tail}",
        )

    pcd_dir = case_path / "pcd"
    npz_files = sorted(pcd_dir.glob("*.npz")) if pcd_dir.exists() else []
    if not npz_files:
        return PcdProjectionReport(success=False, error=f"pcd_projection ran but no .npz files in {pcd_dir}")

    return PcdProjectionReport(success=True, pcd_dir=str(pcd_dir), num_frames=len(npz_files))
