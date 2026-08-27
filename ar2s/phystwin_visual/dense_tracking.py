"""Stage 2: dense 2D tracking with CoTracker.

Wraps ``ar2s.phystwin_visual.dense_track`` (env: ``phystwin_track``).

Outputs under ``<base_path>/<case_name>/cotracker/``:
    <camera_idx>.npz    — tracks (T, N, 2) and visibility (T, N, 1) arrays
    <camera_idx>.mp4    — visualization video
"""
from dataclasses import dataclass
from pathlib import Path

from ar2s.phystwin_visual._runners import env_for_stage, module_for_stage, run_module_in_env


@dataclass
class DenseTrackingInput:
    base_path: str
    case_name: str
    max_query_points: int = 5000


@dataclass
class DenseTrackingReport:
    success: bool
    error: str = ""
    cotracker_dir: str = ""
    num_cameras: int = 0


def run(inp: DenseTrackingInput, *, log_dir: str) -> DenseTrackingReport:
    case_path = Path(inp.base_path) / inp.case_name
    mask_dir = case_path / "mask"
    if not mask_dir.exists():
        return DenseTrackingReport(success=False, error=f"mask/ dir not found: {mask_dir} — run video_segmentation first")

    args = [
        "--base_path",        inp.base_path,
        "--case_name",        inp.case_name,
        "--max_query_points", str(inp.max_query_points),
    ]

    result = run_module_in_env(
        env=env_for_stage("dense_tracking"),
        module=module_for_stage("dense_tracking"),
        args=args,
        log_dir=log_dir,
        stage="dense_tracking",
    )

    if result.returncode != 0:
        return DenseTrackingReport(
            success=False,
            error=f"dense_tracking failed (rc={result.returncode}); see {result.stderr_path}\n{result.stderr_tail}",
        )

    cotracker_dir = case_path / "cotracker"
    npz_files = list(cotracker_dir.glob("*.npz")) if cotracker_dir.exists() else []
    if not npz_files:
        return DenseTrackingReport(success=False, error=f"dense_tracking ran but no .npz files in {cotracker_dir}")

    return DenseTrackingReport(
        success=True,
        cotracker_dir=str(cotracker_dir),
        num_cameras=len(npz_files),
    )
