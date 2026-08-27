"""Stage 5: 3D track filtering and final track data packaging.

Wraps ``ar2s.phystwin_visual.data_process_track`` (env: ``phystwin_proc``).
Reads cotracker/, pcd/, mask/processed_masks.pkl.
Writes ``<case_dir>/track_process_data.pkl``.
"""
from dataclasses import dataclass
from pathlib import Path

from ar2s.phystwin_visual._runners import env_for_stage, module_for_stage, run_module_in_env


@dataclass
class TrackProcessingInput:
    base_path: str
    case_name: str


@dataclass
class TrackProcessingReport:
    success: bool
    error: str = ""
    track_data_path: str = ""


def run(inp: TrackProcessingInput, *, log_dir: str) -> TrackProcessingReport:
    case_path = Path(inp.base_path) / inp.case_name
    processed_masks = case_path / "mask" / "processed_masks.pkl"
    if not processed_masks.exists():
        return TrackProcessingReport(success=False, error=f"processed_masks.pkl not found: {processed_masks}")

    args = ["--base_path", inp.base_path, "--case_name", inp.case_name]

    result = run_module_in_env(
        env=env_for_stage("track_processing"),
        module=module_for_stage("track_processing"),
        args=args,
        log_dir=log_dir,
        stage="track_processing",
    )

    if result.returncode != 0:
        return TrackProcessingReport(
            success=False,
            error=f"track_processing failed (rc={result.returncode}); see {result.stderr_path}\n{result.stderr_tail}",
        )

    out_path = case_path / "track_process_data.pkl"
    if not out_path.exists():
        return TrackProcessingReport(success=False, error=f"track_processing ran but track_process_data.pkl not found: {out_path}")

    return TrackProcessingReport(success=True, track_data_path=str(out_path))
