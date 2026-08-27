"""Stage 4: mask cleanup — depth + semantic + outlier filtering.

Wraps ``ar2s.phystwin_visual.data_process_mask`` (env: ``phystwin_proc``).
Reads pcd/ and mask/ dirs. Writes ``<case_dir>/mask/processed_masks.pkl``.
"""
from dataclasses import dataclass
from pathlib import Path

from ar2s.phystwin_visual._runners import env_for_stage, module_for_stage, run_module_in_env


@dataclass
class MaskCleanupInput:
    base_path: str
    case_name: str
    controller_name: str = "hand"


@dataclass
class MaskCleanupReport:
    success: bool
    error: str = ""
    processed_masks_path: str = ""


def run(inp: MaskCleanupInput, *, log_dir: str) -> MaskCleanupReport:
    case_path = Path(inp.base_path) / inp.case_name
    for required in [case_path / "pcd", case_path / "mask"]:
        if not required.exists():
            return MaskCleanupReport(success=False, error=f"Required dir not found: {required}")

    args = [
        "--base_path",      inp.base_path,
        "--case_name",      inp.case_name,
        "--controller_name", inp.controller_name,
    ]

    result = run_module_in_env(
        env=env_for_stage("mask_cleanup"),
        module=module_for_stage("mask_cleanup"),
        args=args,
        log_dir=log_dir,
        stage="mask_cleanup",
    )

    if result.returncode != 0:
        return MaskCleanupReport(
            success=False,
            error=f"mask_cleanup failed (rc={result.returncode}); see {result.stderr_path}\n{result.stderr_tail}",
        )

    out_path = case_path / "mask" / "processed_masks.pkl"
    if not out_path.exists():
        return MaskCleanupReport(success=False, error=f"mask_cleanup ran but processed_masks.pkl not found: {out_path}")

    return MaskCleanupReport(success=True, processed_masks_path=str(out_path))
