"""Stage 1: multi-camera video segmentation using Grounded-SAM-2.

Wraps ``ar2s.phystwin_visual.segment`` (env: ``phystwin_seg``).

Outputs under ``<base_path>/<case_name>/mask/``:
    mask_info_<camera_idx>.json    — object-id-to-label mapping per camera
    <camera_idx>/<obj_id>/<frame>.png — per-frame binary masks
"""
from dataclasses import dataclass, field
from pathlib import Path

from ar2s.phystwin_visual._runners import env_for_stage, module_for_stage, run_module_in_env


@dataclass
class VideoSegmentationInput:
    base_path: str              # absolute path
    case_name: str
    text_prompt: str            # e.g. "cup.hand"
    camera_indices: list[int] = field(default_factory=list)   # empty = auto-detect
    box_threshold: float = 0.35
    text_threshold: float = 0.25


@dataclass
class VideoSegmentationReport:
    success: bool
    error: str = ""
    case_dir: str = ""
    camera_indices: list[int] = field(default_factory=list)
    mask_dir: str = ""           # <base_path>/<case_name>/mask
    mask_info_paths: dict[int, str] = field(default_factory=dict)   # camera_idx -> mask_info_<idx>.json path


def run(inp: VideoSegmentationInput, *, log_dir: str) -> VideoSegmentationReport:
    case_path = Path(inp.base_path) / inp.case_name
    if not case_path.exists():
        return VideoSegmentationReport(success=False, error=f"case_dir not found: {case_path}")

    depth_dir = case_path / "depth"
    if not depth_dir.exists():
        return VideoSegmentationReport(success=False, error=f"depth/ dir not found: {depth_dir}")

    # Resolve camera indices
    camera_indices = inp.camera_indices
    if not camera_indices:
        camera_indices = sorted(int(p.name) for p in depth_dir.iterdir() if p.is_dir())
    assert camera_indices, f"No camera subdirectories found in {depth_dir}"

    args = [
        "--base_path",    inp.base_path,
        "--case_name",    inp.case_name,
        "--TEXT_PROMPT",  inp.text_prompt,
        "--box_threshold", str(inp.box_threshold),
        "--text_threshold", str(inp.text_threshold),
        "--camera_indices", *[str(i) for i in camera_indices],
    ]

    result = run_module_in_env(
        env=env_for_stage("video_segmentation"),
        module=module_for_stage("video_segmentation"),
        args=args,
        log_dir=log_dir,
        stage="video_segmentation",
    )

    if result.returncode != 0:
        return VideoSegmentationReport(
            success=False,
            error=(
                f"video_segmentation failed (rc={result.returncode}); "
                f"see {result.stderr_path}\n{result.stderr_tail}"
            ),
        )

    mask_dir = case_path / "mask"
    mask_info_paths: dict[int, str] = {}
    for idx in camera_indices:
        p = mask_dir / f"mask_info_{idx}.json"
        if p.exists():
            mask_info_paths[idx] = str(p)

    if not mask_info_paths:
        return VideoSegmentationReport(
            success=False,
            error=f"video_segmentation ran but no mask_info_*.json found in {mask_dir}",
        )

    return VideoSegmentationReport(
        success=True,
        case_dir=str(case_path),
        camera_indices=camera_indices,
        mask_dir=str(mask_dir),
        mask_info_paths=mask_info_paths,
    )
