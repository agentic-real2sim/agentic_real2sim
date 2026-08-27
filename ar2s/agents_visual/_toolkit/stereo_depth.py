"""Stage 2: rectified frame pairs -> depth + pointclouds.

Wraps ``run_stereo_on_frames.py`` (env: ``qianjun_foundation_stereo``).

Input must contain the layout produced by stage 1::

    <sequence_dir>/K.txt
    <sequence_dir>/frames/left/frame_*.png
    <sequence_dir>/frames/right/frame_*.png

Writes back into the same directory::

    <sequence_dir>/depth/frame_NNNNNN_depth.npy   (H, W) float32 meters
    <sequence_dir>/vis/frame_NNNNNN_vis.png       disparity visualisation
    <sequence_dir>/pointcloud/frame_NNNNNN.ply    (when get_pointcloud=True)
"""
from dataclasses import dataclass
from pathlib import Path

from ar2s.agents_visual._toolkit._runners import MODELS_ROOT, env_for_stage, run_in_env, script_path


@dataclass
class StereoDepthInput:
    sequence_dir: str                       # contains K.txt + frames/left + frames/right
    max_frames: int | None = None
    frame_step: int = 1
    get_pointcloud: bool = False            # .ply is debug-only (nothing downstream
                                            # reads it) but accounts for the majority
                                            # of per-frame cost.
    valid_iters: int = 22                   # GRU refinement iterations. Dominant GPU
                                            # cost knob (~linear); 22 is near-lossless
                                            # vs FoundationStereo's default of 32
                                            # (measured ~0.16% mean depth error).


@dataclass
class StereoDepthReport:
    success: bool
    error: str = ""
    depth_dir: str = ""                     # <seq>/depth, files frame_XXXXXX_depth.npy
    vis_dir: str = ""                       # <seq>/vis,   files frame_XXXXXX_vis.png
    pointcloud_dir: str = ""                # <seq>/pointcloud (empty when get_pointcloud=False)
    num_frames_processed: int = 0


def run(inp: StereoDepthInput, *, log_dir: str) -> StereoDepthReport:
    seq_dir = Path(inp.sequence_dir)
    if not (seq_dir / "K.txt").exists():
        return StereoDepthReport(success=False, error=f"K.txt missing in {seq_dir}")

    # FoundationStereo's run_stereo_on_frames.py defaults --ckpt_dir to its own
    # third_party/FoundationStereo/pretrained_models/ — route to the
    # consolidated MODELS_ROOT instead. cfg.yaml sits next to the .pth and is
    # picked up by the script via os.path.dirname(ckpt_dir).
    ckpt = MODELS_ROOT / "foundation_stereo" / "23-51-11" / "model_best_bp2.pth"
    args = [
        "--sequence_dir", inp.sequence_dir,
        "--ckpt_dir",     str(ckpt),
        "--get_pc",       "1" if inp.get_pointcloud else "0",
        "--valid_iters",  str(inp.valid_iters),
    ]
    if inp.max_frames is not None:
        args += ["--max_frames", str(inp.max_frames)]
    if inp.frame_step != 1:
        args += ["--frame_step", str(inp.frame_step)]

    result = run_in_env(
        env=env_for_stage("stereo_depth"),
        script_path=script_path("stereo_depth"),
        args=args,
        log_dir=log_dir,
        stage="stereo_depth",
    )

    if result.returncode != 0:
        return StereoDepthReport(
            success=False,
            error=(
                f"stereo_depth failed (rc={result.returncode}); see {result.stderr_path}\n"
                f"{result.stderr_tail}"
            ),
        )

    depth_dir = seq_dir / "depth"
    vis_dir   = seq_dir / "vis"
    pc_dir    = seq_dir / "pointcloud"

    depth_files = sorted(depth_dir.glob("frame_*_depth.npy"))
    if not depth_files:
        return StereoDepthReport(
            success=False,
            error=f"script reported success but no depth files found in {depth_dir}",
        )

    return StereoDepthReport(
        success=True,
        depth_dir=str(depth_dir),
        vis_dir=str(vis_dir),
        pointcloud_dir=str(pc_dir) if pc_dir.exists() else "",
        num_frames_processed=len(depth_files),
    )
