"""Stage 7: FoundationPose tracking per object.

Wraps ``run_demo_foundationstereo_video.py`` (env: ``qianjun_foundationpose``).

The script writes per-frame 4x4 pose matrices as ``ob_in_cam/<id>.txt``
plus tracking overlay PNGs under ``track_vis/``. ``ObjectInput.pose_source_path``
in the downstream sysid pipeline expects a pose source readable by
``ar2s.droid_sim.pose_store``, so this stage packs the per-frame ``.txt`` into
a single ``poses.npz`` in-process after the subprocess returns.

Output layout::

    <output_dir>/<object>/ob_in_cam/<id>.txt        — written by the script
    <output_dir>/<object>/track_vis/<id>.png        — written by the script
    <output_dir>/<object>/poses.npz                 — packed here (all frames)
"""
from dataclasses import dataclass
from pathlib import Path

import re

import numpy as np

from ar2s.agents_visual._toolkit._runners import env_for_stage, run_in_env, script_path
from ar2s.droid_sim import pose_store


_FRAME_TXT_RE = re.compile(r"frame_(\d+)\.txt$")


@dataclass
class PoseTrackingInput:
    sequence_dir: str                       # FoundationStereo output (K.txt + frames + depth + scale/)
    mask_root: str                          # SAM3 output root (parent of <obj>/masks/)
    mesh_file: str                          # .glb produced in stage 5
    object_name: str
    output_dir: str                         # script's --debug_dir; per-object subdir is created under this
    use_icp_init_scale: bool = True
    icp_scale_num_frames: int = 20
    init_frame: int = -1                    # -1 = auto-pick first frame with a valid mask
    end_frame: int = -1                     # last frame index to track (inclusive). -1 = to end of sequence.
    shorter_side: int | None = None         # downscale frames so min(H,W) == this before registration.
                                            # FoundationPose warps every pose hypothesis at full image
                                            # resolution, so register() memory scales with pixel count:
                                            # a 2208x1242 sensor needs ~3x a 1280x720 one and OOMs a 32 GB
                                            # card. The script rescales K with the image, so geometry is
                                            # unchanged. None = native resolution.


@dataclass
class PoseTrackingReport:
    success: bool
    error: str = ""
    object_name: str = ""
    pose_dir: str = ""                      # <output_dir>/<obj>/poses.npz — feeds ObjectInput.pose_source_path
    track_vis_dir: str = ""                 # <output_dir>/<obj>/track_vis
    first_frame_rgb_path: str = ""          # first available track_vis PNG (init_frame may be > 0)


def run(inp: PoseTrackingInput, *, log_dir: str) -> PoseTrackingReport:
    seq = Path(inp.sequence_dir)
    if not (seq / "K.txt").exists():
        return PoseTrackingReport(
            success=False, object_name=inp.object_name,
            error=f"K.txt missing in {seq}",
        )
    if not (seq / "depth").exists():
        return PoseTrackingReport(
            success=False, object_name=inp.object_name,
            error=f"depth/ missing in {seq}",
        )
    if not Path(inp.mesh_file).exists():
        return PoseTrackingReport(
            success=False, object_name=inp.object_name,
            error=f"mesh_file not found: {inp.mesh_file}",
        )

    out_dir = Path(inp.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "--object_name",         inp.object_name,
        "--sequence_dir",        inp.sequence_dir,
        "--mask_root",           inp.mask_root,
        "--mesh_file",           inp.mesh_file,
        "--use_icp_init_scale",  "1" if inp.use_icp_init_scale else "0",
        "--icp_scale_num_frames", str(inp.icp_scale_num_frames),
        "--init_frame",          str(inp.init_frame),
        "--end_frame",           str(inp.end_frame),
        "--debug",               "1",       # writes track_vis overlays
        "--debug_dir",           str(out_dir),
    ]
    if inp.shorter_side:
        args += ["--shorter_side", str(inp.shorter_side)]

    # Per-object stage name so log files don't clobber across the loop.
    stage = f"pose_tracking_{inp.object_name}"

    result = run_in_env(
        env=env_for_stage("pose_tracking"),
        script_path=script_path("pose_tracking"),
        args=args,
        log_dir=log_dir,
        stage=stage,
    )

    if result.returncode != 0:
        return PoseTrackingReport(
            success=False, object_name=inp.object_name,
            error=(
                f"pose_tracking failed (rc={result.returncode}); see {result.stderr_path}\n"
                f"{result.stderr_tail}"
            ),
        )

    return _collect_outputs(inp, out_dir)


def _collect_outputs(inp: PoseTrackingInput, out_dir: Path) -> PoseTrackingReport:
    obj_dir       = out_dir / inp.object_name
    txt_dir       = obj_dir / "ob_in_cam"
    track_vis_dir = obj_dir / "track_vis"
    poses_path    = obj_dir / "poses.npz"

    txt_files = sorted(txt_dir.glob("frame_*.txt"))
    if not txt_files:
        return PoseTrackingReport(
            success=False, object_name=inp.object_name,
            error=f"no pose .txt files written to {txt_dir}",
        )

    # .txt -> one packed poses.npz. One file per tracked frame per object is
    # the single biggest inode cost in a run dir, and nothing downstream reads
    # the per-frame form directly — ar2s.droid_sim.pose_store abstracts both.
    poses: dict[int, np.ndarray] = {}
    for tp in txt_files:
        pose = np.loadtxt(tp)
        if pose.shape != (4, 4):
            return PoseTrackingReport(
                success=False, object_name=inp.object_name,
                error=f"unexpected pose shape {pose.shape} at {tp}",
            )
        m = _FRAME_TXT_RE.search(tp.name)
        if m is None:
            return PoseTrackingReport(
                success=False, object_name=inp.object_name,
                error=f"cannot parse frame index from {tp.name}",
            )
        poses[int(m.group(1))] = pose
    pose_store.save_poses(poses_path, poses)

    vis_files = sorted(track_vis_dir.glob("*.png"))
    first_rgb = str(vis_files[0]) if vis_files else ""

    return PoseTrackingReport(
        success=True,
        object_name=inp.object_name,
        pose_dir=str(poses_path),
        track_vis_dir=str(track_vis_dir),
        first_frame_rgb_path=first_rgb,
    )
