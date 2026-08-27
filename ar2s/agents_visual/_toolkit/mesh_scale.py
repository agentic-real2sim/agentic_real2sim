"""Stage 6: depth + masks + meshes -> per-object scale.

Wraps ``compute_mesh_scale.py`` (env: ``qianjun_foundation_stereo``).

The script writes per-frame ``frame_*_meta.npz`` files containing a uniform
``scale`` float (bbox-diagonal ratio between masked depth point cloud and
mesh surface). We aggregate median / std ourselves from those npz files
rather than parsing the SUMMARY block in stdout — the npz files are the
authoritative source and trivially loadable.

Output:
    <sequence_dir>/scale/<object>/frame_*_meta.npz

The ``ObjectInput.scale_path`` (.npy with (3,)) is *not* produced here; the
agent layer's resolve step broadcasts ``median_scale`` to ``(s, s, s)``.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ar2s.agents_visual._toolkit._runners import env_for_stage, run_in_env, script_path


_FRAME_RE = re.compile(r"frame_(\d+)_meta\.npz$")

# Frames are kept if their scale is within KEYFRAME_THRESH (relative) of the
# keyframe's scale. 0.15 matches scale_critic's "unstable" warning threshold —
# a frame more than 15% off from keyframe scale is treated as noise / partial
# occlusion / depth glitch.
_KEYFRAME_THRESH = 0.15


@dataclass
class MeshScaleInput:
    sequence_dir: str                       # contains K.txt and depth/
    mesh_dir: str                           # contains <object>.glb
    mask_dir: str                           # SAM3 output root (parent of <obj>/masks/)
    objects: list[str]
    # Phase-B per-object keyframe: an object's mask may live anywhere in the
    # demo (e.g. tri_v0 marker initially inside the mug, mask only appears at
    # frame ~40). The legacy 20-frame ceiling silently dropped such objects
    # (num_frames_used=0 because frames 0-19 had only zero-masks). Scan a
    # large window so any reasonable per-object keyframe is reachable; the
    # script skips zero-mask frames cheaply so the overhead is small.
    num_frames: int = 500
    # Per-object keyframe index (sanitized object name → frame idx). The
    # aggregator filters scales to those within ±KEYFRAME_THRESH of the
    # keyframe's scale; the keyframe is the highest-quality view by design
    # (keyframe_select picked it for clarity), so it's a more trustworthy
    # anchor than a median over all frames (which can drift if 50%+ of frames
    # have partial visibility). When empty / keyframe scale unavailable on
    # disk, falls back to the legacy median-everything behaviour.
    keyframe_by_object: dict[str, int] = field(default_factory=dict)


@dataclass
class ObjectScaleInfo:
    object_name: str
    meta_dir: str                           # <sequence_dir>/scale/<obj>
    median_scale: float = 0.0
    std_scale: float = 0.0
    num_frames_used: int = 0                # frames AFTER keyframe-anchored filter
    total_frames: int = 0                   # frames BEFORE filter (raw on disk)
    keyframe_anchored: bool = False         # True iff keyframe scale was available
                                             # and used as the agreement anchor


@dataclass
class MeshScaleReport:
    success: bool
    error: str = ""
    objects: list[ObjectScaleInfo] = field(default_factory=list)


def run(inp: MeshScaleInput, *, log_dir: str) -> MeshScaleReport:
    if not inp.objects:
        return MeshScaleReport(success=False, error="objects is empty")

    seq = Path(inp.sequence_dir)
    if not (seq / "K.txt").exists():
        return MeshScaleReport(success=False, error=f"K.txt missing in {seq}")
    if not (seq / "depth").exists():
        return MeshScaleReport(success=False, error=f"depth/ missing in {seq}")

    args = [
        "--data_dir",   inp.sequence_dir,
        "--mesh_dir",   inp.mesh_dir,
        "--mask_dir",   inp.mask_dir,
        "--num_frames", str(inp.num_frames),
        "--objects",    *inp.objects,        # nargs="+" — keep last
    ]

    result = run_in_env(
        env=env_for_stage("mesh_scale"),
        script_path=script_path("mesh_scale"),
        args=args,
        log_dir=log_dir,
        stage="mesh_scale",
    )

    if result.returncode != 0:
        return MeshScaleReport(
            success=False,
            error=(
                f"mesh_scale failed (rc={result.returncode}); see {result.stderr_path}\n"
                f"{result.stderr_tail}"
            ),
        )

    return _collect_outputs(inp, seq / "scale")


def _collect_outputs(inp: MeshScaleInput, scale_root: Path) -> MeshScaleReport:
    """Aggregate per-frame scales with keyframe-anchored filtering.

    Algorithm per object:
      1. Load all {frame_idx: scale} pairs the script wrote on disk.
      2. Look up s_keyframe = scales[keyframe_idx] (per-object keyframe from
         keyframe_select, passed in via inp.keyframe_by_object).
      3. If s_keyframe is available, keep only scales s where
            |s - s_keyframe| / s_keyframe < _KEYFRAME_THRESH (0.15)
         The keyframe is always in the kept set (|s_kf - s_kf| = 0).
      4. If no keyframe scale (no entry in keyframe_by_object, or keyframe's
         meta.npz wasn't written — e.g. depth glitch on that exact frame),
         fall back to median over all scales (legacy behaviour).
      5. median_scale + std_scale computed on the kept set.

    Reports both num_frames_used (post-filter) and total_frames (pre-filter)
    so scale_critic can warn when a high fraction of frames disagreed with
    the keyframe (suggests keyframe itself may be suspect, or data is very
    noisy).
    """
    objects: list[ObjectScaleInfo] = []
    for obj in inp.objects:
        obj_dir = scale_root / obj
        if not obj_dir.exists():
            objects.append(ObjectScaleInfo(object_name=obj, meta_dir=""))
            continue

        # Load (frame_idx, scale) pairs
        frame_scales: dict[int, float] = {}
        for npz_path in sorted(obj_dir.glob("frame_*_meta.npz")):
            m = _FRAME_RE.search(npz_path.name)
            if not m:
                continue
            with np.load(npz_path) as d:
                frame_scales[int(m.group(1))] = float(d["scale"])

        if not frame_scales:
            objects.append(ObjectScaleInfo(object_name=obj, meta_dir=str(obj_dir)))
            continue

        total = len(frame_scales)
        keyframe_idx = inp.keyframe_by_object.get(obj)
        s_keyframe = frame_scales.get(keyframe_idx) if keyframe_idx is not None else None

        if s_keyframe is not None and s_keyframe > 0:
            # Keyframe-anchored filter: keep frames whose scale agrees with keyframe.
            kept = [
                s for s in frame_scales.values()
                if abs(s - s_keyframe) / s_keyframe < _KEYFRAME_THRESH
            ]
            keyframe_anchored = True
        else:
            # No keyframe info or keyframe scale unavailable on disk —
            # fall back to median over all frames.
            kept = list(frame_scales.values())
            keyframe_anchored = False

        arr = np.array(kept)
        objects.append(ObjectScaleInfo(
            object_name=obj,
            meta_dir=str(obj_dir),
            median_scale=float(np.median(arr)),
            std_scale=float(arr.std()),
            num_frames_used=len(kept),
            total_frames=total,
            keyframe_anchored=keyframe_anchored,
        ))

    return MeshScaleReport(success=True, objects=objects)
