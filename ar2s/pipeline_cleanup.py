"""Post-run consolidation of an ``outputs/run_pipeline/<run_id>/`` directory.

A finished run leaves ~2400 files / ~1.7 GiB, most of which is either
regeneratable from an artifact we keep, a byte-identical copy of one, or
scratch that no stage ever reads again. HPC filesystems cap inodes as well
as bytes, so both matter.

What this does, per area (each step is independent and never raises — a
cleanup failure must not fail an otherwise-successful run):

  artifacts/sweep-*/   pack samples/ into samples.tar.zst (zstd --long dedups
                       the ~40 near-identical USDs ~11x), EXCEPT the grasp
                       videos for eligible Blender handoff candidates plus
                       best_sample_idx — those stay loose, so the handoff
                       collector can stage them without opening the archive.
                       The two sets are disjoint, so nothing is stored twice.
                       best_result.usd is promoted to the sweep root as a real
                       copy (hand inspection); best_grasp_video.mp4 becomes a
                       symlink into the surviving samples/<best>/.
                       samples_kept.json records what stayed loose so a re-run
                       does not re-tar the survivors. Per-sample scalars
                       already live in summary.yaml, and readers that need the
                       full tree get the union from ``open_samples``.
  seq/<serial>/        depth/*.npy -> depth.npz (uint16 millimetres);
                       drop vis/ (written by FoundationStereo, read by
                       nothing), frames/{left,right} (rebuildable from
                       rectified_left.mp4 / raw_episodes stereo.mp4), and
                       collapse scale/<obj>/frame_*_meta.npz into scale.npz.
  poses/<obj>/         track_vis/*.png -> track_vis.mp4; drop ob_in_cam/*.txt
                       (superseded by the packed pose source, the downstream
                       contract) and FoundationPose's _mesh_exports/ scratch.
  segmentation/        drop every masks/ dir, keeping the overlay mp4s.
  sysid_inputs/        drop the render scratch that lands in the deliverable
                       bundle — orient_tmp/mesh_*.obj (mesh_orient) and
                       axis_align_tmp/mesh_*.obj (mesh_axis_align, ~900 MiB
                       per object) — keeping each stage's VLM panel PNG,
                       hoisted up one level with geometry_priors.json
                       repointed at it. Also the duplicated track_vis/ copy
                       and the visual_pre_*.obj backups the USD colour path
                       no longer needs.
  raw_episodes/        hoist a legacy ``<episode_id>/`` level up to the flat
                       layout build_raw_episode now writes, repointing
                       state.json and the visual_<id>.py stub.

One resume constraint keeps partial runs usable, reported as SKIP lines
rather than silently applied: ``material_classify`` (physical_prior, which
runs AFTER visual finalize) reads ``segmentation/<obj>/masks/`` and
``seq/*/frames/left``, so those survive until the bundle carries
``physical_priors.json``. That is what lets a ``--until visual_processing``
run stay resumable by a later ``--start-from geometry_prior``.

The geometry_prior stages need no such gate: mesh_axis_align guards its
re-runs on ``foundation_pose_pre_align/``, which cleanup never removes. So
``visual_pre_*.obj`` goes as soon as its one remaining reader — usd_exporter's
legacy vertex-color fallback — is provably satisfied.

``scene_view_repair`` rebuilds the scene from the secondary view into a full
sibling run root, ``<run_root>__v2``, so cleaning ``run_root`` cleans that
sibling too (reported under a ``__v2/`` prefix). Same policy, mask/frame gate
rebound: the branch is a repair RESOURCE that stops at apply_geometry_priors
and never reaches physical_prior, so ``geometry_priors.json`` marks it
finished. Keying on that marker rather than the ``__v2`` name keeps a
half-built branch resumable — ``run_repair`` re-enters one through
``run_visual``'s state.json.

Dry-run reports the same plan without touching anything.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ar2s.pipeline_artifacts import (
    episode_bundle_dir,
    raw_episodes_is_flat,
    sample_dir_name,
    visual_input_stub,
)
from ar2s.grasp_candidates import expected_candidate_indices


# uint16 millimetres. Max representable 65.535 m; measured worst-case error on
# a 720p FoundationStereo stack is 0.50 mm absolute / 0.29% relative, which is
# well inside what mesh_scale (median over 20 frames of a point-cloud bbox
# ratio) and FoundationPose ICP resolve.
DEPTH_SCALE_M = 0.001
DEPTH_MAX_M = 65.535

USD_ARCHIVE_NAME = "samples.tar.zst"
# Marker naming the sample files cleanup deliberately left loose. Its presence
# also means "this sweep is done" -- a re-run must not re-tar the survivors.
SAMPLES_KEPT_MARKER = "samples_kept.json"
# The per-sample file needed by the Blender handoff collector. Kept loose for
# eligible candidates (see _kept_sample_indices) so the handoff can stage
# videos without opening the archive.
KEPT_SAMPLE_FILES = ("grasp_video.mp4",)
ZSTD_LEVEL = 10
# 27 => a 128 MiB window, which is exactly zstd's default decoder limit, so
# `zstd -d` reads the archive with no extra flags. Going to 31 (2 GiB) buys
# ~10% (67.7 vs 74.6 MB on a 818 MB / 40-sample sweep) but makes every reader
# pass --long=31 or hit "Frame requires too much memory for decoding". Not
# worth it for an archive whose whole point is to still be openable later.
ZSTD_LONG_WINDOW_LOG = 27

_TRACK_VIS_FPS = 10.0

# Mirrors scene_view_repair._V2_SUFFIX; duplicated rather than imported
# because that module pulls in PIL / pydantic / the VLM stack.
_V2_SUFFIX = "__v2"


@dataclass
class CleanupStats:
    files_removed: int = 0
    bytes_removed: int = 0
    files_written: int = 0
    bytes_written: int = 0
    actions: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def bytes_saved(self) -> int:
        return self.bytes_removed - self.bytes_written


def human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PiB"


# ---------------------------------------------------------------------------
# Filesystem primitives — every mutation funnels through these so --dry-run
# needs no separate code path.
# ---------------------------------------------------------------------------

class _Fs:
    def __init__(self, dry_run: bool, stats: CleanupStats) -> None:
        self.dry_run = dry_run
        self.stats = stats

    def remove_files(self, paths: list[Path]) -> None:
        for p in paths:
            try:
                size = p.stat().st_size
            except OSError:
                continue
            self.stats.files_removed += 1
            self.stats.bytes_removed += size
            if not self.dry_run:
                try:
                    p.unlink()
                except OSError as e:
                    self.stats.errors.append(f"unlink {p}: {e}")

    def remove_tree(self, root: Path) -> None:
        if not root.is_dir():
            return
        self.remove_files([p for p in root.rglob("*") if p.is_file()])
        if not self.dry_run:
            shutil.rmtree(root, ignore_errors=True)

    def prune_empty(self, root: Path) -> None:
        """Remove now-empty directories under ``root``, deepest first."""
        if self.dry_run or not root.is_dir():
            return
        for d in sorted(
            (p for p in root.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts), reverse=True,
        ):
            try:
                d.rmdir()
            except OSError:
                pass

    def record_written(self, path: Path) -> None:
        if self.dry_run:
            return
        try:
            self.stats.files_written += 1
            self.stats.bytes_written += path.stat().st_size
        except OSError:
            pass


def _files(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.glob(pattern) if p.is_file())


def _load_state(run_root: Path) -> dict:
    path = run_root / "state.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _stage_ok(state: dict, name: str) -> bool:
    return bool((state.get("stages") or {}).get(name, {}).get("ok"))


def _bundle_dirs(run_root: Path) -> list[Path]:
    """The run's emitted bundle, as a list so callers stay loop-shaped.

    Empty when nothing has been emitted yet — an un-emitted ``sysid_inputs/``
    has no manifest, and every caller here is gated on the bundle being real.
    """
    bundle = episode_bundle_dir(run_root)
    return [bundle] if (bundle / "manifest.yaml").is_file() else []


def _is_v2_branch(run_root: Path) -> bool:
    """True for a ``<run_root>__v2`` scene_view_repair secondary-view branch."""
    return run_root.name.endswith(_V2_SUFFIX)


def v2_branch_root(run_root: Path | str) -> Path | None:
    """The scene_view_repair sibling of ``run_root``, if one was built.

    A full run root of its own (seq/, poses/, bundle) that nothing writes to
    once the repair is done, and that nothing else reclaims.
    """
    run_root = Path(run_root)
    if _is_v2_branch(run_root):
        return None
    sibling = run_root.parent / (run_root.name + _V2_SUFFIX)
    return sibling if sibling.is_dir() else None


def _terminal_stage(run_root: Path) -> tuple[str, str]:
    """``(stage name, bundle marker)`` for the run's last mask/frame reader.

    A ``__v2`` branch never runs physical_prior, so waiting for its marker
    there would keep the branch's masks and frames forever.
    """
    if _is_v2_branch(run_root):
        return "geometry_prior", "geometry_priors.json"
    return "physical_prior", "physical_priors.json"


def _terminal_stage_done(run_root: Path) -> bool:
    """True once the last stage reading segmentation masks + seq frames has
    emitted its audit trace for every bundle, so deleting them is safe."""
    bundles = _bundle_dirs(run_root)
    marker = _terminal_stage(run_root)[1]
    return bool(bundles) and all((b / marker).is_file() for b in bundles)


# ---------------------------------------------------------------------------
# artifacts/ — promote the best sample, archive the rest
# ---------------------------------------------------------------------------

def _best_sample(summary: dict) -> dict | None:
    """The sample ``summary.yaml`` itself nominates via ``best_sample_idx``.

    Promotion deliberately does NOT apply its own criterion — grasp_sweep owns
    what "best" means, and a second definition living here would make the
    loose ``best_result.usd`` disagree with the summary next to it.
    """
    idx = summary.get("best_sample_idx")
    if idx is None:
        return None
    for s in summary.get("samples") or []:
        if s.get("sample_idx") is not None and int(s["sample_idx"]) == int(idx):
            return s
    return None


def _kept_sample_indices(summary: dict) -> set[int]:
    """Sample indices whose handoff files stay loose after cleanup.

    The eligible grasp candidates (top-K by ``peak_displacement_mm``), plus
    ``best_sample_idx``. ``grasp_sweep`` now picks its winner on the same
    metric, so it is normally the top candidate -- but it is still unioned in
    because the sweep promotes a fallback winner even when nothing grasped.
    """
    samples = summary.get("samples") or []
    # check_video=False: summary entries carry no video_path, and grasp_sweep
    # keeps an mp4 exactly for the grasped samples anyway.
    kept = {int(i) for i in expected_candidate_indices(samples, check_video=False)}
    best = summary.get("best_sample_idx")
    if best is not None:
        kept.add(int(best))
    return kept


def _kept_sample_files(samples_dir: Path, summary: dict) -> set[Path]:
    """Existing paths under ``samples/`` that must survive as real files."""
    n_samples = summary.get("n_samples") or len(summary.get("samples") or [])
    kept = set()
    for idx in _kept_sample_indices(summary):
        sample_dir = samples_dir / sample_dir_name(idx, n_samples)
        kept.update(p for name in KEPT_SAMPLE_FILES
                    if (p := sample_dir / name).is_file())
    return kept


def _archive_samples(sweep_dir: Path, samples_dir: Path, fs: _Fs,
                     *, exclude: set[Path]) -> bool:
    """tar samples/ through ``zstd --long`` into the sweep root.

    The long window is what makes this worth doing: the per-sample USDs embed
    the same scene geometry and only differ in animation, so a window spanning
    several files dedups across them rather than within each one. Members are
    ordered by (filename, sample) rather than by path so all 40 USDs sit
    contiguously inside that window instead of being interleaved with the
    incompressible mp4s.

    ``exclude`` is the set of files staying loose beside the archive. The two
    sets are disjoint, so nothing is stored twice -- which is also why
    ``open_samples`` has to overlay them to reconstruct the full tree.
    """
    zstd = shutil.which("zstd")
    if zstd is None:
        fs.stats.skipped.append(
            f"{sweep_dir.name}: zstd not on PATH, left samples/ uncompressed"
        )
        return False
    archive = sweep_dir / USD_ARCHIVE_NAME
    if archive.exists():
        fs.stats.skipped.append(f"{archive} already exists")
        return False
    members = sorted(
        (p for p in samples_dir.rglob("*") if p.is_file() and p not in exclude),
        key=lambda p: (p.name, p.parent.name),
    )
    if not members:
        return False
    if fs.dry_run:
        # Report the archived bytes as reclaimed: it is the single largest item
        # cleanup removes, and a dry run that omitted it would read as a no-op.
        fs.remove_files(members)
        return True

    cmd = [zstd, "-q", "-T0", f"-{ZSTD_LEVEL}",
           f"--long={ZSTD_LONG_WINDOW_LOG}", "-o", str(archive)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        with proc.stdin:
            with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
                for m in members:
                    tar.add(m, arcname=str(m.relative_to(sweep_dir)))
    except BaseException:
        proc.kill()
        proc.wait()
        archive.unlink(missing_ok=True)
        raise
    if proc.wait() != 0:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"zstd exited {proc.returncode} for {archive}")
    fs.record_written(archive)
    fs.remove_files(members)
    fs.prune_empty(samples_dir)
    return True


def clean_artifacts(run_root: Path, fs: _Fs) -> None:
    import yaml

    root = run_root / "artifacts"
    if not root.is_dir():
        return
    for sweep_dir in sorted(p for p in root.iterdir()
                            if p.is_dir() and p.name.startswith("sweep-")):
        samples_dir = sweep_dir / "samples"
        summary_path = sweep_dir / "summary.yaml"
        if not samples_dir.is_dir() or not summary_path.is_file():
            continue
        if (sweep_dir / SAMPLES_KEPT_MARKER).is_file():
            fs.stats.skipped.append(
                f"{sweep_dir.name}: already packed ({SAMPLES_KEPT_MARKER} present)"
            )
            continue
        summary = yaml.safe_load(summary_path.read_text()) or {}
        best = _best_sample(summary)
        kept = _kept_sample_files(samples_dir, summary)

        best_video: Path | None = None
        if best is not None:
            idx = int(best["sample_idx"])
            src_dir = next(
                (d for d in samples_dir.iterdir()
                 if d.is_dir() and d.name.isdigit() and int(d.name) == idx),
                None,
            )
            if src_dir is not None:
                # A real copy: nothing reads it, but it gets opened by hand and
                # a file is friendlier for that than a member of an archive.
                src_usd = src_dir / "result.usd"
                if src_usd.is_file():
                    dst = sweep_dir / "best_result.usd"
                    if not fs.dry_run:
                        shutil.copy2(src_usd, dst)
                    fs.record_written(dst)
                # The video stays loose under samples/ (best_sample_idx is in
                # the kept set), so this one is a symlink, not a second copy.
                if (src_dir / "grasp_video.mp4") in kept:
                    best_video = src_dir / "grasp_video.mp4"
                if not fs.dry_run:
                    (sweep_dir / "best_sample.json").write_text(json.dumps({
                        "criterion": "summary.yaml best_sample_idx",
                        "sample_idx": idx,
                        "grasped": bool(best.get("grasped")),
                        "peak_lift_mm": float(best.get("peak_lift_mm") or 0.0),
                        "closest_distance_mm": float(best.get("closest_distance_mm") or 0.0),
                        "archive": USD_ARCHIVE_NAME,
                    }, indent=2) + "\n")
                fs.stats.actions.append(
                    f"artifacts/{sweep_dir.name}: promoted sample {idx:02d} "
                    f"(summary best_sample_idx)"
                )
        else:
            fs.stats.skipped.append(
                f"{sweep_dir.name}: summary.yaml names no best_sample_idx to promote"
            )

        if not _archive_samples(sweep_dir, samples_dir, fs, exclude=kept):
            continue
        if not any(p.is_file() for p in samples_dir.rglob("*")):
            # No eligible handoff sample: leaving an empty samples/ behind would make
            # prepare_staging symlink an empty tree instead of expanding.
            fs.remove_tree(samples_dir)

        if best_video is not None:
            link = sweep_dir / "best_grasp_video.mp4"
            if not fs.dry_run and not link.exists():
                link.symlink_to(best_video.relative_to(sweep_dir))
        if not fs.dry_run:
            (sweep_dir / SAMPLES_KEPT_MARKER).write_text(json.dumps({
                "archive": USD_ARCHIVE_NAME,
                "kept_files": sorted(str(p.relative_to(sweep_dir)) for p in kept),
                "kept_file_names": list(KEPT_SAMPLE_FILES),
                "kept_sample_indices": sorted(_kept_sample_indices(summary)),
                "criterion": "handoff candidates + summary.yaml best_sample_idx",
            }, indent=2) + "\n")
        fs.stats.actions.append(
            f"artifacts/{sweep_dir.name}: samples/ -> {USD_ARCHIVE_NAME} "
            f"({len(kept)} file(s) kept loose)"
        )


@contextlib.contextmanager
def open_samples(sweep_dir: Path | str, *, include: Callable[[str], bool] | None = None):
    """Yield a readable ``samples/`` dir for a sweep, cleaned or not.

    Readers that walk sample dirs (scripts/collect_blender_grasp_assets.py)
    call this instead of hardcoding the path. Three shapes:

      uncleaned          samples/ only            -> the real directory
      cleaned            samples/ + the archive   -> a temp UNION of the two
      cleaned (legacy)   the archive only         -> a temp extraction

    Cleanup keeps the handoff files of eligible samples loose under ``samples/``
    and archives everything else, so on a cleaned sweep NEITHER half is the
    whole tree: the union re-joins them, and callers that need only the loose
    half should read ``samples/`` directly rather than come through here.
    Temp trees are removed on exit; the loose half is symlinked in, never
    copied.

    ``include`` narrows the result to files whose BASENAME it accepts, for
    readers that need a few files per sample and not the ~50 MiB
    ``result.usd`` beside them. It applies to the temp shapes ONLY — an
    uncleaned sweep yields its real directory, unfiltered, because filtering
    there would mean copying. So a caller passing ``include`` must treat the
    yielded tree as "at least these files", never as an exhaustive listing.
    """
    sweep_dir = Path(sweep_dir)
    samples_dir = sweep_dir / "samples"
    archive = sweep_dir / USD_ARCHIVE_NAME
    if samples_dir.is_dir() and not archive.is_file():
        yield samples_dir
        return
    if not archive.is_file():
        yield None
        return
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError(f"zstd not on PATH, cannot read {archive}")
    with tempfile.TemporaryDirectory(prefix="ar2s_samples_") as scratch:
        proc = subprocess.Popen([zstd, "-dc", str(archive)], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                if include is None:
                    tar.extractall(scratch, filter="data")
                else:
                    for member in tar:
                        if member.isfile() and not include(
                            posixpath.basename(member.name)
                        ):
                            continue
                        tar.extract(member, scratch, filter="data")
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.wait() != 0:
                raise RuntimeError(f"zstd exited {proc.returncode} reading {archive}")
        out = Path(scratch) / "samples"
        for src in sorted(samples_dir.rglob("*")) if samples_dir.is_dir() else ():
            if not src.is_file() or (include is not None and not include(src.name)):
                continue
            link = out / src.relative_to(samples_dir)
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(src.absolute())
        yield out


def sweep_usd_archive(sweep_dir: Path | str) -> Path | None:
    """The sweep's ``samples.tar.zst`` if cleanup already made one, else None."""
    archive = Path(sweep_dir) / USD_ARCHIVE_NAME
    return archive if archive.is_file() else None


# ---------------------------------------------------------------------------
# seq/ — depth, scale, and the regeneratable frame dumps
# ---------------------------------------------------------------------------

def _pack_depth(seq_dir: Path, fs: _Fs) -> None:
    depth_dir = seq_dir / "depth"
    files = _files(depth_dir, "frame_*_depth.npy")
    if not files:
        return
    out = seq_dir / "depth.npz"
    if out.exists():
        fs.stats.skipped.append(f"{out} already exists")
        return
    indices = [int(p.name.split("_")[1]) for p in files]
    if not fs.dry_run:
        stack = np.stack([np.load(p) for p in files])
        clipped = int(np.count_nonzero(stack > DEPTH_MAX_M))
        quantized = np.round(
            np.clip(stack, 0.0, DEPTH_MAX_M) / DEPTH_SCALE_M
        ).astype(np.uint16)
        np.savez_compressed(
            out,
            depth_mm=quantized,
            frame_indices=np.asarray(indices, dtype=np.int32),
            scale_m_per_unit=np.float32(DEPTH_SCALE_M),
        )
        if clipped:
            fs.stats.actions.append(
                f"seq/{seq_dir.name}: {clipped} depth px above "
                f"{DEPTH_MAX_M}m clipped by uint16 packing"
            )
        fs.record_written(out)
    fs.remove_tree(depth_dir)
    fs.stats.actions.append(
        f"seq/{seq_dir.name}: {len(files)} depth .npy -> depth.npz (uint16 mm)"
    )


def _pack_scale(seq_dir: Path, fs: _Fs) -> None:
    """Collapse per-frame mesh_scale metadata into one npz.

    The per-object aggregate survives — ``state.resolved_objects[].scale_path``
    points at it and outputs.write_episode_folder copies it into the bundle —
    but a legacy one is relocated to where resolve() now writes it; see the
    hoist below.
    """
    scale_root = seq_dir / "scale"
    if not scale_root.is_dir():
        return
    out = seq_dir / "scale_per_frame.npz"
    if out.exists():
        fs.stats.skipped.append(f"{out} already exists")
        return
    payload: dict[str, np.ndarray] = {}
    consumed: list[Path] = []
    for obj_dir in sorted(p for p in scale_root.iterdir() if p.is_dir()):
        metas = _files(obj_dir, "frame_*_meta.npz")
        if not metas:
            continue
        scales, n_pts, frames = [], [], []
        for m in metas:
            with np.load(m) as d:
                scales.append(float(d["scale"]))
                n_pts.append(int(d["n_observed_pts"]) if "n_observed_pts" in d else 0)
            frames.append(int(m.name.split("_")[1]))
        payload[f"{obj_dir.name}__scale"] = np.asarray(scales, dtype=np.float64)
        payload[f"{obj_dir.name}__n_observed_pts"] = np.asarray(n_pts, dtype=np.int64)
        payload[f"{obj_dir.name}__frame_indices"] = np.asarray(frames, dtype=np.int32)
        consumed.extend(metas)
    if not consumed:
        return
    if not fs.dry_run:
        np.savez_compressed(out, **payload)
        fs.record_written(out)
    fs.remove_files(consumed)
    fs.stats.actions.append(
        f"seq/{seq_dir.name}: {len(consumed)} scale meta .npz -> scale_per_frame.npz"
    )

    # Hoist each object's aggregate out of scale/<obj>/, where it is the only
    # surviving file, to <seq>/<obj>_scale.npy — matching what resolve() now
    # writes directly. state.json's scale_path is rewritten to match.
    moved: dict[str, str] = {}
    for obj_dir in sorted(p for p in scale_root.iterdir() if p.is_dir()):
        agg = obj_dir / "aggregate_scale.npy"
        if not agg.is_file():
            continue
        dst = seq_dir / f"{obj_dir.name}_scale.npy"
        if not fs.dry_run and not dst.exists():
            shutil.move(str(agg), str(dst))
        moved[str(agg)] = str(dst)
    if moved:
        _rewrite_state_paths(seq_dir.parent.parent, "scale_path", moved, fs)
        fs.stats.actions.append(
            f"seq/{seq_dir.name}: {len(moved)} aggregate_scale.npy -> <obj>_scale.npy"
        )
    if not fs.dry_run:
        shutil.rmtree(scale_root, ignore_errors=True)


def _rewrite_state_paths(
    run_root: Path, field: str, moved: dict[str, str], fs: _Fs,
) -> None:
    """Repoint ``state.resolved_objects[].<field>`` at relocated files.

    Matching is by ``os.path.realpath`` on both sides, not by string equality:
    ``run_root`` here is already resolved, whereas state.json stores whatever
    string the writing stage happened to hold. Those differ whenever any
    component is a symlink — ``outputs/`` is one in the main checkout — and a
    plain string compare would silently skip the rewrite, leaving the field
    pointing at a file this cleanup just deleted.
    """
    if fs.dry_run or not moved:
        return
    state_path = run_root / "state.json"
    if not state_path.is_file():
        return
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    by_real = {os.path.realpath(src): dst for src, dst in moved.items()}
    changed = False
    for obj in state.get("resolved_objects") or []:
        current = obj.get(field) or ""
        new = by_real.get(os.path.realpath(current)) if current else None
        if new and new != current:
            obj[field] = new
            changed = True
    if changed:
        state_path.write_text(json.dumps(state, indent=2))


def _rewrite_geometry_prior_panels(
    bundle: Path, moved: dict[str, str], fs: _Fs,
) -> None:
    """Repoint ``geometry_priors.json``'s VLM panel paths at hoisted files.

    ``axis_align[].panel_path`` and ``orient[].panel_image`` are the audit
    trail those panels are kept for, so a hoist has to carry them along.

    Matched by ``os.path.realpath``, same reason as ``_rewrite_state_paths``:
    the recorded string carries whatever prefix the writing stage held, which
    differs from the resolved path we moved whenever a component is a symlink.
    """
    if fs.dry_run or not moved:
        return
    path = bundle / "geometry_priors.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    by_real = {os.path.realpath(src): dst for src, dst in moved.items()}
    changed = False
    for key, field in (("axis_align", "panel_path"), ("orient", "panel_image")):
        for entry in data.get(key) or []:
            current = entry.get(field) or ""
            new = by_real.get(os.path.realpath(current)) if current else None
            if new and new != current:
                entry[field] = new
                changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n")


def clean_seq(run_root: Path, fs: _Fs, *, state: dict, quantize_depth: bool) -> None:
    root = run_root / "seq"
    if not root.is_dir():
        return
    frames_removable = _terminal_stage_done(run_root)
    gate = _terminal_stage(run_root)[0]
    depth_removable = _stage_ok(state, "pose_tracking")

    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if quantize_depth and depth_removable:
            _pack_depth(seq_dir, fs)
        elif quantize_depth:
            fs.stats.skipped.append(
                f"seq/{seq_dir.name}/depth: pose_tracking not complete"
            )

        vis_dir = seq_dir / "vis"
        if vis_dir.is_dir():
            fs.remove_tree(vis_dir)
            fs.stats.actions.append(f"seq/{seq_dir.name}: dropped vis/ (unread debug)")

        if frames_removable:
            frames_dir = seq_dir / "frames"
            if frames_dir.is_dir():
                fs.remove_tree(frames_dir)
                fs.stats.actions.append(
                    f"seq/{seq_dir.name}: dropped frames/ (in rectified_left.mp4)"
                )
        else:
            fs.stats.skipped.append(
                f"seq/{seq_dir.name}/frames: {gate} not complete"
            )

        _pack_scale(seq_dir, fs)
        fs.prune_empty(seq_dir)


# ---------------------------------------------------------------------------
# poses/
# ---------------------------------------------------------------------------

def _resolve_ffmpeg() -> str | None:
    """Bundled binary first, matching _toolkit/video_build."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _mp4_frame_count(ffmpeg: str, path: Path) -> int:
    """Frames actually decodable from ``path``; -1 if ffmpeg could not tell."""
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return -1
    matches = re.findall(r"frame=\s*(\d+)", proc.stderr)
    return int(matches[-1]) if matches else -1


def _frames_to_mp4(frames_dir: Path, out_mp4: Path, fps: float) -> bool:
    """Encode ``frame_NNNNNN.png`` into an mp4. False if ffmpeg is unusable.

    Uses the same ``frame_%06d.png`` sequence pattern as
    ``_toolkit/video_build``, which means ffmpeg's image2 demuxer stops at the
    first missing index. video_build can live with that because it leaves the
    PNGs in place; here the caller DELETES them once this returns True, so a
    gap would silently lose frames. Hence the count check: unless the encoded
    mp4 holds every PNG we found, the output is discarded and the frames stay.
    """
    frames = _files(frames_dir, "frame_*.png")
    if not frames:
        return False
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return False
    start = int(frames[0].stem.split("_")[1])
    cmd = [
        ffmpeg, "-y", "-nostdin", "-loglevel", "error",
        "-framerate", str(fps),
        "-start_number", str(start),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_mp4),
    ]
    if subprocess.run(cmd, capture_output=True).returncode != 0 or not out_mp4.is_file():
        return False
    n_encoded = _mp4_frame_count(ffmpeg, out_mp4)
    if n_encoded != len(frames):
        out_mp4.unlink(missing_ok=True)
        return False
    return True


def _pack_pose_dir(pose_dir: Path, dst: Path, label: str, fs: _Fs) -> bool:
    """Collapse a legacy per-frame pose dir into the packed archive at ``dst``.

    Returns True when the directory was replaced, so callers can repoint
    whatever referenced it.

    ``pose_store`` resolves ``<root>.npz`` as well as ``<root>/poses.npz``, so
    ``foundation_pose/`` -> ``foundation_pose.npz`` keeps a manifest pointing
    at the old directory path working for free. That trick does NOT extend to
    ``poses_npy/`` -> ``poses.npz``, where the stem changes — that caller has
    to rewrite its reference explicitly.
    """
    from ar2s.droid_sim import pose_store

    if not pose_dir.is_dir() or dst.exists():
        return False
    sources = _files(pose_dir, "frame_*_transform.npy")
    if not sources:
        return False
    if fs.dry_run:
        fs.stats.actions.append(f"{label}: {len(sources)} pose .npy -> {dst.name}")
        fs.stats.files_removed += len(sources)
        fs.stats.bytes_removed += sum(p.stat().st_size for p in sources)
        return True
    poses = pose_store.load_poses(pose_dir)
    out = pose_store.save_poses(dst, poses)
    fs.record_written(out)
    fs.remove_tree(pose_dir)
    fs.stats.actions.append(f"{label}: {len(sources)} pose .npy -> {dst.name}")
    return True


def clean_poses(run_root: Path, fs: _Fs, *, state: dict) -> None:
    from ar2s.droid_sim import pose_store

    root = run_root / "poses"
    if not root.is_dir():
        return
    fps = float(
        ((state.get("stages") or {}).get("svo_extract", {}).get("outputs") or {})
        .get("fps") or _TRACK_VIS_FPS
    ) or _TRACK_VIS_FPS

    for obj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        track_vis = obj_dir / "track_vis"
        if track_vis.is_dir():
            out_mp4 = obj_dir / "track_vis.mp4"
            if out_mp4.exists():
                fs.stats.skipped.append(f"{out_mp4} already exists")
            elif fs.dry_run or _frames_to_mp4(track_vis, out_mp4, fps):
                fs.record_written(out_mp4)
                fs.remove_tree(track_vis)
                fs.stats.actions.append(
                    f"poses/{obj_dir.name}: track_vis/ -> track_vis.mp4"
                )
            else:
                fs.stats.errors.append(
                    f"poses/{obj_dir.name}: ffmpeg failed or dropped frames, "
                    f"kept track_vis/"
                )

        # One .npy per tracked frame -> one poses.npz beside the object, not
        # inside a poses_npy/ level that would hold a single file. New runs
        # write poses.npz directly, so this only migrates legacy layouts —
        # and because the stem changes, state.json has to be repointed too.
        legacy_dir = obj_dir / "poses_npy"
        packed = obj_dir / "poses.npz"
        if _pack_pose_dir(legacy_dir, packed, f"poses/{obj_dir.name}", fs):
            _rewrite_state_paths(
                run_root, "pose_source_path", {str(legacy_dir): str(packed)}, fs,
            )

        # ob_in_cam/*.txt is FoundationPose's raw output; _toolkit/pose_tracking
        # converts it to the packed pose source, which is the documented
        # downstream contract (ObjectInput.pose_source_path). Nothing reads the
        # .txt after that. Gated on the poses having actually been collected,
        # in EITHER layout — keying this on poses_npy/ alone meant it never
        # fired for a run that wrote poses.npz directly.
        ob_in_cam = obj_dir / "ob_in_cam"
        if ob_in_cam.is_dir() and pose_store.exists(packed):
            fs.remove_tree(ob_in_cam)
            fs.stats.actions.append(f"poses/{obj_dir.name}: dropped ob_in_cam/ .txt")

        mesh_exports = obj_dir / "_mesh_exports"
        if mesh_exports.is_dir():
            fs.remove_tree(mesh_exports)
            fs.stats.actions.append(f"poses/{obj_dir.name}: dropped _mesh_exports/")
        fs.prune_empty(obj_dir)


# ---------------------------------------------------------------------------
# segmentation/
# ---------------------------------------------------------------------------

def clean_segmentation(run_root: Path, fs: _Fs) -> None:
    gated = _terminal_stage_done(run_root)
    gate = _terminal_stage(run_root)[0]
    for name in ("segmentation", "segmentation_secondary"):
        root = run_root / name
        if not root.is_dir():
            continue
        if not gated:
            fs.stats.skipped.append(f"{name}/: {gate} not complete")
            continue
        n_masks = 0
        for masks_dir in sorted(root.glob("*/masks")):
            if not masks_dir.is_dir():
                continue
            n_masks += sum(1 for _ in masks_dir.iterdir())
            fs.remove_tree(masks_dir)
        if n_masks:
            fs.stats.actions.append(
                f"{name}/: dropped {n_masks} mask PNGs (overlay mp4s kept)"
            )
        fs.prune_empty(root)


# ---------------------------------------------------------------------------
# sysid_inputs/ — the deliverable bundle keeps everything downstream reads
# ---------------------------------------------------------------------------

def _visual_obj_carries_colors(visual_obj: Path) -> bool | None:
    """None when we cannot tell (missing dep) — caller then keeps the backup."""
    try:
        from ar2s.droid_sim.usd_exporter import obj_has_explicit_vertex_colors
    except Exception:
        return None
    try:
        return bool(obj_has_explicit_vertex_colors(visual_obj))
    except Exception:
        return None


def clean_sysid_inputs(run_root: Path, fs: _Fs) -> None:
    for bundle in _bundle_dirs(run_root):
        objects_root = bundle / "objects"
        if not objects_root.is_dir():
            continue
        for obj_dir in sorted(p for p in objects_root.iterdir() if p.is_dir()):
            # mesh_orient writes one full .obj per flip hypothesis purely so
            # MuJoCo's loader can read it back; only the panel PNG survives as
            # an artifact worth keeping.
            orient_tmp = obj_dir / "orient_tmp"
            meshes = _files(orient_tmp, "mesh_*.obj") if orient_tmp.is_dir() else []
            if meshes:
                fs.remove_files(meshes)
                fs.stats.actions.append(
                    f"{obj_dir.name}/orient_tmp: dropped {len(meshes)} render-scratch .obj"
                )
            # The panel PNG is all that is left in there; hoist it so the dir
            # goes away (mesh_orient now writes it here directly).
            if orient_tmp.is_dir():
                moved: dict[str, str] = {}
                for panel in _files(orient_tmp, "orient_panel_*.png"):
                    dst = obj_dir / panel.name
                    if not fs.dry_run and not dst.exists():
                        shutil.move(str(panel), str(dst))
                    moved[str(panel)] = str(dst)
                if not fs.dry_run:
                    shutil.rmtree(orient_tmp, ignore_errors=True)
                _rewrite_geometry_prior_panels(bundle, moved, fs)

            # mesh_axis_align does the same, ~50 MiB per candidate: 18 flip
            # hypotheses plus a mesh_dec_* per already-decided object, ~900 MiB
            # per object in all. Ungated — the stage rmtree's this directory on
            # entry, so no resume reads it.
            axis_align_tmp = obj_dir / "axis_align_tmp"
            if axis_align_tmp.is_dir():
                meshes = _files(axis_align_tmp, "mesh_*.obj")
                if meshes:
                    fs.remove_files(meshes)
                    fs.stats.actions.append(
                        f"{obj_dir.name}/axis_align_tmp: dropped {len(meshes)} "
                        f"render-scratch .obj"
                    )
                # Unlike the orient panel, geometry_priors.json records where
                # this one lives, so hoisting it has to repoint that.
                moved = {}
                for panel in _files(axis_align_tmp, "axis_align_panel_*.png"):
                    dst = obj_dir / panel.name
                    if not fs.dry_run and not dst.exists():
                        shutil.move(str(panel), str(dst))
                    moved[str(panel)] = str(dst)
                if not fs.dry_run:
                    shutil.rmtree(axis_align_tmp, ignore_errors=True)
                _rewrite_geometry_prior_panels(bundle, moved, fs)

            _pack_pose_dir(obj_dir / "foundation_pose",
                           obj_dir / "foundation_pose.npz", obj_dir.name, fs)

            # Byte-identical copy of poses/<obj>/track_vis, which clean_poses
            # has already turned into an mp4.
            track_vis = obj_dir / "track_vis"
            if track_vis.is_dir():
                fs.remove_tree(track_vis)
                fs.stats.actions.append(
                    f"{obj_dir.name}: dropped duplicated track_vis/ (see poses/)"
                )

            # One reader left: usd_exporter falls back to these for vertex
            # colors on legacy bundles whose visual.obj carries none, so the
            # color check below is the whole condition. Neither backup is a
            # re-run guard — mesh_axis_align guards on
            # foundation_pose_pre_align/ (never touched here), and
            # mesh_orient's existence check only protects its own backup.
            visual_obj = obj_dir / "visual.obj"
            backups = [obj_dir / n for n in
                       ("visual_pre_align.obj", "visual_pre_orient.obj")]
            backups = [b for b in backups if b.is_file()]
            if backups and visual_obj.is_file():
                has_colors = _visual_obj_carries_colors(visual_obj)
                if has_colors:
                    fs.remove_files(backups)
                    fs.stats.actions.append(
                        f"{obj_dir.name}: dropped {len(backups)} visual_pre_*.obj "
                        f"(visual.obj carries vertex colors)"
                    )
                else:
                    fs.stats.skipped.append(
                        f"{obj_dir.name}: visual.obj has no explicit vertex "
                        f"colors, kept visual_pre_*.obj as the USD fallback"
                    )
            fs.prune_empty(obj_dir)


# ---------------------------------------------------------------------------
# Cross-tree duplicate collapsing
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Only files at least this large are candidates for staging symlinks. The mp4s
# and the raw trajectory h5 carry essentially all of raw_episodes' bytes; the
# sub-MiB sidecars save nothing and would only widen the dangling surface.
_LINK_MIN_BYTES = 1 << 20


def _raw_episode_dirs(
    run_root: Path, raw_root: Path, fs: _Fs,
) -> list[tuple[Path, str]]:
    """``(directory, staged_dir_name)`` pairs for either raw_episodes layout.

    Flat (current): the directory IS ``raw_episodes/`` and the staged folder is
    named after ``state.json``'s episode_id, which is the only place the id
    survives once the ``<episode_id>/`` level is gone. Nested (legacy): one
    pair per child, named after the child.
    """
    if raw_episodes_is_flat(raw_root):
        episode_id = (_load_state(run_root).get("episode_id") or "").strip()
        if not episode_id:
            fs.stats.skipped.append(
                "raw_episodes/: flat layout but state.json has no episode_id, "
                "cannot identify the staged counterpart"
            )
            return []
        return [(raw_root, episode_id)]
    return [(p, p.name) for p in sorted(raw_root.iterdir()) if p.is_dir()]


def link_raw_episodes(run_root: Path, staging_root: Path, fs: _Fs) -> None:
    """Replace raw_episodes copies with symlinks into a PERSISTENT staging root.

    ``build_raw_episode`` byte-copies the stereo/wrist mp4s and their sidecars
    out of the staged raw_data folder, but also *derives* three files that have
    no upstream counterpart (``cameras_extrinsics.npz`` from the refined
    extrinsics, ``robot_traj.h5`` flattened from trajectory.h5, and
    ``camera_selection.json`` recording the camera vote). So the directory as a
    whole cannot become one symlink — only the copies can, individually.

    Matching is content-addressed: a raw_episodes file is linked only when some
    staged file hashes identically. That needs no filename mapping and cannot
    mislink, and the derived files simply never match.

    ``staging_root`` must OUTLIVE the run. Do not point this at the per-job
    scratch that build_raw_episode read from (``/localscratch/<job>/...`` on
    SLURM) — that is deleted at job exit and every link would dangle.
    """
    raw_root = run_root / "raw_episodes"
    if not raw_root.is_dir():
        return
    staging_root = Path(staging_root).expanduser().resolve()
    if not staging_root.is_dir():
        fs.stats.errors.append(f"staging root not a directory: {staging_root}")
        return

    for episode_dir, staged_name in _raw_episode_dirs(run_root, raw_root, fs):
        staged = staging_root / staged_name
        if not staged.is_dir():
            fs.stats.skipped.append(
                f"raw_episodes ({staged_name}): no staged counterpart under "
                f"{staging_root}"
            )
            continue

        by_hash: dict[str, Path] = {}
        for p in sorted(staged.rglob("*")):
            if p.is_file() and not p.is_symlink() and p.stat().st_size >= _LINK_MIN_BYTES:
                by_hash.setdefault(_sha256(p), p)
        if not by_hash:
            fs.stats.skipped.append(f"{staged}: no files >= {human_size(_LINK_MIN_BYTES)}")
            continue

        n_linked = 0
        for p in sorted(episode_dir.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            size = p.stat().st_size
            if size < _LINK_MIN_BYTES:
                continue
            target = by_hash.get(_sha256(p))
            if target is None:
                fs.stats.skipped.append(
                    f"{p.relative_to(run_root)}: no identical file staged, kept as a copy"
                )
                continue
            fs.stats.bytes_removed += size
            n_linked += 1
            if not fs.dry_run:
                try:
                    tmp = p.with_name(p.name + ".relink")
                    tmp.symlink_to(target)
                    os.replace(tmp, p)
                except OSError as e:
                    fs.stats.errors.append(f"symlink {p}: {e}")
        if n_linked:
            fs.stats.actions.append(
                f"raw_episodes ({staged_name}): {n_linked} file(s) -> symlinks "
                f"into {staging_root}"
            )


def dedupe_bundle_copies(run_root: Path, fs: _Fs) -> None:
    """Hardlink the bundle's byte-identical copies back onto their sources.

    write_episode_folder copies the rectified video, first frame, trajectory
    h5 and extrinsics npz into the bundle. Both paths must keep working, so
    these become hardlinks rather than deletions. Verified by hash and skipped
    across device boundaries.

    Runs after the flatten step, so the bundle paths named here are the ones
    write_episode_folder emits today — a legacy ``trajectory/robot.h5`` has
    already been hoisted to ``robot_traj.h5`` by then.
    """
    state = _load_state(run_root)
    for bundle in _bundle_dirs(run_root):
        pairs: list[tuple[Path, Path]] = []
        seq_dirs = sorted((run_root / "seq").glob("*")) if (run_root / "seq").is_dir() else []
        for seq_dir in seq_dirs:
            pairs.append((seq_dir / "rectified_left.mp4", bundle / "real" / "video.mp4"))
        raw = state.get("robot_traj_path") or ""
        if raw:
            pairs.append((Path(raw), bundle / "robot_traj.h5"))
        extr = state.get("cameras_extrinsics_path") or ""
        if extr:
            pairs.append((Path(extr), bundle / "cameras" / "extrinsics.npz"))

        for src, dst in pairs:
            try:
                if not (src.is_file() and dst.is_file()):
                    continue
                s_stat, d_stat = src.stat(), dst.stat()
                if s_stat.st_dev != d_stat.st_dev or s_stat.st_ino == d_stat.st_ino:
                    continue
                if s_stat.st_size != d_stat.st_size or _sha256(src) != _sha256(dst):
                    continue
                # Bytes are freed but the inode stays, so only the byte
                # counter moves here.
                fs.stats.bytes_removed += d_stat.st_size
                if not fs.dry_run:
                    tmp = dst.with_suffix(dst.suffix + ".relink")
                    os.link(src, tmp)
                    os.replace(tmp, dst)
                fs.stats.actions.append(
                    f"hardlinked {dst.relative_to(run_root)} -> "
                    f"{src if not src.is_relative_to(run_root) else src.relative_to(run_root)}"
                )
            except OSError as e:
                fs.stats.errors.append(f"dedupe {dst}: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cleanup_run(
    run_root: str | Path,
    *,
    dry_run: bool = False,
    quantize_depth: bool = True,
    archive_usd: bool = True,
    dedupe: bool = True,
    link_raw_root: str | Path | None = None,
) -> CleanupStats:
    """Consolidate one run directory in place. Never raises.

    A ``<run_root>__v2`` scene_view_repair branch, if present, is cleaned as
    part of the same call and folded into the returned stats with its lines
    prefixed ``__v2/``.

    Args:
        run_root: ``outputs/run_pipeline/<run_id>/``.
        dry_run: report the plan, change nothing.
        quantize_depth: pack ``seq/*/depth`` to uint16-millimetre npz. Off
            keeps the float32 per-frame .npy files untouched.
        archive_usd: pack ``artifacts/sweep-*/samples`` to tar.zst. The best
            sample is promoted to the sweep root either way.
        dedupe: hardlink the bundle's byte-identical copies onto their sources.
        link_raw_root: opt-in. Replace raw_episodes copies with symlinks into
            this staging root, which must outlive the run — see
            ``link_raw_episodes``. Off by default: build_raw_episode's source
            is per-job scratch on SLURM, so linking is only safe against a
            separately-persisted staging tree.

    Returns:
        ``CleanupStats`` — counts, a per-action log, and any errors. Steps are
        independent: one failing does not stop the others.
    """
    run_root = Path(run_root).expanduser().resolve()
    stats = CleanupStats()
    if not run_root.is_dir():
        stats.errors.append(f"not a directory: {run_root}")
        return stats

    fs = _Fs(dry_run, stats)
    state = _load_state(run_root)

    steps: list[tuple[str, object]] = [
        ("seq", lambda: clean_seq(run_root, fs, state=state,
                                  quantize_depth=quantize_depth)),
        ("poses", lambda: clean_poses(run_root, fs, state=state)),
        ("segmentation", lambda: clean_segmentation(run_root, fs)),
        ("sysid_inputs", lambda: clean_sysid_inputs(run_root, fs)),
        # Before dedupe/link_raw, which address files by the path the current
        # writers emit — this is what migrates a legacy run dir onto them.
        ("flatten", lambda: _flatten_single_child_dirs(run_root, fs)),
    ]
    if archive_usd:
        steps.insert(0, ("artifacts", lambda: clean_artifacts(run_root, fs)))
    if dedupe:
        steps.append(("dedupe", lambda: dedupe_bundle_copies(run_root, fs)))
    if link_raw_root is not None:
        steps.append((
            "link_raw", lambda: link_raw_episodes(run_root, Path(link_raw_root), fs),
        ))
    steps.append(("pycache", lambda: _drop_pycache(run_root, fs)))

    for name, fn in steps:
        try:
            fn()  # type: ignore[operator]
        except Exception as e:  # noqa: BLE001 — cleanup must not fail the run
            stats.errors.append(f"{name}: {type(e).__name__}: {e}")

    v2_root = v2_branch_root(run_root)
    if v2_root is not None:
        v2 = cleanup_run(
            v2_root,
            dry_run=dry_run,
            quantize_depth=quantize_depth,
            archive_usd=archive_usd,
            dedupe=dedupe,
            link_raw_root=link_raw_root,
        )
        stats.files_removed += v2.files_removed
        stats.bytes_removed += v2.bytes_removed
        stats.files_written += v2.files_written
        stats.bytes_written += v2.bytes_written
        stats.actions += [f"{_V2_SUFFIX}/ {a}" for a in v2.actions]
        stats.skipped += [f"{_V2_SUFFIX}/ {s}" for s in v2.skipped]
        stats.errors += [f"{_V2_SUFFIX}/ {e}" for e in v2.errors]

    return stats


def _flatten_single_child_dirs(run_root: Path, fs: _Fs) -> None:
    """Hoist files out of directories that exist only to hold one thing.

    ``material_classify/crops/``, the bundle's ``trajectory/`` and
    ``visual_inputs/`` each held a single entry; every writer now emits flat,
    so this only migrates run dirs produced before that change.
    """
    _hoist_visual_input_stub(run_root, fs)
    _hoist_episode_bundle(run_root, fs)

    crops = run_root / "material_classify" / "crops"
    if crops.is_dir():
        n = 0
        for f in sorted(p for p in crops.iterdir() if p.is_file()):
            dst = crops.parent / f.name
            if not fs.dry_run and not dst.exists():
                shutil.move(str(f), str(dst))
            n += 1
        if not fs.dry_run:
            shutil.rmtree(crops, ignore_errors=True)
        if n:
            fs.stats.actions.append(
                f"material_classify: hoisted {n} crop(s) out of crops/"
            )

    for bundle in _bundle_dirs(run_root):
        traj_dir = bundle / "trajectory"
        src = traj_dir / "robot.h5"
        if not src.is_file():
            continue
        dst = bundle / "robot_traj.h5"
        if not fs.dry_run:
            if not dst.exists():
                shutil.move(str(src), str(dst))
            # Unconditional, not tied to the move: trajectory/ is about to go
            # away, so the manifest must stop pointing into it whether this
            # run did the move or an interrupted earlier one did.
            _rewrite_manifest_key(
                bundle, ("robot", "trajectory"), "robot_traj.h5", fs,
            )
            shutil.rmtree(traj_dir, ignore_errors=True)
        fs.stats.actions.append(f"{bundle.name}: trajectory/robot.h5 -> robot_traj.h5")

    _flatten_raw_episodes(run_root, fs)


def _hoist_episode_bundle(run_root: Path, fs: _Fs) -> None:
    """Hoist ``sysid_inputs/<episode_id>_agent/*`` up to ``sysid_inputs/``.

    Runs BEFORE the trajectory hoist below, so that one sees the final bundle
    path. Manifest paths are all bundle-relative, so nothing inside needs
    rewriting — only things outside pointing IN would, and nothing does:
    state.json holds no bundle paths.
    """
    root = run_root / "sysid_inputs"
    if not root.is_dir():
        return
    legacy = [p for p in sorted(root.glob("*_agent")) if p.is_dir()]
    if not legacy:
        return
    if len(legacy) > 1 or [p for p in root.iterdir() if p not in legacy]:
        fs.stats.skipped.append(
            f"sysid_inputs/: unexpected layout "
            f"({[p.name for p in root.iterdir()]}), not flattened"
        )
        return
    bundle = legacy[0]
    n = sum(1 for _ in bundle.iterdir())
    if not fs.dry_run:
        for p in sorted(bundle.iterdir()):
            shutil.move(str(p), str(root / p.name))
        bundle.rmdir()
    fs.stats.actions.append(
        f"sysid_inputs/{bundle.name}/: hoisted {n} entr(ies) up one level"
    )


def _hoist_visual_input_stub(run_root: Path, fs: _Fs) -> None:
    """Hoist ``visual_inputs/visual_<id>.py`` up to the run root.

    The stub resolves its own paths from ``Path(__file__).parent``, so moving
    it one level up requires rewriting that expression too — an unmoved copy
    would resolve _RUN_DIR to the run root's PARENT.
    """
    legacy = run_root / "visual_inputs"
    if not legacy.is_dir():
        return
    stubs = sorted(legacy.glob("visual_*.py"))
    for stub in stubs:
        dst = run_root / stub.name
        if dst.exists():
            fs.stats.skipped.append(f"{dst} already exists")
            continue
        if not fs.dry_run:
            text = stub.read_text().replace(
                "Path(__file__).resolve().parent.parent",
                "Path(__file__).resolve().parent",
            )
            dst.write_text(text)
            stub.unlink()
        fs.stats.actions.append(f"visual_inputs/{stub.name} -> {stub.name}")
    if not fs.dry_run:
        shutil.rmtree(legacy / "__pycache__", ignore_errors=True)
        with contextlib.suppress(OSError):
            legacy.rmdir()


def _flatten_raw_episodes(run_root: Path, fs: _Fs) -> None:
    """Hoist ``raw_episodes/<episode_id>/*`` up to ``raw_episodes/``.

    build_raw_episode now writes flat (one run dir holds one episode, so the
    per-episode level could only ever have a single child), but runs made
    before that carry the extra level. Both collectors already accept either
    shape, so migrating costs nothing and removes a directory.

    Bails out rather than guessing if the layout is not what we expect: an
    already-flat dir, more than one child, or a name collision between the
    child's contents and something already at the top.
    """
    raw_root = run_root / "raw_episodes"
    if not raw_root.is_dir() or raw_episodes_is_flat(raw_root):
        return
    children = [p for p in raw_root.iterdir()]
    nested = [p for p in children if p.is_dir()]
    if len(children) != 1 or len(nested) != 1:
        if children:
            fs.stats.skipped.append(
                f"raw_episodes/: unexpected layout ({len(children)} entries), not flattened"
            )
        return
    episode_dir = nested[0]
    collisions = [p.name for p in episode_dir.iterdir() if (raw_root / p.name).exists()]
    if collisions:
        fs.stats.errors.append(
            f"raw_episodes/{episode_dir.name}: would collide with {collisions}, not flattened"
        )
        return
    n = sum(1 for _ in episode_dir.iterdir())
    if not fs.dry_run:
        for p in sorted(episode_dir.iterdir()):
            shutil.move(str(p), str(raw_root / p.name))
        episode_dir.rmdir()
        _rewrite_raw_episode_refs(run_root, episode_dir.name, fs)
    fs.stats.actions.append(
        f"raw_episodes/{episode_dir.name}/: hoisted {n} entr(ies) up one level"
    )


def _rewrite_raw_episode_refs(run_root: Path, episode_id: str, fs: _Fs) -> None:
    """Repoint everything that named the now-removed ``raw_episodes/<id>/``.

    state.json holds seven absolute paths into it (stereo_stream_path,
    robot_traj_path, cameras_extrinsics_path, ...) and the auto-generated
    ``visual_<id>.py`` stub builds its own from
    ``_RUN_DIR / "raw_episodes" / "<episode_id>"``. Both are plain text, and
    the segment being removed is unambiguous, so a targeted string
    substitution beats re-deriving each field.
    """
    edits: list[tuple[Path, list[tuple[str, str]]]] = [
        (run_root / "state.json", [(f"/raw_episodes/{episode_id}/", "/raw_episodes/")]),
    ]
    stub = visual_input_stub(run_root)
    if stub is not None:
        edits.append((stub, [
            (f'"raw_episodes" / "{episode_id}"', '"raw_episodes"'),
            (f"raw_episodes/{episode_id}/", "raw_episodes/"),
        ]))
    for path, subs in edits:
        if not path.is_file():
            continue
        try:
            text = original = path.read_text()
            for old, new in subs:
                text = text.replace(old, new)
            if text != original:
                path.write_text(text)
        except OSError as e:
            fs.stats.errors.append(f"rewrite {path}: {e}")


def _rewrite_manifest_key(bundle: Path, keys: tuple[str, ...], value: str,
                          fs: _Fs) -> None:
    """Point a nested manifest.yaml key at a relocated file."""
    import yaml

    path = bundle / "manifest.yaml"
    if not path.is_file():
        return
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as e:  # noqa: BLE001 — malformed manifest is the user's
        fs.stats.errors.append(f"manifest {path}: {e}")
        return
    node = data
    for k in keys[:-1]:
        node = node.get(k)
        if not isinstance(node, dict):
            return
    node[keys[-1]] = value
    path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False))


def select_run_dirs(root: Path) -> list[Path]:
    """Every run dir under ``root``, minus the ``__v2`` branches that
    ``cleanup_run`` already reaches through their main run.

    A branch whose main run dir is gone stays in the list — otherwise nothing
    would ever clean it.
    """
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    names = {p.name for p in dirs}
    return [p for p in dirs
            if not (_is_v2_branch(p) and p.name[:-len(_V2_SUFFIX)] in names)]


def _drop_pycache(run_root: Path, fs: _Fs) -> None:
    """__pycache__ is a side effect of exec'ing the entry stub."""
    for cache in sorted(run_root.rglob("__pycache__")):
        if cache.is_dir():
            fs.remove_tree(cache)
            fs.stats.actions.append(f"dropped {cache.relative_to(run_root)}")


def format_report(run_root: Path, stats: CleanupStats, *, dry_run: bool) -> str:
    lines = [
        "=" * 70,
        f"pipeline cleanup {'(DRY RUN)' if dry_run else ''}: {run_root}",
        "=" * 70,
    ]
    for a in stats.actions:
        lines.append(f"  {a}")
    for s in stats.skipped:
        lines.append(f"  SKIP  {s}")
    for e in stats.errors:
        lines.append(f"  ERROR {e}")
    lines.append(
        f"  removed {stats.files_removed} files ({human_size(stats.bytes_removed)}), "
        f"wrote {stats.files_written} ({human_size(stats.bytes_written)}), "
        f"net {human_size(stats.bytes_saved)}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI — `python -m ar2s.pipeline_cleanup`
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Clean finished run directories in place.

    ``ar2s.run_pipeline`` already calls ``cleanup_run`` at the end of a run
    (disable with ``--no-cleanup``), so this is for run dirs that finished
    before that existed, for re-running after a manual resume, and for
    ``--link-raw-to``, which is deliberately not part of the in-pipeline path.
    """
    import argparse

    from ar2s.pipeline_artifacts import pipeline_artifact_root

    default_root = pipeline_artifact_root()
    parser = argparse.ArgumentParser(
        prog="python -m ar2s.pipeline_cleanup",
        description=main.__doc__,
        epilog=(
            "Dry-run by default; pass --yes to apply. Say what to clean with "
            "either --run-id or --all. A run's <run_id>__v2 scene_view_repair "
            "branch is cleaned along with it, reported under a __v2/ prefix. "
            "Safe to rerun — every step no-ops when its output already "
            "exists. See the module docstring for the full policy and for "
            "which steps are gated on pipeline progress."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Actually modify files. Without this, only report the plan.",
    )
    parser.add_argument(
        "--root", type=Path, default=default_root,
        help=f"Directory holding per-run dirs (default: {default_root}; "
             f"override the default with AR2S_RUN_PIPELINE_ROOT).",
    )
    # Which runs to touch is always explicit, so --yes can never be given a
    # wider scope than the --yes-less preview the caller just read.
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--run-id", action="append", default=None,
        help="Clean this run dir under --root. Repeatable.",
    )
    selection.add_argument(
        "--all", action="store_true",
        help="Clean every run dir under --root.",
    )
    parser.add_argument("--no-depth-quantize", action="store_true",
                        help="Keep seq/*/depth as float32 per-frame .npy.")
    parser.add_argument("--no-usd-archive", action="store_true",
                        help="Keep artifacts/sweep-*/samples/ uncompressed.")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Skip hardlinking the bundle's duplicated copies.")
    parser.add_argument(
        "--link-raw-to", type=Path, default=None, metavar="STAGING_ROOT",
        help="Replace raw_episodes copies with symlinks into this staging "
             "root. Only files that hash-identically to a staged file are "
             "linked. STAGING_ROOT must OUTLIVE the run — do NOT pass the "
             "per-job scratch dir build_raw_episode read from.",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"Not a directory: {args.root}", file=sys.stderr)
        return 2

    if args.run_id:
        run_dirs = [args.root / r for r in args.run_id]
        missing = [d for d in run_dirs if not d.is_dir()]
        if missing:
            print(f"No such run dir(s): {', '.join(str(m) for m in missing)}",
                  file=sys.stderr)
            return 2
    else:
        run_dirs = select_run_dirs(args.root)
        if not run_dirs:
            print(f"No run dirs under {args.root}", file=sys.stderr)
            return 2

    dry_run = not args.yes
    print(f"Scanning {len(run_dirs)} run dir(s) under {args.root}")
    print(f"Mode: {'MODIFY' if args.yes else 'DRY RUN (pass --yes to apply)'}\n")

    total_removed = total_written = n_errors = 0
    for i, run_dir in enumerate(run_dirs, 1):
        stats = cleanup_run(
            run_dir,
            dry_run=dry_run,
            quantize_depth=not args.no_depth_quantize,
            archive_usd=not args.no_usd_archive,
            dedupe=not args.no_dedupe,
            link_raw_root=args.link_raw_to,
        )
        print(format_report(run_dir, stats, dry_run=dry_run))
        print()
        total_removed += stats.bytes_removed
        total_written += stats.bytes_written
        n_errors += len(stats.errors)
        if len(run_dirs) > 1 and (i % 25 == 0 or i == len(run_dirs)):
            print(f"  ...{i}/{len(run_dirs)} run dirs processed", file=sys.stderr)

    if len(run_dirs) > 1:
        print(f"Total across {len(run_dirs)} runs: freed {human_size(total_removed)}, "
              f"wrote {human_size(total_written)}, "
              f"net {human_size(total_removed - total_written)}")
    if n_errors:
        print(f"{n_errors} error(s) — see the per-run reports above", file=sys.stderr)
    if dry_run:
        print("\nDry run only — nothing was modified. Rerun with --yes to apply.")
    return 1 if n_errors else 0


__all__ = ["CleanupStats", "cleanup_run", "format_report", "human_size", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
