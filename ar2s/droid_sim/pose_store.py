"""Read/write FoundationPose per-frame object poses in either on-disk layout.

Two layouts exist and both stay readable forever:

  legacy directory   <root>/frame_NNNNNN_transform.npy   — one (4,4) per file
  packed archive     <root>.npz                          — one file per object

The legacy layout costs one inode per tracked frame per object, which for a
2-object 63-frame episode is 252 files in ``poses/`` plus another 126 copied
into the emitted bundle — ~380 inodes holding ~126 unique 4x4 matrices. HPC
filesystems cap inodes as well as bytes, so new runs write the packed form.

Resolution order for a given ``root`` (the path a manifest's
``foundation_pose_dir`` points at):

  1. ``<root>.npz``          — packed; ``<root>`` already ending in ``.npz``
                               resolves to itself
  2. ``<root>/frame_*.npy``  — legacy per-frame files

Naming the archive after the directory it replaces is what lets a manifest
emitted before this module existed keep working untouched: ``foundation_pose``
resolves to ``foundation_pose.npz`` with no manifest edit. Renames that change
the stem (``poses_npy/`` -> ``poses.npz``) get no such help and must repoint
their references — see ``ar2s.pipeline_cleanup.clean_poses``.

The packed archive holds:
    poses          (N, 4, 4) float64, ordered by frame index
    frame_indices  (N,)      int32
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_FRAME_RE = re.compile(r"frame_(\d+)_transform\.npy$")


def packed_path(root: Path | str) -> Path:
    """The packed archive path for ``root``; idempotent on an ``.npz`` path.

    Appends rather than using ``with_suffix``, which would rewrite the last
    dot-segment of an object name that legitimately contains one — object
    dirs are sanitised to ``[A-Za-z0-9._-]``, so ``cup.001`` is reachable and
    ``with_suffix`` would map it to ``cup.npz``.
    """
    root = Path(root)
    return root if root.suffix == ".npz" else root.with_name(root.name + ".npz")


def _resolve_packed(root: Path) -> Path | None:
    packed = packed_path(root)
    return packed if packed.is_file() else None


def exists(root: Path | str) -> bool:
    """True if ``root`` resolves to poses in either layout."""
    root = Path(root)
    if _resolve_packed(root) is not None:
        return True
    return root.is_dir() and any(root.glob("frame_*_transform.npy"))


def load_poses(root: Path | str) -> dict[int, np.ndarray]:
    """Return ``{frame_index: (4,4) float64}``. Empty when nothing resolves.

    Raises ValueError on a malformed entry — a silently-wrong pose would place
    an object at the wrong spot in the sim rather than fail.
    """
    root = Path(root)
    packed = _resolve_packed(root)
    if packed is not None:
        with np.load(packed) as data:
            poses = np.asarray(data["poses"], dtype=np.float64)
            indices = np.asarray(data["frame_indices"], dtype=np.int64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"expected (N,4,4) poses in {packed}, got {poses.shape}"
            )
        if len(indices) != len(poses):
            raise ValueError(
                f"{packed}: {len(indices)} frame_indices for {len(poses)} poses"
            )
        return {int(i): poses[k] for k, i in enumerate(indices)}

    if not root.is_dir():
        return {}
    out: dict[int, np.ndarray] = {}
    for path in sorted(root.glob("frame_*_transform.npy")):
        m = _FRAME_RE.search(path.name)
        if not m:
            continue
        value = np.asarray(np.load(str(path), allow_pickle=True), dtype=np.float64)
        if value.shape != (4, 4):
            raise ValueError(f"Expected (4,4) at {path}, got {value.shape}.")
        out[int(m.group(1))] = value
    return out


def load_pose(root: Path | str, frame_index: int) -> np.ndarray | None:
    """One frame's (4,4), or None if that frame has no pose.

    Reads only the requested entry in the legacy layout; the packed layout is
    read whole (a few hundred KB) since npz has no partial-row access.
    """
    root = Path(root)
    if _resolve_packed(root) is None and root.is_dir():
        path = root / f"frame_{int(frame_index):06d}_transform.npy"
        if not path.is_file():
            return None
        value = np.asarray(np.load(str(path), allow_pickle=True), dtype=np.float64)
        if value.shape != (4, 4):
            raise ValueError(f"Expected (4,4) at {path}, got {value.shape}.")
        return value
    return load_poses(root).get(int(frame_index))


def earliest_frame(root: Path | str) -> int | None:
    """Smallest frame index present, or None when nothing resolves.

    Per-object keyframes mean different objects can start at different frames
    (SAM3 may not ground an object until frame 2), so this is what sets a
    manifest's ``init_frame``.
    """
    root = Path(root)
    packed = _resolve_packed(root)
    if packed is not None:
        with np.load(packed) as data:
            indices = np.asarray(data["frame_indices"], dtype=np.int64)
        return int(indices.min()) if indices.size else None
    if not root.is_dir():
        return None
    indices = [
        int(m.group(1))
        for m in (_FRAME_RE.search(p.name) for p in root.glob("frame_*_transform.npy"))
        if m
    ]
    return min(indices) if indices else None


def save_poses(root: Path | str, poses: dict[int, np.ndarray]) -> Path:
    """Write ``{frame_index: (4,4)}`` as a packed archive. Returns its path.

    ``root`` is normalised through ``packed_path``, so writing back what
    ``load_poses`` was given round-trips whichever form the caller holds.
    """
    out = packed_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    indices = sorted(poses)
    stacked = (
        np.stack([np.asarray(poses[i], dtype=np.float64) for i in indices])
        if indices else np.zeros((0, 4, 4), dtype=np.float64)
    )
    np.savez_compressed(
        out,
        poses=stacked,
        frame_indices=np.asarray(indices, dtype=np.int32),
    )
    return out


__all__ = [
    "earliest_frame",
    "exists",
    "load_pose",
    "load_poses",
    "packed_path",
    "save_poses",
]
