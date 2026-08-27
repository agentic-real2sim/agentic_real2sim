"""File-system inspection skills for the Episode Agent.

All skills are read-only. They return structured info (dicts / strings / lists)
suitable for an LLM to reason about. No side effects.

Path semantics: absolute or relative-to-CWD (the agent prompt asks for absolute).
"""
from __future__ import annotations

from pathlib import Path

import h5py
from langchain_core.tools import tool


# When a directory has more than this many entries, collapse the middle
# (show only first and last) — saves tokens for FP / collision dirs etc.
_COLLAPSE_AT = 10
_KEEP_HEAD = 1
_KEEP_TAIL = 1


def _resolve(path: str) -> Path:
    return Path(path).expanduser()


def _is_pattern_dir(entries: list[dict]) -> bool:
    """Heuristic: True iff this looks like a 'pattern dir' worth collapsing —
    i.e., many files that mostly share the same extension (e.g., a
    FoundationPose dir of frame_*.npy, or a collision dir of *_collision_*.obj).

    A heterogeneous dir (mixed file types — like the raw episode folder)
    is NOT collapsed even if it has > _COLLAPSE_AT entries, because each
    different file type carries information the LLM needs.
    """
    if len(entries) <= _COLLAPSE_AT:
        return False
    file_entries = [e for e in entries if e.get("type") == "file"]
    if not file_entries:
        return False
    exts = [Path(e["name"]).suffix.lower() for e in file_entries]
    most_common = max(set(exts), key=exts.count)
    # ≥ 80% of files share a single extension → pattern dir
    return exts.count(most_common) / len(exts) >= 0.8


def _collapse_entries(entries: list[dict]) -> list[dict]:
    """Collapse to first + last + summary marker, BUT only for 'pattern dirs'
    where most files share an extension. Heterogeneous dirs return as-is.
    """
    if not _is_pattern_dir(entries):
        return entries
    omitted = len(entries) - _KEEP_HEAD - _KEEP_TAIL
    summary = {
        "type": "summary",
        "name": f"... ({omitted} more entries omitted; first and last shown) ...",
        "size_bytes": None,
    }
    return [*entries[:_KEEP_HEAD], summary, *entries[-_KEEP_TAIL:]]


# -------------------------------------------------------------------
# list_dir
# -------------------------------------------------------------------

@tool
def list_dir(path: str, recursive: bool = False) -> dict:
    """List directory contents (files AND subdirs).

    For directories with many entries (e.g., FoundationPose result dirs
    with hundreds of frame_*.npy, or convex-hull collision dirs with
    hundreds of *_collision_*.obj), the middle is auto-omitted —
    only the FIRST and LAST entries are shown, with a summary marker
    in between. This saves tokens; the names of first + last + the count
    are enough to identify the file pattern.

    Args:
        path: directory path
        recursive: if True, walk subdirectories (each entry's name shown
            as relative path from `path`)

    Returns:
        dict:
          - 'path':         resolved directory path
          - 'entries':      list of {name, type ('file' / 'dir' / 'summary'),
                                     size_bytes (None for dirs)}
          - 'total_count':  total number of entries before collapsing
          - 'collapsed':    True iff the middle was omitted
    """
    base = _resolve(path)
    if not base.exists():
        return {"error": f"path does not exist: {base}"}
    if not base.is_dir():
        return {"error": f"not a directory: {base}"}

    iterator = base.rglob("*") if recursive else base.iterdir()
    raw: list[dict] = []
    for p in iterator:
        try:
            is_dir = p.is_dir()
            raw.append({
                "name": str(p.relative_to(base)) if recursive else p.name,
                "type": "dir" if is_dir else "file",
                "size_bytes": None if is_dir else p.stat().st_size,
            })
        except Exception as e:
            raw.append({"name": p.name, "error": str(e)})

    total = len(raw)
    raw.sort(key=lambda e: e.get("name", ""))   # alphabetical for stable head/tail
    entries = _collapse_entries(raw)

    return {
        "path": str(base),
        "entries": entries,
        "total_count": total,
        "collapsed": len(entries) < total,
    }


# -------------------------------------------------------------------
# file_head — peek at text files
# -------------------------------------------------------------------

@tool
def file_head(path: str, n_bytes: int = 512) -> dict:
    """Read the first N bytes of a file as UTF-8 text.

    Useful for sniffing .txt / .yaml / .csv / .json files to identify
    them by content rather than just filename.

    Args:
        path: file to read
        n_bytes: how many bytes to read (default 512, max 4096)
    """
    p = _resolve(path)
    if not p.exists() or not p.is_file():
        return {"error": f"not a file: {p}"}
    n_bytes = min(max(n_bytes, 1), 4096)
    try:
        with open(p, "rb") as f:
            data = f.read(n_bytes)
        text = data.decode("utf-8", errors="replace")
        return {
            "path": str(p),
            "n_bytes_read": len(data),
            "text": text,
            "truncated": p.stat().st_size > n_bytes,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# -------------------------------------------------------------------
# npy_info — inspect numpy array files
# -------------------------------------------------------------------

@tool
def npy_info(path: str) -> dict:
    """Return shape + dtype of a .npy file (without loading the array data
    into memory; uses mmap).

    Tiebreaker for ambiguous .npy files: scale arrays are shape (3,);
    FoundationPose transforms are shape (4, 4); etc.
    """
    import numpy as np
    p = _resolve(path)
    if not p.exists() or not p.is_file():
        return {"error": f"not a file: {p}"}
    try:
        # mmap_mode='r' avoids loading the data — header only.
        arr = np.load(str(p), mmap_mode="r")
        return {
            "path": str(p),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# -------------------------------------------------------------------
# npz_info — inspect numpy .npz archives
# -------------------------------------------------------------------

@tool
def npz_info(path: str) -> dict:
    """Return the list of arrays inside an .npz archive (each with its
    shape + dtype). Does NOT load the array data into memory.

    Use this for files like `cam_extrinsics_*.npz` to see which camera
    IDs are encoded (e.g., keys like `cam_mat_20252535` and
    `cam_mat_25947356`), or any other multi-array dataset bundle.
    """
    import numpy as np
    p = _resolve(path)
    if not p.exists() or not p.is_file():
        return {"error": f"not a file: {p}"}
    try:
        with np.load(str(p), allow_pickle=False) as data:
            arrays = [
                {
                    "name": name,
                    "shape": list(data[name].shape),
                    "dtype": str(data[name].dtype),
                }
                for name in data.files
            ]
        return {"path": str(p), "arrays": arrays, "n_arrays": len(arrays)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# -------------------------------------------------------------------
# h5_info — inspect HDF5 files
# -------------------------------------------------------------------

@tool
def h5_info(path: str) -> dict:
    """Return top-level dataset names + their shapes/dtypes from an HDF5 file."""
    p = _resolve(path)
    if not p.exists() or not p.is_file():
        return {"error": f"not a file: {p}"}
    try:
        datasets = []
        with h5py.File(str(p), "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append({
                        "name": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    })
            f.visititems(visit)
        return {"path": str(p), "datasets": datasets}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


__all__ = ["list_dir", "file_head", "npy_info", "npz_info", "h5_info"]
