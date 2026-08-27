"""Read-only inspection skills for motion .npz files and latent .pkl files.

All tools are stateless and raise on error rather than returning partial data
so the LLM never acts on bad information.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# NPZ inspection
# ---------------------------------------------------------------------------

@tool
def inspect_npz(path: str) -> str:
    """Inspect a NumPy .npz file and return a JSON summary.

    Returns a JSON object with:
      keys        - list of array names in the archive
      shapes      - dict mapping each key to its shape
      dtypes      - dict mapping each key to its dtype string
      n_frames    - value of shape[0] for the key named "dof_positions"
                    (or the first key if "dof_positions" is absent)
      dof_names   - list of joint names if a "dof_names" key exists, else null
      body_names  - list of body names if a "body_names" key exists, else null
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return json.dumps({"error": f"File not found: {p}"})
    if p.suffix != ".npz":
        return json.dumps({"error": f"Not a .npz file: {p}"})

    import numpy as np
    try:
        data = np.load(str(p), allow_pickle=True)
    except Exception as e:
        return json.dumps({"error": f"Failed to load {p}: {e}"})

    keys = list(data.files)
    shapes = {k: list(data[k].shape) for k in keys}
    dtypes = {k: str(data[k].dtype) for k in keys}

    # Prefer "dof_positions" for frame count; fall back to first key
    frame_key = "dof_positions" if "dof_positions" in keys else (keys[0] if keys else None)
    n_frames = int(data[frame_key].shape[0]) if frame_key else None

    dof_names = (
        [str(n) for n in data["dof_names"].tolist()]
        if "dof_names" in keys else None
    )
    body_names = (
        [str(n) for n in data["body_names"].tolist()]
        if "body_names" in keys else None
    )

    return json.dumps({
        "path": str(p),
        "keys": keys,
        "shapes": shapes,
        "dtypes": dtypes,
        "n_frames": n_frames,
        "dof_names": dof_names,
        "body_names": body_names,
    }, indent=2)


# ---------------------------------------------------------------------------
# PKL inspection
# ---------------------------------------------------------------------------

@tool
def inspect_pkl(path: str) -> str:
    """Inspect a latent-context .pkl file and return a JSON summary.

    Handles two common formats:
      1. Raw ndarray (N, z_dim) - from zs_3.pkl style exports
      2. Dict with key "z"      - from single-goal exports (z_exports/)

    Returns a JSON object with:
      structure   - "array" | "dict"
      n_z_rows    - number of rows (N) in the array
      z_dim       - latent dimension
      dtype       - dtype string
      dict_keys   - (dict only) list of keys present
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return json.dumps({"error": f"File not found: {p}"})

    import joblib
    import numpy as np
    try:
        raw = joblib.load(str(p))
    except Exception as e:
        return json.dumps({"error": f"Failed to load {p}: {e}"})

    if isinstance(raw, dict):
        z = raw.get("z")
        if z is None:
            return json.dumps({
                "path": str(p),
                "structure": "dict",
                "dict_keys": list(raw.keys()),
                "error": "No 'z' key found in dict pkl",
            })
        arr = np.asarray(z)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return json.dumps({
            "path": str(p),
            "structure": "dict",
            "dict_keys": list(raw.keys()),
            "n_z_rows": int(arr.shape[0]),
            "z_dim": int(arr.shape[1]),
            "dtype": str(arr.dtype),
        }, indent=2)

    # Raw array
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return json.dumps({
        "path": str(p),
        "structure": "array",
        "n_z_rows": int(arr.shape[0]),
        "z_dim": int(arr.shape[1]),
        "dtype": str(arr.dtype),
    }, indent=2)


# ---------------------------------------------------------------------------
# Directory listing (thin wrapper kept here for convenience)
# ---------------------------------------------------------------------------

@tool
def list_motion_dir(path: str) -> str:
    """List files in a directory, highlighting .npz, .pkl, and .onnx files.

    Returns a JSON object with:
      files  - list of {name, size_kb, ext} dicts
      counts - dict of extension -> count
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return json.dumps({"error": f"Path not found: {p}"})
    if not p.is_dir():
        return json.dumps({"error": f"Not a directory: {p}"})

    files = []
    for f in sorted(p.iterdir()):
        if f.is_file():
            files.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "ext": f.suffix.lower(),
            })

    counts: dict[str, int] = {}
    for f in files:
        counts[f["ext"]] = counts.get(f["ext"], 0) + 1

    return json.dumps({"path": str(p), "files": files, "counts": counts}, indent=2)


__all__ = ["inspect_npz", "inspect_pkl", "list_motion_dir"]
