"""Shared mesh utilities.

Centralizes per-object geometry reads so `ground_alignment`,
`scene_calibrate`, and any future module don't each reimplement them.

Effective scale = base scale (from `scale_path`) × `scale_multiplier`.
The multiplier is the agent-tunable calibration delta (default identity),
base is the upstream data. Anything that needs the "real" scale applied
to the mesh should go through `get_effective_scale`.
"""
from pathlib import Path

import numpy as np

from ar2s.droid_sim.scene.config import SceneObject


def load_obj_vertices(path: str) -> np.ndarray:
    """Minimal .obj vertex reader — returns (N, 3) float64."""
    verts: list[list[float]] = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices found in {path}")
    return np.array(verts, dtype=np.float64)


def get_effective_scale(obj: SceneObject) -> np.ndarray:
    """Return base scale × scale_multiplier, shape (3,)."""
    base = np.asarray(np.load(obj.scale_path), dtype=np.float64).reshape(3)
    mult = np.asarray(obj.scale_multiplier, dtype=np.float64).reshape(3)
    return base * mult


def compute_object_dimensions(obj: SceneObject, local_down: tuple[float, float, float]
                              ) -> dict:
    """Summarize an object's geometry for the calibration agent.

    Returns dims in metres and the bottom-depth along local_down (same
    quantity used for theoretical ground offset)."""
    scale = get_effective_scale(obj)
    verts = load_obj_vertices(obj.visual_mesh_path) * scale
    dims = (verts.max(axis=0) - verts.min(axis=0)).astype(float)
    ld = np.asarray(local_down, dtype=np.float64)
    ld = ld / max(np.linalg.norm(ld), 1e-12)
    bottom_depth = float(np.max(verts @ ld))
    return {
        "dims_m": (float(dims[0]), float(dims[1]), float(dims[2])),
        "bottom_depth_m": bottom_depth,
    }
