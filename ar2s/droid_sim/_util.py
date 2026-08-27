"""Shared utilities for droid_sim + downstream agent packages.

Module-level imports stay LIGHT (just stdlib). Helpers needing numpy /
mujoco lazy-import inside the function so importing this module is cheap.
Consumers can `from ar2s.droid_sim._util import sanitize_name` without paying
for heavy deps they don't use.

Catalogued by the agents_visual cleanup audit
(docs/agents_visual_cleanup_zh.html).
"""
from __future__ import annotations

import re
from pathlib import Path


# =========================================================================
# Filesystem-safe object names
# =========================================================================

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_SNAKE_RE = re.compile(r"[^a-z0-9]+")


def sanitize_name(text: str) -> str:
    """Replace any run of non-[A-Za-z0-9._-] chars with a single underscore,
    strip leading/trailing underscores, fall back to "unnamed" for empties.

    Used as the canonical path-safe form for object / text-prompt names that
    must round-trip through filesystem paths (SAM3 mask dirs, FoundationPose
    output dirs, collision_decompose piece filenames, episode bundle dirs).

    Examples:
        "snack package"       -> "snack_package"
        "pot lid"             -> "pot_lid"
        "object   #1"         -> "object_1"
        ""                    -> "unnamed"

    CALLER CONTRACT: must produce byte-identical output to the duplicate
    copy in ``ar2s/agents_visual/_toolkit/scripts/run_sam3_segment.py`` —
    that script lives in the qianjun_sam3 subprocess env (which may not have
    droid_sim installed) so it cannot import this module directly. Any
    change here MUST be mirrored there.
    """
    return _SANITIZE_RE.sub("_", text).strip("_") or "unnamed"


def snake_case_name(text: str) -> str:
    """Canonical object identifier: lowercase words joined by underscores."""
    return _SNAKE_RE.sub("_", text.strip().lower()).strip("_") or "unnamed"


# =========================================================================
# FoundationPose output directory scanning
# =========================================================================

def earliest_pose_frame(pose_dir: Path | str) -> int | None:
    """Return the smallest frame index in a FoundationPose output, or None.

    visual's per-object keyframe means different objects can start at
    different frames (e.g. snack_package starting at frame 2 because SAM3
    couldn't segment it earlier). Sysid + outputs.py read this to either
    backfill earlier frames or to set the manifest's ``init_frame``.

    Thin re-export of ``ar2s.droid_sim.pose_store.earliest_frame``, which
    handles both the packed ``poses.npz`` and the legacy per-frame
    ``frame_NNNNNN_transform.npy`` layout.
    """
    from ar2s.droid_sim.pose_store import earliest_frame
    return earliest_frame(pose_dir)


# =========================================================================
# OpenCV ↔ MuJoCo camera conversion
# =========================================================================

# OpenCV camera convention: +Z look (forward), +X right, +Y down (image axes).
# MuJoCo  camera convention: -Z look (forward), +X right, +Y up.
# Conversion = 180° rotation about the camera's local X axis = diag(1, -1, -1)
# applied on the right of the world-from-camera rotation.

def opencv_c2w_to_mujoco_camera_pose(c2w_4x4):
    """OpenCV-convention camera world pose (4x4 c2w) → MuJoCo (pos, quat_wxyz).

    Returns ``(pos: np.ndarray(3,), quat_wxyz: np.ndarray(4,))`` ready to
    plug into ``mjSpec.cam.pos`` / ``cam.quat`` or any equivalent.

    Same logic as RobotSceneSim:444 / RobotOnlySim:168 / mesh_orient:195.
    """
    import numpy as np
    import mujoco
    c2w = np.asarray(c2w_4x4, dtype=np.float64)
    R_flip = np.diag([1.0, -1.0, -1.0])
    R_mj = c2w[:3, :3] @ R_flip
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, R_mj.flatten())
    return c2w[:3, 3].copy(), quat


def fovy_deg_from_K(K_3x3, image_height: int | float) -> float:
    """Vertical field of view in degrees from a 3x3 intrinsics matrix.

    ``image_height`` is whatever height MuJoCo's renderer is configured for
    (i.e. the offscreen frame height we'll render to). Callers using the
    "render-square-then-crop-to-K-principal-point" pattern pass S (the
    square side); callers rendering at the native sensor size pass H.
    """
    import numpy as np
    fy = float(K_3x3[1, 1])
    return float(np.degrees(2.0 * np.arctan(image_height / (2.0 * fy))))


__all__ = [
    "earliest_pose_frame",
    "fovy_deg_from_K",
    "opencv_c2w_to_mujoco_camera_pose",
    "sanitize_name",
]
