"""Ground plane derivation helpers.

Two flavours:

- ``derive_ground_pose_world_xy`` (current production path): place the ground
  as MuJoCo's world XY plane at ``z = phase5_dz - ground.offset``. The
  ``geometry_prior`` Stage 0 (``ar2s.agents_geometry_prior``) fills the
  manifest with the right ``ground.offset`` BEFORE sysid runs, so all
  downstream code can read it as truth. This is the only path the
  pipeline uses today.

- ``derive_ground_pose`` (legacy): place the ground along a reference
  object's local-down axis at a signed offset. Used by the v1 alignment
  loop before Stage 0 existed. Kept for the unit test in
  ``tests/test_phase4_ground_alignment.py`` and as a reference for the
  geometry of FP-derived ground placement, but no production caller exists.

See ``GroundSpec`` in ``ar2s/droid_sim/scene/config.py``.
"""
import mujoco
import numpy as np

from ar2s.droid_sim.scene.config import GroundSpec


def derive_ground_pose_world_xy(
    ground_spec: GroundSpec,
    phase5_dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, quat_wxyz) for a world-XY ground plane at z driven by offset.

    With Stage 0 filling ``ground.offset`` from z_anchor's BFS propagation,
    the ground is simply ``MuJoCo world XY plane`` at
    ``z = phase5_dz - offset``. The sign matches the legacy
    ``derive_ground_pose`` convention (positive ``offset`` moves the plane
    along ``local_down_axis`` ≈ world -Z), so ``set_ground_offset`` callers
    behave identically.

    Args:
        ground_spec: ``GroundSpec`` with ``.offset`` (and unused other fields
            in this path; ``local_down_axis`` is honoured only insofar as the
            plane normal lives in ``-local_down``).
        phase5_dz: the sim's phase-5 z-shift (whole scene lifts uniformly
            after grasp-shift; ground tracks it so the scene-to-ground
            relationship is invariant under shift sweeps).

    Returns:
        pos: (3,) world ground position. xy = 0.
        quat_wxyz: (4,) plane normal = +world Z (identity quat for MuJoCo
            planes; default plane normal IS local +Z which equals world +Z
            with identity orientation).
    """
    ground_z = float(phase5_dz) - float(ground_spec.offset)
    pos = np.array([0.0, 0.0, ground_z], dtype=np.float64)
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return pos, quat_wxyz


def derive_ground_pose(
    ref_pos: np.ndarray,
    ref_quat_wxyz: np.ndarray,
    local_down_axis: tuple[float, float, float] | np.ndarray,
    offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, quat_wxyz) for a ground plane aligned to a reference object.

    Args:
        ref_pos: (3,) reference object position in world frame.
        ref_quat_wxyz: (4,) reference object orientation in world frame.
        local_down_axis: (3,) "down" direction in the reference object's
            local frame, e.g. (0, -1, 0) for a mug whose opening points +Y.
        offset: signed distance along local_down_axis from ref_pos to ground.

    Returns:
        pos: (3,) ground plane position.
        quat_wxyz: (4,) ground orientation (MuJoCo plane normal = local Z).
    """
    ref_quat_wxyz = np.asarray(ref_quat_wxyz, dtype=np.float64)
    ref_pos = np.asarray(ref_pos, dtype=np.float64)

    ref_R = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(ref_R, ref_quat_wxyz)
    ref_R = ref_R.reshape(3, 3)

    local_down = np.asarray(local_down_axis, dtype=np.float64)
    local_down /= max(np.linalg.norm(local_down), 1e-12)
    local_up = -local_down

    # MuJoCo plane normal is the plane's local Z. We want plane_Z_world to equal
    # ref_R @ local_up. Equivalently, find a quat in the reference frame that
    # takes [0,0,1] to local_up, then compose with the reference's world quat.
    z_axis = np.array([0.0, 0.0, 1.0])
    cos_a = float(np.clip(np.dot(z_axis, local_up), -1.0, 1.0))
    if cos_a > 1.0 - 1e-9:
        quat_z_to_up = np.array([1.0, 0.0, 0.0, 0.0])
    elif cos_a < -1.0 + 1e-9:
        # 180° flip around any perpendicular axis
        quat_z_to_up = np.array([0.0, 1.0, 0.0, 0.0])
    else:
        axis = np.cross(z_axis, local_up)
        axis /= np.linalg.norm(axis)
        angle = float(np.arccos(cos_a))
        quat_z_to_up = np.array([
            np.cos(angle / 2),
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2),
        ])

    quat_plane = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(quat_plane, ref_quat_wxyz, quat_z_to_up)

    pos = ref_pos + ref_R @ (local_down * offset)
    return pos, quat_plane
