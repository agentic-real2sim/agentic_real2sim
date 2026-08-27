"""Phase-5 group-shift helpers.

User specifies `(dx, dy, dz)` in a basis that's natural for tabletop
grasping but NOT plain world frame:

    dx → projection of world +x onto the ground plane (a vector in-plane)
    dy → projection of world +y onto the ground plane (a vector in-plane)
    dz → world +z                                        (vertical)

The ground plane has normal aligned with the reference object's
local_up direction (mug's local_up), which is approximately — but not
exactly — world +z (FoundationPose tilt of ~10° is typical).

Effects of a shift `(dx, dy, dz)`:
  * Objects translate by  `dx*e1 + dy*e2 + dz*world_z`  in world coords
    (so dy on a tilted table also rises slightly in world z).
  * Ground translates by  `(0, 0, dz)`  in world coords (only vertical;
    its xy is fixed; its normal is unchanged).
"""
import numpy as np


def ground_normal_world(ref_R: np.ndarray, local_down_axis: tuple[float, float, float]) -> np.ndarray:
    """Mug local_up direction expressed in world frame (= ground normal).

    Args:
        ref_R: (3, 3) rotation matrix of the reference object (mug) in world.
        local_down_axis: down direction in mug's local frame, e.g. (0, -1, 0).

    Returns:
        (3,) unit vector. Points roughly along world +z when ref is upright.
    """
    ld = np.asarray(local_down_axis, dtype=np.float64)
    ld /= max(np.linalg.norm(ld), 1e-12)
    n = -ref_R @ ld
    return n / max(np.linalg.norm(n), 1e-12)


def shift_to_world_disp_obj(
    shift: tuple[float, float, float],
    ground_normal: np.ndarray,
) -> np.ndarray:
    """Convert user-spec `(dx, dy, dz)` to a world-frame displacement vector
    for OBJECTS.

    Basis:
        e1 = unit(world_x − (world_x · n) * n)          # world_x ⟂-projected on ground
        e2 = unit(world_y − (world_y · n) * n)          # world_y ⟂-projected on ground
        e3 = world_z

    Returns shape (3,).
    """
    dx, dy, dz = shift
    n = np.asarray(ground_normal, dtype=np.float64)
    n /= max(np.linalg.norm(n), 1e-12)

    wx = np.array([1.0, 0.0, 0.0])
    wy = np.array([0.0, 1.0, 0.0])
    wz = np.array([0.0, 0.0, 1.0])

    e1 = wx - np.dot(wx, n) * n
    e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = wy - np.dot(wy, n) * n
    e2 /= max(np.linalg.norm(e2), 1e-12)

    return float(dx) * e1 + float(dy) * e2 + float(dz) * wz


def shift_to_world_disp_ground(shift: tuple[float, float, float]) -> np.ndarray:
    """Ground translates only in world +z by `dz`. xy unchanged."""
    return np.array([0.0, 0.0, float(shift[2])], dtype=np.float64)


def world_disp_to_shift(
    v_world: np.ndarray,
    ground_normal: np.ndarray,
) -> tuple[float, float, float]:
    """Inverse of `shift_to_world_disp_obj` — express a world-frame
    displacement vector in the user-spec basis ``(dx, dy, dz)``.

    Solves ``v_world = dx*e1 + dy*e2 + dz*e3`` for ``(dx, dy, dz)`` where
    ``e1 e2 e3`` are the basis defined in ``shift_to_world_disp_obj``.
    Works for any non-vertical ground (small FoundationPose tilt fine).
    """
    n = np.asarray(ground_normal, dtype=np.float64)
    n /= max(np.linalg.norm(n), 1e-12)

    wx = np.array([1.0, 0.0, 0.0])
    wy = np.array([0.0, 1.0, 0.0])
    wz = np.array([0.0, 0.0, 1.0])

    e1 = wx - np.dot(wx, n) * n
    e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = wy - np.dot(wy, n) * n
    e2 /= max(np.linalg.norm(e2), 1e-12)
    e3 = wz

    M = np.column_stack([e1, e2, e3])
    s = np.linalg.solve(M, np.asarray(v_world, dtype=np.float64))
    return (float(s[0]), float(s[1]), float(s[2]))
