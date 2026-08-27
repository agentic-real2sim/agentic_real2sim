"""Convert retargeted LAFAN1 clips into reference-motion NPZ files.

Input: a Unitree ``LAFAN1_Retargeting_Dataset`` G1 CSV — one row per frame at
30 fps: ``root x,y,z, qx,qy,qz,qw`` followed by 29 joint angles in
``POLICY_JOINT_NAMES`` order (the dataset's documented order matches the
bundled G1 MJCF exactly; validated at runtime).

Output: the ``save_recording`` NPZ format consumed by MotionTrackingReward /
compare_motion — body poses from MuJoCo FK over the bundled robot XML,
resampled to 50 fps because the sysid loop consumes one reference frame per
50 Hz control step (``sysid._z_at_step`` / ``track_ref_frame`` index by step).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BFM_ROOT = Path(__file__).resolve().parent / "bfm_zero"
_ROBOT_XML = Path(__file__).resolve().parent / "assets" / "robot" / "g1_for_reward_inference.xml"

CONTROL_FPS = 50.0
LAFAN_FPS = 30.0


def _ensure_bfm_on_path() -> None:
    p = str(_BFM_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_unitree_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a Unitree LAFAN1-retargeting G1 CSV.

    Returns:
        root_pos (T,3), root_quat (T,4) w-x-y-z (MuJoCo order), dof_pos (T,29).
    """
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 + 4 + 29:
        raise ValueError(
            f"{csv_path}: expected (T, 36) = root xyz + quat xyzw + 29 dofs, "
            f"got {raw.shape}"
        )
    root_pos = raw[:, 0:3]
    qx, qy, qz, qw = raw[:, 3], raw[:, 4], raw[:, 5], raw[:, 6]
    root_quat = np.stack([qw, qx, qy, qz], axis=1)  # xyzw -> wxyz
    dof_pos = raw[:, 7:36]
    return root_pos, root_quat, dof_pos


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Batched quaternion slerp. q0/q1: (T,4) unit wxyz; t: (T,) in [0,1]."""
    dot = np.sum(q0 * q1, axis=1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)  # shorter arc
    dot = np.abs(dot).clip(max=1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    t = t[:, None]
    # Fall back to lerp for nearly-parallel quaternions.
    near = sin_theta < 1e-6
    w0 = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(near, 1.0, sin_theta))
    w1 = np.where(near, t, np.sin(t * theta) / np.where(near, 1.0, sin_theta))
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def resample(root_pos: np.ndarray, root_quat: np.ndarray, dof_pos: np.ndarray,
             src_fps: float, dst_fps: float = CONTROL_FPS,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a motion to dst_fps: linear interp for positions/angles, slerp
    for the root quaternion. The span is truncated to the last whole 1/dst_fps
    step; when duration*dst_fps is integral the final source frame is kept."""
    T = root_pos.shape[0]
    duration = (T - 1) / src_fps
    # The epsilon keeps exactly-integral products (e.g. same-fps passthrough,
    # 29/50*50 == 28.999...96 in floats) from flooring one frame short.
    n_out = int(np.floor(duration * dst_fps + 1e-6)) + 1
    t_out = np.arange(n_out) / dst_fps * src_fps  # in source-frame units
    i0 = np.floor(t_out).astype(int).clip(max=T - 1)
    i1 = np.minimum(i0 + 1, T - 1)
    frac = t_out - i0

    def lerp(x: np.ndarray) -> np.ndarray:
        return x[i0] + (x[i1] - x[i0]) * frac.reshape((-1,) + (1,) * (x.ndim - 1))

    return lerp(root_pos), _slerp(root_quat[i0], root_quat[i1], frac), lerp(dof_pos)


def write_reference_npz(out_path: str | Path,
                        root_pos: np.ndarray, root_quat: np.ndarray,
                        dof_pos: np.ndarray, fps: float = CONTROL_FPS) -> dict:
    """FK the motion through the bundled G1 MJCF and write the reference NPZ.

    Returns a small summary dict (T, n_bodies, duration_s, out_path).
    """
    import mujoco

    _ensure_bfm_on_path()
    from common import POLICY_JOINT_NAMES  # type: ignore
    from run_mujoco_env_demo import _recording_body_indices, save_recording  # type: ignore

    mjm = mujoco.MjModel.from_xml_path(str(_ROBOT_XML))
    mjd = mujoco.MjData(mjm)

    xml_joints = [mjm.joint(i).name for i in range(1, mjm.njnt)]  # skip free joint
    if xml_joints != list(POLICY_JOINT_NAMES):
        raise RuntimeError(
            "robot XML joint order != POLICY_JOINT_NAMES — refusing to convert:\n"
            f"  xml   : {xml_joints}\n  policy: {list(POLICY_JOINT_NAMES)}"
        )
    body_idx, body_names = _recording_body_indices(mjm)

    T = root_pos.shape[0]
    dof_vel = np.zeros_like(dof_pos)
    dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) * fps

    frames = []
    for i in range(T):
        mjd.qpos[0:3] = root_pos[i]
        mjd.qpos[3:7] = root_quat[i]
        mjd.qpos[7:] = dof_pos[i]
        mujoco.mj_kinematics(mjm, mjd)
        frames.append({
            "body_positions": mjd.xpos[body_idx].copy(),
            "body_rotations": mjd.xquat[body_idx].copy(),
            "dof_positions": dof_pos[i].copy(),
            "dof_velocities": dof_vel[i].copy(),
            "root_pos": root_pos[i].copy(),
            "root_quat": root_quat[i].copy(),
        })

    save_recording(str(out_path), frames, fps, list(POLICY_JOINT_NAMES), body_names)
    return {
        "T": T,
        "n_bodies": len(body_idx),
        "duration_s": round((T - 1) / fps, 2),
        "out_path": str(out_path),
    }


def convert_lafan_csv(csv_path: str | Path, out_path: str | Path,
                      *, src_fps: float = LAFAN_FPS, dst_fps: float = CONTROL_FPS,
                      frame_start: int = 0, frame_end: int | None = None) -> dict:
    """CSV -> (slice) -> resample to dst_fps -> FK -> reference NPZ."""
    root_pos, root_quat, dof_pos = load_unitree_csv(csv_path)
    sl = slice(frame_start, frame_end)
    root_pos, root_quat, dof_pos = root_pos[sl], root_quat[sl], dof_pos[sl]
    if root_pos.shape[0] < 2:
        raise ValueError(f"clip slice too short: {root_pos.shape[0]} frame(s)")
    root_pos, root_quat, dof_pos = resample(
        root_pos, root_quat, dof_pos, src_fps, dst_fps,
    )
    return write_reference_npz(out_path, root_pos, root_quat, dof_pos, dst_fps)
