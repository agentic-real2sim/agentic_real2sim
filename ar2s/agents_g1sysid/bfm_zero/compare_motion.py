#!/usr/bin/env python3
"""
Compare a simulation recording NPZ against a reference motion NPZ.

── Trajectory mode (default) ────────────────────────────────────────────────
  Compare every frame of the recording against the corresponding reference frame.
  Good for evaluating motion tracking.

  python compare_motion.py recordings/sim_dance.npz example_motion/dance1_subject2_50_jpos.npz

  Optional:
    --frame-start 0 --frame-end 500   slice the reference to this range
    --no-root-align                   skip aligning root positions

── Single-frame mode (--ref-frame N) ────────────────────────────────────────
  Compare the LAST frame of the recording against frame N of the reference.
  Good for evaluating goal reaching: did the robot reach the target pose?

  python compare_motion.py recordings/goal_frame250.npz example_motion/dance1_subject2_50_jpos.npz \\
      --ref-frame 250

  Optional:
    --rec-frame -1   which recording frame to compare (default: -1 = last frame)
                     Use e.g. --rec-frame 0 to compare the first recorded frame.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert quaternions [w,x,y,z] (..., 4) → rotation matrices (..., 3, 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], axis=-1)
    return m.reshape(q.shape[:-1] + (3, 3))


def geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Geodesic rotation error in degrees. q: (..., 4) [w,x,y,z].

    Handles the quaternion double-cover: q and -q represent the same rotation,
    so we take the minimum of the angle and its supplement.
    """
    R1 = _quat_to_mat(q1)          # (..., 3, 3)
    R2 = _quat_to_mat(q2)
    Rrel = R1.swapaxes(-2, -1) @ R2          # R1^T · R2
    trace = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))       # (...,) in [0, 180]
    # q and -q give the same rotation; clamp to [0, 90] via symmetry
    return np.minimum(angle, 180.0 - angle)  # (...,) in [0, 90]


# ─── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(rec: dict, ref: dict, root_align: bool = True) -> dict:
    """
    Compute comparison metrics between a simulation recording and a reference motion.

    Both dicts must contain numpy arrays for:
      body_positions  (T, 30, 3)
      body_rotations  (T, 30, 4)  [w,x,y,z]
      dof_positions   (T, 29)

    Returns a dict of metric name → scalar or per-body/per-joint array.
    """
    T = min(rec["body_positions"].shape[0], ref["body_positions"].shape[0])
    bp_rec = rec["body_positions"][:T].astype(np.float64)   # (T, 30, 3)
    bp_ref = ref["body_positions"][:T].astype(np.float64)
    br_rec = rec["body_rotations"][:T].astype(np.float64)   # (T, 30, 4)
    br_ref = ref["body_rotations"][:T].astype(np.float64)
    dof_rec = rec["dof_positions"][:T].astype(np.float64)   # (T, 29)
    dof_ref = ref["dof_positions"][:T].astype(np.float64)

    # ── root-relative body positions (remove global drift) ──
    if root_align:
        root_rec = bp_rec[:, 0:1, :]   # (T, 1, 3) pelvis = body index 0
        root_ref = bp_ref[:, 0:1, :]
        bp_rec_rel = bp_rec - root_rec
        bp_ref_rel = bp_ref - root_ref
    else:
        bp_rec_rel = bp_rec
        bp_ref_rel = bp_ref

    # ── per-body Euclidean distance ──
    body_err = np.linalg.norm(bp_rec_rel - bp_ref_rel, axis=-1)  # (T, 30) metres
    per_body_mean = body_err.mean(axis=0)                          # (30,) metres
    mpjpe = body_err.mean()                                        # scalar metres

    # ── per-DOF angular error ──
    dof_err_deg = np.degrees(np.abs(dof_rec - dof_ref))           # (T, 29) degrees
    per_dof_mean = dof_err_deg.mean(axis=0)                        # (29,) degrees
    mean_dof_err = dof_err_deg.mean()                              # scalar degrees

    # ── body rotation error (geodesic) ──
    rot_err_deg = geodesic_deg(br_rec, br_ref)                    # (T, 30) degrees
    per_body_rot_mean = rot_err_deg.mean(axis=0)                   # (30,) degrees
    mean_rot_err = rot_err_deg.mean()                              # scalar degrees

    # ── root position drift (absolute, no alignment) ──
    root_drift = np.linalg.norm(bp_rec[:, 0, :] - bp_ref[:, 0, :], axis=-1)  # (T,)
    mean_root_drift = root_drift.mean()
    max_root_drift  = root_drift.max()

    return dict(
        T=T,
        # scalars
        mpjpe_m=mpjpe,
        mean_dof_err_deg=mean_dof_err,
        mean_rot_err_deg=mean_rot_err,
        mean_root_drift_m=mean_root_drift,
        max_root_drift_m=max_root_drift,
        # per-body / per-joint (for breakdown tables)
        per_body_pos_err_m=per_body_mean,
        per_body_rot_err_deg=per_body_rot_mean,
        per_dof_err_deg=per_dof_mean,
        root_drift_series=root_drift,
    )


def _quat_to_euler_zyx(q: np.ndarray) -> tuple[float, float, float]:
    """
    Convert a single quaternion [w,x,y,z] to intrinsic ZYX Euler angles (yaw, pitch, roll)
    in degrees.  Convention: R = Rz(yaw) · Ry(pitch) · Rx(roll).
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    return yaw, pitch, roll


def _angle_diff(a: float, b: float) -> float:
    """Signed difference a-b wrapped to (-180, 180]."""
    d = a - b
    return (d + 180.0) % 360.0 - 180.0


def _rz(deg: float) -> np.ndarray:
    """3×3 rotation matrix around the world Z axis by `deg` degrees."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def _rz_quat(deg: float) -> np.ndarray:
    """Unit quaternion [w,x,y,z] for a rotation of `deg` degrees around Z."""
    half = np.radians(deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two [w,x,y,z] quaternions (or arrays of them)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def apply_yaw_correction(body_positions: np.ndarray,
                          body_rotations: np.ndarray,
                          yaw_correction_deg: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate every body position and orientation by `yaw_correction_deg` around
    the world Z axis (applied after root subtraction).

    body_positions : (N, 3)
    body_rotations : (N, 4)  [w,x,y,z]

    Returns corrected copies of the same shape.
    """
    R = _rz(yaw_correction_deg)                        # (3, 3)
    pos_corr = (R @ body_positions.T).T                 # (N, 3)
    q_corr   = _rz_quat(yaw_correction_deg)            # (4,)
    rot_corr = _quat_mul(q_corr[None, :], body_rotations)  # (N, 4)
    # renormalise
    norms = np.linalg.norm(rot_corr, axis=-1, keepdims=True)
    rot_corr = rot_corr / np.where(norms > 0, norms, 1.0)
    return pos_corr, rot_corr


def compute_single_frame_metrics(rec_frame: dict, ref_frame: dict,
                                  root_align: bool = True) -> dict:
    """
    Compare one recorded pose against one reference pose.

    Each dict must contain 1-D / 2-D numpy arrays (no time dimension):
      body_positions  (30, 3)
      body_rotations  (30, 4)  [w,x,y,z]
      dof_positions   (29,)
    """
    bp_rec = rec_frame["body_positions"].astype(np.float64)   # (30, 3)
    bp_ref = ref_frame["body_positions"].astype(np.float64)
    br_rec = rec_frame["body_rotations"].astype(np.float64)   # (30, 4)
    br_ref = ref_frame["body_rotations"].astype(np.float64)
    dof_rec = rec_frame["dof_positions"].astype(np.float64)   # (29,)
    dof_ref = ref_frame["dof_positions"].astype(np.float64)

    if root_align:
        bp_rec = bp_rec - bp_rec[0:1, :]   # subtract pelvis
        bp_ref = bp_ref - bp_ref[0:1, :]

    per_body_pos_err = np.linalg.norm(bp_rec - bp_ref, axis=-1)   # (30,) metres
    mpjpe = per_body_pos_err.mean()

    per_dof_err_deg = np.degrees(np.abs(dof_rec - dof_ref))       # (29,) degrees
    mean_dof_err = per_dof_err_deg.mean()

    per_body_rot_err = geodesic_deg(br_rec, br_ref)                # (30,) degrees
    mean_rot_err = per_body_rot_err.mean()

    root_dist = float(np.linalg.norm(
        rec_frame["body_positions"][0] - ref_frame["body_positions"][0]
    ))

    # ── Pelvis Euler decomposition (ZYX: yaw / pitch / roll) ────────────────
    yaw_rec, pitch_rec, roll_rec = _quat_to_euler_zyx(br_rec[0])
    yaw_ref, pitch_ref, roll_ref = _quat_to_euler_zyx(br_ref[0])

    pelvis_euler = dict(
        rec=dict(yaw=yaw_rec, pitch=pitch_rec, roll=roll_rec),
        ref=dict(yaw=yaw_ref, pitch=pitch_ref, roll=roll_ref),
        diff=dict(
            yaw=_angle_diff(yaw_rec, yaw_ref),
            pitch=_angle_diff(pitch_rec, pitch_ref),
            roll=_angle_diff(roll_rec, roll_ref),
        ),
    )

    # ── Yaw-aligned metrics (cancel the facing-direction error) ─────────────
    yaw_corr_deg = _angle_diff(yaw_ref, yaw_rec)   # rotate rec BY this to match ref yaw
    bp_rec_ya, br_rec_ya = apply_yaw_correction(bp_rec, br_rec, yaw_corr_deg)

    per_body_pos_err_ya = np.linalg.norm(bp_rec_ya - bp_ref, axis=-1)
    per_body_rot_err_ya = geodesic_deg(br_rec_ya, br_ref)

    yaw_aligned = dict(
        mpjpe_m=per_body_pos_err_ya.mean(),
        mean_rot_err_deg=per_body_rot_err_ya.mean(),
        per_body_pos_err_m=per_body_pos_err_ya,
        per_body_rot_err_deg=per_body_rot_err_ya,
        yaw_correction_deg=yaw_corr_deg,
        # corrected arrays (useful for rendering)
        bp_rec=bp_rec_ya,
        br_rec=br_rec_ya,
    )

    return dict(
        mpjpe_m=mpjpe,
        mean_dof_err_deg=mean_dof_err,
        mean_rot_err_deg=mean_rot_err,
        root_dist_m=root_dist,
        per_body_pos_err_m=per_body_pos_err,
        per_body_rot_err_deg=per_body_rot_err,
        per_dof_err_deg=per_dof_err_deg,
        pelvis_euler=pelvis_euler,
        yaw_aligned=yaw_aligned,
    )


def format_single_frame_report(metrics: dict, rec_path: str, ref_path: str,
                                rec_frame_idx: int, ref_frame_idx: int,
                                body_names: list[str] | None = None,
                                dof_names: list[str] | None = None,
                                top_n: int = 5,
                                yaw_align: bool = True) -> str:
    """Return the full report as a string (printed to stdout and optionally saved)."""
    lines = []
    W = 64
    lines.append("=" * W)
    lines.append("Single-frame pose comparison  (goal-reaching accuracy)")
    lines.append(f"  Recording frame : {rec_frame_idx}  ({rec_path})")
    lines.append(f"  Reference frame : {ref_frame_idx}  ({ref_path})")
    lines.append("=" * W)
    lines.append("")

    lines.append("── Summary ──────────────────────────────────────────────────")
    lines.append(f"  MPJPE (root-relative body pos error)  : {metrics['mpjpe_m']*100:.2f} cm")
    lines.append(f"  Mean DOF angular error                 : {metrics['mean_dof_err_deg']:.2f} °")
    lines.append(f"  Mean body rotation error (geodesic)    : {metrics['mean_rot_err_deg']:.2f} °")
    lines.append(f"  Root position offset                   : {metrics['root_dist_m']*100:.2f} cm")
    lines.append("")

    # ── Pelvis Euler breakdown ─────────────────────────────────────────────────
    eu = metrics["pelvis_euler"]
    rec_e, ref_e, diff_e = eu["rec"], eu["ref"], eu["diff"]
    lines.append("── Pelvis orientation breakdown  (ZYX Euler, degrees) ───────")
    lines.append(f"  {'axis':<8}  {'reference':>10}  {'recorded':>10}  {'difference':>12}  {'dominant?':>10}")
    lines.append(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")
    axes = [("Yaw   (Z)", "yaw"), ("Pitch (Y)", "pitch"), ("Roll  (X)", "roll")]
    abs_diffs = {ax: abs(diff_e[ax]) for _, ax in axes}
    dominant = max(abs_diffs, key=abs_diffs.get)
    for label, ax in axes:
        flag = "◄ largest" if ax == dominant else ""
        lines.append(
            f"  {label:<8}  {ref_e[ax]:>10.2f}°  {rec_e[ax]:>10.2f}°  "
            f"{diff_e[ax]:>+12.2f}°  {flag}"
        )
    lines.append(f"  Note: geodesic total = {metrics['per_body_rot_err_deg'][0]:.2f}°  "
                 f"(combines all three axes)")
    lines.append("")

    pbe  = metrics["per_body_pos_err_m"]
    pbre = metrics["per_body_rot_err_deg"]

    lines.append(f"── Top-{top_n} bodies by position error ─────────────────────────")
    for i in np.argsort(pbe)[::-1][:top_n]:
        n = body_names[i] if body_names else f"body_{i}"
        lines.append(f"  [{i:2d}] {n:<35s}  {pbe[i]*100:.2f} cm  |  rot {pbre[i]:.1f}°")
    lines.append("")

    pde = metrics["per_dof_err_deg"]
    lines.append(f"── Top-{top_n} joints by angular error ──────────────────────────")
    for i in np.argsort(pde)[::-1][:top_n]:
        n = dof_names[i] if dof_names else f"dof_{i}"
        lines.append(f"  [{i:2d}] {n:<35s}  {pde[i]:.2f}°")
    lines.append("")

    if yaw_align:
        # ── Yaw-aligned summary ───────────────────────────────────────────────
        ya = metrics["yaw_aligned"]
        lines.append("── Yaw-aligned metrics  (facing-direction error cancelled) ──")
        lines.append(f"  Yaw correction applied                 : {ya['yaw_correction_deg']:+.2f}°")
        lines.append(f"  MPJPE (yaw-aligned)                    : {ya['mpjpe_m']*100:.2f} cm"
                     f"  (was {metrics['mpjpe_m']*100:.2f} cm)")
        lines.append(f"  Mean body rotation error (yaw-aligned) : {ya['mean_rot_err_deg']:.2f} °"
                     f"  (was {metrics['mean_rot_err_deg']:.2f} °)")
        lines.append("")

        # ── Full body breakdown (all 30) — raw and yaw-aligned ───────────────
        pbe_ya  = ya["per_body_pos_err_m"]
        pbre_ya = ya["per_body_rot_err_deg"]
        lines.append("── All bodies  (sorted by yaw-aligned position error) ───────")
        lines.append(f"  {'idx':>3}  {'body name':<35s}  {'raw pos':>8}  {'ya pos':>8}  {'ya rot':>7}")
        lines.append(f"  {'---':>3}  {'-'*35}  {'--------':>8}  {'--------':>8}  {'-------':>7}")
        for i in np.argsort(pbe_ya)[::-1]:
            n = body_names[i] if body_names else f"body_{i}"
            lines.append(
                f"  [{i:2d}] {n:<35s}  {pbe[i]*100:7.2f} cm  {pbe_ya[i]*100:7.2f} cm  {pbre_ya[i]:6.1f}°"
            )
        lines.append("")
    else:
        # plain body breakdown without yaw columns
        lines.append("── All bodies  (sorted by position error) ───────────────────")
        lines.append(f"  {'idx':>3}  {'body name':<35s}  {'pos err':>8}  {'rot err':>8}")
        lines.append(f"  {'---':>3}  {'-'*35}  {'--------':>8}  {'--------':>8}")
        for i in np.argsort(pbe)[::-1]:
            n = body_names[i] if body_names else f"body_{i}"
            lines.append(f"  [{i:2d}] {n:<35s}  {pbe[i]*100:7.2f} cm  {pbre[i]:7.1f}°")
        lines.append("")

    # ── Full DOF breakdown (all 29) ───────────────────────────────────────────
    lines.append("── All joints  (sorted by angular error) ────────────────────")
    lines.append(f"  {'idx':>3}  {'joint name':<35s}  {'ang err':>9}")
    lines.append(f"  {'---':>3}  {'-'*35}  {'---------':>9}")
    for i in np.argsort(pde)[::-1]:
        n = dof_names[i] if dof_names else f"dof_{i}"
        lines.append(f"  [{i:2d}] {n:<35s}  {pde[i]:8.2f}°")
    lines.append("")

    return "\n".join(lines)


def print_single_frame_report(metrics: dict, rec_path: str, ref_path: str,
                               rec_frame_idx: int, ref_frame_idx: int,
                               body_names: list[str] | None = None,
                               dof_names: list[str] | None = None,
                               top_n: int = 5,
                               save_path: str | None = None,
                               yaw_align: bool = True) -> None:
    report = format_single_frame_report(
        metrics, rec_path, ref_path, rec_frame_idx, ref_frame_idx,
        body_names=body_names, dof_names=dof_names, top_n=top_n,
        yaw_align=yaw_align,
    )
    print(report)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
        print(f"  Full report saved → {save_path}")


def print_report(metrics: dict, rec_path: str, ref_path: str,
                 body_names: list[str] | None = None,
                 dof_names: list[str] | None = None,
                 top_n: int = 5) -> None:
    T = metrics["T"]
    print("=" * 64)
    print(f"Motion comparison report")
    print(f"  Recording : {rec_path}")
    print(f"  Reference : {ref_path}")
    print(f"  Frames    : {T}")
    print("=" * 64)
    print()

    print("── Summary ──────────────────────────────────────────────────")
    print(f"  MPJPE (root-relative body pos error)  : {metrics['mpjpe_m']*100:.2f} cm")
    print(f"  Mean DOF angular error                 : {metrics['mean_dof_err_deg']:.2f} °")
    print(f"  Mean body rotation error (geodesic)    : {metrics['mean_rot_err_deg']:.2f} °")
    print(f"  Root position drift  (mean / max)      : "
          f"{metrics['mean_root_drift_m']*100:.2f} cm / {metrics['max_root_drift_m']*100:.2f} cm")
    print()

    # ── worst bodies by position error ──
    pbe = metrics["per_body_pos_err_m"]
    pbre = metrics["per_body_rot_err_deg"]
    print(f"── Top-{top_n} bodies by position error ─────────────────────────")
    idx = np.argsort(pbe)[::-1][:top_n]
    for i in idx:
        n = body_names[i] if body_names else f"body_{i}"
        print(f"  [{i:2d}] {n:<35s}  {pbe[i]*100:.2f} cm  |  rot {pbre[i]:.1f}°")
    print()

    # ── worst joints by DOF error ──
    pde = metrics["per_dof_err_deg"]
    print(f"── Top-{top_n} joints by angular error ──────────────────────────")
    idx = np.argsort(pde)[::-1][:top_n]
    for i in idx:
        n = dof_names[i] if dof_names else f"dof_{i}"
        print(f"  [{i:2d}] {n:<35s}  {pde[i]:.2f}°")
    print()


# ─── rendering ────────────────────────────────────────────────────────────────

def render_pose(env, body_positions: np.ndarray, body_rotations: np.ndarray,
                dof_positions: np.ndarray) -> np.ndarray:
    """
    Kinematically snap the robot into a pose and return a rendered RGB image (H, W, 3).

    body_positions : (30, 3)   pelvis is index 0
    body_rotations : (30, 4)   [w,x,y,z]
    dof_positions  : (29,)
    """
    env.reset()
    env.set_state(
        dof_positions=dof_positions.astype(np.float32),
        root_pos=body_positions[0].astype(np.float32),
        root_quat=body_rotations[0].astype(np.float32),
    )
    import mujoco
    if env.renderer is None:
        env.renderer = mujoco.Renderer(env.mjm, height=480, width=640)
    if env.camera is None:
        env.camera = mujoco.MjvCamera()
        env.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        env.camera.trackbodyid = env.track_body_id
        env.camera.distance = 3.0
        env.camera.elevation = -20.0
        env.camera.azimuth = 45.0
    env.renderer.update_scene(env.mjd, camera=env.camera)
    return env.renderer.render()   # (H, W, 3) uint8


def save_pose_comparison(ref_img: np.ndarray, rec_img: np.ndarray,
                          out_dir: str, ref_label: str, rec_label: str,
                          ya_img: np.ndarray | None = None,
                          ya_label: str = "") -> None:
    """Save individual images and a side-by-side comparison PNG.

    If ya_img is provided, a three-panel comparison (ref | raw | yaw-aligned)
    is saved as comparison_yaw_aligned.png in addition to comparison.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    ref_path = os.path.join(out_dir, "reference_pose.png")
    rec_path = os.path.join(out_dir, "recorded_pose.png")
    cmp_path = os.path.join(out_dir, "comparison.png")

    plt.imsave(ref_path, ref_img)
    plt.imsave(rec_path, rec_img)
    print(f"  Reference pose  → {ref_path}")
    print(f"  Recorded pose   → {rec_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(ref_img);  axes[0].set_title(ref_label, fontsize=11);  axes[0].axis("off")
    axes[1].imshow(rec_img);  axes[1].set_title(rec_label, fontsize=11);  axes[1].axis("off")
    fig.suptitle("Pose comparison (goal reaching vs. reference)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(cmp_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Side-by-side    → {cmp_path}")

    if ya_img is not None:
        ya_path = os.path.join(out_dir, "recorded_pose_yaw_aligned.png")
        cmp_ya_path = os.path.join(out_dir, "comparison_yaw_aligned.png")
        plt.imsave(ya_path, ya_img)
        print(f"  Yaw-aligned pose→ {ya_path}")

        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        axes[0].imshow(ref_img); axes[0].set_title(ref_label, fontsize=10);  axes[0].axis("off")
        axes[1].imshow(rec_img); axes[1].set_title(rec_label, fontsize=10);  axes[1].axis("off")
        axes[2].imshow(ya_img);  axes[2].set_title(ya_label,  fontsize=10);  axes[2].axis("off")
        fig.suptitle("Pose comparison — ref  |  raw recording  |  yaw-aligned recording",
                     fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(cmp_ya_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"  3-panel comparison → {cmp_ya_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Compare a sim recording NPZ to a reference motion NPZ")
    p.add_argument("recording", help="Path to recorded NPZ (from run_mujoco_env_demo.py --record)")
    p.add_argument("reference", help="Path to reference motion NPZ (e.g. example_motion/dance1...)")

    # ── trajectory mode ──
    p.add_argument("--frame-start", type=int, default=0,   help="Reference frame start (trajectory mode)")
    p.add_argument("--frame-end",   type=int, default=None, help="Reference frame end (trajectory mode)")
    p.add_argument("--no-root-align", action="store_true",
                   help="Do NOT subtract root position before computing body pos error")
    p.add_argument("--top", type=int, default=5, help="Show top-N worst bodies/joints (default 5)")

    # ── single-frame mode ──
    p.add_argument("--ref-frame", type=int, default=None,
                   help="Single-frame mode: compare recording frame --rec-frame against "
                        "this reference frame index. Good for goal-reaching evaluation.")
    p.add_argument("--rec-frame", type=int, default=-1,
                   help="Which recording frame to use in single-frame mode "
                        "(default: -1 = last frame, i.e. end of goal-reaching rollout).")
    p.add_argument("--render-dir", default=None,
                   help="(Single-frame mode only) Render both poses and save PNG images to "
                        "this directory: reference_pose.png, recorded_pose.png, comparison.png.")
    p.add_argument("--no-yaw-align", action="store_true",
                   help="Skip the yaw-aligned metrics section and yaw-corrected render.")
    p.add_argument("--save-report", default=None,
                   help="Save the full per-joint / per-body report to this .txt file. "
                        "If --render-dir is set and --save-report is not, a report.txt is "
                        "saved automatically inside the render dir.")

    args = p.parse_args()

    rec_raw = np.load(args.recording)
    ref_raw = np.load(args.reference)

    # body / dof names
    try:
        body_names = list(rec_raw["body_names"])
    except KeyError:
        body_names = None
    try:
        dof_names = list(rec_raw["dof_names"])
    except KeyError:
        try:
            dof_names = list(ref_raw["dof_names"])
        except KeyError:
            dof_names = None

    root_align = not args.no_root_align

    if args.ref_frame is not None:
        # ── Single-frame mode: goal-reaching accuracy ──
        T_rec = rec_raw["body_positions"].shape[0]
        rec_idx = args.rec_frame % T_rec   # support negative indexing

        rec_frame = {
            "body_positions": rec_raw["body_positions"][rec_idx],
            "body_rotations": rec_raw["body_rotations"][rec_idx],
            "dof_positions":  rec_raw["dof_positions"][rec_idx],
        }
        ref_frame = {
            "body_positions": ref_raw["body_positions"][args.ref_frame],
            "body_rotations": ref_raw["body_rotations"][args.ref_frame],
            "dof_positions":  ref_raw["dof_positions"][args.ref_frame],
        }
        metrics = compute_single_frame_metrics(rec_frame, ref_frame, root_align=root_align)

        # auto-place report.txt inside --render-dir if --save-report not explicitly given
        report_path = args.save_report
        if report_path is None and args.render_dir:
            report_path = os.path.join(args.render_dir, "report.txt")

        yaw_align = not args.no_yaw_align
        print_single_frame_report(
            metrics, args.recording, args.reference,
            rec_frame_idx=rec_idx, ref_frame_idx=args.ref_frame,
            body_names=body_names, dof_names=dof_names, top_n=args.top,
            save_path=report_path,
            yaw_align=yaw_align,
        )

        if args.render_dir:
            print(f"\nRendering poses into {args.render_dir} …")
            # lazy import so rendering deps aren't needed in trajectory mode
            repo_root = os.path.dirname(os.path.abspath(__file__))
            os.chdir(repo_root)
            import sys as _sys
            if repo_root not in _sys.path:
                _sys.path.insert(0, repo_root)
            from common import ACTION_RESCALE, ACTION_SCALES, DEFAULT_JOINT_POS, KD_GAINS, KP_GAINS
            from env import MuJoCoBFMZeroEnv

            ROBOT_XML = os.path.join(repo_root, "bfm_zero_inference_code/g1_for_reward_inference.xml")
            env = MuJoCoBFMZeroEnv(
                robot_xml=ROBOT_XML,
                kp_gains=KP_GAINS, kd_gains=KD_GAINS,
                default_joint_pos=DEFAULT_JOINT_POS,
                action_scales=ACTION_SCALES, action_rescale=ACTION_RESCALE,
                enable_video=True,   # needed to initialise renderer
                video_path="/tmp/_cmp_dummy.mp4",
            )

            ref_img = render_pose(env,
                                  ref_raw["body_positions"][args.ref_frame],
                                  ref_raw["body_rotations"][args.ref_frame],
                                  ref_raw["dof_positions"][args.ref_frame])

            rec_img = render_pose(env,
                                  rec_raw["body_positions"][rec_idx],
                                  rec_raw["body_rotations"][rec_idx],
                                  rec_raw["dof_positions"][rec_idx])

            # Yaw-aligned render (skipped when --no-yaw-align)
            if not yaw_align:
                save_pose_comparison(
                    ref_img, rec_img, args.render_dir,
                    ref_label=f"Reference  frame {args.ref_frame}  ({os.path.basename(args.reference)})",
                    rec_label=f"Recording  frame {rec_idx}  ({os.path.basename(args.recording)})",
                )
                return 0

            # Yaw-aligned render: rotate the recording's pelvis by the yaw correction,
            # then render it.  Positions are root-relative in the metric, but for
            # rendering we need world-space: add the reference pelvis world position back.
            ya      = metrics["yaw_aligned"]
            yaw_deg = ya["yaw_correction_deg"]
            ref_pelvis_world = ref_raw["body_positions"][args.ref_frame, 0].astype(np.float64)
            rec_pelvis_world = rec_raw["body_positions"][rec_idx, 0].astype(np.float64)

            # root-relative corrected positions (already computed in metrics)
            bp_ya = ya["bp_rec"]                           # (30, 3) root-relative
            br_ya = ya["br_rec"]                           # (30, 4)
            # put pelvis back at the reference world position for a fair visual
            bp_ya_world = bp_ya + ref_pelvis_world[None, :]
            ya_img = render_pose(env, bp_ya_world, br_ya,
                                 rec_raw["dof_positions"][rec_idx])

            corr_str = f"{yaw_deg:+.1f}°"
            save_pose_comparison(
                ref_img, rec_img, args.render_dir,
                ref_label=f"Reference  frame {args.ref_frame}  ({os.path.basename(args.reference)})",
                rec_label=f"Recording  frame {rec_idx}  ({os.path.basename(args.recording)})",
                ya_img=ya_img,
                ya_label=f"Yaw-aligned ({corr_str} correction)  frame {rec_idx}",
            )
    else:
        # ── Trajectory mode: full sequence comparison ──
        T_rec = rec_raw["body_positions"].shape[0]
        frame_end = args.frame_end if args.frame_end is not None else args.frame_start + T_rec

        ref = {
            "body_positions": ref_raw["body_positions"][args.frame_start:frame_end],
            "body_rotations": ref_raw["body_rotations"][args.frame_start:frame_end],
            "dof_positions":  ref_raw["dof_positions"][args.frame_start:frame_end],
        }
        rec = {
            "body_positions": rec_raw["body_positions"],
            "body_rotations": rec_raw["body_rotations"],
            "dof_positions":  rec_raw["dof_positions"],
        }
        metrics = compute_metrics(rec, ref, root_align=root_align)
        print_report(metrics, args.recording, args.reference,
                     body_names=body_names, dof_names=dof_names, top_n=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
