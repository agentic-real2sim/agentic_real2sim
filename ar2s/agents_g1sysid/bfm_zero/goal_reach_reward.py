#!/usr/bin/env python3
"""
Goal-reaching reward function for BFM-Zero RL fine-tuning.

Designed around the two metrics that best distinguish normal vs. broken physics
from the compare_motion analysis:

  1. DOF angular error  — primary pose accuracy signal
                          (normal: 2.74°  |  broken: 16.00°)
  2. Root position offset — physical stability / fall detection
                          (normal: 12.74 cm  |  broken: 86.15 cm)

Two reward modes selectable via `mode`:
  "simple" (2-component):  r = w_dof * r_dof  +  w_root * r_root
                           Weights default: dof=0.7, root=0.3
  "full"   (4-component):  r = w_dof * r_dof  +  w_root * r_root
                              + w_height * r_height  +  w_alive * r_alive
                           Weights default: dof=0.5, root=0.25, height=0.15, alive=0.1

If ``target_dof_vel`` is set (typical for motion tracking), add
  ``w_vel * r_vel`` with mean absolute joint-velocity error vs reference (rad/s),
  and reduce other weights — defaults listed in ``GoalReachReward.__init__``.

Additional components:
  3. Root height reward  — reward matching the target pelvis height (Z only)
  4. Alive bonus         — flat reward for staying above a minimum height,
                           zeroed out when the robot falls

All components use exponential kernels so the reward is always in [0, 1]
and degrades smoothly rather than clipping.

Typical combined reward (DeepMimic style):
    r = w_dof * r_dof + w_root * r_root + w_height * r_height + w_alive * r_alive

Usage (inside a rollout loop):
    from goal_reach_reward import GoalReachReward
    reward_fn = GoalReachReward(target_dof_pos, target_root_pos, target_root_quat)
    ...
    r, info = reward_fn(current_dof_pos, current_root_pos, current_root_quat)
    # If ``target_dof_vel`` was passed at construction (tracking / NPZ with velocities):
    # r, info = reward_fn(..., dof_vel=current_joint_velocities_rad_s)
"""

from __future__ import annotations
import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """[w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def geodesic_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angular distance in radians between two [w,x,y,z] quaternions."""
    R1 = _quat_to_mat(q1)
    R2 = _quat_to_mat(q2)
    Rrel = R1.T @ R2
    trace = Rrel[0, 0] + Rrel[1, 1] + Rrel[2, 2]
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    return float(min(angle, np.pi - angle))   # handle double-cover


def exp_kernel(error: float, sigma: float) -> float:
    """Exponential reward kernel: exp(-error² / sigma²) ∈ (0, 1]."""
    return float(np.exp(-(error ** 2) / (sigma ** 2)))


# ─── reward function ──────────────────────────────────────────────────────────

class GoalReachReward:
    """
    Reward function for goal-reaching policy training/evaluation.

    Parameters
    ----------
    target_dof_pos : (29,) float array
        Target joint angles in radians.
    target_root_pos : (3,) float array
        Target pelvis world position [x, y, z].
    target_root_quat : (4,) float array
        Target pelvis orientation [w, x, y, z].

    Mode:
        mode         "simple"  →  2-component: DOF + root offset only
                                  weights must sum to 1: w_dof + w_root = 1
                     "full"    →  4-component: DOF + root + height + alive
                                  weights must sum to 1: w_dof+w_root+w_height+w_alive = 1

    Weights (must sum to 1.0):
        w_dof        DOF angular closeness  (primary pose signal)
        w_root       Root XY position closeness  (fall / drift penalty)
        w_height     Root Z height closeness  (full mode only)
        w_alive      Flat bonus for staying upright  (full mode only)
        w_vel        Joint velocity closeness  (only if ``target_dof_vel`` is set)

    Kernel widths (controls how sharply the reward drops off):
        sigma_dof    in radians  (≈ 0.2 rad ≈ 11° → tight; 0.5 rad → loose)
        sigma_root   in metres   (0.2 m → tight; 0.5 m → loose)
        sigma_height in metres   (full mode only)
        sigma_vel    in rad/s    (only if ``target_dof_vel`` is set)

    Other:
        min_height   Pelvis Z must exceed this to receive any reward.
                     Below this the robot is considered "fallen" and r=0.
        yaw_invariant_root  If True, ignore yaw when computing root XY error
                            (useful when global facing direction doesn't matter).
    """

    SIMPLE = "simple"
    FULL   = "full"

    def __init__(
        self,
        target_dof_pos:   np.ndarray,
        target_root_pos:  np.ndarray,
        target_root_quat: np.ndarray,
        *,
        mode: str = "full",           # "simple" (2-component) or "full" (4-component)
        # --- component weights (None → use per-mode defaults) ---
        w_dof:    float | None = None,
        w_root:   float | None = None,
        w_height: float = 0.15,       # full mode only
        w_alive:  float = 0.10,       # full mode only
        w_vel:    float | None = None,  # only when target_dof_vel is set
        # --- kernel widths ---
        sigma_dof:    float = 0.30,   # radians  (~17°)
        sigma_root:   float = 0.30,   # metres
        sigma_height: float = 0.10,   # metres  (full mode only)
        sigma_vel:    float = 2.0,    # rad/s (mean abs joint vel error)
        # --- misc ---
        target_dof_vel: np.ndarray | None = None,
        min_height: float = 0.50,     # robot must keep pelvis above this (metres)
        yaw_invariant_root: bool = True,
    ) -> None:
        if mode not in (self.SIMPLE, self.FULL):
            raise ValueError(f"mode must be 'simple' or 'full', got {mode!r}")
        self.mode = mode

        use_vel = target_dof_vel is not None
        self.target_dof_vel = (
            None if not use_vel else np.asarray(target_dof_vel, dtype=np.float64).reshape(-1)
        )
        if use_vel and self.target_dof_vel.shape != np.asarray(target_dof_pos).reshape(-1).shape:
            raise ValueError(
                "target_dof_vel must match target_dof_pos shape "
                f"got {self.target_dof_vel.shape} vs dof {np.asarray(target_dof_pos).shape}"
            )

        # resolve default weights per mode
        if use_vel:
            if mode == self.SIMPLE:
                w_dof  = w_dof  if w_dof  is not None else 0.55
                w_root = w_root if w_root is not None else 0.25
                w_vel  = w_vel  if w_vel  is not None else 0.20
                assert abs(w_dof + w_root + w_vel - 1.0) < 1e-6, (
                    "simple + velocity: w_dof + w_root + w_vel must equal 1.0")
            else:
                if w_dof is None and w_root is None and w_vel is None:
                    w_dof, w_root, w_height, w_alive, w_vel = 0.38, 0.19, 0.13, 0.10, 0.20
                else:
                    w_dof = w_dof if w_dof is not None else 0.38
                    w_root = w_root if w_root is not None else 0.19
                    w_vel = w_vel if w_vel is not None else 0.20
                    assert abs(w_dof + w_root + w_height + w_alive + w_vel - 1.0) < 1e-6, (
                        "full + velocity: w_dof + w_root + w_height + w_alive + w_vel must equal 1.0 "
                        f"(got dof={w_dof} root={w_root} height={w_height} alive={w_alive} vel={w_vel})"
                    )
            self.w_vel = float(w_vel)
        else:
            if mode == self.SIMPLE:
                w_dof  = w_dof  if w_dof  is not None else 0.70
                w_root = w_root if w_root is not None else 0.30
                assert abs(w_dof + w_root - 1.0) < 1e-6, \
                    "simple mode: w_dof + w_root must equal 1.0"
            else:
                w_dof  = w_dof  if w_dof  is not None else 0.50
                w_root = w_root if w_root is not None else 0.25
                assert abs(w_dof + w_root + w_height + w_alive - 1.0) < 1e-6, \
                    "full mode: w_dof + w_root + w_height + w_alive must equal 1.0"
            self.w_vel = 0.0

        self.target_dof_pos   = np.asarray(target_dof_pos,   dtype=np.float64)
        self.target_root_pos  = np.asarray(target_root_pos,  dtype=np.float64)
        self.target_root_quat = np.asarray(target_root_quat, dtype=np.float64)

        self.w_dof    = w_dof
        self.w_root   = w_root
        self.w_height = w_height
        self.w_alive  = w_alive

        self.sigma_dof    = sigma_dof
        self.sigma_root   = sigma_root
        self.sigma_height = sigma_height
        self.sigma_vel    = sigma_vel

        self.min_height         = min_height
        self.yaw_invariant_root = yaw_invariant_root

    # ------------------------------------------------------------------
    def __call__(
        self,
        dof_pos:   np.ndarray,   # (29,) current joint angles in radians
        root_pos:  np.ndarray,   # (3,)  current pelvis world position
        root_quat: np.ndarray,   # (4,)  current pelvis orientation [w,x,y,z]
        *,
        dof_vel: np.ndarray | None = None,
    ) -> tuple[float, dict]:
        """
        Compute the reward for one simulation step.

        If this instance was built with ``target_dof_vel``, pass ``dof_vel`` (29,)
        joint velocities in rad/s (same order as ``dof_pos``), e.g. ``qvel[6:35]``.

        Returns
        -------
        reward : float   total scalar reward in [0, 1]
        info   : dict    per-component breakdown for logging/debugging
        """
        dof_pos   = np.asarray(dof_pos,   dtype=np.float64)
        root_pos  = np.asarray(root_pos,  dtype=np.float64)
        root_quat = np.asarray(root_quat, dtype=np.float64)
        use_vel = self.target_dof_vel is not None
        if use_vel:
            if dof_vel is None:
                raise ValueError(
                    "dof_vel is required when GoalReachReward was constructed with target_dof_vel"
                )
            dof_vel = np.asarray(dof_vel, dtype=np.float64).reshape(-1)

        # ── 1. Alive check (both modes) ───────────────────────────────
        current_height = float(root_pos[2])
        alive = current_height >= self.min_height

        if not alive:
            info = dict(
                reward=0.0, alive=False, mode=self.mode,
                r_dof=0.0, r_root=0.0, r_height=0.0, r_alive=0.0, r_vel=0.0,
                dof_err_rad=float('nan'), dof_err_deg=float('nan'),
                root_offset_m=float('nan'), height_err_m=float('nan'),
                vel_err_rad_s=float('nan'),
            )
            return 0.0, info

        # ── 2. DOF angular error — used in both modes ─────────────────
        dof_err_rad = float(np.mean(np.abs(dof_pos - self.target_dof_pos)))
        r_dof = exp_kernel(dof_err_rad, self.sigma_dof)

        # ── 3. Root XY position offset — used in both modes ───────────
        xy_err = float(np.linalg.norm(root_pos[:2] - self.target_root_pos[:2]))
        root_offset_m = xy_err
        r_root = exp_kernel(root_offset_m, self.sigma_root)

        # ── 4. Height + alive — full mode only ────────────────────────
        height_err_m = float(abs(root_pos[2] - self.target_root_pos[2]))
        r_height = exp_kernel(height_err_m, self.sigma_height)
        r_alive  = 1.0

        # ── 5. Joint velocity — optional ──────────────────────────────
        if use_vel:
            vel_err_rad_s = float(np.mean(np.abs(dof_vel - self.target_dof_vel)))
            r_vel = exp_kernel(vel_err_rad_s, self.sigma_vel)
        else:
            vel_err_rad_s = float("nan")
            r_vel = float("nan")

        # ── 6. Combine ────────────────────────────────────────────────
        if use_vel:
            if self.mode == self.SIMPLE:
                reward = self.w_dof * r_dof + self.w_root * r_root + self.w_vel * r_vel
            else:
                reward = (
                    self.w_dof    * r_dof    +
                    self.w_root   * r_root   +
                    self.w_height * r_height +
                    self.w_alive  * r_alive  +
                    self.w_vel    * r_vel
                )
        elif self.mode == self.SIMPLE:
            reward = self.w_dof * r_dof + self.w_root * r_root
        else:
            reward = (
                self.w_dof    * r_dof    +
                self.w_root   * r_root   +
                self.w_height * r_height +
                self.w_alive  * r_alive
            )

        info = dict(
            reward        = reward,
            alive         = True,
            mode          = self.mode,
            r_dof         = r_dof,
            r_root        = r_root,
            r_height      = r_height if self.mode == self.FULL else None,
            r_alive       = r_alive  if self.mode == self.FULL else None,
            r_vel         = r_vel if use_vel else None,
            dof_err_rad   = dof_err_rad,
            dof_err_deg   = float(np.degrees(dof_err_rad)),
            root_offset_m = root_offset_m,
            height_err_m  = height_err_m if self.mode == self.FULL else None,
            vel_err_rad_s = vel_err_rad_s if use_vel else None,
        )
        return reward, info

    # ------------------------------------------------------------------
    @classmethod
    def from_npz_frame(
        cls,
        npz_path: str,
        frame_idx: int,
        **kwargs,
    ) -> "GoalReachReward":
        """
        Convenience constructor: load target pose directly from a motion NPZ.

        Example
        -------
        reward_fn = GoalReachReward.from_npz_frame(
            "example_motion/dance1_subject2_50_jpos.npz",
            frame_idx=500,
            w_dof=0.60, w_root=0.20, w_height=0.10, w_alive=0.10,
        )
        """
        data = np.load(npz_path)
        target_dof_pos   = data["dof_positions"][frame_idx].astype(np.float64)
        target_root_pos  = data["body_positions"][frame_idx, 0].astype(np.float64)
        target_root_quat = data["body_rotations"][frame_idx, 0].astype(np.float64)
        extra = {}
        if "dof_velocities" in data:
            extra["target_dof_vel"] = data["dof_velocities"][frame_idx].astype(np.float64)
        return cls(target_dof_pos, target_root_pos, target_root_quat, **extra, **kwargs)

    # ------------------------------------------------------------------
    def print_config(self) -> None:
        """Print a summary of the reward configuration."""
        print(f"GoalReachReward  mode={self.mode!r}")
        use_vel = self.target_dof_vel is not None
        if self.mode == self.SIMPLE:
            print(f"  Weights   : dof={self.w_dof}  root={self.w_root}"
                  + (f"  vel={self.w_vel}" if use_vel else "")
                  + "  (height & alive not used)")
        else:
            print(f"  Weights   : dof={self.w_dof}  root={self.w_root}  "
                  f"height={self.w_height}  alive={self.w_alive}"
                  + (f"  vel={self.w_vel}" if use_vel else ""))
        print(f"  Sigmas    : dof={self.sigma_dof} rad  root={self.sigma_root} m"
              + (f"  height={self.sigma_height} m" if self.mode == self.FULL else "")
              + (f"  vel={self.sigma_vel} rad/s" if use_vel else ""))
        print(f"  Min height: {self.min_height} m   "
              f"yaw_invariant_root={self.yaw_invariant_root}")
        print(f"  Target DOF (first 5): {self.target_dof_pos[:5].round(3)}")
        print(f"  Target root Z: {self.target_root_pos[2]:.3f} m")


def _pearson_corr_bounded(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation in [-1, 1]; 0 if too few samples or degenerate variance."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n < 3:
        return 0.0
    a = a[:n]
    b = b[:n]
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom < 1e-12:
        return 0.0
    r = float(np.dot(a0, b0) / denom)
    return max(-1.0, min(1.0, r))


def root_z_trajectory_shape_reward(
    z_ref: np.ndarray,
    z_sim: np.ndarray,
    *,
    deriv_weight: float = 0.5,
) -> float:
    """
    Map reference vs simulated pelvis height sequences to a reward in [0, 1].

    Combines (1) Pearson correlation on mean-centered ``z`` — insensitive to a
    constant height offset but rewards the same overall up/down pattern — and
    (2) Pearson correlation on first differences ``Δz``, which emphasizes
    matching when the root descends vs rises (e.g. crouch then stand).

    Uses the first ``min(len(z_ref), len(z_sim))`` samples so early termination
    (fall) is judged on the matched prefix only.
    """
    z_ref = np.asarray(z_ref, dtype=np.float64).ravel()
    z_sim = np.asarray(z_sim, dtype=np.float64).ravel()
    n = min(z_ref.size, z_sim.size)
    if n < 3:
        return 0.5
    r_level = _pearson_corr_bounded(z_ref[:n], z_sim[:n])
    r_level_u = (r_level + 1.0) * 0.5
    if n < 4:
        return r_level_u
    d_ref = np.diff(z_ref[:n])
    d_sim = np.diff(z_sim[:n])
    r_vel = _pearson_corr_bounded(d_ref, d_sim)
    r_vel_u = (r_vel + 1.0) * 0.5
    w = float(np.clip(deriv_weight, 0.0, 1.0))
    return (1.0 - w) * r_level_u + w * r_vel_u


class MotionTrackingReward:
    """
    Per-step pose reward for motion tracking: at simulator step ``t`` the target
    is reference frame ``frame_start + t`` (clamped to the last loaded frame).

    ``frame_end`` is **exclusive** (same as ``build_z_from_motion``): reference
    frames are ``frame_start .. frame_end - 1``.

    ``from_npz`` loads ``dof_velocities`` when present; otherwise estimates
    reference joint speeds by finite differencing ``dof_positions`` with ``fps``.

    Call ``reset()`` at the start of each episode (``recover_physics_params`` does
    this automatically when present).

    Optional trajectory term (physics recovery / tracking evaluation):

        w_traj_z (float, default 0)
            If > 0, episode fitness can blend mean per-step reward with
            ``root_z_trajectory_shape_reward`` over logged pelvis Z (see
            ``combined_episode_fitness``, invoked from ``recover_physics_params.run_episode``).
        traj_z_deriv_weight (float, default 0.5)
            In ``[0, 1]``: weight on Δz correlation vs mean-centered z correlation
            inside ``root_z_trajectory_shape_reward`` (higher → more emphasis on
            vertical motion timing, e.g. crouch-down-then-up).
    """

    def __init__(
        self,
        dof_seq: np.ndarray,
        root_pos_seq: np.ndarray,
        root_quat_seq: np.ndarray,
        *,
        dof_vel_seq: np.ndarray | None = None,
        frame_start: int = 0,
        w_traj_z: float = 0.0,
        traj_z_deriv_weight: float = 0.5,
        **kwargs,
    ) -> None:
        self._dof_seq = np.asarray(dof_seq, dtype=np.float64)
        self._rp_seq = np.asarray(root_pos_seq, dtype=np.float64)
        self._rq_seq = np.asarray(root_quat_seq, dtype=np.float64)
        self._dof_vel_seq = (
            None if dof_vel_seq is None else np.asarray(dof_vel_seq, dtype=np.float64)
        )
        self._frame_start = int(frame_start)
        self._w_traj_z = max(0.0, min(1.0, float(w_traj_z)))
        self._traj_z_deriv_w = max(0.0, min(1.0, float(traj_z_deriv_weight)))
        self._kwargs = kwargs
        self._t = 0
        self._sim_root_z: list[float] = []
        self.last_traj_shape_r: float | None = None
        if self._dof_seq.shape[0] == 0:
            raise ValueError("MotionTrackingReward: empty frame range (frame_end <= frame_start?)")

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        frame_start: int,
        frame_end: int,
        **kwargs,
    ) -> "MotionTrackingReward":
        data = np.load(npz_path)
        dof = data["dof_positions"][frame_start:frame_end].astype(np.float64)
        rp = data["body_positions"][frame_start:frame_end, 0].astype(np.float64)
        rq = data["body_rotations"][frame_start:frame_end, 0].astype(np.float64)
        if "dof_velocities" in data:
            dvel = data["dof_velocities"][frame_start:frame_end].astype(np.float64)
        else:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            dvel = np.zeros_like(dof)
            if dof.shape[0] > 1:
                dvel[1:] = (dof[1:] - dof[:-1]) * fps
                dvel[0] = dvel[1]
        return cls(dof, rp, rq, dof_vel_seq=dvel, frame_start=frame_start, **kwargs)

    def reset(self) -> None:
        self._t = 0
        self._sim_root_z = []
        self.last_traj_shape_r = None

    def combined_episode_fitness(self, mean_step_reward: float) -> float:
        """
        Blend mean per-step tracking reward with a trajectory-level pelvis-height
        shape score (see ``root_z_trajectory_shape_reward``). Used by
        ``recover_physics_params.run_episode`` when ``w_traj_z > 0``.
        """
        self.last_traj_shape_r = None
        if self._w_traj_z <= 0.0:
            return mean_step_reward
        z_ref = self._rp_seq[:, 2]
        z_sim = np.asarray(self._sim_root_z, dtype=np.float64)
        r_traj = root_z_trajectory_shape_reward(
            z_ref, z_sim, deriv_weight=self._traj_z_deriv_w)
        self.last_traj_shape_r = float(r_traj)
        w = max(0.0, min(1.0, self._w_traj_z))
        return (1.0 - w) * mean_step_reward + w * r_traj

    def __call__(
        self,
        dof_pos: np.ndarray,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        *,
        dof_vel: np.ndarray | None = None,
    ) -> tuple[float, dict]:
        i = min(self._t, self._dof_seq.shape[0] - 1)
        kw = dict(self._kwargs)
        if self._dof_vel_seq is not None:
            kw["target_dof_vel"] = self._dof_vel_seq[i]
        fn = GoalReachReward(
            self._dof_seq[i],
            self._rp_seq[i],
            self._rq_seq[i],
            **kw,
        )
        self._t += 1
        r, info = fn(dof_pos, root_pos, root_quat, dof_vel=dof_vel)
        info = dict(info)
        info["track_ref_frame"] = self._frame_start + i
        info["track_step"] = self._t - 1
        self._sim_root_z.append(float(root_pos[2]))
        return r, info


# ─── sigma sensitivity table ──────────────────────────────────────────────────

def print_sigma_table() -> None:
    """
    Show how the exponential kernel degrades at different error levels.
    Useful for choosing sigma values.

    DOF error (radians → degrees):
        sigma=0.20 rad (~11°) → tight: rewards only very accurate poses
        sigma=0.30 rad (~17°) → medium: default
        sigma=0.50 rad (~29°) → loose: forgiving

    Root offset (metres):
        sigma=0.20 m → tight: 20 cm drift → reward 0.37
        sigma=0.30 m → medium: 30 cm drift → reward 0.37
        sigma=0.50 m → loose: 50 cm drift → reward 0.37
    """
    print("── DOF reward  r = exp(-(err/sigma)²) ────────────────────────")
    print(f"  {'error':>10}  {'sigma=0.20':>12}  {'sigma=0.30':>12}  {'sigma=0.50':>12}")
    for err_deg in [1, 5, 10, 17, 25, 35]:
        err_rad = np.radians(err_deg)
        vals = [exp_kernel(err_rad, s) for s in [0.20, 0.30, 0.50]]
        print(f"  {err_deg:>4}° ({err_rad:.3f} rad)  "
              + "  ".join(f"{v:12.4f}" for v in vals))
    print()
    print("── Root offset reward  r = exp(-(err/sigma)²) ────────────────")
    print(f"  {'error (m)':>10}  {'sigma=0.20':>12}  {'sigma=0.30':>12}  {'sigma=0.50':>12}")
    for err_m in [0.05, 0.10, 0.20, 0.30, 0.50, 0.87]:
        vals = [exp_kernel(err_m, s) for s in [0.20, 0.30, 0.50]]
        print(f"  {err_m:>9.2f}m  " + "  ".join(f"{v:12.4f}" for v in vals))
    print()
    # highlight the broken-physics values
    print("  ← Normal physics  DOF err ≈ 2.74° (0.048 rad),"
          " root offset ≈ 12.74 cm")
    print("  ← Broken physics  DOF err ≈ 16.00° (0.279 rad),"
          " root offset ≈ 86.15 cm  (robot fell)")


# ─── demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_sigma_table()
    print()

    # Instantiate both modes for comparison
    reward_simple = GoalReachReward.from_npz_frame(
        "example_motion/dance1_subject2_50_jpos.npz",
        frame_idx=500, mode="simple",
    )
    reward_full = GoalReachReward.from_npz_frame(
        "example_motion/dance1_subject2_50_jpos.npz",
        frame_idx=500, mode="full",
    )
    reward_simple.print_config()
    print()
    reward_full.print_config()
    print()

    # Simulate: normal-physics result
    data     = np.load("example_motion/dance1_subject2_50_jpos.npz")
    tgt_dof  = data["dof_positions"][500]
    tgt_pos  = data["body_positions"][500, 0]

    # Approximate "normal physics" result from the report
    cur_dof_normal = tgt_dof + np.random.default_rng(0).normal(
        0, np.radians(2.74), size=tgt_dof.shape)
    cur_pos_normal = tgt_pos + np.array([0.1274, 0.0, 0.0])

    cur_pos_broken = tgt_pos + np.array([0.8615, 0.0, -0.60])  # fell over
    cur_dof_broken = tgt_dof + np.random.default_rng(1).normal(
        0, np.radians(16.0), size=tgt_dof.shape)

    tgt_vel = data["dof_velocities"][500] if "dof_velocities" in data.files else None
    rng2, rng3 = np.random.default_rng(2), np.random.default_rng(3)
    cur_vel_normal = (
        tgt_vel + rng2.normal(0, 0.05, size=tgt_dof.shape) if tgt_vel is not None else None
    )
    cur_vel_broken = (
        tgt_vel + rng3.normal(0, 0.6, size=tgt_dof.shape) if tgt_vel is not None else None
    )

    def _run(fn, dof, pos, dq=None):
        return fn(dof, pos, data["body_rotations"][500, 0], dof_vel=dq)

    _, info_simple_normal = _run(reward_simple, cur_dof_normal, cur_pos_normal, cur_vel_normal)
    _, info_simple_broken = _run(reward_simple, cur_dof_broken, cur_pos_broken, cur_vel_broken)
    _, info_full_normal   = _run(reward_full,   cur_dof_normal, cur_pos_normal, cur_vel_normal)
    _, info_full_broken   = _run(reward_full,   cur_dof_broken, cur_pos_broken, cur_vel_broken)

    print("── Simulated reward comparison ──────────────────────────────")
    print(f"  {'Metric':<25}  {'simple/normal':>14}  {'simple/broken':>14}"
          f"  {'full/normal':>12}  {'full/broken':>12}")
    print(f"  {'-'*25}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*12}")
    for k in ["reward", "r_dof", "r_root", "r_height", "r_alive", "r_vel",
              "dof_err_deg", "root_offset_m", "vel_err_rad_s"]:
        vs_n = info_simple_normal.get(k)
        vs_b = info_simple_broken.get(k)
        vf_n = info_full_normal.get(k)
        vf_b = info_full_broken.get(k)
        fmt  = lambda v: f"{v:>12.4f}" if isinstance(v, float) else f"{'N/A':>12}"
        print(f"  {k:<25}  {fmt(vs_n)}  {fmt(vs_b)}  {fmt(vf_n)}  {fmt(vf_b)}")
