"""Stage 3 - SysID loop orchestrator.

Hill-climbing over alpha-space to recover broken physics parameters, followed by
a one-shot LLM convergence assessment.

Algorithm (ported from BFM-Zero/recover_physics_params.py):
  1. Evaluate baseline (broken physics, alpha = 1)
  2. Evaluate oracle (true ideal alpha)
  3. Random log-space hill-climbing: sample alpha_candidate near alpha_best;
     keep if reward improves.
  4. Run final rollout with alpha_best, save NPZ recording.
  5. Compare recording vs reference motion (trajectory metrics).
  6. Call convergence_agent to assess quality and emit a SysIDResult.

Output layout::

    artifacts/<run-id>/
      summary.yaml
      rollout_recovered.npz
      rollout_oracle.npz    (optional)
      metrics.json

Public API
----------
    from ar2s.agents_g1sysid.sysid import run_sysid
    result = run_sysid(scene, artifacts_root="artifacts/", n_iters=80)
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ar2s.agents_g1sysid.scene import (
    G1Scene,
    PhysicsSnapshot,
    alpha_dim,
    alpha_to_normal_ratio,
    apply_correction,
)
from ar2s.agents_g1sysid.state import SysIDResult
from ar2s.agents_g1sysid.skills.compare_motion import compute_trajectory_metrics

_BFM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bfm_zero")


def _ensure_bfm_on_path() -> None:
    if _BFM_ROOT not in sys.path:
        sys.path.insert(0, _BFM_ROOT)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

N_ITERS_DEFAULT = 80
SIGMA_DEFAULT = 0.25
ALPHA_MIN_DEFAULT = 0.05
ALPHA_MAX_DEFAULT = 20.0
N_STEPS_DEFAULT = 300
SAVE_ORACLE_ROLLOUT_DEFAULT = False


# ---------------------------------------------------------------------------
# Z loading helpers
# ---------------------------------------------------------------------------

def _load_z(scene: G1Scene) -> np.ndarray:
    """Load the latent context from the bundle's pkl into a numpy array.

    Returns:
        (z_dim,)     for goal task  — fixed z used at every sim step
        (T, z_dim)   for tracking   — per-step z sequence

    z selection for goal task (raw-array pkl):
        bundle.z_idx is set     -> use that specific row: pkl[z_idx]
        bundle.z_idx is None    -> use row 0 as default
    """
    import joblib
    bundle = scene.bundle
    raw = joblib.load(bundle.pkl_path)

    # Dict-style pkl (single export, e.g. goal_idx14.pkl or pose_variation_*.pkl)
    if isinstance(raw, dict):
        z_np = np.asarray(raw["z"], dtype=np.float32).flatten()
        return z_np  # (z_dim,)

    # Raw (N, z_dim) array pkl  (e.g. zs_3.pkl)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if bundle.task_type == "tracking":
        rs = bundle.z_row_start or 0
        re = bundle.z_row_end or arr.shape[0]
        return arr[rs:re]  # (T, z_dim)

    # Goal task: pick a single row.
    # z_idx explicitly overrides (intended for large sequence pkls like zs_3.pkl).
    idx = bundle.z_idx if bundle.z_idx is not None else 0
    idx = max(0, min(idx, arr.shape[0] - 1))
    print(f"  z loaded from pkl row {idx}  (n_rows={arr.shape[0]}, z_dim={arr.shape[1]})")
    return arr[idx]  # (z_dim,)


def _z_at_step(z: np.ndarray, t: int) -> np.ndarray:
    """Return the z to use at step t (goal: constant; tracking: indexed)."""
    if z.ndim == 2:
        return z[min(t, z.shape[0] - 1)]
    return z


# ---------------------------------------------------------------------------
# Reward function loading
# ---------------------------------------------------------------------------

def _build_reward_fn(scene: G1Scene):
    """Build the appropriate GoalReachReward or MotionTrackingReward."""
    _ensure_bfm_on_path()
    from goal_reach_reward import GoalReachReward, MotionTrackingReward  # type: ignore

    bundle = scene.bundle
    mh = bundle.min_height

    if bundle.task_type == "tracking":
        fs = bundle.z_row_start or 0
        fe = bundle.z_row_end or bundle.n_frames
        return MotionTrackingReward.from_npz(
            bundle.motion_path, fs, fe, mode="simple", min_height=mh
        )
    else:
        gf = bundle.goal_frame if bundle.goal_frame is not None else bundle.n_frames - 1
        return GoalReachReward.from_npz_frame(
            bundle.motion_path, gf, mode="simple", min_height=mh
        )


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def _run_episode(scene: G1Scene, z: np.ndarray, reward_fn,
                 n_steps: int) -> tuple[float, dict]:
    """Run one episode; return (mean_reward, last_info)."""
    env = scene.env
    policy = scene.policy

    if hasattr(reward_fn, "reset"):
        reward_fn.reset()

    actual_steps = n_steps
    if z.ndim == 2:
        actual_steps = min(n_steps, z.shape[0])

    obs = env.reset()
    rewards = []
    last_info: dict = {}

    for t in range(actual_steps):
        action = policy.get_action(obs, _z_at_step(z, t))
        obs, next_obs, _, _ = env.step(action)
        obs = next_obs
        r, info = reward_fn(
            dof_pos=env.mjd.qpos[7:36],
            root_pos=env.mjd.qpos[0:3],
            root_quat=env.mjd.qpos[3:7],
            dof_vel=env.mjd.qvel[6:35],
        )
        rewards.append(r)
        last_info = info
        if not info.get("alive", True):
            break

    mean_r = float(np.mean(rewards)) if rewards else 0.0
    if hasattr(reward_fn, "combined_episode_fitness"):
        mean_r = reward_fn.combined_episode_fitness(mean_r)
    return mean_r, last_info


# ---------------------------------------------------------------------------
# Hill-climbing
# ---------------------------------------------------------------------------

def _recover_alpha(
    scene: G1Scene,
    z: np.ndarray,
    reward_fn,
    n_steps: int,
    n_iters: int,
    sigma: float,
    alpha_min: float,
    alpha_max: float,
    per_param: bool,
) -> tuple[np.ndarray, list[float]]:
    """Log-space random hill-climbing over alpha-space.

    Returns (alpha_best, history) where history[i] is the best fitness after
    iteration i (history[0] = baseline with alpha = 1).
    """
    broken = scene.broken_snapshot
    dim = alpha_dim(broken) if per_param else 3
    rng = np.random.default_rng(0)

    alpha_best = np.ones(dim)
    apply_correction(scene.env, broken, alpha_best, per_param)
    r_best, _ = _run_episode(scene, z, reward_fn, n_steps)
    history = [r_best]

    print(f"  iter   0  r={r_best:.4f}  (no correction, dim={dim})")

    for i in range(1, n_iters + 1):
        log_cand = np.log(alpha_best) + rng.normal(0, sigma, size=dim)
        alpha_cand = np.clip(np.exp(log_cand), alpha_min, alpha_max)

        apply_correction(scene.env, broken, alpha_cand, per_param)
        r_cand, _ = _run_episode(scene, z, reward_fn, n_steps)
        history.append(r_cand)

        if r_cand > r_best:
            alpha_best = alpha_cand
            r_best = r_cand
            tag = "✓"
        else:
            tag = " "

        if i % 10 == 0 or i == n_iters:
            print(f"  iter {i:3d}  r={r_cand:.4f}  best={r_best:.4f}  {tag}")

    apply_correction(scene.env, broken, alpha_best, per_param)
    return alpha_best, history


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import contextlib as _contextlib


@_contextlib.contextmanager
def _quiet_stderr():
    """Suppress C-library stderr (GTK/Mesa/EGL init noise) at the fd level.

    Python's contextlib.redirect_stderr only catches Python writes; Mesa and
    GTK write directly to fd 2. We dup2 over it to silence those too.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


# ---------------------------------------------------------------------------
# Recording helper (ported from BFM-Zero/run_mujoco_env_demo.py)
# ---------------------------------------------------------------------------

def _run_and_record(scene: G1Scene, z: np.ndarray, n_steps: int,
                    save_path: str | Path,
                    video_path: str | Path | None = None) -> None:
    """Run one episode; save .npz trajectory and optionally an .mp4 video.

    The original sim-based recording is used so the camera follows the robot
    exactly as configured in env.py. GTK/Mesa/EGL init warnings are suppressed
    by redirecting fd 2 around the env constructor (C libraries write there
    directly; Python-level redirect does not catch them).
    """
    _ensure_bfm_on_path()
    from common import (ACTION_RESCALE, ACTION_SCALES, DEFAULT_JOINT_POS,  # type: ignore
                        KD_GAINS, KP_GAINS, POLICY_JOINT_NAMES)
    from env import MuJoCoBFMZeroEnv, apply_random_physics_perturbation  # type: ignore
    from run_mujoco_env_demo import (_recording_body_indices,  # type: ignore
                                     save_recording)

    do_video = video_path is not None
    _ensure_robot_xml()

    # Suppress C-level init noise only during env construction (the one place
    # that initialises the GL context / GTK / libdecor).
    with _quiet_stderr():
        rec_env = MuJoCoBFMZeroEnv(
            robot_xml=_LOCAL_ROBOT_XML,
            kp_gains=KP_GAINS, kd_gains=KD_GAINS,
            default_joint_pos=DEFAULT_JOINT_POS,
            action_scales=ACTION_SCALES, action_rescale=ACTION_RESCALE,
            enable_video=do_video,
            video_path=str(video_path) if do_video else "/tmp/_g1sysid_dummy.mp4",
            video_fps=50.0,
        )

    apply_random_physics_perturbation(rec_env, seed=scene.break_seed)
    rec_env.mjm.body_mass[:]        = scene.env.mjm.body_mass.copy()
    rec_env.mjm.dof_frictionloss[:] = scene.env.mjm.dof_frictionloss.copy()
    rec_env.kp_gains[:]             = scene.env.kp_gains.copy()
    rec_env.kd_gains[:]             = scene.env.kd_gains.copy()

    body_indices, body_names = _recording_body_indices(rec_env.mjm)
    frames = []
    actual_steps = min(n_steps, z.shape[0]) if z.ndim == 2 else n_steps
    obs = rec_env.reset()
    for t in range(actual_steps):
        action = scene.policy.get_action(obs, _z_at_step(z, t))
        obs, next_obs, _, _ = rec_env.step(action)
        obs = next_obs
        frames.append({
            "body_positions": rec_env.mjd.xpos[body_indices].copy(),
            "body_rotations": rec_env.mjd.xquat[body_indices].copy(),
            "dof_positions":  rec_env.mjd.qpos[7:36].copy(),
            "dof_velocities": rec_env.mjd.qvel[6:35].copy(),
            "root_pos":       rec_env.mjd.qpos[0:3].copy(),
            "root_quat":      rec_env.mjd.qpos[3:7].copy(),
        })

    save_recording(str(save_path), frames, 50.0, POLICY_JOINT_NAMES, body_names)

    if do_video and rec_env.frames:
        import imageio
        vp = Path(video_path)
        vp.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(vp), rec_env.frames, fps=50.0)
        print(f"  Video saved -> {vp}  ({len(rec_env.frames)} frames)")


def _ensure_robot_xml() -> None:
    """Use the bundled local XML if available, fall back to BFM-Zero repo."""
    global _LOCAL_ROBOT_XML
    if _LOCAL_ROBOT_XML:
        return
    from ar2s.agents_g1sysid.scene import ROBOT_XML as _bundled
    if os.path.exists(_bundled):
        _LOCAL_ROBOT_XML = _bundled
    else:
        _LOCAL_ROBOT_XML = os.path.join(
            _BFM_ROOT, "bfm_zero_inference_code", "g1_for_reward_inference.xml"
        )


_LOCAL_ROBOT_XML: str = ""


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_sysid(
    scene: G1Scene,
    *,
    artifacts_root: str | Path = "artifacts",
    run_id: str | None = None,
    n_iters: int = N_ITERS_DEFAULT,
    n_steps: int = N_STEPS_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
    alpha_min: float = ALPHA_MIN_DEFAULT,
    alpha_max: float = ALPHA_MAX_DEFAULT,
    per_param: bool = False,
    save_video: bool = True,
    save_oracle_rollout: bool = SAVE_ORACLE_ROLLOUT_DEFAULT,
) -> SysIDResult:
    """Run the Stage-3 SysID loop: hill-climb + compare + convergence assess.

    Args:
        scene:               G1Scene from build_scene().
        artifacts_root:      parent of <run-id>/ output directory.
        run_id:              explicit id; default = timestamp + uuid6.
        n_iters:             hill-climbing iterations (default 80).
        n_steps:             simulation steps per episode (default 300).
        sigma:               log-space perturbation std for alpha (default 0.25).
        alpha_min / max:     alpha search bounds.
        per_param:           if True, search per-body / per-DOF / per-gain alpha
                             (dim = n_bodies + n_dof + 2) instead of 3-dim grouped.
        save_video:          save rollout_recovered.mp4 alongside the NPZ (default True).
        save_oracle_rollout: also record and save the oracle-alpha rollout.

    Returns:
        SysIDResult with outcome, best alpha, rewards, motion metrics, and run_dir.
    """
    artifacts_root = Path(artifacts_root).expanduser().resolve()
    run_id = run_id or (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = scene.bundle
    broken = scene.broken_snapshot
    normal = scene.normal_snapshot

    print(f"\n[stage3-sysid] run_dir    = {run_dir}")
    print(f"[stage3-sysid] episode    = {bundle.motion_path}")
    print(f"[stage3-sysid] task       = {bundle.task_type}")
    print(f"[stage3-sysid] iters={n_iters}  steps={n_steps}  sigma={sigma}  per_param={per_param}")

    # ── Load z and reward function ──────────────────────────────────────────
    _ensure_bfm_on_path()
    z = _load_z(scene)
    reward_fn = _build_reward_fn(scene)

    true_alpha = alpha_to_normal_ratio(broken, normal, per_param)
    dim = alpha_dim(broken) if per_param else 3

    # ── Baseline ────────────────────────────────────────────────────────────
    print(f"\n── Baseline (broken, no correction) ─────────────────────────")
    apply_correction(scene.env, broken, np.ones(dim), per_param)
    r_baseline, _ = _run_episode(scene, z, reward_fn, n_steps)
    print(f"  reward = {r_baseline:.4f}")

    # ── Oracle ──────────────────────────────────────────────────────────────
    print(f"\n── Oracle (true alpha) ───────────────────────────────────────────")
    apply_correction(scene.env, broken, true_alpha, per_param)
    r_oracle, _ = _run_episode(scene, z, reward_fn, n_steps)
    print(f"  reward = {r_oracle:.4f}")

    # ── Hill-climbing ────────────────────────────────────────────────────────
    print(f"\n── Searching {dim}-dim alpha-space ({n_iters} iters, sigma={sigma}) ──")
    alpha_best, history = _recover_alpha(
        scene, z, reward_fn,
        n_steps=n_steps, n_iters=n_iters,
        sigma=sigma, alpha_min=alpha_min, alpha_max=alpha_max,
        per_param=per_param,
    )
    r_recovered = max(history)

    print(f"\n── Best recovered  reward={r_recovered:.4f}  "
          f"(baseline={r_baseline:.4f}  oracle={r_oracle:.4f})")

    # ── Final rollout + comparison ──────────────────────────────────────────
    rollout_path = run_dir / "rollout_recovered.npz"
    video_path   = run_dir / "rollout_recovered.mp4" if save_video else None
    print(f"\n── Recording final rollout -> {rollout_path}"
          + ("  +video" if save_video else ""))
    _run_and_record(scene, z, n_steps, rollout_path, video_path=video_path)

    motion_metrics: dict = {}
    if rollout_path.exists():
        try:
            motion_metrics = compute_trajectory_metrics(
                str(rollout_path), bundle.motion_path
            )
            # Keep only scalar metrics for serialisation
            scalar_metrics = {
                k: float(v) for k, v in motion_metrics.items()
                if isinstance(v, (int, float))
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(scalar_metrics, indent=2)
            )
            print(f"  MPJPE = {motion_metrics.get('mpjpe_m', 'n/a'):.4f} m  "
                  f"DOF err = {motion_metrics.get('mean_dof_err_deg', 'n/a'):.2f}°")
        except Exception as e:
            print(f"  [warn] motion comparison failed: {e}")

    # ── Oracle rollout (optional) ─────────────────────────────────────────
    if save_oracle_rollout:
        oracle_path  = run_dir / "rollout_oracle.npz"
        oracle_video = run_dir / "rollout_oracle.mp4" if save_video else None
        print(f"\n── Recording oracle rollout -> {oracle_path}"
              + ("  +video" if save_video else ""))
        apply_correction(scene.env, broken, true_alpha, per_param)
        _run_and_record(scene, z, n_steps, oracle_path, video_path=oracle_video)

    # ── Convergence assessment (LLM) ────────────────────────────────────────
    print(f"\n── Convergence assessment (LLM) ──────────────────────────────")
    from ar2s.agents_g1sysid.agents.sysid_loop.convergence_agent import assess_convergence
    verdict = assess_convergence(
        baseline_reward=r_baseline,
        oracle_reward=r_oracle,
        recovered_reward=r_recovered,
        n_iters=n_iters,
        reward_curve=history,
        alpha_best=alpha_best,
        motion_metrics={k: float(v) for k, v in motion_metrics.items()
                        if isinstance(v, (int, float))},
    )
    print(f"  outcome={verdict.outcome}  quality={verdict.recovery_quality}")
    print(f"  {verdict.reasoning[:200]}")
    if verdict.suggestions:
        print(f"  suggestion: {verdict.suggestions[:120]}")

    # ── Save summary ─────────────────────────────────────────────────────────
    result = SysIDResult(
        outcome=verdict.outcome,
        alpha_best=tuple(alpha_best.tolist()),
        r_baseline=r_baseline,
        r_oracle=r_oracle,
        r_recovered=r_recovered,
        history=history,
        motion_metrics={k: float(v) for k, v in motion_metrics.items()
                        if isinstance(v, (int, float))},
        n_iters=n_iters,
        run_dir=run_dir,
    )
    summary = {
        "outcome": result.outcome,
        "recovery_quality": verdict.recovery_quality,
        "alpha_best": list(result.alpha_best),
        "rewards": {
            "baseline": round(r_baseline, 4),
            "oracle": round(r_oracle, 4),
            "recovered": round(r_recovered, 4),
        },
        "motion_metrics": result.motion_metrics,
        "n_iters": n_iters,
        "run_dir": str(run_dir),
        "llm_reasoning": verdict.reasoning,
        "llm_suggestions": verdict.suggestions,
    }
    (run_dir / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, default_flow_style=False)
    )
    print(f"\n[stage3-sysid] summary -> {run_dir / 'summary.yaml'}")

    return result


__all__ = [
    "N_ITERS_DEFAULT",
    "N_STEPS_DEFAULT",
    "SIGMA_DEFAULT",
    "run_sysid",
]
