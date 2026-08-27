"""Stage 2 - deterministic G1 scene and policy setup.

Pure deterministic pipeline (no LLM). Loads the MuJoCo environment, wraps
the ONNX policy, applies the physics perturbation, and snapshots both the
normal and broken physics states.

    load_bundle(bundle)
         |
         v   MuJoCoBFMZeroEnv
    OnnxPolicy.load(onnx_path)
         |
         v
    apply_random_physics_perturbation(seed)
         |
         v   snapshot normal + broken physics
    G1Scene  (passed to Stage 3)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ar2s.agents_g1sysid.inputs import MotionBundle

# ---------------------------------------------------------------------------
# Paths — all resolved relative to this file so the package is self-contained.
# bfm_zero/ is a bundled copy of the BFM-Zero Python source (env.py, etc.).
# No separate BFM-Zero clone is required.
# ---------------------------------------------------------------------------
_PKG_DIR    = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_PKG_DIR, "assets")
_BFM_ROOT   = os.path.join(_PKG_DIR, "bfm_zero")
_ROBOT_XML  = os.path.join(_ASSETS_DIR, "robot", "g1_for_reward_inference.xml")
_DEFAULT_ONNX = os.path.join(_ASSETS_DIR, "model", "FBcprAuxModel.onnx")


def _ensure_bfm_on_path() -> None:
    if _BFM_ROOT not in sys.path:
        sys.path.insert(0, _BFM_ROOT)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PhysicsSnapshot:
    """Frozen copy of the env's physics parameters at one instant."""
    body_mass: np.ndarray      # (n_bodies,)
    dof_friction: np.ndarray   # (n_dof,)
    kp_gains: np.ndarray       # (29,)
    kd_gains: np.ndarray       # (29,)


@dataclass
class G1Scene:
    """All handles needed by the Stage-3 SysID loop."""
    env: object                       # MuJoCoBFMZeroEnv
    policy: "OnnxPolicy"
    normal_snapshot: PhysicsSnapshot  # physics before perturbation
    broken_snapshot: PhysicsSnapshot  # physics after perturbation
    bundle: MotionBundle
    break_seed: int


# ---------------------------------------------------------------------------
# ONNX policy wrapper
# ---------------------------------------------------------------------------

class OnnxPolicy:
    """Wrap an ONNX actor model with the same get_action() interface as
    the PyTorch FBcprAuxModel."""

    # Single input 'actor_obs' (1, 721) layout confirmed from g1_deploy/rl_policy/bfm_zero.py:
    #   state (64) + last_action (29) + history_actor (372) + z (256) = 721
    # privileged_state excluded (teacher-only, not available at inference).
    # NOTE: last_action MUST come before history_actor. The first 29 dims of
    # history_actor happen to equal last_action (history_action[0] = last_action),
    # so [64:93] was coincidentally correct before, but [93:465] was shifted by 29
    # — corrupting the full 4-step action/velocity/gravity history fed to the actor.
    _OBS_KEY_ORDER = ("state", "last_action", "history_actor")

    def __init__(self, onnx_path: str, device: str = "cpu"):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # suppress INFO noise
        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_opts, providers=providers
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name    # "actor_obs"
        self.input_dim = inp.shape[1] # 721

    @classmethod
    def load(cls, onnx_path: str, device: str = "cpu") -> "OnnxPolicy":
        return cls(onnx_path, device=device)

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        """Flatten obs keys (excluding privileged_state) into float32 array."""
        parts = []
        for k in self._OBS_KEY_ORDER:
            v = obs[k]
            if hasattr(v, "cpu"):
                v = v.detach().cpu().numpy()
            parts.append(np.asarray(v, dtype=np.float32).flatten())
        return np.concatenate(parts)  # (465,)

    def get_action(self, obs: dict, z: np.ndarray) -> np.ndarray:
        """Run one forward pass.

        Args:
            obs: observation dict from MuJoCoBFMZeroEnv (torch tensors OK)
            z:   latent vector (z_dim,) for this step

        Returns:
            action array of shape (29,)
        """
        obs_flat = self._flatten_obs(obs)                   # (465,)
        z_flat = np.asarray(z, dtype=np.float32).flatten() # (256,)
        actor_obs = np.concatenate([obs_flat, z_flat])[None, :].astype(np.float32)
        outputs = self.session.run(None, {self.input_name: actor_obs})
        return outputs[0][0]  # (29,)


# ---------------------------------------------------------------------------
# Physics helpers (ported from BFM-Zero/recover_physics_params.py)
# ---------------------------------------------------------------------------

def snapshot_physics(env) -> PhysicsSnapshot:
    return PhysicsSnapshot(
        body_mass=env.mjm.body_mass.copy(),
        dof_friction=env.mjm.dof_frictionloss.copy(),
        kp_gains=env.kp_gains.copy(),
        kd_gains=env.kd_gains.copy(),
    )


def alpha_dim(broken: PhysicsSnapshot) -> int:
    """Per-param alpha dimension: n_bodies + n_dof + 2 (kp, kd)."""
    return len(broken.body_mass) + len(broken.dof_friction) + 2


def apply_correction(env, broken: PhysicsSnapshot, alpha: np.ndarray,
                     per_param: bool = False) -> None:
    """Apply multiplicative correction factors to the broken-physics snapshot."""
    if per_param:
        n_m = len(broken.body_mass)
        n_f = len(broken.dof_friction)
        am = alpha[:n_m]
        af = alpha[n_m:n_m + n_f]
        akp = float(alpha[n_m + n_f])
        akd = float(alpha[n_m + n_f + 1])
        env.mjm.body_mass[:]        = broken.body_mass    * am
        env.mjm.dof_frictionloss[:] = broken.dof_friction * af
        env.kp_gains[:]             = broken.kp_gains     * akp
        env.kd_gains[:]             = broken.kd_gains     * akd
    else:
        am, af, ag = float(alpha[0]), float(alpha[1]), float(alpha[2])
        env.mjm.body_mass[:]        = broken.body_mass    * am
        env.mjm.dof_frictionloss[:] = broken.dof_friction * af
        env.kp_gains[:]             = broken.kp_gains     * ag
        env.kd_gains[:]             = broken.kd_gains     * ag


def alpha_to_normal_ratio(broken: PhysicsSnapshot, normal: PhysicsSnapshot,
                           per_param: bool = False) -> np.ndarray:
    """Compute the ideal alpha that exactly maps broken to normal."""
    safe = lambda a, b: np.where(b != 0, a / b, 1.0)
    if per_param:
        am  = safe(normal.body_mass,    broken.body_mass)
        af  = safe(normal.dof_friction, broken.dof_friction)
        akp = float(np.mean(safe(normal.kp_gains, broken.kp_gains)))
        akd = float(np.mean(safe(normal.kd_gains, broken.kd_gains)))
        return np.concatenate([am, af, [akp, akd]])
    else:
        rm = float(np.mean(safe(normal.body_mass,    broken.body_mass)))
        rf = float(np.mean(safe(normal.dof_friction, broken.dof_friction)))
        rg = float(np.mean(safe(normal.kp_gains,     broken.kp_gains)))
        return np.array([rm, rf, rg])


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def build_scene(bundle: MotionBundle, break_seed: int = 42,
                device: str = "cpu") -> G1Scene:
    """Load the G1 env and ONNX policy, apply physics perturbation, return G1Scene.

    This is the Stage-2 entry point. Pure deterministic - no LLM involved.

    Args:
        bundle:     validated MotionBundle from Stage 1.
        break_seed: random seed for apply_random_physics_perturbation.
        device:     "cpu" or "cuda" (ONNX execution provider).

    Returns:
        G1Scene ready for Stage-3 hill-climbing.
    """
    _ensure_bfm_on_path()
    from common import (ACTION_RESCALE, ACTION_SCALES, DEFAULT_JOINT_POS,
                        KD_GAINS, KP_GAINS)
    from env import MuJoCoBFMZeroEnv, apply_random_physics_perturbation

    print(f"[stage2] Loading MuJoCo env  {_ROBOT_XML}")
    env = MuJoCoBFMZeroEnv(
        robot_xml=_ROBOT_XML,
        kp_gains=KP_GAINS,
        kd_gains=KD_GAINS,
        default_joint_pos=DEFAULT_JOINT_POS,
        action_scales=ACTION_SCALES,
        action_rescale=ACTION_RESCALE,
        enable_video=False,
    )
    normal_snap = snapshot_physics(env)
    print(f"[stage2] Normal physics snapshot saved.")

    apply_random_physics_perturbation(env, seed=break_seed)
    broken_snap = snapshot_physics(env)
    print(f"[stage2] Broken physics applied  (seed={break_seed}).")

    print(f"[stage2] Loading ONNX policy  {bundle.onnx_model_path}")
    policy = OnnxPolicy.load(bundle.onnx_model_path, device=device)

    return G1Scene(
        env=env,
        policy=policy,
        normal_snapshot=normal_snap,
        broken_snapshot=broken_snap,
        bundle=bundle,
        break_seed=break_seed,
    )


__all__ = [
    "ASSETS_DIR",
    "DEFAULT_ONNX",
    "G1Scene",
    "OnnxPolicy",
    "PhysicsSnapshot",
    "ROBOT_XML",
    "alpha_dim",
    "alpha_to_normal_ratio",
    "apply_correction",
    "build_scene",
    "snapshot_physics",
]

# Public aliases (uppercase) so callers can import canonical paths
ASSETS_DIR  = _ASSETS_DIR
DEFAULT_ONNX = _DEFAULT_ONNX
ROBOT_XML   = _ROBOT_XML
