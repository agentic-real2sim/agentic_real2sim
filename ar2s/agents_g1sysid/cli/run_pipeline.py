"""CLI: full G1 SysID pipeline - Stage 1 + Stage 2 + Stage 3.

Stage breakdown
---------------
  Stage 1  Motion Reader Agent (LLM) reads a raw data directory containing a
           paired .npz reference motion and .pkl latent context, validates
           the pairing, and produces a MotionBundle.

  Stage 2  build_scene() - deterministic. Loads the MuJoCo G1 environment,
           wraps the ONNX policy, applies physics perturbation, snapshots
           normal and broken physics. No LLM.

  Stage 3  run_sysid() - hill-climbing over alpha-space to recover broken physics
           parameters, followed by a trajectory rollout, motion comparison,
           and one-shot LLM convergence assessment.

Usage
-----
# Minimal (all defaults):
    python -m ar2s.agents_g1sysid.cli.run_pipeline raw_motions/dance_v0/

# Full options:
    python -m ar2s.agents_g1sysid.cli.run_pipeline raw_motions/dance_v0/ \\
        --onnx-model model/exported/FBcprAuxModel.onnx \\
        --break-seed 42 \\
        --n-iters 200 --sigma 0.15 --per-param \\
        --n-steps 300 \\
        --agent-config ar2s/agent_configs/config_anthropic_opus47.yaml \\
        --run-id dance_v0

# Skip Stage 1 using bundled sample data:
    python -m ar2s.agents_g1sysid.cli.run_pipeline ar2s/agents_g1sysid/assets/motions/ \\
        --skip-stage1 \\
        --motion ar2s/agents_g1sysid/assets/motions/pose_variation_0.npz \\
        --pkl    ar2s/agents_g1sysid/assets/motions/pose_variation_0.pkl \\
        --task-type goal --goal-frame 299 \\
        --n-frames 300 --z-dim 256 --n-z-rows 1

Exit code: 0 = converged or good quality; 1 = otherwise.
"""
from __future__ import annotations

import os
# Must be set before `import mujoco` — mujoco caches the GL backend at import
# time. EGL skips the GTK/GLFW/Mesa-Zink chain and uses NVIDIA's headless EGL
# directly, cutting video render time from ~0.3 s/frame to <30 ms/frame.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
import time
from pathlib import Path

from ar2s.agent_configs import set_config_path
from ar2s.pipeline_artifacts import pipeline_run_dir
from ar2s.agents_g1sysid.agents.motion_reader_agent import run_motion_reader_agent
from ar2s.agents_g1sysid.inputs import MotionBundle
from ar2s.agents_g1sysid.scene import DEFAULT_ONNX, ASSETS_DIR, build_scene
from ar2s.agents_g1sysid.sysid import (
    N_ITERS_DEFAULT,
    N_STEPS_DEFAULT,
    SIGMA_DEFAULT,
    ALPHA_MIN_DEFAULT,
    ALPHA_MAX_DEFAULT,
    run_sysid,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run the full G1 SysID pipeline (Stage 1 + 2 + 3)."
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("data_dir",
                   help="Directory containing paired .npz + .pkl (Stage 1 input).")
    p.add_argument("--onnx-model", default=DEFAULT_ONNX,
                   help="Path to ONNX policy model (default: ar2s/agents_g1sysid/assets/model/FBcprAuxModel.onnx).")
    p.add_argument("--run-id", default=None,
                   help="Pipeline run id for default artifacts. Defaults to data_dir basename.")
    p.add_argument("--artifact-root", default=None, type=Path,
                   help="Root containing per-run pipeline artifacts. Defaults "
                        "to outputs/run_pipeline or AR2S_RUN_PIPELINE_ROOT.")
    p.add_argument("--artifacts-root", default=None,
                   help="Parent directory for Stage 3 <run-id>/ output. Defaults "
                        "to outputs/run_pipeline/<run_id>/artifacts.")

    # ── Stage gating ─────────────────────────────────────────────────────────
    p.add_argument("--skip-stage1", action="store_true",
                   help="Skip Stage 1; provide MotionBundle fields manually.")

    # ── Manual MotionBundle (used when --skip-stage1) ─────────────────────
    p.add_argument("--motion", default=None, help="(skip-stage1) .npz motion path.")
    p.add_argument("--pkl", default=None, help="(skip-stage1) .pkl latent path.")
    p.add_argument("--task-type", choices=["goal", "tracking"], default="tracking")
    p.add_argument("--goal-frame", type=int, default=None,
                   help="(goal) Frame index in .npz for GoalReachReward (default: last frame).")
    p.add_argument("--z-idx", type=int, default=None,
                   help="(goal, raw-array pkl) Row index in the pkl file to use as the "
                        "fixed latent z vector.  Use this when the pkl is a large sequence "
                        "file (e.g. zs_3.pkl) and you want a specific single z.  "
                        "Example: --z-idx 0 loads zs_3.pkl[0].")
    p.add_argument("--z-row-start", type=int, default=None)
    p.add_argument("--z-row-end", type=int, default=None)
    p.add_argument("--n-frames", type=int, default=None,
                   help="(skip-stage1) Total frames in .npz. Auto-detected if omitted.")
    p.add_argument("--z-dim", type=int, default=None,
                   help="(skip-stage1) Latent dimension. Auto-detected from pkl if omitted.")
    p.add_argument("--n-z-rows", type=int, default=None,
                   help="(skip-stage1) Total rows in pkl. Auto-detected from pkl if omitted.")
    p.add_argument("--description", default="manually specified motion")
    p.add_argument("--min-height", type=float, default=0.3)

    # ── Physics perturbation ─────────────────────────────────────────────────
    p.add_argument("--break-seed", type=int, default=42,
                   help="Seed for random physics perturbation (default 42).")
    p.add_argument("--device", default="cpu",
                   help="ONNX execution device: cpu or cuda (default: cpu).")

    # ── Stage 3 search ───────────────────────────────────────────────────────
    p.add_argument("--n-iters", type=int, default=N_ITERS_DEFAULT,
                   help=f"Hill-climbing iterations (default {N_ITERS_DEFAULT}).")
    p.add_argument("--n-steps", type=int, default=N_STEPS_DEFAULT,
                   help=f"Simulation steps per episode (default {N_STEPS_DEFAULT}).")
    p.add_argument("--sigma", type=float, default=SIGMA_DEFAULT,
                   help=f"Log-space perturbation std (default {SIGMA_DEFAULT}).")
    p.add_argument("--alpha-min", type=float, default=ALPHA_MIN_DEFAULT)
    p.add_argument("--alpha-max", type=float, default=ALPHA_MAX_DEFAULT)
    p.add_argument("--per-param", action="store_true",
                   help="Per-body/per-DOF alpha search instead of 3-dim grouped.")
    p.add_argument("--no-video", action="store_true",
                   help="Skip MP4 video recording (saves time/disk).")
    p.add_argument("--save-oracle-rollout", action="store_true",
                   help="Also record and save the oracle-corrected rollout.")

    # ── Agent model config ──────────────────────────────────────────────────
    p.add_argument("--agent-config", default=None, type=Path,
                   help="YAML model/runtime config. Defaults to "
                        "ar2s/agent_configs/config_anthropic_opus47.yaml.")

    args = p.parse_args()
    set_config_path(args.agent_config)
    data_dir = Path(args.data_dir).expanduser().resolve()
    run_id = args.run_id or data_dir.name
    if args.artifacts_root is None:
        args.artifacts_root = str(pipeline_run_dir(run_id, args.artifact_root) / "artifacts")

    t_pipeline = time.time()

    print("=" * 70)
    print("G1 SysID pipeline")
    print(f"  data_dir   = {data_dir}")
    print(f"  onnx_model = {args.onnx_model}")
    print(f"  stage1     = {'skip' if args.skip_stage1 else 'run'}")
    if args.skip_stage1:
        z_src = (f"pkl[{args.z_idx}]" if args.z_idx is not None
                 else f"pkl rows [{args.z_row_start}:{args.z_row_end}]"
                 if args.task_type == "tracking" else "pkl (goal)")
        print(f"  task       = {args.task_type}  z={z_src}")
    print(f"  break_seed = {args.break_seed}")
    print(f"  iters={args.n_iters}  steps={args.n_steps}  sigma={args.sigma}"
          f"  per_param={args.per_param}")
    print("=" * 70)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if not args.skip_stage1:
        print("\n[stage 1] Motion Reader Agent: inspect data dir + validate bundle")
        t0 = time.time()
        try:
            bundle = run_motion_reader_agent(
                data_dir=data_dir,
                onnx_model_path=args.onnx_model,
            )
        except Exception as e:
            print(f"[stage 1] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        print(f"[stage 1] done in {time.time()-t0:.1f}s - "
              f"task={bundle.task_type}  n_frames={bundle.n_frames}  "
              f"z_dim={bundle.z_dim}")
        print(f"[stage 1] {bundle.description}")
    else:
        print("\n[stage 1] skipped - building MotionBundle from CLI args")
        if not args.motion or not args.pkl:
            print("[stage 1] ERROR: --motion and --pkl required with --skip-stage1",
                  file=sys.stderr)
            return 1

        # ── Auto-detect n_frames, z_dim, n_z_rows from files when not provided ──
        motion_p = Path(args.motion).resolve()
        pkl_p    = Path(args.pkl).resolve()

        n_frames = args.n_frames
        z_dim    = args.z_dim
        n_z_rows = args.n_z_rows

        if n_frames is None or z_dim is None or n_z_rows is None:
            import numpy as np
            import joblib

            if n_frames is None:
                try:
                    n_frames = int(np.load(str(motion_p))["dof_positions"].shape[0])
                    print(f"[stage 1] auto-detected n_frames={n_frames} from {motion_p.name}")
                except Exception as e:
                    print(f"[stage 1] ERROR: could not read n_frames from {motion_p}: {e}",
                          file=sys.stderr)
                    return 1

            if z_dim is None or n_z_rows is None:
                try:
                    raw = joblib.load(str(pkl_p))
                    if isinstance(raw, dict):
                        arr = np.asarray(raw["z"], dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                    else:
                        arr = np.asarray(raw, dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                    if z_dim is None:
                        z_dim = int(arr.shape[1])
                        print(f"[stage 1] auto-detected z_dim={z_dim} from {pkl_p.name}")
                    if n_z_rows is None:
                        n_z_rows = int(arr.shape[0])
                        print(f"[stage 1] auto-detected n_z_rows={n_z_rows} from {pkl_p.name}")
                except Exception as e:
                    print(f"[stage 1] ERROR: could not read z info from {pkl_p}: {e}",
                          file=sys.stderr)
                    return 1

        bundle = MotionBundle(
            motion_path=str(motion_p),
            pkl_path=str(pkl_p),
            onnx_model_path=str(Path(args.onnx_model).resolve()),
            task_type=args.task_type,
            goal_frame=args.goal_frame,
            z_idx=args.z_idx,
            z_row_start=args.z_row_start,
            z_row_end=args.z_row_end,
            n_frames=n_frames,
            z_dim=z_dim,
            n_z_rows=n_z_rows,
            description=args.description,
            min_height=args.min_height,
        )

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print("\n[stage 2] Scene setup: load env + ONNX policy + break physics")
    t0 = time.time()
    try:
        scene = build_scene(bundle, break_seed=args.break_seed, device=args.device)
    except Exception as e:
        print(f"[stage 2] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"[stage 2] done in {time.time()-t0:.1f}s")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    print(f"\n[stage 3] SysID: {args.n_iters} iters  sigma={args.sigma}")
    t0 = time.time()
    try:
        result = run_sysid(
            scene,
            artifacts_root=args.artifacts_root,
            n_iters=args.n_iters,
            n_steps=args.n_steps,
            sigma=args.sigma,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            per_param=args.per_param,
            save_video=not args.no_video,
            save_oracle_rollout=args.save_oracle_rollout,
        )
    except Exception as e:
        print(f"[stage 3] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    ok = result.outcome in ("converged",)
    print(f"[stage 3] done in {time.time()-t0:.1f}s - outcome={result.outcome}  "
          f"recovered={result.r_recovered:.4f}  oracle={result.r_oracle:.4f}")
    print(f"[stage 3] artifacts: {result.run_dir}")

    elapsed = time.time() - t_pipeline
    print(f"\n[pipeline] TOTAL: {elapsed:.1f}s - "
          f"{'CONVERGED ★' if ok else result.outcome}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
