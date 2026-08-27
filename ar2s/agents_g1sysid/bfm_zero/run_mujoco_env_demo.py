#!/usr/bin/env python3
"""
Run the BFM-Zero MuJoCo environment from the command line (no Jupyter).

Examples (from repo root, conda env activated):

  # Interactive window with dance tracking (on-sphere z from backward map)
  python run_mujoco_env_demo.py --motion example_motion/dance1_subject2_50_jpos.npz --viewer

  # Headless tracking + save video
  python run_mujoco_env_demo.py --motion example_motion/dance1_subject2_50_jpos.npz --video videos/dance_tracking.mp4

  # Motion tracking: per-step z from model (tracking_inference) + record NPZ
  python run_mujoco_env_demo.py --motion example_motion/dance1_subject2_50_jpos.npz \\
      --frame-start 0 --frame-end 300 --record recordings/tracking_model.npz

  # Motion tracking: per-step z from an exported (N,256) array (e.g. Isaac zs_3.pkl)
  # Rows [frame_start:frame_end) must exist; optional --motion only sets fps to match reference NPZ
  python run_mujoco_env_demo.py --z-pkl exported/zs_3.pkl --z-sequence \\
      --frame-start 5100 --frame-end 5400 --motion recordings/ref.npz \\
      --record recordings/tracking_zs3.npz

  # Random z (no motion reference)
  python run_mujoco_env_demo.py --viewer --steps 500

  # Goal reaching: pick one frame from the clip, hold that z for the whole rollout
  python run_mujoco_env_demo.py --motion example_motion/dance1_subject2_50_jpos.npz --goal-frame 250 --viewer

  # Goal reaching + save video + recording
  python run_mujoco_env_demo.py --motion example_motion/dance1_subject2_50_jpos.npz --goal-frame 250 \
      --steps 300 --video videos/goal_frame250.mp4 --record recordings/goal_frame250.npz

  # Same as notebook: goal inference from a saved goal export (dict with motion_path + goal_frame)
  python run_mujoco_env_demo.py --goal-pkl z_exports/goal_idx14.pkl --steps 300 --record recordings/run_a.npz

  # Same as above but with broken physics
  python run_mujoco_env_demo.py --motion example_motion/dance1_subject2_50_jpos.npz --break-motion --video videos/broken.mp4

Requires: model weights under ./model/checkpoint/model (see README / download_hf_model.py).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils._pytree import tree_map

from common import ACTION_RESCALE, ACTION_SCALES, DEFAULT_JOINT_POS, KD_GAINS, KP_GAINS
from env import MuJoCoBFMZeroEnv, apply_random_physics_perturbation, calc_angular_velocity

ROBOT_XML = "bfm_zero_inference_code/g1_for_reward_inference.xml"
DEFAULT_MODEL = "./model/checkpoint/model"

# Bodies to skip when recording (match the 30-body layout of the example NPZ).
# world (index 0) + 8 dummy foot contact bodies + head_link + two rubber hands.
_SKIP_BODIES = {
    "world", "head_link", "left_rubber_hand", "right_rubber_hand",
    "dummy_lf_1", "dummy_lf_2", "dummy_lf_3", "dummy_lf_4",
    "dummy_rf_1", "dummy_rf_2", "dummy_rf_3", "dummy_rf_4",
}


def _resolve_motion_path(motion_rel: str, repo_root: str) -> str:
    """Turn motion_path from a pkl or CLI into an absolute path if it lives under the repo."""
    if os.path.isabs(motion_rel) and os.path.isfile(motion_rel):
        return motion_rel
    cand = os.path.join(repo_root, motion_rel)
    if os.path.isfile(cand):
        return cand
    return os.path.normpath(motion_rel)


def _recording_body_indices(mjm):
    """Return (indices, names) for the 30 bodies saved in the recording NPZ."""
    indices, names = [], []
    for i in range(mjm.nbody):
        name = mjm.body(i).name
        if name not in _SKIP_BODIES:
            indices.append(i)
            names.append(name)
    return indices, names


def save_recording(path: str, frames: list[dict], fps: float,
                   dof_names: list[str], body_names: list[str]) -> None:
    """Save recorded simulation frames to an NPZ matching the example motion format."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(
        path,
        body_positions=np.array([f["body_positions"] for f in frames], dtype=np.float64),
        body_rotations=np.array([f["body_rotations"] for f in frames], dtype=np.float64),
        dof_positions=np.array([f["dof_positions"] for f in frames], dtype=np.float64),
        dof_velocities=np.array([f["dof_velocities"] for f in frames], dtype=np.float64),
        root_pos=np.array([f["root_pos"] for f in frames], dtype=np.float64),
        root_quat=np.array([f["root_quat"] for f in frames], dtype=np.float64),
        fps=np.float64(fps),
        dof_names=np.array(dof_names),
        body_names=np.array(body_names),
    )
    T = len(frames)
    print(f"Recorded {T} frames → {path}")


def get_action_from_policy(model, obs, latent_z):
    if not isinstance(latent_z, torch.Tensor):
        latent_z = torch.tensor(latent_z, dtype=torch.float32)
    obs_torch = tree_map(lambda x: torch.tensor(x).unsqueeze(0), obs)
    with torch.no_grad():
        action = model.act(obs_torch, latent_z.unsqueeze(0), mean=True)
    return action[0].cpu().numpy()


def build_z_goal_from_frame(model, env, npz_path, goal_frame):
    """
    Load a motion NPZ, snap the robot into one frame (zero velocity), read the
    backward-map observation, and return a single projected z via goal_inference.

    This is the CLI equivalent of the notebook's goal-inference-from-motion-frame cell:
      env.set_state(pose)  →  _create_observation_backward()  →  model.goal_inference()

    Returns:
        z_goal: Tensor [z_dim]  (single projected z)
        fps: float
    """
    print(f"Loading motion for goal inference: {npz_path}  frame={goal_frame}")
    data = np.load(npz_path)
    fps = float(data["fps"])

    root_pos  = data["body_positions"][goal_frame, 0, :].astype(np.float32)
    root_quat = data["body_rotations"][goal_frame, 0, :].astype(np.float32)
    dof_pos   = data["dof_positions"][goal_frame, :].astype(np.float32)

    env.reset()
    env.set_state(
        dof_positions=dof_pos,
        root_pos=root_pos,
        root_quat=root_quat,
        # zero velocities — static snapshot of the pose
    )
    goal_obs = env._create_observation_backward()
    # move obs tensors to same device as model before inference
    device = next(model.parameters()).device
    goal_obs = tree_map(lambda x: torch.tensor(x).to(device) if not isinstance(x, torch.Tensor)
                        else x.to(device), goal_obs)
    z_goal = model.goal_inference(goal_obs)          # backward_map + project_z
    z_goal = z_goal[0] if z_goal.dim() > 1 else z_goal
    print(f"  Goal z inferred  norm={z_goal.norm().item():.3f}")
    return z_goal, fps


def build_z_from_motion(model, env, npz_path, frame_start, frame_end):
    """
    Load a motion NPZ, replay it kinematically through the env to collect
    observations, then run tracking_inference to get one z per frame.

    Returns:
        z_sequence: Tensor [T, z_dim]   (one projected z per frame)
        fps: float
    """
    print(f"Loading motion: {npz_path}")
    data = np.load(npz_path)
    fps = float(data["fps"])

    body_positions = data["body_positions"][frame_start:frame_end]   # [T, 30, 3]
    body_rotations = data["body_rotations"][frame_start:frame_end]   # [T, 30, 4]
    dof_positions  = data["dof_positions"][frame_start:frame_end]    # [T, 29]
    T = body_positions.shape[0]
    print(f"  Frames {frame_start}–{frame_end} ({T} frames, {fps} fps, {T/fps:.1f} s)")

    # Kinematic replay: set each frame and collect backward-map observations
    next_obs_list = []
    env.reset()
    for i in range(T):
        root_pos  = body_positions[i, 0, :]
        root_quat = body_rotations[i, 0, :]
        dof_pos   = dof_positions[i, :]

        if i > 0:
            dt = 1.0 / fps
            dof_vel    = (dof_positions[i] - dof_positions[i - 1]) / dt
            root_vel   = (body_positions[i, 0] - body_positions[i - 1, 0]) / dt
            root_ang_vel = calc_angular_velocity(body_rotations[i, 0], body_rotations[i - 1, 0], dt)
        else:
            dof_vel      = np.zeros(29, dtype=np.float32)
            root_vel     = np.zeros(3, dtype=np.float32)
            root_ang_vel = np.zeros(3, dtype=np.float32)

        env.set_state(
            dof_positions=dof_pos,
            dof_velocities=dof_vel,
            root_quat=root_quat,
            root_pos=root_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
        )
        next_obs_list.append(env._create_observation_backward())

    # Stack observations and run tracking inference (backward map + windowed mean + project)
    print("  Running tracking inference (backward map)…")
    next_obs_inf = {}
    for k in next_obs_list[0].keys():
        next_obs_inf[k] = torch.cat([o[k] for o in next_obs_list], dim=0)
    z_sequence = model.tracking_inference(next_obs_inf)
    print(f"  z sequence shape: {tuple(z_sequence.shape)}")
    return z_sequence, fps


def _extract_z_from_pkl(z_raw, idx: int) -> tuple:
    """
    Extract a single (256,) float32 z vector from a pkl payload.

    Supports two formats:
      • numpy array (N, 256)  — idx selects row
      • dict {name: (1, 256)} — idx selects the idx-th key by insertion order

    Returns (z_np, key_name) where key_name is None for array format.
    """
    import numpy as np
    if isinstance(z_raw, dict):
        keys = list(z_raw.keys())
        if idx < 0 or idx >= len(keys):
            raise IndexError(
                f"--z-idx {idx} out of range for dict pkl with {len(keys)} entries. "
                f"Valid range: 0–{len(keys)-1}.\n"
                f"Keys: {keys}")
        key = keys[idx]
        arr = np.array(z_raw[key], dtype=np.float32)
        return arr.reshape(-1), key
    else:
        arr = np.array(z_raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if idx < 0 or idx >= len(arr):
            raise IndexError(
                f"--z-idx {idx} out of range for array pkl with shape {arr.shape}.")
        return arr[idx].reshape(-1), None


def main() -> int:
    p = argparse.ArgumentParser(description="BFM-Zero MuJoCo rollout without Jupyter")
    p.add_argument("--model",  default=DEFAULT_MODEL, help="Path to FBcprAuxModel checkpoint dir")
    p.add_argument("--device", default="cpu",         help="torch device (e.g. cpu, cuda)")
    p.add_argument("--steps",  type=int, default=150,
                   help="Max simulation steps (random-z mode, or cap for motion mode)")
    p.add_argument("--video",  default=None,          help="Save MP4 to this path")
    p.add_argument("--fps",    type=float, default=50.0)

    # --- Motion tracking / goal from motion ---
    p.add_argument("--motion", default=None,
                   help="Path to .npz motion file. Uses tracking_inference z instead of random z.")
    p.add_argument("--frame-start", type=int, default=0,    help="First frame of the clip (tracking mode)")
    p.add_argument("--frame-end",   type=int, default=1000, help="Last frame (exclusive, tracking mode)")
    p.add_argument("--goal-frame",  type=int, default=None,
                   help="If set (requires --motion), pick this single frame as a goal pose and hold "
                        "its z for the whole rollout (goal-reaching mode, not per-step tracking).")
    p.add_argument("--goal-pkl", default=None,
                   help="Optional joblib dict from the notebook export (keys motion_path, goal_frame). "
                        "Fills --motion / --goal-frame when omitted so goal reaching matches "
                        "inference_tutorial (build_z_goal_from_frame + rollout). Ignored if --z-pkl is set.")

    # --- Direct z from pkl ---
    p.add_argument("--z-pkl", default=None,
                   help="Load z from a joblib pkl. Usually a numpy array of shape (N, 256). "
                        "With --z-sequence, rows [frame_start:frame_end) are used as per-step z. "
                        "Otherwise one row (see --z-idx) is used for the whole rollout. "
                        "Overrides --motion / --goal-frame for z *inference* (not for --z-sequence+--motion fps).")
    p.add_argument("--z-sequence", action="store_true",
                   help="With --z-pkl: use a 2D array and rows [frame_start:frame_end) as one z per step "
                        "(motion tracking, like tracking_inference on a full clip). "
                        "Incompatible with --goal-frame. Optional --motion supplies fps for video/NPZ.")
    p.add_argument("--z-idx", type=int, default=0,
                   help="Row index of the z to select from --z-pkl when not using --z-sequence (default: 0).")

    # --- Interactive viewer ---
    p.add_argument("--viewer", action="store_true",
                   help="Open MuJoCo's interactive viewer (needs a display).")

    # --- Break physics ---
    p.add_argument("--break-motion", action="store_true",
                   help="Randomize masses, friction, PD gains to break motion.")
    p.add_argument("--break-seed", type=int, default=None,
                   help="RNG seed for --break-motion.")
    p.add_argument("--record", default=None,
                   help="Save simulation recording to this .npz path "
                        "(body_positions, body_rotations, dof_positions, dof_velocities, root_pos, root_quat, fps, dof_names).")
    p.add_argument("--save-z", default=None,
                   help="Save the inferred per-step z sequence to this joblib "
                        ".pkl as a (T, z_dim) float32 array (tracking mode, "
                        "i.e. --motion without --goal-frame). The pkl pairs "
                        "with the motion NPZ for the g1sysid tracking task.")
    args = p.parse_args()

    if args.save_z:
        if not args.motion or args.goal_frame is not None or args.z_pkl:
            p.error("--save-z requires tracking mode: --motion without "
                    "--goal-frame / --z-pkl")
        if args.frame_start != 0:
            # sysid co-indexes the pkl and the NPZ with one z_row_start/end
            # pair, so a pre-sliced pkl would track 'frame_start' frames off.
            p.error("--save-z pkl must start at NPZ frame 0 to pair with the "
                    "motion; slice the clip at the converter instead")
        # Resolve before the chdir below, else a relative path lands in
        # bfm_zero/ instead of the invoking cwd.
        args.save_z = os.path.abspath(args.save_z)

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    if args.goal_pkl and args.z_pkl is None:
        import joblib
        if not os.path.isfile(args.goal_pkl):
            print(f"goal-pkl not found: {args.goal_pkl}", file=sys.stderr)
            return 1
        gp = joblib.load(args.goal_pkl)
        if not isinstance(gp, dict):
            print("--goal-pkl must be a dict with motion_path (and optional goal_frame).", file=sys.stderr)
            return 1
        if args.motion is None:
            mp = gp.get("motion_path")
            if not mp:
                print("goal-pkl has no motion_path; pass --motion explicitly.", file=sys.stderr)
                return 1
            args.motion = _resolve_motion_path(str(mp), root)
        if args.goal_frame is None and gp.get("goal_frame") is not None:
            args.goal_frame = int(gp["goal_frame"])
        if args.goal_frame is None:
            print(
                "goal-pkl: need goal_frame in the file or pass --goal-frame (e.g. -1 for last frame).",
                file=sys.stderr,
            )
            return 1

    if args.z_sequence and not args.z_pkl:
        print("--z-sequence requires --z-pkl (2D array of shape (N, z_dim)).", file=sys.stderr)
        return 1
    if args.z_sequence and args.goal_frame is not None:
        print("--z-sequence is motion tracking; do not set --goal-frame or --goal-pkl.", file=sys.stderr)
        return 1

    if not os.path.isdir(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        print("Download weights first: python download_hf_model.py", file=sys.stderr)
        return 1

    primary_xml = os.path.normpath(
        os.path.join(root, "..", "assets", "robot", "g1_for_reward_inference.xml")
    )
    xml_path = primary_xml
    if not os.path.isfile(xml_path):
        xml_path = os.path.join(root, ROBOT_XML)  # legacy layout
    if not os.path.isfile(xml_path):
        print(f"Robot XML not found: {primary_xml} (nor legacy {xml_path})",
              file=sys.stderr)
        return 1

    print("Loading model...")
    # Deferred: the model stack needs safetensors from the checkpoint download,
    # and motion_convert imports this module for save_recording/_recording_body_indices
    # in envs without it.
    from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModel
    model = FBcprAuxModel.load(args.model, device=args.device)

    enable_video = args.video is not None
    env = MuJoCoBFMZeroEnv(
        robot_xml=xml_path,
        kp_gains=KP_GAINS,
        kd_gains=KD_GAINS,
        default_joint_pos=DEFAULT_JOINT_POS,
        action_scales=ACTION_SCALES,
        action_rescale=ACTION_RESCALE,
        enable_video=enable_video,
        video_path=args.video or "videos/cli_demo.mp4",
        video_fps=args.fps,
    )

    if args.break_motion:
        apply_random_physics_perturbation(env, seed=args.break_seed)
        print(f"Physics perturbed (seed={args.break_seed}).")

    record_fps = float(args.fps)
    # ---- Build z ----
    if args.z_pkl is not None and args.z_sequence:
        # ── Per-step z from an exported (N, z_dim) table (e.g. Isaac zs_3.pkl) ──
        import joblib
        z_raw = joblib.load(args.z_pkl)
        if isinstance(z_raw, dict):
            print("--z-sequence requires a numpy array of shape (N, z_dim), not a dict.", file=sys.stderr)
            return 1
        arr = np.asarray(z_raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            print(f"--z-sequence: need 2D (N, z_dim), got shape {arr.shape}.", file=sys.stderr)
            return 1
        a, b = int(args.frame_start), int(args.frame_end)
        if b <= a:
            print("frame-end must be greater than frame-start.", file=sys.stderr)
            return 1
        if a < 0 or b > arr.shape[0]:
            print(
                f"z-slice [{a}:{b}) out of range: pkl has {arr.shape[0]} row(s).",
                file=sys.stderr,
            )
            return 1
        dev = next(model.parameters()).device
        z_sequence = torch.tensor(arr[a:b], dtype=torch.float32, device=dev)
        t_steps = int(z_sequence.shape[0])
        num_steps = min(
            t_steps, args.steps if (args.steps != 150) else t_steps
        ) if t_steps else 0
        if num_steps == 0:
            print("Empty z sequence after frame range / steps cap.", file=sys.stderr)
            return 1
        if args.motion:
            if not os.path.isfile(args.motion):
                print(f"--motion not found (for fps): {args.motion}", file=sys.stderr)
                return 1
            _fps_data = np.load(args.motion)
            record_fps = float(_fps_data["fps"])
        env.video_fps = record_fps
        print(
            f"Motion tracking (z from pkl): rows [{a}:{b})  →  {num_steps} sim step(s)  @  {record_fps} fps"
        )
        mode = "motion"
    elif args.z_pkl is not None:
        # ── Direct pkl: single z row (no inference) — ignores frame-start/end
        if int(args.frame_start) != 0 or int(args.frame_end) != 1000:
            print(
                "\n  WARNING: you set --frame-start / --frame-end but did NOT use --z-sequence.\n"
                f"  Using a single z from row --z-idx (default 0) → {args.z_idx!r}.\n"
                "  For a per-step z slice [frame_start:frame_end), re-run with --z-sequence "
                "(and usually --steps to match, e.g. 100 for 100 rows).\n",
                file=sys.stderr,
            )
        import joblib
        print(f"Loading z from pkl: {args.z_pkl}  idx={args.z_idx}")
        z_raw = joblib.load(args.z_pkl)
        z_np, key_name = _extract_z_from_pkl(z_raw, args.z_idx)
        if key_name:
            print(f"  key: '{key_name}'")
        random_z = torch.tensor(z_np, dtype=torch.float32, device=next(model.parameters()).device)
        print(f"  z norm={random_z.norm().item():.3f}")
        num_steps = args.steps
        mode = "random"   # fixed single z — reuse "random" path in get_z()
    elif args.motion:
        if not os.path.isfile(args.motion):
            print(f"Motion file not found: {args.motion}", file=sys.stderr)
            return 1

        if args.goal_frame is not None:
            # ── Goal reaching: one frame → single fixed z ──
            z_goal, motion_fps = build_z_goal_from_frame(
                model, env, args.motion, args.goal_frame
            )
            record_fps = motion_fps
            env.video_fps = motion_fps
            num_steps = args.steps
            print(f"Goal reaching from frame {args.goal_frame}: {num_steps} steps at {motion_fps} fps")
            mode = "goal"
        else:
            # ── Tracking: per-step z via tracking_inference on motion NPZ ──
            z_sequence, motion_fps = build_z_from_motion(
                model, env, args.motion, args.frame_start, args.frame_end
            )
            z_sequence = z_sequence.to(next(model.parameters()).device)
            record_fps = motion_fps
            env.video_fps = motion_fps
            zlen = int(z_sequence.shape[0])
            num_steps = min(
                zlen, args.steps if (args.steps != 150) else zlen
            )
            print(f"Motion tracking (inference on NPZ): {num_steps} steps at {motion_fps} fps")
            if args.save_z:
                import joblib
                os.makedirs(os.path.dirname(args.save_z), exist_ok=True)
                joblib.dump(z_sequence.cpu().numpy().astype(np.float32), args.save_z)
                print(f"Saved z sequence {tuple(z_sequence.shape)} covering NPZ "
                      f"frames [0, {zlen}) → {args.save_z}")
            mode = "motion"
    else:
        rng = np.random.default_rng(0)
        z_dim = model.cfg.archi.z_dim
        random_z = model.project_z(
            torch.tensor(rng.standard_normal(z_dim), dtype=torch.float32, device=next(model.parameters()).device)
        )
        num_steps = args.steps
        print(f"Random z (projected onto hypersphere): {num_steps} steps")
        mode = "random"

    def get_z(t):
        if mode == "motion":
            return z_sequence[t]
        if mode == "goal":
            return z_goal        # same z every step
        return random_z

    rec_body_indices, rec_body_names = _recording_body_indices(env.mjm)
    from common import POLICY_JOINT_NAMES
    rec_frames: list[dict] = []

    def capture_frame():
        rec_frames.append(dict(
            body_positions=env.mjd.xpos[rec_body_indices].copy(),   # (30, 3)
            body_rotations=env.mjd.xquat[rec_body_indices].copy(),  # (30, 4) [w,x,y,z]
            dof_positions=env.mjd.qpos[7:36].copy(),                # (29,)
            dof_velocities=env.mjd.qvel[6:35].copy(),               # (29,)
            root_pos=env.mjd.qpos[0:3].copy(),                      # (3,)  xyz
            root_quat=env.mjd.qpos[3:7].copy(),                     # (4,)  [w,x,y,z]
        ))

    obs = env.reset()

    if args.viewer:
        import mujoco.viewer
        print("Interactive viewer open. Close window or Ctrl+C to stop.")
        with mujoco.viewer.launch_passive(env.mjm, env.mjd) as viewer:
            t = 0
            while viewer.is_running() and t < num_steps:
                action = get_action_from_policy(model, obs, get_z(t))
                obs, _next_obs, _r, info = env.step(action)
                obs = _next_obs
                capture_frame()
                viewer.sync()
                if t % 50 == 0:
                    print(f"  step {t:5d}  root_z={info['root_height']:.3f}")
                t += 1
    else:
        print(f"Headless rollout ({num_steps} steps). Use --viewer for a window.")
        for t in range(num_steps):
            action = get_action_from_policy(model, obs, get_z(t))
            obs, _next_obs, _r, info = env.step(action)
            obs = _next_obs
            capture_frame()
            if t % 50 == 0:
                print(f"  step {t:5d}  root_z={info['root_height']:.3f}")

    if enable_video and env.enable_video:
        env.save_video()
        print(f"Saved video: {args.video}")

    if args.record and rec_frames:
        save_recording(args.record, rec_frames, record_fps, POLICY_JOINT_NAMES, rec_body_names)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
