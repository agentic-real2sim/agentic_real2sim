"""Brute-force grasp sweep — bypasses LLM optimization entirely.

Pipeline (2026-05-31 reframe: pickup-mesh uniform surface sampling):

  1. Run an initial probe to expose engage_frame + sim state.
  2. Compute gripper_anchor = ``ee_frame.xpos`` (the TCP site at
     base_mount.z=+0.160, ~33mm past pad-body midpoint toward the pad
     tip) in world frame at the reference frame (engage_frame or user
     override). Pre-2026-06-29 this used the silicone-pad-body midpoint;
     ee_frame is closer to the actual fingertip contact zone and gives
     more meaningful closest_distance / DET corrective-vector signals.
  3. Uniformly sample N points on the SCALED pickup visual mesh (via
     ``trimesh.sample.sample_surface``); each point is a candidate
     gripper-anchor target in the pickup's local frame.
  4. For each sample point:
         world_pos       = pickup_pos + pickup_R @ local_pos
         world_disp      = gripper_anchor − world_pos
         shift_userspec  = world_disp_to_shift(world_disp, ground_normal)
         run probe with this shift.
  5. Record outcome per sample. Any sample that grasps is a success.

Pre-2026-05-31 the sweep used a 3-D grid along the pickup's "long
axis" (local "y") with radial X/Z spread — only worked for pen-shaped
pickups. Replaced with mesh surface sampling so the same sweep works
for any geometry (lid, mug, bowl, foil packet, pen, ...).

Single-pickup scope: the sweep studies ONE grasp target — ``bundle.pickup_object``,
which is ``pickup_objects[0]``, the object the visual pipeline says the robot
grasps FIRST. Scenes where the gripper picks up several objects in sequence are
swept only for that first one; the others are present and simulated, but never
sampled, aligned to, or scored.
TODO: multi-object sweeping — one sweep per entry in ``bundle.pickup_objects``,
which also needs the trajectory time-segmented per grasp, since a single probe
replays the whole demo as one grasp event and the later pickups aren't in the
gripper at the reference frame.

Costs: N probes × ~6s each = ~240s for N=40. Zero LLM cost.

USD: default ON per user request (warning: ~47MB/sample → ~2GB for N=40).
Disable via ``save_usd=False`` if disk-constrained.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

from ar2s.agents_sysid._gl import configure_headless_gl

# Must run before `import mujoco`: MuJoCo's GL backend is locked in at
# `import mujoco` time (it reads MUJOCO_GL then, not when a Renderer is
# later constructed), so setting the env var afterwards has no effect.
configure_headless_gl()

import mujoco
import numpy as np
import yaml

from ar2s.droid_sim.scene.shift import ground_normal_world, world_disp_to_shift
from ar2s.pipeline_artifacts import sample_dir_name

from ar2s.agents_sysid.probe import _load_or_compute_calibration
from ar2s.agents_sysid.scene import CalibratedScene
from ar2s.agents_sysid.scene_config import build_scene_config
from ar2s.agents_sysid.skills.probe_grasp import (
    GraspProbeResult,
    FrameCache,
    probe_grasp,
)


# -------------------------------------------------------------------
# Tunable defaults
# -------------------------------------------------------------------

N_SAMPLES_DEFAULT = 40               # uniform surface samples on pickup mesh
SAMPLE_SEED_DEFAULT = 0              # rng seed for trimesh.sample.sample_surface
SAVE_USD_DEFAULT = True              # USDs are packed into samples.tar.zst by
                                     # pipeline_cleanup (--no-cleanup to retain)
USD_SAMPLE_EVERY_DEFAULT = 1         # frame-faithful playback for sim_bundle;
                                     # pipeline cleanup packs USDs post-run

# Multi-processing worker count: default = cpu_count - 1 (leave one core for
# main + OS), then capped by available RAM (see EST_WORKER_RSS_BYTES) and the
# number of samples (no idle workers).
N_WORKERS_DEFAULT: int | None = None  # None → resolved at call time

# Measured RSS for a single probe_grasp() call (RobotSceneSim construction +
# EGL renderer dominate; ~2.95GB with save_usd=False, ~3.2GB with
# save_usd=True at usd_sample_every=3).
# Padded above the measured ceiling: workers spawned without this cap can
# OOM-kill the parent process outright at cpu_count-1 workers.
EST_WORKER_RSS_BYTES = 4 * 1024 ** 3

# A dead pool worker's task is lost silently (Pool replaces the process but
# not the task), hanging imap forever. Grace period after the first
# second-wave sample reports in, before declaring stragglers dead.
STRAGGLER_GRACE_SEC = 900.0


# -------------------------------------------------------------------
# Result containers
# -------------------------------------------------------------------

@dataclass
class SampleResult:
    sample_idx: int                          # flat index in samples list
    pickup_local_pos_m: tuple                # (x, y, z) in pickup local frame (on mesh surface)
    pickup_world_pos_m: tuple                # corresponding world point at reference frame
    shift_mm: tuple                          # (dx, dy, dz) user-spec basis, integer mm
    probe: GraspProbeResult                  # full probe result


@dataclass
class GraspSweepResult:
    n_samples: int
    samples: list[SampleResult] = field(default_factory=list)
    n_grasped: int = 0
    best_sample_idx: int | None = None       # max peak_displacement among grasped, OR closest_distance among non-grasped
    gripper_anchor_world: tuple = (0.0, 0.0, 0.0)
    engage_frame: int = 0
    run_dir: Path | None = None


def select_winner_index(samples: list[SampleResult]) -> int | None:
    """Pick the max-displacement successful grasp, else the closest failed one.

    ``peak_displacement_m`` is the metric used by downstream candidate
    selection, so the sweep's own winner is the same sample the handoff keeps
    first. It was max ``peak_lift_m`` until 2026-08-06, which made the two
    disagree.

    The non-grasped fallback stays ``closest_distance_m``: with no grasp there
    is no displacement worth ranking, and there is no eligible candidate
    either way.
    """
    grasped = [s for s in samples if s.probe.grasped]
    if grasped:
        return max(grasped, key=lambda s: s.probe.peak_displacement_m).sample_idx
    return min(samples, key=lambda s: s.probe.closest_distance_m).sample_idx if samples else None


# -------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------

def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    R = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64))
    return R.reshape(3, 3)


def _gripper_anchor_at_frame(
    frame_cache: FrameCache, frame: int, tip_offset_m: float = 0.0,
) -> np.ndarray:
    """World-frame gripper anchor point at a trajectory frame.

    The anchor is the ``ee_frame`` site (RobotSceneSim line ~460), fixed
    at ``base_mount`` local ``[0, 0, +0.160]`` — slightly past the pad-body
    midpoint toward the fingertip, inside the pad envelope. This is what
    sweep aligns with each object surface sample.

    Pre-2026-06-29 the anchor was the silicone-pad-body midpoint, which sat
    ~33 mm *behind* ee_frame (back toward base_mount). Switching to
    ee_frame keeps the `closest_distance` metric (also ee-based) and the
    sweep alignment anchor consistent.

    Args:
        frame_cache: probe FrameCache (provides sim + qpos trajectory).
        frame: trajectory frame to evaluate the gripper pose at.
        tip_offset_m: optional extra shift along base_mount local +Z (the
            approach axis, world-frame). ``0.0`` (default) leaves the
            anchor at ee_frame; positive values push it farther past the
            pad tip; negative values pull it back toward base_mount.
    """
    sim = frame_cache.sim
    # Anchor = the COMMANDED gripper pose (arm joints forced to q_target), NOT
    # the physics qpos. A physics replay lets the commanded gripper drive toward
    # the table and, when the ground plane is placed too high (mis-calibrated
    # object/ground depth), the contact response shoves the arm back up —
    # corrupting the engage-frame ee_frame anchor (measured ~20 cm on
    # RAIL_10_02). Using q_target recovers the true trajectory gripper pose, so
    # the object↔anchor displacement reflects the real height gap; the sweep's
    # shift then lowers object + ground together to meet it. ee_frame sits on
    # base_mount, so only the 7 arm joints matter.
    qpos = np.asarray(frame_cache.qpos_trajectory[frame], dtype=np.float64).copy()
    for i, jid in enumerate(sim.joint_id_map):
        qpos[jid] = sim.q_target[frame][i]
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    anchor = np.asarray(sim.data.site("ee_frame").xpos, dtype=np.float64).copy()
    if tip_offset_m == 0.0:
        return anchor
    bm_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY,
                              sim.profile.ee_site_body)
    approach = sim.data.xmat[bm_id].reshape(3, 3)[:, 2].copy()   # gripper-base local +Z in world
    approach /= max(np.linalg.norm(approach), 1e-12)
    return anchor + float(tip_offset_m) * approach


def _pickup_pose_at_frame(
    frame_cache: FrameCache, pickup_name: str, frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pickup world pos + 3x3 rotation matrix at trajectory frame."""
    sim = frame_cache.sim
    sim.data.qpos[:] = frame_cache.qpos_trajectory[frame]
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    pen_adr = sim.model.joint(f"{pickup_name}_freejoint").qposadr.item()
    pos = sim.data.qpos[pen_adr:pen_adr + 3].copy()
    quat_wxyz = sim.data.qpos[pen_adr + 3:pen_adr + 7].copy()
    R = _quat_wxyz_to_R(quat_wxyz)
    return pos, R


def _ground_normal_at_frame(
    frame_cache: FrameCache,
    scene: CalibratedScene,
    frame: int,
) -> np.ndarray:
    """Compute the world-frame ground normal at this frame.

    For ``geometry_prior`` Stage 0's world-XY-plane mode
    (``reference == "GROUND"``), the normal is -local_down directly. For
    the legacy FP-derived ground (reference is a real scene object),
    rotate ``local_down_axis`` by that body's current world rotation.
    """
    ref_name = scene.ground.reference_object
    if ref_name == "GROUND":
        n_world = -np.asarray(scene.ground.local_down_axis, dtype=np.float64)
        return n_world / max(np.linalg.norm(n_world), 1e-12)
    sim = frame_cache.sim
    sim.data.qpos[:] = frame_cache.qpos_trajectory[frame]
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    ref_id = sim.model.body(ref_name).id
    ref_R = sim.data.xmat[ref_id].reshape(3, 3).copy()
    return ground_normal_world(ref_R, scene.ground.local_down_axis)


def _sample_pickup_mesh_surface(
    scene: CalibratedScene,
    pickup_name: str,
    n_samples: int,
    *,
    rng_seed: int = SAMPLE_SEED_DEFAULT,
) -> np.ndarray:
    """Uniformly sample ``n_samples`` points on the SCALED pickup visual mesh.

    Uses ``trimesh.sample.sample_surface`` (area-weighted face sampling), so
    the density is uniform on the surface for any geometry — pen / lid / mug
    / bowl / foil all work without per-shape tuning. Replaces the old
    long-axis × radial-XZ grid (pen-shape specific).

    Args:
        scene: CalibratedScene (provides bundle.objects[pickup].visual_mesh_path
            and .scale_path).
        pickup_name: which object to sample.
        n_samples: number of surface points to draw.
        rng_seed: numpy seed for reproducibility (default 0). Two runs with the
            same scene + seed return byte-identical samples.

    Returns:
        ``(n_samples, 3)`` float64 array, in the pickup's LOCAL frame
        (metres, scale applied).
    """
    import trimesh
    obj = next(o for o in scene.bundle.objects if o.name == pickup_name)
    mesh = trimesh.load(obj.visual_mesh_path, force="mesh")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise RuntimeError(f"empty pickup mesh: {obj.visual_mesh_path}")

    scale = np.asarray(np.load(obj.scale_path), dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.array([scale[0], scale[0], scale[0]], dtype=np.float64)
    elif scale.size != 3:
        raise ValueError(
            f"scale.npy at {obj.scale_path} has unexpected shape {scale.shape}"
        )

    # trimesh's sample_surface reads from np.random; restore state after so
    # we don't perturb any caller's RNG.
    rng_state = np.random.get_state()
    try:
        np.random.seed(int(rng_seed))
        points, _face_idx = trimesh.sample.sample_surface(mesh, int(n_samples))
    finally:
        np.random.set_state(rng_state)

    return np.asarray(points, dtype=np.float64) * scale


def _job_cpu_count() -> int:
    """Usable CPU count. Only diverges from ``os.cpu_count()`` under SLURM,
    to account for the job's ``--cpus-per-task`` cgroup allocation."""
    if os.environ.get("SLURM_JOB_ID"):
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            return max(1, int(slurm_cpus))
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _cgroup_memory_available_bytes() -> int | None:
    """(memory.max - memory.current) for this process's cgroup v2, or None
    if unconfined / cgroup v1 / unreadable."""
    try:
        with open("/proc/self/cgroup") as f:
            line = next(l for l in f if l.startswith("0::"))
        rel_path = line.strip().split(":", 2)[2]
        base = Path("/sys/fs/cgroup") / rel_path.lstrip("/")
        max_raw = (base / "memory.max").read_text().strip()
        if max_raw == "max":
            return None
        limit = int(max_raw)
        current = int((base / "memory.current").read_text().strip())
        return max(0, limit - current)
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def _available_memory_bytes() -> int:
    """Linux ``MemAvailable``. Under SLURM, additionally clamped to the
    job's own cgroup ``--mem`` budget rather than the whole shared node."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                node_available = int(line.split()[1]) * 1024
                break
        else:
            raise RuntimeError("MemAvailable not found in /proc/meminfo")

    if os.environ.get("SLURM_JOB_ID"):
        cgroup_available = _cgroup_memory_available_bytes()
        if cgroup_available is not None:
            return min(node_available, cgroup_available)
    return node_available


# -------------------------------------------------------------------
# Persistence helpers
# -------------------------------------------------------------------

def _make_run_id() -> str:
    return "sweep-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _probe_worker(
    args: tuple,
) -> tuple[int, GraspProbeResult]:
    """Run probe_grasp for one sample in a worker process.

    Module-level for spawn-mode pickleability. Each worker process imports
    its own copy of v2.skills.probe_grasp + droid_sim, builds its own
    RobotSceneSim from the (picklable) CalibratedScene + shift, runs the
    full trajectory, writes its own USD (if requested), and renders an
    mp4 from the primary cam — but only for samples where the displacement
    success criterion was met (drops the file otherwise so disk stays clean).

    Args (positional tuple):
        scene: CalibratedScene
        shift_m: (dx, dy, dz) shift in meters (user-spec basis)
        sample_idx: int
        usd_save_dir: Path | None
        usd_sample_every: int
        video_save_path: Path | None — primary-cam mp4 destination
        exclude_nonpickup_collision: bool
    """
    (scene, shift_m, sample_idx, usd_save_dir, usd_sample_every,
     video_save_path, exclude_nonpickup_collision) = args
    probe_result, _ = probe_grasp(
        scene,
        cumulative_shift_m=shift_m,
        usd_save_dir=usd_save_dir,
        usd_sample_every=usd_sample_every,
        video_save_path=video_save_path,
        exclude_nonpickup_collision=exclude_nonpickup_collision,
    )
    # Drop the mp4 unless the sample succeeded — failed-grasp videos are
    # uninteresting and would balloon disk usage across the sweep.
    if video_save_path is not None and not probe_result.grasped:
        try:
            Path(video_save_path).unlink(missing_ok=True)
        except Exception:
            pass
    return sample_idx, probe_result


def _save_summary(path: Path, res: GraspSweepResult, *,
                  save_success_videos: bool) -> None:
    """Write the sweep summary — the single source of truth per sample.

    Carries every per-sample scalar ``probe_result.yaml`` has, and the exact
    float ``cumulative_shift_mm`` the probe was run with (never the rounded
    ``shift_mm``, which flips edge grasps). ``save_success_videos`` records the
    ``grasped => grasp_video.mp4 exists`` contract used by the handoff.
    """
    doc = {
        "n_samples": res.n_samples,
        "n_grasped": res.n_grasped,
        "best_sample_idx": res.best_sample_idx,
        "gripper_anchor_world_m": list(res.gripper_anchor_world),
        "engage_frame": res.engage_frame,
        "save_success_videos": save_success_videos,
        "samples": [
            {
                "sample_idx": s.sample_idx,
                "pickup_local_pos_m": list(s.pickup_local_pos_m),
                "pickup_world_pos_m": list(s.pickup_world_pos_m),
                "cumulative_shift_mm": list(s.probe.cumulative_shift_mm),
                "grasped": s.probe.grasped,
                "grasped_loose": s.probe.grasped_loose,
                "peak_displacement_mm": s.probe.peak_displacement_m * 1000,
                "peak_lift_mm": s.probe.peak_lift_m * 1000,
                "closest_distance_mm": s.probe.closest_distance_m * 1000,
                "dist_at_peak_lift_mm": s.probe.dist_at_peak_lift_m * 1000,
                "lift_window_sec": s.probe.lift_window_sec,
                "engage_frame": s.probe.engage_frame,
                "failure_reason": s.probe.failure_reason,
            }
            for s in res.samples
        ],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


# -------------------------------------------------------------------
# Main orchestrator
# -------------------------------------------------------------------

def grasp_sweep(
    episode_dir: str | Path,
    *,
    artifacts_root: str | Path = "artifacts",
    run_id: str | None = None,
    n_samples: int = N_SAMPLES_DEFAULT,
    sample_seed: int = SAMPLE_SEED_DEFAULT,
    reference_frame: int | None = None,
    save_usd: bool = SAVE_USD_DEFAULT,
    usd_sample_every: int = USD_SAMPLE_EVERY_DEFAULT,
    n_workers: int | None = N_WORKERS_DEFAULT,
    tip_offset_m: float = 0.0,
    save_success_videos: bool = True,
    full_collision: bool = True,
) -> GraspSweepResult:
    """Brute-force grasp sweep over uniform surface samples of the pickup mesh.

    Args:
        episode_dir: built episode folder (with manifest.yaml + calibration.yaml).
        artifacts_root: parent of <run-id>/ output (default "artifacts/").
        run_id: explicit id; default = "sweep-<ts>-<hex>".
        n_samples: total uniform surface samples on the pickup visual mesh
            (default 40 — works for arbitrary geometries: pen / lid / mug /
            bowl / foil / ...).
        sample_seed: numpy seed for ``trimesh.sample.sample_surface``
            (reproducibility; default 0).
        reference_frame: trajectory frame to compute gripper_anchor + pickup
            pose against (default = engage_frame).
        save_usd: write USD per sample (default True; ~47 MB × n_samples).
        usd_sample_every: capture every Nth sim frame to USD.

    Returns:
        GraspSweepResult — full per-sample trace + best sample id.
    """
    # EGL offscreen when headless — set before the spawn Pool so every worker
    # inherits it and avoids the GLFW no-display crash + recursive-init hang.
    configure_headless_gl()

    episode_dir = Path(episode_dir).expanduser().resolve()
    artifacts_root = Path(artifacts_root).expanduser().resolve()
    run_id = run_id or _make_run_id()
    run_dir = artifacts_root / run_id
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] episode = {episode_dir}")
    print(f"[sweep] run_dir = {run_dir}")
    print(f"[sweep] n_samples = {n_samples} (uniform mesh surface), "
          f"seed = {sample_seed}, save_usd = {save_usd}")

    # ---- Stage 2: load calibration
    scene: CalibratedScene = _load_or_compute_calibration(episode_dir)
    # Only the first pickup is swept — see the module docstring's TODO.
    pickup = scene.bundle.pickup_object
    others = [p for p in scene.bundle.pickup_objects if p != pickup]
    print(f"[sweep] pickup = {pickup}, primary_cam = {scene.camera_selection.primary_id}")
    if others:
        print(f"[sweep] NOTE: episode has additional pickup objects {others} "
              f"which this sweep does not cover (single-target sweep)")

    # ---- Run an initial probe (no shift) to expose engage_frame + sim state
    print(f"[sweep] initial probe (no shift) ...")
    probe_0, fc_0 = probe_grasp(scene, cumulative_shift_m=(0, 0, 0),
                                  usd_save_dir=None,
                                  exclude_nonpickup_collision=not full_collision)
    # Decide which trajectory frame to anchor the alignment to. engage_frame
    # is where the gripper *starts* closing — pad gap still wide (~58mm). A
    # later frame (e.g. 400 for marker_in_mug) catches the gripper while
    # fingers are mid-close, often more representative of the actual grasp
    # geometry.
    if reference_frame is None:
        reference = fc_0.engage_frame
    else:
        reference = int(max(0, min(reference_frame, fc_0.n_frames - 1)))
    print(f"[sweep] engage_frame = {fc_0.engage_frame}, "
          f"reference_frame = {reference}"
          + ("" if reference == fc_0.engage_frame else "  ← user override"))

    # ---- Geometric pre-compute at reference frame
    gripper_anchor = _gripper_anchor_at_frame(fc_0, reference, tip_offset_m=tip_offset_m)
    pickup_pos, pickup_R = _pickup_pose_at_frame(fc_0, pickup, reference)
    ground_normal = _ground_normal_at_frame(fc_0, scene, reference)
    if tip_offset_m != 0.0:
        print(f"[sweep] tip_offset_m = {tip_offset_m:.4f}  "
              f"(gripper_anchor pushed +{tip_offset_m*1000:.0f}mm along base_mount +Z)")
    print(f"[sweep] gripper_anchor_world = {gripper_anchor}")
    print(f"[sweep] pickup_world_pos     = {pickup_pos}")
    print(f"[sweep] ground_normal_world  = {ground_normal}")

    # ---- Uniformly sample N points on the pickup's scaled visual mesh
    local_samples = _sample_pickup_mesh_surface(
        scene, pickup, n_samples, rng_seed=sample_seed,
    )
    total_samples = len(local_samples)
    print(f"[sweep] uniform surface sample: {total_samples} points on {pickup} mesh "
          f"(seed={sample_seed})")

    # ---- Build per-sample task table (all geometry pre-computed here)
    result = GraspSweepResult(
        n_samples=total_samples,
        gripper_anchor_world=tuple(float(x) for x in gripper_anchor),
        engage_frame=int(reference),
        run_dir=run_dir,
    )
    tasks: list[dict] = []
    for i, local_pos in enumerate(local_samples):
        world_pos = pickup_pos + pickup_R @ local_pos
        world_disp = gripper_anchor - world_pos
        shift_userspec = world_disp_to_shift(world_disp, ground_normal)
        shift_mm = tuple(int(round(x * 1000)) for x in shift_userspec)
        shift_m = tuple(float(x) for x in shift_userspec)
        sample_dir = samples_dir / sample_dir_name(i, total_samples)
        sample_dir.mkdir(exist_ok=True)
        usd_dir = sample_dir if save_usd else None
        video_path = (sample_dir / "grasp_video.mp4") if save_success_videos else None
        tasks.append({
            "i": i,
            "local_pos": local_pos, "world_pos": world_pos,
            "shift_mm": shift_mm, "shift_m": shift_m,
            "usd_dir": usd_dir,
            "video_path": video_path,
        })

    # ---- Resolve worker count
    if n_workers is None:
        n_workers = max(1, _job_cpu_count() - 1)
    available_bytes = _available_memory_bytes()
    mem_cap = max(1, available_bytes // EST_WORKER_RSS_BYTES)
    if mem_cap < n_workers:
        print(f"[sweep] capping workers {n_workers} -> {mem_cap}: "
              f"{available_bytes / 1024**3:.1f}GB available, "
              f"~{EST_WORKER_RSS_BYTES / 1024**3:.1f}GB/worker estimate "
              f"(RobotSceneSim + EGL renderer construction)")
        n_workers = mem_cap
    n_workers = max(1, min(n_workers, total_samples))
    print(f"[sweep] running {total_samples} probes across {n_workers} workers "
          f"(save_usd={save_usd}) ...")

    # ---- Parallel probe pool
    best_closest_mm = float("inf")
    results_by_idx: dict[int, GraspProbeResult] = {}
    shift_mm_by_idx = {t["i"]: t["shift_mm"] for t in tasks}

    worker_args = [
        (scene, t["shift_m"], t["i"], t["usd_dir"], usd_sample_every,
         t["video_path"], not full_collision)
        for t in tasks
    ]
    # First completion past this count proves the wave-1→wave-2 transition
    # (the slow part) is over; single-wave sweeps anchor on the 1st result.
    anchor_count = n_workers + 1 if total_samples > n_workers else 1

    # spawn (not fork) is safer with MuJoCo's OpenGL state.
    ctx = get_context("spawn")
    t_start = time.time()
    n_dead = 0
    with ctx.Pool(processes=n_workers) as pool:
        pending = {pool.apply_async(_probe_worker, (args,)): args[2]
                   for args in worker_args}
        straggler_deadline: float | None = None
        while pending:
            for async_result, sample_idx in list(pending.items()):
                if not async_result.ready():
                    continue
                del pending[async_result]
                _, probe_i = async_result.get()
                results_by_idx[sample_idx] = probe_i
                verdict = ("GRASPED ★" if probe_i.grasped
                           else "LOOSE" if probe_i.grasped_loose else "fail")
                print(f"  s{sample_idx:02d}: {verdict}  "
                      f"closest={probe_i.closest_distance_m*1000:5.1f}mm  "
                      f"disp={probe_i.peak_displacement_m*1000:6.1f}mm  "
                      f"lift={probe_i.peak_lift_m*1000:+5.1f}mm")
                if straggler_deadline is None and len(results_by_idx) >= anchor_count:
                    straggler_deadline = time.time() + STRAGGLER_GRACE_SEC

            if pending and straggler_deadline is not None and time.time() > straggler_deadline:
                for sample_idx in pending.values():
                    print(f"  s{sample_idx:02d}: TIMEOUT — worker presumed dead "
                          f"({STRAGGLER_GRACE_SEC:.0f}s past first post-wave-1 "
                          f"result), marking failed")
                    results_by_idx[sample_idx] = GraspProbeResult(
                        grasped=False, grasped_loose=False,
                        failure_reason="worker_timeout",
                        peak_displacement_m=0.0, peak_lift_m=0.0,
                        peak_lift_time_frac=0.0, lift_window_sec=0.0,
                        dist_at_peak_lift_m=0.0, closest_distance_m=float("inf"),
                        duration_sec=0.0, engage_frame=int(reference),
                        cumulative_shift_mm=shift_mm_by_idx[sample_idx],
                    )
                    n_dead += 1
                pending.clear()
            elif pending:
                time.sleep(0.5)
    parallel_elapsed = time.time() - t_start
    print(f"\n[sweep] {n_samples} probes in {parallel_elapsed:.1f}s "
          f"(~{parallel_elapsed/n_samples:.1f}s/probe avg, "
          f"{n_workers}x parallelism)"
          + (f", {n_dead} worker-timeout" if n_dead else ""))

    # ---- Persist artifacts in deterministic order
    for t in tasks:
        i = t["i"]
        probe_i = results_by_idx[i]
        sample = SampleResult(
            sample_idx=i,
            pickup_local_pos_m=tuple(float(x) for x in t["local_pos"]),
            pickup_world_pos_m=tuple(float(x) for x in t["world_pos"]),
            shift_mm=t["shift_mm"],
            probe=probe_i,
        )
        result.samples.append(sample)

        if probe_i.grasped:
            result.n_grasped += 1
        if probe_i.closest_distance_m * 1000 < best_closest_mm:
            best_closest_mm = probe_i.closest_distance_m * 1000

    result.best_sample_idx = select_winner_index(result.samples)
    _save_summary(run_dir / "summary.yaml", result,
                  save_success_videos=save_success_videos)

    print()
    print(f"[sweep] DONE — {result.n_grasped}/{result.n_samples} grasped, "
          f"best_sample_idx={result.best_sample_idx}, best_closest={best_closest_mm:.1f}mm")
    print(f"[sweep] summary: {run_dir / 'summary.yaml'}")
    return result


__all__ = ["grasp_sweep", "GraspSweepResult", "SampleResult"]
