"""Stage 3 skill: grasp probe with optional shift + per-frame state cache.

Runs the recorded trajectory in a `RobotSceneSim` and returns:
  1. ``GraspProbeResult`` — success-criterion verdict + diagnostic fields
  2. ``FrameCache`` — per-frame qpos snapshot + the live sim object, so
     ``render_view`` (Stage 3 retry loop) can restore the sim to any frame
     and render custom views without replaying the whole trajectory.

v1 success criteria (constants from ar2s/droid_sim/grasp_probe.py):
  - ``grasped`` (tight):    pickup rose ≥ LIFT_THRESHOLD_M, lift sustained
                             ≥ MIN_LIFT_DURATION_S, AND |TCP - obj_origin|
                             ≤ MAX_DIST_AT_LIFT_M at the lift apex.
  - ``grasped_loose``:       lift + duration only.

v1 hack preserved: robot-vs-object collision is kept only for the pickup
objects, so the gripper can pass through container walls (e.g., pen-in-mug).
All of ``bundle.pickup_objects`` are passed, not just the probe target —
an object the robot demonstrably grasps must not be phantom to the gripper.

Shift application: ``cumulative_shift`` is written into ``cfg.phase5_shift``
before building the sim. RobotSceneSim translates all object trajectories
by the user-spec basis displacement (dx/dy along ground plane, dz vertical).
The ground itself is fixed; trajectories shift through it.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

# Constants + lift analysis helpers — shared library, OK to import.
from ar2s.droid_sim.grasp_probe import (
    DISPLACEMENT_THRESHOLD_M,
    LIFT_THRESHOLD_M,
    MAX_DIST_AT_LIFT_M,
    MIN_LIFT_DURATION_S,
    _analyse_lift,
    _build_first_contact_info,
    _build_pickup_geom_set,
    _detect_first_contact,
    _find_close_complete_frame,
    _find_engage_frame,
)
from ar2s.droid_sim.scene.robot_scene_sim import RobotSceneSim
from ar2s.droid_sim.usd_exporter import VertexColorUSDExporter

from ar2s.agents_sysid.scene import CalibratedScene
from ar2s.agents_sysid.scene_config import build_scene_config
from ar2s.droid_sim.scene.robot_profile import get_robot_profile


@dataclass
class FirstContact:
    """First gripper-pickup contact detected during the close phase.

    Used by DET shift heuristic (round 0 of grasp loop): the geometric
    direction from contact-point toward the gripper TCP midpoint, expressed
    in user-spec basis (dx/dy on ground plane, dz vertical), tells us how
    to shift the scene so the next probe's gripper actually engages the
    pickup body instead of brushing past it.

    None when no gripper↔pickup contact occurred during the close window
    (e.g. the gripper missed the pickup entirely).
    """
    frame_idx: int
    side: str                                    # "inner" | "outer" | "tip" | "side"
    body_name: str                               # e.g. "right_pad"
    world_pos: tuple                             # (x, y, z) — contact point in world
    gripper_frame_pos: tuple                     # contact in gripper TCP frame
    corrective_shift_m: tuple                    # (dx, dy, dz) in m, user-spec basis
                                                  #   — full geometric magnitude, NOT
                                                  #   yet clipped to per-round / cum caps


@dataclass
class GraspProbeResult:
    # ---- verdict (3D displacement criterion, 2026-06-06) ----
    grasped: bool
    grasped_loose: bool
    failure_reason: str | None

    # ---- displacement / lift trace ----
    peak_displacement_m: float        # 3D Euclidean displacement (verdict input)
    peak_lift_m: float                # z-only delta at the displacement peak frame
    peak_lift_time_frac: float
    lift_window_sec: float            # legacy field, always 0 with new criterion
    dist_at_peak_lift_m: float
    closest_distance_m: float
    duration_sec: float
    engage_frame: int

    # ---- reference quats at engage_frame (wxyz, world) ----
    pen_quat_at_engage: tuple = (1.0, 0.0, 0.0, 0.0)
    gripper_quat_at_engage: tuple = (1.0, 0.0, 0.0, 0.0)

    # ---- input echo (for audit + downstream history table) ----
    cumulative_shift_mm: tuple = (0.0, 0.0, 0.0)

    # ---- first gripper↔pickup contact (DET shift signal) ----
    first_contact: FirstContact | None = None

    criteria: dict = field(default_factory=lambda: {
        "displacement_threshold_m": DISPLACEMENT_THRESHOLD_M,
        # legacy / diagnostic — no longer gating
        "lift_threshold_m": LIFT_THRESHOLD_M,
        "min_lift_duration_s": MIN_LIFT_DURATION_S,
        "max_dist_at_lift_m": MAX_DIST_AT_LIFT_M,
    })


@dataclass
class FrameCache:
    """Per-frame sim state from a probe run — fast random-access render.

    The ``sim`` field holds the LIVE sim object the probe ran on. ``render_view``
    sets ``sim.data.qpos`` to the row of ``qpos_trajectory`` at the target frame,
    calls ``mj_forward``, then renders. After rendering, the sim is left in
    that state; the next render restores qpos again from the cache.

    Single-round lifetime: a new probe creates a new sim + new FrameCache; the
    previous round's cache is discarded.
    """
    sim: RobotSceneSim
    qpos_trajectory: np.ndarray            # (n_frames, n_qpos)
    n_frames: int
    frame_dt: float
    engage_frame: int
    video_sample_every: int = 3            # Nth-sim-frame video sampling used
                                           #   by probe_grasp (grasp_video.mp4)


def _classify_failure(lite: dict) -> str | None:
    if lite["grasped"]:
        return None
    peak_disp = float(lite.get("peak_displacement", 0.0))
    return (f"pickup displacement insufficient: peak {peak_disp*1000:.1f}mm "
            f"< {DISPLACEMENT_THRESHOLD_M*1000:.0f}mm")


def _build_sim_cfg_for_probe(
    scene: CalibratedScene,
    cumulative_shift_m: tuple[float, float, float],
):
    """Build SceneConfig with Stage-2 calibration + phase5_shift applied."""
    cfg = build_scene_config(scene.bundle)
    cfg.cameras = replace(
        cfg.cameras,
        object_camera_id=scene.camera_selection.primary_id,
    )
    cfg.ground = replace(
        cfg.ground,
        local_down_axis=scene.ground.local_down_axis,
        offset=scene.ground.offset,
    )
    cfg.phase5_shift = tuple(float(s) for s in cumulative_shift_m)
    return cfg


def _make_usd_exporter(sim: RobotSceneSim, save_dir: Path):
    """Build a colored USD exporter rooted at ``save_dir/usd/`` (scratch dir;
    flattened by ``_finalise_usd_export`` into ``save_dir/result.usd``)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    return VertexColorUSDExporter(
        model=sim.model,
        output_directory="usd",
        output_directory_root=str(save_dir),
        verbose=False,
        object_visual_mesh_paths={
            f"{obj.name}_visual": obj.visual_mesh_path
            for obj in sim.config.objects
        },
    )


def _finalise_usd_export(
    usd_exporter: VertexColorUSDExporter,
    save_dir: Path,
) -> None:
    """Save the accumulated USD scene and flatten it to ``save_dir/result.usd``.

    The exporter writes into ``<save_dir>/usd/frames/frame_<N>.usd``
    (N = total frame count) plus an always-empty ``usd/assets/`` (this scene
    has no MuJoCo textures) — both hardcoded by the library. We move the
    single exported file up and discard the now-empty scratch directory.
    """
    usd_exporter.save_scene()
    usd_dir = save_dir / "usd"
    exported = usd_dir / "frames" / f"frame_{usd_exporter.frame_count}.usd"
    exported.rename(save_dir / "result.usd")
    shutil.rmtree(usd_dir)


# USD shows the optimized VISUAL meshes only (default MjvOption shows 0..2;
# our object visual meshes live in groups 4-5). Both collision groups are
# hidden: group 3 is the Franka collision proxies, group 0 the per-object
# coacd sub-meshes. See RobotSceneSim.GEOM_GROUP_* for the convention.
_USD_HIDDEN_GROUPS = {
    RobotSceneSim.GEOM_GROUP_OBJECT_COLLISION,
    RobotSceneSim.GEOM_GROUP_ROBOT_COLLISION,
}
_USD_SCENE_OPT = mujoco.MjvOption()
for _g in range(len(_USD_SCENE_OPT.geomgroup)):
    _USD_SCENE_OPT.geomgroup[_g] = 0 if _g in _USD_HIDDEN_GROUPS else 1


def robot_collision_filter(
    pickup_objects: list[str], pickup: str, *, full_collision: bool,
) -> tuple[str, ...]:
    """Names that stay robot-solid in reduced-collision mode.

    An empty tuple means full robot-vs-all-object collision in
    ``RobotSceneSim``.  Reduced mode never drops a second pickup object.
    """
    return () if full_collision else tuple(pickup_objects or [pickup])


def probe_grasp(
    scene: CalibratedScene,
    *,
    cumulative_shift_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    usd_save_dir: str | Path | None = None,
    usd_sample_every: int = 1,
    video_save_path: str | Path | None = None,
    video_sample_every: int = 3,
    video_fps: float = 30.0,
    video_width: int | None = None,          # None = the episode's native sensor size
    video_height: int | None = None,
    exclude_nonpickup_collision: bool = True,
) -> tuple[GraspProbeResult, FrameCache]:
    """Run a grasp probe with optional scene shift; cache per-frame state.

    Args:
        scene: CalibratedScene from Stage 2.
        cumulative_shift_m: (dx, dy, dz) in METERS — user-spec basis (dx/dy
            project onto ground, dz vertical). Applied as ``cfg.phase5_shift``
            before building the sim.
        usd_save_dir: if set, also accumulate a USD scene capturing every
            sim frame into ``<usd_save_dir>/result.usd``. The USD can be
            opened later in Houdini / Blender / usdview to render the full
            grasp animation. None = no USD export (zero overhead).
        usd_sample_every: capture every Nth sim frame to USD (default 1 =
            every frame). Bump to 5 or 10 if disk is a concern.
        video_save_path: if set, render the PRIMARY camera viewpoint into
            an mp4 written to this path. Adds ~50 ms / sampled frame so
            ``video_sample_every`` defaults to 3 (~10 fps output for a
            30 fps sim). Independent of usd_save_dir — set either, both,
            or neither.
        video_sample_every: capture every Nth sim frame to video.
        video_fps: output mp4 framerate (frames written → video at this
            wall-clock rate).
        video_width / video_height: render resolution. None (default) uses the
            camera's native size — render_object_camera crops around the true
            principal point, so a smaller request slides the window off-centre
            rather than zooming out.

    Returns:
        ``(GraspProbeResult, FrameCache)`` — never raises on grasp failure;
        verdict lives in ``GraspProbeResult.grasped`` / ``failure_reason``.
    """
    bundle = scene.bundle
    pickup = bundle.pickup_object
    # Every grasped object keeps robot collision, even though only `pickup` is
    # probed — a secondary pickup the gripper passes through would fall through
    # the fingers mid-sim. Falls back to the probe target for bundles built
    # directly (tests / hand-written inputs) that never set pickup_objects.
    cfg = _build_sim_cfg_for_probe(scene, cumulative_shift_m)

    sim = RobotSceneSim(
        cfg,
        is_kinematic=False,
        robot_base_pos=scene.robot_alignment.pos,
        robot_base_quat=scene.robot_alignment.quat_wxyz,
        robot_profile=get_robot_profile(scene.bundle.robot_type),
        # Empty tuple keeps robot collision against every object.  In the
        # reduced-collision mode, all pickup objects remain solid to the robot;
        # only non-pickup scene objects are excluded.
        exclude_robot_nonpickup_collision=robot_collision_filter(
            bundle.pickup_objects, pickup,
            full_collision=not exclude_nonpickup_collision,
        ),
    )

    usd_exporter = None
    if usd_save_dir is not None:
        usd_exporter = _make_usd_exporter(sim, Path(usd_save_dir))

    # Optional primary-cam video. Streamed frame-by-frame to an ffmpeg
    # writer rather than buffered in a list — buffering ~700 frames at
    # 1280x720 (~1.85GB/worker) across a grasp_sweep's parallel pool was
    # OOM-killing the parent process.
    video_writer = None
    video_frame_count = 0
    primary_cam_idx: int | None = None
    if video_save_path is not None:
        try:
            primary_cam_idx = list(sim.camera_ids).index(sim.object_camera_id)
        except ValueError:
            print(f"[probe_grasp] video_save_path set but object_camera_id "
                  f"{sim.object_camera_id} not in camera_ids {list(sim.camera_ids)}; "
                  f"disabling video")

    # Resolve the video size once: None means the episode's native sensor size.
    _sz = getattr(sim.config.cameras, "image_size", None) or (1280, 720)
    vid_w = int(video_width) if video_width else int(_sz[0])
    vid_h = int(video_height) if video_height else int(_sz[1])

    # ----- run the trajectory per-frame, cache qpos -----
    n_frames = int(sim.q_target.shape[0])
    engage_frame = _find_engage_frame(sim)
    close_complete_frame = _find_close_complete_frame(sim, start=engage_frame)
    pickup_geom_ids = _build_pickup_geom_set(sim.model, pickup)
    sim.reset()
    pen_adr = sim.model.joint(f"{pickup}_freejoint").qposadr.item()

    pen_pos_trace = np.empty((n_frames, 3), dtype=np.float64)
    dist_trace = np.empty(n_frames, dtype=np.float64)
    qpos_traj = np.empty((n_frames, sim.model.nq), dtype=np.float64)
    closest = float("inf")
    pen_quat_eng: tuple = (1.0, 0.0, 0.0, 0.0)
    grip_quat_eng: tuple = (1.0, 0.0, 0.0, 0.0)
    first_contact_raw = None  # ar2s.agents_sysid.state.FirstContactInfo or None

    for frame in range(n_frames):
        sim.sim_one_frame()
        qpos_traj[frame] = sim.data.qpos
        pen_pos = sim.data.qpos[pen_adr:pen_adr + 3]
        pen_pos_trace[frame] = pen_pos
        ee = np.array(sim.data.site("ee_frame").xpos, dtype=np.float64)
        d = float(np.linalg.norm(ee - pen_pos))
        dist_trace[frame] = d
        if d < closest:
            closest = d
        if frame == engage_frame:
            qp = sim.data.qpos[pen_adr + 3:pen_adr + 7]
            pen_quat_eng = (float(qp[0]), float(qp[1]), float(qp[2]), float(qp[3]))
            R = np.array(sim.data.site("ee_frame").xmat, dtype=np.float64).reshape(3, 3)
            qg = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(qg, R.flatten())
            grip_quat_eng = (float(qg[0]), float(qg[1]), float(qg[2]), float(qg[3]))

        # First gripper↔pickup contact during close window — used by DET shift.
        if first_contact_raw is None and frame <= close_complete_frame:
            hit = _detect_first_contact(sim, pickup, pickup_geom_ids, frame)
            if hit is not None:
                _, body_name, side, world_pos = hit
                first_contact_raw = _build_first_contact_info(
                    sim, frame, body_name, side, world_pos,
                )

        if usd_exporter is not None and (frame % usd_sample_every == 0):
            try:
                usd_exporter.update_scene(sim.data, scene_option=_USD_SCENE_OPT)
            except Exception as e:
                print(f"[probe_grasp] USD update failed at frame {frame}: {e}; disabling USD")
                usd_exporter = None

        if primary_cam_idx is not None and (frame % video_sample_every == 0):
            try:
                rendered = sim.render_object_camera(
                    width=vid_w, height=vid_h,
                )
                if video_writer is None:
                    import imageio.v2 as imageio
                    out_path = Path(video_save_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    video_writer = imageio.get_writer(
                        str(out_path), fps=float(video_fps), quality=8,
                        codec="libx264",
                    )
                video_writer.append_data(rendered)
                video_frame_count += 1
            except Exception as e:
                print(f"[probe_grasp] video render failed at frame {frame}: "
                      f"{e}; disabling video")
                if video_writer is not None:
                    video_writer.close()
                    Path(video_save_path).unlink(missing_ok=True)
                    video_writer = None
                primary_cam_idx = None

    # Free the reused video GL context promptly (don't wait for sim GC).
    sim.close_object_camera_renderer()

    if usd_exporter is not None:
        try:
            _finalise_usd_export(usd_exporter, Path(usd_save_dir))
        except Exception as e:
            print(f"[probe_grasp] USD save_scene failed: {e}")

    if video_writer is not None:
        try:
            video_writer.close()
            print(f"[probe_grasp] video saved: {video_save_path} "
                  f"({video_frame_count} frames @ {video_fps:.1f} fps)")
        except Exception as e:
            print(f"[probe_grasp] video encode failed: {e}")

    lift = _analyse_lift(
        pen_pos_trace, dist_trace, sim.frame_dt,
        DISPLACEMENT_THRESHOLD_M, MIN_LIFT_DURATION_S, MAX_DIST_AT_LIFT_M,
    )
    lite = {
        **lift,
        "closest_distance": float(closest),
        "duration_sec": float(n_frames * sim.frame_dt),
        "engage_frame": int(engage_frame),
        "pen_quat_at_engage": pen_quat_eng,
        "gripper_quat_at_engage": grip_quat_eng,
    }

    # Convert droid_sim FirstContactInfo into the Stage 3 FirstContact type.
    first_contact = None
    if first_contact_raw is not None:
        first_contact = FirstContact(
            frame_idx=int(first_contact_raw.frame_idx),
            side=str(first_contact_raw.side),
            body_name=str(first_contact_raw.body_name),
            world_pos=tuple(float(x) for x in first_contact_raw.world_pos),
            gripper_frame_pos=tuple(float(x) for x in first_contact_raw.gripper_frame_pos),
            corrective_shift_m=tuple(float(x) for x in first_contact_raw.corrective_shift),
        )

    result = GraspProbeResult(
        grasped=bool(lite["grasped"]),
        grasped_loose=bool(lite["grasped_loose"]),
        failure_reason=_classify_failure(lite),
        peak_displacement_m=float(lite.get("peak_displacement", 0.0)),
        peak_lift_m=float(lite["peak_lift"]),
        peak_lift_time_frac=float(lite["peak_lift_time_frac"]),
        lift_window_sec=float(lite["lift_window_sec"]),
        dist_at_peak_lift_m=float(lite["dist_at_peak_lift"]),
        closest_distance_m=float(lite["closest_distance"]),
        duration_sec=float(lite["duration_sec"]),
        engage_frame=int(lite["engage_frame"]),
        pen_quat_at_engage=pen_quat_eng,
        gripper_quat_at_engage=grip_quat_eng,
        cumulative_shift_mm=tuple(float(s * 1000) for s in cumulative_shift_m),
        first_contact=first_contact,
    )

    frame_cache = FrameCache(
        sim=sim,
        qpos_trajectory=qpos_traj,
        n_frames=n_frames,
        frame_dt=float(sim.frame_dt),
        engage_frame=int(engage_frame),
        video_sample_every=int(video_sample_every),
    )

    return result, frame_cache


def get_scene_state_at_frame(
    frame_cache: FrameCache,
    pickup_object: str,
    frame: int | None = None,
) -> dict:
    """Read pickup + gripper world positions at a specific trajectory frame.

    Used by Stage 3 Agent B (shift_decision) to expose engage-frame geometry
    in textual form alongside the rendered images. Restores sim qpos, calls
    mj_forward, reads the freejoint position + ee_frame site.

    Args:
        frame_cache: from probe_grasp.
        pickup_object: which SceneObject's pose to return.
        frame: trajectory frame; default = engage_frame.

    Returns:
        {"pickup_pos": (x, y, z), "gripper_pos": (x, y, z), "frame": int}
    """
    if frame is None:
        frame = frame_cache.engage_frame
    sim = frame_cache.sim
    sim.data.qpos[:] = frame_cache.qpos_trajectory[frame]
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    adr = sim.model.joint(f"{pickup_object}_freejoint").qposadr.item()
    pickup_pos = sim.data.qpos[adr:adr + 3].copy()
    gripper_pos = np.array(sim.data.site("ee_frame").xpos, dtype=np.float64)
    return {
        "pickup_pos": tuple(float(x) for x in pickup_pos),
        "gripper_pos": tuple(float(x) for x in gripper_pos),
        "frame": int(frame),
    }


def compute_det_shift_mm(
    probe_result: GraspProbeResult,
    cumulative_shift_mm: tuple[int, int, int],
    *,
    cumulative_cap_mm: int = 50,
) -> tuple[int, int, int]:
    """Deterministic shift from `first_contact` (v1's round-0 DET strategy).

    Returns the geometric correction (gripper midpoint toward contact, in
    user-spec basis, integer mm), clipped only by the **cumulative** cap
    (no per-round cap — the direction is well-determined, so we apply the
    full geometric move). Returns ``(0, 0, 0)`` when no contact was
    detected (caller can fall through to LLM in that case).

    Args:
        probe_result: the probe whose first_contact we're acting on.
        cumulative_shift_mm: shift accumulated up to (but not including)
            this round, in mm.
        cumulative_cap_mm: per-axis cap on cumulative shift in mm.

    Returns:
        (dx, dy, dz) in integer mm.
    """
    fc = probe_result.first_contact
    if fc is None or fc.side == "none":
        return (0, 0, 0)

    # corrective_shift_m is in METERS, user-spec basis, full magnitude.
    sx_m, sy_m, sz_m = fc.corrective_shift_m
    cx, cy, cz = cumulative_shift_mm   # mm
    cap = int(cumulative_cap_mm)

    def _clip_cum(s_m: float, c_mm: int) -> int:
        s_mm = int(round(s_m * 1000))
        # keep cum + s within ±cap
        return int(np.clip(s_mm, -cap - c_mm, cap - c_mm))

    return (_clip_cum(sx_m, cx), _clip_cum(sy_m, cy), _clip_cum(sz_m, cz))


__all__ = [
    "probe_grasp",
    "GraspProbeResult",
    "FrameCache",
    "FirstContact",
    "get_scene_state_at_frame",
    "compute_det_shift_mm",
]
