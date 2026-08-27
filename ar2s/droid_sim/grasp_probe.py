"""Phase 5 grasp probe — dynamic-physics layer.

Runs the full recorded robot trajectory in `RobotSceneSim(is_kinematic=False)`
mode (true contact dynamics). At three keyframes — engage_frame (gripper
TCP at lowest world z), engage+N, engage+2N — captures images and
per-frame poses.

Per keyframe we render 5 images:
  - real_cam_<id1>, real_cam_<id2>  : both calibrated cameras, full scene
  - view_a, view_b                  : ViewFinder-chosen oblique views, full scene
  - view_a_pickup_only              : view_a re-rendered with all NON-pickup
                                      objects temporarily moved off-scene, so the
                                      pickup object (e.g. pen inside a mug) is
                                      visible without container occlusion.
All renders include the world-frame coordinate axes overlay (red=+x,
green=+y, blue=+z) for direction-call sanity by the LLM.

Lift analysis (tight success criterion):
  pen.z rose ≥ LIFT_THRESHOLD_M above its initial z, sustained ≥
  MIN_LIFT_DURATION_S, AND at the lift apex moment, |gripper - pen| <
  MAX_DIST_AT_LIFT_M. The distance check rules out wrong-object grasps
  (gripper holding the container, pen rising along with the container
  but not actually held).
"""
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from ar2s.agents_sysid.state import FirstContactInfo, FrameSnapshot, GraspProbeReport
from ar2s.droid_sim.scene.robot_scene_sim import RobotSceneSim
from ar2s.droid_sim.scene.shift import ground_normal_world, world_disp_to_shift
from ar2s.droid_sim.scene.views import CameraPose
from ar2s.droid_sim.usd_exporter import VertexColorUSDExporter


# Success criterion (revised 2026-06-06): peak 3D displacement.
# v1 used a sustained lift-in-z criterion (peak z-lift ≥ 30 mm sustained ≥ 0.2 s
# with gripper close to object at the apex). Real droid_300 demos include
# successful pushes, slides, and re-orientations that have minimal z-lift,
# so the criterion now reads: max over the trajectory of
# ||pickup_pos(t) − pickup_pos(0)|| ≥ DISPLACEMENT_THRESHOLD_M.
# The old lift-only fields (LIFT_THRESHOLD_M etc.) are kept as diagnostic
# constants but no longer gate the verdict.
DISPLACEMENT_THRESHOLD_M = 0.05    # any-axis displacement to count as "grasped"

# Diagnostic / legacy — exposed in result fields for back-compat, NOT used
# in the verdict.
LIFT_THRESHOLD_M = 0.03
MIN_LIFT_DURATION_S = 0.2
MAX_DIST_AT_LIFT_M = 0.08

# Frame sampling — 3 frames starting at engage frame, every N frames.
KEYFRAME_OFFSET_FRAMES = 30        # ~0.5 s @ 60 fps

# Threshold on gripper close-command norm (q_target[:, 7] / 255) above which
# the gripper is considered "closing" (grasp event detection).
_CLOSE_THRESHOLD = 0.5

# Threshold above which the gripper is considered "fully closed" — used to
# stop the first-contact detector. Contacts that happen AFTER the gripper has
# finished closing are not relevant for the outer-sweep failure mode (those
# are post-grasp dynamics: pen rising / falling / being held).
_CLOSE_COMPLETE_THRESHOLD = 0.95

# Render sizes
# Fallback only. A real camera must be rendered at its NATIVE size:
# render_cameras crops a window around the true principal point, so asking for
# a smaller frame slides that window off-centre rather than zooming out. The
# episode's actual size comes from CameraSpec.image_size (see _real_cam_size).
_REAL_CAM_W = 1280
_REAL_CAM_H = 720
_FREE_CAM_W = 1280
_FREE_CAM_H = 720

# Where to park hidden objects
_OFFSCREEN_POS = np.array([100.0, 100.0, 100.0], dtype=np.float64)


# -------------------------------------------------------------------
# First-contact (outer-finger sweep) detection
# -------------------------------------------------------------------

# Robotiq 2F-85 body-name prefixes that identify the gripper's bodies.
# Anything matching is considered "gripper". Used to test whether a contact
# involves the gripper at all.
_GRIPPER_BODY_PREFIXES = (
    "base_mount", "base",                                            # wrist
    "right_driver", "right_coupler", "right_spring_link",
    "right_follower", "right_pad", "right_silicone_pad",
    "left_driver", "left_coupler", "left_spring_link",
    "left_follower", "left_pad", "left_silicone_pad",
)

# Substring on a gripper body name that classifies it as inner-pad surface.
# Anything else under the gripper (driver / coupler / follower / spring_link
# / base) is OUTER and is the failure mode we want to detect.
_INNER_PAD_KEY = "pad"


def _real_cam_size(sim) -> tuple[int, int]:
    """Native (W, H) for real-camera renders, or the DROID default."""
    size = getattr(sim.config.cameras, "image_size", None)
    if size:
        return int(size[0]), int(size[1])
    return _REAL_CAM_W, _REAL_CAM_H


def _classify_gripper_body(
    body_name: str, prefixes: tuple[str, ...], inner_key: str
) -> str:
    """Returns 'inner' / 'outer' / '' (not a gripper body). Prefixes and the
    inner-pad key come from the robot profile (see RobotProfile)."""
    if not any(body_name == p or body_name.startswith(p) for p in prefixes):
        return ""
    return "inner" if inner_key in body_name else "outer"


def _build_pickup_geom_set(model: mujoco.MjModel, pickup_body_name: str) -> set:
    """Return a set of geom ids that belong to the pickup object's body."""
    body = model.body(pickup_body_name)
    geom_ids: set[int] = set()
    # Iterate all geoms; pick those whose bodyid equals the pickup body's id.
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == body.id:
            geom_ids.add(gid)
    return geom_ids


def _closedness_trace(sim: RobotSceneSim):
    """Per-frame gripper closedness, 0.0 = fully open .. 1.0 = fully closed."""
    norm = sim.q_target[:, sim.arm_dof] / sim.profile.gripper_ctrl_scale
    return sim.profile.closedness(norm)


def _find_close_complete_frame(sim: RobotSceneSim, start: int = 0) -> int:
    """First frame at or after `start` where the gripper has finished closing.

    Contact detection is restricted to frames `<=` this index — anything later
    is post-grasp dynamics and not actionable for the outer-sweep correction.

    `start` should be the engage frame. Searching from 0 instead would return
    the pre-grasp closed period on a demo that starts closed (AIRBOT ep3 would
    answer 0 and pin contact detection to the very first frame).

    Returns `n_frames` (never within range) when the gripper never fully
    closes — either a sliding/no-grasp demo, or a grasp where the object keeps
    the fingers apart (AIRBOT ep3 holds the spoon at closedness 0.82).
    """
    n_frames = int(sim.q_target.shape[0])
    start = max(0, min(int(start), n_frames))
    above = _closedness_trace(sim)[start:] >= _CLOSE_COMPLETE_THRESHOLD
    if above.any():
        return start + int(np.argmax(above))
    return n_frames


def _detect_first_contact(
    sim: RobotSceneSim,
    pickup_object: str,
    pickup_geom_ids: set,
    frame: int,
) -> tuple[int, str, str, np.ndarray] | None:
    """Scan `sim.data.contact[]` for the first gripper↔pickup contact.

    Returns (geom_other, gripper_body_name, side, world_pos) for the first
    such contact, or None if no gripper-pickup contact this step.
    """
    model = sim.model
    data = sim.data
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in pickup_geom_ids:
            other = g2
        elif g2 in pickup_geom_ids:
            other = g1
        else:
            continue
        body_id = int(model.geom_bodyid[other])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        side = _classify_gripper_body(
            body_name,
            sim.profile.gripper_body_prefixes,
            sim.profile.gripper_inner_pad_key,
        )
        if side:
            return (other, body_name, side, np.array(c.pos, dtype=np.float64))
    return None


def _world_to_gripper_frame(
    sim: RobotSceneSim,
    p_world: np.ndarray,
) -> np.ndarray:
    """Convert a world-frame point to the gripper TCP frame."""
    ee = sim.data.site("ee_frame")
    pos = np.array(ee.xpos, dtype=np.float64)
    R = np.array(ee.xmat, dtype=np.float64).reshape(3, 3)
    return R.T @ (p_world - pos)


def _build_first_contact_info(
    sim: RobotSceneSim,
    frame: int,
    body_name: str,
    side: str,
    world_pos: np.ndarray,
) -> FirstContactInfo:
    """Build a FirstContactInfo from raw detection + compute corrective shift.

    Corrective direction in world frame: ``ee_world - contact_world``
    (from contact toward gripper midpoint). Decomposed into user-spec basis.
    Magnitude = the in-plane component of the contact offset (so the shift
    actually closes the gap; downstream may clamp by per-round limit).
    """
    ee_world = np.array(sim.data.site("ee_frame").xpos, dtype=np.float64)
    contact_gripper = _world_to_gripper_frame(sim, world_pos)

    # Corrective direction: contact → mid (= ee_world)
    v_correction_world = ee_world - world_pos

    # Ground normal. For the world-XY-plane mode set by ``geometry_prior``
    # Stage 0 (``reference == "GROUND"``), the normal is -local_down directly
    # (no reference object to rotate). For the FP-derived ground (legacy
    # reference is a real scene object), rotate ``local_down_axis`` by the
    # reference body's current world rotation.
    ref_name = sim.config.ground.reference_object
    if ref_name == "GROUND":
        n_world = -np.asarray(sim.config.ground.local_down_axis, dtype=np.float64)
        n_world /= max(np.linalg.norm(n_world), 1e-12)
    else:
        ref_id = sim.model.body(ref_name).id
        ref_R = np.array(sim.data.xmat[ref_id], dtype=np.float64).reshape(3, 3)
        n_world = ground_normal_world(ref_R, sim.config.ground.local_down_axis)

    dx, dy, dz = world_disp_to_shift(v_correction_world, n_world)

    return FirstContactInfo(
        frame_idx=int(frame),
        side=side,
        body_name=body_name,
        world_pos=tuple(float(v) for v in world_pos),
        gripper_frame_pos=tuple(float(v) for v in contact_gripper),
        corrective_shift=(float(dx), float(dy), float(dz)),
    )


def _analyse_lift(
    pen_pos_trace: np.ndarray,
    gripper_pen_dist_trace: np.ndarray,
    sim_dt: float,
    displacement_thresh: float,
    min_duration_s: float,
    max_dist_at_lift_m: float,
) -> dict:
    """3D-displacement success criterion (2026-06-06): the pickup is
    considered grasped iff at some frame ``||pos(t) − pos(0)|| ≥ thresh``.

    Args:
        pen_pos_trace: (N, 3) world-frame pickup position per frame.
        gripper_pen_dist_trace: (N,) gripper-TCP ↔ pickup distance.
        sim_dt: per-frame timestep.
        displacement_thresh: success cutoff in meters.
        min_duration_s, max_dist_at_lift_m: kept for back-compat (no longer
            gate the verdict — just reported for the diagnostic dict).

    Returns dict with:
        grasped (3D-displacement verdict), grasped_loose (same alias),
        peak_displacement (max 3D distance), peak_lift (z-component delta
        at the displacement-peak frame, diagnostic), peak_lift_time_frac,
        lift_window_sec (legacy field, zero), dist_at_peak_lift (gripper
        distance at peak-displacement frame).
    """
    pen_pos_trace = np.asarray(pen_pos_trace, dtype=np.float64)
    if pen_pos_trace.ndim != 2 or pen_pos_trace.shape[1] != 3:
        raise ValueError(
            f"pen_pos_trace must be (N, 3); got {pen_pos_trace.shape}"
        )
    p0 = pen_pos_trace[0]
    disp = np.linalg.norm(pen_pos_trace - p0, axis=1)
    peak_idx = int(np.argmax(disp))
    peak_disp = float(disp[peak_idx])
    z_delta_at_peak = float(pen_pos_trace[peak_idx, 2] - p0[2])
    dist_at_peak = float(gripper_pen_dist_trace[peak_idx])

    grasped = peak_disp >= displacement_thresh
    return {
        "grasped": grasped,
        "grasped_loose": grasped,           # alias kept for back-compat
        "peak_displacement": peak_disp,
        # ---- legacy / diagnostic fields ----
        "peak_lift": z_delta_at_peak,
        "peak_lift_time_frac": float(peak_idx / max(len(disp) - 1, 1)),
        "lift_window_sec": 0.0,
        "dist_at_peak_lift": dist_at_peak,
    }


def _render_view_pickup_only(
    sim: RobotSceneSim,
    pickup: str,
    view: CameraPose,
) -> np.ndarray:
    """Render view_free_camera with all NON-pickup scene objects temporarily
    moved off-screen (qpos translated to ~100m). Restores qpos on exit.
    """
    saved: dict[str, tuple[int, np.ndarray]] = {}
    for obj in sim.config.objects:
        if obj.name == pickup or obj.static:
            continue
        adr = sim.model.joint(f"{obj.name}_freejoint").qposadr.item()
        saved[obj.name] = (adr, sim.data.qpos[adr:adr + 3].copy())
        sim.data.qpos[adr:adr + 3] = _OFFSCREEN_POS
    mujoco.mj_forward(sim.model, sim.data)
    try:
        return sim.render_free_camera(
            pos=np.asarray(view.pos),
            lookat=np.asarray(view.lookat),
            fovy=float(view.fovy_deg),
            width=_FREE_CAM_W, height=_FREE_CAM_H,
            up=tuple(view.up),
            show_axes=True,
        )
    finally:
        for _, (adr, pos) in saved.items():
            sim.data.qpos[adr:adr + 3] = pos
        mujoco.mj_forward(sim.model, sim.data)


def _render_keyframe_images(
    sim: RobotSceneSim,
    pickup: str,
    view_a: CameraPose,
    view_b: CameraPose,
    save_dir: Path,
    frame_idx: int,
) -> dict[str, str]:
    """Render 5 cameras at the current sim state. Returns {label: path}.

    Always overlays the world-frame coordinate axes.
    """
    paths: dict[str, str] = {}
    save_dir.mkdir(parents=True, exist_ok=True)

    # Real cameras, rendered at their native size (see _real_cam_size).
    real_w, real_h = _real_cam_size(sim)
    real_imgs = sim.render_cameras(
        width=real_w, height=real_h,
        # Group 0 is OBJECT_COLLISION under this branch's convention (coacd
        # sub-meshes) — master's (0,1,2,4,5) would render the hulls.
        visible_groups=(1, 2, 4, 5),
        show_axes=True,
    )
    for i, cam_id in enumerate(sim.config.cameras.camera_ids):
        label = f"real_cam_{cam_id}"
        p = save_dir / f"{label}_frame_{frame_idx:05d}.jpg"
        imageio.imwrite(str(p), real_imgs[i])
        paths[label] = str(p)

    # Two free-camera views from ViewFinder, full scene
    for label, pose in (("view_a", view_a), ("view_b", view_b)):
        img = sim.render_free_camera(
            pos=np.asarray(pose.pos),
            lookat=np.asarray(pose.lookat),
            fovy=float(pose.fovy_deg),
            width=_FREE_CAM_W, height=_FREE_CAM_H,
            up=tuple(pose.up),
            show_axes=True,
        )
        p = save_dir / f"{label}_frame_{frame_idx:05d}.jpg"
        imageio.imwrite(str(p), img)
        paths[label] = str(p)

    # Bonus: view_a re-rendered with non-pickup objects hidden, so the
    # LLM can see the pickup object (e.g. pen) without container occlusion.
    img_pickup = _render_view_pickup_only(sim, pickup, view_a)
    p = save_dir / f"view_a_pickup_only_frame_{frame_idx:05d}.jpg"
    imageio.imwrite(str(p), img_pickup)
    paths["view_a_pickup_only"] = str(p)

    return paths


def _record_snapshot(
    sim: RobotSceneSim,
    pickup: str,
    frame_idx: int,
    n_frames: int,
    view_a: CameraPose,
    view_b: CameraPose,
    save_dir: Path,
) -> FrameSnapshot:
    """Pull pen/gripper poses from current sim state + render the keyframe images."""
    pen_adr = sim.model.joint(f"{pickup}_freejoint").qposadr.item()
    pen_pos = sim.data.qpos[pen_adr:pen_adr + 3].copy()
    pen_quat = sim.data.qpos[pen_adr + 3:pen_adr + 7].copy()

    gripper_pos = np.array(sim.data.site("ee_frame").xpos, dtype=np.float64)
    R = np.array(sim.data.site("ee_frame").xmat, dtype=np.float64).reshape(3, 3)
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, R.flatten())

    close_frac = float(np.clip(sim.q_target[frame_idx][sim.arm_dof] / sim.profile.gripper_ctrl_scale, 0.0, 1.0))
    dist = float(np.linalg.norm(gripper_pos - pen_pos))
    paths = _render_keyframe_images(sim, pickup, view_a, view_b, save_dir, frame_idx)

    return FrameSnapshot(
        frame_idx=int(frame_idx),
        time_sec=float(frame_idx * sim.frame_dt),
        frac=float(frame_idx / max(n_frames - 1, 1)),
        pen_pos=tuple(float(v) for v in pen_pos),
        pen_quat=tuple(float(v) for v in pen_quat),
        gripper_pos=tuple(float(v) for v in gripper_pos),
        gripper_quat=tuple(float(v) for v in quat),
        gripper_close_frac=close_frac,
        distance=dist,
        image_paths=paths,
    )


# USD: how often to dump a frame to the USD exporter, measured in sim frames
# (each sim frame is 1/sim.fps = 1/60 s by default, with 10 mj_steps inside).
# Setting to 1 captures every interpolated joint-angle frame faithfully (USD
# plays back at 60fps, matching the recorded trajectory rate). Bump higher
# (e.g. 5 or 10) only if disk usage is a concern.
USD_SAMPLE_EVERY = 1

# Scene option used for USD export — show only the optimized VISUAL meshes:
#   group 1 = ground                       (show)
#   group 2 = robot visual                 (show)
#   group 4/5 = object visual mesh         (show)
#   group 0 = object collision sub-meshes  (HIDE — coacd-decomposed pieces)
#   group 3 = robot collision              (HIDE — Franka panda collision proxies)
# See RobotSceneSim.GEOM_GROUP_* for the convention.
_USD_HIDDEN_GROUPS = {
    0,  # GEOM_GROUP_OBJECT_COLLISION
    3,  # GEOM_GROUP_ROBOT_COLLISION
}
_USD_SCENE_OPT = mujoco.MjvOption()
for _g in range(len(_USD_SCENE_OPT.geomgroup)):
    _USD_SCENE_OPT.geomgroup[_g] = 0 if _g in _USD_HIDDEN_GROUPS else 1


def _make_usd_exporter(sim: RobotSceneSim, save_dir: Path):
    """Construct a colored USD exporter that will accumulate animation frames.

    The exporter writes per-frame USDs into `save_dir/usd/frames/` and a
    top-level scene that references them.
    """
    out = save_dir / "usd"
    out.mkdir(parents=True, exist_ok=True)
    return VertexColorUSDExporter(
        model=sim.model,
        output_directory=out.name,
        output_directory_root=str(save_dir),
        verbose=False,
        object_visual_mesh_paths={
            f"{obj.name}_visual": obj.visual_mesh_path
            for obj in sim.config.objects
        },
    )


def _finalise_usd(usd_exporter, save_dir: Path) -> None:
    if usd_exporter is None:
        return
    try:
        usd_exporter.save_scene()
    except Exception as e:
        print(f"[grasp_probe] USD save_scene failed: {e}")


def run_grasp_probe_lite(sim: RobotSceneSim, pickup_object: str) -> dict:
    """Pure-physics version of `run_grasp_probe` — no rendering, no USD,
    no keyframe snapshots. Just runs the trajectory and returns the
    success-criterion verdict.

    Used by box-search / robustness tools where we need many cheap probes.
    Returns dict with the same keys as `_analyse_lift` plus
    `closest_distance` and `duration_sec`.
    """
    if sim.is_kinematic:
        raise RuntimeError("run_grasp_probe_lite requires is_kinematic=False")

    n_frames = int(sim.q_target.shape[0])
    engage_frame = _find_engage_frame(sim)
    sim.reset()
    pen_adr = sim.model.joint(f"{pickup_object}_freejoint").qposadr.item()
    pen_pos_trace = np.empty((n_frames, 3), dtype=np.float64)
    gripper_pen_dist_trace = np.empty(n_frames, dtype=np.float64)
    closest = float("inf")
    pen_quat_at_engage: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    gripper_quat_at_engage: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    for frame in range(n_frames):
        sim.sim_one_frame()
        pen_pos = sim.data.qpos[pen_adr:pen_adr + 3]
        pen_pos_trace[frame] = pen_pos
        ee = np.array(sim.data.site("ee_frame").xpos, dtype=np.float64)
        d = float(np.linalg.norm(ee - pen_pos))
        gripper_pen_dist_trace[frame] = d
        if d < closest:
            closest = d
        if frame == engage_frame:
            qp = sim.data.qpos[pen_adr + 3:pen_adr + 7]
            pen_quat_at_engage = (float(qp[0]), float(qp[1]), float(qp[2]), float(qp[3]))
            R = np.array(sim.data.site("ee_frame").xmat, dtype=np.float64).reshape(3, 3)
            qg = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(qg, R.flatten())
            gripper_quat_at_engage = (float(qg[0]), float(qg[1]), float(qg[2]), float(qg[3]))

    lift = _analyse_lift(
        pen_pos_trace, gripper_pen_dist_trace, sim.frame_dt,
        DISPLACEMENT_THRESHOLD_M, MIN_LIFT_DURATION_S, MAX_DIST_AT_LIFT_M,
    )
    return {
        **lift,
        "closest_distance": float(closest),
        "duration_sec": float(n_frames * sim.frame_dt),
        "engage_frame": int(engage_frame),
        "pen_quat_at_engage": pen_quat_at_engage,        # wxyz, world frame
        "gripper_quat_at_engage": gripper_quat_at_engage,  # wxyz, world frame (ee_frame site)
    }


def _find_engage_frame(sim: RobotSceneSim) -> int:
    """First frame where the gripper closes AFTER having been open — the
    trajectory's grasp moment.

    "After having been open" is load-bearing, not decoration: a demo whose
    gripper starts closed (AIRBOT ep3 sits closed for its first 590 frames)
    would otherwise anchor on frame 0. Polarity comes from the robot profile,
    since 1 means closed on a Franka but OPEN on the AIRBOT G2.

    Why not 'lowest gripper z'? Some demos have multiple low-z passes
    (grasp + place + retract); the lowest z can be the *placement*
    move, several seconds after the actual grasp. The close-command
    rising-edge is a reliable single-event marker for the grasp itself.

    Falls back to argmin of gripper TCP z if no close event is detected
    (e.g. a sliding/poke trajectory with no closing command).
    """
    n_frames = int(sim.q_target.shape[0])
    closed = _closedness_trace(sim) > _CLOSE_THRESHOLD
    # A real rising edge: the first CLOSED frame that has an OPEN frame before
    # it. `argmax(closed)` alone is not that — on a trajectory that starts
    # closed it returns frame 0. AIRBOT ep3 opens at 590, grasps at 745,
    # releases at 1074 and closes empty at 1292; only requiring a preceding
    # open picks 745. Franka demos start open, so first-closed already IS the
    # rising edge and this returns the same frame as before.
    opened = ~closed
    if opened.any() and closed.any():
        first_open = int(np.argmax(opened))
        after = closed[first_open:]
        if after.any():
            return first_open + int(np.argmax(after))
    # Fallback: lowest gripper TCP z via FK
    saved_qpos = sim.data.qpos.copy()
    saved_qvel = sim.data.qvel.copy()
    best_z, best_frame = float("inf"), 0
    try:
        for f in range(n_frames):
            for i, joint_id in enumerate(sim.joint_id_map):
                sim.data.qpos[joint_id] = sim.q_target[f][i]
            mujoco.mj_forward(sim.model, sim.data)
            ee_z = float(sim.data.site("ee_frame").xpos[2])
            if ee_z < best_z:
                best_z, best_frame = ee_z, f
        return best_frame
    finally:
        sim.data.qpos[:] = saved_qpos
        sim.data.qvel[:] = saved_qvel
        mujoco.mj_forward(sim.model, sim.data)


def run_grasp_probe(
    sim: RobotSceneSim,
    pickup_object: str,
    view_a: CameraPose,
    view_b: CameraPose,
    save_dir: Path,
    frame_offset: int = KEYFRAME_OFFSET_FRAMES,
) -> GraspProbeReport:
    """Run the full trajectory in dynamic mode, capture 3 keyframes (5 imgs each).

    `sim` MUST be `is_kinematic=False`.
    """
    if sim.is_kinematic:
        raise RuntimeError("run_grasp_probe requires is_kinematic=False")

    n_frames = int(sim.q_target.shape[0])
    engage_frame = _find_engage_frame(sim)
    keyframe_indices = sorted({
        max(0, min(n_frames - 1, engage_frame + k * frame_offset))
        for k in range(3)
    })

    sim.reset()
    pen_adr = sim.model.joint(f"{pickup_object}_freejoint").qposadr.item()
    pickup_geom_ids = _build_pickup_geom_set(sim.model, pickup_object)
    close_complete_frame = _find_close_complete_frame(sim, start=engage_frame)
    pen_pos_trace = np.empty((n_frames, 3), dtype=np.float64)
    gripper_pen_dist_trace = np.empty(n_frames, dtype=np.float64)
    closest_distance = float("inf")
    snapshots: list[FrameSnapshot] = []
    first_contact: FirstContactInfo | None = None

    # Begin USD animation capture for the whole episode. Frames sampled
    # every USD_SAMPLE_EVERY sim steps. Updated BEFORE keyframe rendering
    # so any temporary qpos mutation in pickup-only renders does not leak
    # into the recorded animation.
    usd_exporter = _make_usd_exporter(sim, save_dir)

    for frame in range(n_frames):
        sim.sim_one_frame()
        pen_pos = sim.data.qpos[pen_adr:pen_adr + 3]
        pen_pos_trace[frame] = pen_pos

        ee_pos = np.array(sim.data.site("ee_frame").xpos, dtype=np.float64)
        d = float(np.linalg.norm(ee_pos - pen_pos))
        gripper_pen_dist_trace[frame] = d
        if d < closest_distance:
            closest_distance = d

        # First gripper↔pickup contact — captured ONCE, then ignored.
        # Restricted to frames before the gripper has finished closing;
        # post-close contacts are dynamics noise, not approach geometry.
        if first_contact is None and frame <= close_complete_frame:
            hit = _detect_first_contact(sim, pickup_object, pickup_geom_ids, frame)
            if hit is not None:
                _, body_name, side, world_pos = hit
                first_contact = _build_first_contact_info(
                    sim, frame, body_name, side, world_pos,
                )

        if usd_exporter is not None and (frame % USD_SAMPLE_EVERY == 0):
            try:
                usd_exporter.update_scene(sim.data, scene_option=_USD_SCENE_OPT)
            except Exception:
                pass

        if frame in keyframe_indices:
            snapshots.append(_record_snapshot(
                sim, pickup_object, frame, n_frames, view_a, view_b, save_dir,
            ))

    _finalise_usd(usd_exporter, save_dir)

    lift = _analyse_lift(
        pen_pos_trace, gripper_pen_dist_trace, sim.frame_dt,
        DISPLACEMENT_THRESHOLD_M, MIN_LIFT_DURATION_S, MAX_DIST_AT_LIFT_M,
    )

    return GraspProbeReport(
        grasped=lift["grasped"],
        grasped_loose=lift["grasped_loose"],
        peak_lift=lift["peak_lift"],
        peak_lift_time_frac=lift["peak_lift_time_frac"],
        lift_window_sec=lift["lift_window_sec"],
        closest_distance_overall=float(closest_distance),
        dist_at_peak_lift=lift["dist_at_peak_lift"],
        snapshots=snapshots,
        duration_sec=float(n_frames * sim.frame_dt),
        first_contact=first_contact,
    )
