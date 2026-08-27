"""Lightweight single-object MuJoCo renderer for the pose_view_critic subagent.

Purpose
-------
Take a single object (mesh + estimated pose in camera frame) and render what
that object looks like from the primary calibrated camera, so a VLM critic
can compare the render against the real frame at the same view.

This is intentionally NOT ``RobotSceneSim`` — that class needs the full
SceneConfig (robot XML, all objects, ground, trajectory) which is not
available yet at ``pose_tracking`` time. We build a minimal MjSpec with:

    - one directional light (fixed pose, matches RobotSceneSim's key_light)
    - one mesh geom (visual only, no collision)
    - one calibrated camera (K + T_cam_in_world)

Reuses the CV↔MuJoCo camera-axis conversion helpers from
``ar2s.droid_sim._util`` so the FoV and orientation match what
``RobotSceneSim.render_object_camera()`` produces for the same K.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from ar2s.droid_sim._util import (
    fovy_deg_from_K,
    opencv_c2w_to_mujoco_camera_pose,
)


_MUJOCO_MESH_EXTS = {".obj", ".stl", ".msh"}


def _mujoco_ready_mesh(mesh_path: Path) -> Path:
    """MuJoCo can't load .ply / .glb natively. Convert to a cached .stl next
    to the source (or fall back to the system tmpdir) and return that path.

    Cache key: sha256 of the source bytes; regeneration only on source change.
    """
    ext = mesh_path.suffix.lower()
    if ext in _MUJOCO_MESH_EXTS:
        return mesh_path
    import trimesh
    digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()[:16]
    cache = mesh_path.parent / f"{mesh_path.stem}.{digest}.stl"
    if not cache.is_file():
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            m = trimesh.load(str(mesh_path), force="mesh")
            m.export(str(cache))
        except (OSError, PermissionError):
            tmp = Path(tempfile.gettempdir()) / f"{mesh_path.stem}.{digest}.stl"
            if not tmp.is_file():
                m = trimesh.load(str(mesh_path), force="mesh")
                m.export(str(tmp))
            return tmp
    return cache


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation → MuJoCo quat (w, x, y, z)."""
    q = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R.flatten(), dtype=np.float64))
    return q


def render_scene_objects(
    objects: list[dict],
    T_cam_in_world: np.ndarray,
    K: np.ndarray,
    *,
    ground_z: float | None = None,
    width: int = 1280,
    height: int = 720,
    usd_save_path: Path | None = None,
) -> np.ndarray:
    """Render MULTIPLE objects at world poses from one calibrated camera.

    Each entry of ``objects``: {"name": str, "mesh_path": Path,
    "T_object_in_world": (4,4), "mesh_scale": (3,) or float}.
    Same camera/light/ground conventions as ``render_object_at_pose`` —
    that function now delegates here with a single-element list.
    """
    T_cam_in_world = np.asarray(T_cam_in_world, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    assert T_cam_in_world.shape == (4, 4), T_cam_in_world.shape
    assert K.shape == (3, 3), K.shape

    cam_pos, cam_quat = opencv_c2w_to_mujoco_camera_pose(T_cam_in_world)
    fovy_deg = fovy_deg_from_K(K, image_height=height)

    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = max(2048, width)
    spec.visual.global_.offheight = max(2048, height)
    world = spec.worldbody

    light = world.add_light()
    light.name = "key_light"
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.dir[:] = [0.2, 0.1, -1.0]
    light.pos[:] = [0.0, -0.4, 1.5]
    light.ambient[:] = [0.35, 0.35, 0.35]
    light.diffuse[:] = [0.85, 0.85, 0.85]
    light.specular[:] = [0.2, 0.2, 0.2]
    light.castshadow = 0

    for i, obj in enumerate(objects):
        T_w = np.asarray(obj["T_object_in_world"], dtype=np.float64)
        assert T_w.shape == (4, 4), (obj.get("name"), T_w.shape)
        mesh_asset = spec.add_mesh()
        mesh_asset.name = f"mesh_{i}"
        mesh_asset.file = str(_mujoco_ready_mesh(Path(obj["mesh_path"])).resolve())
        s = np.atleast_1d(np.asarray(obj.get("mesh_scale", 1.0), dtype=np.float64))
        mesh_asset.scale[:] = float(s[0]) if s.size == 1 else s[:3]

        body = world.add_body()
        body.name = obj.get("name") or f"obj_{i}"
        body.pos[:] = T_w[:3, 3]
        body.quat[:] = _mat_to_quat_wxyz(T_w[:3, :3])
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = f"mesh_{i}"
        geom.contype = 0
        geom.conaffinity = 0
        # Optional flat-color override (e.g. saturated magenta) — kills
        # texture cues but makes silhouette/position/size pop against the
        # gray ground for VLM judging.
        if obj.get("rgba") is not None:
            geom.rgba[:] = np.asarray(obj["rgba"], dtype=np.float64)

    if ground_z is not None:
        gbody = world.add_body()
        gbody.name = "ground"
        gbody.pos[:] = [0.0, 0.0, float(ground_z)]
        ggeom = gbody.add_geom()
        ggeom.type = mujoco.mjtGeom.mjGEOM_PLANE
        ggeom.size[:] = [2.5, 2.5, 0.01]
        ggeom.rgba[:] = [0.55, 0.55, 0.55, 0.7]
        ggeom.contype = 0
        ggeom.conaffinity = 0

    cam = world.add_camera()
    cam.name = "primary"
    cam.pos[:] = cam_pos
    cam.quat[:] = cam_quat
    cam.fovy = fovy_deg

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if usd_save_path is not None:
        import shutil
        from mujoco.usd.exporter import USDExporter
        save_dir = Path(usd_save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        opt = mujoco.MjvOption()
        for g in range(len(opt.geomgroup)):
            opt.geomgroup[g] = 1
        exp = USDExporter(model=model, output_directory="usd_scratch",
                          output_directory_root=str(save_dir), verbose=False)
        exp.update_scene(data, scene_option=opt)
        exp.save_scene()
        scratch = save_dir / "usd_scratch"
        exported = scratch / "frames" / f"frame_{exp.frame_count}.usd"
        if exported.is_file():
            shutil.move(str(exported), str(usd_save_path))
        shutil.rmtree(scratch, ignore_errors=True)

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        cam_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "primary")
        renderer.update_scene(data, camera=cam_idx)
        rgb = renderer.render().copy()
    finally:
        renderer.close()
    return rgb


def render_object_at_pose(
    mesh_path: Path,
    T_object_in_cam: np.ndarray,
    T_cam_in_world: np.ndarray,
    K: np.ndarray,
    *,
    mesh_scale: np.ndarray | float = 1.0,
    obj_pos_offset: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ground_z: float | None = None,
    width: int = 1280,
    height: int = 720,
    usd_save_path: Path | None = None,
) -> np.ndarray:
    """Render a single mesh at the given pose, seen from the calibrated camera.

    Args:
        mesh_path: absolute path to a mesh MuJoCo can load (.ply / .obj / .stl).
        T_object_in_cam: 4x4 object → camera (OpenCV convention).
        T_cam_in_world: 4x4 camera → world (OpenCV convention);
            i.e., ``T_cam_in_world = inv(cam_mat_<serial>)`` where
            ``cam_mat_<serial>`` in ``cameras_extrinsics.npz`` is w2c per
            repo convention (see docs / memory ``cam_mat_convention``).
        K: 3x3 pinhole intrinsics matching ``height``.
        width, height: output RGB size.

    Returns:
        HxWx3 uint8 RGB array.
    """
    T_cam_in_world = np.asarray(T_cam_in_world, dtype=np.float64)
    T_object_in_cam = np.asarray(T_object_in_cam, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    assert T_cam_in_world.shape == (4, 4), T_cam_in_world.shape
    assert T_object_in_cam.shape == (4, 4), T_object_in_cam.shape
    assert K.shape == (3, 3), K.shape

    T_object_in_world = T_cam_in_world @ T_object_in_cam
    # z_anchor writes its per-object shift into manifest.yaml (see
    # ``load_pose_and_camera``) rather than baking it into the FP .npy —
    # the sim runtime composes them at scene build; the renderer has to do
    # the same to reflect the final pipeline output.
    T_w = T_object_in_world.copy()
    T_w[:3, 3] = T_w[:3, 3] + np.asarray(obj_pos_offset, dtype=np.float64)
    return render_scene_objects(
        [{"name": "target", "mesh_path": mesh_path,
          "T_object_in_world": T_w, "mesh_scale": mesh_scale}],
        T_cam_in_world, K,
        ground_z=ground_z, width=width, height=height,
        usd_save_path=usd_save_path,
    )


def load_pose_and_camera(
    run_root: Path,
    object_name: str,
    frame_idx: int = 0,
    *,
    stage: str = "pose_tracking",
) -> dict:
    """Gather everything ``render_object_at_pose`` needs for one object.

    Args:
        stage: which pipeline stage the pose+mesh should reflect.
          - ``"pose_tracking"``: raw FoundationPose output (pre-geometry_prior).
              Reads ``<run_root>/poses/<obj>/scaled_mesh.ply`` +
              ``<run_root>/poses/<obj>/ob_in_cam/frame_XXXXXX.txt``.
          - ``"post_geometry"``: post-geometry_prior state after mesh_orient
              flips + mesh_axis_align rewrites the per-frame FP.
              Reads the flat ``sysid_inputs/objects/<obj>/visual.obj`` bundle
              through the packed/legacy ``pose_store`` interface.
              Pose is still ``T_object_in_cam`` (mesh_axis_align applies R in
              world, converts back to cam). ``pos_offset_z`` from z_anchor
              is NOT applied here — for a strictly-accurate render the caller
              can shift the world position after calling render.

    Returns dict with keys: mesh_path, T_object_in_cam, T_cam_in_world, K,
    primary_serial, real_frame_path.
    """
    import json

    assert stage in ("pose_tracking", "post_geometry"), stage

    if stage == "pose_tracking":
        poses_dir = run_root / "poses" / object_name
        mesh_path = poses_dir / "scaled_mesh.ply"
        assert mesh_path.is_file(), f"missing mesh: {mesh_path}"
        ob_in_cam_path = poses_dir / "ob_in_cam" / f"frame_{frame_idx:06d}.txt"
        assert ob_in_cam_path.is_file(), f"missing pose: {ob_in_cam_path}"
        T_object_in_cam = np.loadtxt(ob_in_cam_path).reshape(4, 4)
        mesh_scale = np.ones(3, dtype=np.float64)  # scaled_mesh.ply is already scaled
        offset = 0.0  # no manifest yet at pose_tracking time
        obj_pos_offset = np.zeros(3, dtype=np.float64)
    else:  # post_geometry
        from ar2s.pipeline_artifacts import episode_bundle_dir
        from ar2s.droid_sim.pose_store import load_pose
        agent_dir = episode_bundle_dir(run_root)
        obj_dir = agent_dir / "objects" / object_name
        mesh_path = obj_dir / "visual.obj"
        assert mesh_path.is_file(), f"missing mesh: {mesh_path}"
        pose_root = obj_dir / "foundation_pose"
        T_object_in_cam = load_pose(pose_root, frame_idx)
        assert T_object_in_cam is not None, f"missing pose: {pose_root} frame={frame_idx}"
        assert T_object_in_cam.shape == (4, 4), T_object_in_cam.shape
        # visual.obj is UN-scaled; scale.npy is applied by sysid at load time.
        scale_path = obj_dir / "scale.npy"
        if scale_path.is_file():
            mesh_scale = np.load(scale_path).astype(np.float64).reshape(-1)
        else:
            mesh_scale = np.ones(3, dtype=np.float64)
        # Ground plane world Z from manifest.yaml (see mesh_axis_align:
        # ground_z = -ground.offset per the sim convention). Also apply the
        # per-object z_anchor pos_offset — z_anchor writes this into
        # manifest.yaml but leaves the FP .npy poses untouched, so any renderer
        # of a "post-geometry" pose has to compose them itself.
        manifest_path = agent_dir / "manifest.yaml"
        obj_pos_offset = np.zeros(3, dtype=np.float64)
        if manifest_path.is_file():
            import yaml
            manifest = yaml.safe_load(manifest_path.read_text()) or {}
            offset = float((manifest.get("ground") or {}).get("offset") or 0.0)
            for o in manifest.get("objects", []) or []:
                if o.get("name") == object_name:
                    po = o.get("pos_offset")
                    if po:
                        obj_pos_offset = np.asarray(po, dtype=np.float64)
                    break
        else:
            offset = 0.0

    state = json.loads((run_root / "state.json").read_text())
    primary_serial = str(state.get("primary_camera_id") or "")
    secondary_serial = str(state.get("secondary_camera_id") or "")
    assert primary_serial, "state.json.primary_camera_id missing"

    raw_dir = run_root / "raw_episodes"
    if not raw_dir.is_dir():
        # last resort: search staged path from state.entry
        entry_path = state.get("entry", {}).get("cameras_extrinsics_path", "")
        assert entry_path, f"cannot resolve raw_episodes for {run_root}"
        raw_dir = Path(entry_path).parent
    extr = np.load(raw_dir / "cameras_extrinsics.npz")

    def _cam_c2w(serial: str) -> np.ndarray:
        key = f"cam_mat_{serial}"
        assert key in extr.files, f"{raw_dir}/cameras_extrinsics.npz missing {key}"
        return np.linalg.inv(np.asarray(extr[key], dtype=np.float64))

    # Camera files via the shared resolver (handles role-keyed v2 + legacy
    # stereo.mp4 layouts). Cross-check serials against state.json — if the
    # staged bundle disagrees with what the pipeline recorded, fail loudly
    # rather than render the wrong view (GuptaLab manual-retry lesson).
    from ar2s.agents_visual.raw_episode import resolve_camera_files
    cams = resolve_camera_files(raw_dir)
    if cams["primary"]["serial"] and cams["primary"]["serial"] != primary_serial:
        raise RuntimeError(
            f"{run_root.name}: staged primary serial "
            f"{cams['primary']['serial']} != state.json primary_camera_id "
            f"{primary_serial} — raw_episodes and pipeline state disagree"
        )

    T_cam_in_world = _cam_c2w(primary_serial)
    K = np.loadtxt(cams["primary"]["K"], max_rows=1).reshape(3, 3)

    # Object pose in world frame (needed to re-express in secondary camera frame).
    T_object_in_world = T_cam_in_world @ T_object_in_cam

    # Secondary view (optional — may be absent for early debug eps).
    T_cam_in_world_secondary: np.ndarray | None = None
    K_secondary: np.ndarray | None = None
    T_object_in_cam_secondary: np.ndarray | None = None
    sec_files = cams.get("secondary")
    if secondary_serial and sec_files is not None and sec_files["K"].is_file():
        try:
            T_cam_in_world_secondary = _cam_c2w(secondary_serial)
            K_secondary = np.loadtxt(sec_files["K"], max_rows=1).reshape(3, 3)
            T_object_in_cam_secondary = np.linalg.inv(T_cam_in_world_secondary) @ T_object_in_world
        except AssertionError:
            T_cam_in_world_secondary = None

    # Real image at PRIMARY / SECONDARY view (NOT od_frames — those come from wrist cam;
    # NOT sysid_inputs/real/first_frame.png either — that's a copy of pose_tracking's
    # track_vis overlay, with FP bbox + mesh contour drawn on top).
    #
    # Primary sources:
    #   1. seq/<serial>/frames/left/frame_XXXXXX.png  (clean rectified, during pipeline)
    #   2. sysid_inputs/<ep>_agent/real/video.mp4     (clean rectified single view, post-finalize)
    #
    # Secondary sources:
    #   1. seq/<secondary>/frames/left/frame_XXXXXX.png (if svo_extract was run on secondary)
    #   2. raw_episodes/stereo_secondary.mp4            (side-by-side stereo — take LEFT half)
    def _extract_frame_from_mp4(video_mp4: Path, cache: Path, *, crop_left_half: bool) -> Path:
        if cache.is_file():
            return cache
        cache.parent.mkdir(parents=True, exist_ok=True)
        import subprocess
        vf = f"select=eq(n\\,{frame_idx})"
        if crop_left_half:
            vf += ",crop=iw/2:ih:0:0"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(video_mp4),
             "-vf", vf, "-vframes", "1",
             str(cache)],
            check=True,
        )
        return cache

    # NOTE: We deliberately do NOT fall back to sysid_inputs/<ep>_agent/real/video.mp4
    # here — that file has been observed to hold the SECONDARY camera view for
    # some eps (seen on GuptaLab_success_2023_04_20_13_13_17 where the pipeline
    # had a manual --camera=ext1 override retry, leaving the sysid-side real
    # video pointing at ext2). Reading via resolve_camera_files() stays honest
    # to the recorded primary/secondary serial contract.
    real_frame_path: Path | None = None
    seq_png = run_root / "seq" / primary_serial / "frames" / "left" / f"frame_{frame_idx:06d}.png"
    if seq_png.is_file():
        real_frame_path = seq_png
    elif cams["primary"]["mp4"].is_file():
        pri_mp4 = cams["primary"]["mp4"]
        cache = raw_dir / f"_extracted_primary_frame_{frame_idx:06d}.png"
        real_frame_path = _extract_frame_from_mp4(
            pri_mp4, cache, crop_left_half=True,
        )

    real_frame_secondary_path: Path | None = None
    if secondary_serial:
        seq_png_s = run_root / "seq" / secondary_serial / "frames" / "left" / f"frame_{frame_idx:06d}.png"
        if seq_png_s.is_file():
            real_frame_secondary_path = seq_png_s
        elif sec_files is not None and sec_files["mp4"].is_file():
            cache = raw_dir / f"_extracted_secondary_frame_{frame_idx:06d}.png"
            real_frame_secondary_path = _extract_frame_from_mp4(
                sec_files["mp4"], cache, crop_left_half=True,
            )

    # Per mesh_axis_align convention, MuJoCo world +Z is up and ground plane
    # sits at world Z = -offset (offset is a signed distance recorded in
    # manifest.yaml).
    ground_z = -offset if offset else None

    return {
        "mesh_path": mesh_path,
        "T_object_in_cam": T_object_in_cam,
        "T_cam_in_world": T_cam_in_world,
        "K": K,
        "mesh_scale": mesh_scale,
        "obj_pos_offset": obj_pos_offset,
        "ground_z": ground_z,
        "primary_serial": primary_serial,
        "real_frame_path": real_frame_path,
        "secondary_serial": secondary_serial,
        "T_object_in_cam_secondary": T_object_in_cam_secondary,
        "T_cam_in_world_secondary": T_cam_in_world_secondary,
        "K_secondary": K_secondary,
        "real_frame_secondary_path": real_frame_secondary_path,
    }
