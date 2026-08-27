"""User-facing entry schema for the visual pipeline.

Users fill out a ``VisualInput`` (typically via a generated file under
``outputs/run_pipeline/<run_id>/`` named ``visual_<id>.py``). The
orchestrator seeds VisualState from this bundle and runs the pipeline; the
final node emits an episode folder under
``outputs/run_pipeline/<run_id>/sysid_inputs/`` which the sysid runner consumes.

Hint fields below are scaffolding for stages where the LLM/VLM agent isn't
in place yet. As those agents land, the corresponding hint field becomes
unused and can be dropped.
"""
from dataclasses import dataclass, field


@dataclass
class VisualInput:
    # ---- always required ----
    stereo_stream_path: str                                  # exported rectified stereo side-by-side .mp4
    stereo_intrinsics_path: str                              # exporter-produced SDK/SVO <serial>-stereo.K.txt
    robot_traj_path: str
    episode_id: str                                          # identifier-safe slug (e.g. raw_id_slug, or legacy droid_100_episode_<idx:03d>_<raw_id_slug>)

    # ---- always optional ----
    scene_name: str = ""                                     # defaults to episode_id
    max_frames: int | None = None                            # cap frames at svo_extract + stereo_depth (None = all)
    frame_step: int = 1                                      # keep every Nth frame (1 = all). Subsamples
                                                             # temporally while spanning the whole demo —
                                                             # svo_extract re-indexes the kept frames to a
                                                             # consecutive 0..N sequence so downstream stays
                                                             # frame-number-consecutive. ~Nx faster.
    keyframe_index: int = -1                                 # force the SAM3 prompt/keyframe to this (re-indexed)
                                                             # frame, skipping the keyframe_select VLM vote.
                                                             # -1 = auto (let keyframe_select pick).
    heavy_max_frames: int | None = None                      # cap frames for the per-frame HEAVY models ONLY:
                                                             # FoundationStereo depth (--max_frames) + FoundationPose
                                                             # tracking (--end_frame). svo_extract stays uncapped, so
                                                             # object_discovery/segment/mesh_recover run over the full
                                                             # sequence as usual. None = all. Use for quick validation.

    # ---- stereo_depth (FoundationStereo) speed/quality knobs ----
    # These trade negligible quality for a large speedup. depth feeds mesh_scale
    # (keyframe-anchored, robust to resolution) and pose_tracking.
    stereo_valid_iters: int = 22                             # GRU refinement iterations (FoundationStereo
                                                             # script default 32). Dominant cost knob; 22 ≈ lossless.
    # ---- pose_tracking (FoundationPose) ----
    pose_shorter_side: int | None = None                     # resize frames so min(H,W) == this before
                                                             # FoundationPose runs. register() warps ~240 pose
                                                             # hypotheses at full image resolution, so memory
                                                             # scales with pixel count: a 2208x1242 sensor needs
                                                             # ~3x a 1280x720 one and OOMs a 32 GB card. The
                                                             # script rescales K with the image, so geometry is
                                                             # unchanged. None = native (fine for DROID's 720p).

    stereo_get_pointcloud: bool = False                      # save per-frame .ply clouds. Debug-only (nothing
                                                             # downstream reads them; ~75% of run size) and the
                                                             # open3d denoise is slow — default off.

    # ---- hints (replace each as the matching LLM agent lands) ----
    # ObjectDiscoveryAgent: hardcoded SAM3 text prompts.
    # Must contain "robot" if you want the robot mask resolved downstream.
    object_texts: list[str] = field(default_factory=list)

    # PickupObjectAgent: which discovered object the robot is grasping.
    pickup_object: str = ""

    # GroundRefAgent: which object defines the floor reference.
    # Empty -> first non-pickup object in discovered_objects.
    ground_reference_object: str = ""
    # NOTE: ground.local_down_axis + ground.offset used to be hint fields here
    # but visual never inferred them. outputs.py emits a [0,0,-1] / 0.0 stub
    # (MuJoCo z-up world); geometry_prior.z_anchor patches `offset` and may
    # flip `reference` to the GROUND sentinel before sysid consumes the
    # manifest. Deleted from VisualInput 2026-05-30.

    # ---- user-provided (no LLM replacement planned) ----
    cameras_extrinsics_path: str = ""                        # required by CameraSpec downstream
    task_description: str = ""                               # one-sentence NL description; emitted as manifest.task.description (default: "pickup <pickup_objects>")
    robot_type: str = "franka_panda"                         # RobotProfile registry name; emitted as
                                                             # manifest.robot.type, which is what sysid uses to
                                                             # pick the arm/gripper model. DROID episodes are all
                                                             # Franka, hence the default; self-collected episodes
                                                             # carry their own (build_raw_episode fills it from
                                                             # the raw_data stage_manifest).

    # ---- dual-view (PR-1 onwards) ----
    # Primary uses ``stereo_stream_path`` + ``stereo_intrinsics_path`` above.
    # Secondary is the runner-up rectified stereo camera, kept so
    # segment_controller can lazy-fallback when primary mask attempts fail.
    # Empty secondary fields mean a single-camera episode.
    secondary_stereo_stream_path: str = ""                   # rectified stereo side-by-side .mp4
    secondary_stereo_intrinsics_path: str = ""               # matching SDK/SVO <serial>-stereo.K.txt
    # Deprecated compatibility alias for incoming PR-3 code. If set, this must
    # name the same rectified MP4 as ``secondary_stereo_stream_path``; it is not
    # an SVO file and there is intentionally no primary ``svo_path`` field.
    secondary_svo_path: str = ""
    # ZED serials matching primary/secondary rectified MP4s. Filled by
    # build_raw_episode from camera_select's vote. Consumed by data_ingest ->
    # state, then by per-camera skills to disambiguate which
    # ``cam_mat_<serial>`` key in cameras_extrinsics.npz belongs to which view.
    primary_camera_id: str = ""
    secondary_camera_id: str = ""

    # Wrist-cam rectified stereo .mp4, present iff the raw_data folder had
    # one. Frame source only (no intrinsics needed) — used by
    # object_discovery in preference to the external camera. Empty =
    # not available.
    wrist_stream_path: str = ""

    # ---- SysID hints: per-object overrides (optional) ----
    # If you have prior knowledge of an object's mass / friction, fill these so
    # SysIDAgent has a better starting point. Missing entries fall back to the
    # ResolvedObject dataclass defaults.
    object_mass_hints: dict[str, float] = field(default_factory=dict)
    object_friction_hints: dict[str, tuple[float, float, float]] = field(default_factory=dict)
