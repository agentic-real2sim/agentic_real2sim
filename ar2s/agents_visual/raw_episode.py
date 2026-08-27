"""Convert a DROID-style raw_data folder into a per-run raw episode bundle.

Inputs (the layout users hand us):

    raw_data/
    ├── <ext1_serial>-stereo.mp4
    ├── <ext1_serial>-stereo.K.txt          (required SDK/SVO intrinsics)
    ├── <ext2_serial>-stereo.mp4
    ├── <ext2_serial>-stereo.K.txt          (required SDK/SVO intrinsics)
    ├── trajectory.h5                      (DROID-formatted, action/joint_position + action/gripper_position)
    ├── metadata_<UUID>.json               (task_description + cam role serials)
    └── <UUID>_cameras.json                (per-camera refined_extrinsics)

Outputs (what the rest of ar2s.agents_visual consumes):

    outputs/run_pipeline/<run_id>/
    └── raw_episodes/
        ├── ext1-stereo.mp4                (physical ext1 camera — ALWAYS ext1
        │                                   content regardless of which camera
        │                                   is chosen primary)
        ├── ext1-stereo.K.txt              (ext1 SDK/SVO intrinsics)
        ├── ext2-stereo.mp4                (physical ext2, iff usable)
        ├── ext2-stereo.K.txt
        ├── wrist.mp4                      (iff a wrist camera was staged)
        ├── robot_traj.h5                  (flat: joint_position (N, arm_dof)
        │                                   + gripper_position (N,1); raw_data's
        │                                   DROID zero-padding to 7 columns is
        │                                   removed here per the episode's robot)
        ├── cameras_extrinsics.npz         (cam_mat_<serial> per staged external:
        │                                   4x4 w2c from refined_extrinsics)
        ├── camera_selection.json          (which role/serial is primary vs
        │                                   secondary + the VLM vote record —
        │                                   "primary" is an ATTRIBUTE here, never
        │                                   a file identity)
        └── source/                        (verbatim backups for traceability)
            ├── metadata.json
            ├── cameras.json
            └── trajectory_raw.h5

Role-keyed naming (2026-07 refactor): files are named by their PHYSICAL camera
role (ext1/ext2 from DROID metadata), so re-staging with a different
--camera flag only rewrites camera_selection.json + the VisualInput stub —
video file contents never change identity. This kills the class of bug where
a manual primary-camera retry left stale same-named files (stereo.mp4) with
old-camera content on disk/remote copies, silently feeding the wrong view to
FoundationPose (seen on GuptaLab_success_2023_04_20_13_13_17).

Legacy layout (stereo.mp4 = chosen primary, stereo_secondary.mp4 = runner-up)
is still readable everywhere via ``resolve_camera_files()`` below.

And as a reproducibility side-effect:

    outputs/run_pipeline/<run_id>/visual_<episode_id>.py
                                            (a VisualInput stub auto-filled
                                             with task_description from metadata.json
                                             and paths pointing at the run-local
                                             raw_episodes/)

Camera choice (which external mp4 to copy):
  - 0 candidates -> raise (no external mp4 + K sidecar next to a usable extrinsics entry)
  - 1 candidate  -> picked directly
  - >=2          -> camera_select_vlm (VLM vote)

Refined extrinsics: cameras.json's top-level dict has one key per ZED serial,
each with a "refined_extrinsics" 4x4 ``T_cam_from_world`` (= world→cam = w2c)
matrix. That matrix is written directly into cameras_extrinsics.npz as
``cam_mat_<serial>`` — droid_sim's RobotSceneSim (and friends) load it as
``xforms_bot2cam`` and invert to get ``cam2world = inv(cam_mat)`` for the
``pose_world = cam2world @ T_obj_in_cam`` lift (see
``droid_sim/scene/robot_scene_sim.py:111-127, 229``).

NOTE: earlier revisions of this docstring claimed cam_mat used the opposite
orientation; runtime consumers use it as w2c, and this builder preserves that
convention.

The DROID metadata.json's 6-DoF ext{1,2}_cam_extrinsics fields are KNOWN-BAD
(metadata 6-DoF is unreliable for sim coord) — we never read them as
extrinsics. We only use metadata for: task_description, wrist_cam_serial
(to exclude), ext1/ext2_cam_serial (to identify roles).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ar2s.pipeline_artifacts import pipeline_run_dir, run_id_for
from ar2s.agents_visual.droid_episode_id import (
    load_stage_manifest,
    resolve_droid_100_episode_id,
)
from ar2s.agents_visual.inputs import VisualInput
from ar2s.agents_visual.subagents.camera_select import (
    CameraCandidate,
    CameraSelectResult,
    camera_select_vlm,
)
from ar2s.droid_sim.scene.robot_profile import get_robot_profile

# Every DROID episode is a Franka; only self-collected raw_data declares
# something else (via stage_manifest.json's robot_type).
DEFAULT_ROBOT_TYPE = "franka_panda"


def _resolve_robot_type(raw_data_dir: Path, explicit: str | None) -> str:
    """Pick the RobotProfile name for this raw_data folder.

    Precedence: caller override > stage_manifest.json > franka_panda. An
    explicit value that contradicts the manifest is an error rather than a
    silent win — the manifest was written by whoever converted the recording
    and is the better-informed source.
    """
    manifest = load_stage_manifest(raw_data_dir) or {}
    from_manifest = manifest.get("robot_type")
    if explicit is not None and from_manifest and explicit != from_manifest:
        raise ValueError(
            f"robot_type {explicit!r} conflicts with "
            f"{raw_data_dir}/stage_manifest.json robot_type {from_manifest!r}"
        )
    return explicit or from_manifest or DEFAULT_ROBOT_TYPE


_REPO_ROOT = Path(__file__).resolve().parents[2]

# This builder is a consumer of already-exported artifacts: it copies and
# restructures files that the export step produced, but it never fetches,
# converts, or synthesizes them on the user's behalf. Every "missing input"
# error below points back here instead of falling back to generating the
# artifact itself.
_EXPORT_HINT = (
    "agents_visual only consumes already-exported artifacts and never "
    "generates or fetches them itself — run the export step "
    "(`python -m scripts.export_episodes_from_raw`; see "
    "docs/droid_data_utils.md#raw-episode-export-artifacts) first"
)


# Path prefixes whose contents do not outlive the job that created them. A
# raw_data folder under any of these must be COPIED into the run dir; anything
# else is durable enough to symlink. SLURM_TMPDIR / TMPDIR are read from the
# environment because their value is per-job.
_EPHEMERAL_ENV_VARS = ("SLURM_TMPDIR", "TMPDIR", "TEMP", "TMP")
_EPHEMERAL_PREFIXES = (
    "/tmp", "/var/tmp", "/dev/shm",
    "/localscratch", "/scratch/local", "/state/partition1",
)


def _ephemeral_roots() -> list[Path]:
    """Job-scoped scratch roots to test paths against.

    Deliberately NOT filtered by existence: the check has to give the same
    answer on a login node as on a compute node, and ``/localscratch`` exists
    only on the latter. Filtering by ``exists()`` would silently classify a
    compute node's scratch as persistent wherever the dir is absent.
    """
    roots = [Path(p) for p in _EPHEMERAL_PREFIXES]
    for var in _EPHEMERAL_ENV_VARS:
        value = os.environ.get(var)
        if value:
            roots.append(Path(value).expanduser())
    return roots


def source_is_persistent(raw_data_dir: Path | str) -> tuple[bool, str]:
    """Whether ``raw_data_dir`` will still exist after this job ends.

    Returns ``(persistent, reason)``. Ephemeral means the path sits under a
    job-scoped scratch root (``$SLURM_TMPDIR``, ``$TMPDIR``, ``/localscratch``,
    ``/tmp``, ...), in which case build_raw_episode must copy — a symlink there
    dangles the moment the allocation is torn down.

    Both the literal and the fully-resolved path are tested, so neither a
    symlink pointing into scratch nor one pointing out of it can hide.
    Deliberately conservative: either form matching means copy, so the failure
    mode is a redundant copy rather than a broken run dir.
    """
    literal = Path(raw_data_dir).expanduser()
    try:
        resolved = literal.resolve()
    except OSError as e:
        return False, f"cannot resolve path ({e})"
    for root in _ephemeral_roots():
        for candidate in (literal, resolved):
            if candidate == root or root in candidate.parents:
                return False, f"under job-scoped scratch {root}"
    return True, "not under any known job-scoped scratch root"


def place_raw_source(src: Path, dst: Path, *, link_source: bool) -> None:
    """Stage one verbatim input as a durable symlink or a real copy."""
    if link_source:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _normalize_serial(value) -> str:
    serial = str(value)
    if serial.isdigit() and len(serial) < 8:
        serial = serial.zfill(8)
    return serial


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass
class RawDataLayout:
    """Resolved file paths inside a raw_data folder."""
    raw_data_dir: Path
    metadata_path: Path
    cameras_json_path: Path
    trajectory_h5_path: Path
    metadata: dict
    cameras_json: dict
    wrist_serial: str
    wrist_mp4_path: Path | None              # present iff a wrist <serial>-stereo.mp4 was staged
    ext_serials: list[str]                  # ordered: ext1, ext2 (or whatever metadata declared)
    available_externals: list[CameraCandidate]   # externals whose .mp4 + .K.txt + extrinsics both exist


def _discover_raw_data(raw_data_dir: Path) -> RawDataLayout:
    """Locate the 4 required files and resolve camera roles. Raises with a
    descriptive message if anything is missing."""
    raw_data_dir = raw_data_dir.resolve()
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"raw_data dir does not exist: {raw_data_dir}")

    metadata_paths = sorted(raw_data_dir.glob("metadata_*.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"no metadata_*.json under {raw_data_dir} — DROID metadata is required "
            f"to identify wrist/ext serials and task description"
        )
    if len(metadata_paths) > 1:
        raise RuntimeError(
            f"ambiguous metadata: {len(metadata_paths)} metadata_*.json files in "
            f"{raw_data_dir}; expected exactly one"
        )
    metadata_path = metadata_paths[0]
    metadata = json.loads(metadata_path.read_text())

    cameras_json_paths = sorted(raw_data_dir.glob("*_cameras.json"))
    if not cameras_json_paths:
        raise FileNotFoundError(
            f"no *_cameras.json (PointWorld camera calibration) under "
            f"{raw_data_dir} — {_EXPORT_HINT}"
        )
    if len(cameras_json_paths) > 1:
        raise RuntimeError(
            f"ambiguous cameras json: {len(cameras_json_paths)} files in "
            f"{raw_data_dir}"
        )
    cameras_json_path = cameras_json_paths[0]
    cameras_json = json.loads(cameras_json_path.read_text())

    trajectory_path = raw_data_dir / "trajectory.h5"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"trajectory.h5 missing in {raw_data_dir}")

    wrist_serial = _normalize_serial(metadata.get("wrist_cam_serial", "") or "")
    wrist_mp4_path: Path | None = (
        raw_data_dir / f"{wrist_serial}-stereo.mp4" if wrist_serial else None
    )
    if wrist_mp4_path is not None and not wrist_mp4_path.exists():
        print(f"[raw_episode] wrist mp4 not present at {wrist_mp4_path.name}, skipping")
        wrist_mp4_path = None
    ext_serials: list[str] = []
    for key in ("ext1_cam_serial", "ext2_cam_serial"):
        v = metadata.get(key)
        if v:
            ext_serials.append(_normalize_serial(v))

    if not ext_serials:
        raise RuntimeError(
            f"metadata.json has no ext1_cam_serial / ext2_cam_serial keys; "
            f"cannot identify external cameras"
        )

    # Match each declared ext serial to files actually present in raw_data/.
    # Skip externals without mp4, K sidecar, or usable extrinsics.
    role_by_serial = {s: f"ext{i+1}" for i, s in enumerate(ext_serials)}
    available: list[CameraCandidate] = []
    for serial in ext_serials:
        mp4 = raw_data_dir / f"{serial}-stereo.mp4"
        sdk_intrinsics_path = mp4.with_suffix(".K.txt")
        cam_entry = cameras_json.get(serial)
        if not mp4.exists():
            print(f"[raw_episode] {serial} ({role_by_serial[serial]}): "
                  f"mp4 not present at {mp4.name}, skipping")
            continue
        if not sdk_intrinsics_path.exists():
            print(
                f"[raw_episode] {serial} ({role_by_serial[serial]}): "
                f"SDK/SVO intrinsics sidecar not present at "
                f"{sdk_intrinsics_path.name}, skipping"
            )
            continue
        if cam_entry is None:
            raise RuntimeError(
                f"cameras.json has no entry for declared ext serial {serial!r}; "
                f"present serials: "
                f"{[k for k in cameras_json.keys() if isinstance(k, str) and k.isdigit()]}"
            )
        if not any(k in cam_entry for k in _EXTRINSICS_KEY_PREF):
            raise RuntimeError(
                f"cameras.json entry for {serial!r} has none of "
                f"{_EXTRINSICS_KEY_PREF} (got {list(cam_entry.keys())}); a "
                f"per-cam optimized matrix is required — metadata's 6-DoF "
                f"extrinsics are not reliable for sim coord frames. Re-run the "
                f"export step with --export_extrinsics ({_EXPORT_HINT}); this "
                f"builder will not synthesize the matrix itself"
            )
        available.append(CameraCandidate(
            serial=serial,
            mp4_path=mp4,
            role=role_by_serial[serial],
            sdk_intrinsics_path=sdk_intrinsics_path,
        ))

    return RawDataLayout(
        raw_data_dir=raw_data_dir,
        metadata_path=metadata_path,
        cameras_json_path=cameras_json_path,
        trajectory_h5_path=trajectory_path,
        metadata=metadata,
        cameras_json=cameras_json,
        wrist_serial=wrist_serial,
        wrist_mp4_path=wrist_mp4_path,
        ext_serials=ext_serials,
        available_externals=available,
    )


# ---------------------------------------------------------------------------
# Reading staged episodes (both layouts)
# ---------------------------------------------------------------------------

def resolve_camera_files(raw_dir: Path) -> dict:
    """Resolve primary/secondary camera files for a ``raw_episodes/<ep>/`` dir.

    Handles both layouts:
      - role-keyed v2 (2026-07+): ``ext1-stereo.mp4`` / ``ext2-stereo.mp4``;
        which one is primary comes from ``camera_selection.json``.
      - legacy: ``stereo.mp4`` (= whichever camera was chosen at staging
        time) / ``stereo_secondary.mp4``.

    Returns::

        {
          "primary":   {"serial": str, "role": str, "mp4": Path, "K": Path},
          "secondary": {...} | None,
          "layout":    "role_keyed_v2" | "legacy",
        }

    Raises FileNotFoundError when the primary video cannot be located.
    """
    raw_dir = Path(raw_dir)
    sel_path = raw_dir / "camera_selection.json"
    sel: dict = {}
    if sel_path.is_file():
        sel = json.loads(sel_path.read_text())

    def _entry(serial: str, role: str, mp4_name: str, k_name: str) -> dict | None:
        mp4 = raw_dir / mp4_name
        if not mp4.is_file():
            return None
        return {
            "serial": _normalize_serial(serial) if serial else "",
            "role": role,
            "mp4": mp4,
            "K": raw_dir / k_name,
        }

    # v2: explicit file names recorded at staging time (or derivable from role)
    chosen_role = sel.get("chosen_role") or ""
    primary = None
    secondary = None
    if sel.get("layout") == "role_keyed_v2" or (
        chosen_role and (raw_dir / f"{chosen_role}-stereo.mp4").is_file()
    ):
        primary = _entry(
            sel.get("chosen_serial", ""), chosen_role,
            sel.get("primary_mp4") or f"{chosen_role}-stereo.mp4",
            sel.get("primary_intrinsics") or f"{chosen_role}-stereo.K.txt",
        )
        sec_role = sel.get("secondary_role") or ""
        if sec_role:
            secondary = _entry(
                sel.get("secondary_serial", ""), sec_role,
                sel.get("secondary_mp4") or f"{sec_role}-stereo.mp4",
                sel.get("secondary_intrinsics") or f"{sec_role}-stereo.K.txt",
            )
        if primary is not None:
            return {"primary": primary, "secondary": secondary,
                    "layout": "role_keyed_v2"}

    # legacy fallback
    primary = _entry(
        sel.get("chosen_serial", ""), chosen_role, "stereo.mp4", "stereo.K.txt",
    )
    if primary is None:
        raise FileNotFoundError(
            f"cannot locate primary camera video under {raw_dir} "
            f"(tried role-keyed <role>-stereo.mp4 and legacy stereo.mp4)"
        )
    secondary = _entry(
        sel.get("secondary_serial", ""), sel.get("secondary_role") or "",
        "stereo_secondary.mp4", "stereo_secondary.K.txt",
    )
    return {"primary": primary, "secondary": secondary, "layout": "legacy"}


# ---------------------------------------------------------------------------
# Building outputs
# ---------------------------------------------------------------------------

# DROID's cameras.json has gone through several optimization passes over the
# dataset's history. We prefer them in this order:
#   - "refined_extrinsics"  : the older robot-point-optimized pass (used
#                              throughout pen_to_mug / snack era).
#   - "vggt_extrinsics"     : the newer VGGT-based geometric optimization
#                              (droid_300 dataset uses this).
# Both are 4x4 w2c (T_cam_from_world).
# We never read "measured_extrinsics" — those come from the ZED SDK and have
# the same 6-DoF reliability problem as metadata.json's ext{1,2}_cam_extrinsics.
_EXTRINSICS_KEY_PREF = ("refined_extrinsics", "vggt_extrinsics")


def _refined_mat(cameras_json: dict, serial: str) -> np.ndarray:
    """Pull per-cam extrinsics for ``serial`` as a (4,4) w2c float64 ndarray.

    Tries ``refined_extrinsics`` first, then ``vggt_extrinsics`` — see
    ``_EXTRINSICS_KEY_PREF`` for the rationale. Raises on missing keys or
    unexpected shape — both are config errors the caller should surface.
    """
    cam_entry = cameras_json[serial]
    for key in _EXTRINSICS_KEY_PREF:
        if key in cam_entry:
            mat = np.asarray(cam_entry[key], dtype=np.float64)
            if mat.shape != (4, 4):
                raise RuntimeError(
                    f"{key} for {serial} has shape {mat.shape}; expected (4, 4)"
                )
            return mat
    raise RuntimeError(
        f"cameras.json entry for {serial!r} has none of {_EXTRINSICS_KEY_PREF} "
        f"(got {list(cam_entry.keys())})"
    )


def _write_extrinsics_npz(
    out_path: Path,
    cameras_json: dict,
    serials: list[str],
) -> dict[str, np.ndarray]:
    """Write cameras_extrinsics.npz with one cam_mat_<serial> per serial.

    Dual-view (PR-1+): pass [primary, secondary] to bundle both views'
    refined extrinsics into the same npz. Single-camera episodes pass
    [primary] only — schema unchanged from before PR-1.

    Returns the dict of {serial: mat} actually written, for logging.
    """
    if not serials:
        raise ValueError("_write_extrinsics_npz called with empty serials")
    payload: dict[str, np.ndarray] = {}
    for s in serials:
        payload[f"cam_mat_{s}"] = _refined_mat(cameras_json, s)
    np.savez(out_path, **payload)
    return {s: payload[f"cam_mat_{s}"] for s in serials}


def _write_flat_trajectory_h5(
    src_path: Path,
    dst_path: Path,
    *,
    arm_dof: int,
) -> int:
    """Read DROID trajectory.h5 ``action/`` block, write the flat format
    visual / sysid expects: joint_position (N, arm_dof) + gripper_position
    (N,1) in float32. Returns the trajectory length N.

    ``raw_data/trajectory.h5`` is the DROID-compatible interchange format, so
    it always carries DROID's 7 joint columns — a 6-DOF arm is zero-padded into
    them (see ``scripts.collected_episode_common``). This is the boundary where
    that padding comes back off: everything downstream is ours, and
    ``RobotSceneSim`` asserts the width matches the profile's ``arm_dof``
    exactly. Dropping the columns here rather than slicing at load time keeps
    that assert meaningful — a genuinely 7-DOF trajectory paired with a 6-DOF
    profile still fails loudly instead of being silently truncated.
    """
    with h5py.File(src_path, "r") as src:
        if "action/joint_position" not in src or "action/gripper_position" not in src:
            raise RuntimeError(
                f"{src_path} missing 'action/joint_position' or 'action/gripper_position'; "
                f"this is not a DROID-format trajectory.h5"
            )
        jp = src["action/joint_position"][:]               # (N, 7) f64
        gp = src["action/gripper_position"][:]             # (N,) f64

    jp = jp.astype(np.float32, copy=False)
    gp = gp.astype(np.float32, copy=False).reshape(-1, 1)
    if jp.shape[0] != gp.shape[0]:
        raise RuntimeError(
            f"joint vs gripper traj length mismatch: {jp.shape[0]} vs {gp.shape[0]}"
        )
    if jp.shape[1] < arm_dof:
        raise RuntimeError(
            f"{src_path} joint_position has {jp.shape[1]} columns but the robot "
            f"profile declares arm_dof={arm_dof}"
        )
    if jp.shape[1] > arm_dof:
        # Only pad columns may be dropped. A non-zero surplus column means the
        # trajectory is for a wider arm than the profile, and truncating it
        # would silently discard a real joint.
        surplus = jp[:, arm_dof:]
        if np.any(surplus != 0.0):
            raise RuntimeError(
                f"{src_path} joint_position has {jp.shape[1]} columns for an "
                f"arm_dof={arm_dof} robot, and columns {arm_dof}..{jp.shape[1] - 1} "
                f"are not all zero (max |value| = {np.abs(surplus).max():.6g}); "
                f"refusing to drop real joint data — check the episode's robot_type"
            )
        print(f"[raw_episode] joint_position {jp.shape[1]} -> {arm_dof} columns "
              f"(dropped {jp.shape[1] - arm_dof} zero pad column(s))")
        jp = jp[:, :arm_dof]

    with h5py.File(dst_path, "w") as dst:
        dst.create_dataset("joint_position", data=jp)
        dst.create_dataset("gripper_position", data=gp)
    return int(jp.shape[0])


_VISUAL_INPUT_TEMPLATE = '''"""VisualInput entry — {episode_id}.

Auto-generated by ``ar2s.agents_visual.cli.build_raw_episode`` from
``{raw_data_rel}``. Primary camera chosen via {selection_label}: serial
{chosen_serial} ({chosen_role}).{secondary_doc_block}

The artefacts under ``<run_dir>/raw_episodes/`` (rectified
stereo .mp4(s) + SDK/SVO K sidecar(s) + flat trajectory h5 + extrinsics .npz)
were derived from the original DROID trajectory.h5 + cameras.json. Source
backups (metadata, cameras json, raw trajectory.h5) live under
``<run_dir>/raw_episodes/source/``.

IMPORTANT: ``cameras_extrinsics.npz`` was built directly from the
*refined* or *vggt* optimized extrinsics in the cameras json as 4x4 w2c
(``T_cam_from_world``) matrices. Do NOT switch to DROID metadata.json's
6-DoF extrinsics; they are unreliable for sim coordinate frames.

Hints (object_texts / pickup / ground) left empty — visual's
subagents decide from the first rectified frame.

Run on the GPU box:
    python -m ar2s.run_pipeline --input <this_file.py>
"""
import os
from pathlib import Path
from ar2s.agents_visual.inputs import VisualInput

_RUN_DIR = Path(__file__).resolve().parent
_RAW = os.environ.get(
    "DROID_RAW_EPISODE",
    str(_RUN_DIR / "raw_episodes"),
)
VISUAL_INPUT = VisualInput(
    stereo_stream_path=f"{{_RAW}}/{primary_mp4_name}",
    stereo_intrinsics_path=f"{{_RAW}}/{primary_intrinsics_name}",
    robot_traj_path=f"{{_RAW}}/robot_traj.h5",
    episode_id="{episode_id}",
    scene_name="{episode_id}",
    max_frames=None,
    frame_step=5,
    object_texts=[],
    pickup_object="",
    ground_reference_object="",
    cameras_extrinsics_path=f"{{_RAW}}/cameras_extrinsics.npz",
    task_description={task_description_literal},
    object_mass_hints={{}},
    object_friction_hints={{}},
    # RobotProfile registry name; becomes manifest.robot.type, which is what
    # sysid uses to pick the arm/gripper model.
    robot_type="{robot_type}",
    # Dual-view: secondary camera kept for segment_controller's lazy
    # fallback (PR-3+). Empty strings = single-camera episode.
    secondary_stereo_stream_path={secondary_stream_literal},
    secondary_stereo_intrinsics_path={secondary_intrinsics_literal},
    primary_camera_id="{chosen_serial}",
    secondary_camera_id="{secondary_serial}",
    # Wrist-cam rectified video, present iff the raw_data folder had a
    # wrist <serial>-stereo.mp4. Used only as an alternate frame source
    # (object_discovery); empty string = not available.
    wrist_stream_path={wrist_stream_literal},
)
'''


def _write_visual_input_stub(
    target_path: Path,
    *,
    episode_id: str,
    raw_data_rel: str,
    chosen_serial: str,
    chosen_role: str,
    task_description: str,
    selection_label: str,
    primary_mp4_name: str = "stereo.mp4",
    primary_intrinsics_name: str = "stereo.K.txt",
    secondary_serial: str = "",
    secondary_role: str = "",
    secondary_mp4_name: str = "",
    secondary_intrinsics_name: str = "",
    wrist_mp4_name: str = "",
    robot_type: str = "franka_panda",
) -> None:
    """Generate <run_dir>/visual_<id>.py from the template above.

    Dual-view: when ``secondary_serial`` is empty (single-camera episode)
    the stub fills canonical secondary fields as empty strings so downstream
    code's "is secondary present?" check stays a single truthiness test.
    Same pattern for ``wrist_mp4_name`` empty (no wrist video staged).
    """
    if secondary_serial:
        secondary_stream_literal = f'f"{{_RAW}}/{secondary_mp4_name}"'
        secondary_intrinsics_literal = f'f"{{_RAW}}/{secondary_intrinsics_name}"'
        role_tag = f" ({secondary_role})" if secondary_role else ""
        secondary_doc_block = (
            f"\nSecondary camera (runner-up; kept for segment fallback): "
            f"serial {secondary_serial}{role_tag}."
        )
    else:
        secondary_stream_literal = '""'
        secondary_intrinsics_literal = '""'
        secondary_doc_block = ""
    wrist_stream_literal = (
        f'f"{{_RAW}}/{wrist_mp4_name}"' if wrist_mp4_name else '""'
    )
    content = _VISUAL_INPUT_TEMPLATE.format(
        episode_id=episode_id,
        raw_data_rel=raw_data_rel,
        chosen_serial=chosen_serial,
        chosen_role=chosen_role,
        primary_mp4_name=primary_mp4_name,
        primary_intrinsics_name=primary_intrinsics_name,
        task_description_literal=repr(task_description),
        selection_label=selection_label,
        secondary_serial=secondary_serial,
        secondary_stream_literal=secondary_stream_literal,
        secondary_intrinsics_literal=secondary_intrinsics_literal,
        secondary_doc_block=secondary_doc_block,
        wrist_stream_literal=wrist_stream_literal,
        robot_type=robot_type,
    )
    target_path.write_text(content)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    episode_id: str
    run_id: str
    run_dir: Path
    raw_episode_dir: Path
    visual_input_path: Path           # <run_dir>/visual_<id>.py
    chosen_serial: str
    chosen_role: str
    task_description: str
    robot_type: str
    n_traj_frames: int
    selection: CameraSelectResult
    # Dual-view: empty when build was single-camera.
    secondary_serial: str = ""
    secondary_role: str = ""


def build_raw_episode(
    raw_data_dir: Path,
    *,
    episode_id: str | None = None,
    run_suffix: str | None = None,
    artifact_root: Path | None = None,
    out_root: Path | None = None,            # default: <run_dir>/raw_episodes
    inputs_dir: Path | None = None,          # default: <run_dir> itself
    forced_camera: str = "auto",             # "auto" | "ext1" | "ext2" | "<serial>"
    overwrite: bool = False,
    link_source: bool | None = None,         # None = auto-detect persistence
    robot_type: str | None = None,           # None = stage_manifest's, else franka_panda
) -> BuildResult:
    """Convert raw_data/ into the current run's raw/input artifacts.

    Default output layout:

        outputs/run_pipeline/<run_id>/
            raw_episodes/
            visual_<episode_id>.py

    ``episode_id`` must follow the DROID-100 convention
    ``droid_100_episode_<processed_idx:03d>_<raw_id_slug>``. If raw_data/
    contains ``stage_manifest.json``, the id is read from that manifest and any
    explicit ``episode_id`` must match it.

    ``forced_camera``:
      - ``"auto"`` (default) — single external picked directly; multiple
        externals trigger the camera_select VLM vote.
      - ``"ext1"`` / ``"ext2"`` — pick the camera with that role from
        metadata.json's ext{1,2}_cam_serial fields.
      - any other string — interpreted as a literal serial; must match one
        of the available externals.

    ``robot_type`` names the RobotProfile this episode was recorded on. It ends
    up in the VisualInput stub and from there in ``manifest.robot.type``, which
    is what sysid reads to pick the arm model. Default: the raw_data
    ``stage_manifest.json``'s ``robot_type`` if present (self-collected
    episodes write it), else ``franka_panda`` (every DROID episode).
    """
    episode_id = resolve_droid_100_episode_id(raw_data_dir, episode_id)
    robot_type = _resolve_robot_type(raw_data_dir, robot_type)
    robot_profile = get_robot_profile(robot_type)
    run_id = run_id_for(episode_id, run_suffix)
    run_dir = pipeline_run_dir(run_id, artifact_root)
    layout = _discover_raw_data(raw_data_dir)
    # One run dir holds one episode (see ar2s.pipeline_artifacts), so neither
    # the old raw_episodes/<episode_id>/ nor visual_inputs/ level could ever
    # hold more than one child.
    inputs_dir = Path(inputs_dir) if inputs_dir else run_dir
    out_dir = Path(out_root) if out_root else (run_dir / "raw_episodes")

    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{out_dir} already exists; pass --overwrite to replace it"
        )
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    print(f"[raw_episode] raw_data: {layout.raw_data_dir}")
    print(f"[raw_episode] metadata: {layout.metadata_path.name}")
    print(f"[raw_episode] cameras json: {layout.cameras_json_path.name}")
    print(f"[raw_episode] trajectory: {layout.trajectory_h5_path.name}")
    print(f"[raw_episode] wrist serial (excluded): {layout.wrist_serial}")
    print(f"[raw_episode] ext serials declared: {layout.ext_serials}")
    print(f"[raw_episode] robot: {robot_profile.name} + {robot_profile.gripper_name} "
          f"(arm_dof={robot_profile.arm_dof})")
    print(f"[raw_episode] externals usable (mp4 + K.txt + optimized extrinsics): "
          f"{[c.serial for c in layout.available_externals]}")

    if not layout.available_externals:
        raise RuntimeError(
            "no usable external camera: none of the declared ext serials had "
            "a <serial>-stereo.mp4 + matching <serial>-stereo.K.txt + an "
            f"optimized extrinsics entry in cameras.json — {_EXPORT_HINT}"
        )

    # ---- pick camera ----
    selection: CameraSelectResult
    task_desc = (layout.metadata.get("current_task") or "").strip()

    if forced_camera != "auto":
        forced_serial: str | None = None
        for c in layout.available_externals:
            if forced_camera in (c.serial, c.role):
                forced_serial = c.serial
                break
        if forced_serial is None:
            raise ValueError(
                f"forced_camera={forced_camera!r} does not match any available "
                f"external (serials: {[c.serial for c in layout.available_externals]}, "
                f"roles: {[c.role for c in layout.available_externals]})"
            )
        chosen = next(c for c in layout.available_externals if c.serial == forced_serial)
        # Dual-view (PR-1+): even in forced mode, pick a runner-up from the
        # remaining candidates so segment fallback has a secondary view
        # available. Tie-break alphabetically — deterministic across reruns.
        other_serials = sorted(c.serial for c in layout.available_externals
                                if c.serial != forced_serial)
        secondary_forced = other_serials[0] if other_serials else ""
        selection = CameraSelectResult(
            chosen_serial=forced_serial,
            rationale=f"forced by --camera={forced_camera}",
            votes={},
            per_model_pick={},
            secondary_serial=secondary_forced,
        )
        selection_label = f"--camera={forced_camera}"
        print(f"[raw_episode] camera forced -> {forced_serial} ({chosen.role})")
        if secondary_forced:
            print(f"[raw_episode] secondary (runner-up): {secondary_forced}")
    else:
        selection = camera_select_vlm(
            layout.available_externals,
            task_description=task_desc,
        )
        chosen = next(
            c for c in layout.available_externals if c.serial == selection.chosen_serial
        )
        selection_label = "VLM (camera_select)" if len(layout.available_externals) > 1 else "single candidate"

    chosen_role = chosen.role

    # Dual-view (PR-1): identify the runner-up so its stereo + extrinsics get
    # bundled alongside primary. ``secondary_serial`` may be "" — single-cam
    # episodes flow through unchanged.
    secondary_serial = selection.secondary_serial or ""
    secondary_candidate: CameraCandidate | None = None
    if secondary_serial:
        for c in layout.available_externals:
            if c.serial == secondary_serial:
                secondary_candidate = c
                break
        if secondary_candidate is None:
            # camera_select returned a serial not in available_externals — this
            # shouldn't happen since the vote is restricted to those serials,
            # but bail loudly rather than silently drop the runner-up.
            raise RuntimeError(
                f"camera_select returned secondary_serial={secondary_serial!r} "
                f"but it is not among available externals "
                f"({[c.serial for c in layout.available_externals]})"
            )

    # ---- write outputs ----
    # Verbatim inputs are symlinked when the source outlives the job and copied
    # otherwise; the derived artifacts below (extrinsics npz, flat trajectory
    # h5, camera_selection.json) are always real files with no upstream twin.
    if link_source is None:
        link_source, why = source_is_persistent(layout.raw_data_dir)
    else:
        why = "caller override"
    verb = "linking" if link_source else "copying"
    print(f"[raw_episode] raw_data is "
          f"{'persistent' if link_source else 'ephemeral'}: {why} -> {verb} inputs")

    def _place(src: Path, dst: Path, label: str) -> None:
        print(f"[raw_episode] {verb} {src.name} -> {dst} ({label})")
        place_raw_source(src, dst, link_source=bool(link_source))

    # Role-keyed staging: EVERY usable external is copied under its physical
    # role name, independent of which one was chosen primary. "Primary" is
    # recorded as an attribute (camera_selection.json + the VisualInput stub),
    # never as a file identity — see module docstring.
    role_file_names: dict[str, tuple[str, str]] = {}   # serial -> (mp4_name, K_name)
    for cand in layout.available_externals:
        mp4_name = f"{cand.role}-stereo.mp4"
        k_name = f"{cand.role}-stereo.K.txt"
        role_file_names[cand.serial] = (mp4_name, k_name)
        tag = "primary" if cand.serial == chosen.serial else (
            "secondary" if cand.serial == secondary_serial else "extra"
        )
        _place(cand.mp4_path, out_dir / mp4_name, f"{cand.role}, {tag}")
        assert cand.sdk_intrinsics_path is not None, cand
        _place(cand.sdk_intrinsics_path, out_dir / k_name, f"{cand.role}, {tag}")

    primary_mp4_name, primary_intrinsics_name = role_file_names[chosen.serial]
    secondary_mp4_name = ""
    secondary_intrinsics_name = ""
    if secondary_candidate is not None:
        secondary_mp4_name, secondary_intrinsics_name = (
            role_file_names[secondary_candidate.serial]
        )

    wrist_mp4_name = ""
    if layout.wrist_mp4_path is not None:
        wrist_mp4_name = "wrist.mp4"
        _place(layout.wrist_mp4_path, out_dir / wrist_mp4_name, "wrist")

    extr_dst = out_dir / "cameras_extrinsics.npz"
    # Extrinsics for every staged external (role-keyed files above), not just
    # chosen+secondary — the npz is serial-keyed so extra entries are free and
    # keep the bundle self-contained if primary is re-picked later.
    extr_serials = [c.serial for c in layout.available_externals]
    mats = _write_extrinsics_npz(extr_dst, layout.cameras_json, extr_serials)
    for s, mat in mats.items():
        print(f"[raw_episode] wrote {extr_dst.name} cam_mat_{s} "
              f"(translation: {mat[:3, 3]})")

    traj_dst = out_dir / "robot_traj.h5"
    n_frames = _write_flat_trajectory_h5(
        layout.trajectory_h5_path, traj_dst, arm_dof=robot_profile.arm_dof
    )
    print(f"[raw_episode] wrote {traj_dst.name} with {n_frames} steps")

    # Traceability backups.
    source_dir = out_dir / "source"
    source_dir.mkdir()
    shutil.copy2(layout.metadata_path, source_dir / "metadata.json")
    shutil.copy2(layout.cameras_json_path, source_dir / "cameras.json")
    _place(layout.trajectory_h5_path, source_dir / "trajectory_raw.h5", "source backup")
    print(f"[raw_episode] backed up source files -> {source_dir}")

    # Persist selection details for reproducibility (rerun --input recipe
    # skips the VLM, but the artifact records why this serial was picked).
    (out_dir / "camera_selection.json").write_text(json.dumps({
        "layout":         "role_keyed_v2",
        "chosen_serial":  selection.chosen_serial,
        "chosen_role":    chosen_role,
        "primary_mp4":    primary_mp4_name,
        "primary_intrinsics": primary_intrinsics_name,
        "secondary_serial": secondary_serial,
        "secondary_role": secondary_candidate.role if secondary_candidate else "",
        "secondary_mp4":  secondary_mp4_name,
        "secondary_intrinsics": secondary_intrinsics_name,
        "rationale":      selection.rationale,
        "votes":          selection.votes,
        "per_model_pick": selection.per_model_pick,
        "fallback_reason": selection.fallback_reason,
        "candidates":     [
            {
                "serial": c.serial,
                "role": c.role,
                "mp4": c.mp4_path.name,
                "sdk_intrinsics": (
                    c.sdk_intrinsics_path.name
                    if c.sdk_intrinsics_path is not None else None
                ),
            }
            for c in layout.available_externals
        ],
        "selection_label": selection_label,
        "frame_indices":  selection.per_camera_frame_indices,
    }, indent=2))

    # ---- write <run_dir>/visual_<id>.py stub ----
    visual_input_path = inputs_dir / f"visual_{episode_id}.py"
    if visual_input_path.exists() and not overwrite:
        existing = visual_input_path.read_text()
        required = ("stereo_stream_path", "stereo_intrinsics_path")
        missing_required = [field for field in required if field not in existing]
        secondary_mentions = (
            "secondary_stereo_stream_path" in existing
            or "secondary_svo_path" in existing
            or "secondary_camera_id" in existing
        )
        missing_secondary = []
        if secondary_mentions:
            missing_secondary = [
                field for field in (
                    "secondary_stereo_stream_path",
                    "secondary_stereo_intrinsics_path",
                )
                if field not in existing
            ]
        if missing_required or missing_secondary:
            raise FileExistsError(
                f"{visual_input_path} already exists but lacks canonical "
                f"VisualInput fields: {missing_required + missing_secondary}; "
                f"pass --overwrite to regenerate it"
            )
        print(f"[raw_episode] WARNING: {visual_input_path} already exists, leaving "
              f"as-is (re-run with --overwrite to regenerate)")
    else:
        inputs_dir.mkdir(parents=True, exist_ok=True)
        raw_data_rel = str(layout.raw_data_dir.relative_to(_REPO_ROOT)) \
            if layout.raw_data_dir.is_relative_to(_REPO_ROOT) \
            else str(layout.raw_data_dir)
        _write_visual_input_stub(
            visual_input_path,
            episode_id=episode_id,
            raw_data_rel=raw_data_rel,
            chosen_serial=chosen.serial,
            chosen_role=chosen_role,
            primary_mp4_name=primary_mp4_name,
            primary_intrinsics_name=primary_intrinsics_name,
            task_description=task_desc,
            selection_label=selection_label,
            secondary_serial=secondary_serial,
            secondary_role=(secondary_candidate.role if secondary_candidate else ""),
            secondary_mp4_name=secondary_mp4_name,
            secondary_intrinsics_name=secondary_intrinsics_name,
            wrist_mp4_name=wrist_mp4_name,
            robot_type=robot_type,
        )
        try:
            display_path = visual_input_path.relative_to(_REPO_ROOT)
        except ValueError:
            display_path = visual_input_path
        print(f"[raw_episode] wrote {display_path}")

    print()
    print(f"[raw_episode] DONE: {out_dir} ready, "
          f"camera = {chosen.serial} ({chosen_role})")

    return BuildResult(
        episode_id=episode_id,
        run_id=run_id,
        run_dir=run_dir,
        raw_episode_dir=out_dir,
        visual_input_path=visual_input_path,
        chosen_serial=chosen.serial,
        chosen_role=chosen_role,
        task_description=task_desc,
        robot_type=robot_type,
        n_traj_frames=n_frames,
        selection=selection,
        secondary_serial=secondary_serial,
        secondary_role=(secondary_candidate.role if secondary_candidate else ""),
    )


def build_visual_input(result: BuildResult) -> VisualInput:
    """Convenience: load VISUAL_INPUT from the freshly-written stub."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"visual_{result.episode_id}",
        result.visual_input_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {result.visual_input_path}")
    mod = importlib.util.module_from_spec(spec)
    # The stub is read once per run; caching it would only litter the artifact
    # dir with a __pycache__.
    prev_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev_dont_write
    if not hasattr(mod, "VISUAL_INPUT"):
        raise RuntimeError(
            f"{result.visual_input_path} does not export VISUAL_INPUT after build"
        )
    return mod.VISUAL_INPUT
