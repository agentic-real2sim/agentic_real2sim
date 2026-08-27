"""Stage 2 skill: resolve the primary camera.

Single-camera default
---------------------
visual already runs ``camera_select`` at the raw_data prep stage
(VLM picks one external camera before the rest of the pipeline). The
emitted episode therefore has ``cameras.cam_ids == [single_serial]``
in every case — there is no "pick" to make here. This skill short-
circuits to a trivial CameraSelection with no IoU computation and no
mask dependency.

Multi-camera interface (future, N≥2)
------------------------------------
The dataclass + entry point are kept so a future multi-camera pipeline
can plug in without rewriting Stage 2 wiring. The selection criterion
in that world will be **per-camera object-pose-estimation quality**
(e.g. FoundationPose reprojection error, mask coverage, depth coverage)
rather than the historical robot-mask IoU below — robot mask IoU only
disambiguates which camera captured the mask, not which camera is best
for downstream sim. Until the multi-cam path is implemented, N≥2 input
falls back to the manifest's ``primary_id`` when robot_mask is missing,
or runs the legacy IoU disambiguator when present.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import imageio.v2 as imageio
import numpy as np

from ar2s.agents_sysid.inputs import InputBundle
from ar2s.droid_sim.scene.robot_only_sim import RobotOnlySim


_DEFAULT_INIT_POS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
_DEFAULT_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w,x,y,z

_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_MIN_CONFIDENCE = 0.05


@dataclass
class CameraSelection:
    primary_id: int
    primary_index: int                       # index in bundle.cameras.camera_ids
    iou_per_camera: dict[int, float] = field(default_factory=dict)
    confidence: float = 0.0                  # best IoU − 2nd-best IoU


def _load_real_mask(path: str) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img > 127


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.count_nonzero(mask_a & mask_b))
    union = int(np.count_nonzero(mask_a | mask_b))
    return intersection / union if union else 0.0


def _render_robot_mask(sim: RobotOnlySim, cam_idx: int,
                       width: int, height: int) -> np.ndarray:
    frames = sim.render_cameras(
        width=width, height=height,
        visible_groups=(RobotOnlySim.GEOM_GROUP_ROBOT_VISUAL,),
    )
    distorted = sim.apply_distortion(frames)
    return np.any(distorted[cam_idx] > 10, axis=2)


def select_primary_camera(
    bundle: InputBundle,
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> CameraSelection:
    """Resolve the primary camera.

    Single-camera path (N=1, the visual default): trivial — the only
    cam is the primary. No mask, no render, no IoU. ``iou_per_camera``
    is left empty and ``confidence`` is set to 1.0 to mark the
    unambiguous case.

    Multi-camera path (N≥2, future):
      * If ``robot_mask_path`` is present, run the legacy IoU-based
        disambiguator (renders the robot from each cam, picks the
        viewpoint whose silhouette best matches ``real/robot_mask.png``).
        Raises if confidence < ``min_confidence``.
      * If ``robot_mask_path`` is absent, fall back to the manifest's
        ``primary_id``. This is a temporary bridge — the proper
        multi-cam selection criterion will be per-camera object-pose
        estimation quality, not robot-mask IoU.

    Args:
        bundle: episode InputBundle (provides robot traj, mask path, cameras).
        width / height: render resolution (defaults match v1 pose_align).
        min_confidence: required gap between best and 2nd-best IoU
            (multi-cam path only).
    """
    cam_ids = list(bundle.cameras.camera_ids)
    if not cam_ids:
        raise RuntimeError("bundle.cameras.camera_ids is empty")

    # ------------- single-camera default path -------------
    if len(cam_ids) == 1:
        only = int(cam_ids[0])
        return CameraSelection(
            primary_id=only,
            primary_index=0,
            iou_per_camera={},
            confidence=1.0,
        )

    # ------------- trust manifest's primary_id when valid -------------
    # The visual pipeline picks the primary camera via VLM during raw_data prep, then
    # keeps BOTH cams' extrinsics in the emitted manifest for dual-view
    # downstream (PR-1). For DROID episodes we don't need to re-disambiguate
    # via robot-mask IoU — the rendered robot at identity rarely overlaps
    # the real arm anyway, and the IoU path was the "needs robot pose align"
    # bit we're now skipping by design. Just trust the manifest.
    manifest_primary = int(bundle.cameras.object_camera_id)
    if manifest_primary in cam_ids:
        return CameraSelection(
            primary_id=manifest_primary,
            primary_index=cam_ids.index(manifest_primary),
            iou_per_camera={},
            confidence=1.0,
        )

    # ------------- multi-camera (legacy + future hook) -------------
    if not bundle.robot_mask_path:
        # No mask → can't run IoU disambiguator; trust the manifest.
        manifest_primary = int(bundle.cameras.object_camera_id)
        if manifest_primary not in cam_ids:
            raise RuntimeError(
                f"N>=2 cameras but no robot_mask to disambiguate; manifest "
                f"primary_id={manifest_primary} is not in cam_ids={cam_ids}"
            )
        return CameraSelection(
            primary_id=manifest_primary,
            primary_index=cam_ids.index(manifest_primary),
            iou_per_camera={},
            confidence=0.0,         # no signal available
        )

    real_mask = _load_real_mask(bundle.robot_mask_path)

    sim = RobotOnlySim(
        robot_traj_path=bundle.robot_traj_path,
        cameras=bundle.cameras,
        robot_base_pos=_DEFAULT_INIT_POS,
        robot_base_quat=_DEFAULT_INIT_QUAT,
    )

    iou_per_camera: dict[int, float] = {}
    for idx, cam_id in enumerate(cam_ids):
        rendered = _render_robot_mask(sim, idx, width, height)
        iou_per_camera[int(cam_id)] = _iou(rendered, real_mask)

    ranked = sorted(iou_per_camera.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best_iou = ranked[0]
    confidence = best_iou - (ranked[1][1] if len(ranked) > 1 else 0.0)

    if confidence < min_confidence:
        raise RuntimeError(
            f"Camera selection ambiguous: confidence={confidence:.4f} "
            f"< {min_confidence:.4f}. Per-camera IoU: {iou_per_camera}. "
            f"Robot mask may be unusable, robot pose may be wildly off, "
            f"or the candidate cameras are too similar."
        )

    primary_index = cam_ids.index(best_id)
    return CameraSelection(
        primary_id=best_id,
        primary_index=primary_index,
        iou_per_camera=iou_per_camera,
        confidence=confidence,
    )


__all__ = ["select_primary_camera", "CameraSelection"]
