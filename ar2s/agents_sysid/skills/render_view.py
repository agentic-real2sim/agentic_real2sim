"""Stage 3 skill: render a custom camera view at a specific trajectory frame.

Used by Agent A (view sufficiency) to inspect the sim from any viewpoint,
and by the orchestrator (rounds 1+) to re-render the round-0 ``poses_chosen``
at the current shifted scene state.

Two flavors:
  - ``render_view(...)`` pure function — returns a uint8 RGB image array.
    Pure(-ish): mutates ``frame_cache.sim``'s ``qpos`` but restores it
    after rendering, so subsequent calls behave identically.
  - ``make_render_view_tool(scene, frame_cache, pickup_object)`` — returns
    a LangChain ``@tool`` that wraps the pure function, exposing the
    multimodal image content blocks an agent can see.

v1 trick preserved: ``hide_non_pickup=True`` moves all non-pickup scene
objects off-screen (qpos[:3] ← (100,100,100)) before rendering, then
restores. Lets the LLM see the pickup directly without container occlusion.
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from langchain_core.tools import tool


_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_OFFSCREEN_POS = np.array([100.0, 100.0, 100.0], dtype=np.float64)

# Camera-pose sanity bounds (v1 carryover; see agents/view_finder.py).
_MIN_DISTANCE_M = 0.2
_MAX_DISTANCE_M = 1.5
_MIN_POS_Z_M = 0.0
_MAX_POS_Z_M = 1.5
_MIN_FOVY = 20.0
_MAX_FOVY = 80.0
_MAX_IMG_DIM_FOR_LLM = 720      # downscale before sending to LLM


def _validate_pose(
    pos: tuple, lookat: tuple, fovy_deg: float,
) -> str | None:
    """Return error string if pose is out of bounds, else None."""
    p = np.asarray(pos, dtype=np.float64)
    l = np.asarray(lookat, dtype=np.float64)
    d = float(np.linalg.norm(p - l))
    if not (_MIN_DISTANCE_M <= d <= _MAX_DISTANCE_M):
        return (f"camera distance {d*100:.1f}cm out of bounds "
                f"[{_MIN_DISTANCE_M*100:.0f}, {_MAX_DISTANCE_M*100:.0f}] cm")
    if not (_MIN_POS_Z_M <= p[2] <= _MAX_POS_Z_M):
        return (f"camera z={p[2]:.2f}m out of bounds "
                f"[{_MIN_POS_Z_M:.2f}, {_MAX_POS_Z_M:.2f}] m")
    if not (_MIN_FOVY <= fovy_deg <= _MAX_FOVY):
        return f"fovy {fovy_deg:.0f}° out of bounds [{_MIN_FOVY}, {_MAX_FOVY}]°"
    return None


def _restore_frame(frame_cache, frame: int) -> None:
    """Restore the sim's qpos to cache[frame] + mj_forward."""
    sim = frame_cache.sim
    sim.data.qpos[:] = frame_cache.qpos_trajectory[frame]
    sim.data.qvel[:] = 0.0          # we only care about kinematics for render
    mujoco.mj_forward(sim.model, sim.data)


def render_view(
    frame_cache,
    frame: int,
    pos: tuple[float, float, float],
    lookat: tuple[float, float, float],
    *,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    fovy_deg: float = 45.0,
    hide_non_pickup: bool = False,
    pickup_object: str | None = None,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> np.ndarray:
    """Render a free-camera view at the given trajectory frame.

    Args:
        frame_cache: FrameCache from probe_grasp.
        frame: frame index (0 <= frame < frame_cache.n_frames).
        pos / lookat / up / fovy_deg: standard MuJoCo free-camera params.
        hide_non_pickup: if True, all non-pickup SceneObjects are moved
            off-screen before rendering, then restored.
        pickup_object: required if hide_non_pickup=True.
        width / height: render resolution.

    Returns:
        (H, W, 3) uint8 RGB image. World coordinate axes overlaid in red/green/blue.
    """
    if not (0 <= frame < frame_cache.n_frames):
        raise ValueError(f"frame {frame} out of range [0, {frame_cache.n_frames})")
    if hide_non_pickup and not pickup_object:
        raise ValueError("hide_non_pickup=True requires pickup_object")

    sim = frame_cache.sim
    _restore_frame(frame_cache, frame)

    saved: dict[str, tuple[int, np.ndarray]] = {}
    if hide_non_pickup:
        for obj in sim.config.objects:
            if obj.name == pickup_object or obj.static:
                continue
            adr = sim.model.joint(f"{obj.name}_freejoint").qposadr.item()
            saved[obj.name] = (adr, sim.data.qpos[adr:adr + 3].copy())
            sim.data.qpos[adr:adr + 3] = _OFFSCREEN_POS
        mujoco.mj_forward(sim.model, sim.data)

    try:
        img = sim.render_free_camera(
            pos=np.asarray(pos, dtype=np.float64),
            lookat=np.asarray(lookat, dtype=np.float64),
            fovy=float(fovy_deg),
            width=width, height=height,
            up=tuple(up),
            show_axes=True,
        )
    finally:
        for _, (adr, qp) in saved.items():
            sim.data.qpos[adr:adr + 3] = qp
        if hide_non_pickup:
            mujoco.mj_forward(sim.model, sim.data)

    return img


# -------------------------------------------------------------------
# LLM-facing wrapper: produces multimodal content blocks
# -------------------------------------------------------------------

def _encode_jpeg(img: np.ndarray, max_dim: int = _MAX_IMG_DIM_FOR_LLM) -> str:
    from PIL import Image
    pil = Image.fromarray(img)
    if max(pil.size) > max_dim:
        pil.thumbnail((max_dim, max_dim))
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _content_text(text: str) -> dict:
    return {"type": "text", "text": text}


def _content_image_b64(b64_jpeg: str) -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"}}


def make_render_view_tool(scene, frame_cache, pickup_object: str,
                          *, save_dir: Path | None = None,
                          save_prefix: str = "custom"):
    """Build a LangChain @tool exposing render_view to a ReAct agent.

    Args:
        scene: CalibratedScene (closure capture).
        frame_cache: from probe_grasp.
        pickup_object: needed for hide_non_pickup.
        save_dir: if set, every rendered image is also written to disk
            so the orchestrator can persist it in the round artifact.
        save_prefix: filename prefix for saved images.

    Returns:
        ``@tool`` function ``render_view`` for the agent.
    """
    n_frames = frame_cache.n_frames
    engage_frame = frame_cache.engage_frame
    counter = {"i": 0}                      # mutable counter for save filenames

    @tool(response_format="content")
    def render_view(
        frame: int,
        pos_x: float, pos_y: float, pos_z: float,
        lookat_x: float, lookat_y: float, lookat_z: float,
        fovy_deg: float = 45.0,
        hide_non_pickup: bool = False,
    ) -> list[dict]:
        """Render the sim from a custom free camera at a specific trajectory frame.

        Use this when the 4 baseline images (2 real cameras × full/pickup-only)
        don't clearly show the gripper↔pickup spatial relation. Pick a camera
        pos + lookat that gives an oblique view; pure side / pure top-down
        rarely work for occluded pickups.

        Args:
            frame: trajectory frame index (0 <= frame < {n_frames}).
                Useful frames: engage_frame (={engage_frame}, gripper starts
                closing); engage_frame+30/+60 (during/after grasp attempt).
            pos_x, pos_y, pos_z: camera position in WORLD coords (meters).
                Bounds: distance from lookat ∈ [0.2, 1.5] m, z ∈ [0, 1.5] m.
            lookat_x, lookat_y, lookat_z: WORLD point the camera looks at.
            fovy_deg: vertical field of view in degrees, [20, 80].
            hide_non_pickup: if True, render with non-pickup objects hidden
                (lets you see the pickup without container occlusion).

        Returns:
            text caption + image content block. World axes overlaid in red/green/blue.
        """
        pos = (float(pos_x), float(pos_y), float(pos_z))
        lookat = (float(lookat_x), float(lookat_y), float(lookat_z))

        err = _validate_pose(pos, lookat, fovy_deg)
        if err:
            return [_content_text(
                f"ERROR — invalid camera pose: {err}. "
                f"Adjust pos/lookat/fovy and try again."
            )]

        try:
            img = render_view_core(
                frame_cache,
                frame=int(frame),
                pos=pos, lookat=lookat, fovy_deg=float(fovy_deg),
                hide_non_pickup=bool(hide_non_pickup),
                pickup_object=pickup_object,
            )
        except Exception as e:
            return [_content_text(f"ERROR — render failed: {type(e).__name__}: {e}")]

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{save_prefix}_{counter['i']:02d}_f{frame:04d}.jpg"
            imageio.imwrite(str(save_dir / fname), img)
            counter["i"] += 1

        caption = (
            f"Rendered: frame={frame} (engage_frame={engage_frame}), "
            f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}), "
            f"lookat=({lookat[0]:+.2f}, {lookat[1]:+.2f}, {lookat[2]:+.2f}), "
            f"fovy={fovy_deg:.0f}°"
            + (", pickup-only (other objects hidden)" if hide_non_pickup else "")
            + ". Red=+x, Green=+y, Blue=+z world axes overlaid."
        )
        return [_content_text(caption), _content_image_b64(_encode_jpeg(img))]

    return render_view


# Public alias for the pure function so callers can disambiguate.
render_view_core = render_view


__all__ = ["render_view", "render_view_core", "make_render_view_tool"]
