"""Stage 3 Agent A — view sufficiency loop.

Runs ONCE per grasp-loop session, at round 0. Given the 4 baseline images
(2 real cameras × full / pickup-only) it judges whether they suffice for
the shift-decision agent. If not, it may request up to 3 custom render
views. The collected camera poses are then reused for every subsequent
round of this same grasp loop.

LangGraph ReAct agent. Tools:
  - render_view (closure-wrapped — records poses, enforces 3-call cap)
  - submit_sufficient (signal exit)

Returns a list of PoseRecord dicts (4 baseline + 0-3 custom). Does NOT
propose shifts.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ar2s.agent_configs.models import chat_model_for
from ar2s.agents_sysid.skills.probe_grasp import FrameCache
from ar2s.agents_sysid.skills.render_view import make_render_view_tool, render_view_core


_RECURSION_LIMIT = 15               # ~7 LLM-tool round trips (Agent A loop)
_AGENT_A_MAX_CUSTOM_RENDERS = 3
_IMAGE_MAX_DIM = 720
_PROMPT_PATH = Path(__file__).parent / "view_sufficiency_prompt.md"

# Default baseline frame offsets from engage_frame — captures the grasp
# evolution: opening contact / mid-close / lift attempt. Mirrors v1.
_DEFAULT_BASELINE_FRAME_OFFSETS = (0, 30, 60)


# -------------------------------------------------------------------
# Pose record + run result
# -------------------------------------------------------------------

@dataclass
class PoseRecord:
    """One camera pose used in the grasp loop. Re-rendered each round."""
    kind: str                                   # "baseline_real" | "custom"
    cam_idx: int | None                         # for baseline_real (real cam index)
    hide_non_pickup: bool
    frame: int                                  # trajectory frame
    pos: tuple | None                           # custom only (None for baseline_real)
    lookat: tuple | None
    up: tuple = (0.0, 0.0, 1.0)
    fovy_deg: float = 45.0
    label: str = ""                             # short human label for filename

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "cam_idx": self.cam_idx,
            "hide_non_pickup": self.hide_non_pickup,
            "frame": self.frame, "pos": list(self.pos) if self.pos else None,
            "lookat": list(self.lookat) if self.lookat else None,
            "up": list(self.up), "fovy_deg": self.fovy_deg,
            "label": self.label,
        }


@dataclass
class ViewSufficiencyResult:
    poses_chosen: list[PoseRecord]              # 4 baseline + 0-3 custom
    image_paths: list[Path]                     # paths to rendered images on disk
    n_custom_renders: int
    final_reasoning: str                        # from submit_sufficient
    terminated_by: str                          # "submit" | "recursion_limit"


# -------------------------------------------------------------------
# Baseline rendering (orchestrator-side; called BEFORE Agent A runs)
# -------------------------------------------------------------------

def _frame_label(frame: int, engage_frame: int, frame_dt: float) -> str:
    """Human-readable label for a frame, anchored at engage_frame.

    Examples:
      engage_frame  → "engage_frame, t=6.52s"
      engage + 30   → "engage+30 (~0.50s after engage), t=7.02s"
      engage − 10   → "engage−10 (~0.17s before engage), t=6.35s"
    """
    t = frame * frame_dt
    delta = frame - engage_frame
    if delta == 0:
        return f"engage_frame, t={t:.2f}s"
    dt = abs(delta) * frame_dt
    side = "after" if delta > 0 else "before"
    return f"engage{'+' if delta > 0 else '−'}{abs(delta)} (~{dt:.2f}s {side} engage), t={t:.2f}s"


def _render_one_baseline_set(
    frame_cache: FrameCache,
    pickup_object: str,
    save_dir: Path,
    frame: int,
) -> tuple[list[PoseRecord], list[Path]]:
    """Render 4 images at ONE frame: 2 real cams × {full, pickup-only}."""
    import mujoco

    sim = frame_cache.sim
    cam_ids = sim.config.cameras.camera_ids
    poses: list[PoseRecord] = []
    paths: list[Path] = []

    for hide in (False, True):
        sim.data.qpos[:] = frame_cache.qpos_trajectory[frame]
        sim.data.qvel[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)

        saved: dict = {}
        if hide:
            for obj in sim.config.objects:
                if obj.name == pickup_object or obj.static:
                    continue
                adr = sim.model.joint(f"{obj.name}_freejoint").qposadr.item()
                saved[obj.name] = (adr, sim.data.qpos[adr:adr + 3].copy())
                sim.data.qpos[adr:adr + 3] = np.array([100., 100., 100.])
            mujoco.mj_forward(sim.model, sim.data)
        try:
            frames = sim.render_cameras(width=1280, height=720,
                                         visible_groups=tuple(range(6)),
                                         show_axes=True)
            distorted = sim.apply_distortion(frames)
            for idx, cam_id in enumerate(cam_ids):
                mode_tag = "pickup" if hide else "full"
                fname = f"baseline_cam{idx}_{cam_id}_{mode_tag}_f{frame:04d}.jpg"
                p = save_dir / fname
                imageio.imwrite(str(p), distorted[idx])
                poses.append(PoseRecord(
                    kind="baseline_real", cam_idx=idx, hide_non_pickup=hide,
                    frame=frame, pos=None, lookat=None,
                    label=f"cam{idx}_{mode_tag}_f{frame}",
                ))
                paths.append(p)
        finally:
            for _, (adr, qp) in saved.items():
                sim.data.qpos[adr:adr + 3] = qp
            if hide:
                mujoco.mj_forward(sim.model, sim.data)

    # Order: cam0_full, cam0_pickup, cam1_full, cam1_pickup ...
    n_cams = len(cam_ids)
    ordered_poses: list[PoseRecord] = []
    ordered_paths: list[Path] = []
    for cam_idx in range(n_cams):
        for mode in ("full", "pickup"):
            for pose, path in zip(poses, paths):
                if (pose.cam_idx == cam_idx
                        and ((mode == "pickup") == pose.hide_non_pickup)):
                    ordered_poses.append(pose)
                    ordered_paths.append(path)
                    break
    return ordered_poses, ordered_paths


def render_baseline_views(
    frame_cache: FrameCache,
    pickup_object: str,
    save_dir: Path,
    *,
    frame_offsets: tuple[int, ...] = _DEFAULT_BASELINE_FRAME_OFFSETS,
) -> tuple[list[PoseRecord], list[Path]]:
    """Render N×4 baseline images: 2 real cams × {full, pickup-only} × N frames.

    Default ``frame_offsets = (0, 30, 60)`` from engage_frame — captures
    grasp evolution (initial contact / mid-close / lift attempt). Out-of-range
    offsets are silently clamped to the last available frame.

    Returns (poses, image_paths), grouped by frame in offset order.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    engage = frame_cache.engage_frame
    last = frame_cache.n_frames - 1

    # Compute concrete frames (clamped + deduped, preserve order)
    seen: set[int] = set()
    frames: list[int] = []
    for off in frame_offsets:
        f = min(max(engage + int(off), 0), last)
        if f not in seen:
            seen.add(f)
            frames.append(f)

    all_poses: list[PoseRecord] = []
    all_paths: list[Path] = []
    for f in frames:
        poses, paths = _render_one_baseline_set(
            frame_cache, pickup_object, save_dir, f,
        )
        all_poses.extend(poses)
        all_paths.extend(paths)
    return all_poses, all_paths


# -------------------------------------------------------------------
# Re-render a pose at any frame (for rounds 1+ + Agent B input)
# -------------------------------------------------------------------

def rerender_pose(
    pose: PoseRecord,
    frame_cache: FrameCache,
    pickup_object: str,
    save_path: Path,
    *,
    frame: int | None = None,
) -> Path:
    """Re-render a stored PoseRecord at a (potentially new) frame.

    For ``kind == "baseline_real"`` uses the calibrated real camera.
    For ``kind == "custom"`` uses the stored free-camera params.
    """
    import mujoco

    if frame is None:
        frame = pose.frame
    sim = frame_cache.sim
    sim.data.qpos[:] = frame_cache.qpos_trajectory[frame]
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)

    saved: dict = {}
    if pose.hide_non_pickup:
        for obj in sim.config.objects:
            if obj.name == pickup_object or obj.static:
                continue
            adr = sim.model.joint(f"{obj.name}_freejoint").qposadr.item()
            saved[obj.name] = (adr, sim.data.qpos[adr:adr + 3].copy())
            sim.data.qpos[adr:adr + 3] = np.array([100., 100., 100.])
        mujoco.mj_forward(sim.model, sim.data)
    try:
        if pose.kind == "baseline_real":
            frames = sim.render_cameras(width=1280, height=720,
                                         visible_groups=tuple(range(6)),
                                         show_axes=True)
            distorted = sim.apply_distortion(frames)
            img = distorted[pose.cam_idx]
        else:
            img = render_view_core(
                frame_cache, frame=frame,
                pos=tuple(pose.pos), lookat=tuple(pose.lookat),
                up=tuple(pose.up), fovy_deg=pose.fovy_deg,
                hide_non_pickup=pose.hide_non_pickup,
                pickup_object=pickup_object,
            )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(save_path), img)
    finally:
        for _, (adr, qp) in saved.items():
            sim.data.qpos[adr:adr + 3] = qp
        if pose.hide_non_pickup:
            mujoco.mj_forward(sim.model, sim.data)

    return save_path


# -------------------------------------------------------------------
# Image embedding for initial Agent A message
# -------------------------------------------------------------------

def _encode_image(path: Path, max_dim: int = _IMAGE_MAX_DIM) -> str:
    img = Image.open(path)
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_initial_message(
    pickup_object: str,
    baseline_paths: list[Path],
    baseline_poses: list[PoseRecord],
    engage_frame: int,
    frame_dt: float,
    cam_ids: tuple[int, ...],
) -> list[dict]:
    # Summarize the baseline coverage: which frames + which cams + which modes.
    unique_frames = sorted({p.frame for p in baseline_poses})
    frame_summary_lines = []
    for f in unique_frames:
        frame_summary_lines.append(f"  - frame {f} ({_frame_label(f, engage_frame, frame_dt)})")

    text = (
        f"# Grasp failed (round 0). Pickup object: {pickup_object}\n\n"
        f"# Baseline images ({len(baseline_paths)} attached)\n"
        f"Coverage: **{len(cam_ids)} real cameras × {{full scene, pickup-only}} × "
        f"{len(unique_frames)} time frames** = {len(baseline_paths)} images total.\n\n"
        f"Time frames in baseline:\n" + "\n".join(frame_summary_lines) + "\n\n"
        f"Real camera ids in order: " + ", ".join(str(c) for c in cam_ids) + "\n\n"
        f"For each frame you see, the order is: "
        f"cam0_full, cam0_pickup-only, cam1_full, cam1_pickup-only.\n"
        f"`pickup-only` = same real camera, non-pickup objects (e.g. mug) hidden, "
        f"so the pickup is directly visible.\n\n"
        f"# Your task\n"
        f"Judge whether these {len(baseline_paths)} baseline images are enough for a "
        f"separate shift-decision agent. If yes → call `submit_sufficient`. If no → "
        f"call `render_view` to add a custom view (max 3) that adds genuine info."
    )
    content: list[dict] = [{"type": "text", "text": text}]
    for path, pose in zip(baseline_paths, baseline_poses):
        b64 = _encode_image(path)
        mode = "pickup-only (mug hidden)" if pose.hide_non_pickup else "full scene"
        cam_id = cam_ids[pose.cam_idx] if pose.cam_idx is not None else "?"
        caption = (
            f"\n[baseline] cam_id={cam_id} (idx={pose.cam_idx}), {mode}, "
            f"frame={pose.frame} [{_frame_label(pose.frame, engage_frame, frame_dt)}]"
        )
        content.append({"type": "text", "text": caption})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return content


# -------------------------------------------------------------------
# Closure-wrapped tools
# -------------------------------------------------------------------

def _make_tools(scene, frame_cache, pickup_object: str, save_dir: Path):
    """Build render_view + submit_sufficient with shared state."""
    state = {
        "custom_renders": 0,
        "max_custom": _AGENT_A_MAX_CUSTOM_RENDERS,
        "custom_poses": [],          # list of PoseRecord
        "custom_paths": [],          # list of Path
        "submitted": False,
        "final_reasoning": "",
    }

    # Inner render_view closure that also records the pose
    @tool(response_format="content")
    def render_view(
        frame: int,
        pos_x: float, pos_y: float, pos_z: float,
        lookat_x: float, lookat_y: float, lookat_z: float,
        fovy_deg: float = 45.0,
        hide_non_pickup: bool = False,
    ) -> list[dict]:
        """Render a custom free-camera view at the given trajectory frame.

        Use after looking at baseline images, to add ONE extra view that
        clarifies a specific ambiguity. Camera pos & lookat are in world
        meters. Up to 3 custom calls per session.

        Args:
            frame: trajectory frame index.
            pos_x/y/z: camera position (m). Distance to lookat in [0.2, 1.5].
                pos.z in [0, 1.5].
            lookat_x/y/z: point the camera looks at (m).
            fovy_deg: vertical FOV in [20, 80].
            hide_non_pickup: if True, hide non-pickup objects before render.
        """
        if state["custom_renders"] >= state["max_custom"]:
            return [{"type": "text", "text":
                f"ERROR: max {state['max_custom']} custom renders reached. "
                f"You MUST call submit_sufficient now."
            }]

        pos = (float(pos_x), float(pos_y), float(pos_z))
        lookat = (float(lookat_x), float(lookat_y), float(lookat_z))

        # Bounds validation
        from ar2s.agents_sysid.skills.render_view import _validate_pose
        err = _validate_pose(pos, lookat, fovy_deg)
        if err:
            return [{"type": "text", "text":
                f"ERROR: invalid camera pose: {err}. Adjust and try again "
                f"(this attempt does NOT count toward the 3-render limit)."
            }]

        try:
            img = render_view_core(
                frame_cache, frame=int(frame),
                pos=pos, lookat=lookat, fovy_deg=float(fovy_deg),
                hide_non_pickup=bool(hide_non_pickup),
                pickup_object=pickup_object,
            )
        except Exception as e:
            return [{"type": "text", "text":
                f"ERROR: render failed: {type(e).__name__}: {e}"
            }]

        # Save + record
        state["custom_renders"] += 1
        idx = state["custom_renders"] - 1
        fname = f"custom_{idx:02d}_f{frame:04d}.jpg"
        path = save_dir / fname
        imageio.imwrite(str(path), img)
        pose = PoseRecord(
            kind="custom", cam_idx=None, hide_non_pickup=bool(hide_non_pickup),
            frame=int(frame), pos=pos, lookat=lookat,
            fovy_deg=float(fovy_deg), label=f"custom_{idx:02d}",
        )
        state["custom_poses"].append(pose)
        state["custom_paths"].append(path)

        b64 = _encode_image(path)
        remaining = state["max_custom"] - state["custom_renders"]
        frame_lbl = _frame_label(int(frame), frame_cache.engage_frame,
                                  frame_cache.frame_dt)
        caption = (
            f"[custom view {state['custom_renders']}/{state['max_custom']}, "
            f"remaining slots: {remaining}]\n"
            f"frame={frame} [{frame_lbl}]\n"
            f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) m, "
            f"lookat=({lookat[0]:+.3f},{lookat[1]:+.3f},{lookat[2]:+.3f}) m, "
            f"fovy={fovy_deg:.0f}°"
            + (", pickup-only (non-pickup objects hidden)" if hide_non_pickup else ", full scene")
        )
        return [
            {"type": "text", "text": caption},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]

    @tool
    def submit_sufficient(reasoning: str) -> str:
        """Declare the current image set sufficient for shift-decision; exit
        the view-sufficiency loop. Pass a 1-2 sentence reasoning.
        """
        state["submitted"] = True
        state["final_reasoning"] = str(reasoning)
        return (
            f"OK — view sufficiency confirmed with "
            f"{state['custom_renders']} custom render(s). "
            f"You may end the turn now."
        )

    return render_view, submit_sufficient, state


# -------------------------------------------------------------------
# Public entry
# -------------------------------------------------------------------

def collect_diagnostic_views(
    scene,
    frame_cache: FrameCache,
    save_dir: Path,
    *,
    recursion_limit: int = _RECURSION_LIMIT,
) -> ViewSufficiencyResult:
    """Run Agent A view-sufficiency loop at round 0.

    Args:
        scene: CalibratedScene.
        frame_cache: from probe_grasp (round 0 probe).
        save_dir: artifacts/<run-id>/rounds/00/images/ — baseline + custom
            images get written here.
        recursion_limit: max LLM ↔ tool round trips for Agent A.

    Returns:
        ViewSufficiencyResult with poses_chosen (4 baseline + 0-3 custom),
        image_paths (same order), terminated_by (submit/recursion_limit),
        and final_reasoning.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pickup_object = scene.bundle.pickup_object

    # ---- 1. Render baseline (orchestrator does this; Agent A starts with these in context)
    baseline_poses, baseline_paths = render_baseline_views(
        frame_cache, pickup_object, save_dir,
    )

    # ---- 2. Build agent
    render_view_tool, submit_tool, state = _make_tools(
        scene, frame_cache, pickup_object, save_dir,
    )
    llm = chat_model_for("grasp_optimization.subagents.grasp_loop_view_sufficiency")
    if llm is None:
        raise RuntimeError("No credential configured for the YAML-selected LLM provider.")

    sys_msg = SystemMessage(content=[{
        "type": "text", "text": _PROMPT_PATH.read_text(),
        "cache_control": {"type": "ephemeral"},
    }])
    agent = create_react_agent(
        model=llm,
        tools=[render_view_tool, submit_tool],
        prompt=sys_msg,
    )

    # ---- 3. Build initial human message and run
    initial_content = _build_initial_message(
        pickup_object, baseline_paths, baseline_poses,
        engage_frame=frame_cache.engage_frame,
        frame_dt=frame_cache.frame_dt,
        cam_ids=tuple(scene.bundle.cameras.camera_ids),
    )
    try:
        agent.invoke(
            {"messages": [HumanMessage(content=initial_content)]},
            config={"recursion_limit": recursion_limit},
        )
        terminated_by = "submit" if state["submitted"] else "recursion_limit"
    except Exception as e:
        # If recursion limit triggered an exception, treat as timeout.
        terminated_by = "recursion_limit"
        if not state["final_reasoning"]:
            state["final_reasoning"] = f"(recursion limit; last error: {e})"

    return ViewSufficiencyResult(
        poses_chosen=baseline_poses + state["custom_poses"],
        image_paths=baseline_paths + state["custom_paths"],
        n_custom_renders=state["custom_renders"],
        final_reasoning=state["final_reasoning"],
        terminated_by=terminated_by,
    )


__all__ = [
    "collect_diagnostic_views",
    "PoseRecord",
    "ViewSufficiencyResult",
    "render_baseline_views",
    "rerender_pose",
]
