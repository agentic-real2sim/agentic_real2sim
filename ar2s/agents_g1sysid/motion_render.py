"""Side-by-side motion-comparison videos for the G1 pipeline.

Kinematically replays two reference-format NPZ files (e.g. the retargeted
LAFAN1 reference vs a recorded closed-loop rollout) through the bundled G1
MJCF and writes one MP4 with the clips rendered side by side — the paper's
qualitative comparison artifact. Frames are rendered and written one at a
time, so memory stays flat for multi-minute clips.

Import only with a GL backend configured (the CLI sets MUJOCO_GL=egl).
"""
from __future__ import annotations

from pathlib import Path

import imageio
import mujoco
import numpy as np

from PIL import Image, ImageDraw

_ROBOT_XML = Path(__file__).resolve().parent / "assets" / "robot" / "g1_for_reward_inference.xml"

# Match env.py's follow camera so rollout videos and comparisons look alike.
_CAM_DISTANCE = 3.0
_CAM_ELEVATION = -15.0
_CAM_AZIMUTH = 180.0
_FRAME_W, _FRAME_H = 640, 480


class _Replay:
    """Per-frame kinematic replay of one reference-format NPZ."""

    def __init__(self, npz_path: str | Path, width: int, height: int):
        data = np.load(npz_path)
        self.root_pos = data["root_pos"]
        self.root_quat = data["root_quat"]
        self.dof_pos = data["dof_positions"]
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        self.n_frames = self.root_pos.shape[0]

        self.mjm = mujoco.MjModel.from_xml_path(str(_ROBOT_XML))
        self.mjd = mujoco.MjData(self.mjm)
        self.renderer = mujoco.Renderer(self.mjm, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = _CAM_DISTANCE
        self.camera.elevation = _CAM_ELEVATION
        self.camera.azimuth = _CAM_AZIMUTH
        self.pelvis = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def frame(self, i: int) -> np.ndarray:
        self.mjd.qpos[0:3] = self.root_pos[i]
        self.mjd.qpos[3:7] = self.root_quat[i]
        self.mjd.qpos[7:] = self.dof_pos[i]
        mujoco.mj_kinematics(self.mjm, self.mjd)
        self.camera.lookat[:] = self.mjd.xpos[self.pelvis]
        self.renderer.update_scene(self.mjd, camera=self.camera)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    """Top-left corner text label."""
    img = Image.fromarray(frame)
    ImageDraw.Draw(img).text((12, 10), text, fill=(255, 255, 255))
    return np.asarray(img)


def render_side_by_side(reference_npz: str | Path, rollout_npz: str | Path,
                        out_path: str | Path, *,
                        labels: tuple[str, str] = ("reference", "rollout"),
                        width: int = _FRAME_W, height: int = _FRAME_H,
                        ) -> dict:
    """Write a side-by-side MP4 of two reference-format NPZ motions.

    Both clips are truncated to the shorter one; fps is taken from the
    reference (the pipeline records rollouts at the same 50 fps).
    """
    ref = _Replay(reference_npz, width, height)
    try:
        roll = _Replay(rollout_npz, width, height)
    except Exception:
        ref.close()  # GL contexts are not reclaimed reliably by GC
        raise
    T = min(ref.n_frames, roll.n_frames)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=ref.fps)
    try:
        for i in range(T):
            writer.append_data(np.hstack([
                _label(ref.frame(i), labels[0]),
                _label(roll.frame(i), labels[1]),
            ]))
    finally:
        writer.close()
        ref.close()
        roll.close()
    return {"T": T, "fps": ref.fps, "out_path": str(out_path)}
