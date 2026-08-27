"""Visual inspection skills for the Episode Agent.

These tools return multimodal content (text + base64 image blocks) so the
LLM can actually SEE the file content in its next turn, not just read text
descriptions of it.

Use cases:
  - view_image("first_frame.jpg") to ground task semantics in scene reality
  - view_video_keyframes("demo.mp4") to see the action / robot behavior

Both tools downscale images to ≤ 1024px on the longest edge to keep token
budgets reasonable (Claude Opus charges ~1500 tokens per image at default).
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from langchain_core.tools import tool


_MAX_DIM = 1024
_JPEG_QUALITY = 85


def _resolve(path: str) -> Path:
    return Path(path).expanduser()


def _encode_pil_image(img) -> str:
    """Resize + encode a PIL image as base64 JPEG."""
    if max(img.size) > _MAX_DIM:
        img.thumbnail((_MAX_DIM, _MAX_DIM))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _content_block_text(text: str) -> dict:
    return {"type": "text", "text": text}


def _content_block_image(b64_jpeg: str) -> dict:
    """Return a LangChain-compatible image content block (data: URL form)."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
    }


# -------------------------------------------------------------------
# view_image — show a single image to the LLM
# -------------------------------------------------------------------

@tool(response_format="content")
def view_image(path: str) -> list[dict]:
    """Show an image (jpg / png) to the agent so it can visually inspect it.

    Useful for:
      - first_frame images (to confirm scene composition)
      - mask / segmentation files
      - any rendered keyframe

    Returns a multimodal content block (text caption + image) that the LLM
    can see in its next turn.
    """
    p = _resolve(path)
    if not p.exists() or not p.is_file():
        return [_content_block_text(f"ERROR: not a file: {p}")]
    try:
        img = Image.open(p)
        w, h = img.size
        b64 = _encode_pil_image(img)
        return [
            _content_block_text(f"Image at {p} (orig {w}x{h}, downscaled to ≤{_MAX_DIM}px):"),
            _content_block_image(b64),
        ]
    except Exception as e:
        return [_content_block_text(f"ERROR loading {p}: {type(e).__name__}: {e}")]


# -------------------------------------------------------------------
# view_video_keyframes — extract N evenly-spaced frames
# -------------------------------------------------------------------

@tool(response_format="content")
def view_video_keyframes(path: str, n_frames: int = 6) -> list[dict]:
    """Extract N evenly-spaced keyframes from a video file and show them
    to the agent in temporal order.

    Use this to understand what action is performed in the video:
      - what objects are in the scene
      - what the robot does (grasps / pours / pushes / ...)
      - the start vs end state

    Args:
        path: video file (mp4 / mov / avi — anything imageio+ffmpeg reads)
        n_frames: how many keyframes to extract (default 6, max 8;
            higher costs more tokens)
    """
    p = _resolve(path)
    if not p.exists() or not p.is_file():
        return [_content_block_text(f"ERROR: not a file: {p}")]
    n_frames = min(max(n_frames, 1), 8)

    try:
        # Step 1: figure out total frame count cheaply.
        # Try improps first (fast, only reads headers); fall back to iterate
        # if the backend doesn't expose n_images.
        total: int | None = None
        try:
            props = iio.improps(str(p), plugin="pyav")
            shape = getattr(props, "shape", None)
            if shape and len(shape) >= 1:
                total = int(shape[0])
        except Exception:
            pass
        if total is None:
            total = sum(1 for _ in iio.imiter(str(p)))   # one-pass count

        if total <= 0:
            return [_content_block_text(f"ERROR: video has no frames: {p}")]

        # Step 2: pick N evenly-spaced indices.
        if total <= n_frames:
            indices = list(range(total))
        else:
            indices = sorted(set(np.linspace(0, total - 1, n_frames,
                                              dtype=int).tolist()))

        # Step 3: stream — iterate once and pick only target indices.
        # Memory stays O(n_frames) instead of O(total).
        wanted = set(indices)
        kept: dict[int, "Image.Image"] = {}
        for i, frame in enumerate(iio.imiter(str(p))):
            if i in wanted:
                kept[i] = Image.fromarray(frame)
                if len(kept) == len(wanted):
                    break

        # Step 4: build multimodal content blocks in order.
        content: list[dict] = [_content_block_text(
            f"{len(indices)} keyframes from {p} "
            f"(total {total} frames, indices {indices}).\n\n"
            "IMPORTANT — how to read these keyframes:\n"
            "1. Each keyframe MAY be a multi-view COMPOSITE (e.g., 2-4 camera "
            "angles stitched horizontally into one wide image, all captured at "
            "the same instant in time). Look at every panel, especially when "
            "objects are small or partly occluded.\n"
            "2. The keyframes are presented in TEMPORAL ORDER. The first "
            "keyframe is the START of the action; the last is the END. "
            "To determine the action, compare the start state with the end "
            "state and ask: which object moved, and from where to where?"
        )]
        for i, idx in enumerate(indices):
            img = kept[idx]
            b64 = _encode_pil_image(img)
            content.append(_content_block_text(
                f"Keyframe {i + 1}/{len(indices)} (frame_idx={idx}):"
            ))
            content.append(_content_block_image(b64))
        return content
    except Exception as e:
        return [_content_block_text(
            f"ERROR reading video {p}: {type(e).__name__}: {e}"
        )]


__all__ = ["view_image", "view_video_keyframes"]
