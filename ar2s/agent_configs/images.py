"""Image helpers for VLM input plumbing.

Shared utility used by any agent that needs to pass an on-disk image into a
VLM call. Lives next to ``vlm_call_first_success`` because the data URL is
the canonical way to attach images to a structured chat message.
"""
from __future__ import annotations

import base64
from pathlib import Path


def image_to_data_url(
    image_path: str | Path,
    *,
    max_dim: int | None = 1024,
    jpeg_quality: int = 85,
) -> str:
    """Read an image off disk and return a base64 data URL ready for VLM input.

    Downsamples to ``max_dim`` on the longer side (default 1024 px) and
    re-encodes as JPEG. Verbose models (Qwen3-VL, gemma-3-27b) hit max_tokens
    on full-res PNG inputs because image token cost dominates the budget;
    downsampling to ≤1024 keeps the prompt under ~500 image tokens and lets
    structured-output JSON fit comfortably.

    Set ``max_dim=None`` to disable resizing (use the raw bytes verbatim;
    the original extension is preserved as the MIME hint).
    """
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")

    if max_dim is None:
        ext = p.suffix.lower().lstrip(".") or "png"
        mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    import io
    from PIL import Image
    img = Image.open(p).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"
