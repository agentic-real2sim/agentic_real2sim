"""SAM3D single-input wrapper (lives inside ar2s, not upstream sam-3d-objects).

Simplified counterpart of ``third_party/sam-3d-objects/run_outdoor_sam3d_batch.py``
— takes a single (image, mask) pair and writes one ``.glb`` plus a sidecar
metadata.json. Kept in this repo (not in the vendored sam-3d-objects tree) so
upstream stays unmodified.

The script depends on ``notebook.inference`` from sam-3d-objects; we inject
``third_party/sam-3d-objects/`` into sys.path so the import resolves. Run via
``ar2s.agents_visual._toolkit._runners.run_in_env('sam3d-objects', ...)``.

Usage::

    python ar2s/agents_visual/_toolkit/scripts/run_sam3d_single.py \\
        --image_path  <path>           \\
        --mask_path   <path>           \\
        --output_glb_path <path>       \\
        [--object_name <name>]         \\
        [--config <yaml>]              \\
        [--seed 42]                    \\
        [--allow_cpu]

Outputs at the given glb_path location:
    <output_glb_path>             — the reconstructed mesh
    <output_glb_path>.json        — success / runtime / mask_pixels metadata
    <output_glb_path>.traceback   — only on failure, full traceback
"""
import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


# Portable cache locations — these env vars must be set BEFORE importing
# notebook.inference (which imports matplotlib).
_TMP = tempfile.gettempdir()
os.environ.setdefault("MPLCONFIGDIR", f"{_TMP}/droid_visual/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", f"{_TMP}/droid_visual")
os.environ.setdefault("CONDA_PREFIX", sys.prefix)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Resolve sam-3d-objects from env var, else the repo's bundled source tree.
# Script lives at ar2s/agents_visual/_toolkit/scripts/run_sam3d_single.py, so
# the repo root is parents[4] (scripts -> _toolkit -> agents_visual -> ar2s -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SAM3D_ROOT = Path(
    os.environ.get("SAM3D_OBJECTS_ROOT")
    or _REPO_ROOT / "third_party" / "sam-3d-objects"
).resolve()
if not _SAM3D_ROOT.is_dir():
    raise FileNotFoundError(
        f"sam-3d-objects not found at {_SAM3D_ROOT}. Either run "
        f"the public installation script with the bundled third_party tree, "
        f"or set SAM3D_OBJECTS_ROOT to point at a local source tree."
    )
# Upstream sam-3d-objects README uses `sys.path.append("notebook")` and
# `from inference import Inference, load_image` — i.e. it puts the
# `notebook/` SUBDIR on sys.path, not the project root, and imports the
# module flat. That's because `notebook/` has no `__init__.py`, AND if the
# env has the Jupyter `notebook` package installed (kaolin pulls it in via
# ipycanvas / ipywidgets), `from notebook.inference import` would resolve
# to Jupyter's package and fail to find `.inference`. We follow upstream.
sys.path.insert(0, str(_SAM3D_ROOT / "notebook"))

from inference import Inference, load_image  # noqa: E402
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils  # noqa: E402
import torch  # noqa: E402

# sm_120 (Blackwell / 5070 Ti) workaround. The first torch.inverse() inside
# sam-3d-objects' preprocessor (pytorch3d Transform3d.inverse via cuSolver)
# crashes with CUSOLVER_STATUS_INTERNAL_ERROR at cusolverDnCreate(handle)
# AFTER the pipeline loads ~13 GB of weights on a 16 GB Blackwell card,
# even though a 4x4 inverse fits comfortably. Probe shows the same op runs
# fine via MAGMA in the exact same state, so we force MAGMA for linalg ops
# here. (The probe / decision is in tmp/probe_cusolver.py.)
torch.backends.cuda.preferred_linalg_library("magma")


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 127


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    pad_frac: float = 0.35,
    min_side: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("mask is empty after thresholding")

    h, w = mask.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    obj_w = x2 - x1
    obj_h = y2 - y1
    pad = max(16, int(max(obj_w, obj_h) * pad_frac))
    x1, x2 = max(0, x1 - pad), min(w, x2 + pad)
    y1, y2 = max(0, y1 - pad), min(h, y2 + pad)

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < min_side:
        extra = min_side - crop_w
        left = min(x1, extra // 2)
        right = min(w - x2, extra - left)
        left += min(x1 - left, extra - left - right)
        x1 -= left
        x2 += right
    if crop_h < min_side:
        extra = min_side - crop_h
        top = min(y1, extra // 2)
        bottom = min(h - y2, extra - top)
        top += min(y1 - top, extra - top - bottom)
        y1 -= top
        y2 += bottom

    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2], [x1, y1, x2, y2]


def install_mesh_only_export(inference: Inference) -> None:
    pipeline = inference._pipeline
    for name in ("slat_decoder_gs", "slat_decoder_gs_4"):
        if name in pipeline.models:
            del pipeline.models[name]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    orig_postprocess = pipeline.postprocess_slat_output

    def _postprocess(outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color):
        if "mesh" in outputs and "gaussian" not in outputs:
            glb = postprocessing_utils.to_glb(
                None,
                outputs["mesh"][0],
                simplify=0.95,
                texture_size=1024,
                verbose=False,
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=False,
                use_vertex_color=use_vertex_color,
                rendering_engine=pipeline.rendering_engine,
            )
            outputs["glb"] = glb
            return outputs
        return orig_postprocess(
            outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
        )

    pipeline.postprocess_slat_output = _postprocess


def run_inference(
    inference: Inference,
    image_path: Path,
    mask: np.ndarray,
    *,
    seed: int,
    mesh_only: bool,
) -> dict:
    if mesh_only:
        install_mesh_only_export(inference)
        image_rgba = inference.merge_mask_to_rgba(load_image(str(image_path)), mask)
        return inference._pipeline.run(
            image_rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=None,
            decode_formats=["mesh"],
        )
    return inference(load_image(str(image_path)), mask, seed=seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAM3D on a single image+mask pair.")
    p.add_argument("--image_path", required=True, help="RGB image PNG/JPG.")
    p.add_argument("--mask_path",  required=True, help="Binary mask PNG (white = object).")
    p.add_argument("--output_glb_path", required=True,
                   help="Destination .glb path. Parent dir is created.")
    p.add_argument("--object_name", default="object",
                   help="Logged in metadata.json for traceability.")
    p.add_argument("--config", default="checkpoints/hf/pipeline.yaml",
                   help="SAM3D pipeline config (relative to sam-3d-objects root).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_crop_to_mask", action="store_true",
                   help="Disable padded object crop before SAM3D inference.")
    p.add_argument("--full_decode", action="store_true",
                   help="Keep upstream Gaussian+mesh decode instead of mesh-only GLB export.")
    p.add_argument("--allow_cpu", action="store_true",
                   help="Allow inference without CUDA (default: fail fast).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    image_path = Path(args.image_path)
    mask_path  = Path(args.mask_path)
    glb_path   = Path(args.output_glb_path)
    meta_path  = glb_path.with_suffix(glb_path.suffix + ".json")

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"mask not found: {mask_path}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; pass --allow_cpu to force CPU inference.")

    glb_path.parent.mkdir(parents=True, exist_ok=True)

    # Config path resolves relative to sam-3d-objects root, matching the
    # upstream batch script's convention.
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(_SAM3D_ROOT / config_path)
    print(f"[sam3d] loading pipeline from {config_path}")
    inference = Inference(config_path, compile=False)

    start = time.time()
    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = load_binary_mask(mask_path)
        if mask.shape[:2] != image.shape[:2]:
            mask_u8 = Image.fromarray((mask.astype(np.uint8) * 255))
            mask_u8 = mask_u8.resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST)
            mask = np.array(mask_u8) > 127

        mask_pixels = int(mask.sum())
        if mask_pixels == 0:
            raise ValueError("mask is empty after thresholding")

        crop_bbox_xyxy = [0, 0, int(image.shape[1]), int(image.shape[0])]
        inference_image = image
        inference_mask = mask
        if not args.no_crop_to_mask:
            inference_image, inference_mask, crop_bbox_xyxy = crop_to_mask(image, mask)
            print(
                f"[sam3d] crop bbox={crop_bbox_xyxy} "
                f"size={inference_image.shape[1]}x{inference_image.shape[0]} "
                f"mask_px={int(inference_mask.sum())}"
            )

        with tempfile.TemporaryDirectory(prefix="sam3d_crop_") as td:
            inference_image_path = Path(td) / image_path.name
            Image.fromarray(inference_image).save(inference_image_path)
            output = run_inference(
                inference,
                inference_image_path,
                inference_mask,
                seed=args.seed,
                mesh_only=not args.full_decode,
            )
        glb = output.get("glb")
        if glb is None:
            raise RuntimeError("inference did not return a 'glb' field")
        glb.export(str(glb_path))

        runtime_sec = time.time() - start
        meta_path.write_text(json.dumps({
            "success": True,
            "object_name": args.object_name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "glb_path": str(glb_path),
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "inference_image_size": [int(inference_image.shape[1]), int(inference_image.shape[0])],
            "crop_bbox_xyxy": crop_bbox_xyxy,
            "mask_foreground_pixels": mask_pixels,
            "inference_mask_foreground_pixels": int(inference_mask.sum()),
            "runtime_sec": runtime_sec,
            "seed": args.seed,
            "config": config_path,
            "mesh_only_decode": not args.full_decode,
        }, indent=2))
        print(f"[sam3d] OK  glb={glb_path}  runtime={runtime_sec:.1f}s  mask_px={mask_pixels}")
        return 0

    except Exception as exc:
        runtime_sec = time.time() - start
        meta_path.write_text(json.dumps({
            "success": False,
            "object_name": args.object_name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "crop_to_mask": not args.no_crop_to_mask,
            "mesh_only_decode": not args.full_decode,
            "runtime_sec": runtime_sec,
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2))
        (glb_path.with_suffix(glb_path.suffix + ".traceback")).write_text(traceback.format_exc())
        print(f"[sam3d] FAIL  {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
