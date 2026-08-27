"""SAM3D batch wrapper — one model load, many (image, mask) pairs.

Single-pass counterpart of the original ``run_sam3d_single.py``: loads the
sam-3d-objects ``Inference`` pipeline ONCE and reconstructs every object in a
jobs list, each with its own RGB frame + mask (per-object keyframe). This is
the SAM3D-side of the batched ``mesh_recover`` skill; the old single-input
script reloaded ~13 GB of weights per object.

Each job is ``{object_name, image_path, mask_path, output_glb_path}``. For each
job the script writes ``<output_glb_path>`` (mesh) and ``<output_glb_path>.json``
(success / runtime / mask_pixels sidecar), plus a ``.traceback`` on failure.
A missing/empty mask or a per-object inference error is recorded in that job's
sidecar with ``success=false`` and does not stop the remaining jobs.

Usage::

    python ar2s/agents_visual/_toolkit/scripts/run_sam3d_batch.py \\
        --jobs_json <path>  \\
        [--config <yaml>]    \\
        [--seed 42]          \\
        [--allow_cpu]

Run via the ``mesh_recover`` stage (env: ``qianjun_sam3d``).
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

# Resolve sam-3d-objects from env var, else the repo's bundled source tree.
# Script lives at ar2s/agents_visual/_toolkit/scripts/run_sam3d_batch.py, so the
# repo root is parents[4] (scripts -> _toolkit -> agents_visual -> ar2s -> droid_sim).
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
import torch  # noqa: E402

# sm_120 (Blackwell / 5070 Ti) workaround. The first torch.inverse() inside
# sam-3d-objects' preprocessor (pytorch3d Transform3d.inverse via cuSolver)
# crashes with CUSOLVER_STATUS_INTERNAL_ERROR at cusolverDnCreate(handle)
# AFTER the pipeline loads ~13 GB of weights on a 16 GB Blackwell card,
# even though a 4x4 inverse fits comfortably. Probe shows the same op runs
# fine via MAGMA in the exact same state, so we force MAGMA for linalg ops
# here. (The probe / decision lives in tmp/probe_cusolver.py.)
torch.backends.cuda.preferred_linalg_library("magma")


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 127


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAM3D on a batch of image+mask jobs.")
    p.add_argument("--jobs_json", required=True,
                   help='JSON list of {object_name, image_path, mask_path, output_glb_path}.')
    p.add_argument("--config", default="checkpoints/hf/pipeline.yaml",
                   help="SAM3D pipeline config (relative to sam-3d-objects root).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_cpu", action="store_true",
                   help="Allow inference without CUDA (default: fail fast).")
    return p.parse_args()


def _write_sidecar(glb_path: Path, payload: dict) -> None:
    glb_path.with_suffix(glb_path.suffix + ".json").write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; pass --allow_cpu to force CPU inference.")

    with open(args.jobs_json) as f:
        jobs = json.load(f)

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(_SAM3D_ROOT / config_path)
    print(f"[sam3d] loading pipeline from {config_path}")
    inference = Inference(config_path, compile=False)

    n_ok = 0
    for job in jobs:
        object_name = job.get("object_name", "object")
        image_path = Path(job["image_path"])
        mask_path = Path(job["mask_path"])
        glb_path = Path(job["output_glb_path"])
        glb_path.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"image not found: {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"mask not found: {mask_path}")

            image = np.array(Image.open(image_path))
            mask = load_binary_mask(mask_path)
            if mask.shape[:2] != image.shape[:2]:
                mask_u8 = Image.fromarray((mask.astype(np.uint8) * 255))
                mask_u8 = mask_u8.resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST)
                mask = np.array(mask_u8) > 127

            mask_pixels = int(mask.sum())
            if mask_pixels == 0:
                raise ValueError("mask is empty after thresholding")

            output = inference(load_image(str(image_path)), mask, seed=args.seed)
            glb = output.get("glb")
            if glb is None:
                raise RuntimeError("inference did not return a 'glb' field")
            glb.export(str(glb_path))

            runtime_sec = time.time() - start
            _write_sidecar(glb_path, {
                "success": True,
                "object_name": object_name,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "glb_path": str(glb_path),
                "image_size": [int(image.shape[1]), int(image.shape[0])],
                "mask_foreground_pixels": mask_pixels,
                "runtime_sec": runtime_sec,
                "seed": args.seed,
                "config": config_path,
            })
            n_ok += 1
            print(f"[sam3d] OK  {object_name}  glb={glb_path}  runtime={runtime_sec:.1f}s  mask_px={mask_pixels}")
        except Exception as exc:
            runtime_sec = time.time() - start
            _write_sidecar(glb_path, {
                "success": False,
                "object_name": object_name,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "runtime_sec": runtime_sec,
                "error": f"{type(exc).__name__}: {exc}",
            })
            glb_path.with_suffix(glb_path.suffix + ".traceback").write_text(traceback.format_exc())
            print(f"[sam3d] FAIL  {object_name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"[sam3d] {n_ok}/{len(jobs)} objects succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
