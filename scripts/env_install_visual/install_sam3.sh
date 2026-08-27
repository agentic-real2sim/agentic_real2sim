#!/usr/bin/env bash
# Install qianjun_sam3 env (Stage 4: text-prompted segmentation via SAM 3).
#
# Based on third_party/sam3/README.md, with these deviations (mirroring the
# H9.3 / foundation_stereo decisions; see docs/agents_visual_plan_zh.html):
#   - Env prefixed `qianjun_` to avoid collisions on shared workstations.
#   - Skip `flash-attn-3`: README marks as "optional for faster inference";
#     ships a Hopper sm_90 kernel that risks the same sm_120 invalid-arg
#     crash we hit in foundation_stereo. torch SDPA fallback is fast enough
#     for stereo so we expect it to also work for sam3.
#   - Skip `cc_torch`: same "optional for faster inference" tier, plus it's a
#     git+pip source build (needs nvcc + torch import-during-build) — the
#     highest-risk shape on sm_120 right now.
#   - Add opencv-python + tqdm: the repo SAM3 wrappers import both at module
#     load, so they must be explicit env deps rather than transitive luck.
#
# Strict mode: `set -eo pipefail` (no `-u`, same as install_foundation_stereo.sh
# — conda activate.d scripts in cu* packages reference unbound vars and crash
# under `set -u`).

set -eo pipefail

ENV=qianjun_sam3
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/ar2s_conda_env.sh"

step() { echo; echo "==[$ENV]== $*"; }

ar2s_source_conda
ar2s_prepare_conda_env_root "$REPO_ROOT"
ENV_PREFIX="$(ar2s_conda_env_prefix "$REPO_ROOT" "$ENV")"

step "1/4  create conda env (python=3.12)  [idempotent]"
ar2s_create_prefix_env_or_skip "$ENV" "$ENV_PREFIX" "3.12"

conda activate "$ENV_PREFIX"

step "2/4  install torch 2.10.0 + torchvision (cu128 wheels, sm_120 compatible)"
pip install \
    torch==2.10.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

step "3/4  install sam3 with [notebooks] extras (one-shot — README's path)"
# Three python-3.12 / numpy-ABI / under-specified-deps gotchas surfaced
# during iterative install. After 4 missing-import iterations we
# stopped cherry-picking deps and switched to sam3 README's official
# recommendation (`pip install -e .[notebooks]`) — it covers all the
# core runtime deps the pyproject mis-categorised plus a small jupyter
# tax we accept (~200 MB) for reliability.
#
#   (a) sam3.model_builder imports pkg_resources. setuptools>=81 stopped
#       shipping pkg_resources (deprecation finalised). torch's transitive
#       pull brings setuptools 82.0.1 which has no pkg_resources. Pin
#       `setuptools<81` (80.x is the last line still shipping pkg_resources)
#       to keep the import path alive without patching upstream sam3.
#
#   (b) sam3's pyproject.toml pins numpy<2. opencv-python>=4.11 forces
#       numpy>=2, which produces a pip dep-resolver warning and risks a
#       numpy-2-ABI runtime crash inside sam3. Pin opencv-python<4.11
#       (last line that supports numpy<2; python 3.12 compatible) and
#       numpy<2 in the same pip call so the resolver picks consistent
#       versions — they over-ride the unpinned opencv-python in [notebooks].
#
#   (c) [notebooks] still doesn't cover psutil (sam3_video_predictor.py
#       imports it at module load). Install it explicitly alongside tqdm,
#       which our wrapper imports at module load. Skip [train] extras
#       (submitit / torchmetrics / tensorboard / fvcore) — those are
#       training-only.
pip install 'setuptools<81'
pip install -e "${REPO_ROOT}/third_party/sam3[notebooks]" \
    'opencv-python<4.11' 'numpy<2'
pip install psutil tqdm

step "4/4  verify imports + GPU compute capability"
cd "$REPO_ROOT"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("WARN: no CUDA device visible — skipping GPU + package verification.")
    print("      Common case: docker build (no GPU access at build time).")
    print("      Re-test inside container via: docker run --gpus all ... python -c 'import torch; assert torch.cuda.is_available()'")
    raise SystemExit(0)
cc = torch.cuda.get_device_capability(0)
print(f"torch          = {torch.__version__}")
print(f"torch.cuda     = {torch.version.cuda}")
print(f"device         = {torch.cuda.get_device_name(0)}")
print(f"compute cap    = sm_{cc[0]}{cc[1]}")

# Basic GPU op (does sm_120 actually work for torch ops in this env)
x = torch.randn(512, 512, device="cuda")
_ = (x @ x).sum().item()
print("basic matmul   = ok")

# sam3 package + the actual entrypoint used by batch_video_segment.py.
# We DON'T instantiate the predictor here (that pulls multi-GB weights from
# huggingface_hub); we just confirm the import surface is wired correctly.
import sam3
print(f"sam3           = {getattr(sam3, '__version__', 'unknown')}")
from sam3.model_builder import build_sam3_video_predictor
print("sam3.model_builder.build_sam3_video_predictor = importable")

import cv2
print(f"cv2            = {cv2.__version__}")
import tqdm
print(f"tqdm           = {tqdm.__version__}")
import os, sys
sys.path.insert(0, os.getcwd())
from ar2s.agents_visual._toolkit.scripts.run_sam3_segment import require_torch_and_predictor
require_torch_and_predictor()
print("run_sam3_segment imports = ok")
PY

echo
echo "✅  $ENV install complete."
