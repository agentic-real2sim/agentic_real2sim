#!/usr/bin/env bash
# Install the conda env for ar2s.agents_visual stereo stages.
#
# Wraps these ar2s.agents_visual._toolkit scripts under one env:
#   third_party/FoundationStereo/scripts/read_from_svo.py
#   third_party/FoundationStereo/scripts/extract_rectified_frames.py
#   third_party/FoundationStereo/scripts/run_stereo_on_frames.py
#   third_party/FoundationStereo/scripts/compute_mesh_scale.py
#
# Adapted from upstream third_party/FoundationStereo/install.sh, with these
# deviations (see also docs/agents_visual_plan_zh.html, Phase H9.3 row):
#   - Env prefixed `qianjun_` to avoid collision with other projects on this box.
#   - Skip flash-attn: a grep across the 4 scripts above shows none import it,
#     and source-building flash-attn for sm_120 (5070 Ti / Blackwell) is slow
#     (~20-30 min) and brittle. If we ever wire a droid script that DOES use
#     flash-attn, install it ad-hoc then.
#   - Add imageio-ffmpeg for MP4 reads in visual's split_rectified_mp4 helper.
#   - Add huggingface-hub (in environment.yml, missing from upstream install.sh).
#   - cu129 wheels so sm_120 PTX is present.
#
# Hard requirement: every step succeeds. Per project rule, on any failure we
# abort (`set -e`) — no silent fallback. If something breaks, inspect, fix,
# re-run.

# NOTE: We intentionally use `set -eo pipefail` (no `-u`). Conda's cuda-nvcc
# package ships an activate.d script that references NVCC_PREPEND_FLAGS
# without a `:-` default — under `set -u` it crashes on first activate. We
# don't use unset variables in this script ourselves, so the strictness loss
# is effectively zero. See discussion in chat history.
set -eo pipefail

ENV=qianjun_foundation_stereo
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/ar2s_conda_env.sh"

step() { echo; echo "==[$ENV]== $*"; }

ar2s_source_conda
ar2s_prepare_conda_env_root "$REPO_ROOT"
ENV_PREFIX="$(ar2s_conda_env_prefix "$REPO_ROOT" "$ENV")"

step "1/6  create conda env (python=3.11)  [idempotent]"
ar2s_create_prefix_env_or_skip "$ENV" "$ENV_PREFIX" "3.11"
conda activate "$ENV_PREFIX"

step "2/6  install torch 2.8 + torchvision (cu129 wheels, sm_120 compatible)"
pip install \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu129

step "3/6  install cuda-nvcc=12.9 from conda nvidia channel"
conda install -c nvidia cuda-nvcc=12.9 -y

step "4/6  install primary pip packages (upstream install.sh + huggingface-hub)"
pip install \
    scikit-image \
    omegaconf \
    opencv-contrib-python \
    imgaug \
    ninja \
    timm \
    albumentations \
    jupyterlab \
    scipy \
    joblib \
    scikit-learn \
    ruamel.yaml \
    trimesh \
    pyyaml \
    imageio \
    imageio-ffmpeg \
    open3d \
    transformations \
    einops \
    gdown \
    nodejs \
    huggingface-hub

step "5/6  install xformers==0.0.32.post2 (cu129 prebuilt; no source build)"
pip install \
    xformers==0.0.32.post2 \
    --index-url https://download.pytorch.org/whl/cu129

step "6/6  verify imports + GPU + dinov2 attention SDPA fallback (XFORMERS_DISABLED=1)"
# We force XFORMERS_DISABLED=1: xformers 0.0.32.post2 cu129 ships a Hopper
# (sm_90) flash-attn kernel that errors on sm_120 (Blackwell). dinov2 has a
# built-in fallback to torch's scaled_dot_product_attention which fully
# supports sm_120. ar2s/agents_visual/_toolkit/_runners.py auto-injects this
# env var when running stereo scripts, so verify must mirror that path.
export XFORMERS_DISABLED=1

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
python - <<PY
import os, sys
sys.path.insert(0, "${REPO_ROOT}/third_party/FoundationStereo")
import torch

assert os.environ.get("XFORMERS_DISABLED") == "1", "XFORMERS_DISABLED must be set"
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

# Basic GPU op
x = torch.randn(512, 512, device="cuda")
_ = (x @ x).sum().item()
print("basic matmul   = ok")

# dinov2 MemEffAttention with XFORMERS_DISABLED -> torch SDPA path
# Important: must set env BEFORE import (dinov2 reads it at import time).
from dinov2.dinov2.layers.attention import MemEffAttention
attn = MemEffAttention(dim=384, num_heads=6).cuda().eval()
x = torch.randn(2, 197, 384, device="cuda")  # B=2, N=197 tokens, D=384
with torch.no_grad():
    out = attn(x)
assert out.shape == x.shape, f"shape mismatch {out.shape} vs {x.shape}"
print(f"dinov2 attn fwd = ok, out shape {tuple(out.shape)} (via torch SDPA fallback)")

# FoundationStereo core import (no checkpoint load)
from core.foundation_stereo import FoundationStereo
print("FoundationStereo import = ok")

print()
print("===  sm_120 fallback path verified  ===")
PY

echo
echo "✅  $ENV install complete."
