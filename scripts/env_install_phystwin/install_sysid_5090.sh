#!/usr/bin/env bash
#
# Minimal phystwin env install for Stage 1 (CMA) + Stage 2 (Warp training)
# of ar2s.phystwin_sysid. Skips visual processing tools (TRELLIS, Grounded-SAM-2,
# SDXL, RealSense) — our data is already pre-processed.
#
# Target: Blackwell GPUs (RTX 5070 Ti / 5080 / 5090 = sm_120) + CUDA 12.8 toolkit.
# Tested 2026-05-17 on RTX 5070 Ti.
#
# Differences vs upstream env_install.sh / 5090_env_install.sh:
#   - Uses pip opencv-python (not conda opencv) — conda's needs newer
#     CXXABI_1.3.15 which system libstdc++ doesn't have.
#   - Adds plyfile + einops (missing from upstream scripts but needed by
#     third_party/gaussian-splatting/).
#   - Compiles CUDA extensions with TORCH_CUDA_ARCH_LIST covering local Ada and
#     Blackwell by default, and --no-build-isolation (so they see the env's torch).
#   - Uses third_party/gaussian-splatting/ directly via a .pth file and an
#     env-local gaussian_splatting namespace loader so upstream imports keep
#     working even though the checkout lives under a hyphenated directory.
#   - Builds diff-gaussian-rasterization from its accelerated 3dgs_accel branch
#     because current gaussian-splatting imports SparseGaussianAdam at top level.
#   - Uses third_party/pytorch3d_phystwin/ (Eric's fork, has compat patches
#     for CUDA 12.8 / sm_120) — built from source. Upstream pytorch3d has no
#     wheel for cu128 + pytorch 2.11.
#
# Usage:
#   cd /path/to/droid_sim
#   bash scripts/env_install_phystwin/install_sysid_5090.sh

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-phystwin}"
GS_ROOT="$REPO_ROOT/third_party/gaussian-splatting"
DGR_ROOT="$GS_ROOT/submodules/diff-gaussian-rasterization"
PYTORCH3D_ROOT="$REPO_ROOT/third_party/pytorch3d_phystwin"

unset LD_LIBRARY_PATH
# Derive AR2S_CONDA_PATH from the repo-local .conda (symlink or, on
# cluster filesystems, a real directory). Guard readlink: a dangling
# symlink must fail loudly, not die silently under set -e.
if [ -z "${AR2S_CONDA_PATH:-}" ] && [ -L "$REPO_ROOT/.conda" ]; then
    AR2S_CONDA_PATH="$(readlink -f "$REPO_ROOT/.conda")" || {
        echo "ERROR: $REPO_ROOT/.conda is a dangling symlink" >&2; exit 1; }
    export AR2S_CONDA_PATH
elif [ -z "${AR2S_CONDA_PATH:-}" ] && [ -d "$REPO_ROOT/.conda" ]; then
    AR2S_CONDA_PATH="$REPO_ROOT/.conda"
    export AR2S_CONDA_PATH
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/ar2s_conda_env.sh"

step() { echo; echo "============== $* =============="; date '+%T'; }

ar2s_source_conda
ar2s_prepare_conda_env_root "$REPO_ROOT"
ENV_PREFIX="$(ar2s_conda_env_prefix "$REPO_ROOT" "$ENV_NAME")"

step "create env $ENV_NAME"
ar2s_create_prefix_env_or_skip "$ENV_NAME" "$ENV_PREFIX" "3.10"
conda activate "$ENV_PREFIX"
python --version

step "PyTorch 2.11 + cu128 (Blackwell-compatible)"
pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

step "CUDA 12.8 toolchain for source builds"
conda install -c nvidia \
    cuda-nvcc=12.8 cuda-cccl=12.8 cuda-cudart=12.8 cuda-cudart-dev=12.8 \
    cuda-libraries-dev=12.8 ninja \
    -y

export NVCC_PREPEND_FLAGS=""
export NVCC_APPEND_FLAGS=""
PHYSTWIN_CUDA_ARCH_LIST="${PHYSTWIN_CUDA_ARCH_LIST:-8.9;12.0}"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="${PHYSTWIN_CC:-/usr/bin/gcc}"
export CXX="${PHYSTWIN_CXX:-/usr/bin/g++}"
export CUDAHOSTCXX="${PHYSTWIN_CUDAHOSTCXX:-$CXX}"

step "Physics + optimization (warp + cma)"
pip install --no-cache-dir warp-lang cma

step "Numerical / data utilities"
pip install --no-cache-dir numpy==1.26.4 scipy scikit-learn kornia tqdm \
    termcolor fvcore stannum ipdb

step "Visualization + I/O"
pip install --no-cache-dir open3d trimesh rtree pyrender matplotlib imageio \
    moviepy "pyglet<2" usd-core opencv-python==4.10.0.84

step "Re-pin numpy after visualization deps"
pip install --no-cache-dir numpy==1.26.4

step "Gaussian Splatting helpers (plyfile, einops, gsplat)"
pip install --no-cache-dir plyfile==1.0.3 einops gsplat==1.4.0

step "Re-pin numpy before CUDA extension builds"
pip install --no-cache-dir numpy==1.26.4

step "Optional wandb"
pip install --no-cache-dir wandb

step "Keyboard input dependency (pynput)"
pip install --no-cache-dir pynput

step "Use vendored Gaussian Splatting sources"
cd "$REPO_ROOT"
# The source tree, its CUDA extensions, and the nested GLM headers are shipped
# in this repository. Do not initialize or update Git metadata here: a release
# copy is intentionally usable without access to any upstream repository.
[ -d "$GS_ROOT/submodules/simple-knn" ] || {
    echo "ERROR: vendored simple-knn is missing at $GS_ROOT/submodules/simple-knn" >&2
    exit 1
}
[ -d "$DGR_ROOT/third_party/glm" ] || {
    echo "ERROR: vendored GLM is missing at $DGR_ROOT/third_party/glm" >&2
    exit 1
}

step "Register third_party/ and gaussian-splatting upstream namespace loader"
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
PTH="$SITE_PACKAGES/ar2s_third_party.pth"
{
    echo "$REPO_ROOT"
    echo "$REPO_ROOT/third_party"
    echo "$GS_ROOT"
} > "$PTH"
export SITE_PACKAGES GS_ROOT
python <<'PY'
import os
from pathlib import Path

shim = Path(os.environ["SITE_PACKAGES"]) / "gaussian_splatting"
shim.mkdir(exist_ok=True)
# The namespace package points at GS_ROOT only; the direct GS_ROOT .pth entry
# separately keeps the upstream modules' root-level imports working.
(shim / "__init__.py").write_text(
    '"""Upstream Gaussian Splatting namespace loader for phystwin installs."""\n'
    f'__path__ = [{os.environ["GS_ROOT"]!r}]\n',
    encoding="utf-8",
)
PY
echo "wrote $PTH and gaussian_splatting namespace loader"

step "Compile simple_knn CUDA extension (sm_120)"
TORCH_CUDA_ARCH_LIST="$PHYSTWIN_CUDA_ARCH_LIST" pip install --no-cache-dir --no-build-isolation \
    "$GS_ROOT/submodules/simple-knn"

HDR="$DGR_ROOT/cuda_rasterizer/rasterizer_impl.h"
# fixes #include <cstdint> bug for gcc 13+ (idempotent: guard against a
# rerun stacking another include)
grep -q '#include <cstdint>' "$HDR" || \
    sed -i '0,/#pragma once/{s/#pragma once/#pragma once\n#include <cstdint>/}' "$HDR"

step "Compile diff-gaussian-rasterization (sm_120)"
TORCH_CUDA_ARCH_LIST="$PHYSTWIN_CUDA_ARCH_LIST" pip install --no-cache-dir --no-build-isolation \
    "$DGR_ROOT"

step "Compile vendored pytorch3d (sm_120) — slowest step, ~5-15 min"
cd "$REPO_ROOT"
TORCH_CUDA_ARCH_LIST="$PHYSTWIN_CUDA_ARCH_LIST" pip install --no-cache-dir --no-build-isolation \
    "$PYTORCH3D_ROOT"

step "VERIFY full ar2s.phystwin_sysid import chain"
cd "$REPO_ROOT"
python <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import torch
print(f"torch    = {torch.__version__}, cuda? {torch.cuda.is_available()}")
print(f"device   = {torch.cuda.get_device_name(0)}, cc={torch.cuda.get_device_capability(0)}")

import warp, cma, kornia, scipy, sklearn, open3d, trimesh, pyrender, cv2, imageio
import ipdb
import wandb, pynput
print(f"cv2      = {cv2.__version__}")
import pytorch3d
print(f"pytorch3d = {pytorch3d.__version__}")
import gaussian_splatting
assert list(gaussian_splatting.__path__) == [os.environ["GS_ROOT"]]
import gaussian_splatting.scene
import gaussian_splatting.gaussian_renderer
import gaussian_splatting.utils
import gaussian_splatting.arguments
from gaussian_splatting.scene import Scene
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getWorld2View2
from gaussian_splatting.arguments import ModelParams
from ar2s.phystwin_sysid.gaussian_utils.dynamic import (
    interpolate_motions_speedup,
    knn_weights,
    knn_weights_sparse,
    get_topk_indices,
    calc_weights_vals_from_indices,
)
from ar2s.phystwin_sysid.gaussian_utils.rotation import quaternion_multiply, matrix_to_quaternion
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizationSettings, SparseGaussianAdam
print("CUDA extensions OK")

from ar2s.phystwin_sysid import (
    PhysTwinConfig, PhysTwinRealData, PhysTwinSyntheticData,
    SpringMassSystemWarp, OptimizerCMA, InvPhyTrainerWarp,
)
print()
print("=" * 50)
print("phystwin env install COMPLETE")
print("=" * 50)
PY
