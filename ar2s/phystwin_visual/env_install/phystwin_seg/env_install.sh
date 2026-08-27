#!/usr/bin/env bash
# Conda env for PhysTwin video_segmentation stage.
# Requires CUDA 12.8 and conda on PATH.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ENV_NAME="${ENV_NAME:-phystwin_seg}"
PYTHON_VERSION="3.10"

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
source "${REPO_ROOT}/scripts/ar2s_conda_env.sh"

step() { echo; echo "============== $* =============="; date '+%T'; }

ar2s_source_conda
ar2s_prepare_conda_env_root "$REPO_ROOT"
ENV_PREFIX="$(ar2s_conda_env_prefix "$REPO_ROOT" "$ENV_NAME")"

step "create env $ENV_NAME"
ar2s_create_prefix_env_or_skip "$ENV_NAME" "$ENV_PREFIX" "$PYTHON_VERSION"
conda activate "$ENV_PREFIX"

# CUDA
step "CUDA 12.8 runtime/toolchain"
conda install -c nvidia \
    cuda-nvcc=12.8 cuda-cccl=12.8 cuda-cudart=12.8 cuda-cudart-dev=12.8 \
    cuda-libraries-dev=12.8 ninja \
    -y

export CUDA_HOME="${CONDA_PREFIX}"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVCC_PREPEND_FLAGS=""
export NVCC_APPEND_FLAGS=""
PHYSTWIN_CUDA_ARCH_LIST="${PHYSTWIN_CUDA_ARCH_LIST:-8.9;12.0}"
export CC="${PHYSTWIN_CC:-/usr/bin/gcc}"
export CXX="${PHYSTWIN_CXX:-/usr/bin/g++}"
export CUDAHOSTCXX="${PHYSTWIN_CUDAHOSTCXX:-$CXX}"

pip install --upgrade pip setuptools wheel

# Torch stack (cu128)
pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Core deps. Keep SciPy on conda here; newer pip SciPy wheels can be incompatible
# with the NumPy 1.26 ABI expected by these legacy vision packages.
conda install -y numpy==1.26.4 scipy==1.11.4 opencv
pip install --no-cache-dir matplotlib pillow supervision

# GroundingDINO is served from HuggingFace (IDEA-Research/grounding-dino-tiny
# via transformers) — no local CUDA build needed. Pre-fetch it into the HF
# cache here so segmentation also works offline (HF_HUB_OFFLINE=1) later.
pip install --no-cache-dir transformers
python - <<'PY'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
print("grounding-dino-tiny cached")
PY

# Grounded-SAM-2 from the bundled third_party source tree.
pip install --no-cache-dir --no-build-isolation -e "${REPO_ROOT}/third_party/Grounded-SAM-2_phystwin"

# Register repo source so ar2s.phystwin_visual is importable in this Python 3.10 env.
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$REPO_ROOT" > "$SITE_PACKAGES/ar2s_repo_root.pth"

python - <<'PY'
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2, build_sam2_video_predictor
print(f"torch = {torch.__version__}, cuda? {torch.cuda.is_available()}")
print("phystwin_seg imports OK")
PY

echo "phystwin_seg env ready."
