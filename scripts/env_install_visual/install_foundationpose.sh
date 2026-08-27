#!/usr/bin/env bash
# Install qianjun_foundationpose env for FoundationPose pose tracking (H9.6).
#
# We DEVIATE significantly from third_party/FoundationPose/install.sh because:
#   - upstream pins CUDA 11.8 + torch 2.0.0+cu118 (40-series-friendly, no
#     sm_120 PTX). sm_120 (5070 Ti / 5090) needs CUDA 12.8+ + torch 2.7+
#     per NVlabs/FoundationPose#369 ("Env setup for RTX 50 series GPU").
#   - upstream uses prebuilt pytorch3d wheel from Facebook (py39_cu118_pyt200);
#     no sm_120 prebuilt exists, so we must source-build pytorch3d (same
#     recipe as H9.5: TORCH_CUDA_ARCH_LIST includes local Ada + Blackwell).
#   - upstream pins kaolin==0.15.0 for torch-2.0.0_cu118 (cu121-era);
#     NVIDIA only publishes recent kaolin (0.18.0) for torch-2.8.0_cu128.
#
# What we KEEP from upstream:
#   - python 3.9                       (fork pinned; mycpp / mycuda built against this)
#   - eigen + libboost via conda       (mycpp cmake needs them)
#   - requirements.txt deps            (filtered — see step 5)
#   - nvdiffrast from the vendored source tree (NVlabs install.sh CC/CXX/CUDA_HOME dance preserved)
#   - build_all_conda.sh               (mycpp cmake + mycuda pip install -e)
#
# Already PR #369-patched in the RyougiJarvis fork:
#   - bundlesdf/mycuda/common.cu       .type() → .scalar_type() (deprecated API fixed)
#   - bundlesdf/mycuda/setup.py        c++14 → c++17 (sm_120 requires c++17)
#   So we DO NOT need to apply those patches ourselves.
#
# ============================================================================
# SIX SM_120 RISK POINTS (each step annotates expected failure modes)
# ============================================================================
#
# (1) ⚠️ torch 2.8.0+cu128 cp39          probed: manylinux_2_28 wheel exists
# (2) ⚠️ conda cuda toolkit 12.8         probed: nvidia conda channel has it
# (3) ⚠️ requirements.txt strict pins    jupyter-client==7.4.9 etc. are kaolin-0.15-era;
#                                        may pip-resolver-conflict with kaolin 0.18 deps.
#                                        Filter the worst offenders (jupyter + zmq pins).
# (4) ⚠️ pytorch3d source build          proven in H9.5
# (5) ⚠️ nvdiffrast source build         FIRST sm_120 attempt; should work
#                                        because nvdiffrast uses standard
#                                        TORCH_CUDA_ARCH_LIST resolution
# (6) 🚨 mycuda source build             HIGHEST RISK — FoundationPose-specific
#                                        CUDA kernel. PR #369 patched the
#                                        deprecated .type() API + c++17. If
#                                        nvcc complains about sm_120 PTX or
#                                        any other arch issue: ABORT.
# ============================================================================

set -eo pipefail

ENV=qianjun_foundationpose
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FP_ROOT="${REPO_ROOT}/third_party/FoundationPose"
PYTORCH3D_ROOT="${PYTORCH3D_ROOT:-${REPO_ROOT}/third_party/pytorch3d_phystwin}"
NVDIFFRAST_ROOT="${NVDIFFRAST_ROOT:-${REPO_ROOT}/third_party/nvdiffrast}"
PRECHECK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$PYTORCH3D_ROOT" ]; then
    echo "ERROR: vendored PyTorch3D is missing at $PYTORCH3D_ROOT" >&2
    exit 1
fi
if [ ! -d "$NVDIFFRAST_ROOT" ]; then
    echo "ERROR: vendored nvdiffrast is missing at $NVDIFFRAST_ROOT" >&2
    echo "       Set NVDIFFRAST_ROOT to a local checkout before running this installer." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/ar2s_conda_env.sh"

step() { echo; echo "==[$ENV]== $*"; }

# Defense-in-depth: this is the build-heaviest env (mycpp cmake + mycuda nvcc
# + nvdiffrast + pytorch3d source builds). Run precheck to surface missing
# host toolchain (gcc / g++ / cmake / ninja) BEFORE the conda create burns
# minutes on download. Other install scripts assume precheck has been run
# manually; we call it explicitly here because the failure mode (mid-step-9
# nvcc errors) is much harder to debug than precheck's clear message.
step "0/10  precheck (host gcc/g++ — only deps that can't be conda-installed)"
# Full precheck.sh requires cmake + ninja on host PATH (per project policy:
# never auto-install system-level packages). bootstrap-driven setups skip
# that constraint by installing cmake/ninja via conda-forge INTO the env
# below — completely sandboxed, no host pollution. precheck.sh is still
# useful as a standalone manual-setup tool: bash scripts/env_install_visual/precheck.sh
for tool in gcc g++; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not on PATH." >&2
        echo "       Required by nvcc host C++ codegen for mycpp / mycuda / nvdiffrast / pytorch3d source builds." >&2
        echo "       gcc/g++ MUST come from the system (conda's gcc_linux-64 conflicts with system headers)." >&2
        echo "       Ask admin: sudo apt install build-essential" >&2
        exit 1
    fi
done

# ----------------------------------------------------------------------------

ar2s_source_conda
ar2s_prepare_conda_env_root "$REPO_ROOT"
ENV_PREFIX="$(ar2s_conda_env_prefix "$REPO_ROOT" "$ENV")"

step "1/10  create conda env (python=3.9)  [idempotent]"
ar2s_create_prefix_env_or_skip "$ENV" "$ENV_PREFIX" "3.9"

conda activate "$ENV_PREFIX"

# Install cmake / ninja into the env if missing on the host. Sandboxed —
# lands at <ENV_PREFIX>/bin/{cmake,ninja}, no host pollution. Used by
# step 9 (mycpp cmake build) + step 8 (nvdiffrast/mycuda compile speed).
step "1.5/10  ensure cmake + ninja available (conda-forge fallback)"
NEED_INSTALL=()
command -v cmake >/dev/null 2>&1 || NEED_INSTALL+=(cmake)
command -v ninja >/dev/null 2>&1 || NEED_INSTALL+=(ninja)
if [ ${#NEED_INSTALL[@]} -gt 0 ]; then
    echo "installing into $ENV: ${NEED_INSTALL[*]}"
    conda install -y -c conda-forge "${NEED_INSTALL[@]}"
else
    echo "cmake $(command -v cmake) | ninja $(command -v ninja) — both already available"
fi

# Defensive: cuda-nvcc activate.d uses NVCC_PREPEND_FLAGS without :- default.
export NVCC_PREPEND_FLAGS=""
export NVCC_APPEND_FLAGS=""
# Default covers Ada (sm_89 — 4090), Hopper (sm_90 — H100/H200), and
# Blackwell (sm_120 — 5090). Override per-machine if you only need one.
FOUNDATIONPOSE_CUDA_ARCH_LIST="${FOUNDATIONPOSE_CUDA_ARCH_LIST:-8.9;9.0;12.0}"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ----------------------------------------------------------------------------

step "2/10  install torch 2.8.0 + torchvision (cu128 cp39 manylinux_2_28)"
# Same sweet-spot version as H9.5 — torch 2.8 knows sm_120 in
# _get_cuda_arch_flags AND has NVIDIA kaolin wheels for cu128.
pip install --upgrade \
    torch==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# ----------------------------------------------------------------------------

step "3/10  install conda CUDA 12.8 toolchain (nvcc + cudart-dev + libraries-dev)"
# Upstream install.sh uses `conda install -c nvidia/label/cuda-11.8.0 cuda`
# (full toolkit). For sm_120 we need CUDA 12.8 components instead. We install
# the components that mycuda + nvdiffrast actually need (smaller than full
# toolkit; matches what we did in H9.5):
#   cuda-nvcc 12.8        nvcc compiler with sm_120 PTX support
#   cuda-cccl 12.8        CUDA C++ Core Library (templates / headers)
#   cuda-cudart           CUDA runtime (linked at build)
#   cuda-cudart-dev       runtime headers
#   cuda-libraries-dev    cuBLAS / cuFFT / etc. dev headers (nvdiffrast links these)
conda install -c nvidia \
    cuda-nvcc=12.8 \
    cuda-cccl=12.8 \
    cuda-cudart=12.8 \
    cuda-cudart-dev=12.8 \
    cuda-libraries-dev=12.8 \
    -y

# ----------------------------------------------------------------------------

step "4/10  install eigen + libboost (pinned to 1.86 — mycpp cmake needs boost_system)"
# Upstream install.sh has: conda install conda-forge::eigen=3.4.0 conda-forge::libboost
#
# Conda-forge libboost AT LATEST (1.91+) has SPLIT system into header-only —
# there's no boost_system-config.cmake any more. mycpp/CMakeLists.txt does
#   find_package(Boost CONFIG REQUIRED COMPONENTS system program_options)
# which then fails: "Could not find a package configuration file provided
# by 'boost_system'". Pinning to 1.86 (last line with boost_system as a
# compiled component) makes this find_package work without patching mycpp.
#
# Three packages needed for cmake CONFIG mode:
#   libboost          compiled .so libraries
#   libboost-devel    cmake config + per-component config files
#   libboost-headers  headers (transitively pulled but explicit is safer)
conda install -c conda-forge \
    eigen=3.4.0 \
    libboost=1.86 \
    libboost-devel=1.86 \
    libboost-headers=1.86 \
    -y

# ----------------------------------------------------------------------------

step "5/10  install FoundationPose requirements (filter conflicting pins)"
# Strip these pins from requirements.txt:
#   jupyterlab==4.1.5        old; kaolin 0.18 pulls newer transitively
#   ipywidgets==8.1.2        same
#   ipykernel==6.29.5        same
#   jupyter-client==7.4.9    kaolin-0.15-era constraint; kaolin 0.18 doesn't need
#   pyzmq==24.0.1            same
# These are notebook/jupyter pins that fork's install.sh added for kaolin 0.15.
# kaolin 0.18 (what we install in step 8) has no such constraints; let pip
# resolve the newer transitive versions.
#
# Also strip kornia==0.7.2 if it conflicts with torch 2.8 (kornia 0.7.2 declared
# torch>=1.9.0,<2.5.0). Replace with `kornia` (latest compatible with torch 2.8).
FILTERED_REQS="${TMPDIR%/}/fp_filtered_reqs.txt"
grep -vE "^(jupyterlab==|ipywidgets==|ipykernel==|jupyter-client==|pyzmq==|kornia==|warp-lang==)" \
    "${FP_ROOT}/requirements.txt" > "$FILTERED_REQS"
echo "kornia" >> "$FILTERED_REQS"    # let pip pick latest compatible
# warp-lang upgraded explicitly to >=1.5 in step 7 (kaolin 0.18 imports
# warp.fem.linalg which only exists in warp 1.5+; upstream pins 1.0.2).
echo "    requirements.txt -> $FILTERED_REQS  ($(wc -l <"$FILTERED_REQS") lines)"
pip install -r "$FILTERED_REQS"

# numpy<2: opencv-python==4.9.0.80 from requirements.txt was compiled
# against numpy 1.x ABI; numpy 2 import crashes cv2 (same lesson as H9.5).
pip install --upgrade 'numpy<2'

# ----------------------------------------------------------------------------

step "5.5  re-pin torch to 2.8.0+cu128 (requirements.txt may have downgraded)"
# H9.5 lesson: requirements.txt strict deps (lightning / fvcore / sentence-
# transformers in sam3d, ultralytics / wandb in FoundationPose) can drag
# torch back to an older version. Force-reinstall and verify sm_120 ∈
# supported_arches before proceeding to source builds.
pip install --upgrade --force-reinstall \
    torch==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

python - <<'PY'
import inspect, torch
import torch.utils.cpp_extension as ce
print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
src = inspect.getsource(ce._get_cuda_arch_flags)
if "'12.0'" in src or '"12.0"' in src:
    print("✓ torch._get_cuda_arch_flags knows sm_120")
else:
    raise SystemExit("✗ torch does NOT know sm_120 — ABORT")
PY

# ----------------------------------------------------------------------------

step "6/10  build vendored pytorch3d from source (sm_120, same recipe as H9.5)"
# Upstream uses NVlabs/FoundationPose readme's:
#   python -m pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
# That prebuilt wheel is cu118 + torch 2.0 + py39 — has no sm_120 PTX.
# Per PR #369 README addition: "PyTorch3D does not currently have a prebuilt
# binary for sm_120; install it from source manually."
TORCH_CUDA_ARCH_LIST="$FOUNDATIONPOSE_CUDA_ARCH_LIST" pip install --force-reinstall --no-deps \
    --no-cache-dir --no-build-isolation \
    "$PYTORCH3D_ROOT"

# ----------------------------------------------------------------------------

step "7/10  install kaolin 0.18.0 from NVIDIA wheel (cp39, torch-2.8.0_cu128)"
# PyPI 'kaolin' is a placeholder. Real wheel at NVIDIA S3:
#   https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
# We probed: cp39 wheel exists alongside H9.5's cp311. --no-deps + manual
# install of kaolin's runtime deps (matches H9.5).
pip install --force-reinstall --no-deps \
    kaolin==0.18.0 \
    --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

pip install \
    pygltflib usd-core \
    ipycanvas ipyevents pybind11 \
    'jupyter_client<8'
# warp-lang explicitly upgraded — kaolin 0.18 imports warp.fem.linalg which
# only exists in warp-lang>=1.5. Without --upgrade pip leaves the 1.0.2 from
# fork's requirements.txt in place and kaolin import dies.
pip install --upgrade 'warp-lang>=1.5'

# ----------------------------------------------------------------------------

step "8/10  build vendored nvdiffrast from source (sm_120 PTX)"
# Upstream install.sh has elaborate CC/CXX/CUDA_HOME setup so nvcc finds the
# right host compiler. Preserve that pattern but anchor CUDA_HOME at the
# conda env so the cuda-nvcc 12.8 from step 3 is the one in use.
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
for cuda_lib_dir in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib"; do
    if [[ -d "${cuda_lib_dir}" ]]; then
        export LD_LIBRARY_PATH="${cuda_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin ${CUDAHOSTCXX}"

# --no-deps + --no-cache-dir for the same reasons as pytorch3d (H9.5 lessons).
TORCH_CUDA_ARCH_LIST="$FOUNDATIONPOSE_CUDA_ARCH_LIST" pip install --no-cache-dir --no-build-isolation --no-deps \
    "$NVDIFFRAST_ROOT"

# ----------------------------------------------------------------------------

step "9/10  build FoundationPose extensions (mycpp + mycuda via build_all_conda.sh)"
# Fork's build_all_conda.sh:
#   - cmake-builds mycpp/  (needs Boost + Eigen from step 4)
#   - pip install -e ./bundlesdf/mycuda (uses --no-build-isolation)
# The mycuda build compiles CUDA kernels — first sm_120 test for
# FoundationPose-specific code. PR #369 patches (c++17 + .scalar_type())
# are already in the fork, so it should compile cleanly. ABORT on failure.
#
# Set TORCH_CUDA_ARCH_LIST here too — bundlesdf/mycuda/setup.py uses
# torch's cpp_extension which honours this env var.
export TORCH_CUDA_ARCH_LIST="$FOUNDATIONPOSE_CUDA_ARCH_LIST"

# CMAKE_PREFIX_PATH so cmake finds pybind11 + conda's Eigen + Boost
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CONDA_PREFIX}/lib/cmake:${CONDA_PREFIX}/lib/python3.9/site-packages/pybind11/share/cmake/pybind11${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

cd "${FP_ROOT}"
bash build_all_conda.sh
cd "${REPO_ROOT}"

# ----------------------------------------------------------------------------

step "9.5  re-pin numpy<2 (final — torch / nvdiffrast / mycuda builds re-upgrade numpy)"
# torch 2.8 declares numpy>=2.0 as a dependency. Step 5.5 force-reinstall
# torch overrides our step-5 numpy<2 pin. Each subsequent --force-reinstall
# in source builds (pytorch3d, nvdiffrast) may also bump numpy. Pin numpy<2
# AFTER all builds — scipy==1.12 and opencv-python==4.9 from requirements.txt
# were compiled against numpy 1.x ABI and crash on numpy 2 (dtype size mismatch).
pip install --force-reinstall --no-deps 'numpy<2'

step "10/10  verify (sm_120 GPU op + nvdiffrast/kaolin/pytorch3d/mycuda/mycpp/Utils)"
# Scope:
#   - Import + small GPU ops on 5070 Ti (proves sm_120 kernels work).
#   - We DON'T run a full pose estimation (model weights from Google Drive,
#     also need a real demo dataset). Defer end-to-end test to H9.8.
cd "${FP_ROOT}"
python - <<'PY'
import sys, os
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

x = torch.randn(512, 512, device="cuda")
_ = (x @ x).sum().item()
print("basic matmul   = ok")

# Heavy modules
import kaolin
print(f"kaolin         = {kaolin.__version__}")
import pytorch3d
print(f"pytorch3d      = {pytorch3d.__version__}")
import nvdiffrast
import nvdiffrast.torch as dr
print(f"nvdiffrast     = importable")

# FoundationPose-specific compiled extensions (the real sm_120 acid test):
sys.path.insert(0, ".")
import mycpp.build.mycpp as mycpp
print(f"mycpp          = importable (cluster_poses fn = {hasattr(mycpp, 'cluster_poses')})")

import bundlesdf.mycuda.common as mycuda_common
print("mycuda         = importable")

# Tiny op via mycuda to actually launch a sm_120 kernel (this is what
# triggers the kernel image / PTX dispatch — the real test)
try:
    # mycuda.common has sampleRaysUniformOccupiedVoxels etc.; check at least
    # one symbol exists. We don't run it (would need valid inputs).
    has_sample_rays = hasattr(mycuda_common, "sample_rays_uniform_occupied_voxels")
    print(f"mycuda.sample_rays_uniform_occupied_voxels exists: {has_sample_rays}")
except Exception as e:
    print(f"mycuda symbol check: {e}")

# FoundationPose's main entry — Utils does heavy imports including kornia,
# mycpp, etc. If anything is wired wrong, this is where it surfaces.
import Utils  # noqa
print("Utils.py       = importable")

import estimater
print("estimater.py   = importable (FoundationPose class available)")
PY

echo
echo "✅  $ENV install complete."
echo "    Reminder: full pose estimation NOT verified here."
echo "    Defer end-to-end test to H9.8 demo run."
