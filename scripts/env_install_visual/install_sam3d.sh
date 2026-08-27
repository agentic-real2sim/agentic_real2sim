#!/usr/bin/env bash
# Install qianjun_sam3d env for sam-3d-objects mesh recovery (H9.5).
#
# We DEVIATE significantly from the upstream environments/default.yml because:
#   - it hard-locks to CUDA 12.1 (cuda-nvcc 12.1.105, libcublas 12.1, ...)
#     which doesn't support sm_120 (Blackwell — 5070 Ti / 5090).
#   - it uses --extra-index-url for cu121 wheels which lack sm_120 PTX.
#     Running on Blackwell would launch kernels with no PTX, failing with
#     "no kernel image is available for execution on the device".
#
# Strategy: build fresh on cu128 (proven sm_120 baseline from H9.3 / H9.4 —
# torch 2.8/2.10 with cu128/cu129 wheels work on 5070 Ti). Build pytorch3d
# and gsplat from source with TORCH_CUDA_ARCH_LIST="12.0". Skip flash-attn
# unless verify proves it's a hard requirement.
#
# ============================================================================
# SIX SM_120 RISK POINTS — each step annotates expected failure modes
# ============================================================================
#
# (1) ✅ torch + cu128                  proven OK in H9.3 / H9.4
# (2) ⚠️ kaolin                         upstream pin 0.17.0 only has cu121
#                                       prebuilt; we try latest PyPI for cu128
# (3) ⚠️ pytorch3d from vendored source  phystwin env's recipe should work
# (4) ⚠️ gsplat from vendored source     same recipe as pytorch3d
# (5) 🚨 flash_attn==2.8.3 in [p3d]      H9.3 lesson: flash-attn likely fails
#                                       on sm_120. SKIP initially. If
#                                       sam3d_objects imports it
#                                       unconditionally, verify will surface
#                                       it and we ABORT per user policy.
# (6) ⚠️ heavy requirements.txt pins    lightning==2.3.3 etc. were tested
#                                       against torch 2.5.1; on torch 2.10
#                                       some may resolve incompatibly.
#                                       We leave those pins alone; let pip
#                                       complain. Each complaint = decision.
# ============================================================================

set -eo pipefail

ENV=qianjun_sam3d
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SAM3D_ROOT="${REPO_ROOT}/third_party/sam-3d-objects"
PYTORCH3D_ROOT="${PYTORCH3D_ROOT:-${REPO_ROOT}/third_party/pytorch3d_phystwin}"
MOGE_ROOT="${MOGE_ROOT:-${REPO_ROOT}/third_party/MoGe}"
GSPLAT_ROOT="${GSPLAT_ROOT:-${REPO_ROOT}/third_party/gsplat}"

for vendored_dir in "$PYTORCH3D_ROOT" "$MOGE_ROOT" "$GSPLAT_ROOT"; do
    if [ ! -d "$vendored_dir" ]; then
        echo "ERROR: vendored dependency is missing at $vendored_dir" >&2
        echo "       Add the source under third_party/ or set its *_ROOT override." >&2
        exit 1
    fi
done

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/ar2s_conda_env.sh"

step() { echo; echo "==[$ENV]== $*"; }

ar2s_source_conda
ar2s_prepare_conda_env_root "$REPO_ROOT"
ENV_PREFIX="$(ar2s_conda_env_prefix "$REPO_ROOT" "$ENV")"

step "1/8  create conda env (python=3.11)  [idempotent]"
# Upstream uses python 3.11; matches H9.3/H9.4 versioning expectations.
ar2s_create_prefix_env_or_skip "$ENV" "$ENV_PREFIX" "3.11"

conda activate "$ENV_PREFIX"

step "2/8  install torch 2.8.0 + torchvision (cu128, sm_120 known + kaolin wheel exists)"
# Tried torch 2.10 first — works for sm_120 compile but NVIDIA only publishes
# kaolin wheels up to torch-2.8.0_cu128 (probed S3 — torch 2.9/2.10 = 404).
# Verified torch 2.8.0+cu128's _get_cuda_arch_flags supported_arches includes
# '12.0', '12.0a', '12.1', '12.1a' — i.e. Blackwell is known. So 2.8 is the
# sweet spot: sm_120 compile works AND kaolin wheel available.
pip install --upgrade \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

step "3/8  install cuda-nvcc + CUDA dev headers from conda nvidia channel"
# Needed for source-built pytorch3d / gsplat to compile sm_120 PTX.
# 12.8 is the lowest CUDA toolkit that knows about Blackwell (sm_120).
conda install -c nvidia \
    cuda-nvcc=12.8 cuda-cccl=12.8 cuda-cudart=12.8 cuda-cudart-dev=12.8 \
    cuda-libraries-dev=12.8 ninja \
    -y

# Same conda activate.d cuda-nvcc bug as H9.3 (NVCC_PREPEND_FLAGS unbound).
# Pre-export an empty value so the activate.d script doesn't crash under set -u
# if anyone ever flips that flag back on. set -u is currently off in this script.
export NVCC_PREPEND_FLAGS=""
export NVCC_APPEND_FLAGS=""
# Default covers Ada (sm_89 — 4090), Hopper (sm_90 — H100/H200), and
# Blackwell (sm_120 — 5090). Override per-machine if you only need one.
SAM3D_CUDA_ARCH_LIST="${SAM3D_CUDA_ARCH_LIST:-8.9;9.0;12.0}"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

step "4/8  install base sam-3d-objects requirements (filter cu121-incompatible pins)"
# Pins we strip and handle ourselves / can't satisfy on cu128:
#   cuda-python==12.1.0           need 12.8+; pip picks compatible
#   nvidia-cuda-nvcc-cu12==12.1.X conflicts with conda cuda-nvcc 12.8 from step 3
#   nvidia-pyindex                broken installer (ModuleNotFoundError: pip
#                                  in its isolated build env); only adds NGC
#                                  URL to pip config, not a runtime dep
#   torchaudio==2.5.1+cu121       cu121 PyPI wheel doesn't exist on cu128 line
#   spconv-cu121==2.3.8           package name itself locks cu121; SKIP entirely
#                                  (if sam3d_objects needs sparse conv, verify
#                                  will surface it and we'll add spconv-cu126
#                                  or whatever ~latest cu version has a wheel)
#   bitsandbytes==0.43.0          cu121-tied; pip picks latest
#   kaolin / gsplat / flash[-_]attn / pytorch3d  handled in steps 5-7 below
FILTERED_REQS="${TMPDIR%/}/sam3d_filtered_reqs.txt"
grep -vE "^(cuda-python==12\.1|nvidia-cuda-nvcc-cu|nvidia-pyindex|torchaudio|spconv-cu|bitsandbytes==0\.43|kaolin|gsplat|flash[-_]attn|pytorch3d|MoGe[[:space:]]+@)" \
    "${SAM3D_ROOT}/requirements.txt" > "$FILTERED_REQS"
echo "    requirements.txt -> $FILTERED_REQS  ($(wc -l <"$FILTERED_REQS") lines)"
pip install -r "$FILTERED_REQS"

step "4.1  install vendored MoGe (no VCS dependency resolution)"
pip install --no-deps -e "$MOGE_ROOT"

step "4.5  re-pin torch to 2.8.0+cu128 (requirements.txt drags torch back to 2.5.1+cu124)"
# requirements.txt pulls in lightning==2.3.3 / fvcore / sentence-transformers / etc.,
# some of which declare dependencies on torch<2.6 — pip's resolver downgrades
# our torch 2.8 to 2.5.1+cu124. torch 2.5.1's _get_cuda_arch_flags maxes at
# '9.0' (Hopper); it has NO knowledge of sm_120 (Blackwell). We need torch 2.8
# for both sm_120 compile AND kaolin wheel availability.
#
# Force-reinstall torch + verify sm_120 ∈ supported_arches before proceeding.
pip install --upgrade --force-reinstall \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

python - <<'PY'
import inspect, torch
import torch.utils.cpp_extension as ce
print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
src = inspect.getsource(ce._get_cuda_arch_flags)
if "'12.0'" in src or '"12.0"' in src or 'Blackwell' in src:
    print("✓ torch._get_cuda_arch_flags knows sm_120 (Blackwell)")
else:
    raise SystemExit("✗ torch does NOT know sm_120 in supported_arches — ABORT")
PY

step "5/8  build vendored pytorch3d from source (sm_120 PTX, against torch 2.8 ABI)"
# Same commit upstream's requirements.p3d.txt pins. We reuse phystwin env's
# proven recipe: TORCH_CUDA_ARCH_LIST includes both this workstation's Ada
# GPU (sm_89) and Blackwell (sm_120) by default, plus --no-build-isolation so
# the build sees our pre-installed torch. --force-reinstall to recompile if a
# previous run built against torch 2.10. --no-deps so pip doesn't yank our
# torch 2.8 and reinstall PyPI's latest torch (2.12+) as a "dep" of pytorch3d
# under --force-reinstall.
TORCH_CUDA_ARCH_LIST="$SAM3D_CUDA_ARCH_LIST" pip install --force-reinstall --no-deps \
    --no-cache-dir --no-build-isolation \
    "$PYTORCH3D_ROOT"

step "6/8  install kaolin 0.18.0 from NVIDIA wheel (torch 2.8.0 + cu128, cp311)"
# PyPI 'kaolin' is a PLACEHOLDER package (version 0.1) that raises
# ImportError when imported, pointing to NVIDIA's wheel server. Real NVIDIA
# kaolin lives at https://nvidia-kaolin.s3.us-east-2.amazonaws.com/ with
# per-(torch, cu) HTML index pages. Probed; only torch-2.8.0_cu128 has a
# wheel for our cp311 / linux_x86_64 combo (kaolin-0.18.0).
#
# Bug from earlier run: if placeholder kaolin 0.1 was already installed,
# `pip install kaolin --find-links X` reports "already satisfied" because
# the placeholder's version 0.1 < 0.18 doesn't trigger an upgrade-by-default.
# We --force-reinstall + pin to ==0.18.0 explicitly. --no-deps so the kaolin
# wheel's torch dep doesn't get re-resolved against PyPI's latest.
pip install --force-reinstall --no-deps \
    kaolin==0.18.0 \
    --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html || {
    echo
    echo "==[$ENV]== ❌ ABORT: kaolin wheel install failed."
    echo "    The NVIDIA wheel page exists (probed 200 OK at script-write time)."
    echo "    Likely follow-up: check S3 still has the URL, network is reachable, or"
    echo "    provide a compatible local kaolin wheel or install it before rerunning."
    exit 1
}

# kaolin's actual runtime deps (from its METADATA Requires-Dist). We did
# --no-deps above to keep torch pinned — now install kaolin's deps EXPLICITLY
# (none of these lists torch as a hard dep, so pip won't yank our torch 2.8).
# Many are already in env (numpy/Pillow/scipy/tqdm/ipython/jupyter_client/
# comm/flask/tornado); pip skips those.
pip install \
    pygltflib usd-core warp-lang \
    ipycanvas ipyevents pybind11

step "6.4  pin numpy<2 (opencv-python==4.9.0.80 was compiled against numpy 1.x ABI)"
# requirements.txt pins opencv-python==4.9.0.80 which was built against
# numpy 1.x. Without an explicit pin, transitive deps from kaolin / scipy /
# usd-core etc. pull numpy 2.4 — cv2 import then crashes with
# "numpy.core.multiarray failed to import". Same lesson as H9.4 sam3 env.
pip install --upgrade 'numpy<2'

step "6.5  install seaborn + gradio + spconv-cu126 (skipped in step 4 filter)"
# requirements.inference.txt has 4 lines: kaolin (step 6) / gsplat (step 7) /
# seaborn / gradio. notebook/inference.py imports seaborn directly. gradio
# is for sam-3d-objects' demo.py web UI — not strictly needed for our
# subprocess call from run_sam3d_single.py, but cheap to add for consistency.
#
# spconv: requirements.txt had spconv-cu121==2.3.8 (we filtered cu121-bound
# pins in step 4). sam3d_objects' sparse module imports spconv at module
# load time, so it's a hard runtime dep. PyPI only publishes cu126 as the
# latest cu variant (spconv-cu128 doesn't exist); cu126 wheel works on
# cu128 runtime via CUDA minor forward compat.
pip install seaborn==0.13.2 gradio==5.49.0 spconv-cu126==2.3.8

step "7/8  build vendored gsplat from source (sm_120 PTX, against torch 2.8 ABI)"
# --force-reinstall to recompile if a previous run built against torch 2.10.
# --no-deps for the same reason as pytorch3d above: prevent pip from yanking
# torch 2.8 and reinstalling latest PyPI torch (2.12+) as a gsplat dep.
TORCH_CUDA_ARCH_LIST="$SAM3D_CUDA_ARCH_LIST" pip install --force-reinstall --no-deps \
    --no-cache-dir --no-build-isolation \
    "$GSPLAT_ROOT"

step "8/8  install sam-3d-objects package (editable, --no-deps)"
# We already installed all deps in step 4; --no-deps prevents pip from
# trying to re-resolve and possibly pulling kaolin 0.17.0 again.
pip install --no-deps -e "${SAM3D_ROOT}"

# Apply hydra patch IF the resolved hydra version is 1.3.2 (upstream's pin).
# patching/hydra is a Python script with `#!/usr/bin/env python` shebang —
# invoke with `python`, not `bash` (was bugged in earlier run; bash parses
# the Python code as shell and dies at line 8 with syntax error).
if python -c "import hydra, sys; sys.exit(0 if hydra.__version__ == '1.3.2' else 1)" 2>/dev/null; then
    step "9  apply hydra patch (replaces hydra/core/utils.py)"
    python "${SAM3D_ROOT}/patching/hydra"
else
    echo
    echo "==[$ENV]== [info] hydra version != 1.3.2; skipping upstream patch."
fi

# Conditionally install flash-attn — sam3d_objects.modules.sparse defaults
# to ATTN="flash_attn" and import fails without it. flash-attn has pre-built
# wheels for Hopper (sm_90) and older but does NOT support Blackwell
# (sm_120) yet (H9.3 finding). Detect GPU compute capability and install
# only on sm_90 or older. On sm_120 the user must set SPARSE_ATTN_BACKEND
# at pipeline runtime (e.g. xformers / sdpa) to bypass the flash_attn path.
GPU_CC_INT=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
                | head -1 | awk -F. '{printf "%d%02d\n", $1, $2}')
if [ -n "$GPU_CC_INT" ] && [ "$GPU_CC_INT" -le 900 ]; then
    step "[H100/sm_90] install flash-attn (pre-built wheel)"
    pip install flash-attn --no-build-isolation
else
    echo
    echo "==[$ENV]== sm_${GPU_CC_INT} > sm_900 — SKIP flash-attn install."
    echo "==[$ENV]== Runtime: set SPARSE_ATTN_BACKEND=xformers to bypass flash_attn path."
fi

step "verify (GPU op + sam3d_objects + kaolin + pytorch3d + gsplat import)"
# Verify-scope clarification:
#   - YES on 5070 Ti (16 GB):  import + small GPU op (proves sm_120 kernels work).
#   - DEFERRED to 5090 (32 GB): full Inference() pipeline load + actual mesh recovery.
# That's why this verify does NOT instantiate Inference() — that'd OOM on
# 5070 Ti and isn't sm_120-related anyway (it's VRAM-related).
#
# sam3d_objects.__init__.py imports `sam3d_objects.init` (Meta-internal LIDRA
# initialiser) unless LIDRA_SKIP_INIT is set. The `.init` module isn't
# shipped in the public release; the escape hatch IS the public usage path.
# We export it here for verify; _runners.py auto-injects it for runtime
# subprocesses (mirrors XFORMERS_DISABLED=1 pattern for the stereo env).
export LIDRA_SKIP_INIT=1
python - <<PY
import sys
from pathlib import Path

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

# Basic GPU op — proves CUDA kernel launch path works in this env. On
# Blackwell this also exercises sm_120; on other visible GPUs it still
# validates the env without pretending the workstation has different hardware.
x = torch.randn(512, 512, device="cuda")
_ = (x @ x).sum().item()
print("basic matmul   = ok")

# Heavy modules — kaolin / pytorch3d / gsplat are sm_120-sensitive
import kaolin
print(f"kaolin         = {kaolin.__version__}")

import pytorch3d
print(f"pytorch3d      = {pytorch3d.__version__}")

import gsplat
print(f"gsplat         = importable")

# sam3d_objects package (transitively imports the above)
import sam3d_objects
print(f"sam3d_objects  = importable")

# inference.py is the entry point used by run_sam3d_single.py. Follow
# upstream README convention: insert sam-3d-objects/notebook on sys.path
# and import flat (from inference). Jupyter's notebook package (pulled
# by kaolin via ipycanvas/ipywidgets) would shadow notebook.inference
# otherwise.
sys.path.insert(0, "${SAM3D_ROOT}/notebook")
try:
    import inference  # noqa
    print("notebook/inference.py = importable (Inference / load_image available)")
except ImportError as exc:
    msg = str(exc).lower()
    if "flash" in msg:
        print(f"❌ FLASH_ATTN IS A HARD DEPENDENCY — ABORT per user policy")
        print(f"   exact error: {exc}")
        print(f"   Options (not auto-attempted):")
        print(f"     - try pip install flash-attn (may fail on sm_120)")
        print(f"     - patch sam3d_objects to skip flash_attn path")
        raise
    raise

# Tiny sm_120 op via pytorch3d / kaolin (smoke test compiled kernels)
from pytorch3d.ops import knn_points
pts = torch.randn(1, 128, 3, device="cuda")
out = knn_points(pts, pts, K=8)
print(f"pytorch3d.knn  = ok  (idx shape {tuple(out.idx.shape)})")
PY

echo
echo "✅  $ENV install complete."
echo "    Reminder: actual sam-3d-objects inference NOT verified here."
echo "    Defer to 5090 (32 GB VRAM) for end-to-end mesh recovery test."
