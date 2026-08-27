#!/usr/bin/env bash
#
# Zero-sudo one-shot setup for droid_sim. Installs 5 conda envs
# (droid_sim + 4 visual envs). Designed to work on any Linux/GPU host:
#
#   - vast.ai container (5090 / H100 / ...)
#   - Lambda Labs / Paperspace / AWS GPU VM
#   - University HPC / shared GPU server (no sudo)
#   - your own workstation
#
# Hard requirement: gcc / g++ on PATH (system-only — conda's gcc_linux-64
# conflicts with system headers). If missing, ask admin to install
# build-essential (Debian/Ubuntu) or equivalent.
#
# Everything else (cmake / ninja / ffmpeg / pip wheels / model code) is
# installed into the conda envs themselves under AR2S_CONDA_PATH —
# sandboxed, no system pollution.
#
# 3-step usage:
#   1. ssh into the machine
#   2. export ANTHROPIC_API_KEY=...    # for LLM-driven agents
#      export HF_TOKEN=...             # for SAM3 (gated) weight download
#      export AR2S_CONDA_PATH=/path    # OPTIONAL — where to put 5 envs
#                                      # (default: <repo>/.conda_envs)
#   3. obtain this repository locally, then enter its root directory
#      bash scripts/setup_precheck.sh        # OPTIONAL but recommended:
#                                            # verifies env in 5 sec; tells you
#                                            # what to fix BEFORE the 90-min install
#      bash scripts/setup.sh   2>&1 | tee ~/setup.log
#
# Total: ~60-90 min (mostly pip installs). Idempotent — re-run is safe.
# Disk: ~50 GB for 5 envs + ~30 GB for caches during install (~80 GB peak).
# Weights NOT bundled — see docs/deployment_from_scratch_zh.md §5.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pin AR2S_CONDA_PATH so all install_*.sh scripts agree where envs go.
# Without this, the install scripts fall back to AR2S_DEFAULT_CONDA_PATH
# in scripts/ar2s_conda_env.sh which is a qianjun-linux-specific path that
# won't exist on cloud / shared servers.
#
# NOTE: must NOT be $REPO_ROOT/.conda — that's reserved for the symlink the
# project creates (ar2s_prepare_conda_env_root) pointing AT this path. Using
# the same path causes "exists but is not a symlink" abort. Use a sibling
# dir (.conda_envs) instead — same disk as repo, distinct from the symlink.
export AR2S_CONDA_PATH="${AR2S_CONDA_PATH:-$REPO_ROOT/.conda_envs}"

# DROID_MODELS_ROOT — project convention for ALL model weights. The
# visual wrappers (ar2s/agents_visual/_toolkit/{stereo_depth,mesh_recover}.py)
# read MODELS_ROOT = $DROID_MODELS_ROOT (or $REPO_ROOT/models if unset).
#
# Per-env HF_HUB_CACHE is auto-set to $DROID_MODELS_ROOT/hf_cache by the
# wrappers in _runners.py, so SAM3 / SAM3D / etc. auto-download lands here
# too. No need to set HF_HOME separately for pipeline runs.
#
# Must also be set at pipeline runtime — written to scripts/env.sh below.
export DROID_MODELS_ROOT="${DROID_MODELS_ROOT:-$REPO_ROOT/models}"
mkdir -p "$DROID_MODELS_ROOT"

LOG_PREFIX() { printf '[%s] ' "$(date +%H:%M:%S)"; }
banner() { echo; echo "============================================================"; LOG_PREFIX; echo "$*"; echo "============================================================"; }
abort()  { LOG_PREFIX; echo "ABORT: $*" >&2; exit 1; }
warn()   { LOG_PREFIX; echo "WARN: $*" >&2; }

# ---------------------------------------------------------------------------
# 0. Banner — show config
# ---------------------------------------------------------------------------
banner "0/8  config"
echo "REPO_ROOT         : $REPO_ROOT"
echo "AR2S_CONDA_PATH   : $AR2S_CONDA_PATH"
echo "USER              : $(whoami)"
echo "HOSTNAME          : $(hostname)"

# ---------------------------------------------------------------------------
# 1. GPU check — hard, this whole thing is pointless without a GPU
# ---------------------------------------------------------------------------
banner "1/8  GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    abort "nvidia-smi not on PATH — GPU not available. Pick a GPU-enabled machine/template."
fi
nvidia-smi | head -3
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "GPU compute cap   : sm_${GPU_CC/./}"

# ---------------------------------------------------------------------------
# 2. Compiler check — hard (only system gcc/g++ play nicely; conda's
#    gcc_linux-64 conflicts with /usr/include headers used by torch/CUDA)
# ---------------------------------------------------------------------------
banner "2/8  gcc / g++ check (hard)"
MISSING_COMPILERS=()
for tool in gcc g++; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        MISSING_COMPILERS+=("$tool")
    else
        echo "$tool             : $(command -v "$tool") ($("$tool" --version | head -1))"
    fi
done
if [ ${#MISSING_COMPILERS[@]} -gt 0 ]; then
    echo
    echo "MISSING: ${MISSING_COMPILERS[*]}"
    echo "These cannot be conda-installed reliably (env's gcc conflicts with system headers)."
    echo "Ask whoever has root to run:"
    echo "    sudo apt install -y build-essential"
    abort "missing system compiler(s)"
fi

# ---------------------------------------------------------------------------
# 3. conda — install miniforge to $HOME if missing (no sudo needed)
# ---------------------------------------------------------------------------
banner "3/8  conda"
if command -v conda >/dev/null 2>&1; then
    echo "conda             : $(command -v conda) ($(conda --version))"
else
    MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
    if [ ! -d "$MINIFORGE_DIR" ]; then
        echo "downloading miniforge → $MINIFORGE_DIR"
        wget -qO /tmp/Miniforge3.sh \
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
            || abort "miniforge download failed (check network)"
        bash /tmp/Miniforge3.sh -b -p "$MINIFORGE_DIR"
        rm /tmp/Miniforge3.sh
    fi
    export PATH="$MINIFORGE_DIR/bin:$PATH"
    conda init bash >/dev/null
    echo "miniforge installed at $MINIFORGE_DIR"
    echo "NOTE: open a new shell after this script for PATH to update permanently."
fi

# ---------------------------------------------------------------------------
# 4. Soft system check — warn but don't abort
# ---------------------------------------------------------------------------
banner "4/8  soft system check (warn-only)"

# Common runtime libs ML packages pull in (opencv / gymnasium rendering /
# matplotlib / glfw). Most GPU server images already have them. If missing,
# pipeline runtime may fail with cryptic "libGL.so.1 not found" etc.
SOFT_LIBS=("libGL.so.1" "libsm" "libxext" "libglib")
if command -v ldconfig >/dev/null 2>&1; then
    for lib in "${SOFT_LIBS[@]}"; do
        if ldconfig -p 2>/dev/null | grep -q "$lib"; then
            echo "$lib            : OK"
        else
            warn "$lib not in ldconfig — runtime may fail. If sudo: sudo apt install -y libgl1 libsm6 libxext6 libglib2.0-0"
        fi
    done
else
    warn "ldconfig not available — skipping system C lib check"
fi

# cmake / ninja: each install_*.sh that needs them conda-installs into its
# env if missing. So we just inform.
for tool in cmake ninja ffmpeg; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "$tool          : $(command -v "$tool")"
    else
        echo "$tool          : not on host PATH — will be conda-installed per env as needed"
    fi
done

# Disk space
AVAIL=$(df -BG "$REPO_ROOT" | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${AVAIL:-0}" -lt 100 ]; then
    warn "only ${AVAIL}G free at $REPO_ROOT — need ~80G for envs + caches; install may fail mid-way"
else
    echo "disk free        : ${AVAIL}G at $(df -BG "$REPO_ROOT" | tail -1 | awk '{print $NF}')"
fi

# ---------------------------------------------------------------------------
# 5. droid_sim base env (~5-10 min)
# ---------------------------------------------------------------------------
banner "5/8  install droid_sim base env"
cd "$REPO_ROOT"
if ! bash scripts/env_install_base/install_droid_sim.sh 2>&1 | tail -30; then
    abort "droid_sim env install failed"
fi
echo "droid_sim env OK"

# ---------------------------------------------------------------------------
# 6-8. 4 visual envs (each ~15-20 min)
# ---------------------------------------------------------------------------
install_visual_env() {
    local idx=$1 name=$2 script=$3 vendored_dir=$4
    banner "${idx}  install $name"

    if [ ! -d "$REPO_ROOT/$vendored_dir" ]; then
        abort "vendored dependency missing: $REPO_ROOT/$vendored_dir"
    fi
    if ! bash "scripts/env_install_visual/$script" 2>&1 | tail -50; then
        abort "$name install failed — see log above"
    fi
    echo "$name env OK"
}

install_visual_env "6a/8" sam3              install_sam3.sh              third_party/sam3
install_visual_env "6b/8" sam3d             install_sam3d.sh              third_party/sam-3d-objects
install_visual_env "7/8"  foundation_stereo install_foundation_stereo.sh third_party/FoundationStereo
install_visual_env "8/8"  foundationpose    install_foundationpose.sh    third_party/FoundationPose

# ---------------------------------------------------------------------------
# Write scripts/env.sh — so any future shell can re-establish the env
# vars by `source scripts/env.sh`. Without this, a fresh shell falls back
# to defaults (~/.cache for HF, AR2S_DEFAULT_CONDA_PATH for envs) and
# weights download to the wrong place.
# ---------------------------------------------------------------------------
banner "writing scripts/env.sh"
ENV_FILE="$REPO_ROOT/scripts/env.sh"
cat > "$ENV_FILE" <<EOF
# Auto-generated by scripts/setup.sh on $(date).
# Source this BEFORE running any droid_sim pipeline command in a fresh shell:
#
#     source scripts/env.sh
#
# It re-establishes the env paths so install_*.sh and the visual wrappers
# find the right conda envs and weight directories.

export AR2S_CONDA_PATH="$AR2S_CONDA_PATH"
export DROID_MODELS_ROOT="$DROID_MODELS_ROOT"

# Optional: also re-export your API keys if you set them at setup time.
# (Commented out — uncomment + paste your keys if you want them auto-sourced.)
# export ANTHROPIC_API_KEY="..."
# export HF_TOKEN="..."
EOF
chmod 0644 "$ENV_FILE"
echo "wrote $ENV_FILE"

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
banner "DONE — summary"
echo "AR2S_CONDA_PATH   : $AR2S_CONDA_PATH"
echo "DROID_MODELS_ROOT : $DROID_MODELS_ROOT"
echo
echo "envs created (project uses flat layout, no envs/ subdir):"
ls -d "$AR2S_CONDA_PATH"/*/ 2>/dev/null | sed 's/^/  /'
echo
df -h "$REPO_ROOT" | tail -2
echo
echo "Smoke test (run from $REPO_ROOT):"
echo "  # base env (no torch, just check ar2s imports + langchain):"
echo "  .conda/droid_sim/bin/python -c 'import ar2s.run_pipeline, langchain_core, mujoco; print(\"droid_sim base OK\")'"
echo
echo "  # 4 visual envs — each must report CUDA True:"
echo "  for e in qianjun_sam3 qianjun_sam3d qianjun_foundation_stereo qianjun_foundationpose; do"
echo "      .conda/\$e/bin/python -c \"import torch; print('\$e', torch.__version__, torch.cuda.is_available())\""
echo "  done"
echo
echo "Weights NOT bundled — see docs/deployment_from_scratch_zh.md §5"
echo
echo "==> ALWAYS source scripts/env.sh before running pipeline commands <=="
echo
echo "setup done at $(date)"
