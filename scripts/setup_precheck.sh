#!/usr/bin/env bash
#
# Pre-flight check for scripts/setup.sh.
#
# Runs the prerequisite checks from setup.sh as CHECKS ONLY — installs nothing. Tells you
# whether your machine is ready before you commit to a ~90 min install.
#
# Usage (on the target machine, after placing the repository locally):
#   bash scripts/setup_precheck.sh
#
# Exit code:
#   0 = all hard checks passed → safe to run scripts/setup.sh
#   1 = at least one hard check failed → fix before running setup.sh
#
# WARN-only items (soft checks) don't change exit code; they just inform
# you of possible runtime issues you may want to address.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# These two appear here ONLY for the printed summary at the end; nothing
# in this script touches conda envs.
export AR2S_CONDA_PATH="${AR2S_CONDA_PATH:-$REPO_ROOT/.conda_envs}"

PASS=()
WARN=()
FAIL=()

pass() { echo "  ✓ $*"; PASS+=("$*"); }
warn() { echo "  ⚠ $*" >&2; WARN+=("$*"); }
fail() { echo "  ✗ $*" >&2; FAIL+=("$*"); }
banner() { echo; echo "── $* ──"; }

banner "0. config"
echo "  REPO_ROOT       : $REPO_ROOT"
echo "  AR2S_CONDA_PATH : $AR2S_CONDA_PATH (default; export to override)"
echo "  USER            : $(whoami)"
echo "  HOSTNAME        : $(hostname)"

# ---------------------------------------------------------------------------
banner "1. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    pass "GPU: $GPU_NAME (sm_${GPU_CC/./}, driver $DRV)"
else
    fail "nvidia-smi not on PATH — no GPU access"
fi

# ---------------------------------------------------------------------------
banner "2. gcc / g++ (system compilers — can't be conda-installed)"
for tool in gcc g++; do
    if command -v "$tool" >/dev/null 2>&1; then
        VER=$("$tool" --version 2>/dev/null | head -1)
        pass "$tool: $(command -v "$tool") ($VER)"
    else
        fail "$tool not on PATH — ask admin: 'sudo apt install -y build-essential'"
    fi
done

# ---------------------------------------------------------------------------
banner "3. conda"
if command -v conda >/dev/null 2>&1; then
    pass "conda: $(command -v conda) ($(conda --version 2>&1))"
elif [ -d "$HOME/miniforge3" ]; then
    warn "conda not on PATH but $HOME/miniforge3 exists — source it or setup.sh will use it"
else
    warn "conda not installed — setup.sh will auto-install miniforge to $HOME/miniforge3 (~150 MB download)"
fi

# ---------------------------------------------------------------------------
banner "4. soft system check (warn-only)"
# Exact .so filenames (ldconfig is case-sensitive: libSM != libsm).
SOFT_LIBS=("libGL.so.1" "libSM.so.6" "libXext.so.6" "libglib-2.0.so.0")
if command -v ldconfig >/dev/null 2>&1; then
    # awk first column (lib name) + fixed-string + full-line match — robust
    # against ldconfig's "\tlibFOO.so.1 (libc6,...)" format.
    SO_NAMES=$(ldconfig -p 2>/dev/null | awk 'NR>1 {print $1}')
    for lib in "${SOFT_LIBS[@]}"; do
        if echo "$SO_NAMES" | grep -Fxq "$lib"; then
            pass "$lib"
        else
            warn "$lib missing — pipeline runtime may fail. If sudo: 'sudo apt install -y libgl1 libsm6 libxext6 libglib2.0-0'"
        fi
    done
else
    warn "ldconfig not available — skipping system C lib check"
fi

for tool in cmake ninja ffmpeg; do
    if command -v "$tool" >/dev/null 2>&1; then
        pass "$tool: $(command -v "$tool")"
    else
        warn "$tool not on host PATH — will be conda-installed per env as needed (this is fine)"
    fi
done

# Disk space
AVAIL=$(df -BG "$REPO_ROOT" | tail -1 | awk '{print $4}' | tr -d 'G')
MNT=$(df -BG "$REPO_ROOT" | tail -1 | awk '{print $NF}')
if [ "${AVAIL:-0}" -ge 150 ]; then
    pass "disk: ${AVAIL}G free at $MNT (recommended ≥ 150G)"
elif [ "${AVAIL:-0}" -ge 80 ]; then
    warn "disk: ${AVAIL}G free at $MNT (≥ 80G OK for single-episode test, ≥ 150G recommended for batch)"
else
    fail "disk: only ${AVAIL}G free at $MNT — setup needs ≥ 80G for envs + caches (peak)"
fi

echo
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "PASS : ${#PASS[@]}"
echo "WARN : ${#WARN[@]}  (won't block setup but worth knowing)"
echo "FAIL : ${#FAIL[@]}  (must fix before setup.sh)"
echo

if [ ${#FAIL[@]} -gt 0 ]; then
    echo "Must fix:"
    for msg in "${FAIL[@]}"; do echo "  ✗ $msg"; done
    echo
    echo "After fixing the above, re-run this precheck. When 0 fails, proceed to:"
    echo "  bash scripts/setup.sh 2>&1 | tee ~/setup.log"
    exit 1
fi

if [ ${#WARN[@]} -gt 0 ]; then
    echo "Worth knowing (not blocking):"
    for msg in "${WARN[@]}"; do echo "  ⚠ $msg"; done
    echo
fi

echo "All hard checks passed — safe to run:"
echo "  bash scripts/setup.sh 2>&1 | tee ~/setup.log"
exit 0
