#!/usr/bin/env bash
#
# Pre-install checks for ar2s.agents_visual setup (5 conda envs).
#
# Usage:
#     bash scripts/env_install_visual/precheck.sh
#
# Runs before any install_*.sh. install_foundationpose.sh also invokes this
# at its start as a defense (it's the build-heaviest env). Other install
# scripts assume the user has run precheck.sh manually.
#
# Per project policy, we NEVER auto-install system-level packages — on any
# missing dep we print what's needed + why, and exit non-zero. The user
# installs via their preferred package manager and re-runs.

set -eo pipefail

step() { echo "== precheck == $*"; }
warn() { echo "WARNING: $*" >&2; }
err()  { echo "ERROR: $*" >&2; }

fail=0

# ---------------------------------------------------------------------------
# Conda — required by every install_*.sh
# ---------------------------------------------------------------------------
if [ -z "${CONDA_EXE:-}" ] && ! command -v conda >/dev/null 2>&1; then
    err "'conda' not on PATH and \$CONDA_EXE unset."
    err "  Every scripts/env_install_visual/install_*.sh needs conda to create envs."
    err "  Install miniforge or miniconda from https://github.com/conda-forge/miniforge"
    err "  (or your preferred conda distribution) and re-run."
    fail=1
else
    step "conda available ✓"
fi

# ---------------------------------------------------------------------------
# Host C/C++ compiler — needed for source-built CUDA / C++ extensions
# ---------------------------------------------------------------------------
# nvcc compiles GPU code but hands C++ codegen to a host compiler. Conda's
# cuda-nvcc package does NOT ship one. Affected source builds:
#   - install_foundationpose.sh step 8/9 (nvdiffrast, mycpp, mycuda)
#   - install_sam3d.sh    step 5/7 (pytorch3d, gsplat)
#   - install_foundationpose.sh step 6 (pytorch3d)
for tool in gcc g++; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        err "'$tool' not on PATH."
        err "  Required by nvcc host C++ codegen for these source builds:"
        err "    - install_foundationpose.sh: nvdiffrast / mycpp / mycuda (steps 8–9)"
        err "    - install_sam3d.sh: pytorch3d / gsplat (steps 5 / 7)"
        err "    - install_foundationpose.sh: pytorch3d (step 6)"
        err "  Install via your distro's package manager and re-run precheck."
        err "  (Debian/Ubuntu ships gcc + g++ in the 'build-essential' meta-package.)"
        fail=1
    else
        step "$tool available ✓"
    fi
done

# ---------------------------------------------------------------------------
# Build tooling — cmake + ninja
# ---------------------------------------------------------------------------
if ! command -v cmake >/dev/null 2>&1; then
    err "'cmake' not on PATH."
    err "  Required by install_foundationpose.sh step 9 (mycpp cmake build,"
    err "  finds Boost + Eigen via find_package(... CONFIG REQUIRED))."
    err "  Install via your distro's package manager (e.g. 'cmake' package)."
    fail=1
else
    step "cmake available ✓"
fi

if ! command -v ninja >/dev/null 2>&1; then
    warn "'ninja' not on system PATH."
    warn "  Used by torch.utils.cpp_extension as the preferred builder for"
    warn "  pytorch3d / nvdiffrast / gsplat source compiles. Without it the"
    warn "  builds fall back to slow GNU Make — works, but build times multiply."
    warn "  install_foundation_stereo.sh pip-installs ninja inside its conda env"
    warn "  (covers stereo source builds); install_foundationpose.sh / sam3d don't."
    warn "  Optional fix: install via your distro's package manager ('ninja-build')."
else
    step "ninja available ✓"
fi

# ---------------------------------------------------------------------------
# NVIDIA driver — warn-only
# ---------------------------------------------------------------------------
# foundation_stereo env uses cu129 wheels which require NVIDIA driver ≥ 535.
# Other 4 envs use cu128 and run on older drivers, so this is a warning
# rather than a hard fail — user might be deliberately doing a partial setup.
if command -v nvidia-smi >/dev/null 2>&1; then
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    if [ -z "$drv" ]; then
        warn "nvidia-smi present but driver_version unparseable."
    elif [ "$drv" -lt 535 ]; then
        warn "NVIDIA driver $drv < 535."
        warn "  install_foundation_stereo.sh uses cu129 wheels which need driver ≥ 535."
        warn "  Other 4 envs use cu128 and likely work on older drivers."
        warn "  If you need foundation_stereo, update the driver via your usual method"
        warn "  (e.g. NVIDIA's .run installer, or your distro's driver package)."
    else
        step "NVIDIA driver $drv ≥ 535 ✓"
    fi
else
    warn "nvidia-smi not found — skipping driver check."
    warn "  No NVIDIA GPU visible; pipeline runtime will not work."
    warn "  install_*.sh may still succeed (CPU-only pip), but verify steps will fail."
fi

# ---------------------------------------------------------------------------
echo
if [ "$fail" -eq 0 ]; then
    step "all hard checks passed ✓"
    exit 0
else
    err "checks failed — install the missing tools and re-run."
    exit 1
fi
