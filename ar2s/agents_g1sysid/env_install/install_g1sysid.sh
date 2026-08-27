#!/usr/bin/env bash
#
# Install the g1sysid env under AR2S_CONDA_PATH.
#
# Usage:
#   bash ar2s/agents_g1sysid/env_install/install_g1sysid.sh
#   AR2S_CONDA_PATH=/some/env/root bash ar2s/agents_g1sysid/env_install/install_g1sysid.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_NAME="${ENV_NAME:-g1sysid}"

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

step "create env $ENV_NAME (python 3.12)"
ar2s_create_prefix_env_or_fail "$ENV_NAME" "$ENV_PREFIX" "3.12"
conda activate "$ENV_PREFIX"
python --version

step "Install g1sysid dependencies"
cd "$SCRIPT_DIR"
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements_llm.txt

step "Install ar2s editable"
cd "$REPO_ROOT"
pip install --no-deps -e .

step "Verify g1sysid imports"
python <<'PY'
import ar2s.agents_g1sysid
import imageio
import imageio.plugins.ffmpeg
import langchain_core
from ar2s.agents_g1sysid.agents.motion_reader_agent import run_motion_reader_agent
from ar2s.agents_g1sysid.agents.sysid_loop.convergence_agent import assess_convergence
from ar2s.agents_g1sysid.bfm_zero.env import MuJoCoBFMZeroEnv
from ar2s.agents_g1sysid.skills.bundle_submit import make_submit_tool
from ar2s.agents_g1sysid.skills.compare_motion import compute_trajectory_metrics
from ar2s.agents_g1sysid.skills.motion_inspect import inspect_npz

print("g1sysid env install COMPLETE")
PY
