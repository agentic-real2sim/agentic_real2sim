#!/usr/bin/env bash
#
# Install the lightweight `visual` env for the ar2s.agents_visual orchestrator
# + subagent LLM calls. No CUDA, no foundation models — those live in the 4
# toolkit envs (foundation_stereo / sam3 / sam3d-objects / foundationpose),
# installed separately.
#
# Usage:
#     bash scripts/env_install_visual/install_visual_orchestrator.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-visual}"

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

step "LangChain / LangGraph + provider clients"
pip install --no-cache-dir \
    langgraph \
    langchain \
    langchain-core \
    langchain-anthropic \
    langchain-openai \
    pydantic

step "Numerical + IO utilities"
pip install --no-cache-dir \
    numpy \
    Pillow \
    pyyaml \
    h5py \
    trimesh \
    imageio \
    imageio-ffmpeg \
    click

step "Verify imports"
cd "$REPO_ROOT"
python <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath('.'))

# Core — use importlib.metadata (langgraph 1.x dropped __version__)
from importlib.metadata import version as _v
import langchain, langgraph, langchain_anthropic, langchain_openai, pydantic
print(f"langchain        = {_v('langchain')}")
print(f"langgraph        = {_v('langgraph')}")
print(f"langchain_anthropic = {_v('langchain-anthropic')}")
print(f"pydantic         = {pydantic.__version__}")

# Util
import numpy, PIL, yaml, h5py, trimesh, imageio, imageio_ffmpeg
print(f"numpy            = {numpy.__version__}")
print(f"PIL              = {PIL.__version__}")
print(f"h5py             = {h5py.__version__}")
print(f"trimesh          = {trimesh.__version__}")
print(f"imageio          = {imageio.__version__}")
print(f"imageio_ffmpeg   = {imageio_ffmpeg.__version__}")

# Visual-agent modules
import ar2s.agents_visual
import ar2s.agents_visual.state
import ar2s.agents_visual.outputs
import ar2s.agents_visual.resolve
import ar2s.agents_visual.skills.svo_extract
import ar2s.agents_visual._toolkit.video_build
print()
print("ar2s.agents_visual modules import OK in visual env")
PY
