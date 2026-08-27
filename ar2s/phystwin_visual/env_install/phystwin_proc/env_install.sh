#!/usr/bin/env bash
# Conda env for PhysTwin processing stages:
#   pcd_projection, mask_cleanup, track_processing, final_export
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ENV_NAME="${ENV_NAME:-phystwin_proc}"
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

pip install --upgrade pip setuptools wheel

conda install -y numpy==1.26.4 opencv
pip install --no-cache-dir open3d trimesh rtree
pip install --no-cache-dir scipy matplotlib pillow tqdm imageio moviepy

# Register repo source so ar2s.phystwin_visual is importable in this Python 3.10 env.
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$REPO_ROOT" > "$SITE_PACKAGES/ar2s_repo_root.pth"

python - <<'PY'
import ar2s.phystwin_visual
import cv2, imageio, open3d, scipy, trimesh
print("phystwin_proc imports OK")
PY

echo "phystwin_proc env ready."
