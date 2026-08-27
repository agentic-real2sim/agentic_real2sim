#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROUNDED_SAM_ROOT="$REPO_ROOT/third_party/Grounded-SAM-2_phystwin"
GROUNDING_DINO_ROOT="$REPO_ROOT/third_party/GroundingDINO_phystwin"
TRELLIS_ROOT="$REPO_ROOT/third_party/TRELLIS_phystwin"
GS_ROOT="$REPO_ROOT/third_party/gaussian-splatting"
PYTORCH3D_ROOT="$REPO_ROOT/third_party/pytorch3d_phystwin"

for vendored_dir in \
    "$GROUNDED_SAM_ROOT" \
    "$GROUNDING_DINO_ROOT" \
    "$TRELLIS_ROOT" \
    "$GS_ROOT" \
    "$PYTORCH3D_ROOT"; do
    if [ ! -d "$vendored_dir" ]; then
        echo "ERROR: vendored dependency is missing at $vendored_dir" >&2
        exit 1
    fi
done

conda install -y numpy==1.26.4
pip install warp-lang
pip install usd-core matplotlib
pip install "pyglet<2"
pip install open3d
pip install trimesh
pip install rtree
pip install pyrender

conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -y opencv
pip install cma
pip install --no-cache-dir --no-build-isolation -e "$PYTORCH3D_ROOT"

# Install the env for realsense camera
pip install Cython
pip install pyrealsense2
pip install atomics
pip install pynput

# Install the vendored grounded-sam-2 and GroundingDINO sources.
pip install --no-cache-dir --no-build-isolation -e "$GROUNDED_SAM_ROOT"
pip install --no-cache-dir --no-build-isolation -e "$GROUNDING_DINO_ROOT"

# Install the env for image upscaler using SDXL
pip install diffusers
pip install accelerate

# TRELLIS is imported directly from its vendored source tree by ar2s. Do not
# run its upstream setup.sh here: that script can clone optional extensions
# which are not part of this release.
echo "using vendored TRELLIS at $TRELLIS_ROOT"

pip install gsplat==1.4.0
pip install kornia
pip install --no-cache-dir --no-build-isolation \
    "$GS_ROOT/submodules/diff-gaussian-rasterization" \
    "$GS_ROOT/submodules/simple-knn"
