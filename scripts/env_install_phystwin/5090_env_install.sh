#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROUNDED_SAM_ROOT="$REPO_ROOT/third_party/Grounded-SAM-2_phystwin"
TRELLIS_ROOT="$REPO_ROOT/third_party/TRELLIS_phystwin"
GS_ROOT="$REPO_ROOT/third_party/gaussian-splatting"
PYTORCH3D_ROOT="$REPO_ROOT/third_party/pytorch3d_phystwin"
BUNDLED_CHECKPOINT_ROOT="$REPO_ROOT/ar2s/phystwin_visual/groundedSAM_checkpoints"

for vendored_dir in \
    "$GROUNDED_SAM_ROOT" \
    "$TRELLIS_ROOT" \
    "$GS_ROOT" \
    "$PYTORCH3D_ROOT"; do
    if [ ! -d "$vendored_dir" ]; then
        echo "ERROR: vendored dependency is missing at $vendored_dir" >&2
        exit 1
    fi
done

stage_checkpoint() {
    local source_path="$1"
    local destination_path="$2"
    [ -f "$source_path" ] || {
        echo "ERROR: bundled checkpoint is missing at $source_path" >&2
        exit 1
    }
    if [ -f "$destination_path" ]; then
        echo "using existing checkpoint $destination_path"
    else
        mkdir -p "$(dirname "$destination_path")"
        cp "$source_path" "$destination_path"
        echo "staged checkpoint $destination_path"
    fi
}

conda install -y numpy==1.26.4
pip install warp-lang
pip install usd-core matplotlib
pip install "pyglet<2"
pip install open3d
pip install trimesh
pip install rtree
pip install pyrender

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -y opencv
pip install cma

# Install the env for realsense camera
pip install Cython
pip install pyrealsense2
pip install atomics
pip install pynput

# Install the vendored grounded-sam-2 sources and stage the bundled weights.
# Do not call the upstream download_ckpts.sh helpers: they fetch from external
# model hosts and the public release already carries the weights we use.
stage_checkpoint \
    "$BUNDLED_CHECKPOINT_ROOT/sam2.1_hiera_large.pt" \
    "$GROUNDED_SAM_ROOT/checkpoints/sam2.1_hiera_large.pt"
stage_checkpoint \
    "$BUNDLED_CHECKPOINT_ROOT/groundingdino_swint_ogc.pth" \
    "$GROUNDED_SAM_ROOT/gdino_checkpoints/groundingdino_swint_ogc.pth"
pip install --no-build-isolation -e "$GROUNDED_SAM_ROOT"
pip install --no-build-isolation -e "$GROUNDED_SAM_ROOT/grounding_dino"

# Install the env for image upscaler using SDXL
pip install diffusers
pip install accelerate

pip install gsplat==1.4.0
pip install kornia
cd "$GS_ROOT/submodules/diff-gaussian-rasterization/"
python setup.py build_ext --inplace
pip install -e .
cd "$GS_ROOT/submodules/simple-knn/"
pip install -e .
cd "$REPO_ROOT"

pip install plyfile

pip install --no-build-isolation -e "$PYTORCH3D_ROOT"

pip install einops

# TRELLIS is imported directly from its vendored source tree by ar2s. Do not
# run its upstream setup.sh here: that script can clone optional extensions.
echo "using vendored TRELLIS at $TRELLIS_ROOT"

# The old recipe fetched a machine-specific FlashAttention wheel from a GitHub
# release. The public installer deliberately does not fetch binary artifacts;
# provide a local wheel explicitly if this legacy environment needs one.
if [ -n "${FLASH_ATTN_WHEEL:-}" ]; then
    [ -f "$FLASH_ATTN_WHEEL" ] || {
        echo "ERROR: FLASH_ATTN_WHEEL does not exist: $FLASH_ATTN_WHEEL" >&2
        exit 1
    }
    pip install "$FLASH_ATTN_WHEEL"
fi
