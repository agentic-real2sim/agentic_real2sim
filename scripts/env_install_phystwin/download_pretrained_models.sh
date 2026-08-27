#!/usr/bin/env bash
# Stage the bundled pretrained checkpoints for the ar2s.phystwin_visual pipeline.
#
#   sam2.1_hiera_large.pt      -> $PHYSTWIN_MODELS_ROOT/grounded_sam_2/
#                                 (default <repo>/models; read by
#                                 segment_util_video.py / segment_util_image.py)
#   superglue_*.pth,
#   superpoint_v1.pth          -> ar2s/phystwin_visual/models/weights/
#                                 (read by models/superglue.py, superpoint.py)
#
# The public release already contains these checkpoints under
# ar2s/phystwin_visual/. This script only stages/verifies those local files; it
# never contacts a model or code repository.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_ROOT="${PHYSTWIN_MODELS_ROOT:-$REPO_ROOT/models}"
BUNDLED_ROOT="$REPO_ROOT/ar2s/phystwin_visual"

stage() {  # stage <source> <dest>
    local source="$1"
    local dest="$2"
    if [ ! -f "$source" ]; then
        echo "ERROR: bundled checkpoint is missing: $source" >&2
        exit 1
    fi
    if [ -f "$dest" ]; then
        echo "[skip] $dest already exists"
    else
        mkdir -p "$(dirname "$dest")"
        cp "$source" "$dest"
        echo "[copy] $source -> $dest"
    fi
}

stage "$BUNDLED_ROOT/groundedSAM_checkpoints/sam2.1_hiera_large.pt" \
    "$MODELS_ROOT/grounded_sam_2/sam2.1_hiera_large.pt"

SG_DIR="$BUNDLED_ROOT/models/weights"
stage "$SG_DIR/superglue_indoor.pth" "$SG_DIR/superglue_indoor.pth"
stage "$SG_DIR/superglue_outdoor.pth" "$SG_DIR/superglue_outdoor.pth"
stage "$SG_DIR/superpoint_v1.pth" "$SG_DIR/superpoint_v1.pth"

echo "[done] phystwin visual checkpoints in place."
