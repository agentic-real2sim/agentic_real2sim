# PhysTwin Visual Third-Party Code

The migrated PhysTwin visual pipeline includes the research source trees it
uses under this repo's `third_party/` directory. The repository column below
records provenance; it is not an installation instruction. Do not replace the
bundled sources with random packages copied out of a conda environment.

Recommended locations:

| Component | Repository (provenance) | Bundled path | Used by |
|---|---|---|---|
| Grounded-SAM-2 | git@github.com:ericchen321/Grounded-SAM-2.git | `third_party/Grounded-SAM-2_phystwin` | `segment_util_video.py`, `segment_util_image.py` |
| GroundingDINO | git@github.com:ericchen321/GroundingDINO.git | `third_party/GroundingDINO_phystwin` | Grounded-SAM-2 detection backend |
| CoTracker | git@github.com:ericchen321/co-tracker.git | `third_party/co-tracker_phystwin` | `dense_track.py` |
| PyTorch3D | git@github.com:ericchen321/pytorch3d.git | `third_party/pytorch3d_phystwin` | `align.py`, `utils/align_util.py` |
| TRELLIS | git@github.com:ericchen321/TRELLIS.git | `third_party/TRELLIS_phystwin` | `shape_prior.py` |
| 3D Gaussian Splatting | https://github.com/graphdeco-inria/gaussian-splatting | `third_party/gaussian-splatting` | TRELLIS/Gaussian rendering support |

Checkpoint files currently bundled under
`ar2s/phystwin_visual/groundedSAM_checkpoints/` are:

- `GroundingDINO_SwinT_OGC.py`
- `groundingdino_swint_ogc.pth`
- `sam2.1_hiera_large.pt`

The PhysTwin-specific repositories use the `_phystwin` suffix so this repo can
carry separate forks for DROID or other datasets later. The source installs may
still need project-specific build commands, CUDA, and editable installs.
`requirements.txt` only covers PyPI-available packages.
