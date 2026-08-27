"""User-facing entry schema for the PhysTwin visual pipeline."""
from dataclasses import dataclass, field


@dataclass
class PhysTwinInput:
    # ---- always required ----
    base_path: str          # absolute path to dataset base dir (e.g. "/data/different_types")
    case_name: str          # name of the case subdirectory (e.g. "case_001")
    category: str           # object category for Grounded-SAM-2 text prompt (e.g. "cup")

    # ---- optional pipeline config ----
    controller_name: str = "hand"   # name of the controller in segmentation mask
    shape_prior: bool = False        # if True, run shape_alignment and use mesh in final_export
    shape_mesh_path: str = ""        # path to pre-generated .glb mesh (required when shape_prior=True)
    pipeline_id: str = ""           # ID for run_root (logs); defaults to case_name if empty

    # ---- per-stage hyperparams ----
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    cotracker_query_points: int = 5000
    num_surface_points: int = 1024
    volume_sample_size: float = 0.005
    camera_indices: list[int] = field(default_factory=list)  # empty = auto-detect from depth/ dir
