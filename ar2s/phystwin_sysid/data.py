import numpy as np
import torch
import pickle
from ._logger import logger
from ._visualize import visualize_pc
from .config import PhysTwinConfig
import matplotlib.pyplot as plt


class PhysTwinRealData:
    def __init__(self, data_path: str, cfg: PhysTwinConfig,
                 device: str = "cuda", base_dir: str | None = None,
                 visualize: bool = False, save_gt: bool = True):
        logger.info(f"[DATA]: loading data from {data_path}")
        self.data_path = data_path
        self.base_dir = base_dir
        self.cfg = cfg
        self.device = device
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        object_points = data["object_points"]
        object_colors = data["object_colors"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        controller_points = data["controller_points"]
        other_surface_points = data["surface_points"]
        interior_points = data["interior_points"]

        # Get the rainbow color for the object_colors
        y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
        y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
        rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

        self.num_original_points = object_points.shape[1]
        self.num_surface_points = (
            self.num_original_points + other_surface_points.shape[0]
        )
        self.num_all_points = self.num_surface_points + interior_points.shape[0]

        # Concatenate the surface points and interior points
        self.structure_points = np.concatenate(
            [object_points[0], other_surface_points, interior_points], axis=0
        )
        self.structure_points = torch.tensor(
            self.structure_points, dtype=torch.float32, device=device
        )

        self.object_points = torch.tensor(
            object_points, dtype=torch.float32, device=device
        )
        # self.object_colors = torch.tensor(
        #     object_colors, dtype=torch.float32, device=device
        # )
        self.original_object_colors = torch.tensor(
            object_colors, dtype=torch.float32, device=device
        )
        # Apply the rainbow color to the object_colors
        rainbow_colors = torch.tensor(
            rainbow_colors, dtype=torch.float32, device=device
        )
        # Make the same rainbow color for each frame
        self.object_colors = rainbow_colors.repeat(self.object_points.shape[0], 1, 1)

        # # Apply the first frame color to all frames
        # first_frame_colors = torch.tensor(
        #     object_colors[0], dtype=torch.float32, device=device
        # )
        # self.object_colors = first_frame_colors.repeat(self.object_points.shape[0], 1, 1)

        self.object_visibilities = torch.tensor(
            object_visibilities, dtype=torch.bool, device=device
        )
        self.object_motions_valid = torch.tensor(
            object_motions_valid, dtype=torch.bool, device=device
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=device
        )

        self.frame_len = self.object_points.shape[0]
        # Visualize/save the GT frames
        self.visualize_data(visualize=visualize, save_gt=save_gt)

    def visualize_data(self, visualize: bool = False, save_gt: bool = True):
        if visualize:
            visualize_pc(
                self.object_points,
                self.cfg,
                self.object_colors,
                self.controller_points,
                self.object_visibilities,
                self.object_motions_valid,
                visualize=True,
            )
        if save_gt and self.base_dir is not None:
            visualize_pc(
                self.object_points,
                self.cfg,
                self.object_colors,
                self.controller_points,
                self.object_visibilities,
                self.object_motions_valid,
                visualize=False,
                save_video=True,
                save_path=f"{self.base_dir}/gt.mp4",
            )


class PhysTwinSyntheticData:
    def __init__(self, data_path: str, cfg: PhysTwinConfig,
                 device: str = "cuda", base_dir: str | None = None,
                 visualize: bool = False):
        logger.info(f"[DATA]: loading data from {data_path}")
        self.data_path = data_path
        self.base_dir = base_dir
        self.cfg = cfg
        self.device = device
        self.data = np.load(self.data_path)
        self.data = torch.tensor(self.data, dtype=torch.float32, device=device)
        self.frame_len = self.data.shape[0]
        self.point_num = self.data.shape[1]
        # Visualize/save the GT frames
        self.visualize_data(visualize=visualize)

    def visualize_data(self, visualize: bool = False):
        if visualize:
            visualize_pc(
                self.data,
                self.cfg,
                visualize=True,
            )
        if self.base_dir is not None:
            visualize_pc(
                self.data,
                self.cfg,
                visualize=False,
                save_video=True,
                save_path=f"{self.base_dir}/gt.mp4",
            )


# Backwards-compatible aliases used internally by optimize.py and train.py
# (removed once those files are fully refactored)
RealData = PhysTwinRealData
SimpleData = PhysTwinSyntheticData
