from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
import typing
from typing import Optional
import numpy as np
import yaml


@dataclass
class PhysTwinConfig:
    data_type: str = "real"
    fps: int = 30
    dt: float = 5e-5
    dashpot_damping: float = 100.0
    drag_damping: float = 3.0
    base_lr: float = 1e-3
    iterations: int = 250
    vis_interval: int = 10
    init_spring_Y: float = 3e3
    collide_elas: float = 0.5
    collide_fric: float = 0.3
    collide_object_elas: float = 0.7
    collide_object_fric: float = 0.3
    object_radius: float = 0.02
    object_max_neighbours: int = 30
    controller_radius: float = 0.04
    controller_max_neighbours: int = 50
    spring_Y_min: float = 0.0
    spring_Y_max: float = 1e5
    reverse_z: bool = True
    vp_front: list = field(default_factory=lambda: [1, 0, -2])
    vp_up: list = field(default_factory=lambda: [0, 0, -1])
    vp_zoom: float = 1.0
    collision_dist: float = 0.06
    collision_learn: bool = True
    self_collision: bool = False
    use_graph: bool = True
    chamfer_weight: float = 1.0
    track_weight: float = 1.0
    acc_weight: float = 0.01
    # visualization / camera (Optional; set by caller before constructing engines)
    overlay_path: Optional[str] = None
    c2ws: Optional[np.ndarray] = None
    w2cs: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    WH: Optional[list] = None
    _numeric_fields = None  # class-level cache

    @property
    def num_substeps(self) -> int:
        return round(1.0 / self.fps / self.dt)

    @classmethod
    def _coerce_types(cls, d: dict) -> dict:
        if cls._numeric_fields is None:
            hints = typing.get_type_hints(cls)
            cls._numeric_fields = {
                f.name: hints[f.name]
                for f in dataclasses.fields(cls)
                if hints.get(f.name) in (int, float)
            }
        out = dict(d)
        for k, t in cls._numeric_fields.items():
            if k in out and out[k] is not None:
                out[k] = t(out[k])
        return out

    @classmethod
    def from_yaml(cls, path: str) -> "PhysTwinConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        if "FPS" in d:
            d["fps"] = d.pop("FPS")
        d.pop("num_substeps", None)
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**cls._coerce_types(d))
    
    def with_optimal_params(self, optimal_params: dict) -> "PhysTwinConfig":
        p = dict(optimal_params)
        if "global_spring_Y" in p:
            p["init_spring_Y"] = p.pop("global_spring_Y")
        p = {k: v for k, v in p.items() if k in self.__dataclass_fields__}
        return dataclasses.replace(self, **type(self)._coerce_types(p))
