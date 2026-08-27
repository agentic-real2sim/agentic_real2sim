"""Result types produced by the G1 SysID pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SysIDResult:
    """Final result of a Stage-3 SysID run.

    Attributes
    ----------
    outcome:         "converged" | "max_iters" | "give_up"
    alpha_best:      best correction vector found by hill-climbing
    r_baseline:      reward with broken physics, no correction (α = 1)
    r_oracle:        reward with the true oracle correction
    r_recovered:     reward achieved by the best found alpha
    history:         fitness value at each hill-climbing iteration (len = n_iters+1)
    motion_metrics:  trajectory comparison metrics (MPJPE etc.) for the final rollout
    n_iters:         actual hill-climbing iterations executed
    run_dir:         artifacts directory (summary.yaml + rollouts)
    """
    outcome: str
    alpha_best: tuple
    r_baseline: float
    r_oracle: float
    r_recovered: float
    history: list[float]
    motion_metrics: dict
    n_iters: int
    run_dir: Optional[Path] = None


__all__ = ["SysIDResult"]
