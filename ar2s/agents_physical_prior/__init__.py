"""ar2s.agents_physical_prior — VLM-driven physical prior inference.

Sits between the visual agent (geometry recovery: mesh + scale + pose) and
the sysid scene-prep stages (collision, camera select, calibration). Job:
infer per-object physical properties (currently mass, via material
classification × density × scaled volume) and patch them into the emitted
episode bundle's ``manifest.yaml``.

**Not** system identification — does not fit dynamics against an observed
trajectory. The output is a *prior* (informed guess based on visual evidence
+ mesh geometry), which subsequent grasp/probing stages can override or
refine.

Public surface:
    ``orchestrator.PhysicalPriorInput``
    ``orchestrator.data_ingest(episode_dir, run_root) -> PhysicalPriorInput``
    ``orchestrator.run_physical_prior(input) -> ApplyReport``

Compatibility helper:
    ``apply.apply_physical_priors(episode_dir, run_root) -> ApplyReport``
"""

from ar2s.agents_physical_prior.apply import ApplyReport, apply_physical_priors
from ar2s.agents_physical_prior.orchestrator import (
    PhysicalPriorInput,
    data_ingest,
    run_physical_prior,
)


__all__ = [
    "ApplyReport",
    "PhysicalPriorInput",
    "apply_physical_priors",
    "data_ingest",
    "run_physical_prior",
]
