"""Ground-plane offset alignment.

Two-stage ground tuning consumed by SimBuildAgent:

  Stage 1 — coarse_align_ground: physics-feedback Strategy A.
      Drop the reference object for a short duration; observe vertical
      drift and contact. Adjust GroundSpec.offset and repeat. Handles
      the "ground above object → no contact, object falls forever" case
      by detecting zero contacts and flipping direction.

  Stage 2 — fine_align_ground: 1D Nelder-Mead around Stage-1 result.

Both stages mutate `sim.config.ground.offset` via `sim.set_ground_offset`
and measure a drift via `sim.simulate_freefall`. They do not rebuild the
model.

A helper, `compute_theoretical_ground_offset`, derives the initial
offset from the reference object's mesh and scale so the pipeline is not
forced to hardcode `_MUG_Y_MIN_UNSCALED`-style constants.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ar2s.droid_sim.scene.config import SceneConfig
from ar2s.droid_sim.scene.mesh_utils import get_effective_scale, load_obj_vertices


# =========================================================================
# Theoretical offset from mesh
# =========================================================================

def compute_theoretical_ground_offset(cfg: SceneConfig) -> float:
    """Ground offset = distance from ref-object origin to its mesh bottom
    along `local_down_axis` (after per-axis scaling).

    offset = max(scaled_vertices @ local_down_axis)
    """
    ref = next(o for o in cfg.objects if o.name == cfg.ground.reference_object)
    local_down = np.asarray(cfg.ground.local_down_axis, dtype=np.float64)
    local_down = local_down / max(np.linalg.norm(local_down), 1e-12)
    verts = load_obj_vertices(ref.visual_mesh_path) * get_effective_scale(ref)
    return float(np.max(verts @ local_down))


# =========================================================================
# Stage 1 — coarse (Strategy A)
# =========================================================================

@dataclass
class GroundAlignResult:
    offset: float
    iterations: int
    final_drift: float                 # |p_N - p_0| of watched object (m)
    vertical_drift: float              # signed world-z drift (m)
    converged: bool
    reason: str
    history: list[dict] = field(default_factory=list)


def coarse_align_ground(
    sim,
    *,
    initial_offset: float | None = None,
    max_iter: int = 5,
    convergence_thresh: float = 0.01,        # 1 cm
    max_offset_delta: float = 0.02,          # ±2 cm hard cap
    step_duration: float = 0.5,              # seconds
    step_scale: float = 0.5,                 # fraction of drift to correct per iter
) -> GroundAlignResult:
    """Physics-feedback ground tuning (Strategy A).

    At each iteration:
      * drop scene for `step_duration` seconds with robot frozen at q_target[0]
      * measure vertical drift (world z) and contact presence
      * update offset:
          drift<0, had_contact=True  → ground too low (object fell & caught) → INCREASE offset
            Wait — re-reading derive_ground_pose: larger |offset| pushes ground
            FURTHER along local_down (away from ref). So "object fell"
            can mean ground is further than object bottom already. Either way
            we want to reduce |drift|; we move offset toward ref center:
              - drift<0 & contact  → offset too large → DECREASE
              - drift>0 & contact  → offset too small → INCREASE
              - drift<0 & no contact → ground above object (wrong sign /
                magnitude) → flip toward positive / large correction

    Capped in `[initial - max_offset_delta, initial + max_offset_delta]`.
    """
    if initial_offset is None:
        initial_offset = sim.config.ground.offset
    current = float(initial_offset)
    history: list[dict] = []

    watch = sim.config.ground.reference_object

    for it in range(max_iter):
        sim.set_ground_offset(current)
        result = sim.simulate_freefall(duration_seconds=step_duration)
        watched = result[watch]
        total = float(watched["drift"])
        vdrift = float(watched["vertical_drift"])
        had_contact = bool(watched["had_contact"])

        history.append({
            "iter": it, "offset": current,
            "drift": total, "vertical_drift": vdrift,
            "had_contact": had_contact,
        })

        if total < convergence_thresh:
            return GroundAlignResult(
                offset=current, iterations=it + 1,
                final_drift=total, vertical_drift=vdrift,
                converged=True,
                reason=f"coarse converged: drift {total*1000:.2f}mm "
                       f"< {convergence_thresh*1000:.0f}mm at iter {it+1}",
                history=history)

        # Decide direction + magnitude
        if not had_contact:
            # Object fell through thin air. Ground likely above / too small.
            # Increase offset aggressively toward twice-current-magnitude
            # (if current ≤ 0, jump to 1cm).
            if current <= 0:
                proposed = current + 0.01
            else:
                proposed = current * 1.5
        elif vdrift < 0:
            # Fell a bit but landed. Ground too far — pull it closer to ref.
            proposed = current - step_scale * abs(vdrift)
        else:
            # Pushed up. Ground too close to ref — push it further.
            proposed = current + step_scale * abs(vdrift)

        # Clip to bounds around initial
        lo = initial_offset - max_offset_delta
        hi = initial_offset + max_offset_delta
        new_offset = float(np.clip(proposed, lo, hi))

        if abs(new_offset - current) < 1e-6:
            return GroundAlignResult(
                offset=current, iterations=it + 1,
                final_drift=total, vertical_drift=vdrift,
                converged=False,
                reason=f"coarse hit offset bound ±{max_offset_delta*100:.1f}cm "
                       f"at iter {it+1}",
                history=history)

        current = new_offset

    return GroundAlignResult(
        offset=current, iterations=max_iter,
        final_drift=history[-1]["drift"],
        vertical_drift=history[-1]["vertical_drift"],
        converged=False,
        reason=f"coarse max_iter={max_iter} reached; "
               f"best drift {history[-1]['drift']*1000:.2f}mm",
        history=history)


# =========================================================================
# Stage 2 — fine (1D Nelder-Mead)
# =========================================================================

def fine_align_ground(
    sim,
    *,
    initial_offset: float,
    search_radius: float = 0.005,            # ±5 mm
    maxiter: int = 30,
    step_duration: float = 0.5,
) -> GroundAlignResult:
    """1D Nelder-Mead: minimize total drift over ref object."""
    watch = sim.config.ground.reference_object
    history: list[dict] = []
    best = {"drift": np.inf, "offset": initial_offset, "vdrift": 0.0}

    def objective(params: np.ndarray) -> float:
        offset = float(np.clip(
            params[0],
            initial_offset - search_radius,
            initial_offset + search_radius))
        sim.set_ground_offset(offset)
        result = sim.simulate_freefall(duration_seconds=step_duration)
        total = float(result[watch]["drift"])
        vdrift = float(result[watch]["vertical_drift"])
        history.append({"offset": offset, "drift": total, "vertical_drift": vdrift})
        if total < best["drift"]:
            best["drift"] = total
            best["offset"] = offset
            best["vdrift"] = vdrift
        return total

    minimize(
        objective, np.array([initial_offset]),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
    )

    sim.set_ground_offset(best["offset"])
    return GroundAlignResult(
        offset=best["offset"],
        iterations=len(history),
        final_drift=best["drift"],
        vertical_drift=best["vdrift"],
        converged=True,
        reason=f"fine best drift {best['drift']*1000:.2f}mm over "
               f"{len(history)} evals",
        history=history)
