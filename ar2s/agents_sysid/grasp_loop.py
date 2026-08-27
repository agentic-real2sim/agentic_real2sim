"""Stage 3 retry-loop orchestrator.

Outer loop:
  probe → (round 0 only) Agent A view-sufficiency → Agent B shift-decision
  → apply shift → repeat until grasped / give_up / max_rounds.

Round 0 runs Agent A to choose camera poses (4 baseline + 0-3 custom);
rounds 1+ re-render those same poses at the new shifted scene state and
go straight to Agent B.

Per-round caps:
  * shift_mm[i] ∈ [-PER_ROUND_SHIFT_CAP_MM, +PER_ROUND_SHIFT_CAP_MM]
  * cumulative_shift_mm[i] ∈ [-CUMULATIVE_SHIFT_CAP_MM, +CUMULATIVE_SHIFT_CAP_MM]
  * total outer rounds ≤ MAX_ROUNDS

Output layout::

    artifacts/<run-id>/
      summary.yaml           # outcome, history, cumulative_shift_mm
      poses_chosen.yaml      # 4 baseline + 0-3 custom (from round 0)
      rounds/
        00/
          probe_result.yaml
          images/baseline_*.jpg, custom_*.jpg
          agent_a_trace.json
          agent_b_decision.json
        01/
          probe_result.yaml
          images/*.jpg  (re-rendered poses_chosen at this round's state)
          agent_b_decision.json
        ...
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ar2s.agents_sysid.agents.grasp_loop.shift_decision_agent import (
    RoundSummary,
    ShiftDecision,
    decide_shift,
)
from ar2s.agents_sysid.agents.grasp_loop.view_sufficiency_agent import (
    PoseRecord,
    ViewSufficiencyResult,
    collect_diagnostic_views,
    rerender_pose,
)
from ar2s.agents_sysid._gl import configure_headless_gl
from ar2s.agents_sysid.probe import _load_or_compute_calibration
from ar2s.agents_sysid.scene import CalibratedScene
from ar2s.agents_sysid.skills.probe_grasp import (
    GraspProbeResult,
    compute_det_shift_mm,
    get_scene_state_at_frame,
    probe_grasp,
)


# -------------------------------------------------------------------
# Tunable defaults (CLI / caller can override)
# -------------------------------------------------------------------

MAX_ROUNDS = 10
PER_ROUND_SHIFT_CAP_MM = 10
CUMULATIVE_SHIFT_CAP_MM = 50
AGENT_A_MAX_CUSTOM = 3                   # also enforced inside Agent A
SAVE_USD_DEFAULT = True                  # write rounds/<NN>/result.usd per round
USD_SAMPLE_EVERY_DEFAULT = 1             # capture every Nth sim frame to USD


@dataclass
class GraspLoopResult:
    outcome: str                         # "success" | "give_up" | "max_rounds" | "cap_exceeded"
    n_rounds: int                        # how many rounds ran (1..MAX_ROUNDS)
    final_cumulative_shift_mm: tuple     # (dx, dy, dz)
    final_probe: GraspProbeResult
    poses_chosen: list[PoseRecord]
    history: list[RoundSummary] = field(default_factory=list)
    run_dir: Path | None = None


# -------------------------------------------------------------------
# Persistence helpers
# -------------------------------------------------------------------

def _make_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _yaml_safe(obj):
    """Convert dataclasses / tuples / Paths into yaml-friendly primitives."""
    if hasattr(obj, "to_dict"):
        return _yaml_safe(obj.to_dict())
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _save_probe_result(path: Path, result: GraspProbeResult) -> None:
    doc = {
        "grasped": result.grasped,
        "grasped_loose": result.grasped_loose,
        "failure_reason": result.failure_reason,
        "peak_lift_mm": result.peak_lift_m * 1000,
        "lift_window_sec": result.lift_window_sec,
        "closest_distance_mm": result.closest_distance_m * 1000,
        "dist_at_peak_lift_mm": result.dist_at_peak_lift_m * 1000,
        "engage_frame": result.engage_frame,
        "duration_sec": result.duration_sec,
        "cumulative_shift_mm": list(result.cumulative_shift_mm),
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


def _save_agent_a_trace(path: Path, result: ViewSufficiencyResult) -> None:
    doc = {
        "terminated_by": result.terminated_by,
        "n_custom_renders": result.n_custom_renders,
        "final_reasoning": result.final_reasoning,
        "poses_chosen": [p.to_dict() for p in result.poses_chosen],
        "image_paths": [str(p) for p in result.image_paths],
    }
    path.write_text(json.dumps(doc, indent=2))


def _save_agent_b_decision(path: Path, decision: ShiftDecision,
                           shift_clipped_mm: tuple,
                           cumulative_after_mm: tuple) -> None:
    doc = {
        "decision": decision.decision,
        "shift_mm_raw": list(decision.shift_mm),
        "shift_mm_applied": list(shift_clipped_mm),
        "cumulative_after_mm": list(cumulative_after_mm),
        "reasoning": decision.reasoning,
        "parse_error": decision.parse_error,
        "raw_response": decision.raw_response,
    }
    path.write_text(json.dumps(doc, indent=2))


def _save_poses_chosen(path: Path, poses: list[PoseRecord]) -> None:
    doc = {"poses": [p.to_dict() for p in poses]}
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


def _save_summary(path: Path, result: GraspLoopResult) -> None:
    doc = {
        "outcome": result.outcome,
        "n_rounds": result.n_rounds,
        "final_cumulative_shift_mm": list(result.final_cumulative_shift_mm),
        "final_probe": {
            "grasped": result.final_probe.grasped,
            "peak_lift_mm": result.final_probe.peak_lift_m * 1000,
            "closest_distance_mm": result.final_probe.closest_distance_m * 1000,
            "failure_reason": result.final_probe.failure_reason,
        },
        "limits": {
            "max_rounds": MAX_ROUNDS,
            "per_round_shift_cap_mm": PER_ROUND_SHIFT_CAP_MM,
            "cumulative_shift_cap_mm": CUMULATIVE_SHIFT_CAP_MM,
        },
        "history": [
            {
                "round_idx": r.round_idx,
                "cumulative_shift_mm_entering": list(r.cumulative_shift_mm),
                "peak_lift_mm": r.peak_lift_mm,
                "closest_distance_mm": r.closest_distance_mm,
                "decision": r.decision,
                "reasoning_summary": r.reasoning_summary,
            }
            for r in result.history
        ],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


# -------------------------------------------------------------------
# Per-round helpers
# -------------------------------------------------------------------

def _clip_per_round(shift: tuple, cap: int) -> tuple:
    return tuple(max(-cap, min(cap, int(round(s)))) for s in shift)


def _accumulate_with_cap(cum: tuple, delta: tuple, cap: int) -> tuple[tuple, bool]:
    """Add delta to cum; clip each axis to ±cap; return (new_cum, was_clipped)."""
    new_cum = tuple(c + d for c, d in zip(cum, delta))
    clipped = tuple(max(-cap, min(cap, c)) for c in new_cum)
    was_clipped = any(c != n for c, n in zip(clipped, new_cum))
    return clipped, was_clipped


# -------------------------------------------------------------------
# Main orchestrator
# -------------------------------------------------------------------

def grasp_loop(
    episode_dir: str | Path,
    *,
    artifacts_root: str | Path = "artifacts",
    run_id: str | None = None,
    max_rounds: int = MAX_ROUNDS,
    per_round_cap_mm: int = PER_ROUND_SHIFT_CAP_MM,
    cumulative_cap_mm: int = CUMULATIVE_SHIFT_CAP_MM,
    save_usd: bool = SAVE_USD_DEFAULT,
    usd_sample_every: int = USD_SAMPLE_EVERY_DEFAULT,
) -> GraspLoopResult:
    """Run the Stage 3 retry loop until grasped / give_up / max_rounds.

    Args:
        episode_dir: built episode folder (manifest.yaml + ideally calibration.yaml).
        artifacts_root: parent of <run-id>/ output (default "artifacts/").
        run_id: explicit id; default = timestamp + uuid6.
        max_rounds: total outer rounds cap.
        per_round_cap_mm: |shift_mm[i]| cap per round.
        cumulative_cap_mm: |cumulative_shift_mm[i]| cap.
    Returns:
        GraspLoopResult with outcome + history + final state.
    """
    # EGL offscreen when headless — avoids the GLFW no-display crash + hang.
    configure_headless_gl()

    episode_dir = Path(episode_dir).expanduser().resolve()
    artifacts_root = Path(artifacts_root).expanduser().resolve()
    run_id = run_id or _make_run_id()
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rounds_dir = run_dir / "rounds"
    rounds_dir.mkdir(exist_ok=True)

    print(f"[stage3-loop] episode = {episode_dir}")
    print(f"[stage3-loop] run_dir = {run_dir}")
    print(f"[stage3-loop] limits: rounds≤{max_rounds}, "
          f"per_round=±{per_round_cap_mm}mm, cumulative=±{cumulative_cap_mm}mm")

    # ----- Stage 2: load calibration -----
    scene: CalibratedScene = _load_or_compute_calibration(episode_dir)
    print(f"[stage3-loop] calibration loaded: primary_id={scene.camera_selection.primary_id}, "
          f"robot_iou={scene.robot_alignment.iou:.3f}")

    cumulative_mm: tuple = (0, 0, 0)
    poses_chosen: list[PoseRecord] = []
    poses_persisted = False
    history: list[RoundSummary] = []
    final_probe: GraspProbeResult | None = None
    outcome: str = "max_rounds"

    for round_idx in range(max_rounds):
        round_dir = rounds_dir / f"{round_idx:02d}"
        images_dir = round_dir / "images"
        round_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)

        cum_m = tuple(c / 1000.0 for c in cumulative_mm)  # mm → m
        print(f"\n[round {round_idx}] probe (cumulative_shift_mm={cumulative_mm}) ...")
        t0 = time.time()
        usd_dir = round_dir if save_usd else None
        probe_result, frame_cache = probe_grasp(
            scene, cumulative_shift_m=cum_m,
            usd_save_dir=usd_dir,
            usd_sample_every=usd_sample_every,
        )
        print(f"  probe done in {time.time()-t0:.1f}s — "
              f"grasped={probe_result.grasped}, "
              f"peak_lift={probe_result.peak_lift_m*1000:.1f}mm, "
              f"closest={probe_result.closest_distance_m*1000:.1f}mm")

        _save_probe_result(round_dir / "probe_result.yaml", probe_result)
        final_probe = probe_result

        if probe_result.grasped:
            outcome = "success"
            history.append(RoundSummary(
                round_idx=round_idx,
                cumulative_shift_mm=cumulative_mm,
                peak_lift_mm=probe_result.peak_lift_m * 1000,
                closest_distance_mm=probe_result.closest_distance_m * 1000,
                decision="success",
                reasoning_summary="probe grasped=True",
            ))
            print(f"  [SUCCESS] grasp succeeded at round {round_idx}")
            break

        # ===========================================================
        # Round 0: DETERMINISTIC shift from first_contact (no LLM)
        # ===========================================================
        if round_idx == 0:
            det_shift = compute_det_shift_mm(
                probe_result, cumulative_mm,
                cumulative_cap_mm=cumulative_cap_mm,
            )
            fc = probe_result.first_contact
            if fc is None:
                reasoning = "no gripper↔pickup contact in close window — DET no-op (defer to LLM next round)"
            else:
                reasoning = (
                    f"DET from first_contact: side={fc.side}, body={fc.body_name}, "
                    f"frame={fc.frame_idx}, corrective_shift_m={fc.corrective_shift_m}"
                )
            print(f"  DET round 0 → shift_mm={det_shift}")
            print(f"    {reasoning[:200]}")

            det_doc = {
                "kind": "deterministic",
                "shift_mm": list(det_shift),
                "reasoning": reasoning,
                "first_contact": (
                    {"detected": False} if fc is None else {
                        "frame_idx": fc.frame_idx,
                        "side": fc.side,
                        "body_name": fc.body_name,
                        "world_pos": list(fc.world_pos),
                        "gripper_frame_pos": list(fc.gripper_frame_pos),
                        "corrective_shift_m": list(fc.corrective_shift_m),
                    }
                ),
            }
            (round_dir / "det_decision.json").write_text(json.dumps(det_doc, indent=2))

            new_cum, was_clipped = _accumulate_with_cap(
                cumulative_mm, det_shift, cumulative_cap_mm,
            )
            history.append(RoundSummary(
                round_idx=round_idx,
                cumulative_shift_mm=cumulative_mm,
                peak_lift_mm=probe_result.peak_lift_m * 1000,
                closest_distance_mm=probe_result.closest_distance_m * 1000,
                decision="det",
                reasoning_summary=(
                    f"DET no-op (no contact)" if fc is None
                    else f"DET shift={det_shift} from contact on {fc.body_name}/{fc.side}"
                ),
            ))
            cumulative_mm = new_cum
            # Snapshot summary after round 0 too
            _save_summary(run_dir / "summary.yaml", GraspLoopResult(
                outcome="(in_progress)",
                n_rounds=round_idx + 1,
                final_cumulative_shift_mm=cumulative_mm,
                final_probe=probe_result,
                poses_chosen=poses_chosen,
                history=history,
                run_dir=run_dir,
            ))
            continue

        # ===========================================================
        # Round 1: FIRST LLM round — Agent A view-sufficiency loop
        # ===========================================================
        if round_idx == 1:
            print(f"  Agent A (view-sufficiency) ...")
            t0 = time.time()
            agent_a_result = collect_diagnostic_views(
                scene, frame_cache, images_dir,
            )
            print(f"  Agent A done in {time.time()-t0:.1f}s — "
                  f"{len(agent_a_result.poses_chosen)} poses "
                  f"(baseline + custom {agent_a_result.n_custom_renders}), "
                  f"terminated_by={agent_a_result.terminated_by}")
            poses_chosen = agent_a_result.poses_chosen
            image_paths = agent_a_result.image_paths
            _save_agent_a_trace(round_dir / "agent_a_trace.json", agent_a_result)

            if not poses_persisted:
                _save_poses_chosen(run_dir / "poses_chosen.yaml", poses_chosen)
                poses_persisted = True
        else:
            # ===========================================================
            # Round 2+: re-render poses_chosen at current shifted state
            # ===========================================================
            print(f"  re-rendering {len(poses_chosen)} poses at current state ...")
            t0 = time.time()
            image_paths = []
            for i, pose in enumerate(poses_chosen):
                ext = f"_{pose.label}" if pose.label else f"_{i:02d}"
                out_path = images_dir / f"{pose.kind}{ext}_f{pose.frame:04d}.jpg"
                rerender_pose(pose, frame_cache, scene.bundle.pickup_object, out_path)
                image_paths.append(out_path)
            print(f"  rerender done in {time.time()-t0:.1f}s")

        # ----- Agent B one-shot shift decision (rounds 1+) -----
        scene_state = get_scene_state_at_frame(frame_cache, scene.bundle.pickup_object)
        scene_state["pickup_name"] = scene.bundle.pickup_object

        print(f"  Agent B (shift_decision) ...")
        t0 = time.time()
        decision = decide_shift(
            images=image_paths,
            probe_result=probe_result,
            cumulative_shift_mm=cumulative_mm,
            scene_state=scene_state,
            history=history,
        )
        elapsed = time.time() - t0
        print(f"  Agent B done in {elapsed:.1f}s — "
              f"decision={decision.decision}, "
              f"shift_mm={decision.shift_mm}")
        if decision.reasoning:
            print(f"    reasoning: {decision.reasoning[:200]}")
        if decision.parse_error:
            print(f"    parse_error: {decision.parse_error}")

        clipped_shift = _clip_per_round(decision.shift_mm, per_round_cap_mm)
        new_cum, was_clipped = _accumulate_with_cap(
            cumulative_mm, clipped_shift, cumulative_cap_mm,
        )
        _save_agent_b_decision(
            round_dir / "agent_b_decision.json", decision, clipped_shift, new_cum,
        )

        history.append(RoundSummary(
            round_idx=round_idx,
            cumulative_shift_mm=cumulative_mm,
            peak_lift_mm=probe_result.peak_lift_m * 1000,
            closest_distance_mm=probe_result.closest_distance_m * 1000,
            decision=decision.decision,
            reasoning_summary=(decision.reasoning[:80] if decision.reasoning
                                else f"({decision.decision} no reasoning)"),
        ))

        if decision.decision == "success":
            outcome = "success"
            print(f"  [SUCCESS] Agent B declared success at round {round_idx}")
            break
        if decision.decision == "give_up":
            outcome = "give_up"
            print(f"  [GIVE_UP] Agent B gave up at round {round_idx}")
            break
        if was_clipped:
            print(f"  cumulative cap hit on at least one axis "
                  f"(would have been {tuple(c + d for c, d in zip(cumulative_mm, clipped_shift))}, "
                  f"clipped to {new_cum}); ending loop")
            outcome = "cap_exceeded"
            cumulative_mm = new_cum
            break

        cumulative_mm = new_cum
        # Snapshot summary after every round (so partial state survives crashes)
        _save_summary(run_dir / "summary.yaml", GraspLoopResult(
            outcome="(in_progress)",
            n_rounds=round_idx + 1,
            final_cumulative_shift_mm=cumulative_mm,
            final_probe=probe_result,
            poses_chosen=poses_chosen,
            history=history,
            run_dir=run_dir,
        ))

    result = GraspLoopResult(
        outcome=outcome,
        n_rounds=len(history),
        final_cumulative_shift_mm=cumulative_mm,
        final_probe=final_probe,
        poses_chosen=poses_chosen,
        history=history,
        run_dir=run_dir,
    )
    _save_summary(run_dir / "summary.yaml", result)
    print(f"\n[stage3-loop] OUTCOME = {outcome}, rounds={len(history)}, "
          f"final_cumulative={cumulative_mm} mm")
    print(f"[stage3-loop] summary: {run_dir / 'summary.yaml'}")
    return result


__all__ = ["grasp_loop", "GraspLoopResult"]
