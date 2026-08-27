"""Top-level DROID Real2Sim pipeline entrypoint.

This is the only full-pipeline DROID runner:

    python -m ar2s.run_pipeline --raw-data <staged_droid_raw>

It replaces the old three-command visual / physical-prior / sysid chain.

A deterministic sequential runner executes the fixed stage order:

    visual_processing -> geometry_prior -> scene_view_repair
    -> physical_prior -> scene_prep -> grasp_optimization

The stage tools own the real work and preserve the existing stage contracts;
LLM/VLM intelligence lives inside the stages (sub-ReAct agents + VLM
subagents), never in the top-level control flow.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import tool

from ar2s.agent_configs import set_config_path
from ar2s.agent_configs.models import get_grasp_optimization_config
from ar2s.agents_sysid._gl import configure_headless_gl

# Must run before any import below that pulls in `mujoco` — MuJoCo's GL
# backend is locked in at `import mujoco` time (it reads MUJOCO_GL then,
# not when a Renderer is later constructed), so calling this later (e.g.
# in main()) has no effect on a backend already imported elsewhere.
configure_headless_gl()

from ar2s.agents_sysid.grasp_sweep import (
    N_SAMPLES_DEFAULT,
    N_WORKERS_DEFAULT,
    SAMPLE_SEED_DEFAULT,
    GraspSweepResult,
    grasp_sweep,
)
from ar2s.agents_sysid.grasp_loop import GraspLoopResult, grasp_loop
from ar2s.agents_visual.inputs import VisualInput
from ar2s.agents_visual.raw_episode import build_raw_episode, build_visual_input
from ar2s.pipeline_artifacts import (
    episode_bundle_dir,
    pipeline_run_dir,
    run_id_for,
)
from ar2s.pipeline_cleanup import cleanup_run, format_report


# Per-stage LangGraph recursion_limits (the top level itself is a plain
# deterministic loop and needs none).
_DEFAULT_VISUAL_MAX_ITERATIONS = 50
_DEFAULT_PHYSICAL_PRIOR_MAX_ITERATIONS = 16
_DEFAULT_SCENE_PREP_MAX_ITERATIONS = 10

# Fixed stage order. ``--start-from`` skips stages before its value;
# ``--until`` stops after its value (inclusive). Both flags are independent
# but compose: ``--start-from X --until X`` runs only stage X.
_STAGE_ORDER = [
    "visual_processing",
    "geometry_prior",
    "scene_view_repair",
    "physical_prior",
    "scene_prep",
    "grasp_optimization",
]


@dataclass
class PipelineConfig:
    visual_input: VisualInput
    episode_id: str
    run_id: str
    run_root: Path
    visual_max_iterations: int = _DEFAULT_VISUAL_MAX_ITERATIONS
    physical_prior_max_iterations: int = _DEFAULT_PHYSICAL_PRIOR_MAX_ITERATIONS
    scene_prep_max_iterations: int = _DEFAULT_SCENE_PREP_MAX_ITERATIONS
    coacd_threshold: float = 0.05
    coacd_force: bool = False
    optimize_robot_base: bool = False
    n_samples: int = N_SAMPLES_DEFAULT
    sample_seed: int = SAMPLE_SEED_DEFAULT
    reference_frame: int | None = None
    n_workers: int | None = N_WORKERS_DEFAULT
    # Robot<->all-object contact in the sweep. Pipeline default ON per
    # review decision; sweep results are not comparable with runs made
    # before this default changed.
    full_collision: bool = True
    tip_offset_m: float = 0.0
    save_success_videos: bool = True
    save_usd: bool = True
    artifacts_root: Path | None = None
    grasp_strategy: str = "sweep"


@dataclass
class PipelineSink:
    run_root: Path | None = None
    episode_dir: Path | None = None
    physical_priors_path: Path | None = None
    scene_prep_ok: bool = False
    grasp_result: GraspSweepResult | GraspLoopResult | None = None
    agent_final: str = ""
    errors: list[str] = field(default_factory=list)


def _load_input_module(input_ref: str):
    """Load an input module from either a filesystem path or import name."""
    path = Path(input_ref).expanduser()
    if path.exists() or input_ref.endswith(".py"):
        path = path.resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None, f"cannot import {path}"
        mod = importlib.util.module_from_spec(spec)
        # Don't leave a __pycache__ next to the entry stub inside run_root.
        prev_dont_write = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = prev_dont_write
        return mod
    return importlib.import_module(input_ref)


def _load_visual_input(input_ref: str) -> VisualInput:
    mod = _load_input_module(input_ref)
    assert hasattr(mod, "VISUAL_INPUT"), f"{input_ref} does not export VISUAL_INPUT"
    visual_input = mod.VISUAL_INPUT
    assert isinstance(visual_input, VisualInput), (
        f"{input_ref}.VISUAL_INPUT is not a VisualInput "
        f"(got {type(visual_input).__name__})"
    )
    return visual_input


def _last_message_text(agent_result: dict) -> str:
    msgs = agent_result.get("messages") or []
    if not msgs:
        return ""
    content = getattr(msgs[-1], "content", "") or ""
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    return str(content)


def _grasp_strategy_from_config() -> str:
    cfg = get_grasp_optimization_config()
    strategy = str(cfg["sweep_or_loop"])
    assert strategy in ("sweep", "loop"), (
        "stages.grasp_optimization.sweep_or_loop must be 'sweep' or 'loop'"
    )
    return strategy


def _build_tools(
    config: PipelineConfig,
    sink: PipelineSink,
    stages_to_include: list[str] | None = None,
) -> list:
    @tool
    def visual_processing(stage_request: str = "run") -> str:
        """Run visual data_ingest, the visual ReAct agent, and finalize emit."""
        from ar2s.agents_visual.orchestrator import (
            data_ingest,
            finalize,
            run_visual,
        )

        try:
            run_root = Path(
                data_ingest(config.visual_input, run_root=str(config.run_root))
            )
            visual_result = run_visual(run_root, max_iterations=config.visual_max_iterations)
            visual_final = _last_message_text(visual_result)
            if not visual_final.startswith("Pipeline complete"):
                raise RuntimeError(
                    "visual ReAct agent did not reach pipeline completion: "
                    f"{visual_final or '(no terminal message; likely hit max_iterations)'}"
                )
            episode_dir = Path(finalize(run_root))
        except Exception as e:  # noqa: BLE001 - tool boundary reports to ReAct.
            msg = f"{type(e).__name__}: {e}"
            sink.errors.append(f"visual_processing: {msg}")
            return json.dumps({"ok": False, "error": msg})

        sink.run_root = run_root
        sink.episode_dir = episode_dir
        return json.dumps({
            "ok": True,
            "run_root": str(run_root),
            "episode_dir": str(episode_dir),
        })

    @tool
    def geometry_prior(stage_request: str = "run") -> str:
        """Run Stage 0 geometry priors (mesh_orient + mesh_axis_align, scheme C).

        Patches the visual-emitted episode bundle in place: rewrites
        objects/<name>/visual.obj when mesh_orient flips, picks each
        object's world-up orientation via the 6-candidate axis_align VLM,
        and updates manifest.yaml's ground.offset + per-object pos_offset
        entries (roots stay put, children snap to parent tops). Pure
        post-processing — no sub-ReAct. See agents_geometry_prior/apply.py.
        """
        if sink.run_root is None or sink.episode_dir is None:
            msg = "visual_processing has not produced run_root and episode_dir"
            sink.errors.append(f"geometry_prior: {msg}")
            return json.dumps({"ok": False, "error": msg})
        from ar2s.agents_geometry_prior.apply import apply_geometry_priors

        try:
            report = apply_geometry_priors(
                episode_dir=sink.episode_dir,
                run_root=str(sink.run_root),
            )
        except Exception as e:  # noqa: BLE001 - tool boundary reports to ReAct.
            msg = f"{type(e).__name__}: {e}"
            sink.errors.append(f"geometry_prior: {msg}")
            return json.dumps({"ok": False, "error": msg})

        return json.dumps({
            "ok": True,
            "orient_flipped": sum(
                1 for o in report.orient if o.chosen_flip != "NONE"
            ),
            "axis_snapped": sum(1 for a in report.axis_align if a.applied),
            "geometry_priors_path": (
                str(report.geometry_priors_path)
                if report.geometry_priors_path else None
            ),
        })

    @tool
    def scene_view_repair(stage_request: str = "run") -> str:
        """Two-view scene repair after geometry_prior.

        Detects (P1) objects that never resolved a mask and (P2) VLM-visible
        real-vs-sim mismatches across both calibrated views. When problems
        exist, rebuilds the ENTIRE scene from the secondary view (swapped
        primary attribute over the same role-keyed raw files) in-process,
        then per problem object asks a VLM which view's reconstruction is
        better, swapping mesh+scale+pose into the main bundle and
        re-grounding its z on the support-tree root. No-op (ok=true,
        repaired=[]) when both detectors pass.
        """
        if sink.run_root is None or sink.episode_dir is None:
            msg = "visual_processing has not produced run_root and episode_dir"
            sink.errors.append(f"scene_view_repair: {msg}")
            return json.dumps({"ok": False, "error": msg})
        from ar2s.agents_visual.subagents.scene_view_repair import run_repair

        try:
            report = run_repair(
                run_root=sink.run_root,
                episode_dir=sink.episode_dir,
                visual_input=config.visual_input,
                visual_max_iterations=config.visual_max_iterations,
            )
        except Exception as e:  # noqa: BLE001
            # Best-effort stage: the main scene is intact whether or not the
            # repair ran, and there is no --skip flag, so a failure here must
            # not halt physical_prior / scene_prep / grasp_optimization. Same
            # degradation contract the v2 build already uses internally
            # (v2_build_error); recorded in sink.errors so the run still
            # reports non-zero at the end.
            msg = f"{type(e).__name__}: {e}"
            sink.errors.append(f"scene_view_repair: {msg}")
            print(f"[scene_view_repair] degraded (continuing): {msg}", file=sys.stderr)
            return json.dumps({"ok": True, "repaired": [], "error": msg})
        return json.dumps({"ok": True, **report})

    @tool
    def physical_prior(stage_request: str = "run") -> str:
        """Run the physical-prior ReAct agent on the visual-emitted episode."""
        if sink.run_root is None or sink.episode_dir is None:
            msg = "visual_processing has not produced run_root and episode_dir"
            sink.errors.append(f"physical_prior: {msg}")
            return json.dumps({"ok": False, "error": msg})
        from ar2s.agents_physical_prior.orchestrator import (
            data_ingest,
            run_physical_prior,
        )

        try:
            physical_input = data_ingest(sink.episode_dir, run_root=sink.run_root)
            report = run_physical_prior(
                physical_input,
                max_iterations=config.physical_prior_max_iterations,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            sink.errors.append(f"physical_prior: {msg}")
            return json.dumps({"ok": False, "error": msg})

        sink.physical_priors_path = report.physical_priors_path
        return json.dumps({
            "ok": True,
            "classified": len(report.classified),
            "skipped": len(report.skipped),
            "manifest_patched_for": report.manifest_patched_for,
            "physical_priors_path": (
                str(report.physical_priors_path)
                if report.physical_priors_path else None
            ),
        })

    @tool
    def scene_prep(stage_request: str = "run") -> str:
        """Run the scene-prep ReAct agent: collision_prep then calibrate_scene."""
        if sink.episode_dir is None or sink.physical_priors_path is None:
            msg = "physical_prior has not completed and recorded physical_priors_path"
            sink.errors.append(f"scene_prep: {msg}")
            return json.dumps({"ok": False, "error": msg})
        from ar2s.agents_sysid.orchestrator import (
            ScenePrepConfig,
            data_ingest,
            run_scene_prep,
        )

        try:
            episode_dir = data_ingest(sink.episode_dir)
            result = run_scene_prep(
                episode_dir,
                config=ScenePrepConfig(
                    coacd_threshold=config.coacd_threshold,
                    coacd_force=config.coacd_force,
                    optimize_robot_base=config.optimize_robot_base,
                ),
                max_iterations=config.scene_prep_max_iterations,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            sink.errors.append(f"scene_prep: {msg}")
            return json.dumps({"ok": False, "error": msg})

        sink.scene_prep_ok = result.ok
        if result.sink.errors:
            sink.errors.extend(f"scene_prep: {err}" for err in result.sink.errors)
        return json.dumps({
            "ok": bool(result.ok),
            "agent_final": result.agent_final,
            "errors": result.sink.errors,
            "calibration_path": str(Path(episode_dir) / "calibration.yaml"),
        })

    @tool
    def grasp_optimization(stage_request: str = "run") -> str:
        """Run the YAML-selected grasp optimization strategy."""
        if sink.episode_dir is None or not sink.scene_prep_ok:
            msg = "scene_prep has not completed successfully"
            sink.errors.append(f"grasp_optimization: {msg}")
            return json.dumps({"ok": False, "error": msg})

        artifacts_root = (
            config.artifacts_root
            if config.artifacts_root is not None
            else config.run_root / "artifacts"
        )
        try:
            if config.grasp_strategy == "sweep":
                result = grasp_sweep(
                    sink.episode_dir,
                    artifacts_root=artifacts_root,
                    n_samples=config.n_samples,
                    sample_seed=config.sample_seed,
                    reference_frame=config.reference_frame,
                    save_usd=config.save_usd,
                    n_workers=config.n_workers,
                    tip_offset_m=config.tip_offset_m,
                    save_success_videos=config.save_success_videos,
                    full_collision=config.full_collision,
                )
                payload = {
                    "ok": result.n_grasped > 0,
                    "strategy": "sweep",
                    **({} if result.n_grasped > 0 else {
                        "error": (
                            f"no successful grasps in sweep "
                            f"(0/{result.n_samples} samples grasped)"
                        ),
                    }),
                    "n_grasped": result.n_grasped,
                    "n_samples": result.n_samples,
                    "best_sample_idx": result.best_sample_idx,
                    "run_dir": str(result.run_dir),
                }
            elif config.grasp_strategy == "loop":
                result = grasp_loop(
                    sink.episode_dir,
                    artifacts_root=artifacts_root,
                    save_usd=config.save_usd,
                )
                payload = {
                    "ok": result.outcome == "success",
                    "strategy": "loop",
                    "outcome": result.outcome,
                    "n_rounds": result.n_rounds,
                    "final_cumulative_shift_mm": result.final_cumulative_shift_mm,
                    "run_dir": str(result.run_dir),
                }
            else:
                raise AssertionError(
                    f"unsupported grasp_strategy {config.grasp_strategy!r}"
                )
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            sink.errors.append(f"grasp_optimization: {msg}")
            return json.dumps({"ok": False, "error": msg})

        sink.grasp_result = result
        return json.dumps(payload)

    all_tools = {
        "visual_processing": visual_processing,
        "geometry_prior": geometry_prior,
        "scene_view_repair": scene_view_repair,
        "physical_prior": physical_prior,
        "scene_prep": scene_prep,
        "grasp_optimization": grasp_optimization,
    }
    if stages_to_include is None:
        stages_to_include = _STAGE_ORDER
    return [all_tools[s] for s in stages_to_include]


def _validate_stage_range(
    start_from: str | None, until: str | None,
) -> tuple[int, int]:
    """Resolve ``--start-from``/``--until`` to inclusive ``[start_idx, end_idx]``.

    Defaults: full range ``[0, len(_STAGE_ORDER)-1]``. Validates ordering
    (start_from must not be after until). Returns indices into _STAGE_ORDER.
    """
    start_idx = 0 if start_from is None else _STAGE_ORDER.index(start_from)
    end_idx = len(_STAGE_ORDER) - 1 if until is None else _STAGE_ORDER.index(until)
    if start_idx > end_idx:
        raise ValueError(
            f"--start-from {start_from!r} is after --until {until!r}; "
            f"stage order is {_STAGE_ORDER}"
        )
    return start_idx, end_idx


def _prepopulate_sink_from_disk(
    sink: PipelineSink,
    run_root: Path,
    episode_dir: Path,
    start_from: str,
) -> None:
    """Fill sink fields from on-disk artifacts left by earlier stages.

    Strict: every skipped stage must have its canonical output present at
    the expected path or this raises ``FileNotFoundError`` with a message
    naming the missing artifact. This prevents silently running a later
    stage on an incomplete bundle.
    """
    start_idx = _STAGE_ORDER.index(start_from)

    if start_idx >= 1:  # visual_processing was skipped → need its outputs
        state_path = run_root / "state.json"
        manifest_path = episode_dir / "manifest.yaml"
        if not state_path.exists():
            raise FileNotFoundError(
                f"--start-from {start_from!r}: visual_processing artifact "
                f"missing — {state_path}"
            )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"--start-from {start_from!r}: visual_processing artifact "
                f"missing — {manifest_path} (need the sysid_inputs/ "
                f"bundle visual_processing emits)"
            )
        sink.run_root = run_root
        sink.episode_dir = episode_dir

    if start_idx >= 2:  # geometry_prior was skipped
        gp_path = episode_dir / "geometry_priors.json"
        if not gp_path.exists():
            raise FileNotFoundError(
                f"--start-from {start_from!r}: geometry_prior artifact "
                f"missing — {gp_path}"
            )

    if start_idx >= 3:  # scene_view_repair was skipped
        # Repair is conditional (no-ops when both detectors pass) and was
        # introduced 2026-07 — legacy bundles predate its report. Warn only.
        repair_report = run_root / "scene_view_repair" / "report.json"
        if not repair_report.exists():
            print(f"[resume] note: no scene_view_repair report at "
                  f"{repair_report} (legacy bundle or repair never ran)")

    if start_idx >= 4:  # physical_prior was skipped
        pp_path = episode_dir / "physical_priors.json"
        if not pp_path.exists():
            raise FileNotFoundError(
                f"--start-from {start_from!r}: physical_prior artifact "
                f"missing — {pp_path}"
            )
        sink.physical_priors_path = pp_path

    if start_idx >= 5:  # scene_prep was skipped
        calib_path = episode_dir / "calibration.yaml"
        if not calib_path.exists():
            raise FileNotFoundError(
                f"--start-from {start_from!r}: scene_prep artifact "
                f"missing — {calib_path}"
            )
        sink.scene_prep_ok = True


def run_pipeline_agent(
    config: PipelineConfig,
    *,
    start_from: str | None = None,
    until: str | None = None,
) -> PipelineSink:
    """Run the DROID pipeline as a DETERMINISTIC sequential runner.

    The stage order is fixed (see ``_STAGE_ORDER``) and every stage runs
    exactly once — there is no decision for an LLM to make here, so the
    former top-level ReAct loop was pure overhead plus a failure mode: its
    recursion budget killed full 6-stage runs after scene_view_repair was
    added ("Sorry, need more steps", REAL_05_12 + REAL_05_31, 2026-07-06).
    Intelligence stays in the leaves (visual / physical / scene_prep
    sub-agents and the VLM subagents inside the stage tools).

    ``start_from``/``until`` (both optional) restrict which stages run.
    ``start_from`` additionally pre-populates sink fields from on-disk
    artifacts left by the skipped stages (strict: raises
    ``FileNotFoundError`` if any expected artifact is missing).
    Any stage returning ``ok=false`` halts the run; subsequent stages are
    skipped and the failure is recorded in ``sink.agent_final``.
    """
    start_idx, end_idx = _validate_stage_range(start_from, until)
    stages_to_run = _STAGE_ORDER[start_idx:end_idx + 1]
    stages_skipped = _STAGE_ORDER[:start_idx]

    sink = PipelineSink()
    if stages_skipped:
        # Canonical episode_dir: <run_root>/sysid_inputs, or the legacy
        # <episode_id>_agent/ child when an older run made one. Resolved rather
        # than built from run_id — finalize() never named the bundle after the
        # run id, so a --run-suffix run used to look for a dir never written.
        episode_dir = episode_bundle_dir(config.run_root)
        _prepopulate_sink_from_disk(
            sink, config.run_root, episode_dir, start_from,
        )
        print(f"[runner] resuming; skipped stages: {stages_skipped}")

    tools = _build_tools(config, sink, stages_to_include=stages_to_run)
    for stage_name, stage_tool in zip(stages_to_run, tools):
        print()
        print(f"[runner] ── stage: {stage_name} ──")
        t0 = time.time()
        raw = stage_tool.invoke({"stage_request": "run"})
        dt = time.time() - t0
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            payload = {"ok": False, "error": f"non-JSON stage return: {raw!r:.200}"}
        if not payload.get("ok", False):
            err = payload.get("error", "(no error message)")
            sink.agent_final = (
                f"DROID pipeline halted at {stage_name}: {err}"
            )
            print(f"[runner] {stage_name} FAILED after {dt:.1f}s: {err}")
            return sink
        print(f"[runner] {stage_name} ok ({dt:.1f}s)")

    if stages_to_run[-1] != _STAGE_ORDER[-1]:
        sink.agent_final = (
            f"DROID pipeline halted after {stages_to_run[-1]} (as requested)."
        )
    else:
        sink.agent_final = "DROID pipeline complete: all stages ok."
    return sink


def _build_or_load_visual_input(args: argparse.Namespace) -> VisualInput:
    """Resolve the VisualInput. The run id is derived from its episode_id by
    the caller — it is never an independent input."""
    if args.raw_data is not None:
        print()
        print("=" * 70)
        print("Step 0: build raw_episode")
        print("=" * 70)
        build_result = build_raw_episode(
            args.raw_data,
            episode_id=args.episode_id,
            run_suffix=args.run_suffix,
            artifact_root=args.artifact_root,
            forced_camera=args.camera,
            overwrite=args.overwrite_raw,
            link_source={"auto": None, "always": True, "never": False}[args.raw_link],
        )
        vi = build_visual_input(build_result)
    else:
        assert args.input is not None, "one of --raw-data or --input is required"
        print(f"[run_pipeline] loading visual entry: {args.input}")
        vi = _load_visual_input(args.input)

    # CLI override: cap ONLY the heavy per-frame models (FoundationStereo depth
    # + FoundationPose tracking) without touching svo_extract/discovery/segment.
    if args.heavy_max_frames is not None:
        vi.heavy_max_frames = args.heavy_max_frames
        print(f"[run_pipeline] heavy_max_frames override -> {vi.heavy_max_frames} "
              f"(caps stereo_depth + pose_tracking only)")
    # CLI override: downscale FoundationPose's input. register() warps ~240 pose
    # hypotheses at full image resolution, so a wide sensor OOMs where 720p fits.
    if args.pose_shorter_side is not None:
        vi.pose_shorter_side = args.pose_shorter_side
        print(f"[run_pipeline] pose_shorter_side override -> {vi.pose_shorter_side} "
              f"(FoundationPose input downscale)")
    return vi


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full DROID Real2Sim pipeline.",
    )
    p.add_argument("--input", default=None,
                   help="Python file or module exporting VISUAL_INPUT.")
    p.add_argument("--raw-data", default=None, type=Path,
                   help="DROID-style raw_data folder.")
    p.add_argument("--episode-id", default=None,
                   help="DROID-100 episode id when using --raw-data.")
    p.add_argument("--camera", default="auto",
                   help="With --raw-data: auto, ext1, ext2, or serial.")
    p.add_argument("--heavy-max-frames", type=int, default=None,
                   help="Cap ONLY the per-frame heavy models (FoundationStereo "
                        "depth + FoundationPose tracking) to the first N frames. "
                        "svo_extract/object_discovery/segment/mesh_recover still "
                        "run over the full sequence. Handy for quick validation.")
    p.add_argument("--pose-shorter-side", type=int, default=None,
                   help="Resize frames so min(H,W) == N before FoundationPose. "
                        "register() warps ~240 pose hypotheses at full image "
                        "resolution, so a 2208x1242 sensor needs ~3x the memory "
                        "of DROID's 1280x720 and OOMs a 32 GB card; 720 matches "
                        "the DROID footprint. K is rescaled with the image, so "
                        "geometry is unchanged. Default: native resolution.")
    p.add_argument("--overwrite-raw", action="store_true",
                   help="Replace raw_episodes/ and visual input artifacts.")
    p.add_argument("--raw-link", default="auto", choices=("auto", "always", "never"),
                   help=(
                       "With --raw-data: whether raw_episodes symlinks the "
                       "verbatim inputs (mp4s, K sidecars, trajectory backup) "
                       "instead of copying them. 'auto' (default) links only "
                       "when --raw-data is NOT under job-scoped scratch "
                       "($SLURM_TMPDIR, $TMPDIR, /localscratch, /tmp, ...), so "
                       "passing a persistent path skips the copy entirely. "
                       "Derived files are always written for real."
                   ))
    p.add_argument("--run-suffix", default=None,
                   help=(
                       "Optional tag distinguishing config variants of the SAME "
                       "episode; the run dir becomes "
                       "<artifact-root>/<episode_id>_<suffix>/ (e.g. "
                       "--run-suffix gemma4_nopyzed). The episode id always "
                       "prefixes the run dir, so one run dir holds one episode "
                       "by construction. Use --artifact-root to relocate runs."
                   ))
    p.add_argument("--artifact-root", default=None, type=Path,
                   help="Root containing per-run outputs/run_pipeline artifacts.")
    p.add_argument("--agent-config", default=None, type=Path,
                   help="YAML agent model config.")
    p.add_argument("--visual-max-iterations", type=int,
                   default=_DEFAULT_VISUAL_MAX_ITERATIONS,
                   help="LangGraph recursion_limit for visual agent.")
    p.add_argument("--physical-prior-max-iterations", type=int,
                   default=_DEFAULT_PHYSICAL_PRIOR_MAX_ITERATIONS,
                   help="LangGraph recursion_limit for physical-prior agent.")
    p.add_argument("--scene-prep-max-iterations", type=int,
                   default=_DEFAULT_SCENE_PREP_MAX_ITERATIONS,
                   help="LangGraph recursion_limit for scene-prep agent.")
    p.add_argument("--no-full-collision", action="store_true",
                   help="Restrict sweep collision to pickup+ground (legacy "
                        "behaviour; default is robot vs ALL objects).")
    p.add_argument("--coacd-threshold", type=float, default=0.05,
                   help="Scene-prep CoACD concavity threshold.")
    p.add_argument("--coacd-force", action="store_true",
                   help="Re-run CoACD even when collision pieces exist.")
    p.add_argument("--optimize-robot-base", action="store_true",
                   help="Run robot-base IoU optimizer during calibration.")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT,
                   help="grasp_sweep uniform surface samples.")
    p.add_argument("--sample-seed", type=int, default=SAMPLE_SEED_DEFAULT,
                   help="grasp_sweep sampling seed.")
    p.add_argument("--reference-frame", type=int, default=None,
                   help="grasp_sweep trajectory frame override.")
    p.add_argument("--n-workers", type=int, default=N_WORKERS_DEFAULT,
                   help=f"grasp_sweep worker count (default {N_WORKERS_DEFAULT}).")
    p.add_argument("--tip-offset-m", type=float, default=0.0,
                   help="grasp_sweep pad-midpoint offset toward gripper tip.")
    p.add_argument("--no-success-videos", action="store_true",
                   help="Skip per-success grasp MP4 renders.")
    p.add_argument("--no-usd", action="store_true",
                   help="Disable USD export for grasp_sweep.")
    p.add_argument("--artifacts-root", default=None, type=Path,
                   help="Parent directory for grasp_sweep artifacts. Defaults "
                        "to <run_root>/artifacts.")
    p.add_argument("--start-from", default=None, choices=_STAGE_ORDER,
                   help=(
                       "Skip stages BEFORE this one. Sink fields are "
                       "pre-populated from on-disk artifacts of the skipped "
                       "stages (strict: missing artifact raises). Typical "
                       "split: 5090 runs '--until visual_processing', local "
                       "runs '--start-from geometry_prior'."
                   ))
    p.add_argument("--until", default=None, choices=_STAGE_ORDER,
                   help=(
                       "STOP after this stage completes. Later stages are "
                       "removed from the agent's tool list so they cannot be "
                       "invoked. Composes with --start-from to run a single "
                       "stage in isolation."
                   ))
    p.add_argument("--no-cleanup", action="store_true",
                   help=(
                       "Skip the post-run consolidation of run_root (depth "
                       "packing, sweep-sample archiving, track_vis stitching, "
                       "mask/frame pruning). See ar2s.pipeline_cleanup."
                   ))
    p.add_argument("--cleanup-dry-run", action="store_true",
                   help="Report what the post-run cleanup would do, change nothing.")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    set_config_path(args.agent_config)
    configure_headless_gl()

    if args.raw_data is None and args.input is None:
        print("[run_pipeline] one of --raw-data or --input is required", file=sys.stderr)
        return 2
    if args.raw_data is not None and args.input is not None:
        print("[run_pipeline] --raw-data and --input are mutually exclusive", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    try:
        visual_input = _build_or_load_visual_input(args)
        episode_id = visual_input.episode_id
        run_id = run_id_for(episode_id, args.run_suffix)
        run_root = pipeline_run_dir(run_id, args.artifact_root).expanduser().resolve()
        config = PipelineConfig(
            visual_input=visual_input,
            episode_id=episode_id,
            run_id=run_id,
            run_root=run_root,
            visual_max_iterations=args.visual_max_iterations,
            physical_prior_max_iterations=args.physical_prior_max_iterations,
            scene_prep_max_iterations=args.scene_prep_max_iterations,
            coacd_threshold=args.coacd_threshold,
            coacd_force=args.coacd_force,
            optimize_robot_base=args.optimize_robot_base,
            n_samples=args.n_samples,
            sample_seed=args.sample_seed,
            reference_frame=args.reference_frame,
            n_workers=args.n_workers,
            tip_offset_m=args.tip_offset_m,
            save_success_videos=not args.no_success_videos,
            save_usd=not args.no_usd,
            artifacts_root=(
                args.artifacts_root.expanduser().resolve()
                if args.artifacts_root is not None else None
            ),
            grasp_strategy=_grasp_strategy_from_config(),
            full_collision=not args.no_full_collision,
        )
        sink = run_pipeline_agent(
            config,
            start_from=args.start_from,
            until=args.until,
        )
    except Exception as e:
        print(f"[run_pipeline] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    if sink.agent_final:
        print(f"\n[run_pipeline] agent final: {sink.agent_final[:600]}")
    for err in sink.errors:
        print(f"[run_pipeline] stage error: {err}", file=sys.stderr)

    print()
    print("=" * 70)
    print("DROID pipeline summary")
    print("=" * 70)
    print(f"run_root        : {sink.run_root}")
    print(f"episode_dir     : {sink.episode_dir}")
    print(f"physical_priors : {sink.physical_priors_path}")
    if sink.episode_dir is not None:
        print(f"calibration     : {sink.episode_dir / 'calibration.yaml'}")
    if sink.grasp_result is not None:
        print(f"grasp_artifacts : {sink.grasp_result.run_dir}")
        if isinstance(sink.grasp_result, GraspSweepResult):
            print("grasp_strategy  : sweep")
            print(f"grasped         : {sink.grasp_result.n_grasped}/{sink.grasp_result.n_samples}")
            print(f"best_sample_idx : {sink.grasp_result.best_sample_idx}")
        else:
            print("grasp_strategy  : loop")
            print(f"outcome         : {sink.grasp_result.outcome}")
            print(f"n_rounds        : {sink.grasp_result.n_rounds}")
    print(f"wall_s          : {time.monotonic() - t0:.1f}")

    # Consolidate run_root before returning. Steps that would destroy inputs a
    # later --start-from still needs are gated inside cleanup_run on which
    # stages actually completed, so this is safe after a partial run too.
    if not args.no_cleanup:
        print()
        stats = cleanup_run(run_root, dry_run=args.cleanup_dry_run)
        print(format_report(run_root, stats, dry_run=args.cleanup_dry_run))

    ok = False
    if isinstance(sink.grasp_result, GraspSweepResult):
        ok = sink.grasp_result.n_grasped > 0 and not sink.errors
    elif isinstance(sink.grasp_result, GraspLoopResult):
        ok = sink.grasp_result.outcome == "success" and not sink.errors
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
