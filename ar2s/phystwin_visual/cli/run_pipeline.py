"""End-to-end CLI for ar2s.phystwin_visual.

Mirrors ``ar2s.agents_visual.cli.run_pipeline`` / ``ar2s.agents_sysid.cli.run_pipeline``:
a two-step pipeline (``data_ingest`` -> ReAct orchestrator), driven from a
single ``PhysTwinInput`` built from CLI flags.

    python -m ar2s.phystwin_visual.cli.run_pipeline \\
        <base_path>/<case_name> --category <text-prompt-category>

``<case_dir>`` is split into ``base_path`` (parent) + ``case_name`` (basename).
The case directory must already exist and contain ``color/``, ``depth/``,
``metadata.json``, ``calibrate.pkl`` (PhysTwin-format raw data).

``--dry-run`` runs ``data_ingest`` and builds the stage tools (surfacing
import/wiring errors) without invoking the ReAct agent or any stage subprocess.

Exit codes: 0 = success, 1 = pipeline halted at a stage, 2 = bad input,
3 = data_ingest failed, 4 = ReAct agent failed to run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ar2s.agent_configs import set_config_path
from ar2s.phystwin_visual.inputs import PhysTwinInput
from ar2s.phystwin_visual.orchestrator import (
    _DEFAULT_MAX_ITERATIONS,
    _StageSink,
    _build_tools,
    _plan,
    data_ingest,
    run_phystwin,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the ar2s.phystwin_visual pipeline end-to-end.",
    )
    parser.add_argument(
        "case_dir",
        help="Path to <base_path>/<case_name> (PhysTwin-format case dir "
             "containing color/, depth/, metadata.json, calibrate.pkl).",
    )
    parser.add_argument(
        "--category", required=True,
        help="Object category for the Grounded-SAM-2 text prompt, e.g. 'sloth'.",
    )
    parser.add_argument(
        "--controller-name", default="hand",
        help="Name of the controller in the segmentation mask (default: hand).",
    )
    parser.add_argument(
        "--shape-prior", action="store_true",
        help="Run shape_alignment and use the aligned mesh in final_export. "
             "Requires --shape-mesh-path. WARNING: data_ingest deletes "
             "<case_dir>/shape/matching/ when this is set.",
    )
    parser.add_argument(
        "--shape-mesh-path", default="",
        help="Path to a pre-generated .glb mesh, copied to "
             "<case_dir>/shape/object.glb. Required with --shape-prior.",
    )
    parser.add_argument(
        "--pipeline-id", default="",
        help="ID for run logs (runs/<pipeline_id>_phystwin/logs). "
             "Defaults to case_name.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.35,
                         help="Grounded-SAM-2 box threshold (default 0.35).")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                         help="Grounded-SAM-2 text threshold (default 0.25).")
    parser.add_argument("--cotracker-query-points", type=int, default=5000,
                         help="Max CoTracker query points (default 5000).")
    parser.add_argument("--num-surface-points", type=int, default=1024,
                         help="final_export surface point count (default 1024).")
    parser.add_argument("--volume-sample-size", type=float, default=0.005,
                         help="final_export volume sample size in meters (default 0.005).")
    parser.add_argument("--camera-indices", type=int, nargs="*", default=[],
                         help="Camera indices to use (default: auto-detect from depth/).")
    parser.add_argument("--agent-config", default=None, type=Path,
                         help="YAML agent model config path. Defaults to "
                              "ar2s/agent_configs/config_default.yaml.")
    parser.add_argument("--max-iterations", type=int, default=_DEFAULT_MAX_ITERATIONS,
                         help=f"LangGraph recursion_limit for the ReAct loop "
                              f"(default {_DEFAULT_MAX_ITERATIONS}).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Run data_ingest + build the stage tools but don't "
                              "invoke the ReAct agent or any stage subprocess.")
    args = parser.parse_args()
    set_config_path(args.agent_config)

    case_dir = Path(args.case_dir).expanduser().resolve()
    inp = PhysTwinInput(
        base_path=str(case_dir.parent),
        case_name=case_dir.name,
        category=args.category,
        controller_name=args.controller_name,
        shape_prior=args.shape_prior,
        shape_mesh_path=args.shape_mesh_path,
        pipeline_id=args.pipeline_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        cotracker_query_points=args.cotracker_query_points,
        num_surface_points=args.num_surface_points,
        volume_sample_size=args.volume_sample_size,
        camera_indices=args.camera_indices,
    )

    print("=" * 70)
    print("Step 1: data_ingest")
    print("=" * 70)
    try:
        inp = data_ingest(inp)
    except ValueError as e:
        print(f"[run_pipeline] data_ingest FAILED: {e}", file=sys.stderr)
        return 3
    print(f"[run_pipeline] base_path={inp.base_path} case_name={inp.case_name} "
          f"category={inp.category} shape_prior={inp.shape_prior}")

    if args.dry_run:
        print()
        print("[run_pipeline] --dry-run: building stage tools to validate imports + wiring")
        plan = _plan(inp)
        tools = _build_tools(inp, _StageSink(), log_dir="/tmp/phystwin_dry_run_logs")
        print(f"[run_pipeline] {len(tools)} tools, plan: {plan}")
        for t in tools:
            print(f"  - {t.name}")
        print("[run_pipeline] dry-run OK")
        return 0

    print()
    print("=" * 70)
    print(f"Step 2: ReAct orchestrator ({len(_plan(inp))} tools)")
    print("=" * 70)
    try:
        result = run_phystwin(inp, max_iterations=args.max_iterations)
    except Exception as e:
        print(f"[run_pipeline] ReAct agent FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 4

    if result.agent_final:
        print(f"\n[run_pipeline] agent final: {result.agent_final[:400]}")
    for err in result.sink.errors:
        print(f"[run_pipeline] stage error: {err}", file=sys.stderr)
    if result.final_data_path:
        print(f"[run_pipeline] final_data: {result.final_data_path}")

    print(f"\n[run_pipeline] {'SUCCESS' if result.ok else 'FAILED'}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
