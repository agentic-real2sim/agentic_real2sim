"""CLI: full phystwin sysid pipeline — Stage 1 + 2 + 3 on one raw case dir.

Usage:

    python -m ar2s.agents_sysid.cli.run_phystwin \
        <base_path>/<case_name>

``<case_name>`` must be a PhysTwin-format raw case dir holding
``final_data.pkl``, ``calibrate.pkl``, ``metadata.json`` and ``color/``
(as produced by ``python -m ar2s.phystwin_visual.cli.run_pipeline``).

Stage breakdown:

    Stage 1: Episode Agent (LLM) scans the raw dir, picks the cloth/real
             config, and writes ``<episode_dir>/phystwin_manifest.yaml``.
    Stage 2: setup_phystwin_scene — builds the Warp sim in the ``phystwin``
             conda env and writes ``<episode_dir>/phystwin_setup.yaml``.
    Stage 3: sysid_agent (LLM dispatcher) → CMA → train → inference, with
             cache skip via phystwin_setup.yaml's cache_state.

An existing ``phystwin_manifest.yaml`` is reused automatically (pass
``--overwrite`` to regenerate it); ``--skip-stage1`` / ``--skip-stage2``
force-skip a stage, and ``--skip-stage3`` stops after sim verification.
When the CLI ``raw_dir`` differs from the manifest's (data moved), the
manifest is retargeted to the CLI path.

Exit code 0 = success; 1 = failure at any stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

from ar2s.agent_configs import set_config_path
from ar2s.pipeline_artifacts import pipeline_run_dir

_MANIFEST_KEYS = ("raw_dir", "case_name", "train_frame", "config_choice")


def _load_manifest(path: Path) -> dict | None:
    """Load phystwin_manifest.yaml; None if unreadable, corrupt, or partial
    (a batch SIGKILL can land mid-write, leaving an empty/truncated file)."""
    try:
        with open(path) as f:
            doc = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(doc, dict) or any(k not in doc for k in _MANIFEST_KEYS):
        return None
    return doc


def _write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the full phystwin sysid pipeline (Stage 1 + 2 + 3).",
    )
    p.add_argument("raw_dir",
                   help="PhysTwin-format raw case dir (final_data.pkl, "
                        "calibrate.pkl, metadata.json, color/).")
    p.add_argument("--episode-dir", default=None,
                   help="Stage 1 output dir (default: "
                        "outputs/run_pipeline/<run_id>/sysid_inputs/<raw_dir.name>).")
    p.add_argument("--run-id", default=None,
                   help="Pipeline run id (default: raw_dir basename).")
    p.add_argument("--artifact-root", default=None, type=Path,
                   help="Root containing per-run pipeline artifacts. Defaults "
                        "to outputs/run_pipeline or AR2S_RUN_PIPELINE_ROOT.")
    p.add_argument("--scene-name", default=None,
                   help="Override scene_name (default = raw_dir basename).")
    p.add_argument("--agent-config", default=None, type=Path,
                   help="YAML model/runtime config for the Stage 1/3 agents.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite an existing phystwin_manifest.yaml.")
    p.add_argument("--skip-stage1", action="store_true",
                   help="Skip the LLM Episode Agent; requires an existing "
                        "phystwin_manifest.yaml in the episode dir.")
    p.add_argument("--skip-stage2", action="store_true",
                   help="Skip sim initialization; requires an existing "
                        "phystwin_setup.yaml in the episode dir.")
    p.add_argument("--skip-stage3", action="store_true",
                   help="Stop after Stage 2 (no CMA/train/inference).")
    p.add_argument("--device", default="cuda:0",
                   help="CUDA device for the Warp sim subprocesses "
                        "(default cuda:0).")
    args = p.parse_args()
    set_config_path(args.agent_config)

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    if not raw_dir.is_dir():
        print(f"[run_phystwin] ERROR: {raw_dir} is not a directory",
              file=sys.stderr)
        sys.exit(1)

    run_id = args.run_id or raw_dir.name
    run_dir = pipeline_run_dir(run_id, args.artifact_root)
    episode_dir = Path(
        args.episode_dir or (run_dir / "sysid_inputs" / raw_dir.name)
    ).expanduser().resolve()
    log_dir = episode_dir / "logs"
    t_pipeline = time.time()

    print("=" * 70)
    print("phystwin sysid pipeline")
    print(f"  raw_dir     = {raw_dir}")
    print(f"  episode_dir = {episode_dir}")
    print(f"  stage1      = {'skip' if args.skip_stage1 else 'run'}  (episode agent)")
    print(f"  stage2      = {'skip' if args.skip_stage2 else 'run'}  (sim init)")
    print(f"  stage3      = {'skip' if args.skip_stage3 else 'run'}  (sysid agent)")
    print("=" * 70)

    # ----- Stage 1: Episode Agent + write manifest to disk -----
    manifest_path = episode_dir / "phystwin_manifest.yaml"
    manifest = _load_manifest(manifest_path) if manifest_path.exists() else None
    if args.skip_stage1 or (manifest is not None and not args.overwrite):
        if manifest is None:
            print(f"\n[stage 1] ERROR: {manifest_path} is missing or corrupt; "
                  f"cannot skip stage 1 (delete it or drop --skip-stage1).",
                  file=sys.stderr)
            sys.exit(1)
        print("\n[stage 1] skipped — reusing existing phystwin_manifest.yaml"
              + ("" if args.skip_stage1 else " (pass --overwrite to regenerate)"))
    else:
        from ar2s.agents_sysid.agents.episode_agent import run_episode_agent
        from ar2s.agents_sysid.phystwin.build import write_phystwin_manifest

        print("\n[stage 1] phystwin Episode Agent: scan raw + write manifest")
        t0 = time.time()
        try:
            scan_result = run_episode_agent(
                raw_dir, scene_name=args.scene_name, kind="phystwin",
            )
            # overwrite unconditionally: reaching here with an existing file
            # means either --overwrite was passed or the file is corrupt.
            report = write_phystwin_manifest(
                scan_result, target_dir=episode_dir, overwrite=True,
            )
        except Exception as e:
            print(f"[stage 1] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
        m = scan_result.manifest
        print(f"[stage 1] done in {time.time()-t0:.1f}s — "
              f"case_name={m['case_name']}, config={m['config_choice']}, "
              f"train_frame={m['train_frame']}, manifest={report.manifest_path.name}")
        manifest = _load_manifest(manifest_path)
        if manifest is None:
            print(f"[stage 1] ERROR: {manifest_path} unreadable after write.",
                  file=sys.stderr)
            sys.exit(1)

    # The manifest's raw_dir goes stale when the data moves (another machine
    # or mount); the CLI argument is validated above, so it wins — for Stage 2
    # here and, via the rewritten manifest, for Stage 3's sysid agent.
    if manifest["raw_dir"] != str(raw_dir):
        if raw_dir.name != manifest["case_name"]:
            print(f"[run_phystwin] ERROR: raw_dir basename {raw_dir.name!r} != "
                  f"manifest case_name {manifest['case_name']!r}; refusing to "
                  f"retarget the manifest.", file=sys.stderr)
            sys.exit(1)
        # A same-named case dir can hold regenerated data the old manifest's
        # LLM decisions no longer fit; require at least the frame budget to
        # still cover train_frame before quietly reusing them.
        try:
            with open(raw_dir / "metadata.json") as f:
                frame_num = int(json.load(f)["frame_num"])
        except (OSError, KeyError, ValueError) as e:
            print(f"[run_phystwin] ERROR: cannot read metadata.json at the "
                  f"new raw_dir ({e}); refusing to retarget.", file=sys.stderr)
            sys.exit(1)
        if int(manifest["train_frame"]) >= frame_num:
            print(f"[run_phystwin] ERROR: manifest train_frame "
                  f"{manifest['train_frame']} >= frame_num {frame_num} at the "
                  f"new raw_dir — the data changed; rerun with --overwrite.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[run_phystwin] manifest raw_dir is stale "
              f"({manifest['raw_dir']}); updating to {raw_dir}")
        manifest["raw_dir"] = str(raw_dir)
        _write_manifest(manifest_path, manifest)

    # ----- Stage 2: sim_initialization -----
    if not args.skip_stage2:
        from ar2s.agents_sysid.phystwin.setup import setup_phystwin_scene

        print("\n[stage 2] phystwin sim_initialization (build sim + sanity check)")
        t0 = time.time()
        try:
            setup = setup_phystwin_scene(
                raw_dir=manifest["raw_dir"],
                case_name=manifest["case_name"],
                train_frame=int(manifest["train_frame"]),
                config_choice=manifest["config_choice"],
                episode_dir=episode_dir,
                log_dir=log_dir,
                device=args.device,
            )
        except Exception as e:
            print(f"[stage 2] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
        si = setup.sim_initialization
        sc = setup.sanity_check
        print(f"[stage 2] done in {time.time()-t0:.1f}s — "
              f"springs={si.num_total_springs}, vertices={si.num_vertices}, "
              f"sanity={'PASS' if sc.passed else 'WARN'}"
              + (f" ({len(sc.issues)} issues)" if sc.issues else "")
              + f", cma_cached={setup.cache_state.cma_done}, "
              f"train_cached={setup.cache_state.train_done}")
    else:
        print("\n[stage 2] skipped")
        setup_path = episode_dir / "phystwin_setup.yaml"
        if not setup_path.exists():
            print("[stage 2] ERROR: phystwin_setup.yaml missing; "
                  "cannot skip stage 2.", file=sys.stderr)
            sys.exit(1)
        with open(setup_path) as f:
            setup_doc = yaml.safe_load(f)
        # The failure yaml is kept for postmortem (ok: false + error); Stage 3
        # would read data_loaded=false and return early as a false SUCCESS.
        if not (isinstance(setup_doc, dict) and setup_doc.get("ok", False)):
            err = setup_doc.get("error") if isinstance(setup_doc, dict) else "corrupt yaml"
            print(f"[stage 2] ERROR: existing phystwin_setup.yaml records a "
                  f"failed setup ({err}); rerun stage 2 (drop --skip-stage2).",
                  file=sys.stderr)
            sys.exit(1)

    # ----- Stage 3: sysid_agent (LLM ReAct) -----
    if args.skip_stage3:
        print("\n[stage 3] skipped (--skip-stage3)")
        print(f"\n[pipeline] TOTAL: {time.time()-t_pipeline:.1f}s")
        sys.exit(0)

    from ar2s.agents_sysid.agents.sysid_agent import run_sysid_agent

    print("\n[stage 3] phystwin sysid_agent (LLM dispatcher → CMA + train + inference)")
    t0 = time.time()
    try:
        result = run_sysid_agent(
            episode_dir, kind="phystwin", device=args.device, log_dir=log_dir,
        )
    except Exception as e:
        print(f"[stage 3] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    cma_msg = "fresh" if result.cma_result else "skipped (cached)"
    train_msg = "fresh" if result.train_result else "skipped (cached)"
    infer_msg = "ran" if result.infer_result else "skipped"
    print(f"[stage 3] done in {time.time()-t0:.1f}s — "
          f"cma={cma_msg}, train={train_msg}, infer={infer_msg}")
    print(f"[stage 3] summary: {result.agent_summary}")

    print(f"\n[pipeline] TOTAL: {time.time()-t_pipeline:.1f}s — SUCCESS ★")
    sys.exit(0)


if __name__ == "__main__":
    main()
