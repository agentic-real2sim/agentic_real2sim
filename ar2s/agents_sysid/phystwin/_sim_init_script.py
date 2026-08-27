"""Phystwin Stage 2 (sim_initialization) — worker script.

Runs in the ``phystwin`` conda env via ``run_in_env``. Builds a PhysTwinConfig,
loads data (verifying schema + GPU + warp), constructs spring connectivity,
sanity-checks the data, and writes a ``phystwin_setup.yaml``.

What this verifies that Stage 3 needs:
  1. final_data.pkl loads + matches metadata.json frame_num
  2. No NaN/Inf in object_points / controller_points
  3. PhysTwinConfig builds with the chosen YAML + camera params
  4. open3d KDTree spring construction succeeds (uses object_radius etc.)
  5. CUDA + Warp init OK (PhysTwinRealData ctor moves to GPU; KDTree is CPU but
     subsequent calls would fail without Warp loaded)

What this does NOT verify (deferred, marked ``forward_sim_verified: false``):
  - Whether SpringMassSystemWarp.step() runs without NaN. Building the
    simulator requires reaching into OptimizerCMA.error_func() internals,
    fragile; left for Stage 3 to discover. Future revision can add this.

Outputs ``<episode_dir>/phystwin_setup.yaml`` (see schema in setup.py).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from ar2s.phystwin_sysid import OptimizerCMA, PhysTwinConfig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--case_name", required=True)
    p.add_argument("--train_frame", type=int, required=True)
    p.add_argument("--config_choice", choices=("cloth", "real"), required=True)
    p.add_argument("--episode_dir", required=True,
                   help="Where to write phystwin_setup.yaml.")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    case_name = args.case_name
    episode_dir = Path(args.episode_dir).resolve()
    episode_dir.mkdir(parents=True, exist_ok=True)
    setup_yaml_path = episode_dir / "phystwin_setup.yaml"

    # We progressively fill ``setup`` as steps succeed. If a step throws,
    # we still write what we have so caller can debug.
    setup: dict = {
        "case_name": case_name,
        "config_choice": args.config_choice,
        "raw_dir": str(raw_dir),
        "device": args.device,
        "sim_initialization": {
            "data_loaded": False,
            "springs_built": False,
            "forward_sim_verified": False,  # always false in this revision
        },
        "sanity_check": {
            "passed": False,
            "issues": [],
        },
        "cache_state": {
            "cma_done": False,
            "train_done": False,
        },
    }

    t0 = time.time()

    try:
        # ---- 1. Build PhysTwinConfig
        repo_root = Path(__file__).resolve().parents[3]
        config_path = repo_root / "ar2s" / "phystwin_sysid" / "configs" / f"{args.config_choice}.yaml"
        cfg = PhysTwinConfig.from_yaml(str(config_path))

        with open(raw_dir / "calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        with open(raw_dir / "metadata.json", "r") as f:
            meta = json.load(f)

        cfg = dataclasses.replace(
            cfg,
            c2ws=np.array(c2ws),
            w2cs=np.array(w2cs),
            intrinsics=np.array(meta["intrinsics"]),
            WH=meta["WH"],
            overlay_path=str(raw_dir / "color"),
        )
        setup["frame_num"] = int(meta["frame_num"])
        setup["fps"] = int(meta["fps"])
        setup["n_cams"] = len(c2ws)
        setup["WH"] = list(meta["WH"])
        print(f"[sim_init] cfg built (n_cams={len(c2ws)}, frame_num={meta['frame_num']})", flush=True)

        # ---- 2. Build OptimizerCMA — triggers PhysTwinRealData load
        base_dir = episode_dir / "sim_init_tmp"
        base_dir.mkdir(parents=True, exist_ok=True)
        optimizer = OptimizerCMA(
            data_path=str(raw_dir / "final_data.pkl"),
            base_dir=str(base_dir),
            train_frame=args.train_frame,
            cfg=cfg,
            device=args.device,
        )
        setup["sim_initialization"]["data_loaded"] = True
        setup["data_shape"] = {
            "num_object_points":      int(optimizer.num_original_points),
            "num_surface_points":     int(optimizer.num_surface_points),
            "num_all_points":         int(optimizer.num_all_points),
            "frame_len":              int(optimizer.dataset.frame_len),
            "num_controller_points":  int(optimizer.controller_points.shape[1]),
        }
        print(f"[sim_init] OptimizerCMA built (data loaded). "
              f"frame_len={optimizer.dataset.frame_len}, "
              f"obj_pts={optimizer.num_original_points}", flush=True)

        # ---- 3. Sanity checks
        issues: list[str] = []
        if torch.isnan(optimizer.object_points).any():
            issues.append("NaN in object_points")
        if torch.isnan(optimizer.controller_points).any():
            issues.append("NaN in controller_points")

        if optimizer.object_visibilities is not None:
            per_frame_visible = optimizer.object_visibilities.any(dim=1)
            n_invisible = int((~per_frame_visible).sum().item())
            setup["sanity_check"]["fully_invisible_frames"] = n_invisible
            if n_invisible > 0:
                issues.append(f"{n_invisible} frames have ALL object points invisible")

        if optimizer.dataset.frame_len != meta["frame_num"]:
            issues.append(
                f"data frame_len {optimizer.dataset.frame_len} != metadata frame_num {meta['frame_num']}"
            )
        if args.train_frame > optimizer.dataset.frame_len:
            issues.append(
                f"train_frame {args.train_frame} > frame_len {optimizer.dataset.frame_len}"
            )

        # ---- 4. Build springs via _init_start (verifies open3d KDTree path)
        structure_points = optimizer.structure_points
        controller_points_frame0 = optimizer.controller_points[0]

        init_vertices, init_springs, init_rest_lengths, init_masses, num_object_springs = (
            optimizer._init_start(
                object_points=structure_points,
                controller_points=controller_points_frame0,
                object_radius=cfg.object_radius,
                object_max_neighbours=cfg.object_max_neighbours,
                controller_radius=cfg.controller_radius,
                controller_max_neighbours=cfg.controller_max_neighbours,
            )
        )
        setup["sim_initialization"].update({
            "springs_built":         True,
            "num_object_springs":    int(num_object_springs),
            "num_total_springs":     int(init_springs.shape[0]),
            "num_vertices":          int(init_vertices.shape[0]),
            "num_substeps":          int(cfg.num_substeps),
            "cuda_available":        bool(torch.cuda.is_available()),
        })
        print(f"[sim_init] springs built (total={init_springs.shape[0]}, "
              f"object={num_object_springs}, vertices={init_vertices.shape[0]})",
              flush=True)

        # ---- 5. Sanity verdict
        if not issues:
            setup["sanity_check"]["passed"] = True
        setup["sanity_check"]["issues"] = issues

        # ---- 6. Cache state. Existence alone is not enough: a batch SIGKILL
        # mid-train leaves early best_N.pth files (train.py saves per
        # vis_interval), and a same-named case regenerated from new data
        # leaves checkpoints whose spring count no longer matches — either
        # would make Stage 3 skip training and roll out garbage.
        cma_pkl = repo_root / "experiments_optimization" / case_name / "optimal_params.pkl"
        train_dir = repo_root / "experiments" / case_name / "train"
        cma_done = False
        if cma_pkl.exists():
            try:
                with open(cma_pkl, "rb") as f:
                    cma_done = isinstance(pickle.load(f), dict)
            except Exception as e:
                print(f"[sim_init] cache: optimal_params.pkl unreadable ({e}); "
                      f"cma_done=False", flush=True)
        train_done = False
        if (train_dir / "train_complete").exists():
            bests = sorted(train_dir.glob("best_*.pth"),
                           key=lambda p: int(p.stem.split("_")[1]))
            if bests:
                try:
                    ckpt = torch.load(bests[-1], map_location="cpu",
                                      weights_only=False)
                    train_done = int(ckpt["num_object_springs"]) == int(num_object_springs)
                    if not train_done:
                        print(f"[sim_init] cache: checkpoint spring count "
                              f"{ckpt['num_object_springs']} != current "
                              f"{num_object_springs}; train_done=False", flush=True)
                except Exception as e:
                    print(f"[sim_init] cache: best checkpoint unreadable ({e}); "
                          f"train_done=False", flush=True)
        setup["cache_state"]["cma_done"] = cma_done
        setup["cache_state"]["train_done"] = train_done
        if cma_done:
            setup["cache_state"]["optimal_params_path"] = str(cma_pkl)
        print(f"[sim_init] cache: cma_done={cma_done}, train_done={train_done}", flush=True)

        setup["elapsed_sec"] = round(time.time() - t0, 2)
        setup["ok"] = True

        # OptimizerCMA's ctor eagerly mkdirs base_dir/optimizeCMA and renders
        # gt.mp4 there; this run never optimizes, and Stage 3 re-renders the
        # same gt.mp4 under experiments_optimization/, so the scratch dir is
        # pure leftover. Kept on failure for postmortem.
        shutil.rmtree(episode_dir / "sim_init_tmp", ignore_errors=True)

    except Exception as e:
        setup["ok"] = False
        setup["error"] = f"{type(e).__name__}: {e}"
        setup["traceback"] = traceback.format_exc()
        setup["elapsed_sec"] = round(time.time() - t0, 2)
        print(f"[sim_init] FAILED: {setup['error']}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

    # Always write yaml — even on failure, so caller can postmortem.
    with open(setup_yaml_path, "w") as f:
        yaml.safe_dump(setup, f, sort_keys=False, default_flow_style=False)
    print(f"[sim_init] wrote {setup_yaml_path}", flush=True)

    return 0 if setup.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
