"""Phystwin Stage 2 (sim_initialization) — orchestrator-side wrapper.

Subprocess-runs ``_sim_init_script.py`` in the phystwin env via ``run_in_env``,
then parses ``<episode_dir>/phystwin_setup.yaml`` and returns a typed dataclass.

This is the phystwin counterpart to droid's ``calibrate_scene()``. The role
shape is the same:

    Stage 1 (Episode Agent) emits a manifest →
    Stage 2 (this) initializes sim + writes setup.yaml →
    Stage 3 (sysid.run_cma / run_train / run_infer) consumes setup.yaml

Failures are surfaced via RuntimeError. The yaml is still written on failure
(with ``ok: false`` + error fields) so caller can postmortem at
``<episode_dir>/phystwin_setup.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ar2s.agents_sysid._runners import (
    REPO_ROOT,
    RunResult,
    env_for_stage,
    run_in_env,
    script_path,
)


# -------------------------------------------------------------------
# Result dataclasses
# -------------------------------------------------------------------

@dataclass
class PhystwinSanityCheck:
    passed: bool
    issues: list[str] = field(default_factory=list)
    fully_invisible_frames: int = 0


@dataclass
class PhystwinSimInit:
    data_loaded: bool
    springs_built: bool
    forward_sim_verified: bool          # always false in current revision; placeholder
    num_object_springs: int | None = None
    num_total_springs: int | None = None
    num_vertices: int | None = None
    num_substeps: int | None = None
    cuda_available: bool | None = None


@dataclass
class PhystwinCacheState:
    cma_done: bool
    train_done: bool
    optimal_params_path: Path | None = None


@dataclass
class PhystwinDataShape:
    num_object_points: int
    num_surface_points: int
    num_all_points: int
    frame_len: int
    num_controller_points: int


@dataclass
class PhystwinSetup:
    """Structured output of Stage 2 (sim_initialization)."""
    case_name: str
    config_choice: str                   # "cloth" or "real"
    raw_dir: Path
    episode_dir: Path
    setup_yaml_path: Path

    # From metadata.json + calibrate.pkl
    frame_num: int
    fps: int
    n_cams: int
    WH: list[int]

    sim_initialization: PhystwinSimInit
    sanity_check: PhystwinSanityCheck
    cache_state: PhystwinCacheState
    data_shape: PhystwinDataShape

    elapsed_sec: float
    run_result: RunResult                # raw stdout/stderr for postmortem


# -------------------------------------------------------------------
# Public entry
# -------------------------------------------------------------------

def setup_phystwin_scene(
    raw_dir: str | Path,
    case_name: str,
    train_frame: int,
    config_choice: str,
    *,
    episode_dir: str | Path,
    log_dir: str | Path,
    device: str = "cuda:0",
) -> PhystwinSetup:
    """Run Stage 2 sim_initialization for a phystwin episode.

    Args:
        raw_dir: raw phystwin folder (with final_data.pkl, calibrate.pkl, ...).
        case_name: episode case name (usually = raw_dir folder name).
        train_frame: how many frames to verify (usually frame_num − 1).
        config_choice: ``"cloth"`` or ``"real"`` (decided by Stage 1).
        episode_dir: where ``phystwin_setup.yaml`` lands (analog of droid's
            ``<episode_dir>/calibration.yaml``).
        log_dir: where ``phystwin_sim_init.{cmd,stdout,stderr,returncode}`` go.
        device: cuda device (default ``"cuda:0"``).

    Returns:
        PhystwinSetup — structured Stage 2 result.

    Raises:
        FileNotFoundError: raw_dir missing required files.
        RuntimeError: subprocess returncode != 0 OR yaml not produced.
    """
    raw_dir = Path(raw_dir).expanduser().resolve()
    episode_dir = Path(episode_dir).expanduser().resolve()

    # Pre-validate raw_dir layout (saves a subprocess spawn if missing files).
    for required in ("calibrate.pkl", "metadata.json", "final_data.pkl"):
        if not (raw_dir / required).is_file():
            raise FileNotFoundError(
                f"phystwin raw_dir {raw_dir} missing required file: {required}"
            )
    if not (raw_dir / "color").is_dir():
        raise FileNotFoundError(f"phystwin raw_dir {raw_dir} missing required dir: color/")

    if config_choice not in ("cloth", "real"):
        raise ValueError(f"config_choice must be 'cloth' or 'real', got {config_choice!r}")

    args = [
        "--raw_dir", str(raw_dir),
        "--case_name", case_name,
        "--train_frame", str(int(train_frame)),
        "--config_choice", config_choice,
        "--episode_dir", str(episode_dir),
        "--device", device,
    ]

    result = run_in_env(
        env=env_for_stage("phystwin_sim_init"),
        script_path=script_path("phystwin_sim_init"),
        args=args,
        log_dir=log_dir,
        stage="phystwin_sim_init",
    )

    setup_yaml_path = episode_dir / "phystwin_setup.yaml"
    if not setup_yaml_path.exists():
        raise RuntimeError(
            f"phystwin sim_init returned {result.returncode} but no "
            f"phystwin_setup.yaml at {setup_yaml_path}.\n"
            f"stderr tail:\n{result.stderr_tail}"
        )

    with open(setup_yaml_path) as f:
        doc = yaml.safe_load(f)

    if result.returncode != 0 or not doc.get("ok", False):
        raise RuntimeError(
            f"phystwin sim_init failed (returncode={result.returncode}).\n"
            f"  setup.yaml.error: {doc.get('error', '(none)')}\n"
            f"  stderr full     : {result.stderr_path}\n"
            f"  stderr tail (last 50 lines):\n{result.stderr_tail}"
        )

    # Parse doc -> dataclass
    sim_init_d = doc.get("sim_initialization", {})
    sanity_d = doc.get("sanity_check", {})
    cache_d = doc.get("cache_state", {})
    shape_d = doc.get("data_shape", {})

    optimal_params_path = cache_d.get("optimal_params_path")
    return PhystwinSetup(
        case_name=doc["case_name"],
        config_choice=doc["config_choice"],
        raw_dir=Path(doc["raw_dir"]),
        episode_dir=episode_dir,
        setup_yaml_path=setup_yaml_path,
        frame_num=int(doc["frame_num"]),
        fps=int(doc["fps"]),
        n_cams=int(doc["n_cams"]),
        WH=list(doc["WH"]),
        sim_initialization=PhystwinSimInit(
            data_loaded=bool(sim_init_d.get("data_loaded", False)),
            springs_built=bool(sim_init_d.get("springs_built", False)),
            forward_sim_verified=bool(sim_init_d.get("forward_sim_verified", False)),
            num_object_springs=sim_init_d.get("num_object_springs"),
            num_total_springs=sim_init_d.get("num_total_springs"),
            num_vertices=sim_init_d.get("num_vertices"),
            num_substeps=sim_init_d.get("num_substeps"),
            cuda_available=sim_init_d.get("cuda_available"),
        ),
        sanity_check=PhystwinSanityCheck(
            passed=bool(sanity_d.get("passed", False)),
            issues=list(sanity_d.get("issues", [])),
            fully_invisible_frames=int(sanity_d.get("fully_invisible_frames", 0)),
        ),
        cache_state=PhystwinCacheState(
            cma_done=bool(cache_d.get("cma_done", False)),
            train_done=bool(cache_d.get("train_done", False)),
            optimal_params_path=Path(optimal_params_path) if optimal_params_path else None,
        ),
        data_shape=PhystwinDataShape(
            num_object_points=int(shape_d["num_object_points"]),
            num_surface_points=int(shape_d["num_surface_points"]),
            num_all_points=int(shape_d["num_all_points"]),
            frame_len=int(shape_d["frame_len"]),
            num_controller_points=int(shape_d["num_controller_points"]),
        ),
        elapsed_sec=float(doc.get("elapsed_sec", 0.0)),
        run_result=result,
    )


__all__ = [
    "PhystwinSetup",
    "PhystwinSimInit",
    "PhystwinSanityCheck",
    "PhystwinCacheState",
    "PhystwinDataShape",
    "setup_phystwin_scene",
]
