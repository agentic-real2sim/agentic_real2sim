"""Phystwin sysid wrappers — Stage 3 (CMA → train → inference).

Each wrapper:
  1. Validates inputs (raw dir exists, required predecessors present).
  2. Resolves config_choice ("cloth" / "real" / path) to a yaml file.
  3. Spawns a subprocess in the ``phystwin`` conda env via ``run_in_env``.
  4. Raises RuntimeError on non-zero returncode (with stderr_tail).
  5. Verifies the expected output file landed on disk.
  6. Returns a structured result dataclass with concrete paths.

Outputs (written relative to REPO_ROOT by the underlying scripts):
  - CMA:        experiments_optimization/<case_name>/optimal_params.pkl
  - Train:      experiments/<case_name>/train/best_*.pth + iter_*.pth
  - Inference:  experiments/<case_name>/inference/... (loaded best_N.pth)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ar2s.agents_sysid._runners import (
    REPO_ROOT,
    RunResult,
    env_for_stage,
    run_in_env,
    script_path,
)


# -------------------------------------------------------------------
# Config resolution
# -------------------------------------------------------------------

_CONFIGS_DIR = REPO_ROOT / "ar2s" / "phystwin_sysid" / "configs"
_NAMED_CONFIGS: dict[str, Path] = {
    "cloth": _CONFIGS_DIR / "cloth.yaml",   # iterations=100, self_collision=true (soft, foldable)
    "real":  _CONFIGS_DIR / "real.yaml",    # iterations=200, no self_collision  (rope, less folding)
}


def _resolve_config_path(config_choice: str) -> Path:
    """Resolve config_choice to an absolute YAML path.

    ``"cloth"`` / ``"real"`` map to the bundled configs. Anything else is
    treated as a filesystem path (absolute or relative to REPO_ROOT).
    Raises FileNotFoundError if the resolved path doesn't exist.
    """
    if config_choice in _NAMED_CONFIGS:
        path = _NAMED_CONFIGS[config_choice]
    else:
        path = Path(config_choice).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"phystwin config not found: {path} "
            f"(config_choice={config_choice!r}; expected one of "
            f"{sorted(_NAMED_CONFIGS)} or a path to a YAML file)"
        )
    return path


def _expected_data_path(base_path: str | Path, case_name: str) -> Path:
    """Path to ``<base_path>/<case_name>/final_data.pkl``. Used for input validation."""
    base = Path(base_path).expanduser()
    if not base.is_absolute():
        base = (REPO_ROOT / base).resolve()
    return base / case_name / "final_data.pkl"


# -------------------------------------------------------------------
# Result dataclasses
# -------------------------------------------------------------------

@dataclass
class PhystwinCmaResult:
    case_name: str
    config_path: Path
    optimal_params_path: Path        # experiments_optimization/<case>/optimal_params.pkl
    log_dir: Path                    # where stage logs (.stdout/.stderr) landed
    run_result: RunResult            # raw RunResult for postmortem


@dataclass
class PhystwinTrainResult:
    case_name: str
    config_path: Path
    best_ckpt_path: Path             # experiments/<case>/train/best_N.pth (largest N)
    train_dir: Path                  # experiments/<case>/train/
    log_dir: Path
    run_result: RunResult


@dataclass
class PhystwinInferResult:
    case_name: str
    config_path: Path
    output_dir: Path                 # experiments/<case>/inference/ (best-effort)
    log_dir: Path
    run_result: RunResult


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _raise_on_failure(result: RunResult, stage: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"phystwin {stage} failed (returncode={result.returncode}).\n"
            f"  full stderr: {result.stderr_path}\n"
            f"  full stdout: {result.stdout_path}\n"
            f"  command   : {' '.join(result.cmd)}\n"
            f"  stderr tail (last 50 lines):\n{result.stderr_tail}"
        )


def _validate_input(base_path: str | Path, case_name: str) -> None:
    """All three stages need final_data.pkl present."""
    data_path = _expected_data_path(base_path, case_name)
    if not data_path.exists():
        raise FileNotFoundError(
            f"phystwin input not found: {data_path} "
            f"(base_path={base_path!r}, case_name={case_name!r}; "
            f"expected raw phystwin folder layout with final_data.pkl, "
            f"calibrate.pkl, metadata.json, color/)"
        )


# -------------------------------------------------------------------
# Public wrappers
# -------------------------------------------------------------------

def run_cma(
    base_path: str | Path,
    case_name: str,
    train_frame: int,
    *,
    max_iter: int = 20,
    config_choice: str = "real",
    log_dir: str | Path,
    device: str = "cuda:0",
) -> PhystwinCmaResult:
    """Stage 1: CMA-ES sparse parameter optimization.

    Wraps ``scripts/phystwin_optimize_cma.py``. Writes its output to
    ``experiments_optimization/<case_name>/optimal_params.pkl`` (relative to
    REPO_ROOT — the subprocess's cwd).

    Args:
        base_path: parent dir of ``<case_name>/`` (raw phystwin data).
        case_name: episode folder name (e.g., ``"double_stretch_sloth"``).
        train_frame: how many frames to fit (typically frame_num − 1).
        max_iter: CMA generations (default 20).
        config_choice: ``"cloth"`` / ``"real"`` or a path to a YAML.
        log_dir: where ``phystwin_cma.{cmd,stdout,stderr,returncode}`` go.
        device: cuda device (default ``"cuda:0"``).

    Returns:
        PhystwinCmaResult with ``optimal_params_path`` set to the produced .pkl.

    Raises:
        RuntimeError: subprocess returncode != 0.
        FileNotFoundError: missing input or config, OR output .pkl not produced.
    """
    _validate_input(base_path, case_name)
    config_path = _resolve_config_path(config_choice)

    args = [
        "--base_path", str(base_path),
        "--case_name", case_name,
        "--train_frame", str(int(train_frame)),
        "--max_iter", str(int(max_iter)),
        "--device", device,
        "--config_path", str(config_path),
    ]

    result = run_in_env(
        env=env_for_stage("phystwin_cma"),
        script_path=script_path("phystwin_cma"),
        args=args,
        log_dir=log_dir,
        stage="phystwin_cma",
    )
    _raise_on_failure(result, "CMA")

    optimal_params = REPO_ROOT / "experiments_optimization" / case_name / "optimal_params.pkl"
    if not optimal_params.exists():
        raise FileNotFoundError(
            f"phystwin CMA returned 0 but optimal_params.pkl not written at {optimal_params}"
        )

    return PhystwinCmaResult(
        case_name=case_name,
        config_path=config_path,
        optimal_params_path=optimal_params,
        log_dir=Path(log_dir),
        run_result=result,
    )


def run_train(
    base_path: str | Path,
    case_name: str,
    train_frame: int,
    *,
    config_choice: str = "real",
    log_dir: str | Path,
    device: str = "cuda:0",
) -> PhystwinTrainResult:
    """Stage 2: Warp differentiable training (consumes CMA output).

    Wraps ``scripts/phystwin_train_warp.py``. Reads
    ``experiments_optimization/<case_name>/optimal_params.pkl`` and writes
    ``experiments/<case_name>/train/{best_N,iter_N}.pth + sim_iter*.mp4``.
    Iteration count comes from the YAML (cloth=100, real=200).

    Raises:
        FileNotFoundError: CMA hasn't been run (no optimal_params.pkl), or
            best_*.pth not produced after the subprocess exits cleanly.
        RuntimeError: subprocess returncode != 0.
    """
    _validate_input(base_path, case_name)
    config_path = _resolve_config_path(config_choice)

    cma_output = REPO_ROOT / "experiments_optimization" / case_name / "optimal_params.pkl"
    if not cma_output.exists():
        raise FileNotFoundError(
            f"phystwin train_warp requires CMA output: {cma_output}. "
            f"Run run_cma() first."
        )

    args = [
        "--base_path", str(base_path),
        "--case_name", case_name,
        "--train_frame", str(int(train_frame)),
        "--device", device,
        "--config_path", str(config_path),
    ]

    result = run_in_env(
        env=env_for_stage("phystwin_train"),
        script_path=script_path("phystwin_train"),
        args=args,
        log_dir=log_dir,
        stage="phystwin_train",
    )
    _raise_on_failure(result, "train_warp")

    train_dir = REPO_ROOT / "experiments" / case_name / "train"
    # Best checkpoint: highest-numbered best_N.pth (training writes best per
    # improvement, so the largest N is the last/best). Numeric sort — the
    # lexicographic order would rank best_9 over best_100.
    bests = sorted(train_dir.glob("best_*.pth"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not bests:
        raise FileNotFoundError(
            f"phystwin train_warp returned 0 but no best_*.pth in {train_dir}"
        )
    # Completion sentinel: interrupted trainings also leave best_N.pth files
    # (saved per vis_interval), so Stage 2's cache check requires this marker
    # written only after a clean train exit.
    (train_dir / "train_complete").touch()

    return PhystwinTrainResult(
        case_name=case_name,
        config_path=config_path,
        best_ckpt_path=bests[-1],
        train_dir=train_dir,
        log_dir=Path(log_dir),
        run_result=result,
    )


def run_infer(
    base_path: str | Path,
    case_name: str,
    *,
    config_choice: str = "real",
    log_dir: str | Path,
    device: str = "cuda:0",
) -> PhystwinInferResult:
    """Stage 3 (inference): forward-only roll-out with the trained model.

    Wraps ``scripts/phystwin_inference_warp.py``. Reads the latest
    ``experiments/<case_name>/train/best_*.pth`` and writes
    ``experiments/<case_name>/inference.{mp4,pkl}`` + ``inference_log.log``
    (the underlying script puts them at the top of the case dir, not in a
    nested ``inference/`` subdir).

    Raises:
        FileNotFoundError: no best_*.pth checkpoint to load.
        RuntimeError: subprocess returncode != 0.
    """
    _validate_input(base_path, case_name)
    config_path = _resolve_config_path(config_choice)

    train_dir = REPO_ROOT / "experiments" / case_name / "train"
    if not any(train_dir.glob("best_*.pth")):
        raise FileNotFoundError(
            f"phystwin inference requires a trained checkpoint: "
            f"no best_*.pth in {train_dir}. Run run_train() first."
        )

    args = [
        "--base_path", str(base_path),
        "--case_name", case_name,
        "--device", device,
        "--config_path", str(config_path),
    ]

    result = run_in_env(
        env=env_for_stage("phystwin_infer"),
        script_path=script_path("phystwin_infer"),
        args=args,
        log_dir=log_dir,
        stage="phystwin_infer",
    )
    _raise_on_failure(result, "inference_warp")

    # phystwin_inference_warp.py writes inference.{mp4,pkl} + inference_log.log
    # to the case-level dir, not a nested inference/ subdir.
    output_dir = REPO_ROOT / "experiments" / case_name
    # Inference may write multiple files; we just point at the directory.
    return PhystwinInferResult(
        case_name=case_name,
        config_path=config_path,
        output_dir=output_dir,
        log_dir=Path(log_dir),
        run_result=result,
    )


__all__ = [
    "PhystwinCmaResult",
    "PhystwinTrainResult",
    "PhystwinInferResult",
    "run_cma",
    "run_train",
    "run_infer",
]
