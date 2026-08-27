"""Conda-env subprocess runner for PhysTwin visual pipeline stages.

Scripts are invoked as Python modules (``python -m ar2s.phystwin_visual.<module>``)
from REPO_ROOT so that intra-package imports resolve correctly in the subprocess.
The conda envs must have this repo installed as editable (``pip install -e
/path/to/droid_sim``) so that ``ar2s.phystwin_visual`` is importable.
"""
import os
import subprocess
from pathlib import Path

from ar2s.agents_visual._toolkit._runners import RunResult, conda_env_root

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AR2S_DEFAULT_CONDA_PATH = Path("/media/eric/data/conda_envs/droid_sim_envs")
STDERR_TAIL_LINES = 50


# ---------------------------------------------------------------------------
# Env / module tables
# ---------------------------------------------------------------------------

# Conda env names
ENVS = {
    "phystwin_seg":   "phystwin_seg",
    "phystwin_track": "phystwin_track",
    "phystwin_proc":  "phystwin_proc",
    "phystwin_align": "phystwin_align",
}

# Fully-qualified Python module names for each stage
MODULES = {
    "video_segmentation": "ar2s.phystwin_visual.segment",
    "dense_tracking":     "ar2s.phystwin_visual.dense_track",
    "pcd_projection":     "ar2s.phystwin_visual.data_process_pcd",
    "mask_cleanup":       "ar2s.phystwin_visual.data_process_mask",
    "track_processing":   "ar2s.phystwin_visual.data_process_track",
    "shape_alignment":    "ar2s.phystwin_visual.align",
    "final_export":       "ar2s.phystwin_visual.data_process_sample",
}

# Stage -> env key
STAGE_ENV = {
    "video_segmentation": "phystwin_seg",
    "dense_tracking":     "phystwin_track",
    "pcd_projection":     "phystwin_proc",
    "mask_cleanup":       "phystwin_proc",
    "track_processing":   "phystwin_proc",
    "shape_alignment":    "phystwin_align",
    "final_export":       "phystwin_align",
}


def module_for_stage(stage: str) -> str:
    """Return the Python module name for ``stage``."""
    env_key = f"PHYSTWIN_VISUAL_MODULE_{stage.upper()}"
    return os.environ.get(env_key, MODULES[stage])


def env_for_stage(stage: str) -> str:
    """Resolve stage name to conda env name."""
    return ENVS[STAGE_ENV[stage]]


def _env_python(env: str) -> Path | None:
    candidates = [REPO_ROOT / ".conda" / env / "bin" / "python"]
    env_root = os.environ.get("AR2S_CONDA_PATH")
    if env_root:
        candidates.append(Path(env_root).expanduser() / env / "bin" / "python")
    for python in candidates:
        if python.is_file():
            return python
    return None


def _conda_env_root() -> Path:
    return conda_env_root(AR2S_DEFAULT_CONDA_PATH)


def _apply_prefix_env_defaults(full_env: dict[str, str], env: str, env_python: Path) -> None:
    env_prefix = env_python.resolve().parent.parent
    env_root = _conda_env_root()
    defaults = {
        "CONDA_PKGS_DIRS": env_root / "pkgs",
        "PIP_CACHE_DIR": env_root / "pip-cache",
        "WARP_CACHE_PATH": env_root / "warp-cache",
        "TMPDIR": env_root / "tmp",
        "TEMP": env_root / "tmp",
        "TMP": env_root / "tmp",
        "XDG_CACHE_HOME": env_root / "xdg-cache",
        "TORCH_HOME": env_root / "torch-cache",
        "HF_HOME": Path(os.environ.get("HF_HOME", str(env_root / "huggingface"))).expanduser(),
    }
    for key, value in defaults.items():
        full_env.setdefault(key, str(value))

    full_env["CONDA_PREFIX"] = str(env_prefix)
    full_env["CONDA_DEFAULT_ENV"] = env
    full_env.setdefault("PYTHONNOUSERSITE", "1")

    env_bin = str(env_prefix / "bin")
    prev_path = full_env.get("PATH", "")
    full_env["PATH"] = f"{env_bin}:{prev_path}" if prev_path else env_bin

    env_target_lib = str(env_prefix / "targets" / "x86_64-linux" / "lib")
    env_lib = str(env_prefix / "lib")
    prev_ld = full_env.get("LD_LIBRARY_PATH", "")
    env_ld = f"{env_target_lib}:{env_lib}"
    full_env["LD_LIBRARY_PATH"] = f"{env_ld}:{prev_ld}" if prev_ld else env_ld


def _tail(path: Path, n_lines: int) -> str:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


def _run_command(
    cmd: list[str],
    *,
    log_dir: str | Path,
    stage: str,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> RunResult:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd_path = log_dir / f"{stage}.cmd"
    stdout_path = log_dir / f"{stage}.stdout"
    stderr_path = log_dir / f"{stage}.stderr"
    rc_path = log_dir / f"{stage}.returncode"

    cmd_path.write_text(" ".join(cmd) + "\n")
    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        proc = subprocess.run(
            cmd, stdout=fout, stderr=ferr, cwd=cwd, env=env, timeout=timeout,
        )
    rc_path.write_text(f"{proc.returncode}\n")

    return RunResult(
        returncode=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stderr_tail=_tail(stderr_path, STDERR_TAIL_LINES),
        log_dir=log_dir,
        cmd=cmd,
    )


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_module_in_env(
    env: str,
    module: str,
    args: list[str],
    *,
    log_dir: str | Path,
    stage: str,
    cwd: str | Path | None = None,
    timeout: float | None = None,
) -> RunResult:
    """Run ``python -m <module> <args>`` inside conda env ``env``.

    Uses REPO_ROOT as cwd by default so that relative paths and intra-package
    imports resolve correctly.

    Wraps ``ar2s.agents_visual._toolkit._runners.run_command``; logs land under
    ``<log_dir>/<stage>.{cmd,stdout,stderr,returncode}``.
    """
    if cwd is None:
        cwd = REPO_ROOT
    env_python = _env_python(env)
    proc_env = None
    if env_python is not None:
        cmd = [str(env_python), "-m", module, *args]
        proc_env = os.environ.copy()
        _apply_prefix_env_defaults(proc_env, env, env_python)
    else:
        cmd = [
            "conda", "run", "-n", env, "--no-capture-output",
            "python", "-m", module, *args,
        ]
    return _run_command(
        cmd, log_dir=log_dir, stage=stage, cwd=cwd, env=proc_env, timeout=timeout,
    )
