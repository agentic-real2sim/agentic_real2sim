"""Conda-env subprocess runner + cross-stage path constants (sysid side).

Mirrors the design in ``ar2s/agents_visual/_toolkit/_runners.py`` so visual + sysid tooling
share a consistent shape: a stage name maps to a script path + a conda env
name, and ``run_in_env`` shells out to that env's Python.

Why subprocess instead of in-process import? Each sysid backend lives in a
different conda env with conflicting package versions (mujoco vs warp vs
torch CUDA arch), so the orchestrator can't import them all at once. Each
``run_in_env`` call spawns a fresh Python in the target env, runs the script
with the given args, and writes stdout/stderr to per-stage log files for
postmortem.

Current scope: only phystwin Stage 3 scripts are registered. Droid Stage 3
still runs in-process from ar2s.agents_sysid; if we move it to subprocess later,
add it to ENVS/SCRIPTS/STAGE_ENV here — the API doesn't change.

Per-script env var defaults (auto-injected unless caller overrides via ``env``):
  - WANDB_MODE=disabled  (phystwin's train.py + inference.py hard-call
    ``wandb.init()`` / ``wandb.Video()``; disabled makes them no-op so we
    don't need a wandb account).
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ar2s.agents_visual._toolkit._runners import conda_env_root


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AR2S_DEFAULT_CONDA_PATH = Path("/media/eric/data/conda_envs/droid_sim_envs")


# Conda env names. Map abstract keys (used in STAGE_ENV) to actual conda
# env names — change here when an env is renamed.
ENVS: dict[str, str] = {
    "phystwin_sysid": "phystwin",
    # Future: "droid_sysid": "droid_sim"  — when we move droid Stage 3 to subprocess too.
}


# Script paths relative to the repo root. Resolved by ``script_path()`` —
# overridable per-stage via ``SYSID_SCRIPT_<STAGE>=/abs/path`` env var.
SCRIPTS: dict[str, str] = {
    "phystwin_sim_init": "ar2s/agents_sysid/phystwin/_sim_init_script.py",
    "phystwin_cma":      "scripts/phystwin_optimize_cma.py",
    "phystwin_train":    "scripts/phystwin_train_warp.py",
    "phystwin_infer":    "scripts/phystwin_inference_warp.py",
}


# Stage -> ENVS key.
STAGE_ENV: dict[str, str] = {
    "phystwin_sim_init": "phystwin_sysid",
    "phystwin_cma":      "phystwin_sysid",
    "phystwin_train":    "phystwin_sysid",
    "phystwin_infer":    "phystwin_sysid",
}


# Default env vars to inject into every ``run_in_env`` invocation unless the
# caller supplies an ``env`` dict (in which case the caller decides everything).
_DEFAULT_ENV_OVERRIDES: dict[str, str] = {
    "WANDB_MODE": "disabled",
}


STDERR_TAIL_LINES = 50


def _conda_env_root() -> Path:
    return conda_env_root(AR2S_DEFAULT_CONDA_PATH)


def _apply_prefix_env_defaults(full_env: dict[str, str], env: str, env_python: str) -> None:
    """Replicate the repo activation-hook defaults for direct prefix execution."""
    env_prefix = Path(env_python).resolve().parent.parent
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


@functools.lru_cache(maxsize=None)
def _env_python(env: str) -> str | None:
    """Resolve ``<env_prefix>/bin/python`` for a conda env name.

    Install scripts create prefix envs under ``REPO_ROOT/.conda/<env>``. Prefer
    that stable repo-local access path, then fall back to conda's env registry
    by basename. Returning None lets ``run_in_env`` use ``conda run -n`` for
    older standard named-env installs.
    """
    candidates: list[Path] = [REPO_ROOT / ".conda" / env / "bin" / "python"]

    env_root = os.environ.get("AR2S_CONDA_PATH")
    if env_root:
        candidates.append(Path(env_root).expanduser() / env / "bin" / "python")

    for py in candidates:
        if py.is_file():
            return str(py)

    conda_exe = os.environ.get("CONDA_EXE") or "conda"
    try:
        out = subprocess.run(
            [conda_exe, "env", "list", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return None
        for prefix in json.loads(out.stdout).get("envs", []):
            if Path(prefix).name == env:
                py = Path(prefix) / "bin" / "python"
                if py.is_file():
                    return str(py)
    except Exception:
        return None

    return None


def script_path(stage: str) -> Path:
    """Absolute path to the script for ``stage``.

    Per-stage override via env var ``SYSID_SCRIPT_<STAGE>`` (uppercase),
    e.g. ``SYSID_SCRIPT_PHYSTWIN_CMA=/abs/path``. Without an override,
    paths resolve under ``REPO_ROOT`` per ``SCRIPTS``.
    """
    env_key = f"SYSID_SCRIPT_{stage.upper()}"
    override = os.environ.get(env_key)
    if override:
        return Path(override).expanduser().resolve()
    if stage not in SCRIPTS:
        raise KeyError(f"unknown stage {stage!r}; known: {sorted(SCRIPTS)}")
    return REPO_ROOT / SCRIPTS[stage]


def env_for_stage(stage: str) -> str:
    """Resolve a stage name to its actual conda env name."""
    if stage not in STAGE_ENV:
        raise KeyError(f"unknown stage {stage!r}; known: {sorted(STAGE_ENV)}")
    return ENVS[STAGE_ENV[stage]]


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    returncode: int
    stdout_path: Path
    stderr_path: Path
    stderr_tail: str             # last N lines of stderr, for inline error reporting
    log_dir: Path
    cmd: list[str]


def _tail(path: Path, n_lines: int) -> str:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


def run_command(
    cmd: list[str],
    *,
    log_dir: str | Path,
    stage: str,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> RunResult:
    """Run an arbitrary command, streaming stdout/stderr to per-stage log files.

    Files written under ``log_dir`` (clobbered on each call):

        <log_dir>/<stage>.cmd         — exact command for manual replay
        <log_dir>/<stage>.stdout
        <log_dir>/<stage>.stderr
        <log_dir>/<stage>.returncode

    Use ``run_in_env`` when the command is a Python script inside a conda env;
    use this function directly for non-Python tools or one-off commands.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd_path    = log_dir / f"{stage}.cmd"
    stdout_path = log_dir / f"{stage}.stdout"
    stderr_path = log_dir / f"{stage}.stderr"
    rc_path     = log_dir / f"{stage}.returncode"

    cmd_path.write_text(" ".join(cmd) + "\n")

    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        proc = subprocess.run(
            cmd, stdout=fout, stderr=ferr,
            cwd=str(cwd) if cwd else None, env=env, timeout=timeout,
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


def run_in_env(
    env: str,
    script_path: str | Path,
    args: list[str],
    *,
    log_dir: str | Path,
    stage: str,
    cwd: str | Path | None = None,
    env_vars: dict[str, str] | None = None,
    timeout: float | None = None,
) -> RunResult:
    """Run ``python <script_path> <args...>`` inside conda env ``env``.

    Args:
        env: conda env name (e.g. ``"phystwin"`` — already resolved from
            ENVS/STAGE_ENV by the caller; use ``env_for_stage(stage)``).
        script_path: path to the Python script. If a relative path is given,
            interpreted relative to ``REPO_ROOT``.
        args: extra CLI args appended after the script path.
        log_dir: directory for ``<stage>.{cmd,stdout,stderr,returncode}``.
        stage: stage name (used as log filename prefix).
        cwd: working directory for the subprocess. Defaults to REPO_ROOT —
            phystwin scripts write to ``experiments_optimization/<case>/`` and
            ``experiments/<case>/`` relative to cwd, so they need to land in
            the same place every time.
        env_vars: extra environment variables to pass to the subprocess. If
            None, defaults to ``_DEFAULT_ENV_OVERRIDES`` merged on top of
            the parent process env. If given, the caller's dict is merged
            on top of ``_DEFAULT_ENV_OVERRIDES`` (so caller can override).
        timeout: kill the subprocess after N seconds; None = no limit.

    Returns:
        RunResult — caller decides what to do with a non-zero ``returncode``.
        Use ``stderr_tail`` to surface the last 50 lines in an error report.
    """
    if cwd is None:
        cwd = REPO_ROOT

    sp = Path(script_path)
    if not sp.is_absolute():
        sp = REPO_ROOT / sp

    env_python = _env_python(env)
    # Build env: parent env + defaults + direct-prefix activation defaults +
    # caller overrides.
    full_env = os.environ.copy()
    full_env.update(_DEFAULT_ENV_OVERRIDES)
    if env_python is not None:
        _apply_prefix_env_defaults(full_env, env, env_python)
    if env_vars:
        full_env.update(env_vars)

    if env_python is not None:
        cmd = [env_python, str(sp), *args]
    else:
        cmd = [
            "conda", "run", "-n", env, "--no-capture-output",
            "python", str(sp), *args,
        ]
    return run_command(
        cmd, log_dir=log_dir, stage=stage,
        cwd=cwd, env=full_env, timeout=timeout,
    )


__all__ = [
    "REPO_ROOT",
    "ENVS",
    "SCRIPTS",
    "STAGE_ENV",
    "RunResult",
    "script_path",
    "env_for_stage",
    "run_command",
    "run_in_env",
]
