"""Conda-env subprocess runner + cross-stage path constants.

Each visual stage lives in a different conda env with conflicting package
versions. ``run_in_env`` invokes them via ``conda run -n <env>`` and persists
stdout / stderr / returncode under ``<log_dir>/<stage>.{stdout,stderr,returncode,cmd}``
for postmortem.

Script paths resolve relative to the repo root. The CV stages are vendored
under ``third_party/`` and are shipped as part of the public release. The
visual-agent ``mesh_recover`` wrapper lives in this repo
at ``ar2s/agents_visual/_toolkit/scripts/run_sam3d_batch.py`` and loads
SAM3D once for a jobs list.

Callers must have run ``module load conda`` in their shell before launching
the orchestrator process; subprocess invocations inherit that env state.
"""
import functools
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"


def conda_env_root(default: Path) -> Path:
    """Resolve the conda env root shared by every subprocess runner.

    Order: $AR2S_CONDA_PATH (empty counts as unset, matching the shell
    scripts' ``:-`` semantics), else the repo's ``.conda`` symlink (the
    env-install scripts' convention for naming the machine-local env root),
    else ``default``. A dangling ``.conda`` symlink falls through to
    ``default`` rather than resolving to a nonexistent directory.
    """
    env_root = os.environ.get("AR2S_CONDA_PATH")
    if env_root:
        return Path(env_root).expanduser()
    dot_conda = (REPO_ROOT / ".conda").resolve()
    if dot_conda.is_dir():
        return dot_conda
    return default

# Consolidated model-weight root. Holds FoundationStereo + sam-3d-objects ckpts
# (routed via per-script CLI flags from stereo_depth.py / mesh_recover.py) and
# the HF hub cache (routed via HF_HUB_CACHE env var in _ENV_EXTRA_VARS below).
# FoundationPose weights live under third_party/FoundationPose/weights/ instead
# — its predictor classes hardcode that path with no override hook, and the
# user prefers leaving them at the vendored tree's expected location over symlinks.
# Override per-machine with DROID_MODELS_ROOT.
MODELS_ROOT = Path(os.environ.get("DROID_MODELS_ROOT", str(REPO_ROOT / "models")))


# Conda env names. Change here to track local env renames.
ENVS = {
    "droid_sim":       "droid_sim",
    "sam3":            "qianjun_sam3",
    "sam3d":           "qianjun_sam3d",
    "stereo":          "qianjun_foundation_stereo",
    "foundationpose":  "qianjun_foundationpose",
}


# Script paths relative to the repo root. All entries except ``mesh_recover``
# live under ``third_party/``; ``mesh_recover`` is our own thin wrapper that
# imports ``notebook.inference`` from the bundled sam-3d-objects source tree.
SCRIPTS = {
    "split_rectified_mp4":  "ar2s/agents_visual/_toolkit/scripts/split_rectified_mp4_frames.py",
    # Vendored copy of third_party/sam3/batch_video_segment.py with a 1-line
    # patch (predictor._ALL_INFERENCE_STATES -> _all_inference_states) to match
    # current sam3 API; see the script's inline note.
    "segmentation":         "ar2s/agents_visual/_toolkit/scripts/run_sam3_segment.py",
    "mesh_recover":         "ar2s/agents_visual/_toolkit/scripts/run_sam3d_batch.py",
    "stereo_depth":     "third_party/FoundationStereo/scripts/run_stereo_on_frames.py",
    "mesh_scale":           "third_party/FoundationStereo/scripts/compute_mesh_scale.py",
    "pose_tracking":        "third_party/FoundationPose/run_demo_foundationstereo_video.py",
}


# Stage -> ENVS key. video_build (ffmpeg) doesn't go through run_in_env.
STAGE_ENV = {
    "split_rectified_mp4":  "droid_sim",
    "segmentation":         "sam3",
    "mesh_recover":         "sam3d",
    "stereo_depth":         "stereo",
    "mesh_scale":           "stereo",
    "pose_tracking":        "foundationpose",
}


# Per-env automatic env-var injection for subprocess invocations.
#
# qianjun_foundation_stereo: XFORMERS_DISABLED=1 forces dinov2 (used inside
#   FoundationStereo's image backbone) to skip xformers and call torch's
#   built-in scaled_dot_product_attention instead. xformers 0.0.32.post2's
#   cu129 wheel ships a Hopper (sm_90) flash-attn kernel that errors out
#   with "invalid argument" on sm_120 (Blackwell / 5070 Ti). torch SDPA
#   has full sm_120 support so the fallback is safe and is the path
#   FoundationStereo's readme already documents for ONNX export.
#
# qianjun_sam3d: LIDRA_SKIP_INIT=1 skips sam3d_objects' Meta-internal LIDRA
#   initialiser, which tries to import a `sam3d_objects.init` submodule that
#   isn't shipped in the public release. The skip is the documented public
#   usage path — sam3d_objects/__init__.py reads this env var explicitly.
_ENV_EXTRA_VARS: dict[str, dict[str, str]] = {
    "qianjun_foundation_stereo": {
        "XFORMERS_DISABLED": "1",
        # FoundationStereo's image backbone (timm/edgenext_small.usi_in1k) is
        # fetched via huggingface_hub. Route to MODELS_ROOT/hf_cache like the
        # other envs so offline runs (HF_HUB_OFFLINE=1) find it instead of
        # reaching for the default ~/.cache/huggingface and failing.
        "HF_HUB_CACHE": str(MODELS_ROOT / "hf_cache"),
    },
    # SAM 3 (segmentation) fetches its weights via huggingface_hub
    # (facebook/sam3 + facebook/sam3.1). Route HF cache into MODELS_ROOT so all
    # ~24 GB of weights live under one workspace dir.
    "qianjun_sam3": {
        "HF_HUB_CACHE": str(MODELS_ROOT / "hf_cache"),
    },
    "qianjun_sam3d": {
        "LIDRA_SKIP_INIT": "1",
        # MoGe (depth prior used by sam-3d-objects Inference()) imports
        # xformers.triton.vararg_kernel, which assigns jitted_fn.src =
        # new_src — protected as of triton 3.4 (Cannot set attribute 'src'
        # directly). Same fix shape as the foundation_stereo dinov2 path:
        # let upstream fall back to non-xformers attention.
        "XFORMERS_DISABLED": "1",
        # MoGe + sam-3d-objects also pull from huggingface_hub
        # (Ruicheng/moge-vitl, facebook/sam-3d-objects, timm/edgenext_small).
        "HF_HUB_CACHE": str(MODELS_ROOT / "hf_cache"),
        # DINOv2 is resolved through torch.hub inside sam-3d-objects.
        "TORCH_HOME": str(MODELS_ROOT / "torch_cache"),
        # SAM3D runs close to the 20 GB card limit; expandable segments reduce
        # allocator fragmentation when the final decoder asks for one more
        # modest block after loading the model stack.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    # FoundationPose loads its own weights from third_party/FoundationPose/
    # weights/ (local), but route HF cache here too defensively in case any
    # timm/torchvision backbone it touches resolves via huggingface_hub.
    "qianjun_foundationpose": {
        "HF_HUB_CACHE": str(MODELS_ROOT / "hf_cache"),
    },
}


# Per-env HPC (Lmod) module sets, loaded inside the subprocess — on top of the
# launcher's inherited module state — right before exec'ing the env's python.
# Keyed by env name (same granularity as _ENV_EXTRA_VARS).
#
# Why this exists: a venv is not always self-contained — some stages pull
# runtime libraries from environment modules. The launcher loads ONE module set
# for the whole process, but stages need *conflicting* versions, so a single
# preamble can't satisfy all of them. Loading per-env here decouples them.
#
# Lmod resolves ``module load X/ver`` against the inherited state as a swap, so
# a version listed here overrides whatever the launcher loaded. An env absent
# here (or mapped to an empty list) runs with the inherited environment
# unchanged — and the wrapper no-ops entirely when ``module`` isn't available
# (off-cluster), so this is safe on machines without Lmod.
_ENV_MODULES: dict[str, list[str]] = {
    "qianjun_sam3d": ["opencv/4.10.0", "vtk/9.4.2"],
    "qianjun_foundationpose": ["hdf5/1.14.2", "scipy-stack/2023b"],
}


def _wrap_with_modules(cmd: list[str], modules: list[str]) -> list[str]:
    """Wrap ``cmd`` so it runs after ``module load <modules>`` in a child shell.

    Lmod exports its ``module`` shell function to child bash processes (via the
    ``BASH_FUNC_module%%`` env var), so a plain ``bash -c`` inherits the
    launcher's loaded modules and these loads apply *on top* of that state
    (version pins act as swaps). We deliberately do NOT use ``bash -lc``:
    re-sourcing the login profile could reset modules back to site defaults.

    ``command -v module`` guards the load so this is a no-op where Lmod isn't
    present (the command still runs, unmodified) — but a genuine load failure
    on a module system aborts loudly rather than running with the wrong libs.
    ``module``'s own chatter is redirected to stderr (the stage's ``.stderr``
    log) so it never contaminates the script's stdout. The real command is
    passed positionally and exec'd via ``"$@"``, avoiding any requoting of it.
    """
    load = "module load " + " ".join(shlex.quote(m) for m in modules)
    script = f'if command -v module >/dev/null 2>&1; then {load} 1>&2 || exit 1; fi; exec "$@"'
    return ["bash", "-c", script, "_", *cmd]


def script_path(stage: str) -> Path:
    """Absolute path to the script for ``stage``.

    Per-stage override via ``DROID_VISUAL_SCRIPT_<STAGE>`` env var (e.g.
    ``DROID_VISUAL_SCRIPT_MESH_RECOVER=/abs/path``) — useful when iterating
    on a script without going through the bundled source tree. Without an override,
    paths resolve under ``REPO_ROOT`` per ``SCRIPTS``.
    """
    env_key = f"DROID_VISUAL_SCRIPT_{stage.upper()}"
    override = os.environ.get(env_key)
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / SCRIPTS[stage]


def env_for_stage(stage: str) -> str:
    """Resolve a stage name to its actual conda env name."""
    return ENVS[STAGE_ENV[stage]]


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    returncode: int
    stdout_path: Path
    stderr_path: Path
    stderr_tail: str             # last STDERR_TAIL_LINES lines, for inline error reporting
    log_dir: Path
    cmd: list[str]


STDERR_TAIL_LINES = 50

# Soft "this stage is taking a while" reminder. Default 30 min; override via
# DROID_STAGE_WARN_SEC env var (e.g. set to 0 to disable, or 3600 for hourly).
# We never KILL — slow runs are usually legit (12 GB HF download, long videos,
# multi-min diffusion). The thread just prints a stderr line and continues,
# so user can decide to Ctrl+C or wait.
_DEFAULT_WARN_EVERY_SEC = int(os.environ.get("DROID_STAGE_WARN_SEC", "1800"))


def _spawn_progress_warner(
    stage: str, log_dir: Path, done: threading.Event, warn_every: int
) -> None:
    """Print a stderr warning every ``warn_every`` seconds until ``done`` is set.

    Cancelled cleanly when the subprocess returns (run_command sets ``done``
    in its ``finally`` block). Daemon thread so it doesn't block process exit
    on the chance the Event is somehow never set.
    """
    if warn_every <= 0:
        return  # disabled

    def loop() -> None:
        elapsed = 0
        while not done.wait(warn_every):
            elapsed += warn_every
            mins = elapsed // 60
            print(
                f"[runner] reminder: stage '{stage}' still running after {mins} min — "
                f"check {log_dir}/{stage}.stderr (HF download / network / deadlock?). "
                f"No timeout will fire; Ctrl+C if needed.",
                file=sys.stderr,
                flush=True,
            )

    threading.Thread(target=loop, daemon=True, name=f"warn-{stage}").start()


def run_command(
    cmd: list[str],
    *,
    log_dir: str | Path,
    stage: str,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    env_overrides: dict[str, str] | None = None,
    warn_every_sec: int | None = None,
) -> RunResult:
    """Run an arbitrary command, streaming stdout/stderr to per-stage log files.

    Files written under ``log_dir`` (clobbered on each call):

        <log_dir>/<stage>.cmd          — exact command for manual replay
        <log_dir>/<stage>.stdout
        <log_dir>/<stage>.stderr
        <log_dir>/<stage>.returncode

    Returns a RunResult; caller decides how to interpret a non-zero returncode.
    Use ``stderr_tail`` to surface the last ~50 lines in a Report.error field.

    Use ``run_in_env`` when the command is a Python script inside a conda env;
    use this function directly for non-Python tools (e.g. ffmpeg).

    ``env_overrides`` are merged onto the parent process environment for the
    child only; useful for forcing toggles like ``XFORMERS_DISABLED=1``.

    ``warn_every_sec`` controls the soft long-run reminder. ``None`` uses the
    module default (DROID_STAGE_WARN_SEC env var, default 1800s). Set to 0
    explicitly to disable. We NEVER kill — Ctrl+C is the only stop mechanism.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd_path    = log_dir / f"{stage}.cmd"
    stdout_path = log_dir / f"{stage}.stdout"
    stderr_path = log_dir / f"{stage}.stderr"
    rc_path     = log_dir / f"{stage}.returncode"

    cmd_path.write_text(" ".join(cmd) + "\n")

    proc_env = None
    if env_overrides:
        proc_env = os.environ.copy()
        proc_env.update(env_overrides)

    warn_every = _DEFAULT_WARN_EVERY_SEC if warn_every_sec is None else warn_every_sec
    done = threading.Event()
    _spawn_progress_warner(stage, log_dir, done, warn_every)

    try:
        with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
            proc = subprocess.run(
                cmd, stdout=fout, stderr=ferr, cwd=cwd, timeout=timeout, env=proc_env,
            )
    finally:
        done.set()  # cancel pending reminders cleanly

    rc_path.write_text(f"{proc.returncode}\n")

    return RunResult(
        returncode=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stderr_tail=_tail(stderr_path, STDERR_TAIL_LINES),
        log_dir=log_dir,
        cmd=cmd,
    )


@functools.lru_cache(maxsize=None)
def _env_python(env: str) -> str | None:
    """Resolve ``<env_prefix>/bin/python`` for a conda env name, else None.

    Why this exists: on some conda/mamba layouts (e.g. vast.ai pytorch images
    with envs under ``/venv/`` + a custom envs_dirs), ``conda run -n <env>
    python`` fails to resolve the *env's own* interpreter and silently falls
    back to the base python — so any script needing the env's site-packages
    dies with ModuleNotFoundError (cv2, torch, ...). Invoking the env's python
    binary directly sidesteps that entirely.

    Prefer this repository's local ``.conda/<env>/bin/python`` prefix before
    asking conda. The env installers create that layout, and on machines with
    a different default conda root, ``conda env list`` may not see it.

    We ask conda for the env list as JSON and match by directory basename.
    Returns None if conda isn't reachable or the env can't be located, in
    which case ``run_in_env`` falls back to the portable ``conda run`` form
    (which works fine on standard conda installs). Cached — the env layout
    doesn't change within a single pipeline run.
    """
    repo_local = REPO_ROOT / ".conda" / env / "bin" / "python"
    if repo_local.is_file():
        return str(repo_local)

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


def run_in_env(
    env: str,
    script_path: str | Path,
    args: list[str],
    *,
    log_dir: str | Path,
    stage: str,
    cwd: str | Path | None = None,
    timeout: float | None = None,
    env_overrides: dict[str, str] | None = None,
    warn_every_sec: int | None = None,
) -> RunResult:
    """Run ``python <script_path> <args...>`` inside conda env ``env``.

    Prefers invoking the env's python binary directly (``<prefix>/bin/python``)
    for robustness against non-standard conda layouts where ``conda run -n
    <env> python`` mis-resolves to the base interpreter. Falls back to the
    portable ``conda run --no-capture-output`` form when the env prefix can't
    be located (standard conda installs handle that fine).

    If ``cwd`` is not given, defaults to ``Path(script_path).parent`` — many
    upstream scripts use cwd-relative paths for checkpoints / configs.

    Per-env automatic env vars from ``_ENV_EXTRA_VARS`` are merged in first;
    caller-supplied ``env_overrides`` win on key conflicts.

    Per-env HPC modules from ``_ENV_MODULES`` are loaded inside the subprocess
    before the interpreter starts (see ``_wrap_with_modules``), so stages with
    conflicting module needs don't all inherit the launcher's single set.
    """
    if cwd is None:
        cwd = Path(script_path).parent

    auto = dict(_ENV_EXTRA_VARS.get(env, {}))
    env_python = _env_python(env)
    if env_python is not None:
        cmd = [env_python, str(script_path), *args]
        # `conda activate` would prepend <prefix>/lib to LD_LIBRARY_PATH; we
        # bypass activation by calling the binary directly, so replicate that
        # for any non-RPATH'd shared libs the env ships. setdefault so an
        # explicit caller override still wins.
        env_lib = Path(env_python).resolve().parent.parent / "lib"
        prev_ld = os.environ.get("LD_LIBRARY_PATH", "")
        auto.setdefault(
            "LD_LIBRARY_PATH",
            f"{env_lib}:{prev_ld}" if prev_ld else str(env_lib),
        )
    else:
        cmd = [
            "conda", "run", "-n", env, "--no-capture-output",
            "python", str(script_path), *args,
        ]

    # Load this env's HPC modules in the child shell, on top of (and swapping
    # against) whatever the launcher loaded. Wraps both the direct-python and
    # the `conda run` forms. No-op for envs without a module list.
    modules = _ENV_MODULES.get(env)
    if modules:
        cmd = _wrap_with_modules(cmd, modules)

    merged = {**auto, **(env_overrides or {})}
    return run_command(
        cmd, log_dir=log_dir, stage=stage, cwd=cwd, timeout=timeout,
        env_overrides=merged or None, warn_every_sec=warn_every_sec,
    )


def _tail(path: Path, n_lines: int) -> str:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])
