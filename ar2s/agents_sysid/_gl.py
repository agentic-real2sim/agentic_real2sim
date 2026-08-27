"""Headless OpenGL backend selection for MuJoCo offscreen rendering.

On a machine without an X display, MuJoCo's default GLFW backend cannot
create an OpenGL context. Every ``mujoco.Renderer`` call then fails with
``X11: The DISPLAY environment variable is missing`` and, worse, the broken
GLFW fallback path can deadlock inside a C++ static initializer
(``libc++abi: __cxa_guard_acquire detected recursive initialization``),
hanging the process indefinitely.

EGL renders offscreen without a window, so we select it when no DISPLAY is
present. ``setdefault`` semantics: an explicit ``MUJOCO_GL`` from the caller
(or a DISPLAY for interactive runs) always wins.

On shared GPU nodes (e.g. cgroup-isolated Slurm allocations), EGL's device
enumeration probes every ``/dev/dri/cardN`` node, including ones this
process has no permission for, and Mesa logs a "libEGL warning" for each
miss before settling on the one device it can actually open. That's just
enumeration noise — harmless, but it floods stderr. We turn it off by
raising Mesa's ``EGL_LOG_LEVEL`` to "fatal" (same opt-out semantics as
``MUJOCO_GL``: an explicit value from the caller wins).

Must run before the first Renderer/GLContext is created in a process. Call
it at CLI entry (inherited by ``spawn`` workers via ``os.environ``) and at
the start of any function that drives rendering directly.
"""
from __future__ import annotations

import os


def configure_headless_gl() -> str:
    """Select EGL when headless. Idempotent; returns the active MUJOCO_GL.

    Rules:
      - Caller's MUJOCO_GL always wins (explicit > default).
      - If DISPLAY is unset → headless → use egl.
      - If DISPLAY is set but the X server unreachable (common on cloud VMs
        / vast containers / SLURM workers where DISPLAY is pre-set for ssh
        forwarding without any actual X server) → still headless, use egl.
        Probe via xset / xdpyinfo; if both absent OR probe fails → headless.
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]

    display = os.environ.get("DISPLAY", "")
    if not display:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
        return "egl"

    # DISPLAY claims an X server — verify with a 2-sec probe. On failure,
    # treat as headless (better than letting GLFW deadlock on init).
    import shutil
    import subprocess

    probe_cmd = None
    if shutil.which("xset"):
        probe_cmd = ["xset", "q"]
    elif shutil.which("xdpyinfo"):
        probe_cmd = ["xdpyinfo"]

    if probe_cmd is None:
        # Neither probe tool installed → can't verify; assume headless.
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
        return "egl"

    try:
        subprocess.run(
            probe_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
        return "egl"

    # Real X session available; let MuJoCo use its default (glfw).
    return ""
