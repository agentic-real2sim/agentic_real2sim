"""Shared filesystem layout for runtime pipeline artifacts.

The canonical runtime layout is:

    outputs/run_pipeline/<run_id>/
        raw_episodes/
        visual_<episode_id>.py
        sysid_inputs/
        logs/
        seq/
        segmentation/
        meshes/
        poses/

One run directory holds exactly ONE episode. ``state.json`` lives at the run
root and carries a single ``episode_id``, so a second episode ingested into
the same run root would overwrite the first's ``support_tree`` (which
``geometry_prior`` reads) while the second emit overwrote the first's
``sysid_inputs/`` bundle — a silently wrong run, not a supported mode.

That is enforced structurally rather than by validation: the run id is always
``<episode_id>`` or ``<episode_id>_<suffix>`` (see ``run_id_for``), so the run
directory name is a pure function of the episode. Two different episodes
cannot name the same directory, and there is no CLI flag that lets them —
``--run-suffix`` only varies the part after the episode id, which is what
separates config variants of the SAME episode (e.g. ``..._gemma4_nopyzed``).

``outputs/`` is intentionally repo-relative; in the main workstation checkout
it is a symlink to ``/media/eric/data/droid_sim/outputs``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_ROOT = "AR2S_RUN_PIPELINE_ROOT"


def pipeline_artifact_root(root: str | Path | None = None) -> Path:
    """Return the root that contains per-run pipeline artifact directories."""
    if root is not None:
        return Path(root).expanduser().resolve()
    configured = os.environ.get(_ENV_ROOT)
    if configured:
        return Path(configured).expanduser().resolve()
    return (_REPO_ROOT / "outputs" / "run_pipeline").resolve()


def pipeline_run_dir(
    run_id: str,
    root: str | Path | None = None,
) -> Path:
    """Return ``outputs/run_pipeline/<run_id>`` for a non-empty run id."""
    assert run_id, "run_id must be non-empty"
    return pipeline_artifact_root(root) / run_id


def run_id_for(episode_id: str, run_suffix: str | None = None) -> str:
    """Derive the run id, and hence the run directory name, from the episode.

    ``run_suffix`` distinguishes config variants of the same episode and never
    replaces the episode id, so the run directory always identifies which
    episode it holds and two episodes can never collide on one directory.

    >>> run_id_for("ep_A")
    'ep_A'
    >>> run_id_for("ep_A", "gemma4_nopyzed")
    'ep_A_gemma4_nopyzed'
    """
    assert episode_id, "episode_id must be non-empty"
    if not run_suffix:
        return episode_id
    suffix = run_suffix.strip().lstrip("_")
    if not suffix:
        return episode_id
    if not re.fullmatch(r"[A-Za-z0-9._-]+", suffix):
        raise ValueError(
            f"--run-suffix must be letters/digits/._- only; got {run_suffix!r}"
        )
    return f"{episode_id}_{suffix}"


def raw_episodes_is_flat(raw_root: Path | str) -> bool:
    """True when ``raw_episodes/`` holds the episode directly (current layout).

    False means the legacy ``raw_episodes/<episode_id>/`` nesting. The stereo
    mp4 is the marker because build_raw_episode always places one, under
    either name it has had: ``<role>-stereo.mp4`` since role-keyed staging
    (ext1 is always staged), ``stereo.mp4`` before it. Both must match — on
    the legacy name alone, every current run answers False and callers treat
    ``source/``, the traceability backup dir, as the episode.

    Listed rather than stat'd so a dangling entry still counts: under
    ``--raw-link`` the marker is a symlink into the staging tree, and a dead
    target still proves this directory is flat.
    """
    try:
        names = os.listdir(str(raw_root))
    except OSError:
        return False
    return any(n == "stereo.mp4" or n.endswith("-stereo.mp4") for n in names)


def episode_bundle_dir(run_root: Path | str) -> Path:
    """The emitted sysid bundle inside a run dir.

    ``<run_root>/sysid_inputs/`` IS the bundle: one run dir holds one episode,
    so the old ``sysid_inputs/<episode_id>_agent/`` level could only ever have
    a single child. Legacy run dirs still have it, and are detected by that
    child existing — a new run never creates one.

    The ``_agent`` suffix survives OUTSIDE the run dir, in aggregate trees like
    ``episodes_droid_300/`` that hold many episodes at once and genuinely need
    the name to disambiguate; their readers glob for it directly.

    Returns the flat path when no legacy child is present, so callers can use
    the result as a write target as well as a read path.
    """
    root = Path(run_root) / "sysid_inputs"
    if not root.is_dir():
        return root
    legacy = sorted(p for p in root.glob("*_agent") if p.is_dir())
    assert len(legacy) <= 1, (
        f"{root} holds {len(legacy)} episode bundles ({[p.name for p in legacy]}); "
        f"a run dir must hold exactly one"
    )
    return legacy[0] if legacy else root


def visual_input_stub(run_root: Path | str) -> Path | None:
    """The run's ``visual_<episode_id>.py`` entry stub, or None.

    build_raw_episode writes it at the run root; ``visual_inputs/`` is the
    legacy location for a directory that could only ever hold one stub.
    """
    run_root = Path(run_root)
    for candidate in (run_root, run_root / "visual_inputs"):
        stubs = sorted(candidate.glob("visual_*.py"))
        if stubs:
            return stubs[0]
    return None


def sample_dir_name(sample_idx: int, n_samples: int) -> str:
    """Name of ``<sweep>/samples/<idx>/`` for a sweep of ``n_samples``.

    grasp_sweep zero-pads to the width of the largest index, minimum 2. Every
    reader that walks samples from ``summary.yaml`` rather than from the
    filesystem needs the same rule, so it lives here.
    """
    width = max(2, len(str(int(n_samples) - 1)))
    return f"{sample_idx:0{width}d}"
