"""Convex decomposition of visual meshes via CoACD.

MuJoCo (and most rigid-body sims) only support convex collision geoms. A
concave visual mesh — bowl, mug, lid, pot — must be split into a set of
convex pieces; this module wraps CoACD (SIGGRAPH 2022, pip-installable) to
do that.

The output layout matches what ``ar2s/agents_sysid/scene_config.py`` expects:

    <episode_dir>/objects/<obj>/collision/<sanitized_name>_collision_<i>.obj

so ``ObjectInput.collision_mesh_dir`` just needs to be set to that folder
and the existing glob ``*_collision_*.obj`` picks them all up in order.

The convention is intentionally **disk-state-driven**: this tool reads
``visual.obj`` as-it-is-on-disk. If an earlier step (planned: mesh_orient)
overwrote ``visual.obj`` with a flipped / rotated version, the collision
parts will match the flipped geometry without any extra plumbing.

Idempotent: if the output dir already contains ``<sanitized>_collision_*.obj``
files, we skip unless ``force=True``.

CLI (one episode):
    python -m ar2s.agents_sysid.skills.collision_decompose <episode_dir> \\
        [--threshold 0.05] [--object <name>] [--force]

CLI (one mesh):
    python -m ar2s.agents_sysid.skills.collision_decompose --mesh <visual.obj> \\
        --out <collision_dir> --name <obj_name> [--threshold 0.05] [--force]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from ar2s.droid_sim._util import sanitize_name as _sanitize


# =========================================================================
# Per-mesh
# =========================================================================

@dataclass
class DecomposeResult:
    object_name: str
    visual_mesh: str
    pieces: list[str] = field(default_factory=list)
    skipped: bool = False                # already populated; force=False
    threshold: float = 0.05
    error: str = ""


def decompose_mesh(
    mesh_path: Path | str,
    out_dir: Path | str,
    *,
    name: str,
    threshold: float = 0.05,
    max_convex_hull: int = -1,
    preprocess_mode: str = "auto",
    seed: int = 0,
    force: bool = False,
    log_level: str = "warn",
) -> DecomposeResult:
    """Run CoACD on a single mesh.

    Args:
        mesh_path: visual .obj / .glb / .stl ... (anything trimesh loads).
        out_dir: directory to write ``<sanitized_name>_collision_<i>.obj``.
            Created if missing.
        name: canonical object name (gets sanitised for the filename).
        threshold: CoACD's concavity cutoff. Higher → fewer, coarser pieces.
            0.05 is the upstream default; 0.03 for fine detail; 0.08 for
            quick-and-coarse.
        max_convex_hull: hard cap on piece count (-1 = unlimited).
        preprocess_mode: "auto" / "on" / "off". "auto" wraps non-manifold
            inputs in a watertight envelope; turn "off" only if the mesh
            is already manifold and you want the input vertices preserved
            verbatim.
        seed: CoACD's MCTS seed; same seed + same mesh + same params →
            byte-identical output.
        force: re-run even if output dir already has matching pieces.
        log_level: CoACD verbosity; "warn" prints only warnings/errors.

    Returns:
        DecomposeResult with the list of written piece paths (or
        ``skipped=True`` if pre-existing pieces were kept).

    Raises:
        FileNotFoundError if mesh_path doesn't exist.
        RuntimeError on CoACD failure (the .obj that triggered the failure
        is included in the message).
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"visual mesh not found: {mesh_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sanitized = _sanitize(name)
    existing = sorted(out_dir.glob(f"{sanitized}_collision_*.obj"))
    if existing and not force:
        return DecomposeResult(
            object_name=name,
            visual_mesh=str(mesh_path),
            pieces=[str(p) for p in existing],
            skipped=True,
            threshold=threshold,
        )

    # Wipe stale pieces so we don't mix old + new (the index suffixes the
    # glob enumerates are positional, not deterministic across reruns).
    for p in existing:
        p.unlink()

    # CoACD is loaded lazily so importing this module doesn't fail in envs
    # where coacd isn't installed (e.g. agents-only inspection).
    import coacd as _coacd
    _coacd.set_log_level(log_level)

    mesh = trimesh.load(str(mesh_path), force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    if V.size == 0 or F.size == 0:
        raise RuntimeError(f"empty mesh after load: {mesh_path}")

    cm = _coacd.Mesh(V, F)
    try:
        parts = _coacd.run_coacd(
            cm,
            threshold=float(threshold),
            max_convex_hull=int(max_convex_hull),
            preprocess_mode=preprocess_mode,
            seed=int(seed),
        )
    except Exception as e:
        raise RuntimeError(f"CoACD failed on {mesh_path}: {e}") from e

    pieces: list[Path] = []
    for i, (verts, faces) in enumerate(parts):
        piece = trimesh.Trimesh(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        path = out_dir / f"{sanitized}_collision_{i}.obj"
        piece.export(str(path))
        pieces.append(path)

    return DecomposeResult(
        object_name=name,
        visual_mesh=str(mesh_path),
        pieces=[str(p) for p in pieces],
        skipped=False,
        threshold=threshold,
    )


# =========================================================================
# Per-episode
# =========================================================================

def decompose_episode(
    episode_dir: Path | str,
    *,
    objects: list[str] | None = None,
    threshold: float = 0.05,
    force: bool = False,
    **kwargs,
) -> dict[str, DecomposeResult]:
    """Walk ``<episode>/objects/<obj>/visual.obj`` and decompose each.

    Args:
        episode_dir: episode root (must have an ``objects/`` subdir).
        objects: explicit list of object subdirs to process. None → every
            subdir of ``objects/`` that has a ``visual.obj``.
        threshold, force, **kwargs: forwarded to ``decompose_mesh``.

    Returns:
        ``{object_name: DecomposeResult}``. ``object_name`` is the
        on-disk directory name (NOT sanitised) so callers can join paths
        back to the same folder.
    """
    episode_dir = Path(episode_dir).resolve()
    objects_root = episode_dir / "objects"
    if not objects_root.is_dir():
        raise FileNotFoundError(f"no objects/ under {episode_dir}")

    if objects is None:
        targets = [
            p.name for p in sorted(objects_root.iterdir())
            if p.is_dir() and (p / "visual.obj").exists()
        ]
    else:
        targets = list(objects)

    results: dict[str, DecomposeResult] = {}
    for obj_name in targets:
        obj_dir = objects_root / obj_name
        visual = obj_dir / "visual.obj"
        if not visual.exists():
            results[obj_name] = DecomposeResult(
                object_name=obj_name,
                visual_mesh=str(visual),
                error=f"visual.obj missing under {obj_dir}",
            )
            continue
        out_dir = obj_dir / "collision"
        try:
            res = decompose_mesh(
                visual, out_dir,
                name=obj_name,
                threshold=threshold,
                force=force,
                **kwargs,
            )
        except Exception as e:
            res = DecomposeResult(
                object_name=obj_name,
                visual_mesh=str(visual),
                threshold=threshold,
                error=str(e),
            )
        results[obj_name] = res
    return results


# =========================================================================
# CLI
# =========================================================================

def _print_result(name: str, r: DecomposeResult) -> None:
    if r.error:
        print(f"  [{name}] ERROR: {r.error}", file=sys.stderr)
    elif r.skipped:
        print(f"  [{name}] kept {len(r.pieces)} existing piece(s) (use --force to rebuild)")
    else:
        print(f"  [{name}] -> {len(r.pieces)} piece(s) (threshold={r.threshold})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convex-decompose visual meshes into collision parts (CoACD).",
    )
    g_episode = ap.add_argument_group("episode mode")
    g_episode.add_argument("episode", nargs="?", help="path to episode dir")
    g_episode.add_argument(
        "--object", action="append", default=None,
        help="object subdir to process (repeatable; default = all)",
    )

    g_one = ap.add_argument_group("single-mesh mode")
    g_one.add_argument("--mesh", help="single .obj input (mutually exclusive with episode arg)")
    g_one.add_argument("--out", help="output dir for single-mesh mode")
    g_one.add_argument("--name", help="object name (sanitised for filenames)")

    ap.add_argument("--threshold", type=float, default=0.05,
                    help="CoACD concavity threshold (default 0.05; lower = finer)")
    ap.add_argument("--max-convex-hull", type=int, default=-1,
                    help="hard cap on piece count (-1 = unlimited)")
    ap.add_argument("--preprocess-mode", default="auto",
                    choices=("auto", "on", "off"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="re-run even if output dir already has matching pieces")
    ap.add_argument("--log-level", default="warn",
                    choices=("off", "error", "warn", "info", "debug"))
    args = ap.parse_args(argv)

    common = dict(
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hull,
        preprocess_mode=args.preprocess_mode,
        seed=args.seed,
        force=args.force,
        log_level=args.log_level,
    )

    if args.mesh:
        if not args.out or not args.name:
            ap.error("--mesh requires --out and --name")
        res = decompose_mesh(args.mesh, args.out, name=args.name, **common)
        _print_result(args.name, res)
        return 0 if not res.error else 1

    if not args.episode:
        ap.error("provide either an episode path or --mesh/--out/--name")
    results = decompose_episode(
        args.episode,
        objects=args.object,
        **common,
    )
    print(f"[collision_decompose] {args.episode}: {len(results)} object(s)")
    for n, r in results.items():
        _print_result(n, r)
    return 0 if all(not r.error for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
