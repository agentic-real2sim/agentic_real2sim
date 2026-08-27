"""Persist a phystwin ScanResult to disk as ``phystwin_manifest.yaml``.

Phystwin counterpart to droid's ``build_episode.py``. Much lighter because
phystwin raw data is already in canonical layout — there's nothing to copy,
just a manifest yaml to write.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from ar2s.agents_sysid.skills.manifest_common import ScanResult


@dataclass
class PhystwinBuildReport:
    target_dir: Path
    manifest_path: Path

    def __str__(self) -> str:
        return f"PhystwinBuildReport(target={self.target_dir}, manifest={self.manifest_path.name})"


def write_phystwin_manifest(
    scan_result: ScanResult,
    target_dir: str | Path,
    *,
    overwrite: bool = True,
) -> PhystwinBuildReport:
    """Write the phystwin manifest to ``target_dir/phystwin_manifest.yaml``.

    Unlike droid's ``build_episode``, this does NOT copy any raw files —
    phystwin sysid reads directly from ``manifest.raw_dir``.

    Args:
        scan_result: from ``run_episode_agent(raw_dir)``.
            ``scan_result.file_moves`` is expected to be empty (phystwin
            convention).
        target_dir: ``sysid_inputs/<case_name>/`` is the run-local convention.
        overwrite: if False and manifest already exists, raise FileExistsError.
    """
    target = Path(target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    manifest_path = target / "phystwin_manifest.yaml"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; pass --overwrite to replace, "
            f"or --skip-stage1 to resume using the existing manifest."
        )

    if scan_result.manifest.get("pipeline_kind") != "phystwin":
        raise ValueError(
            f"Expected manifest with pipeline_kind='phystwin', got "
            f"{scan_result.manifest.get('pipeline_kind')!r}"
        )

    # Atomic: a batch SIGKILL mid-write must not leave a truncated manifest.
    tmp_path = manifest_path.with_suffix(".yaml.tmp")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(scan_result.manifest, f, sort_keys=False,
                       default_flow_style=False)
    os.replace(tmp_path, manifest_path)

    return PhystwinBuildReport(target_dir=target, manifest_path=manifest_path)


__all__ = ["write_phystwin_manifest", "PhystwinBuildReport"]
