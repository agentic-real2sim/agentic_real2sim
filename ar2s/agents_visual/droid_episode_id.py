"""DROID-100 episode id convention for visual-agent.

Canonical visual-agent DROID ids are:

    droid_100_episode_<processed_idx:03d>_<raw_id_slug>

where ``raw_id_slug`` is the DROID raw episode id with non-identifier
characters normalized to underscores.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


DROID_100_EPISODE_PREFIX = "droid_100_episode"
DROID_100_EPISODE_ID_RE = re.compile(
    rf"^{DROID_100_EPISODE_PREFIX}_(?P<processed_idx>\d{{3}})_(?P<raw_id_slug>[A-Za-z0-9_]+)$"
)

# General episode ids must be importable as ``visual_<id>.py``.
# Legacy ``droid_100_episode_<idx>_<slug>`` ids also satisfy this pattern.
EPISODE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def raw_id_slug(raw_id: str) -> str:
    slug = raw_id.replace("-", "_")
    slug = re.sub(r"[^A-Za-z0-9_]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    assert slug, raw_id
    return slug


def episode_id_for(processed_idx: int, raw_id: str) -> str:
    return f"{DROID_100_EPISODE_PREFIX}_{processed_idx:03d}_{raw_id_slug(raw_id)}"


def is_droid_100_episode_id(episode_id: str) -> bool:
    return DROID_100_EPISODE_ID_RE.match(episode_id) is not None


def validate_droid_100_episode_id(episode_id: str) -> str:
    if not is_droid_100_episode_id(episode_id):
        raise ValueError(
            "episode_id must match "
            f"{DROID_100_EPISODE_PREFIX}_<processed_idx:03d>_<raw_id_slug>; "
            f"got {episode_id!r}"
        )
    return episode_id


def is_valid_episode_id(episode_id: str) -> bool:
    return EPISODE_ID_RE.match(episode_id) is not None


def validate_episode_id(episode_id: str) -> str:
    if not is_valid_episode_id(episode_id):
        raise ValueError(
            "episode_id must be identifier-safe "
            "(start with a letter, then letters/digits/underscores); "
            f"got {episode_id!r}"
        )
    return episode_id


def stage_manifest_path(raw_data_dir: Path) -> Path:
    return Path(raw_data_dir) / "stage_manifest.json"


def load_stage_manifest(raw_data_dir: Path) -> dict | None:
    path = stage_manifest_path(raw_data_dir)
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict), path
    return manifest


def resolve_droid_100_episode_id(
    raw_data_dir: Path,
    episode_id: str | None,
) -> str:
    """Resolve/validate the canonical episode id for a DROID staged folder."""
    manifest = load_stage_manifest(raw_data_dir)
    if manifest is not None:
        manifest_episode_id = manifest.get("episode_id")
        assert isinstance(manifest_episode_id, str), manifest
        validate_episode_id(manifest_episode_id)
        if episode_id is not None and episode_id != manifest_episode_id:
            raise ValueError(
                f"episode_id {episode_id!r} conflicts with "
                f"{stage_manifest_path(raw_data_dir)} episode_id "
                f"{manifest_episode_id!r}"
            )
        return manifest_episode_id

    if episode_id is None:
        raise ValueError(
            "DROID visual-agent raw-data builds require either an "
            "identifier-safe --episode-id or a staged raw-data folder with "
            "stage_manifest.json"
        )
    return validate_episode_id(episode_id)
