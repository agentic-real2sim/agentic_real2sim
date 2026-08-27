"""Shared manifest submission types and constants.

These are used by the Episode Agent submission tools and the phystwin build
helper. Keeping them in one module avoids importing the droid-only validator
after that path has been removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1


@dataclass
class ScanResult:
    """Outcome of a successful Episode Agent run."""
    manifest: dict
    file_moves: list[dict]
    raw_dir: Path
    n_validation_attempts: int = 0


@dataclass
class _SubmissionSlot:
    """Captured by the agent runner; submit tools write the result here."""
    submitted: ScanResult | None = None
    failed_attempts: int = 0
    last_error: str = ""
    aborted: bool = False


_MAX_SUBMIT_ATTEMPTS = 3


__all__ = [
    "SCHEMA_VERSION",
    "ScanResult",
    "_SubmissionSlot",
    "_MAX_SUBMIT_ATTEMPTS",
]
