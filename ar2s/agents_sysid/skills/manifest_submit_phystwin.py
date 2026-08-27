"""Phystwin manifest validation + submit tool factory.

Phystwin manifest schema (much simpler than droid since data is already
in canonical layout — no file copies needed)::

    schema_version: 1
    pipeline_kind: phystwin
    case_name: <string>          # = raw folder name (e.g., 'double_stretch_sloth')
    raw_dir: <abs path>          # absolute path to raw folder
    train_frame: <int>           # typically frame_num - 1
    n_cams: <int>                # = len(calibrate.pkl)
    fps: <int>                   # from metadata.json
    config_choice: cloth | real  # agent's choice (cloth = soft+self-coll, real = rope-like)
    config_reason: <string>      # one-line rationale (so a human can audit the choice)

Unlike droid, ``ScanResult.file_moves`` is always ``[]`` for phystwin
because raw data already lives in the canonical layout that
``ar2s.phystwin_sysid`` reads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml
from langchain_core.tools import tool

# Shared submission state and limits.
from ar2s.agents_sysid.skills.manifest_common import (
    SCHEMA_VERSION,
    ScanResult,
    _MAX_SUBMIT_ATTEMPTS,
    _SubmissionSlot,
)


_REQUIRED_RAW_FILES = ("calibrate.pkl", "metadata.json", "final_data.pkl")
_REQUIRED_RAW_DIRS = ("color",)
_VALID_CONFIG_CHOICES = ("cloth", "real")


def validate_phystwin_submission(
    manifest: Mapping[str, Any],
    raw_dir: Path,
) -> list[str]:
    """Return list of error strings (empty = valid)."""
    errors: list[str] = []

    sv = manifest.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION}")

    kind = manifest.get("pipeline_kind")
    if kind != "phystwin":
        errors.append(f"pipeline_kind must be 'phystwin', got {kind!r}")

    case = manifest.get("case_name")
    if not isinstance(case, str) or not case:
        errors.append("case_name must be a non-empty string")
    elif case != raw_dir.name:
        # Soft check: warn but don't fail — user might rename in scene.
        # Actually let's enforce for now to catch accidental mismatches.
        errors.append(
            f"case_name {case!r} doesn't match raw_dir folder name {raw_dir.name!r}"
        )

    # raw_dir field should match the actually-scanned raw_dir
    declared_raw = manifest.get("raw_dir")
    if not declared_raw:
        errors.append("raw_dir must be set")
    else:
        try:
            if Path(declared_raw).resolve() != raw_dir.resolve():
                errors.append(
                    f"raw_dir mismatch: manifest says {declared_raw!r}, "
                    f"scan was at {raw_dir!r}"
                )
        except Exception as e:
            errors.append(f"raw_dir is not a valid path: {e}")

    for k in ("train_frame", "n_cams", "fps"):
        v = manifest.get(k)
        if not isinstance(v, int) or v < 1:
            errors.append(f"{k} must be an integer >= 1, got {v!r}")

    cc = manifest.get("config_choice")
    if cc not in _VALID_CONFIG_CHOICES:
        errors.append(
            f"config_choice must be one of {_VALID_CONFIG_CHOICES}, got {cc!r}"
        )

    reason = manifest.get("config_reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("config_reason must be a non-empty string (audit trail for the cloth/real choice)")

    # Check raw_dir actually has phystwin layout
    for f in _REQUIRED_RAW_FILES:
        if not (raw_dir / f).is_file():
            errors.append(f"required file missing in raw_dir: {f}")
    for d in _REQUIRED_RAW_DIRS:
        if not (raw_dir / d).is_dir():
            errors.append(f"required directory missing in raw_dir: {d}/")

    return errors


def make_submit_phystwin_tool(slot: _SubmissionSlot, raw_dir: Path):
    """Build a `submit_manifest` tool bound to ``slot``. Phystwin variant.

    Unlike the droid version, the submission JSON has only one key:

        {"manifest_yaml": "<yaml string>"}

    (No ``file_moves`` because phystwin data is already in canonical layout.)
    """

    def _too_many() -> str:
        slot.aborted = True
        return (
            f"ABORT: max submission attempts ({_MAX_SUBMIT_ATTEMPTS}) reached. "
            "Stop calling submit_manifest. Common causes: missing required "
            "fields (schema_version/pipeline_kind/case_name/...), wrong "
            "config_choice value (must be 'cloth' or 'real'), or case_name "
            "not matching the raw_dir folder name."
        )

    @tool
    def submit_manifest(submission_json: str) -> str:
        """Submit your phystwin manifest as ONE JSON string.

        Pass a JSON object with exactly one top-level key::

          {"manifest_yaml": "<full manifest.yaml content as a YAML string>"}

        Required fields in the YAML (see system prompt for full schema):
          schema_version, pipeline_kind, case_name, raw_dir,
          train_frame, n_cams, fps, config_choice, config_reason.

        Returns:
            "OK: manifest validated and accepted." → you can stop.
            Otherwise an error message; fix and resubmit.
        """
        def _fail(msg: str, last_err: str | None = None) -> str:
            slot.failed_attempts += 1
            if last_err is not None:
                slot.last_error = last_err
            if slot.failed_attempts >= _MAX_SUBMIT_ATTEMPTS:
                return _too_many()
            return msg

        if slot.aborted:
            return _too_many()

        # Parse outer JSON
        try:
            decoder = json.JSONDecoder()
            data, _end = decoder.raw_decode(submission_json.lstrip())
        except json.JSONDecodeError as e:
            return _fail(
                f"VALIDATION ERROR: submission_json is not valid JSON: {e}. "
                "Pass a JSON string with one key 'manifest_yaml' (str).",
                last_err=f"submission_json parse error: {e}",
            )

        if not isinstance(data, dict):
            return _fail(
                f"VALIDATION ERROR: submission_json must be a JSON object, "
                f"got {type(data).__name__}.",
            )

        manifest_yaml = data.get("manifest_yaml")
        if manifest_yaml is None:
            return _fail(
                "VALIDATION ERROR: submission_json must contain the key "
                f"'manifest_yaml' (string). Got keys: {sorted(data.keys())}.",
                last_err="missing manifest_yaml key",
            )
        if not isinstance(manifest_yaml, str):
            return _fail(
                f"VALIDATION ERROR: 'manifest_yaml' must be a string; "
                f"got {type(manifest_yaml).__name__}.",
            )

        # Parse YAML
        try:
            manifest = yaml.safe_load(manifest_yaml)
        except Exception as e:
            return _fail(
                f"VALIDATION ERROR: cannot parse manifest_yaml as YAML: {e}",
                last_err=f"YAML parse error: {e}",
            )
        if not isinstance(manifest, dict):
            return _fail(
                f"VALIDATION ERROR: manifest_yaml did not parse to a dict; "
                f"got {type(manifest).__name__}",
            )

        # Schema + semantic checks
        errors = validate_phystwin_submission(manifest, raw_dir)
        if errors:
            return _fail(
                f"VALIDATION FAILED ({len(errors)} error(s)):\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\n\nFix and call submit_manifest again.",
                last_err="\n".join(errors),
            )

        # Accept
        slot.submitted = ScanResult(
            manifest=manifest,
            file_moves=[],     # phystwin: data already in canonical layout
            raw_dir=raw_dir,
            n_validation_attempts=slot.failed_attempts + 1,
        )
        return f"OK: phystwin manifest validated and accepted (attempts: {slot.failed_attempts + 1})."

    return submit_manifest


__all__ = [
    "validate_phystwin_submission",
    "make_submit_phystwin_tool",
]
