# Author: Guanxiong Chen

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ar2s.agents_visual.droid_episode_id import (
    episode_id_for,
    raw_id_slug,
)

from scripts.generate_raw_download_lists import (
    extract_file_path,
    to_raw_episode_name,
)


METADATA_RE = re.compile(r"^episode_(\d{6})_metadata\.yaml$")
OPTIMIZED_EXTRINSICS_KEYS = ("refined_extrinsics", "vggt_extrinsics")


def processed_episode_index(metadata_path: Path) -> int:
    match = METADATA_RE.match(metadata_path.name)
    if not match:
        raise ValueError(f"Unexpected processed metadata filename: {metadata_path}")
    return int(match.group(1))


def normalize_serial(value):
    serial = str(value)
    if serial.isdigit() and len(serial) < 8:
        serial = serial.zfill(8)
    return serial


def copy_one(pattern: str, src_dir: Path, dst_dir: Path) -> Path:
    matches = sorted(src_dir.glob(pattern))
    assert len(matches) == 1, (pattern, src_dir, matches)
    dst = dst_dir / matches[0].name
    shutil.copy2(matches[0], dst)
    return dst


def discover_external_mp4s(
    raw_export_dir: Path,
    metadata: dict,
    cameras_json: dict,
):
    mp4s = []
    skipped = []
    for role, key in (("ext1", "ext1_cam_serial"), ("ext2", "ext2_cam_serial")):
        if key not in metadata:
            continue
        serial = normalize_serial(metadata[key])
        cam_entry = cameras_json.get(serial)
        if not isinstance(cam_entry, dict):
            skipped.append(f"{role} {serial}: no cameras.json entry")
            continue
        if not any(k in cam_entry for k in OPTIMIZED_EXTRINSICS_KEYS):
            skipped.append(
                f"{role} {serial}: no optimized extrinsics "
                f"({', '.join(OPTIMIZED_EXTRINSICS_KEYS)})"
            )
            continue

        mp4_path = raw_export_dir / f"{serial}-stereo.mp4"
        intrinsics_path = mp4_path.with_suffix(".K.txt")
        if not mp4_path.exists():
            skipped.append(f"{role} {serial}: missing exported stereo MP4 at {mp4_path}")
            continue
        if not intrinsics_path.exists():
            skipped.append(
                f"{role} {serial}: missing exported SDK/SVO intrinsics sidecar "
                f"at {intrinsics_path}"
            )
            continue
        mp4s.append((role, serial, mp4_path, intrinsics_path))
    return mp4s, skipped


def discover_wrist_mp4(raw_export_dir: Path, metadata: dict):
    """Locate the exported wrist-cam stereo MP4 + intrinsics, if any.

    Wrist has no PointWorld-refined extrinsics, so unlike
    ``discover_external_mp4s`` this needs no ``cameras_json`` check.
    """
    key = "wrist_cam_serial"
    if key not in metadata:
        return None, [f"wrist: no {key} in metadata"]
    serial = normalize_serial(metadata[key])
    mp4_path = raw_export_dir / f"{serial}-stereo.mp4"
    intrinsics_path = mp4_path.with_suffix(".K.txt")
    if not mp4_path.exists():
        return None, [f"wrist {serial}: missing exported stereo MP4 at {mp4_path}"]
    if not intrinsics_path.exists():
        return None, [
            f"wrist {serial}: missing exported SDK/SVO intrinsics sidecar "
            f"at {intrinsics_path}"
        ]
    return ("wrist", serial, mp4_path, intrinsics_path), []


def validate_existing_stage_manifest(manifest: dict, stage_dir: Path) -> list[str]:
    errors: list[str] = []
    if not stage_dir.is_dir():
        errors.append(f"stage_dir missing: {stage_dir}")
    copied = manifest.get("copied")
    if not isinstance(copied, dict):
        return ["manifest.copied missing or malformed"]
    for key in ("metadata", "trajectory", "cameras_json"):
        path = copied.get(key)
        if not path or not Path(path).exists():
            errors.append(f"manifest.copied.{key} missing or not found: {path}")
    mp4s = copied.get("mp4s")
    if not isinstance(mp4s, list) or not mp4s:
        errors.append("manifest.copied.mp4s is empty")
        return errors
    for i, entry in enumerate(mp4s):
        if not isinstance(entry, dict):
            errors.append(f"manifest.copied.mp4s[{i}] is not an object")
            continue
        if entry.get("source") != "raw_export":
            errors.append(
                f"manifest.copied.mp4s[{i}].source is {entry.get('source')!r}, "
                "expected 'raw_export'"
            )
        mp4_path = entry.get("path")
        if not mp4_path or not Path(mp4_path).exists():
            errors.append(f"manifest.copied.mp4s[{i}].path missing or not found: {mp4_path}")
        intrinsics_path = entry.get("sdk_intrinsics_path")
        if not intrinsics_path or not Path(intrinsics_path).exists():
            errors.append(
                f"manifest.copied.mp4s[{i}].sdk_intrinsics_path missing or not found: "
                f"{intrinsics_path}"
            )
    return errors


def stage_episode(
    metadata_path: Path,
    *,
    raw_root: Path,
    raw_export_root: Path,
    output_root: Path,
    overwrite: bool,
):
    processed_idx = processed_episode_index(metadata_path)
    file_path = extract_file_path(metadata_path)
    raw_id = to_raw_episode_name(file_path)
    episode_id = episode_id_for(processed_idx, raw_id)

    raw_dir = raw_root / raw_id
    raw_export_dir = raw_export_root / raw_id
    stage_dir = output_root / episode_id

    missing = []
    if not raw_dir.is_dir():
        missing.append(f"raw dir: {raw_dir}")
    if not raw_export_dir.is_dir():
        missing.append(f"raw export dir: {raw_export_dir}")
    if missing:
        return None, f"{episode_id}: missing " + "; ".join(missing)

    metadata_jsons = sorted(raw_dir.glob("metadata_*.json"))
    cameras_jsons = sorted(raw_export_dir.glob("*_cameras.json"))
    trajectory_path = raw_dir / "trajectory.h5"
    if len(metadata_jsons) != 1:
        return None, f"{episode_id}: expected 1 metadata_*.json, found {len(metadata_jsons)}"
    if len(cameras_jsons) != 1:
        return None, f"{episode_id}: expected 1 *_cameras.json, found {len(cameras_jsons)}"
    if not trajectory_path.exists():
        return None, f"{episode_id}: missing {trajectory_path}"

    metadata = json.loads(metadata_jsons[0].read_text(encoding="utf-8"))
    cameras_json = json.loads(cameras_jsons[0].read_text(encoding="utf-8"))
    mp4s, skipped_mp4s = discover_external_mp4s(
        raw_export_dir,
        metadata,
        cameras_json,
    )
    if not mp4s:
        return None, (
            f"{episode_id}: no usable external exported <serial>-stereo.mp4 "
            f"+ <serial>-stereo.K.txt with optimized extrinsics "
            f"({'; '.join(skipped_mp4s)})"
        )
    wrist_entry, wrist_skipped = discover_wrist_mp4(raw_export_dir, metadata)
    if wrist_entry is not None:
        mp4s.append(wrist_entry)
    skipped_mp4s = skipped_mp4s + wrist_skipped

    if stage_dir.exists():
        if not overwrite:
            manifest_path = stage_dir / "stage_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                had_legacy_build_fields = False
                for key in ("build_camera", "build_command"):
                    had_legacy_build_fields = key in manifest or had_legacy_build_fields
                    manifest.pop(key, None)
                if had_legacy_build_fields:
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2),
                        encoding="utf-8",
                    )
                manifest_errors = validate_existing_stage_manifest(manifest, stage_dir)
                if manifest_errors:
                    return None, (
                        f"{episode_id}: existing stage manifest violates rectified "
                        f"MP4 + K contract; pass --overwrite ({'; '.join(manifest_errors)})"
                    )
                return manifest, None
            return None, f"{episode_id}: stage dir exists without stage_manifest.json"
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    copied = {
        "metadata": str(copy_one("metadata_*.json", raw_dir, stage_dir)),
        "trajectory": str(shutil.copy2(trajectory_path, stage_dir / "trajectory.h5")),
        "cameras_json": str(copy_one("*_cameras.json", raw_export_dir, stage_dir)),
        "mp4s": [],
    }
    for role, serial, mp4_path, intrinsics_path in mp4s:
        dst = stage_dir / mp4_path.name
        shutil.copy2(mp4_path, dst)
        intrinsics_dst = stage_dir / intrinsics_path.name
        shutil.copy2(intrinsics_path, intrinsics_dst)
        copied["mp4s"].append({
            "role": role,
            "serial": serial,
            "source": "raw_export",
            "source_path": str(mp4_path),
            "path": str(dst),
            "sdk_intrinsics_path": str(intrinsics_dst),
        })

    manifest = {
        "processed_episode_index": processed_idx,
        "processed_metadata_path": str(metadata_path),
        "raw_id": raw_id,
        "raw_id_slug": raw_id_slug(raw_id),
        "episode_id": episode_id,
        "raw_dir": str(raw_dir),
        "raw_export_dir": str(raw_export_dir),
        "stage_dir": str(stage_dir),
        "copied": copied,
        "skipped_external_cameras": skipped_mp4s,
    }
    (stage_dir / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest, None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage DROID raw/export artifacts into pipeline raw_data folders "
            "named droid_100_episode_<processed_index:03d>_<raw_id_slug>."
        )
    )
    parser.add_argument(
        "--metadata_root",
        default="outputs/export_episodes_from_processed/droid_100/train",
        help="Root containing episode_*_metadata.yaml from the processed export.",
    )
    parser.add_argument(
        "--raw_root",
        default="droid_data/raw",
        help="Root containing downloaded raw DROID episode directories.",
    )
    parser.add_argument(
        "--raw_export_root",
        default="outputs/export_episodes_from_raw",
        help="Root containing exported raw artifacts.",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/pipeline_droid_raw_data",
        help="Output root for staged pipeline raw_data folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing staged episode directories.",
    )
    args = parser.parse_args()

    metadata_root = Path(args.metadata_root)
    raw_root = Path(args.raw_root)
    raw_export_root = Path(args.raw_export_root)
    output_root = Path(args.output_root)

    assert metadata_root.is_dir(), metadata_root
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_files = sorted(metadata_root.rglob("episode_*_metadata.yaml"))
    assert metadata_files, metadata_root

    manifests = []
    skipped = []
    for metadata_path in metadata_files:
        manifest, reason = stage_episode(
            metadata_path,
            raw_root=raw_root,
            raw_export_root=raw_export_root,
            output_root=output_root,
            overwrite=args.overwrite,
        )
        if manifest is None:
            skipped.append(reason)
        else:
            manifests.append(manifest)

    manifest_path = output_root / "staged_episodes.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")

    print(f"Scanned processed metadata files: {len(metadata_files)}")
    print(f"Staged episodes: {len(manifests)}")
    print(f"Skipped episodes: {len(skipped)}")
    print(f"Wrote manifest to: {manifest_path}")

    if skipped:
        print("\nSkipped:")
        for reason in skipped[:40]:
            print(f"- {reason}")
        if len(skipped) > 40:
            print(f"... and {len(skipped) - 40} more")


if __name__ == "__main__":
    main()
