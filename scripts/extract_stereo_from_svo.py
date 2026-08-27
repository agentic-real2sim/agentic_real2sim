"""Extract rectified stereo MP4 + K.txt from DROID raw SVO recordings.

This is the only step in the raw-prep flow that requires the ZED SDK + pyzed +
a CUDA GPU. It is split out so the orchestrator (scripts.prepare_droid_episodes)
can invoke this as a subprocess. pyzed is imported unconditionally at module
load, and GPU availability is checked before doing any extraction work.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyzed.sl  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.droid_raw_common import (
    MISSING_PREREQ_EXIT,
    find_svo_path,
    load_raw_ids,
    normalize_serial,
)


def check_prerequisites() -> None:
    missing = []

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        missing.append(
            "nvidia-smi not found: ZED SDK SVO rectification requires a CUDA GPU"
        )
    else:
        try:
            subprocess.run([nvidia_smi], check=True, capture_output=True)
        except Exception as exc:
            missing.append(f"nvidia-smi failed ({exc}): no usable NVIDIA GPU")

    if missing:
        print("Cannot extract stereo MP4s from SVO; missing prerequisites:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print(
            "Install the ZED SDK and run on a host with a CUDA GPU, then retry.",
            file=sys.stderr,
        )
        sys.exit(MISSING_PREREQ_EXIT)


def collect_raw_ids(args) -> list[str]:
    raw_ids: set[str] = set(args.raw_id or [])
    from_file = load_raw_ids(Path(args.raw_ids_file) if args.raw_ids_file else None)
    if from_file:
        raw_ids |= from_file
    assert raw_ids, "no raw ids provided (use --raw_id and/or --raw_ids_file)"
    return sorted(raw_ids)


def extract_episode(
    raw_id: str,
    *,
    raw_root: Path,
    output_root: Path,
    extract_fn,
) -> None:
    episode_dir = raw_root / raw_id
    if not episode_dir.is_dir():
        print(f"Skipping {raw_id}: missing raw dir {episode_dir}")
        return

    metadata_files = sorted(episode_dir.glob("metadata_*.json"))
    if len(metadata_files) != 1:
        print(f"Skipping {raw_id}: expected 1 metadata_*.json, found {len(metadata_files)}")
        return
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))

    out_dir = output_root / raw_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for role in ("wrist", "ext1", "ext2"):
        key = f"{role}_cam_serial"
        if key not in metadata:
            continue
        serial = normalize_serial(metadata[key])
        svo_path = find_svo_path(episode_dir, serial)
        if svo_path is None:
            print(f"Skipping {raw_id} {role} {serial}: missing SVO")
            continue
        output_path = out_dir / f"{serial}-stereo.mp4"
        intrinsics_path = output_path.with_suffix(".K.txt")
        if output_path.exists() and intrinsics_path.exists():
            print(
                f"Skipping {raw_id} {role} {serial}: {output_path.name} and "
                f"{intrinsics_path.name} already exist"
            )
            continue
        print(f"Extracting {raw_id} {role} {serial} from {svo_path.name}")
        try:
            extract_fn(svo_path=svo_path, output_path=output_path, camera_serial=serial)
        except Exception as exc:
            print(f"Failed to extract {raw_id} {role} {serial}: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract rectified stereo MP4 + K.txt from DROID raw SVO files."
    )
    parser.add_argument(
        "--raw_id",
        action="append",
        default=[],
        help="Raw DROID episode id to export (repeatable).",
    )
    parser.add_argument(
        "--raw_ids_file",
        default=None,
        help="Optional newline-delimited file of raw episode ids.",
    )
    parser.add_argument("--raw_root", default="droid_data/raw")
    parser.add_argument("--output_root", default="outputs/export_episodes_from_raw")
    args = parser.parse_args()

    check_prerequisites()

    # Import the exporter after the GPU check so missing CUDA is reported cleanly.
    from scripts.export_episodes_from_raw import extract_stereo_mp4_from_svo

    raw_ids = collect_raw_ids(args)
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    for raw_id in raw_ids:
        extract_episode(
            raw_id,
            raw_root=raw_root,
            output_root=output_root,
            extract_fn=extract_stereo_mp4_from_svo,
        )


if __name__ == "__main__":
    main()
