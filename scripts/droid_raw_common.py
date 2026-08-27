"""Pure helpers shared by DROID raw export/staging scripts.

These intentionally pull in no cv2/numpy/pyzed so they can be imported from a
pyzed-free orchestrator as well as from the GPU-bound SVO exporter.
"""

from pathlib import Path

MISSING_PREREQ_EXIT = 2


def find_svo_path(episode_dir: Path, camera_serial: str) -> Path | None:
    svo_dir = episode_dir / "recordings" / "SVO"
    for suffix in (".svo", ".svo2"):
        candidate = svo_dir / f"{camera_serial}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def normalize_serial(value):
    if isinstance(value, int):
        return f"{value:08d}"
    serial = str(value)
    if serial.isdigit() and len(serial) < 8:
        serial = serial.zfill(8)
    return serial


def load_raw_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        raw_ids = {line.strip() for line in f if line.strip()}
    assert raw_ids, path
    return raw_ids


def pointworld_cameras_path(
    episode_dir: Path,
    metadata: dict,
    *,
    pointworld_root: Path,
) -> tuple[Path | None, str]:
    user_id = metadata.get("user_id")
    if not user_id:
        return None, f"missing user_id in metadata"

    parts = episode_dir.name.split("_", maxsplit=2)
    if len(parts) != 3:
        return None, "unexpected episode dir format"

    lab_name, _, timestamp = parts
    timestamp_parts = timestamp.split("-")
    if len(timestamp_parts) != 6:
        return None, "unexpected timestamp format"

    year, month, day, hour, minute, second = timestamp_parts
    cameras_name = (
        f"{lab_name}+{user_id}+{year}-{month}-{day}-"
        f"{hour}h-{minute}m-{second}s_cameras.json"
    )
    path = pointworld_root / cameras_name
    if not path.is_file():
        return None, f"missing {path}"
    return path, ""
