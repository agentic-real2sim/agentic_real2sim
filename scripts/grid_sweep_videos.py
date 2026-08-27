"""Tile per-sample grasp_video.mp4 files into one spatial-grid video.

Instead of stitching the N successful sample videos end-to-end, this
makes a single mosaic where each cell plays one sample's video
synchronously. Each cell is labelled with its sample index ("00",
"01", ...). The grid is roughly square — cols = ceil(sqrt(N)),
rows = ceil(N/cols), with any unused cells painted black.

Output: ``<sweep_dir>/grid.mp4`` (sibling of samples/).
Skips the top-level ``can_be_refined/`` dir.

Usage:
    python scripts/grid_sweep_videos.py --batch-name droid_400
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path.home() / "Research" / "droid_sim_data"
SKIP_TOP_DIRS = {"can_be_refined"}

CELL_W = 320
CELL_H = 180
LABEL_FONT_SIZE = 28


def numeric_sort_key(p: Path) -> tuple:
    m = re.search(r"(\d+)", p.parent.name)
    return (int(m.group(1)) if m else 0, p.parent.name)


def build_filter_graph(input_paths: list[Path]) -> tuple[str, int, int]:
    """Return (filter_complex, cols, rows) for the given input list.

    Each input is scaled to CELL_W x CELL_H, gets a label burned into
    the top-left corner (sample index from the parent dir name), and is
    then placed in an xstack grid. xstack pads any unused cells with
    black via the ``fill=black`` option.
    """
    n = len(input_paths)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))

    parts: list[str] = []
    cell_labels: list[str] = []
    for i, p in enumerate(input_paths):
        sample_id = p.parent.name      # e.g. "00", "01"
        # scale, then drawtext with the sample id.
        # ``box=1:boxcolor=black@0.6`` adds a translucent box behind the
        # text so it stays legible over any background.
        parts.append(
            f"[{i}:v]scale={CELL_W}:{CELL_H},setsar=1,"
            f"drawtext=text='{sample_id}':"
            f"x=6:y=4:fontcolor=white:fontsize={LABEL_FONT_SIZE}:"
            f"box=1:boxcolor=black@0.6:boxborderw=4"
            f"[v{i}]"
        )
        cell_labels.append(f"[v{i}]")

    # xstack layout: x_y per input, in row-major order
    layout_pieces = []
    for i in range(n):
        r, c = divmod(i, cols)
        layout_pieces.append(f"{c * CELL_W}_{r * CELL_H}")
    layout = "|".join(layout_pieces)

    # fill=black handles unused grid cells (when n < cols*rows)
    xstack = (
        "".join(cell_labels) +
        f"xstack=inputs={n}:layout={layout}:fill=black[out]"
    )
    return ";".join(parts) + ";" + xstack, cols, rows


def make_grid(sweep_dir: Path, out_name: str = "grid.mp4",
              overwrite: bool = False) -> tuple[str, int]:
    samples_dir = sweep_dir / "samples"
    if not samples_dir.is_dir():
        return "skip-no-samples", 0
    videos = sorted(samples_dir.glob("*/grasp_video.mp4"), key=numeric_sort_key)
    if not videos:
        return "skip-empty", 0

    out_path = sweep_dir / out_name
    if out_path.exists() and not overwrite:
        return "skip-exists", len(videos)

    filter_complex, cols, rows = build_filter_graph(videos)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for v in videos:
        cmd += ["-i", str(v.resolve())]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    rc = subprocess.run(cmd, capture_output=True).returncode
    if rc != 0:
        return f"FAIL:ffmpeg_rc{rc}", len(videos)
    return f"OK ({cols}x{rows})", len(videos)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-name", default="droid_400")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="Parent of per-batch dirs (default %(default)s).")
    p.add_argument("--out-name", default="grid.mp4")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    artifacts_root = Path(args.data_root).expanduser() / args.batch_name / "artifacts"
    if not artifacts_root.is_dir():
        print(f"ERROR: {artifacts_root} does not exist", flush=True)
        return 2

    n_ok = n_skip = n_fail = n_eps = 0
    for ep_dir in sorted(artifacts_root.iterdir()):
        if not ep_dir.is_dir() or ep_dir.name in SKIP_TOP_DIRS:
            continue
        n_eps += 1
        for pass_label in ("sweep_anchor", "sweep_tip20"):
            pass_root = ep_dir / pass_label
            if not pass_root.is_dir():
                continue
            for sweep_dir in pass_root.iterdir():
                if not sweep_dir.is_dir() or not sweep_dir.name.startswith("sweep-"):
                    continue
                status, n = make_grid(sweep_dir, args.out_name,
                                       overwrite=args.overwrite)
                tag = f"{ep_dir.name}/{pass_label}"
                if status.startswith("OK"):
                    n_ok += 1
                    print(f"  ✓ {tag}: {n} clips → {args.out_name} {status}", flush=True)
                elif status.startswith("skip"):
                    n_skip += 1
                    if status != "skip-empty":
                        print(f"  - {tag}: {status}", flush=True)
                else:
                    n_fail += 1
                    print(f"  ✗ {tag}: {status}", flush=True)

    print()
    print(f"[done] {n_eps} eps walked — {n_ok} OK, {n_skip} skipped, {n_fail} FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
