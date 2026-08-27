#!/usr/bin/env python3
"""
Multi-object tracking with SAM 3.1 Object Multiplex.

Unlike batch_video_segment.py which tracks each text prompt independently
(resetting the session between prompts), this script leverages Object Multiplex
to track ALL objects simultaneously in a single session. Each text prompt can
detect multiple instances, and all are tracked jointly.

Outputs:
- Per-object binary mask PNGs (one directory per object)
- A single combined overlay MP4 showing all tracked objects
- A metadata JSON summarizing the run
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, os.path.dirname(__file__))

# ── Defaults ─────────────────────────────────────────────────────────────────
VIDEO_PATH = "/home/jpeng30/foundation_pose/FoundationStereo/episode_4_raw_videos/25947356.mp4"
OUTPUT_DIR = "/home/jpeng30/foundation_pose/FoundationStereo/mot_results_25947356"
TEXTS = ["pen", "mug"]
PROMPT_FRAME_INDEX = 0
OVERLAY_ALPHA = 0.45

# Per-text threshold overrides (same as batch_video_segment.py)
TEXT_THRESHOLD_OVERRIDES = {
    "pen": {
        "score_threshold_detection": 0.3,
        "new_det_thresh": 0.5,
    },
}

# Distinct colors (BGR) for up to 16 objects; will cycle if more.
MASK_COLORS_BGR = [
    (0, 255, 0),     # green
    (0, 128, 255),   # orange
    (255, 0, 0),     # blue
    (255, 255, 0),   # cyan
    (0, 0, 255),     # red
    (255, 0, 255),   # magenta
    (128, 255, 0),   # lime
    (0, 255, 255),   # yellow
    (255, 128, 0),   # sky blue
    (128, 0, 255),   # purple
]


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_") or "unnamed"


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "This script requires OpenCV (`cv2`). "
            "Install with: pip install opencv-python"
        )


def require_torch_and_predictor():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required.") from exc

    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import SAM 3.1 multiplex video predictor."
        ) from exc

    return torch, build_sam3_multiplex_video_predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-object tracking with SAM 3.1 Object Multiplex.",
    )
    parser.add_argument("--video-path", default=VIDEO_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--texts", nargs="+", default=TEXTS,
                        help="Text prompts. Each may detect multiple instances.")
    parser.add_argument("--prompt-frame-index", type=int, default=PROMPT_FRAME_INDEX)
    parser.add_argument("--overlay-alpha", type=float, default=OVERLAY_ALPHA)
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device id (default: current device).")
    return parser.parse_args()


def get_video_info(video_path: Path):
    require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        fps = 30.0
    return {"fps": float(fps), "width": width, "height": height, "frame_count": frame_count}


def overlay_masks_on_frame(frame_bgr, obj_ids, binary_masks, obj_meta, alpha):
    """Draw all object masks on a single frame."""
    require_cv2()
    output = frame_bgr.copy()
    for i, obj_id in enumerate(obj_ids):
        obj_id = int(obj_id)
        mask = np.asarray(binary_masks[i], dtype=bool)
        if not mask.any():
            continue

        meta = obj_meta.get(obj_id, {})
        color_bgr = meta.get("color", MASK_COLORS_BGR[obj_id % len(MASK_COLORS_BGR)])
        label = meta.get("label", f"obj_{obj_id}")

        # Blend mask color
        color = np.asarray(color_bgr, dtype=np.float32)
        blended = output[mask].astype(np.float32) * (1.0 - alpha) + color * alpha
        output[mask] = blended.astype(np.uint8)

        # Bounding box
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cv2.rectangle(output, (x0, y0), (x1, y1), color_bgr, 2)

        # Label background
        label_bg_top = max(0, y0 - 28)
        label_bg_bottom = max(24, y0)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        cv2.rectangle(
            output,
            (x0, label_bg_top),
            (min(output.shape[1] - 1, x0 + tw + 12), label_bg_bottom),
            color_bgr,
            thickness=-1,
        )
        cv2.putText(
            output, label,
            (x0 + 6, label_bg_bottom - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA,
        )

    return output


def main():
    args = parse_args()
    video_path = resolve_path(args.video_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    require_cv2()
    torch, build_sam3_multiplex_video_predictor = require_torch_and_predictor()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM 3.1.")

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # ── Build multiplex predictor ─────────────────────────────────────────
    print("Building SAM 3.1 multiplex video predictor...")
    t0 = time.time()
    predictor = build_sam3_multiplex_video_predictor(
        max_num_objects=args.max_num_objects,
        use_fa3=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ── Start session ─────────────────────────────────────────────────────
    session_id = None
    try:
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path),
                offload_video_to_cpu=True,
            )
        )
        session_id = response["session_id"]
        print(f"Session started: {session_id}")

        # Get frame info from the inference state
        inference_state = predictor._all_inference_states[session_id]["state"]
        num_frames = int(inference_state["num_frames"])
        orig_h = int(inference_state["orig_height"])
        orig_w = int(inference_state["orig_width"])
        print(f"Video: {video_path}")
        print(f"Frames: {num_frames}, resolution: {orig_w}x{orig_h}")

        if not 0 <= args.prompt_frame_index < num_frames:
            raise ValueError(
                f"prompt_frame_index={args.prompt_frame_index} out of range "
                f"(video has {num_frames} frames)"
            )

        # Save default thresholds for restore
        default_thresholds = {
            "score_threshold_detection": predictor.model.score_threshold_detection,
            "new_det_thresh": predictor.model.new_det_thresh,
        }

        # ── Per-text prompt: detect, propagate, reset, merge ─────────────
        # SAM 3.1 text prompts are global (each overwrites the previous),
        # so we follow the official pattern: run each text prompt as a
        # separate propagation pass, then merge results with offset obj_ids.
        #
        # Maps global_obj_id -> metadata
        obj_meta = {}
        # Maps frame_idx -> {global_obj_id: binary_mask}
        merged_masks = {}
        global_color_idx = 0
        next_global_id = 0
        propagation_direction = "both" if args.prompt_frame_index > 0 else "forward"

        for text in args.texts:
            print(f"\n{'=' * 60}")
            print(f'Processing text prompt: "{text}"')

            # Apply per-text threshold overrides
            thresholds = dict(default_thresholds)
            overrides = TEXT_THRESHOLD_OVERRIDES.get(text.lower())
            if overrides:
                thresholds.update(overrides)
            predictor.model.score_threshold_detection = thresholds["score_threshold_detection"]
            predictor.model.new_det_thresh = thresholds["new_det_thresh"]
            print(f"  Thresholds: score_det={thresholds['score_threshold_detection']}, "
                  f"new_det={thresholds['new_det_thresh']}")

            # Reset session before each new text prompt
            predictor.handle_request(
                request=dict(type="reset_session", session_id=session_id)
            )

            # Add text prompt
            response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=args.prompt_frame_index,
                    text=text,
                )
            )
            prompt_outputs = response["outputs"]
            detected_ids = np.asarray(prompt_outputs["out_obj_ids"])
            detected_scores = np.asarray(prompt_outputs.get("out_probs", []))
            print(f"  Detected {len(detected_ids)} object(s): {detected_ids.tolist()}")

            if len(detected_ids) == 0:
                print(f'  No objects found for "{text}", skipping.')
                continue

            # Propagate this text prompt through the video
            print(f"  Propagating...")
            for resp in tqdm(
                predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        start_frame_index=args.prompt_frame_index,
                        propagation_direction=propagation_direction,
                    )
                ),
                desc=f"  {text}",
                total=num_frames,
            ):
                fidx = resp["frame_index"]
                out = resp["outputs"]
                out_obj_ids = np.asarray(out["out_obj_ids"])
                out_masks = np.asarray(out["out_binary_masks"])

                if fidx not in merged_masks:
                    merged_masks[fidx] = {}

                for i, local_oid in enumerate(out_obj_ids):
                    global_oid = int(local_oid) + next_global_id
                    merged_masks[fidx][global_oid] = out_masks[i]

            # Build metadata for objects from this prompt
            for i, local_oid in enumerate(detected_ids):
                global_oid = int(local_oid) + next_global_id
                score = float(detected_scores[i]) if i < len(detected_scores) else None
                color = MASK_COLORS_BGR[global_color_idx % len(MASK_COLORS_BGR)]
                label = f"{text}#{int(local_oid)}"
                if score is not None:
                    label += f" ({score:.2f})"
                obj_meta[global_oid] = {
                    "text": text,
                    "obj_id": global_oid,
                    "local_obj_id": int(local_oid),
                    "score": score,
                    "color": color,
                    "label": label,
                }
                global_color_idx += 1

            # Offset for next text prompt so obj_ids don't collide
            max_local_id = int(detected_ids.max()) if len(detected_ids) > 0 else 0
            next_global_id += max_local_id + 1

        # Restore default thresholds
        predictor.model.score_threshold_detection = default_thresholds["score_threshold_detection"]
        predictor.model.new_det_thresh = default_thresholds["new_det_thresh"]

        if not obj_meta:
            print("\nNo objects detected for any text prompt. Exiting.")
            return

        print(f"\n{'=' * 60}")
        print(f"Tracked {len(obj_meta)} object(s) across {num_frames} frames:")
        for oid, meta in sorted(obj_meta.items()):
            print(f"  global_id={oid}: {meta['label']}")

        # ── Save outputs ──────────────────────────────────────────────────
        print("\nSaving results...")
        video_info = get_video_info(video_path)
        height, width = video_info["height"], video_info["width"]

        # Create per-object mask directories
        mask_dirs = {}
        for oid in obj_meta:
            d = output_dir / "masks" / f"obj_{oid}"
            d.mkdir(parents=True, exist_ok=True)
            mask_dirs[oid] = d

        # Open video reader and overlay writer
        overlay_path = output_dir / "overlay.mp4"
        cap = cv2.VideoCapture(str(video_path))
        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_info["fps"],
            (width, height),
        )

        frame_idx = 0
        pbar = tqdm(total=video_info["frame_count"], desc="Saving")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_masks = merged_masks.get(frame_idx, {})

            # Build arrays for overlay
            out_obj_ids = []
            out_masks_list = []
            for oid in sorted(frame_masks.keys()):
                out_obj_ids.append(oid)
                out_masks_list.append(frame_masks[oid])

            if out_obj_ids:
                out_obj_ids_arr = np.array(out_obj_ids, dtype=int)
                out_masks_arr = np.stack(out_masks_list)
            else:
                out_obj_ids_arr = np.array([], dtype=int)
                out_masks_arr = np.zeros((0, height, width), dtype=np.uint8)

            # Save per-object masks
            for oid in obj_meta:
                if oid in frame_masks:
                    mask = np.asarray(frame_masks[oid], dtype=np.uint8)
                else:
                    mask = np.zeros((height, width), dtype=np.uint8)
                cv2.imwrite(str(mask_dirs[oid] / f"{frame_idx:06d}.png"), mask * 255)

            # Write overlay frame
            overlay = overlay_masks_on_frame(
                frame, out_obj_ids_arr, out_masks_arr, obj_meta, args.overlay_alpha
            )
            writer.write(overlay)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()

        # ── Save metadata ─────────────────────────────────────────────────
        metadata = {
            "video_path": str(video_path),
            "texts": args.texts,
            "prompt_frame_index": args.prompt_frame_index,
            "num_frames": num_frames,
            "resolution": [orig_w, orig_h],
            "objects": {
                str(oid): {
                    "text": m["text"],
                    "score": m["score"],
                    "mask_dir": str(mask_dirs[oid]),
                }
                for oid, m in obj_meta.items()
            },
            "overlay_video": str(overlay_path),
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Done! Results saved to: {output_dir}")
        print(f"  Overlay video: {overlay_path}")
        print(f"  Per-object masks: {output_dir / 'masks'}")
        print(f"  Metadata: {meta_path}")

    finally:
        if session_id is not None:
            try:
                predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass
        predictor.shutdown()


if __name__ == "__main__":
    main()
