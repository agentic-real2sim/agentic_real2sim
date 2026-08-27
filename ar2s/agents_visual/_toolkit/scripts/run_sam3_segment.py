#!/usr/bin/env python3
"""
Batch video version: segment target objects from an MP4 video with SAM3 video
predictor.

Compared with `batch_segment.py`, this script:
- uses the SAM3 video predictor instead of the image model
- takes an MP4 file as input
- runs one text prompt at a time on a chosen prompt frame
- keeps only the highest-scoring object from that prompt frame
- propagates that object through the video

A job may instead carry a pixel ``box``. Box jobs do NOT use the text path or
propagate_in_video: SAM3's tracker loses low-texture objects (a matte black
bowl) the instant they leave the seeded frame, collapsing propagation to the
prompt frame or nothing. Box jobs re-prompt the box on every frame as an
independent detection (scan_box_through_video), which is robust for the static
scene objects box jobs target.

For each text prompt, it saves:
- one binary mask PNG per frame
- one overlay MP4 showing the tracked object
- one metadata JSON summarizing the run
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

VIDEO_PATH = "../foundation_pose/FoundationStereo/episode_4_raw_videos/25947356.mp4"
OUTPUT_DIR = "../foundation_pose/FoundationStereo/segment_results_video_25947356"
TEXTS = ["pen", "mug"]
PROMPT_FRAME_INDEX = 0
OVERLAY_ALPHA = 0.45
TEXT_THRESHOLD_OVERRIDES = {
    # `pen` is typically smaller/harder, so we use looser detection thresholds.
    "pen": {
        "score_threshold_detection": 0.3,
        "new_det_thresh": 0.5,
    },
}
MASK_COLORS_BGR = [
    (0, 255, 0),
    (0, 128, 255),
    (255, 0, 0),
    (255, 255, 0),
]


def sanitize_name(text: str) -> str:
    # MUST mirror droid_sim._util.sanitize_name (this script runs in the
    # qianjun_sam3 subprocess env which doesn't have droid_sim installed,
    # so we keep a local copy. Any change here MUST be made in both files.)
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_") or "unnamed"


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def require_cv2():
    return None


def require_torch_and_predictor():
    return torch, build_sam3_video_predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM3 video predictor on a single MP4 video."
    )
    parser.add_argument(
        "--video-path",
        default=VIDEO_PATH,
        help="Path to the input MP4 video.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for masks, overlay videos, and metadata.",
    )
    parser.add_argument(
        "--texts",
        nargs="*",
        default=TEXTS,
        help="Text prompts to run (legacy / standalone-debug; ignored when "
             "--jobs-json is set). Pair with --prompt-frame-index / "
             "--text-frame-json to control the grounding frame. Output dirs "
             "are named after sanitize_name(text); use --jobs-json when you "
             "need explicit out_subdir per job.",
    )
    parser.add_argument(
        "--jobs-json",
        default=None,
        help=(
            "Path to a JSON list of segmentation jobs, each "
            '{"text": str, "frame_index": int, "out_subdir": str, '
            '"box"?: [x1,y1,x2,y2]}. When set, '
            "each job grounds its text on its own frame and writes to "
            "<output-dir>/<out_subdir>/ — this is the path used by the "
            "`segment` controller to batch one round of per-object candidates "
            "into a single subprocess (one SAM3 model load). An optional pixel "
            "`box` is sent alongside the text as a hybrid prompt. Supersedes "
            "--texts / --text-frame-json / --prompt-frame-index when set."
        ),
    )
    parser.add_argument(
        "--text-frame-json",
        type=str,
        default=None,
        help=(
            "Optional JSON dict {text: frame_idx} overriding --prompt-frame-index "
            "on a per-text basis (legacy; ignored when --jobs-json is set)."
        ),
    )
    parser.add_argument(
        "--prompt-frame-index",
        type=int,
        default=PROMPT_FRAME_INDEX,
        help="Frame index for --texts legacy path. Ignored when --jobs-json is set.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Specific CUDA GPU id to use. Default: current CUDA device.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=OVERLAY_ALPHA,
        help="Mask overlay alpha in the saved MP4.",
    )
    return parser.parse_args()


def build_jobs(args) -> list[dict]:
    """Resolve CLI args to a list of {text, frame_index, out_subdir, box?} jobs.

    Prefers --jobs-json (per-job frame + output subdir, used by the `segment`
    controller); falls back to the legacy --texts x --prompt-frame-index (with
    optional per-text override via --text-frame-json) for standalone runs.

    Each job's ``out_subdir`` defaults to ``sanitize_name(text)`` when not
    provided — preserving the legacy on-disk layout for --texts callers.
    ``box`` is optional and only ever set by --jobs-json callers.
    """
    if args.jobs_json:
        with open(args.jobs_json) as f:
            jobs = json.load(f)
        for j in jobs:
            j["frame_index"] = int(j["frame_index"])
            j.setdefault("out_subdir", sanitize_name(j["text"]))
        return jobs

    text_frame_overrides: dict = {}
    if args.text_frame_json:
        text_frame_overrides = json.loads(args.text_frame_json)
        if not isinstance(text_frame_overrides, dict):
            raise ValueError("--text-frame-json must be a JSON dict")

    return [
        {
            "text": t,
            "frame_index": int(text_frame_overrides.get(t, args.prompt_frame_index)),
            "out_subdir": sanitize_name(t),
        }
        for t in (args.texts or [])
    ]


def _normalised_xywh(box, W: int, H: int) -> list[float] | None:
    """Convert a pixel ``[x1,y1,x2,y2]`` box to the normalised xywh SAM3 wants.

    Returns None for a malformed or degenerate box so the caller can fall back
    to prompting on text alone.
    """
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0, min(W, x1)), max(0, min(W, x2))))
    y1, y2 = sorted((max(0, min(H, y1)), max(0, min(H, y2))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]


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

    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def select_best_object(outputs):
    out_obj_ids = np.asarray(outputs["out_obj_ids"])
    out_probs = np.asarray(outputs["out_probs"])
    if len(out_obj_ids) == 0:
        return None

    best_idx = int(out_probs.argmax())
    return {
        "best_idx": best_idx,
        "obj_id": int(out_obj_ids[best_idx]),
        "score": float(out_probs[best_idx]),
    }


def get_thresholds_for_text(text: str, default_thresholds):
    thresholds = dict(default_thresholds)
    overrides = TEXT_THRESHOLD_OVERRIDES.get(text.lower())
    if overrides:
        thresholds.update(overrides)
    return thresholds


def apply_thresholds_to_predictor(predictor, thresholds):
    predictor.model.score_threshold_detection = thresholds[
        "score_threshold_detection"
    ]
    predictor.model.new_det_thresh = thresholds["new_det_thresh"]


def remove_other_objects(predictor, session_id: str, obj_ids, keep_obj_id: int):
    removed_obj_ids = []
    for obj_id in np.asarray(obj_ids).tolist():
        obj_id = int(obj_id)
        if obj_id == keep_obj_id:
            continue
        predictor.handle_request(
            request=dict(
                type="remove_object",
                session_id=session_id,
                obj_id=obj_id,
                is_user_action=False,
            )
        )
        removed_obj_ids.append(obj_id)
    return removed_obj_ids


# Box jobs don't use propagate_in_video: SAM3's tracker loses low-texture
# objects (a matte black bowl) the instant they leave the seeded frame, so
# propagation collapses to the prompt frame (or, in forward mode, to nothing).
# Instead we RE-PROMPT the same box on every frame — each is an independent
# box detection, robust to an untrackable-but-static object — and stop a side
# scan when the object stops matching (it moved out of the fixed box).
_BOX_SCORE_DROP_FACTOR = 0.5     # frame score < seed_score * this → stop the scan
_BOX_AREA_CHANGE_MAX = 0.3       # |area - seed_area| / seed_area > this → stop the scan


def reprompt_box_one_frame(predictor, session_id, frame_idx, box_norm, height, width):
    """Box-prompt ``frame_idx`` in isolation; return (outputs, score, area).

    Returns (None, 0.0, 0) when SAM3 finds nothing usable. ``add_prompt``
    resets the session state internally, so each call is an independent
    single-frame box detection with no cross-frame tracking — that is exactly
    what makes it robust where propagate_in_video is not.
    """
    out = predictor.handle_request(
        request=dict(
            type="add_prompt", session_id=session_id, frame_index=frame_idx,
            text="visual", bounding_boxes=[box_norm], bounding_box_labels=[1],
        )
    )["outputs"]
    best = select_best_object(out)
    if best is None:
        return None, 0.0, 0
    area = int((get_mask_for_object(out, best["obj_id"], height, width) > 0).sum())
    if area == 0:
        return None, 0.0, 0
    return out, float(best["score"]), area


def scan_box_through_video(
    predictor, session_id, box_norm, height, width,
    *, prompt_frame_index: int, num_frames: int, seed_score: float, seed_area: int,
):
    """Re-prompt ``box_norm`` on every frame; return ``{frame_idx: (outputs, obj_id)}``.

    Walks forward then backward from the prompt frame, stopping a side scan when
    the per-frame score drops below ``seed_score * _BOX_SCORE_DROP_FACTOR`` or the
    mask area drifts more than ``_BOX_AREA_CHANGE_MAX`` — i.e. the fixed box no
    longer bounds the object (it moved). The prompt frame is not included; the
    caller seeds it.
    """
    per_frame: dict[int, tuple] = {}
    score_floor = seed_score * _BOX_SCORE_DROP_FACTOR

    def _stop(label, frame, score, area):
        if score < score_floor:
            print(f"  box scan {label}: frame {frame} score {score:.3f} < "
                  f"{score_floor:.3f}, stop")
            return True
        if seed_area > 0 and abs(area - seed_area) / seed_area > _BOX_AREA_CHANGE_MAX:
            print(f"  box scan {label}: frame {frame} area {area} drifted from seed "
                  f"{seed_area}, stop")
            return True
        return False

    for rng, label in ((range(prompt_frame_index + 1, num_frames), "forward"),
                       (range(prompt_frame_index - 1, -1, -1), "backward")):
        for f in rng:
            out, score, area = reprompt_box_one_frame(
                predictor, session_id, f, box_norm, height, width,
            )
            if out is None or _stop(label, f, score, area):
                break
            per_frame[f] = (out, select_best_object(out)["obj_id"])

    return per_frame


def collect_propagation_outputs(predictor, session_id: str, prompt_frame_index: int):
    propagation_direction = "both" if prompt_frame_index > 0 else "forward"
    outputs_per_frame = {}

    for response in tqdm(
        predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                start_frame_index=prompt_frame_index,
                propagation_direction=propagation_direction,
            )
        ),
        desc="Propagating",
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def get_mask_for_object(outputs, obj_id: int, height: int, width: int):
    empty_mask = np.zeros((height, width), dtype=np.uint8)
    if outputs is None:
        return empty_mask

    out_obj_ids = np.asarray(outputs["out_obj_ids"])
    if len(out_obj_ids) == 0:
        return empty_mask

    matches = np.where(out_obj_ids == obj_id)[0]
    if len(matches) == 0:
        return empty_mask

    mask = np.asarray(outputs["out_binary_masks"][int(matches[0])], dtype=np.uint8)
    if mask.shape != (height, width):
        raise ValueError(
            f"Unexpected mask shape {mask.shape}, expected {(height, width)}"
        )
    return mask


def overlay_mask(frame_bgr, mask, color_bgr, alpha: float, label: str):
    require_cv2()
    output = frame_bgr.copy()
    if mask.any():
        mask_bool = mask.astype(bool)
        color = np.asarray(color_bgr, dtype=np.float32)
        blended = (
            output[mask_bool].astype(np.float32) * (1.0 - alpha) + color * alpha
        )
        output[mask_bool] = blended.astype(np.uint8)

        ys, xs = np.where(mask_bool)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cv2.rectangle(output, (x0, y0), (x1, y1), color_bgr, 2)

        label_bg_top = max(0, y0 - 28)
        label_bg_bottom = max(24, y0)
        cv2.rectangle(
            output,
            (x0, label_bg_top),
            (min(output.shape[1] - 1, x0 + 260), label_bg_bottom),
            color_bgr,
            thickness=-1,
        )
        cv2.putText(
            output,
            label,
            (x0 + 6, label_bg_bottom - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def save_metadata(
    output_dir: Path,
    video_path: Path,
    text: str,
    prompt_frame_index: int,
    selected_obj_id,
    prompt_score,
    frame_count: int,
    overlay_video_path: Path = None,
    mask_dir: Path = None,
    thresholds: dict = None,
):
    metadata = {
        "video_path": str(video_path),
        "text": text,
        "prompt_frame_index": prompt_frame_index,
        "selected_obj_id": selected_obj_id,
        "prompt_score": prompt_score,
        "frame_count": frame_count,
        "overlay_video_path": str(overlay_video_path) if overlay_video_path else None,
        "mask_dir": str(mask_dir) if mask_dir else None,
        "thresholds": thresholds,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_text_outputs(
    video_path: Path,
    output_root: Path,
    text: str,
    outputs_per_frame,
    selected_obj_id: int,
    prompt_score: float,
    prompt_frame_index: int,
    overlay_alpha: float,
    color_bgr,
    thresholds: dict,
    out_subdir: str | None = None,
    obj_id_by_frame: dict | None = None,
):
    # ``obj_id_by_frame`` (box scan): each re-prompted frame is an independent
    # detection with its own object id, so the single ``selected_obj_id`` can't
    # address them all — override per frame when provided.
    require_cv2()
    # out_subdir: when the segment controller wants two jobs with the same text
    # at different keyframes / rounds, callers pass an explicit unique subdir.
    # Legacy --texts callers pass None → fall back to sanitize_name(text).
    text_name = out_subdir or sanitize_name(text)
    text_dir = output_root / text_name
    mask_dir = text_dir / "masks"
    overlay_video_path = text_dir / f"{text_name}_overlay.mp4"
    text_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    video_info = get_video_info(video_path)
    frame_count = video_info["frame_count"]
    height = video_info["height"]
    width = video_info["width"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to reopen video for saving: {video_path}")

    writer = cv2.VideoWriter(
        str(overlay_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_info["fps"],
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create overlay video: {overlay_video_path}")

    label = f"{text} | id={selected_obj_id} | score={prompt_score:.3f}"
    saved_frames = 0
    progress_bar = tqdm(
        total=frame_count if frame_count > 0 else None,
        desc=f"Saving {text_name}",
        leave=False,
    )

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_obj_id = (obj_id_by_frame or {}).get(frame_idx, selected_obj_id)
            mask = get_mask_for_object(
                outputs_per_frame.get(frame_idx),
                obj_id=frame_obj_id,
                height=height,
                width=width,
            )
            cv2.imwrite(str(mask_dir / f"{frame_idx:06d}.png"), mask * 255)

            overlay = overlay_mask(
                frame_bgr=frame,
                mask=mask,
                color_bgr=color_bgr,
                alpha=overlay_alpha,
                label=label,
            )
            writer.write(overlay)
            saved_frames += 1
            frame_idx += 1
            progress_bar.update(1)
    finally:
        progress_bar.close()
        cap.release()
        writer.release()

    save_metadata(
        output_dir=text_dir,
        video_path=video_path,
        text=text,
        prompt_frame_index=prompt_frame_index,
        selected_obj_id=selected_obj_id,
        prompt_score=prompt_score,
        frame_count=saved_frames,
        overlay_video_path=overlay_video_path,
        mask_dir=mask_dir,
        thresholds=thresholds,
    )


def save_empty_result(
    output_root: Path,
    video_path: Path,
    text: str,
    prompt_frame_index: int,
    thresholds: dict,
    out_subdir: str | None = None,
):
    text_dir = output_root / (out_subdir or sanitize_name(text))
    text_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(
        output_dir=text_dir,
        video_path=video_path,
        text=text,
        prompt_frame_index=prompt_frame_index,
        selected_obj_id=None,
        prompt_score=None,
        frame_count=0,
        thresholds=thresholds,
    )


def main():
    args = parse_args()
    video_path = resolve_path(args.video_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    require_cv2()
    torch, build_sam3_video_predictor = require_torch_and_predictor()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 video predictor requires CUDA, but CUDA is not available.")

    gpus_to_use = [args.gpu] if args.gpu is not None else None

    print("Building SAM3 video predictor...")
    t0 = time.time()
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    session_id = None
    try:
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path),
            )
        )
        session_id = response["session_id"]
        # Vendored from third_party/sam3/batch_video_segment.py (RyougiJarvis fork);
        # upstream sam3 (Meta) renamed _ALL_INFERENCE_STATES -> _all_inference_states
        # (sam3/model/sam3_base_predictor.py:36) and the fork hasn't tracked it.
        inference_state = predictor._all_inference_states[session_id]["state"]
        num_frames = int(inference_state["num_frames"])
        orig_h = int(inference_state["orig_height"])
        orig_w = int(inference_state["orig_width"])

        # build_jobs handles both --jobs-json (new, per-job out_subdir) and
        # --texts (legacy, out_subdir = sanitize_name(text)).
        jobs = build_jobs(args)
        if not jobs:
            raise SystemExit("nothing to do — --jobs-json/--texts is empty")

        # Range-check every job's frame_index against the video.
        for j in jobs:
            if not 0 <= j["frame_index"] < num_frames:
                raise ValueError(
                    f"job frame_index={j['frame_index']} (text={j['text']!r}, "
                    f"out_subdir={j['out_subdir']!r}) is out of range for {num_frames} frames"
                )

        print(f"Session started: {session_id}")
        print(f"Video: {video_path}")
        print(f"Frames: {num_frames}, resolution: {orig_w}x{orig_h}")
        print(f"Jobs: {[(j['text'], j['frame_index'], j['out_subdir']) for j in jobs]}")

        default_thresholds = {
            "score_threshold_detection": predictor.model.score_threshold_detection,
            "new_det_thresh": predictor.model.new_det_thresh,
        }

        for job_idx, job in enumerate(jobs):
            text = job["text"]
            text_frame_idx = int(job["frame_index"])
            out_subdir = job["out_subdir"]
            box = job.get("box")
            print(f"\n{'=' * 60}")
            print(f'Processing job {job_idx + 1}/{len(jobs)}: "{text}" @ frame {text_frame_idx} -> {out_subdir}'
                  + (f" (+box {box})" if box else ""))

            thresholds = get_thresholds_for_text(text, default_thresholds)
            apply_thresholds_to_predictor(predictor, thresholds)
            print(
                "Using thresholds: "
                f"score_threshold_detection={thresholds['score_threshold_detection']}, "
                f"new_det_thresh={thresholds['new_det_thresh']}"
            )

            predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=session_id,
                )
            )

            box_norm = _normalised_xywh(box, orig_w, orig_h) if box else None
            if box and box_norm is None:
                print(f'Bad box {box!r} for "{text}", prompting on text alone.')

            obj_id_by_frame: dict | None = None
            if box_norm is not None:
                # Box job: re-prompt the box on every frame (see
                # scan_box_through_video) rather than propagate — SAM3's tracker
                # drops low-texture objects the moment they leave the seed frame.
                seed_outputs, prompt_score, seed_area = reprompt_box_one_frame(
                    predictor, session_id, text_frame_idx, box_norm, orig_h, orig_w,
                )
                if seed_outputs is None:
                    print(f'Box gave no detection for "{text}" @ frame {text_frame_idx}, '
                          f'writing empty metadata only.')
                    save_empty_result(
                        output_root=output_dir, video_path=video_path, text=text,
                        prompt_frame_index=text_frame_idx, thresholds=thresholds,
                        out_subdir=out_subdir,
                    )
                    continue
                keep_obj_id = select_best_object(seed_outputs)["obj_id"]
                outputs_per_frame = {text_frame_idx: seed_outputs}
                obj_id_by_frame = {text_frame_idx: keep_obj_id}
                scanned = scan_box_through_video(
                    predictor, session_id, box_norm, orig_h, orig_w,
                    prompt_frame_index=text_frame_idx, num_frames=num_frames,
                    seed_score=prompt_score, seed_area=seed_area,
                )
                for f, (fo, oid) in scanned.items():
                    outputs_per_frame[f] = fo
                    obj_id_by_frame[f] = oid
                print(f"Box scan kept {len(outputs_per_frame)}/{num_frames} frame(s) "
                      f"(seed score={prompt_score:.3f}, area={seed_area})")
            else:
                # Text job: ground the text on the prompt frame, keep the best
                # detection, and let propagate_in_video track it (the text
                # re-grounds on every frame).
                response = predictor.handle_request(request=dict(
                    type="add_prompt", session_id=session_id,
                    frame_index=text_frame_idx, text=text,
                ))
                prompt_outputs = response["outputs"]
                print(f"Prompt frame {text_frame_idx}: detected "
                      f"{len(prompt_outputs['out_obj_ids'])} object(s)")

                best = select_best_object(prompt_outputs)
                if best is None:
                    print(f'No object found for "{text}" @ frame {text_frame_idx}, '
                          f'writing empty metadata only.')
                    save_empty_result(
                        output_root=output_dir, video_path=video_path, text=text,
                        prompt_frame_index=text_frame_idx, thresholds=thresholds,
                        out_subdir=out_subdir,
                    )
                    continue

                keep_obj_id = best["obj_id"]
                prompt_score = best["score"]
                print(f"Keeping best object id={keep_obj_id} on prompt frame "
                      f"(score={prompt_score:.3f})")

                removed_obj_ids = remove_other_objects(
                    predictor, session_id=session_id,
                    obj_ids=prompt_outputs["out_obj_ids"], keep_obj_id=keep_obj_id,
                )
                if removed_obj_ids:
                    print(f"Removed other object ids: {removed_obj_ids}")

                outputs_per_frame = collect_propagation_outputs(
                    predictor, session_id=session_id, prompt_frame_index=text_frame_idx,
                )
                print(f"Collected outputs for {len(outputs_per_frame)} frame(s)")

            color_bgr = MASK_COLORS_BGR[job_idx % len(MASK_COLORS_BGR)]
            save_text_outputs(
                video_path=video_path,
                output_root=output_dir,
                text=text,
                outputs_per_frame=outputs_per_frame,
                selected_obj_id=keep_obj_id,
                prompt_score=prompt_score,
                prompt_frame_index=text_frame_idx,
                overlay_alpha=args.overlay_alpha,
                color_bgr=color_bgr,
                thresholds=thresholds,
                out_subdir=out_subdir,
                obj_id_by_frame=obj_id_by_frame,
            )

        print(f"\n{'=' * 60}")
        print(f"All done! Results saved to: {output_dir}")
    finally:
        if session_id is not None:
            try:
                predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )
            except Exception:
                pass
        predictor.shutdown()


if __name__ == "__main__":
    main()
