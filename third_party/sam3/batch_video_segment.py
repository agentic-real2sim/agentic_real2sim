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
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_") or "unnamed"


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "This script requires OpenCV (`cv2`) to read/write video files. "
            "Please install opencv-python in the current environment."
        )


def require_torch_and_predictor():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "This script requires PyTorch in the current environment."
        ) from exc

    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import SAM3 video predictor. Make sure the SAM3 environment "
            "is installed correctly."
        ) from exc

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
        nargs="+",
        default=TEXTS,
        help="Text prompts to run, one by one.",
    )
    parser.add_argument(
        "--prompt-frame-index",
        type=int,
        default=PROMPT_FRAME_INDEX,
        help="Frame index used for the initial text prompt.",
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
):
    require_cv2()
    text_name = sanitize_name(text)
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

            mask = get_mask_for_object(
                outputs_per_frame.get(frame_idx),
                obj_id=selected_obj_id,
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
):
    text_dir = output_root / sanitize_name(text)
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
        inference_state = predictor._ALL_INFERENCE_STATES[session_id]["state"]
        num_frames = int(inference_state["num_frames"])
        orig_h = int(inference_state["orig_height"])
        orig_w = int(inference_state["orig_width"])

        if not 0 <= args.prompt_frame_index < num_frames:
            raise ValueError(
                f"prompt_frame_index={args.prompt_frame_index} is out of range for "
                f"{num_frames} frames"
            )

        print(f"Session started: {session_id}")
        print(f"Video: {video_path}")
        print(f"Frames: {num_frames}, resolution: {orig_w}x{orig_h}")
        print(f"Texts: {args.texts}")

        default_thresholds = {
            "score_threshold_detection": predictor.model.score_threshold_detection,
            "new_det_thresh": predictor.model.new_det_thresh,
        }

        for text_idx, text in enumerate(args.texts):
            print(f"\n{'=' * 60}")
            print(f'Processing text prompt: "{text}"')

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

            response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=args.prompt_frame_index,
                    text=text,
                )
            )
            prompt_outputs = response["outputs"]
            num_detected = len(prompt_outputs["out_obj_ids"])
            print(
                f"Prompt frame {args.prompt_frame_index}: detected {num_detected} object(s)"
            )

            best = select_best_object(prompt_outputs)
            if best is None:
                print(f'No object found for "{text}", writing empty metadata only.')
                save_empty_result(
                    output_root=output_dir,
                    video_path=video_path,
                    text=text,
                    prompt_frame_index=args.prompt_frame_index,
                    thresholds=thresholds,
                )
                continue

            keep_obj_id = best["obj_id"]
            prompt_score = best["score"]
            print(
                f"Keeping best object id={keep_obj_id} on prompt frame "
                f"(score={prompt_score:.3f})"
            )

            removed_obj_ids = remove_other_objects(
                predictor,
                session_id=session_id,
                obj_ids=prompt_outputs["out_obj_ids"],
                keep_obj_id=keep_obj_id,
            )
            if removed_obj_ids:
                print(f"Removed other object ids: {removed_obj_ids}")

            outputs_per_frame = collect_propagation_outputs(
                predictor,
                session_id=session_id,
                prompt_frame_index=args.prompt_frame_index,
            )
            print(f"Collected outputs for {len(outputs_per_frame)} frame(s)")

            color_bgr = MASK_COLORS_BGR[text_idx % len(MASK_COLORS_BGR)]
            save_text_outputs(
                video_path=video_path,
                output_root=output_dir,
                text=text,
                outputs_per_frame=outputs_per_frame,
                selected_obj_id=keep_obj_id,
                prompt_score=prompt_score,
                prompt_frame_index=args.prompt_frame_index,
                overlay_alpha=args.overlay_alpha,
                color_bgr=color_bgr,
                thresholds=thresholds,
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
