import os
import sys
import cv2
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for rel_path in (
    "third_party/Grounded-SAM-2_phystwin",
):
    third_party_path = PROJECT_ROOT / rel_path
    if third_party_path.exists():
        sys.path.insert(0, str(third_party_path))

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import json
from argparse import ArgumentParser

"""
Hyperparam for Ground and Tracking
"""

# Put below base path into args
parser = ArgumentParser()
parser.add_argument("--base_path", type=str, default="/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect")
parser.add_argument("--case_name", type=str)
parser.add_argument("--TEXT_PROMPT", type=str)  # must be lowercase and end with period, e.g. "sloth. hand."
parser.add_argument("--camera_idx", type=int)
parser.add_argument("--output_path", type=str, default="NONE")
parser.add_argument("--box_threshold", type=float, default=0.35)
parser.add_argument("--text_threshold", type=float, default=0.25)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT
camera_idx = args.camera_idx
if args.output_path == "NONE":
    output_path = f"{base_path}/{case_name}"
else:
    output_path = args.output_path

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


CHECKPOINT_DIR = Path(os.environ.get("PHYSTWIN_MODELS_ROOT", str(PROJECT_ROOT / "models"))) / "grounded_sam_2"
BOX_THRESHOLD = args.box_threshold
TEXT_THRESHOLD = args.text_threshold
PROMPT_TYPE_FOR_VIDEO = "box"  # choose from ["point", "box", "mask"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_PATH = f"{base_path}/{case_name}/color/{camera_idx}.mp4"
existDir(f"{base_path}/{case_name}/tmp_data")
existDir(f"{base_path}/{case_name}/tmp_data/{case_name}")
existDir(f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}")

SOURCE_VIDEO_FRAME_DIR = f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}"

"""
Step 1: Environment settings and model initialization for Grounding DINO and SAM 2
"""
# build groundingdino from huggingface
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"   # SwinT-OGC equivalent; matches your swint_ogc checkpoint
processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

# init sam image predictor and video predictor model
sam2_checkpoint = str(CHECKPOINT_DIR / "sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)


video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

# saving video to frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)

# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

ann_frame_idx = 0  # the frame index we interact with
"""
Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
"""

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image = Image.open(img_path).convert("RGB")
image_source = np.array(image)

# caption must be lowercased and end with a period
caption = TEXT_PROMPT.lower().strip()
if not caption.endswith("."):
    caption += "."

inputs = processor(images=image, text=caption, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    target_sizes=[image.size[::-1]],   # (H, W)
)[0]

input_boxes = results["boxes"].cpu().numpy()           # absolute xyxy already
confidences = results["scores"].cpu().numpy().tolist()
class_names = results.get("text_labels", results.get("labels"))  # version-robust

print(input_boxes)

# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(image_source)

# process the detection results
OBJECTS = class_names

print(OBJECTS)

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if DEVICE == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)
# convert the mask shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

assert PROMPT_TYPE_FOR_VIDEO in [
    "point",
    "box",
    "mask",
], "SAM 2 video predictor only support point/box/mask prompt"

if PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes)):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
else:
    raise NotImplementedError(
        "SAM 2 video predictor only support point/box/mask prompts"
    )

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
for (
    out_frame_idx,
    out_obj_ids,
    out_mask_logits,
) in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

"""
Step 5: Visualize the segment results across the video and save them
"""

existDir(f"{output_path}/mask/")
existDir(f"{output_path}/mask/{camera_idx}")

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS)}

# Save the id_to_objects into json
with open(f"{output_path}/mask/mask_info_{camera_idx}.json", "w") as f:
    json.dump(ID_TO_OBJECTS, f)

for frame_idx, masks in video_segments.items():
    for obj_id, mask in masks.items():
        existDir(f"{output_path}/mask/{camera_idx}/{obj_id}")
        # mask is 1 * H * W
        Image.fromarray((mask[0] * 255).astype(np.uint8)).save(
            f"{output_path}/mask/{camera_idx}/{obj_id}/{frame_idx}.png"
        )
