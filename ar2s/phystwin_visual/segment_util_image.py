import cv2
import os
import sys
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for rel_path in (
    "third_party/Grounded-SAM-2_phystwin",
):
    third_party_path = PROJECT_ROOT / rel_path
    if third_party_path.exists():
        sys.path.insert(0, str(third_party_path))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from argparse import ArgumentParser

"""
Hyper parameters
"""

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--output_path", type=str)
parser.add_argument("--TEXT_PROMPT", type=str)  # must be lowercase and end with period, e.g. "sloth."
parser.add_argument("--box_threshold", type=float, default=0.35)
parser.add_argument("--text_threshold", type=float, default=0.25)
args = parser.parse_args()

img_path = args.img_path
output_path = args.output_path
TEXT_PROMPT = args.TEXT_PROMPT

CHECKPOINT_DIR = Path(os.environ.get("PHYSTWIN_MODELS_ROOT", str(PROJECT_ROOT / "models"))) / "grounded_sam_2"
SAM2_CHECKPOINT = str(CHECKPOINT_DIR / "sam2.1_hiera_large.pt")
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
BOX_THRESHOLD = args.box_threshold
TEXT_THRESHOLD = args.text_threshold
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build groundingdino from huggingface (same model segment_util_video.py uses)
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT.lower().strip()
if not text.endswith("."):
    text += "."

image_pil = Image.open(img_path).convert("RGB")
image_source = np.asarray(image_pil)

sam2_predictor.set_image(image_source)

inputs = processor(images=image_pil, text=text, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = grounding_model(**inputs)
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    target_sizes=[image_pil.size[::-1]],
)[0]

input_boxes = results["boxes"].cpu().numpy()  # already xyxy in pixels
if len(input_boxes) == 0:
    print(f"No objects detected for prompt {text!r}", file=sys.stderr)
    sys.exit(1)

h, w, _ = image_source.shape

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if DEVICE == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

print(f"Detected {len(masks)} objects")

raw_img = cv2.imread(img_path)
mask_img = (masks[0] * 255).astype(np.uint8)

ref_img = np.zeros((h, w, 4), dtype=np.uint8)
mask_bool = mask_img > 0
ref_img[mask_bool, :3] = raw_img[mask_bool]
ref_img[:, :, 3] = mask_bool.astype(np.uint8) * 255
cv2.imwrite(output_path, ref_img)
