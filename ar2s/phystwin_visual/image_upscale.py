from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
from argparse import ArgumentParser
import cv2
import numpy as np

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--mask_path", type=str, default=None)
parser.add_argument("--output_path", type=str)
parser.add_argument("--category", type=str)
args = parser.parse_args()

img_path = args.img_path
mask_path = args.mask_path
output_path = args.output_path
category = args.category


# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# let's download an  image
low_res_img = Image.open(img_path).convert("RGB")
if mask_path is not None:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    bbox = np.argwhere(mask > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    low_res_img = low_res_img.crop(bbox)  # type: ignore

prompt = f"Hand manipulates a {category}."

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save(output_path)
