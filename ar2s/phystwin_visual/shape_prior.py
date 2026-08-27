import os
import sys
from pathlib import Path

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

TRELLIS_DIR = Path(__file__).resolve().parents[2] / "third_party" / "TRELLIS_phystwin"
if TRELLIS_DIR.exists():
    sys.path.insert(0, str(TRELLIS_DIR))

import imageio
from PIL import Image
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

img_path = args.img_path
output_dir = args.output_dir

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

final_im = Image.open(img_path).convert("RGBA")
assert not np.all(np.array(final_im)[:, :, 3] == 255)

# Run the pipeline
outputs = pipeline.run(
    final_im,
)

video_gs = render_utils.render_video(outputs["gaussian"][0])["color"]
video_mesh = render_utils.render_video(outputs["mesh"][0])["normal"]
video = [
    np.concatenate([frame_gs, frame_mesh], axis=1)
    for frame_gs, frame_mesh in zip(video_gs, video_mesh)
]
imageio.mimsave(f"{output_dir}/visualization.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs["gaussian"][0],
    outputs["mesh"][0],
    # Optional parameters
    simplify=0.95,  # Ratio of triangles to remove in the simplification process
    texture_size=1024,  # Size of the texture used for the GLB
)
glb.export(f"{output_dir}/object.glb")

# Save Gaussians as PLY files
outputs["gaussian"][0].save_ply(f"{output_dir}/object.ply")
