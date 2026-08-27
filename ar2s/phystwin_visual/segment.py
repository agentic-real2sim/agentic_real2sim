# Process to get the masks of the controller and the object
import glob
import shutil
import subprocess
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--TEXT_PROMPT", type=str, required=True)
parser.add_argument("--box_threshold", type=float, default=0.35)
parser.add_argument("--text_threshold", type=float, default=0.25)
parser.add_argument("--camera_indices", type=int, nargs="*", default=None)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT
camera_indices = args.camera_indices
if camera_indices is None:
    camera_indices = sorted(
        int(path.rsplit("/", 1)[-1])
        for path in glob.glob(f"{base_path}/{case_name}/depth/*")
    )
camera_num = len(camera_indices)
assert camera_num > 0
print(f"Processing {case_name}")

for camera_idx in camera_indices:
    print(f"Processing {case_name} camera {camera_idx}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ar2s.phystwin_visual.segment_util_video",
            "--base_path",
            base_path,
            "--case_name",
            case_name,
            "--TEXT_PROMPT",
            TEXT_PROMPT,
            "--camera_idx",
            str(camera_idx),
            "--box_threshold",
            str(args.box_threshold),
            "--text_threshold",
            str(args.text_threshold),
        ],
        check=True,
    )
    shutil.rmtree(f"{base_path}/{case_name}/tmp_data", ignore_errors=True)
