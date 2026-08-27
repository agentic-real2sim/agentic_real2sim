"""Stage 1: CMA-ES optimization of sparse physics parameters."""
import dataclasses
import importlib.resources
import json
import random
import sys
import numpy as np
import pickle
import torch
from argparse import ArgumentParser

from ar2s.phystwin_sysid import PhysTwinConfig, OptimizerCMA
from ar2s.phystwin_sysid._logger import logger, StreamToLogger, logging


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(42)

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

_CONFIGS = importlib.resources.files("ar2s.phystwin_sysid") / "configs"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--config_path", type=str, default=None,
        help="Path to YAML config. Defaults to bundled cloth.yaml or real.yaml.",
    )
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame
    max_iter = args.max_iter

    if args.config_path is not None:
        config_path = args.config_path
    elif "cloth" in case_name or "package" in case_name:
        config_path = str(_CONFIGS / "cloth.yaml")
    else:
        config_path = str(_CONFIGS / "real.yaml")

    cfg = PhysTwinConfig.from_yaml(config_path)

    # Set camera / visualization params
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)

    cfg = dataclasses.replace(
        cfg,
        c2ws=np.array(c2ws),
        w2cs=np.array(w2cs),
        intrinsics=np.array(data["intrinsics"]),
        WH=data["WH"],
        overlay_path=f"{base_path}/{case_name}/color",
    )

    base_dir = f"experiments_optimization/{case_name}"
    logger.set_log_file(path=base_dir, name="optimize_cma_log")

    optimizer = OptimizerCMA(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
        cfg=cfg,
        device=args.device,
    )
    optimizer.optimize(max_iter=max_iter)
