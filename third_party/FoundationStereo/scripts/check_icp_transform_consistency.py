#!/usr/bin/env python3
"""
Check consistency of ICP transforms across a frame window.

By default this script inspects:
  output_droid_25947356/icp_results/mug/frame_xxxxxx_transform.npy

It reports:
  1) Per-frame transform stats (scale, translation, optional fitness/rmse)
  2) Adjacent-frame pose deltas (translation / rotation / matrix Frobenius)
  3) Relative-to-first-frame pose deltas
  4) Outliers based on configurable thresholds

Notes:
  - In this project, transform.npy contains scale in the top-left 3x3 block.
  - Rotation comparisons are computed after removing isotropic scale and
    projecting to the nearest SO(3).
"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FrameRecord:
    frame_idx: int
    frame_name: str
    transform_path: Optional[str]
    meta_path: Optional[str]
    has_transform: bool
    has_meta: bool
    scale: float
    t: np.ndarray
    R_so3: np.ndarray
    T: np.ndarray
    fitness: Optional[float]
    rmse: Optional[float]


def frame_name(idx: int) -> str:
    return f"frame_{idx:06d}"


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """Return nearest proper rotation matrix (det=+1)."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def decompose_transform(T: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Decompose transform into isotropic scale, translation, and SO(3) rotation.
    Assumes T[:3, :3] ~= s * R.
    """
    A = T[:3, :3]
    detA = np.linalg.det(A)
    if detA <= 0:
        scale = float("nan")
        R_so3 = project_to_so3(A)
    else:
        scale = float(np.cbrt(detA))
        R_so3 = project_to_so3(A / scale)
    t = T[:3, 3].astype(np.float64)
    return scale, t, R_so3


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def load_frame_record(obj_dir: str, idx: int) -> FrameRecord:
    fn = frame_name(idx)
    t_path = os.path.join(obj_dir, f"{fn}_transform.npy")
    m_path = os.path.join(obj_dir, f"{fn}_meta.npz")

    has_t = os.path.exists(t_path)
    has_m = os.path.exists(m_path)

    T = np.full((4, 4), np.nan, dtype=np.float64)
    scale = float("nan")
    t = np.full((3,), np.nan, dtype=np.float64)
    R_so3 = np.eye(3, dtype=np.float64)
    fitness = None
    rmse = None

    if has_t:
        T = np.load(t_path).astype(np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"{t_path} is not 4x4, got {T.shape}")
        scale, t, R_so3 = decompose_transform(T)

    if has_m:
        meta = np.load(m_path, allow_pickle=True)
        if "fitness" in meta:
            fitness = float(meta["fitness"])
        if "inlier_rmse" in meta:
            rmse = float(meta["inlier_rmse"])

    return FrameRecord(
        frame_idx=idx,
        frame_name=fn,
        transform_path=t_path if has_t else None,
        meta_path=m_path if has_m else None,
        has_transform=has_t,
        has_meta=has_m,
        scale=scale,
        t=t,
        R_so3=R_so3,
        T=T,
        fitness=fitness,
        rmse=rmse,
    )


def summarize(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p95": float("nan"),
        }
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
    }


def write_frame_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "frame_idx",
        "frame_name",
        "has_transform",
        "has_meta",
        "scale",
        "tx",
        "ty",
        "tz",
        "fitness",
        "rmse",
        "ref_dt_m",
        "ref_drot_deg",
        "ref_dmat_fro",
        "flag_ref_trans",
        "flag_ref_rot",
        "flag_low_fitness",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pair_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "from_idx",
        "to_idx",
        "from_name",
        "to_name",
        "adj_dt_m",
        "adj_drot_deg",
        "adj_dmat_fro",
        "from_fitness",
        "to_fitness",
        "flag_adj_trans",
        "flag_adj_rot",
        "flag_low_fitness",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ICP transform consistency")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="output_droid_25947356",
        help="Data directory containing icp_results (default: output_droid_25947356)",
    )
    parser.add_argument(
        "--icp_dir",
        type=str,
        default=None,
        help="ICP directory (default: <data_dir>/icp_results)",
    )
    parser.add_argument(
        "--object",
        type=str,
        default="mug",
        help="Object name under icp_results (default: mug)",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--count", type=int, default=60, help="Number of frames to inspect")
    parser.add_argument(
        "--adj_trans_thresh",
        type=float,
        default=0.01,
        help="Adjacent translation threshold in meters (default: 0.01)",
    )
    parser.add_argument(
        "--adj_rot_thresh_deg",
        type=float,
        default=20.0,
        help="Adjacent rotation threshold in degrees (default: 20.0)",
    )
    parser.add_argument(
        "--ref_trans_thresh",
        type=float,
        default=0.01,
        help="Relative-to-first translation threshold in meters (default: 0.01)",
    )
    parser.add_argument(
        "--ref_rot_thresh_deg",
        type=float,
        default=30.0,
        help="Relative-to-first rotation threshold in degrees (default: 30.0)",
    )
    parser.add_argument(
        "--fitness_thresh",
        type=float,
        default=0.1,
        help="Fitness threshold for warning (default: 0.1)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of largest adjacent jumps to print (default: 10)",
    )
    parser.add_argument(
        "--csv_prefix",
        type=str,
        default=None,
        help="If set, write '<prefix>_frames.csv' and '<prefix>_pairs.csv'",
    )
    args = parser.parse_args()

    icp_dir = args.icp_dir or os.path.join(args.data_dir, "icp_results")
    obj_dir = os.path.join(icp_dir, args.object)
    if not os.path.isdir(obj_dir):
        raise FileNotFoundError(f"Object ICP directory not found: {obj_dir}")

    frame_indices = list(range(args.start, args.start + args.count))
    records = [load_frame_record(obj_dir, i) for i in frame_indices]

    missing_transforms = [r.frame_idx for r in records if not r.has_transform]
    missing_meta = [r.frame_idx for r in records if not r.has_meta]

    valid = [r for r in records if r.has_transform]
    if not valid:
        raise RuntimeError("No valid transform files found in the requested frame window.")

    # Build adjacent stats for contiguous valid frame indices only.
    pair_rows: List[Dict[str, object]] = []
    adj_dt = []
    adj_drot = []
    adj_dmat = []

    for i in range(len(records) - 1):
        a = records[i]
        b = records[i + 1]
        if not (a.has_transform and b.has_transform):
            continue

        dt = float(np.linalg.norm(b.t - a.t))
        drot = rotation_angle_deg(a.R_so3, b.R_so3)
        dmat = float(np.linalg.norm(b.T - a.T, ord="fro"))

        low_fitness = False
        if a.fitness is not None and a.fitness < args.fitness_thresh:
            low_fitness = True
        if b.fitness is not None and b.fitness < args.fitness_thresh:
            low_fitness = True

        row = {
            "from_idx": a.frame_idx,
            "to_idx": b.frame_idx,
            "from_name": a.frame_name,
            "to_name": b.frame_name,
            "adj_dt_m": dt,
            "adj_drot_deg": drot,
            "adj_dmat_fro": dmat,
            "from_fitness": a.fitness,
            "to_fitness": b.fitness,
            "flag_adj_trans": dt > args.adj_trans_thresh,
            "flag_adj_rot": drot > args.adj_rot_thresh_deg,
            "flag_low_fitness": low_fitness,
        }
        pair_rows.append(row)
        adj_dt.append(dt)
        adj_drot.append(drot)
        adj_dmat.append(dmat)

    adj_dt = np.asarray(adj_dt, dtype=np.float64)
    adj_drot = np.asarray(adj_drot, dtype=np.float64)
    adj_dmat = np.asarray(adj_dmat, dtype=np.float64)

    # Relative to first valid frame.
    ref0 = valid[0]
    frame_rows: List[Dict[str, object]] = []
    ref_dt = []
    ref_drot = []
    ref_dmat = []
    scales = []
    fitness_vals = []
    rmse_vals = []

    for r in records:
        if not r.has_transform:
            frame_rows.append(
                {
                    "frame_idx": r.frame_idx,
                    "frame_name": r.frame_name,
                    "has_transform": False,
                    "has_meta": r.has_meta,
                    "scale": np.nan,
                    "tx": np.nan,
                    "ty": np.nan,
                    "tz": np.nan,
                    "fitness": r.fitness,
                    "rmse": r.rmse,
                    "ref_dt_m": np.nan,
                    "ref_drot_deg": np.nan,
                    "ref_dmat_fro": np.nan,
                    "flag_ref_trans": False,
                    "flag_ref_rot": False,
                    "flag_low_fitness": (r.fitness is not None and r.fitness < args.fitness_thresh),
                }
            )
            continue

        dt = float(np.linalg.norm(r.t - ref0.t))
        drot = rotation_angle_deg(ref0.R_so3, r.R_so3)
        dmat = float(np.linalg.norm(r.T - ref0.T, ord="fro"))

        frame_rows.append(
            {
                "frame_idx": r.frame_idx,
                "frame_name": r.frame_name,
                "has_transform": True,
                "has_meta": r.has_meta,
                "scale": r.scale,
                "tx": r.t[0],
                "ty": r.t[1],
                "tz": r.t[2],
                "fitness": r.fitness,
                "rmse": r.rmse,
                "ref_dt_m": dt,
                "ref_drot_deg": drot,
                "ref_dmat_fro": dmat,
                "flag_ref_trans": dt > args.ref_trans_thresh,
                "flag_ref_rot": drot > args.ref_rot_thresh_deg,
                "flag_low_fitness": (r.fitness is not None and r.fitness < args.fitness_thresh),
            }
        )

        ref_dt.append(dt)
        ref_drot.append(drot)
        ref_dmat.append(dmat)
        scales.append(r.scale)
        if r.fitness is not None:
            fitness_vals.append(r.fitness)
        if r.rmse is not None:
            rmse_vals.append(r.rmse)

    ref_dt = np.asarray(ref_dt, dtype=np.float64)
    ref_drot = np.asarray(ref_drot, dtype=np.float64)
    ref_dmat = np.asarray(ref_dmat, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    fitness_vals = np.asarray(fitness_vals, dtype=np.float64)
    rmse_vals = np.asarray(rmse_vals, dtype=np.float64)

    # Print summary
    print(f"[Path] object dir: {obj_dir}")
    print(f"[Window] frames {args.start}..{args.start + args.count - 1} ({args.count} frames)")
    print(f"[Files] missing transforms: {len(missing_transforms)} | missing meta: {len(missing_meta)}")
    if missing_transforms:
        print(f"  missing transform indices: {missing_transforms}")
    if missing_meta:
        print(f"  missing meta indices: {missing_meta}")

    s_scale = summarize(scales)
    print(
        "[Scale] min={min:.6f} max={max:.6f} mean={mean:.6f} std={std:.6f}".format(
            **s_scale
        )
    )

    s_adj_t = summarize(adj_dt)
    s_adj_r = summarize(adj_drot)
    s_adj_m = summarize(adj_dmat)
    print(
        "[Adjacent] dt(m): mean={mean:.6f} p95={p95:.6f} max={max:.6f}".format(
            **s_adj_t
        )
    )
    print(
        "[Adjacent] drot(deg): mean={mean:.3f} p95={p95:.3f} max={max:.3f}".format(
            **s_adj_r
        )
    )
    print(
        "[Adjacent] dmat(Fro): mean={mean:.6f} p95={p95:.6f} max={max:.6f}".format(
            **s_adj_m
        )
    )

    s_ref_t = summarize(ref_dt)
    s_ref_r = summarize(ref_drot)
    s_ref_m = summarize(ref_dmat)
    print(
        "[Ref@first] dt(m): mean={mean:.6f} max={max:.6f}".format(
            **s_ref_t
        )
    )
    print(
        "[Ref@first] drot(deg): mean={mean:.3f} max={max:.3f}".format(
            **s_ref_r
        )
    )
    print(
        "[Ref@first] dmat(Fro): mean={mean:.6f} max={max:.6f}".format(
            **s_ref_m
        )
    )

    if fitness_vals.size > 0:
        s_fit = summarize(fitness_vals)
        print(
            "[Fitness] min={min:.4f} max={max:.4f} mean={mean:.4f}".format(
                **s_fit
            )
        )
    if rmse_vals.size > 0:
        s_rmse = summarize(rmse_vals)
        print(
            "[RMSE] min={min:.6f} max={max:.6f} mean={mean:.6f}".format(
                **s_rmse
            )
        )

    bad_adj_pairs = [
        r for r in pair_rows
        if r["flag_adj_trans"] or r["flag_adj_rot"] or r["flag_low_fitness"]
    ]
    bad_ref_frames = [
        r for r in frame_rows
        if r["has_transform"] and (
            r["flag_ref_trans"] or r["flag_ref_rot"] or r["flag_low_fitness"]
        )
    ]

    print(
        f"[Flags] bad adjacent pairs: {len(bad_adj_pairs)}/{len(pair_rows)} | "
        f"bad frames vs first: {len(bad_ref_frames)}/{len(valid)}"
    )
    print(
        "        thresholds: "
        f"adj_dt>{args.adj_trans_thresh}m, adj_drot>{args.adj_rot_thresh_deg}deg, "
        f"ref_dt>{args.ref_trans_thresh}m, ref_drot>{args.ref_rot_thresh_deg}deg, "
        f"fitness<{args.fitness_thresh}"
    )

    # Print top-K adjacent jumps
    if pair_rows:
        topk = min(args.topk, len(pair_rows))
        print(f"[Top {topk}] adjacent translation jumps")
        order = np.argsort(-adj_dt)[:topk]
        for i in order:
            r = pair_rows[int(i)]
            print(
                f"  {r['from_name']} -> {r['to_name']}: "
                f"dt={r['adj_dt_m']:.6f}m, "
                f"drot={r['adj_drot_deg']:.3f}deg, "
                f"dmat={r['adj_dmat_fro']:.6f}, "
                f"fitness=({r['from_fitness']}, {r['to_fitness']})"
            )

    # Print low-fitness frames for convenience.
    low_fit_frames = [
        r for r in frame_rows
        if r["fitness"] is not None and r["fitness"] < args.fitness_thresh
    ]
    if low_fit_frames:
        print("[Low fitness frames]")
        for r in low_fit_frames:
            print(
                f"  {r['frame_name']}: fitness={r['fitness']:.4f}, rmse={r['rmse']}"
            )

    # Optional CSV outputs
    if args.csv_prefix:
        frames_csv = f"{args.csv_prefix}_frames.csv"
        pairs_csv = f"{args.csv_prefix}_pairs.csv"
        os.makedirs(os.path.dirname(os.path.abspath(frames_csv)), exist_ok=True)
        write_frame_csv(frames_csv, frame_rows)
        write_pair_csv(pairs_csv, pair_rows)
        print(f"[CSV] wrote {frames_csv}")
        print(f"[CSV] wrote {pairs_csv}")


if __name__ == "__main__":
    main()
