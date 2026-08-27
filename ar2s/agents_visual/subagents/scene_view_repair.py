"""scene_view_repair — two-view detect / rebuild / per-object swap flow.

User-designed logic (2026-07-06):

1. The pipeline runs on the PRIMARY view all the way through z-grounding
   (geometry_prior) as usual.
2. Two problem detectors:
   - P1 (deterministic): objects in ``state.discovered_objects`` that never
     made it into ``state.resolved_objects`` — SAM3 found no usable mask.
   - P2 (VLM): show [real@view1, sim@view1, real@view2, sim@view2] where the
     sim images are FULL-scene renders of every resolved object at its final
     pose, plus the object name list. VLM reports objects clearly
     inconsistent with real.
3. If any problem: the caller rebuilds the ENTIRE scene from view 2 (a
   swapped-primary stub over the same role-keyed raw_episodes — content
   never moves, only the attribute flips) into a sibling run_root.
4. Per problem object, the VLM compares 6 images
   (real / view1-candidate / view2-candidate, rendered at BOTH cameras)
   and picks the better candidate.
5. If view2 wins: ``swap_object`` replaces mesh+scale+pose(+collision) in the
   main bundle — FP per-frame poses are REWRITTEN into the main primary
   camera frame (RobotSceneSim lifts all objects through one scene-global
   camera) — then ``reground_z`` re-anchors the object's z onto its
   support-tree root.

VLM routing: ``visual.subagents.scene_mismatch_critic`` and
``visual.subagents.object_view_compare`` in the agent config YAML.
"""
from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, Field

from ar2s.agents_visual._models import vlm_call_single
from ar2s.agents_visual.raw_episode import resolve_camera_files
from ar2s.pipeline_artifacts import episode_bundle_dir
from ar2s.droid_sim.pose_store import earliest_frame, load_pose, load_poses, save_poses
from ar2s.agents_visual.subagents._pose_view_render import (
    render_scene_objects,
)

_JPEG_QUALITY = 90
_MISMATCH_PROMPT = Path(__file__).with_name("scene_mismatch_prompt.md")
_COMPARE_PROMPT = Path(__file__).with_name("object_view_compare_prompt.md")
_V2_SUFFIX = "__v2"


# ---------------------------------------------------------------------------
# Stage entry (wired into ar2s.run_pipeline between geometry_prior and
# physical_prior)
# ---------------------------------------------------------------------------

def _swapped_visual_input(vi):
    """VisualInput with primary and secondary camera attributes exchanged.

    Role-keyed staging means this touches ONLY attributes — the underlying
    ext1/ext2 files are shared with the main run verbatim.
    """
    import dataclasses
    assert vi.secondary_stereo_stream_path, (
        "scene_view_repair: no secondary camera available for this episode"
    )
    return dataclasses.replace(
        vi,
        stereo_stream_path=vi.secondary_stereo_stream_path,
        stereo_intrinsics_path=vi.secondary_stereo_intrinsics_path,
        secondary_stereo_stream_path=vi.stereo_stream_path,
        secondary_stereo_intrinsics_path=vi.stereo_intrinsics_path,
        primary_camera_id=vi.secondary_camera_id,
        secondary_camera_id=vi.primary_camera_id,
        episode_id=vi.episode_id + _V2_SUFFIX,
        scene_name=(vi.scene_name or vi.episode_id) + _V2_SUFFIX,
    )


def run_repair(
    run_root: Path,
    episode_dir: Path,
    visual_input,
    *,
    visual_max_iterations: int = 50,
) -> dict:
    """Full repair flow. Returns a JSON-serializable report (also written to
    ``<run_root>/scene_view_repair/report.json`` along with every image the
    VLMs saw, so humans can audit the judgments).
    """
    run_root = Path(run_root)
    repair_dir = run_root / "scene_view_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(visual_input.stereo_stream_path).parent

    ctx_main = load_scene_ctx(run_root, raw_dir)
    report: dict = {"problems": [], "repaired": [], "unrepairable": [],
                    "v2_built": False, "v2_reused": False}

    # ---- detect ----
    missing = detect_missing_objects(ctx_main["state"])
    mm = detect_scene_mismatch(ctx_main, save_debug_dir=repair_dir / "detect")
    mismatch_names: list[str] = []
    if mm is not None and not mm.all_consistent:
        known = {o["name"] for o in ctx_main["objects"]}
        mismatch_names = [m.name for m in mm.mismatched_objects if m.name in known]
        report["mismatch_issues"] = {
            m.name: m.issue for m in mm.mismatched_objects
        }
    problems = list(dict.fromkeys(missing + mismatch_names))   # ordered dedup
    report["problems"] = problems
    report["missing_objects"] = missing
    if not problems:
        (repair_dir / "report.json").write_text(json.dumps(report, indent=2))
        print("[scene_view_repair] no problems detected — scene accepted as-is")
        return report

    # ---- build (or reuse) the view-2 scene ----
    v2_root = run_root.parent / (run_root.name + _V2_SUFFIX)
    v2_ep_dir = episode_bundle_dir(v2_root)
    if (v2_ep_dir / "manifest.yaml").is_file():
        report["v2_reused"] = True
        print(f"[scene_view_repair] reusing existing view-2 scene: {v2_root}")
    else:
        # The v2 scene is a REPAIR RESOURCE, never the deliverable — a failure
        # here (e.g. the pickup object not surviving the secondary view's
        # visual stages, seen on RPL_12_09) must degrade to "no v2 available",
        # not kill an otherwise-healthy main scene.
        try:
            from ar2s.agents_visual.orchestrator import data_ingest, finalize, run_visual
            from ar2s.agents_geometry_prior.apply import apply_geometry_priors

            v2_input = _swapped_visual_input(visual_input)
            print(f"[scene_view_repair] building view-2 scene at {v2_root} ...")
            data_ingest(v2_input, run_root=str(v2_root))
            visual_result = run_visual(v2_root, max_iterations=visual_max_iterations)
            msgs = visual_result.get("messages") or []
            final_text = ""
            if msgs:
                content = getattr(msgs[-1], "content", "") or ""
                final_text = content if isinstance(content, str) else str(content)
            if not final_text.startswith("Pipeline complete"):
                raise RuntimeError(
                    f"view-2 visual agent did not complete: {final_text[:200]}"
                )
            v2_ep_dir = Path(finalize(str(v2_root)))
            apply_geometry_priors(episode_dir=v2_ep_dir, run_root=str(v2_root))
            report["v2_built"] = True
        except Exception as exc:  # noqa: BLE001 — degrade, don't propagate
            report["v2_build_error"] = f"{type(exc).__name__}: {exc}"
            report["unrepairable"] = list(dict.fromkeys(
                report["unrepairable"] + problems
            ))
            (repair_dir / "report.json").write_text(json.dumps(report, indent=2))
            print(f"[scene_view_repair] view-2 build FAILED — problems "
                  f"{problems} recorded as unrepairable, main scene kept: {exc}")
            return report

    ctx_alt = load_scene_ctx(v2_root, raw_dir)

    # ---- per-object compare / swap / adopt, parents before children ----
    relations = (ctx_main["state"].get("support_tree") or {}).get("relations") or []
    root_of = {r.get("object"): r.get("supported_by") for r in relations}
    problems.sort(key=lambda n: 0 if root_of.get(n, "GROUND") == "GROUND" else 1)

    for name in problems:
        main_names = {o["name"] for o in ctx_main["objects"]}
        if name not in main_names:
            if adopt_object(ctx_main, ctx_alt, name):
                ctx_main = load_scene_ctx(run_root, raw_dir)
                reground_z(ctx_main, name)
                report["repaired"].append({"name": name, "action": "adopted_view2"})
            else:
                report["unrepairable"].append(name)
            continue

        choice = compare_object_views(
            ctx_main, ctx_alt, name,
            save_debug_dir=repair_dir / f"compare_{name}",
        )
        if choice is None:
            report["unrepairable"].append(name)
            continue
        entry = {"name": name, "choice": f"view{choice.choice}",
                 "reason": choice.reason}
        if choice.choice == 2:
            swap_object(ctx_main, ctx_alt, name)
            ctx_main = load_scene_ctx(run_root, raw_dir)
            dz = reground_z(ctx_main, name)
            ctx_main = load_scene_ctx(run_root, raw_dir)
            entry["action"] = "swapped_to_view2"
            entry["reground_dz_mm"] = round(dz * 1000, 1)
        else:
            entry["action"] = "kept_view1"
        report["repaired"].append(entry)

    # ---- post-repair renders for human audit ----
    ctx_final = load_scene_ctx(run_root, raw_dir)
    for serial, tag in ((ctx_final["primary_serial"], "primary"),
                        (ctx_final["secondary_serial"], "secondary")):
        if serial and serial in ctx_final["K"]:
            Image.fromarray(render_scene(ctx_final, serial)).save(
                repair_dir / f"after_sim_{tag}.png"
            )

    (repair_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[scene_view_repair] report: {json.dumps(report)[:400]}")
    return report


# ---------------------------------------------------------------------------
# Scene context loading + rendering
# ---------------------------------------------------------------------------

def load_scene_ctx(run_root: Path, raw_dir: Path | None = None) -> dict:
    """Load everything needed to render the emitted bundle from any camera.

    Returns {agent_dir, raw_dir, state, manifest, primary_serial,
    secondary_serial, c2w: {serial: (4,4)}, K: {serial: (3,3)},
    ground_z, objects: [{name, mesh_path, mesh_scale, T_object_in_world}],
    real_frames: {serial: Path}}.
    """
    run_root = Path(run_root)
    state = json.loads((run_root / "state.json").read_text())
    agent_dir = episode_bundle_dir(run_root)
    manifest = yaml.safe_load((agent_dir / "manifest.yaml").read_text())

    if raw_dir is None:
        entry_path = state.get("entry", {}).get("cameras_extrinsics_path", "")
        raw_dir = run_root / "raw_episodes"
        if not raw_dir.is_dir():
            raw_dir = Path(entry_path).parent
    raw_dir = Path(raw_dir)
    assert raw_dir.is_dir(), f"raw_episodes not found: {raw_dir}"

    primary_serial = str(manifest["cameras"]["primary_id"])
    cam_ids = [str(c) for c in manifest["cameras"]["cam_ids"]]
    secondary_serial = next((c for c in cam_ids if c != primary_serial), "")

    extr = np.load(raw_dir / "cameras_extrinsics.npz")
    c2w = {}
    for s in cam_ids:
        key = f"cam_mat_{s}"
        assert key in extr.files, (raw_dir, key)
        c2w[s] = np.linalg.inv(np.asarray(extr[key], dtype=np.float64))

    cams = resolve_camera_files(raw_dir)
    K = {}
    real_mp4 = {}
    for entry in (cams["primary"], cams.get("secondary")):
        if entry is None:
            continue
        s = entry["serial"]
        if s in cam_ids and entry["K"].is_file():
            K[s] = np.loadtxt(entry["K"], max_rows=1).reshape(3, 3)
            real_mp4[s] = entry["mp4"]

    offset = float((manifest.get("ground") or {}).get("offset") or 0.0)
    ground_z = -offset if offset else None

    c2w_pri = c2w[primary_serial]
    objects = []
    for o in manifest.get("objects", []):
        obj_dir = agent_dir / Path(o["visual"]).parent
        # Per-object keyframes mean the first FP frame is not always 0 (e.g. an
        # object SAM3 could only segment from frame 2). Hardcoding frame_000000
        # silently dropped those objects from the scene context, so the P2
        # mismatch detector never saw them.
        pose_dir = agent_dir / o["foundation_pose_dir"]
        first_frame = earliest_frame(pose_dir)
        if first_frame is None:
            continue
        T_cam = load_pose(pose_dir, first_frame)
        assert T_cam is not None, pose_dir
        T_w = c2w_pri @ T_cam
        po = o.get("pos_offset")
        if po:
            T_w[:3, 3] += np.asarray(po, dtype=np.float64)
        scale = np.load(agent_dir / o["scale"]).astype(np.float64).reshape(-1) \
            if (agent_dir / o["scale"]).is_file() else np.ones(3)
        objects.append({
            "name": o["name"],
            "mesh_path": agent_dir / o["visual"],
            "mesh_scale": scale,
            "T_object_in_world": T_w,
        })

    return {
        "run_root": run_root, "agent_dir": agent_dir, "raw_dir": raw_dir,
        "state": state, "manifest": manifest,
        "primary_serial": primary_serial, "secondary_serial": secondary_serial,
        "c2w": c2w, "K": K, "ground_z": ground_z,
        "objects": objects, "real_mp4": real_mp4,
    }


def real_frame(ctx: dict, serial: str, frame_idx: int = 0) -> np.ndarray:
    """Frame ``frame_idx`` of a camera's real video (left half of stereo SBS)."""
    mp4 = ctx["real_mp4"][serial]
    cache = ctx["raw_dir"] / f"_extracted_{serial}_frame_{frame_idx:06d}.png"
    if not cache.is_file():
        import subprocess
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(mp4),
             "-vf", f"select=eq(n\\,{frame_idx}),crop=iw/2:ih:0:0",
             "-vframes", "1", str(cache)],
            check=True,
        )
    return np.array(Image.open(cache).convert("RGB"))


def render_scene(ctx: dict, serial: str, objects: list[dict] | None = None) -> np.ndarray:
    """Full-scene render (all bundle objects) from camera ``serial``."""
    return render_scene_objects(
        objects if objects is not None else ctx["objects"],
        ctx["c2w"][serial], ctx["K"][serial],
        ground_z=ctx["ground_z"],
    )


def _url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


# ---------------------------------------------------------------------------
# P1: deterministic missing-object detection
# ---------------------------------------------------------------------------

def detect_missing_objects(state: dict) -> list[str]:
    """Discovered (non-robot) objects that never resolved to a mask/mesh."""
    discovered = [n for n in (state.get("discovered_objects") or []) if n != "robot"]
    resolved = {o["name"] for o in (state.get("resolved_objects") or [])}
    return [n for n in discovered if n not in resolved]


# ---------------------------------------------------------------------------
# P2: VLM full-scene mismatch check
# ---------------------------------------------------------------------------

class MismatchedObject(BaseModel):
    name: str = Field(description="Object name from the provided list.")
    issue: str = Field(description="One sentence: what is inconsistent (position / orientation / size / shape).")


class SceneMismatchResult(BaseModel):
    all_consistent: bool = Field(description="True if every listed object plausibly matches the real scene in both views.")
    mismatched_objects: list[MismatchedObject] = Field(description="Objects clearly inconsistent with the real scene. Empty when all_consistent.")
    note: str = Field(description="One-sentence overall summary.")


def detect_scene_mismatch(ctx: dict, save_debug_dir: Path | None = None) -> SceneMismatchResult | None:
    pri, sec = ctx["primary_serial"], ctx["secondary_serial"]
    imgs = {
        "real_primary": real_frame(ctx, pri),
        "sim_primary": render_scene(ctx, pri),
    }
    have_secondary = sec and sec in ctx["K"] and sec in ctx["real_mp4"]
    if have_secondary:
        imgs["real_secondary"] = real_frame(ctx, sec)
        imgs["sim_secondary"] = render_scene(ctx, sec)

    if save_debug_dir is not None:
        save_debug_dir.mkdir(parents=True, exist_ok=True)
        for k, v in imgs.items():
            Image.fromarray(v).save(save_debug_dir / f"{k}.png")

    names = [o["name"] for o in ctx["objects"]]
    order = list(imgs.keys())
    user_text = (
        f"Objects present in the sim renders: {names}.\n"
        + "".join(f"IMAGE {i+1} = {k}.\n" for i, k in enumerate(order))
        + "Judge whether any listed object is clearly inconsistent with the "
          "real scene, using both views. Return the structured verdict."
    )
    r = vlm_call_single(
        "visual.subagents.scene_mismatch_critic",
        system=_MISMATCH_PROMPT.read_text(),
        user_text=user_text,
        images=[_url(imgs[k]) for k in order],
        output_schema=SceneMismatchResult,
    )
    if r is not None:
        print(f"[scene_mismatch] all_consistent={r.all_consistent} "
              f"problems={[(m.name, m.issue) for m in r.mismatched_objects]}")
    return r


# ---------------------------------------------------------------------------
# Per-object view1-vs-view2 comparison
# ---------------------------------------------------------------------------

class ObjectViewChoice(BaseModel):
    # ``reasoning`` intentionally precedes ``choice``: autoregressive
    # generation writes it first, forcing an explicit per-axis comparison
    # before committing to a verdict (chain-of-thought via field order).
    reasoning: str = Field(description=(
        "Step-by-step comparison BEFORE deciding: for each axis in priority "
        "order (shape, size, position, orientation), state what the REAL object looks like in "
        "each view, then how candidate 1 and candidate 2 each match or "
        "mismatch it, citing the specific images."
    ))
    choice: int = Field(description="1 if the VIEW-1 candidate matches the real object better, 2 for the VIEW-2 candidate.")
    reason: str = Field(description="One sentence summarizing the deciding axes.")


def _find_alt_name(ctx_alt: dict, name: str) -> str | None:
    """Match a main-scene object name to the alt scene's naming.

    The two scenes come from INDEPENDENT visual runs, so object_discovery may
    name the same physical object differently per view (observed: v1 'pen'
    vs v2 'green_pen'). Resolution order:
      1. exact match;
      2. unique substring containment either way ('pen' ⊂ 'green_pen');
      3. unique shared last-token ('green_pen' → 'pen').
    Ambiguity (>1 candidate) returns None rather than guessing.
    """
    alt_names = [o["name"] for o in ctx_alt["objects"]]
    if name in alt_names:
        return name
    contains = [a for a in alt_names if name in a or a in name]
    if len(contains) == 1:
        return contains[0]
    tok = name.split("_")[-1]
    tokened = [a for a in alt_names if a.split("_")[-1] == tok]
    if len(tokened) == 1:
        return tokened[0]
    return None


def _alt_object_in_main_world(ctx_alt: dict, name: str) -> dict | None:
    """The alt bundle's object entry (fuzzy-name-matched), in WORLD frame.

    Both bundles share one calibrated world (same cameras_extrinsics), so the
    alt scene's world-frame pose drops straight into main-scene renders.
    """
    alt_name = _find_alt_name(ctx_alt, name)
    if alt_name is None:
        return None
    for o in ctx_alt["objects"]:
        if o["name"] == alt_name:
            return o
    return None


# Saturated magenta: absent from DROID scenes (wood/white/gray) and already
# the repo's mask-overlay convention. Applied to BOTH candidates identically
# so color itself can't bias the choice.
_CANDIDATE_RGBA = (1.0, 0.0, 1.0, 1.0)

_IMG_INTROS = {
    "real_primary": (
        "REAL image from the PRIMARY camera. This is ground truth. Locate "
        "the target object and note its position on the table, orientation, "
        "and apparent size."
    ),
    "cand1_primary": (
        "CANDIDATE 1 (reconstructed from view 1), rendered ALONE from the "
        "same PRIMARY camera as the previous real image. Compare it against "
        "the real object you just located."
    ),
    "cand2_primary": (
        "CANDIDATE 2 (reconstructed from view 2), rendered alone from the "
        "same PRIMARY camera. Compare against the same real object."
    ),
    "real_secondary": (
        "REAL image from the SECONDARY camera — the same scene and instant "
        "from the other calibrated viewpoint. Locate the target object again."
    ),
    "cand1_secondary": (
        "CANDIDATE 1 again, rendered from the SECONDARY camera. A correct "
        "reconstruction must match the real object in BOTH views."
    ),
    "cand2_secondary": (
        "CANDIDATE 2 again, rendered from the SECONDARY camera."
    ),
}


def compare_object_views(
    ctx_main: dict,
    ctx_alt: dict,
    name: str,
    save_debug_dir: Path | None = None,
    *,
    interleaved_intros: bool = True,
    candidate_rgba: tuple | None = _CANDIDATE_RGBA,
) -> ObjectViewChoice | None:
    """6-image VLM compare of one object's two candidates.

    Candidate 1 = main bundle's version, candidate 2 = alt bundle's version;
    both rendered ALONE from both calibrated cameras of the main scene.

    Knobs (defaults reflect the 2026-07-06 experiments):
      - schema carries a mandatory ``reasoning`` field ahead of ``choice``;
      - ``interleaved_intros``: each image arrives with its own explanatory
        text block instead of one index list;
      - ``candidate_rgba``: flat-color override for both candidate renders
        (None = keep mesh textures).
    """
    cand1 = next((o for o in ctx_main["objects"] if o["name"] == name), None)
    cand2 = _alt_object_in_main_world(ctx_alt, name)
    if cand2 is None:
        print(f"[object_view_compare] {name}: alt scene has no such object")
        return None
    if candidate_rgba is not None:
        cand1 = {**cand1, "rgba": candidate_rgba} if cand1 is not None else None
        cand2 = {**cand2, "rgba": candidate_rgba}

    pri, sec = ctx_main["primary_serial"], ctx_main["secondary_serial"]
    imgs: dict[str, np.ndarray] = {}
    for serial, tag in ((pri, "primary"), (sec, "secondary")):
        if not serial or serial not in ctx_main["K"]:
            continue
        imgs[f"real_{tag}"] = real_frame(ctx_main, serial)
        if cand1 is not None:
            imgs[f"cand1_{tag}"] = render_scene(ctx_main, serial, objects=[cand1])
        imgs[f"cand2_{tag}"] = render_scene(ctx_main, serial, objects=[cand2])

    if save_debug_dir is not None:
        save_debug_dir.mkdir(parents=True, exist_ok=True)
        for k, v in imgs.items():
            Image.fromarray(v).save(save_debug_dir / f"{k}.png")

    # keep view-grouped order: real → cand1 → cand2 per view
    order = [k for k in _IMG_INTROS if k in imgs]
    color_note = (
        "Both candidates are deliberately rendered in flat magenta so you "
        "judge silhouette, position, and size — not texture.\n"
        if candidate_rgba is not None else ""
    )
    if interleaved_intros:
        images = [
            {"label": f"IMAGE {i+1} — {_IMG_INTROS[k]}", "url": _url(imgs[k])}
            for i, k in enumerate(order)
        ]
        user_text = (
            f"Target object: {name!r}.\n" + color_note +
            "You have now seen all six images. Reason step by step per the "
            "schema, then choose which candidate matches the REAL object "
            "better — judge shape and size first, then position, then orientation."
        )
    else:
        images = [_url(imgs[k]) for k in order]
        user_text = (
            f"Target object: {name!r}.\n" + color_note +
            "".join(f"IMAGE {i+1} = {k}.\n" for i, k in enumerate(order))
            + "cand1 = candidate reconstructed from view 1 (the scene's "
              "primary camera); cand2 = candidate reconstructed from view 2. "
              "Both are rendered alone from the same calibrated cameras as "
              "the real frames. Judge shape and size first, then position, then "
              "orientation. Reason step by step per the schema, then choose."
        )
    r = vlm_call_single(
        "visual.subagents.object_view_compare",
        system=_COMPARE_PROMPT.read_text(),
        user_text=user_text,
        images=images,
        output_schema=ObjectViewChoice,
    )
    if r is not None:
        print(f"[object_view_compare] {name}: choice=view{r.choice} — {r.reason}")
    return r


# ---------------------------------------------------------------------------
# Swap + reground
# ---------------------------------------------------------------------------

def swap_object(ctx_main: dict, ctx_alt: dict, name: str) -> None:
    """Replace ``name``'s assets in the main bundle with the alt bundle's.

    Copies visual.obj / scale.npy / foundation_pose/ / collision/ (if any),
    rewriting every FP per-frame pose from the ALT primary camera frame into
    the MAIN primary camera frame (scene-global lift in RobotSceneSim).
    Originals are backed up to ``objects/<name>_view1_backup/``.
    """
    alt_name = _find_alt_name(ctx_alt, name) or name
    main_dir = ctx_main["agent_dir"] / "objects" / name
    alt_dir = ctx_alt["agent_dir"] / "objects" / alt_name
    assert alt_dir.is_dir(), alt_dir

    backup = ctx_main["agent_dir"] / "objects" / f"{name}_view1_backup"
    if not backup.exists():
        shutil.copytree(main_dir, backup)

    for item in ("visual.obj", "scale.npy"):
        src = alt_dir / item
        if src.is_file():
            shutil.copy2(src, main_dir / item)
    if (alt_dir / "collision").is_dir():
        shutil.rmtree(main_dir / "collision", ignore_errors=True)
        shutil.copytree(alt_dir / "collision", main_dir / "collision")
    else:
        # Alt bundle stopped before scene_prep (no coacd output). The main
        # bundle's old collision decomposition belongs to the REPLACED mesh —
        # keeping it would silently pair new visual with stale collision.
        # Delete the dir AND drop the manifest key: the strict episode loader
        # validates collision_dir paths when present, while collision_prep
        # regenerates + re-adds the key when absent.
        shutil.rmtree(main_dir / "collision", ignore_errors=True)
        manifest_path = ctx_main["agent_dir"] / "manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        for o in manifest.get("objects", []):
            if o.get("name") == name:
                o.pop("collision_dir", None)
        manifest_path.write_text(
            yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False)
        )
        print(f"[swap_object] {name}: stale collision/ + manifest collision_dir "
              f"removed — scene_prep will regenerate")

    # Rewrite FP poses: alt-primary cam frame → main-primary cam frame.
    w2c_main = np.linalg.inv(ctx_main["c2w"][ctx_main["primary_serial"]])
    c2w_alt = ctx_alt["c2w"][ctx_alt["primary_serial"]]
    T_bridge = w2c_main @ c2w_alt
    fp_main = main_dir / "foundation_pose"
    fp_alt = alt_dir / "foundation_pose"
    alt_poses = load_poses(fp_alt)
    save_poses(fp_main, {
        frame: (T_bridge @ pose).astype(np.float32)
        for frame, pose in alt_poses.items()
    })
    n = len(alt_poses)
    print(f"[swap_object] {name}: assets swapped, {n} FP poses rewritten "
          f"alt-cam({ctx_alt['primary_serial']}) → main-cam({ctx_main['primary_serial']})")

    # The alt run picked its own per-object keyframe, so the frame numbering we
    # just copied in generally differs from view 1's. Refresh the manifest's
    # init_frame or downstream keeps view 1's stale value.
    new_init = earliest_frame(fp_main)
    assert new_init is not None, f"swap left no FP frames under {fp_main}"
    manifest_path = ctx_main["agent_dir"] / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    for o in manifest.get("objects", []):
        if o.get("name") == name and o.get("init_frame") != new_init:
            print(f"[swap_object] {name}: init_frame {o.get('init_frame')} → {new_init}")
            o["init_frame"] = new_init
    manifest_path.write_text(
        yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False)
    )


def adopt_object(ctx_main: dict, ctx_alt: dict, name: str) -> bool:
    """Adopt an object that exists only in the alt bundle (P1 missing case).

    Copies the whole objects/<name>/ dir, rewrites FP poses into the main
    primary camera frame, and appends the alt manifest's object entry to the
    main manifest. Returns False when the alt bundle doesn't have it either.
    """
    alt_name = _find_alt_name(ctx_alt, name)
    alt_manifest = ctx_alt["manifest"]
    alt_entry = next(
        (o for o in alt_manifest.get("objects", []) if o["name"] == alt_name),
        None,
    ) if alt_name else None
    alt_dir = ctx_alt["agent_dir"] / "objects" / alt_name if alt_name else None
    if alt_entry is None or alt_dir is None or not alt_dir.is_dir():
        print(f"[adopt_object] {name}: not present in alt bundle either — unrepairable")
        return False

    main_dir = ctx_main["agent_dir"] / "objects" / name
    shutil.rmtree(main_dir, ignore_errors=True)
    shutil.copytree(alt_dir, main_dir)
    # Keep MAIN's name: rewrite the adopted manifest entry's name + paths.
    alt_entry = dict(alt_entry)
    alt_entry["name"] = name
    for key in ("visual", "scale", "foundation_pose_dir", "collision_dir"):
        if alt_entry.get(key):
            alt_entry[key] = alt_entry[key].replace(f"objects/{alt_name}/", f"objects/{name}/")
    # Same collision_dir contract as swap_object: drop the key when the
    # copied dir doesn't actually exist so scene_prep regenerates.
    if alt_entry.get("collision_dir") and not (main_dir / "collision").is_dir():
        alt_entry.pop("collision_dir", None)

    w2c_main = np.linalg.inv(ctx_main["c2w"][ctx_main["primary_serial"]])
    c2w_alt = ctx_alt["c2w"][ctx_alt["primary_serial"]]
    T_bridge = w2c_main @ c2w_alt
    fp_root = main_dir / "foundation_pose"
    save_poses(fp_root, {
        frame: (T_bridge @ pose).astype(np.float32)
        for frame, pose in load_poses(fp_root).items()
    })

    manifest_path = ctx_main["agent_dir"] / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["objects"].append(alt_entry)
    manifest_path.write_text(yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False))
    print(f"[adopt_object] {name}: adopted from alt bundle")
    return True


def reground_z(ctx_main: dict, name: str) -> float:
    """Adjust manifest pos_offset.z so ``name`` rests exactly on its
    support-tree root (GROUND plane, or another object's world top).

    Returns the applied delta-z in meters. Recomputes from the CURRENT
    on-disk assets (call after swap_object).
    """
    import trimesh

    agent_dir = ctx_main["agent_dir"]
    manifest_path = agent_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())

    entry = next(o for o in manifest["objects"] if o["name"] == name)
    obj_dir = agent_dir / Path(entry["visual"]).parent

    # world pose from the current on-disk FP first frame + current pos_offset.
    # Not necessarily frame 0: per-object keyframes start later, and after a
    # swap this dir holds the alt run's frame numbering.
    c2w_pri = ctx_main["c2w"][str(manifest["cameras"]["primary_id"])]
    fp_dir = obj_dir / "foundation_pose"
    first_frame = earliest_frame(fp_dir)
    assert first_frame is not None, f"no FoundationPose frames under {fp_dir}"
    T_cam = load_pose(fp_dir, first_frame)
    assert T_cam is not None, fp_dir
    T_w = c2w_pri @ T_cam
    po = np.asarray(entry.get("pos_offset") or [0.0, 0.0, 0.0], dtype=np.float64)

    mesh = trimesh.load(str(obj_dir / "visual.obj"), force="mesh")
    scale = np.load(obj_dir / "scale.npy").astype(np.float64).reshape(-1)
    V = np.asarray(mesh.vertices, dtype=np.float64) * (scale if scale.size == 3 else float(scale[0]))
    Vw = (T_w[:3, :3] @ V.T).T + T_w[:3, 3] + po
    bottom_z = float(Vw[:, 2].min())

    # target: support-tree root's top
    relations = (ctx_main["state"].get("support_tree") or {}).get("relations") or []
    root = next((r.get("supported_by") for r in relations if r.get("object") == name), "GROUND")
    if root and root != "GROUND":
        root_obj = next((o for o in ctx_main["objects"] if o["name"] == root), None)
        if root_obj is not None:
            rm = trimesh.load(str(root_obj["mesh_path"]), force="mesh")
            rs = np.asarray(root_obj["mesh_scale"], dtype=np.float64)
            RV = np.asarray(rm.vertices, dtype=np.float64) * (rs if rs.size == 3 else float(rs[0]))
            RT = root_obj["T_object_in_world"]
            target_z = float(((RT[:3, :3] @ RV.T).T + RT[:3, 3])[:, 2].max())
        else:
            target_z = float(ctx_main["ground_z"] or 0.0)
    else:
        target_z = float(ctx_main["ground_z"] or 0.0)

    dz = target_z - bottom_z
    new_po = po.copy()
    new_po[2] += dz
    entry["pos_offset"] = [float(x) for x in new_po]
    manifest_path.write_text(yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False))
    print(f"[reground_z] {name}: root={root} bottom_z={bottom_z:.4f} "
          f"target_z={target_z:.4f} Δz={dz*1000:.1f}mm → pos_offset={entry['pos_offset']}")
    return dz
