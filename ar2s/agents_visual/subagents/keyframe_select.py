"""keyframe_select subagent — per-object VLM picks a prompt frame + SAM3 synonyms.

``select_keyframe_for_object`` is the only public entry point; ``segment`` calls
it once per (object, view, round). One VLM call per invocation, against
``_N_CANDIDATES`` evenly-spaced untried frames, returns BOTH the frame to ground
SAM3 on and 3 alternative text prompts for the same target. Different objects
naturally end up at different prompt frames (e.g. a grasped item at frame 0, a
destination container at any frame).

Two selection criteria, chosen by whether ``reference_frame`` is passed:
  - round 1 (no reference): the EARLIEST candidate in which the target is
    clearly visible — most targets are static early and get occluded later.
  - round 2 (reference = the frame round 1 tried): the candidate MOST UNLIKE
    the reference, since a near-duplicate of a failed frame just fails again.

Synonyms exist because SAM3's text grounding is vocabulary-sensitive; the name
object_discovery produced may simply not be a phrase SAM3 knows. The robot
short-circuits to a fixed ``_ROBOT_SYNONYMS`` list instead of asking the VLM.

Reads from state.json:
  - stages.svo_extract.outputs.left_frames_dir            (view="primary")
  - stages.svo_extract_secondary.outputs.left_frames_dir  (view="secondary")
  - discovered_object_descriptions              (per-object description from object_discovery)
  - pickup_objects                              (used to flag an object's role as "grasped")
  - entry.keyframe_index                        (optional force override)

Writes nothing — the controller owns all state for this stage.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ar2s.agents_visual._models import vlm_call_first_success
from ar2s.agent_configs import image_to_data_url
from ar2s.agents_visual.state import load_state


_PROMPT_PATH = Path(__file__).with_name("keyframe_select_prompt.md")
_N_CANDIDATES = 6                 # broader sweep helps "earliest visible" finding
_ROBOT_NAME = "robot"
# The robot is not a manipulation target so it gets no VLM-derived synonyms, but
# downstream pose_align still needs a reliable robot mask — hence a fixed list.
_ROBOT_SYNONYMS = ["robot arm", "robotic arm", "gripper", "franka arm", "robot manipulator"]


class _Choice(BaseModel):
    frame_index: int = Field(
        description="Chosen frame index — must equal one of the candidate indices in the prompt"
    )
    visible: bool = Field(
        description="True iff the target is clearly visible and recognisable in the chosen frame"
    )
    synonyms: list[str] = Field(
        default_factory=list,
        description=(
            "Exactly 3 alternative SAM3 text prompts for the same target, best first: "
            "the bare base noun when the name carries an adjective, then "
            "<noun + exactly 1 adjective> variants each using a different descriptor kind"
        ),
    )
    reasoning: str = Field(default="", description="One short sentence citing why this frame is the earliest visible")


def _evenly_sample_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return [int(i * (n - 1) / (k - 1)) for i in range(k)]


def _description_for(name: str, rel_by: dict[str, dict]) -> tuple[str, str]:
    """Return (role, description) for an object. Robot gets a sensible default."""
    if name == _ROBOT_NAME:
        return "robot", (
            "The robot arm — a white/black Franka Emika Panda arm with a parallel-jaw "
            "gripper at its end-effector, entering from one side of the workspace."
        )
    ro = rel_by.get(name) or {}
    return ro.get("role") or "related", ro.get("description") or f"the {name} on the workspace"


def _pick_one_object(
    name: str, role: str, description: str,
    candidate_paths: list[Path], candidate_indices: list[int],
    system_prompt: str,
    reference_path: Path | None = None,
) -> tuple[int, list[str]] | None:
    """One VLM call: pick this object's prompt frame and propose 3 SAM3 synonyms.

    Returns (frame_index, synonyms) — the index is guaranteed to be in
    ``candidate_indices`` — or None on failure.

    When ``reference_path`` is given it is prepended to the images as a labelled
    reference and the criterion flips from "earliest visible" to "most unlike
    the reference": the reference is a frame already tried and rejected for this
    object, so a near-duplicate would just fail the same way.
    """
    if reference_path is not None:
        criterion = (
            "The FIRST image is the REFERENCE frame — already tried for this target "
            "and rejected. It is NOT a candidate. From the candidates that follow, "
            "pick the one MOST VISUALLY UNLIKE the reference: different angle on "
            "the target, different occlusion, different position."
        )
    else:
        criterion = (
            "Pick the EARLIEST candidate in which the target is clearly visible "
            "and unoccluded. If multiple early ones look equally good, pick the FIRST."
        )
    user_text = (
        f"Candidate frame indices (in temporal order, shown to you in the same order): "
        f"{candidate_indices}.\n"
        f"Target object: {name} (role: {role}) — {description}\n\n"
        f"{criterion}\n"
        f"Also propose 3 alternative SAM3 text prompts for this same target."
    )
    images = [image_to_data_url(p) for p in candidate_paths]
    if reference_path is not None:
        images.insert(0, image_to_data_url(reference_path))
    r = vlm_call_first_success(
        "visual.subagents.keyframe_select",
        system=system_prompt,
        user_text=user_text,
        images=images,
        output_schema=_Choice,
    )
    if r is None:
        return None
    if r.frame_index not in candidate_indices:
        print(f"[keyframe_select] {name!r}: VLM returned unknown index {r.frame_index}, dropping")
        return None
    synonyms = _clean_synonyms(name, r.synonyms)
    print(f"[keyframe_select] {name!r}: frame {r.frame_index}, visible={r.visible}, "
          f"synonyms={synonyms}  ({r.reasoning})")
    return int(r.frame_index), synonyms


_N_SYNONYMS = 3


def _clean_synonyms(name: str, raw: list[str]) -> list[str]:
    """Normalise the VLM's synonyms: strip, lowercase, drop dupes/the name, cap at 3.

    The cap lives here rather than at the call site so the robot's fixed
    ``_ROBOT_SYNONYMS`` list — which never passes through this function — keeps
    all of its entries.
    """
    out: list[str] = []
    for s in raw or []:
        s = str(s).strip().lower()
        if s and s != name.strip().lower() and s not in out:
            out.append(s)
    return out[:_N_SYNONYMS]


# ---------------------------------------------------------------------------
# Per-object helper for the segment controller
# ---------------------------------------------------------------------------

def select_keyframe_for_object(
    run_root: str, object_name: str, tried: set[int] | None = None,
    *, view: str = "primary", reference_frame: int | None = None,
) -> tuple[int | None, list[str]]:
    """Pick ``object_name``'s SAM3 prompt frame and its 3 synonyms.

    Returns ``(frame_index, synonyms)``. ``frame_index`` is None when no
    candidate is available (no frames on disk, all frames tried, or a forced
    keyframe already used); ``synonyms`` is empty whenever the VLM didn't run
    or didn't produce any.

    Called by ``subagents.segment_controller`` once per (object, view, round).
    Resolves the per-object description via ``_description_for``
    (discovered_object_descriptions / robot default) so the VLM has full context.

    ``reference_frame`` (round 2): a frame already tried for this object. Its
    image is shown first and the VLM is asked for the candidate LEAST like it,
    rather than the earliest visible one.

    Dual-view: ``view="primary"`` reads frames from
    ``stages.svo_extract.outputs.left_frames_dir``; ``view="secondary"``
    reads from ``stages.svo_extract_secondary.outputs.left_frames_dir``.
    The forced keyframe override is shared across views (a forced index
    applies to whichever view is asked, so the user has a single knob).
    """
    tried = set(tried or [])
    state = load_state(run_root)

    # Robot gets a fixed vocabulary rather than a VLM synonym call — it is not a
    # manipulation target, but downstream pose_align still needs its mask.
    if object_name == _ROBOT_NAME:
        synonyms = list(_ROBOT_SYNONYMS)
    else:
        synonyms = []

    # Forced keyframe override (VisualInput.keyframe_index >= 0): use it
    # verbatim on the first attempt; once in `tried` there's no alternative.
    forced = int((state.get("entry") or {}).get("keyframe_index", -1))
    if forced >= 0:
        return (None if forced in tried else forced), synonyms

    stage_key = "svo_extract" if view == "primary" else "svo_extract_secondary"
    sve = state.get("stages", {}).get(stage_key, {})
    left_dir = sve.get("outputs", {}).get("left_frames_dir") if sve.get("ok") else None
    if not left_dir:
        return None, synonyms

    all_frames = sorted(Path(left_dir).glob("frame_*.png"))
    if not all_frames:
        return None, synonyms

    frame_idx_by_path = {p: int(p.stem.split("_")[-1]) for p in all_frames}
    path_by_frame_idx = {v: k for k, v in frame_idx_by_path.items()}
    untried = [p for p in all_frames if frame_idx_by_path[p] not in tried]
    if not untried:
        return None, synonyms
    if len(untried) == 1:
        return frame_idx_by_path[untried[0]], synonyms

    sampled_paths = [untried[i] for i in _evenly_sample_indices(len(untried), _N_CANDIDATES)]
    sampled_indices = [frame_idx_by_path[p] for p in sampled_paths]

    descriptions = state.get("discovered_object_descriptions") or {}
    grasped = set(state.get("pickup_objects") or [])
    rel_by = {
        n: {"role": "grasped" if n in grasped else "related", "description": d}
        for n, d in descriptions.items()
    }
    role, description = _description_for(object_name, rel_by)
    system_prompt = _PROMPT_PATH.read_text()

    chosen = _pick_one_object(
        name=object_name, role=role, description=description,
        candidate_paths=sampled_paths, candidate_indices=sampled_indices,
        system_prompt=system_prompt,
        reference_path=path_by_frame_idx.get(reference_frame),
    )
    if chosen is None:
        # VLM failed; fall back to the earliest sampled untried frame.
        return sampled_indices[0], synonyms
    frame_idx, vlm_synonyms = chosen
    return frame_idx, (synonyms or vlm_synonyms)
