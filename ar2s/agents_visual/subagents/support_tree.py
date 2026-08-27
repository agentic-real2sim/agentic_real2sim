"""support_tree subagent — VLM infers per-object physical support relations.

Builds a single-parent tree over the relevant scene objects, rooted at the
sentinel "GROUND" (a virtual node representing the lowest physical surface
visible in the scene — could be the real floor, could be a table top).

Runs AFTER ``object_discovery`` and BEFORE ``segment``: needs only the
first frame plus the names of discovered objects, no masks. Single VLM call
with structured output; if the result fails validation (missing object,
unknown parent, cycle, etc.) a single retry is made; on second failure we
fall back to "everyone on GROUND" with ``support_tree_degraded: True`` so
the pipeline never stalls here.

Reads from state.json:
  - discovered_objects                          (list[str], snake_case; robot excluded by this subagent)
  - stages.svo_extract.outputs.left_frames_dir  (we use frame_000000.png)

Writes to state.json:
  - support_tree = {
        "relations":           [{"object": str, "supported_by": str}, ...],
        "degraded":            bool,
        "degraded_reason":     str | None,
    }
  - history += {"agent": "support_tree", "note": "..."}

Returns small JSON:
  {"ok": bool, "n_relations": int, "degraded": bool, "error"?: str}.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ar2s.agents_visual._models import vlm_call_first_success
from ar2s.agent_configs import image_to_data_url
from ar2s.agents_visual.state import append_history, load_state, save_state
from ar2s.droid_sim._util import snake_case_name


_PROMPT_PATH = Path(__file__).with_name("support_tree_prompt.md")
_AGENT_PATH = "visual.subagents.support_tree"
_GROUND = "GROUND"
_ROBOT_NAMES = {
    "robot", "robot arm", "robotic arm", "arm", "gripper",
    "robot manipulator", "robotic gripper",
}
_ROBOT_NAME_IDS = {snake_case_name(n) for n in _ROBOT_NAMES}


class _Edge(BaseModel):
    object: str = Field(
        description="Object name as snake_case from the input list."
    )
    supported_by: str = Field(
        description="Parent: another snake_case object name from the list, OR 'GROUND'."
    )


class _Tree(BaseModel):
    relations: list[_Edge] = Field(default_factory=list)


def _candidates(state: dict) -> list[str]:
    """Names that must appear in the tree (robot always excluded)."""
    pool = state.get("discovered_objects") or []
    names = [snake_case_name(str(n)) for n in pool if n]
    return [n for n in dict.fromkeys(names) if n not in _ROBOT_NAME_IDS]


def _first_frame(state: dict) -> Path | None:
    svo = state.get("stages", {}).get("svo_extract", {})
    if not svo.get("ok"):
        return None
    left_dir = (svo.get("outputs") or {}).get("left_frames_dir") or ""
    if not left_dir:
        return None
    p = Path(left_dir) / "frame_000000.png"
    return p if p.exists() else None


def _validate(rels: list[dict], expected: list[str]) -> tuple[bool, str]:
    """Return (ok, reason). On ok, ``rels`` is canonical-cased + complete.

    Mutates ``rels`` in place: converts names to snake_case, drops unknown
    'object' entries, drops robot rows. Reports the first concrete problem found.
    """
    expected_set = {snake_case_name(n) for n in expected}
    valid_names = expected_set | {_GROUND}
    seen: set[str] = set()

    # First pass: normalize + collect.
    norm: list[dict] = []
    for r in rels:
        o = snake_case_name(r.get("object") or "")
        p_raw = (r.get("supported_by") or "").strip()
        # Sentinel is case-sensitive in the schema, but accept GROUND/ground/Ground.
        p = _GROUND if p_raw.lower() == "ground" else snake_case_name(p_raw)
        if not o or o in _ROBOT_NAME_IDS:
            continue  # silently drop robot rows
        if o in seen:
            return False, f"duplicate object {o!r}"
        if o not in expected_set:
            return False, f"unknown object {o!r} (not in relevant list)"
        if p not in valid_names:
            return False, f"{o!r} parent {p!r} is not a known object or GROUND"
        seen.add(o)
        norm.append({"object": o, "supported_by": p})

    missing = expected_set - seen
    if missing:
        return False, f"missing objects: {sorted(missing)}"

    # Cycle check via BFS from GROUND down the children.
    children: dict[str, list[str]] = {}
    for row in norm:
        children.setdefault(row["supported_by"], []).append(row["object"])
    reached: set[str] = set()
    stack = list(children.get(_GROUND, []))
    while stack:
        node = stack.pop()
        if node in reached:
            return False, f"cycle or duplicate-support involving {node!r}"
        reached.add(node)
        stack.extend(children.get(node, []))
    not_reached = expected_set - reached
    if not_reached:
        # Either a cycle disconnected them from GROUND, or some node points at
        # itself / forms an island. Either way the tree is invalid.
        return False, f"nodes unreachable from GROUND: {sorted(not_reached)}"

    rels[:] = norm
    return True, ""


def _emit(run_root: str, relations: list[dict], *, degraded: bool,
          degraded_reason: str | None, note: str) -> str:
    state = load_state(run_root)
    state["support_tree"] = {
        "relations":       relations,
        "degraded":        degraded,
        "degraded_reason": degraded_reason,
    }
    save_state(state, run_root)
    append_history(run_root, "support_tree", note=note)
    return json.dumps({
        "ok":           True,
        "n_relations":  len(relations),
        "degraded":     degraded,
    })


def _flat_fallback(run_root: str, candidates: list[str], reason: str) -> str:
    """Degraded fallback: every object on GROUND."""
    relations = [{"object": n, "supported_by": _GROUND} for n in candidates]
    print(f"[support_tree] FALLBACK ({reason}): {len(relations)} objects all -> GROUND")
    return _emit(
        run_root, relations,
        degraded=True,
        degraded_reason=reason,
        note=f"degraded fallback ({reason}); {len(relations)} obj -> GROUND",
    )


def _call_vlm(image_url: str, candidates: list[str], system: str) -> _Tree | None:
    user_text = (
        f"Objects to place in the tree (one row per name, exact spelling): "
        f"{candidates}.\n\n"
        f"Look at the first frame and decide each object's `supported_by`. "
        f"Object names are snake_case. Use the sentinel 'GROUND' for objects "
        f"sitting on the lowest visible surface. Skip any robot/gripper entries."
    )
    return vlm_call_first_success(
        _AGENT_PATH,
        system=system,
        user_text=user_text,
        images=[image_url],
        output_schema=_Tree,
    )


@tool
def support_tree(run_root: str) -> str:
    """Infer per-object physical support relations (single-parent tree, root=GROUND).

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            ``state.discovered_objects`` (from object_discovery) and a
            ``stages.svo_extract.outputs.left_frames_dir`` containing
            ``frame_000000.png``.

    Returns:
        JSON string:
          {"ok": True, "n_relations": int, "degraded": bool}
        Degraded fallback (still ok=True) flattens every object onto
        ``GROUND`` with ``support_tree.degraded = True`` and a reason in
        ``support_tree.degraded_reason``.
    """
    state = load_state(run_root)
    candidates = _candidates(state)
    if not candidates:
        return json.dumps({
            "ok": False,
            "error": "state.discovered_objects empty — run object_discovery first",
        })

    first_frame = _first_frame(state)
    if first_frame is None:
        return _flat_fallback(run_root, candidates,
                              "no svo_extract frame_000000.png on disk")

    print(f"[support_tree] candidates={candidates}")
    image = image_to_data_url(first_frame)
    system = _PROMPT_PATH.read_text()

    last_err = ""
    for attempt in (1, 2):
        result = _call_vlm(image, candidates, system)
        if result is None:
            last_err = "VLM returned no parsable result"
            print(f"[support_tree] attempt {attempt}: {last_err}")
            continue
        rels = [e.model_dump() for e in result.relations]
        ok, why = _validate(rels, candidates)
        if ok:
            print(f"[support_tree] tree ok: "
                  f"{[(r['object'], r['supported_by']) for r in rels]}")
            return _emit(
                run_root, rels,
                degraded=False, degraded_reason=None,
                note=f"tree built ({len(rels)} edges); "
                     f"roots={[r['object'] for r in rels if r['supported_by'] == _GROUND]}",
            )
        last_err = why
        print(f"[support_tree] attempt {attempt}: invalid ({why}); retrying")

    return _flat_fallback(run_root, candidates,
                          f"VLM failed validation twice: {last_err}")
