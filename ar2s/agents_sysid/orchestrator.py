"""Scene Prep ReAct orchestrator for the DROID pipeline.

This stage validates an emitted episode folder, prepares collision geometry,
and calibrates the scene. Grasp search is intentionally outside this
orchestrator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ar2s.agent_configs.models import chat_model_for
from ar2s.agents_sysid.calibrate import calibrate_scene as _calibrate_scene
from ar2s.agents_sysid.collision_prep import CollisionPrepReport, prep_episode
from ar2s.agents_sysid.episode import load_episode
from ar2s.agents_sysid.scene import CalibratedScene


_PROMPT_PATH = Path(__file__).with_name("orchestrator_prompt.md")
_DEFAULT_MAX_ITERATIONS = 8
_TOOL_ORDER = ["collision_prep", "calibrate_scene"]


@dataclass
class ScenePrepConfig:
    """Collision-prep and calibration parameters."""
    coacd_threshold: float = 0.05
    coacd_force: bool = False
    optimize_robot_base: bool = False


@dataclass
class _ScenePrepSink:
    """In-memory capture of each scene-prep tool's typed result."""
    collision: CollisionPrepReport | None = None
    calibration: CalibratedScene | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class ScenePrepResult:
    ok: bool
    sink: _ScenePrepSink
    agent_final: str = ""


def data_ingest(episode_dir: str | Path) -> str:
    """Validate the physical-prior-patched episode folder for scene prep.

    ``load_episode`` runs strict schema-v1 validation and requires object
    masses. The top-level DROID pipeline must run physical prior before this.
    """
    episode_dir = Path(episode_dir).expanduser().resolve()
    if not (episode_dir / "manifest.yaml").exists():
        raise FileNotFoundError(
            f"{episode_dir}/manifest.yaml does not exist. Scene prep expects an "
            "episode emitted by ar2s.agents_visual and patched by "
            "ar2s.agents_physical_prior."
        )
    load_episode(episode_dir)
    return str(episode_dir)


def _build_tools(config: ScenePrepConfig, sink: _ScenePrepSink) -> list:
    @tool
    def collision_prep(episode_dir: str) -> str:
        """CoACD convex-decompose objects and patch manifest collision_dir."""
        try:
            report = prep_episode(
                episode_dir,
                threshold=config.coacd_threshold,
                force=config.coacd_force,
            )
        except Exception as exc:  # noqa: BLE001 - surface tool halt to agent
            sink.errors.append(f"collision_prep: {type(exc).__name__}: {exc}")
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        sink.collision = report
        n_ok = sum(1 for r in report.per_object.values() if not r.error and r.pieces)
        if report.errors:
            sink.errors.extend(f"collision_prep: {err}" for err in report.errors)
        return json.dumps({
            "ok": not report.errors,
            "objects_with_collision": n_ok,
            "n_objects": len(report.per_object),
            "manifest_patched": report.manifest_patched,
            "errors": report.errors,
        })

    @tool
    def calibrate_scene(episode_dir: str) -> str:
        """Calibrate primary camera, robot base, and ground frame."""
        try:
            scene = _calibrate_scene(
                episode_dir,
                write_yaml=True,
                update_manifest=True,
                optimize_robot_base=config.optimize_robot_base,
            )
        except Exception as exc:  # noqa: BLE001
            sink.errors.append(f"calibrate_scene: {type(exc).__name__}: {exc}")
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        sink.calibration = scene
        return json.dumps({
            "ok": True,
            "primary_id": int(scene.camera_selection.primary_id),
            "robot_iou": round(float(scene.robot_alignment.iou), 4),
            "ground_offset_m": round(float(scene.ground.offset), 5),
            "ground_converged": bool(scene.ground.converged),
        })

    return [collision_prep, calibrate_scene]


def _system_message_with_caching() -> SystemMessage:
    return SystemMessage(content=[{
        "type": "text",
        "text": _PROMPT_PATH.read_text(),
        "cache_control": {"type": "ephemeral"},
    }])


def run_scene_prep(
    episode_dir: str | Path,
    *,
    config: ScenePrepConfig,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> ScenePrepResult:
    """Run collision prep and scene calibration under a ReAct orchestrator."""
    episode_dir = str(Path(episode_dir).expanduser().resolve())
    sink = _ScenePrepSink()

    llm = chat_model_for("scene_prep.droid.main.orchestrator")
    if llm is None:
        raise RuntimeError(
            "No API key for the configured scene_prep.droid.main.orchestrator "
            "model. Set the credential env var for the provider selected in "
            "the active agent config."
        )

    agent = create_react_agent(
        model=llm,
        tools=_build_tools(config, sink),
        prompt=_system_message_with_caching(),
    )

    initial_msg = (
        f"episode_dir = {episode_dir}\n\n"
        f"Call these tools exactly once each, in this order, passing "
        f"episode_dir to each: {_TOOL_ORDER}. After calibrate_scene returns "
        f"ok=true, stop and reply with the terminal success line. If any tool "
        f"returns ok=false, stop and reply with the terminal halt line."
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=initial_msg)]},
        config={"recursion_limit": max_iterations},
    )

    agent_final = _last_message_text(result)
    return _result_from_sink(sink, agent_final)


def _last_message_text(agent_result: dict) -> str:
    msgs = agent_result.get("messages") or []
    if not msgs:
        return ""
    content = getattr(msgs[-1], "content", "") or ""
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    return content


def _result_from_sink(sink: _ScenePrepSink, agent_final: str) -> ScenePrepResult:
    ok = bool(sink.collision and sink.calibration and not sink.errors)
    return ScenePrepResult(ok=ok, sink=sink, agent_final=agent_final)


__all__ = [
    "ScenePrepConfig",
    "ScenePrepResult",
    "data_ingest",
    "run_scene_prep",
]
