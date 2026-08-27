"""YAML-backed model/runtime helpers for agent LLM and VLM calls."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_ACTIVE_CONFIG_PATH: Path | None = None
_CONFIG_CACHE: dict[str, Any] | None = None
_AGENT_CACHE: dict[str, dict[str, Any]] | None = None

_SUPPORTED_TYPES = {"llm", "vlm"}
_SUPPORTED_POLICIES = {"single", "fanout_all", "first_success"}
_GRASP_OPTIMIZATION_STRATEGIES = {"sweep", "loop"}
_GRASP_LOOP_REQUIRED_SUBAGENTS = (
    "grasp_optimization.subagents.grasp_loop_view_sufficiency",
    "grasp_optimization.subagents.grasp_loop_shift_decision",
)
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENAI_CREDENTIAL_ENVS = {
    "OPENAI_API_AR2S_KEY",
    "OPENAI_API_UBC_PHYSAI_AR2S_KEY",
}

_PROVIDER_MODEL_ALIASES = {
    "openrouter": {
        "moonshotai/Kimi-K2.6": "moonshotai/kimi-k2.6",
        "Qwen/Qwen3.6-35B-A3B": "qwen/qwen3.6-35b-a3b",
    },
}


def default_config_path() -> Path:
    return Path(__file__).resolve().with_name("config_anthropic_opus47.yaml")


def set_config_path(path: str | Path | None) -> None:
    """Set the active YAML config path for this process."""
    global _ACTIVE_CONFIG_PATH, _CONFIG_CACHE, _AGENT_CACHE
    if path is None:
        _ACTIVE_CONFIG_PATH = None
    else:
        p = Path(path).expanduser().resolve()
        assert p.is_file(), f"agent config not found: {p}"
        _ACTIVE_CONFIG_PATH = p
    _CONFIG_CACHE = None
    _AGENT_CACHE = None


def active_config_path() -> Path:
    return _ACTIVE_CONFIG_PATH or default_config_path()


def reload_config() -> None:
    global _CONFIG_CACHE, _AGENT_CACHE
    _CONFIG_CACHE = None
    _AGENT_CACHE = None


def _load_config() -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        path = active_config_path()
        assert path.is_file(), f"agent config not found: {path}"
        data = yaml.safe_load(path.read_text()) or {}
        assert isinstance(data, dict), f"{path}: top-level YAML must be a mapping"
        _validate_config(data, path)
        _CONFIG_CACHE = data
    return _CONFIG_CACHE


def _validate_config(data: dict[str, Any], path: Path) -> None:
    assert data.get("schema_version") == 1, f"{path}: schema_version must be 1"
    providers = data.get("providers")
    assert isinstance(providers, dict) and providers, f"{path}: missing providers"
    for provider_name, provider_cfg in providers.items():
        assert isinstance(provider_name, str) and provider_name, (
            f"{path}: provider names must be non-empty strings"
        )
        assert isinstance(provider_cfg, dict), (
            f"{path}: providers.{provider_name} must be a mapping"
        )
        assert "use_head_node_relay" not in provider_cfg, (
            f"{path}: providers.{provider_name}.use_head_node_relay is no longer supported"
        )
        stale_flag = "allows_" + "fallback_key"
        assert stale_flag not in provider_cfg, (
            f"{path}: providers.{provider_name}.{stale_flag} is no longer supported"
        )
        credential_env = provider_cfg.get("credential_env")
        assert credential_env is None or (
            isinstance(credential_env, str) and credential_env
        ), f"{path}: providers.{provider_name}.credential_env must be a string"
        plural_credential_key = "credential_" + "envs"
        assert plural_credential_key not in provider_cfg, (
            f"{path}: providers.{provider_name}.{plural_credential_key} is no longer "
            "supported; choose one credential_env"
        )
        assert provider_name != "openai" or credential_env in _OPENAI_CREDENTIAL_ENVS, (
            f"{path}: providers.openai.credential_env must be one of "
            f"{sorted(_OPENAI_CREDENTIAL_ENVS)}"
        )
        base_url = provider_cfg.get("base_url")
        assert base_url is None or (isinstance(base_url, str) and base_url), (
            f"{path}: providers.{provider_name}.base_url must be a string"
        )
    stages = data.get("stages")
    assert isinstance(stages, dict) and stages, f"{path}: missing stages"
    agents = _collect_agent_entries(stages)
    assert agents, f"{path}: no agent entries found"
    sysid = stages.get("sysid") or {}
    assert isinstance(sysid, dict), f"{path}: stages.sysid must be a mapping"
    assert "droid" not in sysid, (
        f"{path}: stages.sysid.droid is obsolete; use stages.grasp_optimization"
    )
    grasp_cfg = stages.get("grasp_optimization")
    assert isinstance(grasp_cfg, dict), (
        f"{path}: missing stages.grasp_optimization"
    )
    strategy = grasp_cfg.get("sweep_or_loop")
    assert strategy in _GRASP_OPTIMIZATION_STRATEGIES, (
        f"{path}: stages.grasp_optimization.sweep_or_loop must be one of "
        f"{sorted(_GRASP_OPTIMIZATION_STRATEGIES)}"
    )
    if strategy == "loop":
        missing = [
            agent_path
            for agent_path in _GRASP_LOOP_REQUIRED_SUBAGENTS
            if agent_path not in agents
        ]
        assert not missing, (
            f"{path}: sweep_or_loop=loop requires configured subagents: {missing}"
        )
    for agent_path, cfg in agents.items():
        assert cfg.get("type") in _SUPPORTED_TYPES, (
            f"{path}: {agent_path}: type must be one of {sorted(_SUPPORTED_TYPES)}"
        )
        models = cfg.get("models")
        assert isinstance(models, list) and models, (
            f"{path}: {agent_path}: models must be a non-empty list"
        )
        assert all(isinstance(m, str) and m for m in models), (
            f"{path}: {agent_path}: every model spec must be a non-empty string"
        )
        assert "temperature" in cfg, f"{path}: {agent_path}: missing temperature"
        temp = cfg.get("temperature")
        assert temp is None or isinstance(temp, (int, float)), (
            f"{path}: {agent_path}: temperature must be null or numeric"
        )
        assert isinstance(cfg.get("max_tokens"), int) and cfg["max_tokens"] > 0, (
            f"{path}: {agent_path}: max_tokens must be a positive int"
        )
        policy = cfg.get("response_policy", "single")
        assert policy in _SUPPORTED_POLICIES, (
            f"{path}: {agent_path}: response_policy must be one of "
            f"{sorted(_SUPPORTED_POLICIES)}"
        )
        if "reasoning_effort" in cfg:
            re_val = cfg["reasoning_effort"]
            assert (
                re_val is None
                or re_val is False
                or (isinstance(re_val, str) and re_val)
            ), (
                f"{path}: {agent_path}: reasoning_effort must be null, false, "
                f"or a non-empty string (OpenRouter-only; ignored for other "
                f"providers)"
            )
        for spec in models:
            provider, _ = _parse_spec(spec, providers)
            assert provider in providers, f"{path}: {agent_path}: unknown provider {provider!r}"


def _collect_agent_entries(node: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, value in node.items():
        if not isinstance(value, dict):
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        if "type" in value and "models" in value:
            out[path] = value
        else:
            out.update(_collect_agent_entries(value, path))
    return out


def _agents() -> dict[str, dict[str, Any]]:
    global _AGENT_CACHE
    if _AGENT_CACHE is None:
        _AGENT_CACHE = _collect_agent_entries(_load_config()["stages"])
    return _AGENT_CACHE


def iter_agent_paths() -> list[str]:
    return sorted(_agents())


def get_agent_config(agent_path: str) -> dict[str, Any]:
    agents = _agents()
    if agent_path not in agents:
        raise KeyError(
            f"unknown agent_path {agent_path!r}; known: {sorted(agents)}"
        )
    return dict(agents[agent_path])


def get_stage_config(stage_path: str) -> dict[str, Any]:
    node: Any = _load_config()["stages"]
    for part in stage_path.split("."):
        assert isinstance(node, dict), (
            f"{active_config_path()}: stages.{stage_path} is not a mapping"
        )
        assert part in node, (
            f"{active_config_path()}: unknown stage config {stage_path!r}"
        )
        node = node[part]
    assert isinstance(node, dict), (
        f"{active_config_path()}: stages.{stage_path} must be a mapping"
    )
    return dict(node)


def get_grasp_optimization_config() -> dict[str, Any]:
    return get_stage_config("grasp_optimization")


def _providers() -> dict[str, dict[str, Any]]:
    return _load_config()["providers"]


def _parse_spec(model_spec: str, providers: dict[str, Any] | None = None) -> tuple[str, str]:
    providers = providers or _providers()
    if ":" in model_spec:
        prefix, rest = model_spec.split(":", 1)
        if prefix in providers:
            return prefix, rest
        raise ValueError(f"unknown provider prefix in model spec {model_spec!r}")
    return "nhr_fau", model_spec


def _provider_cfg(provider: str) -> dict[str, Any]:
    cfg = _providers().get(provider)
    assert isinstance(cfg, dict), f"unknown provider {provider!r}"
    return cfg


def _min_tokens_for(model_id: str) -> int:
    overrides = _load_config().get("model_overrides") or {}
    raw = overrides.get(model_id) or {}
    return int(raw.get("min_max_tokens") or 0)


def _openrouter_reasoning_for(model_id: str) -> dict[str, str] | None:
    overrides = _load_config().get("model_overrides") or {}
    raw = overrides.get(model_id) or {}
    effort = raw.get("openrouter_reasoning_effort")
    if effort is None:
        return None
    assert isinstance(effort, str) and effort, (
        f"{active_config_path()}: model_overrides.{model_id}.openrouter_reasoning_effort "
        "must be a non-empty string"
    )
    return {"effort": effort}


# Per-agent ``reasoning_effort`` values that mean "explicitly turn reasoning
# off for this agent" (as opposed to null/absent, which falls back to the
# model_overrides global). OpenRouter disables reasoning via
# ``reasoning: {"enabled": false}``.
_REASONING_DISABLE_VALUES = {"none", "off", "false"}


def _resolve_openrouter_reasoning(
    reasoning_effort: str | bool | None,
    model_id: str,
) -> dict[str, Any] | None:
    """Compute the OpenRouter ``reasoning`` request field for one agent call.

    ``reasoning_effort`` is the per-agent agent-config override:

    - ``None`` (field null/absent): fall back to the ``model_overrides``
      global effort for this model, preserving legacy behavior.
    - ``False`` or a disable string ("none"/"off"/"false"): explicitly disable
      reasoning for this agent, even if the global sets an effort.
    - any other non-empty string ("low"/"medium"/"high"/...): use that effort,
      overriding the global.

    Returns the dict to place under ``extra_body["reasoning"]``, or ``None`` to
    add no reasoning field at all (no per-agent value and no global).
    """
    if reasoning_effort is None:
        return _openrouter_reasoning_for(model_id)
    if reasoning_effort is False or (
        isinstance(reasoning_effort, str)
        and reasoning_effort.strip().lower() in _REASONING_DISABLE_VALUES
    ):
        return {"enabled": False}
    assert isinstance(reasoning_effort, str) and reasoning_effort, (
        "reasoning_effort must be null, false, or a non-empty string"
    )
    return {"effort": reasoning_effort}


def _resolve_openai_reasoning(
    reasoning_effort: str | bool | None,
) -> dict[str, str] | None:
    """Compute the direct OpenAI Responses API ``reasoning`` request field."""
    if reasoning_effort is None:
        return None
    if reasoning_effort is False or (
        isinstance(reasoning_effort, str)
        and reasoning_effort.strip().lower() in _REASONING_DISABLE_VALUES
    ):
        return {"effort": "none"}
    assert isinstance(reasoning_effort, str) and reasoning_effort, (
        "reasoning_effort must be null, false, or a non-empty string"
    )
    return {"effort": reasoning_effort}


# Supports ${VAR} and ${VAR:-default}. Used so a base_url can be pointed at a
# runtime-discovered endpoint (e.g. a vLLM node on an HPC cluster) without
# editing YAML per run.
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_vars(value: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name) or (default or "")

    return _ENV_VAR_RE.sub(_repl, value)


def _provider_base_url(provider: str) -> str | None:
    raw = _provider_cfg(provider).get("base_url")
    if raw is None:
        return _DEFAULT_OPENROUTER_BASE_URL if provider == "openrouter" else None
    return _expand_env_vars(raw)


def _provider_credential_env(provider: str) -> str | None:
    cfg = _provider_cfg(provider)
    raw = cfg.get("credential_env")
    if raw is not None:
        assert isinstance(raw, str) and raw
        assert provider != "openai" or raw in _OPENAI_CREDENTIAL_ENVS
        return raw

    if provider == "openrouter":
        return "OPENROUTER_API_AR2S_KEY"
    if provider == "openai":
        return "OPENAI_API_AR2S_KEY"
    if provider == "nhr_fau":
        return "NHR_FAU_GATEWAY_API_KEY"
    return None


def _provider_model_id(provider: str, model_id: str) -> str:
    return _PROVIDER_MODEL_ALIASES.get(provider, {}).get(model_id, model_id)


def _build_openai_compatible_chat_model(
    provider: str,
    model_id: str,
    *,
    resolved_max: int,
    oai_opts: dict[str, Any],
):
    credential_env = _provider_credential_env(provider)
    assert credential_env, f"{provider} provider requires credential_env"
    api_key = os.environ.get(credential_env)
    if not api_key:
        return None
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": resolved_max,
        "api_key": api_key,
        **oai_opts,
    }
    base_url = _provider_base_url(provider)
    if base_url is not None:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _build_chat_model_from_spec(
    model_spec: str,
    *,
    max_tokens: int,
    temperature: float | None = None,
    request_timeout: float | None = None,
    reasoning_effort: str | bool | None = None,
):
    """Build a LangChain chat model for an explicit YAML-derived model spec.

    ``reasoning_effort`` is the per-agent reasoning override. Direct OpenAI
    sends it as top-level ``reasoning`` for Responses API calls; OpenRouter
    sends its provider-specific value under ``extra_body["reasoning"]``.
    """
    provider, model_id = _parse_spec(model_spec)
    resolved_max = max(int(max_tokens), _min_tokens_for(model_id))

    oai_opts: dict[str, Any] = {}
    if provider != "anthropic" and temperature is not None:
        oai_opts["temperature"] = float(temperature)
    if provider != "anthropic" and request_timeout is not None:
        oai_opts["request_timeout"] = float(request_timeout)

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_id, max_tokens=resolved_max)

    if provider == "openai":
        reasoning = _resolve_openai_reasoning(reasoning_effort)
        if reasoning is not None:
            oai_opts["reasoning"] = reasoning
        return _build_openai_compatible_chat_model(
            "openai",
            model_id,
            resolved_max=resolved_max,
            oai_opts=oai_opts,
        )

    if provider == "qwen":
        if not os.environ.get("QWEN_API_KEY"):
            return None
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_id,
            max_tokens=resolved_max,
            base_url=_provider_base_url("qwen"),
            api_key=os.environ["QWEN_API_KEY"],
            **oai_opts,
        )

    if provider == "nhr_fau":
        return _build_openai_compatible_chat_model(
            "nhr_fau",
            model_id,
            resolved_max=resolved_max,
            oai_opts=oai_opts,
        )

    if provider == "drac":
        # Self-hosted OpenAI-compatible endpoint (local vLLM on the cluster).
        # base_url and credential_env come from the provider block; base_url
        # is typically pointed at the serving node via ${GEMMA_BASE_URL}.
        return _build_openai_compatible_chat_model(
            "drac",
            model_id,
            resolved_max=resolved_max,
            oai_opts=oai_opts,
        )

    if provider == "openrouter":
        translated_model_id = _provider_model_id(provider, model_id)
        reasoning = _resolve_openrouter_reasoning(reasoning_effort, model_id)
        if reasoning is not None:
            oai_opts["extra_body"] = {"reasoning": reasoning}

        credential_env = _provider_credential_env("openrouter")
        assert credential_env, "openrouter provider requires credential_env"
        openrouter_key = os.environ.get(credential_env)
        if not openrouter_key:
            return None
        from langchain_openai import ChatOpenAI
        kwargs: dict[str, Any] = {
            "model": translated_model_id,
            "max_tokens": resolved_max,
            "api_key": openrouter_key,
            # Retry transient HTTP 5xx/429/timeouts at the client layer. Slow
            # reasoning models on OpenRouter occasionally 5xx; without this an
            # orchestrator agent.invoke halts the whole pipeline (no retry).
            "max_retries": 5,
            "timeout": 180.0,
            **oai_opts,
        }
        base_url = _provider_base_url("openrouter")
        if base_url is not None:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    raise ValueError(f"unsupported provider {provider!r}")


def chat_model_for(agent_path: str, *, request_timeout: float | None = None):
    """Return the single configured chat model for a text or multimodal agent."""
    cfg = get_agent_config(agent_path)
    models = list(cfg["models"])
    assert len(models) == 1, (
        f"{agent_path}: chat_model_for requires exactly one configured model; "
        f"got {len(models)}"
    )
    return _build_chat_model_from_spec(
        models[0],
        max_tokens=int(cfg["max_tokens"]),
        temperature=cfg.get("temperature"),
        request_timeout=request_timeout,
        reasoning_effort=cfg.get("reasoning_effort"),
    )


def _assert_vlm(agent_path: str, cfg: dict[str, Any]) -> None:
    if cfg.get("type") != "vlm":
        raise ValueError(
            f"agent {agent_path!r} has type={cfg.get('type')!r}, not 'vlm'"
        )


def vlm_call(
    agent_path: str,
    *,
    system: str,
    user_text: str,
    images: list[str],
    output_schema: Type[T],
) -> dict[str, T | None]:
    """Fan out one structured VLM query to every YAML-configured model."""
    cfg = get_agent_config(agent_path)
    _assert_vlm(agent_path, cfg)
    from ar2s.agents_visual._vlm import _query_many

    return _query_many(
        agent_path=agent_path,
        models=list(cfg["models"]),
        system=system,
        user_text=user_text,
        images=images,
        output_schema=output_schema,
        max_tokens=int(cfg["max_tokens"]),
        temperature=cfg.get("temperature"),
        reasoning_effort=cfg.get("reasoning_effort"),
    )


def vlm_call_single(
    agent_path: str,
    *,
    system: str,
    user_text: str,
    images: list[str],
    output_schema: Type[T],
) -> T | None:
    """Run a structured VLM query for a single-model YAML entry."""
    cfg = get_agent_config(agent_path)
    _assert_vlm(agent_path, cfg)
    models = list(cfg["models"])
    assert len(models) == 1, (
        f"{agent_path}: vlm_call_single requires exactly one configured model; "
        f"got {len(models)}"
    )
    responses = vlm_call(
        agent_path,
        system=system,
        user_text=user_text,
        images=images,
        output_schema=output_schema,
    )
    return next(iter(responses.values()))


def vlm_call_first_success(
    agent_path: str,
    *,
    system: str,
    user_text: str,
    images: list[str],
    output_schema: Type[T],
) -> T | None:
    """Run configured models and return the first successful parsed response."""
    cfg = get_agent_config(agent_path)
    _assert_vlm(agent_path, cfg)
    responses = vlm_call(
        agent_path,
        system=system,
        user_text=user_text,
        images=images,
        output_schema=output_schema,
    )
    for model_spec in cfg["models"]:
        parsed = responses.get(model_spec)
        if parsed is not None:
            return parsed
    return None


def claude_to_lc_messages(content_blocks: list[dict]) -> list:
    """Convert Claude-style content blocks into one LangChain HumanMessage."""
    from langchain_core.messages import HumanMessage

    norm: list[dict] = []
    for block in content_blocks:
        btype = block.get("type")
        if btype == "text":
            norm.append({"type": "text", "text": block["text"]})
        elif btype == "image":
            src = block["source"]
            data = src["data"]
            media_type = src.get("media_type", "image/jpeg")
            norm.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            })
        elif btype == "image_url":
            image_url = block["image_url"]
            url = image_url if isinstance(image_url, str) else image_url["url"]
            norm.append({"type": "image_url", "image_url": {"url": url}})
        else:
            norm.append(block)
    return [HumanMessage(content=norm)]


def parse_text(response: Any) -> str:
    """Extract response text from a LangChain AIMessage-like response."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


__all__ = [
    "active_config_path",
    "chat_model_for",
    "claude_to_lc_messages",
    "default_config_path",
    "get_agent_config",
    "iter_agent_paths",
    "parse_text",
    "reload_config",
    "set_config_path",
    "vlm_call",
    "vlm_call_first_success",
    "vlm_call_single",
]
