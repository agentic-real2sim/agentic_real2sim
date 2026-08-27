"""Internal vision-task mechanics for YAML-backed agent calls.

This module handles VLM-specific concerns — image data URL encoding,
multi-model fan-out voting, per-model token quirks, health-check pings.
ChatModel construction is driven by ``ar2s/agent_configs/*.yaml`` through
``ar2s.agent_configs.vlm_call*``. Callers should not select raw model specs
through this module.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ar2s.agent_configs.models import (
    _build_chat_model_from_spec,
    _parse_spec,
    _resolve_openai_reasoning,
    _resolve_openrouter_reasoning,
)


# ---------------------------------------------------------------------------
# Internal structured VLM calls
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return value


def _nested_value(value: Any, *path: str) -> Any:
    cur = value
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur


def _normalized_request_reasoning(
    provider: str,
    model_spec: str,
    reasoning_effort: str | bool | None,
) -> dict[str, Any] | None:
    if provider == "openai":
        return _resolve_openai_reasoning(reasoning_effort)
    if provider == "openrouter":
        return _resolve_openrouter_reasoning(reasoning_effort, model_spec.split(":", 1)[-1])
    return None


def _normalized_reasoning_tokens(message: Any) -> int | None:
    usage_metadata = getattr(message, "usage_metadata", None)
    reasoning = _nested_value(usage_metadata, "output_token_details", "reasoning")
    if reasoning is not None:
        return int(reasoning)

    response_metadata = getattr(message, "response_metadata", None) or {}
    response_usage = _nested_value(response_metadata, "usage")
    if response_usage is not None:
        reasoning = _nested_value(response_usage, "output_tokens_details", "reasoning_tokens")
        if reasoning is not None:
            return int(reasoning)
        reasoning = _nested_value(response_usage, "completion_tokens_details", "reasoning_tokens")
        if reasoning is not None:
            return int(reasoning)

    reasoning = _nested_value(response_metadata, "completion_tokens_details", "reasoning_tokens")
    if reasoning is not None:
        return int(reasoning)
    reasoning = _nested_value(response_metadata, "output_tokens_details", "reasoning_tokens")
    if reasoning is not None:
        return int(reasoning)
    return None


def _trace_vlm_call(
    *,
    agent_path: str,
    model_spec: str,
    provider: str,
    request_reasoning: dict[str, Any] | None,
    parsed_ok: bool,
    raw_message: Any | None,
    parsing_error: Any | None,
    trace_path: str,
) -> None:
    payload = {
        "agent_path": agent_path,
        "model_spec": model_spec,
        "provider": provider,
        "request_reasoning": _jsonable(request_reasoning),
        "parsed_ok": parsed_ok,
        "parsing_error": None if parsing_error is None else str(parsing_error),
        "response_metadata": _jsonable(getattr(raw_message, "response_metadata", None))
        if raw_message is not None
        else None,
        "usage_metadata": _jsonable(getattr(raw_message, "usage_metadata", None))
        if raw_message is not None
        else None,
        "reasoning_tokens": _normalized_reasoning_tokens(raw_message)
        if raw_message is not None
        else None,
    }
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _build_vlm_from_spec(
    model_spec: str,
    *,
    max_tokens: int = 1024,
    temperature: float | None = 0.1,
    request_timeout: float | None = None,
    reasoning_effort: str | bool | None = None,
):
    """Build a chat model for a ``provider:model`` spec (or bare NHR model).

    Thin wrapper over the YAML-backed provider factory. Returns the langchain
    chat model, or ``None`` if that provider's API key is unset.
    ``reasoning_effort`` is the per-agent provider-specific reasoning override.
    Direct OpenAI maps it to ``reasoning``; OpenRouter maps it to
    ``extra_body["reasoning"]``.
    """
    # The NHR gateway can be slow on multimodal Kimi-K2.6 (large base64 image
    # prompts); a single 60 s ceiling caused transient timeouts that aborted
    # whole critic loops (e.g. box detection -> object dropped). Default to a
    # generous timeout, overridable via AR2S_VLM_TIMEOUT.
    if request_timeout is None:
        request_timeout = float(os.environ.get("AR2S_VLM_TIMEOUT", "180"))
    return _build_chat_model_from_spec(
        model_spec,
        max_tokens=max_tokens,
        temperature=temperature,
        request_timeout=request_timeout,
        reasoning_effort=reasoning_effort,
    )


def _build_messages(system: str, user_text: str, images: list) -> list:
    """Compose the multimodal HumanMessage.

    ``images`` entries are either plain data-URL strings (legacy — all images
    up front) or ``{"label": str, "url": str}`` dicts, in which case each
    image is immediately preceded by its own text block. Interleaved labels
    let judges receive a rich per-image introduction instead of one index
    list at the end (scene_view_repair experiments showed VLMs track 6+
    images noticeably better this way).
    """
    parts: list[dict] = []
    for item in images:
        if isinstance(item, dict) and "url" in item:
            if item.get("label"):
                parts.append({"type": "text", "text": item["label"]})
            parts.append({"type": "image_url", "image_url": {"url": item["url"]}})
        else:
            parts.append({"type": "image_url", "image_url": {"url": item}})
    return [
        SystemMessage(content=system),
        HumanMessage(content=parts + [{"type": "text", "text": user_text}]),
    ]


def _query_one(
    agent_path: str,
    model_id: str,
    system: str,
    user_text: str,
    images: list[str],
    output_schema: Type[T],
    *,
    max_tokens: int = 1024,
    temperature: float | None = 0.1,
    reasoning_effort: str | bool | None = None,
) -> T:
    """Single-model query with structured Pydantic output. Raises on failure.

    ``model_id`` may be a ``provider:model`` spec. Structured-output method is
    chosen per provider: OpenAI-family (openai / qwen / nhr_fau) use
    ``json_schema``; Anthropic uses langchain's default (tool-use), which is
    what ChatAnthropic supports for structured output.
    """
    provider, _ = _parse_spec(model_id)
    trace_path = os.environ.get("AR2S_VLM_TRACE_PATH")
    request_reasoning = _normalized_request_reasoning(provider, model_id, reasoning_effort)
    include_raw = trace_path is not None or (
        provider == "openai" and request_reasoning is not None
    )
    llm = _build_vlm_from_spec(
        model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    if llm is None:
        raise RuntimeError(f"no API key configured for provider in {model_id!r}")
    if provider == "anthropic":
        structured = llm.with_structured_output(output_schema, include_raw=include_raw)
    else:
        structured = llm.with_structured_output(
            output_schema,
            method="json_schema",
            include_raw=include_raw,
        )
    messages = _build_messages(system, user_text, images)
    # Retry transient gateway failures (timeouts, 5xx, connection resets) with
    # exponential backoff. Without this a single hiccup returns None for the
    # model and silently degrades critic loops. Overridable via AR2S_VLM_RETRIES.
    attempts = max(1, int(os.environ.get("AR2S_VLM_RETRIES", "3")))
    last_exc: Exception | None = None
    last_traced_exc: Exception | None = None
    for i in range(attempts):
        try:
            result = structured.invoke(messages)
            if include_raw:
                assert isinstance(result, dict), (
                    f"{model_id!r}: expected include_raw structured output dict"
                )
                raw_message = result.get("raw")
                parsed = result.get("parsed")
                parsing_error = result.get("parsing_error")
                if trace_path is not None:
                    _trace_vlm_call(
                        agent_path=agent_path,
                        model_spec=model_id,
                        provider=provider,
                        request_reasoning=request_reasoning,
                        parsed_ok=parsing_error is None and parsed is not None,
                        raw_message=raw_message,
                        parsing_error=parsing_error,
                        trace_path=trace_path,
                    )
                if parsing_error is not None:
                    last_traced_exc = parsing_error
                    raise parsing_error
                assert parsed is not None, f"{model_id!r}: parsed structured output is None"
                return parsed
            return result
        except Exception as exc:  # noqa: BLE001 — surface only after all retries
            last_exc = exc
            if trace_path is not None and exc is not last_traced_exc:
                _trace_vlm_call(
                    agent_path=agent_path,
                    model_spec=model_id,
                    provider=provider,
                    request_reasoning=request_reasoning,
                    parsed_ok=False,
                    raw_message=None,
                    parsing_error=exc,
                    trace_path=trace_path,
                )
            if i < attempts - 1:
                time.sleep(2.0 * (2 ** i))
    raise last_exc  # type: ignore[misc]


def _query_many(
    agent_path: str,
    models: list[str],
    system: str,
    user_text: str,
    images: list[str],
    output_schema: Type[T],
    *,
    max_tokens: int = 1024,
    temperature: float | None = 0.1,
    reasoning_effort: str | bool | None = None,
) -> dict[str, T | None]:
    """Fan out the same query to every model in parallel.

    Returns ``{model_id: parsed_response | None}``. Failures (timeout, parse
    error, gateway hiccup) surface as ``None`` for that model — let the
    caller decide voting / fallback policy.
    """
    def _go(model_id: str) -> tuple[str, T | None, str]:
        try:
            return model_id, _query_one(
                agent_path,
                model_id,
                system,
                user_text,
                images,
                output_schema,
                max_tokens=max_tokens, temperature=temperature,
                reasoning_effort=reasoning_effort,
            ), ""
        except Exception as exc:
            return model_id, None, str(exc)[:120]

    out: dict[str, T | None] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = {pool.submit(_go, m): m for m in models}
        for fut in as_completed(futures):
            model_id, parsed, err = fut.result()
            out[model_id] = parsed
            if err:
                errors[model_id] = err

    # Print per-model status for log readability (caller can suppress by
    # redirecting stdout; this is meant for interactive / runner usage).
    for m in models:
        short = m.split("/")[-1][:46]
        if out[m] is None:
            print(f"  [vlm] {short}: FAILED — {errors.get(m, 'unknown error')}")
        else:
            print(f"  [vlm] {short}: OK")
    return out


def _first_success(responses: dict[str, T | None], models: list[str]) -> tuple[T | None, str]:
    """Return the first non-null response in configured model preference order."""
    for model_id in models:
        parsed = responses.get(model_id)
        if parsed is not None:
            return parsed, model_id
    return None, ""


def _ping_models(models: list[str]) -> list[str]:
    """Tiny text-only probe to filter out offline models before real work."""
    ids = list(models)
    if not ids:
        return []
    def _probe(m: str) -> tuple[str, bool]:
        try:
            llm = _build_vlm_from_spec(m, max_tokens=5)
            if llm is None:
                return m, False
            llm.invoke([HumanMessage(content="Reply: OK")])
            return m, True
        except Exception:
            return m, False
    live: list[str] = []
    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        futures = {pool.submit(_probe, m): m for m in ids}
        for fut in as_completed(futures):
            m, ok = fut.result()
            if ok:
                live.append(m)
    return [m for m in ids if m in live]   # preserve declared order
