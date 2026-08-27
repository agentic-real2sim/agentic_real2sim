"""Compatibility shims for YAML-backed agent model configuration."""
from __future__ import annotations

from ar2s.agent_configs.models import (
    chat_model_for,
    get_agent_config,
    reload_config,
    set_config_path,
    vlm_call,
    vlm_call_first_success,
    vlm_call_single,
)

llm_for = chat_model_for

__all__ = [
    "chat_model_for",
    "get_agent_config",
    "llm_for",
    "reload_config",
    "set_config_path",
    "vlm_call",
    "vlm_call_first_success",
    "vlm_call_single",
]
