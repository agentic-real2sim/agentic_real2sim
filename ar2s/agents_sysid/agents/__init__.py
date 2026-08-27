"""v2 agents — high-level LLM-driven orchestration.

Each agent here wraps a ReAct or similar loop, exposes a single callable
entry point, and produces a structured result. They can be embedded as
LangGraph nodes in a top-level pipeline.
"""
