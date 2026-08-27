"""Skills (LangChain tools) for the v2 Episode Agent.

A "skill" here is a `@tool`-decorated function — a self-contained unit of
capability the agent can autonomously invoke during its ReAct loop.

Categories:
  fs_inspect.py     — read-only file system inspection (list / sniff / shape)
  visual_inspect.py — image + video keyframe extraction (multimodal)
  manifest_common.py       — shared submission state + limits
  manifest_submit_phystwin.py — phystwin finalization + strict validator
"""
