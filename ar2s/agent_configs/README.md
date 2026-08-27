# Agent Model Config Schema

`config_anthropic_opus47.yaml` is the default model/backend configuration for agentic
visual, sysid, and G1 sysid agents. It is the single source for selecting model
providers and model ids.

Rules:

- `models` is always a list, including single-model agents.
- `type: llm` means a text-capable chat agent. `type: vlm` means an
  image-capable chat agent; some VLM callers use structured Pydantic fanout,
  while others send ordinary multimodal chat messages.
- Model specs use `provider:model`. Supported provider prefixes are listed in
  the YAML `providers` section.
- Bare model specs are reserved for NHR@FAU-compatible model ids and are
  normalized as `nhr_fau:<model>` by the runtime loader.
- `temperature: null` means no temperature kwarg is passed.
- Anthropic construction never receives `temperature`, even when the YAML value
  is numeric, because current Claude models can reject that kwarg.
- OpenAI, Qwen, NHR@FAU, and OpenRouter receive `temperature` only when the
  YAML value is numeric.
- `model_overrides.<model>.min_max_tokens` preserves per-model minimum token
  budgets for verbose structured-output models such as Kimi, Gemma, and
  Mistral.
- `model_overrides.<model>.openrouter_reasoning_effort` sets a global (per
  model id) OpenRouter reasoning effort applied to every agent using that
  model.
- `<agent>.reasoning_effort` is an optional per-agent override whose request
  shape depends on the provider:
  - direct OpenAI: null / absent sends no `reasoning` field; `false` or
    `"none"`/`"off"`/`"false"` sends `reasoning: {effort: "none"}`; any other
    non-empty string sends `reasoning: {effort: <string>}`.
  - OpenRouter: null / absent falls back to the `model_overrides` global for
    the model; `false` or `"none"`/`"off"`/`"false"` disables reasoning for
    this agent with `reasoning: {enabled: false}`; any other non-empty string
    overrides the global with `reasoning: {effort: <string>}`.
  - other providers ignore the field.
- Direct OpenAI mode uses the single env named by
  `providers.openai.credential_env`, which must be either
  `OPENAI_API_AR2S_KEY` or `OPENAI_API_UBC_PHYSAI_AR2S_KEY`; direct OpenRouter mode uses
  `OPENROUTER_API_AR2S_KEY`; NHR gateway mode uses
  `NHR_FAU_GATEWAY_API_KEY`.
- When tracing VLM calls, `AR2S_VLM_TRACE_PATH` captures the raw structured
  response metadata and usage metadata. Direct OpenAI reasoning token counts
  come from response usage metadata, not from visible text.
- API key environment variables are credentials only. They do not select a
  backend or model.

The Anthropic default config now routes Anthropic-backed agent slots through
Claude Opus 4.7. Non-Anthropic fallback models remain listed where the YAML
explicitly includes them.
