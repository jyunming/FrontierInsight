# Tasks: run Frontier Insight on local-ai-stack (offline LLMs)

Goal: let FI quests run fully offline against **local-ai-stack**
(`C:\dev\local-ai-stack`) — llama.cpp router on `http://localhost:8080/v1`
(OpenAI-compatible, keyless) serving `gemma4-26b` (64K ctx, default),
`qwen3-coder-30b`, `qwen3.6-35b`, `gpt-oss-20b`, etc. Models hot-swap on the
`model` field (~30-90 s first request after a swap). Start it with `las start`;
list models with `las list`.

FI needs **no new provider code**: every chat call already goes through the
OpenAI Chat Completions HTTP transport in `core/provider.py`, and
`provider.base_url` / `provider.model` YAML overrides beat the registry
defaults. The tasks below are config, guard-rails, and nice-to-haves.

## P0 — make it work (config only)

- [ ] **Add a local quest recipe** `examples/local_stack/config.yaml` (or a
  root-level `demo_local_stack.yaml` like the existing `demo_*.yaml`):
  ```yaml
  provider:
    name: openai                # plain OpenAI-compat transport
    base_url: http://localhost:8080/v1
    model: gemma4-26b           # or qwen3-coder-30b for implement-heavy quests
    api_key_env: ""             # keyless — llama-server ignores auth
    node_model_fallbacks: {}    # CRITICAL: default escalates implement/write
                                # to claude-opus-4-7 (core/config.py:182-187)
  ```
  Optionally use `node_models` to route `implement → qwen3-coder-30b` and
  `write → gemma4-26b` — the router hot-swaps per request, but each swap costs
  ~30-90 s, so keep per-node model changes coarse (phase-level, not per-call).
- [ ] **Raise the HTTP timeout for local generation**: `LLMClient` uses a
  ~120 s httpx default (`core/provider.py:1559`); local TG is ~10-20 tok/s, so
  long `write` outputs will exceed it. Set the provider `timeout_s` (or add a
  YAML knob if it isn't exposed) to ≥ 900 for local endpoints.
- [ ] **Trim context for 32-64K models**: cloud prompts here can reach
  hundreds of KB. For local quests set `knowledge.top_k` and literature
  excerpt sizes (`literature_excerpt_chars`, see `core/provider.py:817`)
  so prompts stay well under gemma4-26b's 64K / qwen3.6-35b's 32K tokens.
- [ ] **Smoke test**: `las start`, then run
  `dev/scripts/verify_provider_integration.py` against the local base_url; then
  a small end-to-end quest with `evidence_gate` on to confirm the lenient-JSON
  parsing (`_parse_json_lenient`) holds up on local model output.

## P1 — quality of life

- [ ] **Model discovery for the interview picker**: local-ai-stack serves
  `GET /v1/models` (all registry models) but not Ollama's `/api/tags`
  (`core/provider_models_discover.py:95-122` only probes `/api/tags` for
  local). Add a `/v1/models` probe for any custom `base_url` so `@fi`'s
  dropdown lists gemma4/qwen3/gpt-oss instead of the curated fallback.
- [ ] **Pricing rows**: add `gemma4-26b`, `qwen3-coder-30b`, `qwen3.6-35b`,
  `gpt-oss-20b` at $0 to `MODEL_PRICING` (`core/provider.py:1488-1512`) so
  cost logs read $0.00 instead of `None`.
- [ ] **Interview provider probe**: `core/interview.py:931-939` detects
  providers via `OPENAI_API_KEY` / binaries; add a cheap
  `GET http://localhost:8080/health` probe so "local stack" shows as available
  when the engine is up (or when `las status` would say UP).
- [ ] **Docs**: add a "Local stack (llama.cpp router)" row to
  `docs/PROVIDERS.md` (next to the ollama/vllm rows, :51-52) and a recipe in
  `docs/recipes.md` (:89-96 pattern) — note the keyless auth, hot-swap latency,
  and the `node_model_fallbacks: {}` requirement.

## P2 — offline research stack (separate from FI code)

- [ ] **Axon embeddings/LLM → local**: FI delegates RAG to Axon; wire the
  Axon sidecar config (pattern: `examples/integrator_bakeoff/config.yaml`
  `axon_config:`) to a local embedding model, and its `llm.provider` to
  `http://localhost:8080/v1`. Do NOT add embedding services to FI itself
  (explicitly out of scope per `docs/plan.md:186-192`).
- [ ] **Ensemble on local models**: `node_ensemble` fan-out
  (`core/engine.py:4780`, `core/ensemble.py`) assumes parallel model calls;
  with `--models-max 1` the router serializes them through swaps. Either keep
  ensembles off for local quests, or eval a same-model self-consistency
  ensemble (N samples from one loaded model — no swap cost).

## Notes / gotchas

- Local stack is **keyless**: FI already skips the Bearer header when no key
  (`core/provider.py:1742`) — `api_key_env: ""` is enough.
- Only the OpenAI dialect is needed; local-ai-stack's Anthropic (`:8080
  /v1/messages`) and Gemini (`:4000`) endpoints have no FI consumer (Anthropic
  and Gemini are reached via CLI transports, which keep working unchanged).
- Gemma 4 / GPT-OSS are reasoning models: replies can carry a
  `reasoning_content` field and burn thinking tokens; FI reads
  `choices[0].message.content` (`core/provider.py:1814`) which is correct —
  just budget `max_tokens` generously on JSON-emitting nodes.
- Tests keep mocking `LLMClient.chat` — nothing here changes the test policy.
