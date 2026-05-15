# 10 - Agent Framework Patterns Audit

**Date:** 2026-05-15
**Scope:** Comparative survey of AutoGen v0.4, CrewAI (Crews + Flows),
OpenAI Agents SDK, and the `langchain-ai/open_deep_research` LangGraph
reference app, benchmarked against FrontierInsight's current
`core/engine.py:332-401` graph topology.
**Companion audits:** see siblings in `outputs/audit_2026-05-15/`.

FrontierInsight uses a sliver of what LangGraph and peer frameworks
now offer: a hand-authored `StateGraph` with ~16 nodes
(`engine.py:334-401`), a flat `QuestState` (`engine.py:63-124`),
`AsyncSqliteSaver` checkpointing (`engine.py:33`), four conditional
routers (`engine.py:403-466`), and `interrupt()`-based pause in
clarify and wait_for_data (`engine.py:524`, `978`). The review panel
(`engine.py:1488-1517`) fans personas out with bare `asyncio.gather`,
the provider layer (`core/provider.py`, 13 backends) has no
token/cost accounting, and there is no public extension hook for
custom nodes — `launch.py` and `web/server.py` are the only known
out-of-tree callers and they use `Engine.run()`; direct `_node_*`
access is concentrated in the test suite.

This audit reads four leading frameworks against eight dimensions and
proposes adopt / partial / reject decisions.

---

## Findings

### Framework 1: Microsoft AutoGen v0.4 (+ Magentic-One)

AutoGen v0.4 ([launch blog](https://devblogs.microsoft.com/autogen/autogen-reimagined-launching-autogen-0-4/),
early 2025, still active in 2026 alongside the "Microsoft Agent
Framework" unification with Semantic Kernel) is a ground-up rewrite
around an **async event-driven actor model**. The three-layer stack —
`autogen-core` (event bus), `autogen-agentchat` (teams),
`autogen-ext` (providers + tools) — is the cleanest separation among
the four.

1. **Orchestration.** Agent is the unit, *teams* the composition:
   `RoundRobinGroupChat`, `SelectorGroupChat` (LLM picks next speaker),
   `MagenticOneGroupChat` (planner agent maintains Task Ledger +
   Progress Ledger, re-routes after every step). FI's DAG is more
   static — edges hand-coded at `engine.py:356-401`; AutoGen routes
   dynamically.
2. **State.** Per-agent `ChatHistory`/`Memory`; shared state lives in
   the message stream, not a typed dict. No `QuestState` analog —
   context is reconstituted from history each turn. More flexible but
   far less inspectable.
3. **Retry / repair.** No first-class abstraction. Magentic-One's
   Progress Ledger is closest (re-plan after subtask, possibly
   reassign). FI's `_node_execute_reflect` (`engine.py:1207-1308`) is
   a tighter bounded loop with explicit iteration counter — strictly
   better for code-repair.
4. **Pause/resume.** Weak. `save_state()` / `load_state()` return a
   dict you serialize yourself; no checkpoint-per-step. No native
   `--resume <id>`.
5. **Persona / multi-reviewer.** `SelectorGroupChat` runs N speakers
   *serially*. Parallel fan-out is DIY — same pattern FI uses at
   `engine.py:1513`.
6. **Provider abstraction.** Uniform `ChatCompletionClient` covering
   OpenAI, Azure, Anthropic, Ollama, plus a `LangChainChatCompletionClient`.
   `model_info` carries function-calling / vision capability flags.
   Cleaner interface than FI's `PROXY_PROVIDERS` /
   `CLI_PROVIDERS` split (`provider.py:103`, `194`) but AutoGen lacks
   the agentic-CLI category FI uniquely supports.
7. **Cost telemetry.** Native OpenTelemetry
   ([docs](https://microsoft.github.io/autogen/stable//user-guide/core-user-guide/framework/telemetry.html)).
   `gen_ai.usage.input_tokens` / `output_tokens` emitted as OTEL spans
   ready for Langfuse / SigNoz / Arize. FI has **zero** cost
   telemetry — grep `core/provider.py` for `tokens|cost|usage` returns
   only `max_tokens` plumbing.
8. **Devex.** Hello-world ~6 LOC, all async:
   ```python
   model_client = OpenAIChatCompletionClient(model="gpt-4.1")
   agent = AssistantAgent("assistant", model_client=model_client)
   print(await agent.run(task="Say 'Hello World!'"))
   ```

### Framework 2: CrewAI (Crews + Flows)

CrewAI ([docs](https://docs.crewai.com/en/introduction)) is the most
opinionated: agents are roleplayers, crews are teams, flows are
workflows. Two layers — Crews (high-level, LLM-routed) and Flows
(decorator-driven event graph). Markets 12M+ executions/day.

1. **Orchestration.** *Crew* = `agents + tasks + process`,
   `process ∈ {sequential, hierarchical, consensual}`. Agent schema
   `Agent(role=, goal=, backstory=, tools=[...])` is strictly richer
   than FI's flat-text persona prefix at `engine.py:1493`.
2. **State.** Flows carry a Pydantic `state` model; Crews pass a
   `context` dict between tasks. Both simpler than FI's 26-field
   `QuestState`. Flow decorators (`@start`, `@listen`, `@router`,
   `@or_`, `@and_`) read more declarative than FI's
   `_route_after_design` (`engine.py:403`).
3. **Retry / repair.** `max_iter` and `max_retry_limit` on tool calls;
   on error the LLM gets the trace and retries silently. FI's
   `exec_reflect_history` tracking (`engine.py:1297-1300`) is more
   sophisticated and bounded.
4. **Pause/resume.** `@persist()` for Flow state +
   `restore_from_state_id` / `from_checkpoint`
   ([docs](https://docs.crewai.com/en/guides/flows/mastering-flow-state)).
   The two resume systems are incompatible. Diagrid's analysis
   ([blog](https://www.diagrid.io/blog/checkpoints-are-not-durable-execution-why-langgraph-crewai-google-adk-and-others-fall-short-for-production-agent-workflows))
   notes neither LangGraph nor CrewAI checkpoints are durable in the
   Temporal sense — a hard kill mid-LLM-call loses the call. FI
   inherits this limitation.
5. **Persona / multi-reviewer.** The role/goal/backstory schema *is*
   personas. Crew default is sequential; Flows can parallelize via
   `@listen(or_(...))`. FI's `asyncio.gather` at `engine.py:1513` is
   simpler and already concurrent.
6. **Provider.** Native SDKs (OpenAI, Anthropic, Gemini, Azure,
   Bedrock) + LiteLLM fallback for ~100 more. Cleaner than FI's
   proxy/CLI/HTTP split and LiteLLM gives cost-tracking and
   load-balancing free. Misses FI's agentic-CLI category.
7. **Cost telemetry.** AMP/Tracing tier shows tokens and latency per
   step; baseline logs tokens when `verbose=True`.
8. **Devex.** Hello-world ~15-20 LOC: `Agent` → `Task` → `Crew(...).kickoff()`.
   Most approachable for teaching, but the backstory field is theater
   when applied to research reviewers.

### Framework 3: OpenAI Agents SDK

The OpenAI Agents SDK ([docs](https://openai.github.io/openai-agents-python/),
March 2025 release, April 2026 harness upgrade) is the production
successor to Swarm. Keeps the **handoff** primitive, adds guardrails,
tracing, and a Codex-style harness.

1. **Orchestration.** Agents declare `handoffs=[...]` to other agents.
   The model chooses the handoff target as if it were a tool call. No
   graph — control flows dynamically. Inverse of FI's static-edges
   approach.
2. **State.** A `RunContext` is threaded through; conversation history
   is canonical state. No typed dict.
3. **Retry / repair.** Tool errors go back to the model with traceback;
   the model retries until it succeeds or hits the Runner's call cap.
   FI's `exec_reflect_max_iterations` (`engine.py:1229`) is safer
   against runaway loops.
4. **Pause/resume.** April 2026 harness upgrade introduced resume
   bookkeeping for long-running agents
   ([summary](https://qubittool.com/blog/ai-agent-framework-comparison-2026));
   real durability is unclear in practice. LangGraph's AsyncSqliteSaver
   is still the most battle-tested approach.
5. **Persona / multi-reviewer.** Handoff is sequential. No fan-out
   primitive equivalent to `Send`. Parallel personas would mean three
   separate `Runner.run(...)` calls + DIY aggregation — essentially
   what FI already does at `engine.py:1513`.
6. **Provider.** OpenAI-first; the `LitellmModel` class bridges to
   ~100 providers
   ([tutorial](https://docs.litellm.ai/docs/projects/openai-agents)).
   No agentic-CLI support.
7. **Cost telemetry.** **Best-in-class.**
   `Runner.run(...).context_wrapper.usage` aggregates tokens across
   handoffs and tool calls
   ([Usage docs](https://openai.github.io/openai-agents-python/usage/));
   `request_usage_entries` exposes per-request detail; the Tracing UI
   visualizes the call tree free. The single biggest gap relative to FI.
8. **Devex.** Hello-world ~3 LOC sync (`Runner.run_sync(agent, "...")`);
   handoff ~10 LOC. Lowest ceremony — at the cost of being implicit
   when workflows grow past triage + 2-specialist patterns.

### Framework 4: LangGraph reference app — `langchain-ai/open_deep_research`

[`open_deep_research`](https://github.com/langchain-ai/open_deep_research)
([blog](https://www.langchain.com/blog/open-deep-research)) is
LangChain's flagship LangGraph showcase, specifically a "deep research"
agent — closer to FI's workload than the multi-agent SDKs.

1. **Orchestration.** Two-tier supervisor: a supervisor agent breaks
   the brief into sub-topics, then spawns researcher subagents *in
   parallel* via `Send`. Researchers report back, supervisor
   synthesizes. FI has a fixed pipeline; this has a fixed coordination
   pattern with dynamic fan-out width.
2. **State.** `TypedDict` with `Annotated[list, operator.add]`
   reducers on every field parallel branches write to. FI's
   `review_panel` at `engine.py:103` is a plain `list[dict]` written
   atomically at `engine.py:1546` because the gather is *outside* the
   graph. If reviews ever become real graph nodes, this field needs an
   `operator.add` reducer.
3. **Retry / repair.** Researcher subagents have their own bounded
   tool loops; supervisor decides whether to re-spawn a thin
   researcher — comparable to FI's `execute_reflect` at a higher
   level.
4. **Pause/resume.** Same `AsyncSqliteSaver`. Adds a `langgraph dev`
   server with thread/history UI — FI rolls its own `quest_id` thread
   keying at `engine.py:144` and lives without the studio.
5. **Persona / multi-reviewer.** Parallel-researcher *is* the
   multi-persona pattern. Crucially, it's *checkpointed per persona*:
   a half-finished panel resumes from the crashed persona. FI's
   atomic `asyncio.gather` re-runs *all* personas on resume.
6. **Provider.** `init_chat_model(...)` covers ~10 providers
   (Anthropic, OpenAI, Bedrock, Vertex). FI's `ResolvedEndpoint`
   (`provider.py:204`) is wider in *kind* (proxy + CLI + HTTP).
7. **Cost telemetry.** Inherits LangSmith tracing when configured
   (`LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true`). FI does not
   import any langchain tracing context.
8. **Devex.** Clone + `uvx` + configure. The `langgraph dev` command
   is a DX win FI hasn't exploited.

### Comparison Table

| Dimension              | AutoGen v0.4              | CrewAI Flows         | OpenAI Agents SDK    | open_deep_research    | **FrontierInsight**            |
|------------------------|---------------------------|----------------------|----------------------|-----------------------|--------------------------------|
| Orchestration          | Async actor + teams       | Crews + Flow DAG     | Handoff chain        | Supervisor + `Send` fan-out | Hand-authored StateGraph (`engine.py:334`) |
| State                  | Per-agent memory + msgs   | Pydantic Flow state  | RunContext dict      | TypedDict + reducers  | TypedDict, no reducers (`engine.py:63`) |
| Retry / repair         | Magentic Progress Ledger  | `max_retry_limit`    | Model-driven retry   | Per-subagent bounded  | Bounded `execute_reflect` (`engine.py:1207`) |
| Pause/resume           | `save_state()` (manual)   | `@persist` + checkpoint | Harness resume (new) | AsyncSqliteSaver      | AsyncSqliteSaver (`engine.py:33`) |
| Parallel persona       | Sequential SelectorGroup  | Sequential by default | Sequential handoffs | `Send` fan-out         | `asyncio.gather` (`engine.py:1513`) |
| Provider               | Uniform `ChatCompletionClient` | LiteLLM + native | LiteLLM bridge       | `init_chat_model`     | 13-provider proxy/CLI/HTTP (`provider.py`) |
| Cost telemetry         | OTEL native                | AMP tracing tier     | Built-in `usage`     | LangSmith opt-in      | **None** (grep confirms)       |
| Hello-world LOC        | ~6                        | ~15-20               | ~3                   | clone+config          | N/A (no SDK surface)           |

---

## Recommendations

Numbered, impact/effort tagged. Each recommendation says *what to
borrow*, *which framework it comes from*, and *which FI file:line it
modifies*.

### 1. Add token + cost telemetry — at the provider layer, not in nodes  [impact: HIGH] [effort: M]

**Borrow from:** OpenAI Agents SDK `RunContext.usage` aggregation +
AutoGen's OTEL `gen_ai.usage.*` conventions.

**Modify:** `core/provider.py` — extend `LLMClient.chat(...)` (at
`core/provider.py:759` and the proxy/CLI/HTTP transport
implementations) to return a `(text, usage)` tuple where
`usage = {"prompt_tokens": int, "completion_tokens": int,
"model": str, "node": str, "duration_s": float, "usd_cost": float}`.
Aggregate per-quest into `Engine._usage: list[Usage]` and serialize to
`<quest_root>/.fi/usage.json` at every checkpoint.

Why now: zero of the 13 providers in `core/provider.py:194` report
token counts back to the engine, so a quest that costs $4.20 looks
identical in logs to a quest that costs $42. This is the single
biggest "professionalism gap" relative to the other frameworks. The
USD multiplication can live in a tiny `PROVIDER_PRICING` dict —
LiteLLM already maintains one we can import (`litellm.model_cost`).

The proxy providers (claude_code, copilot-cli, codex-cli) won't return
tokens directly — these need a counter at the wire level
(`tiktoken`-style estimate from prompt + response). For the HTTP/SDK
providers (Anthropic, OpenAI, Gemini), real token counts come back in
the response. Document the proxy-provider counts as estimates with a
`±15%` caveat in `<quest_root>/.fi/usage.json`.

### 2. Convert `review_panel` from `asyncio.gather` to LangGraph `Send`  [impact: MEDIUM] [effort: M]

**Borrow from:** `open_deep_research`'s supervisor + parallel
researcher pattern using `Send` and `operator.add` reducers.

**Modify:**
- `core/engine.py:103` — change
  `review_panel: list[dict[str, Any]]` to
  `review_panel: Annotated[list[dict[str, Any]], operator.add]`.
- `core/engine.py:334` — replace the single `g.add_node("review", ...)`
  with `g.add_node("review_persona", ...)` for one persona at a time,
  plus `g.add_node("review_moderate", ...)` for the aggregator.
- `core/engine.py:393` — wire a conditional edge from `write` that
  emits `Send("review_persona", {**state, "persona": name})` for each
  panel member in `engine.review_panel`.
- `core/engine.py:1488-1517` — delete the `asyncio.gather` block;
  each persona becomes its own checkpointed node invocation.

Why bother: a panel of 5 personas that's half-finished when the user
Ctrl-Cs currently re-runs *all 5* on resume because the gather is
atomic. With `Send`, completed personas are checkpointed individually
and only unfinished ones re-execute. For a $0.50/persona-call panel,
this is real money saved on every interrupted run.

Cost of the change: panel results land in state via the reducer in
arrival order. `_aggregate_panel_reviews` (`core/engine.py:1958`)
preserves input order today for weaknesses, suggestions, and the
first blocking note (it does NOT sort by persona before aggregation),
so arrival-order writes from the reducer would make those fields
nondeterministic. The fix is to either (a) sort the reducer's
combined list by persona name before passing it to the aggregator,
or (b) teach `_aggregate_panel_reviews` to sort before aggregating.
Either resolves the determinism issue at small cost. Snapshot tests
that diff `review_panel[0]` still need updating.

### 3. Adopt CrewAI's role/goal/backstory schema for panel personas  [impact: LOW] [effort: S]

**Borrow from:** CrewAI's `Agent(role=, goal=, backstory=, tools=[...])`
declarative schema.

**Modify:** `core/engine.py:1493`'s `_load_persona_prefix(name)` (and
the persona files it reads from `personas/`) — replace the flat
text-prefix model with a small Pydantic-typed YAML:
```yaml
# personas/methodologist.yaml
role: "Methodologist reviewer"
goal: "Detect statistical and experimental-design flaws before publication"
backstory: "PhD-level statistician with 15 years in causal inference reviewing"
tools: []  # reviewers don't need tools in current FI
review_focus: ["sample_size", "confounders", "preregistration", "p-hacking"]
```
The structured `review_focus` field is the real win — it lets the
moderator prompt at `core/engine.py:1523` cite *which axes* each
persona was meant to grade on, which would improve the rationale
quality. Keep the existing flat-text persona files as a fallback for
back-compat (`_load_persona_prefix` returns string today; have a
`_load_persona(name) -> Persona` wrapper that builds the prompt from
the structured fields).

Partial adopt — the "backstory" field is theater for an LLM and FI
shouldn't pretend otherwise. The role + goal + review_focus triple is
the load-bearing structure.

### 4. Publish a minimal `frontier_insight.sdk` surface  [impact: HIGH] [effort: L]

**Borrow from:** OpenAI Agents SDK's "tiny stable surface" philosophy
(`Agent`, `Runner`, `handoff`) and AutoGen's three-layer separation.

**Modify:** Create `core/sdk.py` (~150 LOC) that exposes:
- `Quest(topic: str, config: Config) -> Quest` — wraps `Engine`
- `Quest.run() -> QuestArtifacts` — calls `Engine.run` under the hood
- `Node` — protocol for custom node callables `(state: QuestState) -> QuestState`
- `register_node(name: str, fn: Node)` — extension hook
- `register_provider(name: str, transport: TransportSpec)` — currently
  there is no registration hook; `PROXY_PROVIDERS` / `CLI_PROVIDERS`
  in `core/provider.py` are already public constants (the
  underscored names remain as back-compat aliases), but adding a new
  provider still requires editing core code rather than registering
  a transport from outside.

The engine is currently *not* a library in the SDK sense — `launch.py`
and `web/server.py` use `Engine.run()` as the public entry point
(good), but there is no extension point for writing custom node
variants without forking `core/engine.py`. Direct `_node_*` access
happens primarily in the test suite and inside `Engine._build_graph`
itself. A real SDK would let users register their own nodes the way
AutoGen v0.4 does via its extensions layer; FI's 2,685-LOC
`core/engine.py:1` owns the entire DAG today.

This is a 2-3 week project, not a one-PR change. It unblocks the
"users write their own engine variants" story that the README
implicitly promises but the codebase blocks.

### 5. Add `langgraph dev` integration / studio UI support  [impact: LOW] [effort: S]

**Borrow from:** `open_deep_research`'s `langgraph dev` command.

**Modify:** Add `langgraph.json` at repo root pointing to
`core.engine:build_graph` (need to expose that as a top-level callable
— currently the graph builder is `Engine._build_graph` at
`core/engine.py:332`, a method).

Why: the LangGraph Studio UI gives free graph visualization, step
debugging, and state inspection. FI users currently debug by tailing
`<quest_root>/.fi/run.log` and `printf`-debugging — Studio gets us
visual graph debugging for ~1 PR of effort.

Caveat: this requires `build_graph` to be a free function taking a
`Config`. Doing the extraction cleanly is also a prerequisite for #4.

### 6. Reject: do NOT switch to AutoGen GroupChat or OpenAI handoffs  [impact: N/A] [effort: N/A]

For completeness: dynamic LLM-routed orchestration (AutoGen's
`SelectorGroupChat`, OpenAI's `handoffs`) is *worse* for FI's
workload than the current static StateGraph. The whole point of FI is
that the pipeline `clarify → ideate → literature → design → implement
→ execute → execute_reflect → analyze → cross_check → write → review`
is the same for every research quest. Letting an LLM pick the next
node would (a) burn tokens on routing decisions that are already
trivially correct, (b) break the user's mental model of the pipeline,
and (c) make resume semantics unpredictable.

Static graphs are the right answer when the workflow is known. Reject
dynamic routing for the main pipeline. Use it (per #2) only inside the
review panel where parallel fan-out is genuinely needed.

### 7. Reject: do NOT adopt Magentic-One's Progress Ledger  [impact: N/A] [effort: N/A]

The Progress Ledger is a great pattern for *open-ended task agents*
(web browsing, file shuffling). FI is not an open-ended task agent —
it has a fixed 10-phase pipeline. Bolting a Progress Ledger onto a
StateGraph would be paying the cost of LLM-driven re-planning every
node without getting the benefit. Reject.

### 8. Partial: Adopt LiteLLM as a 14th provider  [impact: MEDIUM] [effort: S]

**Borrow from:** CrewAI's "native SDKs first, LiteLLM as fallback"
strategy.

**Modify:** `core/provider.py:194` — add `"litellm"` to
`_CLI_SPECS` (or really to a new `LITELLM_PROVIDERS` set). This is
*one* new provider that subsumes ~100 backends FI doesn't currently
support (xAI Grok, Cohere, NVIDIA NIM, Mistral, every model on
Together / Fireworks / Replicate / OpenRouter). Wiring it is
mechanical — LiteLLM speaks OpenAI-format and FI already has an
OpenAI-compatible transport.

This does *not* replace FI's existing 13 providers; in particular the
agentic-CLI providers (claude-code, copilot-cli, codex-cli, gemini-cli)
remain irreplaceable because LiteLLM doesn't model long-running
agentic CLIs. LiteLLM fills the long tail of "I want to try
DeepSeek-R3 for one node" without us writing a new transport every
quarter.

Bonus: enabling `litellm.cost_per_token` gives us recommendation #1's
pricing dict for free.

---

## Adopt / Partial / Reject summary

| Rec | What                                | Decision  | Effort |
|-----|-------------------------------------|-----------|--------|
| 1   | Token + USD telemetry               | **Adopt** | M      |
| 2   | `Send` for review_panel             | **Adopt** | M      |
| 3   | Structured persona schema           | Partial   | S      |
| 4   | Public SDK surface                  | **Adopt** | L      |
| 5   | `langgraph dev` integration         | Adopt     | S      |
| 6   | Dynamic LLM routing for main DAG    | **Reject** | —     |
| 7   | Magentic Progress Ledger            | **Reject** | —     |
| 8   | LiteLLM as 14th provider            | Adopt     | S      |

**Sequencing.** Do #1 and #8 first — they're small, they buy back real
visibility, and #8 enables #1's pricing dictionary. Do #5 next as a
quick devex win. Defer #2 and #3 to the next "review experience" pass.
Do #4 only when there's an actual external user asking to write their
own node (the YAGNI risk is real otherwise).

---

## References

### Frameworks

- AutoGen v0.4 launch: [devblogs.microsoft.com/autogen/autogen-reimagined-launching-autogen-0-4](https://devblogs.microsoft.com/autogen/autogen-reimagined-launching-autogen-0-4/)
- AutoGen telemetry: [microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/telemetry.html](https://microsoft.github.io/autogen/stable//user-guide/core-user-guide/framework/telemetry.html)
- Magentic-One: [github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one)
- CrewAI docs: [docs.crewai.com/en/introduction](https://docs.crewai.com/en/introduction)
- CrewAI Flow state: [docs.crewai.com/en/guides/flows/mastering-flow-state](https://docs.crewai.com/en/guides/flows/mastering-flow-state)
- OpenAI Agents SDK: [openai.github.io/openai-agents-python](https://openai.github.io/openai-agents-python/)
- OpenAI Agents SDK usage tracking: [openai.github.io/openai-agents-python/usage](https://openai.github.io/openai-agents-python/usage/)
- open_deep_research: [github.com/langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)
- LangGraph `Send` API: [docs.langchain.com/oss/python/langgraph/use-graph-api](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- LiteLLM: [github.com/BerriAI/litellm](https://github.com/BerriAI/litellm)
- Diagrid "Checkpoints are not durable execution": [diagrid.io/blog/checkpoints-are-not-durable-execution-...](https://www.diagrid.io/blog/checkpoints-are-not-durable-execution-why-langgraph-crewai-google-adk-and-others-fall-short-for-production-agent-workflows)

### FrontierInsight files referenced

- `core/engine.py:33` — `AsyncSqliteSaver` import
- `core/engine.py:63-124` — `QuestState` TypedDict
- `core/engine.py:103` — `review_panel` field (no reducer)
- `core/engine.py:144` — `quest_id` as LangGraph `thread_id`
- `core/engine.py:332-401` — graph construction + edges
- `core/engine.py:403-466` — `_route_after_*` conditional routers
- `core/engine.py:524`, `978` — `interrupt()` callsites
- `core/engine.py:1207-1308` — `_node_execute_reflect` bounded-repair loop
- `core/engine.py:1437-1560` — `_node_review` with panel fan-out
- `core/engine.py:1488-1517` — `asyncio.gather` for parallel personas
- `core/engine.py:1493` — `_load_persona_prefix` (flat-text persona)
- `core/engine.py:1562-1605` — `_chat` and `_model_for_node` per-node model routing
- `core/engine.py:1958` — `_aggregate_panel_reviews` deterministic voting
- `core/provider.py:103-108` — `PROXY_PROVIDERS` set
- `core/provider.py:194` — `CLI_PROVIDERS` set
- `core/provider.py:204` — `ResolvedEndpoint` dataclass (the provider abstraction; line 217 is its `provider_name` field)
- `core/provider.py:759` — `chat(...)` interface (no usage return)
