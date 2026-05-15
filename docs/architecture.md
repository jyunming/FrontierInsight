# Frontier Insight — Architecture

## Layered design

```
┌────────────────────────────────────────────────────────────────────────┐
│ launch.py                                                              │
│   parse args → Config.from_yaml → Engine(cfg).run() → generators       │
│   --fleet runs N Engines under a bounded asyncio.Semaphore             │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │ core/config.py              │
            │   pydantic schema           │
            │   tilde expansion on paths  │
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────────────────────────┐
            │ core/engine.py — async LangGraph                │
            │                                                 │
            │  clarify → ideate → literature → design         │
            │                                       │         │
            │     simulate path ─────┬──────────────┘         │
            │     no-simulation path ┘                        │
            │                                                 │
            │  ┌─ simulate ──────────────────────────┐        │
            │  │  implement → execute                │        │
            │  │    └─ execute_reflect (retry/proceed)        │
            │  └─────────────────────┬───────────────┘        │
            │                        │                        │
            │  ┌─ no-simulation ─────┘                        │
            │  │  auto_collect_data → wait_for_data           │
            │  │    └─ data_load                              │
            │  └─────────────────────┬──────────────┐         │
            │                        ▼              │         │
            │                     analyze           │         │
            │                        │              │         │
            │                  cross_check ─(redesign)┘       │
            │                        │ (write)               │
            │                     write → review → END        │
            │                              │ (revise)         │
            │                              └── design         │
            │                                                 │
            │  AsyncSqliteSaver (one per quest_root)          │
            └──┬───────────────┬─────────────────────┬────────┘
               │               │                     │
   ┌───────────▼──┐  ┌─────────▼──────┐  ┌──────────▼─────────────────┐
   │ core/        │  │ core/          │  │ core/                      │
   │ provider.py  │  │ execution.py   │  │ knowledge.py + datasets/   │
   │              │  │                │  │                            │
   │ LLMClient    │  │ Venv | Docker  │  │ asearch():                 │
   │ Proxy-       │  │ Executor       │  │  local → Axon →            │
   │ Supervisor   │  │                │  │  router (7 srcs)           │
   │              │  │                │  │ +full-text fetch           │
   │              │  │                │  │ structured write-back      │
   │              │  │                │  │ (5 doc kinds)              │
   │              │  │                │  │                            │
   │              │  │                │  │ datasets/ adapters         │
   │              │  │                │  │  ├─ worldbank.py           │
   │              │  │                │  │  │   (country-indicator    │
   │              │  │                │  │  │    tables, 5-year span) │
   │              │  │                │  │  └─ wikipedia.py           │
   │              │  │                │  │      (opensearch +         │
   │              │  │                │  │       page summaries)      │
   │              │  │                │  │ → auto_collected/<adapter>/│
   └───────┬──────┘  └────────────────┘  └────────────────────────────┘
           │                                          ▲
           │ OpenAI-shaped Chat Completions           │
           ▼                                          │
   ┌──────────────────────────────────────┐           │
   │ Transports                           │           │
   │ ├─ HTTP-direct                       │           │
   │ │   ├─ openai / codex                │           │
   │ │   ├─ gemini                        │           │
   │ │   └─ ollama / vllm                 │           │
   │ ├─ HTTP-via-proxy                    │           │
   │ │   ├─ claude_code (poetry-spawned)  │           │
   │ │   └─ github_copilot_* (npx-spawned)│           │
   │ ├─ CLI exec (one subprocess per call)│           │
   │ │   ├─ claude_cli                    │           │
   │ │   ├─ codex_cli                     │           │
   │ │   ├─ copilot_cli (agentic — warn)  │           │
   │ │   └─ gemini_cli                    │           │
   │ └─ VSCode bridge                     │           │
   │     └─ vscode_extension              │           │
   │         (TCP → vscode.lm.* in the    │           │
   │          chat-participant host)      │           │
   └──────────────────────────────────────┘           │
                                                      │
   ┌────────────────────────────┐                     │
   │ generation/                │─────────────────────┘
   │ ├─ paper.py (pandoc+TeX)   │   consume QuestArtifacts
   │ ├─ slides.py (Marp)        │
   │ ├─ poster.py (beamer)      │
   │ └─ speech.py (LLM-only)    │
   └────────────────────────────┘
```

Conditional edges visible in the graph:

| Edge | Trigger | Bound by |
|---|---|---|
| `execute_reflect → execute` | Script crashed (rc != 0 or no `RESULT_JSON`) | `engine.exec_reflect_max_iterations` (default 3) |
| `cross_check → design` | `analyze.next_step ∈ {re_experiment, broaden_lit}` | `engine.max_iterations` (default 2) |
| `review → design` | `review.verdict == "revise"` | `engine.max_iterations` (default 2) |
| `design → auto_collect_data` | `state.no_simulation_resolved is True` (set at clarify or by YAML) | one-shot — the no-simulation branch never loops back into design |

## Key contracts

### `Config` (YAML → pydantic)

Top-level fields: `topic`, `title`, `provider`, `engine`, `execution`,
`knowledge`, `output`, `extra_directives`. See `core/config.py`.

### `QuestState` (TypedDict)

Defined at `core/engine.py` (just under the `Engine` import block).
`total=False` so every field is optional — nodes patch in what they
produce, LangGraph merges the patches.

Core path (every quest produces these):
`topic`, `title`, `iteration`, `ideas`, `chosen_idea`, `literature`,
`design`, `code`, `deps`, `exec_result`, `figures`, `result_json`,
`analysis`, `paper_md`, `review`.

Feature-specific extras:
- Clarify: `clarify_questions`, `clarify_answers`, `clarify_done`.
- Execute-repair loop: `exec_reflect_iter`, `exec_reflect_history`, `exec_give_up_reason`.
- Analyze re-route + cross-paper check: `cross_check` (per-finding classification list).
- Ideate self-reflection: `ideate_critique`.
- Reviewer panel: `review_panel` (per-persona reviews + moderator synthesis).
- No-simulation routing: `no_simulation_resolved` (bool, decided at clarify or YAML). When True, the engine skips `implement → execute → execute_reflect` and routes through `auto_collect_data → wait_for_data → data_load → analyze`.
- Auto-collect bookkeeping: `auto_collected_count` (int — total files written into `data/auto_collected/` by Axon plus all enabled `engine.dataset_adapters`). Per-adapter writes land under `data/auto_collected/<adapter>/`.

JSON-serializable so `AsyncSqliteSaver` can checkpoint after every
node. **Field names are the contract** between the engine, the
prompts, and any alternate graph subclass — keep them
backwards-compatible.

### `QuestArtifacts`

Returned by `Engine.run()`, consumed by every generator:

| Field | Meaning |
|---|---|
| `quest_id` | Timestamp-prefixed unique ID. |
| `quest_root` | `<output_dir>/<quest_id>/` — owns `paper/`, `figures/`, `code/`, `.fi/`. |
| `paper_md` | Path to `paper/paper.md`, or `None` if the write node skipped. |
| `paper_pdf` | Reserved for the generator stage (engine never compiles PDF). |
| `figures_dir` | `<quest_root>/figures/` if it has at least one PNG/SVG/PDF. |
| `bundle_manifest` | Phase-1 leftover (Axon-driven manifest in the future). |
| `raw_state` | Full final QuestState for debugging. |

### `Executor` protocol

`async setup(quest_root)`, `async install(packages, quest_root)`,
`async execute(cmd, cwd, timeout_s, env)`, `python_path(quest_root)`.
Both `VenvExecutor` and `DockerExecutor` satisfy it. `make_executor`
in `core/execution.py` is the constructor.

### Generator protocol

Each generator exposes a `generate(art, out_dir, ...) -> dict[str, Path]`
returning the files it wrote. `PaperGenerator.generate(art, out_dir)` is
sync (pandoc shell-out only). `SlideGenerator`, `PosterGenerator`, and
`SpeechGenerator` are `async` and additionally accept
`*, supervisor: ProxySupervisor | None = None` because they make LLM
calls and may need the proxy supervisor to resolve the endpoint.
Generators run sequentially after the engine; one failure does not
abort the rest (see `_run_generators` in `launch.py`).

## Provider abstraction

`core.provider.resolve_endpoint_async(provider, supervisor)` returns a
`ResolvedEndpoint` carrying `base_url`, `model`, `api_key`, `transport`
(`"http"` / `"cli"` / `"vscode_bridge"`), and transport-specific
fields (`cli_spec`, `cli_model_override`, `vscode_bridge_port`, etc.).
HTTP-direct providers point straight at a `base_url`; CLI-exec
providers spawn the local CLI per call; proxy providers go through
`ProxySupervisor`, which spawns the child process on a free port and
reference-counts so concurrent quests share one proxy. The
`vscode_extension` transport routes calls through the FI VSCode
extension's `vscode.lm.*` bridge. See [`plan.md`](plan.md) for the
shipped-capability list and [`INSTALL.md`](INSTALL.md) for transport
prerequisites.

## Concurrency model

- One `Engine` per quest. Engines are stateless w.r.t. process globals.
- `httpx.AsyncClient` per `LLMClient`. (Future: a shared pool across
  Engines if profiling justifies it.)
- `ProxySupervisor` is shared across Engines in a process; ref-counts
  proxy subprocesses.
- **Fleet runner** (`launch.py:run_fleet`): schedules N Engines under
  an `asyncio.Semaphore(max_concurrent)`. The cap defaults to
  `min(4, os.cpu_count())` and is overridable via `--max-concurrent`.
  Quests run as independent asyncio tasks; an exception in one quest
  doesn't cascade (each task's `try/except` is local). The shared
  `ProxySupervisor` means 4 concurrent quests using the same provider
  share one proxy subprocess (one bridge port, one HTTP client) —
  ref-counting releases the subprocess only when the last quest
  using it finishes. See [`launch.py`](../launch.py) `_run_one_quest`
  + `run_fleet` for the exact scheduling code.
- Per-quest filesystem isolation: `<quest_root>/{paper,figures,code,.fi}`,
  plus `<quest_root>/config.yaml` (a copy of the source YAML so
  `--resume` / `@fi /resume` need nothing else from the filesystem).
- Checkpoints: `<quest_root>/.fi/state.sqlite`. A killed quest can be
  resumed by re-running with `python launch.py --config <yaml>
  --resume <quest_id>` or via `@fi /resume` in VSCode. The engine
  detects existing state via `graph.aget_state()` and passes `None`
  to `ainvoke` so LangGraph continues from the last completed node
  instead of replaying from `ideate`.
- Memory ceiling: `--memory-cap-mb` on `launch.py --fleet` defers new
  starts when RSS approaches the cap (psutil-based).

## Dataset adapters

The no-simulation `auto_collect_data` node calls
`Engine._run_dataset_adapters` (`core/engine.py:~1055`) which
walks the `engine.dataset_adapters` list, dispatches each name to
the matching module under `core/datasets/` (`worldbank.py`,
`wikipedia.py`), and writes the returned `DatasetRow`s into
`<quest_root>/data/auto_collected/<adapter>/`. Adapters share a
small `DatasetAdapter` protocol (one `fetch(query, top_k)`
coroutine returning a list of rows). Failures inside an adapter
log a WARNING and return an empty list — the parent node sums the
counts across adapters and keeps the quest moving. Adding a new
adapter is a single-file change in `core/datasets/` plus an entry
in the registry; no engine-graph edit needed.

## Why we own the loop

The Phase-0 prototype wrapped DeepScientist (DS), a Linux-only engine
that requires WSL2 on Windows and depends on `bash_exec` with no
documented Git-Bash fallback. With the redesign, FI is Windows-native
and Linux-native, runs entirely in Python, and benefits from the full
LangChain/LangGraph ecosystem (specifically Axon's
`langchain.AxonRetriever`, which would not be reachable from a Rust
rewrite without a separate REST/MCP hop). The DS wrapper code has been
removed.
