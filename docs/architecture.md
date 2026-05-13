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
            ┌──────────────▼──────────────┐
            │ core/engine.py              │
            │   async LangGraph           │
            │   ideate ▸ literature ▸     │
            │   design ▸ implement ▸      │
            │   execute ▸ analyze ▸       │
            │   write ▸ review            │
            │   conditional revise edge   │
            │   AsyncSqliteSaver          │
            └──┬──────────┬──────────┬────┘
               │          │          │
   ┌───────────▼──┐  ┌────▼─────┐  ┌─▼─────────────────┐
   │ core/        │  │ core/    │  │ core/             │
   │ provider.py  │  │ execution│  │ knowledge.py      │
   │              │  │ .py      │  │                   │
   │ LLMClient    │  │ Venv|    │  │ asearch():        │
   │ Proxy-       │  │ Docker   │  │  local → Axon →   │
   │ Supervisor   │  │ Executor │  │  router (7 srcs)  │
   │              │  │          │  │ +full-text fetch  │
   │              │  │          │  │ structured write- │
   │              │  │          │  │ back (5 doc kinds)│
   └───────┬──────┘  └──────────┘  └───────────────────┘
           │                           ▲
           │ OpenAI Chat Completions   │
           ▼                           │
   ┌───────────────────────┐           │
   │ Backends              │           │
   │ ├─ codex / openai     │           │
   │ ├─ gemini             │           │
   │ ├─ ollama / vllm      │           │
   │ ├─ claude_code via    │           │
   │ │  poetry-spawned     │           │
   │ │  proxy              │           │
   │ └─ github_copilot via │           │
   │    npx-spawned proxy  │           │
   └───────────────────────┘           │
                                       │
   ┌────────────────────────┐          │
   │ generation/            │──────────┘
   │ ├─ paper.py (pandoc+TeX)│   consume QuestArtifacts
   │ ├─ slides.py (Marp)    │
   │ ├─ poster.py (beamer)  │
   │ └─ speech.py (LLM-only)│
   └────────────────────────┘
```

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

Phase-specific extras:
- Phase I (clarify): `clarify_questions`, `clarify_answers`, `clarify_done`.
- Phase K (execute-repair loop): `exec_reflect_iter`, `exec_reflect_history`, `exec_give_up_reason`.
- Phase L (analyze re-route + cross-paper check): `cross_check` (per-finding classification list).
- Phase M (ideate self-reflection): `ideate_critique`.
- Phase N (reviewer panel): `review_panel` (per-persona reviews + moderator synthesis).

JSON-serializable so `AsyncSqliteSaver` can checkpoint after every
node. **Field names are the contract** between the engine, the
prompts, and any future Phase-G alternate graph — keep them
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
extension's `vscode.lm.*` bridge. See `docs/plan.md` Phase C and
Phase P for transport-specific spawn details.

## Concurrency model

- One `Engine` per quest. Engines are stateless w.r.t. process globals.
- `httpx.AsyncClient` per `LLMClient`. (Future: a shared pool across
  Engines if profiling justifies it.)
- `ProxySupervisor` is shared across Engines in a process; ref-counts
  proxy subprocesses.
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

## Why we own the loop

The Phase-0 prototype wrapped DeepScientist (DS), a Linux-only engine
that requires WSL2 on Windows and depends on `bash_exec` with no
documented Git-Bash fallback. With the redesign, FI is Windows-native
and Linux-native, runs entirely in Python, and benefits from the full
LangChain/LangGraph ecosystem (specifically Axon's
`langchain.AxonRetriever`, which would not be reachable from a Rust
rewrite without a separate REST/MCP hop). The DS wrapper code has been
removed.
