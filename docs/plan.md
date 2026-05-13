# Frontier Insight — Roadmap

> **Strategy.** FI owns the research loop end-to-end on Windows + Linux
> native. The orchestrator is async LangGraph with a SQLite checkpointer;
> the knowledge layer is delegated to [`jyunming/Axon`](https://github.com/jyunming/Axon).
> Generated experiment code runs in a per-quest venv (default) or Docker
> sandbox (opt-in). The differentiator is the **output layer**: paper
> templates, slides, posters, speech scripts, plus cross-quest memory.

## Architecture in one paragraph

`launch.py` parses YAML → `Config` → constructs an `Engine` per quest →
the Engine compiles an async `StateGraph`
(`clarify → ideate → literature → design → implement → execute →
execute_reflect → analyze → cross_check → write → review`) with three
feedback loops → `AsyncSqliteSaver` checkpoints state at
`<quest_root>/.fi/state.sqlite` → on completion, the generators
(`PaperGenerator`, `SlideGenerator`, `PosterGenerator`,
`SpeechGenerator`) consume the `QuestArtifacts` and produce the
configured outputs. Multiple quests in one process share an `httpx`
connection pool and a reference-counted `ProxySupervisor` for spawned
provider proxies.

## Shipped capabilities

The shipped capability list, organized by area rather than dev order.
For exact YAML field semantics see [`capabilities.md`](capabilities.md);
for the layered design see [`architecture.md`](architecture.md).

### Engine
- 11-node async LangGraph DAG with three feedback loops.
- `AsyncSqliteSaver` checkpointing → resumability via
  `--resume <quest_id>` or `@fi /resume`.
- Stateless `Engine` so N parallel quests can share one process
  (`launch.py --fleet`).

### Self-correction
- **Execute-repair loop** — when an agent-generated script crashes,
  the engine reads the traceback and lets the LLM patch the code,
  bounded by `engine.exec_reflect_max_iterations`.
- **Analyze re-route** — `analyze.next_step ∈ {re_experiment,
  broaden_lit}` sends control back to `design` (bounded by
  `engine.max_iterations`).
- **Review-driven revise** — the classic `review.verdict == "revise"`
  loop, sharing the same `max_iterations` budget.
- **Cross-paper check** — after `analyze`, each key finding gets a
  per-finding literature search and a supporting / conflicting /
  neutral classification surfaced into the write prompt.
- **Ideate self-reflection** — an extra LLM call may swap the
  picked idea before design begins.

### Clarify
- 7-slot pre-flight questionnaire (`comparative_baseline`,
  `empirical_vs_theoretical`, `success_metric`, `budget`,
  `output_kinds`, `study_depth`, `paper_venue`).
- `clarify_mode: off | auto | interactive`. `auto` lets the agent
  self-fill defaults; `interactive` pauses the graph and reads
  answers from the user via the terminal or the VSCode bridge.
- Answers flow into every downstream prompt as `$clarify_block`.

### Provider transports
- **HTTP direct** — `openai` / `codex` / `gemini` / `ollama` / `vllm`.
- **CLI exec** — `claude_cli`, `codex_cli`, `gemini_cli` for headless
  OAuth-based use. (`copilot_cli` is wired but emits a loud warning:
  it's an agent loop, not a stateless chat API, and produces
  conversational replies instead of structured node output.)
- **HTTP via proxy** — `claude_code`, `github_copilot_*` (kept for
  backwards compatibility; emit ToS warnings at engine init).
- **VSCode-extension bridge** (`vscode_extension`) — routes chat
  calls through the FI VSCode extension's `vscode.lm.*` API. Picks
  up whatever model the user selects in the Copilot Chat picker
  (GPT family, Anthropic Claude / Google Gemini etc. when the user's
  Copilot subscription federates them).
- **Per-node model routing** — `provider.node_models` can pick a
  different model per engine node, including per reviewer-panel
  persona.

### Knowledge layer
- Axon-backed three-tier retrieval: pinned `local_papers` → Axon
  store → external router (`arxiv` / `openalex` / `crossref` /
  `semantic_scholar` / `pubmed` / `core` / `google_scholar`,
  parallel + DOI-dedup).
- **Topic-aware source routing** — the LLM picks 1–5 sources from a
  12-entry catalog when `source_routing: auto`; users extend the
  catalog by ingesting `kind=fi_source_catalog` entries.
- **Opportunistic full-text fetch** — publisher PDFs over the host's
  network with login-wall rejection via Content-Type +
  `%PDF-` magic-bytes check. Paywalled venues never hang a quest.
- **Structured ingest after accepted quests** — `fi_paper_spine`,
  `fi_quest_paper`, `fi_quest_summary`, `fi_topic_event`,
  `fi_external_ref_spine` × N.
- **Folder summarizer** — `python launch.py --summarize <folder>`
  and `@fi /summarize <folder>`. Walks a mixed folder (papers,
  source code, study notes, execution logs, prior quest outputs),
  classifies each file, calls the LLM once, and writes
  `outputs/<summary_id>/summary.md`. Input files and the summary
  are ingested into Axon as `fi_summary_input` / `fi_summary`.

### Reviewer panel
- N personas in parallel (`methodologist`, `statistician`,
  `devil_advocate`, `reproducibility`) plus a moderator synthesis.
- Per-persona model routing.
- Per-persona reviews carry `verdict`, `score`, plus the new
  `rigor_score` and `depth_score` axes; the moderator aggregates
  medians for the latter two.

### Output generators
- **Paper** — `paper/paper.md` (always when `paper_md` is in
  `output.kinds`); `paper/paper.pdf` via pandoc + a LaTeX engine
  when `paper_pdf` is requested. Templates ship for `generic` and
  `neurips`; others fall back to pandoc default. LaTeX engine
  resolution: pdflatex on PATH preferred, then tectonic on PATH,
  then `<repo>/tools/tectonic.{exe,}` (the target of
  `python launch.py --install-tectonic` — the no-admin path).
- **Slides** — `slides.md` always; `slides.html` and `slides.pdf`
  via the Marp CLI when on PATH; `slides.pptx` via pandoc.
- **Poster** — `poster.tex` from a `beamerposter` template; optional
  `poster.pdf` via pdflatex.
- **Speech** — `talk.md`, a spoken-form script keyed to slides.

### Sandboxed execution
- **`VenvExecutor`** (default) — creates a per-quest `.venv/`,
  resolves Python at `Scripts/python.exe` (Windows) or
  `bin/python` (POSIX), pip-installs declared deps, runs the
  agent-generated script.
- **`DockerExecutor`** — bind-mounts `<quest_root>` at `/work`
  with networking disabled. Selected via `execution.sandbox: docker`.

### VSCode integration
- Chat participant `@fi` registered via
  `vscode.chat.createChatParticipant`. Commands: `/new` (interview),
  `/start <yaml>`, `/fleet <yaml> <yaml> …`, `/resume [quest_id]`,
  `/summarize <folder> [kind]`, `/help`.
- Every LLM call routes through `vscode.lm.selectChatModels` +
  `model.sendRequest` — the sanctioned API, no third-party proxies.
- Per-chunk streaming progress + 180 s inactivity timeout in the
  bridge so a wedged upstream call surfaces as a clean retry
  instead of an indefinite hang.
- Reasoning-model thinking content (Copilot reasoning models)
  surfaced to the chat panel as `💭` lines for visibility into
  long pre-token waits.
- The engine drops a copy of the YAML at `<quest_root>/config.yaml`
  at startup so `/resume` is a one-step lookup.

### Fleet runner
- Bounded `asyncio.Semaphore` for `launch.py --fleet`.
- `--memory-cap-mb` throttles new quest starts when RSS approaches
  the cap (psutil-based).
- `--profile` wraps each engine run in viztracer when installed.
- Ref-counted `ProxySupervisor` shared across quests.

### Status GUI
- FastAPI + vanilla-JS frontend launched via `launch.py --serve`.
- Per-quest live log stream over SSE; clarify panel for interactive
  mode; paper preview; reviewer-panel cards.

## Cross-cutting concerns

- **Cross-platform.** Validated on Windows-native Python 3.11.9 end
  to end through pytest. Linux/macOS work via the same code paths
  (`pathlib.Path` everywhere, `Scripts/` vs `bin/` selected by
  `sys.platform`).
- **Local vs cloud LLM.** Selected via `provider.name`: cloud
  (`codex`/`openai`/`gemini`/VSCode-bridge), local
  (`ollama`/`vllm`). All paths resolve to a single
  OpenAI-compatible chat surface.
- **Concurrency.** Async-first throughout (LangGraph async nodes,
  `httpx.AsyncClient`, `asyncio.create_subprocess_exec`,
  ref-counted proxies). Many quests in one process is the
  supported scaling axis — see `--fleet`.
- **No admin required.** All system-tool dependencies (pandoc,
  pdflatex, Marp, Node.js) can install per-user. For LaTeX
  specifically, `python launch.py --install-tectonic` drops a
  self-bootstrapping LaTeX binary into `tools/` for corporate
  environments where MiKTeX install is blocked.

## What we are explicitly *not* building

- A new research engine that re-implements LangGraph or wraps
  DeepScientist / AI-Scientist as a hard dependency.
- A new vector store, embedding service, or literature-search
  adapter — Axon owns those concerns.
- An SPA framework (React / Vue) for the UI — vanilla JS with HTMX
  is sufficient and has no build step.
- A custom GitHub Copilot HTTP wrapper. For in-VSCode use, the
  sanctioned `vscode_extension` path covers every model the user's
  Copilot Chat picker exposes. For headless use, the chat-style
  CLIs (`claude_cli` / `codex_cli` / `gemini_cli`) or
  HTTP-direct providers (`openai` / `gemini` / `ollama` / `vllm`)
  are the supported paths.
