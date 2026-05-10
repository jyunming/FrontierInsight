# Frontier Insight — Roadmap

> **Strategy.** FI owns the research loop end-to-end on Windows + Linux
> native. The orchestrator is async LangGraph with a SQLite checkpointer;
> the knowledge layer is delegated to [`jyunming/Axon`](https://github.com/jyunming/Axon).
> Generated experiment code runs in a per-quest venv (default) or Docker
> sandbox (opt-in). The differentiator is the **output layer**: paper
> templates, slides, posters, speech scripts, plus cross-quest memory.

## Architecture in one paragraph

`launch.py` parses YAML → `Config` → constructs an `Engine` per quest →
the Engine compiles an async `StateGraph` (`ideate → literature → design
→ implement → execute → analyze → write → review`) with a conditional
`review→revise` edge → SqliteSaver checkpoints state at
`<quest_root>/.fi/state.sqlite` → on completion, the generators
(`PaperGenerator`, `SlideGenerator`, `PosterGenerator`,
`SpeechGenerator`) consume the `QuestArtifacts` and produce the
configured outputs. Multiple quests in one process share an `httpx`
connection pool and a reference-counted `ProxySupervisor` for spawned
provider proxies.

## Phases

### Phase A — Core scaffolding ✅ landed

- `core/config.py` with new pydantic schema (`provider`, `engine`,
  `execution`, `knowledge`, `output`).
- `core/platform.py` shrunk to `detect_system()`.
- `core/provider.py` with async `LLMClient` over `httpx` and
  `ProxySupervisor` skeleton.
- `core/execution.py` with `VenvExecutor` (cross-platform Scripts/python
  vs bin/python) and a `make_executor` factory.
- `core/engine.py` async LangGraph skeleton; `Engine` owns one quest.
- `launch.py` rewired; `--config` and `--fleet ... --max-concurrent N`.
- Old `core/runner.py` and `core/providers.py` deleted.

### Phase B — Real research loop ✅ landed

- All 8 nodes implemented as real LLM calls.
- `agents/<node>.md` prompts (Python `string.Template`).
- `core/knowledge.py` wraps Axon (`AxonBrain`, `AxonRetriever`); ingest +
  search + post-quest write-back. Tolerates missing axon install.
- `_node_execute` writes generated code to
  `<quest_root>/code/experiment.py`, pip-installs declared deps,
  runs the script, scans `figures/`, parses `RESULT_JSON: {...}` from
  stdout.
- SqliteSaver checkpointer compiled into the graph.
- pytest harness: `tests/test_config.py`, `test_execution.py`,
  `test_engine_smoke.py` (full-DAG fake-LLM smoke).

### Phase C — Provider proxies ✅ structurally landed

`ProxySupervisor._spawn` now ships verbatim invocations:

- `claude_code` → `poetry run python main.py <port>` from
  `RichardAtCT/claude-code-openai-wrapper`. Path is
  `FI_CLAUDE_CODE_WRAPPER_DIR` (defaults to `~/claude-code-openai-wrapper`).
  Auth: `claude auth login` or `ANTHROPIC_API_KEY`.
- `github_copilot_cli` / `github_copilot_vscode` →
  `npx copilot-api@latest start --port <N> --rate-limit 60 --wait`.
  Auth: `npx copilot-api@latest auth` once.
- `gemini` → direct `base_url`
  `https://generativelanguage.googleapis.com/v1beta/openai/`.

Readiness probe is `GET /v1/models` rather than raw TCP. Live
authentication and per-provider bake-off runs are user-validated when
the prereqs are installed.

### Phase D — Docker sandbox ✅ landed

`core/execution.py::DockerExecutor` implements the `Executor` protocol
end-to-end via `docker-py`: ensures the image is pulled, mounts
`<quest_root>` at `/work` with networking disabled, runs the command,
captures stdout/stderr/figures. `make_executor` selects venv vs docker
from `config.execution.sandbox`.

### Phase E — Output formats ✅ landed

- **E-1 paper**: `generation/paper.py` invokes pandoc + LaTeX with a
  template at `templates/paper/<format>/template.tex` when
  `paper_pdf` is requested. Ships `generic` + `neurips` templates fully;
  `iclr`, `ieee_access`, `nature_mi` are stubs that fall back to
  pandoc's default. Skips PDF cleanly when pandoc / pdflatex is missing.
- **E-2 slides**: `generation/slides.py` LLM-compresses paper.md into
  Marp markdown, then shells out to `marp` for `slides.html` /
  `slides.pdf`. Requires Marp CLI; `slides.md` is always produced.
- **E-3 poster**: `generation/poster.py` LLM-fills three columns of a
  fixed 36"×48" `beamerposter` template at
  `templates/poster/poster.tex`; pdflatex compiles. `poster.tex`
  always written.
- **E-4 speech**: `generation/speech.py` single LLM call from paper.md
  (and slides outline if present) → `talk.md` (~10 min spoken).

`launch.py` runs the four generators in sequence; one failure does not
abort the others.

### Phase F — Cross-quest memory ✅ landed via Axon

After each successful quest the engine calls
`Knowledge.add_quest_artifacts(quest_id, paper_md_path, summary)` with
the analysis summary + key findings. The `ideate` node already calls
`Knowledge.search(topic)`, so future quests surface past work
naturally. Validated by `tests/test_knowledge_writeback.py` (with a
stand-in for Axon).

### Phase G — Alternate graph swap ✅ hook only (deferred)

`core/engine.py::Engine._build_graph` is the override point. Subclass
`Engine` and replace it to ship a domain-specific pipeline; keep the
`QuestState` TypedDict field names backwards-compatible. No alternate
graph is shipped today — the hook is structural.

### Phase H — Fleet runner & concurrency ✅ landed

- `launch.py --fleet <a.yaml> <b.yaml> ... --max-concurrent N` runs N
  Engine instances under a bounded `asyncio.Semaphore`.
- `ProxySupervisor` ref-counts spawned proxy subprocesses across quests.
- Per-quest log isolation at `<quest_root>/.fi/run.log`; per-quest
  stdout prefix; fleet status line on each engine completion
  (`[FI fleet] start … running=2 done=1/4 failed=0 elapsed=14s rss=820MB`).
- `--memory-cap-mb N` throttles new quest starts via psutil RSS check.
- `--profile` wraps each engine run in viztracer if installed.
- Validated by `tests/test_fleet.py` (two fake-LLM engines concurrently).

## Cross-cutting concerns

- **Cross-platform.** Validated on Windows-native (Python 3.11.9) end
  to end through pytest. Linux/macOS work via the same code paths
  (`pathlib.Path` everywhere, `Scripts/` vs `bin/` selected by
  `sys.platform`); ad-hoc Linux validation pending.
- **Local vs cloud LLM.** Selected via `provider.name`: cloud
  (`codex`/`openai`/`gemini`/`claude_code`/`github_copilot_*`) or
  local (`ollama`/`vllm`). All paths resolve to a single OpenAI Chat
  Completions client.
- **Concurrency.** Async-first throughout (LangGraph async nodes,
  `httpx.AsyncClient`, `asyncio.create_subprocess_exec`,
  ref-counted proxies). Many quests in one process is the supported
  scaling axis — see `--fleet`.
- **Rust kernels: deferred.** A 2026 benchmark of Rust LLM
  orchestrators showed ~5% latency win (within noise) on
  LLM-API-bound work like FI's. The hybrid model (Python orchestration
  + a Rust kernel via maturin) is the right reach if Phase H profiling
  surfaces a hot spot. Today there is no such hot spot.

## What we are explicitly *not* building

- A new research engine that re-implements LangGraph or wraps
  DeepScientist/AI-Scientist as a hard dependency.
- A new vector store, embedding service, or literature-search adapter
  — Axon owns those concerns.
- A web UI / Docker-deploy / one-click wrapper — Phase ≥7 if ever.
