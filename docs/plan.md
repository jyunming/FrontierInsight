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

### Phase C — Non-OpenAI providers ✅ landed (proxy + CLI transports)

Two transport patterns, both reusing existing CLI OAuth so no API key is
required:

**Proxy transport** (`ProxySupervisor._spawn`, ref-counted across quests):
- `claude_code` → `poetry run python main.py <port>` from
  `RichardAtCT/claude-code-openai-wrapper`. Path is
  `FI_CLAUDE_CODE_WRAPPER_DIR` (defaults to `~/claude-code-openai-wrapper`).
  Auth: `claude login` or `ANTHROPIC_API_KEY`.
- `github_copilot_cli` / `github_copilot_vscode` →
  `npx copilot-api@latest start --port <N> --rate-limit 60 --wait`.
  Auth: `npx copilot-api@latest auth` once.
- Readiness probed via `GET /v1/models` rather than raw TCP.

**CLI-exec transport** (no daemon, no port — `LLMClient` spawns the
binary per chat call):
- `claude_cli` → `claude --print --output-format text` with the prompt
  piped to stdin and the response read from stdout. Auth: `claude login`.
  Live-verified end-to-end against Claude Pro/Max OAuth on Windows.
- `codex_cli` → `codex exec --output-last-message <tmpfile> "<prompt>"`
  reading the final assistant message back from the tmpfile. Auth:
  `codex login`.

**Direct transport** (HTTP, OpenAI-compatible):
- `gemini` → `https://generativelanguage.googleapis.com/v1beta/openai/`.
- `openai` / `codex` → `https://api.openai.com/v1` with `OPENAI_API_KEY`.
- `ollama` / `vllm` → local OpenAI-compat endpoints.

When picking between `claude_code` (proxy) and `claude_cli` (CLI exec):
the CLI path has zero infrastructure cost (no clone, no FastAPI proxy
process) but spawns a new subprocess per chat call (~1–3 s overhead on
top of the LLM latency). For fleet runs of many concurrent quests, the
proxy path amortizes one process across all of them and is the right
choice; for one-off runs or local development, the CLI path is simpler.

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

### Phase I — Pre-flight clarification (`clarify` node) ✅ landed

A new node inserted **before** `ideate`. The agent reads the topic and
produces a short structured questionnaire; the user answers; answers
feed into every downstream prompt as additional constraints. Closes
the gap between "topic in YAML" and "agent decides everything."

Static-shape survey (5 fixed slots, agent fills the per-topic specifics):

1. **Comparative baseline** — what existing method / dataset / regime
   should this be compared against?
2. **Empirical vs theoretical** — does this study run code and measure,
   or derive results analytically?
3. **Success metric** — what number changing in what direction would
   count as the headline result?
4. **Time / compute budget** — a soft cap so the `design` node knows
   whether to propose a 30-second sweep or a multi-hour experiment.
5. **Output kinds** — which generators to run (paper / slides / poster
   / speech); paper-only by default.

**Backwards-compatible.** When `engine.clarify_mode = "off"` (default
for tests and fleet quests), the node skips entirely. When `"auto"`,
the agent self-answers from the topic alone — no human in the loop.
When `"interactive"` and `launch.py --interactive` is set, the CLI or
GUI handler reads answers from the user. The node uses LangGraph's
`interrupt()` so the SqliteSaver already gives us pause/resume for free.

`QuestState` gains three fields (additive — Phase G subclasses keep
working): `clarify_questions: dict`, `clarify_answers: dict`,
`clarify_done: bool`.

### Phase J — Status GUI + interactive runner ✅ landed

Single-process FastAPI server with an HTMX frontend. Read paths
mostly exist on disk already (the SqliteSaver, `.fi/run.log`, the
per-quest `frontier_insight_summary.json`); the server is a glorified
file/SQLite browser plus an SSE log stream and a clarify-resume hook.

**Server** (`web/server.py`):
- `GET /` — single-page HTMX shell.
- `GET /api/quests` — every quest dir under the configured root, with
  current node + verdict + age.
- `GET /api/quests/{id}` — detail (state JSON, figures, paper preview).
- `GET /api/quests/{id}/log/stream` — SSE log tail.
- `GET /api/quests/{id}/clarify` — pending questions, if any.
- `POST /api/quests/{id}/clarify` — submit answers; resumes the graph.
- `POST /api/quests/start` — start a new quest from a posted YAML.

**Frontend** (`web/static/`):
- One `index.html` with HTMX + vanilla JS (no build step, no SPA
  framework). Fleet view → quest detail → live log → clarify panel.

**Run as**: `python launch.py --serve [--host 127.0.0.1] [--port 8765]
[--output-dir ./outputs]`. The server reuses the same `Engine`
machinery — starting a quest from the UI is identical to running
`launch.py --config quest.yaml` plus the optional clarify pause.

FastAPI is loaded lazily — `--serve` is the only entry point that
requires it; existing CLI usage is unaffected.

### Phase K — Execute-repair loop (`execute_reflect` node) ✅ landed

Real research is not one-shot — the agent-written experiment script is
the #1 failure surface. Before Phase K, a typo or wrong import in
`experiment.py` made `_node_execute` return `rc != 0`, analyze saw
empty `RESULT_JSON`, write produced a thin paper, and `review → revise`
would re-design from scratch on iteration 2. That's a full quest
restart for a one-line bug.

Phase K inserts a new node **between `execute` and `analyze`**. It
reads `exec_result` and decides:

* **success** (rc==0 AND parseable `RESULT_JSON`) → proceed to analyze.
* **broken** (rc!=0 OR no `RESULT_JSON`) AND iterations left →
  generate a patched `code` from the traceback + previous code +
  prior reflect history. Route back to `execute`.
* **broken** AND iterations exhausted, OR the LLM emits a
  `give_up_reason` sentinel → proceed to analyze anyway (with the
  broken state so analyze can surface the failure in the paper).

Bounded by `engine.exec_reflect_max_iterations` (default 3). Each
iteration costs one LLM call + one venv execution. `QuestState` gains
`exec_reflect_history: list[{iter, returncode, stderr_tail, patch_summary}]`
so analyze and write can describe what was fixed (and review can mark
down a paper that needed many repairs).

The reflect prompt explicitly forbids "fixes that lower the bar"
(swallowing exceptions, skipping assertions, randomizing the success
metric). If the experiment is genuinely impossible the LLM is
instructed to return `{"give_up_reason": "..."}` and we stop.

### Phase L — Analyze-driven re-route + cross-paper check ✅ landed

Two coupled additions, sharing a new `cross_check` node and a
conditional edge after `analyze`:

**Cross-check** (`_node_cross_check`, new). After analyze produces
`key_findings`, we run a literature search **keyed on each finding
text** (not just the original topic), then ask the LLM to classify
each retrieved doc as **supporting**, **conflicting**, or **neutral**
relative to that finding. Results land as
`state["cross_check"]: list[{finding, supporting, conflicting, neutral}]`
and flow into the write prompt's new `$cross_check_block`. This is
the "search for new resources again when there's a new finding"
loop — the literature query uses the *discovered* claim, not the
*starting* topic.

**Analyze-driven re-route**. `_node_analyze` now emits an optional
`next_step` field with three values:

| value | meaning | edge |
|---|---|---|
| `publish` | results stand on their own | → cross_check → write |
| `re_experiment` | data was inconclusive (noise, no signal, effect too small) | → cross_check → design (re-design with the cross-check evidence in hand) |
| `broaden_lit` | new finding raises questions the original literature didn't cover | → cross_check → design (cross_check already broadened the lit) |

`re_experiment` and `broaden_lit` share the existing
`engine.max_iterations` budget — they DON'T double-count against the
review-loop's `revise` path, but they consume the same counter so the
whole quest is bounded.

### Phase M — Ideate self-reflection ✅ landed

The cheapest win. After `ideate` returns its 3–5 ideas and a
`chosen`, a single extra LLM call critiques the choice — "given the
clarify answers and the chosen direction, what's the strongest
objection? would you pick differently?" — and may swap `chosen_idea`
to a different entry from the list (or refine its rationale).
Implemented inline in `_node_ideate`, not as a new node, so the graph
topology stays simple. Adds one LLM call per quest.

Gated by `engine.ideate_reflect: bool = True`. Test-friendly to flip off.

### Phase O — Per-node model routing ✅ landed

`provider.node_models` maps engine node names (and `review_panel.<persona>`
sub-keys) to model strings. `LLMClient.chat(model=...)` accepts a
per-call override that flows through all three transports (HTTP, CLI
exec, and the new VSCode bridge). Each `Engine._chat(prompt, *, node=N)`
call site is tagged; `_model_for_node` resolves N against `node_models`.
Lets the user pick a cheap model for low-value nodes (clarify,
cross_check) and a strong model for the demanding ones (write, review,
each reviewer-panel persona).

### Phase N — Multi-persona reviewer panel ✅ landed

When `engine.review_panel = []` (default), review behaves as a single
LLM call as before. When non-empty (e.g. `[methodologist, statistician,
devil_advocate]`), each persona runs in parallel via `asyncio.gather`
with a persona-specific prefix prepended to `agents/review.md`. The
results aggregate deterministically:

- verdict: any persona votes `revise` with `score < 3` → revise;
  otherwise majority verdict; ties → revise (conservative).
- score: median of panel scores.
- weaknesses: deduped union.
- strengths: intersection.
- suggestions: deduped union with persona-attribution prefix.

A moderator LLM call then writes the prose `rationale` while the
numeric fields stay locked to the deterministic aggregator. The
GUI's quest-detail panel renders one card per persona with verdict +
strengths + weaknesses + suggestions, plus the moderator's synthesis.

### Phase P — Sanctioned VSCode-extension provider ✅ landed

The `github_copilot_cli` / `github_copilot_vscode` proxies in earlier
phases worked but used the third-party `copilot-api` package, which is
explicit about abuse-detection risk in its own README. Phase P
introduces a sanctioned path: a TypeScript VSCode extension
(`vscode-frontier-insight/`) that hosts the Python engine and routes
every LLM call through `vscode.lm.selectChatModels` /
`model.sendRequest`. No reverse-engineering, user-consented, calls
count against the user's normal Copilot subscription.

Architecture:

```
VSCode (user opens Copilot Chat, types `@fi /start config.yaml`)
   │
   ├─ FI extension binds free TCP port, spawns Python:
   │     python launch.py --vscode-bridge-port <N> --config config.yaml
   │
   ├─ Python engine sets provider.name = vscode_extension automatically
   │  (provider.node_models from YAML still honored)
   │
   └─ Per LLM call:
        Python → {"type":"lm_request", id, node, messages, model_hint}
        Extension → vscode.lm.selectChatModels + model.sendRequest
        Extension → streams back as {"type":"lm_chunk", id, delta} ×N
                                  + {"type":"lm_done", id, content}
```

Wire protocol is newline-delimited JSON over `127.0.0.1:<port>`. The
bridge protocol is documented in `core/vscode_bridge.py`. Python-side
tests use an in-process mock TCP server (`tests/test_vscode_bridge.py`)
to cover the wire protocol without VSCode. Live-validation of the
TypeScript extension is manual (install via "Install from VSIX" or
F5 from the extension folder).

Alongside Phase P, the older `github_copilot_*` proxy providers now
emit a one-time warning at engine init pointing at the sanctioned
alternatives (`vscode_extension` for in-VSCode; for headless, the
chat-style CLIs `claude_cli` / `codex_cli` / `gemini_cli` or an
HTTP-direct provider like `openai`). Set `FI_SUPPRESS_PROXY_WARN=1`
to silence (use at your own risk).

### Post-Phase-P enhancements (2026-05-13 batch)

The Phase P bridge proved usable end-to-end but exposed several
real-world failure modes when a researcher actually ran a long-running
quest. The following PRs landed together to harden it:

| PR | What it ships |
|---|---|
| #25 | Bumps the bridge transient-error retry budget to 6 attempts (~2 min cumulative backoff). Replaces the prior 3-attempt budget that failed to outlast sustained Copilot HTTP/2 outages. |
| #26 | Adds `python launch.py --resume <quest_id>` and the in-engine `Engine(resume_quest_id=...)` parameter. |
| #27 | Drops the JSON-wrapped code format in the `implement` node. The model now emits a fenced Python block + a `DEPS:` line — ~30% fewer output tokens, no escape-sequence brittleness on long streams. |
| #28 | Makes `--resume` actually resume from the LangGraph checkpoint (previously it reused the quest_id but still re-ran from `ideate`). Also adds a loud `_AGENTIC_CLI_PROVIDERS` warning for `copilot_cli`, which is an agent loop, not a chat API, and replies conversationally to FI's prompts. |
| #29 | Adds `@fi /resume` chat command to the VSCode extension — picker of every quest dir under `outputs/` that has a checkpoint, sorted most-recent first. |
| #30 | Per-chunk progress heartbeat in the chat panel; 180 s inactivity timeout on the bridge stream so a wedged Copilot call surfaces as `bridge stalled` and Python retries instead of hanging forever. |
| #31 | Switches the bridge stream from `response.text` to `response.stream` so reasoning-model thinking content (`LanguageModelThinkingPart`) renders as `💭` lines in the chat panel. |
| #32 | The engine now copies the source YAML to `<quest_root>/config.yaml` at startup, so `/resume` is a one-step lookup (no slug match against `_drafts/`). |
| #33 | Slides now also produces `slides.pptx` via pandoc when pandoc is on PATH. Real editable PowerPoint deck. Render targets are independent — a failing pdf/html doesn't block pptx. |
| #34 | Paper-depth prompt refinements: literature-context window 600→2000 chars, new `study_depth` clarify slot, depth-aware target in `write.md`, model authors a proper Title-Case title instead of echoing the YAML slug, reviewer grades on new `rigor_score` + `depth_score` axes. |

## What we are explicitly *not* building

- A new research engine that re-implements LangGraph or wraps
  DeepScientist/AI-Scientist as a hard dependency.
- A new vector store, embedding service, or literature-search adapter
  — Axon owns those concerns.
- An SPA framework (React/Vue) for the UI — HTMX is sufficient and
  has no build step.
- A custom GitHub Copilot HTTP wrapper. We use the sanctioned paths
  only (`copilot_cli` for headless, `vscode_extension` for in-VSCode).
  The pre-existing `github_copilot_*` proxy providers stay in the
  codebase for backwards compatibility but emit a runtime warning.
