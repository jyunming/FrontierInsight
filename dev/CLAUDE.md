# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Frontier Insight (FI) is a Windows + Linux native automated research pipeline. **FI owns the research loop itself**: an async LangGraph DAG drives an LLM through experimentation; agent-generated code runs in a per-quest venv (default) or Docker container (opt-in). Knowledge is delegated to [`jyunming/Axon`](https://github.com/jyunming/Axon) — do not reimplement vector search, literature retrieval, or cross-quest memory inside FI.

The graph has two paths after `design`. The simulate path is `implement → execute → execute_reflect → analyze`; the no-simulation path (when `state.no_simulation_resolved` is True, decided at `clarify` or via `engine.no_simulation: true` in YAML) is `auto_collect_data → wait_for_data → data_load → analyze`. Both converge at `analyze → cross_check → write → review`.

Three feedback loops: (a) `execute_reflect → execute` — fix a crashed script, bounded by `engine.exec_reflect_max_iterations`; (b) `cross_check → design` — `analyze.next_step ∈ {re_experiment, broaden_lit}` re-routes to design; (c) `review → design` — the classic revise loop. Loops (b) and (c) share `engine.max_iterations` as the budget; loop (a) has its own counter so a broken script doesn't eat the design-iteration budget. `no_simulation_resolved` is sticky across re-entries, so a redesign on a no-sim quest goes back through `auto_collect_data`, not `implement`.

## Run / develop

```bash
pip install -r requirements.txt
python -m pytest -v                                         # full suite, ~9 minutes (real venv creates)
python launch.py --config examples/integrator_bakeoff/config.yaml
python launch.py --fleet a.yaml b.yaml --max-concurrent 4 --memory-cap-mb 4096
python launch.py --config my.yaml --resume <quest_id>       # re-enter a crashed quest from its sqlite checkpoint
python launch.py --ingest paper1.pdf paper2.md              # permanent ingest into Axon (no quest)
```

External prerequisites are all optional and feature-gated:

- **Axon** for the knowledge layer — `Knowledge.enabled` falls to `False` if `axon` isn't importable.
- **pandoc + LaTeX** for `paper_pdf` — generator skips with a warning if missing.
- **Marp CLI** (`@marp-team/marp-cli`) for `slides.html` / `slides.pdf` — `slides.md` is still produced.
- **Docker Desktop** for `execution.sandbox: docker`.
- **Provider proxies** (`claude_code`, `github_copilot_*`) only when those providers are selected. See `docs/PROVIDERS.md` for verbatim install commands.

## Architecture (the parts that span files)

**Entry point**: `launch.py` is async. It parses args, builds one or more `Config` objects, constructs an `Engine(cfg)` per quest, runs them under a bounded `asyncio.Semaphore` if `--fleet`, then runs each `generation/*` generator on the resulting `QuestArtifacts`. One generator failure does not abort the rest.

**The engine** (`core/engine.py::Engine`) is stateless w.r.t. process globals — N parallel Engines must coexist in one process for `--fleet`. The graph is async LangGraph compiled with `AsyncSqliteSaver` at `<quest_root>/.fi/state.sqlite`; the LangGraph `thread_id` is the quest_id. A killed run is resumed via `python launch.py --config <yaml> --resume <quest_id>` (or `@fi /resume <quest_id>` in VSCode). On resume `Engine.run()` calls `graph.aget_state()` first; if prior state exists, it passes `None` to `ainvoke` so LangGraph continues from the last completed node instead of replaying from `ideate`. The source YAML is auto-copied to `<quest_root>/config.yaml` at startup so `/resume` is a one-step lookup.

**`QuestState`** (TypedDict in `core/engine.py`) is the contract between every node and the prompts in `agents/*.md`. Field names are stable; any alternate graph must keep them backwards-compatible. `QuestArtifacts` is the contract between the engine and the generators. Self-correction state lives on `QuestState` too: `clarify_*`, `ideate_critique`, `exec_reflect_iter`/`exec_reflect_history`/`exec_give_up_reason`, `cross_check`. No-simulation bookkeeping: `no_simulation_resolved` (bool), `auto_collected_count` (int — total files written into `data/auto_collected/` by Axon + every adapter in `engine.dataset_adapters`).

**Provider layer** (`core/provider.py`): four transports, one `LLMClient.chat(messages) -> str` surface.
- **HTTP direct** (`codex`/`openai`/`gemini`/`ollama`/`vllm`) — `httpx.AsyncClient` against a `base_url`.
- **HTTP via proxy** (`claude_code`/`github_copilot_cli`/`github_copilot_vscode`) — `ProxySupervisor` spawns the proxy on a free port, ref-counted across quests; readiness probed via `GET /v1/models`. The two `github_copilot_*` providers emit a one-time warning at engine init (third-party `copilot-api` proxy, abuse-detection risk).
- **CLI exec** (`claude_cli`/`codex_cli`/`copilot_cli`/`gemini_cli`) — `LLMClient` spawns the local CLI binary per chat call via `asyncio.create_subprocess_exec`. No proxy. Reuses the CLI's own OAuth. Spawn details in `_CLI_SPECS`. `claude_cli` and `codex_cli` and `gemini_cli` are chat-style. **`copilot_cli` is agentic**: it interprets FI's node prompts as user coding tasks and replies conversationally; the engine emits a loud warning when it's selected. For Copilot use `vscode_extension` instead.
- **VSCode bridge** (`vscode_extension`). `VSCodeBridgeClient` sends newline-delimited JSON over a localhost TCP socket the FI VSCode extension spawned us with via `--vscode-bridge-port N`. The extension makes the actual `vscode.lm.selectChatModels` + `model.sendRequest` call and streams chunks back. 180 s inactivity timeout + 6-attempt Python-side retry (cumulative ~2 min backoff) protects against Copilot HTTP/2 stalls.

**Execution layer** (`core/execution.py`): `Executor` protocol with two implementations. `make_executor(sandbox, ...)` is the constructor used by `Engine`. `VenvExecutor` resolves Python at `Scripts/python.exe` on Windows and `bin/python` on POSIX. `DockerExecutor` mounts `<quest_root>` at `/work` with networking disabled.

**Knowledge layer** (`core/knowledge.py`): one in-process `AxonBrain` per quest. Tolerates missing axon install (degrades to no-op + warning). Retrieval (`Knowledge.asearch`) is three-layer: pinned `local_papers` → Axon → external router (`arxiv` / `openalex` / `crossref` / `semantic_scholar` / `pubmed` / `core` / `google_scholar`, parallel + DOI-dedup). With `source_routing: auto` (default) the LLM picks 1–5 sources from a 12-entry catalog (`_SOURCE_CATALOG`); users extend the catalog by ingesting `kind=fi_source_catalog` entries into Axon. With `try_fetch_full_text: true` external hits get opportunistic publisher-PDF fetch via the host's existing network access (VPN/Shibboleth/EZproxy); login walls are rejected by a Content-Type + `%PDF-` magic-bytes two-factor check, so paywalled venues never hang a quest. Write-back (`add_quest_artifacts`) is **gated on `verdict == "accept"`** by default and lays down a *structured bundle*, NOT a flat doc: `fi_paper_spine` (single-chunk card-catalog entry — title/authors/DOI/abstract/key-claims — for title-searchability), `fi_quest_paper` (full body with a 1-line `[Title · Year · Venue · DOI]` citation header prepended to every chunk so retrieval hits are self-describing), `fi_quest_summary` (structured findings JSON; carries `paper_refs` short-id list in metadata), `fi_topic_event` (per-topic rollup pointer keyed by slug; lists the quest + the papers it cited), and `fi_external_ref_spine` × N (curated card-catalog entries for cited externals — only refs that contributed to an accepted quest persist). The `ideate` node reads spines + summaries for cross-quest memory.

**Dataset adapters** (`core/datasets/`): the no-simulation `auto_collect_data` node can additionally pull structured data via adapters listed in `engine.dataset_adapters`. The abstract base is `core/datasets/base.py::DatasetAdapter` — a single `async def search(self, query: str, *, top_k: int) -> list[DatasetRow]`. Shipped adapters: `worldbank.py` (country-indicator tables via the WorldBank Indicators API, stdlib `urllib`) and `wikipedia.py` (opensearch + page-summary lookups for qualitative topics). Adapter failures log a WARNING and return `[]`; `Engine._run_dataset_adapters` sums counts and keeps the quest moving. Adding a new adapter is a single-file change plus a registry entry; no graph edit.

## Conventions and gotchas worth knowing

- **Tilde expansion is manual.** `Config` uses `field_validator(..., mode="before")` to expand `~` in path-shaped YAML fields (`output_dir`, `axon_config` when given as a path, every entry in `knowledge.local_papers`). When adding new path fields, mirror that pattern — pydantic does not expand `~`.
- **All node prompts live in `agents/*.md`** as Python `string.Template` files (`$placeholder`, not f-strings) so prompt content can contain literal `{...}` (e.g., JSON examples). Loaded once at engine init.
- **JSON parsing from LLMs is lenient.** `core.engine._parse_json_lenient` strips fences and finds the largest balanced `{...}`. When you add a new node, return JSON and feed it through this helper.
- **`RESULT_JSON: {...}` contract.** Generated experiment scripts emit a single line beginning with `RESULT_JSON: ` as the **last** line of stdout; `_extract_result_json` parses it back into state. This is the cheap, reliable channel for numerical findings to flow from the experiment to the analysis node.
- **Per-quest filesystem layout:** `<output_dir>/<quest_id>/{paper/, figures/, code/, .fi/run.log, .fi/state.sqlite, .venv/, frontier_insight_summary.json}`. `quest_root` is the source of truth; generators read from / write to subdirs of it.
- **Async everywhere.** New code in `core/` should be `async def` unless it's pure CPU work; HTTP must use `httpx.AsyncClient`; subprocesses use `asyncio.create_subprocess_exec` (see `VenvExecutor`); Docker calls offload to `asyncio.to_thread` since `docker-py` is sync.
- **No process-global state.** Avoid module-level singletons; multiple Engines in one process must not collide. `ProxySupervisor` is the one piece of intentional shared state and is explicitly ref-counted.
- **Provider proxies have caveats.** `copilot-api` carries an explicit GitHub abuse-detection warning; FI defaults `--rate-limit 60 --wait`. `claude-code-openai-wrapper` has no PyPI package — `FI_CLAUDE_CODE_WRAPPER_DIR` points at a clone with `poetry install` already run.
- **Rust is deferred.** Don't add `maturin` / PyO3 / a `frontier_insight._fast` Rust crate until profiling identifies a real hot spot. The 2026 benchmark cited in `docs/plan.md` shows ~5% Rust win on LLM-network-bound work — within noise.
- **No PR-number / Phase-letter refs in user-facing docs** (`README.md`, `docs/**` outside `docs/audits/`, `vscode-frontier-insight/README.md`, `CLAUDE.md`). Describe features in their own terms; keep "PR #N" / "Phase X" lineage in commit messages, PR descriptions, and `docs/audits/` (which IS the historical record). Memory files (`.claude/projects/.../memory/`) and audit reports can keep them.

## Tests

`tests/test_config.py` covers the YAML schema and tilde expansion. `tests/test_execution.py` covers `VenvExecutor` (creates a real venv — slow). `tests/test_engine_smoke.py` runs the full DAG with a fake LLM, real venv, real matplotlib install, real subprocess — a regression detector for the engine plumbing. `tests/test_engine_helpers.py` is the unit-test workhorse for the engine resolver / routing helpers (128+ cases). `tests/test_fleet.py` runs two engines concurrently. `tests/test_knowledge.py` covers source adapters, dedup, router-LLM happy/failure paths, local-paper load + pin, PDF magic-bytes check + login-wall rejection, structured-ingest helpers (spine / header / topic event), external-ref spine writes. `tests/test_knowledge_writeback.py` verifies the cross-quest memory write-back call shape and the accept-gate. `tests/test_docker_executor.py` mocks docker-py for unit tests and gates real-daemon integration tests on `info["OSType"] == "linux"` so Windows-mode daemons skip cleanly. Dataset adapters: `tests/test_datasets_*.py` for worldbank + wikipedia. VSCode extension: `tests/test_vscode_extension_typescript.py` (tsc + vsce package round-trip) and `tests/test_vscode_extension_icon.py` (the 128×128 RGBA contract + README references). Per-PM-command modules have their own files (`tests/test_{digest,portfolio,critique,proposal}.py`). **Tests do not call real LLM APIs**; introduce new tests with `monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)` (or `Knowledge.asearch`) to keep that property.

**CI split** (`.github/workflows/ci.yml`): the **fast tier** runs on every PR and every push to main on both ubuntu-latest and windows-latest with `-m "not slow"` plus an explicit `--ignore` for the three e2e files. The **slow tier** runs on ubuntu only and covers (a) the three e2e files (`test_engine_smoke.py`, `test_self_correction_e2e.py`, `test_web_e2e.py`) and (b) any `@pytest.mark.slow`-tagged test outside them; pytest rc=5 ("no tests collected") is treated as success there. Both tiers gate every PR — the slow tier was historically post-merge only, but two slow-tier-only regressions slipped past in quick succession, so the gate moved to pre-merge. A `paths-ignore: docs/**` filter at the top of the workflow skips the whole thing on docs-only PRs.

`pytest.ini` redirects `tmp_path` to `./.pytest_tmp/` because the default Windows `%TEMP%` path was inaccessible to the harness sandbox. If a test fails on a fresh checkout, that path is the first thing to check.

**Don't claim the VSCode extension is done without running BOTH `npm run compile` AND `npm run package`.** Two distinct steps; one silently missing the other has bitten this codebase twice already. `tests/test_vscode_extension_typescript.py` runs both from pytest as a structural check. The tests are gated on `node`+`npm`+`vscode-frontier-insight/node_modules` being present and skip silently otherwise, so any "extension shipping" message must explicitly state both were run and pin the resulting `.vsix` path. The tracked `.vsix` lives at `vscode-frontier-insight/vscode-frontier-insight.vsix` and is auto-rebuilt on main by `.github/workflows/vsix-rebuild.yml`; you only need to commit a rebuilt copy when iterating on a feature branch where users will install before main merges.

**Audit reports live in `docs/audits/`.** That folder IS the historical record — PR-number and Phase-letter references are fine there. The user-facing prose docs (README, `docs/USAGE.md`, `docs/capabilities.md`, `docs/architecture.md`, `vscode-frontier-insight/README.md`) describe behaviour in its own terms and never reference PRs or phases.
