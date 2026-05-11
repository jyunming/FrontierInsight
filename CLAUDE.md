# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Frontier Insight (FI) is a Windows + Linux native automated research pipeline. **FI owns the research loop itself**: an async LangGraph DAG (`ideate → literature → design → implement → execute → analyze → write → review`) drives an LLM through experimentation; agent-generated code runs in a per-quest venv (default) or Docker container (opt-in). Knowledge is delegated to [`jyunming/Axon`](https://github.com/jyunming/Axon) — do not reimplement vector search, literature retrieval, or cross-quest memory inside FI.

The Phase-0 prototype wrapped DeepScientist; that wrapper has been removed. The historical section of `TEST_RESULTS.md` retains DS-era results for context only — links to `test_runs/` artifacts inside that file are historical and the directory is no longer in the repo.

## Run / develop

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio          # dev
python -m pytest -v                         # 217 tests, ~3 minutes (real venv creates)
python launch.py --config examples/integrator_bakeoff/config.yaml
python launch.py --fleet a.yaml b.yaml --max-concurrent 4 --memory-cap-mb 4096
python launch.py --ingest paper1.pdf paper2.md   # permanent ingest into Axon (no quest)
```

External prerequisites are all optional and feature-gated:

- **Axon** for the knowledge layer — `Knowledge.enabled` falls to `False` if `axon` isn't importable.
- **pandoc + LaTeX** for `paper_pdf` — generator skips with a warning if missing.
- **Marp CLI** (`@marp-team/marp-cli`) for `slides.html` / `slides.pdf` — `slides.md` is still produced.
- **Docker Desktop** for `execution.sandbox: docker`.
- **Provider proxies** (`claude_code`, `github_copilot_*`) only when those providers are selected. See `docs/plan.md` Phase C for verbatim install commands.

## Architecture (the parts that span files)

**Entry point**: `launch.py` is async. It parses args, builds one or more `Config` objects, constructs an `Engine(cfg)` per quest, runs them under a bounded `asyncio.Semaphore` if `--fleet`, then runs each `generation/*` generator on the resulting `QuestArtifacts`. One generator failure does not abort the rest.

**The engine** (`core/engine.py::Engine`) is stateless w.r.t. process globals — N parallel Engines must coexist in one process for `--fleet`. The graph is async LangGraph compiled with `AsyncSqliteSaver` at `<quest_root>/.fi/state.sqlite`; the LangGraph `thread_id` is the quest_id, so a killed run can be resumed by re-running with the same quest output dir.

**`QuestState`** (TypedDict in `core/engine.py`) is the contract between every node and the prompts in `agents/*.md`. Field names are stable; any Phase-G alternate graph must keep them backwards-compatible. `QuestArtifacts` is the contract between the engine and the generators.

**Provider layer** (`core/provider.py`): three transports, one `LLMClient.chat(messages) -> str` surface.
- **HTTP direct** (`codex`/`openai`/`gemini`/`ollama`/`vllm`) — `httpx.AsyncClient` against a `base_url`.
- **HTTP via proxy** (`claude_code`/`github_copilot_cli`/`github_copilot_vscode`) — `ProxySupervisor` spawns the proxy on a free port, ref-counted across quests; readiness probed via `GET /v1/models`.
- **CLI exec** (`claude_cli`/`codex_cli`) — `LLMClient` spawns the local CLI binary per chat call via `asyncio.create_subprocess_exec`. No proxy. Reuses the CLI's own OAuth (`claude login`, `codex login`). Spawn details are in `_CLI_SPECS`; `claude_cli` passes the prompt via stdin and reads stdout; `codex_cli` passes via argv and reads `--output-last-message <tmpfile>`. Retries on `_CliTransientError` (non-zero exit) with the same exponential backoff as the HTTP path.

**Execution layer** (`core/execution.py`): `Executor` protocol with two implementations. `make_executor(sandbox, ...)` is the constructor used by `Engine`. `VenvExecutor` resolves Python at `Scripts/python.exe` on Windows and `bin/python` on POSIX. `DockerExecutor` mounts `<quest_root>` at `/work` with networking disabled.

**Knowledge layer** (`core/knowledge.py`): one in-process `AxonBrain` per quest. Tolerates missing axon install (degrades to no-op + warning). Retrieval (`Knowledge.asearch`) is three-layer: pinned `local_papers` → Axon → external router (`arxiv` / `openalex` / `crossref` / `semantic_scholar` / `pubmed` / `core` / `google_scholar`, parallel + DOI-dedup). With `source_routing: auto` (default) the LLM picks 1–5 sources from a 12-entry catalog (`_SOURCE_CATALOG`); users extend the catalog by ingesting `kind=fi_source_catalog` entries into Axon. With `try_fetch_full_text: true` external hits get opportunistic publisher-PDF fetch via the host's existing network access (VPN/Shibboleth/EZproxy); login walls are rejected by a Content-Type + `%PDF-` magic-bytes two-factor check, so paywalled venues never hang a quest. Write-back (`add_quest_artifacts`) is **gated on `verdict == "accept"`** by default and lays down a *structured bundle*, NOT a flat doc: `fi_paper_spine` (single-chunk card-catalog entry — title/authors/DOI/abstract/key-claims — for title-searchability), `fi_quest_paper` (full body with a 1-line `[Title · Year · Venue · DOI]` citation header prepended to every chunk so retrieval hits are self-describing), `fi_quest_summary` (structured findings JSON; carries `paper_refs` short-id list in metadata), `fi_topic_event` (per-topic rollup pointer keyed by slug; lists the quest + the papers it cited), and `fi_external_ref_spine` × N (curated card-catalog entries for cited externals — only refs that contributed to an accepted quest persist). The `ideate` node reads spines + summaries for cross-quest memory.

## Conventions and gotchas worth knowing

- **Tilde expansion is manual.** `Config` uses `field_validator(..., mode="before")` to expand `~` in path-shaped YAML fields (`output_dir`, `axon_config` when given as a path, every entry in `knowledge.local_papers`). When adding new path fields, mirror that pattern — pydantic does not expand `~`.
- **All node prompts live in `agents/*.md`** as Python `string.Template` files (`$placeholder`, not f-strings) so prompt content can contain literal `{...}` (e.g., JSON examples). Loaded once at engine init.
- **JSON parsing from LLMs is lenient.** `core.engine._parse_json_lenient` strips fences and finds the largest balanced `{...}`. When you add a new node, return JSON and feed it through this helper.
- **`RESULT_JSON: {...}` contract.** Generated experiment scripts emit a single line beginning with `RESULT_JSON: ` as the **last** line of stdout; `_extract_result_json` parses it back into state. This is the cheap, reliable channel for numerical findings to flow from the experiment to the analysis node.
- **Per-quest filesystem layout:** `<output_dir>/<quest_id>/{paper/, figures/, code/, .fi/run.log, .fi/state.sqlite, .venv/, frontier_insight_summary.json}`. `quest_root` is the source of truth; generators read from / write to subdirs of it.
- **Async everywhere.** New code in `core/` should be `async def` unless it's pure CPU work; HTTP must use `httpx.AsyncClient`; subprocesses use `asyncio.create_subprocess_exec` (see `VenvExecutor`); Docker calls offload to `asyncio.to_thread` since `docker-py` is sync.
- **No process-global state.** Avoid module-level singletons; multiple Engines in one process must not collide. `ProxySupervisor` is the one piece of intentional shared state and is explicitly ref-counted.
- **Provider proxies have caveats.** `copilot-api` carries an explicit GitHub abuse-detection warning; FI defaults `--rate-limit 60 --wait`. `claude-code-openai-wrapper` has no PyPI package — `FI_CLAUDE_CODE_WRAPPER_DIR` points at a clone with `poetry install` already run.
- **Rust is deferred.** Don't add `maturin` / PyO3 / a `frontier_insight._fast` Rust crate until Phase H profiling identifies a real hot spot. The 2026 benchmark cited in `docs/plan.md` shows ~5% Rust win on LLM-network-bound work — within noise.

## Tests

`tests/test_config.py` covers the YAML schema and tilde expansion. `tests/test_execution.py` covers `VenvExecutor` (creates a real venv — slow). `tests/test_engine_smoke.py` runs the full 8-node DAG with a fake LLM, real venv, real matplotlib install, real subprocess — a regression detector for the engine plumbing. `tests/test_fleet.py` runs two engines concurrently. `tests/test_knowledge.py` (50 tests) covers source adapters, dedup, router-LLM happy/failure paths, local-paper load + pin, Phase-2 PDF magic-bytes check + login-wall rejection, structured-ingest helpers (spine / header / topic event), external-ref spine writes. `tests/test_knowledge_writeback.py` verifies the cross-quest memory write-back call shape and the accept-gate. **Tests do not call real LLM APIs**; introduce new tests with `monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)` (or `Knowledge.asearch`) to keep that property.

`pytest.ini` redirects `tmp_path` to `./.pytest_tmp/` because the default Windows `%TEMP%` path was inaccessible to the harness sandbox. If a test fails on a fresh checkout, that path is the first thing to check.
