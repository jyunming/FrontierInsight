# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Frontier Insight (FI) is a Windows + Linux native automated research pipeline. **FI owns the research loop itself**: an async LangGraph DAG (`ideate → literature → design → implement → execute → analyze → write → review`) drives an LLM through experimentation; agent-generated code runs in a per-quest venv (default) or Docker container (opt-in). Knowledge is delegated to [`jyunming/Axon`](https://github.com/jyunming/Axon) — do not reimplement vector search, literature retrieval, or cross-quest memory inside FI.

The Phase-0 prototype wrapped DeepScientist; that wrapper has been removed. References to DS in `test_runs/` and the historical section of `TEST_RESULTS.md` are kept for context only.

## Run / develop

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio          # dev
python -m pytest -v                         # 11 tests, ~3 minutes (real venv creates)
python launch.py --config examples/integrator_bakeoff/config.yaml
python launch.py --fleet a.yaml b.yaml --max-concurrent 4 --memory-cap-mb 4096
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

**Provider layer** (`core/provider.py`): all paths terminate at OpenAI Chat Completions over `httpx.AsyncClient`. `resolve_endpoint_async` returns a `ResolvedEndpoint` for direct providers; for proxy providers (`claude_code`, `github_copilot_*`) it goes through `ProxySupervisor`, which spawns a child process on a free port and reference-counts so concurrent quests share one proxy process. Readiness is probed via `GET /v1/models`, not raw TCP.

**Execution layer** (`core/execution.py`): `Executor` protocol with two implementations. `make_executor(sandbox, ...)` is the constructor used by `Engine`. `VenvExecutor` resolves Python at `Scripts/python.exe` on Windows and `bin/python` on POSIX. `DockerExecutor` mounts `<quest_root>` at `/work` with networking disabled.

**Knowledge layer** (`core/knowledge.py`): one in-process `AxonBrain` per quest. Tolerates missing axon install (degrades to no-op + warning). `add_quest_artifacts(quest_id, paper_md_path, summary)` writes the finished paper back into Axon for cross-quest memory; the `ideate` node reads it.

## Conventions and gotchas worth knowing

- **Tilde expansion is manual.** `Config` uses `field_validator(..., mode="before")` to expand `~` in path-shaped YAML fields (`output_dir`, `axon_config` when given as a path). When adding new path fields, mirror that pattern — pydantic does not expand `~`.
- **All node prompts live in `agents/*.md`** as Python `string.Template` files (`$placeholder`, not f-strings) so prompt content can contain literal `{...}` (e.g., JSON examples). Loaded once at engine init.
- **JSON parsing from LLMs is lenient.** `core.engine._parse_json_lenient` strips fences and finds the largest balanced `{...}`. When you add a new node, return JSON and feed it through this helper.
- **`RESULT_JSON: {...}` contract.** Generated experiment scripts emit a single line beginning with `RESULT_JSON: ` as the **last** line of stdout; `_extract_result_json` parses it back into state. This is the cheap, reliable channel for numerical findings to flow from the experiment to the analysis node.
- **Per-quest filesystem layout:** `<output_dir>/<quest_id>/{paper/, figures/, code/, .fi/run.log, .fi/state.sqlite, .venv/, frontier_insight_summary.json}`. `quest_root` is the source of truth; generators read from / write to subdirs of it.
- **Async everywhere.** New code in `core/` should be `async def` unless it's pure CPU work; HTTP must use `httpx.AsyncClient`; subprocesses use `asyncio.create_subprocess_exec` (see `VenvExecutor`); Docker calls offload to `asyncio.to_thread` since `docker-py` is sync.
- **No process-global state.** Avoid module-level singletons; multiple Engines in one process must not collide. `ProxySupervisor` is the one piece of intentional shared state and is explicitly ref-counted.
- **Provider proxies have caveats.** `copilot-api` carries an explicit GitHub abuse-detection warning; FI defaults `--rate-limit 60 --wait`. `claude-code-openai-wrapper` has no PyPI package — `FI_CLAUDE_CODE_WRAPPER_DIR` points at a clone with `poetry install` already run.
- **Rust is deferred.** Don't add `maturin` / PyO3 / a `frontier_insight._fast` Rust crate until Phase H profiling identifies a real hot spot. The 2026 benchmark cited in `docs/plan.md` shows ~5% Rust win on LLM-network-bound work — within noise.

## Tests

`tests/test_config.py` covers the YAML schema and tilde expansion (5 tests). `tests/test_execution.py` covers `VenvExecutor` (3 tests; creates a real venv — slow). `tests/test_engine_smoke.py` runs the full 8-node DAG with a fake LLM, real venv, real matplotlib install, real subprocess — a regression detector for the engine plumbing. `tests/test_fleet.py` runs two engines concurrently. `tests/test_knowledge_writeback.py` verifies cross-quest memory write-back is invoked. **Tests do not call real LLM APIs**; introduce new tests with `monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)` to keep that property.

`pytest.ini` redirects `tmp_path` to `./.pytest_tmp/` because the default Windows `%TEMP%` path was inaccessible to the harness sandbox. If a test fails on a fresh checkout, that path is the first thing to check.
