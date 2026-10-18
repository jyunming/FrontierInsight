# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex and others) working with code in this repository. `CLAUDE.md` only imports this file, so Claude Code reads the same text.

## What this project is

Frontier Insight (FI) is a Windows + Linux native automated research pipeline. **FI owns the research loop itself**: an async LangGraph DAG drives an LLM through experimentation; agent-generated code runs in a per-quest venv (default) or Docker container (opt-in). Knowledge is delegated to [`jyunming/Axon`](https://github.com/jyunming/Axon) — do not reimplement vector search, literature retrieval, or cross-quest memory inside FI.

The graph has two paths after `design`. The simulate path is `implement → execute → execute_reflect → improve → analyze` (`improve`, `core/improve.py`, changes the simulation one step at a time against the protocol's checks of correctness, bounded by `engine.improve_rounds`, and sends a kept version back to `execute` once); the no-simulation path (when `state.no_simulation_resolved` is True, decided at `clarify` or via `engine.no_simulation: true` in YAML) is `auto_collect_data → wait_for_data → data_load → analyze`. Both converge at `analyze → cross_check → write → review`.

Three feedback loops: (a) `execute_reflect → execute` — fix a crashed script, bounded by `engine.exec_reflect_max_iterations`; (b) `cross_check → design` — `analyze.next_step == re_experiment` re-routes to design only when FI's own record shows the plan's precision target missed and a larger run fits the time allowed (`broaden_lit` re-enters `literature`); (c) `review → write` — the paper is written again for what the review found; a revise verdict or an objection to the design that no check of FI's confirms is kept in the quality report (`QUALITY.md`) and never changes the design (`review → design` only for a person's note the rewrite left unanswered). `review` can also go to `implement` (a computed number is wrong; re-run, a small fixed number of times) and to `human_feedback`; a person's refine goes `human_feedback → write`, and `write → design` only when the writer names a point that needs a new experiment; `evidence_gate → literature` broadens (bounded) and `evidence_gate → design` retries once when the run produced no results, except that a script that could not run to the end goes `evidence_gate → implement`: the same design's code is written again, once per design. Loops (b) and (c) share `engine.max_iterations` as the budget; loop (a) has its own counter so a broken script doesn't eat the design-iteration budget. `no_simulation_resolved` is sticky across re-entries, so a redesign on a no-sim quest goes back through `auto_collect_data`, not `implement`.

## A core requirement: rigorous architecture, simple surface

The checks behind a result (the frozen protocol, the engine-judged oracle, the run manifest, the evidence ladder, the
audit trace, `rigor_profile: research`) can be as strict as the science needs. **What a person sees while running FI
must not be.** The target user is a scientist or engineer who will not read this source and will not read a long doc
before their first run: a console line, a config key, a CLI flag, an error message, or a new term must be
understandable at a glance, in the words a domain scientist already uses, not in words this codebase invented for its
own convenience. When rigor and surface simplicity pull against each other, keep the rigor internal and simplify the
surface — do not simplify the check itself to make the surface easier. Concretely: `core/engine.py:STAGE_PROGRESS`
shows one plain sentence per stage while `run.log` keeps every internal detail; `fi tools <name>` moved the ~15
rarest CLI commands out of the default `--help` without removing or renaming a single flag; `core/evidence.py`'s
`summary_line()` defaults to a level's plain-language sentence, never the raw six-way identifier. Before adding a new
user-visible name (a config key, a CLI flag, a file name, a log line), prefer extending something that already exists
over inventing a new one, and prefer a plain question over a technical field name wherever a person has to answer it.

## How to fix, and how to bring a decision to the maintainer

- **Fix the root cause, not the instance.** When a check misjudges a case, ask where the wrong fact comes from and
  move its source, rather than adding a rule for that case. A check that guesses after the fact from free text or
  free code (which result a typed number belongs to, which setting a piece of code reads, what a caption's words
  mean) gets a new exception for every new wording and keeps failing on the next real input; the root fix is that
  FI produces the fact itself from its own records (as `{{seed}}`, `{{run_time}}` and `{{not_done:<id>}}` do in
  `core/paper_facts.py`), or that the heuristic is removed. If a fix is an added exception, say so plainly.
- **Decide trade-offs with a test, not by judgement.** Before choosing between approaches (stop vs go on, strict vs
  lenient, model-typed vs FI-filled), run them on the same cases: real inputs whose verdict is known (good) and the
  same inputs with one planted fault each (bad), scored on faults caught, false flags and false stops, against the
  project goal (no fabricated result; false stops at most 10%). When the model's behaviour decides it, run a small
  A/B with the real model and state the cost first.
- **Bring the maintainer results, not open questions.** A decision goes to the maintainer only after that test:
  each option with its measured gains and costs and a recommendation. Do not hand over a list of untested choices.
- **A fix counts as fixed only when its original failing input is shown fixed** (a replay of the real record or
  quest, or a live run), with a known positive showing the check still fires. Passing unit tests and a review are
  not that evidence. Only shown fixes go into a PR.

## Before building a new feature: check what already exists

`docs/capabilities-reference.md` is the exhaustive, user-facing catalogue of what FI does. `dev/registry.md` is the
contributor-facing index of which module or function already owns an area of the code. Read the relevant section of
both before adding a feature or a new file — a surprising amount of what looks missing is already built somewhere in
`core/`, just not where you first looked. Update `dev/registry.md` in the same PR as a feature that adds or moves a
module-level responsibility (a new `core/*.py` file, a new class, a new node) — it goes stale the moment a PR adds
something and forgets it, so keeping it current is part of finishing the feature, not a follow-up.

## The LLM wiki (`wiki/`)

The project keeps a compiled, cross-referenced wiki at `wiki/` (the Karpathy LLM-wiki pattern:
<https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>), maintained by the maintainer with an `llm-wiki` skill that is not part of this repo. The point is to **compile
understanding once and reuse it**, not to re-read the source every time. Without that skill, read the wiki the same
way, and when a page is missing fall back to `dev/registry.md`, `docs/capabilities-reference.md` and the source;
grow the wiki by hand (compiled pages, `[[slug|Title]]` links, an entry in `wiki/log.md`, never hand-editing
`wiki/index.md`). After adding or renaming a page, rebuild the index with `python scripts/wiki_sync_index.py`;
`--check` reports a stale index or a link to a page that does not exist.

- **Check the wiki first.** Before answering a question or starting a task, look for the relevant page in
  `wiki/index.md` and follow `[[wiki-link]]`s out to about two hops for context.
- **Cite it.** When a wiki page backs an answer, reference it as `[[slug|Page Title]]` (links resolve by file name).
- **Say when it's missing.** If the wiki has no answer, say so, then read `wiki/raw/` or the real source and grow the
  wiki with `llm-wiki ingest` — don't silently answer from a stale or absent page.
- **Compile over search.** Reuse compiled knowledge in `wiki/*.md` rather than re-reading the same source files
  every time; `dev/registry.md` (module-level index) and `docs/capabilities-reference.md` (user-facing feature
  index) are the other two places to check first, per the section above.

Structure: `wiki/index.md` (auto-rebuilt routing layer — never hand-edit), `wiki/{page}.md` (compiled pages,
cross-referenced), `wiki/raw/` (immutable source drop zone), `wiki/log.md` (ingest history). Commands (invoke via
`llm-wiki <command>`): `init`, `ingest <source>`, `query <question>`, `lint` (broken links / orphan pages), `sync`
(rebuild index.md — in this repo use `scripts/wiki_sync_index.py` instead: the skill's own `sync` writes a Korean
header that fails `--check`), `export <template>`, `qmd-index` / `lancedb-sync` (optional hybrid search / vector index).

## Run / develop

```bash
pip install -r requirements.txt
python -m pytest -v tests/test_<changed>.py                 # only the tests of what you changed (see Tests)
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
- **CLI exec** (`claude_cli`/`codex_cli`/`copilot_cli`/`gemini_cli`/`antigravity_cli`) — `LLMClient` spawns the local CLI binary per chat call via `asyncio.create_subprocess_exec`. No proxy. Reuses the CLI's own OAuth. Spawn details in `_CLI_SPECS`. `claude_cli`, `codex_cli`, `gemini_cli` and `antigravity_cli` are chat-style. **`copilot_cli` is agentic**: it interprets FI's node prompts as user coding tasks and replies conversationally; the engine emits a loud warning when it's selected. For Copilot use `vscode_extension` instead.
- **VSCode bridge** (`vscode_extension`). `VSCodeBridgeClient` sends newline-delimited JSON over a localhost TCP socket the FI VSCode extension spawned us with via `--vscode-bridge-port N`. The extension makes the actual `vscode.lm.selectChatModels` + `model.sendRequest` call and streams chunks back. 180 s inactivity timeout + 6-attempt Python-side retry (cumulative ~2 min backoff) protects against Copilot HTTP/2 stalls.

**Execution layer** (`core/execution.py`): `Executor` protocol with two implementations. `make_executor(sandbox, ...)` is the constructor used by `Engine`. `VenvExecutor` resolves Python at `Scripts/python.exe` on Windows and `bin/python` on POSIX. `DockerExecutor` mounts `<quest_root>` at `/work` with networking disabled, memory / CPU / process caps (`DockerLimits`, from `execution.docker_*`), every capability dropped, `no-new-privileges`, and a non-root user where the host allows one.

**Knowledge layer** (`core/knowledge.py`): one in-process `AxonBrain` per quest. Tolerates missing axon install (degrades to no-op + warning). Retrieval (`Knowledge.asearch`) is three-layer: pinned `local_papers` → Axon → external router (`arxiv` / `openalex` / `crossref` / `semantic_scholar` / `pubmed` / `core` / `google_scholar`, parallel + DOI-dedup). With `source_routing: auto` (default) the LLM picks 1–5 sources from a 12-entry catalog (`_SOURCE_CATALOG`); users extend the catalog by ingesting `kind=fi_source_catalog` entries into Axon. With `try_fetch_full_text: true` external hits get opportunistic publisher-PDF fetch via the host's existing network access (VPN/Shibboleth/EZproxy); login walls are rejected by a Content-Type + `%PDF-` magic-bytes two-factor check, so paywalled venues never hang a quest. Write-back (`add_quest_artifacts`) is **gated on `verdict == "accept"`** by default and lays down a *structured bundle*, NOT a flat doc: `fi_paper_spine` (single-chunk card-catalog entry — title/authors/DOI/abstract/key-claims — for title-searchability), `fi_quest_paper` (full body with a 1-line `[Title · Year · Venue · DOI]` citation header prepended to every chunk so retrieval hits are self-describing), `fi_quest_summary` (structured findings JSON; carries `paper_refs` short-id list in metadata), `fi_topic_event` (per-topic rollup pointer keyed by slug; lists the quest + the papers it cited), and `fi_external_ref_spine` × N (curated card-catalog entries for cited externals — only refs that contributed to an accepted quest persist). The `ideate` node reads spines + summaries for cross-quest memory.

**Dataset adapters** (`core/datasets/`): the no-simulation `auto_collect_data` node can additionally pull structured data via adapters listed in `engine.dataset_adapters`. The abstract base is `core/datasets/base.py::DatasetAdapter` — a single `async def search(self, query: str, *, top_k: int) -> list[DatasetRow]`. Shipped adapters: `worldbank.py` (country-indicator tables via the WorldBank Indicators API, stdlib `urllib`) and `wikipedia.py` (opensearch + page-summary lookups for qualitative topics). Adapter failures log a WARNING and return `[]`; `Engine._run_dataset_adapters` sums counts and keeps the quest moving. Adding a new adapter is a single-file change plus a registry entry; no graph edit.

## Test topics, live runs and reports to the maintainer

- **Public in English, internal in Chinese.** Everything committed is public, so it is written in English: code,
  comments, prompts, tests, docs, this file, commit messages and PR descriptions. Chinese stays outside the repository
  (handovers, plans and reports for the maintainer, and every reply to the maintainer, in Traditional Chinese). Inside
  the repository Chinese is allowed only as data: a quoted string or a code span (text a user may type, such as a
  regular expression for Chinese words), test fixtures, the historical record `docs/audits/`, and a `*.zh-TW.md`
  translation next to its English original (the English one is the authority).
  `tests/test_public_text_english.py` fails on anything else and names the file and line.
- **A test topic is only a test.** Real research topics used to validate FI (in live runs and replays) never shape what
  goes into the repo: code, prompts, unit tests, docs, commit messages and PR descriptions do not use that topic's words
  or carry a special case for it. Fixtures use neutral made-up topics, and every fix must be general.
- **Watch a live run while it runs.** A long real run is watched throughout (stops, errors, the size of the study and
  FI's own time estimate); a problem is reported with a proposed way forward as soon as it shows, not after the run ends.
- **Report outcomes, not process.** Say what is done, what is left, how much better it is, what it cost and its quality;
  say "not measured" for what was not measured, and never "verified" for what was not observed directly.
- **What the maintainer may not have seen — only when there is something.** After an answer, add at most three points,
  each one passing all three tests: it is new (not in the answer above, not something the maintainer said or decided);
  it is concrete (it names the run, file, number or rule); and it changes something (a result, a cost, an earlier
  decision, or it needs the maintainer's decision). Decisions come first, each with what every option changes and a
  recommendation; then gaps, risks and conflicts. Each point is one sentence of fact and one of what it changes or what
  is asked. Never progress news ("still running"), a repeat of the answer, reminders of rules, general advice or what is
  already handled. When no point passes, there is no such section; it is never filled to a count.
- **A handover every 48 hours of work.** What is done and what is open, what waits for the maintainer, the tests
  actually rerun (none rerun means saying so, not reusing an old number), and when the next one is due; it is a
  snapshot, the code and records stay the authority. It goes where the maintainer keeps handovers.

## Conventions and gotchas worth knowing

- **Tilde expansion is manual.** `Config` uses `field_validator(..., mode="before")` to expand `~` in path-shaped YAML fields (`output_dir`, `axon_config` when given as a path, every entry in `knowledge.local_papers`). When adding new path fields, mirror that pattern — pydantic does not expand `~`.
- **All node prompts live in `agents/*.md`** as Python `string.Template` files (`$placeholder`, not f-strings) so prompt content can contain literal `{...}` (e.g., JSON examples). Loaded once at engine init.
- **JSON parsing from LLMs is lenient.** `core.engine._parse_json_lenient` strips fences and finds the largest balanced `{...}`. When you add a new node, return JSON and feed it through this helper.
- **`RESULT_JSON: {...}` contract.** Generated experiment scripts emit a single line beginning with `RESULT_JSON: ` as the **last** line of stdout; `_extract_result_json` parses it back into state. This is the cheap, reliable channel for numerical findings to flow from the experiment to the analysis node.
- **Per-quest filesystem layout:** `<output_dir>/<quest_id>/{paper.md, paper.pdf, slides.*, poster.*, paper/, figures/, code/, data/, .fi/run.log, .fi/state.sqlite, .venv/, frontier_insight_summary.json}`. `quest_root` is the source of truth; generators read from it and write their outputs (`paper.pdf`, `slides.*`, `poster.*`) at its top level, while `paper/` holds `paper.md`, the claim ledger and the bibliography exports.
- **Async everywhere.** New code in `core/` should be `async def` unless it's pure CPU work; HTTP must use `httpx.AsyncClient`; subprocesses use `asyncio.create_subprocess_exec` (see `VenvExecutor`); Docker calls offload to `asyncio.to_thread` since `docker-py` is sync.
- **No process-global state.** Avoid module-level singletons; multiple Engines in one process must not collide. `ProxySupervisor` is the one piece of intentional shared state and is explicitly ref-counted.
- **Provider proxies have caveats.** `copilot-api` carries an explicit GitHub abuse-detection warning; FI defaults `--rate-limit 60 --wait`. `claude-code-openai-wrapper` has no PyPI package — `FI_CLAUDE_CODE_WRAPPER_DIR` points at a clone with `poetry install` already run.
- **Rust is deferred.** Don't add `maturin` / PyO3 / a `frontier_insight._fast` Rust crate until profiling identifies a real hot spot. The 2026 benchmark cited in `docs/plan.md` shows ~5% Rust win on LLM-network-bound work — within noise.
- **No PR-number / Phase-letter refs in user-facing docs** (`README.md`, `docs/**` outside `docs/audits/`, `vscode-frontier-insight/README.md`, and this file). Describe features in their own terms; keep "PR #N" / "Phase X" lineage in commit messages, PR descriptions, and `docs/audits/` (which IS the historical record).

## Before opening a PR

For coding agents working in this repo (human contributors: see CONTRIBUTING.md; these steps are recommended, not
merge requirements for you).

- **Docs pass first.** Re-check every doc the branch touches (README, `docs/**`, `vscode-frontier-insight/README.md`) for correctness and readability before opening, not as a follow-up — a doc edited three commits ago can go stale from a later commit in the same branch.
- **One external code review of the diff, always.** Every PR gets at least one review from a tool outside this session (Codex CLI, another Claude session, or `agy`) before it opens, even though it slows the PR down — that slowdown is accepted, not a reason to skip it. Write the diff/prompt to a file and have the reviewer produce a written report, not a verbal summary; read it and decide, finding by finding, what to fix.
- **A review finds bugs; it does not show a fix works.** The evidence that a fix works is its original failing input
  shown fixed (see "How to fix" above); a PR lists that evidence per fix.
- **Re-review after applying that review's fixes, before opening.** A fix for one finding can silently introduce or miss a new bug — a construction site the first fix didn't cover, a fix that only half-applies. Once every finding from the first pass is addressed, run a second, targeted pass that checks each changed file specifically against the finding it was meant to fix (a full fresh review is not required, but skipping this step is not an option). Applying a suggested fix is not the same claim as verifying the fix is correct.
- **Keep `dev/registry.md` current** in the same PR that adds, splits, or renames a module-level responsibility.

## Splitting work for small, fast models

Many fixes are small and mechanical. A coding agent that delegates work can hand those to a small, fast model (a
Haiku-class model) and run several at once, keeping a larger model for judgment calls, the engine's core flow in
`core/engine.py`, and changes that span modules. This works only when each task is small and fully specified.

- **One task, one place.** One file and one function where possible, at most the file plus its test file (and one
  named line elsewhere). Give each task its own git worktree and branch, created from the current branch head, so no
  two agents edit the same file.
- **Say everything the model needs.** The exact location (file, function, approximate line), the real failure it
  fixes (a concrete input and what goes wrong), the rule to implement in plain words, what must NOT change, the tests
  to add (including one that fails on the old code), the exact test command, and the commit message.
- **Name the tests of the callers too.** List not only the changed module's test file but the test files of every
  module that calls the changed function (find them with grep). A change that passes its own tests can break a
  consumer: a filter that hid a quantity's name from model input also hid the engine's own repair instruction, and
  only another module's tests caught it.
- **Pass means the whole file passes.** The task must say: run every listed test file in full, paste the final
  summary line, and treat any failure as "not done". Report anything done beyond the spec instead of doing it quietly.
- **Do not use `git stash`** in a task (the stash stack is shared across worktrees and sessions); to run a test on
  the old code, copy the file aside and restore it.
- **Verify before merging.** Read each diff, rerun the listed test files yourself, and cherry-pick only what passes.
  Reject a change whose rule is riskier than the problem it fixes (for example a rule that removes a loophole but
  stops honest runs). Run the full suite before any evaluation round or PR, whatever the small tasks reported.

## Tests

**Run only the tests of what you changed.** Locally, run the changed or new test files and the tests that call the
changed functions — never the whole suite (it takes about two hours on Windows and nearly every file imports
`core/engine.py`, so "related tests" grows to all of them). After merging branches, run only the tests of functions
the merge touched or conflicted on. CI runs the full suite on every PR.

Test files are mostly named after the module or feature they cover (`tests/test_<name>.py`; grep `tests/` to find them). `tests/test_engine_smoke.py` runs the full DAG with a fake LLM and a real venv — the regression detector for engine plumbing. `tests/test_docker_executor.py` gates real-daemon tests on `info["OSType"] == "linux"` so Windows-mode daemons skip. **Tests do not call real LLM APIs**; introduce new tests with `monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)` (or `Knowledge.asearch`) to keep that property.

**CI split** (`.github/workflows/ci.yml`): the workflow runs on pull requests only (a merge to main is not re-tested). The **fast tier** runs on both ubuntu-latest and windows-latest with `-m "not slow"` plus an explicit `--ignore` for the three e2e files, in parallel workers (`-n auto --dist loadfile`) and split into shards that run side by side (2 on ubuntu, 3 on Windows): `tests/conftest.py` keeps, for `FI_TEST_SHARD=k/n`, only the test files whose path hashes to shard `k`, so one OS's shards together run every fast-tier test once (unset, it does nothing — a local run is unchanged). Each shard uploads its coverage data; a `coverage floor` job per OS joins them and applies the 65% floor to the whole. The **slow tier** runs on ubuntu only, alongside the fast tier and also in parallel workers, and covers (a) the three e2e files (`test_engine_smoke.py`, `test_self_correction_e2e.py`, `test_web_e2e.py`) and (b) any `@pytest.mark.slow`-tagged test outside them; pytest rc=5 ("no tests collected") is treated as success there. Both tiers gate every code PR — the slow tier was historically post-merge only, but two slow-tier-only regressions slipped past in quick succession, so the gate moved to pre-merge. Exceptions: PRs whose title contains `[ext-only]` skip the Python fast and slow tiers (the `vscode extension compile` job covers them); and the top-level `paths-ignore: ["**.md", "docs/**", "outputs/**"]` filter skips the whole workflow for docs-only PRs.

`pytest.ini` redirects `tmp_path` to `./.pytest_tmp/` because the default Windows `%TEMP%` path was inaccessible to the harness sandbox. If a test fails on a fresh checkout, that path is the first thing to check. It also gives every test a 600 s limit (`timeout`, from `pytest-timeout`, a dev dependency in both manifests; the three e2e files mark themselves 1800 s), so a hung test fails with its name instead of stalling a shard. On Windows the limit ends the xdist worker running the test, so the fast tier passes `--max-worker-restart=0`: the shard stops and its summary names the test (with restarts, a replacement worker re-ran the hung test, and under `--dist loadfile` the run could end in an xdist INTERNALERROR that printed no name).

**Don't claim the VSCode extension is done without running BOTH `npm run compile` AND `npm run package`.** Two distinct steps; one silently missing the other has bitten this codebase twice already. `tests/test_vscode_extension_typescript.py` runs both from pytest as a structural check. The tests are gated on `node`+`npm`+`vscode-frontier-insight/node_modules` being present and skip silently otherwise, so any "extension shipping" message must explicitly state both were run and pin the resulting `.vsix` path. The tracked `.vsix` lives at `vscode-frontier-insight/vscode-frontier-insight.vsix` and is auto-rebuilt on main by `.github/workflows/vsix-rebuild.yml`; you only need to commit a rebuilt copy when iterating on a feature branch where users will install before main merges.

**Audit reports live in `docs/audits/`.** That folder IS the historical record — PR-number and Phase-letter references are fine there. The user-facing prose docs (README, `docs/USAGE.md`, `docs/capabilities.md`, `docs/architecture.md`, `vscode-frontier-insight/README.md`) describe behaviour in its own terms and never reference PRs or phases.
