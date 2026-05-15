# 07 — Test Reliability Audit

Date: 2026-05-15 · Branch: `main` (c5acf90) · Scope: `tests/`, `pytest.ini`, `.github/workflows/`, related core hooks.

The brief listed `tests/` at 29,046 LOC across 37 files. Re-counted in this worktree (`wc -l tests/*.py`): **14,523 LOC** across 37 files (`__init__.py` empty). The 29k figure is roughly 2× the actual count; the rest of this audit uses the measured 14,523. Test count is correct: **620 collected** (`pytest --collect-only` finishes in 1.80 s).

## Findings

### Hard numbers — top-10 largest test files (LOC)

| Rank | File | LOC | # tests |
|------|------|-----|---------|
| 1 | `tests/test_engine_helpers.py` | 2242 | 115 |
| 2 | `tests/test_knowledge.py` | 1115 | 48 |
| 3 | `tests/test_paper_gen.py` | 798 | 26 |
| 4 | `tests/test_vscode_bridge.py` | 680 | 18 |
| 5 | `tests/test_provider.py` | 617 | 28 |
| 6 | `tests/test_digest.py` | 615 | 32 |
| 7 | `tests/test_provider_cli.py` | 602 | 27 |
| 8 | `tests/test_summarizer.py` | 535 | 36 |
| 9 | `tests/test_launch.py` | 522 | 32 |
| 10 | `tests/test_slides_speech.py` | 428 | 15 |

`test_engine_helpers.py` is 2× the size of the next-largest file. It is internally well-organized — 15 `# ---- section ----` banners (file:line 32, 86, 206, 242, 267, 303, 332, 387, 464, 601, 634, 670, 788, 1187, 1407, 1651). Each banner is a candidate for a separate module. The brief flagged this; data confirms it.

### Hard numbers — top tests by wall-clock duration (measured)

Profiled in batches with `--durations=15`/`20`. Numbers are single-run on Windows 11, no parallelism.

| Test | Duration | File |
|------|----------|------|
| `test_full_dag_with_execute_repair_loop` | 63.31 s | `tests/test_self_correction_e2e.py` |
| `test_two_engines_run_concurrently` | 57.71 s | `tests/test_fleet.py` |
| `test_full_dag_with_analyze_re_experiment_reroute` | 53.64 s | `tests/test_self_correction_e2e.py` |
| `test_full_dag_via_server_with_clarify_pause_and_resume` | 51.64 s | `tests/test_web_e2e.py` |
| `test_engine_runs_with_fake_llm` | 49.55 s | `tests/test_engine_smoke.py` |
| `test_quest_writeback_runs_on_revise_when_accept_gate_off` | 32.02 s | `tests/test_knowledge_writeback.py` |
| `test_quest_writeback_skipped_when_verdict_revise_and_accept_gate_on` | 31.89 s | `tests/test_knowledge_writeback.py` |
| `test_quest_writeback_invokes_axon_when_enabled` | 30.99 s | `tests/test_knowledge_writeback.py` |
| `test_engine_runs_with_clarify_interactive_via_callback` | 22.98 s | `tests/test_engine_smoke.py` |
| `test_engine_runs_with_clarify_auto` | 13.95 s | `tests/test_engine_smoke.py` |
| `test_chat_error_note_omits_node_when_unset` | 8.03 s | `tests/test_provider.py` |
| `test_chat_error_note_uses_model_override_when_passed` | 8.03 s | `tests/test_provider.py` |
| `test_chat_propagates_http_error` | 8.02 s | `tests/test_provider.py` |
| `test_chat_error_includes_provider_and_node_in_note` | 8.01 s | `tests/test_provider.py` |
| `test_slides_marp_renders_html_and_pdf` | 3.08 s | `tests/test_slides_speech.py` |
| `test_bridge_connect_to_dead_port_raises_immediately` | 2.05 s | `tests/test_vscode_bridge.py` |

The fifteen slowest tests above sum to **~445 s = 7.4 min**, against a full-suite budget of ~11 min. **15 of 620 tests = 2.4 % of the test corpus consume ~67 % of the wall time.** Everything else is fast: `test_engine_helpers.py`'s 115 tests run in 0.99 s total; the digest/critique/portfolio/proposal/summarizer/config bundle of 129 tests runs in 4.89 s.

Three distinct cost categories explain the heavy tail:

1. **End-to-end DAG runs that build a real venv** — `test_self_correction_e2e.py`, `test_engine_smoke.py`, `test_web_e2e.py`, `test_fleet.py`, `test_knowledge_writeback.py`. `VenvExecutor.setup` (`core/execution.py:60-66`) calls `venv.EnvBuilder` synchronously on a background thread; on Windows this is ~10-20 s for the venv plus 5-15 s pip-installs. Five of the top eight are this pattern. There is no caching of the venv across tests in the same session — every `tmp_path` gets its own.
2. **Tenacity retry waits with real `wait_exponential`** — the four `test_chat_*_error_*` tests in `test_provider.py:323-496`. Each forces `LLMClient.chat` to fail and waits for 4 attempts with `wait_exponential(multiplier=1, min=2, max=20)` (`core/provider.py:887-888, 1011-1012`). 2+4 s of forced sleep per test = 8 s × 4 = 32 s of pure idle wait. None of these tests monkeypatch `wait_exponential` to zero.
3. **Real external tool invocation** — `test_slides_marp_renders_html_and_pdf` (3.08 s) invokes the `marp` CLI via subprocess.

### Failing tests on `main`

Two tests fail on `main` today, with the same root cause as the regressions PR #63 was supposed to clean up:

- `tests/test_engine_smoke.py::test_engine_runs_with_clarify_auto` (file:line 173-208) — assigns `empirical_vs_theoretical=empirical` as a clarify slot default, which the β/γ legacy-fallback in `_node_clarify` now interprets as a no-simulation signal → quest pauses, no `paper.md`, assertion fails.
- `tests/test_engine_smoke.py::test_engine_runs_with_clarify_interactive_via_callback` (file:line 211-256) — same cause; the callback hands back `"empirical_vs_theoretical": "empirical"` (line 244) and the no-sim pause path triggers.

The captured stderr is unambiguous: `[clarify] simulatability resolved: NO_SIMULATION (source=clarify_empirical_legacy, reason='empirical_vs_theoretical=empirical, simulatability slot missing')`. Identical mechanism to the two tests fixed in PR #63 (`test_clarify_off_skips_node`, `test_full_dag_via_server_with_clarify_pause_and_resume`) — that PR missed two siblings in `test_engine_smoke.py` because it pattern-matched on file name rather than on the failing predicate.

### Coverage heuristic — `core/` LOC vs. matched test LOC

| Module | core LOC | Direct test file LOC | Heuristic floor (core/10) | Status |
|--------|----------|----------------------|---------------------------|--------|
| `core/engine.py` | 2685 | — (split across 10 files, 4475 LOC) | 268 | OK (split) |
| `core/knowledge.py` | 1657 | 1115 | 165 | OK |
| `core/provider.py` | 1030 | 617 (+ 602 in `test_provider_cli.py`) | 103 | OK |
| `core/digest.py` | 816 | 615 | 81 | OK |
| `core/summarizer.py` | 563 | 535 | 56 | OK |
| `core/portfolio.py` | 373 | 374 | 37 | OK |
| `core/critique.py` | 354 | 351 | 35 | OK |
| `core/config.py` | 344 | 202 | 34 | OK |
| `core/proposal.py` | 283 | 216 | 28 | OK |
| `core/vscode_bridge.py` | 281 | 680 | 28 | OK |
| `core/execution.py` | 257 | 51 | 25 | **OK by LOC, weak by coverage** |

Every module clears the LOC/10 floor. The brief specifically flagged `critique.py`, `portfolio.py`, `proposal.py`, `digest.py`; all four have tests at or above their source size (`portfolio.py` 374/373, `critique.py` 351/354, `proposal.py` 216/283, `digest.py` 615/816). The real coverage gap is `tests/test_execution.py` — 51 LOC, 3 tests, all of which exercise `VenvExecutor`. `DockerExecutor` (134 LOC starting `core/execution.py:123`) is covered separately in `tests/test_docker_executor.py` (327 LOC, 17 tests), so on aggregate `core/execution.py` is fine. What is **not** tested in `test_execution.py`:

- `VenvExecutor.install()` (lines 74-82). No direct test asserts pip-install behavior.
- The `_build_venv` async-offload path under failure (e.g. PermissionError mid-venv-creation).
- Concurrent `VenvExecutor.setup()` on the same path (no documented contract; the fleet test indirectly exercises it but doesn't assert ordering).

### Windows file-lock pattern — anywhere else?

The `_close_quest_logger` fix in `core/engine.py:2660-2685` is correct and the outer `try/finally` in `Engine.run` (`core/engine.py:316-324`) guarantees it fires on every exit path including the Phase B no-simulation pause. Five dedicated regression tests pin the contract (`tests/test_engine_helpers.py:670-786`, sections `_quest_logger lifecycle`).

Searching the rest of `core/` for analogous handle-leaking patterns:

- `core/provider.py:507-511` — `tempfile.NamedTemporaryFile(prefix="fi_cli_out_", delete=False)` for `--output-last-message`. Cleanup is in a single `try/finally` (lines 533-606) that unlinks `tmp_out_path` on every exit path including spawn failure. Comment at line 530 explicitly calls out that an earlier revision leaked on `FileNotFoundError`. **Safe.** Regression test exists (`test_cli_tmpfile_cleaned_up_on_spawn_failure` in `test_provider_cli.py`).
- `core/execution.py:60-66, 84-115` — `VenvExecutor` shells out via `asyncio.create_subprocess_exec`. Stdout/stderr are `PIPE`d and read via `communicate()`, which closes the pipes; the timeout path (`proc.kill()` then `await proc.communicate()` at line 107) drains both pipes too. **Safe.**
- `core/datasets/wikipedia.py:64`, `core/datasets/worldbank.py:107` — both wrap `urllib.request.urlopen(...)` in `with` blocks. **Safe.**
- `core/knowledge.py` — only context-managed `read_text` calls at lines 466 and 1564. The Axon brain handle (`self._brain`) lives for process lifetime; tests construct `Knowledge(enabled=False)` to avoid building it (`tests/test_knowledge.py:30, 37, 45, 53, ...`). **Safe by construction.**
- `core/digest.py`, `core/summarizer.py`, `core/portfolio.py`, `core/critique.py`, `core/proposal.py` — all writes go through `Path.write_text`. No long-lived handles. **Safe.**

The audit found **no other file-handle leak comparable to the pre-PR-#56 logger leak**. The Windows tmp_path `PermissionError` cascades the brief mentions almost certainly originated entirely in the logger. The two failing engine_smoke tests above are a separate (no-simulation routing) issue.

One latent risk: `tests/test_engine_helpers.py:733` explicitly `shutil.rmtree(tmp_path / "run1")` to test re-creating a quest_id. If that test ever fails mid-way (before reaching line 758's `_close_quest_logger(qid)`), pytest's `tmp_path` autoclean will fail on Windows. The test currently passes reliably, but moving the close to a `try/finally` (or using a fixture) would harden it.

### CI absence

`.github/workflows/` contains exactly one file: `vsix-rebuild.yml`. It triggers on `push` to `main` for paths under `vscode-frontier-insight/` and rebuilds the tracked `.vsix`. **No workflow runs pytest at any point** — not on PR, not on push to main. The brief is correct that this is how β/γ-style regressions slip through: the unit tests for PRs #57, #59, #60 each touched different files than the two tests that subsequently broke, so a per-file PR run wouldn't have caught them either; only a full `pytest tests/` would have. With no full run, the breakage only surfaces when a contributor manually invokes pytest after the merges land.

### Mock hygiene — real-init slowdowns

The `_make_no_sim_engine` helper at `tests/test_engine_helpers.py:1414-1446` documents the rationale: `Knowledge(KnowledgeConfig(enabled=False))` to skip the 15 s embedding load, then replace `engine.knowledge` with a `MagicMock`. The same pattern reappears across nearly all `Knowledge`-touching tests (`test_knowledge.py:30, 37, 45, 53, 98, 263, 310, 331, 772, 1088` all use `enabled=False`). That hygiene is good.

Other places where real init slows tests:

- **`VenvExecutor.setup`** — already covered. Five end-to-end tests pay this cost. `tests/test_execution.py:12-14` uses a `scope="module"` fixture to share one venv across the three execution tests; the engine_smoke / web_e2e / self_correction_e2e tests do **not** share, even though many of them only need "a venv with matplotlib". A session-scoped venv fixture would clip ~50-100 s off the suite total.
- **Real `tenacity.wait_exponential`** — the four 8-second `test_chat_*` tests. None of them monkeypatches the wait. A `pytest_collection_modifyitems` hook or a session-scoped `monkeypatch_session` fixture that swaps `wait_exponential` to `wait_fixed(0)` during the test phase would reclaim ~32 s.
- **`marp` CLI subprocess** — `tests/test_slides_speech.py:193` skips when marp isn't on PATH, but when present it takes 3 s per render.

### Pytest markers

`pytest.ini` content (full):

```
[pytest]
asyncio_mode = auto
testpaths = tests
addopts = --basetemp=./.pytest_tmp
```

There is **no marker partition**. Of 180 `@pytest.mark.*` decorators across 32 files, only 10 are `skipif`s for environmental gates (npm absent, marp absent, etc.) — the remainder are explicit `@pytest.mark.asyncio` (167 occurrences in 29 files), which is redundant given `asyncio_mode = auto` but harmless.

No `slow`, `integration`, `e2e`, or `network` markers exist. A `pytest -m "not slow"` invocation today is a no-op. With markers added to the 15 slowest tests, a "fast slice" for PR CI would cut the suite from 11 min to **under 2 min**.

### Test naming + organization

Test files map cleanly onto core modules. Naming convention `tests/test_<core_module>.py` is followed without exception. Engine is the only module whose tests span multiple files (helpers, resume, smoke, execute_retry, plus the per-feature files clarify/cross_check/ideate_reflect/execute_reflect/self_correction_e2e/per_node_model_routing). That split is logical — each feature file maps to one phase letter — and the helpers file remains the catch-all that has grown unwieldy.

Async-test annotation is consistent within files but inconsistent across files: `test_provider.py` has 22 `async def test_` functions and 0 explicit `@pytest.mark.asyncio` markers (relies on `asyncio_mode = auto`); `test_provider_cli.py` has 14 async tests and 14 explicit markers (belt-and-braces). Both styles work; pick one.

Three files have suspiciously low test counts relative to their LOC:

- `tests/test_fleet.py` — 121 LOC, 1 test. The one test (`test_two_engines_run_concurrently`) is the 2nd-slowest in the suite (57.71 s) and exercises a fleet end-to-end. Fleet coverage is light.
- `tests/test_web_e2e.py` — 175 LOC, 1 test (51.64 s).
- `tests/test_self_correction_e2e.py` — 235 LOC, 2 tests (sums to 116.95 s).
- `tests/test_platform.py` — 15 LOC, 1 test.

These five tests (one per file × 3 files plus self_correction's two) account for ~280 s = ~25 % of the full suite — a single low-throughput e2e per file pinning a heavy DAG path.

## Recommendations

1. **[high impact] [low effort] Define a `slow` marker and partition the suite.** Add to `pytest.ini`:

   ```ini
   markers =
       slow: tests >1s on a developer laptop; excluded from PR fast-CI
       e2e: full-DAG end-to-end runs that build a venv
   ```

   Mark the 15 named tests above with `@pytest.mark.slow` (and add `e2e` to the eight that build a venv). Then `pytest -m "not slow"` runs ~605 tests in under 2 min and is the PR-CI default.

2. **[high impact] [low effort] Add a pytest CI workflow.** Minimal proposal as `.github/workflows/ci.yml`:

   ```yaml
   name: pytest
   on:
     pull_request:
     push:
       branches: [main]
   jobs:
     fast:
       strategy:
         fail-fast: false
         matrix:
           os: [ubuntu-latest, windows-latest]
       runs-on: ${{ matrix.os }}
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "3.11" }
         - run: pip install -e ".[dev]"
         - run: pytest -m "not slow" -q --maxfail=5
     slow:
       runs-on: ubuntu-latest
       needs: fast
       if: github.event_name == 'push'
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "3.11" }
         - run: pip install -e ".[dev]"
         - run: pytest -m "slow" -q
   ```

   PRs run the fast slice on Windows + Ubuntu (~2 min × 2 = parallel ~2 min wall). The slow tier runs only on main-push so β/γ-style regressions are caught within minutes of merge. The Windows matrix slot is non-negotiable — the project's known file-lock bugs are Windows-only.

3. **[high impact] [low effort] Fix the two broken `test_engine_smoke.py` tests.** Same mechanical change as PR #63: in `test_engine_runs_with_clarify_auto` (`tests/test_engine_smoke.py:173-208`), change the `empirical_vs_theoretical` default in the `clarify` fake response (line 51, currently `"empirical"`) to `"theoretical"`, or — preferred — add an explicit `simulatability: "yes"` slot. Same fix at line 244 for the interactive callback. Today these fail on every full-suite run and degrade signal.

4. **[medium impact] [low effort] Zero out `tenacity.wait_exponential` for the test session.** Add a session-scoped autouse fixture in `tests/conftest.py`:

   ```python
   @pytest.fixture(autouse=True, scope="session")
   def _no_retry_wait(monkeypatch_session):
       import tenacity
       monkeypatch_session.setattr(tenacity, "wait_exponential",
                                   lambda **kw: tenacity.wait_fixed(0))
   ```

   (Requires `pytest-monkeypatch-session` or the project's own session-monkeypatch helper — three lines.) Reclaims ~32 s on each suite run. Does **not** break the four `test_chat_*` tests — they assert on the post-failure exception notes, not on the wait duration.

5. **[medium impact] [medium effort] Session-share a venv for fake-LLM e2e tests.** A `scope="session"` fixture in a new `tests/conftest_venv.py` builds one venv with `matplotlib` preinstalled and yields its `python_path`. The five end-to-end fake-LLM tests (`test_engine_runs_with_fake_llm`, `test_engine_runs_with_clarify_auto`, `test_engine_runs_with_clarify_interactive_via_callback`, `test_quest_writeback_*`, `test_full_dag_via_server_with_clarify_pause_and_resume`) reuse it. Saves ~50-100 s if each currently re-bootstraps. The pattern already works in `tests/test_execution.py:12-14` (module-scoped); generalize it.

6. **[medium impact] [medium effort] Split `tests/test_engine_helpers.py`** along its existing `# ---- section ----` banners. Suggested split:

   - `tests/test_engine_parsing.py` — `_parse_json_lenient`, `_extract_result_json`, `_strip_outer_fence`, `_parse_implement_response` (lines 32-600).
   - `tests/test_engine_routing.py` — `_route_after_review`, graph topology (lines 332-463).
   - `tests/test_engine_logger.py` — `_quest_logger` lifecycle (lines 670-786).
   - `tests/test_engine_no_sim.py` — no-simulation helpers (lines 788-1186) — already a coherent block.
   - `tests/test_engine_pdf_preflight.py` — paper.pdf preflight (lines 1187-1406).
   - `tests/test_engine_auto_collect.py` — `_node_auto_collect_data` + dataset adapters (lines 1407+).

   Keeps the existing test count and coverage; each new file is ≤400 LOC and maps to a discoverable engine surface.

7. **[low impact] [low effort] Drop redundant `@pytest.mark.asyncio` decorators.** Project-wide `asyncio_mode = auto` makes them no-ops. 167 lines saved. Pick the file with the strictest convention (`test_provider_cli.py`) and either remove decorators everywhere or add them everywhere; today the split is arbitrary.

8. **[low impact] [low effort] Harden `test_quest_logger_can_be_reopened_after_dir_recreate`** (`tests/test_engine_helpers.py:714-758`) by moving the final `_close_quest_logger(qid)` into a `try/finally`. If the test ever fails between `_quest_logger(qid, fi_dir_2)` and the close, the FileHandler leaks and `.pytest_tmp` cleanup fails on Windows — the exact failure mode the test is designed to prevent.

9. **[low impact] [low effort] Add an `e2e` skip-on-Windows flag for tests whose `tmp_path` teardown is fragile.** The brief mentions "`PermissionError` on Windows tmp_path teardown" for e2e tests. With recommendation #1 in place, tag those tests `@pytest.mark.skipif(sys.platform == "win32" and os.environ.get("CI"), reason="WinError 32 on PR-CI tmp_path teardown — runs in nightly slow tier on Ubuntu")` until the underlying race is fixed.

10. **[low impact] [medium effort] Add `pytest-xdist` for parallel execution.** With markers in place, `pytest -n auto -m "not slow"` would push fast-tier wall time below 1 min on a 4-core CI runner. Caveat: a few tests (logger lifecycle, anything that monkeypatches `core.engine` globals) are not parallel-safe today and would need `@pytest.mark.serial`.

## References

- `tests/` — 37 files, 14,523 LOC, 620 collected tests.
- `tests/test_engine_helpers.py` — 2242 LOC, 115 tests, 15 sections.
- `tests/test_engine_smoke.py:173-208` (`test_engine_runs_with_clarify_auto`) — failing on `main`.
- `tests/test_engine_smoke.py:211-256` (`test_engine_runs_with_clarify_interactive_via_callback`) — failing on `main`.
- `tests/test_provider.py:323-496` — four 8-second tenacity-wait tests.
- `tests/test_self_correction_e2e.py`, `tests/test_fleet.py`, `tests/test_web_e2e.py`, `tests/test_knowledge_writeback.py` — venv-building e2e tests.
- `tests/test_engine_helpers.py:670-786` — `_quest_logger` lifecycle regression tests.
- `tests/test_engine_helpers.py:1414-1446` (`_make_no_sim_engine`) — mock-hygiene pattern.
- `tests/test_execution.py:12-14` — existing module-scoped venv fixture pattern.
- `tests/conftest.py` — repo-root sys.path bootstrap only; no shared fixtures.
- `pytest.ini` — `asyncio_mode = auto`, `basetemp = ./.pytest_tmp`, no markers defined.
- `pyproject.toml:72-78` — dev extras include `pytest`, `pytest-asyncio`, `pytest-mock`, `pytest-cov`, `pytest-testmon`.
- `.github/workflows/vsix-rebuild.yml` — the only workflow; no pytest invocation anywhere in `.github/`.
- `core/engine.py:2660-2685` (`_close_quest_logger`) — the PR #56 fix.
- `core/engine.py:316-324` — outer `try/finally` in `Engine.run` that guarantees the close fires on every exit path.
- `core/engine.py:2596-2657` (`_quest_logger`) — reuse-or-rebuild logic that prevents stale handlers across re-runs.
- `core/provider.py:505-606` — `tempfile.NamedTemporaryFile` cleanup in `_run_cli_provider`.
- `core/provider.py:887, 1011` — `wait_exponential(multiplier=1, min=2, max=20)`, the source of the 8-second test waits.
- `core/execution.py:60-115` — `VenvExecutor` setup + execute; main source of e2e test latency.
