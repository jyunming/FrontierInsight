# 01 - Engine Architecture Audit

**Date:** 2026-05-15
**Scope:** `core/engine.py` (2,685 LOC)
**Companion audits:** see siblings in `outputs/audit_2026-05-15/`

`core/engine.py` is the spine of FrontierInsight: one file holds the `Engine`
orchestrator class, the LangGraph DAG topology, 14 `_node_*` coroutines, the
`QuestState` TypedDict, and ~16 module-level helpers (prompt loading, JSON
recovery, persona aggregation, slugification, logging plumbing, provider
warnings, dataset rendering, dep parsing). Six feature PRs (#56-#61) landed in
the last two weeks without any restructuring pass. The file works, but it has
crossed the threshold where its growth is now hurting refactoring confidence,
test isolation, and time-to-comprehension for new contributors.

This audit ranks the structural issues that matter and the cheap refactors
that buy back the most maintainability per unit of effort.

---

## Findings

Findings are ranked by impact-on-future-velocity, not by line count.

### 1. The file is doing four jobs and one of them is a graph definition

`core/engine.py:1-2685` mixes four abstraction levels: (a) the `Engine`
class and its checkpointed run loop, (b) the LangGraph topology + four
routing predicates, (c) 14 async node bodies, and (d) ~16 free helpers
that don't touch `Engine` state. The free helpers alone are ~880 LOC
(`core/engine.py:1806-2685`) and already tested independently
(`tests/test_engine_helpers.py`, 2,242 lines). PRs #60 (Axon
auto-collect) and #61 (dataset adapters) added node bodies plus their
own helpers without splitting the file; net additions across #56-#61
are ~700 LOC over two weeks. This is the headline finding — every
other recommendation below is downstream of one file owning four
abstraction levels.

### 2. `QuestState` has 26 fields, three of which are write-only

`core/engine.py:63-124` defines `QuestState` as a flat 26-field TypedDict
spanning seven phases (clarify / ideate / literature / design / implement
/ execute / analyze / cross_check / write / review / no-simulation /
data-load / panel-review). Grep confirms three fields are written but
never read by any downstream node:

- `auto_collected_count` (`engine.py:124`, written at 768 and 809, never
  consumed). The docstring at `engine.py:117-124` says it "lets the user
  see in run.log / state how much of their data load came from the agent"
  — it's purely observational telemetry leaking into the state contract.
- `data_files` (`engine.py:116`, written at 992, 1038, 1056; `data_load`
  at 1010-1011 explicitly says "Re-walk the dir on every invocation
  rather than trusting `state['data_files']`"). The field exists only
  so users can inspect the resume payload, but downstream code distrusts
  it by design.
- `review_panel` (`engine.py:103`, written at 1546, never read by any
  later node — `_route_after_review` reads `review`, not `review_panel`).

Beyond the dead writes, the state has no grouping. `clarify_questions /
clarify_answers / clarify_done / no_simulation_resolved` are four clarify
fields; `exec_result / exec_reflect_iter / exec_reflect_history /
exec_give_up_reason` are four execute fields; `review / review_panel`
are two review fields. Grouping these into nested TypedDicts
(`ClarifyState`, `ExecState`, `ReviewState`) would make `_build_graph`
read like a phase list and make checkpoint resumes self-documenting.

### 3. `Engine.run` is a 160-line state machine masquerading as a method

`core/engine.py:168-324` runs the entire quest. It has eight
responsibilities visibly interleaved:

1. directory setup + logger init (`193-197`)
2. paper-pdf preflight (`205`)
3. executor setup (`206`)
4. provider endpoint resolution (`208-213`)
5. SQLite checkpointer + graph build (`215-218`)
6. resume detection (`233-244`)
7. interrupt loop with TWO interrupt kinds dispatched in one `while True`
   (`246-290`)
8. artifact collection + write-back + cleanup (`291-324`)

Concurrent `try/finally` blocks at `292-294` (client/proxy release) and
`315-324` (logger close) wrap an inner block that itself contains another
`try` (`216-294`). This is the hardest method in the repo to mentally
diff, and PR #56's logger-close fix had to thread a `data_paused` flag
through all of it to keep cleanup semantics correct on the pause path.
Future pause paths (Phase C, slide regeneration, multi-stage data
acquisition) will keep accreting boolean flags here unless the loop body
becomes a real state machine.

The two interrupt kinds (`clarify` vs `wait_for_data`) are dispatched
on a single dict key check (`intr_value.get("data_required")` at line
262) — adding a third interrupt type means adding another `elif` branch
inside an already-busy `while True`.

### 4. `_node_review` is 125 lines and contains a second pipeline

`core/engine.py:1437-1558` is the longest node. The single-reviewer
legacy path (1471-1486) is 16 lines; everything from 1488 onward is a
distinct mini-pipeline: persona loading, parallel `asyncio.gather` for
N persona reviewers, deterministic aggregator call, moderator LLM call,
prose-vs-numeric merge, iteration bumping. The inner `async def
run_persona` (1491-1511) closes over `base_prompt` and is defined inside
the node — it can never be tested in isolation. The panel/single branch
is also where `node` strings get hierarchical (`review_panel.<name>` at
1501), which the `_model_for_node` helper (1590-1609) has to special-case
with a dotted-prefix fallback. Splitting the panel path into a sibling
`_run_review_panel` method (or `core/engine_review.py` module) would
recover the legacy path's clarity and make `run_persona` reachable from
unit tests.

### 5. Six near-identical "LLM call + lenient parse + fallback" sites

`_parse_json_lenient(text) or {...fallback...}` appears 11 times in the
file (`core/engine.py:501, 650, 666, 728, 1049, 1252, 1325, 1377, 1474,
1502, 1528`). Six of those follow the exact same shape:

```
text = await self._chat(prompt, node="<n>")
parsed = _parse_json_lenient(text) or {<dummy>: "(parse failed)"}
```

This is begging for an `_llm_json(prompt, *, node, fallback)` helper on
`Engine`. The function would centralize: prompt substitution call, the
chat call, lenient parse, fallback application, and structured error
logging when the model emits invalid JSON. Today, error logging differs
per call site (`clarify` warns, `design` returns a string with
`"(parse failed)"`, `analyze` returns a list, `ideate` swallows silently
because it tolerates empty parsed dicts).

This refactor would also fold in the `_strip_outer_fence` call that
several sites duplicate via `_parse_json_lenient` internally — the parse
function does the strip, but `_node_write` (1431) does it again because
markdown output is not JSON. A clean `_llm_text` / `_llm_json`
distinction would make that intent explicit.

### 6. The four routing predicates belong in their own module

`_route_after_design` (403-423), `_route_after_execute_reflect`
(425-441), `_route_after_cross_check` (443-455), and `_route_after_review`
(457-464) are pure functions of `state` plus a few `self.config.engine`
fields. They read like a router table but are buried under a class. They
would be trivially testable as module-level functions taking
`(state, config)` and would shrink `Engine` by ~65 lines. They also
collectively encode the iteration-budget rules in three different
places (439, 453, 462) — the budget invariant should live in one
function.

### 7. `_resolve_no_simulation_from_clarify` is a 92-line precedence ladder inside Engine

`core/engine.py:542-632` implements the YAML-vs-clarify-vs-legacy
precedence for the no-simulation flag with extensive INFO/WARNING
logging. Nothing in it depends on `Engine` except `self.config` and
`self._log`. It's the largest method on the class after `_node_review`
and `_node_execute`. Extracting it into a free function taking
`(answers, config, logger)` would:

- isolate the decision logic for direct unit testing (today
  `tests/test_clarify.py` exercises it through the whole `_node_clarify`
  path),
- make the WARNING-on-unrecognized-decision branch (`engine.py:607-614`)
  reusable for future clarify slots,
- reduce `Engine`'s line count by ~90 lines without behavior change.

### 8. `_node_execute` carries a 60-line Windows DLL-race workaround mid-flight

`core/engine.py:1101-1205` has two distinct concerns: (a) run the
experiment script and shape an `exec_result` patch, and (b) work around
a Windows-specific venv-warmup + fast-fail-retry race that PR #20
documented. The warmup logic (1115-1153) and the retry logic (1161-1185)
are ~70 lines of platform-conditional code that does not need to live
on the node body — it belongs in `core/execution.py` as a wrapper
around `executor.execute`, where it would be exercised independently of
the engine's quest-state plumbing. The node body proper is ~30 lines
and currently obscured by the workaround.

### 9. Hidden import-time module work

`_load_prompts` (1809-1821) reads 12 prompt files from disk at every
`Engine.__init__`. There's no caching; the fleet runner instantiating
N engines pays N file-reads × 12 prompts. The two lazy imports inside
methods — `from generation.paper import PaperGenerator` at 1640 inside
`_preflight_paper_pdf`, and `from .datasets import ADAPTER_REGISTRY` at
881 inside `_run_dataset_adapters` — are good (they avoid pulling
generation/* into the engine module just for the preflight). But the
top-of-file imports already pull `langgraph.checkpoint.sqlite.aio`,
`langgraph.graph`, `langgraph.types`, plus the local `Knowledge`,
`make_executor`, and provider modules unconditionally. A test that
only wants to exercise `_parse_json_lenient` still has to bring up the
whole LangGraph runtime.

### 10. Two module-level singletons leak across engines

`_PROXY_WARN_SHOWN` (line 2371) is module-level `set[str]` suppressing
the unsanctioned-provider warning to once-per-process — process-global
mutable state in a module whose docstring (lines 7-8) explicitly
promises "N instances must coexist in one process for the fleet
runner." `tests/test_engine_resume.py:200` imports the set to reset it
between cases, which is the classic smell of tests reaching into
internal state to undo a singleton. An instance attribute on `Engine`
would scope the suppression correctly and remove the hidden coupling.

The other singleton is the `frontier_insight.<quest_id>` logger
(`_quest_logger` at 2599-2657), which the file itself documents at
length (2602-2625) as a global the file has to manage carefully on
Windows. That one is harder to remove (it's how Python's `logging`
works) but the management code is a 60-line monument to a global the
code wishes it didn't have.

---

## Recommendations

Each recommendation is tagged with `[impact: S/M/L]` (how much it
improves future-feature velocity / test isolation / blast radius) and
`[effort: S/M/L]` (rough size of the patch). Listed by best
impact-per-effort ratio first.

### R1. Extract module-level helpers into `core/engine_helpers.py` `[impact: M][effort: S]`

Move the ~16 module-level free functions (`_parse_json_lenient`,
`_load_persona_prefix`, `_aggregate_panel_reviews`, `_format_*`,
`_strip_outer_fence`, `_default_clarify_questions`,
`_render_auto_collected_md`, `_render_data_readme`,
`_list_user_data_files`, `_parse_implement_response`, `_coerce_dep_list`,
`_extract_result_json`, `_deps_to_warmup_modules`, `_slugify`,
`_new_quest_id`, `_warn_if_unsanctioned_provider`, `_quest_logger`,
`_close_quest_logger`) out of `engine.py` and into a sibling
`core/engine_helpers.py`. Re-export them from `engine.py` so
`tests/test_engine_helpers.py` keeps working without import
churn. Engine.py drops from 2,685 LOC to ~2,050 LOC; the test file's
imports become more truthful (today it imports "helpers" from a
2,685-line file that is overwhelmingly NOT helpers). Pure mechanical
move; no behavior change.

### R2. Introduce `Engine._llm_json` to collapse the parse-or-fallback idiom `[impact: M][effort: S]`

Add a single helper on `Engine`:

```python
async def _llm_json(
    self, prompt: str, *, node: str, fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = await self._chat(prompt, node=node)
    parsed = _parse_json_lenient(text, node=node) or (fallback or {})
    return parsed
```

Replace the six call sites at `engine.py:501, 650, 728, 1252, 1325,
1377, 1474, 1502, 1528` with `await self._llm_json(prompt, node="...",
fallback=...)`. Net deletion of ~30 lines, plus the contract for
"every JSON-parsing LLM call also logs raw output on parse failure" is
enforced at one site instead of left to per-call-site discipline. Three
nodes that customize the fallback (clarify, design, analyze) keep their
custom fallbacks via the kwarg.

### R3. Drop dead `QuestState` fields and group the survivors `[impact: M][effort: S]`

Delete `auto_collected_count`, `data_files`, and `review_panel` from
`QuestState` (`engine.py:103, 116, 124`). Surface the same observational
data via `QuestArtifacts.raw_state` or the run.log rather than the state
contract. Group the survivors into three nested TypedDicts:
`ClarifyState` (questions/answers/done/no_simulation_resolved),
`ExecState` (result/reflect_iter/reflect_history/give_up_reason),
`ReviewState` (review/iteration). The flat field count drops from 26 to
~10 top-level keys. Checkpoint resume logs at `engine.py:237-241` get
substantially more readable (today they print
`sorted((prior_snapshot.values or {}).keys())` — a flat list of 20+
unsorted names).

### R4. Split routing predicates into `core/engine_routing.py` `[impact: S][effort: S]`

Move `_route_after_design`, `_route_after_execute_reflect`,
`_route_after_cross_check`, `_route_after_review` (engine.py:403-464)
to a sibling module as free functions taking `(state, engine_config)`.
The `_build_graph` body (engine.py:363-398) passes them as predicates
unchanged, plus their unit tests stop needing a full `Engine`
instance. Also consolidates the three iteration-budget checks
(`engine.py:439, 453, 462`) into a single helper. Small win on its own;
buys clarity when combined with R3.

### R5. Refactor `Engine.run` into a state machine `[impact: L][effort: M]`

The 160-line method body (engine.py:168-324) splits into:

```python
async def run(self, ...):
    self._prepare()                       # mkdir + log
    self._preflight_paper_pdf()
    async with self._provider_session():  # endpoint resolve + client
        async with self._graph_session() as graph:
            final_state = await self._run_interrupt_loop(graph, ...)
    if self._is_paused(final_state):
        return self._collect_artifacts(final_state)
    artifacts = self._collect_artifacts(final_state)
    self._write_back_knowledge(artifacts, final_state)
    return artifacts
```

The interrupt loop becomes a dispatch table keyed on
`intr_value.get("kind", ...)` so future interrupts (slide regeneration,
multi-stage data acquisition) extend the table instead of adding `elif`
branches. The two existing interrupt kinds get explicit names rather
than the implicit `data_required: True` flag check at engine.py:262.
The two nested `try/finally` blocks become two context managers, each
with the same scoping but explicit names visible at the top of `run`.
Pairs well with the (not-yet-landed) Phase C work that I expect will
add a third interrupt point.

### R6. Extract `_node_review`'s panel path into a sibling method `[impact: M][effort: S]`

Pull the panel-mode branch (engine.py:1488-1558) into
`_run_review_panel(self, state, base_prompt) -> dict`. `_node_review`
keeps the legacy single-reviewer path inline (16 lines) and delegates
to the new method when `self.config.engine.review_panel` is non-empty.
The closed-over `run_persona` helper becomes a module-level function or
a method on `Engine`, reachable from tests. Today
`tests/test_review_panel.py` reaches into `_aggregate_panel_reviews`
directly because `run_persona` is unreachable; this refactor makes the
test surface match the code surface.

### R7. Move Windows venv-race workaround into `core/execution.py` `[impact: S][effort: M]`

The warmup + retry workaround in `_node_execute` (engine.py:1115-1185)
should live behind `executor.execute_with_warmup(args, deps=...)` in
`core/execution.py`. Engine.py's node body becomes ~30 lines and stops
caring about Windows DLL load order. Tests for the race
(`tests/test_engine_execute_retry.py`) move closer to the code they
exercise. The platform-specific logic stops being mixed with the
quest-state plumbing.

### R8. Extract `_resolve_no_simulation_from_clarify` as a free function `[impact: S][effort: S]`

Pull the 92-line method (engine.py:542-632) out as
`resolve_no_simulation(answers, config, logger) -> bool`. The unit test
surface gets cleaner; future clarify slots that need similar precedence
logic (e.g., a `paper_venue` resolver) can follow the pattern. No
behavior change.

### R9. Instance-scope the `_PROXY_WARN_SHOWN` suppression `[impact: S][effort: S]`

Replace the module-level `_PROXY_WARN_SHOWN: set[str] = set()`
(engine.py:2371) with `self._warned_providers: set[str]` on `Engine`.
Tests stop needing to import and reset the module global
(`test_engine_resume.py:200`). The "warn once per process" contract
becomes "warn once per Engine," which is the right scope for the fleet
runner anyway (each fleet quest is its own Engine and wants its own
warning — today, the second-and-later fleet quests in the same process
silently inherit warning suppression from the first quest's provider
choice).

### R10. Cache `_load_prompts` at module level `[impact: S][effort: S]`

Move the `_load_prompts()` call from `Engine.__init__` (engine.py:165)
into a module-level cached call so the fleet runner pays the
12-file-read cost once per process, not N times. `functools.lru_cache`
on `_load_prompts` is one line.

---

## References

- `core/engine.py:1-2685` — entire audited file
- `core/engine.py:63-124` — `QuestState` TypedDict (26 fields)
- `core/engine.py:168-324` — `Engine.run` (160-line state machine)
- `core/engine.py:328-399` — `_build_graph` (LangGraph topology)
- `core/engine.py:403-464` — routing predicates (four functions)
- `core/engine.py:468-1558` — node implementations (14 `_node_*` methods)
- `core/engine.py:542-632` — `_resolve_no_simulation_from_clarify` (92 lines)
- `core/engine.py:731-942` — `_node_auto_collect_data` + `_axon_collect_step`
  + `_run_dataset_adapters` (213 LOC across the no-simulation collection path,
  added in PRs #60 and #61)
- `core/engine.py:1101-1205` — `_node_execute` with Windows DLL-race workaround
- `core/engine.py:1437-1558` — `_node_review` (125 lines; legacy + panel paths)
- `core/engine.py:1806-2685` — module-level helpers (~880 LOC, candidate for
  R1 extraction)
- `core/engine.py:2371` — `_PROXY_WARN_SHOWN` module-level singleton
- `core/engine.py:2599-2685` — `_quest_logger` + `_close_quest_logger` (Windows
  FileHandler lifecycle, ~85 LOC)
- `core/config.py:80-130` — `EngineConfig` (15+ engine-level toggles, all
  read by `engine.py` via `self.config.engine.*`)
- `tests/test_engine_helpers.py` (2,242 lines) — exercises ~half of the
  module-level helpers; co-locating these in `core/engine_helpers.py`
  (per R1) is the obvious follow-up
- `tests/test_engine_resume.py:200` — example of a test reaching into
  `_PROXY_WARN_SHOWN` to reset module global between cases
- Recent PRs that added to `engine.py` without restructuring: #56
  (no-simulation pause), #57 (data_load node), #58 (paper-pdf preflight),
  #59 (simulatability clarify slot), #60 (Axon auto-collect), #61
  (dataset adapters). Net additions: ~700 LOC over two weeks.
