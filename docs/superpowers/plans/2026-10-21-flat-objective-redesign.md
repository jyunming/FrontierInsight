# A search whose objective never moves goes back to the plan once: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When FI's search has measured that its objective does not move across the design space, the plan is
rewritten once from FI's measured fact (as a change FI approves itself, made after results were seen), instead of the
search stopping or running its whole budget for nothing.

**Architecture:** The route copies the one FI already has for a study too long to run (`too_long` → evidence gate →
`_rewrite_plan(fit_request)` → `design_from_plan`). The search records the fact and, while a revision is allowed, stops
early as today; the runner then returns at once (no best-design check, no analysis) with the fact; the execute step
puts it in `exec_result["flat_objective"]`; the evidence gate asks the plan's model to rewrite plan.md with it; the
design step takes the new plan and, when the protocol is frozen, applies the change as an amendment approved by FI.
After the one revision, the search watch still records the fact but never stops the search.

**Tech Stack:** Python 3.11, async LangGraph engine, pytest.

**Spec:** `docs/superpowers/specs/2026-10-21-flat-objective-redesign-design.md`

## Global Constraints

- At most one revision for this reason per quest; it counts against `engine.max_iterations` (it bumps `iteration`).
- No revision in the confirm stage of `engine.phased` (the frozen design runs once).
- No new config key, CLI flag or user-facing term. The threshold stays `NO_EFFECT_DESIGNS` (50).
- FI never names a representation, a resolution or a range: the request states only what FI measured and the rules.
- The revision is recorded as made after results were seen (amendment `results_seen=True`, design-history entry
  `results_seen=True`), and the first design's run is archived (`archive_run`), never deleted.
- The revised design passes `_hold_the_design_to_the_topic` like any design (no special path around it).
- No topic words anywhere (fixtures use neutral made-up topics); committed text is English; run
  `python C:/fi_runs/c1/scan_topic_words.py -` on `git diff --cached` before each commit.
- Run only the listed test files, each in full, with a distinct `--basetemp` (`./.pytest_tmp/<task>`); no `git stash`.

## Review Focus

1. A quest resumed (`--resume`) after the revision: the once-marker is in state, so a second flat search runs its whole
   budget and never routes back. Test in Task 3.
2. The plan's model returns an unusable plan (`_rewrite_plan` raises): the quest must not stop without a paper nor
   loop; it runs the first design's search again with the stop off. Test in Task 3.
3. A search in which the objective moves, or a local-only search, is untouched (no fact, no route). Test in Task 1
   (existing tests kept) and Task 2.
4. A timing result (a number that differs when the same design runs twice) must not be listed as "moved". Test in
   Task 2.
5. Explore stage of `engine.phased` (protocol not frozen yet): the revision happens with no amendment, and the design
   history still says after results. Test in Task 4.

---

### Task 1: The search keeps the fact and stops only while a revision is allowed

**Files:**
- Modify: `core/optimise_search.py` (`__init__` watch state ~573-584, `_take_scan` ~600, `evaluate` ~645,
  `_watch_objective` ~724, `_flat_stop` ~750, `outcome` ~1149, `search()` ~1153 and its state constructor)
- Test: `tests/test_search_flat_objective.py`

**Interfaces:**
- Produces: `search(block, *, seed, method=None, rounds=None, results=None, batch=False, stop_flat=True)`.
  `outcome["flat_objective"]` = `{"designs": int, "objective": float, "feasible": bool, "stayed": list[str],
  "moved": list[str], "stopped": bool}`; `stayed`/`moved` are sorted names of the plan's results (`results=`, the
  objective's quantity left out) whose value was / was not the baseline's in every counted flat design.
- With `stop_flat=False` the watch counts the same way, and once it holds over `NO_EFFECT_DESIGNS` designs moving every
  number it sets `flat_objective` with `"stopped": False`, does not raise `_Stop`, and `stopped_because` is unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/test_search_flat_objective.py` (reuse the file's `_block` helper and
  flat simulation):
  - `test_the_flat_fact_names_the_results_that_stayed_and_those_that_moved`: a simulation returning the objective
    constant, a result `part_a` constant and a result `part_b` equal to the first design variable, `results=["part_a",
    "part_b"]`; assert `outcome["flat_objective"]["stayed"] == ["part_a"]`, `["moved"] == ["part_b"]`,
    `["stopped"] is True`, `["feasible"] is True`.
  - `test_after_the_revision_a_flat_search_runs_its_whole_budget_and_still_records_the_fact`: same flat simulation,
    `stop_flat=False`; assert `outcome["stopped_because"] != "flat_objective"`, `outcome["evaluations"] ==
    outcome["budget"]`, `outcome["flat_objective"]["stopped"] is False` and `["designs"] >= search.NO_EFFECT_DESIGNS`.
  - Update the existing equality on `outcome["flat_objective"]` (line ~52) to the new keys.
- [ ] **Step 2: Run** `python -m pytest tests/test_search_flat_objective.py -v --basetemp=./.pytest_tmp/t1` → the two
  new tests FAIL (`KeyError: 'stayed'`; the search stops early).
- [ ] **Step 3: Implement.** `_watch_objective` takes the design's `values` (the scan item's `values`, the reply's
  `values` in `evaluate`); the baseline's values of the plan's results are kept when the baseline is judged; each
  counted flat design removes from the "stayed" set every result whose value is not `_same_result` to the baseline's
  (a result missing from a reply counts as moved). `_flat_stop` builds the record and raises `_Stop` only when
  `self.stop_flat`; otherwise it records once and keeps searching (the watch then ends: `flat_watch = False`).
- [ ] **Step 4: Run** the same command → all PASS.
- [ ] **Step 5: Commit** `The search records which results stayed while its objective did not move, and stops only
  when the plan can still be revised`.

### Task 2: The runner returns the fact at once and leaves out unsteady numbers

**Files:**
- Modify: `core/optimise.py` (`_run_key` ~146, `run_search` ~520/708/736, `OptimisationRunner.__init__` ~1037,
  `execute` after the `no_effect` branch ~1158)
- Test: `tests/test_optimise_runner.py`

**Interfaces:**
- Consumes: Task 1's `search(..., stop_flat=)` and `outcome["flat_objective"]`.
- Produces: `run_search(..., stop_flat: bool = True)`; `_run_key(..., stop_flat=True)` adds `stop_flat` to `what` only
  when it is False (a key an earlier FI kept stays valid). `OptimisationRunner(..., stop_flat: Any = True)` (a bool or
  a callable asked at each run, like `frozen`). `OptimisationRunner.flat_objective: dict | None`: set when the search
  stopped for `flat_objective`; the fact with `"unsteady": sorted names` added and those names removed from
  `stayed`/`moved`. When set, `execute` returns `ExecutionResult(returncode=1, stdout="", stderr=<one plain sentence
  of the fact>)` before the best-design check and the analysis, and `failed_script = None`.

- [ ] **Step 1: Write the failing tests** in `tests/test_optimise_runner.py` (reuse its scan/search fixtures):
  - `test_a_flat_search_returns_the_fact_without_the_check_or_the_analysis`: a flat `run_cell` simulation; assert the
    result's `returncode == 1`, `runner.flat_objective["designs"] >= 50`, `needs/OPTIMUM_CHECK.json` absent and
    `experiment.py` never ran (its marker file absent).
  - `test_a_timing_is_named_unsteady_not_moved`: the flat simulation also returns `runtime_s` from `time.perf_counter()`;
    assert `"runtime_s" in runner.flat_objective["unsteady"]` and not in `["moved"]`.
  - `test_with_the_stop_off_a_flat_search_goes_on_to_the_check_and_the_analysis`: `stop_flat=False`; assert
    `runner.flat_objective is None` and the analysis ran.
- [ ] **Step 2: Run** `python -m pytest tests/test_optimise_runner.py -v --basetemp=./.pytest_tmp/t2` → the new tests
  FAIL.
- [ ] **Step 3: Implement.** `run_search` keeps the scan's `unsteady` set (computed today inside `scan()`) in an outer
  variable, passes `stop_flat` to `_search.search` and to `_run_key`, and puts `unsteady` in the record next to
  `scan_settings_without_effect`. The runner reads `outcome`'s `evaluations.flat_objective` from `run.record` when
  `stopped_because == "flat_objective"`.
- [ ] **Step 4: Run** the same command plus `tests/test_search_flat_objective.py` → all PASS (the two existing
  `len(rows) + 1` counts stay right).
- [ ] **Step 5: Commit** `A search whose objective did not move hands its fact back at once, without timings`.

### Task 3: The evidence gate sends the fact to the plan once

**Files:**
- Modify: `core/engine.py` — `QuestState` (~497: `flat_objective_redesigned: bool`), the runner construction
  (~17216: `stop_flat=`), the execute patch (~17967: `"flat_objective"`), `_route_after_execute_reflect` (~2790),
  `_no_results_verdict` (~20545), the gate's rewrite block (~20020) and patch (~20086), a new module-level
  `_flat_request(fact: dict, kept: list[str] | None) -> str` beside `_fit_request` (~27289).
- Test: create `tests/test_flat_objective_redesign.py`

**Interfaces:**
- Consumes: Task 2's `runner.flat_objective`.
- Produces: `exec_result["flat_objective"]` (the fact dict, only when the run returned it); state
  `flat_objective_redesigned: True` after the rewrite; `state["flat_objective_fact"]` (the fact, for Task 4).
- `stop_flat` = `lambda: not state.get("flat_objective_redesigned") and not phased-confirming and
  int(state.get("iteration", 0)) < self.config.engine.max_iterations`.
- Route: `_route_after_execute_reflect` returns `"stuck"` when `exec_result.get("flat_objective")` (to the gate; it is
  not added to `_STOPPED_BEFORE_THE_STUDY`). `_no_results_verdict` returns, first, `{"verdict": "insufficient",
  "rationale", "gaps", "redesign": True, "flat_request": _flat_request(...)}`. The gate rewrites plan.md with
  `_rewrite_plan(flat_request, ..., by="engine", note="a design whose objective can move, asked by FI")` and patches
  `iteration + 1`, `design_from_plan: True`, `flat_objective_redesigned: True`, `flat_objective_fact`. If the rewrite
  raises: route `"repair_crash"` with `flat_objective_redesigned: True` (the next run searches with the stop off), and
  run.log says why.
- `_flat_request` states, in plain words: the count of designs across the whole range, the common objective value
  and whether the limits were met, that the starting design gave it too, which results stayed and which moved (and the
  unsteady ones as timings), then the rules: the design must let the objective move; keep every case, setting and
  check the topic asks for (`kept`, from `_topic_req.kept_when_smaller`); do not change what the objective measures to
  get a number; return an objective built from parts with each part as a result.

- [ ] **Step 1: Write the failing tests** in `tests/test_flat_objective_redesign.py` (an engine built as in
  `tests/test_run_checks.py::_sizing`; a fake `LLMClient.chat` for `plan_revise`):
  - `test_a_flat_search_goes_to_the_gate_and_the_plan_is_rewritten_once_with_the_fact`: `exec_result` with
    `flat_objective`; assert the execute-reflect route is `"stuck"`, the verdict has `redesign` and a `flat_request`
    containing `"50 designs"`, `"part_a"` and `"keep every"`, the gate's route is `"redesign"`, the patch has
    `iteration == 1`, `design_from_plan` and `flat_objective_redesigned`, and the fake model was asked once.
  - `test_after_the_one_revision_the_stop_is_off`: the runner built with state `flat_objective_redesigned: True` has
    `stop_flat()` False; same for `iteration == max_iterations`.
  - `test_an_unusable_rewrite_runs_the_search_again_with_the_stop_off`: the fake model returns garbage twice; assert
    route `"repair_crash"` and `flat_objective_redesigned` True, and no `STUCK.json`.
- [ ] **Step 2: Run** `python -m pytest tests/test_flat_objective_redesign.py -v --basetemp=./.pytest_tmp/t3` → FAIL.
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run** the new file plus `tests/test_run_checks.py tests/test_engine_helpers.py
  tests/test_topic_requirements.py` (callers of `_no_results_verdict` and `_fit_request`) → all PASS.
- [ ] **Step 5: Commit** `A search whose objective did not move sends what FI measured back to the plan once`.

### Task 4: The design step takes the rewritten plan as a change FI approves

**Files:**
- Modify: `core/engine.py` — the design node's `design_from_plan` branch (~8506), a new method
  `_amend_for_flat_objective(self, state, design) -> dict`, `_append_design_revision` (~33273: a `reason=` keyword).
- Test: `tests/test_flat_objective_redesign.py`

**Interfaces:**
- Consumes: Task 3's `flat_objective_fact`, `flat_objective_redesigned`.
- `_amend_for_flat_objective`: when a frozen protocol exists and the plan's protocol differs: `_frozen.propose(...,
  source="FI: the search's objective did not move", reason=<the fact in one sentence>, results_seen=True)`, then
  `_frozen.approve(quest_root, "FI: the search's objective did not move across the design space (the plan was revised
  once, after results were seen)", via="fi")`, then `_frozen.apply(...)`; returns the design with the new protocol and
  logs one plain line. No frozen protocol (explore stage): returns the design unchanged.
- The design-history entry for this design has `results_seen=True` and `reason="the search's objective did not move
  (<n> designs gave <value>); the plan was rewritten once from what FI measured (results were seen)"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_the_rewritten_plan_is_an_amendment_fi_approved_after_results`: frozen protocol, plan.md rewritten; run the
    design node; assert `needs/PROTOCOL_AMENDMENT_1.json` has `prespecified is False`, `approved_by` starting `"FI:"`,
    the fact in `reason`; the new frozen protocol is the plan's; `archive/run_1/` exists and holds the first search's
    ledger (`LEDGER_PATH` in `core/optimise.py`; add it to `archive_run`'s copied names if it is not there: the spec
    says the first design's evaluations are kept and reported); no pause.
  - `test_the_paper_is_told_and_the_evidence_names_the_gap`: after that, `frozen_protocol.disclosure()` contains
    `"post-hoc"` and the fact, and `core.evidence` lists the amendment among the gaps.
  - `test_in_the_explore_stage_there_is_no_amendment_but_the_history_says_after_results`: no frozen protocol; assert no
    amendment file and the last design-history entry `results_seen is True` with the reason above.
  - `test_a_revision_that_drops_a_case_the_topic_asks_for_is_still_held_to_the_topic`: the rewritten plan drops a
    topic setting; assert `_hold_the_design_to_the_topic` is called on it (monkeypatch spy) and its result is the design
    that runs.
- [ ] **Step 2: Run** `python -m pytest tests/test_flat_objective_redesign.py -v --basetemp=./.pytest_tmp/t4` → FAIL.
- [ ] **Step 3: Implement** as in Interfaces; call `_amend_for_flat_objective` right after `_design_from_plan()` in
  the `design_from_plan` branch when `state.get("flat_objective_fact")` and the fact's revision is not yet recorded.
- [ ] **Step 4: Run** the new file plus `tests/test_frozen_protocol.py tests/test_design_history.py` (grep `tests/` for
  `_append_design_revision` and `_settle_pending_amendment` and add those files) → all PASS.
- [ ] **Step 5: Commit** `The rewritten plan is a change FI approves itself, recorded as made after results`.

### Task 5: The plan asks for the objective's parts, and the paper says a flat search was revised

**Files:**
- Modify: `core/engine.py` `_STUDY_TYPE_DIRECTIVE` (~27590): one sentence: "When the objective is built from parts
  (the worst of several cases, a sum of terms), return each part as a result of its own, so a part that never moves
  can be seen."
- Modify: `core/best_design_report.py` and `core/optimise.py:summary_lines`: when `flat_objective` is recorded with
  `stopped is False`, one sentence: "the objective did not move: its first <n> designs across the whole range all gave
  <value>; FI had already sent the design back to the plan once for this".
- Modify: `docs/capabilities-reference.md` (the search entry), `dev/registry.md` (the search-outcome line), and
  `wiki/` page for the search if one exists (`python scripts/wiki_sync_index.py --check`).
- Test: `tests/test_search_flat_objective.py`

- [ ] **Step 1: Write the failing tests:** `test_the_plan_asks_for_the_parts_of_an_objective` (the directive contains
  `"each part as a result"`); `test_a_flat_search_after_the_revision_is_said_in_the_report` (a record with
  `flat_objective.stopped == False` → the report text contains `"did not move"` and `"back to the plan once"`).
- [ ] **Step 2: Run** `python -m pytest tests/test_search_flat_objective.py tests/test_public_text_english.py -v
  --basetemp=./.pytest_tmp/t5` → the new tests FAIL.
- [ ] **Step 3: Implement** and update the docs.
- [ ] **Step 4: Run** the same command → all PASS.
- [ ] **Step 5: Commit** `The plan returns an objective's parts; the paper says a flat search was revised once`.

### Task 6: Shown fixed (outside the repository)

- [ ] **Step 1: Replay d21's search record** (`<run folder>`, a copy of its code and
  `.fi/optimisation`): drive `core.optimise_search.search` with the recorded scan and global answers; expected: the
  watch fires at the scan or the global stage with `designs >= 50`, the request text names the stuck results, and the
  engine routes to `design` once (the gate run on the replayed `exec_result`).
- [ ] **Step 2: Known positives:** replay every other search record under `C:/fi_runs/suite/rounds/*/` with
  `optimisation` (the same set as `replay_unsteady_all.py`); expected: no `flat_objective` where the objective moves,
  and the outcomes unchanged.
- [ ] **Step 3: External review** (Sonnet) of the branch diff for Tasks 1-5, then a targeted second pass on each fix.
- [ ] **Step 4: Live run d22** (the topic without 72 nm) on the new HEAD in a new worktree, watched throughout.
