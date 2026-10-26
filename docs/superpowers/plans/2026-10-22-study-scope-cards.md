# Study Scope Cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the literature step, FI writes 3–4 study cards, the person picks, edits or writes one, and the plan is
written for that study. When no one can answer, the cards are recorded and nothing is chosen.

**Architecture:** A pure module `core/scope_cards.py` owns the cards: it normalises the model's reply against FI's
literature record, builds the question and reads the answer, writes `.fi/scope.json`, and compares a plan with a card.
A new engine node `scope` sits between `pause_after_literature` and `select_skills`. It asks through the clarify
answer channel the run already has: the in-process callback, or the answer file. The plan prompt and the study-type
channel read the chosen card. QUALITY.md, `plan.md` and the limitations note read `.fi/scope.json`.

**Tech Stack:** Python 3.11, async LangGraph, pydantic config, `string.Template` prompts in `agents/`, TypeScript
(VS Code extension), web page in `web/`.

**Spec:** `docs/superpowers/specs/2026-10-22-study-scope-cards-design.md` (approved 2026-10-22). Read it before any
task.

## Global Constraints

- Everything committed is in English. `tests/test_public_text_english.py` must pass.
- No test-topic words in code, prompts, tests, fixtures or commit messages. Fixtures use neutral made-up topics.
  Run `python C:/fi_runs/c1/scan_topic_words.py -` on the diff and on the message before each commit.
- Tests never call a real model. Use `monkeypatch.setattr("core.engine.LLMClient.chat", fake)`.
- One test run at a time on this machine: `python C:/fi_runs/suite/run_tests.py <worktree> <fresh-name> <args>`.
  Run each listed file in full, and paste the summary line.
- `core/engine.py` is edited only by the engine-lane owner (Task 3), never by a parallel subagent.
- User-visible words: the person sees "Pick the study to run", never "scope" or "card" as a term. The config key is
  `pauses.scope: ask | off`.
- The plan not following the card never stops a quest. It is disclosed only (core/stop_power.py: a reading).
- Commit messages end with the two attribution lines given in the session.

## Review Focus

1. The model's reply cites a source FI never read. That entry is removed and the removal is recorded. Task 1 test:
   `test_a_literature_line_without_a_read_source_is_removed_and_recorded`.
2. The person types something that is neither a card number nor a sentence (empty, `0`, `9` with 3 cards, or `e`
   with no number). An empty answer counts as "nobody chose"; an out-of-range number is asked once more on the
   terminal, and in the other paths counts as "nobody chose". It never crashes. Task 1 test:
   `test_an_answer_that_names_no_card_is_nobody_chose`.
3. `.fi/clarify_answer.json` already exists from clarify. The scope step must not read it as the study answer.
   Task 3 test: `test_the_setup_answer_file_is_not_read_as_the_study_answer`.
4. Resume after the cards were written but before the plan, with or without an answer, does not ask again and keeps
   the record. Task 3 test: `test_a_resume_after_the_cards_does_not_ask_again`.
5. A YAML `pauses: {scope: off}` without quotes parses as False and must mean `off`. Task 2 test:
   `test_unquoted_off_turns_the_study_cards_off`.

---

### Task 1: `core/scope_cards.py`, the cards, the answer, the record, the comparison

**Files:**
- Create: `core/scope_cards.py`
- Test: `tests/test_scope_cards.py`

**Interfaces:**
- Produces:
  - `STUDY_TYPES = ("measure", "find_best_design")`
  - `RECORD = ".fi/scope.json"`, `ANSWER = ".fi/scope_answer.json"`
  - `normalize(reply: Any, source_ids: set[str]) -> tuple[list[dict], list[str]]`: returns the cards and the
    sentences saying what was removed. Each card is
    `{"question": str, "study_type": str, "main_results": list[str], "compared_settings": list[str],
    "literature": list[{"says": str, "source": str}], "run_time": str}`, with `widest: bool` kept when the reply
    marks it. Rules:
    - a card with a `study_type` outside `STUDY_TYPES` is dropped (`compare` maps to `measure`);
    - a literature entry whose `source` is not in `source_ids` is removed;
    - duplicates are cards with the same `study_type` and the same set of `main_results` after `" ".join(x.lower().split())`;
    - at most 4 cards are kept.
  - `distinct_enough(cards: list[dict]) -> bool`: true when there are at least 3 cards.
  - `has_widest(cards: list[dict]) -> bool`: true when some card has `widest: true` or is `find_best_design`.
  - `question(cards: list[dict]) -> dict`: the clarify-shaped payload
    `{"study": {"question": <text>, "default": "", "cards": cards}}`. The text opens with the line "Pick the study to
    run (a number), or write the study you want in a sentence:" and then one block per card, numbered from 1.
  - `read_answer(answer: Any, cards: list[dict]) -> dict | None`. It reads `answer` (a str, an int, or
    `{"study": ...}`) as:
    - a number from 1 to n: that card;
    - a sentence: a card `{"question": sentence, "study_type": None, "main_results": [], "compared_settings": [],
      "literature": [], "run_time": "", "written_by_person": True}`;
    - a dict with card fields (an edited card): that card;
    - anything else (empty, out of range): `None`.
  - `write_record(quest_root: Path, cards: list[dict], chosen: dict | None, removed: list[str], *,
    widest: bool) -> None`: writes `RECORD` with `{"cards", "chosen", "chosen_by": "person" | "nobody",
    "removed", "widest_offered", "history": []}`. When the file exists, its old `chosen` is first appended to
    `history`.
  - `read_record(quest_root: Path) -> dict | None`
  - `plan_note(card: dict) -> str`: the plan prompt text. It says: the person chose this study; write the plan for
    it; cover each main result and compared setting, or say in the plan why one is left out.
  - `gaps(card: dict, design: dict) -> list[str]`: each `main_results` entry with no metric `id` or `label` in
    `design["metrics"]`, and each `compared_settings` entry with no grid axis or optimisation variable name, compared
    after the same lower/space normalisation. Underscores count as spaces. One plain sentence per gap.
  - `nobody_line() -> str`: "Nobody picked the study to run; FI planned from the topic alone. The studies it could
    also have been:"

- [ ] **Step 1: Write the failing tests.** Neutral topic (for example a cooling cup). The tests are:
  - `test_normalize_keeps_three_distinct_cards_and_maps_compare_to_measure`
  - `test_a_literature_line_without_a_read_source_is_removed_and_recorded` (assert `"src9"` is not in any card, and
    that one removal sentence names it)
  - `test_two_cards_with_the_same_type_and_results_are_one`
  - `test_read_answer_number_sentence_and_edited_card`
  - `test_an_answer_that_names_no_card_is_nobody_chose` (cases `""`, `"0"`, `"9"`, `"e"`, `None`)
  - `test_write_record_keeps_the_earlier_choice_in_history`
  - `test_gaps_names_a_main_result_the_plan_has_no_metric_for` (a design with metrics `[{"id": "cool_rate"}]` and the
    card result "cool rate", plus "final temperature"; exactly one gap, naming "final temperature")
  - `test_question_text_numbers_each_card_and_says_pick_the_study_to_run`
- [ ] **Step 2: Run them.** Expected: they fail on the import.
  `python C:/fi_runs/suite/run_tests.py <wt> sc1 tests/test_scope_cards.py`
- [ ] **Step 3: Implement the module as specified above.** It is pure: no model calls and no engine import.
- [ ] **Step 4: Run.** Expected: all pass.
- [ ] **Step 5: Commit** "The study cards: what FI offers, what a person picks, and what the plan left out of the pick".

### Task 2: the prompt `agents/scope.md` and the setting `pauses.scope`

**Files:**
- Create: `agents/scope.md`
- Modify: `core/config.py`: add `ScopePause = Literal["off", "ask"]` next to `PlanPause` (~line 615), the field
  `scope: ScopePause = "ask"` in the pauses class (~line 663) with a one-line comment ("before the plan, pick the
  study to run from a few FI offers; off skips it"), and a `_norm_scope` validator mapping `False` → `"off"` like
  `_norm_plan`.
- Test: `tests/test_scope_cards.py` (the config tests go there too).

**Interfaces:**
- Produces: `agents/scope.md` with the placeholders `$topic`, `$answers`, `$direction`, `$sources` (one line per
  source: `id: title — key finding`), `$again` (empty, or the sentence naming what was the same).

The prompt asks for JSON `{"cards": [...]}` with the Task 1 card fields. It asks for 3 or 4 cards that differ in
study type or in main results; one card marked `"widest": true`, the widest reading of the topic, which is a
search for the best design when the topic names something to improve; and `literature` entries only with a source
id from `$sources`.

- [ ] **Step 1: Failing tests.**
  - `test_unquoted_off_turns_the_study_cards_off`: `Config.model_validate` with `pauses: {scope: False}` gives
    `"off"`.
  - `test_study_cards_are_on_by_default`
  - `test_the_study_card_prompt_has_its_placeholders`: load the file with `string.Template`, substitute all five
    placeholders, and check that no `$` placeholder is left.
- [ ] **Step 2: Run.** Expected: they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_scope_cards.py` together with the existing config tests (`grep -l "pauses" tests/test_config*.py`).
  Expected: all pass.
- [ ] **Step 5: Commit** "A setting and a prompt for picking the study to run".

### Task 3 (engine lane, serial): the node `scope` and its wiring

**Files:**
- Modify: `core/engine.py`:
  - `STAGE_PROGRESS` (~line 219): `"scope": "Offering the studies this topic could mean."`;
  - the graph: the edge `pause_after_literature → scope → select_skills` (~line 2583);
  - `run()`: keep `self._scope_answerer = clarify_callback`;
  - the new `async def _node_scope(self, state) -> QuestState`;
  - `QuestState`: a `scope_card: dict[str, Any] | None` field;
  - `_study_type_asked` (~line 5712);
  - `_node_plan` (~line 5470), where the plan prompt is built.
- Modify: `core/rerun_from.py`: map `"scope"` into the step table (~line 104) so that `--from scope` reruns it and
  needs `--approve-as`, as the steps before design do.
- Test: `tests/test_scope_node.py`

**Interfaces:**
- Consumes: Task 1 (`normalize`, `question`, `read_answer`, `write_record`, `read_record`, `plan_note`,
  `distinct_enough`, `has_widest`, `ANSWER`); Task 2 (`agents/scope.md`, `config.pauses.scope`).
- Produces:
  - `state["scope_card"]` (the chosen card, or None);
  - `.fi/scope.json`;
  - `_study_type_asked(state)` returns `state["scope_card"]["study_type"]` when a card is chosen and has a type,
    before it looks at clarify's answer.

What `_node_scope` does:
1. It returns `{}` when `pauses.scope == "off"`, when `iteration > 0`, when `analyze_local_first` is set, when
   `plan.md` exists, or when `read_record` finds a record. When the record has a choice, it returns
   `{"scope_card": that}`.
2. It makes one model call, `node="scope"`. `source_ids` are the ids of the literature record the plan reads; reuse
   the helper that `_node_plan` uses for its sources. The reply goes through `normalize`. When
   `not distinct_enough`, it asks once more, with `$again` naming what was the same.
3. It looks for an answer, in this order:
   - the file `ANSWER`, read at this moment;
   - else `self._scope_answerer(question(cards))`, awaited with `pauses.timeout_s` (0 means no limit). The callback
     returns `{"study": value}`. A `TimeoutError` or a `BridgeError` counts as nobody;
   - else nobody.

   It never reads `.fi/clarify_answer.json`.
4. It calls `write_record`, logs one plain line (which study was picked, or "nobody picked the study to run; the plan
   is written from the topic alone"), and returns `{"scope_card": chosen}`.

In `_node_plan`, add `scope_cards.plan_note(card)` to the prompt when `state.get("scope_card")`.

- [ ] **Step 1: Failing tests** (fake model; the engine is built like `tests/test_stop_power.py::_engine`):
  - `test_cards_are_recorded_and_nobody_chose_when_no_one_can_answer`
  - `test_a_staged_answer_picks_the_card_and_the_plan_prompt_names_it`: capture the plan prompt, assert that the
    card's question is in it.
  - `test_the_setup_answer_file_is_not_read_as_the_study_answer`
  - `test_a_resume_after_the_cards_does_not_ask_again`: the second call makes no model call.
  - `test_a_chosen_find_best_design_card_drives_the_study_type`: `_study_type_asked` returns `find_best_design`,
    even with clarify's answer `measure`.
  - `test_scope_off_writes_nothing`
  - `test_a_reply_with_two_alike_cards_is_asked_again_once`
- [ ] **Step 2: Run.** Expected: they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_scope_node.py`, `tests/test_rerun_from*.py`, `tests/test_engine_smoke.py` (the
  graph changed) and the plan-node tests (`grep -l "_node_plan\|study_type_asked" tests/`). Expected: all pass.
- [ ] **Step 5: Commit** "Before the plan, FI offers the studies the topic could mean and plans the one a person picks".

### Task 4: disclosure in QUALITY.md, plan.md and the limitations

**Files:**
- Modify: `core/quality_report.py`: one stage item from `.fi/scope.json`:
  - `chosen_by: nobody`: `nobody_line()` plus the card questions;
  - `chosen_by: person`: one line naming the chosen study;
  - each removed literature line, as "not fixed / removed".
- Modify: `core/run_checks.py::_run_checks_note`. When a card was chosen, for each `gaps(card, design)` sentence:
  "The person chose a study that names {x}; the plan does not cover it and gives no reason. Say so in the
  limitations."
- Modify: `core/engine.py`, after `plan.md` is written (engine lane). Append a section "## The study to run" with the
  chosen card, or `nobody_line()` and the cards.
- Test: `tests/test_scope_cards_disclosure.py`

- [ ] **Step 1: Failing tests:**
  - `test_quality_lists_the_cards_when_nobody_chose`
  - `test_a_plan_gap_against_the_chosen_card_is_in_the_note_and_does_not_stop`: `_cannot_repair_stop()` is None, and
    the note names the gap.
  - `test_plan_md_has_the_study_to_run_section`
- [ ] **Step 2: Run.** Expected: they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new file together with `tests/test_quality_report*.py` and `tests/test_run_checks.py`.
  Expected: all pass.
- [ ] **Step 5: Commit** "What study was picked, or that nobody picked one, is said in the plan, QUALITY.md and the limitations".

### Task 5: the three interfaces

**Files:**
- Modify: `launch.py`:
  - `_cli_clarify_callback` (~line 1992): when a question entry has `cards`, print the text and read one line, and
    ask once more on an out-of-range number;
  - `_web_page_clarify_callback` (~line 1892): add a `name: str = "clarify"` argument that sets the file stems
    (`scope_questions.json`, `scope_answer.json`, `scope_waiting.json`). The engine gets the scope callback through a
    new `Engine.run(scope_callback=...)` argument; when it is None, the clarify callback is used. Add that to Task 3's
    `_scope_answerer` line.
- Modify: `web/server.py` and the quest page: show `scope_questions.json` as radio cards with an edit box and a
  "write my own" box, and post to `scope_answer.json`, following the clarify pattern (~lines 307–346, 2687–2731).
  For a quest the server runs in-process (~line 3235), pass a scope callback built the same way.
- Modify: `vscode-frontier-insight/src/bridge.ts::handleClarifyRequest`: when `entry.cards` is set, use
  `showQuickPick`, with one item per card plus "Edit a card…" and "Write my own…" (the last two open
  `showInputBox`). Answer `{study: <number | card | sentence>}`.
- Test: `tests/test_scope_interfaces.py` (CLI callback with stdin monkeypatched; web callback with `name="scope"`
  writing and reading the scope files and never touching the clarify files), and
  `tests/test_vscode_extension_typescript.py`.

- [ ] **Step 1: Failing tests:** `test_terminal_reads_a_card_number_and_asks_again_on_a_bad_one`,
  `test_web_callback_for_the_study_uses_its_own_files`
- [ ] **Step 2: Run.** Expected: they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new file and `tests/test_vscode_extension_typescript.py`. Then run
  `npm run compile && npm run package` in `vscode-frontier-insight/` and state the `.vsix` path.
- [ ] **Step 5: Commit** "Pick the study to run in the terminal, on the web page and in VS Code".

### Task 6: docs and registry

**Files:** `dev/registry.md` (the `core/scope_cards.py` entry and the `scope` node), `docs/capabilities-reference.md`
(one bullet near the clarify and plan-pause entries), the README pause list, `vscode-frontier-insight/README.md`
(one line), and `wiki/` (a page `study-cards.md`, then `python scripts/wiki_sync_index.py`).

- [ ] **Step 1: Write.** Describe the feature in its own terms; no PR numbers.
- [ ] **Step 2: Run** `python scripts/wiki_sync_index.py --check` and `tests/test_public_text_english.py`.
  Expected: pass.
- [ ] **Step 3: Commit** "Docs: picking the study to run".

### Task 7: acceptance, a live plan-stage A/B (costs model calls; state the cost before running)

Not code. Use the configs already kept outside the repo for the 37-word and 93-word topics and their full version,
`pauses.plan: ask`, two runs each. In each run, the harness picks the card closest to the full plan by its
`study_type` and `main_results`, through `.fi/scope_answer.json`. Record:
(1) whether a card matching the full plan's study type and main results was offered;
(2) the plan's `study_type` after the pick;
(3) the number of plan metrics against the full plan.

It passes when (1) and (2) hold in both runs of the 37-word topic. Known positive: the full topic with its widest
card picked keeps its study type. Results go to `C:/fi_runs/suite/p20/ACCEPTANCE.md`, never into the repo.
