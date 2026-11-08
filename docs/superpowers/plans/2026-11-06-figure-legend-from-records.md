# Figure legend and caption from FI's records: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every line in a delivered figure has an identity in FI's records, and FI writes the legend and caption from them.

**Architecture:** the figure frame (`core/frame_figure.py`) refuses lines that bypass the records (a `value=` constant
that matches no single plan reference; the old `curve()`), formula curves are registered by the analysis in
`derived.json` and drawn by name, the legend is written from what differs between entries, and the caption's facts come
from the figure record. Lines still without identity after one repair round are removed and named in QUALITY.md.

**Tech stack:** Python 3.11, matplotlib, the existing figure frame and analysis frame.

**Spec:** `docs/superpowers/specs/2026-11-06-figure-legend-from-records-design.md` (its "Settled" and "Decided by the
maintainer" sections are binding).

Marks: UNCONFIRMED = not confirmed in the code or data when the plan was drafted; confirm before coding that step.

## Global Constraints

- AGENTS.md rules: English only in the repo; no topic words in code, tests, prompts, docs or commit messages (fixtures
  use neutral ids `r1`, `s1`, `ref_a`); plain user-facing words, no new config key, flag or term; fix the root (FI
  produces the fact), no after-the-fact guessing; a fix counts only when its original failing input is shown fixed on a
  replay; docs and `dev/registry.md` updated in the same piece of work.
- Async: no new blocking calls in the engine. Repair work stays inside `async def _figures_from_results`
  (`core/engine.py:11985`) and its `_FIGURE_DATA_REPAIRS = 2` budget (line 11978); the analysis re-run uses
  `await self.executor.execute(...)` as `_figures_after_give_up` does (line 12033).
- `core/frame_figure.py` imports nothing of FI's: it is copied as text into `code/fi_figure_calls.py`
  (`core/skeleton.py:67-72`, `FRAME_FIGURE_SOURCE`) and uses the standard library only. New frame helpers live in the
  frame file.
- A quest keeps the frame it was written with: `skeleton.VERSION = 5`, `RECORD_VERSION = 2`. A changed record shape
  bumps `RECORD_VERSION` to 3; readers (`engine._call_*`, `figure_claims`) accept 2 and 3.
- The frozen protocol stays frozen: a registered curve is a name in `derived.json`, not a new plan result.
- Maintainer decision: a line with no identity after the repair round is always removed, formula curves included; no
  "drawn but unverified" form; `curve()` is not kept.
- A wrong legend entry is the anti-goal: when unsure, drop the line, never guess its name.
- Tests: `python C:/fi_runs/suite/run_tests.py C:/dev/fi_wt_fx349 <slot> <files>`; run each listed file in full and
  paste the final line. No `git stash`.

## Review Focus (each tested in an owning task)

1. A coincidental `value=` match gives a confidently wrong name (plan references are often per-setting; `_plan_for`,
   `core/analysis_figures.py:530`, exposes only constant ones). Tests: `test_value_matching_two_plan_references_is_refused`,
   `test_value_matches_only_at_the_panels_own_setting` (Task 1).
2. `value=` and `ref_id=` for the same number draw the same line twice (seen in a fc11 record: a stated 0.05 and a plan
   reference 0.05); `_merge_identical` (`frame_figure.py:1095`) merges series only. Test:
   `test_identical_reference_lines_are_one_entry` (Task 3).
3. "What differs" over mixed series (table series, level curve, registered curve in one panel) must never give two
   identical or empty legend texts. Test: `test_entries_never_collide_and_never_empty` (Task 3).
4. A repair that rewrites analysis code must not change results: re-run on the recorded trials, no new simulation.
   Tests: `test_registering_a_curve_leaves_every_result_unchanged` (Task 2), `test_repair_rerun_does_not_simulate` (Task 5).
5. Caption and legend must not disagree with the record after a removal; the removed line is named in QUALITY.md.
   Tests: `test_removed_line_absent_from_record_legend_caption_and_named_in_quality` (Task 5) and the planted-mislabel
   bar (Task 6).

## Code map (real names)

- `core/frame_figure.py`: `FigureCalls.curve` (752-773, label "a curve", ids `curve%d`); `ref` (918-956; `ref_id` uses
  `self.references`; `at=` uses `_table_read`, source "table"; `value=` labels "Reference line", source `REF_STATED`);
  `_entries_keyed` (1299-1338); `_colour_key` (1340); `_merge_identical` (1095); `finish` (1004); `record` (1608);
  `write` (1620); `NEUTRAL_Y = "computed value"` (49); `LEGEND_LEVELS` (48).
- `core/analysis_figures.py`: `_draw_one` (308), `_figure_handle` (390), `_plan_for` (530), `_write_calls` (786),
  `_calls_summary` (800), `run_figures` (816), `default_figure` (1221), `_default_with_calls` (1309).
- `core/analysis_frame.py`: text of `code/experiment.py`; `run` (~1652) writes `data/results/derived.json` (`RECORD`,
  941) with `derived[id] = {status, why, missing, by_setting, role, by, inputs}`; `_COUNTING_ROLES =
  ("result","reference","physical_bound")` (69, 934); `_references` (1166); `_figures` (1576); `SIGNATURES` (85).
- `core/engine.py`: captions `_caption_words` (35624), `_call_group` (35753), `_call_drawn` (35766), `_call_extras`
  (35811), `_caption_from_calls` (35825), `_figure_caption_text` (~35895); records `_read_figure_records` (35040); repair
  `_figures_from_results` (11985), `_repair_typed_figures` (12088), `_figures_after_give_up` (12012);
  `QuestState.figure_data_repairs` (411).
- `core/quality_report.py`: `_analysis_problems` (696), `_paper_problems` (719), `_problem(...)` (235).
- `core/figure_claims.py` ("a curve" at 212), `core/figure_facts.py`.
- Prompts: `agents/figure_primitives.md` (lines 9-11, 21-24 describe `ref value=`, `curve`, error messages), loaded by
  `core/skeleton.py:1435-1437` and `core/analysis_frame.py:249,279`; also `core/run_checks.py:297,527`,
  `core/figure_data_check.py:372`.
- Existing tests touching these: frame `tests/test_frame_figure.py` (`test_curve_recorded_as_kind_curve_with_source`,
  `test_ref_has_three_sources_and_a_stated_value_gets_a_neutral_label`, `test_ref_unknown_id_raises_naming_known`),
  `test_figure_series_levels.py`, `test_figure_layout.py`, `test_figure_curve_xcheck_range.py`, `test_figure_labels.py`,
  `test_figure_mutations.py`, `test_figure_fixtures.py`, `test_figure_mark_roundtrip.py`, `test_figure_data_check.py`,
  `test_skeleton.py`; drawing `test_analysis_figures.py`, `test_analysis_frame.py`; caption `test_figure_captions.py`,
  `test_fi_places_figures.py`, `test_figure_records.py`, `test_replicate_figures.py`, `test_plot_style.py`,
  `test_g18_figure_facts.py`; claims `test_figure_claims.py`.

---

## Phase 0: measure first (no model calls, nothing merged)

Script `C:/fi_runs/c1/p0_1106/fx_349/replay_p0.py` + `README.md` (3 lines) + `out/`; nothing in the repo.

Inputs per quest `Q = C:/fi_runs/suite/rounds/fc{8,10,11}*/<topic>/out/<run>/`: `Q/.fi/figure_records/<stem>.calls.json`
(version 2), `<stem>.json` (plot-style record), `Q/code/derive.py` (bodies), `Q/code/fi_figure_calls.py` (frame),
`Q/needs/FROZEN_PROTOCOL.json` (`metrics[i].reference`, `precision.target_half_width`), `Q/data/results/derived.json`
(`derived[id].by_setting`), captions in `Q/paper.md`.

- [ ] Count per figure and per round: legend placeholders ("Reference line", "a curve", "curve N", "computed value") and
  entry totals (baseline fc11: 137 entries, 49 placeholders).
- [ ] Reference lines by source (plan / table / stated) and curves.
- [ ] For each stated constant: matches 0 / exactly 1 / more than 1 plan reference at that panel's setting, and "matches a
  result value only", under three tolerances side by side: typed digits (recommended), `target_half_width`, 1e-9.
- [ ] Worst-case lines removed = unmatched constants + all curves; share of figures losing a line; panels left empty.
- [ ] Captions whose reference-line list disagrees with the record (count and values, per panel); identical-value ref
  pairs (merge candidates); empty or raw panels the caption does not mention.
- [ ] Re-render recipe (UNCONFIRMED until run once): copy `Q` to `fx_349/copies/<round>_<topic>`; overwrite
  `code/fi_figure_calls.py` (and `fi_figures.py`, `experiment.py` if changed) with the new frame text; run the quest's
  venv python on `code/experiment.py` with cwd = copy (reads `raw/trials.json`, no simulation; do not set
  `FI_FIGURE_FALLBACK=only`); read the new records. Skip quests older than frame version 5 and list them.
- [ ] Output `out/p0_report.md`. Go = placeholders 0 and caption disagreements 0 on the replay, worst-case removals
  reported as the cost. Stop and report to the maintainer if the worst case removes lines from a large share of figures,
  or if no stated constant matches a plan reference.

The replay cannot show whether a model registers a curve (only the A/B can); it reports "lines removed if no repair
succeeded".

## Task 1: a `value=` constant takes a plan reference's identity, or is refused

Files: `core/frame_figure.py` (`ref`, new `_plan_match`), `core/analysis_figures.py` (`_plan_for`, `_figure_handle`);
test `tests/test_ref_value_identity.py`.
Run: `... fx349a tests/test_ref_value_identity.py tests/test_frame_figure.py tests/test_analysis_figures.py tests/test_figure_series_levels.py tests/test_skeleton.py tests/test_figure_curve_xcheck_range.py tests/test_figure_layout.py`

- `FigureCalls._plan_match(self, value, axis, panel) -> list[str]`: plan reference ids whose value at this panel's setting
  equals `value` within the typed-digit tolerance (Phase 0 decides whether the plan precision is also needed).
- `ref(..., value=)`: exactly one match -> `ident=<ref id>`, `source="plan"`, plan label, plan value; else `ValueError`:
  "this constant does not match exactly one reference of the plan. If it is a theory value, add it to the analysis as a
  named curve or reference; if it is a result's value, use `at=`."
- `_plan_for` adds `plan["reference_values"] = {id: {setting key: number}}`; `_figure_handle` passes it (guarded on
  `takes`). UNCONFIRMED: whether `Figure.__init__` in `skeleton.RUNTIME` needs the keyword.

Tests: `test_value_matching_one_plan_reference_takes_its_identity`, `test_value_matching_two_plan_references_is_refused`
(message names both ids, nothing drawn), `test_value_matching_only_a_result_value_is_refused` (message mentions `at=`),
`test_value_matches_only_at_the_panels_own_setting`, `test_value_within_typed_digits` (0.05 vs 0.0500000001 matches;
0.0475 vs 0.05 does not), `test_stated_source_no_longer_written`; update
`test_ref_has_three_sources_and_a_stated_value_gets_a_neutral_label`.

## Task 2: formula curves registered in `derived.json`, drawn by name

Files: `core/analysis_frame.py`, `core/frame_figure.py`, `core/analysis_figures.py`; test `tests/test_registered_curves.py`.
Run: `... fx349b tests/test_registered_curves.py tests/test_frame_figure.py tests/test_analysis_frame.py tests/test_analysis_figures.py tests/test_skeleton.py tests/test_figure_curve_xcheck_range.py tests/test_manifest_derived_lists.py tests/test_figure_claims.py tests/test_figure_data_check.py tests/test_figure_captions.py tests/test_g18_figure_facts.py`

- Registration: `derived[id] = {"status":"ok","role":"curve","carries_result":false,"x":<setting id>,"levels":{<key>:{"xs":[...],"ys":[...]}},"by":"derive_<id>","inputs":[...]}`; `"curve"` is not in `_COUNTING_ROLES`, so never a result.
  Written by a `derive_<id>` function of `derive.py`. UNCONFIRMED: how PLAN/DERIVE declare roles
  (`analysis_frame.py:112,230`); read 100-300 and `core/code_review.py:1487` first.
- `FigureCalls.formula(self, panel, curve_id, *, key=None)` replaces `curve()`: looks up host attribute
  `registered_curves`; refuses an unknown id (naming the registered ones) or a panel level with no entry; records
  `{"id","kind":"curve","label":<registered name>,"registered":true,"level","x","y"}`. `curve()` raises a `ValueError`
  pointing to `fig.formula`. `RECORD_VERSION = 3`.

Tests: `test_formula_draws_registered_curve_by_name`, `test_formula_unknown_id_is_refused_naming_registered`,
`test_formula_without_level_for_panel_is_refused`, `test_old_curve_call_is_refused_with_pointer_to_formula`,
`test_curve_role_is_not_counting_and_carries_no_result`, `test_registering_a_curve_leaves_every_result_unchanged`,
`test_version2_records_still_read`.

## Task 3: legend entries from what differs; identical entries merged

Files: `core/frame_figure.py`; test `tests/test_legend_from_record.py`.
Run: `... fx349c tests/test_legend_from_record.py tests/test_frame_figure.py tests/test_figure_series_levels.py tests/test_figure_layout.py tests/test_figure_labels.py tests/test_figure_mutations.py tests/test_figure_fixtures.py tests/test_figure_mark_roundtrip.py tests/test_figure_captions.py tests/test_plot_style.py tests/test_analysis_figures.py`

- `_identity(self, e_or_ref) -> dict` (`id`, `level`, `ref`, `value`); `_differing(identities) -> set[str]`;
  `_entry_text(self, ident, differing, names) -> str`.
- `_merge_identical` also merges reference lines with the same `(axis, id, value)` and registered curves with the same
  `(id, level)`.
- What all entries share goes to `panel["shared"]` (written in `finish`) for the axis label / caption.
- Colour key stays the identity, so one series keeps one colour across panels.

Tests: `test_entries_name_only_what_differs`, `test_series_of_different_results_keep_result_names`,
`test_identical_reference_lines_are_one_entry`, `test_reference_entry_carries_plan_name_and_value`,
`test_entries_never_collide_and_never_empty`, `test_no_placeholder_text_reaches_the_legend`,
`test_legend_text_comes_from_record_not_artist_labels`.

## Task 4: caption facts written by FI from the record

Files: `core/engine.py` (`_call_extras`, `_call_drawn`, `_caption_from_calls`, new `_call_reference_sentence`,
`_call_empty_panels`); tests `tests/test_caption_facts_from_record.py`, extend `tests/test_figure_captions.py`.
Run: `... fx349d tests/test_caption_facts_from_record.py tests/test_figure_captions.py tests/test_fi_places_figures.py tests/test_figure_records.py tests/test_replicate_figures.py tests/test_plot_style.py tests/test_figure_claims.py tests/test_g18_figure_facts.py tests/test_paper_figures.py tests/test_figure_own_paragraph.py`

- References clause: "<plan name> at <value>" per distinct drawn line after merge, from the record.
- No "a curve" / "N curves" wording; a registered curve by its name.
- Empty or removed panels named ("Panel N draws nothing"; "a line was left out of panel N", from `removed`).
- UNCONFIRMED: the writer of the "Marked:" clause (`grep -n "Marked" core/*.py`); make it read the same list.

Tests: `test_caption_lists_every_drawn_reference_with_name_and_value`, `test_caption_reference_count_follows_record_not_text`,
`test_caption_names_empty_panel`, `test_caption_has_no_a_curve_wording_for_a_registered_curve`,
`test_version2_record_caption_unchanged_for_series_only_figures`.

## Task 5: one repair round, then unregistered lines removed and named in QUALITY.md

Files: `core/engine.py`, `core/frame_figure.py`, `core/analysis_figures.py`, `core/quality_report.py`; test
`tests/test_unregistered_lines_removed.py`.
Run: `... fx349e tests/test_unregistered_lines_removed.py tests/test_analysis_figures.py tests/test_analysis_frame.py tests/test_frame_figure.py tests/test_figure_failure_keeps_results.py tests/test_figure_fi_fault_keeps_run.py tests/test_figure_taken_out_everywhere.py tests/test_figure_data_check.py tests/test_figure_overlap_repair.py tests/test_engine_smoke.py`

- A refusal is a `refused` outcome sent to the existing figure repair; the message names the line and says to register it
  in the analysis. No second repair counter. UNCONFIRMED: which loop a `refused` outcome reaches; trace it and wire the
  registration repair into the loop that already rewrites `derive.py`.
- The analysis re-run reads the recorded trials (no simulation).
- After the round, an internal switch (mirroring `FI_FIGURE_FALLBACK`; kept internal) makes the frame skip such a line and
  append `record["removed"] = [{"panel": n, "what": "...", "why": "..."}]`; never drawn. Empty panels follow the existing
  `_empty_panels` path.
- QUALITY.md: one plain line per removed line from `_analysis_problems` ("Figure 2: a line at 0.05 was left out, because
  FI has no record of where its value comes from.").

Tests: `test_refused_constant_message_names_the_line_and_the_registration_step`,
`test_after_repair_round_unmatched_constant_is_not_drawn`, `test_curve_without_registration_removed_not_kept_as_unverified`,
`test_removed_line_absent_from_record_legend_caption_and_named_in_quality`, `test_repair_rerun_does_not_simulate`,
`test_panel_left_empty_follows_existing_empty_panel_path`, `test_quality_md_names_removed_line_in_plain_words`.

## Task 6: record-vs-legend/caption consistency check

Files: new `core/figure_text_check.py` (or `core/figure_claims.py` if it fits; UNCONFIRMED, read its first 120 lines);
test `tests/test_legend_caption_consistency.py`.
Run: `... fx349f tests/test_legend_caption_consistency.py tests/test_figure_claims.py tests/test_figure_captions.py tests/test_figure_records.py tests/test_g18_figure_facts.py tests/test_paper_figures.py`

- `check_figure_text(calls: dict, legend_texts: list[str], caption: str) -> list[dict]` with kinds `legend_extra`,
  `legend_missing`, `ref_count`, `ref_value`, `empty_panel_unsaid`. A finding goes to QUALITY.md, never ends the quest.
- Planting helper in the test file only: swap two entries, rename one, drop or add a reference in the caption, change a
  value, omit an empty panel.

Tests: `test_true_text_passes_on_every_fixture_record` (neutral fixtures from real records, 0 findings),
`test_planted_mislabel_catch_rate_at_least_95_percent`, `test_each_plant_kind_is_caught`,
`test_no_false_flag_on_merged_entries_and_registered_curves`.

## Task 7: prompts

Files: `agents/figure_primitives.md` (lines 9, 11, 21-24), `core/analysis_frame.py` figure stub text (249),
`core/run_checks.py:297,527`, `core/figure_data_check.py:372`, the repair message in `_repair_typed_figures`
(UNCONFIRMED location); test `tests/test_figure_prompt_rules.py`.
Run: `... fx349g tests/test_figure_prompt_rules.py tests/test_skeleton.py tests/test_analysis_frame.py tests/test_public_text_english.py`

Tests: `test_prompt_mentions_formula_and_not_curve_call`, `test_no_prompt_names_a_removed_call`.

## Task 8: docs

`dev/registry.md` (frame entry ~526), `docs/capabilities-reference.md` (figure legend and caption), user docs naming
"a curve" / "Reference line"; `python scripts/wiki_sync_index.py --check`; `tests/test_public_text_english.py`.

## Verification (nothing merged before these pass)

1. Rerun Phase 0 on the finished frame: placeholders 0, caption disagreements 0, planted mislabels caught >= 95%; report
   worst-case removals beside the gain.
2. Sonnet A/B on 4-6 figures with a formula curve or unmatched constant (cost stated first; a few hundred thousand
   tokens): does the code model register the line after the repair message; lines kept vs removed; rounds used.
3. Small-study round (as fc11): G3 fails in fewer than 4 of 5 papers; no new false stop; results of counting roles
   identical to a run without the change on the same trials; read QUALITY.md removal notes.

## Order

Phase 0, then Tasks 1-8 in order (1 and 2 block the rest; 3 needs 2; 4 needs 3; 5 needs 1-2 and feeds 4; 6 needs 3-5).
One commit per task.
