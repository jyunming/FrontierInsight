# Module registry

What already exists, by area, so a new PR extends something instead of rebuilding it. This is a **module-level**
index (an entry per file, or per tightly-related group of files, never per function) — for the exact function, open
the file and search; for what a feature *does* from a user's side, see `docs/capabilities-reference.md` instead.
Grep this file for a keyword before adding a new module, a new node, or a new top-level concept; update it in the
same PR that adds, splits or renames one.

## The engine and its state

- `core/engine.py` — the async LangGraph DAG (`Engine`), every node, every gate, `QuestState`/`QuestArtifacts`. The
  biggest file in the repo; read `docs/architecture.md` before adding a node here.
- `core/protocol.py` — the typed `ResearchProtocol` a quest's design may declare.
- `core/frozen_protocol.py` — freezing a protocol before the first run, and the amendment approve/apply/decline flow
  (also used, unmodified, for the tamper-recovery restore — see `propose_tamper_recovery`). Who approved the frozen
  protocol (`approved_by`) is decided in `Engine._freeze_protocol_if_due`; an oracle the oracle gate had added to the
  plan (`Engine._declare_oracles`) is recorded in `.fi/oracles_added.json`, and `Engine._hold_added_oracles` stops the
  quest again (the `plan` pause, `_pause_for_plan(added=...)`) before the freeze under `pauses.plan: ask`.
- `core/protocol_check.py` — static check: does the experiment's source hold to the frozen protocol? Also the plan's
  notes on the protocol (`*_notes`; `run_count_notes`: repeated runs of a design that names nothing random).
- `core/run_manifest.py` — runtime check: does what the simulation says it ran (`run_manifest.json`) match the
  protocol's exact Cartesian product? Also the two-script split-analysis lint, the strata of a `given` mean
  (`NESTING`, `DERIVED`, `_given_findings`, `strata_coverage`, read by `metric_spec.coverage_gaps`), and
  `once_per_setting`: for a `run_cell` with no randomness, `runs_per_setting` is left out of the comparison (one run
  per setting; `Engine._run_cell_randomness` and `trial_runner.run_cell_again` decide there is no randomness).
- `core/oracle_check.py` — an oracle the script must pass before its main run, judged by the engine against the
  protocol's own expected value and tolerance, not by the script's self-report. Also owns what a repair may propose
  about a check (`proposals`, never applied without a person; `undisputed` / `disputed_failing` split the problems a
  repair may still fix from the checks it called wrong, which `Engine._oracle_gate` never rewrites the script for)
  and the note that carries the engine's verdicts to
  `analyze` (`analysis_note`). An oracle that names a `case` and a `measure` is measured by the engine, not the script
  (`case_of`); `loose_tolerance` warns when the tolerance would pass a lower-order method than the oracle claims
  (`order`); `script_measured` lists the values that are still the script's own word (`core/evidence.py` never counts them toward
  `independently_validated`, one script or two), and `last_judged` / `engine_passed` / `not_passed` read the record the
  evidence level trusts (at least one value FI measured and passed, none unpassed); `with_run_problems` merges the
  reasons a value could not be measured. Also owns an oracle's kind (`KINDS`, the six; `kind_of` reads the older names)
  and where its expected value comes from: `source_gaps` (a `reference` that is empty or names a source the quest did
  not retrieve; matched against `retrieved_sources` — the `[n]` labels of `_labelled_sources`, titles, DOIs — against
  an equation of the plan's model, or a second implementation's "shares no code") and `model_notes` (the model block
  itself). `Engine._check_plan_sources` warns, and under `rigor_profile: research` stops at the plan
  (`_pause_for_plan(unsourced=...)`), at the plan step and right before the freeze; the freeze keeps the sources as
  numbered then (`sources` and its own `sources_sha256` in `needs/FROZEN_PROTOCOL.json`; `load` sets
  `sources_problem` on an edit), and `Engine._oracle_source_gaps` (called by `_write_evidence`; a record frozen
  without `sources` is judged only on `empty_references`) passes the gaps to `evidence.assess(oracle_source_gaps=...)`.
  What the engine reads is mapped at read time, never renamed in the plan (a frozen protocol hashes as written):
  `oracle_check.reference_of` / `kind_of` / `kind_written` (a reference under `source`/`basis`/`derivation` or inside
  the check's text; a kind under `type`, or in words such as "normalization"), `model_view` / `equation_items` /
  `model_missing` (the model's parts under other names; equations as lines or an id->formula mapping), `unsourced`
  (`(name, why)` per check), `DERIVATION_RULE` (a derivation needs at least one equation with `=`).
  `plan.normalize_model` / `normalize_protocol` convert only shapes they used to refuse (equations as text or a mapping,
  a kind/reference written as a list); `plan.repair_protocol` leaves out one unreadable check, not the list;
  `plan.listed_sources` reads *The sources this quest found* back from `plan.md`; `plan.refresh_model_section` shows the
  model section again from the block after a rewrite. Under research, `Engine._settle_plan_sources` (all four call
  sites) first runs `_fill_plan_sources`: once per quest (`.fi/plan_sources_asked.json`, written once the model has
  answered; not on a later pass, nor on a plan.md that cannot be read) a targeted
  `_rewrite_plan(_fill_request(...), by="engine")`, put back when `_fill_changed_more` finds anything but sources,
  kinds and the model changed. `_rewrite_plan` gives every rewrite `_plan_checks_note` (what FI reads and what is
  missing, `$checks` in `agents/plan_revise.md`). A rewrite whose design block cannot be used goes through
  `Engine._usable_design_block`: `plan.repair_design` / `repair_block` first (only repairs that change no key or value:
  a plain value with `: ` in it as a folded block, tab-only indentation), then ONE call for the block alone
  (`plan.block_fix_request`: `plan.yaml_problem`'s line, column, parser message and the lines around it; the reply is
  read by `plan.block_from_reply` and spliced back with `plan.replace_design_block`). At most two calls a rewrite.
  `oracle_check._code_list` keeps code lists (`y0=[1, 0]`, `[0, 1]`) from being read as citations, and
  `oracle_check._not_found` / `labels_note` name the quest's real source numbers when one cited does not exist. The stop (`_pause_for_plan(unsourced_checks=...)`) offers three ways
  on and writes `needs/UNSOURCED_CHECKS.json`.
- `core/accepted_checks.py` — going on with checks whose expected value has no stated source: `write_pending` /
  `pending` / `clear_pending` (what the stop named, and what the stop before named, so the next stop says what changed),
  `accept(root, who, via=...)` (`needs/UNSOURCED_CHECKS_ACCEPTED.json`; a name is required), `fingerprint` (a choice
  binds to a check's name and numbers: expected, tolerance, case, measure), `covers` / `chose` / `changed_since` (the
  engine goes on only when every unsourced check was chosen as it is now; `Engine._unsourced_oracles` and
  `_not_confirmed_names` mark the evidence gap "source not confirmed" in `_oracle_source_gaps` and the freeze's
  `approved_by`, for checks still without a source), `disclosure(root, names)` (the writer's note). A source the fill
  wrote after the person read the plan is named in the freeze too (`.fi/plan_sources_filled.json`). Surfaces:
  `go_on_by_itself` (FI found no source after asking once: recorded as `AUTOMATIC`, the same marks; `who_text`),
  `went_on_entries` (the gate's automatic go-on for checks that measured nothing too). For quests an earlier FI stopped:
  `launch.py --accept-checks <quest> --approve-as <you>` (`_accept_checks`), `POST /api/quests/{id}/plan/accept-checks`
  and `GET .../plan/unsourced` (the quest page's *Go on as it is*), `@fi /accept-checks` (`skills.ts::runAcceptChecks`).
  Also going on although a known-answer check was measured and failed (the same flag, button and command; `accept`
  reads which stop it answers from `.fi/pause.json`): `offer(found, oracles, judged)` (offered only when every problem
  is a measured failure; else why not), `script_version` (hash of the script the checks ran + helper modules beside it + the model's package),
  `accept_failing` (`needs/FAILED_CHECKS_ACCEPTED.json`, bound to the fingerprint and the code version),
  `failing_pending` (the stop's `go_on` in `needs/ORACLE_CHECK.json`; the web payload's `failed_checks`), `went_on_by`
  / `no_longer_applies`, `gap` (the evidence's sentence, person or `automatic`), `failing_disclosure` (the writer's
  note). Engine side: `_oracle_gate` honours a choice via `_chosen_go_on` before any plan change or repair, an
  exploration goes on by itself via `_goes_on_by_itself`, the record's status is `went_on_failing`
  (`_measuring_code_version`); `core/todo.py::waiting` lists it, `core/evidence.py` keeps the gap.
  `vscode-frontier-insight/src/resume-args.ts` (no VS Code import, run by a test with node): `splitRevisePlan` (`/resume
  <id> --revise-plan "..."` goes the `/plan` way) and `unknownResumeFlags` (any other flag is said, and nothing runs).
  `launch._plan_sources_status` prints, after `--revise-plan`, which checks still lack a source.
  `sources_block` is the numbered list plan.md shows (*The sources this quest found*). The model block's shape (`protocol.model`) is `core/plan.py`'s
  (`normalize_model`, `repair_model`, the *The model behind the numbers* section of `render`). Also owns the equation
  labels: `generating_equations` (the model's `generates` ids), `unlabelled_equations` (those no comment or docstring
  of the simulation carries, `# E1`; read with `tokenize`/`ast`, never the code) and `label_gaps` (the sentence).
  `Engine._label_equations` (end of `_node_implement`) asks once for missing labels and keeps the answer only when the
  syntax tree is unchanged; `Engine._simulation_sources` picks the simulation and its helper modules;
  `Engine._check_equation_labels` (in `_node_execute`, before the oracle gate and again when a repair in the gate
  rewrote the simulation) warns, and under `rigor_profile:
  research` stops (`equation_labels` pause); `Engine._equation_label_gaps` (called by `_write_evidence`) passes them to
  `evidence.assess(equation_label_gaps=...)`, a gap below `independently_validated`.
- `core/oracle_forms.py` — how an oracle is written so the plan and the simulation mean one number by it. The formula
  language of `measure` (`evaluate`, `problem`, `names`, `is_name`, `FUNCTIONS`; parsed with `ast` and walked, never
  run; a returned name alone is the old form), applied by `trial_runner.measure_oracles`. One numeric form per kind
  (`VIOLATION_KINDS` expect 0, `QUANTITY_KINDS` the quantity itself): `enforce` / `apply_to_plan` rewrite what can be
  rewritten without changing a verdict and return precise requests for the rest (`request`), `describe_changes` says
  what a revision changed. `mismatch` / `mismatches` / `dry_run_request` read the oracle gate's first measurement as a
  test run of the checks; `passes_on` says whether a changed check passes on the test run's own values. Engine side:
  `Engine._guide_oracles` / `_hold_oracle_forms` (plan time, from `_node_plan`, once per quest:
  `.fi/oracle_guidance.json`) and `Engine._revise_after_dry_run` (from `_oracle_gate`, once: `.fi/oracle_dry_run.json`;
  its changes and removals go through `_oracles_added_write` with a `reason` and `removed` to the plan stop before the
  freeze, merged by `Engine._note_engine_change`, which `_declare_oracles` uses too); both ask the plan through
  `Engine._revise_checks_only`, which keeps only the changes to `protocol.oracles` and the criteria reading them.
  `core/rerun_from.py` moves both markers aside with the plan (and the test-run one with the design).
  `expected_formula` (a check's expected value as one formula FI computes itself, with `evaluate(..., special=True)`
  on the check's numeric `case` settings): `formula_value` / `formula_findings` (only checks `oracle_triage.correctable`
  allows and that can be judged), `formula_request` (the part of the SAME one request about the checks),
  `apply_formulas` (a formula that still disagrees by more than the check's own tolerance sets `expected`; the
  tolerance, mode, case and measure are never touched), `EXPECTED_FORMULA_LANGUAGE` (generated from the
  calculator's tables, spliced into `_PLAN_DIRECTIVE`). Applied by `Engine._apply_expected_formulas` from
  `_hold_oracle_forms`: before the freeze and any run, once (`formulas` in `.fi/oracle_review.json`), as the
  engine's change (`_note_engine_change`, plan version `by="engine"`). `oracle_triage.blind` withholds it.
  The plan.md section is `HEADING`; `plan.raw_design_block` / `plan.edit_design_block` read and edit the block as
  written, `plan.add_to_section` / `plan.refresh_model_section` keep the prose in step.
- `core/oracle_review.py` — a second opinion on the plan's checks (`agents/oracle_review.md`, node `oracle_review`):
  `prompt_parts`, `parse` (strict: only declared checks and equations), `findings`, `request`, `plan_lines`. Called by
  `Engine._review_oracles` from `_hold_oracle_forms` (one call; its findings share the one `plan_revise` request with
  the form requests; `.fi/oracle_review.json` keeps the answer and how far the look got, so a resume asks nothing
  again, and `core/rerun_from.py` moves it aside with the plan). The reviewer model is `provider.node_models.oracle_review`; the VS Code node picker lists it
  (`OTHER_NODES` in `vscode-frontier-insight/src/interview-core.ts`). Under `rigor_profile: research`,
  `independence_gaps` (with `canonical_model` / `same_model`, `WRITER_NODES`, `NOT_REVIEWED`, `HOW_TO_NAME_ANOTHER`)
  makes the review a condition of `independently_validated`: a usable answer covering the frozen checks
  (`fingerprint(s)`; the record's `read` and `after_look`, written by `_review_oracles` / `_hold_oracle_forms`), and
  `.fi/model_calls.jsonl` naming a reader model (the call `call_id` in the record names) that is none of the models
  that answered `plan` / `plan_revise` / `design` / `design_self_critique` (`PROXY_PROVIDERS` not taken at their
  word). `Engine._independence_gaps` passes it (and `hidden_check.evidence_gaps`) to
  `evidence.assess(independence_gaps=...)`.
- `core/hidden_check.py` — FI's own check at a setting the code never saw (`rigor_profile: research`): `candidates`
  (invariant / symmetry / second implementation with a case), `derive` (a smaller decimal step, else a value between
  the grid's decimal settings, else another whole-number grid value), `run` (through `trial_runner.measure_oracles`,
  judged by `oracle_check.judged`; reused only for the record the trace names, same `code_sha` and `checks_key`),
  `write` / `load` / `record_sha` (`needs/HIDDEN_CHECK.json`), `evidence_gaps` (reads the trace's `hidden_check`
  events, `Engine._hidden_check_written`). Called by `Engine._hidden_check` in `_node_execute` after a successful
  two-script run; never stops a quest.
- `core/criteria.py` — how a quest judges whether its code got better: the protocol's `criteria` (two to five checks of
  correctness, each from a declared oracle or FI's trial record; never a headline metric or the
  script's results). `normalize` (strict, called by `plan.normalize_protocol`) / `repair` (a draft, called by
  `plan.repair_protocol`), `describe` / `plan_notes` / `countable` (the plan's *How we will judge whether the code got better* section,
  rendered by `plan._criteria_lines`), `evaluate` / `se_rate` / `meets` (the values after a run), `record` / `history`
  (`.fi/criteria_history.jsonl`) and `summary_line`. `receipts.design_core` leaves the criteria out of what the
  methodology audit vouches for; `Engine._hold_design_to_frozen` leaves an amendment's unusable criteria out. `Engine._propose_criteria` (one search, one more question,
  `agents/plan_criteria.md`) runs in `_node_plan` when the draft names none; `Engine._record_criteria` runs at the end of
  every `_node_execute` (oracle values from `oracle_check.last_judged`, trial values through
  `trial_runner.recorded_series`, the commit through `code_project.head`; a run that exited 0 with results also stores
  `result_digest`, one hash per result via `improve.headline_digest`, and fills code/CHANGELOG.md's waiting "Did the
  results change" lines through `changelog.run_line` + `code_project.fill_pending`).
- `core/changelog.py` — the words of `code/CHANGELOG.md`: `CATEGORIES` (added / changed / fixed / tidied; `category`
  validates, anything else is `changed`), `heading`, the "Did the results change" line (`PENDING` until a run;
  `run_line` compares a run's criteria row with the last earlier full run that has a `result_digest` — checks before
  and after, the study's results changed or not by name only; `round_not_kept`, `put_back`, `not_measured`,
  `documentation_line`, `together`). The category comes from the caller: `_node_implement` (added / fixed for a
  review's re-run / changed after a redesign), `_record_criteria` (fixed when FI repaired the script, else changed),
  `_node_analyze` (changed), the improve loop (changed; a kept round waits for the full run, a revert passes
  `undo_pending`); `code_project.record_change` makes a change to `DOCUMENTATION` files alone (README.md, requirements.txt, run.py,
  study.json, METHODS.md, tests/test_oracles.py: FI's own runs never read them) `tidied`, and commits a CHANGELOG.md-only change with no entry.
- `core/improve.py` — the improve loop and the ratchet: one edit per round to the simulation against the protocol's
  criteria. `parse_edit` / `check_edit` (the edit refused before it runs: another file, not found once, does not
  parse, `normalized` unchanged = print/comment-only, a version already tried by `fingerprint`, a check's expected
  value or a criterion's target `planted`, `_measure_exprs` changed = the checked number worked out differently,
  `_asks_for_a_setting` = a new comparison with a check's setting), `guard_hashes` / `guard_bytes` / `put_back` (the files a round's run must not change),
  `compare` / `verdict` / `all_met` (per-criterion ratchet by each criterion's own tolerance), `series_of` (a round's
  trial values), the prompt blocks (`criteria_block`, `history_block`, `code_block`; never `result_json` or the
  metrics), `snapshot` / `restore` / `save_snapshot` (byte-exact `code/*.py`; `forget_bytecode` so a same-size version written
  in the same second is never run from Python's stale cache), `tree_bytes` / `put_back_tree` (all of `code/` and its
  git settings and hooks, links never followed), the record `.fi/improve.json`
  (`load` / `save` / `summary`) and `write_note` (the writer's note). Graph node `improve` (after `execute_reflect`;
  `rerun` → `execute` once for a kept version): `Engine._node_improve` / `_improve_skip` / `_improve_loop` /
  `_improve_measure` (engine-run oracle cases and trials, `needs/ORACLE_CHECK.json` untouched) /
  `_improve_stop_for_regression` (research: `improve` pause) / `_improve_resume_after_block` /
  `_improve_put_back_first` (a loop cut short) / `_improve_save_raw` / `_improve_put_back_raw` (the first run's trial
  record, on disk in `.fi/improve/raw/`) / `_improve_fresh_run` (`_FRESH_SCRIPT` and no cached trials for a rerun) /
  `_improve_after_rerun` (the headline "changed" note via `code_project.record_note`), prompt `agents/improve.md`,
  config `engine.improve_rounds`; each round's row is `criteria.record(..., improve=...)`.
- `core/optimisation_plan.py` — the kind of study (`study_type`: `measure` or `find_best_design`; `study_type_of`
  reads a design with an `optimisation` block as a search whatever it says) and the plan's `protocol.optimisation`
  block for a search for the best design: `normalize` / `repair` (hooked into `core/plan.py`'s `normalize_protocol`
  and `repair_protocol`), what FI does with what the plan left out, computed at use and never written into the block
  (`check_levels`: the plan's finer check values or the fixed rule; `effective_method`; `improvement_rule`), the
  evaluation count worked out at plan time (`budget`), the *What is being optimised* section of plan.md
  (`plan_lines`), the topic's numbers in the block (`numbers`, read by `protocol_check.plan_notes`), and the one
  clarify question for an ambiguous topic (`classify_topic`, `add_study_type_question`, `resolve_answer`; added in
  `Engine._node_clarify_questions`, the plan prompt's `_STUDY_TYPE_DIRECTIVE` and `Engine._settle_study_type`).
  `Engine._stop_if_the_search_cannot_start` (in `_node_design`) stops a search that cannot start (no block or budget,
  `split_analysis: false`, `background_jobs`) before anything is implemented or run. Before it stops for a missing or
  unreadable block, `Engine._ask_plan_for_the_search` (called at the plan step, before the person reads the plan, and again
  in `_node_design`) asks the plan's model to write the block (`optimisation_plan.ask_request`; at most twice and six chat calls per plan and
  per model, recorded in `.fi/search_block_asked.json`; only `study_type` and `protocol.optimisation` are kept from the
  rewrite, `Engine._keep_only_the_search`, which also keeps every part of the plan's own block that was readable; a plan saved meanwhile is not overwritten, `_PlanEditedMeanwhile`; "measure instead" is checked by `Engine._check_measure_instead`; `plan.refresh_optimisation_section` shows the section again). The stop speaks in
  plain words (`optimisation_plan.plain_gaps`) and offers another model for the plan step or
  `optimisation_plan.MEASURE_INSTEAD` as a `--revise-plan` request; it never asks a person to write the block.
- `core/optimise.py` — the engine runs the search for the best design: `OptimisationRunner` (the executor stand-in
  `_node_execute` picks when the design has an `optimisation` block and two scripts, instead of
  `trial_runner.TrialsRunner`; it runs the search, then `experiment.py` with `FI_OPTIMISATION` / `FI_BEST_DESIGN`, and
  `FI_TRIALS` for the coarse scan), `run_search` (the coarse scan through `trial_runner.run_trials`, each design through
  the trial harness in its own process with a nonce, a library method's steps through `.fi/optimisation/optimise_search.py --drive`
  in the quest's Python, the budget and `execution.timeout_s`, the run key cache in `.fi/optimisation/run.json`),
  the record FI alone writes (`raw/optimisation_ledger.jsonl`, `results/best_design.json`; `restore` / `read` put
  FI's copy back), `summary_lines` (run.log), `OptimisationRunner._check` (runs `optimum_check.run_check` after the
  search and before the analysis, then `attach_check` puts the verdict and the check file's hash into
  `results/best_design.json` under `check` and the hash into the run record (`check_sha256`), re-saved so `restore`
  keeps it and `optimum_check.run_check` reuses a kept check only when it is that one; the analysis gets
  `FI_OPTIMUM_CHECK`; nothing in `_check` can stop the quest), and
  `complete_case` (an oracle's case gets the fixed conditions, search settings and baseline, in `Engine._oracle_gate`),
  `with_fi_record` (FI's numbers added to the analysis's RESULT_JSON as `fi_search`, the check's under `fi_search.check`),
  `searches_itself` (an optimiser in the simulation's code: a warning). `Engine._node_execute` calls
  `_stop_if_the_search_cannot_start` too.
- `core/best_design_report.py` — what a search for the best design hands over: `section` / `for_paper` (the paper's
  "Best design found" section from `results/best_design.json` and `needs/OPTIMUM_CHECK.json`), `mark_paper` /
  `without_block` (between `<!-- fi:best-design -->` markers; `Engine._mark_best_design` in `_node_write`),
  `strip_for_checks` (the claim check), `records` (what `number_provenance.check(optimisation=)` traces the
  section to), `write_note` (the writer's note), `summary_line` / `files` (the `[FI] best design:` lines in
  `launch._report_best_design`, the web quest API's `best_design`, the VS Code chat), `write_readable`
  (`results/best_design.md`).
- `core/optimise_refine.py` — a refine of a search for the best design, read without a model: `read_request` (search
  further / a new study / unclear, said back in plain words), `add_round` (`.fi/optimisation/continue.json`,
  read by `optimise.read_rounds`; `optimise_search.search(rounds=)` runs the rounds, `optimise.run_search`
  replays the search before them from FI's record), `hint` (the review pause). Engine side:
  `Engine._optimise_refine_request`, `QuestState.optimise_refine`, the `human_feedback` routes `search` (→ `execute`)
  and `ask_again` (→ `human_feedback`).
- `core/optimum_check.py` — the engine checks the search's best design at finer numerical settings, through the same
  harness (`optimise._evaluate`, fresh seeds for `run_trial` via `check_seeds`): `check` (a generator of evaluate
  requests, like `optimise_search.search`; `check_sync` drives it in one process for the tests), `levels_of` (the finer
  levels, the plan's `check` values or `optimisation_plan.check_levels`' fixed rule, and where each came from),
  `numerical_error` (Roache's grid-convergence estimate per design), `candidates` (the best design of each starting
  point), the six checks (refinement, limits, improvement over the baseline against the plan's threshold or the
  numerical error of the two designs, starting points, neighbourhood nudges, the search's budget) and one `verdict`;
  `run_check` writes `needs/OPTIMUM_CHECK.json` (cached in `.fi/optimisation/check.json`; `restore` / `read`),
  `summary_lines` / `summary_line` (run.log), `attach_summary` (the part in `results/best_design.json`), and
  `evidence_gaps` (read by `evidence.assess`: each failed or unfinished part is a gap at the level it bears on). Never
  pauses a quest.
  `Engine._run_manifest_problems` returns `not_applicable` for a search; replicate seeds are not run for it; the
  code-writing prompt gets `_SEARCH_PROTOCOL` (via `_split_block`) and the repair prompts `_SEARCH_REFLECT_*`.
- `core/optimise_search.py` — which design comes next, standard library only and importing nothing from FI (copied
  into `.fi/optimisation/` and into `code/fi_search.py`): `search` (a generator of evaluate / drive requests; `bounded_local`,
  `global_then_local`, `exhaustive`, and a scipy / Optuna method replayed step by step by `drive`), `run_sync` (the
  same search driven in one process: `code/run.py`, the tests), `judge` (feasibility and failures), `best_design`
  (the record), `ledger_lines`, `scan_grid` / `scan_rows` (the coarse scan as trial-runner cells), and copies of
  `optimisation_plan.effective_method` / `check_levels` that `tests/test_optimise_runner.py` keeps equal.
- `core/metric_spec.py` — a metric spec (estimand, estimator, contrasts) per headline number, and the statistics that
  follow from it; `core/stats.py` is the pure-stdlib estimator/interval/test library underneath it.
- `core/evidence.py` — the six-level evidence ladder (`assess`, `summary_line`, `upgrade` for older records; for a search for the best design `assess` adds `optimum_check.evidence_gaps` level by level and `_OPTIMUM_INFO`'s blind spot and artifacts); `_trace_completeness_gaps` reads the trace's `quest_finalized` seal (written last by `Engine._seal_trace`, naming `SEALED_FILES`); `SEALED_LEDGERS` and `SEALED_QUERIES` (the signed record of the search queries, `engine._record_query_set` / `_query_set_standing`; a person's own in `inputs/search_queries.txt` ; the same file also holds each pass's source verdicts, stages `floor` and `screen`: `Engine._record_floor_verdicts` / `_record_source_verdicts`, digest fields `_SOURCE_VERDICT_HASHED`, so no seal change) are required in the seal; `read` / `verify_seal` are how every surface reads `needs/EVIDENCE.json` (a record written before its seal says `trace_seal: pending`).
- `core/disclosure.py` — what the paper says about how its result was reached: `paragraph` builds the methods
  paragraph from `needs/DESIGN_HISTORY.json` (`revisions`: `post_hoc` entries not marked `after_results: false`, which
  `engine._append_design_revision` / `_anything_ran` records; `reason_phrase` into the closed `REASONS`) and
  `.fi/attempts.jsonl` (`runs`: the `run` lines, every one but the last discarded, by outcome); `without_block` (called
  by `Engine._node_write` right after the writer returns: removes markers, blocks and lead-word paragraphs until none is
  left) and `mark_paper` (at the end of the write node: the paragraph at the end of the methods, else before the
  results); `strip_for_checks` (first in `stat_claims.normalise` and `numeric_oracle.extract_paper_numbers`, so
  `number_provenance` too, and the claim check's copy of the paper: removes the block only when `is_engine_paragraph`);
  `unconfirmed_gap` / `confirmed_after_last_change` (`evidence.assess`: a revision after results with no confirm run
  after it, read from `.fi/phased.json` `design_revisions_at_confirm`, beyond the `frozen_protocol.post_hoc`
  amendments, is a `publication_ready` gap unless a `_PHASED_GAPS` entry already names the stage; none for a survey).
  `paper_trim.candidates` never offers a sentence inside an engine note (`<!-- fi:... -->`).
- `core/audit_log.py` — the hash-chained, redacted per-quest trace (`STAGE_PROGRESS`/`_ProgressOnly` for the curated
  console/web view live in `core/engine.py`, next to `_quest_logger`).
- `core/record_anchor.py` — FI's note of how far it wrote the trace and the `.fi/` record files
  (`.fi/record_heads.sqlite`: `appending` around every line, `before_reading` in `AuditLog._open`; copied into
  `QuestState.record_anchor` at each step's end by `Engine._audited`); `check_on_start` moves lines FI did not write
  aside when a quest starts again (`Engine._check_records_changed_outside`), `evidence_gaps` feeds `core/evidence.py`.
- `core/numeric_oracle.py`, `core/stat_claims.py`, `core/number_provenance.py` — the three paper-vs-results audits
  (`needs/{numeric,statistics,provenance}_audit.json`) that `internally_reconciled` requires all three of.
- `core/numeric_warnings.py` — solver/runtime warnings from a run's own stderr that a paper must not be written over.
- `core/plausibility.py` — the design's declared `result_assertions` (legal ranges for a result) checked in code, including caps the script puts on a result (`clamp_constants`; `direct_caps`: a cap written where a value is reported -- proven at its whole path in the printed result, or a warning (`engine._assertion_warnings`); `caps_of_other_values`: caps proven to be another value's, which do not count for a value on its bound).
- `core/goal_coverage.py` — does the experiment use the numbers the topic itself asked for?

## Config, interview, providers

- `core/config.py` — the typed `Config` tree, `rigor_profile: research` (`_RESEARCH_PROFILE`,
  `REQUIRED_REVIEW_ROLES`, `_apply_rigor_profile`), unknown-key auditing.
- `core/attempt_records.py` — what a quest tried and under which conditions: `.fi/attempts.jsonl` (each run's outcome from `OUTCOMES` and `script_hashes`, each failed-check stop, the quest's four fields from `quest_status`; every line with `context_fingerprint` of a `CONTEXT_KINDS` kind, schema `SCHEMA`, a `record_id`) `.fi/branch_ledger.jsonl` (ideas, design revisions, repairs) and `.fi/model_calls.jsonl` (one line per model call attempt: `model_call_row` / `append_model_call`, written by `Engine._recorded_call` (failed attempts from `provider.CALL_ATTEMPTS`) and `provider.append_cost_row` for the generators; `model_call_gaps` compares it with the engine's per-step `model_call_counts` at the seal and the quest's end; after the seal calls go to `model_calls.after_seal.jsonl`); written by `Engine._record` / `_record_stop` / `_attempt_context`, read by nothing that routes yet.
- `core/attempt_memory.py` — shadow recommendations from past failed attempts: `Index` (failed attempts under the output root), `recommend` (BLOCK / VERIFY / INFO / IGNORE for a decision's context), `record` (`.fi/shadow_recommendations.jsonl`, written by `Engine._shadow` at plan / implement / execute / repair), `score` / `report_lines` (`fi tools shadow-report`); per-decision `KEYS`, `failure_signature` (also on every run, stop and crashed-quest record), one locked `index_for` per output root, lineage by `stamp` / `take` / `settle_without_record` (`parent_shadow_ids`; every plan of a quest is kept until it ends); the index is process-shared derived state alongside `ProxySupervisor` (read-only, locked, newest 500 quests, rebuildable); the worker is a daemon thread; `Engine._shadow` / `_shadow_close` (deadline, discard when late, closed before the seal); `engine.attempt_memory: shadow | off`; recorded only, read by nothing that decides.
- `core/phased.py` — explore, then confirm (`engine.phased`, off by default; on under `rigor_profile: research` where the config is silent, `core/config.py:_RESEARCH_DEFAULTS`, which a config may turn off, unlike `_RESEARCH_PROFILE`): `.fi/phased.json` (`load`, `stage`, `status`), `prepare` (at every start, `Engine._phased_prepare`: holds back about 30% of the rows of each CSV/TSV in `inputs/data/`, chosen by a hash of each row's text (once exploration ran, only rows held back before), into `store_dir` = `<output_dir>/_held_back/<quest_id>/{held_back,original}/` outside the quest folder (`_move_out_of_quest` moves the older `.fi/phased/` layout), or says why the confirm run gets new seeds only; finishes a start cut short; in the confirm stage gives the run the held-back part again, `mark_compromised` when it cannot), `restore_inputs` (the whole files back when `Engine.run` ends, paused or not, also with the setting off; names a file it left alone), `turned_off` (`Engine._phased_turned_off`: a start with the setting off on a quest with a record), `note_job_pending` (`_wait_for_job`: a resume checking on the confirm run's own job is not a second run), `note_confirm_result` (`_node_execute`: the gate records only the result the confirm run produced), `Engine._phased_seed_gap` + `mark_unconfirmable` (no data held back and a result new seeds cannot change: no random source (`result_json_no_random_source`; seeds that merely agreed do not count), runs that ignored the seed, `run_cell`, or the experiment and the modules it imports never take `FI_REPLICATE_SEED` into a generator and draw only from fixed seeds (`_own_modules`; FI's `run.py` does not count): status `not_confirmable`; `record_confirm` also when a new-seeds confirm run gives exactly exploration's result), `Engine._phased_fresh_background_job` (sets exploration's `job/state.json` aside for a background job's confirm run; `_JOB_SEED_PROTOCOL`, only when on, tells the driver to pass `FI_REPLICATE_SEED` on to the job), `_write` (temporary copies in the store, never in `inputs/data/`; `_clear_stray_tmp`), `confirm_run_started` (the gate records a confirm result only after the confirm run started; `status` is `confirm_reused` once the confirm data or seeds were run on twice), `mark_compromised` (a file could not be split or put back, a replayed result, seeds the run cannot take: status `compromised`, nothing confirmed; writes a record when there is none), `seed_env` (the replicate seeds in `_node_execute`: remembered in exploration, moved to the confirm base after), `enter_confirm` / `record_confirm` (called by `Engine._phased_route` from `_route_after_evidence_gate`: the evidence gate's `confirm` edge to `execute`, added to the graph only when on; `_freeze_protocol_if_due(at_confirm=True)` freezes there, not before the first run), `evidence_settings` (`evidence.assess` `settings["phased"]`, `_PHASED_GAPS`), `write_note` / `mark_paper` (the writer's note and the paper's note, no numbers in it), `enter_confirm` also records `design_revisions_at_confirm` (the design history's length then; `core/disclosure.py` reads it); `mark_not_applicable` (status `not_applicable`: a quest with no experiment of its own, pinned in the config at the start (`Engine._phased_prepare`) or resolved by the clarify step (`Engine._node_clarify_then_phased`, off the event loop); the held-back rows go back, one run.log sentence, no gap, no paper note, empty writer note; `turned_off` says nothing for it); `kept_off` / `began_before_default` / `ran_without` / `without` (a research quest that ran with the two stages never on (graph steps in the trace, no `STARTED_EVENT` = `phased_started`, which every start with them on writes) and whose `config.yaml` does not set it runs without them, on a copy of the config: `Engine._phased_keep_off_if_began_before` before the approved settings are compared, and `interview_update.approve_settings` before it records them; `plan_settings.config_sets`); `plan_lines` (`Engine._plan_confirm_lines` → `plan.render(confirm=...)`: plan.md's *How the result will be confirmed*, with the confirm run's cost, or why it cannot be confirmed; `off_sentence` / `kept_off_sentence` also in run.log); `interview.smart_default_phased` (on for research and a decision, off when exploring; `InterviewAnswers.phased` is `None` when unanswered, and `False` writes `phased: false` only under research); `_route_after_cross_check` does not redesign in the confirm stage; reads no attempt record or shadow recommendation. Data quests and split semantics: `prepare(data_quest=..., key=..., mode=...)` also holds back the person's table in `data/` (`_data_files`; a file's `folder` in the record, kept under `store_dir/data/`), sets rows aside one by one until `decide_split` (called by `Engine._phased_before_first_run` at the top of `_node_execute`, and by `data_quest_gate` from `Engine._phased_before_data_read` at the top of `_node_data_load`) applies the rule of `core/phased_data.py` (`split` and `split_decided` in the record, a per-file `rule` and `manifest` with its zero-overlap check; overlap → `mark_compromised`); a research quest that cannot tell which rows belong together gets a question (`Engine._phased_ask_split`, pause `data_split`, answered on plan.md's `Rows that belong together:` line, `phased_data.plan_answer`); `split_preview` writes that line into plan.md (`plan_lines(split_note=...)`). `data_quest_gate` counts each reading (exploration `explore_runs`, after it `confirm_executions`) and marks an unsplittable data quest `not_applicable` (`not_applicable_sentence(data_quest=True)`, `GATHERED_ONLY`); `_phased_route` sends a data quest's confirm to `data_load` (edge `confirm_data`), `enter_confirm(design_sha256=..., isolation=...)` / `record_confirm(design_sha256=...)` refuse a changed design and record `confirm_differs` (`phased_data.compare`), which `evidence_settings` passes on (`phased_differs`, `phased_isolation`, `phased_split_gap`; gaps in `evidence.assess`). Kept files go through `_read_kept` / `_write_kept` (encrypted when the engine passes `Engine._phased_key()`; `_rekey` rewrites an earlier run's with the current key from the whole file in place; `_as_supplied` returning None means a killed run's rows are lost → compromised).
- `core/phased_data.py` — which rows of a table are held back for a confirm run (`decide` → `Decision`: the plan's `protocol.split` (`declared`), plan.md's answer line (`plan_answer`, `QUESTION_LINE`), or the column names (one time column, or one unit column); rules row / group / time (latest periods, `embargo`) / spatial, `stratify_by` or a plan variable (`grouping_names`), `MIN_UNITS`, `MIN_HELD_UNITS`; `overlap` (the zero-overlap check on written parts); `how` / `how_in_paper` wording), and a data quest's confirm numbers beside exploration's (`numbers`, `compare`, `compare_lines`).
- `core/phased_isolation.py` — whether held-back rows were out of exploration's reach: `seal` / `unseal` / `new_key` (Fernet, key in the engine's memory only, `MAGIC`), `scan` (a static read of `code/` for paths out of the quest folder), `docker_mounts_clear` (`DockerExecutor._volumes` holds nothing of the store), statuses `DOCKER`, `ENCRYPTED` (`encrypted+scanned`), `UNVERIFIED`; `Engine._phased_isolation` decides it at the confirm stage, `phased.isolation` reads it.
- `core/confirmations.py` — the append-only record of every confirmation (`.fi/confirmations.jsonl`): `append` (one
  verdict line per confirm run of a candidate, chained by `prev_sha256`), `read`, `for_candidate`, `worst` (a failure
  counts), `candidates`, `unsealed` (lines the trace does not name yet: `Engine._phased_seal_confirmations`, called
  from `_phased_log`, writes `confirmation_recorded` for each) and `problems` (a broken chain or a line the trace named
  that is gone). Written by `phased.record_confirm` / `phased.new_candidate`; read by `phased.record_gap` /
  `evidence_settings`, `disclosure.confirmations` (the methods paragraph's count of versions) and `todo.confirm_item`.
  The candidate logic lives in `core/phased.py`: `fingerprint` (code / protocol / environment hashes, without the
  redraw helpers in `_NOT_THE_STUDY`), `candidate`, `new_candidate` (called by `Engine._phased_before_first_run` and
  `data_quest_gate(design_sha256=...)`: a version changed after its confirm run began is candidate N+1, back to
  exploring; `not_confirmable` with `HELD_BACK_USED` when rows were held back), `ONE_SHOT`. In the confirm stage
  `optimise.run_search(frozen=True)` (from `OptimisationRunner(frozen=...)`) keeps exploration's search (`study_key`)
  and never searches again.
- `core/plan_settings.py` — the settings a quest was approved with (`.fi/approved_plan.json`), checked at every start
  (`Engine._stop_for_changed_settings`); `interview_update.approve_settings` records a change `--update` approves.
  The model settings (`MODEL_SETTINGS`: provider, model, per-step models; not the ensemble) are recorded but never
  stop a quest: `model_changes` / `accept_models` / `model_change_lines`, taken by `Engine._take_model_change` (a
  `model_changed` audit event, `before_any_step` when nothing had run yet, run.log, the `[FI] model:` console line the
  VS Code chat shows) and told to the paper by `model_disclosure` (`Engine._write_whole_paper`).
  `Engine._review_models_stop` re-checks a research panel after a model change even when it has reviewed. `--update`
  leaves a model change for that start to record.
  `launch._refresh_quest_config` keeps the quest's own `config.yaml` equal to the config it was resumed with.
- `core/interview_update.run_update_flow` — `--update`: asks the editable setup questions (`ask`), approves the
  settings, resumes. VS Code's `@fi /update` (`extension.ts:runUpdate`, run in the chat through `runLaunchInChat`
  after the person chose to approve config.yaml as it is) sets `FI_UPDATE_APPROVE_AS_IS=1`, which `launch._run_update`
  reads as `ask=False`: nothing is asked and each line starts `[FI] update:`.
- `vscode-frontier-insight/src/extension.ts` `runLaunchInChat` — every chat command that runs `launch.py`
  (`/start`, `/resume`, `/update`, `/generate`, `/ingest`, `/install-tectonic`) runs it as a child with its output in
  the chat and a per-command bridge; `reportQuestEnd` shows how a quest run ended. No chat command opens a terminal;
  the one terminal is the Axon server's "Start in terminal" button.
- `core/engine.py` `_node_clarify` — the first discussion of a quest (what you want to see, the title, scope). `pauses.clarify`
  unset means ask when a callback or staged answer exists, else auto (`Engine._clarify_answerable`); the title chosen there
  is `state["title"]` (`title_confirmed`), which the writer is told to use (`_spell_out_title_options` puts the
  suggested titles in the question's words, `_clean_title` tidies the answer). `Engine.run` resumes the clarify pause from the
  callback, then a staged `.fi/clarify_answer.json` (`_json_file_is_answer` keeps one written as the wait ran out).
- `core/engine.py` `_review_decision` — whether a human-review answer (callback or staged `human_review_answer.json`)
  carries accept / reject / refine; `Engine.run` resumes only with one, and otherwise (or on a closed VS Code prompt)
  pause-exits with the snapshot and `NEXT_STEP.md` kept.
- `core/interview.py` — the single question set shared by the CLI, the web form and VS Code; a question asked only for
  some earlier answer carries `ask_if`, checked by `question_applies` (the web page and VS Code mirror it; today only
  `second_reviewer_model`, for research or a decision); `core/interview_update.py` is mid-quest re-entry. The
  clarify-mode default `CLARIFY_WHEN_PRESENT` is never written, so `pauses.clarify` stays unset unless a person chose.
  What a first-time person sees is defined here once: `FIRST_STEPS` (the three first steps: the research question,
  what the result is for, the model), `REVIEW_CARDS` (the four review cards, each with its shown and advanced rows;
  every question on exactly one), `PLAIN_VALUES` / `plain_value` (no internal names on a card), `COST_NOTES` /
  `cost_note`, `CHECKS_SENTENCES`, `BYLINE_FIELDS` / `byline_text` (the folded "Paper byline (optional)" row; the
  byline is tier 2). The schema exports them (`first_steps`, `review_cards`, ...): the web page reads them there,
  VS Code from `interview-core.ts` `REVIEW_SCREEN` (a generated copy; `tests/test_interview_review_cards.py` keeps it
  equal), the CLI through `launch._build_review_rows` / `_print_review`.
- `core/provider.py` — every transport (`LLMClient`), `ProxySupervisor`, `missing_api_key`, model pricing; `LAST_CALL` (who answered the current task's last call, and why its answer ended); `ModelAnswerTruncated` / `ModelAnswerFiltered` / `outcome_of` (an answer cut off at its limit or withheld is never returned as whole; `Engine._pause_for_model_output` turns either into a `model_output` pause); `node_output_limit` (`provider.node_max_tokens` per step); `quest_run_log` (a failed call made outside the engine still reaches the quest's run.log); `_http_streams` / `_post_streamed` (Moonshot calls are streamed) and `_post_ollama_streamed` (Ollama's native `think` call, NDJSON, assembled for `_ollama_as_openai`; both streamed calls share `_STREAM_TOTAL_FACTOR` = 4 times the step's HTTP timeout as the whole-call budget, the step timeout being the silence limit; `LLMClient` marks a call whose every try timed out with `fi_timeouts`). The HTTP retry policy waits out a server outage or rate limit (`_http_outage_status`, `_http_retry_wait` / `_http_retry_stop` / `_http_retry_sleep`: six attempts on `_HTTP_OUTAGE_WAITS_S`, a sane `Retry-After` up to `_RETRY_AFTER_MAX_S`, at most `_HTTP_OUTAGE_MAX_WAIT_S` per call, the `FI_MAX_CONCURRENT_LLM_CALLS` slot given back while waiting); a used-up quota is `_is_exhausted_quota`; `FallbackLLMClient` sets `short_retry` (`_http_short_retry`) while a later provider's circuit is closed. The claude CLI's own model switch (`_claude_stream_facts`: the answer's model, a `model_refusal_fallback` / `model_fallback` / `model_consent_fallback` event, a refused last turn) is said in run.log every time; after a `model_refusal_fallback` the asked-for model is asked again once with `CLAUDE_NO_REFUSAL_FALLBACK_ENV` (`LLMClient._ask_the_asked_model_again`; `switched_from` in the call's record); a refusal with no other model answering raises `ModelAnswerFiltered` (`finish_reason: refusal`), not retried.
- `core/provider_models_discover.py` — runtime model-list discovery for the provider picker.
- `core/provider_readiness.py` — how far a provider is set up, as a ladder of plain states (`Readiness`: not
  installed → installed → signed in / unknown → reachable → model available; no key / key present / key rejected;
  not running / model missing), each with one sentence and a fix that every surface prints as is (`one_line`).
  `check_local` (PATH and environment only: the quick `--doctor`, the CLI interview's provider list,
  `core/interview.py::available_providers`), `check` (also a CLI's own sign-in status command through
  `_run_command`, a `core/proc_tree.py` tree with a time limit, and a local Ollama's `/api/tags`: the web Settings
  page and the interview's picker via `/api/providers/availability`), `preflight` (also the provider's free
  `GET /models`: before a launch from the CLI interview or the web form, `fi demo`, `--doctor --deep`).
- `core/ensemble.py` — the multi-model fan-out-and-merge primitive a node opts into.

## Knowledge / literature

- `core/knowledge.py` — `Knowledge`, the three-layer retrieval (pinned papers → Axon → external router); `is_open_access(metadata)` is the one rule for "free to download" and `_free_locations(metadata)` the only addresses requested for a free paper (the source's own `free_url`, or an address on a free host; never a DOI or publisher page); `_get_checked` follows redirects by hand and `_redirect_allowed` re-checks every hop (the headless render does the same per hop in `_playwright_fetch_html`, through the route handler `_make_navigation_guard`, whose `_route_call`, on the "page already closed" error only, marks the page's routes "ignore errors" and re-raises so Playwright drops it without printing a traceback); `_looks_scholarly` keeps journal-article web hits out unless free (gates `_fetch_full_text` / `_fetch_web_page_text`; `engine._is_open_access` delegates to it); `add_quest_artifacts` writes accepted or preliminary kinds by `metadata['standing']` (decided in `Engine._write_back_knowledge`; read back by `engine._is_preliminary_memory` / `_preliminary_reminders`); the copy under the other standing (`STANDING_KINDS`) is removed first (`_retire`, `_delete_in_process`), and `retire_stale_standing` / `retire_stale_ref_spines` back `fi tools tidy-knowledge`; a removal that fails stops the write (`last_writeback_problem`, which the engine puts in run.log and `.fi/knowledge_problem.json` for the card); cited papers' entries (`REF_SPINE`) keep every accepted consumer (`_paper_entry`, `_drop_consumer`, under `_paper_entries_lock`).
- `core/axon_http.py`, `core/axon_sidecar.py`, `core/axon_endpoint.py` — talking to the shared Axon service: HTTP
  client (`AxonHTTPBrain`: ingest, delete_documents, list_sources, search_raw), sidecar lifecycle (start/reuse/stale-lock clearing), endpoint discovery.
- `core/passages.py` — relevance-ranked excerpt selection over fetched full text; `embed_model_cached` says whether
  the embedding model is downloaded, from the model caches on disk only (the quick `--doctor`), `_embed_model`
  loads it.
- `core/retractions.py` — the retraction check: `check_literature` looks each DOI up in Crossref (a pinned / dropped paper with no DOI is first matched by exact title via `find_doi_by_title`; `updated-by`
  notices, Retraction Watch included) after the literature node's dedup and sets `metadata["retraction"]`
  (`retracted` / `not_retracted` / `not_checked` / `no_doi`); `apply_to_claims` makes a claim grounded in a retracted
  source, and any sentence citing one, unsupported after the claim check; `retracted_dois` keeps retracted papers off
  `WANTED_PAPERS.md`; `summary_line` is the run.log line; `retracted_in_record` feeds the to-do card. Tests never reach
  Crossref: `tests/conftest.py` answers "not checked" outside `tests/test_retractions.py`. The `[retracted]` mark itself is added by `core/engine.py::_format_lit_header` and `_claim_source_block`.
- `core/source_text.py` — retrieved text is data, not instructions. `fence` puts every block of retrieved text a
  prompt carries (the prior-work blocks `engine._format_lit` / `_format_lit_from_state`, analyze's title list, the
  claim check's sources, the literature screen, relevance guard and requery, plan_criteria's search, figure captions
  for figures_pick and web_figures, web_plots' collected pages, `summarizer._render_content_blocks` for data_load /
  analyze / `fi summarize`, `engine._preliminary_reminders`) between two markers carrying a key hashed from the
  block's text (`markers`), after one sentence saying what it is; `neutralise` also strips a copy of the markers'
  words from the source text and changes nothing else. `scan` flags text addressed to a model or hidden from a reader
  (tag characters, invisible characters inside words, a PDF's `hidden_text`); `flag_and_record` marks each source's
  metadata once per text (`source_text_scanned` / `source_text_flags`, read by `mark` for the prompt's
  `[flagged: ...]` tag), writes one `check_result` event (`check="source_text"`) and one run.log line per call,
  naming the sources not already named with the same findings (`reported`; a literature pass names every flagged source it scanned),
  and never raises or removes anything. `Engine._flag_sources` runs it in a thread for
  the literature node (every pass; one that scanned and found nothing is recorded as `ok`), `_node_pause_after_literature`, ideate,
  cross_check and `_propose_criteria`. figures_read (image + caption pairs) is told in `agents/figures_read.md`
  instead of fenced; the foundational-works notes of write and review list titles outside a fence.
- `core/trial_runner.py` — the trial contract: FI runs `run_trial` / `run_cell` of `simulate.py` for every setting,
  one process per setting, writes `raw/ledger.jsonl` and `raw/trials.json` itself (`TrialsRunner` in `_node_execute`,
  `measure_oracles` in `_oracle_gate`: it calls the simulation on each oracle's case via `run_case`, and `run_oracle` only for an oracle without a case); `recorded_values_by_cell` / `given_values_not_run` hold each reported value to the trials of its own settings; under research `recorded_rows_by_cell` / `given_rows_problems` (via `Engine._given_row_findings`) check a mean over a subset trial by trial; the harness hands every trial the protocol's thresholds as `FI_THRESHOLDS` (`run_oracle` too; the proportion's 0/1 under its id and the averaged quantity under the mean's own id, `RETURN_MEMBERSHIP`); the older self-looping contract stays in `core/split_run.py` as `self_reported`. On a cluster (`execution.background_jobs`) `prepare_cluster` / `collect_cluster` run the settings as a job array submitted by `code/submit.py` (`TrialsRunner(submit=...)`); `code_changed_while_queued` compares the code at submission with the code at collection.
- `core/profile.py` — the person's paper byline, kept in `~/.frontier-insight/profile.json` for the CLI
  (`launch._run_new`), the web page (`/api/profile`, saved on submit) and VS Code (`interview.ts` `loadProfile` /
  `saveProfile`); not asked up front: a review-screen row, and `launch._ask_byline_once` asks it once before the
  first paper (a terminal only, bounded wait). A blank byline is not saved while none was ever given.
- `core/fi_home.py` — `fi_home()`: the per-person state folder `~/.frontier-insight` (`FI_HOME` overrides; the tests set
  it in `tests/conftest.py::_isolate_fi_home`). New per-person files go through it; `vscode-frontier-insight/src/fi-home.ts`
  (`fiHome`) is the same rule. (Older files — profile, skills, caches, pip lock — still build the path themselves.)
- `core/quest_index.py` — every quest FI has run on this computer, `fi_home()/quests.json` (id → quest_root, config,
  working_folder, title, created_at, last_seen). `register` (from `launch._record_quest`, called by `run_one` at start /
  resume and at the end), `set_title` (from `quest_title.rename`), `prune` / `entries` / `listing` (`fi tools quests
  [--prune]`, `launch._list_quests`; `web/server.py::_index_listing` for `GET /api/quest-index`). Every write is
  read-modify-write under a `filelock.FileLock` with temp file + `os.replace` (retried on Windows sharing violations);
  a file whose content is not a quest list is kept aside as `quests.json.unreadable-<time>`; a read error (`OSError`)
  writes nothing. `find(text, local_dirs, allow_folder=, tidy=)` is the one lookup: folder path → `<local>/<id>` →
  index full id → the one id among local and index ids that starts/ends with the text (`MIN_SHORT` 4) →
  `QuestNotFound` with close matches / `AmbiguousQuest` with candidates. `_missing` entries are left out of lists;
  `_gone` ones (missing while their outputs folder is there) are dropped on lookup when `tidy`; `prune` drops
  `_prunable` ones (missing while their drive is there). Only the entries a text names have their folder checked. Used by
  `launch._config_from_quest` (`--resume`/`--watch`/`--rerun` without `--config`), `main_async` (with `--config`;
  sets `cfg.output.output_dir` to the quest's own outputs folder and, via `_started_in`, `os.chdir`s to the folder it
  was started in — one quest per process there; every web launch path gets this through the CLI),
  `interview_update.run_update_flow` (output_dir), `launch._quest_dir` (`--trace`, `--why`, `--rename`) and
  `_locate_quest` (`--update`, `--critique`), and `web/server.py::_resolve_quest_root` (`allow_folder=False,
  tidy=False`; every quest API; a 409 lists an ambiguous id; DELETE refuses a quest outside its outputs folder; the
  resume endpoint also passes `--output <its folder>` and `QuestLauncher.launch_command(cwd=)`). `short_id` (6 hex after the last dash) is shown by the
  start/resume log line, `fi tools quests`, the dashboard and quest page. VS Code reads the file only:
  `vscode-frontier-insight/src/quest-index.ts` (`loadIndex`, `matchIds` = `matching`, `findInIndex`, `shortId`), used by
  `extension.ts::runResume` (`/resume`, `/plan`, `/watch`; `RunWhere` runs a quest from another folder with its config,
  `--output` and cwd) and `quest-map.ts` (`pickQuest`, `fullQuestId`); the other `@fi` commands pass the id to `launch.py`.
- `core/thinking_capture.py` — the model's own account of its reasoning: a task-local holder (`open_holder` / `note_thinking` / `add_thinking`) that each transport fills (`provider._reasoning_of` and the streamed `reasoning_content` deltas for HTTP, `provider._stream_thinking` for the Claude CLI, `vscode_bridge.LAST_BRIDGE_THINKING` fed by `bridge.ts`'s `lm_done.thinking`); `Engine._recorded_call` opens it and `Engine._save_thinking` writes `.fi/thinking.jsonl` (redacted, one line capped at `THINKING_LINE_CHARS` and the file at `THINKING_FILE_BYTES`, `call_id` joined to `.fi/model_calls.jsonl`, `provider`/`model` of the answering provider, not sealed, `output.save_thinking`); `LLMClient._chat_impl` and each `FallbackLLMClient` rung reset the holder so one attempt's reasoning is never filed under another's; `lm-messages.ts::lmDoneMessage` (both bridges) cuts the reasoning so the `lm_done` line stays under 48 KiB. Asking for it: `thinking_capture.wanted()` (a holder open with `want=output.save_thinking`) becomes `lm_request.ask_thinking`, and `lm-messages.ts::ThinkingRequests` (both bridges) sends Copilot's `modelOptions._enableThinking`, asks again once without it on a refusal (`thinking_declined` on `lm_done` → `note_declined` → `Engine._say_once_about_thinking`, which also says once per step that no reasoning came back) and turns a stream that failed before its first part into `THINKING_DECLINED_MARKER`, which `provider._TRANSIENT_BRIDGE_MARKERS` retries; `lm-messages.ts::thinkingText` and `thinking_capture.as_text` take a `string[]` value. Codex: `_CliSpec.reasoning_extractor` / `provider._extract_codex_reasoning` (the `reasoning` items of `codex exec --json`). What a CLI must be told to return its reasoning is `_CliSpec.thinking_args` (claude: `--settings` with `showThinkingSummaries`; codex: `model_reasoning_summary`), added by `_run_cli` only while `thinking_capture.wanted()`. Ollama: with `wanted()` on, `LLMClient._chat_impl`'s `send` calls Ollama's native `/api/chat` with `think` (`provider._ollama_native` / `_ollama_native_body` / `_ollama_as_openai`; a model that refuses is remembered per client and uses `/v1` as before). A reasoning that loops: `thinking_capture.repeating_cycle` (the same block of up to 60 lines ten times in a row, 400 characters at least, or one stretch of up to 800 characters; `LOOP_*` thresholds, checked every `LOOP_CHECK_EVERY` characters) makes `_post_ollama_streamed` close the stream and raise `ThinkingLoop`; `send` catches it (and a `length` finish with reasoning and no answer), calls `note_loop`, and asks once more over `/v1` (`think_off`), so nothing is filed in `.fi/thinking.jsonl`. With `think` on `send` omits `options.temperature` (`_ollama_native_body(keep_temperature=...)`) unless `provider.fixed_temperature` / a `temperature` in `extra_body`/`extra` is set; `LLMClient._ollama_sampling_note` reads the model's `parameters` from `/api/show` for the note. `note_timing` / `note_sampling` / `note_loop` put the call's timing line, sampling note and loop fact on the holder, and `Engine._say_call_notes` writes them to run.log (the timing line each call, sampling once per model, the loop once per step). `BRIDGE_PROTOCOL` (`lm-messages.ts`, on every `lm_done`; `vscode_bridge.REQUIRED_BRIDGE_PROTOCOL` must equal it) tells an extension older than this FI apart (`extension_older` on the holder, said once by `Engine._say_extension_older`), `ThinkingCollector.emptyParts` → `thinking_parts_empty` → `note_empty_parts` makes `Engine._say_once_about_thinking` say once per model that a VS Code model returned no reasoning (and name `codex_cli`), and `lm-messages.ts::stallMessage` names the silent model. `thinking_capture.CANNOT_RETURN` names the connections that cannot return reasoning, which `Engine._say_once_about_thinking` says once instead of the per-step line. `_chat_provenance` also carries `call_id` / `requested_model` / `finish_reason` into every `model_claim` and, through `_last_chat`, into an attempt record's `models_used`, whose `call_id` differs per call.
- `core/quest_title.py` — a quest's title after it ran: `current_title` (paper H1 / front matter, else `config.yaml`), `rename` (paper.md copies, `config.yaml` one-line edit, `frontier_insight_summary.json`, the latest checkpoint's `title` / `title_confirmed` edited in place, all files written or none, the quest index's title (`quest_index.set_title`), then a `title_changed` audit event carrying the paper's hash before/after, which `evidence._trace_completeness_gaps` follows from the seal and `audit_log.after_title_changes` skips for `--trace --follow`; `RenameRefused` / `QuestRunning`), `looks_running`. Called by `launch._rename_quest` (`--rename`, `fi tools rename`), `POST /api/quests/{id}/title` (quest page **Rename**) and `trace.ts::runRename` (`@fi /rename`); `launch._finish_outputs` writes `title` into the summary.
- `core/why.py` — `--why` / web Why? / `@fi /why`: why a quest stopped, why the review asked for a revision, why the evidence is at its level, why a step decided what it did (with `_thinking_line`: whether `.fi/thinking.jsonl` holds that run of the step's reasoning and how many characters), and `reasons` (web **Model's reasons**): every step's stated reasons, from the audit trace and records (no model call). The reasons come from `Engine._audit_claims` and `Engine._audit_stated_reasons` (ideate, select_skills, cross_check, execute_reflect, clarify, the review's `why` and moderator), `Engine._claim` (only under a call made during that run of the step: `_chat_at_node_start`), and the write step's `REFINE_WHY:` line (`engine._take_refine_why`). The model a VS Code quest uses: `launch._apply_vscode_chat_model` (`--vscode-chat-model`, passed by `extension.ts` through `lm-messages.ts::chatModelArgs`), `Engine._say_model_change` / `_note_served_model` (run.log lines and `model_changed` trace events). `launch._follow_trace` is `--trace --follow`; `provider.set_model_call_archive` / `append_cost_row(messages=, response=)` keep every model call in `.fi/io/` when `output.save_model_calls` is on.
- `core/engine.py` refine routes — `_take_refine_points` reads the writer's `NEEDS_EXPERIMENT` / `NEEDS_DATA` / `NEEDS_LAYOUT` lines; `_route_after_write` sends them to `design`, to `implement` in extend mode (`_extend_directive`: the existing scripts are the base) or to `_node_replot_layout` (`agents/replot_layout.md`: redraw from the saved files under `data/results/` and `data/`, no experiment run); `_route_after_human_feedback` sends a refine with notes to `write` and one without (a re-open) to `design`; `_node_write` records `refine_scope` (`paper` / `data` / `layout` / `experiment`) for the route's trace facts; state `refine_extend` / `refine_layout` / `extend_missed` / `layout_missed` (asked but not done, also when collection is off or a redraw has no figures, fails, or prints `NOT_DRAWN:`: `_route_after_replot` sends the paper back to the writer once, which says so and clears both). An extension that ran records `extend_check` (what was asked, the result names before it); every later write checks the paper reports what it added (`Engine._report_what_the_extension_added`, `number_provenance.unreported_results`, no model call), asks the writer once more when it does not, and leaves `extend_unreported` for the review pause's steps. An extension is recorded in `PROTOCOL_CHECK.json` as `extended_by_person` (`_goes_beyond_protocol_by_request`), which the run-manifest check and `_goal_coverage_notes` read (`_protocol_drift_not_asked_for`: only the differences it recorded are excused, any other drift is still reported); `auto_collect_data` searches for the ask; `_node_replot_layout` backs up figures and data in `.fi/layout_backup` and `_restore_layout_backup` undoes a redraw cut short; it shows the model what each saved file holds (`core/data_shape.py`), writes a failed (or no-change) script once more with its error, and when both tries fail sets `layout_not_redrawn`, which `_node_human_feedback` puts first on the review card and clears once answered.
- `core/acceptance.py` — who accepted the result and the one question a person answers first (`QUESTION`, `CHOICES`, `parse_answer`): `shown(record)` is what every interface shows before an accept (the reached level's blind spots, the most important gaps, capped and de-duplicated, never the decision's own gaps), `problem` refuses an accept without an answer, with "no", or a "partly" without its `note` (`engine._review_answer_problem` / `_review_decision`, and `web/server.py` `post_human_review` calls it too), `stamp` / `receipt` / `automatic` are what the run loop resumes the review pause with (a person's receipt from the callback or the answer file — `who`, `at`, `via`, `answer`, `note`, `evidence_sha256` and `limits_shown` from the snapshot's `before_accept` — or `auto_accept_on_pass`; an answer's own `by` is never taken), `_node_human_feedback` puts `before_accept` (with `evidence_sha256`) in the snapshot and on the card (`Engine._write_evidence(..., write=False)`), records state `acceptance` (adding the accepted paper's hash) and the `result_accepted` trace event; `accepted_by` and `review_gap` are read by `core/evidence.py::assess` (no person, or another paper: the gap `NO_PERSON_GAP`; a person's answer other than "yes": `NOT_CHECKED_GAP` or `partly_gap(note)`; each below `publication_ready`, `accepted_by` / `acceptance` in the record, `mark` in `summary_line`). Interfaces: `launch.py` `--accept [ANSWER [NOTE]]` (`_accept_answer`, `_ask_accept_note`, who from `--approve-as` or `_login_name`) and the `--interactive` prompt, `web/server.py` `post_human_review` + `quest.html` `showBeforeAccept`, `vscode-frontier-insight/src/bridge.ts` `handleHumanReviewRequest` + `core/vscode_bridge.py`.
- `core/todo.py` — the to-do card every stop writes (`NEXT_STEP.md` + `.fi/todo.json`): per-kind decision, recommendation and alternatives (`advice`; under a rigor profile, never a setting it refuses: `refused_settings`, and what to do instead by pause kind and freeze: `research_instead`), the other things waiting (`waiting`), printed by `launch.py`, read by the web `/next-step` endpoint and VS Code. A stop with a structured "why it stopped" card (`Item.card`) is rendered from it by `card_lines` (Markdown for NEXT_STEP.md, plain text for the terminal).
- `core/crash_kind.py` — sorts a failure that stops a quest into a passing problem, a setup problem, FI's own programming error or an unexpected one (`classify`, reusing `core/provider.py`'s retry and used-up-quota rules), with what happened and the one thing to do in plain words; a step whose every provider try timed out (`fi_timeouts`, `_always_too_slow`) is a setup problem naming the setting to raise, not a passing one; `RETRY_WAITS_S` / `RETRY_ONLY_UNDER_S` are how long `Engine._try_step_again` waits before running a graph step again after a passing problem, and the longest a step may have run to be run again; `write` / `read` / `clear` keep `.fi/failure.json`, which `Engine._write_quest_failed_diagnostic` (the top of `quest_failed.md`), `core/todo.py::waiting`, `web/server.py::_read_quest_failed_md`, `launch.py::_print_quest_failure` and the VS Code chat (`readFailure`) read.
- `core/oracle_card.py` — what FI found about the known-answer checks that did not pass, built by `Engine._oracle_gate` from what it already has (`build`; no model call): `leaning` (which side FI's own look points to: check / script / unclear / plan; a recheck by the plan's own model is never counted) and `why` (one plain sentence), recorded with the gate's automatic decision as `explained` in `needs/ORACLE_CHECK.json` and said in the console (the gate no longer stops to ask a person), and the card's details: per check that did not pass its plain name, kind, expected value and source, measured value and who measured it, tolerance and `gap`, case and measure, where the script computes the number (`locate`, an excerpt), the error when nothing was measured (`last_error` / `error_at`, `trial_runner.last_frame`); likely causes (a repair's proposal, `oracle_forms.mismatch`, `oracle_check.duplicate_names`), what FI tried (the record's `repair` / `patch_summary`), and 2–3 actions with the exact command per interface. One payload in `.fi/pause.json` (`card`, with `problems` / `proposed_changes`) and `.fi/todo.json`; the web quest page (`renderCardActions`) and VS Code (`src/stop-card.ts`, `offerCardButtons`) add buttons from its `actions`. Says what was judged; never changes a verdict. FI's own look at failing checks (below) adds its findings as causes and tried lines.
- `core/oracle_triage.py` — FI's own look at a failing known-answer check before the first repair is spent on it, called from `Engine._oracle_gate` via `_look_at_failing_checks` / `_recompute_expected` (prompt `agents/oracle_recompute.md`, node `oracle_review.recompute`): the arithmetic a derivation writes out, worked out by FI (`calculate`, `arithmetic_slip`, `arithmetic_entry`, `arithmetic_proposal`, on `oracle_forms.evaluate`; a slip sets the check aside from the repairs; its value is written into the plan before the freeze by `Engine._correct_expected_values` only when another model's blind recheck agrees (`Engine._may_correct`, `_second_source`) — never the measured value, never the tolerance; `.fi/oracle_corrections.json`); a recheck by the plan's own model, or whose own working does not add up, is recorded but never counted; the expected value recomputed blind to the measurement (`recompute_verdict`, `recompute_proposal`, cached in `.fi/oracle_triage.json` by `fingerprint`), a measured value that is half (twice, a quarter, four times) a value FI worked out itself (`multiple_of`, `multiple_entry`; only against FI's own correction of the check; `Engine._multiple_hints` puts it in the repair request), the case at a smaller step (`step_of`, `refined`, `step_verdict`: Richardson; a value that stays the same is `steady_elsewhere`), more seeds (`seeds_verdict`; `trial_runner.run_case(trial=)`), the same exception after a repair (`same_exception`), a timeout retry (`timeout_entry`), and plan.md's `cost_line`. Each returns a triage entry (`tried`, `cause`, `points_to`) the card renders. `Engine._restore_oracle_disputes` reads disputes back on a resume. `oracle_forms.unit_multiple` names a unit/representation factor; `protocol_check` skips the oracle's own code (`_oracle_only_names`).
- `core/receipts.py` — the receipt each required check (evidence gate, design audit, claim check) writes under
  `needs/receipts/`; `core/evidence.py` reads them for `publication_ready`; `engine._stop_once_for_check` is the
  research profile's one stop and retry.
- `core/pdf_text.py` — every PDF FI reads: all pages in reading order (PyMuPDF if installed, else pypdfium2), scanned
  pages by OCR (tesseract, else RapidOCR), a size cap that names its page. Fetched scans are OCR'd after the fetch
  (`knowledge._ocr_scanned`); `engine._literature_entry` / `_content_quality` / `_item_content` keep the whole text on
  disk (`data/literature/.full_text/`) and label what it is. `PdfText.hidden_text` lists text drawn in white, invisibly
  or below 2 pt (`_is_hidden`, `_hidden_pymupdf` / `_hidden_pdfium`; white or invisible text over a coloured fill or an
  image does not count); it reaches the source's `metadata["hidden_text"]` through `knowledge._hidden_pdf_text` for a
  fetched PDF, `engine._extract_pdf` for `inputs/papers/` and `knowledge._extract_pdf_text` for `local_papers`, and
  `core/source_text.py` flags it.
- `core/pdf_figures.py` — captioned figures cut out of a PDF by layout (drawings above the caption; scanned pages from
  `PdfText.ocr_lines`). `knowledge._pdf_figures` caches them, `knowledge._cut_figures` runs after a fetch,
  `engine._save_literature_figures` puts them in `data/literature/figures/`, and
  `engine._read_literature_figures` (in the `pause_after_literature` node; prompts `agents/figures_pick.md` and
  `agents/figures_read.md`, step `figures`) reads the relevant ones into each source's text.
- `core/arxiv_gate.py` — the one shared queue/backoff/cache for every arXiv connection.
- `core/source_failures.py` — per-quest retrieval-failure bookkeeping, surfaced as `[FI] source failures: ...`.
- `core/figure_sources.py` — license-clean web-figure enrichment for the no-simulation path.
- `core/citations.py` — BibTeX / CSL-JSON rendering of the reference list.
- `core/datasets/` — the `DatasetAdapter` base plus shipped adapters (`worldbank.py`, `wikipedia.py`) for the
  no-simulation auto-collect path.

## Execution / sandboxing

- `core/execution.py` — `Executor` protocol, `VenvExecutor` / `SharedInterpreterExecutor` / `DockerExecutor`,
  `make_executor`. `DockerLimits` (from `execution.docker_memory_gb` / `docker_cpus` / `docker_max_processes`)
  and the container's isolation (`DockerExecutor._create_kwargs`: no network, the limits, `cap_drop=ALL`,
  `no-new-privileges`; limits Docker cannot apply left out), the mounts (`_volumes`: the quest at `/work`, `.fi/` and
  `needs/` read-only over it, `.fi/trials` and `.fi/optimisation` writable for the harness's rows with their run
  records read-only again: `_RECORD_DIRS` / `_SCRIPT_WRITABLE` / `_RECORD_FILES`), the container environment (`_env_for`: host system/Python paths dropped (`_HOST_ONLY_VARS`), quest paths
  as `/work`, thread counts; `_to_container` for command arguments too), the
  non-root user choice checked by a write in the quest folder (`_user_candidates` / `_resolve_user`, in `setup`;
  `_warn_root_owned`), and the plain `[FI]` line for a run stopped by a cap (`_limit_note`).
- `core/proc_tree.py` — `ProcessTree`: start a program so that `kill()` stops it with everything it started (Windows:
  started suspended inside its own Job Object that kills on close; POSIX: its own session and process group), then
  waits until all of it is gone. `AsyncProcessTree` is the same around `asyncio.create_subprocess_exec`
  (`await AsyncProcessTree.start(...)`, `await kill()`, `await aclose()` in a `finally` for cancellation; a test's
  stand-in process gets no job or group). Used where a timeout or a cancellation must stop a whole tree:
  `scripts/import_scientist_skills.py` (`_git`), `generation/_office_pdf.py` (`_run`), `core/engine.py`
  (`_record_environment`'s `pip freeze`), `generation/slides.py` (`_run_cli`), `web/quest_launcher.py` (detached:
  `ProcessTree(detached=True)`, no kill-on-close job, only an explicit `kill()` stops it), the provider proxies
  (`core/provider.py::ProxySupervisor._spawn` / `_terminate`, `ProcessTree`), the experiment script
  (`core/execution.py::VenvExecutor.execute`, also `SharedInterpreterExecutor`) and the CLI providers
  (`core/provider.py::_run_cli` and `_kill_and_reap`), both `AsyncProcessTree`.
- `core/experiment_deps.py` — what a quest environment is given before a run: requested packages minus the quest's
  own files, the selected skills' `pip_requires`, library skills on `PYTHONPATH`, one-at-a-time install fallback,
  skill names pip cannot install explained as the skill, the repair note for what could not be installed. Also the one
  pip-name / import-name table (`IMPORT_TO_PIP`, `pip_name`, `import_names`; `code_project.requirements_for` uses it
  too), the import scanner (`imported_modules`, `third_party`, `third_party_of`, `code_sources`, `skill_sources`,
  `local_module_names`), `env_packages` (asks the quest's Python what an installed package provides),
  `plan_installs` (the requested packages the scripts use, plus well-known unrequested imports; called by
  `Engine._node_execute`) and `warmup_modules` (the post-install test import).
- `core/code_project.py` — `refresh` keeps `code/` a runnable project (README, requirements.txt, `run.py`, and `study.json` for the trial contract; `run.py` repeats FI's trial runner for seed 0, one process per setting, stdlib only, working in `run_output/`; for a search for the best design `study.json` holds the optimisation block and `fi_search.py` is `core/optimise_search.py`, so `run.py` repeats FI's search, one process per design); `.fi/installed_deps.json` (written by `_node_execute`) is the requirements source; `attempt_records.script_hashes` skips the unedited generated files; called by `Engine._refresh_code_project` at the end of `implement` and the start of `analyze`; `record_change` makes one git commit + one `CHANGELOG.md` entry per change inside `code/`, with its kind (`category`) and its "Did the results change" line (`results`, or waiting until `fill_pending`; `undo_pending` closes what an earlier version put back never ran; words in `core/changelog.py`) (`record_note`: an entry and its commit when nothing else changed, used by the improve loop; `fill_pending` commits only CHANGELOG.md) (`rerun_from.back_up` carries `code/.git` over a re-run; `script_hashes` ignore `.git` and CHANGELOG.md); `pin` gives the installed versions for requirements.txt; `verify` (called by `Engine._check_code_project` before `write`, once per code version) runs `run.py` in a clean venv and writes `needs/CODE_PROJECT_CHECK.json`, warning only; `unasked_conflicts` / `mark_asked` feed `Engine._ask_about_edited_project_files` (a `code_project` pause when `pauses.review` is `ask`); a file whose hash differs from `.fi/code_project.json` (a person's edit) is never overwritten.
- `Engine._save_run_data` (called in `_node_execute` only once the run is accepted: exit 0 and the run manifest not
  stopped, pending or repairing; again after the replicate seeds for the raw copy only) copies the data tables the run
  wrote in the quest folder (`_RUN_DATA_SUFFIXES`, each up to `_RUN_DATA_MAX_BYTES`, never the secret/setup names in
  `_RUN_DATA_SKIP`) and a two-script quest's `raw/` into `data/results/`; `data/results/` is not counted as the data a
  quest was given (`attempt_records`' data manifest, `Engine._gather_collected_text`), and `_node_replot_layout` reads it.
- `core/engine.py::Engine._chat_json` — a required check's one-JSON-object call (the design audit, the evidence gate, the claim check): the object is read among several (`_parse_json_lenient(..., want=...)`, `_top_level_json_objects`), and an unusable reply is asked for once more, short (`agents/json_reanswer.md`); a second unusable one is left to the check's own could-not-judge rule.
- `core/flat_output.py` — a simulation whose output does not depend on its inputs is broken, not a finding. `flat_search(rows, objective)` (a search whose every evaluated design gave the same objective and limit quantities, within `SAME_TOLERANCE`, over at least `MIN_DESIGNS` designs), `flat_search_check(check, block)` (FI's check at finer settings found that nudging every movable variable changes nothing) and `flat_sweep(cells)` (a grid whose every returned number is the same in every setting). `OptimisationRunner.execute` and `TrialsRunner.execute` return a failed run (`failed_script: simulate.py`, plain symptom in stderr, `runner.flat` set) before any check or analysis, so the existing repair loop (`_node_execute_reflect`) asks for the repair; `_node_execute` copies `runner.flat` into `exec_result['flat_output']`; after the repairs `_no_results_verdict` gives `stuck_reason: flat` and `_node_stuck_no_findings` writes the plain stop (`needs/STUCK.json`, no paper).
- `core/figure_data_check.py` — a figure must be drawn from the run's results. `typed_series(source, script)` reads a script with `ast` (never runs it) and returns the plotting calls (`PLOT_METHODS`) whose data argument is a literal list/tuple of three or more numbers, or a name bound only to one; axis limits, ticks, reference lines, the settings written along the horizontal axis, colours and sizes are not flagged. `Engine._figures_from_results` (called in `_node_execute` before the run) asks `execute_reflect`'s template (node `implement_figures`) to redraw from the saved results, at most `_FIGURE_DATA_REPAIRS` times per script (`figure_data_repairs` in `QuestState`, reset by `_FRESH_SCRIPT`); `_drop_typed_figures` leaves out what is still typed in (`figures_to_drop`: the file the call's own `savefig` names, else every saved figure), writes `needs/FIGURE_DATA_CHECK.json`, and `_typed_figures_note` hands the writer the limitations sentence.
- `core/latex_text.py` — `escape_latex_backslashes(text, yaml=False)`: inside double-quoted string literals of a JSON (or YAML) text, a backslash that is no real escape, or that begins a LaTeX command (`\sigma`, `\text`, `\nabla`), is written `\\`. Used by `core/engine.py::_read_latex_object` (the one place `_parse_json_lenient` and `_top_level_json_objects` go through: on a failed parse, and on a parse whose `\text` read as a tab), `core/plan.py::repair_block` / `keep_latex_in_block` (a rewritten plan's design block) and the source router in `core/knowledge.py`. `Engine._design_reply` asks the `plan` / `design` steps through `_chat_json` and, when still unreadable, `_stop_for_unreadable_design` (a `model_unreadable` pause); `_unreadable_analysis_verdict` makes the evidence gate route an analysis marked `unreadable` to the `stuck_no_findings` node (`_node_stuck_no_findings`: `needs/STUCK.json`, no paper; `core/todo.py::waiting` shows it). The simulation-stuck stop on `fix/loop-guards` uses the same file and a node named `stuck`; merge them into one.
- `core/engine.py::_script_has_random_source` — whether a one-script quest's script (or a sibling module it imports)
  can draw a random number at all; none (and the seed ignored) makes it a deterministic study
  (`result_json_no_random_source`): the remaining replicates are skipped and the analysis is told to report one run,
  not an interval. This is the only early stop in `_node_execute`'s replicate loop: a script with any random source runs
  every configured seed, and only then is an all-agreeing set classified (`result_json_deterministic` = every seed agreed,
  replicates kept; or `result_json_replicate_seed_ignored` = one run repeated, when the seed reaches no generator).
- `core/code_layout.py` — the default shape of a simulation's `code/`, a small research tool: the model's equations in
  a package of their own (`code/<package>/`, name from the title via `package_name`, kept in `.fi/code_layout.json`),
  `simulate.py` as the scenario FI still calls, `tests/test_oracles.py` (`oracle_tests`, the plan's checks as unit tests,
  same seed as `trial_runner.run_case`) and `METHODS.md` (`equation_map` / `methods_text`, each `generates` equation to the
  function carrying its label), both written by FI through `code_project.refresh(extra_files=...)`. The cost over two
  scripts is worked out at plan time (`estimate`), decided against `execution.code_package_max_extra_lines` /
  `code_package_max_extra_calls` (`decide`) and shown in plan.md (`plan_lines`, section *How the code will be laid
  out*, via `plan.render(code_layout=...)`). Engine hooks: `Engine._code_layout` (None off the two-script path),
  `_plan_code_layout` (in `_node_plan`), `_code_package_reply` (in `_node_implement`: reads the package from the reply,
  asks once more, else keeps two scripts and says so), `_package_in_use`, `_check_code_layout` (in `_node_execute`
  beside `_check_equation_labels` and again after an oracle repair: `check` plus `unseeded_rng_calls` over the package, a warning, a stop only under research through the `code_layout` pause, `todo._ADVICE["code_layout"]`).
  `_code_layout` is `None` for code written before the quest decided a layout (no `.fi/code_layout.json` beside an
  existing simulate.py). The request limit is a quest-wide budget (`calls_left` / `spend_call`, counted in the record);
  `_adopt_reply_package` keeps a package the reply named itself when simulate.py imports it; `split_run.simulation_sha`
  and `attempt_records.script_hashes` (FI-written METHODS.md / tests left out) know the package.
  `improve.editable` / `snapshot` / `restore` / the kept-version snapshots include the package's modules by their
  path in code/ (`improve.edited_path`), so the improve loop can change an equation there.
  `trial_runner._run_key` and `Engine._simulation_sources` read the package as part of the simulation.
  Where simulate.py is rewritten the package goes with it: `Engine._package_shown` (shown by `_extend_directive`; an
  extension's package file missing functions it had is not written, `dropped_functions`), `_package_repair_note` /
  `_apply_package_repair` (`repair_note` / `repair_files`: the reflect and oracle repairs of simulate.py may return
  `package_files`), `_package_snapshot` / `_restore_package` (the oracle gate's undo of a repair bent to a disputed
  check), `_label_target` (labels asked for in `model.py`); `code_project.requirements_for` scans the package's imports.
- `core/split_run.py` — the two-script contract (`simulate.py` / `experiment.py`), raw-dir naming, replicate seeding.
  Whether a quest gets it is `Engine._split_on`: under `split_analysis: auto` every quest that runs a simulation
  (`_runs_code`), deterministic (`run_cell`) or stochastic (`run_trial`); `design_is_stochastic` only decides whether
  a one-script quest (`split_analysis: false`) is a gap for having no run manifest.
- `core/job_watch.py` — background jobs (HPC / a cluster) that outlive one `execute` call; `--watch`.
- `core/example_inputs.py` — staging a user's own example files into a quest.
- `core/plot_style.py` — the shared matplotlib house style every generated figure uses.

## Skills

- `core/skills/registry.py` — discovery, self-tests, what may load.
- `core/skills/selection.py` — which skills a quest's catalogue carries (`layers.py` = general vs. domain layer).
- `core/skills/approval.py` — the human sign-off gate (content-hash pinned).
- `core/skills/removal.py` — `--remove-skill` / `--restore-skill`: delete a skill in FI's own folder, or hide an external one (`skills_removed_external.json` beside the ledger, honoured by `discover()`).
- `core/skills/scan.py` — a static review of a skill's contents for the person about to approve it.
- `dev/own-skills/` — the four skills written for FI (`invariant-guards`, `reference-cross-check`, `numerical-property-tests`, `dimensional-consistency`); the import script reads them from here (`LOCAL_SOURCES`, repo key `fi`), no clone.
- `core/skills/known_requirements.py` — the pip packages each preset skill needs (`scripts/import_scientist_skills.py`
  reads it) and the versions a skill must not get (`PINS`); the fallback for a skill imported before provenance
  recorded `pip_requires`.
- `core/skills/importer.py`, `core/skills/scaffold.py` — importing a skill written for another agent, or drafting one
  from existing code.
- `core/skills/known_checks.py` — known-value checks (a physical constant, a reverse complement) that a generated
  self-test runs for a script-less skill whose library is declared, so a broken install fails instead of passing.
  `core/skills/scaffold.py` writes each one into `selftest.py` as a plain function (no `exec`), so the skill scanner
  sees the code it runs and a generated self-test carries no high-severity finding.
- `core/skills/mounts.py` — making an approved external skill reachable inside the Docker sandbox.
- `core/skills/usage.py`, `core/skills/selftest_cache.py` — usage provenance, and which exact contents last passed
  self-test under which Python.

## Outputs

- `generation/paper.py`, `poster.py`, `slides.py`, `speech.py` — the four output generators.
- `generation/_visual_check.py` — screenshot + measurement (+ optional AI check) of every rendered output.
- `generation/_pdf_measure.py`, `_pdf_engine.py`, `_pandoc.py`, `_html_pdf.py`, `_office_pdf.py` — PDF rendering and
  measurement plumbing shared across generators.
- `generation/_pptx_slides.py`, `_pptx_math.py` — native-PowerPoint deck rendering and equations.
- `generation/_marp.py` — Marp CLI discovery for `slides.html` / `slides.pdf`.
- `generation/_keywords.py`, `_figure_captions.py`, `_tables.py`, `_cjk.py`, `_skip_md.py` — small shared renderers.
- `core/replot_figures.py` — redrawing a line figure as the mean-over-seeds version.
- `core/data_shape.py` — `describe_data_files`: what a saved data file holds (`.npz`/`.npy` keys, dtype and shape from the header only, never unpickled; JSON keys; CSV/TSV columns and row count), size-bounded, for the layout redraw's prompt.
- `core/rerun_from.py` — `--resume <id> --from <step>`: one table per step (`STEPS` nodes, `REDOES` sentence, `OUTPUTS` files moved aside, `GROUPS` block for a map, `NEEDS_APPROVAL` for the steps up to the design, `LEADS_INTO` for seeding state before them); `Engine.rerun_steps` returns `step_info` dicts, `MAP_BLOCKS`, `NODES` (every graph node with its block, plain title, sentence, reads, writes and loop note; `tests/test_quest_map.py` pins it to the graph's nodes), `node_map` (adds status and clickability) and `map_payload` (blocks + finished + nodes) turn them into the quest map; `Engine.quest_map` builds it and `Engine.stopped_at` says where the quest is (the checkpoint's next nodes; finished only at the graph's end with no `pause.json` / `NEXT_STEP.md` / `quest_failed.md`, never from the summary file): the web `rerun-steps` API and `launch.py --resume <id> --from --json` return it, `web/static/quest_map.js` + `quest_map.css` draw it (`web/static/quest.html` `renderQuestMap`; the VS Code panel `vscode-frontier-insight/src/quest-map.ts` loads the same two files from `media/`, copied by `scripts/copy-quest-map.js`), `Engine._keep_design_history` restores `DESIGN_HISTORY.json` into the state before a pre-design restart, and `Engine.run(approved_by=...)` refuses those steps without a name. `OUTPUTS` of a step include those of every later step. A graph node with no step must be listed in the module docstring and in `tests/test_rerun_from_nodes.py::NOT_A_STEP`. `skills` picks the skills again from the current config (`Engine._repick_skills`, on the checkpoint just before the code); `checkpoint_before` finds the checkpoint before a step (used by `Engine.run`); `reached`, `REDOES` and `listing` give the steps a quest reached on the line it is on now (a checkpoint of a line left by an earlier `--from` does not count) (`Engine.rerun_steps` for `--from` with no step, the web `rerun-steps` API and `@fi /resume --from`); `back_up` moves that step's and later outputs to `.fi/previous/<time>/` (a move that fails half way puts back what it had moved). A restart before the design records the replaced protocol as a post-hoc amendment (`frozen_protocol.record_replacement`). `in_quest` takes a path the checkpoint holds (the paper's) to the same place in the quest's current folder, so a copied or moved quest rerun from a later step judges its own paper (claim check, review, the review pause, `core/evidence.py`), not the old folder's. The path a map draws comes from `Engine.rerun_no_simulation()` (config flags or what clarify put in the checkpoint); a node is clickable only once the quest reached it.
- `core/paper_patch.py`, `core/paper_trim.py` — a targeted revise (only the flagged passages) and a page-limit trim.

## Command-line tools and the web/VS Code surfaces

- `launch.py:_start_menu` (`START_CHOICES`) — bare `fi`: three next steps in a terminal (`--new`, `--serve`,
  `--doctor`), else the short help on stderr and exit 2.
- `launch.py:_bootstrap_or_reraise`, `_relaunch` — the self-setup a missing dependency triggers at the top-level
  import (installs into an activated venv/conda env in place, silently relaunches into an already-complete
  `.venv/`, or does the full ask-and-install-and-relaunch dance); see `docs/capabilities-reference.md#getting-started`
  for the exact decision tree. Every `vscode-frontier-insight` spawn of `launch.py` opts out (`FI_SKIP_BOOTSTRAP=1`)
  in favor of its own `frontierInsight.pythonPath` contract.
- `core/critique.py`, `core/digest.py`, `core/portfolio.py`, `core/proposal.py`, `core/analyze_cli.py`,
  `core/summarizer.py`, `core/state_dump.py` — the one-shot CLI tools (`fi tools <name>`, see `launch.py:
  _TOOL_SUBCOMMANDS`); `core/proposal_seed.py` seeds an interview from a saved proposal.
- `web/server.py` — the FastAPI status server (quest list, log stream, evidence, trace, amendment banner, ...).
  The Settings page's knowledge-base card: `_knowledge_config_info` (`GET /api/knowledge/info`, settings only) and
  `_knowledge_inventory` (`POST /api/knowledge/inventory`, opens the store; on request, in a thread, time-limited).
- `web/interview_routes.py`, `web/skills_routes.py`, `web/tools_routes.py` — the web UI's interview, skills and
  CLI-tools surfaces. The interview keeps an interview's answers as they are typed (`GET/PUT
  /api/interview/draft/<id>`, `_drafts/.interview/<id>.json`, removed and marked `.launched` on launch) and makes
  submit idempotent (`Idempotency-Key` / `submit_key`: one submit at a time, the first answer kept in
  `_drafts/.submits/<key>.json` and returned to a retry). The page (`web/static/interview.html`) has `fetchJson`
  (time limit, `res.ok`, plain error), a Retry panel (`showStartError`), saved answers (`saveAnswersSoon` /
  `restoreSavedAnswers`, localStorage + server, id in `?d=`) and one Launch at a time.
- `web/static/vendor/` — the fonts, the icon font and the Tailwind script every page loads, served by FI so a page
  works offline (`README.md` there; `scripts/vendor_web_fonts.py` refreshes the fonts). Tests drive the pages in a
  real browser with no network through `tests/web_browser_harness.py` (Playwright routes to the app in-process,
  with fault injection).
- `web/quest_launcher.py` — the subprocess pool for quests started from the web UI, each a detached
  `core/proc_tree.py` `ProcessTree` (it outlives the server; Cancel stops the whole tree). Its children (and the quest page's
  Resume) run with `FI_WEB_ANSWERS=1`, so `launch.py` `_web_page_clarify_callback` asks the setup questions on the
  quest page (`.fi/clarify_questions.json` → `.fi/clarify_answer.json`, with `.fi/clarify_waiting.json` holding the
  waiting child's pid); `web/server.py` `_clarify_run_waiting` reads that file, so `POST /api/quests/{id}/clarify`
  says `run_waiting` and `POST .../resume` refuses (409) while the child waits, even after a server restart.
- `vscode-frontier-insight/src/extension.ts` — the `@fi` chat participant and every slash command.
- `vscode-frontier-insight/src/bridge.ts`, `persistent-bridge.ts` — the `vscode.lm.*` bridge a spawned `launch.py`
  talks to over a local socket; `lm_done` carries `served_model` (`lm-messages.ts` `servedModel`), which
  `core/vscode_bridge.py` hands to the call's own `LAST_SERVED` and `core/provider.py` records as `LAST_CALL`.
- `vscode-frontier-insight/src/roots.ts`, `roots-config.ts` — where FI is and which folder a command works in:
  `locateFi` (setting → an open folder with `launch.py` + `core/engine.py` → the configured Python's
  `find_spec("launch")`, never importing it → the folder picked once, kept in `~/.frontier-insight/fi_location.json`
  → one folder picker), `chooseWorkFolder` (several folders open: the named YAML's folder → the active editor's →
  the first that is not FI's checkout), and `rootsForCommand`, which every command awaits (a found folder is kept
  for the window's session; a cancelled picker asks again on the next command).
- `vscode-frontier-insight/src/interview-core.ts`, `interview.ts` — the VS Code interview (mirrors `core/interview.py`
  question-for-question; keep both in sync when the question set changes).
- `vscode-frontier-insight/scripts/normalize-vsix.js` — run by `npm run package` after vsce: sorts the zip entries,
  fixes their timestamp and compression so a build is byte-reproducible (same Node major), and writes
  `vscode-frontier-insight.vsix.manifest.json` (sha256 per packed file; committed with the `.vsix` by
  `.github/workflows/vsix-rebuild.yml`).

## Platform / diagnostics

- `core/platform.py` — `--doctor`'s machine-capability detection.
- `launch.py:_doctor` — `--doctor`: quick by default (this computer only, each check bounded by `_bounded`, the
  engine not imported: `_quick_doctor_argv` / `_import_runtime`), `--deep` loads the embedding model and runs
  `provider_readiness.preflight` for every provider.
- `core/demo.py`, `launch.py:_demo` — `fi demo`, the first step after installing: writes `fi-demo.yaml` (the example
  is a string in the package; `examples/` is not in the wheel) without overwriting, checks the providers set up on
  this computer in turn, asks before running (no terminal: prints the command). `tests/test_wheel_quickstart.py`
  runs it from a freshly built wheel.
- `core/bridge_path.py`, `vscode-frontier-insight/src/bridge-path.ts` — the canonical persistent-bridge socket path,
  kept identical on both sides. A second VS Code window open at the same time binds `<path>-<pid>` instead
  (`persistent-bridge.ts` `listen`), and its `/update` / `/generate` terminals are handed that address.
- `scripts/wiki_sync_index.py` — rebuilds `wiki/index.md` from the pages (`[[slug|Title]]`, grouped by front-matter
  `type`); `--check` reports a stale index or a wiki link to a missing page. `tests/test_wiki_sync_index.py` and the
  `wiki-check` CI job (runs when `wiki/**` changes) keep the repo's wiki index current.
- `core/replay.py` — answering a quest's model calls from a recording: `Recording` (calls keyed by step and number
  within the step, never by prompt hash; `load` reads a `calls.jsonl`, `from_quest` reads `.fi/io/` with
  `output.save_model_calls`, joined to `.fi/model_calls.jsonl` for who answered), `ReplayClient` (stands in for
  `Engine._client`: `record` / `partial` / `replay`, planted answers, `LAST_CALL` set from the recording so a replayed
  panel keeps its models, an unrecorded call gets `CANNOT_FIX` and a divergence event; writes `calls.jsonl` and
  `replay_events.jsonl` to a folder outside the quest), `CrossrefReplay` (records / replays
  `core/retractions.py::check_dois` per DOI). Shared by the self-benchmark and the record-and-replay of quests.
- `dev/evaluation/bench/` — the self-benchmark (`fi tools bench`, dispatched by `launch.py:_run_bench`; dev-only, not
  in an installed FI): `answers.py` (the `dev/quest-topics/answers/*.answer.json` format, its checks, held-back tasks
  by hash), `catalogue.py` (the 17 planted errors and the gates `GATES`), `plant.py` (R1, R2, N3, S1, L1 planters),
  `validity.py` (the valid-error filter through `core/trial_runner.py::run_trials`), `runner.py` (`BenchEngine`, an
  `Engine` whose `_connect_llm` wraps the client in a `ReplayClient`; `BENCH_SETTINGS`; `fork`), `score.py` (the
  answer from `raw/trials.json` checked against `.fi/trials/run.json`'s hashes, "would be published", gates, the
  metrics with Wilson intervals), `report.py`, `reference.py`, `cli.py`. README there; end-to-end on fake-model
  quests in `tests/test_self_benchmark_e2e.py` (slow).

## Prompts (`agents/*.md`)

One `string.Template` file per node/persona; loaded once at engine init (`core/engine.py`). Grep `agents/` for a
topic before writing a new prompt — most nodes already have one, and a persona variant (`write_persona_*.md`,
`review_persona_*.md`) is usually the right way to add a new voice rather than a new node.

- **The methodology audit acts on FI's own rewrite** - `Engine._audit_the_design_that_runs` (called at the end of `_node_design`): when plan.md's latest version was written by FI (`_FI_PLAN_AUTHORS`: `model` / `engine` in `needs/PLAN_HISTORY.json`, file unchanged since) and no protocol is frozen (`_frozen.load`), the audit may amend the design once per design hash (`.fi/design_audit_acted.json`, so a resume does not act twice); the amendment is written with `_write_audited_design_to_plan` (`plan.edit_design_block`, a new `engine` version) and the audit then looks again taking nothing, so the receipt is for the design that runs. A person's edit or request, a frozen protocol, or any design that is not plan.md's keeps record-only behaviour.
