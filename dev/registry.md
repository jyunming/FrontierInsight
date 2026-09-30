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
  missing, `$checks` in `agents/plan_revise.md`). The stop (`_pause_for_plan(unsourced_checks=...)`) offers three ways
  on and writes `needs/UNSOURCED_CHECKS.json`.
- `core/accepted_checks.py` — going on with checks whose expected value has no stated source: `write_pending` /
  `pending` / `clear_pending` (what the stop named, and what the stop before named, so the next stop says what changed),
  `accept(root, who, via=...)` (`needs/UNSOURCED_CHECKS_ACCEPTED.json`; a name is required), `fingerprint` (a choice
  binds to a check's name and numbers: expected, tolerance, case, measure), `covers` / `chose` / `changed_since` (the
  engine goes on only when every unsourced check was chosen as it is now; `Engine._unsourced_oracles` and
  `_not_confirmed_names` mark the evidence gap "source not confirmed" in `_oracle_source_gaps` and the freeze's
  `approved_by`, for checks still without a source), `disclosure(root, names)` (the writer's note). A source the fill
  wrote after the person read the plan is named in the freeze too (`.fi/plan_sources_filled.json`). Surfaces:
  `launch.py --accept-checks <quest> --approve-as <you>` (`_accept_checks`), `POST /api/quests/{id}/plan/accept-checks`
  and `GET .../plan/unsourced` (the quest page's *Go on as it is*), `@fi /accept-checks` (`skills.ts::runAcceptChecks`).
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
  The plan.md section is `HEADING`; `plan.raw_design_block` / `plan.edit_design_block` read and edit the block as
  written, `plan.add_to_section` / `plan.refresh_model_section` keep the prose in step.
- `core/oracle_review.py` — a second opinion on the plan's checks (`agents/oracle_review.md`, node `oracle_review`):
  `prompt_parts`, `parse` (strict: only declared checks and equations), `findings`, `request`, `plan_lines`. Called by
  `Engine._review_oracles` from `_hold_oracle_forms` (one call; its findings share the one `plan_revise` request with
  the form requests; `.fi/oracle_review.json` keeps the answer and how far the look got, so a resume asks nothing
  again, and `core/rerun_from.py` moves it aside with the plan). The reviewer model is `provider.node_models.oracle_review`; the VS Code node picker lists it
  (`OTHER_NODES` in `vscode-frontier-insight/src/interview-core.ts`).
- `core/criteria.py` — how a quest judges whether its code got better: the protocol's `criteria` (two to five checks of
  correctness, each from a declared oracle or FI's trial record; never a headline metric or the
  script's results). `normalize` (strict, called by `plan.normalize_protocol`) / `repair` (a draft, called by
  `plan.repair_protocol`), `describe` / `plan_notes` / `countable` (the plan's *How we will judge whether the code got better* section,
  rendered by `plan._criteria_lines`), `evaluate` / `se_rate` / `meets` (the values after a run), `record` / `history`
  (`.fi/criteria_history.jsonl`) and `summary_line`. `receipts.design_core` leaves the criteria out of what the
  methodology audit vouches for; `Engine._hold_design_to_frozen` leaves an amendment's unusable criteria out. `Engine._propose_criteria` (one search, one more question,
  `agents/plan_criteria.md`) runs in `_node_plan` when the draft names none; `Engine._record_criteria` runs at the end of
  every `_node_execute` (oracle values from `oracle_check.last_judged`, trial values through
  `trial_runner.recorded_series`, the commit through `code_project.head`).
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
  `split_analysis: false`, `background_jobs`) before anything is implemented or run.
- `core/optimise.py` — the engine runs the search for the best design: `OptimisationRunner` (the executor stand-in
  `_node_execute` picks when the design has an `optimisation` block and two scripts, instead of
  `trial_runner.TrialsRunner`; it runs the search, then `experiment.py` with `FI_OPTIMISATION` / `FI_BEST_DESIGN`, and
  `FI_TRIALS` for the coarse scan), `run_search` (the coarse scan through `trial_runner.run_trials`, each design through
  the trial harness in its own process with a nonce, a library method's steps through `.fi/optimisation/optimise_search.py --drive`
  in the quest's Python, the budget and `execution.timeout_s`, the run key cache in `.fi/optimisation/run.json`),
  the record FI alone writes (`raw/optimisation_ledger.jsonl`, `results/best_design.json`; `restore` / `read` put
  FI's copy back), `summary_lines` (run.log; `CHECKS_AT_FINER_SETTINGS` is the switch the finer check will flip), and
  `complete_case` (an oracle's case gets the fixed conditions, search settings and baseline, in `Engine._oracle_gate`),
  `with_fi_record` (FI's numbers added to the analysis's RESULT_JSON as `fi_search`), `searches_itself` (an optimiser in
  the simulation's code: a warning). `Engine._node_execute` calls `_stop_if_the_search_cannot_start` too.
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
- `core/evidence.py` — the six-level evidence ladder (`assess`, `summary_line`, `upgrade` for older records); `_trace_completeness_gaps` reads the trace's `quest_finalized` seal (written last by `Engine._seal_trace`, naming `SEALED_FILES`); `SEALED_LEDGERS` and `SEALED_QUERIES` (the signed record of the search queries, `engine._record_query_set` / `_query_set_standing`; a person's own in `inputs/search_queries.txt` ; the same file also holds each pass's source verdicts, stages `floor` and `screen`: `Engine._record_floor_verdicts` / `_record_source_verdicts`, digest fields `_SOURCE_VERDICT_HASHED`, so no seal change) are required in the seal; `read` / `verify_seal` are how every surface reads `needs/EVIDENCE.json` (a record written before its seal says `trace_seal: pending`).
- `core/audit_log.py` — the hash-chained, redacted per-quest trace (`STAGE_PROGRESS`/`_ProgressOnly` for the curated
  console/web view live in `core/engine.py`, next to `_quest_logger`).
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
- `core/phased.py` — explore, then confirm (`engine.phased`, off by default): `.fi/phased.json` (`load`, `stage`, `status`), `prepare` (at every start, `Engine._phased_prepare`: holds back about 30% of the rows of each CSV/TSV in `inputs/data/`, chosen by a hash of each row's text (once exploration ran, only rows held back before), into `store_dir` = `<output_dir>/_held_back/<quest_id>/{held_back,original}/` outside the quest folder (`_move_out_of_quest` moves the older `.fi/phased/` layout), or says why the confirm run gets new seeds only; finishes a start cut short; in the confirm stage gives the run the held-back part again, `mark_compromised` when it cannot), `restore_inputs` (the whole files back when `Engine.run` ends, paused or not, also with the setting off; names a file it left alone), `turned_off` (`Engine._phased_turned_off`: a start with the setting off on a quest with a record), `note_job_pending` (`_wait_for_job`: a resume checking on the confirm run's own job is not a second run), `note_confirm_result` (`_node_execute`: the gate records only the result the confirm run produced), `Engine._phased_seed_gap` + `mark_unconfirmable` (no data held back and a result new seeds cannot change: runs agreed, `run_cell`, or the experiment and the modules it imports never take `FI_REPLICATE_SEED` into a generator and draw only from fixed seeds (`_own_modules`; FI's `run.py` does not count): status `not_confirmable`; `record_confirm` also when a new-seeds confirm run gives exactly exploration's result), `Engine._phased_fresh_background_job` (sets exploration's `job/state.json` aside for a background job's confirm run; `_JOB_SEED_PROTOCOL`, only when on, tells the driver to pass `FI_REPLICATE_SEED` on to the job), `_write` (temporary copies in the store, never in `inputs/data/`; `_clear_stray_tmp`), `confirm_run_started` (the gate records a confirm result only after the confirm run started; `status` is `confirm_reused` once the confirm data or seeds were run on twice), `mark_compromised` (a file could not be split or put back, a replayed result, seeds the run cannot take: status `compromised`, nothing confirmed; writes a record when there is none), `seed_env` (the replicate seeds in `_node_execute`: remembered in exploration, moved to the confirm base after), `enter_confirm` / `record_confirm` (called by `Engine._phased_route` from `_route_after_evidence_gate`: the evidence gate's `confirm` edge to `execute`, added to the graph only when on; `_freeze_protocol_if_due(at_confirm=True)` freezes there, not before the first run), `evidence_settings` (`evidence.assess` `settings["phased"]`, `_PHASED_GAPS`), `write_note` / `mark_paper` (the writer's note and the paper's note, no numbers in it); `_route_after_cross_check` does not redesign in the confirm stage; reads no attempt record or shadow recommendation.
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
- `core/provider.py` — every transport (`LLMClient`), `ProxySupervisor`, `missing_api_key`, model pricing; `LAST_CALL` (who answered the current task's last call, and why its answer ended); `ModelAnswerTruncated` / `ModelAnswerFiltered` / `outcome_of` (an answer cut off at its limit or withheld is never returned as whole; `Engine._pause_for_model_output` turns either into a `model_output` pause); `node_output_limit` (`provider.node_max_tokens` per step); `quest_run_log` (a failed call made outside the engine still reaches the quest's run.log); `_http_streams` / `_post_streamed` (Moonshot calls are streamed). The HTTP retry policy waits out a server outage or rate limit (`_http_outage_status`, `_http_retry_wait` / `_http_retry_stop` / `_http_retry_sleep`: six attempts on `_HTTP_OUTAGE_WAITS_S`, a sane `Retry-After` up to `_RETRY_AFTER_MAX_S`, at most `_HTTP_OUTAGE_MAX_WAIT_S` per call, the `FI_MAX_CONCURRENT_LLM_CALLS` slot given back while waiting); a used-up quota is `_is_exhausted_quota`; `FallbackLLMClient` sets `short_retry` (`_http_short_retry`) while a later provider's circuit is closed.
- `core/provider_models_discover.py` — runtime model-list discovery for the provider picker.
- `core/ensemble.py` — the multi-model fan-out-and-merge primitive a node opts into.

## Knowledge / literature

- `core/knowledge.py` — `Knowledge`, the three-layer retrieval (pinned papers → Axon → external router); `is_open_access(metadata)` is the one rule for "free to download" and `_free_locations(metadata)` the only addresses requested for a free paper (the source's own `free_url`, or an address on a free host; never a DOI or publisher page); `_get_checked` follows redirects by hand and `_redirect_allowed` re-checks every hop (the headless render does the same per hop in `_playwright_fetch_html`, through the route handler `_make_navigation_guard`, whose `_route_call`, on the "page already closed" error only, marks the page's routes "ignore errors" and re-raises so Playwright drops it without printing a traceback); `_looks_scholarly` keeps journal-article web hits out unless free (gates `_fetch_full_text` / `_fetch_web_page_text`; `engine._is_open_access` delegates to it); `add_quest_artifacts` writes accepted or preliminary kinds by `metadata['standing']` (decided in `Engine._write_back_knowledge`; read back by `engine._is_preliminary_memory` / `_preliminary_reminders`); the copy under the other standing (`STANDING_KINDS`) is removed first (`_retire`, `_delete_in_process`), and `retire_stale_standing` / `retire_stale_ref_spines` back `fi tools tidy-knowledge`; a removal that fails stops the write (`last_writeback_problem`, which the engine puts in run.log and `.fi/knowledge_problem.json` for the card); cited papers' entries (`REF_SPINE`) keep every accepted consumer (`_paper_entry`, `_drop_consumer`, under `_paper_entries_lock`).
- `core/axon_http.py`, `core/axon_sidecar.py`, `core/axon_endpoint.py` — talking to the shared Axon service: HTTP
  client (`AxonHTTPBrain`: ingest, delete_documents, list_sources, search_raw), sidecar lifecycle (start/reuse/stale-lock clearing), endpoint discovery.
- `core/passages.py` — relevance-ranked excerpt selection over fetched full text.
- `core/retractions.py` — the retraction check: `check_literature` looks each DOI up in Crossref (`updated-by`
  notices, Retraction Watch included) after the literature node's dedup and sets `metadata["retraction"]`
  (`retracted` / `not_retracted` / `not_checked` / `no_doi`); `apply_to_claims` makes a claim grounded in a retracted
  source, and any sentence citing one, unsupported after the claim check; `retracted_dois` keeps retracted papers off
  `WANTED_PAPERS.md`; `summary_line` is the run.log line; `retracted_in_record` feeds the to-do card. Tests never reach
  Crossref: `tests/conftest.py` answers "not checked" outside `tests/test_retractions.py`. The `[retracted]` mark itself is added by `core/engine.py::_format_lit_header` and `_claim_source_block`.
- `core/trial_runner.py` — the trial contract: FI runs `run_trial` / `run_cell` of `simulate.py` for every setting,
  one process per setting, writes `raw/ledger.jsonl` and `raw/trials.json` itself (`TrialsRunner` in `_node_execute`,
  `measure_oracles` in `_oracle_gate`: it calls the simulation on each oracle's case via `run_case`, and `run_oracle` only for an oracle without a case); `recorded_values_by_cell` / `given_values_not_run` hold each reported value to the trials of its own settings; under research `recorded_rows_by_cell` / `given_rows_problems` (via `Engine._given_row_findings`) check a mean over a subset trial by trial; the harness hands every trial the protocol's thresholds as `FI_THRESHOLDS` (`run_oracle` too; the proportion's 0/1 under its id and the averaged quantity under the mean's own id, `RETURN_MEMBERSHIP`); the older self-looping contract stays in `core/split_run.py` as `self_reported`. On a cluster (`execution.background_jobs`) `prepare_cluster` / `collect_cluster` run the settings as a job array submitted by `code/submit.py` (`TrialsRunner(submit=...)`); `code_changed_while_queued` compares the code at submission with the code at collection.
- `core/profile.py` — the person's author line, asked on the first interview and kept in
  `~/.frontier-insight/profile.json` for the CLI (`launch._run_new`), the web page (`/api/profile`, saved on
  submit) and VS Code (`interview.ts` `loadProfile` / `saveProfile`).
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
- `core/thinking_capture.py` — the model's own account of its reasoning: a task-local holder (`open_holder` / `note_thinking` / `add_thinking`) that each transport fills (`provider._reasoning_of` and the streamed `reasoning_content` deltas for HTTP, `provider._stream_thinking` for the Claude CLI, `vscode_bridge.LAST_BRIDGE_THINKING` fed by `bridge.ts`'s `lm_done.thinking`); `Engine._recorded_call` opens it and `Engine._save_thinking` writes `.fi/thinking.jsonl` (redacted, one line capped at `THINKING_LINE_CHARS` and the file at `THINKING_FILE_BYTES`, `call_id` joined to `.fi/model_calls.jsonl`, `provider`/`model` of the answering provider, not sealed, `output.save_thinking`); `LLMClient._chat_impl` and each `FallbackLLMClient` rung reset the holder so one attempt's reasoning is never filed under another's; `lm-messages.ts::lmDoneMessage` (both bridges) cuts the reasoning so the `lm_done` line stays under 48 KiB. Asking for it: `thinking_capture.wanted()` (a holder open with `want=output.save_thinking`) becomes `lm_request.ask_thinking`, and `lm-messages.ts::ThinkingRequests` (both bridges) sends Copilot's `modelOptions._enableThinking`, asks again once without it on a refusal (`thinking_declined` on `lm_done` → `note_declined` → `Engine._say_once_about_thinking`, which also says once per step that no reasoning came back) and turns a stream that failed before its first part into `THINKING_DECLINED_MARKER`, which `provider._TRANSIENT_BRIDGE_MARKERS` retries; `lm-messages.ts::thinkingText` and `thinking_capture.as_text` take a `string[]` value. Codex: `_CliSpec.reasoning_extractor` / `provider._extract_codex_reasoning` (the `reasoning` items of `codex exec --json`). `_chat_provenance` also carries `call_id` / `requested_model` / `finish_reason` into every `model_claim` and, through `_last_chat`, into an attempt record's `models_used`, whose `call_id` differs per call.
- `core/quest_title.py` — a quest's title after it ran: `current_title` (paper H1 / front matter, else `config.yaml`), `rename` (paper.md copies, `config.yaml` one-line edit, `frontier_insight_summary.json`, the latest checkpoint's `title` / `title_confirmed` edited in place, all files written or none, the quest index's title (`quest_index.set_title`), then a `title_changed` audit event carrying the paper's hash before/after, which `evidence._trace_completeness_gaps` follows from the seal and `audit_log.after_title_changes` skips for `--trace --follow`; `RenameRefused` / `QuestRunning`), `looks_running`. Called by `launch._rename_quest` (`--rename`, `fi tools rename`), `POST /api/quests/{id}/title` (quest page **Rename**) and `trace.ts::runRename` (`@fi /rename`); `launch._finish_outputs` writes `title` into the summary.
- `core/why.py` — `--why` / web Why? / `@fi /why`: why a quest stopped, why the review asked for a revision, why the evidence is at its level, why a step decided what it did (with `_thinking_line`: whether `.fi/thinking.jsonl` holds that run of the step's reasoning and how many characters), and `reasons` (web **Model's reasons**): every step's stated reasons, from the audit trace and records (no model call). The reasons come from `Engine._audit_claims` and `Engine._audit_stated_reasons` (ideate, select_skills, cross_check, execute_reflect, clarify, the review's `why` and moderator), `Engine._claim` (only under a call made during that run of the step: `_chat_at_node_start`), and the write step's `REFINE_WHY:` line (`engine._take_refine_why`). The model a VS Code quest uses: `launch._apply_vscode_chat_model` (`--vscode-chat-model`, passed by `extension.ts` through `lm-messages.ts::chatModelArgs`), `Engine._say_model_change` / `_note_served_model` (run.log lines and `model_changed` trace events). `launch._follow_trace` is `--trace --follow`; `provider.set_model_call_archive` / `append_cost_row(messages=, response=)` keep every model call in `.fi/io/` when `output.save_model_calls` is on.
- `core/engine.py` refine routes — `_take_refine_points` reads the writer's `NEEDS_EXPERIMENT` / `NEEDS_DATA` / `NEEDS_LAYOUT` lines; `_route_after_write` sends them to `design`, to `implement` in extend mode (`_extend_directive`: the existing scripts are the base) or to `_node_replot_layout` (`agents/replot_layout.md`: redraw from the saved files under `data/results/` and `data/`, no experiment run); `_route_after_human_feedback` sends a refine with notes to `write` and one without (a re-open) to `design`; `_node_write` records `refine_scope` (`paper` / `data` / `layout` / `experiment`) for the route's trace facts; state `refine_extend` / `refine_layout` / `extend_missed` / `layout_missed` (asked but not done, also when collection is off or a redraw has no figures, fails, or prints `NOT_DRAWN:`: `_route_after_replot` sends the paper back to the writer once, which says so and clears both). An extension that ran records `extend_check` (what was asked, the result names before it); every later write checks the paper reports what it added (`Engine._report_what_the_extension_added`, `number_provenance.unreported_results`, no model call), asks the writer once more when it does not, and leaves `extend_unreported` for the review pause's steps. An extension is recorded in `PROTOCOL_CHECK.json` as `extended_by_person` (`_goes_beyond_protocol_by_request`), which the run-manifest check and `_goal_coverage_notes` read (`_protocol_drift_not_asked_for`: only the differences it recorded are excused, any other drift is still reported); `auto_collect_data` searches for the ask; `_node_replot_layout` backs up figures and data in `.fi/layout_backup` and `_restore_layout_backup` undoes a redraw cut short; it shows the model what each saved file holds (`core/data_shape.py`), writes a failed (or no-change) script once more with its error, and when both tries fail sets `layout_not_redrawn`, which `_node_human_feedback` puts first on the review card and clears once answered.
- `core/todo.py` — the to-do card every stop writes (`NEXT_STEP.md` + `.fi/todo.json`): per-kind decision, recommendation and alternatives (`advice`; under a rigor profile, never a setting it refuses: `refused_settings`, and what to do instead by pause kind and freeze: `research_instead`), the other things waiting (`waiting`), printed by `launch.py`, read by the web `/next-step` endpoint and VS Code.
- `core/receipts.py` — the receipt each required check (evidence gate, design audit, claim check) writes under
  `needs/receipts/`; `core/evidence.py` reads them for `publication_ready`; `engine._stop_once_for_check` is the
  research profile's one stop and retry.
- `core/pdf_text.py` — every PDF FI reads: all pages in reading order (PyMuPDF if installed, else pypdfium2), scanned
  pages by OCR (tesseract, else RapidOCR), a size cap that names its page. Fetched scans are OCR'd after the fetch
  (`knowledge._ocr_scanned`); `engine._literature_entry` / `_content_quality` / `_item_content` keep the whole text on
  disk (`data/literature/.full_text/`) and label what it is.
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
  `no-new-privileges`), the non-root user choice checked by a write in the quest folder
  (`_guess_user` / `_resolve_user`, in `setup`), and the plain `[FI]` line for a run stopped by a cap (`_limit_note`).
- `core/experiment_deps.py` — what a quest environment is given before a run: requested packages minus the quest's
  own files, the selected skills' `pip_requires`, library skills on `PYTHONPATH`, one-at-a-time install fallback,
  skill names pip cannot install explained as the skill, the repair note for what could not be installed. Also the one
  pip-name / import-name table (`IMPORT_TO_PIP`, `pip_name`, `import_names`; `code_project.requirements_for` uses it
  too), the import scanner (`imported_modules`, `third_party`, `third_party_of`, `code_sources`, `skill_sources`,
  `local_module_names`), `env_packages` (asks the quest's Python what an installed package provides),
  `plan_installs` (the requested packages the scripts use, plus well-known unrequested imports; called by
  `Engine._node_execute`) and `warmup_modules` (the post-install test import).
- `core/code_project.py` — `refresh` keeps `code/` a runnable project (README, requirements.txt, `run.py`, and `study.json` for the trial contract; `run.py` repeats FI's trial runner for seed 0, one process per setting, stdlib only, working in `run_output/`; for a search for the best design `study.json` holds the optimisation block and `fi_search.py` is `core/optimise_search.py`, so `run.py` repeats FI's search, one process per design); `.fi/installed_deps.json` (written by `_node_execute`) is the requirements source; `attempt_records.script_hashes` skips the unedited generated files; called by `Engine._refresh_code_project` at the end of `implement` and the start of `analyze`; `record_change` makes one git commit + one `CHANGELOG.md` entry per change inside `code/` (`record_note`: an entry and its commit when nothing else changed, used by the improve loop) (`rerun_from.back_up` carries `code/.git` over a re-run; `script_hashes` ignore `.git` and CHANGELOG.md); `pin` gives the installed versions for requirements.txt; `verify` (called by `Engine._check_code_project` before `write`, once per code version) runs `run.py` in a clean venv and writes `needs/CODE_PROJECT_CHECK.json`, warning only; `unasked_conflicts` / `mark_asked` feed `Engine._ask_about_edited_project_files` (a `code_project` pause when `pauses.review` is `ask`); a file whose hash differs from `.fi/code_project.json` (a person's edit) is never overwritten.
- `Engine._save_run_data` (called in `_node_execute` only once the run is accepted: exit 0 and the run manifest not
  stopped, pending or repairing; again after the replicate seeds for the raw copy only) copies the data tables the run
  wrote in the quest folder (`_RUN_DATA_SUFFIXES`, each up to `_RUN_DATA_MAX_BYTES`, never the secret/setup names in
  `_RUN_DATA_SKIP`) and a two-script quest's `raw/` into `data/results/`; `data/results/` is not counted as the data a
  quest was given (`attempt_records`' data manifest, `Engine._gather_collected_text`), and `_node_replot_layout` reads it.
- `core/engine.py::_script_has_random_source` — whether a one-script quest's script (or a sibling module it imports)
  can draw a random number at all; none (and the seed ignored) makes it a deterministic study
  (`result_json_no_random_source`): the remaining replicates are skipped and the analysis is told to report one run,
  not an interval.
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
- `core/rerun_from.py` — `--resume <id> --from <step>`: one table per step (`STEPS` nodes, `REDOES` sentence, `OUTPUTS` files moved aside, `GROUPS` block for a map, `NEEDS_APPROVAL` for the steps up to the design, `LEADS_INTO` for seeding state before them); `Engine.rerun_steps` returns `step_info` dicts, `MAP_BLOCKS`, `NODES` (every graph node with its block, plain title, sentence, reads, writes and loop note; `tests/test_quest_map.py` pins it to the graph's nodes), `node_map` (adds status and clickability) and `map_payload` (blocks + finished + nodes) turn them into the quest map; `Engine.quest_map` builds it and `Engine.stopped_at` says where the quest is (the checkpoint's next nodes; finished only at the graph's end with no `pause.json` / `NEXT_STEP.md` / `quest_failed.md`, never from the summary file): the web `rerun-steps` API and `launch.py --resume <id> --from --json` return it, `web/static/quest_map.js` + `quest_map.css` draw it (`web/static/quest.html` `renderQuestMap`; the VS Code panel `vscode-frontier-insight/src/quest-map.ts` loads the same two files from `media/`, copied by `scripts/copy-quest-map.js`), `Engine._keep_design_history` restores `DESIGN_HISTORY.json` into the state before a pre-design restart, and `Engine.run(approved_by=...)` refuses those steps without a name. `OUTPUTS` of a step include those of every later step. A graph node with no step must be listed in the module docstring and in `tests/test_rerun_from_nodes.py::NOT_A_STEP`. `skills` picks the skills again from the current config (`Engine._repick_skills`, on the checkpoint just before the code); `checkpoint_before` finds the checkpoint before a step (used by `Engine.run`); `reached`, `REDOES` and `listing` give the steps a quest reached on the line it is on now (a checkpoint of a line left by an earlier `--from` does not count) (`Engine.rerun_steps` for `--from` with no step, the web `rerun-steps` API and `@fi /resume --from`); `back_up` moves that step's and later outputs to `.fi/previous/<time>/` (a move that fails half way puts back what it had moved). A restart before the design records the replaced protocol as a post-hoc amendment (`frozen_protocol.record_replacement`). The path a map draws comes from `Engine.rerun_no_simulation()` (config flags or what clarify put in the checkpoint); a node is clickable only once the quest reached it.
- `core/paper_patch.py`, `core/paper_trim.py` — a targeted revise (only the flagged passages) and a page-limit trim.

## Command-line tools and the web/VS Code surfaces

- `launch.py:_bootstrap_or_reraise`, `_relaunch` — the self-setup a missing dependency triggers at the top-level
  import (installs into an activated venv/conda env in place, silently relaunches into an already-complete
  `.venv/`, or does the full ask-and-install-and-relaunch dance); see `docs/capabilities-reference.md#getting-started`
  for the exact decision tree. Every `vscode-frontier-insight` spawn of `launch.py` opts out (`FI_SKIP_BOOTSTRAP=1`)
  in favor of its own `frontierInsight.pythonPath` contract.
- `core/critique.py`, `core/digest.py`, `core/portfolio.py`, `core/proposal.py`, `core/analyze_cli.py`,
  `core/summarizer.py`, `core/state_dump.py` — the one-shot CLI tools (`fi tools <name>`, see `launch.py:
  _TOOL_SUBCOMMANDS`); `core/proposal_seed.py` seeds an interview from a saved proposal.
- `web/server.py` — the FastAPI status server (quest list, log stream, evidence, trace, amendment banner, ...).
- `web/interview_routes.py`, `web/skills_routes.py`, `web/tools_routes.py` — the web UI's interview, skills and
  CLI-tools surfaces.
- `web/quest_launcher.py` — the subprocess pool for quests started from the web UI. Its children (and the quest page's
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
- `core/bridge_path.py`, `vscode-frontier-insight/src/bridge-path.ts` — the canonical persistent-bridge socket path,
  kept identical on both sides. A second VS Code window open at the same time binds `<path>-<pid>` instead
  (`persistent-bridge.ts` `listen`), and its `/update` / `/generate` terminals are handed that address.
- `scripts/wiki_sync_index.py` — rebuilds `wiki/index.md` from the pages (`[[slug|Title]]`, grouped by front-matter
  `type`); `--check` reports a stale index or a wiki link to a missing page. `tests/test_wiki_sync_index.py` and the
  `wiki-check` CI job (runs when `wiki/**` changes) keep the repo's wiki index current.

## Prompts (`agents/*.md`)

One `string.Template` file per node/persona; loaded once at engine init (`core/engine.py`). Grep `agents/` for a
topic before writing a new prompt — most nodes already have one, and a persona variant (`write_persona_*.md`,
`review_persona_*.md`) is usually the right way to add a new voice rather than a new node.
