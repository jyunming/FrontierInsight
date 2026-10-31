"""A read-only list of every check FI has: where its decision is made, where the fact it judges comes from, and the most
it can do. Nothing in the engine calls it. It exists so that the number of checks can be counted and may only go down:
a new check is added here in the same change that deletes another (``tests/test_checks_registry.py`` keeps each row
pointing at code that exists, and keeps it from drifting from ``core.static_checks.static_checks``).

``source``: ``own_record`` (FI's own ledger, records or seeds), ``measured`` (FI runs code to measure), ``model_text``
(reads text a model wrote), ``model_code`` (reads code a model wrote), ``model_judge`` (asks a model for a verdict).
``power`` is the strongest thing the check can do, at its default settings: ``stop`` (ends the quest or stops before the
run), ``repair`` (sends code, plan or paper back for a rewrite), ``strip`` (removes a result or sentence from the
paper), ``note`` (records or discloses only). Where a config key changes the power, the note names it.
"""

from __future__ import annotations

from dataclasses import dataclass

SOURCES = ("own_record", "measured", "model_text", "model_code", "model_judge")
POWERS = ("stop", "repair", "strip", "note")
STAGES = ("clarify", "ideate", "literature", "plan", "design", "implement", "execute", "execute_reflect", "improve",
          "analyze", "cross_check", "evidence_gate", "write", "claim_check", "citation_check", "review",
          "human_feedback", "several")


@dataclass(frozen=True)
class Check:
    name: str  # short snake_case id, unique
    module: str  # dotted module under core where the decision is made
    function: str  # function, or Class.method, in that module
    stage: str  # the graph node it runs in, or "several"
    source: str  # one of SOURCES
    power: str  # one of POWERS
    note: str  # one line: what it judges


CHECKS: tuple[Check, ...] = (
    Check('undefined_names', 'core.undefined_names', 'problems',
          'execute', 'model_code', 'stop',
          'A name the code uses that nothing defines or imports (symtable); sent back for repair up to 2 times, '
          "then recorded; configured 'stop' but a reading cannot end a quest."),
    Check('grid_keys', 'core.grid_keys', 'problems',
          'execute', 'model_code', 'stop',
          "Code reads a setting by a name the plan's grid/fixed settings/check cases do not have (cell.get('x', "
          'default)); repaired up to 2 times, then noted for the paper.'),
    Check('static_fixed_run_count', 'core.grid_keys', 'counts',
          'execute', 'model_code', 'repair',
          "A module-level constant naming a trial/run count that differs from the protocol's runs per setting "
          'in a run_trial study; sent to repair via the grid_keys gate.'),
    Check('typed_constants', 'core.typed_constants', 'find',
          'execute', 'model_code', 'stop',
          "Numbers the plan never gives typed into the model's functions (inside smooth calls, or 3+ of them); "
          "repaired, then limitations note; 'headline' mode still cannot stop since the source is a reading."),
    Check('independent_check', 'core.independent_check', 'tautologies',
          'execute', 'model_code', 'note',
          'A declared second_implementation check whose two compared sides reduce to the same expression '
          '(difference zero by construction); noted as an evidence gap.'),
    Check('static_gate_standing', 'core.static_checks', 'StaticChecksMixin._static_standing',
          'execute', 'model_code', 'note',
          'What a fault standing after repairs does: note and go on, or hand it to the cannot-repair record, '
          'which ends the quest only for a measured/agreed source (never for these readings).'),
    Check('static_repair_kept', 'core.static_checks', 'StaticChecksMixin._repair_static',
          'execute', 'model_code', 'repair',
          "A static-fault rewrite is kept only if it parses, keeps FI's marks, passes grid_keys.rejects and has "
          'fewer faults of that kind; otherwise the old code stays.'),
    Check('static_repair_substitution', 'core.grid_keys', 'rejects',
          'execute', 'model_code', 'repair',
          'Refuses a repair that replaced a constant the plan itself defines with a setting the study sweeps '
          '(grid_keys.substituted).'),
    Check('protocol_grid_values', 'core.protocol_check', 'check',
          'implement', 'model_code', 'repair',
          "Script's list/loop for a grid or in_each_run axis does not hold exactly the protocol's values; "
          'repaired up to engine.protocol_repair_attempts, stop only if engine.protocol_check=block and source '
          'may stop (it is a reading, so recorded).'),
    Check('protocol_runs_per_setting', 'core.protocol_check', 'check',
          'implement', 'model_code', 'repair',
          "Script fixes a count of runs (NUM_RUNS, n_trials...) that differs from the protocol's "
          'runs_per_setting; resample/MCMC counts and floors are skipped; repaired then recorded.'),
    Check('protocol_thresholds', 'core.protocol_check', 'check',
          'implement', 'model_code', 'repair',
          'A protocol threshold value appears nowhere in the scripts (outside the trial contract, where FI '
          'hands it in); repaired then recorded.'),
    Check('protocol_loop_literal_hint', 'core.protocol_check', 'loop_hints',
          'implement', 'model_code', 'note',
          "A difference resting only on a for-loop's literal values is told to the reader, never repaired or "
          'stopped.'),
    Check('protocol_analysis_loop_suspect', 'core.protocol_check', 'analysis_suspects',
          'implement', 'model_code', 'note',
          'Analysis loops over fewer grid values than the protocol; only logged here, settled after the run by '
          "FI's own trial count (run_manifest.confirmed)."),
    Check('protocol_rng_reuse', 'core.protocol_check', 'rng_notes',
          'implement', 'model_code', 'note',
          'A function builds a random generator from something that does not depend on its inputs (same stream '
          'reused); note in PROTOCOL_CHECK.json, never a repair or stop.'),
    Check('protocol_enforce_repair', 'core.engine', 'Engine._enforce_protocol',
          'implement', 'model_code', 'repair',
          'Holds the written script to the frozen protocol: repair loop, record in needs/PROTOCOL_CHECK.json, '
          'status stopped/warned by engine.protocol_check and stop_power.'),
    Check('protocol_drift_after_extension', 'core.engine', 'Engine._protocol_drift_not_asked_for',
          'several', 'model_code', 'note',
          "After a person's extension, script differences from the frozen protocol that the person did not "
          'request count as drift.'),
    Check('protocol_agent_code', 'core.agent_build', 'AgentBuildMixin._agent_protocol',
          'implement', 'model_code', 'repair',
          "Coding agent's scripts compared with the protocol; with engine.protocol_check=block the code goes "
          'back to the agent, with warn only recorded.'),
    Check('plan_topic_numbers_missing', 'core.protocol_check', 'plan_notes',
          'plan', 'model_text', 'note',
          "A number the topic sets that the plan's protocol does not contain (or no protocol block at all); "
          "said in the plan's Checks already made."),
    Check('plan_precision_vs_trials', 'core.protocol_check', 'precision_notes',
          'plan', 'own_record', 'note',
          'Trials the protocol plans against the trials its target precision (half-width) needs; arithmetic '
          'stated in the plan.'),
    Check('plan_grid_axis_too_few_values', 'core.protocol_check', 'grid_notes',
          'plan', 'model_text', 'note',
          'An axis with too few values for the claim the design makes about it (convergence, scaling, threshold '
          'wording).'),
    Check('plan_repeated_runs_of_deterministic', 'core.protocol_check', 'run_count_notes',
          'plan', 'model_text', 'note',
          'Protocol asks for repeated runs of a design whose description names nothing random.'),
    Check('plan_oracle_notes', 'core.protocol_check', 'oracle_notes',
          'plan', 'own_record', 'note',
          'What the plan says about the oracles (known-answer checks) its protocol declares.'),
    Check('plan_metric_notes', 'core.protocol_check', 'metric_notes',
          'plan', 'own_record', 'note',
          'What the plan says about the metrics the protocol declares (metric_spec).'),
    Check('plan_failure_notes', 'core.protocol_check', 'failure_notes',
          'plan', 'own_record', 'note',
          'What the plan says about a trial that fails when the protocol has runs to fail (run_manifest).'),
    Check('layout_missing_parts', 'core.code_layout', 'check',
          'execute', 'model_code', 'repair',
          'Research-tool layout missing: package, simulate.py not using it, no unit tests, no METHODS.md, '
          'analysis importing the package; repaired as a failed run under rigor_profile research, then recorded '
          '(a reading cannot stop).'),
    Check('layout_module_level_randomness', 'core.code_layout', 'module_level_randomness',
          'execute', 'model_code', 'repair',
          "Package module makes/seeds a random generator at import time, so FI's per-trial seed does not reach "
          'it; part of the layout check.'),
    Check('layout_unseeded_rng', 'core.engine', 'Engine._check_code_layout',
          'execute', 'model_code', 'repair',
          'Package builds a random generator without a seed (outside accepted fallback shapes); reported with '
          'the layout problems.'),
    Check('equation_labels', 'core.engine', 'Engine._check_equation_labels',
          'execute', 'model_code', 'repair',
          "A plan equation has no '# E<n>' label on a function (or outline-named function absent); repaired as "
          'failed run under research profile, otherwise logged.'),
    Check('layout_problems_standing', 'core.engine', 'Engine._code_problems_stop',
          'execute', 'model_code', 'repair',
          'Sends label/layout problems back to repair (exec_reflect_max_iterations), then records what stands '
          'as READ so it does not end the quest.'),
    Check('function_import_check', 'core.function_steps', 'FunctionFiller._check',
          'implement', 'measured', 'repair',
          'Each function is import-checked in the quest environment (parses, outline signature, imports); '
          'failures repaired alone, up to function_steps.REPAIRS.'),
    Check('function_environment_problem', 'core.function_steps', 'environment_problem',
          'implement', 'measured', 'repair',
          'Tells a missing package (installed, then re-checked, costing no repair) from a fault of the '
          "function's code."),
    Check('function_example_test', 'core.function_steps', 'FunctionFiller._example_test',
          'implement', 'measured', 'repair',
          "Runs the plan's worked example against the just-written function (equation or step); failing "
          'function repaired alone, kept failed/partial otherwise.'),
    Check('function_step_static', 'core.step_static', 'check',
          'implement', 'model_code', 'repair',
          "A method step's function that returns a number written into it, or never reads a parameter, fails "
          'and is repaired.'),
    Check('function_steps_partial', 'core.function_steps', 'FunctionFiller.run_all',
          'implement', 'measured', 'note',
          'Functions still failing after repairs end as partial (or fall back to writing the code whole); later '
          'equation tests and honest stop apply.'),
    Check('equation_test', 'core.engine', 'Engine._equation_gate',
          'execute', 'measured', 'repair',
          'One test per plan equation from its worked example, called in a separate process and judged by FI; '
          'failures repaired alone then whole-script repair as a failed run (off when engine.oracle_check=off).'),
    Check('equation_example_judged', 'core.equation_tests', 'example_rows',
          'plan', 'own_record', 'note',
          "FI computes the worked example's expected output with its calculator and flags examples that are "
          'unusable, untestable, or leave a generating equation unprotected.'),
    Check('equation_unprotected_core', 'core.equation_tests', 'untested_notes',
          'execute', 'own_record', 'note',
          'An equation that will not be tested before the run (untestable with no property check, no function '
          'found, cannot judge) is said plainly in the record and limitations.'),
    Check('equation_example_rederive', 'core.engine', 'Engine._rederive_examples',
          'execute', 'model_judge', 'repair',
          "When a failing test may be a fault of the plan's own example, the plan is asked once to work the "
          'example out again from its equation.'),
    Check('equation_example_parameters', 'core.equation_tests', 'parameter_problems',
          'implement', 'own_record', 'repair',
          "An equation whose named function does not take the example's inputs by name is found before the body "
          'is written.'),
    Check('code_review_requirements', 'core.engine', 'Engine._code_review_gate',
          'execute', 'model_judge', 'repair',
          "A second model reads the plan's required outputs against the code (implemented / missing / "
          'different); each finding goes through test-first repair, bounded by MAX_SEND_BACKS and reader-call '
          'budget.'),
    Check('code_review_logic', 'core.code_review', 'parse_logic',
          'execute', 'model_judge', 'repair',
          'Same reading checks logic against the plan (equation, normalisation, constant-reported, skipped '
          'step, check case); problems are repaired one at a time.'),
    Check('code_review_repair_scope', 'core.code_review', 'repair_problem',
          'execute', 'model_code', 'repair',
          'Refuses a repaired file that changes what the findings name or no longer prints RESULT_JSON.'),
    Check('code_review_plan_conflict', 'core.engine', 'Engine._route_plan_conflicts',
          'execute', 'model_judge', 'repair',
          "A finding saying the PLAN contradicts itself is sent to the plan's model, never to the code."),
    Check('code_review_finding_demonstrated', 'core.code_findings', 'shown_by_assertion',
          'execute', 'measured', 'note',
          'A finding counts as demonstrated only when its generated test fails by its own assertion about '
          'values; errors/no answer show nothing.'),
    Check('code_review_shown_fault_gate', 'core.engine', 'Engine._shown_fault_verdict',
          'evidence_gate', 'measured', 'stop',
          'Re-runs the test of each shown, still-open fault on the code that ran (twice if failing); a failing '
          'assertion on a headline quantity, or leaving out leaves no result, ends the quest.'),
    Check('code_review_leave_out_shown', 'core.result_checks', 'ResultChecksMixin._leave_out_shown_faults',
          'evidence_gate', 'measured', 'strip',
          'Quantities a failing test showed the code computes wrongly are left out of the results the paper may '
          'use, and the paper says so.'),
    Check('code_review_computed_differently', 'core.result_checks', 'ResultChecksMixin._code_reader_findings',
          'execute', 'model_judge', 'strip',
          'A quantity the second model found computed differently from the plan (not shown harmless, function '
          'reached) is taken out of the results.'),
    Check('code_review_headline_constant', 'core.code_review', 'resolve_logic',
          'evidence_gate', 'measured', 'stop',
          "High-severity reader finding that the main quantity is constant, confirmed by the run's own results "
          'holding one value everywhere, ends the quest.'),
    Check('code_review_headline_missing', 'core.code_review', 'resolve_headline',
          'evidence_gate', 'model_judge', 'note',
          'Reader says the main quantity is missing from the code and no result is named like it: the paper '
          'says it was not computed; quest goes on (reading only).'),
    Check('code_review_open_findings', 'core.code_review', 'write_note',
          'write', 'model_judge', 'note',
          "Findings still open after repairs, and quantities not computed, are said in the paper's limitations "
          '(also {{not_done:}} items).'),
    Check('results_report_failure', 'core.engine', 'Engine._reports_no_quantity',
          'evidence_gate', 'own_record', 'stop',
          "Run result says it failed (errors/status) and holds a number for none of the protocol's metrics: "
          'counts as no result (insufficient, stuck).'),
    Check('code_unreached_finding', 'core.code_reach', 'never_run_in',
          'evidence_gate', 'model_code', 'note',
          'A finding about a function the run plainly never calls changes no number: kept in the record, not '
          'stopped on, not left out, not in the paper.'),
    Check('second_version_agreement', 'core.engine', 'Engine._second_version_gate',
          'implement', 'measured', 'repair',
          'A second independently written version of key functions is compared on FI-built inputs; a judge says '
          'which is wrong, main repaired only if still passing its tests; unresolved/unconfirmed go to '
          'limitations, never stops.'),
    Check('second_version_judge', 'core.second_impl', 'judge_rows',
          'implement', 'model_judge', 'repair',
          'Verdict on rows where the two versions differ (main_wrong / reference_wrong / card_ambiguous) within '
          'MAX_ROUNDS.'),
    Check('headline_depends_on_functions', 'core.second_impl', 'dependence',
          'execute', 'measured', 'note',
          'Perturbs a tested function and checks the headline results change; a function the headline results '
          'do not use becomes an evidence gap.'),
    Check('hidden_setting_check', 'core.hidden_check', 'run',
          'execute', 'measured', 'note',
          'After the run FI re-runs invariant/symmetry/second-implementation checks at a setting no prompt '
          'named (research profile); a failure is an evidence gap, never a stop.'),
    Check('deps_split_unknown_packages', 'core.experiment_deps', 'split_deps',
          'execute', 'model_code', 'note',
          'Packages the model listed that pip cannot find (own helper files, skill names, nonexistent) are left '
          'out of the install line with a reason, so one bad name costs only itself.'),
    Check('deps_plan_installs', 'core.experiment_deps', 'plan_installs',
          'execute', 'model_code', 'repair',
          'Installs a well-known package a script imports but the list lacks; skips a listed package no script '
          'imports.'),
    Check('script_repair_reply_usable', 'core.script_reply', 'resolve',
          'execute_reflect', 'model_code', 'repair',
          'Builds the script a repair reply makes (whole file or spliced functions); returns nothing when the '
          'reply makes no script.'),
    Check('code_built_from_delta', 'core.code_delta', 'compare',
          'implement', 'own_record', 'repair',
          "From FI's record of what the scripts were built from: unchanged plan parts keep the scripts, changed "
          'parts get only the affected functions rewritten, no record means write whole.'),
    Check('crash_kind', 'core.crash_kind', 'classify',
          'several', 'measured', 'repair',
          'Sorts a quest-ending exception into transient (engine retries), setup, fi, unknown, so the card '
          'names one action; transient failures are retried before stopping.'),
    Check('agent_code_missing', 'core.agent_build', 'AgentBuildMixin._agent_judge',
          'implement', 'own_record', 'repair',
          'No analysis script in code/ after an agent round is a finding for the next round.'),
    Check('agent_two_scripts_contract', 'core.agent_build', 'AgentBuildMixin._agent_judge',
          'implement', 'model_code', 'repair',
          'Split study without run_cell/run_trial in simulate.py is sent back to the agent, not run.'),
    Check('agent_shown_findings', 'core.agent_build', 'AgentBuildMixin._shown_findings',
          'implement', 'measured', 'repair',
          "Each fault another model found that FI's test shows is one check; still flagged goes to the agent "
          'with its test.'),
    Check('agent_round_progress', 'core.agent_build', 'AgentBuildMixin._agent_build',
          'implement', 'own_record', 'stop',
          'Honest stop when agent rounds make no progress (compare_rounds), fix rounds or minutes are used up '
          'and checks still fail.'),
    Check('code_checks_verdicts', 'core.code_checks', 'check_code',
          'implement', 'measured', 'note',
          'Judges code/ with the gate-pass checks (typed results, grid keys, small run, equation tests, known '
          'answers, unread settings, figure data) as verdicts; decides nothing itself.'),
    Check('unread_compared_setting', 'core.run_checks', 'RunChecksMixin._unread_compared_setting',
          'execute', 'model_code', 'repair',
          'A swept setting the study compares that the simulation never reads goes back for repair (bounded); '
          'afterwards the grid is kept as written.'),
    Check('unread_setting_note', 'core.run_checks', 'RunChecksMixin._check_unread_settings',
          'execute', 'model_code', 'note',
          "FI's reading finds no use of a grid setting: one line in plan.md and run.log; every setting is kept "
          'and run.'),
    Check('clarify_topic_shape_mismatch', 'core.engine', 'Engine._log_topic_shape_mismatch',
          'clarify', 'model_judge', 'note',
          'Clarify says the topic is a review/case study/opinion but the engine resolved to run an experiment; '
          'only a run.log warning, design and write keep the experiment minimal.'),
    Check('ideate_reflect_swap', 'core.engine', 'Engine._node_ideate',
          'ideate', 'model_judge', 'repair',
          'A reflection call may swap the chosen idea for another brainstormed one or refine its rationale '
          '(engine.ideate_reflect; skipped when the tournament ran).'),
    Check('ideate_tournament_pick', 'core.engine', 'Engine._run_ideate_tournament',
          'ideate', 'model_judge', 'repair',
          'Pairwise model comparisons of all ideas pick the winner that replaces the first choice '
          '(engine.ideate_tournament); inconclusive falls back to the earliest idea.'),
    Check('literature_novelty_evidence', 'core.novelty', 'check',
          'plan', 'own_record', 'strip',
          "The model's note on whether the retrieved sources already answer the question is checked against FI's "
          'own list of sources: an evidence item whose source is not in the list, or whose quoted sentence is not in '
          "that source's text, is removed and counted in run.log. The note is only disclosed (knowledge.novelty_check)."),
    Check('scope_cards_literature_refs', 'core.scope_cards', 'normalize',
          'plan', 'own_record', 'strip',
          "Study cards from the model are checked against FI's own list of retrieved sources; a card's "
          'literature line not in that list is removed and said.'),
    Check('scope_cards_distinct_enough', 'core.scope_cards', 'distinct_enough',
          'plan', 'model_text', 'repair',
          'The scope node asks the model once more when fewer than the needed number of cards differ in study '
          'type or main results (pauses.scope).'),
    Check('scope_card_not_covered_by_plan', 'core.run_checks', 'RunChecksMixin._scope_gap_note',
          'write', 'model_text', 'note',
          'A result or setting of the study card the person chose that the frozen plan has no name for is told '
          'to the writer for the limitations (name reading via scope_cards.gaps); never stops.'),
    Check('topic_items_name_match', 'core.topic_requirements', 'match',
          'plan', 'model_text', 'note',
          "Looks for each factor, quantity, formula and search the topic text asks for among the plan's grid, "
          'metrics and equations by name; unresolved items go to the judge.'),
    Check('topic_items_judge', 'core.engine', 'Engine._judge_topic_items',
          'plan', 'model_judge', 'note',
          'One bounded model call (topic_cover, node oracle_review.topic) judges the topic items a name match '
          'cannot settle; an unjudged item is recorded as unjudged.'),
    Check('topic_unmet_asked_back', 'core.engine', 'Engine._ask_plan_for_the_topic',
          'plan', 'model_judge', 'repair',
          "What the topic asks for and the plan lacks goes back to the plan's model in one bounded request; "
          'only the asked parts are taken in.'),
    Check('topic_unmet_disclosed', 'core.engine', 'Engine._hold_the_design_to_the_topic',
          'design', 'model_judge', 'note',
          'The design that will run is held to the topic again; what is still missing no longer stops the quest '
          "(earlier stops mostly were false), it is logged and the paper says FI's check did not find it."),
    Check('topic_asks_for_search', 'core.engine', 'Engine._asks_for_best_design',
          'plan', 'model_judge', 'repair',
          "Topic reader's structured 'searches' (or the person's answer) make the plan a find_best_design "
          'study; a draft calling itself a measurement gets study_type forced to find_best_design.'),
    Check('topic_code_files_kept', 'core.engine', 'Engine._plan_code_layout',
          'plan', 'model_judge', 'note',
          "If the topic asks for N code files (reader's structured answer via topic_requirements.code_files) "
          "and FI keeps another layout, the paper must name it as not done; FI's code checks stay on."),
    Check('plan_design_block_unreadable', 'core.engine', 'Engine._usable_design_block',
          'plan', 'model_text', 'repair',
          'A rewritten plan whose design block is not valid YAML is repaired where no key or value changes, '
          'else the model is asked once for the block alone.'),
    Check('plan_md_unusable_pause', 'core.engine', 'Engine._pause_for_plan',
          'plan', 'model_text', 'stop',
          "A plan.md that cannot be read as a design stops the quest (pause) with the parser's reason until it "
          'is fixed; the plan stop for reading is pauses.plan: ask.'),
    Check('plan_protocol_part_unreadable', 'core.plan', 'normalize_protocol',
          'plan', 'model_text', 'repair',
          'Strict reading of the protocol block; a part that cannot be checked is left out with a note '
          "(repair_protocol) and the plan's model is asked for it in a bounded request."),
    Check('plan_model_block_unreadable', 'core.plan', 'normalize_model',
          'plan', 'model_text', 'repair',
          "The plan's model block (equations, assumptions) is read strictly; an equation that cannot be read is "
          'left out and noted (repair_model).'),
    Check('plan_sub_studies_unreadable', 'core.plan', 'readable_sub_studies',
          'plan', 'model_text', 'repair',
          'A sub-study entry that varies a setting the grid lacks or holds an invalid value is dropped on its '
          'own with its reason, not the whole list.'),
    Check('plan_rewrite_lost_settings', 'core.plan', 'lost_settings',
          'design', 'own_record', 'repair',
          'A plan rewrite that drops a setting the plan compared before is sent back to the model with the lost '
          'setting named.'),
    Check('plan_scope_refusal', 'core.plan_scope', 'translate',
          'design', 'own_record', 'repair',
          "A model reply to a plan request that names parts outside the request's scope is refused with a "
          'reason; the rest of the plan stays byte for byte.'),
    Check('plan_grid_unreadable_stop', 'core.engine', 'Engine._stop_if_no_settings_to_vary',
          'design', 'model_text', 'stop',
          "A plan that tried to list settings to vary but whose grid cannot be read, even after the plan's "
          'model was asked again, stops the quest before anything runs rather than running one setting as a '
          'sweep.'),
    Check('plan_grid_too_large_asked_back', 'core.engine', 'Engine._ask_plan_for_the_grid',
          'design', 'own_record', 'repair',
          "A grid with more settings than FI lists (plan's own arithmetic) is sent back to the plan once to "
          'make smaller; if not, settings are planned lazily and run up to the time limit.'),
    Check('plan_grid_too_large_note', 'core.engine', 'Engine._grid_too_large_stop',
          'design', 'own_record', 'note',
          'A grid larger than FI lists no longer stops the quest (always returns None); run.log says FI will '
          "plan the reachable settings in the plan's order."),
    Check('plan_search_block_unreadable', 'core.engine', 'Engine._ask_plan_for_the_search',
          'design', 'model_text', 'repair',
          'A find_best_design plan whose optimisation block is missing or unreadable (or whose coarse scan '
          "cannot run) is asked of the plan's model with the reader's exact reason, bounded."),
    Check('search_missing_parts', 'core.optimisation_plan', 'missing_parts',
          'design', 'model_text', 'repair',
          'What a search for the best design lacks before it could start (objective, design variables, '
          'baseline, budget) is listed in plain words and asked back.'),
    Check('search_scan_cannot_run', 'core.optimisation_plan', 'scan_problem',
          'design', 'own_record', 'repair',
          "The coarse scan of a search is checked by the plan's own arithmetic (too many settings, or too long "
          'at the time one run may take) and sent back to the plan to rewrite.'),
    Check('search_cannot_start_stop', 'core.engine', 'Engine._search_cannot_start_stop',
          'design', 'model_text', 'stop',
          'The honest stop (no paper) when the plan is a search for the best design and its search block still '
          'cannot be made after asking; a search is never run as a plain sweep. A reading-based stop, so see '
          'stop_power.'),
    Check('search_setup_cannot_start', 'core.engine', 'Engine._stop_if_the_search_cannot_start',
          'design', 'own_record', 'stop',
          'Pauses the quest when its setup stops a search (execution.background_jobs on, or '
          'execution.split_analysis off); the person is told which setting to change.'),
    Check('study_type_matches_ask', 'core.engine', 'Engine._settle_study_type',
          'plan', 'model_text', 'repair',
          "Makes the plan's study_type say what was asked (topic or answer): a search hidden as a sweep is set "
          'to find_best_design, and a contrary draft is noted.'),
    Check('criteria_unusable_dropped', 'core.criteria', 'repair',
          'plan', 'model_text', 'repair',
          'A drafted criterion FI cannot compute is left out with a sentence each, and the others kept; at most '
          'five are kept.'),
    Check('plan_no_countable_criterion', 'core.engine', 'Engine._node_plan',
          'plan', 'own_record', 'note',
          'A plan with no criterion FI can measure itself is given criteria made from its own known-answer '
          'checks, else one literature search and one model question; if none, the run goes on with no '
          'criterion (pauses.plan: ask stops for the person).'),
    Check('criteria_evaluated_per_run', 'core.criteria', 'evaluate',
          'execute', 'measured', 'note',
          "Each criterion is computed from FI's measured verdicts and trial record after a run and appended to "
          'criteria_history.jsonl; records only and never stops.'),
    Check('plan_sources_missing', 'core.oracle_check', 'unsourced',
          'plan', 'model_text', 'note',
          'A check whose expected value has no source a reader can check (empty reference, or a source this '
          'quest did not retrieve) is warned; under research the quest goes on by itself with the check marked '
          'source not confirmed (core/accepted_checks go_on_by_itself).'),
    Check('plan_equation_source_problem', 'core.oracle_check', 'equation_problem',
          'plan', 'model_text', 'note',
          "An equation of the plan's model with no source a reader can check is said in the plan's Checks "
          'already made.'),
    Check('plan_model_leaves_out', 'core.oracle_check', 'model_missing',
          'plan', 'model_text', 'note',
          'What the model behind the numbers leaves out (what produces the numbers, assumptions, equations) is '
          'listed in the plan; going on is recorded in the evidence.'),
    Check('plan_check_kind_unknown', 'core.oracle_check', 'model_notes',
          'plan', 'model_text', 'note',
          'A check whose kind is none of the six, an equation with no role or no source: said in the plan '
          'notes.'),
    Check('oracle_forms_numeric_form', 'core.oracle_forms', 'enforce',
          'plan', 'model_text', 'repair',
          'Each known-answer check is held to the numeric form of its kind; changes that cannot alter a verdict '
          'are made by FI, the rest go back to the plan in one request.'),
    Check('oracle_expected_precision', 'core.oracle_forms', 'precision_gaps',
          'plan', 'own_record', 'repair',
          'A check whose tolerance is smaller than half a unit of the last digit of its expected value as typed '
          'is asked back once, then the tolerance is widened to that precision.'),
    Check('oracle_expected_formula', 'core.oracle_forms', 'formula_findings',
          'plan', 'own_record', 'repair',
          "FI computes each check's expected value from the plan's expected_formula itself; "
          'missing/unusable/different formulas go back to the plan and a disagreeing one is corrected to the '
          "formula's value."),
    Check('oracle_stated_arithmetic', 'core.oracle_forms', 'stated_arithmetic',
          'plan', 'own_record', 'repair',
          'Sums the plan writes in a reference or derivation (expr = number) that FI can compute and that come '
          'out otherwise are reported to the plan.'),
    Check('oracle_second_reader', 'core.oracle_review', 'findings',
          'plan', 'model_judge', 'repair',
          "One call (node oracle_review) has a second model judge the plan's known-answer checks; its findings "
          'go in one request to the plan but are applied only when right; errors never stop.'),
    Check('oracle_equation_reader', 'core.oracle_review', 'equation_findings',
          'plan', 'model_judge', 'repair',
          "Equations of the plan's model that another model than the planner says are not the standard "
          "definition go into the one request to the plan; the planner's own reading does not count."),
    Check('oracle_isolation_reader', 'core.oracle_review', 'isolation_findings',
          'plan', 'model_judge', 'repair',
          'Checks whose case another model says does not isolate what the measure claims go back to the plan; '
          'only counted when the reader is another model.'),
    Check('oracle_review_never_looser', 'core.oracle_review', 'never_looser',
          'plan', 'own_record', 'repair',
          'After a rewrite of checks, a tolerance made larger or a mode changed is put back, and dropped '
          'equations are restored (keep_every_equation).'),
    Check('oracle_independence_gap', 'core.oracle_review', 'independence_gaps',
          'evidence_gate', 'own_record', 'note',
          'Under research: checks the second model did not read, or read as the same model as the planner, are '
          'an evidence gap that keeps the result below independently validated.'),
    Check('oracle_dry_run_mismatch', 'core.oracle_forms', 'mismatch',
          'implement', 'measured', 'repair',
          "FI's first test run of the checks before freezing: a number whose size says plan and simulation mean "
          'different things by a check sends one targeted request to the plan (once per quest).'),
    Check('oracle_fitted_to_test_run', 'core.engine', 'Engine._fitted_to_test_run',
          'implement', 'measured', 'note',
          'A check whose expected value or tolerance the plan changed after the test run so it passes on the '
          "test run's own numbers is recorded fitted and counts as unconfirmed."),
    Check('oracle_checks_not_judgeable', 'core.oracle_check', 'unjudgeable',
          'implement', 'own_record', 'repair',
          'Declared checks with no numeric expected and tolerance cannot be judged; the plan is asked for the '
          'numbers (up to engine.oracle_repair_attempts).'),
    Check('oracle_run_problems', 'core.oracle_check', 'problems',
          'implement', 'measured', 'repair',
          'FI runs the checks (the case itself under the trial contract) and judges each against expected and '
          'tolerance; missing or failing checks send the script back for repair (engine.oracle_check: '
          'block|warn|off).'),
    Check('oracle_trials_verdict', 'core.mc_error', 'decide',
          'implement', 'measured', 'repair',
          'A random check is judged on three standard errors of its trial mean against the tolerance: pass, '
          'fail, or undecided (set aside, goes on unconfirmed).'),
    Check('oracle_failed_trial_share', 'core.mc_error', 'failed_share_verdict',
          'implement', 'measured', 'repair',
          "More than a share of a check's trials failing is a fail; any failed or non-finite trial makes a pass "
          'undecided.'),
    Check('oracle_per_trial_reading_unsaid', 'core.mc_error', 'unsaid_reading',
          'design', 'model_text', 'repair',
          "Every check run on random trials is asked of the plan's model once, check by check, whether it holds in "
          'every trial or on average (per_trial); a reading volunteered while writing the plan is not an answer.'),
    Check('oracle_duplicate_names', 'core.oracle_check', 'duplicate_names',
          'implement', 'own_record', 'note',
          'Two checks (or two reported values) sharing a name are warned: a value is matched by name.'),
    Check('oracle_loose_tolerance', 'core.oracle_check', 'loose_tolerance',
          'implement', 'own_record', 'note',
          'A check claiming an order of accuracy whose tolerance could not tell that method from first order is '
          'warned.'),
    Check('oracle_triage_recompute', 'core.oracle_triage', 'recompute_verdict',
          'implement', 'model_judge', 'repair',
          'For a failing check another model, never shown the measurement, works the expected value out again; '
          'a disputed value becomes a proposal, a corrected value only with two independent workings agreeing.'),
    Check('oracle_triage_plan_slip', 'core.oracle_triage', 'plan_slip',
          'implement', 'own_record', 'repair',
          "FI computes the arithmetic written in the check's derivation; a slip sets the check aside from "
          'script repairs and offers its value as a proposal (corrected only with a second source).'),
    Check('oracle_triage_formula_agrees', 'core.oracle_triage', 'formula_value',
          'implement', 'own_record', 'note',
          "A failing check whose expected value FI computed itself from the plan's formula and found right "
          'points to the simulation; no model is asked.'),
    Check('oracle_triage_step_size', 'core.oracle_triage', 'step_verdict',
          'implement', 'measured', 'note',
          "A deterministic failing check is run at half and quarter step; a tolerance tighter than the method's "
          'own error sets the check aside from repairs and unconfirmed.'),
    Check('oracle_triage_seed_noise', 'core.oracle_triage', 'seeds_verdict',
          'implement', 'measured', 'note',
          'A random failing check is run with extra seeds; trial noise larger than the tolerance sets it aside '
          'from repairs and unconfirmed.'),
    Check('oracle_triage_multiple_of', 'core.oracle_triage', 'multiple_of',
          'implement', 'own_record', 'repair',
          'A measured value that is a simple multiple (half, twice) of a value FI worked out adds a hint into '
          'the script repair request.'),
    Check('oracle_triage_same_exception', 'core.oracle_triage', 'same_exception',
          'implement', 'own_record', 'note',
          'The same exception after a repair changed the script ends further repairs of the checks run.'),
    Check('oracle_challenge_exchange', 'core.oracle_challenge', 'settle',
          'implement', 'model_judge', 'repair',
          "The code's objection to the plan's working is answered by the plan (concede/stand) and a judge; FI "
          'computes every formula and corrects the expected value only when both agree.'),
    Check('oracle_numeric_referee', 'core.oracle_challenge', 'settle_referee',
          'implement', 'measured', 'repair',
          'A failing check is refereed by a number FI computes numerically from the definitions (a script that '
          "never saw the plan's working or the measurement); decides which side to fix."),
    Check('oracle_expected_value_corrected', 'core.engine', 'Engine._correct_expected_values',
          'implement', 'model_judge', 'repair',
          "Plan's expected value of a failing check is rewritten only when two independent workings agree, once "
          'per check, before freeze; tolerance and measured value never used.'),
    Check('oracle_disputed_guard', 'core.engine', 'Engine._oracle_gate',
          'implement', 'measured', 'repair',
          'A script repair that makes a disputed check pass is put back (repair bent to a number the repair '
          'itself called wrong).'),
    Check('oracle_go_on_unconfirmed', 'core.accepted_checks', 'offer',
          'implement', 'own_record', 'note',
          'A check still failing after repairs lets the quest go on by itself (oracle_check: block), marked '
          'source/result unconfirmed and bound to the check and code version; the result counts as exploratory.'),
    Check('oracle_added_after_read', 'core.engine', 'Engine._hold_added_oracles',
          'implement', 'own_record', 'note',
          'Under pauses.plan: ask, checks the engine added or changed after the person read the plan are said '
          'once (no stop); the freeze record says nobody approved them.'),
    Check('oracle_card_leaning', 'core.oracle_card', 'leaning',
          'implement', 'own_record', 'note',
          "From FI's own look at failing checks it says which side most likely is wrong (checks or script) in "
          'plain words for the console and record.'),
    Check('oracle_engine_passed_independent', 'core.oracle_check', 'engine_passed',
          'evidence_gate', 'measured', 'note',
          'A check counts as independent evidence only when FI measured it itself and it passed; '
          'script-reported values do not count.'),
    Check('oracle_reader_flagged', 'core.oracle_check', 'with_reader_flags',
          'evidence_gate', 'model_judge', 'note',
          "A check the second model's reading of the code flags is marked flagged and not counted as passed."),
    Check('oracle_partial_pass', 'core.oracle_check', 'partial_checks',
          'evidence_gate', 'measured', 'note',
          'A rule check that passed every trial judged but not all planned (time ran out) is a pass for stop '
          'and repair but reported as not fully checked.'),
    Check('oracle_source_gaps_evidence', 'core.engine', 'Engine._oracle_source_gaps',
          'evidence_gate', 'model_text', 'note',
          'Checks whose expected value has no checkable source (judged against sources as frozen) are evidence '
          'gaps, also marked when a person or FI went on without one.'),
    Check('oracle_labels_evidence_gap', 'core.engine', 'Engine._equation_label_gaps',
          'evidence_gate', 'model_code', 'note',
          'Unlabelled generates equations are an evidence gap.'),
    Check('numeric_oracle_trivial_reference', 'core.numeric_oracle', 'trivial_reference_findings',
          'review', 'model_code', 'note',
          'A reference-named quantity in the results exactly 0 at every sweep setting (a root finder returning '
          'a trivial root) is an advisory warning shown to the human reviewer.'),
    Check('goal_coverage_asked_numbers', 'core.goal_coverage', 'notes',
          'review', 'model_code', 'note',
          "Numbers the topic sets as parameters (braced sets, run counts) are looked for in the script's "
          'literals, generated ranges and result keys; absent ones are advisory notes.'),
    Check('goal_coverage_figure_count', 'core.goal_coverage', 'asked_figure_count',
          'review', 'model_text', 'note',
          'Number of figures the topic asks for (exact/at least/at most) is compared with the figures drawn; '
          'advisory.'),
    Check('goal_unplaced_quantities', 'core.goal_coverage', 'unplaced_answers',
          'review', 'own_record', 'repair',
          'A quantity the topic asks for and FI holds a value for but the paper never places is forced into '
          'must_flag_hits, so the review sends the paper back to write.'),
    Check('plan_ledger_problems', 'core.engine', 'Engine._ledger_problems',
          'design', 'model_text', 'repair',
          'Unusable simplifications, metric definitions or step cards (and a definition quote not in the '
          "sources FI found) are asked of the plan's model once per round; never a stop."),
    Check('plan_ledger_uncovered', 'core.plan_ledger', 'uncovered',
          'design', 'model_text', 'repair',
          'Topic items the plan neither covers nor names in its simplifications are added to the ledger '
          'request.'),
    Check('plan_definition_quote_unfound', 'core.plan_ledger', 'unfound_quotes',
          'design', 'own_record', 'repair',
          "A metric definition's quoted source text not found in the retrieved passages is a problem; no "
          'passages means nothing judged.'),
    Check('plan_step_cards', 'core.plan_steps', 'parse',
          'design', 'model_text', 'repair',
          'Step cards (non-equation steps of the model) that are incomplete or whose worked example cannot be '
          'used are asked back.'),
    Check('redesign_ledger_note', 'core.engine', 'Engine._check_the_ledger_of_a_redesign',
          'design', 'model_text', 'note',
          "A redesign's simplifications, definitions and step cards get the same look; problems are said in "
          'run.log and passed to the writer, nothing stops.'),
    Check('plan_settings_changed', 'core.plan_settings', 'check',
          'several', 'own_record', 'stop',
          'On every start, settings that decide how strictly the quest is checked (check modes, reviewers, '
          'rigor profile) are compared with those approved on the confirm screen; a difference stops before '
          'anything runs (fix with --update).'),
    Check('plan_settings_record_missing', 'core.engine', 'Engine.run',
          'several', 'own_record', 'stop',
          'Stops when the approved-settings record is gone, was edited after approval (hash differs from the '
          'trace), or the trace no longer verifies.'),
    Check('review_models_stop', 'core.engine', 'Engine._review_models_stop',
          'several', 'own_record', 'stop',
          'Under research, stops before anything runs when every review-panel persona is on the same model, '
          'unless one_model_review is set.'),
    Check('model_change_disclosed', 'core.plan_settings', 'model_changes',
          'several', 'own_record', 'note',
          'A changed provider/model is never a stop: it is taken, recorded in the trace and said, and the paper '
          'discloses more than one model.'),
    Check('frozen_protocol_tamper_stop', 'core.engine', 'Engine._resolved_frozen',
          'design', 'own_record', 'stop',
          'A frozen protocol file whose hash no longer matches its content stops the quest under every profile '
          'until a person approves restoring the saved version.'),
    Check('frozen_protocol_amendment_stop', 'core.engine', 'Engine._hold_design_to_frozen',
          'design', 'own_record', 'stop',
          "A redesign after the freeze that asks for a different protocol stops for a person's approval; "
          'resumed without approval the frozen protocol stands; an unrequested change is silently put back.'),
    Check('frozen_amendment_declined', 'core.engine', 'Engine._settle_pending_amendment',
          'design', 'own_record', 'note',
          'Resuming past an amendment request without approval keeps the frozen protocol and records the '
          'declined request.'),
    Check('frozen_amend_by_fi', 'core.engine', 'Engine._amend_by_fi',
          'design', 'own_record', 'note',
          "FI's own rewrites (flat objective, plan made smaller to fit the time) are amendments FI approves "
          'itself; results-seen ones archive the earlier run and the paper discloses it.'),
    Check('frozen_audit_change_lost', 'core.audit_open_items', 'check_against',
          'design', 'own_record', 'note',
          'After the freeze, an audit change reported as made that the frozen protocol does not hold becomes an '
          'open item for the code and the paper.'),
    Check('plan_diverges_from_frozen', 'core.engine', 'Engine._warn_if_plan_diverges',
          'several', 'own_record', 'note',
          'plan.md edited after the freeze differs from the frozen protocol; the frozen one runs, only a '
          'warning.'),
    Check('design_audit', 'core.engine', 'Engine._audit_design',
          'design', 'model_judge', 'repair',
          'Second-pass methodology audit of the drafted design; the changes its objections name are applied '
          '(not those that change what results are judged by); a same-model audit is recorded as self-review.'),
    Check('design_audit_could_not_judge', 'core.engine', 'Engine._stop_once_for_check',
          'design', 'model_judge', 'stop',
          'Under research, a design audit that failed or was no review stops the quest once (resume retries '
          'once, then recorded as a gap); other profiles go on.'),
    Check('design_audit_repair_schema', 'core.engine', 'Engine._hold_audit_repairs_to_the_schema',
          'design', 'own_record', 'repair',
          "An audit change FI's own reading of the plan would refuse is not applied; the audit is asked once to "
          'correct that part.'),
    Check('design_audit_unnamed_change', 'core.design_parts', 'amended_from_audit',
          'design', 'own_record', 'repair',
          "Only parts named by an objection's changes are taken from the audit; a changed part no objection "
          'names is refused.'),
    Check('design_audit_may_not_change_judged', 'core.engine', 'Engine._keep_what_the_results_are_judged_by',
          'design', 'own_record', 'repair',
          'An audit may change how the study is run, never what results are judged by (checks, thresholds, '
          'metrics, precision, model): those keys are put back.'),
    Check('design_audit_open_items', 'core.audit_open_items', 'split',
          'design', 'own_record', 'note',
          'Audit proposals the design that runs does not hold become open items in QUALITY.md, passed to the '
          'code and told to the writer.'),
    Check('redesign_part_scope', 'core.design_parts', 'merge_parts',
          'design', 'own_record', 'repair',
          'A redesign may change only the parts its reason concerns (after a missed precision target only those '
          'sizing the study); other parts are kept as they were.'),
    Check('design_hypothesis_kept_after_empty_run', 'core.engine', 'Engine._keep_hypothesis_after_an_empty_run',
          'design', 'own_record', 'repair',
          "A redesign's new hypothesis is not taken when the run before it produced no results; the hypothesis "
          'stays.'),
    Check('design_unreadable_stop', 'core.engine', 'Engine._stop_for_unreadable_design',
          'design', 'model_text', 'stop',
          'A design answer that cannot be read as a design even after asking once more stops the quest; an '
          'empty design is never run.'),
    Check('review_design_objection_kept', 'core.engine', 'Engine._note_design_objections',
          'review', 'model_judge', 'note',
          'A reviewer objection to the design that no check of FI confirms is kept in QUALITY.md and never '
          'changes the design (repeated ones not counted again).'),
    Check('run_trial_aggregates', 'core.run_estimate', 'pilot_problems',
          'execute', 'model_code', 'repair',
          'A run_trial in simulate.py that reaches run_cell (read in the code, confirmed by two seeds giving '
          'the same numbers) is sent back to the code step before timing.'),
    Check('pilot_metric_undefined', 'core.run_estimate', 'undefined_metrics',
          'execute', 'measured', 'stop',
          'A protocol metric that is not a number in any pilot trial is sent back for repair '
          '(engine.exec_reflect_max_iterations), then ends in the honest stop.'),
    Check('pilot_metric_constant', 'core.run_estimate', 'pilot_problems',
          'execute', 'model_code', 'stop',
          'A mean metric exactly constant in all pilot trials while others move, and built from no setting per '
          'the code, is sent back for repair then the honest stop.'),
    Check('pilot_proportions_stuck', 'core.run_estimate', 'pilot_problems',
          'execute', 'measured', 'stop',
          'Two or more proportions all at the same bound (0 or 1) in every pilot trial, with code not using '
          'settings, are sent back then stop.'),
    Check('pilot_flat_sweep', 'core.run_estimate', 'pilot_problems',
          'execute', 'model_code', 'stop',
          'A sweep that gave the same output at every setting (flat_output) in the pilot trials is sent back '
          'then stop.'),
    Check('run_too_long_stop', 'core.run_checks', 'RunChecksMixin._stop_too_long',
          'execute', 'measured', 'stop',
          'Runs timed settings; when the study cannot fit execution.timeout_s even with the fewest runs and not '
          'one setting fits, stops with no paper (the evidence gate may ask the plan once for a study that '
          'fits).'),
    Check('run_lowered_runs', 'core.run_checks', 'RunChecksMixin._lower_the_runs',
          'execute', 'measured', 'repair',
          'A study too long is made smaller by FI lowering runs per setting (never settings), written to '
          "plan.md; below the plan's minimum the paper says part of the study is not fully done."),
    Check('run_cut_to_time_limit', 'core.run_checks', 'RunChecksMixin._run_up_to_the_limit',
          'execute', 'measured', 'note',
          "A grid too long is run in the plan's order up to the time limit when the cheapest setting fits; the "
          "paper says how many of the plan's settings that was."),
    Check('run_over_topic_limit', 'core.run_checks', 'RunChecksMixin._size_the_run',
          'execute', 'measured', 'note',
          "Over the topic's own time limit but within FI's hard limit, the run goes ahead and the paper says it "
          'was over.'),
    Check('search_scan_too_large_stop', 'core.run_checks', 'RunChecksMixin._size_the_scan',
          'execute', 'own_record', 'stop',
          'A coarse scan with more settings than FI lists, or whose timed total exceeds execution.timeout_s, '
          "stops the quest (FI does not run a smaller scan than the plan's)."),
    Check('small_run_crashed', 'core.run_checks', 'RunChecksMixin._small_run_gate',
          'implement', 'measured', 'repair',
          "FI's small run of simulate.py (few settings, few trials) before the code is read: a load error or "
          "most trials of a setting failing sends it to repair with the traceback; no repair left, the study's "
          'own checks decide.'),
    Check('split_run_raw_stale', 'core.split_run', 'stale_reason',
          'execute', 'own_record', 'repair',
          'Raw files from another version of the simulation, seed or manifest are not reused and the simulation '
          'runs again.'),
    Check('agent_repair_scope', 'core.agent_scope', 'find_violations',
          'execute_reflect', 'model_code', 'repair',
          'An agent repair round that edited code outside the failing function (and callees/callers) without a '
          'NEEDS claim is put back and the agent sent back once.'),
    Check('plausibility_declared_bounds', 'core.plausibility', 'violations',
          'execute_reflect', 'model_code', 'repair',
          'A value breaking a bound the design declared (result_assertions), a number the script caps at the '
          'bound (clamped), a quantity on a bound in several settings (at_bound) or a bounded quantity no '
          'longer reported sends the run back for repair.'),
    Check('plausibility_unproven_cap', 'core.plausibility', 'check_design',
          'execute_reflect', 'model_code', 'note',
          "A value exactly on a constant the script caps that quantity at, with the cap's place in the result "
          'unprovable, is a warning only.'),
    Check('phased_split_overlap', 'core.phased', 'decide_split',
          'execute', 'own_record', 'note',
          'A split of the supplied data that puts any unit in both parts is refused (zero-overlap manifest) and '
          'nothing is held back; the result stays preliminary.'),
    Check('phased_rows_belong_together', 'core.engine', 'Engine._phased_ask_split',
          'execute', 'own_record', 'stop',
          'Under research, when nothing says which rows of the data belong together, the quest stops once to '
          'ask (plan.md answer); other profiles hold nothing back.'),
    Check('phased_isolation_scan', 'core.phased_isolation', 'scan',
          'execute', 'model_code', 'note',
          "Without a container, the quest's code is read for paths out of the quest folder; a hit makes "
          'held-back isolation unverified, keeping the confirm result from counting.'),
    Check('phased_confirm_compromised', 'core.phased', 'mark_compromised',
          'execute', 'own_record', 'note',
          'A confirm run that could not be kept apart from exploration (held-back part not put in place, not '
          "the confirm run's result, seed ignored) leaves nothing confirmed."),
    Check('phased_unconfirmable', 'core.phased', 'mark_unconfirmable',
          'execute', 'own_record', 'note',
          'No data held back and a run on new seeds would only repeat exploration (no randomness or seed '
          'ignored): the quest cannot be confirmed and says why.'),
    Check('phased_new_candidate', 'core.phased', 'new_candidate',
          'execute', 'own_record', 'note',
          'Code or protocol changed after the confirm run began makes a new candidate back in exploration; the '
          'earlier verdict stays.'),
    Check('phased_record_gap', 'core.phased', 'record_gap',
          'evidence_gate', 'own_record', 'note',
          'A candidate whose verdict lines do not match the record is a gap that keeps the result preliminary.'),
    Check('metric_spec_unusable', 'core.metric_spec', 'normalize',
          'plan', 'model_text', 'repair',
          'Strict shape check of protocol.metrics (decides which estimator runs); in a draft one unusable spec '
          'is left out with a sentence (repair), not the list.'),
    Check('metric_known_value_problem', 'core.metric_spec', 'known_value_problems',
          'plan', 'model_text', 'repair',
          'A declared true value that is no number, threshold, constant, grid parameter or computable formula '
          'of them is a problem asked back.'),
    Check('metric_multiplier_unstated', 'core.metric_spec', 'multiplier_unstated',
          'plan', 'model_text', 'note',
          'Metrics declaring a standard error and interval for a known value but no degrees_of_freedom: the '
          'multiplier cannot be held to the declared one.'),
    Check('metric_reference_unusable', 'core.reference_values', 'problems',
          'plan', 'model_text', 'strip',
          "A metric's reference value that cannot be computed or contradicts the plan's own checks is left out "
          '(the metric kept) with a note (without_bad_reference).'),
    Check('metric_pair_ids_missing', 'core.metric_spec', 'missing_pair_ids',
          'execute', 'own_record', 'stop',
          'Paired metrics whose per-trial values print without a pair id can only be joined by position; stops '
          'under research (engine.run_manifest_check: block; warn by default).'),
    Check('metric_statistics_coverage', 'core.metric_spec', 'coverage_gaps',
          'evidence_gate', 'own_record', 'note',
          'What keeps the statistics from being adequate (undeclared estimators, missing pairs) is an evidence '
          'gap.'),
    Check('reference_copies_measured', 'core.reference_values', 'copies_measured',
          'analyze', 'measured', 'strip',
          'A reference that reproduces the measured means closer than they are known (read off the run, not '
          'worked out) is dropped from the comparison.'),
    Check('reference_cross_check', 'core.reference_values', 'cross_check',
          'analyze', 'own_record', 'note',
          'A metric reference is marked cross-checked only when a passing known-answer check confirms it, else '
          'not cross-checked.'),
    Check('refinement_converged', 'core.refinement_check', 'judge',
          'execute', 'measured', 'note',
          'After a run, headline results are run again at finer numerical settings; non-convergence is an '
          'evidence gap, never a stop.'),
    Check('optimum_check_refinement', 'core.optimum_check', 'check',
          'execute', 'measured', 'note',
          'Best designs of a search and the baseline are re-evaluated at finer levels and numerical error '
          'estimated (grid-convergence index); a check cut short is unverified.'),
    Check('optimum_check_constraints', 'core.optimum_check', 'check',
          'execute', 'measured', 'note',
          "Every limit is re-checked at the finest level with the plan's margin; a best design breaking one "
          'falls to the next candidate meeting all.'),
    Check('optimum_check_improvement', 'core.optimum_check', 'check',
          'execute', 'measured', 'note',
          "Improvement over the baseline at the finest level against the plan's threshold or the numerical "
          'error of both designs; with randomness on fresh seeds the search never used.'),
    Check('optimum_check_starts', 'core.optimum_check', 'check',
          'execute', 'measured', 'note',
          'Whether the best designs from different starting points agree at the finest level.'),
    Check('optimum_check_neighbourhood', 'core.optimum_check', 'check',
          'execute', 'measured', 'note',
          'Each variable nudged both ways: a better nearby design means the search had not finished; an unused '
          'variable or a bound-edge design is flagged.'),
    Check('optimum_check_budget', 'core.optimum_check', 'check',
          'execute', 'measured', 'note',
          'Whether the search stopped because it converged or because its budget or time ran out.'),
    Check('optimum_check_record_integrity', 'core.optimum_check', 'evidence_gaps',
          'evidence_gate', 'own_record', 'note',
          'A missing search ledger, evaluations beyond the planned budget, designs outside ranges, a changed '
          'ledger or a check file not matching best_design.json are evidence gaps.'),
    Check('manifest_record_vs_protocol', 'core.run_manifest', 'problems',
          'execute', 'own_record', 'stop',
          "Compares FI's trial ledger (or the script's own manifest) with the frozen protocol: grid swept, runs "
          'per setting, thresholds, failed trials listed; sent back, then stops; engine.run_manifest_check warn '
          '(default) only records non-fabricated differences, block (research) stops'),
    Check('manifest_fabricated_values', 'core.run_manifest', '_value_count_findings',
          'execute', 'own_record', 'stop',
          "Analysis reports a value, count or undefined-trial number FI's trials never produced, or a trial "
          'record changed on disk; sent back and stops in every mode of engine.run_manifest_check'),
    Check('manifest_value_accounting', 'core.run_manifest', 'analysis_output_problems',
          'execute', 'own_record', 'stop',
          'What the analysis lists per setting does not account for every trial FI holds; analysis sent back, '
          'stops under run_manifest_check block, only recorded under warn'),
    Check('manifest_unprinted_list', 'core.run_manifest', 'is_unprinted',
          'execute', 'own_record', 'repair',
          'Analysis prints no per-trial list for a mean FI holds trial by trial; analysis sent back under '
          'block, only logged under warn (not shown in the paper)'),
    Check('manifest_once_per_setting', 'core.run_manifest', 'once_per_setting',
          'execute', 'measured', 'note',
          "A run_cell simulation FI ran once per setting is held to the protocol's run count only if FI saw it "
          'draw random numbers (call it again); otherwise the count is waived and said'),
    Check('manifest_relabelled_lists', 'core.run_manifest', 'relabelled',
          'execute', 'own_record', 'note',
          "A list printed under the wrong name that holds another output's values at its setting is counted as "
          'that output and said in run.log'),
    Check('manifest_derived_metrics', 'core.run_manifest', 'derived_metrics',
          'execute', 'own_record', 'note',
          'Metrics the analysis derives itself (no trial returns them) get no trial-count claim; said in the '
          'run-manifest record'),
    Check('manifest_strata_coverage', 'core.run_manifest', 'strata_coverage',
          'analyze', 'own_record', 'note',
          "A mean per stratum covering fewer than all the protocol's settings is allowed but the analysis "
          'payload says so for the limitations'),
    Check('manifest_split_lint', 'core.run_manifest', 'split_lint',
          'implement', 'model_code', 'note',
          'Reads the two scripts for breaks of the two-script contract; a reading, so under '
          'execution.split_failure block it is recorded and the quest goes on (stop_power READ)'),
    Check('manifest_replicate_seeds', 'core.engine', 'Engine._check_replicate_manifests',
          'execute', 'own_record', 'note',
          "Other seeds' manifests compared with the protocol; a difference is recorded as an evidence gap, not "
          'repaired'),
    Check('manifest_grid_values_left_out', 'core.engine', 'Engine._grid_values_left_out',
          'execute', 'own_record', 'repair',
          "Analysis code names only some of an axis's values and FI's count shows it reports no trials for the "
          'rest; acted on even under run_manifest_check warn'),
    Check('manifest_run_cell_randomness', 'core.engine', 'Engine._run_cell_randomness',
          'execute', 'measured', 'repair',
          'FI calls a once-per-setting simulation again to see if numbers change or code names a random source; '
          "decides whether the protocol's run count is held (feeds the manifest)"),
    Check('manifest_stop', 'core.engine', 'Engine._manifest_stop',
          'execute', 'own_record', 'stop',
          'Run record still differs from the frozen protocol after the repairs (run_manifest_check block, or '
          'fabricated numbers in any mode): honest stop with no paper'),
    Check('manifest_repair_gave_up', 'core.engine', 'Engine._manifest_repair_gave_up',
          'execute_reflect', 'own_record', 'stop',
          'Repair gave up or none left on a run the manifest or trial checks sent back; the difference stands '
          'and the quest stops naming it'),
    Check('trial_failures_per_setting', 'core.trial_checks', 'failures',
          'execute', 'own_record', 'stop',
          "Setting where more than half of FI's trials failed: sent back; after repairs those settings are left "
          'out (strip) or, if over half the settings or calibration unjudged, the quest stops'),
    Check('trial_calibration_vs_truth', 'core.trial_checks', 'calibration',
          'execute', 'measured', 'stop',
          "Reported intervals and standard errors measured against the plan's declared true value over FI's "
          'trials; grossly wrong sent back, then stop (disclosed only if the pooled error is fine)'),
    Check('trial_undefined_values', 'core.trial_checks', 'undefined',
          'execute', 'own_record', 'note',
          'A declared value with no number in trials that ran to the end, per setting; recorded and stated, not '
          'a failed trial'),
    Check('trial_checks_stop', 'core.run_checks', 'RunChecksMixin._trial_checks_stop',
          'execute_reflect', 'own_record', 'stop',
          'When no repair of a trial-check fault is coming: partial failures leave the settings out, any other '
          'fault stops with no paper'),
    Check('search_failed_evaluations', 'core.optimise', 'OptimisationRunner._check_trials',
          'execute', 'own_record', 'stop',
          "Same failed-trial check applied to the search's evaluations and coarse scan; calibration not "
          'checked; sent back then stop'),
    Check('coverage_recount', 'core.result_checks', 'ResultChecksMixin._coverage_recount_findings',
          'execute', 'own_record', 'strip',
          "Coverage the analysis reports differs from FI's own count of the trials against the declared truth; "
          'analysis sent back then the number is left out of the results'),
    Check('reported_settings_match', 'core.reported_settings', 'problems',
          'execute', 'own_record', 'strip',
          "Tables the analysis reports carry one cell's results under another cell's setting, or a collapsed "
          'axis; analysis sent back (max REPAIRS) then the keys are left out and the paper says so'),
    Check('degenerate_figures', 'core.degenerate_results', 'figure_problems',
          'execute', 'own_record', 'strip',
          'Empty figure panels or series all at one x, from the figure records; analysis sent back, then the '
          'figure is deleted'),
    Check('degenerate_quantities', 'core.degenerate_results', 'quantity_problems',
          'execute', 'own_record', 'strip',
          'A reported quantity the plan says is estimated that is identical at all settings or stuck at its '
          'bound while others vary; simulation sent back then the quantity is left out'),
    Check('flat_quantity_kept', 'core.degenerate_results', 'kept_note',
          'execute', 'own_record', 'note',
          'A flat quantity the plan does not say is estimated from data is kept in results and the limitations '
          'say it was identical at all settings'),
    Check('figure_gaps', 'core.degenerate_results', 'figure_gaps',
          'execute', 'own_record', 'note',
          'A figure with no point at some settings: not repaired, the writer is told which settings it does not '
          'show'),
    Check('no_figures_drawn', 'core.result_checks', 'ResultChecksMixin._no_figures_findings',
          'execute', 'own_record', 'note',
          'The plan has figures and the run saved none; results stand, the writer is told and the limitations '
          'say so'),
    Check('result_check_gave_up', 'core.result_checks', 'ResultChecksMixin._result_check_gave_up',
          'execute_reflect', 'own_record', 'note',
          'Repair gave up on a run the results check sent back: the same code runs once more with repairs spent '
          'and what is wrong is left out, never a stop'),
    Check('typed_results_in_code', 'core.run_checks', 'RunChecksMixin._results_from_computation',
          'execute', 'model_code', 'strip',
          'Reads simulate.py with ast for returned numbers typed into the code; sent back twice to compute '
          'them, then the keys are left out and the paper says so'),
    Check('typed_results_text_unread', 'core.run_checks', 'RunChecksMixin._typed_results_stable',
          'execute', 'model_code', 'stop',
          'After an oracle repair rewrote the simulation, a text still unread for typed-in results when rounds '
          'are spent is not run; quest stops'),
    Check('speed_up_study', 'core.run_checks', 'RunChecksMixin._speed_up_rounds',
          'execute', 'measured', 'repair',
          'Edits that make the simulation faster are kept only when every number in a fixed trial sample is '
          'unchanged and it is clearly faster'),
    Check('pilot_reads_trials', 'core.run_checks', 'RunChecksMixin._pilot_gate',
          'execute', 'measured', 'stop',
          'Timed pilot trials read before the freeze: a protocol metric not a number, never moving, or an '
          'output the same everywhere; sent back, then stops'),
    Check('review_crash_run', 'core.run_checks', 'RunChecksMixin._review_crash_run',
          'execute', 'measured', 'repair',
          "A reader's finding test that crashed inside the code goes to the failed-run repair (at most "
          'REVIEW_CRASHES per quest)'),
    Check('gate_crash_repair', 'core.run_checks', 'RunChecksMixin._gate_crash_repair',
          'evidence_gate', 'measured', 'repair',
          "At the gate a shown-fault test crashed inside the program: back to the run step's crash repair while "
          'budget lasts, otherwise only recorded'),
    Check('flat_sweep', 'core.flat_output', 'flat_sweep',
          'execute', 'own_record', 'repair',
          'Every number the simulation returned is identical in every setting AND its code never lets the '
          'settings reach the output (text reading); simulation sent back, honest stop if repairs run out'),
    Check('flat_unchanged_settings', 'core.flat_output', 'unchanged_axes',
          'execute', 'own_record', 'repair',
          "Settings that changed none of the plan's declared results (timings ignored) and which the code never "
          'reads; simulation sent back once per quest, otherwise recorded as observed no difference'),
    Check('flat_identical_results', 'core.flat_output', 'identical_results',
          'execute', 'own_record', 'note',
          'Two declared results that are identical in every trial of every setting; observation given to the '
          'analysis and writer'),
    Check('no_difference_observed', 'core.flat_output', 'observed_flat_sweep',
          'execute', 'own_record', 'note',
          'All settings gave the same numbers but the code does read them: kept as an observed absence of '
          'difference in needs/NO_DIFFERENCE_OBSERVED.json'),
    Check('flat_search_objective', 'core.flat_output', 'flat_search',
          'execute', 'own_record', 'repair',
          'Search where every evaluation gave the same objective and the simulation never uses the design '
          'variables; sent back to be repaired'),
    Check('flat_search_check', 'core.flat_output', 'flat_search_check',
          'execute', 'measured', 'repair',
          "FI's check of the best design: nudging every design variable changes nothing the simulation returns; "
          'sent back'),
    Check('search_no_effect', 'core.flat_output', 'no_effect_search',
          'execute', 'measured', 'repair',
          "First designs of a search that between them moved every design number all gave the baseline's "
          'results; simulation sent back'),
    Check('search_flat_objective_stop', 'core.optimise', 'OptimisationRunner._flat_objective_result',
          'execute', 'measured', 'repair',
          'Objective gave one value across designs sampled over the whole range; search stops early and the '
          'gate asks the plan once for a design whose objective can move'),
    Check('search_stepped_objective', 'core.optimise', 'OptimisationRunner._stepped_result',
          'execute', 'measured', 'repair',
          'Objective counted in whole steps of a numerical setting; simulation sent back once to read it '
          "between the setting's points"),
    Check('search_scan_unchanged', 'core.optimise', 'OptimisationRunner._unchanged_result',
          'execute', 'measured', 'repair',
          'A search setting whose scan changed none of the results; simulation sent back once'),
    Check('search_entry_returns_objective', 'core.optimise', 'search_entry_returns_objective',
          'implement', 'measured', 'repair',
          "After a repair, calls the search's own evaluation of the baseline design to see the entry still "
          'returns the objective; a repair that does not is refused'),
    Check('search_calls_own_optimiser', 'core.optimise', 'searches_itself',
          'execute', 'model_code', 'note',
          "Warns when the simulation's own code imports or calls an optimiser (FI runs the search itself)"),
    Check('search_still_improving', 'core.optimise', 'still_improving',
          'execute', 'own_record', 'note',
          'Best design still improving when the budget ended: FI adds one more round automatically and states '
          'it'),
    Check('unsteady_numbers_excluded', 'core.trial_runner', 'unsteady_names',
          'execute', 'measured', 'note',
          'FI calls the cheapest setting again; numbers that differ (timings) are excluded from the flat and '
          'unchanged-result judgements'),
    Check('run_trial_is_one_trial', 'core.trial_runner', 'trial_aggregates',
          'execute', 'model_code', 'repair',
          'run_trial that reaches run_cell (a mean over trials) is read from code, settled by calling it with '
          'two seeds; sent back to the code step'),
    Check('trial_no_success', 'core.trial_runner', 'TrialsRunner.execute',
          'execute', 'own_record', 'repair',
          'Every trial failed or no trial ran successfully, or the code reads cell[...] with no grid in the '
          'protocol; simulation sent back with the reasons'),
    Check('reported_values_not_run', 'core.trial_runner', 'reported_values_not_run',
          'execute', 'own_record', 'stop',
          "A <name>_values list the analysis printed holds values FI's trials never produced; fabricated "
          'finding, stops in every mode'),
    Check('given_values_not_run', 'core.trial_runner', 'given_values_not_run',
          'execute', 'own_record', 'stop',
          "Values of a mean over a subset (given) that FI's trials never produced; fabricated finding"),
    Check('given_rows_trialwise', 'core.trial_runner', 'given_rows_problems',
          'execute', 'own_record', 'repair',
          "Research profile: a mean over the trials a proportion counts is recomputed trial by trial from FI's "
          'rows; analysis or simulation sent back'),
    Check('trial_record_altered', 'core.trial_runner', 'recorded_values_by_cell',
          'execute', 'own_record', 'stop',
          "FI's trial record changed on disk since it wrote it; trials run again, a fabricated finding"),
    Check('queued_code_changed', 'core.trial_runner', 'code_changed_while_queued',
          'execute', 'own_record', 'note',
          "Code changed while a cluster job was queued; logged and the attempt's record says so"),
    Check('trial_return_format', 'core.in_run', 'read_returned',
          'execute', 'own_record', 'repair',
          'What run_trial/run_cell returned must be a dict of numbers (or per-value numbers); otherwise the '
          'trial counts as failed with the reason'),
    Check('results_distinct_settings', 'core.results_by_setting', 'unread_axes',
          'write', 'model_code', 'note',
          "Grid axes the code never reads (sure reading only) pool into one distinct setting in FI's results "
          'table'),
    Check('results_vs_reference', 'core.results_by_setting', 'comparison',
          'write', 'own_record', 'note',
          "Plain verdict of each setting's mean and 95 percent interval against the plan's reference value in "
          'the results table'),
    Check('trial_table_precision', 'core.trial_table', 'per_setting_precision',
          'evidence_gate', 'own_record', 'note',
          "Whether the plan's precision target holds at each setting's own trials; feeds the evidence gap and "
          "the paper's disclosure"),
    Check('numeric_warnings_scan', 'core.numeric_warnings', 'scan',
          'execute', 'measured', 'stop',
          'Reads stderr and results for overflow, non-convergence, ignored arguments, complex discard, NaN/inf; '
          'repaired then stops; a NaN or infinity only for a result, reference or physical_bound the plan declares (its role), any other disclosed; '
          'engine.numeric_warnings block (default) / warn records / off'),
    Check('degenerate_run_guard', 'core.engine', '_is_degenerate_result',
          'execute_reflect', 'own_record', 'repair',
          'Run exits 0 but every numeric metric is about 0; sent back to repair, then analysis frames an honest '
          'failure note (engine.degenerate_run_guard)'),
    Check('result_assertions_bounds', 'core.engine', '_assertion_violations',
          'execute_reflect', 'own_record', 'repair',
          'Values break the bounds the design declared or are capped by the script; sent back until repairs are '
          'spent, then proceeds with the violation noted'),
    Check('figure_overlap_redraw', 'core.engine', 'Engine._figure_overlaps_to_repair',
          'execute_reflect', 'measured', 'repair',
          'A legend or title measured on the saved canvas overlapping data; one redraw request, never stops the '
          'quest'),
    Check('exec_reflect_give_up', 'core.engine', 'Engine._node_execute_reflect',
          'execute_reflect', 'model_judge', 'note',
          'The repair model gives up (give_up_reason) or returns no code; the failure goes on to analyze and '
          'the evidence gate; manifest/trial/result-check faults are still stopped or left out'),
    Check('numeric_warnings_stop', 'core.engine', 'Engine._numeric_findings',
          'execute_reflect', 'measured', 'stop',
          'Numeric warnings the repairs did not remove stop the quest before a paper (cannot_repair) under '
          'numeric_warnings block'),
    Check('replicate_seed_ignored', 'core.engine', 'Engine._node_execute',
          'execute', 'model_code', 'note',
          'Replicates agree and the code never reaches a generator with the seed: one run repeated, no '
          'aggregate or confidence interval is reported; plus manifest and typed-result steps of this node'),
    Check('exec_timeout_kill', 'core.execution', 'VenvExecutor.execute',
          'execute', 'measured', 'stop',
          'A command running past execution.timeout_s is killed and marked timed_out; the gate then states the '
          'study was too large and stops'),
    Check('docker_resource_caps', 'core.execution', 'DockerExecutor.execute',
          'execute', 'measured', 'stop',
          'Container memory, CPU and process caps (execution.docker_*) end a run that exceeds them; the limit '
          'note is given to the repair or stop'),
    Check('outside_copy_bring_back', 'core.outside_run', 'OutsideCopy.bring_back',
          'execute', 'own_record', 'strip',
          'After a run from a scratch copy only files in the allowed folders come back to the quest; anything '
          "else is said and dropped (FI's records never reachable)"),
    Check('outside_stopped_leftovers', 'core.outside_run', 'execute_keeping_only_finished',
          'execute', 'own_record', 'strip',
          "Files created by a trial FI stopped for its time are removed; only finished settings' files stay"),
    Check('code_receipt_verdict', 'core.gate_receipt', 'verdict',
          'execute', 'own_record', 'stop',
          'Code that would run vs the hash of the version the checks were made for: same or drawing-only '
          'passes, a changed computation re-runs the checks and the run starts only if they pass'),
    Check('record_files_changed_outside', 'core.record_anchor', 'check_on_start',
          'several', 'own_record', 'note',
          "On start compares record files with FI's note of what it wrote; foreign lines moved aside, said in "
          'run.log, evidence not publication-ready'),
    Check('receipt_stale_inputs', 'core.receipts', 'stale_inputs',
          'evidence_gate', 'own_record', 'note',
          'Gate receipt made for other sources than the ones now in the paper is stale: an evidence gap'),
    Check('model_call_record_gaps', 'core.attempt_records', 'model_call_gaps',
          'several', 'own_record', 'note',
          "What keeps the quest's record of model calls incomplete (calls with no named model, unwritten rows); "
          'an evidence gap'),
    Check('source_failure_summary', 'core.source_failures', 'record_failure',
          'literature', 'own_record', 'note',
          'Per-quest record of retrieval failures (HTTP 429/400/timeouts) written to run.log and '
          '.fi/source_failures.json'),
    Check('seed_reaches_code', 'core.run_facts', 'seed_verdict',
          'write', 'model_code', 'note',
          "Reads the code to say whether FI's per-trial seed reaches the simulation; the paper's run paragraph ({{run_facts}}) and a "
          'log warning use it'),
    Check('shared_random_streams', 'core.run_facts', 'shared_stream_notes',
          'write', 'measured', 'note',
          "From FI's trial record: settings that drew the same random numbers; a note only"),
    Check('reader_run_time_verdict', 'core.run_facts', 'reader_verdict',
          'write', 'own_record', 'note',
          "Whether a reader's run of the delivered code is over or within the time limit; given to the paper's "
          'run-time sentence'),
    Check('evidence_ladder', 'core.evidence', 'assess',
          'evidence_gate', 'own_record', 'note',
          'Classifies the quest on the six-level evidence ladder from its records and lists one gap per missing '
          'check; needs no model'),
    Check('evidence_seal_verify', 'core.evidence', 'verify_seal',
          'several', 'own_record', 'note',
          'Research profile: a record is publication-ready only if the trace seal is last, intact and names the '
          'evidence, attempts and paper; else read one level down'),
    Check('evidence_gate_rule', 'core.engine', '_evidence_gate_rule',
          'evidence_gate', 'own_record', 'repair',
          'Source counts decide the gate for simulation topics with retrieval on: no source with text means '
          'broaden, many sources with support means sufficient'),
    Check('evidence_gate_model_verdict', 'core.engine', 'Engine._node_evidence_gate',
          'evidence_gate', 'model_judge', 'repair',
          'Otherwise asks a model sufficient/broaden/insufficient; one bounded broaden re-entry to literature '
          '(evidence_gate_max_broaden); a failed call is recorded as unknown (research: pauses)'),
    Check('evidence_gate_flat_objective', 'core.engine', 'Engine._flat_objective_verdict',
          'evidence_gate', 'measured', 'repair',
          'Search whose objective did not move: plan asked once for a design whose objective can move (spends '
          'one design iteration)'),
    Check('evidence_gate_unreadable_analysis', 'core.engine', 'Engine._unreadable_analysis_verdict',
          'evidence_gate', 'model_text', 'stop',
          "The model's analysis of the results could not be read even when asked twice: no findings, honest "
          'stop with no paper'),
    Check('evidence_gate_no_results', 'core.engine', 'Engine._no_results_verdict',
          'evidence_gate', 'own_record', 'stop',
          "Run produced no number or none of the study's quantities, was stopped before the study, or never "
          'finished: code written again once, design asked again once, then stuck (also constant headline from '
          'a second model plus results)'),
    Check('shown_fault_outcome', 'core.engine', 'Engine._shown_fault_outcome',
          'evidence_gate', 'measured', 'stop',
          'Decides stop vs leave out vs not computed for each shown fault from the records (headline, missing '
          'quantity, keys that can be removed)'),
    Check('evidence_route', 'core.engine', 'Engine._route_after_evidence_gate',
          'evidence_gate', 'own_record', 'repair',
          "Routes the gate's decision to write, literature, design, implement, run-step crash repair or stuck; "
          'phased mode routes a write to the confirm run'),
    Check('stuck_no_findings_stop', 'core.engine', 'Engine._node_stuck_no_findings',
          'evidence_gate', 'own_record', 'stop',
          'States the honest stop (needs/STUCK.json) for reasons no_numbers, crashed, flat, too_long, not_run, '
          'cannot_repair, shown_fault_open, headline_missing; earlier paper set aside'),
    Check('precision_target_missed', 'core.engine', 'Engine._precision_missed',
          'cross_check', 'own_record', 'repair',
          "re_experiment sends the plan back to design only when FI's own record shows a precision target "
          'missed and a larger run fits the time allowed'),
    Check('cross_check_route', 'core.engine', 'Engine._cross_check_route',
          'cross_check', 'own_record', 'repair',
          'write, broaden_lit or redesign from analyze.next_step, the precision record, the phased confirm '
          'stage and engine.max_iterations'),
    Check('cross_check_literature', 'core.engine', 'Engine._node_cross_check',
          'cross_check', 'model_judge', 'note',
          'Each key finding searched in the literature and classed supporting/conflicting/neutral with an '
          'optional verification pass; given to the writer'),
    Check('analysis_unreadable_flag', 'core.engine', 'Engine._node_analyze',
          'analyze', 'model_text', 'note',
          'Analysis reply that cannot be read twice is flagged unreadable (gate then stops); a degenerate '
          'result is framed as a failure note, a single run as single measurement'),
    Check('data_load_no_user_data', 'core.engine', 'Engine._node_data_load',
          'analyze', 'own_record', 'note',
          'No-simulation mode: data/ holds only README so analysis runs with an empty result_json'),
    Check('wait_for_data_pause', 'core.engine', 'Engine._node_wait_for_data',
          'analyze', 'own_record', 'stop',
          'No-simulation mode: data/ empty pauses the quest for a person to supply a dataset'),
    Check('improve_edit_refused', 'core.improve', 'check_edit',
          'improve', 'model_code', 'note',
          "Model's single edit is refused before running if it touches the wrong file, does not parse, changes "
          "nothing computed, repeats a version, plants a check's expected value or alters a measure"),
    Check('improve_tamper_guard', 'core.improve', 'guard_hashes',
          'improve', 'own_record', 'repair',
          'Hashes of the files FI judges by and code/ before and after a round; a round whose run changed them '
          'is aborted, records and best version put back, loop stops'),
    Check('improve_ratchet', 'core.improve', 'verdict',
          'improve', 'measured', 'stop',
          'Keeps a changed version only if some criterion got better beyond tolerance and none worse; a '
          'regression stops the quest for a person under rigor_profile research, a warning otherwise'),
    Check('improve_study_fault', 'core.engine', 'Engine._simulation_repair_fault',
          'improve', 'measured', 'repair',
          "A version meeting the checks is not kept if the study's own function no longer returns the study's "
          'results'),
    Check('improve_only_noise', 'core.improve', 'only_noise',
          'improve', 'own_record', 'note',
          'Loop is skipped when every unmet criterion is one a second model flagged (nothing an edit could fix)'),
    Check('record_claim_optimised_vs_search', 'core.record_claims', 'found',
          'review', 'own_record', 'strip',
          'paper calls a design optimised/best while the search record (best_design, OPTIMUM_CHECK) says no '
          'better design; rewrites capped, then the sentence is taken out'),
    Check('record_claim_caption_optimised', 'core.record_claims', 'found',
          'review', 'own_record', 'strip',
          'a sentence names a figure as the optimised design when the record has no optimised design; rewritten, '
          'then taken out'),
    Check('record_claim_validated_vs_oracle', 'core.record_claims', 'found',
          'review', 'own_record', 'strip',
          'paper says validated/confirmed against known answers while ORACLE_CHECK is unconfirmed or failed; '
          'rewritten, then taken out'),
    Check('record_claims_rewrite_cap', 'core.engine', 'Engine._route_after_review',
          'review', 'own_record', 'repair',
          'record contradiction sends the paper back to write up to _RECORD_REWRITES (2) times outside '
          'max_iterations; then listed in QUALITY.md, never ends the quest (stop_power READ)'),
    Check('stat_interval_method_mismatch', 'core.stat_claims', '_check_interval_method',
          'review', 'own_record', 'repair',
          'an interval labelled with an estimator (e.g. exact binomial) that is really the seed t-interval; '
          'forced text-only rewrite (mislabelled_statistic)'),
    Check('stat_probability_complement', 'core.stat_claims', '_check_probability_complement',
          'review', 'own_record', 'repair',
          'outbreak probability described as the chance the disease vanishes (the complement); forced text-only '
          'rewrite'),
    Check('paper_unknown_placeholder', 'core.paper_facts', '_unknown_hits',
          'review', 'own_record', 'repair',
          "a {{...}} name that is not FI's placeholder, an unknown {{not_done:id}} or {{figure:..}}; paper "
          "would read 'not recorded', sent back to write"),
    Check('paper_result_key_unfilled', 'core.paper_facts', 'value_hits',
          'review', 'own_record', 'strip',
          'a {{v:...}} result key FI cannot fill from trials/plan; rewritten, and with no round left the '
          'sentence or key is taken out (take_out_unfilled); only when the paper was written under result keys'),
    Check('paper_typed_number_not_plan', 'core.typed_numbers', 'hits',
          'review', 'own_record', 'repair',
          'rule R1: a number the writer typed that is not a plan/topic number or a label; rewritten, with no '
          'round left kept as written and listed in QUALITY.md as not traced (typed_numbers.untraced)'),
    Check('paper_typed_number_edit_once', 'core.engine', 'Engine._typed_pass_hits',
          'review', 'own_record', 'repair',
          'when no review round is left, the sentences with typed or unfilled numbers get one passage edit '
          'before anything is taken out'),
    Check('paper_limitations_block', 'core.paper_facts', 'limitations_text',
          'write', 'own_record', 'note',
          "FI adds a list to the end of the paper's Limitations section from its records (what the study did not "
          "do, an unmet target, a page overrun, claims found unsupported, the last review's open points); the "
          "writer's own Limitations text is kept, and with nothing to state nothing is added"),
    Check('extension_result_unreported', 'core.number_provenance', 'unreported_results',
          'write', 'own_record', 'repair',
          "a result a refine's extension added that the paper does not report: writer asked once more, then "
          'said at the review pause'),
    Check('figure_label_shows_code_name', 'core.figure_labels', 'findings',
          'write', 'own_record', 'repair',
          'axis/legend/title text showing a code variable name (Ea_hat, pi1); figures redrawn from saved '
          "numbers with the plan's label (once per script)"),
    Check('figure_drawn_from_typed_numbers', 'core.figure_data_check', 'typed_series',
          'execute', 'model_code', 'strip',
          'plotting call whose data is a literal list typed into the script; script sent back twice, then those '
          'figures are dropped and the paper says so'),
    Check('figure_repair_changes_computation', 'core.plot_scope', 'computation_changes',
          'execute', 'model_code', 'repair',
          'a figure repair that changes anything but drawing (AST of computation differs) is refused and the '
          'script kept as written'),
    Check('replot_script_typed_or_other_numbers', 'core.engine', 'Engine._replot_layout',
          'write', 'model_code', 'repair',
          'redraw script draws typed-in numbers, or other numbers than the run saved for a rename-only figure: '
          'tried twice, figures put back'),
    Check('replot_unrequested_change_reverted', 'core.engine', 'Engine._replot_layout',
          'write', 'measured', 'repair',
          'redraw changed a figure not named, a mean-of-seeds figure, saved numbers, or added files (digest '
          'compare): put back/removed'),
    Check('replot_layout_not_applied', 'core.engine', 'Engine._route_after_replot',
          'write', 'measured', 'repair',
          'a layout note the redraw could not carry out (ran but changed no figure, failed twice) goes back to '
          'the writer once and is shown at the review pause'),
    Check('figure_placed_by_fi', 'core.engine', '_place_figures',
          'write', 'own_record', 'repair',
          'every figure the run drew goes in under its number, with the caption FI writes from its record; the '
          "draft's own images are taken out, no model call"),
    Check('figure_web_license_gate', 'core.figure_sources', '_is_free_license',
          'several', 'measured', 'strip',
          'web/arXiv illustrative figure used only with a free/open license (web_figures node, no-simulation or '
          'survey quests); others are not embedded'),
    Check('abstract_missing', 'core.paper_sections', 'missing_abstract',
          'write', 'model_text', 'repair',
          'scientific paper without a bodied ## Abstract; an Abstract is asked for from the draft once'),
    Check('paper_citation_names_no_source', 'core.engine', '_finalize_paper_sources',
          'write', 'own_record', 'strip',
          'citation numbers that name no source in the prior-work block are removed; engine writes '
          'References/Further reading'),
    Check('foundational_works_uncited', 'core.engine', '_uncited_foundational_works',
          'write', 'own_record', 'note',
          'foundational works in the prior-work block the draft does not cite; logged and advisory in the '
          'review prompt'),
    Check('code_project_clean_run', 'core.engine', 'Engine._check_code_project',
          'write', 'measured', 'note',
          'code/ run in a clean environment once per code version; a failure is a warning only'),
    Check('page_limit_further_reading_dropped', 'core.engine', 'Engine._drop_further_reading_to_fit',
          'review', 'measured', 'strip',
          'draft over the page limit only because of Further reading: last entries dropped until it fits'),
    Check('page_limit_body_trim', 'core.engine', 'Engine._trim_body_to_fit',
          'review', 'model_judge', 'strip',
          'a model ranks sentences, FI cuts background/discussion sentences (never the last citation of a '
          'source or last statement of a number; paper_trim.choose) to fit'),
    Check('claim_grounding_unsupported', 'core.engine', 'Engine._node_claim_check',
          'claim_check', 'model_judge', 'repair',
          'a model grounds each paper claim in run results or one cited source, else unsupported; forces '
          'unsupported_claim rewrite; off when engine.claim_grounding is false (default true)'),
    Check('claim_quote_not_in_source', 'core.engine', '_quote_in_source',
          'claim_check', 'own_record', 'repair',
          "a citation basis whose quote is not in the source's own text (or only a model's figure reading) is "
          'marked unsupported'),
    Check('claim_title_only_source', 'core.engine', '_apply_title_only_rule',
          'claim_check', 'own_record', 'repair',
          'a sentence resting on a source held as title only is marked unsupported, listed or not'),
    Check('claim_retracted_source', 'core.retractions', 'apply_to_claims',
          'claim_check', 'own_record', 'repair',
          'a claim grounded in a source Crossref lists as retracted is marked unsupported'),
    Check('claim_earlier_unsupported_kept', 'core.engine', '_unsupported_still_in',
          'claim_check', 'own_record', 'repair',
          'a sentence an earlier check found unsupported that the paper still holds unchanged stays unsupported '
          'when the new check omits it'),
    Check('claim_check_unreadable_reply', 'core.engine', 'Engine._stop_once_for_check',
          'claim_check', 'model_judge', 'stop',
          'claim check call failed or its reply names no claims list: review forces citations_unchecked; under '
          'rigor_profile research the quest stops once (resume retries)'),
    Check('claim_check_zero_claims', 'core.engine', 'Engine._node_claim_check',
          'claim_check', 'model_judge', 'stop',
          '0 claims extracted from a paper body over 2000 chars is treated as a failed check (same one-time '
          'research-profile stop)'),
    Check('review_unsupported_claim_hit', 'core.engine', '_unsupported_claim_hit',
          'review', 'own_record', 'repair',
          'FI adds unsupported_claim to the must-fix hits when the claim check found unbacked claims, even if '
          'the reviewer did not'),
    Check('review_citations_unchecked_hit', 'core.engine', '_citations_unchecked',
          'review', 'own_record', 'repair',
          'forced citations_unchecked hit when the claim check could not run on this draft'),
    Check('citation_pair_rules', 'core.citation_support', 'pre_problem',
          'citation_check', 'own_record', 'strip',
          "per (sentence, source): retracted, no retrieved text, a number in neither source nor the run's "
          'results, or snippet/title-only text saying more; only ever adds unsupported'),
    Check('citation_quote_not_in_source', 'core.citation_support', 'post_problem',
          'citation_check', 'own_record', 'strip',
          "the passage a reader names must be in the source's own text"),
    Check('citation_reader_judgement', 'core.citation_support', 'check_paper',
          'citation_check', 'model_judge', 'strip',
          'reader (citation_check step) judges each pair; unsupported ones are replaced by another source, '
          'reworded, or dropped; never stops the quest'),
    Check('citation_unfixable_settled', 'core.citation_support', 'settle',
          'citation_check', 'own_record', 'strip',
          'rule with no model: drop the sentence, the clause with foreign numbers, or only the citation when '
          'nobody could fix it'),
    Check('literature_retraction_lookup', 'core.retractions', 'check_literature',
          'literature', 'measured', 'note',
          'each DOI looked up in Crossref; retracted sources are marked, never cited and never fetched; '
          'unreachable is not checked'),
    Check('review_verdict_missing', 'core.engine', 'Engine._node_review',
          'review', 'model_judge', 'stop',
          'reviewer (or a panelist) answered twice without a verdict or could not be asked: pause '
          'review_unavailable, never a stand-in accept'),
    Check('review_models_collapsed', 'core.engine', 'Engine._review_models_collapsed',
          'review', 'measured', 'stop',
          'research profile: panel set up on several models but one provider/model answered all; quest pauses '
          'unless engine.one_model_review'),
    Check('review_models_one_model_panel', 'core.engine', 'Engine._review_models_stop',
          'several', 'own_record', 'stop',
          'research profile: every review_panel persona routed to the same model; stops before anything runs '
          'unless engine.one_model_review'),
    Check('review_panel_verdict', 'core.engine', '_aggregate_panel_reviews',
          'review', 'model_judge', 'repair',
          'deterministic vote of panelists gives the verdict, score and union of must_flag_hits; revise or any '
          'hit consumes an iteration'),
    Check('review_reviewer_page_limit_dropped', 'core.engine', '_without_reviewer_page_limit_hits',
          'review', 'own_record', 'note',
          'over_page_limit hits written by a reviewer are dropped; the page limit is a guideline and forces no round'),
    Check('review_hits_text_only', 'core.engine', '_hits_need_only_a_rewrite',
          'review', 'model_text', 'repair',
          'all must-fix hit names in _TEXT_ONLY_HITS: write again only; else objection kept in QUALITY.md and '
          'still a rewrite, not a redesign'),
    Check('review_unanswered_note_to_design', 'core.engine', 'Engine._route_after_review',
          'review', 'model_judge', 'repair',
          "a person's refine note still flagged user_feedback_unaddressed after a rewrite sends the quest to "
          'design'),
    Check('goal_coverage_notes', 'core.engine', 'Engine._goal_coverage_notes',
          'review', 'own_record', 'note',
          'numbers/figure count the topic sets vs the script, results and figures; advisory, '
          'paper/goal_coverage.json'),
    Check('goal_unplaced_topic_answers', 'core.engine', 'Engine._unplaced_topic_answers',
          'review', 'own_record', 'repair',
          'a quantity the topic asks for that FI has a value for and the paper never places; forced text-only '
          'rewrite'),
    Check('numeric_trivial_reference', 'core.engine', 'Engine._reference_check_hits',
          'review', 'own_record', 'note',
          'a reference quantity exactly 0 at every sweep setting (trivial root); advisory '
          'numeric_oracle_warnings'),
    Check('human_acceptance_recorded', 'core.engine', 'Engine._node_human_feedback',
          'human_feedback', 'own_record', 'note',
          'accept/reject/refine pause; refine with no text counts as accept; acceptance records who and the '
          'paper sha256 (automatic or person)'),
    Check('refine_points_route', 'core.engine', 'Engine._route_after_write',
          'write', 'model_judge', 'repair',
          "writer's own classification of a person's refine points: new experiment goes to design, missing "
          'number extends the script, layout note redraws, else text only'),
    Check('critique_second_pass', 'core.critique', 'generate_critique',
          'several', 'model_judge', 'note',
          'optional /critique adversarial second read of a finished quest into critique.md; not part of the '
          'graph'),
    Check('reviewer_said_nothing', 'core.review_pairs', 'said_nothing',
          'several', 'model_judge', 'note',
          'a reviewer reply with no finding and no list of what was checked is no review: asked once more, then '
          'recorded as not reviewed'),
    Check('review_plan_values_scrubbed', 'core.review_pairs', 'scrub',
          'implement', 'own_record', 'strip',
          "numbers equal to the plan's expected values/tolerances are removed from findings sent to "
          'code-writing'),
    Check('quality_report_problems', 'core.quality_report', 'build',
          'several', 'own_record', 'note',
          "QUALITY.md / quality.json: rule-based problems and suggestions read from FI's records at every stop; "
          'never calls a model'),
    Check('paper_patch_edits_accepted', 'core.paper_patch', 'apply_edits',
          'write', 'model_text', 'repair',
          'passage edits refused (PatchError) when a find is not unique, overlaps, replaces most of the paper, '
          'or changes figure links/headings; falls back to a whole-paper write'),
    Check('paper_scope_splice_accepted', 'core.paper_scope', 'splice',
          'write', 'model_text', 'repair',
          'a section rewrite lacking an asked section or giving it empty is refused (ScopeError); unasked '
          'sections are dropped; whole paper written instead'),
    Check('finding_test_demonstrates', 'core.finding_repair', 'demonstrate',
          'implement', 'measured', 'note',
          'a code-reader finding is repaired only if a test fails by assertion twice on current code; otherwise '
          'recorded not demonstrated'),
    Check('finding_test_swallowed_call', 'core.finding_repair', 'swallowed_call',
          'implement', 'model_code', 'repair',
          'a finding test that calls code inside try/except is no proof; not run, asked once more'),
    Check('finding_repair_regressions', 'core.finding_repair', 'regressions',
          'implement', 'measured', 'repair',
          'a repair that breaks a test that passed before is reverted'),
    Check('finding_repair_static_worse', 'core.finding_repair', 'worse_static',
          'implement', 'measured', 'repair',
          'a repair with more static-check faults than before is reverted'),
    Check('finding_repair_params_kept', 'core.finding_repair', 'params_kept',
          'implement', 'model_code', 'repair',
          'a repaired function must keep every parameter name'),
    Check('finding_repair_unpassed_params', 'core.finding_repair', 'unpassed_params',
          'implement', 'model_code', 'repair',
          'a repair adding parameters that callers leave on defaults is refused'),
    Check('finding_repair_frame_changes', 'core.finding_repair', 'unspoken_frame_changes',
          'implement', 'model_code', 'repair',
          'a repair that changes module/class-level statements the finding does not speak of is refused'),
    Check('finding_repair_caller_edit', 'core.finding_repair', 'caller_edit_problem',
          'implement', 'model_code', 'repair',
          "a repair's change to a direct caller must only pass new arguments, else refused"),
)
