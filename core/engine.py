"""Frontier Insight research engine — async LangGraph DAG.

Real LLM-driven nodes, code generation + execution in a per-quest
venv, Axon-backed knowledge retrieval, and SQLite-checkpointed state for
resumability after stalls (LLM quotas, OS sleep, manual interrupt).

Each Engine instance is stateless w.r.t. process globals; N instances must
coexist in one process for the fleet runner.
"""

from __future__ import annotations

# Every module-level name of the engine (imports, constants, QuestState, helpers) lives in core/engine_base.py; they
# are imported here by name so that `from core.engine import X` and `core.engine.X` read the same objects.
from .engine_base import (  # noqa: F401
    AgentBuildMixin, Any, AsyncProcessTree, AsyncSqliteSaver, Awaitable, BridgeError, Callable, ClarifyCallback,
    Command, Config, Counter, DEFAULT_PAPER_FORMAT, DockerLimits, END, ExecutionResult, FOUNDATIONAL_MAX_SUGGESTED,
    FOUNDATIONAL_SUGGESTED, FallbackLLMClient, GraphBubbleUp, GraphInterrupt, HumanFeedbackCallback, Knowledge,
    LLMClient, ModelAnswerTruncated, ModelUnavailable, NON_SCIENTIFIC_PAPER_FORMATS, PRELIMINARY_KINDS,
    PRELIMINARY_NOTE, PROGRESS_LOG_NAME, PROMPTS_DIR, PROXY_PROVIDERS, Path, ProviderLimitError, ProviderUnusable,
    ProxySupervisor, QuestArtifacts, QuestState, RepairForAgent, ResultChecksMixin, RetrievedDoc, RunChecksMixin,
    SCIENTIFIC_PAPER_FORMATS, STAGE_PROGRESS, START, StateGraph, StaticChecksMixin, TypedDict, WORK_SCOPE_PAPERS,
    WORK_SCOPE_PAPERS_AND_BOOKS, _ABSTRACT_ONLY_CHAR_THRESHOLD, _ANALYSIS_REPAIR_TRIES,
    _ANALYZE_RESULT_JSON_BUDGET_CHARS, _ANY_HEADING_RE, _AgentLines, _BODIES_REPLY_REMINDER, _CALL_ATTEMPTS,
    _CALL_SPANS, _CITATION_BRACKET_RE, _CITATION_PART, _CITATION_SOURCE_CHARS, _CI_BETWEEN_SEEDS, _CI_BOOTSTRAP,
    _CI_WILSON, _CLAIM_ARRAY_MAX_ITEMS, _CLAIM_BASES, _CLAIM_CODE_CHARS, _CLAIM_EVIDENCE_CHARS, _CLAIM_FAULTS,
    _CLAIM_FAULTS_TAKEN_OUT, _CLAIM_MODEL_FAULTS, _CLAIM_PLOT_WORDS, _CLAIM_PROTOCOL_CHARS, _CLAIM_READINGS_CHARS,
    _CLAIM_SOURCE_CHARS, _CLARIFY_LABELS, _CLI_PROVIDERS_SET, _CLUSTER_BODIES_REMINDER, _CLUSTER_PROTOCOL,
    _CLUSTER_REPLY_REMINDER, _CODE_FENCE_RE, _CODE_REEXECUTES, _DEFAULT_CHAT_TEMPERATURE, _DEPS_LINE_RE,
    _DETERMINISTIC_GATE_NODES, _DRAFTED_GRID_ASKED, _DRAFTED_GRID_ASKED_AGAIN, _DRAFTED_PARTS_ASKED,
    _DRAFTED_SEARCH_ASKED, _FAILURE_SOURCE_WORDS, _FAILURE_WORDS, _FENCED_BLOCK_RE, _FENCE_RE, _FIGURES_PER_CALL,
    _FIGURE_FORMAT_ORDER, _FIGURE_MENTION_RE, _FIGURE_PAGE_SHARE, _FIGURE_PICK_BATCH, _FIGURE_READINGS_MARK,
    _FIGURE_READING_CHARS, _FIGURE_SUFFIXES, _FILLED_AFTER_READ, _FILL_FREE_KEYS, _FILL_MARKER,
    _FINISHED_RUN_STATUSES, _FIT_APPROVER, _FIT_SOURCE, _FI_INTERNAL_KINDS, _FI_MEMORY_KINDS, _FI_MPL_PREAMBLE,
    _FI_PLAN_AUTHORS, _FI_PROVIDER_NAMES, _FLAT_APPROVER, _FLAT_SOURCE, _FRACTION_BAR_RE, _FRESH_SCRIPT,
    _FURTHER_ENTRY_RE, _FURTHER_HEADING_LINE_RE, _FURTHER_READING_HEADING_RE, _FurtherReadingBlock,
    _GATE_RULE_MIN_SOURCES, _GRID_ASKED, _GRID_ASKED_AGAIN, _HIT_NAME_RE, _IDENTIFIER_HIT, _INTERVAL_METHOD_RE,
    _IN_EACH_RUN_PROTOCOL, _JOB_NO_SEED_PROTOCOL, _JOB_PROTOCOL, _JOB_SEED_PROTOCOL, _JSON_KEY_RE, _JSON_NEXT_KEY_RE,
    _LAST_CALL, _LEDGER_ASKED, _LEDGER_PARTS, _LITERATURE_QUERIES, _LIT_EXCERPT_CHARS, _MATH_FONT_RE, _MATH_LETTERS,
    _MATH_LETTER_DIGIT_RE, _MATH_PAREN_SCRIPT_RE, _MATH_SCRIPT_MARK_RE, _MATH_SPACE_RE, _MD_FENCE_LINE_RE,
    _MIN_PAPER_CHARS, _ModelAnswerProblem, _NL, _NOT_A_CITATION_NUMBER, _NOT_GIVEN, _NOT_LITERATURE_FAILURES,
    _NOT_PROSE_RE, _NOT_SOLID, _NO_CRITERIA_LINE, _NO_SIM_DESIGN_DIRECTIVE, _NO_SIM_WRITE_NOTE, _ONCE_PARTS,
    _ORACLE_HIDDEN, _OUTSIDE_WORK_STEPS, _PAGE_LIMIT_HIT, _PANEL_COLS, _PANEL_ROWS, _PAPERS_ASKED, _PAPERS_DECLINED,
    _PAPERS_README, _PAPER_BASIS_KEYS, _PAPER_IMAGE_RE, _PAPER_PROMPT_CHARS, _PARA, _PART_ASKED, _PART_HEADING,
    _PERSONA_NAME_RE, _PERSON_QUERIES, _PIP_FREEZE_TIMEOUT_S, _PLAN_DIRECTIVE, _PROPORTION_NAME_RE, _PROSE_NUMBER,
    _PROXY_WARN_SHOWN, _PY_FENCE_RE, _PartsRefused, _PlanEditedMeanwhile, _ProgressOnly, _ProviderName,
    _QUERY_SET_HASHED, _QUEST_MEMORY_KINDS, _QUOTE_MIN_CHARS, _QuerySetsUnreadable, _RANDOM_SOURCE_PATTERN,
    _RECORD_REWRITES, _REDERIVE_BECAUSE, _REDESIGN_OWNED_PARTS, _REFERENCES_HEADING_LINE_RE, _REFERENCES_HEADING_RE,
    _REFERENCE_FORMS, _REFINE_POINT_RE, _REFINE_WHY_RE, _REFUSAL_CHARS, _REFUSED_BLOCK_CHARS, _REPAIR_REPLY_KEYS,
    _REPAIR_TEXT_KEYS, _REPLOT_STYLE_KEYS, _RESULT_LINE_RE, _RESULT_MARKER_RE, _RUN_FIGURE_IMAGE_RE,
    _SAME_STATEMENT_SHARE, _SAME_STATEMENT_WORDS, _SCHEMA_RETRY_ASKS, _SEARCH_ASKED, _SEARCH_ASKS, _SEARCH_CALLS,
    _SEARCH_CALLS_PER_ASK, _SEARCH_PROTOCOL, _SEARCH_REFLECT_ANALYSIS, _SEARCH_REFLECT_SIMULATE, _SEARCH_REFUSED,
    _SEARCH_SIZE_KEYS, _SEEDING_CALLS, _SEED_KEYWORDS, _SEED_REPAIR_DIRECTIVE, _SENTENCE_END_RE, _SETTING_FIELDS,
    _SIMULATABILITY_LEAD, _SIZE_PARTS, _SKILLS_IN_CODE, _SKILL_MOUNTS_FILE, _SOURCE_LIST_HEADING_RE,
    _SOURCE_SLIDE_CHARS_PER_LINE, _SOURCE_SLIDE_LINES, _SOURCE_SLIDE_MAX, _SOURCE_VERDICT_HASHED,
    _SPACED_HYPHEN_BREAK_RE, _SPLIT_PROTOCOL, _SPLIT_REFLECT_ANALYSIS, _SPLIT_REFLECT_SIMULATE, _SPLIT_REPLY_REMINDER,
    _SPLIT_RERUN_NOTE, _SPREAD, _SPREAD_IN_PARENS_RE, _SPREAD_RE, _STATE_TEXT_CHARS, _STOPPED_BEFORE_THE_STUDY,
    _STUDY_TYPE_DIRECTIVE, _SURVEY_DESIGN_DIRECTIVE, _SURVEY_WRITE_NOTE, _TAKE_OUT_PASSES, _TEXT_ONLY_HITS,
    _THIN_MARKS, _THIN_TEXT_CHARS, _TITLE_ONLY_UNEXPLAINED_STEMS, _TOC_LINE, _TOPIC_ASKED, _TOPIC_DESIGN_ASKED,
    _TRIAL_ORACLE_NOTE, _TRIM_ROUNDS, _UNANSWERED_NOTE_HIT, _UNPLACED_FINDING, _UNSANCTIONED_PROXY_PROVIDERS,
    _UNSEEDED_RNG_FACTORIES, _USER_DATA_SUFFIXES, _UnseededRngFinder, _VALUES_POOL_CAP, _WARNING_KINDS,
    _WORDS_PER_PAGE, _acceptance, _accepted, _agent_coder, _aggregate_cost_rows, _aggregate_panel_reviews,
    _aggregate_result_json_replicates, _analysis_frame, _anything_ran, _append_design_revision,
    _append_figure_readings, _append_further_reading, _apply_title_only_rule, _asserting_design,
    _assertion_violations, _assertion_warnings, _attach_unlabelled_lines, _attempts, _audit_log, _audit_open,
    _auto_accepts, _best_report, _bounded_seen_after, _call_drawn, _call_extras, _call_group, _call_levels,
    _call_panels, _call_plots, _call_raw_texts, _caption_from_calls, _caption_safe, _caption_words, _changelog,
    _checked, _ci_bounds, _citation_counts, _citation_labels, _citations_unchecked, _cite_labels, _citing_sentences,
    _claim_cell, _claim_choice, _claim_choice_problem, _claim_code_block, _claim_code_order,
    _claim_faults_from_sources, _claim_fi_record_block, _claim_findings_parts, _claim_keys, _claim_method_block,
    _claim_result_units, _claim_results_block, _claim_source_block, _clean_title, _close_quest_logger, _code_delta,
    _code_findings, _code_layout, _code_names_replicate_seed, _code_project, _code_reach, _code_review,
    _coerce_dep_list, _compact_result_json_block, _confirmations, _constant_expr, _content_quality, _content_stems,
    _contract, _contract_parts, _crash_kind, _criteria, _cross_check_items, _cs, _declare_scope, _declared_metrics,
    _default_clarify_questions, _deps_from_tail, _deps_to_warmup_modules, _depth_change, _design_parts,
    _design_parts_listing, _disclosure, _discover_skill_names, _doc_dedup_keys, _dotted_name, _drawn_metrics,
    _elide_value_lists, _entries_as_written, _entry_identities, _equation_tests, _evidence, _evidence_gate_rule,
    _experiment_deps, _extend_directive, _extension_values, _extract_pdf, _extract_result_json, _fault_from_record,
    _fault_record, _figdata, _figure_caption_text, _figure_claims, _figure_draws_note, _figure_draws_nothing,
    _figure_facts, _figure_files_by_stem, _figure_insertion_line, _figure_label_findings, _figure_label_found,
    _figure_list_for_prompt, _figure_menu_note, _figure_minimum_sentence, _figure_numbers, _figure_offers,
    _figure_overlap_directive, _figure_overlap_findings, _figure_plan, _figure_record_note, _figure_reference_end,
    _figure_with_sentence, _figures_drawn, _figures_section, _fill_changed_more, _fill_scope,
    _finalize_after_citation_edits, _finalize_paper_sources, _finalize_with_numbering, _finding_repair, _fit_request,
    _fit_scope, _flat_fact_sentence, _flat_moved, _flat_request, _flat_shown, _format_claim_grounding,
    _format_clarify, _format_cross_check, _format_evidence_note, _format_lit, _format_lit_excerpt,
    _format_lit_from_state, _format_lit_header, _format_reflect_history, _format_review_for_writer, _forms,
    _foundational_line, _foundational_outcomes, _foundational_review_block, _foundational_sources,
    _foundational_work_text, _foundational_write_block, _frozen, _function_steps, _further_reading_block,
    _further_reading_lines, _gate_inputs, _gate_receipt, _grid_keys, _grounding_after_trim, _guarded_absent,
    _has_full_text, _hidden, _hidden_series, _hit_name, _hits_need_only_a_rewrite, _imported_module_names, _improve,
    _in_run, _ingest_user_dropped_papers, _input_numbers, _is_abstract_only, _is_audience_appropriate,
    _is_bot_check_title, _is_citable, _is_degenerate_result, _is_fatal_provider_error, _is_metric_key,
    _is_not_a_paper, _is_open_access, _is_os_environ, _is_preliminary_memory, _is_replicate_seed_literal,
    _is_scholarly_record, _is_web_page, _item_content, _json_file_is_answer, _labelled_sources, _last_error_line,
    _latex_esc, _layout_views, _leading_simulatability_word, _ledger, _list_user_data_files, _lit_query,
    _literature_entry, _literature_in_citation_order, _literature_key, _literature_query_prompt,
    _literature_text_gaps, _load_persona_prefix, _load_prompts, _log_parse_failure, _lower_first,
    _manifest_repair_budget, _mark_preliminary, _mark_seeded_targets, _mask_prose, _matching_brace_end, _math_folded,
    _math_safe_cut, _mc_error, _memory, _merge_round_robin, _metric_spec, _names_a_verdict,
    _names_seeded_from_replicate_env, _new_quest_id, _normalize_title, _normalized_text, _note_from_calls, _novelty,
    _number_list, _numeric, _object_among_prose, _object_spans, _objection_key, _ok_call_ids, _ok_calls, _one_line,
    _optim, _optimise, _optimise_refine, _optimise_search, _oracle, _oracle_card, _oracle_challenge, _oracle_triage,
    _oracle_with_reader_flags, _outcome_of, _outside_run, _own_modules, _package_block_text, _page_limit_note, _pairs,
    _panel_draws_nothing, _panel_place, _paper_basis, _paper_body, _paper_copies, _paper_display_title, _paper_facts,
    _paper_figure_count, _paper_for_prompt, _paper_gist, _paper_key, _paper_resolve_link, _paper_scope,
    _paper_sections, _paper_values, _papers_dir_has_files, _papers_next_step, _papers_to_ask,
    _parse_implement_response, _parse_json_lenient, _parse_json_lenient_once, _parse_split_deps, _part_in_layouts,
    _passage_stands, _person_queries, _phased, _phased_isolation, _physical_bound_paths,
    _pick_up_user_dropped_datasets, _place_figures, _places_as_placeholders, _plan, _plan_checks_note, _plan_scope,
    _plan_search_mark, _plan_settings, _plan_steps, _planned_figure_order, _plot_scope, _pool_evidence,
    _precision_target_on, _preliminary_reminders, _printed_in_figure, _prose_value, _protocol, _protocol_directive,
    _provider_is_auth_error, _provider_sign_in_advice, _put_back_paper_copies, _quality_report, _query_set_digest,
    _query_set_key, _query_set_standing, _quest_logger, _quote_in_source, _raise_if_required_skills_unusable,
    _random_source_found, _rank_papers_by_relevance, _read_a_search_block, _read_figure_records, _read_hit_places,
    _read_json_or_none_path, _read_key_set, _read_latex_object, _read_object_keys, _read_primary_figures,
    _read_query_sets, _readable_json_keys, _reading, _reads_replicate_env, _receipts, _record_anchor, _record_claims,
    _record_numbers, _record_query_set, _redrawn_band_text, _ref_citation_text, _reference_entry, _reference_values,
    _refine, _refine_count, _refine_round, _rejudges_oracles_after, _rename_code_names, _render_auto_collected_md,
    _render_data_readme, _renumber_citations, _reorder_literature, _repair_reply_json, _repair_template_values,
    _replicate_assertions, _replicate_env, _replicate_line_figure, _replicate_metric_kinds,
    _replicate_result_intervals, _replicate_seed_count, _replicate_seed_reaches_rng, _replicate_seeds, _rerun_check,
    _rerun_directive, _rerun_from, _resolve_selected_skills, _restore_primary_figures, _result_comparison_stats,
    _results_by_setting, _retractions, _review, _review_answer_problem, _review_decision, _review_items,
    _review_sends_the_experiment_back, _review_was_real, _rng_aliases, _rng_call_is_seeded, _rooted, _run_facts,
    _run_manifest, _run_settings, _same_record_numbers, _same_statement, _save_literature_figures, _scope_cards,
    _screen_grades, _script_has_random_source, _script_reads_replicate_seed, _script_reply,
    _search_record_has_abstract, _second_impl, _seed_label, _seeds_only_constants, _series_at,
    _set_model_call_archive, _sf_snapshot, _shown_reach, _shrink_json_value, _simulatability_answer,
    _single_seed_note, _skeleton, _skill_mount_plan, _slugify, _source_ref, _source_text, _spaced_pattern,
    _spell_out_title_options, _split_figure_readings, _split_run, _stamp_figure_credit, _stats, _step_passages,
    _stop_power, _strip_outer_fence, _strip_plan_numbers, _strip_source_lists, _stripped, _study_size,
    _summarise_long_arrays, _take_out_unpointed, _take_papers_declined, _take_refine_points, _take_refine_why,
    _temperature_for_node, _thin_source, _thinking, _title_only_sentences, _titles_match, _todo,
    _top_level_json_objects, _topic_req, _trial_code, _trial_runner, _trial_table, _trim_further_reading,
    _unchecked_quest_memory, _uncited_foundational_works, _unnamed_curve, _unread_results, _unseeded_rng_calls,
    _unseeded_rng_directive, _unsupported_claim_hit, _unsupported_still_in, _usable_queries,
    _user_feedback_review_block, _walk_paths, _warn_if_unsanctioned_provider, _web_labels_cited, _with_newline,
    _with_reminders, _without_comments, _without_figure_embeds, _without_numbers, _without_reviewer_page_limit_hits,
    _words_to_cut, _write_key_set, _write_paper_need_stubs, _write_personas, _write_query_sets,
    _write_wanted_papers_md, _writes_code, _zero_expected_paths, append_cost_row, ast, asyncio, billing_stop,
    blocked_stop, build_further_reading, build_references, cited_references, claude_cli_account_line, concurrent,
    copy, dataclass, derive_protocol, escape_latex_backslashes, field, fnmatch, forget_papers_asked, functools,
    further_reading_listed, get_args, goal_coverage, hashlib, inspect, interrupt, io, json, langchain_core,
    limit_resumes_at, logging, make_executor, math, mint_quest_id, missing_step_api_keys, model_for_node,
    node_provider_for, os, paper_patch, paper_trim, papers_declined_count, pip_failure_summary, platform, re,
    references_web_listed, render_references_marp_slide, replicate_seed_reaches_rng, resolve_endpoint_async,
    resolve_page_limit, route_for_topic_type, scrub_secrets, shutil, step_provider_config, step_provider_lines,
    string, sys, tempfile, threading, time, tokenize, traceback, unicodedata, unseeded_rng_calls, uuid, warnings,
    web_pages_listed, write_cost_summary, yaml,
)
from .engine_ideas import IdeasStageMixin
from .engine_plan import PlanStageMixin
from .engine_data import DataStageMixin
from .engine_code import CodeStageMixin
from .engine_improve import ImproveStageMixin
from .engine_run import RunStageMixin
from .engine_analyze import AnalyzeStageMixin
from .engine_write import WriteStageMixin
from .engine_review import ReviewStageMixin


class Engine(IdeasStageMixin, PlanStageMixin, DataStageMixin, CodeStageMixin, ImproveStageMixin, RunStageMixin, AnalyzeStageMixin, WriteStageMixin, ReviewStageMixin,
             RunChecksMixin, StaticChecksMixin, ResultChecksMixin, AgentBuildMixin):
    """Owns one quest's research graph, executor, knowledge layer, and LLM client."""

    # Who repairs code a check finds wrong; ``__init__`` sets it from the config. The class default is the chat model, so an
    # Engine built without ``__init__`` (as several tests do) reads it too.
    _repairs_go_to = "chat"

    def __init__(
        self,
        config: Config,
        *,
        supervisor: ProxySupervisor | None = None,
        resume_quest_id: str | None = None,
        auto_accept_on_pass: bool | None = None,
    ) -> None:
        self.config = config
        _warn_if_unsanctioned_provider(config.provider.name)
        # `resume_quest_id` lets a caller re-enter an existing quest
        # (LangGraph's AsyncSqliteSaver keys checkpoints by thread_id,
        # which we set to quest_id below — so reusing the id auto-
        # resumes from the last completed node when a prior run died
        # mid-pipeline, e.g. on a sustained upstream Copilot outage).
        # `FI_PRESEED_QUEST_ID` lets a caller pin the quest_id before
        # `Engine` mints one — used by the `--serve` web UI's quest
        # launcher so the post-submit redirect URL `/quest/<id>` is
        # stable. `resume_quest_id` still wins when both are set
        # because explicit-API beats env-var. Unset env var → original
        # behavior unchanged.
        preseed = os.environ.get("FI_PRESEED_QUEST_ID")
        self.quest_id = (
            resume_quest_id
            or (preseed if preseed and preseed.strip() else None)
            or _new_quest_id(config.title or config.topic)
        )
        # quest_root MUST be absolute. When the config sets a relative
        # `output_dir` (e.g. `./outputs`) and the executor later runs a
        # subprocess with `cwd=quest_root`, an absolute argv path is
        # required — otherwise the relative argv path gets cwd-prefixed
        # by the OS, producing a duplicated nonsense path like
        # `<quest_root>/<quest_root>/code/experiment.py` and the
        # subprocess silently falls back to the SYSTEM Python (because
        # the relative venv-python path also fails to resolve relative
        # to its own cwd). Calling `.resolve()` once here pins the path
        # for every downstream consumer.
        self.quest_root: Path = (config.output.output_dir / self.quest_id).resolve()
        self.fi_dir: Path = self.quest_root / ".fi"
        # Where .fi leads now (it may be kept elsewhere through a link): the improve loop touches its own copies only
        # while .fi still leads here, so a run that moves .fi cannot steer those writes (core/improve.py:own_dir).
        self._fi_real: Path = Path(os.path.realpath(self.fi_dir))  # realpath: a link loop does not raise
        self._clarify_answerable = False
        self.supervisor = supervisor or ProxySupervisor()
        self._log = _quest_logger(self.quest_id, self.fi_dir)
        self.executor = make_executor(
            config.execution.sandbox,
            python_version=config.execution.python_version,
            docker_image=config.execution.docker_image,
            system_site_packages=config.execution.system_site_packages,
            shared_interpreter=config.execution.shared_interpreter,
            docker_limits=DockerLimits.from_config(config.execution),
            log=self._log,  # a container stopped by a limit says so in run.log
        )
        # Every version of code/ that runs is a commit in code/'s history (core/code_project.py:CodeRuns).
        self._code_runs = _code_project.CodeRuns(self.quest_root, node=lambda: self._audit_node, log=self._log)
        self.executor.before_run = self._code_runs.before_run
        self.knowledge = Knowledge(config.knowledge)
        self.audit = _audit_log.AuditLog(self.fi_dir / "audit.jsonl", self.quest_id, enabled=config.engine.audit_trace)
        self._audit_node = ""       # the node running now, for events written from inside it
        self._audit_pause = ""      # the pause kind a node is stopping for
        self._audit_seen: dict[str, str] = {}   # watched file -> sha256 last logged
        self._last_progress = ""    # the last curated progress line, so a re-entered node does not repeat itself
        # Provenance of the most recent ``_chat(node=...)`` call, keyed by that ``node`` string (a chat "kind" like
        # "design" or "design_self_critique", not necessarily the graph node name). ``_audit_claims`` reads this so a
        # model_claim event can say which provider/model actually answered and hash the exact prompt/reply, without
        # threading those values through every node body. Absent (not an error) for nodes that never called ``_chat``
        # under that key — a rule-decided shortcut, an ensemble path, or a node with no model_claim to record.
        self._last_chat: dict[str, dict[str, Any]] = {}
        # Skills other agents installed are read where they are. Which folders
        # is this quest's own setting, held here and passed to every lookup, so
        # quests sharing a process (--fleet) never see one another's. The skill
        # commands (--skills, --approve-skill, ...) build the same object from
        # ``--config <this quest's YAML>``; without it they pass none and see the
        # known folders and the environment override instead.
        from core.skills import ExternalSkillDirs
        self._skill_dirs = ExternalSkillDirs.of(
            config.engine.skills_dirs, scan_known=config.engine.skills_scan_known_dirs,
        )
        self._prompts = _load_prompts()
        # Who repairs code a check finds wrong: the chat model (``chat``), or, with ``engine.code_writer: agent``, the coding
        # agent, to which the checks report instead (core/agent_build.py).
        self._repairs_go_to = "agent" if config.engine.code_writer == "agent" else "chat"
        self._client: LLMClient | None = None
        # Throttle bookkeeping for ``_llm_heartbeat``. Keyed by node
        # name so concurrent ensembled calls don't share one bucket.
        # Reset on every node entry would be nice but isn't needed —
        # any new call's elapsed starts at 0 so the "elapsed - last <
        # interval" check trivially passes.
        self._heartbeat_last_logged: dict[str, float] = {}
        self._heartbeat_log_interval_s: float = 30.0
        # When True, the human-review interrupt auto-resumes with
        # ``accept`` for clean papers (verdict == "accept" AND no
        # must-flag hits). Flagged or revise-verdict papers still
        # pause so a human can read them. Used by ``--fleet
        # --auto-accept-on-pass`` so the production fleet runner
        # doesn't block on every clean quest. The constructor arg
        # wins when explicitly set; otherwise the engine reads
        # ``config.pauses.auto_accept_on_pass`` so YAML / fixtures
        # can configure it without touching the constructor.
        self.auto_accept_on_pass = (
            bool(auto_accept_on_pass)
            if auto_accept_on_pass is not None
            else bool(config.pauses.auto_accept_on_pass)
        )
        # Wall-clock cap on the human-review callback await. 0 = wait
        # forever (legacy). On timeout the run loop falls through to the
        # headless answer-file / pause-exit path so an orphaned UI prompt
        # can't park the quest indefinitely.
        self.human_feedback_timeout_s = float(
            config.pauses.timeout_s
        )

    def __del__(self) -> None:
        try:
            _close_quest_logger(self.quest_id)
        except Exception:
            pass

    async def run(
        self,
        *,
        clarify_callback: ClarifyCallback | None = None,
        human_feedback_callback: "HumanFeedbackCallback | None" = None,
        scope_callback: ClarifyCallback | None = None,
        reopen: bool = False,
        from_step: str | None = None,
        approved_by: str | None = None,
    ) -> QuestArtifacts:
        """Run the quest to terminal state.

        ``from_step`` (see ``core/rerun_from.py``) runs it again from that step. The steps up to the design replace the
        frozen protocol, so they need ``approved_by`` (``--approve-as <you>``); without it nothing is changed.

        ``clarify_callback`` is called only when ``engine.clarify_mode``
        is ``"interactive"`` AND the clarify node fires an
        ``interrupt()``. The callback receives the questions dict and
        must return the answers dict; the engine then resumes the graph
        with ``Command(resume=answers)``. When the callback is None and
        clarify is interactive, the engine raises — set the mode to
        ``"auto"`` or ``"off"`` for headless runs.

        The entire body is wrapped in a try/finally that calls
        ``_close_quest_logger(self.quest_id)`` on every exit path —
        success, error, cancellation, or future pause-exit points. On
        Windows the FileHandler holds an exclusive lock on
        ``run.log``; leaking it broke test cleanup (``shutil.rmtree``
        with ``PermissionError [WinError 32]``) and prevented reusing
        the same ``quest_id`` later in the same process. See
        ``_close_quest_logger`` for the full rationale.
        """
        # ``run_config`` is defined inside the AsyncSqliteSaver block
        # below, but the exception handler at the bottom of this method
        # needs to reference it. Pre-bind to ``None`` so a pre-graph
        # failure (preflight, endpoint resolution, executor.setup) doesn't
        # NameError its way into masking the original exception.
        run_config: dict[str, Any] | None = None
        # This run's automatic retries per step (core/crash_kind.py), and when it began: a failure record written
        # before then is an earlier run's (launch.py prints only this run's).
        self._step_retries: dict[str, tuple[str, int]] = {}
        self._run_started_at = time.time()
        self._clarify_answerable = (
            clarify_callback is not None or (self.fi_dir / "clarify_answer.json").is_file())
        # The study cards have an answer channel of their own (its own files on the web page): the setup questions'
        # channel would hand back the setup answer, and a rerun from the cards would reuse an old pick.
        self._scope_answerer = scope_callback
        import time as _time
        # This run's quest line in .fi/attempts.jsonl is not written yet (a failure after it keeps it).
        self._quest_recorded = False
        self.__dict__.pop("_shadow_closed", None)  # a quest run again records again

        from . import source_failures as _source_failures

        # Every literature / full-text source records its failures against
        # this quest. The adapters run in worker threads with no engine
        # handle; asyncio copies this context into them, so under --fleet each
        # quest still gets only its own. Summarised in the finally below.
        _source_failures.reset(self.quest_id)
        # A new run of this quest may try arXiv again even if an earlier run
        # paused it after repeated rate limits.
        from . import arxiv_gate as _arxiv_gate

        _arxiv_gate.reset_quest(self.quest_id)
        _quest_ctx = _source_failures.current_quest.set(self.quest_id)
        _run_started_at = _time.time()
        # This session's marker (core/run_marker.py): an earlier session that ended without recording an end was stopped
        # from outside, which is said, and its leftover cache locks are cleared. Ended in the finally below.
        _marker = None
        _ended_how = "finished"
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            from .run_marker import RunMarker as _RunMarker

            _marker = _RunMarker(self.quest_root, log=self._log)
            _marker.start()
            (self.quest_root / "figures").mkdir(parents=True, exist_ok=True)
            (self.quest_root / "code").mkdir(parents=True, exist_ok=True)
            (self.quest_root / "paper").mkdir(parents=True, exist_ok=True)
            self._log.info("starting quest %s", self.quest_id)
            _set_model_call_archive(self.fi_dir, bool(self.config.output.save_model_calls))
            # Before anything is written: lines another program added to the quest's records since FI last wrote them
            # are moved aside (core/record_anchor.py), so this run does not chain onto them.
            await self._check_records_changed_outside()
            self._audit("quest_started", resumed=self.audit.event_count() > 0, reopen=bool(reopen), title=self.config.title)
            # Research explores first and confirms once by default (engine.phased); a research quest that began before
            # that default goes on as it began. Decided before the approved settings are compared, so a quest approved
            # with it off is compared as it runs.
            self._phased_keep_off_if_began_before()
            # Research needs a reviewer on another model: checked before the settings are recorded as approved, so the
            # model the person adds is part of what is approved, not a change to it. A model change that leaves every
            # reviewer on one model stops here too.
            stopped = self._review_models_stop()
            if stopped is not None:
                return stopped
            # How strictly this quest is checked was approved on the interview's confirm screen; a hand edit since then
            # (a check turned down, a reviewer dropped) stops here, before anything runs. Another model does not stop
            # it: that is taken, recorded and said (_take_model_change).
            record = self.fi_dir / _plan_settings.NAME
            had_record = record.is_file()
            if not had_record and any(
                e.get("kind") == "plan_settings_recorded" for e in _audit_log.read(self.audit.path)
            ):
                # Deleting the record must not re-approve whatever the config now says: the audit trace (hash-chained)
                # remembers that one was written.
                return self._stop_for_changed_settings([
                    f"the record of the settings this quest was approved with ({record}) is gone, so a change since "
                    "then cannot be told from none",
                ])
            if had_record:
                # The record as it was last approved (its hash is in the hash-chained trace): one edited by hand to
                # match a weaker config agrees with that config, so only its own hash shows the edit. Compared before
                # check() runs, since check() itself rewrites the record when an FI default changed.
                trace_ok = _audit_log.verify(self.audit.path).ok if self.audit.path.is_file() else True
                recorded = [e.get("sha256") for e in _audit_log.read(self.audit.path)
                            if e.get("kind") == "plan_settings_recorded"]
                if recorded and not trace_ok:
                    return self._stop_for_changed_settings([
                        f"the quest's trace ({self.audit.path}) no longer checks out, so the hash of the approved "
                        "settings it holds cannot be trusted; approve the settings again with --update",
                    ])
                if recorded and recorded[-1] != hashlib.sha256(record.read_bytes()).hexdigest():
                    return self._stop_for_changed_settings([
                        f"the record of the settings this quest was approved with ({record}) was changed after it was "
                        "approved; approve the settings again with --update",
                    ])
            before = hashlib.sha256(record.read_bytes()).hexdigest() if record.is_file() else None
            changed = _plan_settings.check(self.quest_root, self.fi_dir, self.config)
            if changed:
                # check() may have rewritten the record for an FI default that moved (not an edit): its hash goes
                # into the trace before the stop, so putting the edited setting back is not then read as a hand edit
                # of the record.
                drifted = hashlib.sha256(record.read_bytes()).hexdigest() if record.is_file() else None
                if before and drifted and drifted != before:
                    self._audit("plan_settings_recorded", sha256=drifted)
                return self._stop_for_changed_settings(changed)
            # A different model is not a change to how the quest is checked: taken as it is, recorded, and said.
            self._take_model_change()
            after =hashlib.sha256(record.read_bytes()).hexdigest() if record.is_file() else None
            # A first record, FI's own rewrite of it (an FI default that changed), or a record from before its hash was
            # kept: its hash goes into the trace now, so an edit from here on is found.
            if after and (after != before or not any(
                e.get("kind") == "plan_settings_recorded" for e in _audit_log.read(self.audit.path)
            )):
                self._audit("plan_settings_recorded", sha256=after)
            # Which interpreter is running FI decides which packages it can
            # see; a `pip install` into a different one changes nothing here.
            self._log.info(
                "[env] python=%s (%s) sandbox=%s",
                sys.executable, sys.version.split()[0], self.config.execution.sandbox,
            )
            # Pre-flight: if the user asked for paper_pdf, verify the
            # host can produce one BEFORE spending 15 minutes on LLM
            # calls only to discover at the end that pandoc / LaTeX are
            # missing. Always warn on missing prereqs; raise only when
            # ``output.require_pdf`` is True. See #55 for the
            # silent-skip incident that motivated both this check and
            # the ``paper_pdf_skipped.md`` diagnostic.
            self._preflight_paper_pdf()
            await self._preflight_required_skills()
            await asyncio.to_thread(self._stage_example_inputs)
            if _phased.enabled(self.config):
                self._audit(_phased.STARTED_EVENT)  # this quest ran with it on (phased.ran_without)
                await asyncio.to_thread(self._phased_prepare)
            elif _phased.record_path(self.quest_root).exists():
                await asyncio.to_thread(self._phased_turned_off)
            if (not _phased.enabled(self.config) and self.config.rigor_profile == "research"
                    and not getattr(self, "_phased_kept_off", False)):
                # A research quest that turned explore-then-confirm off: said plainly (plan.md says it too).
                self._log.info("[phased] %s", _phased.off_sentence(self._runs_code_by_config(),
                                                                    self._phased_no_experiment_reason()))
            await self.executor.setup(self.quest_root)
            await self._record_environment()

            await self._connect_llm()

            checkpoint_path = self.fi_dir / "state.sqlite"
            try:
                async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
                    graph = self._build_graph().compile(checkpointer=saver)
                    initial: QuestState = {
                        "topic": self.config.topic,
                        "title": self.config.title or _slugify(self.config.topic)[:60],
                        "iteration": 0,
                    }
                    run_config = {"configurable": {"thread_id": self.quest_id}}

                    # Resume detection: if the thread already has checkpointed
                    # state (a prior run died mid-pipeline), pass `None` to
                    # ainvoke so LangGraph continues from the last-completed
                    # node instead of replaying from `ideate` with the
                    # current YAML's topic. Without this, --resume only
                    # reused the quest_id; LangGraph still treated the
                    # `initial` payload as a fresh START.
                    prior_snapshot = await graph.aget_state(run_config)
                    if getattr(prior_snapshot, "values", None):
                        # A resumed quest: files an older FI left in code/tests/ with expected values in them go first.
                        stale = _equation_tests.remove_stale(self.quest_root / "code")
                        if stale:
                            self._log.info("[run] removed files an older FI left in code/ with expected values in "
                                           "them: %s", ", ".join(stale))
                        # The values an earlier run delivered for a person's tests: gone while code runs again.
                        _code_layout.remove_check_values(self.quest_root / "code")
                    # `values` is empty dict for never-run threads.
                    payload: Any
                    # ``--from <step>``: the first call continues from the checkpoint taken just before that step; every
                    # call after it (a resume after a pause) continues the thread as it now is.
                    fork_config: dict[str, Any] | None = None
                    if from_step:
                        fork_config = await _rerun_from.checkpoint_before(graph, run_config, from_step)
                        if fork_config is None:
                            self._log.warning("[run] --from %s: this quest never reached that step", from_step)
                            print(f"[FI] quest {self.quest_id} never reached the {from_step} step, so there is nothing "
                                  f"to rerun from there; nothing was changed. Resume it with "
                                  f"`python launch.py --resume {self.quest_id}`; "
                                  f"`--resume {self.quest_id} --from` with no step lists the steps it did reach.")
                            return self._collect_artifacts({})
                        replaced_sha = ""
                        if _rerun_from.needs_approval(from_step):
                            if not (approved_by or "").strip():
                                self._log.warning("[run] --from %s replaces the frozen protocol and needs --approve-as", from_step)
                                mine = "plan.md and its " if "plan.md" in _rerun_from.outputs_of(from_step) else ""
                                print(f"[FI] running quest {self.quest_id} again from the {from_step} step replaces its {mine}"
                                      f"experiment plan (frozen) (the old ones are kept in .fi/previous/). Nothing was "
                                      f"changed. To go ahead: --resume {self.quest_id} --from {from_step} "
                                      f"--approve-as <you>")
                                return self._collect_artifacts({})
                            frozen_file = _frozen.frozen_path(self.quest_root)
                            if frozen_file.is_file():
                                replaced_sha = hashlib.sha256(frozen_file.read_bytes()).hexdigest()
                            fork_config = await self._keep_design_history(graph, fork_config, from_step)
                            _frozen.record_replacement(self.quest_root, approved_by=approved_by.strip(), step=from_step)
                        if reopen:
                            self._log.info("[run] --from %s is used; --rerun's re-opening at the review is not", from_step)
                        forget_papers_asked(self.fi_dir, declined=False)
                        if _rerun_from.picks_skills(from_step):
                            fork_config = await self._repick_skills(graph, fork_config)
                        try:
                            await asyncio.to_thread(self._improve_undo_unfinished)
                        except Exception as exc:  # noqa: BLE001 -- the rerun goes on; the log says what was not undone
                            self._log.warning("[improve] could not undo a loop cut short before the rerun: %r", exc)
                        try:
                            where, moved = _rerun_from.back_up(self.quest_root, from_step)
                        except OSError as exc:
                            self._log.warning("[run] --from %s: moving the earlier outputs failed (%s); they were put back", from_step, exc)
                            print(f"[FI] could not move the earlier outputs of quest {self.quest_id} aside ({exc}); they were put "
                                  f"back where they were and the quest was not run again. Close whatever holds those files "
                                  f"and try `--resume {self.quest_id} --from {from_step}` again.")
                            return self._collect_artifacts({})
                        if not (_rerun_from.needs_approval(from_step) or _rerun_from.picks_skills(from_step)):
                            # Make the fork the thread's newest checkpoint now (a copy of it, about to run the same
                            # step), as the steps above already do by writing state: LangGraph saves nothing until the
                            # step's first node finishes, so until then the quest's map read the old line's end and
                            # showed a quest whose outputs were just moved aside as finished (and did so for good when
                            # the run was stopped or failed in that node).
                            fork_config = await graph.aupdate_state(fork_config, None, as_node="__copy__")
                        extra_audit: dict[str, Any] = {}
                        if _rerun_from.needs_approval(from_step):
                            extra_audit = {"approved_by": approved_by, "replaced_protocol_sha256": replaced_sha}
                            print(f"[FI] the old plan.md and experiment plan (frozen) are replaced; they are kept in "
                                  f"{where.relative_to(self.quest_root).as_posix() if where else '.fi/previous/'}")
                        self._audit("rerun_from", step=from_step, moved=moved, **extra_audit)
                        self._log.info("[run] rerunning from the %s step; the earlier outputs are in %s (%s)",
                                       from_step, where or "(nothing to move)", ", ".join(moved) or "none")
                        self._progress(f"Rerunning from the {from_step} step; the earlier outputs are kept in "
                                       f"{where.relative_to(self.quest_root).as_posix() if where else 'nothing to keep'}")
                    if fork_config is not None:
                        payload = None
                    elif prior_snapshot and prior_snapshot.values:
                        self._log.info(
                            "[run] found checkpoint with keys=%s next=%s — resuming",
                            sorted((prior_snapshot.values or {}).keys()),
                            prior_snapshot.next,
                        )
                        payload = None
                        # Re-open a FINISHED quest (next=()) for another pass:
                        # re-enter at design via the refine route so it re-runs
                        # write/review and lands back at the human-review gate,
                        # applying any saved feedback_history. A no-op if the
                        # quest isn't terminal (resume continues normally).
                        #
                        # The injected ``human_feedback`` carries action=refine
                        # with NO feedback text on purpose: re-opening is "run
                        # another pass", not a new user ask. The design node
                        # already reads the accumulated ``feedback_history``
                        # (so a refine dropped past max_iterations finally
                        # applies); a synthetic feedback string would instead
                        # be embedded as spurious USER FEEDBACK whenever the
                        # history is empty. We also bump ``iteration`` to mirror
                        # the real refine path (``_node_human_feedback``), which
                        # consumes a loop slot and guarantees the prior review
                        # JSON is embedded (the design node skips it at
                        # iteration==0).
                        if reopen and not prior_snapshot.next:
                            forget_papers_asked(self.fi_dir, declined=False)
                            self._set_aside_stuck_record()
                            _it = int((prior_snapshot.values or {}).get("iteration", 0))
                            # The notes of earlier refines were answered by earlier passes: this pass redoes the
                            # design, and neither the writer nor the review treats them as new.
                            _answered = _refine_count(prior_snapshot.values or {})  # type: ignore[arg-type]
                            await graph.aupdate_state(
                                run_config,
                                {
                                    "human_feedback": {"action": "refine"},
                                    # The earlier accept was of the earlier pass: this one needs its own.
                                    "acceptance": {},
                                    "iteration": _it + 1,
                                    "refine_written_for": _answered,
                                    "feedback_rounds_from": len((prior_snapshot.values or {}).get("feedback_history") or []),
                                    # A request to search further (core/optimise_refine.py) belongs to its own pass.
                                    "optimise_refine": {},
                                    # An honest stop belongs to the pass that ended in it.
                                    "stuck": {},
                                },
                                as_node="human_feedback",
                            )
                            reopened = await graph.aget_state(run_config)
                            self._log.info(
                                "[run] re-opened finished quest for another pass "
                                "(next=%s)", reopened.next,
                            )
                    else:
                        payload = initial

                    # Clear any stale ``quest_failed.md`` NOW, at the start of
                    # this run — not only on the pause/completion paths below.
                    # A resume that's still in-flight otherwise keeps the prior
                    # run's failure diagnostic on disk for its whole (often
                    # many-minute) duration, so the dashboard / quest page shows
                    # the quest as "failed" even though it's actively running
                    # and has already moved past the node that broke. If THIS
                    # run also fails, the exception handler writes a fresh one;
                    # a fresh START has no file, so this is a no-op there.
                    self._clear_stale_quest_failed_diagnostic()
                    # Same fix for the pause-display markers: a refine/resume
                    # that continues past a pause must not leave the prior
                    # pause's "needs you" markers on disk, or the dashboard
                    # shows "human review" / "user input" for the whole run.
                    # (Answer files are kept — they're this run's input.)
                    self._clear_stale_pause_markers()
                    # This run's model calls belong to the record the quest will seal again when it finishes.
                    try:
                        (self.fi_dir / _attempts.MODEL_CALLS_CLOSED).unlink(missing_ok=True)
                    except OSError:
                        pass

                    # Run, handling interrupts as they fire. Two kinds:
                    #   (a) clarify-interactive — pause to collect answers
                    #       via clarify_callback, then resume the graph.
                    #   (b) wait_for_data (no-simulation mode) — pause and
                    #       EXIT cleanly with rc=0. User drops files
                    #       into <quest_root>/data/, then re-runs
                    #       `fi --resume <quest_id>` which lands here
                    #       again — at which point _node_wait_for_data
                    #       sees the files and proceeds without pausing.
                    data_paused = False
                    while True:
                        self._invoke_started_at = time.time()
                        try:
                            final_state = await graph.ainvoke(payload, config=fork_config or run_config)
                        except (KeyboardInterrupt, SystemExit, GeneratorExit, asyncio.CancelledError):
                            raise
                        except Exception as step_exc:  # noqa: BLE001 -- sorted below; anything else is re-raised
                            if not await self._try_step_again(graph, run_config, step_exc):
                                raise
                            # Continue from the last finished step (the checkpoint), as ``--resume`` would.
                            payload = None
                            fork_config = None
                            continue
                        fork_config = None
                        interrupts = (final_state or {}).get("__interrupt__")
                        if not interrupts:
                            break
                        intr_value = interrupts[0].value or {}
                        if intr_value.get("data_required"):
                            # no-simulation pause-exit. State is already
                            # checkpointed; the next `fi --resume` will
                            # re-enter wait_for_data and proceed.
                            data_paused = True
                            data_dir = intr_value.get(
                                "data_dir", str(self.quest_root / "data"),
                            )
                            self._log.info(
                                "[FI] paused for user data: drop files "
                                "into %s then run `fi --resume %s`",
                                data_dir, self.quest_id,
                            )
                            break
                        if intr_value.get("user_input_required"):
                            # Generic pause-drop-anytime. Pause-exit
                            # clean rc=0; user drops files into
                            # inputs/{papers,data}/ then ``fi --resume``.
                            data_paused = True
                            stage = intr_value.get("stage", "?")
                            inputs_dir = intr_value.get(
                                "inputs_dir",
                                str(self.quest_root / "inputs"),
                            )
                            self._log.info(
                                "[FI] paused at stage=%s for user input: drop "
                                "files into %s/{papers,data}/ then run "
                                "`fi --resume %s`",
                                stage, inputs_dir, self.quest_id,
                            )
                            break
                        if intr_value.get("papers_required"):
                            # literature node's pause-for-user-papers
                            # gate fired. Same pause-exit semantics as
                            # the data_required path: exit rc=0 cleanly,
                            # user drops PDFs into ``inputs/papers/``
                            # and runs ``fi --resume``.
                            data_paused = True
                            papers_dir = intr_value.get(
                                "papers_dir",
                                str(self.quest_root / "inputs" / "papers"),
                            )
                            self._log.info(
                                "[FI] paused for user papers: drop PDFs into "
                                "%s then run `fi --resume %s`",
                                papers_dir, self.quest_id,
                            )
                            break
                        if intr_value.get("plan_stage") or intr_value.get("protocol_stage") or intr_value.get("oracle_stage") or intr_value.get("numeric_stage") or intr_value.get("amendment_stage") or intr_value.get("contract_stage"):
                            # The plan step wrote plan.md and stopped for the
                            # person to read and edit it. Same pause-exit as
                            # the pauses above; without this branch the payload
                            # fell through to the clarify handling below and
                            # left a clarify_questions.json that made the quest
                            # page ask for clarify answers.
                            data_paused = True
                            if intr_value.get("contract_stage"):
                                self._log.info(
                                    "[FI] paused for the %s (%s): read %s, fix what it names, then run `fi --resume %s`",
                                    "equation labels in the simulation"
                                    if (intr_value.get("pause") or {}).get("kind") == "equation_labels"
                                    else "package of equations, unit tests and equation list in code/"
                                    if intr_value.get("layout")
                                    else "two-script contract",
                                    "; ".join(intr_value.get("problems") or [])[:200],
                                    self.quest_root / "NEXT_STEP.md", self.quest_id,
                                )
                                break
                            if intr_value.get("amendment_stage"):
                                self._log.info(
                                    "[FI] paused for the protocol amendment: read %s, approve it with `--approve-amendment \"%s\" "
                                    "--approve-as <you>`, then run `fi --resume %s` (resuming without approving keeps the "
                                    "frozen protocol)",
                                    intr_value.get("amendment_file", "needs/PROTOCOL_AMENDMENT_PENDING.json"),
                                    self.quest_root, self.quest_id,
                                )
                                break
                            if intr_value.get("oracle_stage"):
                                # Not "edit plan.md": the script may have crashed, and the check may be fine. The
                                # card (NEXT_STEP.md, and printed when the run ends) says which, with the numbers.
                                self._log.info(
                                    "[FI] paused: %s (%s). See NEXT_STEP.md, then run `fi --resume %s`",
                                    (intr_value.get("pause") or {}).get("headline")
                                    or "the known-answer checks did not pass",
                                    "; ".join(intr_value.get("problems") or [])[:240], self.quest_id,
                                )
                                break
                            self._log.info(
                                "[FI] paused for the %s: read and edit %s "
                                "(or ask for a change with `--revise-plan`), "
                                "then run `fi --resume %s`",
                                "run's numeric warnings" if intr_value.get("numeric_stage")
                                else "protocol" if intr_value.get("protocol_stage")
                                else "checks FI added to the plan" if intr_value.get("oracles_added") else "plan",
                                intr_value.get("plan_file", "plan.md"),
                                self.quest_id,
                            )
                            break
                        descriptor = intr_value.get("pause") if isinstance(intr_value.get("pause"), dict) else {}
                        if descriptor.get("interaction") == "supply" and "human_review" not in intr_value:
                            # Any other stop that needs something done before a resume (a model that can read
                            # images, a check that could not judge): exit cleanly; NEXT_STEP.md says what to do, and
                            # the resume re-enters the node that stopped. Without this branch such a stop fell through
                            # to the clarify handling below and asked the person for answers to no questions.
                            data_paused = True
                            self._log.info(
                                "[FI] paused: %s. See NEXT_STEP.md, then run `fi --resume %s`",
                                descriptor.get("headline") or descriptor.get("kind") or "action needed", self.quest_id,
                            )
                            break
                        # human_feedback node raised `interrupt(...)`.
                        # Three resolution paths, in order:
                        #   1. ``--auto-accept-on-pass`` AND the paper
                        #      is clean (verdict=accept AND no
                        #      must_flag_hits) → resume with accept
                        #      automatically, no user interaction.
                        #   2. ``human_feedback_callback`` wired → call
                        #      it (CLI --interactive, web in-process,
                        #      VSCode bridge).
                        #   3. No callback → check
                        #      ``<quest_root>/.fi/human_review_answer.json``
                        #      for a pre-staged answer (the file the
                        #      web POST endpoint writes). If present,
                        #      consume and resume; otherwise pause-exit
                        #      cleanly with rc=0 — the user finishes the
                        #      review off-line and re-runs
                        #      ``fi --resume <quest_id>``.
                        if "human_review" in intr_value:
                            snap = intr_value["human_review"]
                            verdict = snap.get("verdict")
                            snapshot_path = self.fi_dir / "human_review.json"
                            answer_path = self.fi_dir / "human_review_answer.json"

                            def _consume_snapshot() -> None:
                                # Remove the on-disk snapshot when the gate
                                # resolves so the dashboard's "snapshot
                                # present + no answer-file" pending check
                                # doesn't continue to show a stale banner.
                                # Best-effort: an unlink failure is not
                                # quest-fatal.
                                for p in (snapshot_path, answer_path):
                                    try:
                                        p.unlink()
                                    except OSError:
                                        pass

                            if self.auto_accept_on_pass and _auto_accepts(snap):
                                self._log.info(
                                    "[run] human_feedback auto-accept "
                                    "(verdict=accept, no must_flag_hits)",
                                )
                                _consume_snapshot()
                                # No person looked: recorded as automatic, which keeps the evidence one level
                                # below publication_ready (core/acceptance.py).
                                payload = Command(
                                    resume={"action": "accept", "feedback": "",
                                            "acceptance": _acceptance.automatic("auto_accept_on_pass")},
                                )
                                continue
                            if human_feedback_callback is not None:
                                self._log.info(
                                    "[run] human_feedback interrupt fired (verdict=%s); "
                                    "invoking callback", verdict,
                                )
                                t = self.human_feedback_timeout_s
                                try:
                                    if t and t > 0:
                                        answer = await asyncio.wait_for(
                                            human_feedback_callback(snap), t,
                                        )
                                    else:
                                        answer = await human_feedback_callback(snap)
                                    if _review_decision(answer):
                                        _consume_snapshot()
                                        # A person answered: an accept carries who, how and their answer.
                                        payload = Command(resume=_acceptance.stamp(answer, "callback", snap))
                                        continue
                                    # No decision in the answer (``{}``, ``None``, a missing or unknown action, an
                                    # accept without the answer to the question, or "no"): stop and ask again rather
                                    # than guess. Resuming with ``{}`` re-fired this pause forever (LangGraph reads it
                                    # as "resume nothing") and ``None`` crashed the run.
                                    self._log.warning(
                                        "[run] %s — the quest stops here; NEXT_STEP.md says how to decide",
                                        _review_answer_problem(answer),
                                    )
                                except BridgeError as e:
                                    # The person closed the VS Code review prompt without choosing: stop and wait
                                    # for a decision, never take it as "accept".
                                    self._log.warning(
                                        "[run] the review prompt was closed without a decision (%s) — the quest "
                                        "stops here; NEXT_STEP.md says how to decide", e,
                                    )
                                except asyncio.TimeoutError:
                                    # Orphaned UI prompt (e.g. a VSCode QuickPick
                                    # tied to a stale chat turn) — stop waiting and
                                    # fall through to the answer-file / pause-exit
                                    # path so the quest checkpoints and can resume
                                    # rather than blocking forever.
                                    self._log.warning(
                                        "[run] human_feedback callback timed out after "
                                        "%ss — falling back to answer-file / pause-exit",
                                        t,
                                    )
                            if answer_path.is_file():
                                try:
                                    answer = json.loads(
                                        answer_path.read_text(encoding="utf-8"),
                                    )
                                except (OSError, json.JSONDecodeError) as e:
                                    self._log.warning(
                                        "[run] couldn't read %s: %r — pausing instead",
                                        answer_path, e,
                                    )
                                    answer = None
                                if _review_decision(answer):
                                    self._log.info(
                                        "[run] consuming pre-staged human-review answer "
                                        "(action=%s)", answer.get("action"),
                                    )
                                    _consume_snapshot()
                                    payload = Command(resume=_acceptance.stamp(answer, "answer file", snap))
                                    continue
                                if answer is not None:
                                    # A staged answer with no decision in it is not "accept"; drop it so the next
                                    # resume does not read it again, and say why.
                                    self._log.warning(
                                        "[run] %s: %s — ignoring it; the quest stops here", answer_path,
                                        _review_answer_problem(answer),
                                    )
                                    try:
                                        answer_path.unlink(missing_ok=True)
                                    except OSError:
                                        pass
                            data_paused = True
                            self._log.info(
                                "[FI] paused for human review. Decide with ONE command:\n"
                                "      accept:  python launch.py --config <yaml> --resume %s --accept yes"
                                "   (have you reviewed the evidence record, and do you accept these claims and the "
                                "limits listed? yes / partly \"what you do not accept\" / not-checked)\n"
                                "      reject:  python launch.py --config <yaml> --resume %s --reject\n"
                                "      refine:  python launch.py --config <yaml> --resume %s --refine \"your feedback\"\n"
                                "      (or, in the web UI / VSCode, click Accept / Reject / Refine.)",
                                self.quest_id, self.quest_id, self.quest_id,
                            )
                            break
                        # Clarify node raised `interrupt(...)`. Resolve answers
                        # via (1) the in-process callback, (2) a pre-staged
                        # on-disk answer file, or (3) clean pause-exit — the
                        # same three-path pattern the human-review gate uses, so
                        # a subprocess- / CLI-launched quest (no callback) pauses
                        # cleanly instead of crashing.
                        questions = intr_value.get("clarify_questions", {})
                        clarify_answer_path = self.fi_dir / "clarify_answer.json"
                        if clarify_callback is not None:
                            try:
                                self._log.info(
                                    "[run] clarify interrupt fired with %d questions; "
                                    "invoking callback", len(questions),
                                )
                                if self.human_feedback_timeout_s > 0:
                                    answers = await asyncio.wait_for(
                                        clarify_callback(questions),
                                        timeout=self.human_feedback_timeout_s,
                                    )
                                else:
                                    answers = await clarify_callback(questions)
                                self._clear_clarify_snapshot()
                                # LangGraph reads an empty dict as an (empty) map of interrupt ids, so nothing is
                                # resumed and the pause re-fires forever; an empty answer is wrapped and the node
                                # fills in the defaults.
                                payload = Command(resume={"clarify_answers": answers or {}})
                                continue
                            except asyncio.TimeoutError:
                                if _json_file_is_answer(clarify_answer_path):
                                    # An answer written just as the wait ran out is used below, never dropped.
                                    pass
                                elif self.config.pauses.clarify is None:
                                    # Nobody set "ask": an unattended start must not stall, so answer for itself.
                                    self._log.warning(
                                        "[run] nobody answered the setup questions within %ss — using the defaults",
                                        self.human_feedback_timeout_s,
                                    )
                                    self._clear_clarify_snapshot()
                                    payload = Command(resume={"clarify_answers": {}})
                                    continue
                                self._log.warning(
                                    "[run] clarify callback timed out after %ss — "
                                    "falling back to answer-file / pause-exit",
                                    self.human_feedback_timeout_s,
                                )
                            except BridgeError as e:
                                # The person dismissed the VS Code prompt: carry on with the defaults.
                                self._log.warning("[run] setup questions dismissed (%s) — using the defaults", e)
                                self._clear_clarify_snapshot()
                                payload = Command(resume={"clarify_answers": {}})
                                continue
                        if clarify_answer_path.is_file():
                            try:
                                answers = json.loads(
                                    clarify_answer_path.read_text(encoding="utf-8"),
                                )
                            except (OSError, json.JSONDecodeError) as e:
                                # A corrupt answer file would re-fail on every
                                # resume and strand the quest — drop it so the
                                # user can re-stage a fresh answer.
                                self._log.warning(
                                    "[run] couldn't read %s: %r — discarding it "
                                    "and pausing instead", clarify_answer_path, e,
                                )
                                try:
                                    clarify_answer_path.unlink(missing_ok=True)
                                except OSError:
                                    pass
                                answers = None
                            if isinstance(answers, dict) and answers:
                                self._log.info(
                                    "[run] consuming pre-staged clarify answers",
                                )
                                self._clear_clarify_snapshot()
                                payload = Command(resume=answers)
                                continue
                        # Genuine pause (no callback, no staged answer): persist
                        # the questions so the web can render a form for this
                        # subprocess quest. Only here — never on the resume path
                        # above — so a consumed answer isn't re-shown as pending.
                        try:
                            self.fi_dir.mkdir(parents=True, exist_ok=True)
                            (self.fi_dir / "clarify_questions.json").write_text(
                                json.dumps(questions, indent=2) + "\n",
                                encoding="utf-8",
                            )
                        except OSError as e:
                            self._log.debug(
                                "[clarify] questions snapshot write failed: %r", e,
                            )
                        data_paused = True
                        self._log.info(
                            "[FI] paused for clarify. Answer in the web / VSCode "
                            "panel, or write your answers into %s and re-run "
                            "`fi --resume %s`.",
                            clarify_answer_path, self.quest_id,
                        )
                        break
                    # No more of the quest's code runs in this call: the values its tests compare with are written now.
                    await self._deliver_check_values(final_state if isinstance(final_state, dict) else {})
            finally:
                # Guard against early failure: if ``resolve_endpoint_async``
                # raised (unknown provider / proxy spawn failure / etc.),
                # ``self._client`` is still None and the original exception
                # is what the caller should see. Unconditionally calling
                # ``aclose()`` on None would raise AttributeError and mask
                # the real error. Same logic for the proxy release —
                # only release a handle we actually acquired.
                await self._close_step_clients()
                if self._client is not None:
                    await self._client.aclose()
                if (
                    self.config.provider.name in PROXY_PROVIDERS
                    and self._client is not None
                ):
                    await self.supervisor.release(self.config.provider.name)
                # Release proxies for any fallback providers that a
                # FallbackLLMClient actually materialised this run.
                for fb_name in getattr(
                    self._client, "built_fallback_providers", (),
                ):
                    if (
                        fb_name in PROXY_PROVIDERS
                        and fb_name != self.config.provider.name
                    ):
                        await self.supervisor.release(fb_name)

            if data_paused:
                # Generic pause-exit. The flag is shared across four
                # interrupt kinds: ``wait_for_data`` (no-simulation
                # missing data), ``papers_required`` (literature pause
                # for user papers), ``user_input_required`` (the
                # pause-drop-anytime gate), and ``human_review``
                # (after-review human gate, when no callback is wired).
                # All of them want the same cleanup: skip
                # ``_write_back_knowledge`` (the quest hasn't been
                # accepted), return a partial QuestArtifacts so callers
                # (launch.py / the VSCode bridge) can surface the
                # "drop files here" / "respond and resume" message, and
                # exit rc=0 cleanly. The specific message has already
                # been logged by the interrupt handler above; this is
                # just the shared "we're pausing" notice.
                self._log.info(
                    "quest %s paused — exiting clean (rc=0)",
                    self.quest_id,
                )
                # Resume-from-failure clears the stale diagnostic too.
                # If the prior run wrote ``quest_failed.md`` and the
                # current resume got far enough to reach wait_for_data,
                # the prior failure was recovered — leaving the file
                # would mislead ("paused for data, but also failed?").
                self._clear_stale_quest_failed_diagnostic()
                self._write_evidence(final_state)
                self._record_stop(final_state, await self._attempt_context(final_state, kind="in_progress"))
                return self._collect_artifacts(final_state)

            artifacts = self._collect_artifacts(final_state)
            # Finishing is two steps: everything the quest keeps is written first (the evidence, the attempt records),
            # then the seal naming their hashes is the trace's last event. The evidence is assessed before the seal it
            # cannot yet see; readers check that seal (core/evidence.py::verify_seal).
            evidence_record = self._write_evidence(final_state, sealing=True)
            end_context = await self._attempt_context(final_state, kind="quest_end")

            def _quest_record() -> dict[str, Any]:
                review = final_state.get("review") if isinstance(final_state.get("review"), dict) else {}
                human = final_state.get("human_feedback") if isinstance(final_state.get("human_feedback"), dict) else {}
                return {
                    "kind": "quest",
                    **_attempts.quest_status(
                        dict(final_state), evidence_record,
                        reviewer_accepted=review.get("verdict") == "accept" and _review_was_real(review),
                        no_experiment=bool(getattr(self.config.engine, "survey_mode", False)
                                           or final_state.get("survey_mode_resolved")),
                        data_analysis=bool(final_state.get("no_simulation_resolved")),
                    ),
                    "review": review.get("verdict"),
                    # What a person decided: an accept made automatically is no person's (core/acceptance.py).
                    "person": (None if human.get("action") == "accept"
                               and (final_state.get("acceptance") or {}).get("by") != "person"
                               else human.get("action")),
                    # As the evidence record says it (no review pause, or another paper, counts as automatic).
                    "accepted_by": (evidence_record or {}).get("accepted_by"),
                    # This run's run records (a resumed quest's earlier ones are in the file before it).
                    "run_record_ids": list(getattr(self, "_run_record_ids", [])),
                    "records_not_written": _attempts.lost(self.fi_dir),
                    "answered_by": self._answering_model(),
                    "context": end_context,
                }
            self._record(_attempts.ATTEMPTS, _quest_record)
            self._quest_recorded = True
            # Completion path only (NOT the pause-exit above): the quest reached
            # its terminal node, so every pause it raised has been resolved —
            # clear the unified NEXT_STEP.md + pause.json + any ANSWER-pause
            # snapshots so a finished quest never shows a stale "Action needed"
            # (a clarify node re-executes on resume, re-writing its snapshot).
            for p in (
                self.quest_root / "NEXT_STEP.md",
                self.fi_dir / "pause.json",
                self.fi_dir / "clarify_questions.json",
                self.fi_dir / "clarify_answer.json",
                self.fi_dir / self._CLARIFY_ASKED,
                # The human-review snapshot too — the node re-writes it on the
                # resume pass, so a finished quest could otherwise look like it's
                # still waiting for review on the dashboard / quest page.
                self.fi_dir / "human_review.json",
                self.fi_dir / "human_review_answer.json",
            ):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            # Nothing may be written to the trace after this.
            self._shadow_close()
            self._seal_trace(final_state)
            if evidence_record is not None:
                sealed = _evidence.verify_seal(self.quest_root, evidence_record)
                if sealed.get("status") != evidence_record.get("status"):
                    self._log.warning("[evidence] after sealing: %s (the seal did not check out: %s)",
                                      _evidence.summary_line(sealed, technical=True),
                                      "; ".join((sealed.get("all_gaps") or {}).get("publication_ready", [])[-3:]))
                evidence_record = sealed
            # Nothing is paused now; what did not stop the quest but is worth a look stays on the card (.fi/todo.json).
            _todo.write(self.quest_root, self.fi_dir, self.quest_id, None)
            self._write_back_knowledge(artifacts, final_state, (evidence_record or {}).get("status"))
            self._record_skill_usage(final_state)
            self._write_cost_summary()
            # Clean up any stale ``quest_failed.md`` from a PRIOR
            # failed run of this quest — the current run succeeded,
            # so leaving the old diagnostic on disk would mislead the
            # user into thinking the just-completed quest broke.
            # Same idempotent-cleanup pattern as the paper generator
            # uses for ``paper_pdf_skipped.md`` on a successful PDF
            # compile.
            self._clear_stale_quest_failed_diagnostic()
            # Reclaim ~150-250 MB per quest by freezing the venv to
            # ``.fi/requirements.lock.txt`` then deleting ``.venv/``.
            # Only fires on the success path — failed/paused quests
            # keep their venv so the user can poke at it. Best-effort:
            # any exception here is swallowed so a venv-cleanup hiccup
            # never masks a successful quest.
            try:
                await self.executor.cleanup_after_success(self.quest_root)
            except Exception as cleanup_exc:
                self._log.warning(
                    "venv cleanup failed (quest still succeeded): %r",
                    cleanup_exc,
                )
            self._log.info("quest %s reached terminal state", self.quest_id)
            return artifacts
        except (KeyboardInterrupt, SystemExit, GeneratorExit, asyncio.CancelledError):
            # Someone stopped the quest on purpose — Ctrl-C, a shutdown
            # signal, a cancelled task. That is not "the quest broke", and
            # writing a failure report for it would tell the user their run
            # crashed when they are the one who ended it. Propagate
            # untouched, exactly as the literature fetch boundary in
            # ``core/knowledge.py`` does with the same four.
            #
            # Listed FIRST so the deliberately wider arm below never sees
            # one of them: order is the whole mechanism here.
            _ended_how = "stopped"
            raise
        except BaseException as exc:  # noqa: BLE001 - deliberately wider; see below
            # Surface the failure as a quest-directory diagnostic the
            # user can discover by opening the quest folder, rather than
            # leaving an empty quest dir whose only breadcrumb is a
            # traceback buried in ``<quest_root>/.fi/launch.log``. Mirrors the
            # ``paper_pdf_skipped.md`` contract from the paper generator.
            #
            # Wider than ``Exception`` on purpose, and for the same reason
            # ``core/knowledge.py``'s fetch boundary is: a quest reaches
            # out through a Playwright-driven Node process, and when that
            # driver dies the failure does not always arrive as a
            # well-behaved Python exception — the sync API drives its pipe
            # through greenlets, so what reaches this frame can be a
            # ``BaseException`` subclass. That fetch boundary CONTAINS the
            # shape for the sources it wraps, so a dying render is one
            # failed source and never reaches here; this arm is for the
            # same shape arriving from anywhere else. Under the old
            # ``except Exception`` such a failure skipped the diagnostic
            # entirely and the user was left with an empty quest folder
            # whose only breadcrumb was a traceback in ``.fi/launch.log``.
            # Cancellation keeps its quiet path in the arm above.
            #
            # Re-raise unconditionally — this handler is for diagnostics
            # only, NOT for swallowing errors. The caller (launch.py)
            # still surfaces the exception in stderr / its own exit code,
            # and swallowing a ``BaseException`` would be worse still.
            #
            # The diagnostic-write itself is wrapped in its own
            # try/except: a failure to write the diagnostic must NEVER
            # mask the original exception (the user wants to see the
            # real error, not "could not open file for diagnostic
            # writing").
            _unusable = exc if isinstance(exc, ProviderUnusable) else None
            if _unusable is not None:
                # A call made outside any step (the run's own, before or after the graph): the same plain stop. It is an
                # ``Exception`` from here on, so the fleet and the web and VS Code runs treat it as any quest that failed.
                exc = self._unusable_stop(_unusable)  # its cause is the provider's error: quest_failed.md is written from it
            _ended_how = f"failed: {type(exc).__name__}"
            if isinstance(exc, ModelUnavailable):
                self._progress(str(exc))  # the one plain sentence: what stopped the quest and how to continue
            try:
                self._log.error("[run] the quest stopped on an error", exc_info=exc)
            except Exception:  # noqa: BLE001 -- the log must never hide the error
                pass
            try:
                await self._write_quest_failed_diagnostic(exc, run_config)
            except Exception as diag_err:
                # Best-effort logging only — re-raising the original
                # exception is the contract.
                self._log.warning(
                    "[run] could not write quest_failed.md: %r", diag_err,
                )
            # The failure is on the to-do card (the run's start cleared the last one). Never in the way of the error.
            try:
                _todo.write(self.quest_root, self.fi_dir, self.quest_id, None)
            except Exception:  # noqa: BLE001
                pass
            # A quest already recorded as finished that fails in its clean-up keeps that record.
            if not getattr(self, "_quest_recorded", False):
                # The conditions as last worked out while the quest ran; only when none were, a context without the
                # quest's state (marked incomplete), and never one that holds up the error for long.
                fail_context = getattr(self, "_last_attempt_context", None)
                if not fail_context:
                    try:
                        fail_context = await asyncio.wait_for(
                            self._attempt_context({"topic": self.config.topic}, kind="in_progress", partial=True), timeout=30)
                    except Exception:  # noqa: BLE001 -- never in the way of the error
                        fail_context = None
                self._record(_attempts.ATTEMPTS, lambda: {
                    "kind": "quest", "execution_status": "crashed", "review_status": "none",
                    "evidence_status": "unknown", "claim_outcome": None, "error": type(exc).__name__,
                    "failure_signature": _memory.failure_signature(error=type(exc).__name__),
                    "run_record_ids": list(getattr(self, "_run_record_ids", [])),
                    "records_not_written": _attempts.lost(self.fi_dir),
                    "answered_by": self._answering_model(),
                    "context": fail_context,
                })
                self._audit("attempts_sealed", files=_attempts.file_digests(self.fi_dir))
            if _unusable is not None:
                raise exc from _unusable.error
            raise
        finally:
            self._shadow_close()
            # Outer cleanup: releases the per-quest run.log FileHandler
            # on EVERY exit path — normal completion, exception from
            # ``graph.ainvoke``, missing-callback RuntimeError, errors
            # in ``_collect_artifacts`` / ``_write_back_knowledge``,
            # and the not-yet-landed Phase-B no-simulation pause-exit.
            # Without this, Windows test cleanup would intermittently
            # fail with PermissionError as soon as ANY of those paths
            # fired. ``_close_quest_logger`` is idempotent.
            #
            # The source-failure summary goes first, while run.log is still
            # open, and on every path: a quest that failed or paused because
            # its sources did is exactly the one whose summary matters.
            try:
                self._emit_source_failure_summary(started_at=_run_started_at)
            except Exception:  # noqa: BLE001 - reporting must not mask the outcome
                pass
            # The quality report (QUALITY.md, .fi/quality.json): where the quest went wrong, who caught it, what was
            # fixed and what it cost, read from the records above. Every way a run ends passes here: finished, an
            # honest stop, a pause, a crash. Never in the way of the outcome.
            _quality_report.write_quietly(self.quest_root)
            try:
                _source_failures.current_quest.reset(_quest_ctx)
            except ValueError:
                pass
            # Explore, then confirm: outside a running quest inputs/data/ holds the person's whole files (the next
            # start holds the same rows back again). Also when it was turned off: a part must never stay in place.
            if _phased.enabled(self.config) or _phased.record_path(self.quest_root).exists():
                self._phased_restore_inputs()
            if _marker is not None:
                try:
                    _marker.finish(_ended_how)
                except Exception:  # noqa: BLE001 -- the marker is a record; never in the way of the outcome
                    pass
            _close_quest_logger(self.quest_id)

    def _emit_source_failure_summary(self, *, started_at: float) -> None:
        """One ``[source-failures]`` line in run.log plus
        ``.fi/source_failures.json`` for this run. A clean run says ``none``:
        silence would be indistinguishable from the report being skipped."""
        from . import source_failures as _source_failures

        try:
            payload = _source_failures.write_summary(
                self.quest_id, self.fi_dir / "source_failures.json",
                started_at=started_at,
            )
        except OSError as e:
            self._log.warning("[source-failures] could not write summary: %r", e)
            payload = _source_failures.snapshot(self.quest_id)
            payload["summary"] = _source_failures.format_summary(payload)
        emit = self._log.warning if payload.get("total") else self._log.info
        emit("[source-failures] %s", payload["summary"])

    async def emit_artifacts_only(self) -> "QuestArtifacts":
        """Load a FINISHED quest's checkpoint READ-ONLY and bundle its
        artifacts — WITHOUT re-running any node. Used by the on-demand
        ``--emit``/Generate-artifacts path: the caller then runs only the
        requested generator (PDF/slides/poster/speech) off the existing
        ``paper.md`` + figures, so a quest that finished as markdown-only can
        produce more formats without re-doing the research.

        Mirrors the read-only snapshot pattern the quest-failed resolver
        uses — open a fresh saver context purely to read the StateSnapshot.
        Returns an artifacts bundle built from whatever the checkpoint holds;
        raises ``FileNotFoundError`` if there is no checkpoint to read.
        """
        _set_model_call_archive(self.fi_dir, bool(self.config.output.save_model_calls))
        checkpoint_path = self.fi_dir / "state.sqlite"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"no checkpoint at {checkpoint_path} — nothing to emit"
            )
        run_config = {"configurable": {"thread_id": self.quest_id}}
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            graph = self._build_graph().compile(checkpointer=saver)
            snap = await graph.aget_state(run_config)
        state = dict((snap.values if snap else None) or {})
        return self._collect_artifacts(state)

    # ---- audit trace (core/audit_log.py) ---------------------------------
    #
    # ``<quest>/.fi/audit.jsonl``: what ran, in order, with each check's verdict, each route's reasons and the model's own
    # stated rationale (provenance ``model_claim``, never mixed with a check). A record, not a control: nothing here can stop a
    # quest.

    # Files whose bytes are worth a sha256 in the trace, hashed after every node and logged when they change.
    _AUDIT_CACHE_BYTES = 8 * 1024 * 1024
    _AUDIT_WATCHED = (
        "plan.md", "code/simulate.py", "code/experiment.py", "code/submit.py", "needs/FROZEN_PROTOCOL.json",
        "paper/paper.md", "paper/paper.pdf",
        # The records the evidence level is read from: a change to one after it was written shows in the trace.
        ".fi/approved_plan.json", "needs/receipts/evidence_gate.json", "needs/receipts/design_audit.json",
        "needs/receipts/claim_check.json", "needs/ENVIRONMENT.json", "needs/RUN_MANIFEST_CHECK.json",
        "needs/ORACLE_CHECK.json", "needs/PROTOCOL_CHECK.json", "raw/ledger.jsonl", "raw/trials.json",
        "needs/HIDDEN_CHECK.json", ".fi/oracle_review.json",
        ".fi/trials/run.json", "paper/claims.json",
        # FI's record of a search for the best design (core/optimise.py).
        "raw/optimisation_ledger.jsonl", "results/best_design.json", ".fi/optimisation/run.json",
        # FI's check of the best design at finer numerical settings (core/optimum_check.py).
        "needs/OPTIMUM_CHECK.json", ".fi/optimisation/check.json",
        # The rounds of a continued search a person asked for (core/optimise_refine.py).
        ".fi/optimisation/continue.json",
        # What the quest tried (core/attempt_records.py): anchored in the trace, so an edit after the fact shows.
        ".fi/attempts.jsonl", ".fi/branch_ledger.jsonl",
        # The search queries used: a change between steps shows in the trace, the seal covers the rest.
        ".fi/literature_queries.json",
    )

    async def _check_records_changed_outside(self) -> None:
        """Compare the quest's record files with FI's note of how far it wrote them and its copy in the last checkpoint
        (core/record_anchor.py): what FI did not write is moved aside (never used), said in run.log and recorded in the
        trace; the evidence then reads the result as not publication-ready."""
        try:
            found = await asyncio.to_thread(_record_anchor.check_on_start, self.fi_dir)
        except Exception as e:  # noqa: BLE001 -- a check that cannot run is said, and the quest goes on
            self._log.warning("[records] FI could not compare the quest's records with its own note of them: %r", e)
            return
        for f in found:
            self._log.warning("[records] %s", f.line())
            self._audit(_record_anchor.EVENT, file=f.file, reason=f.reason, lines=f.lines, moved_to=f.moved_to)
        if found:
            moved = any(f.moved_to for f in found)
            self._progress("The quest's records were changed outside FI since it last ran" +
                           (": what FI did not write was set aside and is not used" if moved else "") +
                           ", so the result will not count as publication-ready (see run.log).")

    def _audit(self, kind: str, *, node: str | None = None, provenance: str = _audit_log.DETERMINISTIC, **fields: Any) -> None:
        try:
            self.audit.append(kind, node=node or self._audit_node, provenance=provenance, **fields)
        except Exception as e:  # noqa: BLE001 -- the trace is a record; it must never stop a quest
            self._log.debug("[audit] could not record %s: %r", kind, e)

    async def _flag_sources(self, items: list[Any], stage: str, *, record_clean: bool = False) -> None:
        """Flag text in these sources that is hidden from a reader or addressed to an AI model (core/source_text.py):
        kept, marked in the prompts, recorded once per source in the audit trace and run.log. In a thread: a source's
        whole text can be 10 MB, and a --fleet shares the event loop."""
        await asyncio.to_thread(
            _source_text.flag_and_record, items, stage=stage, audit=self._audit, log=self._log,
            content_of=_item_content, record_clean=record_clean,
            reported=self.__dict__.setdefault("_sources_flagged", set()),
        )

    def _seal_trace(self, state: QuestState) -> None:
        """The last event of a finished quest (``quest_finalized``), written after everything else the quest keeps:
        how many events the trace holds before it, how many events and attempt records could not be written, which
        steps completed, and the hashes of the evidence record, the attempt records and the paper. Under
        ``rigor_profile: research`` a trace without it as its last event, or whose named files changed since, is a gap
        (core/evidence.py::verify_seal): a valid prefix is not a complete record."""
        try:
            events = _audit_log.read(self.audit.path)
        except Exception:  # noqa: BLE001 -- a trace that cannot be read gets no seal, which is itself the gap
            return
        # The quest's own paper (where _collect_artifacts finds it), not a path the state kept from before a move.
        paper_rel = "paper/paper.md"
        try:
            paper_sha = hashlib.sha256((self.quest_root / paper_rel).read_bytes()).hexdigest()
        except OSError:
            paper_sha = None
        self._seed_call_counts(state)
        counts = dict(self.__dict__.get("_model_call_counts") or {})
        for rel in _evidence.SEALED_LEDGERS:
            # A record the quest never wrote to is an empty file with a hash, not a name the seal leaves out. Not the
            # record of model calls of a quest that made calls: that one missing is itself the gap.
            path = self.quest_root / rel
            if path.exists() or (rel.endswith(_attempts.MODEL_CALLS) and sum(counts.values())):
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
            except OSError:
                pass
        # The record of the search queries likewise: an empty one when no query was ever derived (every derivation,
        # failed or not, leaves an entry); missing after an answered derivation, it is the gap.
        if not (self.fi_dir / _LITERATURE_QUERIES).exists() and not (
                _ok_call_ids(self.fi_dir, "literature_query") or _ok_call_ids(self.fi_dir, "ideate_query")):
            _write_query_sets(self.fi_dir, [])
        call_gaps, call_summary = _attempts.model_call_gaps(
            self.quest_root, {n: dict(v) for n, v in dict(getattr(self, "_last_chat", {}) or {}).items()}, counts)
        files = {}
        for rel in _evidence.SEALED_FILES:
            try:
                files[rel] = hashlib.sha256((self.quest_root / rel).read_bytes()).hexdigest()
            except OSError:
                files[rel] = None
        self._audit(
            "quest_finalized",
            events_before=len(events),
            write_errors=max(int(getattr(self.audit, "write_errors", 0) or 0), _audit_log.lost_writes(self.audit.path)),
            records_not_written=_attempts.lost(self.fi_dir),
            model_calls_not_written=_attempts.lost(self.fi_dir, _attempts.MODEL_CALLS_LOST),
            # Kept here, not only in the evidence record the seal protects: whether the seal must be checked.
            rigor_profile=self.config.rigor_profile,
            nodes_completed=sorted({str(e.get("node")) for e in events if e.get("kind") == "node_completed" and e.get("node")}),
            paper_path=paper_rel,
            paper_sha256=paper_sha,
            files=files,
            sealed_records=_evidence.SEAL_RECORDS,
            model_calls={"lines": call_summary.get("lines", 0), "counts": counts, "gaps": call_gaps},
        )
        # From here, a model call (an output made from the finished quest) goes to a separate record the seal does not
        # name; a new run of the quest removes this and seals again.
        try:
            (self.fi_dir / _attempts.MODEL_CALLS_CLOSED).write_text("sealed\n", encoding="utf-8")
        except OSError:
            pass

    def _audit_artifacts(self, node: str) -> None:
        # A large file (the trial ledger) whose size and modification time are those it had when last hashed is not
        # read again, since this runs after every step; a small one is hashed every time, so an edit that keeps its size
        # and resets its time still shows.
        stats = self.__dict__.setdefault("_audit_stat", {})
        for rel in self._AUDIT_WATCHED:
            path = self.quest_root / rel
            try:
                stat = path.stat()
                mark = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                mark = None
            if mark is not None and mark[0] >= self._AUDIT_CACHE_BYTES and rel in self._audit_seen and stats.get(rel) == mark:
                continue
            digest = _audit_log.file_sha256(path) if path.is_file() else None
            if digest is not None:
                stats[rel] = mark
            if digest is None or self._audit_seen.get(rel) == digest:
                continue
            self._audit_seen[rel] = digest
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            self._audit("artifact_created", node=node, path=rel, sha256=digest, bytes=size)

    def _audit_check(self, check: str, path: Path, *, status: str, summary: str = "", problems: Any = None) -> None:
        """One check's verdict, with the record that holds the detail and its sha256."""
        try:
            rel = path.relative_to(self.quest_root).as_posix()
        except ValueError:
            rel = str(path)
        found = [str(p) for p in problems] if isinstance(problems, (list, tuple)) else []
        self._audit(
            "check_result", check=check, status=status, summary=summary or (found[0] if found else ""),
            problems=found[:5], record=rel, sha256=_audit_log.file_sha256(path),
        )

    def _pop_rationale_into_claims(self, node: str, design: dict[str, Any]) -> dict[str, Any]:
        """``design`` without its ``rationale`` block (if it has one), after turning it into ``model_claim`` events under
        ``node``. Shared by the ``design`` node (via :meth:`_audit_claims`, working from its returned state patch) and
        ``plan`` (called inline — a plan-drafted design is written straight to ``plan.md`` and never returned in the
        state patch at all, so ``_audit_claims`` never sees it there; without this, its rationale block reached the
        persisted file verbatim instead of the audit trace, since ``plan`` asks the same design prompt as ``design``
        and can get the same block back)."""
        if "rationale" not in design:
            return design
        design = dict(design)
        why = design.pop("rationale")
        if isinstance(why, dict):
            prov = self._chat_provenance(node)
            for a in why.get("assumptions") or []:
                self._audit("model_claim", node=node, provenance=_audit_log.MODEL_CLAIM, topic="assumption", claim=a, **prov)
            for alt in why.get("alternatives_considered") or []:
                if isinstance(alt, dict):
                    self._audit(
                        "model_claim", node=node, provenance=_audit_log.MODEL_CLAIM, topic="alternative",
                        claim=alt.get("option", ""), decision=alt.get("decision", ""), reason=alt.get("reason", ""),
                        **prov,
                    )
        return design

    def _audit_claims(self, node: str, out: Any) -> Any:
        """Turn what a model said about why into ``model_claim`` events: the design's ``rationale`` (which is then dropped from
        the design in the state, so the later prompts do not carry it), a review's verdict and weaknesses, the evidence
        gate's verdict and gaps (only when a model actually decided it — see below), and analyze's stated reason for its
        ``next_step``. Every event is merged with :meth:`_chat_provenance` for the same key, so a reader can see which
        provider/model produced the claim and hash the exact prompt/reply it came from, not just that some call did."""
        if not isinstance(out, dict):
            return out
        if node == "design" and isinstance(out.get("design"), dict) and "rationale" in out["design"]:
            out = {**out, "design": self._pop_rationale_into_claims(node, out["design"])}
        if node == "review" and isinstance(out.get("review"), dict):
            review = out["review"]
            prov = self._chat_provenance("review")
            # The reviewer's one-sentence reason for its verdict: recorded here, then dropped from the state (as the
            # design's rationale is), so the later prompts that read the review do not carry it.
            why = review.get("why")
            if "why" in review:
                review = {k: v for k, v in review.items() if k != "why"}
                out = {**out, "review": review}
            if review.get("verdict"):
                self._audit(
                    "model_claim", node=node, provenance=_audit_log.MODEL_CLAIM, topic="review_verdict",
                    claim=str(review.get("verdict")) + (f": {review['blocking']}" if review.get("blocking") else ""),
                    **({"reason": str(why).strip()} if isinstance(why, str) and why.strip() else {}),
                    **prov,
                )
            for w in review.get("weaknesses") or []:
                self._audit("model_claim", node=node, provenance=_audit_log.MODEL_CLAIM, topic="review_weakness", claim=w, **prov)
            # What this round said, kept apart from the trace: when no round is left after it, the paper states its points.
            _disclosure.write_last_review(self.quest_root, review)
            # A review panel's moderator says why the panel's verdict stands, given how the reviewers disagreed.
            if isinstance(review.get("rationale"), str):
                self._claim(node, "review_moderator", "review_moderator", review["rationale"],
                            decision=str(review.get("verdict") or ""))
        try:
            self._audit_stated_reasons(node, out)
        except Exception as e:  # noqa: BLE001 -- the trace is a record; an answer of an odd shape must never stop a quest
            self._log.debug("[audit] reasons not recorded for %s: %r", node, e)
        if node == "evidence_gate" and isinstance(out.get("evidence_assessment"), dict):
            assessment = out["evidence_assessment"]
            # A rule-decided verdict (``core.engine._evidence_gate_rule``) never called a model — recording it as a
            # model_claim would misattribute an engine decision to the model, which audit_log's own contract forbids
            # (provenance ``model_claim`` is never mixed with a check/rule result). An "unknown" status has no real
            # rationale to record either way (a parse failure or a provider error, not a considered verdict).
            if assessment.get("decided_by") == "model" and assessment.get("status") == "ok":
                prov = self._chat_provenance("evidence_gate")
                rationale = str(assessment.get("rationale") or "").strip()
                if rationale:
                    self._audit(
                        "model_claim", node=node, provenance=_audit_log.MODEL_CLAIM, topic="verdict",
                        claim=rationale, decision=str(assessment.get("verdict") or ""), **prov,
                    )
                for gap in assessment.get("gaps") or []:
                    self._audit("model_claim", node=node, provenance=_audit_log.MODEL_CLAIM, topic="gap", claim=gap, **prov)
        if node == "analyze" and isinstance(out.get("analysis"), dict):
            analysis = out["analysis"]
            reason = str(analysis.get("next_step_reason") or "").strip()
            if reason:
                self._audit(
                    "model_claim", node=node, provenance=_audit_log.MODEL_CLAIM, topic="next_step",
                    claim=reason, decision=str(analysis.get("next_step") or ""), **self._chat_provenance("analyze"),
                )
        return out

    def _claim(self, node: str, key: str, topic: str, claim: Any, **fields: Any) -> None:
        """One ``model_claim`` under ``node``: ``claim`` (the model's own words, cut to a readable length) with the
        provenance of the call it came from (``key``, the ``_chat`` node key). Nothing is recorded for an empty claim
        or when no model call was made under ``key`` (a rule, a person, or a path that asked no model)."""
        text = " ".join(str(claim or "").split())
        prov = self._chat_provenance(key)
        if not text or not prov:
            return
        # Only a call made during this run of the step: a reason carried over in the state from an earlier run is not
        # recorded again under a call that did not write it.
        # (``_chat`` stores a new record for each answered call, so the same record as at the step's start means no
        # call under ``key`` answered during this run; a failed call leaves the old record in place.)
        before = self.__dict__.get("_chat_at_node_start")
        if isinstance(before, dict) and self._last_chat.get(key) is before.get(key):
            return
        extra = {k: (" ".join(str(v).split())[:600] if isinstance(v, str) else v) for k, v in fields.items()
                 if v not in (None, "", [], {})}
        self._audit("model_claim", node=node, provenance=_audit_log.MODEL_CLAIM, topic=topic, claim=text[:1200],
                    **extra, **prov)

    def _audit_stated_reasons(self, node: str, out: dict[str, Any]) -> None:
        """The reasons a model already wrote in its answer at the steps that choose between options, as ``model_claim``
        events (no extra call, no extra tokens): the idea it chose (and, when they ran, what the pairwise comparison
        and the self-critique said), the skills it picked or declined, how each finding sits against the literature,
        what a repair of the script changed and why (or why it gave up), and the reasons of the setup questions'
        answers it chose. Each carries the provenance of the call that wrote it. A reason is the model's own account,
        not evidence."""
        if node == "ideate":
            chosen = out.get("chosen_idea") if isinstance(out.get("chosen_idea"), dict) else {}
            ideas = [i for i in out.get("ideas") or [] if isinstance(i, dict)]
            options = [str(i.get("title") or "") for i in ideas if i.get("title")]
            tournament = out.get("ideate_tournament") if isinstance(out.get("ideate_tournament"), dict) else {}
            critique = out.get("ideate_critique") if isinstance(out.get("ideate_critique"), dict) else {}
            first = str(chosen.get("rationale") or "")
            decision = chosen.get("title")
            if tournament:
                first = first.split("[tournament]", 1)[0]  # the tournament's reasons follow, under their own call
                if tournament.get("outcome") == "swapped":
                    decision = "its first pick (the pairwise comparison then chose another idea)"
            elif critique.get("swap_to") or critique.get("refined_rationale"):
                first = ""  # the self-critique replaced it; its own words are recorded under its call below
            self._claim(node, "ideate", "idea_chosen", first, decision=decision,
                        options=options if len(options) > 1 else None)
            winner = tournament.get("winner_idx")
            for m in tournament.get("matches") or []:
                if not isinstance(m, dict):
                    continue
                won = m.get("a_idx") if m.get("winner") == "A" else m.get("b_idx") if m.get("winner") == "B" else None
                if winner is not None and won == winner:
                    self._claim(node, "ideate_tournament", "idea_compared", m.get("reason"),
                                decision=chosen.get("title"), margin=m.get("margin"))
            if critique:
                self._claim(node, "ideate_reflect", "idea_reconsidered",
                            critique.get("refined_rationale") or critique.get("strongest_objection"),
                            decision=(f"switched to {critique['swap_to']}" if critique.get("swap_to")
                                      else "kept the first choice"))
        elif node == "select_skills":
            record = out.get("skill_selection") if isinstance(out.get("skill_selection"), dict) else {}
            forced = set(record.get("forced") or [])  # required by the config: the engine's decision, not the model's
            for name, why in (record.get("reasons") or {}).items():
                if name not in forced:
                    self._claim(node, "select_skills", "skill_chosen", why, decision=f"use {name}")
            for name, why in (record.get("declined") or {}).items():
                self._claim(node, "select_skills", "skill_declined", why, decision=f"do not use {name}")
        elif node == "cross_check":
            for entry in out.get("cross_check") or []:
                if not isinstance(entry, dict):
                    continue
                key = "cross_check_verify" if "first_pass" in entry else "cross_check"
                finding = str(entry.get("finding") or "")[:200]
                candidates = entry.get("candidates") or []
                verdicts = [b for b in ("supporting", "conflicting") if entry.get(b)]
                # a result FI held no value of was not searched: the trace must not read that as a check that found nothing
                self._claim(node, key, "literature_verdict", entry.get("summary"), finding=finding,
                            decision="not checked" if entry.get("status") == "not_checked"
                            else ", ".join(verdicts) or "neutral")
                for bucket in ("supporting", "conflicting"):
                    for item in entry.get(bucket) or []:
                        if not isinstance(item, dict):
                            continue
                        idx = item.get("index")
                        title = (candidates[idx - 1].get("title") if isinstance(idx, int) and 0 < idx <= len(candidates)
                                 and isinstance(candidates[idx - 1], dict) else "") or f"source {idx}"
                        self._claim(node, key, f"literature_{bucket}", item.get("why"), finding=finding,
                                    decision=f"{bucket}: {title}")
        elif node == "execute_reflect":
            # The engine's own placeholders are in parentheses ("(no summary)", "(LLM produced no patched code)"):
            # never recorded as the model's words. A give-up the run-record check turned into a stop is only in the
            # history, as "(gave up: <the model's reason>)".
            give_up = str(out.get("exec_give_up_reason") or "")
            written = out.get("exec_reflect_history") or []
            entry = written[-1] if written and isinstance(written[-1], dict) else None
            attempt = (entry or {}).get("iter", out.get("exec_reflect_iter"))
            summary = str((entry or {}).get("patch_summary") or "")
            if give_up and not give_up.startswith("("):
                self._claim(node, "execute_reflect", "repair", give_up, decision="gave up")
            elif summary.startswith("(gave up: "):
                body = summary[len("(gave up: "):]
                self._claim(node, "execute_reflect", "repair", body[:-1] if body.endswith(")") else body,
                            decision="gave up")
            elif summary and not summary.startswith("(") and not give_up:
                self._claim(node, "execute_reflect", "repair", summary,
                            decision=f"repaired the script (attempt {attempt})")
        elif node == "clarify":
            questions = out.get("clarify_questions") if isinstance(out.get("clarify_questions"), dict) else {}
            answers = out.get("clarify_answers") if isinstance(out.get("clarify_answers"), dict) else {}
            for slot, q in questions.items():
                if not isinstance(q, dict) or not isinstance(q.get("reason"), str):
                    continue
                answered = answers.get(slot)
                self._claim(node, "clarify", f"setup/{slot}", q["reason"], decision=str(q.get("default") or ""),
                            **({"answer_used": str(answered)} if answered not in (None, "") and str(answered) != str(q.get("default")) else {}))

    def _progress(self, text: str) -> None:
        """One line for whoever is watching the quest, not debugging it: the CLI console, VS Code's chat, and the web
        page's live log show only lines logged this way (``core/engine.py:STAGE_PROGRESS``, ``_ProgressOnly``); a repeat
        of the same line (a node re-entered after a pause, say) is silently dropped rather than shown twice."""
        if text == self._last_progress:
            return
        self._last_progress = text
        self._log.info(text, extra={"progress": True})

    def _progress_stage(self, node: str) -> None:
        phrase = STAGE_PROGRESS.get(node)
        if phrase:
            self._progress(f"[{node}] {phrase}")

    def _sign_in_message(self, exc: BaseException, requested_model: str | None = None) -> str:
        """The plain sign-in message for a call that a provider refused (which provider and model, what to run)."""
        served = _crash_kind.served_by(exc)
        prov = getattr(getattr(self, "config", None), "provider", None)
        name = served[0] if served else str(getattr(prov, "name", "") or "")
        model = served[1] if served else str(requested_model or getattr(prov, "model", "") or "")
        return _provider_sign_in_advice(name, model)

    def _unavailable_message(self, gone: str = "", limit: BaseException | None = None) -> str:
        """What a stop because the model could not be used says, in plain words: a dead sign-in says how to sign in
        again; a usage limit says which provider and model, when it ends if the provider said so, and how to continue.
        No step id and none of the provider's own text (its links and codes)."""
        signin = self.__dict__.pop("_provider_signin", "")
        self.__dict__.pop("_provider_limit", None)
        if signin:
            return (f"{signin} Nothing the current step decided was kept. After signing in, continue the quest with "
                    f"`--resume {self.quest_id}` and it runs this step again.")
        served = _crash_kind.served_by(limit) if limit is not None else None
        prov = getattr(getattr(self, "config", None), "provider", None)
        name = (getattr(limit, "provider", "") or (served[0] if served else "") or str(getattr(prov, "name", "") or ""))
        model = (getattr(limit, "model", "") or (served[1] if served else "") or str(getattr(prov, "model", "") or ""))
        who = ", ".join(x for x in (name, model) if x)
        when = limit_resumes_at(str(limit) if limit is not None else "") or limit_resumes_at(gone)
        resume = f"python launch.py --config {(self.quest_root / 'config.yaml').as_posix()} --resume {self.quest_id}"
        empty = billing_stop(limit, name) if limit is not None else None
        if empty:  # an empty balance is not a usage limit: no wait brings it back
            return f"{empty[0]}. Nothing the current step decided was kept. {empty[1]}, then resume with: {resume}"
        blocked = blocked_stop(limit) if limit is not None else None
        if blocked:  # refused before the service: not a usage limit and not a sign-in
            return f"{blocked[0]}. Nothing the current step decided was kept. {blocked[1]}: {resume}"
        return (f"The model provider's usage limit was reached{f' ({who})' if who else ''}. Nothing the current step decided "
                f"was kept. " + (f"Resume after {when} with: {resume}" if when else
                                 f"Resume with: {resume} once the limit has reset"))

    def _unavailable_stop(self, gone: str = "", error: BaseException | None = None) -> "ModelUnavailable":
        """The quest's stop for a model that cannot be used, with the provider's own error as its cause (so the failure
        card and quest_failed.md read the limit from it)."""
        limit = error if error is not None else self.__dict__.get("_provider_limit")
        stop = ModelUnavailable(self._unavailable_message(gone, limit))
        stop.__cause__ = limit
        return stop

    def _unusable_stop(self, u: ProviderUnusable, node: str = "") -> "ModelUnavailable":
        """The stop for a ``ProviderUnusable`` (the one plain sentence)."""
        self.__dict__.pop("_provider_unavailable", None)
        return self._unavailable_stop(error=u.error)

    def _audited(self, name: str, fn: Any) -> Any:
        """``fn`` (a graph node) with its start, end, failure or pause in the audit trace. An interrupted node is run again
        from its start when the quest resumes, so a second ``node_started`` after ``node_paused`` is what happened."""

        @functools.wraps(fn)
        async def wrapper(state: QuestState) -> Any:
            # The quest root is FI's own fact: every node (a resumed one too) sees it, whatever its state holds.
            state = {**state, "quest_root": str(self.quest_root)}  # type: ignore[assignment]
            self._seed_call_counts(state)
            self.audit.event_count()      # open the chain first: after a restart it says which node was waiting for a person
            resumed = self.audit.paused_node == name
            self._audit_node, self._audit_pause = name, ""
            self._audit("node_started", node=name, iteration=state.get("iteration", 0), **({"resumed": True} if resumed else {}))
            self.__dict__["_chat_at_node_start"] = dict(self._last_chat)
            self._progress_stage(name)
            try:
                self._heal_frame(state)
            except Exception:  # noqa: BLE001 -- a repair of the frame's list must never stop a node
                pass
            began = time.monotonic()
            self.__dict__.pop("_provider_unavailable", None)
            self.__dict__.pop("_provider_signin", None)
            self.__dict__.pop("_provider_limit", None)
            try:
                try:
                    out = await fn(state)
                except _ModelAnswerProblem as e:
                    # An answer the model did not finish (or the provider withheld) stops for a person, who can give
                    # the step a larger output limit or another model; the step runs again on resume.
                    self._pause_for_model_output(name, e)
                    raise  # unreachable: the pause interrupts
                if gone := self.__dict__.pop("_provider_unavailable", None):
                    # A call in this step met a usage limit or a dead login and the step went on without its answer
                    # (a "could not ask" path): what it decided is not kept, and the step runs again on resume.
                    raise self._unavailable_stop(gone)
            except GraphBubbleUp as e:
                if isinstance(e, GraphInterrupt) and (gone := self.__dict__.pop("_provider_unavailable", None)):
                    # A pause that follows a call its model could not take (r4: "the settings to vary could not be
                    # read" after the plan's model hit its usage limit) asks a person about a question never answered:
                    # the step stops instead, and runs again on resume.
                    error = self._unavailable_stop(gone)
                    self._audit("node_failed", node=name, error=f"{type(error).__name__}: {error}")
                    raise error
                if isinstance(e, GraphInterrupt):
                    self._audit("node_paused", node=name, pause=self._audit_pause)
                raise
            except asyncio.CancelledError:
                self._audit("node_failed", node=name, error="cancelled")
                raise
            except ProviderUnusable as u:
                # A call of the step met a limit or a refused sign-in. Nothing caught it on the way (it is not an
                # ``Exception``): the quest stops with the plain sentence, the step is not kept, and it runs again on resume.
                error = self._unusable_stop(u, name)
                self._audit("node_failed", node=name, error=f"{type(error).__name__}: {error}")
                raise error from u.error
            except Exception as e:  # noqa: BLE001 -- recorded, then raised again unchanged
                self._audit("node_failed", node=name, error=f"{type(e).__name__}: {e}")
                raise
            try:
                out = self._audit_claims(name, out)
            except Exception as e:  # noqa: BLE001 -- the trace is a record; an odd answer never fails a finished step
                self._log.debug("[audit] claims not recorded for %s: %r", name, e)
            self._audit_artifacts(name)
            self._audit(
                "node_completed", node=name, duration_s=round(time.monotonic() - began, 2),
                wrote=sorted(out) if isinstance(out, dict) else [],
            )
            counts = dict(self.__dict__.get("_model_call_counts") or {})
            if isinstance(out, dict) and counts != dict(state.get("model_call_counts") or {}):
                out = {**out, "model_call_counts": counts}  # kept in the state, so a resumed quest keeps its tally
            if isinstance(out, dict):
                anchor = _record_anchor.snapshot(self.fi_dir)  # in the checkpoint: checked when the quest starts again
                if anchor and anchor != state.get("record_anchor"):
                    out = {**out, "record_anchor": anchor}
            return out

        return wrapper

    def _route_facts(self, source: str, state: QuestState) -> dict[str, Any]:
        """The values a router read to choose, for the trace."""
        review = state.get("review") or {}
        result = state.get("exec_result") or {}
        analysis = state.get("analysis") or {}
        facts: dict[str, dict[str, Any]] = {
            "design": {
                "no_simulation": bool(state.get("no_simulation_resolved")), "survey_mode": bool(state.get("survey_mode_resolved")),
            },
            "execute_reflect": {
                "returncode": result.get("returncode", 0), "has_result_json": bool(state.get("result_json")),
                "attempt": _numeric.repairs_made(state), "gave_up": state.get("exec_give_up_reason") or "",
                "patch_pending": bool(state.get("exec_patch_pending")),
            },
            "cross_check": {
                "next_step": analysis.get("next_step", "publish"), "iteration": state.get("iteration", 0),
                "max_iterations": self.config.engine.max_iterations, "findings_checked": len(state.get("cross_check") or []),
            },
            "evidence_gate": {k: v for k, v in (state.get("evidence_assessment") or {}).items() if isinstance(v, (str, int, float, bool))},
            "review": {
                "verdict": review.get("verdict", ""), "must_flag_hits": list(review.get("must_flag_hits") or []),
                "iteration": state.get("iteration", 0), "max_iterations": self.config.engine.max_iterations,
                "pauses_review": self.config.pauses.review,
            },
            "human_feedback": {"action": (state.get("human_feedback") or {}).get("action", "accept")},
            "write": {
                "refine_scope": state.get("refine_scope") or "",
                "needs_experiment": list(state.get("refine_needs_experiment") or []),
                "extend": list(state.get("refine_extend") or []),
                "layout": list(state.get("refine_layout") or []),
            },
        }
        return facts.get(source, {})

    def _audited_route(self, source: str, fn: Any) -> Any:
        """``fn`` (a conditional edge) with the route it chose and the facts it read in the audit trace."""

        @functools.wraps(fn)
        def wrapper(state: QuestState) -> str:
            chosen = fn(state)
            try:
                facts = self._route_facts(source, state)
            except Exception:  # noqa: BLE001 -- a record must not change a route
                facts = {}
            self._audit("route_decision", node=source, chosen=chosen, facts=facts)
            return chosen

        return wrapper

    async def _keep_design_history(self, graph: Any, fork_config: dict[str, Any], step: str) -> dict[str, Any]:
        """A restart at or before the design goes back to a checkpoint from before the design history existed, and
        possibly from a later round. The history on disk (``needs/DESIGN_HISTORY.json``) is put back into the state
        first, so the new design is recorded after the old ones as a change made after results were seen, never as a
        fresh start; a history that is missing or unreadable for a quest that did design counts as one earlier design.
        The round counter goes back to 0 and the design is cleared, so the plan is written again (or read again, from
        ``design``) as on a first pass."""
        path = self.quest_root / "needs" / "DESIGN_HISTORY.json"
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        except (OSError, ValueError):
            on_disk = []
        if not isinstance(on_disk, list):
            on_disk = []
        snapshot = await graph.aget_state(fork_config)
        in_state = list((snapshot.values or {}).get("design_history") or [])
        history = on_disk if len(on_disk) > len(in_state) else in_state
        if not history and "design" in await _rerun_from.reached(graph, {"configurable": {"thread_id": self.quest_id}}):
            history = [{"revision": 0, "iteration": 0, "post_hoc": False, "hypothesis": "",
                        "reason": "an earlier design existed, but its record was missing or unreadable"}]
        return await graph.aupdate_state(
            fork_config, {"design_history": history, "iteration": 0, "design": None},
            as_node=_rerun_from.LEADS_INTO[step])

    async def rerun_steps(self) -> list[dict[str, Any]]:
        """Every step ``--from`` can run the quest again from, in graph order: name, block, plain sentence, whether it
        needs approval, whether this quest reached it, and what would be moved aside. Read from the checkpoint history;
        no model call, nothing written."""
        reached = await self._reached_steps()
        return [_rerun_from.step_info(step, reached=step in reached) for step in _rerun_from.STEPS]

    async def rerun_no_simulation(self) -> bool:
        """Whether this quest takes the no-simulation path: what the clarify node resolved into the checkpoint, or what
        the config says (a survey or a local-data analysis has no experiment either). No model call, nothing written."""
        cfg = self.config
        if cfg.engine.no_simulation or cfg.engine.survey_mode or cfg.engine.analyze_local_first:
            return True
        values = await self._checkpoint_values()
        return bool(values.get("no_simulation_resolved") or values.get("survey_mode_resolved"))

    async def quest_map(self, steps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """What the quest map draws (``core.rerun_from.map_payload``): the CLI's ``--from --json`` (the VS Code panel)
        and the web page both take it from here, so they agree on where the quest stopped. ``steps`` is
        ``rerun_steps()`` when the caller already has it. No model call, nothing written."""
        if steps is None:
            steps = await self.rerun_steps()
        finished, at = await self.stopped_at()
        return _rerun_from.map_payload(steps, finished=finished, no_simulation=await self.rerun_no_simulation(), at=at)

    async def stopped_at(self) -> tuple[bool, list[str]]:
        """Where the quest is, for its map: ``(finished, nodes)``. ``nodes`` are the graph nodes the checkpoint is about
        to run (the one a pause waits in, the one that failed, the one a killed run was in); ``finished`` is true only
        when there is none left AND nothing is waiting: no pause (``.fi/pause.json``, ``NEXT_STEP.md``) and no failure
        (``quest_failed.md``). A pause writes ``frontier_insight_summary.json`` too, so that file never says "finished".
        A quest with no checkpoint, or one that cannot be read, is not finished. No model call, nothing written."""
        nxt = await self._checkpoint_next()
        if nxt is None:
            return False, []
        waiting = any(p.is_file() for p in (self.fi_dir / "pause.json", self.quest_root / "NEXT_STEP.md",
                                             self.quest_root / "quest_failed.md"))
        return (not nxt and not waiting), list(nxt)

    async def _checkpoint_next(self) -> tuple[str, ...] | None:
        """The nodes the latest checkpoint is about to run (``()`` at the end of the graph), or ``None`` when the quest
        has no checkpoint or it cannot be read."""
        try:
            opened = await self._open_readonly_graph()
            if opened is None:
                return None
            graph, conn = opened
            try:
                snap = await graph.aget_state({"configurable": {"thread_id": self.quest_id}})
            finally:
                await conn.close()
        except Exception as e:  # noqa: BLE001 -- a map that cannot read the checkpoint shows "not finished"
            self._log.debug("[map] could not read the checkpoint: %r", e)
            return None
        # A thread with no checkpoint at all also has nothing next: that is "never ran", not "finished".
        if not ((snap.config or {}).get("configurable") or {}).get("checkpoint_id"):
            return None
        return tuple(snap.next or ())

    async def _checkpoint_values(self) -> dict[str, Any]:
        graph_and_close = await self._open_readonly_graph()
        if graph_and_close is None:
            return {}
        graph, conn = graph_and_close
        try:
            snap = await graph.aget_state({"configurable": {"thread_id": self.quest_id}})
            return dict(snap.values or {})
        finally:
            await conn.close()

    async def _open_readonly_graph(self) -> tuple[Any, Any] | None:
        import aiosqlite

        path = self.fi_dir / "state.sqlite"
        if not path.is_file():
            return None
        # Read-only, and with the tables taken as there: the quest may be running in another process, and the saver's
        # own setup would write to its database.
        conn = await aiosqlite.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            saver = AsyncSqliteSaver(conn)
            saver.is_setup = True
            return self._build_graph().compile(checkpointer=saver), conn
        except BaseException:
            await conn.close()
            raise

    async def _reached_steps(self) -> list[str]:
        opened = await self._open_readonly_graph()
        if opened is None:
            return []
        graph, conn = opened
        try:
            return await _rerun_from.reached(graph, {"configurable": {"thread_id": self.quest_id}})
        finally:
            await conn.close()

    # ---- graph topology --------------------------------------------------

    def _build_graph(self) -> StateGraph:
        # Subclassing `Engine` and overriding `_build_graph` is the
        # supported way to ship a domain-specific pipeline (e.g.,
        # a lithography graph) without forking the full Engine class.
        # The QuestState TypedDict is the contract — keep field names
        # backwards-compatible if you add a graph here.
        g: StateGraph[QuestState] = StateGraph(QuestState)
        g.add_node("clarify", self._audited("clarify", self._node_clarify_then_phased))
        g.add_node("ideate", self._audited("ideate", self._node_ideate))
        g.add_node("literature", self._audited("literature", self._node_literature))
        g.add_node("pause_after_literature", self._audited("pause_after_literature", self._node_pause_after_literature))
        g.add_node("scope", self._audited("scope", self._node_scope))
        g.add_node("select_skills", self._audited("select_skills", self._node_select_skills))
        g.add_node("plan", self._audited("plan", self._node_plan))
        g.add_node("design", self._audited("design", self._node_design))
        # design → implement_outline → implement → execute (two-stage
        # implement). The outline node produces a scaffold + function
        # signatures + constants + RESULT_JSON template, which the
        # body node fills in. Splitting the work this way lets each
        # call be smaller and gives the model a feedback opportunity
        # before committing to a full ~200-line experiment.py. On a
        # pre-Phase-2 resume where ``implement_outline`` is empty, the
        # body node falls back to the legacy single-shot prompt.
        g.add_node("implement_outline", self._audited("implement_outline", self._node_implement_outline))
        g.add_node("implement", self._audited("implement", self._node_implement))
        g.add_node("execute", self._audited("execute", self._node_execute))
        # execute → execute_reflect (loops back to execute on failure)
        g.add_node("execute_reflect", self._audited("execute_reflect", self._node_execute_reflect))
        # execute_reflect → improve: the simulation changed one step at a time against its checks of correctness
        # (core/improve.py); a passthrough unless the protocol has criteria FI measures and one is not met.
        g.add_node("improve", self._audited("improve", self._node_improve))
        g.add_node("analyze", self._audited("analyze", self._node_analyze))
        # analyze → cross_check (always) → write OR design
        g.add_node("cross_check", self._audited("cross_check", self._node_cross_check))
        # evidence_gate sits on the cross_check → write happy path: it
        # weighs the assembled evidence and either proceeds to write or
        # sends the quest back for ONE bounded literature broaden. A
        # logged passthrough when engine.evidence_gate is off.
        g.add_node("evidence_gate", self._audited("evidence_gate", self._node_evidence_gate))
        g.add_node("stuck_no_findings", self._audited("stuck_no_findings", self._node_stuck_no_findings))
        g.add_edge("stuck_no_findings", END)
        g.add_node("write", self._audited("write", self._node_write))
        g.add_node("claim_check", self._audited("claim_check", self._node_claim_check))
        g.add_node("citation_check", self._audited("citation_check", self._node_citation_check))
        g.add_node("replot_layout", self._audited("replot_layout", self._node_replot_layout))
        g.add_node("review", self._audited("review", self._node_review))
        g.add_node("human_feedback", self._audited("human_feedback", self._node_human_feedback))
        # no-simulation mode: design → auto_collect_data → wait_for_data
        # → (pause + resume) → data_load → analyze. All three new nodes
        # are conditional and only fire when
        # ``state.no_simulation_resolved`` is True.
        g.add_node("auto_collect_data", self._audited("auto_collect_data", self._node_auto_collect_data))
        g.add_node("wait_for_data", self._audited("wait_for_data", self._node_wait_for_data))
        g.add_node("data_load", self._audited("data_load", self._node_data_load))
        # no-simulation mode only: turn the collected web/data content into
        # figures so the paper/poster/slides aren't text-only. Passthrough
        # in the simulation path (which makes its own figures in execute).
        g.add_node("web_plots", self._audited("web_plots", self._node_web_plots))
        g.add_node("web_figures", self._audited("web_figures", self._node_web_figures))

        g.add_edge(START, "clarify")
        g.add_edge("clarify", "ideate")
        g.add_edge("ideate", "literature")
        # Selection sits between literature and design so it sees the
        # chosen direction and the retrieved sources — the direction is
        # what actually names the instrument a quest needs.
        # ``pause_after_literature`` is a passthrough unless pauses.supply asks
        # for a stop there: the search is saved, and a resume starts here
        # rather than searching again.
        g.add_edge("literature", "pause_after_literature")
        g.add_edge("pause_after_literature", "scope")
        g.add_edge("scope", "select_skills")
        g.add_edge("select_skills", "plan")
        g.add_edge("plan", "design")
        # design → implement (normal sim path) OR auto_collect_data
        # (no-simulation, agent attempts auto-collect via Axon first
        # then wait_for_data handles the pause-if-still-empty case).
        g.add_conditional_edges(
            "design",
            self._audited_route("design", self._route_after_design),
            {
                # The route key stays ``implement`` for resume
                # compatibility (the 609990 checkpoint pins
                # ``next=("implement",)``); the underlying target is
                # the outline node, which then chains to the body node
                # named ``implement``. Routing decisions don't see
                # the outline; the legacy ``implement`` resume path
                # bypasses the outline and falls into the body's
                # legacy single-shot prompt.
                "implement": "implement_outline",
                "auto_collect_data": "auto_collect_data",
                # survey mode: no experiment AND no dataset — jump straight
                # to web_figures (illustrative images) → analyze, skipping
                # the whole data-collection chain.
                "synthesize": "web_figures",
            },
        )
        g.add_edge("implement_outline", "implement")
        g.add_edge("implement", "execute")
        g.add_edge("execute", "execute_reflect")
        g.add_conditional_edges(
            "execute_reflect",
            self._audited_route("execute_reflect", self._route_after_execute_reflect),
            {"retry": "execute", "proceed": "improve", "stuck": "evidence_gate"},
        )
        # improve → execute once, when it kept a changed version: that version runs the ordinary way (every check, the
        # full set of trials), and its run is the one analysed. It then comes back here and goes on.
        g.add_conditional_edges(
            "improve",
            self._audited_route("improve", self._route_after_improve),
            {"rerun": "execute", "proceed": "analyze"},
        )
        # auto_collect_data: best-effort Axon retrieval that
        # writes hits into <quest_root>/data/auto_collected/<idx>_<slug>.md
        # so wait_for_data's rglob walk picks them up. Always proceeds —
        # if Axon is disabled, returns zero docs, or the feature flag
        # is off, the node is a logged passthrough.
        g.add_edge("auto_collect_data", "wait_for_data")
        # wait_for_data uses LangGraph's ``interrupt()`` to pause when
        # the user hasn't dropped any files yet AND auto_collect_data
        # didn't land any either. On resume (with files present), the
        # node returns and we proceed to data_load.
        g.add_edge("wait_for_data", "data_load")
        # data_load → web_plots → analyze. web_plots is a no-op in the
        # simulation path; in no-simulation mode it derives figures from
        # the collected sources before analyze/write consume them.
        g.add_edge("data_load", "web_plots")
        g.add_edge("web_plots", "web_figures")
        g.add_edge("web_figures", "analyze")
        g.add_edge("analyze", "cross_check")
        g.add_conditional_edges(
            "cross_check",
            self._audited_route("cross_check", self._route_after_cross_check),
            {
                # The happy-path "write" label now lands on evidence_gate,
                # which makes the real write-vs-broaden call.
                "write": "evidence_gate",
                "redesign": "design",
                "broaden_lit": "literature",
            },
        )
        # evidence_gate → write (sufficient / insufficient-but-write) OR
        # back to literature for ONE bounded broaden pass.
        after_gate = {"write": "write", "broaden_lit": "literature", "redesign": "design", "stuck": "stuck_no_findings",
                      "repair_crash": "execute", "rewrite_code": "implement"}
        if _phased.enabled(self.config):
            # Explore, then confirm (core/phased.py): when exploration ends, the frozen design runs once more; a quest
            # that analyses data reads the held-back rows once more with it.
            after_gate["confirm"] = "execute"
            after_gate["confirm_data"] = "data_load"
        g.add_conditional_edges(
            "evidence_gate",
            self._audited_route("evidence_gate", self._route_after_evidence_gate),
            after_gate,
        )
        # write → claim_check → review. claim_check grounds each paper claim to
        # evidence (a no-op passthrough when engine.claim_grounding is off).
        # A person's refine that the writer says needs a new experiment goes back to the design; everything else goes
        # on to the checks and the review as always.
        g.add_conditional_edges(
            "write",
            self._audited_route("write", self._route_after_write),
            {"check": "claim_check", "redesign": "design", "extend": "implement", "collect": "auto_collect_data",
             "replot": "replot_layout"},
        )
        g.add_conditional_edges(
            "replot_layout",
            self._audited_route("replot_layout", self._route_after_replot),
            {"check": "claim_check", "write": "write"},
        )
        g.add_edge("claim_check", "review")
        g.add_conditional_edges(
            "review",
            self._audited_route("review", self._route_after_review),
            {
                "revise": "design",
                # Every must-flag is a problem with the text: the experiment
                # stands, so only the paper is written again.
                "rewrite": "write",
                # A must-flag named something the run computed: no rewrite of
                # the paper can fix that, so the experiment is written and run
                # again (implement → execute → analyze → … → write).
                "re_execute": "implement",
                # The paper that is about to be delivered (or shown to a person) has every citation read against the
                # retrieved text of its source last, after the review, the rewrites and the page-limit fit.
                "done": "citation_check",
                "human_feedback": "citation_check",
                # The paper still claims what FI's own records contradict after its rewrites: no paper (needs/STUCK.json).
                "stuck": "stuck_no_findings",
            },
        )
        g.add_conditional_edges(
            "citation_check",
            self._audited_route("citation_check", self._route_after_citation_check),
            {"done": END, "human_feedback": "human_feedback"},
        )
        # human_feedback resolves to one of three outcomes after the callback returns: accept / reject → END,
        # refine → write (the writer answers the notes; one that needs a new experiment goes on to design).
        g.add_conditional_edges(
            "human_feedback",
            self._audited_route("human_feedback", self._route_after_human_feedback),
            # A refine of a search for the best design that asks to search further goes back to the search itself;
            # one that asks for a new study (or cannot be read) changes nothing and asks again (core/optimise_refine.py).
            {"rewrite": "write", "revise": "design", "done": END, "search": "execute", "ask_again": "human_feedback"},
        )
        return g

    # ---- conditional edges --------------------------------------------------

    def _route_after_design(self, state: QuestState) -> str:
        """When ``no_simulation_resolved`` is set, skip the
        implement → execute → execute_reflect chain entirely. Instead
        route to ``auto_collect_data``, which best-effort-pulls
        relevant docs from Axon into ``<quest_root>/data/auto_collected/``,
        then hands off to ``wait_for_data`` → ``data_load`` → ``analyze``.

        ``no_simulation_resolved`` is set by the clarify node from
        either the ``engine.no_simulation`` YAML flag (wins) or the
        ``empirical_vs_theoretical == 'empirical'`` clarify answer.
        See ``_resolve_no_simulation_from_clarify``.

        Note: this routing decision is made AFTER design runs, so the
        no-simulation flow still benefits from the LLM's experimental
        design (variables, hypotheses, measurement plan) — it just
        skips the simulate-and-execute half and treats the user's
        real-world data as the experimental result instead.

        Before returning, the topic-type ROUTE_MATRIX is consulted: the
        resolved ``no_simulation`` flag stays authoritative, but a
        topic-type-vs-route divergence is surfaced as a WARNING.
        """
        # Survey mode is the strongest form of no-simulation: no experiment
        # AND no dataset. Skip the data-collection chain entirely
        # (auto_collect_data → wait_for_data → data_load → web_plots) and go
        # straight to web_figures → analyze, so the paper synthesises the
        # literature directly. web_figures still runs (it is guarded on
        # no_simulation_resolved, which survey implies) and sources its
        # license-clean illustrative images from the topic + literature.
        if bool(state.get("survey_mode_resolved")):
            self._warn_route_matrix_mismatch(state, "no_simulation")
            return "synthesize"
        no_sim = bool(state.get("no_simulation_resolved"))
        self._warn_route_matrix_mismatch(
            state, "no_simulation" if no_sim else "simulation")
        if no_sim:
            return "auto_collect_data"
        return "implement"

    def _warn_route_matrix_mismatch(
        self, state: QuestState, actual_path: str,
    ) -> None:
        """Consult the topic-type ``ROUTE_MATRIX`` at the design fork and
        log a WARNING when the topic type's expected path disagrees with
        the resolved route. The resolved ``no_simulation`` flag stays
        authoritative — routing is NOT overridden here; this only makes
        the divergence visible in run.log.

        Distinct from ``_log_topic_shape_mismatch`` (which fires earlier,
        at clarify time, and only for non-experimental shape + SIMULATE):
        this is bidirectional and keyed on the richer
        ``ResearchProtocol.topic_type``, so it also catches a future
        ``engineering`` topic that resolved to NO_SIMULATION. Pure + no
        LLM call; fail-open (never blocks routing on its own error)."""
        try:
            protocol = derive_protocol(dict(state), self.config)
            expected = route_for_topic_type(protocol.topic_type)
        except Exception:  # never let a diagnostic break routing
            return
        if expected == actual_path:
            return
        self._log.warning(
            "[route] topic_type=%r expects the %s path but the quest "
            "resolved to the %s path. Routing follows the resolved "
            "no_simulation flag (config/clarify wins); set "
            "``simulatability`` in clarify_overrides to align. Declared "
            "source policy for this type: %s.",
            protocol.topic_type, expected, actual_path, protocol.source_policy,
        )

    def _figure_overlaps_to_repair(self, state: QuestState) -> list[str]:
        """The figure overlaps the run is to be sent back for: those of
        ``_figure_overlap_findings``, when the repair budget can spare an
        attempt for them.

        A redraw rewrites a script that ran, and a rewrite can break it, so it
        is offered only while one more attempt is left after it to repair that.
        Without the reserve, a redraw on the last attempt that crashed would
        turn a run with an overlapping legend into a run with no result.
        """
        if int(state.get("exec_reflect_iter", 0) or 0) + 1 >= self.config.engine.exec_reflect_max_iterations:
            return []
        return _figure_overlap_findings(state)

    def _route_after_improve(self, state: QuestState) -> str:
        """``rerun`` when the improve loop kept a changed version (it is run once more, the ordinary way), else
        ``proceed`` to the analysis."""
        return "rerun" if state.get("improve_rerun") else "proceed"

    def _repair_budget_left(self, state: QuestState) -> bool:
        """Whether the run's repairs are not spent (the repair of a crash or a figure goes on). The router after the repair and
        ``_node_execute`` (which holds back the work of a final run while a repair is coming) both ask this."""
        return (int(state.get("exec_reflect_iter", 0) or 0) < self.config.engine.exec_reflect_max_iterations
                and not state.get("exec_give_up_reason"))

    def _route_after_execute_reflect(self, state: QuestState) -> str:
        """Route based on whether the reflect node patched the
        code (→ retry execute) or accepted the failure / success
        (→ proceed to analyze)."""
        result = state.get("exec_result") or {}
        if any(result.get(k) for k in _STOPPED_BEFORE_THE_STUDY):
            # Decided before anything ran (the plan never covered the topic, the study would not fit the time allowed, FI's
            # repairs ended with a problem standing ...): there is nothing to improve, analyse or cross-check, and a
            # redesign now would be recorded as a change made after results were seen. The gate states the stop (no
            # paper) from the record, and the quest ends there.
            return "stuck"
        if result.get("flat_objective"):
            # The search ran, and its objective did not move: nothing in the script to repair. The gate sends what FI
            # measured back to the plan.
            return "stuck"
        rc = result.get("returncode", 0)
        # ``_node_execute`` stores ``result_json or {}``, so a script that
        # exits 0 WITHOUT a RESULT_JSON marker lands as an empty dict — which
        # ``is not None`` wrongly counted as "parsed", skipping repair. Treat
        # an empty result_json as no usable result so execute_reflect retries.
        has_result_json = bool(state.get("result_json"))
        # Give-up sentinel set by the reflect node OR iterations exhausted
        # always win — otherwise the degenerate-retry below could loop
        # past the budget.
        if state.get("exec_give_up_reason"):
            return "proceed"
        # A patch nobody has run yet is run, the one that used the last
        # attempt included. The reflect node writes no patch once the attempts
        # are spent, so this cannot loop.
        if state.get("exec_patch_pending"):
            return "retry"
        # A run that ran to the end and only warned is repaired on the warnings' own budget or stopped for them by the
        # reflect node, whatever is left of the crash repairs (a repair that gave up has gone on above, as before).
        if self._numeric_turn(state):
            return "retry"
        if not self._repair_budget_left(state):
            return "proceed"
        if rc == 0 and has_result_json:
            # Clean exit + parsed metrics — but an all-zero/degenerate
            # result is a soft failure: loop back so the reflect node can
            # patch the underlying bug before we write a paper.
            if (
                self.config.engine.degenerate_run_guard
                and _is_degenerate_result(state.get("result_json") or {})
            ):
                return "retry"
            # A run can exit 0 with perfectly plausible numbers and still be
            # physically wrong (unit error, factor of two, sign flip). The
            # design declared what its outputs may legally be; enforce it
            # here, before a paper gets written from them.
            if _assertion_violations(self._in_force(state), self._walk_memo()):
                return "retry"
            # Figures with a legend or a title drawn over what a reader needs go
            # back once: the reflect node writes the redraw, and the flag it sets
            # ends the finding, so this cannot loop.
            if self._figure_overlaps_to_repair(state):
                return "retry"
            return "proceed"
        return "retry"

    def _route_after_cross_check(self, state: QuestState) -> str:
        """Route based on analyze's ``next_step`` field.

        Three outcomes:

        * ``next_step == "broaden_lit"`` → ``broaden_lit`` (re-enter
          ``literature`` so the next pass can fetch fresh evidence;
          design then re-runs with the accumulated literature block).
          Until this routing existed, ``broaden_lit`` collapsed onto
          ``redesign`` and the design node re-ran with the SAME
          literature it already had — defeating the signal.
        * ``next_step == "re_experiment"`` → ``redesign`` (re-enter
          ``design`` with the same literature; only the experimental
          plan changes).
        * Anything else (or budget exhausted, or rerouting disabled
          via ``engine.enable_analyze_reroute: false``) → ``write``.

        Both re-entry branches share the same ``engine.max_iterations``
        budget so the whole quest stays bounded.

        ``_node_cross_check`` returns ``{"cross_check": []}`` early only when ``cross_check_per_finding_k <= 0`` (the draft
        profile skips this step); that return also carries ``cross_check_route: "write"``. Every other pass bumps the
        iteration itself on a route that is not ``write``, so a ``broaden_lit`` loop is bounded by
        ``engine.max_iterations`` whatever the number of results looked up (a run with none routes by ``next_step``).

        The cross_check node decides the route (``cross_check_route``) before it spends a round, and this router reads
        it, as the evidence gate's router does: a router that decided again after the node raised ``iteration`` saw the
        last round as already spent and wrote instead. Without the field (a state the node did not make), the route is
        decided here from the state as it is.
        """
        route = state.get("cross_check_route")
        if route in ("write", "broaden_lit", "redesign"):
            return route
        if not state.get("cross_check"):
            return "write"  # a checkpoint the new node did not write: this router bumps no round, so it never loops
        return self._cross_check_route(state)

    def _cross_check_route(self, state: QuestState) -> str:
        """``write``, ``broaden_lit`` or ``redesign`` from the analysis's ``next_step``, FI's precision record, the
        confirm stage of ``engine.phased`` and the rounds left (``iteration`` before this round is spent)."""
        if not self.config.engine.enable_analyze_reroute:
            return "write"
        analysis = state.get("analysis") or {}
        next_step = analysis.get("next_step", "publish")
        if next_step in ("broaden_lit", "re_experiment") and self._phased_confirming():
            self._log.info("[phased] confirm stage: the frozen design is run once, so the analysis's request to %s is "
                           "not followed; the confirm result is written up as it is",
                           "look at the literature again" if next_step == "broaden_lit" else "redesign")
            return "write"
        if state.get("iteration", 0) >= self.config.engine.max_iterations:
            return "write"
        if next_step == "broaden_lit":
            return "broaden_lit"
        if next_step == "re_experiment":
            missed = self._precision_missed(state)
            if missed:
                self._log.info("[cross_check] the analysis asks to run the experiment again, and FI's record agrees: the "
                               "plan's precision target was not reached (%s), and a run large enough for it fits the time "
                               "allowed", ", ".join(missed[:4]))
                return "redesign"
            # The model's reasons were nearly always a run that failed or left settings out: a fault of the code, which
            # the evidence check sends back from its own record, never a reason to redesign after results were seen. A
            # precision target missed that a larger run cannot reach in time is said in the paper's limitations
            # (core/disclosure.py:run_record).
            self._log.info("[cross_check] the analysis asks to run the experiment again (%s), but FI's record shows no "
                           "precision target missed that a larger run could reach in the time allowed; the evidence "
                           "check decides from the record",
                           _one_line(str(analysis.get("next_step_reason") or analysis.get("summary") or "no reason given"), 160))
        return "write"

    def _flat_redesign_allowed(self, state: QuestState) -> bool:
        """Whether a search whose objective does not move may still stop early and send what FI measured back to the
        plan: once per quest, within ``engine.max_iterations``, never in the confirm stage of ``engine.phased`` (its
        frozen design runs once). Otherwise the search runs its whole budget and records the fact."""
        return (not state.get("flat_objective_redesigned") and not self._phased_confirming()
                and int(state.get("iteration", 0) or 0) < self.config.engine.max_iterations)

    def _precision_shortfalls(self, state: QuestState) -> list[dict[str, Any]]:
        """The pooled quantities (a probability from its counts, a mean from the values the run lists) whose interval is
        wider than the plan's precision target in this run (the record ``_annotate_precision`` writes), each ``{"name",
        "half_width", "target", "trials"}``; empty when the plan sets none, the run has no repeats, or all were reached.
        A mean the run gives without its values has no interval from the trials and is not listed."""
        try:
            replicates = state.get("result_json_replicates") or []
            if not (len(replicates) > 1 or (replicates and state.get("result_json_trials"))):
                return []
            aggregate = _aggregate_result_json_replicates(
                replicates, assertions=_replicate_assertions(state), kinds=_replicate_metric_kinds(state))
            self._annotate_precision(state, aggregate)
            return [{"name": k, "half_width": float(v["ci_half_width"]), "target": float(v["target_half_width"]),
                     "trials": v.get("n_trials") if v.get("n_trials") is not None else v.get("n_values")}
                    for k, v in aggregate.items() if isinstance(v, dict) and v.get("precision_reached") is False]
        except Exception as e:  # noqa: BLE001 -- a record that cannot be read never sends a quest back
            self._log.warning("[precision] the precision record could not be read (%r)", e)
            return []

    def _precision_missed(self, state: QuestState, *, quiet: bool = False) -> list[str]:
        """The probabilities whose precision target this run missed and a larger run could reach in the time allowed:
        the trials scale with the square of the interval's width, so the run is ``(half_width / target) ** 2`` times
        longer, and FI's own estimate of this run (``.fi/run_estimate.json``, which already carries its safety margin)
        times that must fit the time the study is allowed. The estimate counts only when it is of this run's size (its
        settings and trials; FI times only the first design's run, so after a redesign it is another run's). No such
        estimate, a run FI already had to make smaller, or one that would not fit: empty (the paper says the target was
        missed). ``quiet``: the same answer, without saying why in run.log (the router says it)."""
        from . import run_estimate as _estimate

        missed = self._precision_shortfalls(state)
        if not missed:
            return []
        say = (lambda *a: None) if quiet else self._log.info
        record = _estimate.read_record(self.quest_root)
        estimate_s, limit_s = record.get("estimate_s"), record.get("limit_s")
        try:
            grid, runs, _thresholds = self._grid_and_runs(state)
            cells = _estimate.cell_count(grid)
        except Exception:  # noqa: BLE001 -- a plan whose size cannot be read has no estimate of this run
            cells, runs = -1, 0
        if record.get("reduced") or not record.get("fits") or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in (estimate_s, limit_s)) \
                or record.get("cells") != cells or record.get("trials") not in (cells, cells * runs):
            say("[cross_check] the precision target was not reached (%s); FI has no estimate of this run showing a "
                "larger run fits the time allowed", ", ".join(m["name"] for m in missed[:4]))
            return []
        scale = max((m["half_width"] / m["target"]) ** 2 for m in missed)
        if float(estimate_s) * scale > float(limit_s):  # type: ignore[arg-type]
            say("[cross_check] the precision target was not reached (%s); a run large enough for it would take about "
                "%.0f s, over the %.0f s allowed", ", ".join(m["name"] for m in missed[:4]),
                float(estimate_s) * scale, float(limit_s))  # type: ignore[arg-type]
            return []
        return [m["name"] for m in missed]

    def _precision_redesign_note(self, state: QuestState) -> str:
        """For the redesign the precision record sent: each quantity whose interval was wider than the plan's target,
        with the trials it had and the trials that reach the target (they grow with the square of the width ratio), and
        what to keep. From FI's own record of the last run."""
        import math

        lines = []
        for m in self._precision_shortfalls(state):
            trials = m.get("trials")
            ratio = (m["half_width"] / m["target"]) ** 2 if m["target"] > 0 else None
            need = (f"; about {math.ceil(float(trials) * ratio)} trials reach it" if isinstance(trials, (int, float))
                    and not isinstance(trials, bool) and trials > 0 and ratio else "")
            lines.append(f"- {m['name']}: interval half-width {m['half_width']:.3g} against the target "
                         f"{m['target']:.3g}" + (f", with {trials} trials" if trials else "") + need)
        body = "\n".join(lines) or "- (the record of which quantities missed it could not be read again)"
        return (f"FI's record of the last run:\n{body}\n"
                "A run that large fits the time allowed (FI's own estimate of this run). Keep the question, the "
                "settings and what is measured; change the number of trials so each target is reached.")

    def _route_after_evidence_gate(self, state: QuestState) -> str:
        """The evidence_gate node already decided write vs broaden (it
        owns the bounded-broaden bookkeeping); the router just reads it.
        Fails open to ``write`` when the gate was a passthrough. Under ``engine.phased`` a write at the end of
        exploration goes to the confirm run first (``_phased_route``)."""
        route = (state.get("evidence_assessment") or {}).get("route", "write")
        if route in ("stuck", "repair_crash", "rewrite_code"):
            return route  # no findings to write up, a crash to repair first, code to write again: the same in every mode
        if _phased.enabled(self.config):
            return self._phased_route(state, route)
        return route

    # ---- explore, then confirm (engine.phased; core/phased.py) --------------------------------------------------

    def _phased_log(self, lines: list[str]) -> None:
        for line in lines:
            self._log.info("[phased] %s", line)
            if "keep your own copy" in line:
                # Said where the person watching sees it too (console, web page, VS Code): it is about their data.
                self._progress("[phased] Part of your data is held back, encrypted with a key only this run of FI "
                               "holds: if FI is killed rather than stopped, those rows cannot be put back, so keep your "
                               "own copy of the file.")
        self._phased_seal_confirmations()

    def _phased_seal_confirmations(self) -> None:
        """Each verdict line of ``.fi/confirmations.jsonl`` the decision trace does not name yet gets its event
        (``confirmation_recorded``, with the line's hash), so a line changed or removed later is found
        (``core/confirmations.problems``)."""
        if not _confirmations.path(self.quest_root).is_file():
            return
        try:
            events = _audit_log.read(self.audit.path) if self.audit.path.is_file() else []
            for digest, entry in _confirmations.unsealed(self.quest_root, events):
                self._audit(_confirmations.EVENT, line_sha256=digest, candidate=entry.get("candidate"),
                            verdict=entry.get("verdict"))
        except Exception as e:  # noqa: BLE001 -- the trace is a record; it must never stop a quest
            self._log.debug("[audit] could not record the confirmations: %r", e)

    def _phased_keep_off_if_began_before(self) -> None:
        """A research quest whose config does not set ``engine.phased`` has it on by default. One that began before that
        default (it has run with it never on, and has no record of the two stages; ``phased.kept_off``) goes on as it
        began, without them: for this engine only, never the caller's config object. Said in run.log at every start."""
        if not (_phased.enabled(self.config) and self.config.rigor_profile == "research"):
            return  # nothing of core/phased.py beyond ``enabled`` runs for a quest with it off
        if not _phased.kept_off(self.quest_root, self.config):
            return
        self.config = _phased.without(self.config)
        self._phased_kept_off = True
        self._log.info("[phased] %s", _phased.kept_off_sentence(self._runs_code_by_config()))

    def _runs_code_by_config(self) -> bool:
        """Whether, as far as the config says, the quest runs an experiment of its own (the clarify step may still
        decide it does not)."""
        return not self._phased_no_experiment_reason()

    def _plan_confirm_lines(self, state: QuestState) -> list[str]:
        """At plan time: plan.md's lines on whether and how the result will be confirmed, and what that costs; the same
        said in run.log. Nothing under the default profile with explore-then-confirm off."""
        on = _phased.enabled(self.config)
        research = self.config.rigor_profile == "research"
        if not on and not research:
            return []
        data_quest = self._phased_data_quest(state)
        try:
            # How the held-back rows will be chosen, or the question of which rows belong together (read back from
            # plan.md when the data is first read).
            note = _phased.split_preview(self.quest_root, self.quest_id, **self._phased_split_inputs(state),
                                         key=self._phased_key()) if on else ""
            lines = _phased.plan_lines(
                _phased.load(self.quest_root) if on else None, on=on, research=research,
                runs_code=self._runs_code(state), why=self._phased_no_experiment_reason(state),
                kept_off=bool(getattr(self, "_phased_kept_off", False)), data_quest=data_quest, split_note=note)
        except Exception as e:  # noqa: BLE001 -- a line of the plan must never stop the quest
            self._log.warning("[phased] the plan's section on confirming the result could not be written: %r", e)
            return []
        if on and (self._runs_code(state) or data_quest) and len(lines) > 2:
            self._log.info("[phased] plan: %s", lines[2])
        return lines

    def _phased_no_experiment_reason(self, state: QuestState | None = None) -> str:
        """Why this quest has nothing to run once more (empty when it has, or it is not known yet): the config pins it,
        or the clarify step resolved it (``state``). A quest that collects and analyses data instead of simulating has:
        its data-reading step, on held-back rows (``_phased_data_quest``)."""
        engine = self.config.engine
        if engine.analyze_local_first:
            return "the quest analyses data you supplied (--analyze) and designs no experiment of its own"
        if engine.survey_mode or (state or {}).get("survey_mode_resolved"):
            return "this quest is a literature survey and runs no experiment"
        return ""

    def _phased_data_quest(self, state: QuestState | None = None) -> bool:
        """Whether this quest analyses data instead of simulating (not a survey, not ``--analyze``): the config pins it,
        the clarify step resolved it (``state``), or the record of the two stages says so (a resume)."""
        if self._phased_no_experiment_reason(state):
            return False
        if self.config.engine.no_simulation or (state or {}).get("no_simulation_resolved"):
            return True
        return bool((_phased.load(self.quest_root) or {}).get("data_quest"))

    def _phased_key(self) -> bytes | None:
        """This run's key for the held-back rows (``core/phased_isolation.py``): made once per run of FI and kept only in
        this object's memory, never on disk, in the state or in the environment of the quest's code. None in a
        container, which keeps the kept files out of reach by not mounting them."""
        if self.config.execution.sandbox == "docker" or getattr(self, "_phased_no_crypto", False):
            return None
        key = getattr(self, "_phased_key_value", None)
        if key is None:
            try:
                key = _phased_isolation.new_key()
            except ImportError:
                # Installed before FI needed it: the rows are still held back, kept plain, and the record says they
                # were not out of reach (isolation_unverified).
                self._phased_no_crypto = True
                self._log.warning("[phased] the `cryptography` package is not installed, so the held-back rows are kept "
                                  "unencrypted and the result cannot count as confirmed on unseen data; install it "
                                  "with `pip install -r requirements.txt`")
                return None
            self._phased_key_value = key
        return key

    def _phased_mode(self) -> str:
        """How this run keeps the held-back rows out of reach: ``docker``, ``encrypted``, or ``plain`` (neither)."""
        if self.config.execution.sandbox == "docker":
            return "docker"
        return "encrypted" if self._phased_key() is not None else "plain"

    def _phased_isolation(self) -> tuple[str, str]:
        """``(status, why)`` for the confirm stage: ``docker`` when every start ran the quest's code in a container that
        is not given the kept files; ``encrypted+scanned`` when the kept files were encrypted at every start without a
        container and the quest's code reads no path out of the quest folder; otherwise ``isolation_unverified`` with
        the reason in plain words."""
        record = _phased.load(self.quest_root) or {}
        starts = set(record.get("starts") or [])
        if not starts or "unrecorded" in starts:
            return (_phased_isolation.UNVERIFIED,
                    "the quest held its rows back before FI recorded how they were kept from exploration")
        if "plain" in starts:
            return (_phased_isolation.UNVERIFIED, "a run of the quest kept the held-back rows unencrypted without a "
                                                  "container (the `cryptography` package was not installed)")
        if starts == {"docker"}:
            why = _phased_isolation.docker_mounts_clear(self.executor, self.quest_root, _phased.store_dir(self.quest_root))
            return (_phased_isolation.DOCKER, "") if not why else (_phased_isolation.UNVERIFIED, why)
        if starts <= {"docker", "encrypted"}:
            hits = _phased_isolation.scan(self.quest_root)
            if hits:
                more = f" and {len(hits) - 3} more" if len(hits) > 3 else ""
                return (_phased_isolation.UNVERIFIED,
                        "the quest's code may read outside the quest folder: " + "; ".join(hits[:3]) + more)
            return _phased_isolation.ENCRYPTED, ""
        return (_phased_isolation.UNVERIFIED,
                "a run of the quest kept the held-back rows neither in a container nor encrypted")

    def _phased_split_inputs(self, state: QuestState) -> dict[str, Any]:
        """What decides which rows are held back: the plan's ``protocol.split`` (plan.md's design block, else the design
        in the state), the answer on plan.md's line ``Rows that belong together:``, and the plan's variables."""
        from . import phased_data as _phased_data

        planned, _why = _plan.load_design(self.quest_root)
        declared = _phased_data.declared((planned or {}).get("protocol")) or _phased_data.declared(
            (state.get("design") or {}).get("protocol") if isinstance(state.get("design"), dict) else None)
        try:
            text = _plan.plan_path(self.quest_root).read_text(encoding="utf-8")
        except OSError:
            text = ""
        return {"declared": declared, "answer": _phased_data.plan_answer(text),
                "grouping": _phased_data.grouping_names(state.get("design"))}

    def _phased_ask_split(self, question: str) -> None:
        """Stop and ask which rows of the data belong together (one plain question, answered in plan.md). Nothing has
        read the data yet; the resume reads the answer and holds rows back by it."""
        plan = _plan.plan_path(self.quest_root)
        self._pause_for_human(
            kind="data_split",
            interaction="supply",
            headline="say which rows of your data belong together",
            steps=[
                "Part of your data is held back so that the result can be confirmed on data the exploration never read. "
                "Rows of one subject, site or device must stay on the same side, and FI cannot tell from the table "
                "which rows those are.",
                question,
                f"In `{plan}`, write your answer on the line `Rows that belong together:` (add the line if it is not "
                "there): the column that names the subject, site, device or other unit several rows can share, or "
                "`independent` if every row is a separate case. Then resume.",
            ],
            recommended="Name the column that identifies the subject, site or device, then resume.",
            alternatives=["`independent`, if every row is a separate case (no subject, site or device is measured twice).",
                          "Turn explore-then-confirm off for this quest (`engine.phased: false`): the result is then "
                          "not confirmed on unseen data."],
            payload={"quest_id": self.quest_id, "plan_file": str(plan), "data_split": True},
        )

    def _phased_before_first_run(self, state: QuestState) -> str:
        """Right before the experiment's code first runs: which rows are held back is decided
        (``core/phased.decide_split``). Returns the question to ask when a research quest cannot tell which rows belong
        together, else ``""``. After the confirm stage began, a study whose code or protocol changed since it was frozen
        is a new version first (``core/phased.new_candidate``): it is confirmed on its own, never as a second try of the
        earlier version."""
        try:
            self._phased_log(_phased.new_candidate(self.quest_root, key=self._phased_key(), quest_id=self.quest_id))
            _phased.note_confirm_handed(self.quest_root)
            _record, lines, question = _phased.decide_split(
                self.quest_root, self.quest_id, **self._phased_split_inputs(state),
                research=self.config.rigor_profile == "research", key=self._phased_key(),
                outlives_run=bool(self.config.execution.background_jobs) and self._phased_key() is not None)
        except OSError as e:
            self._log.warning("[phased] the data held back could not be kept apart from exploration (%r); nothing in "
                              "this quest can be confirmed, so its numbers stay exploratory", e)
            _phased.mark_compromised(self.quest_root, f"a data file could not be written before it was read ({e!r})")
            return ""
        self._phased_log(lines)
        return question

    def _phased_prepare_data_quest(self) -> None:
        """The clarify step found the quest analyses data: its data in ``data/`` is held back too (and what was held back
        of ``inputs/data/`` stays), before anything reads it."""
        try:
            _record, lines = _phased.prepare(self.quest_root, self.quest_id, data_quest=True, key=self._phased_key())
        except OSError as e:
            self._log.warning("[phased] the data held back could not be kept apart from exploration (%r); nothing in "
                              "this quest can be confirmed, so its numbers stay exploratory", e)
            _phased.mark_compromised(self.quest_root, f"a data file could not be written at a start ({e!r})")
            return
        self._phased_log(lines)

    def _phased_before_data_read(self, state: QuestState) -> str:
        """Right before a data quest's data-reading step reads the data (``core/phased.data_quest_gate``): the split is
        decided the first time, or the quest says in one sentence why its data cannot be split; in the confirm stage the
        held-back rows are put in place. Every reading is counted. Returns a question to ask, else ``""``."""
        try:
            lines, question = _phased.data_quest_gate(
                self.quest_root, self.quest_id, **self._phased_split_inputs(state),
                research=self.config.rigor_profile == "research", key=self._phased_key(),
                design_sha256=_frozen.sha256(state.get("design") or {}))
        except OSError as e:
            self._log.warning("[phased] the data held back could not be kept apart from exploration (%r); nothing in "
                              "this quest can be confirmed, so its numbers stay exploratory", e)
            _phased.mark_compromised(self.quest_root, f"a data file could not be written before it was read ({e!r})")
            return ""
        self._phased_log(lines)
        return question

    def _phased_mark_not_applicable(self, why: str) -> None:
        """The two stages do not apply to this quest (``why``): any held-back rows go back, and it is said once."""
        try:
            self._phased_log(_phased.mark_not_applicable(self.quest_root, why, key=self._phased_key()))
        except OSError as e:
            self._log.warning("[phased] the whole data files could not be put back in inputs/data/ (%r); they are "
                              "kept in %s", e, _phased.store_dir(self.quest_root) / "original")

    def _phased_prepare(self) -> None:
        """At every start: the exploration stage begins, or takes in the data supplied since (a part held back). A quest
        whose config says it runs no experiment of its own holds nothing back: there is nothing to confirm."""
        if why := self._phased_no_experiment_reason():
            self._phased_mark_not_applicable(why)
            return
        # Turned on after the experiment already ran: every row was seen, so nothing can be held back. Any step of the
        # experiment in the decision trace counts, a paused or failed one too (a background job reads the data when it
        # is submitted, before the step pauses). Without the trace (engine.audit_trace off) a saved state is taken to
        # mean it may have run.
        already_ran = False
        if _phased.load(self.quest_root) is None and not (self.fi_dir / _phased.RECORD).exists():
            if self.audit.path.is_file():
                # The experiment's step, or a data quest's data-reading step (it reads the data).
                already_ran = any(str(e.get("kind") or "").startswith("node_") and e.get("node") in ("execute", "data_load")
                                  for e in _audit_log.read(self.audit.path))
            else:
                already_ran = (self.fi_dir / "state.sqlite").is_file()
        try:
            _record, lines = _phased.prepare(self.quest_root, self.quest_id, already_ran=already_ran,
                                             data_quest=self._phased_data_quest(), key=self._phased_key(),
                                             mode=self._phased_mode())
        except OSError as e:
            self._log.warning("[phased] the data held back could not be kept apart from exploration (%r); nothing in "
                              "this quest can be confirmed, so its numbers stay exploratory", e)
            _phased.mark_compromised(self.quest_root, f"a data file could not be written at a start ({e!r})")
            return
        self._phased_log(lines)

    def _phased_turned_off(self) -> None:
        """A start with engine.phased off on a quest that ran with it on: the whole files back in inputs/data/, and
        what it means for confirming said in run.log."""
        try:
            self._phased_log(_phased.turned_off(self.quest_root, key=self._phased_key()))
        except OSError as e:
            self._log.warning("[phased] the whole data files could not be put back in inputs/data/ (%r); they are "
                              "kept in %s", e, _phased.store_dir(self.quest_root))

    def _phased_restore_inputs(self) -> None:
        """When a run stops (finished, paused or failed): the person's whole files back in inputs/data/. Not while a
        background job the run submitted is waiting: its tasks read inputs/data/ when they start."""
        if getattr(self, "_phased_job_pending", False):
            record = _phased.load(self.quest_root) or {}
            # Only while a part of a file is in place (exploring, or the confirm run): after the confirm result the
            # whole files are back already, and a compromised record has nothing left to lose.
            if self._phased_key() is None or not record.get("files"):
                self._log.info("[phased] a background job is waiting, so inputs/data/ keeps the part of the data this "
                               "stage may see until the quest is resumed")
                return
            # Without a container the kept rows are encrypted with this run's key, which ends with this run: a part left
            # in place for the job could never be completed again, so the whole files go back (put back below). While a
            # part is in place (exploring, or the confirm run) the job may then read every row, and nothing in this
            # quest can be confirmed; after the confirm result the files are whole already and nothing changes.
            if record.get("stage") in (_phased.EXPLORE, _phased.CONFIRM) and not record.get("compromised"):
                _phased.mark_compromised(self.quest_root, "a background job outlived the run of FI that held the key to "
                                                          "the held-back rows, so the whole files were put back for it")
                self._log.warning("[phased] a background job is waiting and outlives this run of FI, which holds the "
                                  "only key to the held-back rows: the whole files are put back now, so nothing in this "
                                  "quest can be confirmed")
        try:
            for line in _phased.restore_inputs(self.quest_root, key=self._phased_key()):
                self._log.warning("[phased] %s", line)
        except OSError as e:
            self._log.warning("[phased] the whole data files could not be put back in inputs/data/ (%r); they are "
                              "kept in %s", e, _phased.store_dir(self.quest_root) / "original")

    def _phased_seed_gap(self, state: QuestState) -> str:
        """Why a confirm run on new seeds could not differ from exploration's run (empty when it can). A deterministic
        trial contract (``run_cell``), or a script ``_node_execute`` found has no source of random numbers, or one whose
        identical runs it found ignore the seed. Seeds that merely agreed do not count: a script that can draw random
        numbers can agree on every seed it ran and still differ on a new one. Then the code: no file in ``code/`` that
        reads ``FI_REPLICATE_SEED`` while the experiment draws random numbers only from fixed seeds (a generator built
        without a seed draws new numbers on every run, so that one does differ). Under ``run_trial`` FI hands every
        trial its seed itself."""
        deterministic = ("the study has no randomness (every run gives the same numbers), so a run on new seeds only "
                         "repeats exploration's run")
        code = self.quest_root / "code"
        simulate = code / _split_run.SIMULATE_NAME
        entries = _trial_runner.entries(simulate) if simulate.is_file() else set()
        if "run_trial" in entries:
            return ""
        if "run_cell" in entries or state.get("result_json_no_random_source"):
            return deterministic
        script = simulate if simulate.is_file() else code / "experiment.py"
        if not script.is_file():
            return ""
        if state.get("result_json_replicate_seed_ignored"):
            return (f"the seed FI gives code/{script.name} (FI_REPLICATE_SEED) reaches none of its random numbers, and "
                    "its runs gave the same numbers, so a run on new seeds would repeat exploration's run")
        replicates = list(state.get("result_json_replicates") or [])
        this_pass = bool(replicates) and isinstance(replicates[0], dict) and \
            {k: v for k, v in replicates[0].items() if k != "_seed"} == (state.get("result_json") or {})
        if len(replicates) > 1 and this_pass:
            # This pass's runs are samples on several seeds (the list is this pass's: its seed 0 is the result in hand,
            # not an earlier script's left on the state): they differed, or they agreed while the script can draw
            # random numbers, which new seeds may still change.
            return ""
        # The seed read in the script or a module it imports from code/ (a multi-module project reads it in a helper);
        # FI's own helpers there (run.py sets a default seed) do not count.
        modules = _own_modules(script)
        if any(_unseeded_rng_calls(p) for p in modules):
            return ""  # a generator built without a seed draws new numbers on every run
        if any(_script_reads_replicate_seed(p) for p in modules):
            if self.config.execution.background_jobs or not _seeds_only_constants(modules):
                return ""  # a background job's driver passes the seed on to the job, not to a generator of its own
            return (f"code/{script.name} names FI_REPLICATE_SEED but seeds every random generator with a fixed number, "
                    "so a run on new seeds would repeat exploration's run")
        if not self.config.execution.background_jobs and not any(_script_has_random_source(p) for p in modules):
            return deterministic
        return (f"code/{script.name} never reads the seed FI gives it (FI_REPLICATE_SEED), so a run on new seeds "
                "would repeat exploration's run")

    def _phased_fresh_background_job(self) -> None:
        """The confirm run of a background job (not FI's own job array, which plans new tasks for the new seeds): the
        job exploration submitted is set aside (``job/state.explore.json``), so the driver submits a new one with the
        confirm run's seeds and data instead of reporting exploration's job as done."""
        if not self.config.execution.background_jobs:
            return
        if (self.quest_root / "code" / _trial_runner.SUBMIT_NAME).is_file():
            return
        state_file = self.quest_root / "job" / "state.json"
        if state_file.is_file():
            try:
                state_file.replace(state_file.with_name("state.explore.json"))
            except OSError as e:
                self._log.warning("[phased] exploration's background job record could not be set aside (%r); nothing "
                                  "in this quest can be confirmed", e)
                _phased.mark_compromised(self.quest_root, f"exploration's background job could not be set aside ({e!r})")
                return
            self._log.info("[phased] confirm stage: exploration's background job is set aside (job/state.explore.json), "
                           "so the confirm run submits its own")

    def _phased_confirming(self) -> bool:
        return _phased.enabled(self.config) and _phased.stage(self.quest_root) == _phased.CONFIRM

    def _phased_route(self, state: QuestState, route: str) -> str:
        """Where the quest goes after the evidence gate in two stages. In exploration, a write on an accepted result
        ends exploration: the protocol is frozen and the frozen design runs once more (``confirm``). In the confirm
        stage the confirm run's result is recorded and the quest writes it up, never going back to change the design
        (that would choose on data meant only for confirming)."""
        record = _phased.load(self.quest_root)
        current = str(record.get("stage")) if record else None
        if current is None:
            self._log.warning("[phased] the record of the two stages is missing or cannot be read; nothing is "
                              "confirmed, so the paper is written from exploration and says its numbers are exploratory")
            return route
        if record and record.get("not_applicable"):
            # The quest runs no experiment of its own (said once in run.log): it goes on exactly as it would without
            # explore-then-confirm, and the paper says nothing about stages it never had.
            return route
        if record and (record.get("compromised") or record.get("not_confirmable")):
            # Nothing can be confirmed any more (said in run.log when it happened): the quest goes on as it would
            # without explore-then-confirm (its protocol frozen before it writes), and the paper says its numbers are
            # exploratory.
            if route == "write" and current == _phased.EXPLORE:
                self._freeze_protocol_if_due(state, at_confirm=True)
            return route
        data_quest = bool((record or {}).get("data_quest"))
        confirm_edge = "confirm_data" if data_quest else "confirm"
        if current == _phased.CONFIRM and not _phased.confirm_run_started(self.quest_root):
            # The confirm stage began but its run has not (a stop before this step was saved): run it now, rather
            # than take the exploration result in hand for the confirm result.
            self._log.info("[phased] confirm stage: the confirm run has not been made yet; running it now")
            if not data_quest:
                self._phased_fresh_background_job()
            return "write" if (_phased.load(self.quest_root) or {}).get("compromised") else confirm_edge
        if current == _phased.CONFIRM and route != "write":
            self._log.info("[phased] confirm stage: the frozen design is run once, so the evidence gate's %s is not "
                           "followed; the confirm result is written up as it is", route)
            route = "write"
        if route != "write":
            return route
        if current in (None, _phased.EXPLORE):
            if not (self._runs_code(state) or data_quest) or not state.get("result_json"):
                self._log.info("[phased] exploration stage: there is no accepted result to confirm, so the paper is "
                               "written from exploration and says its numbers are exploratory")
                return "write"
            if data_quest and (record or {}).get("strategy") != _phased.HELD_BACK:
                # Its data was never split (the data-reading step says why in run.log): nothing to read once more.
                self._phased_log(_phased.mark_not_applicable(
                    self.quest_root, str((record or {}).get("why_no_data") or "the data could not be split by rows"),
                    key=self._phased_key()))
                return "write"
            self._freeze_protocol_if_due(state, at_confirm=True)
            seed_gap = "" if data_quest else (
                self._phased_seed_gap(state) if (record or {}).get("strategy") == _phased.FRESH_SEEDS else "")
            if seed_gap:
                self._log.warning("[phased] %s; no data was held back either, so this quest cannot be confirmed and the "
                                  "paper says its numbers are exploratory", seed_gap)
                _phased.mark_unconfirmable(self.quest_root, seed_gap)
                return "write"
            frozen = _frozen.load(self.quest_root)
            step = "data_load" if data_quest else "execute"
            explore_runs = sum(1 for e in _audit_log.read(self.audit.path)
                               if e.get("kind") == "node_completed" and e.get("node") == step) \
                if self.audit.path.is_file() else 0
            try:
                _record, lines = _phased.enter_confirm(
                    self.quest_root, explore_result=state.get("result_json"),
                    frozen_sha256=str(frozen["sha256"]) if frozen else None,
                    stride=max(1, int(self.config.engine.replicate_seed_stride)),
                    replicates=max(1, int(self.config.engine.execute_replicates)), explore_runs=explore_runs,
                    design_sha256=_frozen.sha256(state.get("design") or {}) if data_quest else None,
                    key=self._phased_key(),
                    # Whether the held-back rows were out of exploration's reach (a container, or encrypted with the
                    # quest's code read for paths out of the quest folder): judged now, before the confirm run.
                    isolation=self._phased_isolation() if (record or {}).get("strategy") == _phased.HELD_BACK else None,
                )
            except OSError as e:
                self._log.warning("[phased] the confirm run could not be given the data held back (%r); nothing in this "
                                  "quest can be confirmed, so the paper is written from exploration", e)
                _phased.mark_compromised(self.quest_root, f"the held-back data could not be put in place ({e!r})")
                return "write"
            self._phased_log(lines)
            if (_record or {}).get("compromised"):
                return "write"
            if not data_quest:
                self._phased_fresh_background_job()
            return "write" if (_phased.load(self.quest_root) or {}).get("compromised") else confirm_edge
        try:
            _record, lines = _phased.record_confirm(
                self.quest_root,
                state.get("result_json") if self._exec_produced(state.get("exec_result") or {}, 0) else None,
                design_sha256=_frozen.sha256(state.get("design") or {}) if data_quest else None,
                key=self._phased_key())
        except OSError as e:
            # The record was saved before the files are put back; the next stop puts them back again.
            self._log.warning("[phased] the whole data files could not be put back in inputs/data/ yet (%r)", e)
            return "write"
        self._phased_log(lines)
        return "write"

    def _route_after_review(self, state: QuestState) -> str:
        # Ordering is load-bearing:
        #
        # 1. ``must_flag_hits`` from any reviewer is non-bypassable.
        #    A non-empty list forces another revise pass even when
        #    ``review_loop = false`` would otherwise short-circuit
        #    to done. This is the gate the methodologist persona's
        #    MUST-FLAG checks (circular evaluation, single-point
        #    evaluation, weak baseline without re-run, pseudo-units)
        #    rely on to actually take effect. When every hit is a
        #    problem with the text (a claim nothing backs, a caption
        #    that describes what its figure does not show), the
        #    experiment stands and the route is ``rewrite``: back to
        #    ``write`` only, not to ``design``. When instead the review's
        #    must-fix evidence names something the run computed (a key of
        #    ``result_json``, a figure's underlying data, the experiment file),
        #    no rewrite can fix it and the route is ``re_execute``: the
        #    experiment is written and run again. That costs an iteration, so
        #    it happens at most ``_CODE_REEXECUTES`` times in a quest and a
        #    flag that survives the re-run falls back to the text routes.
        # 2. ``human_feedback_gate == "after_review"`` routes through
        #    the human-feedback node so the user gets a final say.
        # 3. Otherwise fall through to the legacy verdict-driven routing.
        review = state.get("review") or {}
        # An objection to the design that FI's record already keeps (.fi/review_objections.json, printed in QUALITY.md) is
        # not raised again: a rewrite cannot answer it, and the design is not changed for it.
        repeated = set(review.get("objections_repeated") or [])
        must_flag = [h for h in review.get("must_flag_hits") or [] if _objection_key(h) not in repeated]
        # The paper claims what FI's own records contradict: it is written again, with the contradiction named, up to
        # ``_RECORD_REWRITES`` times (outside engine.max_iterations, so even a one-round quest gets them). What is still
        # flagged after them is taken out of the paper (:meth:`_record_contradiction_hits`); what could not be taken out
        # is listed in QUALITY.md. The check reads the paper's words, so it never ends the quest (core/stop_power.py).
        if any(_hit_name(h) == _record_claims.HIT for h in must_flag):
            if int(state.get("record_rewrites") or 0) <= _RECORD_REWRITES:
                self._log.info(
                    "[route] the paper claims what FI's own records contradict — writing it again "
                    "(rewrite %d of %d, outside engine.max_iterations)",
                    int(state.get("record_rewrites") or 0), _RECORD_REWRITES)
                return "rewrite"
            self._log.warning(
                "[route] after %d rewrites the paper still has sentences FI's check of its records flags; they are listed "
                "in QUALITY.md, and the check reads the paper's words, so the quest goes on", _RECORD_REWRITES)
            must_flag = [h for h in must_flag if _hit_name(h) != _record_claims.HIT]
        if must_flag and state.get("iteration", 0) < self.config.engine.max_iterations:
            # Before the text routes: a flag about a computed value is not a
            # writing problem, and sending it to ``write`` is what let a paper
            # ship a deterministic limit of 0.0 that two checks had found.
            rerun_for = _review_sends_the_experiment_back(review, state)
            reexecutes = int(state.get("code_reexecutes") or 0)
            if rerun_for and reexecutes <= _CODE_REEXECUTES:
                self._log.info(
                    "[route] must_flag_hits=%s are about %r, which this run computed — "
                    "running the experiment again (re-execute %d of %d)",
                    must_flag, rerun_for, reexecutes, _CODE_REEXECUTES,
                )
                return "re_execute"
            if rerun_for:
                self._log.info(
                    "[route] must_flag_hits=%s are about %r again, but this quest has "
                    "already re-run its experiment — writing the paper again instead",
                    must_flag, rerun_for,
                )
            if (any(_hit_name(h) == _UNANSWERED_NOTE_HIT for h in must_flag)
                    and state.get("feedback_rewrite_for") == _refine_count(state) and _refine_count(state)):
                self._log.info(
                    "[route] the review still finds the person's note unanswered after a rewrite for it — "
                    "going back to the design",
                )
                return "revise"
            if _hits_need_only_a_rewrite(must_flag):
                self._log.info(
                    "[route] must_flag_hits=%s are all about the text — rewriting the paper",
                    must_flag,
                )
                return "rewrite"
            # An objection to the design (circular evaluation, a single-point evaluation, a name the reviewer made up)
            # that no check of FI's confirms: the paper is written again with the review in hand, the objection is kept
            # in FI's record (.fi/review_objections.json, printed in QUALITY.md; core/disclosure.py:run_record), and the
            # design is not changed for it. A redesign for it used to cost a whole round and recorded the new design as
            # one made after the results were seen.
            self._log.info(
                "[route] must_flag_hits=%s include objections to the design that no check of FI's confirms — writing "
                "the paper again; the objections are kept in QUALITY.md and the design is not changed", must_flag,
            )
            return "rewrite"
        # No round is left and the draft has numbers FI did not fill: those sentences are edited once (outside
        # engine.max_iterations) before any is recorded as untraced or taken out. The review held that back for this edit, so
        # every route below, each of which delivers the draft, waits for it; the routes above (a contradiction of FI's
        # records, the experiment run again) write or run again, and the next review asks for the edit again.
        if review.get("typed_pass_hits"):
            self._log.info(
                "[route] no round is left and %d sentence(s) hold numbers FI did not fill — editing those sentences once "
                "before any number is recorded as untraced (outside engine.max_iterations)", len(review["typed_pass_hits"]))
            return "rewrite"
        if self.config.pauses.review == "ask":
            return "human_feedback"
        if not self.config.engine.review_loop:
            return "done"
        verdict = review.get("verdict", "accept")
        if verdict == "revise" and review.get("reason_repeated"):
            return "done"
        if verdict == "revise" and state.get("iteration", 0) < self.config.engine.max_iterations:
            # The reviewer's own judgement, with no check of FI's behind it: the paper is written again with the review
            # in hand, the reviewer's reason is kept in FI's record (printed in QUALITY.md), and the design is not changed.
            self._log.info("[route] the review asks for a revision that no check of FI's confirms — writing the paper "
                           "again; the reviewer's reason is kept in QUALITY.md and the design is not changed")
            return "rewrite"
        return "done"

    def _route_after_citation_check(self, state: QuestState) -> str:
        """Where the review was sending the paper (to a person's decision, or out), as the citation check found it."""
        return "human_feedback" if state.get("citation_next") == "human_feedback" else "done"

    def _route_after_human_feedback(self, state: QuestState) -> str:
        """``refine`` goes to the writing step with the person's notes; ``accept`` / ``reject`` finalise.

        The notes are answered where they are cheapest first: the writer fixes what the text can fix and lists what
        needs a new or different experiment, and only that sends the quest back to the design
        (:meth:`_route_after_write`); after the rewrite the review's own routing applies. A refine used to go to the
        design every time, so a note about one sentence redesigned the study (and could drift its protocol).

        A human ``refine`` is honoured **regardless of ``max_iterations``** — it
        is a deliberate, interactive request, not the unattended review loop the
        cap exists to bound. The user stays in control: the refine runs one
        pass and lands back at the human-review gate, where they
        can refine again or accept. (Earlier this was gated on the iteration
        budget, so a refine submitted past the cap was silently dropped — the
        user clicked Refine and nothing happened.)"""
        hf = state.get("human_feedback") or {}
        action = hf.get("action", "accept")
        if action == "refine":
            kind = (state.get("optimise_refine") or {}).get("kind")
            if kind == _optimise_refine.SEARCH:
                return "search"
            if kind in (_optimise_refine.NEW_STUDY, _optimise_refine.UNCLEAR):
                return "ask_again"
            # A refine with no notes is FI re-opening a finished quest (``--reopen``, ``--update``): the steps its
            # changed settings affect run again, from the design, as before.
            notes = str(hf.get("feedback") or "").strip()
            history = [str(h.get("text") or "").strip() for h in (state.get("feedback_history") or [])
                       if isinstance(h, dict)]
            # Notes the history does not hold (a state saved before refines were recorded there) reach only the
            # design, which reads them from ``human_feedback``: send them there rather than drop them.
            return "rewrite" if notes and history and history[-1] == notes else "revise"
        return "done"

    def _route_after_write(self, state: QuestState) -> str:
        """After a draft: back to the design when the writer, answering a person's refine, named points that need a
        new or different experiment; to the code-writing step (the existing script extended) for a number the study
        lacks; to a redraw of the figures from the saved data for a layout note; on to the claim check and the review
        otherwise."""
        if state.get("refine_needs_experiment"):
            return "redesign"
        if state.get("refine_extend"):
            # A quest with no experiment has no script to extend: the missing data is collected again instead.
            return "collect" if state.get("no_simulation_resolved") else "extend"
        return "replot" if state.get("refine_layout") else "check"

    def _route_after_replot(self, state: QuestState) -> str:
        """A layout note the redraw could not carry out goes back to the writer once, so the paper says so."""
        return "write" if state.get("layout_missed") else "check"

    #: A top-level key of the design block that is not under ``protocol``; every other key of a flat answer is a protocol part.
    _DESIGN_TOP_KEYS = ("study_type", "variables")

    #: The most raw answers of the plan's model kept in ``.fi/plan_answers/``, and the most characters of each.
    _PLAN_ANSWERS_KEPT = 40
    _PLAN_ANSWER_CHARS = 60000

    #: The most characters kept of a whole rewritten plan (``_rewrite_plan``): the file is long, and the place a parser
    #: stopped at is usually far into it.
    _PLAN_FILE_ANSWER_CHARS = 400000

    # The key in ``provider.extra`` launch.py sets when the model picked in the VS Code chat panel replaced the config's
    # ``provider.model`` for this run: ``{"before": <the config's>, "after": <the chat panel's>}``.
    CHAT_MODEL_KEY = "chat_panel_model"

    #: What a rewrite for a flat objective may change in the protocol: how the design is described and searched. Every
    #: other key (the checks and their tolerances, thresholds, criteria, metrics, precision, the model) and the search's
    #: limits are what the results are judged by, and are put back as they were.
    _FLAT_MAY_CHANGE = frozenset({"optimisation", "grid", "in_each_run", "sub_studies", "numerical_settings"})
    #: Inside the search block, what judges a design (its limits, the margin and tolerance of "better", the target, the
    #: runs of the finer check, the assertions on its results): put back as it was too. The objective may become one
    #: that grades how close a design comes; which way is better may not change.
    _FLAT_SEARCH_JUDGED = ("constraints", "constraint_margin", "improvement_tolerance", "target", "check_runs",
                           "result_assertions")

    #: How many times, in all, a script that draws typed-in numbers is sent back.
    _FIGURE_DATA_REPAIRS = 2

    @staticmethod
    def _search_proposals_by_part(props: list[dict[str, Any]], before: Any) -> list[dict[str, Any]]:
        """A proposal that replaces the whole search block (or its objective) is compared and kept part by part: the parts
        FI applied are applied, and an open item names the part it put back and why, not the block. A part the proposal
        leaves as the design had it is not a change."""
        out: list[dict[str, Any]] = []
        for p in props:
            value = p["value"]
            if p["path"] in ("protocol.optimisation", "protocol.optimisation.objective") and isinstance(value, dict):
                parts = []
                for k, v in value.items():
                    found, now = _design_parts.get_part(before if isinstance(before, dict) else {}, f"{p['path']}.{k}")
                    if not found or json.dumps(now, sort_keys=True, default=str) != json.dumps(v, sort_keys=True, default=str):
                        parts.append({**p, "path": f"{p['path']}.{k}", "value": v})
                out.extend(Engine._search_proposals_by_part(parts, before))
            else:
                out.append(p)
        return out

    #: The protocol keys an audit may change: how the study is run, never what its results are judged by.
    _AUDIT_MAY_CHANGE = frozenset({"grid", "sub_studies", "in_each_run", "seed_policy", "failure_policy", "ci_method"})

    #: Inside the search block (``protocol.optimisation``), what judges a design. The objective's own parts are below.
    #: Besides the limits and the margin of "better": the conditions held fixed and the cases each design is judged over
    #: (they define what the objective is computed over), and the numerical settings (their search value and the finer
    #: values are the points the check's numerical error and its verdict are read from).
    _AUDIT_SEARCH_JUDGED = (*(k for k in _FLAT_SEARCH_JUDGED if k != "check_runs"), "numerical_settings", "fixed", "in_each_run")
    #: The two keys that give the runs of the finer check: ``check_runs``, or ``runs_per_evaluation`` when it is absent.
    _AUDIT_CHECK_RUNS_KEYS = ("check_runs", "runs_per_evaluation")
    _AUDIT_OBJECTIVE_JUDGED = ("quantity", "unit", "direction")

    _CANNOT_REPAIR_FILE = "cannot_repair.json"

    _CORRECTIONS = "oracle_corrections.json"

    @staticmethod
    def _run_has_no_result(state: QuestState) -> bool:
        """The run gave up, did not produce its results, or printed no result (the improve loop's test). A figure that failed
        after its repairs is not that: the results were produced."""
        result = state.get("exec_result") or {}
        produced = Engine._exec_produced(result, 1)
        figure_only = produced and result.get("returncode", 1) != 0
        return bool((state.get("exec_give_up_reason") and not figure_only) or not produced or not state.get("result_json"))

    #: The most different repairs one pass of the gate runs the study with; a finding beyond them is "not rerun".
    MAX_RERUNS = 6

    #: A reply up to this long is shown back with the request for a re-answer, so the model can restate what it said; a
    #: longer one (a real Haiku audit reply was 67,963 characters) is not repeated, and the request is answered afresh.
    _REANSWER_ECHO_CHARS = 12000

    # The setup questions the clarify node put to the person, kept across its pause (see _node_clarify_questions).
    # Not the web form's ``clarify_questions.json``: that one is cleared as soon as an answer is taken, before the
    # node runs again, and its presence means "a form is waiting".
    _CLARIFY_ASKED = "clarify_asked.json"

    _RUN_DATA_SUFFIXES = frozenset({
        ".csv", ".tsv", ".json", ".npy", ".npz", ".parquet", ".feather", ".pkl", ".pickle", ".h5", ".hdf5", ".xlsx", ".txt", ".dat",
    })
    _RUN_DATA_MAX_BYTES = 200 * 1024 * 1024
    # Names that are secrets or setup files, never a result table; matched whole, so `tokens.csv` is kept.
    _RUN_DATA_SKIP = (
        ".env*", "id_rsa*", "*.pem", "*.key", "token.json", "tokens.json", "access_token*", "refresh_token*",
        "credentials*.json", "client_secret*.json", "secret.json", "secrets*.json", "requirements*.txt", "constraints*.txt",
    )


# The engine's module-level names are one namespace, as they were when this was one file: a name set on ``core.engine``
# (a test that replaces a helper or a constant, say) is set in every part that holds it, and a name new to the engine in
# every part, so the code that moved to ``core/engine_base.py`` and ``core/engine_<stage>.py`` reads the same value as
# the code here. A module's own ``__dunder__`` names are never passed on. Setting a name on a part itself changes that
# part only.
_PARTS = tuple(sys.modules[__name__.rsplit(".", 1)[0] + "." + _part] for _part in (
    "engine_base", "engine_ideas", "engine_plan", "engine_data", "engine_code", "engine_improve", "engine_run", "engine_analyze", "engine_write", "engine_review"))


class _OneNamespace(type(sys)):
    def __setattr__(self, name: str, value: Any) -> None:
        new_here = name not in vars(self)
        super().__setattr__(name, value)
        if name.startswith("__") and name.endswith("__"):
            return
        for part in _PARTS:
            if new_here or name in vars(part):
                setattr(part, name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        if name.startswith("__") and name.endswith("__"):
            return
        for part in _PARTS:
            if name in vars(part):
                delattr(part, name)


sys.modules[__name__].__class__ = _OneNamespace
