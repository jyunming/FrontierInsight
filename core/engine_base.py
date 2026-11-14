"""What the stages of the research engine share: the imports, the constants, ``QuestState``, the small classes and
every module-level function of the engine. ``core/engine.py`` holds the ``Engine`` class, its run loop and its graph,
and imports every name of this module, so ``from core.engine import X`` gives the same object as before; the stages'
methods are in ``core/engine_<stage>.py`` and import the names they use from here.
"""

from __future__ import annotations

import ast
import copy
import asyncio
import fnmatch
import concurrent.futures
import functools
import hashlib
import inspect
import json
import logging
import math
import os
import platform
import io
import re
import tokenize
import shutil
import tempfile
import string
import sys
import threading
import time
import traceback
import unicodedata
import uuid

import yaml
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, TypedDict, get_args

# User-supplied async function that collects answers to clarify-node
# questions. Receives the ``clarify_questions`` dict and must return
# the answers dict (same keys, resolved values).
ClarifyCallback = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

# Human-feedback gate callback. Receives the review snapshot dict
# (verdict / score / strengths / weaknesses / suggestions / paper_md_path)
# and returns ``{"action": <accept|reject|refine>, "feedback": "..."}``.
HumanFeedbackCallback = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

import warnings

# langchain_core adds its own "always show" filter for its deprecation warnings when it is first imported: imported first,
# so the one below comes before it.
import langchain_core._api.deprecation  # noqa: E402,F401

with warnings.catch_warnings():
    # langgraph's checkpoint serializer builds langchain_core's Reviver at import without `allowed_objects`, which warns
    # that the default will change. FI's checkpoints hold only plain state (dicts, lists, text, numbers), never a
    # LangChain object, so the new default does not touch them. Only this warning, only for this import: it was printed
    # before even `--help`.
    warnings.filterwarnings("ignore", message=r"The default value of `allowed_objects` will change")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.types import Command, interrupt

from .latex_text import escape_latex_backslashes
from .vscode_bridge import BridgeError
from .proc_tree import AsyncProcessTree
from . import acceptance as _acceptance
from . import cite_labels as _cite_labels
from . import agent_coder as _agent_coder
from . import audit_log as _audit_log
from . import audit_open_items as _audit_open
from . import confirmations as _confirmations
from . import cross_check_items as _cross_check_items
from . import record_anchor as _record_anchor
from . import evidence as _evidence
from . import frozen_protocol as _frozen
from . import math_safe_cut as _math_safe_cut
from . import design_parts as _design_parts
from . import todo as _todo
from . import crash_kind as _crash_kind
from . import plan_scope as _plan_scope
from . import oracle_card as _oracle_card
from . import oracle_triage as _oracle_triage
from . import oracle_challenge as _oracle_challenge
from . import code_layout as _code_layout
from . import changelog as _changelog
from . import code_project as _code_project
from . import contract as _contract
from . import contract_parts as _contract_parts
from . import rerun_from as _rerun_from
from . import attempt_records as _attempts
from . import attempt_memory as _memory
from . import phased as _phased
from . import phased_isolation as _phased_isolation
from . import disclosure as _disclosure
from . import figure_claims as _figure_claims
from . import paper_facts as _paper_facts
from . import paper_values as _paper_values
from . import run_facts as _run_facts
from . import thinking_capture as _thinking
from . import metric_spec as _metric_spec
from . import mc_error as _mc_error
from . import run_manifest as _run_manifest
from . import trial_runner as _trial_runner
from . import goal_coverage
from . import citation_support as _cs
from . import figure_facts as _figure_facts
from . import paper_patch
from . import paper_sections as _paper_sections
from . import paper_scope as _paper_scope
from . import paper_trim
from . import answer_reading as _reading
from . import plan as _plan
from . import criteria as _criteria
from . import improve as _improve
from . import in_run as _in_run
from . import plan_settings as _plan_settings
from . import receipts as _receipts
from . import novelty as _novelty
from . import source_text as _source_text
from . import experiment_deps as _experiment_deps
from . import figure_data_check as _figdata
from . import gate_receipt as _gate_receipt
from . import plot_scope as _plot_scope
from .agent_build import AgentBuildMixin, RepairForAgent
from .agent_build import _Lines as _AgentLines
from .agent_build import writes_code as _writes_code
from .run_checks import RunChecksMixin
from .result_checks import ResultChecksMixin
from .static_checks import StaticChecksMixin
from . import record_claims as _record_claims
from . import numeric_warnings as _numeric
from . import oracle_check as _oracle
from . import oracle_forms as _forms
from . import oracle_review as _review
from . import topic_requirements as _topic_req
from . import trial_table as _trial_table
from . import reference_values as _reference_values
from . import results_by_setting as _results_by_setting
from . import plan_ledger as _ledger
from . import plan_steps as _plan_steps
from . import equation_tests as _equation_tests
from . import outside_run as _outside_run
from . import code_reach as _code_reach
from . import shown_reach as _shown_reach
from . import rerun_check as _rerun_check
from . import code_review as _code_review
from . import review_pairs as _pairs
from . import quality_report as _quality_report
from . import code_findings as _code_findings
from . import write_personas as _write_personas
from . import grid_keys as _grid_keys
from . import code_delta as _code_delta
from . import finding_repair as _finding_repair
from . import function_steps as _function_steps
from . import second_impl as _second_impl
from . import step_passages as _step_passages
from . import hidden_check as _hidden
from . import refinement_check as _refine
from . import accepted_checks as _accepted
from . import optimisation_plan as _optim
from . import scope_cards as _scope_cards
from . import optimise as _optimise
from . import optimise_refine as _optimise_refine
from . import optimise_search as _optimise_search
from . import best_design_report as _best_report
from . import protocol_check as _protocol
from . import retractions as _retractions
from . import script_reply as _script_reply
from . import skeleton as _skeleton
from . import analysis_frame as _analysis_frame
from . import split_run as _split_run
from . import stop_power as _stop_power
from . import stats as _stats
from .config import (
    DEFAULT_PAPER_FORMAT,
    Config,
    NON_SCIENTIFIC_PAPER_FORMATS,
    SCIENTIFIC_PAPER_FORMATS,
    resolve_page_limit,
)
from .execution import DockerLimits, ExecutionResult, make_executor, pip_failure_summary, scrub_secrets
from .knowledge import (
    FOUNDATIONAL_MAX_SUGGESTED,
    FOUNDATIONAL_SUGGESTED,
    WORK_SCOPE_PAPERS,
    WORK_SCOPE_PAPERS_AND_BOOKS,
    Knowledge,
    RetrievedDoc,
    _content_stems,
    _doc_dedup_keys,
    _is_bot_check_title,
    _normalize_title,
    _titles_match,
    foundational_work_text as _foundational_work_text,
)
from .protocol import derive_protocol, route_for_topic_type
from .provider import (
    CLI_PROVIDERS as _CLI_PROVIDERS_SET,
    CALL_ATTEMPTS as _CALL_ATTEMPTS,
    LAST_CALL as _LAST_CALL,
    FallbackLLMClient,
    LLMClient,
    PROXY_PROVIDERS,
    ProxySupervisor,
    append_cost_row,
    billing_stop,
    blocked_stop,
    claude_cli_account_line,
    missing_step_api_keys,
    model_for_node,
    node_provider_for,
    outcome_of as _outcome_of,
    step_provider_config,
    step_provider_lines,
    _ModelAnswerProblem,
    _is_fatal_provider_error,
    is_auth_error as _provider_is_auth_error,
    sign_in_advice as _provider_sign_in_advice,
    ModelAnswerTruncated,
    ProviderLimitError,
    ProviderUnusable,
    limit_resumes_at,
    resolve_endpoint_async,
    set_model_call_archive as _set_model_call_archive,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents"

# ---- what a person watching a quest sees by default -------------------------------------------------------------------
#
# The engine logs hundreds of internal lines per quest (a skill that failed its self-test, a reply that needed a repair,
# an internal decision's raw fields) — real, useful for debugging, and unreadable to someone who did not write this code.
# `run.log` in the quest folder keeps every one of them, unfiltered, always. What a person watching the quest sees while
# it runs — the CLI console, the web page's live log, and VS Code's chat — is instead a short, fixed set of one-line
# stage updates, marked with `progress=True` on the log record and picked out by `_ProgressOnly` below. A stage not in
# this map (an internal passthrough, e.g. `pause_after_literature`) simply has no line: silence there is correct, not a
# gap. `PROGRESS_LOG_NAME` is a second file with only these lines, for the web page and for `--progress <quest_id>`.
STAGE_PROGRESS: dict[str, str] = {
    "clarify": "Checking the setup.",
    "ideate": "Choosing the angle to take.",
    "literature": "Searching the literature.",
    "scope": "Offering the studies this topic could mean.",
    "select_skills": "Choosing which tools to use.",
    "plan": "Writing the plan.",
    "design": "Designing the experiment.",
    "implement_outline": "Writing the code.",
    "implement": "Writing the code.",
    "replot_layout": "Arranging the figures again.",
    "execute": "Running the experiment.",
    "analyze": "Analyzing the results.",
    "cross_check": "Checking the results against the literature.",
    "write": "Writing the paper.",
    "claim_check": "Checking every claim against the evidence.",
    "citation_check": "Checking every citation against its source.",
    "review": "Reviewing the paper.",
    "auto_collect_data": "Looking for data.",
    "data_load": "Loading the data.",
}
PROGRESS_LOG_NAME = "progress.log"


class _ProgressOnly(logging.Filter):
    """Lets through only the log records :meth:`Engine._progress` marked — the short list above, plus the handful of
    one-off events (a pause, the final evidence line) that are rare and important enough to earn a place beside them.
    Everything else (the great majority of what the engine logs) still reaches ``run.log``, just not this handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "progress", False))

_FIGURE_SUFFIXES = frozenset({".png", ".svg", ".jpg", ".jpeg", ".pdf"})

# Under the Docker sandbox, the skills mounted into the experiment's container
# (names), kept in ``.fi/`` for a ``--watch`` that has no quest state.
_SKILL_MOUNTS_FILE = "skill_mounts.json"

# Default sampling temperature for generative nodes (ideate, write, analyze…).
_DEFAULT_CHAT_TEMPERATURE = 0.2

# How long the environment record waits for `pip freeze` before it records the timeout and stops it (with everything it
# started) instead.
_PIP_FREEZE_TIMEOUT_S = 30.0
# Judgment / gate / classifier nodes: their job is to reach a *verdict* or a
# routing decision (sufficient vs broaden, accept vs revise, supported vs not),
# where run-to-run flakiness means the same corpus can flip the route and
# trigger wasted broaden loops or spurious revise cycles (audit: "gates are
# non-deterministic"). These run at temperature 0 so the decision is
# reproducible wherever the transport honours it (HTTP + vscode_bridge; CLI
# transports that don't expose temperature are unaffected — no regression).
# The ``review_panel.`` prefix covers every per-persona panel call.
_DETERMINISTIC_GATE_NODES = frozenset({
    "evidence_gate",
    "review",
    "claim_check",
    "citation_check",  # every citation read against the retrieved text of its source, last, on the final paper
    "oracle_review.citations",  # ... while no step of its own is set for it
    "cross_check",
    "relevance_guard",
    "literature_screen",
    "literature_novelty",  # a judgement of the sources, not a draft
    "figures",
    "write.trim",
    "oracle_review",  # a referee's verdict on the plan's checks
    "design_review",  # the design audit, when another step reads the design than the one that wrote it
    "code_review",  # the code read against the plan, when it has a step of its own
    "oracle_review.recompute",  # a failing check's expected value worked out again (core/oracle_triage.py)
    "oracle_review.code",  # the code read against the plan, while no step of its own is set for it
    "second_impl.adjudicate",  # which of two versions of a function that disagree is wrong (core/second_impl.py)
    "oracle_review.topic",  # what the topic asks for, read from its text and looked for in the plan (core/topic_requirements.py)
})


def _temperature_for_node(node: str | None) -> float:
    """Temperature for a node's chat call: 0 for gate/verdict/classifier nodes
    (deterministic routing), the generative default otherwise."""
    if not node:
        return _DEFAULT_CHAT_TEMPERATURE
    if node in _DETERMINISTIC_GATE_NODES or node.startswith("review_panel."):
        return 0.0
    return _DEFAULT_CHAT_TEMPERATURE


@dataclass
class QuestArtifacts:
    quest_id: str
    quest_root: Path
    paper_md: Path | None = None
    paper_pdf: Path | None = None
    figures_dir: Path | None = None
    bundle_manifest: Path | None = None
    raw_state: dict[str, Any] = field(default_factory=dict)


class QuestState(TypedDict, total=False):
    quest_root: str  # set by ``Engine._audited`` for every node; the engine's own root, never a node's output
    topic: str
    title: str
    iteration: int
    # Skills chosen for this quest by ``select_skills`` — the names that
    # actually reach the design and implement prompts. Distinct from
    # ``config.engine.skills``, which only bounds the candidate set.
    selected_skills: list[str]
    # The selection record: chosen names, per-skill reasons, how many
    # candidates there were, and any names the model invented. Kept so
    # "why did this quest not use ambit?" is answerable after the fact.
    skill_selection: dict[str, Any]
    # Range assertions contributed by the selected skills, merged with the
    # design's own by ``_assertion_violations``.
    _skill_assertions: list[dict[str, Any]]
    # Clarify-node state. Both dicts share the same 5 keys
    # (`comparative_baseline`, `empirical_vs_theoretical`,
    # `success_metric`, `budget`, `output_kinds`); `clarify_questions`
    # carries `{question, default}` per slot, `clarify_answers` carries
    # the resolved values (default or user-overridden).
    clarify_questions: dict[str, Any]
    clarify_answers: dict[str, Any]
    clarify_done: bool
    # The study a person picked before the plan (core/scope_cards.py), or None: nobody picked one.
    scope_card: dict[str, Any] | None
    title_confirmed: bool
    ideas: list[dict[str, Any]]
    chosen_idea: dict[str, Any]
    # Ideate self-reflection result. Optional; describes what
    # the agent considered before locking in `chosen_idea`.
    ideate_critique: dict[str, Any]
    # Ideate tournament result. Optional; present only when
    # `engine.ideate_tournament: true`. Carries the match table, win
    # counts, and outcome label ("confirmed" / "swapped" /
    # "inconclusive_fallback") for visibility + future Axon write-back.
    ideate_tournament: dict[str, Any]
    #: Whether the retrieved sources already answer the study's question (core/novelty.py): the checked record, or absent
    #: when the call was off, failed or gave nothing usable. Disclosed only; kept also in .fi/literature_novelty.json.
    literature_novelty: dict[str, Any]
    # How chosen_idea was picked: "ensemble", "tournament", "reflection" or "model" (the ideate node's own pick).
    idea_selection: str
    literature: list[dict[str, Any]]
    # Iterative-literature counter. Incremented on every entry into
    # ``_node_literature``. First entry sets it to 1; ``broaden_lit``
    # re-entries from ``cross_check`` bump it. Used to (a) emit a
    # distinct INFO log per pass, (b) decide whether to dedup-merge
    # vs replace the literature list, and (c) optionally cap
    # additional retrievals at ``engine.max_iterations + 1``.
    literature_iter: int
    # The search query the literature node actually sent -- derived from the
    # topic by the model when it could be, else the old title+topic string.
    # With facet queries this is the first facet; all of them are in
    # ``literature_queries``.
    literature_query: str
    literature_queries: list[str]
    # The literature step's query set as a record: the queries, what they answer (``key``: the derivation prompt and
    # the search pass), the prompt's hash, the model, why they were derived, and whether they were reused.
    literature_query_set: dict[str, Any]
    # True when ``literature_query`` is a keyword query (derived from the topic, or the requery that found sources),
    # False when it is the fallback ``title + topic`` concatenation.
    literature_query_derived: bool
    design: dict[str, Any]
    # Every version of the design, in order. Entry 0 is the pre-registration
    # (stated before any result existed); later entries are flagged
    # ``post_hoc`` and carry what sent the engine back to design. `review` and
    # `cross_check` can both re-enter design, so a hypothesis CAN legitimately
    # be rewritten after its results are known -- but a finished paper looks
    # identical either way, which is what makes the record necessary.
    design_history: list[dict[str, Any]]
    # How many times the simulation was sent back because its run manifest differed from the frozen protocol.
    run_manifest_failures: int
    #: The run record's last difference was in the analysis alone (so its repairs have a budget of their own, and spend no
    #: crash repair); kept in the state so a resume knows.
    run_manifest_analysis_only: bool
    figure_data_repairs: int
    record_rewrites: int
    #: The draft was written under the rule that FI fills every result number (the writer was given the sheet of keys), so
    #: the typed-number check holds it to that rule; a draft from an older build, or a study with no trial record, is not.
    paper_keys: bool
    #: The draft was made by editing passages of the earlier one (the passage editor, or the one edit of the sentences with
    #: numbers FI did not fill that a whole-paper draft gets when no round is left, ``Engine._typed_pass_hits``), not by the
    #: whole-paper writer: a number FI cannot trace that is left in it stays and is recorded as untraced (QUALITY.md);
    #: an unfilled result key is taken out.
    draft_edited: bool
    # Two-stage implement scaffold from ``_node_implement_outline``.
    # Carries ``{scaffold, functions, data_flow, constants,
    # result_json_template, deps}`` for the body node to consume.
    # Empty dict on legacy resume (pre-Phase-2 checkpoint), in which
    # case ``_node_implement`` falls back to the original
    # ``agents/implement.md`` one-shot prompt.
    implement_outline: dict[str, Any]
    #: The hash of the design the outline was written for (:meth:`Engine._outline_design_sha`): a redesign gets a new outline,
    #: a resume of the same design keeps the one it has. Absent in a checkpoint made before it existed (then reused).
    implement_outline_for: str
    code: str
    deps: list[str]
    exec_result: dict[str, Any]
    figures: list[str]
    # Attribution for license-clean illustrative figures pulled from the web
    # (knowledge.fetch_web_figures): one dict per figure with file / caption /
    # source_url / license / attribution, so FI's caption credits them and the
    # references record their provenance.
    figure_credits: list[dict[str, Any]]
    # What each experiment figure draws, by file name, as the plot-style
    # bootstrap recorded it on savefig (core/plot_style.py).
    figure_records: dict[str, Any]
    result_json: dict[str, Any]
    # Multi-seed replication: when ``engine.execute_replicates > 1``,
    # ``_node_execute`` runs the script N times with different seeds
    # and aggregates the N ``RESULT_JSON`` outputs into this list
    # (one dict per replicate). The primary ``result_json`` field
    # carries the first replicate so existing single-seed paths
    # downstream stay unchanged. ``_node_analyze`` reads this list
    # when present and aggregates numeric fields with mean ± std.
    result_json_replicates: list[dict[str, Any]]
    result_json_trials: bool  # the one result holds every trial FI ran (the trial contract): pool it as the replicates
    # True when every configured seed ran and every one gave byte-identical
    # results while the script can draw random numbers (the replicates are kept).
    # Despite the name it is not proof of determinism; it distinguishes "no
    # spread because every seed agreed" from "no error bars because nothing could
    # be aggregated", and is never read as "new seeds cannot change the result".
    result_json_deterministic: bool
    # True when the generated script's runs were ONE run repeated: it never reads
    # ``FI_REPLICATE_SEED``, or reads it without the value reaching any generator,
    # or (with the flag below) has no random source at all. No replicate list is
    # published in that case; this records WHY, so analyze can tell the paper it
    # holds a single measurement rather than quietly losing its error bars.
    result_json_replicate_seed_ignored: bool
    # True (only alongside the flag above) when the script has no random source at all: a deterministic study, where
    # one run is the whole result and the analyze note says so instead of calling it unreplicated.
    result_json_no_random_source: bool
    # The design's asserted paths some run of this quest has reported (``plausibility.bounded_paths``). A later result
    # that no longer reports one of them is a plausibility violation: a repair must not rename or drop a bounded quantity.
    bounded_seen: list[str]
    # How many trials the last run's own manifest lists as failed; None when no manifest was read. A count of failures
    # the protocol's ``failure_policy`` names is not held to be a trivial answer at 0 when this is 0 (``_zero_expected_paths``).
    manifest_failed_trials: int | None
    # Execute-repair loop counter + history. The reflect
    # node increments `exec_reflect_iter` and appends a one-line
    # record per attempt, so analyze/write/review can describe what
    # was fixed.
    exec_reflect_iter: int
    # How many times the gate sent a fault of the analysis on the recorded results back to be repaired (its own budget,
    # ``run_checks.GATE_REPAIRS``; the crash repairs above are not spent by it and do not spend it).
    gate_repairs: int
    exec_reflect_history: list[dict[str, Any]]
    exec_give_up_reason: str
    # The quest stopped with no paper (``needs/STUCK.json``): what it could not do and what it tried.
    stuck: dict[str, Any]
    # True from the moment the reflect node writes a patch until ``execute``
    # runs it. The router runs a pending patch even when it used the last
    # repair attempt; otherwise the paper is written from the run before it,
    # next to code that never ran.
    exec_patch_pending: bool
    # True when the person, at the stop for numeric warnings, resumed without changing the script: the run is used
    # as it is and its warnings are not raised again. Cleared by the next run of the experiment.
    numeric_warnings_accepted: bool
    # True once the run's figures have been sent back for a redraw because a
    # legend or a title was drawn over what a reader needs (see
    # ``_figure_overlap_findings``). One such round per quest, so a model that
    # cannot lay a figure out is not asked again.
    figure_overlap_repaired: bool
    # Set once figure text that shows the code's name for a quantity was sent to the redraw from the saved numbers
    # (``_node_write``, core/figure_labels.py): once per script, since a new script starts it over (``_FRESH_SCRIPT``).
    figure_labels_redrawn: bool
    # Set by the execute-repair loop when a script exits 0 but every
    # numeric metric is ~0 (a degenerate run) and the repair attempts
    # couldn't produce real numbers. analyze/write/review read this so a
    # broken run is framed as a failure note, not a real result.
    degenerate_result: bool
    analysis: dict[str, Any]
    # Cross-paper check per finding. List of per-finding
    # records carrying supporting / conflicting / neutral classifications.
    cross_check: list[dict[str, Any]]
    # Where the quest goes after cross_check ("write" / "broaden_lit" / "redesign"), decided by the node before it
    # spends a round; the router reads it.
    cross_check_route: str
    # The typed research contract (core.protocol.ResearchProtocol, stored
    # as a dict) the quest is held to: topic_type, source_policy, baseline,
    # success_metric, … Derived at the evidence_gate from clarify + config.
    research_protocol: dict[str, Any]
    # Evidence-sufficiency assessment from the evidence_gate node:
    # ``{"verdict": "sufficient"|"broaden"|"insufficient", "rationale",
    # "gaps": [...], "n_sources", "n_supporting"}``. Read by the writer so
    # a thin-evidence paper frames its limits honestly.
    evidence_assessment: dict[str, Any]
    # How many times the evidence_gate has sent the quest back to broaden
    # the literature. Bounds the broaden loop (``evidence_gate_max_broaden``).
    evidence_broaden_count: int
    # How many times the evidence_gate has given a simulation with no results another try (at most once): back to design,
    # or, when the script could not run to the end, its code written again from the same design.
    evidence_no_result_retries: int
    # Times the plan was asked to fit the time FI measured, after the study was too long to run (at most once; it
    # spends no design iteration: nothing ran).
    evidence_fit_redesigns: int
    # plan.md was rewritten for the next design (a study that fits the time FI measured): the design node reads its
    # design block again instead of asking for a new design. Cleared by the design node.
    design_from_plan: bool
    # The search's objective did not move across the design space (core/optimise_search.py, ``flat_objective``) and
    # plan.md was rewritten once from what FI measured: never again for this quest, and the search runs its whole budget.
    flat_objective_redesigned: bool
    # What FI measured of that search (the runner's fact), read by the design node to record the change it approved.
    flat_objective_fact: dict[str, Any]
    # That try was the code written again from the same design (the script crashed), not a new design.
    evidence_code_rewritten_at: int  # the design round (``iteration``) whose code was written again after no result
    paper_md: str
    # A fingerprint of the study the current draft was written from (design,
    # analysis, results, figures, cross-check and the user's feedback). A revise
    # edits the flagged passages of that draft only while the fingerprint still
    # matches, i.e. while nothing the paper is based on has been run again.
    paper_basis: str
    # Claim grounding: which paper claims trace to evidence
    # (experiment / citation / unsupported), from the claim_check node.
    claim_grounding: dict[str, Any]
    # Why the claim check could not run on the current draft (empty once it
    # runs). The review then forces ``citations_unchecked``: a draft whose
    # citations nobody checked cannot be accepted.
    claim_check_failed: str
    # The last citation check (core/citation_support.py): what it found and what it did, in counts.
    citation_check: dict[str, Any]
    # Where the review sent the paper that the citation check then read: "done" or "human_feedback".
    citation_next: str
    # How many times the review has sent the experiment back to be written and
    # run again because a must-flag named something this run computed. At most
    # ``_CODE_REEXECUTES``; each one costs an iteration.
    code_reexecutes: int
    review: dict[str, Any]
    # Human-feedback gate state. Populated by ``_node_human_feedback``
    # when ``engine.human_feedback_gate == "after_review"``. ``action``
    # is one of "accept" / "reject" / "refine"; ``feedback`` carries
    # the user's freeform text on refine, which the design node reads
    # on the next revise loop. Pre-resume the dict is empty.
    human_feedback: dict[str, Any]
    # Who accepted the result (core/acceptance.py): a person's receipt {"by": "person", "via": <interface>, "who",
    # "at", "question", "answer", "note", "evidence_sha256", "limits_shown", "paper_sha256"; and, when the record worked
    # out on resume differs from the one shown, "evidence_sha256_at_accept" and "limits_at_accept"} or
    # {"by": "automatic", "via": ...}. Set by ``_node_human_feedback`` on an accept, cleared on a refine; none
    # (no review pause) counts as automatic, which keeps the evidence one level below publication_ready.
    acceptance: dict[str, Any]
    # Cumulative refinement asks across the quest's revise iterations.
    # One entry per refine round: ``{"iteration": int, "text": str}``.
    # The design node reads ALL of these on every revise pass so a
    # later iteration doesn't drop an earlier ask. Pre-resume empty.
    feedback_history: list[dict[str, Any]]
    # A person's refine goes to the writing step first (``_route_after_human_feedback``). ``refine_written_for``: how
    # many refines the writer has answered (so each is answered once). ``refine_needs_experiment``: the points
    # of it the writer said need a new or different experiment (then the quest goes back to the design).
    # ``refine_scope``: what the refine came to, ``paper`` or ``experiment`` (empty on a write that answered none).
    # ``feedback_rounds_from``: the first round of ``feedback_history`` the review is shown (the rounds since the last
    # gate answer). ``feedback_rewrite_for``: the refine count when a rewrite was made for an unanswered note; the
    # same note flagged again after it sends the quest to the design.
    refine_written_for: int
    feedback_rounds_from: int
    # The engine's own tally of the model calls it made, per step (``node`` key), for the quest's whole life: the record
    # of its calls (.fi/model_calls.jsonl) must hold at least this many lines per step (core/attempt_records.py).
    model_call_counts: dict[str, int]
    # How far FI wrote each of the quest's record files at the end of the last step (core/record_anchor.py): checked when
    # the quest starts again, so a line another program added is moved aside instead of taken for FI's.
    record_anchor: dict[str, Any]
    # The improve loop (core/improve.py): the rounds spent in this quest (one budget, ``engine.improve_rounds``, shared by
    # every pass), whether the next run is the kept version's full run (``improve_rerun``), and what the loop did.
    improve_rounds_used: int
    improve_rerun: bool
    improve_fell_back: bool
    improve_summary: dict[str, Any]
    feedback_rewrite_for: int
    refine_needs_experiment: list[str]
    # The points of a refine that only need a number the study lacks (``refine_extend``: the existing script is extended
    # and run again) or the figures arranged again (``refine_layout``: they are redrawn from the saved data).
    refine_extend: list[str]
    refine_layout: list[str]
    # What a refine asked to have added that the run could not add (the extension did not come back, or the collection
    # found nothing new); the paper says so.
    extend_missed: list[str]
    # The layout notes the redraw could not carry out (a note about the paper's text or the order of its figures, or
    # nothing to redraw from); the paper says so.
    layout_missed: list[str]
    # The layout notes whose redraw failed on both tries (the figures are as they were): the review gate tells the
    # person their request was not applied, and clears this once they answer.
    layout_not_redrawn: list[str]
    # A refine of a search for the best design, read without a model (core/optimise_refine.py): ``kind`` "search" (the
    # search goes on from the best design so far: human_feedback -> execute), "new_study" or "unclear" (nothing is
    # changed and the review pause says why), with its plain sentence ``says``. Empty for any other refine.
    optimise_refine: dict[str, Any]
    # A script extended for a refine: what was asked (``asked``) and the result names the results held before it
    # (``before``), so each write after the run can check that what it added is in the paper. ``extend_unreported``:
    # what the paper still leaves out after the writer was asked once more; the review pause says so.
    extend_check: dict[str, Any]
    extend_unreported: list[str]
    refine_scope: str
    # Names of pause-points the engine has already paused at on this
    # quest (e.g., ``"after_design"`` / ``"after_paper"``). Used as a
    # secondary "already paused" signal for test paths that mock the
    # interrupt out; the authoritative signal in the real engine is
    # the disk marker ``<quest_root>/.fi/paused_at_<stage>.flag``
    # because LangGraph's ``interrupt()`` raises BEFORE a node returns
    # a state patch, so any partial state-list update never lands in
    # the checkpoint. Pre-resume empty.
    user_pauses_fired: list[str]
    # Files the user dropped into ``<quest_root>/inputs/data/``. The
    # analyze node reads these on resume so its prompt can reference
    # user-supplied datasets alongside the engine's RESULT_JSON.
    # Populated by ``_pick_up_user_dropped_datasets`` on the
    # post-pause re-entry. Empty on quests that never paused.
    user_supplied_datasets: list[str]
    # Per-persona reviews from the panel, before moderation.
    # One entry per `engine.review_panel` member, each with the same
    # JSON shape the single reviewer produces plus a `persona` field.
    review_panel: list[dict[str, Any]]
    # no-simulation mode — the engine doesn't write/run experiment
    # Python; instead it pauses after `design`, asks the user to drop
    # real-world data into `<quest_root>/data/`, then resumes with
    # `data_load` synthesizing the result_json from those files.
    # Resolved at the clarify node: True if `engine.no_simulation: true`
    # in YAML OR the clarify answer for `empirical_vs_theoretical` is
    # "empirical". See `_resolve_no_simulation_from_clarify`.
    no_simulation_resolved: bool
    # survey mode — a descriptive literature / history synthesis with NO
    # experiment AND NO dataset. A stronger form of no_simulation: the
    # engine skips the experiment (implement/execute) AND the data path
    # (auto_collect_data/wait_for_data/data_load/web_plots), routing
    # design → web_figures → analyze so the paper synthesises the
    # literature directly. Resolved at clarify: True if `engine.survey_mode:
    # true` in YAML OR the clarify agent classified `topic_shape == 'survey'`
    # (a history/overview/"evolution of X" topic). Survey implies
    # no_simulation_resolved. See `_resolve_survey_from_clarify`.
    survey_mode_resolved: bool
    # Populated by the wait_for_data node's interrupt-resume payload.
    # List of absolute paths the user dropped into `<quest_root>/data/`.
    # The data_load node walks them, classifies, and synthesizes a
    # result_json compatible with downstream nodes (analyze, write).
    data_files: list[str]
    # Number of docs the auto_collect_data node successfully
    # wrote into `<quest_root>/data/auto_collected/`. 0 means the node
    # was a passthrough (auto-collect disabled, knowledge disabled, or
    # Axon returned no hits), in which case wait_for_data falls back
    # to pausing for user-supplied data. Positive values let the user
    # see in run.log / state how much of their data load came from
    # the agent vs from manual drops.
    auto_collected_count: int


# Injected into the design/write prompts when a quest runs in
# ``no_simulation`` mode. Without these, the design node plans an
# "executable Python experiment" (the default prompt) that the no-sim
# path never runs, and the writer then narrates the engine's own
# mechanics — "the automated pipeline failed to run the simulation",
# "the system executed a literature search instead" — as if that were a
# scientific finding. The paper must read as a standalone analytical
# study, never as a report on the tool that produced it.
_NO_SIM_DESIGN_DIRECTIVE = (
    "> **THIS IS A NO-SIMULATION STUDY.** The quest is configured "
    "`no_simulation`: no experiment or simulation will be run. "
    "**Ignore the \"executable Python experiment\" / \"must produce a "
    "figure\" instructions below** — they do not apply. Instead design an "
    "**analytical or observational study** grounded in the literature (and "
    "any sources the user supplies): a derivation, a structured comparison, "
    "a synthesis, or an interpretation of existing evidence. `method` "
    "describes how that existing evidence is analyzed, NOT a program to "
    "run. Set `\"dependencies\": []` and `\"figures_planned\": []` — do not "
    "plan an executable experiment to produce figures (any figures come "
    "later from charting the numbers your sources actually report, not "
    "from code you specify here). There is no 'planned experiment' that "
    "could later be 'not run' — frame the study as complete in its own "
    "right.\n"
)
_NO_SIM_WRITE_NOTE = (
    "**This is a no-simulation study — by design, no experiment or "
    "simulation was run.** Write it as a deliberate literature / "
    "observational analysis that stands on its own. Do NOT describe a "
    "'planned experiment', a 'simulation', or their absence as a failure, "
    "and do NOT treat the lack of an experiment as a gap to apologise for "
    "— this study was never meant to run an executable experiment. (Any "
    "figures present were charted from numbers the sources report; cite "
    "them normally.)\n"
)
# Injected into the design/write prompts when a quest runs in ``survey``
# mode — a descriptive literature / history synthesis with no experiment
# AND no dataset. Stronger than the no-sim directives above: there is not
# even a dataset to analyse, so ``method`` is a plan for how the existing
# published sources are organised and synthesised (by era, theme, technique,
# school …), never a measurement.
_SURVEY_DESIGN_DIRECTIVE = (
    "> **THIS IS A SURVEY / HISTORICAL SYNTHESIS.** The quest is configured "
    "`survey_mode`: there is NO experiment AND NO dataset — the paper is a "
    "descriptive synthesis of the published literature (a history, overview, "
    "or evolution of the topic). **Ignore the \"executable Python "
    "experiment\" / \"must produce a figure\" instructions below.** Do NOT "
    "invent a measurable hypothesis, a clustering/ML method, variables to "
    "measure, or a dataset to collect. Instead produce a SYNTHESIS OUTLINE: "
    "use `hypothesis` for the framing thesis or through-line, `method` for "
    "how the sources are organised and synthesised (e.g. chronological eras, "
    "techniques, movements, sub-questions to cover), and leave `variables` "
    "empty or as thematic axes. Set `\"dependencies\": []` and "
    "`\"figures_planned\": []` — any figures are license-clean illustrative "
    "images gathered separately, not experiment output. Frame the study as "
    "a complete descriptive synthesis in its own right.\n"
)
_SURVEY_WRITE_NOTE = (
    "**This is a survey / historical synthesis — by design there is no "
    "experiment and no dataset.** Write a descriptive, well-structured "
    "narrative (organised by era / theme / technique as fits the topic) "
    "grounded ONLY in the cited literature. Do NOT report results, metrics, "
    "accuracies, clustering, or any computed numbers; do NOT describe a "
    "'planned experiment', a 'dataset', or their absence as a gap or "
    "failure — this study was never meant to run one. Any figures present "
    "are license-clean illustrative images; FI places them with their "
    "captions and credit, and the text refers to each by its number.\n"
)


#: Graph steps that may hand work to background or cluster jobs (``execution.background_jobs``): a passing problem in
#: one of them is never retried by itself (core/crash_kind.py), so the same jobs are never submitted twice.
_OUTSIDE_WORK_STEPS = frozenset({"execute", "execute_reflect", "improve"})


def _lower_first(text: str) -> str:
    """``text`` with its first letter lowered for use mid-sentence, unless the first word is a name or an acronym
    (``FI worked out ...`` stays ``FI``)."""
    return text if text[1:2].isupper() else text[:1].lower() + text[1:]


from core.config import ProviderName as _ProviderName  # noqa: E402

#: Every provider name FI itself uses; an answer whose provider is not one of them came through the VS Code bridge,
#: which reports the vendor that answered (copilot, customendpoint, ollama-models).
_FI_PROVIDER_NAMES = frozenset(get_args(_ProviderName)) - {"vscode_extension"}

#: Why FI asks the plan to work an equation's example out again (core/engine.py ``_rederive_examples``), by source.
_REDERIVE_BECAUSE = {
    "tests": "could not be met by code that follows its formula",
}


# ---- module-level helpers ------------------------------------------------


def write_cost_summary(fi_dir: Path) -> None:
    """Roll ``<fi_dir>/cost.jsonl`` up into ``<fi_dir>/cost.summary.json``
    (schema in :meth:`Engine._write_cost_summary`). Quest finalization writes
    it, and the output pass writes it again: the slides, poster, talk script
    and visual checks come after the quest and log their calls too. A missing
    log writes nothing; no exception bubbles out."""
    path = fi_dir / "cost.jsonl"
    try:
        if not path.is_file():
            return
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        summary = _aggregate_cost_rows(rows)
        try:  # the analysis gate's time in this quest (only a quest that ran it has the file)
            gate = json.loads((fi_dir / "pipeline_gate.json").read_text(encoding="utf-8"))
            summary["gate_runs"], summary["gate_seconds"] = int(gate.get("runs") or 0), float(gate.get("seconds_total") or 0.0)
        except (OSError, ValueError, TypeError):
            pass
        (fi_dir / "cost.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        logging.getLogger("frontier_insight.engine").debug("[cost] failed to write cost.summary.json: %r", e)


def _aggregate_cost_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up a list of cost.jsonl rows into a one-shot summary dict.

    Skips ensemble breadcrumb rows (``ensemble: True`` from
    ``cost_jsonl_entries``) — those mirror per-call rows already
    accounted for via ``_log_chat_cost``, so counting them again
    would double-count requests and tokens. Pure function so it's
    also callable from the web UI's cost endpoint without spinning
    up an Engine.
    """
    totals = {
        "total_requests": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "total_cost_usd_partial": False,
        "estimated_rows": 0,
        "unpriced_requests": 0,
    }
    has_cost_value = False
    by_node: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}

    def _bucket() -> dict[str, Any]:
        return {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "estimated_rows": 0,
            "unpriced_requests": 0,
        }

    for row in rows:
        if row.get("ensemble") is True:
            continue
        usage = row.get("usage") or {}
        prompt_t = int(usage.get("prompt_tokens", 0) or 0)
        completion_t = int(usage.get("completion_tokens", 0) or 0)
        total_t = int(usage.get("total_tokens", 0) or 0) or (prompt_t + completion_t)
        cost = row.get("cost_usd")
        is_priced = isinstance(cost, (int, float))
        is_estimated = bool(usage.get("estimated"))

        totals["total_requests"] += 1
        totals["total_prompt_tokens"] += prompt_t
        totals["total_completion_tokens"] += completion_t
        totals["total_tokens"] += total_t
        if is_priced:
            totals["total_cost_usd"] += float(cost)
            has_cost_value = True
        else:
            totals["unpriced_requests"] += 1
        if is_estimated:
            totals["estimated_rows"] += 1

        node = str(row.get("node") or "unknown")
        model = str(row.get("model") or "unknown")
        for d, key in ((by_node, node), (by_model, model)):
            bucket = d.setdefault(key, _bucket())
            bucket["requests"] += 1
            bucket["prompt_tokens"] += prompt_t
            bucket["completion_tokens"] += completion_t
            bucket["total_tokens"] += total_t
            if is_priced:
                bucket["cost_usd"] += float(cost)
            else:
                bucket["unpriced_requests"] += 1
            if is_estimated:
                bucket["estimated_rows"] += 1

    if not has_cost_value:
        # No real pricing data hit any row — surface that explicitly
        # instead of pretending the total is $0.00.
        totals["total_cost_usd"] = None  # type: ignore[assignment]
        for bucket in list(by_node.values()) + list(by_model.values()):
            bucket["cost_usd"] = None  # type: ignore[assignment]
    elif totals["unpriced_requests"]:
        # Some rows were priced and some were not (a model the price table
        # does not list). The total above counts the priced rows only, so it
        # is a lower bound; say so rather than let it pass for the whole cost.
        totals["total_cost_usd_partial"] = True

    return {
        **totals,
        "by_node": by_node,
        "by_model": by_model,
        "generated_at": time.time(),
    }


def _design_parts_listing(design: Any) -> str:
    """The design's own part paths, listed from the design object, for a reviewer to cite (``design_parts``)."""
    if not isinstance(design, dict) or not design:
        return ""
    return ("\n\nThe parts of the design, as paths (cite these in `design_parts`): "
            + ", ".join(f"`{p}`" for p in _design_parts.part_paths(design)))


def _load_prompts() -> dict[str, string.Template]:
    names = (
        "clarify", "ideate", "ideate_reflect", "ideate_tournament", "literature_novelty",
        "design", "design_self_critique",   # second-pass methodology audit
        "design_parts_map",                 # a note with no path: which listed parts of the design it is about
        "design_parts",                     # a redesign: ONLY the parts the reason concerns, the rest carried over
        "plan_revise",                      # --revise-plan: rewrite plan.md as the person asked
        "plan_parts",                       # FI asks for ONLY the missing parts of the plan, as JSON (never the whole file)
        "plan_criteria",                    # the plan named no check of correctness: ask once more
        "plan_roles", "plan_names", "plan_comparisons",  # the contract's parts, asked one at a time (core/contract_parts.py)
        "plan_formulas",                    # ... and a formula FI cannot work out as written, when FI writes the analysis
        "plan_figures",                     # ... and a figure the plan names only by file name, when FI writes the analysis
        "plan_answers",                     # ... and which result answers each quantity the topic asks for
        "plan_dependents",                  # ... and what a dependent variable that is no declared result is (adds checks only)
        "json_reanswer",                    # a required check's reply could not be read: one short JSON re-answer
        "json_reanswer_part",               # ... when part of it could be read: only the unreadable keys are asked again
        "oracle_review",                    # a second model reads the plan's checks against known answers
        "topic_requirements",               # a second model lists what the topic asks for, from its text alone
        "topic_cover",                      # ... and judges the items a name match could not find in the plan
        "code_review",                      # a second model reads the code against the plan's required outputs
        "oracle_recompute",                 # a failing check's expected value worked out again, blind to the measurement
        "oracle_challenge_plan",            # the plan's model answers the code's objection to its known answer
        "oracle_challenge_judge",           # a third model judges between the two workings
        "oracle_numeric_reference",         # ... and, still unsettled, a script that computes the answer numerically
        "implement",                      # legacy one-shot (resume fallback)
        "implement_outline",                # two-stage implement: scaffold
        "scope",                # the few studies the topic could mean, for a person to pick before the plan
        "select_skills",        # pick which skills this quest carries
        "implement_body",                   # two-stage implement: fills bodies
        "implement_update",                 # a revised plan: ONLY the functions of the scripts the revision touched
        "implement_agent",                  # engine.code_writer: agent -- the coding agent's task (core/agent_build.py)
        "implement_function",               # the model's package: ONE function body for ONE equation
        "implement_function_repair",        # ONE function repaired alone (never the whole script)
        "implement_module_repair",          # the failing functions of the model's package, with the passing ones named
        "code_review_repair",               # the code has not crashed: a second model's finding, to be fixed
        "code_finding_test",                # a small test that fails on the current code because of one finding
        "execute_reflect", "analyze",
        "improve",              # one change to the simulation against its checks of correctness
        "improve_speed",        # one change that makes a long study's simulation faster, same numbers
        "cross_check",
        "cross_check_verify",   # CoVe-style second-pass verification
        "claim_check",          # ground each paper claim to evidence
        "citation_check",       # a different model reads every citation against its source's retrieved text
        "citation_repair",      # what that reading found unsupported, fixed without a person
        "evidence_gate",        # weigh evidence sufficiency before write
        "literature_screen",    # grade retrieved sources 0-3 before they reach the corpus
        "figures_pick",         # which of the papers' figures carry values this question can use
        "figures_read",         # read those values off the images
        "write", "review",
        "write_patch",      # a revise for flagged passages: edits, not a new paper
        "write_trim",       # a draft a little over its page limit: sentences to take out
        "write_abstract",   # a scientific paper whose draft has no Abstract: the Abstract alone, from the draft
        "replot_layout",    # a refine that only asks for the figures arranged again: redraw from saved data
        "review_moderate",  # review-panel moderator prompt
        "data_load",        # no-simulation mode — synthesize result_json
                            # from user-supplied data
        "web_plots",        # no-simulation mode — chart the collected
                            # web/data content (text-only outputs otherwise)
        "second_impl_write",    # a second, independent version of a key function, from its card alone
        "second_impl_adjudicate",  # which of two versions that disagree is wrong
        "execute_reflect_failure",  # the failed function's own failure, whole, in a repair prompt
    )
    out: dict[str, string.Template] = {}
    for n in names:
        path = PROMPTS_DIR / f"{n}.md"
        out[n] = string.Template(path.read_text(encoding="utf-8"))
    return out


# Per-paper excerpt size when rendering retrieved literature into the
# write / ideate / design prompts. Was 600 chars — too short for the
# model to extract specific findings or methods, so citations stayed
# generic. 2000 chars is roughly an abstract + intro, which is enough
# to discuss prior work by content rather than by title alone.
_LIT_EXCERPT_CHARS = 2000

# A record the retrieval layer kept no full text of holds its title and whatever
# followed it: an abstract, or nothing. Over the 2,321 sources of the 131 stored
# quests, the text after the title is empty for 508 of them and 22-85 characters
# for five more; the next-shortest holds 100, so 100 is a clean cut for a
# title-only record. A book is the other kind: it has no abstract, so the text
# kept for one is a blurb or the opening of a review (62 records of 760-1,376
# characters; the other 59 books are title-only). Length alone would not
# separate a blurb from an abstract: 137 articles have an abstract of 700-1,499
# characters. Together 575 records (25%) are flagged, and none of them has full
# text or an abstract.
_THIN_TEXT_CHARS = 100
_THIN_MARKS = {"title": "title only", "blurb": "short blurb only"}


#: The kinds a finished quest writes to Axon (accepted and preliminary standing).
_QUEST_MEMORY_KINDS = frozenset({"fi_quest_summary", "fi_quest_paper", "fi_paper_spine", "fi_preliminary_summary",
                                 "fi_preliminary_paper", "fi_preliminary_spine"})


def _unchecked_quest_memory(meta: dict[str, Any]) -> bool:
    """Whether ``meta`` is an entry a quest wrote before its findings were taken from FI's record: written by a quest
    (it names one), of a kind a quest writes, and without ``claims_from == "record"``. Never a store write: read only."""
    return (meta.get("kind") in _QUEST_MEMORY_KINDS and bool(meta.get("quest_id"))
            and meta.get("claims_from") != "record")


def _has_full_text(meta: dict[str, Any]) -> bool:
    """Whether a source's text is its full text: its ``content_quality`` when it has one (a fetch that returned only
    the abstract or a table of contents is not), else whether a fetch returned anything (an entry from before
    content_quality told the two apart)."""
    quality = meta.get("content_quality")
    return quality == "full_text" if quality else bool(meta.get("fetched_full_text"))


def _thin_source(meta: dict[str, Any], content: str) -> str | None:
    """``"title"`` when a source's stored text is only its title, ``"blurb"``
    when it is a book without full text (all it has is a short description), and
    ``None`` for every other source, full text and abstracts included. Such a
    record cannot show a finding, a number or a mechanism, whatever it is cited
    for."""
    if _has_full_text(meta):
        return None
    title = str(meta.get("title") or meta.get("source") or "")
    # The source's own text: a model's reading of its figures is not text the source has.
    if len(_format_lit_excerpt(_split_figure_readings(content or "")[0], title).strip()) < _THIN_TEXT_CHARS:
        return "title"
    return "blurb" if str(meta.get("work_type") or "") == "book" else None


def _format_lit_header(meta: dict[str, Any], i: int, thin: str | None = None) -> str:
    """Render the header line of a prior-work block entry.

    Includes title + authors + year + venue + DOI/URL when the
    retrieval layer surfaced them. Previously this only emitted the
    title, which left the writer LLM with no choice but to produce
    bare-title References like ``"1. Stratonovich-type integral ..."``
    with no author, year, or DOI — useless for actual citation
    lookup.

    Format example:
        [3] Lipton-Lifschitz, 2003. Closed-form approximations ...
            Quantitative Finance. DOI: 10.1088/1469-7688/3/1/305.

    ``thin`` (see :func:`_thin_source`) adds ``[title only]`` or ``[short
    blurb only]`` after the title, on the first line.
    """
    title = meta.get("title") or meta.get("source") or f"item-{i}"
    authors = meta.get("authors") or []
    year = meta.get("year") or (meta.get("published") or "")[:4]
    venue = meta.get("venue") or meta.get("publisher") or ""
    doi = meta.get("doi") or ""
    arxiv_id = meta.get("arxiv_id") or ""
    url = meta.get("url") or ""

    # Author block: prefer "First, Second & Third" for 2-3 authors,
    # collapse to "First et al." beyond 3. Keeps the prior-work block
    # readable for the LLM without truncating the citation handle.
    if isinstance(authors, list) and authors:
        clean = [a for a in authors if a]
        if len(clean) == 1:
            author_str = clean[0]
        elif len(clean) == 2:
            author_str = f"{clean[0]} & {clean[1]}"
        elif len(clean) == 3:
            author_str = f"{clean[0]}, {clean[1]} & {clean[2]}"
        elif len(clean) > 3:
            author_str = f"{clean[0]} et al."
        else:
            author_str = ""
    else:
        author_str = ""

    parts: list[str] = [f"[{i}]"]
    head = ""
    if author_str:
        head = author_str
        if year:
            head += f" ({year})"
        head += f". {title}"
    elif year:
        head = f"({year}) {title}"
    else:
        head = title
    parts.append(head)
    line1 = " ".join(parts)
    if thin in _THIN_MARKS:
        # The record holds no more than this says, and the writer is told so
        # where it reads the entry (see agents/write.md, "Citing sources").
        line1 += f" [{_THIN_MARKS[thin]}]"
    if _unchecked_quest_memory(meta):
        # An entry an earlier quest wrote from the model's own text: a precedent only as far as that text is.
        line1 += " [findings taken from the model's text, not checked against FI's record]"
    if _retractions.is_retracted(meta):
        # Crossref lists a retraction of it (core/retractions.py): every block that names it says so.
        line1 += " [retracted]"

    extras: list[str] = []
    if venue:
        extras.append(venue)
    if doi:
        extras.append(f"DOI: {doi}")
    elif arxiv_id:
        extras.append(f"arXiv:{arxiv_id}")
    elif url:
        extras.append(url)
    if extras:
        return f"{line1}\n    {'. '.join(extras)}."
    return line1


def _format_lit_excerpt(
    content: str,
    title: str,
    *,
    query: str = "",
    budget: int | None = None,
    mode: str = "lexical",
) -> str:
    """Excerpt a source's content for a prompt. When ``query`` is given and
    the content exceeds ``budget``, the excerpt is the passages most
    RELEVANT to the question (relevance-ranked, see
    :mod:`core.passages`) rather than the first ``budget`` chars — so a
    result, table, or number buried mid-document still reaches the LLM.
    Without a query it degrades to the leading slice (back-compatible).

    Also trims the leading-title duplication out of arXiv-style excerpts:
    most loaders set ``content = f"{title}\\n\\n{abstract}"`` so the LLM
    used to see ``[i] Title\\nTitle. abstract...`` and propagated the
    title-twice pattern into the References section."""
    budget = _LIT_EXCERPT_CHARS if budget is None else budget
    # What a model read off the source's figures is excerpted apart from the source's own text and always shown under
    # its label: a passage picked for relevance must never carry a reading without the words saying what it is.
    content, readings = _split_figure_readings(content)
    # A quarter of the budget, or all the source's own text leaves unused.
    readings_budget = min(len(readings), max(budget // 4, budget - len(content))) if readings.strip() else 0

    def pick(text: str, room: int) -> str:
        if query and len(text) > room:
            from .passages import select_relevant_excerpt
            return select_relevant_excerpt(text, query, budget_chars=room, mode=mode)
        return text[:room]

    excerpt = pick(content, budget - readings_budget)
    if title and excerpt.lstrip().startswith(title):
        # Drop the leading title + immediately-following separator
        # (newline or ". "). Keep everything after as the real abstract.
        excerpt = excerpt.lstrip()[len(title):].lstrip(".\n ")
    if readings_budget:
        excerpt = excerpt.rstrip() + _PARA + _FIGURE_READINGS_MARK + _PARA + pick(readings, readings_budget).strip()
    return excerpt


#: What a preliminary FI result is written to Axon as (core/knowledge.py::add_quest_artifacts).
PRELIMINARY_KINDS = frozenset({"fi_preliminary_spine", "fi_preliminary_paper", "fi_preliminary_summary"})


def _is_preliminary_memory(meta: dict[str, Any]) -> bool:
    """An earlier FI quest's result kept as a preliminary observation (an exploration, a result with evidence gaps):
    a reminder of what was tried, never a source to cite."""
    return str(meta.get("kind") or "") in PRELIMINARY_KINDS or str(meta.get("standing") or "") == "preliminary"


def _preliminary_reminders(items: list[Any], limit: int = 3, chars: int = 300) -> str:
    """The earlier preliminary FI results among ``items``, as a short block set apart from the citable sources, or
    ``""`` when there are none."""
    lines: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        meta = (item.get("metadata") if isinstance(item, dict) else getattr(item, "metadata", None)) or {}
        if not _is_preliminary_memory(meta):
            continue
        key = str(meta.get("quest_id") or meta.get("title") or "")
        if key in seen:
            continue
        seen.add(key)
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
        text = " ".join(str(content or "").split())[:chars]
        lines.append(f"- {meta.get('title') or 'an earlier quest'} (quest {meta.get('quest_id') or '?'}, "
                     f"result for {meta.get('result_use') or '?'}, evidence {meta.get('evidence_status') or '?'}): {text}")
        if len(lines) >= limit:
            break
    if not lines:
        return ""
    return ("Earlier FI results kept as PRELIMINARY (an exploration, or a result with evidence gaps). They say what was "
            "tried; they are not evidence. Never cite them or state their findings as established:\n"
            + _source_text.fence("\n".join(lines), "earlier FI results read back from the knowledge base"))


def _with_reminders(block: str, items: list[Any]) -> str:
    reminders = _preliminary_reminders(items)
    return f"{block}\n\n{reminders}" if reminders else block


def _is_citable(meta: dict[str, Any]) -> bool:
    """An entry is citable if it has at least a real title AND
    one identifying field (authors, year, venue, doi/url/arxiv).
    An earlier FI result kept as preliminary is never citable (see :func:`_preliminary_reminders`).

    Filters out partial loader output where only a path/slug exists
    — those entries would render as ``[i] item-i`` or ``[i] (no title)``
    and the LLM tends to fabricate author names / URLs to complete
    the slot. Honesty > completeness on a thin retrieval pull.
    """
    if _is_preliminary_memory(meta):
        return False
    title = (meta.get("title") or "").strip()
    if not title:
        return False
    # A bot-check page's title ("Checking your browser - reCAPTCHA") names the
    # wall a crawler met, not a source, whatever brought the entry in (web
    # search, a corpus hit, a checkpoint saved before web search dropped it).
    if _is_bot_check_title(title):
        return False
    has_id = bool(
        meta.get("authors")
        or meta.get("year")
        or meta.get("published")
        or meta.get("venue")
        or meta.get("publisher")
        or meta.get("doi")
        or meta.get("arxiv_id")
        or meta.get("url")
    )
    return has_id


# FI-internal kinds that are cross-quest memory artifacts — NOT public
# sources. When the paper's audience is "external" (a journal, the open
# web), the writer must not cite these because the reader cannot look
# them up. ``fi_local_paper`` is intentionally OMITTED: that kind is
# how the user feeds real (paywalled or local) papers into Axon; whether
# such an entry survives depends on its own metadata (real DOI/URL).
_FI_INTERNAL_KINDS = frozenset({
    "fi_preliminary_spine",
    "fi_preliminary_paper",
    "fi_preliminary_summary",
    "fi_critique",
    "fi_digest",
    "fi_portfolio",
    "fi_proposal",
    "fi_summary",
    "fi_summary_input",
    "fi_source_catalog",
    "fi_paper_spine",
})


#: FI's own cross-quest memory: never recorded as an external paper a quest built on.
_FI_MEMORY_KINDS = _FI_INTERNAL_KINDS | {"fi_quest_paper", "fi_quest_summary", "fi_topic_event"}


def _is_audience_appropriate(meta: dict[str, Any], audience: str) -> bool:
    """When the paper is external-facing, drop cross-quest memory
    artifacts so the References section only contains sources an
    outside reader could actually look up. Internal-facing papers
    keep everything (the audience expects to see prior internal work).
    """
    if audience == "internal":
        return True
    kind = (meta.get("kind") or "").strip()
    return kind not in _FI_INTERNAL_KINDS


def _lit_query(state: QuestState) -> str:
    """The relevance query for selecting literature passages into a prompt:
    the quest topic plus, when available, the chosen idea title and the
    design hypothesis, so passages are picked for THIS question."""
    parts = [state.get("topic") or ""]
    chosen = state.get("chosen_idea") or {}
    if isinstance(chosen, dict) and chosen.get("title"):
        parts.append(str(chosen["title"]))
    design = state.get("design") or {}
    if isinstance(design, dict) and design.get("hypothesis"):
        parts.append(str(design["hypothesis"]))
    return " ".join(p for p in parts if p)[:500]


def _screen_grades(parsed: Any, n: int) -> dict[int, int] | None:
    """Read the literature screen's reply -- ``{"grades": [{"i": 0, "grade":
    3}, ...]}`` or ``{"grades": {"0": 3}}`` -- into ``{index: grade}``. None
    when the reply carries no grades at all. Entries naming an index outside
    the ``n`` candidates, or a grade outside 0-3, are ignored."""
    if not isinstance(parsed, dict):
        return None
    raw = parsed.get("grades")
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, list):
        pairs = [(e.get("i"), e.get("grade")) for e in raw if isinstance(e, dict)]
    else:
        return None
    out: dict[int, int] = {}
    for i, g in pairs:
        try:
            i, g = int(i), int(g)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and 0 <= g <= 3:
            out[i] = g
    return out


def _merge_round_robin(result_lists: list[list]) -> list:
    """Interleave several searches' results -- the first of each, then the
    second of each, and so on -- keeping the first copy of a work found by
    more than one. Each search keeps its own ranking, and no search's best
    results end up behind another search's tail."""
    seen: set[str] = set()
    out: list = []
    for rank in range(max((len(r) for r in result_lists), default=0)):
        for results in result_lists:
            if rank >= len(results):
                continue
            doc = results[rank]
            keys = _doc_dedup_keys(doc)
            if any(k in seen for k in keys):
                continue
            seen.update(keys)
            out.append(doc)
    return out


def _format_lit(
    docs: list[RetrievedDoc],
    audience: str = "external",
    *,
    query: str = "",
    budget: int | None = None,
    mode: str = "lexical",
) -> str:
    if not docs:
        return "(no prior work surfaced from the knowledge base)"
    lines: list[str] = []
    keep_idx = 0
    for d in docs:
        meta = d.metadata or {}
        if not _is_citable(meta):
            continue
        if not _is_audience_appropriate(meta, audience):
            continue
        keep_idx += 1
        title = meta.get("title") or meta.get("source") or f"item-{keep_idx}"
        header = _source_text.mark(_format_lit_header(meta, keep_idx), meta)
        excerpt = _format_lit_excerpt(
            d.content, title, query=query, budget=budget, mode=mode,
        )
        lines.append(f"{header}\n{excerpt}" if excerpt else header)
    if not lines:
        return _with_reminders("(no prior work surfaced from the knowledge base)", docs)
    # Retrieved text is material to read, never instructions (core/source_text.py); FI's reminders stay outside.
    return _with_reminders(_source_text.fence("\n\n".join(lines)), docs)


def _format_lit_from_state(
    state: QuestState,
    audience: str = "external",
    *,
    query: str = "",
    budget: int | None = None,
    mode: str = "lexical",
    mark_thin: bool = False,
) -> str:
    """The prior-work block from ``state['literature']``. Entries carry the
    labels the paper's References ([1], [2]…) and Further reading ([W1],
    [W2]…) use, taken from the same de-duplicated list
    (:func:`_labelled_sources`), so the writer's [2] is the claim check's and
    the bib's [2]. With ``mark_thin`` (the writer's block) an entry whose stored
    text is only a title or a short blurb says so (:func:`_thin_source`)."""
    items = state.get("literature") or []
    if not items:
        return "(no prior work surfaced from the knowledge base)"
    lines: list[str] = []
    for label, meta, item in _labelled_sources(items, audience):
        title = meta.get("title") or meta.get("source") or f"item-{label}"
        content = _item_content(item)  # the whole text, from disk when the state holds only its first part
        header = _format_lit_header(meta, label, _thin_source(meta, content) if mark_thin else None)
        header = _source_text.mark(header, meta)
        excerpt = _format_lit_excerpt(
            content, title,
            query=query, budget=budget, mode=mode,
        )
        lines.append(f"{header}\n{excerpt}" if excerpt else header)
    if not lines:
        return _with_reminders("(no prior work surfaced from the knowledge base)", items)
    # Retrieved text is material to read, never instructions (core/source_text.py); FI's reminders stay outside.
    return _with_reminders(_source_text.fence("\n\n".join(lines)), items)


def _is_web_page(meta: dict[str, Any]) -> bool:
    """A source found by general web search. It is Further reading, not a
    numbered Reference, even when the page names a DOI: what the writer read
    is the page."""
    return str(meta.get("source") or "") == "web_search"


def _labelled_sources(
    literature: list[Any], audience: str = "external",
) -> list[tuple[str, dict[str, Any], Any]]:
    """The quest's citable sources, de-duplicated once and labelled in
    literature order: scholarly records ``"1"``, ``"2"``… and web pages
    ``"W1"``, ``"W2"``…. The writer's prior-work block, the References,
    Further reading, claim grounding, the poster, the slides and the bib all
    label from this one list, so a label names the same source everywhere.
    Applies :func:`_is_citable` and :func:`_is_audience_appropriate`, so
    FI-internal cross-quest memory never appears. Returns ``(label,
    metadata, item)`` triples."""
    out: list[tuple[str, dict[str, Any], Any]] = []
    seen: set[str] = set()
    papers = pages = 0
    for item in literature or []:
        meta = (item.get("metadata") if isinstance(item, dict)
                else getattr(item, "metadata", None)) or {}
        if not _is_citable(meta) or not _is_audience_appropriate(meta, audience):
            continue
        key = str(
            meta.get("doi") or meta.get("arxiv_id") or meta.get("pmid")
            or meta.get("url") or meta.get("title") or ""
        ).lower().strip()
        if not key:
            continue
        norm_title = _normalize_title(meta.get("title") or "")
        keys = [key] + ([f"title:{norm_title}"] if norm_title else [])
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        if _is_web_page(meta):
            pages += 1
            out.append((f"W{pages}", meta, item))
        else:
            papers += 1
            out.append((str(papers), meta, item))
    return out


def _foundational_outcomes(
    suggestions: list[dict[str, Any]], new: list[RetrievedDoc],
    found: list[RetrievedDoc], docs: list[RetrievedDoc],
) -> tuple[list[str], list[str], list[str]]:
    """What became of each suggested work, as :func:`_foundational_work_text`
    lines: the ones the pass added (``new``), the ones already among the search
    results (the lookup found them, or the search had them, so they are not new)
    and the ones OpenAlex did not have. A suggestion is matched to a record by
    the same title rule the lookup used, so a record counts for the suggestion
    that found it; a record found by author and year, whose title is not the
    suggested one, names the suggestions it answers (``suggested_works``: title,
    year and authors, so two suggestions with one title are told apart), and
    its line says so: ``suggested (year, author) -> the record's title, by author
    and year``."""
    def record_for(work: dict[str, Any], candidates: list[RetrievedDoc]) -> RetrievedDoc | None:
        title = str(work.get("title") or "")
        for d in candidates:
            md = d.metadata or {}
            if _titles_match(title, str(md.get("title") or "")):
                return d
            if title and _foundational_work_text(work) in (md.get("suggested_works") or []):
                return d
        return None

    def line(work: dict[str, Any], doc: RetrievedDoc) -> str:
        text = _foundational_work_text(work)
        found_title = " ".join(str((doc.metadata or {}).get("title") or "").split())
        if _titles_match(str(work.get("title") or ""), found_title):
            return text
        return f"{text} -> {found_title}, by author and year"

    added: list[str] = []
    already: list[str] = []
    dropped: list[str] = []
    for work in suggestions:
        if (doc := record_for(work, new)) is not None:
            added.append(line(work, doc))
        elif (doc := record_for(work, found) or record_for(work, docs)) is not None:
            already.append(line(work, doc))
        else:
            dropped.append(_foundational_work_text(work))
    return added, already, dropped


def _foundational_sources(
    literature: list[Any], audience: str = "external",
) -> list[tuple[str, dict[str, Any]]]:
    """The foundational works in the paper's citable set, as ``(label,
    metadata)`` in the writer's label order. The citable set is the one
    :func:`_labelled_sources` gives the prior-work block, so a label here is that
    block's label. A work is foundational when the literature pass marked it
    (``metadata["foundational"]``): an original paper or standard textbook the
    model named and OpenAlex holds, or a work several retrieved papers cite. It
    is in this set only if the literature screen kept it."""
    return [
        (label, meta)
        for label, meta, _item in _labelled_sources(literature, audience)
        # A retracted work is not to be cited, so it is never asked for (core/retractions.py).
        if meta.get("foundational") and not label.startswith("W") and not _retractions.is_retracted(meta)
    ]


def _foundational_line(
    label: str, meta: dict[str, Any], *, numbered: bool, thin: str | None = None,
) -> str:
    """One foundational work as the prior-work block's header line gives it
    (authors, year, title), with ``[label]`` when ``numbered``, and a note of
    what kind of work it is: a book, or how many retrieved papers cite it. A
    ``thin`` record (:func:`_thin_source`) carries the same ``[title only]`` /
    ``[short blurb only]`` as its entry in the block."""
    line = _format_lit_header(meta, label, thin).split("\n", 1)[0]
    if not numbered:
        line = re.sub(r"^\[[^\]]*\]\s*", "", line)
    notes: list[str] = []
    if meta.get("work_type") in ("book", "book-chapter"):
        notes.append("book")
    how = str(meta.get("foundational") or "")
    if how.startswith("cited by"):
        notes.append(how)
    return f"- {line}" + (f" ({', '.join(notes)})" if notes else "")


def _foundational_write_block(literature: list[Any], audience: str = "external") -> str:
    """The write prompt's note on the foundational works in the prior-work block,
    or "" when there are none, so a prompt without any is unchanged. It lists each
    with the label the writer cites it by, and asks for the ones that bear on the
    paper. The works were retrieved, kept by the literature screen and marked
    foundational, and on stored quests the papers left 175 of 312 such works
    uncited. The ask is only for the ones that bear on the paper, so it never
    pushes the writer to cite a work the paper does not use."""
    works = _foundational_sources(literature, audience)
    if not works:
        return ""
    thin = {
        label: _thin_source(meta, _item_content(item))
        for label, meta, item in _labelled_sources(literature, audience)
    }
    return (
        "\n\n### Foundational works in the prior-work block\n"
        "The literature search marked these entries as foundational: the original papers "
        "and standard textbooks the topic rests on, or works several of the retrieved papers "
        "cite. Cite each one that bears on this paper's claims, where it belongs: the original "
        "paper for a method, model or relation the paper uses, the standard textbook for the "
        "field. A work that does not bear on the paper is not to be cited just to be cited.\n\n"
        + "\n".join(
            _foundational_line(label, meta, numbered=True, thin=thin.get(label)) for label, meta in works
        )
    )


def _uncited_foundational_works(
    literature: list[Any], paper_md: str, audience: str = "external",
) -> list[tuple[str, dict[str, Any]]]:
    """The foundational works in the citable set that the paper's text does not
    cite. The paper's citation numbers are the labels of ``literature`` in the
    order the write node leaves it in (:func:`_finalize_paper_sources`), the same
    numbers :func:`cited_references` reads."""
    cited = {str(n) for n in _citation_counts(paper_md)}
    return [(label, meta) for label, meta in _foundational_sources(literature, audience) if label not in cited]


def _foundational_review_block(literature: list[Any], paper_md: str, audience: str = "external") -> str:
    """The review prompt's advisory line on foundational works the paper does not
    cite, or "" when it cites every one (or there are none). Advisory: it is
    handed to the reviewer as something it may ask for, and nothing here adds a
    must-flag hit, because whether a work bears on a paper is a judgement, not a
    set difference."""
    if not (paper_md or "").strip():
        return ""  # nothing was read, so nothing can be said to be left out
    missing = _uncited_foundational_works(literature, paper_md, audience)
    if not missing:
        return ""
    # The prior-work block marks a record FI holds no more than the title (or a
    # blurb) of, and the writer is told to cite it only for what that text says.
    # The reviewer was not told, and asked for one to back a specific claim: a
    # rewrite then cited it, and the claim check marked the sentence unsupported.
    thin = {
        label: _thin_source(meta, _item_content(item))
        for label, meta, item in _labelled_sources(literature, audience)
    }
    marked = any(thin.get(label) for label, _meta in missing)
    return (
        "\n\n## Foundational works this paper does not cite (advisory)\n"
        "These works were retrieved, kept by the literature screen and marked as foundational "
        "(an original paper or standard textbook the topic rests on, or a work several of the "
        "retrieved papers cite), but the paper does not cite them:\n"
        + "\n".join(
            _foundational_line(label, meta, numbered=False, thin=thin.get(label)) for label, meta in missing
        )
        + "\nAdvisory only: if one of them bears on a claim the paper makes (the original paper "
        "for a method or relation the paper uses, the standard textbook for the field), you may "
        "ask for it under `suggestions`. Do not add anything to `must_flag_hits` and do not "
        "choose `revise` because of this alone. A work that does not bear on the paper needs "
        "no citation."
        + (
            " An entry marked `[title only]` or `[short blurb only]` has no text beyond that: ask "
            "for it only to say the work exists or for what its title states, never to back a "
            "finding, a number or a mechanism, which it cannot show."
            if marked else ""
        )
    )


def _reference_entry(label: str, meta: dict[str, Any]) -> dict[str, Any]:
    authors = meta.get("authors") or []
    if not isinstance(authors, list):
        authors = [str(authors)]
    entry = {
        "title": (meta.get("title") or "").strip(),
        "authors": [a for a in authors if a],
        "year": meta.get("year") or (meta.get("published") or "")[:4] or "",
        "venue": meta.get("venue") or meta.get("publisher") or "",
        "doi": meta.get("doi") or "",
        "arxiv_id": meta.get("arxiv_id") or "",
        "url": meta.get("url") or "",
        "site": meta.get("site") or "",
        "source": meta.get("source") or "",
    }
    if label.startswith("W"):
        return {"label": label, **entry}
    return {"n": int(label), **entry}


def build_references(
    literature: list[Any],
    *,
    audience: str = "external",
    max_n: int | None = None,
) -> list[dict[str, Any]]:
    """The numbered References: the quest's citable scholarly sources from
    ``state['literature']`` (dict items ``{content, metadata}`` or
    ``RetrievedDoc`` objects), de-duplicated and numbered as in
    :func:`_labelled_sources`. Web pages are not here; they are
    :func:`build_further_reading`. Shared by the claim check, the poster, the
    slides and the bib export so every output cites the same numbers. Each
    entry: ``{n, title, authors, year, venue, doi, arxiv_id, url, site,
    source}``."""
    refs = [
        _reference_entry(label, meta)
        for label, meta, _ in _labelled_sources(literature, audience)
        if not label.startswith("W")
    ]
    return refs[:max_n] if max_n else refs


def build_further_reading(
    literature: list[Any],
    *,
    audience: str = "external",
    max_n: int | None = None,
) -> list[dict[str, Any]]:
    """Further reading: the web pages the quest drew on, labelled ``W1``,
    ``W2``… as in :func:`_labelled_sources`. The writer may quote them, but
    they are listed apart from the scholarly References. Each entry carries
    ``label`` instead of ``n``, with the same other fields."""
    further = [
        _reference_entry(label, meta)
        for label, meta, _ in _labelled_sources(literature, audience)
        if label.startswith("W")
    ]
    return further[:max_n] if max_n else further


def _ref_citation_text(r: dict[str, Any]) -> str:
    """One-line human citation: 'Authors (Year). Title. Venue. URL/DOI'.
    Falls back gracefully when a web result has only title + URL."""
    bits: list[str] = []
    authors = r.get("authors") or []
    if authors:
        who = f"{authors[0]} et al." if len(authors) > 3 else ", ".join(authors)
        bits.append(who + (f" ({r['year']})" if r.get("year") else ""))
    elif r.get("year"):
        bits.append(f"({r['year']})")
    if r.get("title"):
        bits.append(r["title"])
    line = ". ".join(b for b in bits if b)
    tail: list[str] = []
    if r.get("venue"):
        tail.append(r["venue"])
    if r.get("doi"):
        tail.append(f"DOI: {r['doi']}")
    elif r.get("arxiv_id"):
        tail.append(f"arXiv:{r['arxiv_id']}")
    elif r.get("url"):
        tail.append(r["url"])
    if tail:
        line = f"{line}. {'. '.join(tail)}" if line else ". ".join(tail)
    return line.strip()


def _latex_esc(s: str) -> str:
    """Escape LaTeX specials in citation text (URLs carry ``_ & % # ~``).
    Backslashes become ``/`` (URLs only) so we never collide with the
    escape sequences we insert; braces are escaped before anything that
    introduces a backslash."""
    s = s.replace("\\", "/")
    for a, b in (
        ("{", r"\{"), ("}", r"\}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
        ("~", r"\textasciitilde "), ("^", r"\textasciicircum "),
    ):
        s = s.replace(a, b)
    return s


# The deck ends on one slide of sources: the ones the paper cites most, as many
# as fit, and a pointer to the paper for the rest. The validation quest's
# nine-slide talk ended on three References and three Further reading slides
# that no slide cited. The list is set at 0.96em (18 pt) in two columns
# (templates/slides/fi.css): a column holds about 35 characters of monospace
# text a line, and the slide about 27 lines across both columns, so change one
# with the other. Each entry costs its wrapped lines plus half a line of
# spacing. Measured with the real Marp theme, the four sources of a stored quest
# (26 lines) end at the slide's bottom padding edge, and a set of 30 lines or
# more ran past it; the deck lists as many as the budget holds.
_SOURCE_SLIDE_CHARS_PER_LINE = 35
_SOURCE_SLIDE_LINES = 27
_SOURCE_SLIDE_MAX = 6
# A citation in a paper's text: "[3]", "[1, 4]", "[2-5]", "[W1]" or "[2, W1]".
_CITATION_PART = r"(?:W\d+|\d+(?:\s*[–-]\s*\d+)?)"
_CITATION_BRACKET_RE = re.compile(rf"\[({_CITATION_PART}(?:\s*[,;]\s*{_CITATION_PART})*)\]")
_REFERENCES_HEADING_RE = re.compile(r"^#{1,6}\s*references\s*$", re.IGNORECASE | re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def _citation_labels(inside: str) -> list[str]:
    """The source labels one citation bracket names, ranges spelled out."""
    labels: list[str] = []
    for part in re.split(r"[,;]", inside):
        part = part.strip()
        if part.startswith("W"):
            labels.append(part)
            continue
        ends = [int(x) for x in re.split(r"[–-]", part) if x.strip()]
        labels += [str(n) for n in range(ends[0], min(ends[-1], ends[0] + 50) + 1)]
    return labels


def _paper_body(paper_md: str) -> str:
    """The paper without its own reference list and what follows it."""
    return _REFERENCES_HEADING_RE.split(paper_md or "", maxsplit=1)[0]


def _citation_counts(paper_md: str) -> dict[int, int]:
    """How often the paper's text cites each numbered reference."""
    counts: dict[int, int] = {}
    for match in _CITATION_BRACKET_RE.finditer(_paper_body(paper_md)):
        for label in _citation_labels(match.group(1)):
            if label.isdigit():
                counts[int(label)] = counts.get(int(label), 0) + 1
    return counts


def _citing_sentences(paper_md: str) -> dict[str, list[str]]:
    """Each source label the paper's text cites, with the sentences that cite it."""
    found: dict[str, list[str]] = {}
    for block in re.split(r"\n\s*\n", _paper_body(paper_md)):
        for sentence in _SENTENCE_END_RE.split(" ".join(block.split())):
            for match in _CITATION_BRACKET_RE.finditer(sentence):
                for label in _citation_labels(match.group(1)):
                    sentences = found.setdefault(label, [])
                    if sentence not in sentences:
                        sentences.append(sentence)
    return found


# A source FI holds only the title of contains no statement, so it cannot show a
# finding, a number or a mechanism whatever it is cited for. The claim check reads
# the sentence and the source and says whether the citation holds, and it misses
# some: on the 72 stored SIR papers 9 sentences cite only title-only records and say
# more than the title, the check had marked 6 of them unsupported and had not listed
# the other 3 as claims at all.
# A sentence is said to go beyond the title when it names this many words that its
# cited sources' titles, authors and stored text do not. A record's words come from
# :func:`_content_stems`, the word test the foundational-work lookup uses, so
# "epidemic" and "epidemics" are one word. Attribution ("Kermack and McKendrick [1]
# introduced the SIR model") adds one or two such words to a title that names its
# subject; a specific result adds more.
_TITLE_ONLY_UNEXPLAINED_STEMS = 4
# A claim and a sentence are the same statement when this share of the shorter
# one's content words are in the other, and at least this many words are: a
# three-word title shares all three with any claim that names the same things.
_SAME_STATEMENT_SHARE = 0.6
_SAME_STATEMENT_WORDS = 4


def _title_only_sentences(
    paper_md: str, sources: dict[str, tuple[dict[str, Any], str]],
) -> list[tuple[str, list[str], list[str]]]:
    """``(sentence, labels, words)`` for each sentence of the paper's text that
    cites sources, cites only ones whose stored text is nothing but the title
    (:func:`_thin_source`), and says at least ``_TITLE_ONLY_UNEXPLAINED_STEMS``
    words that neither those sources' titles, authors nor stored text contain.
    The words are what the sentence says beyond the sources' text. A book's blurb
    is not this: it is a description the check can read."""
    titled = {label for label, (meta, text) in sources.items() if _thin_source(meta, text) == "title"}
    if not titled:
        return []
    known: dict[str, set[str]] = {}
    for label in titled:
        meta, text = sources[label]
        authors = meta.get("authors") or []
        if not isinstance(authors, list):
            authors = [authors]
        known[label] = _content_stems(" ".join(
            [str(meta.get("title") or ""), *(str(a) for a in authors), str(text or "")]
        ))
    found: list[tuple[str, list[str], list[str]]] = []
    seen: set[str] = set()
    for block in re.split(r"\n\s*\n", _paper_body(paper_md)):
        for sentence in _SENTENCE_END_RE.split(" ".join(block.split())):
            labels: list[str] = []
            for match in _CITATION_BRACKET_RE.finditer(sentence):
                labels += [x for x in _citation_labels(match.group(1)) if x not in labels]
            if not labels or any(label not in titled for label in labels) or sentence in seen:
                continue
            seen.add(sentence)
            said = _content_stems(_CITATION_BRACKET_RE.sub(" ", sentence))
            extra = said - set().union(*(known[label] for label in labels))
            if len(extra) >= _TITLE_ONLY_UNEXPLAINED_STEMS:
                found.append((sentence, labels, sorted(extra)))
    return found


# The claim check's closed choices (agents/claim_check.md): a claim's basis, and what is wrong with an unsupported one.
_CLAIM_BASES = ("experiment", "citation", "unsupported")
_CLAIM_MODEL_FAULTS = ("number", "scope", "wording")
# FI's own faults: ``no_source``, a claim whose source can back nothing (held as its title only, retracted); ``unread``, a
# claim whose basis or fault could not be read after the one ask (it counts as unsupported and its sentence goes).
_CLAIM_FAULTS = (*_CLAIM_MODEL_FAULTS, "no_source", "unread")
# The faults whose sentence is taken out of the paper; a ``wording`` claim stays and is stated in the Limitations.
_CLAIM_FAULTS_TAKEN_OUT = frozenset({"number", "scope", "no_source", "unread"})


def _claim_choice(value: Any) -> str | None:
    """A closed choice as written (spaces and case aside); ``None`` for JSON null; anything else as its text."""
    if value is None:
        return None
    return str(value).strip().lower()


def _claim_choice_problem(c: dict[str, Any]) -> str:
    """``""`` when the claim check's ``basis`` and ``fault`` for claim ``c`` are among the closed choices and agree, else
    what is wrong, in plain words. A supported claim with no ``fault`` key is read as ``null``: its basis already says it is
    supported. Nothing else is read leniently: an unknown word is asked again, never mapped to the nearest choice."""
    basis = _claim_choice(c.get("basis"))
    if basis not in _CLAIM_BASES:
        return f"its basis {c.get('basis')!r} is not one of experiment, citation, unsupported"
    fault = _claim_choice(c.get("fault"))
    if basis == "unsupported":
        if fault not in _CLAIM_MODEL_FAULTS:
            return (f"it is unsupported and its fault {c.get('fault')!r} is not one of number, scope, wording"
                    if "fault" in c else "it is unsupported and gives no fault (one of number, scope, wording)")
    elif fault is not None:
        return f"its basis is {basis} and its fault {c.get('fault')!r} is given; a supported claim's fault is null"
    return ""


def _claim_keys(value: Any) -> list[str]:
    """The ``{{v:...}}`` keys a claim's fault is about, as the check wrote them, without their braces."""
    out: list[str] = []
    for k in value if isinstance(value, list) else []:
        key = str(k or "").strip()
        if key.startswith("{{") and key.endswith("}}"):
            key = key[2:-2].strip()
        if key and key not in out:
            out.append(key)
    return out


def _claim_faults_from_sources(claims: list[dict[str, Any]], sources: dict[str, tuple[dict[str, Any], str]],
                               typed_untraced: Callable[[str], bool]) -> list[dict[str, Any]]:
    """``claims`` with the fault FI's own checks set: a claim the retracted-source rule made unsupported (or that rests on a
    retracted source) is ``no_source``; a ``wording`` claim, or one with no fault, whose stored passage states a number the
    writer typed that FI cannot trace to the plan or its records (``typed_untraced(passage)``) is ``number`` (an old record
    with no fault too: the take-out cut such a sentence before faults were named). A fault the check gave that already
    takes the sentence out is kept. A ``no_source`` claim with no stored passage (one FI's rule added) gets its own
    sentence as its passage. The caller's claims are not changed in place."""
    retracted = {label for label, (meta, _text) in sources.items() if _retractions.is_retracted(meta)}
    out: list[dict[str, Any]] = []
    for c in claims:
        c = dict(c)
        if c.get("basis") == "unsupported" and c.get("fault") not in _CLAIM_FAULTS_TAKEN_OUT:
            if str(c.get("citation_index")) in retracted:
                c["fault"] = "no_source"
            elif any(typed_untraced(str(w.get("passage") or "")) for w in c.get("where") or []
                     if isinstance(w, dict) and str(w.get("passage") or "").strip()):
                c["fault"] = "number"
        c.setdefault("fault", None)
        c.setdefault("keys", [])
        if c["fault"] == "no_source" and not c.get("where"):
            # A sentence FI's own rule added (it cites a title-only or retracted source and the check did not list it) is
            # the paper's sentence itself: it is its own passage, so the take-out finds it exactly.
            c["where"] = [{"passage": " ".join(_paper_facts.to_placeholders(str(c.get("claim") or "")).split())}]
        out.append(c)
    return out


def _unsupported_still_in(paper_text: str, earlier: list[Any], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The claims of an earlier check of this quest marked ``unsupported`` whose text, or a passage the check stored for
    it (:func:`_passage_stands`), the paper still holds word for word (spacing aside) and that ``claims`` (this check's)
    does not state again (:func:`_same_statement`): carried over, so a check that leaves a sentence out cannot clear it.
    A sentence the writer changed is not carried (it is judged afresh)."""
    flat = " ".join(str(paper_text or "").split())
    out: list[dict[str, Any]] = []
    for c in earlier or []:
        if not isinstance(c, dict) or c.get("basis") != "unsupported":
            continue
        text = " ".join(str(c.get("claim") or "").split())
        if not text or (text not in flat and not _passage_stands(c, flat)):
            continue
        if any(_same_statement(text, str(n.get("claim") or "")) for n in [*claims, *out] if isinstance(n, dict)):
            continue
        out.append({**c, "claim": text,
                    "evidence": f"{str(c.get('evidence') or '').strip()} (an earlier check found it unsupported; the "
                                "sentence is unchanged and this check did not judge it again)".strip()})
    return out


def _passage_stands(claim: dict[str, Any], *texts: str) -> bool:
    """Whether a passage the claim check stored for ``claim`` (``where``) is still in one of ``texts`` (the writer's text,
    spaces made single): the claim is found by that exact passage, never by its wording."""
    for w in claim.get("where") or []:
        passage = " ".join(str(w.get("passage") or "").split()) if isinstance(w, dict) else ""
        if passage and any(passage in t for t in texts):
            return True
    return False


def _same_statement(a: str, b: str) -> bool:
    """Whether two texts state one thing: the shorter one's content words are
    mostly in the other. A claim the check quotes or paraphrases is compared
    with the paper's sentence this way."""
    wa, wb = _content_stems(_CITATION_BRACKET_RE.sub(" ", a)), _content_stems(_CITATION_BRACKET_RE.sub(" ", b))
    shared = len(wa & wb)
    return shared >= _SAME_STATEMENT_WORDS and shared / min(len(wa), len(wb)) >= _SAME_STATEMENT_SHARE


def _apply_title_only_rule(
    paper_md: str,
    sources: dict[str, tuple[dict[str, Any], str]],
    claims: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """``claims`` with the sentences that rest on title-only sources marked
    ``unsupported`` with fault ``no_source`` (their sentence is taken out of the paper), and how many were changed or added.

    For each sentence :func:`_title_only_sentences` finds: a claim the check
    grounded in one of its citations is marked ``unsupported`` (the quote it gave
    can only have been the title), and a sentence the check did not list as a claim
    at all is added as one. A claim the check already marked ``unsupported`` keeps
    its basis and gets fault ``no_source`` (unless its fault already takes the
    sentence out); one it grounded in the run's results is left as it is: a result
    the experiment produced does not need the citation beside it."""
    found = _title_only_sentences(paper_md, sources)
    if not found:
        return claims, 0
    out = [dict(c) for c in claims]
    changed = 0
    for sentence, labels, words in found:
        same = [c for c in out if _same_statement(c["claim"], sentence)]
        if any(c["basis"] == "experiment" for c in same):
            continue
        weaker = [c for c in same if c["basis"] == "unsupported" and c.get("fault") not in _CLAIM_FAULTS_TAKEN_OUT]
        if any(c["basis"] == "unsupported" for c in same):
            # The check found it unsupported already; the source it cites backs nothing, so its fault is FI's.
            for claim in weaker:
                claim["fault"] = "no_source"
                claim.setdefault("where", [])
                if not claim["where"]:
                    claim["where"] = [{"passage": " ".join(_paper_facts.to_placeholders(sentence).split())}]
            changed += bool(weaker)
            continue
        first = labels[0]
        why = (
            f"[{first}] is held as its title only ({str(sources[first][0].get('title') or '').strip()!r}), "
            f"which cannot show what the sentence says beyond it: {', '.join(words[:6])}"
        )
        cited = [c for c in same if c["basis"] == "citation" and str(c.get("citation_index")) in labels]
        if cited:
            for claim in cited:
                claim.update(basis="unsupported", fault="no_source", quote="", evidence=why)
                if not claim.get("where"):
                    claim["where"] = [{"passage": " ".join(_paper_facts.to_placeholders(sentence).split())}]
        elif not same:
            out.append({
                "claim": sentence,
                "basis": "unsupported",
                "fault": "no_source",
                "keys": [],
                "citation_index": int(first) if first.isdigit() else first,
                "quote": "",
                "evidence": why,
            })
        else:
            continue
        changed += 1
    return out, changed


# Claim grounding shows the model this much of each cited source's text: the
# passages most related to the sentences citing it. It saw only titles before,
# and grounded "explicit Euler is unstable in oscillatory systems" in a paper
# on discrete gradients whose abstract never mentions Euler. At 1,500 only the
# opening chunk and one 1,200-character passage fit, so a source cited by six
# sentences showed one passage, and supported citations were rejected.
_CLAIM_SOURCE_CHARS = 6000
# The citation check shows the reader this much of each cited source: the passages most related to the sentences citing it.
_CITATION_SOURCE_CHARS = 2500
# A quote shorter than this could be found in almost any source.
_QUOTE_MIN_CHARS = 25
# A paper has a title and sections. Text this short with no heading at all is
# the provider's error message ("You've hit your weekly limit ...") that
# arrived as content; three Sonnet quests ended rc=0 with a review "accept" on
# exactly that.
_MIN_PAPER_CHARS = 300

# What the design and code-writing prompts say when execution.background_jobs is
# on. The contract itself, and how FI acts on it, is core/job_watch.py.
_JOB_PROTOCOL = """\
## The experiment is a background job (execution.background_jobs is on)

The real simulation runs on a cluster or takes longer than the wall-time limit, so `experiment.py` must NOT wait for it. Write it as an idempotent driver that FI runs again and again, following the selected skill and the user's example files for how to submit, how to tell that the job finished, failed or is still running, and how to read its results:
1. Keep the job's state in `job/state.json` (relative to the working directory) so every run finds what the last one did.
2. First run: prepare the inputs, submit the job, save its id in `job/state.json`, print exactly one line `RESULT_JSON: {"fi_job": {"status": "pending", "id": "<job id>", "note": "<short state, e.g. queued>", "poll_s": <seconds worth waiting before the next check>}}` and exit 0.
3. Every later run: read `job/state.json` and check the job. Still running: print the same pending line with an updated `note` and exit 0. Never submit a second job.
4. Job finished: read its outputs, draw the figures into `figures/` as usual, and print the real `RESULT_JSON: {...}` (no `fi_job` key) in the format this prompt asks for results.
5. Job failed: print the reason on stderr and exit non-zero.
Never sleep-wait for the job. """

# Appended to _JOB_PROTOCOL (same line), so the prompt with engine.phased off is exactly what it always was.
_JOB_NO_SEED_PROTOCOL = "Ignore FI_PILOT and FI_REPLICATE_SEED: no pilot or replicate run is made for a background job.\n"

# engine.phased only: the confirm run's job is given a seed exploration never used, and must use it.
_JOB_SEED_PROTOCOL = (
    "Ignore FI_PILOT: no pilot or replicate run is made for a background job. When you submit, read the integer in the "
    "environment variable FI_REPLICATE_SEED (0 when unset), pass it to the job and build every random generator the "
    "job uses from it (a task may add its own index to it); save it in `job/state.json` with the job id. FI gives a "
    "later run of the same study a different value, so its job draws new random numbers.\n"
)

_CLUSTER_PROTOCOL = """\

## The trials run on a cluster (execution.background_jobs is on)

The simulation is too long to run here, so FI runs the trials as a job array on the cluster: one task per setting of
the protocol's grid, each running every trial of that setting through FI's own harness. FI writes the tasks; you write
how to submit them. Reply with a THIRD fenced block, starting with the line `# file: submit.py`, after the other two.

**submit.py** is an idempotent driver FI runs again and again from the quest folder. Follow the selected skill and the
user's example files for how to submit an array job on this cluster, how to tell that it finished, failed or is still
running, and which Python the tasks use.
- The environment variable `FI_TASKS` names a JSON file: `{"count": N, "tasks": [{"index": i, "setting": "...",
  "argv": [harness, spec, out]}]}`. Task i must run `<cluster python> <argv...>` from the quest folder; nothing else.
  Submit ONE array job of N tasks (task i = the array index i). Do not change, merge or skip tasks.
- Keep the job's state in `job/state.json`. First run: submit, save the job id there, print exactly one line
  `RESULT_JSON: {"fi_job": {"status": "pending", "id": "<job id>", "note": "<short state>", "poll_s": <seconds>}}` and
  exit 0. Every later run: check the job; still running: print the pending line again; never submit a second job.
- All tasks finished (whatever each one's own result): print `RESULT_JSON: {"fi_job": {"status": "done", "id": "<job id>"}}`
  and exit 0. FI then reads every task's results itself, writes the record and runs experiment.py. FI may run
  submit.py again after that: it must print the same done line again, so never delete `job/state.json`.
- The job could not be submitted or was cancelled: print the reason on stderr and exit non-zero.
submit.py never reads or writes the tasks' results, and experiment.py does not submit anything: it reads FI_TRIALS as
above.
"""

_CLUSTER_REPLY_REMINDER = (
    "\n\nThis quest runs its trials on a cluster: reply with THREE fenced Python blocks, `# file: simulate.py`, "
    "`# file: experiment.py` and `# file: submit.py`, as the section on the cluster above says."
)
#: :data:`_CLUSTER_REPLY_REMINDER` when FI writes the analysis (core/analysis_frame.py): experiment.py is FI's, never asked for.
_CLUSTER_BODIES_REMINDER = (
    "\n\nThis quest runs its trials on a cluster: the reply also holds the fenced Python block `# file: submit.py`, after the "
    "other blocks, as the section on the cluster above says."
)

_IN_EACH_RUN_PROTOCOL = """
**What each run compares inside it** (`protocol.in_each_run`: $factors). These are not settings of their own: every run
goes through ALL their values together, on the same random numbers (one fit across all of them, or every method applied
to the same simulated data). FI gives each run the whole list in its cell, in the protocol's order (`cell["<name>"]` is
that list); the function uses every value of it, never a value or a list of its own. A number the run gives once (fitted
across all the values, a difference between two of them on the same data) is returned as a number. A number it gives FOR
EACH value is returned as a mapping of each value to its number (`"power": {"bonferroni": 0.41, "bh": 0.52}`), or as a list
of one number per value in the order the run was given them; every value is there (null where it does not exist, with
the reason under `"why_missing"`). $key_rule
A check of FI's against a known answer calls the same function on its own case with `FI_ORACLE=1` in the environment; the case is
given the same lists as any cell.
In FI_TRIALS each setting is followed by one record per value (its `cell` and `key` name the value too, as in
`"key": "m=20,procedure=bh"`, with `"level"`): a number per value is in the value's own record and a number per run in the
setting's own, each once. The analysis reports each from where it is, and never counts a number per run once per value.
"""

_SPLIT_PROTOCOL = """\
## The experiment is two scripts, and FI runs the trials (execution.split_analysis is on)

This quest keeps the simulation and its analysis apart, and FI itself runs every trial and keeps their record, so the
record of what ran is never the simulation's own statement. That CHANGES the output format and the "single file" rule
above: reply with TWO fenced Python blocks, the first starting with the line `# file: simulate.py` and the second with
the line `# file: experiment.py`, then the `DEPS:` line, and nothing else.

**simulate.py defines functions; it has no loop over settings or trials and no `if __name__ == "__main__"` block.**
- For a study with randomness: `def run_trial(cell: dict, trial_id: int, seed: int) -> dict` runs ONE trial of ONE
  setting. `cell` holds one value of each grid parameter (`{"R0": 1.5, "N": 1000}`); build ONE generator from `seed`
  (`numpy.random.default_rng(seed)` or `random.Random(seed)`) and use only it. FI calls it `runs_per_setting` times for
  every setting of the protocol's grid, each call with its own seed: the `FI_REPLICATE_SEED` rule above does not apply
  here, and the function never picks, derives or fixes a seed of its own.
- Before every call FI sets the environment variable `FI_CELL_KIND` (`cell` itself holds only the study's settings):
  `"scan"` (a setting of the grid), `"search"` (a design a search evaluates) or `"check"` (a case of one of the plan's
  checks; also when the variable is unset). Read it, as `os.environ.get("FI_CELL_KIND", "check")`, when the function
  must treat these differently; never infer the kind from the values. A child process the function starts inherits the variable, so it sees the kind of
  the call that started it.
- For a deterministic study (no randomness): `def run_cell(cell: dict) -> dict` computes one setting, called once per
  setting. It takes no seed and reads no `FI_REPLICATE_SEED`: FI calls it once for each setting, with nothing repeated.
- Each equation of the design's `model` whose role is `generates` carries its id in a comment (`# E1`) on or above the
  code that computes it (in simulate.py, or in the model's package when the code has one).
- The function RETURNS a flat dict of numbers, one entry per quantity the analysis needs from that trial
  (`{"outbreak": 1.0, "peak_day": 38.0, "final_size": 812.0}`); a trial whose code fails RAISES an exception with the
  reason instead of returning a made-up value. A quantity the method cannot give at this setting (it diverged, there is
  no crossing in the range, the feature does not print) is `null`, with the reason under `"why_missing"`
  (`"why_missing": {"peak_day": "no outbreak"}`), never a stand-in number (0, -1, a bound): FI keeps the reason beside the
  missing value and the tables say it. It does not write files, print results, or keep state between calls.
- When a protocol metric is a mean over the trials a proportion counts (`"given": "p_outbreak"`), the dict also holds,
  under that proportion's own id, 1 (this trial is in the subset) or 0 (it is not) for EVERY trial, and the value being
  averaged under that mean metric's own id (mean id `final_size_given_outbreak`:
  `{"p_outbreak": 1.0, "final_size_given_outbreak": 812.0}`): FI picks the subset trial by trial from its own record and
  looks only at those two names.
- When a protocol metric has a `known_value`, the dict holds, under EXACTLY the names it gives (`estimate`, `standard_error`, the two ends of `interval`), the trial's own estimate and its own uncertainty, computed from that trial's data, on the scale of the true value; never whether the interval covered the truth (FI counts that itself from the ends and the true value). Where a trial has no estimate it raises with the reason, unless the metric's `undefined_when` names that setting.
  A cut-off that decides it (what counts as a major outbreak) is read from the protocol's thresholds,
  `json.loads(os.environ["FI_THRESHOLDS"])`, never written into the script as a number of its own.
- When the protocol lists oracles: an oracle that names a `case` (the settings of one run) and a `measure` is run by
  FI itself: it calls run_trial / run_cell on that case and computes `measure` (one returned name, or a formula of
  them) from the dict returned, so that function must return every name the `measure` uses, computed by the real
  simulation (never the check's number already worked out in another form). When simulate.py defines both functions,
  FI measures a check on many calls of run_trial (each one trial, with FI's seed: the worst counts for a rule every run
  keeps, the average for any other check), since the study runs only run_trial and the check must test it; `run_trial` is always ONE trial and never calls or returns `run_cell`, whose mean over trials may call
  `run_trial`. Only an oracle without a `case` needs `def oracle() -> dict`,
  which computes, with the SAME simulation code, the values those oracles name, returned as a dict keyed by the oracle
  names; this replaces the `FI_ORACLE` rule above. `FI_PILOT` does not apply.

**experiment.py is the analysis.** FI runs it once, after all the trials; its top-level run (calling `main()`, printing
`RESULT_JSON`) sits under `if __name__ == "__main__":`, so importing it runs nothing. It reads FI's record of them from the file
the environment variable `FI_TRIALS` names: `json.load(open(os.environ["FI_TRIALS"]))` gives
`{"thresholds": {"<name>": <value>}, "cells": [{"cell": {...}, "key": "R0=1.5,N=1000", "planned": n, "ok": k,
"failed": f, "metrics": {"<name>": {"values": [...], "trials": [...], "count": k, "total": ...}}}]}`, one entry per
setting, the values of the trials that succeeded (`trials` gives the trial each value came from), and the protocol's
thresholds: a threshold the analysis applies (what counts as an outbreak,
a success, a pass) is read from there, never written into a script as a number of its own. It never imports or calls simulate.py. It computes the summary statistics from those values, draws the
figures into `figures/` and prints the `RESULT_JSON: {...}` last line, exactly as the rules above ask: the figure
rules, the stratification rule and the no-clamping rule are its rules, and every `<name>_values` / `_count` / `_total`
it reports comes from those lists, never from a number of its own. A result reported per setting or per group of
settings puts each one in a mapping of its own keyed like `R0=1.5` (or `R0=1.5,N=500`), with the metric's own names
inside it, never the setting inside the name (`p_outbreak_R0_1_5_count`). For a PAIRED metric it also prints, beside each
`<name>_values`, `<name>_pair_id`: the `trials` of those values, so the settings' trials are joined by id. A setting with failed trials is reported with the
trials that succeeded and says how many failed.

If an outline is given, it describes the whole experiment as one program: put the code of one trial in `run_trial` (or
`run_cell`) and the statistics, figure and `RESULT_JSON` code in experiment.py, keeping every name and signature.
"""

_SEARCH_PROTOCOL = """
## This study is a search for the best design, and FI runs the search (the plan's `optimisation` block)

FI, not the scripts, searches for the design that makes the objective as low or as high as possible: it decides which
design to try next, calls the simulation for each one, counts every call against the plan's evaluation budget and keeps
the record. The rules above still hold, with these differences:
- `cell` is ONE design: a value for every design variable (whole numbers for an integer variable), every condition the
  plan holds fixed, and each numerical setting at the value the search uses. Read them all from `cell` by name; do not
  write any of them into the script as a number of its own, and do not choose a mesh, a time step or a tolerance
  yourself.
- FI says what it is asking for in the environment variable `FI_CELL_KIND`, set before every call (the cell holds only
  the study's settings): `"search"` is a design the search evaluates (the design variables are the design to compute),
  `"scan"` is a row of the coarse scan or of the comparison grid (its design variables are that row's values, the
  baseline's where the row leaves one out), `"check"` is a case of one of the plan's checks, and so is a call with the
  variable unset. Decide which of these you were called with from `os.environ.get("FI_CELL_KIND", "check")` alone,
  never from the values (a grid row at the baseline and a search design at the baseline have the same values). A
  child process the function starts inherits the variable, so it sees the kind of the call that started it.
- The returned dict holds the objective and every constrained quantity, under the exact names the plan gives them
  (`objective.quantity`, each `constraints[].quantity`), plus anything else the analysis wants.
- simulate.py never searches, optimises or loops over designs itself (no `scipy.optimize`, no Optuna, no grid of
  designs): one call computes one design.
- experiment.py does not read `FI_TRIALS` for the search. It reads FI's record: `json.load(open(os.environ["FI_BEST_DESIGN"]))`
  gives `{"best": {"design": {...}, "objective": ..., "constraints": {...}} or null, "baseline": {"design": {...},
  "objective": ..., "feasible": ...}, "improvement": {"value": ..., "relative": ...} or null, "evaluations": {"search":
  n, "budget": m, "stopped_because": "..."}, "method": {"used": "..."}, ...}`, and the file `os.environ["FI_OPTIMISATION"]`
  names holds one JSON object per line; those with `"event": "evaluation"` are the evaluations in order (`n`, `stage`,
  `start`, `design`, `objective`, `constraints`, `feasible`, `status`). Draw the best objective found so far against
  the evaluation number (one line per starting point, `start`), and a table of the baseline against the best design.
  When the plan has a coarse scan (`optimisation.grid`), `FI_TRIALS` names its per-setting results as described above:
  draw them as a map (a condition held fixed that the scan compares is one of its axes: report the study across it). Its `RESULT_JSON` reports the best design, its objective, the baseline's and the improvement,
  read from that file; a best of null means no design met every limit, and it says so. It never writes either file.
- Before the analysis runs, FI checks the best design at finer numerical settings: `best["check"]` holds `"says"`
  (one plain sentence: report THIS, in its words, never the internal `"verdict"` code, which is only for branching),
  `"design"`, `"objective"` and `"baseline_objective"` at the finest settings, `"improvement"`,
  `"improvement_numerical_error"` (the numerical error of the two designs added) and `"best_numerical_error"` (the
  best design's own); the whole record is the file `os.environ["FI_OPTIMUM_CHECK"]` names. Report those checked numbers
  beside the search's: an improvement that disappears at finer settings is the result, not something to hide.
"""

# How an oracle's expected value may be sourced, for a request to rewrite the plan (core/oracle_check.py reads it).
_REFERENCE_FORMS = (
    "`derivation: <how the number follows, with at least one equation using =>`, a source from the plan's list of the sources this quest found, by its [n], an equation "
    "of the plan's `model` by its id (E1), or, for a second implementation, what it is and that it shares no code with "
    "the simulation. A source that is not in that list does not count."
)

#: The plain ending of a stop for a code finding FI's records cannot tie to a reported result.
_UNPLACED_FINDING = ("the finding's test fails and FI cannot find which of the results it changes, so it can neither keep the "
                     "results nor leave some out; fix the code the finding names, or say the finding does not affect the "
                     "results")

# Written once FI has asked the plan step to fill in where the checks' expected values come from (``_fill_plan_sources``).
_FILL_MARKER = "plan_sources_asked.json"
# Written each time FI asks the plan's model to write the search part of a plan for the best design
# (``_ask_plan_for_the_search``): how many times it asked of this plan, so a resume does not ask again; at most this many.
# The kinds of stop before the study (``Engine._stop_before_run``): the key in ``exec_result`` that holds the plain text.
# ``cannot_repair``: FI's repairs of the code ended and the problem stands (the run record differs from the protocol, the
# numerics still warn, the code still lacks what the plan requires ...): the quest ends without a paper, nobody debugs.
_STOPPED_BEFORE_THE_STUDY = ("too_long", "not_run", "cannot_repair", "no_search")


def _fit_request(why: str, fit: dict[str, Any], kept: list[str] | None = None) -> str:
    """The request to the plan's model when the study was too long to run (Engine._no_results_verdict): what FI measured,
    how many settings fit, and what may and may not change. FI itself leaves no setting out. ``kept``: what the topic
    asks to vary, from FI's record of it (``topic_requirements.kept_when_smaller``), which the smaller study still varies."""
    from . import run_estimate as _run_estimate

    per = float(fit.get("seconds_per_setting") or 0.0)
    n = int(fit.get("settings_that_fit") or 1)
    limit = _run_estimate.plain_duration(float(fit.get("limit_s") or 0.0))
    each = (f"at least {per:.0f} s" if fit.get("at_least") else f"about {per:.0f} s") if per >= 1 else f"about {per:.2g} s"
    check = int(fit.get("check") or 0)
    room = max(1, n - check)
    search = ""
    if fit.get("search"):
        search = (" A search's evaluations count as settings too: each costs about as much as one setting, so its "
                  "evaluation_budget (starts times per_start) is part of the same total.")
        build = fit.get("start_build")
        if isinstance(build, dict) and build.get("numbers"):
            search += (f" Each starting point of the search first moves each of the design's {int(build['numbers']):,} "
                       f"numbers once ({int(build['evaluations']):,} of its {int(build.get('per_start') or 0):,} "
                       "evaluations).")
        alternatives = int(fit.get("alternatives") or 0)
        if alternatives > 1:
            per_alt = int(fit.get("per_alternative") or 0)
            conditions = ", ".join(str(c) for c in fit.get("alternative_conditions") or [])
            searched = int(fit.get("alternatives_searched") or alternatives)
            search += (f" The search's evaluations come from {alternatives} alternatives the plan compares"
                       + (f" (the names of {conditions} the study has)" if conditions else "")
                       + (f", {searched} of which give results different from the first one's and are searched"
                          if searched < alternatives else "")
                       + f": each searched one is searched on its own with {per_alt:,} evaluations, "
                       f"{searched * per_alt:,} in all"
                       + (f", and {int(fit.get('probes') or 0):,} more evaluations show whether an alternative gives the "
                          "first one's results" if fit.get("probes") else "")
                       + ". The total falls with fewer alternatives or a smaller evaluation_budget for each; FI does not "
                       "choose which.")
        if check:
            search += (f" After the search, FI checks the best designs at finer settings: about {check} more "
                       f"evaluations, not part of the search's budget, so the coarse scan and the search together "
                       f"have at most {room}.")
    once = float(fit.get("once_s") or 0.0)
    if once > 0:
        axes = [str(a) for a in fit.get("once_axes") or []]
        payer = (f"Every new value of {', '.join(axes)} costs extra the first time" if axes
                 else "The first setting costs extra")
        search += (f" {payer} (about {_run_estimate.plain_duration(once)} in all for this study): later settings reuse "
                   "what it computed, so " + ("fewer values of " + ", ".join(axes) + " save the most time." if axes else
                                              "that cost does not grow with the settings."))
    return (f"FI timed the study before running it: {why}. One setting takes {each} on this machine, so about {n} "
            f"settings fit the {limit} allowed with FI's margin. Change the parts of the design that set the size of the study "
            f"so that the whole study has at most {n} settings in all, counting every setting of the grid, of every "
            "sub-study and of a search's coarse scan."
            f"{search} Keep every comparison and every quantity the topic asks for, and every check, threshold and "
            "tolerance as it is: take fewer values of a swept setting, spread over the same range, give a part of the study "
            "fewer settings, or give a search fewer evaluations. A number worked out a second way only to check the method "
            "need not be worked out at every setting or evaluation: it may be worked out at a few, as long as every check "
            "of the plan still runs. Add one entry to the simplifications (a new id) that says what was made smaller and that "
            "it was made smaller to fit the time FI measured."
            + ("\n\nWhat the topic asks to vary stays varied, in whichever part of the study varies it now:\n"
               + "\n".join(kept) if kept else ""))


#: What sets the size of a study: the settings it sweeps, the division of them into sub-studies, what every run goes through
#: together, the number of runs of each setting, and the constants a run is given.
_SIZE_PARTS = ("grid", "sub_studies", "in_each_run", "runs_per_setting", "constants")
#: What sets the size of a search, inside its ``optimisation`` block.
_SEARCH_SIZE_KEYS = ("evaluation_budget", "grid", "sub_studies", "in_each_run")


def _fit_scope(fit: dict[str, Any], protocol: Any) -> _plan_scope.PartScope:
    """The parts of the plan the request of :func:`_fit_request` concerns: the parts that set the size of the study that the
    protocol has (the settings it sweeps and their division, what every run goes through together, the runs of each
    setting, the constants), for a search the keys of its ``optimisation`` block that set the number of evaluations, and
    a new entry in the simplifications saying what was made smaller. Derived from what the request asks to change and from
    the protocol's own keys, never from the topic."""
    protocol = protocol if isinstance(protocol, dict) else {}
    parts = tuple(k for k in _SIZE_PARTS if k in protocol)
    block = protocol.get("optimisation")
    inside = ({"optimisation": tuple(k for k in _SEARCH_SIZE_KEYS if k == "evaluation_budget" or k in block)}
              if fit.get("search") and isinstance(block, dict) else {})
    return _plan_scope.PartScope(parts=parts, inside=inside,
                                 entries={"simplifications": _plan_scope.EntryScope("id", frozenset(), True)})


#: The source and the approver of the change FI makes to a frozen protocol whose search's objective did not move.
#: The source and the approver of the change FI makes to a frozen protocol when the study did not fit the time allowed.
_FIT_SOURCE = "FI: the study did not fit the time allowed"
_FIT_APPROVER = "FI: the study did not fit the time allowed; the plan was rewritten once to fit the time FI measured"
_FLAT_SOURCE = "FI: the search's objective did not move, or moved only between a few values, across the design space"
_FLAT_APPROVER = ("FI: the search's objective did not move, or moved only between a few values, across the design space; "
                  "the plan was rewritten once, after results were seen")


def _flat_moved(fact: Any) -> str:
    """How the objective behaved, as a clause of the console line: it did not move, or only between a few values."""
    return ("moved only between a few values" if isinstance(fact, dict) and fact.get("values")
            else "did not move anywhere")


def _flat_shown(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def _flat_fact_sentence(fact: dict[str, Any]) -> str:
    """What FI measured of a search whose objective did not move, or moved only between a few values (``values``), in
    one plain sentence (the runner's fact)."""
    if fact.get("values"):
        levels = ", ".join(f"{_flat_shown(v)} ({n} of them{'' if met else ', not meeting every limit'})"
                           for v, n, met in fact["values"])
        start = f"; the starting design gave {_flat_shown(fact['start'])}" if fact.get("start") is not None else ""
        return (f"the search's first {int(fact.get('designs') or 0)} designs across the whole range of the design, which "
                f"between them changed every number of it, gave the objective only {len(fact['values'])} different "
                f"values: {levels}{start}")
    shown = _flat_shown(fact.get("objective"))
    limits = "met every limit" if fact.get("feasible") else "did not meet every limit"
    return (f"the search's first {int(fact.get('designs') or 0)} designs across the whole range of the design, which "
            f"between them changed every number of it, all gave the objective {shown} and {limits}, as the starting "
            "design did")


def _flat_request(fact: dict[str, Any], kept: list[str] | None = None) -> str:
    """The request to the plan's model when the search's objective did not move (Engine._no_results_verdict): what FI
    measured, which results stayed and which moved, and what may and may not change. FI names no representation,
    resolution or range: the plan chooses. ``kept``: what the topic asks to vary (``topic_requirements.kept_when_smaller``)."""
    stayed = [str(n) for n in fact.get("stayed") or []]
    moved = [str(n) for n in fact.get("moved") or []]
    unsteady = [str(n) for n in fact.get("unsteady") or []]
    parts = []
    if stayed:
        parts.append("These results stayed at the starting design's value in every one of those designs: "
                     + ", ".join(stayed) + ".")
    if moved:
        parts.append("These results moved: " + ", ".join(moved) + ".")
    if unsteady:
        parts.append("These differ when the same design runs twice (timings), so they say nothing about the design: "
                     + ", ".join(unsteady) + ".")
    steps = bool(fact.get("values"))
    if not (stayed or moved or steps):
        parts.append("The simulation returns no result besides the objective and the limits, so FI cannot tell which "
                     "part of the objective is stuck.")
    said = ("So the objective moves only in jumps between a few levels, and the search had no direction to climb: a small "
            "change of the design leaves it where it was. An objective counted in whole steps of a numerical setting "
            "(the spacing of a scan, a grid) instead of read between them does this, for example. " if steps else
            "So nothing in the design space as it is written lets the objective move, and the search had nothing to "
            "climb. ")
    return ("FI ran the search for the best design: " + _flat_fact_sentence(fact) + ". " + said + " ".join(parts) + " "
            "Rewrite the design so that its objective can move: you choose how, from what FI measured and the topic (how "
            "the design is described, its range, an objective read finely enough to change with a small change of the "
            "design, or one that grades how close a design comes as well as whether it passes). Keep every case, setting, comparison and quantity the topic asks for, and every check, "
            "threshold and tolerance as it is; do not change what the objective measures to get a number above the "
            "starting design's. When the objective is built from parts (the worst of several cases, a sum of terms), "
            "return each part as a result of its own. Say in the simplifications what was changed and that it was "
            "changed after FI found that the first design's objective "
            + ("moved only between a few values." if steps else "did not move.")
            + ("\n\nWhat the topic asks to vary stays varied:\n" + "\n".join(kept) if kept else ""))


def _plan_search_mark(plan_text: str | None) -> str:
    """A short hash of the search block of the plan ``plan_text`` as written (``protocol.optimisation``; none for no text,
    or a plan without one): what a held search block is tied to (:meth:`Engine._keep_search_refused`)."""
    raw = _plan.raw_design_block(plan_text) if plan_text else None
    protocol = raw.get("protocol") if isinstance(raw, dict) else None
    block = protocol.get("optimisation") if isinstance(protocol, dict) else None
    return hashlib.sha256(json.dumps(block, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


_SEARCH_ASKED = "search_block_asked.json"
_SEARCH_REFUSED = "search_refused.json"  # the search block that could not be used, whole, with its faults by path
_DRAFTED_SEARCH_ASKED = "drafted_search_asked.json"  # the same bound, for a design drafted at the design step
#: The parts of a protocol that a redesign writes itself: it is never given the plan's in their place by
#: :meth:`Engine._inherit_plan_parts` (a search for the best design is a different kind of study, not a missing part).
_REDESIGN_OWNED_PARTS = frozenset({"grid", "sub_studies", "in_each_run", "metrics", "model", "topic_map", "optimisation",
                                   "form", "parts_left_out", "names_not_asked"})
_DRAFTED_PARTS_ASKED = "drafted_parts_asked.json"  # the same bound, for protocol parts a redesign drafted that could not be read
_LEDGER_ASKED = "ledger_asked.json"  # which ledger parts were asked about, in which design round
_LEDGER_PARTS = ("simplifications", "definitions", "steps")  # the plan's ledger parts: each asked once per quest
# Asked once per design round, never a stop: the ledger parts, and whether a check on an average holds in every trial.
_ONCE_PARTS = (*_LEDGER_PARTS, "per_trial")
_DRAFTED_GRID_ASKED = "drafted_grid_asked.json"  # one request per redesign round, for a drafted grid of more settings than FI lists
_DRAFTED_GRID_ASKED_AGAIN = "drafted_grid_asked_again.json"  # one more request per redesign round, after what the topic asked for was added
_GRID_ASKED = "grid_asked.json"  # the same bounds, for a grid of more settings than FI lists (one request per plan)
_GRID_ASKED_AGAIN = "grid_asked_again.json"  # one more request, after what the topic asked for was added to the plan
_PART_ASKED = "protocol_part_asked.json"  # the same bounds, for the protocol parts that could not be read
_PART_HEADING = "Checks already made"
_TOPIC_ASKED = "topic_asked.json"  # the same bounds, for what the topic asks for and the plan lacks
_TOPIC_DESIGN_ASKED = "topic_design_asked.json"  # one request per redesign, for a design drafted at the design step
_SEARCH_ASKS = 2
#: The most characters of a refusal kept to be sent back to the plan's model with the next request. A block with several
#: problems gives one sentence for each, and every one must reach the model, so this is room for many, not for a long one.
_REFUSAL_CHARS = 8000
#: The most characters of the plan's model's last refused block that go back with the next request, and are kept on disk.
_REFUSED_BLOCK_CHARS = 12000
#: How many times the analysis alone is sent back for a difference FI counted in its values (the simulation stands, so
#: no experiment is run again); each try uses one of the execute-reflect repairs, which stay bounded as before.
_ANALYSIS_REPAIR_TRIES = 3
#: A model's answer the reader refused for its form (a schema error, quoted back to it exactly) earns this many more asks
#: than ``_SEARCH_ASKS``, and the chat calls they need: a reason that names the field can usually be fixed in one more try.
#: An answer that never came (a failed call, no complete answer) earns nothing extra.
_SCHEMA_RETRY_ASKS = 2
_SEARCH_CALLS_PER_ASK = 2  # one request makes up to two chat calls (the asked parts as JSON, and once more if it could not be read)
_SEARCH_CALLS = 4  # at most this many chat calls in all, per plan and model


def _manifest_repair_budget(configured: int, analysis_only: bool) -> int:
    """How many times a run-manifest difference is sent back: ``engine.run_manifest_repair_attempts``, but up to
    ``_ANALYSIS_REPAIR_TRIES`` when only the analysis is at fault (a repair of its code reruns no experiment). Zero stays
    zero: a person who asked for no repairs gets none."""
    return max(configured, _ANALYSIS_REPAIR_TRIES) if analysis_only and configured > 0 else configured


class _PartsRefused(ValueError):
    """The plan's model's answer to a request for some parts of the plan was not used (FI's reason is the message)."""


class _PlanEditedMeanwhile(Exception):
    """plan.md was changed (by a person) while the plan's model was answering: the model's answer is not written."""


class ModelUnavailable(RuntimeError):
    """A step's model call met a limit that does not clear within the run (a usage limit, a dead login) and the step
    went on without the answer: the step is not kept, so the quest stops and the step runs again on resume."""

#: What a plan with no way to judge whether the code got better means, in one line (never a stop of its own).
_NO_CRITERIA_LINE = ("The plan has no measure of whether the code got better (FI looked in the literature once and found "
                     "none), so FI will not try to improve the simulation step by step. Nothing to do: the quest goes on.")
# The checks whose source that rewrite wrote after the person had read the plan (the freeze record says so).
_FILLED_AFTER_READ = "plan_sources_filled.json"
# Plan versions FI wrote itself (core/plan.py::record_version): the model's draft and the engine's own rewrites. A
# person's edit is "user" and a person's rewrite request is "request"; those, and any other value, are a person's plan.
_FI_PLAN_AUTHORS = frozenset({"model", "engine"})
# What a rewrite that fills in the checks' sources may change: where each value comes from and its kind (under any of
# the keys the engine reads them from). The words of a check only by adding to them (a "(derivation: ...)" at its end),
# and the model behind the numbers only in the parts it left out.
_FILL_FREE_KEYS = frozenset({*_oracle._KIND_KEYS, *_oracle._REFERENCE_KEYS, "derivation", "check", "expected_formula"})


def _plan_checks_note(plan_md: str) -> str:
    """For a request to rewrite the plan: what FI reads in the design block's checks and model, and, when the plan has
    them, what is still missing (each check whose expected value has no source a reader can check, the parts of the model
    left out). Empty for a plan with neither checks nor a model. A request that says only "look at the literature and the
    formulas" is then answered where FI reads it: in the block, not in the section shown from it."""
    parsed = _plan.parse(plan_md)
    protocol = (parsed.design or {}).get("protocol") if parsed.design else None
    if not isinstance(protocol, dict) or not (protocol.get("oracles") or protocol.get("model")):
        return ""
    lines = [
        "# The checks and the model (FI reads these in the design block)",
        "",
        "- *The model behind the numbers* above the block is only SHOWN from the block and is never read: a change to a "
        "check or to the model is made in the block (`protocol.oracles`, `protocol.model`), and the section is shown "
        "again from it.",
        f"- Each check's `kind` is one of: {', '.join(_oracle.KINDS)}.",
        f"- Each check's `reference` says where its `expected` value comes from: {_REFERENCE_FORMS} A derivation must "
        f"{_oracle.DERIVATION_RULE}.",
        "- `model` has `summary` (one sentence: what produces the numbers), `assumptions` (a list), `holds_for`, and "
        "`equations` (a list, each with `id` E1, E2 ..., `formula`, `role` generates or analyses, and `source`: a [n] from "
        "the plan's list of the sources this quest found, or `derivation` with the steps in `derivation`).",
        "- A check may carry `expected_formula`: its `expected` as one formula FI computes itself before anything runs "
        "(see the plan's instructions for what a formula may use); where it and `expected` differ by more than `tolerance`, "
        "FI uses the formula's value.",
        "- Keep each check's `name`, `expected`, `tolerance`, `case` and `measure` unless the request is about them.",
    ]
    try:
        named = _oracle.unsourced(protocol, _plan.listed_sources(plan_md))
        missing = _oracle.model_missing(protocol)
    except Exception:  # noqa: BLE001 -- the rules above are still worth saying
        named, missing = [], []
    if named or missing:
        lines += ["", "Missing right now (fill these in if the request is about the checks, their sources or the model):"]
        lines += [f"- {why}" for _n, why in named]
        if missing:
            lines.append(f"- the model behind the numbers leaves out {', '.join(missing)}")
    return "\n".join(lines) + "\n"


def _declare_scope(incomplete: bool) -> _plan_scope.PartScope:
    """The parts the request for the plan's checks concerns (``Engine._declare_oracles``): the numbers of the declared
    checks it names (nothing else of a check), or, when the protocol declares none, the new checks and the criteria that read them."""
    if incomplete:
        fields = frozenset({"expected", "tolerance", "tolerance_mode", "expected_formula", "derivation",
                            *_oracle._KIND_KEYS, *_oracle._REFERENCE_KEYS})
        return _plan_scope.PartScope(entries={"oracles": _plan_scope.EntryScope("name", fields)})
    return _plan_scope.PartScope(parts=("criteria",), entries={"oracles": _plan_scope.EntryScope("name", None, True)})


def _fill_scope(checks: bool, model: bool) -> _plan_scope.PartScope:
    """The parts the request to fill in the checks' sources concerns (``Engine._fill_plan_sources``): the source and kind of
    each check it names (the other fields of a check are the plan's own), and the parts of the model behind the numbers."""
    return _plan_scope.PartScope(
        inside={"model": tuple(dict.fromkeys([*_oracle._SUMMARY_KEYS, *_oracle._ASSUMPTION_KEYS, *_oracle._HOLDS_KEYS,
                                              *_oracle._EQUATION_KEYS]))} if model else {},
        entries={"oracles": _plan_scope.EntryScope("name", frozenset(_FILL_FREE_KEYS))} if checks else {})


def _fill_changed_more(before: Any, after: Any) -> str:
    """Why a rewrite that was only to fill in the checks' sources changed more than that (``""`` when it did not): the
    design outside ``protocol.model``, and each check except for its source and kind, must be as before (a check's words
    may only be added to); a part of the model that was already written must be as it was."""
    if not isinstance(before, dict):
        return "was made on a plan whose design block could not be read"
    if not isinstance(after, dict):
        return "left a design block that cannot be read"
    old_p = before.get("protocol") if isinstance(before.get("protocol"), dict) else None
    new_p = after.get("protocol") if isinstance(after.get("protocol"), dict) else None
    old_checks = {str(o.get("name")): " ".join(str(o.get("check") or "").split()) for o in _oracle.declared(old_p)}
    for o in _oracle.declared(new_p):
        old = old_checks.get(str(o.get("name")))
        if old is not None and not " ".join(str(o.get("check") or "").split()).startswith(old):
            return f"rewrote what the check {str(o.get('name'))!r} measures"
    old_model = _oracle.model_view((old_p or {}).get("model")) or {}
    new_model = _oracle.model_view((new_p or {}).get("model")) or {}
    for part in ("summary", "assumptions", "holds_for"):
        if old_model.get(part) and old_model.get(part) != new_model.get(part):
            return f"changed a part of the model behind the numbers that was already written ({part})"
    kept = {str(e.get("id")).upper(): (e.get("formula"), e.get("role")) for e in new_model.get("equations") or []}
    for eq in old_model.get("equations") or []:
        if kept.get(str(eq.get("id")).upper()) != (eq.get("formula"), eq.get("role")):
            return f"changed or removed equation {eq.get('id')} of the model behind the numbers"
    # The model's other keys (its parameters, say) are not the fill's to change.
    raw_old = (old_p or {}).get("model")
    raw_new = (new_p or {}).get("model")
    if isinstance(raw_old, dict):
        parts = {*_oracle._SUMMARY_KEYS, *_oracle._LABEL_KEYS, *_oracle._ASSUMPTION_KEYS, *_oracle._HOLDS_KEYS,
                 *_oracle._EQUATION_KEYS}
        for key, value in raw_old.items():
            if key not in parts and (raw_new.get(key) if isinstance(raw_new, dict) else None) != value:
                return f"changed `{key}` of the model behind the numbers"
    # `source`, `sources` and `basis` on a check may mean the thing simulated: free only where they were empty or
    # already held a reference.
    free_maybe = {str(o.get("name")): {k for k in _oracle._MAYBE_REFERENCE_KEYS
                                       if not _oracle._text(o.get(k))
                                       or _oracle._LOOKS_LIKE_REFERENCE_RE.search(_oracle._text(o.get(k)))}
                  for o in _oracle.declared(old_p)}

    def core(design: dict[str, Any]) -> Any:
        out = json.loads(json.dumps(design, default=str))
        protocol = out.get("protocol")
        if isinstance(protocol, dict):
            protocol.pop("model", None)
            if isinstance(protocol.get("oracles"), list):
                protocol["oracles"] = [
                    {k: v for k, v in o.items()
                     if k not in _FILL_FREE_KEYS and k not in free_maybe.get(str(o.get("name")), set())}
                    if isinstance(o, dict) else o for o in protocol["oracles"]]
        return out

    was, now = core(before), core(after)
    if was == now:
        return ""
    old_o = {str(o.get("name")): o for o in ((was.get("protocol") or {}).get("oracles") or []) if isinstance(o, dict)}
    new_o = {str(o.get("name")): o for o in ((now.get("protocol") or {}).get("oracles") or []) if isinstance(o, dict)}
    if set(old_o) != set(new_o):
        return "added, removed or renamed a check"
    moved = [n for n in old_o if old_o[n] != new_o[n]]
    if moved:
        return f"changed {', '.join(repr(n) for n in moved)} beyond where the expected value comes from and its kind"
    return "changed the plan beyond the checks' sources and the model"

_PLAN_DIRECTIVE = """

## Also write the plan (added to the design; it replaces nothing above)

Before you design, read the Prior work above the way a reviewer would. Then add ONE more key, `plan`, to the JSON object you return, next to the design fields:

"plan": {
  "in_short": "<two or three sentences: what this quest will find out, and how>",
  "literature": [{"source": "<the title, or the [n], of a source listed above>", "says": "<what it establishes that bears on this topic>"}],
  "gap": "<what the literature leaves open, or where it disagrees, that this experiment addresses>",
  "success_criteria": ["<what result would support the hypothesis, and what would count against it>"],
  "risks": ["<how this could go wrong or mislead>"],
  "out_of_scope": ["<what this quest will not do>"]
}

Name three to eight sources, and only ones that are listed above; if the Prior work holds little that bears on the topic, say that in `gap` instead of inventing sources. The design has to follow from the plan: the hypothesis answers the gap, and the method is what would show it. The person reading this may edit the design before it runs, so write each field so that it can be edited on its own.

Also add a design key `protocol` (a sibling of `hypothesis`, NOT inside `plan`). It fixes what the experiment may not change on its own; the code that is written next is checked against it, and a script that differs is sent back:

"protocol": {
  "grid": {"<parameter, named as the code will name it>": [<every value it takes>]},
  "sub_studies": [{"name": "<a separate part of the study, in a few words>", "vary": ["<the settings of `grid` this part crosses with one another>"], "hold": {"<each other setting of `grid` with several values>": <the value this part keeps it at>}}],
  "in_each_run": {"<what ONE run goes through all of together: the conditions one fit spans (every dose of one dose-response fit, say), or alternative methods applied to the same simulated data (two estimators of one quantity)>": [<all its values; a quantity the run returns is a `metrics` entry, never a value here>]},
  "numerical_settings": {"<each numerical resolution of the simulation, named as the code names it: a sampling density, a step, a mesh size, a number of points>": {"search": <the one value the study runs at>, "check": [<finer values FI runs the headline results again at>], "finer": "<smaller | larger>", "unit": "<its unit, e.g. nm or s; leave out for a count>"}},
  "runs_per_setting": <a whole number: stochastic runs or samples per setting; leave out for a deterministic study>,
  "min_runs_per_setting": <the fewest runs per setting that still answer the question; when the study would not finish in time FI lowers the runs toward it and keeps every setting; only the topic's own time limit may take them below it (down to 10), and the paper then says so; leave out for a deterministic study>,
  "thresholds": {"<name>": <value>},
  "constants": {"<name>": <value>},
  "seed_policy": "<how randomness is seeded: one independent stream per setting and run, or say why streams are shared>",
  "shared_draws": <true when trial i of every setting must use the same random draws (common random numbers); false otherwise. It alone makes the settings share them; a metric marked paired is then compared trial by trial>,
  "failure_policy": "<how a trial that fails (a solver that does not converge, an exception) is treated: counted as a failure, excluded and reported, or retried; every failed trial is listed by the run>",
  "ci_method": "<how uncertainty is estimated, matched to what is estimated: for a proportion over pooled runs, a binomial interval; for a mean over a subset of runs, a bootstrap; a spread across a few batches is not a sample size>",
  "acceptance": ["<a rule fixed now that decides whether the hypothesis is supported, such as how close counts as converged>"],
  "precision": {"target_half_width": <the 95% half-width the headline probability needs, for example 0.03>, "metric": "<which number>", "reason": "<why that width is what the claim needs>"},
  "labels": {"<each grid parameter and each other quantity a figure plots, named as the code names it>": "<the reader's name for it: words, its symbol in $...$ and its unit, for example Activation energy $E_a$ (kJ/mol)>"},
  "metrics": [{"id": "<the name the code uses for the number in RESULT_JSON>", "label": "<the reader's name for the number, written the same way>", "estimand": "<what it estimates, for example P(outbreak | R0)>", "kind": "<proportion | mean>", "unit": "<what one observation is: a trajectory, a run, a household>", "cluster": <null, or true when observations come in clusters that are not independent (trials of one household, steps of one trajectory), or the name of the RESULT_JSON list that holds each observation's cluster>, "exact": <true only when the number is a count or a share of counts, so every run gives the same value; false or left out for a measurement or a value worked out with decimal arithmetic>, "paired": <true when the settings are compared trial by trial, which needs shared_draws true at the protocol level; false otherwise, including two estimates made from the same trial's data (what one run compares on the same data goes in in_each_run)>, "family": "<the set of comparisons a multiplicity correction covers, for example R0 contrasts>", "given": "<only for a mean over a subset of the trials (the final size of the runs that became major outbreaks, say): the id of the proportion metric whose successes are that subset; leave it out otherwise. Reported per stratum, each stratum is a mapping of its own keyed like R0=1.5, holding <given>_count and <id>_values under their own names>", "known_value": {"true_value": <the true value the simulation generates the data with, on the estimate's own scale: a number, or the name of the threshold, of the `constants` entry or of the grid parameter that holds it>, "estimate": "<the name run_trial returns for the trial's estimate of it>", "standard_error": "<the name run_trial returns for that estimate's standard error>", "interval": ["<the name of the lower end of the trial's interval>", "<the name of its upper end>"], "level": <the interval's level, for example 0.95>, "degrees_of_freedom": <only when the interval is the estimate plus and minus one multiplier of its standard error in every setting: the whole number of the t distribution whose quantile is the multiplier (the plan's own statement, for example 3), or "normal" for the normal quantile; when the study compares values of a factor that give the interval different multipliers, a mapping {"<factor>": {"<value>": <whole number or "normal">}} with one entry per value; leave it out when the multiplier changes between settings in any other way>, "scale": <only "log", when the interval is symmetric on the log scale of the estimate (exp(ln(estimate) -/+ multiplier x s), the standard error above being the estimate's own); leave it out otherwise>}, "undefined_when": {"<grid parameter>": [<the values where this number truly has no value, so a trial returns none there>]}, @@REFERENCE_FIELD@@}],
  "oracles": [{"name": "<short name>", "kind": "<special_case | invariant | symmetry | second_implementation | convergence_rate | published_value>", "check": "<what the script measures, on which small case>", "expected": <the number the measurement must agree with, computed by you from the closed form, the limit or the invariant, not by the script; 0 for an invariant's worst violation or for the difference between two implementations>, "expected_formula": "<for a check that is not an invariant, a symmetry or a second implementation: @@FORMULA@@>", "tolerance": <a number: how far from `expected` still counts as agreeing>, "tolerance_mode": "<absolute (default) | relative>", "reference": "<where the expected value comes from, in one of the four forms below>", "case": {"<grid parameter>": <its value in one small, fast run, e.g. "dt": 0.1>}, "measure": "<how the number is computed from what that run returns: one name the simulation function returns, or a formula of them, e.g. abs(P_out - P_in) / P_in>"}],
  "model": {
    "summary": "<in one sentence, the model that produces the numbers, e.g. classical RK4 on the linear ODE y' = -y>",
    "assumptions": ["<what the model assumes>"],
    "holds_for": "<the range of parameters where it holds>",
    "equations": [{"id": "E1", "formula": "<the equation as the source writes it>", "role": "<generates | analyses>", "source": "<the [n] of a source listed above, or derivation>", "derivation": "<only when source is derivation: the steps, written out>", "example": {"inputs": {"<the equation's input, named as the code will name its argument>": <a number>}, "expected_formula": "<the equation's output for those inputs, as a formula>"}}]
  },
  "criteria": [{"name": "<short name>", "what": "<one sentence>", "oracle": "<the name of one of the oracles above>", "use": "<error (how far it lands from its expected value, the default) | value (the measured value itself)>", "direction": "<lower | higher | target>", "target": <a number: for lower the most it may be, for higher the least, for target the value aimed at>, "tolerance": <a number: how close to the target counts as met, and how much a later version may change before it counts as worse>}]
}

`grid` is crossed: the study runs every combination of the values of ALL its settings, so each setting multiplies every other. When the topic is several separate studies (for example one comparison of methods, and a different sweep of a condition that only matters for one method), do not put all their settings into one `grid`: list every setting with its values in `grid`, and divide them in `sub_studies`, one entry for each part, naming in `vary` the settings that part crosses (a setting can be in several parts). The study then runs the settings of each part in turn, so its size is the sum of the parts, not the product of every setting; a setting a part does not vary stays at the one value you give it in `hold` (write one for every other setting of `grid` that has several values). Every setting of `grid` that takes more than one value must be varied by some part, or held at each of its values by the parts between them, or its values would never be run. Leave `sub_studies` out when every setting is meant to be crossed with every other. FI runs the settings of `grid` (and of `sub_studies`) and nothing else: when the method says a setting is also run (a second step size at two ratios, an extra condition), that run must be a value of a `grid` setting or a part of `sub_studies`, never only a sentence in the method; a run the method describes and the grid does not hold is not run. (In a search for the best design the `grid` and `sub_studies` are those of its `optimisation` block, described below.)

`constants` lists every other number the method uses that is neither a grid value nor a threshold nor a setting of the model: a quantile of a distribution (1.96 for a 95% interval), a physical constant (the gas constant R), a conversion factor. Give each a name and its value. The paper may state these numbers and only these (and the plan's own settings and thresholds): any other number in the paper's text must come from FI's records, and FI refuses a number the plan does not state. Leave `constants` out when the method uses none.

`model` says what produces the numbers, so that a wrong number can be traced to the model or to the code. List its core equations, E1, E2, and so on: `generates` for an equation the simulation computes the data with, `analyses` for one used on the results (a fitted rate, an estimator). Each equation's `source` is a source listed above, by its [n], or `derivation` with the steps written out in `derivation`. A source you remember but that is not listed above does not count: derive the equation instead. @@EXAMPLE@@

Each oracle's `kind` is one of six: `special_case` (a special or limiting case with a known answer, such as an exact solution), `invariant` (a conserved quantity or other invariant), `symmetry` (a symmetry or scaling law), `second_implementation` (an independent second implementation), `convergence_rate` (a convergence rate) or `published_value` (a benchmark value a source reports). Each kind has ONE numeric form, and FI holds the plan to it: for `invariant`, `symmetry` and `second_implementation` the number is the worst violation (absolute, or relative when you divide by the reference in the formula) and `expected` is 0 (a conservation check measures `abs(P_out - P_in) / P_in` expecting 0, never the ratio expecting 1); for `special_case`, `published_value` and `convergence_rate` the number is the quantity itself (the solution at that step, the benchmark quantity, the observed order) with its known value as `expected`. On a random simulation FI runs each check's case many times; do not write how a check is read over those runs (FI asks you that afterwards, check by check). `measure` says how the number is computed from what the simulation function returns on the `case`: one returned name, or a formula of them written with numbers, + - * / ** %, parentheses and the functions abs, sqrt, exp, log, log10, log2, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, hypot, floor, ceil, min, max, sum (and pi); nothing else, and never a formula that takes nothing the simulation returns. A check on a quantity across trials (a coverage fraction, a spread) writes it with `mean(x)` or `sd(x)` of the per-trial result `x` the simulation returns, for example `abs(mean(covered) - 0.95)` or `abs(sd(estimate) / reference_sd - 1)`; FI works these out over the trials together, and such a check is not `per_trial`. FI computes it itself from what the simulation returns, so the simulation must return every name the formula uses. An `expected` compared at a finite step is the value AT that step (worked out, or the limit plus the method's known error there), never the limit as the step goes to 0; and it is written to full precision (0.36787944117144233, not 0.367879) when the tolerance is tight. Its `reference` says where `expected` comes from, in one of four forms: `derivation: <the steps that give the number, written as relations, e.g. r = sqrt(2) = 1.4142>`; a source listed above, by its [n] (and where in it); an equation of `model`, by its id (E1), whose own source counts; or, for a second implementation, what it is and that it shares no code with the simulation. FI checks this: a reference that is empty, or names a source that is not listed above, is reported, and a strict quest stops at the plan until it is fixed.

`criteria` are how a later version of the code will be judged better or worse: two to five checks of correctness that FI computes itself after every run, never the study's own finding. A criterion on a number in `metrics` or `precision` is refused, because judging the code by its result would reward bending the code towards the result. Each takes its number from exactly one of: an oracle above, by its name (`"use": "error"` for how far it lands from its expected value, such as the gap between an observed and a claimed convergence order; `"use": "value"` for the measured value, such as the worst drift of a conserved quantity), best one with a `case` so that FI runs it itself (a number the script's own `oracle()` reports is shown but never counts); or `"trials": "<a number every trial returns>"` in place of `oracle`, whose value is how fast the standard error of its mean shrinks as trials are added (use `"direction": "target", "target": 0.5`; it needs 256 trials or more, in settings of 32 trials or more). Give each its own `tolerance`, from the method's known error or the noise of the measurement. Leave `criteria` out only when nothing FI can compute says whether the code is right.

@@LEDGER@@

@@STEPS@@

`numerical_settings` names every numerical resolution of the simulation whose value could change a result (how finely a source, a pupil, a field or a time is sampled, a step, a mesh size, a number of terms), with the one value the study uses: the code reads each from the setting it is given, as it reads a setting of `grid`, never from a number of its own. After the run FI itself runs the settings where each headline result is lowest and highest again at the finer `check` values (each at least 1.1 times finer than the one before; left out, FI uses half and a quarter, or twice and four times for a count) and says in the paper whether the results are converged; `tolerance` (optional, a fraction below 1) is how much of a result's spread over the study its numerical error may be (5% when left out). A resolution left out of `numerical_settings` is one nobody checked. A resolution the study varies on purpose (a convergence study of the step itself) is a setting of `grid` with its values, and is not listed in `numerical_settings`.

`known_value` is for a number the simulation estimates and knows the true value of (a parameter it generates the data with and then recovers): give the true value, the names run_trial returns for the estimate and for its standard error and/or the two ends of its interval, and the interval's level. FI then counts itself, from the trials, how far the estimates lie from the true value, whether the standard errors match the spread of the estimates, and, when the two ends of the interval are named, how often that interval covers the true value (FI judges coverage only from the interval the trial reports, so name its two ends when the code computes one, for example a t interval); a code whose uncertainty is wrong is sent back. When your design fixes the interval's multiplier (t with 3 degrees of freedom, or the normal quantile), give it as `degrees_of_freedom`: FI then recovers the multiplier the code used from the trials' own intervals and standard errors and sends the code back if it differs from yours. Give the true value on the scale of the estimate (the log of a value when the estimate is a log). A true value the study fixes (an activation energy of 50 kJ/mol the data are generated with, and the log of a frequency factor that follows from it) goes into `constants` under a name, with its number, and `true_value` names it; a true value that follows from other constants may instead be written as a formula of their names (for example `log(k_ref) + Ea / (R * T_ref)`), which FI computes itself. Leave `known_value` out for a number with no known true value; the paper then says its uncertainty was not checked against the truth. Leave `undefined_when` out unless the number truly has no value at some settings.

@@KNOWN_RULE@@

`precision` says how tight the claim has to be, and the runs follow from it, not the other way round: a probability near 0.5 needs about 0.96/h^2 trials for a 95% half-width of h (about 1070 for 0.03, 385 for 0.05). `runs_per_setting` is the runs each seed executes and the engine runs several seeds (their counts are pooled), so say how many trials you mean. Use a grid of at least five values for any parameter you make a claim about how a result changes with (convergence, scaling, a threshold), with values close together where the behaviour changes. Say in `seed_policy` that every setting and run draws from its own stream (derived from a base seed and the setting), unless you mean common random numbers, in which case set `shared_draws` to true (that alone makes the settings share their draws), say so, and mark the metrics compared trial by trial `paired`.

`oracles` is required for an experiment that computes anything: at least one check that does not rely on the script's own numbers being right, such as a closed form the simulation must reproduce, a limiting case with a known answer, a conservation law or other invariant every run must satisfy, a small case whose exact answer can be computed another way, or a second independent implementation. Give every oracle a `case` and a `measure`: FI calls the simulation's own function (run_cell for a deterministic simulation, run_trial for a random one) on that case itself before the main run and judges the number it returns, so the check does not rest on anything the script reports about itself. A deterministic simulation's oracles always get a case. For a random simulation a case is ONE trial with one seed, so give a case to a check one trial shows exactly (an invariant every trial keeps, a setting whose outcome is certain, such as no outbreak at R0 = 0); a check that needs many trials (a probability, a mean) either has no case and is answered by the script's own oracle(), or has a case and a `measure` written with `mean(x)` or `sd(x)` of a per-trial result, which FI runs on many trials of that case. A check without a `case` is the script's own word and does not count as independent validation. (A quest set to keep one script, `execution.split_analysis: false`, has no such function: its script answers every check itself when FI_ORACLE=1, and none of them counts.) A run whose checks do not pass never reaches the main sweep.

A result that is a yes or no, or a count of settings past a threshold, needs a check on each side: a setting where it must come out yes and one where it must come out no (each a `special_case` in `oracles`), and the plan says over which settings it changes; a result that is the same at nearly every setting of the study measures nothing there.

When a quantity is calibrated separately for each condition the study compares (set to meet a target, anchored, normalised per condition), the calibration can make or remove the difference being compared: add a part of the study in which one shared calibration is used for every condition, and compare the two.

Mistakes that stop a run, each seen in real quests:
- A thing the topic names that you leave out or approximate, with no entry in `simplifications`, or a metric with no entry in `definitions`, is sent back to you to write.
- A `quote` in `definitions` that is not in the sources listed above is sent back to you: quote the source word for word, or leave `quote` empty and name your own definition in `simplifications` (put its id in `differs`).
- `grid` holds only the settings the simulation runs separately; each grid cell is simulated and its trials are counted. A value used only to classify or summarise the same runs afterwards (a cut-off, a threshold) is not a grid axis: put it in `thresholds`.
- A grid axis lists every value it takes: `[0, 99]` means two settings, not 0 to 99. Repeated random instances of one setting are trials, counted in `runs_per_setting`, not a grid axis.
- An oracle's `tolerance` is what the method can reach on that case, worked out from the method's known error at the step, size or sample count the check uses, not an ideal: a method of order p has an error of order h^p, and a conserved quantity is conserved only to that error, so a tolerance below it fails a correct script.
- An order or ratio of convergence is read where the method is in its asymptotic regime: name the time or point and the step sizes, and keep the steps small and the horizon short; far from that regime the measured order is not the method's order.
- An oracle that compares two methods compares the same quantity: two iterative solvers started differently can converge to different solutions of one problem, so compare a residual, or solutions found from the same start or in the same bracket.
- Each oracle check runs on a small, fast case: all of them together must finish in seconds (the pre-check has a time limit of a fraction of the run's timeout).
- An `expected` of 0 takes an absolute `tolerance`; a relative one around 0 cannot be judged.
- `expected_formula` is the working of `expected` as one formula FI computes itself (before anything runs, so that a value written from memory, such as the complete elliptic integral K(m), cannot be wrong): @@FORMULA@@. For example `2*pi*sqrt(1/9.81)*(2/pi)*ellipk(sin(15*pi/180)**2)`. `expected` is that formula's value to full precision; where the two differ by more than `tolerance`, FI uses the formula.

`labels` and each metric's `label` are what a reader sees on a figure's axes and in its tables: the code's names (`Ea_hat`, `pi1`) are not, and a figure axis that shows one is sent back to be renamed to its label.

@@REFERENCE_RULE@@

Every number the topic sets (a set in braces, a count of runs, a threshold) must appear in `protocol` exactly as the topic gives it, and the design's method must use those values. Add values the topic does not name only when the method needs them, and say why in `method`. Leave out the keys that do not apply (an analytical study has no grid).
"""

_PLAN_DIRECTIVE = _PLAN_DIRECTIVE.replace("@@FORMULA@@", _forms.EXPECTED_FORMULA_LANGUAGE)
_PLAN_DIRECTIVE = _PLAN_DIRECTIVE.replace("@@LEDGER@@", _ledger.DIRECTIVE_TEXT)
_PLAN_DIRECTIVE = _PLAN_DIRECTIVE.replace("@@STEPS@@", _plan_steps.DIRECTIVE_TEXT)
_PLAN_DIRECTIVE = _PLAN_DIRECTIVE.replace("@@EXAMPLE@@", _equation_tests.EXAMPLE_RULE)
_PLAN_DIRECTIVE = _PLAN_DIRECTIVE.replace("@@KNOWN_RULE@@", _metric_spec.PLAN_RULE)
# A metric's reference value, a formula FI computes at every setting and prints beside the result (core/reference_values.py).
_PLAN_DIRECTIVE = _PLAN_DIRECTIVE.replace("@@REFERENCE_FIELD@@", _reference_values.PLAN_FIELD)
_PLAN_DIRECTIVE = _PLAN_DIRECTIVE.replace("@@REFERENCE_RULE@@", _reference_values.PLAN_RULE)

# The kind of study, and the block of a search for the best design (core/optimisation_plan.py). Added after the plan
# directive, so the rules above stay as they are for a measurement.
_STUDY_TYPE_DIRECTIVE = """

## A measurement, or a search for the best design?

Add a design key `study_type` (a sibling of `hypothesis`, NOT inside `plan` or `protocol`):
- `measure` when the topic asks how a result changes, or which of some fixed alternatives does better, over settings you choose (the protocol has a `grid`);
- `find_best_design` when it asks for the design, setting or parameter values that make one result as low or as high as possible, perhaps within limits: the fin spacing and thickness that give the lowest heat-sink temperature, the lightest truss cross-section whose stress stays under a limit, the controller gain with the least overshoot.
- A topic that asks for both (how results change over settings it names, AND the best design) is `find_best_design`: the settings it compares are the block's coarse `grid` below, run and plotted before the search, and the design FI searches for is a design variable. FI runs the search itself and records every evaluation; the simulation computes the result of ONE design and never searches, fits a surrogate or loops over designs itself.

A search for the best design has NO top-level `grid`: its protocol has an `optimisation` block instead (`oracles` and `model` are still required, as above):

"optimisation": {
  "objective": {"quantity": "<the name the simulation returns it under, e.g. max_base_temperature>", "direction": "<minimise | maximise>", "unit": "<K>", "meaning": "<what it is, in words>"},
  "design_variables": [{"name": "<e.g. fin_spacing>", "low": <number>, "high": <number>, "unit": "<mm>", "kind": "<continuous | integer | choice>", "values": [<only for a choice: the values it may take>]}, {"name": "<an array, e.g. source_pixels>", "shape": [<whole numbers, e.g. 5, 5>], "kind": "<continuous | integer>", "low": <a number, or a list of that shape>, "high": <a number, or a list of that shape>}],
  "fixed": {"<a condition the simulation needs that the search may not change, e.g. heat_load_W>": <value>},
  "constraints": [{"quantity": "<e.g. mass_g>", "limit": "<'<= 120' or '>= 3'>", "unit": "<g>"}],
  "baseline": {"values": {"<every design variable>": <its value>}, "source": "<where this design comes from: a design in a source listed above by its [n], a standard design, or the one in use now>"},
  "numerical_settings": {"<a mesh size, a time step, a number of elements>": {"search": <the value the search uses>, "check": [<finer values the best designs and the baseline are recomputed at>], "finer": "<smaller | larger>", "unit": "<its unit, e.g. mm or s; leave out for a count>"}},
  "evaluation_budget": {"starts": <whole number of starting points>, "per_start": <whole number of evaluations each>, "seconds_per_evaluation": <your estimate of one evaluation, or leave out>},
  "search_method": "<bounded_local | global_then_local | exhaustive | scipy:<function> | optuna:<sampler>>",
  "grid": {"<design variable, a condition listed in fixed that the study should also be compared across, or a setting whose values are names>": [<a coarse scan run before the search, plotted; only a cell at the value held in fixed can start the search; leave out when not wanted>]},
  "sub_studies": [{"name": "<one part of the scan, e.g. one family of alternatives>", "vary": [<the settings of grid this part varies, crossed>], "hold": {"<every other setting of grid with more than one value>": <the value this part holds it at>}}],
  "in_each_run": {"<a setting every evaluation of one design goes through together, e.g. load_case>": [<all its values: the objective is one result over all of them>]},
  "improvement_tolerance": {"value": <number, in the objective's unit; a fraction when relative, 0.05 = 5%>, "mode": "<absolute | relative>"},
  "target": <the value of the objective the person hopes to reach, only when the topic names one>
}

Rules for the block:
- There is ONE objective. A second goal is written as a constraint: the lightest truss whose stress stays under 250 MPa minimises mass with the constraint stress <= 250.
- When the objective is built from parts (the worst of several cases, a sum of terms), the simulation returns each part as a result of its own too, so a part that never moves can be seen.
- Each design variable's range has a physical meaning and lies where the model holds (`model.holds_for`).
- A design that is a set of numbers (the pixels of a source, a profile) is ONE design variable with a `shape` (a list of 1 to 3 whole numbers, at most 1024 elements in all), `kind` `continuous` or `integer`, and `low` / `high` as one number for every element or a list of that shape; its baseline value is a list of that shape. The search changes every element. Never write a list for `low`, `high` or a baseline value without the `shape`, and never put an array under `grid` (the grid scans single numbers).
- The baseline is a real, feasible design (it meets every constraint): it gives a value for every design variable, inside its range, and says where it comes from. Make one oracle check the baseline design.
- `numerical_settings` lists every numerical resolution of the simulation that could change a result (how finely a source, a pupil, a field or a time is sampled, a step, a mesh size): FI also runs the first scan's lowest and highest results again at the finer values and says whether they are converged.
- The search and the check use the same model; what differs is that the check recomputes the best designs and the baseline at finer numerical settings (and, with randomness, with new seeds), so an improvement that is only a numerical error does not count. `check` values must be really finer than `search`, each level at least 1.1 times finer than the one before (a mesh size of 0.5 is checked at 0.25 and 0.125). Leave `check` out and FI uses half, then a quarter, of the search value.
- Leave out `improvement_tolerance` unless the topic or the field says how much better counts: a design then counts as better only when it beats the baseline by more than the numerical error of the two designs.
- `search_method`: name a built-in one unless the topic asks for a library; a `scipy:` or `optuna:` method is used only when the quest's environment has that library. A coarse `grid` goes with `global_then_local`.
- For a study with randomness, add `runs_per_evaluation` (runs per evaluation during the search) and `check_runs` (new runs for each design at the check).
- Every setting of the scan is crossed with every other, unless `sub_studies` divides it, as in a measurement: alternatives that each have their own parameters are separate parts (each varies its own parameters and holds the rest), so they do not multiply one another. Leave `sub_studies` out when the scan is one part.
- A setting of the scan that is a number and not a design variable is a condition: list it under `fixed` as well, with the value the search holds it at (a setting whose values are names needs no such entry).
- `in_each_run` holds only what the objective itself is computed over: the cases one design is judged across at once (several load cases, say), which the objective combines into one number. FI gives the simulation each list whole. It is not a setting of the scan, and it is not where alternatives are compared: two ways of computing the same number are an oracle check (`second_implementation`), and alternatives the study compares are settings of the scan, in parts of their own. Every evaluation runs every value in it, so each value adds to the time of every evaluation.

Do not write `study_type: find_best_design` without an `optimisation` block (the quest stops until it has one), nor an `optimisation` block beside a top-level `grid` (that grid is then moved into the block as its coarse scan). A measurement leaves `optimisation` out.
"""

_SPLIT_REPLY_REMINDER = """

## Your last reply did not hold both scripts
Reply with exactly two fenced Python blocks: the first starting with the line `# file: simulate.py`, the second starting with the line `# file: experiment.py`, then the `DEPS:` line. Nothing else.
"""

#: :data:`_SPLIT_REPLY_REMINDER` when FI writes the analysis (core/analysis_frame.py): the reply's own blocks are simulate.py and
#: the functions of derive.py; experiment.py is FI's and is never asked for.
_BODIES_REPLY_REMINDER = """

## Your last reply did not hold the `# file: simulate.py` block
Reply with exactly the blocks the output format above names: a fenced Python block starting with the line `# file: simulate.py` (the function `trial` and its helpers), then, when the format lists functions of derive.py, a fenced Python block starting with the line `# file: derive.py` holding only those functions, then the `DEPS:` line. Nothing else.
"""

_TRIAL_ORACLE_NOTE = """

THE TRIAL CONTRACT: simulate.py defines run_trial (or run_cell). An oracle that names a `case` and a `measure` is NOT
measured by anything you write for the check: FI calls run_trial (or run_cell) itself on that case and computes `measure`
(one returned name, or a formula of them) from the dict it returns, so the simulation function must return every name
the `measure` uses, computed by the real simulation. Which function FI calls: `run_trial` is ONE trial and `run_cell` is
the setting's own number (a mean or rate over many trials, or a deterministic value). When simulate.py defines both, FI
calls `run_trial(cell, trial_id, seed)` for a check of kind `invariant` (it holds in every single trial) and for every other
check unless `run_cell(cell)` is built from calls to `run_trial` (the study runs only `run_trial`, so a check must test it): then
a check on a mean or a rate is read from `run_cell`, which must return that name. NEVER make `run_trial` return
or call `run_cell`, or average many trials inside it: FI runs it once per trial, thousands of times, and keeps each
result as one trial. Put the mean over trials in `run_cell` (it may call `run_trial`, never the other way round). Only an oracle
without a `case` is measured by `def oracle() -> dict` in simulate.py, which must compute, with the same simulation code,
each such check and return them as a dict keyed by the check's name, e.g. `{"closed_form_limit": 0.4987}`. There is no
FI_ORACLE variable and no ORACLE_JSON line. Keep every comment that marks where an equation of the plan's model is
computed (`# E1`). When you fix the script, return the whole of simulate.py; when the only thing
wrong is a check itself, return its `oracle_change` and leave `code` empty, as said above.
"""

_SPLIT_RERUN_NOTE = """
The experiment is two scripts. Give simulate.py back exactly as it is unless the problem the review found is IN the simulation: the trials FI already ran are then used as they are, and only experiment.py runs again. Rewrite simulate.py only when the simulation itself has to change.
"""

def _with_newline(text: str) -> str:
    """``text`` as a paragraph of its own in a prompt, or "" when there is none."""
    return f"\n{text}\n" if text else ""


_SPLIT_REFLECT_SIMULATE = """\
SPLIT EXPERIMENT: this script is simulate.py, the simulation half of a two-script experiment. FI calls its run_trial(cell, trial_id, seed) (or run_cell(cell)) once per trial and keeps the record itself; the trials failed, or what FI ran does not match the frozen protocol (the reason is below). Fix simulate.py and return the whole of it: the function returns a dict of numbers for one trial and raises on a failed one, with no loop over settings, no files and no RESULT_JSON. Keep every comment that marks where an equation of the plan's model is computed (`# E1`).
"""

_SPLIT_REFLECT_ANALYSIS = """\
SPLIT EXPERIMENT: this script is experiment.py, the analysis half of a two-script experiment. FI has already run the trials; this script reads FI's per-setting results from the file the environment variable FI_TRIALS names (in the folder FI_RAW_DIR names) and must not run the simulation again. Fix experiment.py and return the whole of it. The files there are:
{listing}
"""


_SEARCH_REFLECT_SIMULATE = """\
THIS STUDY IS A SEARCH FOR THE BEST DESIGN: FI calls run_cell(cell) (or run_trial) once for each design it tries; `cell` holds the design variables, the conditions the plan holds fixed and the search's numerical settings, by name. The function returns the objective and every constrained quantity under the plan's own names, and never searches or loops over designs itself.
"""

_SEARCH_REFLECT_ANALYSIS = """\
SPLIT EXPERIMENT, A SEARCH FOR THE BEST DESIGN: this script is experiment.py, the analysis. FI has already run the search; this script reads FI's record of it (the file FI_BEST_DESIGN names: the best design, the baseline and the improvement; the file FI_OPTIMISATION names: one JSON object per line, the lines with "event": "evaluation" in order; with a coarse scan, FI_TRIALS names its per-setting results), must not run the simulation again, and never writes either file. Fix experiment.py and return the whole of it.
"""


def _is_not_a_paper(text: str) -> bool:
    body = text.strip()
    return len(body) < _MIN_PAPER_CHARS and not re.search(r"(?m)^#{1,6}\s", body)
# And this much of the quest's OWN evidence. Separate from the per-source
# budget above: this one bounds the findings, the supported claims and the
# results the check grounds an "experiment" claim against. The block used to
# be the first 4,000 characters of all three serialised together, and one real
# run serialised 113,460 — so 3.5% of it reached the checker, the cut landed
# inside the 12th of 14 key findings, and the 14th, an exact finite-state
# validation, never appeared at all. The check called that validation
# unsupported, the rewrite deleted the table that proved it, and the next
# review asked for the table back.
_CLAIM_EVIDENCE_CHARS = 14000
# A numeric array longer than this is shown as its length and its range. A
# probability mass function of 101 floats costs 2,000 characters and grounds
# no claim, while the fact that it is there, and what it spans, is what
# reading the results needs.
_CLAIM_ARRAY_MAX_ITEMS = 10


def _item_content(item: Any) -> str:
    """A source's whole text: from ``full_text_path`` when the state holds only its first part (``_literature_entry``),
    else the state's copy. A file that is gone falls back to the state's copy."""
    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
    meta = (item.get("metadata") if isinstance(item, dict) else getattr(item, "metadata", None)) or {}
    path = meta.get("full_text_path") if isinstance(meta, dict) else None
    if path:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            text = None
        if text is not None:
            # Values read off the figures that could not be appended to the file are still in the state's copy.
            state_text = str(content or "")
            if _FIGURE_READINGS_MARK in state_text:
                tail = state_text[state_text.index(_FIGURE_READINGS_MARK) + len(_FIGURE_READINGS_MARK):]
                # Compared with the readings the file already holds, not its whole text: a short reading paragraph
                # ("Conclusion:") that happens to occur in the paper is still a reading.
                on_disk = _split_figure_readings(text)[1]
                missing = [b.strip() for b in tail.split(_PARA)
                           if b.strip() and b.strip() != _FIGURE_READINGS_MARK and b.strip() not in on_disk]
                if missing:
                    text += _PARA + _FIGURE_READINGS_MARK + _PARA + _PARA.join(missing) + _NL
            return text
    return str(content or "")


# The code as the claim check sees it. A sentence about how the study was run (the model, the solver, a calibration, how
# many oracle trials) is checked against the code that ran it, all of it: a real run's model lived in a package the check
# never saw, and its Methods described an imaging model and a chemistry the code did not compute. The budget is the one
# the code reader has (``core/code_review.py:_MAX_CODE_CHARS``).
# It is larger than the reader's: the paper alone may take 120,000 characters of this prompt, and a model package of a
# real run was 48,000 with its scripts.
_CLAIM_CODE_CHARS = 100_000
_CLAIM_PROTOCOL_CHARS = 3000
#: A file whose name has one of these as a whole part (split on ``_``, ``-`` and ``.``) only draws; it comes last.
_CLAIM_PLOT_WORDS = frozenset({"plot", "plots", "figure", "figures", "fig", "figs", "viz", "chart", "charts"})


def _claim_code_order(rel: str) -> tuple[int, int, str]:
    """Where a file comes first in what the claim check is shown: simulate.py, experiment.py, then the model and
    simulation modules, then the entry points, and the files that only draw last."""
    name = rel.rsplit("/", 1)[-1].lower()
    if rel == _split_run.SIMULATE_NAME:
        return (0, 0, rel)
    if rel == "experiment.py":
        return (0, 1, rel)
    if set(re.split(r"[_.\-]+", name)) & _CLAIM_PLOT_WORDS:
        return (3, 0, rel)
    if name in ("run.py", "__init__.py", "__main__.py", "main.py"):
        return (2, 0, rel)
    return (1, 0, rel)


def _claim_fi_record_block(quest_root: Path) -> str:
    """FI's own record of what ran, from its trial ledger: the settings run and the trials per setting. FI's engine runs
    the trials (``core/trial_runner.py``), so the code does not show them. ``""`` when FI wrote no ledger (a script that
    drove its own runs)."""
    try:
        rows = _trial_runner.read_ledger(quest_root)
    except Exception:  # noqa: BLE001 - no readable record adds nothing; it never loosens a check
        return ""
    if not rows:
        return ""
    per_cell: dict[str, Counter] = {}
    for row in rows:
        per_cell.setdefault(str(row.get("cell")), Counter())[str(row.get("status"))] += 1
    total: Counter = Counter()
    for c in per_cell.values():
        total.update(c)
    axes: dict[str, list[str]] = {}
    for key in per_cell:
        for part in key.split(","):
            name, eq, value = part.partition("=")
            if eq and name.strip():
                seen = axes.setdefault(name.strip(), [])
                if value.strip() not in seen:
                    seen.append(value.strip())
    sizes = sorted({sum(c.values()) for c in per_cell.values()})
    per = (f"{sizes[0]:,}" if len(sizes) == 1 else f"{sizes[0]:,} to {sizes[-1]:,}")
    lines = [
        "WHAT FI RAN (FI's own record of the trials it ran; the engine runs them, so the code does not show them):",
        f"- {len(per_cell):,} setting(s), {sum(total.values()):,} trial(s) in all"
        f" ({total.get('ok', 0):,} finished, {total.get('failed', 0):,} failed); {per} trial(s) per setting.",
    ]
    for name, values in axes.items():
        shown = ", ".join(values[:30]) + (f", ... ({len(values)} values)" if len(values) > 30 else "")
        lines.append(f"- {name}: {shown}")
    return "\n".join(lines)


def _claim_code_block(code_dir: Path) -> str:
    """The code that ran, for the claim check: every ``.py`` of ``code/`` except tests and caches, with line numbers, the
    model and simulation files first and the plotting files last, as much as ``_CLAIM_CODE_CHARS`` holds. What does not
    fit is cut or left out and said so by name."""
    files = _grid_keys.code_files(code_dir)
    if not files:
        return ""
    parts: list[str] = []
    cut: list[str] = []
    used = 0
    full = True
    for rel in sorted(files, key=_claim_code_order):
        if not full:
            cut.append(f"code/{rel} (not shown)")
            continue
        numbered = "\n".join(f"{i:4d}  {line}" for i, line in enumerate(files[rel].splitlines(), start=1))
        block = f"### code/{rel}\n```python\n{numbered}\n```"
        room = _CLAIM_CODE_CHARS - used
        if len(block) <= room:
            parts.append(block)
            used += len(block)
        elif room >= 2000:
            keep = numbered[: room - 200]
            parts.append(f"### code/{rel}\n```python\n{keep}\n# ... ({len(numbered) - len(keep):,} more characters of this file not shown)\n```")
            cut.append(f"code/{rel} (cut)")
            full = False
        else:
            cut.append(f"code/{rel} (not shown)")
            full = False
    text = "The code that ran (every script and module of code/ except its tests):\n\n" + "\n\n".join(parts)
    if cut:
        text += ("\n\nNOT SHOWN (the code is longer than what fits): " + "; ".join(cut)
                 + ". What this part does is not known from here; a claim that only it could show is grounded as it"
                 " would be without the code, unless a finding below says the code does not do it.")
    return text


def _claim_method_block(quest_root: Path, state: QuestState) -> str:
    """How the study was run, for the claim check: the frozen protocol (labelled as the plan), the code that ran
    (:func:`_claim_code_block`), the open findings of the second model's reading of that code, the engine's check
    verdicts, and what each figure draws."""
    parts: list[str] = []
    protocol = _frozen.protocol_of(quest_root)
    if protocol:
        text = json.dumps(protocol, indent=1, default=str)
        if len(text) > _CLAIM_PROTOCOL_CHARS:
            text = text[:_CLAIM_PROTOCOL_CHARS] + f"\n... ({len(text) - _CLAIM_PROTOCOL_CHARS:,} more characters)"
        parts.append("THE PLAN (the frozen protocol: what was planned, NOT evidence of what ran):\n" + text)
    record = _claim_fi_record_block(quest_root)
    if record:
        parts.append(record)
    # FI's run record (core/disclosure.py:run_record; printed in QUALITY.md, not in the paper): a sentence of the paper
    # that states a seed or the run time is grounded in it, with the measured run time (core/run_facts.py).
    try:
        wall, least = _paper_facts.measured_run_time(quest_root, (state.get("exec_result") or {}).get("duration_s"))
        run = _disclosure.run_record(quest_root, topic=str(state.get("topic") or ""), duration_s=wall)
        if wall is None and least is not None:
            # A quest whose trials ran before FI recorded the run's wall time: the ledger still says how long the trials
            # alone took, so a paper's "ran in under a minute" can be weighed against it.
            run += (f" FI did not record how long the whole experiment took; the trials alone took at least "
                    f"{_disclosure._plain_seconds(least)}.")
    except Exception as e:  # noqa: BLE001 -- evidence FI cannot read is left out
        logging.getLogger("frontier_insight.engine").warning(
            "[claim_check] FI's record of the run could not be read (%s: %s)", type(e).__name__, str(e)[:200])
        run = ""
    if run:
        parts.append("WHAT FI RECORDED ABOUT THIS RUN (FI's own records, measured or read by FI; the writer is given the same facts):\n"
                     + run.replace("**What FI recorded about this run.** ", ""))
    code = _claim_code_block(quest_root / "code")
    if code:
        parts.append(code)
    parts.extend(_claim_findings_parts(quest_root))
    try:
        record = json.loads((quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    # The engine's own oracle verdicts: a real run's script re-read its oracle results from the wrong folder and
    # reported them failed, and the check called the paper's true "every oracle passed" unsupported.
    oracle_note = _oracle.analysis_note(_oracle_with_reader_flags(quest_root, _oracle.last_judged(record))).strip()
    if oracle_note:
        parts.append(oracle_note.replace("[FI NOTE] ", "", 1).replace("the results below", "the results above"))
    if state.get("figures"):
        parts.append("What each figure draws:\n" + _figure_list_for_prompt(state, root=quest_root))
    else:  # a paper that says so is right: the figure list is FI's record, and an empty one is a fact too
        parts.append("What each figure draws: FI's record holds no figure from this run.")
    return "\n\n".join(parts) or "(nothing recorded)"


#: The run-manifest statuses of a main run that finished: the known-answer checks the gate could not decide are judged again
#: on its trials (``off`` and ``not_applicable``: no manifest was compared, but the run and its trials are there).
_FINISHED_RUN_STATUSES = ("ok", "warned", "extended_by_person", "off", "not_applicable")


def _rejudges_oracles_after(manifest_status: Any, failed_script: Any) -> bool:
    """Whether ``_node_execute`` judges the undecided known-answer checks again: the main run finished, its manifest was
    accepted, compared or not compared, and no script is being sent back."""
    return manifest_status in _FINISHED_RUN_STATUSES and not failed_script


def _claim_findings_parts(quest_root: Path) -> list[str]:
    """What the second model's reading of the code left open, for the claim check: the engine's own list (the same one
    the paper's limitations carry) under a header that says how to weigh it; when no second model read the code and
    nothing else is open, one neutral line."""
    code_dir = quest_root / "code"
    try:
        f = _code_review.open_findings(quest_root, code_dir)
        text = _code_review.findings_block(quest_root, code_dir).strip()
    except Exception:  # noqa: BLE001 - a record that cannot be read adds nothing; it never loosens a check
        return []
    if not text:
        return []
    if f["unread"] and not any(f[k] for k in ("missing", "different", "flagged", "unshown", "conflicts", "not_reviewed", "second_version")):
        return ["No second model compared the code with the plan, so no findings about the code are listed."]
    if f["second_version"] and not f["reviewed"] and not any(f[k] for k in ("reread", "missing", "different", "flagged", "unshown",
                                                       "conflicts", "not_reviewed", "stopped")):
        return [
            "TWO INDEPENDENTLY WRITTEN VERSIONS OF KEY FUNCTIONS DISAGREED, or could not be compared (FI's own check; no second "
            "model read the code). A claim that rests on a function named below is not confirmed by an independent version "
            "and is `unsupported` unless the results show it by themselves.\n" + text]
    return [
        "A SECOND MODEL'S READING OF THE CODE AGAINST THE PLAN (it can be wrong). A claim that a point below says the "
        "code does not do, or does differently, is `unsupported` unless the code above plainly does it.\n" + text
    ]


def _oracle_with_reader_flags(quest_root: Path, judged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``judged`` with each check marked that an open finding of the second model's reading of the code concerns."""
    try:
        flagged = _code_review.flagged_checks(quest_root, quest_root / "code", [str(j.get("name")) for j in judged])
    except Exception:  # noqa: BLE001 - a record that cannot be read marks nothing; it never loosens a check
        return judged
    return _oracle.with_reader_flags(judged, flagged)


def _claim_source_block(label: str, meta: dict[str, Any], text: str, sentences: list[str], *, readings: str = "") -> str:
    """One source for the claim check: its label and title, and, when the
    paper cites it, the passages of its text most related to the citing
    sentences, then what a model read off its figures, labelled as that."""
    title = str(meta.get("title") or "").strip()
    ident = meta.get("url") if label.startswith("W") else (f"DOI:{meta['doi']}" if meta.get("doi") else "")
    mark = " [retracted]" if _retractions.is_retracted(meta) else ""
    head = f"[{label}] {title}{mark}" + (f" · {ident}" if ident else "")
    if not sentences:
        return head
    if not text.strip():
        return head + "\n(no text of this source was retrieved, so nothing can be quoted from it)"
    excerpt = _format_lit_excerpt(text, title, query=" ".join(sentences), budget=_CLAIM_SOURCE_CHARS)
    block = head + "\nText:\n" + excerpt.strip()
    if readings.strip():
        block += (
            "\nValues a model read off this source's figures (NOT the source's words: a claim resting only on these is"
            " unsupported, and they are never a quote):\n" + readings.strip()[:_CLAIM_READINGS_CHARS]
        )
    return block


def _summarise_long_arrays(obj: Any, *, max_items: int = _CLAIM_ARRAY_MAX_ITEMS) -> Any:
    """``obj`` with every all-numeric array longer than ``max_items`` replaced
    by its length and range, so the results a claim can be checked against are
    not crowded out by the raw arrays behind them."""
    if isinstance(obj, dict):
        return {k: _summarise_long_arrays(v, max_items=max_items) for k, v in obj.items()}
    if isinstance(obj, list):
        numbers = [x for x in obj if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if len(obj) > max_items and len(numbers) == len(obj):
            return f"[{len(obj)} values, {min(numbers):.4g} to {max(numbers):.4g}]"
        return [_summarise_long_arrays(v, max_items=max_items) for v in obj]
    return obj


def _claim_results_block(result_json: Any, *, query: str, budget: int) -> str:
    """As much of the run's results as ``budget`` holds: all of them when they
    fit, otherwise the top-level branches most related to ``query``, each one
    whole and in its original order.

    Keeping whole branches rather than the leading slice is what reaches a
    result the sweep buries. One real run's results opened with an
    80,000-character sweep, so every leading slice was that sweep and the
    validation branch the paper's claims rested on never appeared, at any
    budget. Long numeric arrays are summarised first, which is what makes the
    branch that matters small enough to fit."""
    if not result_json or budget <= 0:
        return ""
    summarised = _summarise_long_arrays(result_json)
    whole = json.dumps(summarised, indent=2)
    head = "\n\nThe run's results (result_json)"
    if len(head) + 2 + len(whole) <= budget:
        return f"{head}:\n{whole}"
    # Leave room for the section's own heading line, so the block as a whole
    # stays inside the budget its caller had left over.
    room = budget - 200

    def _leading_slice() -> str:
        # What the check saw before, so this can never show less than it did.
        return (f"{head}, the first {max(room, 0):,} of {len(whole):,} "
                f"characters:\n{whole[:max(room, 0)]}")[:budget]

    if not isinstance(summarised, dict) or not summarised:
        return _leading_slice()
    from .passages import rank_by_relevance
    names = list(summarised)
    units = [u for name in names for u in _claim_result_units(name, name, summarised[name], max(room, 1), top=True)]
    # A table's rows are ranked with its column names, which say what each number in a row is.
    scores = rank_by_relevance([f"{h}\n{t}" if h else t for _g, h, t in units], query)
    # The branches that fit whole come first, as before a large branch was shown in parts, so the check never sees less
    # than it did: ranked with the many rows of a split table, the headline values were pushed out once the budget held
    # most of the rows. A branch too large to fit is shown in parts in the room left.
    pieces = Counter(group for group, _h, _t in units)
    first = {i for i, (group, _h, _t) in enumerate(units) if pieces[group] == 1}
    kept: set[int] = set()
    headers: set[str] = set()
    used = 0
    for i in sorted(range(len(units)), key=lambda i: (i in first, scores[i]), reverse=True):
        _group, header, text = units[i]
        cost = len(text) + 1 + (len(header) + 1 if header and header not in headers else 0)
        if used + cost <= room:
            kept.add(i)
            used += cost
            if header:
                headers.add(header)
    if not kept:
        return _leading_slice()
    lines: list[str] = []
    shown: set[str] = set()
    for i, (_group, header, text) in enumerate(units):
        if i in kept:
            if header and header not in shown:
                lines.append(header)
                shown.add(header)
            lines.append(text)
    body = "\n".join(lines)
    total = Counter(g for g, _h, _t in units)
    kept_of = Counter(units[i][0] for i in kept)
    left_out = [name for name in names if not kept_of[name]]
    partly = [f"{name} ({kept_of[name]} of {total[name]} parts)" for name in names if 0 < kept_of[name] < total[name]]
    # Name a few of the branches that did not fit; a sweep can have hundreds,
    # and the note must not itself crowd out the results.
    missed = ", ".join(left_out[:3]) + (
        f" and {len(left_out) - 3} more" if len(left_out) > 3 else "")
    notes = ([f"{missed} did not fit"] if left_out else []) + ([f"shown in part: {', '.join(partly[:3])}"] if partly else [])
    note = f", the parts most related to the paper ({'; '.join(notes)})" if notes else ""
    return f"{head}{note}:\n{body}"[:budget]


def _claim_cell(value: Any) -> bool:
    """A value one cell of a results table holds: a number, a word, or a short list of them (an interval)."""
    plain = (str, int, float, bool)
    if value is None or isinstance(value, plain):
        return True
    return isinstance(value, list) and len(value) <= 4 and all(v is None or isinstance(v, plain) for v in value)


def _claim_result_units(group: str, path: str, value: Any, room: int, *, top: bool = False) -> list[tuple[str, str, str]]:
    """``value`` (at ``path``, under the top-level branch ``group``) as pieces of at most ``room`` characters, each
    ``(group, table header or "", text)``. A branch that fits is one piece, whole. One that does not is split into its
    children rather than dropped: one real run's 25 settings were a single 168,000-character branch, so not one
    setting reached the check and every claim about the grid was called unsupported. Children that are mappings
    become the rows of one table, whose column names are said once in its header, so a setting is one short line."""
    if top:
        text = json.dumps({path: value}, indent=2)
    else:
        text = f"{path}: {json.dumps(value, separators=(', ', ': '))}"
    if len(text) <= room or not isinstance(value, (dict, list)) or len(value) < 2:
        return [(group, "", text[:room])]
    children = list(value.items()) if isinstance(value, dict) else list(enumerate(value))

    def label(key: Any) -> str:
        return f"{path}.{key}" if isinstance(value, dict) else f"{path}[{key}]"

    rows = [child for _key, child in children if isinstance(child, dict)]
    columns: list[str] = []
    for row in rows:
        columns += [c for c, cell in row.items() if _claim_cell(cell) and c not in columns]
    header = ""
    if len(rows) >= 2 and columns:
        header = f"{path}: a table, one row per item ({len(children)} items); columns: " + " | ".join(columns)
    out: list[tuple[str, str, str]] = []
    for key, child in children:
        if not (header and isinstance(child, dict)):
            out += _claim_result_units(group, label(key), child, room)
            continue
        cells = " | ".join(json.dumps(child[c], separators=(",", ":")) if c in child and _claim_cell(child[c]) else "-"
                           for c in columns)
        out.append((group, header, f"{label(key)}: {cells}"[:room]))
        for c, cell in child.items():
            if not _claim_cell(cell):
                out += _claim_result_units(group, f"{label(key)}.{c}", cell, room)
    return out


def _normalized_text(text: str) -> str:
    """Lower case, one space between words, straight dashes, no quote marks,
    and words a PDF hyphenated across lines joined again."""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[‘’‚‛“”„\"'`´]", "", text.lower())
    text = re.sub(r"[‐-―−]", "-", text)
    return " ".join(text.split())


# NFKC already folds the mathematical letters themselves: every character of
# the Letterlike Symbols and Mathematical Alphanumeric Symbols blocks except
# the turned F becomes its ASCII letter (script R to R, script l to l, bold and
# double-struck A to A). These two it does not: the Weierstrass p folds to
# itself, and the reduced Planck constant folds to a stroked Latin h.
_MATH_LETTERS = str.maketrans({"℘": "p", "ħ": "h"})
# A LaTeX font command wraps a name without changing what it names, so the
# braced name is the name.
_MATH_FONT_RE = re.compile(
    r"\\(?:mathcal|mathrm|mathbf|mathbb|mathit|mathsf|boldsymbol|bm|text|operatorname)"
    r"\s*\{([^{}]*)\}"
)
# ``^`` and ``_`` mark a script the source may have rendered as layout instead.
_MATH_SCRIPT_MARK_RE = re.compile(r"[\^_]")
# Spacing around an operator is typesetting, not content. Deliberately no ``.``
# and no ``,``, and no digit and no letter: what a formula *says* -- its
# numbers, its operators and their order -- has to keep mattering, or a
# quotation that changed one of them would start matching.
_MATH_SPACE_RE = re.compile(r"\s*([()\[\]{}+\-*/=<>|])\s*")
# A subscript a PDF broke onto its own line comes back as ``R 0``.
_MATH_LETTER_DIGIT_RE = re.compile(r"\b([a-z])\s+(\d)")
# One a PDF parenthesised instead comes back as ``R(0)``, and a parenthesised
# superscript as ``)( i )``. Braces because a paper writing the same subscript
# in LaTeX writes ``R_{0}``, which the script marks above leave as ``R{0}``.
_MATH_PAREN_SCRIPT_RE = re.compile(r"(?<=[a-z0-9)])[({]([a-z0-9]{1,3})[)}]")


def _math_folded(text: str) -> str:
    """``text`` with the ways a PDF flattens mathematics into characters folded
    together, so a quotation of a formula matches the formula it quotes.

    One source wrote its reproduction number ``ℛ(0)`` in its abstract and
    ``R``-newline-``0`` in its full text, and the same exponent as ``( i )``
    and as a bare ``i``; the paper quoting it wrote ``R0`` and ``^i``. Every
    one of those is the same formula, and none is a substring of another.

    Only layout is folded. The numbers, the operators and the order of the
    tokens are left exactly as they are, so this lets a *correct* quotation
    match without letting an incorrect one match. Expects
    :func:`_normalized_text` output: lower case, single-spaced, and with the
    dashes already straightened, which is what makes folding the spaces around
    a ``-`` safe to do here."""
    text = text.translate(_MATH_LETTERS)
    text = _MATH_FONT_RE.sub(r"\1", text.replace("$", ""))
    text = _MATH_SCRIPT_MARK_RE.sub("", text)
    text = _MATH_SPACE_RE.sub(r"\1", text)
    text = _MATH_LETTER_DIGIT_RE.sub(r"\1\2", text)
    text = _MATH_PAREN_SCRIPT_RE.sub(r"\1", text)
    return " ".join(text.split())


# What text extraction does to the layout of a source, and a quotation of it
# does not repeat. Each is a further chance for a part that is not there as
# written (see _quote_in_source), and none lets a different word match.
#
# A PDF hyphenates at the end of a line and can leave a space before the
# hyphen ("inap -" newline "propriate"); _normalized_text joins only a hyphen
# that touches the word.
_SPACED_HYPHEN_BREAK_RE = re.compile(r"(\w)[ \t]+-[ \t]*\r?\n\s*(\w)")
# A stacked fraction comes out as numerator, line break, denominator ("( 1",
# "R0", ")i") with no bar; the paper quoting it writes the bar. Only a bar with
# an operand on each side is one.
_FRACTION_BAR_RE = re.compile(r"(?<=[\w)\]])\s*/\s*(?=[\w(\[])")


def _spaced_pattern(text: str) -> re.Pattern[str]:
    """``text`` as a pattern that also matches it with one stray space inside
    any of its words. A PDF puts a space into a word ("equati on", "p
    robability") and the quotation, written from the text, does not have it.

    One direction only, on purpose: the source may have a space the quotation
    does not, but the words the quotation separates stay separate, so "the
    rapist" is not found in "therapist"."""
    return re.compile(" ".join(" ?".join(map(re.escape, word)) for word in text.split()))


def _layout_views(source: str, haystack: str, folded: str) -> list[tuple[str, bool]]:
    """The source as :func:`_quote_in_source` looks in it: as normalised and
    with the mathematics folded, and, when it has a hyphen a PDF broke a word
    at with a space before it, again with that word joined. Each with whether
    the mathematics is folded, which the part being looked for must match."""
    views = [(haystack, False), (folded, True)]
    if _SPACED_HYPHEN_BREAK_RE.search(source):
        joined = _normalized_text(_SPACED_HYPHEN_BREAK_RE.sub(r"\1\2", source))
        views += [(joined, False), (_math_folded(joined), True)]
    return views


def _part_in_layouts(part: str, views: list[tuple[str, bool]]) -> bool:
    """Whether ``part`` is in the source once the layout of the source is
    allowed for: a stray space inside a word, or a line break where the
    quotation has a fraction bar.

    A formula the source has lost altogether is not a layout difference. A web
    page's text can read "Given  and , the epidemic ends at time t, when ."
    for "Given S(t) and I(t), the epidemic ends at time t, when I(t)=0.", and
    nothing in it says what the formulas were, so a quotation that spans one is
    not found: the words around it could be, but the formula between them could
    be anything."""
    shapes = [part]
    if "/" in part:
        # The one place an operator is dropped, and only from the quotation,
        # only as a last chance, and only for a bar between two operands:
        # "(1/R0)" is looked for as "(1 R0)", so "(R0/1)" is still not found
        # in "(1 R0)".
        shapes.append(_FRACTION_BAR_RE.sub(" ", part))
    for shape in shapes:
        for text, is_folded in views:
            needle = _math_folded(shape) if is_folded else shape
            # Folding can leave nothing (a "quotation" of dollar signs), and
            # nothing is in every text.
            if needle and (needle in text or _spaced_pattern(needle).search(text)):
                return True
    return False


def _quote_in_source(quote: str, source: str) -> bool:
    """Whether ``quote`` is words ``source`` has. Parts an ellipsis joins are
    looked for one by one, and together they must be long enough to mean
    something.

    A part that is not there as written is looked for again, each time with
    something a source's text extraction is known to do to its layout allowed
    for, and never with a different word allowed:

    * the mathematics folded (:func:`_math_folded`), because a source that
      renders a formula one way and a paper that quotes it another are still
      quoting it;
    * a stray space inside a word of the source (:func:`_spaced_pattern`), and a
      hyphen a PDF broke a word at with a space before it;
    * a fraction bar in the quotation where the source has the numerator and
      the denominator on separate lines.

    Every one is only ever a further chance: a quote found as written is
    accepted on that alone, and the length floor is measured before any of
    them, so no quote a source really does contain can be rejected because of
    them. The letters, digits and operators the quotation has, and the order
    they come in, must all be there; so a word changed, a word left out, two
    sentences that are not neighbours run together, a formula the source has
    lost (:func:`_part_in_layouts`) and a quotation of another source are all
    still rejected."""
    haystack = _normalized_text(source)
    parts = [p.strip(" .,;:[]") for p in re.split(r"\.\.\.", _normalized_text(quote))]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < _QUOTE_MIN_CHARS:
        return False
    if all(p in haystack for p in parts):
        return True
    folded = _math_folded(haystack)
    # Folding can leave nothing (a "quotation" of dollar signs), and nothing is
    # in every text.
    if all(p in haystack or (bool(m := _math_folded(p)) and m in folded) for p in parts):
        return True
    views = _layout_views(source, haystack, folded)
    return all(_part_in_layouts(p, views) for p in parts)


def render_references_marp_slide(refs: list[dict[str, Any]], *, paper_md: str = "") -> str:
    """One Marp slide listing the sources ``paper_md`` cites most, in
    reference order and as many as fit, appended after the LLM's deck so a
    References slide always lands when sources exist."""
    if not refs:
        return ""
    counts = _citation_counts(paper_md)
    ranked = sorted(refs, key=lambda r: (-counts.get(r["n"], 0), r["n"]))
    chosen: list[tuple[int, str]] = []
    used = 0.0
    for ref in ranked[:_SOURCE_SLIDE_MAX]:
        entry = f"- [{ref['n']}] {_ref_citation_text(ref)}"
        cost = -(-len(entry) // _SOURCE_SLIDE_CHARS_PER_LINE) + 0.5
        if chosen and used + cost > _SOURCE_SLIDE_LINES:
            break
        chosen.append((ref["n"], entry))
        used += cost
    lines = ["---", "", "## References", "", *(entry for _n, entry in sorted(chosen))]
    rest = len(refs) - len(chosen)
    if rest:
        lines += ["", f"_({rest} more {'source' if rest == 1 else 'sources'} in the paper)_"]
    return "\n".join(lines)


# A "Further reading" heading at any level, however the writer capitalised it.
_FURTHER_READING_HEADING_RE = re.compile(
    r"^#{1,6}\s*further\s+reading\s*$", re.IGNORECASE | re.MULTILINE,
)


def _further_reading_lines(further: list[dict[str, Any]]) -> list[str]:
    return [f"- [{w['label']}] {_ref_citation_text(w)}" for w in further]


def _append_further_reading(markdown: str, further: list[dict[str, Any]]) -> str:
    """Append a ``## Further reading`` section listing the web pages, unless
    there are none or the paper already has that heading. The engine writes
    it rather than the writer, so every page is listed and none is invented."""
    if not further or _FURTHER_READING_HEADING_RE.search(markdown):
        return markdown
    section = "\n".join(["## Further reading", "", *_further_reading_lines(further)])
    return markdown.rstrip() + "\n\n" + section + "\n"


# The heading of a source list a writer put in the paper. The engine writes
# both lists itself, so the writer's go.
_SOURCE_LIST_HEADING_RE = re.compile(
    r"^(#{1,6})[ \t]*(?:references|bibliography|works[ \t]+cited|literature[ \t]+cited"
    r"|further[ \t]+reading)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_ANY_HEADING_RE = re.compile(r"^(#{1,6})[ \t]", re.MULTILINE)
# Inline code and math: a bracket in them is not a citation.
_NOT_PROSE_RE = re.compile(r"`[^`\n]*`|\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)
# A bracketed number this large is a year or a count ("[1990–2020]"), not a
# citation.
_NOT_A_CITATION_NUMBER = 1000


def _strip_source_lists(markdown: str) -> str:
    """``markdown`` without the References and Further reading sections its
    writer put in. Each runs from its heading to the next heading of the same
    or a higher level."""
    while m := _SOURCE_LIST_HEADING_RE.search(markdown):
        level = len(m.group(1))
        end = next(
            (h.start() for h in _ANY_HEADING_RE.finditer(markdown, m.end()) if len(h.group(1)) <= level),
            len(markdown),
        )
        head, tail = markdown[:m.start()].rstrip(), markdown[end:]
        markdown = f"{head}\n\n{tail}" if tail else f"{head}\n"
    return markdown


# The heading line of the Further reading the engine writes, and one of its entries
# ("- [W3] Title. https://…").
_FURTHER_HEADING_LINE_RE = re.compile(r"^(#{1,6})[ \t]*further[ \t]+reading[ \t]*$", re.IGNORECASE | re.MULTILINE)
_FURTHER_ENTRY_RE = re.compile(r"^- \[(W\d+)\] ")


@dataclass(frozen=True)
class _FurtherReadingBlock:
    """The paper's ``## Further reading`` section, as offsets into the text."""

    start: int  # where the heading begins
    body_start: int  # the end of the heading line
    end: int  # where the section ends
    lines: list[str]  # the section's body, split at line ends
    entries: list[int]  # the indices in ``lines`` of its entries, in order
    labels: list[str]  # the labels of those entries: "W1", "W2"...


def _further_reading_block(markdown: str) -> _FurtherReadingBlock | None:
    """The Further reading section of ``markdown`` when it is a list the engine
    wrote: nothing in it but ``- [W1] …`` lines and blank ones. A section with
    any other line (a page's prose, a list the writer made) is not the
    engine's, so it is ``None`` and no code here touches it."""
    m = _FURTHER_HEADING_LINE_RE.search(markdown)
    if m is None:
        return None
    level = len(m.group(1))
    end = next(
        (h.start() for h in _ANY_HEADING_RE.finditer(markdown, m.end()) if len(h.group(1)) <= level),
        len(markdown),
    )
    lines = markdown[m.end():end].split("\n")
    entries: list[int] = []
    labels: list[str] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        entry = _FURTHER_ENTRY_RE.match(line)
        if entry is None:
            return None
        entries.append(i)
        labels.append(entry.group(1))
    return _FurtherReadingBlock(m.start(), m.end(), end, lines, entries, labels)


def further_reading_listed(markdown: str) -> list[str] | None:
    """The labels (``"W1"``…) of the web pages the paper's Further reading
    lists, in order; ``None`` when it has no such section written by the
    engine. What the ``further_reading`` bib export follows. A web page the
    text cites is under References instead (:func:`references_web_listed`)."""
    block = _further_reading_block(markdown)
    return None if block is None else block.labels


_REFERENCES_HEADING_LINE_RE = re.compile(r"^(#{1,6})[ \t]*references[ \t]*$", re.IGNORECASE | re.MULTILINE)


def references_web_listed(markdown: str) -> list[str]:
    """The labels (``"W6"``…) of the web pages listed under the paper's
    References, in order: the pages its text cites, which the engine lists
    there after the numbered papers, each as ``- [W6] …``."""
    m = _REFERENCES_HEADING_LINE_RE.search(markdown)
    if m is None:
        return []
    level = len(m.group(1))
    end = next(
        (h.start() for h in _ANY_HEADING_RE.finditer(markdown, m.end()) if len(h.group(1)) <= level),
        len(markdown),
    )
    return [e.group(1) for line in markdown[m.end():end].split("\n") if (e := _FURTHER_ENTRY_RE.match(line))]


def web_pages_listed(markdown: str) -> list[str] | None:
    """The labels of every web page the paper lists, under References (cited
    pages) or Further reading (the others); ``None`` when it lists none."""
    cited, further = references_web_listed(markdown), further_reading_listed(markdown)
    if not cited and further is None:
        return None
    return cited + (further or [])


def _gate_inputs(state: Any) -> dict[str, Any]:
    """What the evidence gate weighs (core/receipts.py ``gate_inputs``: one definition, for the receipt and for the
    check that the receipt is still for what is there)."""
    return _receipts.gate_inputs(state)


def _trim_further_reading(markdown: str, keep: int) -> str:
    """``markdown`` with only the first ``keep`` entries of its Further reading:
    the rest are dropped from the end, one line each. At 0 the section goes,
    heading and all. Everything outside the section, and every entry kept, is
    as it was; a paper with no such section, or ``keep`` at or above the
    number of entries, comes back unchanged."""
    block = _further_reading_block(markdown)
    if block is None or keep >= len(block.entries):
        return markdown
    if keep <= 0:
        head, tail = markdown[:block.start].rstrip(), markdown[block.end:]
        return f"{head}\n\n{tail.lstrip()}" if tail.strip() else f"{head}\n"
    dropped = set(block.entries[keep:])
    body = "\n".join(line for i, line in enumerate(block.lines) if i not in dropped)
    return markdown[:block.body_start] + body + markdown[block.end:]


def _web_labels_cited(markdown: str) -> set[str]:
    """The web-page labels (``"W2"``) the text of the paper cites, outside its
    source lists, inline code and math."""
    body = _strip_source_lists(markdown)
    spans = [(s.start(), s.end()) for s in _NOT_PROSE_RE.finditer(body)]
    labels: set[str] = set()
    for m in _CITATION_BRACKET_RE.finditer(body):
        if body[m.end():m.end() + 1] == "(" or any(s <= m.start() < e for s, e in spans):
            continue
        labels.update(label for label in _citation_labels(m.group(1)) if label.startswith("W"))
    return labels


def _number_list(numbers: list[int]) -> str:
    """``[1, 2, 3, 5]`` as ``"1–3, 5"``: three or more in a row become a range."""
    parts: list[str] = []
    i = 0
    while i < len(numbers):
        j = i
        while j + 1 < len(numbers) and numbers[j + 1] == numbers[j] + 1:
            j += 1
        parts += [f"{numbers[i]}–{numbers[j]}"] if j - i >= 2 else [str(n) for n in numbers[i:j + 1]]
        i = j + 1
    return ", ".join(parts)


def _renumber_citations(
    body: str, n_papers: int, web_labels: set[str] | None = None,
) -> tuple[str, dict[int, int], list[int | str]]:
    """Number the scholarly sources ``body`` cites 1, 2, 3… in the order it
    first cites them, and rewrite its citations to match. Web-page labels
    ([W1]) stay as they are. A label no source has (above ``n_papers``, or a
    web page not in ``web_labels``) is taken out of its citation, and a
    citation left empty goes.

    A paper written with ``{{cite:N}}`` (core/cite_labels.py) cites by those
    placeholders alone: FI renders them, and no bracket is read as a citation.
    A paper with no placeholder was written before that, and its brackets are
    read as follows (kept only so an old record can be rendered again). Not
    citations, so left alone: brackets in inline code or math, a bracket
    followed by "(" (a link), one right after a name or another bracket (an
    index: "pixels[2,3]", "m[1][4]"), and one holding a 0 or a year-sized
    number ("[0, 1]", "[1990–2020]").

    Returns the new text, the map from old number to new, and the labels
    that were taken out."""
    spans = [(s.start(), s.end()) for s in _NOT_PROSE_RE.finditer(body)]
    found: list[tuple[int, int, list[str], list[str]]] = []
    placeholders = _cite_labels.has_placeholder(body, spans)
    if placeholders:
        for m in _cite_labels.CITE.finditer(body):
            if not any(s <= m.start() < e for s, e in spans):
                labels, bad = _cite_labels.split_labels(m.group(1), lambda part: _citation_labels(part))
                found.append((m.start(), m.end(), labels, bad))
    else:
        for m in _CITATION_BRACKET_RE.finditer(body):
            before = body[m.start() - 1:m.start()]
            if (body[m.end():m.end() + 1] == "(" or (before.isalnum() or before in "_])" and before != "")
                    or any(s <= m.start() < e for s, e in spans)):
                continue
            labels = _citation_labels(m.group(1))
            if any(n == 0 or n >= _NOT_A_CITATION_NUMBER for n in (int(x) for x in labels if x.isdigit())):
                continue
            found.append((m.start(), m.end(), labels, []))
    # A label with no source behind it is removed, never turned into a number.
    mapping: dict[int, int] = {}
    for _s, _e, labels, _bad in found:
        for n in (int(x) for x in labels if x.isdigit()):
            if 0 < n <= n_papers and n not in mapping:
                mapping[n] = len(mapping) + 1
    dropped: list[int | str] = []
    pieces: list[str] = []
    last = 0
    for start, end, labels, bad in found:
        numbers = [int(label) for label in labels if label.isdigit()]
        unknown_web = [x for x in labels if x.startswith("W") and web_labels is not None and placeholders
                       and x not in web_labels]
        dropped += [n for n in numbers if n not in mapping and n not in dropped]
        dropped += [x for x in [*unknown_web, *bad] if x not in dropped]
        new = _number_list(sorted({mapping[n] for n in numbers if n in mapping}))
        pages = list(dict.fromkeys(label for label in labels if not label.isdigit() and label not in unknown_web))
        inside = ", ".join(([new] if new else []) + pages)
        if not inside:
            while start > last and body[start - 1] in " 	":
                start -= 1
        # A rendered citation never touches the word before it, so no later pass reads it as an index.
        lead = " " if inside and placeholders and start > last and body[start - 1].isalnum() else ""
        pieces += [body[last:start], lead + (f"[{inside}]" if inside else "")]
        last = end
    pieces.append(body[last:])
    return "".join(pieces), mapping, dropped


def _literature_in_citation_order(
    literature: list[Any], mapping: dict[int, int], audience: str,
) -> list[Any]:
    """``literature`` with the scholarly sources the paper cites first, in the
    order of their new numbers, then the uncited ones; every other entry
    (web pages, second copies of a work, internal records) keeps its order
    after them. Numbering from this list gives each source the number the
    paper now uses, in every output and in the writer's next prior-work
    block."""
    papers = [
        (int(label), item)
        for label, _meta, item in _labelled_sources(literature, audience)
        if not label.startswith("W")
    ]
    cited = sorted(((mapping[n], item) for n, item in papers if n in mapping), key=lambda t: t[0])
    moved = {id(item) for _n, item in papers}
    return (
        [item for _new, item in cited]
        + [item for n, item in papers if n not in mapping]
        + [item for item in literature if id(item) not in moved]
    )


def _finalize_paper_sources(
    markdown: str, literature: list[Any], audience: str,
) -> tuple[str, list[Any], list[int]]:
    """The paper with the source lists the engine writes, and the literature
    in the order those lists number it.

    The writer's own References and Further reading are removed. The
    scholarly sources the text cites are numbered 1, 2, 3… in the order it
    first cites them, the citations are rewritten to those numbers, and
    ``## References`` lists exactly the sources the text cites: the papers
    numbered, then the web pages it cites under their own ``[W…]`` labels (a
    source the text cites is always under References); ``## Further reading``
    lists the web pages it does not cite. Returns the paper, the reordered
    literature and the citation numbers that named no source."""
    labelled = _labelled_sources(literature, audience)
    n_papers = sum(1 for label, _m, _i in labelled if not label.startswith("W"))
    body, mapping, dropped = _renumber_citations(
        _strip_source_lists(markdown), n_papers, {label for label, _m, _i in labelled if label.startswith("W")})
    ordered = _literature_in_citation_order(literature, mapping, audience)
    refs = build_references(ordered, audience=audience)[:len(mapping)]
    web = build_further_reading(ordered, audience=audience)
    cited_web = _web_labels_cited(body)
    lines = [f"{r['n']}. {_ref_citation_text(r)}" for r in refs]
    lines += _further_reading_lines([w for w in web if w["label"] in cited_web])
    if lines:
        body = body.rstrip() + "\n\n" + "\n".join(["## References", "", *lines]) + "\n"
    body = _append_further_reading(body, [w for w in web if w["label"] not in cited_web])
    return body, ordered, dropped


def _literature_key(item: Any, index: int) -> str:
    """What names a source in the record of the citation check, so the order the paper numbers them in can be put back."""
    meta = (item.get("metadata") if isinstance(item, dict) else getattr(item, "metadata", None)) or {}
    key = str(meta.get("doi") or meta.get("arxiv_id") or meta.get("pmid") or meta.get("url") or meta.get("title") or "")
    return key.lower().strip() or f"#{index}"


def _reorder_literature(literature: list[Any], order: list[str]) -> list[Any]:
    """``literature`` in the order ``order`` names (keys from :func:`_literature_key`); a source it does not name keeps its
    place after them. The same list comes back when it is already in that order."""
    pool: dict[str, list[Any]] = {}
    for n, item in enumerate(literature):
        pool.setdefault(_literature_key(item, n), []).append(item)
    out: list[Any] = []
    for key in order:
        if pool.get(key):
            out.append(pool[key].pop(0))
    placed = {id(i) for i in out}
    return out + [item for item in literature if id(item) not in placed]


def _finalize_after_citation_edits(markdown: str, literature: list[Any], audience: str) -> tuple[str, list[Any]]:
    text, ordered, _mapping = _finalize_with_numbering(markdown, literature, audience)
    return text, ordered


def _finalize_with_numbering(
    markdown: str, literature: list[Any], audience: str,
) -> tuple[str, list[Any], dict[int, int]]:
    """The paper after the citation check took citations out or put one in: the sources it cites are numbered again in
    the order the text first cites them, ``## References`` lists exactly those, and the literature is reordered to
    match. The web pages the text cites are listed under References after the papers; Further reading (as the
    page-limit fit left it) keeps the pages the text does not cite, and everything after it is kept as it is."""
    start = _cs.cited_text_end(markdown)
    body, rest = markdown[:start], markdown[start:]
    listed_before = references_web_listed(rest)
    n_papers = sum(1 for label, _m, _i in _labelled_sources(literature, audience) if not label.startswith("W"))
    body, mapping, _dropped = _renumber_citations(body.rstrip(), n_papers)
    ordered = _literature_in_citation_order(literature, mapping, audience)
    refs = build_references(ordered, audience=audience)[:len(mapping)]
    web = build_further_reading(ordered, audience=audience)
    cited_web = _web_labels_cited(body)
    after = ""
    if rest:
        if _REFERENCES_HEADING_LINE_RE.match(rest):
            newline = rest.find("\n")
            heading = _ANY_HEADING_RE.search(rest, newline) if newline >= 0 else None
            after = rest[heading.start():] if heading else ""
        else:
            after = rest  # no References: the text ends at a Further reading list
    block = _further_reading_block(after) if after else None
    if block is not None:
        # A page the text cites moves to References; one it no longer cites, or that a fix took out of References,
        # goes back to Further reading; a page the page-limit fit dropped stays dropped.
        listed = set(listed_before) | set(block.labels)
        kept = _further_reading_lines([w for w in web if w["label"] in listed and w["label"] not in cited_web])
        tail = after[block.end:]
        new = ("## Further reading\n\n" + "\n".join(kept) + ("\n\n" if tail else "\n")) if kept else ""
        after = after[:block.start] + new + tail
    elif not _FURTHER_HEADING_LINE_RE.search(after):
        # No Further reading yet (every page was cited): a page listed under References that the text no longer cites
        # is not lost, it starts the section.
        gone = _further_reading_lines([w for w in web if w["label"] in set(listed_before) and w["label"] not in cited_web])
        if gone:
            section = "## Further reading\n\n" + "\n".join(gone) + "\n"
            after = (after.rstrip() + "\n\n" if after.strip() else "") + section
    lines = [f"{r['n']}. {_ref_citation_text(r)}" for r in refs]
    lines += _further_reading_lines([w for w in web if w["label"] in cited_web])
    text = body
    if lines:
        text = text + "\n\n" + "\n".join(["## References", "", *lines]) + "\n"
    else:
        text = text + "\n"
    if after.strip():
        text = text.rstrip() + "\n\n" + after
    return text, ordered, mapping


def cited_references(
    literature: list[Any], paper_md: str, *, audience: str = "external",
) -> list[dict[str, Any]]:
    """The numbered References the paper's text cites: what its References
    section lists. The slides and the bib export use this, so none of them
    lists a source the paper does not cite."""
    counts = _citation_counts(paper_md)
    return [r for r in build_references(literature, audience=audience) if r["n"] in counts]


_CLARIFY_LABELS = {
    "want_to_see": "What the user wants to see",
    "comparative_baseline": "Comparative baseline",
    "empirical_vs_theoretical": "Empirical / theoretical",
    "success_metric": "Success metric",
    "budget": "Time / compute budget",
    "output_kinds": "Desired output kinds",
    "study_depth": "Study depth",
    "paper_venue": "Paper venue / template",
    "topic_shape": "Topic shape",
    "study_type": "Measure, or find the best design (the person's answer)",
}


def _format_clarify(state: QuestState) -> str:
    """Render the resolved clarify-answer slots as a small bulleted block
    for downstream prompts. Returns an empty-marker string when clarify
    was skipped (mode=off) so the prompt still parses cleanly."""
    answers = state.get("clarify_answers") or {}
    if not answers:
        return "(none — clarify mode is off)"
    lines: list[str] = []
    for key, label in _CLARIFY_LABELS.items():
        if key not in answers:
            continue
        value = answers[key]
        if key == "study_type":
            value = _optim.answer_label(value)
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value) or "(empty)"
        lines.append(f"- **{label}**: {value}")
    return "\n".join(lines) or "(no answers recorded)"


def _clean_title(value: Any) -> str:
    """A title the user or agent gave, tidied: one line, no wrapping quotes, at most 120 characters."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split()).strip("\"'`“”‘’ ")
    if text.startswith("<") and text.endswith(">"):
        return ""
    return text[:120].rstrip()


def _simulatability_answer(sim: Any) -> tuple[str, str]:
    """``(decision, reason)`` of a ``simulatability`` answer in any of its shapes: a slot dict ``{default, reason}``,
    a bare string, or a bool (PyYAML reads an unquoted ``yes`` / ``no`` as ``True`` / ``False``, the commonest way
    the slot is written in a YAML; ``True`` means "yes"). The decision is lower-cased and stripped, not yet parsed."""
    reason = ""
    if isinstance(sim, dict):
        reason = str(sim.get("reason", "") or "").strip()
        sim = sim.get("default", "")
    if isinstance(sim, bool):
        return ("yes" if sim else "no"), reason
    if sim is None:
        return "", reason
    return str(sim).strip().lower(), reason


_SIMULATABILITY_LEAD = re.compile(
    # "No idea", "no clue", "no preference" are not a "no".
    r"^\W*(yes|uncertain|no(?!\s+(?:idea|clue|opinion|preference|comment|answer|way\s+to\s+tell)\b))\b",
    re.IGNORECASE,
)


def _leading_simulatability_word(text: str) -> str:
    """``yes`` / ``no`` / ``uncertain`` when ``text`` starts with that word ("Yes, but more than 100 lines" is "yes";
    "not sure" and "no idea" are none of them), else ``""``."""
    m = _SIMULATABILITY_LEAD.match(str(text or ""))
    return m.group(1).lower() if m else ""


def _spell_out_title_options(questions: dict[str, Any]) -> None:
    """Put the suggested titles into the title question's own words, so every interface (terminal, web, VS Code)
    shows them without knowing about a separate options field; the first suggestion is the default."""
    slot = questions.get("title")
    if not isinstance(slot, dict):
        return
    options = [t for t in (_clean_title(o) for o in (slot.pop("options", None) or [])) if t][:5]
    if options and not _clean_title(slot.get("default")):
        slot["default"] = options[0]
    slot["suggestions"] = options
    if options:
        slot["question"] = (
            f"{slot.get('question') or 'What should this study be called?'} "
            f"Suggestions: {' | '.join(options)} (or type your own)")


def _default_clarify_questions(topic: str) -> dict[str, Any]:
    """Minimal fallback when the LLM produces unparseable JSON. Keeps
    the same 5-slot shape so downstream code doesn't branch."""
    topic_hint = topic[:80].replace("\n", " ")
    return {
        "comparative_baseline": {
            "question": f"What existing method or dataset should this study be compared against, for: {topic_hint}?",
            "default": "(none specified — agent will pick a sensible baseline)",
        },
        "empirical_vs_theoretical": {
            "question": "Does this study run code and measure something, or derive results analytically?",
            "default": "empirical",
        },
        "success_metric": {
            "question": "What number changing in what direction would count as the headline result?",
            "default": "(none specified — agent will pick a metric)",
        },
        "budget": {
            "question": "Soft cap on experiment wall-clock?",
            "default": "a few minutes on a laptop CPU",
        },
        "output_kinds": {
            "question": "Which deliverables matter for this study?",
            "default": ["paper_md"],
        },
        "study_depth": {
            "question": "How deep should this study go? (brief preprint / journal-length / comprehensive review)",
            "default": "journal-length",
        },
        "paper_venue": {
            "question": "Which paper template should we use? (article / generic / neurips / iclr / ieee_access / nature_mi)",
            "default": DEFAULT_PAPER_FORMAT,
        },
        "topic_shape": {
            "question": "What's the intellectual shape of this topic? (experimental / survey / review / case_study / opinion)",
            "default": "experimental",
        },
    }


def _format_reflect_history(history: list[dict[str, Any]]) -> str:
    """Render the per-iteration repair history for the reflect
    prompt — keeps the LLM from re-trying patches that already failed."""
    if not history:
        return "(no prior repair attempts on this experiment)"
    lines: list[str] = []
    for h in history:
        lines.append(
            f"- iter {h.get('iter')}: rc={h.get('returncode')} "
            f"→ {h.get('patch_summary', '(no summary)')[:200]}"
        )
        stderr_tail = (h.get("stderr_tail") or "").strip()
        if stderr_tail:
            lines.append(f"    stderr_tail: {stderr_tail[-300:]}")
    return "\n".join(lines)


_PERSONA_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _load_persona_prefix(name: str, *, category: str = "review") -> str:
    """Load a persona-specific prefix from
    ``agents/<category>_persona_<name>.md`` (used by both ``review``
    and ``write`` categories). Falls back to a generic prefix when no
    per-persona file exists so users can declare a custom persona
    name in YAML without shipping a new file. The fallback path only
    applies to the ``review`` category — write personas without a
    file return an empty string so the caller can decide whether to
    use the default voice."""
    if not _PERSONA_NAME_RE.match(name):
        raise ValueError(
            f"invalid persona name {name!r}: must match [a-z][a-z0-9_]*"
        )
    persona_path = PROMPTS_DIR / f"{category}_persona_{name}.md"
    if persona_path.exists():
        return _write_personas.fill(name, persona_path.read_text(encoding="utf-8").strip())
    if category != "review":
        # Write personas don't have a generic-fallback prompt — return
        # empty so callers fall back to the prompt's default voice.
        return ""
    generic_path = PROMPTS_DIR / "review_persona_generic.md"
    if not generic_path.exists():
        return f"**Persona: {name}.**"
    template = string.Template(generic_path.read_text(encoding="utf-8"))
    return template.safe_substitute(persona_name=name).strip()


# A metric named as a probability or a proportion. When every seed's value
# lies in [0, 1], its interval stays there too: the validation quest reported
# an outbreak probability of 0.007 with a 95% CI of -0.002 to 0.015.
_PROPORTION_NAME_RE = re.compile(
    r"(?:^|_)(?:prob|probability|fraction|frac|proportion|share|accuracy|precision|recall|auc|f1)(?:_|$)",
    re.IGNORECASE,
)


def _replicate_env(
    exec_env: dict[str, str] | None, index: int, stride: int,
) -> dict[str, str]:
    """The environment for replicate ``index``: the seed it draws from, and
    which replicate it is.

    ``FI_REPLICATE_SEED`` strides by ``stride``, so replicate ``i`` owns
    ``[i*stride, (i+1)*stride)``. A script derives one seed per trial from the
    base it is handed, and ``base + counter`` is the obvious way to write that:
    bases one apart hand consecutive runs almost exactly the same trials, and
    the spread between them is then not sampling error at all.

    ``FI_REPLICATE_INDEX`` stays 0, 1, 2 ... It names which replicate this is,
    which is what the per-seed figure records are keyed on; it is deliberately
    NOT the seed, because the seeds are now far apart.

    Merges with the parent environment rather than replacing it:
    ``asyncio.create_subprocess_exec(env=...)`` overrides the child's whole env
    when given a dict, so passing only these two would strip PATH, PYTHONPATH,
    LANG and SystemRoot (Windows) and the venv python.exe would fail at its
    DLL-load step.
    """
    env = {
        **(exec_env or scrub_secrets()),
        "FI_REPLICATE_SEED": str(index * stride),
        "FI_REPLICATE_INDEX": str(index),
    }
    if index > 0:
        # A replicate's figures are put back to the first run's, or redrawn as the mean over the seeds: a paper copy
        # of them (core/plot_style.py) would be drawn for nothing, and on a large figure that costs as much as the
        # figure itself.
        env["FI_PAPER_FIGURES"] = "0"
    return env


def _script_reads_replicate_seed(code_path: Path) -> bool:
    """Whether the generated experiment reads ``FI_REPLICATE_SEED`` at all.

    A script that never names the variable cannot have responded to it, so its
    replicate runs are one run repeated rather than independent samples. That
    is not hypothetical: three graded quests shipped a script with a hardcoded
    ``RNG_SEED``, 300 stochastic trajectories per cell, and every replicate
    identical to the last leaf but for the ordinal the engine itself injects.

    Deliberately a source scan rather than a runtime probe. It is exact for the
    case that occurs -- the name is simply absent from the file -- and costs
    nothing. Its blind spot is a script that NAMES the variable without obeying
    it (reads and discards it, or mentions it only in a comment): that one
    passes here, and ``replicate_seed_reaches_rng`` is what asks whether the
    value reaches a generator. An unreadable file returns ``True``, because silence
    is not evidence of a fault and the runtime check still runs.
    """
    try:
        return "FI_REPLICATE_SEED" in code_path.read_text(
            encoding="utf-8", errors="replace",
        )
    except OSError:
        return True


def _code_names_replicate_seed(code_path: Path) -> bool:
    """Whether a module's code (comments aside) names ``FI_REPLICATE_SEED``. Used for the modules a script imports,
    where a note saying the seed is not used must not count as reading it. An unreadable file names nothing."""
    try:
        return "FI_REPLICATE_SEED" in _without_comments(code_path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return False


_RANDOM_SOURCE_PATTERN = re.compile(
    r"random|seed|secrets|\brng\b|default_rng|SeedSequence|\brvs\b|randn|randint|randrange|randperm|shuffle|permutation"
    r"|multinomial|gillespie|bootstrap|scramble|sobol|halton|latin_?hypercube|dropout|erdos_renyi|watts_strogatz"
    r"|barabasi_albert|torch|tensorflow|keras|\bjax\b"
    # Library calls that sample unless given a seed, with no random word in their name.
    r"|differential_evolution|dual_annealing|basinhopping|kmeans|train_test_split|spring_layout|\buuid"
    # The source is scanned with its tokens joined by spaces (``_without_comments``): ``df.sample(`` reads ``df . sample (``.
    r"|\.\s*(rand|normal|uniform|binomial|poisson|exponential|choice|choices|sample|integers|standard_normal)\s*\(",
    re.IGNORECASE,
)


def _without_comments(text: str) -> str:
    """The source with ``#`` comments removed, so a note saying "no random seed is needed" is not a random source."""
    try:
        kept = [
            tok for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            if tok.type != tokenize.COMMENT
        ]
        return " ".join(tok.string for tok in kept)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text


def _imported_module_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
            names += [f"{node.module}.{a.name}" for a in node.names]
    return names


def _script_has_random_source(code_path: Path) -> bool:
    """Whether the script, or a module it imports from its own folder, can draw a random number at all.

    A script with none (an ODE integrator, a closed-form sweep, a lattice sum) is deterministic by construction,
    so a repeat count is not what makes its result trustworthy. A deliberately generous scan of the code (comments
    aside): any mention of a random module, seed argument, sampler or graph generator counts, so "none" is only said
    when nothing could draw. It follows the script's own imports into sibling files (a multi-module project keeps its
    sampler elsewhere) but not the whole folder, which also holds the analysis script and FI's own helpers.
    Over-detecting only brings back the old warning, or holds a ``run_cell`` to the protocol's run count. Missing a
    source is not the only guard there: a ``run_cell`` this finds none in is also called a second time on one setting
    (:meth:`Engine._run_cell_randomness`). An unreadable or unparsable file returns ``True`` because silence is not
    evidence of determinism.
    """
    return bool(_random_source_found(code_path))


def _random_source_found(code_path: Path) -> str:
    """What :func:`_script_has_random_source` found, in words a person can act on (``"`seed` in simulate.py"``), or
    ``""`` when it found nothing."""
    folder = code_path.parent
    seen: set[Path] = set()
    todo = [code_path]
    path = code_path
    try:
        while todo and len(seen) < 100:
            path = todo.pop()
            if path in seen:
                continue
            seen.add(path)
            text = path.read_text(encoding="utf-8", errors="replace")
            hit = _RANDOM_SOURCE_PATTERN.search(_without_comments(text))
            if hit is not None:
                return f"`{hit.group(0).strip('.( ')}` in {path.name}"
            try:
                tree = ast.parse(text)
            except (SyntaxError, ValueError):
                return f"{path.name} could not be read as Python to check it"
            for name in _imported_module_names(tree):
                stem = folder.joinpath(*name.split("."))
                for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                    if candidate.is_file():
                        todo.append(candidate)
        return ""
    except OSError:
        return f"{path.name} could not be read to check it"


def _own_modules(code_path: Path) -> list[Path]:
    """The script and the modules it imports from its own folder, followed as ``_script_has_random_source`` does
    (not the whole folder, which also holds FI's own helpers such as ``run.py``). An unparsable file ends the walk
    there."""
    folder = code_path.parent
    seen: list[Path] = []
    todo = [code_path]
    while todo and len(seen) < 100:
        path = todo.pop()
        if path in seen or not path.is_file():
            continue
        seen.append(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        stems = [folder.joinpath(*name.split(".")) for name in _imported_module_names(tree)]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                # A relative import inside a package of the project: resolved from the importing file's own folder.
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                parts = node.module.split(".") if node.module else []
                stems.append(base.joinpath(*parts) if parts else base)
                stems += [base.joinpath(*parts, a.name) for a in node.names]
        root = folder.resolve()
        for stem in stems:
            todo += [c for c in (stem.with_suffix(".py"), stem / "__init__.py")
                     if c.is_file() and root in c.resolve().parents]  # never outside code/
    return seen


# Calls that seed a random generator (the name of what is called, its last part).
_SEEDING_CALLS = frozenset({"seed", "manual_seed", "default_rng", "RandomState", "Random", "SeedSequence",
                            "PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64", "PRNGKey", "key"})
# Keyword arguments that carry a seed (``Random(x=)``, ``random.seed(a=)``, ``SeedSequence(entropy=)``, scikit-learn and
# scipy's ``random_state=``).
_SEED_KEYWORDS = frozenset({"seed", "x", "a", "entropy", "random_state"})


def _constant_expr(node: ast.AST | None) -> bool:
    """A literal number, or arithmetic on literals only (``42``, ``-1``, ``2**31 - 1``)."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return _constant_expr(node.operand)
    if isinstance(node, ast.BinOp):
        return _constant_expr(node.left) and _constant_expr(node.right)
    return False


def _seeds_only_constants(paths: list[Path]) -> bool:
    """Whether every call in these files that seeds a random generator is given a literal number (``Random(42)``,
    ``np.random.seed(0)``), so the seed FI gives the run can reach none of them. A seed it cannot tell (a variable, a
    function's argument, a value worked out) is not a constant: only a certainly fixed seed counts. No seeding call at
    all is not "only constants" either."""
    seeded = 0
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            # A seed handed to anything by keyword (``random_state=seed`` for scikit-learn or scipy, ``entropy=``): not
            # a constant, so not "only constants". ``x=`` and ``a=`` are seeds only for a seeding call
            # (``random.seed(a=s)``, ``Random(x=s)``; ``plt.axvline(x=mu)`` is not one).
            carriers = _SEED_KEYWORDS if name in _SEEDING_CALLS else _SEED_KEYWORDS - {"x", "a"}
            if any(k.arg in carriers and not _constant_expr(k.value) for k in node.keywords):
                return False
            if name not in _SEEDING_CALLS:
                continue
            arg = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in _SEED_KEYWORDS), None)
            if arg is None:
                continue  # built without a seed: draws from the operating system (handled apart)
            if not _constant_expr(arg):
                return False
            seeded += 1
    return seeded > 0


# Generators that draw from OS entropy when they are handed no seed.
# ``np.random.Generator`` is deliberately absent: it takes a bit generator, not
# a seed, and is always constructed from one that was seeded or was not.
_UNSEEDED_RNG_FACTORIES = frozenset({
    "default_rng",      # numpy: np.random.default_rng()
    "RandomState",      # numpy legacy: np.random.RandomState()
    "Random",           # stdlib: random.Random()
    "SeedSequence",     # numpy: np.random.SeedSequence()
    "PCG64", "MT19937", "Philox", "SFC64",  # numpy bit generators
})


def _rng_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """The names, in this script, that reach a random module or one of the
    factories above: ``({"np"}, {"default_rng"})`` for a file that wrote
    ``import numpy as np`` and ``from numpy.random import default_rng``.

    Read from the imports rather than guessed, so a script's own ``Random``
    class or ``default_rng`` helper is not mistaken for numpy's, and an alias
    nobody would guess (``import numpy.random as nr``) still resolves.
    """
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in {"numpy", "random"}:
                    modules.add(alias.asname or root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] not in {"numpy", "random"}:
                continue
            for alias in node.names:
                if alias.name in _UNSEEDED_RNG_FACTORIES:
                    names.add(alias.asname or alias.name)
                elif alias.name == "random":  # from numpy import random as nr
                    modules.add(alias.asname or alias.name)
    return modules, names


def _guarded_absent(test: ast.AST, *, absent: bool = True) -> set[str]:
    """The names a branch says it does not have.

    With ``absent=True``, the names the ``if`` body runs without: ``rng`` in
    ``if rng is None``, ``if not rng``, ``if rng is None or force``. With
    ``absent=False``, the ones its ``else`` runs without, which is the same
    guard written the other way round: ``if seed is not None: ... else:``.
    """
    names: set[str] = set()
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.left, ast.Name):
        other = test.comparators[0]
        if isinstance(other, ast.Constant) and other.value is None:
            if isinstance(test.ops[0], ast.Is if absent else ast.IsNot):
                names.add(test.left.id)
    elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and absent:
        if isinstance(test.operand, ast.Name):
            names.add(test.operand.id)
    elif isinstance(test, ast.Name) and not absent:
        names.add(test.id)
    elif isinstance(test, ast.BoolOp):
        for value in test.values:
            names |= _guarded_absent(value, absent=absent)
    return names


class _UnseededRngFinder(ast.NodeVisitor):
    """Collects the generators a script builds without a seed, skipping the
    ones that stand in a ``if <param> is None:`` fallback branch.

    That branch is the idiom for "my caller hands me a seeded generator"; in
    every archived quest that writes it, every call site does hand one, and
    asking a model to rewrite a reproducible script is the expensive mistake.
    Three blind spots come with it, all deliberate: a fallback that IS taken,
    because some call site omitted the argument, reads as seeded here; the
    branch is skipped wholesale, so an unrelated generator built inside one is
    skipped with it; and a lambda's parameters are not tracked, so a generator
    a lambda falls back to is reported. None has occurred in the stored quests,
    and the first two would need the call sites to settle.
    """

    def __init__(self, modules: set[str], names: set[str]) -> None:
        self.found: list[tuple[int, str]] = []
        self._modules = modules
        self._names = names
        self._params: list[set[str]] = [set()]
        self._guarded: list[set[str]] = [set()]

    def _visit_function(self, node: Any) -> None:
        args = node.args
        params = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                params.add(extra.arg)
        self._params.append(params)
        self._guarded.append(set())
        self.generic_visit(node)
        self._params.pop()
        self._guarded.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_If(self, node: ast.If) -> None:
        outer = set(self._guarded[-1])
        for branch, absent in ((node.body, True), (node.orelse, False)):
            self._guarded[-1] = outer | (
                _guarded_absent(node.test, absent=absent) & self._params[-1]
            )
            for child in branch:
                self.visit(child)
        self._guarded[-1] = outer
        self.visit(node.test)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute):
            dotted = _dotted_name(func)
            if not dotted or dotted.split(".")[0] not in self._modules:
                return
            if func.attr in _UNSEEDED_RNG_FACTORIES:
                kind = "factory"
            elif func.attr == "seed":  # np.random.seed() reseeds from entropy
                kind = "reseed"
            else:
                return
            expr = dotted
        elif isinstance(func, ast.Name) and func.id in self._names:
            kind, expr = "factory", func.id
        else:
            return
        if self._guarded[-1]:
            return
        bare = not node.args and not node.keywords
        if kind == "reseed":
            if bare:
                self.found.append((node.lineno, f"{expr}()"))
        elif not _rng_call_is_seeded(node):
            self.found.append((node.lineno, f"{expr}()" if bare else f"{expr}(None)"))


def _dotted_name(node: ast.AST) -> str:
    """``np.random.default_rng`` for an attribute chain, or "" when its base is
    not a plain name (a call, a subscript, an attribute of a literal)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _rng_call_is_seeded(call: ast.Call) -> bool:
    """Whether a generator was handed a seed. An explicit ``None`` is not one:
    ``default_rng(None)`` draws from entropy exactly as ``default_rng()`` does.
    ``default_rng(**kwargs)`` is unknowable, so it counts as seeded."""
    def _real(value: ast.AST) -> bool:
        return not (isinstance(value, ast.Constant) and value.value is None)

    if call.args:
        return _real(call.args[0])
    for kw in call.keywords:
        if kw.arg in {"seed", "entropy"}:
            return _real(kw.value)
        if kw.arg is None:
            return True
    return False


def unseeded_rng_calls(code: str) -> list[tuple[int, str]]:
    """``(line, expression)`` for every random generator the script builds
    without a seed, sorted by line.

    A generator built from entropy ignores ``FI_REPLICATE_SEED`` however
    faithfully the rest of the script reads it, so the seed the engine hands
    each replicate never reaches the randomness the run measures and a rerun
    cannot land on the same numbers. Seeding numpy's legacy global does not
    reach a Generator, which is the shape this misses most often: one archived
    quest called ``np.random.seed(seed)`` from the environment variable and
    built ``np.random.default_rng()`` ninety lines earlier, so its 300
    Gillespie trajectories per cell were drawn from OS entropy.

    A source scan, like ``_script_reads_replicate_seed``, and for the same
    reason: it is exact for the case that occurs and costs nothing. A file that
    does not parse returns no calls, because a script that is not valid Python
    has a louder problem than its seeding.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    finder = _UnseededRngFinder(*_rng_aliases(tree))
    finder.visit(tree)
    return sorted(set(finder.found))


def _unseeded_rng_calls(code_path: Path) -> list[tuple[int, str]]:
    """``unseeded_rng_calls`` for a file. An unreadable one reports nothing:
    silence is not evidence of a fault, and a repair costs a model call."""
    try:
        return unseeded_rng_calls(code_path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []


def _is_os_environ(node: ast.AST) -> bool:
    """``os.environ`` (however imported) or a bare ``environ`` from ``from os import environ``."""
    return _dotted_name(node) == "os.environ" or (isinstance(node, ast.Name) and node.id == "environ")


def _is_replicate_seed_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "FI_REPLICATE_SEED"


def _reads_replicate_env(expr: ast.AST) -> bool:
    """Whether this expression subtree reads ``FI_REPLICATE_SEED`` straight out of the environment:
    ``os.environ["FI_REPLICATE_SEED"]``, ``os.environ.get("FI_REPLICATE_SEED", ...)``,
    ``os.getenv("FI_REPLICATE_SEED", ...)`` (however ``os`` or ``getenv`` were imported), under any wrapping
    (``int(...)``, ``str(...)``, a default via ``or``)."""
    for node in ast.walk(expr):
        if isinstance(node, ast.Subscript) and _is_os_environ(node.value):
            key = node.slice.value if isinstance(node.slice, ast.Index) else node.slice  # py3.8 ast.Index compat
            if _is_replicate_seed_literal(key):
                return True
        elif isinstance(node, ast.Call):
            func = node.func
            is_environ_get = isinstance(func, ast.Attribute) and func.attr == "get" and _is_os_environ(func.value)
            is_getenv = (isinstance(func, ast.Attribute) and func.attr == "getenv" and _dotted_name(func.value) == "os") or (
                isinstance(func, ast.Name) and func.id == "getenv"
            )
            if (is_environ_get or is_getenv) and node.args and _is_replicate_seed_literal(node.args[0]):
                return True
    return False


def _mark_seeded_targets(target: ast.AST, value: ast.AST, seeded: set[str]) -> None:
    """Add every plain name ``target`` binds a value that reads ``FI_REPLICATE_SEED`` to. A tuple/list target
    (``seed, label = os.environ.get(...), "run"``) is matched element-by-element against a tuple/list value of the
    same length, so only the elements that actually read the environment are marked -- not every name the
    assignment happens to bind. Anything less literal on the value side (unpacking a call's return value, say)
    cannot be matched this way and is silently skipped, the same "cannot confirm, so do not grant" stance as
    everywhere else in this pair of checks."""
    if isinstance(target, ast.Name):
        if _reads_replicate_env(value):
            seeded.add(target.id)
    elif (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
    ):
        for t_el, v_el in zip(target.elts, value.elts):
            _mark_seeded_targets(t_el, v_el, seeded)


def _names_seeded_from_replicate_env(tree: ast.Module) -> set[str]:
    """Every variable name whose value comes straight from reading ``FI_REPLICATE_SEED``, so a later reference to
    that name can be recognised as carrying the seed even through a plain reassignment (``seed = ...; s = seed``
    is not followed further — one hop is what every script that reads the seed correctly needs)."""
    seeded: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign) and node.value is not None:
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for t in targets:
            _mark_seeded_targets(t, value, seeded)
    return seeded


def replicate_seed_reaches_rng(code: str) -> bool:
    """Whether the value ``FI_REPLICATE_SEED`` names actually flows into EVERY call this file makes that seeds a
    random generator -- not merely whether the string ``FI_REPLICATE_SEED`` appears in the file at all, and not
    merely whether it reaches at least one of several.

    ``_script_reads_replicate_seed`` is a plain substring search, and its own docstring names the blind spot this
    closes: a script that reads the variable into an unused name, or reads it and seeds a generator from an
    unrelated hardcoded constant instead, passes that search and is then indistinguishable from a script that
    genuinely computes the same answer every time -- the runtime comparison (two seeds, identical output) can only
    call both "deterministic". This walks every call this file's ``_rng_aliases``/``_UNSEEDED_RNG_FACTORIES`` would
    recognise as seeding a generator (a factory call's first positional argument, or its ``seed=``/``entropy=``
    keyword; a bare ``.seed(x)`` reseed, or an unseeded reseed/factory call, which cannot connect by construction)
    and asks whether that seed expression itself reads the environment variable directly, or names a variable this
    file assigned straight from it, ONE hop back (an assignment inside a helper function the seed reaches only as a
    parameter is not followed, and such a call counts as not connecting).

    Every recognised call must connect, not just one: a script that correctly seeds one generator from
    ``FI_REPLICATE_SEED`` and a SECOND from a hardcoded constant, using the second for the randomness that actually
    matters, would pass an "any one connects" version of this check while its real result stays disconnected from
    the seed. Requiring all of them is the conservative direction to be wrong in -- a legitimate script whose seed
    reaches its generator through something this cannot trace (the indirect-parameter case above; two independent
    generators for two genuinely unrelated purposes) is treated as "does not reach", which downstream costs an
    error bar it earned, not a false claim of determinism it did not.

    A script with NO such call at all (a purely numerical method -- an ODE solve, say -- that reads the seed but
    has nothing to seed) reports ``True``: there is no randomness for the value to be disconnected from, so this
    check has nothing to say and the runtime comparison's old "it reads the seed" reasoning still applies. A file
    that does not parse also reports ``True`` -- silence is not evidence of the fault, matching
    ``unseeded_rng_calls``'s own stance on both.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return True
    seeded_names = _names_seeded_from_replicate_env(tree)
    modules, names = _rng_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            dotted = _dotted_name(func)
            if not dotted or dotted.split(".")[0] not in modules:
                continue
            if func.attr not in _UNSEEDED_RNG_FACTORIES and func.attr != "seed":
                continue
        elif isinstance(func, ast.Name) and func.id in names:
            pass
        else:
            continue
        seed_arg = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg in ("seed", "entropy")), None,
        )
        connects = seed_arg is not None and (
            _reads_replicate_env(seed_arg)
            or any(isinstance(n, ast.Name) and n.id in seeded_names for n in ast.walk(seed_arg))
        )
        if not connects:
            return False
    return True  # every call seen connected (or there were none to see)


def _replicate_seed_reaches_rng(code_path: Path) -> bool:
    """``replicate_seed_reaches_rng`` for a file. An unreadable one gets the benefit of the doubt for the same
    reason ``_script_reads_replicate_seed`` does: silence is not evidence of a fault."""
    try:
        return replicate_seed_reaches_rng(code_path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return True


def _unseeded_rng_directive(calls: list[tuple[int, str]]) -> str:
    """The ``execute_reflect`` directive for a script that reads
    ``FI_REPLICATE_SEED`` and then builds a generator that cannot hear it."""
    where = "; ".join(f"line {line}: {expr}" for line, expr in calls[:6])
    return (
        "This script has NOT been run, and it has not failed: the account of a crash "
        "above does not apply. It needs one change before it is run, and no other.\n\n"
        "The engine runs this script several times, each run handed its own integer "
        "in the environment variable FI_REPLICATE_SEED, so that the spread between "
        "the runs can be reported and any one of them can be reproduced. This script "
        "does read that variable, but it also builds a random generator without a "
        f"seed, which draws from the operating system's entropy instead: {where}. "
        "Nothing that generator produces can be reproduced, and the seed handed to "
        "the run never reaches it. Note that seeding numpy's legacy global "
        "(np.random.seed) does NOT seed a Generator made by np.random.default_rng: "
        "they are separate streams.\n\n"
        "Change exactly this: derive the seed of EVERY random generator the script "
        "creates from the integer in FI_REPLICATE_SEED (default 0 when it is unset), "
        "including the ones listed above. Where a generator is created per call, per "
        "trial or inside a loop, hand it that integer plus the trial index (or pass "
        "one generator in from the caller) so no two draws repeat and the run stays "
        "reproducible. Keep everything else in the script unchanged: the same "
        "functions, parameters, outputs and figures, and the same final RESULT_JSON "
        "line. Do not add randomness the experiment does not already have, and do not "
        "shorten or simplify anything.\n\n"
        "Return only the functions you change in `functions` (each one whole; FI keeps every other line), one sentence in `patch_summary`, and "
        "leave `give_up_reason` empty."
    )


# Stands where the traceback would be in the ``execute_reflect`` prompt, for the
# one repair a script that ignores ``FI_REPLICATE_SEED`` is offered before it
# has run (see ``Engine._repair_ignored_replicate_seed``).
def _stripped(protocol: dict[str, Any]) -> dict[str, Any]:
    copy_ = copy.deepcopy(protocol)
    _strip_plan_numbers(copy_)
    return copy_


def _strip_plan_numbers(protocol: dict[str, Any], *, names_only: bool = False) -> None:
    """In place: the plan's own numbers for what the code must reproduce, removed from a protocol that is shown to a
    request that writes or repairs code (see :meth:`Engine._code_design_block`)."""
    model = protocol.get("model")
    for eq in (model.get("equations") if isinstance(model, dict) and isinstance(model.get("equations"), list) else []):
        if isinstance(eq, dict) and "example" in eq:
            example = eq.pop("example")
            inputs = example.get("inputs") if isinstance(example, dict) else None
            if names_only and isinstance(inputs, dict):
                eq["example"] = {"input_names": list(inputs), **({"random": True} if _equation_tests.is_random(
                    {**eq, "example": example}) else {})}
    # The value each check compares with, and how close it must be, are not shown: not as the check's fields, not as a
    # criterion's target built on a check, and not in any prose of the plan (written as words there, so the prose keeps
    # its meaning). A number the code is given as an input (a grid value, a threshold, a check's setting) stays: the
    # code cannot run without it, even where it is also what a check recovers.
    oracles = [o for o in (protocol.get("oracles") if isinstance(protocol.get("oracles"), list) else [])
               if isinstance(o, dict)]
    criteria = [c for c in (protocol.get("criteria") if isinstance(protocol.get("criteria"), list) else [])
                if isinstance(c, dict) and c.get("oracle")]
    hidden = [o.get(k) for o in oracles for k in ("expected", "tolerance", "limit")]
    hidden += [c.get(k) for c in criteria for k in ("target", "tolerance")]
    inputs = _input_numbers(protocol)
    hidden = [v for v in hidden if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) not in inputs]
    for oracle in oracles:
        for key in _ORACLE_HIDDEN:
            oracle.pop(key, None)
    for criterion in criteria:
        criterion.pop("target", None)
        criterion.pop("tolerance", None)
    _mask_prose(protocol, hidden)


#: The fields of a check a request that writes or repairs code never sees.
_ORACLE_HIDDEN = ("expected", "expected_formula", "tolerance", "tolerance_mode", "limit")


#: The fields of a protocol that are settings the code runs with: a number written in them, also as text ("q = 0.01"),
#: is an input and is never masked (with each check's ``case``).
_SETTING_FIELDS = ("grid", "fixed", "in_each_run")


def _input_numbers(protocol: dict[str, Any]) -> set[float]:
    """The numbers the plan gives the code to run with: every number of the protocol outside its checks, its criteria
    built on a check and its model's worked examples; and every number, also one written as text, of its settings
    (:data:`_SETTING_FIELDS`) and of each check's setting (``case``)."""
    def walk(node: Any, text: bool = False) -> set[float]:
        if isinstance(node, dict):
            return set().union(*(walk(v, text) for v in node.values())) if node else set()
        if isinstance(node, list):
            return set().union(*(walk(v, text) for v in node)) if node else set()
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            return {float(node)}
        if text and isinstance(node, str):
            return {_prose_value(m) for m in _PROSE_NUMBER.finditer(node)} - {None}
        return set()

    found = walk({k: v for k, v in protocol.items() if k not in ("oracles", "criteria", "model")})
    found |= walk({k: protocol.get(k) for k in _SETTING_FIELDS if protocol.get(k) is not None}, text=True)
    for o in protocol.get("oracles") if isinstance(protocol.get("oracles"), list) else []:
        if isinstance(o, dict):
            found |= walk(o.get("case") or {}, text=True)
    return found


def _prose_value(m: re.Match[str]) -> float | None:
    """The number a match of :data:`_PROSE_NUMBER` writes (a percentage as the fraction it states)."""
    try:
        x = float(m.group(0).rstrip("%"))
    except ValueError:
        return None
    return x / 100 if m.group(1) else x


def _mask_prose(node: Any, hidden: list[Any]) -> Any:
    """In place: every string under ``node`` with the ``hidden`` numbers written as words (:func:`_without_numbers`)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str):
                node[k] = _without_numbers(v, hidden)
            else:
                _mask_prose(v, hidden)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str):
                node[i] = _without_numbers(v, hidden)
            else:
                _mask_prose(v, hidden)
    return node
#: A number in prose: not part of a word or a longer number; a full stop after it ends the sentence ("is 0.6415.").
_PROSE_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(%?)(?!\w|\.\d)")


def _without_numbers(text: str, hidden: list[Any]) -> str:
    """``text`` with every number that is one of ``hidden`` (a check's expected value or tolerance, also written as a
    percentage) put as words: what is checked stays readable, the number it is held to does not."""
    # 0 and 1 stay: every setting is written with them ("sigma=0"), and a check held to 0 says so in its words anyway.
    values = [float(v) for v in hidden if isinstance(v, (int, float)) and not isinstance(v, bool) and v not in (0, 1)]
    if not values:
        return text

    def swap(m: re.Match[str]) -> str:
        x = _prose_value(m)
        if x is None:
            return m.group(0)
        if any(math.isclose(x, v, rel_tol=1e-9, abs_tol=1e-15) for v in values):
            return "[the value FI checks it against]"
        return m.group(0)

    return _PROSE_NUMBER.sub(swap, text)


def _protocol_directive(protocol: dict[str, Any], mismatches: list[Any], frame: bool = False) -> str:
    """What stands where a traceback would in the repair request: the protocol the plan fixed and how the script differs."""
    return (
        "This script has NOT been run, and it has not failed: the account of a crash above does not apply. It "
        "contradicts the experiment protocol that was fixed in the plan, and needs the changes below before it is "
        "run, and no other.\n\n"
        "The protocol:\n" + json.dumps(_stripped(protocol), indent=2) + "\n\n"
        "How the script differs from it:\n" + "\n".join(f"- {m.message()}" for m in mismatches) + "\n\n"
        "Change exactly this: make the script use the protocol's values, every one of them, and none the protocol does "
        "not name. Keep everything else unchanged: the same functions, outputs and figures, "
        + ("and the frame FI wrote" if frame else "the handling of FI_PILOT and FI_REPLICATE_SEED, and the same final RESULT_JSON line")
        + ". Do not shorten or simplify anything. If the "
        "timeout makes the full protocol impossible, keep the protocol's values anyway and say so in `patch_summary`: "
        "the person reading it decides, not this script.\n\n"
        "Return only the functions you change in `functions` (each one whole; FI keeps every other line), one sentence in `patch_summary`, and leave `give_up_reason` empty."
    )


_SEED_REPAIR_DIRECTIVE = (
    "This script has NOT been run, and it has not failed: the account of a crash "
    "above does not apply. It needs one change before it is run, and no other.\n\n"
    "The engine runs this script several times, each run handed its own integer "
    "in the environment variable FI_REPLICATE_SEED, so that the spread between "
    "the runs can be reported. This script never reads that variable, so every "
    "run would repeat one measurement and no mean or confidence interval could "
    "be reported over them.\n\n"
    "Change exactly this: read the integer from the environment variable "
    "FI_REPLICATE_SEED (default 0 when it is unset) and derive the seed of "
    "every random generator the script creates from it (random.seed, "
    "np.random.seed, np.random.default_rng, torch.manual_seed, ...), replacing "
    "any seed constant the script wrote for itself. Where it derives one seed "
    "per trial, derive it as that integer plus the trial index. Keep everything "
    "else in the script unchanged: the same functions, parameters, outputs and "
    "figures, and the same final RESULT_JSON line. Do not add randomness the "
    "experiment does not already have, and do not shorten or simplify anything.\n\n"
    "Return only the functions you change in `functions` (each one whole; FI keeps every other line), one sentence in `patch_summary`, and "
    "leave `give_up_reason` empty."
)


def _replicate_assertions(state: QuestState) -> list[Any]:
    """The design's and the selected skills' ``result_assertions``, parsed, to
    bound the replicate confidence intervals. Never raises: a bound only
    refines an interval."""
    try:
        from core import plausibility

        design = state.get("design")
        declared = list(design.get("result_assertions") or []) if isinstance(design, dict) else []
        return plausibility.parse_assertions(
            {"result_assertions": declared + list(state.get("_skill_assertions") or [])},
        )
    except Exception:  # noqa: BLE001
        return []


def _ci_bounds(path: str, vals: list[float], assertions: list[Any]) -> dict[str, float]:
    """The values a metric's confidence interval must stay within: the range
    the first ``result_assertions`` entry for its path declares, and [0, 1]
    for a probability or proportion whose every seed lies there."""
    from core.plausibility import _matches

    bounds: dict[str, float] = {}
    declared = next((a for a in assertions if _matches(a.path, path)), None)
    if declared is not None:
        if declared.min is not None:
            bounds["lower"] = declared.min
        if declared.max is not None:
            bounds["upper"] = declared.max
    if _PROPORTION_NAME_RE.search(path.rsplit(".", 1)[-1]) and vals and all(0.0 <= v <= 1.0 for v in vals):
        bounds.setdefault("lower", 0.0)
        bounds.setdefault("upper", 1.0)
    return bounds


def _aggregate_result_json_replicates(
    replicates: list[dict[str, Any]],
    *,
    assertions: list[Any] | None = None,
    kinds: dict[str, str] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Compute mean ± sample-std for every numeric scalar field that
    appears in EVERY replicate's ``RESULT_JSON``.

    Each entry in ``replicates`` is one ``RESULT_JSON`` dict augmented
    with a synthetic ``_seed`` field. We skip the ``_seed`` key, scan
    the union of remaining keys, and emit an aggregate only for keys
    whose values are scalar floats/ints in ALL replicates — mixed-type
    keys (strings, lists) are skipped silently so the analyze
    LLM can still reason about them from the raw per-seed JSON.

    **Nested results are flattened to dotted paths.** Scanning only the top
    level was a silent no-op for any experiment that groups its results, which
    is the normal shape for a parameter sweep — a real quest reported
    ``0 numeric keys`` from three replicates because every top-level value was
    a dict (``by_h``, ``by_integrator``). Now
    ``by_h["0.5"]["RK4"]["trajectory_error"]`` aggregates under
    ``by_h.0.5.RK4.trajectory_error``. Path segments containing dots stay
    as-is, so a literal ``{"a.b": 1}`` and a nested ``{"a": {"b": 1}}`` would
    collide; that is accepted because the output is read, not indexed.

    Returns ``{path: {"mean": float, "std": float, "n": int, "min": float, "max": float}}``.
    Empty input → empty dict. n=1 (single replicate) → emits min=max=value,
    std=0.0.
    """
    if not replicates:
        return {}

    def _flat(obj: Any, prefix: str = "") -> dict[str, float]:
        """Numeric leaves of a nested mapping, keyed by dotted path."""
        found: dict[str, float] = {}
        if not isinstance(obj, dict):
            return found
        for k, v in obj.items():
            if not prefix and k == "_seed":
                continue
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                found.update(_flat(v, path))
            elif not isinstance(v, bool) and isinstance(v, (int, float)):
                found[path] = float(v)
        return found

    flattened = [_flat(r) for r in replicates]

    # Union of all paths; a path must be numeric in EVERY replicate to be
    # aggregated, so a key that appears in only some seeds is skipped.
    all_keys: set[str] = set()
    for f in flattened:
        all_keys.update(f)

    out: dict[str, dict[str, float | int]] = {}
    for key in sorted(all_keys):
        vals: list[float] = []
        all_scalar_numeric = True
        for f in flattened:
            if key not in f:
                all_scalar_numeric = False
                break
            vals.append(f[key])
        if not all_scalar_numeric or not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            # Sample std (n-1 denominator) — the usual reported quantity.
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        out[key] = {
            "mean": mean,
            "std": std,
            "n": n,
            "min": min(vals),
            "max": max(vals),
            # Standard error + 95% CI of the mean (None for n<2 — a single
            # seed has no spread to estimate, so we don't fake an interval),
            # kept within the metric's declared or proportion bounds.
            **_stats.confidence_interval(vals, **_ci_bounds(key, vals, assertions or [])),
            # What the interval above is an interval OF: the spread of this quantity across the seeds. With a few
            # seeds that is a statement about batches, not about the number of trials behind each of them.
            "ci_method": _CI_BETWEEN_SEEDS,
        }
    _pool_evidence(out, replicates, kinds=kinds)
    return out


def _replicate_metric_kinds(state: "QuestState") -> dict[str, str]:
    """``{metric id: kind}`` for the metrics the design's protocol declares (``proportion`` / ``mean``)."""
    protocol = (state.get("design") or {}).get("protocol")
    metrics = protocol.get("metrics") if isinstance(protocol, dict) else None
    return {
        str(m["id"]): str(m.get("kind"))
        for m in metrics or [] if isinstance(m, dict) and m.get("id") and m.get("kind")
    }


# What a newly written script starts from. The repair loop's counters belong to one script: carried into the script a
# redesign wrote, they read as spent, and in two live quests the new script's first crash logged "iterations
# exhausted" with no repair at all, so both papers were written with no results. Redesigns stay bounded by
# engine.max_iterations, and each script's own repairs by engine.exec_reflect_max_iterations.
_FRESH_SCRIPT: dict[str, Any] = {
    "exec_reflect_iter": 0, "gate_repairs": 0, "exec_reflect_history": [], "exec_give_up_reason": "", "exec_patch_pending": False,
    "run_manifest_failures": 0, "run_manifest_analysis_only": False, "figure_data_repairs": 0, "figure_overlap_repaired": False, "figure_labels_redrawn": False, "bounded_seen": [],
}


_CI_BETWEEN_SEEDS = "t_between_seeds"
_CI_WILSON = "wilson_pooled_counts"
_CI_BOOTSTRAP = "bootstrap_pooled_values"
_VALUES_POOL_CAP = 20000


def _walk_paths(obj: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    """The numeric scalars and the lists of numbers in a nested results mapping, keyed by their path."""
    found: dict[tuple[str, ...], Any] = {}
    if not isinstance(obj, dict):
        return found
    for k, v in obj.items():
        if not prefix and k == "_seed":
            continue
        path = prefix + (str(k),)
        if isinstance(v, dict):
            found.update(_walk_paths(v, path))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            found[path] = float(v)
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            found[path] = [float(x) for x in v]
    return found


def _is_metric_key(key: str, name: str, declared: set[str] | frozenset[str] = frozenset()) -> bool:
    """Whether a pooled entry's key is the metric ``name``: the name itself, or the end of a path to it (``by_x.0.9.name``;
    a name with a dot in it included), unless a longer declared metric ends the key (``by_x.risk.rate`` is ``risk.rate``,
    not ``rate``)."""
    def ends(d: str) -> bool:
        return key == d or key.endswith("." + d)

    return ends(name) and not any(len(d) > len(name) and ends(d) for d in declared)


def _declared_metrics(protocol: dict[str, Any]) -> set[str]:
    """The ids of the metrics the plan declares."""
    return {str(m.get("id")) for m in protocol.get("metrics") or [] if isinstance(m, dict) and m.get("id")}


def _precision_target_on(protocol: dict[str, Any]) -> str:
    """The metric the plan's precision target is on, when it is one the plan declares (``precision.metric``, or the plan's
    only metric); ``""`` when the target names none of them (prose, say), and every probability is held to it."""
    from . import run_estimate as _run_estimate

    on, _kind = _run_estimate.precision_metric(protocol)
    return str(on) if on and str(on) in _declared_metrics(protocol) else ""


def _pool_evidence(
    out: dict[str, dict[str, float | int]], replicates: list[dict[str, Any]], *, kinds: dict[str, str] | None = None,
) -> None:
    """Replace the between-seed interval of a metric by one that uses the trials, when the script gave them.

    A probability that a script estimates from repeated trials is reported with its counts beside it,
    ``<name>``, ``<name>_count`` and ``<name>_total``, and a mean over the trials that met a condition with the
    observations behind it, ``<name>_values``. The engine ran the script once per seed, so each seed holds a share of
    the trials: the shares are pooled, and the interval is a Wilson interval for the pooled counts, or a bootstrap
    interval for the pooled values, instead of a t interval over the seeds' own estimates (three numbers, however many
    trials there were). The mean becomes the pooled estimate, and the seeds' own mean and spread are kept as
    ``batch_mean`` and ``batch_std``. A metric with no counts or values keeps the t interval, labelled as such.

    Which of the two a metric is comes from the protocol (``kinds``, its declared ``kind``) and only for an undeclared
    one from the names. A live quest's conditional mean (final size over major outbreaks) printed ``_count`` and
    ``_total`` both equal to the number of outbreaks, beside its ``_values``: read by the names it became a probability
    of 1.0 with a Wilson interval, and the paper printed a half-width of ±0.245 for a mean whose values varied by a
    hundredth. A declared ``mean`` is never pooled as counts; an undeclared one whose count equals its total in every
    seed while it also lists its values is a mean too."""
    kinds = kinds or {}
    walked = [_walk_paths(r) for r in replicates]
    if not walked:
        return
    for path, first in walked[0].items():
        key = ".".join(path)
        entry = out.get(key)
        if entry is None or not isinstance(first, float):
            continue
        parent, name = path[:-1], path[-1]
        count_path, total_path = parent + (f"{name}_count",), parent + (f"{name}_total",)
        values_path = parent + (f"{name}_values",)
        counts = [(w.get(count_path), w.get(total_path)) for w in walked]
        lists = [w.get(values_path) for w in walked]
        kind = kinds.get(name)
        # Undeclared, a count equal to its total beside values that are not all 0/1 indicators is a mean's own
        # denominator; a proportion of 1.0 printed with its 0/1 indicators is still a proportion.
        a_mean_by_its_values = (
            all(isinstance(v, list) for v in lists)
            and all(c == t for c, t in counts)
            and any(x not in (0.0, 1.0) for v in lists for x in v)
        )
        as_counts = all(isinstance(c, float) and isinstance(t, float) for c, t in counts) and (
            kind == "proportion" or (kind != "mean" and not a_mean_by_its_values)
        )
        if as_counts:
            if all(c.is_integer() and t.is_integer() and 0 <= c <= t for c, t in counts):
                k, n = sum(c for c, _t in counts), sum(t for _c, t in counts)
                ci = _stats.wilson_interval(k, n)
                if ci is not None and n > 0:
                    p_hat = k / n
                    entry.update({
                        "batch_mean": entry["mean"], "batch_std": entry["std"],
                        "mean": p_hat, "se": math.sqrt(p_hat * (1.0 - p_hat) / n),
                        "ci_lower": ci[0], "ci_upper": ci[1],
                        "n_trials": int(n), "n_successes": int(k), "ci_method": _CI_WILSON,
                    })
                    # The counts now live in the probability's own entry; a between-seed interval of the counts
                    # themselves would only read as a second, wrong, uncertainty.
                    out.pop(".".join(count_path), None)
                    out.pop(".".join(total_path), None)
            continue
        if all(isinstance(v, list) for v in lists):
            pooled = [x for v in lists for x in v]
            ci = _stats.bootstrap_mean_interval(pooled, cap=_VALUES_POOL_CAP)
            if ci is not None:
                entry.update({
                    "batch_mean": entry["mean"], "batch_std": entry["std"],
                    "mean": sum(pooled) / len(pooled),
                    "ci_lower": ci[0], "ci_upper": ci[1],
                    "n_values": len(pooled), "ci_method": _CI_BOOTSTRAP,
                })


def _elide_value_lists(obj: Any) -> Any:
    """``obj`` with each long ``<name>_values`` list replaced by a note: the observations are pooled into the
    aggregate's interval, and thousands of numbers per seed would crowd the analysis prompt."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if str(k).endswith("_values") and isinstance(v, list) and len(v) > 20:
                out[k] = f"<{len(v)} values, pooled in aggregate_mean_std>"
            else:
                out[k] = _elide_value_lists(v)
        return out
    if isinstance(obj, list):
        return [_elide_value_lists(x) for x in obj]
    return obj


def _series_at(
    replicates: list[dict[str, Any]], path: tuple[str, ...],
) -> list[float] | None:
    """The per-seed values at ``path`` — only when it's a scalar number in
    EVERY replicate (else None: a value some seeds lack can't be compared)."""
    vals: list[float] = []
    for r in replicates:
        node: Any = r
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            return None
        vals.append(float(node))
    return vals


def _replicate_result_intervals(state: QuestState) -> dict[str, dict[str, Any]]:
    """Each result's mean and 95% CI over the seeds, by dotted path, for the
    checks that compare the paper with the results: a paper reports the mean
    and its interval, and seed 0's ``RESULT_JSON`` holds neither. Only the
    results that differ between seeds; the rest are seed 0's values."""
    replicates = state.get("result_json_replicates") or []
    if len(replicates) < 2:
        return {}
    aggregate = _aggregate_result_json_replicates(
        replicates, assertions=_replicate_assertions(state), kinds=_replicate_metric_kinds(state),
    )
    # ``ci_method`` says what the interval is an interval OF (``t_between_seeds`` or a pooled Wilson / bootstrap interval),
    # so a check that asks "is this printed interval FI's own seed-level t interval under another name" only asks it
    # about the ones that are.
    return {
        path: {
            **{key: stats[key] for key in ("mean", "ci_lower", "ci_upper") if stats.get(key) is not None},
            **({"ci_method": stats["ci_method"]} if isinstance(stats.get("ci_method"), str) else {}),
        }
        for path, stats in aggregate.items() if stats.get("std")
    }


# An interval the experiment's own figure named ("Gillespie (Wilson 95% CI)", "mean +/- 1 SD"). A redrawn band is FI's
# 95% interval over the seed means, whatever the seed-0 figure's legend said it was: a real quest's redraw kept "Wilson
# 95% CI" on bands that were the interval over 3 seed means, and the caption repeated it. A method name alone ("a
# binomial model") is left as it is: only one given as an interval (a percent or a CI word beside it) is rewritten.
_INTERVAL_METHOD_RE = re.compile(
    r"(?P<pre>\b\d{2}(?:\.\d+)?\s*%\s*)?"
    r"\b(?:Wilson(?:\s+score)?|Clopper[\s\-\u2013]*Pearson|Agresti[\s\-\u2013]*Coull|Jeffreys|bootstrap(?:ped)?"
    r"|binomial|exact|normal[\s\-\u2013]*approximation|Poisson)\b"
    r"(?P<post>\s*\d{2}(?:\.\d+)?\s*%)?"
    r"(?P<what>\s*(?:CIs?\b|C\.I\.|confidence\s+intervals?|intervals?|bands?|error\s+bars?))?"
    r"(?P<paren>\s*\(\s*\d{2}(?:\.\d+)?\s*%\s*\))?",
    re.IGNORECASE,
)
_SPREAD = (
    r"(?:\u00b1|\+/-|\+-)\s*(?:1\s*)?(?:SD|s\.d\.|std(?:\.|\s+dev(?:iation)?)?|SE|SEM|s\.e\.(?:m\.)?"
    r"|standard\s+(?:deviation|error))\b"
)
_SPREAD_RE = re.compile(r"\s*" + _SPREAD, re.IGNORECASE)
_SPREAD_IN_PARENS_RE = re.compile(r"\(\s*" + _SPREAD + r"\s*\)", re.IGNORECASE)


def _redrawn_band_text(text: Any, n: int) -> Any:
    """``text`` (a legend label, a title, an axis label) with any interval the experiment named replaced by what a
    redrawn band is: the 95% CI over ``n`` seeds."""
    if not isinstance(text, str) or not text:
        return text
    ours = f"95% CI over {n} seeds"

    def method(m: re.Match[str]) -> str:
        if not (m.group("pre") or m.group("post") or m.group("what") or m.group("paren")):
            return m.group(0)
        return ours

    out = _INTERVAL_METHOD_RE.sub(method, text)
    # "(+/- 1 SD)" in parentheses is replaced whole, so no "(, 95% ...)" is left behind.
    out = _SPREAD_IN_PARENS_RE.sub(f"({ours})", out)
    return _SPREAD_RE.sub(f", {ours}", out)


# The line style a redraw keeps (core/replot_figures.py draws with the same keys).
_REPLOT_STYLE_KEYS = (
    "color", "linestyle", "marker", "markersize", "markerfacecolor",
    "linewidth", "alpha", "drawstyle",
)


def _replicate_line_figure(
    name: str, runs: list[Any], assertions: list[Any],
) -> dict[str, Any] | None:
    """What ``code/replot_figures.py`` draws for figure ``name``, from the lines
    the recorder kept at each seed (``runs``): every line's mean and 95% CI at
    each x, bounded as the results are, by the panel's y label. ``None``
    unless every panel of every run holds only lines, the panels match across
    the runs in position and their lines in label, kind and x, and some line
    differs between the seeds."""
    if len(runs) < 2 or not all(isinstance(r, dict) and isinstance(r.get("axes"), list) for r in runs):
        return None
    first = runs[0]["axes"]
    if not first or any(len(r["axes"]) != len(first) for r in runs):
        return None
    panels: list[dict[str, Any]] = []
    varies = False
    for i, panel in enumerate(first):
        same = [r["axes"][i] for r in runs]
        if not all(
            isinstance(p, dict) and p.get("line_only") and isinstance(p.get("lines"), list)
            and isinstance(p.get("grid"), list) and len(p["grid"]) == 6
            and p["grid"] == panel["grid"] and len(p["lines"]) == len(panel["lines"])
            for p in same
        ):
            return None
        bound_name = re.sub(r"[^0-9a-z]+", "_", str(panel.get("ylabel") or "").lower()).strip("_")
        lines = []
        for j, line in enumerate(panel["lines"]):
            at_seeds = [p["lines"][j] for p in same]
            x = line.get("x") if isinstance(line, dict) else None
            if not isinstance(x, list) or not all(
                isinstance(s, dict) and s.get("label") == line.get("label")
                and s.get("kind") == line.get("kind") and s.get("x") == x
                and isinstance(s.get("y"), list) and len(s["y"]) == len(x)
                for s in at_seeds
            ):
                return None
            ys = [s["y"] for s in at_seeds]
            varies = varies or any(y != ys[0] for y in ys[1:])
            mean, lower, upper = [], [], []
            for k in range(len(x)):
                vals = [float(y[k]) for y in ys]
                m = sum(vals) / len(vals)
                ci = _stats.confidence_interval(vals, **_ci_bounds(bound_name, vals, assertions))
                mean.append(m)
                lower.append(m if ci["ci_lower"] is None else ci["ci_lower"])
                upper.append(m if ci["ci_upper"] is None else ci["ci_upper"])
            lines.append({
                **{key: line.get(key) for key in _REPLOT_STYLE_KEYS},
                "label": _redrawn_band_text(line.get("label"), len(runs)), "kind": line.get("kind"),
                "x": x, "mean": mean, "lower": lower, "upper": upper,
            })
        panels.append({
            **{key: panel.get(key) for key in ("grid", "xlabel", "xscale", "yscale", "legend")},
            **{key: _redrawn_band_text(panel.get(key), len(runs)) for key in ("title", "ylabel")},
            "lines": lines,
        })
    if not varies:
        return None
    return {
        "file": name, "size": runs[0].get("size"), "suptitle": _redrawn_band_text(runs[0].get("suptitle") or "", len(runs)),
        "n": len(runs), "axes": panels,
    }


#: The line an exploration's paper carries under its title (what the result is for: explore, or not said).
PRELIMINARY_NOTE = (
    "> **Preliminary result.** This quest was set up to explore, or did not say what its result is for, so what it "
    "found is a first look, not a publication-ready result. Run it again for research (`result_use: research`) to "
    "check it."
)


def _mark_preliminary(markdown: str) -> str:
    """``markdown`` with :data:`PRELIMINARY_NOTE` under its title, once. With no title it goes at the top, after any
    YAML front matter; a ``# comment`` inside a fenced code block is not a title."""
    if PRELIMINARY_NOTE in markdown:
        return markdown
    lines = markdown.split("\n")
    start = 0
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
        if end is not None:
            start = end + 1
    at, fenced = None, False
    for i in range(start, len(lines)):
        if lines[i].lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced and lines[i].startswith("# "):
            at = i + 1
            break
    at = start if at is None else at
    return "\n".join([*lines[:at], "", PRELIMINARY_NOTE, "", *lines[at:]])


def _result_comparison_stats(
    replicates: list[dict[str, Any]], *, max_effect_sizes: int = 24,
    max_depth: int = 6, assertions: list[Any] | None = None, metric_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Per-stratum CIs + pairwise effect sizes (Cohen's d) between the strata
    the experiment broke results down by, plus a multiple-comparison guard.
    Lets the paper report *how big* a between-group difference is and whether
    it survives correction for the number of comparisons — not just its
    direction.

    Strata nest. A crossed design writes ``by_h → step size → method →
    metric``, and the comparison a reader wants — method A against method B
    at one step size — lives BELOW the ``by_*`` key. So every level under a
    ``by_*`` key is compared as well, labelled by its dotted path
    (``by_h.0.5``), with each stratum's numeric leaves as its metrics
    (``RK4.err`` at the step-size level, ``err`` at the method level). A
    nested level counts only when its children share one key structure:
    ``{"errors": {...}, "timing": {...}}`` groups unlike things, and pairing
    them would be noise. Unlike groups can share a key structure too
    (``errors`` and ``timing`` each with ``mean`` and ``max``): a nested level
    whose children are named for the protocol's declared metrics
    (``metric_ids``) groups measures, not settings, and is not compared. The
    ``by_*`` level itself is always a factor.

    Returns ``{"strata": {factor: {stratum: {metric: {mean, ci_lower,
    ci_upper, n}}}}, "effect_sizes": [{factor, metric, a, b, cohens_d,
    magnitude}], "comparisons": {"n", "bonferroni_alpha", "many"}}``. Empty
    when no level has ≥2 numeric strata to compare."""
    if not replicates or not all(isinstance(r, dict) for r in replicates):
        return {}
    strata_out: dict[str, Any] = {}
    effect_sizes: list[dict[str, Any]] = []
    n_comparisons = 0

    def _at(obj: Any, path: tuple[str, ...]) -> Any:
        for key in path:
            obj = obj.get(key) if isinstance(obj, dict) else None
        return obj

    def _shape(obj: Any, depth: int = 0) -> Any:
        # Keys only: a method that emitted null for a metric still has the
        # same shape as one that emitted a number.
        if isinstance(obj, dict) and depth < max_depth:
            return tuple(sorted((k, _shape(v, depth + 1)) for k, v in obj.items()))
        return None

    def _leaf_paths(obj: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
        if not isinstance(obj, dict):
            return [prefix] if prefix else []
        if len(prefix) >= max_depth:
            return []
        return [p for k, v in obj.items() for p in _leaf_paths(v, (*prefix, k))]

    def _visit(path: tuple[str, ...], depth: int) -> None:
        nonlocal n_comparisons
        nodes = [_at(r, path) for r in replicates]
        if not all(isinstance(n, dict) for n in nodes):
            return
        # Strata present — and themselves dicts — in every replicate.
        strata = sorted(
            k for k in set(nodes[0]).intersection(*(set(n) for n in nodes[1:]))
            if all(isinstance(n[k], dict) for n in nodes)
        )
        label = ".".join(path)
        comparable = len(strata) >= 2 and (
            depth == 0 or (
                len({_shape(nodes[0][s]) for s in strata}) == 1
                and not (metric_ids and any(s.lower() in metric_ids for s in strata))
            )
        )
        if comparable:
            metrics = sorted({
                m for s in strata for m in _leaf_paths(nodes[0][s])
                if _series_at(replicates, (*path, s, *m)) is not None
            })
            per_stratum: dict[str, Any] = {}
            for s in strata:
                cell: dict[str, Any] = {}
                for m in metrics:
                    series = _series_at(replicates, (*path, s, *m))
                    if series is None:
                        continue
                    mean, _, n = _stats.mean_std(series)
                    bounds = _ci_bounds(".".join((*path, s, *m)), series, assertions or [])
                    cell[".".join(m)] = {"mean": mean, "n": n,
                                         **_stats.confidence_interval(series, **bounds)}
                if cell:
                    per_stratum[s] = cell
            if per_stratum:
                strata_out[label] = per_stratum
            # Pairwise effect sizes between strata, per shared metric.
            for m in metrics:
                for i in range(len(strata)):
                    for j in range(i + 1, len(strata)):
                        a = _series_at(replicates, (*path, strata[i], *m))
                        b = _series_at(replicates, (*path, strata[j], *m))
                        if a is None or b is None:
                            continue
                        n_comparisons += 1
                        d = _stats.cohens_d(a, b)
                        effect_sizes.append({
                            "factor": label, "metric": ".".join(m),
                            "a": strata[i], "b": strata[j],
                            "cohens_d": d,
                            "magnitude": _stats.effect_magnitude(d),
                        })
        if depth + 1 < max_depth:
            for s in strata:
                _visit((*path, s), depth + 1)

    for factor in sorted(k for k in replicates[0] if str(k).startswith("by_")):
        _visit((factor,), 0)
    if not strata_out and not effect_sizes:
        return {}
    # Surface the largest effects first; cap the list so the analyze prompt
    # isn't flooded on a heavily-stratified run.
    effect_sizes.sort(key=lambda e: abs(e["cohens_d"] or 0.0), reverse=True)
    return {
        "strata": strata_out,
        "effect_sizes": effect_sizes[:max_effect_sizes],
        "comparisons": {
            "n": n_comparisons,
            "bonferroni_alpha": _stats.bonferroni_alpha(n_comparisons),
            "many": n_comparisons > 5,
        },
    }


def _aggregate_panel_reviews(
    panel: list[dict[str, Any]], *, fallback_verdict: str = "accept",
) -> dict[str, Any]:
    """Deterministic aggregator the moderator prompt is also
    instructed to follow. We compute the canonical answer programmatically
    so tests can pin the rules even when the LLM moderator is unavailable
    or produces malformed JSON. The moderator's output, when usable, is
    preferred for prose (`rationale`, `suggestions` attribution) but the
    numeric verdict/score is recomputed here to enforce the rules.

    Rules:
      - Any persona votes `revise` with `score < 3` → final verdict revise.
      - Otherwise majority verdict; ties → revise (conservative).
      - Score = median of panel scores, rounded.
      - Weaknesses = deduped union.
      - Strengths = intersection.
      - Suggestions = deduped union, persona-attributed.
      - Agreement = "unanimous" / "split" / "controversial".

    Only a panelist with a real review votes: an entry whose ``status`` says it
    did not review (``error``, ``call_failed``, ``unreviewed``) is left out of
    every rule above, and ``status`` on the result is ``"ok"`` only when every
    panelist reviewed. Two failed panelists' stand-in accepts once outvoted a real
    revise. An entry with no ``status`` (a checkpoint from before it existed)
    counts as reviewed.
    """
    failed = [r for r in panel if r.get("status", "ok") != "ok"]
    panel = [r for r in panel if r.get("status", "ok") == "ok"]
    status = "ok" if panel and not failed else "error"
    if not panel:
        return {"verdict": fallback_verdict, "score": 3,
                "rigor_score": 3, "depth_score": 3,
                "agreement": "unanimous", "strengths": [],
                "weaknesses": [], "suggestions": [], "blocking": "",
                "must_flag_hits": [], "status": status}

    verdicts = [(r.get("verdict") or "accept") for r in panel]

    def _median_score(key: str) -> int:
        """Median of a numeric per-persona score, or 3 if no persona
        returned one. Used for `score` plus the new `rigor_score`/
        `depth_score` axes so the aggregated verdict carries them
        through to the revise-loop signal."""
        vals: list[int] = []
        for r in panel:
            s = r.get(key)
            if isinstance(s, (int, float)):
                vals.append(int(round(float(s))))
        vals = vals or [3]
        vals.sort()
        return vals[len(vals) // 2]

    median = _median_score("score")
    rigor_median = _median_score("rigor_score")
    depth_median = _median_score("depth_score")

    # Any low-confidence revise vote dominates.
    low_revise = any(
        (r.get("verdict") == "revise"
         and isinstance(r.get("score"), (int, float))
         and float(r["score"]) < 3)
        for r in panel
    )
    if low_revise:
        verdict = "revise"
    else:
        n_revise = sum(1 for v in verdicts if v == "revise")
        n_accept = sum(1 for v in verdicts if v == "accept")
        if n_revise > n_accept:
            verdict = "revise"
        elif n_accept > n_revise:
            verdict = "accept"
        else:
            verdict = "revise"  # ties favor revision (conservative)

    n = len(panel)
    if all(v == verdicts[0] for v in verdicts):
        agreement = "unanimous"
    elif n - max(verdicts.count("accept"), verdicts.count("revise")) <= 1:
        agreement = "split"
    else:
        agreement = "controversial"

    # Strengths: intersection across panel.
    strength_sets = [set(r.get("strengths") or []) for r in panel]
    intersected: set[str] = strength_sets[0] if strength_sets else set()
    for s in strength_sets[1:]:
        intersected &= s
    strengths = sorted(intersected)

    # Weaknesses: deduped union.
    weaknesses: list[str] = []
    seen_w: set[str] = set()
    for r in panel:
        for w in (r.get("weaknesses") or []):
            key = str(w).strip().lower()
            if key and key not in seen_w:
                seen_w.add(key)
                weaknesses.append(str(w))

    # Suggestions: deduped union with persona attribution.
    suggestions: list[str] = []
    seen_s: set[str] = set()
    for r in panel:
        persona = r.get("persona", "?")
        for sug in (r.get("suggestions") or []):
            key = str(sug).strip().lower()
            if key and key not in seen_s:
                seen_s.add(key)
                suggestions.append(f"[{persona}] {sug}")

    # Blocking: first non-empty blocking note across the panel.
    blocking = ""
    for r in panel:
        b = (r.get("blocking") or "").strip()
        if b:
            blocking = b
            break

    # must_flag_hits: union across the panel, deduped, persona-attributed
    # so the downstream router and the human-review UI can show which
    # reviewer raised which fatal-methodology flag. A non-empty list
    # forces a revise even if ``review_loop = false`` (see
    # ``_route_after_review``).
    mfh_out: list[str] = []
    seen_mfh: set[str] = set()
    for r in panel:
        persona = r.get("persona", "?")
        for h in (r.get("must_flag_hits") or []):
            key = str(h).strip().lower()
            if not key or key in seen_mfh:
                continue
            seen_mfh.add(key)
            mfh_out.append(f"[{persona}] {h}")

    cited_out: list[str] = []  # the design paths any panelist cited (checked against the design where they are used)
    for r in panel:
        for x in r.get("design_parts") or []:
            if isinstance(x, str) and x not in cited_out:
                cited_out.append(x)

    return {
        "verdict": verdict, "score": median,
        "rigor_score": rigor_median, "depth_score": depth_median,
        "agreement": agreement,
        "strengths": strengths, "weaknesses": weaknesses,
        "suggestions": suggestions, "blocking": blocking,
        "design_parts": cited_out,
        "must_flag_hits": mfh_out,
        "status": status,
    }


def _names_a_verdict(reply: Any) -> bool:
    """A review reply is usable only when it names one of the verdicts the review prompt allows (``accept`` or
    ``revise``, any case). Anything else ("minor revision", a missing key) is asked for again: a verdict the router
    and the accept checks would not recognise is not a review either."""
    return (
        isinstance(reply, dict)
        and isinstance(reply.get("verdict"), str)
        and reply["verdict"].strip().lower() in ("accept", "revise")
    )


def _auto_accepts(snapshot: dict[str, Any]) -> bool:
    """Whether ``auto_accept_on_pass`` may accept this human-review snapshot for the person: a reviewer's own accept
    (``review_status`` "ok") with no must-flag hit. A stand-in accept is never accepted automatically, and neither is a
    paper that leaves out a number the person asked for and the run computed (``not_in_paper``)."""
    return (
        snapshot.get("verdict") == "accept"
        and not (snapshot.get("must_flag_hits") or [])
        and not (snapshot.get("not_in_paper") or [])
        and snapshot.get("review_status", "ok") == "ok"
        # A figure request the redraw could not apply is told to the person, not accepted for them.
        and not (snapshot.get("layout_not_redrawn") or [])
        # Nor is a refine of a search for the best design FI did not carry out (core/optimise_refine.py).
        and not (snapshot.get("refine_not_done") or [])
    )


def _review_decision(answer: Any) -> bool:
    """Whether a human-review answer (from a callback or a staged answer file) carries a decision: a dict whose
    ``action`` is accept, reject or refine (a refine with its notes; an accept with the person's answer to the
    question asked before accepting, and not "no": core/acceptance.py). Anything else is no decision, and the quest
    stops for one instead of taking it as an accept (a refine with no notes would otherwise become one)."""
    return not _review_answer_problem(answer)


def _review_answer_problem(answer: Any) -> str:
    """Why a human-review answer is not a decision, in plain words ('' when it is one)."""
    if not isinstance(answer, dict):
        return "the review answer had no decision in it (accept, reject or refine)"
    action = str(answer.get("action") or "").strip().lower()
    if action == "refine":
        return "" if str(answer.get("feedback") or "").strip() else "the refine had no notes in it"
    if action == "accept":
        return _acceptance.problem(answer) or ""
    if action == "reject":
        return ""
    return "the review answer had no decision in it (accept, reject or refine)"


def _review_was_real(review: dict[str, Any] | None) -> bool:
    """True when the recorded review is one a reviewer actually gave (``status == "ok"``), not a stand-in.

    An automatic accept, the skills' usage record and the Axon write-back take an ``accept`` as a reviewer's sign-off;
    this is the check they share. A review from before ``status`` existed has none and is taken as real, as the
    aggregator does."""
    return isinstance(review, dict) and review.get("status", "ok") == "ok"


#: Sources a simulation quest needs before one supported finding settles the
#: evidence gate without a model call.
_GATE_RULE_MIN_SOURCES = 15


def _evidence_gate_rule(
    topic_type: str, n_sources: int, n_supporting: int, *,
    analyze_local_first: bool, retrieval_on: bool,
) -> dict[str, Any] | None:
    """The evidence gate's verdict where the counts already decide it, else
    ``None`` and the model is asked.

    From 68 gate calls over 28 quests: the model answered ``broaden`` for
    both simulation quests that had no source with text, and ``sufficient``
    for all six with at least 15 sources and a finding a source supports.
    Between those it gave different verdicts to identical counts, so the
    rest stays with the model. Every one of those quests was a simulation
    with retrieval on, so everything else is always asked: other topic
    types; ``--analyze``, where the literature step is skipped; and a quest
    with retrieval off, where no source is the normal case and a broaden
    would re-run the experiment for sources it cannot find.
    """
    if analyze_local_first or not retrieval_on or topic_type != "simulation":
        return None
    if n_sources == 0:
        return {
            "verdict": "broaden",
            "rationale": "no source with readable text was retrieved.",
            "gaps": ["sources on the research question"],
        }
    if n_sources >= _GATE_RULE_MIN_SOURCES and n_supporting >= 1:
        return {
            "verdict": "sufficient",
            "rationale": (
                f"{n_sources} sources with text, and {n_supporting} "
                f"finding(s) backed by a source."
            ),
        }
    return None


def _format_evidence_note(
    evidence_assessment: dict[str, Any] | None, *, is_survey: bool,
) -> str:
    """The pre-write 'evidence note' handed to the writer when the evidence gate
    judged the assembled evidence thin (``insufficient`` / ``broaden``), so the
    paper frames its limits honestly. Returns ``""`` when the gate was satisfied
    / disabled.

    Suppressed entirely for **survey** topics: a survey is a descriptive
    literature synthesis, not an evidence-gated experiment, so the gate's
    experiment-oriented criteria (results, cross-check support) don't apply and
    this note would just make the essay apologise for experiment-grade evidence
    it never needed. The ``_SURVEY_WRITE_NOTE`` already enforces honest,
    source-only framing for surveys. (The broaden *routing* still ran, so a
    survey that broadened simply gathered more sources.)"""
    ev = evidence_assessment or {}
    if is_survey:
        return ""
    if ev.get("status") == "unknown":
        # The gate itself could not be evaluated (provider/parse failure) —
        # a distinct claim from "judged thin": nobody actually judged it.
        return (
            "**Evidence note (pre-write evidence gate).** The evidence gate could "
            "not be evaluated before this draft (a technical failure, not a "
            "judgement) — whether the assembled evidence is sufficient for this "
            "research question is UNKNOWN, not confirmed. Do not claim or imply "
            "the evidence was checked and found adequate.\n"
        )
    if ev.get("verdict") not in ("insufficient", "broaden"):
        return ""
    gaps = "; ".join(ev.get("gaps") or [])
    return (
        "**Evidence note (pre-write evidence gate).** The assembled "
        f"evidence was judged *{ev.get('verdict')}* for this research question"
        + (f": {ev.get('rationale')}" if ev.get("rationale") else "")
        + (f" Specific gaps: {gaps}." if gaps else "")
        + " Honour this: report only what the sources/data actually support, "
        "state the limitation plainly (an honest, scoped paper is the goal), "
        "and do NOT manufacture confidence the evidence doesn't carry. This is "
        "a scientific limitation of the study — do not phrase it as a failure "
        "of any tool or pipeline.\n"
    )


def _format_claim_grounding(state: QuestState) -> str:
    """Render the claim-grounding result for the review prompt's
    `$claim_grounding_block`, calling out unsupported claims so the reviewer
    must-flags them."""
    failed = str(state.get("claim_check_failed") or "").strip()
    if failed:
        return (
            f"(claim grounding FAILED on this draft: {failed[:200]}. None of its "
            "citations was checked against its source.)"
        )
    g = state.get("claim_grounding") or {}
    if not g:
        return "(claim grounding not run)"
    unsupported = g.get("unsupported") or []
    lines = [
        f"{g.get('grounded', 0)}/{g.get('total', 0)} substantive claims trace "
        "to evidence (this quest's results or a cited source).",
    ]
    if g.get("summary"):
        lines.append(g["summary"])
    if unsupported:
        lines.append("")
        lines.append(
            f"UNSUPPORTED CLAIMS ({len(unsupported)}) — neither this quest's "
            "results nor a cited source backs these:"
        )
        lines.extend(f"  - {c}" for c in unsupported)
        lines.append("")
        lines.append(
            "You MUST add an `unsupported_claim` entry to must_flag_hits and "
            "set verdict=revise for the unsupported claims listed above."
        )
    else:
        lines.append("No unsupported claims were detected.")
    return "\n".join(lines)


def _grounding_after_trim(
    grounding: dict[str, Any], taken: list[str], paper_md: str,
) -> dict[str, Any] | None:
    """``grounding`` without the claims whose sentence a trim took out of the
    paper (``taken``), or ``None`` when none of them is affected. A claim stays
    when another sentence of the paper still says it, since the same statement
    is often made twice and the check lists it once."""
    # Headings are left out: a title names the same things a claim does.
    body = re.sub(r"(?m)^#{1,6}[ \t].*$", "", _paper_body(paper_md))
    remaining = [
        sentence
        for block in re.split(r"\n\s*\n", body)
        for sentence in _SENTENCE_END_RE.split(" ".join(block.split()))
    ]
    kept: list[dict[str, Any]] = []
    for claim in grounding.get("claims") or []:
        text = str(claim.get("claim") or "")
        gone = any(_same_statement(text, sentence) for sentence in taken)
        if gone and not any(_same_statement(text, sentence) for sentence in remaining):
            continue
        kept.append(claim)
    if len(kept) == len(grounding.get("claims") or []):
        return None
    unsupported = [c["claim"] for c in kept if c.get("basis") == "unsupported"]
    return {**grounding, "claims": kept, "total": len(kept), "grounded": len(kept) - len(unsupported),
            "unsupported": unsupported}


# Must-flag hits that are problems with the paper's text, not with the study:
# fixing one means writing the paper again, not running the experiment again.
# ``figure_caption`` and ``figure_missing`` stay here so a reviewer's hit of either still reads as text: FI places the
# figures and writes their captions, so neither is a problem with the study.
_TEXT_ONLY_HITS = frozenset({
    "unsupported_claim", "figure_caption", "figure_missing", "citations_unchecked", "over_page_limit",
    # A statistic the paper describes as something the run did not compute:
    # the numbers are right and the words around them are wrong, so the paper
    # is written again and the experiment is left alone.
    "mislabelled_statistic",
    # A person's refine point the paper does not answer: the writer said it needs no new experiment (a point that
    # does goes to the design from the writing step), so the paper is written again.
    "user_feedback_unaddressed",
    # A claim FI's own records contradict (core/record_claims.py): the paper is written again with it named.
    "record_contradiction",
    # What the paper states about the run against FI's records (a run time, a limit claim, a part the study did not do that
    # the paper leaves unnamed; core/paper_facts.py): the paper is written again, the design is not touched.
    "paper_record_mismatch",
})
_UNANSWERED_NOTE_HIT = "user_feedback_unaddressed"


# How many times a paper that claims what FI's own records contradict is written again before the quest ends with no paper.
_RECORD_REWRITES = 2
#: The folder of ``code/`` that holds a copy of the selected skills' files for the coding agent (FI's own, like ``tests``).
_SKILLS_IN_CODE = "skills"
#: Passes of taking out what the record check still flags once the rewrites are used up (each pass can bring up the next
#: sentence of a kind the check names one of).
_TAKE_OUT_PASSES = 5

# A paper with a page limit: the review renders each draft the way paper.pdf
# is rendered and counts its pages. The limit is a guideline: no hit is forced and
# no rewrite made for it. A reviewer's own ``over_page_limit`` hit is dropped.
_PAGE_LIMIT_HIT = "over_page_limit"
# The most calls one trim makes to the model: a list that falls short of the words to
# cut, or a draft still over the limit after the sentences were taken, is another
# round on the paper as it then stands.
_TRIM_ROUNDS = 3
# Words a full page holds and the share of a page one figure takes, from the
# graded SIR papers at the 1 in layout. The page-limit layout holds more, so a
# budget worked out from these leaves room.
_WORDS_PER_PAGE = 450
_FIGURE_PAGE_SHARE = 0.37


def _without_reviewer_page_limit_hits(hits: list[Any], log: Any = None) -> list[Any]:
    """``hits`` without any the reviewer named ``over_page_limit``. Only the
    engine's page count forces that hit: a reviewer that names it has not
    measured the PDF, and since its hit never moves the shortening counter,
    the rewrite it routes to could repeat without end."""
    kept = [hit for hit in hits if _hit_name(hit) != _PAGE_LIMIT_HIT]
    if log is not None and len(kept) != len(hits):
        log.warning(
            "[review] dropped %d over_page_limit hit(s) the reviewer wrote; only the "
            "measured page count forces that hit", len(hits) - len(kept),
        )
    return kept


def _words_to_cut(pages: int, limit: int, last_page_empty: float) -> int:
    """About how many words take a draft of ``pages`` pages down to ``limit``:
    a page's worth for each full page past the first one over, and the filled
    part of the last page. At least 100, in steps of 50."""
    filled = 1.0 - min(1.0, max(0.0, last_page_empty))
    over = max(0.0, (pages - limit - 1) + filled)
    return max(100, int(over * _WORDS_PER_PAGE / 50.0 + 0.5) * 50)


def _paper_figure_count(state: QuestState) -> int:
    """The figures the paper will carry: the ones the run drew, else the ones
    the design planned."""
    figures = state.get("figures") or []
    if figures:
        return len(figures)
    planned = (state.get("design") or {}).get("figures_planned")
    return len([f for f in planned if str(f).strip()]) if isinstance(planned, list) else 0


def _page_limit_note(limit: int | None, n_figures: int) -> str:
    """The write prompt's ``$page_limit_note``: nothing without a page limit, so
    that prompt stays as it was; with one, the length the paper must keep to."""
    if limit is None:
        return ""
    budget = limit * _WORDS_PER_PAGE - n_figures * _FIGURE_PAGE_SHARE * _WORDS_PER_PAGE
    words = max(150, int(round(budget / 50.0)) * 50)
    pages = f"{limit} page" + ("" if limit == 1 else "s")
    figures = (
        f", less about a third of a page for each of the {n_figures} figures"
        if n_figures > 1 else ", less about a third of a page for the figure" if n_figures == 1 else ""
    )
    return (
        f"\n\n**Page limit: the rendered paper must fit in {pages}.** This overrides the page "
        f"and word counts of the `Study depth` lengths above. Write about {words} words for the "
        f"whole paper, abstract through the last section ({limit} × {_WORDS_PER_PAGE} words a "
        f"page{figures}). The References and Further reading lists added after your last "
        "section take room on those pages too, so do not pad. Each draft is rendered and its "
        "pages counted; a draft over the limit comes back to be shortened."
    )

# A node that judges the paper reads all of it; the cap only guards against a
# runaway file. The old 16,000-character cut hid the second half of a real
# 34,910-character paper from both the claim check and the review.
_PAPER_PROMPT_CHARS = 120_000


def _paper_for_prompt(paper_md: str, node: str, log: Any = None) -> str:
    """``paper_md`` whole, or its first ``_PAPER_PROMPT_CHARS`` characters with
    a note saying so (and a warning in the log) when it is longer."""
    if len(paper_md) <= _PAPER_PROMPT_CHARS:
        return paper_md
    if log is not None:
        log.warning(
            "[%s] the paper is %d characters; the model reads the first %d",
            node, len(paper_md), _PAPER_PROMPT_CHARS,
        )
    return (
        paper_md[:_PAPER_PROMPT_CHARS]
        + f"\n\n[The paper continues: only its first {_PAPER_PROMPT_CHARS:,} of "
        f"{len(paper_md):,} characters are shown.]"
    )


def _citations_unchecked(state: QuestState) -> list[str]:
    """The forced hit for a draft the claim check could not run on. Text-only:
    the rewrite runs the claim check again."""
    reason = str(state.get("claim_check_failed") or "").strip()
    if not reason:
        return []
    return [
        f"citations_unchecked: the claim check could not run on this draft ({reason[:160]}), "
        "so no citation was checked against its source; the paper cannot be accepted "
        "until the check runs"
    ]
# The name a hit starts with, after a panel's ``[persona] `` prefix:
# ``unsupported_claim``, ``[methodologist] figure_caption: Figure 2 ...``.
_HIT_NAME_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?[`'\"]?([A-Za-z_]+)")


def _read_hit_places(raw: Any, table: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """The reviewer's ``must_flag_places`` (``[{"hit": ..., "ids": [...]}]``) as ``{hit: places}``, each place read from FI's
    own table of ids; a hit with no valid id has no entry (core/paper_scope.py)."""
    out: dict[str, list[dict[str, str]]] = {}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        places = _places_as_placeholders(_paper_scope.read_places(item.get("ids"), table))
        hit = str(item.get("hit") or "").strip()
        if hit and places:
            out[hit] = places
    return out


def _places_as_placeholders(places: list[dict[str, str]]) -> list[dict[str, str]]:
    """Places (core/paper_scope.py) with each passage in the writer's placeholder form, the form the earlier draft is
    read in when a section is written again."""
    return [{**p, "passage": _paper_facts.to_placeholders(p["passage"])} for p in places]


def _hit_name(hit: Any) -> str:
    """The identifier a must-flag hit starts with, in lower case and without a
    plural s; empty when it starts with none."""
    m = _HIT_NAME_RE.match(str(hit))
    return m.group(1).lower().removesuffix("s") if m else ""


_IDENTIFIER_HIT = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?[`'\"]?([A-Za-z]+_[A-Za-z_]+|[A-Za-z]+(?=[`'\"]?\s*$))")


def _objection_key(hit: Any) -> str:
    """How FI's record names an objection: its identifier (``circular_evaluation``, ``circular_evaluation: ...``, a hit
    that is one word), or else its whole text in lower case, so two objections written as prose ("Baseline: weak",
    "Baseline: missing") are never taken for one."""
    if _IDENTIFIER_HIT.match(str(hit)):
        return _hit_name(hit)
    text = re.sub(r"^\s*\[[^\]]*\]\s*", "", str(hit))
    return " ".join(text.lower().split())[:120]


def _unsupported_claim_hit(state: QuestState, hits: list[Any]) -> list[str]:
    """``unsupported_claim`` when the claim check found claims nothing backs and the reviewer did not name it: FI's own
    record, so it does not depend on the reviewer copying it."""
    unsupported = _review_items((state.get("claim_grounding") or {}).get("unsupported"))
    if not unsupported or any(_hit_name(h) == "unsupported_claim" for h in hits):
        return []
    return [f"unsupported_claim: the claim check found {len(unsupported)} claim(s) that neither this study's results "
            "nor a cited source backs"]


def _hits_need_only_a_rewrite(hits: list[Any]) -> bool:
    """True when every must-flag hit names a problem with the paper's text, so
    the review sends the quest back to ``write`` instead of ``design``."""
    names = [_hit_name(hit) for hit in hits]
    return bool(names) and all(name in _TEXT_ONLY_HITS for name in names)


# How many times one quest may have its experiment written and run again
# because the review found a problem with what it computed. A re-run costs an
# iteration and a full implement → execute → analyze → write chain, so a flag
# the reviewer keeps raising cannot spend the budget on re-runs: after this
# many, the same flag takes the text routes it may also belong to.
_CODE_REEXECUTES = 1


def _review_sends_the_experiment_back(review: dict[str, Any], state: QuestState) -> str:
    """What a review sends back to be run again: always ``""``. A review is free text, and matching the names of the
    run's result keys, figure files or ``experiment.py`` against its prose guessed which number the reviewer meant;
    in eight real quests the name was a seed, an oracle, a figure file or a result key that a caption or a source
    attribution mentioned, and the study ran again for a sentence of the paper. Every check that measures a computed
    number as wrong (the results check, the known-answer checks, the run manifest, the evidence gate) runs before the
    review and repairs or leaves out what it finds, so a review that names a number adds no measurement. A review of
    the paper goes to ``write``. The function stays as the one place the routes and the implement prompt ask, so a
    future measured finding can be returned here."""
    return ""


def _rerun_directive(review: dict[str, Any], named: str, analysis: bool = False) -> str:
    """What the implement prompt is told when a review sent the experiment
    back: what the review named, and the review's own must-fix prose. ``analysis``: FI writes the analysis and reports the
    results itself (core/analysis_frame.py), so the code is not asked to report them."""
    notes = [
        item
        for field in ("blocking", "weaknesses", "suggestions")
        for item in _review_items(review.get(field))
    ]
    return "\n".join([
        "",
        "## The review sent this experiment back",
        f"The review of the last draft found a problem with `{named}` — something "
        "this run computed, which no rewrite of the paper can fix. Write the "
        + ("code so that it is computed correctly (FI reports it from its own record), "
           if analysis else "experiment so that it is computed correctly and reported in RESULT_JSON, ")
        + "and keep everything the design asks for.",
        *(f"  - {note}" for note in notes),
    ])


def _extend_directive(scripts: dict[str, str], points: list[str], submit: str = "",
                      package: tuple[str, dict[str, str]] | None = None, frame: bool = False,
                      analysis: bool = False) -> str:
    """What the implement prompt is told when a person's refine asked for a number the study lacks: the scripts that
    exist are the base, and only what the number needs is changed. ``package``: the model's package the simulation
    computes with (``core/code_layout.py``), shown too so the extension sees the equations it calls. ``analysis``: FI writes
    the analysis (core/analysis_frame.py): ``scripts`` are simulate.py and derive.py, and the reply gives functions only."""
    if analysis:
        shown = "\n\n".join(f"### code/{name}\n```python\n{code}\n```" for name, code in scripts.items())
        if submit.strip():
            shown += f"\n\n### code/{_trial_runner.SUBMIT_NAME}\n```python\n{submit}\n```"
        if package and package[1]:
            shown += "\n\n" + _code_layout.sources_block(package[1]) + "\n\n" + _code_layout.extend_note(package[0])
        return "\n".join([
            "",
            "## Extend the code that exists",
            "This study was already run and written up. A person read it and asked for a number it lacks. Do not write a new "
            "experiment: start from the code below and change as little as you can so that it also gives what is asked. "
            "Keep the protocol, the design, every setting and every function that exists exactly as they are; FI keeps its "
            "frame, runs the trials and writes the analysis that reports the results. " + _skeleton.EXTEND_REPLY_ANALYSIS,
            "What the person asked for:",
            *(f"  - {p}" for p in points),
            "",
            shown or "(no code was saved: write it from the design)",
        ])
    shown = "\n\n".join(f"### code/{name}\n```python\n{code}\n```" for name, code in scripts.items())
    if submit.strip():
        shown += f"\n\n### code/{_trial_runner.SUBMIT_NAME}\n```python\n{submit}\n```"
    if package and package[1]:
        shown += "\n\n" + _code_layout.sources_block(package[1]) + "\n\n" + _code_layout.extend_note(package[0])
    return "\n".join([
        "",
        "## Extend the scripts that exist",
        "This study was already run and written up. A person read it and asked for a number it lacks. Do not write a new "
        "experiment: start from the script(s) below and change as little as you can so that they also produce what is "
        "asked. Keep the protocol, the design, every setting, the seeds, the RESULT_JSON keys that exist and every figure "
        "that exists exactly as they are (same file names, same content); only add the measurement, the setting or the "
        "figure the ask needs, and report it in RESULT_JSON. "
        + (_skeleton.EXTEND_REPLY if frame else "Return the whole script(s), in the reply format above."),
        "What the person asked for:",
        *(f"  - {p}" for p in points),
        "",
        shown or "(no script was saved: write it from the design)",
    ])


def _review_items(value: Any) -> list[str]:
    """A review field as its non-empty strings: the model may give a list, one
    string, or nothing."""
    items = value if isinstance(value, list) else [value] if value else []
    return [s for s in (str(i).strip() for i in items) if s]


# What the writer is given about the study, apart from the review and the source
# lists. While none of it has changed, an earlier draft still describes the
# study and can be edited; once the design, the run or the analysis has been done
# again, it cannot.
_PAPER_BASIS_KEYS = (
    "design", "analysis", "result_json", "figures", "figure_records", "cross_check",
    "feedback_history",
)


def _paper_basis(state: QuestState) -> str:
    """A fingerprint of the study a draft is written from (``_PAPER_BASIS_KEYS``)."""
    import hashlib

    basis = {key: state.get(key) for key in _PAPER_BASIS_KEYS}
    return hashlib.sha256(json.dumps(basis, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


# A line of the writer's reply naming a point that needs more than text: bare, quoted, bulleted, numbered (``1.`` /
# ``1)``), bold or in backticks.
_REFINE_POINT_RE = re.compile(
    r"^[ \t>*_`-]*(?:\d+[.)][ \t]*)?[*_`]*NEEDS_(EXPERIMENT|DATA|LAYOUT)[*_`]*:[*_`]*[ \t]*(.*?)[ \t*_`]*$", re.MULTILINE)


def _take_refine_points(markdown: str) -> tuple[str, dict[str, list[str]]]:
    """The paper without its ``NEEDS_EXPERIMENT:`` / ``NEEDS_DATA:`` / ``NEEDS_LAYOUT:`` lines, and the points they
    name, by kind (``experiment``, ``data``, ``layout``)."""
    points: dict[str, list[str]] = {"experiment": [], "data": [], "layout": []}
    for m in _REFINE_POINT_RE.finditer(markdown):
        if m.group(2).strip():
            points[m.group(1).lower()].append(m.group(2).strip())
    if not _REFINE_POINT_RE.search(markdown):
        return markdown, points
    cleaned = _REFINE_POINT_RE.sub("", markdown)
    cleaned = _strip_outer_fence(re.sub(r"\n{3,}", "\n\n", cleaned).strip())
    return cleaned.rstrip() + "\n", points


# The writer's one-sentence reason for how it answered a person's refine (text only, more data, a new layout or a new
# experiment), on a line of its own; decorated the same ways a NEEDS_ line may be.
_REFINE_WHY_RE = re.compile(r"^[ \t>*_`-]*[*_`]*REFINE_WHY[*_`]*:[*_`]*[ \t]*(.*?)[ \t*_`]*$", re.MULTILINE)


def _take_refine_why(markdown: str) -> tuple[str, str]:
    """The paper without its ``REFINE_WHY:`` line(s), and the reason they give (the last non-empty one)."""
    reasons = [m.group(1).strip() for m in _REFINE_WHY_RE.finditer(markdown) if m.group(1).strip()]
    if not _REFINE_WHY_RE.search(markdown):
        return markdown, ""
    cleaned = _strip_outer_fence(re.sub(r"\n{3,}", "\n\n", _REFINE_WHY_RE.sub("", markdown)).strip())
    return cleaned.rstrip() + "\n", (reasons[-1] if reasons else "")


def _extension_values(group: str, values: list[float]) -> str:
    """``max_abs_errors = 0.02714, 0.01321``: one result an extension added, as the writer is shown it (cut to a
    readable length)."""
    text = ", ".join(f"{v:.6g}" for v in values) or "(see the results)"
    return f"{group} = {text[:600]}{' ...' if len(text) > 600 else ''}"


def _refine_round(state: QuestState) -> bool:
    """This write answers a person's refine it has not answered yet: a round of ``feedback_history`` the writer has
    not answered (a refine whose notes are only in ``human_feedback``, an older state's shape, is not one)."""
    return ((state.get("human_feedback") or {}).get("action") == "refine"
            and _refine_count(state) > int(state.get("refine_written_for") or 0))


def _refine_count(state: QuestState) -> int:
    """How many refines the person has sent (each is a round of ``feedback_history``)."""
    return sum(1 for h in state.get("feedback_history") or [] if isinstance(h, dict) and str(h.get("text") or "").strip())


def _user_feedback_review_block(state: QuestState) -> str:
    """The review prompt's ``$user_feedback_block``: what the person asked for at the last gate (the rounds since
    ``feedback_rounds_from``; only the latest one for an older state), so a point the paper leaves unanswered is a
    must-fix finding. Earlier rounds were answered at an earlier gate and are not raised again. Empty when there is
    none."""
    history = [h for h in state.get("feedback_history") or [] if isinstance(h, dict)]
    start = state.get("feedback_rounds_from")
    history = history[int(start):] if isinstance(start, int) else history[-1:]
    # (A note the search for the best design answered is not one for the paper: core/optimise_refine.py.)
    history = [h for h in history if not h.get("answered_by")]
    rounds = [str(h.get("text") or "").strip() for h in history]
    rounds = [r for r in rounds if r]
    if not rounds:
        return ""
    return (
        "\n\n## What the person asked for (their refine notes; a point the paper does not answer is a must-fix "
        "finding: add `user_feedback_unaddressed` to must_flag_hits)\n" + "\n".join(f"- {r}" for r in rounds)
    )


def _format_review_for_writer(state: QuestState, *, refine_round: bool = False) -> str:
    """The write prompt's ``$review_feedback``: what the review of the previous
    draft asks for (its must-fix hits, its weaknesses and suggestions, and the
    claims the claim check found unsupported), and every round of the user's
    feedback. A
    first draft has none."""
    review = state.get("review") or {}
    lines: list[str] = []
    if review:
        score = review.get("score")
        lines.append(
            f"Verdict: {review.get('verdict', 'revise')}"
            + (f" (score {score})" if score is not None else "")
        )
    for heading, key in (
        ("Must fix", "must_flag_hits"),
        ("Weaknesses", "weaknesses"),
        ("Suggestions", "suggestions"),
    ):
        items = _review_items(review.get(key))
        if items:
            lines += [f"{heading}:", *(f"  - {item}" for item in items)]
    objections = [h for h in _review_items(review.get("must_flag_hits")) if _hit_name(h) not in _TEXT_ONLY_HITS]
    blocking = str(review.get("blocking") or "").strip() if review.get("verdict") == "revise" else ""
    if objections or blocking:
        lines += [
            "Objections to the design that no check of FI's confirms (the study is not changed for them; FI keeps them "
            "in its quality report, not in the paper. Where one names a real limit of this study, say so in the "
            "Limitations in your own words; never write that the study answers them):",
            *(f"  - {item}" for item in [*objections, *([blocking] if blocking else [])]),
        ]
    unsupported = _review_items((state.get("claim_grounding") or {}).get("unsupported"))
    if unsupported:
        lines += [
            "Claims that neither this study's results nor a cited source backs:",
            *(f"  - {claim}" for claim in unsupported),
        ]
    # A rewrite for another reason (the page limit) dropped a citation the check had confirmed against its source's
    # text, and cited title-only books instead: the confirmed ones are named so a rewrite keeps them.
    confirmed = [
        c for c in (state.get("claim_grounding") or {}).get("claims") or []
        if isinstance(c, dict) and c.get("basis") == "citation" and c.get("citation_index") is not None
    ]
    if confirmed and review:
        lines += [
            "Citations the check confirmed against the source's own text (keep each where its sentence stays):",
            *(f"  - [{c['citation_index']}] for: {str(c.get('claim') or '').strip()[:160]}" for c in confirmed[:20]),
        ]
    rounds = [
        h for h in state.get("feedback_history") or []
        if isinstance(h, dict) and str(h.get("text") or "").strip() and not h.get("answered_by")
    ]
    if rounds:
        lines += [
            "The user's feedback (honour every round):",
            *(f"  - (round {h.get('iteration', '?')}) {str(h['text']).strip()}" for h in rounds),
        ]
    if refine_round and rounds:
        lines += [
            "Answer the user's feedback in the text wherever the text can answer it. A point the text cannot answer "
            "gets one line at the end of your reply (FI removes these lines), and you write no numbers for it. Start the "
            "line with NEEDS_DATA: when the study only lacks a number or a result (the same study run with one more "
            "value, one more measure; FI extends the existing script and runs it again); with NEEDS_LAYOUT: when the "
            "results are there and only the figures should be arranged, resized or redrawn differently (FI redraws them "
            "from the saved data and does not run the experiment again); with NEEDS_EXPERIMENT: only when it is a "
            "different study (FI goes back to the design). Follow each with the point in one sentence. Write no such "
            "line when the text can answer every point. Always end your reply with one more line, starting with "
            "REFINE_WHY:, saying in one sentence why the feedback needs only the text, more data, a new figure layout "
            "or a new experiment (FI removes this line too).",
        ]
    return "\n".join(lines) or "(none — first draft)"


def _format_cross_check(state: QuestState) -> str:
    """Render the per-finding cross-paper-check results as a
    bulleted block for the write prompt's `$cross_check_block`."""
    checks = state.get("cross_check") or []
    if not checks:
        return "(none — cross-check disabled or FI holds no result to look up)"
    lines: list[str] = []
    for c in checks:
        lines.append(f"### Finding: {c.get('finding', '(?)')}")
        if c.get("status") == "not_checked":
            lines.append(f"  not checked: {c.get('why_not') or 'FI holds no value of it'}")
            lines.append("")
            continue
        for bucket in ("supporting", "conflicting"):
            items = c.get(bucket) or []
            if not items:
                continue
            lines.append(f"  {bucket} ({len(items)}):")
            for it in items[:5]:
                idx = it.get("index")
                why = (it.get("why") or "").strip()
                ref = "(?)"
                if isinstance(idx, int) and 0 < idx <= len(c.get("candidates") or []):
                    cand = c["candidates"][idx - 1]
                    bits = [cand.get("title") or "(untitled)"]
                    if cand.get("doi"):
                        bits.append(f"DOI:{cand['doi']}")
                    elif cand.get("url"):
                        bits.append(cand["url"])
                    ref = " · ".join(bits)
                lines.append(f"    - [{idx}] {ref} — {why}")
        if c.get("summary"):
            lines.append(f"  summary: {c['summary'][:300]}")
        lines.append("")
    return "\n".join(lines).strip() or "(no classifiable hits)"


_FENCE_RE = re.compile(r"^```(?:\w+)?\n(.*)\n```$", re.DOTALL)


_REPAIR_REPLY_KEYS = ("functions", "code", "package_files", "oracle_change", "patch_summary", "give_up_reason")


_CODE_FENCE_RE = re.compile(r"```[ \t]*(\w*)[^\n]*\n.*?(?:```|\Z)", re.DOTALL)


def _repair_template_values() -> set[str]:
    """The placeholder text FI's own repair prompt (agents/execute_reflect.md) puts in ``functions`` and ``code`` of its
    format example, read from the prompt file so a reply that echoes the format is told from one that fixes something."""
    try:
        text = (PROMPTS_DIR / "execute_reflect.md").read_text(encoding="utf-8")
    except OSError:
        return set()
    return {m.group(1).strip() for m in re.finditer(r'^\s*"(?:functions|code)":\s*"(<.*>)",?\s*$', text, re.MULTILINE)}


def _repair_reply_json(text: str, *, node: str) -> dict[str, Any]:
    """The JSON object of a repair's reply, or ``{}`` when the reply holds none. A reply that is the object (bare or in one
    fence) is read as it is. A reply that opens with a sentence of the model's own ("I have not run this ...") is read from
    the objects it holds that carry a key of the repair format: the last one that carries a fix (functions, code, a change
    to a check or a package file), else the last one that carries any such key (a give-up). An object inside a code block
    that is not JSON (a dict in the script) is not the reply, and a reply that is a whole script in a code block gives
    ``{}``, for the caller to read as code."""
    template = _repair_template_values()

    def blanked(o: dict[str, Any]) -> dict[str, Any]:
        # The prompt's own format echoed back (its placeholder in `functions` or `code`) is not a fix. A field that must be
        # text (`give_up_reason`, `patch_summary`) and is not text is absent; a list of strings in `give_up_reason` is read
        # as one reason (the strings joined with "; "), any other value is not a reason.
        out = {k: ("" if k in ("functions", "code") and isinstance(v, str) and v.strip() in template else v)
               for k, v in o.items() if k not in _REPAIR_TEXT_KEYS or isinstance(v, str)}
        listed = o.get("give_up_reason")
        if isinstance(listed, list) and listed and all(isinstance(x, str) for x in listed) and any(x.strip() for x in listed):
            out["give_up_reason"] = "; ".join(x.strip() for x in listed if x.strip())
        return out

    if _strip_outer_fence(text).lstrip().startswith("{"):
        parsed = _parse_json_lenient(text, node=node) or {}
        return blanked(parsed) if all(isinstance(parsed.get(k), str) or parsed.get(k) is None for k in ("functions", "code")) else {}
    prose = _CODE_FENCE_RE.sub(lambda m: m.group(0) if m.group(1).lower() in ("", "json") else "\n", text or "")
    found = [o for o in _top_level_json_objects(prose) if any(k in o for k in _REPAIR_REPLY_KEYS)]

    def carries_fix(o: dict[str, Any]) -> bool:
        return any(isinstance(o.get(k), str) and o[k].strip() for k in ("functions", "code")) or any(
            o.get(k) for k in ("oracle_change", "package_files"))

    # An object whose `functions` or `code` is not a string is not a repair reply. An object left with no fix and no
    # give-up is not a reply either.
    found = [blanked(o) for o in found if all(isinstance(o.get(k), str) or o.get(k) is None for k in ("functions", "code"))]
    found = [o for o in found if carries_fix(o) or str(o.get("give_up_reason") or "").strip()]
    fixes = [o for o in found if carries_fix(o)]
    return (fixes or found or [{}])[-1]


_REPAIR_TEXT_KEYS = ("give_up_reason", "patch_summary")


def _strip_outer_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text


# Frontier Insight house figure style, prepended to every web-plots script so
# the matplotlib figures match the paper / poster / slides identity (Palatino
# serif, teal-anchored palette, warm off-white ground, hairline grid, no top/
# right spines) instead of the default matplotlib look. Self-contained so it
# runs in the quest's sandbox venv; the palette names are exposed for a script
# that needs to colour specific series on-brand.
_FI_MPL_PREAMBLE = '''# --- Frontier Insight house figure style (auto-injected) ---
import matplotlib as _mpl
_mpl.use("Agg")
from cycler import cycler as _cycler
# Brand palette — teal-anchored with warm, muted neutrals. Reference these
# names if you must colour specific series; otherwise let the cycle apply.
FI_TEAL  = "#0E6E6B"   # primary
FI_DEEP  = "#0A4F4D"   # deep teal — titles
FI_GOLD  = "#C28A2C"
FI_CLAY  = "#B5654A"   # terracotta
FI_SLATE = "#41706D"
FI_SAGE  = "#8FA98A"
FI_SAND  = "#C9A66B"
FI_PLUM  = "#7C5E72"
FI_INK   = "#16302E"
FI_GROUND = "#FBFAF7"  # warm off-white
FI_GRID  = "#E2DED4"
_mpl.rcParams.update({
    "figure.facecolor": FI_GROUND, "axes.facecolor": FI_GROUND,
    "savefig.facecolor": FI_GROUND, "savefig.edgecolor": FI_GROUND,
    "figure.figsize": (8.0, 5.2),
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "Book Antiqua",
                   "Georgia", "DejaVu Serif"],
    "font.size": 11.5,
    # Title: large, deep teal, left-aligned with generous breathing room.
    "axes.titlesize": 16.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": FI_DEEP, "axes.titlepad": 16,
    "axes.labelcolor": FI_SLATE, "axes.labelsize": 11.5, "axes.labelpad": 7,
    "axes.edgecolor": "#6E6A60", "axes.linewidth": 1.0,
    # A single baseline; the light horizontal grid carries the values, so
    # the left/top/right spines and the tick marks are dropped for a clean,
    # editorial read.
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": True,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": FI_GRID, "grid.linewidth": 0.9,
    "xtick.color": FI_SLATE, "ytick.color": FI_SLATE,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "ytick.left": False, "xtick.bottom": True,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "text.color": FI_INK,
    "legend.frameon": False, "legend.fontsize": 10.5,
    "lines.linewidth": 2.2, "lines.markersize": 6,
    "patch.edgecolor": FI_GROUND, "patch.linewidth": 0.8,  # thin gap between bars
    "axes.prop_cycle": _cycler(color=[FI_TEAL, FI_GOLD, FI_CLAY, FI_SLATE,
                                      FI_SAGE, FI_SAND, FI_PLUM]),
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})
# --- end Frontier Insight style ---
'''


def _stamp_figure_credit(path: Path, source_url: str, license_str: str) -> None:
    """Burn a small attribution line onto the bottom of a fetched figure so
    the source + license travel with the image wherever it is embedded.
    Best-effort: no-ops if PIL is unavailable or the image won't open (the
    writer's caption still carries the credit)."""
    try:
        from urllib.parse import urlparse

        from PIL import Image, ImageDraw  # type: ignore[import-not-found]

        img = Image.open(path).convert("RGB")
        host = urlparse(source_url).hostname or source_url
        text = f"Source: {host}  -  {license_str}"
        bar = 20
        out = Image.new("RGB", (img.width, img.height + bar), (251, 250, 247))
        out.paste(img, (0, 0))
        ImageDraw.Draw(out).text((8, img.height + 4), text, fill=(65, 112, 109))
        out.save(path)
    except Exception:  # noqa: BLE001 — attribution also lives in the caption
        pass


def _replicate_seed_count(state: QuestState) -> int | None:
    """How many replicate seeds the results are means over, or ``None`` when
    the state does not say (fewer than two)."""
    n = len(state.get("result_json_replicates") or [])
    return n if n > 1 else None


def _seed_label(seed: Any, n_seeds: int | None) -> str:
    """``seed 0 of 3``, or ``seed 0`` when the number of seeds is not known."""
    return f"seed {seed} of {n_seeds}" if n_seeds else f"seed {seed}"


def _replicate_seeds(n_seeds: int | None) -> str:
    return f"the {n_seeds} replicate seeds" if n_seeds else "the replicate seeds"


def _single_seed_note(seed: Any, n_seeds: int | None) -> str:
    """What a figure the seeds could not redraw as their mean shows. A seed is
    a replicate of the whole experiment, so the figure holds every run that
    seed made."""
    return (
        f"replicate {_seed_label(seed, n_seeds)} only (every run that seed made), "
        f"not the mean over {_replicate_seeds(n_seeds)}"
    )


def _rooted(state: QuestState, root: Any = None) -> QuestState:
    """``state`` with the quest's own root in it. The quest root is a fact of FI's (the engine's ``quest_root``), never
    something a node's state is trusted to hold: the live quest state has no such key, and every figure helper that reads the
    call records takes the root from its caller (``root=``) so that none can silently take the older path. Without ``root``
    the state's own key is what is read (the slides and the poster put it there)."""
    if root is None:
        return state
    return {**state, "quest_root": str(root)}  # type: ignore[return-value]


def _figure_list_for_prompt(state: QuestState, *, root: Any = None) -> str:
    """The figure list for a writer/analysis prompt. Plain charts are listed
    by path; license-clean web figures (``figure_credits``) additionally
    carry their caption + source + license, and are flagged ILLUSTRATIVE so
    the writer credits them and does NOT present them as the quest's own
    results."""
    state = _rooted(state, root)
    figs = state.get("figures") or []
    if not figs:
        return "(none)"
    credits = {c.get("file"): c for c in (state.get("figure_credits") or [])}
    records = state.get("figure_records") or {}
    n_seeds = _replicate_seed_count(state)
    numbers = _figure_numbers(state)
    shown = _figure_files_by_stem(state)
    settings = _run_settings(state)
    lines: list[str] = []
    for f in figs:
        c = credits.get(f)
        number = numbers.get(Path(str(f)).name)
        other = shown.get(Path(str(f)).stem)
        if number is None and other in numbers:
            lines.append(f"- figures/{f} — the same drawing as Figure {numbers[other]} (figures/{other}), saved in "
                         "another format")
        elif number is None:
            lines.append(f"- figures/{f} — draws nothing (FI's record): FI leaves it out of the paper")
        elif c:
            lines.append(
                f"- Figure {number}: figures/{f} — ILLUSTRATIVE (a sourced figure, not your "
                f"own data): {str(c.get('caption', ''))[:140]} "
                f"[FI's caption credits it: Source {c.get('source_url', '')}, "
                f"{c.get('license', '')}]"
            )
        else:
            calls = _figure_claims.load_calls(state["quest_root"], Path(str(f)).stem) if state.get("quest_root") else None
            lines.append(f"- Figure {number}: figures/{f}" + _figure_record_note(
                records.get(f), n_seeds=n_seeds, results=state.get("result_json"), settings=settings, calls=calls)
                + _figure_draws_note(calls) + _figure_menu_note(state, Path(str(f)).stem, calls))
    if any(_panel_place(ax.get("grid")) for f in figs for ax in ((records.get(f) or {}).get("axes") or [])
           if isinstance(ax, dict)):
        lines.append(
            "Where a panel sits (left, right, top, bottom) is FI's record of the saved figure: a sentence "
            "that points to a panel by place uses these words, and one that does not need them names the panel by "
            "what it shows."
        )
    if any((records.get(f) or {}).get("replicate_mean") for f in figs):
        lines.append(
            "A figure drawn as the mean of several seeds shows each line at its mean, "
            "shaded with its 95% confidence interval, and FI's caption says so. FI drew that band: it is the "
            "interval over the seed means, not a Wilson, bootstrap or other interval the experiment's own "
            "figure may have named, so the text does not call it one."
        )
    if any((records.get(f) or {}).get("single_seed") is not None for f in figs):
        lines.append(
            "A figure that shows replicate seed 0 only could not be drawn as the mean over "
            f"{_replicate_seeds(n_seeds)}. A replicate seed repeats the whole experiment, so the "
            "figure draws every run seed 0 made: the text about it quotes seed 0's "
            f"values, or say it shows a single replicate ({_seed_label(0, n_seeds)}), not the "
            "means over the seeds."
        )
    if any(isinstance(ax.get("unlabelled_curves"), dict) for f in figs
           for ax in ((records.get(f) or {}).get("axes") or []) if isinstance(ax, dict)):
        lines.append(
            "A figure that draws curves its legend does not name holds more curves than the legend lists: the text "
            "gives the number of curves FI's record counts, and never the number of names in the legend as the "
            "number drawn."
        )
    if any(_hidden_series(records.get(f)) for f in figs):
        lines.append(
            "A series marked FLAT is drawn at one value on its axis, and one NOT SHOWN "
            "is not visible at all: the text describes what its figure shows, so "
            "it does not describe how such a series changes."
        )
    return "\n".join(lines)


def _read_primary_figures(
    figures_dir: Path, records_dir: Path, figures: list[str],
) -> dict[str, tuple[bytes, bytes | None]]:
    """Each figure file the primary run drew, with the record of what it draws
    (``None`` when it has none), by file name, before the replicate runs draw
    over them."""
    kept: dict[str, tuple[bytes, bytes | None, dict[str, bytes]]] = {}
    for name in figures:
        try:
            image = (figures_dir / name).read_bytes()
        except OSError:
            continue
        try:
            record: bytes | None = (records_dir / f"{Path(name).stem}.json").read_bytes()
        except OSError:
            record = None
        kept[name] = (image, record, _paper_copies(figures_dir, name))
    return kept


def _put_back_paper_copies(source: Path, figures_dir: Path, name: str) -> None:
    """Figure ``name``'s paper copy as it is in the folder ``source`` (a backup of ``figures/``); one that has none
    there has none after."""
    from .plot_style import PAPER_DIRNAME

    for suffix in (".pdf", ".png"):
        kept = source / PAPER_DIRNAME / f"{Path(name).stem}{suffix}"
        target = figures_dir / PAPER_DIRNAME / kept.name
        if kept.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(kept, target)
        else:
            target.unlink(missing_ok=True)


def _paper_copies(figures_dir: Path, name: str) -> dict[str, bytes]:
    """The paper copy of figure ``name`` (``figures/paper/<stem>.pdf`` and ``.png``, ``core/plot_style.py``), by file
    name: put back with the figure, so the paper and the slides show the same run."""
    from .plot_style import PAPER_DIRNAME

    out: dict[str, bytes] = {}
    for suffix in (".pdf", ".png"):
        path = figures_dir / PAPER_DIRNAME / f"{Path(name).stem}{suffix}"
        try:
            out[path.name] = path.read_bytes()
        except OSError:
            continue
    return out


def _restore_primary_figures(
    figures_dir: Path, records_dir: Path, kept: dict[str, tuple[bytes, bytes | None, dict[str, bytes]]],
    *, redrawn: Any,
) -> list[str]:
    """Put back the primary run's file and record of every kept figure that is
    not in ``redrawn``, so it shows seed 0 rather than the last replicate.
    Returns the figures put back."""
    restored: list[str] = []
    from .plot_style import PAPER_DIRNAME

    for name, (image, record, *paper) in kept.items():
        if name in redrawn:
            continue
        target = records_dir / f"{Path(name).stem}.json"
        try:
            figures_dir.mkdir(parents=True, exist_ok=True)
            (figures_dir / name).write_bytes(image)
            copies = paper[0] if paper else {}
            for suffix in (".pdf", ".png"):
                path = figures_dir / PAPER_DIRNAME / f"{Path(name).stem}{suffix}"
                if path.name in copies:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(copies[path.name])
                else:
                    # A replicate's paper copy would show another run than the figure.
                    path.unlink(missing_ok=True)
            if record is None:
                # A replicate's record would describe a figure no longer on disk.
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(record)
        except OSError:
            continue
        restored.append(name)
    return restored


def _unread_results(records: dict[str, Any]) -> list[str]:
    """One line per figure whose body made no read that returned data of a result the plan declared it shows (``declared``
    of its record, set by :mod:`core.analysis_figures`). It is a gap for run.log, not a claim for the caption: the body
    may draw the quantity from other reads, and only the saved axes and legends say what is drawn."""
    lines: list[str] = []
    for name, record in sorted(records.items()):
        declared = record.get("declared") if isinstance(record, dict) else None
        if not isinstance(declared, dict):
            continue
        names = declared.get("labels") if isinstance(declared.get("labels"), dict) else {}
        not_drawn = declared.get("not_drawn")
        missing = [str(names.get(str(r)) or str(r)) for r in (not_drawn if isinstance(not_drawn, list) else [])]
        if missing:
            lines.append(f"{name}: no read of the figure's body returned a value of {', '.join(missing)}")
    return lines


def _read_figure_records(folder: Path, figures: list[str]) -> dict[str, Any]:
    """The plot-style bootstrap's record of what each figure draws, by file
    name. A figure saved without the bootstrap (or in a sandbox that could not
    write the record) has none."""
    records: dict[str, Any] = {}
    for name in figures:
        try:
            data = json.loads((folder / f"{Path(name).stem}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("axes"), list):
            # What the paper copy changed (sizes in inches, text sizes in pt) stays in the file for a person: its
            # numbers are not the figure's, and the number check would otherwise trace a paper's "6.3" or "15" to them.
            data.pop("paper", None)
            _attach_unlabelled_lines(data, folder / f"{Path(name).stem}.seed0.json")
            _figure_facts.attach_series_drawn(data, folder / f"{Path(name).stem}.seed0.json")
            records[name] = data
    return records


#: matplotlib's names for a line style that is not a solid line, as a reader calls it.
_NOT_SOLID = {"--": "dashed", "dashed": "dashed", ":": "dotted", "dotted": "dotted", "-.": "dash-dotted",
              "dashdot": "dash-dotted"}


def _attach_unlabelled_lines(record: dict[str, Any], lines_path: Path) -> None:
    """Add to each panel of ``record`` the dashed, dotted or dash-dotted lines the figure draws that its legend does not
    name, as ``unlabelled_lines`` (``style``, ``count``, ``min``, ``max`` of their y values), from the full line list the
    recorder keeps next to it (``<stem>.seed0.json``). The series list holds the labelled series only, so a theory or
    reference line drawn without a label was nowhere in the record, and a caption could deny it unchecked. A solid
    unlabelled line (a family of runs), a marker or an error-bar cap is not one. Nothing is added when the line list is
    missing or does not match the panels."""
    try:
        full = json.loads(lines_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    panels, drawn = record.get("axes"), full.get("axes") if isinstance(full, dict) else None
    if not isinstance(panels, list) or not isinstance(drawn, list) or len(panels) != len(drawn):
        return
    for panel, got in zip(panels, drawn):
        if not isinstance(panel, dict) or not isinstance(got, dict):
            continue
        groups: dict[str, dict[str, Any]] = {}
        # What the panel draws, counted from the full line list (not from the legend): a panel with no line, no shape,
        # no legend and no text is blank, and the curves drawn without a legend label are counted, so a caption states
        # how many curves the saved figure holds and not only the ones its legend names.
        data = [ln for ln in got.get("lines") or []
                if isinstance(ln, dict) and ln.get("kind") == "data" and len(ln.get("x") or []) >= 2]
        if (not got.get("lines") and not got.get("draws_other") and not got.get("legend") and not got.get("has_text")
                and not panel.get("series")):
            panel["blank"] = True
        loose = [ln for ln in data if str(ln.get("label") or "").startswith("_")
                 and not (_NOT_SOLID.get(str(ln.get("linestyle"))) is not None
                          and str(ln.get("marker") or "None").lower() in ("none", "", " "))]
        if loose:
            panel["unlabelled_curves"] = {"count": len(loose), "drawn": len(data)}
        for line in got.get("lines") or []:
            if not isinstance(line, dict) or not str(line.get("label") or "").startswith("_"):
                continue
            style = _NOT_SOLID.get(str(line.get("linestyle")))
            if style is None or str(line.get("marker") or "None").lower() not in ("none", "", " ") \
                    or len(line.get("x") or []) < 2:
                continue
            ys = [float(v) for v in line.get("y") or []
                  if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
            if not ys:
                continue
            group = groups.setdefault(style, {"style": style, "count": 0, "min": min(ys), "max": max(ys)})
            group["count"] += 1
            group["min"], group["max"] = min(group["min"], min(ys)), max(group["max"], max(ys))
        if groups:
            panel["unlabelled_lines"] = list(groups.values())


def _record_numbers(record: Any) -> list[tuple[float, ...]]:
    """The numbers a figure record says its series draw, labels and panels aside: each series' lowest and highest
    value, each reference line's value, sorted. Empty when the record has none (an old record, or no labelled series),
    and then there is nothing to compare."""
    out: list[tuple[float, ...]] = []
    axes = record.get("axes") if isinstance(record, dict) else None
    for ax in axes if isinstance(axes, list) else []:
        for s in (ax.get("series") or []) if isinstance(ax, dict) else []:
            if not isinstance(s, dict):
                continue
            values = [s.get(k) for k in ("min", "max")] if "min" in s or "max" in s else [s.get("y", s.get("x"))]
            if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                out.append(tuple(float(v) for v in values))
    return sorted(out)


def _same_record_numbers(old: list[tuple[float, ...]], new: list[tuple[float, ...]]) -> bool:
    """Whether two figures draw the same numbers (:func:`_record_numbers`), to the precision a saved table keeps."""
    return len(old) == len(new) and all(
        len(a) == len(b) and all(math.isclose(x, y, rel_tol=1e-3, abs_tol=1e-12) for x, y in zip(a, b))
        for a, b in zip(old, new))


def _rename_code_names(text: Any, renames: dict[str, str]) -> Any:
    """``text`` with each code name in ``renames`` written as the plan names it, whole words only; a unit the plan's
    label already carries is not written twice (``Ea_hat (kJ/mol)`` becomes ``Activation energy $E_a$ (kJ/mol)``)."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for name, new in renames.items():
        out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", lambda _m, new=new: new, out)
    return re.sub(r"(\([^()]*\))(?:\s*\1)+", r"\1", out) if out != text else out


def _hidden_series(record: dict[str, Any] | None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """``(axes, series)`` for each labelled series a figure draws flat or not at all."""
    return [
        (ax, s)
        for ax in ((record or {}).get("axes") or []) if isinstance(ax, dict)
        for s in (ax.get("series") or [])
        if isinstance(s, dict) and s.get("shows") not in ("yes", "vertical line", "reference line")
    ]


def _read_a_search_block(block: Any) -> tuple[dict[str, Any] | None, list[str], list[Any]]:
    """``(block, left_out, stopping)`` for a search block a model wrote: ``block`` is what FI uses (every part and entry that
    could be read, the faulted ones left out alone, the block's own parts as written when it reads whole), ``left_out`` one
    sentence per part left out, and ``stopping`` the faults that make the block unusable (``block`` is then ``None``).
    What a search cannot do without stops it: an objective, a baseline, a limit, a design variable whose value at the
    baseline is not given (a variable that has one is held there and said), or every entry of its scan's settings or parts;
    every other fault costs only its own part (:func:`core.optimisation_plan.read_block`,
    :func:`core.optimisation_plan.repair`)."""
    reading = _optim.read_block(block)
    if not reading.faults:
        return block, [], []
    repaired, notes = _optim.repair(block)
    stopping = [f for f in reading.faults if f.required and len(_reading.split(f.path)) > 2
                and str(_reading.split(f.path)[2]) in ("objective", "baseline", "constraints")]
    # The scan's settings and its parts are the question the search starts from: every entry of one lost would leave the
    # scan running as a different (larger) one, so the answer is asked again, as for an objective.
    for part in ("grid", "sub_studies", "in_each_run"):
        if block.get(part) and repaired is not None and not repaired.get(part):
            stopping += [f for f in reading.faults if len(_reading.split(f.path)) > 2 and str(_reading.split(f.path)[2]) == part]
    if repaired is None or stopping or reading.nothing_read:
        return None, [], stopping or list(reading.faults)
    return repaired, notes, []


def _fault_record(fault: Any) -> dict[str, Any]:
    """A fault as plain data (for a record or a refused-change item)."""
    return {"path": fault.path, "kind": fault.kind, "reason": fault.reason, "allowed": list(fault.allowed or []) or None,
            "as_written": fault.as_written, "required": bool(fault.required)}


def _fault_from_record(record: dict[str, Any]) -> Any:
    """The :class:`~core.answer_reading.Fault` of :func:`_fault_record`."""
    return _reading.Fault(record["path"], record["kind"], record["reason"],
                          tuple(record["allowed"]) if record.get("allowed") else None, record.get("as_written"),
                          bool(record.get("required")))


def _entries_as_written(protocol: Any, faults: Any, *, you: bool = False) -> str:
    """The entries of ``protocol`` that ``faults`` (each at its path) are about, as written, in one sentence; ``""`` when none
    is found. One entry gives ``the entry as written: ...`` (``you``: ``the entry as you wrote it: ...``). Each entry is
    found by its fault's path (:func:`core.plan.offending_entry`), not by words in a reason."""
    found: list[str] = []
    for fault in faults or []:
        entry = _plan.offending_entry(protocol, fault)
        if entry and entry not in found:
            found.append(entry)
    if not found:
        return ""
    lead = (("the entry as you wrote it" if you else "the entry as written") if len(found) == 1
            else ("the entries as you wrote them" if you else "the entries as written"))
    return f"{lead}: " + " | ".join(found)


def _one_line(text: Any, limit: int = 60) -> str:
    """One line of ``text``, at most ``limit`` characters, for a message."""
    one = " ".join(str(text or "").split())
    return one if len(one) <= limit else one[: limit - 1] + "…"


def _figure_overlap_findings(state: QuestState) -> list[str]:
    """What the run's saved figures draw over what a reader needs, one sentence
    per finding: a legend over the data it labels, or a figure title over a
    panel title. The plot-style recorder measured each on the finished canvas
    (``layout`` in the figure's record), since only the drawn positions can say.

    Empty once the redraw has been asked for, which happens once per quest, and
    for a figure drawn as the mean over the seeds: the engine drew that one, so
    a repair of the experiment's own script would not change it. A record with
    no ``layout`` (an older one, or a figure that could not be measured) has
    nothing to report. Never raises: a defect here must not stall a quest.
    """
    if state.get("figure_overlap_repaired"):
        return []
    records = state.get("figure_records") or {}
    found: list[str] = []
    try:
        for name in state.get("figures") or []:
            record = records.get(name)
            if not isinstance(record, dict) or record.get("replicate_mean"):
                continue
            for item in record.get("layout") or []:
                if not isinstance(item, dict):
                    continue
                if item.get("check") == "title_overlap":
                    found.append(
                        f'figures/{name}: the figure title "{_one_line(item.get("text"))}" is drawn '
                        f'over the panel title "{_one_line(item.get("over"))}"'
                    )
                elif item.get("check") == "label_overlap":
                    found.append(
                        f'figures/{name}: the {_one_line(item.get("where"), 40)} '
                        f'"{_one_line(item.get("text"), 40)}" and "{_one_line(item.get("over"), 40)}" '
                        "run into each other"
                    )
                elif item.get("check") == "axis_label_overlap":
                    found.append(
                        f'figures/{name}: the {_one_line(item.get("axis"), 4)}-axis label '
                        f'"{_one_line(item.get("text"))}" is drawn over the tick label '
                        f'"{_one_line(item.get("over"), 40)}"'
                    )
                elif item.get("check") == "label_clipped":
                    found.append(
                        f'figures/{name}: the label "{_one_line(item.get("text"))}" runs past the edge '
                        "of the figure"
                    )
                elif item.get("check") == "legend_over_panel":
                    names = [_one_line(n, 30) for n in (item.get("legend") or [])[:4]]
                    more = ", ..." if len(item.get("legend") or []) > 4 else ""
                    found.append(
                        f"figures/{name}: the legend ({', '.join(names)}{more}) is drawn over "
                        f'"{_one_line(item.get("over"), 40)}" of another panel'
                    )
                elif item.get("check") == "legend_off_canvas":
                    names = [_one_line(n, 30) for n in (item.get("legend") or [])[:4]]
                    more = ", ..." if len(item.get("legend") or []) > 4 else ""
                    found.append(
                        f"figures/{name}: the legend ({', '.join(names)}{more}) runs past the edge of the figure"
                    )
                elif item.get("check") == "legend_over_data":
                    names = [_one_line(n, 30) for n in (item.get("legend") or [])[:4]]
                    more = ", ..." if len(item.get("legend") or []) > 4 else ""
                    panel = f'in the panel "{_one_line(item.get("axes_title"))}", ' if item.get("axes_title") else ""
                    found.append(
                        f"figures/{name}: {panel}the legend ({', '.join(names)}{more}) is drawn over "
                        "the lines or points it labels"
                    )
            # Geometry the recorder wrote (a legend off the page, error bars cut off): same repair round.
            # (a legend the layout checks already report is not told twice: the prompt shows 8 findings, the disclosure 3)
            legend_seen = any(isinstance(i, dict) and i.get("check") in ("legend_over_panel", "legend_off_canvas")
                              for i in record.get("layout") or [])
            found.extend(f"figures/{name}: {sentence}" for sentence in _figure_facts.geometry_findings(record)
                         if not (legend_seen and sentence.startswith("The legend of ")))
    except Exception:  # noqa: BLE001
        return []
    return found


def _figure_label_findings(state: QuestState, code_dir: Path | None = None, *, root: Any = None) -> list[str]:
    """Axis labels, panel titles and legend entries of the run's figures that show the code's name for a quantity
    (``Ea_hat``, ``pi1``), one sentence each, with the plan's label to use when the plan gives one
    (``core/figure_labels.py``). A name counts when the plan or the experiment's code (the ``.py`` files in
    ``code_dir`` the run wrote, not FI's redraw scripts) names it and it is written as code writes a name. They go to
    the redraw from the saved numbers (``_node_write``), not to a repair of the script. Never raises."""
    try:
        from . import figure_labels as _figure_labels

        return [_figure_labels.sentence(f) for f in _figure_label_found(state, code_dir, root=root)]
    except Exception:  # noqa: BLE001
        return []


def _figure_label_found(state: QuestState, code_dir: Path | None = None, *, root: Any = None) -> list[dict[str, Any]]:
    """The findings behind :func:`_figure_label_findings`, as ``core.figure_labels.findings`` gives them. Never
    raises."""
    state = _rooted(state, root)
    try:
        from . import figure_labels as _figure_labels
        from .plan import plan_labels, plan_names

        sources: list[str] = []
        if code_dir is not None and Path(code_dir).is_dir():
            for path in sorted(Path(code_dir).rglob("*.py")):
                if path.name in ("replot_layout.py", "replot_figures.py") or len(sources) >= 50 \
                        or path.relative_to(code_dir).parts[0] == _SKILLS_IN_CODE:
                    continue
                try:
                    sources.append(path.read_text(encoding="utf-8", errors="replace")[:400_000])
                except OSError:
                    continue
        protocol = (state.get("design") or {}).get("protocol")
        figures = list(state.get("figures") or [])
        root = state.get("quest_root")
        calls = {f: c for f in figures if root and (c := _figure_claims.load_calls(str(root), Path(str(f)).stem))}
        return _figure_labels.findings(state.get("figure_records") or {}, figures,
                                       plan_labels(protocol), plan_names(protocol),
                                       _figure_labels.code_identifiers(sources), calls)
    except Exception:  # noqa: BLE001
        return []


def _figure_overlap_directive(findings: list[str]) -> str:
    """Stands where the traceback would be in the ``execute_reflect`` prompt, for
    the one redraw a run whose figures overlap is offered (see
    ``Engine._node_execute_reflect``)."""
    return (
        "FIGURE LAYOUT: the script exited 0, printed its RESULT_JSON and drew its figures, "
        "and those results stand: the account of a crash above does not apply. But the saved "
        "figures have text, a legend or a label drawn over what a reader needs, measured on the "
        "finished canvas:\n"
        + "\n".join(f"- {finding}" for finding in findings[:8])
        + "\n\nChange only how these figures are laid out, so that nothing is drawn over "
        "anything else. For a legend over the data: put it outside the axes (loc=\"upper left\" "
        "with bbox_to_anchor=(1.02, 1.0), or loc=\"lower center\" with bbox_to_anchor=(0.5, 1.02) "
        "and ncol), or in a corner the data does not reach, or make it smaller (fontsize, ncol). "
        "loc=\"best\" alone is not a fix: a bare ax.legend() already uses it, and it only picks the "
        "emptiest corner, which is not always empty. Keep the legend inside the saved image "
        "(bbox_inches=\"tight\" does that). For a figure title over a panel title: make "
        "room for it (fig.suptitle(..., y=1.02) with bbox_inches=\"tight\", or "
        "fig.tight_layout(rect=(0, 0, 1, 0.94)), or a constrained layout). For tick labels that run "
        "into each other: rotate them (plt.setp(ax.get_xticklabels(), rotation=30, ha=\"right\")), "
        "or show fewer ticks. For an axis label drawn over tick labels, or a label past the edge of "
        "the figure: make room (fig.subplots_adjust, or bbox_inches=\"tight\"). "
        "When error bars reach past the plotted range and are cut off: widen the y limits "
        "(ax.set_ylim) to include them.\n\n"
        "Do NOT change any computation, data, parameter, use of the random seed, printed value, "
        "the final RESULT_JSON line, the file name of any figure, or what any panel plots: only "
        "positions, sizes, the number of legend columns and the y limits. Return only the functions you change in `functions` (each one whole; FI keeps every other line), one "
        "sentence in `patch_summary`, and leave `give_up_reason` empty."
    )


_PANEL_ROWS = {2: ("top", "bottom"), 3: ("top", "middle", "bottom")}
_PANEL_COLS = {2: ("left", "right"), 3: ("left", "centre", "right")}


def _study_size(design: Any) -> int:
    """How many settings the design's protocol names (0 when it names none): what is compared before and after FI adds to a plan."""
    protocol = design.get("protocol") if isinstance(design, dict) else None
    grid = _optimise_search.study_grid(protocol)
    try:
        return _optimise_search.grid_size(grid) if grid else 0
    except (TypeError, ValueError):
        return 0


def _panel_place(grid: Any) -> str:
    """Where a panel sits on its figure, in a caption's words ("left", "top right", "row 2 of 4"), from the cell of the
    subplot grid the saved figure put it in (``[rows, cols, row start, row stop, col start, col stop]``, recorded on
    savefig by core/plot_style.py); "" for a figure's only panel, a panel that spans the grid, or no grid recorded."""
    if not (isinstance(grid, list) and len(grid) == 6 and all(isinstance(v, int) for v in grid)):
        return ""
    rows, cols, r0, r1, c0, c1 = grid

    def axis(n: int, start: int, stop: int, named: dict[int, tuple[str, ...]], kind: str) -> str:
        if n <= 1 or (start == 0 and stop >= n) or not 0 <= start < n:
            return ""
        return named[n][start] if n in named else f"{kind} {start + 1} of {n}"

    return " ".join(w for w in (axis(rows, r0, r1, _PANEL_ROWS, "row"), axis(cols, c0, c1, _PANEL_COLS, "column")) if w)


def _run_settings(state: QuestState) -> Any:
    """The run's own settings (the protocol's grid, in-each-run values and constants): the frozen protocol when there is
    one, else the design's. A number a figure prints that is one of them is not a result FI lacks."""
    root = state.get("quest_root")
    protocol = _frozen.protocol_of(Path(str(root))) if root else None
    design = state.get("design")
    return protocol or (design.get("protocol") if isinstance(design, dict) else None) or {}


def _printed_in_figure(strings: Any, results: Any, settings: Any = None) -> str:
    """``"text"`` for each string a figure prints, a number in it that is none of the run's recorded results marked as
    such (``results`` is the run's ``result_json``)."""
    from core import number_provenance as _number_provenance

    out = []
    for text in (strings if isinstance(strings, list) else []):
        text = _one_line(str(text), 160)
        if not text:
            continue
        loose = _number_provenance.unrecorded_numbers(text, results, settings)
        out.append(f'"{text}"' + (f" [printed in the figure, not a recorded result: {', '.join(loose)}]" if loose else ""))
    return "; ".join(out)


def _figure_record_note(record: dict[str, Any] | None, *, n_seeds: int | None = None, results: Any = None,
                        settings: Any = None, calls: dict[str, Any] | None = None) -> str:
    """What a figure draws, for the writer: each panel's title, y axis and series. With ``calls`` (FI's record of the
    figure's drawing calls) the panels are what the calls drew, never the saved axes' own words.
    ``n_seeds`` is how many replicate seeds ran, when known; ``results`` is the run's ``result_json``, to tell a number
    the figure prints from one the run recorded."""
    panels = _note_from_calls(calls) if calls else []
    for ax in [] if panels else (record or {}).get("axes") or []:
        if not isinstance(ax, dict):
            continue
        bits = [f'"{ax["title"]}"'] if ax.get("title") else []
        where = _panel_place(ax.get("grid"))
        if where:
            bits.insert(0, f"{where} panel")
        ylim = ax.get("ylim") or []
        if len(ylim) == 2:
            bits.append(
                f'y axis "{ax.get("ylabel") or "y"}" ({ax.get("yscale") or "linear"}, '
                f"{ylim[0]:.3g} to {ylim[1]:.3g})"
            )
        series = []
        for s in ax.get("series") or []:
            if not isinstance(s, dict):
                continue
            shows = s.get("shows")
            if shows == "vertical line" and isinstance(s.get("x"), (int, float)):
                series.append(f"{s.get('label')} a vertical line at x = {s['x']:.3g}")
                continue
            if shows == "reference line" and isinstance(s.get("y"), (int, float)):
                series.append(f"{s.get('label')} a reference line at y = {s['y']:.3g}")
                continue
            span = f'{s["min"]:.3g} to {s["max"]:.3g}' if "min" in s else "no points"
            tail = "" if shows == "yes" else ", FLAT on this axis" if shows == "flat" else ", NOT SHOWN"
            series.append(f"{s.get('label')} {span}{tail}")
        if series:
            bits.append("series: " + "; ".join(series))
        if ax.get("series_drawn"):
            more = ax.get("series_drawn_more")
            shown = [str(t) for t in ax["series_drawn"]]
            if more:
                shown.append(f"and {more} more")
            if ax.get("series_drawn_tail"):
                shown.append(str(ax["series_drawn_tail"]))
            bits.append("how each is drawn (FI's record): " + "; ".join(shown))
        loose = ax.get("unlabelled_curves")
        if isinstance(loose, dict) and loose.get("count"):
            bits.append(f'draws {loose["drawn"]} curves in all, of which {loose["count"]} carry no legend label '
                        "(the legend names the others only)")
        for group in ax.get("unlabelled_lines") or []:
            k = int(group["count"])
            bits.append(f'also draws {k} {group["style"]} line{"s" if k > 1 else ""} with no legend label '
                        f'({group["min"]:.3g} to {group["max"]:.3g})')
        # A legend entry that only repeats a series already named above is not said twice.
        named = {str(s.get("label")) for s in ax.get("series") or [] if isinstance(s, dict)}
        printed = _printed_in_figure([t for t in ax.get("texts") or [] if str(t) not in named], results, settings)
        if printed:
            bits.append("the panel itself prints: " + printed)
        if bits:
            panels.append(", ".join(bits))
    note = (" — " + " | ".join(panels)) if panels else ""
    printed = _printed_in_figure((record or {}).get("texts"), results, settings)
    if printed:
        note += " — the figure itself prints: " + printed
    mean_of = ((record or {}).get("replicate_mean") or {}).get("n")
    single_seed = (record or {}).get("single_seed")
    if mean_of:
        note += f" — each line is the mean of {mean_of} seeds, shaded with its 95% confidence interval"
    elif single_seed is not None:
        note += f" — shows {_single_seed_note(single_seed, n_seeds)}"
    # The paper copy drops a figure's title (core/plot_style.py), and FI's caption does not copy a title (it states the
    # axes and the lines FI recorded): what the title said does not reach the reader.
    gone = [" ".join(str(t).split()) for t in ((record or {}).get("paper_dropped") or []) if str(t).strip()]
    if gone:
        note += (" — in a one-column paper.pdf this figure does not show its title" + ("s " if len(gone) > 1 else " ")
                 + "; ".join(f'"{_one_line(t, 120)}"' for t in gone)
                 + ": FI's caption does not repeat it")
    return note


_PAPER_IMAGE_RE = re.compile(r"!\[(?P<alt>(?:[^\[\]]|\[[^\[\]]*\])*)\]\((?P<src>[^)\s]+)")


def _planned_figure_order(state: QuestState) -> list[str]:
    """The figures the design planned and the run drew, in the order planned.

    A figure a later experiment version superseded is not one of them: every
    ``execute`` empties ``figures/`` first (``_clear_stale_figures``) and
    ``state["figures"]`` is the scan of what the run then drew, so a plan entry
    naming a figure this run did not draw matches nothing here.
    """
    planned = (state.get("design") or {}).get("figures_planned")
    planned_names = [_contract.figure_file(f) for f in planned if str(f).strip()] if isinstance(planned, list) else []
    produced = {Path(str(f)).name for f in (state.get("figures") or [])}
    records = state.get("figure_records") or {}
    return [name for name in dict.fromkeys(planned_names)
            if name in produced and not _figure_draws_nothing(records.get(name))]


def _panel_draws_nothing(ax: dict[str, Any]) -> bool:
    """Whether FI's record of one panel says it shows no data (see :func:`core.figure_facts.panel_draws_nothing`, the one
    rule the caption's description and its layout sentence share)."""
    return _figure_facts.panel_draws_nothing(ax)


def _figure_draws_nothing(record: dict[str, Any] | None) -> bool:
    """Whether a figure's own record says no panel of it draws a point. Such a figure is not put in the paper by FI:
    it shows the reader nothing."""
    axes = [ax for ax in (record or {}).get("axes") or [] if isinstance(ax, dict)]
    return bool(axes) and all(_panel_draws_nothing(ax) for ax in axes)


# Which file of a drawing saved in several formats the paper shows: one every output of the paper (PDF, HTML, slides)
# can embed first.
_FIGURE_FORMAT_ORDER = (".png", ".jpg", ".jpeg", ".svg", ".pdf")


def _figure_files_by_stem(state: QuestState) -> dict[str, str]:
    """The file the paper shows of each drawing the run saved, by file stem: a drawing saved as ``fig.png`` and
    ``fig.pdf`` is one figure."""
    chosen: dict[str, tuple[int, str]] = {}
    for name in dict.fromkeys(Path(str(f)).name for f in state.get("figures") or []):
        suffix = Path(name).suffix.lower()
        rank = _FIGURE_FORMAT_ORDER.index(suffix) if suffix in _FIGURE_FORMAT_ORDER else len(_FIGURE_FORMAT_ORDER)
        stem = Path(name).stem
        if stem not in chosen or rank < chosen[stem][0]:
            chosen[stem] = (rank, name)
    return {stem: name for stem, (_rank, name) in chosen.items()}


def _figure_numbers(state: QuestState) -> dict[str, int]:
    """The number of each figure FI puts in the paper, fixed before the paper is written so the prose refers to the
    figures by FI's numbers (the writer is given them with the figure list): the figures the design planned, in the order
    planned, then the run's other figures in the order drawn, one per drawing (:func:`_figure_files_by_stem`). A figure
    whose own record says it draws nothing has none: it shows the reader nothing."""
    records = state.get("figure_records") or {}
    shown = _figure_files_by_stem(state)
    order = [shown[Path(name).stem] for name in _planned_figure_order(state)]
    order += [shown[Path(str(f)).stem] for f in state.get("figures") or []]
    keep = [name for name in dict.fromkeys(order) if not _figure_draws_nothing(records.get(name))]
    return {name: i + 1 for i, name in enumerate(keep)}


_RUN_FIGURE_IMAGE_RE = _figure_facts.RUN_FIGURE_RE


def _without_figure_embeds(markdown: str) -> tuple[str, int]:
    """``markdown`` with every image of a run figure (``](figures/...)``) taken out, and how many there were: the
    figures and their captions are FI's (:func:`_place_figures`). A line that held only images goes, and with it the
    blank line after it when one stands before it too, so taking out the figures FI placed gives back the text it placed
    them in; no other line changes. Code fences are left as they are."""
    out: list[str] = []
    fence: str | None = None
    taken = 0
    skip_blank = False
    for line in (markdown or "").split("\n"):
        fence_line = _MD_FENCE_LINE_RE.match(line)
        if fence_line:
            fence = None if fence == fence_line.group(1) else (fence or fence_line.group(1))
        elif fence is None and _RUN_FIGURE_IMAGE_RE.search(line):
            taken += len(_RUN_FIGURE_IMAGE_RE.findall(line))
            rest = _RUN_FIGURE_IMAGE_RE.sub("", line).rstrip()
            if not rest.strip():
                skip_blank = not out or not out[-1].strip()
                continue
            line = rest
        if skip_blank and not line.strip():
            skip_blank = False
            continue
        skip_blank = False
        out.append(line)
    return "\n".join(out), taken


def _place_figures(markdown: str, state: QuestState, *,
                   facts: dict[str, str] | None = None, root: Any = None) -> tuple[str, list[str], int]:
    """``markdown`` with the run's figures in it, put there by FI: ``(markdown, names placed, images the draft wrote)``.

    Every figure FI numbered (:func:`_figure_numbers`) goes in, under its number, with the caption FI writes from its
    records (:func:`_figure_caption_text`); every image of a run figure the draft wrote itself is taken out first, caption
    and all, so no caption in the paper is the writer's. Each figure goes after the paragraph whose prose first names it
    by number, and never before the figure numbered one less, so the paper shows them in number order; one the prose
    never names goes before the source lists. Idempotent: placing the figures of a paper FI placed them in gives the
    same paper."""
    state = _rooted(state, root)
    numbers = _figure_numbers(state)
    text, written = _without_figure_embeds(markdown)
    if not numbers:
        return text, [], written
    records = state.get("figure_records") or {}
    credits = {Path(str(c.get("file") or "")).name: c for c in state.get("figure_credits") or [] if isinstance(c, dict)}
    n_seeds = _replicate_seed_count(state)
    lines = text.split("\n")
    placed: list[str] = []
    last = -1
    for name, number in sorted(numbers.items(), key=lambda kv: kv[1]):
        caption = _figure_caption_text(name, records.get(name), n_seeds=n_seeds, facts=facts, credit=credits.get(name),
                                       calls=(_figure_claims.load_calls(state["quest_root"], Path(name).stem)
                                              if state.get("quest_root") else None))
        embed = f"![**Figure {number}.** {caption}](figures/{name})"
        said = _figure_minimum_sentence(lines, number, name, state)
        at = max(_figure_insertion_line(lines, number), last + 1)
        if at >= len(lines):
            if not lines or lines[-1].strip():
                lines.append("")  # the paper ends in a line break, so placing its figures again gives the same paper
            at = len(lines) - 1
        before_sources = at < len(lines) and bool(_SOURCE_LIST_HEADING_RE.match(lines[at]))
        block = [embed, ""] if before_sources else ["", embed]
        if said is not None:
            block = _figure_with_sentence(lines, said, block)
        lines[at:at] = block
        last = at + block.index(embed)
        placed.append(name)
    return "\n".join(lines), placed, written


def _figure_minimum_sentence(lines: list[str], number: int, name: str, state: QuestState) -> tuple[int | None, str] | None:
    """The one sentence FI composes for a figure whose drawing it checked, when the draft places none of the figure's marks:
    ``(line of the paragraph that first names it, sentence)``, the line being ``None`` when no paragraph names it. The
    sentence is the figure's ``drawn`` statement (else its ``range``), a mark FI fills from its record, then "(Figure N)"
    written by FI: the pointer rule never takes out a statement of FI's with its own closing pointer. ``None`` for a figure with no call record, one FI
    drew itself from the plan's description, one with no such statement (a custom drawing alone), or one whose marks the
    draft already holds."""
    stem = Path(name).stem
    calls = _figure_claims.load_calls(state["quest_root"], stem) if state.get("quest_root") else None
    if not calls or calls.get("fallback"):
        return None
    text = "\n".join(lines)
    held = (re.search(r"\{\{\s*figure:\s*" + re.escape(stem) + r"(?:\.\w+)?\s*:", text, re.I)
            or re.search(r"fi:fact-figure-\w+?-" + re.escape(stem) + r"(?:-args[0-9a-f]*)?\s*-->", text))
    if held:
        return None
    offers = _figure_offers(state, stem, calls)
    offer = next((o for kind in ("drawn", "range") for o in offers if o["claim"] == kind), None)
    if offer is None:
        return None
    return _figure_reference_end(lines, number), f"{offer['mark']} (Figure {number})."


def _figure_with_sentence(lines: list[str], said: tuple[int | None, str], block: list[str]) -> list[str]:
    """``block`` (the figure and its blank line) with the composed sentence in the paragraph that names the figure, after its
    last sentence; with a paragraph of its own before the figure when no paragraph names it or the one that does is not
    running prose."""
    end, sentence = said
    if end is not None and lines[end].rstrip().endswith((".", "!", "?")) and not _paper_facts._NOT_PROSE.match(lines[end]):
        lines[end] = lines[end].rstrip() + " " + sentence
        return block
    return ["", sentence] + block


# A figure the prose names: "Figure 2 shows", "Figures 1 and 2", "Fig. 3".
_FIGURE_MENTION_RE = re.compile(r"(?<![A-Za-z])(?:figures?|figs?\.?)\s*(\d+)", re.IGNORECASE)
_MD_FENCE_LINE_RE = re.compile(r"^\s*(```|~~~)")


def _caption_words(text: Any, limit: int = 120) -> str:
    """A label or title from FI's record, as a caption may quote it: one line, at most ``limit`` characters."""
    words = " ".join(str(text or "").split())
    if len(words) <= limit:
        return words
    return _math_safe_cut.head_before_math(words, len(words[:limit].rsplit(" ", 1)[0])).rstrip()


def _caption_safe(text: str) -> str:
    """``text`` as an image's caption can hold it: each square bracket as a parenthesis (a title's ``[0.1, 0.2]`` reads
    ``(0.1, 0.2)``). Markdown ends a caption at an unpaired one, and the readers of the paper's images (the poster, the
    slides) end it at the first closing one."""
    return text.replace("[", "(").replace("]", ")")


def _figures_drawn(state: QuestState, *, root: Any = None) -> dict[int, set[str]]:
    """``{figure number: the metrics its drawing calls drew}`` for each numbered figure that has a call record: the ``what``
    FI checked of each series a panel FI drew itself holds (a joined "a / b" is each of them; "other" is none), never a series'
    id or label. A panel drawn by hand adds none (what it draws is not
    checked). A figure with no call record is not in it: its sentences are not judged."""
    state = _rooted(state, root)
    out: dict[int, set[str]] = {}
    for name, number in _figure_numbers(state).items():
        calls = _figure_claims.load_calls(state["quest_root"], Path(name).stem) if state.get("quest_root") else None
        if not calls:
            continue
        out[number] = _drawn_metrics(calls)
    return out


def _drawn_metrics(calls: dict[str, Any]) -> set[str]:
    """The results a figure's drawing calls drew, as FI checked them: the ``what`` of each series of a panel FI drew itself (a
    joined "a / b" is each of them; "other" is none), never a series' id or label. The one function behind both the line the
    writer reads about a figure and the pointer rule's test of "a quantity that figure draws", so they cannot disagree."""
    metrics: set[str] = set()
    for p in calls.get("panels") or []:
        if not isinstance(p, dict) or p.get("raw"):
            continue
        for s in p.get("series") or []:
            if isinstance(s, dict):
                metrics.update(w.strip() for w in str(s.get("what") or "").split(" / "))
    return metrics - {"", "other"}


def _figure_draws_note(calls: dict[str, Any] | None) -> str:
    """The writer's line about the results a figure FI checked draws, by the names its ``{{v:<name>}}`` marks use; ``""`` when
    it draws none FI checked."""
    names = sorted(_drawn_metrics(calls)) if calls else []
    return "\n    Results this figure draws: " + ", ".join(names) + "." if names else ""


def _take_out_unpointed(text: str, state: QuestState, *, root: Any = None) -> tuple[str, list[dict[str, str]]]:
    """``text`` without the sentences that point at a figure FI drew and are not FI's to stand behind
    (:func:`core.figure_claims.apply_pointer_rule`), and the entries to list, shaped like those of ``take_out_unfilled`` with
    ``key`` ``figure:<stem>:claim`` and the figure's ``figure`` number, so the figure gets one disclosure. Nothing changes for
    a quest none of whose figures has a call record."""
    state = _rooted(state, root)
    drawn = _figures_drawn(state)
    if not drawn:
        return text, []
    numbers = _figure_numbers(state)
    places = {n: _figure_claims.drawn_places(_figure_claims.load_calls(state["quest_root"], Path(name).stem))
              for name, n in numbers.items() if n in drawn}
    out, taken = _figure_claims.apply_pointer_rule(text, numbers, drawn, _FIGURE_MENTION_RE, places)
    by_number = {n: Path(name).stem for name, n in numbers.items()}
    entries: list[dict[str, str]] = []
    for t in taken:
        rest = {k: v for k, v in t.items() if k != "figures"}
        checked = [n for n in t.get("figures") or [] if n in drawn and n in by_number]
        # one entry for every figure with a call record that the sentence names, so each gets its log line and its disclosure
        entries += [{**rest, "key": f"figure:{by_number[n]}:claim", "figure": str(n)} for n in checked] or [rest]
    return out, entries


def _figure_plan(root: Any) -> dict[str, Any] | None:
    """The plan the figure claims read (:func:`core.figure_claims.plan_of`), from the quest's frozen protocol; ``None``
    without one, and then ``not_drawn`` and ``missing_x`` state nothing."""
    try:
        return _figure_claims.plan_of(_paper_facts.plan_protocol(root)) if root else None
    except Exception:  # noqa: BLE001 -- no plan: those two claims fail closed
        return None


def _figure_offers(state: QuestState, stem: str, calls: dict[str, Any]) -> list[dict[str, Any]]:
    """The statements about the figure ``stem`` that are true of its call record (:func:`core.figure_claims.menu`), worded as
    the paper will render them."""
    return _figure_claims.cap_menu(
        _figure_claims.menu(calls, _figure_plan(state.get("quest_root")), _paper_values.sig, stem=stem), calls)


def _figure_menu_note(state: QuestState, stem: str, calls: dict[str, Any] | None) -> str:
    """What the writer may say about a figure whose drawing calls FI recorded: the marks it may place (FI writes the
    sentence), a panel drawn by hand having none. ``""`` for a figure with no call record, which is listed as before."""
    if not calls:
        return ""
    parts = ["\n    Marks FI has checked about this figure (FI writes the sentence):"]
    if calls.get("fallback"):
        parts.append("    this figure was drawn by FI from the plan's description")
    for p in _call_panels(calls):
        if p.get("raw"):
            parts.append(f"    no statements can be made about panel {p.get('panel')} (custom drawing)")
    for offer in _figure_offers(state, stem, calls):
        parts.append(f'    {offer["mark"]}  FI writes: "{offer["text"]}"')
    return "\n".join(parts)


#: What an error bar or a band spans, in a reader's words, by the meaning the call recorded.
_CALL_SPANS = {"ci95": "a 95% confidence interval", "sd": "one standard deviation", "range": "the range of the values",
               "other": "an amount the figure does not state"}


def _call_panels(calls: dict[str, Any]) -> list[dict[str, Any]]:
    """The panels of a call record that draw something: a custom drawing, a series, a reference line or a band."""
    return [p for p in calls.get("panels") or []
            if isinstance(p, dict) and (p.get("raw") or [s for s in p.get("series") or [] if isinstance(s, dict)]
                                        or p.get("refs") or p.get("bands"))]


def _call_plots(p: dict[str, Any]) -> str:
    """What a non-raw panel plots, from the labels FI's calls recorded (the plan's words, never the body's)."""
    y, x = (_caption_words((p.get(a) or {}).get("label")) for a in ("y", "x"))
    return f"{y} against {x}" if y and x else y or x


def _call_raw_texts(p: dict[str, Any]) -> list[str]:
    """The words a custom drawing prints that carry a digit, verbatim (one line each): what a reader could take for a result."""
    texts = [" ".join(str(t).split()) for t in p.get("texts_raw") or []]
    return [f"text in the panel, not checked by FI: '{t}'" for t in texts if any(c.isdigit() for c in t)][:3]


def _call_group(s: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    """The name of a drawn series and the series merged into it, as the legend reads it ("a = b"). Only a series whose label
    FI holds is named; a merged curve's label is the body's words, so it reads "a curve", and an id the record does not
    draw is left out."""
    by_id = {str(e.get("id")): e for e in entries}
    names = [_caption_words(s.get("label"), 60)]
    for m in s.get("merged_with") or []:
        e = by_id.get(str(m))
        if e is not None:
            names.append("a curve" if _unnamed_curve(e) else _caption_words(e.get("label"), 60))
    return " = ".join(n for n in names if n)


def _unnamed_curve(s: dict[str, Any]) -> bool:
    """A curve FI cannot name: one an older record holds that the figure computed itself. A reference of the plan drawn
    as a curve carries the plan's name (``source`` "plan") and is named like a series."""
    return s.get("kind") == "curve" and s.get("source") != "plan"


def _call_drawn(p: dict[str, Any]) -> list[str]:
    """The series a panel's calls drew, one name per drawn id, by the plan's label: a series merged into an identical one
    reads "a = b" (the legend's form); a curve is "a curve": its own label is the body's words."""
    entries = [s for s in p.get("series") or [] if isinstance(s, dict)]
    absorbed = {str(m) for s in entries for m in s.get("merged_with") or []}
    names: list[str] = []
    seen: set[str] = set()
    curves = 0
    levelled: dict[tuple[str, str], list[dict[str, Any]]] = {}  # a figure with one curve per level of a setting: said once for each result
    for s in entries:
        ident = str(s.get("id"))
        if isinstance(s.get("level"), dict):
            levelled.setdefault((str(s["level"].get("result")), str(s["level"].get("result_label"))), []).append(s)
        elif _unnamed_curve(s):
            curves += 1
        elif ident not in absorbed and _call_group(s, entries):
            key = ident + ("\0curve" if s.get("kind") == "curve" else "")  # a plan's curve of a result is its own line
            if key not in seen:
                seen.add(key)
                names.append(_call_group(s, entries))
    names += [_call_levels(group) for group in levelled.values()]
    more = ("plus " if levelled else "") + (f"{curves} curves" if curves > 1 else "a curve")
    return names + ([more] if curves else [])


def _call_levels(group: list[dict[str, Any]]) -> str:
    """What one result's curves are when the figure draws one per level of a setting, in FI's words from its own record: "34 curves
    of X, one per Y". A level drawn as a line and as points is one curve. More levels than the legend names say their range; one
    level says which it is."""
    first = group[0]["level"]
    result = _caption_words(first.get("result_label"), 60) or "the result"
    setting = _caption_words(first.get("setting_label"), 60) or "level"
    by_level: dict[str, dict[str, Any]] = {}
    for g in group:
        by_level.setdefault(str(g["level"].get("text")), g)  # one entry per level, in drawn order
    distinct = list(by_level.values())
    if len(distinct) == 1:
        return f"{result} at {setting} = {_caption_words(first.get('text'), 30)}"
    kinds = {str(g.get("kind")) for g in group}
    unit = {"points": "series of points", "bars": "series of bars"}.get(next(iter(kinds)), "curves") if len(kinds) == 1 else "curves"
    values = [g["level"].get("value") for g in distinct]
    span = ""
    if len(distinct) > 12 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        span = f" (from {min(values):.6g} to {max(values):.6g}; the legend names the lowest and the highest)"
    return f"{len(distinct)} {unit} of {result}, one per {setting}{span}"


def _call_extras(p: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """``(bands, error bars, references)`` of one panel, each as a clause, from the calls only."""
    label = {str(s.get("id")): _caption_words(s.get("label"), 60) for s in p.get("series") or [] if isinstance(s, dict)}
    bands = [f"{label.get(str(b.get('series'))) or 'a series'} has a band: "
             f"{_CALL_SPANS.get(str(b.get('meaning')), _CALL_SPANS['other'])}"
             for b in p.get("bands") or [] if isinstance(b, dict)]
    spans = list(dict.fromkeys(_CALL_SPANS.get(str(s.get("yerr_meaning")), _CALL_SPANS["other"])
                               for s in p.get("series") or [] if isinstance(s, dict) and s.get("yerr")))
    bars = [" and ".join(spans)] if spans else []
    refs = [f"{_caption_words(r.get('label'), 60) or 'a reference'} at {r['value']:.3g}"
            for r in p.get("refs") or [] if isinstance(r, dict) and isinstance(r.get("value"), (int, float))]
    return bands, bars, refs


def _caption_from_calls(name: str, calls: dict[str, Any], record: dict[str, Any] | None, *, n_seeds: int | None,
                        facts: dict[str, str] | None) -> str | None:
    """The caption of a figure whose drawing calls FI recorded: what the calls drew (the series, bands, error bars and
    reference lines, in the plan's words) and nothing else; a series is drawn exactly when a call drew it. A panel the
    body drew itself is "custom drawing", with the words it prints that carry a digit quoted as FI's check did not
    reach them. ``None`` when no panel of the record draws anything."""
    panels = _call_panels(calls)
    if not panels:
        return None
    stem = " ".join(Path(name).stem.replace("_", " ").replace("-", " ").split()) or "the figure"
    plotted = list(dict.fromkeys(t for t in (_call_plots(p) for p in panels if not p.get("raw")) if t))
    description = f"{stem} ({len(panels)} panels)" if len(panels) > 3 else "; ".join(plotted) or stem
    parts = [description.rstrip(" .;,") + "."]
    many = len(panels) > 1
    marks: list[str] = []
    from .figure_claims import panel_name  # the panel's own setting, when it was declared, names it
    for p in panels:
        who = panel_name(p)  # "panel 2", or "the panel at Mass = 200"
        named = many or who != f"panel {p.get('panel')}"  # a panel that was declared to show a setting is named, alone or not
        if p.get("raw"):
            parts.append("; ".join([f"{who}: custom drawing", *_call_raw_texts(p)]) + ".")
            continue
        drawn = _call_drawn(p)
        bands, bars, refs = _call_extras(p)
        marks += refs
        if 1 <= len(drawn) <= 5:
            parts.append((f"{who[0].upper()}{who[1:]} draws " if named else "It draws ") + ", ".join(drawn) + ".")
        for band in bands:
            parts.append((f"In {who}, " if named else "") + band + ".")
        if bars:
            parts.append((f"In {who}, error bars show " if named else "Error bars show ") + bars[0] + ".")
    for p in calls.get("panels") or []:
        if isinstance(p, dict) and not p.get("raw") and not p.get("series") and not p.get("refs") and not p.get("bands"):
            parts.append(f"Panel {p.get('panel')} draws nothing.")
    marks = list(dict.fromkeys(marks))
    if marks:
        parts.append("Marked: " + "; ".join(marks) + ".")  # every reference line drawn, never a cut list
    mean_of = ((record or {}).get("replicate_mean") or {}).get("n")
    single_seed = (record or {}).get("single_seed")
    if mean_of:
        parts.append(f"Each line is the mean of {mean_of} seeds, shaded with its 95% confidence interval.")
    elif single_seed is not None:
        parts.append(f"Shows {_single_seed_note(single_seed, n_seeds)}.")
    return _caption_safe(" ".join(parts))


def _note_from_calls(calls: dict[str, Any]) -> list[str]:
    """What each panel's calls drew, for the writer: one entry per panel that draws something."""
    out = []
    from .figure_claims import panel_name
    for p in _call_panels(calls):
        who = panel_name(p)
        if p.get("raw"):
            out.append("; ".join([f"{who}: custom drawing", *_call_raw_texts(p)]))
            continue
        bits = []
        entries = [s for s in p.get("series") or [] if isinstance(s, dict)]
        absorbed = {str(m) for s in entries for m in s.get("merged_with") or []}
        for s in entries:
            if str(s.get("id")) in absorbed:
                continue
            ys = [v for v in s.get("y") or [] if isinstance(v, (int, float)) and not isinstance(v, bool)]
            span = f" {min(ys):.3g} to {max(ys):.3g}" if ys else " no points"
            bits.append("a curve" if _unnamed_curve(s)
                        else f"{_call_group(s, entries)} ({s.get('kind')}){span}")
        bands, bars, refs = _call_extras(p)
        bits += bands + [f"error bars: {b}" for b in bars] + [f"reference line {r}" for r in refs]
        out.append(f"{who} ({_call_plots(p)}): " + "; ".join(bits))
    return out


def _figure_caption_text(
    name: str, record: dict[str, Any] | None, *, n_seeds: int | None, facts: dict[str, str] | None = None,
    credit: dict[str, Any] | None = None, calls: dict[str, Any] | None = None,
) -> str:
    """The caption FI writes for a figure, from its own records only: what each panel that draws data is (its axes: what
    it plots against what, as the saved figure's axis labels say), the series it draws when there are at most five (one
    drawn flat, or outside the plotted range, said so), its reference lines, the panels' layout (each panel named by its
    title, quoted, else by its axis labels) and the line styles (``facts``, :func:`core.figure_facts.facts`), and whether
    a line is a mean over seeds. A script's title is free text and is quoted only in the layout sentence, as the figure
    shows it. A sourced figure (``credit``) says where it is from. Never empty: with no record at all, the file's own name
    describes it."""
    stem = " ".join(Path(name).stem.replace("_", " ").replace("-", " ").split()) or "the figure"
    if credit:
        what = _caption_words(credit.get("caption"), 200) or stem
        return _caption_safe(f"{what.rstrip(' .;,')}. Source: {credit.get('source_url') or 'not recorded'}, "
                             f"{credit.get('license') or 'licence not recorded'}; an illustration from that source, not "
                             "this study's data.")
    if calls:  # FI's record of the drawing calls: the caption says only what the calls drew
        from_calls = _caption_from_calls(name, calls, record, n_seeds=n_seeds, facts=facts)
        if from_calls is not None:
            return from_calls
    # A panel keeps its place in the saved figure ("panel 3" is the third one drawn) when an empty one is left out.
    numbered = [(k, ax) for k, ax in enumerate((record or {}).get("axes") or [], start=1)
                if isinstance(ax, dict) and not _panel_draws_nothing(ax)]
    axes = [ax for _, ax in numbered]
    # What a panel plots is what its saved axes say: the labels of the panels that hold drawn data, as FI measured them on
    # the figure it saved (an empty panel is left out above). Nothing is inferred from which declared results the body
    # read: a quantity the body worked out from several reads (an error from two values, a spread of a mean) is drawn
    # though it was not read under its own id, and a caption built from the reads named the wrong quantity or said a drawn
    # one was missing. A declared result no read of the body gave is recorded in run.log (``_unread_results``), not
    # claimed in the caption. A panel's title is quoted in the layout sentence (``core.figure_facts``), not here.
    def plots(ax: dict[str, Any]) -> str:
        y, x = _caption_words(ax.get("ylabel")), _caption_words(ax.get("xlabel"))
        return f"{y} against {x}" if y and x else y or x

    def named(k: int, ax: dict[str, Any]) -> str:
        return _caption_words(ax.get("ylabel")) or _caption_words(ax.get("xlabel")) or f"panel {k}"

    plotted = list(dict.fromkeys(t for t in (plots(ax) for ax in axes) if t))
    if len(axes) > 3:
        # A grid of panels: the file's own name says what the grid shows.
        description = f"{stem} ({len(axes)} panels)"
    else:
        description = "; ".join(plotted) or stem
    parts = [description.rstrip(" .;,") + "."]
    appears: dict[str, list[str]] = {}
    flat: dict[str, list[str]] = {}
    outside: dict[str, list[str]] = {}
    marks: list[str] = []
    for k, ax in numbered:
        title = named(k, ax) if len(plotted) > 1 else f"panel {k}"
        for s in ax.get("series") or []:
            if not isinstance(s, dict):
                continue
            label = _caption_words(s.get("label"), 60)
            shows = s.get("shows")
            if label.startswith("_"):
                label = ""  # matplotlib's name for a line given no label
            if shows == "reference line" and isinstance(s.get("y"), (int, float)):
                marks.append(f"{label or 'a reference line'} at {s['y']:.3g}")
            elif shows == "vertical line" and isinstance(s.get("x"), (int, float)):
                marks.append(f"{label or 'a vertical line'} at {s['x']:.3g}")
            elif label and shows != "no points":
                appears.setdefault(label, []).append(title)
                if shows == "flat":
                    flat.setdefault(label, []).append(title)
                elif shows == "outside the axis":
                    outside.setdefault(label, []).append(title)
    if 1 <= len(appears) <= 5:
        hidden = [label for label, where in outside.items() if len(where) == len(appears[label])]
        drawn = [label for label in appears if label not in hidden]
        if drawn:
            parts.append("It draws " + ", ".join(drawn) + ".")
        for label in hidden:
            parts.append(f"{label} is not visible: it lies outside the plotted range.")
        for found, words in ((flat, "is flat at this scale"), (outside, "lies outside the plotted range")):
            for label, where in found.items():
                if found is outside and label in hidden:
                    continue
                if len(where) == len(appears[label]):
                    parts.append(f"{label} {words}.")
                else:
                    parts.append(f"{label} {words} in " + "; ".join(dict.fromkeys(where)) + ".")
    for k, ax in numbered:
        loose = ax.get("unlabelled_curves")
        if isinstance(loose, dict) and loose.get("count"):
            who = "It also" if len(axes) == 1 else f"Panel {k} also"
            n = int(loose["count"])
            parts.append(f"{who} draws {n} further curve{'s' if n > 1 else ''} with no legend label "
                         f"({loose['drawn']} curves in all).")
    marks = list(dict.fromkeys(marks))
    if marks:
        parts.append("Marked: " + "; ".join(marks) + ".")  # every reference line drawn, never a cut list
    key = Path(name).stem
    for kind in ("layout", "lines"):
        sentence = (facts or {}).get(f"{key}:{kind}", "")
        if sentence and (kind == "lines" or len(axes) > 1):
            parts.append(sentence[0].upper() + sentence[1:].rstrip(" .") + ".")
    mean_of = ((record or {}).get("replicate_mean") or {}).get("n")
    single_seed = (record or {}).get("single_seed")
    if mean_of:
        parts.append(f"Each line is the mean of {mean_of} seeds, shaded with its 95% confidence interval.")
    elif single_seed is not None:
        parts.append(f"Shows {_single_seed_note(single_seed, n_seeds)}.")
    return _caption_safe(" ".join(parts))


def _figure_reference_end(lines: list[str], number: int) -> int | None:
    """The last line of the paragraph whose prose first names the figure numbered ``number``, else ``None``. Code fences and
    the captions of figures already in the paper are not prose and are skipped, and so are the source lists."""
    fence: str | None = None
    for i, line in enumerate(lines):
        fence_line = _MD_FENCE_LINE_RE.match(line)
        if fence_line:
            fence = None if fence == fence_line.group(1) else (fence or fence_line.group(1))
            continue
        if fence is None and _SOURCE_LIST_HEADING_RE.match(line):
            break  # the source lists are not the paper's prose
        if fence is not None or _PAPER_IMAGE_RE.search(line) or _RUN_FIGURE_IMAGE_RE.search(line):
            continue
        if any(int(m.group(1)) == number for m in _FIGURE_MENTION_RE.finditer(line)):
            end = i
            while end + 1 < len(lines) and lines[end + 1].strip():
                end += 1
            return end
    return None


def _figure_insertion_line(lines: list[str], number: int) -> int:
    """Where the figure numbered ``number`` goes: after the paragraph whose
    prose first names it, else before the paper's source lists, else at the
    end."""
    end = _figure_reference_end(lines, number)
    if end is not None:
        return end + 1
    for i, line in enumerate(lines):
        if _SOURCE_LIST_HEADING_RE.match(line):
            return i
    return len(lines)


def _parse_json_lenient(
    text: str, *, node: str = "", _log_truncate_chars: int = 500, want: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Find and parse a JSON object inside arbitrary LLM output.

    ``want`` (opt-in): the keys the asked-for object carries. When the usual parse below fails or returns an object
    without them, every JSON object written at the top level of the reply is read, and the last one that carries all
    of them is returned. A real Haiku audit reply held two fenced objects (the amended design alone, then the asked-for
    object with its objections); the first-``{``-to-last-``}`` slice spanned both and the audit was called unreadable."""
    if want:
        parsed = _parse_json_lenient_once(text, node=node, _log_truncate_chars=_log_truncate_chars, _quiet=True,
                                         _prose=False)
        if isinstance(parsed, dict) and all(k in parsed for k in want):
            return parsed
        found = [o for o in _top_level_json_objects(text or "") if all(k in o for k in want)]
        if found:
            return found[-1]
        candidate = _strip_outer_fence(text or "").strip()
        if parsed is None and candidate:
            _log_parse_failure(candidate, node, _log_truncate_chars)
        return parsed
    return _parse_json_lenient_once(text, node=node, _log_truncate_chars=_log_truncate_chars)


def _read_latex_object(chunk: str, *, strict: bool = False) -> dict[str, Any] | None:
    """The JSON object in ``chunk`` after the backslashes of LaTeX a model wrote in its strings (``$\\sigma$``,
    ``\\text{SNR}``) are kept as the LaTeX they are (:func:`core.latex_text.escape_latex_backslashes`), or ``None`` when
    that does not make it parse. It also catches a reply that parses but whose LaTeX came out as an escape (``\\text`` read as a
    tab and ``ext``): any other valid reply gives ``None`` here and is used exactly as parsed. ``strict`` is for a chunk
    that parsed as written: only what cannot be a real newline or tab followed by a word is taken for LaTeX (a script
    in a JSON string has lines that start with ``nu`` or ``to``)."""
    fixed = escape_latex_backslashes(chunk, strict=strict)
    if fixed == chunk:
        return None
    try:
        result = json.loads(fixed)
    except (ValueError, RecursionError):
        return None
    if isinstance(result, dict):
        logging.getLogger("frontier_insight.engine").info(
            "a model's JSON reply held LaTeX with single backslashes (as in \\sigma or \\text{...}); read with them kept"
            " as written%s", " (it had parsed, with some read as a tab or a form feed)" if strict else "")
        return result
    return None


def _top_level_json_objects(text: str) -> list[dict[str, Any]]:
    """Every JSON object written at the top level of ``text`` (an object inside one already read is not counted
    again), in order. Prose, fences and objects that do not parse between them are skipped."""
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    i = text.find("{")
    while i >= 0:
        try:
            obj, end = decoder.raw_decode(text, i)
        except (ValueError, RecursionError):
            # LaTeX written with single backslashes in a string (`$\sigma$`): read once more with them kept.
            closes = _matching_brace_end(text, i)
            repaired = _read_latex_object(text[i:closes])
            if repaired is not None:
                out.append(repaired)
                i = text.find("{", closes)
                continue
            # An object that does not parse (a trailing comma, a reply cut off) is skipped whole, to its matching
            # brace: an object inside it is never read as one of the reply's own (an evidence gate's per-source
            # "verdict" taken for the gate's). A brace in prose ("{ opens a set") is not an object: only the next
            # character is skipped, so it cannot swallow the reply that follows it.
            looks_like_object = text[i + 1:].lstrip()[:1] in ('"', "}")
            i = text.find("{", _matching_brace_end(text, i) if looks_like_object else i + 1)
            continue
        if isinstance(obj, dict):
            # `\text`, `\frac`, `\beta`, `\nabla`, `\rho` are valid JSON escapes: the text came out as a tab, a form feed...
            out.append(_read_latex_object(text[i:end], strict=True) or obj)
        i = text.find("{", end)
    return out


_NOT_GIVEN = object()
_JSON_KEY_RE = re.compile(r'\s*,?\s*"((?:[^"\\]|\\.)*)"\s*:\s*')
_JSON_NEXT_KEY_RE = re.compile(r',\s*"((?:[^"\\]|\\.)*)"\s*:\s*')


def _depth_change(chunk: str) -> int:
    """How many more objects or lists ``chunk`` opens than it closes (brackets inside JSON strings not counted)."""
    depth = 0
    in_string = escaped = False
    for ch in chunk:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return depth


def _read_object_keys(body: str, pos: int) -> tuple[dict[str, Any], list[tuple[str, ...]], str, bool]:
    """The keys of the object whose ``{`` is just before ``pos`` that can be read: ``(readable, key paths that could
    not be read, the parser's reason, whether a key is written twice)``. A value that does not parse (a quote left
    unescaped, a reply cut off) is skipped to the next key written at the same level of the object; inside an object
    that is broken only in one place, the readable keys are kept and the path of the broken one is named
    (``("protocol", "tolerance")``). A value that ends exactly where the reply ends is not trusted (``0.12`` may have
    been ``0.125``)."""
    decoder = json.JSONDecoder()
    readable: dict[str, Any] = {}
    bad: list[tuple[str, ...]] = []
    reason = ""
    seen: set[str] = set()
    while (m := _JSON_KEY_RE.match(body, pos)):
        try:
            key = json.loads('"' + m.group(1) + '"')
        except ValueError:
            break
        if key in seen:
            return readable, bad, reason, True
        seen.add(key)
        start = m.end()
        failure = ""
        try:
            value, end = decoder.raw_decode(body, start)
            if body[end:].lstrip()[:1] not in (",", "}", "]", ""):  # a quote left unescaped ends the string early
                failure = f"the value goes on past where it was read, at character {end}"
            elif not isinstance(value, (dict, list, str)) and not body[end:].strip():
                failure = f"the reply ends right after this value, at character {end}, so it may be cut short"
        except ValueError as e:
            failure = f"{getattr(e, 'msg', 'it is not valid JSON')} at character {getattr(e, 'pos', start)}"
        if not failure:
            readable[key] = value
            pos = end
            continue
        reason = reason or failure
        kept = False
        if body[start:start + 1] == "{":
            inner, inner_bad, _inner_reason, twice = _read_object_keys(body, start + 1)
            if twice:
                return readable, bad, reason, True
            if inner and inner_bad:
                readable[key] = inner
                bad.extend((key, *path) for path in inner_bad)
                kept = True
        if not kept:
            bad.append((key,))
        nxt = None
        for cand in _JSON_NEXT_KEY_RE.finditer(body, start):
            if _depth_change(body[start:cand.start()]) != 0:
                continue
            try:
                decoder.raw_decode(body, cand.end())
            except ValueError:
                continue
            nxt = cand.start()
            break
        if nxt is None:
            break
        pos = nxt
    return readable, bad, reason, False


def _readable_json_keys(text: str) -> tuple[dict[str, Any], list[tuple[str, ...]], str] | None:
    """What of ONE JSON object that does not parse as a whole can still be read, key by key
    (:func:`_read_object_keys`): ``(readable, key paths that could not be read, the parser's reason)``. ``None`` when
    nothing is readable, when the reply holds more than one object (which one is the answer is then not known), or
    when a key is written twice (which value is meant is not known): the caller then asks for the whole answer."""
    if not (text or "").strip() or len(_top_level_json_objects(text)) > 1:
        return None
    body = _strip_outer_fence(text).strip()
    start = body.find("{")
    if start < 0:
        return None
    readable, bad, reason, twice = _read_object_keys(body, start + 1)
    if twice or not readable:
        return None
    return readable, bad, reason


def _matching_brace_end(text: str, start: int) -> int:
    """The index just past the ``}`` that closes the ``{`` at ``start`` (braces inside JSON strings not counted), or
    the end of ``text`` when it is never closed."""
    depth = 0
    in_string = escaped = False
    for k in range(start, len(text)):
        ch = text[k]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return len(text)


_FENCED_BLOCK_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n(.*?)```", re.DOTALL)


def _object_spans(text: str) -> tuple[list[tuple[int, int, dict[str, Any]]], bool]:
    """``(spans, doubt)``: ``(start, end, object)`` of every non-empty JSON object that parses at the top level of ``text``,
    in order, and whether an object-like ``{`` failed to parse (a reply cut short, a raw newline in a string). An object
    that does not parse is skipped whole, so one inside it is never taken for the reply's own."""
    decoder = json.JSONDecoder()
    out: list[tuple[int, int, dict[str, Any]]] = []
    cut = False
    i = text.find("{")
    while i >= 0:
        try:
            obj, end = decoder.raw_decode(text, i)
        except (ValueError, RecursionError):
            if text[i + 1:].lstrip()[:1] in ('"', "}"):
                closes = _matching_brace_end(text, i)
                repaired = _read_latex_object(text[i:closes])
                if repaired:
                    out.append((i, closes, repaired))
                else:
                    cut = True  # an object that does not parse, closed or not: which one the reply means is in doubt
                i = text.find("{", closes)
            else:
                i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and obj:
            out.append((i, end, _read_latex_object(text[i:end], strict=True) or obj))
        i = text.find("{", end)
    return out, cut


def _object_among_prose(text: str) -> dict[str, Any] | None:
    """The JSON object a reply means when prose around it holds braces of its own, taken only where the reply leaves no
    doubt which object that is, else ``None``. Called only where the whole-text parse and the first-``{``-to-last-``}``
    slice both failed, so a reply that parsed before is read exactly as it was. Refused: a reply cut short (an odd number
    of fences, or an object opened that never closes). Taken: the one object that is the whole content of a fenced block;
    else the one object the whole reply holds (not one inside a list). Never "the last of several"."""
    log = logging.getLogger("frontier_insight.engine")
    if text.count("```") % 2:
        return None
    spans, cut = _object_spans(text)
    if cut or not spans:
        return None
    whole = []
    for m in _FENCED_BLOCK_RE.finditer(text):
        body = m.group(1).strip()
        objs, _ = _object_spans(body)
        if len(objs) == 1 and objs[0][0] == 0 and objs[0][1] == len(body):
            whole.append(objs[0][2])
    if len(whole) == 1:
        chosen, how = whole[0], "the one object that is a whole fenced block"
    elif len(whole) == 0 and len(spans) == 1:
        start, end, chosen = spans[0]
        if text[:start].rstrip()[-1:] in ("[", ",") or text[end:].lstrip()[:1] == "]":
            return None
        how = "the one object the reply holds"
    else:
        return None
    log.info("a model's JSON reply held prose with braces around its object; read %s (keys: %s)",
             how, ", ".join(list(chosen)[:6]))
    return chosen


def _parse_json_lenient_once(
    text: str, *, node: str = "", _log_truncate_chars: int = 500, _quiet: bool = False,
    _prose: bool = True,
) -> dict[str, Any] | None:
    """Find and parse a JSON object inside arbitrary LLM output.

    LLMs frequently wrap JSON in markdown fences, prose, or trailing
    commentary. This tries strict parse first; on failure, slices from
    the first `{` to the last `}` and parses that. The slice is NOT
    balance-aware — if the LLM emits two top-level objects in one
    response, the slice will span both and parsing will fail.

    When parsing definitively fails (both the strict parse AND the
    fence-slice fallback give up, AND the input had content to parse),
    we log a WARNING with the raw text truncated to
    ``_log_truncate_chars`` characters. Callers typically use the
    ``parsed or {fallback}`` idiom to keep the quest running, but the
    fallback values are dummies (``"(parse failed)"``) — without this
    log line, a developer debugging a prompt change has no way to see
    what the model actually emitted. The optional ``node`` kwarg is
    included verbatim in the log line so the message identifies which
    engine node's JSON broke.

    Empty input and non-dict JSON (case where ``json.loads`` returns a
    list/string/number) return None silently — those are edge cases,
    not bugs in the model's output.
    """
    if not text:
        return None
    candidate = _strip_outer_fence(text).strip()
    # Treat "whitespace-only" and "empty-fence-only" inputs the same
    # as truly empty input — return silently, don't fire a WARNING.
    # Without this guard, ``text = "   "`` or ``text = "```\n```"``
    # passes the ``not text`` check above, collapses to "" here, then
    # falls through to the "no braces found" warning path with an
    # empty raw-output snippet — pure log spam, no signal.
    if not candidate:
        return None
    try:
        result = json.loads(candidate)
        # Successful parse but wrong shape: return None without a
        # WARNING — the JSON itself was valid, the prompt told the
        # model to return an object, the contract is upstream of
        # this function.
        # `\text`, `\frac`, `\beta`, `\nabla`, `\rho` are valid JSON escapes, so LaTeX written with single backslashes can
        # parse as a tab or a form feed: read as the LaTeX it is.
        return (_read_latex_object(candidate, strict=True) or result) if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    # Find the first '{' and last '}' and try the slice.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            result = json.loads(candidate[start : end + 1])
            return (_read_latex_object(candidate[start : end + 1], strict=True) or result) if isinstance(result, dict) else None
        except json.JSONDecodeError:
            repaired = _read_latex_object(candidate[start : end + 1])
            if repaired is not None:
                return repaired
            # Prose around the object that quotes braces of its own (`a case such as {"threshold": 0.01}`) makes the
            # first-to-last slice span the prose too: read the objects written at the top level instead.
            chosen = _object_among_prose(candidate) if _prose else None
            if chosen is not None:
                return chosen
            if not _quiet:
                _log_parse_failure(candidate, node, _log_truncate_chars)
            return None
    # No braces found at all — the model didn't return JSON-ish output.
    # That's just as much a parse failure as the slice-attempt-failed
    # case above, and worth logging at the same WARNING level.
    if not _quiet:
        _log_parse_failure(candidate, node, _log_truncate_chars)
    return None


def _log_parse_failure(text: str, node: str, max_chars: int) -> None:
    """Emit one WARNING line when JSON parsing of an LLM response
    definitively fails. Truncated raw text helps a developer compare
    the model's output against the prompt schema without scrolling
    through gigabytes of run.log.

    Uses the package-level ``frontier_insight.engine`` logger rather
    than a per-quest logger because ``_parse_json_lenient`` is a
    free function called from many places — threading a logger handle
    through every call site would touch 12+ lines and isn't worth the
    surface area. The per-quest run.log file inherits from this
    logger via ``logging.basicConfig``-style propagation, so the
    warning still lands in the right run.log."""
    snippet = text[:max_chars]
    if len(text) > max_chars:
        snippet += f"… [+{len(text) - max_chars} chars truncated]"
    node_tag = f" node={node}" if node else ""
    logging.getLogger("frontier_insight.engine").warning(
        "JSON parse failed in _parse_json_lenient%s; falling back to "
        "the caller's default. Raw LLM output (%d chars): %r",
        node_tag, len(text), snippet,
    )


# Implement-node response parsing.
#
# Why a dedicated parser: the prior `{"code": "...", "deps": [...]}`
# JSON-wrapped format was the worst possible shape for an LLM stream —
# a 150-line script became a 6-10 KB single JSON string with every
# newline / quote / backslash escaped, costing ~30% more output tokens
# AND forcing the whole response to be well-formed (any truncation
# silently fell back to `print("RESULT_JSON: {}")` and crashed execute).
# Long streams are exactly where Copilot's HTTP/2 drops happen, so the
# `implement` node accounted for most of the bridge retry-exhaustions
# users saw in real quests. The new format is a fenced Python block
# plus a `DEPS:` line — partial truncation still yields recoverable code.
_PY_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)
_DEPS_LINE_RE = re.compile(
    r"^[ \t]*deps\s*[:=]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _coerce_dep_list(value: Any) -> list[str]:
    """Normalize a deps value from various LLM-output shapes to ``list[str]``.

    Accepts ``list[str]`` (canonical), a comma-separated string
    (``"numpy, scipy"``), a single bare name (``"numpy"``), or
    anything else (returns ``[]``). Filters out empty / whitespace-only
    entries and stringifies non-str elements as a last resort.

    Returning a real ``list[str]`` is important because the caller
    set-unions with ``{*deps, *design_deps}`` — unpacking a bare string
    into a set yields per-character entries ({"n","u","m","p","y"}).
    """
    if isinstance(value, list):
        return [str(d).strip() for d in value if str(d).strip()]
    if isinstance(value, str):
        return [d.strip() for d in value.split(",") if d.strip()]
    return []


def _deps_from_tail(tail: str) -> list[str]:
    """The packages of the first ``DEPS:`` line in ``tail`` (the text after the last
    fenced block, so a ``deps = [...]`` inside the code is never read)."""
    deps_match = _DEPS_LINE_RE.search(tail)
    if not deps_match:
        return []
    raw = deps_match.group(1).strip()
    # Tolerate "numpy, matplotlib" / "[numpy, matplotlib]" /
    # "['numpy', 'matplotlib']" — peel exactly ONE matched pair
    # of outer brackets, not every leading/trailing bracket.
    # The naive `.strip("[](){}")` would chew the trailing `]`
    # off PEP 508 extras like `pandas[performance]`, leaving a
    # broken spec `pandas[performance` that pip can't install.
    for opener, closer in (("[", "]"), ("(", ")"), ("{", "}")):
        if raw.startswith(opener) and raw.endswith(closer):
            raw = raw[1:-1].strip()
            break
    return [
        d.strip().strip("'\"")
        for d in raw.split(",")
        if d.strip().strip("'\"")
    ]


def _package_block_text(files: dict[str, str]) -> str:
    return ("\n\n").join(f"### code/{rel}\n```python\n{text}\n```" for rel, text in files.items())


def _parse_split_deps(text: str) -> list[str]:
    """The ``DEPS:`` line of a reply that holds two fenced blocks: read after the last one."""
    last = None
    for last in _PY_FENCE_RE.finditer(text or ""):
        pass
    return _deps_from_tail(text[last.end():] if last else text or "")


def _parse_implement_response(text: str) -> tuple[str, list[str]]:
    """Extract ``(code, deps)`` from the implement-node LLM response.

    Format expected (the new shape after agents/implement.md was rewritten):

        ```python
        <code>
        ```
        DEPS: numpy, matplotlib

    Falls back to the legacy ``{"code": ..., "deps": [...]}`` JSON shape
    if no fenced code block is found, so a model that drifts back to the
    old format still works.

    Returns ``("", [])`` if neither shape parses — caller is expected to
    handle the empty-code path (it writes a stub experiment and lets the
    execute node fail loudly rather than silently swallow the breakage).
    """
    if not text:
        return "", []

    # Primary: fenced Python block + `DEPS:` line.
    fence = _PY_FENCE_RE.search(text)
    if fence:
        code = fence.group(1).strip("\n")
        deps: list[str] = []
        # Search the DEPS line only in the AFTER-fence tail. Searching
        # the whole text would falsely match Python statements like
        # `deps = [...]` INSIDE the fenced experiment code itself (the
        # prompt explicitly puts DEPS after the closing ```).
        deps = _deps_from_tail(text[fence.end():])
        return code, deps

    # Fallback: legacy JSON-wrapped shape.
    legacy = _parse_json_lenient(text) or {}
    code = legacy.get("code") or ""
    deps = _coerce_dep_list(legacy.get("deps"))
    return code, deps


_RESULT_LINE_RE = re.compile(r"RESULT_JSON:\s*(\{.*\})\s*$", re.MULTILINE)
_RESULT_MARKER_RE = re.compile(r"RESULT_JSON:\s*")


def _extract_result_json(stdout: str) -> dict[str, Any] | None:
    if not stdout:
        return None
    # Fast path: single-line ``RESULT_JSON: {...}`` (the format the prompt asks
    # for). Take the LAST such marker so a replicate run's final line wins.
    matches = list(_RESULT_LINE_RE.finditer(stdout))
    if matches:
        try:
            return json.loads(matches[-1].group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: the marker is present but the JSON spans MULTIPLE lines — the
    # common case where the script used ``json.dumps(..., indent=2)``. The
    # single-line regex above (MULTILINE, no DOTALL) can't span newlines, so
    # brace-balance the first complete JSON object after the last marker. This
    # keeps a perfectly good experiment from being scored ``result_json=False``
    # purely over pretty-printing.
    marker_hits = list(_RESULT_MARKER_RE.finditer(stdout))
    if not marker_hits:
        return None
    tail = stdout[marker_hits[-1].end():]
    start = tail.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(tail)):
        c = tail[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(tail[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# Char budget for the ``result_json`` block spliced into the analyze prompt.
# The full result_json stays in state (it feeds results.csv at full fidelity);
# only the *prompt copy* is bounded. A pathological experiment can dump raw
# arrays/images into its RESULT_JSON — one low-k1 DUV quest emitted a 36 MB
# result_json, which blew past codex_cli's 1,048,576-char turn limit and
# hard-failed analyze. 200 KB is generous for genuine summary statistics
# (real ones are a few KB) while staying well under any CLI input cap, so
# the rest of the analyze prompt (design, framing, figure list) still fits.
_ANALYZE_RESULT_JSON_BUDGET_CHARS = 200_000


def _shrink_json_value(value: Any, *, list_keep: int, str_keep: int) -> Any:
    """Recursively shrink the bulky parts of a JSON-able value so it serializes
    smaller while staying VALID, representative JSON (not a mid-structure hard
    cut). Long lists keep ``list_keep`` head + ``list_keep`` tail elements with
    a count-of-elided marker between them; long strings are truncated with a
    char-count suffix. Dict keys and scalar summary stats are preserved — those
    are what analyze actually reasons over; raw arrays are the disposable bulk."""
    if isinstance(value, dict):
        return {k: _shrink_json_value(v, list_keep=list_keep, str_keep=str_keep)
                for k, v in value.items()}
    if isinstance(value, list):
        if len(value) <= list_keep * 2:
            return [_shrink_json_value(v, list_keep=list_keep, str_keep=str_keep)
                    for v in value]
        head = [_shrink_json_value(v, list_keep=list_keep, str_keep=str_keep)
                for v in value[:list_keep]]
        tail = [_shrink_json_value(v, list_keep=list_keep, str_keep=str_keep)
                for v in value[-list_keep:]]
        elided = len(value) - 2 * list_keep
        return head + [f"... {elided} more elements elided to fit prompt budget ..."] + tail
    if isinstance(value, str) and len(value) > str_keep:
        return value[:str_keep] + f"... ({len(value) - str_keep} chars elided)"
    return value


def _compact_result_json_block(
    data: dict[str, Any], budget_chars: int = _ANALYZE_RESULT_JSON_BUDGET_CHARS,
) -> tuple[str, int]:
    """Serialize ``data`` to indented JSON bounded to ~``budget_chars``.

    Returns ``(block, original_chars)`` — ``original_chars`` is the size of the
    untrimmed dump (so the caller can log when compaction actually fired). If
    the full dump already fits, it is returned unchanged. Otherwise the bulky
    arrays/strings are progressively shrunk (tighter on each pass) until the
    dump fits, with a final hard cap as a last resort so the return value can
    never exceed the budget."""
    full = json.dumps(data, indent=2, ensure_ascii=False)
    if len(full) <= budget_chars:
        return full, len(full)
    for list_keep, str_keep in ((20, 2000), (8, 800), (4, 400), (2, 200), (1, 100)):
        shrunk = _shrink_json_value(data, list_keep=list_keep, str_keep=str_keep)
        out = json.dumps(shrunk, indent=2, ensure_ascii=False)
        if len(out) <= budget_chars:
            return out, len(full)
    return out[:budget_chars], len(full)


def _skill_mount_plan(config: Any, usable: list[Any]) -> "Any | None":
    """Where each approved external skill among ``usable`` is mounted in the
    Docker sandbox (``core.skills.mounts``), or None when the experiment does
    not run in Docker: every other sandbox sees the host's own paths."""
    if config.execution.sandbox != "docker":
        return None
    from core.skills.mounts import plan_mounts

    return plan_mounts(usable)


def _discover_skill_names(external_dirs: Any = None) -> list:
    """Every skill this quest can see, for the "candidates = all trusted"
    default — those on this machine plus the folders of other agents the quest
    itself names (``external_dirs``, its own ``ExternalSkillDirs``).

    Returns Skill objects; the caller re-resolves them through
    ``loadable_skills`` so the promotion gate runs exactly once, in one place.
    Never raises — a broken skills directory must not stall a quest.
    """
    try:
        from core.skills import discover

        return discover(external_dirs=external_dirs)
    except Exception:  # noqa: BLE001
        return []


def _raise_if_required_skills_unusable(
    required: list[str], usable: list[Any], rejected: list[Any],
) -> None:
    """Naming a skill in ``engine.skills_required`` is an instruction, so one
    that cannot be used stops the quest with the reason, rather than being
    skipped the way an unusable candidate is."""
    if not required:
        return
    ok = {st.skill.name for st in usable}
    by_name = {st.skill.name: st for st in rejected}
    problems: list[str] = []
    for name in required:
        if name in ok:
            continue
        st = by_name.get(name)
        if st is None or st.skill.source == "missing":
            problems.append(f"{name} (no skill by that name was found)")
        else:
            problems.append(f"{name} ({st.status.value}: {st.reason})")
    if problems:
        raise RuntimeError(
            "engine.skills_required names skill(s) that cannot be used: "
            + "; ".join(problems)
            + ". An external or newly written skill is approved one at a time: "
            "python launch.py --approve-skill <name> --approve-as <you> "
            "(--skills lists them and their status). For a skill that lives in a "
            "folder this quest's engine.skills_dirs names, add --config <this "
            "quest's YAML> to those commands. Or remove the name."
        )


def _resolve_selected_skills(
    state: QuestState | None, log: Any, *, use: str, external_dirs: Any = None,
) -> tuple[list[Any], dict[str, str]]:
    """The selected skills for one ``use`` that are still usable, and why
    each was selected.

    ``use`` is what selection said the skill is for: ``"experiment"`` for
    design and the implement stages, ``"writing"`` for the writer. A skill
    with no recorded use is an experiment skill, as every skill was before
    selection recorded one. ``external_dirs`` is the quest's own
    ``ExternalSkillDirs``: a selected skill is looked up where this quest looks,
    not where another quest in the process does. Never raises: a broken
    registry leaves the quest generating its own code, as before skills.
    """
    selection = (state or {}).get("skill_selection") or {}
    uses = dict(selection.get("uses") or {})
    names = [
        n for n in list((state or {}).get("selected_skills") or [])
        if uses.get(n, "experiment") == use
    ]
    if not names:
        return [], {}
    # The selection's reasons travel with it: design sees why each skill
    # was picked and may decline one it judges inapplicable.
    reasons = dict(selection.get("reasons") or {})

    try:
        from core.skills import loadable_skills
    except Exception as e:  # noqa: BLE001 - a broken registry must not stall a quest
        log.warning("[skills] registry unavailable (%s); generating instead", e)
        return [], {}

    try:
        usable, rejected = loadable_skills(names, external_dirs=external_dirs)
    except Exception as e:  # noqa: BLE001
        log.warning("[skills] resolution failed (%s); generating instead", e)
        return [], {}

    for rej in rejected:
        # Selection only ever names trusted skills, so reaching here means
        # a skill broke between selection and use — its tool moved, or its
        # package upgraded. Skip it and say so; a skill is additive, never
        # a precondition.
        log.warning(
            "[skills] %s selected but no longer usable (%s) — %s",
            rej.skill.name, rej.status.value, rej.reason,
        )
    return list(usable), reasons


def _trial_code(code_dir: Path) -> str:
    """The code that makes the trial results: the study's scripts and package files (``grid_keys.code_files``: no tests, no
    file of FI's that is unchanged) without FI's analysis (experiment.py, derive.py, the table reader and the figure step)."""
    fi_files = {_code_layout.ANALYSIS_SCRIPT, _code_layout.TABLE_SCRIPT, _code_layout.FIGURE_CALLS_SCRIPT,
                _analysis_frame.DERIVE_FILE, "fi_frame.py",
                f"{_analysis_frame.FIGURES_MODULE}.py"}
    return "\n".join(text for rel, text in _grid_keys.code_files(code_dir).items()
                     if rel not in fi_files and not rel.startswith(f"{_code_layout.FI_CORE_FOLDER}/"))


def _checked(state: "QuestState", memo: Any = None) -> list:
    """The findings of the range check. With FI's analysis frame a trial result is checked against the trial code and a
    derived one against derive.py (``_in_force`` sets both); otherwise every result against ``state["code"]``."""
    from core import plausibility

    rj = state.get("result_json") or {}
    design = _asserting_design(state)
    args = dict(zero_expected=_zero_expected_paths(state), edge_ok=_physical_bound_paths(state))
    seen = list(state.get("bounded_seen") or ())
    trial, derived = state.get("_trial_code"), set(state.get("_derived_ids") or ())
    if trial is None:
        return plausibility.check_design(rj, design, code=state.get("code") or "", seen=seen,
                                         walk=memo.get(rj) if memo is not None else None, **args)

    def of_derived(path: str) -> bool:
        return any(seg in derived for seg in plausibility._segments(path))

    def path_of(a: Any) -> str:
        return str(a.get("path", "") if isinstance(a, dict) else getattr(a, "path", ""))

    asserted = list(design.get("result_assertions") or [])
    out: list = []
    for code, mine in ((state.get("code") or "", True), (trial, False)):
        part = {**design, "result_assertions": [a for a in asserted if of_derived(path_of(a)) == mine]}
        out += plausibility.check_design(rj, part, code=code, seen=[p for p in seen if of_derived(str(p)) == mine],
                                         walk=memo.get(rj) if memo is not None else None, **args)
    return out


# Findings of the plausibility check that are said, never acted on: a cap whose place in the result cannot be shown, a
# physical_bound result on its edge everywhere, and a result on a bound the plan did not declare honest (a guess about
# the computation, so it is disclosed in the paper and does not send the code back).
_WARNING_KINDS = ("clamped_unproven", "edge_everywhere", "at_bound")


def _assertion_violations(state: "QuestState", memo: Any = None) -> list:
    """Range/unit assertions the run broke, as declared by its own design.

    Returns [] when the design declared nothing — silence means "nothing was
    claimed", never "everything checked out". Never raises: a defect in the
    checker must not stall a quest, so any failure degrades to "no violations"
    and lets the run proceed.

    Separate from ``_is_degenerate_result`` on purpose. That one catches a run
    that produced nothing; this one catches a run that produced the wrong
    thing convincingly.
    """
    try:
        from core import plausibility

        # The script rides along so a value capped exactly at a non-zero
        # bound is caught too; ``seen`` so a bounded quantity an earlier run
        # reported cannot quietly disappear. See core/plausibility.py for both.
        found = _checked(state, memo)
        # A cap whose place in the result cannot be shown, and a physical_bound result on its edge in every setting, are
        # warnings (_assertion_warnings), not violations.
        return [v for v in found if getattr(v, "kind", "") not in _WARNING_KINDS]
    except Exception:  # noqa: BLE001 - a checker bug must not block a quest
        return []


def _assertion_warnings(state: "QuestState", memo: Any = None) -> list:
    """The findings of the same check that are warnings, not violations: a value exactly on a constant the script
    caps a quantity of that name at, where that cap's place in the result cannot be shown
    (``plausibility.direct_caps(..., proven=False)``). Logged and recorded in the trace; they never send a run back."""
    try:
        from core import plausibility

        found = _checked(state, memo)
        return [v for v in found if getattr(v, "kind", "") in _WARNING_KINDS]
    except Exception:  # noqa: BLE001
        return []


def _zero_expected_paths(state: "QuestState") -> set[str]:
    """The asserted quantities whose 0 is the answer, not a trivial one: a count of failures, known as one because the
    protocol's ``failure_policy`` names it, and not contradicted by the run's own manifest (it lists no failed trial, or
    the run kept none). A live quest reported ``failed_run_count`` = 0 in every setting, its manifest agreed, and the
    at-bound rule sent a correct script back twice, until the repair replaced the zeros with nulls."""
    protocol = (state.get("design") or {}).get("protocol")
    policy = str((protocol or {}).get("failure_policy") or "") if isinstance(protocol, dict) else ""
    metrics = (protocol or {}).get("metrics") if isinstance(protocol, dict) else None
    # A result that declares a role says it there (physical_bound), not in the prose of the failure policy.
    roled = {str(m["id"]) for m in metrics if isinstance(m, dict) and m.get("role") and m.get("id")} \
        if isinstance(metrics, list) else set()
    failed = state.get("manifest_failed_trials")
    # The zero must be corroborated: a manifest was read and it lists no failed trial.
    if not policy.strip() or failed != 0 or isinstance(failed, bool):
        return set()
    from core import plausibility

    out: set[str] = set()
    for a in plausibility.parse_assertions(_asserting_design(state)):
        leaf = a.path.rsplit(".", 1)[-1]
        if not leaf or leaf in roled:
            continue
        # The policy names the quantity as a name: an identifier (`failed_run_count`) as a whole word, or any name in
        # quotes. A bare word ("rate", "count") in the policy's prose names nothing.
        quoted = re.search(rf"[`'\"]{re.escape(leaf)}[`'\"]", policy)
        identifier = re.search(r"[_\d]", leaf) and re.search(rf"(?<![\w]){re.escape(leaf)}(?![\w])", policy)
        if quoted or identifier:
            out.add(a.path)
    return out


def _physical_bound_paths(state: "QuestState") -> set[str]:
    """The asserted quantities whose result the plan declares with the role ``physical_bound`` (:mod:`core.contract`): such
    a value may sit exactly on the edge of its range, which is its physics and not a fault."""
    design = _asserting_design(state)
    protocol = design.get("protocol")
    if not isinstance(protocol, dict):
        return set()
    from core import plausibility

    physical = {i for i, role in _contract.raw_roles(protocol).items() if role == "physical_bound"}
    return {a.path for a in plausibility.parse_assertions(design) if any(plausibility._matches(i, a.path) for i in physical)}


def _asserting_design(state: "QuestState") -> dict[str, Any]:
    """The design with a called skill's own range assertions added.

    A skill knows its own valid domain, so a design that calls one need not
    restate its bounds. Design assertions come first: a design that
    deliberately narrows a skill's range keeps its say."""
    design = dict(state.get("design") or {})
    extra = state.get("_skill_assertions") or []
    if extra:
        design["result_assertions"] = list(design.get("result_assertions") or []) + list(extra)
    return design


def _bounded_seen_after(state: "QuestState", result_json: dict[str, Any], memo: Any = None) -> list[str]:
    """``bounded_seen`` once this result is counted: what was seen before, plus what it reports."""
    seen = list(state.get("bounded_seen") or [])
    try:
        from core import plausibility

        for path in plausibility.bounded_paths(
                result_json, _asserting_design(state), walk=memo.get(result_json) if memo is not None else None):
            if path not in seen:
                seen.append(path)
    except Exception:  # noqa: BLE001 - bookkeeping must not block a quest
        pass
    return seen


def _is_degenerate_result(rj: dict[str, Any]) -> bool:
    """True when a RESULT_JSON has numeric metrics that are ALL ~0.

    A script can exit 0 yet still be broken — e.g. a lithography
    aerial-image sim that reports ``contrast=NILS=CD=...=0`` because the
    grid missed the diffraction orders, or a threshold/edge finder that
    returns a zero sentinel when it finds no crossing. We flag
    ">=2 numeric leaves, every one within 1e-12 of zero" as degenerate so
    the execute-repair loop can fix the bug before a paper is written.

    Deliberately conservative to avoid false positives: any non-zero
    value, a NaN, or fewer than two numeric leaves → not degenerate.
    Booleans are not counted as metrics. Walks nested dicts and lists.
    """
    if not isinstance(rj, dict) or not rj:
        return False
    for k in ("degenerate", "_degenerate"):
        val = rj.get(k)
        if val is False or (isinstance(val, str) and val.lower() == "false"):
            return False
    nums: list[float] = []

    def _walk(v: Any) -> None:
        if isinstance(v, bool):
            return  # a flag, not a metric
        if isinstance(v, (int, float)):
            nums.append(float(v))
        elif isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _walk(x)

    # FI's own numbers of a search for the best design (core/optimise.py::with_fi_record) are not the script's result.
    _walk({k: v for k, v in rj.items() if k != "fi_search"})
    if len(nums) < 2:
        return False
    return all(abs(n) <= 1e-12 for n in nums)


def _deps_to_warmup_modules(deps: list[str], installed: dict[str, list[str] | None] | None = None) -> str:
    """The modules to import in the test import after installing, as ``a, b, c`` for ``python -c "import ..."``.
    Only names that are known (core/experiment_deps.py: the package-name table, or ``installed``, the environment's
    record of what each package provides); an unknown package is left out rather than imported under a guess."""
    return ", ".join(_experiment_deps.warmup_modules(deps, installed))


_UNSANCTIONED_PROXY_PROVIDERS = frozenset({
    "github_copilot_cli",
    "github_copilot_vscode",
})

_PROXY_WARN_SHOWN: set[str] = set()


def _warn_if_unsanctioned_provider(name: str) -> None:
    """Print a one-time warning (per process) when the user picks a
    provider that is risky for FI's pipeline: `github_copilot_cli` and
    `github_copilot_vscode` (`_UNSANCTIONED_PROXY_PROVIDERS`) route through
    a third-party reverse-engineered `copilot-api` proxy. The proxy's own
    README warns: "Excessive automated or scripted use of Copilot ... may
    trigger GitHub's abuse-detection systems." Premium-request volume from
    an automated research loop is exactly the pattern that trips that
    detector. (`copilot_cli` does not warn: FI calls it answer-only, with
    every tool off and the prompt on stdin.)

    Suppression: each provider warns once per process; set the
    ``FI_SUPPRESS_PROXY_WARN`` env var to silence entirely.
    """
    import os
    if os.environ.get("FI_SUPPRESS_PROXY_WARN"):
        return
    if name in _PROXY_WARN_SHOWN:
        return

    if name in _UNSANCTIONED_PROXY_PROVIDERS:
        _PROXY_WARN_SHOWN.add(name)
        logging.getLogger("frontier_insight").warning(
            "\n%s\n"
            "  provider=%r uses the third-party `copilot-api` proxy, which is\n"
            "  NOT officially supported by GitHub and may violate the GitHub\n"
            "  Copilot acceptable-use policy under heavy automation. From\n"
            "  copilot-api's own README:\n"
            "    \"Excessive automated or scripted use of Copilot ... may\n"
            "     trigger GitHub's abuse-detection systems. You may receive\n"
            "     a warning from GitHub Security, and further anomalous\n"
            "     activity could result in temporary suspension of your\n"
            "     Copilot access.\"\n"
            "\n"
            "  Sanctioned alternatives for Copilot in FI:\n"
            "    - provider.name: vscode_extension  (in-VSCode via vscode.lm.*)\n"
            "    - provider.name: claude_cli / codex_cli / gemini_cli (OAuth CLIs)\n"
            "\n"
            "  Set FI_SUPPRESS_PROXY_WARN=1 to silence this warning.\n"
            "%s",
            "─" * 72, name, "─" * 72,
        )
        return


def _new_quest_id(seed: str) -> str:
    base = _slugify(seed)[:32] or "quest"
    return f"{int(time.time())}-{base}-{uuid.uuid4().hex[:6]}"


# Public alias for callers outside the engine module that need to
# mint a quest_id BEFORE constructing an Engine (e.g. the
# `--serve` web UI's quest launcher needs the id up-front so the
# post-submit redirect URL `/quest/<id>` is stable). Forwards to the
# internal `_new_quest_id`; the rename lets external callers stop
# depending on a underscore-prefixed private symbol.
def mint_quest_id(seed: str) -> str:
    """Generate a new quest_id from a topic / title / slug seed.
    Same algorithm `Engine.__init__` uses when no explicit
    `resume_quest_id` or `FI_PRESEED_QUEST_ID` is provided."""
    return _new_quest_id(seed)


def _slugify(s: str) -> str:
    """Quest-id slug. ASCII-only by contract — the quest_id regex used
    by digest / critique / --resume / interview_update is
    ``^\\d{10}-[a-z0-9-]+-[0-9a-f]{6}$``, so a non-ASCII slug would
    create a quest directory that those downstream paths can't see.

    Policy for a topic in a non-Latin script (Traditional Chinese,
    Cyrillic, ...):
    1. Lowercase + extract the ASCII-letter / digit runs (handles
       mixed-script topics like ``Genetic 遺傳 impact`` → ``genetic-impact``).
    2. If nothing ASCII survives, fall back to a stable 8-hex digest of
       the original string so every distinct CJK topic still gets a
       distinct quest_id (instead of every CJK quest colliding on the
       constant ``"untitled"``).

    For ``interview.slugify`` (used for human-readable YAML titles +
    folder names downstream), the Unicode-letter policy is the right
    one — that helper keeps CJK intact.
    """
    s = s.strip().lower()
    # ASCII-only character class — re.UNICODE is irrelevant here.
    ascii_only = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if ascii_only:
        return ascii_only
    # No ASCII letters/digits survived. Hash-fallback only when the
    # original input carries at least one Unicode letter/digit —
    # otherwise pure-punctuation / empty input still produces
    # ``untitled`` (the long-standing fallback the tests pin).
    if re.search(r"\w", s, flags=re.UNICODE):
        import hashlib
        digest = hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
        return f"i18n-{digest}"
    return "untitled"


def _save_literature_figures(figures: Any, folder: Path, stem: str) -> list[dict[str, Any]]:
    """Copy a source's figures (cut from its PDF by core/pdf_figures.py, kept in a cache) into ``folder`` as
    ``<stem>_fig<N>_p<page>.png`` for people to look at. Returns the figures copied, each with ``file`` (its name in
    ``folder``); one that cannot be copied is left out."""
    out: list[dict[str, Any]] = []
    for fig in figures if isinstance(figures, list) else []:
        if not isinstance(fig, dict) or not fig.get("image"):
            continue
        name = f"{stem}_fig{fig.get('number')}_p{fig.get('page')}.png"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(fig["image"]), folder / name)
        except OSError:
            continue
        fig["file"] = name
        out.append(fig)
    return out


def _figures_section(figures: list[dict[str, Any]]) -> str:
    """The ``## Figures`` section of a literature file: each figure's caption and where its image is."""
    lines = ["## Figures", ""]
    for fig in figures:
        where = " (from a scanned page, read by OCR)" if fig.get("scanned") else ""
        lines.append(f"- **Figure {fig.get('number')}**, page {fig.get('page')}{where}: "
                     f"`figures/{fig['file']}`  ")
        lines.append(f"  {str(fig.get('caption') or '').strip()}")
    return "\n".join(lines)


#: Figures sent to the model in one call, and the most kept of what it read off one figure.
_FIGURES_PER_CALL = 4
#: Captions shown to the picker in one call (thirty figure-heavy papers can hold over a thousand).
_FIGURE_PICK_BATCH = 150
_FIGURE_READING_CHARS = 2000
#: What a model read off one source's figures, shown to the claim check beside the source's own text.
_CLAIM_READINGS_CHARS = 1500
_FIGURE_READINGS_MARK = "---VALUES READ FROM THE FIGURES (by a model, from the images)---"
_NL = chr(10)
_PARA = _NL + _NL


def _split_figure_readings(text: str) -> tuple[str, str]:
    """``(the source's own text, what a model read off its figures)``. The readings are appended to a source's text for
    the design, the analysis and the writer to read (:func:`_append_figure_readings`); they are a model's observation,
    not the source's words, so nothing that checks a quotation against a source may take them for its text."""
    if _FIGURE_READINGS_MARK not in text:
        return text, ""
    head, _, tail = text.partition(_FIGURE_READINGS_MARK)
    readings = [b.strip() for b in tail.split(_FIGURE_READINGS_MARK) if b.strip()]
    return head.rstrip(), _PARA.join(readings)


def _append_figure_readings(item: dict[str, Any]) -> int:
    """Append what was read off a source's figures to its text (in the state, and in its whole text on disk), each
    under its caption, so the design, the analysis and the writer read them like the rest of the paper. A figure is
    appended once (``in_text``)."""
    meta = item.get("metadata") or {}
    new = [f for f in meta.get("figures") or [] if isinstance(f, dict) and f.get("reading") and not f.get("in_text")]
    if not new:
        return 0
    blocks = [
        f"Figure {f.get('number')} (page {f.get('page')}): {str(f.get('caption') or '').strip()[:600]}{_NL}"
        f"Read from the image: {f['reading']}"
        for f in new
    ]
    block = _PARA.join(blocks)
    text = f"\n\n{_FIGURE_READINGS_MARK}\n\n{block}\n"
    item["content"] = str(item.get("content") or "") + text
    path = meta.get("full_text_path")
    if path:
        try:
            # A resume after a stop part-way through re-applies readings already appended to the file: never twice.
            on_disk = Path(path).read_text(encoding="utf-8")
            fresh = [b for b in blocks if b.strip() not in on_disk]
            if fresh:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(_PARA + _FIGURE_READINGS_MARK + _PARA + _PARA.join(fresh) + _NL)
        except OSError:
            pass  # the state's copy has them, and _item_content adds them back
    for f in new:
        f["in_text"] = True
    return len(new)


def _render_auto_collected_md(idx: int, meta: dict[str, Any], content: str) -> str:
    """Render an Axon retrieval hit as a Markdown file with YAML front
    matter. Used by ``_node_auto_collect_data``.

    Why a proper YAML dump (not Python ``repr``): a metadata value
    containing single quotes (``"O'Brien"``), backslashes, or non-ASCII
    characters would be backslash-escaped by ``repr`` and the resulting
    string would be unsafe to round-trip through a YAML parser. The
    data_load node downstream may decide to parse the front matter for
    provenance; if it can't, the user loses cite-back fidelity. Using
    ``yaml.safe_dump`` produces a guaranteed-parseable block regardless
    of the metadata content.
    """
    front: dict[str, Any] = {"auto_collected": True, "rank": idx}
    # Render every non-empty metadata key the caller passed — caller
    # decides what's provenance-worthy (Axon docs use source/path/
    # title/url/kind/year; dataset adapters add adapter/indicator_id/
    # countries/score/etc.). Dropping unknown keys here would silently
    # eat dataset-adapter provenance.
    #
    # ALL values are coerced to YAML scalars (str/int/float/bool):
    # this is a front-matter renderer, not a deep-config dump. A
    # ``list`` or ``dict`` in metadata would otherwise produce
    # nested YAML that changes the shape downstream consumers
    # (data_load, paper.md cite-back) expect to read. Coerce by
    # ``str(value)`` for anything non-scalar so the file head stays
    # flat. Caller's responsibility if they want richer types — they
    # can pre-format into a JSON string.
    _YAML_SCALARS = (str, int, float, bool)
    for key, value in meta.items():
        if value is None or value == "":
            continue
        # Don't allow caller-supplied keys to overwrite the two
        # we set as the contract — rank+auto_collected are
        # engine-set, not metadata-source-set.
        if key in ("auto_collected", "rank"):
            continue
        if isinstance(value, bool):
            front[key] = value
        elif isinstance(value, (int, float)):
            front[key] = value
        elif isinstance(value, str):
            # Strip newlines that would break YAML's scalar rules.
            front[key] = value.replace("\n", " ").strip()
        else:
            # list/dict/anything else → coerce to a flat string so
            # the front matter shape stays predictable.
            front[key] = str(value).replace("\n", " ").strip()
    yaml_block = yaml.safe_dump(front, default_flow_style=False, sort_keys=False)
    return f"---\n{yaml_block}---\n{content.strip()}\n"


_PAPERS_README = """\
# Papers needed

This quest's literature search returned only abstracts for a handful
of papers: they are not confirmed free to read, and FI downloads only
papers that are free.

See **``../needs/WANTED_PAPERS.md``** for the ranked list (most relevant
first) with a download link for each. **Drop the PDFs into THIS directory**
(or anywhere under it — FI walks recursively).
{after}

Accepted formats: ``.pdf`` / ``.md`` / ``.txt``. Other formats are
ignored. The README itself never counts as a paper.
"""


def _entry_identities(entry: dict[str, Any]) -> list[str]:
    """The keys a literature entry is known by: DOI, else URL, else the first
    200 characters of its text, and its normalised title. The literature node
    dedups on the same keys."""
    md = entry.get("metadata") or {}
    ident = (
        str(md.get("doi") or "").strip()
        or str(md.get("url") or "").strip()
        or (entry.get("content") or "")[:200]
    )
    norm_title = _normalize_title(md.get("title") or "")
    return [i for i in (ident, f"title:{norm_title}" if norm_title else "") if i]


def _ingest_user_dropped_papers(
    quest_root: Path,
    merged: list[dict[str, Any]],
    seen: set[str],
    log: logging.Logger,
) -> tuple[list[dict[str, Any]], int]:
    """Walk ``<quest_root>/inputs/papers/`` for user-supplied PDFs /
    MDs / TXTs and append them to the literature list as
    ``source=user_supplied`` entries with their full text in
    ``content``. Returns ``(merged, count_added)``.

    Dedups against the ``seen`` set the caller is already building
    (first 200 chars of content keyed). Files smaller than 200 bytes
    are skipped — empty stubs aren't worth ingesting."""
    papers_dir = quest_root / "inputs" / "papers"
    if not papers_dir.is_dir():
        return merged, 0
    count = 0
    for p in sorted(papers_dir.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.name == "README.md" and p.parent == papers_dir:
            continue
        suffix = p.suffix.lower()
        if suffix not in (".pdf", ".md", ".txt"):
            continue
        ocr_note = ""
        figures: list[dict[str, Any]] = []
        hidden: list[str] = []
        try:
            if suffix == ".pdf":
                content, ocr_note, figures, hidden = _extract_pdf(p)
                log.info("[literature] %s: %s%s", p.name, ocr_note,
                         f"; {len(figures)} figure(s) with captions" if figures else "")
            else:
                content = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError) as e:
            log.warning("[literature] could not read %s: %r", p, e)
            continue
        content = (content or "").strip()
        if len(content) < 200:
            if suffix == ".pdf":
                log.warning("[literature] %s gave no text to read (%s)", p.name, ocr_note or "empty")
            continue
        ident = content[:200]
        if ident in seen:
            continue
        seen.add(ident)
        # Whole, as every source now is: the text on disk, its first _STATE_TEXT_CHARS in the state. The old 8,000-
        # character cut dropped most of a paper the person had gone to the trouble of supplying.
        merged.append(_literature_entry(quest_root, content, {
            "source": "user_supplied",
            "filename": p.name,
            "path": str(p),
            "abstract_only": False,
            "fetched_full_text": True,
            **({"full_text_ocr": True} if "by OCR" in ocr_note else {}),
            **({"figures": figures} if figures else {}),
            **({"hidden_text": hidden} if hidden else {}),  # flagged by core/source_text.py
        }, quality="full_text"))
        count += 1
    return merged, count


def _extract_pdf(path: Path) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    """Text of a PDF the person dropped in ``inputs/papers/`` (core/pdf_text.py: every page, scanned pages read by
    OCR, up to 10 MB), its one-line summary, its captioned figures (core/knowledge.py ``_pdf_figures``: cut once
    and cached; a scanned page's figures too when OCR read it), and the passages it draws so a reader cannot see
    them."""
    from core import pdf_text
    from core.knowledge import _pdf_figures

    data = path.read_bytes()
    result = pdf_text.extract(data)
    try:
        figures = _pdf_figures(data, ocr_lines=result.ocr_lines or None)
    except Exception:  # noqa: BLE001 -- the text is what the source is for; its figures are extra
        figures = []
    return result.text, result.summary(), figures, list(result.hidden_text)


#: What the quest state keeps of a source's text: the first this many characters, as it always held (64 KB). The
#: whole text, up to ``knowledge.full_text_max_kb``, is on disk and read back by ``_item_content``.
_STATE_TEXT_CHARS = 64 * 1024

_TOC_LINE = re.compile(r"^.{2,120}?\s\d{1,4}$")


def _content_quality(content: str, meta: dict[str, Any]) -> str:
    """What a source's text is: ``full_text``, ``abstract_only`` (a fetch that brought back little more than the
    abstract), ``preview_only`` (mostly a table of contents: lines ending in page numbers), or ``snippet_only``.

    A fetch used to count as full text at 300 characters, so an abstract returned again, or a book's table of
    contents, was read as a whole source by the claim check and the writer."""
    if not meta.get("fetched_full_text"):
        return "snippet_only"
    head, sep, text = content.partition("---FULL TEXT")
    body = text.split("---", 1)[-1] if sep else content
    snippet = head if sep else ""
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    # A preview is short: a whole paper's reference list (lines ending in page numbers) must not make it one.
    if (len(lines) >= 10 and len(body) < 20_000
            and sum(1 for ln in lines if _TOC_LINE.match(ln)) / len(lines) > 0.4):
        return "preview_only"
    if len(body.strip()) < max(4000, 3 * len(snippet.strip())):
        return "abstract_only"
    return "full_text"


def _search_record_has_abstract(content: str) -> bool:
    """Whether a search record's text holds more than its title: an index's abstract (OpenAlex rebuilds it from its
    word index) runs to hundreds of characters; a title, or a search engine's one-line snippet, does not."""
    _title, _sep, rest = str(content or "").partition("\n")
    return len(rest.strip()) >= 200


def _sf_snapshot(quest_id: str) -> dict[str, Any]:
    from core import source_failures
    try:
        return source_failures.snapshot(quest_id)
    except Exception:  # noqa: BLE001 -- the report is extra; a quest never stops on it
        return {}


#: Failure-ledger sources that are not about the literature: web search engines and figure images.
_NOT_LITERATURE_FAILURES = frozenset({"duckduckgo", "brave", "wikimedia_commons", "image"})
#: The failure ledger's names for download steps, in words (a search source keeps its own name: openalex, crossref).
_FAILURE_SOURCE_WORDS = {
    "oa_copy": "free copies", "free_page": "free-copy pages", "publisher_pdf": "publisher PDFs",
    "full_text": "full-text step", "web_page": "web pages", "unpaywall": "Unpaywall", "europepmc": "Europe PMC",
    "pmc": "PubMed Central",
}
_FAILURE_WORDS = {
    "http_429": "too many requests", "http_4xx": "refused", "http_5xx": "server error", "http_other": "failed",
    "timeout": "timed out", "network": "unreachable", "parse": "unreadable answer", "error": "failed",
    "paused_skip": "skipped while arXiv asked FI to wait", "throttled_skip": "skipped, arXiv's queue was too long",
    "ocr_budget": "OCR time ran out", "fetch_crashed": "crashed", "budget_abandoned": "download time ran out",
}


def _literature_text_gaps(
    entries: list[dict[str, Any]], snap: dict[str, Any], *, downloads_on: bool = True,
) -> str:
    """One plain line on why the kept sources lack their full text: how many are not marked free to read, how many
    are free with no download link, how many are free but FI did not get their text, the requests that failed (by
    source, with their hosts), and a missing key. ``""`` when every scholarly source has its full text."""
    from core.knowledge import has_free_route, is_open_access

    paywalled = free_no_link = free_failed = 0
    openalex_used = False
    for e in entries:
        md = e.get("metadata") or {}
        openalex_used = openalex_used or md.get("source") in ("openalex", "arxiv")
        if (md.get("content_quality") or "snippet_only") == "full_text":
            continue
        if not _is_scholarly_record(md):
            continue
        if not is_open_access(md):
            paywalled += 1
        elif not has_free_route(md):
            free_no_link += 1
        else:
            free_failed += 1
    if not (paywalled or free_no_link or free_failed):
        return ""
    parts: list[str] = []
    if paywalled:
        parts.append(f"{paywalled} are not marked free to read (most likely behind a subscription), so FI did not "
                     "download them and a person has to")
    if free_no_link:
        parts.append(f"{free_no_link} are marked free to read but the index gave no download link, only the "
                     "publisher's DOI page, which FI does not fetch")
    if free_failed:
        parts.append(f"{free_failed} are free to read but FI did not get their text"
                     + ("" if downloads_on else " (downloading is off: knowledge.try_fetch_full_text)"))
    by_source = snap.get("by_source") or {}
    hosts: dict[str, list[str]] = {}
    for ex in snap.get("examples") or []:
        host = str(ex.get("host") or "")
        if host and host not in hosts.setdefault(str(ex.get("source")), []):
            hosts[str(ex.get("source"))].append(host)
    failed: list[str] = []
    for source, kinds in by_source.items():
        if source in _NOT_LITERATURE_FAILURES:
            continue
        said = ", ".join(f"{n} {_FAILURE_WORDS.get(k, 'failed')}"
                         for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]))
        at = hosts.get(source) or []
        failed.append(f"{_FAILURE_SOURCE_WORDS.get(source, source)} {said}" + (f" at {', '.join(at[:3])}" if at else ""))
    if failed:
        line = "requests that failed while searching and downloading: " + "; ".join(failed)
        if ("http_429" in (by_source.get("semantic_scholar") or {})
                and not os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()):
            line += " (no SEMANTIC_SCHOLAR_API_KEY is set; without one Semantic Scholar shares a busy public limit)"
        parts.append(line)
    if (openalex_used or "openalex" in by_source) and not os.environ.get("OPENALEX_API_KEY", "").strip():
        parts.append("no OPENALEX_API_KEY is set, so OpenAlex allows only about 100 searches a day (docs/INSTALL.md)")
    return "; ".join(parts)


def _is_scholarly_record(md: dict[str, Any]) -> bool:
    """A record from a scholarly index (a paper or a book), not a web page, a file the person gave, or FI's own notes
    from earlier quests."""
    kind = str(md.get("kind") or "")
    return not (md.get("source") in ("local_paper", "user_supplied", "web_search")
                or kind in _FI_INTERNAL_KINDS or kind.startswith("fi_"))


def _literature_entry(
    quest_root: Path, content: str, meta: dict[str, Any], *, quality: str | None = None,
) -> dict[str, Any]:
    """A literature entry for the state: the first ``_STATE_TEXT_CHARS`` of ``content``, and, when there is more, the
    whole text in ``data/literature/.full_text/`` with its path in ``full_text_path`` (read back by ``_item_content``)."""
    content = content or ""
    meta = {**meta, "content_quality": quality or _content_quality(content, meta)}
    if len(content) > _STATE_TEXT_CHARS:
        key = hashlib.sha1(
            (str(meta.get("doi") or meta.get("url") or meta.get("path") or "") + content[:2000]).encode("utf-8", "replace")
        ).hexdigest()[:16]
        # A dot folder: data_load walks data/ and skips dot folders, so the same text is not read twice (lit_*.md
        # already carries it for people to read).
        target = quest_root / "data" / "literature" / ".full_text" / f"{key}.txt"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            meta["full_text_path"] = str(target)
        except OSError:
            pass  # the state's first part is still there
        content = content[:_STATE_TEXT_CHARS]
    return {"content": content, "metadata": meta}


# Threshold below which a retrieved doc is treated as abstract-only
# (full text not available). Most arXiv / OpenAlex / Crossref / S2
# returns are abstracts in the 800–1400 char range; a real paper body
# is well above 5000 chars even when truncated. 1500 splits the two
# comfortably without over- or under-flagging.
_ABSTRACT_ONLY_CHAR_THRESHOLD = 1500


def _last_error_line(text: str) -> str:
    """The last line of a traceback that names an error (``NotADirectoryError: [WinError 267] ...``), else the last
    non-empty line, at most 200 characters: what a record says, never the whole traceback."""
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip() and not set(ln.strip()) <= set("^~ ")]
    named = [ln for ln in lines if re.match(r"^[A-Za-z_][\w.]*(Error|Exception)\b", ln)]
    return ((named or lines or [""])[-1])[:200]


def _read_json_or_none_path(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _anything_ran(state: "QuestState", quest_root: Path) -> bool:
    """Whether the experiment has run or the data been analysed: a result or analysis in the state, an earlier design
    version recorded after results, a ``run`` line in ``.fi/attempts.jsonl``, or results or a paper that a rerun from an
    earlier step moved aside to ``.fi/previous/`` (a data analysis writes no run line)."""
    if state.get("result_json") or state.get("analysis") or state.get("exec_result"):
        return True
    if any(isinstance(e, dict) and e.get("after_results") for e in state.get("design_history") or []):
        return True
    if any(r.get("kind") == "run" for r in _attempts.read(quest_root / ".fi", _attempts.ATTEMPTS)):
        return True
    # What a rerun moved aside counts only by content: a rerun moves the empty folders a quest starts with too, and the
    # oracle gate's own runs (raw/oracle_check) come before the first full run.
    previous = quest_root / ".fi" / "previous"
    if not previous.is_dir():
        return False
    for earlier in previous.iterdir():
        if (earlier / "results.json").is_file() or (earlier / "paper" / "paper.md").is_file():
            return True
        if any((earlier / "data" / "auto_collected").glob("*")):
            return True
        if any(p.name != "oracle_check" for p in (earlier / "raw").glob("*")):
            return True
    return False


def _append_design_revision(
    state: "QuestState", design: dict, quest_root: Path, log: logging.Logger, plan_sha: str = "",
    results_seen: bool = True, reason: str = "",
) -> list[dict[str, Any]]:
    """Record this version of the design, returning the full history.

    A first entry is the pre-registration: the hypothesis as stated before any
    result existed. Later entries are revisions, and each carries the reason
    the engine came back -- ``analyze.next_step`` when cross_check re-routed,
    or the reviewer verdict when the review loop did.

    That distinction is the whole point. "The hypothesis was stated up front"
    and "the hypothesis was rewritten after seeing the numbers" are different
    scientific claims, and a finished paper looks identical either way. The
    history is written to ``needs/DESIGN_HISTORY.json`` so it survives
    alongside the quest rather than only in state.

    ``results_seen``: False when the run that led back here produced no results (FI's own record), which the reason
    then says instead of "results were seen". ``reason``: what sent the engine back, when FI knows it better than the
    analysis or the review (a search whose objective did not move).
    """
    prior = list(state.get("design_history") or [])
    hypothesis = str((design or {}).get("hypothesis") or "").strip()

    if not prior:
        reason = "initial design, before any result existed"
        if plan_sha:
            reason = f"initial design, taken from plan.md (sha256 {plan_sha[:12]}) before any result existed"
    else:
        analysis = state.get("analysis") or {}
        next_step = str(analysis.get("next_step") or "").strip()
        verdict = str((state.get("review") or {}).get("verdict") or "").strip()
        seen = "results were seen" if results_seen else "the run before it produced no results"
        if reason:
            pass
        elif next_step in ("re_experiment", "broaden_lit"):
            reason = f"analyze.next_step={next_step} ({seen})"
        elif verdict:
            reason = f"review verdict={verdict} ({seen})"
        else:
            reason = f"re-entered design ({seen})"

    entry = {
        "revision": len(prior),
        "iteration": int(state.get("iteration", 0) or 0),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "post_hoc": bool(prior),
        # Whether the experiment had run (or the data been analysed) by now: a design done again before anything ran
        # (a rerun from the plan after a stop before the first run) is not a revision after results (core/disclosure.py).
        "after_results": bool(prior) and _anything_ran(state, quest_root),
        "reason": reason,
        "results_seen": bool(prior) and results_seen,
        "hypothesis": hypothesis[:600],
    }
    if plan_sha:
        entry["plan_sha256"] = plan_sha
    history = prior + [entry]

    if entry["post_hoc"]:
        changed = hypothesis != str(prior[-1].get("hypothesis") or "")
        log.warning(
            "[design] revision %d recorded (%s)%s -- %s; DESIGN_HISTORY.json keeps "
            "this auditable.",
            entry["revision"], reason,
            "; hypothesis TEXT CHANGED" if changed else "; hypothesis unchanged",
            "the paper's hypothesis is being set after results were seen" if results_seen
            else "the run produced no results, so the design is not a response to any",
        )

    try:
        d = quest_root / "needs"
        d.mkdir(parents=True, exist_ok=True)
        (d / "DESIGN_HISTORY.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        log.debug("[design] could not write DESIGN_HISTORY.json: %r", e)
    return history


def _is_open_access(doc: "RetrievedDoc") -> bool:
    """True when the source is confirmed free to download without a subscription (see
    ``core.knowledge.is_open_access``).

    This is deliberately separate from :func:`_is_abstract_only`. That predicate answers "did we end up with only
    an abstract?", which stays true for an arXiv paper whose full-text fetch failed. What changes is what the
    papers gate does with it: asking a person to hand-download an arXiv PDF is asking them to work around a fetch
    failure (blocked network, proxy), not a paywall. A paper that is not confirmed free is the one a person is
    asked to download; FI never fetches it from the publisher itself."""
    from core.knowledge import has_free_route, is_open_access
    # A record marked free with no free address and no id to look a free copy up by cannot be fetched: it goes
    # to the person like any paper not confirmed free.
    return is_open_access(doc.metadata) and has_free_route(doc.metadata)


def _is_abstract_only(doc: "RetrievedDoc") -> bool:
    """Heuristic: a doc is abstract-only if the retriever explicitly
    set ``metadata.abstract_only`` to truthy OR the content is short
    enough to be only an abstract (< 1500 chars). The explicit flag
    wins so future retrievers can set it precisely; the length check
    is the fallback for today's retrievers, which don't carry the flag
    on every hit."""
    md = doc.metadata or {}
    if md.get("abstract_only"):
        return True
    # A fetch that brought back only the abstract again, or a preview, still needs the paper (the papers gate asks).
    if md.get("content_quality") in ("abstract_only", "preview_only"):
        return True
    if _has_full_text(md):
        return False
    if md.get("source") in ("local_paper", "user_supplied"):
        return False
    # A page whose own metadata said it is a journal article, and that nothing confirmed free: its text was
    # discarded, so a person is asked for the paper.
    if md.get("scholarly_page"):
        return True
    # FI-internal cross-quest memory (prior papers / summaries / spines stored
    # in Axon) is NOT a paper to download — it's already in the corpus. Never
    # flag it: even when a spine carries a DOI/URL, the user has nothing to
    # fetch, and it would otherwise pollute WANTED_PAPERS.md with our own
    # artifacts instead of the external papers the user actually needs.
    kind = str(md.get("kind") or "")
    if kind in _FI_INTERNAL_KINDS or kind.startswith("fi_"):
        return False
    # Flag only a real, RESOLVABLE paper the user can actually download: it
    # carries a DOI / arXiv-id / PMID (so the manifest has a download link),
    # or it's an academic-adapter hit WITH a real title (so the manifest entry
    # is actionable). A generic web page, or a doc with neither id nor title,
    # is content the agent uses as-is — not something to ask the user to fetch.
    if md.get("doi") or md.get("arxiv_id") or md.get("pmid"):
        return True
    title = str(md.get("title") or "").strip()
    if not title:
        return False
    # An academic-adapter hit (crossref / openalex / pubmed / …) is a paper.
    if md.get("source") not in (None, "", "web_search"):
        return True
    # A web-search hit on a known academic publisher domain (SPIE / IEEE /
    # Elsevier / Wiley / …) is also a real paywalled paper the user can fetch.
    from core.knowledge import _is_academic_source
    return _is_academic_source(str(md.get("url") or ""))




def _literature_query_prompt(topic: str, idea_title: str = "", hypothesis: str = "", *,
                             work_scope: str = WORK_SCOPE_PAPERS) -> str:
    """The prompt that turns a topic (and, at the literature step, the chosen direction and the hypothesis) into three
    keyword search queries: one per facet, in the vocabulary of the kind of quest. See
    ``Engine._derive_literature_queries``."""
    if work_scope == WORK_SCOPE_PAPERS_AND_BOOKS:
        engines = "OpenAlex, Crossref, CORE, OpenAIRE, DOAJ"
        angle = "a specific period, place, work, group or case within the topic"
        frame = "the discipline or theoretical frame the topic is studied in"
        vocabulary = (
            "Use the words scholars of this subject write under: the names "
            "of the works, people, periods, places, movements, genres and "
            "concepts involved, and the discipline's own terms. Do not add "
            "method words such as model, simulation, dataset or analysis "
            "unless the topic itself is about them."
        )
    else:
        engines = "arXiv, OpenAlex, Crossref"
        angle = "the method, mechanism or measured quantity"
        frame = "the broader problem or application area it belongs to"
        vocabulary = (
            "Use the terms researchers in this field publish under: the "
            "standard names of the methods, the system studied and the "
            "quantity measured."
        )
    return (
        "Write THREE literature search queries for this research topic, one "
        f"per facet below. They go to academic search engines ({engines}); "
        "the first also goes to web search.\n\n"
        f"RESEARCH TOPIC:\n{topic[:1200]}\n\n"
        + (f"CHOSEN RESEARCH DIRECTION:\n{idea_title[:200]}\n\n" if idea_title else "")
        + (f"HYPOTHESIS UNDER TEST:\n{hypothesis[:300]}\n\n" if hypothesis else "")
        + "FACETS:\n"
        "1. the core subject, in the field's standard terms\n"
        f"2. {angle}\n"
        f"3. {frame}\n\n"
        "A search engine matches keywords, not sentences. " + vocabulary
        + " Spell out acronyms. No full sentences, quotes, boolean "
        "operators or wildcards. 3 to 8 words each; shorter queries match "
        "more.\n\n"
        'Reply as JSON only: {"queries": ["<facet 1>", "<facet 2>", "<facet 3>"]}'
    )


#: The query sets the literature searches used, one line per derivation (``.fi/literature_queries.json``).
_LITERATURE_QUERIES = "literature_queries.json"


def _query_set_key(prompt: str, iteration: int, model: str = "") -> str:
    """What a saved query set answers: the exact derivation prompt (topic, chosen direction, hypothesis, kind of
    quest), the search pass and the model that derives them. A new pass (a broaden-the-literature re-entry) or another
    model is a new key, so it derives afresh."""
    return hashlib.sha256(f"{iteration}\n{model}\n{prompt}".encode("utf-8")).hexdigest()


def _usable_queries(value: Any) -> bool:
    """A saved query list that can be reused: a non-empty list of non-empty strings (not a string, not a hand edit)."""
    return (isinstance(value, list) and bool(value)
            and all(isinstance(q, str) and q.strip() for q in value))


#: The fields of a saved query set its digest covers: what was searched, why, and which model call produced it.
_QUERY_SET_HASHED = ("stage", "key", "iteration", "queries", "prompt_sha256", "model", "call_id", "reason",
                     "revised_by", "source_sha256")

#: Search queries a person gives the quest themselves, one per line (the first three are used), in its inputs folder.
_PERSON_QUERIES = Path("inputs") / "search_queries.txt"


#: Extra fields the entries that record how the retrieved sources were judged (``stage`` ``floor``, ``screen`` and
#: ``retractions``)
#: carry; covered by the digest only when present, so the digest of an entry written without them is unchanged.
_SOURCE_VERDICT_HASHED = ("outcome", "threshold", "minimum", "sources")


def _query_set_digest(entry: dict[str, Any]) -> str:
    """The SHA-256 over an entry's hashed fields (:data:`_QUERY_SET_HASHED`, and for a source-judgement entry
    :data:`_SOURCE_VERDICT_HASHED`)."""
    hashed = {k: entry.get(k) for k in _QUERY_SET_HASHED}
    hashed.update({k: entry[k] for k in _SOURCE_VERDICT_HASHED if k in entry})
    return hashlib.sha256(json.dumps(hashed, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _query_set_standing(entry: dict[str, Any]) -> str:
    """``verified`` (its digest checks out), ``unverified`` (written before entries carried a digest) or ``edited``
    (changed after FI wrote it). The digest has no key: it catches an accidental or uninformed edit, not a deliberate
    one that recomputes it; the trace (each step's hash of the file) and, once the quest is closed, the seal are what
    protect the record. Only a verified entry is ever reused."""
    digest = entry.get("digest")
    if not digest:
        return "unverified"
    return "verified" if digest == _query_set_digest(entry) else "edited"


def _ok_calls(fi_dir: Path, node: str) -> dict[str, str | None]:
    """The answered calls of ``node`` in the quest's record of model calls: call id -> the hash of its prompt."""
    return {str(r.get("call_id")): r.get("prompt_sha256") for r in _attempts.read(fi_dir, _attempts.MODEL_CALLS)
            if r.get("node") == node and r.get("outcome") == "ok" and r.get("call_id")}


def _ok_call_ids(fi_dir: Path, node: str) -> set[str]:
    """The ids of the answered calls of ``node`` in the quest's record of model calls."""
    return set(_ok_calls(fi_dir, node))


def _person_queries(quest_root: Path) -> tuple[list[str], str] | None:
    """The queries a person wrote in ``inputs/search_queries.txt`` (up to three non-empty lines, ``#`` lines skipped)
    and the file's SHA-256; ``None`` when there is no such file or it names no query."""
    path = Path(quest_root) / _PERSON_QUERIES
    try:
        data = path.read_bytes()
    except OSError:
        return None
    lines = [" ".join(line.split()) for line in data.decode("utf-8", errors="replace").splitlines()]
    queries = [q for q in lines if q and not q.startswith("#")][:3]
    return (queries, hashlib.sha256(data).hexdigest()) if queries else None


class _QuerySetsUnreadable(Exception):
    """The record exists but could not be read just now (locked, access denied): it is left alone."""


def _read_query_sets(fi_dir: Path, *, log: Any = None, strict: bool = False) -> list[dict[str, Any]]:
    """The saved query sets. A file that is not the record (bad JSON, wrong shape) is set aside as
    ``.bad.<time>`` (kept, never silently overwritten) and a warning says so; nothing is reused from it. A file that
    cannot be read just now is left where it is (``strict``: raise :class:`_QuerySetsUnreadable` so nothing
    overwrites it)."""
    path = fi_dir / _LITERATURE_QUERIES
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        if log is not None:
            log.warning("[literature] %s could not be read just now (%s); it is left as it is, and the search "
                        "queries are derived again", path.name, e)
        if strict:
            raise _QuerySetsUnreadable(str(e)) from None
        return []
    try:
        data = json.loads(text)
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            raise ValueError("no list of entries")
    except ValueError as e:
        bad = path.with_name(f"{path.name}.bad.{time.strftime('%Y%m%dT%H%M%S')}")
        try:
            path.replace(bad)
            where = f"it was kept as {bad.name}"
        except OSError:
            where = "it could not be moved aside"
            if strict:
                raise _QuerySetsUnreadable(str(e)) from None
        if log is not None:
            log.warning("[literature] %s could not be read (%s); %s, and the search queries are derived again",
                        path.name, e, where)
        return []
    return [e for e in entries if isinstance(e, dict)]


def _record_query_set(fi_dir: Path, entry: dict[str, Any], *, log: Any = None) -> None:
    """Add one entry to ``.fi/literature_queries.json``, with its digest (:func:`_query_set_digest`). Entries are only
    ever added: the ones already there are written back exactly as read (best-effort: a record never stops a search,
    and a record that could not be read is never overwritten)."""
    try:
        entries = _read_query_sets(fi_dir, log=log, strict=True)
    except _QuerySetsUnreadable:
        return
    entry = {k: v for k, v in entry.items() if k not in ("digest", "reused")}
    entry["digest"] = _query_set_digest(entry)
    _write_query_sets(fi_dir, [*entries, entry])


def _write_query_sets(fi_dir: Path, entries: list[dict[str, Any]]) -> None:
    try:
        fi_dir.mkdir(parents=True, exist_ok=True)
        tmp = fi_dir / (_LITERATURE_QUERIES + ".tmp")
        tmp.write_text(json.dumps({"schema": 2, "entries": entries}, indent=1), encoding="utf-8")
        tmp.replace(fi_dir / _LITERATURE_QUERIES)
    except OSError:
        pass


_PAPERS_ASKED = "papers_asked.json"
_PAPERS_DECLINED = "papers_declined.json"


def _source_ref(doc: Any) -> dict[str, str]:
    """A retrieved source as the verdict records name it: its identity (:func:`_paper_key`) and its title."""
    md = getattr(doc, "metadata", None) or {}
    return {"source": _paper_key(doc), "title": " ".join(str(md.get("title") or md.get("url") or "").split())[:200]}


def _paper_key(doc: "RetrievedDoc") -> str:
    """A paper's identity across retrievals: its DOI, else its link, else its title."""
    md = doc.metadata or {}
    for k in ("doi", "url", "source_url", "title"):
        v = str(md.get(k) or "").strip().lower()
        if v:
            return f"{k}:{v}"
    return "text:" + hashlib.sha256((doc.content or "")[:500].encode("utf-8")).hexdigest()


def _papers_to_ask(needed: list["RetrievedDoc"], fi_dir: Path) -> tuple[list["RetrievedDoc"], int]:
    """The paywalled papers to stop and ask the person for -- those not declined before -- and how many were left out
    because they were. The ones asked for are recorded (``.fi/papers_asked.json``) so a resume from this pause can
    tell which papers it answered."""
    declined = _read_key_set(fi_dir / _PAPERS_DECLINED)
    new = [d for d in needed if _paper_key(d) not in declined]
    if new:
        _write_key_set(fi_dir / _PAPERS_ASKED, {_paper_key(d) for d in new})
    return new, len(needed) - len(new)


def _take_papers_declined(fi_dir: Path, quest_root: Path, resumed_from: str | None) -> set[str]:
    """When this run goes on from the papers pause (``resumed_from``, the pause kind the engine read before clearing
    the markers) and no paper was added to ``inputs/papers/``, the papers it asked for were declined: they are moved
    to ``.fi/papers_declined.json`` and returned. Anything else (another pause, an ``--update``, a run from a step:
    :func:`forget_papers_asked`) declines nothing."""
    asked = _read_key_set(fi_dir / _PAPERS_ASKED)
    if not asked or _papers_dir_has_files(quest_root) or resumed_from != "papers":
        return set()
    _write_key_set(fi_dir / _PAPERS_DECLINED, _read_key_set(fi_dir / _PAPERS_DECLINED) | asked)
    try:
        (fi_dir / _PAPERS_ASKED).unlink()
    except OSError:
        pass
    return asked


def forget_papers_asked(fi_dir: Path, *, declined: bool = True) -> None:
    """An unanswered papers pause is not answered by changing the quest (``--update``) or running it again from a step
    or re-opening it. ``declined``: also forget the papers the person went on without (a changed quest may ask for them
    again); a run from a step or a re-opened quest does not search the literature again, so it keeps them."""
    for name in ((_PAPERS_ASKED, _PAPERS_DECLINED) if declined else (_PAPERS_ASKED,)):
        try:
            (Path(fi_dir) / name).unlink()
        except OSError:
            pass


def papers_declined_count(fi_dir: Path) -> int:
    return len(_read_key_set(Path(fi_dir) / _PAPERS_DECLINED))


def _read_key_set(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {str(k) for k in data} if isinstance(data, list) else set()


def _write_key_set(path: Path, keys: set[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(keys), indent=1) + "\n", encoding="utf-8")
    except OSError:
        pass


def _papers_dir_has_files(quest_root: Path) -> bool:
    """True iff ``<quest_root>/inputs/papers/`` already carries
    user-dropped PDFs/MDs/TXTs (ignoring the README we write). Used
    by the literature pause gate to avoid re-pausing on resume."""
    papers_dir = quest_root / "inputs" / "papers"
    if not papers_dir.is_dir():
        return False
    for p in papers_dir.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.name == "README.md" and p.parent == papers_dir:
            continue
        if p.suffix.lower() in (".pdf", ".md", ".txt"):
            return True
    return False


_USER_DATA_SUFFIXES: frozenset[str] = frozenset({
    ".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx",
    ".npy", ".npz", ".txt",
})


def _pick_up_user_dropped_datasets(quest_root: Path) -> list[str]:
    """Walk ``<quest_root>/inputs/data/`` and return a sorted list of
    relative file paths for tabular / scientific-data files the user
    dropped on a pause cycle. Skips dotfiles, the auto-generated
    README.md, and unknown extensions (we don't want to feed a stray
    binary to the analyze prompt).

    Returns paths relative to ``quest_root`` so downstream consumers
    can render a short list without leaking absolute paths into the
    prompt. Empty list when the dir is missing or empty.
    """
    data_dir = quest_root / "inputs" / "data"
    if not data_dir.is_dir():
        return []
    out: list[str] = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.name == "README.md":
            continue
        if p.suffix.lower() not in _USER_DATA_SUFFIXES:
            continue
        out.append(str(p.relative_to(quest_root)).replace("\\", "/"))
    return out


def _paper_resolve_link(md: dict[str, Any]) -> str:
    """A clickable link to find/download a paper: DOI resolver, arXiv abs,
    else the source URL."""
    if md.get("doi"):
        return f"https://doi.org/{str(md['doi']).strip()}"
    if md.get("arxiv_id"):
        return f"https://arxiv.org/abs/{str(md['arxiv_id']).strip()}"
    return str(md.get("url") or md.get("pdf_url") or "").strip()


def _paper_display_title(md: dict[str, Any], fallback: str) -> str:
    """A non-empty, actionable label for a wanted paper. A real title wins; a
    title-less but resolvable hit (e.g. a DOI-only crossref record) falls back
    to its identifier so the manifest never shows a bare ``paper-N``."""
    title = str(md.get("title") or md.get("name") or "").strip()
    if title:
        return title
    if md.get("doi"):
        return f"DOI {str(md['doi']).strip()}"
    if md.get("arxiv_id"):
        return f"arXiv:{str(md['arxiv_id']).strip()}"
    if md.get("pmid"):
        return f"PMID {str(md['pmid']).strip()}"
    return fallback


def _paper_gist(content: str, md: dict[str, Any]) -> str:
    """One-line 'what it's about' for the wanted-papers manifest — the first
    sentence of the abstract (title stripped)."""
    text = (content or "").strip()
    title = str(md.get("title") or "")
    if title and text.startswith(title):
        text = text[len(title):].strip()
    m = re.match(r"(.{40,240}?[.!?])(?:\s|$)", text)
    return re.sub(r"\s+", " ", (m.group(1) if m else text[:200])).strip()


def _rank_papers_by_relevance(
    docs: list["RetrievedDoc"], query: str,
) -> list["RetrievedDoc"]:
    """Order papers most-relevant-first by lexical overlap of their
    title+abstract with the quest question (deterministic; no model)."""
    if not query.strip() or len(docs) <= 1:
        return list(docs)
    from core.passages import _lexical_scores
    blobs = [f"{(d.metadata or {}).get('title', '')} {d.content or ''}" for d in docs]
    scores = _lexical_scores(blobs, query)
    return [docs[i] for i in sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)]


def _papers_next_step(quest_id: str, paused: bool) -> str:
    """What to do after adding papers: resume when the quest is waiting for them, otherwise say the quest
    did not stop, so there is nothing to resume."""
    if paused:
        return (
            "Then continue the quest:\n\n```\nfi --resume " + quest_id + "\n```\n\n"
            "The literature step picks up the new files, extracts their text and merges them into the "
            "literature list, so the design and writing steps see real full text instead of bare abstracts."
        )
    return (
        "This quest did not stop for these papers, so there is nothing to resume: it went on with their "
        "abstracts. Files placed here are read whenever this quest's literature step runs again."
    )


def _write_wanted_papers_md(
    quest_root: Path, ranked: list["RetrievedDoc"], *, topic: str,
    oa_unfetched: list["RetrievedDoc"] | None = None, paused: bool = True,
) -> None:
    """Write a single human-friendly, RANKED ``needs/WANTED_PAPERS.md`` —
    most relevant first, each with a download link, what it's about, and the
    open-access status — so the user can grab the few that matter.

    ``oa_unfetched`` lists open-access sources whose full text FI failed to
    download. They are reported separately and never counted as paywalled:
    conflating the two produces the nonsensical request "please go download
    this arXiv paper for me"."""
    lines = [
        f"# Papers to download — {topic[:120].strip()}",
        "",
    ]
    if ranked:
        lines += [
            "The agent found these papers relevant but could only get the "
            "abstract: they are not confirmed free to read, and FI downloads only "
            "papers that are free. Download the ones that matter — **most relevant "
            "first** — and drop the PDFs into `inputs/papers/`. You don't have to "
            "get them all; even the top few sharpen the research.",
            "",
        ]
    for i, doc in enumerate(ranked, 1):
        md = doc.metadata or {}
        title = _paper_display_title(md, f"paper-{i}")
        lines.append(f"## {i}. {title}")
        link = _paper_resolve_link(md)
        if link:
            lines.append(f"- **Get it:** {link}")
        authors = md.get("authors") or md.get("author")
        if authors:
            if isinstance(authors, (list, tuple)):
                authors = ", ".join(str(a) for a in authors[:4])
            lines.append(f"- **Authors:** {str(authors)[:160]}")
        gist = _paper_gist(doc.content or "", md)
        if gist:
            lines.append(f"- **What it's about:** {gist}")
        lines.append(
            "- **Status:** not confirmed free to read, so FI did not download it — "
            "manual download needed"
        )
        lines.append("")
    if oa_unfetched:
        lines += [
            "",
            "---",
            "",
            "## Open access — FI's download failed",
            "",
            f"These {len(oa_unfetched)} source(s) are **free to read** (arXiv / "
            "PMC / preprint servers), so they are NOT paywalled. FI tried to "
            "fetch the full text and could not — usually the host is "
            "unreachable from this machine (proxy, firewall, or TLS "
            "interception), which is a network problem rather than something "
            "you need to buy or request.",
            "",
            "The quest was **not** paused for these. Fixing the network "
            "connection is the real fix and lets FI fetch them itself next "
            "run. Listed here only because a browser often succeeds where the "
            "agent's HTTP client is blocked — if you want them in this run, "
            "download and drop them into `inputs/papers/` like the others.",
            "",
        ]
        for i, doc in enumerate(oa_unfetched, 1):
            md = doc.metadata or {}
            title = _paper_display_title(md, f"oa-paper-{i}")
            lines.append(f"### {i}. {title}")
            link = _paper_resolve_link(md)
            if link:
                lines.append(f"- **Get it (free):** {link}")
            gist = _paper_gist(doc.content or "", md)
            if gist:
                lines.append(f"- **What it's about:** {gist}")
            lines.append("")
    lines.append(
        "Drop the PDFs into `inputs/papers/` (any subfolder works)."
        + (f" Then `fi --resume {quest_root.name}`." if paused else
           " This quest did not stop for them, so there is nothing to resume: it went on with their abstracts.")
    )
    try:
        (quest_root / "needs" / "WANTED_PAPERS.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )
    except OSError:
        pass


def _write_paper_need_stubs(
    quest_root: Path,
    needed: list["RetrievedDoc"],
    log: logging.Logger,
    *,
    query: str = "",
    oa_unfetched: list["RetrievedDoc"] | None = None,
    paused: bool = True,
) -> None:
    """Per missing paper, write ``<quest_root>/needs/<slug>.json`` with the
    metadata FI knows (title, authors, DOI, URL, source) so the user can
    resolve the citation and download the right PDF, PLUS a single ranked,
    human-friendly ``needs/WANTED_PAPERS.md`` (most relevant to ``query``
    first). Also creates ``<quest_root>/inputs/papers/README.md`` with resume
    instructions so the user knows what to do next."""
    needs_dir = quest_root / "needs"
    papers_dir = quest_root / "inputs" / "papers"
    needs_dir.mkdir(parents=True, exist_ok=True)
    papers_dir.mkdir(parents=True, exist_ok=True)
    needed = _rank_papers_by_relevance(needed, query)
    oa_ranked = _rank_papers_by_relevance(list(oa_unfetched or []), query)
    _write_wanted_papers_md(
        quest_root, needed, topic=query or quest_root.name, oa_unfetched=oa_ranked,
        paused=paused,
    )
    readme = papers_dir / "README.md"
    readme_text = _PAPERS_README.format(after=_papers_next_step(quest_root.name, paused))
    try:
        # Rewritten when the quest's state changed (a later run can pause where the first did not), so the
        # next-step text never contradicts the current run.
        if not readme.exists() or readme.read_text(encoding="utf-8") != readme_text:
            readme.write_text(readme_text, encoding="utf-8")
    except OSError as e:
        log.debug("[literature] papers README write failed: %r", e)
    for i, doc in enumerate(needed):
        md = doc.metadata or {}
        title = _paper_display_title(md, f"paper-{i+1}")
        slug = _slugify(title)[:48] or f"paper-{i+1}"
        stub_path = needs_dir / f"{slug}.json"
        # Don't overwrite an existing stub — preserves user notes.
        if stub_path.exists():
            continue
        try:
            stub_path.write_text(
                json.dumps({
                    "title": md.get("title"),
                    "authors": md.get("authors") or md.get("author"),
                    "doi": md.get("doi"),
                    "arxiv_id": md.get("arxiv_id"),
                    "url": md.get("url") or md.get("pdf_url"),
                    "source": md.get("source"),
                    "abstract": (doc.content or "")[:600],
                }, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            log.debug("[literature] needs stub write failed: %r", e)


def _list_user_data_files(data_dir: Path) -> list[Path]:
    """Files the user has dropped into ``<quest_root>/data/``. Excludes
    the README.md we auto-write and any dot-prefixed files. Order is
    deterministic (sorted by path) so re-walks across resumes return
    the same list and the data_load prompt is stable."""
    if not data_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "README.md" and p.parent == data_dir:
            continue
        if p.name.startswith("."):
            continue
        out.append(p)
    return out


def _render_data_readme(state: QuestState, quest_id: str) -> str:
    """The README.md FI writes into ``<quest_root>/data/`` to instruct
    the user what to drop. Includes the topic + the LLM-designed
    measurement plan so the user knows which data points actually
    answer the research question.

    Permissive about formats: csv / json / md notes / pdf / xlsx /
    txt / images — the ``data_load`` node walks whatever's there and
    synthesizes a result_json via one LLM call."""
    topic = state.get("topic", "(no topic recorded)").strip()
    design = state.get("design") or {}
    hypothesis = design.get("hypothesis", "(not recorded — design node may have failed)")
    plan = design.get("method", design.get("plan", "(see design.md)"))
    variables = design.get("variables", {})
    return (
        f"# Drop your data here\n\n"
        f"This quest is running in **no-simulation mode** "
        f"(`engine.no_simulation: true` OR clarify answered "
        f"`empirical_vs_theoretical: empirical`). The engine has\n"
        f"finished the planning half (clarify → ideate → literature → "
        f"design) and is paused waiting for you to supply real-world "
        f"data.\n\n"
        f"## What to drop into this folder\n\n"
        f"Anything that answers the research question. Permissive about "
        f"format — FI walks the dir and synthesizes a `result_json` "
        f"from the contents via one LLM call. Common shapes:\n\n"
        f"- **`.csv` / `.tsv`** — tabular measurements, one row per observation\n"
        f"- **`.json` / `.jsonl`** — structured data (survey responses, API dumps)\n"
        f"- **`.md` / `.txt`** — your own field notes, interview transcripts, observations\n"
        f"- **`.pdf`** — supporting documents (reports, papers, archival sources)\n"
        f"- **`.xlsx`** — spreadsheets (will be converted via pandas)\n"
        f"- **`.png` / `.jpg`** — images / charts. Captioned descriptions in "
        f"  an accompanying `.md` are more useful than raw images alone.\n\n"
        f"This `README.md` is auto-written by FI; you can delete or "
        f"overwrite it freely. FI ignores it when scanning the dir.\n\n"
        f"## The topic\n\n"
        f"{topic}\n\n"
        f"## The hypothesis\n\n"
        f"> {hypothesis}\n\n"
        f"## What the design asked for\n\n"
        f"{plan if isinstance(plan, str) else json.dumps(plan, indent=2)}\n\n"
        + (
            f"### Variables the design wants you to measure\n\n"
            f"```json\n{json.dumps(variables, indent=2)}\n```\n\n"
            if variables else ""
        ) +
        f"## How to resume\n\n"
        f"Once you've dropped your data into this folder, re-run:\n\n"
        f"```bash\n"
        f"fi --resume {quest_id}\n"
        f"# or:\n"
        f"python launch.py --config <your_config.yaml> --resume {quest_id}\n"
        f"```\n\n"
        f"FI will pick up at the `data_load` node, walk every file in "
        f"this folder, synthesize a result_json, and then continue "
        f"through `analyze → cross_check → write → review` exactly like "
        f"a simulation-driven quest would.\n"
    )


def _json_file_is_answer(path: Path) -> bool:
    """Whether ``path`` holds a non-empty JSON object (a staged answer that can be used)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and bool(data)


def _quest_logger(quest_id: str, fi_dir: Path) -> logging.Logger:
    """Construct (or refresh) the per-quest logger.

    Loggers in Python's ``logging`` module are global by name —
    ``logging.getLogger("frontier_insight.<qid>")`` returns the SAME
    Logger object across the process lifetime. The FileHandler we add
    here opens ``<fi_dir>/run.log`` and keeps the file descriptor
    open for the life of the Logger; on Windows that lock prevents
    tests from deleting the quest tree after the test ends, AND
    prevents reusing the same quest_id with a fresh fi_dir on a
    later run (a stale handler keeps writing to a now-deleted path).

    To fix: when the logger already has handlers, check whether the
    existing FileHandler points at the *current* run.log path. If
    yes, reuse — this is the common case where an Engine is
    re-instantiated within one process to call ``run()`` twice. If
    no, close + drop the stale handlers and rebuild them. The
    test-cleanup case (delete the dir, recreate Engine) then works
    without the second Engine inheriting a broken handler.

    Pair this with ``_close_quest_logger`` (below) in
    ``Engine.run``'s outer ``try/finally`` so the file lock is released
    on every exit path — normal completion, exception from any node,
    artifact-collection failure, OR the no-simulation pause-exit.
    The outer try/finally is what delivers the cleanup invariant
    (closing the handler only on the success path is not enough).
    """
    fi_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"frontier_insight.{quest_id}")
    logger.setLevel(logging.INFO)
    # Don't propagate to the root logger. Some libs FI imports (httpx,
    # langgraph, etc.) configure their own root-logger StreamHandlers,
    # and propagation duplicates every quest log line on stderr —
    # which the VSCode extension's chat panel then shows TWICE.
    # The file handler below + the per-process stream handler are
    # the only two sinks we want.
    logger.propagate = False

    target_log_path = (fi_dir / "run.log").resolve()
    if logger.handlers:
        # Reuse only if the existing FileHandler still points at the
        # right file. Otherwise wipe and rebuild.
        existing_fh = next(
            (h for h in logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )
        if existing_fh is not None and Path(existing_fh.baseFilename).resolve() == target_log_path:
            return logger
        # Stale handlers — close and detach them.
        _close_quest_logger(quest_id)

    fh = logging.FileHandler(target_log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    # The curated view (STAGE_PROGRESS, `Engine._progress`): a second, much shorter file the web page's live log reads by
    # default, and a second console handler filtered to the same lines. `run.log` above still gets everything, unfiltered.
    pfh = logging.FileHandler((fi_dir / PROGRESS_LOG_NAME).resolve(), encoding="utf-8")
    pfh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    pfh.addFilter(_ProgressOnly())
    logger.addHandler(pfh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(f"[{quest_id[:24]}] %(message)s"))
    sh.addFilter(_ProgressOnly())
    logger.addHandler(sh)
    return logger


def _close_quest_logger(quest_id: str) -> None:
    """Close + detach every handler from the per-quest logger so the
    underlying ``run.log`` file lock is released. Safe to call
    repeatedly (no-op if the logger has no handlers) and safe to call
    from ``finally:`` in any return path.

    Why this matters on Windows: an open ``FileHandler`` holds an
    exclusive write lock on the file. Without this close, a test that
    creates an Engine, completes it, and then ``shutil.rmtree``s the
    quest directory will fail with ``PermissionError: [WinError 32]
    The process cannot access the file because it is being used by
    another process``. We've hit that cascade across several test
    sessions; the no-simulation pause-exit adds another return path
    where the same leak would happen, so the fix lands here."""
    logger = logging.getLogger(f"frontier_insight.{quest_id}")
    for handler in list(logger.handlers):
        try:
            handler.flush()
        except (OSError, ValueError):
            pass
        try:
            handler.close()
        except (OSError, ValueError):
            pass
        logger.removeHandler(handler)
