"""Frontier Insight research engine — async LangGraph DAG.

Real LLM-driven nodes, code generation + execution in a per-quest
venv, Axon-backed knowledge retrieval, and SQLite-checkpointed state for
resumability after stalls (LLM quotas, OS sleep, manual interrupt).

Each Engine instance is stateless w.r.t. process globals; N instances must
coexist in one process for the fleet runner.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import json
import logging
import math
import os
import re
import shutil
import string
import sys
import time
import unicodedata
import uuid

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, TypedDict

# User-supplied async function that collects answers to clarify-node
# questions. Receives the ``clarify_questions`` dict and must return
# the answers dict (same keys, resolved values).
ClarifyCallback = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

# Human-feedback gate callback. Receives the review snapshot dict
# (verdict / score / strengths / weaknesses / suggestions / paper_md_path)
# and returns ``{"action": <accept|reject|refine>, "feedback": "..."}``.
HumanFeedbackCallback = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import paper_patch
from . import stats as _stats
from .config import (
    Config,
    NON_SCIENTIFIC_PAPER_FORMATS,
    SCIENTIFIC_PAPER_FORMATS,
    resolve_page_limit,
)
from .execution import ExecutionResult, make_executor, pip_failure_summary
from .knowledge import (
    FOUNDATIONAL_MAX_SUGGESTED,
    FOUNDATIONAL_SUGGESTED,
    WORK_SCOPE_PAPERS,
    WORK_SCOPE_PAPERS_AND_BOOKS,
    Knowledge,
    RetrievedDoc,
    _doc_dedup_keys,
    _is_bot_check_title,
    _normalize_title,
    _titles_match,
)
from .protocol import derive_protocol, route_for_topic_type
from .provider import (
    FallbackLLMClient,
    LLMClient,
    PROXY_PROVIDERS,
    ProxySupervisor,
    append_cost_row,
    model_for_node,
    resolve_endpoint_async,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents"

_FIGURE_SUFFIXES = frozenset({".png", ".svg", ".jpg", ".jpeg", ".pdf"})

# Under the Docker sandbox, the skills mounted into the experiment's container
# (names), kept in ``.fi/`` for a ``--watch`` that has no quest state.
_SKILL_MOUNTS_FILE = "skill_mounts.json"

# Default sampling temperature for generative nodes (ideate, write, analyze…).
_DEFAULT_CHAT_TEMPERATURE = 0.2
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
    "cross_check",
    "relevance_guard",
    "literature_screen",
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
    design: dict[str, Any]
    # Every version of the design, in order. Entry 0 is the pre-registration
    # (stated before any result existed); later entries are flagged
    # ``post_hoc`` and carry what sent the engine back to design. `review` and
    # `cross_check` can both re-enter design, so a hypothesis CAN legitimately
    # be rewritten after its results are known -- but a finished paper looks
    # identical either way, which is what makes the record necessary.
    design_history: list[dict[str, Any]]
    # Two-stage implement scaffold from ``_node_implement_outline``.
    # Carries ``{scaffold, functions, data_flow, constants,
    # result_json_template, deps}`` for the body node to consume.
    # Empty dict on legacy resume (pre-Phase-2 checkpoint), in which
    # case ``_node_implement`` falls back to the original
    # ``agents/implement.md`` one-shot prompt.
    implement_outline: dict[str, Any]
    code: str
    deps: list[str]
    exec_result: dict[str, Any]
    figures: list[str]
    # Attribution for license-clean illustrative figures pulled from the web
    # (knowledge.fetch_web_figures): one dict per figure with file / caption /
    # source_url / license / attribution, so the writer captions them and the
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
    # True when two seeds produced byte-identical results, so replication
    # stopped early. Distinguishes "no error bars because the experiment is
    # deterministic" from "no error bars because nothing could be aggregated".
    result_json_deterministic: bool
    # True when the generated script never reads ``FI_REPLICATE_SEED``, so its
    # replicate runs repeated ONE run instead of sampling. No replicate list is
    # published in that case; this records WHY, so analyze can tell the paper it
    # holds a single measurement rather than quietly losing its error bars.
    result_json_replicate_seed_ignored: bool
    # Execute-repair loop counter + history. The reflect
    # node increments `exec_reflect_iter` and appends a one-line
    # record per attempt, so analyze/write/review can describe what
    # was fixed.
    exec_reflect_iter: int
    exec_reflect_history: list[dict[str, Any]]
    exec_give_up_reason: str
    # True from the moment the reflect node writes a patch until ``execute``
    # runs it. The router runs a pending patch even when it used the last
    # repair attempt; otherwise the paper is written from the run before it,
    # next to code that never ran.
    exec_patch_pending: bool
    # True once the run's figures have been sent back for a redraw because a
    # legend or a title was drawn over what a reader needs (see
    # ``_figure_overlap_findings``). One such round per quest, so a model that
    # cannot lay a figure out is not asked again.
    figure_overlap_repaired: bool
    # Set by the execute-repair loop when a script exits 0 but every
    # numeric metric is ~0 (a degenerate run) and the repair attempts
    # couldn't produce real numbers. analyze/write/review read this so a
    # broken run is framed as a failure note, not a real result.
    degenerate_result: bool
    analysis: dict[str, Any]
    # Cross-paper check per finding. List of per-finding
    # records carrying supporting / conflicting / neutral classifications.
    cross_check: list[dict[str, Any]]
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
    # How many drafts the review has sent back to be shortened because the
    # rendered PDF ran over the page limit. At most ``_PAGE_LIMIT_REWRITES``;
    # these rewrites do not use ``engine.max_iterations``.
    page_limit_rewrites: int
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
    # Cumulative refinement asks across the quest's revise iterations.
    # One entry per refine round: ``{"iteration": int, "text": str}``.
    # The design node reads ALL of these on every revise pass so a
    # later iteration doesn't drop an earlier ask. Pre-resume empty.
    feedback_history: list[dict[str, Any]]
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
    "are license-clean illustrative images; caption and attribute them "
    "normally.\n"
)


class Engine:
    """Owns one quest's research graph, executor, knowledge layer, and LLM client."""

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
        self.supervisor = supervisor or ProxySupervisor()
        self.executor = make_executor(
            config.execution.sandbox,
            python_version=config.execution.python_version,
            docker_image=config.execution.docker_image,
            system_site_packages=config.execution.system_site_packages,
            shared_interpreter=config.execution.shared_interpreter,
        )
        self.knowledge = Knowledge(config.knowledge)
        self._log = _quest_logger(self.quest_id, self.fi_dir)
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
        reopen: bool = False,
    ) -> QuestArtifacts:
        """Run the quest to terminal state.

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
        import time as _time

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
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.quest_root / "figures").mkdir(parents=True, exist_ok=True)
            (self.quest_root / "code").mkdir(parents=True, exist_ok=True)
            (self.quest_root / "paper").mkdir(parents=True, exist_ok=True)
            self._log.info("starting quest %s", self.quest_id)
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
            await self.executor.setup(self.quest_root)

            endpoint = await resolve_endpoint_async(self.config.provider, self.supervisor)
            self._log.info(
                "provider %s -> %s (%s)",
                self.config.provider.name, endpoint.base_url, endpoint.model,
            )
            self._client = LLMClient(
                endpoint,
                timeout_s=self.config.provider.http_timeout_s,
                cli_timeout_s=self.config.provider.cli_timeout_s,
                cli_inactivity_timeout_s=(
                    self.config.provider.cli_inactivity_timeout_s
                ),
                node_cli_timeout_s=self.config.provider.node_cli_timeout_s,
                node_http_timeout_s=self.config.provider.node_http_timeout_s,
                node_model_fallbacks=(
                    self.config.provider.node_model_fallbacks
                ),
                max_prompt_chars=self.config.provider.max_prompt_chars,
                heartbeat_cb=self._llm_heartbeat,
            )
            # Wrap in a fallback chain so a single provider's outage doesn't
            # forfeit the quest. No-op (unwrapped) when no fallback configured.
            if self.config.provider.fallback:
                specs = [
                    (name, self._make_fallback_factory(name))
                    for name in self.config.provider.fallback
                ]
                self._client = FallbackLLMClient(
                    self._client, specs, log=self._log,
                )
                self._log.info(
                    "provider fallback chain: %s -> %s",
                    self.config.provider.name,
                    " -> ".join(self.config.provider.fallback),
                )

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
                    # `values` is empty dict for never-run threads.
                    payload: Any
                    if prior_snapshot and prior_snapshot.values:
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
                            _it = int((prior_snapshot.values or {}).get("iteration", 0))
                            await graph.aupdate_state(
                                run_config,
                                {
                                    "human_feedback": {"action": "refine"},
                                    "iteration": _it + 1,
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
                        final_state = await graph.ainvoke(payload, config=run_config)
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
                            mfh = snap.get("must_flag_hits") or []
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

                            if (
                                self.auto_accept_on_pass
                                and verdict == "accept"
                                and not mfh
                            ):
                                self._log.info(
                                    "[run] human_feedback auto-accept "
                                    "(verdict=accept, no must_flag_hits)",
                                )
                                _consume_snapshot()
                                payload = Command(
                                    resume={"action": "accept", "feedback": ""},
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
                                    _consume_snapshot()
                                    payload = Command(resume=answer)
                                    continue
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
                                if isinstance(answer, dict) and "action" in answer:
                                    self._log.info(
                                        "[run] consuming pre-staged human-review answer "
                                        "(action=%s)", answer.get("action"),
                                    )
                                    _consume_snapshot()
                                    payload = Command(resume=answer)
                                    continue
                            data_paused = True
                            self._log.info(
                                "[FI] paused for human review. Decide with ONE command:\n"
                                "      accept:  python launch.py --config <yaml> --resume %s --accept\n"
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
                                payload = Command(resume=answers)
                                continue
                            except asyncio.TimeoutError:
                                self._log.warning(
                                    "[run] clarify callback timed out after %ss — "
                                    "falling back to answer-file / pause-exit",
                                    self.human_feedback_timeout_s,
                                )
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
            finally:
                # Guard against early failure: if ``resolve_endpoint_async``
                # raised (unknown provider / proxy spawn failure / etc.),
                # ``self._client`` is still None and the original exception
                # is what the caller should see. Unconditionally calling
                # ``aclose()`` on None would raise AttributeError and mask
                # the real error. Same logic for the proxy release —
                # only release a handle we actually acquired.
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
                return self._collect_artifacts(final_state)

            artifacts = self._collect_artifacts(final_state)
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
            self._write_back_knowledge(artifacts, final_state)
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
            try:
                await self._write_quest_failed_diagnostic(exc, run_config)
            except Exception as diag_err:
                # Best-effort logging only — re-raising the original
                # exception is the contract.
                self._log.warning(
                    "[run] could not write quest_failed.md: %r", diag_err,
                )
            raise
        finally:
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
            try:
                _source_failures.current_quest.reset(_quest_ctx)
            except ValueError:
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

    # ---- graph topology --------------------------------------------------

    def _build_graph(self) -> StateGraph:
        # Subclassing `Engine` and overriding `_build_graph` is the
        # supported way to ship a domain-specific pipeline (e.g.,
        # a lithography graph) without forking the full Engine class.
        # The QuestState TypedDict is the contract — keep field names
        # backwards-compatible if you add a graph here.
        g: StateGraph[QuestState] = StateGraph(QuestState)
        g.add_node("clarify", self._node_clarify)
        g.add_node("ideate", self._node_ideate)
        g.add_node("literature", self._node_literature)
        g.add_node("pause_after_literature", self._node_pause_after_literature)
        g.add_node("select_skills", self._node_select_skills)
        g.add_node("design", self._node_design)
        # design → implement_outline → implement → execute (two-stage
        # implement). The outline node produces a scaffold + function
        # signatures + constants + RESULT_JSON template, which the
        # body node fills in. Splitting the work this way lets each
        # call be smaller and gives the model a feedback opportunity
        # before committing to a full ~200-line experiment.py. On a
        # pre-Phase-2 resume where ``implement_outline`` is empty, the
        # body node falls back to the legacy single-shot prompt.
        g.add_node("implement_outline", self._node_implement_outline)
        g.add_node("implement", self._node_implement)
        g.add_node("execute", self._node_execute)
        # execute → execute_reflect (loops back to execute on failure)
        g.add_node("execute_reflect", self._node_execute_reflect)
        g.add_node("analyze", self._node_analyze)
        # analyze → cross_check (always) → write OR design
        g.add_node("cross_check", self._node_cross_check)
        # evidence_gate sits on the cross_check → write happy path: it
        # weighs the assembled evidence and either proceeds to write or
        # sends the quest back for ONE bounded literature broaden. A
        # logged passthrough when engine.evidence_gate is off.
        g.add_node("evidence_gate", self._node_evidence_gate)
        g.add_node("write", self._node_write)
        g.add_node("claim_check", self._node_claim_check)
        g.add_node("review", self._node_review)
        g.add_node("human_feedback", self._node_human_feedback)
        # no-simulation mode: design → auto_collect_data → wait_for_data
        # → (pause + resume) → data_load → analyze. All three new nodes
        # are conditional and only fire when
        # ``state.no_simulation_resolved`` is True.
        g.add_node("auto_collect_data", self._node_auto_collect_data)
        g.add_node("wait_for_data", self._node_wait_for_data)
        g.add_node("data_load", self._node_data_load)
        # no-simulation mode only: turn the collected web/data content into
        # figures so the paper/poster/slides aren't text-only. Passthrough
        # in the simulation path (which makes its own figures in execute).
        g.add_node("web_plots", self._node_web_plots)
        g.add_node("web_figures", self._node_web_figures)

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
        g.add_edge("pause_after_literature", "select_skills")
        g.add_edge("select_skills", "design")
        # design → implement (normal sim path) OR auto_collect_data
        # (no-simulation, agent attempts auto-collect via Axon first
        # then wait_for_data handles the pause-if-still-empty case).
        g.add_conditional_edges(
            "design",
            self._route_after_design,
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
            self._route_after_execute_reflect,
            {"retry": "execute", "proceed": "analyze"},
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
            self._route_after_cross_check,
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
        g.add_conditional_edges(
            "evidence_gate",
            self._route_after_evidence_gate,
            {"write": "write", "broaden_lit": "literature"},
        )
        # write → claim_check → review. claim_check grounds each paper claim to
        # evidence (a no-op passthrough when engine.claim_grounding is off).
        g.add_edge("write", "claim_check")
        g.add_edge("claim_check", "review")
        g.add_conditional_edges(
            "review",
            self._route_after_review,
            {
                "revise": "design",
                # Every must-flag is a problem with the text: the experiment
                # stands, so only the paper is written again.
                "rewrite": "write",
                # A must-flag named something the run computed: no rewrite of
                # the paper can fix that, so the experiment is written and run
                # again (implement → execute → analyze → … → write).
                "re_execute": "implement",
                "done": END,
                "human_feedback": "human_feedback",
            },
        )
        # human_feedback resolves to one of three outcomes after the
        # callback returns: accept / reject → END, refine → design.
        g.add_conditional_edges(
            "human_feedback",
            self._route_after_human_feedback,
            {"revise": "design", "done": END},
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

    def _route_after_execute_reflect(self, state: QuestState) -> str:
        """Route based on whether the reflect node patched the
        code (→ retry execute) or accepted the failure / success
        (→ proceed to analyze)."""
        result = state.get("exec_result") or {}
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
        iters = state.get("exec_reflect_iter", 0)
        if iters >= self.config.engine.exec_reflect_max_iterations:
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
            if _assertion_violations(state):
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

        Empty-cross_check guard: ``_node_cross_check`` returns
        ``{"cross_check": []}`` early when ``analysis.key_findings`` is
        empty OR when ``cross_check_per_finding_k <= 0`` — both BEFORE
        the iteration-bump block runs. If analyze still emits
        ``next_step: "broaden_lit"`` in that case, the loop literature →
        design → implement → execute → analyze → cross_check → broaden_lit
        never increments ``iteration`` and the cap never fires. Terminate
        on empty cross_check: there are no findings to broaden literature
        against, so writing the paper with what we have is the safe
        choice (audit BLOCK #11).
        """
        if not self.config.engine.enable_analyze_reroute:
            return "write"
        if not state.get("cross_check"):
            return "write"
        analysis = state.get("analysis") or {}
        next_step = analysis.get("next_step", "publish")
        if state.get("iteration", 0) >= self.config.engine.max_iterations:
            return "write"
        if next_step == "broaden_lit":
            return "broaden_lit"
        if next_step == "re_experiment":
            return "redesign"
        return "write"

    def _route_after_evidence_gate(self, state: QuestState) -> str:
        """The evidence_gate node already decided write vs broaden (it
        owns the bounded-broaden bookkeeping); the router just reads it.
        Fails open to ``write`` when the gate was a passthrough."""
        return (state.get("evidence_assessment") or {}).get("route", "write")

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
        must_flag = review.get("must_flag_hits") or []
        # The draft is over the page limit and nothing else is to be fixed:
        # write it again, shorter. These rewrites have their own cap (the
        # review stops forcing them after ``_PAGE_LIMIT_REWRITES``), so the
        # iteration budget neither pays for them nor stops them.
        page_rewrites = int(state.get("page_limit_rewrites") or 0)
        if _only_page_limit_hits(must_flag) and page_rewrites <= _PAGE_LIMIT_REWRITES:
            self._log.info(
                "[route] the draft is over the page limit — rewriting it shorter "
                "(shortening %d of %d, outside engine.max_iterations)",
                page_rewrites, _PAGE_LIMIT_REWRITES,
            )
            return "rewrite"
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
            if _hits_need_only_a_rewrite(must_flag):
                self._log.info(
                    "[route] must_flag_hits=%s are all about the text — rewriting the paper",
                    must_flag,
                )
                return "rewrite"
            self._log.info(
                "[route] must_flag_hits=%s — forcing revise even if review_loop=False",
                must_flag,
            )
            return "revise"
        if self.config.pauses.review == "ask":
            return "human_feedback"
        if not self.config.engine.review_loop:
            return "done"
        verdict = review.get("verdict", "accept")
        if verdict == "revise" and state.get("iteration", 0) < self.config.engine.max_iterations:
            return "revise"
        return "done"

    def _route_after_human_feedback(self, state: QuestState) -> str:
        """``refine`` loops back to design with the user's feedback in state;
        ``accept`` / ``reject`` finalise.

        A human ``refine`` is honoured **regardless of ``max_iterations``** — it
        is a deliberate, interactive request, not the unattended review loop the
        cap exists to bound. The user stays in control: the refine runs one
        design→…→review pass and lands back at the human-review gate, where they
        can refine again or accept. (Earlier this was gated on the iteration
        budget, so a refine submitted past the cap was silently dropped — the
        user clicked Refine and nothing happened.)"""
        hf = state.get("human_feedback") or {}
        action = hf.get("action", "accept")
        if action == "refine":
            return "revise"
        return "done"

    # ---- nodes -----------------------------------------------------------

    async def _node_clarify(self, state: QuestState) -> QuestState:
        """Pre-flight clarification.

        Three modes, controlled by `engine.clarify_mode`:

        * ``"off"`` (default) — skip entirely. Returns an empty patch so
          downstream nodes see ``clarify_done=False`` and ignore the slot.
        * ``"auto"`` — agent generates the 5-question survey AND uses
          its own `default` values as answers. No human loop. Cheap way
          to sharpen framing.
        * ``"interactive"`` — generates the questions, then calls
          LangGraph's ``interrupt()`` so a CLI prompt (`launch.py
          --interactive`) or the web UI's clarify panel can collect
          answers from the user. ``interrupt()`` returns the payload
          the caller resumed with, which becomes ``clarify_answers``.

        Idempotent across restart: when ``clarify_done`` is already True
        (e.g. resuming after a kill), the node passes through.
        """
        mode = self.config.pauses.clarify
        if state.get("clarify_done"):
            return {}
        if mode == "off":
            self._log.info("[clarify] mode=off; skipping")
            # When clarify is skipped, only the YAML flag can switch on
            # no_simulation — there's no clarify answer to inspect.
            return {
                "clarify_done": True,
                **self._resolve_modes({}),
            }

        # Proposal short-circuit: when this quest was started from a
        # ``/proposal``-generated YAML (detected by a ``*-proposal.md``
        # pinned in ``knowledge.local_papers``), parse the proposal's
        # structured H2 sections directly into ``clarify_answers``
        # and skip the clarify LLM call entirely. The user already
        # approved the hypothesis + success criteria when they ran
        # ``/proposal``; re-asking is wasted compute.
        #
        # Gated on ``mode == "auto"`` only — ``interactive`` mode's
        # contract is "let the human confirm / override every slot",
        # so silently bypassing the prompt because a proposal MD
        # happens to be pinned would surprise users. In interactive
        # mode the regular ``interrupt()`` path still runs; the
        # human can copy values from the proposal MD into the modal
        # if they want.
        if mode == "auto":
            seeded = self._maybe_seed_clarify_from_proposal()
            if seeded is not None:
                proposal_path, answers = seeded
                # Even when seeded from a proposal, YAML-pinned
                # clarify_overrides still win — the interview /
                # --update path is meant to override proposal defaults.
                answers = {**answers, **dict(self.config.engine.clarify_overrides)}
                self._log.info(
                    "[clarify] mode=auto; seeded from proposal %s "
                    "(skipped LLM call)",
                    proposal_path.name,
                )
                return {
                    "clarify_questions": {},
                    "clarify_answers": answers,
                    "clarify_done": True,
                    **self._resolve_modes(answers),
                }

        # When the interview / --update has pinned EVERY known clarify
        # slot in YAML AND mode is auto, the LLM call is pure waste —
        # the answers are already known. Skip it. The 8-slot list
        # matches _CLARIFY_LABELS + the proposal-seed contract.
        overrides = dict(self.config.engine.clarify_overrides)
        # ``known_slots`` defines when ``clarify_mode=auto`` can
        # short-circuit (all answers known from clarify_overrides, no
        # LLM call needed). Kept at the ORIGINAL 7 slots so configs
        # produced by the existing interview machinery — which doesn't
        # pin ``topic_shape`` yet — continue to short-circuit instead
        # of regressing to a wasted LLM call on every quest.
        # ``topic_shape`` is handled by the safety-default a few lines
        # below: when not pinned, the engine populates a sensible
        # ``experimental`` default so downstream consumers always see
        # a value.
        known_slots = {
            "comparative_baseline", "empirical_vs_theoretical",
            "simulatability", "success_metric", "budget",
            "output_kinds", "study_depth", "paper_venue",
        }
        if mode == "auto" and known_slots.issubset(overrides.keys()):
            # Auto-populate ``topic_shape`` when the user pinned the
            # other 7 but not this one. ``experimental`` is the right
            # safe default — most quests are experiment-shaped — and
            # the mismatch helper / design prompt simply act as no-ops
            # for ``experimental``. The override path leaves the
            # caller's value intact when they DID pin it.
            overrides.setdefault("topic_shape", "experimental")
            self._log.info(
                "[clarify] mode=auto; all %d slots pinned via "
                "clarify_overrides (skipped LLM call)", len(known_slots),
            )
            modes = self._resolve_modes(overrides)
            self._log_topic_shape_mismatch(
                overrides, no_simulation_resolved=modes["no_simulation_resolved"])
            return {
                "clarify_questions": {},
                "clarify_answers": overrides,
                "clarify_done": True,
                **modes,
            }

        prompt = self._prompts["clarify"].substitute(topic=state["topic"])
        text = await self._chat(prompt, node="clarify")
        questions = _parse_json_lenient(text) or {}
        if not isinstance(questions, dict) or not questions:
            # Degrade gracefully: if the LLM produced unparseable JSON,
            # synthesize a minimal default questionnaire from the topic
            # alone so the downstream nodes get *something*.
            self._log.warning("[clarify] LLM returned no parseable questions; using minimal defaults")
            questions = _default_clarify_questions(state["topic"])

        if mode == "auto":
            agent_answers = {
                k: v.get("default") for k, v in questions.items()
                if isinstance(v, dict)
            }
            agent_count = len(agent_answers)  # pre-merge count for honest logging
            answers = dict(agent_answers)
            # User-pinned overrides (from the interview / --update flow)
            # win over the agent's self-answers. Logged so run.log
            # tells the user exactly which slots they pre-pinned.
            if overrides:
                pinned = sorted(overrides.keys() & answers.keys())
                answers = {**agent_answers, **overrides}
                # EXCEPT ``simulatability`` / ``topic_shape``: these are
                # TOPIC JUDGMENTS the clarify agent is better placed to make
                # than the interview's format heuristic. When the agent
                # actually answered one and it conflicts with the pinned
                # value, the agent's judgment wins back — otherwise a
                # review-shaped topic the agent flagged as no-simulation
                # gets force-run as an experiment by a stale
                # ``simulatability: "yes"`` override (audit #1). The hard
                # ``engine.no_simulation: true`` YAML pin still wins
                # absolutely (see _resolve_no_simulation_from_clarify);
                # there is intentionally no hard "force simulate" — the
                # engine trusts the agent's "can a simulation answer this?"
                # call over a format guess.
                reclaimed: list[str] = []
                for slot in ("simulatability", "topic_shape"):
                    agent_val = agent_answers.get(slot)
                    if (agent_val not in (None, "")
                            and slot in overrides
                            and overrides[slot] != agent_val):
                        answers[slot] = agent_val
                        reclaimed.append(slot)
                self._log.info(
                    "[clarify] mode=auto; agent self-answered %d slots, "
                    "user-pinned overrides applied to %d (%s)%s",
                    agent_count, len(pinned), ",".join(pinned),
                    (f"; agent judgment kept over heuristic for "
                     f"{','.join(reclaimed)}") if reclaimed else "",
                )
            else:
                self._log.info("[clarify] mode=auto; agent self-answered %d slots", agent_count)
            modes = self._resolve_modes(answers)
            self._log_topic_shape_mismatch(
                answers, no_simulation_resolved=modes["no_simulation_resolved"])
            return {
                "clarify_questions": questions,
                "clarify_answers": answers,
                "clarify_done": True,
                **modes,
            }

        # Interactive: pre-fill any user-pinned answers as the
        # default for each question's interrupt payload, so the human
        # sees the interview / --update value already in the slot and
        # only has to confirm. Cheap merge; an empty overrides dict
        # is a no-op and preserves prior behavior.
        if overrides:
            for slot, value in overrides.items():
                if isinstance(questions.get(slot), dict):
                    questions[slot]["default"] = value
        # NB: the on-disk ``.fi/clarify_questions.json`` snapshot (so a
        # subprocess quest can render a web form) is written by ``Engine.run``'s
        # clarify pause-exit branch, NOT here — the node re-executes on resume
        # (interrupt() returns the answer), so writing it here would re-create
        # the file the run loop just consumed and falsely re-show the form.
        # Interactive: pause the graph until the caller resumes with answers.
        payload = self._pause_for_human(
            kind="clarify",
            interaction="answer",
            headline="confirm the research setup",
            steps=[
                f"Answer the {len(questions)} clarifying question(s) in the "
                "panel (Web / VSCode) or at the CLI prompt.",
                "Headless run? Write your answers into "
                "`.fi/clarify_answer.json`, then resume.",
            ],
            payload={"clarify_questions": questions},
        )
        # payload is whatever the resume call sent. Accept either a
        # dict (the answers) or a dict that includes a "clarify_answers"
        # key (the GUI wraps it that way for forward-compat).
        if isinstance(payload, dict) and "clarify_answers" in payload:
            answers = payload["clarify_answers"]
        elif isinstance(payload, dict):
            answers = payload
        else:
            # Resumed with a non-dict (or None) — fall through to defaults.
            answers = {k: v.get("default") for k, v in questions.items() if isinstance(v, dict)}
        modes = self._resolve_modes(answers)
        self._log_topic_shape_mismatch(
            answers, no_simulation_resolved=modes["no_simulation_resolved"])
        return {
            "clarify_questions": questions,
            "clarify_answers": answers,
            "clarify_done": True,
            **modes,
        }

    def _maybe_seed_clarify_from_proposal(
        self,
    ) -> tuple[Path, dict[str, Any]] | None:
        """Check ``knowledge.local_papers`` for a ``*-proposal.md`` and
        parse it into a clarify_answers shape. Returns
        ``(proposal_path, answers)`` when a proposal is found and
        parses cleanly; ``None`` otherwise.

        Imported lazily so a quest with no local_papers (the common
        case) doesn't pay the import cost. The proposal_seed module
        is pure-Python with no heavy deps so the import is cheap
        once paid."""
        local_papers = list(self.config.knowledge.local_papers or [])
        if not local_papers:
            return None
        from core.proposal_seed import seed_clarify_from_local_papers
        return seed_clarify_from_local_papers(local_papers)

    def _log_topic_shape_mismatch(
        self, answers: dict[str, Any], *, no_simulation_resolved: bool,
    ) -> None:
        """Log a WARNING when the clarify-detected topic shape disagrees
        with the engine's already-computed simulatability decision.

        Specifically: if the topic shape is ``review`` / ``case_study``
        / ``opinion`` BUT the engine resolved to SIMULATE, the quest
        is about to run a Python experiment on a topic that doesn't
        really want one. The downstream design and write prompts
        already read ``topic_shape`` from ``clarify_answers`` and will
        keep the experiment minimal + shift weight to the literature
        synthesis — but the mismatch is worth surfacing in run.log so
        the user can hand-pivot ``simulatability=no`` next time if
        they prefer the no-experiment flow.

        Takes the already-computed ``no_simulation_resolved`` so we
        don't double-call ``_resolve_no_simulation_from_clarify``
        (which logs at INFO each call). No-op when the slot is
        missing (legacy quests pre-dating ``topic_shape``) or the
        shape is ``experimental``.
        """
        answers = answers or {}
        shape_raw = answers.get("topic_shape")
        shape = ""
        if isinstance(shape_raw, dict):
            shape = str(shape_raw.get("default", "")).strip().lower()
        elif isinstance(shape_raw, str):
            shape = shape_raw.strip().lower()
        if not shape or shape == "experimental":
            return
        if no_simulation_resolved:
            # NO_SIMULATION path is consistent with non-experimental
            # shapes — no warning needed.
            return
        self._log.warning(
            "[clarify] topic_shape=%r but engine resolved to SIMULATE — "
            "the quest will run an experiment on a topic that doesn't "
            "want one. design + write stages will keep the experiment "
            "minimal and shift weight to literature synthesis; set "
            "``simulatability: \"no\"`` in clarify_overrides (the quotes "
            "matter — PyYAML reads bare ``no`` as boolean False) if you "
            "prefer the no-experiment flow.", shape,
        )

    def _resolve_no_simulation_from_clarify(
        self, answers: dict[str, Any],
    ) -> bool:
        """Decide whether the ``no_simulation`` flag should be on for
        this quest, based on the YAML config + the clarify answers.

        Decision precedence (first match wins):

        1. ``engine.no_simulation: true`` in YAML → always True. The
           user's explicit override beats any LLM judgement.
        2. ``simulatability`` answer in clarify (new slot, see
           ``agents/clarify.md``):
            - ``"no"`` → True. The LLM judged that Python can't
              produce data that answers this question.
            - ``"yes"`` or ``"uncertain"`` → False. The simulation
              path runs; ``uncertain`` adds a review-time caveat (not
              implemented in this method — happens in the review prompt).
        3. **Legacy fallback** — ``empirical_vs_theoretical == "empirical"``
           → True. Kept for back-compat with quests started before the
           ``simulatability`` slot existed (resumes from old
           checkpoints, hand-written YAML answers, etc.). New quests
           should always have the simulatability slot populated.

        Every resolution is logged at INFO level with the reason
        (when available) so the user can see exactly why the engine
        took whichever path it took — log line format:
        ``[clarify] simulatability resolved: NO_SIMULATION|SIMULATE
        (source=yaml|clarify_simulatability|clarify_empirical_legacy|default,
        reason='<quote>')``.
        """
        answers = answers or {}
        if self.config.engine.no_simulation:
            self._log.info(
                "[clarify] simulatability resolved: NO_SIMULATION "
                "(source=yaml, reason='engine.no_simulation: true')",
            )
            return True

        # ``simulatability`` may arrive in either shape:
        # * dict ``{default, reason}`` — when callers pass through the
        #   full clarify slot (the unit-test path in
        #   test_engine_helpers, and any caller that explicitly
        #   preserves the slot's reason).
        # * bare string ``"yes" | "no" | "uncertain"`` — what
        #   ``clarify_mode="auto"`` produces, because the reducer at
        #   line ~530 collapses ``{k: v["default"]}`` for every slot
        #   to keep ``clarify_answers`` flat for downstream prompt
        #   substitution. The reason is dropped in that path; we log
        #   "(no reason provided)" so the source is still greppable.
        # Both shapes route the same — the contract is the decision.
        sim = answers.get("simulatability")
        decision = ""
        reason = ""
        if isinstance(sim, dict):
            raw_default = sim.get("default", "")
            # Coerce bool to the documented string vocabulary — PyYAML
            # parses unquoted ``yes`` / ``no`` as ``True`` / ``False``,
            # which is the most common way users write the slot in
            # their YAML. Without this coercion, ``simulatability: no``
            # (unquoted) silently fell through to the legacy fallback
            # and the engine ran the simulation path despite the user
            # asking for no-simulation. ``True`` / ``False`` are the
            # only sensible bool mappings: ``True`` ≈ "yes",
            # ``False`` ≈ "no". String values are still preferred.
            if isinstance(raw_default, bool):
                raw_default = "yes" if raw_default else "no"
            decision = str(raw_default).strip().lower()
            reason = str(sim.get("reason", "")).strip()
        elif isinstance(sim, bool):
            decision = "yes" if sim else "no"
        elif isinstance(sim, str):
            decision = sim.strip().lower()
        if decision or isinstance(sim, dict):
            if decision == "no":
                self._log.info(
                    "[clarify] simulatability resolved: NO_SIMULATION "
                    "(source=clarify_simulatability, reason=%r)",
                    reason or "(no reason provided)",
                )
                return True
            if decision in ("yes", "uncertain"):
                self._log.info(
                    "[clarify] simulatability resolved: SIMULATE "
                    "(source=clarify_simulatability, decision=%s, reason=%r)",
                    decision, reason or "(no reason provided)",
                )
                return False
            # Unknown / empty decision — fall through to legacy fallback.
            # Surface it though: a misformed LLM response (typo, "maybe",
            # blank, anything outside the documented {yes, no, uncertain}
            # set) silently downgrading to the legacy path is a routing
            # bug waiting to bite. Logging a WARNING here keeps the
            # decision visible in run.log so the user can see "the LLM
            # returned X which we didn't recognize, so we fell through
            # to the empirical_vs_theoretical fallback" without having
            # to diff the clarify answers against the engine source.
            if decision:
                self._log.warning(
                    "[clarify] simulatability.default=%r is not in the "
                    "documented set {yes, no, uncertain}; falling through "
                    "to the empirical_vs_theoretical legacy check. "
                    "Check agents/clarify.md and the LLM's clarify output "
                    "for drift.", decision,
                )

        # Legacy fallback for quests scoped before the simulatability
        # slot was added.
        evt = answers.get("empirical_vs_theoretical")
        if isinstance(evt, str) and evt.strip().lower() == "empirical":
            self._log.info(
                "[clarify] simulatability resolved: NO_SIMULATION "
                "(source=clarify_empirical_legacy, "
                "reason='empirical_vs_theoretical=empirical, "
                "simulatability slot missing')",
            )
            return True

        self._log.info(
            "[clarify] simulatability resolved: SIMULATE "
            "(source=default, no signal from YAML or clarify)",
        )
        return False

    def _resolve_survey_from_clarify(self, answers: dict[str, Any]) -> bool:
        """Decide whether ``survey`` mode is on: a descriptive literature /
        history synthesis with NO experiment AND NO dataset. True when
        ``engine.survey_mode: true`` is pinned in YAML OR the clarify agent
        classified ``topic_shape == 'survey'`` (a history / overview /
        "evolution of X" humanities topic). Survey is a stronger form of
        no_simulation — the caller (:meth:`_resolve_modes`) forces
        ``no_simulation_resolved`` True whenever this is True, so the graph
        skips both the experiment and the data-collection path."""
        if self.config.engine.survey_mode:
            self._log.info(
                "[clarify] survey resolved: SURVEY "
                "(source=yaml, reason='engine.survey_mode: true')",
            )
            return True
        answers = answers or {}
        shape = str(answers.get("topic_shape") or "").strip().lower()
        if shape == "survey":
            self._log.info(
                "[clarify] survey resolved: SURVEY "
                "(source=clarify_topic_shape, reason='topic_shape=survey')",
            )
            return True
        return False

    def _resolve_modes(self, answers: dict[str, Any]) -> dict[str, bool]:
        """Resolve the two run-mode flags together and return them as a
        state patch. ``survey_mode_resolved`` implies ``no_simulation_resolved``
        (a survey has no experiment), so callers can spread this into their
        clarify return dict and get a consistent pair. Passing ``{}`` (no
        clarify answers, e.g. ``clarify=off``) still honours the YAML
        ``engine.no_simulation`` / ``engine.survey_mode`` flags via the
        underlying resolvers."""
        survey = self._resolve_survey_from_clarify(answers)
        no_sim = self._resolve_no_simulation_from_clarify(answers) or survey
        return {
            "no_simulation_resolved": no_sim,
            "survey_mode_resolved": survey,
        }

    async def _node_ideate(self, state: QuestState) -> QuestState:
        if self.config.engine.analyze_local_first:
            # --analyze: the user supplied the data; there's nothing to
            # ideate. Passthrough (no LLM call) — downstream reads
            # chosen_idea via .get() and tolerates its absence.
            self._log.info("[ideate] analyze_local_first — skipping ideation")
            return {}
        self._log.info("[ideate] topic=%s", state["topic"][:80].replace("\n", " "))
        # Pull a few related items from the knowledge base to ground ideation.
        # No chosen_idea yet — pass chat_fn so the source-router (if
        # enabled) can still pick sources from the catalog using the
        # topic alone.
        seeded = await self.knowledge.asearch(
            state["topic"], top_k=3,
            chat_fn=functools.partial(self._chat_messages, node="source_router"),
            work_scope=self._work_scope(state),
        )
        prompt = self._prompts["ideate"].substitute(
            topic=state["topic"],
            literature_block=_format_lit(seeded, **self._lit_kwargs(state)),
            clarify_block=_format_clarify(state),
        )
        # Multi-model ensemble path: when the YAML carries
        # provider.node_ensemble["ideate"], fan out N models in parallel
        # and tournament-pick the best ideas-JSON. We skip the downstream
        # ``ideate_tournament`` + ``ideate_reflect`` steps in that case —
        # the ensemble's moderator already does that job (picking the
        # best of N candidate idea-sets) and re-running tournament/reflect
        # on top would double-bill for no quality gain.
        ensemble_cfg = self._ensemble_for_node("ideate")
        if ensemble_cfg is not None:
            from core.ensemble import EnsembleError
            try:
                result = await self._ensemble_chat(
                    prompt, node="ideate", ensemble_cfg=ensemble_cfg,
                )
                text = result.merged if isinstance(result.merged, str) else json.dumps(result.merged)
                # Ensemble already did the "pick the best" work — skip
                # tournament + reflect downstream.
                ideate_skip_post_processing = True
            except EnsembleError as e:
                self._log.warning(
                    "[ideate] ensemble all-failed (%s); falling back to single-call path", e,
                )
                text = await self._chat(prompt, node="ideate")
                ideate_skip_post_processing = False
        else:
            text = await self._chat(prompt, node="ideate")
            ideate_skip_post_processing = False
        parsed = _parse_json_lenient(text) or {}
        ideas = parsed.get("ideas") or []
        chosen = parsed.get("chosen") or (ideas[0] if ideas else {"title": "fallback", "rationale": ""})

        # Pairwise tournament. When enabled AND there are
        # at least 2 ideas to compare, REPLACES the single critique
        # call below with C(N, 2) parallel pairwise comparisons and
        # picks the highest-win-count idea. See
        # ``_run_ideate_tournament`` for the aggregation policy.
        critique: dict[str, Any] = {}
        tournament_result: dict[str, Any] | None = None
        tournament_ran = False
        if (not ideate_skip_post_processing
            and self.config.engine.ideate_tournament and len(ideas) >= 2):
            try:
                chosen, tournament_result = await self._run_ideate_tournament(
                    state, ideas, initial_chosen=chosen,
                )
                tournament_ran = True
            except Exception as e:
                self._log.warning(
                    "[ideate] tournament failed: %s — falling through "
                    "to ideate_reflect if enabled", e,
                )
        # Self-reflection. Single extra LLM call that may
        # swap chosen_idea to a different entry from the brainstormed
        # list. Skipped ONLY when the tournament actually ran (its
        # pick subsumes the critique's purpose). If the tournament
        # was enabled but couldn't run (N<2 ideas) or raised, reflect
        # still gets its chance to refine the single idea.
        if (not ideate_skip_post_processing
            and not tournament_ran
            and self.config.engine.ideate_reflect and ideas):
            try:
                critique_prompt = self._prompts["ideate_reflect"].substitute(
                    topic=state["topic"],
                    clarify_block=_format_clarify(state),
                    ideas_block=json.dumps(ideas, indent=2),
                    chosen_block=json.dumps(chosen, indent=2),
                )
                ctext = await self._chat(critique_prompt, node="ideate_reflect")
                critique = _parse_json_lenient(ctext) or {}
                swap_to = (critique.get("swap_to") or "").strip()
                if swap_to:
                    swapped = next(
                        (i for i in ideas if i.get("title") == swap_to), None,
                    )
                    if swapped is not None:
                        self._log.info(
                            "[ideate] reflection swapped chosen: %r -> %r",
                            chosen.get("title", "?"), swap_to,
                        )
                        chosen = {
                            **swapped,
                            "rationale": critique.get("refined_rationale")
                            or swapped.get("rationale", ""),
                        }
                    else:
                        self._log.info(
                            "[ideate] reflection wanted unknown title %r; keeping original pick",
                            swap_to,
                        )
                elif critique.get("refined_rationale"):
                    chosen = {**chosen, "rationale": critique["refined_rationale"]}
            except Exception as e:
                self._log.warning("[ideate] reflection skipped: %s", e)

        out: QuestState = {"ideas": ideas, "chosen_idea": chosen}
        if critique:
            out["ideate_critique"] = critique
        if tournament_result:
            out["ideate_tournament"] = tournament_result
        return out

    async def _run_ideate_tournament(
        self,
        state: QuestState,
        ideas: list[dict[str, Any]],
        *,
        initial_chosen: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run C(N, 2) pairwise comparisons across ``ideas`` and return
        ``(winner_idea, tournament_record)``.

        The record carries the full match table for run.log + Axon
        write-back so a future quest can see "we already tried these
        ideas; the tournament picked X over Y because Z."

        Each comparison fires one ``self._chat`` call to the
        ``ideate_tournament`` prompt, returning ``{winner: A|B,
        reason: ..., margin: decisive|narrow}``. The N matches run
        concurrently via ``asyncio.gather`` so total wall-clock is one
        LLM round-trip (plus parsing) regardless of N. For N=3
        (FI default), that's 3 calls in parallel vs the prior 1
        critique call serially.

        Tie-breaking: highest total wins. Ties are then broken by
        more "decisive" margins, then by EARLIEST original-list
        position (the ``-i`` term in the sort key). The earliest-
        position fallback is deterministic across runs but does NOT
        prefer ``initial_chosen`` — that fallback only fires when NO
        match resolved a clean winner (the "inconclusive_fallback"
        outcome below), so the engine always has a chosen_idea for
        downstream nodes.
        """
        from itertools import combinations

        pairs = list(combinations(range(len(ideas)), 2))
        self._log.info(
            "[ideate] tournament: %d ideas, %d pairwise matches",
            len(ideas), len(pairs),
        )

        async def play(a_idx: int, b_idx: int) -> dict[str, Any]:
            prompt = self._prompts["ideate_tournament"].substitute(
                topic=state["topic"],
                clarify_block=_format_clarify(state),
                idea_a=json.dumps(ideas[a_idx], indent=2),
                idea_b=json.dumps(ideas[b_idx], indent=2),
            )
            text = await self._chat(prompt, node="ideate_tournament")
            parsed = _parse_json_lenient(text) or {}
            winner_label = str(parsed.get("winner", "")).strip().upper()
            return {
                "a_idx": a_idx, "b_idx": b_idx,
                "winner": winner_label,                  # "A" or "B" or ""
                "reason": parsed.get("reason", ""),
                "margin": parsed.get("margin", "narrow"),
            }

        matches = await asyncio.gather(
            *(play(a, b) for a, b in pairs), return_exceptions=True,
        )

        # Tally wins. Skip failed matches (Exception or empty winner).
        wins = [0] * len(ideas)
        decisive_wins = [0] * len(ideas)
        valid_matches: list[dict[str, Any]] = []
        for m in matches:
            if isinstance(m, Exception):
                self._log.warning("[ideate] tournament match raised: %s", m)
                continue
            valid_matches.append(m)
            w = m["winner"]
            idx = m["a_idx"] if w == "A" else m["b_idx"] if w == "B" else -1
            if idx >= 0:
                wins[idx] += 1
                if m["margin"] == "decisive":
                    decisive_wins[idx] += 1

        if not any(wins):
            self._log.info(
                "[ideate] tournament inconclusive (no valid match outcomes) "
                "— keeping initial chosen=%r",
                initial_chosen.get("title", "?"),
            )
            return initial_chosen, {
                "matches": valid_matches, "winner_idx": None,
                "wins": wins, "decisive_wins": decisive_wins,
                "outcome": "inconclusive_fallback",
            }

        # Pick by (wins, decisive_wins, original-order) so ties break
        # deterministically.
        winner_idx = max(
            range(len(ideas)),
            key=lambda i: (wins[i], decisive_wins[i], -i),
        )
        winner = dict(ideas[winner_idx])
        # Preserve the original rationale; append the tournament reason
        # for grep-ability in run.log + paper writeback.
        reasons = [
            m["reason"]
            for m in valid_matches
            if (m["winner"] == "A" and m["a_idx"] == winner_idx)
            or (m["winner"] == "B" and m["b_idx"] == winner_idx)
        ]
        winner["rationale"] = (
            initial_chosen.get("rationale", "") + " "
            + " ".join(f"[tournament] {r}" for r in reasons[:2])
        ).strip()
        self._log.info(
            "[ideate] tournament resolved: winner=%r wins=%d (decisive=%d) "
            "vs initial=%r",
            winner.get("title", "?"), wins[winner_idx],
            decisive_wins[winner_idx], initial_chosen.get("title", "?"),
        )
        # Compute the original-list index of ``initial_chosen`` so the
        # "swapped" / "confirmed" outcome is decided by position, not
        # title. Title comparison would mis-report a swap as
        # "confirmed" when two ideas share a title (e.g. both fall
        # back to the synthesized {"title": "fallback", ...}).
        initial_title = initial_chosen.get("title")
        initial_idx = next(
            (i for i, idea in enumerate(ideas)
             if idea.get("title") == initial_title),
            -1,
        )
        return winner, {
            "matches": valid_matches, "winner_idx": winner_idx,
            "wins": wins, "decisive_wins": decisive_wins,
            "outcome": "swapped" if winner_idx != initial_idx else "confirmed",
        }

    def _lit_kwargs(self, state: QuestState) -> dict[str, Any]:
        """Relevance-selection kwargs for the ``_format_lit*`` helpers, from
        the quest state + knowledge config — so each node's prompt gets the
        passages most relevant to the question, within the configured
        per-source character budget."""
        k = self.config.knowledge
        return {
            "query": _lit_query(state),
            "budget": k.literature_excerpt_chars,
            "mode": k.passage_ranking,
        }

    @staticmethod
    def _work_scope(state: QuestState) -> str:
        """Which scholarly record types academic search keeps for this quest.
        A quest with an experiment cites papers; one without (a survey, a
        history, a humanities or policy question) also keeps books and book
        chapters, where much of that scholarship is published. Keyed on the
        resolved run mode, not on a separate subject classifier, so a
        misclassified topic gets the other rule rather than no search."""
        if state.get("no_simulation_resolved"):
            return WORK_SCOPE_PAPERS_AND_BOOKS
        return WORK_SCOPE_PAPERS

    async def _node_literature(self, state: QuestState) -> QuestState:
        if self.config.engine.analyze_local_first:
            # --analyze: local-data-first, so NO external literature is
            # retrieved. Passthrough — the analysis rests on the user's
            # data, not on fetched sources.
            self._log.info("[literature] analyze_local_first — no retrieval")
            return {}
        chosen = state.get("chosen_idea") or {}
        prior = list(state.get("literature") or [])
        prior_iter = int(state.get("literature_iter") or 0)
        this_iter = prior_iter + 1
        # On a broaden_lit re-entry, the design has already been
        # written — fold its hypothesis into the query so the second
        # pass searches for evidence that addresses the SPECIFIC
        # design, not just the original chosen idea. First pass keeps
        # the lean ``title + topic`` query because design hasn't run
        # yet.
        hypothesis = ""
        if this_iter > 1 and state.get("design"):
            hypothesis = str(state["design"].get("hypothesis") or "")[:200]
            query = (chosen.get("title") or "") + " " + hypothesis + " " + state["topic"][:160]
            self._log.info(
                "[literature] iteration=%d (broaden_lit re-entry); query incorporates design.hypothesis",
                this_iter,
            )
        else:
            query = (chosen.get("title") or "") + " " + state["topic"][:200]
        # A topic statement is written for a person; a search engine matches
        # keywords. A real quest's 365-character topic came back from the
        # academic sources with HTTP 200 and zero hits, where a 79-character
        # keyword query found three on-target papers. One small call turns the
        # topic into the terms the field publishes under. On any failure the
        # concatenation above is kept, so a flaky model degrades to the old
        # query rather than to no search at all.
        scope = self._work_scope(state)
        queries = await self._derive_literature_queries(
            state["topic"], chosen.get("title") or "", hypothesis, work_scope=scope,
        )
        if queries:
            self._log.info("[literature] search queries derived from the topic: %r", queries)
            query = queries[0]
        else:
            queries = [query.strip()]
        chat_fn = functools.partial(self._chat_messages, node="source_router")
        # One routing decision for all facets: they are phrasings of one
        # topic, and routing each would spend a call re-deriving the same list.
        sources = await self.knowledge.choose_sources(
            query, chosen_idea=chosen, chat_fn=chat_fn,
        )
        self._log.info(
            "[literature] searching sources=%s (knowledge.enabled=%s, "
            "source_routing=%s)",
            sources or "none", self.config.knowledge.enabled,
            self.config.knowledge.source_routing,
        )

        async def _retrieve(q: str, *, web: bool = True) -> list:
            return await self.knowledge.asearch(
                q.strip(),
                top_k=self.config.knowledge.top_k,
                # The literature node is the one path that explicitly wants
                # broad external retrieval when Axon misses — pass the
                # config's external cap so a web miss returns ~20 abstracts
                # instead of being silently capped at the Axon top_k.
                external_top_k=self.config.knowledge.external_top_k,
                chosen_idea=chosen,
                chat_fn=chat_fn,
                work_scope=scope,
                sources=sources,
                # Web search runs for the first facet only: the keyless
                # DuckDuckGo backend throttles a burst of queries, and the
                # other facets are there to reach scholarly work.
                web=web,
                # Full text is fetched once, below, for the sources that
                # survive the relevance screen -- not for every facet's hits.
                fetch_full_text=False,
            )

        per_facet = await asyncio.gather(*(
            _retrieve(q, web=(i == 0)) for i, q in enumerate(queries)
        ))
        docs = _merge_round_robin(list(per_facet))
        if docs:
            why_none = ""
        elif not self.config.knowledge.enabled:
            why_none = " — knowledge.enabled is false, so nothing was searched"
        else:
            why_none = (
                " — no source returned anything; look for [source-failures] at "
                "the end of this log, and check network access"
            )
        self._log.info(
            "[literature] hits per query=%s, %d after merging%s",
            [len(x) for x in per_facet], len(docs), why_none,
        )
        # Relevance floor: drop off-topic sources the retriever returned before
        # they reach the corpus. This is the ONLY relevance filter on the
        # literature path — the LLM guard runs only under auto_collect, which
        # survey / simulation quests skip — so without this a humanities topic
        # carries e.g. change-point-math papers into analyze/write. Scored vs
        # the raw topic; fail-open + never-starve (see _filter_docs_by_relevance).
        rel_topic = (state.get("topic") or "").strip() or query
        stats: dict = {}
        filtered = self._filter_docs_by_relevance(rel_topic, docs, stats=stats)

        # Re-search with different keywords when the whole retrieval missed.
        # ``above_floor == 0`` means not one source cleared the threshold on
        # its own merits and only never-starve retention kept anything — a
        # symptom of a badly-worded query, not of a topic with no literature.
        # Proceeding here is what produces "it gave me three irrelevant
        # papers": min_keep pads the set and the writer treats the padding as
        # evidence. Retrying with the model's alternative phrasings is the
        # principled fix. Bounded, and skipped when unscored (see config).
        kn = self.config.knowledge
        tried_queries = list(queries)
        if kn.requery_on_low_relevance and stats.get("scored") and docs:
            attempt = 0
            while stats.get("above_floor", 0) == 0 and attempt < kn.requery_max:
                attempt += 1
                alt = await self._propose_literature_queries(
                    rel_topic, tried_queries, docs, work_scope=scope,
                )
                if not alt:
                    self._log.info(
                        "[literature] requery %d: no alternative query proposed; "
                        "keeping the original results", attempt,
                    )
                    break
                self._log.info(
                    "[literature] requery %d/%d: nothing cleared the relevance "
                    "floor (best cosine=%.2f) — retrying with %r",
                    attempt, kn.requery_max, stats.get("best", 0.0), alt,
                )
                tried_queries.append(alt)
                more = await _retrieve(alt)
                if not more:
                    continue
                # Merge rather than replace: the first pass may still hold the
                # single on-topic hit, and dedup happens downstream anyway.
                docs = docs + more
                stats = {}
                filtered = self._filter_docs_by_relevance(
                    rel_topic, docs, stats=stats,
                )
            if stats.get("above_floor", 0) > 0 and attempt:
                self._log.info(
                    "[literature] requery succeeded after %d retry(ies): "
                    "%d doc(s) now clear the floor",
                    attempt, stats.get("above_floor", 0),
                )
        docs = filtered
        # The original papers and textbooks a keyword search does not reach
        # join the candidates, and the screen judges them like the rest.
        docs = docs + await self._foundational_works(rel_topic, docs, work_scope=scope)
        # The floor scores word overlap; the screen asks whether the paper
        # could cite each source for a claim (see _screen_literature).
        n_before_screen = len(docs)
        docs = await self._screen_literature(rel_topic, docs, work_scope=scope)
        self._log.info(
            "[literature] kept %d of %d after the relevance floor and the screen",
            len(docs), n_before_screen,
        )
        # Legal full text for the scholarly sources that were kept (web pages
        # already carry their page text). Once here rather than per facet.
        docs = await self.knowledge.fetch_full_text(docs)
        # Keep the FULL fetched text (no truncation): it lands uncapped on
        # disk under data/literature/ for audit, and the prompt builders
        # relevance-select the passages each node needs (see
        # _format_lit_excerpt / knowledge.literature_excerpt_chars) rather
        # than the old first-2000-chars slice that usually held only the
        # abstract. content_quality records whether real full text was
        # recovered or only the search snippet survived.
        new_entries = [
            {
                "content": d.content,
                "metadata": {
                    **d.metadata,
                    "content_quality": (
                        "full_text" if d.metadata.get("fetched_full_text")
                        else "snippet_only"
                    ),
                },
            }
            for d in docs
        ]
        # Dedup-merge: identity is DOI when present, otherwise the
        # canonical URL, otherwise the first 200 chars of content.
        # On the first iteration ``prior`` is empty so this is a
        # straight assignment; on broaden_lit re-entries we accumulate
        # so the design node sees the full corpus FI has seen for
        # this quest.
        # A specific-enough title is a second identity: the same text reached
        # under several DOIs (or once with a DOI, once without) is one source.
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for entry in (*prior, *new_entries):
            md = entry.get("metadata") or {}
            ident = (
                str(md.get("doi") or "").strip()
                or str(md.get("url") or "").strip()
                or (entry.get("content") or "")[:200]
            )
            norm_title = _normalize_title(md.get("title") or "")
            idents = [i for i in (ident, f"title:{norm_title}" if norm_title else "") if i]
            if any(i in seen for i in idents):
                continue
            seen.update(idents)
            merged.append(entry)
        added = len(merged) - len(prior)
        # Pull in any PDFs the user dropped under ``inputs/papers/`` on
        # a previous pause cycle. Indexes them as new ``user_supplied``
        # literature entries so the design / write nodes see real
        # full text instead of only the upstream abstracts.
        merged, user_added = _ingest_user_dropped_papers(
            self.quest_root, merged, seen, self._log,
        )
        if user_added:
            self._log.info(
                "[literature] picked up %d user-supplied paper(s) from inputs/papers/",
                user_added,
            )
        self._log.info(
            "[literature] retrieved %d docs (iter=%d, +%d new after dedup, "
            "+%d user-supplied, total=%d)",
            len(docs), this_iter, added, user_added, len(merged),
        )
        full_n = sum(
            1 for e in merged
            if (e.get("metadata") or {}).get("content_quality") == "full_text"
        )
        self._log.info(
            "[literature] full-text coverage: %d/%d sources have real full "
            "text (%d snippet-only)",
            full_n, len(merged), len(merged) - full_n,
        )

        # Pause-for-user-papers gate. Fires only when the user opted in
        # AND we have abstract-only hits (heuristic: content shorter
        # than ~1500 chars or carrying an explicit ``abstract_only``
        # flag from the retriever). On a re-entry after the user
        # dropped PDFs, ``inputs/papers/`` is non-empty and we don't
        # pause again — the resume path picks up the new files and
        # proceeds.
        if (
            self.config.pauses.papers
            and not _papers_dir_has_files(self.quest_root)
        ):
            abstract_only = [d for d in docs if _is_abstract_only(d)]
            # Split the genuinely paywalled from open-access sources we simply
            # failed to fetch. Only the former justify stopping the quest to
            # ask a person for help: an arXiv/PMC paper we could not download
            # is OUR network problem, and a pause that asks the user to fetch
            # a free paper is both confusing and usually futile (the same host
            # is behind the same proxy). The OA ones are still listed in
            # WANTED_PAPERS.md as a manual fallback -- a browser often works
            # where httpx does not -- but they never trigger the pause.
            oa_unfetched = [d for d in abstract_only if _is_open_access(d)]
            needed = [d for d in abstract_only if not _is_open_access(d)]
            if oa_unfetched:
                self._log.warning(
                    "[literature] %d open-access source(s) (arXiv/PMC/preprint) "
                    "came back abstract-only -- full-text fetch failed for "
                    "sources that are free to download. This usually means "
                    "the host is unreachable (proxy/firewall), not a paywall. "
                    "Listed in WANTED_PAPERS.md as a manual fallback; not "
                    "pausing the quest for them.",
                    len(oa_unfetched),
                )
            if needed or oa_unfetched:
                _write_paper_need_stubs(
                    self.quest_root, needed, self._log,
                    query=_lit_query(state), oa_unfetched=oa_unfetched,
                )
            if needed:
                self._pause_for_human(
                    kind="papers",
                    interaction="supply",
                    headline=f"download {len(needed)} paywalled paper(s)",
                    steps=[
                        f"{len(needed)} relevant paper(s) came back abstract-only "
                        "(paywalled). They're listed, most-relevant-first, in "
                        "`needs/WANTED_PAPERS.md` with a download link each.",
                        "Download the few that matter and drop the PDFs into "
                        "`inputs/papers/` — they're ingested as full text.",
                    ],
                    payload={
                        "papers_required": True,
                        "quest_id": self.quest_id,
                        "papers_dir": str(self.quest_root / "inputs" / "papers"),
                        "needed_count": len(needed),
                    },
                    upload_targets=["papers"],
                )
                # Unreachable in practice (see ``wait_for_data`` for the
                # same pattern): interrupt() raises GraphInterrupt; on
                # resume this node is re-invoked from the checkpoint,
                # ``_ingest_user_dropped_papers`` picks up the new
                # files, and the gate's else-branch falls through.
                return {}

        # Download the retrieved sources to disk for EVERY quest (sim and
        # no-sim alike) so the collected web/academic literature is always
        # auditable on disk under ``data/literature/``. The no-simulation
        # path additionally reuses the same corpus as analysis data via
        # auto_collect_data (``data/auto_collected/``); this write
        # guarantees a downloaded corpus even on the SIMULATION path,
        # where auto_collect_data never runs (design → implement →
        # execute …). Empty corpus → no files, no empty directory.
        downloaded = self._write_literature_files(
            merged, self.quest_root / "data" / "literature",
        )
        if downloaded:
            self._log.info(
                "[literature] downloaded %d source(s) to data/literature/",
                downloaded,
            )

        return {
            "literature": merged,
            "literature_iter": this_iter,
            "literature_query": query.strip(),
            "literature_queries": queries,
        }

    async def _node_design(self, state: QuestState) -> QuestState:
        if self.config.engine.analyze_local_first:
            # --analyze: no experiment to design — the data already exists.
            # Passthrough; routing keys on no_simulation_resolved (set by
            # clarify), so an empty design still flows to data_load.
            self._log.info("[design] analyze_local_first — no experiment design")
            return {}
        iteration = state.get("iteration", 0)
        self._log.info("[design] iteration=%d", iteration)
        review_feedback = ""
        if iteration > 0:
            review_feedback = json.dumps(state.get("review", {}), indent=2)
        # When this redesign was triggered by analyze's ``re_experiment``
        # reroute (the previous experiment produced non-physical or unsupported
        # results), the review block above is empty — the review node only runs
        # AFTER the paper is written, never before a pre-write re_experiment
        # loop. Without the analyze/cross_check diagnosis the redesign is blind
        # to WHY the last experiment was rejected and tends to regenerate the
        # same flaw. Fold the analysis summary + key findings in so the design
        # LLM can diagnose and correct the root cause rather than repeat it.
        analysis = state.get("analysis") or {}
        if iteration > 0 and analysis.get("next_step") == "re_experiment":
            findings_txt = json.dumps(analysis.get("key_findings") or [], indent=2)
            diag = (
                "--- PRIOR EXPERIMENT REJECTED BY ANALYSIS "
                "(diagnose and fix the root cause; do NOT reproduce it) ---\n"
                "The previous experiment ran but its results were judged "
                "non-physical or unsupported. Before redesigning, work out why "
                "the results were wrong — common causes are unit or scale "
                "inconsistencies, comparing two quantities defined on different "
                "scales, degenerate or biased estimators, and parameters that "
                "make an effect vanish — then change the experimental plan so "
                "the numbers become physically sensible and internally "
                "consistent.\n"
                f"Analysis summary: {analysis.get('summary', '')}\n"
                f"Key findings from the rejected run:\n{findings_txt}"
            )
            review_feedback = f"{review_feedback}\n\n{diag}".strip()
        # Human-feedback refinement (when the gate is configured AND the
        # user picked "refine"). Folded into the same review_feedback
        # block the design prompt already reads — explicitly attributed
        # so the LLM understands this came from a real user, not the
        # auto-review. Uses the accumulated ``feedback_history`` so a
        # later revise pass honours every prior ask, not just the most
        # recent one. Falls back to the single-shot ``human_feedback``
        # dict for legacy state shapes (resumed pre-history checkpoints).
        history = list(state.get("feedback_history") or [])
        hf = state.get("human_feedback") or {}
        if not history and hf.get("action") == "refine" and hf.get("feedback"):
            history = [{"iteration": state.get("iteration", 1) - 1,
                        "text": hf["feedback"]}]
        if history:
            blocks = "\n\n".join(
                f"  (round {h.get('iteration', '?')}) {h.get('text', '')}".rstrip()
                for h in history if (h.get("text") or "").strip()
            )
            if blocks:
                review_feedback = (
                    f"{review_feedback}\n\n"
                    f"--- USER FEEDBACK (priority over auto-review above; "
                    f"honour every round below, not only the most recent) ---\n"
                    f"{blocks}\n"
                ).strip()
        prompt = self._prompts["design"].substitute(
            topic=state["topic"],
            chosen_idea=json.dumps(state.get("chosen_idea") or {}, indent=2),
            # The full excerpts, as analyze and write get. At 800 characters a
            # source, design chose a 1% major-outbreak threshold in 6 of 10
            # replays of a real quest (0 of 5 at the full budget): short
            # excerpts cost it the context a sound design rests on.
            literature_block=_format_lit_from_state(state, **self._lit_kwargs(state)),
            review_feedback=review_feedback or "(none — first iteration)",
            timeout_s=str(self.config.execution.timeout_s),
            clarify_block=_format_clarify(state),
            skills_block=(
                self._skills_summary_block(state)
                or "(no skills selected for this quest)"
            ),
            inputs_block=self._inputs_block(),
            job_block=self._job_block(),
            study_mode_directive=(
                _SURVEY_DESIGN_DIRECTIVE
                if state.get("survey_mode_resolved")
                else _NO_SIM_DESIGN_DIRECTIVE
                if state.get("no_simulation_resolved")
                else ""
            ),
        )
        text = await self._chat(prompt, node="design")
        design = _parse_json_lenient(text) or {"hypothesis": "(parse failed)", "dependencies": []}

        # Second-pass methodology audit. The draft design just produced is
        # passed back to the LLM with a fixed checklist of common-but-fatal
        # design errors (circular evaluation, single-point eval, weak
        # baseline plans, pseudo-units, natural-stratum collapse) and a
        # mandate to either patch them or confirm non-applicability.
        #
        # Cost: +1 LLM call per design pass. For a typical 2-iteration
        # quest, +2 calls in the design lane. Worth it because design
        # errors are O(quest cost) to fix at review time but O(critique
        # call cost) to fix here, BEFORE implement / execute / analyze
        # / write / review have spent compute building on a bad design.
        #
        # Failure isolation: this whole block is wrapped in
        # try/except so a transient provider/network error on the
        # critique call (``_chat`` itself can raise) NEVER blocks the
        # quest. Parse failures and shape drift on the response are
        # also non-fatal — the original draft survives in those
        # cases. The audit is strictly advisory.
        critique_prompt = self._prompts["design_self_critique"].substitute(
            topic=state["topic"],
            chosen_idea=json.dumps(state.get("chosen_idea") or {}, indent=2),
            clarify_block=_format_clarify(state),
            draft_design=json.dumps(design, indent=2),
        )
        critique: dict[str, Any] = {}
        try:
            critique_text = await self._chat(
                critique_prompt, node="design_self_critique",
            )
            critique = _parse_json_lenient(
                critique_text, node="design_self_critique",
            ) or {}
        except Exception as e:  # noqa: BLE001 — see "Failure isolation" above
            self._log.warning(
                "[design_self_critique] chat/parse failed (%r); keeping "
                "un-audited draft design", e,
            )
        amended = critique.get("amended_design") if isinstance(critique, dict) else None
        objections = critique.get("objections_addressed") if isinstance(critique, dict) else None
        if isinstance(amended, dict) and amended:
            # The amended design must keep the original design's shape —
            # otherwise downstream consumers (implement / analyze /
            # write) will silently mis-read missing keys. Require the
            # SAME set of top-level keys the draft had; on schema
            # drift, fall back to the draft rather than ship a partial
            # design. (Stricter than the prior "hypothesis only" check,
            # which would silently drop variables / method /
            # figures_planned / dependencies.)
            draft_keys = set(design.keys())
            amended_keys = set(amended.keys())
            if draft_keys.issubset(amended_keys):
                design = amended
            else:
                self._log.warning(
                    "[design_self_critique] amended_design dropped keys "
                    "%s; keeping draft", sorted(draft_keys - amended_keys),
                )
        n_addressed = len(objections) if isinstance(objections, list) else 0
        self._log.info(
            "[design_self_critique] iteration=%d objections_addressed=%d",
            iteration, n_addressed,
        )

        out: dict[str, Any] = {"design": design}
        # Provenance for the hypothesis itself. The DAG lets `review` and
        # `cross_check` route back here, so a design CAN be rewritten after
        # its results are known. That iteration is legitimate research, but a
        # paper that presents a post-hoc hypothesis as though it were
        # pre-specified is not -- it is the thing methodologists call HARKing,
        # and the reader has no way to detect it from the finished paper.
        #
        # So record every version: what it was, when, and what sent the engine
        # back here. Nothing is blocked -- the record exists so the revision is
        # auditable rather than invisible.
        out["design_history"] = _append_design_revision(
            state, design, self.quest_root, self._log,
        )
        # Range assertions contributed by the trusted skills this quest may
        # call. Stashed in state because ``_assertion_violations`` is a
        # module-level function with no access to the engine, and the
        # execute-repair loop needs them before any node with `self` runs
        # again. A skill declares its own valid domain once; the design does
        # not have to restate it.
        skill_assertions = self._skill_assertions(state)
        if skill_assertions:
            out["_skill_assertions"] = skill_assertions
            self._log.info(
                "[skills] %d range assertion(s) contributed by skills",
                len(skill_assertions),
            )
        if isinstance(objections, list):
            # Surfaced into state so it can be inspected post-quest (run.log
            # already carries the count; the full list lives here for any
            # caller that wants to render it).
            out["design_objections"] = objections

        # Generic pause-drop-anytime: when configured, pause AFTER the
        # design lands so the user can drop reference papers and/or
        # datasets BEFORE the implement/execute spends LLM + venv
        # compute. On a post-pause resume, ``user_pauses_fired``
        # carries "before_build" and the gate falls through. Files
        # the user dropped under ``inputs/data/`` get walked into
        # ``user_supplied_datasets`` so the analyze node sees them.
        self._maybe_pause_for_user_input(state, "before_build")
        # _maybe_pause_for_user_input either fires interrupt() (which
        # never returns) or no-ops. If it no-ops and there are user
        # data files to pick up, surface them on the patch.
        ds_added = _pick_up_user_dropped_datasets(self.quest_root)
        if ds_added:
            existing = list(state.get("user_supplied_datasets") or [])
            merged = list(dict.fromkeys(existing + ds_added))
            out["user_supplied_datasets"] = merged
            self._log.info(
                "[design] picked up %d user-supplied dataset(s) from inputs/data/",
                len(ds_added),
            )
        return out

    def _write_next_step(
        self,
        *,
        kind: str,
        interaction: str,
        headline: str,
        steps: list[str],
    ) -> None:
        """Write the one consistent ``NEXT_STEP.md`` the user looks at whenever
        the quest pauses — same shape for every pause, whether the quest is
        asking a question (ANSWER) or waiting on files (SUPPLY). It says why it
        stopped, exactly what to do, and the resume command. Best-effort; a
        write failure is logged but never quest-fatal."""
        verb = "ANSWER" if interaction == "answer" else "SUPPLY"
        body = [
            f"# Action needed — {headline}",
            "",
            f"Quest **{self.quest_id}** is paused and waiting for you "
            f"(**{verb}**).",
            "",
            "## What to do",
            *[f"{i}. {s}" for i, s in enumerate(steps, 1)],
            "",
            "## Then resume",
            f"- **CLI:** `fi --resume {self.quest_id}`",
            "- **Web / VSCode:** open the quest and click **Resume** — an "
            "*Action needed* banner shows there too.",
            "",
        ]
        try:
            (self.quest_root / "NEXT_STEP.md").write_text(
                "\n".join(body), encoding="utf-8",
            )
        except OSError as e:
            self._log.warning("[%s] couldn't write NEXT_STEP.md: %r", kind, e)

    def _pause_for_human(
        self,
        *,
        kind: str,
        interaction: str,
        headline: str,
        steps: list[str],
        payload: dict[str, Any],
        upload_targets: list[str] | None = None,
    ) -> Any:
        """The single way the engine stops for a human. Writes the unified
        ``NEXT_STEP.md``, logs a consistent line, then fires LangGraph's
        ``interrupt()``.

        ``kind`` is the pause id (clarify / papers / supply / data / review);
        ``interaction`` is ``"answer"`` (the quest asks; resume returns the
        answer) or ``"supply"`` (the quest needs files; resume re-enters and
        re-checks). ``payload`` keeps each pause's existing keys so the CLI /
        web / VSCode resume handlers are unchanged; a unified ``pause``
        descriptor is added so those surfaces can render one *Action needed*
        affordance without knowing each pause's bespoke shape.

        Returns whatever the resume sent (ANSWER pauses). SUPPLY pauses
        pause-exit: ``interrupt()`` raises ``GraphInterrupt`` and never returns.
        """
        self._write_next_step(
            kind=kind, interaction=interaction, headline=headline, steps=steps,
        )
        descriptor = {
            "kind": kind,
            "interaction": interaction,
            "headline": headline,
            "quest_id": self.quest_id,
            "next_step_file": "NEXT_STEP.md",
            # For a SUPPLY pause: which upload target(s) the web banner should
            # offer (papers → inputs/papers, data → inputs/data, root_data →
            # data/). Empty for an ANSWER pause (the web reveals the form).
            "upload_targets": list(upload_targets or []),
        }
        # Authoritative on-disk descriptor so the web can render the right
        # affordance for a subprocess quest (the in-process payload below isn't
        # visible across processes). Cleared with NEXT_STEP.md on completion.
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / "pause.json").write_text(
                json.dumps(descriptor, indent=2) + "\n", encoding="utf-8",
            )
        except OSError as e:
            self._log.debug("[%s] pause.json write failed: %r", kind, e)
        self._log.info("[%s] paused — %s", kind, headline)
        return interrupt({**payload, "pause": descriptor})

    def _pause_stage_enabled(self, stage: str) -> bool:
        """Whether ``pauses.supply`` asks for a stop at ``stage``. ``both`` is
        before_build + before_review, as it always was; ``all`` adds the stop
        after the literature."""
        value = self.config.pauses.supply
        if value == "all":
            return True
        if value == "both":
            return stage in ("before_build", "before_review")
        return value == stage

    async def _node_pause_after_literature(self, state: QuestState) -> QuestState:
        """The literature is done and saved. When ``pauses.supply`` asks for it,
        stop here: skills, design and the experiment start on ``--resume`` with
        the literature already in state (the search is not run again). Papers
        dropped in ``inputs/papers/`` meanwhile join it. A node of its own so a
        resume re-enters this cheap node, not the search."""
        if not self._pause_stage_enabled("after_literature"):
            return {}
        self._maybe_pause_for_user_input(state, "after_literature")
        merged = list(state.get("literature") or [])
        seen = {i for entry in merged for i in _entry_identities(entry)}
        merged, added = _ingest_user_dropped_papers(
            self.quest_root, merged, seen, self._log,
        )
        if not added:
            return {}
        self._log.info(
            "[after_literature] picked up %d paper(s) dropped in inputs/papers/ "
            "while paused", added,
        )
        return {"literature": merged}

    def _maybe_pause_for_user_input(
        self, state: QuestState, stage: str,
    ) -> None:
        """Fire a LangGraph ``interrupt()`` at the named stage when
        ``pauses.supply`` opts in AND this stage hasn't already fired for
        this quest. ``stage`` is one of ``"before_build"`` / ``"before_review"``.
        ``state['user_pauses_fired']`` tracks per-quest fired pause names so a
        resume re-entry doesn't force a second pause at the same stage.

        On pause, the engine writes the unified ``<quest_root>/NEXT_STEP.md``
        (via ``_pause_for_human``) explaining the drop zones, then raises
        interrupt(). The user drops files and re-runs ``fi --resume <id>``;
        the resume picks up the files via the normal literature / analyze paths.
        """
        if not self._pause_stage_enabled(stage):
            return
        # Disk marker is the authoritative "already paused at this
        # stage" signal — same pattern as wait_for_data uses with the
        # data dir's file count. Using a state list (``user_pauses_fired``)
        # alone would fail on the pause-exit + --resume flow because
        # LangGraph's ``interrupt()`` raises BEFORE the node returns
        # a state patch, so the partial-state update never lands in
        # the checkpoint. Disk markers survive any resume path.
        inputs_dir = self.quest_root / "inputs"
        papers_dir = inputs_dir / "papers"
        data_dir = inputs_dir / "data"
        papers_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        marker = self.fi_dir / f"paused_at_{stage}.flag"
        already = list(state.get("user_pauses_fired") or [])
        if marker.is_file() or stage in already:
            self._log.info(
                "[%s] pause-for-user-input already fired this quest; "
                "skipping (resume path)", stage,
            )
            return
        # Commit the "this stage paused" marker BEFORE firing interrupt
        # so a --resume of the (checkpointed) graph re-enters this node
        # and the marker.is_file() check above falls through.
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(stage, encoding="utf-8")
        except OSError as e:
            self._log.warning(
                "[%s] couldn't write pause marker %s: %r", stage, marker, e,
            )
        when = {
            "after_literature": "after the literature search, before skills, "
                                "design and the experiment",
            "before_build": "before the experiment is built / run",
        }.get(stage, "before the draft goes to review")
        steps = [
            f"Optional — paused {when} so you can add your own sources.",
            "Drop reference papers (PDF / Markdown) into `inputs/papers/` "
            "— they become citable literature.",
            "Drop datasets (CSV / JSON / TSV / Parquet) into `inputs/data/` "
            "— the analyze node reasons over them.",
        ]
        if stage in ("after_literature", "before_build"):
            steps.append(
                "Drop example files for the experiment (simulation settings, "
                "input decks, configs, scripts, documents — any type) into "
                "`inputs/examples/` — the design and the experiment code are "
                "written from them, combined with the selected skills."
            )
        if stage == "after_literature":
            steps.insert(
                1, "The literature is saved: resuming continues with it and "
                "does not search again.",
            )
        self._pause_for_human(
            kind="supply",
            interaction="supply",
            headline=f"add any papers or data ({stage})",
            steps=steps,
            payload={
                "user_input_required": True,
                "stage": stage,
                "quest_id": self.quest_id,
                "inputs_dir": str(inputs_dir),
                "papers_dir": str(papers_dir),
                "data_dir": str(data_dir),
            },
            upload_targets=["papers", "data"],
        )

    async def _node_auto_collect_data(self, state: QuestState) -> QuestState:
        """Agent-side data collection via Axon, run BEFORE
        the wait_for_data pause in no-simulation mode.

        Why this exists: the user said *"data can be collected by
        agent as well, not users only"* — for many no-simulation
        topics (literature reviews, cross-cultural comparisons,
        history surveys) Axon already holds enough to answer the
        question, and there's no reason to interrupt the user when
        the corpus already covers the topic. This node tries to
        pull ``top_k`` relevant docs and writes each one as a
        Markdown file under ``<quest_root>/data/auto_collected/`` so
        the downstream nodes (``wait_for_data`` → ``data_load``) see
        them as ordinary user data and proceed without pausing.

        Passthrough discipline (each case is distinguishable in run.log):
        * ``engine.auto_collect_data`` is False — user opted out. No
          Axon call. Logged INFO.
        * ``knowledge.enabled`` is False — no Axon to query. No Axon
          call. Logged INFO.
        * ``Knowledge.asearch`` IS called but raises — the exception
          is caught, logged WARNING, and the node returns
          ``auto_collected_count=0`` so ``wait_for_data`` takes over.
        * ``Knowledge.asearch`` returns zero docs — logged INFO. No
          files written. Empty ``auto_collected/`` directory is NOT
          left behind.

        On resume the node re-runs and re-walks Axon — files at the
        same rank slot get overwritten with the latest retrieval. The
        user's manually dropped files (anywhere in ``data/`` other
        than ``auto_collected/``) are untouched on each pass.
        """
        if not self.config.engine.auto_collect_data:
            self._log.info(
                "[auto_collect] engine.auto_collect_data=False — "
                "skipping; will pause for user data",
            )
            return {"auto_collected_count": 0}

        # Build the query once and reuse for BOTH Axon and dataset
        # adapters. Topic alone is enough for first-pass retrieval;
        # hypothesis sharpens it on later iterations.
        topic = state.get("topic", "")
        design = state.get("design") or {}
        hypothesis = ""
        if isinstance(design, dict):
            hypothesis = str(design.get("hypothesis", "")).strip()
        query = f"{topic} {hypothesis}".strip() or topic
        auto_dir = self.quest_root / "data" / "auto_collected"

        # ---- Reuse already-downloaded literature -------------------
        # The literature node ran earlier in the graph and already fetched
        # (and, with web_fetch_pages, read the full text of) a set of
        # web/Axon sources into state["literature"] AND wrote them to
        # data/literature/. data_load walks the whole data/ tree, so we
        # reuse those directly — counting them here (so the no-sim gate
        # doesn't pause) WITHOUT writing a duplicate copy under
        # auto_collected/. Avoids re-querying the web (wasteful, and the
        # keyless DuckDuckGo backend can rate-limit the Nth query) and
        # avoids double-feeding the same content into the analysis.
        # Reuse already-downloaded literature — but DROP off-topic sources
        # so an irrelevant hit (the classic "banana bread" result for a
        # "SpaceX revenue" topic) can't silently satisfy the no-sim gate
        # without the relevance check the Axon top-up path already applies.
        lit_docs = [
            d for d in (state.get("literature") or [])
            if isinstance(d, dict)
            and ((d.get("metadata") or {}).get("title")
                 or (d.get("metadata") or {}).get("url"))
            and (d.get("metadata") or {}).get("kind") not in _FI_INTERNAL_KINDS
        ]
        if lit_docs and self.config.knowledge.relevance_guard:
            relevant = await self._filter_relevant_docs(query, lit_docs)
            lit_written = len(relevant)
            if lit_written < len(lit_docs):
                self._log.info(
                    "[auto_collect] relevance guard: %d/%d reused literature "
                    "doc(s) on-topic — off-topic ones don't satisfy the "
                    "no-sim gate", lit_written, len(lit_docs),
                )
            elif lit_written:
                self._log.info(
                    "[auto_collect] reusing %d already-downloaded literature "
                    "doc(s) from data/literature/", lit_written,
                )
        else:
            lit_written = self._literature_seed_step(state)

        # ---- Axon / web retrieval (top-up) -------------------------
        # Only fire a fresh knowledge-layer query when the literature
        # node gave us nothing to reuse — avoids the redundant,
        # rate-limit-prone re-search in the common case.
        axon_written = 0
        if lit_written == 0:
            axon_written = await self._axon_collect_step(
                query, auto_dir, work_scope=self._work_scope(state),
            )

        # ---- Dataset adapters --------------------------------------
        adapter_written = await self._run_dataset_adapters(query, auto_dir)

        # lit_written is a COUNT of sources reused from data/literature/ (no
        # files written there); only axon + adapters write into auto_dir.
        files_written = axon_written + adapter_written
        written = lit_written + files_written
        # If nothing landed in auto_collected/, clean up any empty top-level
        # dir that might have been created mid-flight.
        if files_written == 0 and auto_dir.is_dir() and not any(auto_dir.iterdir()):
            try:
                auto_dir.rmdir()
            except OSError as e:
                self._log.warning(
                    "[auto_collect] zero writes AND could not rmdir "
                    "leftover %s: %s", auto_dir, e,
                )
        self._log.info(
            "[auto_collect] %d source(s) available for analysis "
            "(literature_reuse=%d from data/literature/, axon=%d + "
            "adapters=%d written to auto_collected/)",
            written, lit_written, axon_written, adapter_written,
        )
        return {"auto_collected_count": written}

    def _literature_seed_step(self, state: QuestState) -> int:
        """Count the web/Axon sources the literature node already fetched
        (``state['literature']``) so the no-simulation gate
        (``auto_collected_count``) treats the quest as having data and
        doesn't pause for user uploads.

        Does NOT write a second copy: those sources were already downloaded
        to ``data/literature/`` (the literature node does this for every
        quest), and ``data_load`` walks the whole ``data/`` tree — so a
        duplicate copy under ``auto_collected/`` only double-feeds the same
        content into the analysis. Counts the same citable entries
        ``_write_literature_files`` would keep (a title or URL, not an
        FI-internal artifact)."""
        n = 0
        for item in state.get("literature") or []:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") or {}
            if not (meta.get("title") or meta.get("url")):
                continue
            if meta.get("kind") in _FI_INTERNAL_KINDS:
                continue
            n += 1
        if n:
            self._log.info(
                "[auto_collect] reusing %d already-downloaded literature "
                "doc(s) from data/literature/ (no duplicate copy written)", n,
            )
        return n

    def _write_literature_files(self, lit: list, out_dir: Path) -> int:
        """Write each citable literature entry to ``out_dir`` as a
        ``lit_NNN_slug.md`` file (YAML front matter + content, via
        ``_render_auto_collected_md``). Called by the ``literature`` node's
        always-on download into ``data/literature/`` (runs for EVERY quest,
        sim or no-sim) — the single on-disk copy of the retrieved sources,
        which ``data_load`` then reuses on the no-simulation path.

        Skips entries with no title/url and FI-internal cross-quest
        artifacts. ``out_dir`` is created lazily so a corpus with nothing
        citable leaves no empty directory behind. Returns the count
        written."""
        written = 0
        for idx, item in enumerate(lit, start=1):
            if isinstance(item, dict):
                meta = item.get("metadata") or {}
                content = item.get("content") or ""
            else:
                meta = getattr(item, "metadata", {}) or {}
                content = getattr(item, "content", "") or ""
            if not (meta.get("title") or meta.get("url")):
                continue
            if meta.get("kind") in _FI_INTERNAL_KINDS:
                continue
            slug = _slugify(
                str(meta.get("title") or meta.get("source") or f"lit{idx}")
            )[:40] or f"lit{idx}"
            target = out_dir / f"lit_{idx:03d}_{slug}.md"
            body = _render_auto_collected_md(idx, meta, content)
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
                written += 1
            except OSError as e:
                self._log.warning(
                    "[literature] failed to write %s: %s — skipping",
                    target, e,
                )
        return written

    async def _axon_collect_step(
        self, query: str, auto_dir: Path, *, work_scope: str = WORK_SCOPE_PAPERS,
    ) -> int:
        """Axon-backed retrieval. Returns the
        count of files written under ``auto_dir`` (not in a
        sub-directory). Returns 0 on any of: knowledge disabled,
        asearch raised, zero hits, or every write failed. The
        caller is responsible for combining this with the dataset-
        adapter count and for the final empty-dir cleanup.
        """
        if not self.knowledge.enabled:
            self._log.info(
                "[auto_collect] knowledge.enabled=False — skipping "
                "Axon retrieval (dataset adapters may still run)",
            )
            return 0
        top_k = self.config.engine.auto_collect_top_k
        self._log.info(
            "[auto_collect] querying knowledge layer (Axon + web): "
            "top_k=%d query=%r",
            top_k, query[:120],
        )
        try:
            # Pass chat_fn so the source router (source_routing="auto") can
            # pick the right sources for the topic — crucially, for a
            # non-academic question ("SpaceX revenue") it selects general
            # web search and skips the scholarly adapters that would
            # otherwise return irrelevant nearest-neighbour papers. The
            # web layer runs in parallel regardless; routing just spares
            # the wasted academic calls.
            docs = await self.knowledge.asearch(
                query,
                top_k=top_k,
                chat_fn=functools.partial(self._chat_messages, node="source_router"),
                work_scope=work_scope,
            )
        except Exception as e:
            self._log.warning(
                "[auto_collect] knowledge search raised: %s — dataset "
                "adapters may still run", e,
            )
            return 0
        if not docs:
            self._log.info(
                "[auto_collect] knowledge layer returned 0 docs — dataset "
                "adapters may still run",
            )
            return 0

        # Relevance guard: a retrieval can come back full of confident-but-
        # irrelevant hits (the classic failure: academic adapters returning
        # physics preprints for a "SpaceX revenue" question). Drop the
        # off-topic ones so we don't write a paper on garbage; if NOTHING
        # is on-topic, return 0 so wait_for_data pauses for real user data
        # instead of proceeding.
        docs = await self._filter_relevant_docs(query, docs)
        if not docs:
            self._log.info(
                "[auto_collect] relevance guard dropped every auto-collected "
                "doc as off-topic — pausing for user-supplied sources",
            )
            return 0

        written_targets: list[Path] = []
        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            source = str(meta.get("source") or meta.get("path") or "")
            slug_basis = Path(source).stem if source else f"doc{idx}"
            slug = _slugify(slug_basis)[:40] or f"doc{idx}"
            target = auto_dir / f"{idx:03d}_{slug}.md"
            body = _render_auto_collected_md(idx, meta, doc.content or "")
            try:
                auto_dir.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
                written_targets.append(target)
            except OSError as e:
                self._log.warning(
                    "[auto_collect] failed to write %s: %s — skipping doc",
                    target, e,
                )
        return len(written_targets)

    async def _filter_relevant_docs(self, topic: str, docs: list) -> list:
        """Relevance guard: ask the model which of the retrieved docs are
        actually on-topic and keep only those. A single cheap judging call
        (not one per doc). On any failure — guard disabled, LLM error,
        unparseable response — return the docs unchanged (fail-open: never
        silently drop a quest's whole corpus because the judge hiccuped).
        """
        if not docs or not self.config.knowledge.relevance_guard:
            return docs
        listing_lines: list[str] = []
        for i, d in enumerate(docs):
            # Dict-aware: state["literature"] entries are dicts, while the
            # Axon path passes RetrievedDoc objects. Without this the guard
            # judged blind on dicts (empty title/excerpt) and fail-opened.
            if isinstance(d, dict):
                meta = d.get("metadata") or {}
                content = d.get("content") or ""
            else:
                meta = getattr(d, "metadata", {}) or {}
                content = getattr(d, "content", "") or ""
            title = (
                (meta.get("title") or meta.get("source") or "")
                if isinstance(meta, dict) else ""
            )
            excerpt = str(content)[:200].replace("\n", " ")
            listing_lines.append(f"[{i}] {title} :: {excerpt}")
        prompt = (
            "You are a relevance filter for a research assistant. The user's "
            "topic is:\n\n"
            f"{topic[:600]}\n\n"
            "Below are candidate sources the retriever returned. Some may be "
            "confidently retrieved yet completely off-topic (e.g. physics "
            "papers returned for a corporate-finance question). Return ONLY "
            "the indices that are genuinely relevant to the topic.\n\n"
            "# Candidates\n"
            + "\n".join(listing_lines)
            + "\n\nRespond with a single JSON object, no prose:\n"
            '{"relevant_indices": [<int>, ...]}\n'
            "If NONE are relevant, return an empty list."
        )
        try:
            raw = await self._chat(prompt, node="relevance_guard")
            parsed = _parse_json_lenient(raw, node="relevance_guard")
            if not parsed or "relevant_indices" not in parsed:
                return docs
            idxs = parsed.get("relevant_indices") or []
            keep_idx = {int(i) for i in idxs if isinstance(i, (int, float))}
        except Exception as e:
            self._log.info("[auto_collect] relevance guard failed: %s — keeping all", e)
            return docs
        kept = [d for i, d in enumerate(docs) if i in keep_idx]
        dropped = len(docs) - len(kept)
        if dropped:
            self._log.info(
                "[auto_collect] relevance guard kept %d/%d docs (%d off-topic dropped)",
                len(kept), len(docs), dropped,
            )
        return kept

    async def _screen_literature(
        self, topic: str, docs: list, *, work_scope: str = WORK_SCOPE_PAPERS,
    ) -> list:
        """Grade every retrieved source 0-3 in one batched call and keep the
        citable ones (rubric in ``agents/literature_screen.md``).

        The embedding floor that runs first scores word overlap, so a table of
        contents or a paper on another system that shares the search terms
        passes it; this asks whether the paper could cite the source for a
        claim. Scholarly records need a 2. Web pages are dropped only at 0:
        they are kept for the text the writer quotes, and a page of general
        background is exactly what a 1 describes. At least
        ``knowledge.relevance_min_keep`` sources survive, best grades first,
        so a thin retrieval is not emptied (the evidence gate can broaden).
        Fail-open: when the screen is off, the call fails or the reply cannot
        be read, every source is kept, and so is any source left ungraded.
        Graded sources carry ``screen_grade``.
        """
        kn = self.config.knowledge
        # Papers the user supplied are theirs to judge: never shown, never dropped.
        own = {
            i for i, d in enumerate(docs)
            if (d.metadata or {}).get("source") in ("local_paper", "user_supplied")
        }
        if not kn.literature_screen or len(own) == len(docs):
            return docs
        lines: list[str] = []
        for i, d in enumerate(docs):
            if i in own:
                continue
            md = d.metadata or {}
            kind = "web page" if md.get("source") == "web_search" else "paper"
            if md.get("foundational"):
                kind = "foundational " + ("book" if md.get("work_type") in ("book", "book-chapter") else "paper")
            title = " ".join(str(md.get("title") or md.get("url") or "(untitled)").split())
            facts = ", ".join(
                str(v) for v in (md.get("venue"), md.get("year"), md.get("work_type"), md.get("foundational")) if v
            )
            excerpt = " ".join(str(d.content or "").split())[:300]
            lines.append(
                f"[{i}] ({kind}) {title}" + (f" — {facts}" if facts else "") + f" :: {excerpt}"
            )
        if work_scope == WORK_SCOPE_PAPERS_AND_BOOKS:
            guidance = (
                "This quest has no experiment, so books, book chapters and "
                "humanities or social-science scholarship count as fully as "
                "journal articles: judge them by subject, not by format."
            )
        else:
            guidance = (
                "This quest runs an experiment. Work on the same system, method "
                "or measured quantity is the most useful; a paper from another "
                "field that only shares vocabulary is a 0 or a 1."
            )
        prompt = self._prompts["literature_screen"].substitute(
            topic=topic[:1200], kind_guidance=guidance, candidates="\n".join(lines),
        )
        try:
            raw = await self._chat(prompt, node="literature_screen")
            parsed = _parse_json_lenient(raw, node="literature_screen")
        except Exception as e:  # noqa: BLE001 — the screen must never cost the corpus
            self._log.info("[literature] screen failed (%r); keeping all %d sources", e, len(docs))
            return docs
        grades = _screen_grades(parsed, len(docs))
        if grades is None:
            self._log.info("[literature] screen reply unreadable; keeping all %d sources", len(docs))
            return docs
        # A grade the model gave a user-supplied paper anyway does not count.
        grades = {i: g for i, g in grades.items() if i not in own}
        keep = [
            i for i, d in enumerate(docs)
            if i not in grades
            or grades[i] >= (1 if (d.metadata or {}).get("source") == "web_search" else 2)
        ]
        minimum = min(kn.relevance_min_keep, len(docs))
        if len(keep) < minimum:
            # Stable sort: equal grades stay in retrieval order.
            rest = sorted((i for i in range(len(docs)) if i not in keep),
                          key=lambda i: -grades.get(i, 0))
            keep = sorted(keep + rest[:minimum - len(keep)])
        out = []
        for i in keep:
            md = dict(docs[i].metadata or {})
            if i in grades:
                md["screen_grade"] = grades[i]
            out.append(RetrievedDoc(content=docs[i].content, metadata=md))
        values = list(grades.values())
        self._log.info(
            "[literature] screen kept %d/%d sources (grade counts: %s)",
            len(out), len(docs), {g: values.count(g) for g in sorted(set(values))},
        )
        return out

    async def _foundational_works(
        self, topic: str, docs: list, *, work_scope: str = WORK_SCOPE_PAPERS,
    ) -> list:
        """Candidates a keyword search does not reach: the original papers and
        standard textbooks of what a topic rests on. One call asks the model
        for up to eight, each is looked up by title in OpenAlex and dropped when
        not found, and the works several retrieved papers cite are added. They
        are labelled foundational and go through the literature screen. Off
        with ``knowledge.foundational_works: false``; any failure adds nothing.

        The run log names every work the model suggested and what became of
        each, because a work the model never named cannot be told from one
        OpenAlex does not hold without it."""
        kn = self.config.knowledge
        if not (kn.enabled and kn.foundational_works):
            return []
        suggestions = await self._suggest_foundational_works(topic, work_scope=work_scope)
        self._log.info(
            "[literature] foundational works suggested (%d): %s",
            len(suggestions), "; ".join(_foundational_work_text(w) for w in suggestions) or "(none)",
        )
        try:
            found = await self.knowledge.find_foundational_works(suggestions, docs)
        except Exception as e:  # noqa: BLE001 — never costs the retrieval
            self._log.info("[literature] foundational works lookup failed: %r", e)
            return []
        have = {_normalize_title((d.metadata or {}).get("title") or "") for d in docs} - {""}
        new = [d for d in found if _normalize_title(d.metadata.get("title") or "") not in have]
        self._log.info(
            "[literature] foundational works: %d suggested, %d new candidate(s)%s",
            len(suggestions), len(new),
            (": " + "; ".join(f"{d.metadata.get('title')} ({d.metadata.get('year')}, "
                              f"{d.metadata.get('foundational')})" for d in new))[:600] if new else "",
        )
        if suggestions:
            added, kept_already, dropped = _foundational_outcomes(suggestions, new, found, docs)
            self._log.info(
                "[literature] foundational works: found in OpenAlex and added (%d): %s | "
                "already among the search results (%d): %s | dropped, not found in OpenAlex "
                "(%d): %s",
                len(added), "; ".join(added) or "-",
                len(kept_already), "; ".join(kept_already) or "-",
                len(dropped), "; ".join(dropped) or "-",
            )
        return new

    async def _suggest_foundational_works(
        self, topic: str, *, work_scope: str = WORK_SCOPE_PAPERS,
    ) -> list[dict]:
        """Up to eight foundational works the model names for ``topic``, each
        ``{title, authors, year}``; any beyond the eighth are ignored. Empty
        when the call fails."""
        if work_scope == WORK_SCOPE_PAPERS_AND_BOOKS:
            what = "the classic books and articles scholars of this subject cite as its foundations"
        else:
            what = ("the original papers that introduced the methods, models or effects it "
                    "relies on, and the standard textbooks on them")
        prompt = (
            f"List up to EIGHT foundational works for this research topic: {what}. "
            "Name only works you are sure exist, with their exact titles: each one is "
            "looked up by title, and a work that is not found is dropped.\n\n"
            f"RESEARCH TOPIC:\n{topic[:1200]}\n\n"
            'Reply as JSON only: {"works": [{"title": "<exact title>", '
            '"authors": "<first author\'s surname>", "year": <year>}]}'
        )
        try:
            raw = await self._chat(prompt, node="literature_foundational")
            parsed = _parse_json_lenient(raw, node="literature_foundational")
        except Exception as e:  # noqa: BLE001 — best-effort
            self._log.info("[literature] foundational works suggestion failed: %r", e)
            return []
        works = parsed.get("works") if isinstance(parsed, dict) else None
        return [
            w for w in (works if isinstance(works, list) else [])
            if isinstance(w, dict) and str(w.get("title") or "").strip()
        ][:FOUNDATIONAL_MAX_SUGGESTED]

    async def _derive_literature_queries(
        self, topic: str, idea_title: str = "", hypothesis: str = "",
        *, work_scope: str = WORK_SCOPE_PAPERS,
    ) -> list[str]:
        """Turn a topic statement into up to three keyword search queries,
        one per facet of the topic.

        A search engine matches keywords, and one query reaches only the work
        written in its own words. Three facets -- the core subject, a specific
        angle and the wider frame -- reach work a single query misses; the
        results are merged and screened afterwards. The vocabulary follows the
        kind of quest. One with an experiment searches under the names of
        methods, systems and measured quantities. One without searches under
        the names scholars of that subject write about -- works, people,
        periods, places, movements -- because wording decides what a database
        returns: on one humanities topic the same database gave 0 of 10
        on-topic hits for the STEM-style query and 8 of 10 for the other.

        Returns [] when retrieval is off (nothing would read the queries), when
        the model fails, or when no reply is a keyword query (a long sentence
        is the topic echoed back). The caller then keeps its own query, so
        this changes what gets sent but never stops the search.
        """
        if not self.config.knowledge.enabled:
            return []
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
        prompt = (
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
        try:
            raw = await self._chat(prompt, node="literature_query")
            parsed = _parse_json_lenient(raw, node="literature_query")
        except Exception as e:  # noqa: BLE001 — best-effort; caller degrades
            self._log.info("[literature] query derivation failed: %r", e)
            return []
        if not isinstance(parsed, dict):
            return []
        replies = parsed.get("queries")
        if not isinstance(replies, list):
            # A reply in the one-query shape still yields one facet.
            replies = [parsed.get("query")]
        out: list[str] = []
        for reply in replies:
            q = " ".join(str(reply or "").split())[:300]
            if q and len(q.split()) <= 20 and q.lower() not in {o.lower() for o in out}:
                out.append(q)
        return out[:3]

    async def _propose_literature_queries(
        self, topic: str, tried: list[str], missed: list,
        *, work_scope: str = WORK_SCOPE_PAPERS,
    ) -> str:
        """Ask for ONE better search query after a retrieval missed entirely.

        Shows the model what was already tried and a sample of what came back,
        so it can tell "wrong vocabulary" (the usual cause — a field's papers
        use different terms than the topic statement) from "too narrow".
        Returns "" on any failure; the caller then keeps the original results
        rather than looping, so a flaky model degrades to today's behaviour.
        """
        titles = []
        for d in missed[:6]:
            meta = (d.get("metadata") if isinstance(d, dict) else getattr(d, "metadata", {})) or {}
            t = str(meta.get("title") or "").strip()
            if t:
                titles.append(f"- {t[:120]}")
        prompt = (
            "A literature search returned results that are all off-topic.\n\n"
            f"RESEARCH TOPIC:\n{topic[:600]}\n\n"
            "QUERIES ALREADY TRIED (do not repeat these):\n"
            + "\n".join(f"- {q[:160]}" for q in tried)
            + "\n\nWHAT CAME BACK (all judged off-topic):\n"
            + ("\n".join(titles) if titles else "- (no titles)")
            + "\n\nThe likely cause is vocabulary: this field's papers may use "
            "different terminology than the topic statement does. Propose ONE "
            "alternative search query that uses the terms researchers in this "
            "field would actually publish under. "
            + ("Prefer the names scholars of this subject write under -- works, "
               "people, periods, places, movements, genres, concepts -- over "
               "method words. "
               if work_scope == WORK_SCOPE_PAPERS_AND_BOOKS else
               "Prefer domain-standard terms. ")
            + "Spell out acronyms. Keep it under 20 words.\n\n"
            'Reply as JSON only: {"query": "<your query>"}'
        )
        try:
            raw = await self._chat(prompt, node="literature_requery")
            parsed = _parse_json_lenient(raw, node="literature_requery")
            q = str((parsed or {}).get("query") or "").strip()
        except Exception as e:  # noqa: BLE001 — best-effort; caller degrades
            self._log.info("[literature] requery proposal failed: %r", e)
            return ""
        if not q or q.lower() in {t.lower() for t in tried}:
            return ""
        return q[:300]

    def _filter_docs_by_relevance(
        self, topic: str, docs: list, *, stats: dict | None = None,
    ) -> list:
        """Deterministic relevance floor for the literature node: score each
        retrieved doc by embedding cosine against the TOPIC and drop the
        off-topic tail. Unlike the LLM ``_filter_relevant_docs`` guard — which
        runs only on the auto_collect path — this runs on EVERY quest's fresh
        literature, so survey / simulation quests (which skip auto_collect)
        don't carry off-topic sources (e.g. change-point-math papers retrieved
        for a sculpture-history topic) into analyze/write.

        No LLM call — reuses the all-MiniLM cosine helper from ``passages``.
        Fail-open and never-starve: returns docs unchanged when embeddings are
        unavailable (FI_OFFLINE / no model / error), when ``relevance_min_score``
        is 0, or when docs is empty; and always retains at least
        ``relevance_min_keep`` top-scoring docs so a wholly-borderline corpus is
        not emptied (the evidence_gate can then broaden). Scores against the raw
        TOPIC (not the retrieval query, whose folded-in hypothesis keywords are
        exactly what drags off-topic hits in)."""
        min_score = self.config.knowledge.relevance_min_score
        min_keep = self.config.knowledge.relevance_min_keep
        if not docs or min_score <= 0.0:
            return docs
        from core.passages import _embed_scores  # shared MiniLM; FI_OFFLINE-safe

        blobs: list[str] = []
        for d in docs:
            # Dict- and RetrievedDoc-aware, mirroring _filter_relevant_docs.
            if isinstance(d, dict):
                meta = d.get("metadata") or {}
                content = d.get("content") or ""
            else:
                meta = getattr(d, "metadata", {}) or {}
                content = getattr(d, "content", "") or ""
            title = (
                (meta.get("title") or meta.get("source") or "")
                if isinstance(meta, dict) else ""
            )
            excerpt = str(content)[:300].replace("\n", " ")
            blobs.append(f"{title} {excerpt}".strip())
        scores = _embed_scores(blobs, str(topic or "")[:600])
        if scores is None:  # embeddings unavailable — never filter blind
            if stats is not None:
                stats["scored"] = False
            return docs
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        keep_idx = {i for i in range(len(docs)) if scores[i] >= min_score}
        if stats is not None:
            # ``above_floor`` counts docs that cleared the threshold on their
            # own merits, BEFORE never-starve retention pads the set back up.
            # The requery loop needs that distinction: "3 docs kept" can mean
            # "3 good hits" or "everything missed and min_keep held 3 back",
            # and only the second is worth re-searching for.
            stats["scored"] = True
            stats["above_floor"] = len(keep_idx)
            stats["best"] = max(scores) if scores else 0.0
        keep_idx.update(order[:max(0, min_keep)])  # never-starve retention
        kept = [d for i, d in enumerate(docs) if i in keep_idx]
        dropped = len(docs) - len(kept)
        if dropped:
            # ``lowest_kept`` can be below the threshold when the never-starve
            # retention (min_keep) held back a doc — label the values + surface
            # min_keep so that isn't misread as the floor being violated.
            lowest_kept = min((scores[i] for i in keep_idx), default=0.0)
            self._log.info(
                "[literature] relevance floor kept %d/%d docs (%d dropped; "
                "threshold=%.2f, min_keep=%d, lowest kept cosine=%.2f)",
                len(kept), len(docs), dropped, min_score, min_keep, lowest_kept,
            )
        return kept

    async def _run_dataset_adapters(
        self, query: str, auto_dir: Path,
    ) -> int:
        """Iterate the user's enabled
        ``engine.dataset_adapters``, ask each for up to
        ``engine.dataset_adapter_top_k`` rows, and render each row as
        a Markdown file under ``<auto_dir>/<adapter_name>/``.

        Returns the count of files actually written so the calling
        node can fold it into ``auto_collected_count``. Failures (an
        adapter raises, returns nothing, or writes can't land on
        disk) are logged and skipped — the engine never aborts the
        no-simulation flow because of a flaky dataset API.
        """
        adapter_names = self.config.engine.dataset_adapters or []
        if not adapter_names:
            return 0
        from .datasets import ADAPTER_REGISTRY

        top_k = self.config.engine.dataset_adapter_top_k
        total_written = 0
        for name in adapter_names:
            adapter_cls = ADAPTER_REGISTRY.get(name)
            if adapter_cls is None:
                self._log.warning(
                    "[auto_collect] unknown dataset adapter %r "
                    "(known: %s) — skipping",
                    name, sorted(ADAPTER_REGISTRY),
                )
                continue
            self._log.info(
                "[auto_collect] running dataset adapter %s (top_k=%d)",
                name, top_k,
            )
            try:
                rows = await adapter_cls().search(query, top_k=top_k)
            except Exception as e:
                self._log.warning(
                    "[auto_collect] dataset adapter %s raised: %s — "
                    "skipping its results",
                    name, e,
                )
                continue
            if not rows:
                self._log.info(
                    "[auto_collect] dataset adapter %s returned 0 rows",
                    name,
                )
                continue
            sub_dir = auto_dir / name
            for idx, row in enumerate(rows, start=1):
                slug_basis = (
                    row.metadata.get("indicator_id")
                    or row.metadata.get("title")
                    or f"row{idx}"
                )
                slug = _slugify(str(slug_basis))[:40] or f"row{idx}"
                fname = f"{idx:03d}_{slug}.md"
                target = sub_dir / fname
                body = _render_auto_collected_md(
                    idx, {**row.metadata, "adapter": name}, row.content,
                )
                try:
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    target.write_text(body, encoding="utf-8")
                    total_written += 1
                except OSError as e:
                    self._log.warning(
                        "[auto_collect] adapter %s: failed to write %s: %s",
                        name, target, e,
                    )
            # If every write for this adapter failed, clean the empty
            # subdirectory like we do for Axon results.
            if sub_dir.is_dir() and not any(sub_dir.iterdir()):
                try:
                    sub_dir.rmdir()
                except OSError:
                    pass
        return total_written

    async def _node_wait_for_data(self, state: QuestState) -> QuestState:
        """no-simulation pause point. Creates ``<quest_root>/data/``,
        writes a README explaining what to drop into it, then either:

        * If the dir is empty (apart from the README we just wrote) —
          fire ``interrupt(...)`` so ``Engine.run`` can exit cleanly
          with rc=0 and tell the user to drop files and re-run.

        * If files are already present — return state with
          ``data_files`` populated, letting the graph proceed to
          ``data_load`` → ``analyze``.

        On resume after the user dropped files, LangGraph re-enters
        this node from the checkpoint. The dir now has files, so we
        proceed without pausing.
        """
        data_dir = self.quest_root / "data"
        data_dir.mkdir(exist_ok=True)
        readme = data_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                _render_data_readme(state, self.quest_id), encoding="utf-8",
            )
        user_files = _list_user_data_files(data_dir)
        if not user_files:
            # Pause via LangGraph's interrupt mechanism. Engine.run
            # catches this and exits rc=0 cleanly. On resume the
            # interrupt re-fires; if the user has dropped files by
            # then, the resume payload carries them. (We don't trust
            # the payload — we re-walk the dir on every resume.)
            self._pause_for_human(
                kind="data",
                interaction="supply",
                headline="add a dataset to analyse",
                steps=[
                    "This observational quest has no data yet and runs no "
                    "simulation.",
                    "Drop your dataset (CSV / JSON / TSV / Parquet) into "
                    "`data/` — see `data/README.md` for the specifics it expects.",
                ],
                payload={
                    "data_required": True,
                    "quest_id": self.quest_id,
                    "data_dir": str(data_dir),
                },
                upload_targets=["root_data"],
            )
            # Unreachable in practice: interrupt() raises GraphInterrupt
            # which Engine.run catches above; on resume, the node is
            # re-invoked from the checkpoint and the early-return below
            # fires because user_files is now non-empty.
            return {}
        self._log.info(
            "[wait_for_data] found %d user file(s) under %s — proceeding "
            "to data_load", len(user_files), data_dir,
        )
        return {"data_files": [str(p) for p in user_files]}

    async def _node_data_load(self, state: QuestState) -> QuestState:
        """no-simulation mode: synthesize a ``result_json`` from the
        files the user dropped into ``<quest_root>/data/``. Reuses
        the ``core/summarizer.py`` patterns (file walk + classify +
        content-budget-aware prompt assembly) so we don't duplicate
        the mixed-format walker.

        The LLM call produces a JSON object with the same shape an
        analyze-node prompt expects: a top-level dict of findings
        + supporting evidence cited back to the user's source files.
        """
        from .summarizer import (
            _walk_folder, _render_file_manifest, _render_content_blocks,
        )

        data_dir = self.quest_root / "data"
        # Re-walk the dir on every invocation rather than trusting
        # ``state["data_files"]`` — the user may have edited / added
        # files between pause and resume.
        #
        # Filter out the FI-authored README.md at the top of the data
        # dir before rendering the prompt. ``_walk_folder`` doesn't
        # know it's there as instructions, not data; without this
        # filter the README's "drop your data here" text + the
        # hypothesis + the resume command would land in the prompt
        # and the LLM might cite it as a "primary source" in
        # key_findings. ``_list_user_data_files`` already excludes
        # the README (used by _node_wait_for_data's pause check);
        # data_load now matches that contract by re-numbering idents
        # after the filter so the manifest IDs stay sequential.
        all_entries = _walk_folder(data_dir)
        entries = [
            e for e in all_entries
            if not (e.rel_path == "README.md" and (data_dir / "README.md").is_file())
        ]
        # Re-number ident so the prompt sees [1], [2], … in order;
        # FileEntry is a dataclass so we mutate the ident field.
        for new_id, entry in enumerate(entries, start=1):
            entry.ident = new_id
        if not entries:
            self._log.warning(
                "[data_load] %s has no user data (only README.md) — "
                "analyze will run with an empty result_json", data_dir,
            )
            return {"result_json": {}, "data_files": []}

        manifest = _render_file_manifest(entries)
        content_blocks = _render_content_blocks(entries)
        prompt = self._prompts["data_load"].substitute(
            topic=state["topic"],
            design_block=json.dumps(state.get("design") or {}, indent=2),
            file_manifest=manifest,
            content_blocks=content_blocks,
        )
        text = await self._chat(prompt, node="data_load")
        result_json = _parse_json_lenient(text, node="data_load") or {}
        self._log.info(
            "[data_load] synthesized result_json with %d keys from %d files",
            len(result_json), len(entries),
        )
        return {
            "result_json": result_json,
            "data_files": [str(e.path) for e in entries],
            # Mark the no-sim path explicitly so analyze can flavor its
            # interpretation ("the user collected this; treat citations
            # as primary sources, not simulation outputs").
            "exec_result": {
                "returncode": 0,
                "source": "no_simulation_user_data",
                "n_files": len(entries),
            },
            # No figures from a simulation — leave the list empty.
            "figures": [],
        }

    async def _node_implement_outline(self, state: QuestState) -> QuestState:
        """First half of the two-stage implement flow: produce a
        structural outline (scaffold + function signatures + constants
        + RESULT_JSON template + deps) BEFORE the body stage spends its
        thinking budget on algorithm details.

        Failure-isolated like ``design_self_critique``: a transient
        provider error or parse failure on the outline call doesn't
        block the quest. We return an empty ``implement_outline`` dict
        and the body stage falls back to the legacy single-shot
        ``agents/implement.md`` prompt.

        Skipped on resume from a pre-Phase-2 checkpoint whose
        ``implement_outline`` slot is already populated — the body
        stage uses what's there.
        """
        if state.get("implement_outline"):
            # Already populated (resume after a failed body call, or a
            # checkpoint that was advanced by a future-version engine).
            # Don't re-bill an outline call.
            self._log.info("[implement_outline] cached outline present; skipping")
            return {}
        self._log.info("[implement_outline] drafting scaffold + signatures")
        try:
            prompt = self._prompts["implement_outline"].substitute(
                design_block=json.dumps(state.get("design") or {}, indent=2),
                clarify_block=_format_clarify(state),
                timeout_s=str(self.config.execution.timeout_s),
                skills_block=self._skills_block(state) or "(no skills selected for this quest)",
                inputs_block=self._inputs_block(),
                job_block=self._job_block(),
            )
        except KeyError:
            # Prompt not loaded (e.g. running a build that doesn't ship
            # the new agents/implement_outline.md). Leave the slot empty
            # so the body stage falls back to the legacy single-shot.
            self._log.warning(
                "[implement_outline] prompt template missing; "
                "body stage will use the legacy one-shot path"
            )
            return {"implement_outline": {}}
        try:
            text = await self._chat(prompt, node="implement_outline")
        except Exception as exc:  # noqa: BLE001 — best-effort, strictly advisory
            self._log.warning(
                "[implement_outline] chat failed (%r); body stage will "
                "use the legacy one-shot path", exc,
            )
            return {"implement_outline": {}}
        # Tag the lenient-parse with the node so its WARNING lines in
        # run.log carry the originator — multiple nodes call this
        # helper and the unprefixed warnings were hard to grep.
        outline = _parse_json_lenient(text, node="implement_outline")
        if not isinstance(outline, dict) or "scaffold" not in outline:
            self._log.warning(
                "[implement_outline] response not parseable or missing "
                "'scaffold' (kept legacy fallback). LLM head: %r",
                (text or "")[:300],
            )
            return {"implement_outline": {}}
        n_funcs = len(outline.get("functions") or [])
        n_const = len(outline.get("constants") or [])
        self._log.info(
            "[implement_outline] scaffold ready — %d functions, %d constants",
            n_funcs, n_const,
        )
        return {"implement_outline": outline}

    async def _node_implement(self, state: QuestState) -> QuestState:
        outline = state.get("implement_outline") or {}
        body_prompt = self._prompts.get("implement_body")
        if outline and outline.get("scaffold") and body_prompt is not None:
            self._log.info(
                "[implement] filling scaffold (%d functions to body)",
                len(outline.get("functions") or []),
            )
            prompt = body_prompt.substitute(
                design_block=json.dumps(state.get("design") or {}, indent=2),
                clarify_block=_format_clarify(state),
                outline_block=json.dumps(outline, indent=2),
                timeout_s=str(self.config.execution.timeout_s),
                skills_block=self._skills_block(state) or "(no skills selected for this quest)",
                inputs_block=self._inputs_block(),
                job_block=self._job_block(),
            )
        else:
            # Legacy single-shot path: no outline available (pre-Phase-2
            # checkpoint resume, the outline call failed, or — defensive
            # case — a future checkpoint advanced past outline but the
            # CURRENT running build doesn't ship ``agents/implement_body.md``.
            # Use the original prompt verbatim so existing behaviour is
            # preserved on resume across mixed-version builds.
            if outline and outline.get("scaffold") and body_prompt is None:
                self._log.warning(
                    "[implement] outline cached but implement_body prompt "
                    "not loaded; falling back to legacy one-shot. This "
                    "happens when an older build resumes a checkpoint "
                    "produced by a newer build — rebuild the agents/ "
                    "directory to enable the two-stage path."
                )
            else:
                self._log.info("[implement] generating experiment code (legacy one-shot)")
            prompt = self._prompts["implement"].substitute(
                design_block=json.dumps(state.get("design") or {}, indent=2),
                timeout_s=str(self.config.execution.timeout_s),
                skills_block=self._skills_block(state) or "(no skills selected for this quest)",
                inputs_block=self._inputs_block(),
                job_block=self._job_block(),
            )
        # A review that named something this run computed sent the experiment
        # back here (``re_execute``). Both prompts above carry the design and
        # the outline but no review, so without this the model would regenerate
        # the same code — and with it the same wrong value.
        rerun_for = _review_sends_the_experiment_back(state.get("review") or {}, state)
        if rerun_for:
            self._log.info(
                "[implement] the review sent the experiment back over %r", rerun_for,
            )
            prompt += _rerun_directive(state.get("review") or {}, rerun_for)
        text = await self._chat(prompt, node="implement")
        code, deps = _parse_implement_response(text)
        extracted = bool(code)
        if not code:
            # Empty-code path: log the LLM head so the user can see WHAT
            # came back rather than silently shipping a stub experiment
            # that crashes downstream with no signal.
            self._log.warning(
                "[implement] no code extracted; LLM head: %r",
                text[:400] if text else "<empty>",
            )
            code = 'print("RESULT_JSON: {}")'
        # Defensive: if the design listed deps and the impl skipped them, union.
        # design_deps comes from a JSON-leniently-parsed LLM response — it
        # MAY be a list[str], a comma-separated string, or something weirder.
        # Coerce to a list[str] before set-union; otherwise unpacking a bare
        # string into the set produces per-character entries ("numpy" -> {"n","u",...}).
        design_deps = _coerce_dep_list(
            (state.get("design") or {}).get("dependencies")
        )
        deps = sorted({*deps, *design_deps})

        code_path = self.quest_root / "code" / "experiment.py"
        code_path.write_text(code, encoding="utf-8")
        self._log.info("[implement] wrote %s (%d bytes)", code_path, len(code))
        if extracted:  # nothing to seed in the stub written above
            code, deps = await self._repair_ignored_replicate_seed(state, code_path, code, deps)
        return {"code": code, "deps": deps}

    async def _repair_ignored_replicate_seed(
        self, state: QuestState, code_path: Path, code: str, deps: list[str],
    ) -> tuple[str, list[str]]:
        """Ask for ONE repair of a script whose randomness the seed cannot reach.

        Two shapes, one call. A script that never reads ``FI_REPLICATE_SEED``
        makes every replicate run the same run: ``execute`` finds the results
        identical, publishes no aggregate, and the paper can report one
        measurement with no interval (four of six codex quests did exactly
        that). A script that reads it and still builds a generator from OS
        entropy has the opposite failure and hides it better: the replicates
        differ, so an aggregate is published, but the seed reached none of the
        randomness and no run can be reproduced. The instruction not to write a
        seed of its own is in the implement prompts; a model that ignores it
        either way is asked again, once, here.

        The request reuses the ``execute_reflect`` template (the directive
        stands where a traceback would) and is applied the way its patches
        are, but it is not a repair iteration: the node's
        ``exec_reflect_*`` budget is untouched. Never a call when replication
        is off, for a background job (no replicate runs are made), or when the
        script already names the variable.

        A repair is kept only if it parses as Python and the script would no
        longer be sent back -- it reads the seed AND builds no generator
        without one. A model that rewrote the script and still missed the
        target, or returned half of it, would leave a script worse than the one
        it had, so the original stays and the fallbacks in ``execute`` (the
        single-measurement one, or the warning that the runs cannot be
        reproduced) handle it. Failure never blocks the quest: one warning,
        then on.
        """
        reads_seed = _script_reads_replicate_seed(code_path)
        unseeded = _unseeded_rng_calls(code_path)
        if (
            self.config.execution.background_jobs
            or max(1, int(self.config.engine.execute_replicates)) <= 1
            or state.get("no_simulation_resolved")
            or state.get("survey_mode_resolved")
            or (reads_seed and not unseeded)
        ):
            return code, deps
        if reads_seed:
            self._log.info(
                "[implement] %s reads FI_REPLICATE_SEED but builds a generator with no "
                "seed (%s), so none of its %d replicate runs could be reproduced; "
                "asking for one repair",
                code_path.name,
                "; ".join(f"line {line}: {expr}" for line, expr in unseeded[:6]),
                self.config.engine.execute_replicates,
            )
        else:
            self._log.info(
                "[implement] %s never reads FI_REPLICATE_SEED, so its %d replicate runs "
                "would repeat one run; asking for one repair",
                code_path.name, self.config.engine.execute_replicates,
            )
        prompt = self._prompts["execute_reflect"].substitute(
            # Whole: a repair that returns the script must not lose the tail of it.
            previous_code=code,
            returncode="(not run yet)",
            stdout_tail=_unseeded_rng_directive(unseeded) if reads_seed else _SEED_REPAIR_DIRECTIVE,
            stderr_tail="",
            duration_s="0.00",
            figures_count="0",
            result_json_present="no (not run yet)",
            reflect_history_block=_format_reflect_history([]),
            design_block=json.dumps(state.get("design") or {}, indent=2),
            clarify_block=_format_clarify(state),
        )
        why = ""
        try:
            text = await self._chat(prompt, node="implement_seed")
        except Exception as exc:  # noqa: BLE001 -- a repair is best-effort
            text, why = "", f"the repair call failed ({exc!r})"
        parsed: dict[str, Any] = {}
        if _strip_outer_fence(text).lstrip().startswith("{"):
            parsed = _parse_json_lenient(text, node="implement_seed") or {}
        new_code, new_deps = parsed.get("code"), _coerce_dep_list(parsed.get("deps"))
        if not (isinstance(new_code, str) and new_code.strip()):
            new_code, new_deps = _parse_implement_response(text)  # fence drift
        if not why:
            try:
                ast.parse(new_code)
                usable = bool(new_code.strip())
            except (SyntaxError, ValueError):
                usable = False
            if not usable:
                why = "the model returned no usable script"
            elif "FI_REPLICATE_SEED" not in new_code:  # the test _script_reads_replicate_seed applies
                why = "the repaired script still does not read it"
            elif unseeded_rng_calls(new_code):
                why = "the repaired script still builds a generator without a seed"
        if why:
            if reads_seed:
                self._log.warning(
                    "[implement] %s: keeping the script as written. Its replicates will "
                    "differ, so the quest keeps its aggregate, but the randomness is "
                    "drawn from OS entropy and a rerun will not reproduce these numbers",
                    why,
                )
            else:
                self._log.warning(
                    "[implement] %s: keeping the script as written. Its replicates will "
                    "repeat one run, so the quest will be reported as a single "
                    "measurement with no confidence interval", why,
                )
            return code, deps
        code_path.write_text(new_code, encoding="utf-8")
        self._log.info(
            "[implement] the script now seeds every generator it builds from "
            "FI_REPLICATE_SEED (%s); rewrote %s (%d bytes)",
            str(parsed.get("patch_summary") or "no summary")[:120], code_path, len(new_code),
        )
        return new_code, sorted({*deps, *new_deps})

    async def _node_execute(self, state: QuestState) -> QuestState:
        # Docker sandbox: the selected, approved external skills are mounted
        # read-only in every container this node starts (a thread: resolving
        # the skills can run their self-tests).
        if self.config.execution.sandbox == "docker":
            await asyncio.to_thread(self._mount_selected_skills, state)
        deps = state.get("deps") or []
        if deps:
            self._log.info("[execute] pip install %s", deps)
            install = await self.executor.install(deps, quest_root=self.quest_root)
            if install.returncode != 0:
                self._log.warning(
                    "[execute] pip install rc=%d: %s",
                    install.returncode, pip_failure_summary(install.stderr),
                )

        py = self.executor.python_path(self.quest_root)
        code_path = self.quest_root / "code" / "experiment.py"

        # Clear figures from a PRIOR experiment version before this run. On a
        # re_experiment / broaden / repair loop the implement node rewrites
        # experiment.py, which may emit a DIFFERENT set of figure filenames —
        # the old ones aren't overwritten, so without this they linger, get
        # counted in figures=N, and pollute the paper/slides/poster with a mix
        # of current and stale (possibly degenerate) plots.
        self._clear_stale_figures()

        # House plot style: drop a guarded `sitecustomize.py` bootstrap and
        # put its dir on the experiment subprocess's PYTHONPATH. Python
        # auto-imports `sitecustomize` at startup — before the script imports
        # matplotlib — so every figure gets the FrontierInsight look (brand
        # palette, despined axes, branded teal heatmap, paper-matched
        # backdrop) with zero edits to the LLM-authored code. Failure-isolated:
        # if anything goes wrong we run with the plain inherited env.
        exec_env: dict[str, str] | None = None
        records_dir = self.fi_dir / "figure_records"
        try:
            from .plot_style import RECORDS_DIRNAME, write_boot

            records_dir = self.fi_dir / RECORDS_DIRNAME
            # The same bootstrap records what each figure draws; a previous
            # version's records go with its figures.
            shutil.rmtree(records_dir, ignore_errors=True)
            boot_dir = write_boot(self.fi_dir, self.config.output.paper_style)
            exec_env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p
                ),
                "FI_FIGURE_RECORDS": str(records_dir),
            }
        except Exception as exc:  # styling must never break execution
            self._log.warning("[execute] plot-style bootstrap skipped: %s", exc)

        from core.example_inputs import ENV_VAR as _INPUT_ENV, examples_dir, list_inputs

        if list_inputs(self.quest_root):
            exec_env = {**(exec_env or os.environ), _INPUT_ENV: str(examples_dir(self.quest_root))}

        # Venv warmup: invoke the freshly-installed Python and import the
        # declared deps before the real experiment. This consumes the
        # rc=2 fast-fail race specific to the first invocation of a
        # fresh venv on Windows — the race is in the C-extension DLL
        # load of newly-installed packages (matplotlib, numpy, etc.),
        # not generic Python startup. A pure `import sys` warmup does
        # NOT trigger the same DLL-load path; importing the deps does.
        # Gated so we don't pay the cost where the race cannot occur:
        #   - Docker sandbox: each execute() spawns a fresh container,
        #     so a warmup call is one full container spin-up wasted.
        #   - No deps: nothing was just pip-installed to race against.
        warmup_modules = _deps_to_warmup_modules(deps)
        if self.config.execution.sandbox == "venv" and warmup_modules:
            warmup_code = f"import sys; import {warmup_modules}"
            warmup = await self.executor.execute(
                [str(py), "-c", warmup_code],
                cwd=self.quest_root,
                timeout_s=60,
            )
            if (
                warmup.returncode != 0
                and not warmup.timed_out
                and warmup.duration_s < 0.5
            ):
                self._log.info(
                    "[execute] warmup fast-failed (rc=%d t=%.2fs); retrying once",
                    warmup.returncode, warmup.duration_s,
                )
                warmup = await self.executor.execute(
                    [str(py), "-c", warmup_code],
                    cwd=self.quest_root,
                    timeout_s=60,
                )
            if warmup.returncode != 0:
                self._log.warning(
                    "[execute] venv warmup failed rc=%d stderr_tail=%s; "
                    "proceeding to real script anyway",
                    warmup.returncode, warmup.stderr[-200:],
                )

        # Pilot pass: run the experiment small before running it for real.
        #
        # ``execute_reflect`` already repairs a script that CRASHES. What it
        # cannot catch is a script that runs fine and answers the wrong
        # question -- a sweep over the wrong parameter range, a resolution too
        # coarse to show the effect. Today that costs the full timeout to
        # discover, and a researcher would never work that way: they run a
        # cheap version, look at whether the numbers are the right order of
        # magnitude, and only then commit the compute.
        #
        # ``FI_PILOT=1`` is honoured by the generated script the same way
        # ``FI_REPLICATE_SEED`` is (the implement prompt instructs it), so
        # this needs no separate code path in the experiment itself. The
        # pilot's numbers are DISCARDED -- it is a smoke test of the design,
        # not a measurement.
        # No pilot for a background job: the script would submit it.
        if self.config.engine.pilot_run and not self.config.execution.background_jobs:
            pilot_timeout = max(
                30, int(self.config.execution.timeout_s * self.config.engine.pilot_timeout_frac)
            )
            self._log.info(
                "[execute] pilot pass (FI_PILOT=1, timeout=%ds) before the "
                "full run", pilot_timeout,
            )
            pilot_env = {**(exec_env or os.environ), "FI_PILOT": "1"}
            try:
                pilot = await self.executor.execute(
                    [str(py), str(code_path)],
                    cwd=self.quest_root,
                    timeout_s=pilot_timeout,
                    env=pilot_env,
                )
            except Exception as e:  # noqa: BLE001 — a pilot must never abort the quest
                self._log.info("[execute] pilot could not run (%r); continuing", e)
                pilot = None
            if pilot is not None:
                pilot_rj = _extract_result_json(pilot.stdout)
                if pilot.returncode != 0 or pilot_rj is None:
                    # Not fatal: the full run still happens, and if the fault
                    # is real, execute_reflect repairs it there with the
                    # traceback it needs. Saying so early is the value.
                    self._log.warning(
                        "[execute] pilot did not produce a usable RESULT_JSON "
                        "(rc=%s). Continuing to the full run, where "
                        "execute_reflect can repair it. stderr_tail=%s",
                        pilot.returncode, (pilot.stderr or "")[-300:],
                    )
                else:
                    # ``_assertion_violations`` reads ``result_json`` off the
                    # state, so hand it the PILOT's numbers rather than the
                    # real ones (which don't exist yet) via a shallow copy.
                    violations = _assertion_violations(
                        {**state, "result_json": pilot_rj},
                    )
                    if violations:
                        self._log.warning(
                            "[execute] pilot produced out-of-range values: %s. "
                            "The full run proceeds, but this usually means the "
                            "DESIGN is wrong (parameter range, units) rather "
                            "than the code -- worth reading before the results.",
                            "; ".join(str(v) for v in violations[:3]),
                        )
                    else:
                        self._log.info(
                            "[execute] pilot passed: RESULT_JSON parsed and "
                            "within declared ranges; running full scale",
                        )

        # Run from quest_root so figures/ is the relative target. Wrapped in a
        # heartbeat: the experiment subprocess can run for many minutes (a
        # parameter sweep) while the executor just blocks on communicate(),
        # emitting nothing — and the dashboard infers a quest's status from
        # run.log recency, so a long silent run reads as "pending"/idle. The
        # heartbeat keeps that signal fresh.
        # The first run's seed is FI's to choose too. Left unset, it fell back
        # to whatever default the script had written down (42 in a graded
        # quest) while the replicates were handed 1 and 2 -- three bases a few
        # apart, which for a script deriving per-trial seeds as ``base +
        # counter`` means three runs of almost exactly the same trials.
        stride = max(1, int(self.config.engine.replicate_seed_stride))
        primary_env = _replicate_env(exec_env, 0, stride)
        result: ExecutionResult = await self._await_with_heartbeat(
            self.executor.execute(
                [str(py), str(code_path)],
                cwd=self.quest_root,
                timeout_s=self.config.execution.timeout_s,
                env=primary_env,
            ),
            label="running experiment.py",
        )
        # Observed on Windows-native: the first invocation of a freshly-
        # created venv's python.exe — even after a warmup `python -c
        # "import <deps>"` — sometimes exits with rc != 0 and duration <
        # 0.5 s. The process never reaches user code; a repeat in the
        # same venv works. Suspect a Windows file-cache / DLL-load race.
        # Retry once on the fast-fail signature (rc != 0, not timed-out,
        # duration < 0.5 s). We INTENTIONALLY don't gate on empty
        # stdout/stderr — a real deterministically-broken script will
        # fail the same way on retry, costing ~5 s of wall clock, but
        # the gain is reliably catching the race even when its tail
        # output is non-empty (e.g., a DLL-load message on stderr).
        if (
            result.returncode != 0
            and not result.timed_out
            and result.duration_s < 0.5
        ):
            self._log.warning(
                "[execute] suspicious fast-fail (rc=%d t=%.2fs); retrying once",
                result.returncode, result.duration_s,
            )
            result = await self.executor.execute(
                [str(py), str(code_path)],
                cwd=self.quest_root,
                timeout_s=self.config.execution.timeout_s,
                env=primary_env,
            )
        figures = sorted(
            p.name for p in (self.quest_root / "figures").iterdir()
            if p.is_file() and p.suffix.lower() in _FIGURE_SUFFIXES
        ) if (self.quest_root / "figures").is_dir() else []
        result_json = _extract_result_json(result.stdout)
        self._log.info(
            "[execute] rc=%d duration=%.1fs figures=%d result_json=%s",
            result.returncode, result.duration_s, len(figures), bool(result_json),
        )

        # A background job (HPC, a cluster): the script submitted it, or checked
        # it, and says the job has not finished. That is not a failure and there
        # is nothing to repair. The quest pauses, exits cleanly, and a resume (or
        # `--watch`) runs this node again, which runs the same idempotent script.
        from core import job_watch

        job = job_watch.job_of(result_json)
        if job is not None and job["status"] == job_watch.PENDING and result.returncode == 0:
            self._wait_for_job(job, code_path)
        elif job_watch.clear_pending(self.fi_dir):
            self._log.info("[execute] the background job has finished; using its results")

        # Multi-seed replication. Only fires when (a) the primary run
        # succeeded — we don't want to spend N×cost re-running a script
        # that's already broken — and (b) ``execute_replicates > 1``.
        # The first run we just did counts as seed 0; we now run N-1
        # MORE with seeds 1..N-1 via ``FI_REPLICATE_SEED``. Failures
        # in any individual replicate are recorded but don't abort the
        # rest — better to have N-1 good replicates than zero. The
        # primary ``result_json`` carries seed 0 so downstream
        # single-seed code paths are unchanged.
        # Replicates would submit the job again, once per seed.
        replicates_n = (
            1 if self.config.execution.background_jobs
            else max(1, int(self.config.engine.execute_replicates))
        )
        result_json_replicates: list[dict[str, Any]] = []
        deterministic = False
        # Whether the script can respond to the seed at all, and whether it
        # hands any of its randomness to a generator the seed cannot reach.
        # Both read once from the source that is about to run.
        reads_seed = _script_reads_replicate_seed(code_path)
        unseeded_rng = _unseeded_rng_calls(code_path)
        seed_ignored = False
        replicates_ran = False
        primary_figures: dict[str, tuple[bytes, bytes | None]] = {}
        if result.returncode == 0 and result_json is not None:
            # Tag seed 0 explicitly so the aggregator can attribute it.
            result_json_replicates.append({"_seed": 0, **result_json})

        # Error bars are worth paying for only on a result that is going to
        # survive. ``execute_reflect`` judges plausibility AFTER this node, so
        # replicating first meant every futile repair iteration cost three runs
        # instead of one -- measured on a real quest, where a diverging
        # integrator tripped the gate three times over. With no repair attempt
        # left, though, nothing regenerates the result: it is the one the paper
        # is written from, so it still gets its error bars.
        gate_violations = (
            _assertion_violations({**state, "result_json": result_json})
            if result.returncode == 0 and result_json is not None
            else []
        )
        repair_left = (
            int(state.get("exec_reflect_iter", 0) or 0)
            < self.config.engine.exec_reflect_max_iterations
            and not state.get("exec_give_up_reason")
        )
        skip_replicates = bool(gate_violations) and repair_left
        if skip_replicates and replicates_n > 1:
            self._log.info(
                "[execute] skipping %d replicate(s): the primary result already "
                "violates %d assertion(s), so execute_reflect is about to "
                "regenerate it",
                replicates_n - 1, len(gate_violations),
            )
        elif gate_violations and replicates_n > 1:
            self._log.info(
                "[execute] the result violates %d assertion(s) and no repair "
                "attempt is left, so it is the one written up; replicating it",
                len(gate_violations),
            )
        if (
            replicates_n > 1
            and result.returncode == 0
            and result_json is not None
            and not skip_replicates
        ):
            self._log.info(
                "[execute] replicating: %d additional seeds (1..%d)",
                replicates_n - 1, replicates_n - 1,
            )
            # Every run draws into the same figures/, so the replicates draw
            # over the primary run's figures, a replicate that crashes halfway
            # included. Kept here, a figure the seeds do not redraw as their
            # mean goes back to seed 0's, with seed 0's record of what it draws.
            primary_figures = _read_primary_figures(self.quest_root / "figures", records_dir, figures)
            for seed in range(1, replicates_n):
                replicates_ran = True
                rep_env = _replicate_env(exec_env, seed, stride)
                rep_result = await self.executor.execute(
                    [str(py), str(code_path)],
                    cwd=self.quest_root,
                    timeout_s=self.config.execution.timeout_s,
                    env=rep_env,
                )
                rep_rj = _extract_result_json(rep_result.stdout)
                if rep_result.returncode == 0 and rep_rj is not None:
                    result_json_replicates.append({"_seed": seed, **rep_rj})
                    self._log.info(
                        "[execute] replicate seed=%d rc=0 duration=%.1fs",
                        seed, rep_result.duration_s,
                    )
                    # A deterministic experiment yields the same numbers at
                    # every seed, so further replicates buy nothing but wall
                    # clock -- and the aggregate can only ever report std=0.
                    # Honouring FI_REPLICATE_SEED is not the same as consuming
                    # randomness: a real quest was observed seeding numpy and
                    # then integrating an ODE, so all three runs were byte
                    # identical. One extra run is the cheapest way to find out,
                    # and unlike asking the design to declare itself, it cannot
                    # be wrong about what the script actually did.
                    #
                    # Two seeds agreeing has two causes that are not the same
                    # fact about the experiment, and calling both
                    # "deterministic" is what let a hardcoded seed reach a
                    # paper. A script that never reads FI_REPLICATE_SEED cannot
                    # have responded to it, so its runs are ONE run repeated --
                    # not a computation that happens to be deterministic, and
                    # not samples anything may be averaged over. Its own source
                    # says which case this is.
                    if seed == 1 and rep_rj == result_json:
                        if reads_seed:
                            deterministic = True
                            if replicates_n > 2:
                                self._log.info(
                                    "[execute] seeds 0 and 1 produced identical "
                                    "results -- experiment is deterministic; "
                                    "skipping the remaining %d replicate(s)",
                                    replicates_n - 2,
                                )
                        else:
                            seed_ignored = True
                            self._log.warning(
                                "[execute] seeds 0 and 1 produced identical results "
                                "and %s never reads FI_REPLICATE_SEED, so these are "
                                "one run repeated, not %d samples. No aggregate, "
                                "standard error or confidence interval will be "
                                "reported over them: the quest stands on a single "
                                "measurement. Skipping the remaining %d replicate(s).",
                                code_path.name, replicates_n, max(0, replicates_n - 2),
                            )
                        break
                else:
                    self._log.warning(
                        "[execute] replicate seed=%d FAILED rc=%d duration=%.1fs "
                        "(skipping in aggregate)",
                        seed, rep_result.returncode, rep_result.duration_s,
                    )

        # A line figure drawn by one seed shows that run's noise. With the
        # seeds in hand it is drawn again as their mean, shaded with its 95%
        # confidence interval, and so is one of lines with error bars, whose
        # bars the interval over the seeds replaces. A bar chart, a histogram
        # or any other figure goes back to the primary run's (seed 0), and its
        # record says it shows that one run, so the text can quote seed 0's
        # values for it rather than the means.
        # The seed did not reach the randomness yet the runs differ: they are
        # drawing from OS entropy, either because the script never reads the
        # seed or because it reads it and builds a generator without one
        # anyway. Those replicates ARE independent samples, so the aggregate
        # stands -- but nothing about the run is reproducible, and a rerun will
        # not land on these numbers.
        if (not reads_seed or unseeded_rng) and not seed_ignored and len(result_json_replicates) > 1:
            self._log.warning(
                "[execute] %s draws from OS entropy (%s), so its replicates are "
                "independent samples but not reproducible: a rerun of these seeds "
                "will not land on these numbers",
                code_path.name,
                "; ".join(f"line {line}: {expr}" for line, expr in unseeded_rng[:6])
                if unseeded_rng else "it never reads FI_REPLICATE_SEED",
            )
        replotted: dict[str, int] = {}
        # A figure drawn as "the mean of the seeds" over replicates that are one
        # run repeated would be a mean of one number, captioned as several.
        redraw = len(result_json_replicates) > 1 and not deterministic and not seed_ignored
        if redraw:
            replotted = await self._replot_replicate_figures(
                figures, records_dir, [int(r["_seed"]) for r in result_json_replicates],
                python=py, env=exec_env, assertions=_replicate_assertions(state),
            )
        if replicates_ran:
            restored = _restore_primary_figures(
                self.quest_root / "figures", records_dir, primary_figures, redrawn=replotted,
            )
            if restored:
                self._log.info(
                    "[execute] %d figure(s) not drawn as the mean of the seeds show seed 0: %s",
                    len(restored), ", ".join(restored),
                )
        figure_records = _read_figure_records(records_dir, figures)
        for name, n_seeds in replotted.items():
            if name in figure_records:
                figure_records[name]["replicate_mean"] = {"n": n_seeds}
        if redraw:
            for name in figures:
                if name not in replotted and name in figure_records:
                    figure_records[name]["single_seed"] = 0
        patch: dict[str, Any] = {
            "exec_result": {
                "returncode": result.returncode,
                "duration_s": result.duration_s,
                "timed_out": result.timed_out,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            },
            "figures": figures,
            "figure_records": figure_records,
            "result_json": result_json or {},
            "exec_patch_pending": False,
        }
        # Only populate ``result_json_replicates`` when replication
        # actually ran AND produced more than the primary entry. This
        # keeps the field absent on default single-seed quests so
        # downstream code can use ``state.get("result_json_replicates")``
        # as a "did we run multi-seed" sentinel.
        #
        # Replicates a script produced without ever reading the seed are not
        # published at all. Leaving them on the state is what let "the mean over
        # 3 replicate seeds" and a 95% CI be written about a single run: with
        # the field absent ``_replicate_seed_count`` returns None, and every
        # downstream path -- the analyze aggregate, the figure captions, the
        # writer's "seed 0 of N" language, the number and claim checks -- falls
        # back to its honest single-run behaviour on its own.
        if len(result_json_replicates) > 1 and not seed_ignored:
            patch["result_json_replicates"] = result_json_replicates
            # Lets ``analyze`` say "every seed agreed" instead of reporting an
            # empty aggregate, which reads like the aggregator broke.
            patch["result_json_deterministic"] = deterministic
            # This node runs again on a repair and on a re_experiment, and
            # these are last-value channels: a key a pass leaves out keeps the
            # PREVIOUS pass's value. Clearing the flag stops a quest whose
            # earlier script ignored the seed from carrying "there is one
            # measurement here" alongside this script's full aggregate.
            patch["result_json_replicate_seed_ignored"] = False
        if seed_ignored:
            patch["result_json_replicate_seed_ignored"] = True
            # The same hazard the other way round. An earlier pass may have
            # left a replicate list on the state, and merely withholding the
            # key would KEEP it -- so analyze would aggregate the previous
            # script's seeds against this script's result. An empty list is
            # what every consumer already reads as "no replication", and
            # ``_replicate_seed_count`` returns None for it.
            patch["result_json_replicates"] = []
            patch["result_json_deterministic"] = False
        return patch

    async def _replot_replicate_figures(
        self, figures: list[str], records_dir: Path, seeds: list[int], *,
        python: Any, env: dict[str, str] | None, assertions: list[Any],
    ) -> dict[str, int]:
        """Draw each line figure again as the mean over ``seeds``, shaded with
        its 95% confidence interval, from the lines the plot-style recorder
        kept at every seed. The engine computes the numbers and
        ``code/replot_figures.py`` draws them in the quest's Python, under the
        house style, over the same file. Returns each figure drawn, with its
        number of seeds. Error bars count as lines: each series is drawn at its
        mean without them. A figure that holds anything else, or whose lines
        differ between seeds in label or x, is not drawn, and neither is a
        figure the redraw fails on."""

        def recorded(path: Path) -> Any:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None

        plans = []
        for name in figures:
            runs = [recorded(records_dir / f"{Path(name).stem}.seed{seed}.json") for seed in seeds]
            plan = _replicate_line_figure(name, runs, assertions)
            if plan is not None:
                plans.append(plan)
        if not plans:
            return {}
        code_dir = self.quest_root / "code"
        script = code_dir / "replot_figures.py"
        try:
            code_dir.mkdir(parents=True, exist_ok=True)
            script.write_text(
                (Path(__file__).parent / "replot_figures.py").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (code_dir / "replot_figures.json").write_text(
                json.dumps({"figures": plans}), encoding="utf-8",
            )
        except OSError as e:
            self._log.warning("[execute] could not write the figure redraw: %s", e)
            return {}
        result = await self.executor.execute(
            [str(python), str(script)],
            cwd=self.quest_root,
            timeout_s=min(self.config.execution.timeout_s, 300),
            env=env,
        )
        drawn = {
            line.split(":", 1)[1].strip()
            for line in (result.stdout or "").splitlines() if line.startswith("REPLOTTED:")
        }
        if result.returncode != 0:
            self._log.warning(
                "[execute] redrawing figures as the mean of the seeds rc=%d: %s",
                result.returncode, (result.stderr or "")[-300:],
            )
        replotted = {plan["file"]: len(seeds) for plan in plans if plan["file"] in drawn}
        if replotted:
            self._log.info(
                "[execute] drew %d figure(s) as the mean of %d seeds with 95%% CI bands: %s",
                len(replotted), len(seeds), ", ".join(sorted(replotted)),
            )
        return replotted

    async def _node_execute_reflect(self, state: QuestState) -> QuestState:
        """Post-execute repair node.

        If the experiment ran cleanly (rc==0 AND RESULT_JSON parsed),
        this is a no-op pass-through. Otherwise we read the traceback,
        ask the LLM to patch the code, and write the new code into
        state. The conditional edge after this node routes back to
        `execute` (bounded by `engine.exec_reflect_max_iterations`).

        If the LLM emits a `give_up_reason`, we record it and let the
        graph proceed to analyze with the failure intact — `analyze`
        will surface it in the paper and `review` will mark it down.
        """
        exec_result = state.get("exec_result") or {}
        rc = exec_result.get("returncode", 0)
        # ``_node_execute`` stores ``result_json or {}``, so a script that
        # exits 0 WITHOUT a RESULT_JSON marker lands as an empty dict — which
        # ``is not None`` wrongly counted as "parsed", skipping repair. Treat
        # an empty result_json as no usable result so execute_reflect retries.
        has_result_json = bool(state.get("result_json"))
        degenerate = bool(
            rc == 0
            and has_result_json
            and self.config.engine.degenerate_run_guard
            and _is_degenerate_result(state.get("result_json") or {})
        )
        implausible = _assertion_violations(state)
        # A legend drawn over its data, or a figure title over a panel title, is
        # sent back for a redraw once, and only when nothing else is wrong with
        # the run: a repair of a crash, an all-zero result or a broken bound
        # regenerates the figures anyway, and they are measured again after it.
        overlaps = (
            self._figure_overlaps_to_repair(state)
            if rc == 0 and has_result_json and not degenerate and not implausible
            else []
        )

        # Clean success (and not a degenerate all-zero run) → no-op.
        if rc == 0 and has_result_json and not degenerate and implausible:
            for v in implausible:
                self._log.warning("[execute_reflect] implausible: %s", v.describe())
        if rc == 0 and has_result_json and not degenerate and not implausible and not overlaps:
            left = _figure_overlap_findings({**state, "figure_overlap_repaired": False})
            if left:
                self._log.warning(
                    "[execute_reflect] figure_overlap left as drawn (%s): %s",
                    "the one redraw has been made" if state.get("figure_overlap_repaired")
                    else "no repair attempt to spare", "; ".join(left[:3]),
                )
            self._log.info("[execute_reflect] script succeeded; skipping repair")
            return {}

        iters = state.get("exec_reflect_iter", 0)
        if iters >= self.config.engine.exec_reflect_max_iterations:
            # Proceed regardless; analyze owns the authoritative degenerate
            # flag against the final result (covers max_iterations=0 too).
            self._log.warning(
                "[execute_reflect] %s after %d attempt(s); proceeding to analyze",
                "result still degenerate (all metrics ~0)" if degenerate
                else "iterations exhausted",
                iters,
            )
            return {}

        history = list(state.get("exec_reflect_history") or [])
        history_block = _format_reflect_history(history)

        # For a degenerate (rc=0) run, reframe the repair prompt: the
        # script DIDN'T crash, it ran and produced silent zeros. Tell the
        # LLM to fix the underlying bug (sampling/grid, threshold sentinel,
        # normalization) — reusing the crash-repair template's existing
        # fields rather than adding a new placeholder.
        if degenerate:
            self._log.warning(
                "[execute_reflect] rc=0 but result is degenerate (all metrics ~0) "
                "— attempting repair (iter %d)",
                iters + 1,
            )
            rj_preview = json.dumps(state.get("result_json") or {}, indent=2)[:1500]
            stdout_for_prompt = (
                "DEGENERATE RESULT: the script exited 0 but EVERY numeric metric "
                "is ~0. This is almost certainly a bug — e.g. a grid/sampling "
                "mismatch so the signal of interest is missed, a threshold/edge "
                "finder returning a zero sentinel when it finds no crossing, or "
                "a normalization that flattens the output — NOT a real result. "
                "Diagnose and fix the code so it emits real, non-zero metrics. If "
                "the quantity is genuinely zero/unresolvable, make the script SAY "
                "so explicitly (print a clear diagnostic) instead of emitting "
                "silent zeros.\n\nRESULT_JSON was:\n"
                f"{rj_preview}\n\nOriginal stdout tail:\n"
                + exec_result.get("stdout_tail", "")[:1000]
            )
            returncode_for_prompt = "0 (ran, but result is degenerate)"
            result_json_note = "yes (degenerate — all metrics ~0)"
        elif rc == 0 and has_result_json and implausible:
            # Say WHICH declared bound broke. Without this the model saw rc=0,
            # a parsed RESULT_JSON and "the script just failed", had nothing
            # to repair, and invented a code fault -- then capped the diverging
            # value at the bound on the next pass so the gate would take it.
            # The repair has to be told that an honest null beats a plausible
            # number.
            clamped = any(getattr(v, "kind", "") == "clamped" for v in implausible)
            at_bound = any(getattr(v, "kind", "") == "at_bound" for v in implausible)
            self._log.warning(
                "[execute_reflect] rc=0 but %d value(s) break the design's "
                "declared bounds%s — attempting repair (iter %d)",
                len(implausible),
                " (at least one is capped at its bound)" if clamped else "",
                iters + 1,
            )
            rj_preview = json.dumps(state.get("result_json") or {}, indent=2)[:1500]
            stdout_for_prompt = (
                "IMPLAUSIBLE RESULT: the script exited 0 and printed RESULT_JSON, "
                "but these values break the bounds the DESIGN declared for them:\n"
                + "\n".join(f"- {v.describe()}" for v in implausible[:10])
                + "\n\nFind the cause before changing anything. If it is a bug "
                "(wrong units, a factor of two, a sign error, a mis-set "
                "parameter), fix it. If the value is genuinely what the method "
                "produces here (a scheme that diverges at this step size, an "
                "unstable fit), that is a finding, not a bug: emit null for that "
                "value and a flag saying why (for example \"diverged\": true). "
                "NEVER clamp, cap or clip a result to the bound or replace it "
                "with any constant: a capped value is detected and rejected, and "
                "it would state something false."
                + ("\n\nA value above is already capped at its bound by the "
                   "script itself; remove that cap." if clamped else "")
                + ("\n\nA quantity above sits exactly on a bound in several "
                   "settings. That is what a computation returning a trivial "
                   "answer looks like: a root finder settling on the solution at "
                   "the starting state, a sentinel, a guard branch, a threshold "
                   "compared on the wrong scale. Check how it is computed and fix "
                   "that if it is the cause; if the value really is the bound in "
                   "those settings, leave the code as it is." if at_bound else "")
                + "\n\nRESULT_JSON was:\n"
                f"{rj_preview}\n\nOriginal stdout tail:\n"
                + exec_result.get("stdout_tail", "")[:1000]
            )
            returncode_for_prompt = "0 (ran, but values break the declared bounds)"
            result_json_note = "yes (rejected — see stdout)"
        elif overlaps:
            self._log.warning(
                "[execute_reflect] figure_overlap: rc=0 and the results stand, but %d overlap(s) "
                "were measured on the saved canvas -- asking for one redraw (iter %d): %s",
                len(overlaps), iters + 1, "; ".join(overlaps[:3]),
            )
            stdout_for_prompt = _figure_overlap_directive(overlaps)
            returncode_for_prompt = "0 (ran; the figures overlap)"
            result_json_note = "yes (stands: change only the figures' layout)"
        else:
            stdout_for_prompt = exec_result.get("stdout_tail", "")[:2000]
            returncode_for_prompt = str(rc)
            result_json_note = "yes" if has_result_json else "no"

        # The one round the redraw gets, spent whatever the model answers.
        spent: QuestState = {"figure_overlap_repaired": True} if overlaps else {}
        prompt = self._prompts["execute_reflect"].substitute(
            # Whole for a redraw: it returns the script, and a script cut at the
            # limit would lose its tail.
            previous_code=(state.get("code") or "") if overlaps else (state.get("code") or "")[:8000],
            returncode=returncode_for_prompt,
            stdout_tail=stdout_for_prompt,
            stderr_tail=exec_result.get("stderr_tail", "")[:2000],
            duration_s=f"{exec_result.get('duration_s', 0):.2f}",
            figures_count=str(len(state.get("figures") or [])),
            result_json_present=result_json_note,
            reflect_history_block=history_block,
            design_block=json.dumps(state.get("design") or {}, indent=2),
            clarify_block=_format_clarify(state),
        )
        try:
            text = await self._chat(prompt, node="execute_reflect")
        except Exception as exc:  # noqa: BLE001
            if not overlaps:
                raise
            # A redraw is a nicety on a run that worked: failing to ask for it
            # never stops the quest.
            self._log.warning(
                "[execute_reflect] the figure redraw call failed (%r); keeping the figures as drawn", exc,
            )
            return spent
        parsed = _parse_json_lenient(text) or {}

        # A run whose figures merely overlap has a result that stands, so a
        # refusal, an empty answer or a redraw that is not Python leaves the
        # script and its figures as they are: no give-up sentinel (which would
        # switch off every later repair) and no repair attempt spent.
        give_up = (parsed.get("give_up_reason") or "").strip()
        if give_up:
            self._log.warning("[execute_reflect] LLM gave up: %s", give_up[:200])
            if overlaps:
                return spent
            history.append({
                "iter": iters + 1,
                "returncode": rc,
                "stderr_tail": exec_result.get("stderr_tail", "")[-400:],
                "patch_summary": f"(gave up: {give_up[:120]})",
            })
            return {
                "exec_give_up_reason": give_up,
                "exec_reflect_iter": iters + 1,
                "exec_reflect_history": history,
            }

        new_code = parsed.get("code") or ""
        if not new_code.strip():
            self._log.warning(
                "[execute_reflect] LLM returned no `code` field; proceeding without repair"
            )
            if overlaps:
                return spent
            return {
                "exec_give_up_reason": "(LLM produced no patched code)",
                "exec_reflect_iter": iters + 1,
            }
        if overlaps:
            try:
                ast.parse(new_code)
            except (SyntaxError, ValueError):
                self._log.warning(
                    "[execute_reflect] the figure redraw is not a Python script; keeping the "
                    "script as written, with its figures as drawn",
                )
                return spent

        patch_summary = parsed.get("patch_summary") or "(no summary)"
        history.append({
            "iter": iters + 1,
            "returncode": rc,
            "stderr_tail": exec_result.get("stderr_tail", "")[-400:],
            "patch_summary": patch_summary[:200],
        })
        self._log.info(
            "[execute_reflect] iter=%d patch=%s",
            iters + 1, patch_summary[:120],
        )

        # Write the patched code to disk so the next `execute` picks it
        # up. We mirror the implement node's behavior.
        code_path = self.quest_root / "code" / "experiment.py"
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(new_code, encoding="utf-8")

        patch: QuestState = {
            **spent,
            "code": new_code,
            "exec_reflect_iter": iters + 1,
            "exec_reflect_history": history,
            "exec_patch_pending": True,
        }
        # If the agent declared additional deps for the fix, merge them
        # so the next `execute` pip-installs them.
        new_deps = parsed.get("deps") or []
        if isinstance(new_deps, list) and new_deps:
            existing = list(state.get("deps") or [])
            patch["deps"] = list(dict.fromkeys(existing + [str(d) for d in new_deps]))
        return patch

    def _list_figures(self) -> list[str]:
        fdir = self.quest_root / "figures"
        if not fdir.is_dir():
            return []
        return sorted(
            p.name for p in fdir.iterdir()
            if p.is_file() and p.suffix.lower() in _FIGURE_SUFFIXES
        )

    def _gather_collected_text(self, state: QuestState, *, limit: int = 12) -> str:
        """Concatenate the web/data content the collector gathered (full
        page text + auto-collected files), each labelled with its source,
        for the web-plots prompt to mine numbers from. Each source's text is
        relevance-selected to the quest question (the passages carrying the
        numbers), not truncated to its first N chars."""
        from .passages import select_relevant_excerpt
        opts = self._lit_kwargs(state)
        query, budget, mode = opts["query"], opts["budget"], opts["mode"]
        parts: list[str] = []
        for item in (state.get("literature") or [])[:limit]:
            meta = item.get("metadata") or {}
            if meta.get("source") == "web_search" or meta.get("kind") == "web_page":
                title = meta.get("title") or ""
                url = meta.get("url") or ""
                content = select_relevant_excerpt(
                    item.get("content") or "", query,
                    budget_chars=budget, mode=mode,
                )
                if content.strip():
                    parts.append(f"[SOURCE] {title} ({url})\n{content}")
        auto_dir = self.quest_root / "data" / "auto_collected"
        if auto_dir.is_dir():
            for p in sorted(auto_dir.rglob("*.md"))[:limit]:
                try:
                    parts.append(f"[FILE] {p.name}\n{p.read_text(encoding='utf-8')[:budget]}")
                except OSError:
                    continue
        # The user's OWN data files (e.g. a dropped data/ridership.csv, an
        # --analyze-staged spreadsheet) were ignored — so web_plots reported
        # "no collected content to plot" even when the numbers to chart were
        # sitting right there. Read the ordinary data files too (via the
        # summarizer, so csv/json AND xlsx/parquet all yield real content),
        # skipping the literature / auto_collected subtrees covered above.
        from .summarizer import _classify_extension, _read_text
        data_dir = self.quest_root / "data"
        if data_dir.is_dir():
            _skip_top = {"literature", "auto_collected"}
            # Collect matching files lazily and STOP after limit*2 — don't
            # materialize + sort the whole data/ tree (could be large). Sort
            # only the small collected set for deterministic prompt order,
            # and filter BEFORE the cap so non-data files don't crowd it out.
            collected: list[Path] = []
            for p in data_dir.rglob("*"):
                if not p.is_file() or p.name == "README.md":
                    continue
                rel = p.relative_to(data_dir)
                if rel.parts and rel.parts[0] in _skip_top:
                    continue
                if _classify_extension(p.suffix) == "other":
                    continue
                collected.append(p)
                if len(collected) >= limit * 2:
                    break
            for p in sorted(collected):
                content = _read_text(p, _classify_extension(p.suffix))
                if content.strip():
                    rel = p.relative_to(data_dir)
                    parts.append(f"[USER DATA] {rel.as_posix()}\n{content[:budget]}")
        return "\n\n".join(parts)

    async def _node_web_plots(self, state: QuestState) -> QuestState:
        """No-simulation mode: derive figures from the collected web/data
        content. The no-sim path runs no experiment, so without this a
        literature-driven quest is text-only. An LLM extracts the concrete
        quantitative data it finds in the collected sources and writes a
        self-contained matplotlib script (data inline — no file parsing,
        no fabrication); we run it in the SAME sandbox the simulation path
        uses and stamp each figure with its source. Best-effort: no
        plottable data, a script error, or a missing matplotlib all leave
        the quest figure-less and proceed — this never aborts the quest.
        """
        if not self.config.engine.web_derived_plots:
            self._log.info("[web_plots] engine.web_derived_plots=False — skipping")
            return {}
        if not state.get("no_simulation_resolved"):
            # The simulation path makes its own figures in `execute`.
            return {}
        sources_text = self._gather_collected_text(state)
        if not sources_text.strip():
            self._log.info("[web_plots] no collected content to plot — skipping")
            return {}

        # Hard wall-clock cap on the whole node. web_plots is best-effort
        # (skipping it just leaves the quest figure-less), so it must NEVER
        # hang the quest — a slow/stuck codex_cli call once wedged a run for
        # ~2 h here when the provider-level timeout didn't catch it. On
        # timeout we skip plots and proceed.
        budget = float(self.config.engine.web_plots_timeout_s)
        try:
            return await asyncio.wait_for(
                self._web_plots_render(state, sources_text), timeout=budget,
            )
        except asyncio.TimeoutError:
            self._log.warning(
                "[web_plots] exceeded %.0fs budget — skipping plots "
                "(quest continues figure-less)", budget,
            )
            return {}
        except Exception as e:
            self._log.warning("[web_plots] failed: %s — skipping plots", e)
            return {}

    async def _web_plots_render(self, state: QuestState, sources_text: str) -> QuestState:
        """The slow part of web_plots (LLM script + install + execute),
        wrapped by ``_node_web_plots`` in a hard timeout."""
        prompt = self._prompts["web_plots"].safe_substitute(
            topic=str(state.get("topic", ""))[:500],
            result_json=json.dumps(
                state.get("result_json") or {}, ensure_ascii=False,
            )[:2000],
            sources=sources_text[:12000],
        )
        try:
            raw = await self._chat(prompt, node="web_plots")
        except Exception as e:
            self._log.warning("[web_plots] LLM call failed: %s — skipping", e)
            return {}
        script = _strip_outer_fence(raw).strip()
        if (
            not script
            or script.upper().startswith("NO_PLOT")
            or "matplotlib" not in script
        ):
            self._log.info(
                "[web_plots] model reported no plottable data — skipping",
            )
            return {}

        # Prepend the FI house style so the figures match the paper/poster/
        # slides identity. It sets rcParams (font, palette, ground, grid),
        # which the script's own `import matplotlib.pyplot` then inherits.
        script = _FI_MPL_PREAMBLE + "\n" + script

        code_dir = self.quest_root / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        plot_script = code_dir / "web_plots.py"
        plot_script.write_text(script, encoding="utf-8")

        figs_before = set(self._list_figures())
        install = await self.executor.install(
            ["matplotlib"], quest_root=self.quest_root,
        )
        if install.returncode != 0:
            self._log.warning(
                "[web_plots] matplotlib install rc=%d stderr=%s — skipping plots",
                install.returncode, install.stderr[-200:],
            )
            return {}
        py = self.executor.python_path(self.quest_root)
        result = await self.executor.execute(
            [str(py), str(plot_script)],
            cwd=self.quest_root,
            timeout_s=min(self.config.execution.timeout_s, 120),
        )
        new_figs = [f for f in self._list_figures() if f not in figs_before]
        if result.returncode != 0:
            self._log.warning(
                "[web_plots] plot script rc=%d stderr=%s",
                result.returncode, result.stderr[-300:],
            )
        self._log.info(
            "[web_plots] produced %d figure(s): %s", len(new_figs), new_figs,
        )
        if not new_figs:
            return {}
        existing = list(state.get("figures") or [])
        merged = existing + [f for f in new_figs if f not in existing]
        return {"figures": merged}

    def _arxiv_ids_from_literature(self, state: QuestState) -> list[str]:
        """arXiv ids of the papers in this quest's literature (from metadata
        or an arxiv.org URL) — candidate sources for a cited-paper figure."""
        ids: list[str] = []
        for item in state.get("literature") or []:
            md = (item.get("metadata") or {}) if isinstance(item, dict) else {}
            aid = str(md.get("arxiv_id") or "").strip()
            if not aid:
                # New-style (2007+) 2401.01234 AND old-style hep-th/9701001.
                m = re.search(
                    r"arxiv\.org/(?:abs|pdf|html)/"
                    r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})",
                    str(md.get("url") or ""), re.I,
                )
                aid = m.group(1) if m else ""
            aid = re.sub(r"v\d+$", "", aid)
            if aid and aid not in ids:
                ids.append(aid)
        return ids

    async def _select_relevant_figures(self, topic: str, cands: list) -> list:
        """LLM gate: keep only figures genuinely relevant + appropriate as an
        illustration for the topic. A wrong image is worse than none, so on
        any error we fail SAFE — keep cited-paper (arXiv) figures, which are
        inherently on-topic, and drop the keyword-searched Commons ones."""
        lines = "\n".join(
            f"[{i}] ({c.kind}) {c.caption[:160]}" for i, c in enumerate(cands)
        )
        prompt = (
            f"Topic: {topic[:400]}\n\nCandidate illustrative figures:\n{lines}\n\n"
            "Return ONLY a JSON array of the indices that are genuinely "
            "relevant AND appropriate as an illustration for THIS topic. Drop "
            "anything off-topic, misleading, or only superficially keyword-"
            'matching (e.g. atmospheric "gravity waves" for a gravitational-'
            "wave topic). Fewer, on-point figures are better. Example: [0, 2]"
        )
        try:
            raw = await self._chat(prompt, node="web_figures")
            m = re.search(r"\[[^\]]*\]", raw or "")
            idxs = json.loads(m.group(0)) if m else []
            keep = [cands[i] for i in idxs
                    if isinstance(i, int) and 0 <= i < len(cands)]
            return keep
        except Exception as e:
            self._log.info(
                "[web_figures] selection failed (%s); keeping cited-paper "
                "figures only", e,
            )
            return [c for c in cands if c.kind == "oa_paper"]

    async def _node_web_figures(self, state: QuestState) -> QuestState:
        """No-simulation mode: embed a few LICENSE-CLEAN illustrative figures
        (a real figure from a cited CC-licensed arXiv paper + Wikimedia
        Commons diagrams) so the study carries a visual point of view beyond
        its own charts. Each figure is attribution-stamped and recorded in
        ``figure_credits`` for the writer + references. Best-effort: any miss
        leaves the quest unchanged."""
        k = self.config.knowledge
        # Survey mode is a text-only path (no experiment, no data plots), so
        # illustrative images ARE the paper's only visuals — force web_figures
        # on for survey even when ``knowledge.fetch_web_figures`` is off (its
        # default), still honouring ``web_figures_max``. Other modes keep the
        # explicit opt-in.
        survey = bool(state.get("survey_mode_resolved"))
        if (not k.fetch_web_figures and not survey) or k.web_figures_max <= 0:
            return {}
        if not state.get("no_simulation_resolved"):
            return {}
        from .figure_sources import collect_topic_figures

        topic = str(state.get("topic", ""))
        out_dir = self.quest_root / "figures"
        maxn = k.web_figures_max
        try:
            cands = await asyncio.to_thread(
                collect_topic_figures, topic,
                self._arxiv_ids_from_literature(state),
                out_dir=out_dir, max_n=maxn + 2,   # over-fetch for selection
                timeout_s=float(k.full_text_fetch_timeout_s),
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            self._log.warning("[web_figures] collection failed: %s — skipping", e)
            return {}
        if not cands:
            self._log.info("[web_figures] no license-clean figures found")
            return {}

        keep = (await self._select_relevant_figures(topic, cands))[:maxn]
        kept = {f.local_path for f in keep}
        for c in cands:  # drop the unselected downloads
            if c.local_path not in kept and c.local_path.exists():
                try:
                    c.local_path.unlink()
                except OSError:
                    pass
        if not keep:
            self._log.info("[web_figures] no candidate passed the relevance gate")
            return {}

        credits: list[dict[str, Any]] = []
        for f in keep:
            _stamp_figure_credit(f.local_path, f.source_url, f.license)
            credits.append({
                "file": f.local_path.name, "caption": f.caption,
                "source_url": f.source_url, "license": f.license,
                "attribution": f.attribution, "kind": f.kind,
            })
        existing = list(state.get("figures") or [])
        merged = existing + [c["file"] for c in credits if c["file"] not in existing]
        # Accumulate credits (don't clobber any already in state on a re-entry),
        # deduped by file so a re-run doesn't double-list the same figure.
        prior_credits = list(state.get("figure_credits") or [])
        seen = {c.get("file") for c in prior_credits}
        all_credits = prior_credits + [c for c in credits if c["file"] not in seen]
        self._log.info(
            "[web_figures] embedded %d license-clean illustrative figure(s): %s",
            len(credits), [c["file"] for c in credits],
        )
        return {"figures": merged, "figure_credits": all_credits}

    async def _node_analyze(self, state: QuestState) -> QuestState:
        self._log.info("[analyze] interpreting results")
        exec_result = state.get("exec_result") or {}
        # Authoritative degenerate-run detection against the FINAL result
        # the analysis will describe (the execute-repair loop may have
        # exhausted its budget without producing real numbers). When set,
        # we (a) prepend a banner to the analyze prompt so the write node
        # frames an honest failure note, and (b) flag state for review.
        degenerate = bool(
            self.config.engine.degenerate_run_guard
            and _is_degenerate_result(state.get("result_json") or {})
        )
        if degenerate:
            self._log.warning(
                "[analyze] result is degenerate (all metrics ~0) — framing as a "
                "failure note, not a real result",
            )
        # Pick up any datasets the user dropped into inputs/data/ (e.g.
        # via the pause-drop-anytime gate). The design node also picks
        # these up on a post-pause resume; we run again here so a quest
        # that hit ``after_paper`` (where design didn't re-fire) still
        # surfaces fresh drops to analyze. Merge dedup'd with what's
        # already on state.
        ds_added = _pick_up_user_dropped_datasets(self.quest_root)
        existing_ds = list(state.get("user_supplied_datasets") or [])
        all_ds = list(dict.fromkeys(existing_ds + ds_added))
        # Multi-seed replication: when the execute node produced more
        # than one replicate, render the per-seed JSONs PLUS a mean ± std
        # aggregate of numeric fields into the analyze prompt. The
        # primary ``result_json`` (seed 0) is the headline value; the
        # ``replicates`` block lets the LLM discuss variance honestly
        # rather than treating a single-seed result as ground truth.
        # We also splice user-supplied datasets (when the user dropped
        # any via the pause-drop gate) into the same block so analyze
        # sees them alongside.
        replicates = state.get("result_json_replicates") or []
        if replicates and len(replicates) > 1:
            assertions = _replicate_assertions(state)
            agg = _aggregate_result_json_replicates(replicates, assertions=assertions)
            # Flattening a crossed design yields one entry per numeric leaf —
            # easily hundreds on a parameter sweep, each carrying seven stats.
            # Only the ones that actually MOVED between seeds tell the reader
            # anything, and the rest would crowd out the analysis they are
            # meant to inform (``_compact_result_json_block`` would otherwise
            # truncate at an arbitrary point). Entries that never varied are
            # dropped and counted; the per-seed JSON above still carries them.
            constant = {k for k, v in agg.items() if v.get("n", 0) > 1 and not v.get("std")}
            varying = {k: v for k, v in agg.items() if k not in constant}
            payload: dict[str, Any] = {
                "result_json_seed_0": state.get("result_json") or {},
                "replicates": replicates,
                # Each metric carries mean/std/n/min/max + se + 95% CI bounds.
                "aggregate_mean_std": varying,
                "n_replicates": len(replicates),
            }
            if constant:
                payload["aggregate_note"] = (
                    f"{len(constant)} further metric(s) were identical across "
                    f"all {len(replicates)} seeds and are omitted here; see "
                    f"result_json_seed_0 for their values."
                )
            # Per-stratum CIs + pairwise effect sizes + multiple-comparison
            # guard for any by_<factor> breakdowns (empty otherwise).
            comparison_stats = _result_comparison_stats(replicates, assertions=assertions)
            if comparison_stats:
                payload["comparison_stats"] = comparison_stats
            if all_ds:
                payload["_user_supplied_datasets"] = all_ds
            result_json_block, _rj_orig = _compact_result_json_block(payload)
            if state.get("result_json_deterministic"):
                # Without this the line reads "0 numeric keys", which is what a
                # broken aggregator looks like. Every seed agreeing is a fact
                # about the experiment, not a failure to measure.
                self._log.info(
                    "[analyze] replicates agreed exactly (n=%d): the experiment "
                    "is deterministic, so there are no error bars to report",
                    len(replicates),
                )
            else:
                self._log.info(
                    "[analyze] using replicate aggregate (n=%d, %d varying "
                    "metric(s), %d identical, %d comparison(s))",
                    len(replicates), len(varying), len(constant),
                    (comparison_stats.get("comparisons") or {}).get("n", 0),
                )
        else:
            result_json_block_data: dict[str, Any] = dict(state.get("result_json") or {})
            if all_ds:
                result_json_block_data["_user_supplied_datasets"] = all_ds
            result_json_block, _rj_orig = _compact_result_json_block(result_json_block_data)
        if _rj_orig > len(result_json_block):
            self._log.warning(
                "[analyze] result_json was %d chars; compacted to %d to fit the "
                "prompt budget — raw arrays in the experiment output were elided. "
                "The interpretation rests on the summary stats; if detail is lost, "
                "have the experiment emit summary statistics (not raw arrays) in "
                "its RESULT_JSON.", _rj_orig, len(result_json_block),
            )
        stdout_for_analyze = exec_result.get("stdout_tail", "")[:2000]
        if degenerate:
            stdout_for_analyze = (
                "[FI NOTE] The experiment ran but every numeric metric is ~0 — "
                "a degenerate/broken run, not a real result. Write this up as an "
                "honest failure note: state plainly that no usable measurement was "
                "produced, hypothesize the likely cause (sampling/grid, threshold, "
                "or normalization bug), and do NOT report the zeros as findings.\n\n"
                + stdout_for_analyze
            )
        if state.get("result_json_replicate_seed_ignored"):
            stdout_for_analyze = (
                "[FI NOTE] Replication was configured, but the experiment script "
                "never reads FI_REPLICATE_SEED, so every replicate repeated the "
                "SAME run and returned the same numbers. There is ONE measurement "
                "here, not several. Report it as a single run: do NOT report a "
                "mean over seeds, a standard error, a confidence interval, or any "
                "spread across replicates, and say plainly in the limitations that "
                "the experiment was not replicated because it does not vary with "
                "the seed it is given.\n\n"
                + stdout_for_analyze
            )
        # Read the ACTUAL contents of any user-dropped data (the pause-drop
        # gate writes to inputs/data/). Previously only the file PATHS were
        # surfaced, so analyze never saw the numbers — a dropped latency.csv
        # showed up as a filename, not its rows. Render the real content via
        # the same summarizer the no-sim data_load uses, budgeted.
        user_data_block = "(none)"
        drop_dir = self.quest_root / "inputs" / "data"
        if drop_dir.is_dir():
            from .summarizer import (
                _render_content_blocks,
                _render_file_manifest,
                _walk_folder,
            )
            try:
                # Filter the FI-authored README.md (the "drop your data here"
                # instructions) so it doesn't leak into the prompt as a
                # "source" — same guard as _node_data_load — and re-number
                # idents so the manifest stays sequential.
                entries = [
                    e for e in _walk_folder(drop_dir)
                    if not (e.rel_path == "README.md"
                            and (drop_dir / "README.md").is_file())
                ]
                for new_id, e in enumerate(entries, start=1):
                    e.ident = new_id
                if entries:
                    # Manifest first so _render_content_blocks' "files
                    # elided" note ("the manifest above") is truthful and
                    # binary/unreadable files are still visible to the model.
                    manifest = _render_file_manifest(entries)
                    blocks = _render_content_blocks(entries, total_budget_chars=8000)
                    user_data_block = f"{manifest}\n\n{blocks}".strip() or "(none)"
            except OSError as e:
                self._log.debug("[analyze] could not read dropped data: %r", e)
        prompt = self._prompts["analyze"].substitute(
            clarify_block=_format_clarify(state),
            design_block=json.dumps(state.get("design") or {}, indent=2),
            returncode=str(exec_result.get("returncode")),
            duration_s=f"{exec_result.get('duration_s', 0):.1f}",
            timed_out=str(exec_result.get("timed_out", False)),
            stdout_tail=stdout_for_analyze,
            stderr_tail=exec_result.get("stderr_tail", "")[:1000],
            result_json=result_json_block,
            user_data_block=user_data_block,
            figure_list=_figure_list_for_prompt(state),
            # The literature is analyze's ONLY evidence in survey / no-sim
            # mode (no experiment, and — in survey mode — no dataset either),
            # so it must reach the analyze prompt to be synthesised. Harmless
            # extra grounding in the simulation path.
            literature_block=_format_lit_from_state(state, **self._lit_kwargs(state)),
        )
        # Multi-model ensemble path: when the YAML carries
        # provider.node_ensemble["analyze"], fan out N parallel calls
        # and merge. The moderator's synthesized output is then parsed
        # the same way as the single-call path, so the rest of the
        # pipeline (cross_check, write) sees an identical shape.
        ensemble_cfg = self._ensemble_for_node("analyze")
        if ensemble_cfg is not None:
            from core.ensemble import EnsembleError
            try:
                result = await self._ensemble_chat(
                    prompt, node="analyze", ensemble_cfg=ensemble_cfg,
                )
                text = result.merged if isinstance(result.merged, str) else json.dumps(result.merged)
                if result.disagreement_score > 0:
                    self._log.info(
                        "[analyze] ensemble disagreement_score=%.2f (%d models)",
                        result.disagreement_score, len(ensemble_cfg.models),
                    )
            except EnsembleError as e:
                self._log.warning(
                    "[analyze] ensemble all-failed (%s); falling back to single-call path", e,
                )
                text = await self._chat(prompt, node="analyze")
        else:
            text = await self._chat(prompt, node="analyze")
        analysis = _parse_json_lenient(text) or {"summary": "(parse failed)", "key_findings": []}
        # Default `next_step` to publish when the LLM omits it (older
        # prompts, parse failures) so the route doesn't break.
        analysis.setdefault("next_step", "publish")
        patch: QuestState = {"analysis": analysis}
        if degenerate:
            patch["degenerate_result"] = True
        if all_ds:
            patch["user_supplied_datasets"] = all_ds
        return patch

    async def _node_cross_check(self, state: QuestState) -> QuestState:
        """For each key finding, search literature with the
        finding text as the query, then classify hits as supporting /
        conflicting / neutral. Results land in ``state['cross_check']``
        and are surfaced in the write prompt's ``$cross_check_block``."""
        findings = list((state.get("analysis") or {}).get("key_findings") or [])
        if not findings:
            self._log.info("[cross_check] no key_findings; skipping")
            return {"cross_check": []}

        per_finding_k = self.config.engine.cross_check_per_finding_k
        if per_finding_k <= 0:
            self._log.info("[cross_check] disabled by config (per_finding_k=0)")
            return {"cross_check": []}

        out: list[dict[str, Any]] = []
        for finding in findings[:10]:  # cap to avoid runaway LLM cost
            text = str(finding).strip()
            if not text:
                continue
            self._log.info("[cross_check] searching for: %s", text[:80])
            try:
                hits = await self.knowledge.asearch(
                    text, top_k=per_finding_k,
                    chat_fn=functools.partial(self._chat_messages, node="source_router"),
                    work_scope=self._work_scope(state),
                )
            except Exception as e:
                self._log.warning("[cross_check] retrieval failed: %s", e)
                hits = []
            if not hits:
                out.append({
                    "finding": text,
                    "supporting": [], "conflicting": [], "neutral": [],
                    "summary": "(no related literature surfaced)",
                    "candidates": [],
                })
                continue
            # Classify. Single call by default; multi-model vote when
            # provider.node_ensemble["cross_check"] is configured —
            # majority verdict wins per-finding, ties surfaced. Either
            # way ``parsed`` carries the same shape downstream.
            cand_block = _format_lit(hits, **self._lit_kwargs(state))
            prompt = self._prompts["cross_check"].substitute(
                topic=state.get("topic", "")[:1000],
                finding=text,
                candidate_literature=cand_block,
            )
            ensemble_cfg = self._ensemble_for_node("cross_check")
            try:
                if ensemble_cfg is not None:
                    from core.ensemble import EnsembleError
                    try:
                        result = await self._ensemble_chat(
                            prompt, node="cross_check", ensemble_cfg=ensemble_cfg,
                        )
                        # merge_vote returns a dict {verdict, tally, tie};
                        # we still need the supporting/conflicting/neutral
                        # lists, so we re-parse one of the survivors that
                        # voted with the majority. Fallback: first survivor.
                        majority = result.merged if isinstance(result.merged, dict) else {}
                        survivors = [r for r in result.raw if r.ok]
                        # Pick a survivor whose JSON agrees with the majority verdict
                        # so we get a consistent supporting/conflicting/neutral block.
                        winner_text = ""
                        for s in survivors:
                            try:
                                obj = json.loads(s.text.strip())
                                if obj.get("verdict") == majority.get("verdict"):
                                    winner_text = s.text
                                    break
                            except Exception:
                                continue
                        parsed = _parse_json_lenient(winner_text or (survivors[0].text if survivors else "")) or {}
                        if majority.get("tie"):
                            self._log.warning("[cross_check] vote tie on finding %r — picked first occurrence", text[:60])
                    except EnsembleError as e:
                        self._log.warning("[cross_check] ensemble all-failed (%s); falling back to single call", e)
                        resp = await self._chat(prompt, node="cross_check")
                        parsed = _parse_json_lenient(resp) or {}
                else:
                    resp = await self._chat(prompt, node="cross_check")
                    parsed = _parse_json_lenient(resp) or {}
            except Exception as e:
                self._log.warning("[cross_check] classify call failed: %s", e)
                parsed = {}
            candidates = [
                {
                    "title": d.metadata.get("title", "")[:200],
                    "source": d.metadata.get("source", ""),
                    "doi": d.metadata.get("doi", ""),
                    "url": d.metadata.get("url", ""),
                }
                for d in hits
            ]

            verification_notes: list[dict[str, Any]] = []
            # CoVe-style verification pass. Skipped when disabled by
            # config OR when the first pass produced no non-neutral
            # assignments (nothing to verify). Cost: +1 LLM call per
            # finding. The pass can downgrade over-claimed supports
            # to neutral, flip sign errors, or add a verification
            # note that explains why the original direction held up.
            do_verify = (
                self.config.engine.cross_check_verify
                and (parsed.get("supporting") or parsed.get("conflicting"))
            )
            if do_verify:
                try:
                    verify_prompt = self._prompts["cross_check_verify"].substitute(
                        topic=state.get("topic", "")[:1000],
                        finding=text,
                        first_pass_block=json.dumps({
                            "verdict": parsed.get("verdict") or "neutral",
                            "supporting": parsed.get("supporting") or [],
                            "conflicting": parsed.get("conflicting") or [],
                            "neutral": parsed.get("neutral") or [],
                            "summary": parsed.get("summary") or "",
                        }, indent=2),
                        candidate_literature=cand_block,
                    )
                    verify_resp = await self._chat(
                        verify_prompt, node="cross_check_verify",
                    )
                    verified = _parse_json_lenient(verify_resp) or {}
                    # Require BOTH a usable revised classification AND a
                    # non-None verdict in the verified payload before
                    # adopting it — a parse that drops verdict (or
                    # returns ``None`` for it) would leave the merged
                    # ``parsed`` and ``first_pass`` carrying ``None``
                    # verdicts which downstream verdict tally and audit
                    # logs read as "no opinion" rather than the actual
                    # first-pass direction.
                    has_classification = isinstance(verified, dict) and (
                        verified.get("supporting") is not None
                        or verified.get("conflicting") is not None
                        or verified.get("neutral") is not None
                    )
                    has_verdict = (
                        isinstance(verified, dict)
                        and isinstance(verified.get("verdict"), str)
                        and verified.get("verdict")
                    )
                    if has_classification and has_verdict:
                        # Verification produced a usable revision —
                        # adopt it. Track the original classification
                        # alongside for the write/audit downstream. The
                        # ``verdict`` fallback in the snapshot mirrors
                        # the first-pass prompt's default behaviour so
                        # a malformed first-pass that omitted verdict
                        # still leaves a defensible audit record.
                        first_pass_snapshot = {
                            "verdict": parsed.get("verdict") or "neutral",
                            "supporting": parsed.get("supporting") or [],
                            "conflicting": parsed.get("conflicting") or [],
                            "neutral": parsed.get("neutral") or [],
                        }
                        parsed = verified
                        notes = verified.get("verification_notes") or []
                        if isinstance(notes, list):
                            verification_notes = [
                                n for n in notes if isinstance(n, dict)
                            ]
                        parsed["first_pass"] = first_pass_snapshot
                        self._log.info(
                            "[cross_check] verified finding %r (notes=%d)",
                            text[:60], len(verification_notes),
                        )
                    elif isinstance(verified, dict):
                        # Verification call succeeded but the payload
                        # was unusable. Log + fall through with the
                        # first-pass result preserved.
                        self._log.warning(
                            "[cross_check] verification dropped (verdict=%r, "
                            "classifications=%s) — keeping first pass",
                            verified.get("verdict"), has_classification,
                        )
                except Exception as e:
                    # Verification is advisory — a transient failure
                    # MUST NOT block the quest; the first-pass result
                    # is still valid output.
                    self._log.warning(
                        "[cross_check] verification pass failed (kept first pass): %s", e,
                    )

            entry: dict[str, Any] = {
                "finding": text,
                "supporting": parsed.get("supporting") or [],
                "conflicting": parsed.get("conflicting") or [],
                "neutral": parsed.get("neutral") or [],
                "summary": parsed.get("summary") or "",
                "candidates": candidates,
                "verification_notes": verification_notes,
            }
            # ``first_pass`` is only emitted when verification actually
            # ran AND produced a usable revision (the ``parsed`` dict
            # then carries it). When verification was disabled / skipped
            # / dropped, the key is omitted entirely rather than written
            # as ``null`` — keeps the JSON tighter and lets consumers
            # use ``"first_pass" in entry`` as the "was verified"
            # sentinel.
            if parsed.get("first_pass") is not None:
                entry["first_pass"] = parsed.get("first_pass")
            out.append(entry)
        self._log.info("[cross_check] checked %d findings", len(out))
        patch: QuestState = {"cross_check": out}
        # Cross-check iteration accounting: if analyze flagged a re-route AND
        # there's budget left, bump the shared iteration counter here so
        # the design node sees the new iteration on its next visit. This
        # mirrors how `_node_review` bumps on `verdict=revise`.
        if self.config.engine.enable_analyze_reroute:
            next_step = (state.get("analysis") or {}).get("next_step", "publish")
            if (
                next_step in ("re_experiment", "broaden_lit")
                and state.get("iteration", 0) < self.config.engine.max_iterations
            ):
                patch["iteration"] = state.get("iteration", 0) + 1
                self._log.info(
                    "[cross_check] analyze.next_step=%s -> redesign (iteration %d)",
                    next_step, patch["iteration"],
                )
        return patch

    def _resolve_write_persona(self, state: QuestState) -> str:
        """Pick a persona prefix for the ``write`` node based on the
        format hint in clarify answers + output config.

        Scientific venues return empty so ``write.md``'s built-in
        IMRAD framing carries the prompt unchanged. Non-scientific
        formats load their persona prefix from
        ``agents/write_persona_<name>.md`` via ``_load_persona_prefix``
        — same pattern as the ``review_persona_*`` files. Unknown
        format values fall back to empty (defense-in-depth for the
        clarify path, which accepts free-form strings from the LLM).

        Resolution order (first match wins):
        1. ``state["clarify_answers"]["paper_venue"]`` — clarify
           agent's pick (LLM-generated; normalized via ``.strip().lower()``).
        2. ``self.config.output.paper_format`` — YAML default
           (Pydantic-validated against ``PaperFormat`` so case is
           already canonical).
        """
        venue = ""
        answers = state.get("clarify_answers") or {}
        if isinstance(answers, dict):
            venue = str(answers.get("paper_venue") or "").strip().lower()
        if not venue:
            venue = self.config.output.paper_format or ""

        if venue in SCIENTIFIC_PAPER_FORMATS:
            return ""
        if venue not in NON_SCIENTIFIC_PAPER_FORMATS:
            return ""
        return _load_persona_prefix(venue, category="write")

    async def _node_evidence_gate(self, state: QuestState) -> QuestState:
        """Weigh the assembled evidence against the research question
        BEFORE writing. Returns a verdict the writer (and router) act on.

        Fails OPEN — when the flag is off, or anything goes wrong, the
        node is a passthrough and routing defaults to ``write`` (the
        downstream review + claim_check still guard quality). The only
        action it can take that changes control flow is ONE bounded
        ``broaden_lit`` re-entry (capped by ``evidence_gate_max_broaden``).
        """
        if not self.config.engine.evidence_gate:
            return {}
        # The typed contract this quest is held to (topic type, source
        # policy, baseline, success metric). derive_protocol is pure +
        # total — it guards every state access and has defaults — so it
        # won't raise on a malformed checkpoint; building it here means
        # it's always recorded in state AND available to the gate prompt
        # below so the sufficiency call is contract-aware.
        protocol = derive_protocol(state, self.config)
        # EVERYTHING below is inside one fail-open guard, including the
        # signal-gathering: a resumed / malformed checkpoint can hand us
        # non-dict literature items, a non-dict analysis, or non-dict
        # cross_check entries, and the gate must degrade to "write" rather
        # than abort the whole quest on an AttributeError. Each access is
        # also type-guarded so one bad entry doesn't poison the tally.
        parsed: dict[str, Any] = {}
        decided_by = "model"
        n_sources = n_supporting = n_findings = 0
        try:
            lit = state.get("literature")
            lit = lit if isinstance(lit, list) else []
            n_sources = sum(
                1 for d in lit
                if isinstance(d, dict) and str(d.get("content") or "").strip()
            )
            # The gate is asked to judge whether sources are real / on-topic /
            # off-topic — so it must actually SEE them. Pass each source's
            # title + a short snippet (not just the count), capped so the
            # prompt stays bounded. Without this the agent was judging
            # relevance blind (an off-topic "Banana Bread" source for a
            # "SpaceX revenue" topic looked identical to an on-topic one).
            source_previews: list[dict[str, str]] = []
            for d in lit[:10]:
                if not isinstance(d, dict):
                    continue
                md = d.get("metadata")
                md = md if isinstance(md, dict) else {}
                title = str(md.get("title") or "").strip()
                # Slice BEFORE normalising whitespace so we don't .split()
                # a 16k-char body just to keep 240 chars (Copilot, #199).
                snippet = " ".join(str(d.get("content") or "")[:400].split())[:240]
                if title or snippet:
                    source_previews.append({
                        "title": title[:160] or "(untitled)",
                        "snippet": snippet,
                    })
            analysis = state.get("analysis")
            analysis = analysis if isinstance(analysis, dict) else {}
            findings = analysis.get("key_findings")
            findings = findings if isinstance(findings, list) else []
            n_findings = len(findings)
            cc = state.get("cross_check")
            cc = cc if isinstance(cc, list) else []
            n_conflicting = 0
            for rec in cc:
                if not isinstance(rec, dict):
                    continue
                # ``_node_cross_check`` writes per-finding records shaped
                # ``{supporting: [...], conflicting: [...], neutral: [...]}``
                # — read THOSE buckets. (The old code looked for
                # ``hits``/``results`` with per-hit ``stance`` fields, which
                # this structure never carries, so both counters were always
                # 0 and the gate was blind to real support/conflict.) Keep a
                # fallback for the legacy stance shape just in case.
                supporting = rec.get("supporting")
                conflicting = rec.get("conflicting")
                if supporting is None and conflicting is None:
                    hits = rec.get("hits") or rec.get("results") or []
                    stances = [
                        str(h.get("stance") or h.get("classification") or "").lower()
                        for h in hits if isinstance(h, dict)
                    ] if isinstance(hits, list) else []
                    supporting = [s for s in stances if s == "supporting"]
                    conflicting = [s for s in stances if s == "conflicting"]
                if supporting:
                    n_supporting += 1
                if conflicting:
                    n_conflicting += 1
            summary = {
                "n_sources_with_text": n_sources,
                "sources": source_previews,
                "n_key_findings": n_findings,
                "n_findings_with_supporting_lit": n_supporting,
                "n_findings_with_conflicting_lit": n_conflicting,
                "has_results": bool(state.get("result_json") or {}),
                "mode": (
                    "no_simulation"
                    if state.get("no_simulation_resolved") else "simulation"
                ),
                "key_findings_preview": [str(f)[:200] for f in findings[:6]],
            }
            ruled = _evidence_gate_rule(
                protocol.topic_type, n_sources, n_supporting,
                analyze_local_first=self.config.engine.analyze_local_first,
                retrieval_on=self.config.knowledge.enabled,
            )
            if ruled is not None:
                parsed, decided_by = ruled, "rule"
            else:
                prompt = self._prompts["evidence_gate"].substitute(
                    topic=str(state.get("topic") or ""),
                    clarify_block=_format_clarify(state),
                    protocol_block=protocol.as_block(),
                    evidence_summary=json.dumps(summary, indent=2),
                )
                text = await self._chat(prompt, node="evidence_gate")
                parsed = _parse_json_lenient(text, node="evidence_gate") or {}
        except Exception as e:  # noqa: BLE001 — gate must never abort the quest
            self._log.warning(
                "[evidence_gate] assessment failed (%r); failing open to write", e,
            )
        verdict = str(parsed.get("verdict") or "").strip().lower()
        if verdict not in ("sufficient", "broaden", "insufficient"):
            verdict = "sufficient"  # fail-open
        broadened = int(state.get("evidence_broaden_count", 0) or 0)
        will_broaden = (
            verdict == "broaden"
            and broadened < self.config.engine.evidence_gate_max_broaden
        )
        assessment = {
            "verdict": verdict,
            "route": "broaden_lit" if will_broaden else "write",
            "rationale": str(parsed.get("rationale") or ""),
            "gaps": [str(g) for g in (parsed.get("gaps") or []) if str(g).strip()],
            "n_sources": n_sources,
            "n_supporting": n_supporting,
            "decided_by": decided_by,
        }
        self._log.info(
            "[evidence_gate] verdict=%s route=%s decided=%s type=%s policy=%s "
            "(sources=%d, findings=%d, supporting=%d, broadened=%d)",
            verdict, assessment["route"], decided_by, protocol.topic_type,
            protocol.source_policy, n_sources, n_findings,
            n_supporting, broadened,
        )
        patch: QuestState = {
            "evidence_assessment": assessment,
            "research_protocol": protocol.model_dump(),
        }
        if will_broaden:
            patch["evidence_broaden_count"] = broadened + 1
        return patch

    async def _write_whole_paper(self, state: QuestState, persona_block: str) -> str:
        """The paper's markdown as the writer gives it: the whole paper, written
        from the study, with the review of an earlier draft (when there is one)
        in the prompt. A first draft always comes from here, and so does a revise
        that :meth:`_patch_flagged_passages` cannot answer with edits."""
        # When the evidence gate judged the evidence thin (insufficient,
        # or "broaden" with the broaden budget exhausted), hand the writer
        # an explicit note so the paper frames its limits honestly instead
        # of over-claiming. Suppressed for survey topics (see the helper).
        evidence_note = _format_evidence_note(
            state.get("evidence_assessment"),
            is_survey=bool(state.get("survey_mode_resolved")),
        )
        prompt = self._prompts["write"].substitute(
            persona_block=persona_block,
            topic=state["topic"],
            title=state.get("title", "Untitled"),
            design_block=json.dumps(state.get("design") or {}, indent=2),
            analysis_block=json.dumps(state.get("analysis") or {}, indent=2),
            # Write node is the ONE place audience filtering applies:
            # this is the literature block that flows into the paper's
            # References. ideate/design/cross_check still see the full
            # pull because those nodes are about choosing what to do,
            # not what to publish.
            literature_block=_format_lit_from_state(
                state, audience=self.config.output.audience,
                mark_thin=True, **self._lit_kwargs(state),
            ),
            # Empty when the prior-work block holds no foundational work.
            foundational_block=_foundational_write_block(
                state.get("literature") or [], self.config.output.audience,
            ),
            figure_list=_figure_list_for_prompt(state),
            clarify_block=_format_clarify(state),
            cross_check_block=_format_cross_check(state),
            study_mode_note=(
                _SURVEY_WRITE_NOTE
                if state.get("survey_mode_resolved")
                else _NO_SIM_WRITE_NOTE
                if state.get("no_simulation_resolved")
                else ""
            ),
            evidence_note=evidence_note,
            # Empty without a page limit, so that prompt is unchanged.
            page_limit_note=_page_limit_note(
                resolve_page_limit(self.config), _paper_figure_count(state),
            ),
            review_feedback=_format_review_for_writer(state),
            skills_block=(
                self._writing_skills_block(state)
                or "(no writing guidance selected for this quest)"
            ),
        )
        markdown = await self._chat(prompt, node="write")
        # The model may wrap with a fence; strip it.
        markdown = _strip_outer_fence(markdown)
        if _is_not_a_paper(markdown):
            raise RuntimeError(
                f"the writer returned {len(markdown.strip())} characters, which "
                f"cannot be a paper: {markdown.strip()[:200]!r}. If that reads "
                f"like a provider message (a usage limit, an expired login), fix "
                f"the provider and resume the quest."
            )
        return markdown

    async def _node_write(self, state: QuestState) -> QuestState:
        persona_block = self._resolve_write_persona(state)
        self._log.info(
            "[write] authoring paper.md (persona=%s)",
            persona_block.split("\n", 1)[0][:80] if persona_block else "default",
        )
        # A revise whose every must-fix hit names a passage edits those passages
        # of the earlier draft and leaves the rest as it was; anything else
        # (a first draft, a hit about the whole paper, edits the engine cannot
        # apply) writes the whole paper.
        markdown = await self._patch_flagged_passages(state, persona_block)
        if markdown is None:
            markdown = await self._write_whole_paper(state, persona_block)
        from generation._keywords import keep_one_keywords_form

        # A scientific paper shows its keywords; a persona's paper keeps them
        # in a comment. A writer that gave both keeps the one its format uses.
        markdown = keep_one_keywords_form(markdown, visible=not persona_block)
        # The engine writes the source lists, so every source listed is one the
        # text cites and none is invented: References numbers the cited papers
        # in the order the text first cites them, and Further reading lists
        # every web page. The literature is reordered to those numbers, so the
        # claim check, the slides, the poster, the bib and the next write pass
        # all use them.
        markdown, literature, dropped = _finalize_paper_sources(
            markdown, state.get("literature") or [], self.config.output.audience,
        )
        if dropped:
            self._log.warning(
                "[write] removed citations of sources the prior-work block does not have: %s",
                ", ".join(f"[{n}]" for n in dropped),
            )
        # What the foundational works came to: how many were in the prior-work
        # block and which of them the draft cites, so a paper that leaves the
        # field's original papers out can be found in the run log.
        foundational = _foundational_sources(literature, self.config.output.audience)
        if foundational:
            left_out = _uncited_foundational_works(literature, markdown, self.config.output.audience)
            self._log.info(
                "[write] foundational works in the prior-work block: %d; the draft cites %d; "
                "not cited: %s",
                len(foundational), len(foundational) - len(left_out),
                "; ".join(str(meta.get("title") or "") for _label, meta in left_out) or "-",
            )
        # A figure the design planned and the run drew that this draft leaves
        # out goes back in here, before the review reads it. The review checks
        # for it too, but only by forcing a rewrite that spends one of
        # ``engine.max_iterations`` — and on the last iteration it cannot spend
        # one, so the figure is simply lost. Putting it back costs no LLM call
        # and no iteration.
        markdown, placed = _place_missing_figures(markdown, state)
        if placed:
            self._log.warning(
                "[write] the draft left out %d figure(s) the design planned and the run "
                "drew; placed %s in the paper before the review",
                len(placed), ", ".join(placed),
            )
        paper_path = self.quest_root / "paper" / "paper.md"
        paper_path.write_text(markdown, encoding="utf-8")
        self._log.info("[write] wrote %s (%d bytes)", paper_path, len(markdown))
        # Generic pause-drop-anytime: when configured, pause AFTER
        # paper.md lands so the user can drop reference papers (for
        # the next revise pass to cite) and/or datasets (for analyze
        # on the next iteration to incorporate). On a post-pause
        # resume, ``user_pauses_fired`` carries "before_review" and the
        # gate falls through to review.
        self._maybe_pause_for_user_input(state, "before_review")
        return {
            "paper_md": str(paper_path), "literature": literature,
            "paper_basis": _paper_basis(state),
        }

    async def _patch_flagged_passages(self, state: QuestState, persona_block: str) -> str | None:
        """The earlier draft with the passages the review named edited, or
        ``None`` when this round writes the whole paper.

        Eligible when the review's must-fix hits all name one passage of the
        paper (an unsupported claim, a figure caption, a number nothing accounts
        for, a mislabelled statistic), the earlier draft is on disk, and nothing
        it was written from has been run again since (``paper_basis``). The model
        gets that draft and only those passages and answers with
        ``find``/``replace`` edits; :mod:`core.paper_patch` applies each edit only
        where its ``find`` occurs exactly once and leaves every other character
        as it was. The source lists are then rebuilt from the citations in the
        edited text by the same step a whole paper goes through.

        A first draft, a hit about the whole paper (``over_page_limit``,
        ``figure_missing``), a study that changed, a reply that is not usable
        edits, and an edit that cannot be applied all return ``None``, and the log
        says which."""
        review = state.get("review") or {}
        hits = [str(h).strip() for h in review.get("must_flag_hits") or [] if str(h).strip()]
        if not hits:
            return None  # a first draft, or a revise with no hit to answer
        previous = ""
        try:
            if state.get("paper_md"):
                previous = Path(str(state["paper_md"])).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            previous = ""
        grounding = state.get("claim_grounding") or {}
        claims = [
            {"claim": str(c.get("claim") or ""), "evidence": str(c.get("evidence") or "")}
            for c in grounding.get("claims") or []
            if isinstance(c, dict) and c.get("basis") == "unsupported"
        ] or [{"claim": str(c), "evidence": ""} for c in _review_items(grounding.get("unsupported"))]
        start = _SOURCE_LIST_HEADING_RE.search(previous)
        end = start.start() if start else len(previous)
        reason = ""
        passages: list[paper_patch.Passage] = []
        if not previous.strip():
            reason = "there is no earlier draft on disk to edit"
        elif state.get("paper_basis") != _paper_basis(state):
            reason = "the study the draft was written from has been run again since"
        elif len(previous) > _PAPER_PROMPT_CHARS:
            reason = f"the draft is longer than the {_PAPER_PROMPT_CHARS:,} characters a model reads"
        else:
            passages, reason = paper_patch.plan_passages(
                previous, end,
                hits=[(_hit_name(h), h) for h in hits],
                claims=claims,
                captions=_review_items(review.get("figure_caption_warnings")),
            )
        if reason:
            self._log.info("[write] writing the whole paper again: %s", reason)
            return None
        located = sum(1 for p in passages if p.located)
        prompt = self._prompts["write_patch"].substitute(
            persona_block=persona_block,
            topic=state["topic"],
            analysis_block=json.dumps(state.get("analysis") or {}, indent=2),
            literature_block=_format_lit_from_state(
                state, audience=self.config.output.audience, mark_thin=True, **self._lit_kwargs(state),
            ),
            passages_block=paper_patch.format_passages(passages),
            paper_block=previous,
        )
        self._log.info(
            "[write] editing the earlier draft: %d passage(s) named by the review, %d found in it",
            len(passages), located,
        )
        try:
            reply = await self._chat(prompt, node="write.patch")
        except Exception as e:  # noqa: BLE001 - the whole-paper call below is what a real outage stops
            self._log.warning(
                "[write] the call for edits failed (%s: %s); writing the whole paper again",
                type(e).__name__, str(e)[:200],
            )
            return None
        try:
            patched = paper_patch.apply_edits(previous, end, paper_patch.parse_edits(reply))
        except paper_patch.PatchError as e:
            self._log.warning(
                "[write] the edits could not be used (%s); writing the whole paper again. "
                "The reply began: %r", e, " ".join(str(reply).split())[:160],
            )
            return None
        self._log.info(
            "[write] applied %d edit(s) to the earlier draft (%d characters replaced by %d, %d unchanged "
            "as given); every other character of the paper is as it was",
            patched.applied, patched.removed_chars, patched.added_chars, patched.ignored,
        )
        return patched.text

    async def _node_claim_check(self, state: QuestState) -> QuestState:
        """Ground each substantive claim in the written paper to evidence — the
        quest's own experiment results, a cited reference, or flag it
        ``unsupported``. Writes a ``paper/claims.json`` + ``paper/CLAIMS.md``
        ledger and stashes the result in state so the reviewer must-flags any
        unsupported claims (forcing a bounded revise). No-op passthrough when
        ``engine.claim_grounding`` is off."""
        if not self.config.engine.claim_grounding:
            return {}
        paper_md = state.get("paper_md")
        if not paper_md or not Path(paper_md).is_file():
            self._log.info("[claim_check] no paper to check; skipping")
            return {}
        paper_text = _paper_for_prompt(Path(paper_md).read_text(encoding="utf-8"), "claim_check", self._log)
        literature = state.get("literature") or []
        audience = self.config.output.audience
        refs = build_references(literature, audience=audience)
        # Web pages are Further reading, labelled W1, W2...; a claim resting on
        # one is grounded in a source too.
        further = build_further_reading(literature, audience=audience)
        # Each source the paper cites comes with its text, and a citation must
        # quote it: the model checks what the source says, and the quote is
        # then looked up in the source.
        sources = {
            label: (meta, _item_content(item))
            for label, meta, item in _labelled_sources(literature, audience)
        }
        citing = _citing_sentences(paper_text)
        refs_block = "\n\n".join(
            _claim_source_block(label, meta, text, citing.get(label) or [])
            for label, (meta, text) in sorted(
                sources.items(), key=lambda kv: (kv[0].startswith("W"), int(kv[0].lstrip("W")))
            )
        ) or "(no references)"
        analysis = state.get("analysis") or {}
        # The findings and the supported claims go in whole and first — they
        # are what an "experiment" basis is checked against — and the run's
        # results then fill whatever is left of the budget.
        evidence_block, dropped = _claim_distilled_block(analysis, _CLAIM_EVIDENCE_CHARS)
        if dropped:
            self._log.warning(
                "[claim_check] %d finding(s)/supported claim(s) did not fit the "
                "%d-character evidence budget and were left out",
                dropped, _CLAIM_EVIDENCE_CHARS,
            )
        evidence_block += _claim_results_block(
            state.get("result_json") or {},
            query=evidence_block,
            budget=_CLAIM_EVIDENCE_CHARS - len(evidence_block),
        )
        # The paper reports means over the seeds with their intervals, which
        # seed 0's RESULT_JSON holds neither of. They follow the results, one
        # line each, and outside the budget above: they are a few lines that
        # the results must never be able to push out.
        intervals = _replicate_result_intervals(state)
        if intervals:
            n_seeds = len(state.get("result_json_replicates") or [])
            evidence_block += f"\n\nMean over the {n_seeds} seeds, with its 95% CI:\n" + "\n".join(
                f"- {path}: {s['mean']:.4g}"
                + (f" (95% CI {s['ci_lower']:.4g} to {s['ci_upper']:.4g})" if "ci_lower" in s and "ci_upper" in s else "")
                for path, s in list(intervals.items())[:40]
            )
        prompt = self._prompts["claim_check"].substitute(
            topic=state["topic"],
            evidence_block=evidence_block,
            references=refs_block,
            paper=paper_text,
        )
        # claim_check runs AFTER the paper is already written, so a provider
        # failure here (a bridge stall, a server out of capacity once its
        # retries are spent) must NOT abort the quest and forfeit the paper and
        # every downstream output. It must not pass silently either: a rewrite
        # whose check failed kept the previous draft's grounding, and nothing
        # checked its new citations. The failure is recorded, the stale
        # grounding cleared, and the review forces ``citations_unchecked``.
        try:
            text = await self._chat(prompt, node="claim_check")
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"[:300]
            self._log.warning(
                "[claim_check] grounding call failed (%s); the review will mark "
                "this draft's citations unchecked", reason,
            )
            return {"claim_grounding": {}, "claim_check_failed": reason}
        parsed = _parse_json_lenient(text) or {}
        raw_claims = parsed.get("claims") if isinstance(parsed, dict) else None
        n_refs = len(refs)
        claims: list[dict[str, Any]] = []
        for c in (raw_claims or []):
            if not isinstance(c, dict) or not str(c.get("claim") or "").strip():
                continue
            basis = str(c.get("basis") or "unsupported").strip().lower()
            if basis not in ("experiment", "citation", "unsupported"):
                basis = "unsupported"
            # A "citation" basis only counts if it points at a real source: a
            # References number in [1, n_refs] or a Further reading label
            # W1..W<len(further)>. Anything else is effectively unsupported (a
            # claim that names no source isn't grounded).
            raw_idx = c.get("citation_index")
            cite_idx: int | str | None = None
            web = (re.fullmatch(r"\[?\s*[Ww](\d+)\s*\]?", str(raw_idx).strip())
                   if raw_idx is not None else None)
            if web:
                if 1 <= int(web.group(1)) <= len(further):
                    cite_idx = f"W{int(web.group(1))}"
            else:
                try:
                    number = int(raw_idx)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    number = 0
                if 1 <= number <= n_refs:
                    cite_idx = number
            quote = " ".join(str(c.get("quote") or "").split())
            evidence = str(c.get("evidence") or "").strip()
            if basis == "citation" and cite_idx is None:
                basis = "unsupported"
            elif basis == "citation" and not _quote_in_source(quote, sources[str(cite_idx)][1]):
                # The model had the source's text; a citation it cannot quote
                # from that text is not one the source supports.
                basis = "unsupported"
                why = f"no quote from [{cite_idx}] given" if not quote else f"the quote is not in the text of [{cite_idx}]"
                evidence = f"{evidence} ({why})".strip()
            claims.append({
                "claim": str(c["claim"]).strip(),
                "basis": basis,
                "citation_index": cite_idx,
                "quote": quote,
                "evidence": evidence,
            })
        unsupported = [c["claim"] for c in claims if c["basis"] == "unsupported"]
        grounding = {
            "claims": claims,
            "summary": str(parsed.get("summary") or "").strip(),
            "total": len(claims),
            "grounded": len(claims) - len(unsupported),
            "unsupported": unsupported,
        }
        self._log.info(
            "[claim_check] %d/%d claims grounded (%d unsupported)",
            grounding["grounded"], grounding["total"], len(unsupported),
        )
        self._write_claims_ledger(grounding)
        return {"claim_grounding": grounding, "claim_check_failed": ""}

    async def _node_select_skills(self, state: QuestState) -> QuestState:
        """Pick which skills this quest carries.

        Replaces blanket injection. Every trusted skill used to be pushed into
        four prompts regardless of topic, which cost ~465 tokens per skill per
        node — so the library growing, which is the whole point, made itself
        the largest single token cost.

        Here one small call reads a catalogue (name, kind, a description and a
        scope limit each) and narrows it, once, against ~93,000 for the
        blanket version.

        The catalogue itself is layered before the call. With no domain filter
        — FI works in any field — every importable skill would otherwise be a
        candidate for every quest, which is the same accumulation problem one
        level up: ~28,000 tokens for a hundred-odd skills, growing with the
        library. So untagged skills (the general ones — statistics, units,
        figures) are always candidates, and domain-tagged ones are admitted
        only when the topic looks related. See ``core.skills.layers``.

        Never fatal: a quest must not fail because an additive step could not
        answer. Every failure path ends in "no skills", which is the behaviour
        that existed before skills.
        """
        from core.skills import loadable_skills
        from core.skills.layers import render_layer_report, select_layers
        from core.skills.selection import (
            build_catalogue, near_misses, parse_selection,
        )

        cfg = self.config.engine
        exclude = list(getattr(cfg, "skills_exclude", []) or [])
        requested = list(getattr(cfg, "skills", []) or [])
        required = list(dict.fromkeys(getattr(cfg, "skills_required", []) or []))

        def _force(
            chosen: list[str], reasons: dict[str, str], uses: dict[str, str],
        ) -> list[str]:
            """The pick plus every required skill, whatever the pick said."""
            out = list(chosen)
            for name in required:
                if name not in out:
                    out.append(name)
                reasons.setdefault(name, "required by engine.skills_required")
                uses.setdefault(name, "experiment")
            return out

        try:
            # An empty `skills` means "every trusted skill is a candidate" —
            # the user should not have to remember what they have taught it.
            # Required skills are always looked up, so narrowing the
            # candidates with `skills` cannot hide one.
            names = requested or [s.name for s in _discover_skill_names(self._skill_dirs)]
            names = list(dict.fromkeys([*names, *required]))
            # Self-tests are subprocesses that can take minutes on a cold
            # cache; in a thread, so the event loop (and the web server on
            # it) keeps answering while they run.
            usable, rejected = await asyncio.to_thread(
                loadable_skills, names, external_dirs=self._skill_dirs,
            )
        except Exception as e:  # noqa: BLE001 - the registry must never stall a quest
            if required:
                raise RuntimeError(
                    f"engine.skills_required names {required}, but the skill "
                    f"registry is unavailable ({e})"
                ) from e
            self._log.warning("[skills] registry unavailable (%s); none used", e)
            return {"selected_skills": [], "skill_selection": {"error": str(e)}}
        _raise_if_required_skills_unusable(required, usable, rejected)

        survey = bool(
            state.get("survey_mode_resolved") or state.get("no_simulation_resolved")
        )
        # Layer before cataloguing: the topic decides which domain-tagged
        # skills are even worth describing to the selector.
        layered, layer_report = select_layers(usable, str(state.get("topic") or ""))
        self._log.info(
            "[skills] %s",
            render_layer_report(layer_report).replace("\n", " | "),
        )
        catalogue = build_catalogue(layered, exclude=exclude, survey_mode=survey)

        # Tell the user about skills that would have been candidates but are
        # not approved — otherwise one can sit unapproved forever while every
        # quest quietly does without it.
        # Skills found in other agents' folders that nobody approved are not
        # near misses: nobody asked for them, and there can be hundreds. One line
        # says they exist; an approved one whose content then changed is still
        # reported by name below.
        never_asked = [
            st for st in rejected
            if st.skill.external and st.approved_hash is None
        ]
        for st in near_misses([s for s in rejected if s not in never_asked], [], catalogue):
            self._log.warning(
                "[skills] note: %r is not usable (%s) — %s",
                st.skill.name, st.status.value, st.reason,
            )
        if never_asked:
            folders = {st.skill.path.parent for st in never_asked}
            self._log.info(
                "[skills] found %d skill(s) in %d folder(s) of other agents; none is "
                "approved yet, so they are not candidates (`python launch.py --skills` "
                "lists them, `--approve-skill <name>` approves one; add `--config "
                "<this quest's YAML>` to see those in a folder its engine.skills_dirs "
                "names)",
                len(never_asked), len(folders),
            )

        if not catalogue:
            self._log.info(
                "[skills] no candidate skills%s", " (survey mode)" if survey else "",
            )
            if required:
                reasons: dict[str, str] = {}
                uses: dict[str, str] = {}
                forced = _force([], reasons, uses)
                self._log.info("[skills] required: %s", ", ".join(forced))
                return {"selected_skills": forced, "skill_selection": {
                    "candidates": 0, "chosen": forced, "reasons": reasons,
                    "uses": uses, "forced": required,
                }}
            return {"selected_skills": [], "skill_selection": {"candidates": 0}}

        prompt = self._prompts["select_skills"].substitute(
            topic=state["topic"],
            clarify_block=_format_clarify(state),
            chosen_idea=json.dumps(state.get("chosen_idea") or {}, indent=2),
            catalogue_block=catalogue.render(),
        )

        # Two attempts: the call is small, and a transient failure costing the
        # quest its skills is a poor trade for one retry.
        text = ""
        for attempt in (1, 2):
            try:
                text = await self._chat(prompt, node="select_skills")
                break
            except Exception as e:  # noqa: BLE001
                self._log.warning(
                    "[skills] selection call failed (attempt %d/2): %s", attempt, e,
                )
        else:
            self._log.warning("[skills] selection unavailable; none used")
            if required:
                reasons = {}
                uses = {}
                forced = _force([], reasons, uses)
                self._log.info("[skills] required: %s", ", ".join(forced))
                return {"selected_skills": forced, "skill_selection": {
                    "error": "call failed", "chosen": forced, "reasons": reasons,
                    "uses": uses, "forced": required,
                }}
            return {"selected_skills": [], "skill_selection": {"error": "call failed"}}

        sel = parse_selection(text, catalogue)
        for name in sel.unknown:
            self._log.warning(
                "[skills] selection named %r, which is not a candidate — ignored",
                name,
            )
        if required:
            sel.chosen = _force(sel.chosen, sel.reasons, sel.uses)
            self._log.info("[skills] required by engine.skills_required: %s", ", ".join(required))
        if sel.chosen:
            self._log.info(
                "[skills] selected %d of %d candidate(s): %s",
                len(sel.chosen), len(catalogue.entries), ", ".join(sel.chosen),
            )
            for name in sel.chosen:
                self._log.info(
                    "[skills]   %s (%s) — %s", name, sel.uses.get(name, "experiment"),
                    sel.reasons.get(name) or "(no reason given)",
                )
        else:
            self._log.info(
                "[skills] none of %d candidate(s) selected; generating instead",
                len(catalogue.entries),
            )

        record = sel.to_dict() | {
            "candidates": len(catalogue.entries),
            "layers": layer_report,
        }
        if required:
            record["forced"] = required
        return {"selected_skills": sel.chosen, "skill_selection": record}

    def _skills_block(self, state: QuestState | None = None) -> str:
        """Instructions for the skills this quest actually selected.

        This is the step that makes the registry worth having: a skill FI
        knows about but never tells ``design`` / ``implement`` about changes
        nothing.

        Renders **only what ``select_skills`` chose**, not everything trusted.
        Rendering the whole library cost ~465 tokens per skill per node across
        four nodes, so a growing library — the entire point of the subsystem —
        became its own largest cost.

        Only TRUSTED skills can be selected in the first place; anything else
        never reaches this point, and the quest falls back to generating the
        code itself, which is the behaviour that existed before skills.
        """
        usable, reasons = _resolve_selected_skills(
            state, self._log, use="experiment", external_dirs=self._skill_dirs,
        )
        if not usable:
            return ""

        # The preamble has to match what the skills actually are. "Call into
        # them instead of re-deriving the physics" is right for an importable
        # library and meaningless for a command-line tool, which has no
        # physics and cannot be imported — telling a model to import pandoc
        # teaches it to write code that cannot work.
        from core.skills import Kind

        kinds = {st.skill.kind for st in usable}
        lead = ["The following skills are available and trusted. Prefer them "
                "over working the same thing out yourself."]
        if Kind.LIBRARY in kinds:
            lead.append(
                "For a **library** skill, import it and call its functions "
                "rather than re-deriving what it computes — it is tested code, "
                "and re-deriving it is where wrong physics enters."
            )
        if Kind.TOOL in kinds:
            lead.append(
                "For a **tool** skill, drive the tool the way its instructions "
                "record — the invocations, flags and output-checking below are "
                "known to work. Do not guess flags, and do not reimplement what "
                "the tool already does."
            )
        lead.append(
            "Each skill below records why it was selected. If you judge one "
            "inapplicable after seeing the full design, you may leave it "
            "unused — say why in `method`."
        )
        parts = [" ".join(lead)]
        # Under the Docker sandbox the experiment cannot see a host path. Each
        # approved external skill is mounted read-only in the container, and
        # every path the model is given for it is the container's.
        plan = _skill_mount_plan(self.config, usable)
        for st in usable:
            skill = st.skill
            self._log.info(
                "[skills] loaded %s (%s, %s)",
                skill.name, skill.kind.value, skill.maturity.value,
            )
            mount = plan.mounts.get(skill.name) if plan is not None else None
            not_mounted = plan.refused.get(skill.name) if plan is not None else None
            parts.append(f"\n## Skill: {skill.name} ({skill.kind.value})\n")
            why = reasons.get(skill.name)
            if why:
                parts.append(f"*Selected because:* {why}\n")
            if mount is not None:
                parts.append(
                    f"*In the Docker sandbox this skill's folder is `{mount.container}`, "
                    f"mounted read-only: read or run what it ships from there, and "
                    f"write only under the working directory (the quest folder).*\n"
                )
            elif not_mounted:
                parts.append(
                    f"*This skill's folder is NOT available in the Docker sandbox "
                    f"({not_mounted}), so the experiment cannot read or run the "
                    f"files it ships. Use what is written here only.*\n"
                )
            instructions = skill.instructions().strip()
            surface = skill.api_surface().strip()
            if mount is not None:
                instructions, surface = mount.translate(instructions), mount.translate(surface)
            parts.append(instructions)
            if surface:
                parts.append(f"\n### {skill.name} — API surface\n")
                parts.append(surface)

            # Bundled files are named, never inlined. A references directory
            # can be larger than the whole quest, and inlining it would undo
            # the selection step this block exists to serve. Naming them is
            # enough: the generated code opens what it needs. The skill's
            # folder is not its working directory: every run of the experiment
            # has the quest folder as its cwd (`cwd=self.quest_root`; `/work`
            # in Docker), so a file a script writes without a path lands beside
            # `paper.md`, and the listed paths are relative to the skill's
            # folder, not to the cwd. The block says so, gives the folder in
            # full, and says to launch a script with `sys.executable`: the
            # experiment runs on FI's interpreter (`SharedInterpreterExecutor`)
            # or the container's, not necessarily the first `python` on PATH,
            # which is what a skill's own text says. One wording fits both
            # sandboxes.
            bundled = skill.bundled_scripts()
            refs = skill.reference_files()
            if bundled or refs:
                parts.append(f"\n### {skill.name} — bundled files\n")
                if not_mounted:
                    parts.append(
                        "These ship with the skill but are not reachable from the "
                        "sandbox, so do not write code that opens or runs them:"
                    )
                else:
                    base = mount.container if mount is not None else skill.path
                    parts.append(
                        f"These ship with the skill, at paths relative to "
                        f"`{base}`. Read or run them as needed; their "
                        f"contents are deliberately not reproduced here."
                    )
                    in_docker = " (`/work` in the Docker sandbox)" if plan is not None else ""
                    parts.append(
                        f"The experiment's working directory is the quest folder"
                        f"{in_docker}, not the skill's folder: a file a script "
                        f"writes without a path lands in the quest folder, and a "
                        f"relative path will not find these files, so give each "
                        f"one in full (`{base}` plus the path listed). To run one "
                        f"of these scripts as a subprocess, launch it with "
                        f"`sys.executable` (the Python running the experiment), "
                        f"not the bare command `python`, even where the skill's "
                        f"text above writes `python`."
                    )
                for rel in bundled:
                    parts.append(f"- `{rel}` (executable)")
                for rel in refs:
                    parts.append(f"- `{rel}` (reference)")
        return "\n".join(parts).strip()

    def _mount_selected_skills(self, state: Any, *, record: bool = True) -> None:
        """Docker sandbox only: mount the folders of the approved external skills
        this quest selected for its experiment, read-only, into every container
        from here on. Only those: an approved skill that was not selected, an
        unapproved one, and FI's own skills are not mounted, and a folder that is
        a link or leads outside its skills folder is refused with the reason in
        the log. The names are written to ``.fi/skill_mounts.json`` so a
        ``--watch`` (which has no quest state) mounts the same ones."""
        setter = getattr(self.executor, "set_skill_mounts", None)
        if self.config.execution.sandbox != "docker" or setter is None:
            return
        usable, _ = _resolve_selected_skills(
            state, self._log, use="experiment", external_dirs=self._skill_dirs,
        )
        plan = _skill_mount_plan(self.config, usable)
        setter(list(plan.mounts.values()))
        found_at = {st.skill.name: st.skill.path for st in usable}
        for name, why in plan.refused.items():
            self._log.warning(
                "[skills] %s is NOT mounted into the Docker sandbox: %s (%s)",
                name, why, found_at.get(name),
            )
        for m in plan.mounts.values():
            self._log.info(
                "[skills] mounted read-only in the Docker sandbox: %s -> %s (%s)",
                m.name, m.container, m.host,
            )
        if not record:
            return
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / _SKILL_MOUNTS_FILE).write_text(
                json.dumps({"skills": list(plan.mounts)}), encoding="utf-8",
            )
        except OSError as exc:
            self._log.debug("[skills] could not record the mounted skills: %s", exc)

    def _skills_summary_block(self, state: QuestState | None = None) -> str:
        """The selected skills as ``design`` needs them: what each is for,
        where it does not apply, and why it was selected.

        Design decides which skill a method rests on; the implementation
        stages, which write the calls, receive the full instructions and API
        from :meth:`_skills_block`. On the SIR validation quest the full text
        of five skills was 98,147 of design's 146,904 prompt characters.
        """
        usable, reasons = _resolve_selected_skills(
            state, self._log, use="experiment", external_dirs=self._skill_dirs,
        )
        if not usable:
            return ""
        from core.skills.selection import describe, scope_limit

        parts = [
            "The following skills are available and trusted. Build the "
            "experiment on them where they fit: a **library** skill is tested "
            "code the experiment imports and calls, a **tool** skill is "
            "external software FI drives. The implementation step receives "
            "each one's full instructions and API. If one does not fit the "
            "design, leave it unused and say why in `method`."
        ]
        for st in usable:
            skill = st.skill
            parts.append(
                f"\n- **{skill.name}** ({skill.kind.value}): "
                f"{describe(skill) or '(no description)'}"
            )
            limit = scope_limit(skill)
            if limit:
                parts.append(f"  NOT for: {limit}")
            why = reasons.get(skill.name)
            if why:
                parts.append(f"  Selected because: {why}")
        return "\n".join(parts).strip()

    def _writing_skills_block(self, state: QuestState | None = None) -> str:
        """The instructions of the skills selected for writing, for ``write``.

        A writing skill guides how the paper is written — structure, style,
        diagrams in the text — so it goes to the writer and nowhere else. On
        the SIR validation quest two were selected to draft and format the
        paper, then sent to design and the implement stages five times, where
        no paper is written, while the writer never saw them. Only the
        instructions go: the writer produces text and runs nothing, so an API
        surface or bundled script is no use to it.
        """
        usable, reasons = _resolve_selected_skills(
            state, self._log, use="writing", external_dirs=self._skill_dirs,
        )
        if not usable:
            return ""
        parts = [
            "Guidance selected for writing this paper. Follow it where it "
            "applies; the rules above on honesty, citations and the output "
            "format take precedence over it."
        ]
        for st in usable:
            skill = st.skill
            self._log.info("[skills] writing guidance: %s", skill.name)
            parts.append(f"\n## Writing skill: {skill.name}\n")
            why = reasons.get(skill.name)
            if why:
                parts.append(f"*Selected because:* {why}\n")
            parts.append(skill.instructions().strip())
        return "\n".join(parts).strip()

    def _skill_assertions(self, state: QuestState | None = None) -> list[dict[str, Any]]:
        """Range assertions contributed by the trusted skills.

        A skill knows its own valid domain, so a design that calls one need
        not restate it. These are merged with the design's own
        ``result_assertions`` by ``_assertion_violations``.
        """
        names = list((state or {}).get("selected_skills") or [])
        if not names:
            return []
        try:
            from core.skills import loadable_skills

            usable, _ = loadable_skills(names, external_dirs=self._skill_dirs)
        except Exception:  # noqa: BLE001
            return []
        out: list[dict[str, Any]] = []
        for st in usable:
            out.extend(st.skill.assertions())
        return out

    def _numeric_oracle_hits(
        self, paper_md: str, state: QuestState,
    ) -> list[str]:
        """Deterministic number check; returns one advisory string per finding.

        Runs on the FULL paper text, not the 16 000-character slice the
        review prompt gets — a wrong number in a late results table is
        exactly the kind this is here to catch.

        Writes ``paper/numeric_audit.json`` either way, so a clean run
        leaves evidence that the check ran rather than silence that could
        equally mean it was skipped. Best-effort throughout: an oracle
        failure must never be quest-fatal, because it would block a paper
        over a bug in the checker rather than a bug in the paper.
        """
        try:
            from core import numeric_oracle

            # A paper reports the mean over the seeds and its interval, which
            # seed 0's RESULT_JSON does not hold.
            intervals = _replicate_result_intervals(state)
            results = {**(state.get("result_json") or {})}
            if intervals:
                results["mean_over_seeds"] = intervals
            # The settings the run was given (the topic's grid, the design's
            # variables and method): a paper that prints ``R0 = 1.5`` is
            # quoting its setup, not a result that happens to sit near one.
            report = numeric_oracle.check(
                paper_md, results,
                declared=numeric_oracle.declared_numbers(
                    state.get("design"), state.get("topic"),
                ),
            )
        except Exception as e:  # noqa: BLE001 - never fail a quest over the checker
            self._log.warning("[numeric_oracle] check failed (%s); skipping", e)
            return []

        try:
            paper_dir = self.quest_root / "paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "numeric_audit.json").write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            self._log.warning("[numeric_oracle] could not write audit: %s", e)

        if report.skipped:
            self._log.info("[numeric_oracle] skipped — %s", report.skip_reason)
            return []
        if report.ok:
            self._log.info(
                "[numeric_oracle] %d paper numbers vs %d results: all consistent",
                report.paper_numbers, report.result_numbers,
            )
            return []

        for f in report.findings:
            self._log.warning("[numeric_oracle] %s: %s", f.kind, f.describe())
        # One hit per contradicted number, each naming the value and the
        # result path, so the rewrite has something specific to act on
        # instead of "a number is wrong somewhere".
        #
        # Labelled by the finding's OWN kind — ``transposed``,
        # ``near_miss``, ``trivial_reference`` — rather than one blanket
        # ``unverified_number:`` prefix over all of them. That prefix
        # argues with the trivial-reference finding it is pasted onto:
        # "unverified_number: `deterministic_final_size` is exactly 0 at
        # every one of its settings" describes a number that was in fact
        # computed and exported correctly. What is unverified there is not
        # the number but the bracket the root finder was handed, which is
        # exactly what the finding's own text says. A label that
        # contradicts the sentence under it makes the reader decide which
        # half to believe, and the label is the half they read first.
        return [
            f"{f.kind}: {f.describe()}" for f in report.findings
        ]

    def _statistics_claim_hits(
        self, paper_md: str, state: QuestState,
    ) -> list[str]:
        """Statistics the paper describes as something the run did not compute.

        The numeric oracle asks whether a number MATCHES a result. This asks
        whether it is the QUANTITY the paper says it is — a Bonferroni
        threshold printed as a p-value, an effect-size claim the run's own
        comparisons contradict, a t-interval labelled "exact binomial", a
        figure's axis limit printed as a bin count, a major-outbreak
        probability called the chance the disease vanishes.

        Forced, unlike the numeric oracle: every finding is a recomputation
        against this run's own replicates that can NAME the contradiction,
        rather than a regex hoping a number means what it looks like. The hit
        is text-only (``_TEXT_ONLY_HITS``) because the experiment computed the
        right numbers and the paper described them wrongly — so the route is
        ``rewrite`` and a mislabel never sends the experiment back.

        Writes ``paper/statistics_audit.json`` either way, so a clean run
        leaves evidence the check ran. Best-effort throughout: a bug in the
        checker must never block a paper.
        """
        try:
            from core import stat_claims

            replicates = list(state.get("result_json_replicates") or [])
            comparison_stats: dict[str, Any] = {}
            if len(replicates) >= stat_claims.MIN_SEEDS:
                # Uncapped, unlike the analyze aggregate: "for ALL pairwise
                # comparisons" is a claim about every comparison, and the
                # cap of 24 hid 12 of the 36 one real quest made.
                comparison_stats = _result_comparison_stats(
                    replicates,
                    max_effect_sizes=10**9,
                    assertions=_replicate_assertions(state),
                )
            report = stat_claims.check(
                paper_md,
                result_json=state.get("result_json") or {},
                intervals=_replicate_result_intervals(state),
                comparison_stats=comparison_stats,
                figure_records=state.get("figure_records") or {},
                n_seeds=len(replicates),
            )
        except Exception as e:  # noqa: BLE001 - never fail a quest over the checker
            self._log.warning("[stat_claims] check failed (%s); skipping", e)
            return []

        try:
            paper_dir = self.quest_root / "paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "statistics_audit.json").write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            self._log.warning("[stat_claims] could not write audit: %s", e)

        if report.skipped:
            self._log.info("[stat_claims] skipped — %s", report.skip_reason)
            return []
        if report.ok:
            self._log.info(
                "[stat_claims] every statistic the paper names matches what the "
                "run computed",
            )
            return []

        for f in report.findings:
            self._log.warning("[stat_claims] %s", f.describe())
        return [f"mislabelled_statistic: {f.describe()}" for f in report.findings]

    def _number_provenance_hits(
        self, paper_md: str, state: QuestState,
    ) -> list[str]:
        """Numbers the paper prints that nothing in this run accounts for.

        The numeric oracle asks whether a number is NEAR a result and ignores
        one that is far from every result; this asks whether the number came
        from anywhere at all. A graded paper's Table 1 printed eight cells
        matching no value the run computed, no replicate and none of the
        three-seed aggregates, and every existing check passed it.

        The aggregates are the reason this is possible rather than a machine
        for flagging correct papers: a paper reports the MEAN over the seeds
        and its interval, and seed 0's ``RESULT_JSON`` holds neither. Measured
        on that paper, checking against ``RESULT_JSON`` alone makes all eleven
        of its *correct* headline numbers untraceable too.

        Forced, and text-only (``_TEXT_ONLY_HITS``): the experiment ran and
        recorded its results, so the route is ``rewrite`` and an unsourced
        number never sends the experiment back to be run again.

        Writes ``paper/provenance_audit.json`` either way, so a clean run
        leaves evidence the check ran. Best-effort throughout: a bug in the
        checker must never block a paper.
        """
        try:
            from core import number_provenance

            replicates = list(state.get("result_json_replicates") or [])
            assertions = _replicate_assertions(state)
            comparison_stats: dict[str, Any] = {}
            aggregate: dict[str, Any] = {}
            if replicates:
                aggregate = _aggregate_result_json_replicates(
                    replicates, assertions=assertions,
                )
            if len(replicates) >= 2:
                comparison_stats = _result_comparison_stats(
                    replicates, max_effect_sizes=10**9, assertions=assertions,
                )
            try:
                config_dump: Any = self.config.model_dump(mode="json")
            except Exception:  # noqa: BLE001 - the config is a convenience here
                config_dump = None
            report = number_provenance.check(
                paper_md,
                result_json=state.get("result_json") or {},
                replicates=replicates,
                intervals=_replicate_result_intervals(state),
                aggregate=aggregate,
                figure_records=state.get("figure_records") or {},
                comparison_stats=comparison_stats,
                config=config_dump,
                design=state.get("design") or {},
                n_seeds=len(replicates),
            )
        except Exception as e:  # noqa: BLE001 - never fail a quest over the checker
            self._log.warning("[number_provenance] check failed (%s); skipping", e)
            return []

        try:
            paper_dir = self.quest_root / "paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "provenance_audit.json").write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            self._log.warning("[number_provenance] could not write audit: %s", e)

        if report.skipped:
            self._log.info("[number_provenance] skipped — %s", report.skip_reason)
            return []
        if report.ok:
            self._log.info(
                "[number_provenance] all %d paper numbers trace to this run "
                "(%d values it can account for)",
                report.paper_numbers, report.traceable_values,
            )
            return []

        for f in report.findings:
            self._log.warning("[number_provenance] %s", f.describe())
        return [f"unsourced_number: {f.describe()}" for f in report.findings]

    def _figure_caption_hits(self, paper_md: str, state: QuestState) -> list[str]:
        """Captions naming a series their figure draws flat or not at all.
        The review prompt carries the same list; this logs it and keeps it for
        the human review."""
        findings = _figure_caption_findings(paper_md, state.get("figure_records") or {})
        for finding in findings:
            self._log.warning("[figure_check] %s", finding)
        return findings

    def _write_claims_ledger(self, grounding: dict[str, Any]) -> None:
        """Persist the claim-grounding result as a transparency ledger:
        ``paper/claims.json`` (structured) + ``paper/CLAIMS.md`` (readable).
        Best-effort; a write failure is logged, never quest-fatal."""
        paper_dir = self.quest_root / "paper"
        try:
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "claims.json").write_text(
                json.dumps(grounding, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            lines = [
                "# Claim grounding",
                "",
                f"**{grounding['grounded']} of {grounding['total']}** substantive "
                "claims trace to evidence "
                f"({len(grounding['unsupported'])} unsupported).",
                "",
                grounding.get("summary", ""),
                "",
                "## Claims",
            ]
            for c in grounding["claims"]:
                tag = (f"cite [{c['citation_index']}]"
                       if c["basis"] == "citation" else c["basis"])
                lines.append(f"- **[{tag}]** {c['claim']}"
                             + (f" — {c['evidence']}" if c["evidence"] else "")
                             + (f" — quoted: \"{c['quote']}\"" if c.get("quote") else ""))
            (paper_dir / "CLAIMS.md").write_text(
                "\n".join(lines) + "\n", encoding="utf-8",
            )
        except OSError as e:
            self._log.warning("[claim_check] ledger write failed: %r", e)

    async def _measure_draft_pages(self, paper_path: Path, state: QuestState) -> dict[str, Any] | None:
        """Render the draft the way the final ``paper.pdf`` is rendered and
        measure it: ``{"pages", "words", "last_page_lines", "last_page_empty"}``.

        The render is ``PaperGenerator._compile_pdf``, so the template, the
        pandoc flags and the page-limit layout are the final paper's, in the
        venue the final render uses (the clarify ``paper_venue`` when the
        config keeps ``generic``). It runs in ``.fi/page_check/`` with the
        quest's figures copied beside the source; nothing is written to the
        quest folder itself. Returns ``None``, with one warning per quest, when
        the pages cannot be counted: the paper renders through HTML
        (``paper_style: briefing`` with a browser present), the LaTeX render
        fails or has no engine (the HTML fallback is off for this render), or
        the PDF cannot be read."""
        from generation._html_pdf import find_html_browser
        from generation._pdf_measure import _empty_share, measure_pdf, paper_report
        from generation.paper import PaperGenerator

        output = self.config.output
        if output.paper_style == "briefing" and find_html_browser() is not None:
            return self._page_check_skipped(
                "paper_style is briefing, which renders through HTML, not the LaTeX templates"
            )
        fmt = output.paper_format
        answers = state.get("clarify_answers")
        venue = answers.get("paper_venue") if isinstance(answers, dict) else None
        if fmt == "generic" and venue in SCIENTIFIC_PAPER_FORMATS | NON_SCIENTIFIC_PAPER_FORMATS:
            fmt = venue
        config = self.config.model_copy(update={"output": output.model_copy(update={
            "html_pdf_fallback": False, "paper_style": "latex", "paper_format": fmt,
        })})
        scratch = self.fi_dir / "page_check"
        quest_figures = self.quest_root / "figures"

        def render() -> tuple[dict[str, Any] | None, str]:
            shutil.rmtree(scratch, ignore_errors=True)
            scratch.mkdir(parents=True, exist_ok=True)
            if quest_figures.is_dir():
                shutil.copytree(quest_figures, scratch / "figures")
            pdf, skip = PaperGenerator(config)._compile_pdf(Path(paper_path), scratch)
            if pdf is None:
                return None, skip.summary if skip is not None else "the render produced no PDF"
            doc = measure_pdf(pdf)
            if doc is None or not doc.pages:
                return None, "the rendered PDF could not be read"
            metrics = paper_report(doc)["metrics"]
            return {
                "pages": int(metrics["pages"]),
                "words": metrics["words"],
                "last_page_lines": metrics["last_page_lines"],
                "last_page_empty": round(_empty_share(doc.pages[-1]), 3),
            }, ""

        try:
            measured, why = await asyncio.to_thread(render)
        except Exception as e:  # noqa: BLE001 — a page count never stops the review
            measured, why = None, repr(e)
        if measured is None:
            return self._page_check_skipped(why)
        return measured

    def _page_check_skipped(self, why: str) -> None:
        """Log, once per quest, that the page limit is not checked and why."""
        if not getattr(self, "_page_check_warned", False):
            self._page_check_warned = True
            self._log.warning(
                "[page_limit] the draft's pages are not counted, so the page limit is not checked: %s", why,
            )
        return None

    async def _drop_further_reading_to_fit(
        self, state: QuestState, paper_path: Path, limit: int,
    ) -> tuple[dict[str, Any], list[str], int] | None:
        """Bring a draft that is over the page limit within it by dropping
        entries from the end of its Further reading, when that alone does it.

        FI writes that list itself and sets it after the body, so on a paper
        cut to the limit it is what spills onto one page more: a real quest
        (limit 4) ended on 5 pages, page 5 holding two entries of it, and the
        whole-paper rewrites that followed changed a correct citation and added
        unsupported ones. The body is not touched here and no model is called.

        First the draft is rendered with as many entries removed as may go (an
        entry the text cites is never one of them, since the citation would then
        name nothing); if it is still over, the body is what is too long and
        nothing is changed. Otherwise one more entry is put back at a time,
        rendering each time, and the first draft that fits is written over
        ``paper_path``. That is one render per entry at most, each the render
        the page check makes. Returns the measurement of the draft written, the
        labels dropped and how many entries there were, or ``None`` when
        nothing was dropped."""
        try:
            text = paper_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        block = _further_reading_block(text)
        if block is None:
            return None
        cited = _web_labels_cited(text)
        droppable = 0
        for label in reversed(block.labels):
            if label in cited:
                break
            droppable += 1
        if not droppable:
            return None
        n_entries = len(block.labels)
        trial = self.fi_dir / "page_check_trial.md"
        self.fi_dir.mkdir(parents=True, exist_ok=True)

        async def render(keep: int) -> dict[str, Any] | None:
            trial.write_text(_trim_further_reading(text, keep), encoding="utf-8")
            return await self._measure_draft_pages(trial, state)

        try:
            fewest = n_entries - droppable
            found = await render(fewest)
            if found is None or int(found["pages"]) > limit:
                self._log.info(
                    "[page_limit] the draft is still over the limit of %d with its Further reading "
                    "%s, so the body is what is too long; the list is left as written",
                    limit, "removed" if not fewest else f"cut to its first {fewest} entries",
                )
                return None
            keep = fewest
            for more in range(n_entries - 1, fewest, -1):
                measured = await render(more)
                if measured is None:
                    return None
                if int(measured["pages"]) <= limit:
                    keep, found = more, measured
                    break
        finally:
            trial.unlink(missing_ok=True)
        paper_path.write_text(_trim_further_reading(text, keep), encoding="utf-8")
        return found, block.labels[keep:], n_entries

    async def _page_limit_review(
        self, state: QuestState, paper_path: str | Path | None,
    ) -> tuple[list[str], dict[str, Any] | None]:
        """The page-limit part of the review: the forced hit (a list of at most
        one) and the record for ``review["page_limit"]``. Without a page limit
        nothing is rendered and both are empty, as they are when the draft
        cannot be measured. A draft over the limit only because of its Further
        reading is not sent back: the list's last entries are dropped from
        ``paper_path`` until it fits (:meth:`_drop_further_reading_to_fit`), and
        the record names them. Any other draft over the limit is forced while
        fewer than ``_PAGE_LIMIT_REWRITES`` shortening rewrites have been made;
        after that the overrun is only recorded."""
        limit = resolve_page_limit(self.config)
        if limit is None or not paper_path or not Path(paper_path).is_file():
            return [], None
        measured = await self._measure_draft_pages(Path(paper_path), state)
        if measured is None:
            return [], None
        pages = int(measured["pages"])
        done = int(state.get("page_limit_rewrites") or 0)
        record: dict[str, Any] = {"pages": pages, "limit": limit, "rewrites": done}
        if pages > limit:
            # The engine's own Further reading may be all that is over. Dropping
            # its last entries costs no model call and no rewrite of the body.
            fitted = await self._drop_further_reading_to_fit(state, Path(paper_path), limit)
            if fitted is not None:
                before, (measured, dropped, n_entries) = pages, fitted
                pages = int(measured["pages"])
                record["pages"] = pages
                record["further_reading_dropped"] = dropped
                self._log.warning(
                    "[page_limit] the draft renders to %d pages, over the limit of %d, and only its "
                    "Further reading was over: dropped %d of its %d entries from the end (%s); it now "
                    "renders to %d pages",
                    before, limit, len(dropped), n_entries, ", ".join(f"[{w}]" for w in dropped), pages,
                )
        if pages <= limit:
            self._log.info("[page_limit] the draft renders to %d pages; the limit is %d", pages, limit)
            return [], record
        if done >= _PAGE_LIMIT_REWRITES:
            record["exceeded_after_rewrites"] = True
            self._log.warning(
                "[page_limit] the draft renders to %d pages, over the limit of %d, after %d shortening "
                "rewrites; recorded in the review, not forced again", pages, limit, done,
            )
            return [], record
        words = _words_to_cut(pages, limit, float(measured.get("last_page_empty") or 0.0))
        record["words_to_cut"] = words
        hit = _page_limit_hit(pages, limit, words)
        self._log.warning("[page_limit] %s (shortening %d of %d)", hit, done + 1, _PAGE_LIMIT_REWRITES)
        return [hit], record

    async def _node_review(self, state: QuestState) -> QuestState:
        """Single-reviewer (default) OR panel-mode review.

        When ``engine.review_panel`` is empty, behave exactly as before:
        one LLM call, one verdict.

        When non-empty, fire N parallel persona-prefixed reviews
        (`asyncio.gather`), aggregate the results deterministically,
        and call a moderator LLM for the prose `rationale`. Each
        persona's per-call model is resolvable via
        ``provider.node_models["review_panel.<name>"]``.
        """
        self._log.info("[review] judging paper")
        paper_path = state.get("paper_md")
        # With a page limit, this draft is rendered the way paper.pdf will be
        # and its pages counted, for both review paths below. Without one,
        # nothing is rendered.
        page_hits, page_record = await self._page_limit_review(state, paper_path)
        # Read after that: it may have dropped Further reading entries from the
        # file, and the review reads the paper as it now is.
        paper_md = ""
        if paper_path:
            try:
                paper_md = Path(paper_path).read_text(encoding="utf-8")
            except OSError:
                paper_md = ""
        base_prompt = self._prompts["review"].substitute(
            topic=state["topic"],
            clarify_block=_format_clarify(state),
            design_block=json.dumps(state.get("design") or {}, indent=2),
            analysis_block=json.dumps(state.get("analysis") or {}, indent=2),
            claim_grounding_block=_format_claim_grounding(state),
            figure_check_block=_format_figure_check(paper_md, state),
            # Advisory, unlike the figure check: what the reviewer may ask for,
            # never a must-flag hit. Empty when the paper cites every foundational
            # work it has, or has none.
            foundational_check_block=_foundational_review_block(
                state.get("literature") or [], paper_md, self.config.output.audience,
            ),
            # The whole paper. A 16 KB cut hid the second half of a real
            # 34,910-character paper, so the review graded a draft it had not
            # read; the cap now only guards against a runaway file.
            paper_md=_paper_for_prompt(paper_md, "review", self._log),
        )

        panel_names = list(self.config.engine.review_panel or [])
        if not panel_names:
            # Legacy single-reviewer path. Review runs AFTER the paper is
            # written, and the output generators (pdf/slides/poster/speech) run
            # only after run() returns — so a transient provider failure here
            # must NOT abort the quest and forfeit the finished paper + its
            # outputs. Fail open to "accept" (the same degrade the parse-miss
            # path below already uses, and the pattern applied to claim_check).
            try:
                text = await self._chat(base_prompt, node="review")
            except Exception as e:
                self._log.warning(
                    "[review] review call failed (%s); accepting the paper "
                    "as-is so its outputs still render", e,
                )
                text = ""
            review = _parse_json_lenient(text) or {
                "verdict": "accept", "score": 3, "suggestions": [],
            }
            mfh = review.get("must_flag_hits") or []
            if not isinstance(mfh, list):
                mfh = []
            review["must_flag_hits"] = _without_reviewer_page_limit_hits(
                [str(h).strip() for h in mfh if str(h).strip()], self._log,
            )
            # Arithmetic, not judgement: compare the paper's numbers against
            # the ones the run actually produced. Every other gate here ends
            # in a model reading text, so a mis-transcription (2.14 computed,
            # 2.41 written) survives all of them.
            #
            # Reported, NOT forced. These used to ride must_flag_hits onto the
            # non-bypassable revise path, and a real quest showed the cost of
            # that when the finding is wrong: a false positive (a DOI prefix
            # read as a measurement; a mantissa read without its exponent)
            # triggered a full re-design, a re-implement, twelve further
            # experiment runs, and a rewrite that left the paper WORSE than
            # the draft it replaced -- 11 of 11 claims grounded became 9 of 11.
            # The parser bugs behind that instance are fixed, but the exposure
            # is structural: this check is a regex over prose, and prose keeps
            # inventing new ways to write a number. A wrong number in the
            # paper is bad; silently burning the iteration budget on a
            # correct one is worse, so the findings are surfaced to the human
            # (log, paper/numeric_audit.json, and the human-review panel in all
            # three interfaces) and the verdict is left to the reviewer.
            numeric_warnings = self._numeric_oracle_hits(paper_md, state)
            if numeric_warnings:
                review["numeric_oracle_warnings"] = numeric_warnings
            figure_warnings = self._figure_caption_hits(paper_md, state)
            if figure_warnings:
                review["figure_caption_warnings"] = figure_warnings
            # Forced, unlike the advisory number check above: a statistic the
            # paper mislabels is caught by recomputing it, so the finding can
            # name the contradiction instead of guessing at one.
            review["must_flag_hits"] += self._statistics_claim_hits(paper_md, state)
            # Forced too, and for the same reason: a number that matches
            # nothing the run computed, nothing it configured and nothing it
            # drew is a set membership test, not a reading of prose. It goes
            # quiet on a run that recorded no results.
            review["must_flag_hits"] += self._number_provenance_hits(paper_md, state)
            # A figure the design planned and the run drew that the paper
            # leaves out is a set difference, not a reading of prose, so it
            # cannot misfire the way the number check can: it is forced.
            missing_figures = _missing_planned_figures(paper_md, state)
            for hit in missing_figures:
                self._log.warning("[figure_check] %s", hit)
            review["must_flag_hits"] += missing_figures
            # Forced as well: nothing checked this draft's citations.
            review["must_flag_hits"] += _citations_unchecked(state)
            # And a draft over the page limit.
            review["must_flag_hits"] += page_hits
            if page_record is not None:
                review["page_limit"] = page_record
            update: QuestState = {"review": review}
            if page_hits:
                update["page_limit_rewrites"] = int(state.get("page_limit_rewrites") or 0) + 1
            # Counted here, like the shortening rewrites above, so the router
            # can cap it: a must-flag about something the run computed sends
            # the experiment back, at most ``_CODE_REEXECUTES`` times.
            if _review_sends_the_experiment_back(review, state):
                update["code_reexecutes"] = int(state.get("code_reexecutes") or 0) + 1
            # Iteration is consumed when EITHER the verdict says revise
            # OR the must-flag hits force one. Bumping on must_flag_hits
            # alone (even with verdict=accept) makes ``_route_after_review``'s
            # non-bypassable revise path deterministic with respect to the
            # ``max_iterations`` budget — without this, a malformed
            # ``revise`` route from must-flag wouldn't have consumed the
            # iteration and the loop could run unbounded. A hit over the page
            # limit consumes none: its rewrites have their own counter, capped
            # at ``_PAGE_LIMIT_REWRITES``.
            other_hits = [h for h in review["must_flag_hits"] if _hit_name(h) != _PAGE_LIMIT_HIT]
            if review.get("verdict") == "revise" or other_hits:
                update["iteration"] = state.get("iteration", 0) + 1
                self._log.info(
                    "[review] verdict=%s must_flag_hits=%s -> iteration %d",
                    review.get("verdict"), review["must_flag_hits"],
                    update["iteration"],
                )
            else:
                self._log.info(
                    "[review] verdict=%s score=%s",
                    review.get("verdict"), review.get("score"),
                )
            return update

        # Panel path. Fire each persona in parallel; aggregate.
        self._log.info("[review] panel mode: %s", panel_names)

        async def run_persona(name: str) -> dict[str, Any]:
            try:
                prefix = _load_persona_prefix(name)
            except ValueError as e:
                self._log.warning("[review] %s; skipping", e)
                return {"persona": name, "verdict": "accept", "score": 3,
                        "strengths": [], "weaknesses": [],
                        "suggestions": [], "blocking": "",
                        "error": str(e)}
            prompt = f"{prefix}\n\n{base_prompt}"
            try:
                text = await self._chat(prompt, node=f"review_panel.{name}")
            except Exception as e:
                self._log.warning(
                    "[review] panelist %s failed (%s); recording a neutral "
                    "accept for this persona", name, e,
                )
                text = ""
            parsed = _parse_json_lenient(text) or {}
            mfh = parsed.get("must_flag_hits") or []
            if not isinstance(mfh, list):
                mfh = []
            return {
                "persona": name,
                "verdict": parsed.get("verdict") or "accept",
                "score": parsed.get("score") if isinstance(parsed.get("score"), (int, float)) else 3,
                "strengths": parsed.get("strengths") or [],
                "weaknesses": parsed.get("weaknesses") or [],
                "suggestions": parsed.get("suggestions") or [],
                "blocking": parsed.get("blocking") or "",
                "must_flag_hits": [str(h).strip() for h in mfh if str(h).strip()],
            }

        # return_exceptions=True is defense in depth: run_persona already
        # degrades a failed persona to a neutral accept, but a panelist must
        # never be able to abort the whole (post-write) review and forfeit the
        # paper's outputs. Drop any unexpected raise, propagate genuine
        # cancellation, and if EVERY panelist somehow failed, accept as-is.
        panel_results_raw = await asyncio.gather(
            *(run_persona(n) for n in panel_names),
            return_exceptions=True,
        )
        panel_results: list[dict[str, Any]] = []
        for r in panel_results_raw:
            if isinstance(r, asyncio.CancelledError):
                raise r
            if isinstance(r, BaseException):
                self._log.warning("[review] a panelist raised unexpectedly: %r", r)
            elif isinstance(r, dict):
                panel_results.append(r)
        if not panel_results:
            self._log.warning(
                "[review] all panelists failed; accepting the paper as-is",
            )
            panel_results = [{
                "persona": panel_names[0], "verdict": "accept", "score": 3,
                "strengths": [], "weaknesses": [], "suggestions": [],
                "blocking": "", "must_flag_hits": [],
            }]
        agg = _aggregate_panel_reviews(list(panel_results))

        # Moderator call — best effort for the rationale + suggestion
        # attribution prose. Numeric fields are taken from `agg`.
        panel_block = json.dumps(panel_results, indent=2)
        try:
            mod_prompt = self._prompts["review_moderate"].substitute(
                topic=state["topic"],
                panel_block=panel_block,
            )
            mod_text = await self._chat(mod_prompt, node="review_moderator")
            mod_parsed = _parse_json_lenient(mod_text) or {}
        except Exception as e:
            self._log.warning("[review] moderator call failed: %s", e)
            mod_parsed = {}

        # Merge: numeric/voting fields from the deterministic aggregator
        # always win; prose fields prefer the moderator's version when
        # present.
        review: dict[str, Any] = {**agg}
        review["must_flag_hits"] = _without_reviewer_page_limit_hits(
            review.get("must_flag_hits") or [], self._log,
        )
        rationale = mod_parsed.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            review["rationale"] = rationale.strip()
        # Use the moderator's suggestions if they're well-formed and
        # carry the persona-attribution prefix; otherwise keep agg's.
        mod_suggs = mod_parsed.get("suggestions")
        if isinstance(mod_suggs, list) and mod_suggs:
            review["suggestions"] = [str(s) for s in mod_suggs]
        # The same advisory arithmetic check as the single-reviewer path, and
        # kept out of must_flag_hits for the same reason: the iteration bump
        # below reads only must_flag_hits, so a finding costs no budget.
        numeric_warnings = self._numeric_oracle_hits(paper_md, state)
        if numeric_warnings:
            review["numeric_oracle_warnings"] = numeric_warnings
        figure_warnings = self._figure_caption_hits(paper_md, state)
        if figure_warnings:
            review["figure_caption_warnings"] = figure_warnings
        # Forced, as on the single-reviewer path: a recomputed contradiction,
        # not a pattern match over prose.
        stat_hits = self._statistics_claim_hits(paper_md, state)
        if stat_hits:
            review["must_flag_hits"] = [*(review.get("must_flag_hits") or []), *stat_hits]
        # Forced on this path as well: panel mode once skipped the numeric
        # oracle entirely, so asking for more reviewers meant fewer checks.
        provenance_hits = self._number_provenance_hits(paper_md, state)
        if provenance_hits:
            review["must_flag_hits"] = [
                *(review.get("must_flag_hits") or []), *provenance_hits,
            ]
        # Forced, as on the single-reviewer path.
        missing_figures = _missing_planned_figures(paper_md, state)
        for hit in missing_figures:
            self._log.warning("[figure_check] %s", hit)
        if missing_figures:
            review["must_flag_hits"] = [*(review.get("must_flag_hits") or []), *missing_figures]
        unchecked = _citations_unchecked(state)
        if unchecked:
            review["must_flag_hits"] = [*(review.get("must_flag_hits") or []), *unchecked]
        # And a draft over the page limit, as on the single-reviewer path.
        if page_hits:
            review["must_flag_hits"] = [*(review.get("must_flag_hits") or []), *page_hits]
        if page_record is not None:
            review["page_limit"] = page_record

        update: QuestState = {"review": review, "review_panel": panel_results}
        if page_hits:
            update["page_limit_rewrites"] = int(state.get("page_limit_rewrites") or 0) + 1
        # As on the single-reviewer path: count a review that sends the
        # experiment back, so the router can cap those re-executes.
        if _review_sends_the_experiment_back(review, state):
            update["code_reexecutes"] = int(state.get("code_reexecutes") or 0) + 1
        # Bump iteration on EITHER verdict=revise OR a non-empty
        # must_flag_hits list. Without the must-flag clause, a malformed
        # persona response that recorded ``verdict=accept`` alongside a
        # must-flag hit would route to revise (via _route_after_review)
        # without consuming iteration budget — the loop could spin. A hit
        # over the page limit bumps nothing: its rewrites have their own
        # counter, capped at ``_PAGE_LIMIT_REWRITES``.
        other_hits = [
            h for h in (review.get("must_flag_hits") or []) if _hit_name(h) != _PAGE_LIMIT_HIT
        ]
        if review.get("verdict") == "revise" or other_hits:
            update["iteration"] = state.get("iteration", 0) + 1
            self._log.info(
                "[review] panel verdict=%s must_flag_hits=%s (agreement=%s, score=%s) -> iteration %d",
                review.get("verdict"), review.get("must_flag_hits") or [],
                agg.get("agreement"), agg.get("score"), update["iteration"],
            )
        else:
            self._log.info(
                "[review] panel verdict=%s (agreement=%s, score=%s)",
                agg.get("verdict"), agg.get("agreement"), agg.get("score"),
            )
        return update

    async def _node_human_feedback(self, state: QuestState) -> QuestState:
        """Pause after the review node and ask the user (CLI / web /
        VSCode) to accept / reject / refine the result before finalising.

        Only fires when ``engine.human_feedback_gate == "after_review"``.
        Writes a snapshot of the current review at
        ``<quest_root>/.fi/human_review.json`` so a UI can read it and
        post back; then raises ``interrupt()`` carrying the same
        payload for the in-process callback path (CLI / VSCode bridge).

        The interrupt payload is a dict ``{"action": "...", "feedback": "..."}``;
        ``action`` ∈ ``{"accept", "reject", "refine"}``. The router
        consumes the resolved value from ``state["human_feedback"]``
        and either ends the quest or bumps iteration → design with the
        feedback text stuffed into state so the design node can read it.
        """
        review = state.get("review") or {}
        verdict = review.get("verdict", "accept")
        # Prefer the path the write node actually recorded in state —
        # accommodates custom pipelines / future relocations of the
        # rendered paper. Falls back to the conventional path if the
        # write node didn't populate it (older quest checkpoints).
        paper_md_state = state.get("paper_md") or ""
        paper_md_path = (
            str(paper_md_state)
            if paper_md_state
            else str(self.quest_root / "paper" / "paper.md")
        )
        snapshot = {
            "quest_id": self.quest_id,
            "iteration": state.get("iteration", 0),
            "verdict": verdict,
            "score": review.get("score"),
            "strengths": review.get("strengths") or [],
            "weaknesses": review.get("weaknesses") or [],
            "suggestions": review.get("suggestions") or [],
            "must_flag_hits": review.get("must_flag_hits") or [],
            # Advisory, never blocking: numbers the arithmetic check could not
            # reconcile with the results. Kept apart from must_flag_hits so
            # every UI can label them differently -- a regex over prose is
            # not grounds for a forced rewrite, but a human should see it.
            "numeric_oracle_warnings": review.get("numeric_oracle_warnings") or [],
            # Captions that name a series their figure does not show; the
            # reviewer was asked to must-flag them.
            "figure_caption_warnings": review.get("figure_caption_warnings") or [],
            "rationale": review.get("rationale", ""),
            "paper_md_path": paper_md_path,
            # Accumulated user-feedback history across refine
            # iterations. Surfaced to the human-review UI so a
            # reviewer can see what was asked for last time.
            "feedback_history": list(state.get("feedback_history") or []),
        }
        # Best-effort disk snapshot so a web UI / VSCode chat can render
        # the gate state without re-loading the LangGraph checkpoint.
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / "human_review.json").write_text(
                json.dumps(snapshot, indent=2) + "\n", encoding="utf-8",
            )
        except OSError as e:
            self._log.debug("[human_feedback] snapshot write failed: %r", e)

        payload = self._pause_for_human(
            kind="review",
            interaction="answer",
            headline=f"review the result (verdict: {verdict}, "
                     f"score: {review.get('score')})",
            steps=[
                "Accept, reject, or refine the paper in the panel "
                "(Web / VSCode), or at the CLI prompt.",
                "Headless run? "
                f"`fi --resume {self.quest_id} --accept` (or `--reject` / "
                "`--refine \"what to change\"`).",
            ],
            payload={"human_review": snapshot},
        )
        # Resume: ``payload`` is what the callback / web POST returned.
        # Validate + normalise so a malformed answer doesn't propagate
        # into the routing layer.
        action = "accept"
        feedback = ""
        if isinstance(payload, dict):
            raw_action = str(payload.get("action") or "accept").lower()
            if raw_action in ("accept", "reject", "refine"):
                action = raw_action
            feedback = str(payload.get("feedback") or "").strip()
        # ``refine`` with no text falls back to ``accept`` — a 0-char
        # refinement is indistinguishable from approval.
        if action == "refine" and not feedback:
            action = "accept"

        update: QuestState = {
            "human_feedback": {"action": action, "feedback": feedback},
        }
        # When the user refines, bump iteration so the loop budget is
        # consumed and the design node sees an explicit "we're in a
        # revise pass" signal (same convention the verdict-driven
        # revise loop uses). Also append to ``feedback_history`` so
        # the design node sees the cumulative refinement requests
        # instead of only the latest one — important when a quest
        # goes through multiple revise passes and the user wants the
        # rewriter to honour all prior asks, not just the last.
        if action == "refine":
            update["iteration"] = state.get("iteration", 0) + 1
            history = list(state.get("feedback_history") or [])
            history.append({
                "iteration": state.get("iteration", 0),
                "text": feedback,
            })
            update["feedback_history"] = history
            self._log.info(
                "[human_feedback] refine → iteration %d (feedback len=%d, total entries=%d)",
                update["iteration"], len(feedback), len(history),
            )
        elif action == "reject":
            # Match the documented contract: the user "rejected" the
            # result, so the review verdict is overwritten to
            # ``rejected`` (distinct from ``accept`` and ``revise``).
            # Downstream artifacts + the cost report can see the
            # rejection without consulting state.human_feedback.
            rejected_review = {**review, "verdict": "rejected"}
            update["review"] = rejected_review
            self._log.info("[human_feedback] action=reject — review.verdict=rejected")
        else:
            self._log.info("[human_feedback] action=%s — finalising", action)
        return update

    # ---- helpers ---------------------------------------------------------

    async def _chat(
        self, prompt: str, *, node: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Single-user-message chat. ``node`` is the engine node name
        (e.g. ``"ideate"``, ``"review"``); when present and the YAML
        config sets ``provider.node_models[node]``, that model is sent
        on this call only. Otherwise the endpoint default applies.

        ``temperature`` defaults to per-node routing: gate/verdict/classifier
        nodes (evidence_gate, review, review_panel.*, claim_check, cross_check,
        relevance_guard) run at 0 for reproducible decisions; generative nodes
        use the 0.2 default. Pass an explicit value to override."""
        assert self._client is not None
        temp = (
            temperature if temperature is not None
            else _temperature_for_node(node)
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self._client.chat(
            messages, temperature=temp, model=self._model_for_node(node),
            node=node or "",
        )
        self._log_chat_cost(node=node or "")
        return response

    def _make_fallback_factory(self, name: str):
        """Build an async factory that lazily resolves+constructs an
        ``LLMClient`` for fallback provider ``name`` (used by
        :class:`FallbackLLMClient`). Nothing is resolved and no proxy spawned
        until the primary provider actually fails and the chain reaches this
        rung. The derived config keeps the primary's timeouts (pure seconds,
        provider-agnostic) but resets model/base_url/api_key and drops
        node_model_fallbacks — those name provider-specific models that would
        be wrong for a different provider."""
        async def _factory() -> LLMClient:
            derived = self.config.provider.model_copy(update={
                "name": name,
                "model": None,
                "base_url": None,
                "api_key_env": None,
                "node_model_fallbacks": {},
                "fallback": [],
            })
            ep = await resolve_endpoint_async(derived, self.supervisor)
            self._log.info(
                "[fallback] resolved %s -> %s (%s)", name, ep.base_url, ep.model,
            )
            return LLMClient(
                ep,
                timeout_s=self.config.provider.http_timeout_s,
                cli_timeout_s=self.config.provider.cli_timeout_s,
                cli_inactivity_timeout_s=(
                    self.config.provider.cli_inactivity_timeout_s
                ),
                node_cli_timeout_s=self.config.provider.node_cli_timeout_s,
                node_http_timeout_s=self.config.provider.node_http_timeout_s,
                node_model_fallbacks={},
                max_prompt_chars=self.config.provider.max_prompt_chars,
                heartbeat_cb=self._llm_heartbeat,
            )
        return _factory

    def _llm_heartbeat(self, payload: dict[str, Any]) -> None:
        """Receive a periodic progress beat from ``LLMClient`` during
        a long-running CLI call (Sonnet 4.6 extended-thinking spans of
        thinking_delta events look identical to a hung process on
        ``--output-format text``; this hook is what makes them visible
        in run.log).

        Called every ~1 s by the CLI streaming reader. We THROTTLE
        emission to once every ``_heartbeat_log_interval_s`` (default
        30 s) so a 9-minute implement call writes ~18 progress lines,
        not 540. Errors are swallowed by the caller — never re-raises.
        """
        if payload.get("kind") != "cli_progress":
            return
        elapsed = float(payload.get("elapsed_s", 0.0))
        idle = float(payload.get("idle_s", 0.0))
        node = str(payload.get("node") or "?")
        # Throttle: only log every N seconds of WALL-CLOCK time. We
        # compare against ``time.monotonic()`` directly, NOT against
        # the call's local elapsed — the call's elapsed resets to 0
        # at the top of every new chat invocation, which would make a
        # second call's "elapsed - last_logged" go negative and
        # suppress every heartbeat after the first call's last log
        # (the bug Copilot review on PR #154 flagged). Keying by node
        # gives concurrent ensembled fan-out calls independent buckets.
        now = time.monotonic()
        last = self._heartbeat_last_logged.get(node, 0.0)
        if now - last < self._heartbeat_log_interval_s:
            return
        self._heartbeat_last_logged[node] = now
        thinking = int(payload.get("thinking_tokens", 0))
        # Provider's heartbeat payload counts CHARACTERS, not UTF-8
        # bytes (text aggregator runs at str-level). Older field name
        # was ``text_bytes`` and was misleading; ``text_chars`` is the
        # honest label and the value matches.
        text_chars = int(payload.get("text_chars", 0))
        # Phrasing: "still thinking" when we have thinking events but no
        # text yet (the OPC case); "still streaming" once text begins;
        # "no events yet" when idle is already past the inactivity
        # window's halfway mark (caller will kill soon).
        if text_chars > 0:
            phase = f"streaming ({text_chars} text chars so far)"
        elif thinking > 0:
            phase = f"thinking ({thinking} thinking-token events)"
        else:
            phase = "no events yet"
        self._log.info(
            "[%s] still waiting on LLM — %s, elapsed=%.0fs, idle=%.0fs",
            node, phase, elapsed, idle,
        )

    def _ensemble_for_node(self, node: str) -> "NodeEnsembleConfig | None":  # type: ignore[name-defined]
        """Return the ensemble config for ``node`` if the YAML carries
        one, else None. Single-call nodes (no ensemble configured)
        keep today's path — no cost or latency regression."""
        ne = self.config.provider.node_ensemble or {}
        return ne.get(node)

    async def _ensemble_chat(
        self, prompt: str, *, node: str, ensemble_cfg: "NodeEnsembleConfig",  # type: ignore[name-defined]
    ) -> "EnsembleResult":  # type: ignore[name-defined]
        """Fan out ``node``'s chat across ``ensemble_cfg.models``, merge
        per ``ensemble_cfg.merge``, return the EnsembleResult.

        Each fan-out call is logged to cost.jsonl via
        ``_log_chat_cost(node=<node>.ensemble[<model>])``; the moderator
        call (if any) under ``<node>.ensemble.moderator``. The cost
        tool buckets these by the ``.ensemble`` substring later.

        Lenient: per-model failures are captured (not raised); the
        merger sees survivors only. All-fail bubbles ``EnsembleError``
        up to the caller, which decides whether to fall back to a
        single-call path or hard-fail."""
        from core.ensemble import (
            cost_jsonl_entries, fanout_chat, merge_synthesize,
            merge_tournament, merge_vote,
        )
        assert self._client is not None

        # The chat_fn ensemble.py calls. Match the keyword shape of
        # LLMClient.chat so the primitive stays transport-agnostic.
        #
        # Concurrency note: fan-out calls run via ``asyncio.gather`` and
        # all share ``self._client.last_model`` / ``last_usage``. We
        # snapshot those two fields IMMEDIATELY after our own chat
        # returns — no awaits in between — so the cost row attributes
        # spend to the model we just used, not the one another in-flight
        # call has since written.
        async def _chat_fn(
            messages: list[dict[str, str]], *,
            temperature: float = 0.2, model: str | None = None,
            node: str = "",
        ) -> str:
            assert self._client is not None
            text = await self._client.chat(
                messages, temperature=temperature, model=model, node=node,
            )
            snap_model = (
                getattr(self._client, "last_model", None) or model or ""
            )
            snap_usage = getattr(self._client, "last_usage", None)
            self._log_chat_cost(
                node=node, model=snap_model, usage=snap_usage,
            )
            return text

        messages = [{"role": "user", "content": prompt}]
        raw = await fanout_chat(
            messages, ensemble_cfg.models,
            chat_fn=_chat_fn, node=node,
        )

        if ensemble_cfg.merge == "tournament":
            result = await merge_tournament(
                raw, moderator_model=(ensemble_cfg.moderator or ensemble_cfg.models[0]),
                chat_fn=_chat_fn, node=node, prompt_summary=prompt[:200],
            )
        elif ensemble_cfg.merge == "synthesize":
            result = await merge_synthesize(
                raw, moderator_model=(ensemble_cfg.moderator or ensemble_cfg.models[0]),
                chat_fn=_chat_fn, node=node, prompt_summary=prompt[:200],
            )
        elif ensemble_cfg.merge == "vote":
            # Vote consumes structured (JSON) responses; the engine
            # caller — typically cross_check — passes JSON-emitting
            # prompts, and merge_vote parses each survivor. The key is
            # currently hard-coded to ``"verdict"`` because that's the
            # only shape the engine produces for this merger today; if
            # we add a second vote caller we'll thread the key through
            # ``NodeEnsembleConfig`` rather than continue to assume.
            result = merge_vote(raw, key="verdict")
        else:
            raise ValueError(f"unknown ensemble merge strategy: {ensemble_cfg.merge!r}")

        # Write the ensemble breadcrumb rows — these sit alongside the
        # per-call rows emitted by ``_log_chat_cost`` above and carry
        # the metadata (role/merge/ok/error/disagreement) the cost tool
        # uses to break spend down by ensemble vs single-call.
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            breadcrumbs = cost_jsonl_entries(
                result, base_node=node, merge_strategy=ensemble_cfg.merge,
            )
            with (self.fi_dir / "cost.jsonl").open("a", encoding="utf-8") as f:
                for row in breadcrumbs:
                    row["ts"] = time.time()
                    f.write(json.dumps(row) + "\n")
        except OSError as e:
            self._log.debug("[cost] failed to write ensemble breadcrumbs: %r", e)
        return result

    async def _chat_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        node: str | None = None,
    ) -> str:
        """Lower-level chat hook for callers (e.g. the knowledge layer's
        source-router) that build their own messages array. Honors the
        same Phase-O per-node model routing as ``_chat``."""
        assert self._client is not None
        response = await self._client.chat(
            messages, temperature=temperature, model=self._model_for_node(node),
            node=node or "",
        )
        self._log_chat_cost(node=node or "")
        return response

    def _clear_clarify_snapshot(self) -> None:
        """Remove the on-disk clarify question/answer files once the clarify
        gate resolves, so the web's "questions present + no answer-file"
        pending check stops showing a stale form. Mirrors the human-review
        ``_consume_snapshot``. Best-effort; an unlink failure is not fatal."""
        for name in ("clarify_questions.json", "clarify_answer.json"):
            try:
                (self.fi_dir / name).unlink()
            except OSError:
                pass

    async def _await_with_heartbeat(
        self, coro: Awaitable[Any], *, label: str, interval_s: float = 30.0,
    ) -> Any:
        """Await ``coro`` while emitting a periodic ``[execute] … still running,
        Ns elapsed`` line to run.log. Long-running work that blocks silently —
        chiefly the experiment subprocess (the executor just waits on
        ``communicate()``) — otherwise produces no log output, and the
        dashboard infers a quest's status from run.log recency, so a busy quest
        reads as "pending"/idle the whole time. The heartbeat keeps that signal
        fresh. Failure-isolated: the heartbeat task can't affect the awaited
        result, and its cancellation is awaited so it never leaks."""
        done = asyncio.Event()
        start = time.monotonic()

        async def _beat() -> None:
            while not done.is_set():
                try:
                    await asyncio.wait_for(done.wait(), timeout=interval_s)
                except asyncio.TimeoutError:
                    self._log.info(
                        "[execute] %s — still running, %ds elapsed",
                        label, int(time.monotonic() - start),
                    )

        beat = asyncio.ensure_future(_beat())
        try:
            return await coro
        finally:
            done.set()
            try:
                await beat
            except asyncio.CancelledError:
                pass

    def _clear_stale_quest_failed_diagnostic(self) -> None:
        """Remove a stale ``quest_failed.md`` from a PRIOR failed run.

        Called from the two non-failed exit paths (clean success and
        data-pause-exit). Same idempotent-cleanup pattern the paper
        generator uses for ``paper_pdf_skipped.md`` on a successful
        PDF compile. Failures to unlink are logged but never raise —
        a stale file is annoying but not fatal.
        """
        stale = self.quest_root / "quest_failed.md"
        if stale.is_file():
            try:
                stale.unlink()
            except OSError as e:
                self._log.warning(
                    "[run] could not remove stale %s: %r", stale, e,
                )

    def _clear_stale_figures(self) -> int:
        """Delete figure files from ``<quest_root>/figures/`` before a (re-)run
        of the experiment, so a re-implemented script's new figure set can't be
        mixed with the previous version's leftovers. Returns the count removed.
        Only files with a figure suffix are touched (``results.csv`` and other
        non-figure artifacts live here too on some quests and must survive).
        Best-effort: unlink errors are swallowed, a no-op on a fresh quest."""
        fig_dir = self.quest_root / "figures"
        if not fig_dir.is_dir():
            return 0
        removed = 0
        for f in fig_dir.iterdir():
            if f.is_file() and f.suffix.lower() in _FIGURE_SUFFIXES:
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed:
            self._log.info(
                "[execute] cleared %d stale figure(s) from a prior experiment "
                "version before re-executing", removed,
            )
        return removed

    def _clear_stale_pause_markers(self) -> None:
        """Remove the on-disk 'needs you' DISPLAY markers at the START of a run.

        A resume that continues past a pause (e.g. a refine that re-enters the
        graph) otherwise leaves these on disk for the quest's whole run, so the
        dashboard / quest page shows a stale "human review" / "user input" /
        "needs you" badge while the quest is actively running and has moved well
        past the pause. Same staleness class as the ``quest_failed.md`` one.

        Crucially this does NOT clear the *answer* files
        (``human_review_answer.json`` / ``clarify_answer.json``): those are the
        INPUT this resume is about to consume (the human-feedback / clarify
        nodes read them), and dropping them here would discard the user's
        decision. It also leaves the ``paused_at_<stage>.flag`` idempotency
        markers alone, since those are state ("already paused here"), not
        display. Any node that genuinely re-pauses re-writes its own marker, so
        a real pending pause re-appears within the same run."""
        for p in (
            self.quest_root / "NEXT_STEP.md",
            self.fi_dir / "pause.json",
            self.fi_dir / "human_review.json",
            self.fi_dir / "clarify_questions.json",
        ):
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                self._log.warning(
                    "[run] could not remove stale pause marker %s: %r", p, e,
                )

    async def _write_quest_failed_diagnostic(
        self,
        exc: BaseException,
        run_config: dict[str, Any] | None,
    ) -> None:
        """Write ``<quest_root>/quest_failed.md`` so a node-raise is
        discoverable from the quest directory itself, not just from
        ``<quest_root>/.fi/launch.log``.

        Captures:

        * The failing node, when LangGraph's state snapshot is
          available (``run_config`` is None for pre-graph failures —
          preflight, endpoint resolution, executor setup — in which
          case the diagnostic notes the pre-graph stage).
        * Exception type + message (no full traceback in the .md —
          that's already in run.log; the .md is a breadcrumb).
        * Tail of the per-quest ``.fi/run.log`` (last ~80 lines), so
          the user has the immediate cause without a separate ``tail``.
        * Provider + model context, so the failure mode (e.g. CLI
          wall-clock timeout) is interpretable in light of the
          transport choice.
        * A copy-pasteable resume command, since most node-raise
          failures (transient API errors, timeouts) recover cleanly
          on resume.

        Best-effort: writing the diagnostic must NEVER mask the
        original exception. The caller wraps THIS call in its own
        try/except and re-raises the original ``exc`` regardless.
        """
        # Resolve the failing node. ``aget_state`` returns a
        # ``StateSnapshot`` whose ``.next`` is a tuple of node names
        # that were about to run — when ``ainvoke`` raised on a
        # node, that's the one. For pre-graph failures (where the
        # saver context never opened) ``run_config`` is None and we
        # report the pre-graph stage instead.
        failing_node = "(pre-graph stage — preflight / endpoint resolution / setup)"
        if run_config is not None and self._client is not None:
            try:
                # Re-open a saver context purely to read the snapshot.
                # The original saver context is already torn down by
                # the time we get here (the inner ``finally`` ran).
                checkpoint_path = self.fi_dir / "state.sqlite"
                async with AsyncSqliteSaver.from_conn_string(
                    str(checkpoint_path),
                ) as saver:
                    graph = self._build_graph().compile(checkpointer=saver)
                    snap = await graph.aget_state(run_config)
                    nxt = getattr(snap, "next", None) or ()
                    if nxt:
                        failing_node = ", ".join(nxt)
            except Exception as e:  # noqa: BLE001
                # Snapshot read failed (saver locked, corrupted, etc.).
                # Fall back to a generic label — the .md is still
                # useful with just the exception + log tail.
                self._log.warning(
                    "[run] could not resolve failing node from "
                    "checkpoint snapshot: %r", e,
                )
                failing_node = "(unknown — could not read checkpoint snapshot)"

        # Tail the per-quest run.log. The full trace is at the end of
        # the file; ~80 lines is comfortably more than any single
        # traceback but small enough to not bury the user.
        #
        # Bounded read: seek to the last 64 KB rather than reading the
        # whole file into memory. Quest run.logs occasionally grow to
        # tens of MB (LangGraph trace verbosity + per-iteration retry
        # spam), and we don't want a diagnostic write — which fires
        # exactly when the user is already having a bad day — to slow
        # down further on a giant log. 64 KB is more than enough for
        # 80 lines of even very long tracebacks. Mirrors the helper
        # in ``web/server.py::_read_log_tail``.
        run_log_path = self.fi_dir / "run.log"
        log_tail = "(run.log not on disk — likely a pre-logging failure)"
        if run_log_path.is_file():
            try:
                size = run_log_path.stat().st_size
                with run_log_path.open("rb") as f:
                    f.seek(max(0, size - 65536))
                    tail_bytes = f.read()
                lines = tail_bytes.decode("utf-8", errors="replace").splitlines()
                log_tail = "\n".join(lines[-80:])
            except OSError as e:
                log_tail = f"(could not read run.log: {e!r})"

        # Provider context — separate the YAML transport from any
        # bridge override that landed in ``provider.extra``.
        provider_name = self.config.provider.name
        provider_model = self.config.provider.model or "(provider default)"
        bridge_extras = []
        for key in ("bridge_port", "bridge_socket"):
            val = (self.config.provider.extra or {}).get(key)
            if val:
                bridge_extras.append(f"{key}={val!r}")
        provider_extra_str = ", ".join(bridge_extras) or "(none)"

        # Normalize the topic for header rendering. YAML block-scalar
        # topics can contain embedded newlines, which would break the
        # single-line ``**Topic:**`` header and split the markdown
        # structure across multiple bullets. Collapse all internal
        # whitespace (newlines, tabs, runs of spaces) to a single
        # space before truncating to 200 chars.
        topic_one_line = " ".join(self.config.topic.split())[:200]
        body = (
            f"# Quest failed before producing a paper\n"
            f"\n"
            f"**Quest ID:** `{self.quest_id}`\n"
            f"**Topic:** {topic_one_line}\n"
            f"**Failing node:** `{failing_node}`\n"
            f"**Provider:** `{provider_name}` / model `{provider_model}`"
            f" / extras: {provider_extra_str}\n"
            f"\n"
            f"## What broke\n"
            f"\n"
            f"```\n"
            f"{type(exc).__name__}: {exc}\n"
            f"```\n"
            f"\n"
            f"## Last ~80 lines of `.fi/run.log`\n"
            f"\n"
            f"```\n"
            f"{log_tail}\n"
            f"```\n"
            f"\n"
            f"## How to resume\n"
            f"\n"
            f"Most node failures are transient (rate-limit, CLI "
            f"wall-clock timeout, network blip). The LangGraph "
            f"checkpoint at `.fi/state.sqlite` lets the engine "
            f"continue from the failing node on resume:\n"
            f"\n"
            f"```bash\n"
            f"python launch.py --config "
            f"{(self.quest_root / 'config.yaml').as_posix()} "
            f"--resume {self.quest_id}\n"
            f"```\n"
            f"\n"
            f"If the same node fails repeatedly, the cause is likely "
            f"systematic. Common follow-ups:\n"
            f"\n"
            f"- **CLI wall-clock timeout** — switch to a smaller model "
            f"via `provider.node_models.<failing_node>` (e.g. Haiku "
            f"for `implement`), or shrink the prompt by disabling "
            f"the ensemble preset.\n"
            f"- **Bridge error** — the bridge dumps the available "
            f"`id|family` model catalog on failed lookups; look for "
            f"that line in the embedded log tail above to confirm "
            f"the YAML's `provider.model` matches what Copilot "
            f"actually exposes.\n"
            f"- **Provider auth / quota** — re-authenticate "
            f"(`claude login`, `gh auth refresh`, etc.) and retry.\n"
            f"\n"
            f"This file is auto-deleted on the next successful run "
            f"of this quest.\n"
        )
        diag_path = self.quest_root / "quest_failed.md"
        try:
            # Defensive: the failure might have fired before ``Engine.run``
            # got past the first mkdir, so the quest_root may not exist
            # yet. Cheap to create it here — exist_ok=True keeps the
            # common case (root already there) a no-op.
            self.quest_root.mkdir(parents=True, exist_ok=True)
            diag_path.write_text(body, encoding="utf-8")
            self._log.warning(
                "[run] quest_failed diagnostic written to %s", diag_path,
            )
        except OSError as e:
            self._log.warning(
                "[run] could not write %s: %r", diag_path, e,
            )

    def _write_cost_summary(self) -> None:
        """Aggregate ``.fi/cost.jsonl`` into ``.fi/cost.summary.json``
        at quest finalization. The summary carries totals and per-node
        / per-model breakdowns so the cost tool can render a one-line
        "this quest cost N tokens / $X across M requests" without
        re-walking the raw log on every render.

        Schema (per the JSON file):

            {
              "total_requests": <int>,
              "total_prompt_tokens": <int>,
              "total_completion_tokens": <int>,
              "total_tokens": <int>,
              "total_cost_usd": <float | null>,
              "total_cost_usd_partial": <bool>,  # some rows priced, some not
              "estimated_rows": <int>,        # rows from char-based fallback
              "unpriced_requests": <int>,     # rows whose cost_usd is null
              "by_node":  { <node>: {...same shape...} },
              "by_model": { <model>: {...same shape...} },
              "generated_at": <epoch>,
            }

        ``total_cost_usd`` (and each bucket's ``cost_usd``) sums the rows
        that carry a price; it is ``null`` only when no row does. A model the
        price table does not list logs ``cost_usd: null`` with its tokens, so
        a quest that mixes priced and unpriced models has a total that is a
        lower bound: ``total_cost_usd_partial`` is ``true`` then, and
        ``unpriced_requests`` says how many calls the total leaves out.

        Best-effort: a missing / unreadable cost.jsonl produces a
        summary with zeros; no exception bubbles out.
        """
        write_cost_summary(self.fi_dir)

    def _log_chat_cost(
        self, *, node: str,
        model: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Append one row to ``<quest_root>/.fi/cost.jsonl`` per chat
        call. Pulled out of ``_chat`` / ``_chat_messages`` so both
        call sites stay tight.

        Captures: timestamp, node, model, usage dict (when the
        transport returned one), estimated USD cost (when the model
        has a pricing row). CLI / vscode_bridge transports leave
        ``last_usage = None`` — we still write the row so the chart
        on ``/quest/<id>`` can show "call count" + node breakdown
        even when token-level data is unavailable.

        ``model`` / ``usage`` overrides exist for concurrent callers
        (ensemble fan-out): the shared ``last_model`` / ``last_usage``
        on the LLMClient can be overwritten between awaits when N
        chat calls run via ``asyncio.gather``. Concurrent paths
        snapshot the values immediately after their own chat returns
        and pass them in explicitly so the row attributes spend to the
        correct model.
        """
        assert self._client is not None
        # Tolerate test stubs that pre-date the cost-tracking fields —
        # engine + test suite share dozens of fake LLMClient
        # implementations, and patching each one to add ``last_usage``
        # is mechanical work that doesn't add coverage.
        # ``getattr(..., None)`` returns the new field when the real
        # LLMClient is in play, or skips the cost-log row entirely
        # when a stub is used.
        if usage is None:
            usage = getattr(self._client, "last_usage", None)
        if model is None:
            model = getattr(self._client, "last_model", None) or ""
        append_cost_row(self.fi_dir, node=node, model=model, usage=usage)

    def _model_for_node(self, node: str | None) -> str | None:
        """Resolve the effective model for a node via the shared
        ``core.provider.model_for_node`` lookup — None when the lookup
        misses so the transport falls through to the endpoint default.
        Accepts hierarchical keys like ``"review_panel.methodologist"``
        (review-panel personas)."""
        if not node:
            return None
        return model_for_node(self.config.provider.node_models, node)

    async def _preflight_required_skills(self) -> None:
        """A skill named in ``engine.skills_required`` that cannot be used stops
        the quest here, before any LLM call, rather than after the literature
        has been paid for."""
        required = list(dict.fromkeys(self.config.engine.skills_required or []))
        if not required:
            return
        from core.skills import loadable_skills

        usable, rejected = await asyncio.to_thread(
            loadable_skills, required, external_dirs=self._skill_dirs,
        )
        _raise_if_required_skills_unusable(required, usable, rejected)
        self._log.info("[skills] required skills are usable: %s", ", ".join(required))

    def _stage_example_inputs(self) -> None:
        """Copy ``execution.inputs`` into the quest. A path that is missing or
        far too large stops the quest here, before any LLM call."""
        sources = list(self.config.execution.inputs or [])
        if not sources:
            return
        from core.example_inputs import stage_inputs

        stage_inputs(sources, self.quest_root, self._log)

    def _job_block(self) -> str:
        """The background-job contract for the design and code-writing prompts,
        or nothing when ``execution.background_jobs`` is off."""
        return _JOB_PROTOCOL if self.config.execution.background_jobs else ""

    def _wait_for_job(self, job: dict[str, Any], code_path: Path) -> None:
        """The experiment reported its job as pending. Record what is being
        waited for and pause; a resume runs the experiment node again, which
        re-runs the (idempotent) script. Never returns: ``interrupt()`` raises."""
        from core import job_watch

        info = job_watch.write_pending(
            self.fi_dir, job, code_path.relative_to(self.quest_root).as_posix(),
        )
        detail = job_watch.describe(info)
        self._log.info("[execute] the job is pending (%s)", detail)
        self._pause_for_human(
            kind="results",
            interaction="supply",
            headline="waiting for the background job",
            steps=[
                f"The experiment submitted a background job and is waiting for it ({detail}).",
                "Nothing to do while it runs; FI stops here and keeps everything so far.",
                f"To be woken automatically: `fi --watch {self.quest_id} --config <the quest's yaml>` "
                "(or `@fi /watch`, or the Watch button on the quest page). Every check is printed "
                "and written to run.log.",
                f"Or check once yourself when it should be done: `fi --resume {self.quest_id}`. "
                "That re-runs `code/experiment.py`, which reports the job pending again or "
                "collects its results.",
            ],
            payload={"job_pending": True, "quest_id": self.quest_id, "job": job},
            upload_targets=[],
        )

    async def poll_job(self) -> tuple[str, dict[str, Any]]:
        """Run the experiment script once, as a resume would, and say whether the
        job it submitted is still pending: ``("pending" | "done" | "failed", info)``.
        No LLM call; this is what ``--watch`` does on a timer."""
        from core import job_watch
        from core.example_inputs import ENV_VAR, examples_dir, list_inputs

        await self.executor.setup(self.quest_root)
        if self.config.execution.sandbox == "docker":
            # The same skills the quest mounted when it ran the experiment: a
            # script that reads /fi-skills/... needs them on every check.
            try:
                recorded = json.loads(
                    (self.fi_dir / _SKILL_MOUNTS_FILE).read_text(encoding="utf-8")
                ).get("skills") or []
            except (OSError, ValueError, AttributeError):
                recorded = []
            await asyncio.to_thread(
                self._mount_selected_skills,
                {"selected_skills": [str(n) for n in recorded]}, record=False,
            )
        py = self.executor.python_path(self.quest_root)
        code_path = self.quest_root / "code" / "experiment.py"
        env = (
            {**os.environ, ENV_VAR: str(examples_dir(self.quest_root))}
            if list_inputs(self.quest_root) else None
        )
        result = await self.executor.execute(
            [str(py), str(code_path)], cwd=self.quest_root,
            timeout_s=self.config.execution.timeout_s, env=env,
        )
        result_json = _extract_result_json(result.stdout)
        job = job_watch.job_of(result_json)
        if job is not None and job["status"] == job_watch.PENDING and result.returncode == 0:
            return job_watch.PENDING, job_watch.write_pending(
                self.fi_dir, job, code_path.relative_to(self.quest_root).as_posix(),
            )
        if result.returncode == 0 and result_json is not None:
            return job_watch.DONE, {"note": "the results are ready"}
        return job_watch.FAILED, {
            "note": f"the script exited {result.returncode}: {(result.stderr or '')[-200:].strip()}",
        }

    def _inputs_block(self) -> str:
        """What the design and code-writing prompts say about the user's example
        files. Read from disk each time, so files dropped in during a pause
        count."""
        from core.example_inputs import render_block

        return render_block(self.quest_root) or "(none supplied)"

    def _preflight_paper_pdf(self) -> None:
        """Verify the host can produce ``paper.pdf`` BEFORE the quest
        runs any LLM calls.

        Skipped entirely when:
        * ``paper_pdf`` is not in ``output.kinds`` — nothing to check.
        * The check passes — pandoc on PATH AND at least one LaTeX
          engine reachable (pdflatex / tectonic on PATH, or a repo-
          local ``tools/tectonic[.exe]``).

        Warns vs raises based on ``output.require_pdf``:
        * ``require_pdf: false`` (default) — emit a WARNING with the
          install recipe and continue. The user still gets paper.md
          and the ``paper_pdf_skipped.md`` diagnostic that the paper
          generator writes at the end (see #55).
        * ``require_pdf: true`` — raise ``RuntimeError`` immediately,
          aborting the quest before any LLM cost is incurred. The
          error message carries the exact same install recipe the
          warning version would have logged.

        The check uses the same engine-discovery logic that
        ``generation/paper.py:_find_pdf_engine`` uses at compile time,
        so a pre-flight pass is a strong predictor of a post-LLM-call
        compile success.
        """
        if "paper_pdf" not in self.config.output.kinds:
            return
        # Lazy import to avoid pulling generation/* into the engine
        # module just for a pre-flight; engine imports stay small.
        from generation.paper import PaperGenerator
        from generation._pandoc import find_pandoc
        pandoc_exe = find_pandoc()
        # ``PaperGenerator._find_pdf_engine`` is an instance method but
        # doesn't touch ``self.config`` for its lookup. Instantiate a
        # cheap one for the engine discovery.
        engine_lookup = PaperGenerator(self.config)._find_pdf_engine()

        missing: list[str] = []
        if pandoc_exe is None:
            missing.append("pandoc")
        if engine_lookup is None:
            # No LaTeX engine — but the HTML/Chromium fallback can still
            # produce paper.pdf (pandoc + a system browser). So a missing
            # LaTeX engine is only a real problem when that fallback is
            # disabled or no browser is reachable.
            from generation._html_pdf import find_html_browser
            fallback_browser = (
                find_html_browser()
                if (self.config.output.html_pdf_fallback and pandoc_exe)
                else None
            )
            if fallback_browser is not None:
                self._log.info(
                    "[preflight] no LaTeX engine found, but the HTML/Chromium "
                    "fallback is available (pandoc + %s headless) — paper.pdf "
                    "will render via HTML.", fallback_browser[0],
                )
            else:
                missing.append(
                    "a LaTeX engine (pdflatex/tectonic) — or, for the HTML "
                    "fallback, a Chromium-family browser (Edge/Chrome/Chromium)"
                )
        if not missing:
            self._log.info(
                "[preflight] paper.pdf prereqs OK: pandoc=%s pdf_engine=%s",
                pandoc_exe, engine_lookup[1] if engine_lookup else None,
            )
            return

        recipe = (
            "Install pandoc: Windows `winget install --id JohnMacFarlane.Pandoc`, "
            "macOS `brew install pandoc`, Linux via package manager. "
            "No-admin alternative (any OS): `pip install pypandoc_binary` — it "
            "ships a real pandoc and FI finds it automatically; or drop the "
            "portable pandoc binary into `tools/`. "
            "For the LaTeX engine, the no-admin path is "
            "`python launch.py --install-tectonic` (drops a 70 MB binary "
            "into `tools/`); standard alternative is MiKTeX/TeX Live."
        )
        what_missing = " AND ".join(missing)
        # Wording note: pandoc lookup is just ``shutil.which`` (PATH only),
        # but the LaTeX engine lookup is broader — it also accepts the
        # repo-local ``tools/tectonic[.exe]`` written by
        # ``python launch.py --install-tectonic``. So "not found" is the
        # honest description across both; "not on PATH" alone would send
        # users looking in the wrong place when their tools/tectonic was
        # removed or never installed.
        if self.config.output.require_pdf:
            raise RuntimeError(
                f"[preflight] paper_pdf requested with "
                f"output.require_pdf=True but {what_missing} not found "
                f"on this host (pandoc is searched on PATH; the LaTeX "
                f"engine also accepts a repo-local tools/tectonic). "
                f"Aborting before LLM calls. {recipe}"
            )
        self._log.warning(
            "[preflight] paper_pdf requested but %s not found "
            "(pandoc is searched on PATH; LaTeX engine also accepts "
            "repo-local tools/tectonic). Quest will continue "
            "(paper.md will still be produced) but paper.pdf will be "
            "skipped with a diagnostic file. Set output.require_pdf=True "
            "in YAML to abort early on this condition instead. %s",
            what_missing, recipe,
        )

    def _collect_artifacts(self, state: QuestState) -> QuestArtifacts:
        paper_md = self.quest_root / "paper" / "paper.md"
        figures = self.quest_root / "figures"
        manifest = self.quest_root / "paper" / "paper_bundle_manifest.json"
        figures_present = (
            figures.is_dir()
            and any(
                p.is_file() and p.suffix.lower() in _FIGURE_SUFFIXES
                for p in figures.iterdir()
            )
        )
        return QuestArtifacts(
            quest_id=self.quest_id,
            quest_root=self.quest_root,
            paper_md=paper_md if paper_md.exists() else None,
            paper_pdf=None,
            figures_dir=figures if figures_present else None,
            bundle_manifest=manifest if manifest.exists() else None,
            raw_state=dict(state),
        )

    def _record_skill_usage(self, state: QuestState) -> None:
        """Write this quest into the provenance of the skills it used.

        Gated on an accepted review, matching ``write_back_only_on_accept``
        next door: a paper that cleared the numeric oracle, the plausibility
        gate and review is the strongest evidence available that the skill
        contributed something sound. A rejected quest teaches nothing about
        the skill — only about that attempt.

        Best-effort throughout. The quest has already succeeded by the time
        this runs, so a bookkeeping failure must never turn that into an
        error.
        """
        names = list(state.get("selected_skills") or [])
        if not names:
            return
        verdict = str((state.get("review") or {}).get("verdict", "")).lower()
        if verdict != "accept":
            self._log.info(
                "[skills] usage not recorded (verdict=%s) — a rejected quest "
                "says nothing about the skill", verdict or "unknown",
            )
            return
        try:
            from core.skills.usage import record_quest

            written = record_quest(
                names, self.quest_id, "accept", external_dirs=self._skill_dirs,
            )
        except Exception as e:  # noqa: BLE001 - bookkeeping is never fatal
            self._log.warning("[skills] usage not recorded: %s", e)
            return
        if written:
            self._log.info("[skills] recorded use in: %s", ", ".join(written))

    def _write_back_knowledge(self, artifacts: QuestArtifacts, state: QuestState) -> None:
        if not self.knowledge.enabled or not self.config.knowledge.write_back_quests:
            return
        if artifacts.paper_md is None:
            return

        review = state.get("review") or {}
        verdict = review.get("verdict", "accept")

        # Gate on the accept verdict so the long-term store accumulates
        # only research the review node signed off on. With
        # `write_back_only_on_accept = False`, every finished quest
        # lands — useful while bootstrapping an empty corpus.
        if self.config.knowledge.write_back_only_on_accept and verdict != "accept":
            self._log.info(
                "[write-back] skipped: verdict=%s (write_back_only_on_accept=True)",
                verdict,
            )
            return

        analysis = state.get("analysis") or {}
        design = state.get("design") or {}
        summary_parts: list[str] = []
        if analysis.get("summary"):
            summary_parts.append(str(analysis["summary"]))
        for kf in analysis.get("key_findings", []) or []:
            summary_parts.append(f"- {kf}")
        summary = "\n".join(summary_parts)

        # Distill the literature the agent actually saw into a curated
        # `external_refs` list. The Knowledge layer writes a spine doc
        # per ref so Axon becomes title-/topic-searchable after accept.
        external_refs: list[dict[str, Any]] = []
        for d in (state.get("literature") or []):
            m = d.get("metadata") or {}
            if not (m.get("title") or m.get("doi") or m.get("arxiv_id") or m.get("pmid")):
                continue
            external_refs.append({
                "title": m.get("title", ""),
                "authors": m.get("authors", []),
                "year": m.get("year") or (m.get("published") or "")[:4] or None,
                "venue": m.get("venue") or m.get("publisher") or "",
                "doi": m.get("doi", ""),
                "arxiv_id": m.get("arxiv_id", ""),
                "pmid": m.get("pmid", ""),
                "source": m.get("source", ""),
                "url": m.get("url", ""),
                "abstract": (d.get("content") or "")[:1500],
            })

        # Rich, indexable metadata that future quests' ideate node can
        # filter / rank past work by. Keep the keys flat and JSON-safe.
        #
        # `paper_md_relpath` is the path RELATIVE TO `quest_root` (not
        # absolute) — Axon stores this; the absolute path would leak
        # the user's home-directory layout into the long-term corpus
        # and make exported / shared Axon corpora non-portable. Callers
        # that need the on-disk file should join with `quest_root`.
        try:
            paper_md_relpath = str(
                artifacts.paper_md.relative_to(artifacts.quest_root)
            )
        except ValueError:
            # paper_md outside quest_root (shouldn't happen, but be safe).
            paper_md_relpath = artifacts.paper_md.name
        # The writer's keywords: a line under the abstract, or a comment in
        # the formats that show none. The index card carries them in its text.
        from generation._keywords import paper_keywords

        try:
            keywords = paper_keywords(artifacts.paper_md.read_text(encoding="utf-8"))
        except OSError:
            keywords = []
        meta: dict[str, Any] = {
            "title": state.get("title", ""),
            "topic": state.get("topic", "")[:1000],
            "keywords": keywords,
            "verdict": verdict,
            "score": review.get("score"),
            "iteration": state.get("iteration", 0),
            "hypothesis": design.get("hypothesis", ""),
            "method_summary": design.get("method", ""),
            "key_findings": list(analysis.get("key_findings", []) or [])[:20],
            "result_json": state.get("result_json") or {},
            "figures": list(state.get("figures", []) or []),
            "provider": self.config.provider.name,
            "model": self.config.provider.model or "(cli-default)",
            "paper_md_relpath": paper_md_relpath,
            "external_refs": external_refs,
        }
        ok = self.knowledge.add_quest_artifacts(
            quest_id=self.quest_id,
            paper_md_path=artifacts.paper_md,
            summary=summary,
            metadata=meta,
        )
        self._log.info(
            "[write-back] axon ingest=%s (verdict=%s, score=%s)",
            ok, verdict, review.get("score"),
        )


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
        (fi_dir / "cost.summary.json").write_text(
            json.dumps(_aggregate_cost_rows(rows), indent=2) + "\n", encoding="utf-8",
        )
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


def _load_prompts() -> dict[str, string.Template]:
    names = (
        "clarify", "ideate", "ideate_reflect", "ideate_tournament",
        "design", "design_self_critique",   # second-pass methodology audit
        "implement",                        # legacy one-shot (resume fallback)
        "implement_outline",                # two-stage implement: scaffold
        "select_skills",        # pick which skills this quest carries
        "implement_body",                   # two-stage implement: fills bodies
        "execute_reflect", "analyze",
        "cross_check",
        "cross_check_verify",   # CoVe-style second-pass verification
        "claim_check",          # ground each paper claim to evidence
        "evidence_gate",        # weigh evidence sufficiency before write
        "literature_screen",    # grade retrieved sources 0-3 before they reach the corpus
        "write", "review",
        "write_patch",      # a revise for flagged passages: edits, not a new paper
        "review_moderate",  # review-panel moderator prompt
        "data_load",        # no-simulation mode — synthesize result_json
                            # from user-supplied data
        "web_plots",        # no-simulation mode — chart the collected
                            # web/data content (text-only outputs otherwise)
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


def _thin_source(meta: dict[str, Any], content: str) -> str | None:
    """``"title"`` when a source's stored text is only its title, ``"blurb"``
    when it is a book without full text (all it has is a short description), and
    ``None`` for every other source, full text and abstracts included. Such a
    record cannot show a finding, a number or a mechanism, whatever it is cited
    for."""
    if meta.get("content_quality") == "full_text" or meta.get("fetched_full_text"):
        return None
    title = str(meta.get("title") or meta.get("source") or "")
    if len(_format_lit_excerpt(content or "", title).strip()) < _THIN_TEXT_CHARS:
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
    if query and len(content) > budget:
        from .passages import select_relevant_excerpt
        excerpt = select_relevant_excerpt(
            content, query, budget_chars=budget, mode=mode,
        )
    else:
        excerpt = content[:budget]
    if title and excerpt.lstrip().startswith(title):
        # Drop the leading title + immediately-following separator
        # (newline or ". "). Keep everything after as the real abstract.
        return excerpt.lstrip()[len(title):].lstrip(".\n ")
    return excerpt


def _is_citable(meta: dict[str, Any]) -> bool:
    """An entry is citable if it has at least a real title AND
    one identifying field (authors, year, venue, doi/url/arxiv).

    Filters out partial loader output where only a path/slug exists
    — those entries would render as ``[i] item-i`` or ``[i] (no title)``
    and the LLM tends to fabricate author names / URLs to complete
    the slot. Honesty > completeness on a thin retrieval pull.
    """
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
    "fi_critique",
    "fi_digest",
    "fi_portfolio",
    "fi_proposal",
    "fi_summary",
    "fi_summary_input",
    "fi_source_catalog",
    "fi_paper_spine",
})


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
        header = _format_lit_header(meta, keep_idx)
        excerpt = _format_lit_excerpt(
            d.content, title, query=query, budget=budget, mode=mode,
        )
        lines.append(f"{header}\n{excerpt}" if excerpt else header)
    if not lines:
        return "(no prior work surfaced from the knowledge base)"
    return "\n\n".join(lines)


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
        content = item.get("content", "") or ""
        header = _format_lit_header(meta, label, _thin_source(meta, content) if mark_thin else None)
        excerpt = _format_lit_excerpt(
            content, title,
            query=query, budget=budget, mode=mode,
        )
        lines.append(f"{header}\n{excerpt}" if excerpt else header)
    if not lines:
        return "(no prior work surfaced from the knowledge base)"
    return "\n\n".join(lines)


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


def _foundational_work_text(work: dict[str, Any]) -> str:
    """A work the model suggested as ``title (year, authors)``, on one line."""
    title = " ".join(str(work.get("title") or "").split())
    who = work.get("authors")
    if isinstance(who, list):
        who = ", ".join(str(a) for a in who if a)
    bits = [b for b in (str(work.get("year") or "").strip(), " ".join(str(who or "").split())) if b]
    return f"{title} ({', '.join(bits)})" if bits else title


def _foundational_outcomes(
    suggestions: list[dict[str, Any]], new: list[RetrievedDoc],
    found: list[RetrievedDoc], docs: list[RetrievedDoc],
) -> tuple[list[str], list[str], list[str]]:
    """What became of each suggested work, as :func:`_foundational_work_text`
    lines: the ones the pass added (``new``), the ones already among the search
    results (the lookup found them, or the search had them, so they are not new)
    and the ones OpenAlex did not have. A suggestion is matched to a record by
    the same title rule the lookup used, so a record counts for the suggestion
    that found it."""
    def matches(work: dict[str, Any], candidates: list[RetrievedDoc]) -> bool:
        title = str(work.get("title") or "")
        return any(_titles_match(title, str((d.metadata or {}).get("title") or "")) for d in candidates)

    added: list[str] = []
    already: list[str] = []
    dropped: list[str] = []
    for work in suggestions:
        text = _foundational_work_text(work)
        if matches(work, new):
            added.append(text)
        elif matches(work, found) or matches(work, docs):
            already.append(text)
        else:
            dropped.append(text)
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
        if meta.get("foundational") and not label.startswith("W")
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
    return (
        "\n\n## Foundational works this paper does not cite (advisory)\n"
        "These works were retrieved, kept by the literature screen and marked as foundational "
        "(an original paper or standard textbook the topic rests on, or a work several of the "
        "retrieved papers cite), but the paper does not cite them:\n"
        + "\n".join(_foundational_line(label, meta, numbered=False) for label, meta in missing)
        + "\nAdvisory only: if one of them bears on a claim the paper makes (the original paper "
        "for a method or relation the paper uses, the standard textbook for the field), you may "
        "ask for it under `suggestions`. Do not add anything to `must_flag_hits` and do not "
        "choose `revise` because of this alone. A work that does not bear on the paper needs "
        "no citation."
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


# Claim grounding shows the model this much of each cited source's text: the
# passages most related to the sentences citing it. It saw only titles before,
# and grounded "explicit Euler is unstable in oscillatory systems" in a paper
# on discrete gradients whose abstract never mentions Euler. At 1,500 only the
# opening chunk and one 1,200-character passage fit, so a source cited by six
# sentences showed one passage, and supported citations were rejected.
_CLAIM_SOURCE_CHARS = 6000
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
Never sleep-wait for the job. Ignore FI_PILOT and FI_REPLICATE_SEED: no pilot or replicate run is made for a background job.
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
    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
    return str(content or "")


def _claim_source_block(label: str, meta: dict[str, Any], text: str, sentences: list[str]) -> str:
    """One source for the claim check: its label and title, and, when the
    paper cites it, the passages of its text most related to the citing
    sentences."""
    title = str(meta.get("title") or "").strip()
    ident = meta.get("url") if label.startswith("W") else (f"DOI:{meta['doi']}" if meta.get("doi") else "")
    head = f"[{label}] {title}" + (f" · {ident}" if ident else "")
    if not sentences:
        return head
    if not text.strip():
        return head + "\n(no text of this source was retrieved, so nothing can be quoted from it)"
    excerpt = _format_lit_excerpt(text, title, query=" ".join(sentences), budget=_CLAIM_SOURCE_CHARS)
    return head + "\nText:\n" + excerpt.strip()


def _claim_distilled_block(analysis: dict[str, Any], budget: int) -> tuple[str, int]:
    """The analysis's own key findings and the claims it says the run
    supports, every item WHOLE, as much as ``budget`` holds. Returns the block
    and how many items did not fit.

    These are what an "experiment" basis is checked against — they carry the
    run's numbers at the precision the paper writes them — so an item is
    either shown entire or not at all. Cutting the serialised evidence at a
    character count severed a finding mid-sentence and silently dropped the
    ones after it."""
    kept: dict[str, list[Any]] = {"key_findings": [], "claims_supported": []}
    dropped = 0
    for key in ("key_findings", "claims_supported"):
        for item in (analysis.get(key) or []):
            kept[key].append(item)
            if len(json.dumps(kept, indent=2)) > budget:
                kept[key].pop()
                dropped += 1
    return json.dumps(kept, indent=2), dropped


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

    if not isinstance(summarised, dict) or len(summarised) < 2:
        return _leading_slice()
    from .passages import rank_by_relevance
    names = list(summarised)
    blocks = {name: json.dumps({name: summarised[name]}, indent=2) for name in names}
    scores = rank_by_relevance([blocks[name] for name in names], query)
    kept: set[str] = set()
    used = 0
    for i in sorted(range(len(names)), key=lambda i: scores[i], reverse=True):
        if used + len(blocks[names[i]]) <= room:
            kept.add(names[i])
            used += len(blocks[names[i]])
    if not kept:
        return _leading_slice()
    body = "\n".join(blocks[name] for name in names if name in kept)
    left_out = [name for name in names if name not in kept]
    # Name a few of the branches that did not fit; a sweep can have hundreds,
    # and the note must not itself crowd out the results.
    missed = ", ".join(left_out[:3]) + (
        f" and {len(left_out) - 3} more" if len(left_out) > 3 else "")
    note = f", the branches most related to the findings ({missed} did not fit)" if left_out else ""
    return f"{head}{note}:\n{body}"[:budget]


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
    engine. What the ``further_reading`` bib export follows."""
    block = _further_reading_block(markdown)
    return None if block is None else block.labels


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


def _renumber_citations(body: str, n_papers: int) -> tuple[str, dict[int, int], list[int]]:
    """Number the scholarly sources ``body`` cites 1, 2, 3… in the order it
    first cites them, and rewrite its citations to match. Web-page labels
    ([W1]) stay as they are. A number no source has (above ``n_papers``) is
    taken out of its citation, and a citation left empty goes.

    Not citations, so left alone: brackets in inline code or math, a bracket
    followed by "(" (a link), and one holding a 0 or a year-sized number
    ("[0, 1]", "[1990–2020]").

    Returns the new text, the map from old number to new, and the numbers
    that were taken out."""
    spans = [(s.start(), s.end()) for s in _NOT_PROSE_RE.finditer(body)]
    found: list[tuple[re.Match[str], list[str]]] = []
    mapping: dict[int, int] = {}
    for m in _CITATION_BRACKET_RE.finditer(body):
        if body[m.end():m.end() + 1] == "(" or any(s <= m.start() < e for s, e in spans):
            continue
        labels = _citation_labels(m.group(1))
        numbers = [int(label) for label in labels if label.isdigit()]
        if any(n == 0 or n >= _NOT_A_CITATION_NUMBER for n in numbers):
            continue
        found.append((m, labels))
        for n in numbers:
            if n <= n_papers and n not in mapping:
                mapping[n] = len(mapping) + 1
    dropped: list[int] = []
    pieces: list[str] = []
    last = 0
    for m, labels in found:
        numbers = [int(label) for label in labels if label.isdigit()]
        dropped += [n for n in numbers if n not in mapping and n not in dropped]
        new = _number_list(sorted({mapping[n] for n in numbers if n in mapping}))
        pages = list(dict.fromkeys(label for label in labels if not label.isdigit()))
        inside = ", ".join(([new] if new else []) + pages)
        start = m.start()
        if not inside:
            while start > last and body[start - 1] in " \t":
                start -= 1
        pieces += [body[last:start], f"[{inside}]" if inside else ""]
        last = m.end()
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
    ``## References`` lists exactly those sources; ``## Further reading``
    lists every web page. Returns the paper, the reordered literature and the
    citation numbers that named no source."""
    n_papers = sum(1 for label, _m, _i in _labelled_sources(literature, audience) if not label.startswith("W"))
    body, mapping, dropped = _renumber_citations(_strip_source_lists(markdown), n_papers)
    ordered = _literature_in_citation_order(literature, mapping, audience)
    refs = build_references(ordered, audience=audience)[:len(mapping)]
    if refs:
        lines = [f"{r['n']}. {_ref_citation_text(r)}" for r in refs]
        body = body.rstrip() + "\n\n" + "\n".join(["## References", "", *lines]) + "\n"
    body = _append_further_reading(body, build_further_reading(ordered, audience=audience))
    return body, ordered, dropped


def cited_references(
    literature: list[Any], paper_md: str, *, audience: str = "external",
) -> list[dict[str, Any]]:
    """The numbered References the paper's text cites: what its References
    section lists. The slides and the bib export use this, so none of them
    lists a source the paper does not cite."""
    counts = _citation_counts(paper_md)
    return [r for r in build_references(literature, audience=audience) if r["n"] in counts]


_CLARIFY_LABELS = {
    "comparative_baseline": "Comparative baseline",
    "empirical_vs_theoretical": "Empirical / theoretical",
    "success_metric": "Success metric",
    "budget": "Time / compute budget",
    "output_kinds": "Desired output kinds",
    "study_depth": "Study depth",
    "paper_venue": "Paper venue / template",
    "topic_shape": "Topic shape",
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
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value) or "(empty)"
        lines.append(f"- **{label}**: {value}")
    return "\n".join(lines) or "(no answers recorded)"


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
            "question": "Which paper template should we use? (generic / neurips / iclr / ieee_access / nature_mi)",
            "default": "generic",
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
        return persona_path.read_text(encoding="utf-8").strip()
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
    return {
        **(exec_env or os.environ),
        "FI_REPLICATE_SEED": str(index * stride),
        "FI_REPLICATE_INDEX": str(index),
    }


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
    passes here and falls through to the runtime comparison, which can only
    call it deterministic. An unreadable file returns ``True``, because silence
    is not evidence of a fault and the runtime check still runs.
    """
    try:
        return "FI_REPLICATE_SEED" in code_path.read_text(
            encoding="utf-8", errors="replace",
        )
    except OSError:
        return True


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
        "Return the whole script in `code`, one sentence in `patch_summary`, and "
        "leave `give_up_reason` empty."
    )


# Stands where the traceback would be in the ``execute_reflect`` prompt, for the
# one repair a script that ignores ``FI_REPLICATE_SEED`` is offered before it
# has run (see ``Engine._repair_ignored_replicate_seed``).
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
    "Return the whole script in `code`, one sentence in `patch_summary`, and "
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
        }
    return out


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


def _replicate_result_intervals(state: QuestState) -> dict[str, dict[str, float]]:
    """Each result's mean and 95% CI over the seeds, by dotted path, for the
    checks that compare the paper with the results: a paper reports the mean
    and its interval, and seed 0's ``RESULT_JSON`` holds neither. Only the
    results that differ between seeds; the rest are seed 0's values."""
    replicates = state.get("result_json_replicates") or []
    if len(replicates) < 2:
        return {}
    aggregate = _aggregate_result_json_replicates(replicates, assertions=_replicate_assertions(state))
    return {
        path: {key: stats[key] for key in ("mean", "ci_lower", "ci_upper") if stats.get(key) is not None}
        for path, stats in aggregate.items() if stats.get("std")
    }


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
                "label": line.get("label"), "kind": line.get("kind"),
                "x": x, "mean": mean, "lower": lower, "upper": upper,
            })
        panels.append({
            **{key: panel.get(key) for key in ("grid", "title", "xlabel", "ylabel", "xscale", "yscale", "legend")},
            "lines": lines,
        })
    if not varies:
        return None
    return {
        "file": name, "size": runs[0].get("size"), "suptitle": runs[0].get("suptitle") or "",
        "n": len(runs), "axes": panels,
    }


def _result_comparison_stats(
    replicates: list[dict[str, Any]], *, max_effect_sizes: int = 24,
    max_depth: int = 6, assertions: list[Any] | None = None,
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
    them would be noise. The ``by_*`` level itself is always a factor.

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
            depth == 0 or len({_shape(nodes[0][s]) for s in strata}) == 1
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
    """
    if not panel:
        return {"verdict": fallback_verdict, "score": 3,
                "rigor_score": 3, "depth_score": 3,
                "agreement": "unanimous", "strengths": [],
                "weaknesses": [], "suggestions": [], "blocking": "",
                "must_flag_hits": []}

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

    return {
        "verdict": verdict, "score": median,
        "rigor_score": rigor_median, "depth_score": depth_median,
        "agreement": agreement,
        "strengths": strengths, "weaknesses": weaknesses,
        "suggestions": suggestions, "blocking": blocking,
        "must_flag_hits": mfh_out,
    }


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
    if is_survey or ev.get("verdict") not in ("insufficient", "broaden"):
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


# Must-flag hits that are problems with the paper's text, not with the study:
# fixing one means writing the paper again, not running the experiment again.
# ``figure_missing`` is one: the figure exists, the paper left it out.
_TEXT_ONLY_HITS = frozenset({
    "unsupported_claim", "figure_caption", "figure_missing", "citations_unchecked", "over_page_limit",
    # A statistic the paper describes as something the run did not compute:
    # the numbers are right and the words around them are wrong, so the paper
    # is written again and the experiment is left alone.
    "mislabelled_statistic",
    # A number the paper prints that nothing in the run accounts for. The
    # experiment ran and recorded its results; the paper quotes a figure none
    # of them holds, so what needs rewriting is the prose, not the experiment.
    "unsourced_number",
})

# A paper with a page limit: the review renders each draft the way paper.pdf
# is rendered and counts its pages. A draft over the limit is sent back to be
# shortened at most this many times, and those rewrites do not use
# ``engine.max_iterations``.
_PAGE_LIMIT_REWRITES = 2
_PAGE_LIMIT_HIT = "over_page_limit"
# Words a full page holds and the share of a page one figure takes, from the
# graded SIR papers at the 1 in layout. The page-limit layout holds more, so a
# budget worked out from these leaves room.
_WORDS_PER_PAGE = 450
_FIGURE_PAGE_SHARE = 0.37


def _only_page_limit_hits(hits: list[Any]) -> bool:
    """True when every must-flag hit says the draft is over the page limit."""
    return bool(hits) and all(_hit_name(hit) == _PAGE_LIMIT_HIT for hit in hits)


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


def _page_limit_hit(pages: int, limit: int, words: int) -> str:
    """The forced hit for a draft over the page limit, which is also what the
    writer reads under "Must fix"."""
    return (
        f"over_page_limit: the rendered PDF is {pages} pages; the limit is {limit}. "
        f"Cut about {words} words, keep every figure and all numbers; shorten "
        "background and discussion first"
    )


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


def _hit_name(hit: Any) -> str:
    """The identifier a must-flag hit starts with, in lower case and without a
    plural s; empty when it starts with none."""
    m = _HIT_NAME_RE.match(str(hit))
    return m.group(1).lower().removesuffix("s") if m else ""


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

# A results key short enough to double as an English word ("mean", "n",
# "ci_upper") would match review prose by accident; a compound one
# ("deterministic_final_size") is never written by chance.
_RESULT_KEY_MIN_LEN = 8


def _nested_key_names(value: Any) -> set[str]:
    """Every key name in a nested results structure, at any depth."""
    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            names.add(str(key))
            names |= _nested_key_names(child)
    elif isinstance(value, list):
        for child in value:
            names |= _nested_key_names(child)
    return names


def _rerun_evidence(review: dict[str, Any], state: QuestState) -> str:
    """What the review's must-fix evidence names that only running the
    experiment again can fix — ``""`` when it names nothing of the sort.

    A must-flag hit is a *short identifier* (``agents/review.md`` asks for one),
    so what it is about is spelled out in the review's own prose: its
    ``blocking`` sentence, its ``weaknesses`` and its ``suggestions``. That
    prose is matched against names this quest actually produced, not against a
    list of English words — a key of ``result_json`` (a value the run computed),
    a file name in ``figure_records`` (a figure's underlying data), or the
    experiment file. "ensure the solver is correctly integrated so the
    'deterministic_final_size' is not reported as 0.0" names a results key and
    no rewrite can fix it; "remove the attribution to [2]" and "say in the
    caption that this is one seed" name none, and stay a rewrite.

    The advisory checks are deliberately not read: every
    ``numeric_oracle_warnings`` finding quotes a result path, and those findings
    are advisory precisely because a pattern match over prose misreads a DOI
    often enough that a forced re-run costs more than a flagged number. Nor are
    the hits themselves: they are short identifiers, and the ones the engine
    forces quote the figure they checked (``figure_caption: the caption of
    figures/x.png names …``, ``figure_missing: figures/x.png``), which would
    read as a problem with that figure's data when it is a problem with the
    paper's words.
    """
    if not state.get("code"):
        return ""  # no experiment to run again (an analyze-only or survey quest)
    names = {
        name for name in _nested_key_names(state.get("result_json") or {})
        if "_" in name and len(name) >= _RESULT_KEY_MIN_LEN
    }
    names |= {str(name) for name in state.get("figure_records") or {}}
    names.add("experiment.py")
    prose = " ".join(
        item
        for field in ("blocking", "weaknesses", "suggestions")
        for item in _review_items(review.get(field))
    ).lower()
    return next((name for name in sorted(names) if name.lower() in prose), "")


def _review_sends_the_experiment_back(review: dict[str, Any], state: QuestState) -> str:
    """What a review that can only be answered by running the experiment again
    names — ``""`` when this review is not one.

    Only a review whose must-flags would otherwise be answered by rewriting the
    paper is considered. A hit that already sends the quest back to ``design``
    (``circular_evaluation`` and the rest) re-runs the experiment on its way
    through ``implement`` anyway, and a flawed design is not fixed by writing
    the same experiment again. A draft that is merely over the page limit is
    not one either: it has its own route and its own counter.
    """
    hits = review.get("must_flag_hits") or []
    if not hits or _only_page_limit_hits(hits) or not _hits_need_only_a_rewrite(hits):
        return ""
    return _rerun_evidence(review, state)


def _rerun_directive(review: dict[str, Any], named: str) -> str:
    """What the implement prompt is told when a review sent the experiment
    back: what the review named, and the review's own must-fix prose."""
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
        "experiment so that it is computed correctly and reported in RESULT_JSON, "
        "and keep everything the design asks for.",
        *(f"  - {note}" for note in notes),
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


def _format_review_for_writer(state: QuestState) -> str:
    """The write prompt's ``$review_feedback``: what the review of the previous
    draft asks for (its must-fix hits, the captions that describe what their
    figure does not show, its weaknesses and suggestions, and the claims the
    claim check found unsupported), and every round of the user's feedback. A
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
        ("Captions that describe what their figure does not show", "figure_caption_warnings"),
        ("Weaknesses", "weaknesses"),
        ("Suggestions", "suggestions"),
    ):
        items = _review_items(review.get(key))
        if items:
            lines += [f"{heading}:", *(f"  - {item}" for item in items)]
    unsupported = _review_items((state.get("claim_grounding") or {}).get("unsupported"))
    if unsupported:
        lines += [
            "Claims that neither this study's results nor a cited source backs:",
            *(f"  - {claim}" for claim in unsupported),
        ]
    rounds = [
        h for h in state.get("feedback_history") or []
        if isinstance(h, dict) and str(h.get("text") or "").strip()
    ]
    if rounds:
        lines += [
            "The user's feedback (honour every round):",
            *(f"  - (round {h.get('iteration', '?')}) {str(h['text']).strip()}" for h in rounds),
        ]
    return "\n".join(lines) or "(none — first draft)"


def _format_cross_check(state: QuestState) -> str:
    """Render the per-finding cross-paper-check results as a
    bulleted block for the write prompt's `$cross_check_block`."""
    checks = state.get("cross_check") or []
    if not checks:
        return "(none — cross-check disabled or no key findings)"
    lines: list[str] = []
    for c in checks:
        lines.append(f"### Finding: {c.get('finding', '(?)')}")
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


def _figure_list_for_prompt(state: QuestState) -> str:
    """The figure list for a writer/analysis prompt. Plain charts are listed
    by path; license-clean web figures (``figure_credits``) additionally
    carry their caption + source + license, and are flagged ILLUSTRATIVE so
    the writer credits them and does NOT present them as the quest's own
    results."""
    figs = state.get("figures") or []
    if not figs:
        return "(none)"
    credits = {c.get("file"): c for c in (state.get("figure_credits") or [])}
    records = state.get("figure_records") or {}
    n_seeds = _replicate_seed_count(state)
    lines: list[str] = []
    for f in figs:
        c = credits.get(f)
        if c:
            lines.append(
                f"- figures/{f} — ILLUSTRATIVE (a sourced figure, not your "
                f"own data): {str(c.get('caption', ''))[:140]} "
                f"[the caption MUST credit: Source {c.get('source_url', '')}, "
                f"{c.get('license', '')}]"
            )
        else:
            lines.append(f"- figures/{f}" + _figure_record_note(records.get(f), n_seeds=n_seeds))
    if any((records.get(f) or {}).get("replicate_mean") for f in figs):
        lines.append(
            "A figure drawn as the mean of several seeds shows each line at its mean, "
            "shaded with its 95% confidence interval, and its caption says so."
        )
    if any((records.get(f) or {}).get("single_seed") is not None for f in figs):
        lines.append(
            "A figure that shows replicate seed 0 only could not be drawn as the mean over "
            f"{_replicate_seeds(n_seeds)}. A replicate seed repeats the whole experiment, so the "
            "figure draws every run seed 0 made: the text and caption about it quote seed 0's "
            f"values, or say it shows a single replicate ({_seed_label(0, n_seeds)}), not the "
            "means over the seeds."
        )
    if any(_hidden_series(records.get(f)) for f in figs):
        lines.append(
            "A series marked FLAT is drawn at one value on its axis, and one NOT SHOWN "
            "is not visible at all: a caption must describe what its figure shows, so "
            "it cannot describe how such a series changes."
        )
    return "\n".join(lines)


def _read_primary_figures(
    figures_dir: Path, records_dir: Path, figures: list[str],
) -> dict[str, tuple[bytes, bytes | None]]:
    """Each figure file the primary run drew, with the record of what it draws
    (``None`` when it has none), by file name, before the replicate runs draw
    over them."""
    kept: dict[str, tuple[bytes, bytes | None]] = {}
    for name in figures:
        try:
            image = (figures_dir / name).read_bytes()
        except OSError:
            continue
        try:
            record: bytes | None = (records_dir / f"{Path(name).stem}.json").read_bytes()
        except OSError:
            record = None
        kept[name] = (image, record)
    return kept


def _restore_primary_figures(
    figures_dir: Path, records_dir: Path, kept: dict[str, tuple[bytes, bytes | None]],
    *, redrawn: Any,
) -> list[str]:
    """Put back the primary run's file and record of every kept figure that is
    not in ``redrawn``, so it shows seed 0 rather than the last replicate.
    Returns the figures put back."""
    restored: list[str] = []
    for name, (image, record) in kept.items():
        if name in redrawn:
            continue
        target = records_dir / f"{Path(name).stem}.json"
        try:
            figures_dir.mkdir(parents=True, exist_ok=True)
            (figures_dir / name).write_bytes(image)
            if record is None:
                # A replicate's record would describe a figure no longer on disk.
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(record)
        except OSError:
            continue
        restored.append(name)
    return restored


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
            records[name] = data
    return records


def _hidden_series(record: dict[str, Any] | None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """``(axes, series)`` for each labelled series a figure draws flat or not at all."""
    return [
        (ax, s)
        for ax in ((record or {}).get("axes") or []) if isinstance(ax, dict)
        for s in (ax.get("series") or []) if isinstance(s, dict) and s.get("shows") != "yes"
    ]


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
                elif item.get("check") == "legend_over_data":
                    names = [_one_line(n, 30) for n in (item.get("legend") or [])[:4]]
                    more = ", ..." if len(item.get("legend") or []) > 4 else ""
                    panel = f'in the panel "{_one_line(item.get("axes_title"))}", ' if item.get("axes_title") else ""
                    found.append(
                        f"figures/{name}: {panel}the legend ({', '.join(names)}{more}) is drawn over "
                        "the lines or points it labels"
                    )
    except Exception:  # noqa: BLE001
        return []
    return found


def _figure_overlap_directive(findings: list[str]) -> str:
    """Stands where the traceback would be in the ``execute_reflect`` prompt, for
    the one redraw a run whose figures overlap is offered (see
    ``Engine._node_execute_reflect``)."""
    return (
        "FIGURE LAYOUT: the script exited 0, printed its RESULT_JSON and drew its figures, "
        "and those results stand: the account of a crash above does not apply. But the saved "
        "figures have text or a legend drawn over what a reader needs, measured on the "
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
        "fig.tight_layout(rect=(0, 0, 1, 0.94)), or a constrained layout).\n\n"
        "Do NOT change any computation, data, parameter, use of the random seed, printed value, "
        "the final RESULT_JSON line, the file name of any figure, or what any panel plots: only "
        "positions, sizes and the number of legend columns. Return the whole script in `code`, one "
        "sentence in `patch_summary`, and leave `give_up_reason` empty."
    )


def _figure_record_note(record: dict[str, Any] | None, *, n_seeds: int | None = None) -> str:
    """What a figure draws, for the writer: each panel's title, y axis and series.
    ``n_seeds`` is how many replicate seeds ran, when known."""
    panels = []
    for ax in (record or {}).get("axes") or []:
        if not isinstance(ax, dict):
            continue
        bits = [f'"{ax["title"]}"'] if ax.get("title") else []
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
            span = f'{s["min"]:.3g} to {s["max"]:.3g}' if "min" in s else "no points"
            shows = s.get("shows")
            tail = "" if shows == "yes" else ", FLAT on this axis" if shows == "flat" else ", NOT SHOWN"
            series.append(f"{s.get('label')} {span}{tail}")
        if series:
            bits.append("series: " + "; ".join(series))
        if bits:
            panels.append(", ".join(bits))
    note = (" — " + " | ".join(panels)) if panels else ""
    mean_of = ((record or {}).get("replicate_mean") or {}).get("n")
    single_seed = (record or {}).get("single_seed")
    if mean_of:
        note += f" — each line is the mean of {mean_of} seeds, shaded with its 95% confidence interval"
    elif single_seed is not None:
        note += f" — shows {_single_seed_note(single_seed, n_seeds)}"
    return note


_PAPER_IMAGE_RE = re.compile(r"!\[(?P<alt>(?:[^\[\]]|\[[^\[\]]*\])*)\]\((?P<src>[^)\s]+)")


def _embedded_figure_names(paper_md: str) -> set[str]:
    """The figure files ``paper_md`` embeds, by file name."""
    return {Path(m.group("src")).name for m in _PAPER_IMAGE_RE.finditer(paper_md or "")}


def _planned_figure_order(state: QuestState) -> list[str]:
    """The figures the design planned and the run drew, in the order planned.

    A figure a later experiment version superseded is not one of them: every
    ``execute`` empties ``figures/`` first (``_clear_stale_figures``) and
    ``state["figures"]`` is the scan of what the run then drew, so a plan entry
    naming a figure this run did not draw matches nothing here.
    """
    planned = (state.get("design") or {}).get("figures_planned")
    planned_names = [Path(str(f)).name for f in planned if str(f).strip()] if isinstance(planned, list) else []
    produced = {Path(str(f)).name for f in (state.get("figures") or [])}
    return [name for name in dict.fromkeys(planned_names) if name in produced]


def _planned_figures_left_out(paper_md: str, state: QuestState) -> list[str]:
    """Figures the design planned and the run drew that the paper leaves out,
    by file name, in the order the design planned them.

    The write prompt says to include every available figure, and nothing
    checked it: 3 of 12 real papers left out at least one, and one of them
    kept 1 of its 3. A planned figure the run did not produce is the writer's
    to describe in prose, and a figure nobody planned may stay out, so
    neither counts here.
    """
    embedded = _embedded_figure_names(paper_md)
    return [name for name in _planned_figure_order(state) if name not in embedded]


def _missing_planned_figures(paper_md: str, state: QuestState) -> list[str]:
    """The review's forced hit for each figure the paper leaves out. The write
    node puts these back before the review sees the draft, so a hit here means
    the paper lost a figure some other way."""
    return [
        f"figure_missing: the paper leaves out figures/{name}, which the design planned and "
        "the run drew; include it with a numbered caption and discuss it in the text"
        for name in _planned_figures_left_out(paper_md, state)
    ]


# A figure the prose names: "Figure 2 shows", "Figures 1 and 2", "Fig. 3".
_FIGURE_MENTION_RE = re.compile(r"(?<![A-Za-z])(?:figures?|figs?\.?)\s*(\d+)", re.IGNORECASE)
_MD_FENCE_LINE_RE = re.compile(r"^\s*(```|~~~)")


def _figure_caption_text(
    name: str, record: dict[str, Any] | None, *, n_seeds: int | None,
) -> str:
    """The caption for a figure the engine places in the paper itself: what the
    figure's own record says it draws.

    Written from the panel titles and never from the series labels, so a
    caption the engine wrote cannot name a series its figure draws flat or not
    at all. That is a forced ``figure_caption`` hit at review time, which would
    spend the very iteration this repair exists to save. The record is checked
    all the same: the first wording it does not flag is the one used.
    """
    titles: list[str] = []
    for ax in (record or {}).get("axes") or []:
        if not isinstance(ax, dict):
            continue
        title = " ".join(str(ax.get("title") or "").replace("[", "").replace("]", "").split())
        if title and title not in titles:
            titles.append(title)
    stem = " ".join(Path(name).stem.replace("_", " ").replace("-", " ").split())
    if len(titles) > 3:
        # A grid of panels: its titles are the settings each panel was drawn
        # at, a list rather than a sentence about the figure, and the file's
        # own name says what the grid shows.
        joined = f"{stem} ({len(titles)} panels)"
    else:
        joined = "; ".join(titles)
        if len(joined) > 200:
            joined = joined[:200].rsplit(" ", 1)[0]
    tail = ""
    mean_of = ((record or {}).get("replicate_mean") or {}).get("n")
    single_seed = (record or {}).get("single_seed")
    if mean_of:
        tail = f" Each line is the mean of {mean_of} seeds, shaded with its 95% confidence interval."
    elif single_seed is not None:
        tail = f" Shows {_single_seed_note(single_seed, n_seeds)}."
    # A figure whose record the sandbox could not write has no panel title to
    # use, so the file's own name describes it; the bare number is the last
    # resort, when even that would name a series the figure hides.
    for description in (*(d for d in (joined, stem) if d), ""):
        caption = ((description.rstrip(" .;,") + "." if description else "") + tail).strip()
        if not _figure_caption_findings(f"![{caption}](figures/{name})", {name: record}):
            return caption
    return ""


def _figure_numbers_to_place(
    paper_md: str, state: QuestState, missing: list[str],
) -> dict[str, int]:
    """The number each left-out figure takes: its place in the design's plan,
    or the next free one when a figure the draft did embed already carries it."""
    order = _planned_figure_order(state)
    used = {
        int(m.group(1))
        for image in _PAPER_IMAGE_RE.finditer(paper_md or "")
        for m in [_FIGURE_MENTION_RE.search(image.group("alt"))] if m
    }
    numbers: dict[str, int] = {}
    nxt = max(used, default=0) + 1
    for name in missing:
        number = order.index(name) + 1 if name in order else 0
        if number <= 0 or number in used:
            number = nxt
        used.add(number)
        nxt = max(nxt, number + 1)
        numbers[name] = number
    return numbers


def _figure_insertion_line(lines: list[str], number: int) -> int:
    """Where the figure numbered ``number`` goes: after the paragraph whose
    prose first names it, else before the paper's source lists, else at the
    end. Code fences and the captions of figures already in the paper are not
    prose and are skipped."""
    fence: str | None = None
    for i, line in enumerate(lines):
        fence_line = _MD_FENCE_LINE_RE.match(line)
        if fence_line:
            fence = None if fence == fence_line.group(1) else (fence or fence_line.group(1))
            continue
        if fence is not None or _PAPER_IMAGE_RE.search(line):
            continue
        if any(int(m.group(1)) == number for m in _FIGURE_MENTION_RE.finditer(line)):
            end = i
            while end + 1 < len(lines) and lines[end + 1].strip():
                end += 1
            return end + 1
    for i, line in enumerate(lines):
        if _SOURCE_LIST_HEADING_RE.match(line):
            return i
    return len(lines)


def _place_missing_figures(markdown: str, state: QuestState) -> tuple[str, list[str]]:
    """``markdown`` with every figure the design planned and the run drew that
    the draft left out put back, and the names of the ones put back.

    The write prompt tells the writer to include every figure it is given, and
    a draft that ignores it was caught only at review time — which costs one of
    the two iterations, and catches nothing at all on the last one, so real
    runs delivered papers that discuss "Figure 1" through "Figure 3" with no
    figure in them while the images sat in ``figures/``. Which figures are
    missing is a set difference rather than a reading of prose, so the engine
    puts them back itself, the way it writes the source lists itself.

    Each figure goes after the paragraph that already discusses it by number,
    which is where the writer would have put it: the drafts that dropped every
    figure still said "Figure 1 shows ..." of each one. A figure whose number
    the prose never names goes before the source lists, so it is still in the
    paper. Idempotent: a draft with every planned figure is returned unchanged.
    """
    missing = _planned_figures_left_out(markdown, state)
    if not missing:
        return markdown, []
    records = state.get("figure_records") or {}
    n_seeds = _replicate_seed_count(state)
    numbers = _figure_numbers_to_place(markdown, state, missing)
    lines = markdown.split("\n")
    for name in missing:
        caption = _figure_caption_text(name, records.get(name), n_seeds=n_seeds)
        label = f"**Figure {numbers[name]}.**"
        embed = f"![{label} {caption}](figures/{name})" if caption else f"![{label}](figures/{name})"
        at = _figure_insertion_line(lines, numbers[name])
        before_sources = at < len(lines) and bool(_SOURCE_LIST_HEADING_RE.match(lines[at]))
        lines[at:at] = [embed, ""] if before_sources else ["", embed]
    return "\n".join(lines), missing


def _plain_words(text: str) -> str:
    return " ".join(re.sub(r"[_\-*`]+", " ", text.lower()).split())


# What a caption says of a series that lies flat, which is what its figure shows.
_FLAT_WORDS_RE = re.compile(
    r"\b(?:flat|constant|unchanged|negligible|indistinguishable|invisible|not visible|overlap\w*"
    r"|at this scale|(?:at|near|around|close to) zero)\b"
)
# A caption's clauses: "Euler starts flat, while RK4 climbs" says nothing flat of RK4.
_CAPTION_CLAUSE_RE = re.compile(r"[.;:,]\s|\s(?:while|whereas|but)\s")


def _panels_where_flat_and_where_not(
    record: dict[str, Any] | None, label: str,
) -> tuple[list[str], list[str]]:
    """For one series label: the panels of a multi-panel figure that draw it flat
    (with its value) and the panels that draw it varying (with its range).

    Both are empty for a one-panel figure, and for a series every panel draws flat,
    because then "flat" is true of the whole figure and there is nothing to tell
    the panels apart by."""
    axes = [ax for ax in ((record or {}).get("axes") or []) if isinstance(ax, dict)]
    if len(axes) < 2:
        return [], []
    flat: list[str] = []
    varies: list[str] = []
    for index, ax in enumerate(axes, start=1):
        title = str(ax.get("title") or f"panel {index}")
        for s in ax.get("series") or []:
            if not isinstance(s, dict) or s.get("label") != label or "min" not in s:
                continue
            if s.get("shows") == "flat":
                flat.append(f'"{title}" ({s["min"]:.3g})')
            elif s.get("shows") == "yes":
                varies.append(f'"{title}" ({s["min"]:.3g} to {s["max"]:.3g})')
    return (flat, varies) if flat and varies else ([], [])


# What a caption says when it calls a series flat: the reverse of ``_FLAT_WORDS_RE``,
# which lists what excuses a caption for naming a series the figure draws flat. Only
# claims of a constant value count here ("negligible", "overlap" and "at zero" do not),
# and "horizontal" as an axis ("on the horizontal axis") is not a claim about a series.
_SAYS_FLAT_RE = re.compile(
    r"\b(?:flat|constant|unchanged"
    r"|horizontal(?!\s+(?:axis|axes|scale|position|direction|extent|bars?|error)))\b"
)
# A claim that stops short of "flat" or denies it: "nearly flat", "approximately
# constant", "not flat", "non-constant" ("-" reads as a space, as in ``_plain_words``).
_SOFTENED_FLAT_RE = re.compile(
    r"\b(?:nearly|almost|approximately|roughly|essentially|virtually|practically"
    r"|effectively|quasi|not|non|never|hardly|barely|no longer)"
    r"(?:\s+\w+){0,2}\s+(?:flat|constant|unchanged|horizontal)\b"
    r"|n['’]t(?:\s+\w+){0,2}\s+(?:flat|constant|unchanged|horizontal)\b"
    # "near flat", but not "near the flat line", which places a flat line and hedges nothing.
    r"|\bnear\s+(?:flat|constant|unchanged|horizontal)\b"
)
# A clause that says a series is flat for a part of its run ("starts flat", "flat until
# t = 5", "flat only at R0 = 3") is not calling the whole series flat.
_PARTIAL_FLAT_RE = re.compile(
    r"\b(?:starts?|started|starting|begins?|beginning|initially|at first|early|until|before"
    r"|after|then|later|eventually|briefly|only|except|plateaus?|beyond)\b"
)
# The share of a y axis's span a series must cover for its figure to draw it varying.
# The recorder calls a series "flat" under 1%; a caption that calls a line flat is
# wrong once the eye can read a slope, which the sample put at about 5%.
_VARYING_SHARE = 0.05


def _share_of_axis(ax: dict[str, Any], s: dict[str, Any]) -> float:
    """The share of ``ax``'s y span that series ``s`` covers: the recorder's own test
    for "flat" (``core/plot_style.py``), clipped to the axis and measured in log
    units on a log axis. 0.0 when the record cannot say (no limits, a malformed
    range, a log axis reaching zero)."""
    try:
        lo, hi = (float(v) for v in ax["ylim"])
        low, high = float(s["min"]), float(s["max"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(v) for v in (lo, hi, low, high)):
        return 0.0
    bottom, top = min(lo, hi), max(lo, hi)
    low, high = max(low, bottom), min(high, top)
    pos: Callable[[float], float] = float
    if ax.get("yscale") == "log":
        if bottom <= 0:
            return 0.0
        pos = math.log10
    span = pos(top) - pos(bottom)
    return (pos(high) - pos(low)) / span if span > 0 and high > low else 0.0


def _panels_drawn_varying(
    axes: list[dict[str, Any]], label: str,
) -> tuple[list[tuple[str, float, float]], list[str]]:
    """For one series label: ``(title, min, max)`` of each panel that draws it varying
    over more than ``_VARYING_SHARE`` of the axis, and the titles of the panels that
    do not (flat there, or under 5%).

    A series the recorder marks min = max = 0 "yes" has a range of 0 and is never
    varying here. That is a measurement artefact and not a drawing: the recorder
    reads ``ax.hlines`` and ``fill_between`` through ``get_offsets()`` and gives 18 of
    918 series in the stored quests that range, so the rule must stay silent on them."""
    varying: list[tuple[str, float, float]] = []
    calm: list[str] = []
    for index, ax in enumerate(axes, start=1):
        title = str(ax.get("title") or f"panel {index}")
        for s in ax.get("series") or []:
            if not isinstance(s, dict) or str(s.get("label")) != label:
                continue
            if s.get("shows") == "yes" and _share_of_axis(ax, s) > _VARYING_SHARE:
                varying.append((title, float(s["min"]), float(s["max"])))
            else:
                calm.append(title)
    return varying, calm


def _clause_names_a_panel(clause: str, titles: list[str]) -> bool:
    """True when ``clause`` names one of the panels ``titles`` by its title, ignoring
    case, spacing and punctuation ("R0 = 3" and "R₀=3" are the same panel).
    Titles of fewer than two letters or digits ("A") would match anything."""
    def squeeze(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", text).lower())

    body = squeeze(clause)
    return any(len(t) >= 2 and t in body for t in map(squeeze, titles))


def _flat_claim_findings(name: str, record: dict[str, Any] | None, caption: str) -> list[str]:
    """Findings for a caption that calls a series flat when its figure draws it varying.

    ``caption`` is the image's alt text as ``_plain_words`` leaves it. A clause counts
    when it names the series by its exact label, says flat / constant / unchanged /
    horizontal of it (the label's own words are not the claim: a series labelled
    "constant N" is not called constant by being named), and does not hedge or deny
    it ("nearly flat", "not flat") or confine it to part of the run ("starts flat"),
    and the figure draws that series varying over more than 5% of an axis in at least
    one panel. A clause that names a panel where the series is not varying is scoping
    its claim to that panel, which is what the finding asks for, so it is left alone.

    Exact labels only: a caption that names the series in other words ("flat
    deterministic predictions" for a series labelled "Deterministic ODE") is not read."""
    axes = [ax for ax in ((record or {}).get("axes") or []) if isinstance(ax, dict)]
    findings: list[str] = []
    seen: set[str] = set()
    for ax in axes:
        for s in ax.get("series") or []:
            raw = str(s.get("label") or "") if isinstance(s, dict) else ""
            if not raw or raw in seen:
                continue
            seen.add(raw)
            label = _plain_words(raw)
            named = rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])"
            if not label or not re.search(named, caption):
                continue
            varying, calm = _panels_drawn_varying(axes, raw)
            if not varying:
                continue
            for clause in _CAPTION_CLAUSE_RE.split(caption):
                if not re.search(named, clause):
                    continue
                said = re.sub(named, " ", clause)
                claim = _SOFTENED_FLAT_RE.sub(" ", said)
                if (
                    not _SAYS_FLAT_RE.search(claim) or _PARTIAL_FLAT_RE.search(said)
                    or _clause_names_a_panel(clause, calm)
                ):
                    continue
                head = f'figure_caption: the caption of figures/{name} calls "{raw}" flat, but the figure draws it varying'
                if len(axes) < 2:
                    ylabel = str(axes[0].get("ylabel") or "y")
                    findings.append(
                        f'{head} ({varying[0][1]:.3g} to {varying[0][2]:.3g}) on its axis "{ylabel}". '
                        "Say that it varies"
                    )
                else:
                    where = ", ".join(f'"{t}" ({lo:.3g} to {hi:.3g})' for t, lo, hi in varying)
                    findings.append(
                        f"{head} in {where}. Say that it varies there, or describe only the panels "
                        "where it is flat"
                    )
                break
    return findings


def _figure_caption_findings(paper_md: str, records: dict[str, Any]) -> list[str]:
    """Captions that name a series their figure does not show, or one it draws
    flat without saying so, and captions that call a series flat when the figure
    draws it varying (``_flat_claim_findings``).

    A series a multi-panel figure draws flat in only some panels is flagged with
    those panels named, and the panels where it varies listed. A finding that named
    no panel got the caption rewritten to call the series flat in EVERY panel, and
    the paper then said of the other panels what its figure does not show."""
    findings: list[str] = []
    for match in _PAPER_IMAGE_RE.finditer(paper_md or ""):
        name = Path(match.group("src")).name
        record = records.get(name)
        caption = f" {_plain_words(match.group('alt'))} "
        for ax, s in _hidden_series(record):
            label = _plain_words(str(s.get("label") or ""))
            named = rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])"
            if not label or not re.search(named, caption):
                continue
            if s.get("shows") == "flat" and any(
                re.search(named, clause) and _FLAT_WORDS_RE.search(clause)
                for clause in _CAPTION_CLAUSE_RE.split(caption)
            ):
                continue
            how = "draws it flat at one value" if s.get("shows") == "flat" else "does not show it"
            # A shared y axis is labelled on one panel only: take the first label there is.
            ylabel = next(
                (str(a.get("ylabel")) for a in [ax, *((record or {}).get("axes") or [])]
                 if isinstance(a, dict) and a.get("ylabel")),
                "y",
            )
            finding = (
                f'figure_caption: the caption of figures/{name} names "{s.get("label")}", but the '
                f'figure {how} on its axis "{ylabel}"'
            )
            if s.get("shows") == "flat":
                flat, varies = _panels_where_flat_and_where_not(record, str(s.get("label")))
                if flat:
                    finding += (
                        f" in only some panels: flat in {', '.join(flat)}, but varying in "
                        f"{', '.join(varies)}. Describe it as flat only where it is flat"
                    )
            if finding not in findings:
                findings.append(finding)
        for claim_finding in _flat_claim_findings(name, record, caption):
            if claim_finding not in findings:
                findings.append(claim_finding)
    return findings


def _format_figure_check(paper_md: str, state: QuestState) -> str:
    """The figure check for the review prompt's ``$figure_check_block``."""
    records = state.get("figure_records") or {}
    if not records:
        return "(no record of what the figures draw)"
    # The results are means over the seeds; these figures hold replicate seed 0 alone.
    n_seeds = _replicate_seed_count(state)
    seed_only = [
        f"figures/{name} shows {_single_seed_note(record['single_seed'], n_seeds)}: its caption "
        f"and the text about it must quote seed {record['single_seed']}'s values or say it shows "
        f"a single replicate ({_seed_label(record['single_seed'], n_seeds)})."
        for name, record in records.items()
        if isinstance(record, dict) and record.get("single_seed") is not None
    ]
    findings = _figure_caption_findings(paper_md, records)
    if not findings:
        return "\n".join([*seed_only, "No caption names a series its figure does not show."])
    return "\n".join([
        *seed_only,
        f"CAPTIONS THAT DESCRIBE WHAT THEIR FIGURE DOES NOT SHOW ({len(findings)}):",
        *(f"  - {f}" for f in findings),
        "",
        "You MUST add a `figure_caption` entry to must_flag_hits and set verdict=revise, "
        "asking for the caption to describe what the figure shows (or the figure to show "
        "what the caption describes).",
    ])


def _parse_json_lenient(
    text: str, *, node: str = "", _log_truncate_chars: int = 500,
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
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    # Find the first '{' and last '}' and try the slice.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            result = json.loads(candidate[start : end + 1])
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            _log_parse_failure(candidate, node, _log_truncate_chars)
            return None
    # No braces found at all — the model didn't return JSON-ish output.
    # That's just as much a parse failure as the slice-attempt-failed
    # case above, and worth logging at the same WARNING level.
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
        deps_match = _DEPS_LINE_RE.search(text[fence.end():])
        if deps_match:
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
            deps = [
                d.strip().strip("'\"")
                for d in raw.split(",")
                if d.strip().strip("'\"")
            ]
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


def _assertion_violations(state: "QuestState") -> list:
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

        design = dict(state.get("design") or {})
        # A skill knows its own valid domain, so a design that calls one
        # need not restate its bounds. Design assertions come first: a
        # design that deliberately narrows a skill's range keeps its say.
        extra = state.get("_skill_assertions") or []
        if extra:
            design["result_assertions"] = list(
                design.get("result_assertions") or []
            ) + list(extra)
        # The script rides along so a value capped exactly at a non-zero
        # bound is caught too; see core/plausibility.py for why.
        return plausibility.check_design(
            state.get("result_json") or {}, design, code=state.get("code") or "",
        )
    except Exception:  # noqa: BLE001 - a checker bug must not block a quest
        return []


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

    _walk(rj)
    if len(nums) < 2:
        return False
    return all(abs(n) <= 1e-12 for n in nums)


_PKG_TO_MODULE = {
    # pip package name -> import name when they differ. Conservative —
    # only fills in cases we've observed our `implement` node produce.
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "beautifulsoup4": "bs4",
    "pyyaml": "yaml",
}


# PEP 508 splits the package name from version specifiers / extras /
# markers on the first occurrence of any of these. We strip on this set
# rather than just `>=`/`==`/`<` so deps like `numpy!=1.26.0`,
# `pandas~=2.0`, `urllib3<2;python_version<"3.10"` all yield a clean name.
_DEP_NAME_BOUNDARY = re.compile(r"[\s;<>=!~\[]")
# A valid Python module identifier (or dotted import path). After
# pip→module remapping + dash→underscore substitution, the final token
# MUST match this; anything else gets dropped rather than splatted
# into `-c "import ..."` where it would SyntaxError.
_PY_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _deps_to_warmup_modules(deps: list[str]) -> str:
    """Convert a pip-style deps list into a comma-separated module list
    safe for `python -c "import a, b, c"`. Strips version pins / extras
    / environment markers (PEP 508), remaps known name-mismatched
    packages, and validates each token against `_PY_MODULE_RE` so a
    malformed dep can never produce invalid import syntax."""
    out: list[str] = []
    for d in deps:
        head = _DEP_NAME_BOUNDARY.split(d, 1)[0].strip()
        if not head or "/" in head:
            continue  # blank or URL/path dep
        module = _PKG_TO_MODULE.get(head.lower(), head.replace("-", "_"))
        if not _PY_MODULE_RE.match(module):
            # Anything that didn't reduce to a clean identifier gets
            # dropped silently rather than risk a SyntaxError in `-c`.
            continue
        out.append(module)
    return ", ".join(out)


_UNSANCTIONED_PROXY_PROVIDERS = frozenset({
    "github_copilot_cli",
    "github_copilot_vscode",
})

# Agentic CLIs that interpret prompts as user tasks (engage their own
# tool-using agent loop) instead of running stateless LLM inference.
# Their outputs include conversational replies like "Are you trying
# to debug X?" rather than the structured prompt-driven output FI's
# nodes expect. Empirically broken as chat backends for FI's pipeline.
_AGENTIC_CLI_PROVIDERS = frozenset({"copilot_cli"})

_PROXY_WARN_SHOWN: set[str] = set()


def _warn_if_unsanctioned_provider(name: str) -> None:
    """Print a one-time warning (per process) when the user picks a
    provider that's known-broken or risky for FI's pipeline. Two
    categories trigger:

    1. **`_UNSANCTIONED_PROXY_PROVIDERS`** — `github_copilot_cli` and
       `github_copilot_vscode` route through a third-party
       reverse-engineered `copilot-api` proxy. The proxy's own README
       warns: "Excessive automated or scripted use of Copilot ... may
       trigger GitHub's abuse-detection systems." Premium-request
       volume from an automated research loop is exactly the pattern
       that trips that detector.

    2. **`_AGENTIC_CLI_PROVIDERS`** — `copilot_cli`. The standalone
       Copilot CLI is an agentic tool: it interprets FI's node prompts
       as user coding tasks and replies conversationally (real symptom
       seen: paper.md filled with "Are you trying to debug X?",
       experiment.py reduced to the empty stub). Use `vscode_extension`,
       `claude_cli`, `codex_cli`, `gemini_cli`, or an HTTP-direct
       provider (`openai`, `gemini`, `ollama`, `vllm`) instead.

    Suppression: each category warns once per process; set the
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

    if name in _AGENTIC_CLI_PROVIDERS:
        _PROXY_WARN_SHOWN.add(name)
        logging.getLogger("frontier_insight").warning(
            "\n%s\n"
            "  provider=%r is an AGENTIC CLI, not a stateless chat backend.\n"
            "  It interprets FI's node prompts (\"You are the Implementation\n"
            "  node, output JSON ...\") as user tasks and replies\n"
            "  conversationally — e.g. asking which file you want edited,\n"
            "  or outputting a chat-style answer instead of the structured\n"
            "  payload FI's nodes need. Real symptoms observed in the wild:\n"
            "  paper.md filled with \"Are you trying to debug X?\",\n"
            "  experiment.py reduced to the empty stub.\n"
            "\n"
            "  Recommended provider for Copilot users:\n"
            "    - provider.name: vscode_extension  (uses vscode.lm.* via\n"
            "                                        the FI VSCode extension)\n"
            "  Other working chat-style CLIs:\n"
            "    - provider.name: claude_cli  (Claude --print --output-format text)\n"
            "    - provider.name: codex_cli   (codex exec --output-last-message)\n"
            "    - provider.name: gemini_cli  (gemini --yolo -o json)\n"
            "  Or HTTP-direct: openai, gemini, ollama, vllm.\n"
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
of papers — full text wasn't available from the open-web sources FI
tries (arXiv / OpenAlex / Crossref / Semantic Scholar / ...).

See **``../needs/WANTED_PAPERS.md``** for the ranked list (most relevant
first) with a download link for each. **Drop the PDFs into THIS directory**
(or anywhere under it — FI walks recursively). Then re-run:

```
fi --resume {quest_id}
```

The literature node will pick up the new files, extract their text,
and merge them into the existing literature list — the design /
write nodes then see real full text instead of bare abstracts.

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
        try:
            if suffix == ".pdf":
                content = _extract_pdf_text(p)
            else:
                content = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError) as e:
            log.warning("[literature] could not read %s: %r", p, e)
            continue
        content = (content or "").strip()
        if len(content) < 200:
            continue
        ident = content[:200]
        if ident in seen:
            continue
        seen.add(ident)
        merged.append({
            "content": content[:8000],  # cap to keep prompt manageable
            "metadata": {
                "source": "user_supplied",
                "filename": p.name,
                "path": str(p),
                "abstract_only": False,
            },
        })
        count += 1
    return merged, count


def _extract_pdf_text(path: Path) -> str:
    """Pull plain text out of a PDF via pypdf. Returns empty string on
    failure (caller decides whether to skip). pypdf is a soft
    dependency — when missing, returns the path name so the LLM at
    least knows the file exists."""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return f"[pypdf not installed; user-supplied paper at {path.name}]"
    try:
        reader = PdfReader(str(path))
        return "\n\n".join(
            (page.extract_text() or "").strip() for page in reader.pages
        )
    except Exception as e:  # noqa: BLE001
        # pypdf raises a zoo of exceptions on malformed PDFs.
        return f"[pypdf could not parse {path.name}: {e}]"


# Threshold below which a retrieved doc is treated as abstract-only
# (full text not available). Most arXiv / OpenAlex / Crossref / S2
# returns are abstracts in the 800–1400 char range; a real paper body
# is well above 5000 chars even when truncated. 1500 splits the two
# comfortably without over- or under-flagging.
_ABSTRACT_ONLY_CHAR_THRESHOLD = 1500


def _append_design_revision(
    state: "QuestState", design: dict, quest_root: Path, log: logging.Logger,
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
    """
    prior = list(state.get("design_history") or [])
    hypothesis = str((design or {}).get("hypothesis") or "").strip()

    if not prior:
        reason = "initial design, before any result existed"
    else:
        analysis = state.get("analysis") or {}
        next_step = str(analysis.get("next_step") or "").strip()
        verdict = str((state.get("review") or {}).get("verdict") or "").strip()
        if next_step in ("re_experiment", "broaden_lit"):
            reason = f"analyze.next_step={next_step} (results were seen)"
        elif verdict:
            reason = f"review verdict={verdict} (results were seen)"
        else:
            reason = "re-entered design (results were seen)"

    entry = {
        "revision": len(prior),
        "iteration": int(state.get("iteration", 0) or 0),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "post_hoc": bool(prior),
        "reason": reason,
        "hypothesis": hypothesis[:600],
    }
    history = prior + [entry]

    if entry["post_hoc"]:
        changed = hypothesis != str(prior[-1].get("hypothesis") or "")
        log.warning(
            "[design] revision %d recorded (%s)%s -- the paper's hypothesis "
            "is being set after results were seen; DESIGN_HISTORY.json keeps "
            "this auditable.",
            entry["revision"], reason,
            "; hypothesis TEXT CHANGED" if changed else "; hypothesis unchanged",
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
    """True when the source is freely downloadable without a subscription.

    arXiv / PMC / bioRxiv / medRxiv are open access by construction, and
    OpenAlex reports an explicit ``open_access`` flag we preserve on the doc.

    This is deliberately separate from :func:`_is_abstract_only`. That
    predicate answers "did we end up with only an abstract?", which stays
    true for an arXiv paper whose full-text fetch failed — the doc really is
    abstract-only. What changes is what the pause gate does with it: asking a
    person to hand-download an arXiv PDF is asking them to work around a
    FETCH failure (blocked network, proxy, TLS interception), not a paywall,
    and the request reads as nonsense to anyone who knows arXiv is free.
    """
    md = doc.metadata or {}
    if md.get("open_access") is True:
        return True
    if md.get("arxiv_id") or md.get("pmcid"):
        return True
    if str(md.get("doi") or "").startswith("10.1101/"):  # bioRxiv / medRxiv
        return True
    url = str(md.get("url") or "").lower()
    return any(
        host in url for host in
        ("arxiv.org", "ncbi.nlm.nih.gov/pmc", "biorxiv.org", "medrxiv.org")
    )


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
    if md.get("fetched_full_text") or md.get("content_quality") == "full_text":
        return False
    if md.get("source") in ("local_paper", "user_supplied"):
        return False
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


def _write_wanted_papers_md(
    quest_root: Path, ranked: list["RetrievedDoc"], *, topic: str,
    oa_unfetched: list["RetrievedDoc"] | None = None,
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
            "abstract; no open-access full text was reachable. Download the ones "
            "that matter — **most relevant first** — drop the PDFs into "
            "`inputs/papers/`, then re-run the quest. You don't have to get them "
            "all; even the top few sharpen the research.",
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
            "- **Status:** full text not retrieved (likely paywalled / "
            "subscription) — manual download needed"
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
        "Drop the PDFs into `inputs/papers/` (any subfolder works), then "
        f"`fi --resume {quest_root.name}`."
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
    )
    readme = papers_dir / "README.md"
    if not readme.exists():
        try:
            readme.write_text(
                _PAPERS_README.format(quest_id=quest_root.name),
                encoding="utf-8",
            )
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
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(f"[{quest_id[:24]}] %(message)s"))
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
