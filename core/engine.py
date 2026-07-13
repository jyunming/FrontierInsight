"""Frontier Insight research engine — async LangGraph DAG.

Real LLM-driven nodes, code generation + execution in a per-quest
venv, Axon-backed knowledge retrieval, and SQLite-checkpointed state for
resumability after stalls (LLM quotas, OS sleep, manual interrupt).

Each Engine instance is stateless w.r.t. process globals; N instances must
coexist in one process for the fleet runner.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import re
import shutil
import string
import time
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

from . import stats as _stats
from .config import (
    Config,
    NON_SCIENTIFIC_PAPER_FORMATS,
    SCIENTIFIC_PAPER_FORMATS,
)
from .execution import ExecutionResult, make_executor
from .knowledge import Knowledge, RetrievedDoc
from .protocol import derive_protocol, route_for_topic_type
from .provider import (
    LLMClient,
    PROXY_PROVIDERS,
    ProxySupervisor,
    resolve_endpoint_async,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents"

_FIGURE_SUFFIXES = frozenset({".png", ".svg", ".jpg", ".jpeg", ".pdf"})


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
    design: dict[str, Any]
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
    result_json: dict[str, Any]
    # Multi-seed replication: when ``engine.execute_replicates > 1``,
    # ``_node_execute`` runs the script N times with different seeds
    # and aggregates the N ``RESULT_JSON`` outputs into this list
    # (one dict per replicate). The primary ``result_json`` field
    # carries the first replicate so existing single-seed paths
    # downstream stay unchanged. ``_node_analyze`` reads this list
    # when present and aggregates numeric fields with mean ± std.
    result_json_replicates: list[dict[str, Any]]
    # Execute-repair loop counter + history. The reflect
    # node increments `exec_reflect_iter` and appends a one-line
    # record per attempt, so analyze/write/review can describe what
    # was fixed.
    exec_reflect_iter: int
    exec_reflect_history: list[dict[str, Any]]
    exec_give_up_reason: str
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
    # Claim grounding: which paper claims trace to evidence
    # (experiment / citation / unsupported), from the claim_check node.
    claim_grounding: dict[str, Any]
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
        )
        self.knowledge = Knowledge(config.knowledge)
        self._log = _quest_logger(self.quest_id, self.fi_dir)
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
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.quest_root / "figures").mkdir(parents=True, exist_ok=True)
            (self.quest_root / "code").mkdir(parents=True, exist_ok=True)
            (self.quest_root / "paper").mkdir(parents=True, exist_ok=True)
            self._log.info("starting quest %s", self.quest_id)
            # Pre-flight: if the user asked for paper_pdf, verify the
            # host can produce one BEFORE spending 15 minutes on LLM
            # calls only to discover at the end that pandoc / LaTeX are
            # missing. Always warn on missing prereqs; raise only when
            # ``output.require_pdf`` is True. See #55 for the
            # silent-skip incident that motivated both this check and
            # the ``paper_pdf_skipped.md`` diagnostic.
            self._preflight_paper_pdf()
            await self.executor.setup(self.quest_root)

            endpoint = await resolve_endpoint_async(self.config.provider, self.supervisor)
            self._log.info(
                "provider %s -> %s (%s)",
                self.config.provider.name, endpoint.base_url, endpoint.model,
            )
            self._client = LLMClient(
                endpoint,
                cli_timeout_s=self.config.provider.cli_timeout_s,
                cli_inactivity_timeout_s=(
                    self.config.provider.cli_inactivity_timeout_s
                ),
                node_cli_timeout_s=self.config.provider.node_cli_timeout_s,
                node_model_fallbacks=(
                    self.config.provider.node_model_fallbacks
                ),
                heartbeat_cb=self._llm_heartbeat,
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
        except Exception as exc:
            # Surface the failure as a quest-directory diagnostic the
            # user can discover by opening the quest folder, rather than
            # leaving an empty quest dir whose only breadcrumb is a
            # traceback buried in ``<quest_root>/.fi/launch.log``. Mirrors the
            # ``paper_pdf_skipped.md`` contract from the paper generator.
            #
            # Re-raise unconditionally — this handler is for diagnostics
            # only, NOT for swallowing errors. The caller (launch.py)
            # still surfaces the exception in stderr / its own exit code.
            #
            # The diagnostic-write itself is wrapped in its own
            # try/except: a failure to write the diagnostic must NEVER
            # mask the original exception (the user wants to see the
            # real error, not "could not open file for diagnostic
            # writing"). ``CancelledError`` and ``KeyboardInterrupt``
            # are NOT caught here (they inherit from BaseException, not
            # Exception) so user-initiated cancellation skips the
            # diagnostic — those are not "the quest broke" events.
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
            _close_quest_logger(self.quest_id)

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
        g.add_edge("literature", "design")
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
            {"revise": "design", "done": END, "human_feedback": "human_feedback"},
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
        #    rely on to actually take effect.
        # 2. ``human_feedback_gate == "after_review"`` routes through
        #    the human-feedback node so the user gets a final say.
        # 3. Otherwise fall through to the legacy verdict-driven routing.
        review = state.get("review") or {}
        must_flag = review.get("must_flag_hits") or []
        if must_flag and state.get("iteration", 0) < self.config.engine.max_iterations:
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
                "no_simulation_resolved": self.config.engine.no_simulation,
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
                    "no_simulation_resolved": self._resolve_no_simulation_from_clarify(answers),
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
            no_sim = self._resolve_no_simulation_from_clarify(overrides)
            self._log_topic_shape_mismatch(overrides, no_simulation_resolved=no_sim)
            return {
                "clarify_questions": {},
                "clarify_answers": overrides,
                "clarify_done": True,
                "no_simulation_resolved": no_sim,
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
            no_sim = self._resolve_no_simulation_from_clarify(answers)
            self._log_topic_shape_mismatch(answers, no_simulation_resolved=no_sim)
            return {
                "clarify_questions": questions,
                "clarify_answers": answers,
                "clarify_done": True,
                "no_simulation_resolved": no_sim,
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
        no_sim = self._resolve_no_simulation_from_clarify(answers)
        self._log_topic_shape_mismatch(answers, no_simulation_resolved=no_sim)
        return {
            "clarify_questions": questions,
            "clarify_answers": answers,
            "clarify_done": True,
            "no_simulation_resolved": no_sim,
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
        if this_iter > 1 and state.get("design"):
            hypothesis = str(state["design"].get("hypothesis") or "")[:200]
            query = (chosen.get("title") or "") + " " + hypothesis + " " + state["topic"][:160]
            self._log.info(
                "[literature] iteration=%d (broaden_lit re-entry); query incorporates design.hypothesis",
                this_iter,
            )
        else:
            query = (chosen.get("title") or "") + " " + state["topic"][:200]
        docs = await self.knowledge.asearch(
            query.strip(),
            top_k=self.config.knowledge.top_k,
            # The literature node is the one path that explicitly wants
            # broad external retrieval when Axon misses — pass the
            # config's external cap so a web miss returns ~20 abstracts
            # instead of being silently capped at the Axon top_k.
            external_top_k=self.config.knowledge.external_top_k,
            chosen_idea=chosen,
            chat_fn=functools.partial(self._chat_messages, node="source_router"),
        )
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
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for entry in (*prior, *new_entries):
            md = entry.get("metadata") or {}
            ident = (
                str(md.get("doi") or "").strip()
                or str(md.get("url") or "").strip()
                or (entry.get("content") or "")[:200]
            )
            if ident and ident in seen:
                continue
            if ident:
                seen.add(ident)
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
            needed = [d for d in docs if _is_abstract_only(d)]
            if needed:
                _write_paper_need_stubs(
                    self.quest_root, needed, self._log, query=_lit_query(state),
                )
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
            literature_block=_format_lit_from_state(state, **self._lit_kwargs(state)),
            review_feedback=review_feedback or "(none — first iteration)",
            timeout_s=str(self.config.execution.timeout_s),
            clarify_block=_format_clarify(state),
            study_mode_directive=(
                _NO_SIM_DESIGN_DIRECTIVE
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
        cfg_value = self.config.pauses.supply
        if cfg_value == "never":
            return
        if cfg_value not in ("both", stage):
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
        when = ("before the experiment is built / run" if stage == "before_build"
                else "before the draft goes to review")
        self._pause_for_human(
            kind="supply",
            interaction="supply",
            headline=f"add any papers or data ({stage})",
            steps=[
                f"Optional — paused {when} so you can add your own sources.",
                "Drop reference papers (PDF / Markdown) into `inputs/papers/` "
                "— they become citable literature.",
                "Drop datasets (CSV / JSON / TSV / Parquet) into `inputs/data/` "
                "— the analyze node reasons over them.",
            ],
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
            axon_written = await self._axon_collect_step(query, auto_dir)

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

    async def _axon_collect_step(self, query: str, auto_dir: Path) -> int:
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
            )
        text = await self._chat(prompt, node="implement")
        code, deps = _parse_implement_response(text)
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
        return {"code": code, "deps": deps}

    async def _node_execute(self, state: QuestState) -> QuestState:
        deps = state.get("deps") or []
        if deps:
            self._log.info("[execute] pip install %s", deps)
            install = await self.executor.install(deps, quest_root=self.quest_root)
            if install.returncode != 0:
                self._log.warning(
                    "[execute] pip install rc=%d stderr_tail=%s",
                    install.returncode, install.stderr[-400:],
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
        try:
            from .plot_style import write_boot

            boot_dir = write_boot(self.fi_dir, self.config.output.paper_style)
            exec_env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p
                ),
            }
        except Exception as exc:  # styling must never break execution
            self._log.warning("[execute] plot-style bootstrap skipped: %s", exc)

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

        # Run from quest_root so figures/ is the relative target. Wrapped in a
        # heartbeat: the experiment subprocess can run for many minutes (a
        # parameter sweep) while the executor just blocks on communicate(),
        # emitting nothing — and the dashboard infers a quest's status from
        # run.log recency, so a long silent run reads as "pending"/idle. The
        # heartbeat keeps that signal fresh.
        result: ExecutionResult = await self._await_with_heartbeat(
            self.executor.execute(
                [str(py), str(code_path)],
                cwd=self.quest_root,
                timeout_s=self.config.execution.timeout_s,
                env=exec_env,
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
                env=exec_env,
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

        # Multi-seed replication. Only fires when (a) the primary run
        # succeeded — we don't want to spend N×cost re-running a script
        # that's already broken — and (b) ``execute_replicates > 1``.
        # The first run we just did counts as seed 0; we now run N-1
        # MORE with seeds 1..N-1 via ``FI_REPLICATE_SEED``. Failures
        # in any individual replicate are recorded but don't abort the
        # rest — better to have N-1 good replicates than zero. The
        # primary ``result_json`` carries seed 0 so downstream
        # single-seed code paths are unchanged.
        replicates_n = max(1, int(self.config.engine.execute_replicates))
        result_json_replicates: list[dict[str, Any]] = []
        if result.returncode == 0 and result_json is not None:
            # Tag seed 0 explicitly so the aggregator can attribute it.
            result_json_replicates.append({"_seed": 0, **result_json})
        if (
            replicates_n > 1
            and result.returncode == 0
            and result_json is not None
        ):
            self._log.info(
                "[execute] replicating: %d additional seeds (1..%d)",
                replicates_n - 1, replicates_n - 1,
            )
            for seed in range(1, replicates_n):
                # Merge with the parent's environment, not REPLACE it.
                # ``asyncio.create_subprocess_exec(env=...)`` overrides
                # the child's whole env when given a dict — passing
                # only FI_REPLICATE_SEED would strip PATH, PYTHONPATH,
                # LANG, SystemRoot (Windows), etc., and the venv
                # python.exe would fail at the DLL-load step. Build
                # a merged env so the child sees the inherited values
                # plus our replicate seed.
                rep_env = {**(exec_env or os.environ), "FI_REPLICATE_SEED": str(seed)}
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
                else:
                    self._log.warning(
                        "[execute] replicate seed=%d FAILED rc=%d duration=%.1fs "
                        "(skipping in aggregate)",
                        seed, rep_result.returncode, rep_result.duration_s,
                    )

        patch: dict[str, Any] = {
            "exec_result": {
                "returncode": result.returncode,
                "duration_s": result.duration_s,
                "timed_out": result.timed_out,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            },
            "figures": figures,
            "result_json": result_json or {},
        }
        # Only populate ``result_json_replicates`` when replication
        # actually ran AND produced more than the primary entry. This
        # keeps the field absent on default single-seed quests so
        # downstream code can use ``state.get("result_json_replicates")``
        # as a "did we run multi-seed" sentinel.
        if len(result_json_replicates) > 1:
            patch["result_json_replicates"] = result_json_replicates
        return patch

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

        # Clean success (and not a degenerate all-zero run) → no-op.
        if rc == 0 and has_result_json and not degenerate:
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
        else:
            stdout_for_prompt = exec_result.get("stdout_tail", "")[:2000]
            returncode_for_prompt = str(rc)
            result_json_note = "yes" if has_result_json else "no"

        prompt = self._prompts["execute_reflect"].substitute(
            previous_code=(state.get("code") or "")[:8000],
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
        text = await self._chat(prompt, node="execute_reflect")
        parsed = _parse_json_lenient(text) or {}

        give_up = (parsed.get("give_up_reason") or "").strip()
        if give_up:
            self._log.warning("[execute_reflect] LLM gave up: %s", give_up[:200])
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
            return {
                "exec_give_up_reason": "(LLM produced no patched code)",
                "exec_reflect_iter": iters + 1,
            }

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
            "code": new_code,
            "exec_reflect_iter": iters + 1,
            "exec_reflect_history": history,
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
        if not k.fetch_web_figures or k.web_figures_max <= 0:
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
            agg = _aggregate_result_json_replicates(replicates)
            payload: dict[str, Any] = {
                "result_json_seed_0": state.get("result_json") or {},
                "replicates": replicates,
                # Each metric carries mean/std/n/min/max + se + 95% CI bounds.
                "aggregate_mean_std": agg,
                "n_replicates": len(replicates),
            }
            # Per-stratum CIs + pairwise effect sizes + multiple-comparison
            # guard for any by_<factor> breakdowns (empty otherwise).
            comparison_stats = _result_comparison_stats(replicates)
            if comparison_stats:
                payload["comparison_stats"] = comparison_stats
            if all_ds:
                payload["_user_supplied_datasets"] = all_ds
            result_json_block, _rj_orig = _compact_result_json_block(payload)
            self._log.info(
                "[analyze] using replicate aggregate (n=%d, %d numeric keys, "
                "%d comparison(s))", len(replicates), len(agg),
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
        }
        self._log.info(
            "[evidence_gate] verdict=%s route=%s type=%s policy=%s "
            "(sources=%d, findings=%d, supporting=%d, broadened=%d)",
            verdict, assessment["route"], protocol.topic_type,
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

    async def _node_write(self, state: QuestState) -> QuestState:
        persona_block = self._resolve_write_persona(state)
        self._log.info(
            "[write] authoring paper.md (persona=%s)",
            persona_block.split("\n", 1)[0][:80] if persona_block else "default",
        )
        # When the evidence gate judged the evidence thin (insufficient,
        # or "broaden" with the broaden budget exhausted), hand the writer
        # an explicit note so the paper frames its limits honestly instead
        # of over-claiming. Empty when the gate was satisfied / disabled.
        _ev = state.get("evidence_assessment") or {}
        evidence_note = ""
        if _ev.get("verdict") in ("insufficient", "broaden"):
            _gaps = "; ".join(_ev.get("gaps") or [])
            evidence_note = (
                "**Evidence note (pre-write evidence gate).** The assembled "
                f"evidence was judged *{_ev.get('verdict')}* for this research "
                "question"
                + (f": {_ev.get('rationale')}" if _ev.get("rationale") else "")
                + (f" Specific gaps: {_gaps}." if _gaps else "")
                + " Honour this: report only what the sources/data actually "
                "support, state the limitation plainly (an honest, scoped paper "
                "is the goal), and do NOT manufacture confidence the evidence "
                "doesn't carry. This is a scientific limitation of the study — "
                "do not phrase it as a failure of any tool or pipeline.\n"
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
                **self._lit_kwargs(state),
            ),
            figure_list=_figure_list_for_prompt(state),
            clarify_block=_format_clarify(state),
            cross_check_block=_format_cross_check(state),
            study_mode_note=(
                _NO_SIM_WRITE_NOTE
                if state.get("no_simulation_resolved")
                else ""
            ),
            evidence_note=evidence_note,
        )
        markdown = await self._chat(prompt, node="write")
        # The model may wrap with a fence; strip it.
        markdown = _strip_outer_fence(markdown)
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
        return {"paper_md": str(paper_path)}

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
        paper_text = Path(paper_md).read_text(encoding="utf-8")[:16000]
        refs = build_references(
            state.get("literature") or [], audience=self.config.output.audience,
        )
        refs_block = "\n".join(
            f"[{r['n']}] {r['title']}"
            + (f" · DOI:{r['doi']}" if r.get("doi") else "")
            for r in refs
        ) or "(no references)"
        analysis = state.get("analysis") or {}
        evidence = {
            "key_findings": analysis.get("key_findings") or [],
            "claims_supported": analysis.get("claims_supported") or [],
            "result_json": state.get("result_json") or {},
        }
        prompt = self._prompts["claim_check"].substitute(
            topic=state["topic"],
            evidence_block=json.dumps(evidence, indent=2)[:4000],
            references=refs_block,
            paper=paper_text,
        )
        text = await self._chat(prompt, node="claim_check")
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
            # A "citation" basis only counts if it points at a real reference:
            # validate the index is an int in [1, n_refs], else it's effectively
            # unsupported (a claim that names no source isn't grounded).
            cite_idx = c.get("citation_index")
            try:
                cite_idx = int(cite_idx)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                cite_idx = None
            if basis == "citation" and not (cite_idx and 1 <= cite_idx <= n_refs):
                basis = "unsupported"
                cite_idx = None
            claims.append({
                "claim": str(c["claim"]).strip(),
                "basis": basis,
                "citation_index": cite_idx,
                "evidence": str(c.get("evidence") or "").strip(),
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
        return {"claim_grounding": grounding}

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
                             + (f" — {c['evidence']}" if c["evidence"] else ""))
            (paper_dir / "CLAIMS.md").write_text(
                "\n".join(lines) + "\n", encoding="utf-8",
            )
        except OSError as e:
            self._log.warning("[claim_check] ledger write failed: %r", e)

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
        paper_md = ""
        paper_path = state.get("paper_md")
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
            # 16 KB ≈ ~4 K tokens — fits a comprehensive-review-length
            # paper plus an abstract + references block. The 8 KB cap
            # was truncating mid-Discussion on journal-length papers
            # so the reviewer was grading on an incomplete read, which
            # made the depth axis unreliable.
            paper_md=paper_md[:16000],
        )

        panel_names = list(self.config.engine.review_panel or [])
        if not panel_names:
            # Legacy single-reviewer path — unchanged behavior.
            text = await self._chat(base_prompt, node="review")
            review = _parse_json_lenient(text) or {
                "verdict": "accept", "score": 3, "suggestions": [],
            }
            mfh = review.get("must_flag_hits") or []
            if not isinstance(mfh, list):
                mfh = []
            review["must_flag_hits"] = [str(h).strip() for h in mfh if str(h).strip()]
            update: QuestState = {"review": review}
            # Iteration is consumed when EITHER the verdict says revise
            # OR the must-flag hits force one. Bumping on must_flag_hits
            # alone (even with verdict=accept) makes ``_route_after_review``'s
            # non-bypassable revise path deterministic with respect to the
            # ``max_iterations`` budget — without this, a malformed
            # ``revise`` route from must-flag wouldn't have consumed the
            # iteration and the loop could run unbounded.
            if review.get("verdict") == "revise" or review["must_flag_hits"]:
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
            text = await self._chat(prompt, node=f"review_panel.{name}")
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

        panel_results = await asyncio.gather(
            *(run_persona(n) for n in panel_names),
            return_exceptions=False,
        )
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
        rationale = mod_parsed.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            review["rationale"] = rationale.strip()
        # Use the moderator's suggestions if they're well-formed and
        # carry the persona-attribution prefix; otherwise keep agg's.
        mod_suggs = mod_parsed.get("suggestions")
        if isinstance(mod_suggs, list) and mod_suggs:
            review["suggestions"] = [str(s) for s in mod_suggs]

        update: QuestState = {"review": review, "review_panel": panel_results}
        # Bump iteration on EITHER verdict=revise OR a non-empty
        # must_flag_hits list. Without the must-flag clause, a malformed
        # persona response that recorded ``verdict=accept`` alongside a
        # must-flag hit would route to revise (via _route_after_review)
        # without consuming iteration budget — the loop could spin.
        if review.get("verdict") == "revise" or (review.get("must_flag_hits") or []):
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

    async def _chat(self, prompt: str, *, node: str | None = None) -> str:
        """Single-user-message chat. ``node`` is the engine node name
        (e.g. ``"ideate"``, ``"review"``); when present and the YAML
        config sets ``provider.node_models[node]``, that model is sent
        on this call only. Otherwise the endpoint default applies."""
        assert self._client is not None
        messages = [{"role": "user", "content": prompt}]
        response = await self._client.chat(
            messages, temperature=0.2, model=self._model_for_node(node),
            node=node or "",
        )
        self._log_chat_cost(node=node or "")
        return response

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
              "estimated_rows": <int>,        # rows from char-based fallback
              "by_node":  { <node>: {...same shape...} },
              "by_model": { <model>: {...same shape...} },
              "generated_at": <epoch>,
            }

        Best-effort: a missing / unreadable cost.jsonl produces a
        summary with zeros; no exception bubbles out.
        """
        try:
            path = self.fi_dir / "cost.jsonl"
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
            (self.fi_dir / "cost.summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8",
            )
        except OSError as e:
            self._log.debug("[cost] failed to write cost.summary.json: %r", e)

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
        # Lazy-import: web UI quests touch this file every chat call;
        # importing json + time once and keeping a bound reference at
        # the function level is the same cost path most stdlib code
        # uses. The provider module is the source of `estimate_cost_usd`.
        from core.provider import estimate_cost_usd
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
        cost = None
        if usage:
            cost = estimate_cost_usd(
                model,
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
            )
        record = {
            "ts": time.time(),
            "node": node,
            "model": model,
            "usage": usage,
            "cost_usd": cost,
        }
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            with (self.fi_dir / "cost.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            # Cost logging is best-effort — a disk-full or
            # permission failure must NOT crash the quest. Log to the
            # quest's own logger and move on; the chart will just
            # show fewer rows.
            self._log.debug("[cost] failed to write cost.jsonl: %r", e)

    def _model_for_node(self, node: str | None) -> str | None:
        """Resolve the effective model for a node — empty string when
        the lookup misses so the transport falls through to the
        endpoint default. Accepts hierarchical keys like
        ``"review_panel.methodologist"`` (review-panel personas)."""
        if not node:
            return None
        node_models = self.config.provider.node_models or {}
        if not node_models:
            return None
        # Exact match wins; then prefix match (e.g., "review_panel"
        # catches "review_panel.methodologist" when no persona-specific
        # entry exists).
        if node in node_models:
            return node_models[node]
        if "." in node:
            base = node.split(".", 1)[0]
            if base in node_models:
                return node_models[base]
        return None

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
        pandoc_exe = shutil.which("pandoc")
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
        meta: dict[str, Any] = {
            "title": state.get("title", ""),
            "topic": state.get("topic", "")[:1000],
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
        "estimated_rows": 0,
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
        }

    for row in rows:
        if row.get("ensemble") is True:
            continue
        usage = row.get("usage") or {}
        prompt_t = int(usage.get("prompt_tokens", 0) or 0)
        completion_t = int(usage.get("completion_tokens", 0) or 0)
        total_t = int(usage.get("total_tokens", 0) or 0) or (prompt_t + completion_t)
        cost = row.get("cost_usd")
        is_estimated = bool(usage.get("estimated"))

        totals["total_requests"] += 1
        totals["total_prompt_tokens"] += prompt_t
        totals["total_completion_tokens"] += completion_t
        totals["total_tokens"] += total_t
        if isinstance(cost, (int, float)):
            totals["total_cost_usd"] += float(cost)
            has_cost_value = True
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
            if isinstance(cost, (int, float)):
                bucket["cost_usd"] += float(cost)
            if is_estimated:
                bucket["estimated_rows"] += 1

    if not has_cost_value:
        # No real pricing data hit any row — surface that explicitly
        # instead of pretending the total is $0.00.
        totals["total_cost_usd"] = None  # type: ignore[assignment]
        for bucket in list(by_node.values()) + list(by_model.values()):
            bucket["cost_usd"] = None  # type: ignore[assignment]

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
        "implement_body",                   # two-stage implement: fills bodies
        "execute_reflect", "analyze",
        "cross_check",
        "cross_check_verify",   # CoVe-style second-pass verification
        "claim_check",          # ground each paper claim to evidence
        "evidence_gate",        # weigh evidence sufficiency before write
        "write", "review",
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


def _format_lit_header(meta: dict[str, Any], i: int) -> str:
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
) -> str:
    items = state.get("literature") or []
    if not items:
        return "(no prior work surfaced from the knowledge base)"
    lines: list[str] = []
    keep_idx = 0
    for item in items:
        meta = item.get("metadata") or {}
        if not _is_citable(meta):
            continue
        if not _is_audience_appropriate(meta, audience):
            continue
        keep_idx += 1
        title = meta.get("title") or meta.get("source") or f"item-{keep_idx}"
        header = _format_lit_header(meta, keep_idx)
        excerpt = _format_lit_excerpt(
            item.get("content", "") or "", title,
            query=query, budget=budget, mode=mode,
        )
        lines.append(f"{header}\n{excerpt}" if excerpt else header)
    if not lines:
        return "(no prior work surfaced from the knowledge base)"
    return "\n\n".join(lines)


def build_references(
    literature: list[Any],
    *,
    audience: str = "external",
    max_n: int | None = None,
) -> list[dict[str, Any]]:
    """Build a clean, de-duplicated, numbered citation list from the
    quest's retrieved literature (``state['literature']`` dict items
    ``{content, metadata}`` or ``RetrievedDoc`` objects).

    Applies the same citability + audience rules as the paper's
    References (:func:`_is_citable` / :func:`_is_audience_appropriate`),
    so a **web page** (title + URL) is a first-class citation and
    FI-internal cross-quest memory artifacts are dropped. Shared by the
    poster + slides generators so every output surfaces the same sources
    the writer cited. Each entry: ``{n, title, authors, year, venue,
    doi, arxiv_id, url, site, source}``."""
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in literature or []:
        if isinstance(item, dict):
            meta = item.get("metadata") or {}
        else:
            meta = getattr(item, "metadata", {}) or {}
        if not _is_citable(meta) or not _is_audience_appropriate(meta, audience):
            continue
        key = (
            meta.get("doi") or meta.get("arxiv_id") or meta.get("pmid")
            or meta.get("url") or (meta.get("title") or "")
        ).lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        authors = meta.get("authors") or []
        if not isinstance(authors, list):
            authors = [str(authors)]
        refs.append({
            "n": len(refs) + 1,
            "title": (meta.get("title") or "").strip(),
            "authors": [a for a in authors if a],
            "year": meta.get("year") or (meta.get("published") or "")[:4] or "",
            "venue": meta.get("venue") or meta.get("publisher") or "",
            "doi": meta.get("doi") or "",
            "arxiv_id": meta.get("arxiv_id") or "",
            "url": meta.get("url") or "",
            "site": meta.get("site") or "",
            "source": meta.get("source") or "",
        })
        if max_n and len(refs) >= max_n:
            break
    return refs


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
        line = f"{line}. {' '.join(tail)}" if line else " ".join(tail)
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


def render_references_marp_slide(refs: list[dict[str, Any]], *, max_n: int = 18) -> str:
    """A standalone Marp slide listing sources, appended after the LLM's
    deck so a References slide always lands when sources exist."""
    if not refs:
        return ""
    lines = ["---", "", "## References", ""]
    lines += [f"{r['n']}. {_ref_citation_text(r)}" for r in refs[:max_n]]
    if len(refs) > max_n:
        lines.append(f"\n_(+{len(refs) - max_n} more sources)_")
    return "\n".join(lines)


def render_poster_references_latex(
    refs: list[dict[str, Any]], *, max_n: int = 12,
) -> str:
    """A compact full-width Sources band for the poster footer — injected
    by the template (not the LLM) so references always render."""
    if not refs:
        return ""
    shown = refs[:max_n]
    parts = [f"[{r['n']}]~{_latex_esc(_ref_citation_text(r))}" for r in shown]
    more = "" if len(refs) <= max_n else f" \\quad (+{len(refs) - max_n} more)"
    return (
        "\\vspace{0.4em}\\hrule\\vspace{0.3em}\n"
        "{\\scriptsize\\textbf{Sources:}~ " + " \\quad ".join(parts) + more + "}\n"
    )


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
            "question": "What's the intellectual shape of this topic? (experimental / review / case_study / opinion)",
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


def _aggregate_result_json_replicates(
    replicates: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Compute mean ± sample-std for every numeric scalar field that
    appears in EVERY replicate's ``RESULT_JSON``.

    Each entry in ``replicates`` is one ``RESULT_JSON`` dict augmented
    with a synthetic ``_seed`` field. We skip the ``_seed`` key, scan
    the union of remaining keys, and emit an aggregate only for keys
    whose values are scalar floats/ints in ALL replicates — mixed-type
    keys (strings, lists, dicts) are skipped silently so the analyze
    LLM can still reason about them from the raw per-seed JSON.

    Returns ``{key: {"mean": float, "std": float, "n": int, "min": float, "max": float}}``.
    Empty input → empty dict. n=1 (single replicate) → emits min=max=value,
    std=0.0.
    """
    if not replicates:
        return {}

    # Union of all keys (excluding _seed).
    all_keys: set[str] = set()
    for r in replicates:
        if not isinstance(r, dict):
            continue
        for k in r.keys():
            if k != "_seed":
                all_keys.add(k)

    out: dict[str, dict[str, float | int]] = {}
    for key in sorted(all_keys):
        vals: list[float] = []
        all_scalar_numeric = True
        for r in replicates:
            if not isinstance(r, dict) or key not in r:
                all_scalar_numeric = False
                break
            v = r[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                all_scalar_numeric = False
                break
            vals.append(float(v))
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
            # seed has no spread to estimate, so we don't fake an interval).
            **_stats.confidence_interval(vals),
        }
    return out


def _stratum_series(
    replicates: list[dict[str, Any]], factor: str, stratum: str, metric: str,
) -> list[float] | None:
    """The per-seed values of ``replicates[*][factor][stratum][metric]`` — only
    when it's a scalar number in EVERY replicate (else None: can't compare it)."""
    vals: list[float] = []
    for r in replicates:
        by = r.get(factor) if isinstance(r, dict) else None
        st = by.get(stratum) if isinstance(by, dict) else None
        v = st.get(metric) if isinstance(st, dict) else None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        vals.append(float(v))
    return vals


def _result_comparison_stats(
    replicates: list[dict[str, Any]], *, max_effect_sizes: int = 24,
) -> dict[str, Any]:
    """Per-stratum CIs + pairwise effect sizes (Cohen's d) between the
    ``by_<factor>`` strata the experiment broke results down by, plus a
    multiple-comparison guard. Lets the paper report *how big* a between-group
    difference is and whether it survives correction for the number of
    comparisons — not just its direction.

    Returns ``{"strata": {factor: {stratum: {metric: {mean, ci_lower,
    ci_upper, n}}}}, "effect_sizes": [{factor, metric, a, b, cohens_d,
    magnitude}], "comparisons": {"n", "bonferroni_alpha", "many"}}``. Empty
    when no ``by_*`` factor has ≥2 numeric strata to compare."""
    if not replicates:
        return {}
    # Factors present (and a dict) in EVERY replicate.
    factors = sorted({
        k for k in (replicates[0] if isinstance(replicates[0], dict) else {})
        if k.startswith("by_")
        and all(isinstance(r.get(k), dict) for r in replicates)
    })
    strata_out: dict[str, Any] = {}
    effect_sizes: list[dict[str, Any]] = []
    n_comparisons = 0
    for factor in factors:
        # Strata present in every replicate's factor dict.
        strata = sorted(
            set(replicates[0][factor])
            .intersection(*(set(r[factor]) for r in replicates))
        )
        if len(strata) < 2:
            continue
        # Metrics scalar-numeric across all reps for at least one stratum.
        metrics: set[str] = set()
        for st in strata:
            metrics.update(
                m for m in replicates[0][factor].get(st, {})
                if _stratum_series(replicates, factor, st, m) is not None
            )
        per_stratum: dict[str, Any] = {}
        for st in strata:
            cell: dict[str, Any] = {}
            for m in sorted(metrics):
                series = _stratum_series(replicates, factor, st, m)
                if series is None:
                    continue
                mean, _, n = _stats.mean_std(series)
                cell[m] = {"mean": mean, "n": n,
                           **_stats.confidence_interval(series)}
            if cell:
                per_stratum[st] = cell
        if per_stratum:
            strata_out[factor] = per_stratum
        # Pairwise effect sizes between strata, per shared metric.
        for m in sorted(metrics):
            for i in range(len(strata)):
                for j in range(i + 1, len(strata)):
                    a = _stratum_series(replicates, factor, strata[i], m)
                    b = _stratum_series(replicates, factor, strata[j], m)
                    if a is None or b is None:
                        continue
                    n_comparisons += 1
                    d = _stats.cohens_d(a, b)
                    effect_sizes.append({
                        "factor": factor, "metric": m,
                        "a": strata[i], "b": strata[j],
                        "cohens_d": d,
                        "magnitude": _stats.effect_magnitude(d),
                    })
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


def _format_claim_grounding(state: QuestState) -> str:
    """Render the claim-grounding result for the review prompt's
    `$claim_grounding_block`, calling out unsupported claims so the reviewer
    must-flags them."""
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
            lines.append(f"- figures/{f}")
    return "\n".join(lines)


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
) -> None:
    """Write a single human-friendly, RANKED ``needs/WANTED_PAPERS.md`` —
    most relevant first, each with a download link, what it's about, and the
    open-access status — so the user can grab the few that matter."""
    lines = [
        f"# Papers to download — {topic[:120].strip()}",
        "",
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
    _write_wanted_papers_md(quest_root, needed, topic=query or quest_root.name)
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
