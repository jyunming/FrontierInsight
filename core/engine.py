"""Frontier Insight research engine — async LangGraph DAG.

Phase B: real LLM-driven nodes, code generation + execution in a per-quest
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
import re
import shutil
import string
import time
import uuid

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, TypedDict

ClarifyCallback = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
"""User-supplied async function that collects answers to clarify-node
questions. Receives the `clarify_questions` dict and must return the
answers dict (same keys, resolved values)."""

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .config import Config
from .execution import ExecutionResult, make_executor
from .knowledge import Knowledge, RetrievedDoc
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
    # Phase I clarify-node state. Both dicts share the same 5 keys
    # (`comparative_baseline`, `empirical_vs_theoretical`,
    # `success_metric`, `budget`, `output_kinds`); `clarify_questions`
    # carries `{question, default}` per slot, `clarify_answers` carries
    # the resolved values (default or user-overridden).
    clarify_questions: dict[str, Any]
    clarify_answers: dict[str, Any]
    clarify_done: bool
    ideas: list[dict[str, Any]]
    chosen_idea: dict[str, Any]
    # Phase M — ideate self-reflection result. Optional; describes what
    # the agent considered before locking in `chosen_idea`.
    ideate_critique: dict[str, Any]
    literature: list[dict[str, Any]]
    design: dict[str, Any]
    code: str
    deps: list[str]
    exec_result: dict[str, Any]
    figures: list[str]
    result_json: dict[str, Any]
    # Phase K — execute-repair loop counter + history. The reflect
    # node increments `exec_reflect_iter` and appends a one-line
    # record per attempt, so analyze/write/review can describe what
    # was fixed.
    exec_reflect_iter: int
    exec_reflect_history: list[dict[str, Any]]
    exec_give_up_reason: str
    analysis: dict[str, Any]
    # Phase L — cross-paper check per finding. List of per-finding
    # records carrying supporting / conflicting / neutral classifications.
    cross_check: list[dict[str, Any]]
    paper_md: str
    review: dict[str, Any]
    # Phase N — per-persona reviews from the panel, before moderation.
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
    # Phase D1 — number of docs the auto_collect_data node successfully
    # wrote into `<quest_root>/data/auto_collected/`. 0 means the node
    # was a passthrough (auto-collect disabled, knowledge disabled, or
    # Axon returned no hits), in which case wait_for_data falls back
    # to pausing for user-supplied data. Positive values let the user
    # see in run.log / state how much of their data load came from
    # the agent vs from manual drops.
    auto_collected_count: int


class Engine:
    """Owns one quest's research graph, executor, knowledge layer, and LLM client."""

    def __init__(
        self,
        config: Config,
        *,
        supervisor: ProxySupervisor | None = None,
        resume_quest_id: str | None = None,
    ) -> None:
        self.config = config
        _warn_if_unsanctioned_provider(config.provider.name)
        # `resume_quest_id` lets a caller re-enter an existing quest
        # (LangGraph's AsyncSqliteSaver keys checkpoints by thread_id,
        # which we set to quest_id below — so reusing the id auto-
        # resumes from the last completed node when a prior run died
        # mid-pipeline, e.g. on a sustained upstream Copilot outage).
        self.quest_id = resume_quest_id or _new_quest_id(config.title or config.topic)
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

    async def run(
        self,
        *,
        clarify_callback: ClarifyCallback | None = None,
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
            self._client = LLMClient(endpoint)

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
                    else:
                        payload = initial

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
                        # Clarify node raised `interrupt(...)`. Hand the
                        # questions to the caller's callback for answers.
                        questions = intr_value.get("clarify_questions", {})
                        if clarify_callback is None:
                            raise RuntimeError(
                                f"quest {self.quest_id} paused at clarify node but no "
                                f"clarify_callback was supplied; set clarify_mode to "
                                f"'auto' or 'off' for headless runs."
                            )
                        self._log.info(
                            "[run] clarify interrupt fired with %d questions; "
                            "invoking callback", len(questions),
                        )
                        answers = await clarify_callback(questions)
                        payload = Command(resume=answers)
            finally:
                await self._client.aclose()
                if self.config.provider.name in PROXY_PROVIDERS:
                    await self.supervisor.release(self.config.provider.name)

            if data_paused:
                # no-simulation pause-exit. The graph is checkpointed
                # at the wait_for_data interrupt; return early with a
                # partial QuestArtifacts so callers (launch.py / the
                # VSCode bridge) can surface the "drop files here"
                # message but don't try to compile a paper that doesn't
                # exist yet. Skip _write_back_knowledge — we have
                # nothing to write back; the quest hasn't been
                # reviewed and accepted.
                self._log.info(
                    "quest %s paused for user data — exiting clean (rc=0)",
                    self.quest_id,
                )
                return self._collect_artifacts(final_state)

            artifacts = self._collect_artifacts(final_state)
            self._write_back_knowledge(artifacts, final_state)
            self._log.info("quest %s reached terminal state", self.quest_id)
            return artifacts
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

    # ---- graph topology --------------------------------------------------

    def _build_graph(self) -> StateGraph:
        # Phase G hook: subclassing `Engine` and overriding `_build_graph`
        # is the supported way to ship a domain-specific pipeline (e.g.,
        # a lithography graph) without forking the full Engine class.
        # The QuestState TypedDict is the contract — keep field names
        # backwards-compatible if you add a graph here.
        g: StateGraph[QuestState] = StateGraph(QuestState)
        g.add_node("clarify", self._node_clarify)
        g.add_node("ideate", self._node_ideate)
        g.add_node("literature", self._node_literature)
        g.add_node("design", self._node_design)
        g.add_node("implement", self._node_implement)
        g.add_node("execute", self._node_execute)
        # Phase K: execute → execute_reflect (loops back to execute on failure)
        g.add_node("execute_reflect", self._node_execute_reflect)
        g.add_node("analyze", self._node_analyze)
        # Phase L: analyze → cross_check (always) → write OR design
        g.add_node("cross_check", self._node_cross_check)
        g.add_node("write", self._node_write)
        g.add_node("review", self._node_review)
        # no-simulation mode: design → auto_collect_data → wait_for_data
        # → (pause + resume) → data_load → analyze. All three new nodes
        # are conditional and only fire when
        # ``state.no_simulation_resolved`` is True.
        g.add_node("auto_collect_data", self._node_auto_collect_data)
        g.add_node("wait_for_data", self._node_wait_for_data)
        g.add_node("data_load", self._node_data_load)

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
            {"implement": "implement", "auto_collect_data": "auto_collect_data"},
        )
        g.add_edge("implement", "execute")
        g.add_edge("execute", "execute_reflect")
        g.add_conditional_edges(
            "execute_reflect",
            self._route_after_execute_reflect,
            {"retry": "execute", "proceed": "analyze"},
        )
        # auto_collect_data: Phase D1. Best-effort Axon retrieval that
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
        g.add_edge("data_load", "analyze")
        g.add_edge("analyze", "cross_check")
        g.add_conditional_edges(
            "cross_check",
            self._route_after_cross_check,
            {"write": "write", "redesign": "design"},
        )
        g.add_edge("write", "review")
        g.add_conditional_edges(
            "review",
            self._route_after_review,
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
        """
        if state.get("no_simulation_resolved"):
            return "auto_collect_data"
        return "implement"

    def _route_after_execute_reflect(self, state: QuestState) -> str:
        """Phase K: route based on whether the reflect node patched the
        code (→ retry execute) or accepted the failure / success
        (→ proceed to analyze)."""
        result = state.get("exec_result") or {}
        rc = result.get("returncode", 0)
        has_result_json = state.get("result_json") is not None
        # Success path: nothing to repair.
        if rc == 0 and has_result_json:
            return "proceed"
        # Give-up sentinel set by the reflect node OR iterations exhausted.
        if state.get("exec_give_up_reason"):
            return "proceed"
        iters = state.get("exec_reflect_iter", 0)
        if iters >= self.config.engine.exec_reflect_max_iterations:
            return "proceed"
        return "retry"

    def _route_after_cross_check(self, state: QuestState) -> str:
        """Phase L: route based on analyze's `next_step` field. If the
        config disables analyze-rerouting, always go to write."""
        if not self.config.engine.enable_analyze_reroute:
            return "write"
        analysis = state.get("analysis") or {}
        next_step = analysis.get("next_step", "publish")
        if next_step in ("re_experiment", "broaden_lit"):
            # Share the review-loop iteration budget so the whole quest
            # stays bounded.
            if state.get("iteration", 0) < self.config.engine.max_iterations:
                return "redesign"
        return "write"

    def _route_after_review(self, state: QuestState) -> str:
        if not self.config.engine.review_loop:
            return "done"
        review = state.get("review") or {}
        verdict = review.get("verdict", "accept")
        if verdict == "revise" and state.get("iteration", 0) < self.config.engine.max_iterations:
            return "revise"
        return "done"

    # ---- nodes -----------------------------------------------------------

    async def _node_clarify(self, state: QuestState) -> QuestState:
        """Phase I pre-flight clarification.

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
        mode = self.config.engine.clarify_mode
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
            answers = {k: v.get("default") for k, v in questions.items() if isinstance(v, dict)}
            self._log.info("[clarify] mode=auto; agent self-answered %d slots", len(answers))
            return {
                "clarify_questions": questions,
                "clarify_answers": answers,
                "clarify_done": True,
                "no_simulation_resolved": self._resolve_no_simulation_from_clarify(answers),
            }

        # Interactive: pause the graph until the caller resumes with answers.
        self._log.info(
            "[clarify] mode=interactive; pausing for human input (%d questions)",
            len(questions),
        )
        payload = interrupt({"clarify_questions": questions})
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
        return {
            "clarify_questions": questions,
            "clarify_answers": answers,
            "clarify_done": True,
            "no_simulation_resolved": self._resolve_no_simulation_from_clarify(answers),
        }

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

        sim = answers.get("simulatability")
        if isinstance(sim, dict):
            decision = str(sim.get("default", "")).strip().lower()
            reason = str(sim.get("reason", "")).strip()
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
            literature_block=_format_lit(seeded),
            clarify_block=_format_clarify(state),
        )
        text = await self._chat(prompt, node="ideate")
        parsed = _parse_json_lenient(text) or {}
        ideas = parsed.get("ideas") or []
        chosen = parsed.get("chosen") or (ideas[0] if ideas else {"title": "fallback", "rationale": ""})

        # Phase M — self-reflection. Single extra LLM call that may swap
        # chosen_idea to a different entry from the brainstormed list.
        critique: dict[str, Any] = {}
        if self.config.engine.ideate_reflect and ideas:
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
        return out

    async def _node_literature(self, state: QuestState) -> QuestState:
        chosen = state.get("chosen_idea") or {}
        query = (chosen.get("title") or "") + " " + state["topic"][:200]
        docs = await self.knowledge.asearch(
            query.strip(),
            top_k=self.config.knowledge.top_k,
            chosen_idea=chosen,
            chat_fn=functools.partial(self._chat_messages, node="source_router"),
        )
        self._log.info("[literature] retrieved %d docs", len(docs))
        return {
            "literature": [
                {"content": d.content[:2000], "metadata": d.metadata} for d in docs
            ]
        }

    async def _node_design(self, state: QuestState) -> QuestState:
        iteration = state.get("iteration", 0)
        self._log.info("[design] iteration=%d", iteration)
        review_feedback = ""
        if iteration > 0:
            review_feedback = json.dumps(state.get("review", {}), indent=2)
        prompt = self._prompts["design"].substitute(
            topic=state["topic"],
            chosen_idea=json.dumps(state.get("chosen_idea") or {}, indent=2),
            literature_block=_format_lit_from_state(state),
            review_feedback=review_feedback or "(none — first iteration)",
            timeout_s=str(self.config.execution.timeout_s),
            clarify_block=_format_clarify(state),
        )
        text = await self._chat(prompt, node="design")
        design = _parse_json_lenient(text) or {"hypothesis": "(parse failed)", "dependencies": []}
        return {"design": design}

    async def _node_auto_collect_data(self, state: QuestState) -> QuestState:
        """Phase D1 — agent-side data collection via Axon, run BEFORE
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
        if not self.knowledge.enabled:
            self._log.info(
                "[auto_collect] knowledge.enabled=False — skipping "
                "(no Axon to query); will pause for user data",
            )
            return {"auto_collected_count": 0}

        # Build a query from topic + the design hypothesis (when
        # design has run). Topic alone is enough for first-pass
        # retrieval; hypothesis sharpens it on later iterations.
        topic = state.get("topic", "")
        design = state.get("design") or {}
        hypothesis = ""
        if isinstance(design, dict):
            hypothesis = str(design.get("hypothesis", "")).strip()
        query = f"{topic} {hypothesis}".strip() or topic
        top_k = self.config.engine.auto_collect_top_k
        self._log.info(
            "[auto_collect] querying Axon: top_k=%d query=%r",
            top_k, query[:120],
        )
        try:
            docs = await self.knowledge.asearch(query, top_k=top_k)
        except Exception as e:
            self._log.warning(
                "[auto_collect] Axon search raised — falling through to "
                "user-data pause (no files written): %s", e,
            )
            return {"auto_collected_count": 0}

        if not docs:
            self._log.info(
                "[auto_collect] Axon returned 0 docs — falling through "
                "to user-data pause",
            )
            return {"auto_collected_count": 0}

        # Lazy mkdir: defer creating ``auto_collected/`` until at least
        # one write succeeds. Without this, a permissions / disk-full
        # failure where ALL writes raise OSError would still leave an
        # empty directory behind, misleading the user about whether
        # auto-collection produced anything.
        auto_dir = self.quest_root / "data" / "auto_collected"
        written_targets: list[Path] = []
        for idx, doc in enumerate(docs, start=1):
            # Build a slug from the source filename (when known) so
            # the file is greppable from the user's side later.
            meta = doc.metadata or {}
            source = str(meta.get("source") or meta.get("path") or "")
            slug_basis = Path(source).stem if source else f"doc{idx}"
            slug = _slugify(slug_basis)[:40] or f"doc{idx}"
            fname = f"{idx:03d}_{slug}.md"
            target = auto_dir / fname
            body = _render_auto_collected_md(idx, meta, doc.content or "")
            try:
                # mkdir(exist_ok=True) is idempotent — safe to call
                # once per doc; this is the "lazy on first write" hook.
                auto_dir.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
                written_targets.append(target)
            except OSError as e:
                self._log.warning(
                    "[auto_collect] failed to write %s: %s — skipping doc",
                    target, e,
                )

        written = len(written_targets)
        # Hard guarantee: when ZERO writes succeeded, there must be no
        # empty ``auto_collected/`` directory left behind (could happen
        # if the first mkdir succeeded then every write failed — rare
        # but possible on a near-full disk). Try to remove; if the
        # rmdir fails (e.g. concurrent file appeared), log and move on.
        if written == 0 and auto_dir.is_dir():
            try:
                auto_dir.rmdir()
            except OSError as e:
                self._log.warning(
                    "[auto_collect] all writes failed AND could not "
                    "rmdir leftover %s: %s — directory may appear "
                    "incorrectly as a partial-success artifact",
                    auto_dir, e,
                )

        self._log.info(
            "[auto_collect] wrote %d/%d Axon doc(s) under %s",
            written, len(docs), auto_dir,
        )
        return {"auto_collected_count": written}

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
            self._log.info(
                "[wait_for_data] no user data yet in %s — pausing for "
                "user to drop files and re-run", data_dir,
            )
            # Pause via LangGraph's interrupt mechanism. Engine.run
            # catches this and exits rc=0 cleanly. On resume the
            # interrupt re-fires; if the user has dropped files by
            # then, the resume payload carries them. (We don't trust
            # the payload — we re-walk the dir on every resume.)
            interrupt({
                "data_required": True,
                "quest_id": self.quest_id,
                "data_dir": str(data_dir),
            })
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

    async def _node_implement(self, state: QuestState) -> QuestState:
        self._log.info("[implement] generating experiment code")
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

        # Run from quest_root so figures/ is the relative target.
        result: ExecutionResult = await self.executor.execute(
            [str(py), str(code_path)],
            cwd=self.quest_root,
            timeout_s=self.config.execution.timeout_s,
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
        return {
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

    async def _node_execute_reflect(self, state: QuestState) -> QuestState:
        """Phase K: post-execute repair node.

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
        has_result_json = state.get("result_json") is not None

        if rc == 0 and has_result_json:
            self._log.info("[execute_reflect] script succeeded; skipping repair")
            return {}

        iters = state.get("exec_reflect_iter", 0)
        if iters >= self.config.engine.exec_reflect_max_iterations:
            self._log.warning(
                "[execute_reflect] iterations exhausted (%d); proceeding to analyze with broken state",
                iters,
            )
            return {}

        history = list(state.get("exec_reflect_history") or [])
        history_block = _format_reflect_history(history)

        prompt = self._prompts["execute_reflect"].substitute(
            previous_code=(state.get("code") or "")[:8000],
            returncode=str(rc),
            stdout_tail=exec_result.get("stdout_tail", "")[:2000],
            stderr_tail=exec_result.get("stderr_tail", "")[:2000],
            duration_s=f"{exec_result.get('duration_s', 0):.2f}",
            figures_count=str(len(state.get("figures") or [])),
            result_json_present="yes" if has_result_json else "no",
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

    async def _node_analyze(self, state: QuestState) -> QuestState:
        self._log.info("[analyze] interpreting results")
        exec_result = state.get("exec_result") or {}
        prompt = self._prompts["analyze"].substitute(
            clarify_block=_format_clarify(state),
            design_block=json.dumps(state.get("design") or {}, indent=2),
            returncode=str(exec_result.get("returncode")),
            duration_s=f"{exec_result.get('duration_s', 0):.1f}",
            timed_out=str(exec_result.get("timed_out", False)),
            stdout_tail=exec_result.get("stdout_tail", "")[:2000],
            stderr_tail=exec_result.get("stderr_tail", "")[:1000],
            result_json=json.dumps(state.get("result_json") or {}, indent=2),
            figure_list="\n".join(f"- figures/{f}" for f in state.get("figures", [])) or "(none)",
        )
        text = await self._chat(prompt, node="analyze")
        analysis = _parse_json_lenient(text) or {"summary": "(parse failed)", "key_findings": []}
        # Default `next_step` to publish when the LLM omits it (older
        # prompts, parse failures) so the route doesn't break.
        analysis.setdefault("next_step", "publish")
        return {"analysis": analysis}

    async def _node_cross_check(self, state: QuestState) -> QuestState:
        """Phase L: for each key finding, search literature with the
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
            # Classify with one LLM call.
            cand_block = _format_lit(hits)
            prompt = self._prompts["cross_check"].substitute(
                topic=state.get("topic", "")[:1000],
                finding=text,
                candidate_literature=cand_block,
            )
            try:
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
            out.append({
                "finding": text,
                "supporting": parsed.get("supporting") or [],
                "conflicting": parsed.get("conflicting") or [],
                "neutral": parsed.get("neutral") or [],
                "summary": parsed.get("summary") or "",
                "candidates": candidates,
            })
        self._log.info("[cross_check] checked %d findings", len(out))
        patch: QuestState = {"cross_check": out}
        # Phase L iteration accounting: if analyze flagged a re-route AND
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

    async def _node_write(self, state: QuestState) -> QuestState:
        self._log.info("[write] authoring IMRAD paper.md")
        prompt = self._prompts["write"].substitute(
            topic=state["topic"],
            title=state.get("title", "Untitled"),
            design_block=json.dumps(state.get("design") or {}, indent=2),
            analysis_block=json.dumps(state.get("analysis") or {}, indent=2),
            literature_block=_format_lit_from_state(state),
            figure_list="\n".join(f"- figures/{f}" for f in state.get("figures", [])) or "(none)",
            clarify_block=_format_clarify(state),
            cross_check_block=_format_cross_check(state),
        )
        markdown = await self._chat(prompt, node="write")
        # The model may wrap with a fence; strip it.
        markdown = _strip_outer_fence(markdown)
        paper_path = self.quest_root / "paper" / "paper.md"
        paper_path.write_text(markdown, encoding="utf-8")
        self._log.info("[write] wrote %s (%d bytes)", paper_path, len(markdown))
        return {"paper_md": str(paper_path)}

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
            update: QuestState = {"review": review}
            if review.get("verdict") == "revise":
                update["iteration"] = state.get("iteration", 0) + 1
                self._log.info("[review] verdict=revise -> iteration %d", update["iteration"])
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
            return {
                "persona": name,
                "verdict": parsed.get("verdict") or "accept",
                "score": parsed.get("score") if isinstance(parsed.get("score"), (int, float)) else 3,
                "strengths": parsed.get("strengths") or [],
                "weaknesses": parsed.get("weaknesses") or [],
                "suggestions": parsed.get("suggestions") or [],
                "blocking": parsed.get("blocking") or "",
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
        if review.get("verdict") == "revise":
            update["iteration"] = state.get("iteration", 0) + 1
            self._log.info(
                "[review] panel verdict=revise (agreement=%s, score=%s) -> iteration %d",
                agg.get("agreement"), agg.get("score"), update["iteration"],
            )
        else:
            self._log.info(
                "[review] panel verdict=%s (agreement=%s, score=%s)",
                agg.get("verdict"), agg.get("agreement"), agg.get("score"),
            )
        return update

    # ---- helpers ---------------------------------------------------------

    async def _chat(self, prompt: str, *, node: str | None = None) -> str:
        """Single-user-message chat. ``node`` is the engine node name
        (e.g. ``"ideate"``, ``"review"``); when present and the YAML
        config sets ``provider.node_models[node]``, that model is sent
        on this call only. Otherwise the endpoint default applies."""
        assert self._client is not None
        messages = [{"role": "user", "content": prompt}]
        return await self._client.chat(
            messages, temperature=0.2, model=self._model_for_node(node),
            node=node or "",
        )

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
        return await self._client.chat(
            messages, temperature=temperature, model=self._model_for_node(node),
            node=node or "",
        )

    def _model_for_node(self, node: str | None) -> str | None:
        """Resolve the effective model for a node — empty string when
        the lookup misses so the transport falls through to the
        endpoint default. Accepts hierarchical keys like
        ``"review_panel.methodologist"`` (Phase N panel personas)."""
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
            missing.append("a LaTeX engine (pdflatex or tectonic)")
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


def _load_prompts() -> dict[str, string.Template]:
    names = (
        "clarify", "ideate", "ideate_reflect", "design", "implement",
        "execute_reflect", "analyze", "cross_check", "write", "review",
        "review_moderate",  # Phase N — panel-moderator prompt
        "data_load",        # no-simulation mode — synthesize result_json
                            # from user-supplied data
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


def _format_lit(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "(no prior work surfaced from the knowledge base)"
    lines: list[str] = []
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title") or d.metadata.get("source") or f"item-{i}"
        lines.append(f"[{i}] {title}\n{d.content[:_LIT_EXCERPT_CHARS]}")
    return "\n\n".join(lines)


def _format_lit_from_state(state: QuestState) -> str:
    items = state.get("literature") or []
    if not items:
        return "(no prior work surfaced from the knowledge base)"
    lines: list[str] = []
    for i, item in enumerate(items, start=1):
        meta = item.get("metadata") or {}
        title = meta.get("title") or meta.get("source") or f"item-{i}"
        lines.append(f"[{i}] {title}\n{item.get('content', '')[:_LIT_EXCERPT_CHARS]}")
    return "\n\n".join(lines)


_CLARIFY_LABELS = {
    "comparative_baseline": "Comparative baseline",
    "empirical_vs_theoretical": "Empirical / theoretical",
    "success_metric": "Success metric",
    "budget": "Time / compute budget",
    "output_kinds": "Desired output kinds",
    "study_depth": "Study depth",
    "paper_venue": "Paper venue / template",
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
    }


def _format_reflect_history(history: list[dict[str, Any]]) -> str:
    """Phase K: render the per-iteration repair history for the reflect
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


def _load_persona_prefix(name: str) -> str:
    """Phase N: load the persona-specific prefix that gets prepended to
    `agents/review.md` when this persona reviews. Falls back to a
    generic prefix when no per-persona file exists, so users can
    declare a custom persona name in YAML without shipping a new file."""
    if not _PERSONA_NAME_RE.match(name):
        raise ValueError(
            f"invalid persona name {name!r}: must match [a-z][a-z0-9_]*"
        )
    persona_path = PROMPTS_DIR / f"review_persona_{name}.md"
    if persona_path.exists():
        return persona_path.read_text(encoding="utf-8").strip()
    generic_path = PROMPTS_DIR / "review_persona_generic.md"
    if not generic_path.exists():
        return f"**Persona: {name}.**"
    template = string.Template(generic_path.read_text(encoding="utf-8"))
    return template.safe_substitute(persona_name=name).strip()


def _aggregate_panel_reviews(
    panel: list[dict[str, Any]], *, fallback_verdict: str = "accept",
) -> dict[str, Any]:
    """Phase N: deterministic aggregator the moderator prompt is also
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
                "weaknesses": [], "suggestions": [], "blocking": ""}

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

    return {
        "verdict": verdict, "score": median,
        "rigor_score": rigor_median, "depth_score": depth_median,
        "agreement": agreement,
        "strengths": strengths, "weaknesses": weaknesses,
        "suggestions": suggestions, "blocking": blocking,
    }


def _format_cross_check(state: QuestState) -> str:
    """Phase L: render the per-finding cross-paper-check results as a
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


def _extract_result_json(stdout: str) -> dict[str, Any] | None:
    if not stdout:
        return None
    matches = list(_RESULT_LINE_RE.finditer(stdout))
    if not matches:
        return None
    try:
        return json.loads(matches[-1].group(1))
    except json.JSONDecodeError:
        return None


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


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "untitled"


def _render_auto_collected_md(idx: int, meta: dict[str, Any], content: str) -> str:
    """Render an Axon retrieval hit as a Markdown file with YAML front
    matter. Used by ``_node_auto_collect_data`` (Phase D1).

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
    for key in ("source", "path", "title", "url", "kind", "year"):
        value = meta.get(key)
        if value:
            # Normalize: strip newlines that would break YAML's scalar
            # rules (multi-line provenance fields would otherwise need
            # quoted block style which clutters the file head).
            front[key] = str(value).replace("\n", " ").strip()
    yaml_block = yaml.safe_dump(front, default_flow_style=False, sort_keys=False)
    return f"---\n{yaml_block}---\n{content.strip()}\n"


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
    artifact-collection failure, OR the no-simulation pause-exit
    added in Phase B. (Earlier revisions of this fix only closed the
    handler on the success path; the outer try/finally is the version
    that actually delivers the cleanup invariant.)
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
    sessions; Phase B's no-simulation pause-exit adds another return
    path where the same leak would happen, so the fix lands here
    first."""
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
