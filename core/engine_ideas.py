"""Methods of :class:`core.engine.Engine` for the questions to the person, the ideas and the literature.

``Engine`` inherits :class:`IdeasStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, Counter, Path, QuestState, WORK_SCOPE_PAPERS, WORK_SCOPE_PAPERS_AND_BOOKS, _FILLED_AFTER_READ,
    _LITERATURE_QUERIES, _NO_CRITERIA_LINE, _NO_SIM_DESIGN_DIRECTIVE, _PERSON_QUERIES, _SURVEY_DESIGN_DIRECTIVE,
    _attempts, _audit_log, _audit_open, _clean_title, _default_clarify_questions, _design_parts, _fault_from_record,
    _fault_record, _format_clarify, _format_lit, _format_lit_excerpt, _format_lit_from_state, _frozen,
    _ingest_user_dropped_papers, _is_abstract_only, _is_open_access, _is_scholarly_record, _item_content,
    _labelled_sources, _leading_simulatability_word, _lit_query, _literature_entry, _literature_query_prompt,
    _literature_text_gaps, _merge_round_robin, _normalize_title, _novelty, _ok_calls, _one_line, _optim, _pairs,
    _papers_dir_has_files, _papers_to_ask, _parse_json_lenient, _person_queries, _phased, _plan, _query_set_key,
    _query_set_standing, _read_query_sets, _reading, _receipts, _record_query_set, _retractions,
    _search_record_has_abstract, _sf_snapshot, _simulatability_answer, _source_text, _spell_out_title_options,
    _take_papers_declined, _topic_req, _usable_queries, _write_paper_need_stubs, asyncio, functools, hashlib, json,
)


class IdeasStageMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    # ---- nodes -----------------------------------------------------------

    async def _node_clarify_then_phased(self, state: QuestState) -> QuestState:
        """The clarify step, then: when it found the quest runs no experiment of its own, explore-then-confirm has no
        design to run once more, so the rows held back at the start go back before the data is collected and analysed
        (off the event loop: the files may be large), and it is said once."""
        out = await self._node_clarify(state)
        if _phased.enabled(self.config) and isinstance(out, dict):
            merged = {**state, **out}
            why = self._phased_no_experiment_reason(merged)  # type: ignore[arg-type]
            if why:
                await asyncio.to_thread(self._phased_mark_not_applicable, why)
            elif self._phased_data_quest(merged):  # type: ignore[arg-type]
                # It analyses data instead of simulating: its data in data/ is held back too, before it is read.
                await asyncio.to_thread(self._phased_prepare_data_quest)
        return out

    async def _node_clarify(self, state: QuestState) -> QuestState:
        """Pre-flight clarification, then the quest's title.

        The title the user approved in the discussion (or the agent's own pick in ``auto``) replaces the raw
        topic as the quest's display title; a ``title`` set in the YAML always wins.
        """
        patch = await self._node_clarify_questions(state)
        answers = patch.get("clarify_answers") if isinstance(patch, dict) else None
        if self.config.title or not isinstance(answers, dict):
            return patch
        title = _clean_title(answers.get("title"))
        picks = (patch.get("clarify_questions") or {}).get("title", {}).get("suggestions") or []
        if title.isdigit() and 1 <= int(title) <= len(picks):
            title = picks[int(title) - 1]
        if title:
            patch = {**patch, "title": title}
            if self.config.pauses.clarify == "ask" or (
                    self.config.pauses.clarify is None and self._clarify_answerable):
                patch["title_confirmed"] = True
            self._log.info("[clarify] quest title: %s", title)
        return patch

    async def _node_clarify_questions(self, state: QuestState) -> QuestState:
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
        if mode is None:
            # Not set in the YAML: talk it over when someone is there to answer, else answer for yourself.
            mode = "ask" if self._clarify_answerable else "auto"
        if state.get("clarify_done"):
            return {}
        if mode == "off":
            self._log.info("[clarify] mode=off; skipping")
            # When clarify is skipped, only the YAML can switch on
            # no_simulation — ``engine.no_simulation`` or a pinned
            # ``clarify_overrides.simulatability``; there's no clarify answer.
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

        # The answers resume this node from its first line (LangGraph re-runs a node after its interrupt), so the
        # questions put to the person are kept on disk before the pause and read back here: the model is not asked
        # again, and the answers are matched to the questions the person actually saw (a title picked by number).
        questions = self._asked_clarify_questions(state["topic"])
        if questions is not None:
            self._log.info("[clarify] using the %d question(s) already asked (no second model call)", len(questions))
        else:
            prompt = self._prompts["clarify"].substitute(topic=state["topic"])
            text = await self._chat(prompt, node="clarify")
            questions = _parse_json_lenient(text) or {}
            if not isinstance(questions, dict) or not questions:
                # Degrade gracefully: if the LLM produced unparseable JSON,
                # synthesize a minimal default questionnaire from the topic
                # alone so the downstream nodes get *something*.
                self._log.warning("[clarify] LLM returned no parseable questions; using minimal defaults")
                questions = _default_clarify_questions(state["topic"])
            _spell_out_title_options(questions)
            # Measure, or find the best design: asked only when the topic's words leave it open
            # (core/optimisation_plan.py).
            if _optim.add_study_type_question(questions, state["topic"]):
                self._log.info("[clarify] the topic does not say whether it measures or looks for the best design: "
                               "that question is added")
        if self.config.title:
            questions.pop("title", None)

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

        # Kept as the model wrote them, before the pinned answers below are filled in: a resume after
        # ``--update`` changed a pin shows the new one. Removed when the quest finishes (not here: the node's
        # result is saved only after it returns, and a crash in between would ask the model again).
        self._keep_asked_clarify_questions(state["topic"], questions)
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
            answers = None
        if not isinstance(answers, dict) or not answers:
            # Resumed with nothing (or a non-dict): the questions' own defaults are the answers.
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
            An answer in the person's own words counts by its first word
            ("yes, but more than 100 lines" is "yes").
        3. **Legacy fallback** — ``empirical_vs_theoretical == "empirical"``
           → True. Kept for back-compat with quests started before the
           ``simulatability`` slot existed (resumes from old
           checkpoints, hand-written YAML answers, etc.). New quests
           should always have the simulatability slot populated.
        4. The YAML's ``engine.clarify_overrides.simulatability`` (what the
           interview wrote), when the clarify answer is missing, empty or
           does not start with yes / no / uncertain. Also with clarify off.

        Every resolution is logged at INFO level with the reason
        (when available) so the user can see exactly why the engine
        took whichever path it took — log line format:
        ``[clarify] simulatability resolved: NO_SIMULATION|SIMULATE
        (source=yaml|clarify_simulatability|clarify_empirical_legacy|yaml_clarify_overrides|default,
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
        decision, reason = _simulatability_answer(sim)
        # A person answers in their own words ("yes, but more than 100 lines"): the leading yes / no / uncertain is
        # the answer, the rest is their note.
        decision = _leading_simulatability_word(decision) or decision
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
            # Unknown / empty decision — fall through to the YAML's answer,
            # then the legacy fallback. Surface it though: an answer
            # outside the documented {yes, no, uncertain} set ("maybe", a
            # typo) silently changing the route is a routing bug waiting
            # to bite, so a WARNING names the value in run.log.
            if decision:
                self._log.warning(
                    "[clarify] the answer to \"can a simulation answer this?\" was %r, which does not start with "
                    "one of {yes, no, uncertain}; using the empirical_vs_theoretical legacy check, else the "
                    "setting in the YAML.", decision,
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

        # The YAML's own answer (engine.clarify_overrides.simulatability, which the interview writes): used when the
        # clarify answer left the question open (the person cleared the slot, wrote something with no yes / no in
        # front, or their answer did not include it) and the legacy check above did not decide. After that check, so
        # the interview's own "yes" (written for every quest not set to no-simulation) never outranks a topic judgment.
        pinned = (getattr(self.config.engine, "clarify_overrides", None) or {}).get("simulatability")
        pinned_decision = _leading_simulatability_word(_simulatability_answer(pinned)[0])
        if pinned_decision:
            self._log.info(
                "[clarify] simulatability resolved: %s (source=yaml_clarify_overrides, decision=%s)",
                "NO_SIMULATION" if pinned_decision == "no" else "SIMULATE", pinned_decision,
            )
            return pinned_decision == "no"

        self._log.info(
            "[clarify] simulatability resolved: SIMULATE "
            "(source=default, neither the clarify answer nor the YAML says yes, no or uncertain)",
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
        # topic alone. The search engines behind it match keywords, not the
        # topic's sentences: the same one-call keyword derivation the
        # literature step uses turns the topic into the field's terms first
        # (the topic as written when it gives none, as before).
        self.__dict__.setdefault("_last_call_id", {}).pop("ideate_query", None)
        self.__dict__.setdefault("_last_call_outcome", {}).pop("ideate_query", None)
        seed_queries = await self._derive_literature_queries(
            state["topic"], work_scope=self._work_scope(state), node="ideate_query",
        ) if self.knowledge.enabled else []
        if self.knowledge.enabled:
            if seed_queries:
                self._log.info("[ideate] seed search query derived from the topic: %r", seed_queries[0])
            ideate_outcome = self.__dict__.get("_last_call_outcome", {}).get("ideate_query")
            # Kept for a person to see (a failed one too); the literature step derives its own, from the topic AND the
            # chosen direction.
            _record_query_set(self.fi_dir, {
                "stage": "ideate", "queries": seed_queries,
                "model": (self._chat_provenance("ideate_query") or {}).get("model"),
                "call_id": self.__dict__.get("_last_call_id", {}).get("ideate_query"),
                "reason": "the idea step's grounding search (topic only)" + (
                    "" if seed_queries else "; the derivation gave no query "
                    + ("(its answer held none)" if ideate_outcome == "ok" else "(the call failed)")
                    + ", so the topic was searched as written"),
                "revised_by": "fi"}, log=self._log)
        seeded = await self.knowledge.asearch(
            seed_queries[0] if seed_queries else state["topic"], top_k=3,
            chat_fn=functools.partial(self._chat_messages, node="source_router"),
            work_scope=self._work_scope(state),
        )
        await self._flag_sources(seeded, "ideate")
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

        selection = ("ensemble" if ideate_skip_post_processing else "tournament" if tournament_ran
                     else "reflection" if critique else "model")
        out: QuestState = {"ideas": ideas, "chosen_idea": chosen, "idea_selection": selection}
        if critique:
            out["ideate_critique"] = critique
        if tournament_result:
            out["ideate_tournament"] = tournament_result
        self._record(_attempts.LEDGER, lambda: {
            "kind": "ideas",
            "candidates": [{"title": i.get("title"), "hypothesis": i.get("hypothesis")}
                           for i in (ideas if isinstance(ideas, list) else []) if isinstance(i, dict)],
            "chosen": chosen.get("title") if isinstance(chosen, dict) else None,
            "rule": selection,
            "why": (chosen.get("rationale") if isinstance(chosen, dict) else "") or "",
            "reflection": {"swap_to": critique.get("swap_to")} if isinstance(critique, dict) and critique else None,
            "tournament": tournament_result or None,
        })
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
            if isinstance(m, BaseException) and not isinstance(m, Exception):
                raise m  # a cancellation or a model that cannot be used (ProviderUnusable) is not a failed match
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
        # The same search pass with the same inputs uses the queries it derived before (a resume after a pause runs
        # this step again): no second call, and no drift to other queries finding other papers. A new pass, or
        # changed inputs, derives them again, and the record says why.
        title = chosen.get("title") or ""
        derive_prompt = _literature_query_prompt(state["topic"], title, hypothesis, work_scope=scope)
        # The model that derives them is part of what they answer: a different model derives them again.
        deriving_model = self._model_for_node("literature_query") or self.config.provider.model or ""
        key = _query_set_key(derive_prompt, this_iter, deriving_model)
        saved = [e for e in _read_query_sets(self.fi_dir, log=self._log) if e.get("stage") == "literature"]
        # Only an entry whose digest checks out is used again. One changed after FI wrote it is said so, never used.
        edited = [e for e in saved if e.get("key") == key and _query_set_standing(e) == "edited"]
        if edited:
            self._log.warning(
                "[literature] the saved search queries for this pass in .fi/%s were changed after FI wrote them; they "
                "are not used, and FI derives them again (to search with your own queries, put them in %s, one per "
                "line)", _LITERATURE_QUERIES, _PERSON_QUERIES.as_posix())
            self._audit("check_result", check="literature_queries", status="edited",
                        summary=f"{len(edited)} saved query set(s) for this pass no longer match their digest")
        ok_calls = _ok_calls(self.fi_dir, "literature_query")
        prompt_sha = hashlib.sha256(derive_prompt.encode("utf-8")).hexdigest()

        def _backed(e: dict[str, Any]) -> bool:
            # The entry names an answered call of this same prompt, and no other entry names that call.
            cid = e.get("call_id")
            return (cid in ok_calls and ok_calls[cid] == prompt_sha == e.get("prompt_sha256")
                    and sum(1 for o in saved if o.get("call_id") == cid) == 1)

        same = next((e for e in reversed(saved) if e.get("key") == key and _usable_queries(e.get("queries"))
                     and e.get("revised_by") != "person" and _query_set_standing(e) == "verified"
                     and _backed(e)), None)
        person = _person_queries(self.quest_root) if self.config.knowledge.enabled else None
        if person is not None:
            # A person's own queries: used as given, recorded as their revision (once per version of the file).
            queries, source_sha = person
            known = next((e for e in reversed(saved) if e.get("revised_by") == "person"
                          and e.get("source_sha256") == source_sha and e.get("iteration") == this_iter
                          and _query_set_standing(e) == "verified"), None)
            query_set = known or {
                "stage": "literature", "key": key, "iteration": this_iter, "queries": queries,
                "prompt_sha256": hashlib.sha256(derive_prompt.encode("utf-8")).hexdigest(), "model": None,
                "call_id": None, "reason": f"your queries in {_PERSON_QUERIES.as_posix()}", "revised_by": "person",
                "source_sha256": source_sha,
            }
            query_set = {**query_set, "reused": known is not None}
            if known is None:
                _record_query_set(self.fi_dir, query_set, log=self._log)
            self._audit("check_result", check="literature_queries", status="person",
                        summary=f"searched with the person's own queries ({_PERSON_QUERIES.as_posix()})",
                        source=_PERSON_QUERIES.as_posix(), source_sha256=source_sha, queries=queries)
            self._log.info("[literature] searching with your queries from %s: %r", _PERSON_QUERIES.as_posix(), queries)
        elif same is not None and self.config.knowledge.enabled:
            queries = [q.strip() for q in same["queries"]][:3]
            query_set = {**same, "reused": True}
            self._log.info("[literature] reusing the search queries derived for this pass (by %s): %r",
                           same.get("model") or "the model then", queries)
        else:
            if edited:
                reason = "the saved queries for this pass were changed after FI wrote them"
            elif any(e.get("key") == key and _query_set_standing(e) == "unverified" for e in saved):
                reason = "the saved queries for this pass carry no digest (written before FI recorded one)"
            elif any(e.get("key") == key and e.get("revised_by") != "person" and _usable_queries(e.get("queries"))
                     and not _backed(e) for e in saved):
                reason = ("the saved queries for this pass do not match an answered call in the record of model calls "
                          "(none, another prompt, or a call another entry also names)")
            elif any(e.get("iteration") == this_iter for e in saved):
                reason = "the topic, the chosen direction, the hypothesis or the model changed"
            elif saved:
                reason = f"a new search pass ({this_iter})"
            else:
                reason = "the first search"
            self.__dict__.setdefault("_last_call_id", {}).pop("literature_query", None)
            self.__dict__.setdefault("_last_call_outcome", {}).pop("literature_query", None)
            queries = await self._derive_literature_queries(
                state["topic"], title, hypothesis, work_scope=scope,
            )
            call_id = self.__dict__.get("_last_call_id", {}).get("literature_query")
            call_outcome = self.__dict__.get("_last_call_outcome", {}).get("literature_query")
            query_set = {
                "stage": "literature", "key": key, "iteration": this_iter, "queries": queries,
                "prompt_sha256": hashlib.sha256(derive_prompt.encode("utf-8")).hexdigest(),
                "model": ((self._chat_provenance("literature_query") or {}).get("model") or deriving_model or None)
                if queries else None,
                "call_id": call_id, "reason": reason, "revised_by": "fi", "reused": False,
            }
            if not queries:
                query_set["reason"] = (f"{reason}; the derivation gave no query "
                                       + ("(its answer held none)" if call_outcome == "ok" else "(the call failed)")
                                       + ", so the search used the topic and the chosen direction as written")
            # Every derivation is kept, a failed one too (its empty list is never reused, so the next run tries again).
            if self.config.knowledge.enabled:
                _record_query_set(self.fi_dir, query_set, log=self._log)
        derived = bool(queries)
        if queries:
            if not query_set.get("reused") and query_set.get("revised_by") != "person":
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
                # The query that found sources is the one later searches reuse, not the one that found none.
                query, derived = tried_queries[-1], True
        self._record_floor_verdicts(docs, filtered, stats, this_iter)
        docs = filtered
        # The original papers and textbooks a keyword search does not reach
        # join the candidates, and the screen judges them like the rest.
        docs = docs + await self._foundational_works(rel_topic, docs, work_scope=scope)
        # The floor scores word overlap; the screen asks whether the paper
        # could cite each source for a claim (see _screen_literature).
        n_before_screen = len(docs)
        docs = await self._screen_literature(rel_topic, docs, work_scope=scope, iteration=this_iter)
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
        # The whole text goes to disk (data/literature/.full_text/, up to knowledge.full_text_max_kb); the state keeps
        # its first _STATE_TEXT_CHARS, as it always held, so the checkpoint does not grow with a 10 MB source. Readers
        # that need the whole text (the passage selection, the claim check's quote check) read it back through
        # _item_content. content_quality says what the text is: full_text, abstract_only (not much more than the
        # abstract), preview_only (a table of contents), or snippet_only (nothing fetched).
        new_entries = [_literature_entry(self.quest_root, d.content, d.metadata) for d in docs]
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
        # A thread: a scanned paper is read by OCR here, about ten seconds a page, and the event loop runs the rest of
        # a --fleet meanwhile.
        merged, user_added = await asyncio.to_thread(
            _ingest_user_dropped_papers, self.quest_root, merged, seen, self._log,
        )
        if user_added:
            self._log.info(
                "[literature] picked up %d user-supplied paper(s) from inputs/papers/",
                user_added,
            )
        # Each DOI looked up in Crossref for a retraction (core/retractions.py); a retracted source is marked, never
        # used as support. No answer is "could not be checked", and nothing here stops the quest.
        if self.config.knowledge.enabled:
            merged, retraction_rows = await _retractions.check_literature(merged)
            if retraction_rows:
                self._record_source_verdicts("retractions", this_iter, "checked", retraction_rows)
                self._log.info("[literature] %s", _retractions.summary_line(retraction_rows))
        self._log.info(
            "[literature] retrieved %d docs (iter=%d, +%d new after dedup, "
            "+%d user-supplied, total=%d)",
            len(docs), this_iter, added, user_added, len(merged),
        )
        # Text in a source that is hidden from a reader or addressed to an AI model is flagged, never removed
        # (core/source_text.py): marked in the prompts, recorded in the audit trace, and the quest goes on.
        await self._flag_sources(merged, "literature", record_clean=True)
        qualities = Counter((e.get("metadata") or {}).get("content_quality") or "snippet_only" for e in merged)
        full_n = qualities.get("full_text", 0)
        # "Nothing fetched" is not "no abstract": a search record from OpenAlex / Crossref / Semantic Scholar carries
        # the paper's abstract. The line used to call all of those "snippet only", which read as if not even the
        # abstract had come back.
        not_fetched = [e for e in merged
                       if ((e.get("metadata") or {}).get("content_quality") or "snippet_only") == "snippet_only"]
        scholarly = [e for e in not_fetched if _is_scholarly_record(e.get("metadata") or {})]
        with_abstract = sum(1 for e in scholarly if _search_record_has_abstract(e.get("content") or ""))
        self._log.info(
            "[literature] full-text coverage: %d/%d sources have real full text (%d by OCR of a scanned PDF); "
            "%d papers have the abstract only (from the search, nothing downloaded), %d only a title or a one-line "
            "snippet, %d a download that held little more than the abstract, %d a preview or table of contents, "
            "%d web pages or notes from earlier quests",
            full_n, len(merged), sum(1 for e in merged if (e.get("metadata") or {}).get("full_text_ocr")),
            with_abstract, len(scholarly) - with_abstract,
            qualities.get("abstract_only", 0), qualities.get("preview_only", 0),
            len(not_fetched) - len(scholarly),
        )
        why = _literature_text_gaps(
            merged, _sf_snapshot(self.quest_id),
            downloads_on=bool(self.config.knowledge.enabled and self.config.knowledge.try_fetch_full_text))
        if why:
            self._log.info("[literature] why the full text is missing: %s", why)

        # Pause-for-user-papers gate. Fires only when the user opted in
        # AND we have abstract-only hits (heuristic: content shorter
        # than ~1500 chars or carrying an explicit ``abstract_only``
        # flag from the retriever). On a re-entry after the user
        # dropped PDFs, ``inputs/papers/`` is non-empty and we don't
        # pause again — the resume path picks up the new files and
        # proceeds.
        if not _papers_dir_has_files(self.quest_root):
            # A retracted paper is never asked for: it is not to be cited (core/retractions.py).
            withdrawn = _retractions.retracted_dois(merged)
            abstract_only = [d for d in docs if _is_abstract_only(d)
                             and _retractions.normalize_doi((d.metadata or {}).get("doi")) not in withdrawn]
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
            # Resuming from the papers pause without adding files is the answer "go on without them": those papers
            # are not asked for again. Only the papers not yet declined are asked for.
            already = 0
            if self.config.pauses.papers:
                declined_now = _take_papers_declined(self.fi_dir, self.quest_root,
                                                     getattr(self, "_resumed_from_pause", None))
                if declined_now:
                    self._audit("papers_declined", count=len(declined_now), papers=sorted(declined_now)[:40])
                    self._log.info("[literature] going on without %d paper(s) you were asked for (none was added to "
                                   "inputs/papers/): they are read from their abstracts only", len(declined_now))
                needed, already = _papers_to_ask(needed, self.fi_dir)
            if needed or oa_unfetched:
                _write_paper_need_stubs(
                    self.quest_root, needed, self._log,
                    query=_lit_query(state), oa_unfetched=oa_unfetched,
                    paused=bool(needed and self.config.pauses.papers),
                )
            if needed and not self.config.pauses.papers:
                # Unattended: FI downloads only free papers and never fetches a paywalled one itself.
                self._log.info(
                    "[literature] %d paper(s) are not confirmed free to read, so FI did not download them. They "
                    "are read from their abstracts; the list with a link each is in needs/WANTED_PAPERS.md if you "
                    "want to download some by hand.",
                    len(needed),
                )
            if needed and self.config.pauses.papers:
                self._pause_for_human(
                    kind="papers",
                    interaction="supply",
                    headline=f"download {len(needed)} paper(s) not confirmed free to read",
                    steps=[
                        f"{len(needed)} relevant paper(s) came back abstract-only "
                        "(not confirmed free to read). They're listed, most-relevant-first, in "
                        "`needs/WANTED_PAPERS.md` with a download link each."
                        + (f" ({already} other(s) you went on without earlier are not asked for again.)"
                           if already else ""),
                        "Download the few that matter and drop the PDFs into "
                        "`inputs/papers/` — they're ingested as full text.",
                        "Or go on without them: resume without adding files, and those papers are used from "
                        "their abstracts only (FI will not ask for them again).",
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
        else:
            left = sum(1 for d in docs if _is_abstract_only(d) and not _is_open_access(d))
            if left:
                self._log.info(
                    "[literature] %d paper(s) are not confirmed free to read and FI did not download them; "
                    "inputs/papers/ already has files, so no new list was written.", left,
                )

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

        out: QuestState = {
            "literature": merged,
            "literature_iter": this_iter,
            "literature_query": query.strip(),
            "literature_queries": queries,
            "literature_query_derived": derived,
            "literature_query_set": query_set,
        }
        novelty = await self._literature_novelty(state, merged, this_iter)
        if novelty:
            out["literature_novelty"] = novelty
        return out

    async def _literature_novelty(self, state: QuestState, merged: list, iteration: int) -> dict[str, Any] | None:
        """One call: do the retrieved sources already answer the study's question, and what does the study add?
        Disclosed only (core/novelty.py): the checked answer goes to ``.fi/literature_novelty.json`` and into the plan and
        writer prompts. It never stops the quest; a call that fails or gives nothing usable leaves no record."""
        if not self.config.knowledge.novelty_check or not merged:
            return None
        try:
            audience = self.config.output.audience  # the labels the writer cites by, so the checked ids can be cited
            sources = []
            for label, meta, item in _labelled_sources(merged, audience):
                full = _item_content(item)
                title = str(meta.get("title") or meta.get("source") or "")
                sources.append({
                    "id": label, "title": title, "year": meta.get("year") or "", "text": full,
                    "excerpt": " ".join(_format_lit_excerpt(full, title, budget=_novelty.EXCERPT_CHARS).split()),
                })
            if not sources:
                return None
            chosen = state.get("chosen_idea") or {}
            question = " ".join(str(chosen.get(k) or "").strip() for k in ("title", "rationale")).strip() or state["topic"]
            prompt = self._prompts["literature_novelty"].substitute(
                topic=state["topic"][:1500], question=question[:1200],
                # Retrieved text is material to read, never instructions (core/source_text.py).
                sources=_source_text.fence(_novelty.format_sources(sources)),
            )
            self.__dict__.setdefault("_last_call_id", {}).pop("literature_novelty", None)
            raw = await self._chat(prompt, node="literature_novelty")
            call_id = self.__dict__.get("_last_call_id", {}).get("literature_novelty")
            record, counts = _novelty.check(_parse_json_lenient(raw, node="literature_novelty"), sources)
            if record is None:
                self._log.warning("[literature] the note on what the sources already answer could not be read; "
                                  "the quest goes on without it")
                return None
            dropped = counts["dropped_unknown_source"] + counts["dropped_quote_not_found"]
            self._log.info(
                "[literature] sources already answer the question: %s; %d of %d evidence quotes checked and kept"
                "%s", record["already_answered"], counts["kept"], counts["given"],
                f" ({dropped} dropped: {counts['dropped_unknown_source']} unknown source, "
                f"{counts['dropped_quote_not_found']} quote not in the source)" if dropped else "")
            if not record["backed"]:
                self._log.info("[literature] no quote backs the %r answer, so it is kept in the record only",
                               record["already_answered"])
            _novelty.write_record(self.quest_root, record, call_id=call_id, iteration=iteration,
                                  model=(self._chat_provenance("literature_novelty") or {}).get("model")
                                  or self._model_for_node("literature_novelty") or self.config.provider.model)
            return record
        except Exception as e:  # noqa: BLE001 - a note that cannot be made never stops a quest
            self._log.warning("[literature] could not make the note on what the sources already answer (%s: %s); "
                              "the quest goes on without it", type(e).__name__, str(e)[:200])
            return None

    def _design_feedback(self, state: QuestState) -> str:
        """What a later design pass is told: the review, FI's precision record, the person's words, the points that need a
        new experiment, and the frozen protocol. Empty on the first pass."""
        iteration = state.get("iteration", 0)
        review_feedback = ""
        if iteration > 0:
            review_feedback = json.dumps(state.get("review", {}), indent=2)
        # A redesign sent from cross_check comes before any review, so the review block above is empty; without what
        # FI measured, the redesign would repeat the same size.
        if iteration > 0 and state.get("cross_check_route") == "redesign":
            # The only redesign after results is FI's own precision record (_cross_check_route): the design is told
            # which target was missed and the size its record says reaches it, not that the results were wrong.
            diag = "--- THE LAST RUN MISSED THE PLAN'S PRECISION TARGET ---\n" + self._precision_redesign_note(state)
            review_feedback = f"{review_feedback}\n\n{diag}".strip()
        # Human-feedback refinement (when the gate is configured AND the
        # user picked "refine"). Folded into the same review_feedback
        # block the design prompt already reads — explicitly attributed
        # so the LLM understands this came from a real user, not the
        # auto-review. Uses the accumulated ``feedback_history`` so a
        # later revise pass honours every prior ask, not just the most
        # recent one. Falls back to the single-shot ``human_feedback``
        # dict for legacy state shapes (resumed pre-history checkpoints).
        # (A note the search for the best design answered is not a change to the design: core/optimise_refine.py.)
        history = [h for h in state.get("feedback_history") or [] if not (isinstance(h, dict) and h.get("answered_by"))]
        hf = state.get("human_feedback") or {}
        if not history and hf.get("action") == "refine" and hf.get("feedback") and not any(
                isinstance(h, dict) and h.get("answered_by") for h in state.get("feedback_history") or []):
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
        needs = [str(p).strip() for p in state.get("refine_needs_experiment") or [] if str(p).strip()]
        if needs:
            review_feedback = (
                f"{review_feedback}\n\n"
                "--- POINTS OF THE USER'S FEEDBACK THAT NEED A NEW OR DIFFERENT EXPERIMENT "
                "(the writer could not answer them in the text) ---\n"
                + "\n".join(f"  - {p}" for p in needs) + "\n"
            ).strip()
        frozen_protocol = _frozen.protocol_of(self.quest_root)
        if frozen_protocol:
            review_feedback = (
                f"{review_feedback}\n\n"
                "--- FROZEN PROTOCOL (fixed before the first full run; keep it) ---\n"
                "Copy this `protocol` into your design unchanged. Do not change a grid value, the runs per setting, a "
                "threshold, the precision target, the stream policy or an oracle to make a result come out differently. If "
                "the redesign truly cannot be done under it, keep `protocol` as it is and add a `protocol_amendment` field: "
                "{\"protocol\": <the complete amended protocol>, \"reason\": \"<one or two sentences: what to change and "
                "why>\"}. A person then approves or refuses the amendment; if approved, the results seen so far are archived "
                "and this study is described as amended after its results were known.\n"
                f"{json.dumps(frozen_protocol, indent=2)}\n"
            ).strip()
        return review_feedback

    async def _redesign_scope(self, state: QuestState) -> list[str] | None:
        """The design parts a redesign may change, or ``None`` when no part is named (the whole design is then asked for
        again, as before). A missed precision target allows ONLY the parts that size the study, whatever else was said.
        Otherwise the parts the notes cite: a review cites the paths of the design it is about (``design_parts``, checked
        against the real design); a note with no path (a person's words, or a review that cited none) is mapped to the
        design's listed paths by the model, once (:meth:`_map_notes_to_parts`). No word of a note is matched by FI."""
        if state.get("cross_check_route") == "redesign":
            return _design_parts.allowed_for_precision()
        previous = state.get("design") if isinstance(state.get("design"), dict) else {}
        review = state.get("review") or {}
        parts: list[str] = []
        loose: list[str] = []  # notes that cite no path
        if isinstance(review, dict):
            cited = _design_parts.cited_parts(review.get("design_parts"), previous)
            parts += cited
            if not cited:
                loose += [str(t) for k in ("blocking", "why") for t in [review.get(k)] if t]
                loose += [str(t) for k in ("weaknesses", "suggestions") for t in (review.get(k) or []) if t]
        history = [h.get("text") for h in state.get("feedback_history") or [] if isinstance(h, dict) and not h.get("answered_by")]
        hf = state.get("human_feedback") or {}
        if not history and isinstance(hf, dict) and hf.get("action") == "refine" and hf.get("feedback"):
            history.append(hf["feedback"])
        loose += [str(t) for t in history if t] + [str(p) for p in state.get("refine_needs_experiment") or []]
        if loose:
            parts += [p for p in await self._map_notes_to_parts(loose, previous) if p not in parts]
        return parts or None

    async def _map_notes_to_parts(self, notes: list[str], design: dict[str, Any]) -> list[str]:
        """The listed design paths the model says the notes are about (one call, structured JSON, checked against the real
        design). A failed call, an unreadable answer or an empty one maps to nothing."""
        listed = _design_parts.part_paths(design)
        if not listed or not notes:
            return []
        prompt = self._prompts["design_parts_map"].substitute(
            notes="\n".join(f"- {n}" for n in notes), paths="\n".join(f"- `{p}`" for p in listed))
        try:
            text, obj = await self._chat_json(prompt, node="design", want=("parts",),
                                              usable=lambda p: isinstance(p.get("parts"), list))
        except Exception as e:  # noqa: BLE001 -- nothing mapped: the whole design is asked for, as before
            self._log.warning("[design] the notes could not be mapped to parts of the design (%r)", e)
            return []
        self._keep_plan_answer(text)
        mapped = _design_parts.cited_parts(obj.get("parts") if isinstance(obj, dict) else None, design)
        self._log.info("[design] the notes were mapped to: %s", ", ".join(f"`{m}`" for m in mapped) or "(no part of the design)")
        return mapped

    async def _redesign_by_parts(self, state: QuestState) -> dict[str, Any] | None:
        """A redesign (the design step run again after a review, a person's note or a missed precision target) that shows
        the previous design and asks the model for ONLY the parts the reason concerns; every other part is carried over
        exactly as it was, and a part the model returns that was not asked for is refused. ``None`` when this is not a
        redesign, no part is named, or nothing asked came back: the whole design is then asked for, as before."""
        previous = state.get("design")
        if (state.get("iteration", 0) <= 0 or not isinstance(previous, dict) or not previous or self._client is None
                or state.get("survey_mode_resolved")):
            return None
        allowed = await self._redesign_scope(state)
        if allowed is None:
            self._log.info("[design] no note names a part of the design, so the whole design is asked for again")
            return None
        self._log.info("[design] the redesign may change only: %s; every other part is carried over as it was",
                       ", ".join(f"`{a}`" for a in allowed))
        prompt = self._prompts["design_parts"].substitute(
            topic=state["topic"], design=json.dumps(previous, indent=2, default=str),
            reason=self._design_feedback(state) or "(no reason was recorded)",
            allowed="\n".join(f"- `{a}`" for a in allowed), timeout_s=str(self.config.execution.timeout_s),
            time_note=self._time_note(state), clarify_block=_format_clarify(state))
        text, obj = await self._chat_json(prompt, node="design", want=(), usable=lambda p: bool(p))
        self._keep_plan_answer(text)
        design, applied, refused = _design_parts.merge_parts(previous, obj or {}, allowed)
        needs, not_needs = _design_parts.accepted_needs(previous, obj or {}, allowed)
        for n in needs:
            self._log.info("[design] the redesign also needed `%s`: %s", n["path"], _one_line(n["why"], 200))
        for r in not_needs:
            self._log.warning("[design] the redesign's need for `%s` was not accepted: %s", r["key"], r["reason"])
        for r in refused:
            self._log.warning("[design] the redesign's change to `%s` was not made: %s", r["key"], r["reason"])
        if not applied:
            self._log.warning("[design] none of the asked parts came back changed; the whole design is asked for again")
            return None
        self._log.info("[design] the redesign changed %s and kept every other part as it was",
                       ", ".join(f"`{a}`" for a in applied))
        return design

    def _design_prompt(self, state: QuestState) -> str:
        """The design prompt for this state: the topic, the chosen direction, the literature, the
        clarifications, the skills and example files, and, on a later pass, what the review and the person said."""
        review_feedback = self._design_feedback(state)
        prompt = self._prompts["design"].substitute(
            topic=state["topic"],
            chosen_idea=json.dumps(state.get("chosen_idea") or {}, indent=2),
            # The full excerpts, as analyze and write get. At 800 characters a
            # source, design chose a 1% major-outbreak threshold in 6 of 10
            # replays of a real quest (0 of 5 at the full budget): short
            # excerpts cost it the context a sound design rests on.
            literature_block=_format_lit_from_state(state, **self._lit_kwargs(state)),
            novelty_block=_novelty.design_block(state.get("literature_novelty")),
            review_feedback=review_feedback or "(none — first iteration)",
            timeout_s=str(self.config.execution.timeout_s),
            time_note=self._time_note(state),
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
        checklist = self._topic_checklist(state)
        return f"{prompt}\n\n{checklist}\n" if checklist else prompt

    def _topic_checklist(self, state: QuestState) -> str:
        """What the topic asks for, as plain lines the plan or design is shown before it is written
        (:func:`core.topic_requirements.checklist_block`), read from the kept reading (``.fi/topic_requirements.json``).
        Empty when the quest runs no code, the protocol is frozen, or the topic was not read (a failed or empty reading):
        the prompt is then as it was."""
        if not self._runs_code(state) or _frozen.load(self.quest_root) is not None:
            return ""
        req = self._topic_requirements_read()
        if not isinstance(req, dict) or req.get("error"):
            return ""
        try:
            return _topic_req.checklist_block(req, topic=state.get("topic") or self.config.topic)
        except Exception as e:  # noqa: BLE001 -- a block that cannot be built leaves the prompt as it was
            self._log.warning("[plan] the list of what the topic asks for could not be built: %r", e)
            return ""

    async def _read_topic_before_the_prompt(self, state: QuestState) -> None:
        """Read what the topic asks for (:meth:`_read_topic_requirements`: one call, kept, never asked again on a resume)
        BEFORE the plan or the design is asked for, so the request can show it. It depends only on the topic. A reading
        that fails is kept as an error and the request goes out without the list, as before."""
        if not self._runs_code(state) or self._client is None or _frozen.load(self.quest_root) is not None:
            return
        await self._read_topic_requirements()

    async def _audit_design(
        self, state: QuestState, design: dict[str, Any], *, adopt: bool = True, writer: str = "design",
    ) -> tuple[dict[str, Any], Any]:
        """The second-pass methodology audit of a drafted design: ``(design, objections addressed)``. With
        ``adopt=False`` it only checks a design that is already settled (one a person edited in plan.md, or an amended
        one): nothing it proposes is taken, and the receipt passes only when it had no objection.

        It is the design's reviewer (core/review_pairs.py): the step ``design_review`` when a person set a model or a
        provider for it, else the main model's own audit as before. ``writer`` is the step that wrote the design (the
        plan step, or the design step). What the reviewer is, against that writer, is recorded with the audit: a
        different model is an independent review, the same model a self-review, and never the one as the other."""
        iteration = state.get("iteration", 0)
        # The design's ``rationale`` is the model's own account for the audit trace, not part of what the audit reviews or
        # what its amended design must repeat (its reply shape does not list it): it is set aside here and put back after.
        rationale = design.get("rationale")
        design = {k: v for k, v in design.items() if k != "rationale"}
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
        review_node = self._review_node("design_review")
        critique_prompt = self._prompts["design_self_critique"].substitute(
            topic=state["topic"],
            chosen_idea=json.dumps(state.get("chosen_idea") or {}, indent=2),
            clarify_block=_format_clarify(state),
            draft_design=json.dumps(design, indent=2),
        )
        critique: dict[str, Any] = {}
        failure = ""
        not_reviewed = False
        # A reviewer that is shown to be another model than the design's writer must find problems or say what it
        # checked: a reply with neither is no review (asked once more, then recorded as not reviewed).
        strict = self._reader_vs_planner(review_node, after_call=False, writer=writer)[0] is False

        def _nothing_said(p: dict[str, Any]) -> bool:
            return _pairs.said_nothing(_pairs.findings(p.get("objections_addressed")), _pairs.checked(p.get("checked_no_problem")))

        audit_started = _receipts.now()
        before = json.loads(json.dumps(design, default=str))
        try:
            # An unreadable reply is asked for once more, short (_chat_json); a second unreadable one is a failure.
            raw_reply, critique = await self._chat_json(
                critique_prompt, node=review_node, want=("objections_addressed",),
                usable=lambda p: isinstance(p.get("objections_addressed"), list) and not (strict and _nothing_said(p)),
                why_unusable=lambda p: ("it listed no objection and did not say what it checked: put every problem under "
                                        "objections_addressed, and what you checked and found no problem with under "
                                        "checked_no_problem") if strict and isinstance(p.get("objections_addressed"), list)
                and _nothing_said(p) else "",
            )
            self._keep_plan_answer(raw_reply)  # the audit's own answer, kept so a refusal of part of it can be replayed
            if strict and isinstance(critique, dict) and isinstance(critique.get("objections_addressed"), list)                     and _nothing_said(critique):
                failure = f"the design was not reviewed: {_pairs.NOT_A_REVIEW}"
                not_reviewed = True
                critique = {}
        except Exception as e:  # noqa: BLE001 — see "Failure isolation" above
            failure = f"the audit call or its reply could not be used: {e!r}"
            self._log.warning(
                "[design_self_critique] chat/parse failed (%r); keeping "
                "un-audited draft design", e,
            )
        whole = critique.get("amended_design") if isinstance(critique, dict) else None
        objections = critique.get("objections_addressed") if isinstance(critique, dict) else None
        not_applied: list[dict[str, str]] = []  # repairs FI did not apply: they would not pass its own checks
        amended: dict[str, Any] | None = None
        unnamed: list[dict[str, str]] = []  # changes no objection named: refused before the schema reading
        if not failure and not isinstance(objections, list):
            # An audit judges by naming its objections (none is an empty list). An unreadable reply, or one that names
            # none, is not an audit that found nothing.
            failure = "the audit's reply did not list its objections"
            objections = None
        # The audit names the parts it changes (``changes`` under each objection); FI makes those and no other. A part of
        # a whole amended design (an older reply's shape) that no objection names is refused (core/design_parts.py).
        if isinstance(objections, list) and not failure:
            if isinstance(whole, dict) and whole and not set(design).issubset(set(whole)):
                failure = f"the amended design dropped the keys {sorted(set(design) - set(whole))}; the draft was kept"
                self._log.warning("[design_self_critique] amended_design dropped keys %s; keeping draft",
                                  sorted(set(design) - set(whole)))
                whole = None
            if not failure:
                amended, unnamed, _applied = _design_parts.amended_from_audit(design, objections, whole)
                for n in unnamed:
                    self._log.warning("[design_self_critique] the audit's change to `%s` was not made: %s", n["key"], n["reason"])
        proposed_change = (
            isinstance(amended, dict) and bool(amended)
            and _receipts.design_core(amended) != _receipts.design_core(design)
        ) or bool(unnamed)
        if not adopt:
            amended = None
        if isinstance(amended, dict) and amended:
            # Only the parts the audit named differ from the draft, so the shape is the draft's. On a part FI's own
            # reading of the plan would refuse, the audit is asked once to correct that part.
            draft_design = design
            design, not_applied = self._hold_audit_repairs_to_the_schema(draft_design, amended)
            if any(n.get("part") for n in not_applied):
                design, not_applied = await self._ask_the_audit_to_correct_parts(
                    draft_design, amended, design, not_applied, review_node)
        if adopt:
            not_applied = [*not_applied, *unnamed]
        n_addressed = len(objections) if isinstance(objections, list) else 0
        self._log.info(
            "[design_self_critique] iteration=%d objections_addressed=%d",
            iteration, n_addressed,
        )
        same, reviewer, writer_model, named = self._reader_vs_planner(review_node, after_call=True, writer=writer)
        review_info = {"kind": "not_reviewed" if not_reviewed else "independent_review" if same is False
                       else "self_review" if same else "unclear",
                       "reviewer": reviewer, "writer": writer_model, "same_model": same, "reported": named,
                       "step": review_node}
        self._record_design_critique(iteration, before, design, objections, failure, node=review_node, review=review_info,
                                     not_applied=not_applied)
        if not failure and isinstance(objections, list):
            self._keep_audit_open_items(
                objections, design, reasons=self._audit_reasons(not_applied),
                default_reason="FI did not apply it" if adopt else "the design was already settled, so the audit only checked it",
                iteration=int(iteration or 0), before=before)
        if not failure:
            self._log.info(
                "[design_self_critique] the design was read by %s: %s", reviewer,
                "a different model than the one that wrote it" if same is False else
                "the model that wrote it (a self-review, not an independent one)" if same else
                f"FI cannot tell whether it is another model than the one that wrote it ({writer_model})")
        objected = [o.get("objection") if isinstance(o, dict) else o for o in objections or []] if not adopt or not_applied else []
        if not adopt and not failure and proposed_change and not objected:
            objected = ["it proposed a different design without naming why"]
        if not_applied and not objected:  # a repair FI could not apply is an objection that stands, named or not
            objected = [f"the audit's change to `{n['key']}` could not be applied: {n['reason']}" for n in not_applied]
        self._write_receipt(
            "design_audit", "unknown" if failure else "fail" if objected else "pass", started_at=audit_started,
            inputs={"design": before}, output=_receipts.design_core(design), error=failure,
            detail=(("it objected to the design that ran: " if not adopt else "its repair could not be applied in full, so the "
                     "objection stands: ") + "; ".join(str(o)[:160] for o in objected[:3])
                    + ("".join(f" [FI did not {'fully ' if n.get('applied') else ''}apply `{n['key']}`: {n['reason'][:160]}]" for n in not_applied[:3])
                       if not_applied else "")) if objected else "",
        )
        if failure and not_reviewed:
            # The reviewer answered but did not review: a gap the evidence level records, not a stop for a person.
            self._log.warning("[design_self_critique] %s; the receipt records it as a gap", failure)
        elif failure:
            self._stop_once_for_check(
                "design_audit", failure, look="needs/DESIGN_CRITIQUE.json and the [design_self_critique] entry in .fi/run.log",
                inputs={"design": before},
            )
        else:
            self._clear_check_stop("design_audit")
        if rationale is not None:
            design = {**design, "rationale": rationale}
        return design, objections

    def _hold_audit_repairs_to_the_schema(
        self, before: dict[str, Any], amended: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """``(design, not_applied)``: ``amended`` with every repair that FI's own reading of the plan would refuse put back
        as it was in ``before``, and one ``{"key", "reason"}`` per repair left out. A repair is refused when the design
        read as the plan step reads it (:func:`core.plan.repair_protocol`, :func:`core.plan.normalize_design`) loses a part
        with it that it kept without it: the reviewer's repair of a value must not cost the block it is in. The reviewer's
        objection stays reported; only the change is not made."""
        not_applied: list[dict[str, str]] = []
        out = dict(amended)
        old_p, new_p = before.get("protocol"), amended.get("protocol")
        if isinstance(old_p, dict) and isinstance(new_p, dict):
            # The settings to compare and the parts of the study are made together or not at all, in the protocol and in
            # a search's block alike (a change to one is read with the other).
            new_p = dict(new_p)
            for where, mine, theirs in (("protocol", old_p, new_p),
                                        ("protocol.optimisation", old_p.get("optimisation"), new_p.get("optimisation"))):
                if where == "protocol" and isinstance(old_p.get("optimisation"), dict):
                    continue  # a search keeps its scan in its block
                if not isinstance(mine, dict) or not isinstance(theirs, dict):
                    continue
                kept_division, why_not, refused = _plan.audit_scan_division(mine, theirs, where)
                if why_not is None:
                    continue
                if where == "protocol":
                    new_p = kept_division
                else:
                    new_p["optimisation"] = kept_division
                not_applied.extend({"key": f"{where}.{part}", "reason": why_not, "part": "yes", "faults": [_fault_record(
                    _reading.Fault(f"{where}.{part}", "together", why_not, as_written=theirs.get(part), required=True))]}
                    for part in refused)
            kept_p, held = self._read_the_audits_protocol(old_p, new_p)
            not_applied.extend(held)
            out["protocol"] = kept_p

        def readable(d: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
            """The design as the plan step reads it: with the protocol's unreadable keys left out, as it does."""
            got, why = _plan.normalize_design(d)
            if got is None and "`protocol" in (why or ""):
                fixed_p, notes_p = _plan.repair_protocol(d.get("protocol"))
                if fixed_p is None:  # a protocol nothing of which can be read is not a design the plan step reads
                    return None, why or (notes_p[0] if notes_p else "the protocol cannot be read")
                got, why = _plan.normalize_design({**{k: v for k, v in d.items() if k != "protocol"},
                                                   **({"protocol": fixed_p} if fixed_p else {})})
            return got, why

        if readable(before)[0] is not None:
            usable, why = readable(out)
            if usable is None:  # the design was readable before the repair and is not with it: nothing of it is applied
                changed = sorted(k for k in {*out, *before} if out.get(k) != before.get(k))
                out = dict(before)
                not_applied.append({"key": ", ".join(changed) or "design", "reason": " ".join(str(why).split())[:300]})
        for item in not_applied:
            self._log.warning("[design_self_critique] the audit's change to `%s` was %s: %s", item["key"],
                              "applied except one part" if item.get("applied") else "not applied", item["reason"])
        return out, not_applied

    def _read_the_audits_protocol(
        self, old_p: dict[str, Any], new_p: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """``(protocol, items)``: the protocol the audit rewrote, read once, entry by entry (:func:`core.plan.read_protocol`),
        against the plan's own. Only a fault the plan's own reading did not have counts, and only at the path it is at: a
        part the audit changed that cannot be read is left out alone (its path named), and a key of the plan's own block
        that the rewrite lacks is put back as the plan had it. A fault in a part the study cannot do without (a scan's
        parts, a search's objective, baseline or limit) keeps the plan's own of that key whole. ``items``: one ``{"key",
        "reason", "part", "faults"}`` per key the audit changed that was not made in full (the faults, each at its path, are
        what the audit is asked to correct once, :meth:`_ask_the_audit_to_correct_parts`)."""
        base_faults = _plan.read_protocol(old_p).faults
        known = {(f.path, f.reason) for f in base_faults}

        def new_in(faults: list[Any], current: dict[str, Any]) -> list[Any]:
            """The faults the plan's own reading did not have: at a path it had no fault at, or with another reason at the
            same path (a different unreadable value is the audit's own; a change to another field of the same entry is not)."""
            return [f for f in faults if (f.path, f.reason) not in known]
        changed = sorted(k for k in {*old_p, *new_p} if (k in old_p) != (k in new_p) or new_p.get(k) != old_p.get(k))
        kept_p = dict(new_p)
        items: list[dict[str, Any]] = []

        def key_of(fault: Any) -> str:
            steps = _reading.split(fault.path)
            return str(steps[1]) if len(steps) > 1 else ""

        def say(key: str, faults: list[Any], parts: dict[str, str] | None = None) -> dict[str, Any]:
            """The item of a key not made in full: its faults, each at its path (the audit is asked about them once). It says
            the change was made in part (``parts``: each part left out, by its path) only when something of it was made."""
            why = "; ".join(" ".join(str(f.reason).split()) for f in faults[:3])
            item: dict[str, Any] = {"key": f"protocol.{key}", "reason": why[:300], "part": "yes",
                                    "faults": [_fault_record(f) for f in faults]}
            if parts is not None and kept_p.get(key) != old_p.get(key):
                item.update(applied="all of it except the part named", parts=parts)
            return item

        def left_out(faults: list[Any]) -> dict[str, str]:
            """Each part left out, by the part's own path (the part of the block, not the place inside it)."""
            parts: dict[str, str] = {}
            for f in faults:
                at = "protocol"
                for step in _reading.split(f.path)[1:3]:
                    at = _reading.join(at, step)
                parts.setdefault(at, " ".join(f"could not be read: {f.reason}".split())[:300])
            return parts

        def stops(fault: Any, key: str, kept_now: dict[str, Any]) -> bool:
            """A fault in a part the study cannot do without (a part of a list of limits or of scan parts, a search
            block that can no longer drive a search): the plan's own of that key stays whole."""
            steps = _reading.split(fault.path)
            return bool(fault.required and fault.kind != "conflict"
                        and (len(steps) <= 2 or any(isinstance(x, int) for x in steps) or key not in kept_now))

        for key in changed:
            if key in old_p and key not in new_p:
                kept_p[key] = old_p[key]
                items.append({"key": f"protocol.{key}", "reason": f"the audit left `protocol.{key}` out"})
        refused: set[str] = set()
        # The keys with a fault that stops them are put back first, one read after another: a conflict between two keys
        # (a scan written beside a block that has its own) may be the other key's fault alone.
        for _round in range(len(changed) + 1):
            reading = _plan.read_protocol(kept_p)
            kept_now = reading.kept if isinstance(reading.kept, dict) else {}
            fresh = new_in(reading.faults, kept_p)
            stopping = [k for k in changed if k not in refused and k in kept_p and any(
                key_of(f) == k and stops(f, k, kept_now) for f in fresh)]
            if not stopping:
                break
            for key in stopping:
                faults = [f for f in fresh if key_of(f) == key]
                trial = self._plans_own_at(old_p, kept_p, key, [f for f in faults if stops(f, key, kept_now)])
                if trial is not None and not any(
                        stops(f, key, {key: 1}) for f in new_in(_plan.read_protocol(trial).faults, trial)
                        if key_of(f) == key):
                    # Only the part that cannot be read goes back as the plan had it; the rest of the change is made.
                    kept_p = trial
                    items.append(say(key, faults, left_out(faults)))
                    refused.add(key)
                    continue
                if key in old_p:
                    kept_p[key] = old_p[key]
                else:
                    kept_p.pop(key, None)
                refused.add(key)
                items.append(say(key, faults))
        reading = _plan.read_protocol(kept_p)
        kept_now = reading.kept if isinstance(reading.kept, dict) else {}
        fresh = new_in(reading.faults, kept_p)
        for key in changed:
            faults = [f for f in fresh if key_of(f) == key]
            if key in refused or key not in kept_p or kept_p.get(key) == old_p.get(key) or not faults:
                continue
            if key not in kept_now:  # the whole key cannot be read: the plan's own stays
                if key in old_p:
                    kept_p[key] = old_p[key]
                else:
                    kept_p.pop(key, None)
                items.append(say(key, faults))
                continue
            kept_p[key] = kept_now[key]
            for name, cause in self._put_back_the_plans_keys(old_p, kept_p, key, new_p.get(key)):
                self._log.warning("[design_self_critique] the audit's `protocol.%s.%s` %s, so that change was not applied: "
                                  "the plan's own is kept as it was", key, name, cause)
            items.append(say(key, faults, left_out(faults)))
        # A fault the plan's own reading did not have, left in the protocol as it is now, is the audit's change somewhere
        # else (a part it left as it was can no longer be read beside it): that change is put back, one key at a time.
        for _round in range(len(changed) + 1):
            stray = new_in(_plan.read_protocol(kept_p).faults, kept_p)
            if not stray:
                break
            for key in changed:
                if kept_p.get(key) == old_p.get(key):
                    continue
                trial = {k: v for k, v in kept_p.items() if k != key}
                if key in old_p:
                    trial[key] = old_p[key]
                if len(new_in(_plan.read_protocol(trial).faults, trial)) < len(stray):
                    kept_p = trial
                    items.append(say(key, stray))
                    break
            else:
                break
        return kept_p, items

    @staticmethod
    def _plans_own_at(old_p: dict[str, Any], kept_p: dict[str, Any], key: str, faults: list[Any]) -> dict[str, Any] | None:
        """``kept_p`` with the plan's own value put back at the path of each of ``faults`` (a part the study cannot do without
        that the audit's version of cannot be read), or ``None`` when the plan has no readable one there to put back. An
        entry of a list goes back at its own number only when the audit's list is as long as the plan's (a change made in
        place); a longer or shorter list is not matched entry by entry."""
        seen = set(_plan.read_protocol(old_p).paths())
        out = kept_p
        for fault in faults:
            steps = _reading.split(fault.path)
            if fault.path in seen or len(steps) < 3:
                return None
            at = steps[0]
            for step in steps[1:]:  # every list on the way to the fault must be as long as the plan's
                if isinstance(step, int):
                    theirs, mine = _reading.value_at(old_p, at, "protocol"), _reading.value_at(out, at, "protocol")
                    if not (isinstance(theirs, list) and isinstance(mine, list) and len(theirs) == len(mine)):
                        return None
                at = _reading.join(at, step)
            own = _reading.value_at(old_p, fault.path, "protocol")
            put = None if own is None else _reading.put_at({"protocol": out}, fault.path, own)
            if put is None:
                return None
            out = put["protocol"]
        return out

    def _put_back_the_plans_keys(self, old_p: dict[str, Any], kept_p: dict[str, Any], key: str,
                                 written: Any) -> list[tuple[str, str]]:
        """Each key of the plan's own ``old_p[key]`` (a mapping) that the rewrite's reading ``kept_p[key]`` lacks, put back as
        the plan had it into ``kept_p`` when the protocol reads with it no worse than without. A rewrite of one key of a
        block must not cost the block a key it had (a real audit rewrote a search's block with an evaluation budget that
        could not be read, and the plan's own readable budget went with it, so the search could not start). Returns
        ``(key, cause)`` for each one put back."""
        old = old_p.get(key)
        if not isinstance(old, dict) or not isinstance(kept_p.get(key), dict):
            return []
        wrote = written if isinstance(written, dict) else {}
        back: list[tuple[str, str]] = []
        for name, value in old.items():
            if name in kept_p[key]:
                continue
            worse = {f.path for f in _plan.read_protocol(kept_p).faults}
            trial = {**kept_p, key: {**kept_p[key], name: value}}
            reading = _plan.read_protocol(trial)
            held = reading.kept.get(key) if isinstance(reading.kept, dict) else None
            if not isinstance(held, dict) or name not in held or any(f.path not in worse for f in reading.faults):
                continue
            kept_p[key] = trial[key]
            back.append((str(name), "could not be read" if name in wrote else "was left out of its rewrite"))
        return back

    async def _ask_the_audit_to_correct_parts(
        self, before: dict[str, Any], amended: dict[str, Any], design: dict[str, Any], not_applied: list[dict[str, Any]],
        node: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """``(design, not_applied)`` after the audit's model has been asked ONCE to correct the parts of its change that FI
        could not read: one request that names only the faulted paths, each with FI's reason, the entry as the audit wrote
        it and the values allowed. What comes back is placed at those paths in the audit's change and read again; whatever
        is still unreadable stays left out (the plan's own of a part the study cannot do without) and is named in
        ``not_applied``. A failed call or an unreadable answer leaves the first reading as it was. The faults, what was
        asked and what was left out are kept in ``needs/audit_readings.json``."""
        asked = [_fault_from_record(r) for n in not_applied for r in n.get("faults") or []]
        paths = list(dict.fromkeys(f.path for f in asked))
        asked_keys = {str(_reading.split(p)[1]) for p in paths if len(_reading.split(p)) > 1}
        new_p = amended.get("protocol") if isinstance(amended.get("protocol"), dict) else {}
        self._log.warning("[design_self_critique] asking the audit once to correct %s, with FI's reason for each",
                          ", ".join(f"`{p}`" for p in paths))
        request = _reading.ask_text(asked, "Your amended design could not be used in full.") + (
            '\n\nReturn ONLY one JSON object of the form {"fixes": {"<path>": <the whole corrected part at that path>}} '
            "with one entry for each path above, written exactly as named. Fix only what the reason names; keep every other "
            "part exactly as you wrote it and add nothing. If the fix also needs another part of the same protocol key (for "
            "example the settings a part of the study varies), give that part too, by its path.")
        try:
            raw_correction, reply = await self._chat_json(
                request, node=node, want=("fixes",), usable=lambda p: isinstance(p.get("fixes"), dict))
        except Exception as e:  # noqa: BLE001 — the audit is advisory: the first reading stands
            self._log.warning("[design_self_critique] the correction could not be asked for (%r); the first reading stands", e)
            self._keep_audit_reading(asked, paths, not_applied, answered=False)
            return design, not_applied
        self._keep_plan_answer(raw_correction)
        given = reply.get("fixes") if isinstance(reply, dict) else None
        placed = dict(new_p)
        used = 0
        for path, value in (given or {}).items():
            steps = _reading.split(path) if isinstance(path, str) else []
            if not steps:
                # A path read only in part names no place: nothing of it is written (never over the part around it).
                self._log.warning("[design_self_critique] the correction for `%s` was not used: FI cannot read that path "
                                  "in full", path)
                continue
            if not (path in paths or (len(steps) > 1 and str(steps[1]) in asked_keys)):
                self._log.warning("[design_self_critique] the correction for `%s` was not used: it was not asked about", path)
                continue
            at = _reading.put_at(placed, path, value, "protocol")
            if at is None:
                self._log.warning("[design_self_critique] the correction for `%s` was not used: the design has no such "
                                  "place", path)
                continue
            placed, used = at, used + 1
        if not used:
            self._log.warning("[design_self_critique] the correction named none of the asked parts; the first reading stands")
            self._keep_audit_reading(asked, paths, not_applied, answered=False)
            return design, not_applied
        design2, not_applied2 = self._hold_audit_repairs_to_the_schema(before, {**amended, "protocol": placed})
        left = {f["path"] for n in not_applied2 for f in n.get("faults") or []}
        for path in paths:
            self._log.info("[design_self_critique] `%s` after the correction: %s", path,
                           "still not fully readable" if path in left else "read in full")
        self._keep_audit_reading(asked, paths, not_applied2, answered=True, dropped=[p for p in paths if p in left])
        return design2, not_applied2

    def _keep_plan_reading(self, reading: Any, step: str) -> None:
        """What a plan answer's reading left out (each fault at its path) and moved, kept in ``needs/plan_readings.json``:
        every part ends kept, asked or left out, on record. A record that cannot be written raises, as for the audit's."""
        if not (reading.faults or reading.moved):
            return
        entry = _reading.record_entry(reading, step=step, dropped=[f.path for f in reading.faults])
        _reading.write_record(self.quest_root / "needs" / "plan_readings.json", entry)

    def _keep_audit_reading(self, faults: list[Any], asked: list[str], refused: list[dict[str, Any]], *, answered: bool,
                            dropped: list[str] | None = None) -> None:
        """The audit's reading kept (``needs/audit_readings.json``, and best-effort in the trace): the faults, the paths
        asked about and the paths left out. A record that cannot be written raises: the step stops as on any disk fault and
        a resume repeats it, so no part goes missing without a word."""
        left = list(dropped if answered else asked)
        entry = _reading.record_entry(_reading.Reading(kept=None, faults=list(faults)), step="design_self_critique",
                                      asked=asked, dropped=left)
        entry["refused_keys"] = list(dict.fromkeys(str(n.get("key")) for n in refused))
        entry["answered"] = answered
        _reading.write_record(self.quest_root / "needs" / "audit_readings.json", entry)
        self._audit("answer_read", topic="design_self_critique", faults=entry["faults"], asked=entry["asked"],
                    dropped=entry["dropped"])

    @staticmethod
    def _audit_reasons(not_applied: list[dict[str, Any]]) -> dict[str, str]:
        """``{path: why}`` for the open items of an audit's refused changes: the reason of each change left out whole, and of
        each part of a block that was applied except that part (by the part's own path, so the block's other parts are not
        given it)."""
        out = {n["key"]: n["reason"] for n in not_applied if n.get("key") and not n.get("applied")}
        for n in not_applied:
            out.update({k: v for k, v in (n.get("parts") or {}).items() if k not in out})
        return out

    def _keep_audit_open_items(
        self, objections: Any, design: Any, *, reasons: dict[str, str], default_reason: str, iteration: int = 0,
        before: Any = None,
    ) -> list[dict[str, Any]]:
        """Compare each change the audit proposed with ``design``, the design FI holds, and keep every one it does not hold as
        an open item with the audit's full text (core/audit_open_items.py): the code-writing step and the code reader are
        shown it, QUALITY.md lists it and the paper's limitations say it. A proposal the design holds is remembered as
        applied, to be compared with the frozen protocol. FI's own records decide; no word of the audit is matched."""
        _audit_open.close_held(self.quest_root, design)  # an item the design holds now is closed, repeated or not
        props = self._search_proposals_by_part(_audit_open.proposals(objections), before)
        if not props:
            return []
        held, missing = _audit_open.split(props, design)
        items = _audit_open.record(self.quest_root, held, missing, reasons=reasons, default_reason=default_reason,
                                   iteration=iteration)
        for p in missing:
            self._log.warning("[design_self_critique] the audit's change to `%s` is not in the design; it is kept as an open "
                              "item for the code and the paper (%s)", p["path"],
                              _audit_open.reason_for(p["path"], reasons) or default_reason)
        return items

    def _audit_changes_held_back(self) -> list[str]:
        """One plain sentence per change the last methodology audit proposed that FI did not (fully) apply, from FI's own
        record of what it applied, for the plan's prose: the audit's text about its change may describe more than the design
        block below holds, and the block is what runs."""
        try:
            last = json.loads((self.quest_root / "needs" / "DESIGN_CRITIQUE.json").read_text(encoding="utf-8"))[-1]
        except (OSError, ValueError, IndexError, TypeError):
            return []
        out = []
        for n in last.get("not_applied") or []:
            if not isinstance(n, dict):
                continue
            out.append((f"FI applied the methodology audit's change to `{n.get('key')}` except one part: {n.get('reason')}. "
                        "Where the text of this plan describes that part as changed, the design block below is what runs.")
                       if n.get("applied") else
                       (f"FI did not apply the methodology audit's change to `{n.get('key')}`: {n.get('reason')}. "
                        "Where the text of this plan describes that change as made, the design block below is what runs."))
        return out

    def _critique_summary(self) -> list[str]:
        """What the last methodology audit did, for the plan's *Checks already made*: which parts of the design it changed,
        or that it did not run."""
        try:
            path = self.quest_root / "needs" / "DESIGN_CRITIQUE.json"
            last = json.loads(path.read_text(encoding="utf-8"))[-1]
        except (OSError, ValueError, IndexError, TypeError):
            return []
        if last.get("status") == "failed":
            return [f"The methodology audit did not complete: {last.get('failure')} (see needs/DESIGN_CRITIQUE.json)."]
        review = last.get("review") if isinstance(last.get("review"), dict) else {}
        who = {"independent_review": f"The design was read by a different model ({review.get('reviewer')}), not the one "
                                     f"that wrote it ({review.get('writer')}).",
               "self_review": "The design was read again by the model that wrote it: a self-review, not an independent "
                              "one."}.get(str(review.get("kind")), "")
        held_back = [(f"The audit's change to `{n.get('key')}` was applied except a part FI could not read: {n.get('reason')}"
                      if n.get("applied") else
                      f"The audit proposed changing `{n.get('key')}`, but FI did not apply it: {n.get('reason')}")
                     for n in last.get("not_applied") or [] if isinstance(n, dict)]
        if held_back:
            who = " ".join([*held_back, who]).strip()
        if last.get("changed_keys"):
            return [
                "The methodology audit amended the design: " + ", ".join(last["changed_keys"])
                + " (before and after are in needs/DESIGN_CRITIQUE.json)." + (f" {who}" if who else "")
            ]
        return [who] if who else []

    def _record_design_critique(
        self, iteration: int, before: dict[str, Any], after: dict[str, Any], objections: Any, failure: str,
        *, node: str = "design_self_critique", review: dict[str, Any] | None = None,
        not_applied: list[dict[str, str]] | None = None,
    ) -> None:
        """Keep what the methodology audit said and did in ``needs/DESIGN_CRITIQUE.json``, one entry per pass: the
        objections in full, the design before and after and the keys that changed, and whether the audit ran at all. Until
        this existed only the number of objections reached the run log, so nobody could tell afterwards what the audit had
        found or changed, and an audit that failed (it is advisory, and fails open) looked the same as one that found
        nothing."""
        try:
            after_plain = json.loads(json.dumps(after, default=str))
            changed = sorted(k for k in set(before) | set(after_plain) if before.get(k) != after_plain.get(k))
            entry = {
                "iteration": iteration,
                "status": "failed" if failure else "ok",
                "failure": failure,
                "objections": objections if isinstance(objections, list) else [],
                **({"review": review} if review else {}),
                **({"not_applied": not_applied} if not_applied else {}),
                "changed_keys": changed,
                "before": before,
                "after": after_plain,
            }
            path = self.quest_root / "needs" / "DESIGN_CRITIQUE.json"
            history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
            history.append(entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self._audit_check(
                "design_self_critique", path, status=entry["status"],
                summary="changed " + (", ".join(changed) or "nothing") if not failure else failure,
            )
            prov = self._chat_provenance(node)
            for o in entry["objections"]:
                o = o if isinstance(o, dict) else {"objection": o}
                self._audit(
                    "model_claim", provenance=_audit_log.MODEL_CLAIM, topic=f"design_self_critique/{o.get('check', '')}".rstrip("/"),
                    claim=o.get("objection", ""), fix=o.get("fix", ""), **prov,
                )
        except (OSError, ValueError, TypeError) as e:
            self._log.debug("[design_self_critique] could not record the audit: %r", e)

    def _design_from_plan(self) -> tuple[dict[str, Any] | None, str]:
        """The design block of ``plan.md`` and its hash, or ``(None, "")`` when there is no plan or it cannot be read
        (the plan step stops the quest on an unreadable one; this only guards a resume that skipped it)."""
        path = _plan.plan_path(self.quest_root)
        if not path.is_file():
            return None, ""
        if _frozen.load(self.quest_root) is None:
            self._move_misplaced_parts()  # a part written beside `protocol` is put under it before the plan is read
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            self._log.warning("[design] plan.md could not be read (%r); designing from the topic", e)
            return None, ""
        parsed = _plan.parse(text)
        if parsed.design is None:
            self._log.warning("[design] plan.md cannot be used (%s); designing from the topic", parsed.error)
            return None, ""
        sha = _plan.sha256(text)
        self._log.info("[design] using the design block of plan.md (sha256 %s)", sha[:12])
        return parsed.design, sha

    def _pause_for_plan(self, *, error: str = "", no_criteria: bool = False) -> None:
        """Stop so the person can read and edit ``plan.md`` (``pauses.plan: ask``). Once per quest (a marker on disk, as
        for the other supply pauses), unless the file cannot be read: then every resume stops again, with the reason.

        Nothing about a check's expected value, its source or its tolerance stops the quest here: FI works those out
        itself (``_fill_plan_sources``, ``_hold_added_oracles``, the triage at the known-answer checks) and marks what
        it cannot confirm. ``no_criteria``: the plan has no way to judge whether the code got better; one plain line
        says so (never a stop of its own)."""
        marker = self.fi_dir / "paused_at_plan.flag"
        # A plan with no way to judge whether the code got better is never a stop of its own: one line says what that
        # means, in the console, and at the stop the person asked for (``pauses.plan: ask``).
        if no_criteria:
            self._say_no_criteria()
        if not error and marker.is_file():
            return
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text("plan", encoding="utf-8")
        except OSError as e:
            self._log.warning("[plan] couldn't write pause marker %s: %r", marker, e)
        path = _plan.plan_path(self.quest_root)
        # The person reads the plan at this stop, including any source FI wrote into a check after they last read it.
        try:
            (self.fi_dir / _FILLED_AFTER_READ).unlink(missing_ok=True)
        except OSError:
            pass
        criteria_steps = [_NO_CRITERIA_LINE] if no_criteria else []
        steps = [
            *([f"`plan.md` could not be used: {error}. Fix it, then resume."] if error else []),
            *criteria_steps,
            f"Read `plan.md` in the quest folder ({path}): what the literature says, the gap, and the design the "
            "experiment will run.",
            f"Edit it. The block under \u201c{_plan.DESIGN_HEADING}\u201d is the design, exactly: what it says is what runs. "
            "The prose above it is for you and for the paper's methods.",
            "Or ask for a change and let FI rewrite it: "
            f"`python launch.py --config <quest.yaml> --resume {self.quest_id} --revise-plan \"what to change\"` "
            "(the quest page's Plan box on the web, `@fi /plan` in VSCode).",
        ]
        self._pause_for_human(
            kind="plan",
            interaction="supply",
            headline=("plan.md cannot be read" if error else "read and edit the plan"),
            steps=steps,
            payload={"plan_stage": True, "quest_id": self.quest_id, "plan_file": str(path), "error": error,
                     **({"no_criteria": True} if no_criteria else {})},
        )

    def _say_no_criteria(self) -> None:
        """Once per quest, in the console and run.log: the plan has no way to judge whether the code got better, so the
        step that improves the simulation does not start. Nothing to do; never a stop."""
        if self.__dict__.get("_said_no_criteria"):
            return
        self._said_no_criteria = True
        self._log.info("[plan] %s", _NO_CRITERIA_LINE)
        print(f"[FI] {_NO_CRITERIA_LINE}")
