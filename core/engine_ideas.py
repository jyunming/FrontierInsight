"""Methods of :class:`core.engine.Engine` for the first node (the study's run modes), the design prompt and the literature.

``Engine`` inherits :class:`IdeasStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, Counter, Path, QuestState, WORK_SCOPE_PAPERS, WORK_SCOPE_PAPERS_AND_BOOKS, _FILLED_AFTER_READ,
    _LITERATURE_QUERIES, _NO_CRITERIA_LINE, _NO_SIM_DESIGN_DIRECTIVE, _SURVEY_DESIGN_DIRECTIVE,
    _audit_log, _audit_open, _design_parts, _fault_from_record,
    _fault_record, _format_clarify, _format_lit_excerpt, _format_lit_from_state, _frozen,
    _ingest_user_dropped_papers, _item_content,
    _labelled_sources, _lit_query, _literature_entry, 
    _normalize_title, _ok_calls, _one_line, _optim, _pairs,
    _parse_json_lenient, _phased, _plan, 
    _reading, _receipts, _retractions, _write_query_sets,
    _source_text,
    _topic_req, asyncio, functools, hashlib, json,
)
from . import commission as _commission


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
        """The study's two run modes, from the commission's ``method`` (``Config`` sets ``engine.no_simulation`` and
        ``engine.survey_mode`` from it, ``core/config.py``). No question to the person and no model call.

        ``survey_mode_resolved`` implies ``no_simulation_resolved`` (a survey has no experiment). Passes through when a
        resume finds it done (``clarify_done``)."""
        if state.get("clarify_done"):
            return {}
        engine = self.config.engine
        survey = bool(engine.survey_mode)
        no_sim = bool(engine.no_simulation) or survey
        self._log.info("[clarify] study mode: %s", "review of the literature (no experiment, no data)" if survey
                       else "analysis of the data you give (no experiment)" if no_sim else "experiment")
        return {"clarify_done": True, "no_simulation_resolved": no_sim, "survey_mode_resolved": survey}

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
        """The literature step: a study starts from a commission, so its sources are the ones the commission gives."""
        return await self._commission_literature(state)

    async def _commission_literature(self, state: QuestState) -> QuestState:
        """A study's literature: only the full texts it supplies (inputs/papers/), nothing searched for."""
        prior = list(state.get("literature") or [])
        seen = {str((e.get("content") if isinstance(e, dict) else "") or "")[:200] for e in prior}
        merged, added = await asyncio.to_thread(_ingest_user_dropped_papers, self.quest_root, prior, seen, self._log)
        if self._commissioned():  # type: ignore[attr-defined]  # an engine built without one (a test, a program) reads only the dropped papers
            merged = _commission.given_sources(merged, self._commission_data())  # type: ignore[attr-defined]
        # Each DOI looked up in Crossref for a retraction (core/retractions.py); a retracted source is marked, never used as
        # support. No answer is "could not be checked", and nothing here stops the quest.
        if self.config.knowledge.enabled:  # type: ignore[attr-defined]
            merged, retraction_rows = await _retractions.check_literature(merged)
            if retraction_rows:
                self._record_retractions(retraction_rows, int(state.get("literature_iter") or 0) + 1)
                self._log.info("[literature] %s", _retractions.summary_line(retraction_rows))  # type: ignore[attr-defined]
        await self._flag_sources(merged, "literature", record_clean=True)  # type: ignore[attr-defined]
        self._log.info("[literature] the commission gives the sources: %d full text(s) read from inputs/papers/, "
                       "nothing searched for", added)
        return {"literature": merged, "literature_iter": int(state.get("literature_iter") or 0) + 1}

    def _record_retractions(self, rows: list[dict[str, Any]], iteration: int) -> None:
        """Keep what Crossref says of each source's retraction in ``.fi/literature_queries.json`` (the record the to-do card
        and the seal read). Entries are only added. Best-effort: a record never stops a quest."""
        fi_dir = self.fi_dir  # type: ignore[attr-defined]
        path = fi_dir / _LITERATURE_QUERIES
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            entries = [e for e in (data.get("entries") if isinstance(data, dict) else None) or [] if isinstance(e, dict)]
            entries.append({"stage": "retractions", "iteration": iteration, "outcome": "checked", "sources": rows})
            _write_query_sets(fi_dir, entries)
        except (OSError, ValueError) as e:
            self._log.debug("[literature] could not record the retraction lookups: %r", e)  # type: ignore[attr-defined]

    def _study_given(self, state: QuestState) -> str:
        """The study, for the prompts that name it (skills, design, the design's review): on a commissioned quest the
        commission's question and scope, which fix the study; else the direction a checkpoint of an earlier FI chose
        (FI makes no research ideas now), or ``(none)``."""
        if self._commissioned():  # type: ignore[attr-defined]
            data = self._commission_data()  # type: ignore[attr-defined]
            return f"Question: {data.get('question')}\nScope: {data.get('scope')}"
        chosen = state.get("chosen_idea")
        return json.dumps(chosen, indent=2) if isinstance(chosen, dict) and chosen else "(none)"

    def _commission_block(self, state: QuestState) -> str:
        """What the commission fixes, for the plan's model, with each known value's source by its label ([n])."""
        data = self._commission_data()  # type: ignore[attr-defined]
        by_name = {name: raw for raw, name in _commission.staged_names(data).items()}
        labels = {by_name[Path(str(meta.get("path") or "")).name]: label
                  for label, meta, _item in _labelled_sources(state.get("literature") or [])
                  if meta.get("source") == "user_supplied" and Path(str(meta.get("path") or "")).name in by_name}
        return _commission.prompt_block(data, labels)

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
        """The design prompt for this state: the topic, the study, the literature, the
        clarifications, the skills and example files, and, on a later pass, what the review and the person said."""
        review_feedback = self._design_feedback(state)
        prompt = self._prompts["design"].substitute(
            topic=state["topic"],
            study=self._study_given(state),
            # The full excerpts, as analyze and write get. At 800 characters a
            # source, design chose a 1% major-outbreak threshold in 6 of 10
            # replays of a real quest (0 of 5 at the full budget): short
            # excerpts cost it the context a sound design rests on.
            literature_block=_format_lit_from_state(state, **self._lit_kwargs(state)),
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
        if self._commissioned():  # type: ignore[attr-defined]
            prompt = f"{prompt}\n\n{self._commission_block(state)}\n"
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
            study=self._study_given(state),
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
