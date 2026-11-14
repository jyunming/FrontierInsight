"""Methods of :class:`core.engine.Engine` for the plan and the design.

``Engine`` inherits :class:`PlanStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, BridgeError, FallbackLLMClient, LLMClient, ModelAnswerTruncated, PROXY_PROVIDERS, Path, ProviderUnusable,
    QuestArtifacts, QuestState, _CLI_PROVIDERS_SET, _DRAFTED_GRID_ASKED, _DRAFTED_GRID_ASKED_AGAIN,
    _DRAFTED_PARTS_ASKED, _DRAFTED_SEARCH_ASKED, _FIGURES_PER_CALL, _FIGURE_PICK_BATCH, _FIGURE_READING_CHARS,
    _FILLED_AFTER_READ, _FILL_MARKER, _GRID_ASKED, _GRID_ASKED_AGAIN, _LEDGER_ASKED, _LEDGER_PARTS,
    _ModelAnswerProblem, _ONCE_PARTS, _PART_ASKED, _PART_HEADING, _PLAN_DIRECTIVE, _PartsRefused,
    _PlanEditedMeanwhile, _REDESIGN_OWNED_PARTS, _REFERENCE_FORMS, _REFUSAL_CHARS, _REFUSED_BLOCK_CHARS,
    _SCHEMA_RETRY_ASKS, _SEARCH_ASKED, _SEARCH_ASKS, _SEARCH_CALLS, _SEARCH_CALLS_PER_ASK, _SEARCH_REFUSED,
    _STUDY_TYPE_DIRECTIVE, _TOPIC_ASKED, _TOPIC_DESIGN_ASKED, _accepted, _append_design_revision,
    _append_figure_readings, _attempts, _audit_log, _audit_open, _code_delta, _code_layout, _contract,
    _contract_parts, _criteria, _deps_from_tail, _disclosure, _entries_as_written, _entry_identities,
    _fault_from_record, _fault_record, _fill_changed_more, _fill_scope, _finding_repair, _flat_fact_sentence,
    _format_clarify, _forms, _frozen, _function_steps, _in_run, _ingest_user_dropped_papers, _item_content,
    _labelled_sources, _ledger, _lit_query, _mc_error, _one_line, _optim, _optimise, _optimise_search, _oracle,
    _package_block_text, _pairs, _parse_json_lenient, _pick_up_user_dropped_datasets, _plan, _plan_checks_note,
    _plan_scope, _plan_search_mark, _plan_settings, _plan_steps, _protocol, _read_a_search_block,
    _read_json_or_none_path, _reading, _receipts, _review, _scope_cards, _script_reply, _skeleton, _source_text,
    _split_run, _study_size, _todo, _top_level_json_objects, _topic_req, _trial_runner, asyncio,
    claude_cli_account_line, copy, hashlib, inspect, interrupt, json, missing_step_api_keys, model_for_node,
    node_provider_for, os, re, resolve_endpoint_async, step_provider_config, step_provider_lines,
)


class PlanStageMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    async def _node_plan(self, state: QuestState) -> QuestState:
        """Review the literature and write the plan (``plan.md``) before the experiment is designed.

        The design prompt gets the plan directive appended, so one call returns the design and the prose around it;
        the methodology audit runs on the design; ``plan.md`` is written; and, with ``pauses.plan: ask``, the quest
        stops for the person. A ``plan.md`` that is already there (a resume, or one the person wrote) is not written
        over: it is read, and an unreadable one stops the quest again with the reason. Later passes of the design
        (a revise after results) do not come back here."""
        if self.config.engine.analyze_local_first:
            return {}
        if int(state.get("iteration", 0) or 0) > 0:
            return {}
        path = _plan.plan_path(self.quest_root)
        ask = self.config.pauses.plan == "ask"
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if _plan.note_edit(self.quest_root, text) is not None:
                self._log.info("[plan] plan.md was edited on disk; keeping that version")
            parsed = _plan.parse(text)
            if parsed.design is None:
                self._pause_for_plan(error=parsed.error or "no design")
                return {}
            if not self._oracle_guidance_path().is_file():
                # A plan written before its checks were looked at (a person's own, or a quest begun before this):
                # looked at once, as a plan FI writes is. When the person already read it at the plan stop, what this
                # changes is theirs to read again before the freeze.
                read_already = (self.fi_dir / "paused_at_plan.flag").is_file()
                before = self._planned_oracles()
                await self._guide_oracles(state)
                after = self._planned_oracles()
                changed = [n for n, o in after.items() if before.get(n) != o]
                removed = [n for n in before if n not in after]
                if read_already and (changed or removed):
                    self._note_engine_change(changed, removed, reason=(
                        "After you read the plan, FI looked at its checks once (see “A second look at the checks "
                        "against known answers” in plan.md), and they changed: "
                        + "; ".join(_forms.describe_changes(before, after)) + "."))
                parsed = _plan.parse(path.read_text(encoding="utf-8"))
                if parsed.design is None:
                    self._pause_for_plan(error=parsed.error or "no design")
                    return {}
            protocol = parsed.design.get("protocol")
            no_criteria = (ask and self._runs_code(state) and isinstance(protocol, dict)
                           and self._split_on({**state, "design": parsed.design}) and not _criteria.countable(protocol))
            await self._settle_plan_sources(state, protocol=protocol if isinstance(protocol, dict) else None,
                                            no_criteria=no_criteria)
            await self._complete_the_search_in_the_plan(state)
            if ask:
                self._pause_for_plan(no_criteria=no_criteria)
            return {}

        # The rules for a search for the best design only when the topic or the person's answer points that way: a
        # topic with no word of looking for the best is planned as a measurement, as before, without their cost.
        await self._read_topic_before_the_prompt(state)
        seek = self._seeks_best_design(state)
        card = state.get("scope_card")
        prompt = (self._design_prompt(state) + _PLAN_DIRECTIVE + self._study_type_part(state, seek=seek)
                  + (f"\n\n{_scope_cards.plan_note(card)}\n" if isinstance(card, dict) else ""))
        obj = await self._design_reply(prompt, node="plan")
        extra = obj.pop("plan", None)
        design = obj
        design, objections = await self._audit_design(state, design, writer="plan")
        if isinstance(design, dict):
            # ``plan`` asks the same design prompt as ``design`` (``_design_prompt`` + ``_PLAN_DIRECTIVE``) and can get
            # the same ``rationale`` block back. Unlike ``design``, this node writes straight to ``plan.md`` and never
            # returns the design in its state patch — ``_audit_claims`` (which strips ``design``'s rationale) never
            # sees it here, so without this the raw block would reach the persisted file (and a person reading it
            # under ``pauses.plan: ask``) instead of the audit trace.
            design = self._pop_rationale_into_claims("plan", design)
        # A draft's study_type that cannot be read, or that contradicts its own block, is put right with a note.
        moved_parts: list[str] = []
        if isinstance(design, dict):
            for fault in _plan.read_hoist(design).faults:  # a part written beside `protocol` AND in it, the two different
                self._log.warning("[plan] %s (both versions: %s)", fault.reason, _reading.shown(fault.as_written, 600))
            design, moved_parts = _plan.hoist_protocol_parts(design)  # a part written beside `protocol` belongs under it
            if moved_parts:
                self._log.warning("[plan] the draft wrote %s beside `protocol`; FI moved %s under it",
                                  ", ".join(f"`{m}`" for m in moved_parts), "it" if len(moved_parts) == 1 else "them")
        study_repairs = _optim.repair_study_type(design)
        normalized, why = _plan.normalize_design(design)
        repaired_notes: list[str] = list(study_repairs)
        if normalized is None and "`protocol" in (why or "") and isinstance(design, dict):
            # The protocol is the part a model writes freest. Keep the design, drop the keys that cannot be checked, and
            # say so in the plan, rather than losing the plan for one of them.
            held_notes: list[str] = []
            drafted = design.get("protocol")
            if isinstance(drafted, dict) and isinstance(drafted.get("optimisation"), dict):
                held_notes = self._hold_search_block_back(drafted["optimisation"])
                if held_notes:  # a limit, objective or baseline that cannot be read: no search without it
                    design = {**design, "study_type": "find_best_design",
                              "protocol": {k: v for k, v in drafted.items() if k != "optimisation"}}
            reading, repaired_notes = _plan.read_protocol_with_notes(design.get("protocol"))
            repaired_notes = [*repaired_notes, *held_notes]
            fixed_protocol = reading.kept
            self._keep_plan_reading(reading, "plan")
            trial = {k: v for k, v in design.items() if k != "protocol"}
            if fixed_protocol:
                trial["protocol"] = fixed_protocol
            normalized, why = _plan.normalize_design(trial)
            for note in repaired_notes:
                self._log.warning("[plan] %s", note)
                if "`protocol.grid`" in note:
                    print(f"[FI] {note}")
        if normalized is None:
            self._log.warning("[plan] the drafted design is not usable (%s); the design step will draft it again", why)
            return {}
        study_notes = self._settle_study_type(state, design, normalized)
        # How the code will be judged (core/criteria.py): a draft that names no criterion is judged by its own checks
        # against known answers that FI runs; only a draft with none of those gets one search of the literature for
        # how such studies are checked, and one more question to the model.
        no_criteria = False
        from_checks: list[str] = []  # the names of the criteria FI made from the plan's own checks
        fi_runs = self._split_on({**state, "design": normalized})
        if self._runs_code(state) and isinstance(normalized.get("protocol"), dict) and not fi_runs:
            self._log.warning("[criteria] this quest runs its experiment as one script, so FI runs none of its checks "
                              "itself: the plan's criteria are shown after each run but none counts")
        elif self._runs_code(state) and isinstance(normalized.get("protocol"), dict) and not _criteria.countable(
                normalized["protocol"]):
            drafted = _criteria.declared(normalized["protocol"])
            proposed = _criteria.from_checks(normalized["protocol"])
            from_checks = [c["name"] for c in proposed]
            if proposed:
                criteria_notes = [
                    "The plan named no criterion FI can measure itself for judging whether the code got better, so FI "
                    "uses the plan's own checks against known answers that it runs: "
                    + ", ".join(f"“{c['name']}”" for c in proposed)
                    + " (how far each lands from the value it expects; met within the check's own tolerance)."]
                self._log.info("[criteria] the plan names no criterion FI can measure itself; FI judges the code by the "
                               "plan's own checks it runs (%s), without asking the model",
                               ", ".join(c["name"] for c in proposed))
            else:
                proposed, criteria_notes = await self._propose_criteria(state, normalized["protocol"])
            repaired_notes += criteria_notes
            if proposed:
                # What the model proposed comes first, then what the draft had (none FI could measure itself), up to five.
                merged = proposed + [c for c in drafted if c["name"].lower() not in {p["name"].lower() for p in proposed}]
                for c in merged[_criteria.MAX:]:
                    repaired_notes.append(f"the criterion {c['name']!r} was left out of the plan: at most "
                                          f"{_criteria.MAX} are kept, those FI can measure itself first")
                renormalized, why_not = _plan.normalize_protocol(
                    {**normalized["protocol"], "criteria": merged[:_criteria.MAX]})
                if renormalized is not None:
                    normalized = {**normalized, "protocol": renormalized}
                else:
                    note = f"the plan with the criteria the model proposed could not be read ({why_not}); they were left out"
                    repaired_notes.append(note)
                    self._log.warning("[criteria] %s", note)
            no_criteria = not _criteria.countable(normalized["protocol"])
            if no_criteria:
                self._log.warning(
                    "[criteria] the plan has no check of correctness FI can compute to judge whether the code got better "
                    "(one search of the literature and a second question found none)%s",
                    "; the quest stops at the plan for you to write one" if ask else
                    "; going on without one: every run is recorded as having no criterion",
                )
                if not ask:
                    print("[FI] The plan has no way to judge whether the code got better (no check of correctness FI can "
                          "compute); going on without one. See plan.md and .fi/run.log")
        # The plan writes the audited design in its own canonical form: the audit's verdict covers it.
        _receipts.carry_output(self.quest_root, "design_audit", _receipts.design_core(design),
                               _receipts.design_core(normalized), "the plan wrote the audited design in its own form")
        # Only FI writes `form`: a model's own word is dropped. The questions come after the receipt is carried, so what
        # the methodology audit did not see (the answers) is not covered by its verdict.
        normalized = _contract_parts.strip_fi_keys(normalized)
        if self._runs_code(state) and self._client is not None and isinstance(normalized.get("protocol"), dict):
            normalized, contract_notes = await self._ask_contract_parts(normalized)
            repaired_notes += contract_notes
        audit = [str(item) if not isinstance(item, dict) else "; ".join(str(v) for v in item.values() if v)
                 for item in (objections if isinstance(objections, list) else [])]
        audit += repaired_notes
        audit += [f"{', '.join('`' + m + '`' for m in moved_parts)} was written beside `protocol` in the draft and FI moved "
                  "it under `protocol`, where the plan is read" for _ in [0] if moved_parts]
        audit += study_notes
        audit += self._critique_summary()
        # What the topic sets and the protocol leaves out: said in the plan, where a person reads it before compute is spent.
        audit += _protocol.plan_notes(state.get("topic") or self.config.topic, normalized.get("protocol"))
        if isinstance(normalized.get("protocol"), dict):
            audit += _protocol.oracle_notes(normalized.get("protocol"))
            audit += _protocol.failure_notes(normalized.get("protocol"))
            audit += _protocol.metric_notes(normalized.get("protocol"))
            fi_runs = self._split_on({**state, "design": normalized})
            once = _protocol.run_count_notes(normalized, fi_runs=fi_runs)
            # A design that names nothing random is not asked for a precision its repeats would buy.
            if not once:
                # FI's own trials (two scripts) are every trial of a setting, each with its own seed, and run once: the trials
                # of a setting are its runs, as the run step counts them (replicates_n there), not runs times the seeds.
                seeds = 1 if fi_runs or self.config.execution.background_jobs else int(self.config.engine.execute_replicates)
                audit += _protocol.precision_notes(normalized.get("protocol"), seeds)
            audit += _protocol.grid_notes(normalized)
            audit += once
            audit += _criteria.plan_notes(normalized.get("protocol"), fi_runs=fi_runs)
            # Where the model's equations and each check's expected value come from (core/oracle_check.py).
            source_gaps, model_notes = self._plan_source_findings(state, normalized.get("protocol"))
            audit += model_notes
            audit += [f"{g[0].upper()}{g[1:]}: {_oracle.SOURCE_FORMS}." for g in source_gaps]
        body = _plan.render(state.get("topic") or self.config.topic, extra if isinstance(extra, dict) else {},
                            normalized, audit,
                            sources=self._retrieved_sources(state) if self._runs_code(state) else None,
                            code_layout=self._plan_code_layout({**state, "design": normalized}),
                            confirm=self._plan_confirm_lines(state), held_back=self._audit_changes_held_back())
        study = _scope_cards.plan_section(_scope_cards.read_record(self.quest_root))
        if study:
            body = body.rstrip("\n") + "\n\n" + study
        path.write_text(body, encoding="utf-8")
        _plan.record_version(self.quest_root, body, by="model", note="written from the topic and the literature")
        self._log.info("[plan] wrote %s (%d sources named, %d checks)", path,
                       len((extra or {}).get("literature") or []) if isinstance(extra, dict) else 0, len(audit))
        await self._shadow("plan", {**state, "design": normalized},
                           taken="held the plan for the person" if ask else "went on with the plan")
        # The checks against known answers, each in its kind's numeric form, before the person reads the plan.
        await self._guide_oracles(state)
        if from_checks:
            self._criteria_follow_checks(from_checks)
        guided, _why = _plan.load_design(self.quest_root)
        if isinstance(guided, dict):
            normalized = guided
            if from_checks and isinstance(guided.get("protocol"), dict):
                no_criteria = not _criteria.countable(guided["protocol"])  # as the checks were left
        protocol = normalized.get("protocol")
        await self._settle_plan_sources(state, protocol=protocol if isinstance(protocol, dict) else None,
                                        no_criteria=no_criteria and ask)
        await self._complete_the_search_in_the_plan(state)
        if ask:
            self._pause_for_plan(no_criteria=no_criteria)
        return {"design_objections": objections} if isinstance(objections, list) else {}

    async def _ask_contract_parts(self, design: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """``(design, notes)``: ``design`` with the contract parts the plan's model gave in a form that could be used
        (:func:`core.contract_parts.ask_parts`), and a sentence for the parts it did not. What was asked and answered is kept
        in ``.fi/contract_parts.json`` (and shown in QUALITY.md). Never a stop. A part whose call raised is left out alone
        and said three ways: run.log with the traceback, one console line, and a note in plan.md."""
        async def ask(prompt: str) -> str:
            return await self._chat(prompt, node="plan_contract")

        try:
            out = await _contract_parts.ask_parts(design, ask, lambda t: _parse_json_lenient(t, node="plan_contract"), self._prompts,
                                                  fi_analysis=self._fis_analysis_planned(design))
        except Exception:  # noqa: BLE001 -- the plan goes on without the parts
            self._log.warning("[plan] the contract parts could not be asked; the plan stays in the older form", exc_info=True)
            print("[FI] FI could not ask the plan about the study's results; the study goes on with the plan as written")
            return design, ["The plan was not asked what kind of number each result is (the call failed); the plan stays in "
                            "the older form."]
        for part, trace in out.errors.items():
            what = _contract_parts.PART_WORDS[part]
            self._log.warning("[plan] FI could not ask the plan about %s:\n%s", what, trace)
            print(f"[FI] FI could not ask the plan about {what}; the study goes on with the plan as written")
            out.notes.append(f"FI could not ask the plan about {what} (the call failed); the study goes on with the plan as "
                             "written.")
        try:
            (self.fi_dir / "contract_parts.json").write_text(json.dumps(
                {"form": out.form, "parts": out.parts, "names_not_asked": out.design.get("protocol", {}).get("names_not_asked", [])},
                indent=1, default=str) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[plan] couldn't write .fi/contract_parts.json: %r", e)
        self._log.info("[plan] contract parts: %s", "; ".join(
            f"{k} {'ok' if v['valid'] else 'left out'} after {v['asks']} ask(s)" for k, v in out.parts.items()) or "none asked")
        return out.design, out.notes

    def _criteria_follow_checks(self, made: list[str]) -> None:
        """The criteria FI made from the plan's own checks (``made``, their names), made again from the checks as FI's look
        at them left them (a check renamed, its tolerance changed or its case taken away), before the person reads the
        plan and before anything is frozen. Other criteria are kept as they are. Best-effort."""
        try:
            path = _plan.plan_path(self.quest_root)
            text = path.read_text(encoding="utf-8")
            design = _plan.parse(text).design
            protocol = design.get("protocol") if isinstance(design, dict) else None
            if not isinstance(protocol, dict):
                return
            now = _criteria.declared(protocol)
            again = _criteria.from_checks(protocol)
            others = [c for c in now if c["name"] not in made and c["name"] not in {a["name"] for a in again}]
            wanted = (again + others)[:_criteria.MAX]
            if wanted == now:
                return

            def change(block: dict[str, Any]) -> dict[str, Any]:
                proto = dict(block.get("protocol") or {})
                if wanted:
                    proto["criteria"] = wanted
                else:
                    proto.pop("criteria", None)
                return {**block, "protocol": proto}

            edited = _plan.edit_design_block(text, change)
            if edited is None or _plan.parse(edited).design is None:
                return
            edited = _plan.refresh_model_section(edited)
            path.write_text(edited, encoding="utf-8")
            _plan.record_version(self.quest_root, edited, by="engine",
                                 note="the criteria made from the plan's own checks follow the checks as FI left them")
            self._log.info("[criteria] the criteria made from the plan's own checks were made again from the checks as "
                           "FI left them: %s", ", ".join(c["name"] for c in again) or "none")
        except Exception as e:  # noqa: BLE001 -- the criteria made first stay; a check they name that is gone is said after a run
            self._log.warning("[criteria] the criteria could not be made again from the checks (%r)", e)

    async def _propose_criteria(self, state: QuestState, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        """The plan named no criterion: search the literature once for how studies like this one are checked, then ask
        the model again with what the quest already found and what the search turned up. ``(criteria, notes)``: the
        criteria FI can compute (possibly none) and a sentence per one left out. A failed search or call is the same as
        finding none; only a model answer the engine stops for anyway (``_ModelAnswerProblem``) stops the quest."""
        topic = " ".join(str(state.get("topic") or self.config.topic or "").split())
        query = f"how the correctness of a simulation or numerical method is verified: {topic[:200]}"
        known = self._retrieved_sources(state)
        seen = {str(s.get("doi") or "").strip().lower() or " ".join(str(s.get("title") or "").split()).lower()
                for s in known}
        hits: list[Any] = []
        try:
            # One bounded search, without asking the model to route it (the fallback sources): a routing call is a third
            # of a quest's calls, and this one only needs a few titles.
            hits = await self.knowledge.asearch(query, top_k=5, external_top_k=5, work_scope=self._work_scope(state),
                                                fetch_full_text=False)
        except Exception as e:  # noqa: BLE001 -- a search that fails leaves the model with what the quest already found
            self._log.warning("[criteria] the search for how such studies are checked failed: %s", e)
        new = []
        for doc in hits or []:
            meta = getattr(doc, "metadata", {}) or {}
            key = str(meta.get("doi") or "").strip().lower() or " ".join(str(meta.get("title") or "").split()).lower()
            if key and key not in seen:
                seen.add(key)
                new.append(doc)
        self._log.info("[criteria] the plan names no check of correctness FI can measure itself; searched once (%r: %d "
                       "new source(s)) and asking the model again", query, len(new))
        lines = [f"- [{s['label']}] {s.get('title') or '(untitled)'}" for s in known[:25]] or ["- (none)"]
        lines += ["", f"One more search ({query!r}) found:"]
        await self._flag_sources(new, "criteria")
        for i, doc in enumerate(new, start=1):
            meta = getattr(doc, "metadata", {}) or {}
            text = " ".join(str(getattr(doc, "content", "") or "").split())[:400]
            head = _source_text.mark(f"- [C{i}] {' '.join(str(meta.get('title') or '(untitled)').split())}", meta)
            lines.append(f"{head}: {text}")
        if not new:
            lines.append("- (nothing new)")
        prompt = self._prompts["plan_criteria"].substitute(
            topic=topic[:1500], protocol=json.dumps(protocol, indent=2, default=str)[:8000],
            # Retrieved titles and text are material to read, never instructions (core/source_text.py).
            found=_source_text.fence("\n".join(lines)),
        )
        try:
            reply = await self._chat(prompt, node="plan_criteria")
        except _ModelAnswerProblem:
            raise
        except Exception as e:  # noqa: BLE001 -- no answer is the same as no criterion
            self._log.warning("[criteria] asking the model for criteria failed: %s", e)
            return [], []
        obj = _parse_json_lenient(reply) or {}
        kept, notes = _criteria.repair(obj.get("criteria") if isinstance(obj, dict) else None, protocol)
        for note in notes:
            self._log.warning("[criteria] %s", note)
        return kept, notes
    @staticmethod
    def _study_type_asked(state: QuestState) -> str | None:
        """The kind of study the person named (``measure`` / ``find_best_design``), or ``None``: the study card they
        picked before the plan comes first, then their answer at the clarify step."""
        card = state.get("scope_card")
        if isinstance(card, dict) and card.get("study_type") in _scope_cards.STUDY_TYPES:
            return str(card["study_type"])
        return _optim.resolve_answer((state.get("clarify_answers") or {}).get("study_type"))

    def _asks_for_best_design(self, state: QuestState, *, topic: bool = True) -> str:
        """Why this quest is a search for the best design whatever its design says: the person answered so (``answer``),
        or (with ``topic``, at the plan step only) FI's reading of the topic lists a design the question cannot be
        answered without that the study itself must search for (``topic``: the reader's ``searches``,
        :func:`core.topic_requirements.searched`); else ``""``. The topic is read only when a fresh plan is written, where
        the plan says so and a person can change it: never to stop a quest whose design was written before."""
        if self._study_type_asked(state) == "find_best_design":
            return "answer"
        if (topic and self._study_type_asked(state) is None
                and any(s.get("main") for s in _topic_req.searched(self._topic_requirements_read()))):
            return "topic"
        return ""

    def _seeks_best_design(self, state: QuestState) -> bool:
        """Whether the study-type rules (measurement or search for the best design) belong in a design prompt: the topic
        points at a best design, the person answered so, or FI's reading of the topic lists a design to search for. The
        one rule behind the plan prompt and a design drafted from the topic (a redesign)."""
        return bool(_optim.may_seek_best(state.get("topic") or self.config.topic)
                    or self._study_type_asked(state) == "find_best_design"
                    or _topic_req.searched(self._topic_requirements_read()))

    def _study_type_part(self, state: QuestState, *, seek: bool) -> str:
        """The study-type rules and the person's own answer, appended to a design prompt (plan and redesign alike)."""
        return (_STUDY_TYPE_DIRECTIVE if seek else "") + self._study_type_note(state)

    def _study_type_note(self, state: QuestState) -> str:
        """For the plan prompt: the person's own answer on the kind of study, which the plan follows."""
        asked = self._study_type_asked(state)
        if asked is None:
            return ""
        return (f"\n\nThe person answered that this study should {_optim.STUDY_TYPES[asked]}: write "
                f"`\"study_type\": \"{asked}\"`"
                + (" and give the protocol an `optimisation` block, not a `grid`." if asked == "find_best_design"
                   else " and give the protocol a `grid`, not an `optimisation` block.") + "\n")

    def _settle_study_type(self, state: QuestState, draft: Any, design: dict[str, Any]) -> list[str]:
        """Make the plan's ``study_type`` say what was asked for, so a search for the best design is never written down
        (and later run) as a plain sweep. ``design`` is changed in place; the sentences say what was changed, for the
        plan's *Checks already made*."""
        notes: list[str] = []
        protocol = draft.get("protocol") if isinstance(draft, dict) else None
        if (design.get("study_type") is None and isinstance(protocol, dict) and protocol.get("optimisation") is not None
                and not _optim.has_block(design)):
            design["study_type"] = "find_best_design"
            notes.append("The draft asked for a search for the best design, but its optimisation block could not be used "
                         "(see above); the plan keeps `study_type: find_best_design` so it is not run as a plain sweep. "
                         "Write the block again, or set `study_type: measure` and give a `grid`.")
        asked = self._study_type_asked(state)
        kind = _optim.study_type_of(design)
        why = self._asks_for_best_design(state)
        # The topic's reading counts as the person's answer does, over a draft that calls itself a measurement: the draft's
        # word for its kind is the plan model's guess, and a design the topic asks to be searched for written as a value
        # of a setting is the search hidden from FI.
        if why and kind != "find_best_design":
            design["study_type"] = "find_best_design"
            wanted = [s["name"] for s in _topic_req.searched(self._topic_requirements_read()) if s.get("main")]
            said = ("You answered that this study should find the best design" if why == "answer"
                    else f"The topic asks the study to search for {', '.join(wanted[:3])}")
            notes.append(f"{said}, but the draft is a measurement over chosen settings; the plan says "
                         "`study_type: find_best_design`. Give it an `optimisation` block, or set `study_type: measure` "
                         "if a measurement is what you want.")
        elif asked == "measure" and kind != "measure":
            notes.append("You answered that this study should measure how the result changes, but the draft searches for "
                         "the best design; replace its `optimisation` block with a `grid` and set `study_type: measure` "
                         "if a measurement is what you want.")
        return notes

    def _search_problems(self, state: QuestState, design: Any) -> tuple[bool, list[str], str | None]:
        """``(wanted, what the search lacks, why its block cannot be read)`` for ``design``: ``wanted`` is whether the
        design is a search for the best design (its own ``study_type`` or block, the person's answer, or, for a design
        that says nothing of its own, a topic that plainly asks for it); the list is the technical sentences for
        run.log and the request to the plan's model; the last is ``normalize``'s reason for a block that cannot be used."""
        explicit = isinstance(design, dict) and design.get("study_type") is not None
        # The person's answer, or a topic that plainly asks for the best design, counts for a design that says nothing
        # of its own (one drafted at this step, say); a design that says `measure` is a choice someone made and runs.
        asked = bool(self._asks_for_best_design(state, topic=False)) and not explicit and not _optim.has_block(design)
        if _optim.study_type_of(design) != "find_best_design" and not asked:
            return False, [], None
        # Read back through the check, since a design drafted at this step has not been through it.
        _block, block_why = (_optim.normalize(design["protocol"]["optimisation"]) if _optim.has_block(design)
                             else (None, None))
        # A coarse scan that cannot run (too many settings, or too long at the plan's own time for one) is a part of the
        # search the plan must fix, here, minutes after it was written; the run step still times one real evaluation.
        missing = _optim.missing_parts(design, self._scan_limit_s()) if not asked else [
            "you answered that this study should find the best design, but the design is a measurement over chosen settings"]
        held = self._search_refused()
        if held and held.get("faults") and not _optim.has_block(design):
            missing.append("its optimisation block could not be read: " + "; ".join(
                f"`{r['path']}` ({' '.join(str(r['reason']).split())[:200]})" for r in held["faults"][:4]))
        if block_why:
            entry = _entries_as_written(design.get("protocol"), _optim.block_faults(design["protocol"]["optimisation"])[1])
            missing.append(f"its optimisation block cannot be used ({block_why})"
                           + (f"; {entry}" if entry else ""))
        return True, missing, block_why

    def _keep_search_refused(self, block: Any, faults: list[Any], plan_text: str | None = None) -> None:
        """The search block a model wrote that could not be used, whole and as written, with each fault at its path
        (``.fi/search_refused.json``): the next request names only those paths, and the plan and the stop say which part
        could not be read. The record is tied to the plan's own search block it stands beside (``plan_text``'s; none for a
        block held back, which leaves the plan without one) and to the plan's model, as the records of what the plan's
        model was asked are. A record that cannot be written raises, as every caller's record does."""
        self.fi_dir.mkdir(parents=True, exist_ok=True)
        (self.fi_dir / _SEARCH_REFUSED).write_text(
            json.dumps({"block": block, "faults": [_fault_record(f) for f in faults],
                        "plan_block": _plan_search_mark(plan_text), "model": self._plan_model_name()},
                       ensure_ascii=False, default=str),
            encoding="utf-8")

    def _search_refused(self, plan_text: str | None = None) -> dict[str, Any] | None:
        """The record of :meth:`_keep_search_refused`, or ``None`` when it does not stand beside this plan: the plan's own
        search block (``plan_text``, else plan.md as it is now) is no longer the one it was kept beside (a person wrote one,
        or another request did), or another model asks now. Its block is then an old draft, never asked about or rebuilt
        into the plan. A record tied to no plan (kept before records were tied) cannot be told from such a one and counts
        as none. The whole plan's hash is not the tie: FI's own notes change plan.md between the record and its use."""
        try:
            record = json.loads((self.fi_dir / _SEARCH_REFUSED).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(record, dict) or "plan_block" not in record or record.get("model") != self._plan_model_name():
            return None
        if plan_text is None:
            try:
                plan_text = _plan.plan_path(self.quest_root).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return record  # no plan written yet: nothing has changed beside the record
        return record if _plan_search_mark(plan_text) == record["plan_block"] else None

    def _clear_search_refused(self) -> None:
        try:
            (self.fi_dir / _SEARCH_REFUSED).unlink()
        except OSError:
            pass

    def _path_only_search_request(self, refused: dict[str, Any]) -> str:
        """The request about a search block that could not be used: only the faulted paths, each with FI's reason, the entry
        as written and the values allowed; the rest of the block is kept as it was written."""
        faults = [_fault_from_record(r) for r in refused.get("faults") or []]
        return (_reading.ask_text(faults, "This study is a search for the best design. The block you wrote for it is kept "
                                          "exactly as you wrote it; only these parts of it could not be read.")
                + '\n\nAnswer with ONE JSON object of the form {"protocol": {"fixes": {"<path>": <the whole corrected part at '
                  'that path>}}}, one entry for each path above, the path written exactly as named. Do not write the block again.')

    def _hold_search_block_back(self, block: Any) -> list[str]:
        """For a search block written in a plan or a draft: ``[]`` when it can be used (every unreadable optional part left
        out alone, as :func:`_read_a_search_block` says); else one sentence naming each part that could not be read, after
        the block was kept whole beside its faults (:meth:`_keep_search_refused`). A search must not run without a limit, an
        objective or a baseline it declared, so the caller leaves the block out of the plan, the plan's model is asked for
        those parts alone, and the search is not run if they still cannot be read."""
        if not isinstance(block, dict):
            return []
        taken, _left_out, stopping = _read_a_search_block(block)
        if taken is not None:
            self._clear_search_refused()  # a block that reads replaces any held one: nothing stale is asked about
            return []
        faults = _optim.block_faults(block)[1] or stopping
        self._keep_search_refused(block, faults)
        named = "; ".join(f"`{f.path}` ({' '.join(str(f.reason).split())[:200]})" for f in faults[:4])
        entry = _entries_as_written({"optimisation": block}, faults[:4])
        sentence = (f"the search's block was not kept because a part of it could not be read: {named}"
                    + (f"; {entry}" if entry else "")
                    + "; the plan's model is asked for those parts alone, and if they still cannot be read the search is not run, "
                      "and it is not run as a plain sweep instead")
        self._log.warning("[plan] %s", sentence)
        return [sentence]

    def _scan_limit_s(self) -> float:
        """The time one run may take (``execution.timeout_s``), which the coarse scan of a search must fit."""
        return float(self.config.execution.timeout_s)

    def _plan_model_name(self) -> str:
        """The model the plan step's requests go to (for the record of what FI asked it, so another model is asked afresh)."""
        return str(self._model_for_node("plan_revise") or self.config.provider.model or "")

    def _search_record(self, sha: str, file: str | None = None) -> tuple[int, int]:
        """``(requests, chat calls)`` FI has made of the plan's model to write the search part of the plan that ``plan.md``
        (``sha``) still lacks, on the model asked now. Zero for a plan someone changed since (a person, another request) or a
        model changed since: each is a new plan, or a new asker, and is asked afresh."""
        try:
            record = json.loads((self.fi_dir / (file or _SEARCH_ASKED)).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0, 0
        if (isinstance(record, dict) and record.get("sha") == sha and record.get("model") == self._plan_model_name()):
            try:
                count = max(0, int(record.get("count") or 0))
                return count, max(0, int(record.get("calls") if record.get("calls") is not None else count * _SEARCH_CALLS_PER_ASK))
            except (TypeError, ValueError):
                return 0, 0
        return 0, 0

    def _search_problem(self, sha: str, file: str | None = None) -> str:
        """Why the plan's model's last answer was not used, as the reader that refused it said (kept in the same record;
        empty for another plan or model, and when the last answer was used)."""
        try:
            record = json.loads((self.fi_dir / (file or _SEARCH_ASKED)).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        if isinstance(record, dict) and record.get("sha") == sha and record.get("model") == self._plan_model_name():
            return str(record.get("problem") or "")
        return ""

    def _search_refused_block(self, sha: str, file: str | None = None) -> str:
        """The block the plan's model wrote that the reader refused last time, as written (cut at ``_REFUSED_BLOCK_CHARS``),
        kept in the same record as the reason; empty for another plan or model, and when the last answer was used."""
        try:
            record = json.loads((self.fi_dir / (file or _SEARCH_ASKED)).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        if (isinstance(record, dict) and record.get("sha") == sha and record.get("model") == self._plan_model_name()
                and record.get("problem")):
            return str(record.get("refused_block") or "")
        return ""

    def _search_schema_refused(self, sha: str, file: str | None = None) -> bool:
        """Whether the last answer of the plan's model was refused for its form (the record keeps ``schema``), for this plan
        and model: such an answer is asked again within ``_SCHEMA_RETRY_ASKS`` more tries."""
        try:
            record = json.loads((self.fi_dir / (file or _SEARCH_ASKED)).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return bool(isinstance(record, dict) and record.get("sha") == sha and record.get("model") == self._plan_model_name()
                    and record.get("schema") and record.get("problem"))

    def _search_asked(self, sha: str) -> int:
        """How many times FI has asked the plan's model to write the search (see :meth:`_search_record`)."""
        return self._search_record(sha)[0]

    def _parts_asked(self, sha: str) -> list[str]:
        """The protocol parts FI has already asked the plan's model about, for the plan (``sha``) and model asked now (the
        record is kept by part name too, because FI's own note in plan.md changes the plan's hash between two steps)."""
        try:
            record = json.loads((self.fi_dir / _PART_ASKED).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if isinstance(record, dict) and record.get("sha") == sha and record.get("model") == self._plan_model_name():
            return [str(p) for p in record.get("parts") or []]
        return []

    def _record_search_asked(self, count: int, calls: int, sha: str, outcome: str, file: str | None = None,
                             extra: dict[str, Any] | None = None) -> bool:
        """Write the record; ``False`` when it could not be written."""
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / (file or _SEARCH_ASKED)).write_text(json.dumps({
                "count": count, "calls": calls, "sha": sha, "model": self._plan_model_name(), "outcome": outcome,
                "at": _frozen.now(), **(extra or {}),
            }) + "\n", encoding="utf-8")
            return True
        except OSError as e:
            self._log.warning("[plan] couldn't record that FI asked the plan's model for the search: %r", e)
            return False

    async def _ask_plan_for_the_search(self, state: QuestState, design: Any, plan_sha: str) -> tuple[Any, str]:
        """Before a search for the best design stops because the plan's ``optimisation`` block is missing or cannot be
        used, ask the plan's model itself to write it (at most twice per plan; the count is kept in
        ``.fi/search_block_asked.json``, so a resume does not ask again): a scientist does not write FI's block, and the
        topic already says what is to be made as low or as high as possible. Only a rewrite of the block (and
        ``study_type``) is kept; a change to anything else is put back. Returns the design to use and the hash of its
        plan: the plan as rewritten once the block is usable, else as it was (the stop that follows says what was tried)."""
        if (not self._runs_code(state) or self._client is None or _frozen.load(self.quest_root) is not None
                or not plan_sha):
            return design, plan_sha

        def lacks(now: Any) -> list[str]:
            wanted, missing, why = self._search_problems(state, now)
            # A first scan the plan wrote and FI's reading left out is asked back, not lost without a word: the search
            # could start without it, but what it was to compare would not be run (it never stops the quest).
            left = self._scan_left_out(now) if wanted else []
            if not (wanted and (missing or left)):
                return []
            # The plain words for the card, and the reader's exact reason for run.log (what the ask quotes to the model).
            return [*_optim.plain_gaps(now, why, self._scan_limit_s()), *([f"the block cannot be read: {why}"] if why else []),
                    *[f"the first scan was left out: {n}" for n in left]]

        def request(now: Any, text: str) -> str:
            _wanted, missing, _why = self._search_problems(state, now)
            topic = state.get("topic") or self.config.topic
            if not missing:  # only a first scan left out: that part alone is asked for, with the reader's reasons
                return _optim.scan_request(topic, self._scan_left_out(now), left_out=True)
            scan = [m for m in missing if m.startswith("the coarse scan cannot run")]
            if scan and len(scan) == len(missing):
                return _optim.scan_request(topic, scan)  # the block is complete: only the scan is to be rewritten
            left_out = [m.group(0).lstrip("- ").strip() for m in re.finditer(
                r"^.*the optimisation block was left out.*$", text, re.MULTILINE)][:3]
            return _optim.ask_request(topic, missing, left_out)

        # A scan that cannot run is the only thing wrong: the plan is asked once to rewrite it (a rewrite that still does not
        # fit is not asked for again; the honest stop that follows says why, at design, not hours into the run).
        _wanted, now_missing, _why = self._search_problems(state, design)
        scan_only = (all(m.startswith("the coarse scan cannot run") for m in now_missing)
                     if now_missing else bool(self._scan_left_out(design)))
        return await self._ask_plan_to_complete(
            design, plan_sha, lacks=lacks, request=request, keep=self._keep_only_the_search, file=_SEARCH_ASKED,
            what="search for the best design", short="the search",
            note="the search for the best design, written by the plan's model", ask_limit=1 if scan_only else None)

    async def _ask_plan_for_the_grid(self, state: QuestState, design: Any, plan_sha: str, *,
                                     again: bool = False) -> tuple[Any, str]:
        """Before any code is written, a ``protocol.grid`` of more settings than FI lists (:func:`core.optimise_search.grid_too_large`,
        the same count the run stops on) is asked of the plan's model once to rewrite, with the plan's own arithmetic (at most
        once per plan, counted in ``.fi/grid_asked.json``, so a resume does not ask again). Only the grid and the parts a
        smaller grid changes with it (``sub_studies``, ``in_each_run``, ``constants``) are kept from the answer. A plan that fits is not asked,
        and nothing is logged for it. The stop that follows when it still does not fit is :meth:`_grid_too_large_stop`.
        ``again``: the count is made once more after what the topic asks for was added to the plan (an added setting can make
        the study larger), with one request of its own, so the plan is counted after every edit FI makes to it.
        Returns the design to use and the hash of its plan."""
        if (not self._runs_code(state) or self._client is None or _frozen.load(self.quest_root) is not None
                or not plan_sha or _optim.has_block(design)):
            return design, plan_sha
        asked_file = _GRID_ASKED_AGAIN if again else _GRID_ASKED

        def problem(now: Any) -> str:
            protocol = now.get("protocol") if isinstance(now, dict) else None
            return "" if _optim.has_block(now) else _optimise_search.grid_too_large(_optimise_search.study_grid(protocol))

        if not problem(design) or self._grid_asked_of(asked_file) > 0:
            return design, plan_sha  # a fit grid is not asked; a grid asked once is not asked again (FI's own notes change the hash)

        def lacks(now: Any) -> list[str]:
            return [p for p in [problem(now)] if p]

        def request(now: Any, _text: str) -> str:
            return _plan.grid_request(state.get("topic") or self.config.topic, problem(now))

        def keep(before: str, answer: dict[str, Any]) -> str:
            kept = self._keep_only_the_parts(before, answer, ["grid", "sub_studies", "in_each_run", "constants"])
            if kept is None:
                raise ValueError("your answer has none of the asked parts in a form that can be read; plan.md is unchanged")
            return kept

        return await self._ask_plan_to_complete(
            design, plan_sha, lacks=lacks, request=request, keep=keep, file=asked_file,
            what="what fits the settings limit", short="the settings",
            note="the grid of settings made small enough to run, written by the plan's model", ask_limit=1,
            extra={"parts": ["grid"]})

    def _grid_asked_of(self, file: str, iteration: int | None = None) -> int:
        """How many times FI has asked about the grid's size on the model asked now (``iteration``: for a drafted design, in
        that design round only). Kept by name, not by the plan's hash, because FI's own notes in plan.md change the hash."""
        try:
            record = json.loads((self.fi_dir / file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        if (not isinstance(record, dict) or record.get("model") != self._plan_model_name()
                or (iteration is not None and record.get("iteration") != iteration)):
            return 0
        try:
            return max(0, int(record.get("count") or 0))
        except (TypeError, ValueError):
            return 0

    async def _ask_design_for_the_grid(self, state: QuestState, design: Any, *, iteration: int, again: bool = False) -> Any:
        """A design drafted at this step (a redesign: it did not come from ``plan.md``) whose ``protocol.grid`` holds more
        settings than FI lists is asked of the plan's model once per design round (counted in ``.fi/drafted_grid_asked.json``
        BEFORE the call, so a resume does not ask again), the way the plan's own grid is (:meth:`_ask_plan_for_the_grid`).
        Only a grid, ``in_each_run`` and ``constants`` that pass the same reading as the plan's are taken; the rest of the
        design is untouched. A grid still too large is the stop of :meth:`_grid_too_large_stop`. ``again``: the one more request,
        after what the topic asked for was added to the redesign made its grid larger (counted in its own file, as the plan's
        is: :meth:`_ask_plan_for_the_grid`). Returns the design to use."""
        asked_file = _DRAFTED_GRID_ASKED_AGAIN if again else _DRAFTED_GRID_ASKED
        protocol = design.get("protocol") if isinstance(design, dict) else None
        if (not isinstance(protocol, dict) or protocol.get("optimisation") or not self._runs_code(state)
                or self._client is None or _frozen.load(self.quest_root) is not None):
            return design
        problem = _optimise_search.grid_too_large(_optimise_search.study_grid(protocol))
        if not problem or self._grid_asked_of(asked_file, iteration) > 0:
            return design
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / asked_file).write_text(json.dumps(
                {"count": 1, "model": self._plan_model_name(), "iteration": iteration, "at": _frozen.now()}) + "\n",
                encoding="utf-8")
        except OSError as e:
            self._log.warning("[design] FI could not write down that it was asking about the redesign's grid, so it did not "
                              "ask: %r", e)
            return design
        self._log.warning("[design] the redesign's grid is too large to run (%s); FI asked the plan's model to rewrite it "
                          "(1 of 1%s)", _one_line(problem, 300), ", after what the topic asks for was added" if again else "")
        self._progress("Asking the model to make the settings of the redesign small enough to run")
        ask = (_plan.grid_request(state.get("topic") or self.config.topic, problem)
               + "\n\nThe protocol drafted so far (keep what is usable in it):\n"
               + json.dumps(protocol, ensure_ascii=False, default=str)[:4000]
               + "\n\nAnswer with the asked parts alone, as one JSON object whose keys are their names "
               + "(for example {\"grid\": {...}}).")
        try:
            obj = _parse_json_lenient(await self._chat(ask, node="plan_revise"))
        except Exception as e:  # noqa: BLE001 -- the stop that follows says what was tried
            self._log.warning("[design] the plan's model could not be asked about the redesign's grid: %s", e)
            return design
        if isinstance(obj, dict) and isinstance(obj.get("protocol"), dict):
            obj = obj["protocol"]
        if not isinstance(obj, dict) or not isinstance(obj.get("grid"), dict):
            self._log.warning("[design] the plan's model's answer about the redesign's grid had no grid")
            return design
        candidate = {k: v for k, v in protocol.items() if k not in ("grid", "sub_studies", "in_each_run", "constants")}
        for part in ("grid", "sub_studies", "in_each_run", "constants"):
            # a rewritten grid replaces the old division into sub-studies: only the answer's own is taken
            value = obj.get(part) if obj.get(part) is not None else (
                protocol.get(part) if part not in ("grid", "sub_studies") else None)
            if value is not None:
                candidate[part] = value
        if isinstance(candidate.get("in_each_run"), dict) and isinstance(candidate.get("grid"), dict):
            moved = [k for k in candidate["grid"] if k in candidate["in_each_run"]]
            candidate["grid"] = {k: v for k, v in candidate["grid"].items() if k not in moved}
            if isinstance(candidate.get("sub_studies"), list) and moved:
                candidate["sub_studies"] = _plan.sub_studies_without(candidate["sub_studies"], moved) or None
                if candidate["sub_studies"] is None:
                    del candidate["sub_studies"]
        fixed, why = _plan.normalize_protocol(candidate)
        if fixed is None or not fixed.get("grid"):
            self._log.warning("[design] the redesign's grid, as the plan's model wrote it, cannot be read: %s", why)
            return design
        self._log.info("[design] the redesign's grid was rewritten by the plan's model: %s",
                       _optimise_search.grid_too_large(_optimise_search.study_grid(fixed)) or "it fits now")
        kept = {**protocol, **{k: fixed[k] for k in ("grid", "sub_studies", "in_each_run", "constants") if k in fixed}}
        if "sub_studies" not in fixed:
            kept.pop("sub_studies", None)
        return {**design, "protocol": kept}

    def _move_misplaced_parts(self) -> list[str]:
        """A protocol part the plan's model wrote beside ``protocol`` (at the top level of the design block, as the search
        block was in a real plan) is moved under it in plan.md, as a version FI wrote itself, and said in run.log: the
        readers look only under ``protocol``, so a part left there read as absent. A moved part that cannot be read is
        treated as the plan step treats one (:func:`core.plan.repair_protocol`): the unreadable piece is left out and plan.md
        says why, quoting the entry as written, which is what the next request to the plan's model shows. Returns the
        names moved (empty when none, or when plan.md cannot be rewritten)."""
        path = _plan.plan_path(self.quest_root)
        if not path.is_file():
            return []
        try:
            text = path.read_text(encoding="utf-8")
            written = _plan.raw_design_block(text, as_written=True)
            moved = _plan.hoist_protocol_parts(written)[1]
            if not moved:
                return []
            notes: list[str] = []

            def change(design: dict[str, Any]) -> dict[str, Any]:
                protocol = design.get("protocol")
                if not isinstance(protocol, dict):
                    return design
                protocol = dict(protocol)
                if isinstance(protocol.get("optimisation"), dict):
                    # The search block on its own first: a block that cannot be read is left out with the reason and the
                    # entry as written, and the plan's own `grid` stays (it is not merged into a block that is not kept).
                    held_back = self._hold_search_block_back(protocol["optimisation"])
                    block, block_notes = (None, held_back) if held_back else _optim.repair(protocol["optimisation"])
                    notes.extend(block_notes)
                    if held_back:
                        design = {**design, "study_type": "find_best_design"}
                    if block is None:
                        del protocol["optimisation"]
                    else:
                        protocol["optimisation"] = block
                if _plan.normalize_protocol(protocol)[0] is None:
                    reading, notes_made = _plan.read_protocol_with_notes(protocol)
                    fixed = reading.kept
                    self._keep_plan_reading(reading, "plan.md, a part written beside `protocol`")
                    notes.extend(notes_made)
                    protocol = fixed or {k: v for k, v in protocol.items() if k not in moved}
                return {**design, "protocol": protocol}

            edited = _plan.edit_design_block(text, change)  # reads the block with the parts under `protocol`
            if edited is None or edited == text or _plan.parse(edited).design is None:
                return []
            if notes:
                edited = _plan.add_to_section(edited, _PART_HEADING, ["", *[f"- {n}" for n in notes]])
            edited = _plan.refresh_optimisation_section(_plan.refresh_model_section(edited))
            path.write_text(edited, encoding="utf-8")
            _plan.record_version(self.quest_root, edited, by="engine",
                                 note="a part of the protocol written beside `protocol` was moved under it")
        except OSError as e:
            self._log.warning("[plan] couldn't move %s under `protocol` in plan.md: %r", "the misplaced protocol part", e)
            return []
        self._log.warning("[plan] plan.md had %s beside `protocol:`; FI moved %s under `protocol` (the readers look there)",
                          ", ".join(f"`{m}`" for m in moved), "it" if len(moved) == 1 else "them")
        for note in notes:
            self._log.warning("[plan] %s", note)
        return moved

    async def _ask_plan_to_complete(
        self, design: Any, plan_sha: str, *, lacks: Any, request: Any, keep: Any, file: str, what: str, short: str,
        note: str, extra: dict[str, Any] | None = None, ask_limit: int | None = None,
    ) -> tuple[Any, str]:
        """The bounded request to the plan's model to write a part of the plan that is missing or cannot be read, shared by
        every such part (the search block, the settings to sweep, the metrics ...): at most ``_SEARCH_ASKS`` requests and
        ``_SEARCH_CALLS`` chat calls per plan and model, counted in ``.fi/<file>`` BEFORE each call so a resume never asks
        again. ``lacks(design)``: what the plan still lacks, in plain sentences (empty when complete);
        ``request(design, text)``: what to ask; ``keep(before_text, parts)``: the text to write, which keeps only the
        asked parts of ``parts`` (the model's answer, a JSON object shaped like the design block: ``None`` or ``ValueError``:
        the answer is not used). The model is asked for those parts alone, never for the whole plan; FI writes the YAML. Returns the design to use and its plan's hash."""
        path = _plan.plan_path(self.quest_root)
        gaps = lacks(design)
        if inspect.isawaitable(gaps):
            gaps = await gaps
        if not gaps or not path.is_file():
            return design, plan_sha
        count, calls = self._search_record(plan_sha, file)
        last_problem = self._search_problem(plan_sha, file)  # why the last answer was not used (kept on disk)
        last_block = self._search_refused_block(plan_sha, file) if last_problem else ""  # ... and the block it refused
        schema = self._search_schema_refused(plan_sha, file)  # ... and whether it was refused for its form
        while (count < (_SEARCH_ASKS if ask_limit is None else min(_SEARCH_ASKS, ask_limit))
               + (_SCHEMA_RETRY_ASKS if schema else 0)
               and calls + _SEARCH_CALLS_PER_ASK
               <= _SEARCH_CALLS + (_SCHEMA_RETRY_ASKS * _SEARCH_CALLS_PER_ASK if schema else 0)):
            text = path.read_text(encoding="utf-8")
            sha_before = _plan.sha256(text)
            gaps_asked = list(gaps)  # what this request asks for: a part refused again is a repeat only if nothing else came
            # `what` is a part of the plan ("settings the experiment needs") or, when it begins "what", the thing the plan lacks.
            lacking = f"the plan lacks {what}" if what.startswith("what ") else f"the plan's {what} was missing"
            self._log.warning("[plan] %s: %s; FI asked the plan to complete it (%d of %d)", lacking,
                              "; ".join(gaps), count + 1,
                              (_SEARCH_ASKS if ask_limit is None else min(_SEARCH_ASKS, ask_limit))
                              + (_SCHEMA_RETRY_ASKS if schema else 0))
            self._progress(f"Asking the model to complete {what} in the plan" if what.startswith("what ")
                           else f"Asking the model to complete the {what} in the plan")
            ask = request(design, text)
            refused = self._search_refused(text) if file == _SEARCH_ASKED else None
            if refused and refused.get("faults") and isinstance(refused.get("block"), dict):
                # Only the faulted paths are asked about: the block the model wrote is kept as it wrote it, whole, beside
                # its faults, and what comes back is placed at those paths and read again.
                ask = self._path_only_search_request(refused)
            elif last_problem:
                # The reader's exact reason for refusing the last answer goes back to the model, not only what is missing.
                if last_block:
                    # The refused block is the base of this request: the model changes what the reason names and keeps the
                    # rest, so a part the last answer got right stays right.
                    ask += ("\n\nThe block you wrote last time, exactly as you wrote it (shown up to "
                            f"{_REFUSED_BLOCK_CHARS} characters):\n" + last_block)
                ask += ("\n\nYour last answer was not used. The reason, exactly:\n" + last_problem
                        + "\nFix that, and keep everything the plan already has."
                        + (" Start from the block you wrote last time (shown above): change only what the reason names, "
                           "and keep every other part of it exactly as you wrote it." if last_block else ""))
            extra_now = {**(extra or {}), **({"problem": last_problem, "schema": schema,
                                             **({"refused_block": last_block} if last_block else {})}
                                            if last_problem else {})}
            # Recorded BEFORE the call, with the most chat calls one request can make reserved: a call that fails, or a quest
            # killed during it, still counts, so a broken model is never asked again and again across resumes.
            if not self._record_search_asked(count + 1, calls + _SEARCH_CALLS_PER_ASK, sha_before, "asking", file, extra_now):
                # No durable count, no request: a request that could not be counted could be asked again on every resume.
                self._log.warning("[plan] FI could not write down that it was asking the plan's model, so it did not ask")
                return design, plan_sha
            used_before = getattr(self, "_plan_chat_calls", 0)
            outcome = "kept"
            earlier_problem = last_problem
            last_problem = ""
            last_block = ""
            self._answer_block = None  # the block of the answer, set by a keep that reads one (:meth:`_keep_only_the_search`)
            schema = False
            repeated = False
            try:
                await self._complete_plan_parts(ask, path, expect=text, keep=keep, note=note)
                refused = list(getattr(self, "_answer_refused", []) or [])
                if refused:
                    # Taken in part: what could not be read goes back with the next request, as a refusal does.
                    outcome = "kept in part"
                    schema = True
                    last_problem = ("Part of your last answer was used. This part was not: "
                                    + " ".join(_optim.ALSO.join(" ".join(one.split()) for one in r.split(_optim.ALSO))
                                               for r in refused))[:_REFUSAL_CHARS]
                    self._log.warning("[plan] the plan step's answer to complete %s was used in part; not used: %s",
                                      short, "; ".join(refused)[:600])
                    repeated = bool(earlier_problem) and last_problem == earlier_problem
            except _PlanEditedMeanwhile:
                outcome = "plan changed meanwhile"
                self._log.warning("[plan] plan.md was changed while FI waited for the plan's model; FI kept the version "
                                  "you saved and did not write the model's answer over it")
                print("[FI] plan.md was changed while FI waited for the plan's model: your version is kept.")
            except _ModelAnswerProblem as e:
                outcome = "no complete answer"
                self._log.warning("[plan] the plan step's answer to complete %s was not complete: %s", short, e)
            except ValueError as e:
                outcome = "not usable"
                schema = True
                last_problem = re.sub(r";? ?plan\.md is unchanged\.?$", "",
                                      _optim.ALSO.join(" ".join(one.split()) for one in str(e).split(_optim.ALSO)))[:_REFUSAL_CHARS]
                if isinstance(getattr(self, "_answer_block", None), dict):
                    last_block = json.dumps(self._answer_block, default=str, ensure_ascii=False)[:_REFUSED_BLOCK_CHARS]
                self._log.warning("[plan] the plan step's answer to complete %s could not be used: %s", short, e)
                if earlier_problem and last_problem == earlier_problem:
                    outcome = "same refusal again"
                    schema = False  # kept so: a resume does not ask once more for the same refusal
            except ProviderUnusable:
                # Never asked (a usage limit, a dead login): the ask is given back, and the step stops to be run
                # again on resume (_audited), so the plan is not judged on a question its model never answered.
                self._record_search_asked(count, calls, sha_before, "model could not be used", file, extra_now)
                raise
            except Exception as e:  # noqa: BLE001 -- the stop that follows says what is missing
                outcome = "no answer"
                self._log.warning("[plan] the plan step could not be asked to complete %s: %s", short, e)
            count += 1
            calls += max(1, getattr(self, "_plan_chat_calls", 0) - used_before)  # the calls really made
            self._record_search_asked(count, calls, _plan.sha256(path.read_text(encoding="utf-8")), outcome, file,
                                      {**(extra or {}), **({"problem": last_problem, "schema": schema,
                                                            **({"refused_block": last_block} if last_block else {})}
                                                           if last_problem else {})})
            if outcome == "same refusal again":
                # The same reason twice: the model repeats an answer the plan's rules cannot take, so another call would
                # cost a few minutes and return the same refusal. The stop that follows says what is missing.
                self._log.warning("[plan] the plan's model gave the same answer that could not be used for %s twice, so FI "
                                  "stopped asking (%d request(s) made)", short, count)
                return design, plan_sha
            if outcome in ("no complete answer", "no answer", "plan changed meanwhile"):
                again, sha = self._design_from_plan()
                return (again, sha) if again is not None and outcome == "plan changed meanwhile" else (design, plan_sha)
            again, sha = self._design_from_plan()
            if again is None:
                return design, plan_sha
            design, plan_sha = again, sha
            gaps = lacks(design)
            if inspect.isawaitable(gaps):
                gaps = await gaps
            if not gaps:
                whole = f"the plan now has {what}" if what.startswith("what ") else f"the plan's {what} is complete now"
                self._log.info("[plan] %s (asked %d time(s))", whole, count)
                print(f"[FI] {whole}; going on")
                return design, plan_sha
            if repeated and list(gaps) == gaps_asked:
                # The same part refused twice for the same reason, and nothing else came with it: asking again would return
                # it once more. Recorded so, so a resume does not ask once more either.
                self._record_search_asked(count, calls, plan_sha, "same refusal again", file,
                                          {**(extra or {}), "problem": last_problem, "schema": False,
                                           **({"refused_block": last_block} if last_block else {})})
                self._log.warning("[plan] the plan's model gave the same part that could not be used for %s twice, so FI "
                                  "stopped asking (%d request(s) made)", short, count)
                return design, plan_sha
        return design, plan_sha

    @classmethod
    def _as_design_parts(cls, obj: dict[str, Any]) -> dict[str, Any]:
        """The model's answer shaped like the design block: parts written beside ``protocol`` (or an answer with no
        ``protocol`` at all, just the parts) are put under it, as the plan's readers look only there."""
        inner = obj.get("protocol") if isinstance(obj.get("protocol"), dict) else {}
        beside = {k: v for k, v in obj.items() if k not in cls._DESIGN_TOP_KEYS and k != "protocol"}
        top = {k: v for k, v in obj.items() if k in cls._DESIGN_TOP_KEYS}
        # a part written under `protocol` wins over the same part written beside it
        return {**top, "protocol": {**beside, **inner}}

    def _keep_plan_answer(self, text: str | None, *, chars: int | None = None) -> None:
        """Write the plan's model's raw answer to a request to complete or rewrite the plan to ``.fi/plan_answers/NNN.txt``,
        whether or not the quest keeps every model call (``output.save_model_calls``, off by default). When such an answer is
        refused, the refusal is only worth reading beside the answer itself: without it nobody can tell whether a rule was
        unclear or the model wrote something unreadable. Small, bounded, and a failure to write never stops the quest."""
        try:
            folder = self.fi_dir / "plan_answers"
            folder.mkdir(parents=True, exist_ok=True)
            number = len(list(folder.glob("*.txt"))) + 1
            if number > self._PLAN_ANSWERS_KEPT:
                return
            (folder / f"{number:03d}.txt").write_text((text or "")[: chars or self._PLAN_ANSWER_CHARS], encoding="utf-8")
        except OSError as e:
            self._log.warning("[plan] couldn't keep the plan model's answer: %r", e)

    async def _keep_refused_repair(self, path: str, script: str, new_code: str | None, reason: str,
                                   result: dict[str, Any] | None = None) -> None:
        """A repair that was refused is not lost: the script it proposed, why it was refused (``reason.txt``) and what is
        known of the run it was meant to fix (``result.json``) go to ``refs/fi/refused/<NNN>`` in ``code/``'s own git
        history (:func:`core.code_project.keep_refused`); the branch and the folder stay as the refusal left them. Best
        effort: never stops or changes the repair, and a missing git is not an error."""
        if not (isinstance(new_code, str) and new_code.strip()):
            return
        try:
            from . import code_project as _project

            body = {"repair_path": path, "script": script, "reason": reason, **(result or {})}
            await asyncio.to_thread(
                _project.keep_refused, self.quest_root, None, {script: new_code},
                f"{path} repair refused: {script}: {reason}",
                extras={"result.json": json.dumps(body, default=str)}, log=self._log)
        except Exception:  # noqa: BLE001 -- the record is a convenience
            pass

    def _keep_repair_answer(self, kind: str, text: str | None) -> None:
        """Write a repair's raw reply to ``.fi/repair_answers/<kind>_NNN.txt`` (bounded, like :meth:`_keep_plan_answer`), so a
        refusal of the repair can be read beside what the model wrote."""
        try:
            folder = self.fi_dir / "repair_answers"
            folder.mkdir(parents=True, exist_ok=True)
            number = len(list(folder.glob(f"{kind}_*.txt"))) + 1
            if number > self._PLAN_ANSWERS_KEPT:
                return
            (folder / f"{kind}_{number:03d}.txt").write_text((text or "")[: self._PLAN_ANSWER_CHARS], encoding="utf-8")
        except OSError as e:
            self._log.warning("[%s] couldn't keep the repair's answer: %r", kind, e)

    def _scripts_built_from(self, state: QuestState, outline: dict[str, Any]) -> dict[str, str]:
        """The parts of the plan the two scripts are written from now (:mod:`core.code_delta`)."""
        try:
            design = json.loads(self._code_design_block(state))
        except ValueError:
            design = {}
        protocol = self._protocol_block(state)
        specs, _notes = _function_steps.model_functions(outline, protocol if isinstance(protocol, dict) else None)
        layout = self._code_layout(state) or {}
        hashes = _function_steps.function_hashes(specs, protocol if isinstance(protocol, dict) else None)
        return _code_delta.parts(design, hashes, str(layout.get("package") or ""))

    async def _incremental_scripts(self, state: QuestState, outline: dict[str, Any]) -> dict[str, Any] | None:
        """On a revised plan, the two scripts as they are with ONLY the functions the revision touched replaced
        (``{"simulate", "analysis", "deps"}``), or ``None`` when that cannot be told or done and the scripts are written whole
        as before (the reason is said in run.log). FI compares the plan parts the scripts were written from with the new
        ones (:mod:`core.code_delta`): none changed keeps both scripts and every repair they received, without a request;
        some changed asks the model for the functions to change and splices them in (:func:`core.code_delta.splice_functions`)."""
        template = self._prompts.get("implement_update")
        layout = self._code_layout(state)
        disk = self._scripts_on_disk()
        if (template is None or not layout or layout["shape"] != _code_layout.PACKAGE or not self._split_on(state)
                or not disk.get(_split_run.SIMULATE_NAME, "").strip() or not disk.get("experiment.py", "").strip()):
            return None
        if not _function_steps.is_ready(self.quest_root, layout["package"]):
            return None
        delta = _code_delta.compare(self.quest_root, self._scripts_built_from(state, outline))
        if not delta.known:
            self._log.info("[implement] the scripts are written whole: %s", delta.reason)
            return None
        deps = list(state.get("deps") or [])
        kept = {"simulate": disk[_split_run.SIMULATE_NAME], "analysis": disk["experiment.py"], "deps": deps}
        if not delta.changed:
            self._log.info("[implement] the revised plan changes nothing the scripts read: simulate.py and experiment.py "
                           "are kept as they are, with every repair they received")
            return kept
        self._log.info("[implement] the revised plan changes %s; asking only for the functions of the scripts that must change",
                       ", ".join(delta.changed))
        shown = self._package_shown(state)
        prompt = template.substitute(
            topic_block=self._topic_block(), changes_block=_code_delta.describe(delta),
            package_block=_package_block_text(shown[1] if shown else {}) or "(none)",
            simulate_code=kept["simulate"], analysis_code=kept["analysis"], design_block=self._code_design_block(state),
            clarify_block=_format_clarify(state))
        prompt = _skeleton.repair_note(kept["simulate"]) + prompt
        try:
            text = await self._chat(prompt, node="implement")
        except Exception as exc:  # noqa: BLE001 -- the scripts are then written whole
            self._log.warning("[implement] asking for the changed functions failed (%r); the scripts are written whole", exc)
            return None
        self._keep_repair_answer("implement_update", text)
        if (text or "").strip().upper().startswith("NO CHANGE"):
            self._log.info("[implement] the model says no function of the scripts must change for the revision; kept as they are")
            return kept
        if "FULL REWRITE NEEDED" in (text or "").upper():
            self._log.info("[implement] the model says the revision needs a module-level change; the scripts are written whole")
            return None
        _main, marked = _finding_repair.file_blocks(text)
        names = {"simulate.py": "simulate", "experiment.py": "analysis"}
        done = {n: marked[n] for n in names if n in marked}
        if not done:
            self._log.warning("[implement] the reply held no function of simulate.py or experiment.py; the scripts are written whole")
            return None
        out = dict(kept)
        for name, block in done.items():
            old = kept[names[name]]
            new, replaced, why = _code_delta.splice_functions(old, block, code=self.quest_root / "code", rel=name)
            broken = why or (_trial_runner.repair_breaks_contract(old, new, name) if new is not None else "")
            if broken:
                self._log.warning("[implement] the changed functions of %s cannot be used (%s); the scripts are written whole", name, broken)
                return None
            out[names[name]] = new
            self._log.info("[implement] %s: %d function(s) replaced or added (%s); the rest is as it was", name, len(replaced), ", ".join(replaced[:8]))
        out["deps"] = sorted({*deps, *_deps_from_tail(text.rsplit("```", 1)[-1])})
        return out

    def _code_rel(self, path: Path) -> str:
        """``path`` relative to the quest's code folder (posix), or ``""`` when it is not inside it."""
        try:
            return path.resolve().relative_to((self.quest_root / "code").resolve()).as_posix()
        except ValueError:
            return ""

    def _script_from_reply(self, node: str, old: str, code: object, parsed: Any, name: str,
                           rel: str = "") -> tuple[str, str]:
        """``(the script a repair's reply makes, why none)``: the reply's ``functions`` are put into ``old`` and every other
        byte of it stays (:mod:`core.script_reply`); a reply that is a whole script is taken whole. The one contract
        check (``repair_breaks_contract``) still judges the result, in the caller."""
        functions = parsed.get("functions") if isinstance(parsed, dict) else ""
        # the other files of the study's code folder that use a repaired function are read, so a changed signature that
        # binds every use is accepted; a script outside the folder (``rel`` empty) keeps its parameters
        made = _script_reply.resolve(old, code, functions, code_dir=self.quest_root / "code" if rel else None, rel=rel)
        if made.kind == "functions":
            self._log.info("[%s] %s: the reply's %d function(s) are put into the script, the rest of it is as it was (%s)",
                           node, name, len(made.replaced), ", ".join(made.replaced[:6]))
        elif made.why:
            self._log.warning("[%s] the repair of %s holds no usable script (%s)", node, name, made.why)
        return _skeleton.without_raise(made.code), made.why

    async def _ask_for_plan_parts(self, prompt: str) -> dict[str, Any]:
        """One JSON object of the asked parts from the plan's model (node ``plan_revise``, so it is the same model, limits
        and call count as every other request of the plan step). A reply that holds no such object is asked once more,
        short (``agents/json_reanswer.md``): at most two calls. ``ValueError`` (the reader's reason, sent back with the next
        request) when neither reply can be read."""
        text = await self._chat(prompt, node="plan_revise")
        self._keep_plan_answer(text)
        obj = _parse_json_lenient(text, node="plan_revise")
        if isinstance(obj, dict) and obj:
            return self._as_design_parts(obj)
        objects = _top_level_json_objects(text or "")
        if not (text or "").strip():
            wrong = "it was empty"
        elif not objects:
            wrong = f"it was {len(text):,} characters with no JSON object in it"
        elif len(objects) > 1:
            wrong = f"it held {len(objects)} separate JSON objects, not one"
        else:
            wrong = "its JSON object was empty"
        if not objects and (text or "").strip():
            merged = await self._reask_unreadable_keys(prompt, text, node="plan_revise", want=("protocol",), ok=bool)
            if merged is not None and merged[1]:
                return self._as_design_parts(merged[1])
        self._log.warning("[plan] the plan step's reply could not be used (%s); asking once more for the whole answer (one "
                          "short JSON object): nothing in it could be read", wrong)
        echo = (f"\nYour previous reply, for reference (say the same things, in the asked-for shape):\n\n{text}\n"
                if text and len(text) <= self._REANSWER_ECHO_CHARS else "")
        retry = self._prompts["json_reanswer"].substitute(
            original_prompt=prompt, what_was_wrong=wrong,
            keys="`protocol` (the asked parts under their names), and `study_type` or `variables` only when asked", earlier_reply=echo)
        text = await self._chat(retry, node="plan_revise")
        self._keep_plan_answer(text)
        obj = _parse_json_lenient(text, node="plan_revise")
        if isinstance(obj, dict) and obj:
            self._log.info("[plan] the second, shorter answer could be read")
            return self._as_design_parts(obj)
        raise ValueError("your answer was not one JSON object holding the asked parts (`protocol` with the parts under their "
                         "names); plan.md is unchanged")

    async def _revise_plan_parts(
        self, request: str, path: Path, *, scope: _plan_scope.PartScope, note: str, by: str = "engine",
        finish: Any = None, passages: str = "",
    ) -> dict[str, Any]:
        """The way every repair of the plan asks its model for a change: the plan is shown read-only and the model returns
        ONLY the parts ``scope`` names, as one small JSON object (:meth:`_complete_plan_parts`); FI merges them into the plan
        (:meth:`_keep_only_the_parts`, each part read as the plan reads it) and writes plan.md, so every other byte of the
        design block and of the prose stays as it was, and the sections FI shows from the block are shown again by FI. A reply
        that names a part outside ``scope``, or a part that cannot be read, is refused with FI's reason and asked once more
        with that reason. ``finish(text)``: may return the text to write instead, or ``None`` / raise ``ValueError`` for an
        answer that must not be used. ``ValueError`` (plan.md unchanged) when neither answer can be used."""
        expect = path.read_text(encoding="utf-8")
        asked = request.rstrip() + "\n\n" + scope.describe()

        def keep(before: str, answer: dict[str, Any]) -> str:
            old = (_plan.raw_design_block(before) or {}).get("protocol")
            not_applied: list[str] = []
            translated, problems = _plan_scope.translate(scope, old if isinstance(old, dict) else {}, answer, not_applied)
            if problems:
                raise _PartsRefused("; ".join(problems))
            for line in not_applied:
                self._log.warning("[plan] the answer's %s", line)
            try:
                kept = self._keep_only_the_parts(before, translated, scope.names(), passages)
                if kept is not None:
                    # The sections FI shows from the block are shown again whatever the parts, finish or not.
                    kept = _plan.refresh_optimisation_section(_plan.refresh_model_section(kept))
                    if finish is not None:
                        kept = finish(kept)
            except _PartsRefused:
                raise
            except ValueError as e:
                raise _PartsRefused(str(e)) from e
            if kept is None:
                raise _PartsRefused("none of the asked parts came back in a form that can be used")
            return kept

        problem = ""
        for attempt in (1, 2):
            ask = asked + (("\n\nYour last answer was not used. The reason, exactly:\n" + problem
                            + "\nFix that, and keep everything the plan already has.") if problem else "")
            try:
                return await self._complete_plan_parts(ask, path, expect=expect, keep=keep, note=note, by=by)
            except ValueError as e:  # FI's refusal, or a reply with no JSON object (already re-asked once, shorter)
                problem = " ".join(str(e).split())[:1500]
                self._log.warning("[plan] the plan's model answer was refused (attempt %d of 2): %s", attempt, problem)
        raise ValueError(f"the model's answer could not be used ({problem}); plan.md is unchanged")

    async def _complete_plan_parts(self, request: str, path: Path, *, expect: str, keep: Any, note: str,
                                   by: str = "engine") -> dict[str, Any]:
        """The plan's model is asked for ONLY the missing parts, as JSON (:meth:`_ask_for_plan_parts`); FI is the only writer
        of the YAML: ``keep(plan_text, parts)`` splices the parts it accepts into the plan as it stands (every check on them
        is the keep's own) and returns the text to write, or ``None`` / raises ``ValueError`` for an answer that must not be
        used. ``expect``: the plan as it was when the request was made; when plan.md differs just before the write (a
        person saved a change meanwhile) nothing is written (:class:`_PlanEditedMeanwhile`)."""
        current = path.read_text(encoding="utf-8")
        _plan.note_edit(self.quest_root, current)  # a hand edit made before this request is its own version
        self._answer_refused = []  # what of the answer the keep did not take, with the reader's reasons
        prompt = self._prompts["plan_parts"].substitute(topic=self.config.topic, plan_md=current, request=request)
        parts = await self._ask_for_plan_parts(prompt)
        kept = keep(current, parts)
        if kept is None:
            raise ValueError("your answer has none of the asked parts in a form that can be read; plan.md is unchanged")
        # The model and search sections are shown from the block and never read back: shown again from the block as it is now.
        kept = _plan.refresh_optimisation_section(_plan.refresh_model_section(kept))
        if path.read_text(encoding="utf-8") != expect:
            raise _PlanEditedMeanwhile("plan.md was changed while the request was being answered")
        path.write_text(kept, encoding="utf-8")
        entry = _plan.record_version(self.quest_root, kept, by=by, note=(note() if callable(note) else note)[:300])
        self._log.info("[plan] completed by the plan's model (the asked parts only): version %d", entry["version"])
        return {"version": entry["version"], "sha256": entry["sha256"], "path": str(path)}

    async def _complete_the_search_in_the_plan(self, state: QuestState) -> None:
        """At the plan step, before a person reads the plan: a search for the best design whose block is missing is
        asked of the plan's model (:meth:`_ask_plan_for_the_search`), and so is a protocol part that could not be read
        (:meth:`_ask_plan_for_the_parts`), so the plan a person reads is complete. Then the plan is held against what the
        topic asks for (:meth:`_cover_the_topic_in_the_plan`)."""
        design, sha = self._design_from_plan()
        if design is not None and sha:
            design, sha = await self._ask_plan_for_the_search(state, design, sha)
            design, sha = await self._ask_plan_for_the_parts(state, design, sha)
            design, sha = await self._ask_plan_for_the_grid(state, design, sha)
            before = _study_size(design)
            await self._cover_the_topic_in_the_plan(state)
            again, again_sha = self._design_from_plan()  # what the topic added may have made the study larger
            if again is not None and again_sha and _study_size(again) > before:
                await self._ask_plan_for_the_grid(state, again, again_sha, again=True)
            self._write_the_ledger(state)

    def _write_the_ledger(self, state: QuestState, design: Any = None) -> None:
        """Write ``.fi/ledger.json`` (:func:`core.plan_ledger.write_ledger`): the plan's simplifications, each with the ids of the
        relations that test it (none yet) and its status, and the metrics whose definition is the plan's own
        (``unsourced_definitions``). At the end of the plan step it is read from plan.md as it stands, so a plan a person edited
        is the one recorded; for a redesign (``design`` given) it is that design. An item already recorded keeps its relation
        ids; a record for the same plan is left as it is. A failure is a warning and never stops the quest."""
        if not self._runs_code(state) or _frozen.load(self.quest_root) is not None:
            return
        try:
            if design is None:
                design, sha = self._design_from_plan()
            else:
                sha = _plan.sha256(json.dumps(design, sort_keys=True, default=str))
            if not isinstance(design, dict) or not sha:
                return
            path = self.fi_dir / _ledger.FILE
            kept = _ledger.read_ledger(path)
            if kept is not None and kept.get("plan_sha") == sha:
                return
            simplifications, _problems = _ledger.parse_simplifications(design.get("protocol"))
            relations = {str(i.get("id")): [str(r) for r in i.get("relation_ids") or []]
                         for i in (kept or {}).get("items") or [] if isinstance(i, dict)}
            record = _ledger.ledger_record(sha, simplifications, relations)
            record["unsourced_definitions"] = self._own_definitions(design.get("protocol"))
            _ledger.write_ledger(path, record)
        except Exception as e:  # noqa: BLE001 -- a record that cannot be written never stops a quest
            self._log.warning("[plan] couldn't write the plan's ledger of simplifications: %r", e)

    def _check_the_ledger_of_a_redesign(self, state: QuestState, design: Any) -> None:
        """A design drafted at the design step (a redesign) gets the same look at its ledger parts as the plan: what is wrong
        with its simplifications, definitions and step cards (and what the topic asks for that it neither covers nor names) is
        said in run.log and kept for the writer, and its ledger is written. Nothing is asked of the model here (the redesign's
        parts are completed by :meth:`_complete_drafted_parts`) and nothing stops."""
        if not self._runs_code(state) or _frozen.load(self.quest_root) is not None or not isinstance(design, dict):
            return
        try:
            left = self._ledger_unread(design, self._literature_passages(state))
            gaps = [f"`protocol.{part}` is incomplete: {why}" for part, why in left.items() if part in _LEDGER_PARTS]
            for gap in gaps:
                self._log.warning("[design] the redesign: %s", gap)
            self._note_the_ledger_for_the_writer(design, gaps)
            self._write_the_ledger(state, design)
        except Exception as e:  # noqa: BLE001 -- never a stop
            self._log.warning("[design] the redesign's ledger could not be checked: %r", e)

    # -- what the topic asks for (core/topic_requirements.py) ------------------------------------------------------

    def _topic_requirements_read(self) -> dict[str, Any] | None:
        """``.fi/topic_requirements.json`` (what the topic asks for, or the error that kept it from being read), or ``None``
        when it was not made."""
        try:
            record = json.loads((self.fi_dir / _topic_req.FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    def _topic_covered_read(self) -> dict[str, Any]:
        try:
            record = json.loads((self.fi_dir / _topic_req.COVERED_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return record if isinstance(record, dict) else {}

    def _topic_covered_write(self, **changes: Any) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / _topic_req.COVERED_FILE).write_text(
                json.dumps({**self._topic_covered_read(), **changes}, indent=2, default=str) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[plan] couldn't record how far the plan covers the topic: %r", e)

    def _note_not_done(self, entries: list[dict[str, Any]], *, kind: str = "") -> None:
        """Keep what the study did not do as the plan wrote it in ``topic_covered.json`` ``not_done`` (``{name, sentence,
        kind}``): each is an item the paper's limitations must name (core/paper_facts.py checks it) and the run record says
        (core/disclosure.py, QUALITY.md). With ``kind``, the entries of that kind are replaced by ``entries`` (none: taken
        away); without, an entry replaces the one of the same name. Unlike ``left``, the plan and design checks do not
        rewrite it."""
        old = [e for e in self._topic_covered_read().get("not_done") or [] if isinstance(e, dict)]
        names = {str(e.get("name")) for e in entries}
        kept = [e for e in old if (e.get("kind") != kind if kind else str(e.get("name")) not in names)]
        new = [*kept, *entries]
        if new != old:
            self._topic_covered_write(not_done=new)

    def _refresh_not_done(self, state: QuestState) -> None:
        """Before the run: take away each ``not_done`` entry the plan as it now stands no longer supports (an approved
        change, a person's edit of plan.md), so the paper is never made to state a limitation the study does not have.
        Every entry has a ``kind``; an entry of an unknown kind (none is written any more) is taken away too."""
        entries = [e for e in self._topic_covered_read().get("not_done") or [] if isinstance(e, dict)]
        if not entries:
            return
        from . import in_run as _in_run

        protocol = self._draft_protocol(state) or {}
        layout = _code_layout.load(self.quest_root)
        still = {
            "in_each_run": not _in_run.of(protocol),
            "code_files": bool(_code_layout.files_note(layout)),
            "split_overridden": self._split_overridden(state),
        }
        kept = [e for e in entries if still.get(str(e.get("kind")), False)]
        if kept != entries:
            self._topic_covered_write(not_done=kept)
            gone = ", ".join(str(e.get("name")) for e in entries if e not in kept)
            self._log.info("[execute] the plan as it now stands does what was recorded as not done (%s); taken away", gone)

    def _in_each_run_not_done(self, why: Any, *, asked_again: bool) -> None:
        """``protocol.in_each_run`` could not be used (``why``, the reader's reason; ``None``: it can be, or there is none):
        an item the paper must name. Said as what FI recorded, not as what the code did (the code may compare those values
        in each run without the protocol saying so)."""
        if not why:
            self._note_not_done([], kind="in_each_run")
            return
        reason = str(why).split("; " + _plan.ASK_MARK)[0].strip().rstrip(".")
        again = ", even when the plan was written again once" if asked_again else ""
        sentence = (f"{reason}{again}. So no comparison of those values inside each run was recorded, and there is no "
                    "result per value of them")
        before = [e for e in self._topic_covered_read().get("not_done") or [] if isinstance(e, dict)
                  and e.get("kind") == "in_each_run"]
        self._note_not_done([{"name": "comparison inside each run", "sentence": sentence, "kind": "in_each_run"}],
                            kind="in_each_run")
        if not before or before[0].get("sentence") != sentence:
            self._log.warning("[plan] %s; the paper must say so", sentence)

    def _topic_block(self) -> str:
        """The topic and the formulas it states, as every request that writes or repairs code is shown them
        (:func:`core.topic_requirements.code_block`): a sentence that states a result is left out."""
        frozen = _frozen.load(self.quest_root)
        protocol = frozen.get("protocol") if isinstance(frozen, dict) else None
        if not isinstance(protocol, dict):
            planned, _why = _plan.load_design(self.quest_root)
            protocol = planned.get("protocol") if isinstance(planned, dict) else None
        # A number of the topic that is also a number of the plan's checks (an expected value, a tolerance, a worked
        # example) is taken out whether or not the topic marked it as a result; formula constants and settings stay.
        values = _pairs.plan_values(protocol if isinstance(protocol, dict) else None)
        return _topic_req.code_block(self.config.topic, self._topic_requirements_read(),
                                     scrub=lambda text: _pairs.scrub(text, values))

    def _reader_and_planner(self, node: str) -> tuple[str, str, bool | None]:
        """``(reviewer, planner, same_model)`` for the last call under ``node``, as :meth:`_review_oracles` decides it."""
        answered, planned = self._chat_provenance(node), self._chat_provenance("plan")
        reviewer_asked = self._model_for_node(node) or self.config.provider.model or ""
        planner_asked = self._model_for_node("plan") or self.config.provider.model or ""
        reviewer = str(answered.get("model") if answered.get("reported") else reviewer_asked or "")
        planner = str(planned.get("model") if planned.get("reported") else planner_asked or "")
        if answered.get("reported") and planned.get("reported"):
            same: bool | None = _review.same_model(answered.get("model"), planned.get("model"))
        elif reviewer_asked and planner_asked:
            same = _review.same_model(reviewer_asked, planner_asked)
        elif not self._model_for_node(node) and not self._model_for_node("plan"):
            same = True
        else:
            same = None
        return reviewer, planner, same

    async def _read_topic_requirements(self) -> dict[str, Any]:
        """What the topic asks for, read once (one call, node ``oracle_review.topic``) and kept in
        ``.fi/topic_requirements.json`` as soon as it is had, so a resume never asks again. A reading that could not be had is
        kept as ``{"error": ...}`` and never stops the quest."""
        kept = self._topic_requirements_read()
        if kept is not None:
            return kept
        topic = str(self.config.topic or "").strip()
        req: dict[str, Any] | None = None
        error = ""
        if not topic:
            error = "the quest has no topic text"
        else:
            prompt = self._prompts["topic_requirements"].substitute(topic=topic[:8000])
            try:
                reply = await self._chat(prompt, node=_topic_req.NODE)
                req = _topic_req.parse(_parse_json_lenient(reply, node=_topic_req.NODE))
                if req is None:
                    error = "the reply named nothing the topic asks for"
            except _ModelAnswerProblem as e:
                error = f"the provider withheld or cut off its answer ({str(e)[:160] or type(e).__name__})"
            except Exception as e:  # noqa: BLE001 -- a reading that could not be had never stops a quest
                error = f"the call failed ({str(e)[:160] or type(e).__name__})"
        reviewer, planner, same = self._reader_and_planner(_topic_req.NODE)
        record: dict[str, Any] = {**(req or {"factors": [], "quantities": [], "formulas": [], "comparisons": [],
                                             "constraints": [], "stated_results": []}),
                                  "reviewer": reviewer, "planner": planner, "same_model": same, "at": _frozen.now(),
                                  **({"error": error} if error else {})}
        if error:
            self._log.warning("[plan] what the topic asks for could not be read: %s", error)
        else:
            self._log.info("[plan] the topic asks for %d factor(s), %d quantity(ies), %d formula(s)%s",
                           len(record["factors"]), len(record["quantities"]), len(record["formulas"]),
                           "; read by the plan's own model" if same else "")
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / _topic_req.FILE).write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[plan] couldn't record what the topic asks for: %r", e)
        return record

    async def _judge_topic_items(self, req: dict[str, Any], plan: dict[str, Any], unresolved: list[str]) -> dict[str, str]:
        """The verdicts (``covered`` / ``replaced`` / ``missing``) on the items a name match could not settle: ONE call, node
        ``oracle_review.topic``, kept by the items and the plan it saw (so it is not asked again for the same), and at most
        :data:`core.topic_requirements.MAX_JUDGE_CALLS` calls in a quest (counted before each). ``{}`` when nothing could
        be judged; an item nobody judged is never a gap."""
        if not unresolved:
            return {}
        items_text, plan_text = _topic_req.judge_parts(req, plan, unresolved)
        key = hashlib.sha256((items_text + plan_text).encode("utf-8")).hexdigest()[:20]
        record = self._topic_covered_read()
        judged = record.get("judged") if isinstance(record.get("judged"), dict) else {}
        if isinstance(judged.get(key), dict):
            return {str(k): str(v) for k, v in judged[key].items()}
        calls = int(record.get("judge_calls") or 0)
        if self._client is None or calls >= _topic_req.MAX_JUDGE_CALLS:
            return {}
        self._topic_covered_write(judge_calls=calls + 1)  # counted before the call: a resume never asks again for it
        prompt = self._prompts["topic_cover"].substitute(
            topic=" ".join(str(self.config.topic or "").split())[:3000], items=items_text, plan=plan_text)
        try:
            reply = await self._chat(prompt, node=_topic_req.NODE)
            verdicts = _topic_req.parse_judgement(_parse_json_lenient(reply, node=_topic_req.NODE), unresolved)
        except _ModelAnswerProblem as e:
            self._log.warning("[plan] the plan could not be held against the topic (the provider withheld its answer: %s)", e)
            return {}
        except Exception as e:  # noqa: BLE001 -- a reading that could not be had never stops a quest
            self._log.warning("[plan] the plan could not be held against the topic: %s", e)
            return {}
        self._topic_covered_write(judged={**judged, key: verdicts})
        return verdicts

    def _gaps_to_say(self, gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The gaps the paper may state (:func:`core.topic_requirements.stands`): one a reader called missing and also covered in
        another reading is kept out of ``left`` and named in run.log. Plan.md and the redesign asks still use all of ``gaps``."""
        history = _topic_req.verdict_history(self._topic_covered_read())
        said = [g for g in gaps if _topic_req.stands(g, history)]
        for g in gaps:
            if g not in said:
                self._log.info("[plan] a reader called %r missing but also covered in another reading; the paper does not say "
                               "the study lacks it", g.get("name"))
        return said

    async def _topic_gaps(self, req: dict[str, Any], design: Any,
                          protocol: Any = None) -> tuple[list[dict[str, Any]], list[str]]:
        """``(gaps, unjudged)``: what the topic asks for that ``design`` (with ``protocol``, the one that runs, when that is
        another) lacks, and the names of what nobody could check against it (the plan and the paper say so)."""
        if _topic_req.is_empty(req):
            return [], []
        plan = _topic_req.view(design, protocol)
        mapped = _topic_req.mapped_items(req, plan)
        record = self._topic_covered_read()
        ever = [str(n) for n in record.get("ever_missing") or []]
        if mapped:
            # What the plan itself names as covering an item (and FI checked exists and covers it) stands over a reading
            # that called the item missing: the disagreement is kept.
            self._topic_covered_write(mapped=mapped, overruled=sorted(set(mapped) & set(ever)))
        _gaps, unresolved = _topic_req.match(req, plan)
        verdicts = await self._judge_topic_items(req, plan, unresolved)
        found, unjudged = _topic_req.gaps_from(req, plan, verdicts)
        fresh = [g["name"] for g in found if g["name"] not in ever]
        if fresh:
            self._topic_covered_write(ever_missing=[*ever, *fresh])
        return found, unjudged

    def _keep_only_what_the_topic_asked(self, before_text: str, parts: dict[str, Any], gaps: list[dict[str, Any]]) -> str:
        """``before_text`` with only what the topic asked for taken from ``parts`` (the model's answer, shaped like the design
        block: :func:`core.topic_requirements.merge`); everything else is the plan's as it was. ``ValueError`` when none of it
        came back in a form that can be read."""
        old, new = _plan.raw_design_block(before_text), parts
        merged = _topic_req.merge(old, new, gaps) if isinstance(old, dict) and isinstance(new, dict) else None
        if merged is None:
            raise ValueError((_topic_req.restated(new, gaps)
                              or "your answer has none of what was asked in a form that can be read") + "; plan.md is unchanged")
        for moved in _topic_req.moved_holds(old.get("protocol"), merged.get("protocol")):
            self._log.warning("[plan] what the topic asks for changed a part of the study: %s", moved)
        # Part by part and entry by entry: what can be read is taken, and what cannot goes back with its reason (the
        # topic's gaps are worked out again, so a refused part is asked for again).
        before = old.get("protocol") if isinstance(old.get("protocol"), dict) else {}
        fixed, refused = _plan.accept_answer(before, merged.get("protocol"))
        if fixed is None:
            raise ValueError(f"what the plan added cannot be read ({' '.join(refused)}); plan.md is unchanged")
        self._answer_refused = refused
        edited = _plan.edit_design_block(before_text, lambda _design: {**merged, "protocol": fixed})
        if edited is None or _plan.parse(edited).design is None:
            raise ValueError("the plan with your answer in it cannot be read; plan.md is unchanged")
        return _plan.refresh_model_section(edited)

    async def _ask_plan_for_the_topic(self, state: QuestState, design: Any, plan_sha: str,
                                      req: dict[str, Any]) -> tuple[Any, str]:
        """What the topic asks for and the plan lacks is asked of the plan's model, in place of asking a person: the same
        bounded request as the other missing parts (:meth:`_ask_plan_to_complete`, counted in ``.fi/topic_asked.json``).
        Only what was asked is kept from the rewrite. Returns the design to use and its plan's hash."""
        if (not self._runs_code(state) or self._client is None or _frozen.load(self.quest_root) is not None
                or not plan_sha or _topic_req.is_empty(req)):
            return design, plan_sha
        current: dict[str, list[dict[str, Any]]] = {"gaps": []}

        async def lacks(now: Any) -> list[str]:
            current["gaps"] = (await self._topic_gaps(req, now))[0]
            return [g["sentence"] for g in current["gaps"]]

        def request(now: Any, _text: str) -> str:
            return _topic_req.request(self.config.topic, current["gaps"], search=_optim.has_block(now), req=req)

        return await self._ask_plan_to_complete(
            design, plan_sha, lacks=lacks, request=request,
            keep=lambda before, parts: self._keep_only_what_the_topic_asked(before, parts, current["gaps"]),
            file=_TOPIC_ASKED, what="what the topic asks for", short="what the topic asks for",
            note="what the topic asks for and the plan lacked, written by the plan's model")

    async def _ask_plan_for_the_answers(self, design: Any, plan_sha: str, req: dict[str, Any]) -> tuple[Any, str]:
        """The main quantities of the topic that the plan's own map (``protocol.topic_map``) does not yet place are asked of the
        plan's model in one question (:func:`core.contract_parts.ask_parts`, the ``answers`` part, at most ``MAX_ASKS`` asks):
        which result answers each, and at which value of a setting where the topic asks for it at one. The answers block of the
        paper is built from this map. A quantity the plan cannot map stays unmapped (the block says so). Never a stop. Returns
        the design to use and its plan's hash."""
        protocol = design.get("protocol") if isinstance(design, dict) else None
        asked = _topic_req.quantities_to_place(req, protocol) if isinstance(protocol, dict) else {}
        if (not asked or self._client is None or _frozen.load(self.quest_root) is not None
                or not plan_sha):
            return design, plan_sha

        async def ask(prompt: str) -> str:
            return await self._chat(prompt, node="plan_contract")

        try:
            out = await _contract_parts.ask_parts(design, ask, lambda t: _parse_json_lenient(t, node="plan_contract"),
                                                  self._prompts, only=("answers",), topic_quantities=asked)
        except Exception:  # noqa: BLE001 -- the plan goes on with its map as it was
            self._log.warning("[plan] the plan could not be asked which results answer the topic's quantities", exc_info=True)
            return design, plan_sha
        part = out.parts.get("answers") or {}
        self._log.info("[plan] results answering the topic's quantities: %s after %s ask(s)",
                       "ok" if part.get("valid") else "left out", part.get("asks", 0))
        try:
            (self.fi_dir / "topic_answers.json").write_text(json.dumps(out.parts, indent=1, default=str) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[plan] couldn't write .fi/topic_answers.json: %r", e)
        new_map = out.design["protocol"].get("topic_map") if part.get("valid") else None
        if new_map is None or new_map == protocol.get("topic_map"):
            return design, plan_sha
        path = _plan.plan_path(self.quest_root)
        text = path.read_text(encoding="utf-8")
        edited = _plan.edit_design_block(text, lambda block: {**block, "protocol": {**(block.get("protocol") or {}),
                                                                                    "topic_map": new_map}})
        if edited is None or _plan.parse(edited).design is None:
            return design, plan_sha
        path.write_text(edited, encoding="utf-8")
        _plan.record_version(self.quest_root, edited, by="engine", note="which result answers each quantity the topic asks for")
        again, sha = self._design_from_plan()
        return (again, sha) if again is not None and sha else (design, plan_sha)

    async def _cover_the_topic_in_the_plan(self, state: QuestState) -> None:
        """At the plan step: read what the topic asks for (:meth:`_read_topic_requirements`), look for each in the plan, ask the
        plan's model for what is missing (:meth:`_ask_plan_for_the_topic`), and say in plan.md, in plain words, what the
        topic asks for and whether the plan has it. Once per plan: a resume does not do it again. It stops nothing: what
        is still missing is decided at the design step, on the design that will run (:meth:`_hold_the_design_to_the_topic`)."""
        if (not self._runs_code(state) or self._client is None or _frozen.load(self.quest_root) is not None
                or self._topic_covered_read().get("plan_done")):
            return
        design, sha = self._design_from_plan()
        path = _plan.plan_path(self.quest_root)
        if design is None or not sha or not path.is_file():
            return
        req = await self._read_topic_requirements()
        if req.get("error"):
            # Never a stop, and plan.md is not touched (a plan a person reads changes only for what is in it): the record,
            # run.log and the paper's limitations say the plan was not held against the topic.
            print("[FI] the topic could not be read for what it asks for; the plan was not held against it (see .fi/run.log)")
            self._topic_covered_write(plan_done=True, unreadable=str(req["error"]))
            return
        if _topic_req.is_empty(req):
            self._topic_covered_write(plan_done=True)
            return
        design, sha = await self._ask_plan_for_the_topic(state, design, sha, req)
        design, sha = await self._ask_plan_for_the_answers(design, sha, req)
        gaps, unjudged = await self._topic_gaps(req, design)
        text = path.read_text(encoding="utf-8")
        lines = _topic_req.plan_lines(req, gaps, reviewer=str(req.get("reviewer") or ""), same_model=req.get("same_model"),
                                      unjudged=unjudged, topic=str(state.get("topic") or self.config.topic or ""))
        asked, calls = self._search_record(sha, _TOPIC_ASKED)
        if asked:
            lines += ["", "- FI asked the plan's model to add what was missing" + (
                "; it is still missing" if gaps else "; it is in the plan now") + "."]
        if gaps:
            lines += ["", "- Not found in the plan (the study runs, and the paper says FI's check did not find it):",
                      *[f"  - {g['sentence']}" for g in gaps]]
        updated = _plan.add_to_section(text, _topic_req.HEADING, lines)
        path.write_text(updated, encoding="utf-8")
        _plan.record_version(self.quest_root, updated, by="engine", note="what the topic asks for, and whether the plan has it")
        if asked:  # the record names the plan it was made on: the plan as FI wrote its note on it
            self._record_search_asked(asked, calls, _plan.sha256(updated), "told in the plan", _TOPIC_ASKED)
        self._topic_covered_write(plan_done=True, left=self._gaps_to_say(gaps), unjudged=unjudged, asked=asked)

    async def _ask_design_for_the_topic(self, state: QuestState, design: dict[str, Any], req: dict[str, Any],
                                        gaps: list[dict[str, Any]], *, iteration: int) -> dict[str, Any]:
        """A design drafted at the design step (a redesign, not from plan.md) that lacks what the topic asks for: the design
        step's model is asked to add it, at most ``_SEARCH_ASKS`` times per design round (one chat call each, counted in
        ``.fi/topic_design_asked.json`` BEFORE the call, so a resume never asks again). An answer that cannot be read goes
        back to the model with the reader's exact reason. Only what was asked is kept (:func:`core.topic_requirements.merge`)."""
        path = self.fi_dir / _TOPIC_DESIGN_ASKED
        try:
            done = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            done = {}
        done = done if isinstance(done, dict) else {}
        asks = 0
        problem = ""
        schema = False  # the last answer was refused for its form: it is asked again, within _SCHEMA_RETRY_ASKS more tries
        if int(done.get("iteration", -1)) == iteration:
            asks = int(done.get("asks", 1) or 0)  # a record from before the count was kept is one ask
            problem = str(done.get("problem") or "")
            schema = bool(done.get("schema") and problem)

        def save(count: int) -> bool:
            try:
                self.fi_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"iteration": iteration, "asks": count, "problem": problem, "schema": schema,
                                            "at": _frozen.now()}) + "\n", encoding="utf-8")
                return True
            except OSError as e:
                self._log.warning("[design] FI could not write down that it was asking for what the topic asks, so it did not ask: %r", e)
                return False

        while asks < _SEARCH_ASKS + (_SCHEMA_RETRY_ASKS if schema else 0):
            if not save(asks + 1):  # counted BEFORE the call
                return design
            asks += 1
            self._log.warning("[design] the redesign lacks what the topic asks for (%s); FI asked the design step's model to add it (%d of %d)",
                              "; ".join(g["sentence"] for g in gaps)[:300], asks,
                              _SEARCH_ASKS + (_SCHEMA_RETRY_ASKS if schema else 0))
            self._progress("Asking the model to add what the topic asks for to the design")
            ask = (_topic_req.request(self.config.topic, gaps, search=_optim.has_block(design), req=req)
                   + "\n\nThe design drafted so far:\n" + json.dumps(design, ensure_ascii=False, default=str)[:8000]
                   + "\n\nAnswer with the whole design as one JSON object, with what was asked added.")
            if problem:
                ask += ("\n\nYour last answer was not used. The reason, exactly:\n" + problem
                        + "\nFix that, and keep everything the design already has.")
            try:
                reply = await self._chat(ask, node="design")
                obj = _parse_json_lenient(reply)
            except _ModelAnswerProblem:
                raise
            except ProviderUnusable:
                save(asks - 1)  # never asked (a usage limit, a dead login): the ask is given back
                raise
            except Exception as e:  # noqa: BLE001 -- what is still missing is decided after this
                self._log.warning("[design] the design step could not be asked to add what the topic asks for: %s", e)
                problem = " ".join(f"the call to the model failed ({str(e)[:300] or type(e).__name__})".split())
                schema = False  # no answer came: nothing for the form to be fixed in
                save(asks)
                continue
            merged = _topic_req.merge(design, obj, gaps) if isinstance(obj, dict) else None
            for moved in _topic_req.moved_holds(design.get("protocol"), (merged or {}).get("protocol")):
                self._log.warning("[design] what the topic asks for changed a part of the study: %s", moved)
            before = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
            fixed, refused = (_plan.accept_answer(before, merged.get("protocol")) if merged is not None
                              else (None, [_topic_req.restated(obj, gaps) or "nothing came back"]))
            why = " ".join(refused)
            if merged is not None and fixed is not None and refused:
                # Taken in part: what can be read is kept; what the topic still lacks is asked again with the reason.
                design = {**merged, "protocol": fixed}
                gaps, _unjudged = await self._topic_gaps(req, design)
                self._log.warning("[design] the design step's answer to add what the topic asks for was used in part; "
                                  "not used: %s", why[:600])
                if not gaps:
                    problem, schema = "", False
                    save(asks)
                    return design
                problem = ("Part of your last answer was used. This part was not: " + " ".join(why.split()))[:1500]
                schema = True
                save(asks)
                continue
            if merged is not None and fixed is not None:
                problem, schema = "", False  # used: an older refusal must not widen the bound of a later request
                save(asks)
                return {**merged, "protocol": fixed}
            self._log.warning("[design] the design step's answer to add what the topic asks for could not be used: %s", why)
            earlier = problem
            problem = " ".join(str(why or "the answer had none of what was asked in a form that can be read").split())[:1500]
            schema = True  # the reader named what it could not read: that goes back for another try
            repeated = bool(earlier) and problem == earlier
            if repeated:
                schema = False  # kept so: a later pass of this round (or a resume) does not ask once more
            save(asks)  # the reason is kept with the count, so the stop can say why
            if repeated:
                self._log.warning("[design] the design step's model gave the same answer that could not be used twice, so FI "
                                  "stopped asking (%d request(s) made)", asks)
                break
        return design

    async def _hold_the_design_to_the_topic(self, state: QuestState, design: Any, plan_sha: str) -> tuple[Any, str]:
        """At the design step, on the design that will RUN (the frozen protocol once there is one, else the design with the
        protocol the gates will hold it to): every factor, quantity and formula the topic asks for is looked for again, so
        a redesign or an amendment that drops a setting is caught, and a design a person edited is held to the topic too.
        What is missing is asked of the model once more (the plan's, within its bound; a redesign's, once per round). What
        is still missing stops the quest before anything runs when it is what the question is about; the rest is kept
        for the paper's limitations. Returns the design to run and its plan's hash."""
        req = self._topic_requirements_read()
        if (not self._runs_code(state) or not isinstance(design, dict) or req is None or req.get("error")
                or _topic_req.is_empty(req)):
            return design, plan_sha
        frozen = _frozen.load(self.quest_root) is not None

        def running(now: dict[str, Any]) -> Any:
            return self._protocol_block({**state, "design": now}, adopted=False)

        gaps, unjudged = await self._topic_gaps(req, design, running(design))
        if gaps and not frozen:
            if plan_sha:
                design, plan_sha = await self._ask_plan_for_the_topic(state, design, plan_sha, req)
            elif self._client is not None:
                design = await self._ask_design_for_the_topic(state, design, req, gaps,
                                                              iteration=int(state.get("iteration", 0) or 0))
            gaps, unjudged = await self._topic_gaps(req, design, running(design))
        self._topic_covered_write(left=self._gaps_to_say(gaps), unjudged=unjudged, checked_at_design=True)
        if gaps:
            # Never a stop: FI's reading of a plan against a topic misses a setting held under another name or form often
            # enough (in 8 recorded stops, most of what it called missing was there) that what it found is said, not acted
            # on. The paper says FI's check did not find it (core.paper_facts), not that the study did not do it.
            self._log.warning("[design] FI's check did not find in the design that will run all the topic asks for; the "
                              "run goes on and the paper says so: %s", "; ".join(g["sentence"] for g in gaps)[:400])
            print("[FI] the plan may not cover all the topic asks for; the study runs, and the paper says what FI's "
                  "check did not find")
        return design, plan_sha

    def _unread_parts(self, design: Any) -> dict[str, str]:
        """``{part: the sentence of plan.md that says it could not be read}`` for each protocol part that changes what is
        measured (:data:`core.plan.ASKED_PARTS`: the settings to sweep, the metrics, the runs per setting, the precision)
        that the plan's own notes say could not be read and that the design block still lacks. Only structured parts and
        the notes FI wrote about them: no prose of the plan is read. A search for the best design has no separate sweep."""
        path = _plan.plan_path(self.quest_root)
        if not isinstance(design, dict) or not path.is_file():
            return {}
        protocol = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
        found: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if _plan.ASK_MARK not in line:
                continue
            for part in _plan.ASKED_PARTS:
                if f"`protocol.{part}`" in line and part not in protocol and part not in found:
                    if part == "grid" and _optim.study_type_of(design) == "find_best_design":
                        continue
                    found[part] = line.strip().lstrip("-").strip()
        return found

    def _scan_left_out(self, design: Any) -> list[str]:
        """The notes FI wrote into plan.md when its reading of the search block left out a part of the first scan
        (``grid``, ``sub_studies``, ``in_each_run``: :func:`core.optimisation_plan.repair`) that the block still lacks, or
        one setting of the scan that the scan still lacks, each with the reader's exact reason; empty for a design with no
        search block. Only FI's own notes are read."""
        path = _plan.plan_path(self.quest_root)
        if not _optim.has_block(design) or not path.is_file():
            return []
        block = design["protocol"]["optimisation"]
        scanned = {str(k) for k in block.get("grid") or {}} if isinstance(block.get("grid"), dict) else set()
        found: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            note = line.strip().lstrip("-").strip()
            if note in found:
                continue
            if any(f"`protocol.optimisation.{part}` {_optim.LEFT_OUT_UNREAD}" in line
                   and block.get(part) is None for part in ("grid", "sub_studies", "in_each_run")):
                found.append(note)
                continue
            # One setting of the scan left out, the rest kept: asked back while the scan still lacks it.
            setting = _optim.scan_setting_left_out(note)
            if setting is not None and setting not in scanned:
                found.append(note)
        return found

    def _keep_only_the_parts(self, before_text: str, answer: dict[str, Any], parts: list[str], passages: str = "") -> str | None:
        """``before_text`` with only the asked ``parts`` of the protocol taken from ``answer`` (the model's JSON, shaped like
        the design block; each part only when it can be read as the plan reads it); everything else, a check, a threshold, a tolerance, is the plan's own as it was.
        ``None`` when none of the asked parts came back usable.

        The plan's ledger parts (``simplifications``, ``definitions``, the step cards ``steps``) are held to their own readers
        (:mod:`core.plan_ledger`, :mod:`core.plan_steps`, and a quoted definition against ``passages``) and not only to
        :func:`core.plan.normalize_protocol`, which does not check them: one that does not pass is not taken, and its exact
        reasons go back to the model with the next request."""
        old, new = _plan.raw_design_block(before_text), answer
        if not isinstance(old, dict) or not isinstance(new, dict):
            return None
        old_protocol = old.get("protocol") if isinstance(old.get("protocol"), dict) else {}
        new_protocol = new.get("protocol") if isinstance(new.get("protocol"), dict) else {}
        merged = dict(old_protocol)
        taken: list[str] = []
        reasons: list[str] = []
        for part in parts:
            if part == "per_trial":
                # Only `per_trial` of the checks asked about is taken (core/mc_error.py::unsaid_reading): never a
                # tolerance, an expected value or anything else of a check. The checks asked about are those of the
                # plan as it was (`old_protocol`), never those a part merged from this answer (new metrics) would add.
                names = set(_mc_error.unsaid_reading(old_protocol))
                said = new_protocol.get("oracles") if isinstance(new_protocol.get("oracles"), list) else (
                    new.get("oracles") if isinstance(new.get("oracles"), list) else [])
                answers = {str(e.get("name") or "").strip(): _mc_error.per_trial_flag(e) for e in said
                           if isinstance(e, dict)}
                answers = {k: v for k, v in answers.items() if k in names and v is not None}
                if not answers:
                    reasons.append("no check named in the question came back with `per_trial` true or false")
                    continue
                # A check asked about takes its answer and is marked answered; one the model left out loses the reading it
                # volunteered while writing the plan (it was not an answer to this question).
                candidate = {**merged, "oracles": [
                    {**o, "per_trial": answers[str(o.get("name") or "").strip()], _mc_error.ASKED_KEY: True}
                    if isinstance(o, dict) and str(o.get("name") or "").strip() in answers else
                    {k: v for k, v in o.items() if k != "per_trial"}
                    if isinstance(o, dict) and str(o.get("name") or "").strip() in names else o
                    for o in merged.get("oracles") or []]}
                fixed, why = _plan.normalize_protocol(candidate)
                if fixed is None:
                    reasons.append(f"the checks with `per_trial` as you wrote it cannot be read: {why}")
                    continue
                merged["oracles"] = candidate["oracles"]
                taken.append(part)
                continue
            if part == "steps":
                # The step cards sit inside the model block: only the steps are taken (from `protocol.model.steps`, or
                # `protocol.steps` written beside it), the rest of the model is the plan's.
                new_model = new_protocol.get("model")
                steps = new_model.get("steps") if isinstance(new_model, dict) else None
                if steps is None:
                    steps = new_protocol.get("steps")
                if steps is None:
                    reasons.append("the step cards are missing: write them as `protocol.model.steps`")
                    continue
                old_model = merged.get("model") if isinstance(merged.get("model"), dict) else {}
                candidate = {**merged, "model": {**old_model, "steps": steps}}
                fixed, why = _plan.normalize_protocol(candidate)
                if fixed is None:
                    self._log.warning("[plan] the plan's model wrote the step cards again but they still cannot be read: %s", why)
                    reasons.append(f"`protocol.model.steps` as you wrote it cannot be read: {why}")
                    continue
                problems = _plan_steps.parse(fixed)[1]
                if problems:
                    self._log.warning("[plan] the step cards still have problems: %s", "; ".join(problems))
                    reasons.append("`protocol.model.steps` as you wrote it is not complete: " + "; ".join(problems))
                    continue
                merged["model"] = fixed["model"]
                taken.append(part)
                continue
            if new_protocol.get(part) is None:
                continue
            if part == "criteria" and isinstance(new_protocol[part], list) and len(new_protocol[part]) > _criteria.MAX:
                # More than the limit is not a list that cannot be read (the plan draft keeps the first five in the same
                # way, core.criteria.repair): the first five in the model's own order are kept, the rest are named.
                extra = new_protocol[part][_criteria.MAX:]
                new_protocol = {**new_protocol, part: new_protocol[part][:_criteria.MAX]}
                self._log.warning("[criteria] the plan's model wrote %d criteria; the first %d, in its order, are kept and "
                                  "%d left out: %s", len(extra) + _criteria.MAX, _criteria.MAX, len(extra),
                                  ", ".join(repr(c.get("name")) if isinstance(c, dict) else repr(c) for c in extra))
            moved: list[str] = []
            if part == "in_each_run" and isinstance(new_protocol[part], dict) and isinstance(merged.get("grid"), dict):
                # A factor every run goes through together is not also a setting run on its own: it leaves the grid (the
                # request says so), or each run would be repeated once per value.
                moved = [str(k) for k in new_protocol[part] if str(k) in merged["grid"]]
            if moved:
                kept_grid = {k: v for k, v in merged["grid"].items() if k not in moved}
                left = _plan.sub_studies_without(merged.get("sub_studies"), moved) if merged.get("sub_studies") else []
                candidate = {**{k: v for k, v in merged.items() if k not in ("grid", "sub_studies")},
                             **({"grid": kept_grid} if kept_grid else {}), **({"sub_studies": left} if left else {}),
                             part: new_protocol[part]}
            elif part == "grid":
                # a rewritten grid replaces the old division into sub-studies (the answer's own comes next, as its own part)
                candidate = {**{k: v for k, v in merged.items() if k != "sub_studies"}, part: new_protocol[part]}
            else:
                candidate = {**merged, part: new_protocol[part]}
            fixed, fault = _plan.protocol_fault(candidate)
            why = fault.reason if fault is not None else None
            if (fixed is None or part not in fixed) and part == "sub_studies" and isinstance(new_protocol[part], list):
                # One entry that cannot be read does not cost the other parts of the study their answer: the entries that
                # read are tried together (the rule that every value of the grid is run is still asked), and the reason
                # for each entry left out goes back to the model.
                entries, entry_reasons = _plan.readable_sub_studies(new_protocol[part], candidate.get("grid"),
                                                                    candidate.get("in_each_run"))
                if entries and entry_reasons:
                    again, _again_why = _plan.normalize_protocol({**candidate, part: entries})
                    if again is not None and part in again:
                        fixed, why = again, None
                        reasons.extend(entry_reasons)
                        self._log.warning("[plan] `protocol.sub_studies`: %d entr%s of the answer could not be read and "
                                          "%s left out: %s", len(entry_reasons), "y" if len(entry_reasons) == 1 else "ies",
                                          "was" if len(entry_reasons) == 1 else "were", "; ".join(entry_reasons)[:600])
            if fixed is None or part not in fixed:
                self._log.warning("[plan] the plan's model wrote `protocol.%s` again but it still cannot be read: %s", part, why)
                entry = _plan.offending_entry({part: new_protocol[part]}, fault) if fault is not None else ""
                reasons.append(f"`protocol.{part}` as you wrote it cannot be read: {why}"
                               + (f" The entry as you wrote it: {entry}" if entry else ""))
                continue
            if part == "oracles" and getattr(self, "_test_run_returned", None):
                # After the test run of the checks the code is not rewritten: a check may not start to read a name nothing
                # defines (the plan does not declare it and the simulation did not return it), or it could never be judged.
                # The reason goes back to the plan's model, so it writes the measure from names that exist.
                unknown = _contract.unknown_measure_names(old_protocol, fixed, self._test_run_returned)
                if unknown:
                    ids = ", ".join(f"`{i}`" for i in sorted(self._test_run_returned))
                    reasons.append("; ".join(
                        f"the check `{n}` measures {', '.join('`' + w + '`' for w in ws)}, which the plan does not declare "
                        "and the simulation did not return" for n, ws in unknown.items())
                        + f". The simulation returned: {ids}. Write each measure as a formula of those names, or leave the "
                        "check as it was")
                    self._log.warning("[plan] `protocol.oracles` reads names nothing defines: %s", "; ".join(
                        f"{n}: {', '.join(ws)}" for n, ws in unknown.items()))
                    continue
            if part in _LEDGER_PARTS:
                problems = self._ledger_problems(fixed, passages).get(part)
                if problems:
                    self._log.warning("[plan] `protocol.%s` still has problems: %s", part, problems)
                    reasons.append(f"`protocol.{part}` as you wrote it is not complete: {problems}")
                    continue
            merged[part] = fixed[part]
            if part == "grid":
                merged.pop("sub_studies", None)
            if moved:
                if fixed.get("grid"):
                    merged["grid"] = fixed["grid"]
                else:
                    merged.pop("grid", None)
                if fixed.get("sub_studies"):
                    merged["sub_studies"] = fixed["sub_studies"]
                else:
                    merged.pop("sub_studies", None)
                self._log.warning("[plan] %s now go%s through every run together (`protocol.in_each_run`), so FI took "
                                  "%s out of the settings run one by one (`protocol.grid`)", ", ".join(moved),
                                  "es" if len(moved) == 1 else "", "it" if len(moved) == 1 else "them")
            taken.append(part)
        # A rewrite of the settings is taken whole or not at all: the grid and the division of it go together, and no setting
        # the plan compared is dropped (core.plan.lost_settings); the reasons go back to the model with the next request.
        settings_parts = ("grid", "sub_studies", "in_each_run", "constants")
        lost = _plan.lost_settings(old_protocol, merged) if any(p in taken for p in settings_parts) else []
        split = ("grid" in taken and isinstance(new_protocol.get("sub_studies"), list) and "sub_studies" not in taken)
        if lost or split:
            if lost:
                reasons.append(_plan.lost_settings_reason(lost))
            elif split:
                reasons.append("`protocol.grid` and `protocol.sub_studies` are taken together, and `protocol.sub_studies` was not taken")
            for part in settings_parts:
                if part in old_protocol:
                    merged[part] = old_protocol[part]
                else:
                    merged.pop(part, None)
            taken = [p for p in taken if p not in settings_parts]
        if not taken:
            if reasons:  # the reader's exact reason goes back to the model with the next request
                raise ValueError(" ".join(reasons))
            return None
        self._answer_refused = reasons  # parts taken beside parts not: the reasons still go back
        edited = _plan.edit_design_block(before_text, lambda design: {**design, "protocol": merged})
        if edited is None or _plan.parse(edited).design is None:
            return None
        return _plan.refresh_model_section(edited)

    def _literature_passages(self, state: QuestState) -> str:
        """The text of every source this quest retrieved (the passages the plan was shown), as one string, for checking that a
        quoted definition is really in the sources. Empty when nothing was retrieved or the text cannot be had."""
        try:
            return "\n".join(_item_content(item) for _label, _meta, item in _labelled_sources(state.get("literature") or []))
        except Exception as e:  # noqa: BLE001 -- a quote that cannot be looked for is not judged; never a stop
            self._log.warning("[plan] the sources' text could not be gathered to check the quoted definitions: %r", e)
            return ""

    def _ledger_problems(self, protocol: Any, passages: str = "") -> dict[str, str]:
        """``{part: what is wrong with it}`` for the structure of the plan's ledger parts: ``simplifications`` and
        ``definitions`` (:mod:`core.plan_ledger`) and ``model.steps``, named ``steps`` (:mod:`core.plan_steps`), when they are
        written but cannot be used, or when a metric has no definition. A definition's quote that is not in ``passages`` (the
        sources FI found) is one more problem; with no passages nothing is judged. A definition marked as the plan's own (its
        ``differs`` names a simplification) needs no source. Nothing here stops a quest."""
        if not isinstance(protocol, dict):
            return {}
        found: dict[str, str] = {}
        try:
            _sims, p_sims = _ledger.parse_simplifications(protocol)
            defs, p_defs = _ledger.parse_definitions(protocol)
            own = [d["metric"] for d in defs if d.get("differs")]
            p_defs = [p for p in p_defs if not any(p.startswith(f"definition not taken from a source: {m} ") for m in own)]
            if passages.strip():
                p_defs += [f"the quote for {m} is not in the sources FI found; quote the source word for word, or leave "
                           "`quote` empty and mark the definition as your own in `simplifications` (put that entry's id in "
                           "`differs`)" for m in _ledger.unfound_quotes(defs, passages)]
            _cards, p_steps = _plan_steps.parse(protocol)
        except Exception as e:  # noqa: BLE001 -- a reader that fails leaves the plan as it was
            self._log.warning("[plan] the plan's list of simplifications and definitions could not be read: %r", e)
            return {}
        for part, problems in (("simplifications", p_sims), ("definitions", p_defs), ("steps", p_steps)):
            if problems:
                found[part] = "; ".join(problems)[:1200].rstrip()
        return found

    def _ledger_uncovered(self, design: Any, *, said: bool = False) -> list[str]:
        """Plain sentences for what the topic asks for that the plan neither covers nor names in ``protocol.simplifications``
        (:func:`core.plan_ledger.uncovered`, from the kept reading of the topic). Empty when the topic was not read. ``said``: for
        the paper, leaving out what the reader called covered; what the plan's model is asked is not cut by readings that come
        later, so a resume asks the same again (and no more)."""
        try:
            req = self._topic_requirements_read()
            if not isinstance(req, dict) or req.get("error") or not isinstance(design, dict):
                return []
            protocol = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
            return [f"the topic asks for {g['name']}, which the plan neither includes nor names in `simplifications`"
                    for g in _ledger.uncovered(req, protocol, design=design,
                                           history=_topic_req.verdict_history(self._topic_covered_read()) if said else None)]
        except Exception as e:  # noqa: BLE001 -- never a stop
            self._log.warning("[plan] what the topic asks for could not be held against the simplifications: %r", e)
            return []

    def _ledger_unread(self, design: Any, passages: str = "") -> dict[str, str]:
        """The parts the plan's ledger asks for that are wrong or incomplete (:meth:`_ledger_problems`), with the topic items the
        plan neither covers nor names in ``simplifications`` (:meth:`_ledger_uncovered`) added to the ``simplifications`` part. An
        old plan with none of these keys, no metric and nothing uncovered has no problem. Never a stop by itself: the topic's
        own stop rules are unchanged."""
        protocol = design.get("protocol") if isinstance(design, dict) and isinstance(design.get("protocol"), dict) else None
        if protocol is None:
            return {}
        found = self._ledger_problems(protocol, passages)
        uncovered = self._ledger_uncovered(design)
        if uncovered:
            # Cut at its limit without a trailing blank: plan.md holds the line without one, and a resume looks for it there
            # (one cut right after a blank was written again at every resume, changing plan.md and restarting its counts).
            found["simplifications"] = "; ".join(filter(None, [found.get("simplifications"), *uncovered]))[:1600].rstrip()
        return found

    @staticmethod
    def _reading_unread(design: Any) -> dict[str, str]:
        """``{"per_trial": why}`` when a known-answer rule uses a number the protocol declares as an average over trials and
        does not say whether it holds in every single trial or only on average (:func:`core.mc_error.unsaid_reading`):
        asked of the plan's model once, never guessed by FI and never a stop."""
        protocol = design.get("protocol") if isinstance(design, dict) and isinstance(design.get("protocol"), dict) else None
        names = _mc_error.unsaid_reading(protocol)
        if not names:
            return {}
        return {"per_trial": (f"the checks {', '.join(repr(n) for n in names)} run on many random trials, and the plan "
                              "has not been asked whether each holds in every single trial or only on average over "
                              "trials")}

    @staticmethod
    def _own_definitions(protocol: Any) -> list[str]:
        """The metrics whose definition is the plan's own: marked so (``differs``), or with no source or no quote."""
        try:
            defs, _problems = _ledger.parse_definitions(protocol)
        except Exception:  # noqa: BLE001
            return []
        return sorted({d["metric"] for d in defs if d.get("metric") and (d.get("differs") or not (d.get("source") and d.get("quote")))})

    def _ledger_seen(self, iteration: int) -> dict[str, str]:
        """``{part: hash of the problem text it was last asked about}`` for this design round and the model asked now
        (``.fi/ledger_asked.json``): the plan's ledger parts are asked once per round, whatever FI's own notes did to plan.md's
        hash; a plan written anew (a revised plan, another round, another model) is asked afresh."""
        try:
            record = json.loads((self.fi_dir / _LEDGER_ASKED).read_text(encoding="utf-8"))
            if (isinstance(record, dict) and record.get("model") == self._plan_model_name()
                    and int(record.get("iteration") or 0) == iteration and isinstance(record.get("seen"), dict)):
                return {str(k): str(v) for k, v in record["seen"].items()}
        except (OSError, ValueError, TypeError):  # an "iteration" that is not a number: the record is unreadable
            return {}
        return {}

    def _ledger_seen_write(self, iteration: int, seen: dict[str, str]) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / _LEDGER_ASKED).write_text(json.dumps(
                {"iteration": iteration, "model": self._plan_model_name(), "seen": seen, "at": _frozen.now()}) + "\n",
                encoding="utf-8")
        except OSError as e:
            self._log.warning("[plan] couldn't record which ledger parts FI asked about: %r", e)

    async def _ask_plan_for_the_parts(self, state: QuestState, design: Any, plan_sha: str) -> tuple[Any, str]:
        """A protocol part that could not be read (the settings to sweep, the metrics, ...) is asked of the plan's model, in
        place of asking a person to write it: the same bounded request as the search block
        (:meth:`_ask_plan_to_complete`, counted in ``.fi/protocol_part_asked.json``), quoting what could not be read and
        the rule. Only the asked parts are kept from the rewrite. plan.md says what was done.

        The plan's ledger parts (``simplifications``, ``definitions``, the step cards, and the topic items the plan neither
        covers nor names: :meth:`_ledger_unread`) ride in the same request, each asked once per design round
        (``.fi/ledger_asked.json``; a request made of nothing else is one request, plus the retries a refusal for its form
        earns). A gap that survives never stops the quest: plan.md says it under *What the topic asks for*, and the paper's
        limitations say it (:func:`core.topic_requirements.write_note`)."""
        if (not self._runs_code(state) or self._client is None or _frozen.load(self.quest_root) is not None
                or not plan_sha):
            # Nobody is asked here: what each run was to compare inside it and could not be read is still not silently
            # dropped (repair_protocol has taken it out of the protocol already).
            if self._runs_code(state) and _frozen.load(self.quest_root) is None:
                self._in_each_run_not_done(self._unread_parts(design).get("in_each_run"), asked_again=False)
            return design, plan_sha
        passages = self._literature_passages(state)
        iteration = int(state.get("iteration", 0) or 0)

        def unread(now: Any) -> dict[str, str]:
            # How a rule holds is asked only in the first design round, before anything ran: later, the model would
            # answer having seen results, and `per_trial: false` could loosen a failing rule.
            return {**self._unread_parts(now), **self._ledger_unread(now, passages),
                    **(self._reading_unread(now) if iteration == 0 else {})}

        def digest(text: str) -> str:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        before = self._parts_asked(plan_sha)
        seen = self._ledger_seen(iteration)
        current = unread(design)
        # A part asked about at the plan step is not asked about again at the design step of the same run; a ledger part is
        # asked once for the problem it has (a plan written anew has other problems and is asked afresh).
        asked = [p for p in current if p not in before and not (p in _ONCE_PARTS and seen.get(p) == digest(current[p]))]
        if asked:
            ledger_asked = [p for p in asked if p in _ONCE_PARTS]
            if ledger_asked:  # recorded BEFORE the request: a killed run does not ask again
                self._ledger_seen_write(iteration, {**seen, **{p: digest(current[p]) for p in ledger_asked}})
            record_parts = {"parts": sorted({*before, *asked}), "iteration": iteration}

            def lacks(now: Any) -> list[str]:
                return [f"`protocol.{part}` ({'could not be read' if part in _plan.ASKED_PARTS else 'is incomplete'})"
                        for part in unread(now) if part in asked]

            def request(now: Any, _text: str) -> str:
                return _plan.part_request(state.get("topic") or self.config.topic,
                                          {k: v for k, v in unread(now).items() if k in asked})

            def keep(before: str, answer: dict[str, Any]) -> str:
                kept = self._keep_only_the_parts(before, answer, asked, passages)
                if kept is None:
                    raise ValueError("your answer has none of the asked parts in a form that can be read; plan.md is unchanged")
                return kept

            design, plan_sha = await self._ask_plan_to_complete(
                design, plan_sha, lacks=lacks, request=request, keep=keep, file=_PART_ASKED,
                what="settings the experiment needs", short="the protocol",
                note="the parts of the protocol that could not be read, written by the plan's model", extra=record_parts,
                # What each run compares inside it is asked again once, like the ledger parts and the per_trial
                # question: what is still unusable is said in the paper, never a stop.
                ask_limit=1 if all(p in _ONCE_PARTS or p == "in_each_run" for p in asked) else None)
            if ledger_asked:  # what the answer left is the problem it was asked about: not asked about again
                after = unread(design)
                self._ledger_seen_write(iteration, {**seen, **{p: digest(after[p]) for p in ledger_asked if p in after},
                                                    **{p: "" for p in ledger_asked if p not in after}})
        else:
            record_parts = {"parts": sorted(before), "iteration": iteration}
        path = _plan.plan_path(self.quest_root)
        if path.is_file():
            left = unread(design)
            lines = []
            for part in asked:
                if part == "per_trial":
                    lines.append("FI asked the plan's model whether each known-answer check holds in every "
                                 "single trial or only on average over trials, and "
                                 + ("a check is still unanswered (it is read as an average over trials)" if part in left else "every check is answered now") + ".")
                elif part in _LEDGER_PARTS:
                    lines.append(f"FI asked the plan's model to write `protocol.{part}` again, and "
                                 + ("it still has gaps (the experiment goes on; the paper says so)" if part in left
                                    else "it is complete now") + ".")
                else:
                    lines.append(f"FI asked the plan's model to write `protocol.{part}` again, and it is "
                                 + (f"still not readable ({'; the quest stops' if part == 'grid' else 'the experiment goes on without it'})"
                                    if part in left else "now in the plan") + ".")
            # What each run was meant to compare inside it and could not be used, even when asked again, is a requirement of
            # the study not done: kept for the writer (core/topic_requirements.py::write_note), so the paper says it.
            if "in_each_run" in {*asked, *before}:
                self._in_each_run_not_done(left.get("in_each_run"), asked_again=True)
            elif "in_each_run" not in left:
                self._note_not_done([], kind="in_each_run")  # a plan written anew that no longer lacks it
            # What is still wrong with the ledger parts is said under *What the topic asks for*, and kept for the writer.
            gaps = [f"`protocol.{part}` is incomplete: {why}" for part, why in left.items() if part in _LEDGER_PARTS]
            self._note_the_ledger_for_the_writer(design, gaps)
            text = path.read_text(encoding="utf-8")
            new_lines = [line for line in lines if line not in text]
            new_gaps = [g for g in gaps if g not in text]
            if new_lines or new_gaps:
                count, calls = self._search_record(_plan.sha256(text), _PART_ASKED)
                updated = text
                if new_lines:
                    updated = _plan.add_to_section(updated, _PART_HEADING, ["", *[f"- {line}" for line in new_lines]])
                if new_gaps:
                    updated = _plan.add_to_section(updated, _topic_req.HEADING, [
                        "", "- The plan did not give all of what it was asked for about its simplifications, definitions and "
                        "steps (the paper will say so in its limitations):", *[f"  - {g}" for g in new_gaps]])
                path.write_text(updated, encoding="utf-8")
                if count:  # the record names the plan it was made on: the plan as FI wrote its note on it
                    self._record_search_asked(count, calls, _plan.sha256(updated), "told in the plan", _PART_ASKED,
                                              record_parts)
                plan_sha = _plan.sha256(updated)
                _plan.record_version(self.quest_root, updated, by="engine", note="what FI did about protocol parts that could not be read")
        return design, plan_sha

    def _note_the_ledger_for_the_writer(self, design: Any, gaps: list[str]) -> None:
        """Keep, for the writer (:func:`core.topic_requirements.write_note`), what is still wrong with the plan's ledger parts
        (``ledger_left``) and which metrics are the study's own definitions, not a source's (``unsourced_definitions``)."""
        protocol = design.get("protocol") if isinstance(design, dict) else None
        own = self._own_definitions(protocol) if isinstance(protocol, dict) else []
        record = self._topic_covered_read()
        head = "`protocol.simplifications` is incomplete: "
        if isinstance(protocol, dict) and any(g.startswith(head) for g in gaps):
            # What the paper says of the simplifications leaves out the topic items the reader called covered (the plan has
            # them; saying it neither includes nor names them was false): the sentence is made again without those.
            now = "; ".join(filter(None, [self._ledger_problems(protocol).get("simplifications"),
                                          *self._ledger_uncovered(design, said=True)]))[:1600].rstrip()
            gaps = [(head + now if g.startswith(head) else g) for g in gaps if now or not g.startswith(head)]
        if gaps != (record.get("ledger_left") or []) or own != (record.get("unsourced_definitions") or []):
            self._topic_covered_write(ledger_left=gaps, unsourced_definitions=own)

    async def _ask_figures_of_an_edited_plan(self, state: QuestState, design: Any, plan_sha: str) -> tuple[Any, str]:
        """The design read from plan.md with the figures nothing declares asked about (a person added or renamed one), and the
        answer written into plan.md's design block, so a resume reads it and does not ask again."""
        before = design.get("protocol", {}).get("figure_declarations") if isinstance(design, dict) and isinstance(
            design.get("protocol"), dict) else None
        asked = await self._ask_figures_of_a_redesign(state, design)
        now = asked.get("protocol", {}).get("figure_declarations") if isinstance(asked, dict) and isinstance(
            asked.get("protocol"), dict) else None
        path = _plan.plan_path(self.quest_root)
        if not plan_sha or now == before or not now or not path.is_file():
            return asked, plan_sha

        def change(block: dict[str, Any]) -> dict[str, Any]:
            protocol = dict(block.get("protocol") or {})
            protocol["figure_declarations"] = now
            return {**block, "protocol": protocol}

        try:
            updated = _plan.edit_design_block(path.read_text(encoding="utf-8"), change)
            if updated is None:
                return asked, plan_sha
            path.write_text(updated, encoding="utf-8")
            _plan.record_version(self.quest_root, updated, by="engine", note="what the plan's model said each new figure shows")
            return asked, _plan.sha256(updated)
        except OSError as e:
            self._log.warning("[design] couldn't write the figure declarations into plan.md: %r", e)
            return asked, plan_sha

    def _figure_declarations_after_the_freeze(self, state: QuestState, frozen_protocol: Any) -> dict[str, Any]:
        """What the plan's model said about each figure, once the protocol is frozen: the declarations the frozen protocol
        holds, then those of the design in the state (a redesign after the freeze asked about its new files). They are kept in
        the design, never in the frozen record (:data:`core.frozen_protocol.UNFROZEN_KEYS`)."""
        out: dict[str, Any] = {}
        design = state.get("design") if isinstance(state, dict) else None
        for protocol in (frozen_protocol, design.get("protocol") if isinstance(design, dict) else None):
            said = protocol.get("figure_declarations") if isinstance(protocol, dict) else None
            if isinstance(said, dict):
                out.update({k: v for k, v in said.items() if isinstance(v, dict)})
        return out

    async def _ask_figures_of_a_redesign(self, state: QuestState, design: Any) -> Any:
        """``design`` of a redesign (or of a plan a person edited) with the figures it plans that no declaration covers asked
        about, in one question (:func:`core.contract_parts.ask_parts` with ``only=("figures",)``, at most ``MAX_ASKS``
        asks): what the plan's model said about a file the design still plans was inherited (:meth:`_inherit_plan_parts`, or,
        after the freeze, :meth:`_figure_declarations_after_the_freeze`), a new file is asked. Never a stop; a figure left
        undeclared is drawn by nobody and the contract's gaps say so.

        This runs after the freeze too, and only for figures: a figure is not a result, a setting or a check, the declarations
        are outside the frozen hash, and the frozen protocol itself (its results, settings and checks) is not touched here."""
        protocol = design.get("protocol") if isinstance(design, dict) else None
        frozen = _frozen.protocol_of(self.quest_root)
        if isinstance(protocol, dict) and frozen is not None:
            said = self._figure_declarations_after_the_freeze(state, frozen)
            said.update(protocol.get("figure_declarations") or {})  # the design's own, last
            if said:
                design = {**design, "protocol": {**protocol, "figure_declarations": said}}
                protocol = design["protocol"]
        if (not isinstance(protocol, dict) or protocol.get("form") != "contract" or self._client is None
                or not self._runs_code(state)
                or not self._fis_analysis_planned(design) or not _contract_parts.figure_faults(design)):
            return design

        async def ask(prompt: str) -> str:
            return await self._chat(prompt, node="plan_contract")

        # After the freeze no result can be added, so the numbers the plan only names are not offered to a figure (nor declared)
        named = protocol.get("named_results") if frozen is not None else None
        shown = {**design, "protocol": {k: v for k, v in protocol.items() if k != "named_results"}} if named else design
        try:
            out = await _contract_parts.ask_parts(shown, ask, lambda t: _parse_json_lenient(t, node="plan_contract"),
                                                  self._prompts, fi_analysis=True, only=("figures",))
            if named:
                out.design = {**out.design, "protocol": {**out.design["protocol"], "named_results": named}}
        except Exception:  # noqa: BLE001 -- the redesign goes on with its figures undeclared
            self._log.warning("[design] the figures of the redesign could not be asked about", exc_info=True)
            return design
        for note in out.notes:
            self._log.warning("[design] %s", note)
        self._log.info("[design] figures of the redesign: %s", "; ".join(
            f"{k} {'ok' if v['valid'] else 'left out'} after {v['asks']} ask(s)" for k, v in out.parts.items()) or "none asked")
        return out.design

    def _inherit_plan_parts(self, state: QuestState, design: Any, *, iteration: int) -> Any:
        """A redesign (a design drafted at this step, not read from ``plan.md``) keeps every part of the plan's protocol that it
        does not write itself: the runs per setting, the precision, the thresholds, the checks against known answers, the
        criteria, the seed policy. A redesign that leaves one out did not decide to drop it: a real redesign wrote only the
        grid, the metrics and the model, the frozen protocol then had no run count, and the engine ran one trial per setting
        where the plan said 2500. Nothing is asked of a model, so this is bounded and the same on a resume. A part is taken
        only when it reads as part of the redesign's protocol (:func:`core.plan.normalize_protocol`); the parts the redesign
        owns (settings to vary, metrics, the model behind the numbers, a search for the best design) are never taken here.
        Said in ``run.log`` and kept in ``.fi/drafted_parts_asked.json`` (``inherited``)."""
        if not isinstance(design, dict) or not self._runs_code(state) or _frozen.load(self.quest_root) is not None:
            return design
        base = design.get("protocol")
        if not isinstance(base, dict) or not base:
            return design  # no protocol of its own: the plan's is the one in use (:meth:`_draft_protocol`)
        planned, _why = _plan.load_design(self.quest_root)
        planned_protocol = planned.get("protocol") if isinstance(planned, dict) else None
        if not isinstance(planned_protocol, dict) or not planned_protocol:
            return {**design, "protocol": _contract_parts.settle_form(base, None)}
        protocol: dict[str, Any] = dict(base)
        taken: list[str] = []
        # The criteria go last: one that names a check is only kept with that check.
        for part, value in sorted(planned_protocol.items(), key=lambda kv: kv[0] == "criteria"):
            if part in _REDESIGN_OWNED_PARTS or part in protocol or value in (None, "", [], {}):
                continue
            if part == "oracles" and isinstance(value, list):
                value, left_out = self._oracles_of_this_model(value, protocol, planned_protocol)
                if left_out:
                    self._log.warning("[design] the redesign's model changed or left out the equation(s) these checks rest on, "
                                      "so FI did not keep them from the plan: %s", ", ".join(repr(n) for n in left_out))
                if not value:
                    continue
            elif part == "criteria" and isinstance(value, list):
                kept_names = {str(o.get("name") or "").strip().lower() for o in protocol.get("oracles") or [] if isinstance(o, dict)}
                dropped = [c for c in value if isinstance(c, dict) and str(c.get("oracle") or "").strip()
                           and str(c["oracle"]).strip().lower() not in kept_names]
                if dropped:
                    self._log.warning("[design] FI did not keep these criteria from the plan, because the check they read is not "
                                      "in the redesign: %s", ", ".join(repr(c.get("name")) for c in dropped))
                value = [c for c in value if c not in dropped]
                if not value:
                    continue
            fixed, _problem = _plan.normalize_protocol({**protocol, part: value})
            if fixed is not None and part in fixed:
                protocol[part] = fixed[part]
                taken.append(part)
        # `form` is FI's, worked out again from the redrafted results: `contract` only when every one has a validated
        # role (core/contract_parts.py::settle_form); nothing is asked, and a redesign after the freeze stays as frozen.
        protocol = _contract_parts.settle_form(protocol, planned_protocol)
        if not taken:
            return {**design, "protocol": protocol}
        self._log.warning("[design] the redesign's protocol leaves out %s, which the plan's protocol has; FI kept the plan's "
                          "(nothing is asked of a model for this)", ", ".join(f"`protocol.{p}`" for p in taken))
        path = self.fi_dir / _DRAFTED_PARTS_ASKED
        try:
            record = _read_json_or_none_path(path)
            record = dict(record) if isinstance(record, dict) else {}
            record.setdefault("iteration", iteration)
            record["inherited"] = taken
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[design] FI could not write down which protocol parts the redesign kept from the plan: %r", e)
        return {**design, "protocol": protocol}

    @staticmethod
    def _oracles_of_this_model(oracles: list[Any], protocol: dict[str, Any],
                               planned_protocol: dict[str, Any]) -> tuple[list[Any], list[str]]:
        """``(the checks, the names left out)``: a check of the plan's is kept for a redesign only when every equation it names
        (:func:`core.oracle_check.equation_ids_in`) is in the redesign's model with the same formula as in the plan's; a check
        that names none is kept. A check written for a model the redesign changed would be judged against another model."""
        def formulas(p: dict[str, Any]) -> dict[str, str]:
            return {str(e.get("id")).strip().upper(): " ".join(str(e.get("formula") or "").split())
                    for e in _review.equations_of(p)}

        have, was = formulas(protocol), formulas(planned_protocol)
        kept: list[Any] = []
        left_out: list[str] = []
        for o in oracles:
            needs = _oracle.equation_ids_in(o) if isinstance(o, dict) else set()
            if all(i in have and have[i] == was.get(i) for i in needs):
                kept.append(o)
            else:
                left_out.append(str(o.get("name") or "?"))
        return kept, left_out

    def _drafted_parts_record(self) -> tuple[int, str, int]:
        """``(requests, model, iteration)`` of the last time FI asked the plan's model to write the protocol parts a
        redesign drafted that could not be read (``.fi/drafted_parts_asked.json``); zeros when it never did."""
        try:
            record = json.loads((self.fi_dir / _DRAFTED_PARTS_ASKED).read_text(encoding="utf-8"))
            return max(0, int(record.get("count") or 0)), str(record.get("model") or ""), int(record.get("iteration") or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            return 0, "", 0

    def _drafted_parts_asked(self, iteration: int) -> int:
        """How many times FI has asked about a redesign's unreadable parts, on the model and design round asked now."""
        count, model, at = self._drafted_parts_record()
        return count if model == self._plan_model_name() and at == iteration else 0

    async def _complete_drafted_parts(self, state: QuestState, design: Any, *, iteration: int) -> Any:
        """A design drafted at this step (a redesign: it did not come from ``plan.md``) whose protocol lacks a part that
        plan.md says could not be read (the settings to sweep, the metrics ...) is completed here, the way the plan's own
        parts are (:meth:`_ask_plan_for_the_parts`), before the quest may stop for it. In this order: (1) the plan's model
        is asked to write just those parts for the redesign (at most ``_SEARCH_ASKS`` requests of one chat call each per
        design round and model, counted in ``.fi/drafted_parts_asked.json`` BEFORE each call so a resume never asks
        again; only a part that passes the same reading as the plan's is kept, the rest of the design is untouched);
        (2) a part still missing is taken from the plan's own design block when that block holds it readable, since the
        plan step already settled it. After the freeze the frozen protocol is the one that runs: nothing is asked."""
        if not isinstance(design, dict) or not self._runs_code(state) or _frozen.load(self.quest_root) is not None:
            return design
        unread = self._unread_parts(design)
        if not unread:
            self._forget_drafted_parts_substituted()  # this round's design reads in full: an earlier round's note is stale
            return design
        base = design["protocol"] if isinstance(design.get("protocol"), dict) else {}
        protocol: dict[str, Any] = dict(base)

        def take(part: str, value: Any) -> bool:
            fixed, why = _plan.normalize_protocol({**protocol, part: value})
            if fixed is None or part not in fixed:
                self._log.warning("[design] the redesign's `protocol.%s` still cannot be read: %s", part, why)
                return False
            protocol[part] = fixed[part]
            return True

        model = self._plan_model_name()
        while unread and self._client is not None and self._drafted_parts_asked(iteration) < _SEARCH_ASKS:
            count = self._drafted_parts_asked(iteration) + 1
            try:
                self.fi_dir.mkdir(parents=True, exist_ok=True)
                (self.fi_dir / _DRAFTED_PARTS_ASKED).write_text(json.dumps(
                    {"count": count, "model": model, "iteration": iteration, "parts": sorted(unread), "at": _frozen.now()}
                ) + "\n", encoding="utf-8")
            except OSError as e:
                self._log.warning("[design] FI could not write down that it was asking for the protocol parts, so it did "
                                  "not ask: %r", e)
                break
            self._log.warning("[design] the redesign lacks %s; FI asked the plan's model to write %s (%d of %d)",
                              ", ".join(f"`protocol.{p}`" for p in unread), "it" if len(unread) == 1 else "them",
                              count, _SEARCH_ASKS)
            self._progress("Asking the model to complete the settings of the redesign")
            ask = (_plan.part_request(state.get("topic") or self.config.topic, unread)
                   + "\n\nThe protocol drafted so far (keep what is usable in it):\n"
                   + json.dumps(base, ensure_ascii=False, default=str)[:4000]
                   + "\n\nAnswer with the asked parts alone, as one JSON object whose keys are their names "
                   + "(for example {\"grid\": {...}}).")
            try:
                obj = _parse_json_lenient(await self._chat(ask, node="plan_revise"))
            except Exception as e:  # noqa: BLE001 -- the next ask, the plan's own part, or the stop says what is missing
                self._log.warning("[design] the plan's model could not be asked for the redesign's parts: %s", e)
                continue
            if isinstance(obj, dict) and isinstance(obj.get("protocol"), dict):
                obj = obj["protocol"]
            if not isinstance(obj, dict):
                self._log.warning("[design] the plan's model's answer for the redesign's parts was not a JSON object")
                continue
            for part in list(unread):
                if obj.get(part) is not None and take(part, obj[part]):
                    unread.pop(part)
            if not unread:
                self._log.info("[design] the redesign's protocol parts are complete now (asked %d time(s))", count)
                print("[FI] the settings of the redesign are complete now; going on")
        used: list[str] = []
        if unread:
            planned, _sha = self._design_from_plan()
            planned_protocol = (planned.get("protocol") if isinstance(planned, dict)
                                and isinstance(planned.get("protocol"), dict) else {})
            for part in list(unread):
                if planned_protocol.get(part) is not None and take(part, planned_protocol[part]):
                    self._log.warning("[design] the redesign's `protocol.%s` could not be read, even when the plan's model was "
                                      "asked; FI used the one in plan.md, which was read and settled at the plan step", part)
                    unread.pop(part)
                    used.append(part)
        if used and not self._record_drafted_parts_substituted(used, iteration):
            # A substitution that was not written down would reach neither the quality report nor the writer: it does
            # not happen. The parts stay as the redesign had them (unread), and the quest stops or goes on as it does
            # for any part that could not be read.
            self._log.warning("[design] the plan's own %s was not used for the redesign, because FI could not write down "
                              "that it was", ", ".join(f"`protocol.{p}`" for p in used))
            for part in used:
                if part in base:
                    protocol[part] = base[part]
                else:
                    protocol.pop(part, None)
        return {**design, "protocol": protocol} if protocol != base else design

    def _record_drafted_parts_substituted(self, parts: list[str], iteration: int) -> bool:
        """Write down, next to the count of asks, which parts of a redesign's protocol FI took from the plan because the
        redesign's own could not be read: the quest then runs what the plan settled, not what the redesign meant. Read back
        by the quality report and by the note the writer receives (:meth:`_drafted_parts_note`), so the paper never describes
        the redesign's intended settings as what ran. Returns False (and says so in the log) when the record cannot be
        written: the caller then does not substitute."""
        path = self.fi_dir / _DRAFTED_PARTS_ASKED
        try:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {}
            record = record if isinstance(record, dict) else {}
            record.setdefault("iteration", iteration)
            record["substituted"] = sorted(set(parts))
            record["substituted_at"] = _frozen.now()
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[design] FI could not write down that it used the plan's own %s for the redesign: %r",
                              ", ".join(parts), e)
            return False
        return True

    def _forget_drafted_parts_substituted(self) -> None:
        """Drop the "used the plan's own" record when the design now running reads in full (it was an earlier round's)."""
        path = self.fi_dir / _DRAFTED_PARTS_ASKED
        record = _read_json_or_none_path(path)
        if not isinstance(record, dict) or "substituted" not in record:
            return
        record.pop("substituted", None)
        record.pop("substituted_at", None)
        try:
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[design] FI could not update its record of the redesign's parts: %r", e)

    def _drafted_parts_note(self) -> str:
        """For the paper: the parts of the design that ran are the plan's, not the redesign's, because the redesign's could
        not be read. Empty when FI substituted nothing."""
        record = _read_json_or_none_path(self.fi_dir / _DRAFTED_PARTS_ASKED)
        parts = record.get("substituted") if isinstance(record, dict) else None
        if not isinstance(parts, list) or not parts:
            return ""
        names = {"grid": "settings to vary", "in_each_run": "what each run compares inside it", "metrics": "metrics",
                 "runs_per_setting": "runs per setting"}
        listed = ", ".join(names.get(str(p), str(p).replace("_", " ")) for p in parts)
        return (f"The redesign's {listed} could not be read, even when the model was asked to write them again, so FI ran the "
                "plan's own instead. What ran is the plan's, not what the redesign meant: do NOT describe the redesign's "
                f"intended {listed} as what was run, and say so plainly (in the limitations).")

    def _stop_if_no_settings_to_vary(self, state: QuestState, design: Any) -> None:
        """A study whose plan tried to list settings to vary (``protocol.grid`` was drafted and could not be read, even when
        the plan's model was asked to write it again) must not run one setting and call it a sweep: stop plainly, with no
        paper, saying what was tried. A plan that never listed a sweep is a study with one setting and goes on. The other
        unreadable parts (the metrics ...) go on, said in plan.md."""
        if not self._runs_code(state) or _frozen.load(self.quest_root) is not None:
            return
        if "grid" not in self._unread_parts(design):
            return
        path = _plan.plan_path(self.quest_root)
        tried = self._search_record(_plan.sha256(path.read_text(encoding="utf-8")), _PART_ASKED)[0] if path.is_file() else 0
        tried = tried or self._drafted_parts_record()[0]  # a redesign is asked about its own parts (not keyed to plan.md)
        steps = [
            "Nothing was run. The plan lists settings to vary, but FI could not read them, so the experiment would have run "
            "one setting only and called it a study of several.",
            (f"FI asked the plan's model {tried} time(s) to write them again, and it still could not."
             if tried else "FI did not ask the plan's model to write them again (no model was available to ask, or the plan "
             "was already settled)."),
            "You do not need to write them yourself. What to do:",
            "1. Try another model for the plan step: give `provider.node_models.plan_revise` (or `provider.model`) another "
            f"model in the quest's config.yaml, then `python launch.py --update {self.quest_id}`.",
            f"2. Or ask for a change in words: `--resume {self.quest_id} --revise-plan \"<the settings to vary and their values>\"`.",
        ]
        self._log.warning("[design] the settings to sweep could not be read, even when the plan's model was asked (%d time(s)); "
                          "stopping, nothing was run", tried)
        print(f"[FI] quest {self.quest_id}: the settings to vary could not be read, even when the plan's model was asked; "
              "nothing was run (see NEXT_STEP.md)")
        self._pause_for_human(
            kind="plan",
            interaction="supply",
            headline="the settings to vary could not be read",
            steps=steps,
            recommended="Try another model for the plan step.",
            alternatives=[steps[-1][3:]],
            payload={"quest_id": self.quest_id, "plan_file": str(path), "settings_stage": True},
        )

    def _keep_only_the_search(self, before_text: str, answer: dict[str, Any]) -> str | None:
        """``before_text`` with only the search part of ``answer`` (the model's JSON, shaped like the design block: ``study_type`` and ``protocol.optimisation``)
        taken in, its sections shown again from the block; a plain ``protocol.grid`` beside the block is dropped. Every part
        of the plan's own block that could already be read is kept as the plan had it (a tolerance, a target, a margin, a
        range, a baseline): the model supplies only what was missing or could not be read, and a value it changed for a part
        the plan already had is put back. ``None`` when the rewrite has no search block or the result cannot be used."""
        old, new = _plan.raw_design_block(before_text), answer
        protocol = new.get("protocol") if isinstance(new, dict) else None
        block = protocol.get("optimisation") if isinstance(protocol, dict) else None
        if not isinstance(old, dict):
            return None
        fixes = protocol.get("fixes") if isinstance(protocol, dict) and isinstance(protocol.get("fixes"), dict) else None
        if not isinstance(block, dict) and fixes is None:
            raise ValueError("your answer has no `protocol.optimisation` block: write it as a mapping (`objective`, "
                             "`design_variables`, `baseline`, `evaluation_budget`) under `protocol` in the design block")
        if fixes is not None:
            # The answer to a request that named only the faulted paths: each value is placed at its path in the block the
            # plan's model wrote last time (kept whole beside the faults), and the block is then read as any other answer.
            refused = self._search_refused(before_text)
            if refused is None or not isinstance(refused.get("block"), dict):
                raise ValueError("there is no block with unreadable parts to correct; write the `protocol.optimisation` block")
            held_block = dict(refused["block"])
            if isinstance(held_block.get("constraints"), dict):  # one limit written as a mapping reads as a list of one
                held_block["constraints"] = [held_block["constraints"]]
            wrapped: Any = {"protocol": {"optimisation": held_block}}
            asked = [str(f.get("path")) for f in refused.get("faults") or [] if isinstance(f, dict)]
            placed = 0
            for path, value in fixes.items():
                # Only a path that was asked about (or a part inside one) is placed: nothing else of the block changes. A
                # path read only in part names no place, so nothing of it is written (never over the part around it).
                if not _reading.split(str(path)):
                    self._log.warning("[plan] the fix for `%s` was not used: FI cannot read that path in full", path)
                    continue
                if not any(str(path) == a or str(path).startswith((a + ".", a + "[")) for a in asked):
                    self._log.warning("[plan] the fix for `%s` was not used: it was not asked about", path)
                    continue
                at = _reading.put_at(wrapped, str(path), value)
                if at is None:
                    self._log.warning("[plan] the fix for `%s` was not used: the search block has no such place", path)
                    continue
                wrapped, placed = at, placed + 1
            if not placed:
                raise ValueError("none of the asked paths was written: give each as `{\"fixes\": {\"<path>\": <the part>}}` "
                                 "under `protocol`, with the path exactly as named")
            block = wrapped["protocol"]["optimisation"]
        self._answer_block = block  # kept as written, so a refusal sends it back as the base of the next request
        old_protocol = old.get("protocol") if isinstance(old.get("protocol"), dict) else {}
        old_block = old_protocol.get("optimisation")
        if isinstance(old_block, dict):
            _fixed, why = _optim.normalize(old_block)
            # Every part the reader names, not only the first: the model was asked for all of them at once.
            unreadable = {(str(_reading.split(f.path)[2]) if len(_reading.split(f.path)) > 2 else "")
                          for f in _optim.block_faults(old_block)[1]}
            if not isinstance(old_block.get("evaluation_budget"), dict):
                unreadable.add("evaluation_budget")
            if _optim.scan_problem(old, self._scan_limit_s()):
                unreadable.add("grid")  # a scan that cannot run is what the plan was asked to rewrite
            one_left_out = any(_optim.scan_setting_left_out(n) for n in self._scan_left_out(old))
            if one_left_out:
                # A setting of the scan FI's reading left out (the rest kept) is what the plan was asked for: the scan and
                # its parts are the answer's, with the conditions it brings under `fixed`; the settings of the scan the
                # plan already had are put back below.
                unreadable |= {"grid", "sub_studies"}
            if "" in unreadable:
                unreadable = set(old_block)  # the block as a whole could not be read: nothing of it is kept
            merged = dict(block)
            put_back: list[str] = []
            for key, value in old_block.items():
                if key in unreadable or value is None or value == "" or value == [] or value == {}:
                    continue
                if merged.get(key) != value:
                    if key in merged:
                        put_back.append(str(key).replace("_", " "))
                    merged[key] = value
            # A scan the answer writes anew (the plan had none that could be used) brings the conditions it is compared
            # across: each one the plan did not hold yet is added under `fixed`, at the value the answer holds it at, as
            # the reader requires; every value the plan already held stays the plan's own.
            new_grid, new_fixed = block.get("grid"), block.get("fixed")
            if (("grid" in unreadable or not old_block.get("grid")) and isinstance(new_grid, dict)
                    and isinstance(new_fixed, dict) and isinstance(merged.get("fixed"), dict)):
                added = {k: v for k, v in new_fixed.items() if k in new_grid and k not in merged["fixed"]}
                if added:
                    if {k: v for k, v in new_fixed.items() if k not in added} == merged["fixed"]:
                        put_back = [p for p in put_back if p != "fixed"]  # the answer only added: nothing was put back
                    merged["fixed"] = {**merged["fixed"], **added}
            if one_left_out and isinstance(old_block.get("grid"), dict) and isinstance(merged.get("grid"), dict):
                # The settings the plan's scan already had, as the plan had them: the answer only adds the one asked for.
                changed = [str(k) for k, v in old_block["grid"].items() if merged["grid"].get(k) != v]
                if changed:
                    put_back.append("the scan's " + ", ".join(f"`{k}`" for k in changed))
                merged["grid"] = {**merged["grid"], **old_block["grid"]}
            if put_back:
                self._log.warning("[plan] the plan's model also changed what the plan already said about %s; FI kept the "
                                  "plan's own", ", ".join(put_back))
            block = merged
        taken, left_out, stopping = _read_a_search_block(block)
        if taken is None:
            faults = _optim.block_faults(block)[1] or stopping  # every problem the block has, as the old reason did
            # the next request names only these paths; the plan, not written, keeps its own block (or none)
            self._keep_search_refused(block, faults, before_text)
            entry = _entries_as_written({"optimisation": block}, faults, you=True)
            raise ValueError("the block you wrote cannot be read: " + _optim.ALSO.join(f.reason for f in faults)
                             + (f" {entry[0].upper()}{entry[1:]}" if entry else ""))
        if left_out:
            # Taken in part: the entries that cannot be read are left out alone, said in the plan and sent back with the next
            # request if the plan still lacks anything; the rest of the answer is kept.
            self._log.warning("[plan] the search block was used in part; left out: %s", "; ".join(left_out)[:600])
            block = taken
        self._clear_search_refused()

        def change(design: dict[str, Any]) -> dict[str, Any]:
            # The plan's own top-level scan parts cannot stand beside the block (the reader refuses them: a search keeps
            # them inside its block), and what they compared is the block's now; they are removed with the grid.
            protocol_now = design.get("protocol") or {}
            gone = [k for k in _optim.BLOCK_OWN_PARTS if k != "grid" and protocol_now.get(k) is not None]
            kept = {k: v for k, v in protocol_now.items() if k not in _optim.BLOCK_OWN_PARTS}
            if gone:
                self._log.warning("[plan] a search keeps %s inside its own block, so FI removed what the plan wrote beside "
                                  "it", ", ".join(f"`{k}`" for k in gone))
            out = {**design, "protocol": {**kept, "optimisation": block}}
            out["study_type"] = "find_best_design"
            return out

        merged_text = _plan.edit_design_block(before_text, change)
        if merged_text is None:
            return None
        parsed = _plan.parse(merged_text)
        if parsed.design is None:
            raise ValueError(f"the plan with your block in it cannot be read: {parsed.error}")
        if left_out:
            merged_text = _plan.add_to_section(merged_text, _PART_HEADING, ["", *[f"- {n}" for n in left_out]])
        return _plan.refresh_optimisation_section(_plan.refresh_model_section(merged_text))

    def _drafted_search_record(self) -> tuple[int, str, int]:
        """``(requests, model, iteration)`` of the last time FI asked the design step's model to write a drafted design's
        search block (``.fi/drafted_search_asked.json``); zeros when it never did."""
        try:
            record = json.loads((self.fi_dir / _DRAFTED_SEARCH_ASKED).read_text(encoding="utf-8"))
            return max(0, int(record.get("count") or 0)), str(record.get("model") or ""), int(record.get("iteration") or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            return 0, "", 0

    def _drafted_search_asked(self, iteration: int) -> int:
        """How many times FI has asked the design step's model, on the model and design round asked now."""
        count, model, at = self._drafted_search_record()
        return count if (model == str(self._model_for_node("design") or self.config.provider.model or "")
                         and at == iteration) else 0

    def _plan_block_for(self, design: Any) -> dict[str, Any] | None:
        """The search block of ``plan.md`` when it can be used and describes the same search as ``design``: the same
        settings to change and the same quantity and direction. ``None`` otherwise (a different search is never swapped in)."""
        planned, _sha = self._design_from_plan()
        if planned is None or not _optim.has_block(planned) or _optim.normalize(planned["protocol"]["optimisation"])[0] is None:
            return None
        theirs = planned["protocol"]["optimisation"]
        mine = design["protocol"].get("optimisation") if isinstance(design.get("protocol"), dict) else None
        if not isinstance(mine, dict):
            return None

        def names(block: dict[str, Any]) -> set[str]:
            return {str(v.get("name")) for v in block.get("design_variables") or [] if isinstance(v, dict) and v.get("name")}

        if not names(mine) or names(mine) != names(theirs):
            return None
        # Only a redesign that states a complete objective (quantity and direction) that is the plan's own matches it: an
        # objective left out or half stated is not "the same search", so the model is asked for the block instead.
        a, b = mine.get("objective"), theirs.get("objective")
        if not (isinstance(a, dict) and a.get("quantity") and a.get("direction") and isinstance(b, dict)
                and a.get("quantity") == b.get("quantity") and a.get("direction") == b.get("direction")):
            return None
        return theirs

    async def _complete_a_drafted_search(self, state: QuestState, design: Any, *, iteration: int) -> Any:
        """A design drafted at this step (a redesign: it did not come from ``plan.md``) that is a search for the best
        design whose block is missing or cannot be used is completed here, the way the plan's is
        (:meth:`_ask_plan_for_the_search`), instead of stopping the quest. In this order: (1) the block of ``plan.md``, when
        it is usable and describes the same search (same settings to change, same quantity and direction), is taken as it
        is, since the plan step or a person already settled it; (2) otherwise the design step's model is asked to write just
        that block (at most ``_SEARCH_ASKS`` requests of one chat call each per design round and model, counted in
        ``.fi/drafted_search_asked.json`` BEFORE each call so a resume never asks again), and only a block that
        passes the same reading as the plan's is kept; the rest of the design is untouched. What is still missing after
        that stops the quest as before; a search is never run as a sweep."""
        if not isinstance(design, dict) or not self._runs_code(state):
            return design
        wanted, missing, why = self._search_problems(state, design)
        if not wanted or not missing:
            return design
        if _frozen.load(self.quest_root) is not None:
            # After the freeze the block that runs is the frozen one (as at the run step): no model is asked.
            frozen_block = _optimise.block_of(self._protocol_block(state))
            if frozen_block is None:
                return design
            protocol = {k: v for k, v in (design.get("protocol") or {}).items() if k != "grid"}
            self._log.warning("[design] the redesign's search block was missing %s; FI used the frozen protocol's block",
                              "; ".join(missing))
            return {**design, "study_type": "find_best_design", "protocol": {**protocol, "optimisation": frozen_block}}
        if self._client is None:
            return design

        def with_block(block: dict[str, Any]) -> dict[str, Any]:
            protocol = {k: v for k, v in (design.get("protocol") or {}).items() if k != "grid"}
            return {**design, "study_type": "find_best_design", "protocol": {**protocol, "optimisation": block}}

        if isinstance(design.get("protocol"), dict):
            planned = self._plan_block_for(design)
            if planned is not None:
                self._log.warning("[design] the redesign's search block was missing %s; FI took the block of plan.md, "
                                  "which describes the same search", "; ".join(missing))
                return with_block(planned)
        model = str(self._model_for_node("design") or self.config.provider.model or "")
        # A scan that cannot run, with the rest of the block complete, is asked once (as at the plan step).
        asks = 1 if all(m.startswith("the coarse scan cannot run") for m in missing) else _SEARCH_ASKS
        while self._drafted_search_asked(iteration) < asks:
            count = self._drafted_search_asked(iteration) + 1
            try:
                self.fi_dir.mkdir(parents=True, exist_ok=True)
                (self.fi_dir / _DRAFTED_SEARCH_ASKED).write_text(json.dumps(
                    {"count": count, "model": model, "iteration": iteration, "at": _frozen.now()}) + "\n", encoding="utf-8")
            except OSError as e:
                self._log.warning("[design] FI could not write down that it was asking for the search block, so it did "
                                  "not ask: %r", e)
                return design
            gaps = _optim.plain_gaps(design, why, self._scan_limit_s())
            self._log.warning("[design] the redesign is a search for the best design and lacks %s; FI asked the design "
                              "step's model to write it (%d of %d)", "; ".join(gaps), count, asks)
            self._progress("Asking the model to complete the search for the best design")
            current = design["protocol"].get("optimisation") if isinstance(design.get("protocol"), dict) else None
            scan_only = all(m.startswith("the coarse scan cannot run") for m in missing)
            ask = ((_optim.scan_request(state.get("topic") or self.config.topic, missing) if scan_only
                    else _optim.ask_request(state.get("topic") or self.config.topic, missing))
                   + "\n\nThe block drafted so far (keep what is usable in it, and write what is missing or cannot be "
                   "read):\n" + json.dumps(current, ensure_ascii=False, default=str)[:4000]
                   + "\n\nAnswer with the `optimisation` block alone, as one JSON object.")
            try:
                reply = await self._chat(ask, node="design")
                obj = _parse_json_lenient(reply)
            except Exception as e:  # noqa: BLE001 -- the stop that follows says what is missing
                self._log.warning("[design] the design step could not be asked to complete the search: %s", e)
                continue
            block = obj.get("optimisation") if isinstance(obj, dict) and isinstance(obj.get("optimisation"), dict) else obj
            taken, left_out, stopping = _read_a_search_block(block) if isinstance(block, dict) else (None, [], [])
            if taken is None:
                self._log.warning("[design] the design step's answer to complete the search could not be used: %s",
                                  _optim.ALSO.join(f.reason for f in _optim.block_faults(block)[1] or stopping)
                                  if isinstance(block, dict) else "not a JSON object")
                continue
            if left_out:
                self._log.warning("[design] the design step's search block was used in part; left out: %s",
                                  "; ".join(left_out)[:600])
                self._keep_plan_reading(_optim.read_block(block), "design step: the search block it was asked for")
            design = with_block(taken)
            wanted, missing, why = self._search_problems(state, design)
            if not missing:
                self._log.info("[design] the redesign's search block is complete now (asked %d time(s))", count)
                print("[FI] the search for the best design is complete now; going on")
                break
        return design

    async def _ask_again_for_a_search_lost_since(self, state: QuestState, design: Any, plan_sha: str) -> tuple[Any, str]:
        """At the end of the design step, for a design from plan.md: when the audit or what the topic asks for rewrote the
        plan after its search was completed (:meth:`_ask_plan_for_the_search`, earlier in the step) and the search can no
        longer start, the plan as it is now is asked for its search, within the same bound per plan and model
        (``.fi/search_block_asked.json``: a plan already asked that often is not asked again), instead of the code step
        stopping with nobody asked. The design step calls it once, so it cannot loop. Returns the design and its plan's hash."""
        if not plan_sha:
            return design, plan_sha
        wanted, missing, _why = self._search_problems(state, design)
        if not (wanted and missing) or self._search_asked(plan_sha):
            return design, plan_sha  # a search that can start, or this plan was asked already (the stop says so)
        self._log.warning("[design] the plan changed after its search was completed, and the search can no longer start "
                          "(%s); FI asks the plan's model for it again", "; ".join(_one_line(m, 200) for m in missing[:3]))
        return await self._ask_plan_for_the_search(state, design, plan_sha)

    def _search_cannot_start_stop(self, state: QuestState, design: Any) -> QuestState | None:
        """The honest stop (no paper) for a design that is a search for the best design whose search FI could not make:
        the plan's model was asked, within its bound, to write the part that is missing or cannot be read, quoting the
        reader's exact reason each time, and the search is still not there. Nothing is run and nobody is asked to pick
        another model or to write any of it. ``None`` when the design is not a search, the search can start, or the quest
        runs no experiment of its own. A search is never run as a plain sweep instead."""
        if not self._runs_code(state):
            return None
        wanted, missing, block_why = self._search_problems(state, design)
        if not (wanted and missing):
            return None
        path = _plan.plan_path(self.quest_root)
        tried_plan = self._search_asked(_plan.sha256(path.read_text(encoding="utf-8"))) if path.is_file() else 0
        drafted = self._drafted_search_record()[0]
        asked = tried_plan or drafted
        said = "; ".join(_one_line(m, 500) for m in missing[:3])
        lacks = "; ".join(_optim.plain_gaps(design, block_why, self._scan_limit_s())) or "a search description FI can read"
        text = (f"the plan is a search for the best design, but FI could not make a search of it: it still lacks {lacks}"
                + (f" ({_one_line(block_why, 300)})" if block_why else ""))
        tried = [
            (f"asked the {'plan' if tried_plan else 'design step'}'s model {asked} time(s) to write the search, each time "
             "quoting what could not be read and why") if asked else
            "had no model to ask for the search (or the plan was already settled)",
            *[_one_line(m, 400) for m in missing[:3]],
            "did not run the plan as a plain sweep: that would answer a different question",
        ]
        self._log.warning("[design] the design is a search for the best design that cannot start (%s); stopping, nothing "
                          "was run", said)
        print(f"[FI] quest {self.quest_id}: the plan is a search for the best design, but FI could not make the search "
              f"({_one_line(missing[0], 200)}); nothing was run (see needs/STUCK.json)")
        return self._stop_before_run(text, "no_search", tried=tried)

    def _stop_for_unusable_search(self, runner: Any) -> QuestState | None:
        """The honest stop (no paper, no repair) for a study in its second stage whose kept search cannot be used
        (``core/optimise.py::run_search``): the runner's own words say what happened and what a person can do, and are what the
        quest reports. Nothing in any script can change that, so the run never goes to the repair step (which would spend model
        calls, or blame FI's analysis, on a run that did not start). ``None`` when the run stopped for no such reason."""
        plain = runner.plain_stop if isinstance(runner, _optimise.OptimisationRunner) else None
        if not (isinstance(plain, str) and plain.strip()):
            return None
        return self._stop_before_run(
            plain, "no_search", advice=plain,
            tried=["looked for the saved result of the search that chose this study's best design, to check that design on "
                   "new data"])

    def _grid_too_large_stop(self, state: QuestState) -> QuestState | None:
        """A protocol whose ``grid`` holds more settings than FI lists is no longer a stop (reached before any code is written
        and again before it runs, after the plan's model was asked once to make the grid smaller,
        :meth:`_ask_plan_for_the_grid`): it is said in run.log and the quest goes on. The run is timed on a few real trials
        (:meth:`_size_the_run`), the settings the time allowed can reach are planned lazily in the plan's order, run up to
        the time limit, and the paper says how many of the plan's settings that was; the quest stops only when not even one
        setting fits. Always ``None``: kept so the two places that asked for the stop still read the same."""
        if not self._runs_code(state):
            return None
        protocol = self._protocol_block(state)
        if not isinstance(protocol, dict) or protocol.get("optimisation"):
            return None
        said = _optimise_search.grid_too_large(_optimise_search.study_grid(protocol))
        if said:
            self._log.warning("[design] the plan's grid is more than FI lists (%s); FI will plan the settings it can reach in "
                              "the plan's order and run them up to the time limit", _one_line(said, 400))
        return None

    def _stop_if_the_search_cannot_start(self, state: QuestState, design: Any) -> None:
        """Pause for a person only when the quest's setup stops a search for the best design: the simulation is not a
        script of its own (``execution.split_analysis: false``), or the trials go to a cluster (``execution.background_jobs``:
        the search runs here, one design after another). A plan whose search block is missing or cannot be read is not a
        setup problem: it was asked of the plan's model (:meth:`_ask_plan_for_the_search`) and, if it is still not there,
        ends in the honest stop (:meth:`_search_cannot_start_stop`, reached when the code would be written and again when it
        would run). A no-op for a measurement, for a search that can start (:mod:`core.optimise`), and for a quest that
        runs no experiment of its own."""
        if not self._runs_code(state):
            return
        wanted, _missing, _block_why = self._search_problems(state, design)
        if not wanted:
            return
        path = _plan.plan_path(self.quest_root)
        block = _optim.normalize(design["protocol"]["optimisation"])[0] if _optim.has_block(design) else None
        goal = ""
        if block is not None:
            objective = block["objective"]
            goal = (f" (make {objective['quantity']} as {'low' if objective['direction'] == 'minimise' else 'high'} as "
                    f"possible by changing {', '.join(v['name'] for v in block['design_variables'])})")
        setup: list[str] = []
        if self._split_overridden(state, design):
            self._note_split_overridden()
        if self.config.execution.background_jobs:
            setup.append("the search runs here, one design after another, not as a job array on a cluster "
                         "(`execution.background_jobs` is on): turn it off in the quest's config and resume")
        if not setup:
            return
        steps = [
            f"Nothing was run. This quest is a search for the best design{goal}, and the search cannot start yet. "
            "Running the plan as a plain sweep over fixed settings would answer a different question, so the quest "
            "stops here.",
            *[f"In the quest's setup: {s}." for s in setup],
        ]
        reason = setup[0].split(" (`")[0]
        self._log.warning("[design] the design is a search for the best design that cannot start (%s); stopping", reason)
        print(f"[FI] quest {self.quest_id}: the plan is a search for the best design, but the search cannot start "
              f"({reason}); nothing was run (see NEXT_STEP.md)")
        self._pause_for_human(
            kind="plan",
            interaction="supply",
            headline="the search for the best design cannot start",
            steps=steps,
            recommended="Complete the quest's setup as the steps below say, then resume.",
            alternatives=["Ask for a change in words: `--revise-plan \"<what to change>\"`."],
            payload={"quest_id": self.quest_id, "plan_file": str(path), "best_design_stage": True},
        )

    def _split_overridden(self, state: QuestState, design: Any = None) -> bool:
        """Whether ``execution.split_analysis: false`` is overridden: the quest's plan is a search for the best design,
        which calls the simulation as a function of its own, so FI keeps the run and the analysis apart anyway. Not when
        the trials go to a cluster (``background_jobs``): that stays an honest stop."""
        if self.config.execution.split_analysis is not False or self.config.execution.background_jobs:
            return False
        if not self._runs_code(state):
            return False
        try:
            return bool(self._search_problems(state, design if design is not None else state.get("design"))[0])
        except Exception:  # noqa: BLE001 -- a design that cannot be read is not a search to override for
            return False

    def _note_split_overridden(self) -> None:
        """Say once, in run.log and in the record the paper's limitations and QUALITY.md are built from, that
        ``split_analysis: false`` was not followed for a search."""
        sentence = ("the setting `execution.split_analysis: false` was not followed: the study is a search for the best "
                    "design, which needs the run and the analysis kept apart, so FI ran it as two scripts")
        if not getattr(self, "_split_override_logged", False):
            self._split_override_logged = True
            self._log.warning("[design] %s", sentence)
        self._note_not_done([{"name": "execution.split_analysis override", "sentence": sentence,
                              "kind": "split_overridden"}], kind="split_overridden")

    def _runs_code(self, state: QuestState) -> bool:
        """Whether the quest runs an experiment of its own (not the no-simulation path, a survey or ``--analyze``)."""
        return not (state.get("no_simulation_resolved") or state.get("survey_mode_resolved")
                    or self.config.engine.analyze_local_first)

    def _retrieved_sources(self, state: QuestState) -> list[dict[str, str]]:
        """The sources this quest retrieved, labelled as the plan's literature and the paper label them ([1], [W1])."""
        return _oracle.retrieved_sources(
            [(label, meta) for label, meta, _item in _labelled_sources(state.get("literature") or [])])

    def _plan_source_findings(self, state: QuestState, protocol: Any = None) -> tuple[list[str], list[str]]:
        """``(checks whose expected value has no source a reader can check, notes on the model behind the numbers)`` for
        ``protocol`` (the plan's, read from ``plan.md``, when not given). Empty for a quest that runs no experiment."""
        if not self._runs_code(state):
            return [], []
        if not isinstance(protocol, dict):
            planned, _why = _plan.load_design(self.quest_root)
            protocol = planned.get("protocol") if isinstance(planned, dict) else None
        if not isinstance(protocol, dict):
            return [], []
        try:
            sources = self._retrieved_sources(state)
            return _oracle.source_gaps(protocol, sources), _oracle.model_notes(protocol, sources)
        except Exception as e:  # noqa: BLE001 -- a check of the plan must never stop a quest by crashing
            self._log.warning("[plan] where the checks' expected values come from could not be checked: %r", e)
            return [], [f"Where the checks' expected values come from could not be checked ({type(e).__name__})."]

    def _check_plan_sources(self, state: QuestState, *, stop: bool, protocol: Any = None,
                            no_criteria: bool = False) -> list[str]:
        """Say, in run.log, which checks of the plan expect a value with no source a reader can check (no ``reference``,
        or a source this quest did not retrieve); under ``rigor_profile: research`` with ``stop``, stop the quest at the
        plan until it says. A no-op once the protocol is frozen: the evidence record carries the gap from then on."""
        if _frozen.load(self.quest_root) is not None:
            return []
        gaps, _notes = self._plan_source_findings(state, protocol)
        for why in gaps:
            self._log.warning("[plan] %s: %s", why, _oracle.SOURCE_FORMS)
        if not gaps:
            _accepted.clear_pending(self.quest_root)
        # Only a plan a person can edit can be stopped for: without plan.md (the draft could not be used) the evidence
        # record carries the gap instead.
        if gaps and stop and self.config.rigor_profile == "research" and _plan.plan_path(self.quest_root).is_file():
            current = self._plan_protocol(protocol)
            try:
                named = _oracle.unsourced(current, self._retrieved_sources(state)) if current else []
                missing = _oracle.model_missing(current) if current else []
            except Exception as e:  # noqa: BLE001 -- the stop still says what it can
                self._log.warning("[plan] which checks lack a source could not be listed: %r", e)
                named, missing = [], []
            by_name = {str(o["name"]).strip(): o for o in _oracle.declared(current)} if current else {}
            # The person chose to go on with exactly these checks as they are now, numbers included (``--accept-checks``):
            # no stop. The gap stays in the evidence, marked "source not confirmed", and the paper is told to say so.
            choice = (_accepted.covers(self.quest_root, [by_name.get(n) or {"name": n} for n, _why in named])
                      if named else None)
            if choice is not None:
                self._go_on_unsourced(choice, [n for n, _why in named])
                return gaps
            checks = [{"name": n, "why": why + ("; the quest went on with it before, and its numbers have changed since"
                                                if _accepted.changed_since(self.quest_root, by_name.get(n) or {"name": n})
                                                else ""),
                       "expected": (by_name.get(n) or {}).get("expected"),
                       "fingerprint": _accepted.fingerprint(by_name[n]) if n in by_name else ""} for n, why in named]
            # Where a check's expected value comes from is not a question a person is asked: FI asked the plan step once
            # and it found none. The quest goes on by
            # itself, recorded as automatic: each check still runs and is still judged, and is marked "source not
            # confirmed" in the evidence, the freeze and the paper, so the result never counts as checked against an
            # independent source (nothing is relaxed). A check whose numbers change afterwards is marked again.
            history = _plan.history(self.quest_root)
            version = int(history[-1].get("version") or 0) if history else None
            went_on = _accepted.go_on_by_itself(self.quest_root, checks, plan_version=version)
            if went_on is not None:
                self._go_on_unsourced(went_on, [n for n, _why in named])
            if missing:
                self._log.warning("[plan] the model behind the numbers still leaves out %s; going on, and the evidence "
                                  "says so", ", ".join(missing))
        elif gaps and not self.__dict__.get("_said_unsourced"):
            self._said_unsourced = True
            print(f"[FI] {len(gaps)} check(s) in the plan do not say where their expected value comes from, or cite a "
                  "source this quest did not find; see plan.md and .fi/run.log")
        return gaps

    def _plan_protocol(self, protocol: Any = None) -> dict[str, Any] | None:
        """``protocol`` when given, else the protocol of the design block in ``plan.md`` (``None`` when there is none)."""
        if not isinstance(protocol, dict):
            planned, _why = _plan.load_design(self.quest_root)
            protocol = planned.get("protocol") if isinstance(planned, dict) else None
        return protocol if isinstance(protocol, dict) else None

    def _go_on_unsourced(self, choice: dict[str, Any], names: list[str]) -> None:
        """The person chose to go on with these checks as they are: say so once (run.log, the console, the audit trace),
        and take the stop's record away so no surface offers the choice again."""
        _accepted.clear_pending(self.quest_root)
        if self.__dict__.get("_went_on_unsourced"):
            return
        self._went_on_unsourced = True
        listed = ", ".join(repr(n) for n in names)
        if choice.get("by") == _accepted.AUTOMATIC:
            self._log.warning("[plan] %d check(s) still do not say where their expected value comes from after FI asked "
                              "a model once (%s): going on by itself; each is marked %r in the evidence", len(names),
                              listed, _accepted.NOT_CONFIRMED)
            print(f"[FI] {len(names)} check(s) do not say where their expected value comes from, and FI could not find "
                  f"it ({listed}). Nothing to do: FI goes on, and the result says these checks are not confirmed.")
            summary = f"FI went on by itself with {listed}: no source could be found ({_accepted.NOT_CONFIRMED})"
        else:
            self._log.warning("[plan] going on with %d check(s) whose expected value has no stated source (%s): %s chose "
                              "to go on (%s); each is marked %r in the evidence", len(names), listed, choice.get("by"),
                              choice.get("via"), _accepted.NOT_CONFIRMED)
            print(f"[FI] going on with {len(names)} check(s) whose expected value has no stated source ({listed}), as "
                  f"{choice.get('by')} chose; the result says their source is not confirmed")
            summary = f"{choice.get('by')} chose to go on with {listed} as they are ({_accepted.NOT_CONFIRMED})"
        self._audit_check("oracle_sources", self.quest_root / "needs" / _accepted.ACCEPTED_NAME,
                          status="accepted_without_source", summary=summary, problems=names)

    def _fill_request(self, named: list[tuple[str, str]], missing: list[str]) -> str:
        """The one request FI makes of the plan step before it stops for the checks' sources: fill in exactly what is
        missing, and nothing else."""
        parts = ["Fill in where each check's expected value comes from, and the model behind the numbers, IN THE DESIGN "
                 "BLOCK (`protocol.oracles` and `protocol.model`): the section *The model behind the numbers* above it is "
                 "only shown from the block and is not read."]
        if named:
            parts.append("These checks do not yet say where their expected value comes from:\n"
                         + "\n".join(f"- {why}" for _n, why in named))
            parts.append(
                "For each, write `reference` in one of these forms: " + _REFERENCE_FORMS + " A derivation must "
                + _oracle.DERIVATION_RULE + ". A value true by definition is written the same way (`derivation: sum of "
                "the normalised weights = 1 by definition`). Also give each check a `kind`, one of: "
                + ", ".join(_oracle.KINDS) + ".")
        if missing:
            parts.append(f"The model behind the numbers leaves out {', '.join(missing)}: fill `protocol.model` with "
                         "`summary` (one sentence: what produces the numbers), `assumptions` (a list) and `equations` "
                         "(a list, each with `id` E1, E2 ..., `formula`, `role` generates or analyses, and `source`: the "
                         "[n] of a source in the plan's list of the sources this quest found, or `derivation` with the "
                         "steps in `derivation`).")
        parts.append("Do not change any check's `name`, `expected`, `tolerance`, `tolerance_mode`, `case` or `measure`, "
                     "and change nothing else in the plan.")
        return "\n\n".join(parts)

    async def _settle_plan_sources(self, state: QuestState, *, protocol: Any = None, no_criteria: bool = False) -> list[str]:
        """:meth:`_check_plan_sources` after FI has asked the plan step, once per quest, to fill in what is missing
        (:meth:`_fill_plan_sources`): a stop comes only when the model could not."""
        filled = await self._fill_plan_sources(state, protocol)
        return self._check_plan_sources(state, stop=True, protocol=filled, no_criteria=no_criteria)

    async def _fill_plan_sources(self, state: QuestState, protocol: Any = None) -> Any:
        """Under ``rigor_profile: research``, before the quest stops because a check does not say where its expected
        value comes from (or the model behind the numbers is left empty), ask the plan step ONCE to fill exactly that in.
        A rewrite that changes anything else (a check's numbers, the grid, the hypothesis) is put back: filling in a
        source must never bend the check. Returns the protocol to check (the rewritten one when it was kept)."""
        if (self.config.rigor_profile != "research" or _frozen.load(self.quest_root) is not None
                or self._client is None or not self._runs_code(state)
                # A later pass checks the design it made, not plan.md: a rewrite of plan.md would not reach it.
                or int(state.get("iteration", 0) or 0) > 0):
            return protocol
        path = _plan.plan_path(self.quest_root)
        marker = self.fi_dir / _FILL_MARKER
        if not path.is_file() or marker.is_file():
            return protocol
        before_text = path.read_text(encoding="utf-8")
        before = _plan.parse(before_text).design
        if not isinstance(before, dict) or not isinstance(before.get("protocol"), dict):
            return protocol  # a plan that cannot be read is not rewritten: its stop says why
        current = before["protocol"]
        try:
            sources = self._retrieved_sources(state)
            named = _oracle.unsourced(current, sources)
            missing = _oracle.model_missing(current)
        except Exception as e:  # noqa: BLE001 -- the check that follows says what it can
            self._log.warning("[plan] what the plan leaves out could not be read: %r", e)
            return protocol
        if not named and not missing:
            return protocol
        by_name = {str(o["name"]).strip(): o for o in _oracle.declared(current)}
        if named and not missing and _accepted.covers(self.quest_root, [by_name.get(n) or {"name": n} for n, _w in named]):
            return protocol  # the person already chose to go on with these checks as they are
        self._log.info("[plan] %d check(s) do not say where their expected value comes from%s; asking the plan step "
                       "once to fill that in before stopping", len(named),
                       f", and the model behind the numbers leaves out {', '.join(missing)}" if missing else "")
        self._progress("Asking the model to say where each check's expected value comes from")

        def asked(outcome: str) -> None:
            # Written once the model has answered (used or not), so the stop can truthfully say FI already asked.
            try:
                self.fi_dir.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"at": _frozen.now(), "checks": [n for n, _why in named],
                                              "model_missing": missing, "outcome": outcome}) + "\n", encoding="utf-8")
            except OSError as e:
                self._log.warning("[plan] couldn't record that FI asked for the checks' sources: %r", e)

        # The plan step fills it in (never the model named for `oracle_review`: that one rechecks the expected values
        # later, and must not be the one that wrote their sources).
        try:
            fill_request = self._fill_request(named, missing)
            await self._revise_plan_parts(fill_request, path, by="engine", note=fill_request,
                                          scope=_fill_scope(bool(named), bool(missing)))
        except _ModelAnswerProblem as e:
            # A cut-off or withheld answer to a step the person never asked for is not a stop of its own: the stop for
            # the sources follows and says what is missing, and a resume asks again.
            self._log.warning("[plan] the plan step's answer to fill in the checks' sources was not complete: %s", e)
            return protocol
        except ValueError as e:
            asked("not usable")
            self._log.warning("[plan] the plan step could not fill in the checks' sources: %s", e)
            return protocol
        except Exception as e:  # noqa: BLE001 -- the stop that follows says what is missing; a resume asks again
            self._log.warning("[plan] the plan step could not be asked to fill in the checks' sources: %s", e)
            return protocol
        after_text = path.read_text(encoding="utf-8")
        after = _plan.parse(after_text).design
        why_not = _fill_changed_more(before, after)
        if why_not:
            asked("put back")
            path.write_text(before_text, encoding="utf-8")
            _plan.record_version(self.quest_root, before_text, by="engine",
                                 note=f"put back: the rewrite that was to fill in the checks' sources {why_not}")
            self._log.warning("[plan] the rewrite that was to fill in the checks' sources %s; it was put back", why_not)
            return protocol
        asked("kept")
        filled = (after or {}).get("protocol")
        filled = filled if isinstance(filled, dict) else None
        try:
            still = _oracle.unsourced(filled, sources) if filled else named
        except Exception:  # noqa: BLE001
            still = named
        self._log.info("[plan] the plan step filled in the sources of %d of %d check(s)%s", len(named) - len(still),
                       len(named), "" if not _oracle.model_missing(filled) else "; the model behind the numbers is "
                       "still incomplete")
        # A person who already read the plan at its stop did not read what this rewrite wrote (the checks' numbers are
        # as they read them): the freeze record says so.
        if (self.fi_dir / "paused_at_plan.flag").is_file():
            was = {str(o["name"]): _oracle.reference_of(o) for o in _oracle.declared(current)}
            written = [str(o["name"]) for o in _oracle.declared(filled) if was.get(str(o["name"])) != _oracle.reference_of(o)]
            if written:
                try:
                    (self.fi_dir / _FILLED_AFTER_READ).write_text(json.dumps({"checks": written, "at": _frozen.now()}) + "\n",
                                                                   encoding="utf-8")
                except OSError:
                    pass
        return filled if filled is not None else protocol

    @staticmethod
    def same_model(name: str, served: str, family: str = "") -> bool:
        """Whether a model name a config (or a person) gave means the model a connection reported: the same id, its
        family, or the id with a date added (``gpt-5.6-luna`` for ``gpt-5.6-luna-2026-09``) or ``-preview`` / ``-latest``. Not the other way
        round, and not one name merely inside another (``gpt-5`` is not ``gpt-5-mini``)."""
        n, s, f = (str(x or "").strip().lower() for x in (name, served, family))
        if not n or not s:
            return False
        return n in (s, f) or re.fullmatch(re.escape(n) + r"(?:[-_.@](?:\d{4}[\d.-]*|\d{2}-\d{2}|preview|latest))+", s) is not None

    def _earlier_main_model(self) -> str | None:
        """The model that answered the latest earlier call made on the quest's main model (not a step's own model,
        not a fallback provider), as the connection reported it; ``None`` when there is none."""
        earlier = None
        try:
            for row in _attempts.read(self.fi_dir, _attempts.MODEL_CALLS):
                if (row.get("outcome") == "ok" and row.get("reported") and row.get("served_model")
                        and not row.get("fallback") and not self._model_for_node(str(row.get("node") or ""))):
                    earlier = str(row["served_model"])
        except Exception:  # noqa: BLE001 -- a note, never in the way of the quest
            earlier = None
        return earlier

    def _say_model_change(self) -> None:
        """When the chat panel's model replaced the config's for this run (``launch.py --vscode-chat-model``): one plain
        run.log line, and a ``model_changed`` event in the trace when it is not the model the quest's earlier calls
        already used (a quest resumed from the chat again on the same model has not changed model). Once per run. Also
        takes note of that earlier model, so the first call of this run can say whether the model changed partway."""
        if self.__dict__.get("_model_change_said"):
            return
        self.__dict__["_model_change_said"] = True
        earlier = self._earlier_main_model()
        self.__dict__["_earlier_served_model"] = earlier
        change = (self.config.provider.extra or {}).get(self.CHAT_MODEL_KEY)
        if isinstance(change, dict) and change.get("after") and change.get("after") != change.get("before"):
            before = str(change.get("before") or "") or "no model"
            self._log.warning("[model] the chat panel's model %s replaces %s from config.yaml", change["after"], before)
            # A change of config.yaml's model is recorded apart (_take_model_change compares the config's own model,
            # never the chat panel's): this is the change this run makes. Nothing made yet (a new quest started from
            # the chat): the event says so, so the paper is not told two models produced the quest.
            # With no earlier call naming its model (the picker on Auto, say), the trace's latest change says which
            # model the quest was last moved to: resumed again on that one, nothing changed.
            last = earlier or self._last_model_in_trace()
            if not (last and self.same_model(str(change["after"]), last)):
                taken = self.__dict__.get("_model_change_taken")
                # Before this run: the model the earlier calls named, else the one config.yaml had before its change,
                # else config.yaml's (never a model config.yaml names now but this run does not use, when it changed).
                was = earlier or (taken[0] if isinstance(taken, tuple) else None) or change.get("before") or None
                self._audit("model_changed", before=was, after=change["after"],
                            source="the model picked in the VS Code chat panel",
                            before_any_step=not self._made_anything())
                self.__dict__["_chat_change_audited"] = True

    def _last_model_in_trace(self) -> str | None:
        """The model the quest's latest ``model_changed`` event (before this run's) moved it to: the chat panel's or the
        served model (``after``), else the model config.yaml's change left running (``run_uses``); ``None`` when none."""
        last = None
        before_run = self.__dict__.get("_model_events_before_run")  # this run's own config.yaml change is not earlier
        try:
            earlier = [e for e in _audit_log.read(self.audit.path) if e.get("kind") == "model_changed"]
            for e in earlier[:before_run] if isinstance(before_run, int) else earlier:
                main = next((c.get("to") for c in e.get("changes") or []
                             if isinstance(c, dict) and c.get("setting") == "provider.model"), None)
                last = e.get("after") or e.get("run_uses") or main or last
        except Exception:  # noqa: BLE001 -- a note, never in the way of the quest
            return None
        return str(last) if last else None

    def _note_served_model(self, node: str, served: dict[str, Any]) -> None:
        """Once per run, on the first answered call that used the quest's main model on its own provider (not a fallback):
        which model actually answered, as the connection named it; a plain warning when that is not the model the
        config names, and when it is not the model that answered this quest's earlier calls (the results before and
        after were then made by different models, which the trace records, once)."""
        if self.__dict__.get("_served_model_said") or self._model_for_node(node) or served.get("fallback"):
            return
        model = str(served.get("model") or "")
        if not served.get("reported") or not model:
            return
        self.__dict__["_served_model_said"] = True
        family = str(served.get("family") or "")
        vendor = f" (vendor: {served['vendor']})" if served.get("vendor") else ""
        self._log.info("[model] the first call of this run (%s) was answered by %s%s, as the connection reports it",
                       node, model, vendor)
        from .vscode_bridge import is_router_alias

        asked = str(self.config.provider.model or "").strip()
        # The claude CLI's short names (``haiku``, ``opus[1m]``) name the newest model of their family: the full id it
        # reports for one is the same model, not a warning.
        short = re.sub(r"\[[^\]]*\]$", "", asked.lower())
        alias = (self.config.provider.name == "claude_cli" and short in ("haiku", "sonnet", "opus", "fable")
                 and short in re.split(r"[-_.\[\]\s]+", model.lower()))
        if asked and not alias and not is_router_alias({"id": asked}) and not self.same_model(asked, model, family):
            self._log.warning("[model] the config names %s; %s served %s", asked,
                              "VS Code" if self.config.provider.name == "vscode_extension" else "the connection", model)
        earlier = self.__dict__.get("_earlier_served_model")
        if earlier and earlier.strip().lower() != model.strip().lower():
            self._log.warning("[model] this quest's earlier results were made by %s and its results from here on by %s: "
                              "the quest's results were made by different models", earlier, model)
            change = (self.config.provider.extra or {}).get(self.CHAT_MODEL_KEY)
            taken = self.__dict__.get("_model_change_taken")
            already = ((self.__dict__.get("_chat_change_audited") and isinstance(change, dict)
                        and self.same_model(str(change.get("after") or ""), model, family))
                       # config.yaml's own model change from the earlier model to this one, with no chat panel model.
                       or (isinstance(taken, tuple) and not isinstance(change, dict)
                           and self.same_model(taken[1], model, family) and self.same_model(taken[0], earlier)))
            # The chat panel's change, or config.yaml's change to this model, is in the trace already, from the start
            # of this run.
            if not already:
                self._audit("model_changed", before=earlier, after=model, source="the model that answered the calls",
                            node=node)

    async def _connect_llm(self) -> None:
        """Resolve the provider and build the LLM client (with its fallback chain), as a quest run does."""
        endpoint = await resolve_endpoint_async(self.config.provider, self.supervisor)
        # Where the calls go and which model the config asks for; the model that actually answers is named on the first
        # call (``_note_served_model``), since a connection may serve another one.
        where = endpoint.base_url or {
            "vscode_bridge": "the VS Code chat models", "cli": "the local command-line tool",
        }.get(getattr(endpoint, "transport", ""), "(no address)")
        self._log.info(
            "provider %s -> %s (model asked for: %s)",
            self.config.provider.name, where, endpoint.model,
        )
        self._say_model_change()
        self._say_step_providers()
        self._say_claude_account()
        # A connection that names the model that answered each call (an HTTP API, the claude CLI): a call on it whose
        # model went unnamed is a gap in the quest's record of its calls (core/attempt_records.py::model_call_gaps).
        self._reports_model = (getattr(endpoint, "transport", "") == "http"
                               or self.config.provider.name == "claude_cli")
        self._client = self._new_llm_client(endpoint, node_model_fallbacks=self.config.provider.node_model_fallbacks)
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

    def _say_claude_account(self) -> None:
        """When the Claude CLI serves any step (the main provider, a fallback, or a step's own provider), say once in run.log
        and on the console which account it will bill: an ``ANTHROPIC_API_KEY`` in the environment makes it the API account
        and not the subscription. FI does not change which account is used; the key itself is never printed."""
        provider = self.config.provider
        steps = list((getattr(provider, "node_providers", None) or {}).values())
        names = {provider.name, *provider.fallback, *(s.name for s in steps), *(f for s in steps for f in s.fallback)}
        if "claude_cli" not in names or self.__dict__.get("_claude_account_said"):
            return
        self.__dict__["_claude_account_said"] = True
        self._progress(f"[provider] {claude_cli_account_line()}")

    def _say_step_providers(self) -> None:
        """``provider.node_providers``: before anything is spent, a step whose provider's key is not set stops the quest
        with the plain sentence; then one run.log line per step on its own provider, and one for a step that is in
        ``provider.node_models`` or ``provider.node_ensemble`` too (the provider wins; the ensemble stays on the main
        one). Once per engine."""
        provider = self.config.provider
        steps = getattr(provider, "node_providers", None) or {}
        if not steps or self.__dict__.get("_step_providers_said"):
            return
        self.__dict__["_step_providers_said"] = True
        missing = missing_step_api_keys(provider)
        if missing:
            raise RuntimeError(missing[0] + " Nothing was started.")
        for line in step_provider_lines(provider):
            self._log.info("[provider] %s", line)
        for step in steps:
            if provider.node_models and node_provider_for(provider.node_models, step) is not None:
                self._log.warning("[provider] `%s` is in provider.node_models and provider.node_providers: the step "
                                  "uses its own provider's model (node_providers wins)", step)
            if provider.node_ensemble and node_provider_for(provider.node_ensemble, step) is not None:
                self._log.warning("[provider] `%s` is a multi-model ensemble step: its models are asked on the main "
                                  "provider, not on the one set in provider.node_providers", step)

    async def _client_for(self, node: str | None) -> tuple[Any, bool]:
        """The model client for ``node`` and whether its connection names the model that answered: the quest's own,
        or, for a step in ``provider.node_providers``, that step's provider's, built on its first call and kept for
        the quest (one per entry, not per step, so ``implement`` and ``implement_oracle`` named by one key share one;
        concurrent first calls build it once)."""
        step = node_provider_for(getattr(self.config.provider, "node_providers", None), node)
        if step is None:
            return self._client, bool(getattr(self, "_reports_model", False))
        key, entry = step
        built = self.__dict__.setdefault("_step_clients", {})
        if key not in built:
            lock = self.__dict__.setdefault("_step_client_lock", asyncio.Lock())
            async with lock:
                if key not in built:
                    built[key] = await self._build_step_client(key, entry)
        return built[key]

    async def _build_step_client(self, key: str, entry: Any) -> tuple[Any, bool]:
        derived = step_provider_config(self.config.provider, entry)
        endpoint = await resolve_endpoint_async(derived, self.supervisor)
        if derived.name in PROXY_PROVIDERS:
            self.__dict__.setdefault("_step_proxies", []).append(derived.name)
        self._log.info("[provider] step `%s`: %s -> %s (model asked for: %s)", key, derived.name,
                       endpoint.base_url or getattr(endpoint, "transport", ""), endpoint.model)
        client: Any = self._new_llm_client(endpoint)
        if derived.fallback:
            client = FallbackLLMClient(
                client, [(n, self._make_fallback_factory(n, base=derived)) for n in derived.fallback], log=self._log)
        reports = getattr(endpoint, "transport", "") == "http" or derived.name == "claude_cli"
        return client, reports

    def _new_llm_client(self, endpoint: Any, *, node_model_fallbacks: dict[str, str] | None = None) -> LLMClient:
        provider = self.config.provider
        return LLMClient(
            endpoint,
            timeout_s=provider.http_timeout_s,
            cli_timeout_s=provider.cli_timeout_s,
            cli_inactivity_timeout_s=provider.cli_inactivity_timeout_s,
            node_cli_timeout_s=provider.node_cli_timeout_s,
            node_http_timeout_s=provider.node_http_timeout_s,
            node_model_fallbacks=node_model_fallbacks or {},
            max_prompt_chars=provider.max_prompt_chars,
            heartbeat_cb=self._llm_heartbeat,
            run_log=self._log,
        )

    async def _close_step_clients(self) -> None:
        """Close the step providers' clients and release the proxies they started (the fallbacks they built too)."""
        built = self.__dict__.pop("_step_clients", {})
        for client, _reports in built.values():
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 -- best-effort cleanup
                pass
            for fb_name in getattr(client, "built_fallback_providers", ()):
                if fb_name in PROXY_PROVIDERS:
                    await self.supervisor.release(fb_name)
        for name in self.__dict__.pop("_step_proxies", []):
            await self.supervisor.release(name)

    async def revise_plan(self, request: str, *, by: str = "request", scope: _plan_scope.PartScope | None = None,
                          note: str | None = None) -> dict[str, Any]:
        """Rewrite ``plan.md`` as the person asked (``--revise-plan``): the whole file goes to the model with the
        request, and the reply replaces it only if its design block can be used (one retry with the reason). A free-text
        request of a person cannot be mapped to parts of the plan, so it keeps the whole rewrite. FI's own repairs pass
        ``scope``: the model is then asked for the parts it names alone (:meth:`_revise_plan_parts`), never for the file.
        The quest stays where it was; resuming runs the revised design. Every version is kept.

        Connects its own model client when the engine has none (a one-shot command), and closes it, and releases
        a proxy provider it started, when it is done, as a quest run does."""
        request = (request or "").strip()
        if not request:
            raise ValueError("say what to change in the plan")
        path = _plan.plan_path(self.quest_root)
        if not path.is_file():
            raise FileNotFoundError(
                f"{self.quest_id} has no plan.md yet: it has not reached the plan step",
            )
        connected_here = self._client is None
        if connected_here:
            await self._connect_llm()
        try:
            if request.strip().rstrip(".").lower() == _optim.MEASURE_INSTEAD:
                before = path.read_text(encoding="utf-8")
                return await self._rewrite_plan(
                    request, path, by=by, expect=before,
                    finish=lambda revised: self._check_measure_instead(before, revised))
            if scope is not None:
                return await self._revise_plan_parts(request, path, scope=scope, by=by, note=note or request)
            return await self._rewrite_plan(request, path, by=by)
        finally:
            if connected_here and self._client is not None:
                await self._close_step_clients()
                await self._client.aclose()
                if self.config.provider.name in PROXY_PROVIDERS:
                    await self.supervisor.release(self.config.provider.name)
                self._client = None

    def _measure_instead_possible(self, text: str) -> bool:
        """Whether "measure over the planned settings" can be done from this plan: it lists settings, each with a range or a
        list of values (so :meth:`_check_measure_instead` can hold the measured values to them). Only then is it offered."""
        design = _plan.raw_design_block(text)
        if not isinstance(design, dict):
            return False
        block = (design.get("protocol") or {}).get("optimisation") if isinstance(design.get("protocol"), dict) else None
        items = block.get("design_variables") if isinstance(block, dict) else None
        if not isinstance(items, list) or not items:
            return False
        named: set[str] = set()
        for index, item in enumerate(items, start=1):
            variable = _optim._variable(item, index)[0]
            if variable is None:
                return False
            named.add(str(variable["name"]))
        independent = (design.get("variables") or {}).get("independent") if isinstance(design.get("variables"), dict) else None
        return all(str(n) in named for n in independent) if isinstance(independent, list) else True

    def _check_measure_instead(self, before_text: str, after_text: str) -> str:
        """``after_text`` when a rewrite to "measure over the planned settings instead of searching" kept the plan's question
        and settings: its hypothesis, expected outcome and measured quantity are unchanged, and every setting it measures
        over is one the plan already listed (a design variable) with values inside the range or list the plan gave. Otherwise
        ``ValueError`` (plan.md stays as it was), in plain words."""
        old, new = _plan.raw_design_block(before_text), _plan.raw_design_block(after_text)
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise ValueError("the plan could not be changed to a measurement: the rewrite could not be read; plan.md is unchanged")

        def flat(value: Any) -> str:
            return " ".join(str(value or "").split())

        for key in ("hypothesis", "expected_outcome"):
            if flat(old.get(key)) != flat(new.get(key)):
                raise ValueError(f"the plan was not changed to a measurement: the rewrite also changed the plan's {key.replace('_', ' ')}, "
                                 "and a measurement over the same settings must keep the question; plan.md is unchanged")
        old_dep = (old.get("variables") or {}).get("dependent") if isinstance(old.get("variables"), dict) else None
        new_dep = (new.get("variables") or {}).get("dependent") if isinstance(new.get("variables"), dict) else None
        if old_dep != new_dep:
            raise ValueError("the plan was not changed to a measurement: the rewrite changed what is measured; plan.md is unchanged")
        old_block = (old.get("protocol") or {}).get("optimisation") if isinstance(old.get("protocol"), dict) else None
        listed: dict[str, Any] = {}
        if isinstance(old_block, dict):
            for item in old_block.get("design_variables") or []:
                if isinstance(item, dict) and item.get("name"):
                    listed[str(item["name"])] = item
        # A plan whose search block could not be read lists its settings only as the independent variables: no range to hold
        # the values to, but still the only settings that may be measured over.
        independent = (old.get("variables") or {}).get("independent") if isinstance(old.get("variables"), dict) else None
        for name in independent if isinstance(independent, list) else []:
            listed.setdefault(str(name), None)
        grid = (new.get("protocol") or {}).get("grid") if isinstance(new.get("protocol"), dict) else None
        if not listed or not isinstance(grid, dict) or not grid:
            raise ValueError("the plan was not changed to a measurement: the rewrite does not say which of the plan's own "
                             "settings to measure over, at which values; plan.md is unchanged")
        for name, values in grid.items():
            variable = listed.get(str(name))
            known = str(name) in listed
            values = values if isinstance(values, list) else [values]
            if not values:
                raise ValueError(f"the plan was not changed to a measurement: the rewrite lists no values for {name}; plan.md is unchanged")
            if known and variable is None:
                raise ValueError(f"the plan was not changed to a measurement: the plan lists {name} as a setting but not the "
                                 "values it may take, so FI cannot tell whether the values to measure are the plan's own; "
                                 "FI did not change the plan")
            if not known:
                raise ValueError(f"the plan was not changed to a measurement: the rewrite measures over {name}, which the plan "
                                 "did not list as a setting; plan.md is unchanged")
            for value in values:
                if variable is not None and not _optim._inside(variable, value):
                    raise ValueError(f"the plan was not changed to a measurement: the rewrite measures {name} at {value}, "
                                     "outside what the plan listed for it; plan.md is unchanged")
        return after_text

    async def _usable_design_block(self, revised: str, *, ask: bool) -> tuple[str, str, bool]:
        """``(the rewritten plan, "", asked)`` when its design block can be used, else ``(it, why not, asked)``; ``asked``
        says whether the block alone was asked for. A block that is not valid YAML is first repaired where that changes
        no key and no value (:func:`core.plan.repair_block`); then, with ``ask``, the model is asked ONCE for the block
        alone, told exactly where it is broken (only the block and the parser's message are sent, not the plan again).
        A file with no design block, or a block that is valid YAML but not a usable design (a missing hypothesis: fixing
        it needs the plan and the request), is left to the caller's whole-file retry."""
        parsed = _plan.parse(revised)
        if parsed.design is not None:
            return revised, "", False
        why = parsed.error or "no design"
        found = _plan.design_block(revised)
        if found is None or _plan.yaml_problem(found[2]) is None:
            # Not a YAML fault of the block, so asking for the block alone would not help (the whole-file retry does).
            self._log.warning("[plan] the rewritten plan's design block could not be used (%s); %s", why,
                              "the reply has no design block" if found is None else
                              "it is valid YAML but not a usable design, so the block alone is not asked for")
            return revised, why, False
        fixed, notes = _plan.repair_design(revised)
        if fixed is not None:
            repaired = _plan.parse(fixed)
            if repaired.design is None:
                # Valid YAML once repaired, but not a usable design (a missing hypothesis): the whole-file retry.
                return revised, repaired.error or "no design", False
            self._log.info("[plan] the rewritten plan's design block was not valid YAML (%s); repaired without asking "
                           "again: %s", why, "; ".join(notes))
            return fixed, "", False
        if not ask:
            self._log.warning("[plan] the rewritten plan's design block could not be used (%s); the block alone was "
                              "already asked for once", why)
            return revised, why, False
        self._log.warning("[plan] the rewritten plan's design block could not be used (%s); asking the model once for "
                          "the block alone, with where it is broken", why)
        self._progress("Asking the model to correct the plan's design block")
        reply = await self._chat(_plan.block_fix_request(found[2], why), node="plan_revise")
        self._keep_plan_answer(reply)
        block = _plan.block_from_reply(reply)
        candidate = _plan.replace_design_block(revised, block) if block is not None else None
        if candidate is None:
            return revised, f"{why}; asked once for the corrected block alone, and none came back", True
        again = _plan.parse(candidate)
        if again.design is None:
            fixed, notes = _plan.repair_design(candidate)
            if fixed is None or _plan.parse(fixed).design is None:
                return revised, (f"{why}; asked once for the corrected block alone, and it still could not be used "
                                 f"({again.error or 'no design'})"), True
            candidate = fixed
        self._log.info("[plan] the model corrected the plan's design block (asked once, for the block alone)")
        return candidate, "", True

    async def _rewrite_plan(
        self, request: str, path: Path, *, by: str = "request", expect: str | None = None,
        finish: Any = None, note: str | None = None,
    ) -> dict[str, Any]:
        """``expect``: the plan as it was when the request was made; when plan.md differs just before the write (a person
        saved a change meanwhile) nothing is written (:class:`_PlanEditedMeanwhile`). ``finish(revised)`` may return the text
        to write instead, or ``None``/raise ``ValueError`` for an answer that must not be used."""
        current = path.read_text(encoding="utf-8")
        _plan.note_edit(self.quest_root, current)  # a hand edit made before this request is its own version
        prompt = self._prompts["plan_revise"].substitute(
            topic=self.config.topic, plan_md=current, request=request, checks=_plan_checks_note(current),
        )
        why = ""
        revised = ""
        asked_block = False
        for _attempt in (1, 2):
            reply = await self._chat(
                prompt + (
                    f"\n\n## Your last reply could not be used\nThe design block: {why}. "
                    "Return the whole file again, with that fixed.\n" if why else ""
                ),
                node="plan_revise",
            )
            self._keep_plan_answer(reply, chars=self._PLAN_FILE_ANSWER_CHARS)
            revised = _plan.strip_outer_fence(reply)
            revised = _plan.keep_latex_in_block(revised)  # `"$\text{SNR}$"` is LaTeX, not a tab and `ext`
            # A block that is not valid YAML is sent back once, alone and with the parser's message, on whichever answer
            # it is first found on (an answer with no block, or a valid one that is no design, is asked for again as a
            # whole file): at most three calls in all.
            revised, why, asked = await self._usable_design_block(revised, ask=not asked_block)
            asked_block = asked_block or asked
            if not why or asked:
                break  # used, or a block that could not be used even when asked for alone: not asked for again
        if why:
            raise ValueError(f"the revised plan could not be used ({why}); plan.md is unchanged")
        # The model section is shown from the block and never read back: shown again from the block as rewritten, so the
        # plan never shows a model or a check's source the block does not hold (or leaves out one it does).
        revised = _plan.refresh_model_section(revised)
        revised = _plan.refresh_optimisation_section(revised)
        if finish is not None:
            kept = finish(revised)
            if kept is None:
                raise ValueError("the revised plan has no search block that can be used; plan.md is unchanged")
            revised = kept
        if expect is not None and path.read_text(encoding="utf-8") != expect:
            raise _PlanEditedMeanwhile("plan.md was changed while the request was being answered")
        path.write_text(revised, encoding="utf-8")
        entry = _plan.record_version(self.quest_root, revised, by=by, note=(note or request)[:300])
        self._log.info("[plan] revised %s: version %d",
                       "by the engine for its oracle check" if by == "engine" else "on request", entry["version"])
        return {"version": entry["version"], "sha256": entry["sha256"], "path": str(path)}

    async def _flat_rewrite(self, state: QuestState, request: str) -> dict[str, Any]:
        """plan.md rewritten from what FI measured of a search whose objective did not move, checked before it is taken:
        what the results are judged by is put back as it was (``_FLAT_MAY_CHANGE``; the search's constraints too), the
        search must be one FI can run, the protocol must change, and what the topic asks for is judged on the rewritten
        protocol itself. The frozen protocol cannot be repaired after the change is approved, so a rewrite that fails
        one of these is not taken (``ValueError``; plan.md is as it was)."""
        path = _plan.plan_path(self.quest_root)
        before = path.read_text(encoding="utf-8")
        basis = self._protocol_block(state) or ((_plan.parse(before).design or {}).get("protocol") or {})
        put_back: list[str] = []
        kept_keys: list[str] = []

        def held(design: dict[str, Any]) -> dict[str, Any]:
            new = dict(design.get("protocol") or {})
            for key in {*basis, *new} - self._FLAT_MAY_CHANGE:
                if new.get(key) != basis.get(key):
                    put_back.append(key)
                    if key in basis:
                        new[key] = basis[key]
                    else:
                        new.pop(key, None)
            old_block, new_block = basis.get("optimisation"), new.get("optimisation")
            if isinstance(old_block, dict) and isinstance(new_block, dict):
                block = dict(new_block)
                for key in self._FLAT_SEARCH_JUDGED:
                    if block.get(key) != old_block.get(key):
                        put_back.append(f"optimisation.{key}")
                        if key in old_block:
                            block[key] = old_block[key]
                        else:
                            block.pop(key, None)
                old_obj, new_obj = old_block.get("objective"), block.get("objective")
                if isinstance(old_obj, dict) and isinstance(new_obj, dict) and \
                        new_obj.get("direction") != old_obj.get("direction"):
                    put_back.append("optimisation.objective.direction")
                    block["objective"] = {**new_obj, "direction": old_obj.get("direction")}
                new["optimisation"] = block
            return {**design, "protocol": new}

        def finish(revised: str) -> str | None:
            put_back.clear()  # the reply is judged again when it is asked for twice
            kept_keys.clear()
            design = _plan.parse(revised).design
            if design is None:
                return None
            # The answer is read as the keys it changes (a redesign rewrites only the changed parts): a key of the search
            # block that the plan has and the answer leaves out is kept from the plan. A key it sets to empty is a removal.
            block_now = (design.get("protocol") or {}).get("optimisation")
            old_block = basis.get("optimisation")
            if isinstance(block_now, dict) and isinstance(old_block, dict):
                kept_keys.extend(sorted(k for k in old_block if k not in block_now and old_block[k] not in (None, "", [], {})))
                if kept_keys:
                    revised = _plan.edit_design_block(
                        revised, lambda b: {**b, "protocol": {**(b.get("protocol") or {}), "optimisation": {
                            **{k: copy.deepcopy(old_block[k]) for k in kept_keys}, **(b.get("protocol") or {})["optimisation"]}}}
                    ) or ""
                    design = _plan.parse(revised).design if revised else None
                    if design is None:
                        raise ValueError("what the answer left out could not be kept from the plan")
                    revised = _plan.refresh_optimisation_section(_plan.refresh_model_section(revised))
            design = held(design)
            if put_back:
                revised = _plan.edit_design_block(revised, lambda block: held(block)) or ""
                design = _plan.parse(revised).design if revised else None
                if design is None:
                    raise ValueError("what the results are judged by could not be put back as it was")
                # The model and search sections are shown from the block: shown again from the block as put back.
                revised = _plan.refresh_optimisation_section(_plan.refresh_model_section(revised))
            wanted, missing, why = self._search_problems(state, design)
            if not wanted or missing or why:
                raise ValueError("its search for the best design cannot be run: "
                                 + "; ".join([*missing, *([why] if why else [])] or ["it is not a search"]))
            if _frozen.sha256(design.get("protocol") or {}) == _frozen.sha256(basis):
                # What of the answer could not be read is named (the field, what was written, the allowed values), not only
                # that nothing changed: the model is asked again with it.
                unread = " ".join(str(r) for r in getattr(self, "_answer_refused", None) or [])
                raise ValueError("it changed nothing of the protocol" + (f"; what could not be read: {unread}" if unread else ""))
            return revised

        # Only how the design is described and searched is asked for (what the results are judged by is not), and a new
        # entry of the simplifications that says what was changed; `held` above still puts back what the reply touched.
        scope = _plan_scope.PartScope(
            parts=tuple(sorted(self._FLAT_MAY_CHANGE)),
            entries={"simplifications": _plan_scope.EntryScope("id", frozenset(), True)})
        closed = _optim.closed_choices_note(scope.names())
        done = await self._revise_plan_parts(
            request.rstrip() + (("\n\n" + closed) if closed else ""), path, scope=scope, by="engine", finish=finish,
            note=lambda: "a design whose objective can move, asked by FI"
            + (f"; kept from the plan: {', '.join(kept_keys)}" if kept_keys else ""))
        if kept_keys:
            self._log.warning("[evidence_gate] the answer left out %s of the search block; kept from the plan as they were",
                              ", ".join(kept_keys))
        if put_back:
            self._log.warning("[evidence_gate] the rewrite changed what the results are judged by (%s); FI put it back as "
                              "it was", ", ".join(sorted(set(put_back))))
        req = self._topic_requirements_read()
        design = _plan.parse(path.read_text(encoding="utf-8")).design or {}
        if self._runs_code(state) and req is not None and not req.get("error") and not _topic_req.is_empty(req):
            try:
                gaps, _unjudged = await self._topic_gaps(req, design, design.get("protocol"))
                why = ("the rewrite dropped what the topic asks for: "
                       + "; ".join(str(g.get("sentence") or g.get("name")) for g in gaps)[:300]) if gaps else ""
            except Exception as exc:  # noqa: BLE001 -- not checked against the topic: not taken
                why = f"the rewrite could not be checked against the topic ({_one_line(str(exc), 160)})"
            if why:
                path.write_text(before, encoding="utf-8")
                _plan.record_version(self.quest_root, before, by="engine",
                                     note="the rewrite for a flat objective was not taken; the plan is put back")
                raise ValueError(why)
        return done

    def _run_gave_no_results(self, state: QuestState) -> bool:
        """True when FI's own record shows the experiment that led back to the design produced no results: a simulation
        ran (``exec_result``) and its result holds no number, or only a number that is none of the study's quantities and a
        word that it failed (the same reading as the gate's, :meth:`_no_results_verdict`). False when it cannot be told, for
        a study of data or of the literature, and when an earlier round of this quest did have results."""
        if any(isinstance(e, dict) and e.get("results_seen", e.get("after_results")) for e in state.get("design_history") or []):
            return False
        exec_result = state.get("exec_result")
        if (not isinstance(exec_result, dict) or not exec_result or state.get("no_simulation_resolved")
                or state.get("survey_mode_resolved") or self.config.engine.analyze_local_first):
            return False
        if exec_result.get("flat_objective"):
            return False  # the search ran and gave its objective at every design it tried: results were seen
        from . import numeric_oracle as _numeric_oracle

        result = state.get("result_json")
        no_numbers = not any(True for _ in _numeric_oracle.flatten_numbers(result, keep_zero=True))
        return bool(no_numbers or self._reports_no_quantity(state, result))

    def _keep_hypothesis_after_an_empty_run(self, state: QuestState, design: Any) -> Any:
        """The redesign keeps the hypothesis of the design that ran when that run produced no results
        (:meth:`_run_gave_no_results`): a hypothesis is not changed by a run that showed nothing. A reply that changes it has
        that part not taken, and run.log says so. With results, the hypothesis is the model's, recorded as post hoc."""
        previous = state.get("design")
        if not (isinstance(design, dict) and isinstance(previous, dict)) or not state.get("design_history"):
            return design
        before = str(previous.get("hypothesis") or "").strip()
        if not before or str(design.get("hypothesis") or "").strip() == before or not self._run_gave_no_results(state):
            return design
        self._log.warning("[design] the redesign's new hypothesis was not taken: the run before it produced no results, so "
                          "nothing was learned that could change it; the hypothesis stays as it was")
        return {**design, "hypothesis": previous.get("hypothesis")}

    async def _node_design(self, state: QuestState) -> QuestState:
        if self.config.engine.analyze_local_first:
            # --analyze: no experiment to design — the data already exists.
            # Passthrough; routing keys on no_simulation_resolved (set by
            # clarify), so an empty design still flows to data_load.
            self._log.info("[design] analyze_local_first — no experiment design")
            return {}
        iteration = state.get("iteration", 0)
        self._log.info("[design] iteration=%d", iteration)
        _disclosure.clear_objections(self.quest_root)  # a reviewer's objections were to the design this one replaces
        design: dict[str, Any] | None = None
        objections: Any = None
        plan_sha = ""
        # A resume after an amendment request settles it first, and does not ask the model for another design.
        settled = self._settle_pending_amendment(state)
        if settled is not None:
            design = settled
        elif (iteration == 0 and not state.get("design")) or state.get("design_from_plan"):
            # The plan step wrote plan.md and, when asked, stopped for the person to read and edit it. Its design
            # block is the design, exactly: nothing is asked of the model again, and what a person changed runs.
            # Only when no design exists yet: every way back into this node after a result bumps ``iteration``,
            # and a design already in state must never be replaced by the plan on a repair or a refine. The one
            # exception is a plan rewritten for the next design (``design_from_plan``: a study that fits the time FI
            # measured, before anything ran).
            design, plan_sha = self._design_from_plan()
            objections = state.get("design_objections")
        if design is None:
            await self._read_topic_before_the_prompt(state)
            # A redesign changes the parts its reason concerns and carries the rest over unchanged; only when nothing
            # names a part is the whole design asked for again (said in run.log).
            design = await self._redesign_by_parts(state)
            if design is None:
                # The same study-type rules as the plan's prompt: a redesign of a "find the best" topic must be able to
                # write the search block, or it comes out as a plain sweep.
                prompt = self._design_prompt(state) + self._study_type_part(state, seek=self._seeks_best_design(state))
                design = await self._design_reply(prompt, node="design")
            # Before the audit, so the receipt it leaves is for the design that runs (the hypothesis is part of it).
            design = self._keep_hypothesis_after_an_empty_run(state, design)
            design, objections = await self._audit_design(state, design)
            # After the freeze a redesign keeps the frozen protocol; a different one is an amendment request.
            design = self._hold_design_to_frozen(state, design)
            design = self._keep_hypothesis_after_an_empty_run(state, design)

        # A search for the best design never runs as a plain sweep: every design, from the plan or drafted here, passes
        # this point before anything is implemented or run, and one the engine cannot search stops here
        # (core/optimisation_plan.py; the search itself: core/optimise.py).
        if plan_sha:
            # A plan that lacks the search part is completed by the plan's model before this stops (a person is not
            # asked to write it); the design to run is then the plan as rewritten.
            design, plan_sha = await self._ask_plan_for_the_search(state, design, plan_sha)
            design, plan_sha = await self._ask_plan_for_the_parts(state, design, plan_sha)
            design, plan_sha = await self._ask_plan_for_the_grid(state, design, plan_sha)
            design, plan_sha = await self._ask_figures_of_an_edited_plan(state, design, plan_sha)
        else:
            # A design drafted here (a redesign) did not come from plan.md: its search block is completed the same way.
            design, moved_parts = _plan.hoist_protocol_parts(design)
            if moved_parts:
                self._log.warning("[design] the redesign wrote %s beside `protocol`; FI moved %s under it",
                                  ", ".join(f"`{m}`" for m in moved_parts), "it" if len(moved_parts) == 1 else "them")
            design = await self._complete_a_drafted_search(state, design, iteration=iteration)
            design = await self._complete_drafted_parts(state, design, iteration=iteration)
            design = self._inherit_plan_parts(state, design, iteration=iteration)
            design = await self._ask_figures_of_a_redesign(state, design)
            design = await self._ask_design_for_the_grid(state, design, iteration=iteration)
            self._check_the_ledger_of_a_redesign(state, design)
        self._stop_if_no_settings_to_vary(state, design)
        self._stop_if_the_search_cannot_start(state, design)
        audited = await self._audit_the_design_that_runs(state, design)
        if audited is not design:
            # FI changed its own design to meet the audit (written into plan.md): the rest of the node uses that one.
            design = audited
            plan_sha = _plan.sha256(_plan.plan_path(self.quest_root).read_text(encoding="utf-8"))
            self._stop_if_the_search_cannot_start(state, design)
        # The design that will run is held to what the topic asks for (a redesign or an amendment that dropped a setting
        # is caught here); what is still missing and is the question's own stops the quest before anything runs.
        size_before = _study_size(design)
        design, plan_sha = await self._hold_the_design_to_the_topic(state, design, plan_sha)
        if plan_sha and _study_size(design) > size_before:  # what the topic added here made the study larger: counted again
            design, plan_sha = await self._ask_plan_for_the_grid(state, design, plan_sha, again=True)
        elif not plan_sha and _study_size(design) > size_before:  # a redesign: the same, with its own request
            design = await self._ask_design_for_the_grid(state, design, iteration=iteration, again=True)
        design, plan_sha = await self._ask_again_for_a_search_lost_since(state, design, plan_sha)
        # The plan rewritten from what FI measured of a search whose objective did not move: a change FI approves itself,
        # recorded as made after results were seen (the first design's run is archived, the paper is told).
        flat_pass = bool(state.get("design_from_plan") and state.get("flat_objective_fact"))
        if flat_pass:
            design = self._amend_for_flat_objective(state, design)
        elif state.get("design_from_plan"):
            # The plan rewritten to fit the time FI measured, after a protocol frozen for a run that was stopped (a resume of
            # a run cut short, say): FI approves the change itself, or the frozen protocol would still hold the study that
            # did not fit and the smaller one would never run.
            design = self._amend_to_fit(design)
        _audit_open.close_held(self.quest_root, design)  # the design handed on, after its last change in this node
        out: dict[str, Any] = {"design": design, "design_from_plan": False}
        if flat_pass:
            out["flat_objective_fact"] = {}  # used once: a later rewrite of the plan (to fit the time) is not this one
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
        design = out["design"] = self._keep_hypothesis_after_an_empty_run(state, design)  # a later amendment too
        out["design_history"] = _append_design_revision(
            state, design, self.quest_root, self._log, plan_sha,
            results_seen=flat_pass or not self._run_gave_no_results(state),
            reason=(_flat_fact_sentence(state.get("flat_objective_fact") or {}) + "; plan.md was rewritten once from what "
                    "FI measured (results were seen)") if flat_pass else "",
        )
        def _design_record() -> dict[str, Any]:
            latest = (out["design_history"] or [{}])[-1]
            revision = latest.get("revision")
            return {
                "kind": "design",
                "revision": revision,  # as in DESIGN_HISTORY.json: 0 for the first design
                "parent": (revision - 1) if isinstance(revision, int) and revision > 0 else None,
                "reason": latest.get("reason") or "",
                "design_sha256": _attempts._json_sha(design),
            }
        self._last_ledger_id = self._record(_attempts.LEDGER, _design_record) or getattr(self, "_last_ledger_id", None)
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

    def _take_model_change(self) -> None:
        """The quest's model (``provider.model``, ``provider.name`` or the per-step models) differs from
        the one recorded when it was approved or last run: never a stop. The new model is used from here on, recorded
        (``.fi/approved_plan.json``, a ``model_changed`` event in the audit trace, run.log) and said in one plain line
        per change on the console (``[FI] model: ...``, which the VS Code chat shows). The paper is told that more than
        one model produced the quest (plan_settings.model_disclosure). Best-effort: a failed write never stops a
        quest."""
        try:
            changes = _plan_settings.model_changes(self.fi_dir, self.config)
            if not changes:
                return
            _plan_settings.accept_models(self.fi_dir, self.config, self.quest_root)
        except OSError as e:
            self._log.warning("[model] could not record the model change: %r", e)
            return
        # Nothing made on the old model yet (its first start failed before any step, say): nothing to disclose.
        made = self._made_anything()
        lines = _plan_settings.model_change_lines(changes, self.config.provider.node_models, made=made)
        chat = (self.config.provider.extra or {}).get(self.CHAT_MODEL_KEY)
        chat_after = chat.get("after") if isinstance(chat, dict) else None
        try:
            self.__dict__["_model_events_before_run"] = sum(
                1 for e in _audit_log.read(self.audit.path) if e.get("kind") == "model_changed")
        except Exception:  # noqa: BLE001
            pass
        # With the chat panel's model on this run, config.yaml's new model is not what runs: the paper is told of the
        # chat panel's model instead (_say_model_change), not of one that never ran.
        self._audit("model_changed", changes=changes, before_any_step=not made,
                    **({"run_uses": chat_after} if chat_after else {}))
        main = next((c for c in changes if c.get("setting") == "provider.model"), None)
        if main is not None:
            # The first call confirming this very change is not recorded again (_note_served_model).
            self.__dict__["_model_change_taken"] = (str(main.get("from") or ""), str(main.get("to") or ""))
        if chat_after and main is not None:
            # config.yaml's model changed, but this run uses the chat panel's (_say_model_change records that): one
            # sentence, not "from here on B" followed by "not B".
            said = f"the model changes from {main.get('from') or 'none'} to "
            lines = [f"config.yaml's model changes from {main.get('from') or 'none'} to {main.get('to') or 'none'}; "
                     f"this run uses the chat panel's model {chat_after}" if line.startswith(said) else line
                     for line in lines]
        for line in lines:
            self._log.info("[model] %s", line)
            print(f"[FI] model: {line}")

    def _made_anything(self) -> bool:
        """Whether anything was made on the quest's earlier model: a call in ``cost.jsonl`` or a completed step in the
        trace. When unsure, yes (a change is then disclosed rather than hidden)."""
        try:
            cost = self.fi_dir / "cost.jsonl"
            return (cost.is_file() and cost.stat().st_size > 0) or any(
                e.get("kind") == "node_completed" for e in _audit_log.read(self.audit.path))
        except Exception:  # noqa: BLE001 -- when unsure, say it as a change
            return True

    def _stop_for_changed_settings(self, changed: list[str]) -> QuestArtifacts:
        """Stop before anything runs because a setting that decides how strictly the quest is checked differs from
        the one approved when it started (core/plan_settings.py). Nothing is run and nothing is changed: the person
        either puts the setting back, or approves the change with ``--update``, which records it."""
        self._log.warning("[plan] stopped: %d setting(s) changed since the quest was approved: %s",
                          len(changed), "; ".join(changed))
        self._audit("plan_settings_changed", changed=changed)
        self._write_next_step(
            kind="plan_changed",
            interaction="answer",
            headline="settings that decide how strictly this quest is checked changed after you approved it",
            steps=[
                "Nothing was run. Changed since the quest was approved:\n" + "\n".join(f"  - {c}" for c in changed),
                "To keep the change, approve it: `python launch.py --update " + self.quest_id + "` (web: the quest page's "
                "Update, VS Code: `@fi /update " + self.quest_id + "`); it shows the settings and records them.",
                f"To undo it, put the setting back in `{self.quest_root / 'config.yaml'}` (or resume with the config "
                f"you started with), then `python launch.py --resume {self.quest_id}`.",
            ],
        )
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / "pause.json").write_text(json.dumps({
                "kind": "plan_changed", "interaction": "answer", "quest_id": self.quest_id,
                "headline": "settings that decide how strictly this quest is checked changed after you approved it",
                "next_step_file": "NEXT_STEP.md", "upload_targets": [],
            }, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
        print(f"[FI] stopped before running: settings changed since quest {self.quest_id} was approved; see NEXT_STEP.md")
        return self._collect_artifacts({})

    def _write_next_step(
        self,
        *,
        kind: str,
        interaction: str,
        headline: str,
        steps: list[str],
        recommended: str | None = None,
        alternatives: list[str] | None = None,
        card: dict[str, Any] | None = None,
    ) -> "_todo.Item":
        """Write the to-do card (``NEXT_STEP.md`` and ``.fi/todo.json``, :mod:`core.todo`) whenever the quest stops for
        the person: why it stopped, what there is to decide, the recommendation, the alternatives, what to do, and
        everything else waiting, then how to go on. ``interaction`` (answer / supply) is kept in ``pause.json``.
        ``card``: the structured "why it stopped" card (core/oracle_card.py), when the stop has one. Best-effort; a
        write failure never stops a quest."""
        item = _todo.pause_item(kind, headline, steps, recommended=recommended, alternatives=alternatives,
                                profile=getattr(self.config, "rigor_profile", "default"),
                                frozen=_frozen.load(self.quest_root) is not None, card=card)
        _todo.write(self.quest_root, self.fi_dir, self.quest_id, item)
        return item

    def _provider_for_step(self, node: str | None) -> tuple[str, str, str]:
        """``(provider name, model, the provider.node_providers key)`` of the provider that serves ``node``: the step's own
        when it is routed (``provider.node_providers``; the key is ``""`` otherwise), else its model from
        ``provider.node_models``, else the quest's main provider and model. A failure or stop text names this one, never
        the main provider when the step runs on another."""
        first = (node or "").split(",")[0].strip()
        provider = getattr(getattr(self, "config", None), "provider", None)
        step = node_provider_for(getattr(provider, "node_providers", None), first) if first else None
        if step is not None:
            return str(step[1].name), str(step[1].model), str(step[0])
        models = getattr(provider, "node_models", None)
        model = model_for_node(models, first) if first and models else None
        return str(getattr(provider, "name", "") or ""), str(model or getattr(provider, "model", "") or ""), ""

    def _routed_words(self, text: str) -> str:
        """A step's model is set in ``provider.node_providers.<step>`` when the step is routed there (which wins over
        ``provider.node_models``): advice that names the other setting would do nothing. Rewritten to the right one."""
        if not text or "node_models" not in text:
            return text
        providers = getattr(self.config.provider, "node_providers", None)
        if not providers:
            return text

        def dotted(m: Any) -> str:
            hit = node_provider_for(providers, m.group(1))
            return f"`provider.node_providers.{hit[0]}.model`" if hit else m.group(0)

        def snippet(m: Any) -> str:
            hit = node_provider_for(providers, m.group(1))
            return (f"provider: {{node_providers: {{{hit[0]}: {{name: <provider>, model: <model>}}}}}}" if hit
                    else m.group(0))

        text = re.sub(r"`provider\.node_models\.(\w+)`", dotted, text)
        return re.sub(r"provider: \{node_models: \{(\w+): <model>\}\}", snippet, text)

    def _pause_for_human(
        self,
        *,
        kind: str,
        interaction: str,
        headline: str,
        steps: list[str],
        payload: dict[str, Any],
        upload_targets: list[str] | None = None,
        recommended: str | None = None,
        alternatives: list[str] | None = None,
        card: dict[str, Any] | None = None,
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
        steps = [self._routed_words(x) for x in steps]
        recommended = self._routed_words(recommended) if recommended else recommended
        alternatives = [self._routed_words(x) for x in alternatives] if alternatives else alternatives
        self._audit_pause = kind
        self._audit("pause_requested", pause=kind, interaction=interaction, headline=headline)
        item = self._write_next_step(
            kind=kind, interaction=interaction, headline=headline, steps=steps,
            recommended=recommended, alternatives=alternatives, card=card,
        )
        descriptor: dict[str, Any] = {
            "kind": kind,
            "interaction": interaction,
            "headline": headline,
            # The card's parts, so the web page and VS Code show the same card NEXT_STEP.md is (core/todo.py).
            "decide": item.decide,
            "recommended": item.recommended,
            "alternatives": item.alternatives,
            "steps": list(item.steps),
            "quest_id": self.quest_id,
            "next_step_file": "NEXT_STEP.md",
            # For a SUPPLY pause: which upload target(s) the web banner should
            # offer (papers → inputs/papers, data → inputs/data, root_data →
            # data/). Empty for an ANSWER pause (the web reveals the form).
            "upload_targets": list(upload_targets or []),
        }
        if item.card:
            # The structured "why it stopped" card: the web page and VS Code show the same card and offer its actions.
            descriptor["card"] = item.card
        for key in ("problems", "proposed_changes"):
            # What a check found and what a repair proposed, for the surfaces that cannot read the interrupt payload.
            if isinstance(payload.get(key), list) and payload[key]:
                descriptor[key] = list(payload[key])
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
        self._progress(f"Waiting for you: {headline}")
        return interrupt({**payload, "pause": descriptor})

    def _pause_for_model_output(self, step: str, e: Any) -> None:
        """Stop because a model's answer at ``step`` was cut off at its output limit or withheld by a content filter:
        say which call it was and what to change. A SUPPLY pause: resume runs the step again with the new settings."""
        call = str(getattr(e, "node", "") or step)
        key = call.split(".", 1)[0] if call.startswith("review_panel.") else call
        who = " / ".join(str(x) for x in (getattr(e, "provider", None), getattr(e, "model", None)) if x) or "the model"
        truncated = isinstance(e, ModelAnswerTruncated)
        limit = getattr(e, "limit", None)
        pname = str(getattr(e, "provider", None) or self.config.provider.name or "")
        is_cli = pname in _CLI_PROVIDERS_SET
        refused = bool(getattr(e, "refused", False))
        steps_names = (
            "Step names you can use: `plan` (the design), `implement`, `write`, `review`, `review_panel`, "
            "`claim_check`, `cross_check`, `analyze`; a name with a dot (`write.patch`) uses its first part."
        )
        change_model = (
            f"Give that step a model with a larger output: `provider: {{node_models: {{{key}: <model>}}}}` in the "
            "quest's `config.yaml`, then approve the change with "
            f"`python launch.py --update {self.quest_id}` (it resumes the quest too)."
        )
        if truncated and refused:
            headline = f"the model refused the output limit set for the {call} step"
            steps = [
                f"{who} would not take an output limit of {limit} tokens at `{call}`: the model's own maximum is lower, "
                "so the step could not run with room to finish.",
                f"Give `{key}` a smaller limit the model accepts (`provider: {{node_max_tokens: {{{key}: <smaller>}}}}`) "
                "and resume; if that is too small for the answer to fit, use a model with a larger output instead.",
                change_model,
            ]
            recommended = f"Use another model for `{key}`: `provider.node_models.{key}`."
            alternatives = [f"Lower `provider.node_max_tokens.{key}` to what the model accepts, then resume."]
        elif truncated:
            headline = f"the answer at the {call} step was cut off at its output limit"
            used = (getattr(e, "usage", None) or {}).get("completion_tokens")
            base = max(int(used or 0), int(limit or 0))
            suggested = min(base * 2, 65536) if base else 0
            steps = [
                f"The answer {who} gave at `{call}` stopped at its output limit"
                + (f" ({limit} tokens)" if limit else " (the model's own; no limit was set)")
                + ", so it is incomplete and was not used: a cut-off script, plan or paper would be passed on as whole.",
            ]
            if is_cli:
                steps.append(
                    f"`provider.node_max_tokens` does not reach a CLI provider ({pname}). {change_model}"
                )
                recommended = f"Use another model for `{key}`: `provider.node_models.{key}`."
                alternatives = ["Ask for a shorter piece of work (the topic, or the plan with `--revise-plan`), then resume."]
            else:
                ask = (f"{suggested}" if suggested else "<a number larger than the answer needs>")
                steps += [
                    f"Give that step a larger output limit in the quest's `config.yaml`: "
                    f"`provider: {{node_max_tokens: {{{key}: {ask}}}}}`"
                    + (f" (twice the {base} tokens the cut-off answer used)" if suggested else "")
                    + ". If the model refuses that number, lower it: every model has its own maximum. Then resume.",
                    change_model,
                ]
                if "ollama" in pname.lower():
                    steps.append(
                        "Ollama can also report `length` when the context window is full, not only the output limit; "
                        "then `num_ctx` needs raising in Ollama (not checked here)."
                    )
                recommended = f"Raise `provider.node_max_tokens.{key}`, then go on."
                alternatives = [f"Give `{key}` another model: `provider.node_models.{key}`, then go on."]
            steps += [steps_names, "Resuming runs this step again with the new setting; what the quest did before it is kept."]
        else:
            declined = getattr(e, "finish_reason", None) == "refusal"  # the model itself declined (the claude CLI)
            headline = (f"the model declined to answer at the {call} step" if declined
                        else f"the provider withheld the answer at the {call} step with its content filter")
            steps = [
                (f"{who} declined to answer `{call}`, so there is nothing to use. " if declined else
                 f"{who} answered `{call}`, but the provider's content filter withheld the answer, so there is nothing "
                 "to use. ") + "Asking the same model the same thing again gives the same result.",
                f"Give that step another model: `provider: {{node_models: {{{key}: <model>}}}}` in the quest's "
                f"`config.yaml`, then approve the change with `python launch.py --update {self.quest_id}` (it resumes "
                "the quest too).",
                "Or change what the quest asks (the topic, or the plan with `--revise-plan \"...\"`) if its wording is "
                + ("what the model declined" if declined else "what the filter refused") + ", then resume.",
            ]
            recommended = f"Give `{key}` another model: `provider.node_models.{key}`, then go on."
            alternatives = ["Change the topic or the plan's wording, then go on."]
        self._log.warning("[%s] %s: %s", step, headline, e)
        self._pause_for_human(
            kind="model_output", interaction="supply", headline=headline, steps=steps,
            recommended=recommended, alternatives=alternatives,
            payload={"quest_id": self.quest_id, "step": call, "setting": key,
                     "reason": "truncated" if truncated else "content_filtered"},
        )

    async def _design_reply(self, prompt: str, *, node: str) -> dict[str, Any]:
        """The design object a model answers ``prompt`` with (the ``plan`` and ``design`` steps). An answer that cannot be
        read is asked for once more (:meth:`_chat_json`); one that cannot be read the second time stops the quest
        (:meth:`_stop_for_unreadable_design`): an empty design is never run."""
        text, obj = await self._chat_json(
            prompt, node=node, want=("hypothesis",), usable=lambda p: any(k != "plan" for k in p))
        if not any(k != "plan" for k in obj):
            self._stop_for_unreadable_design(node, text)
        return obj

    def _stop_for_unreadable_design(self, step: str, text: str) -> None:
        """Stop because the model's design at ``step`` could not be read, twice. A SUPPLY pause: resume asks again."""
        headline = f"the model's design at the {step} step could not be read, even when asked twice"
        steps = [
            f"Nothing was run. The model's answer at the `{step}` step could not be read as a design, and asking it once "
            "more for one short answer gave a reply that could not be read either. FI has no design to run, and it does "
            "not run an empty one.",
            "What FI tried: it read the reply as written, read it again with the backslashes of formulas kept as "
            "written, and asked the model a second time.",
            f"Give that step another model: `provider: {{node_models: {{{step}: <model>}}}}` in the quest's "
            f"`config.yaml`, then approve the change with `python launch.py --update {self.quest_id}` (it resumes the "
            "quest too). Or resume as it is to ask the same model again.",
        ]
        self._log.warning("[%s] the design could not be read, even when asked twice (%d characters); stopping, nothing "
                          "was run", step, len(text or ""))
        print(f"[FI] quest {self.quest_id}: the model's design at the {step} step could not be read, even when asked "
              "twice; nothing was run (see NEXT_STEP.md)")
        self._pause_for_human(
            kind="model_unreadable", interaction="supply", headline=headline, steps=steps,
            recommended=f"Use another model for `{step}`: `provider.node_models.{step}`, then go on.",
            alternatives=["Resume as it is: the same model is asked again."],
            payload={"quest_id": self.quest_id, "step": step, "setting": step, "reason": "unreadable"},
        )

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
        resume re-enters this cheap node, not the search.

        Then the papers' figures are read (``_read_literature_figures``), here for the same reason: a stop to change
        the figure model re-enters this node, not the search."""
        merged = list(state.get("literature") or [])
        added = 0
        if self._pause_stage_enabled("after_literature"):
            self._maybe_pause_for_user_input(state, "after_literature")
            seen = {i for entry in merged for i in _entry_identities(entry)}
            merged, added = await asyncio.to_thread(  # a scanned paper is read by OCR here
                _ingest_user_dropped_papers, self.quest_root, merged, seen, self._log,
            )
            if added:
                self._log.info(
                    "[after_literature] picked up %d paper(s) dropped in inputs/papers/ "
                    "while paused", added,
                )
        read = await self._read_literature_figures(state, merged)
        if not added and not read:
            return {}
        await self._flag_sources(merged, "after_literature")  # the papers added and the readings appended
        if read:
            self._write_literature_files(merged, self.quest_root / "data" / "literature")
        return {"literature": merged}

    async def _node_scope(self, state: QuestState) -> QuestState:
        """Before the plan, the few different studies the topic could mean, for a person to pick one
        (:mod:`core.scope_cards`). Runs once per quest: a record in ``.fi/scope.json`` (picked or not) is read, never
        asked again. With nobody to answer the cards are still recorded, and the plan is written from the topic alone."""
        if (self.config.pauses.scope == "off" or self.config.engine.analyze_local_first
                or int(state.get("iteration", 0) or 0) > 0 or _plan.plan_path(self.quest_root).is_file()):
            return {}
        record = _scope_cards.read_record(self.quest_root)
        if record is not None:
            chosen = record.get("chosen")
            return {"scope_card": chosen} if isinstance(chosen, dict) else {}
        sources = _labelled_sources(state.get("literature") or [])
        ids = {f"[{label}]" for label, _m, _i in sources} | {label for label, _m, _i in sources}
        listing = "\n".join(
            f"[{label}] {meta.get('title') or meta.get('source') or 'untitled'} — "
            + _one_line(_item_content(item), 300) for label, meta, item in sources) or "(none)"
        again = ""
        cards: list[dict[str, Any]] = []
        removed: list[str] = []
        for _ in range(2):
            prompt = self._prompts["scope"].substitute(
                topic=state.get("topic") or self.config.topic, answers=_format_clarify(state),
                direction=json.dumps(state.get("chosen_idea") or {}, indent=2), sources=listing, again=again)
            reply = await self._chat(prompt, node="scope")
            cards, removed = _scope_cards.normalize(_parse_json_lenient(reply) or reply, ids)
            if _scope_cards.distinct_enough(cards):
                break
            again = (f"Your last reply gave {len(cards)} different studies"
                     + (": " + "; ".join(c["question"] for c in cards) if cards else "")
                     + ". Two cards with the same study type and the same main results are one study. "
                       "Write 3 or 4 studies that differ in their study type or in their main results.")
        chosen = await self._scope_answer(cards) if cards else None
        try:
            _scope_cards.write_record(self.quest_root, cards, chosen, removed, widest=_scope_cards.has_widest(cards))
        except OSError as exc:
            self._log.warning("[scope] could not write %s (%r); the choice is kept for this run only",
                              _scope_cards.RECORD, exc)
        if chosen is not None:
            self._log.info("[scope] the study to run: %s", chosen.get("question", ""))
        else:
            self._log.info("[scope] nobody picked the study to run; the plan is written from the topic alone")
        return {"scope_card": chosen}

    async def _scope_answer(self, cards: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The card a person picked: an answer staged in ``.fi/scope_answer.json`` (read now), else the answer channel
        the clarify questions use, waited for up to ``pauses.timeout_s`` (0: no limit). ``None``: nobody picked."""
        path = self.quest_root / _scope_cards.ANSWER
        if path.is_file():
            try:
                return _scope_cards.read_answer(json.loads(path.read_text(encoding="utf-8")), cards)
            except (OSError, ValueError) as exc:
                self._log.warning("[scope] could not read %s (%r)", _scope_cards.ANSWER, exc)
        ask = getattr(self, "_scope_answerer", None)
        if ask is None:
            return None
        try:
            waiting = ask(_scope_cards.question(cards))
            limit = float(self.config.pauses.timeout_s)
            answer = await (asyncio.wait_for(waiting, timeout=limit) if limit > 0 else waiting)
        except (asyncio.TimeoutError, BridgeError) as exc:
            self._log.warning("[scope] no answer to the study cards (%s)", type(exc).__name__)
            return None
        return _scope_cards.read_answer(answer, cards)

    async def _read_literature_figures(self, state: QuestState, literature: list[Any]) -> int:
        """Read the values off the papers' figures (``knowledge.read_figures``), in place: the model for the step
        ``figures`` picks, from the captions, the figures that show values this question can use, then reads those,
        ``_FIGURES_PER_CALL`` images a call. A figure picked and read gets ``relevant`` and ``reading``; one not picked
        gets ``relevant: False``; one whose reading failed twice is left unjudged, to be tried on a later pass. Every
        decision is written at once to ``.fi/figure_readings.json`` (by the image's hash), so a stop part-way through
        never pays for a figure twice. A source whose figures were read gets them appended to its text. A model that
        cannot take images stops the quest (a SUPPLY pause: resume re-enters this node). Returns how many were read."""
        if not self.config.knowledge.read_figures:
            return 0
        ledger_path = self.fi_dir / "figure_readings.json"
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
        except (OSError, ValueError):
            ledger = {}
        ledger = ledger if isinstance(ledger, dict) else {}

        def key_of(fig: dict[str, Any], item: dict[str, Any]) -> str:
            # The image and the source and figure it belongs to: the same image in two sources is two decisions.
            meta = item.get("metadata") or {}
            return "|".join(str(x) for x in (fig.get("sha256") or fig.get("image"), meta.get("doi") or meta.get("url")
                                               or meta.get("title"), fig.get("number")))

        def save_ledger() -> None:
            try:
                self.fi_dir.mkdir(parents=True, exist_ok=True)
                tmp = ledger_path.with_name(f"{ledger_path.name}.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(ledger, indent=1, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, ledger_path)
            except OSError as e:
                self._log.info("[figures] the reading ledger could not be saved (%r)", e)

        todo: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for idx, item in enumerate(literature, start=1):
            meta = item.get("metadata") if isinstance(item, dict) else None
            for fig in (meta or {}).get("figures") or []:
                if not isinstance(fig, dict):
                    continue
                known = ledger.get(key_of(fig, item))
                if isinstance(known, dict) and "relevant" not in fig:  # judged on an earlier pass of this quest
                    fig["relevant"] = bool(known.get("relevant"))
                    if known.get("reading"):
                        fig["reading"] = str(known["reading"])
                if "relevant" not in fig and Path(str(fig.get("image") or "")).is_file():
                    todo[f"{idx}:{fig.get('number')}"] = (item, fig)
        # Readings the ledger brought back (a stop part-way through a pass): counted, so the node returns them.
        read = sum(_append_figure_readings(item) for item in literature if isinstance(item, dict))
        if not todo:
            return read
        topic = _lit_query(state)[:1200]
        wanted: list[str] = []
        ids = list(todo)
        for start in range(0, len(ids), _FIGURE_PICK_BATCH):
            chunk = ids[start:start + _FIGURE_PICK_BATCH]
            lines = [f"[{fid}] {str(todo[fid][0]['metadata'].get('title') or '')[:120]} :: "
                     f"{str(todo[fid][1].get('caption') or '')[:400]}" for fid in chunk]
            try:
                raw = await self._chat(
                    self._prompts["figures_pick"].substitute(
                        topic=topic, figures=_source_text.fence("\n".join(lines), "papers' titles and figure captions"),
                    ), node="figures",
                )
                picked = (_parse_json_lenient(raw, node="figures") or {}).get("pick")
            except Exception as e:  # noqa: BLE001 -- these figures stay unjudged; the quest goes on
                self._log.info("[figures] could not pick among %d figure(s) (%r); left for a later pass", len(chunk), e)
                continue
            if not isinstance(picked, list):
                self._log.info("[figures] the pick reply named no list; %d figure(s) left for a later pass", len(chunk))
                continue
            chosen = {str(p) for p in picked} & set(chunk)
            for fid in chunk:
                if fid in chosen:
                    wanted.append(fid)
                else:
                    todo[fid][1]["relevant"] = False
                    ledger[key_of(todo[fid][1], todo[fid][0])] = {"relevant": False}
            save_ledger()
        self._log.info("[figures] %d of %d figure(s) picked to read (model: %s)", len(wanted), len(todo),
                       self._model_for_node("figures") or "the main model")
        from core.provider import ImageInputUnsupported, image_part

        for start in range(0, len(wanted), _FIGURES_PER_CALL):
            batch = wanted[start:start + _FIGURES_PER_CALL]
            content: list[dict[str, Any]] = [{"type": "text", "text": self._prompts["figures_read"].substitute(
                count=len(batch), topic=topic)}]
            try:
                for fid in batch:
                    item, fig = todo[fid]
                    content.append({"type": "text", "text": f"[{fid}] {str(item['metadata'].get('title') or '')[:120]} "
                                                            f":: {str(fig.get('caption') or '')[:600]}"})
                    content.append(image_part(Path(str(fig["image"])).read_bytes()))
            except OSError as e:
                self._log.info("[figures] %d figure image(s) could not be read from disk (%r); left for a later pass",
                               len(batch), e)
                continue
            pending = list(batch)
            for attempt in (1, 2):  # a failed call, an unreadable reply or a figure left out is tried once more
                if attempt == 2:
                    content = [content[0]] + [part for i, fid in enumerate(batch) if fid in pending
                                              for part in content[1 + 2 * i:3 + 2 * i]]
                try:
                    raw = await self._chat_messages([{"role": "user", "content": content}], node="figures",
                                                    temperature=0.0)
                except ImageInputUnsupported as e:
                    save_ledger()
                    self._pause_for_human(
                        kind="figures",
                        interaction="supply",
                        headline="the model for reading figures cannot read images",
                        steps=[
                            f"{len(wanted) - start} figure(s) from the papers show values this question can use, "
                            f"but the model cannot take images ({e}). The figures are saved in "
                            "`data/literature/figures/`; the ones already read are kept.",
                            "To have them read: in the quest's `config.yaml` set a model that reads images for this "
                            "step, `provider: {node_models: {figures: <model>}}`, then resume.",
                            "To go on without reading them: set `knowledge: {read_figures: false}`, then resume.",
                        ],
                        payload={"quest_id": self.quest_id, "figures_waiting": len(wanted) - start},
                    )
                    return read  # unreachable: a SUPPLY pause does not return
                except Exception as e:  # noqa: BLE001 -- retried once, then left for a later pass
                    self._log.info("[figures] reading %d figure(s) failed (attempt %d: %r)", len(pending), attempt, e)
                    continue
                parsed = _parse_json_lenient(raw, node="figures")
                for entry in (parsed.get("figures") or []) if isinstance(parsed, dict) else []:
                    if not isinstance(entry, dict) or str(entry.get("id")) not in pending:
                        continue
                    reading = str(entry.get("reading") or "").strip()
                    if not reading:
                        continue
                    item, fig = todo[str(entry["id"])]
                    fig["relevant"], fig["reading"] = True, reading[:_FIGURE_READING_CHARS]
                    ledger[key_of(fig, item)] = {"relevant": True, "reading": fig["reading"]}
                    pending.remove(str(entry["id"]))
                    read += 1
                if not pending:
                    break
                self._log.info("[figures] %d figure(s) not read on attempt %d", len(pending), attempt)
            save_ledger()
        for item in {id(item): item for item, _fig in todo.values()}.values():
            _append_figure_readings(item)
        self._log.info("[figures] read %d figure(s); their values are in each source's text", read)
        return read

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
