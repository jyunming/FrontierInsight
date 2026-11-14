"""Methods of :class:`core.engine.Engine` for writing the paper, its claims and its citations.

``Engine`` inherits :class:`WriteStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, Callable, DEFAULT_PAPER_FORMAT, NON_SCIENTIFIC_PAPER_FORMATS, PRELIMINARY_NOTE, Path, QuestState,
    SCIENTIFIC_PAPER_FORMATS, _CITATION_SOURCE_CHARS, _CLAIM_EVIDENCE_CHARS, _CLAIM_FAULTS_TAKEN_OUT,
    _CODE_REEXECUTES, _ModelAnswerProblem, _NO_SIM_WRITE_NOTE, _PAGE_LIMIT_HIT, _PAPER_PROMPT_CHARS, _REFINE_POINT_RE,
    _REFINE_WHY_RE, _SKILLS_IN_CODE, _SKILL_MOUNTS_FILE, _SOURCE_LIST_HEADING_RE, _SURVEY_WRITE_NOTE, _TEXT_ONLY_HITS,
    _TRIM_ROUNDS, _UNANSWERED_NOTE_HIT, _accepted, _analysis_frame, _apply_title_only_rule, _attempts, _audit_log,
    _audit_open, _best_report, _cite_labels, _citing_sentences, _claim_choice, _claim_choice_problem,
    _claim_faults_from_sources, _claim_keys, _claim_method_block, _claim_results_block, _claim_source_block,
    _code_project, _code_review, _cs, _disclosure, _discover_skill_names, _equation_tests, _experiment_deps,
    _extension_values, _figure_facts, _figure_label_findings, _figure_list_for_prompt, _figure_plan, _figures_drawn,
    _finalize_paper_sources, _finalize_with_numbering, _format_clarify, _format_cross_check, _format_evidence_note,
    _format_lit_excerpt, _format_lit_from_state, _format_review_for_writer, _foundational_sources,
    _foundational_write_block, _frozen, _further_reading_block, _grounding_after_trim, _has_full_text, _hit_name,
    _improve, _is_not_a_paper, _item_content, _labelled_sources, _literature_key, _mark_preliminary, _metric_spec,
    _novelty, _objection_key, _one_line, _optim, _optimise, _optimise_refine, _page_limit_note, _paper_basis,
    _paper_facts, _paper_figure_count, _paper_for_prompt, _paper_scope, _paper_sections, _paper_values,
    _parse_json_lenient, _passage_stands, _phased, _place_figures, _places_as_placeholders, _plan_settings,
    _quote_in_source, _raise_if_required_skills_unusable, _read_json_or_none_path, _receipts, _record_claims, _refine,
    _refine_count, _refine_round, _reorder_literature, _replicate_result_intervals, _rerun_from,
    _resolve_selected_skills, _results_by_setting, _retractions, _review_items, _review_sends_the_experiment_back,
    _run_facts, _same_statement, _second_impl, _skeleton, _skill_mount_plan, _source_text, _split_figure_readings,
    _split_run, _stop_power, _strip_outer_fence, _take_out_unpointed, _take_refine_points, _take_refine_why,
    _topic_req, _trial_table, _trim_further_reading, _uncited_foundational_works, _unsupported_still_in,
    _web_labels_cited, _words_to_cut, asyncio, build_further_reading, build_references, goal_coverage, json, os,
    paper_patch, paper_trim, pip_failure_summary, re, resolve_page_limit, shutil, web_pages_listed,
)


class WriteStageMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    async def _write_whole_paper(
        self, state: QuestState, persona_block: str, *, refine_round: bool = False, extra_note: str = "",
        scope_block: str = "",
    ) -> str:
        """The paper's markdown as the writer gives it: the whole paper, written
        from the study, with the review of an earlier draft (when there is one)
        in the prompt. A first draft always comes from here, and so does a round
        whose request FI cannot map to sections (:meth:`_write_sections`).

        With ``scope_block`` (the earlier draft, read-only, and the sections to write) the reply is those sections
        only, not a paper: the caller puts them back into the draft (core/paper_scope.py)."""
        # When the evidence gate judged the evidence thin (insufficient,
        # or "broaden" with the broaden budget exhausted), hand the writer
        # an explicit note so the paper frames its limits honestly instead
        # of over-claiming. Suppressed for survey topics (see the helper).
        evidence_note = _format_evidence_note(
            state.get("evidence_assessment"),
            is_survey=bool(state.get("survey_mode_resolved")),
        )
        amended = _frozen.disclosure(self.quest_root)
        if amended:
            evidence_note = f"{evidence_note}\n\n{amended}".strip()
        not_confirmed = _accepted.disclosure(
            self.quest_root, self._not_confirmed_names(state, self._protocol_block(state)))
        if not_confirmed:
            evidence_note = f"{evidence_note}\n\n{not_confirmed}".strip()
        # A known-answer check failed and the quest went on (FI's automatic decision, or a person's earlier choice).
        went_on_note = _accepted.failing_disclosure(self._oracle_record_read())
        if went_on_note:
            evidence_note = f"{evidence_note}\n\n{went_on_note}".strip()
        # An expected value FI corrected in the plan (from the plan's own arithmetic or another model's, never from the
        # measurement): the paper says so where the checks are described.
        corrected = self._corrections_read()
        if corrected:
            def rests_on(c: dict[str, Any]) -> str:
                if c.get("source") == "arithmetic" and c.get("confirmed_by"):
                    return ("the plan's own arithmetic, confirmed by another model never shown the measurement or the "
                            "plan's working")
                if c.get("source") == "arithmetic":
                    return "the plan's own arithmetic alone (not confirmed independently)"
                if c.get("source") == "numeric":
                    return ("a script that computed it from the model's definitions alone, never shown the plan's "
                            "working or the measurement, whose number agrees with the simulation (two computations "
                            "from the same definitions, not confirmed independently of the model)")
                if c.get("source") == "challenge":
                    return ("the code's objection to the plan's working, settled by the plan's model, a third model or "
                            "a numerical computation, none shown the measurement")
                return "another model's value alone (not confirmed independently)"

            listed = "; ".join(f"'{n}' from {c.get('from')} to {c.get('to')}, resting on {rests_on(c)} ({c.get('why')})"
                               for n, c in corrected.items() if isinstance(c, dict))
            evidence_note = (f"{evidence_note}\n\nFI corrected the expected value of these known-answer checks in the "
                             f"plan before the run, never from the measurement: {listed}. "
                             "Say so where the checks are described.").strip()
        # The model was changed during the quest: the paper says more than one model produced it.
        try:
            models_note = _plan_settings.model_disclosure(_audit_log.read(self.audit.path))
        except Exception:  # noqa: BLE001 -- an unreadable trace is reported by the evidence check, not here
            models_note = ""
        if models_note:
            evidence_note = f"{evidence_note}\n\n{models_note}".strip()
        if _phased.enabled(self.config) and (phased_note := _phased.write_note(self.quest_root)):
            evidence_note = f"{evidence_note}\n\n{phased_note}".strip()
        improved = _improve.write_note(self.quest_root)
        if improved:
            evidence_note = f"{evidence_note}\n\n{improved}".strip()
        if _optimise.block_of(self._protocol_block(state)) is not None and (best_note := _best_report.write_note(
                self.quest_root)):
            evidence_note = f"{evidence_note}\n\n{best_note}".strip()
        if once_note := self._ran_once_note():
            evidence_note = f"{evidence_note}\n\n{once_note}".strip()
        if _refine.settings(self._protocol_block(state)) and (refined := _refine.lines(_refine.read(self.quest_root))):
            evidence_note = (f"{evidence_note}\n\nFI ran the headline results again at finer numerical settings "
                             "(needs/REFINEMENT_CHECK.json); the paper says what this found:\n"
                             + "\n".join(f"- {line}" for line in refined)).strip()
        if depends := _second_impl.dependence_sentences(_second_impl.read_dependence(self.quest_root)):
            evidence_note = (f"{evidence_note}\n\nFI changed the result of each tested function of the model where "
                             "the simulation calls it, to see whether the headline results are computed with it "
                             "(.fi/second_impl/dependence.json); say this plainly where the paper says what was checked:\n"
                             + "\n".join(f"- {line}" for line in depends)).strip()
        if typed_note := self._typed_figures_note():
            evidence_note = f"{evidence_note}\n\n{typed_note}".strip()
        if audit_note := _audit_open.writer_note(self.quest_root):
            evidence_note = f"{evidence_note}\n\n{audit_note}".strip()
        if code_note := _code_review.write_note(self.quest_root, self.quest_root / "code"):
            evidence_note = f"{evidence_note}\n\n{code_note}".strip()
        if topic_note := _topic_req.write_note(self.quest_root):
            evidence_note = f"{evidence_note}\n\n{topic_note}".strip()
        if core_note := _equation_tests.write_note(self._protocol_block(state)):
            evidence_note = f"{evidence_note}\n\n{core_note}".strip()
        if run_note := self._run_checks_note():
            evidence_note = f"{evidence_note}\n\n{run_note}".strip()
        if drafted_note := self._drafted_parts_note():
            evidence_note = f"{evidence_note}\n\n{drafted_note}".strip()
        try:
            trial_rows = _trial_table.write(self.quest_root, _skeleton.diagnostic_names(self._protocol_block(state)),
                                            _metric_spec.proportion_ids(self._protocol_block(state)))
            try:  # every setting with its reference value (core/results_by_setting.py), for a table too long to give
                by_setting = _results_by_setting.write(
                    self.quest_root, self._protocol_block(state), self._judged_checks(),
                    {lb for lb, _meta, _item in _labelled_sources(state.get("literature") or [], self.config.output.audience)})
            except Exception as e:  # noqa: BLE001 -- the same: a help, never a reason to stop
                self._log.warning("[write] FI's table of every setting could not be written (%s)", e)
                by_setting = None
                # An earlier run's tables must not go into this paper.
                (self.quest_root / _results_by_setting.PAPER_BLOCK).unlink(missing_ok=True)
            for d in (by_setting.diagnostics if by_setting else []):  # FI's own reading of the code: the log, not the paper
                self._log.info("[write] %s", d)
            table_note = _results_by_setting.writer_note(trial_rows, by_setting)
        except Exception as e:  # noqa: BLE001 -- the table is a help to the writer, never a reason to stop
            self._log.warning("[write] FI's table of the trials could not be written (%s)", e)
            table_note = ""
        try:  # the claim check reads what the writer was given
            _results_by_setting.keep_writer_note(self.quest_root, table_note)
        except OSError as e:
            self._log.warning("[write] FI could not keep the tables the writer was given (%s)", e)
        if table_note:
            evidence_note = f"{evidence_note}\n\n{table_note}".strip()
        record_note = self._paper_record_section(state)
        if record_note:
            evidence_note = f"{evidence_note}\n\n{record_note}".strip()
        missed = [str(p) for p in state.get("extend_missed") or [] if str(p).strip()]
        if missed:
            evidence_note = (
                f"{evidence_note}\n\nThe reader asked for this to be added: {'; '.join(p[:200] for p in missed)}. "
                "It could not be added, so the numbers in this paper do not include it. Say so plainly (in the "
                "limitations), and do not write a value for it."
            ).strip()
        layout_missed = [str(p) for p in state.get("layout_missed") or [] if str(p).strip()]
        if layout_missed:
            evidence_note = (
                f"{evidence_note}\n\nThe reader asked for this about the figures: {'; '.join(p[:200] for p in layout_missed)}. "
                "Redrawing the figures could not do it (it needs a change in the paper's text or in the order of its "
                "figures, or there was nothing to redraw from). If the text can do it, do it in the text; otherwise say "
                "plainly (in the limitations) that it was not done."
            ).strip()
        if extra_note:
            evidence_note = f"{evidence_note}\n\n{extra_note}".strip()
        prompt = self._prompts["write"].substitute(
            persona_block=persona_block,
            topic=state["topic"],
            title=(
                f"The user chose this title; use it as the paper's title: {state.get('title')}"
                if state.get("title_confirmed") and state.get("title")
                else f"Filename slug (you author the paper's title): {state.get('title', 'Untitled')}"),
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
            novelty_block=_novelty.write_block(state.get("literature_novelty")),
            figure_list=_figure_list_for_prompt(state, root=self.quest_root),
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
            review_feedback=_format_review_for_writer(state, refine_round=refine_round),
            skills_block=(
                self._writing_skills_block(state)
                or "(no writing guidance selected for this quest)"
            ),
            scope_block=scope_block,
        )
        markdown = await self._chat(prompt, node="write")
        # The model may wrap with a fence; strip it.
        markdown = _strip_outer_fence(markdown)
        if not scope_block and _is_not_a_paper(markdown):
            raise RuntimeError(
                f"the writer returned {len(markdown.strip())} characters, which "
                f"cannot be a paper: {markdown.strip()[:200]!r}. If that reads "
                f"like a provider message (a usage limit, an expired login), fix "
                f"the provider and resume the quest."
            )
        return markdown

    def _paper_record_section(self, state: QuestState) -> str:
        """The placeholders the paper's facts of the run go through, and the ``not_done`` ones the Limitations must hold,
        for the writer (core/paper_facts.py). The precision record is kept first (it is one of the items), and the code's
        seed verdict is put on record (``.fi/run_facts.json``). A record FI cannot read leaves its part out."""
        try:
            _disclosure.write_precision_record(self.quest_root, self._precision_shortfalls(state))
        except Exception as e:  # noqa: BLE001 -- the rest of the record is still given
            self._log.warning("[write] the record of the precision reached could not be kept (%s: %s)",
                              type(e).__name__, str(e)[:200])
        try:
            verdict = _run_facts.record_seed(self.quest_root)
            if verdict.get("verdict") in (_run_facts.DOES_NOT_REACH, _run_facts.UNKNOWN):
                self._log.warning("[write] the seed FI gives each trial %s: %s",
                                  "does not reach the simulation" if verdict["verdict"] == _run_facts.DOES_NOT_REACH
                                  else "may not reach the simulation (FI cannot tell)", verdict.get("why"))
        except Exception as e:  # noqa: BLE001 -- the paper's run paragraph reads the code again when it is filled
            self._log.warning("[write] the code's seed could not be put on record (%s: %s)", type(e).__name__, str(e)[:200])
        try:
            for note in _run_facts.shared_stream_notes_for(self.quest_root):  # measured from the trials; a note only
                self._log.info("[write] %s", note)
        except Exception as e:  # noqa: BLE001 -- a note FI cannot compute is left out
            self._log.warning("[write] the trials could not be compared for shared random numbers (%s: %s)",
                              type(e).__name__, str(e)[:200])
        try:
            section = _paper_facts.writer_section(
                self.quest_root, topic=str(self.config.topic or ""),
                fallback_s=(state.get("exec_result") or {}).get("duration_s"))
        except Exception as e:  # noqa: BLE001 -- a record FI cannot read is left out, never a reason to stop
            self._log.warning("[write] FI's record of the run could not be given to the writer (%s: %s)",
                              type(e).__name__, str(e)[:200])
            section = ""
        sheet = self._values_sheet(state)
        return f"{section}\n\n{sheet}".strip() if sheet else section

    def _paper_records(self, state: QuestState) -> Any:
        """The trial record and plan the paper's result keys are filled from (core/paper_values.py); ``None`` without a
        trial record or when it cannot be read."""
        try:
            rec = _paper_values.records_of(self.quest_root, self._protocol_block(state))
            if rec is not None:
                rec.topic = str(self.config.topic or "")  # a {{calc:...}} may use the topic's numbers, as R1 allows them
            return rec
        except Exception as e:  # noqa: BLE001 -- the keys then read "not recorded"; the paper is not held to the rule
            self._log.warning("[write] FI's trial record could not be read for the paper's result numbers (%s: %s)",
                              type(e).__name__, str(e)[:200])
            return None

    def _values_sheet(self, state: QuestState) -> str:
        """The "Results you can place" block the writer is given (the rule in plain words and the sheet of keys,
        core/paper_values.py), and the note that this draft is written under it (``self._sheet_given``). ``""`` for a study
        with no trial record: its paper is not held to the rule."""
        sheet = _paper_values.writer_block(self._paper_records(state))
        self._sheet_given = bool(sheet)
        if not sheet:
            self._log.info("[write] this study has no per-setting trial record, so FI fills in no result number and the "
                           "paper is not held to the rule that the writer types none (QUALITY.md says so)")
        return sheet

    def _render_paper_facts(self, markdown: str, state: QuestState) -> str:
        """``markdown`` with every placeholder of the run's facts filled from FI's records (core/paper_facts.py); a
        placeholder FI cannot fill reads "not recorded" and is said in the run log. Never raises: a failure leaves the
        placeholders as they are, and the review's check then finds them unfilled."""
        try:
            out = _paper_facts.render(markdown, self.quest_root, topic=str(self.config.topic or ""),
                                      fallback_s=(state.get("exec_result") or {}).get("duration_s"),
                                      records=self._paper_records(state), plan=_figure_plan(self.quest_root))
        except Exception as e:  # noqa: BLE001
            self._log.warning("[write] FI could not fill the paper's facts of the run from its records (%s: %s)",
                              type(e).__name__, str(e)[:200])
            return markdown
        if out.unfilled:
            self._log.warning("[write] FI has no record for %s in the paper; it reads \"not recorded\" there (listed in "
                              "QUALITY.md)", ", ".join("{{" + u + "}}" for u in out.unfilled))
        _paper_values.write_record(self.quest_root, list(out.values))
        for v in out.values:
            if v.hit:
                self._log.warning("[write] %s", v.hit.text())
        return out.text

    def _restate_limitations(self, state: QuestState, why: str, *, strict: bool = False) -> bool:
        """The paper's Limitations are FI's block (``{{limitations}}``, core/paper_facts.py), written from its records as
        they are now: called after a step that changed one of them, so the paper that is delivered says what the records
        say. FI's own text and no claim of its own, so the claim check's and the citation check's receipts move to the
        paper that carries it (``why`` is the reason they keep). No model call; best-effort. Returns whether the paper changed."""
        path = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
        if not path or not Path(path).is_file():
            return False
        try:
            before = Path(path).read_text(encoding="utf-8")
            after = self._render_paper_facts(
                _paper_facts.place_blocks(_disclosure.without_block(before), self.quest_root,
                                          topic=str(self.config.topic or ""),
                                          fallback_s=(state.get("exec_result") or {}).get("duration_s"),
                                          limitations_heading=self._limitations_heading(state)), state)
            if after == before:
                return False
            checked = Path(path).read_bytes()
            Path(path).write_text(after, encoding="utf-8")
            for check in ("claim_check", "citation_check"):
                _receipts.carry_over(self.quest_root, check, "paper", checked, Path(path).read_bytes(), why)
            return True
        except (OSError, UnicodeDecodeError) as e:
            if strict:
                raise
            self._log.warning("[review] the paper's Limitations could not be written again from FI's records (%s)", e)
            return False

    def _verify_value_marks(self, markdown: str, state: QuestState) -> str:
        """``markdown`` with every FI value mark worked out again from FI's record (core/paper_values.py::verify_marks): a
        value a later round changed is put back, said in the run log and kept for QUALITY.md; a mark whose key FI can no
        longer work out is left as it is and listed as unverified. Never raises."""
        try:
            check = _paper_values.verify_marks(markdown, self._paper_records(state))
        except Exception as e:  # noqa: BLE001 -- the render that follows still fills every key from the record
            self._log.warning("[review] FI's value marks could not be checked again (%s: %s)", type(e).__name__, str(e)[:200])
            return markdown
        for r in check.repaired:
            self._log.warning("[review] a value FI had filled was changed inside its mark, so FI's value was put back: "
                              "%s was \"%s\", now \"%s\"", r["key"][:120], r["was"][:60], r["now"][:60])
        for r in check.unverified:
            self._log.warning("[review] FI cannot work out %s again, so its text \"%s\" is not checked (%s)",
                              r["key"][:120], r["text"][:60], r["why"][:160])
        _paper_values.keep_marks_record(self.quest_root, check)
        if check.checked:
            self._log.info("[review] %d value mark(s) checked against FI's record: %d put back, %d not checkable",
                           check.checked, len(check.repaired), len(check.unverified))
        return check.text

    def _refresh_paper_facts(self, state: QuestState) -> None:
        """Before a paper is checked: a paper on disk that still carries FI's old blocks (a quest resumed from before they
        left the paper) or a placeholder not filled is cleaned and filled again from the records, so the checks read what
        a reader will. Nothing is written when nothing changes."""
        path = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
        if not path or not Path(path).is_file():
            return
        try:
            before = Path(path).read_text(encoding="utf-8")
            before = self._verify_value_marks(before, state)
            placed = _paper_facts.place_blocks(
                _disclosure.without_block(before), self.quest_root, topic=str(self.config.topic or ""),
                fallback_s=(state.get("exec_result") or {}).get("duration_s"),
                limitations_heading=self._limitations_heading(state))
            after = self._render_paper_facts(placed, state)
            if before.rstrip() != after.rstrip() and (
                    _disclosure.without_block(before) != before or "{{" in before or "fi:fact-" in before
                    or "fi:fact-" in after or "fi:v " in before or "fi:v " in after):
                Path(path).write_text(after, encoding="utf-8")
                self._log.info("[review] took FI's old blocks out of the paper and filled its facts of the run from the "
                               "records again before checking it")
        except (OSError, UnicodeDecodeError) as e:
            self._log.warning("[review] the paper's facts of the run could not be filled again (%s)", e)

    def _paper_record_hits(self, paper_md: str, state: QuestState) -> list[str]:
        """Must-fix hits for the paper against FI's records (core/paper_facts.py): a required placeholder missing (the seed,
        the run time, whether the run kept to the topic's limit; each thing the study did not do, in the Limitations) and,
        for a paper written under the rule that FI fills every result number, a result key FI could not fill and a number
        the writer typed that is not a number of the plan or the topic (core/typed_numbers.py). Text-only: the paper is
        written again, the design is not touched and the quest is not stopped.

        When no review round is left to write the paper again, each typed number still in it stays as the writer wrote it and
        is listed in ``.fi/paper_record_check.json`` (QUALITY.md prints it) as not traced to the plan or FI's records: a number
        may be right though FI cannot trace it, so it is never deleted. What else is still unmet is kept there too, and said
        in the run log."""
        keyed = bool(state.get("paper_keys"))
        topic = str(self.config.topic or "")
        fallback = (state.get("exec_result") or {}).get("duration_s")
        protocol = self._protocol_block(state)
        hits = _paper_facts.check(paper_md, self.quest_root, topic=topic, fallback_s=fallback, keyed=keyed,
                                  protocol=protocol)
        left = int(self.config.engine.max_iterations) - int(state.get("iteration", 0) or 0) - 1
        # A draft the whole-paper writer wrote gets one edit of those passages first (_typed_pass_hits): nothing is recorded
        # as untraced or taken out until that edit has been made and checked.
        deferred = bool(self._typed_pass_hits(paper_md, state))
        untraced: list[tuple[str, str]] = []
        if keyed and left <= 0 and not deferred:
            untraced = _paper_facts.untraced_typed(
                paper_md, _paper_facts.plan_protocol(self.quest_root, protocol), topic, self.quest_root)
            if untraced:
                hits = _paper_facts.check(paper_md, self.quest_root, topic=topic, fallback_s=fallback, keyed=keyed,
                                          protocol=protocol, typed=False)
                self._log.warning("[record_check] no round is left to write the paper again, so %d number(s) the writer typed "
                                  "that FI cannot trace to the plan or its records stay as written and are listed in "
                                  "QUALITY.md", len(untraced))
        # A typed number FI cannot trace stays in the paper and is listed above as untraced. A result key FI could not fill
        # is different: once no round is left to fix it, it is the only thing taken out (a sentence that carried only such
        # keys goes, any other place reads the same marker) and listed.
        dropped: list[dict[str, str]] = []
        # A figure statement FI could not fill, and a sentence that points at a figure FI drew and is not FI's to stand behind
        # (core/figure_claims.py::apply_pointer_rule), go the same way.
        if keyed and left <= 0 and not deferred and (_paper_facts.unfilled_in(paper_md) or self._figures_drawn_safely(state)):
            dropped = self._take_out_unfilled_keys(state)
            if dropped:
                path = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
                paper_md = Path(path).read_text(encoding="utf-8")
                hits = _paper_facts.check(paper_md, self.quest_root, topic=topic, fallback_s=fallback, keyed=keyed,
                                          protocol=protocol)
        # A paper not held to the rule says why in QUALITY.md: the study has no trial record to fill from, or the paper
        # was written before FI filled result numbers.
        not_filled = "" if keyed else (
            "older_paper" if (self.quest_root / "raw" / "trials.json").is_file() else "no_records")
        _paper_facts.keep_check(self.quest_root, hits, paper_md, untraced=untraced, not_filled=not_filled, dropped=dropped)
        if any(str(d.get("key", "")).startswith("figure:") for d in dropped) and self._restate_limitations(
                state, "FI said in the Limitations that statements about a figure were left out and added no claim"):
            # The Limitations are written before the take-out: the figure's disclosure reaches the paper in this same write.
            paper_md = Path(_rerun_from.in_quest(state.get("paper_md"), self.quest_root)).read_text(encoding="utf-8")
            hits = _paper_facts.check(paper_md, self.quest_root, topic=topic, fallback_s=fallback, keyed=keyed,
                                      protocol=protocol)
            _paper_facts.keep_check(self.quest_root, hits, paper_md, untraced=untraced, not_filled=not_filled, dropped=dropped)
        for hit in hits:
            self._log.warning("[record_check] %s", hit)
        if hits and left <= 0 and deferred:
            self._log.info("[record_check] no review round is left, so the sentences with numbers FI did not fill are "
                           "edited once more (outside engine.max_iterations) before any is recorded as untraced")
        elif hits and left <= 0:
            self._log.warning("[record_check] %d point(s) of FI's record are still wrong or missing in the paper and no review "
                              "round is left to rewrite it; they are in .fi/paper_record_check.json and QUALITY.md", len(hits))
        return hits

    def _typed_pass_hits(self, paper_md: str, state: QuestState) -> list[str]:
        """The hits of the one edit a draft gets before FI records its untraced numbers or takes out unfilled keys: when no review round is left, a draft
        under the rule that FI fills every result number, written by the whole-paper writer (not by this edit), with a
        number the writer typed that R1 does not allow or a key FI could not fill. ``[]`` otherwise.

        Every such case in real quests followed a whole-paper rewrite in the last round (to shorten the paper, after the study
        was run again, for a hit no passage holds): the new draft typed numbers and no round was left to fix them. Edited
        once, with FI's record of the run and ``{{calc:...}}`` at hand, almost all of them come back as FI's values or in
        words; a typed number still there after the edit stays in the paper and is recorded as untraced, and only an
        unfilled result key is taken out (:meth:`_paper_record_hits`). One edit per whole-paper draft,
        and those are capped (engine.max_iterations, the shortening and record rewrites), so this always ends."""
        if not state.get("paper_keys") or state.get("draft_edited"):
            return []
        if int(self.config.engine.max_iterations) - int(state.get("iteration", 0) or 0) - 1 > 0:
            return []
        try:
            topic = str(self.config.topic or "")
            plan = _paper_facts.plan_protocol(self.quest_root, self._protocol_block(state))
            hits = _paper_facts.typed_hits(paper_md, plan, topic, self.quest_root) + _paper_facts.value_hits(paper_md, self.quest_root)
            # Only a hit that quotes its sentence can be edited (the passage editor refuses a round with any other, such as
            # the count of typed numbers past the first sentences); the numbers of the rest are recorded as untraced after the edit.
            return [h for h in hits if paper_patch._QUOTED in h]
        except Exception as e:  # noqa: BLE001 -- no edit: the untraced numbers and unfilled keys are handled as before
            self._log.warning("[record_check] the sentences to edit once more could not be found (%s: %s)",
                              type(e).__name__, str(e)[:200])
            return []

    def _take_out_unfilled_keys(self, state: QuestState, *, figures_only: bool = False) -> list[dict[str, str]]:
        """The paper with each result key FI could not fill taken out (core/paper_facts.py::take_out_unfilled), written to
        disk: ``key``, ``sentence`` and ``how`` for each. Nothing is put in their place, so no number is invented, and the
        claim check's receipt moves to the new draft, which only lost text and added no claim.

        ``figures_only`` (every write, not only the last round): only the statements about a figure, a figure statement FI
        could not fill and a sentence that points at a figure FI drew and is not FI's to stand behind; the result keys wait
        for the last round. The paper is a new draft then, with no receipt or claim list to move."""
        path = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
        if not path or not Path(path).is_file():
            return []
        try:
            before = Path(path).read_bytes()
            text, taken = _paper_facts.take_out_unfilled(before.decode("utf-8"), figures_only=figures_only)
            try:
                text, pointed = _take_out_unpointed(text, state, root=self.quest_root)
            except Exception as e:  # noqa: BLE001 -- the pointer rule never fails the write node: the paper stays as written
                self._figure_pass_failed(e)
                pointed = []
            taken = taken + pointed
            if not taken:
                return []
            Path(path).write_bytes(text.encode("utf-8"))  # bytes back as bytes (bytes back as bytes: text mode writes the draft's CRLF as CR CR LF on Windows)
            if not figures_only:
                _receipts.carry_over(self.quest_root, "claim_check", "paper", before, Path(path).read_bytes(),
                                     "FI took out values it could not fill and added no claim")
                self._prune_ledger_after_take_out([str(t.get("sentence") or "") for t in taken], text)
        except (OSError, UnicodeDecodeError) as e:
            self._log.warning("[record_check] the values FI could not fill could not be taken out of the paper (%s)", e)
            return []
        if not figures_only:
            self._log.warning("[record_check] no round is left to write the paper again, so %d value(s) FI could not fill were "
                              "taken out (%d sentence(s) removed, the rest replaced by \"%s\") and are listed in QUALITY.md",
                              len(taken), sum(1 for t in taken if t["how"] == "sentence removed"), _paper_facts.REPLACED)
        for stem, number in _paper_facts._figure_left_out_items(taken):
            gone = [str(t.get("sentence") or "") for t in taken
                    if str(t.get("key", "")).startswith(f"figure:{stem}:") or t.get("figure") == number]
            self._log.warning("[record_check] statements about Figure %s (%s) could not be checked against what it draws and "
                              "were left out (%d): %s; the Limitations say so", number, stem, len(gone),
                              "; ".join('"' + g[:80] + '"' for g in gone))
        return taken

    def _take_out_figure_statements(self, state: QuestState, paper_path: Path) -> None:
        """Every paper the write node writes (the first draft, a review's rewrite, a person's refine) has the sentences about
        a figure FI cannot stand behind taken out before anything reads it, whatever round it is: the review, the slides and
        the poster all read the paper as it is left here. What was left out is kept for the Limitations, which are written
        again in the same write (the record of an earlier paper is replaced, so nothing older is said)."""
        self.__dict__.pop("_figure_pass_error", None)
        at = {**state, "paper_md": str(paper_path)}
        try:
            taken = self._take_out_unfilled_keys(at, figures_only=True)
            if _paper_facts.keep_figure_left_out(self.quest_root, taken):
                self._restate_limitations(at, "FI said in the Limitations which statements about a figure were left out and "
                                              "added no claim")
        except Exception as e:  # noqa: BLE001 -- the figure pass never fails the write node: the paper stays as written
            self._figure_pass_failed(e)
        # The record describes the paper as finally written: a fault of an earlier write's pass is gone when this one ran clean.
        failed = self.__dict__.get("_figure_pass_error") is not None
        try:
            if failed or _paper_facts.keep_figure_pass_fault(self.quest_root, False):
                if not failed:
                    self._clear_cannot_repair("figure_pass_fault")
                self._restate_limitations(at, "FI said in the Limitations whether the sentences about the figures were "
                                              "checked and added no claim")
        except Exception as e:  # noqa: BLE001 -- nor does saying so: the record stays for QUALITY.md and the next write
            self._log.warning("[record_check] the Limitations could not be written again to say whether the sentences about "
                              "the figures were checked (%s: %s)", type(e).__name__, e)

    def _figures_drawn_safely(self, state: QuestState) -> dict[int, set[str]]:
        """:func:`_figures_drawn` for the write node; an error in it is the figure pass's fault, never the node's."""
        try:
            return _figures_drawn(state, root=self.quest_root)
        except Exception as e:  # noqa: BLE001
            self._figure_pass_failed(e)
            return {}

    def _figure_pass_failed(self, error: BaseException) -> None:
        """The check of the sentences about the figures stopped with an error of FI's own: the paper stays as it was written,
        run.log has the error, and QUALITY.md and the next Limitations say once, plainly, that those sentences were not checked."""
        self._log.warning("[write] the check of the sentences about the figures stopped (%r); the paper stays as written", error)
        self.__dict__["_figure_pass_error"] = repr(error)
        _paper_facts.keep_figure_pass_fault(self.quest_root, True)
        try:
            self._record_cannot_repair(
                "FI could not check the sentences about the figures; the paper keeps them as written",
                ["the check of the sentences about the figures stopped with an error of FI's own"],
                source=_stop_power.READ, check="figure_pass_fault")
        except Exception:  # noqa: BLE001 -- a disclosure that cannot be written is already in run.log
            pass

    async def _check_code_project(self) -> None:
        """Run ``code/`` as a person would (clean environment, only requirements.txt) once per version of the code and
        say the outcome in plain words. A failure is a warning only: it never changes the result or stops the quest."""
        if self.config.execution.sandbox != "venv" or self.config.execution.background_jobs:
            return
        try:
            digest = json.dumps(_attempts.script_hashes(self.quest_root), sort_keys=True)
            marker = self.fi_dir / "code_project_checked.json"
            if marker.is_file() and marker.read_text(encoding="utf-8") == digest:
                return
            limit = float(min(self.config.execution.clean_check_timeout_s, self.config.execution.timeout_s))
            self._progress(
                f"Trying code/ in a clean environment (at most {limit / 60:g} min; it does not change the results)"
            )
            if getattr(self, "_code_runs", None) is not None:  # the clean check runs code/run.py itself, outside
                await self._code_runs.before_run()  # the executor
            result = await asyncio.to_thread(_code_project.verify, self.quest_root, run_timeout_s=limit)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(digest, encoding="utf-8")
            (self._log.info if result.get("ok") or result.get("status") == "not_finished" else self._log.warning)(
                "[code] %s", result.get("says"))
        except Exception as exc:  # noqa: BLE001 -- a check of a convenience, never a reason to stop
            self._log.warning("[code] could not try code/ in a clean environment: %s", exc)

    async def _node_write(self, state: QuestState) -> QuestState:
        await self._check_code_project()
        persona_block = self._resolve_write_persona(state)
        self._log.info(
            "[write] authoring paper.md (persona=%s)",
            persona_block.split("\n", 1)[0][:80] if persona_block else "default",
        )
        # A revise whose every must-fix hit names a passage edits those passages
        # of the earlier draft and leaves the rest as it was; anything else
        # (a first draft, a hit about the whole paper, edits the engine cannot
        # apply) writes the whole paper.
        refine_round = _refine_round(state)
        # A person's notes are answered by the whole-paper writer, which reads them (the passage editor does not).
        # A request the run could not carry out is told to the whole-paper writer, which the passage editor is not.
        reported = bool(state.get("extend_missed") or state.get("layout_missed"))
        # No round is left and the review held back the record of numbers FI did not fill (_typed_pass_hits): only those
        # sentences are edited, and a draft that cannot be edited stays as it was (a whole new paper would type new ones).
        # Not when the review sent the paper back for a contradiction of FI's records (that rewrite answers it, and the
        # review after it asks for this edit again), nor for a request the run could not carry out, nor on a draft that
        # was itself edited (a review's hits left in the state are not asked for twice).
        review_now = state.get("review") or {}
        typed_pass = [] if (refine_round or reported or state.get("draft_edited") or any(
            _hit_name(h) == _record_claims.HIT for h in review_now.get("must_flag_hits") or [])) \
            else list(review_now.get("typed_pass_hits") or [])
        if typed_pass:
            self._log.info("[write] editing the %d sentence(s) with numbers FI did not fill, once, before any is recorded as untraced",
                           len(typed_pass))
            markdown = await self._patch_flagged_passages(
                {**state, "review": {**(state.get("review") or {}), "must_flag_hits": typed_pass}}, persona_block,
                otherwise="the draft is kept as it was and the review takes those numbers out")
            if markdown is None:
                markdown = self._kept_draft(state) or None
        else:
            markdown = None if refine_round or reported else await self._patch_flagged_passages(state, persona_block)
        edited = markdown is not None
        # A draft edited from an earlier one is under the rule its earlier draft was; a whole paper is under it when the
        # writer was given the sheet of keys (a study with no trial record has none, and an older paper is never held to it).
        keyed = bool(state.get("paper_keys")) if markdown is not None else False
        if markdown is None and not reported:
            # A round that answers a review or a person's notes writes only the sections it touches (_write_sections);
            # the rest of the earlier draft is kept as it was.
            markdown = await self._write_sections(state, persona_block, refine_round=refine_round)
            # The sections kept are under the rule their earlier draft was; the ones written now are checked like any.
            keyed = bool(state.get("paper_keys")) if markdown is not None else False
        if markdown is None:
            self._sheet_given = False
            markdown = await self._write_whole_paper(state, persona_block, refine_round=refine_round)
            keyed = bool(getattr(self, "_sheet_given", False))
        # Whatever the writer put where the engine's paragraph goes is not the paper's text: out before any step reads
        # the draft (its citations, figures and numbers), and the engine's own goes in at the end (core/disclosure.py).
        markdown = _best_report.without_block(_disclosure.without_block(markdown))
        needs_experiment: list[str] = []
        extend: list[str] = []
        layout: list[str] = []
        route_why = ""
        if refine_round:
            markdown, route_why = _take_refine_why(markdown)
            markdown, points = _take_refine_points(markdown)
            needs_experiment, extend, layout = points["experiment"], points["data"], points["layout"]
            if extend and (needs_experiment or state.get("survey_mode_resolved")):
                # A survey has no script and no data to add to: what is missing is found by looking at the literature again.
                needs_experiment, extend = needs_experiment + extend, []
            if needs_experiment:
                what = ": %d point(s) need a new experiment, so the quest goes back to the design (%s)" % (
                    len(needs_experiment), "; ".join(p[:80] for p in needs_experiment))
            elif extend:
                what = ": %d number(s) are missing, so the existing script is extended and run again (%s)" % (
                    len(extend), "; ".join(p[:80] for p in extend))
            elif layout:
                what = ": the figures are arranged again from the saved data, the experiment is not run again (%s)" % (
                    "; ".join(p[:80] for p in layout))
            else:
                what = " in the text; the experiment stands"
            self._log.info("[write] answered the person's notes%s", what)
            # The writer's own reason for the route its answer takes (asked for in the same reply, no extra call).
            route = ("a new experiment" if needs_experiment else "more data" if extend
                     else "a new figure layout" if layout else "text only")
            self._claim("write", "write", "refine_route", route_why, decision=route)
        unreported: list[str] = []
        reasked = False
        if not refine_round:
            asked_of = markdown
            markdown, unreported, reasked = await self._report_what_the_extension_added(state, persona_block, markdown)
            if markdown is not asked_of:
                edited = False  # the whole-paper writer wrote it again for what the extension added
        from generation._keywords import keep_one_keywords_form

        # A scientific paper has an Abstract, whichever pass wrote this draft (first, revise or shortening): its structure is
        # checked here, not its wording, and a draft without one has it written from the draft itself.
        if not persona_block:
            markdown = await self._ensure_abstract(state, markdown)
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
        _cite_labels.write_removed(self.quest_root, [str(n) for n in dropped])  # listed in QUALITY.md
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
        # The figures are FI's: every figure of the run goes in under the number the writer was given, with the caption
        # FI writes from its record of what the figure draws, and every image the draft wrote itself is taken out. No
        # model call; the same on every write, so a rewrite gets the same figures back.
        markdown, placed, written = _place_figures(markdown, state, facts=_figure_facts.facts(self.quest_root), root=self.quest_root)
        if placed or written:
            self._log.info("[write] FI placed %d figure(s) with captions from its records (%s); the draft's own images "
                           "taken out: %d", len(placed), ", ".join(placed) or "-", written)
        markdown = (_mark_preliminary(markdown) if self.config.effective_result_use == "explore"
                    else markdown.replace("\n" + PRELIMINARY_NOTE + "\n", "\n"))
        if _phased.enabled(self.config):
            markdown = _phased.mark_paper(markdown, _phased.load(self.quest_root))
        # The design history, the run record and the code's open findings are not in the paper: the writer places what the
        # study did not do with FI's placeholders (core/paper_facts.py), the review checks it, and the quality report
        # (QUALITY.md) holds the records themselves. A block an earlier draft carried was taken out above.
        # A search for the best design: the engine's "Best design found" section, from FI's own records.
        markdown = self._mark_best_design(state, markdown)
        # FI's tables of every setting (core/results_by_setting.py): where the writer put {{results_table}}, else before
        # the discussion; filled from FI's records with the facts below, so no number of them is typed by a model.
        markdown = _paper_facts.place_blocks(
            markdown, self.quest_root, topic=str(self.config.topic or ""),
            fallback_s=(state.get("exec_result") or {}).get("duration_s"),
            limitations_heading=self._limitations_heading(state))
        # A new draft is not yet measured against the page limit: an earlier draft's measurement is not this one's.
        (self.quest_root / _paper_facts.PAGE_RECORD).unlink(missing_ok=True)
        # The new draft exists (the writer answered): what was taken out of the earlier draft is not stated about this one.
        # Before this point a writer that fails leaves the earlier draft and its record as they were.
        # The record is cleared for the render and put back if the new draft is not written.
        record_file = self.quest_root / _paper_facts.UNSUPPORTED_RECORD
        record_before = record_file.read_bytes() if record_file.is_file() else None
        _paper_facts.forget_removed_unsupported(self.quest_root)
        try:
            # The facts of the run go in last, from FI's records, so every check after this reads the values, never a
            # placeholder (core/paper_facts.py).
            markdown = self._render_paper_facts(markdown, state)
            # A figure shares its paragraph with text (a run-in label, a sentence): pandoc then draws it inline, with no
            # caption and no float. Each goes into a paragraph of its own here, so the PDF, slides and poster all read it so.
            from generation._figure_captions import figures_in_own_paragraph

            markdown, n_figures_moved = figures_in_own_paragraph(markdown)
            if n_figures_moved:
                self._log.info("[write] %d figure(s) moved into a paragraph of their own", n_figures_moved)
            paper_path = self.quest_root / "paper" / "paper.md"
            paper_path.write_text(markdown, encoding="utf-8")
        except BaseException:
            if record_before is None:
                record_file.unlink(missing_ok=True)
            else:
                record_file.write_bytes(record_before)
            raise
        self._take_out_figure_statements(state, paper_path)
        self._log.info("[write] wrote %s (%d bytes)", paper_path, len(markdown))
        # Generic pause-drop-anytime: when configured, pause AFTER
        # paper.md lands so the user can drop reference papers (for
        # the next revise pass to cite) and/or datasets (for analyze
        # on the next iteration to incorporate). On a post-pause
        # resume, ``user_pauses_fired`` carries "before_review" and the
        # gate falls through to review.
        self._maybe_pause_for_user_input(state, "before_review")
        out: QuestState = {
            "paper_md": str(paper_path), "literature": literature,
            "paper_basis": _paper_basis(state), "paper_keys": keyed,
            # An edited draft had its chance to fix the numbers FI did not fill; what is left stays and is recorded as
            # untraced (an unfilled result key is taken out).
            "draft_edited": edited,
            "refine_needs_experiment": needs_experiment,
            "refine_extend": extend if not needs_experiment else [],
            # A layout note waits through an extended run (its figures are drawn again after the paper is written).
            "refine_layout": layout if refine_round else list(state.get("refine_layout") or []),
        }
        if not refine_round and not needs_experiment and not extend and not state.get("figure_labels_redrawn"):
            # Figure text that shows the code's name for a quantity (core/figure_labels.py): the figures are drawn
            # again from the saved numbers, once per script (a new script starts it over), by the same redraw as a
            # person's layout note. The experiment is not run again and no repair attempt is spent.
            labels = _figure_label_findings(state, self.quest_root / "code", root=self.quest_root)
            if labels:
                out["refine_layout"] = list(out["refine_layout"]) + labels
                out["figure_labels_redrawn"] = True
                self._log.info("[write] %d figure label(s) show the code's name for a quantity; the figures are "
                               "drawn again from the saved numbers to rename them", len(labels))
        if reported:
            out["extend_missed"], out["layout_missed"] = [], []
        # A new refine starts over: an extension it asks for records its own check (``_node_implement``).
        out["extend_unreported"] = unreported
        if reasked:
            out["extend_check"] = {**(state.get("extend_check") or {}), "reasked": True}
        if refine_round:
            out["extend_check"] = {}
            out["refine_written_for"] = _refine_count(state)
            out["refine_scope"] = ("experiment" if needs_experiment else "data" if extend
                                   else "layout" if layout else "paper")
        else:
            out["refine_scope"] = ""
            if any(_hit_name(h) == _UNANSWERED_NOTE_HIT for h in (state.get("review") or {}).get("must_flag_hits") or []):
                # A rewrite for a note the review said the paper leaves unanswered: flagged again, it goes to the design.
                out["feedback_rewrite_for"] = _refine_count(state)
        return out

    async def _report_what_the_extension_added(
        self, state: QuestState, persona_block: str, markdown: str,
    ) -> tuple[str, list[str], bool]:
        """The paper, what it still leaves out, and whether the writer was asked again, after a script was extended
        for a person's refine.

        The results the extension added for the request must be in the paper (``number_provenance.unreported_results``,
        no model call). A draft that leaves them out is written once more with the missing results named (once per
        extension: ``extend_check["reasked"]``); when that one leaves them out too, the run log and the review pause say
        so plainly instead of leaving it to the review (a real quest computed the number and its paper never said it;
        the review sent the whole round back to the person). Nothing happens without an extension, or when the paper
        has what was asked."""
        check = state.get("extend_check")
        result = state.get("result_json")
        if not isinstance(check, dict) or not check.get("asked") or not isinstance(result, dict) or not result:
            return markdown, [], False
        from core import number_provenance

        asked = [str(p).strip() for p in check.get("asked") or [] if str(p).strip()]
        before = check.get("before") or {}
        missing = number_provenance.unreported_results(markdown, result, before, asked)
        if not missing:
            return markdown, [], False
        request = "; ".join(p[:200] for p in asked)

        def said(names: list[str]) -> list[str]:
            return [f'Your request "{request[:300]}" was computed ({", ".join(names)}) but is not in the paper']

        if check.get("reasked"):
            # The writer was asked once for this extension already: say it, without another model call.
            self._log.warning("[write] %s; the values are in the run's results", said(missing)[0])
            return markdown, said(missing), False
        groups, _grown, _old = number_provenance.added_results(result, before)
        computed = "; ".join(_extension_values(g, groups.get(g) or []) for g in missing)
        self._log.info(
            "[write] the paper leaves out what the person asked for and the run computed (%s); asking the writer "
            "once more", ", ".join(missing),
        )
        note = (
            f"The reader asked for this: {request}. The experiment was extended and run again, and it computed it: "
            f"{computed}. Your draft does not report it. Report these values in the paper (as a table when there are "
            "several), rounded as the results allow, and say in a sentence what they show. If they equal values the "
            "paper already reports under another name, say so in words."
        )
        still = missing
        try:
            again = await self._extension_sections(state, persona_block, markdown, note)
            if again is None:
                again = _disclosure.without_block(await self._write_whole_paper(state, persona_block, extra_note=note))
            still = number_provenance.unreported_results(again, result, before, asked)
            if len(still) < len(missing):
                markdown = again
            else:
                still = missing  # no better: the first draft stays
        except Exception as exc:  # noqa: BLE001 -- the draft that exists is kept; what it leaves out is said below
            self._log.warning("[write] the writer could not be asked again (%s); the draft is kept", exc)
        if not still:
            return markdown, [], True
        self._log.warning(
            "[write] %s, also after the writer was asked again; the values are in the run's results", said(still)[0],
        )
        return markdown, said(still), True

    async def _ensure_abstract(self, state: QuestState, markdown: str) -> str:
        """``markdown`` with an ``## Abstract`` (core/paper_sections.py). A draft without one is not sent whole again: the
        Abstract alone is asked for, from the draft, once; a reply that is empty or fails leaves the draft as it was and the
        log says so."""
        if not _paper_sections.missing_abstract(markdown):
            return markdown
        self._log.warning("[write] the draft has no Abstract; asking for it from the draft")
        try:
            reply = await self._chat(self._prompts["write_abstract"].substitute(
                topic=state.get("topic") or self.config.topic, paper=markdown), node="write.abstract")
        except Exception as exc:  # noqa: BLE001 -- the draft that exists is kept
            self._log.warning("[write] the Abstract could not be written (%s); the draft is kept without one", exc)
            return markdown
        lines = [ln for ln in str(reply).strip().splitlines() if not ln.lstrip().startswith("#")]
        text = " ".join(" ".join(lines).split())
        if len(text.split()) < 40:
            self._log.warning("[write] the reply was not an Abstract; the draft is kept without one")
            return markdown
        return _paper_sections.with_abstract(markdown, text)

    def _kept_draft(self, state: QuestState) -> str:
        """The draft on disk as the passage editor would have been given it (FI's values as their placeholders), for a write
        that keeps it as it was; ``""`` when there is none."""
        try:
            path = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
            return _paper_facts.to_placeholders(Path(path).read_text(encoding="utf-8")) if path else ""
        except (OSError, UnicodeDecodeError):
            return ""

    def _patch_record_block(self, state: QuestState) -> str:
        """What the passage editor is given of FI's records: FI's record of the run (its placeholders, such as
        ``{{settings}}``) and the sheet of result keys, as the whole-paper writer is given them, so a number the check names
        can be put back through FI's own placeholder instead of being worded away."""
        try:
            section = _paper_facts.writer_section(
                self.quest_root, topic=str(self.config.topic or ""),
                fallback_s=(state.get("exec_result") or {}).get("duration_s"))
        except Exception as e:  # noqa: BLE001 -- a record FI cannot read is left out, never a reason to stop
            self._log.warning("[write] FI's record of the run could not be given to the editor (%s: %s)",
                              type(e).__name__, str(e)[:200])
            section = ""
        sheet = self._values_sheet(state) if state.get("paper_keys") else ""
        return (f"{section}\n\n{sheet}".strip() or
                "(none: this paper was not written with FI's result keys)")

    def _earlier_draft(self, state: QuestState) -> tuple[str, str]:
        """``(on disk, with FI's placeholders)``: the draft the last write left in the quest, as a reader has it and as the
        writer works on it. ``("", "")`` when there is none."""
        try:
            if state.get("paper_md"):
                disk = Path(_rerun_from.in_quest(state["paper_md"], self.quest_root)).read_text(encoding="utf-8")
                return disk, _paper_facts.to_placeholders(disk)
        except (OSError, UnicodeDecodeError):
            pass
        return "", ""

    def _sections_with_changed_values(self, disk: str, state: QuestState) -> set[int] | None:
        """The sections of ``disk`` that cite a value of FI's record that is not the value the draft shows, found by filling
        the draft's values in again from the records as they are now and comparing the sections (core/paper_scope.py). It
        compares FI's values, not words. ``None`` when that cannot be done."""
        try:
            now = _paper_facts.render(
                disk, self.quest_root, topic=str(self.config.topic or ""),
                fallback_s=(state.get("exec_result") or {}).get("duration_s"), records=self._paper_records(state)).text
            return _paper_scope.changed(disk, now)
        except Exception as e:  # noqa: BLE001 -- the sections are then chosen by the request alone, or the paper is written whole
            self._log.warning("[write] FI could not tell which sections cite a changed value (%s: %s)",
                              type(e).__name__, str(e)[:200])
            return None

    def _scope_of_round(
        self, state: QuestState, previous: str, disk: str, *, refine_round: bool,
    ) -> tuple[list[_paper_scope.Section], int, dict[int, list[str]], dict[int, int], str]:
        """``(sections, end, why each is asked, words to cut each to, reason)`` for a round that answers a review or a
        person's notes: the sections the round touches, or the ``reason`` it touches the whole paper (a request FI cannot
        place in a section is not guessed at). A section is asked because a check quotes or locates a passage in it, because
        the request names it by its heading or quotes it, because it cites a value of FI's record that changed since the
        draft, or (a page limit) because it is long enough to give words (core/paper_scope.py)."""
        review = state.get("review") or {}
        start = _SOURCE_LIST_HEADING_RE.search(previous)
        end = start.start() if start else len(previous)
        sections = _paper_scope.split(previous, end)
        asked: dict[int, list[str]] = {}
        targets: dict[int, int] = {}
        if not previous.strip():
            return sections, end, asked, targets, "there is no earlier draft on disk"
        if len(sections) < 2:
            return sections, end, asked, targets, "the earlier draft has no sections to tell apart"
        if len(previous) > _PAPER_PROMPT_CHARS:
            return sections, end, asked, targets, (
                f"the draft is longer than the {_PAPER_PROMPT_CHARS:,} characters a model reads")

        def add(index: int, why: str) -> None:
            asked.setdefault(index, [])
            if why not in asked[index]:
                asked[index].append(why)

        # Values of FI's record that changed since the draft: the sections that cite them.
        changed = self._sections_with_changed_values(disk, state) if disk else None
        study_changed = state.get("paper_basis") != _paper_basis(state)
        if study_changed and (changed is None or not state.get("paper_keys")):
            return sections, end, asked, targets, (
                "the study the draft was written from has been run again and FI cannot tell which sections cite a result "
                "that changed")
        for index in sorted(changed or ()):
            add(index, "it cites a value of FI's record that is not the value the draft shows")
        # What the review asks for. Every request is placed by an id FI issued (core/paper_scope.py) or by a passage FI itself
        # quoted; a request with neither cannot be placed, and the paper is then written whole.
        hits = [str(h).strip() for h in review.get("must_flag_hits") or [] if str(h).strip()]
        grounding = state.get("claim_grounding") or {}
        unsupported = [c for c in grounding.get("claims") or [] if isinstance(c, dict) and c.get("basis") == "unsupported"]
        unplaced: list[str] = []
        names = {_hit_name(h) for h in hits}
        if "unsupported_claim" in names:
            if not unsupported:
                unplaced.append("unsupported claims the check recorded without ids")
            for c in unsupported:
                where = _paper_scope.place(previous, sections, c.get("where"))
                if where is None:
                    unplaced.append(f"the claim \"{str(c.get('claim') or '')[:60]}\" (no valid id)")
                for at in sorted(where or ()):
                    add(at, "the check found a claim in it unsupported")
        quoted = [h for h in hits if _hit_name(h) in ("paper_record_mismatch", "mislabelled_statistic")]
        if quoted:
            passages, why = paper_patch.plan_passages(
                previous, end, hits=[(_hit_name(h), h) for h in quoted], claims=[])
            if why:
                unplaced.append(why)
            for q in passages:
                at = _paper_scope.holding(previous, sections, q.located) if q.how == "exact" else None
                if at is None:
                    unplaced.append(f"the {q.kind} \"{q.flagged[:60]}\"")
                else:
                    add(at, f"FI's check flags a {q.kind} in it")
        places = review.get("hit_places") if isinstance(review.get("hit_places"), dict) else {}
        for h in hits:
            name = _hit_name(h)
            if name in paper_patch.PATCHABLE_HITS:
                continue
            if name in ("figure_missing", "citations_unchecked"):
                unplaced.append(name)
            else:
                bare = re.sub(r"^\[[^\]]*\]\s*", "", h)
                where = _paper_scope.place(previous, sections, places.get(h) or places.get(bare))
                if where is None:
                    unplaced.append(f"\"{h[:60]}\" (no valid id)")
                for at in sorted(where or ()):
                    add(at, f"the review's point is about a passage in it: {h[:140]}")
        if refine_round:
            for h in state.get("feedback_history") or []:
                if not isinstance(h, dict) or not str(h.get("text") or "").strip() or h.get("answered_by"):
                    continue
                note = str(h["text"]).strip()
                named = _paper_scope.naming(previous, sections, note)
                if not named and not changed:
                    unplaced.append(f"the person's note \"{note[:60]}\" (it names no section and no value that changed)")
                for index in sorted(named):
                    add(index, "the person's note names it")
        if unplaced:
            return sections, end, {}, {}, "FI cannot place in a section: " + "; ".join(dict.fromkeys(unplaced))
        if not asked:
            return sections, end, asked, targets, "no request names a section and no value of FI's record changed"
        return sections, end, asked, targets, ""

    async def _write_sections(
        self, state: QuestState, persona_block: str, *, refine_round: bool = False,
    ) -> str | None:
        """The earlier draft with only the sections this round touches written again, or ``None`` when the round writes the
        whole paper (a first draft, a request FI cannot place in a section, a reply that cannot be put back; the log says
        which). The rest of the draft is kept byte for byte. Where :meth:`_patch_flagged_passages` edits sentences, this
        writes sections, for the rounds whose request is bigger than a sentence."""
        if not (state.get("review") or refine_round):
            return None  # a first draft
        disk, previous = self._earlier_draft(state)
        sections, end, asked, targets, why = self._scope_of_round(state, previous, disk, refine_round=refine_round)
        if why:
            self._log.info("[write] writing the whole paper: %s", why)
            return None
        return await self._write_scoped(state, persona_block, previous, sections, end, asked, targets,
                                        refine_round=refine_round)

    async def _write_scoped(
        self, state: QuestState, persona_block: str, previous: str, sections: list[_paper_scope.Section], end: int,
        asked: dict[int, list[str]], targets: dict[int, int], *, refine_round: bool = False, extra_note: str = "",
    ) -> str | None:
        """Ask the writer for the sections ``asked`` of ``previous`` (shown read-only) and put them back
        (core/paper_scope.py). A section the reply gives that was not asked for is dropped (the earlier text stands, the
        log says which); a reply without an asked section cannot be put back, so ``None`` (the whole paper is written)."""
        order = sorted(asked)
        lines = []
        for n, index in enumerate(order, 1):
            goal = f" (about {targets[index]} words)" if index in targets else ""
            lines.append(f"{n}. `## {sections[index].title}`{goal}. Why: " + "; ".join(asked[index]))
        scope_block = (
            "## This round: write only these sections\n\n"
            "FI keeps every section you are not asked for exactly as it is in the earlier draft below, which is read-only "
            "context. Do not write those sections, a title, a References list or anything else. Write again only the "
            "sections listed, for the reasons given: in each, change only what the reason needs and keep the rest of its "
            "text as it is. Keep every `{{...}}` placeholder, figure link and citation label of the text you keep. "
            "This round's instruction overrides \"Produce a single Markdown paper\" above.\n\n"
            "Reply with the sections listed, in this order, each starting with its heading line exactly as given "
            "(`## Title`), its subsections inside it, and nothing else. If the fix truly needs a section that the "
            "earlier draft does not have, write it right after the listed section it belongs with and put a line "
            "`NEW SECTION: <heading> -- why it is needed` before it; a new section with no such line is dropped.\n\n"
            "Sections to write:\n" + "\n".join(lines) + "\n\n<earlier_draft>\n" + previous[:end].rstrip()
            + "\n</earlier_draft>\n"
        )
        self._log.info(
            "[write] writing %d of %d section(s) again (%s); every other section of the earlier draft is kept as it was",
            len(order), len(sections) - 1, ", ".join(sections[i].title for i in order))
        self._sheet_given = False
        try:
            reply = await self._write_whole_paper(state, persona_block, refine_round=refine_round, extra_note=extra_note,
                                                  scope_block=scope_block)
        except Exception as e:  # noqa: BLE001 -- the whole-paper call is what a real outage stops
            self._log.warning("[write] the call for sections failed (%s: %s); writing the whole paper again",
                              type(e).__name__, str(e)[:200])
            return None
        # The lines that route a person's refine ride at the end of the reply; they are kept whatever section they follow.
        routing = [m.group(0).strip() for rx in (_REFINE_POINT_RE, _REFINE_WHY_RE) for m in rx.finditer(reply)]
        reply = _REFINE_WHY_RE.sub("", _REFINE_POINT_RE.sub("", reply))
        try:
            spliced = _paper_scope.splice(previous, sections, set(order), reply)
        except _paper_scope.ScopeError as e:
            self._log.warning("[write] the sections could not be put back (%s); writing the whole paper again. "
                              "The reply began: %r", e, " ".join(str(reply).split())[:160])
            return None
        if spliced.added:
            self._log.info("[write] the writer declared %s as new section(s) the fix needs; kept, each right after the "
                           "section it came with", ", ".join(f"\"{t}\"" for t in spliced.added))
        if spliced.ignored:
            self._log.info("[write] the writer also gave %s, which were not asked for; the earlier text stands",
                           ", ".join(f"\"{t}\"" for t in spliced.ignored))
        self._log.info("[write] put back %d section(s) (%s); the other %d are as they were",
                       len(spliced.written), ", ".join(spliced.written), len(sections) - 1 - len(spliced.written))
        return spliced.text.rstrip("\n") + "\n" + ("\n" + "\n".join(routing) + "\n" if routing else "")

    async def _extension_sections(
        self, state: QuestState, persona_block: str, draft: str, note: str,
    ) -> str | None:
        """``draft`` with the sections that report results written again so that they report what the extension added, or
        ``None`` (the whole paper is written). The sections are the ones that carry a result key or a table of FI's records
        (read from the draft's structure, not its wording); a draft with none has no section to put the results in."""
        start = _SOURCE_LIST_HEADING_RE.search(draft)
        end = start.start() if start else len(draft)
        sections = _paper_scope.split(draft, end)
        holders = _paper_scope.holds_values(draft, sections)
        if len(draft) > _PAPER_PROMPT_CHARS or not holders:
            self._log.info("[write] writing the whole paper again: no section of the draft carries FI's result keys to "
                           "put the added results in")
            return None
        asked = {i: ["it reports results, and the results the reader asked for are not in the paper"] for i in holders}
        again = await self._write_scoped(state, persona_block, draft, sections, end, asked, {}, extra_note=note)
        return None if again is None else _disclosure.without_block(again)

    async def _patch_flagged_passages(self, state: QuestState, persona_block: str, *,
                                      otherwise: str = "writing the whole paper again") -> str | None:
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

        A first draft, a hit about the whole paper (``figure_missing``), a study that changed, a reply that is not usable
        edits, and an edit that cannot be applied all return ``None``, and the log
        says which."""
        review = state.get("review") or {}
        hits = [str(h).strip() for h in review.get("must_flag_hits") or [] if str(h).strip()]
        if not hits:
            return None  # a first draft, or a revise with no hit to answer
        previous = ""
        try:
            if state.get("paper_md"):
                previous = Path(_rerun_from.in_quest(state["paper_md"], self.quest_root)).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            previous = ""
        # The editor works on the writer's placeholders, not on the values FI filled in: an edit cannot change a value of
        # FI's record, and the write step fills every placeholder again from the records (core/paper_facts.py).
        previous = _paper_facts.to_placeholders(previous)
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
            )
        if reason:
            self._log.info("[write] %s: %s", otherwise, reason)
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
            values_sheet=self._patch_record_block(state),
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
                "[write] the call for edits failed (%s: %s); %s",
                type(e).__name__, str(e)[:200], otherwise,
            )
            return None
        try:
            patched = paper_patch.apply_edits(previous, end, paper_patch.parse_edits(reply))
        except paper_patch.PatchError as e:
            self._log.warning(
                "[write] the edits could not be used (%s); %s. "
                "The reply began: %r", e, otherwise, " ".join(str(reply).split())[:160],
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
        # This quest's own paper, also when the folder was copied or moved after the paper was written.
        paper_md = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
        if not paper_md or not Path(paper_md).is_file():
            self._log.info("[claim_check] no paper to check; skipping")
            # An earlier draft's receipt must not stand for a paper that is not there.
            self._write_receipt("claim_check", "unknown", started_at=_receipts.now(), error="there is no paper to check")
            return {}
        started = _receipts.now()
        # A paper from before FI's facts were placeholders (a resumed quest) is cleaned and filled first.
        self._refresh_paper_facts(state)
        checked_bytes = Path(paper_md).read_bytes()
        # The engine's paragraph on how the result was reached is not a claim of the paper's (core/disclosure.py).
        paper_raw = Path(paper_md).read_text(encoding="utf-8")
        # Nor is the "Best design found" section, when it is exactly FI's (core/best_design_report.py).
        paper_raw = _best_report.strip_for_checks(paper_raw, self._best_design_section(state, paper_raw))
        # Nor are FI's tables of every setting (core/paper_facts.py ``{{results_table}}``).
        paper_raw = _paper_facts.without_blocks(paper_raw)
        paper_text = _paper_for_prompt(_disclosure.strip_for_checks(paper_raw), "claim_check", self._log)
        literature = state.get("literature") or []
        audience = self.config.output.audience
        refs = build_references(literature, audience=audience)
        # Web pages are Further reading, labelled W1, W2...; a claim resting on
        # one is grounded in a source too.
        further = build_further_reading(literature, audience=audience)
        # Each source the paper cites comes with its text, and a citation must
        # quote it: the model checks what the source says, and the quote is
        # then looked up in the source.
        # A quote is looked up in the source's own text only: what a model read off its figures is shown beside it,
        # labelled, and a claim that rests on it is an unverified model observation, never a citation.
        split_sources = {
            label: (meta, *_split_figure_readings(_item_content(item)))
            for label, meta, item in _labelled_sources(literature, audience)
        }
        sources = {label: (meta, text) for label, (meta, text, _readings) in split_sources.items()}
        readings_of = {label: readings for label, (_meta, _text, readings) in split_sources.items()}
        citing = _citing_sentences(paper_text)
        # The sources' text is material to check claims against, never instructions (core/source_text.py).
        refs_block = _source_text.fence("\n\n".join(
            _claim_source_block(label, meta, text, citing.get(label) or [], readings=readings_of.get(label, ""))
            for label, (meta, text) in sorted(
                sources.items(), key=lambda kv: (kv[0].startswith("W"), int(kv[0].lstrip("W")))
            )
        )) or "(no references)"
        analysis = state.get("analysis") or {}
        # The evidence is the run's own results, never the analysis step's prose: a number the model wrote there that
        # the run does not hold (one real paper said 1,920 where the run's value was 1849.0) would "support" the same
        # number in the paper, which is circular. The analysis text only ranks which result branches to keep.
        ranking = "\n".join(
            str(x) for key in ("key_findings", "claims_supported") for x in (analysis.get(key) or [])
        )
        evidence_block = _claim_results_block(
            state.get("result_json") or {},
            # The claims are the paper's: the branches that hold the numbers it writes are the ones to keep.
            query=paper_text + "\n" + ranking,
            budget=_CLAIM_EVIDENCE_CHARS,
        )
        # The paper reports means over the seeds with their intervals, which
        # seed 0's RESULT_JSON holds neither of. They follow the results, one
        # line each, and outside the budget above: they are a few lines that
        # the results must never be able to push out.
        # FI's tables of every setting, the ones the writer was given: a sentence about one setting's value is checked
        # against its row, not against the run's own summary alone. Outside the budget above, as the intervals below.
        try:
            tables = _results_by_setting.claim_note(self.quest_root, self._protocol_block(state), self._judged_checks())
        except Exception as e:  # noqa: BLE001 -- evidence FI cannot read is left out, never a reason to stop
            self._log.warning("[claim_check] FI's tables of every setting could not be read (%s: %s)",
                              type(e).__name__, str(e)[:200])
            tables = ""
        if tables.strip():
            evidence_block += ("\n\nFI'S TABLES OF EVERY SETTING (FI's own record of the trials, the tables the writer "
                               "was given; a number the paper states for a setting is in the evidence when its row "
                               "here shows it):\n" + tables.strip())
        intervals = _replicate_result_intervals(state)
        if intervals:
            n_seeds = len(state.get("result_json_replicates") or [])
            evidence_block += f"\n\nMean over the {n_seeds} seeds, with its 95% CI:\n" + "\n".join(
                f"- {path}: {s['mean']:.4g}"
                + (f" (95% CI {s['ci_lower']:.4g} to {s['ci_upper']:.4g})" if "ci_lower" in s and "ci_upper" in s else "")
                for path, s in list(intervals.items())[:40]
            )
        # The paper as the checker is shown it carries FI's id before each sentence (core/paper_scope.py): a claim names the
        # passage it is about by that id, and FI reads the passage from its own table.
        # FI's figures are left out: their captions are FI's record of what each figure draws, not the writer's claims.
        marked_paper, id_table = _paper_scope.mark(
            _figure_facts.RUN_FIGURE_RE.sub("", _disclosure.strip_for_checks(paper_raw)))
        prompt = self._prompts["claim_check"].substitute(
            topic=state["topic"],
            evidence_block=evidence_block,
            method_block=_claim_method_block(self.quest_root, state),
            references=refs_block,
            paper=_paper_for_prompt(marked_paper, "claim_check"),
        )
        # claim_check runs AFTER the paper is already written, so a provider
        # failure here (a bridge stall, a server out of capacity once its
        # retries are spent) must NOT abort the quest and forfeit the paper and
        # every downstream output. It must not pass silently either: a rewrite
        # whose check failed kept the previous draft's grounding, and nothing
        # checked its new citations. The failure is recorded, the stale
        # grounding cleared, and the review forces ``citations_unchecked``.
        try:
            # An unreadable reply is asked for once more, short (_chat_json); a second unreadable one is a failure.
            reply, parsed = await self._chat_json(
                prompt, node="claim_check", want=("claims",), usable=lambda p: isinstance(p.get("claims"), list),
            )
        except _ModelAnswerProblem:
            raise
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"[:300]
            self._log.warning(
                "[claim_check] grounding call failed (%s); the review will mark "
                "this draft's citations unchecked", reason,
            )
            # The receipt too: an earlier draft's must not stand for this one.
            self._write_receipt("claim_check", "unknown", started_at=started, inputs={"paper": checked_bytes},
                                error=reason)
            self._stop_once_for_check("claim_check", reason, look="the [claim_check] entry in .fi/run.log",
                                      inputs={"paper": checked_bytes})
            return {"claim_grounding": {}, "claim_check_failed": reason}
        # A reply that parses as JSON but never names a "claims" list at all is
        # not the same thing as one that names an EMPTY list — the former is
        # the model failing to answer in the expected shape, the latter is it
        # genuinely reporting nothing to check. Only the latter is a clean 0.
        parsed_ok = isinstance(parsed, dict) and isinstance(parsed.get("claims"), list)
        raw_claims = parsed.get("claims") if isinstance(parsed, dict) else None
        n_refs = len(refs)
        # Each claim's basis and fault are closed choices: a claim whose choice cannot be read is asked once more, alone
        # with the others that failed (the rest of the answer is kept); one still unreadable is ``unread``.
        raw_claims = await self._reask_claim_choices(prompt, reply, raw_claims or [])
        claims: list[dict[str, Any]] = []
        for c in (raw_claims or []):
            if not isinstance(c, dict) or not str(c.get("claim") or "").strip():
                continue
            unread = _claim_choice_problem(c)
            basis = "unsupported" if unread else str(_claim_choice(c.get("basis")))
            fault = "unread" if unread else _claim_choice(c.get("fault"))
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
            if unread:
                evidence = (f"{evidence} (FI could not read the check's verdict on this claim after asking once: {unread}; "
                            "it counts as unsupported)").strip()
            # FI's own look-up of a citation: a claim it cannot place in a source's text is unsupported, and its sentence
            # stays and is stated in the Limitations (fault ``wording``): a quote FI cannot find does not show the sentence
            # false, only that FI could not confirm it.
            if basis == "citation" and cite_idx is None:
                basis, fault = "unsupported", "wording"
            elif basis == "citation" and not _quote_in_source(quote, sources[str(cite_idx)][1]):
                # The model had the source's text; a citation it cannot quote
                # from that text is not one the source supports.
                basis, fault = "unsupported", "wording"
                why = f"no quote from [{cite_idx}] given" if not quote else f"the quote is not in the text of [{cite_idx}]"
                if quote and _quote_in_source(quote, readings_of.get(str(cite_idx), "")):
                    why = (f"the quote is a model's reading of a figure of [{cite_idx}], not the source's text: an "
                           "unverified model observation")
                evidence = f"{evidence} ({why})".strip()
            claims.append({
                "claim": str(c["claim"]).strip(),
                "basis": basis,
                "fault": fault,
                "keys": _claim_keys(c.get("keys")),
                "citation_index": cite_idx,
                "quote": quote,
                "evidence": evidence,
                "where": _places_as_placeholders(_paper_scope.read_places(c.get("where"), id_table)),
            })
        # The rules below add claims of their own; an empty answer from the check is judged before them.
        model_found_none = not claims
        # A source with nothing but its title cannot back what a sentence says
        # beyond it, whatever the check made of the quote.
        claims, held = _apply_title_only_rule(paper_text, sources, claims)
        if held:
            self._log.info(
                "[claim_check] %d claim(s) rest on a source held as its title only; marked unsupported", held,
            )
        # A retracted source supports nothing, whatever the check made of its text (core/retractions.py).
        claims, withdrawn = _retractions.apply_to_claims(claims, sources, citing, _same_statement)
        if withdrawn:
            self._log.info("[claim_check] %d claim(s) rest on a retracted source; marked unsupported", withdrawn)
        # A sentence an earlier check marked unsupported that the paper still holds word for word (its wording, or the
        # passage the check stored for it) stays unsupported when
        # this check does not list it: leaving a sentence out is not a judgement of it (a reply that judges it again decides).
        kept = _unsupported_still_in(paper_text, (state.get("claim_grounding") or {}).get("claims") or [], claims)
        if kept:
            claims.extend(kept)
            self._log.info("[claim_check] %d sentence(s) an earlier check found unsupported are still in the paper unchanged "
                           "and this check did not judge them again: they stay unsupported", len(kept))
        # FI's own checks set the fault where they fired: a retracted source backs nothing (``no_source``), and a sentence
        # the check found unsupported that states a number the writer typed and FI cannot trace is ``number``.
        plan_of_typed = _paper_facts.plan_protocol(self.quest_root, self._protocol_block(state))
        topic_of_typed = str(self.config.topic or "")
        claims = _claim_faults_from_sources(
            claims, sources,
            lambda passage: bool(_paper_facts.untraced_typed(passage, plan_of_typed, topic_of_typed, self.quest_root)))
        n_unread = sum(1 for c in claims if c.get("fault") == "unread")
        if n_unread:
            self._log.warning("[claim_check] %d claim(s) whose verdict could not be read after asking once count as "
                              "unsupported, and their sentences are taken out of the paper", n_unread)
        unsupported = [c["claim"] for c in claims if c["basis"] == "unsupported"]
        # A paper with a real body that yields zero claims is suspicious, not
        # clean: the check almost certainly missed something rather than the
        # paper genuinely making no substantive claims. Thresholded on the
        # BODY text actually sent (not the raw file), so a short abstract-only
        # draft isn't flagged for having little to check.
        suspicious_empty = parsed_ok and model_found_none and len(paper_text) > 2000
        failed = ""
        if not parsed_ok:
            failed = f"the grounding reply did not name a claims list: {str(parsed)[:200]!r}"
        elif suspicious_empty:
            failed = f"0 claims were extracted from a {len(paper_text)}-character paper body"
        grounding = {
            "claims": claims,
            "summary": str(parsed.get("summary") or "").strip(),
            "total": len(claims),
            "grounded": len(claims) - len(unsupported),
            "unsupported": unsupported,
            "status": "unknown" if failed else "ok",
        }
        if failed:
            self._log.warning(
                "[claim_check] %s; the review will mark this draft's citations unchecked",
                failed,
            )
        else:
            self._log.info(
                "[claim_check] %d/%d claims grounded (%d unsupported)",
                grounding["grounded"], grounding["total"], len(unsupported),
            )
        self._write_claims_ledger(grounding)
        self._write_receipt(
            "claim_check", "unknown" if failed else "fail" if unsupported else "pass", started_at=started,
            inputs={"paper": checked_bytes}, output=grounding, error=failed,
            detail=f"{len(unsupported)} of {len(claims)} claim(s) in the final draft are unsupported" if unsupported else "",
        )
        if failed:
            self._stop_once_for_check("claim_check", failed, look="the [claim_check] entry in .fi/run.log",
                                      inputs={"paper": checked_bytes})
        else:
            self._clear_check_stop("claim_check")
        return {"claim_grounding": grounding, "claim_check_failed": failed}

    async def _reask_claim_choices(self, prompt: str, reply: str, raw_claims: list[Any]) -> list[Any]:
        """The claim check's claims with the ones whose ``basis`` or ``fault`` is not a closed choice asked ONCE more, those
        claims alone and each by what was wrong (``agents/json_reanswer_part.md``): every other claim is kept as written.
        A new answer is matched to its claim by the claim's words; only when one claim was asked and one answer came back
        is an answer with other words taken for it (two or more are never matched by order, which could mix them up). A
        claim still unreadable, or a call that fails, leaves the claim as it was, and the reader then makes it ``unread``."""
        asked = [(i, c) for i, c in enumerate(raw_claims)
                 if isinstance(c, dict) and str(c.get("claim") or "").strip() and _claim_choice_problem(c)]
        if not asked:
            return raw_claims
        wrong = "; ".join(f"the claim {' '.join(str(c['claim']).split())[:160]!r}: {_claim_choice_problem(c)}"
                          for _i, c in asked)
        what = (f"{wrong}. Give `claims` as a list holding only these {len(asked)} claim(s), each with `claim` copied as "
                "you wrote it, its `basis` (experiment, citation or unsupported), its `fault` (JSON null for experiment "
                "or citation; number, scope or wording for unsupported), `keys`, `citation_index`, `quote` and `evidence`")
        self._log.warning("[claim_check] %d claim(s) gave a basis or fault that is not one of the choices; asking once "
                          "more for those claims only", len(asked))
        echo = (f"\nYour previous reply, for reference (say the same things for these claims):\n\n{reply}\n"
                if len(reply or "") <= self._REANSWER_ECHO_CHARS else "")
        retry = self._prompts["json_reanswer_part"].substitute(
            original_prompt=prompt, what_was_wrong=what, keys="`claims`", earlier_reply=echo)
        try:
            again = await self._chat(retry, node="claim_check")
        except _ModelAnswerProblem:
            raise
        except Exception as e:  # noqa: BLE001 -- a failed ask leaves those claims unread, never the whole check failed
            self._log.warning("[claim_check] the ask for those claims failed (%s: %s); they count as unsupported",
                              type(e).__name__, str(e)[:200])
            return raw_claims
        answer = _parse_json_lenient(again, node="claim_check", want=("claims",)) or {}
        given = [c for c in (answer.get("claims") if isinstance(answer, dict) else None) or [] if isinstance(c, dict)]

        def words(c: dict[str, Any]) -> str:
            return " ".join(str(c.get("claim") or "").split()).lower()

        out = list(raw_claims)
        left = list(asked)
        unmatched: list[dict[str, Any]] = []
        for g in given:
            hit = next(((i, c) for i, c in left if words(c) == words(g)), None)
            if hit is None:
                unmatched.append(g)
                continue
            left.remove(hit)
            out[hit[0]] = {**hit[1], **{k: v for k, v in g.items() if k != "claim"}}
        if len(left) == len(asked) == len(unmatched) == len(given) == 1:
            for (i, c), g in zip(left, unmatched):
                out[i] = {**c, **{k: v for k, v in g.items() if k != "claim"}}
        still = sum(1 for i, _c in asked if _claim_choice_problem(out[i]))
        if still:
            self._log.warning("[claim_check] %d of those claim(s) still could not be read", still)
        else:
            self._log.info("[claim_check] the %d claim(s) asked again could be read", len(asked))
        return out

    async def _node_citation_check(self, state: QuestState) -> QuestState:
        """The last step before the paper is delivered: its citations are read against their sources
        (:meth:`_check_citations`), then the paper as delivered is measured against the page limit, and one still over it
        goes on record for the quality report (:meth:`_state_page_overrun`). The review states an overrun only
        when it has used up its shortening rewrites, and a paper can also leave the review loop earlier (its other
        findings out of rounds), so the delivered paper is measured here, whatever route brought it."""
        out = await self._check_citations(state)
        # The paper goes out as the last review saw it when no round follows (the graph reaches this step only from the
        # review): the points that review raised and nothing answered are stated in the Limitations and listed in QUALITY.md.
        if out.get("citation_next") == "done" and _disclosure.mark_last_review_final(self.quest_root, state.get("review")):
            self._log.info("[review] the last review's points that no later round addressed are stated in the paper's "
                           "Limitations and listed in QUALITY.md")
        # The Limitations are FI's block, from the records as they stand now (the claim check may have been off, so the
        # refresh above may not have run): the paper that is delivered says what they say.
        self._restate_limitations(state, "FI wrote the Limitations from its records again and added no claim")
        # Before the page count is measured: its sentence is a few lines of the paper's own.
        self._state_unsupported_claims({**state, **out})  # type: ignore[arg-type]
        limit = resolve_page_limit(self.config)
        paper_path = _rerun_from.in_quest(state.get("paper_md"), self.quest_root) or None
        if limit is not None and paper_path and Path(paper_path).is_file():
            measured = await self._measure_draft_pages(Path(paper_path), {**state, **out})  # type: ignore[arg-type]
            if measured is not None and int(measured["pages"]) > limit:
                pages = await self._state_page_overrun({**state, **out}, Path(paper_path), limit,  # type: ignore[arg-type]
                                                       int(measured["pages"]))
                self._log.warning("[page_limit] the paper as delivered renders to %d pages, over the limit of %d: "
                                  "QUALITY.md says so", pages, limit)
        return out

    async def _check_citations(self, state: QuestState) -> QuestState:
        """Read every citation of the paper that is about to be delivered against the retrieved text of its source, and
        fix what that text does not support (core/citation_support.py). It runs last: after the review, the rewrites and
        the page-limit fit, so nothing can add a citation after it. A person is never asked; the paper is never stopped
        for this. A no-op when ``engine.claim_grounding`` is off."""
        nxt = "human_feedback" if self._route_after_review(state) == "human_feedback" else "done"
        out: dict[str, Any] = {"citation_next": nxt}
        if not self.config.engine.claim_grounding:
            return out  # type: ignore[return-value]
        paper_md = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
        if not paper_md or not Path(paper_md).is_file():
            return out  # type: ignore[return-value]
        path = Path(paper_md)
        # The paper this check reads and leaves is the one a reader gets: FI's old blocks out and its facts of the run
        # filled (the claim check it runs again would otherwise change the paper after this check's receipt).
        self._refresh_paper_facts(state)
        started = _receipts.now()
        paper = path.read_text(encoding="utf-8")
        sha = _cs.paper_hash(paper)
        record_path = self.fi_dir / _cs.FILE
        try:
            previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
        except (OSError, ValueError):
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
        if previous.get("final_sha") == sha and previous.get("order") is not None:
            # This paper is what an earlier pass of the check left (a resume after the record was written, or the same
            # paper shown again). Nothing is asked again; what the state would have held is put back: the sources in the
            # order the paper now numbers them, the receipts for the text as it is, the claim check on it.
            return await self._settle_citation_check(  # type: ignore[return-value]
                state, path, previous, lambda rec: self._citation_record_save(record_path, rec), out,
                _reorder_literature(state.get("literature") or [], previous.get("order") or []))
        record = previous if previous.get("paper_sha") == sha else _cs.new_record(paper, previous)

        def save(rec: dict[str, Any]) -> None:
            self._citation_record_save(record_path, rec)

        literature = state.get("literature") or []
        audience = self.config.output.audience
        listed = set(web_pages_listed(paper) or [])
        sources: dict[str, _cs.Source] = {}
        for label, meta, item in _labelled_sources(literature, audience):
            text, _readings = _split_figure_readings(_item_content(item))
            who = meta.get("authors") or []
            sources[label] = _cs.Source(
                label=label, title=str(meta.get("title") or ""), text=text, full=_has_full_text(meta),
                authors=[str(a) for a in (who if isinstance(who, list) else [who])],
                retracted=_retractions.is_retracted(meta), usable=(not label.startswith("W")) or label in listed)
        own = _cs.OwnNumbers(re.findall(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", " ".join(
            json.dumps(x, default=str) for x in (
                state.get("result_json"), state.get("result_json_replicates"), state.get("analysis"),
                state.get("design"), state.get("topic"), _frozen.protocol_of(self.quest_root)))))
        unsupported_claims = [str(c.get("claim") or "") for c in ((state.get("claim_grounding") or {}).get("claims") or [])
                              if isinstance(c, dict) and c.get("basis") == "unsupported"]
        reader_node = self._review_node("citation_check")
        same, reader, _writer_model, _named = self._reader_vs_planner(reader_node, after_call=False, writer="write")

        def passage_block(label: str, sentences: list[str], srcs: dict[str, _cs.Source]) -> str:
            src = srcs.get(label)
            if src is None:
                return f"[{label}] (not one of the paper's sources)"
            head = f"[{label}] {src.title} ({src.kind()})"
            if not src.text.strip():
                return head + "\n(no text of this source was retrieved, so nothing can be quoted from it)"
            return head + "\nText:\n" + _format_lit_excerpt(
                src.text, src.title, query=" ".join(sentences), budget=_CITATION_SOURCE_CHARS).strip()

        def judge_prompt(asked: list[tuple[str, Any, str]], srcs: dict[str, _cs.Source]) -> str:
            by_label: dict[str, list[str]] = {}
            for _pid, site, label in asked:
                by_label.setdefault(label, []).append(site.text)
            return self._prompts["citation_check"].substitute(
                pairs="\n".join(f"{pid} | cites [{label}] | {' '.join(site.text.split())}" for pid, site, label in asked),
                sources=_source_text.fence("\n\n".join(
                    passage_block(lb, by_label[lb], srcs)
                    for lb in sorted(by_label, key=lambda x: (x.startswith("W"), int(x.lstrip("W") or 0))))) or "(none)")

        def repair_prompt(targets: list[dict[str, Any]]) -> str:
            items: list[str] = []
            for t in targets:
                lines = [f"Item {t['id']} ({t['kind']}): {' '.join(t['sentence'].split())}"]
                lines += [f"  not supported: [{lb}] -- {why}" for lb, why in t["why"].items()]
                if t["kept"]:
                    lines.append("  already supported, keep: " + ", ".join(f"[{x}]" for x in t["kept"]))
                lines += [f"  nearest passage of the retrieved text of [{lb}]: {' '.join(p.split())}" for lb, p in t["cited"]]
                for lb, p in t["others"]:
                    lines.append(f"  other source [{lb}] {sources[lb].title} ({sources[lb].kind()}): {' '.join(p.split())}")
                items.append("\n".join(lines))
            return self._prompts["citation_repair"].substitute(items=_source_text.fence("\n\n".join(items)) or "(none)")

        async def judge(prompt: str) -> str:
            return await self._chat(prompt, node=reader_node)

        async def repair(prompt: str) -> str:
            return await self._chat(prompt, node="write.citations")

        result = await _cs.check_paper(
            paper, sources, own_numbers=own, quote_ok=_quote_in_source, judge=judge, repair=repair, record=record,
            save=save, judge_prompt=judge_prompt, repair_prompt=repair_prompt,
            claim_unsupported=lambda sentence: any(_same_statement(c, sentence) for c in unsupported_claims))
        final = paper
        new_literature = literature
        if result.changed:
            final, new_literature, numbering = _finalize_with_numbering(result.paper, literature, audience)
        data = _cs.ledger(result, reader=reader, same_model=same)
        if result.changed:
            data = _cs.relabel_ledger(data, numbering)  # named by the numbers the delivered paper has
        c = result.counts
        summary = {**c, "reader": reader, "reader_is_writer": same, "judge_failed": result.judge_failed}
        # Everything a resume needs to put the state back is in the record BEFORE the paper is written.
        record.update(
            final_sha=_cs.paper_hash(final), changed=bool(result.changed), summary=summary, done=False,
            order=[_literature_key(i, n) for n, i in enumerate(new_literature)],
            receipt={"status": "pass" if result.clean else "unknown",
                     "error": "" if result.clean else (record.get("judge_error") or f"{c['unchecked']} citation(s) were not judged"),
                     "detail": f"{c['citations']} citation(s): {c['supported']} backed as written, {c['found_unsupported']} fixed"})
        save(record)
        if result.changed:
            path.write_text(final, encoding="utf-8")
        try:
            (path.parent / "citations.json").write_text(_cs.dump(data), encoding="utf-8")
            (path.parent / "CITATIONS.md").write_text(_cs.render_md(data), encoding="utf-8")
        except OSError as e:
            self._log.warning("[citation_check] its ledger could not be written: %r", e)
        self._log.info(
            "[citation_check] %d citation(s): %d backed by a passage as written, %d found unsupported and fixed (%d to another "
            "source, %d reworded, %d citation dropped, %d clause dropped, %d sentence dropped)%s%s", c["citations"], c["supported"],
            c["found_unsupported"], c["replaced"], c["reworded"], c["citation_dropped"], c["clause_dropped"], c["sentence_dropped"],
            f", {c['unchecked']} could not be checked" if c["unchecked"] else "",
            f"; read by {reader}" + (" (the model that wrote the paper)" if same else ""))
        return await self._settle_citation_check(state, path, record, save, out, new_literature, started=started)  # type: ignore[return-value]

    def _citation_record_save(self, record_path: Path, rec: dict[str, Any]) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            tmp = record_path.with_name(record_path.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            os.replace(tmp, record_path)
        except OSError as e:
            self._log.warning("[citation_check] its record could not be written: %r", e)

    async def _settle_citation_check(
        self, state: QuestState, path: Path, record: dict[str, Any], save: Callable[[dict[str, Any]], None],
        out: dict[str, Any], literature: list[Any], *, started: str = "",
    ) -> dict[str, Any]:
        """The part of the citation check that follows the paper being final, run on a first pass and again on a resume:
        the receipt for the text as it is, the claim check on it when the check changed it and the claim check's receipt is
        not for this text, and the record marked done. A step already done is not done again, so no model call is repeated."""
        final_bytes = path.read_bytes()
        digest = _receipts.sha256(final_bytes)
        status, receipt, problem = _receipts.read(self.quest_root, "citation_check")
        if problem or receipt is None or (receipt.get("input_hashes") or {}).get("paper") != digest:
            try:
                data = json.loads((path.parent / "citations.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = dict(record.get("summary") or {})
            kept = record.get("receipt") or {}
            self._write_receipt("citation_check", kept.get("status") or "unknown", started_at=started or _receipts.now(),
                                inputs={"paper": final_bytes}, output=data, error=kept.get("error") or "",
                                detail=kept.get("detail") or "")
        out.update({"citation_check": record.get("summary") or {}, "literature": literature})
        if record.get("changed"):
            c_status, c_receipt, c_problem = _receipts.read(self.quest_root, "claim_check")
            if c_problem or c_receipt is None or (c_receipt.get("input_hashes") or {}).get("paper") != digest:
                # The claim ledger and its receipt are for the paper as it is delivered.
                try:
                    out.update(await self._node_claim_check({**state, "literature": literature, **out}))  # type: ignore[arg-type]
                except Exception as e:  # noqa: BLE001 -- the citations are already settled; this only refreshes the ledger
                    self._log.warning("[citation_check] the claim check could not be run again on the fixed paper: %r", e)
        record["done"] = True
        save(record)
        return out

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

    @staticmethod
    def _dropped_skills(state: Any) -> list[dict[str, Any]]:
        return [d for d in ((state or {}).get("skill_selection") or {}).get("dropped") or [] if isinstance(d, dict)]

    def _dropped_tool_names(self, state: Any) -> set[str]:
        """Normalised names of the dropped skills that are tools (never Python packages). A library skill is often named
        after its pip package, so its name is not taken out of what pip is asked for."""
        return {_experiment_deps.normalize(str(d.get("name"))) for d in self._dropped_skills(state)
                if str(d.get("kind")) == "tool" and d.get("name")}

    def _dropped_skills_note(self, state: Any) -> str:
        """Tells the code-writing steps which skills this quest no longer carries, so a name the frozen plan or an
        earlier pick used is not imported, run or listed as a package."""
        names = [str(d["name"]) for d in self._dropped_skills(state) if d.get("name")]
        if not names:
            return ""
        return ("\n\nNot available to this quest (do not use, import, run or list as a dependency): "
                + ", ".join(names) + ". If the design mentions one, do the same work another way and say so in `method`.")

    def _skill_kinds(self, names: list[str]) -> tuple[dict[str, str], dict[str, str]]:
        """``(kinds, unusable)``: the kind of each named skill that can be found, and why each unusable one is."""
        from core.skills import loadable_skills

        kinds: dict[str, str] = {}
        unusable: dict[str, str] = {}
        if not names:
            return kinds, unusable
        try:
            usable, rejected = loadable_skills(names, external_dirs=self._skill_dirs)
        except Exception as exc:  # noqa: BLE001 -- the kind is only used to narrow what is dropped
            self._log.warning("[skills] could not look the earlier pick up (%s)", exc)
            return kinds, unusable
        for st in usable:
            kinds[st.skill.name] = str(getattr(st.skill.kind, "value", st.skill.kind)).lower()
        for st in rejected:
            kinds[st.skill.name] = str(getattr(st.skill.kind, "value", st.skill.kind)).lower()
            unusable[st.skill.name] = f"{st.status.value}: {st.reason}"
        return kinds, unusable

    async def _repick_skills(self, graph: Any, fork_config: dict[str, Any]) -> dict[str, Any]:
        """``--from skills``: pick the skills again from the quest's CURRENT config, on the checkpoint just before the
        code (or the data) is made, and return the config of the checkpoint that carries the new pick.

        Only ``selected_skills``, ``skill_selection`` and the outline (which was written for the old pick) change. The
        literature, plan.md, the design and the frozen protocol are not touched: the plan is not re-run, so nothing
        that a person approved is rewritten. A skill the new pick drops is written to ``skill_selection["dropped"]``
        with the reason, so the code-writing steps are told it is not available."""
        snapshot = await graph.aget_state(fork_config)
        values = dict(snapshot.values or {})
        old = [str(n) for n in values.get("selected_skills") or []]
        exclude = set(self.config.engine.skills_exclude or [])
        kinds, unusable = await asyncio.to_thread(self._skill_kinds, old)
        fresh = await self._node_select_skills(values)  # type: ignore[arg-type]
        record = dict(fresh.get("skill_selection") or {})
        new = [str(n) for n in fresh.get("selected_skills") or []]
        if record.get("error") and not new:
            # The registry or the pick call failed: keep what still stands of the old pick rather than none.
            new = [n for n in old if n not in exclude and n not in unusable]
            record = dict(values.get("skill_selection") or {}) | {"error": record.get("error")}
            self._log.warning("[skills] the pick could not be made again (%s); the earlier pick is kept without the "
                              "skills that are excluded or no longer usable", record.get("error"))
        dropped: list[dict[str, str]] = []
        for name in old:
            if name in new:
                continue
            if name in exclude:
                why = "it is excluded in this quest's config (engine.skills_exclude)"
            elif name in unusable:
                why = f"it is not usable now ({unusable[name]})"
            else:
                why = "the new pick did not choose it"
            dropped.append({"name": name, "why": why, "kind": kinds.get(name, "")})
            self._log.info("[skills] dropped %s: %s", name, why)
            self._progress(f"[FI] skill {name} is no longer used by this quest: {why}.")
        added = [n for n in new if n not in old]
        self._log.info("[skills] picked again: kept %s; added %s; dropped %s",
                       [n for n in new if n in old] or "none", added or "none", [d["name"] for d in dropped] or "none")
        if dropped:
            record["dropped"] = dropped
        else:
            record.pop("dropped", None)
        # A frozen plan is never rewritten here. If it names a dropped skill, say so plainly and how to change it.
        plan_path = self.quest_root / "plan.md"
        text = (plan_path.read_text(encoding="utf-8", errors="replace") if plan_path.is_file() else "")
        text += " " + json.dumps(values.get("design") or {}, ensure_ascii=False)
        text = text.lower()
        for d in dropped:
            spelled = r"[-_ ]+".join(re.escape(part) for part in re.split(r"[-_ ]+", d["name"].lower()) if part)
            if spelled and re.search(rf"(?<!\w)(?<!\w-){spelled}(?!\w)(?!-\w)", text):
                self._log.warning("[skills] the plan or design names %s, which is no longer used; the plan is left as "
                                  "it is, and the code is told the skill is not available", d["name"])
                self._progress(f"[FI] the plan names the skill {d['name']}, which this quest no longer uses. The plan is "
                               f"not changed; the code is written without it. To change the plan, use --revise-plan.")
        writes = list(((snapshot.metadata or {}).get("writes") or {}).keys())
        as_node = next((w for w in writes if w in ("design", "synthesize", "data_load")), "design")
        return await graph.aupdate_state(
            fork_config,
            {"selected_skills": new, "skill_selection": record, "implement_outline": {}},
            as_node=as_node,
        )

    def _skills_block(self, state: QuestState | None = None, *, in_code: bool = False) -> str:
        """Instructions for the skills this quest actually selected.

        ``in_code``: for the coding agent, whose code may not reach outside its folder (``core/agent_coder.py``): each
        skill's bundled files are named under ``skills/<name>/`` in the code folder, where :meth:`_skill_files_to_code`
        puts a copy of them, and not at the skill's own folder.

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
            in_code_base = f"{_SKILLS_IN_CODE}/{skill.name}" if in_code and mount is None and not not_mounted else ""
            if in_code_base:
                spellings = {str(skill.path), Path(skill.path).as_posix()}
                try:  # a skill's text often names its folder in the home form (`~/.../skills/<name>`)
                    home_rel = Path(skill.path).resolve().relative_to(Path.home().resolve())
                    spellings |= {f"~/{home_rel.as_posix()}", f"~\\{home_rel}"}
                except (ValueError, OSError, RuntimeError):
                    pass
                for spelling in sorted(spellings, key=len, reverse=True):
                    instructions, surface = (instructions.replace(spelling, in_code_base),
                                             surface.replace(spelling, in_code_base))
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
                elif in_code_base:
                    # The agent's code may not reach outside its folder (a path into the person's home is refused), so
                    # the skill's own folder is never named: FI put a copy beside the code.
                    parts.append(
                        f"FI put a copy of these in this folder, under `{in_code_base}/` (FI's files: do not change "
                        f"them). Every run has the quest folder as its working directory, not this folder, so open or "
                        f"run each one from the script's own folder (`Path(__file__).parent / \"{in_code_base}\"` plus "
                        f"the path listed), and launch a script with `sys.executable` (the Python running the "
                        f"experiment), not the bare command `python`, even where the skill's text above writes `python`."
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

    def _skill_files_to_code(self, state: Any) -> list[str]:
        """Copy the bundled files of each selected skill into ``code/skills/<name>/`` (the coding agent's code may not
        reach outside its folder, :meth:`_skills_block` with ``in_code``): the paths copied, relative to ``code/``.
        A file that cannot be copied is left out and said in run.log; the quest goes on."""
        copied: list[str] = []
        try:
            usable, _reasons = _resolve_selected_skills(state, self._log, use="experiment",
                                                        external_dirs=self._skill_dirs)
        except Exception as exc:  # noqa: BLE001 -- no skill files: the agent writes the code without them
            self._log.warning("[skills] the selected skills could not be resolved to copy their files: %s", exc)
            return copied
        for st in usable:
            skill = st.skill
            for rel in [*skill.bundled_scripts(), *skill.reference_files()]:
                target = self.quest_root / "code" / _SKILLS_IN_CODE / skill.name / rel
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(Path(skill.path) / rel, target)
                    copied.append(target.relative_to(self.quest_root / "code").as_posix())
                except OSError as exc:
                    self._log.warning("[skills] %s of the skill %s could not be copied into code/ (%s)", rel,
                                      skill.name, exc)
        if copied:
            self._log.info("[skills] put a copy of %d file(s) of the selected skills in code/%s/ for the coding agent",
                           len(copied), _SKILLS_IN_CODE)
        return copied

    def _experiment_skill_states(self, state: Any) -> list[Any]:
        """The approved skills this quest selected for its experiment (as ``_resolve_selected_skills`` returns them).
        Resolving them can run their self-tests; a failure here only means no skill is offered."""
        selection = (state or {}).get("skill_selection") or {}
        uses = dict(selection.get("uses") or {})
        wanted = [n for n in list((state or {}).get("selected_skills") or []) if uses.get(n, "experiment") == "experiment"]
        try:
            usable, _ = _resolve_selected_skills(state, self._log, use="experiment", external_dirs=self._skill_dirs)
        except Exception as exc:  # noqa: BLE001 -- the run goes on without skills rather than not at all
            self._log.warning("[execute] the selected skills could not be resolved: %s", exc)
            usable = []
        # A skill selected but not usable now (its self-test fails, it was changed since approval) was only logged;
        # the code written for it then failed on an import nobody explained. The repair step is told (repair_note).
        found = {st.skill.name for st in usable}
        self._unusable_skills = [n for n in wanted if n not in found]
        return list(usable)

    async def _install_packages(self, packages: list[str]) -> list[tuple[str, str]]:
        """Install ``packages`` into the quest environment; return the ones that could not be, each with the reason.

        One line first; if it fails, one package at a time, so a single name pip cannot find no longer leaves every
        other package (numpy, matplotlib) uninstalled."""
        if not packages:
            return []
        self._log.info("[execute] pip install %s", packages)
        batch = await self.executor.install(packages, quest_root=self.quest_root)
        if batch.returncode == 0:
            return []
        self._log.warning(
            "[execute] pip install rc=%d: %s; installing one package at a time",
            batch.returncode, pip_failure_summary(batch.stderr),
        )
        failed: list[tuple[str, str]] = []
        for pkg in packages:
            one = await self.executor.install([pkg], quest_root=self.quest_root)
            if one.returncode != 0:
                failed.append((pkg, _experiment_deps.failure_reason(one.stderr)))
        for pkg, why in failed:
            self._log.warning("[execute] could not install %s: %s", pkg, why)
        return failed

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

    def _outline_skills_block(self, state: QuestState | None = None) -> str:
        """The selected skills as ``implement_outline`` needs them.

        The outline fixes the structure (the functions, the constants, the result
        template); the body stage, which writes the calls, still receives each
        skill's full text from :meth:`_skills_block`. So the outline gets the summary
        the design node gets. Replayed from three stored quests, outlines written
        from the summary named the same functions and result template as ones written
        from the full text (18 of 18 parsed, and the two arms agreed with each other
        as much as two outlines of one arm did) for 2,066 prompt tokens instead of
        22,894. The full text stays for a run that submits a background job or carries
        the user's example files: there the skills say how the job is submitted and how
        the examples are combined, and that is structure."""
        from core.example_inputs import render_block

        if self.config.execution.background_jobs or render_block(self.quest_root):
            return self._skills_block(state)
        return self._skills_summary_block(state) or self._skills_block(state)

    def _analyze_literature_block(self, state: QuestState, *, degenerate: bool = False) -> str:
        """The reviewed-literature block of the analyze prompt.

        Whole for a survey or no-simulation study (the literature is what it
        interprets), for a run on the user's own data (``--analyze``), and for a run
        with no results or a degenerate one. When an experiment produced results, the
        analysis interprets those, and the block is the titles of the sources
        retrieved (about 700 tokens instead of 8,000 to 13,000): the write step
        receives the literature and each key finding is cross-checked against it
        afterwards. Replayed from three stored quests (two calls each), analyses
        written with only the titles matched the ones written with the whole block
        (5.0 against 5.3 key findings, 2.0 against 2.5 supported claims, every
        ``next_step`` still ``publish``) at 25,098 prompt tokens instead of 34,702."""
        sources = _labelled_sources(state.get("literature") or [], self.config.output.audience)
        if (
            state.get("no_simulation_resolved")
            or state.get("survey_mode_resolved")
            or self.config.engine.analyze_local_first
            or degenerate
            or not state.get("result_json")
            or not sources
        ):
            return _format_lit_from_state(state, **self._lit_kwargs(state))
        titles = "\n".join(
            f"- [{label}] {' '.join(str(meta.get('title') or '').split())} ({meta.get('year') or 'n.d.'})"
            for label, meta, _item in sources
        )
        return (
            "Titles of the sources retrieved for this study (their text is not shown at this step: the "
            "write step receives it, and each key finding is cross-checked against the literature "
            "afterwards):\n" + _source_text.fence(titles, "the titles of retrieved papers and web pages")
        )

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

    def _goal_coverage_notes(self, state: QuestState) -> list[str]:
        """Advisory: what the experiment does differently from the numbers the
        topic sets, one line each (:mod:`core.goal_coverage`).

        A topic that names its settings (``R0 in {0.9, 1.5, 3.0}``), a count of
        runs (``300 runs``) or a number of figures is compared with the
        experiment's script, the keys of its results and the figures it drew.
        Nothing here is a must-flag hit: a topic is prose, and sending the
        experiment back costs a whole cycle. Writes ``paper/goal_coverage.json``
        either way, so a clean run leaves evidence that the check ran and what it
        looked at. Skipped for a study with no experiment and for a run with no results. Best-effort: a failure
        here never touches the quest."""
        if (
            state.get("no_simulation_resolved")
            or state.get("survey_mode_resolved")
            or self.config.engine.analyze_local_first
            # A run with no results failed (one stored run's script was
            # ``print("RESULT_JSON: {}")``), which other checks report; its numbers
            # would only be listed as missing.
            or not state.get("result_json")
        ):
            return []
        topic = str(state.get("topic") or self.config.topic or "")
        try:
            source = (self.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
            simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME
            if simulate.is_file():
                # The sweep and the number of runs are in the simulation.
                source = simulate.read_text(encoding="utf-8") + "\n" + source
        except (OSError, UnicodeDecodeError):
            return []
        try:
            figures = len(state.get("figures") or [])
            asked = goal_coverage.asked_numbers(topic)
            asked_figures = goal_coverage.asked_figure_count(topic)
            found = goal_coverage.notes(topic, [source], state.get("result_json"), figures)
            # The protocol was checked when the script was written; a repair after a failed run may have changed a
            # constant since. Said here too, where the reviewer reads it.
            protocol = self._protocol_of(state)
            if protocol is not None:
                found = found + [
                    f"The script differs from the plan's protocol: {msg}"
                    for msg in self._protocol_drift_not_asked_for(protocol)
                ]
        except Exception as e:  # noqa: BLE001 - never fail a quest over an advisory check
            self._log.warning("[goal_coverage] check failed (%s); skipping", e)
            return []
        try:
            paper_dir = self.quest_root / "paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "goal_coverage.json").write_text(
                json.dumps({
                    "numbers_the_topic_sets": [a.value for a in asked],
                    "figures_the_topic_asks_for": (
                        {"count": asked_figures[0], "kind": asked_figures[1]} if asked_figures else None
                    ),
                    "figures_drawn": figures,
                    "notes": found,
                }, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            self._log.warning("[goal_coverage] could not write audit: %s", e)
        for note in found:
            self._log.warning("[goal_coverage] %s", note)
        if not found:
            self._log.info(
                "[goal_coverage] %d number(s) the topic sets and %s checked against the experiment: nothing differs",
                len(asked), "its figure count" if asked_figures else "no figure count",
            )
        return found

    def _keep_analysis_answers(self, result_json: Any, state: QuestState, runs: list[Any] | None = None,
                               scheduled: int | None = None) -> None:
        """Keep the value the run recorded under each name the plan gives a quantity, so the paper's key for a quantity no
        trial returns can be filled (core/paper_values.py::write_answers). Best-effort. When FI wrote the analysis
        (core/analysis_frame.py), the numbers come from FI's own record of it (``data/results/derived.json``): every quantity
        it worked out once for the study, by the plan's formula or a function of derive.py, as well as the printed results."""
        try:
            if self._analysis_on():
                record = _analysis_frame.read_record(self.quest_root)
                if not result_json or record.get("from_earlier_run"):
                    return  # a run sent back, stopped or killed: its numbers are not this run's answers
                protocol = self._protocol_block(state)
                values = _analysis_frame.study_values(record)
                printed = result_json if isinstance(result_json, dict) else {}
                ids = list(dict.fromkeys([*(str(m["id"]) for m in (protocol or {}).get("metrics") or []
                                            if isinstance(m, dict) and m.get("id")), *values]))
                # one record for the whole study (derived.json), whatever number of runs was scheduled
                _paper_values.write_answers(self.quest_root, {**printed, **values}, protocol, None, None, ids=ids)
                return
            _paper_values.write_answers(self.quest_root, result_json, self._protocol_block(state), runs, scheduled)
        except Exception as e:  # noqa: BLE001 -- the quantity then has no key; the paper is not held to place it
            self._log.warning("[execute] the run's recorded answers could not be kept for the paper's keys (%s: %s)",
                              type(e).__name__, str(e)[:200])

    def _unplaced_topic_answers(self, paper_md: str, state: QuestState) -> list[str]:
        """A quantity the topic asks for, that FI holds a value for and the paper never places (core/goal_coverage.py::
        unplaced_answers): a set difference over FI's own records, so it is forced like a planned figure the paper leaves
        out. Kept in ``paper/goal_coverage.json``. Never raises."""
        try:
            rec = self._paper_records(state)
            if rec is None or not rec.cells or not paper_md:
                return []
            req = _disclosure._read_json(self.fi_dir / _topic_req.FILE)
            found = goal_coverage.unplaced_answers(
                req, self._protocol_block(state), rec.fillable(), _paper_values.placed_metrics(paper_md))
            path = self.quest_root / "paper" / "goal_coverage.json"
            record = _disclosure._read_json(path) if path.is_file() else {}
            if not isinstance(record, dict):
                record = {}
            if found or record.get("quantities_not_placed"):
                record["quantities_not_placed"] = found
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception as e:  # noqa: BLE001 -- an advisory-grade failure never touches the quest
            self._log.warning("[goal_coverage] the quantities the paper never gives could not be checked (%s)", e)
            return []
        for f in found:
            self._log.warning("[goal_coverage] %s", f["sentence"])
        return [f["sentence"] for f in found]

    def _reference_check_hits(self, state: QuestState) -> list[str]:
        """A reference quantity of the results that is exactly 0 at every setting of a sweep (core/numeric_oracle.py): what a
        root finder returning the trivial root of its bracket looks like. A check on the results, not on the paper; one
        advisory string per finding, kept in the review as ``numeric_oracle_warnings``. Never raises."""
        try:
            from core import numeric_oracle

            findings = numeric_oracle.trivial_reference_findings(state.get("result_json"))
        except Exception as e:  # noqa: BLE001 - never fail a quest over the checker
            self._log.warning("[numeric_oracle] check failed (%s); skipping", e)
            return []
        for f in findings:
            self._log.warning("[numeric_oracle] %s: %s", f.kind, f.describe())
        return [f"{f.kind}: {f.describe()}" for f in findings]

    def _statistics_claim_hits(
        self, paper_md: str, state: QuestState,
    ) -> list[str]:
        """Statistics the paper describes as something the run did not compute.

        The numeric oracle asks whether a number MATCHES a result. This asks
        whether it is the QUANTITY the paper says it is — a Bonferroni
        threshold printed as a p-value, an effect-size claim the run's own
        comparisons contradict, a t-interval labelled "exact binomial", a
        major-outbreak probability called the chance the disease vanishes.

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
            report = stat_claims.check(
                _paper_facts.without_blocks(paper_md),
                result_json=state.get("result_json") or {},
                intervals=_replicate_result_intervals(state),
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

    def _best_design_section(self, state: QuestState, markdown: str) -> str:
        """The "Best design found" section FI puts into ``markdown`` (core/best_design_report.py); ``""`` for a study
        that is not a search for the best design, or one with no record of its search."""
        block = _optimise.block_of(self._protocol_block(state))
        if block is None:
            return ""
        try:
            normal, _why = _optim.normalize(block)
            return _best_report.for_paper(self.quest_root, normal or block, markdown)
        except Exception as exc:  # noqa: BLE001 -- a section FI cannot build is left out, never a reason to stop
            self._log.warning("[write] the \"Best design found\" section could not be built (%s: %s)",
                              type(exc).__name__, str(exc)[:200])
            return ""

    def _mark_best_design(self, state: QuestState, markdown: str) -> str:
        """``markdown`` with FI's "Best design found" section (and ``results/best_design.md``, the same text as a file),
        or with any copy of it taken out when there is none."""
        text = self._best_design_section(state, markdown)
        if text:
            block = _optimise.block_of(self._protocol_block(state))
            try:
                normal, _why = _optim.normalize(block)
                _best_report.write_readable(self.quest_root, normal or block)
            except Exception:  # noqa: BLE001 -- a convenience copy
                pass
            self._log.info("[write] FI's \"Best design found\" section is in the paper (from results/best_design.json "
                           "and needs/OPTIMUM_CHECK.json)")
        return _best_report.mark_paper(markdown, text)

    def _optimise_refine_request(self, state: QuestState, feedback: str) -> dict[str, Any]:
        """What a refine asks of a search for the best design (core/optimise_refine.py); ``{}`` for any other study or
        an ordinary refine. A request to search further is recorded as a round of the search here."""
        block_raw = _optimise.block_of(self._protocol_block(state))
        if block_raw is None or not str(feedback or "").strip():
            return {}
        try:
            block, _why = _optim.normalize(block_raw)
            if block is None:
                return {}
            request = _optimise_refine.read_request(feedback, block, _best_report.records(self.quest_root)["best_design"])
            if request.get("kind") == _optimise_refine.SEARCH and _phased.enabled(self.config) and _phased.stage(
                    self.quest_root) in (_phased.CONFIRM, _phased.CONFIRMED):
                # After the freeze the search is never run again (core/phased.py): searching on would choose the design
                # on the data meant to confirm it.
                return {"kind": _optimise_refine.UNCLEAR, "says": (
                    "Your refine was not carried out: this study's best design was frozen and confirmed on runs the "
                    "search never saw, and searching further now would choose the design on those runs. Nothing was "
                    "changed. To search further, start a new quest from this one's plan.")}
            if request.get("kind") == _optimise_refine.SEARCH:
                rounds = _optimise_refine.add_round(self.quest_root, block,
                                                    {**request["round"], "refine": _refine_count(state) + 1})
                self._log.info("[optimise] the person asked to search further (round %d): %s%s", len(rounds),
                               request["says"], f" Read as: {request['round']['read_as']}." if
                               request["round"].get("read_as") else "")
                # (The round's file is in the audit trace's watched files; the route the request takes is a route
                # decision there.)
                self._progress(str(request["says"]))
        except Exception as exc:  # noqa: BLE001 -- an unreadable request is an ordinary refine, never a stop
            self._log.warning("[optimise] the refine could not be read as a request to the search (%s: %s); it goes to "
                              "the writer", type(exc).__name__, str(exc)[:200])
            return {}
        return {k: v for k, v in request.items() if k in ("kind", "says", "round")}

    def _optimise_refine_hint(self, state: QuestState) -> str:
        """How to ask a search for the best design for more, for the review pause; ``""`` for any other study."""
        try:
            block_raw = _optimise.block_of(self._protocol_block(state))
            if block_raw is None:
                return ""
            block, _why = _optim.normalize(block_raw)
            return _optimise_refine.hint(block)
        except Exception:  # noqa: BLE001 -- a hint
            return ""

    def _prune_ledger_after_take_out(self, taken: list[str], paper_md: str, *, strict: bool = False) -> None:
        """A claim whose sentence FI took out of the paper leaves ``paper/claims.json`` with it, as with the page-limit
        trim: the slides, poster and talk read that list, and must not state what the paper no longer says. A claim
        another sentence still says stays. Best-effort."""
        try:
            ledger = _read_json_or_none_path(self.quest_root / "paper" / "claims.json")
            pruned = _grounding_after_trim(ledger, [t for t in taken if t], paper_md) if isinstance(ledger, dict) else None
            if pruned is not None:
                self._write_claims_ledger(pruned, strict=strict)
        except Exception as e:  # noqa: BLE001 -- the ledger is a transparency record, never a reason to stop
            if strict:
                raise
            self._log.warning("[claims] the claim list could not be pruned after a take-out (%s)", e)

    def _write_claims_ledger(self, grounding: dict[str, Any], *, strict: bool = False) -> None:
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
                tag = (f"cite [{c['citation_index']}]" if c["basis"] == "citation"
                       else f"{c['basis']}: {c['fault']}" if c.get("fault") else c["basis"])
                lines.append(f"- **[{tag}]** {c['claim']}"
                             + (f" — {c['evidence']}" if c["evidence"] else "")
                             + (f" — quoted: \"{c['quote']}\"" if c.get("quote") else ""))
            (paper_dir / "CLAIMS.md").write_text(
                "\n".join(lines) + "\n", encoding="utf-8",
            )
        except OSError as e:
            if strict:
                raise
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
        if fmt == DEFAULT_PAPER_FORMAT and venue in SCIENTIFIC_PAPER_FORMATS | NON_SCIENTIFIC_PAPER_FORMATS:
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
            pdf, skip, record = PaperGenerator(config).compile_for_page_count(Path(paper_path), scratch)
            if pdf is None:
                return None, skip or "the render produced no PDF"
            if not record.get("typeset", True):
                self._log.warning(
                    "[page_limit] LaTeX could not typeset the draft (%s); its pages are counted on the "
                    "browser render instead",
                    record.get("reason") or "no reason given",
                )
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

    async def _trim_body_to_fit(
        self, state: QuestState, paper_path: Path, limit: int, measured: dict[str, Any],
    ) -> tuple[dict[str, Any], paper_trim.Trimmed] | None:
        """Bring a draft whose body is a little over the page limit within it by
        taking whole sentences out of its background and discussion, when that
        alone does it, and write the result over ``paper_path``.

        The alternative is the whole-paper rewrite the review forces, the largest
        part of the revise loop (:mod:`core.paper_trim` has the numbers) and the
        step that changes correct citations. Here the model is shown the paper with
        a number before each sentence that may go (:func:`paper_trim.candidates`
        offers no Abstract, Method or Result, nothing that says what the paper does
        or finds, no number stated nowhere else, no figure or table) and answers
        with the numbers of sentences that together are about the words to cut. The
        engine takes them out, skipping one that holds a source's only citation,
        and renders the draft. A list that falls short, or a draft that is still
        over, is another round on the paper as it now stands, at most
        ``_TRIM_ROUNDS`` calls. Nothing the model writes reaches the paper.

        Returns the measurement of the draft written and what was taken out, or
        ``None`` (with the reason in the log, the paper untouched) when the draft
        is more than ``paper_trim.MAX_WORDS`` over, the sections that may lose a
        sentence hold too few words, a reply is not a list of numbers, or the
        rounds do not bring it within the limit."""
        need = _words_to_cut(int(measured["pages"]), limit, float(measured.get("last_page_empty") or 0.0))
        try:
            current = paper_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        trial = self.fi_dir / "page_check_trial.md"
        self.fi_dir.mkdir(parents=True, exist_ok=True)
        spent = 0
        taken: list[paper_trim.Sentence] = []
        try:
            for _ in range(_TRIM_ROUNDS):
                if spent + need > paper_trim.MAX_WORDS:
                    self._log.info(
                        "[page_limit] the draft is about %d words over the limit, more than the %d a trim takes out",
                        spent + need, paper_trim.MAX_WORDS,
                    )
                    return None
                source_lists = _SOURCE_LIST_HEADING_RE.search(current)
                end = source_lists.start() if source_lists else len(current)
                offered = paper_trim.candidates(current, end)
                if sum(sentence.words for sentence in offered) < need:
                    self._log.info(
                        "[page_limit] the draft is about %d words over the limit and its background and discussion "
                        "offer %d words that could go without losing what the paper does or finds, a number, a "
                        "figure or a paragraph's opening",
                        need, sum(sentence.words for sentence in offered),
                    )
                    return None
                prompt = self._prompts["write_trim"].substitute(
                    topic=state.get("topic") or self.config.topic,
                    words=need,
                    upper=need + 40,
                    paper_block=paper_trim.marked_body(current, end, offered),
                )
                try:
                    ranking = paper_trim.parse_ranking(await self._chat(prompt, node="write.trim"), len(offered))
                except paper_patch.PatchError as e:
                    self._log.warning("[page_limit] the sentences to take out could not be used (%s)", e)
                    return None
                except Exception as e:  # noqa: BLE001 - the rewrite the review forces is what a real outage stops
                    self._log.warning(
                        "[page_limit] the call for sentences to take out failed (%s: %s)",
                        type(e).__name__, str(e)[:200],
                    )
                    return None
                cut = paper_trim.choose(current, end, offered, ranking, need, partial=True)
                if cut is None:
                    self._log.info(
                        "[page_limit] none of the sentences the model chose can go (each holds a source's only "
                        "citation or a number stated nowhere else)",
                    )
                    return None
                current, spent = cut.text, spent + cut.words
                taken += cut.taken
                if cut.words < need:
                    need -= cut.words  # the list fell short of the words: ask again on what is left
                    continue
                trial.write_text(current, encoding="utf-8")
                found = await self._measure_draft_pages(trial, state)
                if found is None:
                    return None
                if int(found["pages"]) <= limit:
                    paper_path.write_text(current, encoding="utf-8")
                    return found, paper_trim.Trimmed(current, tuple(taken), spent)
                need = _words_to_cut(int(found["pages"]), limit, float(found.get("last_page_empty") or 0.0))
        finally:
            trial.unlink(missing_ok=True)
        self._log.info("[page_limit] the draft is still over the limit after %d sentences were taken out", len(taken))
        return None

    def _state_unsupported_claims(self, state: QuestState) -> int:
        """The paper as delivered: the sentences the last claim check found supported by neither the run's results nor a
        cited source are kept in ``.fi/unsupported_claims.json`` and stated in the Limitations (core/paper_facts.py
        ``UNSUPPORTED_CLAIMS``), sentence by sentence, from FI's own record: no model call and nothing taken out of the
        paper. Until now these findings were only in QUALITY.md, so a reader of the paper was not told. A claim the paper no
        longer holds is not listed; a record with none empties, and a sentence an earlier pass placed is taken out. A claim
        check that did not run to an answer leaves the record as it was. The sentence adds no claim, so the claim check's
        and the citation check's receipts move to the paper that carries it. Returns how many claims are stated."""
        if not self.config.engine.claim_grounding:
            return 0
        grounding = state.get("claim_grounding") or {}
        paper_path = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
        if not isinstance(grounding, dict) or grounding.get("status") != "ok" or not paper_path                 or not Path(paper_path).is_file():
            return 0
        try:
            before = Path(paper_path).read_text(encoding="utf-8")
            # FI's own blocks quote claims too (the Limitations block does): only the writer's text says what the paper holds
            flat = " ".join(_paper_facts.without_blocks(before).split())
            sentences = re.split(r"(?<=[.!?])\s+", flat)
            # The passage the check stored for a claim (``where``) is the exact way to find it again; the claim's own
            # wording is only the fallback for a claim with no stored passage in the paper.
            keyed = " ".join(_paper_facts.to_placeholders(_paper_facts.without_blocks(before)).split())
            by_wording = [c for c in grounding.get("claims") or []
                          if isinstance(c, dict) and c.get("basis") == "unsupported" and str(c.get("claim") or "").strip()
                          and (" ".join(str(c["claim"]).split()) in flat
                               or any(_same_statement(str(c["claim"]), s) for s in sentences))]
            # The passage match widens the Limitations listing only; the take-out of sentences with a number below is given
            # exactly the claims that matched by wording, as before.
            standing = [c for c in grounding.get("claims") or []
                        if isinstance(c, dict) and c.get("basis") == "unsupported" and str(c.get("claim") or "").strip()
                        and (any(c is b for b in by_wording) or _passage_stands(c, flat, keyed))]
            # A sentence whose claim's fault takes it out (a number not in the record, a true value under a wrong setting or
            # label, a source that backs nothing, a verdict FI could not read) is not delivered (core/record_claims.py): it
            # is taken out of the paper and the Limitations say so. A ``wording`` claim stays and is stated as unconfirmed.
            # Only the fault decides: what the sentence holds is not read. An unsupported claim with no fault was judged
            # before faults were named (a record an older FI wrote, read when its quest is resumed; this FI's check gives
            # every unsupported claim one): it is decided by the rule that FI applied, so its sentence goes when it states a
            # number, one FI filled in or one the writer typed that FI cannot trace.
            # The paper, the claim list, the receipt, the record and the Limitations change together: if any write fails,
            # every file is put back as it was and nothing is reported as taken out.
            snapshot = self._snapshot_files(paper_path)
            removed: list[dict[str, Any]] = []
            try:
                text, removed = self._plan_take_out_unsupported_numbers(
                    state, paper_path, [c for c in by_wording if c.get("fault") in _CLAIM_FAULTS_TAKEN_OUT],
                    unnamed=[c for c in by_wording if c.get("fault") is None])
                if removed:
                    Path(paper_path).write_bytes(text.encode("utf-8"))  # bytes back as bytes (CRLF stays as it was)
                    _receipts.carry_over(self.quest_root, "claim_check", "paper", snapshot[Path(paper_path)] or b"",
                                         Path(paper_path).read_bytes(),
                                         "FI took out sentences with a number its claim check found unsupported and added no claim")
                    self._prune_ledger_after_take_out([str(r["sentence"]) for r in removed], text, strict=True)
                    gone = {str(r["claim"]) for r in removed}
                    left = [c for c in standing if str(c.get("claim")) not in gone]
                else:
                    left = standing
                if not _paper_facts.write_unsupported_record(self.quest_root, left, removed=removed):
                    raise OSError("the record of unsupported claims could not be written")
                self._restate_limitations(state, "FI stated the claims its own check found unsupported in the Limitations and "
                                                 "added no claim", strict=bool(removed))
                standing = left
            except Exception as e:  # noqa: BLE001 -- nothing is taken out unless every part could be written
                if not removed:
                    raise
                self._restore_files(snapshot)
                self._log.warning("[claim_check] the sentences with an unsupported number could not be taken out (%s: %s); "
                                  "every file was put back as it was", type(e).__name__, str(e)[:200])
                if not _paper_facts.write_unsupported_record(self.quest_root, standing):
                    return 0
                self._restate_limitations(state, "FI stated the claims its own check found unsupported in the Limitations and "
                                                 "added no claim")
                removed = []
            if removed:
                self._log.warning("[claim_check] %d sentence(s) that state a number the claim check found unsupported were "
                                  "taken out of the paper; the Limitations say so and QUALITY.md lists them", len(removed))
        except (OSError, UnicodeDecodeError) as e:
            self._log.warning("[claim_check] the paper could not be given its sentence on unsupported claims (%s)", e)
            return 0
        if standing:
            self._log.warning("[claim_check] %d sentence(s) of the paper as delivered were not found supported; its "
                              "Limitations say so", len(standing))
        return len(standing)

    def _snapshot_files(self, paper_path: Path | str) -> dict[Path, bytes | None]:
        """The bytes of every file the take-out of unsupported sentences can change (``None``: it did not exist)."""
        root = self.quest_root
        files = [Path(paper_path), root / "paper" / "claims.json", root / "paper" / "CLAIMS.md",
                 root / "needs" / "receipts" / "claim_check.json", root / "needs" / "receipts" / "citation_check.json",
                 root / ".fi" / "value_marks.json", root / _paper_facts.UNSUPPORTED_RECORD]
        return {f: (f.read_bytes() if f.is_file() else None) for f in files}

    @staticmethod
    def _restore_files(snapshot: dict[Path, bytes | None]) -> None:
        for f, data in snapshot.items():
            if data is None:
                f.unlink(missing_ok=True)
            else:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(data)

    def _plan_take_out_unsupported_numbers(self, state: QuestState, paper_path: Path | str,
                                           standing: list[dict[str, Any]], *, unnamed: list[dict[str, Any]] | None = None,
                                           ) -> tuple[str, list[dict[str, Any]]]:
        """``(the paper without them, the entries taken out)``: the writer's sentences of ``standing`` (claims whose fault
        takes the sentence out, chosen by the caller), found exactly (core/record_claims.py::take_out_unsupported_numbers).
        The claim's fault decides, never what the sentence holds, so every exactly placed sentence of these claims goes.
        ``unnamed``: unsupported claims judged before faults were named (a record an older FI wrote); a placed sentence of
        one goes by the rule that FI applied: when it holds a value FI filled in, or a number the writer typed that FI cannot
        trace to the plan or its records. Nothing is written here; :meth:`_state_unsupported_claims` writes the paper, the
        claim list, the receipt and the record together."""
        before = Path(paper_path).read_bytes().decode("utf-8")

        def plain(raw: str) -> str:
            return " ".join(_paper_facts.without_blocks(_paper_values.without_marks(raw)).split())

        def keyed(raw: str) -> str:
            return " ".join(_paper_facts.to_placeholders(raw).split())

        text, removed = _record_claims.take_out_unsupported_numbers(
            before, standing, same=_same_statement, holds_number=lambda _sentence: True, plain=plain, keyed=keyed)
        if not unnamed:
            return text, removed
        plan = _paper_facts.plan_protocol(self.quest_root, self._protocol_block(state))
        topic = str(self.config.topic or "")

        def holds_number(sentence: str) -> bool:
            if _paper_values._MARK.search(sentence) or _paper_values._SPAN.search(sentence):
                return True
            return bool(_paper_facts.untraced_typed(sentence, plan, topic, self.quest_root))

        text, older = _record_claims.take_out_unsupported_numbers(
            text, unnamed, same=_same_statement, holds_number=holds_number, plain=plain, keyed=keyed)
        return text, removed + older

    async def _state_page_overrun(self, state: QuestState, paper_path: Path, limit: int, pages: int) -> int:
        """The paper stays over its page limit: FI's measurement goes on record (``.fi/page_limit.json``), which the
        quality report prints (core/paper_facts.py ``page_limit_text``). The limit is a guideline, so the paper itself
        carries no sentence about it. Returns the page count."""
        _paper_facts.write_page_record(self.quest_root, pages, limit, final=True)
        return pages

    async def _page_limit_review(
        self, state: QuestState, paper_path: str | Path | None,
    ) -> dict[str, Any] | None:
        """The page-limit part of the review: the record for ``review["page_limit"]``.
        Without a page limit nothing is rendered and the record is empty, as it is
        when the draft cannot be measured. The limit is a guideline: a draft over it
        is never sent back to be shortened. Its Further reading list loses its last
        entries (:meth:`_drop_further_reading_to_fit`), then a few sentences of
        background and discussion are taken out (:meth:`_trim_body_to_fit`), and what is
        still over goes on record for the quality report and the log."""
        limit = resolve_page_limit(self.config)
        if limit is None or not paper_path or not Path(paper_path).is_file():
            return None
        measured = await self._measure_draft_pages(Path(paper_path), state)
        if measured is None:
            return None
        pages = int(measured["pages"])
        record: dict[str, Any] = {"pages": pages, "limit": limit}
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
        if pages > limit:
            # Then the body: a few sentences of background and discussion out, no
            # rewrite and no second claim check; it costs the model one short call.
            cut = await self._trim_body_to_fit(state, Path(paper_path), limit, measured)
            if cut is not None:
                before, (measured, taken) = pages, cut
                pages = int(measured["pages"])
                record["pages"] = pages
                record["trimmed"] = {"words": taken.words, "sentences": [sentence.text for sentence in taken.taken]}
                self._log.warning(
                    "[page_limit] the draft renders to %d pages, over the limit of %d: took %d sentences "
                    "(%d words) out of its background and discussion instead of writing the paper again; "
                    "it now renders to %d pages. Taken out: %s",
                    before, limit, len(taken.taken), taken.words, pages, paper_trim.summary(taken),
                )
        if pages <= limit:
            self._log.info("[page_limit] the draft renders to %d pages; the limit is %d", pages, limit)
            _paper_facts.write_page_record(self.quest_root, pages, limit, final=False)
            return record
        self._log.warning(
            "[page_limit] the draft renders to %d pages, over the limit of %d; the limit is a guideline, "
            "so the paper is not written again for it (QUALITY.md says so)", pages, limit,
        )
        record["pages"] = await self._state_page_overrun(state, Path(paper_path), limit, pages)
        return record

    def _review_models_collapsed(self, panel_results: list[dict[str, Any]]) -> str | None:
        """Under ``rigor_profile: research``: the one provider/model that actually answered every reviewer who reviewed,
        when the panel was set up on more than one model (a fallback or an alias made them one); ``None`` otherwise."""
        if self.config.rigor_profile != "research" or getattr(self.config.engine, "one_model_review", False):
            return None
        ok = [r for r in panel_results if r.get("status") == "ok"]
        if len(ok) < 2 or len({str(r.get("requested_model")) for r in ok}) < 2:
            return None
        # Only a model the provider reported can show a collapse; a transport that reports none (some CLIs, a VS Code
        # extension that sends no model name, a router alias such as "auto") leaves it unknown, which is not taken for
        # one model.
        if not all(r.get("actual_model_reported") for r in ok):
            return None
        actual = {f"{r.get('actual_provider') or '?'}/{r.get('actual_model') or '?'}" for r in ok}
        return next(iter(actual)) if len(actual) == 1 else None

    def _pause_for_review_models_collapsed(self, paper_path: Any, panel_results: list[dict[str, Any]],
                                           actual: str) -> None:
        """Stop because reviewers set up on different models were all answered by one (a fallback took over, or two
        model names reach the same model): the panel is one model's view in several roles. Resuming asks again; with
        ``engine.one_model_review: true`` the review goes on and the evidence level lists the gap."""
        asked = ", ".join(sorted({str(r.get("requested_model")) for r in panel_results if r.get("status") == "ok"}))
        self._pause_for_human(
            kind="review_models_collapsed",
            interaction="supply",
            headline="the reviewers were set up on different models, but one model answered them all",
            steps=[
                f"Asked for: {asked}. Answered by: {actual} (after a fallback took over, or because two model names reach "
                "the same model).",
                f"The paper is at `{paper_path}`. Resume once the other model can be reached again "
                f"(`python launch.py --resume {self.quest_id}`): the review runs from the start.",
                "To go on with one model's review instead, add `one_model_review: true` under the `engine:` section of "
                "the quest's config.yaml and resume: the review then runs on the models you have, and the result is not "
                "publication-ready, and says why.",
            ],
            payload={"review_models_collapsed": True, "quest_id": self.quest_id, "actual": actual},
        )

    def _note_design_objections(self, state: QuestState, review: dict[str, Any]) -> list[Any]:
        """The review's must-fix hits that count (a page-limit hit aside, and an objection FI's record already keeps),
        and FI's record of the reviewer's objections to the design.

        An objection is a hit that names no problem FI checks in the text (circular evaluation, a single-point evaluation,
        a name the reviewer made up) or, for a revise verdict, the reviewer's own reason (``blocking``): no check of FI's
        confirms it, so the design is not changed for it. Every one goes into FI's record (``.fi/review_objections.json``),
        which the quality report prints (core/disclosure.py:run_record), the last round's too; the paper is not required
        to carry them (the writer may name one that is a real limit of the study, in its own words). One the record
        already holds is in ``review["objections_repeated"]`` and is not counted: a rewrite cannot answer it. A review
        that sends the experiment back to be run again records no reason of its own (it names a wrong number, not the
        design). Best-effort: a record FI cannot read or write counts every hit, as before."""
        hits = [h for h in review.get("must_flag_hits") or [] if _hit_name(h) != _PAGE_LIMIT_HIT]
        objections = [h for h in hits if _hit_name(h) not in _TEXT_ONLY_HITS]
        try:
            stated = _disclosure.stated_objections(self.quest_root)
            repeated = sorted({_objection_key(h) for h in objections} & stated)
            if repeated:
                review["objections_repeated"] = repeated
                self._log.info("[review] objections FI's record already keeps (QUALITY.md), not counted again — a "
                               "rewrite cannot answer them: %s", ", ".join(repeated))
            counted = [h for h in hits if _objection_key(h) not in set(repeated)]
            new = [(_objection_key(h), _one_line(str(h), 300)) for h in objections if _objection_key(h) not in stated]
            said = str(review.get("blocking") or "").strip() if review.get("verdict") == "revise" else ""
            if said and not counted and (self.config.pauses.review == "ask" or not self.config.engine.review_loop):
                said = ""  # a person decides, or no review loop runs: the reason alone is no objection FI keeps
            rerun = (_review_sends_the_experiment_back(review, state)
                     and int(state.get("code_reexecutes") or 0) + 1 <= _CODE_REEXECUTES)
            # The same reason again, with nothing measured behind it: the first rewrite already answered it, and its
            # words are in QUALITY.md. Another rewrite for it would change nothing, so the router does not send one.
            if (said and not counted and not rerun and _disclosure.reason_key(said) in _disclosure.stated_reasons(self.quest_root)):
                review["reason_repeated"] = True
                # The verdict stays revise: no more rewrites are spent, but the reason is not answered, and the record
                # (QUALITY.md) and the paper's Limitations say so.
                _disclosure.mark_unresolved(self.quest_root, said)
                self._log.info("[review] the reviewer's reason repeats one FI's record already keeps (QUALITY.md) and no "
                               "check of FI's backs it — not writing the paper again for it")
            # Kept whether or not the paper is written again after this review: an objection raised only in the last
            # round is in QUALITY.md too.
            if new or (said and not rerun):
                _disclosure.add_objections(self.quest_root, new, "" if rerun else said)
            return counted
        except Exception as e:  # noqa: BLE001 -- FI's record is a help, never a reason to stop
            self._log.warning("[review] the record of the reviewer's objections could not be kept (%r)", e)
            return hits

    def _pause_for_review_unavailable(self, paper_path: Any, why: list[str]) -> None:
        """Stop because the review could not be asked for (a usage limit, a provider that is down), and ask again on
        resume. Never an accept: a failed call used to be recorded as ``accept, score 3``, which is what the person was
        shown at the stop, what an automatic accept would have taken, and what a write-back to Axon trusts."""
        self._pause_for_human(
            kind="review_unavailable",
            interaction="supply",
            headline="the paper is written, but its review could not run",
            steps=[
                "Nothing was reviewed and nothing was accepted: " + "; ".join(why) + ".",
                f"The paper is at `{paper_path}`. Resume when the model can be reached again (after a usage limit resets, "
                "for example): the review then runs from the start.",
            ],
            payload={"review_unavailable": True, "quest_id": self.quest_id, "problems": why},
        )
