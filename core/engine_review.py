"""Methods of :class:`core.engine.Engine` for the review, a person's feedback and the end of the quest.

``Engine`` inherits :class:`ReviewStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, AsyncProcessTree, AsyncSqliteSaver, Awaitable, Callable, LLMClient, Path, ProviderUnusable, QuestArtifacts,
    QuestState, RepairForAgent, _CALL_ATTEMPTS, _CLUSTER_PROTOCOL, _FIGURE_SUFFIXES, _FI_MEMORY_KINDS,
    _FI_PROVIDER_NAMES, _FRESH_SCRIPT, _IN_EACH_RUN_PROTOCOL, _JOB_NO_SEED_PROTOCOL, _JOB_PROTOCOL,
    _JOB_SEED_PROTOCOL, _LAST_CALL, _ModelAnswerProblem, _NOT_GIVEN, _OUTSIDE_WORK_STEPS, _PIP_FREEZE_TIMEOUT_S,
    _SEARCH_PROTOCOL, _SKILL_MOUNTS_FILE, _SPLIT_PROTOCOL, _acceptance, _aggregate_panel_reviews, _attempts,
    _audit_log, _citations_unchecked, _code_layout, _code_review, _crash_kind, _design_parts_listing,
    _extract_result_json, _format_claim_grounding, _format_clarify, _foundational_review_block, _function_steps,
    _grounding_after_trim, _in_run, _is_fatal_provider_error, _is_preliminary_memory, _last_error_line,
    _load_persona_prefix, _memory, _names_a_verdict, _one_line, _optim, _optimise_refine, _oracle, _outcome_of,
    _pairs, _paper_for_prompt, _paper_scope, _paper_sections, _parse_json_lenient, _phased, _plan_settings,
    _plan_steps, _provider_is_auth_error, _raise_if_required_skills_unusable, _read_hit_places,
    _read_json_or_none_path, _readable_json_keys, _receipts, _refine_count, _rerun_from,
    _review_sends_the_experiment_back, _review_was_real, _split_run, _temperature_for_node, _thinking, _todo,
    _top_level_json_objects, _trial_runner, _unsupported_claim_hit, _user_feedback_review_block,
    _without_reviewer_page_limit_hits, _writes_code, append_cost_row, asyncio, fnmatch, hashlib, json, model_for_node,
    node_provider_for, os, platform, resolve_endpoint_async, scrub_secrets, shutil, sys, threading, time, traceback,
    uuid, write_cost_summary,
)


class ReviewStageMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

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
        # A paper from before FI's facts were placeholders (a resumed quest) is cleaned and filled first.
        self._refresh_paper_facts(state)
        paper_path = _rerun_from.in_quest(state.get("paper_md"), self.quest_root) or None
        # With a page limit, this draft is rendered the way paper.pdf will be
        # and its pages counted, for both review paths below. Without one,
        # nothing is rendered.
        checked = Path(paper_path).read_bytes() if paper_path and Path(paper_path).is_file() else None
        page_record = await self._page_limit_review(state, paper_path)
        if checked is not None and Path(paper_path).is_file() and Path(paper_path).read_bytes() != checked:
            # The fit only takes Further reading entries or whole sentences out of the draft the claim check judged,
            # and the claims of a sentence taken out leave its grounding with it (the review reads the grounding of
            # the paper it reads): no claim is added and none loses its source, so the verdict covers what is left.
            _receipts.carry_over(self.quest_root, "claim_check", "paper", checked, Path(paper_path).read_bytes(),
                                 "the page-limit fit only took entries or sentences out of the checked draft")
        # Read after that: it may have dropped Further reading entries from the
        # file, and the review reads the paper as it now is.
        paper_md = ""
        if paper_path:
            try:
                paper_md = Path(paper_path).read_text(encoding="utf-8")
            except OSError:
                paper_md = ""
        # A sentence the trim took out may be a claim the check listed, and one it
        # marked unsupported would still be forced against a paper that no longer
        # says it: the grounding is read as of the paper the review reads.
        grounding = state.get("claim_grounding")
        cut = (page_record or {}).get("trimmed")
        pruned = _grounding_after_trim(grounding, cut["sentences"], paper_md) if cut and grounding else None
        if pruned is not None:
            state = {**state, "claim_grounding": pruned}  # type: ignore[typeddict-item]
            self._write_claims_ledger(pruned)
        # The paper as the reviewer is shown it carries FI's id before each sentence (core/paper_scope.py): a hit names the
        # passage it is about by that id.
        review_marked, review_ids = _paper_scope.mark(paper_md)
        base_prompt = self._prompts["review"].substitute(
            topic=state["topic"],
            clarify_block=_format_clarify(state),
            design_block=json.dumps(state.get("design") or {}, indent=2) + _design_parts_listing(state.get("design")),
            analysis_block=json.dumps(state.get("analysis") or {}, indent=2),
            claim_grounding_block=_format_claim_grounding(state),
            # Advisory: what the reviewer may ask for,
            # never a must-flag hit. Empty when the paper cites every foundational
            # work it has, or has none.
            foundational_check_block=_foundational_review_block(
                state.get("literature") or [], paper_md, self.config.output.audience,
            ),
            user_feedback_block=_user_feedback_review_block(state),
            # What a second model's reading of the code left open: a claim resting on it is not established.
            code_findings_block=_code_review.review_block(self.quest_root, self.quest_root / "code"),
            # The whole paper. A 16 KB cut hid the second half of a real
            # 34,910-character paper, so the review graded a draft it had not
            # read; the cap now only guards against a runaway file.
            paper_md=_paper_for_prompt(review_marked, "review", self._log),
        )

        panel_names = list(self.config.engine.review_panel or [])
        if not panel_names:
            # Legacy single-reviewer path. Review runs AFTER the paper is
            # written, and the output generators (pdf/slides/poster/speech) run
            # only after run() returns — so a transient provider failure here
            # must NOT abort the quest and forfeit the finished paper + its
            # outputs. Fail open to "accept" (the same degrade the parse-miss
            # path below already uses, and the pattern applied to claim_check).
            # A reply that parses as JSON but never names a "verdict" at all
            # (a real case, not hypothetical: a reply shaped like
            # {"comment": "..."}) is exactly as unusable as no reply — checked
            # explicitly rather than trusting "the dict is non-empty", which a
            # provider that answers off-format would still satisfy. It is asked
            # once more; a second reply without a verdict stops the quest like a
            # failed call. A stand-in "accept" used to be recorded instead, and
            # an automatic accept and the Axon write-back took it as a review.
            # The pause normally ends the run here. If it ever returned a value instead (no resume sends one to this
            # pause today), that is read as "ask again", never as a review.
            review: Any = None
            while review is None:
                problem = "the reviewer answered twice without a verdict (accept or revise)"
                for _ask in range(2):
                    try:
                        text = await self._chat(base_prompt, node="review")
                    except _ModelAnswerProblem:
                        raise
                    except (Exception, ProviderUnusable) as e:  # the review's own stop for a reviewer that cannot be asked
                        problem = f"the review call failed ({_one_line(getattr(e, 'error', e), 300)})"
                        break
                    parsed_review = _parse_json_lenient(text)
                    if _names_a_verdict(parsed_review):
                        review = parsed_review
                        break
                    self._log.warning("[review] the reply named no verdict%s", "; asking again" if _ask == 0 else "")
                if review is None:
                    self._pause_for_review_unavailable(paper_path, [problem])
            review["verdict"] = review["verdict"].strip().lower()
            review["status"] = "ok"
            mfh = review.get("must_flag_hits") or []
            if not isinstance(mfh, list):
                mfh = []
            review["must_flag_hits"] = _without_reviewer_page_limit_hits(
                [str(h).strip() for h in mfh if str(h).strip()], self._log,
            )
            review["hit_places"] = _read_hit_places(review.get("must_flag_places"), review_ids)
            review["must_flag_hits"] += _unsupported_claim_hit(state, review["must_flag_hits"])
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
            # (log and the human-review panel in all
            # three interfaces) and the verdict is left to the reviewer.
            numeric_warnings = self._reference_check_hits(state)
            if numeric_warnings:
                review["numeric_oracle_warnings"] = numeric_warnings
            # Advisory too: numbers and figures the topic asked for that the
            # experiment does not have. Shown to the human, never forced.
            goal_notes = self._goal_coverage_notes(state)
            if goal_notes:
                review["goal_coverage_notes"] = goal_notes
            review["must_flag_hits"] += self._unplaced_topic_answers(paper_md, state)
            # Forced, unlike the advisory number check above: a statistic the
            # paper mislabels is caught by recomputing it, so the finding can
            # name the contradiction instead of guessing at one.
            review["must_flag_hits"] += self._statistics_claim_hits(paper_md, state)
            # Forced: a claim FI's own records contradict (core/record_claims.py).
            record_hits = self._record_contradiction_hits(paper_md, state)
            review["must_flag_hits"] += record_hits
            # Forced, and text-only: what the paper states about the run against FI's records (core/paper_facts.py).
            review["must_flag_hits"] += self._paper_record_hits(paper_md, state)
            typed_pass = self._typed_pass_hits(paper_md, state)
            if typed_pass:
                review["typed_pass_hits"] = typed_pass
            # Forced as well: nothing checked this draft's citations.
            review["must_flag_hits"] += _citations_unchecked(state)
            if page_record is not None:
                review["page_limit"] = page_record
            update: QuestState = {"review": review}
            if pruned is not None:
                update["claim_grounding"] = pruned
            # Counted here so the router can cap the rewrites for it, and end the quest honestly after them.
            update["record_rewrites"] = (int(state.get("record_rewrites") or 0) + 1) if record_hits else 0
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
            # iteration and the loop could run unbounded.
            other_hits = self._note_design_objections(state, review)
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
                        "must_flag_hits": [], "status": "error", "error": str(e)}
            prompt = f"{prefix}\n\n{base_prompt}"
            # A reply that parses but names no "verdict" at all (e.g.
            # {"comment": "..."}) is exactly as unusable as no reply — a
            # non-empty dict alone is not proof the persona actually answered.
            # Asked once more; then the panelist has not reviewed, the same as
            # one that could not be asked.
            parsed: Any = None
            for _ask in range(2):
                try:
                    text = await self._chat(prompt, node=f"review_panel.{name}")
                except _ModelAnswerProblem:
                    raise
                except Exception as e:  # a model that cannot be used (ProviderUnusable) is not caught here: the gather raises it
                    self._log.warning("[review] panelist %s could not be asked (%s)", name, e)
                    return {"persona": name, "status": "call_failed", "error": _one_line(e, 300)}
                parsed = _parse_json_lenient(text)
                if _names_a_verdict(parsed):
                    break
                self._log.warning("[review] panelist %s named no verdict%s", name, "; asking again" if _ask == 0 else "")
            else:
                return {"persona": name, "status": "call_failed", "no_verdict": True,
                        "error": "answered twice without a verdict (accept or revise)"}
            mfh = parsed.get("must_flag_hits") or []
            if not isinstance(mfh, list):
                mfh = []
            return {
                "persona": name,
                "verdict": parsed["verdict"].strip().lower(),
                "score": parsed.get("score") if isinstance(parsed.get("score"), (int, float)) else 3,
                "strengths": parsed.get("strengths") or [],
                "weaknesses": parsed.get("weaknesses") or [],
                "suggestions": parsed.get("suggestions") or [],
                "blocking": parsed.get("blocking") or "",
                "design_parts": [str(x).strip() for x in (parsed.get("design_parts") or []) if str(x).strip()]
                if isinstance(parsed.get("design_parts"), list) else [],
                "must_flag_hits": [str(h).strip() for h in mfh if str(h).strip()],
                "hit_places": _read_hit_places(parsed.get("must_flag_places"), review_ids),
                "status": "ok",
                # Which model was asked for, and which provider and model actually answered (after a fallback, the
                # fallback's): panelists answered by one model are one reviewer's view in several roles.
                "requested_model": self._reviewer_model(name),
                "actual_provider": self._chat_provenance(f"review_panel.{name}").get("provider"),
                "actual_model": self._chat_provenance(f"review_panel.{name}").get("model"),
                "actual_vendor": self._chat_provenance(f"review_panel.{name}").get("vendor"),
                "fallback": bool(self._chat_provenance(f"review_panel.{name}").get("fallback")),
                # False when the transport does not say which model served the call (some CLIs, a VS Code extension
                # that sends no model name, a router alias): ``actual_model`` is then only the model asked for.
                "actual_model_reported": bool(self._chat_provenance(f"review_panel.{name}").get("reported")),
            }

        # return_exceptions=True is defense in depth: a panelist must never be
        # able to abort the whole (post-write) review and forfeit the paper's
        # outputs. Drop any unexpected raise (it stops the quest below, as a
        # reviewer that could not be asked) and propagate genuine cancellation.
        panel_results_raw = await asyncio.gather(
            *(run_persona(n) for n in panel_names),
            return_exceptions=True,
        )
        panel_results: list[dict[str, Any]] = []
        for r in panel_results_raw:
            if isinstance(r, asyncio.CancelledError):
                raise r
            if isinstance(r, _ModelAnswerProblem):
                raise r
            if isinstance(r, ProviderUnusable):
                raise r
            if isinstance(r, BaseException):
                self._log.warning("[review] a panelist raised unexpectedly: %r", r)
            elif isinstance(r, dict):
                panel_results.append(r)
        # A panelist the model could not be asked for (a usage limit, a provider that is down) has not reviewed the
        # paper. Standing in a neutral accept for it read as a review; the quest stops instead and asks again on resume.
        unasked = [r for r in panel_results if r.get("status") == "call_failed"]
        if unasked or len(panel_results) < len(panel_names):
            self._pause_for_review_unavailable(paper_path, [
                f"the {r['persona']} reviewer "
                + (r["error"] if r.get("no_verdict") else f"could not be asked ({r.get('error') or 'no reply'})")
                for r in unasked
            ] or ["a reviewer could not be asked"])
            # Reached only if a resume ever answered this pause with a value (none does today): the panelists that did
            # not review stay in panel_results with their status, cast no vote in the aggregator, and make the
            # review's status "error", so nothing takes the result as a panel's accept.
        collapsed = self._review_models_collapsed(panel_results)
        if collapsed is not None:
            self._pause_for_review_models_collapsed(paper_path, panel_results, collapsed)
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
        except _ModelAnswerProblem:
            raise
        except Exception as e:
            self._log.warning("[review] moderator call failed: %s", e)
            mod_parsed = {}

        # Merge: numeric/voting fields from the deterministic aggregator
        # always win; prose fields prefer the moderator's version when
        # present.
        review: dict[str, Any] = {**agg}
        review["hit_places"] = {k: v for r in panel_results for k, v in (r.get("hit_places") or {}).items()}
        review["must_flag_hits"] = _without_reviewer_page_limit_hits(
            review.get("must_flag_hits") or [], self._log,
        )
        review["must_flag_hits"] += _unsupported_claim_hit(state, review["must_flag_hits"])
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
        numeric_warnings = self._reference_check_hits(state)
        if numeric_warnings:
            review["numeric_oracle_warnings"] = numeric_warnings
        goal_notes = self._goal_coverage_notes(state)
        if goal_notes:
            review["goal_coverage_notes"] = goal_notes
        unplaced = self._unplaced_topic_answers(paper_md, state)
        if unplaced:
            review["must_flag_hits"] = [*(review.get("must_flag_hits") or []), *unplaced]
        # Forced, as on the single-reviewer path: a recomputed contradiction,
        # not a pattern match over prose.
        stat_hits = self._statistics_claim_hits(paper_md, state)
        if stat_hits:
            review["must_flag_hits"] = [*(review.get("must_flag_hits") or []), *stat_hits]
        record_hits = self._record_contradiction_hits(paper_md, state)
        if record_hits:
            review["must_flag_hits"] = [*(review.get("must_flag_hits") or []), *record_hits]
        stated_hits = self._paper_record_hits(paper_md, state)
        if stated_hits:
            review["must_flag_hits"] = [*(review.get("must_flag_hits") or []), *stated_hits]
        typed_pass = self._typed_pass_hits(paper_md, state)
        if typed_pass:
            review["typed_pass_hits"] = typed_pass
        unchecked = _citations_unchecked(state)
        if unchecked:
            review["must_flag_hits"] = [*(review.get("must_flag_hits") or []), *unchecked]
        if page_record is not None:
            review["page_limit"] = page_record

        update: QuestState = {"review": review, "review_panel": panel_results}
        if pruned is not None:
            update["claim_grounding"] = pruned
        update["record_rewrites"] = (int(state.get("record_rewrites") or 0) + 1) if record_hits else 0
        # As on the single-reviewer path: count a review that sends the
        # experiment back, so the router can cap those re-executes.
        if _review_sends_the_experiment_back(review, state):
            update["code_reexecutes"] = int(state.get("code_reexecutes") or 0) + 1
        # Bump iteration on EITHER verdict=revise OR a non-empty
        # must_flag_hits list. Without the must-flag clause, a malformed
        # persona response that recorded ``verdict=accept`` alongside a
        # must-flag hit would route to revise (via _route_after_review)
        # without consuming iteration budget — the loop could spin.
        other_hits = self._note_design_objections(state, review)
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
        paper_md_state = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
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
            # Whether a reviewer actually gave this verdict ("ok"), and each panelist's own status, so a stand-in
            # verdict is never shown, or auto-accepted, as a review.
            "review_status": "ok" if _review_was_real(review) else str(review.get("status")),
            "panel": [
                {"persona": r.get("persona"), "verdict": r.get("verdict"), "score": r.get("score"),
                 "status": r.get("status", "ok"), **({"error": r["error"]} if r.get("error") else {})}
                for r in (state.get("review_panel") or []) if isinstance(r, dict)
            ],
            "strengths": review.get("strengths") or [],
            "weaknesses": review.get("weaknesses") or [],
            "suggestions": review.get("suggestions") or [],
            "must_flag_hits": review.get("must_flag_hits") or [],
            # Advisory, never blocking: numbers the arithmetic check could not
            # reconcile with the results. Kept apart from must_flag_hits so
            # every UI can label them differently -- a regex over prose is
            # not grounds for a forced rewrite, but a human should see it.
            "numeric_oracle_warnings": review.get("numeric_oracle_warnings") or [],
            # Advisory: numbers and figures the topic asked for that the
            # experiment does not have (core/goal_coverage.py).
            "goal_coverage_notes": review.get("goal_coverage_notes") or [],
            "rationale": review.get("rationale", ""),
            "paper_md_path": paper_md_path,
            # Accumulated user-feedback history across refine
            # iterations. Surfaced to the human-review UI so a
            # reviewer can see what was asked for last time.
            "feedback_history": list(state.get("feedback_history") or []),
            # What a refine asked for that the run computed and the paper still leaves out (``_node_write``).
            "not_in_paper": list(state.get("extend_unreported") or []),
        }
        # A figure request of the person's last refine that the redraw could not carry out (it failed twice).
        not_redrawn = [str(n) for n in state.get("layout_not_redrawn") or [] if str(n).strip()]
        if not_redrawn:
            snapshot["layout_not_redrawn"] = not_redrawn
        # A search for the best design: the person's last refine FI did not carry out (a new study, or unclear), and
        # how to ask for more (core/optimise_refine.py).
        refused = state.get("optimise_refine") or {}
        not_done = ([str(refused.get("says"))] if refused.get("kind") in (_optimise_refine.NEW_STUDY,
                                                                          _optimise_refine.UNCLEAR) else [])
        if not_done:
            snapshot["refine_not_done"] = not_done
        search_hint = self._optimise_refine_hint(state)
        if search_hint:
            snapshot["search_further"] = search_hint
        # Before a person accepts: what the result does not guarantee, its most important evidence gaps, and the one
        # question they answer (core/acceptance.py). Worked out here, written nowhere.
        before_accept = _acceptance.shown(self._write_evidence(state, write=False))
        snapshot["before_accept"] = before_accept
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
            headline=(
                f"review the result (verdict: {verdict}, score: {review.get('score')})"
                if _review_was_real(review) else
                f"review the result yourself: no reviewer gave a verdict (review status {review.get('status')})"
            ),
            steps=[
                *not_done,
                *([f"Your figure request was NOT applied: {'; '.join(n[:200] for n in not_redrawn)}. Two tries at "
                   "redrawing the figures did not work, so they are as they were (the error is in .fi/run.log). Refine again to "
                   "retry, or say it another way."] if not_redrawn else []),
                *(f"{item}; the values are in the run's results. Refine again to have them written in, or accept "
                  "without them."
                  for item in state.get("extend_unreported") or []),
                "Accept, reject, or refine the paper in the panel "
                "(Web / VSCode), or at the CLI prompt.",
                "Before you accept, read what this result does not guarantee and its gaps: "
                + " ".join(line if line.endswith((".", ")", "…")) else line + "." for line in _acceptance.lines(before_accept))
                + f" Then answer: {_acceptance.QUESTION} Only yes lets the result count as publication-ready; "
                "partly needs a short note saying what you do not accept, and it and \"I did not check\" each leave a "
                "gap; \"no\" does not accept: refine instead, or look again.",
                "Headless run? "
                f"`fi --resume {self.quest_id} --accept yes` (or `--accept partly \"what you do not accept\"` / "
                "`--accept not-checked`; or `--reject` / `--refine \"what to change\"`). Add `--approve-as <your name>` "
                "to record your name rather than your login name.",
                "Refine sends your notes back to the writing step first; if a point needs a new "
                "experiment, FI goes back to the design.",
                *([search_hint] if search_hint else []),
            ],
            payload={"human_review": snapshot},
        )
        # Resume: ``payload`` is what the callback / web POST returned.
        # The node wrote the snapshot again on this pass (a resumed node runs from its start), but the review is
        # answered now: the snapshot goes, or the evidence assessed before the quest ends reads "the review is waiting
        # for your decision" for a review already accepted (automatically, by ``auto_accept_on_pass``, included).
        try:
            (self.fi_dir / "human_review.json").unlink(missing_ok=True)
        except OSError:
            pass
        # Validate + normalise so a malformed answer doesn't propagate
        # into the routing layer.
        action = "accept"
        feedback = ""
        if isinstance(payload, dict):
            raw_action = str(payload.get("action") or "accept").strip().lower()
            if raw_action in ("accept", "reject", "refine"):
                action = raw_action
            feedback = str(payload.get("feedback") or "").strip()
        # ``refine`` with no text falls back to ``accept`` — a 0-char
        # refinement is indistinguishable from approval.
        if action == "refine" and not feedback:
            action = "accept"

        update: QuestState = {
            "human_feedback": {"action": action, "feedback": feedback},
            # Told at this review; the next review tells only what the next refine could not do.
            "layout_not_redrawn": [],
            "optimise_refine": {},
            # Who accepted it (the run loop stamps it: a person with their answer, or automatic). An accept the run
            # loop did not stamp is recorded as automatic: nobody is recorded as having looked.
            "acceptance": {},
        }
        if action == "accept":
            given = payload.get("acceptance") if isinstance(payload, dict) else None
            acceptance = (dict(given) if isinstance(given, dict) and given.get("by") in ("person", "automatic")
                          else _acceptance.automatic("not recorded"))
            # Which paper was accepted: a later change to it (a re-run) is not covered by this accept.
            try:
                acceptance["paper_sha256"] = _receipts.sha256(Path(paper_md_path).read_bytes())
            except OSError:
                acceptance["paper_sha256"] = ""
            update["acceptance"] = acceptance
            self._audit("result_accepted", **acceptance)
            self._log.info("[human_feedback] accepted by %s (via %s%s%s%s)", acceptance.get("by"), acceptance.get("via"),
                           f", who: {acceptance['who']}" if acceptance.get("who") else "",
                           f", answer: {acceptance['answer']}" if acceptance.get("answer") else "",
                           f", not accepted: {acceptance['note']}" if acceptance.get("note") else "")
            if acceptance.get("evidence_sha256_at_accept"):
                self._log.warning("[human_feedback] the evidence record worked out when the quest resumed differs from "
                                  "the one whose limits were shown with the question; the receipt keeps both "
                                  "fingerprints (needs/EVIDENCE.json, acceptance)")
        # When the user refines, bump iteration so the loop budget is
        # consumed and the design node sees an explicit "we're in a
        # revise pass" signal (same convention the verdict-driven
        # revise loop uses). Also append to ``feedback_history`` so
        # the design node sees the cumulative refinement requests
        # instead of only the latest one — important when a quest
        # goes through multiple revise passes and the user wants the
        # rewriter to honour all prior asks, not just the last.
        if action == "refine":
            # A search for the best design reads the refine first (core/optimise_refine.py). A request it does not
            # carry out (a new study, or unclear) is not kept: no later step may act on it, and the pause says why.
            request = self._optimise_refine_request(state, feedback)
            update["optimise_refine"] = request
            if request.get("kind") in (_optimise_refine.NEW_STUDY, _optimise_refine.UNCLEAR):
                self._log.warning("[optimise] %s", request.get("says"))
                self._progress(str(request.get("says")))
                return update
        if action == "refine":
            update["iteration"] = state.get("iteration", 0) + 1
            history = list(state.get("feedback_history") or [])
            # The review is shown the notes of this gate answer, not the ones an earlier gate answered.
            update["feedback_rounds_from"] = len(history)
            history.append({
                "iteration": state.get("iteration", 0),
                "text": feedback,
            })
            update["feedback_history"] = history
            if (update.get("optimise_refine") or {}).get("kind") == _optimise_refine.SEARCH:
                # The search answers this refine, not the writer: the paper written after it is a whole new draft,
                # whose "Best design found" section says what the continued search found and whether the target was
                # reached. The note is kept (it counts as a refine) but marked, so neither the writer, the review nor a
                # later design reads it as a change to make; the run starts with fresh repair counters, as after a new
                # script.
                history[-1] = {**history[-1], "answered_by": "search"}
                update["feedback_rounds_from"] = len(history)
                update["refine_written_for"] = _refine_count({**state, **update})  # type: ignore[arg-type]
                update.update(_FRESH_SCRIPT)  # type: ignore[typeddict-item]
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
        if self._repairs_go_to == "agent" and _writes_code(node):
            raise RepairForAgent(str(node))  # the code is the coding agent's: the check reports to it (core/agent_build.py)
        assert self._client is not None
        if node == "plan_revise":
            self._plan_chat_calls = getattr(self, "_plan_chat_calls", 0) + 1  # counted before it is made, so a failure counts
        temp = (
            temperature if temperature is not None
            else _temperature_for_node(node)
        )
        messages = [{"role": "user", "content": prompt}]
        client, _reports = await self._client_for(node)
        response, served = await self._recorded_call(
            node or "", messages,
            lambda: client.chat(messages, temperature=temp, model=self._model_for_node(node), node=node or ""),
        )
        # The model that answered THIS call (not the client's shared latest, which a call made meanwhile can change).
        self._log_chat_cost(node=node or "", messages=messages, response=response, model=served.get("model") or None,
                            usage=served.get("usage") if isinstance(served.get("usage"), dict) else None)
        if node:
            # Who answered THIS call (core/provider.py::LAST_CALL, the task's own), so a fallback that actually served
            # it is recorded truthfully and calls made at the same time do not overwrite each other. The client's
            # shared attributes are the fallback for a transport that sets nothing.
            self._last_chat[node] = {
                "provider": (served.get("provider") or getattr(client, "last_provider", None)
                             or (step[1].name if (step := node_provider_for(getattr(self.config.provider, "node_providers", None), node))
                                 else self.config.provider.name)),
                "model": (served.get("model") or getattr(client, "last_model", None) or self._model_for_node(node)
                          or self.config.provider.model),
                "fallback": bool(served.get("fallback")),
                "reported": bool(served.get("reported")),
                # The model's vendor as the connection named it (the VS Code extension's), when it did.
                **({"vendor": served["vendor"]} if served.get("vendor") else {}),
                # The same hashes as this call's line in .fi/model_calls.jsonl, so the two can be joined -- and its id, the
                # model asked for and why the answer ended (as the engine recorded them; nothing the model said).
                **({"call_id": served["call_id"]} if served.get("call_id") else {}),
                **({"requested_model": rm} if (rm := self._model_for_node(node) or self.config.provider.model) else {}),
                **({"finish_reason": served["finish_reason"]} if served.get("finish_reason") else {}),
                "prompt_hash": _attempts.prompt_sha(messages),
                "response_hash": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            }
        return response

    async def _chat_json(
        self, prompt: str, *, node: str, want: tuple[str, ...], usable: Callable[[dict[str, Any]], bool] | None = None,
        why_unusable: Callable[[dict[str, Any]], str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """A call whose answer is one JSON object carrying ``want`` (a required check's: the design audit, the
        evidence gate, the claim check). When the reply cannot be used (no such object in it, or ``usable`` says it
        is not), the same request is asked ONCE more with a short instruction: one object of the asked-for shape, no
        fence, no prose, no second object, nothing long. Returns the reply and the object read from it (``{}`` when
        neither answer could be used: the caller's own rule for an unreadable answer then applies, unchanged)."""
        ok = usable or (lambda p: all(k in p for k in want))
        text = await self._chat(prompt, node=node)
        parsed = _parse_json_lenient(text, node=node, want=want) or {}
        if isinstance(parsed, dict) and ok(parsed):
            return text, parsed
        objects = _top_level_json_objects(text or "")
        if not (text or "").strip():
            wrong = "it was empty"
        elif not objects:
            wrong = f"it was {len(text):,} characters with no JSON object in it"
        elif len(objects) > 1:
            wrong = (f"it was {len(text):,} characters holding {len(objects)} separate JSON objects, not one object of "
                     "the asked-for shape")
        elif all(k in objects[0] for k in want):
            wrong = f"it was {len(text):,} characters and its {', '.join(want)} was not one of the values asked for"
        else:
            wrong = f"it was {len(text):,} characters and its JSON object did not carry {', '.join(want)} as asked"
        if isinstance(parsed, dict) and parsed and why_unusable is not None and (reason := why_unusable(parsed)):
            wrong = reason  # a reply that is an object of the right shape but says nothing: the caller names why
        if not objects and (text or "").strip():
            merged = await self._reask_unreadable_keys(prompt, text, node=node, want=want, ok=ok)
            if merged is not None:
                return merged[0], merged[1]
        self._log.warning("[%s] the reply could not be used (%s); asking once more for the whole answer (one short JSON "
                          "object): nothing in it could be read", node, wrong)
        echo = (f"\nYour previous reply, for reference (say the same things, in the asked-for shape):\n\n{text}\n"
                if len(text) <= self._REANSWER_ECHO_CHARS else "")
        retry = self._prompts["json_reanswer"].substitute(original_prompt=prompt, what_was_wrong=wrong, keys=", ".join(want), earlier_reply=echo)
        text = await self._chat(retry, node=node)
        parsed = _parse_json_lenient(text, node=node, want=want) or {}
        if isinstance(parsed, dict) and ok(parsed):
            self._log.info("[%s] the second, shorter answer could be read", node)
            return text, parsed
        self._log.warning("[%s] the second answer could not be used either (%d characters)", node, len(text or ""))
        return text, parsed if isinstance(parsed, dict) else {}

    async def _reask_unreadable_keys(
        self, prompt: str, text: str, *, node: str, want: tuple[str, ...], ok: Callable[[dict[str, Any]], bool],
    ) -> tuple[str, dict[str, Any]] | None:
        """A reply whose JSON object does not parse as a whole: what IS readable is kept (:func:`_readable_json_keys`),
        and the model is asked again only for the keys that were unreadable or are missing, each by its path
        (``protocol.tolerance``: the key ``tolerance`` inside ``protocol``), never for the whole answer
        (``agents/json_reanswer_part.md``). The new values are merged at those paths, so the readable keys around them
        are not touched. ``(text of the merged object, the merged object)``; ``None`` when nothing in the reply is
        readable, or a key is written twice, so the caller asks for the whole answer."""
        found = _readable_json_keys(text)
        if found is None:
            return None
        readable, bad, reason = found
        missing = list(bad) + [(k,) for k in want if k not in readable and (k,) not in bad]
        if not missing:
            if ok(readable):
                self._log.info("[%s] the reply did not parse as a whole, but every key asked for could be read from it", node)
                return json.dumps(readable, ensure_ascii=False), readable
            return None
        names = ", ".join(f"`{'.'.join(path)}`" for path in missing)
        wrong = f"{names} could not be read ({reason})" if bad else f"it did not carry {names}"
        self._log.warning("[%s] the reply was read except %s; asking again only for %s", node, wrong, names)
        echo = (f"\nYour previous reply, for reference (say the same things for these keys):\n\n{text}\n"
                if len(text) <= self._REANSWER_ECHO_CHARS else "")
        retry = self._prompts["json_reanswer_part"].substitute(
            original_prompt=prompt, what_was_wrong=wrong, keys=names, earlier_reply=echo)
        again = await self._chat(retry, node=node)
        if node == "plan_revise":
            self._keep_plan_answer(again)
        reply = _parse_json_lenient(again, node=node, want=tuple(dict.fromkeys(path[0] for path in missing))) or {}
        merged = json.loads(json.dumps(readable))
        for path in missing:
            value: Any = reply
            for key in path:
                value = value.get(key, _NOT_GIVEN) if isinstance(value, dict) else _NOT_GIVEN
            if value is _NOT_GIVEN:
                continue
            holder: Any = merged
            for key in path[:-1]:
                holder = holder.setdefault(key, {}) if isinstance(holder, dict) else None
                if not isinstance(holder, dict):
                    break
            if isinstance(holder, dict):
                holder[path[-1]] = value
        if ok(merged):
            self._log.info("[%s] the keys asked again could be read and were merged into the readable part", node)
        else:
            self._log.warning("[%s] the merged answer could not be used either", node)
        return json.dumps(merged, ensure_ascii=False), merged

    def _seed_call_counts(self, state: Any) -> None:
        """Start this run's tally of model calls from the one the quest's state kept (a resumed quest)."""
        if self.__dict__.get("_model_call_counts_seeded"):
            return
        self.__dict__["_model_call_counts_seeded"] = True
        kept = state.get("model_call_counts") if isinstance(state, dict) else None
        counts = self.__dict__.setdefault("_model_call_counts", {})
        for node, n in dict(kept or {}).items():
            counts[node] = max(int(counts.get(node, 0) or 0), int(n or 0))

    async def _recorded_call(self, node: str, messages: Any, call: Any, *,
                             requested_model: str | None = None, tolerated: bool = False) -> tuple[Any, dict[str, Any]]:
        """Make one model call (``call()``) and write every attempt of it in the quest's record of its calls: each
        attempt that failed before it answered or gave up (a retry, a provider the fallback chain passed over), then
        the answer or the final failure. Returns the answer and who answered it (this task's LAST_CALL). ``tolerated``:
        a call whose failure the caller leaves out by design (an ensemble member), so a usage limit does not stop the step."""
        attempts: list[dict[str, Any]] = []
        token = _CALL_ATTEMPTS.set(attempts)
        _LAST_CALL.set(None)
        # This call's own id: one left by an earlier call of the node must not be read if this call's record fails.
        self.__dict__.setdefault("_last_call_id", {}).pop(node, None)
        holder, holder_token = _thinking.open_holder(want=self._keeps_thinking())
        try:
            try:
                response = await call()
            except Exception as exc:
                failed = _LAST_CALL.get() or {}
                if isinstance(exc, _ModelAnswerProblem) and not exc.node:
                    exc.node = node  # the step a pause names (a CLI does not know it)
                final_id = self._record_attempts(node, messages, attempts, requested_model, final=exc)
                if not (attempts and attempts[-1].get("exc") is exc):
                    usage = getattr(exc, "usage", None) if isinstance(getattr(exc, "usage", None), dict) else None
                    finish = getattr(exc, "finish_reason", None)
                    # Only the final record's id is this call's; a failed attempt's id must not stand in for it.
                    self.__dict__.setdefault("_last_call_id", {}).pop(node, None)
                    final_id = self._record_model_call(
                        node, messages, None,
                        served={**failed, "reported": False, **({"finish_reason": finish} if finish else {})},
                        outcome=_outcome_of(exc), requested_model=requested_model, usage=usage)
                    self._cost_of_failed_attempt(node, getattr(exc, "model", None) or failed.get("model"), usage,
                                                 messages)
                self._save_thinking(node, holder, requested_model, failed, outcome=_outcome_of(exc), call_id=final_id)
                if _is_fatal_provider_error(exc) and not tolerated:
                    # A usage limit or a dead login is no answer: the step that asked is not kept, whatever it catches
                    # (see _audited), so it runs again when the quest resumes instead of deciding on a call never made.
                    self.__dict__["_provider_unavailable"] = f"{node or 'a step'}: {' '.join(str(exc).split())[:400]}"
                    self.__dict__["_provider_limit"] = exc
                    self.__dict__["_provider_signin"] = self._sign_in_message(exc, requested_model) if _provider_is_auth_error(exc) else ""
                    raise ProviderUnusable(exc) from exc  # not catchable as a failure: see ProviderUnusable
                raise
            served = dict(_LAST_CALL.get() or {})
            self._record_attempts(node, messages, attempts, requested_model)
            self.__dict__.setdefault("_last_call_id", {}).pop(node, None)
            final_id = self._record_model_call(node, messages, response, served=served, usage=served.get("usage"),
                                               requested_model=requested_model)
            self._save_thinking(node, holder, requested_model, served, outcome="ok", call_id=final_id)
            if not requested_model:
                try:
                    self._note_served_model(node, served)
                except Exception:  # noqa: BLE001 -- a note never touches the quest (nor a stand-in engine's call)
                    pass
            if final_id:
                served["call_id"] = final_id  # this call's own id, not the node's shared latest (a same-node call may run at once)
            return response, served
        finally:
            _thinking.close_holder(holder_token)
            _CALL_ATTEMPTS.reset(token)

    def _keeps_thinking(self) -> bool:
        """``output.save_thinking``: whether a model's reasoning is kept, and so whether a connection that has to ask
        for it (the VS Code bridge) asks."""
        try:
            return bool(self.config.output.save_thinking)
        except AttributeError:  # a stand-in config
            return False

    def _say_extension_older(self, holder: dict[str, Any]) -> None:
        """The FI extension in VS Code answered with an older bridge protocol than this FI expects: one plain line, in
        run.log and on the console, said once per engine. Never stops the quest."""
        said = self.__dict__.setdefault("_thinking_said", set())
        if holder.get("extension_older") and "extension_older" not in said:
            said.add("extension_older")
            from .vscode_bridge import EXTENSION_OLDER_NOTE
            self._progress(EXTENSION_OLDER_NOTE)

    def _say_call_notes(self, node: str, holder: dict[str, Any], served: dict[str, Any]) -> None:
        """What the streamed Ollama call noted on the call's holder, in plain run.log lines: when its output began (each
        call), which sampling a request to think used (once per model), and that the model's reasoning went nowhere so
        FI asked again without it (once per step)."""
        said = self.__dict__.setdefault("_thinking_said", set())
        for line in holder.get("timings") or []:
            self._log.info("[%s] %s", node, line)
        sampling = str(holder.get("sampling") or "")
        if sampling and ("sampling", sampling) not in said:
            said.add(("sampling", sampling))
            self._log.info("[thinking] %s", sampling)
        loop = holder.get("loop")
        if isinstance(loop, dict) and ("loop", node) not in said:
            said.add(("loop", node))
            if loop.get("kind") == "length":
                what = (f"used up all the room there was ({loop.get('chars')} characters of reasoning) without giving an "
                        "answer")
            else:
                unit = "lines" if loop.get("kind") == "lines" else "characters"
                what = f"went round in circles (the same {loop.get('block')} {unit} repeated {loop.get('repeats')} times)"
            self._log.info("[thinking] The model's reasoning at the step `%s` %s, so FI asked again without reasoning; "
                           "this step's reasoning is not kept.", node, what)

    def _say_once_about_thinking(self, node: str, holder: dict[str, Any], served: dict[str, Any], outcome: str,
                                 text: str) -> None:
        """Plain run.log lines about a call's reasoning, each said once: the model refused the request for its
        reasoning (once per model), or this step's connection returned none (once per step)."""
        said = self.__dict__.setdefault("_thinking_said", set())
        declined = str(holder.get("declined") or "")
        model = str(served.get("model") or served.get("provider") or "the model")
        empty = holder.get("empty_parts") or 0
        provider = str(served.get("provider") or "")
        # The connection, not the vendor that answered: through VS Code the answer names Copilot's vendor (copilot,
        # customendpoint, ollama-models), never "vscode_extension".
        # A fallback provider that answered instead (it names itself, e.g. "ollama") is not the VS Code route.
        configured = getattr(getattr(getattr(self, "config", None), "provider", None), "name", "")
        via_vscode = provider == "vscode_extension" or (
            configured == "vscode_extension" and provider not in _FI_PROVIDER_NAMES)
        if declined and ("declined", model) not in said:
            said.add(("declined", model))
            self._log.info("[thinking] %s did not accept FI's request for its reasoning, so FI asked again without it "
                           "(%s); its answers are used as usual", model, declined[:200])
        if (via_vscode and outcome == "ok" and not str(holder.get("text") or "").strip()
                and ("vscode_none", model) not in said):
            said.add(("vscode_none", model))
            why = ("VS Code passed it on with no text in it (for an OpenAI model Copilot does not ask for the reasoning "
                   "summary, so it arrives encrypted only)" if empty else
                   "VS Code did not pass any on (not every model in it returns its reasoning)")
            self._log.info("[thinking] %s returned no reasoning through VS Code: %s. To keep an OpenAI model's reasoning "
                           "use the codex_cli connection (it saves the reasoning summary); only its answers are kept.",
                           model, why)
        if (outcome == "ok" and not text.strip() and provider in _thinking.CANNOT_RETURN
                and ("cannot", provider) not in said):
            said.add(("cannot", provider))
            self._log.info("[thinking] %s cannot return a model's reasoning: %s; only its answers are kept (set "
                           "output.save_thinking: false to stop this note)", provider, _thinking.CANNOT_RETURN[provider])
        if (outcome == "ok" and not text.strip() and provider not in _thinking.CANNOT_RETURN
                and not holder.get("loop") and not via_vscode and ("none", node) not in said):
            said.add(("none", node))
            self._log.info("[thinking] %s: this model/connection returned no reasoning for this step (only its answer "
                           "and any reasons it wrote in it are kept)", node)

    def _save_thinking(self, node: str, holder: dict[str, Any], requested_model: str | None,
                       served: dict[str, Any], *, outcome: str, call_id: str | None = None) -> None:
        """One line in ``.fi/thinking.jsonl`` for a call whose connection handed back the model's reasoning: the text
        (credentials and the home folder removed), which call it belongs to (``call_id``, the line of
        ``.fi/model_calls.jsonl``), who answered it and a note that it is the model's own account. One line is cut at
        :data:`thinking_capture.THINKING_LINE_CHARS` and the file stops growing at ``THINKING_FILE_BYTES``. Not sealed,
        not evidence, and never in the way of the quest."""
        text = holder.get("text") or ""
        fi_dir = getattr(self, "fi_dir", None)
        try:
            self._say_extension_older(holder)
        except Exception:  # noqa: BLE001 -- a log line never touches the quest (nor a stand-in engine's call)
            pass
        try:
            self._say_call_notes(node, holder, served)
        except Exception:  # noqa: BLE001 -- a log line never touches the quest (nor a stand-in engine's call)
            pass
        if fi_dir is None or not self._keeps_thinking():
            return
        try:
            self._say_once_about_thinking(node, holder, served, outcome, text)
        except Exception:  # noqa: BLE001 -- a log line never touches the quest (nor a stand-in engine's call)
            pass
        if not text.strip():
            return
        try:
            path = fi_dir / _thinking.THINKING_FILE
            if path.exists() and path.stat().st_size >= _thinking.THINKING_FILE_BYTES:
                return
            kept = _audit_log.redact_text(text, whole=True)
            if len(kept) > _thinking.THINKING_LINE_CHARS:
                kept = (kept[:_thinking.THINKING_LINE_CHARS]
                        + f"... [{len(kept) - _thinking.THINKING_LINE_CHARS} more characters not kept]")
            _attempts.append(fi_dir, _thinking.THINKING_FILE, {
                "quest_id": getattr(self, "quest_id", ""), "node": node,
                "call_id": call_id,
                "provider": served.get("provider"), "model": served.get("model"), "outcome": outcome,
                "requested_model": requested_model or self._model_for_node(node) or self.config.provider.model or None,
                "note": _thinking.THINKING_NOTE,
                "thinking": kept,
            })
        except Exception as e:  # noqa: BLE001 -- a keepsake never touches the quest
            self._log.debug("[thinking] not kept: %r", e)

    def _record_attempts(self, node: str, messages: Any, attempts: list[dict[str, Any]], requested_model: str | None,
                         *, final: BaseException | None = None) -> str | None:
        """Write each failed attempt; returns the id of the LAST attempt's record, or None when that record could not be
        written (an earlier attempt's id is never returned nor left as the node's latest in its place)."""
        last_id: str | None = None
        for a in attempts:
            self.__dict__.setdefault("_last_call_id", {}).pop(node, None)
            usage = a.get("usage") if isinstance(a.get("usage"), dict) else None
            last_id = self._record_model_call(
                node, messages, None, outcome=str(a.get("error") or "error"), requested_model=requested_model,
                served={"provider": a.get("provider"), "model": a.get("model"), "fallback": bool(a.get("fallback")),
                        **({"finish_reason": a["finish_reason"]} if a.get("finish_reason") else {})},
                usage=usage,
            )
            self._cost_of_failed_attempt(node, a.get("model"), usage, messages)
        return last_id

    def _cost_of_failed_attempt(self, node: str, model: Any, usage: dict[str, Any] | None, messages: Any) -> None:
        """A failed attempt that did answer (an answer cut off at its limit) was paid for: its cost row too."""
        if not usage or getattr(self, "fi_dir", None) is None:
            return
        try:
            append_cost_row(self.fi_dir, node=node, model=str(model or ""), usage=usage, messages=messages,
                            response=None, ledger=False)
        except Exception as e:  # noqa: BLE001 -- a cost row never touches the quest
            self._log.debug("[cost] failed attempt not recorded: %r", e)

    def _reports_for(self, node: str) -> bool:
        """Whether the connection ``node``'s calls go on names the model that answered (a step on its own provider has
        its own connection; known once its first call has built the client)."""
        step = node_provider_for(getattr(self.config.provider, "node_providers", None), node)
        if step is not None:
            built = self.__dict__.get("_step_clients", {}).get(step[0])
            if built is not None:
                return bool(built[1])
        return bool(getattr(self, "_reports_model", False))

    def _record_model_call(self, node: str, messages: Any, response: Any, *, served: dict[str, Any] | None = None,
                           outcome: str = "ok", usage: dict[str, Any] | None = None,
                           requested_model: str | None = None) -> str | None:
        """One line in the quest's record of every model call it made (``.fi/model_calls.jsonl``,
        core/attempt_records.py::model_call_row): the hashes of the prompt and the answer, never their text.
        Best-effort: a line that cannot be written is counted, never in the way of the quest."""
        fi_dir = getattr(self, "fi_dir", None)
        if fi_dir is None:  # an engine without a quest folder (a stand-in) keeps no record
            return None
        try:
            counts = self.__dict__.setdefault("_model_call_counts", {})
            counts[node] = counts.get(node, 0) + 1
            row = _attempts.model_call_row(
                # A call of a step paused and run again after a resume is numbered on from the record's own lines.
                node=node, attempt=max(counts[node], _attempts.next_attempt(fi_dir, node)), served=served,
                requested_model=requested_model or self._model_for_node(node) or self.config.provider.model or None,
                reports_model=self._reports_for(node), messages=messages, response=response,
                outcome=outcome, usage=usage if isinstance(usage, dict) else None,
            )
            _attempts.append_model_call(fi_dir, getattr(self, "quest_id", ""), row)
            # Which line of the record the latest call of a step is, for a record built on it (the search queries).
            self.__dict__.setdefault("_last_call_id", {})[node] = row["call_id"]
            self.__dict__.setdefault("_last_call_outcome", {})[node] = outcome
            return row["call_id"]
        except Exception as e:  # noqa: BLE001 -- a record never touches the quest
            _attempts.count_lost(fi_dir, _attempts.MODEL_CALLS_LOST)
            self._log.debug("[attempts] model call not recorded: %r", e)
            return None

    def _chat_provenance(self, node: str) -> dict[str, Any]:
        """``provider``/``model`` (and ``vendor`` when the connection named one)/``prompt_hash``/``response_hash`` of the most recent ``_chat(node=...)`` call for this
        key, or ``{}`` when none was made under it (a rule-decided shortcut, an ensemble path) — ``_audit_claims`` merges
        this into a ``model_claim`` event so the trace says which model actually produced the claim, not just that some
        model did."""
        return dict(self._last_chat.get(node, {}))

    def _make_fallback_factory(self, name: str, base: Any = None):
        """Build an async factory that lazily resolves+constructs an
        ``LLMClient`` for fallback provider ``name`` (used by
        :class:`FallbackLLMClient`). Nothing is resolved and no proxy spawned
        until the primary provider actually fails and the chain reaches this
        rung. The derived config keeps the primary's timeouts (pure seconds,
        provider-agnostic) but resets model/base_url/api_key and drops
        node_model_fallbacks — those name provider-specific models that would
        be wrong for a different provider."""
        async def _factory() -> LLMClient:
            derived = (base or self.config.provider).model_copy(update={
                "name": name,
                "model": None,
                "base_url": None,
                "api_key_env": None,
                "node_model_fallbacks": {},
                "fallback": [],
                # Sampling settings and output limits belong to the primary's model, not to a fallback's.
                "fixed_temperature": None,
                "extra_body": {},
                "node_max_tokens": {},
            })
            ep = await resolve_endpoint_async(derived, self.supervisor)
            self._log.info(
                "[fallback] resolved %s -> %s (%s)", name, ep.base_url, ep.model,
            )
            return self._new_llm_client(ep)
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
            try:
                # A member that cannot be used is left out by design (the merger sees the others), so its usage limit
                # alone does not stop the step; when no member answers, it does (below).
                text, served = await self._recorded_call(
                    node, messages,
                    lambda: self._client.chat(messages, temperature=temperature, model=model, node=node),
                    requested_model=model, tolerated=True,
                )
            except Exception as exc:
                if _is_fatal_provider_error(exc):
                    unusable.append(f"{node}: {' '.join(str(exc).split())[:400]}")
                    unusable_errors.append(exc)
                    if _provider_is_auth_error(exc) and not unusable_signin:
                        unusable_signin.append(self._sign_in_message(exc, model))
                raise
            # This call's own model and counts (LAST_CALL is the task's own record); the client's shared attributes
            # only when the transport set nothing.
            self._log_chat_cost(
                node=node, model=served.get("model") or getattr(self._client, "last_model", None) or model or "",
                usage=served.get("usage") if served.get("usage") is not None
                else getattr(self._client, "last_usage", None),
            )
            return text

        messages = [{"role": "user", "content": prompt}]
        unusable: list[str] = []
        unusable_signin: list[str] = []
        unusable_errors: list[BaseException] = []
        try:
            return await self._merged_ensemble(prompt, messages, node, ensemble_cfg, _chat_fn)
        except Exception:
            if unusable:
                self.__dict__["_provider_unavailable"] = unusable[0]  # no member answered and one met a usage limit
                self.__dict__["_provider_signin"] = unusable_signin[0] if unusable_signin else ""
                # the member's own error is the stop's cause, so its failure card names the limit and its time
                self.__dict__["_provider_limit"] = unusable_errors[0]
            raise

    async def _merged_ensemble(self, prompt: str, messages: list[dict[str, str]], node: str,
                               ensemble_cfg: Any, _chat_fn: Any) -> Any:
        """:meth:`_ensemble_chat`'s fan-out, merge and breadcrumb rows."""
        from core.ensemble import cost_jsonl_entries, fanout_chat, merge_synthesize, merge_tournament, merge_vote

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
        client, _reports = await self._client_for(node)
        response, _served = await self._recorded_call(
            node or "", messages,
            lambda: client.chat(messages, temperature=temperature, model=self._model_for_node(node),
                                node=node or ""),
        )
        self._log_chat_cost(node=node or "", messages=messages, response=response, model=_served.get("model") or None,
                            usage=_served.get("usage") if isinstance(_served.get("usage"), dict) else None)
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

    def _asked_clarify_questions(self, topic: str) -> dict[str, Any] | None:
        """The questions already put to the person for ``topic``, when the clarify node paused on them; else None."""
        try:
            data = json.loads((self.fi_dir / self._CLARIFY_ASKED).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("topic_sha256") != hashlib.sha256(str(topic or "").encode("utf-8")).hexdigest():
            return None
        questions = data.get("questions")
        return questions if isinstance(questions, dict) and questions else None

    def _keep_asked_clarify_questions(self, topic: str, questions: dict[str, Any]) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.fi_dir / (self._CLARIFY_ASKED + ".tmp")
            tmp.write_text(json.dumps({"topic_sha256": hashlib.sha256(str(topic or "").encode("utf-8")).hexdigest(), "questions": questions},
                                      indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp, self.fi_dir / self._CLARIFY_ASKED)
        except (OSError, TypeError, ValueError) as e:
            # Not kept: the resume asks the model again, as before.
            self._log.debug("[clarify] could not keep the asked questions: %r", e)

    async def _await_with_heartbeat(
        self, coro: Awaitable[Any], *, label: str, interval_s: float = 30.0, progress: Any = None,
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
                    where = ""
                    try:  # where the study is, in plain words (core/run_estimate.py); never stops the beat
                        where = str(progress() or "") if callable(progress) else ""
                    except Exception:  # noqa: BLE001
                        where = ""
                    if where:
                        from .run_estimate import plain_duration

                        self._log.info("[execute] %s — still running, %s elapsed: %s", label,
                                       plain_duration(time.monotonic() - start), where)
                        continue
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

    def _model_calls_since(self, began: float) -> int:
        """How many model calls ``.fi/model_calls.jsonl`` records since ``began`` (a Unix time): what a retry repeats."""
        n = 0
        fi_dir = getattr(self, "fi_dir", None)
        if fi_dir is None:
            return 0
        try:
            with (fi_dir / "model_calls.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        n += float(json.loads(line).get("at") or 0.0) >= began
                    except (ValueError, AttributeError, TypeError):
                        continue
        except OSError:
            return 0
        return n

    async def _try_step_again(self, graph: Any, run_config: dict[str, Any], exc: Exception) -> bool:
        """Whether to run the step that just failed again by itself (core/crash_kind.py): only for a passing problem (a
        timeout, a dropped connection, a 429 or 5xx the provider's own retries did not outlast), at most
        ``len(RETRY_WAITS_S)`` times for the same step, and only when the step had not been running long (a long
        experiment is never repeated for a problem that may not have passed). Waits first, and says so on the screen.
        Never raises: a doubt means no retry, and the error goes on as it was."""
        from datetime import datetime

        try:
            snap = await graph.aget_state(run_config)
            step = ", ".join(getattr(snap, "next", None) or ()) or "(start)"
            # Counted per entry into the step: the checkpoint it starts from. A later pass through the same step (after
            # a review sent the quest back, say) starts from another checkpoint and has its own retries; the count kept
            # is always this entry's, so the failure card never names an earlier pass's retries.
            entry = str(((getattr(snap, "config", None) or {}).get("configurable") or {}).get("checkpoint_id") or "")
            seen_entry, done = self._step_retries.get(step, ("", 0))
            if seen_entry != entry:
                done = 0
                self._step_retries[step] = (entry, 0)
            # The same provider and model the failure card is sorted with, so the retry and the card never disagree.
            config = getattr(self, "config", None)
            prov = getattr(config, "provider", None)
            served_name, served_model, routed_key = self._provider_for_step(step)
            kind = _crash_kind.classify(exc, node=step, provider=served_name or str(getattr(prov, "name", "") or ""),
                                        model=served_model or str(getattr(prov, "model", "") or ""),
                                        routed=routed_key).kind
            if kind != "transient":
                return False
            if done >= len(_crash_kind.RETRY_WAITS_S):
                return False
            jobs = bool(getattr(getattr(config, "execution", None), "background_jobs", False))
            if jobs and any(n in _OUTSIDE_WORK_STEPS for n in (getattr(snap, "next", None) or ())):
                # A step that hands work to a cluster or a background job is never run again by itself: a second pass
                # could submit the same jobs twice.
                self._log.warning("[run] a passing problem stopped the step %s, which hands work to background jobs; "
                                  "not running it again by itself", step)
                return False
            # How long the step had been working: since it last started in this process (never counting the time
            # between an earlier run and a resume, nor a wait before a retry), or since the checkpoint it started from
            # when that is later (a step after others in the same call).
            began = float(getattr(self, "_invoke_started_at", 0.0) or 0.0)
            created = getattr(snap, "created_at", None)
            if created:
                began = max(began, datetime.fromisoformat(str(created)).timestamp())
            ran_s = time.time() - began if began else 0.0
            if ran_s > _crash_kind.RETRY_ONLY_UNDER_S:
                self._log.warning("[run] a passing problem stopped the step %s after %d s of work; not running it "
                                  "again by itself", step, int(ran_s))
                return False
            wait_s = _crash_kind.RETRY_WAITS_S[done]
            self._step_retries[step] = (entry, done + 1)
            self._log.warning("[run] a passing problem stopped the step %s (%s); trying it again in %d s (%d of %d); "
                              "the step's %d model call(s) since it started are made again",
                              step, _crash_kind._one_line(exc, 160), int(wait_s), done + 1,
                              len(_crash_kind.RETRY_WAITS_S), self._model_calls_since(began))
            minutes = max(1, round(wait_s / 60))
            self._progress(f"The model service or the network had a passing problem; FI tries the step again in "
                           f"{minutes} minute{'s' if minutes != 1 else ''}.")
            await asyncio.sleep(wait_s)
            return True
        except Exception as doubt:  # noqa: BLE001 -- never in the way of the real error
            self._log.warning("[run] could not decide whether to run the step again: %r", doubt)
            return False

    def _clear_stale_quest_failed_diagnostic(self) -> None:
        """Remove a stale ``quest_failed.md`` from a PRIOR failed run.

        Called from the two non-failed exit paths (clean success and
        data-pause-exit). Same idempotent-cleanup pattern the paper
        generator uses for ``paper_pdf_skipped.md`` on a successful
        PDF compile. Failures to unlink are logged but never raise —
        a stale file is annoying but not fatal.
        """
        _crash_kind.clear(self.fi_dir)
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
        # The figures' paper copies (figures/paper/, core/plot_style.py) go with them.
        from .plot_style import PAPER_DIRNAME

        shutil.rmtree(fig_dir / PAPER_DIRNAME, ignore_errors=True)
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
        # Which pause this run goes on from, read before its marker goes: a node that needs to know what the person
        # answered by resuming (the papers pause) asks the engine, not the file.
        try:
            self._resumed_from_pause = json.loads((self.fi_dir / "pause.json").read_text(encoding="utf-8")).get("kind")
        except (OSError, ValueError, AttributeError):
            self._resumed_from_pause = None
        for p in (
            self.quest_root / "NEXT_STEP.md",
            self.fi_dir / "pause.json",
            self.fi_dir / _todo.TODO_NAME,
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
        # The provider and model that served the failing step: the step's own when it runs on one
        # (``provider.node_providers``), else what the error's own note says, else the quest's main one.
        provider_name, model_raw, routed_key = self._provider_for_step(failing_node)
        served = _crash_kind.served_by(exc)
        step_name = failing_node
        if served is not None:
            provider_name, model_raw = served[0], served[1]
            if served[2]:  # the call's own step, which is the one to name when no checkpoint says which step failed
                routed_key = routed_key or self._provider_for_step(served[2])[2]
                step_name = served[2] if failing_node.startswith("(") else failing_node
        provider_model = model_raw or "(provider default)"
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
        # ``format_exception_only`` (not a bare f"{type}: {exc}") so a note an LLM call attached
        # via ``Exception.add_note`` — provider/transport/model/node context, and, for an
        # HTTP auth failure, which key looked wrong — reaches this file. Without it, a note the
        # codebase already writes to every provider-call exception was silently dropped here and
        # only ever reached a full ``run.log`` traceback (a 401 read as a bare, unexplained
        # ``HTTPStatusError``, with the note that would have named the likely key sitting one
        # frame away, unread).
        what_broke = "".join(traceback.format_exception_only(type(exc), exc)).rstrip("\n")
        # The failure as the person sees it first (core/crash_kind.py): what happened and the one thing to do. The
        # exception, the provider and the log tail are details for a bug report, below.
        failure = _crash_kind.classify(
            exc, node=step_name, provider=provider_name, model=model_raw,
            routed=routed_key, retries=int((getattr(self, "_step_retries", None) or {}).get(failing_node, ("", 0))[1]))
        try:
            _crash_kind.write(self.fi_dir, failure)
        except OSError as e:
            self._log.warning("[run] could not write %s: %r", self.fi_dir / _crash_kind.FAILURE_FILE, e)
        resume_cmd = (f"python launch.py --config {(self.quest_root / 'config.yaml').as_posix()} "
                      f"--resume {self.quest_id}")
        body = (
            f"# {_crash_kind.TITLES.get(failure.kind, 'The quest stopped')}\n"
            f"\n"
            f"**What happened:** {failure.say}\n"
            f"\n"
            f"**What to do:** {failure.do}\n"
            f"\n"
            f"To continue (it picks up at the step that stopped; the finished steps are not done again):\n"
            f"\n"
            f"```bash\n"
            f"{resume_cmd}\n"
            f"```\n"
            f"\n"
            f"(or `fi --resume {self.quest_id}`, or Resume on the web quest page or in VS Code.)\n"
            f"\n"
            f"## Details (for FI's maintainers)\n"
            f"\n"
            f"**Quest ID:** `{self.quest_id}`\n"
            f"**Topic:** {topic_one_line}\n"
            f"**Failing node:** `{failing_node}`\n"
            f"**Provider:** `{provider_name}` / model `{provider_model}`"
            f" / extras: {provider_extra_str}\n"
            f"**Kind of failure:** {failure.kind}"
            + (f" (tried again {failure.retries} time(s) first)" if failure.retries else "") + "\n"
            f"\n"
            f"## What broke\n"
            f"\n"
            f"```\n"
            f"{what_broke}\n"
            f"```\n"
            f"\n"
            f"## Last ~80 lines of `.fi/run.log`\n"
            f"\n"
            f"```\n"
            f"{log_tail}\n"
            f"```\n"
            f"\n"
            f"This file is deleted on the next successful run of this quest.\n"
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
        messages: Any = None,
        response: Any = None,
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
        # The engine writes its own line in the call record (``_record_model_call``), with more than a cost row knows.
        append_cost_row(self.fi_dir, node=node, model=model, usage=usage, messages=messages, response=response,
                        ledger=False)

    async def _attempt_context(self, state: QuestState, *, kind: str, partial: bool = False) -> dict[str, Any]:
        """The conditions an attempt ran under (core/attempt_records.py; ``kind`` is one of its ``CONTEXT_KINDS``); an
        empty dict if they cannot be read. Hashing
        the quest's files and asking git for FI's commit run in a worker thread, off the event loop."""
        try:
            models_used = {node: dict(v) for node, v in dict(getattr(self, "_last_chat", {}) or {}).items()}
            cache = self.__dict__.setdefault("_attempt_digests", {})
            context = await asyncio.to_thread(
                _attempts.context_fingerprint,
                self.config, self.quest_root, dict(state), prompts=self._prompts,
                fi_repo=Path(__file__).resolve().parent.parent, models_used=models_used,
                kind=kind, cache=cache, partial=partial,
                model_call_counts=dict(self.__dict__.get("_model_call_counts") or {}) or None,
            )
            if not partial:
                self._last_attempt_context = context
            return context
        except Exception as e:  # noqa: BLE001 -- a record never touches the quest
            self._log.debug("[attempts] context not read: %r", e)
            return {}

    async def _shadow(self, decision: str, state: QuestState | dict[str, Any], *, taken: str) -> None:
        """Shadow mode (core/attempt_memory.py, ``engine.attempt_memory: shadow``): what past failed attempts would
        recommend at this decision, written to ``.fi/shadow_recommendations.jsonl`` and nothing else. It never changes
        the state, the route, a prompt or the trace. It is worked out on this engine's own worker thread within
        ``attempt_memory.DEADLINE_S``; a result that comes later is discarded, never written, and counted, as is any
        failure. A decision made while the previous one is still being worked out is skipped (counted)."""
        if getattr(self.config.engine, "attempt_memory", "shadow") != "shadow":
            return
        try:
            closed = self.__dict__.setdefault("_shadow_closed", threading.Event())
            if closed.is_set():
                return
            busy = self.__dict__.get("_shadow_future")
            if busy is not None and not busy.done():
                _attempts.count_lost(self.fi_dir, _memory.SHADOW_LOST)
                return
            record_id = uuid.uuid4().hex
            decided_at = time.time()
            # Stamped now, before the run or stop it is about can be written: lineage, not wall clock.
            _memory.stamp(self.fi_dir, decision, record_id)
            snapshot = dict(state)
            models_used = {node: dict(v) for node, v in dict(getattr(self, "_last_chat", {}) or {}).items()}
            asked = {"plan": self._model_for_node("plan"), "implement": self._model_for_node("implement")}.get(
                decision) or self.config.provider.model
            # The same "provider/model" form the records keep for the model that answered.
            requested = f"{self.config.provider.name}/{asked}" if asked else None
            signature = None
            exclude_ids: set[str] = set()
            exclude_lineage = None
            if decision == "repair":
                er = snapshot.get("exec_result") if isinstance(snapshot.get("exec_result"), dict) else {}
                signature = getattr(self, "_last_run_signature", None) or _memory.failure_signature(
                    returncode=er.get("returncode"), stderr=str(er.get("stderr_tail") or "")[-2000:],
                    timed_out=bool(er.get("timed_out")))
                if getattr(self, "_last_run_record_id", None):
                    exclude_ids.add(self._last_run_record_id)
                history = snapshot.get("design_history")
                exclude_lineage = (self.quest_id, (len(history) - 1) if history else None)
            cancel = threading.Event()
            lock = self.__dict__.setdefault("_shadow_lock", threading.Lock())
            index = _memory.index_for(self.quest_root.parent)
            cache = self.__dict__.setdefault("_shadow_digests", {})

            def work() -> None:
                context = _attempts.context_fingerprint(
                    self.config, self.quest_root, snapshot, kind="in_progress", prompts=self._prompts,
                    models_used=models_used, cache=cache,
                )
                cand = _memory.parts(context, decision, signature=signature, requested_model=requested)
                if cancel.is_set():
                    return
                rec = _memory.recommend(cand, index.attempts(also=self.fi_dir), decision=decision,
                                        exclude_ids=exclude_ids, exclude_lineage=exclude_lineage)
                with lock:
                    if cancel.is_set() or closed.is_set():
                        return  # too late: dropped, never written with a wrong time
                    _memory.record(
                        self.fi_dir, self.quest_id, record_id=record_id, decided_at=decided_at, decision=decision,
                        taken=taken, candidate=cand,
                        # FI's own source is deliberately not read here (no decision is compared on it).
                        context_missing=[m for m in context.get("missing") or [] if not m.startswith("FI's source")],
                        model_family=str(cand.get("model_plan") or cand.get("model_implement") or requested or "?"),
                        recommendation=rec, excluded=sorted(exclude_ids),
                    )

            loop = asyncio.get_running_loop()
            future: asyncio.Future[None] = loop.create_future()

            def settle(error: BaseException | None) -> None:
                if not future.done():
                    future.set_exception(error) if error is not None else future.set_result(None)

            def runner() -> None:
                # A daemon thread: a hash that never returns cannot keep the process from ending.
                error: BaseException | None = None
                try:
                    work()
                except BaseException as e:  # noqa: BLE001
                    error = e
                try:
                    loop.call_soon_threadsafe(settle, error)
                except RuntimeError:
                    pass  # the loop is gone: nothing is waiting

            self.__dict__["_shadow_future"] = future
            threading.Thread(target=runner, name="fi-shadow", daemon=True).start()
            try:
                await asyncio.wait_for(asyncio.shield(future), timeout=_memory.DEADLINE_S)
            except asyncio.TimeoutError:
                cancel.set()
                _attempts.count_lost(self.fi_dir, _memory.SHADOW_LOST)
                self._log.debug("[shadow] the recommendation at %s took too long; dropped", decision)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- a shadow recommendation never touches the quest
            try:
                _attempts.count_lost(self.fi_dir, _memory.SHADOW_LOST)
            except Exception:  # noqa: BLE001
                pass
            self._log.debug("[shadow] no recommendation recorded at %s: %r", decision, e)

    def _shadow_close(self) -> None:
        """No shadow recommendation is written after this (the quest is sealing or stopping); a worker still running
        is left to finish on its own and writes nothing."""
        closed = self.__dict__.setdefault("_shadow_closed", threading.Event())
        lock = self.__dict__.setdefault("_shadow_lock", threading.Lock())
        with lock:
            closed.set()

    def _record(self, name: str, build: Any) -> str | None:
        """Append to one of the quest's attempt records (core/attempt_records.py). ``build`` returns the record (or is
        it); building it and writing it are both best-effort, so a record never touches the quest. Records only: read
        by nothing that decides a route."""
        try:
            record = build() if callable(build) else build
            if not record:
                return None
            if (name == _attempts.ATTEMPTS and record.get("kind") in ("run", "stop", "quest")
                    and getattr(self.config.engine, "attempt_memory", "shadow") == "shadow"):
                # The shadow recommendations this record is about (core/attempt_memory.py): how they are scored.
                ids = _memory.take(self.fi_dir, str(record.get("kind")))
                if ids:
                    record = {**record, "parent_shadow_ids": ids}
            record_id = _attempts.append(self.fi_dir, name, {"quest_id": self.quest_id, **record})
            if not record_id:
                _attempts.count_lost(self.fi_dir)
                self._log.debug("[attempts] %s not written", name)
            return record_id
        except Exception as e:  # noqa: BLE001 -- a record never touches the quest
            _attempts.count_lost(self.fi_dir)
            self._log.debug("[attempts] %s not built: %r", name, e)
            return None

    def _answering_model(self) -> dict[str, Any]:
        """The provider and model that answered the quest's last model call (after a fallback, the fallback's): the
        last call's, not necessarily the one that did the work the record is about."""
        client = self._client
        return {"provider": getattr(client, "last_provider", None) or None,
                "model": getattr(client, "last_model", None) or None}

    def _record_stop(self, state: QuestState | dict[str, Any], context: dict[str, Any]) -> None:
        """A quest that stopped for a check it failed (:data:`core.attempt_records.STOP_OUTCOMES`) is an attempt with
        that outcome; a stop that is not a failed check (the plan waiting for you, papers asked for) is not recorded."""
        def build() -> dict[str, Any] | None:
            if not (self.fi_dir / "pause.json").is_file():
                return None
            pause = json.loads((self.fi_dir / "pause.json").read_text(encoding="utf-8"))
            kind = str((pause or {}).get("kind") or "")
            outcome = _attempts.STOP_OUTCOMES.get(kind)
            if outcome is None:
                # Waiting for you, not a failed check: what was recommended before it is not about the run after it.
                if getattr(self.config.engine, "attempt_memory", "shadow") == "shadow":
                    _memory.settle_without_record(self.fi_dir)
                return None
            return {"kind": "stop", "pause": kind, "outcome": outcome, "execution_status": "stopped",
                    "failure_signature": _memory.failure_signature(pause=kind),
                    "run_record_id": (getattr(self, "_last_run_record_id", None)
                                      or _attempts.last_id(self.fi_dir, _attempts.ATTEMPTS, ("run",))),
                    "answered_by": self._answering_model(), "context": context}
        self._record(_attempts.ATTEMPTS, build)

    def _step_is_set(self, step: str) -> bool:
        """Whether a model or a provider is set for ``step`` (``provider.node_models`` / ``provider.node_providers``)."""
        provider = self.config.provider
        return bool(node_provider_for(getattr(provider, "node_providers", None), step)
                    or model_for_node(provider.node_models, step))

    def _review_node(self, step: str) -> str:
        """The node a reviewer's calls are made under: its own step (``design_review``, ``code_review``) when a person set
        a model or provider for it, else the step it falls back to (core/review_pairs.py)."""
        return _pairs.node_for(step, self._step_is_set)

    def _model_for_node(self, node: str | None) -> str | None:
        """Resolve the effective model for a node via the shared
        ``core.provider.model_for_node`` lookup — None when the lookup
        misses so the transport falls through to the endpoint default.
        Accepts hierarchical keys like ``"review_panel.methodologist"``
        (review-panel personas)."""
        if not node:
            return None
        step = node_provider_for(getattr(self.config.provider, "node_providers", None), node)
        if step is not None:  # the step's own provider's model (a step in node_models too uses this one)
            return step[1].model
        return model_for_node(self.config.provider.node_models, node)

    def _one_model_panel(self) -> bool:
        """``engine.one_model_review`` is set and every reviewer on the panel is asked on the same model."""
        if not getattr(self.config.engine, "one_model_review", False):
            return False
        panel = list(self.config.engine.review_panel or [])
        return len({self._reviewer_model(p) for p in panel}) <= 1

    def _review_identity_unverified(self, state: QuestState) -> list[str]:
        """Under ``rigor_profile: research``, the reviewers set up on different models whose connection did not say
        which model actually answered (one that reports none): that they were different models is unproven, so the
        evidence level lists it. Empty when the panel knowingly runs on one model (its own gap) or every answer came
        with the model that gave it."""
        if self.config.rigor_profile != "research" or self._one_model_panel():
            return []
        ok = [r for r in (state.get("review_panel") or []) if isinstance(r, dict) and r.get("status", "ok") == "ok"]
        if len({str(r.get("requested_model")) for r in ok}) < 2:
            return []
        return sorted(str(r.get("persona") or "?") for r in ok if not r.get("actual_model_reported"))

    def _reviewer_model(self, persona: str) -> str:
        """The model a review-panel persona is asked on: its own ``provider.node_models`` entry, else the quest's."""
        return self._model_for_node(f"review_panel.{persona}") or self.config.provider.model or "(the provider's default)"

    def _review_models_stop(self) -> QuestArtifacts | None:
        """``rigor_profile: research``: the review panel needs at least one reviewer on a different model. Personas all
        on one model are one model's view in several roles; its errors are shared, not caught. Stops before anything
        runs and says how to give one persona another model (the person chooses which). ``None`` when there is one."""
        if self.config.rigor_profile != "research" or not self.config.engine.review_panel:
            return None
        if getattr(self.config.engine, "one_model_review", False):
            return None
        models = {p: self._reviewer_model(p) for p in self.config.engine.review_panel}
        if len(set(models.values())) > 1:
            return None
        # A panel that has already reviewed (a quest resumed at the human-review gate, say) is not asked again, unless
        # its models changed since (a model change is taken without a stop, so it would otherwise put every later
        # review on one model unasked).
        if (any(e.get("kind") == "node_completed" and e.get("node") == "review" for e in _audit_log.read(self.audit.path))
                and not _plan_settings.model_changes(self.fi_dir, self.config)):
            return None
        only = next(iter(models.values()))
        persona = "statistician" if "statistician" in models else next(iter(models))
        self._log.warning("[review] stopped: every review-panel persona is on %s; research needs one on another model", only)
        self._audit("pause_requested", pause="review_models", model=only, panel=list(models))
        headline = "the review panel needs one reviewer on a different model"
        approved = (self.fi_dir / _plan_settings.NAME).is_file()
        how = (
            f"Choose another model for one of them, then approve it with `python launch.py --update {self.quest_id}` "
            f"(web: the quest page's Update, VS Code: `@fi /update {self.quest_id}`): in \"Per-node model overrides\" "
            f"enter `review_panel.{persona}:<another model your provider offers>` (in VS Code choose \"Type node:model "
            f"pairs\" there)."
            if approved else
            f"Choose another model for one of them and add it to the quest's config.yaml, under its `provider:` section "
            f"(add one if there is none; never a second one), inside `node_models:` if that is already there:\n"
            f"  node_models:\n    review_panel.{persona}: <another model your provider offers>"
        )
        self._write_next_step(
            kind="review_models",
            interaction="answer",
            headline=headline,
            steps=[
                f"Nothing was run. Every reviewer on this research quest's panel ({', '.join(models)}) would be asked on "
                f"the same model ({only}): one model's view in several roles, whose mistakes the others share.",
                how,
                f"Then `python launch.py --resume {self.quest_id}`. (Without rigor_profile: research there is no such "
                "requirement.)",
                "If you have only one model: add `one_model_review: true` under the `engine:` section of the quest's "
                "config.yaml and resume. The quest then runs on it, and its result says the review was one model's view "
                "and is not marked publication-ready.",
            ],
        )
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / "pause.json").write_text(json.dumps({
                "kind": "review_models", "interaction": "answer", "quest_id": self.quest_id, "headline": headline,
                "next_step_file": "NEXT_STEP.md", "upload_targets": [],
            }, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
        return self._collect_artifacts({})

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

    async def _record_environment(self, stage: str = "at the start of the run") -> None:
        """The interpreter, platform and installed packages this quest's experiment actually runs on, and whether
        that environment is this quest's own or shared with every other quest on the machine.

        ``rigor_profile: research`` now forces isolation (``execution.shared_interpreter`` / ``system_site_packages``
        both false, or ``sandbox: docker``) — the re-audit's P1-1 fix. The default profile still leaves those as the
        earlier decision keeps them (a previous quest's `pip install` CAN affect this run); this method records
        which is true either way, so a result can be reproduced from the exact package list it ran on, and so the
        evidence level can say plainly when it wasn't isolated rather than being silent about it (the original
        re-audit's P1-2 finding). Best-effort throughout: a diagnostic that cannot be written must never stop a
        quest.
        """
        if stage == "at the start of the run":
            # A resumed quest (after the review pause, say) keeps the record of the environment its experiment ran
            # on: one made after the experiment's packages were installed. A run that installs them again records
            # it again then.
            earlier = _read_json_or_none_path(self.quest_root / "needs" / "ENVIRONMENT.json")
            if isinstance(earlier, dict) and str(earlier.get("recorded") or "").startswith("after installing"):
                return
        isolated = self.config.execution.sandbox == "docker" or not (
            self.config.execution.shared_interpreter or self.config.execution.system_site_packages
        )
        record: dict[str, Any] = {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "sandbox": self.config.execution.sandbox,
            "shared_interpreter": self.config.execution.shared_interpreter,
            "system_site_packages": self.config.execution.system_site_packages,
            "isolated": isolated,
            "recorded": stage,
        }
        try:
            py = self.executor.python_path(self.quest_root)
            # The whole tree (core/proc_tree.py): on Windows a venv's python.exe is a launcher that starts the real
            # interpreter, and killing the launcher alone left a hung `pip freeze` running for the machine's uptime.
            tree = await AsyncProcessTree.start(
                str(py), "-m", "pip", "freeze",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=scrub_secrets(),  # the quest's packages load here (.pth files): no FI keys
            )
            proc = tree.proc
            aborted = False
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=_PIP_FREEZE_TIMEOUT_S)
            except asyncio.TimeoutError:
                # Stop it with everything it started before re-raising to the outer handler below, which records
                # the timeout as ``packages_error``.
                await tree.aclose(aborted=True)
                raise
            except asyncio.CancelledError:
                aborted = True
                raise
            finally:
                await tree.aclose(aborted=aborted)
            if proc.returncode != 0:
                # The returncode is not optional to check: pip can exit non-zero with EMPTY stdout (a real case,
                # found on this codebase's own dev machine — a corrupted dist-info makes `pip freeze` crash there),
                # which would otherwise look exactly like "confirmed zero packages" instead of "unknown."
                # Another real case: an editable install whose folder was deleted makes `pip freeze` crash in its
                # version-control probe (NotADirectoryError). The installed distributions are then listed from the
                # interpreter itself, which needs no probe; only when that fails too is the list unknown.
                why = _last_error_line(err.decode("utf-8", "replace")) or f"exit code {proc.returncode}"
                listed = await self._installed_distributions(py)
                if listed is not None:
                    record["packages"] = listed
                    record["packages_source"] = f"the interpreter's installed distributions (pip freeze failed: {why})"
                else:
                    record["packages_error"] = f"pip freeze failed ({why}), and the installed distributions could not be listed"
            else:
                record["packages"] = sorted(out.decode("utf-8", "replace").splitlines())
        except Exception as e:  # noqa: BLE001 -- a diagnostic must never stall a quest
            record["packages_error"] = repr(e)
        try:
            path = self.quest_root / "needs" / "ENVIRONMENT.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            self._audit_check(
                "environment", path, status="isolated" if isolated else "shared",
                summary="" if isolated else "shared_interpreter/system_site_packages: another quest's packages could affect this run",
            )
        except OSError:
            pass

    def _job_block(self, state: QuestState | None = None) -> str:
        """The background-job contract for the design and code-writing prompts,
        or nothing when ``execution.background_jobs`` is off. A quest whose trials FI runs as a job array gets that
        contract with the two-script one instead (:meth:`_split_block`)."""
        if not self.config.execution.background_jobs:
            return ""
        if state is not None and self._split_on(state):
            return ""
        # Explore, then confirm: the confirm run's job must draw new random numbers, so the driver passes the seed on.
        return _JOB_PROTOCOL + (_JOB_SEED_PROTOCOL if _phased.enabled(self.config) else _JOB_NO_SEED_PROTOCOL)

    def _save_run_data(self, since: float, split: bool, root_files: bool = True,
                       wrote: set[str] | frozenset[str] | None = None) -> list[str]:
        """Copy what a successful run produced into ``data/results/``, so the numbers behind a paper are in ``data/`` too.

        The scripts run from the quest folder, so a table a script writes lands beside ``plan.md``; a two-script quest's
        raw files are in ``raw/``. Both are copied (the originals stay where the scripts and FI's records expect them),
        each file and the raw folder only while small enough not to double a large quest's disk use. The raw copy is
        replaced, not merged, so it never keeps a file an earlier run wrote. ``root_files=False`` refreshes only the raw
        copy (after the replicate seeds ran, which add raw files but overwrite the root tables). Never raises.

        A file is this run's when FI's own record of what the run wrote names it (``wrote``, paths relative to the quest:
        core/outside_run.py tells a file written again with the content it had, which is not copied back and so keeps its
        old modification time), or when its modification time is not older than the run's start (a file written in the
        quest folder itself, which has no such record). A file neither is an earlier run's and is removed."""
        wrote = wrote or frozenset()
        saved: list[str] = []
        dest = self.quest_root / "data" / "results"
        try:
            if root_files and dest.is_dir():
                # Tables an earlier iteration left are not this run's; what a script wrote here itself is newer.
                for old in dest.iterdir():
                    if (old.is_file() and f"data/results/{old.name}" not in wrote
                            and old.stat().st_mtime < since - 2):
                        old.unlink(missing_ok=True)
            for path in sorted(self.quest_root.iterdir() if root_files else []):
                if any(fnmatch.fnmatch(path.name.lower(), pat) for pat in self._RUN_DATA_SKIP):
                    if path.suffix.lower() in self._RUN_DATA_SUFFIXES:
                        self._log.info("[execute] %s looks like a secret or setup file; not copied to data/results/", path.name)
                    continue
                if (
                    path.is_file() and path.suffix.lower() in self._RUN_DATA_SUFFIXES
                    and (path.name in wrote or path.stat().st_mtime >= since - 2)
                    and path.stat().st_size <= self._RUN_DATA_MAX_BYTES
                ):
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest / path.name)
                    saved.append(path.name)
            if split:
                raw = _split_run.raw_root_of(self.quest_root, self.config.execution.raw_dir)
                target = dest / "raw"
                if raw.is_dir() and not raw.resolve().is_relative_to(dest.resolve()):
                    total = 0
                    for p in raw.rglob("*"):
                        if p.is_file():
                            total += p.stat().st_size
                            if total > self._RUN_DATA_MAX_BYTES:
                                break
                    if total <= self._RUN_DATA_MAX_BYTES:
                        shutil.rmtree(target, ignore_errors=True)
                        shutil.copytree(raw, target)
                        saved.append("raw/")
                    else:
                        shutil.rmtree(target, ignore_errors=True)  # an earlier, smaller copy would be a partial one
                        self._log.info("[execute] the raw files in %s are large; they stay there and are not copied to data/results/", raw)
        except OSError as exc:
            self._log.warning("[execute] could not copy the run's data into data/results/: %s", exc)
        if saved:
            self._log.info("[execute] the run's data is kept in %s: %s", dest, ", ".join(saved))
        return saved

    def _split_on(self, state: QuestState) -> bool:
        """Whether this quest keeps its simulation and its analysis in two scripts (``execution.split_analysis``).

        ``true`` and ``false`` decide for every quest. ``auto`` is on for every quest that runs a simulation of its own,
        stochastic or deterministic: the simulation is then a function FI calls itself (``run_trial``, or ``run_cell``
        for a deterministic study), which is what lets FI run an oracle's case itself and so reach
        ``independently_validated``. It is off for a study with no experiment (the no-simulation path, a survey) and a
        run that only analyses data. ``split_analysis: false`` is the way back to one script, and gives that up."""
        mode = self.config.execution.split_analysis
        if mode is False and self._split_overridden(state):
            return True  # a search needs the run and the analysis apart (see _note_split_overridden)
        if mode != "auto":
            return bool(mode)
        # A background job (a cluster) runs FI's trials as a job array (core/trial_runner.py): two scripts and submit.py.
        return self._runs_code(state)

    def _split_block(self, state: QuestState, bodies: list[str] | None = None) -> str:
        """The two-script contract for the code-writing prompts, or nothing when
        ``execution.split_analysis`` is off. ``bodies``: FI writes the analysis (:meth:`_analysis_bodies`), so the request
        asks for the functions of derive.py, never experiment.py."""
        if not self._split_on(state):
            return ""
        search = _optim.has_block(state.get("design")) and not self.config.execution.background_jobs
        inside = _in_run.of_study(self._protocol_block(state))
        block = self._frame_protocols(state, search, inside, bodies) or (
                 _SPLIT_PROTOCOL + (_CLUSTER_PROTOCOL if self.config.execution.background_jobs else "")
                 + (_SEARCH_PROTOCOL if search else "")
                 + (_IN_EACH_RUN_PROTOCOL.replace("$factors", ", ".join(f"`{k}`" for k in inside))
                    .replace("$key_rule", "Key a mapping " + _in_run.key_forms(_in_run.level_cells(inside)) + ".") if inside else ""))
        layout = self._code_layout(state)
        # An extension of code kept as two scripts is not asked to restructure it into a package.
        extending_two_scripts = bool(state.get("refine_extend")) and not self._package_in_use(state)
        if layout and layout["shape"] == _code_layout.PACKAGE and not extending_two_scripts:
            package = layout["package"]
            if _function_steps.is_ready(self.quest_root, package):
                # FI wrote and checked the package function by function: only the two scripts are asked for.
                block += _code_layout.prefilled_block(
                    package, (self.quest_root / "code" / package / _code_layout.MODEL_NAME).read_text(encoding="utf-8"),
                    steps=self._written_steps(state), frame=self._frame_variant(state), analysis=bodies is not None)
                block += self._function_steps_partial_note()
            else:
                block += _code_layout.prompt_block(package, _oracle.generating_equations(self._protocol_block(state)),
                                                   frame=self._frame_variant(state), analysis=bodies is not None)
                block += self._function_steps_note(package)
        return block

    def _written_steps(self, state: QuestState) -> list[tuple[str, str, str]]:
        """``(function, step id, step name)`` for each function of the package that FI wrote for a step of the plan's method
        (``.fi/function_steps.json`` and the plan's step cards): what the simulation is told to call and not write again."""
        try:
            record = _function_steps.load(self.quest_root)
            names = {c["id"]: c.get("name") or "" for c in _plan_steps.parse(self._protocol_block(state))[0]}
            return [(str(fn), str(e.get("equation")), names.get(str(e.get("equation")), ""))
                    for fn, e in (record.get("functions") or {}).items()
                    if isinstance(e, dict) and e.get("kind") == _function_steps.STEP]
        except Exception:  # noqa: BLE001 -- a help to the prompt, never a stop
            return []

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
        # Explore, then confirm: the job's tasks read inputs/data/ when they start, so it keeps this stage's part; and a
        # resume that checks on the confirm run's own job is that run going on, not a second run on the confirm data.
        self._phased_job_pending = True
        if _phased.enabled(self.config):
            _phased.note_job_pending(self.quest_root)
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
            {**scrub_secrets(), ENV_VAR: str(examples_dir(self.quest_root))}
            if list_inputs(self.quest_root) else None
        )
        tasks = self.quest_root / _trial_runner.CLUSTER_DIR / _trial_runner.TASKS_NAME
        submit = self.quest_root / "code" / _trial_runner.SUBMIT_NAME
        if tasks.is_file() and submit.is_file():
            # FI's trials run as a job array: submit.py says whether it is done; the resume then collects the results.
            code_path = submit
            env = {**(env or scrub_secrets()),
                   _trial_runner.TASKS_ENV: (_trial_runner.CLUSTER_DIR / _trial_runner.TASKS_NAME).as_posix()}
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
        if result.returncode == 0 and result_json is not None and (job is None or job["status"] == job_watch.DONE):
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
        # A quest that ended in an honest stop (needs/STUCK.json, written by this run; a rerun or a refine moves it aside)
        # has no paper: one an earlier round left on disk is not this quest's result, so no PDF, slides or poster are
        # made from it. The record is read, not the checkpoint, so a rerun that went on is not held back by it.
        stuck = (self.quest_root / "needs" / "STUCK.json").is_file()
        return QuestArtifacts(
            quest_id=self.quest_id,
            quest_root=self.quest_root,
            paper_md=paper_md if paper_md.exists() and not stuck else None,
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
        if not _review_was_real(state.get("review")):
            self._log.info("[skills] usage not recorded: the accept was not a reviewer's (review status %s)",
                           (state.get("review") or {}).get("status"))
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

    def _write_back_knowledge(self, artifacts: QuestArtifacts, state: QuestState,
                              evidence_status: str | None = None) -> None:
        unreachable = getattr(self.knowledge, "unavailable_reason", None)
        if unreachable and self.config.knowledge.write_back_quests:
            # Asked for but not there: what the knowledge base holds cannot be seen, so its note is left as it is.
            if self._knowledge_note() is not None:
                self._log.warning("[write-back] %s, so this quest's knowledge-base note (.fi/knowledge_problem.json) "
                                  "was not checked again", unreachable)
            return
        if not self.knowledge.enabled or not self.config.knowledge.write_back_quests:
            # This quest no longer writes to the knowledge base: a note from an earlier write-back is not its to act on.
            self._settle_knowledge_note(None)
            return
        if artifacts.paper_md is None:
            self._settle_skipped_knowledge_note()
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
            self._settle_skipped_knowledge_note()
            return
        # An accept no reviewer gave is not research the review signed off on.
        if self.config.knowledge.write_back_only_on_accept and not _review_was_real(review):
            self._log.info("[write-back] skipped: the accept was not a reviewer's (review status %s)", review.get("status"))
            self._settle_skipped_knowledge_note()
            return

        # What the result may stand for in the long-term store: accepted evidence only when the quest was a study
        # (research or decision), a reviewer accepted it and its evidence level is publication_ready. Anything else
        # (an exploration, a result with evidence gaps, a written-back result nobody accepted) is kept as a preliminary
        # observation: a later quest sees it as a reminder of what was tried, never as evidence to cite.
        # ``evidence_status`` is this run's evidence record (Engine.run); no record means unknown, which is preliminary.
        evidence_status = str(evidence_status or "")
        result_use = getattr(self.config, "effective_result_use", "") or "explore"
        accepted = verdict == "accept" and _review_was_real(review)
        standing = ("accepted" if accepted and result_use in ("research", "decision")
                    and evidence_status == "publication_ready" else "preliminary")

        design = state.get("design") or {}
        # What memory holds as the study's claims is FI's own: the sentences it wrote from its record of each result it
        # looked up (the cross-check's entries), and the paper's own abstract. The analysis's text (its summary and key
        # findings, where FI's notes were restated as findings) is never stored.
        record_claims = [str(e["finding"]) for e in (state.get("cross_check") or [])
                         if isinstance(e, dict) and e.get("status") in ("checked", "no_literature") and e.get("finding")]
        try:
            summary = _paper_sections.abstract_of(artifacts.paper_md.read_text(encoding="utf-8"))
        except OSError:
            summary = ""

        # Distill the literature the agent actually saw into a curated
        # `external_refs` list. The Knowledge layer writes a spine doc
        # per ref so Axon becomes title-/topic-searchable after accept.
        external_refs: list[dict[str, Any]] = []
        for d in (state.get("literature") or []):
            m = d.get("metadata") or {}
            if not (m.get("title") or m.get("doi") or m.get("arxiv_id") or m.get("pmid")):
                continue
            # FI's own memory (a preliminary result, an earlier quest's records) is not an external paper; a paper the
            # person supplied (fi_local_paper) or one an earlier quest recorded (fi_external_ref_spine) is.
            if _is_preliminary_memory(m) or m.get("kind") in _FI_MEMORY_KINDS:
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
        chosen = state.get("chosen_idea")
        chosen_title = str(chosen.get("title") or "") if isinstance(chosen, dict) else ""
        ideas_list = state.get("ideas") if isinstance(state.get("ideas"), list) else []
        meta: dict[str, Any] = {
            "title": state.get("title", ""),
            "topic": state.get("topic", "")[:1000],
            "keywords": keywords,
            "verdict": verdict,
            "score": review.get("score"),
            "iteration": state.get("iteration", 0),
            "hypothesis": design.get("hypothesis", ""),
            "method_summary": design.get("method", ""),
            "key_findings": record_claims[:20],
            "claims_from": "record",
            # The idea this study pursued and the ones it chose it over, with how: what a later quest reads to know
            # the direction was one of several (the full record, pairwise verdicts included, is .fi/branch_ledger.jsonl).
            "chosen_idea": chosen_title,
            "alternative_ideas": [str(i.get("title") or "") for i in ideas_list
                                  if isinstance(i, dict) and i.get("title") and i.get("title") != chosen_title][:10],
            "idea_selection": (str(state.get("idea_selection") or "") or ("model" if chosen_title else "none")),
            "result_json": state.get("result_json") or {},
            "figures": list(state.get("figures", []) or []),
            "provider": self.config.provider.name,
            "model": self.config.provider.model or "(cli-default)",
            "paper_md_relpath": paper_md_relpath,
            "external_refs": external_refs,
            "standing": standing,
            "result_use": result_use,
            "evidence_status": evidence_status or "unknown",
            "review_status": str(review.get("status") or "ok"),
        }
        ok = self.knowledge.add_quest_artifacts(
            quest_id=self.quest_id,
            paper_md_path=artifacts.paper_md,
            summary=summary,
            metadata=meta,
        )
        self._log.info(
            "[write-back] axon ingest=%s (verdict=%s, score=%s, kept as %s: result for %s, evidence %s)",
            ok, verdict, review.get("score"), standing, result_use, evidence_status or "unknown",
        )
        # What kept the knowledge base from holding exactly one current copy of this result: in run.log and on the
        # quest's card until a later write-back settles it.
        self._settle_knowledge_note(getattr(self.knowledge, "last_writeback_problem", None),
                                    kind=getattr(self.knowledge, "last_writeback_problem_kind", None) or "unwritten")

    def _settle_skipped_knowledge_note(self) -> None:
        """A finished quest that skips its write-back: its note says only what is still true. A note that the result
        is not written stays (only a write-back that succeeds settles it); one that both copies are held stays while
        they are, goes when they are shown not to be, and stays when that cannot be told."""
        try:
            check = getattr(self.knowledge, "standing_conflict", None)
            both = check(self.quest_id) if callable(check) else None
        except Exception:  # noqa: BLE001 -- a note about the knowledge base never touches the quest
            both = None
        if both is None:
            return
        note = self._knowledge_note()
        if note is not None and note.get("kind", "unwritten") != "both":
            return
        if both:
            self._settle_knowledge_note(getattr(self.knowledge, "BOTH_COPIES", None) or
                                        "the knowledge base holds both an accepted and a preliminary copy of this "
                                        "quest's result", kind="both")
        elif note is not None:
            self._settle_knowledge_note(None)

    def _knowledge_note(self) -> dict[str, Any] | None:
        if getattr(self, "fi_dir", None) is None:
            return None
        try:
            note = json.loads((self.fi_dir / "knowledge_problem.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return note if isinstance(note, dict) else None

    def _settle_knowledge_note(self, problem: str | None, *, kind: str = "unwritten") -> None:
        """Keep ``.fi/knowledge_problem.json`` (and the card) to what is true now: ``problem`` written (with its
        ``kind``, see Knowledge.last_writeback_problem_kind), or the note removed when there is none."""
        if problem:
            self._log.warning("[write-back] %s", problem)
        if getattr(self, "fi_dir", None) is None:
            return
        record = self.fi_dir / "knowledge_problem.json"
        try:
            if problem:
                record.write_text(json.dumps({"quest_id": self.quest_id, "problem": problem, "kind": kind},
                                             indent=2) + "\n", encoding="utf-8")
            elif not record.is_file():
                return
            else:
                record.unlink(missing_ok=True)
            _todo.write(self.quest_root, self.fi_dir, self.quest_id, None)
        except OSError as e:
            self._log.debug("[write-back] could not record the knowledge-base note: %r", e)
