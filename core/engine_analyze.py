"""Methods of :class:`core.engine.Engine` for the analysis, the cross-check and the evidence gate.

``Engine`` inherits :class:`AnalyzeStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, NON_SCIENTIFIC_PAPER_FORMATS, Path, QuestState, SCIENTIFIC_PAPER_FORMATS, _FRESH_SCRIPT, _ModelAnswerProblem,
    _RECORD_REWRITES, _STOPPED_BEFORE_THE_STUDY, _UNPLACED_FINDING, _aggregate_result_json_replicates,
    _analysis_frame, _code_findings, _code_project, _code_reach, _code_review, _compact_result_json_block,
    _cross_check_items, _elide_value_lists, _evidence_gate_rule, _figure_list_for_prompt, _finding_repair,
    _fit_request, _fit_scope, _flat_fact_sentence, _flat_moved, _flat_request, _format_clarify, _format_lit, _frozen,
    _function_steps, _gate_inputs, _hit_name, _is_degenerate_result, _is_preliminary_memory, _load_persona_prefix,
    _numeric, _one_line, _oracle, _pairs, _parse_json_lenient, _pick_up_user_dropped_datasets, _plan, _receipts,
    _record_claims, _replicate_assertions, _replicate_metric_kinds, _rerun_from, _run_manifest,
    _shown_reach, _split_figure_readings, _stop_power, _topic_req, _write_personas, asyncio, copy, derive_protocol,
    functools, json, scrub_secrets, shutil, tempfile, time,
)


class AnalyzeStageMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    async def _node_analyze(self, state: QuestState) -> QuestState:
        self._log.info("[analyze] interpreting results")
        await self._refresh_code_project(
            state, list(state.get("deps") or []),
            "changes since the code was last recorded (the project files FI keeps, or your own edits)",
            category="changed",
        )
        self._ask_about_edited_project_files()
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
        payload_notes: list[str] = []  # FI's notes on the aggregate (precision, coverage, omitted metrics): `fi_notes`
        if replicates and (len(replicates) > 1 or state.get("result_json_trials")):
            assertions = _replicate_assertions(state)
            agg = _aggregate_result_json_replicates(replicates, assertions=assertions, kinds=_replicate_metric_kinds(state))
            # Flattening a crossed design yields one entry per numeric leaf —
            # easily hundreds on a parameter sweep, each carrying seven stats.
            # Only the ones that actually MOVED between seeds tell the reader
            # anything, and the rest would crowd out the analysis they are
            # meant to inform (``_compact_result_json_block`` would otherwise
            # truncate at an arbitrary point). Entries that never varied are
            # dropped and counted; the per-seed JSON above still carries them.
            constant = {k for k, v in agg.items() if v.get("n", 0) > 1 and not v.get("std")}
            varying = {k: v for k, v in agg.items() if k not in constant}
            precision_note = self._annotate_precision(state, varying)
            payload: dict[str, Any] = {
                "result_json_seed_0": state.get("result_json") or {},
                "replicates": _elide_value_lists(replicates),
                # Each metric carries mean/std/n/min/max + se + 95% CI bounds, and ``ci_method``: what that
                # interval is an interval of.
                "aggregate_mean_std": varying,
                "n_replicates": len(replicates),
            }
            if precision_note:
                payload_notes.append(str(precision_note))
            coverage = _run_manifest.strata_coverage(self._protocol_block(state), state.get("result_json"))
            if coverage:
                payload_notes.append(
                    "; ".join(coverage) + ". Say so in the limitations and in the summary: the result is for those "
                    "settings, not the whole design."
                )
            if constant:
                payload_notes.append(
                    f"{len(constant)} further metric(s) were identical across "
                    f"all {len(replicates)} seeds and are omitted from the result block; see "
                    f"result_json_seed_0 for their values."
                )
            # Per-stratum CIs + pairwise effect sizes + multiple-comparison
            # guard for any by_<factor> breakdowns (empty otherwise).
            comparison_stats = self._comparison_stats(state, replicates, assertions=assertions)
            if comparison_stats:
                payload["comparison_stats"] = comparison_stats
            if all_ds:
                payload["_user_supplied_datasets"] = all_ds
            result_json_block, _rj_orig = _compact_result_json_block(payload)
            if state.get("result_json_deterministic"):
                # Without this the line reads "0 numeric keys", which is what a
                # broken aggregator looks like. Despite the flag's name,
                # result_json_deterministic only says every seed agreed: that is
                # what these runs showed, not a failure to measure, and not proof
                # that the experiment has no randomness (a script with no random
                # source stops early and publishes no replicates).
                self._log.info(
                    "[analyze] all %d seeds gave the same numbers, so there is no "
                    "spread over them to report",
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
        # The script's own stdout, alone: FI's notes about the run go in their own section of the prompt (``fi_notes``
        # below), never in front of the stdout or inside the result payload, where the model read them as the script's
        # output and gave them back as findings about the results.
        stdout_for_analyze = exec_result.get("stdout_tail", "")[:2000]
        fi_notes: list[str] = []
        # The engine's own oracle verdicts: the script's results may re-judge the same oracles with a stale tolerance.
        if oracle_note := _oracle.analysis_note(self._judged_checks()).strip():
            fi_notes.append(oracle_note)
        if state.get("result_json_replicate_seed_ignored") and state.get("result_json_no_random_source"):
            fi_notes.append(
                "[FI NOTE] This study is deterministic: the script draws no random numbers, so one run is the whole "
                "result and repeating it returns the same numbers. Report it as a single deterministic run and do NOT "
                "report a mean over seeds, a standard error or a confidence interval. Do not list missing repeats as "
                "a limitation. Say instead that its trust rests on convergence, conservation and analytic-limit "
                "checks, and name which of those were and were not done."
            )
        elif state.get("result_json_replicate_seed_ignored"):
            fi_notes.append(
                "[FI NOTE] Replication was configured, but the experiment script "
                "never reads FI_REPLICATE_SEED (or reads it without the value "
                "reaching any random generator), so every replicate repeated the "
                "SAME run and returned the same numbers. There is ONE measurement "
                "here, not several. Report it as a single run: do NOT report a "
                "mean over seeds, a standard error, a confidence interval, or any "
                "spread across replicates, and say plainly in the limitations that "
                "the experiment was not replicated because it does not vary with "
                "the seed it is given."
            )
        if no_difference := self._no_difference_note():
            fi_notes.append(f"[FI NOTE] {no_difference}")
        if once_note := self._ran_once_note():
            fi_notes.append(f"[FI NOTE] {once_note}")
        if degenerate:
            fi_notes.append(
                "[FI NOTE] The experiment ran but every numeric metric is ~0 — "
                "a degenerate/broken run, not a real result. Write this up as an "
                "honest failure note: state plainly that no usable measurement was "
                "produced, hypothesize the likely cause (sampling/grid, threshold, "
                "or normalization bug), and do NOT report the zeros as findings."
            )
        elif not state.get("result_json") and not (
            state.get("no_simulation_resolved") or state.get("survey_mode_resolved")
            or self.config.engine.analyze_local_first
        ):
            # The degenerate note above is for an all-zero result; a script that crashed printed none at all, and the
            # analysis then read like a literature review of the question the experiment was meant to answer.
            fi_notes.append(
                f"[FI NOTE] The experiment produced no results (exit code {exec_result.get('returncode')}). Nothing "
                "was measured. Say so plainly: no number of this run is a result and what failed is said by FI, not "
                "what the literature says about the question, and no number may be reported as a result of this run."
            )
        fi_notes.extend(f"[FI NOTE] {n}" for n in payload_notes)
        fi_record_block = "\n\n".join(fi_notes) or "(FI has nothing to add about this run.)"
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
            fi_record_block=fi_record_block,
            stdout_tail=stdout_for_analyze,
            stderr_tail=_function_steps.traceback_end(exec_result.get("stderr_tail", ""), 1000),
            result_json=result_json_block,
            user_data_block=user_data_block,
            figure_list=_figure_list_for_prompt(state, root=self.quest_root),
            # The literature is analyze's ONLY evidence in survey / no-sim
            # mode (no experiment, and — in survey mode — no dataset either),
            # so it must reach the analyze prompt whole to be synthesised. With
            # an experiment's results to interpret it is the titles only.
            literature_block=self._analyze_literature_block(state, degenerate=degenerate),
        )
        # Multi-model ensemble path: when the YAML carries
        # provider.node_ensemble["analyze"], fan out N parallel calls
        # and merge. The moderator's synthesized output is then parsed
        # the same way as the single-call path, so the rest of the
        # pipeline (cross_check, write) sees an identical shape.
        ensemble_cfg = self._ensemble_for_node("analyze")
        analysis: dict[str, Any] = {}
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
                analysis = _parse_json_lenient(text, node="analyze") or {}
            except EnsembleError as e:
                self._log.warning(
                    "[analyze] ensemble all-failed (%s); falling back to single-call path", e,
                )
        if not isinstance(analysis, dict):  # a reply that is not an object gets the unreadable rule below
            analysis = {}
        if not analysis:
            # One call; an answer that cannot be read is asked for once more. (When the ensemble's merged answer could
            # not be read, this is a fresh ask, so up to two more calls follow it.)
            _, analysis = await self._chat_json(prompt, node="analyze", want=("summary",), usable=bool)
            if not isinstance(analysis, dict):
                analysis = {}
        if not analysis:
            # Unreadable twice: no finding is invented. The evidence gate reads `unreadable` and fails closed.
            self._log.warning("[analyze] the model's analysis could not be read, even when asked once more; the run is "
                              "treated as having no findings (the evidence gate says so)")
            analysis = {"summary": "(parse failed)", "key_findings": [], "unreadable": True}
        # The readers after this take these fields as lists (text, or objects for a finding with a record): findings
        # that are a bare number or string are put in that shape here, once, whichever call gave the answer.
        for key in ("key_findings", "claims_supported"):
            if key in analysis:
                value = analysis[key]
                if isinstance(value, str):
                    analysis[key] = [value]
                elif isinstance(value, list):
                    analysis[key] = [
                        item if isinstance(item, dict) else str(item)
                        for item in value if isinstance(item, (dict, str, int, float, bool))
                    ]
                else:
                    analysis[key] = []
        # Default `next_step` to publish when the LLM omits it (older
        # prompts, parse failures) so the route doesn't break.
        analysis.setdefault("next_step", "publish")
        # Written on every pass, true or false: a later real run clears what an earlier all-zero run set.
        patch: QuestState = {"analysis": analysis, "degenerate_result": degenerate}
        if all_ds:
            patch["user_supplied_datasets"] = all_ds
        return patch

    async def _node_cross_check(self, state: QuestState) -> QuestState:
        """For each result FI's own record holds a value of (:mod:`core.cross_check_items`), search the literature with the
        result's name and the literature step's query, then classify the hits as supporting / conflicting / neutral
        against the sentence FI wrote from the record. Nothing is read from the analysis's text, so no note of FI's can
        become a query, a sentence to classify or a stored claim. A result with no value is entered as not checked, with
        FI's reason, and not searched. Results land in ``state['cross_check']`` and are surfaced in the write prompt's
        ``$cross_check_block``."""
        per_finding_k = self.config.engine.cross_check_per_finding_k
        if per_finding_k <= 0:
            self._log.info("[cross_check] disabled by config (per_finding_k=0)")
            return {"cross_check": [], "cross_check_route": "write"}

        try:
            what = _cross_check_items.items(
                quest_root=self.quest_root, protocol=self._protocol_block(state),
                requirements=self._topic_requirements_read(), covered=self._topic_covered_read(),
                design=state.get("design"),
                context=str(state.get("literature_query") or "").strip()
                or _topic_req.topic_sentence(state.get("topic") or self.config.topic),
                no_simulation=bool(state.get("no_simulation_resolved")), result_json=state.get("result_json"),
                # read from the run being checked, as analyze reads it: a flag an earlier all-zero run left on the state
                # (or a checkpoint written before analyze wrote it on every pass) never marks a later real run as broken
                degenerate=bool(self.config.engine.degenerate_run_guard
                                and _is_degenerate_result(state.get("result_json") or {})))
        except Exception as e:  # noqa: BLE001 -- a record FI cannot read is a literature check not made, never a stop
            self._log.warning("[cross_check] FI's record of the results could not be read (%s: %s); nothing to look up",
                              type(e).__name__, str(e)[:200])
            what = []
        if not what:
            self._log.info("[cross_check] FI's record holds no result to look up; skipping")

        out: list[dict[str, Any]] = []
        for item in what:
            text = item.sentence
            if item.status != _cross_check_items.TO_CHECK:
                self._log.info("[cross_check] not checked: %s (%s)", item.label[:60], item.why_not[:80])
                out.append({
                    "finding": item.label, "result": item.result, "quantity": item.quantity, "status": "not_checked",
                    "supporting": [], "conflicting": [], "neutral": [],
                    "summary": f"(not checked: {item.why_not})", "why_not": item.why_not, "candidates": [],
                })
                continue
            base = {"result": item.result, "quantity": item.quantity, "query": item.query}
            self._log.info("[cross_check] searching for: %s", item.query[:80])
            try:
                hits = await self.knowledge.asearch(
                    item.query, top_k=per_finding_k,
                    chat_fn=functools.partial(self._chat_messages, node="source_router"),
                    work_scope=self._work_scope(state),
                )
            except Exception as e:
                self._log.warning("[cross_check] retrieval failed: %s", e)
                hits = []
            if not hits:
                out.append({
                    "finding": text, **base, "status": "no_literature",
                    "supporting": [], "conflicting": [], "neutral": [],
                    "summary": "(no related literature surfaced)",
                    "candidates": [],
                })
                continue
            # Classify. Single call by default; multi-model vote when
            # provider.node_ensemble["cross_check"] is configured —
            # majority verdict wins per-finding, ties surfaced. Either
            # way ``parsed`` carries the same shape downstream.
            await self._flag_sources(hits, "cross_check")
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
            except _ModelAnswerProblem:
                raise
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
                "finding": text, **base, "status": "checked",
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
        self._log.info("[cross_check] checked %d of %d results", sum(1 for e in out if e["status"] != "not_checked"),
                       len(out))
        patch: QuestState = {"cross_check": out}
        # The route is decided here, once, before the round is spent (the router reads it): a re-route raises the
        # shared iteration counter so the design node sees the new iteration on its next visit, as `_node_review`
        # does on a revise; a re-route FI does not follow (its precision record does not agree, the confirm stage of
        # engine.phased, no round left) costs no round.
        route = self._cross_check_route({**state, **patch})
        patch["cross_check_route"] = route
        if route != "write":
            patch["iteration"] = state.get("iteration", 0) + 1
            self._log.info("[cross_check] analyze.next_step=%s -> %s (iteration %d)",
                           (state.get("analysis") or {}).get("next_step", "publish"), route, patch["iteration"])
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
        venue = self._write_venue(state)
        if venue in SCIENTIFIC_PAPER_FORMATS:
            return ""
        if venue not in NON_SCIENTIFIC_PAPER_FORMATS:
            return ""
        return _load_persona_prefix(venue, category="write")

    def _write_venue(self, state: QuestState) -> str:
        """The paper format the write persona is chosen by (see :meth:`_resolve_write_persona`)."""
        venue = ""
        answers = state.get("clarify_answers") or {}
        if isinstance(answers, dict):
            venue = str(answers.get("paper_venue") or "").strip().lower()
        return venue or self.config.output.paper_format or ""

    def _limitations_heading(self, state: QuestState) -> bool:
        """Whether FI may put a ``## Limitations`` heading in this paper: not in a format whose persona forbids it."""
        return not _write_personas.forbids_limitations(self._write_venue(state))

    async def _node_evidence_gate(self, state: QuestState) -> QuestState:
        """Weigh the assembled evidence against the research question
        BEFORE writing. Returns a verdict the writer (and router) act on.

        When the flag is off, the node is a passthrough. When the gate
        itself cannot be evaluated (provider failure, an unparseable
        reply), routing still defaults to ``write`` — a broken gate call
        must never forfeit the quest's output — but the assessment is
        recorded with ``status: "unknown"``, never ``"sufficient"``: the
        two are not the same claim, and ``core.evidence`` reads the
        difference (an "unknown" gate is a publication-ready gap on every
        profile). Under ``rigor_profile: research`` an unknown status
        additionally pauses the quest for a human to resolve or accept
        the risk, rather than silently writing on unverified evidence.
        The only action a genuinely-decided gate can take that changes
        control flow is ONE bounded ``broaden_lit`` re-entry (capped by
        ``evidence_gate_max_broaden``).
        """
        # With the gate turned off (``engine.evidence_gate: false``) the model's judgement of the evidence is skipped,
        # but not FI's own facts: a search whose objective never moved, an analysis FI cannot read, a run with no
        # results and a fault a test has shown still decide the route. Turning the gate off once skipped the shown-fault
        # stop as well, and a paper was written over a fault FI had shown.
        gate_off = not self.config.engine.evidence_gate
        # The typed contract this quest is held to (topic type, source
        # policy, baseline, success metric). derive_protocol is pure +
        # total — it guards every state access and has defaults — so it
        # won't raise on a malformed checkpoint; building it here means
        # it's always recorded in state AND available to the gate prompt
        # below so the sufficiency call is contract-aware.
        protocol = derive_protocol(state, self.config)
        started = _receipts.now()
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
            lit = [d for d in (lit if isinstance(lit, list) else [])
                   if not (isinstance(d, dict) and _is_preliminary_memory(d.get("metadata") or {}))]
            n_sources = sum(
                1 for d in lit
                # the source's own text: a model's reading of its figures is not a source with text
                if isinstance(d, dict) and _split_figure_readings(str(d.get("content") or ""))[0].strip()
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
                snippet = " ".join(_split_figure_readings(str(d.get("content") or ""))[0][:400].split())[:240]
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
            n_conflicting = n_not_checked = n_quantities_checked = 0
            for rec in cc:
                if not isinstance(rec, dict):
                    continue
                if rec.get("status") == "not_checked":
                    n_not_checked += 1  # no value in FI's record: nothing was searched, so it is neither support nor conflict
                    continue
                if rec.get("quantity"):
                    n_quantities_checked += 1  # the plan's map to the person's quantity stands; else a plan result only
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
                "n_results_checked": sum(1 for r in cc if isinstance(r, dict)) - n_not_checked,
                "n_results_not_checked": n_not_checked,
                "n_topic_quantities_checked": n_quantities_checked,
                "has_results": bool(state.get("result_json") or {}),
                "mode": (
                    "no_simulation"
                    if state.get("no_simulation_resolved") else "simulation"
                ),
                "key_findings_preview": [str(f)[:200] for f in findings[:6]],
            }
            ruled = self._flat_objective_verdict(state) or self._unreadable_analysis_verdict(state) or (
                # The code reader's findings are its opinion: listed in QUALITY.md, never a reason to hold a result back.
                self._no_results_verdict(state)) or (None if gate_off else _evidence_gate_rule(
                protocol.topic_type, n_sources, n_supporting,
                analyze_local_first=self.config.engine.analyze_local_first,
                retrieval_on=self.config.knowledge.enabled,
            ))
            if ruled is None and gate_off:
                return {}
            if ruled is not None:
                parsed, decided_by = ruled, "rule"
            else:
                prompt = self._prompts["evidence_gate"].substitute(
                    topic=str(state.get("topic") or ""),
                    clarify_block=_format_clarify(state),
                    protocol_block=protocol.as_block(),
                    evidence_summary=json.dumps(summary, indent=2),
                )
                _, parsed = await self._chat_json(
                    prompt, node="evidence_gate", want=("verdict",),
                    usable=lambda p: str(p.get("verdict") or "").strip().lower() in (
                        "sufficient", "broaden", "insufficient"),
                )
        except Exception as e:  # noqa: BLE001 — gate must never abort the quest
            claim_stop: dict[str, Any] | None = None
            if gate_off:
                # FI's own stop conditions could not be checked: the same route as a failed gate call (recorded as
                # unknown, a gap in the evidence record), never a silent passthrough to the writer.
                self._log.warning("[evidence_gate] FI's own checks before the writing could not be run (%r); the gate "
                                  "is off, but the failure is recorded as status=unknown", e)
            status = "unknown"
            failure = f"{type(e).__name__}: {e}"[:300]
            self._log.warning(
                "[evidence_gate] assessment failed (%r); routing to write with "
                "status=unknown (not sufficient)", e,
            )
            if claim_stop is not None:
                parsed, decided_by, status, failure = claim_stop, "rule", "ok", ""
        else:
            status, failure = "ok", ""
        verdict = str(parsed.get("verdict") or "").strip().lower()
        if verdict not in ("sufficient", "broaden", "insufficient"):
            if status == "ok":
                # The call succeeded but the reply named no recognised verdict —
                # equally unknown as a raised exception, not a silent "sufficient".
                status = "unknown"
                failure = f"the gate's reply named no recognised verdict: {parsed!r}"[:300]
            verdict = "sufficient"  # keeps existing routing (write, not broaden)
        broadened = int(state.get("evidence_broaden_count", 0) or 0)
        # A broaden goes back through the literature step for more sources. With retrieval off (or --analyze, where
        # that step is skipped) nothing can come back, and the loop only redesigned the experiment after its results
        # were seen: a live quest threw away a complete first run and changed its hypothesis that way. The verdict
        # stands (the writer is told the evidence is thin, and the evidence level names it as a gap); the route is write.
        can_broaden = self.config.knowledge.enabled and not self.config.engine.analyze_local_first
        will_broaden = (
            status == "ok"
            and verdict == "broaden"
            and can_broaden
            and broadened < self.config.engine.evidence_gate_max_broaden
        )
        if status == "ok" and verdict == "broaden" and not can_broaden:
            self._log.info("[evidence_gate] broaden asked, but there is no literature step to broaden (retrieval is off); writing with the gap recorded")
        # An experiment that produced no results goes back to design once while there is an iteration left (the new
        # script gets its own repairs); after that the verdict stays insufficient and the writer is told why.
        redesign = bool(parsed.get("redesign")) and status == "ok"
        fitted = ""
        if redesign and parsed.get("fit_request"):
            # The study was too long to run: plan.md is rewritten to fit the time FI measured before the design is read
            # again. An answer that cannot be used leaves the stop as it was.
            try:
                plan_file = _plan.plan_path(self.quest_root)
                fit = (state.get("exec_result") or {}).get("too_long_fit")
                done = await self._revise_plan_parts(
                    str(parsed["fit_request"]), plan_file, by="engine",
                    scope=_fit_scope(fit if isinstance(fit, dict) else {},
                                     (_plan.raw_design_block(plan_file.read_text(encoding="utf-8")) or {}).get("protocol")),
                    note="a study that fits the time FI measured, asked by FI")
                fitted = f"plan.md was rewritten (version {(done or {}).get('version')}) to fit the time FI measured"
                self._log.warning("[evidence_gate] %s; the new design is timed again before it runs", _one_line(fitted, 300))
            except Exception as exc:  # noqa: BLE001 -- the stop stands
                self._log.warning("[evidence_gate] the plan could not be rewritten to fit the time FI measured (%s); "
                                  "stopping without a paper", _one_line(str(exc), 200))
                redesign = False
                parsed = {**parsed, "stuck": True, "stuck_reason": "too_long"}
        flat_fact = (state.get("exec_result") or {}).get("flat_objective") if parsed.get("flat_request") else None
        flat_retry = False
        if redesign and parsed.get("flat_request"):
            # The search's objective did not move: plan.md is rewritten from what FI measured before the design is read
            # again. An answer that cannot be used runs the first design's search again, to its whole budget.
            try:
                done = await self._flat_rewrite(state, str(parsed["flat_request"]))
                self._log.warning("[evidence_gate] plan.md was rewritten (version %s) from what FI measured of the "
                                  "search; the new design is checked and timed before it runs", (done or {}).get("version"))
                self._progress("[FI] The search's objective " + _flat_moved(flat_fact) + " in the design's range; the "
                               "plan was revised once so that it can move, and the new design runs next.")
            except Exception as exc:  # noqa: BLE001 -- the first design is searched again, with no early stop
                said = (f"the plan could not be rewritten from what FI measured of the search ({_one_line(str(exc), 200)}); "
                        "the first design's search runs again to its whole budget")
                self._log.warning("[evidence_gate] %s", said)
                self._progress("[FI] The search's objective " + _flat_moved(flat_fact) + " in the design's range, and "
                               "the plan could not be revised for it; the search runs again to its whole budget.")
                redesign = False
                flat_retry = True
                # Back to the run step (the route a crash repair takes), which runs the search again with no early stop.
                parsed = {**parsed, "repair_crash": True, "rationale": said, "gaps": [said]}
        # A script that could not run to the end, even after its repairs, is the code's fault, not the design's: the same
        # design is written into code again (core/engine.py::_no_results_verdict).
        rewrite_code = bool(parsed.get("rewrite_code")) and status == "ok"
        no_findings = bool(parsed.get("stuck")) and status == "ok"
        # The run step's crash repair (a search whose objective did not move, run again with no early stop).
        crash_repair = bool(parsed.get("repair_crash")) and status == "ok" and not no_findings
        assessment = {
            "verdict": verdict,
            "status": status,
            "failure": failure,
            "route": ("stuck" if no_findings else "repair_crash" if crash_repair else "rewrite_code" if rewrite_code
                      else "redesign" if redesign else "broaden_lit" if will_broaden else "write"),
            "rationale": str(parsed.get("rationale") or ""),
            "gaps": [str(g) for g in (parsed.get("gaps") or []) if str(g).strip()],
            "n_sources": n_sources,
            "n_supporting": n_supporting,
            "decided_by": decided_by,
            **({"stuck_reason": str(parsed["stuck_reason"])} if no_findings and parsed.get("stuck_reason") else {}),
        }
        self._log.info(
            "[evidence_gate] verdict=%s status=%s route=%s decided=%s type=%s policy=%s "
            "(sources=%d, findings=%d, supporting=%d, broadened=%d)",
            verdict, status, assessment["route"], decided_by, protocol.topic_type,
            protocol.source_policy, n_sources, n_findings,
            n_supporting, broadened,
        )
        self._write_receipt(
            "evidence_gate", "pass" if status == "ok" and verdict == "sufficient" else "fail" if status == "ok" else "unknown",
            started_at=started, inputs=_gate_inputs(state),
            output=assessment, error=failure,
            detail=(assessment["rationale"] if flat_retry else
                    f"the code went back to the crash repair: {assessment['rationale']}" if crash_repair else
                    f"it judged the evidence {verdict} and the paper was written on it (gaps: "
                    f"{'; '.join(assessment['gaps']) or 'none named'})" if status == "ok" and verdict != "sufficient" else ""),
        )
        if status == "unknown":
            self._stop_once_for_check(
                "evidence_gate", failure or "no usable reply from the gate call",
                look="the [evidence_gate] entry in .fi/run.log",
                inputs=_gate_inputs(state),
            )
        else:
            self._clear_check_stop("evidence_gate")
        patch: QuestState = {
            "evidence_assessment": assessment,
            "research_protocol": protocol.model_dump(),
        }
        if will_broaden:
            patch["evidence_broaden_count"] = broadened + 1
        patch.update(self._figures_after_taken_out(state))
        if rewrite_code:
            # The design stays as it is (no design iteration is spent); the new code gets its own repairs.
            patch.update(_FRESH_SCRIPT)
            # Once per design (a new design's code gets its own); the design's own retry (below) is counted apart.
            patch["evidence_code_rewritten_at"] = int(state.get("iteration", 0) or 0)
        elif flat_retry:
            # The plan could not be rewritten: the first design's search runs again, never stopped for this again.
            patch["flat_objective_redesigned"] = True
        elif redesign and parsed.get("flat_request"):
            # A change made after results were seen: it spends a design iteration (engine.max_iterations), once per quest.
            patch["iteration"] = int(state.get("iteration", 0) or 0) + 1
            patch["design_from_plan"] = True
            patch["flat_objective_redesigned"] = True
            patch["flat_objective_fact"] = dict(flat_fact or {})
        elif redesign and parsed.get("fit_request"):
            # Nothing of the study ran and no result was seen: the rewritten plan is still the plan's first design, so no
            # design iteration is spent, and it is timed and checked before the run as the first one was.
            patch["evidence_fit_redesigns"] = int(state.get("evidence_fit_redesigns") or 0) + 1
            patch["design_from_plan"] = True
        elif redesign:
            patch["iteration"] = int(state.get("iteration", 0) or 0) + 1
            patch["evidence_no_result_retries"] = int(state.get("evidence_no_result_retries") or 0) + 1
        return patch

    def _figures_after_taken_out(self, state: QuestState) -> QuestState:
        """The state's ``figures`` and ``figure_records`` without the figures FI took out at the gate (the record is
        ``needs/RESULT_CHECK.json``). The paper's numbers, captions, in-text citations, slides and poster all start from these
        two, so a figure whose file is gone must leave them too; nothing else is changed. Nothing when none was taken out."""
        from . import result_checks as _result_checks

        gone = _result_checks.removed_figure_stems(self.quest_root)
        figures = [f for f in state.get("figures") or []]
        if not gone or not any(Path(str(f)).stem in gone for f in figures):
            return {}
        records = {k: v for k, v in (state.get("figure_records") or {}).items() if Path(str(k)).stem not in gone}
        return {"figures": [f for f in figures if Path(str(f)).stem not in gone], "figure_records": records}

    def _unreadable_analysis_verdict(self, state: QuestState) -> dict[str, Any] | None:
        """The gate's verdict when the model's analysis of the results could not be read (twice): ``insufficient``, never
        a count over findings that were not read. ``None`` for every other analysis."""
        analysis = state.get("analysis")
        if not (isinstance(analysis, dict) and analysis.get("unreadable")):
            return None
        why = "the model's analysis of the results could not be read, even when asked twice, so there are no findings"
        self._log.warning("[evidence_gate] %s; the evidence is judged insufficient", why)
        return {"verdict": "insufficient", "rationale": why, "gaps": [why], "stuck": True}

    def _set_aside_stuck_record(self) -> None:
        """A refine that re-opens a quest which ended in an honest stop moves ``needs/STUCK.json`` (and so its to-do card)
        aside, to ``.fi/set_aside_by_stuck/<time>/``, kept for the history: the pass that follows is not that stop's."""
        record = self.quest_root / "needs" / "STUCK.json"
        if not record.is_file():
            return
        aside = self.fi_dir / "set_aside_by_stuck" / time.strftime("%Y%m%d-%H%M%S")
        try:
            aside.mkdir(parents=True, exist_ok=True)
            shutil.move(str(record), str(aside / "STUCK.json"))
            self._log.info("[run] the earlier stop (needs/STUCK.json) is kept in %s; this pass starts without it",
                           aside.relative_to(self.quest_root).as_posix())
        except OSError as exc:
            self._log.warning("[run] could not set needs/STUCK.json aside (%r)", exc)

    async def _node_stuck_no_findings(self, state: QuestState) -> QuestState:
        """The experiment ran but the model's reading of its results could not be read, even when asked twice: FI has no
        findings to write up, so the quest stops here without a paper. The same stop as a simulation that would not work
        (``needs/STUCK.json``, which the to-do card reads); what FI tried is in the record; nobody is asked to debug.
        The run's results stay on disk; a paper an earlier round left is set aside (kept, not deleted)."""
        reason = (state.get("evidence_assessment") or {}).get("stuck_reason")
        no_numbers = reason == "no_numbers"
        crashed = reason == "crashed"
        headline_missing = reason == "headline_missing"
        claims = [str(h) for h in (state.get("review") or {}).get("must_flag_hits") or []
                  if _hit_name(h) == _record_claims.HIT]
        repairs = _numeric.repairs_made(state)
        # Said only when it happened: a quest with no iteration left stops on the first pass, with no second design.
        redesigned = int(state.get("evidence_no_result_retries") or 0) >= 1
        rewritten = state.get("evidence_code_rewritten_at") is not None  # its own try, counted apart from the design's
        timed_out = bool((state.get("exec_result") or {}).get("timed_out"))  # stopped for its time, not by an error
        ran_again = redesigned or rewritten
        tried_more = [x for x in ("the code was written again from the same design" if rewritten else "",
                                  "the design was asked for again" if redesigned else "") if x]
        again = (", even after " + " and ".join(tried_more)) if tried_more else ""
        repaired = f"repaired the script {repairs} time{'s' if repairs != 1 else ''}" if repairs else ""
        after = " and ".join(x for x in (repaired, "wrote the code again from the same design" if rewritten else "",
                                         "asked for the design again" if redesigned else "") if x)
        # A result that holds numbers, but none of the study's quantities and says it failed (see _reports_no_quantity).
        bare = self._reports_no_quantity(state, state.get("result_json")) if no_numbers else ""
        not_done = "ran past the time it is allowed" if timed_out else "could not run to the end"
        problem = ((f"the experiment ran, but its result holds none of the study's quantities ({bare}){again}, so FI has "
                    "nothing measured to write up" if bare else
                    f"the experiment ran, but its result has no number in it{again}, so FI has nothing measured to write up")
                   if no_numbers else
                   f"the experiment {not_done}{', even after FI ' + after if after else ''}, "
                   "so FI has nothing measured to write up" if crashed else
                   "the experiment ran, but the model's reading of its results could not be read, even when asked twice, "
                   "so FI has no findings to write up")
        redo = ((["wrote the code again once from the same design (the design was not changed)"] if rewritten else [])
                + (["sent the quest back to the design once, so the script was written again"] if redesigned else []))
        tried = ([("ran the experiment and read the result it printed: it says it failed and has no number for the study's "
                   "quantities" if bare else "ran the experiment and read the result it printed: no number in it"), *redo,
                  *([("ran it again: the result still holds none of the study's quantities" if bare else
                      "ran it again: the result still has no number in it")] if ran_again else [])] if no_numbers else
                 [*(["asked the model to repair the script when it failed"
                     + (f" ({repairs} repair{'s' if repairs != 1 else ''})" if repairs else "")] if repairs else
                    [f"ran the experiment: {'it was stopped for running past the time allowed' if timed_out else 'it stopped with an error'}"]),
                  *redo, *([f"ran it again: it still {not_done}"] if ran_again else [])] if crashed else
                 ["asked the model to analyse the results",
                  "read its reply as written, then again with the backslashes of formulas kept as written",
                  "asked the model a second time for one short answer; that reply could not be read either"])
        ended = (("the run gave no result with a number for the study's quantities" if bare else
                  "the run gave no result with a number in it") + again if no_numbers else
                 f"the script {not_done}" + again if crashed else "the analysis could not be read twice")
        if claims and reason not in ("flat", "no_numbers", "crashed"):
            # The paper claimed what FI's own records contradict, even after it was written again (core/record_claims.py).
            problem = (f"the paper still claims what FI's own records contradict, even after it was written again "
                       f"{_RECORD_REWRITES} times, so FI will not hand it over as a finished paper")
            tried = ["compared the paper with FI's records of the search, the known-answer checks and the figures the run drew",
                     *[_one_line(c, 400) for c in claims[:3]],
                     f"asked for the paper to be written again with the contradiction named ({_RECORD_REWRITES} times); "
                     "the claim was still there"]
            ended = "the paper still claimed what FI's records contradict"
        if headline_missing:
            # A second model compared the code with the plan and the main quantity is not in it, even after repairs.
            missing = _code_review.headline_missing(self.quest_root, self.quest_root / "code")
            constant = [] if missing else _code_review.headline_constant(self.quest_root, self.quest_root / "code")
            left = missing or constant
            if constant:
                problem = ("the study's main quantity is a constant in the run's results, as a second model found in the code"
                           f" ({_one_line('; '.join(constant), 200)}){', even after FI ' + after if after else ''}, so "
                           "there is nothing measured to write up")
            else:
                problem = ("the code does not compute the study's main quantity"
                           f"{' (' + _one_line('; '.join(left), 200) + ')' if left else ''}"
                           f"{', even after FI ' + after if after else ''}, so there is nothing measured to write up")
            review_record = _code_review.load(self.quest_root)
            rounds = int(review_record.get("send_backs") or 0)
            tried = ["asked a second model to read the code against what the plan requires it to compute",
                     *[_one_line(x, 300) for x in left[:3]],
                     *([f"sent the findings to be repaired ({rounds} round{'s' if rounds != 1 else ''}); "
                        + (_code_review.metrics_line(review_record) or "the findings stood")] if rounds else []),
                     *(["asked the model to repair the script" + (f" ({repairs} repair{'s' if repairs != 1 else ''})")]
                       if repairs else [])]
            ended = "the code still did not compute the study's main quantity"
        if reason == "shown_fault_open":
            # A fault that changes a reported number was shown by a failing test; the test still fails on the final code.
            said = _one_line(str((state.get("evidence_assessment") or {}).get("rationale") or ""), 400).rstrip(".")
            problem = f"{said}{', even after FI ' + after if after else ''}, so FI cannot vouch for the numbers"
            review_record = _code_review.load(self.quest_root)
            rounds = int(review_record.get("send_backs") or 0)
            tried = ["asked a second model to read the code against the plan",
                     "wrote a test for each problem it found and ran it: this one failed on the code",
                     *([f"asked for the code to be repaired ({rounds} round{'s' if rounds != 1 else ''}); the problem stood"]
                       if rounds else []),
                     "ran the test again on the code that produced the results: it did not pass"]
            ended = "a fault shown by a failing test was never shown to be gone from the code"
        if reason == "not_run":
            said = str((state.get("exec_result") or {}).get("not_run") or "").strip().rstrip(".")
            problem = f"{said}, so nothing was run for the study"
            tried = ["rewrote the simulation to compute the results it had typed in, and ran the known-answer checks on it",
                     said]
            ended = "the simulation's text still could not be checked for typed-in results"
        if reason == "too_long":
            # The study would not finish in the time allowed, even made smaller (core/run_estimate.py): nothing was run.
            said = str((state.get("exec_result") or {}).get("too_long") or "").strip().rstrip(".")
            problem = f"{said}, so nothing was run for the study" if said else "the experiment would not fit the time allowed"
            tried = [str(t) for t in (state.get("exec_result") or {}).get("too_long_tried") or []] or [
                "timed a few real trials of the experiment and added them up",
                said or "the total is more than the time allowed"]
            ended = ("the settings of the plan were still more than FI lists"
                     if (state.get("exec_result") or {}).get("too_long_tried") else
                     "the experiment would still take longer than the time allowed")
        # A study whose kept search cannot be used says, in its own words, what happened and what to do
        # (core/optimise.py): those words are the whole report, with nothing added to them.
        own_words = reason == "no_search" and bool((state.get("exec_result") or {}).get("no_search_advice"))
        if reason == "no_search":
            # The plan is a search for the best design and the plan's model could not write a search FI can run.
            said = str((state.get("exec_result") or {}).get("no_search") or "").strip().rstrip(".")
            problem = _one_line(said, 500) if own_words else f"{_one_line(said, 500)}, so nothing was run for the study"
            tried = [str(t) for t in (state.get("exec_result") or {}).get("no_search_tried") or []] or [said]
            ended = ("the search this study keeps could not be used" if own_words
                     else "the plan's model could not write a search FI can run")
        if reason == "cannot_repair":
            said = str((state.get("exec_result") or {}).get("cannot_repair") or "").strip().rstrip(".")
            problem = f"{_one_line(said, 400)}, and FI's repairs did not remove it, so there is nothing it can vouch for"
            tried = [str(t) for t in (state.get("exec_result") or {}).get("cannot_repair_tried") or []] or [
                "repaired the code as far as its limit allows", said]
            ended = "FI's repairs ended and the problem stood"
            if (state.get("exec_result") or {}).get("fis_fault"):
                # A fault of FI's own analysis (core/analysis_frame.py): the trials ran, and no repair of the study's code was tried
                problem = f"{_one_line(said, 400)}; the trials that ran are kept"
                ended = "the fault is in FI's own analysis, which no repair of the study's code can change"
        if reason == "flat":
            # The simulation gives the same numbers whatever it is given (core/flat_output.py): not a finding.
            symptom = str((state.get("exec_result") or {}).get("flat_output") or "").strip().rstrip(".")
            problem = ("the simulation gives the same result whatever it is given"
                       f"{', even after FI ' + after if after else ''}, so there is nothing measured to write up")
            tried = ["ran the simulation: " + (_one_line(symptom, 300) or "its result did not change with its inputs"),
                     *(["asked the model to repair the simulation" + (f" ({repairs} repair{'s' if repairs != 1 else ''})"
                                                                      if repairs else "")] if repairs else []),
                     *redo, *(["ran it again: its result still did not change with its inputs"] if redesigned else [])]
            ended = "the simulation's result still did not depend on its inputs" + again
        record: dict[str, Any] = {"at": time.time(), "kind": "no_findings", "problem": problem, "tried": tried,
                                  "say": f"FI stopped: {problem}. No paper was written; the results are kept.",
                                  "why_repairs_ended": ended}
        aside = self.fi_dir / "set_aside_by_stuck" / time.strftime("%Y%m%d-%H%M%S")
        for name in [n for n in _rerun_from._PAPER
                     if "*" not in n and n not in ("frontier_insight_summary.json", "NEXT_STEP.md")]:
            src = self.quest_root / name
            if src.exists():
                try:
                    aside.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(aside / name))
                except OSError as exc:
                    self._log.warning("[stuck] could not set %s aside (%r)", name, exc)
        if aside.is_dir():
            record["set_aside"] = aside.relative_to(self.quest_root).as_posix()
        try:
            needs = self.quest_root / "needs"
            needs.mkdir(parents=True, exist_ok=True)
            (needs / "STUCK.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[stuck] could not write needs/STUCK.json (%r)", exc)
        self._log.warning("[stuck] %s; stopping without a paper (needs/STUCK.json)", problem)
        for line in tried:
            self._log.info("[stuck] tried: %s", line)
        offered = str((state.get("exec_result") or {}).get("cannot_repair_advice") or "") if reason == "cannot_repair" else ""
        if offered:
            record["choice"] = offered
            try:
                (self.quest_root / "needs" / "STUCK.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
            except OSError:
                pass
        print(f"[FI] quest {self.quest_id}: {problem}. No paper was written; the results are kept "
              "(needs/STUCK.json says what FI tried)." if own_words else self._routed_words(
            f"[FI] quest {self.quest_id}: {problem}. No paper was written; the results are kept. Nothing for you to "
            "fix: needs/STUCK.json says what FI tried; " + (offered + " Or " if offered else "") + "run the quest again, "
            + ("or try another model for the design (`provider.node_models.design`)." if no_numbers or crashed
               or reason == "flat" or headline_missing else
               "or try another model for the code (`provider.node_models.implement`)." if reason == "shown_fault_open" else
               (str((state.get("exec_result") or {}).get("too_long_advice") or "")
                or "or allow more time (`execution.timeout_s`), or a smaller study in the plan.") if reason == "too_long" else
               "or try another model for the code (`provider.node_models.implement`)." if reason == "cannot_repair" else
               "or ask for a change in words (`--revise-plan \"<what to change>\"`)." if reason == "no_search" else
               "or try another model for the design (`provider.node_models.design`)." if reason == "not_run" else
               "or ask for the paper again with another model for `write` (`provider.node_models.write`)." if claims else
               "or ask for the analysis again with another model for `analyze` (`provider.node_models.analyze`).")))
        self._audit("stuck", problem=problem, repairs=len(tried))
        return {"stuck": record}

    def _reports_no_quantity(self, state: QuestState, result: Any) -> str:
        """When the run's result says it failed (``errors`` text, a ``status`` of invalid or failed) and holds a number for
        none of the quantities the protocol measures (``protocol.metrics``), a sentence saying so; else ``""``. A protocol
        with no metrics, or a result that does not say it failed, is not judged here."""
        protocol = self._protocol_block(state)
        metrics = protocol.get("metrics") if isinstance(protocol, dict) else None
        ids = [str(m.get("id")).strip() for m in metrics or [] if isinstance(m, dict) and str(m.get("id") or "").strip()]
        if not ids or not isinstance(result, dict):
            return ""
        said = _code_review.reports_failure(result)
        if not said or len(_code_review.absent_from_results(ids, result)) != len(ids):
            return ""
        return f"it says {said[:100]}, and has no number for {', '.join(ids[:4])}"























    def _flat_objective_verdict(self, state: QuestState) -> dict[str, Any] | None:
        """The gate's verdict for a search whose objective did not move across the design space (the runner hands this
        back only while the plan may still be revised for it, ``_flat_redesign_allowed``), else ``None``: the plan's model
        is told what FI measured and asked, once, for a design whose objective can move. FI names no representation,
        resolution or range. Decided before every other rule: the run of this round is what it says, whatever an earlier
        round's analysis held."""
        exec_result = state.get("exec_result")
        fact = exec_result.get("flat_objective") if isinstance(exec_result, dict) else None
        if not (isinstance(fact, dict) and fact):
            return None
        why = _flat_fact_sentence(fact)
        self._log.warning("[evidence_gate] %s; the plan is asked once for a design whose objective can move", why)
        return {"verdict": "insufficient", "rationale": why + "; the plan was asked for a design whose objective can "
                "move", "gaps": [why], "redesign": True,
                "flat_request": _flat_request(fact, _topic_req.kept_when_smaller(self._topic_requirements_read()))}

    def _no_results_verdict(self, state: QuestState) -> dict[str, Any] | None:
        """The gate's verdict for a simulation whose experiment produced no results, else ``None``.

        The count rule said "sufficient" for two live quests whose final script had crashed: with no results the
        analysis summarised the literature, the cross-check found sources for those summaries, and the counts cleared
        the rule. The writer got no note, and the papers went out on no experiment. Now the quest goes back to design
        once while an iteration is left; after that the verdict is ``insufficient`` and the evidence note says why."""
        exec_result = state.get("exec_result")
        stopped = next((k for k in _STOPPED_BEFORE_THE_STUDY if isinstance(exec_result, dict) and exec_result.get(k)), "")
        if stopped:
            # Stopped before anything ran (the execute step's record of it): the gate states that, whatever the code or the
            # results hold, so the reason the quest reports is the one that stopped it.
            why = str(exec_result[stopped])  # type: ignore[index]
            fit = exec_result.get("too_long_fit") if stopped == "too_long" else None  # type: ignore[union-attr]
            if (isinstance(fit, dict) and int(state.get("evidence_fit_redesigns") or 0) < 1
                    and int(state.get("iteration", 0) or 0) == 0 and _plan.plan_path(self.quest_root).is_file()):
                # The plan was written before anything was timed: its model is told what FI measured and asked, once, for
                # a study that fits (FI itself never chooses which settings to leave out). The new design is timed again.
                self._log.warning("[evidence_gate] %s; the plan is asked once for a study that fits the time FI measured",
                                  why)
                return {"verdict": "insufficient", "rationale": why + "; the plan was asked for a study that fits",
                        "gaps": [why], "redesign": True,
                        "fit_request": _fit_request(why, fit, _topic_req.kept_when_smaller(self._topic_requirements_read()))}
            self._log.warning("[evidence_gate] %s; stopping without a paper", why)
            return {"verdict": "insufficient", "rationale": why, "gaps": [why], "stuck": True, "stuck_reason": stopped}
        if isinstance(exec_result, dict) and exec_result and not state.get("no_simulation_resolved") and (
                _missing_headline :=_code_review.resolve_headline(
                    self.quest_root, self.quest_root / "code", state.get("result_json"))):
            # A second model found the study's main quantity missing from the code, the repairs did not add it, and no
            # result of the run is named like it. That rests on one reading and a match of names (a quantity the results
            # hold under other words is not found by it), so it does not end the quest (core/stop_power.py): the paper
            # says, in its limitations, that the quantity was not computed (core/code_review.py::write_note).
            self._log.warning(
                "[evidence_gate] a second model found the study's main quantity missing from the code, and no result is "
                "named like it (%s); this is %s, so the paper says it is not computed and the quest goes on",
                "; ".join(_missing_headline)[:200], _stop_power.source_words(_stop_power.READ))
        if isinstance(exec_result, dict) and exec_result and not state.get("no_simulation_resolved") and (
                _constant := _code_review.resolve_logic(self.quest_root, self.quest_root / "code", state.get("result_json"))):
            # A second model found, with high severity, that the study's main quantity is a constant in the code, and the
            # run's own results hold it at one value everywhere: the results confirm it (a reader's word alone never stops).
            # A constant can be the finding itself (a quantity that does not depend on the settings), so this needs the
            # reader's word to mean a fault: it is listed in QUALITY.md, and the quest goes on.
            self._log.warning("[evidence_gate] the study's main quantity is a constant in the run's results, as a second "
                              "model found in the code (%s); QUALITY.md lists it and the quest goes on",
                              "; ".join(_constant)[:200])
        from . import numeric_oracle as _numeric_oracle

        result = state.get("result_json")
        # A result with no number in it (the script printed its RESULT_JSON line with nothing measured) is no result.
        # A run whose result has no number in it (none printed, `{}`, `[]`, only words) is no result; a `0` is a number.
        no_numbers = not any(True for _ in _numeric_oracle.flatten_numbers(result, keep_zero=True))
        # A number that is not one of the study's own quantities (a count of the records it refused, the trials it
        # required) does not make a result: a script that says it failed, and holds none of the quantities the protocol
        # measures, produced none. Its own word about itself, and every quantity missing (a name written another way
        # still counts), are both needed.
        failed_itself = "" if no_numbers else self._reports_no_quantity(state, result)
        if (
            not (no_numbers or failed_itself)
            or not isinstance(exec_result, dict) or not exec_result  # no experiment has run: not this case
            or state.get("no_simulation_resolved")
            or state.get("survey_mode_resolved")
            or self.config.engine.analyze_local_first
        ):
            return None
        flat = str(exec_result.get("flat_output") or "")
        why = (f"the experiment's result has no number in it (exit code {exec_result.get('returncode')!s}"
               if result else f"the experiment produced no results (exit code {exec_result.get('returncode')!s}")
        if failed_itself:
            why = (f"the experiment's result holds none of the study's quantities ({failed_itself}; exit code "
                   f"{exec_result.get('returncode')!s}")
        if flat:
            why = f"the simulation's result does not depend on what it is given (exit code {exec_result.get('returncode')!s}"
        if state.get("exec_give_up_reason"):
            why += f"; the repair gave up: {_one_line(state.get('exec_give_up_reason'), 160)}"
        why += ")"
        # "Could not run to the end" only when the script really exited with an error; one that finished (exit 0) and printed
        # nothing measured is the other wording, whether or not a repair was tried.
        crashed = exec_result.get("returncode") not in (0, None)
        # A script stopped for running past the time allowed did not crash: the study is too large for it, which a new
        # design can change and new code of the same design cannot.
        code_fault = crashed and not flat and not exec_result.get("timed_out")
        if code_fault:
            if state.get("evidence_code_rewritten_at") == int(state.get("iteration", 0) or 0):
                # The code was already written again from this design and still could not run to the end: a new design
                # is no answer to a crash (see below), so nothing is left to try.
                return {"verdict": "insufficient", "rationale": why, "gaps": [why], "stuck": True,
                        "stuck_reason": "crashed"}
            # The script could not run to the end, even after its repairs: the code's fault, not the design's. In the
            # quest records of every no-result run sent back to design, the cause was a script that crashed; where a quest
            # then recovered, the new design varied the same settings and measured the same quantities (only the code was
            # new), and where the design was written anew it brought in new problems of its own. So the design stays and
            # its code is written once more (with its own repairs); the design is never changed for a crash.
            self._log.warning("[evidence_gate] %s; the design stays as it is and its code is written again once", why)
            return {"verdict": "insufficient", "rationale": why + "; the code is written again from the same design",
                    "gaps": [why], "rewrite_code": True}
        # A script that finished with nothing measured (or ran past its time) is asked of the design once, whether or not
        # its code was written again before: that is a different failure from a crash.
        if (
            int(state.get("evidence_no_result_retries") or 0) < 1
            and int(state.get("iteration", 0) or 0) < self.config.engine.max_iterations
        ):
            self._log.warning("[evidence_gate] %s; sending the quest back to design once", why)
            return {"verdict": "insufficient", "rationale": why + "; tried once more from the design", "gaps": [why],
                    "redesign": True}
        # Tried once more and still nothing measured: there is nothing to write up, so no paper is written.
        return {"verdict": "insufficient", "rationale": why, "gaps": [why], "stuck": True,
                "stuck_reason": "flat" if flat else "crashed" if crashed else "no_numbers"}
