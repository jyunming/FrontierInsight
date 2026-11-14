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
    _record_claims, _replicate_assertions, _replicate_metric_kinds, _rerun_check, _rerun_from, _run_manifest,
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
                self._no_results_verdict(state)) or (
                await self._shown_fault_verdict(state)) or (None if gate_off else _evidence_gate_rule(
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
            try:  # what the gate had not decided about a second model's claims is decided now, as if no rerun could be made
                self._settle_undecided_reader_claims(state)
            except Exception as settle_exc:  # noqa: BLE001
                # Failing closed: what cannot be settled is left out, and a leave-out that cannot be recorded stops the quest.
                self._log.warning("[evidence_gate] the second model's claims could not be settled (%r); they are left out", settle_exc)
                claim_stop = self._leave_out_undecided_claims(state, f"{type(settle_exc).__name__}: {settle_exc}"[:200])
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

    async def _shown_fault_verdict(self, state: QuestState) -> dict[str, Any] | None:
        """The gate's stop for a fault in the code that changes a reported number, was shown by a failing test, is still
        flagged by the latest reading, and was never shown to be gone from the code that produced the results; else
        ``None``.

        The code may have changed since the last reading (a repair of the run, a rewrite), and a reader's word alone never
        stops a quest, so the test written for each such fault (at most ``MAX_SHOWN_RERUNS``) is run again on the code as
        it is now, each in its own process with its own time limit; a test that does not pass is run once more (a flaky
        test does not stop a quest). Only a test's own assertion about values shows a fault, when it is first shown and
        here (core/finding_repair.py): a test that still fails by its assertion counts; one that errors (in the code or in
        the test), gives no answer or is gone from disk shows nothing and stops nothing, and the record keeps where it
        errored (``gate_error``). An error inside this check itself stops the quest.
        Repaired and no-longer-flagged faults are not run again (see ``code_findings.shown_open``). Deterministic given the
        files on disk, so a re-entry of the node gives the same answer."""
        exec_result = state.get("exec_result")
        if not (isinstance(exec_result, dict) and exec_result) or state.get("no_simulation_resolved"):
            return None
        record = _code_review.load(self.quest_root)
        # What an earlier pass of the gate decided about an issue is decided again on this run's code and results.
        stale = [i for i in _code_findings.issues_of(record).values()
                 if isinstance(i, dict) and any([i.pop("at_gate", None), i.pop("gate_error", None)])]
        if stale:
            _code_review.save(self.quest_root, record)
        names = _code_review.headline_names(state.get("design"), self._protocol_block(state))
        measured = await self._measure_reader_findings(record, state, self._shown_fault_reach(names, state), names)
        if measured is not None:
            return measured
        aside_before = {str(x.get("id")) for x in record.get("about_fi") or [] if isinstance(x, dict)}
        if _code_findings.set_aside_own(record, self.quest_root / "code", _code_review.version(self.quest_root / "code")):  # a finding about FI's own lines stops nothing
            _code_review.save(self.quest_root, record)
        self._log_about_fi(record, aside_before)
        shown = _code_findings.shown_open(record)
        if not shown:
            return None
        # A fault in a function the run never calls changes no reported number (core/code_reach.py): it stays in the record.
        never_run = _code_reach.never_run_in(self.quest_root)
        unrun = [i for i in shown if never_run(i.get("file"), i.get("function"))]
        if unrun:
            for issue in unrun:
                saved = (record.get("issues") or {}).get(str(issue.get("id")))
                if isinstance(saved, dict):
                    saved["never_run"] = True
                self._log.warning("[evidence_gate] %s is about %s %s, which the run never calls: it changes no reported "
                                  "number and does not stop the quest", issue.get("id"), issue.get("file"), issue.get("function"))
            _code_review.save(self.quest_root, record)
            shown = [i for i in shown if i not in unrun]
            if not shown:
                return None
        # A quantity the reader called missing that the results hold under another name is a difference of names, not a
        # fault that changes a reported number: the results contradict the reader, and the issue no longer stops anything.
        renamed = {str(i.get("id")): keys for i in shown
                   if (keys := _code_review.reported_under_another_name(record, i, state.get("result_json")))}
        if renamed:
            for fid, keys in renamed.items():
                issue = (record.get("issues") or {}).get(fid)
                if isinstance(issue, dict):
                    issue["changes_result"] = False
                    issue["reported_as"] = keys
                self._log.warning("[evidence_gate] %s calls a quantity missing that the results hold as %s: a difference "
                                  "of names, not a wrong number; it does not stop the quest", fid, ", ".join(keys)[:200])
            _code_review.save(self.quest_root, record)
            shown = [i for i in shown if str(i.get("id")) not in renamed]
            if not shown:
                return None
        if len(shown) > _code_findings.MAX_SHOWN_RERUNS:
            self._log.warning("[evidence_gate] %d shown issue(s) beyond the first %d are not run again: %s", len(shown)
                              - _code_findings.MAX_SHOWN_RERUNS, _code_findings.MAX_SHOWN_RERUNS,
                              ", ".join(str(i.get("id")) for i in shown[_code_findings.MAX_SHOWN_RERUNS:])[:200])
        shown = shown[:_code_findings.MAX_SHOWN_RERUNS]
        try:
            design = state.get("design")
            return await self._shown_fault_check(shown, record, state.get("result_json"),
                                                 _code_review.headline_names(design, self._protocol_block(state)), state)
        except Exception as exc:  # noqa: BLE001 - a broken check must not let a flagged fault through
            flagged = [i for i in shown if _code_findings.flagged_now(i)]
            if not flagged:
                self._log.warning("[evidence_gate] the check of the shown faults failed (%s); none is still flagged", exc)
                return None
            return self._shown_fault_stop(flagged[0], f"FI could not run its test again ({type(exc).__name__}: {exc})")

    async def _shown_fault_check(self, shown: list[dict[str, Any]], record: dict[str, Any] | None = None,
                                 result: Any = None, headline: list[str] | None = None,
                                 state: Any = None) -> dict[str, Any] | None:
        verdicts: list[tuple[dict[str, Any], str]] = []
        errored = False
        declared = (_code_review.declared_names(state.get("design"), self._protocol_block(state),
                                                _analysis_frame.read_record(self.quest_root))
                    if isinstance(state, dict) else None)
        reach = self._shown_fault_reach(headline or [], state)
        for issue in shown:
            rel = str(issue.get("test") or "")
            saved = ((record or {}).get("issues") or {}).get(str(issue.get("id")))
            if not (rel and (self.quest_root / rel).is_file()):
                if isinstance(saved, dict):  # nothing shows it fixed either: the paper says so (core/paper_facts.py)
                    saved["gate_error"] = {"kind": "missing", "detail": f"its test is no longer on disk ({rel or 'none'})"}
                    errored = True
                # Only a failing assertion on the code that ran shows the fault (core/code_findings.py): no test, no stop.
                self._log.warning("[evidence_gate] the test that showed %s is no longer on disk (%s): nothing shows the "
                                  "fault now, and it does not stop the quest", issue.get("id"), rel or "no test recorded")
                continue
            runs = [await self._rerun_finding_test(rel)]
            if runs[0] is None or runs[0].status != "pass":
                runs.append(await self._rerun_finding_test(rel))  # a test that does not pass is run once more
            # a pass in either run clears it (a flaky test stops nothing); else a failing assertion in either run stands
            # (an error in the other run does not hide it); else an error; else no answer at all
            given = [r for r in runs if r is not None]
            run = next((r for status in ("pass", "fail", "error") for r in given if r.status == status), None)
            if run is None:
                self._log.warning("[evidence_gate] the test of %s gave no answer on the code that ran (it ran past its time "
                                  "or ended the process): it shows nothing and does not stop the quest", issue.get("id"))
                if isinstance(saved, dict):
                    saved["gate_error"] = {"kind": "no_answer", "detail": "it gave no answer: it ran past its time or ended "
                                                                          "the process"}
                    errored = True
                continue
            if run.status == "pass":
                self._log.info("[evidence_gate] the test that showed %s passes on the code that ran", issue.get("id"))
            elif run.status == "fail":
                verdicts.append((issue, "its test still fails on it"))
            else:
                # An error shows nothing, wherever it was raised (core/finding_repair.py): recorded, never a stop and never
                # a crash of the study's run (a test calling the code on made-up data says something about the test).
                if record and isinstance(saved, dict):
                    # ... but it is asked for once more, as an assertion about values
                    again = await self._ask_finding_test_again(saved, run, record, state)
                    if again is not None and again.status == "fail":
                        verdicts.append((issue, "a test written again for it, asserting on the values, fails on it"))
                        continue
                    if again is not None and again.status == "pass":
                        continue
                if isinstance(saved, dict):
                    saved["gate_error"] = {"kind": run.kind, "place": run.place, "at": run.at,
                                           "detail": _one_line(run.detail or "", 300)}
                    errored = True
                self._log.warning("[evidence_gate] the test of %s errors on the code that ran (%s): an error shows "
                                  "nothing, so it does not stop the quest", issue.get("id"), _one_line(run.detail or "", 160))
        if errored and record:
            _code_review.save(self.quest_root, record)
        # Whether a fault changes a reported number is shown by running the study with a repair of it, not by reasoning about
        # the test that failed (core/rerun_check.py).
        reruns = await self._rerun_shown(verdicts, record, state) if verdicts and record is not None else {}
        if not verdicts:
            return None
        return self._shown_fault_outcome(verdicts, record or {}, result, headline or [], declared, reach=reach, reruns=reruns)

    async def _rerun_shown(self, verdicts: list[tuple[dict[str, Any], str]], record: dict[str, Any],
                           state: Any, strict: bool = False) -> dict[str, Any]:
        """For each fault a failing test showed and nothing showed gone: what running the study with a repair of it says
        (:class:`core.rerun_check.Verdict`), by finding id. An answer kept for this version of the code is not asked again."""
        out: dict[str, Any] = {}
        version = _code_review.version(self.quest_root / "code")
        done: dict[str, Any] = {}  # the same repair of two findings is run once
        allowed = _rerun_check.time_allowed(self.quest_root, self.config.execution.timeout_s)
        budget = [_rerun_check.TOTAL_FACTOR * (allowed or 0.0)]  # all the runs of one check together
        runs = 0
        fields = _rerun_check.Verdict.__dataclass_fields__
        from . import reuse_key as _reuse_key

        for issue, _how in verdicts:
            fid = str(issue.get("id"))
            saved = (record.get("issues") or {}).get(fid)
            kept = saved.get("rerun") if isinstance(saved, dict) else None
            disputed = sorted(str(x) for x in issue.get("disputed") or []) if strict else []
            # The key of the answer (core/reuse_key.py): every file of the study, the finding's own test, the repairs kept for it
            # and how it was asked. A strict answer (the disputed quantities compared key by key) also stands for an ordinary
            # one, never the other way round. Only a measured answer is reused: "could not" is asked again.
            repairs = [p for _n, p in await asyncio.to_thread(_code_project.refused_patches, self.quest_root, fid, log=self._log)]
            test_rel = str(issue.get("test") or "")
            # also what the comparison is made against: the plan's protocol, the time allowed, the results the study delivered, and
            # what the finding says
            results = self.quest_root / "data" / "results"
            extra = {"protocol": _reuse_key.digest_text(json.dumps(_frozen.protocol_of(self.quest_root) or {}, sort_keys=True, default=str)),
                     "allowed": int(allowed or 0),
                     "delivered": {p.relative_to(self.quest_root).as_posix(): _reuse_key.digest(p)
                                   for folder in (results, self.quest_root / "raw") if folder.is_dir()
                                   for p in sorted(folder.glob("*.json"))},
                     "finding": _reuse_key.digest_text(json.dumps({k: issue.get(k) for k in ("text", "test", "file", "line", "where", "note", "fix", "name")},
                                                                  sort_keys=True, default=str))}
            mine = _reuse_key.rerun(self.quest_root, fid, test=test_rel, repairs=repairs, strict=strict, disputed=disputed, extra=extra)
            plain = mine if not strict else _reuse_key.rerun(self.quest_root, fid, test=test_rel, repairs=repairs, extra=extra)
            if (isinstance(kept, dict) and kept.get("status") in ("changed", "unchanged") and mine.complete
                    and (kept.get("key") == mine.key or (not strict and kept.get("stands_for") == plain.key))):
                out[fid] = _rerun_check.Verdict(**{k: v for k, v in kept.items() if k in fields})
                continue
            if runs >= self.MAX_RERUNS:
                out[fid] = _rerun_check.Verdict("could_not", "the study was already run again for other findings in this check")
            else:
                runs += 1
                try:
                    out[fid] = await self._rerun_one(issue, record, state, done, budget, strict)
                except Exception as exc:  # noqa: BLE001 - a rerun that fails is "could not rerun", never a stop by itself
                    self._log.warning("[evidence_gate] the study could not be run again for %s (%s: %s)", fid,
                                      type(exc).__name__, exc)
                    out[fid] = _rerun_check.Verdict("could_not", f"{type(exc).__name__}: {exc}"[:200])
            if isinstance(saved, dict) and out[fid].mode != "not_asked":  # a request not made is asked again next time
                saved["rerun"] = {**out[fid].record(), "version": version, "strict": strict,
                                  "disputed": sorted(str(x) for x in issue.get("disputed") or []) if strict else [],
                                  "key": mine.key, "stands_for": plain.key}
            self._log_rerun(fid, out[fid])
        _code_review.save(self.quest_root, record)
        return out

    def _log_rerun(self, fid: str, v: Any) -> None:
        if v.status == "could_not":
            self._log.warning("[evidence_gate] %s: the study could not be run again with a repair of it (%s)", fid, v.why)
        else:
            self._log.info("[evidence_gate] %s: the study was run again with a repair of it (%s, %.0f s): %s", fid, v.mode,
                           v.seconds, ("reported numbers differ: " + ", ".join(v.metrics)[:160]) if v.status == "changed"
                           else "no reported number differs beyond its uncertainty")

    async def _repair_passes_own_test(self, issue: dict[str, Any], copy_dir: Path) -> tuple[bool, str]:
        """Whether the finding's own test passes on the repaired copy (a repair that does not make it pass is not a repair of
        the fault): ``(passes, what it said otherwise)``."""
        rel = str(issue.get("test") or "")
        if not (rel and (self.quest_root / rel).is_file()):
            return False, "the finding's own test is not on disk, so the repair cannot be checked"
        try:
            _rc, text, _err = await self._run_in_env(["-c", _finding_repair.harness_source(str(copy_dir), [rel])],
                                                     timeout_s=_code_findings.SHOWN_TEST_TIMEOUT_S,
                                                     env=self._finding_test_env(stand_ins=False))
        except Exception as exc:  # noqa: BLE001 - a test that gave no answer shows nothing
            return False, f"its test gave no answer on the repair ({type(exc).__name__})"
        if _finding_repair.MARK not in (text or ""):
            return False, "its test gave no answer on the repair"
        run = _finding_repair.parse_runs(text, [rel]).get(rel)
        if run is not None and run.status == "pass":
            return True, ""
        return False, "its test still " + ("fails" if run is not None and run.status == "fail" else "does not pass") + (
            ": " + _one_line(run.detail or "", 160) if run is not None and run.detail else "")

    async def _rerun_one(self, issue: dict[str, Any], record: dict[str, Any], state: Any, done: dict[str, Any],
                         budget: list[float] | None = None, strict: bool = False) -> Any:
        """Run the study on a copy of the code with a repair of one shown fault, as :func:`core.rerun_check.judge`. The repair is
        the last one FI tried for the finding that makes its own test pass, even one its own rules refused (kept in ``code/``'s
        history); a finding with none that fits the code as it is is asked for one repair, with the same prompt as the
        code-review loop (counted in the record as ``rerun_calls``). A repair that does not make the finding's test pass is
        never run: the answer is "could not rerun". ``strict``: the finding is a reader's claim about the quantities in
        ``issue["disputed"]``: the answer is "unchanged" only when those numbers were compared, and a repair that leaves a
        delivered number unwritten is "could not rerun"."""
        code = self.quest_root / "code"
        if str(getattr(self.config.execution, "sandbox", "venv")) == "docker":
            # the copy would run in a container that has none of the quest's data: no rerun, and no request spent on one
            return _rerun_check.Verdict("could_not", "the study runs in Docker, where a copy of the code has no data to run on")
        protocol = _frozen.protocol_of(self.quest_root) or (self._protocol_block(state) if isinstance(state, dict) else None)
        allowed = _rerun_check.time_allowed(self.quest_root, self.config.execution.timeout_s)
        if allowed is None:
            return _rerun_check.Verdict("could_not", "FI has no record of how long the study took, so a rerun cannot be bounded")
        fid = str(issue.get("id"))
        work = Path(tempfile.mkdtemp(prefix="fi_repair_"))
        try:
            source = ""
            copy_dir: Path | None = None
            problem = ""
            for number, patch in await asyncio.to_thread(_code_project.refused_patches, self.quest_root, fid, log=self._log):
                trial = await asyncio.to_thread(_rerun_check.copy_code, self.quest_root, work / f"p{number}")
                if await asyncio.to_thread(_code_project.apply_patch, trial, patch):
                    ok, said = await self._repair_passes_own_test(issue, trial)
                    if ok:
                        copy_dir, source = trial, f"its last repair FI tried (try {number})"
                        break
                    problem = said
                await asyncio.to_thread(shutil.rmtree, trial.parent, True)
            env: dict[str, str] = {**scrub_secrets()}
            from .example_inputs import ENV_VAR as _INPUT_ENV, examples_dir, list_inputs

            if list_inputs(self.quest_root):
                env[_INPUT_ENV] = str(examples_dir(self.quest_root))
            python = self.executor.python_path(self.quest_root)
            verdict = _rerun_check.Verdict("could_not", problem or "no repair of it could be written")
            for _round in range(2):
                if copy_dir is None:
                    # nothing kept fits the code as it is and passes its test, or the repair did not run: one repair is asked
                    # for (again), told what went wrong with the last one
                    rel = _finding_repair.target_file(issue, code)
                    if not rel or not (code / rel).is_file():
                        return _rerun_check.Verdict("could_not", "the file it concerns is not in the code")
                    if int(record.get("rerun_calls") or 0) >= _code_review.MAX_FINDING_CALLS:
                        return _rerun_check.Verdict("could_not", problem or "the requests allowed for repairs were spent",
                                                    crashed=bool(problem))
                    record["rerun_calls"] = int(record.get("rerun_calls") or 0) + 1
                    _code_review.save(self.quest_root, record)
                    trial = await asyncio.to_thread(_rerun_check.copy_code, self.quest_root, work / f"a{_round}")
                    try:
                        tried = await self._ask_repair_on_copy(issue, rel, trial, protocol, state, problem)
                    except _finding_repair.NotAsked as exc:
                        # a failed call is no answer: the request is given back and nothing is kept for this code
                        record["rerun_calls"] = max(0, int(record.get("rerun_calls") or 0) - 1)
                        _code_review.save(self.quest_root, record)
                        return _rerun_check.Verdict("could_not", f"not asked: {exc}"[:200], mode="not_asked")
                    if not tried:  # a reply that could not be placed was already asked for once more inside the request
                        return _rerun_check.Verdict("could_not", "no repair of it could be written")
                    for target, text in tried.items():
                        (trial / target).write_text(text, encoding="utf-8")
                    ok, said = await self._repair_passes_own_test(issue, trial)
                    if not ok:
                        problem = ("FI ran the finding's own test on your repair and " + said
                                   + ". The repair must make that test pass.")
                        verdict = _rerun_check.Verdict("could_not", "the repair does not make the finding's own test pass: " + said)
                        continue
                    copy_dir = trial
                    source = "a repair asked for once more" if _round == 0 else "a repair asked for again after the first did not work"
                key = _code_review.version(copy_dir)
                if key in done:
                    return done[key]
                if budget is not None and min(allowed, budget[0]) < _rerun_check.FLOOR_S:  # no time left: not even the small run
                    return _rerun_check.Verdict("could_not", "the time FI allows for running the study again in one check is spent")
                began = time.monotonic()
                # The run contract is held to before the long run: FI's small run of the repaired simulation (seconds) refuses
                # a form the study's own run would refuse, and its reason goes back to the model as a crash's does.
                refused = await self._repair_small_run_refusal(copy_dir, protocol, env)
                if budget is not None:
                    budget[0] -= time.monotonic() - began  # the small run is part of the time of the check
                    began = time.monotonic()
                left = allowed if budget is None else min(allowed, budget[0])
                if left < _rerun_check.FLOOR_S and not refused:
                    return _rerun_check.Verdict("could_not", "the time FI allows for running the study again in one check is spent")
                if refused:
                    verdict = _rerun_check.Verdict("could_not", "the repaired simulation does not run: " + refused,
                                                   source=source, files=sorted(_rerun_check.changed_files(self.quest_root, copy_dir)),
                                                   crashed=True)
                else:
                    verdict = await _rerun_check.judge(self.executor, python, self.quest_root, copy_dir, protocol, left, env,
                                                       source=source, results_root=self.quest_root,
                                                       disputed=list(issue.get("disputed") or []) if strict else None,
                                                       finding=str(issue.get("id") or ""))
                if budget is not None:
                    budget[0] -= time.monotonic() - began
                if verdict.status == "could_not" and verdict.crashed and _round == 0:
                    problem = ("FI ran the study with your repair and it did not run to the end: " + verdict.why
                               + ". The repair may use only the results the plan declares.")
                    copy_dir = None
                    continue
                done[key] = verdict
                return verdict
            return verdict
        finally:
            await asyncio.to_thread(shutil.rmtree, work, True)

    async def _repair_small_run_refusal(self, copy_dir: Path, protocol: Any, env: dict[str, str]) -> str:
        """Why FI's small run (:func:`core.code_checks.small_run_refusal`) refuses the repaired code in ``copy_dir``, or ``""``
        when it runs, cannot be judged, or the repair changed only the analysis (which the simulation's contract does not read)."""
        from . import code_checks as _code_checks

        files = sorted(await asyncio.to_thread(_rerun_check.changed_files, self.quest_root, copy_dir))
        if all(Path(f).name in _rerun_check.ANALYSIS_FILES and "/" not in f for f in files):
            return ""
        async def refusal(root: Path) -> str:
            # The ``code/`` of ``root`` is run in a folder of its own: the small run writes its files (``.fi/small_run``,
            # FI's harness) beside the code it runs, and the study's run.py takes a folder that holds ``.fi/`` for a quest
            # and writes to ``../run_output``. Made beside the repaired copy, the study's own rerun of that copy would write
            # its results outside it, and a working repair would be read as one that wrote no results.
            scratch = Path(tempfile.mkdtemp(prefix="fi_small_"))
            try:
                await asyncio.to_thread(_rerun_check.copy_code, root, scratch)
                return await _code_checks.small_run_refusal(
                    scratch, protocol if isinstance(protocol, dict) else None, executor=self.executor,
                    python=self.executor.python_path(self.quest_root), env=dict(env),
                    timeout_s=int(self.config.execution.timeout_s))
            finally:
                await asyncio.to_thread(shutil.rmtree, scratch, True)

        try:
            refused = await refusal(copy_dir.parent)
            if not refused:
                return ""
            # The copy holds only ``code/``: a study that reads a file of the quest outside it fails there whatever the repair.
            # A refusal counts only for what the repair changed: the code as it is must pass the same small run in the same place.
            if await refusal(self.quest_root):
                return ""
            return refused
        except Exception as exc:  # noqa: BLE001 -- a check that cannot be made refuses nothing: the study's own run decides
            self._log.warning("[rerun] the small run of the repair could not be made (%r); the study's own run decides", exc)
            return ""

    async def _ask_repair_on_copy(self, issue: dict[str, Any], rel: str, copy_dir: Path, protocol: Any,
                                  state: Any, problem: str = "") -> dict[str, str]:
        """One repair of ``issue`` from the code model, written on ``copy_dir`` only: the files the model gave."""
        values = _pairs.plan_values(protocol if isinstance(protocol, dict) else None)
        package = self._package_in_use(state) or ""
        ids = _oracle.generating_equations(protocol if isinstance(protocol, dict) else None)
        scratch = copy.deepcopy(issue)
        scratch.update(calls={"test": 0, "repair": 0, "other": 0})
        fid = str(issue.get("id"))

        async def ask(prompt: str, who: str, kind: str) -> str:
            return await self._chat(prompt, node="implement.finding_repair")

        async def passes(paths: list[str]) -> dict[str, Any]:
            return {p: _finding_repair.TestRun("pass") for p in paths}  # the copy is not held to the finding's own test

        async def nothing() -> dict[str, str]:
            return {}

        env = _finding_repair.Env(
            root=self.quest_root, code=copy_dir, record={"finding_calls": 0, "issues": {fid: scratch}}, save=lambda: None,
            ask=ask, run_tests=passes, snapshot=nothing, static=lambda: {},
            keeps=lambda old, new, r: _code_review.repair_problem(old, new, r, equations=ids, package=package),
            prompts=self._prompts, topic=_pairs.scrub(self._topic_block(), values) or "(not stated)",
            equations=_code_review.equations_view(protocol if isinstance(protocol, dict) else None, values),
            scrub=lambda text: _pairs.scrub(text, values), log=self._log, package=package)
        return await _finding_repair.repair_for_rerun(env, scratch, rel, problem)

    async def _demonstrate_reader_finding(self, issue: dict[str, Any], record: dict[str, Any], state: Any) -> bool:
        """Ask, as for any finding (:func:`core.finding_repair.demonstrate`, within its bound of requests), for a test that
        fails on the delivered code because of the difference the reader claims. ``True`` when one fails by its assertion on
        two runs (the test is kept in ``issue``); else the finding is not shown."""
        code = self.quest_root / "code"
        protocol = _frozen.protocol_of(self.quest_root) or (self._protocol_block(state) if isinstance(state, dict) else None)
        values = _pairs.plan_values(protocol if isinstance(protocol, dict) else None)

        async def ask(prompt: str, who: str, kind: str) -> str:
            return await self._chat(prompt, node="implement.finding_test")

        async def run_tests(paths: list[str]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for path in paths:
                _rc, text, _err = await self._run_in_env(["-c", _finding_repair.harness_source("code", [path])], timeout_s=180,
                                                         env=self._finding_test_env())
                out.update(_finding_repair.parse_runs(text, [path]))
            return out

        async def nothing() -> dict[str, str]:
            return {}

        env = _finding_repair.Env(
            root=self.quest_root, code=code, record=record, save=lambda: _code_review.save(self.quest_root, record),
            ask=ask, run_tests=run_tests, snapshot=nothing, static=lambda: {}, keeps=lambda old, new, rel: "",
            prompts=self._prompts, topic=_pairs.scrub(self._topic_block(), values) or "(not stated)",
            equations=_code_review.equations_view(protocol if isinstance(protocol, dict) else None, values),
            scrub=lambda text: _pairs.scrub(text, values), log=self._log, package=self._package_in_use(state) or "")
        rel = _finding_repair.target_file(issue, code)
        if not rel or not (code / rel).is_file():
            issue.update(demonstrated=False, outcome=_code_findings.NOT_DEMONSTRATED, reason="the file it concerns is not in the code")
            return False
        return await _finding_repair.demonstrate(env, issue, rel)

    async def _measure_reader_findings(self, record: dict[str, Any], state: Any, reach: "_shown_reach.Reach | None",
                                       headline: list[str] | None = None) -> dict[str, Any] | None:
        """The one place where a second model's word that a quantity is computed differently from the plan is measured before
        it keeps a result out of the paper, on the same footing as a shown fault: a test is asked for that fails on the
        delivered code because of the claimed difference (:meth:`_demonstrate_reader_finding`), the repair is held to that
        test, and the study is run again with it (:meth:`_rerun_shown`, :mod:`core.rerun_check`). A quantity the results check
        took out on that word alone (``needs/RESULT_CHECK.json``, kind ``code_reader``) is

        * given back, the method difference said for the paper, when the test shows the difference, the repair passes it and no
          reported number moved beyond its allowance;
        * kept out when the test is shown and a number moved (with what is worked out from it, as a shown fault); the paper
          stops when the main result is among them, by FI's records or, without them, by the plan's headline names;
        * given back with the claim disclosed when no test FI could write shows the difference (not shown: a word never
          withholds a result);
        * kept out as before when no rerun could be made, the record saying that nothing measured it.

        Returns the gate's stop, else ``None``."""
        from . import result_checks as _result_checks

        path = self.quest_root / "needs" / _result_checks.RECORD
        rc = _result_checks._read_json(path)
        checks = [c for c in rc.get("checks") or [] if isinstance(c, dict) and c.get("kind") == "code_reader" and c.get("quantity")
                  and not (isinstance(c.get("measured"), dict) and c["measured"].get("version") == _code_review.version(self.quest_root / "code")
                           and c["measured"].get("status") in ("changed", "could_not"))]
        if not checks:
            return None
        differing = {q["name"]: q for q in _code_review.differing_quantities(self.quest_root, self.quest_root / "code")}
        issues = record.get("issues") if isinstance(record.get("issues"), dict) else {}
        pairs: list[tuple[dict[str, Any], str]] = []
        by_fid: dict[str, list[dict[str, Any]]] = {}
        shown: dict[str, bool] = {}
        kept: list[dict[str, Any]] = []
        not_asked: set[str] = set()  # findings whose test request could not be made in this pass
        version = _code_review.version(self.quest_root / "code")
        for c in checks:
            fid = str((differing.get(str(c["quantity"])) or {}).get("fid") or "")
            saved = issues.get(fid)
            if not isinstance(saved, dict):
                c["measured"] = {"status": "could_not", "why": "the reader's finding is no longer in FI's record, so no test or repair of it can be made"}
                continue
            saved.setdefault("id", fid)
            if fid not in shown:
                has_test = bool(saved.get("test") and (self.quest_root / str(saved["test"])).is_file())
                try:
                    shown[fid] = (saved.get("demonstrated") is True and has_test) or await self._demonstrate_reader_finding(saved, record, state)
                except _finding_repair.NotAsked as exc:  # no answer is no finding of "not shown": undecided, asked again next time
                    shown[fid] = False
                    not_asked.add(fid)
                    for x in checks:
                        if str((differing.get(str(x["quantity"])) or {}).get("fid") or "") == fid:
                            x["measured"] = {"status": "could_not", "why": f"not asked: {exc}"[:200], "not_asked": True}
                _code_review.save(self.quest_root, record)
                if shown[fid]:
                    pairs.append(({**saved, "disputed": [str(x["quantity"]) for x in checks if
                                   str((differing.get(str(x["quantity"])) or {}).get("fid") or "") == fid]},
                                  "a second model found it computed differently from the plan"))
            by_fid.setdefault(fid, []).append(c)
        reruns = await self._rerun_shown(pairs, record, state, strict=True) if pairs else {}
        stop: dict[str, Any] | None = None
        result = state.get("result_json") if isinstance(state, dict) else None
        main = {_shown_reach._norm(h) for h in (headline or [])}
        for fid, group in by_fid.items():
            v = reruns.get(fid)
            for c in group:
                if not shown.get(fid):
                    if "measured" not in c or (c["measured"].get("not_asked") and fid not in not_asked):
                        c["measured"] = {"status": "not_shown", "why": "no test FI could write fails on the delivered code because of it"}
                        kept.append(c)
                        self._log.warning("[evidence_gate] no test shows that `%s` is computed differently from the plan: a word alone "
                                          "keeps no result out, so it stays in the results and the paper says the claim stands "
                                          "unshown", c["quantity"])
                    continue
                c["measured"] = ({"status": v.status, "why": v.why, "metrics": v.metrics, "notes": v.notes, "mode": v.mode,
                                  "version": version} if v is not None else {"status": "could_not", "why": "not run", "version": version})
                if v is not None and v.mode == "not_asked":  # a request not made is no answer for this code: asked again next time
                    c["measured"] = {k: x for k, x in c["measured"].items() if k != "version"} | {"not_asked": True}
                if v is not None and v.status == "unchanged":
                    kept.append(c)
                elif v is not None and v.status == "changed":
                    why = ""
                    placed = reach.effect_of_changes(v.metrics) if reach is not None else None
                    if placed is not None and (v.unplaced or "failed trials" in v.metrics):
                        # a changed number that belongs to no result FI can name is not cleared by the rerun: the finding is also
                        # placed by what it reaches, as the ordinary shown fault is
                        other = reach.effect(record, issues[fid], result)  # type: ignore[union-attr]
                        placed = None if other is None else _shown_reach.union(placed, other)
                    if placed is not None:
                        keys = sorted({*(c.get("keys") or []), *reach.printed_keys(placed.left_out, result), *placed.extra_keys})  # type: ignore[union-attr]
                        why = placed.stop
                    else:
                        # no records to place it by: every changed number is left out, and the plan's headline names decide the stop
                        keys = sorted({*(c.get("keys") or []), *(m for m in v.metrics if m != "failed trials")})
                        if main and any(_shown_reach._norm(m) in main for m in v.metrics):
                            why = "the main result is among them (the plan's headline: " + ", ".join(headline or [])[:120] + ")"
                    drawn = reach.figures_drawing(list(placed.left_out) if placed is not None else [], keys) if reach is not None else []
                    if keys or drawn:
                        how_moved = "the study run again with a repair of it reports other numbers for " + ", ".join(v.metrics)[:160]
                        self._leave_out_shown_faults([(c["quantity"], keys, how_moved)] if keys else [], [], result,
                                                     figures=[(f, how_moved) for f in drawn])
                    if why and stop is None:
                        text = (f"a second model found `{c['quantity']}` computed differently from the plan, a test shows it, and the "
                                f"study run again with a repair of it reports other numbers for {', '.join(v.metrics)[:160]}: {why}")
                        self._log.warning("[evidence_gate] %s; stopping without a paper", text)
                        stop = {"verdict": "insufficient", "rationale": text, "gaps": [text], "stuck": True,
                                "stuck_reason": "shown_fault_open"}
                elif v is None or v.status == "could_not":
                    self._log.warning("[evidence_gate] `%s` stays out of the results on the reader's word: the study could not be "
                                      "run again with a repair of it (%s); the record says nothing measured it", c["quantity"],
                                      (v.why if v is not None else "not run")[:160])
        if kept:
            self._log.warning("[evidence_gate] %s stay in the results: the reader's claim is not shown to move a number (the paper says "
                              "the method differs from the plan's definition, or that the claim stands unshown)",
                              ", ".join(f"`{c['quantity']}`" for c in kept))
        self._close_reader_claims(path, rc, checks, kept, result, reach)
        return stop

    def _close_reader_claims(self, path: Path, rc: dict[str, Any], checks: list[dict[str, Any]], kept: list[dict[str, Any]],
                             result: Any, reach: "_shown_reach.Reach | None") -> None:
        """Write what the gate decided about the second model's claims into ``needs/RESULT_CHECK.json``. A claim a rerun could not
        measure stays out (the execute step recorded it without taking the keys out, so only a measurement or this decides), with
        the figures that draw it, and the writer is told so instead of the execute step's "stays in the results for now"; a claim
        that is kept returns its keys (:func:`core.result_checks._keep_measured`)."""
        from . import result_checks as _result_checks

        rc = _result_checks._read_json(path) or rc  # _leave_out_shown_faults may have added what a changed number takes out
        for c in rc.get("checks") or []:
            m = next((x for x in checks if x.get("quantity") == c.get("quantity") and x.get("kind") == c.get("kind")), None)
            if m is not None and m.get("measured"):
                c["measured"] = m["measured"]
        unmeasured = [c for c in rc.get("checks") or [] if c.get("kind") == "code_reader" and isinstance(c.get("measured"), dict)
                      and c["measured"].get("status") == "could_not"]
        # The execute step's words for a claim still in the results ("stays in the results for now") are gone once the gate has
        # decided that its numbers stay out; the left-out words that follow are the writer's only instruction for them.
        out = {str(c["quantity"]) for c in rc.get("checks") or [] if c.get("kind") == "code_reader" and isinstance(c.get("measured"), dict)
               and c["measured"].get("status") in ("changed", "could_not")}
        if out:
            rc["say"] = "\n\n".join(p for p in str(rc.get("say") or "").split("\n\n")
                                    if p.strip() and not any(f"found that `{n}` is " in p for n in out))
        out_keys = {str(k) for c in unmeasured for k in c.get("keys") or []} - {str(k) for k in rc.get("removed_quantities") or []}
        if out_keys:
            rc["removed_quantities"] = sorted({*(str(k) for k in rc.get("removed_quantities") or []), *out_keys})
            rc["removed_values"] = {**(rc.get("removed_values") or {}),
                                    **_result_checks._values_left_out(result, out_keys, self._derived_record())}
            drawn = [f for f in (reach.figures_drawing([], sorted(out_keys)) if reach is not None else [])
                     if f not in (rc.get("removed_figures") or [])]
            if drawn:
                self._drop_figures(drawn, {f: _result_checks.SHOWN_FAULT_FIGURE_WHY for f in drawn})
                rc["removed_figures"] = sorted({*(rc.get("removed_figures") or []), *drawn})
            paragraphs = [p for p in str(rc.get("say") or "").split("\n\n") if p.strip()]
            for c in unmeasured:
                if not ({str(k) for k in c.get("keys") or []} & out_keys):
                    continue
                q = c["quantity"]
                taken = ", ".join("`" + str(k) + "`" for k in (c.get("keys") or [])[:6])
                paragraphs.append(
                    f"A second model found that `{q}` is computed differently from the plan ({str(c.get('note') or '').rstrip('.')}). "
                    f"FI could not run the study again with a repair of it to find out whether the number is wrong, so it took {taken} "
                    f"out of the results. Do NOT report, describe or quote any value for `{q}`, and do not call another number by the "
                    f"plan's name for it. Say plainly, in the limitations, that `{q}` is left out of this paper because the code may "
                    "not compute it as the plan defines it and FI could not establish whether it does.")
            rc["say"] = "\n\n".join(paragraphs)
        _result_checks._keep_measured(rc, [c for c in rc.get("checks") or [] if any(
            c.get("quantity") == k.get("quantity") and c.get("kind") == k.get("kind") for k in kept)])
        path.write_text(json.dumps(rc, indent=1), encoding="utf-8")  # a leave-out that is not recorded is not enforced: it raises

    def _leave_out_undecided_claims(self, state: Any, why: str) -> dict[str, Any] | None:
        """Settling the second model's claims failed: every key of every claim still undecided is left out (with the figures that
        draw it, when those can be found), the writer is told not to report it and the quality report says FI could not decide
        it. Returns the gate's stop when even that cannot be recorded, else ``None``."""
        from . import result_checks as _result_checks

        path = self.quest_root / "needs" / _result_checks.RECORD
        try:
            rc = _result_checks._read_json(path)
            undecided = [c for c in rc.get("checks") or [] if isinstance(c, dict) and c.get("kind") == "code_reader"
                         and c.get("quantity") and not isinstance(c.get("measured"), dict)]
            if not undecided:
                return None
            keys = {str(k) for c in undecided for k in c.get("keys") or []}
            result = state.get("result_json") if isinstance(state, dict) else None
            rc["removed_quantities"] = sorted({*(str(k) for k in rc.get("removed_quantities") or []), *keys})
            try:
                rc["removed_values"] = {**(rc.get("removed_values") or {}), **_result_checks._values_left_out(result, keys, self._derived_record())}
            except Exception:  # noqa: BLE001 - the values shown in the record are an extra
                pass
            paragraphs = [p for p in str(rc.get("say") or "").split("\n\n") if p.strip()
                          and not any(f"found that `{c['quantity']}` is " in p for c in undecided)]
            for c in undecided:
                c["measured"] = {"status": "could_not", "why": "FI could not decide this claim: " + why}
                c["note"] = str(c.get("note") or "").rstrip(".") + "; FI could not decide whether the claim holds, so the quantity is left out"
                paragraphs.append(
                    f"A second model found that `{c['quantity']}` is computed differently from the plan. FI could not decide whether "
                    f"that holds, so it took {', '.join('`' + str(k) + '`' for k in (c.get('keys') or [])[:6])} out of the results. Do NOT "
                    f"report, describe or quote any value for `{c['quantity']}`, and do not call another number by the plan's name for it. "
                    f"Say plainly, in the limitations, that `{c['quantity']}` is left out of this paper because FI could not decide "
                    "whether the code computes it as the plan defines it.")
            rc["say"] = "\n\n".join(paragraphs)
            try:
                names = _code_review.headline_names(state.get("design") if isinstance(state, dict) else None, self._protocol_block(state))
                reach = self._shown_fault_reach(names, state)
                drawn = [f for f in (reach.figures_drawing([], sorted(keys)) if reach is not None else [])
                         if f not in (rc.get("removed_figures") or [])]
                if drawn:
                    self._drop_figures(drawn, {f: _result_checks.SHOWN_FAULT_FIGURE_WHY for f in drawn})
                    rc["removed_figures"] = sorted({*(rc.get("removed_figures") or []), *drawn})
            except Exception:  # noqa: BLE001 - the figures are an extra source
                pass
            path.write_text(json.dumps(rc, indent=1), encoding="utf-8")
            return None
        except Exception as exc:  # noqa: BLE001
            text = ("a second model's claim about how a result is computed could not be settled, and FI could not record that the "
                    f"result is left out ({type(exc).__name__}), so the quest stops here instead of writing a paper that may report it")
            self._log.warning("[evidence_gate] %s", text)
            return {"verdict": "insufficient", "rationale": text, "gaps": [text], "stuck": True, "stuck_reason": "shown_fault_open"}

    def _settle_undecided_reader_claims(self, state: Any) -> None:
        """The gate could not run to the end (it raised): the second model's claims it had not decided are decided as if no rerun
        could be made, so that whether a result is left out never depends on whether the gate's own code crashed. A claim whose
        test failed on the code that produced the results stays out and is disclosed; a claim no test shows is kept and disclosed
        (a word alone never withholds a result)."""
        from . import result_checks as _result_checks

        path = self.quest_root / "needs" / _result_checks.RECORD
        rc = _result_checks._read_json(path)
        checks = [c for c in rc.get("checks") or [] if isinstance(c, dict) and c.get("kind") == "code_reader" and c.get("quantity")
                  and not isinstance(c.get("measured"), dict)]
        if not checks:
            return
        differing = {q["name"]: q for q in _code_review.differing_quantities(self.quest_root, self.quest_root / "code")}  # raises: fails closed
        issues = _code_review.load(self.quest_root).get("issues")
        issues = issues if isinstance(issues, dict) else {}
        kept: list[dict[str, Any]] = []
        for c in checks:
            saved = issues.get(str((differing.get(str(c["quantity"])) or {}).get("fid") or ""))
            shown = isinstance(saved, dict) and saved.get("demonstrated") is True and bool(
                saved.get("test") and (self.quest_root / str(saved["test"])).is_file())
            if shown:
                c["measured"] = {"status": "could_not", "why": "the evidence check stopped before it could run the study again"}
            else:
                c["measured"] = {"status": "not_shown", "why": "no test FI could write fails on the delivered code because of it"}
                kept.append(c)
        reach = None
        try:
            names = _code_review.headline_names(state.get("design") if isinstance(state, dict) else None, self._protocol_block(state))
            reach = self._shown_fault_reach(names, state)
        except Exception:  # noqa: BLE001 - the figures are an extra source
            reach = None
        self._close_reader_claims(path, rc, checks, kept, state.get("result_json") if isinstance(state, dict) else None, reach)

    def _shown_fault_reach(self, headline: list[str], state: Any) -> "_shown_reach.Reach | None":
        """FI's records of the quest, read for the gate (:mod:`core.shown_reach`); ``None`` when they cannot be read, and
        the gate then judges each finding by the results it names."""
        try:
            design = state.get("design") if isinstance(state, dict) else None
            protocol = _frozen.protocol_of(self.quest_root) or (self._protocol_block(state) if isinstance(state, dict) else None)
            declared = _code_review.write_declared_results(self.quest_root, design, protocol)  # the paper's side reads it
            return _shown_reach.Reach(self.quest_root, protocol, headline, declared)
        except Exception as exc:  # noqa: BLE001 - the records are an extra source: without them the gate judges as before
            self._log.warning("[evidence_gate] FI's records could not be read for the shown faults (%r)", exc)
            return None

    def _shown_fault_outcome(self, verdicts: list[tuple[dict[str, Any], str]], record: dict[str, Any],
                             result: Any, headline: list[str] | None = None,
                             declared: list[str] | None = None, reach: "_shown_reach.Reach | None" = None,
                             reruns: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """What the faults a failing test showed, and nothing showed gone, do to the quest, from the records alone. A fault
        about the study's main quantity (the plan's first dependent variable or its precision metric) stops it with no
        paper: there is nothing to report. A plan quantity the code does not compute (and the results do not hold under
        another name) is a part not done, which the paper states. A fault in another quantity takes that quantity out of
        the results the paper may use, and the paper states it. A fault whose quantity the results do not hold, or whose
        removal leaves no result, stops the quest: nothing can be left out, and a wrong number may stand. So does one
        whose values the results also hold under keys that stay (FI could not tell them apart in the paper), and one that
        may be about the main quantity by its names or its keys (:func:`core.code_review.about_headline`).

        With ``reach`` (FI's records of the quest, :mod:`core.shown_reach`), a fault that test places by the function it
        calls (a result worked out afterwards, a figure, a helper) is judged from those
        records first: it stops only when the main result is among what it changes or nothing would be left; the rest
        is left out, not done or kept in the quality report. A finding the records cannot place is judged as above."""
        from . import result_checks as _result_checks

        stops: list[tuple[dict[str, Any], str]] = []
        left_out: list[tuple[dict[str, Any], str, list[str], str | None]] = []
        not_done: list[tuple[dict[str, Any], str, list[str]]] = []
        figures: list[tuple[dict[str, Any], str, list[str]]] = []
        about_plan: list[dict[str, Any]] = []
        untied: list[tuple[dict[str, Any], list[str]]] = []  # a failing test on text FI wrote, about a plan entry FI placed on no result
        changed: dict[str, set[str]] = {}  # per finding: the printed keys the study run again with a repair of it showed changed
        unchanged: list[dict[str, Any]] = []
        moved_checks: dict[str, bool] = {}  # unchanged because the rerun moved only check measures -> its draws changed
        for issue, how in verdicts:
            rerun = (reruns or {}).get(str(issue.get("id")))
            if rerun is not None and rerun.status == "unchanged":
                if self._unchanged_is_shown(issue, record, result, declared, reach):
                    # The study run again with a repair of it reports the same numbers: a difference that changes no reported
                    # number.
                    unchanged.append(issue)
                    continue
                # A finding that a quantity is not computed, or about a figure, is not answered by numbers that did not move:
                # it is placed as it was before a rerun
                rerun = None
            if rerun is not None and rerun.status == "changed" and self._rerun_moved_only_checks(rerun, reach, result):
                if self._unchanged_is_shown(issue, record, result, declared, reach):
                    # The study run again with a repair of it moved only the numbers FI measures its checks with, not one
                    # declared result and no key of the results: no reported number changes, whatever the finding says.
                    unchanged.append(issue)
                    moved_checks[str(issue.get("id"))] = bool(getattr(rerun, "redrawn", False))
                    continue
                rerun = None  # a finding that a quantity is not computed, or about a figure, is placed as before a rerun
            # A fault in a function that makes one declared result, one figure or a helper they use is placed from FI's
            # records (core/shown_reach.py), not from the finding's words.
            rerun_keys: set[str] = set()  # the keys among what is left out whose recorded value the rerun shows changed
            try:
                if rerun is not None and rerun.status == "changed" and reach is not None:
                    # The numbers the rerun showed changed are what is left out, and the main result among them stops the paper.
                    how = how + "; the study run again with a repair of it reports other numbers for " + ", ".join(rerun.metrics)[:160]
                    placed = reach.effect_of_changes(rerun.metrics)
                    # Only these keys were shown to change; what else is left out (results worked out from them, what the
                    # finding's test reaches) is left out without that having been shown.
                    rerun_keys = reach.changed_keys([m for m in (_rerun_check.metric_of(k)[0] for k in rerun.keys) if m])
                    if rerun.unplaced or "failed trials" in rerun.metrics:
                        # A changed number that belongs to no result FI can name has not been cleared by the rerun: the finding
                        # is also placed as it was before a rerun (what its test reaches), in addition to the results that moved
                        other = reach.effect(record, issue, result)
                        placed = None if other is None else _shown_reach.union(placed, other)
                    if placed is not None and not (placed.stop or placed.left_out or self._in_results(placed.extra_keys, result)):
                        # the numbers that moved leave out no result and no key of the results: never delivered for that; the
                        # finding is judged as before a rerun (what its own names and the results' keys say)
                        placed = None
                    if placed is None:
                        rerun_keys = set()
                elif reach is not None and (not _code_review.missing_quantities(record, issue)
                                            or reach.reads_held_key(issue, result)):
                    # No rerun to say (it could not be made): leave out what the records place, and stop only for an empty paper.
                    keep_rule, reach.headline_stops = reach.headline_stops, rerun is None
                    try:
                        placed = reach.effect(record, issue, result)
                    finally:
                        reach.headline_stops = keep_rule
                    if rerun is not None and placed is not None:
                        if placed.stop and not placed.empty:
                            placed.stop = ""  # results left out for want of a rerun stop the paper only when none is left
                        how = how + "; the study could not be run again with a repair of it (" + rerun.why[:100] + ")"
                else:
                    placed = None
                if placed is not None and not placed.stop:
                    # the main result left out as not computed is as much a paper without it as one that changes it
                    main = {_shown_reach._norm(h) for h in (headline or [])}
                    if any(_shown_reach._norm(n) in main for n in placed.not_done):
                        placed.stop = "the main result is among what the code does not compute (" + ", ".join(placed.not_done)[:120] + ")"
                placed_keys = (sorted({*reach.printed_keys(placed.left_out, result), *placed.extra_keys})
                               if placed is not None and (placed.left_out or placed.extra_keys) else [])
                placed_named = (placed.named or (placed.left_out[0] if placed.left_out else "")) if placed is not None else ""
            except Exception as exc:  # noqa: BLE001 - the records are an extra source: without them the finding is judged as before
                self._log.warning("[evidence_gate] %s could not be placed from FI's records (%r)", issue.get("id"), exc)
                placed, placed_keys, placed_named = None, [], ""
            if placed is not None:
                if placed.stop:
                    stops.append((issue, how + "; " + placed.stop))
                    continue
                if placed.plan_text:
                    about_plan.append(issue)
                    continue
                if placed_keys:
                    left_out.append((issue, how, placed_keys, placed_named or None))
                    changed[str(issue.get("id"))] = rerun_keys & set(placed_keys)
                files = [reach.figure_files.get(f) or "" for f in placed.figures]
                if any(files):
                    figures.append((issue, how, [f for f in files if f]))
                names = [*placed.not_done, *(f"figure {f}" for f, file in zip(placed.figures, files) if not file)]
                if names:
                    not_done.append((issue, how, names))
                if not (placed.left_out or placed.figures or placed.not_done):
                    self._log.info("[evidence_gate] %s: %s; it changes no reported result", issue.get("id"), placed.why)
                continue
            # a rerun that could not run (a failed request, a refused repair, no time left) shows nothing moving: only one
            # that moved numbers (even ones it could not place on a result) keeps the finding from the plan-text release
            moved = rerun is not None and (rerun.status == "changed" or bool(getattr(rerun, "keys", None))
                                           or bool(getattr(rerun, "metrics", None)))
            if not moved and (
                    names := self._untied_plan_text_only(record, issue, result, headline or [], reach)):
                untied.append((issue, names))
                continue
            if _code_review.about_headline(record, issue, result, headline or []):
                stops.append((issue, how))
            elif _code_review.missing_quantities(record, issue):
                self._part_not_done_or_stop(stops, not_done, issue, how, record, result, declared, reach)
            elif not (reach is not None and reach.ids) and (keys := _code_review.keys_of_issue(record, issue, result)):
                # Only a quest with no frozen plan (no records of FI's to place a finding by): the finding's own quantity names the
                # keys. A key that holds no number has nothing another key could hold too: leaving it out removes all of it.
                if _result_checks._values_left_out(result, set(keys)) or _code_review.holds_no_number(result, set(keys)):
                    left_out.append((issue, how, keys, None))
                else:
                    stops.append((issue, how + "; the results hold its values under other names too, so FI cannot leave "
                                               "them out"))
            elif _code_review.not_obtained(record, issue, result, declared):
                self._part_not_done_or_stop(stops, not_done, issue, how, record, result, declared, reach)
            else:
                # The records of FI's (the plan's frozen protocol, the trials, the call graph) do not place the finding, and the
                # reader's words are not a record: nothing is left out or kept on them.
                stops.append((issue, how + "; " + _UNPLACED_FINDING))
        if not stops and left_out:
            from . import numeric_oracle as _numeric_oracle
            from . import typed_results as _typed

            gone = {k for _i, _h, keys, _q in left_out for k in keys}
            if not any(True for _ in _numeric_oracle.flatten_numbers(_typed.strip_keys(result, gone), keep_zero=True)):
                stops.append((left_out[0][0], left_out[0][1] + "; leaving out what it changes leaves no result"))
        if stops:
            return self._shown_fault_stop(*stops[0])
        issues = record.get("issues") if isinstance(record.get("issues"), dict) else {}
        for issue in unchanged:
            saved = issues.get(str(issue.get("id")))
            if isinstance(saved, dict):
                saved["changes_result"] = False
                saved["rerun_unchanged"] = True
            if str(issue.get("id")) in moved_checks:
                if isinstance(saved, dict):
                    saved["rerun_moved_checks"] = True
                    saved["rerun_redrawn"] = moved_checks[str(issue.get("id"))]
                self._log.warning("[evidence_gate] %s: its test fails, but the study run again with a repair of it moves only the "
                                  "numbers FI checks the code with, not one result: it changes no reported number and does not "
                                  "stop the quest", issue.get("id"))
                continue
            self._log.warning("[evidence_gate] %s: its test fails, but the study run again with a repair of it reports no other "
                              "number beyond its uncertainty: it changes no reported number and does not stop the quest",
                              issue.get("id"))
        if unchanged and not (left_out or not_done or figures):
            _code_review.save(self.quest_root, record)
        for issue in about_plan:
            # Text FI wrote from the plan differs from what the reader wanted: no computed number is wrong. It stays in the
            # quality report and changes neither the plan nor the paper's results.
            saved = issues.get(str(issue.get("id")))
            if isinstance(saved, dict):
                saved["changes_result"] = False
                saved["about_plan_text"] = True
            self._log.warning("[evidence_gate] %s is about text FI wrote from the plan, not a number the code computed: it "
                              "changes no reported number and does not stop the quest", issue.get("id"))
        if about_plan and not (left_out or not_done or figures):
            _code_review.save(self.quest_root, record)
        for issue, names in untied:
            # The plan lists a dependent variable that FI placed on no declared result, and the failing test reads only text FI
            # wrote from the plan: no computed number is shown wrong. Kept in the quality report only, never in the paper.
            saved = issues.get(str(issue.get("id")))
            if isinstance(saved, dict):
                saved["changes_result"] = False
                saved["untied_plan_item"] = names
            self._log.warning("[evidence_gate] %s is about %s, which FI did not place on any declared result, and its test reads "
                              "only text FI wrote from the plan: it changes no reported number and does not stop the quest",
                              issue.get("id"), ", ".join(names)[:200])
        if untied and not (left_out or not_done or figures):
            _code_review.save(self.quest_root, record)
        for issue, how, keys, _q in left_out:
            self._log.warning("[evidence_gate] %s: %s; FI leaves %s out of the results the paper may use, and the paper says "
                              "so", issue.get("id"), how, ", ".join(keys)[:200])
        for issue, how, files in figures:
            self._log.warning("[evidence_gate] %s: %s; FI takes %s out of the paper's figures, and the paper says so",
                              issue.get("id"), how, ", ".join(files)[:200])
        for issue, how, names in not_done:
            self._log.warning("[evidence_gate] %s: the code does not compute %s (%s); the paper says so, as a part not done",
                              issue.get("id"), ", ".join(names)[:200], how)
        recorded: set[str] = set()
        for issue, how, names, outcome, named in [*((i, h, k, "left_out", q) for i, h, k, q in left_out),
                                                  *((i, h, f, "left_out", f[0]) for i, h, f in figures),
                                                  *((i, h, n, "not_computed", None) for i, h, n in not_done)]:
            saved = issues.get(str(issue.get("id")))
            if isinstance(saved, dict) and outcome == "left_out" and str(issue.get("id")) in recorded:
                # one finding that leaves out a result and a figure: the record names both
                saved["at_gate"]["keys"] = [*saved["at_gate"].get("keys", []), *names]
                continue
            if isinstance(saved, dict):
                recorded.add(str(issue.get("id")))
                saved["at_gate"] = {"outcome": outcome, "how": how, "keys" if outcome == "left_out" else "names": names,
                                    "quantity": named or (_code_review.quantity_names(record, issue) or names)[0]}
                if _shown_reach._norm(saved["at_gate"]["quantity"]) in {_shown_reach._norm(k) for k in changed.get(str(issue.get("id")), ())}:
                    saved["at_gate"]["rerun"] = "changed"
        if left_out or not_done or figures:
            _code_review.save(self.quest_root, record)
            # per finding: a key is said changed only for the finding whose own rerun moved it
            changed_by_quantity = {q or (_code_review.quantity_names(record, i) or k)[0]: changed.get(str(i.get("id")), set())
                                   for i, _h, k, q in left_out}
            self._leave_out_shown_faults(
                [(q or (_code_review.quantity_names(record, i) or k)[0], k, h) for i, h, k, q in left_out],
                [(n, h) for _i, h, names in not_done for n in names], result,
                figures=[(f, h) for _i, h, files in figures for f in files],
                changed=changed_by_quantity)
        return None

    def _untied_plan_text_only(self, record: dict[str, Any], issue: dict[str, Any], result: Any, headline: list[str],
                               reach: "_shown_reach.Reach | None") -> list[str]:
        """The plan's dependent variables a finding is about, when FI's own records show the one thing the gate may release it
        for: they are placed on no declared result (:func:`core.code_review.untied_dependents`), the frozen plan is of the
        contract form, the run's results and the analysis record exist, and the failing test reads only text FI wrote from the
        plan: no key that holds a value, no trial run, no figure, no result. Anything missing returns ``[]`` (the finding is
        judged as before)."""
        contract = getattr(reach, "contract", None)
        if reach is None or getattr(contract, "form", "") != "contract" or not reach.ids:
            return []
        if not (isinstance(result, dict) and result) or not _analysis_frame.read_record(self.quest_root):
            return []
        names = _code_review.untied_dependents(record, issue, reach.ids, headline)
        if not names:
            return []
        try:
            if reach.reads_held_key(issue, result):
                return []
            placed = reach.effect(record, issue, result)
        except Exception as exc:  # noqa: BLE001 - what cannot be read is not released
            self._log.warning("[evidence_gate] %s: FI's records could not place the test (%r)", issue.get("id"), exc)
            return []
        if placed is None or not placed.plan_text or placed.stop or placed.left_out or placed.not_done or placed.figures                 or placed.probes or placed.extra_keys or placed.cannot:
            return []
        return names

    @staticmethod
    def _part_not_done_or_stop(stops: list, not_done: list, issue: dict[str, Any], how: str, record: dict[str, Any], result: Any,
                               declared: list[str] | None, reach: "_shown_reach.Reach | None") -> None:
        """A finding that a plan quantity is not computed: a part not done only when FI's own record of the run shows no value
        was worked out for it AND the failing test reads no reported key that holds a value (the test would place the finding on
        that key). Otherwise the test may stand for a value reported under another name, and the quest stops."""
        absent = _code_review.not_obtained(record, issue, result, declared)
        if absent and not (reach is not None and reach.reads_held_key(issue, result)):
            not_done.append((issue, how, absent))
        else:
            stops.append((issue, how + "; " + _UNPLACED_FINDING))

    @staticmethod
    def _in_results(names: list[str], result: Any) -> bool:
        """Whether any of ``names`` is a key somewhere in the printed results."""
        have: set[str] = set()

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                for k, v in x.items():
                    have.add(_shown_reach._norm(k))
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(result)
        return any(_shown_reach._norm(n) in have for n in names)

    def _rerun_moved_only_checks(self, rerun: Any, reach: "_shown_reach.Reach | None", result: Any) -> bool:
        """Whether a rerun that changed numbers moved only measures that are no declared result and no key of the results (the
        numbers FI's own checks are made of, each a measure of the trials' own record): then no reported number changes. A rerun that also moved a number FI cannot
        attribute to a result (``unplaced``) or counted failed trials has not been cleared, and the finding is placed as before."""
        if reach is None or rerun.unplaced or "failed trials" in rerun.metrics or not rerun.metrics:
            return False
        if getattr(rerun, "unwritten", None):
            return False  # a delivered result file the repaired run did not write was not compared: its numbers are not cleared
        if not rerun.keys or not all(str(k).startswith("trials/") for k in rerun.keys):
            return False  # a number FI cannot show to be one a trial measures (a derived number, no key at all) is not cleared
        try:
            placed = reach.effect_of_changes(rerun.metrics)
        except Exception as exc:  # noqa: BLE001 - what cannot be placed is not shown to leave the results alone
            self._log.warning("[evidence_gate] the numbers a rerun moved could not be placed (%r)", exc)
            return False
        return not (placed.stop or placed.left_out or placed.figures or placed.not_done or placed.plan_text
                    or self._in_results(placed.extra_keys, result))

    def _unchanged_is_shown(self, issue: dict[str, Any], record: dict[str, Any], result: Any, declared: list[str] | None,
                            reach: "_shown_reach.Reach | None") -> bool:
        """Whether numbers that did not move answer the finding: only a finding about a number the code computes. A finding
        that a quantity is not computed (or not obtained), or that is about a figure, says something no reported number can
        show gone."""
        try:
            if _code_review.missing_quantities(record, issue) or _code_review.not_obtained(record, issue, result, declared):
                return False
            if reach is not None:
                placed = reach.effect(record, issue, result)
                if placed is not None and (placed.figures or placed.not_done):
                    return False
        except Exception as exc:  # noqa: BLE001 - what cannot be read cannot show the finding answered
            self._log.warning("[evidence_gate] %s could not be read for the rerun's answer (%r)", issue.get("id"), exc)
            return False
        return True

    async def _rerun_finding_test(self, rel: str) -> Any:
        """One finding test run on the code as it is, in its own process: its ``TestRun``, or ``None`` when the process
        printed no answer (it ran past its time, or ended the process)."""
        try:
            _rc, text, _err = await self._run_in_env(["-c", _finding_repair.harness_source("code", [rel])],
                                                     timeout_s=_code_findings.SHOWN_TEST_TIMEOUT_S,
                                                     env=self._finding_test_env(stand_ins=False))
        except Exception as exc:  # noqa: BLE001 - a run past its time limit raises from the executor
            self._log.info("[evidence_gate] the test %s gave no answer (%s)", rel, _one_line(str(exc), 120))
            return None
        if _finding_repair.MARK not in (text or ""):
            return None
        return _finding_repair.parse_runs(text, [rel]).get(rel)

    def _shown_fault_stop(self, issue: dict[str, Any], how: str) -> dict[str, Any]:
        where = " ".join(str(x) for x in (issue.get("file"), issue.get("function")) if x)
        what = str(issue.get("text") or issue.get("why") or issue.get("quantity") or "")
        if _UNPLACED_FINDING in how:
            # The finding and what the person can do come first: the record of the stop keeps only its first 400 characters.
            detail = how.replace("; " + _UNPLACED_FINDING, "").replace(_UNPLACED_FINDING, "").strip("; ")
            why = (f"code finding ({_one_line((where + ': ' if where else '') + what, 90)}): {_UNPLACED_FINDING}. A fault that "
                   f"changes a reported number was shown by a failing test and never shown to be gone from the code: {detail}")
        else:
            why = (f"code finding ({_one_line((where + ': ' if where else '') + what, 90)}): a fault in the code that changes a "
                   "reported number was shown by a failing test, and was never shown to be gone from the code that produced "
                   f"the results: {how}")
        self._log.warning("[evidence_gate] %s; stopping without a paper", why)
        return {"verdict": "insufficient", "rationale": why, "gaps": [why], "stuck": True,
                "stuck_reason": "shown_fault_open"}

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
            why = ("the study's main quantity is a constant in the run's results, as a second model found in the code "
                   "(" + "; ".join(_constant)[:200] + ")")
            self._log.warning("[evidence_gate] %s; stopping without a paper", why)
            return {"verdict": "insufficient", "rationale": why, "gaps": [why], "stuck": True,
                    "stuck_reason": "headline_missing"}
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
