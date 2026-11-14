"""Methods of :class:`core.engine.Engine` for writing the code and the checks made before the run.

``Engine`` inherits :class:`CodeStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, AsyncProcessTree, Callable, Path, QuestState, _BODIES_REPLY_REMINDER, _CI_BOOTSTRAP, _CI_WILSON,
    _CLUSTER_BODIES_REMINDER, _CLUSTER_REPLY_REMINDER, _FIGURE_SUFFIXES, _FILLED_AFTER_READ, _FIT_APPROVER,
    _FIT_SOURCE, _FI_PLAN_AUTHORS, _FLAT_APPROVER, _FLAT_SOURCE, _FRESH_SCRIPT, _ModelAnswerProblem,
    _PIP_FREEZE_TIMEOUT_S, _PY_FENCE_RE, _RECORD_REWRITES, _REDERIVE_BECAUSE, _SPLIT_REPLY_REMINDER,
    _SPLIT_RERUN_NOTE, _TAKE_OUT_PASSES, _accepted, _agent_coder, _aggregate_result_json_replicates, _analysis_frame,
    _assertion_violations, _attempts, _audit_open, _best_report, _changelog, _code_delta, _code_findings,
    _code_layout, _code_project, _code_review, _coerce_dep_list, _contract, _criteria, _declared_metrics,
    _design_parts, _disclosure, _equation_tests, _evidence, _experiment_deps, _extend_directive, _extract_result_json,
    _figdata, _finding_repair, _flat_fact_sentence, _format_clarify, _format_reflect_history, _frozen,
    _function_steps, _grid_keys, _improve, _in_run, _is_degenerate_result, _is_metric_key, _ledger, _lower_first,
    _mc_error, _metric_spec, _numeric, _one_line, _optim, _optimise, _optimise_search, _oracle, _oracle_card,
    _oracle_challenge, _oracle_triage, _outside_run, _pairs, _paper_facts, _parse_implement_response,
    _parse_json_lenient, _parse_split_deps, _phased, _plan, _plot_scope, _precision_target_on, _protocol,
    _protocol_directive, _random_source_found, _read_figure_records, _read_json_or_none_path, _receipts,
    _record_claims, _refine, _repair_reply_json, _replicate_assertions, _replicate_env, _replicate_metric_kinds,
    _rerun_directive, _rerun_from, _result_comparison_stats, _review, _review_sends_the_experiment_back,
    _run_manifest, _second_impl, _skeleton, _split_run, _step_passages, _stop_power, _topic_req, _trial_code,
    _trial_runner, _trial_table, ast, asyncio, hashlib, json, math, node_provider_for, re, scrub_secrets, shutil,
    time, unseeded_rng_calls,
)


class CodeStageMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

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
        design_sha = self._outline_design_sha(state)
        if state.get("implement_outline") and state.get("implement_outline_for", design_sha) == design_sha:
            # Already populated for THIS design (resume after a failed body call, or a
            # checkpoint that was advanced by a future-version engine).
            # Don't re-bill an outline call.
            self._log.info("[implement_outline] cached outline present; skipping")
            return {}
        if state.get("implement_outline"):
            self._log.info("[implement_outline] the design changed since the outline was written; writing it again")
        self._log.info("[implement_outline] drafting scaffold + signatures")
        try:
            # FI writes the analysis too (a plan in the contract form): the outline lists the functions of derive.py instead
            prompt = self._frame_outline_template(state, self._prompts["implement_outline"],
                                                  self._analysis_bodies(state)).substitute(
                topic_block=self._topic_block(),
                design_block=self._code_design_block(state, names_only=True),
                clarify_block=_format_clarify(state),
                timeout_s=self._planning_limit(state),
                skills_block=(self._outline_skills_block(state) + self._dropped_skills_note(state)) or "(no skills selected for this quest)",
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
        outline = self._outline_with_example_names(state, outline)
        n_funcs = len(outline.get("functions") or [])
        n_const = len(outline.get("constants") or [])
        self._log.info(
            "[implement_outline] scaffold ready — %d functions, %d constants",
            n_funcs, n_const,
        )
        return {"implement_outline": outline, "implement_outline_for": design_sha}

    def _outline_design_sha(self, state: QuestState) -> str:
        """The hash of the design as the outline request sees it (:meth:`_code_design_block`): the outline is kept only for
        this design."""
        return hashlib.sha256(self._code_design_block(state, names_only=True).encode("utf-8")).hexdigest()[:16]

    def _outline_with_example_names(self, state: QuestState, outline: dict[str, Any]) -> dict[str, Any]:
        """The outline with the parameters of each equation's function set to the inputs of the plan's worked example for it
        (:func:`core.function_steps.align_outline`): the plan's example is its own record, frozen before any code, and the
        outline is only the code-writing model's reading of the same equation. FI sets the names itself, so the outline, the
        equation tests and the repair guard (:meth:`_outline_signatures`) name the same parameters, and no one is asked to
        rename anything. Nothing is changed when the names already agree."""
        try:
            protocol = self._protocol_block(state)
            aligned, changed = _function_steps.align_outline(outline, protocol if isinstance(protocol, dict) else None)
        except Exception as e:  # noqa: BLE001 -- the outline as written is still usable
            self._log.warning("[implement_outline] could not set the functions' parameters from the plan's examples (%r)", e)
            return outline
        for line in changed:
            self._log.info("[implement_outline] the parameters of %s are the inputs of the plan's worked example", line)
        return aligned

    async def _node_implement(self, state: QuestState) -> QuestState:
        self._forget_cannot_repair()  # the code is written again: an earlier problem does not stop the new code
        if (search_stop := self._search_cannot_start_stop(state, state.get("design"))) is not None:
            return search_stop  # no code is written for a search FI could not make (the run step stops the same way)
        if (grid_stop := self._grid_too_large_stop(state)) is not None:
            return grid_stop  # no code is written for settings FI would not list (the run step stops the same way)
        await self._shadow("implement", state, taken="implemented the design")
        kept = self._adopt_scripts_fixed_by_hand(state)
        if kept is not None:
            return {**_FRESH_SCRIPT, **kept, "refine_extend": []}
        extend = [str(p).strip() for p in state.get("refine_extend") or [] if str(p).strip()]
        # A number the person asked to have added extends the script that exists; the scaffold is for a first script.
        outline = {} if extend else self._outline_with_example_names(state, state.get("implement_outline") or {})
        if self._agent_mode():
            # One coding agent writes the whole program; FI's checks before the run judge it (core/agent_build.py).
            return await self._implement_by_agent(state, outline or self._outline_with_example_names(
                state, state.get("implement_outline") or {}), extend)
        body_prompt = self._prompts.get("implement_body")
        # FI writes the analysis as well (core/analysis_frame.py; a plan in the contract form, the frame on): the functions of
        # derive.py the model is asked for, else ``None``. Read once, so the request and what is written of its reply agree.
        bodies = self._analysis_bodies(state)
        # The functions that already ran and passed are kept as they ran; the model is asked for the rest (core/kept_bodies.py)
        kept = None if extend else self._bodies_kept(state, bodies, outline)
        if kept is not None:
            bodies = [b for b in bodies or [] if b not in kept.names]
        if outline and outline.get("scaffold") and body_prompt is not None:
            self._log.info(
                "[implement] filling scaffold (%d functions to body)",
                len(outline.get("functions") or []),
            )
            # The model's package, one function at a time (core/function_steps.py): when it is written and checked, the
            # request below asks for the two scripts only.
            await self._fill_model_functions(state, outline)
            prompt = self._frame_template(state, body_prompt, bodies).substitute(
                topic_block=self._topic_block(),
                design_block=self._code_design_block(state),
                clarify_block=_format_clarify(state),
                outline_block=json.dumps(self._frame_outline(state, outline), indent=2),
                timeout_s=self._planning_limit(state),
                skills_block=(self._skills_block(state) + self._dropped_skills_note(state)) or "(no skills selected for this quest)",
                inputs_block=self._inputs_block(),
                job_block=self._job_block(state),
                split_block=self._split_block(state, bodies),
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
            prompt = self._frame_template(state, self._prompts["implement"], bodies).substitute(
                topic_block=self._topic_block(),
                design_block=self._code_design_block(state),
                timeout_s=self._planning_limit(state),
                skills_block=(self._skills_block(state) + self._dropped_skills_note(state)) or "(no skills selected for this quest)",
                inputs_block=self._inputs_block(),
                job_block=self._job_block(state),
                split_block=self._split_block(state, bodies),
            )
        prompt = self._frame_note(state) + prompt  # FI writes the frame of simulate.py: said first (core/run_checks.py)
        if kept is not None:
            prompt = self._keep_note(kept) + prompt
        # A review that named something this run computed sent the experiment
        # back here (``re_execute``). Both prompts above carry the design and
        # the outline but no review, so without this the model would regenerate
        # the same code — and with it the same wrong value.
        rerun_for = _review_sends_the_experiment_back(state.get("review") or {}, state)
        if rerun_for:
            self._log.info(
                "[implement] the review sent the experiment back over %r", rerun_for,
            )
            prompt += _rerun_directive(state.get("review") or {}, rerun_for, analysis=bodies is not None)
            if self._split_on(state) and (self.quest_root / "code" / _split_run.SIMULATE_NAME).is_file():
                prompt += (_skeleton.RERUN_NOTE_ANALYSIS if bodies is not None
                           else _skeleton.RERUN_NOTE if self._frame_variant(state) else _SPLIT_RERUN_NOTE)
            kept_asks = [str(p) for p in (state.get("extend_check") or {}).get("asked") or [] if str(p).strip()]
            if kept_asks and not extend:
                # The script was extended for a person's request; this rerun must not drop what was added for it.
                prompt += ("\n\nThe script was extended earlier for the reader's request: "
                           + "; ".join(p[:200] for p in kept_asks)
                           + (". Keep what was added for it.\n" if bodies is not None
                              else ". Keep what was added for it, and keep its results in RESULT_JSON.\n"))
        if extend:
            self._log.info("[implement] extending the existing script for: %s", "; ".join(p[:80] for p in extend))
            prompt += _extend_directive(self._study_scripts_on_disk() if bodies is not None else self._scripts_on_disk(),
                                        extend, self._submit_on_disk(), package=self._package_shown(state),
                                        frame=bool(self._frame_variant(state)), analysis=bodies is not None)
            if self._split_on(state) and (self.quest_root / "code" / _split_run.SIMULATE_NAME).is_file():
                prompt += (_skeleton.RERUN_NOTE_ANALYSIS if bodies is not None
                           else _skeleton.RERUN_NOTE if self._frame_variant(state) else _SPLIT_RERUN_NOTE)
        # A revised plan: the scripts that exist are changed only where the revision touched what they read (the model's
        # functions, written above, were kept or written again by their own record). Not when FI writes the analysis: the
        # revision's path gives back an experiment.py, which is FI's there; the request above asks for the bodies instead.
        incremental = None
        if not extend and not rerun_for and outline and outline.get("scaffold") and bodies is None:
            incremental = await self._incremental_scripts(state, outline)
        # Nothing left to ask for when the simulation and every function of derive.py are kept as they ran
        everything_kept = kept is not None and kept.trial and not bodies
        text = "" if incremental is not None or everything_kept else await self._chat(prompt, node="implement")
        # Two scripts (execution.split_analysis): the simulation, and the analysis, which
        # keeps the name ``experiment.py`` because everything else reads that one. A reply
        # that does not hold both is asked for once more; if it still does not, the quest
        # runs as one script and says so, rather than stopping.
        simulate_code = ""
        submit_code = ""
        package_files: dict[str, str] = {}
        if self._split_on(state):
            fis = bodies is not None  # FI writes the analysis: the reply holds simulate.py and derive.py's functions
            scripts = ({"simulate": incremental["simulate"], "analysis": incremental["analysis"]} if incremental is not None
                       else _split_run.parse_split_response(text, _PY_FENCE_RE, bodies=fis,
                                                            simulate_kept=kept.simulate if kept is not None and kept.trial else ""))
            if scripts is None:
                self._log.warning(
                    "[implement] execution.split_analysis is on but the reply did not hold %s; asking once more",
                    "the `# file: simulate.py` block" if fis
                    else "both scripts (`# file: simulate.py` and `# file: experiment.py`)",
                )
                text = await self._chat(prompt + (_BODIES_REPLY_REMINDER if fis else _SPLIT_REPLY_REMINDER), node="implement")
                scripts = _split_run.parse_split_response(text, _PY_FENCE_RE, bodies=fis)
            if (scripts is not None and incremental is None and self.config.execution.background_jobs
                    and "submit" not in scripts):
                self._log.warning("[implement] a cluster quest's reply did not hold submit.py; asking once more")
                text = await self._chat(prompt + (_BODIES_REPLY_REMINDER + _CLUSTER_BODIES_REMINDER if fis
                                                  else _SPLIT_REPLY_REMINDER + _CLUSTER_REPLY_REMINDER), node="implement")
                scripts = _split_run.parse_split_response(text, _PY_FENCE_RE, bodies=fis) or scripts
            if scripts is not None:
                # The research-tool layout (core/code_layout.py): the model's package, asked for once more when missing.
                scripts, text, package_files = await self._code_package_reply(state, prompt, text, scripts,
                                                                              extend=bool(extend), bodies=fis)
                simulate_code = self._framed(state, scripts["simulate"])
                # FI's own experiment.py and the study's functions put into FI's frame of derive.py (core/analysis_frame.py)
                code = (self._analysis_written(state, scripts.get("derive", ""), simulate_code, kept.names if kept else ()) if fis
                        else scripts["analysis"])
                submit_code = scripts.get("submit", "")
                deps = (incremental["deps"] if incremental is not None else list(state.get("deps") or []) if everything_kept
                        else _parse_split_deps(text))
                broken = list(_run_manifest.split_lint({_split_run.SIMULATE_NAME: simulate_code, "experiment.py": code}))
                model_package = next(iter(package_files), "").split("/")[0] or self._package_in_use(state)
                if model_package and _code_layout.imports(code, model_package):
                    broken.append(f"experiment.py imports the model's package {model_package}: the analysis must read "
                                  "only FI's record of the trials")
                if broken:
                    self._split_contract_broken(broken, what="the two scripts break the two-script contract",
                                                source=_stop_power.READ)
            else:
                if self.config.execution.split_failure == "block":
                    self._split_contract_broken(
                        ["the reply did not hold the `# file: simulate.py` block, also when asked again" if fis else
                         "the reply did not hold both scripts (`# file: simulate.py` and `# file: experiment.py`), also when asked again"],
                        what="the code-writing step did not return both scripts", source=_stop_power.MEASURED,
                    )
                self._log.warning(
                    "[implement] the reply still did not hold both scripts; this quest runs as "
                    "ONE script (execution.split_analysis has no effect on it)",
                )
                code, deps = _parse_implement_response(text)
        else:
            code, deps = _parse_implement_response(text)
        if extend and self._scripts_on_disk().get("experiment.py", "").strip() and (
            not code.strip()
            or (self._split_on(state) and (self.quest_root / "code" / _split_run.SIMULATE_NAME).is_file()
                and not simulate_code.strip())
        ):
            # The extension is only worth having if it came back whole. A reply with no script, or one script for a
            # two-script quest, must not replace the scripts that ran: they stay as they were, and the run says so.
            self._log.warning(
                "[implement] the extension did not come back (LLM head: %r); the scripts that exist are kept as they "
                "are, so the number asked for (%s) is still missing", text[:200] if text else "<empty>",
                "; ".join(p[:80] for p in extend),
            )
            disk = self._scripts_on_disk().get("experiment.py", "")
            return {**_FRESH_SCRIPT, "code": disk, "deps": list(state.get("deps") or []), "refine_extend": [],
                    "extend_missed": extend, "extend_check": {}, "extend_unreported": []}
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
        deps = sorted({*deps, *design_deps, *(self._fis_analysis_packages(state) if bodies is not None else [])})
        left_out = self._dropped_tool_names(state)
        if left_out:
            kept = [d for d in deps if _experiment_deps.normalize(_experiment_deps.requirement_name(d) or d) not in left_out]
            for d in deps:
                if d not in kept:
                    self._log.info("[implement] not asking pip for %r: it is a skill this quest no longer carries", d)
            deps = kept

        code_path = self.quest_root / "code" / "experiment.py"
        simulate_path = self.quest_root / "code" / _split_run.SIMULATE_NAME
        sim_kept = False  # the simulation came back unchanged: its trials already run stay in use
        if simulate_code.strip():
            # A script written again identically is left exactly as it was: the raw files
            # it wrote are recorded against its bytes, and are used only while they match.
            previous = simulate_path.read_text(encoding="utf-8") if simulate_path.is_file() else ""
            if not _split_run.same_script(previous, simulate_code):
                simulate_path.write_text(simulate_code, encoding="utf-8")
                self._log.info("[implement] wrote %s (%d bytes)", simulate_path, len(simulate_code))
            else:
                self._log.info("[implement] %s is unchanged; its raw files stay in use", simulate_path.name)
                sim_kept = True
        elif simulate_path.is_file():
            # This pass wrote ONE script: a simulation left from an earlier pass would run beside it and mix two
            # iterations. Under ``auto`` a quest with a simulation asks for two scripts every time (``_split_on``), so
            # this is reached only when the reply held one script, or when ``split_analysis`` is false.
            simulate_path.unlink()
            self._log.warning("[implement] removed the earlier %s: this pass wrote one script", simulate_path.name)
        submit_path = self.quest_root / "code" / _trial_runner.SUBMIT_NAME
        if submit_code.strip():
            submit_path.write_text(submit_code, encoding="utf-8")
            self._log.info("[implement] wrote %s (%d bytes)", submit_path, len(submit_code))
        code_path.write_text(code, encoding="utf-8")
        self._log.info("[implement] wrote %s (%d bytes)", code_path, len(code))
        package_written: list[str] = []
        if package_files:
            package_written = _code_layout.write_package(code_path.parent, package_files, same=_split_run.same_script)
            if package_written:
                self._log.info("[implement] wrote the model's package: %s",
                               ", ".join(f"code/{w}" for w in package_written))
                self._write_plan_constants(state, package_written[0].replace("\\", "/").split("/")[0])
        if extracted and not extend:  # nothing to seed in the stub written above; an extension changes as little as it can
            if simulate_code.strip() and _trial_runner.entries(simulate_path) & {"run_trial", "run_cell"}:
                pass  # the trial contract: FI hands every trial its seed, there is no seed for the script to read
            elif simulate_code.strip():
                # The seed belongs to the simulation: the analysis is deterministic.
                _unused, deps = await self._repair_ignored_replicate_seed(
                    state, simulate_path, simulate_path.read_text(encoding="utf-8"), deps,
                )
            else:
                code, deps = await self._repair_ignored_replicate_seed(state, code_path, code, deps)
        if extracted:
            code, deps = await self._enforce_protocol(state, code_path, simulate_path, code, deps, extended=bool(extend))
            # The labels of the plan's equations (`# E1`): asked for once more when the reply left one out.
            # Not when the simulation came back unchanged: adding comments changes its bytes, and the trials FI already ran
            # are kept against those bytes, so they would all run again. The check before the run still says what is missing.
            # With the model's package, the labels go in it: a package written again this pass runs its trials again anyway.
            if (not sim_kept or package_written) and await self._label_equations(state) == code_path:
                code = code_path.read_text(encoding="utf-8")  # a one-script quest: the labelled script is the one that runs
        if not extend and outline and outline.get("scaffold"):
            _code_delta.save(self.quest_root, self._scripts_built_from(state, outline),
                             package=(_code_layout.load(self.quest_root) or {}).get("package", ""))
        # The kind of change for code/CHANGELOG.md, from what this step knows (core/changelog.py).
        if extend:
            note, kind = "added what a refine asked for: " + "; ".join(extend)[:200], "added"
        elif rerun_for:
            note, kind = f"what the review found wrong in {rerun_for} (a number the run computes)"[:300], "fixed"
        elif int(state.get("iteration", 0) or 0) > 0:
            note, kind = f"revised after review (round {int(state.get('iteration', 0) or 0)})", "changed"
        else:
            note, kind = "code written", "added"
        await self._refresh_code_project(state, deps, note, category=kind)
        from core import number_provenance

        out: QuestState = {**_FRESH_SCRIPT, "code": code, "deps": deps, "refine_extend": [], "extend_missed": []}
        if extend:
            # What the results held before the extension, by name: what it adds is then known, and each write after
            # the run checks that it is in the paper (``_node_write``).
            out["extend_check"] = {"asked": extend, "before": number_provenance.result_paths(state.get("result_json"))}
            out["extend_unreported"] = []
        elif not rerun_for:
            # A script written from the design: an earlier extension's results are not what this run computes. A rerun
            # the review asked for is told to keep the extension (above) and keeps the check, so the paper written
            # after it is still checked for the number asked for.
            out["extend_check"], out["extend_unreported"] = {}, []
        return out

    async def _study_follows_protocol(self, state: QuestState) -> None:
        """Before any trial runs: the settings in ``code/study.json`` (the runs per setting and the settings to vary) are the
        protocol's. The protocol is the source of truth, so a study that differs is written again from it (a file a person
        edited is left, and said). Never stops a quest: FI's own run reads the protocol, not this file."""
        try:
            protocol = self._protocol_block(state)
            found = _code_project.study_differences(self.quest_root / "code", protocol, split=self._split_on(state),
                                                        design=state.get("design"))
            if not found:
                return
            self._log.warning("[execute] code/study.json is not what the protocol says (%s); writing it again from the protocol",
                              "; ".join(found)[:400])
            await self._refresh_code_project(state, list(state.get("deps") or []))
            if _code_project.study_differences(self.quest_root / "code", protocol, split=self._split_on(state),
                                                        design=state.get("design")):
                self._log.warning("[execute] code/study.json still differs from the protocol (you edited it, so FI left it); "
                                  "the study FI runs follows the protocol, not that file")
        except Exception as exc:  # noqa: BLE001 -- a convenience file, never a reason to stop
            self._log.warning("[execute] could not compare code/study.json with the protocol: %r", exc)

    async def _deliver_check_values(self, state: QuestState) -> None:
        """Write the values FI's tests in ``code/tests/`` compare with into ``.fi/checks/`` (a person's copy; FI judged
        with the values it holds). Only once the quest's code has run for the last time in this call of :meth:`run`:
        written earlier, code run without isolation could open them by an absolute path. Never stops a quest."""
        try:
            package = self._package_in_use(state)
            if package:
                await asyncio.to_thread(_code_layout.write_check_values, self.quest_root / "code",
                                        self._protocol_block(state), self._outline_equation_names(state))
        except Exception as exc:  # noqa: BLE001 -- a convenience for a person, not part of the result
            self._log.warning("[code] could not write the values code/tests/ compares with: %s", exc)

    async def _refresh_code_project(self, state: QuestState, deps: list[str], note: str = "", *,
                                    category: str = "changed") -> None:
        """Keep ``code/`` a project that runs on its own (README, requirements, run.py) and, with a ``note``, record
        what changed in its history and CHANGELOG.md (as ``category``: added / changed / fixed / tidied); never stops
        a quest."""
        try:
            idea = state.get("chosen_idea") if isinstance(state.get("chosen_idea"), dict) else {}
            question = str(idea.get("question") or idea.get("title") or state.get("topic") or "")[:500]
            # git and the file scan block: keep them off the event loop so a --fleet's other quests keep going.
            protocol = self._protocol_block(state)
            package = self._package_in_use(state)
            # A quest begun by an older FI (a resume) can still hold expected values in code/tests/: removed, and FI's
            # test files written again below in the current form.
            stale = await asyncio.to_thread(_equation_tests.remove_stale, self.quest_root / "code")
            if stale:
                self._log.info("[code] removed files an older FI left in code/ with expected values in them: %s",
                               ", ".join(stale))
            # What the tests FI writes into code/ compare with is not on disk while the code can still run: FI judges
            # with the values it holds, and a person's copy is written at the end (_deliver_check_values).
            await asyncio.to_thread(_code_layout.remove_check_values, self.quest_root / "code")
            await asyncio.to_thread(
                _code_project.refresh,
                self.quest_root, deps=deps, protocol=protocol,
                title=str(state.get("title") or ""), question=question, split=self._split_on(state),
                log=self._log, design=state.get("design"),
                extra_files=(_code_layout.project_files(self.quest_root / "code", protocol, package,
                                                        self._outline_equation_names(state)) if package else None),
                readme_files=_code_layout.readme_lines(package) if package else None,
            )
            if note:
                await asyncio.to_thread(_code_project.record_change, self.quest_root, note, category=category,
                                        log=self._log)
        except Exception as exc:  # noqa: BLE001 -- these files are a convenience, not part of the result
            self._log.warning("[code] could not update the runnable project in code/: %s", exc)

    def _ask_about_edited_project_files(self) -> None:
        """A file of ``code/`` that a person edited and FI would now change: ask once before touching it (a pause when
        the quest is interactive), otherwise keep it and say so. FI never replaces it either way."""
        try:
            names = _code_project.unasked_conflicts(self.quest_root)
            if not names:
                return
            _code_project.mark_asked(self.quest_root, names)
        except Exception as exc:  # noqa: BLE001 -- a convenience, never a reason to stop
            self._log.warning("[code] could not check your edits to code/: %s", exc)
            return
        listed = ", ".join(f"code/{n}" for n in names)
        hint = ("FI kept your version. To let FI update it, delete or rename the file; it writes its own on the next "
                "step.")
        if self.config.pauses.review != "ask":
            self._log.warning("[code] you edited %s and FI has a newer version. %s", listed, hint)
            return
        self._pause_for_human(
            kind="code_project",
            interaction="supply",
            headline="you edited files that FI would update",
            steps=[
                f"You edited {listed}, and FI has a newer version of it.",
                "To keep your version, just resume: FI leaves it as it is.",
                "To let FI update it, delete or rename the file, then resume: FI writes its own.",
            ],
            payload={"code_project": True, "quest_id": self.quest_id, "files": names},
        )

    # ---- the protocol gate ---------------------------------------------------

    def _protocol_block(self, state: QuestState, *, adopted: bool = True) -> dict[str, Any] | None:
        """The protocol to hold the experiment to, when there is an experiment.

        Once the protocol has been frozen (right before the first full run, ``core/frozen_protocol.py``) it is the frozen
        record and nothing else: a later design that leaves the protocol out or changes it does not change what the gates
        check. Before that it is the draft (:meth:`_draft_protocol`)."""
        if (
            state.get("no_simulation_resolved")
            or state.get("survey_mode_resolved")
            or self.config.engine.analyze_local_first
        ):
            return None
        frozen = self._resolved_frozen()
        if frozen is not None:
            protocol = frozen.get("protocol")
            protocol = protocol if isinstance(protocol, dict) and protocol else None
            said = self._figure_declarations_after_the_freeze(state, protocol)
            if protocol is not None and said:  # not part of the frozen hash (core/frozen_protocol.py::UNFROZEN_KEYS)
                protocol = {**protocol, "figure_declarations": said}
        else:
            protocol = self._draft_protocol(state)
        self._note_contract(state, protocol, frozen, adopted)
        return protocol

    def _note_contract(self, state: QuestState, protocol: dict[str, Any] | None, frozen: dict[str, Any] | None,
                       adopted: bool) -> "_contract.StudyContract | None":
        """The study contract the protocol being read gives (:func:`core.contract.from_design`), built once for the
        design in force; what the plan left out is written to ``.fi/contract.json`` (``gaps``, and in ``shown`` the ones
        the run record says) for the quality report. It is built when the protocol is read and never written back into
        it (the frozen protocol's hash would change). A candidate design (``adopted`` false) is built but never written."""
        if not isinstance(protocol, dict):
            return None
        design = state.get("design")
        if not isinstance(design, dict) or not design:
            return None
        try:
            plan_file = self.quest_root / "plan.md"
            stat = plan_file.stat() if plan_file.is_file() else None
            # Cheap to look up on every call: the frozen record's own hash, else the plan file's stamp, and the design
            # object (every part of it that from_design reads: the assertions, the notes, the planned figures).
            key = (frozen.get("sha256") if frozen is not None else None, stat.st_mtime_ns if stat else None,
                   stat.st_size if stat else None, id(design), len(protocol))
            cache = self.__dict__.setdefault("_contract_cache", {})
            if key in cache and cache[key][0] is design and (cache[key][2] or not adopted):
                return cache[key][1]
            contract = cache[key][1] if key in cache and cache[key][0] is design else _contract.from_design(
                {**design, "protocol": protocol}, fi_analysis=self._fis_analysis_planned({**design, "protocol": protocol}))
            if len(cache) > 8:
                cache.clear()
            cache[key] = (design, contract, adopted)
            if adopted:
                path = self.fi_dir / "contract.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                sha = frozen.get("sha256") if frozen is not None else hashlib.sha256(
                    json.dumps(protocol, sort_keys=True, default=str).encode("utf-8")).hexdigest()
                path.write_text(json.dumps({
                    "protocol_sha": sha, "form": contract.form, "gaps": list(contract.gaps),
                    "shown": [g for g in contract.gaps if not _contract.is_routine_gap(g)]}, indent=2) + "\n",
                    encoding="utf-8")
            return contract
        except Exception as e:  # noqa: BLE001 -- the contract is read for disclosure; it never stops a quest
            self._log.warning("[contract] could not read the plan's contract (%r)", e)
            return None

    def _in_force(self, state: dict[str, Any]) -> Any:
        """``state`` with the protocol in force (the frozen one once there is one) as its design's protocol, for the
        checks of the results that read the design (:func:`_assertion_violations`)."""
        protocol = self._protocol_block(state)  # type: ignore[arg-type]
        if isinstance(protocol, dict):
            state = {**state, "design": {**(state.get("design") or {}), "protocol": protocol}}
        # When FI writes the analysis, state["code"] is FI's own experiment.py: what the checks read to learn what the study
        # did (a cap on a value) is the study's code, never FI's.
        study = _analysis_frame.study_code(self.quest_root / "code")
        if study is not None:
            state = {**state, "code": study}
            # A result FI works out from the trials is capped only in the trial code, a derived one only in derive.py;
            # which is which comes from the protocol in force, merged above.
            try:
                plan = _analysis_frame.plan_of(state["design"]) if _analysis_frame.applies(state.get("design")) == "" else None
            except Exception:  # noqa: BLE001 - the split is an aid; without it the study's code is read for every result
                plan = None
            if plan is not None:
                state["_derived_ids"] = sorted({d["id"] for d in plan["derived"]})
                state["_trial_code"] = _trial_code(self.quest_root / "code")
        return state

    def _draft_protocol(self, state: QuestState) -> dict[str, Any] | None:
        """The protocol before it is frozen (:meth:`_draft_protocol_base`), with the worked examples the plan worked out again
        when asked (:meth:`_rederive_examples`) in place of the ones a redesign would have kept."""
        protocol = self._draft_protocol_base(state)
        if not isinstance(protocol, dict):
            return protocol
        examples = self._examples_rederived().get("examples")
        return _equation_tests.with_examples(protocol, examples) if isinstance(examples, dict) and examples else protocol

    def _draft_protocol_base(self, state: QuestState) -> dict[str, Any] | None:
        """The protocol before it is frozen: on the first pass it is read from ``plan.md``, which is the source of truth
        for it (a person who edits the protocol while the quest is stopped, or asks for a change, is heard at the next
        implement or execute); a later pass, which only happens for a quest that was never frozen (one begun before the
        freeze existed), uses the design it made, or the plan's protocol when the design has none."""
        design = state.get("design") or {}
        planned, _why = _plan.load_design(self.quest_root)
        planned_protocol = planned.get("protocol") if isinstance(planned, dict) else None
        planned_protocol = planned_protocol if isinstance(planned_protocol, dict) and planned_protocol else None
        if int(state.get("iteration", 0) or 0) == 0 and planned_protocol:
            return planned_protocol
        protocol = design.get("protocol")
        if isinstance(protocol, dict) and protocol:
            return protocol
        return planned_protocol

    def _freeze_protocol_if_due(self, state: QuestState, *, at_confirm: bool = False) -> None:
        """Freeze the protocol right before the first full run, after the oracle gate has settled it (an oracle the
        engine added counts as part of it). A no-op once frozen. The record says who approved it: a person, when the
        plan was held for them to read (``pauses.plan: ask``), and otherwise the engine on its own. Under
        ``engine.phased`` it is frozen when exploration ends instead (``at_confirm``), before the confirm run."""
        if _frozen.load(self.quest_root) is not None:
            return
        if (not at_confirm and _phased.enabled(self.config) and _phased.stage(self.quest_root) == _phased.EXPLORE
                and not _phased.unconfirmable(self.quest_root)):
            self._log.info("[phased] exploration stage: the protocol is not frozen yet (the design may still change); "
                           "it is frozen when exploration ends, before the confirm run")
            return
        protocol = self._draft_protocol(state)
        mode = self.config.pauses.plan
        # A person approved the plan only if the quest actually stopped for them to read it (the plan stop leaves its
        # marker), not merely because ``pauses.plan`` says ``ask`` now: the setting can change between the two.
        held = mode == "ask" and (self.fi_dir / "paused_at_plan.flag").is_file()
        approved_by = (
            "human: the plan was held for the person to read and edit (pauses.plan=ask)"
            if held
            else f"auto: nobody approved this protocol before the run (pauses.plan={mode})"
        )
        # The oracles the gate had written into the plan after it was read (``_declare_oracles``) that are still in the
        # protocol being frozen: a person approved them only if the quest stopped again for them (``_hold_added_oracles``).
        added = self._oracles_added_read() or {}
        frozen_names = {str(o["name"]) for o in _oracle.declared(protocol if isinstance(protocol, dict) else None)}
        engine_added = [str(n) for n in added.get("oracles") or [] if str(n) in frozen_names]
        engine_removed = [str(n) for n in added.get("removed") or [] if str(n) not in frozen_names]
        oracle_note = ""
        if engine_added or engine_removed:
            # A test run of the checks changes or removes them (it has a reason); the gate adds them.
            how = "changed or removed" if added.get("reason") else "added"
            names = ", ".join(f"'{n}'" for n in [*engine_added, *engine_removed])
            engine_added = engine_added or engine_removed
            oracle_note = (
                f"the oracles {names} were {how} by the engine after the plan was written and held for the person to read"
                if added.get("shown") else
                f"the oracles {names} were {how} by the engine after the plan was written and nobody approved them"
            )
            if not added.get("shown"):
                approved_by = f"auto: {oracle_note}; " + (
                    "the rest of the plan was held for the person to read (pauses.plan=ask)" if held
                    else f"nobody approved the rest of the protocol either (pauses.plan={mode})"
                )
            elif held:
                approved_by += f"; {oracle_note} too"
            else:
                approved_by = f"auto: nobody approved the rest of this protocol (pauses.plan={mode}); {oracle_note}"
        if int(state.get("evidence_fit_redesigns") or 0) and approved_by.startswith("human"):
            # The person read plan.md; FI then had it rewritten to fit the time it measured, so they did not read this one.
            approved_by = ("auto: FI had the plan rewritten to fit the time it measured after the person read it "
                           f"(pauses.plan={mode}), and nobody approved the protocol it rewrote")
        if approved_by.startswith("human") and self._size_edit_after_read():
            # The person read plan.md; FI then made the study smaller to fit the time it measured, so they did not read this one.
            approved_by = ("auto: FI made the search smaller to fit the time it measured after the person read the plan "
                           f"(pauses.plan={mode}), and nobody approved the protocol it changed")
        iteration = int(state.get("iteration", 0) or 0)
        source = "plan.md" if iteration == 0 else f"design at iteration {iteration} (a quest begun before the protocol was frozen)"
        if at_confirm:
            source = f"the design exploration settled on ({source}), frozen when exploration ended"
            if iteration > 0 and approved_by.startswith("human"):
                # The person read plan.md; exploration then changed the design, so they did not read this protocol.
                approved_by = ("auto: exploration changed the design after the person read the plan "
                               f"(pauses.plan={mode}), and nobody approved the protocol it settled on")
        unsourced_note = ""
        gone_on = self._not_confirmed_names(state, protocol)
        if gone_on:
            chosen = _accepted.accepted(self.quest_root) or {}
            who = sorted({_accepted.who_text(((chosen.get("chosen") or {}).get(" ".join(n.split()).lower()) or {})
                                             .get("by") or chosen.get("by")) for n in gone_on})
            unsourced_note = (f"; the checks {', '.join(repr(n) for n in gone_on)} say nowhere where their expected "
                              f"value comes from ({_accepted.NOT_CONFIRMED}): {' and '.join(who)} went on "
                              "without one")
        try:
            filled = json.loads((self.fi_dir / _FILLED_AFTER_READ).read_text(encoding="utf-8")).get("checks") or []
        except (OSError, ValueError, AttributeError):
            filled = []
        filled = [str(n) for n in filled if str(n) in {str(o["name"]) for o in _oracle.declared(
            protocol if isinstance(protocol, dict) else None)}]
        if filled and approved_by.startswith("human"):
            unsourced_note += (f"; FI wrote where the expected value of {', '.join(repr(n) for n in filled)} comes from "
                               "after the plan was read (their numbers are as read)")
        approved_by += unsourced_note
        replaced = _frozen.open_replacement(self.quest_root)
        if replaced is not None:
            approved_by = (f"human: {replaced.get('approved_by')} approved replacing the protocol (--approve-as)"
                           + unsourced_note)
            if engine_added:
                # They approved going back to a step, before these oracles existed: say what they did not see.
                approved_by += f"; {oracle_note}"
            source = f"{replaced.get('source')}; it replaces the one frozen before (amendment {replaced.get('n')})"
            record = _frozen.freeze(
                self.quest_root, protocol, approved_by=approved_by, source=source,
                version=int(replaced.get("from_version", 1) or 1) + 1, amendments=int(replaced["n"]),
                sources=self._retrieved_sources(state))
            _frozen.close_replacement(self.quest_root, replaced, str(record["sha256"]))
        else:
            record = _frozen.freeze(self.quest_root, protocol, approved_by=approved_by, source=source,
                                    sources=self._retrieved_sources(state))
        for lost in _audit_open.check_against(
                self.quest_root, record.get("protocol"),
                why="the protocol that was frozen does not have it, though the audit's change was reported as made"):
            self._log.warning("[protocol] the audit's change to `%s` was reported as made and the frozen protocol does not "
                              "have it; it is kept as an open item for the code and the paper", lost["path"])
        self._log.info(
            "[protocol] frozen %s (%s, sha256 %s, run %s)%s",
            "at the end of exploration" if at_confirm else "before the first full run",
            record["source"], str(record["sha256"])[:12], record["run_id"],
            "" if protocol else " -- there is no protocol: nothing holds the experiment to a grid, runs or thresholds",
        )

    def _warn_if_plan_diverges(self) -> None:
        """Say so when ``plan.md`` no longer holds the protocol that is frozen: the frozen one is what the gates check, so an
        edit of the file after the freeze changes nothing until it goes through an amendment."""
        frozen = _frozen.load(self.quest_root)
        if frozen is None:
            return
        planned, _why = _plan.load_design(self.quest_root)
        protocol = planned.get("protocol") if isinstance(planned, dict) and isinstance(planned.get("protocol"), dict) else None
        if protocol and _frozen.sha256(protocol) != frozen.get("sha256"):
            self._log.warning(
                "[protocol] the protocol in plan.md differs from the frozen protocol (%s); the frozen one is what runs. A "
                "change is made by asking for it when the quest is refined, which stops for your approval.",
                "; ".join(_frozen.diff(frozen.get("protocol"), protocol)[:3]) or "no listed difference",
            )

    def _run_manifest_record(self, payload: dict[str, Any]) -> None:
        try:
            path = self.quest_root / "needs" / "RUN_MANIFEST_CHECK.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
            self._audit_check("run_manifest", path, status=str(payload.get("status")), problems=payload.get("problems"))
        except OSError:
            pass  # a record that cannot be written must never stop a quest

    def _given_row_findings(self, protocol: dict[str, Any], ledger: list[dict[str, Any]] | None,
                            result_json: Any) -> tuple[list[str], list[str]]:
        """Under ``rigor_profile: research``, a mean over a subset of the trials checked trial by trial against FI's
        own record (``core/trial_runner.py::given_rows_problems``): ``(the analysis's to fix, the simulation's)``.
        Outside research, nothing."""
        if self.config.rigor_profile != "research":
            return [], []
        ok_trials = sum(1 for r in ledger or [] if isinstance(r, dict) and r.get("status") == "ok")
        return _trial_runner.given_rows_problems(
            protocol, _trial_runner.recorded_rows_by_cell(self.quest_root), result_json, ok_trials=ok_trials)

    def _figure_scripts_with_typed_numbers(self) -> dict[Path, list[_figdata.TypedSeries]]:
        """Each script in ``code/`` that draws a figure (it saves one, or it is a derive.py whose figure functions FI saves)
        and has a plotting call drawing numbers typed into the code."""
        return _figdata.typed_series_in(self.quest_root / "code")

    async def _figures_from_results(self, state: QuestState) -> tuple[int, bool]:
        """Send a script that draws typed-in numbers back to draw its figures from the run's saved results: ``(the
        repairs used so far for this script, whether a script was rewritten)``. Bounded by ``_FIGURE_DATA_REPAIRS``; what
        is still typed in after that is left out of the paper (:meth:`_drop_typed_figures`)."""
        used = int(state.get("figure_data_repairs") or 0)
        rewrote = False
        self._typed_figures = self._figure_scripts_with_typed_numbers()
        while self._typed_figures and used < self._FIGURE_DATA_REPAIRS:
            used += 1
            for path, found in self._typed_figures.items():
                self._log.warning(
                    "[execute] %s draws figures from numbers typed into the code (%s); asking for them to be drawn from the "
                    "run's saved results (%d of %d)", path.name, "; ".join(f.says() for f in found[:4]), used,
                    self._FIGURE_DATA_REPAIRS)
                rewrote = await self._repair_typed_figures(state, path, found) or rewrote
            self._typed_figures = self._figure_scripts_with_typed_numbers()
        return used, rewrote

    async def _result_check_gave_up_with_figures(self, state: QuestState,
                                                 history: list[dict[str, Any]] | None = None) -> QuestState | None:
        """:meth:`_result_check_gave_up`, and the figures that failed drawn from the plan's own description besides: the run
        it sends back for one more run kept its results (the helper does not ask whether they were produced)."""
        kept = self._result_check_gave_up(state, history)
        if kept is None:
            return None
        return {**kept, **await self._figures_after_give_up(state, sent_back=True)}

    async def _figures_after_give_up(self, state: QuestState, *, sent_back: bool = False) -> dict[str, Any]:
        """The repairs ended by a give-up (the model refused or returned no code), after a run whose results stand and whose
        figure functions failed: the run that failed was not the one flagged for the plan's own figure, so the analysis is run
        once more WITHOUT the simulation (``experiment.py`` reads FI's record of the trials from disk) and draws only the
        figures that failed, in the plan's own description (``FI_FIGURE_FALLBACK=only``; a figure that was drawn keeps its file).
        The results and the derived values of the run are not touched: the analysis in this mode runs none of the study's
        functions again and writes back the earlier record with only the figures changed. The figures that failed are read from
        the primary run's record (replicates write derived.json again). ``sent_back``: the results check sent the run back, so
        ``results_produced`` is false and one more run follows. Returns the state's ``figures`` and ``figure_records`` as they
        are then, or nothing when there is nothing to draw."""
        try:
            record = self.__dict__.get("_primary_figure_record") or _analysis_frame.read_record(self.quest_root)
            lost = [o for o in record.get("figures") or [] if isinstance(o, dict)
                    and o.get("status") in ("failed", "refused", "not_written")]
            if (not lost or record.get("from_earlier_run") or not self._analysis_on()
                    or not (sent_back or (state.get("exec_result") or {}).get("results_produced"))):
                return {}
            self._log.info("[execute_reflect] the repair gave up with %d figure(s) not drawn; FI draws them from the plan's "
                           "own description, from the results it already holds (the simulation is not run again)", len(lost))
            env = {**(self.__dict__.get("_last_exec_env") or scrub_secrets()), "FI_FIGURE_FALLBACK": "only"}
            await self.executor.execute(
                [str(self.executor.python_path(self.quest_root)), str(self.quest_root / "code" / _analysis_frame.EXPERIMENT_FILE)],
                cwd=self.quest_root, timeout_s=self.config.execution.timeout_s, env=env)
            fig_dir = self.quest_root / "figures"
            figures = (sorted(p.name for p in fig_dir.iterdir() if p.is_file() and p.suffix.lower() in _FIGURE_SUFFIXES)
                       if fig_dir.is_dir() else [])
            records = {**(state.get("figure_records") or {}), **_read_figure_records(self.fi_dir / "figure_records", figures)}
            records = {k: v for k, v in records.items() if k in figures}
            self._figure_fallbacks(records)
            return {"figures": figures, "figure_records": records}
        except Exception as exc:  # noqa: BLE001 -- the figures are best-effort; the repair's give-up stands as it was
            self._log.warning("[execute_reflect] the plan's own figures could not be drawn after the give-up (%r)", exc)
            return {}

    def _figure_fallbacks(self, figure_records: dict[str, Any], record: dict[str, Any] | None = None) -> None:
        """What FI's analysis drew in place of a figure body no repair is left for (``FI_FIGURE_FALLBACK``, set in
        :meth:`_node_execute`; the figure is drawn by ``analysis_figures.default_figure`` from the plan's own description).
        A figure so drawn is marked ``fallback`` in its record, said in run.log. One the plan did not describe enough to draw
        is not in the paper: QUALITY.md says so (a row that stops nothing), and run.log. Both are read from FI's record of
        the run (derived.json), so a run that drew every body itself leaves nothing here. ``record``: the primary run's, when
        replicates have written derived.json since."""
        record = record if record is not None else _analysis_frame.read_record(self.quest_root)
        if record.get("from_earlier_run"):
            return
        self._clear_cannot_repair("figure_dropped")
        self._clear_cannot_repair("figure_fi_fault")
        self._clear_cannot_repair("figure_line_removed")
        for outcome in record.get("figures") or []:
            for gone in (outcome.get("removed") or []) if isinstance(outcome, dict) else []:
                # A line the figure's last run left out (core/frame_figure.py ``_no_identity``): FI could not say what it is,
                # so it is not drawn under a made-up name. QUALITY.md says which line and why; nothing stops.
                if not isinstance(gone, dict):
                    continue
                text = (f"Figure `{outcome.get('id')}`, panel {gone.get('panel')}: {gone.get('what')} was left out, because "
                        f"{gone.get('why')}.")
                self._log.warning("[execute] %s", text)
                self._record_cannot_repair(text, ["its repair did not give the line a name from FI's records"],
                                           source=_stop_power.READ, check="figure_line_removed")
            if isinstance(outcome, dict) and (outcome.get("status") == "fi_fault" or (
                    outcome.get("fallback_fault") and outcome.get("status") in ("failed", "refused", "not_written"))):
                # FI's own fault while drawing this figure: the results stand, the figure is missing, and the cause is FI's.
                # A default figure that failed in FI's own code after the body's last repair (``fallback_fault``, beside the
                # body's failure) is the same fault, disclosed the same way.
                fid = str(outcome.get("id"))
                reason = (str(outcome.get("why") or "").strip().rstrip(".") if outcome.get("status") == "fi_fault"
                          else str(outcome.get("fallback_fault") or "").strip().rstrip(".")) or "no reason was recorded"
                text = f"Figure `{fid}` was not drawn: FI's own drawing step failed ({reason}); this is FI's fault, not the study's."
                self._log.warning("[execute] %s", text)
                self._record_cannot_repair(text, ["FI's own drawing of the figure stopped; the study's functions were not at fault"],
                                           source=_stop_power.READ, check="figure_fi_fault")
                continue
            if not (isinstance(outcome, dict) and outcome.get("fallback")):
                continue
            fid, name = str(outcome.get("id")), Path(str(outcome.get("file") or "")).name
            if outcome.get("status") == "ok":
                if name in figure_records:
                    figure_records[name]["fallback"] = True
                self._log.warning("[execute] figure %s: its function could not be repaired, so FI drew it from the plan's own "
                                  "description (%s)", fid, outcome.get("note") or "one line for each result it shows")
            elif outcome.get("status") == "dropped":
                text = (f"Figure `{fid}` could not be drawn from the run's results; its numbers are in the results table. "
                        f"{outcome.get('why') or ''}").strip()
                self._log.warning("[execute] %s", text)
                self._record_cannot_repair(text, ["its function raised in every repair that was allowed",
                                                  "drew it from the plan's own description: " + str(outcome.get("why") or "")],
                                           source=_stop_power.READ, check="figure_dropped")

    async def _repair_typed_figures(self, state: QuestState, path: Path, found: list[_figdata.TypedSeries]) -> bool:
        """ONE repair of one script's plotting code: kept only if it parses and has fewer typed-in series than before."""
        code = path.read_text(encoding="utf-8")
        prompt = self._prompts["execute_reflect"].substitute(
            topic_block=self._topic_block(), own_failure_block="",
            previous_code=code, returncode="(not run yet)", stdout_tail=_figdata.directive(found), stderr_tail="",
            duration_s="0.00", figures_count="0", result_json_present="no (not run yet)",
            reflect_history_block=_format_reflect_history([]),
            design_block=self._code_design_block(state), clarify_block=_format_clarify(state),
        )
        body = next((name for name, (first, last, _file) in _figdata.fi_figure_bodies(code).items()
                     if first <= found[0].line <= last), "") if found else ""
        # A figure function of derive.py (core/analysis_frame.py) is repaired as a function of derive.py, never as a script.
        prompt = (_skeleton.derive_prompt(code, prompt, body) if path.name == _analysis_frame.DERIVE_FILE
                  else _skeleton.framed_prompt(code, prompt))
        try:
            text = await self._chat(prompt, node="implement_figures")
        except Exception as exc:  # noqa: BLE001 -- a repair is best-effort
            self._log.warning("[execute] the call to redraw the figures from the results failed (%r); keeping %s as written",
                              exc, path.name)
            return False
        parsed = _repair_reply_json(text, node="implement_figures")
        new_code = parsed.get("code")
        if not (isinstance(new_code, str) and new_code.strip()):
            new_code, _deps = _parse_implement_response(text)
        # The same script, redrawn: one that lost what FI reads from it (its result line, the oracle branch, the
        # seed, the simulation's entry points) is another script and is not used.
        self._keep_repair_answer("figures", text)
        new_code, split_why = self._script_from_reply("execute", code, new_code, parsed, path.name, self._code_rel(path))
        broken = split_why or _trial_runner.repair_breaks_contract(code, new_code, path.name)
        usable = not broken and all(mark in new_code for mark in _figdata.KEEP_MARKS if mark in code) and (
            not _code_review.prints_result_json(code) or _code_review.prints_result_json(new_code))
        if not usable:
            self._log.warning("[execute] the redraw of %s is not the same script with its figures drawn from the results "
                              "(%s); keeping it as written", path.name, broken or "it lost a part FI reads from the script")
            await self._keep_refused_repair("figures", path.name, new_code,
                                            broken or "it lost a part FI reads from the script")
            return False
        left = _figdata.typed_series_of_script(new_code, path.name) or _figdata.typed_series(new_code, path.name)
        if len(left) >= len(found):
            self._log.warning("[execute] the rewritten %s still draws %d figure series from typed-in numbers; keeping it "
                              "as written", path.name, len(left))
            await self._keep_refused_repair("figures", path.name, new_code,
                                            f"{len(left)} figure series still drawn from typed-in numbers")
            return False
        # A figure repair changes how a figure is drawn, never what the study computes: the checks were made on that
        # computation (core/plot_scope.py). A reply that changed any of it is not used.
        changed = _plot_scope.computation_changes(code, new_code)
        if changed:
            self._log.warning("[execute] the redraw of %s also changes what it computes (%s); keeping it as written",
                              path.name, "; ".join(changed[:4]))
            return False
        path.write_text(new_code, encoding="utf-8")
        self._log.info("[execute] rewrote %s so its figures come from the run's results (%d typed-in series left): %s",
                       path.name, len(left), str(parsed.get("patch_summary") or "no summary")[:120])
        return True

    def _drop_typed_figures(self, figures: list[str], records_dir: Path) -> list[str]:
        """The run's figures without those still drawn from typed-in numbers after the repairs: removed from ``figures/``
        (and their records), said in run.log and in ``needs/FIGURE_DATA_CHECK.json``, which the writer's note reads. An
        earlier pass's record goes first: it is this run's alone."""
        record_path = self.quest_root / "needs" / "FIGURE_DATA_CHECK.json"
        record_path.unlink(missing_ok=True)
        typed = getattr(self, "_typed_figures", None) or {}
        self._typed_figures = {}
        if not typed:
            return figures
        wherever = [f for found in typed.values() for f in found]
        drop: set[str] = set()
        for found in typed.values():
            drop |= set(_figdata.figures_to_drop(found, figures))
        removed = [n for n in figures if Path(n).name in drop]
        from .plot_style import remove_paper_copies

        for name in removed:
            targets = [self.quest_root / "figures" / name]
            # The figure's record and its paper copy are shared by its other formats: they go with the last of them.
            if not any(Path(n).stem == Path(name).stem and n not in removed for n in figures):
                targets += [records_dir / f"{Path(name).stem}.json", self.quest_root / "figures" / f"{Path(name).stem}.json"]
                remove_paper_copies(self.quest_root / "figures", name)
            for target in targets:
                try:
                    target.unlink(missing_ok=True)
                except OSError as exc:
                    self._log.warning("[execute] could not remove %s (%r)", target.name, exc)
        note = _figdata.plain_note(sorted(removed), wherever)
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(_figdata.record(sorted(removed), wherever), indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/FIGURE_DATA_CHECK.json (%r)", exc)
        self._log.warning("[execute] %s The paper does not use %s and says so in its limitations.", note,
                          "it" if len(removed) == 1 else "them")
        return [n for n in figures if n not in removed]

    def _typed_figures_note(self) -> str:
        """For the writer: figures left out because they showed typed-in numbers, so the limitations say so."""
        record = _read_json_or_none_path(self.quest_root / "needs" / "FIGURE_DATA_CHECK.json")
        note = record.get("note") if isinstance(record, dict) else None
        if not isinstance(note, str) or not note.strip():
            return ""
        return (f"{note} Do not show, describe or quote anything from "
                f"{'that figure' if len(record.get('removed_figures') or []) == 1 else 'those figures'}: say plainly, in "
                "the limitations, that a figure was left out because it was not drawn from the run's results.")

    def _record_contradiction_hits(self, paper_md: str, state: QuestState) -> list[str]:
        """Must-fix hits for what the paper claims against FI's own records (core/record_claims.py). FI's own "Best design
        found" section is left out (it is FI's record, printed). Never raises: a check that cannot be made flags nothing.

        Once the paper has been written again ``_RECORD_REWRITES`` times for such hits, what is still flagged is taken out
        of the paper instead (:meth:`_take_out_contradicted`): the check reads free text and has flagged sentences that say
        what the record says, so it never stops the quest; a claim the records contradict is not delivered either way."""
        found = self._record_contradictions(paper_md, state)
        if not found or int(state.get("record_rewrites") or 0) < _RECORD_REWRITES:
            return [hit for hit, _span in found]
        # The check names the first sentence of each kind it finds: what it flags once that one is out is taken out too,
        # a bounded number of times.
        for _ in range(_TAKE_OUT_PASSES):
            taken = self._take_out_contradicted(state, found)
            left = [hit for hit, span in found if span not in taken]
            path = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
            if not taken or not path or not Path(path).is_file():
                return self._record_claims_left(left)
            found = self._record_contradictions(Path(path).read_text(encoding="utf-8"), state)
            if not found:
                self._clear_cannot_repair("record_claims")
                return []
        return self._record_claims_left([hit for hit, _span in found])

    def _record_claims_left(self, left: list[str]) -> list[str]:
        """What the claim check still flags after the rewrites and the take-out passes: recorded, so QUALITY.md lists it (a
        reading of the paper's words, so it never stops the quest, :mod:`core.stop_power`)."""
        self._clear_cannot_repair("record_claims")
        for hit in left:
            self._record_cannot_repair(
                "the paper still has a sentence FI's records contradict: " + _one_line(str(hit), 300),
                [f"wrote the paper again {_RECORD_REWRITES} times, naming the contradiction",
                 "took out what could be taken out"], source=_stop_power.READ, check="record_claims")
        return left

    def _take_out_contradicted(self, state: QuestState, found: list[tuple[str, str]]) -> list[str]:
        """The paper with each flagged sentence (or figure) taken out (core/record_claims.py::take_out), written to disk and
        listed in ``.fi/record_claims_taken_out.json`` (QUALITY.md prints it): the spans taken out. The claim check's
        receipt moves to the new draft, which only lost text and added no claim."""
        path = _rerun_from.in_quest(state.get("paper_md"), self.quest_root)
        if not path or not Path(path).is_file():
            return []
        try:
            before = Path(path).read_bytes()
            text, taken = _record_claims.take_out(before.decode("utf-8"), [span for _hit, span in found])
            if not taken:
                return []
            Path(path).write_bytes(text.encode("utf-8"))  # bytes back as bytes (bytes back as bytes: text mode writes the draft's CRLF as CR CR LF on Windows)
            _receipts.carry_over(self.quest_root, "claim_check", "paper", before, Path(path).read_bytes(),
                                 "FI took out sentences its records contradict and added no claim")
            self._prune_ledger_after_take_out(list(taken), text)
            record = self.quest_root / _record_claims.TAKEN_OUT
            record.parent.mkdir(parents=True, exist_ok=True)
            # Added to what earlier passes took out (a later review's pass, the next sentence of a kind): never replaced.
            earlier = _read_json_or_none_path(record)
            entries = [e for e in (earlier.get("taken_out") or []) if isinstance(e, dict)] \
                if isinstance(earlier, dict) else []
            seen = {str(e.get("text")) for e in entries}
            entries += [{"text": span, "why": hit} for hit, span in found if span in taken and span not in seen]
            record.write_text(json.dumps({"at": time.time(), "taken_out": entries}, indent=1, ensure_ascii=False) + "\n",
                              encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            self._log.warning("[record_check] the sentences FI's records contradict could not be taken out (%s)", e)
            return []
        self._log.warning("[record_check] the paper was written again %d times and still says what FI's records "
                          "contradict, so %d sentence(s) or figure(s) were taken out of it; they are listed in QUALITY.md",
                          _RECORD_REWRITES, len(taken))
        return taken

    def _record_contradictions(self, paper_md: str, state: QuestState) -> list[tuple[str, str]]:
        """:func:`core.record_claims.found` for this paper and quest: each hit with the text it is about."""
        try:
            recs = _best_report.records(self.quest_root)
            text = _paper_facts.without_blocks(
                _best_report.strip_for_checks(paper_md, self._best_design_section(state, paper_md)))
            hits = _record_claims.found(
                text, best_design=recs["best_design"], optimum_check=recs["optimum_check"],
                oracle=self._oracle_record_read())
        except Exception as exc:  # noqa: BLE001 -- a check that cannot be made flags nothing
            self._log.warning("[record_check] could not compare the paper with FI's records (%r)", exc)
            return []
        for hit, _span in hits:
            self._log.warning("[record_check] %s", hit)
        return hits

    def _ran_once_note(self) -> str:
        """For the analysis and the paper: the frozen protocol still says N runs per setting, but FI ran each setting
        once (a simulation with no randomness), so the text must say what ran. The run check's record is written again
        on every run, so a note there is this run's. Empty when there is none."""
        record = _read_json_or_none_path(self.quest_root / "needs" / "RUN_MANIFEST_CHECK.json")
        note = record.get("note") if isinstance(record, dict) else None
        if not isinstance(note, str) or not note.strip():
            return ""
        return (f"{note[0].upper()}{note[1:]}. Report one run per setting: do NOT write that each setting was repeated, "
                "and do NOT report a mean over repeats, a standard error or a confidence interval.")

    def _runs_once_per_setting(self) -> bool:
        """Whether FI ran the simulation once per setting: it defines ``run_cell`` and no ``run_trial``."""
        entries = getattr(self, "_trial_entries", None) or set()
        return bool(getattr(self, "_trial_mode", False)) and "run_cell" in entries and "run_trial" not in entries

    async def _run_cell_randomness(self, state: QuestState, python: Path | str,
                                   env: dict[str, str] | None) -> tuple[str, str, str]:
        """``(why the simulation may draw random numbers, what FI checked, what a study with none changes)`` for a
        simulation FI ran once per setting under a protocol that asked for several runs per setting: its code names a
        source of random numbers, or a setting called again returns other numbers (or could not be called again). The
        first is empty when FI found no sign of randomness, or nothing needs checking. No setting is called again for a
        cluster job, whose one setting may take hours: there only the code is read. The answer is kept for as long as
        FI's record of the trials is the same one (an analysis repaired on the same trials does not call them again)."""
        protocol = self._protocol_block(state) or {}
        runs = protocol.get("runs_per_setting")
        if (not self._runs_once_per_setting() or not _run_manifest.checkable(protocol)
                or not isinstance(runs, (int, float)) or isinstance(runs, bool) or runs <= 1):
            return "", "", ""
        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME
        found = _random_source_found(simulate)
        if found.startswith("`"):
            return (f"FI found {found}, a source of random numbers", "",
                    "take that out of the code (a docstring or a string counts too)")
        if found:
            return found, "", "make sure the code can be read, then resume"
        if self.config.execution.background_jobs:
            return "", "none in its code; a cluster job's settings are not run again to check", ""
        try:
            record_text = (self.quest_root / _trial_runner.RUN_RECORD).read_bytes()
            key = hashlib.sha256(record_text + simulate.read_bytes()).hexdigest()
        except OSError:
            key = ""
        cached = getattr(self, "_run_cell_again_cache", None)
        if key and isinstance(cached, tuple) and cached[0] == key:
            return cached[1]
        self._log.info("[run_manifest] the protocol asks for %g runs per setting and simulate.py defines run_cell: "
                       "calling a setting again to check that it returns the same numbers", runs)
        try:
            same, why = await self._await_with_heartbeat(_trial_runner.run_cell_again(
                self.executor, python, self.quest_root, f"code/{_split_run.SIMULATE_NAME}",
                timeout_s=self.config.execution.timeout_s, env=env,
                thresholds=protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None,
                in_each_run=_in_run.of_study(protocol), log=self._log,
            ), label="calling a setting of simulate.py again")
        except Exception as e:  # noqa: BLE001 -- the run itself succeeded; this check must not end the quest
            same, why = None, f"{type(e).__name__}: {e}"[:300]
        if same:
            answer = ("", "none in its code, and settings run a second time returned the same numbers", "")
        elif same is False:
            answer = (f"FI called it a second time and {why}", "",
                      "make run_cell return the same numbers on every call (no timings, a fixed seed for any library "
                      "that samples)")
        else:
            answer = (f"FI could not call it a second time to check that it returns the same numbers ({why})", "",
                      "make sure run_cell can be called on its own for one setting, or resume so that FI checks again")
        if key and same is not None:
            self._run_cell_again_cache = (key, answer)
        return answer

    def _run_manifest_problems(self, state: QuestState, split: bool, result: Any,
                               run_cell_random: tuple[str, str, str] | None = None) -> tuple[str, list[str]]:
        """``(status, differences)`` of the finished first run: what its manifest says against the frozen protocol.
        Statuses other than ``ok`` and ``differs`` say why nothing was compared. ``run_cell_random``: what
        :meth:`_run_cell_randomness` found (``None`` when it was not asked, which holds the run to the protocol's count)."""
        self._manifest_failed_trials: int | None = None  # None: no manifest was read, so nothing corroborates a 0
        self._manifest_analysis_problems: list[str] = []
        # The findings of numbers FI's trials never produced (core/run_manifest.py FABRICATED): never record-only.
        self._manifest_fabricated: list[str] = []
        # The findings that only say the analysis printed no per-trial list for a mean FI holds trial by trial: kept apart
        # from the differences of the record (core/run_manifest.py UNPRINTED).
        self._manifest_unprinted: list[str] = []
        self._manifest_differences: list[dict[str, Any]] = []  # what FI counted: where the analysis lists more than ran
        self._manifest_note = ""  # a plain sentence for the record, when the run was compared in a way worth saying
        if self.config.engine.run_manifest_check == "off":
            return "off", []
        protocol = self._protocol_block(state)
        if protocol is None or not _run_manifest.checkable(protocol):
            return "not_applicable", []
        if _optimise.block_of(protocol) is not None:
            # A search for the best design has no grid of settings to count: FI ran every evaluation itself, within the
            # budget it counted, and wrote their record (core/optimise.py).
            return "not_applicable", []
        if not split:
            # One script (split_analysis: false, a reply that held one, or a quest begun before every simulation got its
            # own script): no manifest. That is a gap for a stochastic study; a deterministic one (an ODE sweep) has no
            # per-trial outcomes to record.
            return ("single_script" if _split_run.design_is_stochastic(state.get("design")) else "not_applicable"), []
        if not self._run_produced(result):
            return "not_run", []
        result_json = _extract_result_json(getattr(result, "stdout", "") or "")
        if getattr(self, "_trial_mode", False):
            # FI ran the trials and wrote the ledger: its rows are the record, counted against the frozen protocol.
            ledger = _trial_runner.read_ledger(self.quest_root)
            if ledger is None:
                return "differs", ["FI's own trial record (raw/ledger.jsonl) is missing: the trials did not run through FI"]
            thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
            manifest, row_problems = _run_manifest.manifest_from_ledger(protocol, ledger, thresholds)
            from . import run_estimate as _estimate_record

            cut = _estimate_record.read_record(self.quest_root).get("truncated")
            grid_cut = (isinstance(cut, dict) and isinstance(cut.get("planned_settings"), int)
                        and cut["planned_settings"] < int(cut.get("settings") or 0))
            if grid_cut:
                # More settings than FI lists: only the first of the plan's settings were planned and run (the paper says
                # so, core/disclosure.py::cut_short_text). The run cannot cover the whole grid, so it is not compared with
                # it (the manifest's coverage of the grid is left out); every value the analysis reports is still held to
                # the trials FI ran, one by one, below.
                self._manifest_note = (f"only the first {cut['planned_settings']:,} of the plan's {int(cut['settings']):,} "
                                       "settings were planned, so the run was not compared with the whole grid")
            # FI holds every trial's value: a list the analysis reports for one of the trials' own outcomes is checked
            # value by value, not only counted.
            per_cell, altered = _trial_runner.recorded_values_by_cell(self.quest_root)
            recorded = _trial_runner.pooled(per_cell)
            not_run = (_trial_runner.reported_values_not_run(recorded, result_json)
                       + _trial_runner.given_values_not_run(protocol, per_cell, result_json))
            # Research: a mean over a subset of the trials is recomputed from FI's own rows, trial by trial (the
            # subset's membership and value of the same trial), not from two separate pools of values.
            row_analysis, row_sim = self._given_row_findings(protocol, ledger, result_json)
            not_run = not_run + row_analysis
            # run_cell: FI ran each setting once. With no randomness a repeat returns the same numbers, so the
            # protocol's runs_per_setting is not a count this run can miss (core/run_manifest.py::once_per_setting).
            counted, random_run_cell = protocol, ""
            if self._runs_once_per_setting():
                source, checked, without = run_cell_random or (
                    "FI did not call it a second time to check that it returns the same numbers", "",
                    "resume, so that FI checks it")
                counted, self._manifest_note, random_run_cell = _run_manifest.once_per_setting(
                    protocol, random_source=source, checked=checked, **({"without": without} if without else {}))
                if self._manifest_note:
                    self._log.info("[run_manifest] %s", self._manifest_note)
            # A measurement the trials make once and return under every value of a setting is owed once.
            copied = _trial_runner.copied_settings(
                self.quest_root, {**_optimise_search.study_grid(counted),
                                  **(counted.get("in_each_run") if isinstance(counted.get("in_each_run"), dict) else {})},
                [str(m.get("id")) for m in counted.get("metrics") or [] if isinstance(m, dict) and m.get("id")])
            found = (row_problems + altered + ([random_run_cell] if random_run_cell else [])
                     + ([] if grid_cut else _run_manifest.problems(
                         counted, manifest, result_json=result_json, trial_mode=True,
                         recorded=recorded, recorded_by_cell=per_cell, copied=copied)) + not_run
                     + row_sim)
            # A list printed under the wrong name that holds another output's values at its setting is counted as that
            # output, and said once.
            renamed = _run_manifest.relabelled(counted, result_json, per_cell)
            if renamed:
                self._log.info("[run_manifest] %s", "; ".join(renamed))
            derived = _run_manifest.derived_metrics(counted, recorded)
            if derived:
                # Said plainly, so a person sees which means FI could not count against the trials' own values.
                note = ("the analysis derives " + ", ".join(f"`{m}`" for m in derived) + " itself (no trial returns it "
                        "under that name), so no trial count is claimed for it")
                self._manifest_note = (self._manifest_note + "; " if self._manifest_note else "") + note
                self._log.info("[run_manifest] %s", note)
            self._manifest_failed_trials = _run_manifest.failure_count(manifest)
            # Values the analysis made up are the analysis's to fix; a record changed on disk is not (the trials run
            # again).
            # A paired design joins its trials by id in research; with none it would join them by position.
            no_ids = (_metric_spec.missing_pair_ids(protocol, result_json, self.quest_root)
                      if self.config.rigor_profile == "research" else [])
            found = found + no_ids
            self._manifest_analysis_problems = (
                ([] if grid_cut else _run_manifest.analysis_output_problems(
                    counted, manifest, result_json, trial_mode=True, recorded=recorded,
                    recorded_by_cell=per_cell, copied=copied))
                + not_run + no_ids
            )
            self._manifest_differences = [] if grid_cut else _run_manifest.differences(
                counted, manifest, result_json, recorded=recorded, recorded_by_cell=per_cell, copied=copied)
            # The kind each finding was given where it was made; from here on the findings are plain sentences.
            self._manifest_fabricated = [str(f) for f in found if _run_manifest.kind_of(f) == _run_manifest.FABRICATED]
            self._manifest_unprinted = [str(f) for f in found if _run_manifest.kind_of(f) == _run_manifest.UNPRINTED]
            self._manifest_analysis_problems = [str(f) for f in self._manifest_analysis_problems]
            return ("differs" if found else "ok"), [str(f) for f in found]
        if self.config.rigor_profile == "research":
            # The older contract: the simulation ran its own loop and wrote its own record. Research needs FI's record,
            # so the repair asks for the trial contract, and after it the quest stops.
            return "differs", [
                "simulate.py runs its own loop and writes its own trial record; this quest needs FI to run the trials: "
                "simulate.py must define run_trial(cell, trial_id, seed) (or run_cell(cell) for a deterministic study) "
                "that returns a dict of numbers for one trial, with no loop, no files and no RESULT_JSON, and "
                "experiment.py must read FI's per-setting results from the file FI_TRIALS names"
            ]
        raw_dir = _split_run.raw_dir_for(self._raw_root(), 0)
        # v2: a per-trial ledger, when the script kept one, is counted by the engine instead of trusting a
        # self-reported summary -- the ledger row problems (an out-of-protocol cell, a duplicate trial id, a row
        # missing its status) come first, since those are wrong before the derived manifest is even compared with
        # the protocol. thresholds_used has no natural home in a ledger row, so it still comes from run_manifest.json
        # when the script also wrote one.
        ledger, ledger_why = _run_manifest.read_ledger(raw_dir)
        if ledger is not None:
            legacy, legacy_why = _run_manifest.read(raw_dir)
            thresholds_used = (legacy or {}).get("thresholds_used") if isinstance(legacy, dict) else None
            manifest, row_problems = _run_manifest.manifest_from_ledger(protocol, ledger, thresholds_used)
            # The two-script contract asks for BOTH files; a ledger with no run_manifest.json at all is
            # still a contract break, not something the ledger's own presence excuses.
            if legacy is None:
                row_problems = [f"run_manifest.json: {legacy_why}", *row_problems]
            found = row_problems + _run_manifest.problems(protocol, manifest, result_json=result_json)
        else:
            manifest, why = _run_manifest.read(raw_dir)
            found = _run_manifest.problems(protocol, manifest, why, result_json=result_json)
        self._manifest_failed_trials = _run_manifest.failure_count(manifest) if manifest is not None else None
        self._manifest_analysis_problems = _run_manifest.analysis_output_problems(protocol, manifest, result_json)
        # The older contract's record is the simulation's own statement, never the level FI's own record reaches.
        return ("differs" if found else "self_reported"), found

    def _grid_values_left_out(self, state: QuestState) -> list[str]:
        """Grid values the analysis script names only some of (a ``for`` loop over literals, which the protocol check does
        not stop on under the trial contract) that FI's own count of the run shows it reports no trials for. Empty when
        there is no such loop, or the analysis reports every setting."""
        protocol = self._protocol_of(state)
        if protocol is None or not getattr(self, "_trial_mode", False):
            return []
        suspects = _protocol.analysis_suspects(protocol, self._scripts_on_disk())
        return _run_manifest.confirmed(suspects, list(getattr(self, "_manifest_differences", None) or [])) if suspects else []

    def _manifest_stop(self, found: list[str], attempts: int) -> QuestState:
        """The run's record still differs from the frozen protocol after the repairs (``engine.run_manifest_check: block``):
        the quest ends in the honest stop (no paper), with what differed and how often the script was sent back. Nobody is
        asked to change a script, the protocol or a setting."""
        analysis_only = set(found) <= set(getattr(self, "_manifest_analysis_problems", None) or [])
        trials = bool(getattr(self, "_trial_mode", False))
        said = ("the analysis does not report the values the protocol's metrics need" if analysis_only else
                "the simulation does not do what the protocol fixed")
        differences = list(getattr(self, "_manifest_differences", None) or []) if analysis_only else []
        plain = _run_manifest.plain_difference(differences)  # the first sentence a scientist reads; the numbers follow it
        return self._stop_before_run(
            (f"{plain} Details: " if plain else f"{said}: ") + "; ".join(found),
            "cannot_repair",
            tried=[
                ("compared FI's record of the trials it ran (`raw/ledger.jsonl`) with the frozen protocol" if trials
                 else "compared the run's own manifest with the frozen protocol"),
                *[_one_line(f, 400) for f in found[:3]],
                f"sent {_split_run.ANALYSIS_NAME if analysis_only else _split_run.SIMULATE_NAME} back to be repaired "
                f"{attempts} time(s) with the missing items named; the difference stood",
            ])

    def _check_replicate_manifests(self, state: QuestState, split: bool, replicates: int) -> None:
        """The other seeds' manifests against the same protocol. A difference is recorded (and is a gap of the evidence
        level), not repaired: the primary run was already held to it."""
        if not split or replicates < 2:
            return
        record = _read_json_or_none_path(self.quest_root / "needs" / "RUN_MANIFEST_CHECK.json")
        # Under warn the first run's difference was recorded, not repaired: the other seeds are compared all the same.
        if not isinstance(record, dict) or record.get("status") not in ("ok", "warned"):
            return
        protocol = self._protocol_block(state)
        if protocol is None:
            return
        seeds: dict[str, list[str]] = {}
        for index in range(1, replicates):
            raw_dir = _split_run.raw_dir_for(self._raw_root(), index)
            if not raw_dir.is_dir():
                continue  # that replicate was not run (a script with no random source stops after two seeds)
            manifest, why = _run_manifest.read(raw_dir)
            found = _run_manifest.problems(protocol, manifest, why)
            if found:
                seeds[str(index)] = found
        if seeds:
            record = {**record, "status": "warned" if record.get("status") == "warned" else "differs_in_replicates",
                      "seeds": seeds}
            self._log.warning(
                "[run_manifest] %d replicate(s) differ from the frozen protocol: %s", len(seeds),
                "; ".join(f"seed {k}: {v[0]}" for k, v in seeds.items()),
            )
            self._run_manifest_record(record)

    def _split_contract_broken(self, found: list[str], *, what: str, source: str) -> None:
        """A quest that should keep two scripts got scripts that break the contract, or a reply without both: ``warn`` says
        so and goes on, ``block`` records it with what it rests on (``source``): a reply without both scripts (measured)
        stops the quest (a resume asks for the scripts again); a contract FI read from the code is recorded and the
        quest goes on (:mod:`core.stop_power`)."""
        if self.config.execution.split_failure != "block":
            self._log.warning("[implement] %s: %s", what, "; ".join(found))
            return
        self._record_cannot_repair(
            what[0].upper() + what[1:] + ": " + "; ".join(found),
            ["asked for both scripts, and once more when the reply did not hold both (`execution.split_failure: block`)",
             *[_one_line(f, 300) for f in found[:3]]], source=source, check="split")

    def _pause_for_contract(self, *, kind: str, headline: str, steps: list[str], problems: list[str]) -> None:
        self._pause_for_human(
            kind=kind,
            interaction="supply",
            headline=headline,
            steps=steps,
            payload={"contract_stage": True, "quest_id": self.quest_id, "problems": problems},
        )

    def _raw_root(self) -> Path:
        return _split_run.raw_root_of(self.quest_root, self.config.execution.raw_dir)

    def _settle_pending_amendment(self, state: QuestState) -> dict[str, Any] | None:
        """A resume after an amendment request: the design that asked for it, with the approved protocol when a person
        approved that request and with the frozen one when they did not (resuming is not approving). ``None`` when no
        request is waiting.

        A tamper-recovery request (:attr:`_frozen.TAMPER_SOURCE`) is left alone here: resuming without approving it must
        keep the quest blocked, not fall back to "the frozen one" the way a declined ordinary amendment does — the frozen
        record is exactly what is not trusted. :meth:`_resolved_frozen` (through ``_hold_design_to_frozen`` /
        ``_protocol_block``) is what settles it."""
        pending = _frozen.load_pending(self.quest_root)
        if pending is None or pending.get("source") == _frozen.TAMPER_SOURCE:
            return None
        design = dict(pending.get("design") or {})
        approval = _frozen.approval_for(self.quest_root, pending)
        if approval is not None:
            record = _frozen.apply(self.quest_root, pending, approval, raw_root=self._raw_root())
            design["protocol"] = pending["proposed_protocol"]
            self._log.warning(
                "[protocol] amendment %d approved by %s (%s): %s%s",
                record["n"], record["approved_by"],
                "made after results were seen" if not record["prespecified"] else "before any result",
                "; ".join(record["changes"]) or "no listed change",
                f"; the earlier run is archived in {record['archived_to']}" if record.get("archived_to") else "",
            )
            return design
        _frozen.decline(self.quest_root, pending)
        frozen_protocol = _frozen.protocol_of(self.quest_root)
        if frozen_protocol:
            design["protocol"] = frozen_protocol
        else:
            design.pop("protocol", None)
        self._log.warning(
            "[protocol] the amendment request was resumed past without an approval; the frozen protocol stays: %s",
            "; ".join(pending.get("changes") or []),
        )
        return design

    def _hold_design_to_frozen(self, state: QuestState, design: dict[str, Any]) -> dict[str, Any]:
        """A design made after the freeze keeps the frozen protocol: without one it gets it back, with the same one it is
        left alone, and with a different one it is an amendment request, which stops the quest for a person."""
        frozen = self._resolved_frozen()
        if frozen is None or not isinstance(design, dict):
            return design
        request = design.get("protocol_amendment")
        design = {k: v for k, v in design.items() if k != "protocol_amendment"}
        asked = request.get("protocol") if isinstance(request, dict) and isinstance(request.get("protocol"), dict) else None
        proposed = asked or (design.get("protocol") if isinstance(design.get("protocol"), dict) and design["protocol"] else None)
        def _frozen_design() -> dict[str, Any]:
            if frozen.get("protocol"):
                return {**design, "protocol": frozen["protocol"]}
            return {k: v for k, v in design.items() if k != "protocol"}

        if proposed is None or _frozen.sha256(proposed) == frozen.get("sha256"):
            # No protocol, or the frozen one: the frozen protocol is what the design carries.
            return _frozen_design()
        if asked is None:
            # A protocol that differs without an amendment request: the design, or the methodology audit that amends it
            # in place, rewrote it. That is not a request a person is asked about; the frozen protocol stands, and the
            # log says what the design wanted.
            self._log.warning(
                "[protocol] the redesign changed the protocol without asking for an amendment (%s); the frozen protocol stands",
                "; ".join(_frozen.diff(frozen.get("protocol"), proposed)[:4]) or "no listed difference",
            )
            return _frozen_design()
        iteration = int(state.get("iteration", 0) or 0)
        reason = str(request.get("reason") or "").strip() if isinstance(request, dict) else ""
        reason = reason or f"the redesign at iteration {iteration} asked for this change without giving a reason"
        if proposed.get("criteria") not in (None, "", []):
            rest = {k: v for k, v in proposed.items() if k != "criteria"}
            raw = proposed["criteria"] if isinstance(proposed["criteria"], list) else [proposed["criteria"]]
            _kept, dropped_c = _criteria.repair(raw, rest)
            if dropped_c:
                # The entries FI can use, as they were written (a clean copy would show as changes that are not).
                usable = [item for i, item in enumerate(raw, start=1) if _criteria.normalize([item], rest)[0]]
                proposed = {**rest, "criteria": usable[:_criteria.MAX]} if usable else rest
                reason += " (FI left out what it cannot use as a criterion: " + "; ".join(dropped_c) + ")"
                for note in dropped_c:
                    self._log.warning("[criteria] the amendment request: %s", note)
                if _frozen.sha256(proposed) == frozen.get("sha256"):
                    self._log.warning("[criteria] the amendment request changed nothing FI can use; the frozen protocol "
                                      "stands")
                    return _frozen_design()
        design = {**design, "protocol": proposed}
        pending = _frozen.propose(
            self.quest_root, proposed, design, source=f"redesign at iteration {iteration}", reason=reason,
            results_seen=iteration > 0 or bool(state.get("result_json")),
        )
        self._pause_for_amendment(pending)
        return design  # not reached: the pause exits

    def _amend_for_flat_objective(self, state: QuestState, design: dict[str, Any]) -> dict[str, Any]:
        """The plan rewritten from what FI measured of a search whose objective did not move, made the frozen protocol:
        an amendment FI approves itself (the maintainer's decision: unattended and fleet runs get the revision too), not
        pre-specified, with FI's fact as its reason; the earlier run is archived (``frozen_protocol.apply``) and the
        paper says so (``frozen_protocol.disclosure``; the evidence ladder names it, ``core/evidence.py``). With no
        frozen protocol yet (the explore stage of ``engine.phased``) there is nothing to amend: the design history says
        the change came after results."""
        why = (_flat_fact_sentence(state.get("flat_objective_fact") or {})
               + "; plan.md was rewritten once from what FI measured")
        return self._amend_by_fi(design, source=_FLAT_SOURCE, approver=_FLAT_APPROVER, why=why, results_seen=True)

    def _amend_to_fit(self, design: dict[str, Any]) -> dict[str, Any]:
        """The plan rewritten to fit the time FI measured, made the frozen protocol when one is frozen: an amendment FI
        approves itself. Made before results only when no result of the study was recorded (FI's own ledgers of the trials
        and of a search are empty); otherwise it is said to be made after them, the earlier run is archived and the paper
        is told, as for any amendment."""
        return self._amend_by_fi(design, source=_FIT_SOURCE, approver=_FIT_APPROVER,
                                 why="the study did not fit the time allowed; plan.md was rewritten once to fit the time FI "
                                     "measured", results_seen=self._study_results_recorded())

    def _study_results_recorded(self) -> bool:
        """Whether FI's record of the study holds a result: a finished trial in the trials' ledger, or an evaluation in the
        search's."""
        raw = self._raw_root()
        for path, event in ((raw / _trial_runner.LEDGER_NAME, "trial"), (raw / _optimise.LEDGER_PATH.name, "evaluation")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("event") == event and row.get("status") == "ok":
                    return True
        return False

    def _amend_by_fi(self, design: dict[str, Any], *, source: str, approver: str, why: str,
                     results_seen: bool) -> dict[str, Any]:
        """The design's protocol made the frozen one by an amendment FI approves itself, with ``why`` as its reason;
        unchanged when nothing is frozen or nothing changed. ``results_seen``: the earlier run is archived
        (``frozen_protocol.apply``) and the paper says the change came after results."""
        frozen = self._resolved_frozen()
        proposed = design.get("protocol") if isinstance(design, dict) and isinstance(design.get("protocol"), dict) else None
        if frozen is None or not proposed or _frozen.sha256(proposed) == frozen.get("sha256"):
            return design
        pending = _frozen.propose(self.quest_root, proposed, design, source=source, reason=why, results_seen=results_seen)
        _frozen.approve(self.quest_root, approver, via="fi")
        approval = _frozen.approval_for(self.quest_root, pending)
        if approval is None:  # pragma: no cover -- approve() wrote it just above
            raise RuntimeError("FI could not record its approval of the change to the frozen protocol")
        record = _frozen.apply(self.quest_root, pending, approval, raw_root=self._raw_root())
        self._log.warning(
            "[protocol] amendment %d approved by FI, made %s results were seen: %s; %s%s", record["n"],
            "after" if results_seen else "before", why, "; ".join(record["changes"][:6]) or "no listed change",
            f"; the earlier run is archived in {record['archived_to']}" if record.get("archived_to") else "")
        return {**design, "protocol": proposed}

    def _pause_for_amendment(self, pending: dict[str, Any]) -> None:
        changes = pending.get("changes") or []
        steps = [
            "The redesign changes the protocol that was frozen before the first full run: " + ("; ".join(changes) or "(no listed change)") + ".",
            f"Read the request in `{_frozen.pending_path(self.quest_root)}`. To approve it: "
            f"`python launch.py --approve-amendment \"{self.quest_root}\" --approve-as <you>` (the quest page in the web UI "
            f"has an Approve button; in VSCode: `@fi /approve-amendment {self.quest_id}`), then resume. Resuming WITHOUT approving keeps "
            "the frozen protocol and the study goes on with it.",
            "An approved amendment made after results were seen archives this run under `archive/run_<n>/` and starts a new one, "
            "and the paper says the change was made after the results were known.",
        ]
        self._pause_for_human(
            kind="amendment",
            interaction="supply",
            headline="the redesign changes the frozen protocol",
            steps=steps,
            payload={
                "amendment_stage": True, "quest_id": self.quest_id, "changes": changes,
                "amendment_file": str(_frozen.pending_path(self.quest_root)),
            },
        )

    def _resolved_frozen(self) -> dict[str, Any] | None:
        """The frozen record, guaranteed untampered — never the raw ``needs/FROZEN_PROTOCOL.json`` a person (or a bug)
        could have edited by hand. A record whose hash no longer matches its own content stops the quest here, under
        every profile, until a person approves restoring the version saved right when it was locked
        (``core/frozen_protocol.py:propose_tamper_recovery``, the same approval act as any other change to the frozen
        protocol). Unlike a declined amendment, resuming without approving does not go on: an untrustworthy file has
        nothing safe to fall back to."""
        frozen = _frozen.load(self.quest_root)
        if frozen is None or not frozen.get("problem"):
            return frozen
        pending = _frozen.load_pending(self.quest_root)
        if pending is None or pending.get("source") != _frozen.TAMPER_SOURCE:
            pending = _frozen.propose_tamper_recovery(self.quest_root, frozen)
        approval = _frozen.approval_for(self.quest_root, pending)
        if approval is None:
            self._pause_for_tamper(pending)
            return None  # not reached: the pause exits
        record = _frozen.apply(self.quest_root, pending, approval, raw_root=self._raw_root())
        self._log.warning(
            "[protocol] restored the version saved when the protocol was locked, approved by %s (the saved file no "
            "longer matched its own checksum)", record["approved_by"],
        )
        return _frozen.load(self.quest_root)

    def _pause_for_tamper(self, pending: dict[str, Any]) -> None:
        self._pause_for_human(
            kind="amendment",
            interaction="supply",
            headline="the saved protocol file was changed after it was locked",
            steps=[
                "`needs/FROZEN_PROTOCOL.json` no longer matches its own checksum, so it may not say what was actually "
                "agreed when the protocol was locked. The experiment does not run against a file that might have been edited by hand.",
                f"To restore the version saved right when it was locked: `python launch.py --approve-amendment \"{self.quest_root}\" "
                f"--approve-as <you>` (the quest page's Approve button; in VSCode: `@fi /approve-amendment {self.quest_id}`), then resume.",
                "Resuming WITHOUT approving does not go on: the quest stays paused here until the restore is approved.",
            ],
            payload={
                "amendment_stage": True, "quest_id": self.quest_id, "changes": pending.get("changes") or [],
                "amendment_file": str(_frozen.pending_path(self.quest_root)),
            },
        )

    def _protocol_of(self, state: QuestState) -> dict[str, Any] | None:
        """The protocol the script is compared with (``engine.protocol_check`` not off)."""
        if self.config.engine.protocol_check == "off":
            return None
        return self._protocol_block(state)

    def _submit_on_disk(self) -> str:
        path = self.quest_root / "code" / _trial_runner.SUBMIT_NAME
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def _scripts_on_disk(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in (_split_run.SIMULATE_NAME, "experiment.py"):
            path = self.quest_root / "code" / name
            if path.is_file():
                out[name] = path.read_text(encoding="utf-8")
        return out

    def _study_scripts_on_disk(self) -> dict[str, str]:
        """The study's own files when FI writes the analysis (core/analysis_frame.py): simulate.py and derive.py; experiment.py
        is FI's and is not shown as the study's code."""
        out: dict[str, str] = {}
        for name in (_split_run.SIMULATE_NAME, "derive.py"):
            path = self.quest_root / "code" / name
            if path.is_file():
                out[name] = path.read_text(encoding="utf-8")
        return out

    def _protocol_record(self, payload: dict[str, Any]) -> None:
        try:
            path = self.quest_root / "needs" / "PROTOCOL_CHECK.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            self._audit_check("protocol", path, status=str(payload.get("status")), problems=payload.get("differences"))
        except OSError:
            pass  # a record that cannot be written must never stop a quest

    def _goes_beyond_protocol_by_request(self) -> list[str]:
        """What a person's extension added beyond the frozen protocol (``[]`` when the script was not extended, or the
        extension stayed inside it): the run manifest and the goal-coverage note do not fight it."""
        record = _read_json_or_none_path(self.quest_root / "needs" / "PROTOCOL_CHECK.json")
        if isinstance(record, dict) and record.get("status") == "extended_by_person":
            return [str(d) for d in record.get("differences") or []]
        if isinstance(record, dict):
            # The coding agent's later check keeps the person's additions beside a difference of its own
            # (AgentBuildMixin._agent_protocol).
            return [str(d) for d in record.get("added_by_person") or []]
        return []

    def _protocol_drift_not_asked_for(self, protocol: dict[str, Any]) -> list[str]:
        """The differences from the frozen protocol that a person's extension did not add: what the scripts differ by
        now, less what ``PROTOCOL_CHECK.json`` recorded as the person's request. A constant changed by a repair later
        is still drift."""
        excused = set(self._goes_beyond_protocol_by_request())
        return [m.message() for m in _protocol.check(protocol, self._scripts_on_disk()) if m.message() not in excused]

    async def _enforce_protocol(
        self, state: QuestState, code_path: Path, simulate_path: Path, code: str, deps: list[str],
        *, extended: bool = False,
    ) -> tuple[str, list[str]]:
        """Hold the script to the plan's protocol: repair a script that differs, up to
        ``engine.protocol_repair_attempts`` times, and record what still differs. The comparison reads the script's text
        (``protocol_check.SOURCE``), so a difference that stands ends the quest only if that source may stop one
        (:mod:`core.stop_power`); otherwise it is recorded, the paper says it, and the quest goes on.

        The repair asks for one script at a time and keeps a rewrite only if it parses and the differences named for
        that script are gone, so a repair never leaves a script worse than the one it had. Every attempt is recorded in
        ``needs/PROTOCOL_CHECK.json``."""
        protocol = self._protocol_of(state)
        if protocol is None:
            return code, deps
        if extended:
            # A number a person asked for after reading the result goes beyond the frozen protocol on purpose. A repair
            # would take it out again and a pause would stop the refine, so the differences are recorded, not fought:
            # the frozen protocol itself stays as it is, and the record shows what the person added.
            added = [m.message() for m in _protocol.check(protocol, self._scripts_on_disk())]
            self._protocol_record({
                "status": "extended_by_person", "attempts": [], "differences": added, "protocol": protocol,
            })
            if added:
                self._log.warning(
                    "[protocol] the script was extended at a person's request and now goes beyond the frozen protocol "
                    "(%s); the frozen protocol is unchanged and the difference is recorded", "; ".join(added),
                )
            return code, deps
        attempts: list[dict[str, Any]] = []
        budget = int(self.config.engine.protocol_repair_attempts)
        # A difference with no place named (the script holds no such list at all) belongs where the design's loops are:
        # simulate.py in a two-script quest. Sending it to experiment.py asked the analysis to add a grid it never runs.
        unplaced_to = _split_run.SIMULATE_NAME if simulate_path.is_file() else "experiment.py"
        call_failures_left = 1  # a repair call that got no answer (a provider timeout) is tried again, once, uncounted
        for attempt in range(budget + 1):
            mismatches = _protocol.check_repairable(protocol, self._scripts_on_disk())
            attempts.append({"attempt": attempt, "differences": [m.message() for m in mismatches]})
            if not mismatches:
                break
            if attempt == budget:
                break
            self._log.warning(
                "[protocol] the script differs from the plan's protocol (%d): %s; asking for a repair (%d of %d)",
                len(mismatches), "; ".join(m.message() for m in mismatches), attempt + 1, budget,
            )
            names = {m.where.split(" in ")[-1].split(",")[0] for m in mismatches if m.where}
            names = {n if n in (_split_run.SIMULATE_NAME, "experiment.py") else "experiment.py" for n in names}
            if any(not m.where for m in mismatches):
                names.add(unplaced_to)
            for name in sorted(names):
                path = simulate_path if name == _split_run.SIMULATE_NAME else code_path
                if not path.is_file():
                    continue
                own = [m for m in mismatches if (name in m.where if m.where else name == unplaced_to)]
                text, deps, call_failed = await self._repair_script_for_protocol(state, path, path.read_text(encoding="utf-8"), deps, protocol, own)
                if call_failed and call_failures_left > 0:
                    call_failures_left -= 1
                    text, deps, _ = await self._repair_script_for_protocol(state, path, path.read_text(encoding="utf-8"), deps, protocol, own)
                if name == "experiment.py":
                    code = text
        final = _protocol.check_repairable(protocol, self._scripts_on_disk())
        record = {
            "status": "ok" if not final else (
                "stopped" if self.config.engine.protocol_check == "block" and _stop_power.may_stop(_protocol.SOURCE)
                else "warned"),
            "attempts": attempts,
            "differences": [m.message() for m in final],
            "protocol": protocol,
        }
        notes = [*_protocol.rng_notes(protocol, self._scripts_on_disk()),  # read from free code: a note, never a stop or a repair
                 *_protocol.loop_hints(protocol, self._scripts_on_disk())]
        if notes:
            record["notes"] = notes
            self._log.info("[protocol] %s", "; ".join(notes))
        self._protocol_record(record)
        if not final:
            self._clear_cannot_repair("protocol")
            self._log.info(
                "[protocol] the script holds to the plan's protocol%s",
                "" if len(attempts) == 1 else f" (after {len(attempts) - 1} repair(s))",
            )
            for suspect in _protocol.analysis_suspects(protocol, self._scripts_on_disk()):
                self._log.info("[protocol] the analysis loops over only %s of %s; after the run FI checks, from its own count of "
                               "the trials, that the analysis reports every setting", suspect.found, suspect.name)
            return code, deps
        if record["status"] == "warned":
            self._log.warning("[protocol] the script still differs from the plan's protocol: %s; this is %s, so the run "
                              "goes on and the paper says so", "; ".join(m.message() for m in final),
                              _stop_power.source_words(_protocol.SOURCE))
        self._stop_for_protocol(final)
        return code, deps  # a stopping difference ends the run step in the honest stop without running it

    async def _repair_script_for_protocol(
        self, state: QuestState, path: Path, code: str, deps: list[str],
        protocol: dict[str, Any], mismatches: list[Any],
    ) -> tuple[str, list[str], bool]:
        """ONE repair of one script that contradicts the protocol: ``(code, deps, the repair call itself failed)``. Kept
        only if it parses and no longer contradicts it."""
        prompt = self._prompts["execute_reflect"].substitute(
            topic_block=self._topic_block(), own_failure_block="",
            previous_code=code,
            returncode="(not run yet)",
            stdout_tail=_protocol_directive(protocol, mismatches, frame=path.name == _split_run.SIMULATE_NAME and _skeleton.has_frame(code)),
            stderr_tail="",
            duration_s="0.00",
            figures_count="0",
            result_json_present="no (not run yet)",
            reflect_history_block=_format_reflect_history([]),
            design_block=self._code_design_block(state),
            clarify_block=_format_clarify(state),
        )
        prompt = _skeleton.framed_prompt(code if path.name == _split_run.SIMULATE_NAME else "", prompt)
        try:
            text = await self._chat(prompt, node="implement_protocol")
        except Exception as exc:  # noqa: BLE001 -- a repair is best-effort
            self._log.warning("[protocol] the repair call failed (%r); keeping %s as written", exc, path.name)
            return code, deps, True
        parsed = _repair_reply_json(text, node="implement_protocol")
        new_code, new_deps = parsed.get("code"), _coerce_dep_list(parsed.get("deps"))
        if not (isinstance(new_code, str) and new_code.strip()):
            new_code, new_deps = _parse_implement_response(text)
        self._keep_repair_answer("protocol", text)
        new_code, split_why = self._script_from_reply("protocol", code, new_code, parsed, path.name, self._code_rel(path))
        broken = split_why or _trial_runner.repair_breaks_contract(code, new_code, path.name)
        if broken:
            self._log.warning("[protocol] the repair of %s is not usable (%s)", path.name, broken)
        usable = not broken
        scripts = {**self._scripts_on_disk(), path.name: new_code} if usable else {}
        remaining = [m for m in _protocol.check_repairable(protocol, scripts) if not m.where or path.name in m.where] if usable else mismatches
        if not usable or len(remaining) >= len(mismatches):
            self._log.warning("[protocol] the repair of %s did not remove the differences; keeping it as written", path.name)
            await self._keep_refused_repair(
                "protocol", path.name, new_code,
                broken or f"the differences were not fewer ({len(mismatches)} -> {len(remaining)})")
            return code, deps, False
        package_was = self._package_snapshot()
        path.write_text(new_code, encoding="utf-8")
        if await self._simulation_kept_or_back(state, path, code, package_was, "protocol"):
            return code, deps, False
        self._log.info(
            "[protocol] rewrote %s (%d bytes): %d difference(s) left in it (%s)",
            path.name, len(new_code), len(remaining), str(parsed.get("patch_summary") or "no summary")[:120],
        )
        return new_code, sorted({*deps, *new_deps}), False

    def _comparison_stats(self, state: QuestState, replicates: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        """The per-stratum intervals and effect sizes over the batches (:func:`_result_comparison_stats`), and, when the protocol
        declares metric specs, the engine's own estimates and contrasts with their p-values (``spec_statistics``,
        :mod:`core.metric_spec`). The audits that compare the paper with the numbers read the same dictionary."""
        protocol = self._protocol_block(state)
        metric_ids = {str(m.get("id")).lower() for m in _metric_spec.declared(protocol) if m.get("id")} or None
        out = _result_comparison_stats(replicates, metric_ids=metric_ids, **kw)
        if protocol is not None and len(replicates) >= 1 and _metric_spec.declared(protocol):
            try:
                spec = _metric_spec.statistics(replicates, protocol, progress=self._log.info, quest_root=self.quest_root)
            except Exception as e:  # noqa: BLE001 -- a report about the numbers must never stall the quest
                self._log.warning("[stats] the metric specs' statistics could not be computed: %r", e)
                spec = {}
            if spec:
                out = {**out, "spec_statistics": spec}
        return out

    def _fi_wrote_the_plan_unfrozen(self, design: dict[str, Any]) -> bool:
        """Whether the design about to run is plan.md's design block, that block was last written by FI itself (the model's
        draft or an engine rewrite, never a person's edit or request), and the protocol is not frozen yet. Anything
        else, or anything that cannot be told, is ``False``: the audit then only records, as it always did."""
        if _frozen.load(self.quest_root) is not None:
            return False
        path = _plan.plan_path(self.quest_root)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return False
        rows = _plan.history(self.quest_root)
        if not rows or rows[-1].get("by") not in _FI_PLAN_AUTHORS or rows[-1].get("sha256") != _plan.sha256(text):
            return False  # a hand edit not yet recorded, a person's request, or no history: a person's plan
        planned = _plan.parse(text).design
        return isinstance(planned, dict) and _receipts.design_core(planned) == _receipts.design_core(design)

    def _design_audit_acted_on(self, key: str) -> bool:
        """Whether the audit already acted on the design with this hash (a resume must not act twice)."""
        try:
            done = json.loads((self.fi_dir / "design_audit_acted.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(done, list) and key in done

    def _mark_design_audit_acted(self, key: str) -> None:
        marker = self.fi_dir / "design_audit_acted.json"
        try:
            done = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else []
            done = done if isinstance(done, list) else []
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps([*done, key], indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError) as e:
            self._log.warning("[design] could not record that the audit was acted on (%r)", e)

    def _write_audited_design_to_plan(self, amended: dict[str, Any], note: str) -> dict[str, Any] | None:
        """Write the design the audit amended into plan.md's design block (the way the engine's other plan edits are
        made) and keep that version as FI's own; the design that runs is read back from the file. ``None`` when it could
        not be written, and plan.md is then unchanged."""
        path = _plan.plan_path(self.quest_root)
        try:
            text = path.read_text(encoding="utf-8")
            normalized, why = _plan.normalize_design({k: v for k, v in amended.items() if k != "rationale"})
            if normalized is None:
                self._log.warning("[design] the audit's amended design cannot be used (%s); the design is kept as it was", why)
                return None
            new = _plan.edit_design_block(text, lambda _block: normalized)
            if new is None:
                return None
            new = _plan.refresh_model_section(new)
            parsed = _plan.parse(new)
            if parsed.design is None:
                self._log.warning("[design] plan.md with the audit's change cannot be read (%s); it is unchanged", parsed.error)
                return None
            path.write_text(new, encoding="utf-8")
        except OSError as e:
            self._log.warning("[design] plan.md could not be updated with the audit's change (%r); it is unchanged", e)
            return None
        _plan.record_version(self.quest_root, new, by="engine", note=note[:300])
        return parsed.design

    async def _audit_the_design_that_runs(self, state: QuestState, design: Any) -> Any:
        """The methodology audit's receipt must be for the design that runs. A design a person edited in plan.md, an
        amended one, or one the frozen protocol held changed after the audit saw it: the audit checks it again, taking
        none of its proposals (one more call, only when the design changed).

        The exception: the change was FI's own (it rewrote plan.md itself) and the protocol is not frozen, so nothing has
        run yet. Then the audit's objections are met before anything is implemented: it may amend the design once (per
        design), the amendment is written into plan.md, and the audit looks once more, taking nothing, so the receipt is
        for the design that runs. Returns the design to run."""
        if self.config.engine.analyze_local_first or not isinstance(design, dict):
            return design
        _audit_open.close_held(self.quest_root, design)  # every design that will run: a person's edit may apply an open item
        writer = "plan" if _plan.plan_path(self.quest_root).is_file() else "design"  # the step that wrote the design
        _status, record, problem = _receipts.read(self.quest_root, "design_audit")
        key = _receipts.sha256(_receipts.design_core(design))
        if not problem and record is not None and record.get("output_hash") == key:
            return design
        if self._design_audit_acted_on(key) or not self._fi_wrote_the_plan_unfrozen(design):
            self._log.info("[design] the design that will run is not the one the methodology audit saw; auditing it (no change taken)")
            await self._audit_design(state, design, adopt=False, writer=writer)
            return design
        self._log.info("[design] the design that will run is not the one the methodology audit saw; FI rewrote it "
                       "itself and nothing has run, so the audit may change it once")
        self._mark_design_audit_acted(key)
        amended, objections = await self._audit_design(state, design, adopt=True, writer=writer)
        kept_checks = False
        written: dict[str, Any] | None = None
        if isinstance(amended, dict):
            amended, kept_checks = self._keep_what_the_results_are_judged_by(design, amended)
        if isinstance(amended, dict) and _receipts.design_core(amended) != _receipts.design_core(design):
            first = next((o.get("objection") if isinstance(o, dict) else o for o in objections or [] if o), "")
            written = self._write_audited_design_to_plan(
                amended, "the methodology audit objected to the design before anything ran; FI changed the design: "
                         + str(first)[:200])
            if written is not None:
                self._log.info(
                    "[design] before the experiment ran, a second look at the design found a problem and FI changed "
                    "the design to fix it: %s", str(first)[:160] or "see needs/DESIGN_CRITIQUE.json")
                design = written
        self._reconcile_audit_changes(design, amended, objections, written, iteration=int(state.get("iteration", 0) or 0))
        if kept_checks:
            self._log.info("[design] the audit also suggested changing the checks the results are judged by; FI kept "
                           "them as they were")
        await self._audit_design(state, design, adopt=False, writer=writer)
        if kept_checks:
            _receipts.add_detail(self.quest_root, "design_audit",
                                 "The audit also suggested changing the checks the results are judged by (the protocol's "
                                 "checks, thresholds, acceptance rules, metrics, precision target and model); FI kept "
                                 "them as they were.")
        return design

    def _reconcile_audit_changes(
        self, before: dict[str, Any], amended: Any, objections: Any, written: dict[str, Any] | None, *, iteration: int,
    ) -> None:
        """After the audit's changes were made: compare each with the design FI holds now. ``amended`` is what FI meant to
        write (the checks the results are judged by put back); ``written`` is what plan.md holds when read back, ``None``
        when nothing was written. A proposal missing from ``amended`` was held back on purpose; one in ``amended`` and not
        in ``written`` was lost in the writing. Both become open items (a change reported as made is never only a log line)."""
        if written is not None:
            _audit_open.close_held(self.quest_root, written)
        props = self._search_proposals_by_part(_audit_open.proposals(objections), before)
        if not props:
            return
        meant = amended if isinstance(amended, dict) and written is not None else before
        _held, held_back = _audit_open.split(props, meant)
        reasons = {p["path"]: ("the audit may change how the study is run, not what its results are judged by"
                               if self._audit_may_not_change(str(p["path"]))
                               and not _design_parts.invalid_reason(p["path"], before) else "") for p in held_back}
        for p in held_back:
            if p["path"] == "protocol.optimisation" and not isinstance(p["value"], dict):
                _block, why = _optim.normalize(p["value"])
                reasons[p["path"]] = (f"the proposal is not a valid search block ({why}); the whole block was kept as it was")
        reasons = {k: v for k, v in reasons.items() if v}
        lost: list[dict[str, Any]] = []
        if written is not None and isinstance(amended, dict):
            expected, _why = _plan.normalize_design({k: v for k, v in amended.items() if k != "rationale"})
            for p in _held:
                found, now = _design_parts.get_part(written, p["path"])
                was, want = _design_parts.get_part(expected or {}, p["path"])
                if was and (not found or json.dumps(now, sort_keys=True, default=str) != json.dumps(want, sort_keys=True, default=str)):
                    lost.append(p)
        held = [p for p in _held if p not in lost]
        for p in lost:
            reasons[p["path"]] = "plan.md was written with it and read back without it"
        self._audit_open_record(held, held_back + lost, reasons, iteration)

    def _audit_open_record(self, held: list[dict[str, Any]], missing: list[dict[str, Any]], reasons: dict[str, str],
                           iteration: int) -> None:
        _audit_open.record(self.quest_root, held, missing, reasons=reasons, default_reason="", iteration=iteration)
        for p in missing:
            self._log.warning("[design] the audit's change to `%s` is not in the design that runs (%s); it is kept as an open "
                              "item for the code and the paper", p["path"], reasons.get(p["path"]) or "FI did not apply it")
        if missing:
            print(f"[FI] The methodology audit proposed {len(missing)} change(s) the plan does not hold; they are listed in "
                  f"QUALITY.md and passed to the code. See .fi/{_audit_open.FILE}")

    def _keep_what_the_results_are_judged_by(self, before: dict[str, Any], amended: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """``(design, changed)``: ``amended`` with every protocol key but those in ``_AUDIT_MAY_CHANGE`` put back as it was
        in ``before`` (the checks and their tolerances, thresholds, acceptance rules, metrics, precision target, criteria,
        model and the rest), and whether the audit tried to change one. The method, the variables and the grid may change."""
        old = before.get("protocol") if isinstance(before.get("protocol"), dict) else None
        new = amended.get("protocol") if isinstance(amended.get("protocol"), dict) else None
        if old is None and new is None:
            return amended, False
        old, new = old or {}, new or {}
        merged = {k: v for k, v in old.items() if k not in self._AUDIT_MAY_CHANGE}
        merged.update({k: new[k] for k in new if k in self._AUDIT_MAY_CHANGE})
        merged.update({k: old[k] for k in old if k in self._AUDIT_MAY_CHANGE and k not in new})
        searched = isinstance(old.get("optimisation"), dict) and isinstance(new.get("optimisation"), dict)
        changed = any(new.get(k) != old.get(k) for k in {*old, *new}
                      if k not in self._AUDIT_MAY_CHANGE and not (searched and k == "optimisation"))
        if searched:
            # A search for the best design: the design to beat, the variables and their ranges, the search and the
            # settings it runs at may change; what a design is judged by (the objective's quantity, unit and direction,
            # the limits, the margin of "better", the target, the check's runs, the assertions) stays as it was.
            block, judged_changed = self._keep_what_a_search_is_judged_by(old["optimisation"], new["optimisation"])
            merged["optimisation"] = block
            changed = changed or judged_changed
        return {**amended, "protocol": merged}, changed

    def _audit_may_not_change(self, path: str) -> bool:
        """Whether ``path`` names what the results are judged by, which the audit may not change (see ``_AUDIT_MAY_CHANGE``)."""
        parts = path.split(".")
        if len(parts) < 2 or parts[0] != "protocol" or parts[1] in self._AUDIT_MAY_CHANGE:
            return False
        if parts[1] != "optimisation":
            return True
        if len(parts) < 3:
            return False  # the whole block: only its judged parts are put back, the rest is applied
        return parts[2] in (*self._AUDIT_SEARCH_JUDGED, *self._AUDIT_CHECK_RUNS_KEYS) or (parts[2] == "objective" and (
            len(parts) == 3 or parts[3] in self._AUDIT_OBJECTIVE_JUDGED))

    def _keep_what_a_search_is_judged_by(self, old: dict[str, Any], new: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """``(block, changed)``: the audit's search block with what judges a design put back as it was."""
        block = dict(new)
        changed = False
        for key in self._AUDIT_SEARCH_JUDGED:
            if block.get(key) != old.get(key):
                changed = True
                if key in old:
                    block[key] = old[key]
                else:
                    block.pop(key, None)
        old_obj, new_obj = old.get("objective"), block.get("objective")
        if isinstance(old_obj, dict) and isinstance(new_obj, dict):
            obj = dict(new_obj)
            for key in self._AUDIT_OBJECTIVE_JUDGED:
                if obj.get(key) != old_obj.get(key):
                    changed = True
                    if key in old_obj:
                        obj[key] = old_obj[key]
                    else:
                        obj.pop(key, None)
            block["objective"] = obj
        elif new_obj != old_obj:
            changed = True
            if old_obj is None:
                block.pop("objective", None)
            else:
                block["objective"] = old_obj
        if self._effective_check_runs(old) != self._effective_check_runs(block):
            # The runs of the finer check are ``check_runs``, else ``runs_per_evaluation``: only what keeps them as they
            # were is put back. With ``check_runs`` written, that is it alone (``runs_per_evaluation`` is then how many
            # runs the search makes of a design); without it the default comes from ``runs_per_evaluation``.
            changed = True
            if "check_runs" in old:
                block["check_runs"] = old["check_runs"]
            else:
                block.pop("check_runs", None)
                if "runs_per_evaluation" in old:
                    block["runs_per_evaluation"] = old["runs_per_evaluation"]
                else:
                    block.pop("runs_per_evaluation", None)
        return block, changed

    @staticmethod
    def _effective_check_runs(block: dict[str, Any]) -> Any:
        """The runs of the finer check as FI reads them (``optimisation_plan.budget``, the reader the check's budget and
        the check itself use); the two keys as written when the block cannot be read."""
        try:
            return _optim.budget(block)["check_runs"]
        except Exception:  # noqa: BLE001 -- an unreadable block is compared as written
            return (block.get("check_runs"), block.get("runs_per_evaluation"))

    def _write_receipt(self, check: str, status: str, **kwargs: Any) -> None:
        """Leave the receipt of a required check (core/receipts.py). A receipt that cannot be written is logged: the
        evidence ladder then reads the check as not shown to have run, which is the safe side."""
        try:
            _receipts.write(self.quest_root, check, status=status, producer=check, **kwargs)
        except Exception as e:  # noqa: BLE001 -- the check's own result stands; only its record is missing
            self._log.warning("[%s] its receipt could not be written: %r", check, e)

    def _clear_check_stop(self, check: str) -> None:
        """A check that judged: a later failure (another iteration, other inputs) stops the quest again."""
        marker = self.fi_dir / "check_retries.json"
        try:
            tried = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
            if isinstance(tried, dict) and check in tried:
                tried.pop(check)
                marker.write_text(json.dumps(tried, indent=2) + chr(10), encoding="utf-8")
        except (OSError, ValueError):
            pass

    def _stop_once_for_check(
        self, check: str, failure: str, *, look: str, inputs: dict[str, Any] | None = None,
    ) -> None:
        """Under ``rigor_profile: research``, a required check that could not judge stops the quest once, and resume
        tries it again; a second failure is recorded as a gap and the quest goes on (a person asked for one stop, never
        an endless one). Other profiles go on at once; the evidence level names the gap either way."""
        name = _receipts.REQUIRED.get(check, check)
        if self.config.rigor_profile != "research":
            self._log.warning("[%s] %s could not judge (%s); the evidence level records it as a gap", check, name, failure)
            return
        marker = self.fi_dir / "check_retries.json"
        try:
            tried = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        except (OSError, ValueError):
            tried = {}
        tried = tried if isinstance(tried, dict) else {}
        # One stop per check for the same inputs: the resume's retry of that stop goes on if it fails again, and a
        # later failure on other inputs (another iteration, another draft) stops again.
        key = _receipts.sha256(inputs or {})
        if tried.get(check) == key:
            self._log.warning(
                "[%s] %s failed again after the retry (%s); recorded as a gap, the quest goes on", check, name, failure,
            )
            return
        tried[check] = key
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(tried, indent=2) + "\n", encoding="utf-8")
        except OSError as e:
            # Without the marker a resume would stop here again; going on with the gap is the bounded choice.
            self._log.warning("[%s] could not record the stop (%r); going on with the gap", check, e)
            return
        self._pause_for_human(
            kind=f"{check}_unknown",
            interaction="supply",
            headline=f"{name[0].upper()}{name[1:]} could not judge, and the research profile needs it",
            steps=[
                f"What failed: {failure}.",
                f"Look at {look}: this is usually a provider outage, a rate limit, or a reply that could not be read.",
                "Choose another model for it, or retry later, then resume: the check is tried once more.",
                "If it fails again, the quest goes on and the result is not marked publication-ready "
                f"({name} is listed as a gap in needs/EVIDENCE.json).",
            ],
            payload={"check": check, "failure": failure},
        )

    def _oracle_source_gaps(self, state: QuestState, protocol: Any) -> list[str]:
        """The checks whose expected value has no source a reader can check, for the evidence record. The sources are
        the ones the freeze recorded, numbered as the plan cited them (the paper later cites a subset under new numbers).
        A protocol frozen before the freeze recorded them (or one whose record of them was edited) is judged only on
        what needs no list: an empty ``reference``."""
        pairs, extra = self._unsourced_oracles(state, protocol)
        # A check the person chose to go on with as it is stays a gap, and says so ("source not confirmed", and who).
        out = []
        for oracle, why in pairs:
            who = _accepted.chose(self.quest_root, oracle)
            out.append(f"{why} ({_accepted.NOT_CONFIRMED}: {_accepted.who_text(who)} went on without one)" if who
                       else why)
        return out + extra

    def _unsourced_oracles(self, state: QuestState, protocol: Any) -> tuple[list[tuple[dict[str, Any], str]], list[str]]:
        """``([(oracle, why)], other gaps)``: each check of ``protocol`` whose expected value has no source a reader can
        check, judged as :meth:`_oracle_source_gaps` says."""
        if not isinstance(protocol, dict) or not self._runs_code(state):
            return [], []
        by_name = {str(o["name"]).strip(): o for o in _oracle.declared(protocol)}
        frozen = _frozen.load(self.quest_root)
        if frozen is None:
            named = _oracle.unsourced(protocol, self._retrieved_sources(state))
        elif isinstance(frozen.get("sources"), list) and not frozen.get("sources_problem"):
            named = _oracle.unsourced(protocol, frozen["sources"])
        else:
            named = _oracle.empty_reference_pairs(protocol)
        extra = [str(frozen["sources_problem"])] if frozen is not None and frozen.get("sources_problem") else []
        return [(by_name.get(n) or {"name": n}, why) for n, why in named], extra

    def _not_confirmed_names(self, state: QuestState, protocol: Any) -> list[str]:
        """The checks that still have no stated source and that a person chose to go on with as they are."""
        try:
            pairs, _extra = self._unsourced_oracles(state, protocol)
        except Exception:  # noqa: BLE001 -- a record must never stop a quest
            return []
        return [str(o.get("name")) for o, _why in pairs if _accepted.chose(self.quest_root, o)]

    def _simulation_sources(self, state: QuestState) -> tuple[Path, dict[str, str]]:
        """The script that computes the data and the text of it and of its helper modules in ``code/``: ``simulate.py``
        when the quest runs two scripts, else ``experiment.py`` (a ``simulate.py`` left beside a one-script quest is not
        what runs). The analysis, the cluster's submit script and FI's own ``run.py`` and ``fi_search.py`` are not the
        simulation."""
        code = self.quest_root / "code"
        simulate = code / _split_run.SIMULATE_NAME
        split = self._split_on(state) and simulate.is_file()
        main = simulate if split else code / "experiment.py"
        skip = {"experiment.py" if split else _split_run.SIMULATE_NAME, _trial_runner.SUBMIT_NAME, "run.py",
                _code_project.SEARCH, "replot_figures.py", "replot_layout.py", "web_plots.py"}
        texts: dict[str, str] = {}
        for path in [main, *sorted(p for p in code.glob("*.py") if p != main and p.name not in skip)]:
            try:
                texts[path.name] = path.read_text(encoding="utf-8")
            except OSError:
                continue
        if split and main.name in texts:
            texts.update(_code_layout.package_sources(code))  # the model's package is part of the simulation
        return main, (texts if main.name in texts else {})

    def _unlabelled(self, state: QuestState, protocol: Any) -> tuple[Path, list[str]]:
        """``(the simulation script, the plan's generates equations it does not mark)``; nothing is missing for a quest
        that runs no experiment, a plan with no such equation, or a quest with no script yet."""
        if not isinstance(protocol, dict) or not self._runs_code(state) or not _oracle.generating_equations(protocol):
            return self.quest_root / "code" / "experiment.py", []
        main, texts = self._simulation_sources(state)
        return main, (_oracle.unlabelled_equations(protocol, texts, self._outline_equation_names(state)) if texts else [])

    def _equation_label_gaps(self, state: QuestState, protocol: Any) -> list[str]:
        """The plan's ``generates`` equations the simulation does not mark (``# E1`` on or above the code that
        implements it), as one sentence. Only the labelled mapping is read, never the mathematics."""
        return self._label_sentences(*self._unlabelled(state, protocol))

    def _label_target(self, state: QuestState, main: Path) -> Path:
        """Where an equation's label belongs: in the model's package (``code/<package>/model.py``) when the simulation
        computes with one, since that is where the equations are; else the simulation script itself."""
        package = self._package_in_use(state)
        model = self.quest_root / "code" / package / _code_layout.MODEL_NAME if package else None
        return model if model is not None and model.is_file() else main

    @staticmethod
    def _label_sentences(main: Path, missing: list[str]) -> list[str]:
        if not missing:
            return []
        where = f"code/{main.parent.name}/{main.name}" if main.parent.name != "code" else f"code/{main.name}"
        return _oracle.label_gaps({"model": {"equations": [{"id": m, "role": "generates"} for m in missing]}}, "",
                                  where)

    async def _label_equations(self, state: QuestState) -> Path | None:
        """Right after the code is written: when the simulation does not mark every ``generates`` equation of the plan,
        ask the model once to add the labels as comments, and keep its answer only when the code is otherwise the same
        (the same syntax tree; a changed docstring counts as a change). Never stops a quest; the check before the run says what is
        still missing. Returns the script it rewrote, or ``None``."""
        try:
            main, missing = self._unlabelled(state, self._protocol_block(state))
            target = self._label_target(state, main)
            if target != main:
                # With the model's package the labels belong on its functions (a label in simulate.py does not say
                # which function computes the equation): what is missing is read there, as the layout check reads it.
                package = target.parent.name
                sources = {rel: text for rel, text in _code_layout.package_sources(self.quest_root / "code").items()
                           if rel.startswith(f"{package}/")}
                missing = [row["id"] for row in _code_layout.equation_map(self._protocol_block(state), sources,
                                                                          self._outline_equation_names(state))
                           if not row["function"]]
            if not missing:
                return None
            main = target
            before = main.read_text(encoding="utf-8")
            view = _oracle.model_view((self._protocol_block(state) or {}).get("model")) or {}
            equations = [e for e in view.get("equations") or []
                         if isinstance(e, dict) and str(e.get("id") or "").strip() in missing]
            listed = "\n".join(f"- {e.get('id')}: {e.get('formula') or ''}" for e in equations) or "\n".join(
                f"- {m}" for m in missing)
            prompt = (
                f"# Label the equations in {main.name}\n\n"
                f"The plan's model says the simulation computes its data with these equations:\n{listed}\n\n"
                f"Return {main.name} with a comment carrying each equation's id (`# {missing[0]}`) on or above the code "
                "that computes it. Change nothing else: no code, no names, no values; only add comments. Reply with "
                "the whole script in one fenced Python block.\n\n"
                f"```python\n{before}\n```\n"
            )
            text = await self._chat(prompt, node="implement")
            after, _deps = _parse_implement_response(text)
            after = re.sub(r"\A\s*#\s*file\s*:[^\n]*\n", "", after)  # a `# file: simulate.py` line is not the script's
            if not after.strip() or ast.dump(ast.parse(before)) != ast.dump(ast.parse(after)):
                self._log.warning("[implement] the request to label the equations %s changed more than comments; "
                                  "%s is kept as it was", ", ".join(missing), main.name)
                return None
            main.write_text(after, encoding="utf-8")
            self._log.info("[implement] added the equation labels to %s (%s)", main.name, ", ".join(missing))
            return main
        except Exception as e:  # noqa: BLE001 -- a convenience; the check before the run is what counts
            self._log.warning("[implement] could not ask for the equation labels: %r", e)
            return None

    def _planning_limit(self, state: QuestState) -> str:
        """The run time the code-writing prompts are given (their ``$timeout_s``): the topic's own limit when it states a
        shorter one than ``execution.timeout_s`` (the topic's constraints win over FI's default), else FI's. The design
        prompt keeps FI's limit beside :meth:`_time_note`, which states the topic's."""
        own = int(self.config.execution.timeout_s)
        limit = self._topic_limit_s(state)
        if limit is None or not limit < own:
            return str(own)
        return (f"{_disclosure._plain_seconds(limit).removesuffix(' s')} (the topic's own limit, which the code is written "
                f"for; FI stops a run that takes longer than {own} s, its execution.timeout_s)")

    def _code_layout(self, state: QuestState, protocol: Any = None) -> dict[str, Any] | None:
        """How this quest's code is laid out (:mod:`core.code_layout`): a small research tool or two scripts, with what
        that costs. ``None`` for a quest that keeps no simulation of its own in two scripts (the no-simulation path, a
        survey, ``split_analysis: false``), and for code written before this quest decided a layout (a quest begun
        before the layout existed, resumed or refined): those are left exactly as they were."""
        if not self._runs_code(state) or not self._split_on(state):
            return None
        record = _code_layout.load(self.quest_root)
        if record is None and (self.quest_root / "code" / _split_run.SIMULATE_NAME).is_file():
            return None
        if protocol is None:
            protocol = self._protocol_block(state)
        ex = self.config.execution
        package = str((record or {}).get("package") or "") or _code_layout.package_name(
            str(self.config.title or self.config.topic or ""))
        # How many code files the topic asks for (the topic reader's structured answer, never a reading of its prose):
        # kept and said, never a reason to turn FI's checks of the code off (core/code_layout.py::decide).
        required = _disclosure._read_json(self.quest_root / ".fi" / "topic_requirements.json")
        decision = _code_layout.decide(protocol if isinstance(protocol, dict) else None, enabled=ex.code_package,
                                       max_extra_lines=ex.code_package_max_extra_lines,
                                       max_extra_calls=ex.code_package_max_extra_calls, package=package,
                                       code_files=_topic_req.code_files(required))
        # A quest laid out as a package stays one: a later redesign that adds equations does not switch it back to two
        # scripts (``code_package: false`` still does, and a plan written again decides again). Two scripts are decided
        # again each time, so raising the limit, as plan.md says, takes effect on a resume.
        if (ex.code_package and record and record.get("shape") == _code_layout.PACKAGE
                and not state.get("_deciding_layout")):
            decision["shape"], decision["reason"] = _code_layout.PACKAGE, ""
        return decision

    def _plan_code_layout(self, state: QuestState) -> list[str]:
        """At plan time: decide the layout of the code from the plan, say it in run.log and return plan.md's lines."""
        try:
            protocol = (state.get("design") or {}).get("protocol")
            layout = self._code_layout({**state, "_deciding_layout": True},  # type: ignore[typeddict-unknown-key]
                                       protocol if isinstance(protocol, dict) else {})
            if layout is None:
                return []
            if not (self.quest_root / "code" / _split_run.SIMULATE_NAME).is_file():
                layout["extra_calls_spent"] = 0  # no code yet (a new quest, or a restart from the plan): a new budget
            _code_layout.save(self.quest_root, layout)
            self._log.info("[plan] %s", _code_layout.summary(layout))
            files = _code_layout.files_note(layout)
            if files:
                self._log.info("[plan] %s", files)
            # The topic's request for fewer code files than FI keeps is an item the paper must name.
            self._note_not_done([{"name": "code in the files the topic asked for", "sentence": files,
                                  "kind": "code_files"}] if files else [], kind="code_files")
            return _code_layout.plan_lines(layout)
        except Exception as e:  # noqa: BLE001 -- how the code is laid out must never stop the plan
            self._log.warning("[plan] could not work out how the code will be laid out: %r", e)
            return []

    def _adopt_reply_package(self, text: str, scripts: dict[str, str], package: str) -> tuple[str, dict[str, str]] | None:
        """The reply wrote the model's package under a name of its own (``sir/model.py`` where FI chose
        ``sir_outbreaks``) and simulate.py imports that name: ``(that name, its files)``, so the code that was written
        is the code that runs. ``None`` when there is no such package, or FI's own is already on disk."""
        if (self.quest_root / "code" / package).is_dir():
            return None
        for name in _code_layout.packages_in_reply(text, _PY_FENCE_RE):
            if name == package or not _code_layout.usable_name(name):
                continue
            if not _code_layout.imports(scripts.get("simulate") or "", name):
                continue
            files = _code_layout.reply_files(text, _PY_FENCE_RE, name)
            if _code_layout.complete(files, name):
                self._log.info("[implement] the reply wrote the model's package as code/%s/ (not code/%s/) and "
                               "simulate.py uses it: that name is kept", name, package)
                return name, files
        return None

    async def _code_package_reply(self, state: QuestState, prompt: str, text: str, scripts: dict[str, str], *,
                                  extend: bool, bodies: bool = False) -> tuple[dict[str, str], str, dict[str, str]]:
        """``(scripts, reply, package files)`` for a two-script reply: the model's package in it, asked for again when the
        reply left it out (while ``execution.code_package_max_extra_calls`` allows). A reply that still has none keeps
        the code it has, and says so. ``bodies``: FI writes the analysis, so the reply holds simulate.py and the functions of
        derive.py, never experiment.py."""
        layout = self._code_layout(state)
        if layout is None:
            return scripts, text, {}
        if layout["shape"] != _code_layout.PACKAGE:
            self._log.info("[implement] %s", _code_layout.summary(layout))
            _code_layout.save(self.quest_root, layout)
            return scripts, text, {}
        package = layout["package"]
        code = self.quest_root / "code"
        on_disk = (code / package / _code_layout.MODEL_NAME).is_file()
        if not on_disk and not any((code / n).is_file() for n in (_split_run.SIMULATE_NAME, "experiment.py")):
            # The code is written from nothing (a new quest, or a restart from the code step): a new request budget.
            _code_layout.save(self.quest_root, {**(_code_layout.load(self.quest_root) or layout), "extra_calls_spent": 0})
        if extend and not on_disk:
            # An extension changes as little as it can: code kept as two scripts is not restructured for it.
            return scripts, text, {}
        if not extend and _function_steps.is_ready(self.quest_root, package):
            self._write_plan_constants(state, package)  # a package kept from before; an extension's files are written by the caller
            if _code_layout.reply_files(text, _PY_FENCE_RE, package):
                self._log.info("[implement] the reply also wrote the model's package; FI keeps the package it wrote and "
                               "checked function by function")
            _code_layout.save(self.quest_root, {**layout, "fell_back": "", "function_steps": True})
            return scripts, text, {}
        files = _code_layout.reply_files(text, _PY_FENCE_RE, package)
        if not _code_layout.complete(files, package) and not extend:
            adopted = self._adopt_reply_package(text, scripts, package)
            if adopted:
                package, files = adopted
                layout = {**layout, "package": package}
        if extend:
            # A package file given back without names it has now is a part of the file (the new function alone), and
            # writing it would lose the rest: the package is then kept, and the run says why.
            for rel in list(files):
                old_path = code / rel
                old = old_path.read_text(encoding="utf-8") if old_path.is_file() else ""
                dropped = _code_layout.dropped_functions(old, files[rel]) if old else []
                if dropped:
                    self._log.warning("[implement] the extension gave code/%s back without %s; the package is kept as "
                                      "it is", rel, ", ".join(dropped))
                    files = {}
                    break
        previous = (code / _split_run.SIMULATE_NAME).read_text(encoding="utf-8") if (
            code / _split_run.SIMULATE_NAME).is_file() else ""
        sim_changed = not _split_run.same_script(previous, scripts.get("simulate") or "")
        limit = self.config.execution.code_package_max_extra_calls
        if not _code_layout.complete(files, package) and not extend and (not on_disk or sim_changed):
            if _code_layout.calls_left(self.quest_root, limit) > 0:
                self._log.warning("[implement] the reply left out the model's package (code/%s/); asking once more",
                                  package)
                _code_layout.spend_call(self.quest_root)
                try:
                    again = await self._chat(prompt + _code_layout.reminder(package, analysis=bodies), node="implement")
                except Exception as e:  # noqa: BLE001 -- the package is asked for as a help; the code that came stays
                    self._log.warning("[implement] asking again for the model's package failed: %r", e)
                    again = ""
                again_scripts = _split_run.parse_split_response(again, _PY_FENCE_RE, bodies=bodies)
                again_files = _code_layout.reply_files(again, _PY_FENCE_RE, package)
                if again_scripts is not None and _code_layout.complete(again_files, package):
                    scripts, text, files = again_scripts, again, again_files
            else:
                self._log.info("[implement] the reply left out the model's package; not asking again: the %d extra "
                               "requests allowed (execution.code_package_max_extra_calls) are spent", limit)
        fell_back = ""
        if not _code_layout.complete(files, package):
            if not on_disk:
                fell_back = "the reply left out the model's package"
                self._log.warning("[implement] the reply left out the model's package (code/%s/): this time the code "
                                  "has no package of equations of its own, only simulate.py and experiment.py", package)
            elif sim_changed and not extend:
                self._log.warning(
                    "[implement] simulate.py was written again but the model's package (code/%s/) was not: the package "
                    "from before is kept, so check that it still computes the plan's model", package)
        else:
            self._log.info("[implement] %s", _code_layout.summary(layout))
        # A package simulate.py imports that neither this reply nor code/ holds (the reply named it otherwise, or wrote
        # part of it): said plainly, since the first run will fail on it and be sent for repair.
        kept = package if _code_layout.complete(files, package) else ""
        absent = [n for n in dict.fromkeys([package, *_code_layout.reply_folders(text, _PY_FENCE_RE)])
                  if n != kept and not (code / n / "__init__.py").is_file()
                  and _code_layout.imports(scripts.get("simulate") or "", n)]
        if absent:
            self._log.warning("[implement] simulate.py imports the package %s, but %s not written: the first run will "
                              "fail on it and be sent for repair", ", ".join(f"`{n}`" for n in absent),
                              " and ".join(f"code/{n}/" for n in absent) + (" was" if len(absent) == 1 else " were"))
        _code_layout.save(self.quest_root, {**layout, "fell_back": fell_back})
        return scripts, text, files if _code_layout.complete(files, package) else {}

    # --- the model's functions, one at a time, and a small test per equation (core/function_steps.py, core/equation_tests.py)

    # --- a different model reads the code against the plan's required outputs (core/code_review.py)

    def _reader_vs_planner(self, node: str, *, after_call: bool, writer: str = "plan") -> tuple[bool | None, str, str, bool]:
        """``(same model?, reader, writer's model, the connection named the reader)`` for the step ``node`` against the
        step that wrote what it reads (``writer``: the plan by default), as :meth:`_review_oracles` decides it: from the
        models the connections named after a call, else the ones asked for. ``None`` when it cannot be told."""
        answered = self._chat_provenance(node) if after_call else {}
        planned = self._chat_provenance(writer)
        reader_asked = self._model_for_node(node) or self.config.provider.model or ""
        planner_asked = self._model_for_node(writer) or self.config.provider.model or ""
        reader = str(answered.get("model") if answered.get("reported") else reader_asked or "")
        planner = str(planned.get("model") if planned.get("reported") else planner_asked or "")
        if answered.get("reported") and planned.get("reported"):
            same: bool | None = _review.same_model(answered.get("model"), planned.get("model"))
        elif reader_asked and planner_asked:
            same = _review.same_model(reader_asked, planner_asked)
        elif not self._model_for_node(node) and not self._model_for_node(writer):
            same = True
        else:
            same = None
        return same, reader or "the provider's default model", planner or "the provider's default model", bool(answered.get("reported"))

    async def _code_review_gate(self, state: QuestState, *, final: bool = False) -> QuestState | None:
        """A different model reads the plan's required outputs and its logic against the scripts, and what it finds is
        repaired (core/code_review.py). Called twice in ``_node_execute``: early (``final=False``), where a version is read,
        its findings are kept by stable id and repaired test first, one at a time (:meth:`_repair_review_findings`), the
        changed version is read again against the findings still open, and so on while the open issues that change a
        result keep dropping, within ``code_review.MAX_SEND_BACKS`` passes; and just before the run
        (``final=True``), where the version that will run is read with the one reading the early call may not spend, and
        nothing is repaired any more (a repaired version would be unread). ``.fi/code_review.json`` keeps the readings, the
        open findings in each, whether the repairs converged and whether the final version was read, so a resume does not
        repeat a reading. What is open at the end is said in the limitations; a high-severity finding about a headline
        quantity ends in the honest stop after the run unless the results contradict it. Always ``None``: the code goes on."""
        code = self.quest_root / "code"
        if (self.config.engine.oracle_check == "off" or not self._runs_code(state)
                or not (code / "experiment.py").is_file() or self.config.execution.background_jobs):
            return None
        design = state.get("design") if isinstance(state.get("design"), dict) else {}
        protocol = self._protocol_block(state)
        reqs = _code_review.requirements(design, protocol if isinstance(protocol, dict) else None)
        if not reqs:
            return None
        review_node = self._review_node("code_review")
        for _round in range(_code_review.MAX_SEND_BACKS + 2):
            if _round:
                # A plan the plan's model rewrote to resolve a contradiction is the plan the next reading is made against.
                protocol = self._protocol_block(state)
                reqs = _code_review.requirements(design, protocol if isinstance(protocol, dict) else None) or reqs
            version = _code_review.version(code)
            record = _code_review.load(self.quest_root)
            if record.get("read_version") == version and record.get("version") == version:
                # Read already. What it found stands unless this call may repair it (and it can be read again after).
                problems = [p for p in record.get("problems") or [] if isinstance(p, dict)]
                if record.get("final") or not problems:
                    break
                why = "this is the version that will run" if final else _code_review.may_repair(record)
                if why:
                    record["final"] = True
                    record["repairs_stopped"] = why  # QUALITY.md says why the repairs ended
                    _code_review.save(self.quest_root, record)
                    self._log.warning("[code_review] %d finding(s) are open and are not repaired: %s", len(problems), why)
                    self._code_review_outcome(record)
                    break
                if not await self._repair_review_findings(state, record, protocol, review_node):
                    record = _code_review.load(self.quest_root)
                    record["final"] = True
                    _code_review.save(self.quest_root, record)
                    self._code_review_outcome(record)
                    break
                continue
            if record.get("version") == version and record.get("skipped"):
                break  # the reader is the model that wrote the code (or FI cannot tell): nothing to read again
            after_run_repair = final and _numeric.repairs_made(state) > 0
            if not _code_review.may_read(record, final=final, after_run_repair=after_run_repair):
                if final:
                    cap = _code_review.read_cap(final=True, after_run_repair=after_run_repair)
                    # What the last reading left open stays open for this version, which no one read (carried).
                    record = {**_code_review.carried(record), "version": version, "requirements": len(reqs),
                              "skipped": f"the {cap} readings of the code allowed in a quest are spent"}
                    _code_review.save(self.quest_root, record)
                    self._log.warning("[code_review] %s%s", _code_review.plain_lines(record),
                                      "; the final version of the code was not read again by another model, so what the "
                                      "last reading left open stays open" if record.get("read_version") else "")
                    self._code_review_outcome(record)
                else:
                    self._log.info("[code_review] the code was repaired; the last reading is kept for the version that will "
                                   "run, so it is read just before the run")
                break
            if not await self._read_code_for_review(state, record, version, reqs, protocol, design, review_node):
                break
        record = _code_review.load(self.quest_root)
        if record:
            record["final_read"] = record.get("read_version") == _code_review.version(code)
            _code_review.save(self.quest_root, record)
        return None

    async def _read_code_for_review(self, state: QuestState, old: dict[str, Any], version: str, reqs: list[dict[str, Any]],
                                    protocol: Any, design: dict[str, Any], review_node: str) -> bool:
        """ONE reading of the code version ``version`` (at most two requests, one more for each that failed once). ``True``
        when it found problems that were recorded as open (the caller may repair them); ``False`` when it found none, when
        the reader is the model that wrote the code, or when it could not be had."""
        code = self.quest_root / "code"
        previous = [p for p in old.get("problems") or [] if isinstance(p, dict)] if old.get("repaired") else []
        record: dict[str, Any] = {"version": version, **_code_review.carried(old), "requirements": len(reqs)}
        record.setdefault("reviews_total", 0)
        record.setdefault("send_backs", 0)
        same, reader, planner, _named = self._reader_vs_planner(review_node, after_call=False, writer="implement")
        if same is True:
            record["skipped"] = f"the reader is the model that wrote the code ({planner})"
            _code_review.save(self.quest_root, record)
            self._log.info("[code_review] %s", _code_review.plain_lines(record))
            return False
        record["reviews_total"] = int(record["reviews_total"]) + 1
        _code_review.save(self.quest_root, record)  # counted before the request: a resume does not ask again
        plan_dict = protocol if isinstance(protocol, dict) else None
        calls = _code_review.reading_calls(code, design, plan_dict, open_items=_audit_open.for_code(self.quest_root), reqs=reqs)
        topic = " ".join(str(state.get("topic") or self.config.topic or "").split())
        values = _pairs.plan_values(plan_dict)  # taken out of FI's own small-run text
        shown_values = _code_review.reportable_values(design, plan_dict)  # taken out of what the reader writes
        # What each request is given, from FI's own record: the files and line ranges, the plan's sections, a line cut.
        record["files_read"] = sorted({f for c in calls for f in c["files"]})
        record["code_parts"] = 1 + max(c["code_index"] for c in calls)  # parts of the code, not requests
        record["plan_parts"] = 1 + max(c["plan_index"] for c in calls)
        record["requests"] = [{"files": {f: r for f, r in c["shown"].items()}, "plan_sections": c["plan_sections"],
                               "asked": len(reqs) if c.get("asked") is None else len(c["asked"]),
                               "chars": len(c["code"]) + len(c["plan"]) + len(c["open_items"]),
                               **({"open_items": {"chars": [i["chars"] for i in c["items"]],
                                                  "cut": [i["part"] for i in c["items"] if i["cut"]]}}
                                  if c["items"] else {})} for c in calls]
        if any(c["left_out"] for c in calls):
            record["plan_not_shown"] = sorted({n for c in calls for n in c["left_out"]})
        if any(c["cut"] for c in calls):
            record["lines_cut"] = [x for c in calls for x in c["cut"]]
        earlier = _code_review.earlier_block(previous)
        small_run = (_pairs.scrub(getattr(self, "_small_run_note", ""), values, printed=True)
                     or "(FI did not run the simulation before this reading)")
        # A request whose text is the same as one made before, to the same model, is not asked again: the reply kept for it is
        # read again with today's rules (core/reuse_key.py ``answer_key``; the reply is stored only when this request's own
        # reply was usable).
        from . import reuse_key as _reuse_key

        kept = {k: v for k, v in _code_review.load_reading_cache(self.quest_root).items() if isinstance(v, dict) and "reply" in v}

        def build_prompt(call: dict[str, Any], left: list[dict[str, Any]], again: str = "") -> str:
            return self._prompts["code_review"].substitute(
                topic=topic, requirements=json.dumps([{k: r[k] for k in ("id", "kind", "text")} for r in left], indent=2),
                plan=call["plan"], open_items=call["open_items"] or "(none)",
                code=_code_review.part_note(call["code_index"] + 1, record["code_parts"]) + call["code"], previous=earlier,
                small_run=small_run) + again

        keys = [_reuse_key.answer_key(build_prompt(c, [r for r in reqs if c.get("asked") is None or r["id"] in c["asked"]]),
                                      str(reader or "")) for c in calls]
        verdicts: list[dict[str, Any]] | None = None
        logic: list[dict[str, Any]] = []
        listed: list[str] = []
        outside: list[dict[str, Any]] = []  # what a request said about a place it was not shown
        part_verdicts: list[list[dict[str, Any]]] = []
        p_verdicts: list[dict[str, Any]] | None = None
        many = len(calls) > 1

        def spend() -> None:
            used = int(record.get("reader_calls") or 0)
            if used >= _code_review.MAX_READER_CALLS:
                raise _code_review.BudgetSpent(f"the {_code_review.MAX_READER_CALLS} reader requests of a quest are used")
            record["reader_calls"] = used + 1
            _code_review.save(self.quest_root, record)  # counted before the request, as the repair requests are

        try:
            # The code is read whole; when it does not fit in one request it is read in parts, each against all the
            # requirements, and the verdicts are merged from FI's own record (:func:`core.code_review.merge_parts`).
            # Per part: at most two requests, one, and one more for the requirements a reply left unanswered, or, when the
            # first answered every requirement but named no problem and said nothing it checked, for a review.
            # A request is made for each part, and made once more if it fails: those 2 per part are reserved before the
            # reading starts, so a reading is never admitted and then dropped for want of requests. The follow-up ask of a
            # part (up to 2 more) is made only while the requests the later parts need stay reserved.
            # Only the parts that must be asked are reserved for, one request each: a part read before and not changed is
            # not asked again. A request that fails is made once more only while the parts still to come keep theirs; when
            # none is left for them the reading stops partway and the parts not read are recorded (``parts_not_read``).
            fresh = len({k for k in keys if k not in kept})
            if int(record.get("reader_calls") or 0) + fresh > _code_review.MAX_READER_CALLS:
                raise _code_review.BudgetSpent(f"the {_code_review.MAX_READER_CALLS} reader requests of a quest are "
                                               f"used; this reading needs {fresh} more")
            for p_index, call in enumerate(calls, start=1):
                key = keys[p_index - 1]
                p_verdicts = None
                asked_again = False
                # only the requirements this request was given the plan's definition of (:func:`core.code_review.asked_ids`)
                call_reqs = [r for r in reqs if call.get("asked") is None or r["id"] in call["asked"]]
                for _ask in range(2):
                    later = len({k for k in keys[p_index:] if k not in kept and k != key})
                    if _ask and int(record.get("reader_calls") or 0) + 1 + later > _code_review.MAX_READER_CALLS:
                        break  # the follow-up ask would use what the parts still to come need
                    left = [r for r in call_reqs if r["id"] not in {v["id"] for v in p_verdicts or []}]
                    again = ""
                    if not left and not call_reqs and _ask == 0:
                        pass  # a request with nothing to judge is still made once: it is read for problems of logic
                    elif not left:
                        if asked_again or p_verdicts is None or not _code_review.reply_said_nothing(p_verdicts, logic, listed):
                            break
                        self._log.warning("[code_review] the reader named no problem and did not say what it checked; "
                                          "asking once more")
                        again, left, asked_again = _pairs.REASK, list(call_reqs), True
                    prompt = build_prompt(call, left, again)
                    answer_key = _reuse_key.answer_key(prompt, str(reader or ""))
                    from_cache = answer_key in kept
                    if from_cache:
                        reply = kept[answer_key]["reply"]
                        if _ask == 0:
                            record["requests"][p_index - 1]["reused"] = True
                            record["parts_reused"] = int(record.get("parts_reused") or 0) + 1
                    else:
                        reply = await self._ask_reader(prompt, review_node, spend)
                    parsed = _parse_json_lenient(reply, node=review_node)
                    got = _code_review.parse(parsed, left, shown_values)
                    # a verdict or a finding must cite a file and a line this request was shown (not a line that was cut)
                    had = got is not None
                    got = _code_review.keep_seen(got, call["shown"], outside, call["cut"]) if got else got
                    if again:
                        before = [v for v in p_verdicts or [] if v["id"] not in {g["id"] for g in got or []}]
                        p_verdicts = [*before, *(got or [])] or None
                    else:
                        p_verdicts = [*(p_verdicts or []), *(got or [])] or ([] if had else None)
                    if not call_reqs:
                        p_verdicts = p_verdicts or []  # nothing was asked of it: no verdicts is the whole answer
                    found = _code_review.parse_logic(parsed, reqs, shown_values)
                    seen = [f for f in found if _code_review.shown_in_part(call["shown"], f["file"], f["line"], call["cut"])]
                    outside.extend(_code_review.outside_entry(f, "finding", call["shown"], call["cut"])
                                   for f in found if f not in seen)
                    found = seen
                    logic = _code_review.merge_logic(logic, found)
                    said = _pairs.checked(parsed.get("checked") if isinstance(parsed, dict) else None)
                    listed = [*listed, *(c for c in said if c not in listed)]
                    # Kept only when this request's own reply was usable: every requirement it was asked judged, or (none asked)
                    # a finding or a thing it checked of its own, never because an earlier part said something.
                    usable = (not {r["id"] for r in left} - {v["id"] for v in got or []}) if left else bool(found or said)
                    if usable and not from_cache and isinstance(reply, str):
                        kept[answer_key] = {"reply": reply}
                        _code_review.save_reading_cache(self.quest_root, kept)
                if p_verdicts is None and many:
                    # A part that was not read could hold what the others call missing: no verdict is made from the rest.
                    raise RuntimeError(f"part {p_index} of {len(calls)} of the code got no usable answer")
                part_verdicts.append(p_verdicts or [])
            verdicts = _code_review.merge_parts(part_verdicts) or None
            if verdicts is None:
                record["error"] = ("its answers cited no place in the code it was shown" if outside
                                   else "its answer named none of the plan's requirements")
            elif _code_review.reply_said_nothing(verdicts, logic, listed):
                verdicts, record["error"] = None, _pairs.NOT_A_REVIEW
        except _code_review.BudgetSpent as e:
            record["error"] = f"the code was not read in full: {str(e)[:160]}"
            verdicts = None
        except _ModelAnswerProblem as e:
            record["error"] = f"the provider withheld or cut off its answer, twice ({str(e)[:120] or type(e).__name__})"
            verdicts = p_verdicts if not many else None  # one part: what the first request answered is kept
        except Exception as e:  # noqa: BLE001 -- a reading that could not be had never stops a quest
            record["error"] = f"the call failed, twice ({str(e)[:120] or type(e).__name__})"
            verdicts = p_verdicts if not many else None
        if verdicts is None:
            unread = [_code_review.part_label(c, record["code_parts"]) for i, c in enumerate(calls) if keys[i] not in kept]
            if unread:
                record["parts_not_read"] = list(dict.fromkeys(unread))
        if outside:
            record["cited_outside_what_was_shown"] = outside[:50]
        same, reader, planner, named = self._reader_vs_planner(review_node, after_call=True, writer="implement")
        record.update(reviewer=reader, planner=planner, reported=named, same_model=same)
        if verdicts is not None and same is False:
            unanswered = [r for r in reqs if r["id"] not in {v["id"] for v in verdicts}]
            if unanswered:
                record["not_reviewed"] = [r["text"] for r in unanswered]
                record["not_reviewed_headline"] = any(r["headline"] for r in unanswered)
                self._log.warning("[code_review] the reader did not answer, even when asked again, for: %s",
                                  "; ".join(r["text"] for r in unanswered))
        if verdicts is None:
            _code_review.save(self.quest_root, record)
            self._log.warning("[code_review] %s", _code_review.plain_lines(record))
            return False
        if same is not False:
            record["skipped"] = ("the reader is the model that wrote the code" if same
                                 else "FI cannot tell whether the reader is another model than the one that wrote the code")
            _code_review.save(self.quest_root, record)
            self._log.info("[code_review] %s", _code_review.plain_lines(record))
            return False
        record["verdicts"] = verdicts
        record["read_version"] = version
        record["logic"] = logic
        record["checked"] = listed[:40]
        record["problems"] = [*_code_review.problems(verdicts), *logic]
        record["read_problems"] = list(record["problems"])  # carried: a later reading that fails does not lose the flags
        record.pop("final", None)  # this reading's findings replace the carried ones, and may be repaired
        aside_before = {str(x.get("id")) for x in record.get("about_fi") or [] if isinstance(x, dict)}
        _code_review.note_reading(record, version, record["problems"], code)
        self._log_about_fi(record, aside_before)
        self._log.info("[code_review] %s; %s", _code_review.plain_lines(record), _code_review.metrics_line(record))
        if not record["problems"]:
            record["final"] = True
            _code_review.save(self.quest_root, record)
            return False
        _code_review.save(self.quest_root, record)
        return True

    def _log_about_fi(self, record: dict[str, Any], before: set[str]) -> None:
        """One line in run.log for each finding of the code reader that names FI's own lines (not the study's code): it is
        kept in the quality report so FI can be fixed, and it is not repaired and stops nothing."""
        for row in record.get("about_fi") or []:
            if isinstance(row, dict) and str(row.get("id")) not in before:
                self._log.warning("[code_review] %s is about FI's own lines (%s), not the study's code: it is not repaired, "
                                  "takes no result out and stops nothing; it is kept in QUALITY.md so FI can be fixed (%s)",
                                  row.get("id"), row.get("where") or "?", _one_line(str(row.get("text") or ""), 160))

    async def _ask_reader(self, prompt: str, review_node: str, spend: Callable[[], None] | None = None) -> str:
        """One request to the reader; a request that failed is made once more (a reading that fails twice is recorded as
        such, and the quest goes on). ``spend`` is called before each request made (the quest's reader requests are
        counted and bounded, :data:`core.code_review.MAX_READER_CALLS`)."""
        if spend:
            spend()
        try:
            return await self._chat(prompt, node=review_node)
        except Exception as e:  # noqa: BLE001 -- once more; the second failure is the caller's to record
            self._log.warning("[code_review] the request to the reader failed (%s); asking once more",
                              str(e)[:120] or type(e).__name__)
            if spend:
                spend()
            return await self._chat(prompt, node=review_node)

    async def _repair_review_findings(self, state: QuestState, record: dict[str, Any], protocol: Any,
                                      review_node: str) -> bool:
        """One pass of repair over what the last reading found, finding by finding and test first
        (:mod:`core.finding_repair`). The findings are folded into issues with stable ids (:mod:`core.code_findings`); a
        contradiction inside the plan itself goes to the plan's model (:meth:`_route_plan_conflicts`), never to the code;
        for each of the others, the ones that can change a reported result first, the code model is asked for a test that
        fails on the current code (a finding whose test passes is "not demonstrated" and left alone), then for a repair of
        the function(s) involved; a change is kept only when the new test passes and nothing that passed before fails. A
        function that still fails may be rewritten once by the other model configured for review. Every request is counted
        in ``.fi/code_review.json`` before it is made. ``True`` when a file changed (the next reading reads it)."""
        code = self.quest_root / "code"
        await self._route_plan_conflicts(state, record, review_node)
        values = _pairs.plan_values(protocol if isinstance(protocol, dict) else None)
        package = self._package_in_use(state) or ""
        ids = _oracle.generating_equations(protocol if isinstance(protocol, dict) else None)
        same = record.get("same_model")
        reader = str(record.get("reviewer") or "")
        # A round is counted in ``send_backs`` when it finishes (below), so a quest killed in the middle of one gets that round
        # again on resume. Starts are counted too, and bounded (``code_review.may_repair``): a round that keeps dying ends.
        record["round_starts"] = int(record.get("round_starts") or 0) + 1
        _code_review.save(self.quest_root, record)
        fixed_before = sum(1 for i in _code_findings.issues_of(record).values() if i.get("outcome") == _code_findings.FIXED)

        async def ask(prompt: str, who: str, kind: str) -> str:
            node = (review_node + ".repair") if who == "other" else f"implement.finding_{'test' if kind == 'test' else 'repair'}"
            return await self._chat(prompt, node=node)

        async def run_tests(paths: list[str]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for path in paths:  # each test in its own process: one test cannot change what another sees
                _rc, text, _err = await self._run_in_env(["-c", _finding_repair.harness_source("code", [path])],
                                                         timeout_s=180, env=self._finding_test_env())
                out.update(_finding_repair.parse_runs(text, [path]))
            return out

        async def snapshot() -> dict[str, str]:
            return await self._finding_snapshot(state, protocol, record)

        def static() -> dict[str, int]:
            return self._finding_static_counts(state, protocol)

        def on_accept(rel: str, old: str, new: str, issue: dict[str, Any]) -> None:
            record["repaired"] = True
            _code_review.save(self.quest_root, record)
            if Path(rel).name == "experiment.py":
                self._review_experiment_changed = True
            self._gate_commit(state)  # the analysis gate's baseline is the code that is kept (nothing with the switch off)
            self._record(_attempts.LEDGER, lambda: {
                "kind": "repair", "script": Path(rel).name, "attempt": int(record.get("repair_calls") or 0),
                "parent_sha256": _attempts._sha(old.encode("utf-8")), "sha256": _attempts._sha(new.encode("utf-8")),
                "summary": ("the second model's finding " + str(issue.get("id")) + ": " + str(issue.get("text") or ""))[:300]})

        notes: list[str] = []  # what went wrong in the measures of a try (core/finding_repair.py: ``Env.notes``)
        env = _finding_repair.Env(
            root=self.quest_root, code=code, record=record, save=lambda: _code_review.save(self.quest_root, record),
            ask=ask, run_tests=run_tests, snapshot=snapshot, static=static,
            keeps=lambda old, new, rel: (
                _code_review.repair_problem(old, new, rel, equations=ids, package=package)
                or _grid_keys.rejects({rel: old}, {rel: new}, *self._static_inputs(state))),
            prompts=self._prompts, topic=_pairs.scrub(self._topic_block(), values) or "(not stated)",
            equations=_code_review.equations_view(protocol if isinstance(protocol, dict) else None, values),
            scrub=lambda text: _pairs.scrub(text, values), log=self._log, package=package,
            other=(reader if same is False else ""), code_model=self._model_for_node("implement") or self.config.provider.model or "",
            on_accept=on_accept, small_run=self._small_run_again, small_values=self._small_values_now,
            small_missing=self._small_missing_now,
            gate=lambda: self._finding_gate(state),
            known_answers=lambda: self._finding_known_answers(protocol, notes),
            design_response=lambda: self._finding_design_response(protocol, notes), notes=notes)
        await self._gate_baseline(state)
        with self._hold_code_runs():  # each try is its own commit or side reference (finding_repair.keep_try)
            changed = await _finding_repair.repair_issues(env)
        record["repaired"] = bool(changed)
        record["send_backs"] = int(record.get("send_backs") or 0) + 1  # the round is finished
        # Progress is what FI's own test runs show: the faults a failing test showed that now pass it (code_review.note_reading).
        record["round_fixed"] = sum(1 for i in _code_findings.issues_of(record).values()
                                    if i.get("outcome") == _code_findings.FIXED) - fixed_before
        _code_review.save(self.quest_root, record)
        for issue in _code_findings.issues_of(record).values():
            if issue.get("outcome") in (_code_findings.NOT_DEMONSTRATED, _code_findings.GAVE_UP):
                self._log.warning("[code_review] %s (%s) %s: %s", issue.get("id"),
                                  issue.get("function") or issue.get("file") or "?", issue.get("outcome"),
                                  issue.get("reason") or "")
        if changed:
            self._log.info("[code_review] repaired %s; the changed version is read again", ", ".join(sorted(set(changed))))
        return bool(changed)

    async def _finding_snapshot(self, state: QuestState, protocol: Any, record: dict[str, Any]) -> dict[str, str]:
        """What must keep passing while a finding is repaired: the model's equation tests (core/equation_tests.py, from the
        plan's own worked examples) and every finding test accepted before, as ``{"equation:E1": "pass", ...}``."""
        out: dict[str, str] = {}
        if isinstance(protocol, dict) and _equation_tests.eligible(protocol):
            rows = _equation_tests.example_rows(protocol)
            located = _equation_tests.locate(protocol, self._equation_sources(), self._outline_equation_names(state))
            case_list = _equation_tests.cases(rows, located)
            if case_list:
                _equation_tests.write(self.quest_root / "code", case_list)
                env = scrub_secrets()
                raw_dir = self._raw_root() / "equation_tests"
                raw_dir.mkdir(parents=True, exist_ok=True)
                env[_split_run.RAW_DIR_ENV] = _split_run.env_value(raw_dir, self.quest_root)
                # Measured from a copy outside the quest, judged here (core/equation_measure.py).
                results, _rc, _err = await _equation_tests.run_and_judge(self._run_in_env, case_list, timeout_s=300,
                                                                         env=env, outside=True)
                for r in results or []:
                    out[f"equation:{r.get('id')}"] = "pass" if r.get("status") == _equation_tests.OK else "fail"
        accepted = [p for p in record.get("accepted_tests") or [] if isinstance(p, str)]
        for path in accepted:  # each test in its own process: one test cannot change what another sees
            _rc, text, _err = await self._run_in_env(["-c", _finding_repair.harness_source("code", [path])], timeout_s=300,
                                                     env=self._finding_test_env())
            for got, run in _finding_repair.parse_runs(text, [path]).items():
                out[f"finding:{got}"] = run.status
        return out

    async def _finding_known_answers(self, protocol: Any, notes: list[str] | None = None) -> dict[str, dict[str, Any]] | None:
        """What the plan's known-answer checks give on the code as it is now, run by FI as it runs them after the run
        (:meth:`_oracle_gate`; a search is called with the conditions held fixed): ``{check: {"passed", "value"}}``, for the
        repair of a finding to compare before and after. ``None`` when they cannot be run here (the simulation does not follow
        the trial contract, or no check has numbers FI can judge by); the record of the quest's own checks (``needs/ORACLE_CHECK.json``) is not touched."""
        if not isinstance(protocol, dict) or self.config.engine.oracle_check == "off":
            return None
        from . import code_checks as _code_checks

        stride = max(1, int(self.config.engine.replicate_seed_stride))
        return await _code_checks.known_answer_values(
            self.quest_root, protocol, executor=self.executor, python=self.executor.python_path(self.quest_root),
            env=_replicate_env(self.__dict__.get("_last_exec_env"), 0, stride),
            timeout_s=max(30, int(self.config.execution.timeout_s * self.config.engine.pilot_timeout_frac)), log=self._log,
            notes=notes)

    async def _finding_design_response(self, protocol: Any, notes: list[str] | None = None) -> dict[str, Any] | None:
        """For a search for the best design: the objective the study's entry gives on two designs, as
        :func:`core.optimise.design_response`; ``None`` for a study without a search."""
        block = _optimise.block_of(protocol) if isinstance(protocol, dict) else None
        if block is None:
            return None
        normal, _why = _optim.normalize(block)
        if normal is None:
            return None
        from .small_run import budget as _small_run_budget

        return await _optimise.design_response(
            self.executor, self.executor.python_path(self.quest_root), self.quest_root,
            f"code/{_split_run.SIMULATE_NAME}", normal,
            timeout_s=int(_small_run_budget(self.config.execution.timeout_s)),
            env=dict(self.__dict__.get("_last_exec_env") or {}),
            thresholds=protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None, log=self._log,
            notes=notes)

    def _finding_static_counts(self, state: QuestState, protocol: Any) -> dict[str, int]:
        """The number of faults each static check (core/static_checks.py) finds in the code as it is, read without a model."""
        files = self._static_files()
        drafted, design = self._static_inputs(state)
        counts: dict[str, int] = {}
        for name, module, _on in self._static_checks():
            try:
                counts[name] = len(module.problems(files, protocol if isinstance(protocol, dict) else drafted, design))
            except Exception:  # noqa: BLE001 -- a check that cannot be made counts nothing
                counts[name] = 0
        return counts

    async def _route_plan_conflicts(self, state: QuestState, record: dict[str, Any], review_node: str) -> None:
        """A finding that says the plan contradicts itself (two parts define one thing differently) is not a fault of the
        code. It is recorded as a plan problem and put to the plan's model through the same bounded request as the other
        missing parts of the plan (:meth:`_ask_plan_to_complete`); nothing is sent to the code for it. Once the protocol is
        frozen it is only recorded. The plan's model may rewrite only the definitions a contradiction can sit in (the
        metrics, the interval method, the model's equations); a check, a threshold or a tolerance is never changed."""
        conflicts = [i for i in _code_findings.open_issues(record) if i.get("plan_conflict")
                     and i.get("outcome") in (_code_findings.OPEN, _code_findings.REOPENED)]
        if not conflicts:
            return
        for issue in conflicts:
            issue.update(outcome=_code_findings.PLAN_PROBLEM, demonstrated=None,
                         reason="the plan contradicts itself; this is not a fault of the code")
        record["plan_conflicts"] = [{"id": i["id"], "text": i.get("text"), "why": i.get("why")} for i in conflicts]
        _code_review.save(self.quest_root, record)
        self._log.warning("[code_review] %d finding(s) say the plan contradicts itself, so they are not sent to the code: %s",
                          len(conflicts), "; ".join(str(i.get("text"))[:100] for i in conflicts[:3]))
        design, sha = self._design_from_plan()
        if (design is None or not sha or self._client is None or _frozen.load(self.quest_root) is not None
                or not self._runs_code(state)):
            for issue in conflicts:
                issue["plan_asked"] = False
            _code_review.save(self.quest_root, record)
            return
        started = {"sha": sha}
        protocol_now = self._protocol_block(state)
        values = _pairs.plan_values(protocol_now if isinstance(protocol_now, dict) else None)

        twice_before = bool(_code_review.double_definition_gaps(design))  # the design's own protocol, as in lacks() below

        def lacks(_now: Any) -> list[str]:
            if twice_before:  # complete when the name defined two ways is defined once (FI's own fact, not a changed file)
                return _code_review.double_definition_gaps(_now if isinstance(_now, dict) else None)
            try:
                changed = _plan.sha256(_plan.plan_path(self.quest_root).read_text(encoding="utf-8")) != started["sha"]
            except OSError:
                changed = True
            return [] if changed else [_pairs.scrub(str(i.get("why") or i.get("text") or ""), values) for i in conflicts]

        def request(_now: Any, _text: str) -> str:
            return _code_findings.plan_conflict_request(
                state.get("topic") or self.config.topic,
                [(str(i.get("text") or ""), str(i.get("why") or "")) for i in conflicts], values)

        def keep(before: str, answer: dict[str, Any]) -> str:
            kept = self._keep_only_the_parts(before, answer, list(_code_findings.PLAN_PARTS))
            if kept is None or _plan.sha256(kept) == _plan.sha256(before):
                raise ValueError("your answer resolves none of the contradictions in a form that can be read; "
                                 "plan.md is unchanged")
            return kept

        await self._ask_plan_to_complete(
            design, sha, lacks=lacks, request=request, keep=keep, file="code_plan_conflict_asked.json",
            what="a plan that does not contradict itself", short="a contradiction in the plan",
            note="a contradiction found by the code reader, resolved by the plan's model")
        try:
            after = _plan.sha256(_plan.plan_path(self.quest_root).read_text(encoding="utf-8"))
        except OSError:
            after = sha
        for issue in conflicts:
            issue["plan_asked"] = True
            issue["plan_changed"] = after != sha
        _code_review.save(self.quest_root, record)

    def _code_review_failed_run(self, message: str, *, give_up: str = "", script: str | None = None) -> QuestState:
        """A run that did not happen, for the repair path to take: the directive stands where a traceback would."""
        patch: QuestState = {
            "exec_result": {
                "returncode": 1, "duration_s": 0.0, "timed_out": False, "stdout_tail": "", "stderr_tail": message, "directive": True,
                "packages_note": getattr(self, "_packages_note", ""), "failed_script": script, "flat_output": "",
                "numeric_warnings": [],
            },
            "figures": [], "figure_records": {}, "result_json": {}, "exec_patch_pending": False,
            "result_json_replicates": [], "result_json_deterministic": False, "result_json_trials": False,
            "result_json_replicate_seed_ignored": False, "result_json_no_random_source": False,
        }
        if give_up:
            patch["exec_give_up_reason"] = give_up
        return patch

    def _code_review_outcome(self, record: dict[str, Any]) -> QuestState | None:
        """The code, read and repaired as far as the budget goes, still has findings open in the reader's view. The reader
        can be wrong, so nothing stops before the run. After it, a high-severity finding about a headline quantity that
        the run's results do not contradict ends in the honest stop (:meth:`_no_results_verdict`); what the results do
        hold is recorded as a disagreement; anything else open is said in the paper's limitations."""
        problems = [v for v in record.get("problems") or [] if isinstance(v, dict)]
        for v in problems:
            self._log.warning("[code_review] still %s after the repairs: %s%s",
                              "flagged" if v.get("kind") == "logic" else
                              "missing" if v.get("status") == _code_review.MISSING else "implemented differently", v["text"],
                              " (the run's results decide whether the study stops)" if v.get("headline") else "")
        return None

    def _finding_test_env(self, *, stand_ins: bool = True) -> dict[str, str]:
        """The environment a finding test runs in: the process's own plus the ``FI_*`` variables the run gives the code
        (:func:`core.finding_repair.run_environment`), so a module that reads them when it is imported can be tested. Before
        the run a stand-in holds a file the run has not written yet; at the evidence gate (``stand_ins=False``) only real files
        are used."""
        return {**scrub_secrets(), **_finding_repair.run_environment(
            self.quest_root, _split_run.env_value(self._raw_root(), self.quest_root), stand_ins=stand_ins)}

    def _time_note(self, state: QuestState) -> str:
        """What the plan is told about the topic's own runtime request next to FI's enforced limit (read the way the
        paper's run record reads it, :func:`core.disclosure._topic_limit`). FI sizes the run to the shorter of the two
        (``RunChecksMixin._size_the_run``): over it, the runs per setting are lowered, never the settings."""
        required = _disclosure._read_json(self.quest_root / ".fi" / "topic_requirements.json")
        stated = [c if isinstance(c, str) else c.get("stated", "") if isinstance(c, dict) else ""
                  for c in (required.get("constraints") if isinstance(required, dict) else None) or []]
        limit = _disclosure._topic_limit([c for c in stated if isinstance(c, str)], str(state.get("topic") or ""))
        asked = (f"The topic asks for a run of at most {_disclosure._plain_seconds(limit)}" if limit is not None
                 else "If the topic asks for a shorter run")
        return (f"This is the limit FI enforces on one run. {asked}: plan to stay within the shorter of the two. Before the "
                "run FI times a few trials; a run over the shorter limit gets fewer runs per setting from FI (never fewer "
                "settings), below the plan's min_runs_per_setting (down to 10) only for the topic's own limit, and the paper says so. "
                "Each setting runs in a process of its own, whose start is paid once whatever the runs; the paper "
                "reports how long the run took.")

    async def _run_in_env(self, argv: list[str], *, timeout_s: int = 180,
                          env: dict[str, str] | None = None, outside: bool = False,
                          apart: bool = False) -> tuple[int, str, str]:
        """Run ``python <argv>`` from the quest folder, in the quest's environment (a venv, the shared interpreter, or the
        Docker container): ``(returncode, stdout, stderr)``. ``outside``: from a copy of ``code/`` outside the quest
        folder (core/outside_run.py), so FI's records are not next to the code that runs. ``apart``: from such a copy
        that does not share the study's kept values and gives back only the paths ``argv`` names."""
        py = self.executor.python_path(self.quest_root)
        # A function rewritten within the same second to the same size would be read from its stale compiled copy.
        for cache in (self.quest_root / "code").glob("**/__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        executor = (_outside_run.OutsideExecutor(self.executor, self.quest_root, log=self._log, share=not apart,
                                                 only_named=apart) if outside or apart else self.executor)
        env = {k: v for k, v in env.items() if k != "FI_CHECKS_DIR"} if env is not None else None
        try:
            # From the quest folder with paths relative to it: in Docker that folder is what is mounted as /work.
            ran = await executor.execute([str(py), "-B", *argv], cwd=self.quest_root, timeout_s=timeout_s, env=env)
        except Exception as e:  # noqa: BLE001 -- a check that could not start is a failed check, said as such
            return -1, "", repr(e)
        return ran.returncode, ran.stdout or "", (ran.stderr or "") + ("\n(timed out)" if ran.timed_out else "")

    def _run_apart_from_study(self, exec_env: dict[str, str] | None, *, comparison: bool = False) -> Any:
        """The run of a check that calls the study's own code (what the tested functions are called with, what depends on
        them, the second version of a function): apart from the study, so it neither reads the values the study kept
        (data/results/cache/) nor leaves its own there, and with the environment a trial gets. ``extra_s``: time FI
        measured for something the call does first (the study's own trial), added to the limit, ``comparison``: the run of the
        second-version comparison, which had a fixed 180 s and an environment of ``None`` (the process's own) before it ran
        apart: its base limit is never under :data:`core.second_impl.COMPARE_BASE_S` and a ``None`` stays ``None``; the
        capture's runs have always been given an empty dict."""
        from .small_run import budget as _small_run_budget

        async def run(argv: list[str], extra_s: float = 0.0) -> Any:
            return await self._run_in_env(
                argv, timeout_s=max(_second_impl.COMPARE_BASE_S if comparison else 0,
                                    _small_run_budget(self.config.execution.timeout_s)) + int(math.ceil(max(0.0, extra_s))),
                env=(None if exec_env is None and comparison else dict(exec_env or {})), apart=True)

        return run

    async def _run_outside(self, argv: list[str], **kwargs: Any) -> tuple[int, str, str]:
        """:meth:`_run_in_env` from a copy of ``code/`` outside the quest folder (core/outside_run.py)."""
        return await self._run_in_env(argv, outside=True, **kwargs)

    def _function_steps_partial_note(self) -> str:
        """For the code-writing prompt when the package was kept with functions that still fail their check."""
        record = _function_steps.load(self.quest_root)
        names = _function_steps.failing(record) if record.get("status") == _function_steps.PARTIAL else []
        if not names:
            return ""
        return ("\nThe function" + ("s " if len(names) > 1 else " ") + ", ".join(f"`{n}`" for n in names)
                + (" do" if len(names) > 1 else " does") + " not yet pass FI's check against the plan's equation (FI tests "
                "it again before the run). Call the package's functions as they are; do not copy an equation into simulate.py.\n")

    def _function_steps_note(self, package: str) -> str:
        """For the code-writing prompt when the function-by-function attempt ended: which function it could not get right
        (its name and equation, nothing else)."""
        record = _function_steps.load(self.quest_root)
        if record.get("package") != package or record.get("status") != "fell_back":
            return ""
        failed = [f"`{n}` (equation {e.get('equation')})" for n, e in (record.get("functions") or {}).items()
                  if isinstance(e, dict) and e.get("status") == _function_steps.FAILED]
        return ("\nFI first tried to write the package one function at a time and " + ", ".join(failed or ["one function"])
                + " still failed its checks: take extra care that each function computes exactly its equation.\n")

    def _equation_sources(self) -> dict[str, str]:
        """The files of ``code/`` that can hold a function an equation test imports: the packages and ``simulate.py``
        (``experiment.py`` runs when it is imported, so it is never one)."""
        code = self.quest_root / "code"
        out = dict(_code_layout.package_sources(code))
        simulate = code / _split_run.SIMULATE_NAME
        if simulate.is_file():
            out[_split_run.SIMULATE_NAME] = simulate.read_text(encoding="utf-8")
        return out

    def _new_function_filler(self, outline: dict[str, Any], package: str, protocol: Any, specs: list[Any],
                             passages: dict[str, str], deps: list[str]) -> Any:
        """The filler that writes, and later tests or repairs, the model's functions: the quest-wide count of its requests
        (``function_steps.max_calls_for``) is the one in ``.fi/function_steps.json``."""
        view = _oracle.model_view((protocol or {}).get("model") if isinstance(protocol, dict) else None) or {}
        constants = "\n".join(f"- {c.get('name')} = {c.get('value')} ({c.get('source')})"
                              for c in outline.get("constants") or [] if isinstance(c, dict) and c.get("name"))

        async def chat(prompt: str) -> str:
            return await self._chat(prompt, node="implement")

        return _function_steps.FunctionFiller(
            quest_root=self.quest_root, package=package, protocol=protocol if isinstance(protocol, dict) else None,
            summary=str(view.get("summary") or ""),
            prompts={"fill": self._prompts["implement_function"], "repair": self._prompts["implement_function_repair"],
                     "module": self._prompts["implement_module_repair"]},
            chat=chat, run=self._run_outside, log=self._log, repairs=_function_steps.REPAIRS,
            max_calls=_function_steps.max_calls_for(len(specs)), constants=constants, install=self._install_packages,
            declared=deps, topic_block=self._topic_block(), passages=passages,
            values=_function_steps.plan_constants(outline, protocol if isinstance(protocol, dict) else None, self._log))

    def _write_plan_constants(self, state: QuestState, package: str) -> None:
        """FI's block of the plan's numbers in every module of the package kept from an earlier pass is made again from the
        plan (:func:`core.function_steps.write_package_constants`), so a newly written simulate.py need not set them and a
        number the plan changed is changed there."""
        protocol = self._protocol_block(state)
        values = _function_steps.plan_constants(state.get("implement_outline"), protocol if isinstance(protocol, dict) else None,
                                                self._log)
        _function_steps.write_package_constants(self.quest_root / "code" / package, values, self._log)

    async def _step_passages(self, protocol: dict[str, Any] | None, specs: list[Any]) -> dict[str, str]:
        """``{step or equation id: passages}`` for the functions of ``specs`` (core/step_passages.py): the best hits of the
        quest's knowledge for the card's name and what it does, kept to the source the card names, and the plan's quoted
        definitions of the metrics it computes. No model call; cached in ``.fi/step_passages.json``. An id with nothing is
        left out; a search that fails leaves the function to be written without passages, as before."""
        out: dict[str, str] = {}
        if not isinstance(protocol, dict):
            return out
        try:
            knowledge = self.knowledge if getattr(self.knowledge, "enabled", False) else None
            definitions = _ledger.parse_definitions(protocol)[0]
            equations = {_equation_tests._text(e["id"]): e for e in _equation_tests.eligible(protocol)}
            for spec in specs:
                if spec.kind == _function_steps.STEP and spec.card:
                    card = spec.card
                else:
                    e = equations.get(spec.equation) or {}
                    card = {"id": spec.equation, "name": spec.purpose, "formula": _equation_tests._text(e.get("formula")),
                            "source": _equation_tests._text(e.get("source"))}
                if not _step_passages._source_terms(str(card.get("source") or "")):
                    # "derivation", "textbook": no paper to look for, so none is searched; the plan's quoted definitions stay
                    card = {**card, "source": ""}
                text = await _step_passages.for_card(card, knowledge, self.fi_dir / "step_passages.json",
                                                     definitions=definitions)
                if text:
                    out[spec.equation] = text
        except Exception as e:  # noqa: BLE001 -- passages are a help to the writer, never a stop
            self._log.info("[implement] the literature passages for the model's functions could not be found: %r", e)
        return out

    def _family_of_node(self, node: str) -> str | None:
        """Which kind of model answers ``node`` (the vendor family, from the model's name, else the provider's), so that
        agreement between two versions of a function can be told from agreement between two models of one kind."""
        step = node_provider_for(getattr(self.config.provider, "node_providers", None), node)
        name = str(step[1].name if step else self.config.provider.name or "")
        model = str((step[1].model if step else None) or self._model_for_node(node) or self.config.provider.model or "")
        for text in (model.lower(), name.lower()):
            for family, keys in (("openai", ("gpt", "codex", "openai", "terra")), ("anthropic", ("claude", "anthropic", "haiku", "sonnet", "opus")),
                                 ("google", ("gemini", "gemma", "antigravity", "agy")), ("moonshot", ("kimi", "moonshot"))):
                if any(k in text for k in keys):
                    return family
        return name or None

    async def _equation_test_failures(self, protocol: dict[str, Any], exec_env: Any,
                                      names: dict[str, str] | None = None) -> str:
        """The plan's equation tests run once on the package as it is (no repair, no request): ``""`` when none fails or the plan
        has none to run, else the failing equations' ids (or why the started tests gave no results)."""
        if self.config.engine.oracle_check == "off" or not (self.quest_root / "code" / _split_run.SIMULATE_NAME).is_file():
            return ""
        if not _equation_tests.eligible(protocol):
            return ""
        rows = _equation_tests.example_rows(protocol)
        located = _equation_tests.locate(protocol, self._equation_sources(), names)
        case_list = _equation_tests.cases(rows, located)
        if not case_list:
            return ""
        _equation_tests.write(self.quest_root / "code", case_list)
        env = {**(exec_env or scrub_secrets())}
        raw_dir = self._raw_root() / "equation_tests"
        raw_dir.mkdir(parents=True, exist_ok=True)
        env[_split_run.RAW_DIR_ENV] = _split_run.env_value(raw_dir, self.quest_root)
        # A run that crashed, hung or printed no result fails every test (equation_tests.judged), never a pass: a repair
        # that leaves a model the tests cannot even run is not kept.
        results, _rc, _err = await _equation_tests.run_and_judge(self._run_in_env, case_list, timeout_s=300, env=env,
                                                                 outside=True)
        return ", ".join(str(r.get("id")) for r in results if r.get("status") != _equation_tests.OK)

    async def _second_version_gate(self, state: QuestState, py: Any, exec_env: Any) -> bool:
        """After the equation tests and before the oracle gate: for up to ``second_impl.MAX_KEY`` key functions of the
        model, a second version is written by another request (steps ``second_impl`` and ``second_impl.adjudicate``, which
        may be other models) from the function's description alone, and the two are run on the same inputs in the quest's
        environment (core/second_impl.py). A difference is settled by a judge; the first function is repaired only if the
        repair still passes its own checks. What cannot be settled is recorded (``.fi/second_impl.json``) and said in the
        paper's limitations; nothing here stops a quest or asks anyone. Counts are saved before each request, so a resume
        asks nothing twice. ``True`` when a repair changed the model's package."""
        ex = self.config.execution
        try:
            if not ex.code_function_steps or not self._runs_code(state):
                return False
            package = self._package_in_use(state)
            protocol = self._protocol_block(state)
            if not package or not isinstance(protocol, dict):
                return False
            record = _function_steps.load(self.quest_root)
            if record.get("package") != package:
                return False  # the package was not written function by function: there are no cards to write a second version from
            outline = state.get("implement_outline") or {}
            specs, _notes = _function_steps.model_functions(outline, protocol)
            done = {n for n, e in (record.get("functions") or {}).items()
                    if isinstance(e, dict) and e.get("status") == _function_steps.DONE}
            specs = [s for s in specs if s.name in done]
            if not specs or not _second_impl.key_functions(specs, protocol)[0]:
                return False  # no card or equation to check (or only random ones): nothing is asked, nothing is said
            if ex.sandbox == "venv" and not self.executor.python_path(self.quest_root).is_file():
                return False
            deps = _coerce_dep_list(outline.get("deps")) + _coerce_dep_list((state.get("design") or {}).get("dependencies"))
            passages = await self._step_passages(protocol, specs)
            filler = self._new_function_filler(outline, package, protocol, specs, passages, deps)
            filler.know(specs)
            by_name = {s.name: s for s in specs}

            async def chat(node: str, messages: list[dict[str, str]]) -> str:
                return await self._chat("\n\n".join(str(m.get("content") or "") for m in messages), node=node)

            async def repair_main(name: str, problem: str, _cases: list[dict[str, Any]]) -> bool:
                return await filler.repair_function(by_name[name], problem)

            async def still_passes(name: str) -> bool:
                return await filler.still_passes(by_name[name])

            async def verify_package() -> str:
                return await self._equation_test_failures(protocol, exec_env, self._outline_equation_names(state))

            captured = await self._captured_calls(state, protocol, package, specs, exec_env)
            before = self._package_snapshot()
            result = await _second_impl.second_versions(
                verify_package=verify_package, captured=captured,
                quest_root=self.quest_root, package=package, specs=specs, protocol=protocol,
                prompts={"write": self._prompts["second_impl_write"], "adjudicate": self._prompts["second_impl_adjudicate"]},
                chat=chat, run=self._run_apart_from_study(exec_env, comparison=True), repair_main=repair_main, still_passes=still_passes, passages=passages,
                family_of=self._family_of_node, topic_block=self._topic_block(), log=self._log)
            for name, e in (result.get("functions") or {}).items():
                if not isinstance(e, dict):
                    continue
                status, adj = e.get("status"), e.get("adjudication") or {}
                if adj.get("outcome") == "repair accepted":
                    self._log.info("[implement] a second version of %s showed the first was wrong: the first was repaired and "
                                   "still passes its own checks", name)
                if status == _second_impl.AGREE:
                    self._log.info("[implement] a second, independently written version of %s agrees on %d inputs%s", name,
                                   int(e.get("inputs_tried") or 0),
                                   " (both versions were written by the same kind of model, so this counts for less)"
                                   if e.get("same_family") else "")
                elif status == _second_impl.UNRESOLVED:
                    self._log.warning("[implement] a second version of %s differs from the first and FI could not settle which "
                                      "is right; the paper's limitations say so", name)
                elif status == _second_impl.UNCONFIRMED:
                    self._log.info("[implement] a second version of %s could not be compared with the first (%s); the paper's "
                                   "limitations say so", name, e.get("why") or "no reason recorded")
                elif status == _second_impl.SKIPPED:
                    self._log.info("[implement] %s was not compared with a second version: %s", name, e.get("why") or "skipped")
            return self._package_snapshot() != before
        except _ModelAnswerProblem:
            raise
        except Exception as e:  # noqa: BLE001 -- a second version is a check on top; its failure never stops a quest
            self._log.warning("[implement] the check with a second version of the model's functions could not be made: %r", e)
            return False

    async def _captured_calls(self, state: QuestState, protocol: dict[str, Any], package: str, specs: list[Any],
                              exec_env: Any, *, depends: bool = False) -> dict[str, list[dict[str, Any]]]:
        """What the key functions are called with when FI runs the simulation (core/second_impl.py::capture_inputs): at
        the settings of FI's small run (core/small_run.py::settings; each part of a study takes other paths through the
        code), a search's baseline design first; with ``depends`` then whether the headline results depend on each of
        them (core/second_impl.py::dependence: each one's result changed where it is called). ``{}`` when no run can be
        made."""
        if depends:
            (self.quest_root / _second_impl.DEPENDENCE).unlink(missing_ok=True)  # a record of earlier code says nothing
        selected, _skipped = _second_impl.key_functions(specs, protocol)
        names = [s.name for s in selected]
        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME
        if not names or not simulate.is_file():
            return {}
        try:
            from . import small_run as _small

            block = _optimise.block_of(protocol)
            if block is not None:
                normal, _why = _optim.normalize(block)
                if normal is None:
                    return {}
                cells = [_optimise_search.cell_of(normal, (normal.get("baseline") or {}).get("values") or {})]
                kinds = ["search"]  # the baseline design is a design of the search; the scan's cells are scan rows
                if isinstance(normal.get("grid"), dict) and normal["grid"]:  # a search's first scan: a measurement
                    cells += _small.settings(_optimise_search.scan_grid(normal))
                    kinds += ["scan"] * (len(cells) - 1)
                # The scan's cells leave them out and the trial runner gives each cell all of them (the baseline cell
                # holds them already, with the same values).
                inside = _optimise_search.in_each_run(normal) or None
            else:
                grid = _optimise_search.study_grid(protocol) if isinstance(protocol.get("grid"), dict) else {}
                cells = _small.settings(grid)
                kinds = ["scan"] * len(cells)
                inside = protocol.get("in_each_run") if isinstance(protocol.get("in_each_run"), dict) else None
            if not cells:
                return {}
            deterministic = "run_trial" not in _trial_runner.entries(simulate)
            thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None

            from .small_run import budget as _small_run_budget

            run = self._run_apart_from_study(exec_env)

            budget = 2 * _small_run_budget(self.config.execution.timeout_s)  # each of the two checks, in all
            got = await self._await_with_heartbeat(
                _second_impl.capture_inputs(run, self.quest_root, package, names, cells, deterministic=deterministic,
                                            in_each_run=inside, thresholds=thresholds, budget_s=budget, kinds=kinds),
                label="recording what the tested functions are called with")
        except Exception as e:  # noqa: BLE001 -- a record of the calls is an aid to a check; never a stop
            self._log.info("[implement] what the model's functions are called with could not be recorded: %r", e)
            return {}
        if got:
            self._log.info("[implement] FI ran the simulation and recorded what %d checked function(s) are called with: %s",
                           len(got), ", ".join(sorted(got)))
        if not depends:
            return got
        try:
            found = await self._await_with_heartbeat(
                _second_impl.dependence(run, self.quest_root, package, names,
                                        self._headline_result_names(state, protocol, dependent=False),
                                        deterministic=deterministic,
                                        in_each_run=inside, thresholds=thresholds, budget_s=budget),
                label="checking whether the headline results use the tested functions")
            for line in _second_impl.dependence_sentences(found):
                self._log.warning("[execute] %s", line)
        except Exception as e:  # noqa: BLE001 -- a check on top; never a stop
            self._log.info("[execute] whether the headline results depend on the checked functions could not be "
                           "checked: %r", e)
        return got

    async def _dependence_check(self, state: QuestState, exec_env: Any) -> None:
        """After the study ran on code no step changes any more: whether its headline results are computed with the
        model's tested functions (:meth:`_captured_calls` with ``depends``; core/second_impl.py::dependence). A function
        the headline results do not use is a gap in the evidence. Never a stop; skipped on a resume when the record was
        made on this code."""
        try:
            if not self.config.execution.code_function_steps or not self._runs_code(state):
                return
            if _second_impl.read_dependence(self.quest_root):
                return  # already made on this code (a resume)
            package = self._package_in_use(state)
            protocol = self._protocol_block(state)
            record = _function_steps.load(self.quest_root)
            if not package or not isinstance(protocol, dict) or record.get("package") != package:
                return
            specs, _notes = _function_steps.model_functions(state.get("implement_outline") or {}, protocol)
            done = {n for n, e in (record.get("functions") or {}).items()
                    if isinstance(e, dict) and e.get("status") == _function_steps.DONE}
            specs = [s for s in specs if s.name in done]
            if specs:
                await self._captured_calls(state, protocol, package, specs, exec_env, depends=True)
        except Exception as e:  # noqa: BLE001 -- a check on top; never a stop
            self._log.info("[execute] whether the headline results use the tested functions could not be checked: %r", e)

    def _first_study_cell(self, protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]] | None] | None:
        """The plan's first setting for a call of the study's entry: a search's baseline design, else the middle of every
        axis of the grid (core/small_run.py::settings), the values each run compares inside it, and what the call is for
        (``"search"`` or ``"scan"``). ``None`` when the plan gives none."""
        from . import small_run as _small

        try:
            block = _optimise.block_of(protocol)
            if block is not None:
                normal, _why = _optim.normalize(block)
                if normal is None:
                    return None
                cell = _optimise_search.cell_of(normal, (normal.get("baseline") or {}).get("values") or {})
                return dict(cell), (_optimise_search.in_each_run(normal) or None), "search"
            grid = _optimise_search.study_grid(protocol) if isinstance(protocol.get("grid"), dict) else {}
            cells = _small.settings(grid)
            if not cells:
                return None
            inside = protocol.get("in_each_run") if isinstance(protocol.get("in_each_run"), dict) else None
            return dict(cells[0]), inside, "scan"
        except (KeyError, TypeError, ValueError):
            return None

    async def _simulation_repair_fault(self, state: QuestState, new_code: str) -> str:
        """After a repair of simulate.py is written to disk: why it cannot stay, or ``""``: the checks of
        :meth:`_simulation_entry_fault`, then the analysis gate (core/pipeline_gate.py; nothing with the switch off)."""
        return await self._simulation_entry_fault(state, new_code) or await self._gate_refusal(state)

    async def _simulation_entry_fault(self, state: QuestState, new_code: str) -> str:
        """The checks of the study's entry (without the analysis gate). Two facts, both from FI's own calls
        of the code: a ``run_trial`` that its code shows reaching ``run_cell`` is kept when FI called it with two seeds on
        the plan's first setting and the numbers were the same (one trial is the setting's number); and the function FI
        calls for the study, called once on that setting within the small run's time, must return at least one of the
        study's headline results (FI's record of the plan), or a repair that routes every setting to a check branch
        leaves a study with no result."""
        # The same code (the script and the model's package) that was judged fine is not called again.
        seen = _trial_runner.code_inputs_sha(self.quest_root, new_code)
        if self.__dict__.get("_entry_judged_fine") == seen:
            return ""
        fault = await self._simulation_repair_fault_uncached(state, new_code)
        if not fault:
            self._entry_judged_fine = seen
        return fault

    async def _simulation_kept_or_back(self, state: QuestState, path: Path, old_code: str,
                                       package_before: dict[str, str], node: str) -> str:
        """For every place that adopts a changed simulation: the check of :meth:`_simulation_repair_fault` on the file now on
        disk (``path``: simulate.py; any other file is not this check's). On a fault the old text and package are put back and
        the sentence is returned; ``""`` keeps the change."""
        # The baseline is the gate's result for the code as it was just before this write (nothing with the switch off).
        await self._gate_baseline_before(state, path, old_code, package_before)
        if path.name != _split_run.SIMULATE_NAME:
            # not the entry check's, but the analysis runs this file (experiment.py, a helper): the analysis gate
            fault = await self._gate_refusal(state)
        else:
            fault = await self._simulation_repair_fault(state, path.read_text(encoding="utf-8"))
        if fault:
            path.write_text(old_code, encoding="utf-8")
            self._restore_package(package_before)
            self._log.warning("[%s] the change to %s is not kept: %s", node, path.name, fault)
        else:
            self._gate_commit(state)
        return fault

    async def _simulation_repair_fault_uncached(self, state: QuestState, new_code: str) -> str:
        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME
        if not _trial_runner.study_entry(_trial_runner.entries(simulate)):
            return ""
        protocol = self._protocol_block(state)
        first = self._first_study_cell(protocol) if isinstance(protocol, dict) else None
        aggregate = _trial_runner.trial_aggregates(new_code, self.quest_root)
        if first is None:
            if aggregate:
                self._log.warning("[execute] FI could not measure whether run_trial runs one trial (the plan gives no first "
                                  "setting to call it on); the reading of the code alone does not refuse it")
            return ""
        cell, inside, kind = first
        py = self.executor.python_path(self.quest_root)
        env = dict(self.__dict__.get("_last_exec_env") or {})
        from . import run_estimate as _estimate
        from .small_run import budget as _small_run_budget

        # One budget for the whole check of one repair (the small run's time): the two seed calls and the call of the
        # study's entry share it.
        budget = float(_small_run_budget(self.config.execution.timeout_s))
        began = time.monotonic()

        def left() -> int:
            return int(budget - (time.monotonic() - began))

        module = f"code/{_split_run.SIMULATE_NAME}"
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
        try:
            if aggregate:
                settled = await self._await_with_heartbeat(
                    _trial_runner.settle_aggregate(
                        self.executor, py, self.quest_root, module, cell=cell, inside=inside,
                        timeout_s=max(1, min(int(self.config.execution.timeout_s), left())), env=env,
                        thresholds=thresholds, log=self._log, kind=kind),
                    label="calling the repaired trial function with two seeds")
                if settled is False:
                    return _trial_runner.TRIAL_AGGREGATE_WHY
                if settled is None:
                    return ""  # not judged, and no third call
            if left() < 1:
                self._log.warning("[execute] the check of the repaired code ran out of its %d s before the study's entry "
                                  "was called: not judged", int(budget))
                return ""
            block = _optimise.block_of(protocol)
            if block is not None:
                normal, _why = _optim.normalize(block)
                if normal is None:
                    return ""
                return await self._await_with_heartbeat(
                    _optimise.search_entry_returns_objective(
                        self.executor, py, self.quest_root, module, normal, {**cell, **(inside or {})}, timeout_s=left(),
                        env=env, thresholds=thresholds, log=self._log),
                    label="calling the repaired code on the search's baseline design")
            # Only the headlines FI's own record shows the entry returns (its small run's record of what it gave): a metric the
            # analysis derives later is not the entry's to return. With no such headline there is nothing to judge.
            from . import small_run as _small

            gave = {str(n) for n in _small.read_record(self.quest_root).get("returned") or []}
            wanted = {str(m.get("id")).strip() for m in protocol.get("metrics") or []
                      if isinstance(m, dict) and str(m.get("id") or "").strip()
                      and gave & set(_metric_spec.result_forms(str(m.get("id")).strip()))}
            if not wanted:
                self._log.info("[execute] the repaired code was not judged on the plan's first setting: FI has no record "
                               "that the study's entry returns any headline metric itself")
                return ""
            return await self._await_with_heartbeat(
                _estimate.entry_returns_study(self.executor, py, self.quest_root, module, cell, wanted=wanted,
                                              timeout_s=left(), env=env, thresholds=thresholds, in_each_run=inside),
                label="calling the repaired code on the plan's first setting")
        finally:
            self._log.info("[execute] checking the repaired code took %.0f s of the %d s allowed",
                           time.monotonic() - began, int(budget))

    def _headline_result_names(self, state: QuestState, protocol: dict[str, Any] | None, *,
                               dependent: bool = True) -> set[str]:
        """The headline results: the protocol's headline metrics and (with ``dependent``) the design's dependent
        variables, and for a search what it optimises and what limits it. Without ``dependent`` only names a run
        returns its results under: a dependent variable is often written as a phrase."""
        protocol = protocol if isinstance(protocol, dict) else {}
        design = state.get("design") if isinstance(state.get("design"), dict) else {}
        variables = design.get("variables") if isinstance(design.get("variables"), dict) else {}
        names = set(_criteria.headline_names(protocol)) | (
            {str(v) for v in variables.get("dependent") or []} if dependent else set())
        block = protocol.get("optimisation") if isinstance(protocol.get("optimisation"), dict) else None
        if block is not None:
            names |= {str((block.get("objective") or {}).get("quantity") or "")} | {
                str(c.get("quantity") or "") for c in block.get("constraints") or [] if isinstance(c, dict)}
        names.discard("")
        return names

    async def _fill_model_functions(self, state: QuestState, outline: dict[str, Any]) -> bool:
        """After the outline, before the code is written whole: fill the model's package one function at a time, each
        import-checked and tested against the plan's worked example for its equation, a failing function repaired alone
        (bounded). ``True`` when the package is written and checked, so the code-writing request asks for the two scripts
        only; ``False`` (the code is written whole, as before) when the step does not apply or a function still fails
        after its repairs. Never stops a quest."""
        ex = self.config.execution
        try:
            layout = self._code_layout(state)
            if (not ex.code_function_steps or not layout
                    or layout["shape"] != _code_layout.PACKAGE):
                return False
            protocol = self._protocol_block(state)
            specs, notes = _function_steps.model_functions(outline, protocol if isinstance(protocol, dict) else None)
            if not specs:
                self._log.info("[implement] the model's functions are written with the rest of the code (%s)",
                               "; ".join(notes) or "the outline names no function per equation")
                return False
            if ex.sandbox == "venv" and not self.executor.python_path(self.quest_root).is_file():
                self._log.info("[implement] the model's functions are written with the rest of the code: the quest's "
                               "environment is not there to check them in")
                return False
            package = layout["package"]
            record = _function_steps.load(self.quest_root)
            if (self.quest_root / "code" / package / _code_layout.MODEL_NAME).is_file() and record.get("package") != package:
                return False  # a package written whole earlier is not replaced by this step
            deps = _coerce_dep_list(outline.get("deps")) + _coerce_dep_list((state.get("design") or {}).get("dependencies"))
            install_list, _dropped = _experiment_deps.split_deps(
                deps, local_modules=[p.stem for p in (self.quest_root / "code").glob("*.py")])
            if install_list:
                failed = await self._install_packages(install_list)
                for dep, why in failed:
                    self._log.info("[implement] %r could not be installed for the function checks: %s", dep, why)
            # Each function is written from its step card or equation and the passages of the source the plan cites for it
            # (found here, without a model call, and kept in .fi/step_passages.json: a resume does not search again).
            passages = await self._step_passages(protocol if isinstance(protocol, dict) else None, specs)
            filler = self._new_function_filler(outline, package, protocol, specs, passages, deps)
            outcome = await filler.run_all(specs)
            if outcome.status == "done":
                self._log.info("[implement] the model's package is written: %d function(s), each checked "
                               "(%d request(s) to the model in this quest)", len(specs), outcome.calls)
                return True
            if outcome.status == _function_steps.PARTIAL:
                self._log.info("[implement] the model's package is written with %d of %d function(s) passing their checks "
                               "(%d request(s) to the model in this quest)",
                               *_function_steps.counts(_function_steps.load(self.quest_root)), outcome.calls)
                return True
            return False
        except _ModelAnswerProblem:
            raise
        except Exception as e:  # noqa: BLE001 -- writing the code function by function must never stop a quest
            self._log.warning("[implement] writing the model's functions one at a time failed (%r); the code is written "
                              "whole", e)
            return False

    async def _equation_gate(self, state: QuestState, py: Any, exec_env: Any) -> QuestState | None:
        """Before the oracle gate and the study: one small test per equation of the plan's model, written by FI from the
        plan's own worked example (core/equation_tests.py), run in the quest's environment. A failing function is
        repaired alone (the function and its equation, never the expected value), at most
        ``function_steps.REPAIRS`` times and within ``function_steps.MAX_CALLS`` in the quest;
        what still fails is returned as a failed run, so the existing whole-script repair and the honest stop apply.
        An equation that cannot be tested this way is said plainly and the quest goes on. ``None`` when nothing fails."""
        if self.config.engine.oracle_check == "off" or not self._runs_code(state):
            return None
        protocol = self._protocol_block(state)
        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME
        if not isinstance(protocol, dict) or not simulate.is_file() or not _equation_tests.eligible(protocol):
            return None
        ex = self.config.execution
        rows = _equation_tests.example_rows(protocol)
        record: dict[str, Any] = {"equations": {r["id"]: {"state": r["state"], "why": r["why"]} for r in rows},
                                  "untested_core": [r["id"] for r in rows if r["state"] == "unprotected"]}
        env = {**(exec_env or scrub_secrets())}
        raw_dir = self._raw_root() / "equation_tests"
        raw_dir.mkdir(parents=True, exist_ok=True)
        env[_split_run.RAW_DIR_ENV] = _split_run.env_value(raw_dir, self.quest_root)
        spent: dict[str, int] = {}
        failures: list[dict[str, Any]] = []
        rederived: list[str] = []
        fixed = self._outline_signatures(state)
        # One more pass than the repairs allow: it is made only when the plan has worked a failing example out again.
        for attempt in range(_function_steps.REPAIRS + 2):
            sources = self._equation_sources()
            located = _equation_tests.locate(protocol, sources, self._outline_equation_names(state))
            case_list = _equation_tests.cases(rows, located)
            if self._match_example_names(case_list, located, sources):  # FI's reading of the code is of the renamed code
                sources = self._equation_sources()
                located = _equation_tests.locate(protocol, sources, self._outline_equation_names(state))
                case_list = _equation_tests.cases(rows, located)
            notes = _equation_tests.untested_notes(rows, located, self._outline_equation_names(state))
            record["not_tested"] = notes
            if attempt == 0:
                for n in notes:
                    self._log.info("[equations] %s", n)
            if not case_list:
                _equation_tests.write(self.quest_root / "code", [])
                failures = []
                break
            _equation_tests.write(self.quest_root / "code", case_list)
            # The functions are called in a process of their own, from a copy of code/ outside the quest, given only their
            # inputs; what they return is compared with the plan's examples here, in FI's process (core/equation_measure.py).
            results, rc, err = await _equation_tests.run_and_judge(self._run_in_env, case_list, timeout_s=300, env=env,
                                                                   outside=True)
            if any(r.get("detail") == _equation_tests.NO_RESULT for r in results):
                # A crash, a hang or an early exit fails the tests (never "going on without them"); what it printed is kept.
                self._log.warning("[equations] the run that calls the model's functions gave no result (%s); the tests "
                                  "count as failed", _function_steps.exception_line(err or "", 160) if (err or "").strip() else f"exit code {rc}")
                record["error"] = _function_steps.traceback_end(err or "", 400)
            by_id = {c["id"]: c for c in case_list}
            failures = [{**r, "formula": by_id.get(str(r.get("id")), {}).get("formula", ""),
                         "module": by_id.get(str(r.get("id")), {}).get("module", "")}
                        for r in results if r.get("status") not in (_equation_tests.OK, _equation_tests.OUTPUTS)]
            # The test could not judge it (one value for outputs returned by name): not tested before the run, said so in the
            # record and the limitations; never a failure that stops the study (FI's tool, not the code, fell short).
            shape = [str(r.get("id")) for r in results if r.get("status") == _equation_tests.OUTPUTS]
            for eid in shape:
                note = _equation_tests.cannot_judge_note(eid)
                if note not in notes:
                    notes.append(note)
                    self._log.info("[equations] %s", note)
            record["not_tested"] = notes
            record["tested"] = [str(r.get("id")) for r in results if str(r.get("id")) not in shape]
            if not failures:
                self._log.info("[equations] the model's %d tested equation(s) agree with the plan's worked examples",
                               len(record["tested"]))
                break
            if attempt >= _function_steps.REPAIRS:
                # Code that follows the formula could not meet the plan's example: the example is the plan's own, so the
                # plan is asked (once per equation) to work it out again from the formula, and the tests run once more,
                # unchanged. Nothing about the code or the number it gave goes into the request.
                if rederived or attempt > _function_steps.REPAIRS:
                    break
                rederived = await self._rederive_failed_examples(state, protocol, failures)
                if not rederived:
                    break
                protocol = self._protocol_block(state)
                rows = _equation_tests.example_rows(protocol)
                record["equations"] = {r["id"]: {"state": r["state"], "why": r["why"]} for r in rows}
                fixed = self._outline_signatures(state)  # the reworked examples' names, which the tests now call by
                continue
            for f in failures:
                eid = str(f.get("id"))
                if spent.get(eid, 0) >= _function_steps.REPAIRS:
                    continue
                if not _function_steps.spend(self.quest_root, _function_steps.MAX_CALLS):
                    self._log.warning("[equations] the most requests the function steps may make in this quest are spent; "
                                      "equation %s is not repaired alone", eid)
                    spent[eid] = _function_steps.REPAIRS
                    continue
                spent[eid] = spent.get(eid, 0) + 1
                await self._repair_equation_function(f, sources, located, fixed)
        record["failed"] = [{"id": f.get("id"), "function": f.get("function"), "status": f.get("status"),
                             "reason": _equation_tests.failure_reason(f)} for f in failures]
        record["repairs"] = spent
        if rederived:
            record["example_rederived"] = rederived
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / "equation_tests.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
        except OSError:
            pass
        if not failures:
            return None
        message = " ".join(_equation_tests.failure_message(f, str(f.get("formula") or "")) for f in failures)
        self._log.warning("[equations] %s still fail their worked example after %d repair(s) each: %s; the whole script "
                          "is repaired next",
                          ", ".join(f"{f.get('id')} ({f.get('function')})" for f in failures), _function_steps.REPAIRS,
                          "; ".join(f"{f.get('id')}: {_equation_tests.failure_reason(f)}" for f in failures))
        return {
            "exec_result": {
                "returncode": 1, "duration_s": 0.0, "timed_out": False, "stdout_tail": "",
                "stderr_tail": ("FI tested the model's equations before the study ran and "
                                + ("one does" if len(failures) == 1 else "some do") + " not compute what the plan says. "
                                + message)[-2000:],
                "packages_note": getattr(self, "_packages_note", ""), "failed_script": _split_run.SIMULATE_NAME,
                "flat_output": "", "numeric_warnings": [],
            },
            "figures": [], "figure_records": {}, "result_json": {}, "exec_patch_pending": False,
            # Last-value channels the full run writes on every pass: a failed pass must not leave an earlier script's.
            "result_json_replicates": [], "result_json_deterministic": False, "result_json_trials": False,
            "result_json_replicate_seed_ignored": False, "result_json_no_random_source": False,
        }

    def _examples_rederived_path(self) -> Path:
        return self.fi_dir / "examples_rederived.json"

    def _examples_rederived(self) -> dict[str, Any]:
        """What FI asked the plan about worked examples that code could not meet (``.fi/examples_rederived.json``):
        ``asked`` (``"<equation>:<why>"``, each asked once) and ``examples`` (the example the plan gave back for an
        equation, with the formula it was worked out from). ``{}`` when nothing was asked."""
        try:
            record = json.loads(self._examples_rederived_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return record if isinstance(record, dict) else {}

    async def _rederive_failed_examples(self, state: QuestState, protocol: dict[str, Any] | None,
                                        failures: list[dict[str, Any]]) -> list[str]:
        """An equation test that still fails after its repairs is not a crashed experiment (nothing ran): the example is
        the plan's own, so the plan is asked, once per equation, to work it out again from the equation. The ids whose
        example changed (:meth:`_rederive_examples`)."""
        by_id = {str(eq["id"]).strip(): eq for eq in _equation_tests.eligible(protocol)}
        # An example the equation as written gives too (FI computed both) is right: the code is wrong, and the plan has
        # nothing to work out again.
        agrees = {r["id"] for r in _equation_tests.example_rows(protocol) if r.get("equation_agrees") is True}
        for f in failures:
            if str(f.get("id")) in agrees:
                self._log.info("[equations] the worked example of %s is what its equation as written gives for the same "
                               "inputs, so the code, not the example, is wrong; the plan is not asked about it", f.get("id"))
        # The function's parameters are the first example's inputs (FI set them when it read the outline) and the code is
        # written against them: the plan is told them, so a reworked example keeps the same input names.
        try:
            outline_specs, _notes = _function_steps.model_functions(state.get("implement_outline") or {}, protocol)
        except Exception:  # noqa: BLE001 -- without a readable outline there are no fixed parameters to tell
            outline_specs = []
        fixed_names = {s.equation: s.params() for s in outline_specs if s.kind == _function_steps.EQUATION}
        asks = [{"id": str(f.get("id")), "formula": " ".join(str(f.get("formula") or "").split()),
                 "example": _equation_tests.example_as_written(protocol, str(f.get("id"))), "why": f.get("status"),
                 "detail": f.get("detail"), "source": "tests", "names": fixed_names.get(str(f.get("id")))}
                for f in failures if str(f.get("id")) in by_id and str(f.get("id")) not in agrees
                and _equation_tests.repairable_by_example(f)]
        return await self._rederive_examples(state, protocol, asks)

    async def _rederive_examples(self, state: QuestState, protocol: dict[str, Any] | None,
                                 asks: list[dict[str, Any]]) -> list[str]:
        """Ask the plan, once per equation and per reason (``tests``: code could not meet the example), to work the worked
        example of each equation in ``asks`` out again from the equation as written: ``id``, ``formula``, ``example`` (as
        written), ``why``.

        The request holds the equation, the example as written and the kind of failure, never the number the code gave and
        never the code. Only the asked equations' ``example`` is kept from the rewrite (:meth:`_revise_checks_only` keeps
        only examples; another equation's example is put back as it was). Before the protocol is frozen only: a frozen
        protocol is never changed here. The new example is kept in ``.fi/examples_rederived.json`` and used by
        :meth:`_draft_protocol` (a redesign does not bring the old one back), and plan.md says what was done. The test
        itself is unchanged: FI computes the new example's value as before. The ids whose example changed."""
        path = _plan.plan_path(self.quest_root)
        if (not asks or self.config.engine.oracle_check == "off" or not self._runs_code(state)
                or _frozen.load(self.quest_root) is not None or not path.is_file()):
            return []
        record = self._examples_rederived()
        asked = record.get("asked") if isinstance(record.get("asked"), dict) else {}
        def ask_key(a: dict[str, Any]) -> str:  # a redesign that gives the id a new formula may be asked again
            return f"{a['id']}:{a['source']}:{a['formula']}"
        asks = [a for a in asks if ask_key(a) not in asked]
        if not asks:
            return []
        # Recorded before the request: a killed run does not ask again.
        asked = {**asked, **{ask_key(a): True for a in asks}}
        record = {**record, "asked": asked}
        self._write_json_quiet(self._examples_rederived_path(), record)
        ids = [a["id"] for a in asks]
        for a in asks:
            self._log.info("[plan] the plan's worked example for %s %s; FI asks the plan to work the example out again from "
                           "the formula", a["id"], _REDERIVE_BECAUSE[a["source"]])
        design_before, _why = _plan.load_design(self.quest_root)
        protocol_before = design_before.get("protocol") if isinstance(design_before, dict) else None
        examples_before = {str(e["id"]).strip(): e.get("example") for e in _equation_tests.eligible(
            protocol_before if isinstance(protocol_before, dict) else None)}
        failed = await self._revise_checks_only(_equation_tests.rederive_request(asks), equations="example")
        if failed:
            self._log.warning("[plan] %s", failed)
            return []
        # Only a usable new example of an asked equation is kept: every other example, an asked one the plan left
        # unusable or `untestable` included, is put back as it was, so a reply can never take an equation out of the test.
        design_after, _why = _plan.load_design(self.quest_root)
        protocol_after = design_after.get("protocol") if isinstance(design_after, dict) else None
        formulas = {str(a["id"]): a["formula"] for a in asks}
        protocol_after = protocol_after if isinstance(protocol_after, dict) else None
        usable = {r["id"] for r in _equation_tests.example_rows(protocol_after) if r["state"] == "ok"}
        taken: dict[str, dict[str, Any]] = {}
        for eq in _equation_tests.eligible(protocol_after):
            eid = str(eq["id"]).strip()
            if (eid in ids and eid in usable and eq.get("example") != examples_before.get(eid)
                    and " ".join(str(eq.get("formula") or "").split()) == formulas[eid]):
                taken[eid] = {"formula": formulas[eid], "example": eq["example"]}
        after_text = path.read_text(encoding="utf-8")

        def put_back(block: dict[str, Any]) -> dict[str, Any] | None:
            proto = dict(block.get("protocol") or {})
            model = proto.get("model")
            if not isinstance(model, dict):
                return None
            key = next((k for k in _oracle._EQUATION_KEYS if _oracle.equation_items(model.get(k))), None)
            if key is None:
                return None
            items = [dict(e) for e in _oracle.equation_items(model[key])]
            moved = False
            for eq in items:
                eid = str(eq.get("id") or "").strip()
                if eid not in taken and eid in examples_before and eq.get("example") != examples_before[eid]:
                    if examples_before[eid] is None:
                        eq.pop("example", None)
                    else:
                        eq["example"] = examples_before[eid]
                    moved = True
            return {**block, "protocol": {**proto, "model": {**model, key: items}}} if moved else None

        restored = _plan.edit_design_block(after_text, put_back)
        if restored is not None and _plan.parse(restored).design is not None:
            restored = _plan.refresh_model_section(restored)
            path.write_text(restored, encoding="utf-8")
            _plan.record_version(self.quest_root, restored, by="engine",
                                 note="the examples FI did not take from the plan's reply put back as they were")
        lines = []
        for eid in ids:
            lines.append(f"- FI asked the plan to work the example of equation {eid} out again from its formula, because it "
                         f"{_REDERIVE_BECAUSE[next(a['source'] for a in asks if a['id'] == eid)]}, and "
                         + ("the plan gave a new one." if eid in taken
                            else "the plan gave no new one FI can use, so the example is kept as it was."))
        self._write_plan_section(path, ["", *lines], note="what FI asked the plan about a worked example it could not meet")
        for eid in ids:
            if eid in taken:
                self._log.info("[plan] the plan worked the example of %s out again; the equation is tested once more with it", eid)
            else:
                self._log.warning("[plan] the plan gave no new example of %s that FI can use; the example is kept as it was "
                                  "and the equation's test goes on as before", eid)
        if taken:
            self._write_json_quiet(self._examples_rederived_path(),
                                   {**self._examples_rederived(), "examples": {**(record.get("examples") or {}), **taken}})
        return sorted(taken)

    def _write_json_quiet(self, path: Path, value: dict[str, Any]) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[plan] could not write %s: %r", path.name, e)

    def _outline_equation_specs(self, state: QuestState) -> list[Any]:
        """The outline's function for each equation of the plan's model (:func:`core.function_steps.model_functions`);
        ``[]`` when the code was not written from an outline that names a function per equation."""
        protocol = self._protocol_block(state)
        try:
            specs, _notes = _function_steps.model_functions(state.get("implement_outline") or {},
                                                            protocol if isinstance(protocol, dict) else None)
        except Exception:  # noqa: BLE001 -- without a readable outline no function is named for an equation
            return []
        return [s for s in specs if s.kind == _function_steps.EQUATION]

    def _outline_equation_names(self, state: QuestState) -> dict[str, str]:
        """Each equation's id to the function the outline names for it: the one the equation tests call
        (:func:`core.equation_tests.locate`), so a ``# E<n>`` label on a helper never makes the helper the tested one."""
        return {s.equation: s.name for s in self._outline_equation_specs(state)}

    def _outline_signatures(self, state: QuestState) -> dict[str, str]:
        """The parameters a repair must keep for each function that implements an equation of the plan's model: the
        function's name to its ``def`` line (:func:`core.function_steps.held_signatures`). For a function an equation test
        calls and the code cannot yet be called by, the outline's: the tests call it by them, so a repair keeps them,
        never the parameters some earlier repair left. For one the test already calls, or no test calls, the parameters
        it has now in ``code/``: the code its callers were written against is what a repair must not break (a
        first-written function can differ from the outline, in its names or only in their order, and holding it to the
        outline refused every repair of the file). ``{}`` when the code was not written from an outline that names a
        function per equation."""
        specs = self._outline_equation_specs(state)
        if not specs:
            return {}
        fixed = {s.name: s.signature for s in specs}
        try:
            protocol = self._protocol_block(state)
            protocol = protocol if isinstance(protocol, dict) else None
            sources = self._equation_sources()
            located = _equation_tests.locate(protocol, sources, {s.equation: s.name for s in specs})
            cases = _equation_tests.cases(_equation_tests.example_rows(protocol), located)
        except Exception:  # noqa: BLE001 -- unreadable code: every function held to the outline, as before
            return fixed
        return _function_steps.held_signatures(fixed, sources, cases)

    def _match_example_names(self, case_list: list[dict[str, Any]], located: dict[str, dict[str, str] | None],
                             sources: dict[str, str]) -> bool:
        """Before the equation tests run: a function whose parameter is named otherwise than the plan's worked example names
        that input (``rate`` for ``Rate``), when only capitals or underscores differ (:func:`core.function_steps.match_example_names`),
        has the parameter renamed by FI, in the function and where it reads it. The code is often written whole with
        the parameters it prefers, and a repair of one function is held to the example's names, which the model that wrote the
        function has twice not used; FI already knows both names, so it sets them. The value is still judged by the test.
        ``True`` when a file was changed."""
        changed = False
        for case in case_list:
            where = located.get(str(case.get("id")))
            if not where or not case.get("inputs"):
                continue
            try:
                path = self.quest_root / "code" / where["file"]
                text = path.read_text(encoding="utf-8")
                new, renames = _function_steps.match_example_names(
                    text, where["function"], list(case["inputs"]), random=bool(case.get("random")),
                    callers=[f.read_text(encoding="utf-8", errors="replace") for f in (self.quest_root / "code").glob("*.py")]
                    + [s for rel, s in sources.items() if rel != where["file"]])
                if new is None:
                    continue
                path.write_text(new, encoding="utf-8")
            except Exception as e:  # noqa: BLE001 -- a rename FI could not make leaves the code as it was
                self._log.warning("[equations] could not set the parameter names of %s: %r", where.get("function"), e)
                continue
            changed = True
            self._log.info("[equations] FI set the parameters of %s to the names of the plan's worked example for %s (%s)",
                           where["function"], case.get("id"), ", ".join(f"{o} -> {n}" for o, n in renames.items()))
        return changed

    async def _repair_equation_function(self, failure: dict[str, Any], sources: dict[str, str],
                                        located: dict[str, dict[str, str] | None],
                                        fixed: dict[str, str] | None = None) -> bool:
        """ONE repair of the function a failed equation test names, alone: it is shown that function, its equation and the
        problem (never the expected value or the study's script). Kept only when the file still parses and the function
        has the parameters the outline fixed (``fixed``, :meth:`_outline_signatures`), or, for a function the outline
        does not name, the parameters it has now."""
        where = located.get(str(failure.get("id")))
        if not where:
            return False
        path = self.quest_root / "code" / where["file"]
        try:
            source = path.read_text(encoding="utf-8")
            node = _function_steps._find(source, where["function"])
            if node is None:
                return False
            # The contract the test calls it by: the outline's parameters, else the parameters exactly as it has them now.
            header = (fixed or {}).get(where["function"]) or f"def {where['function']}({ast.unparse(node.args)}):"
            problem = _equation_tests.failure_message(failure, str(failure.get("formula") or ""))
            equation = str(failure.get("id"))
            block = f"{equation}: {failure.get('formula') or ''}"
            new, why = await _function_steps.repair_one(
                source=source, name=where["function"], signature=header, equation=equation, equation_block=block,
                problem=problem, prompt=self._prompts["implement_function_repair"],
                chat=lambda prompt: self._chat(prompt, node="implement"), package=where["file"].split("/")[0],
                topic_block=self._topic_block())
        except _ModelAnswerProblem:
            raise
        except Exception as e:  # noqa: BLE001 -- a repair that could not be asked for leaves the code as it was
            self._log.warning("[equations] the repair of %s could not be asked for: %r", failure.get("function"), e)
            return False
        if new is None:
            self._log.warning("[equations] the repair of %s was not used: %s", failure.get("function"), why)
            return False
        new = _skeleton.without_raise(new)
        framed = _skeleton.repair_problem(source, new, where["file"])  # a framed simulate.py keeps FI's frame
        if framed:
            self._log.warning("[equations] the repair of %s is not used: %s", failure.get("function"), framed)
            return False
        path.write_text(new, encoding="utf-8")
        self._log.info("[equations] rewrote the function %s alone (equation %s)", failure.get("function"), failure.get("id"))
        return True

    def _package_in_use(self, state: QuestState) -> str | None:
        """The model's package when this quest's code is laid out as a research tool and the package is there."""
        layout = self._code_layout(state)
        if not layout or layout["shape"] != _code_layout.PACKAGE:
            return None
        package = layout["package"]
        return package if (self.quest_root / "code" / package / "__init__.py").is_file() else None

    def _package_shown(self, state: QuestState) -> tuple[str, dict[str, str]] | None:
        """``(package, {"<package>/model.py": text})`` for a quest whose simulation computes with the model's package:
        what a prompt that rewrites simulate.py shows beside it, so the model sees the equations it calls."""
        try:
            package = self._package_in_use(state)
            if not package:
                return None
            sources = {rel: text for rel, text in _code_layout.package_sources(self.quest_root / "code").items()
                       if rel.startswith(f"{package}/")}
            return (package, sources) if sources else None
        except Exception as e:  # noqa: BLE001 -- showing the package is a help to the prompt, never a stop
            self._log.warning("[implement] could not read the model's package: %r", e)
            return None

    def _package_repair_note(self, state: QuestState) -> str:
        shown = self._package_shown(state)
        if not shown:
            return ""
        fixed = self._outline_signatures(state)
        return _code_layout.repair_note(*shown) + (
            "Keep the parameters of these functions of the package exactly as they are (FI's tests call them by these "
            "names before the study runs; a package file that changes them is not used): "
            + "; ".join(f"`{sig.rstrip(':')}`" for sig in fixed.values()) + "\n" if fixed else "")

    def _package_files_kept(self, state: QuestState, package: str, parsed: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
        """The package files a repair gave back that can be used, and the ones left out with why: a file that does not
        parse, or that changes the parameters the outline fixed for an equation's function (:meth:`_outline_signatures`)."""
        files, dropped = _code_layout.repair_files(parsed.get("package_files"), package)
        fixed = self._outline_signatures(state)
        for rel in list(files):
            changed = _function_steps.changed_signatures(files[rel], fixed) if fixed else []
            if changed:
                dropped.append(f"{rel} ({'; '.join(changed)})")
                del files[rel]
        return files, dropped

    def _package_refused(self, state: QuestState, parsed: Any) -> list[str]:
        """The package files a repair gave back that are not used because they change a parameter list the outline fixed
        (:meth:`_package_files_kept`), each with why; ``[]`` when there is none."""
        shown = self._package_shown(state)
        if not shown or not isinstance(parsed, dict):
            return []
        dropped = self._package_files_kept(state, shown[0], parsed)[1]
        return [d for d in dropped if "the outline fixed" in d]

    def _apply_package_repair(self, state: QuestState, parsed: Any, *, node: str) -> list[str]:
        """Write the package files a repair gave back (its JSON ``package_files``); returns the files written."""
        shown = self._package_shown(state)
        if not shown or not isinstance(parsed, dict):
            return []
        files, dropped = self._package_files_kept(state, shown[0], parsed)
        if dropped:
            self._log.warning("[%s] the repair's package files not used: %s", node, "; ".join(dropped))
        if not files:
            return []
        try:
            wrote = _code_layout.write_package(self.quest_root / "code", files, same=_split_run.same_script)
        except OSError as e:
            self._log.warning("[%s] could not write the repair's package files: %r", node, e)
            return []
        if wrote:
            self._log.info("[%s] the repair rewrote the model's package: %s", node, ", ".join(f"code/{w}" for w in wrote))
        return wrote

    def _package_repair_files(self, state: QuestState, parsed: Any) -> dict[str, str]:
        """The usable package files a repair reply holds that differ from the package on disk (nothing is written)."""
        shown = self._package_shown(state)
        if not shown or not isinstance(parsed, dict):
            return {}
        files = self._package_files_kept(state, shown[0], parsed)[0]
        return {rel: text for rel, text in files.items()
                if rel not in shown[1] or not _split_run.same_script(shown[1][rel], text)}

    def _package_snapshot(self) -> dict[str, str]:
        return _code_layout.package_sources(self.quest_root / "code")

    def _restore_package(self, before: dict[str, str]) -> None:
        """Put the model's package back as ``before`` held it (a repair that is undone takes its package change with it)."""
        code = self.quest_root / "code"
        try:
            for rel, text in _code_layout.package_sources(code).items():
                if rel not in before:
                    (code / rel).unlink(missing_ok=True)
                elif text != before[rel]:
                    (code / rel).write_text(before[rel], encoding="utf-8")
            for rel, text in before.items():
                if not (code / rel).is_file():
                    (code / rel).parent.mkdir(parents=True, exist_ok=True)
                    (code / rel).write_text(text, encoding="utf-8")
        except OSError as e:
            self._log.warning("[oracle] could not put the model's package back as it was: %r", e)

    def _record_cannot_repair(self, text: str, tried: list[str], *, source: str, check: str) -> None:
        """Write down, before the study runs, that FI's repairs of the code ended with a problem that stands, what the
        problem rests on (``source``, :mod:`core.stop_power`) and which check found it (``check``). One FI measured, or that
        two sources agree on, ends the run step in the honest stop (no paper) without running the script. One that rests
        on FI's reading of the code is kept for QUALITY.md and the paper's limitations, and the quest goes on. A reading
        never replaces a measured problem of the same text. Nobody is asked to change code."""
        problems = self._cannot_repair_problems()
        same = next((p for p in problems if p.get("text") == text), None)
        if same is not None and _stop_power.may_stop(same.get("source")) and not _stop_power.may_stop(source):
            return
        problems = [p for p in problems if p is not same]
        problems.append({"text": text, "tried": tried, "source": source, "check": check,
                         "code_version": _code_review.version(self.quest_root / "code")})
        self._write_cannot_repair(problems)

    def _clear_cannot_repair(self, check: str) -> None:
        """The check ``check`` ran again and found nothing (the code was repaired since): its problems are taken out of the
        record, so neither the paper nor QUALITY.md reports a problem that is fixed."""
        problems = self._cannot_repair_problems()
        kept = [p for p in problems if p.get("check") != check]
        if len(kept) != len(problems):
            self._write_cannot_repair(kept)

    def _write_cannot_repair(self, problems: list[dict[str, Any]]) -> None:
        path = self.fi_dir / self._CANNOT_REPAIR_FILE
        if not problems:
            self._forget_cannot_repair()
            return
        stopping = next((p for p in problems if _stop_power.may_stop(p.get("source"))), None)
        record: dict[str, Any] = {"problems": problems, "at": _frozen.now()}
        if stopping is not None:
            record.update(text=stopping["text"], tried=stopping["tried"])
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[stuck] couldn't write down the problem FI could not repair: %r", e)

    def _cannot_repair_problems(self) -> list[dict[str, Any]]:
        """Every problem :meth:`_record_cannot_repair` wrote down for this text of the code, oldest first."""
        try:
            record = json.loads((self.fi_dir / self._CANNOT_REPAIR_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(record, dict):
            return []
        if "problems" not in record and record.get("text"):  # written before problems said what they rest on
            return [{"text": str(record["text"]), "tried": list(record.get("tried") or []), "source": _stop_power.MEASURED}]
        return [p for p in record.get("problems") or [] if isinstance(p, dict) and p.get("text")]

    def _forget_cannot_repair(self) -> None:
        """The code is written again: an earlier problem does not stop the new code."""
        try:
            (self.fi_dir / self._CANNOT_REPAIR_FILE).unlink(missing_ok=True)
        except OSError:
            pass

    def _cannot_repair_stop(self) -> QuestState | None:
        """The honest stop for a problem the code step recorded (:meth:`_record_cannot_repair`) that may end the quest, or
        ``None``. A problem that rests on FI's reading of the code is said once in run.log and the quest goes on."""
        problems = self._cannot_repair_problems()
        said = self.__dict__.setdefault("_went_on_said", set())
        for p in problems:
            if not _stop_power.may_stop(p.get("source")) and p["text"] not in said:
                said.add(p["text"])
                self._log.warning("[execute] %s; this is %s, so the run goes on and the paper says so",
                                  p["text"], _stop_power.source_words(p.get("source")))
        stopping = next((p for p in problems if _stop_power.may_stop(p.get("source"))), None)
        if stopping is None:
            return None
        return self._stop_before_run(str(stopping["text"]), "cannot_repair", tried=[str(t) for t in stopping.get("tried") or []])

    def _code_problems_stop(self, state: QuestState) -> QuestState | None:
        """Under ``rigor_profile: research``, before anything runs: a simulation that does not say where it implements each
        equation of the plan, or a ``code/`` folder missing its package, tests or equation list, goes to the existing repair
        like a failed run, with a plain directive naming what is missing (at most ``engine.exec_reflect_max_iterations``
        repairs); what is still missing after them is FI's reading of the code, so it is recorded, the paper says it and
        the quest goes on (:mod:`core.stop_power`). Nobody is asked to edit code. ``None`` to go on."""
        labels = self._check_equation_labels(state)
        layout = self._check_code_layout(state)
        if self.config.rigor_profile != "research" or not (labels or layout):
            self._clear_cannot_repair("layout")
            return None
        problems = [*labels, *layout]
        iters = int(state.get("exec_reflect_iter") or 0)
        if iters < self.config.engine.exec_reflect_max_iterations:
            self._log.warning("[implement] the code is not laid out as this quest needs (%s); sending it back to be repaired",
                              "; ".join(problems))
            main, missing = self._unlabelled(state, self._protocol_block(state)) if labels else (None, [])
            where = (f"code/{self._label_target(state, main).relative_to(self.quest_root / 'code').as_posix()}"
                     if main is not None else "code/")
            directive = (
                "FI read the code before running it, and it is not laid out as this quest needs:\n"
                + "\n".join(f"- {p}" for p in problems[:8])
                + ("\n\nPut a comment with the equation's id (for example `# " + missing[0] + "`) on or above the code in "
                   f"{where} that computes it. Only the label is checked, not the mathematics. " if labels and missing else "\n\n")
                + ("Add what the folder lacks (the model's equations in their own package, each function labelled with its "
                   "equation and used by simulate.py, with unit tests and an equation list). " if layout else "")
                + "Keep everything else as it is."
            )
            return self._code_review_failed_run(
                directive, script=_split_run.SIMULATE_NAME if self._split_on(state) else "experiment.py")
        self._record_cannot_repair(
            "; ".join(problems),
            ["read the code against what the plan requires of it before running it",
             *[_one_line(p, 300) for p in problems[:3]],
             f"sent the code back to be repaired {iters} time(s); it still lacked this"], source=_stop_power.READ,
            check="layout")
        return self._cannot_repair_stop()

    def _check_code_layout(self, state: QuestState) -> list[str]:
        """After the code is written, before anything runs: say in run.log what the research-tool layout of ``code/`` is
        missing (the package, the unit tests, which function computes each equation); under ``rigor_profile: research``
        stop until it is there. On a resume the folder is read again."""
        research = self.config.rigor_profile == "research"
        try:
            layout = self._code_layout(state)
            code = self.quest_root / "code"
            if not layout or layout["shape"] != _code_layout.PACKAGE or not (code / _split_run.SIMULATE_NAME).is_file():
                return []
            package = layout["package"]
            fell_back = (_code_layout.load(self.quest_root) or {}).get("fell_back") and not (code / package).is_dir()
            if fell_back and not research:
                return []  # said when the code was written: this quest keeps two scripts
            problems = _code_layout.check(code, self._protocol_block(state), package, self._outline_equation_names(state))
            # A generator built without a seed in the package ignores the seed FI gives each trial.
            for rel, text in sorted(_code_layout.package_sources(code).items()):
                if rel.startswith(f"{package}/"):
                    fallback = _code_layout.fallback_rng_lines(text)  # `rng = rng or default_rng()`: asked for
                    unseeded = [(line, expr) for line, expr in unseeded_rng_calls(text) if line not in fallback]
                    if unseeded:
                        problems.append(f"code/{rel} builds a random generator without a seed ("
                                        + "; ".join(f"line {line}: {expr}" for line, expr in unseeded[:3])
                                        + "), so the seed FI gives each trial does not reach it")
        except Exception as e:  # noqa: BLE001 -- a check of the folder must never stop a quest by crashing
            self._log.warning("[implement] could not check the layout of code/: %r", e)
            return []
        for why in problems:
            self._log.warning("[implement] code/ as a small research tool: %s", why)
        return problems

    def _check_equation_labels(self, state: QuestState) -> list[str]:
        """Say, in run.log, which equations of the plan's model the simulation does not mark where it implements them;
        under ``rigor_profile: research`` stop until it does. Checked before every run and again after the oracle
        repairs (a repair rewrites the script); on a resume the script on disk is read again, so a label a person added
        is seen."""
        try:
            main, missing = self._unlabelled(state, self._protocol_block(state))
            gaps = self._label_sentences(self._label_target(state, main), missing)  # named where the label belongs
        except Exception as e:  # noqa: BLE001 -- a check of the script must never stop a quest by crashing
            self._log.warning("[implement] could not check where the simulation implements the plan's equations: %r", e)
            return []
        for why in gaps:
            self._log.warning("[implement] %s", why)
        return gaps

    def _write_evidence(self, state: QuestState, *, sealing: bool = False, write: bool = True) -> dict[str, Any] | None:
        """Work out how much of the result has been checked against something other than itself
        (:mod:`core.evidence`) and keep it in ``needs/EVIDENCE.json``. Best-effort: it never touches the quest.
        ``sealing``: the quest is finishing and seals its trace right after (the seal names this record). ``write``
        False: only work it out (what the review pause shows before an accept), writing and recording nothing."""
        try:
            missed: list[str] = []
            on = _precision_target_on(self._protocol_block(state) or {})  # the metric the plan's target is on, when declared
            replicates = state.get("result_json_replicates") or []
            if len(replicates) > 1 or (replicates and state.get("result_json_trials")):
                aggregate = _aggregate_result_json_replicates(
                    replicates, assertions=_replicate_assertions(state), kinds=_replicate_metric_kinds(state),
                )
                self._annotate_precision(state, aggregate)
                missed = [k for k, v in aggregate.items() if isinstance(v, dict) and v.get("precision_reached") is False]
                # Judged means the quantity the target is on was held to it: a probability judged beside a mean the target
                # names, given without its values, left the mean's target unjudged and unsaid.
                judged = any(isinstance(v, dict) and "target_half_width" in v
                             and (not on or _is_metric_key(k, on, _declared_metrics(self._protocol_block(state) or {})))
                             for k, v in aggregate.items())
            else:
                judged = False
            # The plan's precision target that no result was held to (each setting ran once, so no interval; or the quantity
            # it is on has no interval): a gap, kept in the record, never read as a target reached.
            precision = (self._protocol_block(state) or {}).get("precision")
            target = precision.get("target_half_width") if isinstance(precision, dict) else None
            not_judged = ({"target": float(target), "metric": on or str(precision.get("metric") or "").strip()}
                          if not judged and (state.get("result_json") or replicates)
                          and isinstance(target, (int, float)) and not isinstance(target, bool)
                          and 0 < target < 1e300 else None)
            # FI's own trial record has the trials of each setting: the target is held to each setting's own interval
            # (never one pooled over settings whose true values differ). Said in the record, never a route back to design.
            per_setting = (_trial_table.per_setting_precision(self.quest_root, on, float(target),
                                                              _metric_spec.proportion_ids(self._protocol_block(state)))
                           if not_judged and on else None)
            if per_setting:
                not_judged = None
            replicates_for_specs = state.get("result_json_replicates") or []
            protocol_now = self._protocol_block(state)
            try:
                computed = (
                    _metric_spec.statistics(replicates_for_specs, protocol_now, progress=self._log.info, quest_root=self.quest_root)
                    if replicates_for_specs and _metric_spec.declared(protocol_now) else None
                )
                statistics_gaps = _metric_spec.coverage_gaps(protocol_now, replicates_for_specs, computed) if replicates_for_specs else []
            except Exception as e:  # noqa: BLE001 -- a report about the quest must never touch it
                self._log.warning("[evidence] the statistics coverage could not be worked out: %r", e)
                # Never "no gap": a check that crashed has not shown the statistics are adequate.
                statistics_gaps = [f"the statistics could not be worked out ({type(e).__name__}: {str(e)[:160]})"]
            # Where each check's expected value comes from, judged against the sources as they were numbered when the
            # protocol was frozen (the paper later cites a subset under new numbers).
            try:
                oracle_source_gaps = self._oracle_source_gaps(state, protocol_now)
            except Exception as e:  # noqa: BLE001 -- a report about the quest must never touch it
                self._log.warning("[evidence] where the checks' expected values come from could not be checked: %r", e)
                oracle_source_gaps = [f"where the checks' expected values come from could not be checked ({type(e).__name__})"]
            try:
                label_gaps = self._equation_label_gaps(state, protocol_now)
            except Exception as e:  # noqa: BLE001 -- a report about the quest must never touch it
                self._log.warning("[evidence] where the simulation implements the plan's equations could not be checked: %r", e)
                label_gaps = [f"where the simulation implements the plan's equations could not be checked ({type(e).__name__})"]
            try:
                independence_gaps = self._independence_gaps(_frozen.protocol_of(self.quest_root) or protocol_now)
            except Exception as e:  # noqa: BLE001 -- a report about the quest must never touch it
                self._log.warning("[evidence] whether the checks are independent of their writer could not be checked: %r", e)
                independence_gaps = [f"whether the checks are independent of the model that wrote them could not be "
                                     f"checked ({type(e).__name__})"]
            independence_gaps = [*independence_gaps, *self._static_check_gaps(),  # core/static_checks.py
                                 *_refine.evidence_gaps(self.quest_root, self._protocol_block(state)),
                                 *_second_impl.dependence_gaps(self.quest_root)]
            record = _evidence.assess(
                self.quest_root, dict(state), precision_missed=missed, statistics_gaps=statistics_gaps,
                oracle_source_gaps=oracle_source_gaps, equation_label_gaps=label_gaps,
                independence_gaps=independence_gaps, precision_not_judged=not_judged,
                precision_per_setting=per_setting,
                settings={
                    "protocol_check": self.config.engine.protocol_check,
                    "oracle_check": self.config.engine.oracle_check,
                    "numeric_warnings": self.config.engine.numeric_warnings,
                    "run_manifest_check": self.config.engine.run_manifest_check,
                    "rigor_profile": self.config.rigor_profile,
                    "result_use": getattr(self.config, "effective_result_use", "") or getattr(self.config, "result_use", ""),
                    # The panel really runs on one model and the person said that is all they have: a gap. The flag
                    # alone (set, then a second model added) is not.
                    "one_model_review": self._one_model_panel(),
                    "review_identity_unverified": self._review_identity_unverified(state),
                    "sealing": sealing,
                    "evidence_gate": "on" if self.config.engine.evidence_gate else "off",
                    "claim_check": "on" if self.config.engine.claim_grounding else "off",
                    "citation_check": "on" if self.config.engine.claim_grounding else "off",
                    # --analyze has no experiment to design, so there is no design to audit.
                    "design_audit": "not_applicable" if self.config.engine.analyze_local_first else "on",
                    **(_phased.evidence_settings(self.quest_root) if _phased.enabled(self.config) else {}),
                    # A coding agent wrote the code (engine.code_writer: agent): whether it ran in a container.
                    **_agent_coder.evidence_settings(self.quest_root),
                },
            )
            if not write:
                return record
            path = self.quest_root / "needs" / "EVIDENCE.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            self._log.info("[evidence] %s", _evidence.summary_line(record, technical=True))
            self._progress(f"[evidence] {_evidence.summary_line(record)}")
            self._audit_check("evidence", path, status=str(record.get("status")), summary=_evidence.summary_line(record))
            return record
        except Exception as e:  # noqa: BLE001 -- a report about the quest must never stall it
            self._log.warning("[evidence] could not assess the quest: %r", e)
            if not write:
                return None
            # Replace any earlier record: a stale "publication_ready" from a previous pass must not outlive a failed
            # assessment of this one.
            try:
                record = _evidence.unassessed(f"{type(e).__name__}: {str(e)[:200]}")
                path = self.quest_root / "needs" / "EVIDENCE.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
            return None

    def _annotate_precision(self, state: QuestState, aggregate: dict[str, Any]) -> str:
        """For each quantity with an interval pooled from the trials (a probability from its counts, a mean from the values
        the run lists), its half-width, and whether the protocol's target was reached. Both are held to the target the same
        way. Returns a sentence for the analysis when a target was missed, else an empty string.

        Which quantities the target is for: the probabilities, or the one the protocol names (``precision.metric``); a
        mean only when the target is on it (``run_estimate.precision_metric`` gives a ``mean``: the metric named, or the
        plan's only metric), since a mean's scale is its own."""
        from . import run_estimate as _run_estimate

        protocol = self._protocol_block(state) or {}
        precision = protocol.get("precision")
        target = precision.get("target_half_width") if isinstance(precision, dict) else None
        on, kind = _run_estimate.precision_metric(protocol)
        mean_on = str(on) if on and kind == "mean" else ""

        declared = _declared_metrics(protocol)

        def is_(key: str, name: str) -> bool:
            return _is_metric_key(key, name, declared)

        # The target is for the metric the protocol names (``precision.metric``), when it names one: a live quest's
        # target for its outbreak probability was held against every pooled interval in the run.
        named = str(precision.get("metric") or "").strip() if isinstance(precision, dict) else ""
        # A metric the plan declares keeps the target even with no pooled interval (a mean given without its values), and so
        # does the plan's only metric when none is named: no other quantity is held to a target set on another scale, and
        # the target is left unjudged, a gap.
        on_declared = _precision_target_on(protocol)
        if on_declared:
            named = on_declared
        elif named and not any(
            isinstance(e, dict) and e.get("ci_method") in (_CI_WILSON, _CI_BOOTSTRAP) and is_(k, named)
            for k, e in aggregate.items()
        ):
            named = ""  # a name that matches no pooled quantity (prose, say): every probability is held to the target
        missed: list[str] = []
        for key, entry in aggregate.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("ci_method") == _CI_BOOTSTRAP:
                if not (mean_on and is_(key, mean_on)):
                    continue
            elif entry.get("ci_method") != _CI_WILSON or (named and not is_(key, named)):
                continue
            lo, hi = entry.get("ci_lower"), entry.get("ci_upper")
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                continue
            entry["ci_half_width"] = (hi - lo) / 2.0
            if isinstance(target, (int, float)) and not isinstance(target, bool):
                entry["target_half_width"] = float(target)
                entry["precision_reached"] = entry["ci_half_width"] <= float(target)
                if not entry["precision_reached"]:
                    missed.append(f"{key} (±{entry['ci_half_width']:.3f})")
        if not missed:
            # No pooled interval for the quantity the target is on: FI's trial record gives each setting's own.
            judged = any(isinstance(e, dict) and "target_half_width" in e for e in aggregate.values())
            if on_declared and not judged and isinstance(target, (int, float)) and not isinstance(target, bool) and 0 < target < 1e300:
                per_setting = _trial_table.per_setting_precision(self.quest_root, on_declared, float(target),
                                                                 _metric_spec.proportion_ids(protocol))
                words = _evidence.precision_per_setting_words(per_setting) if per_setting else ""
                if words:
                    return (f"{words[0].upper()}{words[1:]}. Say so in the limitations and do not claim a precision the "
                            f"trials do not give.")
            return ""
        return (
            f"The protocol's target 95% half-width of ±{float(target):g} was not reached for {len(missed)} of the "
            f"quantities it is on: {', '.join(missed[:6])}. Say so in the limitations and do not claim a precision the "
            f"trials do not give."
        )

    def _scan_numeric_warnings(self, state: QuestState, stderr: str, result_json: Any) -> list[dict[str, str]]:
        """The numeric warnings a run reported, as plain records for the state; logged and written to
        ``needs/NUMERIC_WARNINGS.json`` whatever ``engine.numeric_warnings`` says (unless it is ``off``)."""
        if (
            self.config.engine.numeric_warnings == "off"
            or state.get("no_simulation_resolved")
            or state.get("survey_mode_resolved")
            or self.config.engine.analyze_local_first
        ):
            return []
        try:
            # The protocol the run is held to: the frozen one, as the evidence reads it (core/evidence.py).
            frozen = _frozen.load(self.quest_root)
            protocol = frozen.get("protocol") if frozen is not None else (state.get("design") or {}).get("protocol")
            results = _numeric.declared_results(protocol)
            found = _numeric.scan(stderr, result_json, results=results)
            try:
                own = _numeric.own_measurements(protocol, self._oracle_record_read())
            except Exception:  # noqa: BLE001 -- FI's value beside a disclosed number is a convenience
                own = {}
            disclosed = _numeric.disclosed_non_finite(result_json, results, own)
        except Exception as e:  # noqa: BLE001 -- a defect in the scan must never stall a quest
            self._log.warning("[numeric] the warning scan failed (%r); skipping", e)
            # A scanner that crashed is not the same as one that ran and found nothing: the first means the run's
            # numerics were never actually checked. Recorded so the evidence level can tell the two apart (a crash that
            # went unrecorded here previously read exactly like a clean scan — the audit's P1-1).
            try:
                path = self.quest_root / "needs" / "NUMERIC_WARNINGS.json"
                history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
                history.append({"error": repr(e), "mode": self.config.engine.numeric_warnings})
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(history[-20:], indent=2) + "\n", encoding="utf-8")
                self._audit_check("numeric_warnings", path, status="error", summary=repr(e))
            except (OSError, ValueError):
                pass
            return []
        if disclosed:
            # Not a number the paper stands on (a diagnostic, a timing, a name the plan does not declare): disclosed in
            # the evidence, not repaired for (a NaN in a number nobody reports must not stop every result).
            self._log.info("[numeric] not finite, not a result the paper stands on (disclosed): %s", "; ".join(disclosed[:8]))
        if not found:
            if disclosed:
                try:
                    path = self.quest_root / "needs" / "NUMERIC_WARNINGS.json"
                    history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
                    history.append({"warnings": [], "disclosed": disclosed[:20], "mode": self.config.engine.numeric_warnings})
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(history[-20:], indent=2) + "\n", encoding="utf-8")
                except (OSError, ValueError):
                    pass
                return []
            # A clean scan writes nothing when there is no history yet (a run with nothing to report leaves no
            # trace file at all). But when the file already ends on a scanner crash, this clean scan is itself
            # news: without it, that stale ``{"error": ...}`` would stay the last entry forever, and
            # ``core/evidence.py`` reads the last entry to decide whether the numerics were ever actually
            # checked — a later clean scan must be able to say so. A clean scan after WARNINGS (not a crash)
            # stays silent, unchanged: the repair-loop tests below need the recorded history of what was found,
            # not a synthetic "and then it was fine" entry for every successful repair.
            path = self.quest_root / "needs" / "NUMERIC_WARNINGS.json"
            if path.is_file():
                try:
                    history = json.loads(path.read_text(encoding="utf-8"))
                    # The same holds for an entry that disclosed a number the run no longer prints as NaN.
                    if history and isinstance(history[-1], dict) and ("error" in history[-1] or history[-1].get("disclosed")):
                        history.append({"warnings": [], "mode": self.config.engine.numeric_warnings})
                        path.write_text(json.dumps(history[-20:], indent=2) + "\n", encoding="utf-8")
                except (OSError, ValueError, IndexError):
                    pass
            return []
        for w in found:
            self._log.warning("[numeric] %s", w.describe())
        try:
            path = self.quest_root / "needs" / "NUMERIC_WARNINGS.json"
            history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
            history.append({"warnings": [{"kind": w.kind, "text": w.text} for w in found],
                            **({"disclosed": disclosed[:20]} if disclosed else {}), "mode": self.config.engine.numeric_warnings})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(history[-20:], indent=2) + "\n", encoding="utf-8")
            self._audit_check(
                "numeric_warnings", path, status=self.config.engine.numeric_warnings,
                problems=[f"{w.kind}: {w.text}" for w in found],
            )
        except (OSError, ValueError):
            pass
        return [{"kind": w.kind, "text": w.text} for w in found]

    def _walk_memo(self) -> Any:
        """This Engine's memo of the last result's leaf walk, so the plausibility checks that read one result in turn walk
        it once (``plausibility.WalkMemo``)."""
        from core import plausibility

        return self.__dict__.setdefault("_plaus_walk_memo", plausibility.WalkMemo())

    def _numeric_findings(self, state: QuestState) -> list[Any]:
        """The numeric warnings of the last run that call for a repair (``block``), none once the person accepted the run."""
        if self.config.engine.numeric_warnings != "block" or state.get("numeric_warnings_accepted"):
            return []
        recorded = (state.get("exec_result") or {}).get("numeric_warnings") or []
        return [
            _numeric.NumericWarning(str(r.get("kind") or ""), str(r.get("text") or ""))
            for r in recorded if isinstance(r, dict)
        ]

    def _numeric_turn(self, state: QuestState) -> bool:
        """Whether the last run ran to the end, its results stand on their own (not all zero, inside their bounds) and
        its numerics warned: the repair is then for the warnings, on their own budget."""
        if (state.get("exec_result") or {}).get("returncode", 0) != 0 or not state.get("result_json"):
            return False
        if not self._numeric_findings(state):
            return False
        if self.config.engine.degenerate_run_guard and _is_degenerate_result(state.get("result_json") or {}):
            return False
        return not _assertion_violations(self._in_force(state), self._walk_memo())

    def _scripts_sha(self, rule: int = 1) -> str:
        if rule >= 2:
            # Rule 2: every file of the study (core/reuse_key.py), so a repair in the package or a parameter file counts.
            from . import reuse_key as _reuse_key

            found = _reuse_key.files(self.quest_root, "study")
            return _reuse_key.digest_text(json.dumps(found.digests, sort_keys=True) + ("" if found.complete else "|unreadable"))
        import hashlib

        digest = hashlib.sha256()
        for name in (_split_run.SIMULATE_NAME, "experiment.py"):
            path = self.quest_root / "code" / name
            if path.is_file():
                digest.update(name.encode("utf-8") + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
        return digest.hexdigest()

    def _resume_after_numeric_stop(self) -> QuestState | None:
        """After a stop for numeric warnings: a script the person changed is run again, and one they did not change
        is accepted as it is. ``None`` when the quest was not stopped for them."""
        marker = self.fi_dir / "numeric_stop.json"
        if not marker.is_file():
            return None
        try:
            noted = json.loads(marker.read_text(encoding="utf-8"))
            was, rule = noted.get("scripts_sha"), int(noted.get("scripts_rule") or 1)  # a marker is read by its own rule
            marker.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError, AttributeError):
            was, rule = None, 1
        if was is not None and was != self._scripts_sha(rule):
            code_path = self.quest_root / "code" / "experiment.py"
            self._log.info("[numeric] the script was changed while the quest was stopped; running it again")
            return {"code": code_path.read_text(encoding="utf-8"), "exec_patch_pending": True}
        self._log.warning("[numeric] resumed with the script unchanged: the run's warnings are accepted")
        return {"numeric_warnings_accepted": True}

    async def _oracle_gate(self, state: QuestState, py: Any, exec_env: Any, seed_path: Path) -> str | None:
        """Run the script's oracle checks (FI_ORACLE=1) and hold the quest at them.

        Returns the text of ``code/experiment.py`` when a repair rewrote it (so it reaches the state), else ``None``.
        A protocol that declares no oracle asks the plan for one first (:meth:`revise_plan`); a check that is missing
        or fails asks for a repair of the script; each is up to ``engine.oracle_repair_attempts`` attempts. If the
        oracles still do not pass after the repairs and FI's corrections (:meth:`_correct_expected_values`), ``block`` goes
        on with them marked unconfirmed (no person is asked) and ``warn`` records it and goes on.
        Every attempt is in ``needs/ORACLE_CHECK.json``."""
        # A background job's one script cannot be run here for a check; a job array's oracle() can (it is a function).
        # Decided by the layout the quest really has: a quest that asks for two scripts can still have one (a reply
        # that held one, or a quest begun before every simulation got its own script), and running a one-script job's
        # driver here would submit the job.
        if self.config.engine.oracle_check == "off" or (
            self.config.execution.background_jobs
            and not (self._split_on(state) and (self.quest_root / "code" / _split_run.SIMULATE_NAME).is_file())
        ):
            self._oracle_record_clear()
            return None
        protocol = self._protocol_block(state)
        if protocol is None and self._runs_code(state) and _frozen.load(self.quest_root) is None:
            # A plan with no protocol at all declares no known-answer check either: the plan is asked for one below
            # (as for a protocol that has none), never skipped, so a simulation never runs with nothing checking it.
            protocol = {}
        if protocol is None:
            self._oracle_record_clear()
            return None
        stride = max(1, int(self.config.engine.replicate_seed_stride))
        # A two-script quest's simulate.py reads FI_RAW_DIR unconditionally at module level (the split-experiment
        # directive's own example: ``raw = pathlib.Path(os.environ["FI_RAW_DIR"])``) — Python runs that line on
        # import regardless of which branch the script takes, so it fires during this oracle pre-check too, well
        # before FI_ORACLE is even read. Without a real value here every such script crashed with
        # ``KeyError: 'FI_RAW_DIR'`` before printing ORACLE_JSON, and the resulting stderr (a bare KeyError, nothing
        # about oracles) sent oracle_repair_attempts chasing the wrong problem. A dedicated, disposable folder — never
        # a real seed's raw dir — keeps this pre-check honest without the prompt needing to special-case it.
        oracle_raw_dir = self._raw_root() / "oracle_check"
        oracle_raw_dir.mkdir(parents=True, exist_ok=True)
        # What the repairs and FI's own look said about the checks themselves, latest per oracle: kept in the record; only
        # a value worked out independently of the measurement is ever applied (``_correct_expected_values``).
        self._oracle_proposals: dict[str, dict[str, Any]] = {}
        # The checks a repair called wrong while they were not passing: the script is not changed for them. (A proposal
        # about a check that passed is shown to the person but does not excuse that check failing later.)
        self._oracle_disputed: set[str] = set()
        # The failing checks FI's own look (core/oracle_triage.py) found are not the script's to fix: a tolerance tighter
        # than the method's own error at its step, or than one random trial's noise. Name -> the finding. They still fail;
        # the script is just not rewritten for them.
        self._oracle_set_aside: dict[str, str] = {}
        self._oracle_noisy: set[str] = set()  # those set aside as one random trial's noise
        self._oracle_undecided: set[str] = set()  # those set aside because their trials did not decide them yet
        # The code questioned the plan's own working (core/oracle_challenge.py): name -> the challenge, and what the
        # exchange decided once it was had; a check the plan's answer sends back to the code carries its reason.
        self._oracle_challenges: dict[str, dict[str, Any]] = {}
        self._oracle_measured: dict[str, float] = {}
        self._oracle_challenge_hints: dict[str, str] = {}
        # The failing checks FI asked the numerical referee about itself (:meth:`_referee_failing_checks`), once each.
        self._oracle_refereed: set[str] = set()
        self._oracle_kept_proposals: dict[str, dict[str, Any]] = {}
        # A dispute outlives a stop: the checks the last run of this gate found disputed, while they are as they were then
        # (a person who changed one, or accepted its proposal, starts it afresh).
        self._restore_oracle_disputes(_oracle.declared(protocol))
        self_check_calls, self_check_runs = 0, 0  # what FI's own look at the failing checks cost this run
        looked: set[str] = set()  # the checks it looked at (once each)
        timeout_retried = False  # a run of the checks that ran out of time is run once more, at twice the time
        env = {
            **_replicate_env(exec_env, 0, stride), "FI_ORACLE": "1",
            _split_run.RAW_DIR_ENV: _split_run.env_value(oracle_raw_dir, self.quest_root),
        }
        timeout = max(30, int(self.config.execution.timeout_s * self.config.engine.pilot_timeout_frac))
        budget = int(self.config.engine.oracle_repair_attempts)
        # Three things can use up a turn here, and they are counted apart: a plan rewrite (an oracle missing, or given
        # no numbers), a rewrite of the script, and a repair call that never got an answer (a provider timeout). Counted
        # together, a quest that first needed its plan completed, then its FI_ORACLE branch added, had no repair left
        # for the numerical problem that only showed once the checks finally ran.
        plans_left, repairs_left, call_failures_left = budget, budget, 1
        # A repair whose code cannot be used (it does not parse, or lacks the function FI calls) changed nothing: not
        # counted as one of the repairs, once per gate run (a second one is, so the loop stays bounded).
        unusable_left = 1
        hinted: set[str] = set()
        attempts: list[dict[str, Any]] = []
        found: list[str] = []
        new_code: str | None = None
        # After a repair is applied while some checks are disputed: (the script before it, new_code before it, the
        # disputed checks that were failing). Read once, right after the next run.
        guard: tuple[str, str | None, set[str], dict[str, str]] | None = None
        attempt = 0
        test_run_read = False  # the first measurement of the checks has been read as a test run of them
        went_on: list[dict[str, Any]] = []  # the failed checks the quest goes on with (a person's choice, or automatic)
        while True:
            protocol = self._protocol_block(state) or protocol  # a plan edit, or the oracle declared just now
            oracles = _oracle.declared(protocol)
            incomplete = _oracle.unjudgeable(oracles)  # the engine judges: an oracle with no numbers to judge by is not run
            # A search for the best design calls the simulation with the conditions held fixed and the search's
            # numerical settings: an oracle's case (the baseline, say) is run with them too.
            run_oracles = [{**o, "case": _optimise.complete_case(protocol, o["case"])}
                           if _optimise.block_of(protocol) is not None and isinstance(o.get("case"), dict) else o
                           for o in oracles]

            async def measure(limit_s: int) -> tuple[dict[str, Any] | None, int, bool, str, list[str]]:
                """One run of the checks: (what was reported, exit code, ran out of time, the error output, the trial
                contract's problems)."""
                reported: dict[str, Any] | None = None
                returncode, timed_out, stderr_tail = 0, False, ""
                trial_problems: list[str] = []
                if getattr(self, "_trial_mode", False):
                    # The trial contract. An oracle with a case is measured by the ENGINE: it calls the simulation function
                    # on that case in its own process and reads the measure from what comes back, so no number of the
                    # script's own making stands in for the simulation. The rest fall back to the script's oracle().
                    try:
                        # (A case that ran out of time says so among the problems: the retry below reads it there.)
                        checks, trial_problems, _ = await _trial_runner.measure_oracles(
                            self.executor, py, self.quest_root, seed_path.relative_to(self.quest_root).as_posix(),
                            run_oracles, timeout_s=limit_s, env=env,
                            thresholds=protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None,
                            case_env=dict(_replicate_env(exec_env, 0, stride)), in_each_run=_in_run.of_study(protocol),
                            mean_names=_mc_error.mean_names(protocol), paired=_mc_error.paired(protocol, self.quest_root),
                            proportion_names=_mc_error.proportion_names(protocol),
                            repeated=_mc_error.repeats_trials(protocol), protocol=protocol,
                            log=self._log,
                        )
                    except Exception as e:  # noqa: BLE001 -- a run that cannot start is a problem to report
                        checks, trial_problems = [], [f"the oracles could not be run: {e!r}"[:300]]
                    reported = {"checks": checks, "engine_measured": True}
                    stderr_tail = getattr(self, "_packages_note", "") + "; ".join(trial_problems)
                    return reported, returncode, False, stderr_tail, trial_problems
                try:
                    ran = await self.executor.execute(
                        [str(py), str(seed_path)], cwd=self.quest_root, timeout_s=limit_s, env=env,
                    )
                    reported, returncode, timed_out = _oracle.parse(ran.stdout), ran.returncode, ran.timed_out
                    if isinstance(reported, dict):
                        reported.pop("engine_measured", None)  # only the engine's own measurement carries this mark
                    # What could not be installed comes first: the repair then fixes the import, not a symptom.
                    stderr_tail = getattr(self, "_packages_note", "") + (ran.stderr or "")[-2000:]
                except Exception as e:  # noqa: BLE001 -- an oracle run that cannot start is a problem to report
                    returncode, stderr_tail = -1, repr(e)
                if reported is None and returncode not in (0, -1):
                    self._log.info("[oracle] the script exited %s without ORACLE_JSON; stderr_tail=%s", returncode, stderr_tail[-300:])
                return reported, returncode, timed_out, stderr_tail, trial_problems

            reported: dict[str, Any] | None = None
            returncode, timed_out, stderr_tail = 0, False, ""
            trial_problems: list[str] = []
            retried: dict[str, Any] | None = None
            if oracles and not incomplete:
                reported, returncode, timed_out, stderr_tail, trial_problems = await measure(timeout)
                ran_out = timed_out or any("ran out of time" in p for p in trial_problems)
                if ran_out and not timeout_retried:
                    # A case that needs a little longer than the pre-check allows is not a failure of the script: one
                    # more run at twice the time, once per gate, before it counts.
                    timeout_retried = True
                    self._log.warning("[oracle] the run of the checks ran out of time (%d s); running it once more with "
                                      "%d s", timeout, timeout * 2)
                    self_check_runs += 1
                    reported, returncode, timed_out, stderr_tail, trial_problems = await measure(timeout * 2)
                    retried = _oracle_triage.timeout_entry(
                        timeout, timed_out or any("ran out of time" in p for p in trial_problems))
                    timeout *= 2  # the rest of this gate's runs get the time the case needed
            found = incomplete or _oracle.problems(oracles, reported, returncode, timed_out)
            if trial_problems and not incomplete:
                # The trial contract: there is no FI_ORACLE run and no ORACLE_JSON line; say why a value is missing.
                found = _oracle.with_run_problems(trial_problems, found, oracles)
            judged_list = _oracle.judged(oracles, reported) if reported is not None else []
            # The last exception of a run that left a check unmeasured: on the card, in the repair request, and compared
            # with the next run's (the same one after a repair stops the repairs early).
            exception = (_oracle_triage.exception_of(stderr_tail, found)
                         if found and oracles and not incomplete
                         and (reported is None or any(j.get("value") is None for j in judged_list)) else "")
            if re.match(r"^[\w.]*Warning\b", exception):
                exception = ""  # a warning is not what stopped it
            attempts.append({
                "attempt": attempt, "oracles": [o["name"] for o in oracles], "problems": found,
                "checks": (reported or {}).get("checks"),
                # What the engine made of each: the script's value against the protocol's expected value and tolerance.
                "judged": judged_list,
                **({"exception": exception} if exception else {}),
                **({"triage": [retried]} if retried else {}),
            })
            attempt += 1
            if guard is not None:
                # The repair just applied was told to leave the disputed checks alone. One that made such a check pass
                # changed the script towards a number the repair itself called wrong: that is put back.
                prev_code, prev_new_code, held, prev_package = guard
                guard = None
                flipped = [j["name"] for j in attempts[-1]["judged"] if j.get("name") in held and j.get("passed_by_engine") is True]
                if flipped:
                    seed_path.write_text(prev_code, encoding="utf-8")
                    self._restore_package(prev_package)  # a change to the model's package goes back with it
                    new_code = prev_new_code
                    attempts[-2]["repair"] = "reverted_disputed_changed"
                    self._log.warning(
                        "[oracle] put %s back as it was: the repair made the disputed check(s) %s pass, and the script may "
                        "not be changed for them (FI found the check, not the script, to be wrong)", seed_path.name,
                        ", ".join(repr(n) for n in flipped),
                    )
                    continue
            if found and oracles and not incomplete and self.config.engine.oracle_check == "block":
                # A person chose to go on although these checks failed, bound to the checks as they are and to this
                # code: honoured before anything (a plan change, a repair) could change either and undo the choice.
                went_on = self._chosen_go_on(found, run_oracles, attempts[-1]["judged"], seed_path)
                if went_on:
                    break
            if oracles and not incomplete and not test_run_read:
                # The first measurement of the checks is a test run of them: a definition mismatch goes back to the plan
                # once, before any repair of the script is spent on it; when the plan changed the checks they are
                # measured again as it now states them.
                test_run_read = True
                if found and _frozen.load(self.quest_root) is None and await self._revise_after_dry_run(oracles, attempts[-1]):
                    continue
            if not found:
                break
            if not oracles or incomplete:
                if _frozen.load(self.quest_root) is not None or plans_left == 0:
                    break  # a frozen protocol takes an oracle only through an amendment
                self._log.warning("[oracle] %s; asking the plan for one (%d of %d)", found[0], budget - plans_left + 1, budget)
                plans_left -= 1
                if not await self._declare_oracles(incomplete or None):
                    break
                continue
            # A check a repair has called wrong is not the script's to fix: its proposal is recorded, and the script
            # is never rewritten towards it. When every problem left is such a check, the script is kept as it is and no
            # more repairs are spent (a real quest's repairs, told the check was wrong, kept rewriting a correct script
            # until it crashed and the quest had no result at all).
            # Before a repair is spent on a failing check, FI looks itself whether the check is what is wrong (its
            # expected value worked out again by another model, its case at a smaller step or with more seeds): see
            # core/oracle_triage.py. Once per check per gate run; nothing here changes a verdict.
            calls, runs = await self._look_at_failing_checks(
                state, py, seed_path, oracles, run_oracles, attempts[-1], looked, protocol=protocol, timeout=timeout,
                case_env=dict(_replicate_env(exec_env, 0, stride)))
            self_check_calls, self_check_runs = self_check_calls + calls, self_check_runs + runs
            # A check FI's own look found wrong, with a value two independent workings agree on (the plan's own
            # arithmetic and another model that never saw the measured value): FI corrects the plan's expected value
            # itself (never to the measured value, never its tolerance) and measures the checks again.
            if self._oracle_challenges:
                await self._answer_challenges(state, oracles, attempts[-1])
            corrected = self._correct_expected_values(attempts[-1]["judged"], oracles)
            if corrected:
                looked.difference_update(corrected)
                attempts[-1]["corrected"] = corrected
                continue
            disputed = [n for n in self._oracle_proposals if n in self._oracle_disputed]
            # Not the script's to fix either: a check whose tolerance FI found tighter than its method's error or noise.
            aside = [n for n in self._oracle_set_aside if n not in disputed]
            to_fix = _oracle.undisputed(found, disputed + aside)
            judged_now = attempts[-1]["judged"]
            if ((not to_fix or repairs_left == 0)
                    and await self._referee_failing_checks(state, oracles, judged_now, protocol, attempts[-1])):
                # Before FI goes on with a check the plan worked out still failing, the number computed from the
                # definitions decides which side is wrong: a corrected check is measured again, a wrong code repaired.
                corrected = self._correct_expected_values(judged_now, oracles)
                for n in [n for n, p in self._oracle_proposals.items() if p.get("source") == "numeric"]:
                    # A correction that could not be made is not left as a proposal; the repair's own comes back.
                    self._oracle_proposals.pop(n, None)
                    if n in self._oracle_kept_proposals and n not in corrected:
                        self._oracle_proposals[n] = self._oracle_kept_proposals[n]
                self._oracle_kept_proposals.clear()
                if corrected:
                    looked.difference_update(corrected)
                    attempts[-1]["corrected"] = corrected
                    continue
                disputed = [n for n in self._oracle_proposals if n in self._oracle_disputed]
                aside = [n for n in self._oracle_set_aside if n not in disputed]
                to_fix = _oracle.undisputed(found, disputed + aside)
            if not to_fix:
                self._log.warning(
                    "[oracle] the script is kept as it is: %s; FI corrects the check only where two independent workings agree, or "
                    "goes on with it marked unconfirmed",
                    "; ".join(
                        [f"the check {n!r} is disputed ({'FI worked its expected value out again' if (self._oracle_proposals.get(n) or {}).get('source') == 'recompute' else 'the repair says it is wrong'})"
                         for n in _oracle.disputed_failing(judged_now, disputed)]
                        + [f"the check {n!r}: {self._oracle_set_aside[n]}" for n in aside]),
                )
                break
            if repairs_left == 0:
                break
            measured_to_fix = [j for j in judged_now if j.get("passed_by_engine") is False
                               and str(j.get("name")) not in disputed + aside]
            again = "" if measured_to_fix else _oracle_triage.same_exception(attempts)
            if again:
                # The repair changed the script and the same error came back: another rewrite of the same kind would not
                # reach it. The repairs left are kept, and the card says what the error is and where.
                attempts[-1].setdefault("triage", []).append(_oracle_triage.same_exception_entry(again, repairs_left))
                self._log.warning("[oracle] the same error came back after the repair (%s): no more repairs are spent on "
                                  "it", again)
                break
            self._log.warning(
                "[oracle] %d problem(s): %s; asking for a repair (%d of %d)",
                len(to_fix), "; ".join(to_fix), budget - repairs_left + 1, budget,
            )
            code_before, new_code_before = seed_path.read_text(encoding="utf-8"), new_code
            package_before = self._package_snapshot()
            hints = self._multiple_hints(oracles, measured_to_fix, attempts[-1], hinted)
            hints += [self._oracle_challenge_hints.pop(n) for n in list(self._oracle_challenge_hints)
                      if n in {str(j.get('name')) for j in measured_to_fix}]
            self._oracle_measured = {str(j.get("name") or "").strip(): float(j["value"]) for j in judged_now
                                     if isinstance(j.get("value"), (int, float)) and not isinstance(j.get("value"), bool)}
            text, call_failed, outcome = await self._repair_script_for_oracle(
                state, seed_path, oracles, to_fix,
                # What FI found about the numbers (a measured value that is half the value it worked out, say), then
                # the error that stopped the run, said first: the repair goes to it, not to a symptom.
                "".join(f"FI found: {h}\n" for h in hints)
                + (f"The run of the checks stopped with: {attempts[-1]['exception']}\n" if attempts[-1].get("exception")
                   else "") + stderr_tail,
                disputed=disputed + aside,
                passing=[str(j["name"]) for j in judged_now if j.get("passed_by_engine") is True],
            )
            attempts[-1]["repair"] = outcome
            if getattr(self, "_oracle_patch_summary", ""):
                # What the repair said it changed: the card's "what FI already tried" reads it from the record.
                attempts[-1]["patch_summary"] = self._oracle_patch_summary
            package_after = self._package_snapshot()
            if package_after != package_before:
                # A fix in the model's package: the files it changed, by their new hash.
                attempts[-1]["package_files"] = {
                    rel: _attempts._sha(body.encode("utf-8")) for rel, body in sorted(package_after.items())
                    if package_before.get(rel) != body}
            if call_failed and call_failures_left > 0:
                call_failures_left -= 1  # no answer came back: the script was not rewritten, so the repair is not spent
            elif outcome == "not_usable" and unusable_left > 0:
                unusable_left -= 1
                self._log.warning("[oracle] the repair's code could not be used, so it does not count as one of the %d "
                                  "repairs (once per run of the checks)", budget)
            elif outcome != "set_aside_disputed":
                # A set-aside answer is not spent either: the next request, without that check, is the repair. It is
                # bounded: a check can be newly disputed only once.
                repairs_left -= 1
            if text is not None and seed_path.name == "experiment.py":
                new_code = text
            # Every disputed check not passing now, measured or not: a repair that restores a missing measurement must
            # not restore it bent to the disputed number either.
            passing_now = {str(j["name"]) for j in judged_now if j.get("passed_by_engine") is True}
            # A check its trials had not decided is not held either: decided as a pass after the repair, three standard
            # errors inside its tolerance, it is a real pass, never a sign the repair bent the script to it.
            held = {n for n in disputed + aside if n not in passing_now and n not in self._oracle_noisy
                    and n not in self._oracle_undecided}
            if outcome == "applied" and held:
                guard = (code_before, new_code_before, held, package_before)
        # What the stop may offer: "mark it unconfirmed and go on", only when every problem is a check that was measured
        # and failed (core/accepted_checks.py::offer), bound to the checks' conditions and to this version of the code.
        go_on_checks, go_on_why_not = (_accepted.offer(found, run_oracles, attempts[-1].get("judged") or [])
                                       if found else ([], ""))
        version = self._measuring_code_version(seed_path)
        if found and not went_on and self._goes_on_by_itself():
            # Whether a check is right is not a question a person is asked: when FI could neither fix the simulation
            # nor correct the check from an independent value, it goes on by itself with each failed check marked
            # unconfirmed (one that measured nothing too), recorded as automatic and bound, as a person's choice was, to
            # the check and to this version of the code. The evidence keeps the result below "independently validated"
            # and the paper says so: nothing about the check is relaxed.
            # A check the trials could not decide carries ``undecided`` from its verdict (accepted_checks._mc_said).
            went_on = [{**c, "by": _accepted.AUTOMATIC, "via": _accepted.AUTOMATIC, "script": self._rel_to_quest(seed_path),
                        "script_version": version,
                        "trial_basis": _oracle.trial_basis(protocol)}
                       for c in (go_on_checks or _accepted.went_on_entries(found, run_oracles,
                                                                           attempts[-1].get("judged") or []))]
        # A check the code and the plan's own working still disagree on (core/oracle_challenge.py) is said as that, with
        # every side's reason: not as a failure of the code.
        for c in went_on or []:
            settled = (self._oracle_challenges.get(str(c.get("name") or "")) or {}).get("outcome") or {}
            if settled.get("outcome") == "unsettled":
                c["disagreement"] = settled["text"]
        status = ("ok" if not found else "warned" if self.config.engine.oracle_check == "warn"
                  else _accepted.WENT_ON if went_on else "stopped")
        proposed = list(self._oracle_proposals.values())
        # The checks still failing (a value measured and judged outside tolerance) whose expected value a repair
        # disputes: the record, the analysis and the paper say the check failed and that its expected value is disputed
        # (a person has not approved the proposed one).
        disputed_failing = _oracle.disputed_failing(attempts[-1].get("judged") or [], sorted(self._oracle_disputed)) if found else []
        if disputed_failing:
            by_name = {p["name"]: p for p in proposed}
            for j in attempts[-1].get("judged") or []:
                if j.get("name") in disputed_failing:
                    j["disputed_expected"] = by_name[j["name"]]["expected"]
                    if by_name[j["name"]].get("source") == "recompute":
                        # Said differently when the recheck was not independent (the plan's own model, or working that
                        # does not add up): the analysis and the paper never call it another model.
                        proposal = by_name[j["name"]]
                        j["disputed_by"] = ("recompute" if self._independent_value(proposal)
                                            else "recompute_unsound" if proposal.get("how_slip")
                                            else "recompute_same_model")
        loose = [w for w in (_oracle.loose_tolerance(o) for o in oracles) if w]
        for warning in loose:
            self._log.warning("[oracle] %s", warning)
        aside_failing = [n for n in self._oracle_set_aside if found and n not in disputed_failing and any(
            j.get("name") == n and (j.get("passed_by_engine") is False or j.get("undecided"))
            for j in attempts[-1].get("judged") or [])]
        for duplicate in _oracle.duplicate_names(oracles, {"checks": attempts[-1].get("checks") or []}):
            self._log.warning("[oracle] %s", duplicate)
        # Which side FI's own look points to, and why, in a scientist's words (core/oracle_card.py): recorded with the
        # automatic decision and said in the console; the numbers stay in the record and run.log.
        explained: dict[str, str] = {}
        if found:
            all_disputed = (bool(disputed_failing or aside_failing)
                            and not _oracle.undisputed(found, disputed_failing + aside_failing))
            try:
                card = _oracle_card.build(
                    quest_id=self.quest_id, quest_root=self.quest_root, script=seed_path, found=found, oracles=oracles,
                    judged=attempts[-1].get("judged") or [], attempts=attempts, proposals=proposed,
                    disputed=disputed_failing, trial=bool(getattr(self, "_trial_mode", False)),
                    frozen=_frozen.load(self.quest_root) is not None,
                    research=getattr(self.config, "rigor_profile", "default") == "research",
                    repairs=budget, stderr_tail=stderr_tail,
                    kept=("as_it_was" if not any(a.get("repair") == "applied" for a in attempts) else "for_disputed")
                    if all_disputed else None)
                explained = {"leaning": str(card.get("leaning") or ""), "why": str(card.get("why") or "")}
            except Exception as e:  # noqa: BLE001 -- an explanation that cannot be made never stops a quest
                self._log.warning("[oracle] could not say why the checks did not pass (%r); run.log has them", e)
        self._oracle_record({
            "status": status, "judged_by": "engine", "attempts": attempts, "problems": found,
            **({"contract": "trial"} if getattr(self, "_trial_mode", False) else {}),
            **({"warnings": loose} if loose else {}),
            **({"proposed_changes": proposed} if proposed else {}),
            **({"disputed": disputed_failing} if disputed_failing else {}),
            **({"set_aside": {n: self._oracle_set_aside[n] for n in aside_failing}} if aside_failing else {}),
            # Each check's version, so the next run keeps a dispute only while the check is as it was.
            "fingerprints": {str(o["name"]).strip(): _oracle_triage.fingerprint(o) for o in oracles},
            **({"self_checks": {"model_calls": self_check_calls, "extra_runs": self_check_runs}}
               if self_check_calls or self_check_runs else {}),
            **({"went_on": went_on, "summary": "; ".join(_accepted.gap(e) for e in went_on)} if status == _accepted.WENT_ON
               else {}),
            **({"explained": explained} if explained else {}),
            **({"corrected": self._corrections_read()} if self._corrections_read() else {}),
            **({"fitted_to_test_run": fitted} if (fitted := self._fitted_to_test_run(oracles)) else {}),
            **({"go_on": {"offered": bool(go_on_checks), "why_not": go_on_why_not, "checks": go_on_checks,
                          "script": self._rel_to_quest(seed_path), "script_version": version}}
               if status == "stopped" else {}),
        })
        if self_check_calls or self_check_runs:
            line = _oracle_triage.cost_line(self_check_calls, self_check_runs)
            noted = _oracle_triage.read(self.fi_dir)
            # Written when this run asked a model, or the first time: a resume that only ran the cases again is not.
            if noted.get("plan_line") != line and (self_check_calls or not noted.get("plan_line")):
                try:
                    self._write_plan_section(_plan.plan_path(self.quest_root), ["", "- " + line],
                                             note="what FI's own look at the failing known-answer checks cost")
                    _oracle_triage.write(self.fi_dir, {**noted, "plan_line": line})
                except OSError as e:
                    self._log.warning("[oracle] couldn't write what the look at the checks cost into plan.md: %r", e)
        if not found:
            self._log.info(
                "[oracle] %d oracle(s) passed before the main run%s", len(attempts[-1]["oracles"]),
                "" if len(attempts) == 1 else f" (after {len(attempts) - 1} repair(s))",
            )
            return new_code
        if status == _accepted.WENT_ON:
            for entry in went_on:
                sentence = _accepted.gap(entry)
                self._log.warning("[oracle] going on: %s (the check stays marked %s; the result does not count as checked "
                                  "against known answers)", sentence, _accepted.UNCONFIRMED)
            names = ", ".join(repr(str(e.get("name"))) for e in went_on if e.get("name")) or "the known-answer checks"
            why = explained.get("why") or ""
            line = (f"[FI] FI could not confirm {names}" + (f" ({why})" if why else "") + ", so the result counts as "
                    "exploratory, not as checked against a known answer; the paper says so. Nothing to do (details: "
                    "needs/ORACLE_CHECK.json and run.log).")
            self._say_once(line)
            return new_code
        if self.config.engine.oracle_check == "warn":
            self._log.warning(
                "[oracle] the oracles did not pass: %s;%s going on (engine.oracle_check: warn)", "; ".join(found),
                f" the expected value of {', '.join(repr(n) for n in disputed_failing)} is disputed (not approved, so "
                "the check counts as failed);" if disputed_failing else "",
            )
            return new_code
        # Not reached with the checks on: the gate goes on by itself above (``_goes_on_by_itself``).
        return new_code

    def _multiple_hints(self, oracles: list[dict[str, Any]], measured: list[dict[str, Any]],
                        attempt: dict[str, Any], hinted: set[str]) -> list[str]:
        """For each failing check measured at a simple multiple (half, twice ...) of a value FI worked out
        independently of the plan's word (its correction from the plan's own arithmetic, or from an independent
        recheck): a triage entry pointing to the script, and the sentence for the repair request. Never against the
        plan's own formula alone (a plan whose physics is off by two would send the repair to bend a correct script).
        Once per check per gate run; never a verdict."""
        corrected = self._corrections_read()
        declared = {str(o.get("name") or "").strip(): o for o in oracles}
        hints: list[str] = []
        for j in measured:
            name = str(j.get("name") or "").strip()
            oracle = declared.get(name)
            if oracle is None:
                continue
            fix = corrected.get(name) if isinstance(corrected.get(name), dict) else None
            if not fix or fix.get("source") != "arithmetic" or not fix.get("confirmed_by"):
                continue
            reference = _oracle_triage.num(fix.get("to"))
            source = "its correction of the plan's arithmetic, confirmed by another model"
            expected_now, limit, _mode = _oracle.limit_of(oracle)
            if reference is None or expected_now is None or limit is None or abs(expected_now - reference) > limit:
                continue  # a correction no longer in force (the check was written again since): not FI's value now
            found = _oracle_triage.multiple_of(j.get("value"), reference, limit)
            if found is None:
                continue
            entry = _oracle_triage.multiple_entry(oracle, float(j["value"]), float(reference), found, source)
            if name not in hinted:
                hinted.add(name)
                attempt.setdefault("triage", []).append(entry)
                self._log.warning("[oracle] %s", entry["hint"])
            hints.append(entry["hint"])
        return hints

    def _say_once(self, line: str) -> None:
        """Print ``line`` to the console unless this run already printed it (the checks run again after a later repair
        of the run, and would say the same sentence twice)."""
        said = getattr(self, "_said_lines", set())
        if line not in said:
            print(line)
            self._said_lines = said | {line}

    @staticmethod
    async def _installed_distributions(py: Any) -> list[str] | None:
        """``name==version`` of every distribution the interpreter ``py`` has installed (importlib.metadata, no pip, no
        version-control probe), sorted; ``None`` when it cannot be listed."""
        # The first distribution of a name on sys.path is the one imported (as pip lists it); an editable or direct
        # install is written as pip would (``name @ url``) and marked, since its version alone cannot bring it back.
        code = ("import importlib.metadata as m, json\n"
                "rows = {}\n"
                "for d in m.distributions():\n"
                "    n = (d.metadata['Name'] or '').strip()\n"
                "    if not n or n in rows:\n"
                "        continue\n"
                "    line = f'{n}=={d.version}'\n"
                "    try:\n"
                "        u = json.loads(d.read_text('direct_url.json') or 'null')\n"
                "    except Exception:\n"
                "        u = None\n"
                "    if isinstance(u, dict) and u.get('url'):\n"
                "        ed = (u.get('dir_info') or {}).get('editable')\n"
                "        line = f\"{n} @ {u['url']}\" + (' (editable)' if ed else '')\n"
                "    rows[n] = line\n"
                "print('\\n'.join(sorted(rows.values())))\n")
        try:
            tree = await AsyncProcessTree.start(str(py), "-c", code, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE, env=scrub_secrets())
            aborted = False
            try:
                out, _err = await asyncio.wait_for(tree.proc.communicate(), timeout=_PIP_FREEZE_TIMEOUT_S)
            except BaseException:
                aborted = True
                raise
            finally:
                await tree.aclose(aborted=aborted)
        except Exception:  # noqa: BLE001 -- a diagnostic must never stall a quest
            return None
        if tree.proc.returncode != 0:
            return None
        return sorted(line for line in out.decode("utf-8", "replace").splitlines() if line.strip())

    def _goes_on_by_itself(self) -> bool:
        """Whether a known-answer check that still fails after FI's repairs and corrections lets the quest go on by
        itself, recorded as automatic: always, with the checks on (``engine.oracle_check: block``). Whether the simulation
        or the check is wrong is FI's to work out, never a person's; what FI could not confirm is marked unconfirmed,
        so the result counts as exploratory whatever the quest was set up for (a research quest included)."""
        return self.config.engine.oracle_check == "block"

    def _fitted_to_test_run(self, oracles: list[dict[str, Any]]) -> list[str]:
        """The checks the plan changed after FI's test run so that they pass on its own numbers, still in the protocol."""
        # Kept with the record of the engine's changes to plan.md (.fi/oracles_added.json): it goes with plan.md.
        fitted = (self._oracles_added_read() or {}).get("fitted") or []
        names = {str(o.get("name") or "").strip() for o in oracles}
        return sorted(str(n) for n in fitted if str(n) in names)

    def _unconfirmed_checks(self) -> set[str]:
        """The known-answer checks this quest could not confirm: gone on with failed, disputed, set aside as wrong, or
        fitted to the test run. The improve step never optimises against one of them."""
        record = self._oracle_record_read() or {}
        out = {str(e.get("name")) for e in record.get("went_on") or [] if isinstance(e, dict) and e.get("name")}
        out |= {str(n) for n in record.get("disputed") or []}
        # A check whose formula gives another number than the plan typed, left as typed because correcting it would have
        # made the plan unreadable: nothing confirms either number.
        out |= {str(d.get("name")) for d in (self._oracles_added_read() or {}).get("formula_disagreements") or []
                if isinstance(d, dict) and d.get("name")}
        out |= {str(n) for n in (record.get("set_aside") or {})}
        out |= {str(n) for n in record.get("fitted_to_test_run") or []}
        # Under ``engine.oracle_check: warn`` a failing check is only warned about: still not confirmed.
        if record.get("status") == "warned":
            last = (record.get("attempts") or [{}])[-1] if isinstance(record.get("attempts"), list) else {}
            out |= {str(j.get("name")) for j in (last or {}).get("judged") or []
                    if isinstance(j, dict) and j.get("name") and j.get("passed_by_engine") is False}
        # A check corrected to another model's value: nothing independent confirms it (core/evidence.py says so too).
        out |= {str(n) for n, c in (record.get("corrected") or {}).items()
                if isinstance(c, dict) and (c.get("source") in ("recompute", "numeric")
                                            or (c.get("source") == "arithmetic" and not c.get("confirmed_by")))}
        return out

    def _corrections_read(self) -> dict[str, Any]:
        try:
            data = json.loads((self.fi_dir / self._CORRECTIONS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _independent_value(proposal: dict[str, Any]) -> bool:
        """Whether a proposed expected value was worked out independently of the measurement and of the plan's own model:
        the plan's own arithmetic worked out by FI, or another model that never saw the measured value and whose own
        working adds up. A repair's proposal (it saw the run) and the plan's model asked again are never used."""
        if proposal.get("source") == "arithmetic":
            return True
        # A record that does not say which model answered (one an earlier FI wrote) is not counted as independent.
        return (proposal.get("source") == "recompute" and proposal.get("same_model") is False
                and not proposal.get("how_slip"))

    @staticmethod
    def _may_correct(proposal: dict[str, Any]) -> bool:
        """Whether FI may rewrite a check's expected value: only when TWO workings independent of each other agree on it -- the
        plan's own arithmetic worked out by FI AND another model that never saw the measured value (``confirmed_by``,
        :meth:`_second_source`). The arithmetic alone can misread a correct derivation (a truncated series written with
        ``=``, a function with a peak inside the range of its rounded constants), and one model alone can be wrong: on
        either alone the check is only set aside and goes on unconfirmed, which can lower a result, never raise one."""
        confirm = proposal.get("confirmed_by")
        if proposal.get("source") == "numeric":
            # The simulation's measurement and a script that computed the number from the definitions alone, never
            # shown the plan's working or the measurement (:meth:`_referee_failing_checks`).
            return (isinstance(confirm, dict) and confirm.get("source") == "numeric"
                    and _oracle_triage.num(confirm.get("value")) is not None)
        if proposal.get("source") == "challenge":
            # The code's formula, computed by FI, and the plan's corrected formula or the judge's, computed by FI too.
            return (isinstance(confirm, dict) and confirm.get("source") == "challenge"
                    and _oracle_triage.num(confirm.get("value")) is not None)
        return (proposal.get("source") == "arithmetic" and isinstance(confirm, dict)
                and confirm.get("source") == "recompute" and _oracle_triage.num(confirm.get("value")) is not None)

    @staticmethod
    def _still_holds(oracle: dict[str, Any], proposal: dict[str, Any]) -> bool:
        """The correction worked out again against the check as it is NOW (never trusting stored fields alone): the
        plan's current derivation still has the slip and gives the proposed value, and the second source's value lies
        within the check's own tolerance of it."""
        if proposal.get("source") == "numeric":
            # No formula to work out again: the number is the script's, kept for the check as it was when it ran.
            confirm = proposal.get("confirmed_by") or {}
            value = _oracle_triage.num(proposal.get("expected"))
            return (value is not None and value == _oracle_triage.num(confirm.get("value"))
                    and confirm.get("fingerprint") == _oracle_triage.fingerprint(oracle))
        if proposal.get("source") == "challenge":
            confirm = proposal.get("confirmed_by") or {}
            value = _oracle_triage.num(proposal.get("expected"))
            again = _oracle_challenge.computes(oracle, proposal.get("expected_formula"))
            if value is None or again is None or confirm.get("fingerprint") != _oracle_triage.fingerprint(oracle):
                return False
            _expected, limit, _mode = _oracle.limit_of({**oracle, "expected": value})
            return limit is not None and abs(again - value) <= limit
        slip = _oracle_triage.plan_slip(oracle)
        value = _oracle_triage.num(proposal.get("expected"))
        confirm = proposal.get("confirmed_by") or {}
        second = _oracle_triage.num(confirm.get("value"))
        if slip is None or value is None or second is None:
            return False
        if confirm.get("fingerprint") != _oracle_triage.fingerprint(oracle):
            return False  # confirmed for the check as it was (another case, measure or statement): not this one
        _expected, limit, _mode = _oracle.limit_of({**oracle, "expected": value})
        return limit is not None and abs(slip["computes"] - value) <= limit and abs(second - value) <= limit

    @staticmethod
    def _second_source(oracle: dict[str, Any], proposal: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any] | None:
        """The independent recheck (``entry``) as the second source for the plan's own arithmetic (``proposal``): another
        model (never the plan's own; :meth:`_recompute_expected` fails closed), whose own working adds up, that got the
        value the arithmetic gives within the check's own tolerance (never widened). ``None`` otherwise."""
        recomputed = _oracle_triage.num(entry.get("recomputed"))
        value = _oracle_triage.num(proposal.get("expected"))
        if (recomputed is None or value is None or entry.get("same_model") is not False or entry.get("how_slip")
                or proposal.get("source") != "arithmetic" or entry.get("verdict") == "agrees"
                or not entry.get("blind")):
            return None  # agreeing with what the plan wrote, or shown the plan's working, confirms nothing
        _expected, limit, _mode = _oracle.limit_of({**oracle, "expected": value})
        if limit is None or abs(recomputed - value) > limit:
            return None
        return {"source": "recompute", "model": str(entry.get("model") or ""), "value": recomputed,
                "fingerprint": entry.get("fingerprint") or _oracle_triage.fingerprint(oracle)}

    def _correct_expected_values(self, judged: list[dict[str, Any]], oracles: list[dict[str, Any]]) -> list[str]:
        """Correct, in ``plan.md``, the expected value of each failing check FI's own look found wrong, to the value two
        independent workings agree on (:meth:`_may_correct`: the plan's own arithmetic and another model's recheck). Only the ``expected`` number changes (never
        the tolerance, the case or the measure), only before the protocol is frozen, and each check at most once per
        quest. Every correction is a plan version by the engine, an audit event, a line in run.log and the console, and a
        record (``.fi/oracle_corrections.json``) the to-do list and the freeze read. Returns the checks corrected."""
        if _frozen.load(self.quest_root) is not None:
            return []
        path = _plan.plan_path(self.quest_root)
        if not path.is_file():
            return []
        done = self._corrections_read()
        failing = {str(j.get("name") or "").strip() for j in judged
                   if j.get("passed_by_engine") is False and _oracle_triage.num(j.get("value")) is not None}
        declared = {str(o.get("name") or "").strip(): o for o in oracles}
        todo: dict[str, dict[str, Any]] = {}
        for name, proposal in self._oracle_proposals.items():
            value = _oracle_triage.num(proposal.get("expected"))
            if (name in failing and name in declared and name not in done and value is not None
                    and self._may_correct(proposal) and self._still_holds(declared[name], proposal)):
                todo[name] = {**proposal, "expected": value}
        if not todo:
            return []
        before: dict[str, Any] = {}

        def change(block: dict[str, Any]) -> dict[str, Any] | None:
            protocol = block.get("protocol")
            rows = protocol.get("oracles") if isinstance(protocol, dict) else None
            if not isinstance(rows, list):
                return None
            for row in rows:
                name = str(row.get("name") or "").strip() if isinstance(row, dict) else ""
                if name in todo:
                    before[name] = row.get("expected")
                    row["expected"] = todo[name]["expected"]  # the expected value only (tolerance, case, measure kept)
                    if todo[name].get("source") == "challenge" and str(todo[name].get("expected_formula") or "").strip():
                        # A challenge settled it with a working FI computed: the plan's formula is that working now, so
                        # the plan never shows a value and a formula that disagree (nor recomputes the old value back).
                        row["expected_formula"] = str(todo[name]["expected_formula"]).strip()
                    elif todo[name].get("source") == "numeric" and "expected_formula" in row:
                        # The plan's working was found wrong and nothing replaces it: it is taken out, so the plan never
                        # shows a value and a formula that disagree. What it was stays in the plan's history.
                        row.pop("expected_formula", None)
                    # The plan keeps saying why the number is what it is: what FI changed, after the derivation.
                    note = (f"(FI corrected the expected value from {before[name]} to "
                            f"{_oracle.fmt_digits(todo[name]['expected'])}: {str(todo[name].get('reason') or '')[:200]})")
                    reference = row.get("reference")
                    row["reference"] = f"{reference} {note}".strip() if isinstance(reference, str) else note
            return block if before else None

        text = path.read_text(encoding="utf-8")
        new = _plan.edit_design_block(text, change)
        if new is None or not before:
            return []
        names = sorted(before)
        why = {n: str(todo[n].get("reason") or "")[:300] for n in names}
        note = "; ".join(f"FI corrected the expected value of '{n}' from {before[n]} to "
                         f"{_oracle.fmt_digits(todo[n]['expected'])}: {why[n]}" for n in names)
        path.write_text(new, encoding="utf-8")
        _plan.record_version(self.quest_root, new, by="engine", note=note[:300])
        self._note_engine_change(names, reason="FI corrected an expected value two independent workings agree on "
                                               "(the plan's own arithmetic and another model), never the measurement")
        measured = {str(j.get("name") or "").strip(): _oracle_triage.num(j.get("value")) for j in judged}

        def agrees(n: str) -> bool:
            """Whether the corrected value lies within the check's own tolerance of what was measured: said in the
            record and the console, so a reader knows the check now confirms the plan's own formula was computed, not a
            fact found independently of both."""
            value, new_expected = measured.get(n), todo[n]["expected"]
            tol = _oracle_triage.num(declared[n].get("tolerance")) or 0.0
            if value is None:
                return False
            if str(declared[n].get("tolerance_mode") or "").strip().lower() == "relative":
                return abs(value - new_expected) <= tol * abs(new_expected)
            return abs(value - new_expected) <= tol

        record = {**done}
        for n in names:
            record[n] = {"from": before[n], "to": todo[n]["expected"], "source": todo[n].get("source"),
                         "confirmed_by": todo[n].get("confirmed_by"), "why": why[n], "at": _frozen.now(),
                         **({"agrees_with_measurement": True} if agrees(n) else {})}
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            (self.fi_dir / self._CORRECTIONS).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[oracle] couldn't record the corrected expected values: %r", e)
        self._audit_check("oracle_correction", path, status="corrected", summary=note[:400], problems=names)
        for n in names:
            # The check changed: what FI's look found about its old value no longer applies to it.
            self._oracle_proposals.pop(n, None)
            self._oracle_disputed.discard(n)
            self._oracle_set_aside.pop(n, None)
            self._oracle_noisy.discard(n)
            self._oracle_undecided.discard(n)
            self._log.warning("[oracle] corrected the expected value of %r: %s -> %s (%s); measuring the checks again",
                              n, before[n], _oracle.fmt_digits(todo[n]["expected"]), why[n])
            second = todo[n].get("confirmed_by") or {}
            also = ("" if second.get("source") == "challenge" else
                    " A script that computed it from the definitions alone, never shown the plan's working or the "
                    "measured value, gave it." if second.get("source") == "numeric" else
                    f" Another model ({second.get('model') or 'not the one that wrote the plan'}), never shown the "
                    f"measured value, got {_oracle.fmt_digits(second.get('value'))} too.")
            print(f"[FI] FI corrected an expected value the plan had worked out wrongly: '{n}' {before[n]} -> "
                  f"{_oracle.fmt_digits(todo[n]['expected'])}, because {_lower_first(why[n])}.{also}")
            if record[n].get("agrees_with_measurement"):
                print(f"[FI] The corrected value of '{n}' matches what was measured: this check now shows the simulation "
                      "computes the plan's own formula correctly, not that the formula itself is right.")
        return names

    def _rel_to_quest(self, path: Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.quest_root.resolve()).as_posix()
        except ValueError:
            return Path(path).name

    def _measuring_code_version(self, seed_path: Path) -> str:
        """The version of the code that measured the known-answer checks: the script the checks ran, the helper modules beside it and the model's
        package in ``code/`` (core/accepted_checks.py::script_version). A choice to go on with a failed check is bound to
        it."""
        files: dict[str, str] = {}
        code = self.quest_root / "code"
        # The seed and the helper modules beside it it may import (FI's own scripts and the other main script are not
        # what measured the checks), then the model's package.
        other = "experiment.py" if seed_path.name == _split_run.SIMULATE_NAME else _split_run.SIMULATE_NAME
        skip = {other, _trial_runner.SUBMIT_NAME, "run.py", _code_project.SEARCH, "replot_figures.py",
                "replot_layout.py", "web_plots.py"}
        for path in [seed_path, *sorted(p for p in code.glob("*.py") if p.name not in skip and p != seed_path)]:
            try:
                files[self._rel_to_quest(path)] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
        files.update({f"code/{rel}": text for rel, text in self._package_snapshot().items()})
        return _accepted.script_version(files)

    def _chosen_go_on(self, found: list[str], oracles: list[dict[str, Any]], judged: list[dict[str, Any]],
                      seed_path: Path) -> list[dict[str, Any]]:
        """The failed checks a person chose to go on with, when that choice covers every problem ``found`` names and each
        check is as it was then, measured by the same code; else ``[]`` (and run.log says which choice no longer
        applies, and why: the check is judged again)."""
        checks, _why_not = _accepted.offer(found, oracles, judged)
        if not checks or _accepted.failing_accepted(self.quest_root) is None:
            return []
        version = self._measuring_code_version(seed_path)
        out: list[dict[str, Any]] = []
        covered = True
        for check in checks:
            entry = _accepted.went_on_by(self.quest_root, check, version)
            if entry is None:
                covered = False
                why = _accepted.no_longer_applies(self.quest_root, check, version)
                said = self.__dict__.setdefault("_go_on_lapsed_said", set())
                if why and (check["name"], why) not in said:  # once per check and reason, not on every repair round
                    said.add((check["name"], why))
                    self._log.warning("[oracle] the choice to go on with the check %r no longer applies: %s; it is "
                                      "judged again", check["name"], why)
                    print(f"[FI] the choice to go on with the known-answer check {check['name']!r} no longer applies "
                          f"({why}): it is judged again.")
                continue
            out.append({**check, "by": entry.get("by"), "via": entry.get("via"), "at": entry.get("at"),
                        "script": entry.get("script"), "script_version": version})
        return out if covered else []

    async def _record_criteria(self, state: QuestState, *, attempt: str | None = None, result: Any = None,
                               repaired: bool = False) -> None:
        """Compute each of the frozen protocol's criteria (core/criteria.py) from what FI measured in this run, and append
        a row to ``.fi/criteria_history.jsonl`` with the commit of ``code/`` that ran. Numbers FI measured only: the oracle
        gate's verdicts (a value the script's own ``oracle()`` gave is shown, not counted) and FI's own record of the
        trials; never the script's results, and no run of its own. Records, decides nothing, and never stops a quest.
        ``result``: the study's results when the run produced them; the entries of code/CHANGELOG.md waiting for a run
        then say whether the results changed (core/changelog.py). ``repaired``: FI repaired the script since it was last
        recorded (a crash fix, the check against known answers), so an unrecorded change is a fix."""
        try:
            frozen = _frozen.load(self.quest_root)
            protocol = _frozen.protocol_of(self.quest_root) or self._draft_protocol(state)
            items = _criteria.declared(protocol) if isinstance(protocol, dict) else []
            if isinstance(protocol, dict) and protocol.get("criteria") not in (None, "", []):
                for note in _criteria.repair(protocol["criteria"], protocol)[1]:
                    self._log.warning("[criteria] not computed: %s", note)
            trial_mode = bool(getattr(self, "_trial_mode", False))
            why_missing: dict[str, str] = {}
            judged = self._judged_checks()
            if not judged:
                why_missing["oracle"] = ("the checks against known answers are off (engine.oracle_check: off)"
                                         if self.config.engine.oracle_check == "off" else
                                         "the checks against known answers gave no verdict this run")
            series: dict[str, dict[str, list[float]]] = {}
            names = {c["trials"] for c in items if c.get("trials")}
            if names and trial_mode:
                series = _trial_runner.recorded_series(self.quest_root, names)
            elif names:
                why_missing["trials"] = "FI keeps its own record of the trials only when it runs them itself"
            results = _criteria.evaluate(items, judged=judged, series=series, why_missing=why_missing)
            commit, changed = await asyncio.to_thread(_code_project.head, self.quest_root)
            if changed:
                # A repair rewrote the code after it was recorded (the check against known answers, or a person): the
                # row names the commit of the code that actually ran.
                if await asyncio.to_thread(
                        _code_project.record_change, self.quest_root,
                        "the code as it ran (changed since it was last recorded)",
                        category="fixed" if repaired else "changed", log=self._log):
                    commit, changed = await asyncio.to_thread(_code_project.head, self.quest_root)
            digest = _improve.headline_digest(result)
            row = _criteria.record(
                self.quest_root, run=_frozen.run_id(self.quest_root), code_commit=commit, results=results,
                code_changed=changed, protocol_version=int(frozen.get("version", 1) or 1) if frozen else None,
                protocol_sha256=str(frozen.get("sha256")) if frozen else None,
                protocol_problem=str(frozen.get("problem") or "") or None if frozen else None,
                attempt=attempt, result_digest=digest or None,
            )
            if digest:
                # The changes this run measured: their "Did the results change" line, against the last run with results.
                line = _changelog.run_line(_criteria.history(self.quest_root), row)
                await asyncio.to_thread(_code_project.fill_pending, self.quest_root, line, log=self._log)
            line = _criteria.summary_line(results)
            if results or self.__dict__.get("_said_no_criteria"):
                self._log.info("[criteria] run %s: %s", row["n"], line)
            else:
                self._said_no_criteria = True
                self._log.warning("[criteria] run %s: %s", row["n"], line)
        except Exception as e:  # noqa: BLE001 -- a record of the criteria must never stop a quest
            self._log.warning("[criteria] could not be computed this run: %r", e)

    # ---- the improve loop and the ratchet (core/improve.py) ---------------------------------------------------------

    def _improve_trial_mode(self, state: QuestState) -> bool:
        """Whether FI runs the simulation's trials itself: from this run, or read from the code after a resume."""
        known = self.__dict__.get("_trial_mode")
        if known is not None:
            return bool(known)
        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME
        return (self._split_on(state) and simulate.is_file()
                and bool(_trial_runner.entries(simulate) & {"run_trial", "run_cell"}))

    def _run_produced(self, result: Any) -> bool:
        """Whether a run of the experiment produced its results: it exited 0, or it exited 1 only because a figure of the study's
        failed. The second is read from FI's own record of the run (derived.json) and only when FI wrote the analysis and the run
        printed its RESULT line. Decided once per run (``exec_result["results_produced"]``); the figure's repair keeps reading the
        return code and the figure statuses."""
        rc = result.returncode
        return _analysis_frame.run_produced(
            rc, bool(rc != 0 and _extract_result_json(getattr(result, "stdout", "") or "")),
            _analysis_frame.read_record(self.quest_root) if rc == 1 else {}, rc == 1 and self._analysis_on())

    @staticmethod
    def _exec_produced(exec_result: dict[str, Any], default_rc: int) -> bool:
        """The run's ``results_produced`` fact, or (a record without it) whether its return code is 0."""
        if "results_produced" in exec_result:
            return bool(exec_result["results_produced"])
        return exec_result.get("returncode", default_rc) == 0

    def _improve_skip(self, state: QuestState) -> str | None:
        """Why the improve loop does not run after this run, or ``None`` when it may."""
        limit = int(self.config.engine.improve_rounds or 0)
        if limit <= 0:
            return "engine.improve_rounds is 0"
        if self._agent_mode():
            return "the coding agent wrote this code, and FI does not change it with a chat model"
        if state.get("no_simulation_resolved") or state.get("survey_mode_resolved"):
            return "this quest runs no simulation"
        if _phased.enabled(self.config) and _phased.stage(self.quest_root) in (_phased.CONFIRM, _phased.CONFIRMED):
            return "this is the confirm run, which runs the frozen design as it is"
        if self.config.execution.background_jobs:
            return "the simulation runs as a background job"
        if _optim.has_block(state.get("design")) or _optimise.block_of(self._protocol_block(state)) is not None:
            return "this study is a search for the best design, whose runs FI's search makes itself"
        if self._run_has_no_result(state):
            return "the run did not produce a result"
        if not self._improve_trial_mode(state):
            return ("FI does not run the simulation's trials itself (one script, or a simulation without run_trial / "
                    "run_cell), so it cannot measure the checks between changes")
        used = int(state.get("improve_rounds_used") or 0)
        if used >= limit:
            return f"the {limit} round(s) of engine.improve_rounds were used earlier in this quest"
        return None
