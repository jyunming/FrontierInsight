"""Methods of :class:`core.engine.Engine` for the run, its repair and the plots.

``Engine`` inherits :class:`RunStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, ExecutionResult, Path, QuestState, RepairForAgent, _AgentLines, _FIGURE_SUFFIXES, _FI_MPL_PREAMBLE,
    _SEARCH_REFLECT_ANALYSIS, _SEARCH_REFLECT_SIMULATE, _SPLIT_REFLECT_ANALYSIS, _SPLIT_REFLECT_SIMULATE,
    _STOPPED_BEFORE_THE_STUDY, _analysis_frame, _assertion_violations, _assertion_warnings, _attempts,
    _bounded_seen_after, _changelog, _code_names_replicate_seed, _code_project, _deps_to_warmup_modules,
    _experiment_deps, _extract_result_json, _figdata, _figure_label_found, _figure_overlap_directive,
    _figure_overlap_findings, _format_clarify, _format_reflect_history, _gate_receipt, _in_run, _is_degenerate_result,
    _manifest_repair_budget, _memory, _numeric, _one_line, _optim, _optimise, _optimise_search, _outside_run,
    _own_modules, _parse_implement_response, _phased, _put_back_paper_copies, _read_figure_records,
    _read_json_or_none_path, _read_primary_figures, _record_numbers, _rejudges_oracles_after, _rename_code_names,
    _repair_reply_json, _replicate_assertions, _replicate_env, _replicate_line_figure, _replicate_seed_reaches_rng,
    _restore_primary_figures, _run_manifest, _same_record_numbers, _script_has_random_source,
    _script_reads_replicate_seed, _seeds_only_constants, _skeleton, _skill_mount_plan, _source_text, _split_run,
    _stamp_figure_credit, _strip_outer_fence, _trial_runner, _unread_results, _unseeded_rng_calls, _with_newline, ast,
    asyncio, hashlib, json, os, re, scrub_secrets, shutil, time,
)


class RunStageMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    async def _node_execute(self, state: QuestState) -> QuestState:
        if not self._agent_mode():
            return await self._node_execute_steps(state)
        lines = _AgentLines()
        self._log.addHandler(lines)
        try:
            return await self._node_execute_steps(state)
        except RepairForAgent as asked:
            # A check after the checks before the run asked for a repair of the coding agent's code: a failed run, for the
            # agent's round in the repair step (core/agent_build.py).
            self._log.warning("[execute] %s did not pass; the coding agent is sent what it found", asked.what)
            return self._agent_failure_as_run(asked, lines.lines)
        finally:
            self._log.removeHandler(lines)

    async def _node_execute_steps(self, state: QuestState) -> QuestState:
        # Unconditional tamper check (the re-audit's P1-6): every other frozen-protocol reader
        # (_protocol_of, _run_manifest_problems, the oracle gate) returns early -- skipping
        # _resolved_frozen() entirely -- when ITS OWN check is configured "off", so a hand-edited
        # needs/FROZEN_PROTOCOL.json went completely unvalidated whenever protocol_check,
        # oracle_check and run_manifest_check were all off. This call happens regardless of any of
        # those flags, before the script that would run against that file: it pauses (does not
        # return) if the file was tampered with and restoring it has not been approved. It is also
        # what protects a `--resume` that continues straight into execute without design re-running
        # (design's own tamper check, in `_hold_design_to_frozen`, only fires when design executes).
        self._resolved_frozen()
        recorded = self._cannot_repair_stop()  # the code step ended in a problem FI could not repair: nothing is run
        if recorded is not None:
            return recorded
        # A search for the best design that cannot run (the setup changed since the design, or a resume past it) stops
        # here too: it is never run as a plain script instead (core/optimise.py).
        design_now = state.get("design")
        frozen_block = _optimise.block_of(self._protocol_block(state))
        if frozen_block is not None and not _optim.has_block(design_now):
            # The frozen protocol is a search even when the design in the state no longer says so.
            design_now = {**(design_now if isinstance(design_now, dict) else {}),
                          "protocol": {**((design_now or {}).get("protocol") or {}), "optimisation": frozen_block}}
        self._stop_if_the_search_cannot_start(state, design_now)
        if (search_stop := self._search_cannot_start_stop(state, design_now)) is not None:
            return search_stop
        if (grid_stop := self._grid_too_large_stop(state)) is not None:
            return grid_stop
        if _phased.enabled(self.config):
            # Explore, then confirm: before any code reads the data for the first time, which rows are held back is
            # decided (or, in a research quest that cannot tell which rows belong together, the person is asked).
            question = await asyncio.to_thread(self._phased_before_first_run, state)
            if question:
                self._phased_ask_split(question)
        await self._shadow("execute", state, taken="ran the experiment")
        # Docker sandbox: the selected, approved external skills are mounted
        # read-only in every container this node starts (a thread: resolving
        # the skills can run their self-tests).
        if self.config.execution.sandbox == "docker":
            await asyncio.to_thread(self._mount_selected_skills, state)
        deps = state.get("deps") or []
        usable_skills = self._experiment_skill_states(state)
        skills = [st.skill for st in usable_skills]
        # What the quest environment is given (core/experiment_deps.py): the requested packages that are packages, the
        # selected skills' own packages, and the library skills' folders on the path. What cannot be installed is
        # written as a note the repair steps read before the run's own error.
        install_list, dropped = _experiment_deps.split_deps(
            deps, local_modules=[p.stem for p in (self.quest_root / "code").glob("*.py")],
        )
        for dep, why in dropped:
            self._log.info("[execute] not asking pip for %r: it %s", dep, why)
        # The list is what the model said it would use; the scripts are what runs. A listed package no script uses is
        # not installed, and a well-known one a script imports unlisted is (core/experiment_deps.py:plan_installs).
        # After a run that failed on an import, everything listed is installed: a package the code needs only through
        # another library is then not left out a second time.
        prior = state.get("exec_result") or {}
        import_failed = bool(re.search(r"ModuleNotFoundError|ImportError|No module named",
                                       f"{prior.get('stderr_tail') or ''}\n{prior.get('stdout_tail') or ''}"))
        py = self.executor.python_path(self.quest_root)
        plan = await asyncio.to_thread(
            _experiment_deps.plan_installs, install_list, _experiment_deps.code_sources(self.quest_root / "code"),
            local=_experiment_deps.local_module_names(self.quest_root / "code"),
            python=py if self.config.execution.sandbox == "venv" else None,
            keep=[s.name for s in skills], keep_listed=import_failed,
            skill_files=_experiment_deps.skill_sources(skills),
        )
        install_list = plan.install
        if plan.all_because and install_list:
            self._log.info("[execute] installing every package the model listed: %s", plan.all_because)
        elif import_failed and install_list:
            self._log.info("[execute] installing every package the model listed: the last run failed on an import")
        for asked, new in plan.renamed:
            self._log.info("[execute] the model listed %s, which is what the code imports; installing the package %s",
                           asked, new)
        for dep in plan.unused:
            self._log.info("[execute] the model listed %s but the code does not import it; not installed", dep)
            dropped.append((dep, "the code does not import it, so it was not installed; if the code needs it, "
                                 "import it in the script"))
        for module, pip in plan.added:
            self._log.info("[execute] the code imports %s but the model did not list it; installing %s", module, pip)
        if not getattr(self.config.execution, "shared_interpreter", False) or self.config.execution.sandbox == "docker":
            # A quest's own environment (a clean venv, a container) gets the skills' packages. On FI's shared
            # interpreter they are already there from the skill's approval, and a quest does not change it.
            install_list = list(dict.fromkeys([*install_list, *_experiment_deps.skill_requirements(skills)]))
        failed_installs = await self._install_packages(install_list)
        # What the experiment runs on is the environment after these installs: recorded now, over the start's record.
        await self._record_environment(stage="after installing the experiment's packages")
        # A skill's name that pip could not install (a tool skill, a library skill not published as a package): the
        # repair is told how the skill is used, not to drop an import that may work from the path.
        skill_failures, failed_installs = _experiment_deps.explain_failures(failed_installs, skills)
        dropped = [*dropped, *skill_failures]
        for dep, why in skill_failures:
            self._log.info("[execute] %r could not be installed: it %s", dep, why)
        # A skill whose own declared packages could not be installed cannot be used in this environment: it is dropped
        # from the pick, with a plain line, rather than offered again to every repair and every later step. A failure
        # counts only when it repeats: one dropped connection must not remove a skill from the quest for good.
        wanted_again = [r for sk in skills for r in _experiment_deps.skill_requirements([sk])
                        if r in {dep for dep, _ in failed_installs}]
        if wanted_again:
            still_failing = await self._install_packages(list(dict.fromkeys(wanted_again)))
            recovered = set(wanted_again) - {dep for dep, _ in still_failing}
            if recovered:
                await self._record_environment(stage="after installing the experiment's packages again")
                self._log.info("[skills] installed on the second try: %s", ", ".join(sorted(recovered)))
            failed_installs = [(dep, why) for dep, why in failed_installs if dep not in recovered]
        failed_names = {dep for dep, _ in failed_installs}
        broken_skills = [
            (sk, [r for r in _experiment_deps.skill_requirements([sk]) if r in failed_names]) for sk in skills
        ]
        broken_skills = [(sk, bad) for sk, bad in broken_skills if bad]
        for sk, bad in broken_skills:
            self._log.info("[skills] dropped %s: its package %s could not be installed here", sk.name, ", ".join(bad))
            self._unusable_skills = [*getattr(self, "_unusable_skills", []), sk.name]
        self._packages_note = _experiment_deps.repair_note(
            dropped, failed_installs, getattr(self, "_unusable_skills", []),
        )
        # The warmup below imports what was installed; a name pip could not install would only fail it again.
        not_installed = {dep for dep, _ in (*skill_failures, *failed_installs)}
        deps = [d for d in install_list if d not in not_installed]

        py = self.executor.python_path(self.quest_root)
        pinned = deps
        if self.config.execution.sandbox == "venv":
            try:
                pinned = await asyncio.to_thread(_code_project.pin, py, deps)
            except Exception as exc:  # noqa: BLE001 -- versions in requirements.txt are a convenience
                self._log.warning("[code] could not read the installed versions for requirements.txt: %s", exc)
        _code_project.record_installed(self.quest_root, pinned)
        self._keep_fis_analysis(state)
        code_path = self.quest_root / "code" / "experiment.py"
        # Two scripts (execution.split_analysis): every run of ``experiment.py`` below goes
        # through ``runner``, which runs ``simulate.py`` first unless the raw files it wrote
        # are still good. The seed is read by the simulation, so that is the script the
        # seed checks read.
        simulate_path = self.quest_root / "code" / _split_run.SIMULATE_NAME
        runner: Any = self.executor
        split = self._split_on(state) and simulate_path.is_file()
        if self._split_on(state) and not split:
            self._log.info(
                "[execute] execution.split_analysis is on but this quest has no %s "
                "(the code-writing step returned one script); running %s alone",
                simulate_path.name, code_path.name,
            )
        # The trial contract: simulate.py defines run_trial (or run_cell) and FI runs every trial and keeps their record
        # (core/trial_runner.py). A simulation that still loops by itself is the older contract, whose record is its own.
        trial_entries = _trial_runner.entries(simulate_path) if split else set()
        self._trial_mode = bool(trial_entries & {"run_trial", "run_cell"})
        self._trial_entries = set(trial_entries)
        # A search for the best design (core/optimise.py): FI runs the search and calls the simulation for each design it
        # tries. A simulate.py that defines no function to call is sent back to be repaired by the runner itself, and a
        # reply that held one script is a failed run: neither is ever run as a script that searches by itself.
        searching = self._split_on(state) and (_optim.has_block(state.get("design"))
                                               or _optimise.block_of(self._protocol_block(state)) is not None)
        if searching:
            runner = _optimise.OptimisationRunner(
                self.executor, quest_root=self.quest_root, protocol=lambda: self._protocol_block(state) or {},
                simulate=simulate_path, analysis=code_path, log=self._log,
                # Frozen for its confirm run (or after it): exploration's best design is kept, never searched for again.
                frozen=lambda: _phased.enabled(self.config) and _phased.stage(self.quest_root) in (
                    _phased.CONFIRM, _phased.CONFIRMED),
                # FI's check of the failed trials (core/trial_checks.py) sends a fault back only while a repair is left.
                repairs_left=lambda: (int(state.get("exec_reflect_iter", 0) or 0)
                                      < self.config.engine.exec_reflect_max_iterations
                                      and not state.get("exec_give_up_reason")),
                at_once=_trial_runner.settings_at_once(self.config.execution.settings_at_once),
                # A search whose objective does not move stops early and its fact goes back to the plan, once per quest.
                stop_flat=lambda: self._flat_redesign_allowed(state),
                design=state.get("design"),
            )
        elif split and self._trial_mode:
            runner = _trial_runner.TrialsRunner(
                self.executor, quest_root=self.quest_root, protocol=lambda: self._protocol_block(state) or {},
                deterministic="run_trial" not in trial_entries, simulate=simulate_path, analysis=code_path,
                log=self._log, leave_out=self._left_out_quantities, design=lambda: state.get("design"),
                # FI's checks of the trials (core/trial_checks.py) send a fault back only while a repair is left.
                repairs_left=lambda: (int(state.get("exec_reflect_iter", 0) or 0)
                                      < self.config.engine.exec_reflect_max_iterations
                                      and not state.get("exec_give_up_reason")),
                submit=(self.quest_root / "code" / _trial_runner.SUBMIT_NAME
                        if self.config.execution.background_jobs else None),
                at_once=_trial_runner.settings_at_once(self.config.execution.settings_at_once),
            )
        elif split:
            runner = _split_run.SplitRunner(
                self.executor, quest_root=self.quest_root,
                raw_root=_split_run.raw_root_of(self.quest_root, self.config.execution.raw_dir),
                simulate=simulate_path, analysis=code_path, log=self._log,
            )
        seed_path = simulate_path if split else code_path
        if not (searching or (split and self._trial_mode)):
            # FI's checks of the trials (core/trial_checks.py) run only on trials FI runs: the record says they did not.
            from . import trial_checks as _trial_checks

            _trial_checks.record_not_run(self.quest_root, "own_loop" if split else "one_script")

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
        records_dir = self.fi_dir / "figure_records"
        try:
            from .plot_style import RECORDS_DIRNAME, write_boot

            records_dir = self.fi_dir / RECORDS_DIRNAME
            # The same bootstrap records what each figure draws; a previous
            # version's records go with its figures.
            shutil.rmtree(records_dir, ignore_errors=True)
            boot_dir = write_boot(self.fi_dir, self.config.output.paper_style)
            exec_env = {
                **scrub_secrets(),
                "PYTHONPATH": os.pathsep.join(
                    p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p
                ),
                "FI_FIGURE_RECORDS": str(records_dir),
            }
        except Exception as exc:  # styling must never break execution
            self._log.warning("[execute] plot-style bootstrap skipped: %s", exc)

        # A library skill is imported by the experiment: its folder goes on the path, as its self-test's did.
        lib_paths = [str(p) for p in _experiment_deps.library_paths(skills)]
        if self.config.execution.sandbox == "docker":
            mounts = _skill_mount_plan(self.config, usable_skills).mounts
            lib_paths = [str(mounts[s.name].container) for s in skills if s.name in mounts
                         and str(getattr(s.kind, "value", s.kind)) == "library"]
        if lib_paths:
            base_env = exec_env or scrub_secrets()
            exec_env = {**base_env, "PYTHONPATH": os.pathsep.join(
                p for p in (*lib_paths, base_env.get("PYTHONPATH", "")) if p
            )}
            self._log.info("[execute] library skills on the experiment's path: %s", ", ".join(lib_paths))

        from core.example_inputs import ENV_VAR as _INPUT_ENV, examples_dir, list_inputs

        if list_inputs(self.quest_root):
            exec_env = {**(exec_env or scrub_secrets()), _INPUT_ENV: str(examples_dir(self.quest_root))}
        # The last run before the repairs are spent: a figure function that fails is then drawn from the plan's own description
        # by FI's analysis (core/analysis_figures.py::default_figure) instead of being lost.
        if self._analysis_on() and not self._repair_budget_left(state):
            exec_env = {**(exec_env or scrub_secrets()), "FI_FIGURE_FALLBACK": "1"}
        # The improve loop runs the simulation in the same environment (its skills' paths, its example inputs).
        self._last_exec_env = exec_env

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
        # Only names that are known are imported (the package-name table, or the environment's own record of what
        # each installed package provides): a guessed name that is not a module only printed a frightening traceback.
        warmup_modules = ""
        if self.config.execution.sandbox == "venv" and deps:
            after = await asyncio.to_thread(_experiment_deps.env_packages, py, deps)
            warmup_modules = _deps_to_warmup_modules(deps, after.dists if after else {})
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
                last = next((ln.strip() for ln in reversed((warmup.stderr or "").splitlines()) if ln.strip()), "")
                self._log.warning(
                    "[execute] a test import of the installed packages (%s) failed (exit code %d): %s; "
                    "running the experiment anyway",
                    warmup_modules, warmup.returncode, last[:200] or "no message",
                )

        self._warn_if_plan_diverges()
        # The oracles the plan's protocol declares are answered before anything is run for real (or a quest
        # stops here, with what is missing): the script's own numbers are not evidence that they are right.
        # A plan edited while the quest was stopped is checked before the oracle run spends anything on it.
        await self._settle_plan_sources(state, protocol=self._draft_protocol(state))
        # A number the simulation returns must be computed, never typed into its code (core/typed_results.py): sent back
        # to be computed, a bounded number of times, before any check tests the code; what is still typed in is left out.
        if self._agent_mode():
            # The coding agent fixes what the checks find, round by round, until they pass (core/agent_build.py).
            stop, oracle_code, gate_repaired = await self._agent_build(state, py, exec_env, seed_path)
        else:
            stop, oracle_code, gate_repaired = await self._gate_pass(state, py, exec_env, seed_path)
        if stop is not None:
            return stop
        # A setting of the grid FI's reading finds no use of, once the checks of the code are clean (a misspelt read is the
        # code's fault, found by them): the code sent back when the study compares it, else one line in plan.md and
        # run.log; the grid is always kept (core/run_checks.py).
        if split and self._trial_mode and not searching:
            if (back := self._check_unread_settings(state, simulate_path)) is not None:
                return back
        self._refresh_not_done(state)
        if oracle_code is None and seed_path == code_path and state.get("code"):
            # A repair the gate wrote before an earlier stop (the oracle stop, or the one below) is on disk but never
            # reached the state, and the resumed gate passes without repairing again: the script on disk is what runs,
            # so it is what later repairs must start from.
            try:
                on_disk = code_path.read_text(encoding="utf-8")
            except OSError:
                on_disk = None
            if on_disk and on_disk != state.get("code"):
                oracle_code = on_disk
        # A check's expected value must say where it comes from before the freeze: the gate may have added checks to the
        # plan since it was read, and a person may have edited it (rigor_profile: research stops here until it says).
        await self._settle_plan_sources(state, protocol=self._draft_protocol(state))
        # An oracle the gate had added to the plan is read by the person before the freeze (pauses.plan: ask).
        self._hold_added_oracles()
        # The checks passed on this version of the code: the receipt the run is held to (core/gate_receipt.py). A rewrite
        # after this point that changes what the code computes is checked again before the run, or the run does not start.
        _gate_receipt.write(self.quest_root)
        # The whole run is timed on a few real trials and has its runs per setting lowered by FI when it would not fit the time
        # allowed (core/run_estimate.py); one that still would not fit stops the quest here, with no paper.
        # The run's own environment (its seed and replicate index, the phased stage's seed) is worked out once, before the
        # check that runs the study in shared processes: that check is given exactly this, so its first run can stand for the
        # run. Nothing is recorded for the phased stage until the run starts.
        too_long = await self._size_the_run(state, runner, py, self._first_run_env(exec_env, record=False))
        if too_long is not None:
            return too_long
        # From here on the protocol is what the record says (core/frozen_protocol.py).
        self._freeze_protocol_if_due(state)
        # What `code/study.json` (the project's own `run.py`) will run is what the protocol says, before any trial runs.
        await self._study_follows_protocol(state)
        # A figure is drawn from what the run computed, never from numbers typed into the plotting code: the scripts are
        # read (not run) and sent back to draw it from the saved results, a bounded number of times
        # (core/figure_data_check.py).
        figure_repairs, figure_rewrote = await self._figures_from_results(state)
        # The code can have changed since it was read (a repair by the equation or oracle gate, the figures): the version
        # that will run is read now, with the reading the early call keeps for it (at most code_review.MAX_REVIEWS in a
        # quest). Nothing is repaired here: a repaired version would be one nobody has read.
        await self._code_review_gate(state, final=True)
        # The version that runs is the version the checks were made for: a figure repair may change only the drawing, and
        # anything else that changed the computation since the checks (a person's edit, a rewrite) is checked again here.
        stop, regated, regate_rewrote = await self._verify_gated_code(state, py, exec_env, seed_path)
        if stop is not None:
            return stop
        if regated is not None:
            oracle_code = regated
        gate_repaired = gate_repaired or regate_rewrote

        # Pilot pass: run the experiment small before running it for real.
        #
        # ``execute_reflect`` already repairs a script that CRASHES. What it
        # cannot catch is a script that runs fine and answers the wrong
        # question -- a sweep over the wrong parameter range, a resolution too
        # coarse to show the effect. Today that costs the full timeout to
        # discover, and a researcher would never work that way: they run a
        # cheap version, look at whether the numbers are the right order of
        # magnitude, and only then commit the compute.
        #
        # ``FI_PILOT=1`` is honoured by the generated script the same way
        # ``FI_REPLICATE_SEED`` is (the implement prompt instructs it), so
        # this needs no separate code path in the experiment itself. The
        # pilot's numbers are DISCARDED -- it is a smoke test of the design,
        # not a measurement.
        # No pilot for a background job: the script would submit it.
        if self.config.engine.pilot_run and split:
            self._log.info("[execute] no pilot pass: the simulation and its analysis run as two scripts")
        if self.config.engine.pilot_run and not self.config.execution.background_jobs and not split:
            pilot_timeout = max(
                30, int(self.config.execution.timeout_s * self.config.engine.pilot_timeout_frac)
            )
            self._log.info(
                "[execute] pilot pass (FI_PILOT=1, timeout=%ds) before the "
                "full run", pilot_timeout,
            )
            pilot_env = {**(exec_env or scrub_secrets()), "FI_PILOT": "1"}
            try:
                pilot = await self.executor.execute(
                    [str(py), str(code_path)],
                    cwd=self.quest_root,
                    timeout_s=pilot_timeout,
                    env=pilot_env,
                )
            except Exception as e:  # noqa: BLE001 — a pilot must never abort the quest
                self._log.info("[execute] pilot could not run (%r); continuing", e)
                pilot = None
            if pilot is not None:
                pilot_rj = _extract_result_json(pilot.stdout)
                if pilot.returncode != 0 or pilot_rj is None:
                    # Not fatal: the full run still happens, and if the fault
                    # is real, execute_reflect repairs it there with the
                    # traceback it needs. Saying so early is the value.
                    self._log.warning(
                        "[execute] pilot did not produce a usable RESULT_JSON "
                        "(rc=%s). Continuing to the full run, where "
                        "execute_reflect can repair it. stderr_tail=%s",
                        pilot.returncode, (pilot.stderr or "")[-300:],
                    )
                else:
                    # ``_assertion_violations`` reads ``result_json`` off the
                    # state, so hand it the PILOT's numbers rather than the
                    # real ones (which don't exist yet) via a shallow copy.
                    violations = _assertion_violations(
                        self._in_force({**state, "result_json": pilot_rj}),
                    )
                    if violations:
                        self._log.warning(
                            "[execute] pilot produced out-of-range values: %s. "
                            "The full run proceeds, but this usually means the "
                            "DESIGN is wrong (parameter range, units) rather "
                            "than the code -- worth reading before the results.",
                            "; ".join(str(v) for v in violations[:3]),
                        )
                    else:
                        self._log.info(
                            "[execute] pilot passed: RESULT_JSON parsed and "
                            "within declared ranges; running full scale",
                        )

        # Run from quest_root so figures/ is the relative target. Wrapped in a
        # heartbeat: the experiment subprocess can run for many minutes (a
        # parameter sweep) while the executor just blocks on communicate(),
        # emitting nothing — and the dashboard infers a quest's status from
        # run.log recency, so a long silent run reads as "pending"/idle. The
        # heartbeat keeps that signal fresh.
        # The first run's seed is FI's to choose too. Left unset, it fell back
        # to whatever default the script had written down (42 in a graded
        # quest) while the replicates were handed 1 and 2 -- three bases a few
        # apart, which for a script deriving per-trial seeds as ``base +
        # counter`` means three runs of almost exactly the same trials.
        stride = max(1, int(self.config.engine.replicate_seed_stride))
        primary_env = self._first_run_env(exec_env)  # the environment the size check ran with; the run is counted now
        if _phased.enabled(self.config):
            self._log.info("[phased] %s", "exploration stage: this run may be changed after its results are seen"
                           if _phased.stage(self.quest_root) == _phased.EXPLORE else
                           "confirm stage: running the frozen design on data or seeds exploration never saw")
        run_started = time.time()
        run_mark = _outside_run.write_mark(self.quest_root)  # FI's own count of what the run writes (core/outside_run.py)
        # An earlier pass's results check is not this run's: the analysis this run makes must not read what that pass left out.
        from . import result_checks as _result_checks

        (self.quest_root / "needs" / _result_checks.RECORD).unlink(missing_ok=True)
        try:
            result: ExecutionResult = await self._await_with_heartbeat(
                runner.execute(
                    [str(py), str(code_path)],
                    cwd=self.quest_root,
                    timeout_s=self.config.execution.timeout_s,
                    env=primary_env,
                ),
                label="running simulate.py and experiment.py" if split else "running experiment.py",
                progress=getattr(runner, "progress_text", None),
            )
        except _optimise_search.TooManySettings as exc:
            # A plan that names more settings than FI lists, and no earlier size check met it (a frozen or resumed run): the
            # plain stop with its count, never a crash and never a smaller study than the plan's.
            return self._stop_too_long(str(exc))
        if (unusable := self._stop_for_unusable_search(runner)) is not None:
            # The search this study keeps cannot be used: no script can change that, so no repair and no second try.
            return unusable
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
            result = await runner.execute(
                [str(py), str(code_path)],
                cwd=self.quest_root,
                timeout_s=self.config.execution.timeout_s,
                env=primary_env,
            )
        # Which of the two scripts failed, for the repair; read now, before a replicate
        # runs through the same runner.
        failed_script = runner.failed_script if split else None
        # Whether the results were produced, decided once from this run (rc 0, or only a figure failed: FI's record of it).
        produced = self._run_produced(result)
        result_after_run = result  # the fact is not true of a result a check below replaces (sent back)
        # A run that produced its results but exits 1 for a failed figure is replaced by the figure's repair while repairs are
        # left (the router, ``_repair_budget_left``): the work of a final run waits for the run that ends the loop.
        provisional = produced and result.returncode != 0 and self._repair_budget_left(state)
        # What the simulation says it did, against the frozen protocol (core/run_manifest.py): a run that differs is sent
        # back like a failed one, and after the repairs the quest stops.
        manifest_stop = self.fi_dir / "run_manifest_stop.json"
        if manifest_stop.is_file():  # a resume after a stop starts the repairs again
            manifest_stop.unlink(missing_ok=True)
            manifest_attempts = 0
        else:
            manifest_attempts = int(state.get("run_manifest_failures", 0) or 0)
        from core import job_watch as _job_watch

        _job = _job_watch.job_of(_extract_result_json(result.stdout or ""))
        if _job is not None and _job["status"] == _job_watch.PENDING and result.returncode == 0:
            # A job still running has recorded nothing yet: its record is checked when it is done.
            manifest_status, manifest_found, self._manifest_note = "pending", [], ""
        else:
            run_cell_random = (await self._run_cell_randomness(state, py, primary_env)
                               if split and produced and self.config.engine.run_manifest_check != "off"
                               else None)
            manifest_status, manifest_found = self._run_manifest_problems(state, split, result, run_cell_random)
        # A run that crashed or has not recorded yet says nothing about the record: the repairs already spent stay spent
        # (an analysis repair that crashes must not earn a fresh budget when the crash is fixed).
        manifest_attempts_next = manifest_attempts if (not produced or manifest_status in ("not_run", "pending")) else 0
        manifest_analysis_only = False
        manifest_fis_fault = False  # the difference is in what FI's own analysis printed (core/analysis_frame.py)
        acting: list[str] = []
        unprinted_recorded: list[str] = []  # warn: what the analysis did not print, kept in the record for QUALITY.md
        if manifest_found:
            mode = self.config.engine.run_manifest_check
            # What is acted on: every finding under block; under warn only the numbers FI's trials never produced (they
            # are sent back and stop the quest like under block), the rest being recorded.
            fabricated_found = set(getattr(self, "_manifest_fabricated", []) or [])
            acting = [f for f in manifest_found if f in fabricated_found] if mode == "warn" else list(manifest_found)
            if mode == "warn" and (left_out := self._grid_values_left_out(state)):
                # The analysis names only some of an axis's values (read from its code) and FI's own count shows it reports
                # no trials for them: a fact about the report, not a note. Its findings are acted on like under block.
                self._log.warning("[run_manifest] %s; the analysis is sent back", "; ".join(left_out))
                acting = list(dict.fromkeys([*acting, *(f for f in manifest_found if f in set(self._manifest_analysis_problems))]))
            budget = int(self.config.engine.run_manifest_repair_attempts)
            by_request = self._goes_beyond_protocol_by_request()
            protocol_now = self._protocol_of(state)
            if by_request and protocol_now is not None and self._protocol_drift_not_asked_for(protocol_now):
                by_request = []
            if by_request:
                # A person's request excuses what it added to the protocol, never numbers no trial produced.
                acting = [f for f in manifest_found if f in fabricated_found]
            if by_request and not acting:
                # The person asked for what the frozen protocol does not have; a repair would take it out again.
                manifest_status = "extended_by_person"
                self._log.warning(
                    "[run_manifest] the run differs from the frozen protocol: %s; going on, because the script was "
                    "extended at a person's request (%s); the difference is recorded",
                    "; ".join(manifest_found), "; ".join(by_request),
                )
            elif (acting and set(acting) <= set(self._manifest_analysis_problems) and split
                  and _analysis_frame.is_fis_file(self.quest_root / "code" / _analysis_frame.EXPERIMENT_FILE)):
                # FI wrote the analysis (core/analysis_frame.py): what it prints of the trials is FI's own table, so a
                # difference only in what the analysis printed is a fault of FI's. No function of the study's could repair it:
                # it is recorded (the paper and QUALITY.md say it) and the quest goes on.
                manifest_status, manifest_fis_fault = "warned", True
                self._log.warning("[run_manifest] the results FI's own analysis printed differ from the record: %s; a fault of "
                                  "FI's, not of the study's code: recorded, not sent back", "; ".join(acting))
            elif mode == "warn" and not acting:
                # What the analysis does not print of trials FI holds whole is not a difference in the record: the log has
                # it, the paper (core/paper_facts.py) and the quality report do not.
                unprinted = [f for f in manifest_found if f in set(self._manifest_unprinted)]
                manifest_found = [f for f in manifest_found if f not in unprinted]
                unprinted_recorded = unprinted
                manifest_status = "warned" if manifest_found else "ok"
                self._log.warning("[run_manifest] the run differs from the frozen protocol: %s; going on (engine.run_manifest_check: warn)", "; ".join([*manifest_found, *unprinted]))
            elif manifest_attempts < _manifest_repair_budget(
                    budget, set(acting) <= set(self._manifest_analysis_problems)):
                manifest_status = "repairing"
                manifest_attempts_next = manifest_attempts + 1
                # Only what the analysis failed to print is the analysis's to fix; anything else (the simulation swept
                # another grid, a claimed count its own values do not back) sends simulate.py back, as it always did.
                analysis_only = set(acting) <= set(self._manifest_analysis_problems)
                budget = _manifest_repair_budget(budget, analysis_only)
                manifest_analysis_only = analysis_only
                to_fix = _split_run.ANALYSIS_NAME if analysis_only else _split_run.SIMULATE_NAME
                self._log.warning(
                    "[run_manifest] the run differs from the frozen protocol (%d): %s; sending %s back (%d of %d)",
                    len(acting), "; ".join(acting), to_fix, manifest_attempts_next, budget,
                )
                # The rejected run's numbers must not reach the analysis if the repair gives up: no RESULT_JSON line, no figures.
                kept = "\n".join(line for line in (result.stdout or "").splitlines() if not line.startswith("RESULT_JSON:"))
                why = (
                    _run_manifest.analysis_directive(acting, self._manifest_differences) if analysis_only
                    else _run_manifest.directive(acting, self._protocol_block(state) or {},
                                                 # research asks the older contract for run_trial as well
                                                 trial_mode=bool(getattr(self, "_trial_mode", False))
                                                 or self.config.rigor_profile == "research")
                )
                result = ExecutionResult(1, kept, (result.stderr or "") + "\n[FI] " + why, result.duration_s, result.timed_out)
                failed_script = to_fix
                self._clear_stale_figures()
            else:
                manifest_status = "stopped"
        # Under warn the difference goes on into the paper, which must say it (core/paper_facts.py): in plain words when
        # FI counted it (the analysis lists more, or leaves trials out), else its first finding.
        warned_headline = (_run_manifest.plain_difference(list(getattr(self, "_manifest_differences", None) or []))
                           if manifest_status == "warned" else "")
        self._run_manifest_record({
            "status": manifest_status, "problems": manifest_found, "attempts": manifest_attempts_next or manifest_attempts,
            "failed_trials": getattr(self, "_manifest_failed_trials", 0) or 0,
            **({"unprinted": list(unprinted_recorded)} if unprinted_recorded else {}),
            **({"note": self._manifest_note} if getattr(self, "_manifest_note", "") else {}),
            **({"headline": warned_headline} if warned_headline else {}),
            **({"fault": "FI's own analysis"} if manifest_fis_fault else {}),
        })
        # The run's data is kept only once the run is accepted: not a crash, a run the manifest sent back or stopped,
        # nor a job still pending (its raw files are half written).
        run_accepted = produced and not provisional and manifest_status not in ("stopped", "pending", "repairing")
        if run_accepted:
            await asyncio.to_thread(self._save_run_data, run_started, split,
                                    wrote=_outside_run.written_since(self.quest_root, run_mark))
            if split and self.config.rigor_profile == "research":
                await self._hidden_check(state, py, primary_env)  # the checks at a setting the code never saw
            if split:
                await self._refinement_check(state, py, primary_env)  # the headline results at finer numerical settings
            # Are the headline results computed with the model's functions FI tested? On the code as it ran.
            await self._dependence_check(state, primary_env)
        if manifest_status == "stopped":
            try:
                plain_stop = (_run_manifest.plain_difference(self._manifest_differences)
                              if set(acting) <= set(self._manifest_analysis_problems) else "")
                manifest_stop.write_text(json.dumps({
                    "problems": acting, **({"headline": plain_stop, "differences": self._manifest_differences}
                                                   if plain_stop else {})}) + "\n", encoding="utf-8")
            except OSError:
                pass
            # The rejected run's figures go, as when it is sent back: a stopped quest shows none of its output.
            self._clear_stale_figures()
            return self._manifest_stop(acting, manifest_attempts_next or manifest_attempts)
        # What the run reports is compared with what ran; a fault is sent back like a failed run, and what stays is left out
        # (core/result_checks.py).
        ran_with_results = produced
        result_before_check = result
        result, failed_script = await self._results_match_run(state, result, failed_script, split)
        result_check_sent_back = ran_with_results and result is not result_before_check
        figures = sorted(
            p.name for p in (self.quest_root / "figures").iterdir()
            if p.is_file() and p.suffix.lower() in _FIGURE_SUFFIXES
        ) if (self.quest_root / "figures").is_dir() else []
        # What the primary run's figures say of themselves; the replicates below write derived.json again.
        primary_record = self._primary_figure_record = _analysis_frame.read_record(self.quest_root)
        raw_result_json = _extract_result_json(result.stdout)
        self._note_typed_values(raw_result_json)
        result_json = self._without_typed_results(raw_result_json)
        self._log.info(
            "[execute] rc=%d duration=%.1fs figures=%d result_json=%s",
            result.returncode, result.duration_s, len(figures), bool(result_json),
        )

        # A background job (HPC, a cluster): the script submitted it, or checked
        # it, and says the job has not finished. That is not a failure and there
        # is nothing to repair. The quest pauses, exits cleanly, and a resume (or
        # `--watch`) runs this node again, which runs the same idempotent script.
        from core import job_watch

        job = job_watch.job_of(result_json)
        if job is not None and job["status"] == job_watch.PENDING and result.returncode == 0:
            self._wait_for_job(job, code_path)
        elif job_watch.clear_pending(self.fi_dir):
            self._log.info("[execute] the background job has finished; using its results")

        # Multi-seed replication. Only fires when (a) the primary run
        # succeeded — we don't want to spend N×cost re-running a script
        # that's already broken — and (b) ``execute_replicates > 1``.
        # The first run we just did counts as seed 0; we now run N-1
        # MORE with seeds 1..N-1 via ``FI_REPLICATE_SEED``. Failures
        # in any individual replicate are recorded but don't abort the
        # rest — better to have N-1 good replicates than zero. The
        # primary ``result_json`` carries seed 0 so downstream
        # single-seed code paths are unchanged.
        # Replicates would submit the job again, once per seed.
        replicates_n = (
            1 if self.config.execution.background_jobs or getattr(self, "_trial_mode", False) or searching
            else max(1, int(self.config.engine.execute_replicates))
        )  # under the trial contract runs_per_setting is every trial of a setting, each with its own seed; a search
        # for the best design is run once, within its budget
        result_json_replicates: list[dict[str, Any]] = []
        deterministic = False
        # Whether the script can respond to the seed at all, whether it hands
        # any of its randomness to a generator the seed cannot reach, and
        # (distinct from just naming the variable) whether that value
        # actually flows into a call that seeds a generator -- a script can
        # name FI_REPLICATE_SEED, read it into an unused variable, and still
        # seed every generator from a hardcoded constant; that is not
        # "responding to the seed" even though it mentions it. All three read
        # once from the source that is about to run.
        reads_seed = _script_reads_replicate_seed(seed_path)
        unseeded_rng = _unseeded_rng_calls(seed_path)
        seed_reaches_rng = _replicate_seed_reaches_rng(seed_path)
        seed_ignored = False
        no_random_source = not _script_has_random_source(seed_path)
        replicates_ran = False
        all_agreed = True  # every replicate that finished gave seed 0's numbers
        primary_figures: dict[str, tuple[bytes, bytes | None, dict[str, bytes]]] = {}
        if produced and result_json is not None:
            # Tag seed 0 explicitly so the aggregator can attribute it.
            result_json_replicates.append({"_seed": 0, **result_json})

        # Error bars are worth paying for only on a result that is going to
        # survive. ``execute_reflect`` judges plausibility AFTER this node, so
        # replicating first meant every futile repair iteration cost three runs
        # instead of one -- measured on a real quest, where a diverging
        # integrator tripped the gate three times over. With no repair attempt
        # left, though, nothing regenerates the result: it is the one the paper
        # is written from, so it still gets its error bars.
        gate_violations = (
            _assertion_violations(self._in_force({
                **state, "result_json": result_json,
                "manifest_failed_trials": getattr(self, "_manifest_failed_trials", None),
            }), self._walk_memo())
            if produced and result_json is not None
            else []
        )
        repair_left = self._repair_budget_left(state)
        skip_replicates = (bool(gate_violations) and repair_left) or provisional
        if provisional and replicates_n > 1:
            self._log.info("[execute] skipping %d replicate(s): a figure failed and its repair is coming, so this run is replaced",
                           replicates_n - 1)
        elif skip_replicates and replicates_n > 1:
            self._log.info(
                "[execute] skipping %d replicate(s): the primary result already "
                "violates %d assertion(s), so execute_reflect is about to "
                "regenerate it",
                replicates_n - 1, len(gate_violations),
            )
        elif gate_violations and replicates_n > 1:
            self._log.info(
                "[execute] the result violates %d assertion(s) and no repair "
                "attempt is left, so it is the one written up; replicating it",
                len(gate_violations),
            )
        if (
            replicates_n > 1
            and produced
            and result_json is not None
            and not skip_replicates
        ):
            self._log.info(
                "[execute] replicating: %d additional seeds (1..%d)",
                replicates_n - 1, replicates_n - 1,
            )
            # Every run draws into the same figures/, so the replicates draw
            # over the primary run's figures, a replicate that crashes halfway
            # included. Kept here, a figure the seeds do not redraw as their
            # mean goes back to seed 0's, with seed 0's record of what it draws.
            primary_figures = _read_primary_figures(self.quest_root / "figures", records_dir, figures)
            for seed in range(1, replicates_n):
                replicates_ran = True
                rep_env = _replicate_env(exec_env, seed, stride)
                if _phased.enabled(self.config):
                    rep_env = _phased.seed_env(self.quest_root, rep_env)
                rep_result = await runner.execute(
                    [str(py), str(code_path)],
                    cwd=self.quest_root,
                    timeout_s=self.config.execution.timeout_s,
                    env=rep_env,
                )
                rep_rj = self._without_typed_results(_extract_result_json(rep_result.stdout))
                if self._run_produced(rep_result) and rep_rj is not None:
                    result_json_replicates.append({"_seed": seed, **rep_rj})
                    self._log.info(
                        "[execute] replicate seed=%d rc=0 duration=%.1fs",
                        seed, rep_result.duration_s,
                    )
                    # The one case where two agreeing runs (seed 0 and the first
                    # replicate that finished) may end the
                    # replication: the script (and the modules it imports from
                    # its own folder) has no source of random numbers at all,
                    # so a repeat can only return the same numbers. A script
                    # that does draw random numbers can agree on two seeds by
                    # chance -- a yes/no outcome, a rare event, a count, a
                    # rounded metric -- and differ on the third, so it runs
                    # every seed the quest asked for and is classified below
                    # from all of them. (A run_cell simulation never gets here:
                    # FI runs it once per setting itself.)
                    if rep_rj != result_json:
                        all_agreed = False
                    elif no_random_source and all_agreed:  # the first replicate that finished agreed
                        seed_ignored = True
                        self._log.info(
                            "[execute] %s draws no random numbers, so this study is deterministic: one run "
                            "is the whole result, and its trust comes from convergence, conservation and "
                            "analytic-limit checks rather than repeat counts. Skipping the remaining %d "
                            "replicate(s).",
                            seed_path.name, replicates_n - 1 - seed,
                        )
                        break
                else:
                    self._log.warning(
                        "[execute] replicate seed=%d FAILED rc=%d duration=%.1fs "
                        "(skipping in aggregate)",
                        seed, rep_result.returncode, rep_result.duration_s,
                    )

        # Every seed ran and every one gave seed 0's numbers. That has two
        # causes that are not the same fact about the experiment, and calling
        # both "deterministic" is what let a hardcoded seed reach a paper. A
        # script whose seed reaches its generators, or that draws from OS
        # entropy, produced independent samples that agreed: they stay the
        # replicates, reported as what these seeds showed. A script whose
        # randomness the seed never reaches (it never reads FI_REPLICATE_SEED,
        # or reads it into a name no generator is seeded from) ran ONE run
        # repeated -- not samples anything may be averaged over. Its own source
        # says which case this is.
        # The modules the script imports from its own folder count too: a multi-module project reads the seed, or
        # builds a generator without one, in a helper (a mention in a comment is not a read).
        own_modules = _own_modules(seed_path) if len(result_json_replicates) > 1 and not seed_ignored else []
        helpers = own_modules[1:]
        helper_reads = any(_code_names_replicate_seed(p) for p in helpers)
        helper_unseeded = [(f"{p.name} line {line}", expr) for p in helpers for line, expr in _unseeded_rng_calls(p)]
        if replicates_ran and not seed_ignored and all_agreed and len(result_json_replicates) > 1:
            n_runs = len(result_json_replicates)
            if (
                (reads_seed and seed_reaches_rng) or unseeded_rng or helper_unseeded
                # A helper reads the seed: unless every generator is seeded with a fixed number, it may reach them.
                or (not reads_seed and helper_reads and not _seeds_only_constants(own_modules))
            ):
                deterministic = True
                self._log.info(
                    "[execute] all %d seeds gave the same numbers. %s can draw random numbers, so this is what "
                    "these seeds showed, not proof that other seeds would give the same.",
                    n_runs, seed_path.name,
                )
            elif reads_seed:
                # Named but not obeyed: it reads FI_REPLICATE_SEED (so it is not the
                # "never reads it" case below), but that value reaches no generator this
                # script seeds -- a hardcoded constant, or a value read into a name that
                # is never used. It is one run repeated, exactly like the case below.
                seed_ignored = True
                self._log.warning(
                    "[execute] all %d runs produced identical results, and although "
                    "%s reads FI_REPLICATE_SEED, that value never reaches a generator it "
                    "seeds -- so this is one run repeated, not %d samples. No aggregate, "
                    "standard error or confidence interval will be reported over them: "
                    "the quest stands on a single measurement.",
                    n_runs, seed_path.name, n_runs,
                )
            else:
                seed_ignored = True
                self._log.warning(
                    "[execute] all %d runs produced identical results "
                    "and %s never reads FI_REPLICATE_SEED, so these are "
                    "one run repeated, not %d samples. No aggregate, "
                    "standard error or confidence interval will be "
                    "reported over them: the quest stands on a single "
                    "measurement.",
                    n_runs, seed_path.name, n_runs,
                )

        # A line figure drawn by one seed shows that run's noise. With the
        # seeds in hand it is drawn again as their mean, shaded with its 95%
        # confidence interval, and so is one of lines with error bars, whose
        # bars the interval over the seeds replaces. A bar chart, a histogram
        # or any other figure goes back to the primary run's (seed 0), and its
        # record says it shows that one run, so the text can quote seed 0's
        # values for it rather than the means.
        # The seed did not reach the randomness yet the runs differ: they are
        # drawing from OS entropy, either because the script never reads the
        # seed or because it reads it and builds a generator without one
        # anyway. Those replicates ARE independent samples, so the aggregate
        # stands -- but nothing about the run is reproducible, and a rerun will
        # not land on these numbers.
        if (
            ((not reads_seed and not helper_reads) or unseeded_rng or helper_unseeded)
            and not seed_ignored and len(result_json_replicates) > 1
        ):
            found = [(f"line {line}", expr) for line, expr in unseeded_rng] + helper_unseeded
            self._log.warning(
                "[execute] %s draws from OS entropy (%s), so its replicates are "
                "independent samples but not reproducible: a rerun of these seeds "
                "will not land on these numbers",
                seed_path.name,
                "; ".join(f"{where}: {expr}" for where, expr in found[:6])
                if found else "it never reads FI_REPLICATE_SEED",
            )
        if replicates_ran and split and run_accepted:
            await asyncio.to_thread(self._save_run_data, run_started, split, root_files=False)
        replotted: dict[str, int] = {}
        # A figure drawn as "the mean of the seeds" over replicates that are one
        # run repeated would be a mean of one number, captioned as several.
        redraw = len(result_json_replicates) > 1 and not deterministic and not seed_ignored
        if redraw:
            replotted = await self._replot_replicate_figures(
                figures, records_dir, [int(r["_seed"]) for r in result_json_replicates],
                python=py, env=exec_env, assertions=_replicate_assertions(state),
            )
        if replicates_ran:
            restored = _restore_primary_figures(
                self.quest_root / "figures", records_dir, primary_figures, redrawn=replotted,
            )
            if restored:
                self._log.info(
                    "[execute] %d figure(s) not drawn as the mean of the seeds show seed 0: %s",
                    len(restored), ", ".join(restored),
                )
        figures = self._drop_typed_figures(figures, records_dir)
        figure_records = _read_figure_records(records_dir, figures)
        self._figure_fallbacks(figure_records, primary_record)
        for line in _unread_results(figure_records):
            self._log.info("[execute] declared result not read by a figure: %s (the caption states only what the saved axes "
                           "and legend label)", line)
        for name, n_seeds in replotted.items():
            if name in figure_records:
                figure_records[name]["replicate_mean"] = {"n": n_seeds}
        if redraw:
            for name in figures:
                if name not in replotted and name in figure_records:
                    figure_records[name]["single_seed"] = 0
        self._record_no_difference(getattr(runner, "no_difference", None) if produced and not result_check_sent_back else None)
        self._keep_raw_stdout(result.stdout)
        self._keep_analysis_answers(
            result_json, state,
            result_json_replicates if (len(result_json_replicates) > 1 and not seed_ignored and not searching
                                       and not getattr(self, "_trial_mode", False)) else None,
            replicates_n if (replicates_ran and not seed_ignored and not searching
                             and not getattr(self, "_trial_mode", False)) else 1)
        patch: dict[str, Any] = {
            "exec_result": {
                "returncode": result.returncode,
                "duration_s": result.duration_s,
                "timed_out": result.timed_out,
                # What the models read of the run's output: without its RESULT_JSON line and without any line that names a
                # quantity left out of the results (core/run_checks.py). The raw output is kept in .fi/last_run_stdout.txt.
                "stdout_tail": self._stdout_for_models(result.stdout)[-2000:],
                "stderr_tail": self._stderr_for_models(result.stderr)[-2000:],
                # What could not be installed (core/experiment_deps.py), kept apart from the run's own stderr so the
                # analysis does not read it as an error and no slice of the stderr cuts the traceback's last line.
                # The repair step reads it first, on every run: one that exits 0 on a caught ImportError still goes
                # there as degenerate.
                "packages_note": getattr(self, "_packages_note", ""),
                # The script a two-script quest's failure is in ("simulate.py" or
                # "experiment.py"): the repair rewrites that one. None for one script.
                "failed_script": failed_script,
                # Set when the run was sent back because the simulation's numbers did not depend on what it was given
                # (core/flat_output.py): the honest stop after the repairs says so in those words.
                "flat_output": (str(getattr(runner, "flat", None) or "") if result.returncode != 0 else ""),
                # A fault FI's checks of the trials found (core/trial_checks.py): the stop once the repairs are spent (a reported
                # uncertainty measured not to match is not one: the run goes on, DISCLOSE).
                "trial_checks": (dict(getattr(runner, "trial_fault", None) or {})
                                 if result.returncode != 0 and not produced else {}),
                # The search's objective did not move across the design space, and the plan may still be revised for it
                # (core/optimise.py): the gate sends what FI measured back to the plan.
                "flat_objective": (dict(getattr(runner, "flat_objective", None) or {}) if result.returncode != 0 else {}),
                "numeric_warnings": self._scan_numeric_warnings(state, result.stderr, result_json),
                # The run finished with results and the results check sent it back (core/result_checks.py): a repair
                # that gives up runs the same code once more, its results kept. Held on this run's own record, so it
                # never outlives the run.
                "result_check_sent_back": result_check_sent_back,
                # Whether the run produced its results (rc 0, or only a figure failed): every reader that means "the run
                # produced its results" reads this, not the return code.
                "results_produced": bool(produced and result is result_after_run and not result_check_sent_back),
                # FI's analysis (core/analysis_frame.py): what it was doing when it stopped (a function of derive.py, or FI's
                # own step), read from its whole stderr, and the function of derive.py a check after the run named.
                **({"analysis_step": _analysis_frame.running_step(result.stderr or "", self.quest_root)}
                   if result.returncode != 0 and self._analysis_on() else {}),
                **({"failed_body": self._analysis_sendback_body}
                   if result.returncode != 0 and getattr(self, "_analysis_sendback_body", "") else {}),
            },
            "figures": figures,
            "figure_records": figure_records,
            "result_json": result_json or {},
            "exec_patch_pending": False,
        }
        if _phased.enabled(self.config):
            # Explore, then confirm: the result the confirm run produced is the only one the gate may record.
            _phased.note_confirm_result(self.quest_root, patch["result_json"])
        if broken_skills:
            gone = {sk.name for sk, _ in broken_skills}
            selection = dict(state.get("skill_selection") or {})
            selection["dropped"] = [
                *[d for d in self._dropped_skills(state) if d.get("name") not in gone],
                *[{"name": sk.name, "kind": _experiment_deps._kind(sk),
                   "why": f"its package {', '.join(bad)} could not be installed here"} for sk, bad in broken_skills],
            ]
            patch["selected_skills"] = [n for n in state.get("selected_skills") or [] if n not in gone]
            patch["skill_selection"] = selection
        patch["numeric_warnings_accepted"] = False
        said_edges = []
        for w in _assertion_warnings(self._in_force({**state, **patch}), self._walk_memo()):
            self._log.warning("[plausibility] warning, not a stop: %s", w.describe())
            if getattr(w, "kind", "") in ("edge_everywhere", "at_bound"):
                said_edges.append(w.describe() if w.kind == "edge_everywhere"
                                  else {"text": w.disclosure(), "short": w.disclosure_short()})
            self._audit("check_result", check="plausibility_unproven_cap", status="warn",
                        summary=w.describe()[:300], path=getattr(w, "path", ""), value=getattr(w, "value", None))
        try:  # what the run record says of this run's results that sit on their bound everywhere
            edge_path = self.fi_dir / "edge_everywhere.json"
            if said_edges:
                edge_path.parent.mkdir(parents=True, exist_ok=True)
                edge_path.write_text(json.dumps({"results": said_edges}, indent=2) + "\n", encoding="utf-8")
            else:
                edge_path.unlink(missing_ok=True)
        except OSError:
            pass
        patch["run_manifest_failures"] = manifest_attempts_next
        patch["run_manifest_analysis_only"] = manifest_analysis_only
        if _rejudges_oracles_after(manifest_status, failed_script):
            await self._rejudge_oracles_on_delivered_run(state, seed_path)
        run_context = await self._attempt_context({**state, **patch}, kind="after_run")

        def _run_record() -> dict[str, Any] | None:
            try:
                oracle = json.loads((self.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
                oracle_status = str((oracle or {}).get("status") or "") if isinstance(oracle, dict) else ""
            except (OSError, ValueError):
                oracle_status = ""
            outcome = _attempts.run_outcome(
                returncode=result.returncode, has_result=bool(result_json), manifest_status=str(manifest_status or ""),
                oracle_status=oracle_status,
            )
            # The one signature of this run: the record keeps it and a repair of this run reads it back, so both are
            # made from the same inputs (the last 2000 characters of stderr, the return code, the time limit, the check).
            signature = _memory.failure_signature(
                returncode=result.returncode, stderr=str(result.stderr or "")[-2000:],
                timed_out=bool(getattr(result, "timed_out", False)),
                pause=outcome if outcome in ("protocol_mismatch", "oracle_failure") else None)
            self._last_run_signature = signature
            if outcome is None:
                return None
            frozen = _attempts._read_json(self.quest_root / "needs" / "FROZEN_PROTOCOL.json")
            return {
                "kind": "run", "outcome": outcome, "returncode": result.returncode,
                "failure_signature": signature,
                "scripts": dict(run_context.get("code") or {}),
                # What it came from: the frozen protocol's run, and the design revision or repair recorded last.
                "run_id": frozen.get("run_id") if isinstance(frozen, dict) else None,
                "parent_id": (getattr(self, "_last_ledger_id", None)
                              or _attempts.last_id(self.fi_dir, _attempts.LEDGER, ("design", "repair"))),
                "oracle_status": oracle_status or None, "manifest_status": manifest_status or None,
                "answered_by": self._answering_model(), "context": run_context,
            }
        run_record_id = self._record(_attempts.ATTEMPTS, _run_record)
        if run_record_id:
            self._last_run_record_id = run_record_id
            self.__dict__.setdefault("_run_record_ids", []).append(run_record_id)
        patch["manifest_failed_trials"] = getattr(self, "_manifest_failed_trials", None)
        if result_json:
            patch["bounded_seen"] = _bounded_seen_after(state, result_json, self._walk_memo())
        self._check_replicate_manifests(state, split, replicates_n)
        if oracle_code is None and getattr(self, "_review_experiment_changed", False) and code_path.is_file():
            oracle_code = code_path.read_text(encoding="utf-8")  # a repair of the second model's findings
        self._review_experiment_changed = False
        if oracle_code is not None:
            patch["code"] = oracle_code  # a repair of the script made by the oracle gate
        patch["figure_data_repairs"] = figure_repairs
        if (sized_design := self._design_after_sizing(state)) is not None:
            patch["design"] = sized_design  # the plan's smaller run (core/run_estimate.py)
        if figure_rewrote and code_path.is_file():
            patch["code"] = code_path.read_text(encoding="utf-8")  # the plotting code was rewritten to use the results
        await self._record_criteria(
            state, attempt=run_record_id, result=patch["result_json"] if run_accepted else None,
            repaired=gate_repaired or _numeric.repairs_made(state) > 0)
        # Only populate ``result_json_replicates`` when replication
        # actually ran AND produced more than the primary entry. This
        # keeps the field absent on default single-seed quests so
        # downstream code can use ``state.get("result_json_replicates")``
        # as a "did we run multi-seed" sentinel.
        #
        # Replicates a script produced without ever reading the seed are not
        # published at all. Leaving them on the state is what let "the mean over
        # 3 replicate seeds" and a 95% CI be written about a single run: with
        # the field absent ``_replicate_seed_count`` returns None, and every
        # downstream path -- the analyze aggregate, the figure captions, the
        # writer's "seed 0 of N" language, the number and claim checks -- falls
        # back to its honest single-run behaviour on its own.
        # A deterministic simulation (run_cell) ran once per setting: there is one measurement per setting and no trials
        # to pool, so no interval or replicate statistics are computed for it.
        # A search's result is one best design, not trials to pool (core/optimise.py).
        patch["result_json_trials"] = bool(getattr(self, "_trial_mode", False)) and bool(result_json) and "run_trial" in (
            getattr(self, "_trial_entries", None) or set()) and not searching
        # Written on every pass (set below only when this pass's seeds showed it): a pass whose extra seeds did not run
        # must not keep an earlier script's "every seed agreed".
        patch["result_json_deterministic"] = False
        if patch["result_json_trials"]:
            # The trial contract: the one result holds every trial of every setting FI ran, each with its own seed; the
            # intervals, precision targets and metric statistics are computed from it (pooled counts and values).
            patch["result_json_replicates"] = [{"_seed": 0, **result_json}]
        elif getattr(self, "_trial_mode", False) or searching:
            # No result this pass: the previous pass's pooled trials must not stand in for it.
            patch["result_json_replicates"] = []
        elif len(result_json_replicates) > 1 and not seed_ignored:
            patch["result_json_replicates"] = result_json_replicates
            # Lets ``analyze`` say "every seed agreed" instead of reporting an
            # empty aggregate, which reads like the aggregator broke.
            patch["result_json_deterministic"] = deterministic
            # This node runs again on a repair and on a re_experiment, and
            # these are last-value channels: a key a pass leaves out keeps the
            # PREVIOUS pass's value. Clearing the flag stops a quest whose
            # earlier script ignored the seed from carrying "there is one
            # measurement here" alongside this script's full aggregate.
            patch["result_json_replicate_seed_ignored"] = False
        # Both flags are written on every pass: a pass that ran no replicates (every extra run failed, trial mode,
        # a background job) must not inherit the previous script's verdict.
        patch["result_json_no_random_source"] = bool(seed_ignored and no_random_source)
        patch["result_json_replicate_seed_ignored"] = bool(seed_ignored)
        if seed_ignored:
            # The same hazard the other way round. An earlier pass may have
            # left a replicate list on the state, and merely withholding the
            # key would KEEP it -- so analyze would aggregate the previous
            # script's seeds against this script's result. An empty list is
            # what every consumer already reads as "no replication", and
            # ``_replicate_seed_count`` returns None for it.
            patch["result_json_replicates"] = []
            patch["result_json_deterministic"] = False
        return patch

    async def _node_replot_layout(self, state: QuestState) -> QuestState:
        """A person's refine that only asks for the figures to be arranged or drawn differently: the named figures are
        drawn again from the numbers the run saved, and the experiment is not run again. A figure the script does not
        write is left as it was, and so is every figure when the script fails. The model is shown what each saved file
        holds (``core.data_shape``); a script that fails, or changes no figure, is written once more with its error, and
        when that fails too the person is told at the review that their request was not applied.

        The redraw is recorded and checked as the run's own figures are: the plot-style recorder writes each figure's
        record again, a script that draws numbers typed into it fails its try, and so does one that draws other
        numbers than a figure's record held for a figure FI asked to rename; the figures, their paper copies and their
        records are then put back. A figure drawn as the mean of the seeds is never sent to the model, which cannot see
        each seed's numbers: FI renames its text itself (:meth:`_rename_mean_figures`). ``figure_records`` comes back
        read again, so every later check reads what the figures now show, and ``code/replot_layout.py`` is removed, so
        the next run's typed-number scan never reads it."""
        try:
            return await self._replot_layout(state)
        finally:
            (self.quest_root / "code" / "replot_layout.py").unlink(missing_ok=True)

    async def _replot_layout(self, state: QuestState) -> QuestState:
        """:meth:`_node_replot_layout`'s work."""
        notes = [str(p).strip() for p in state.get("refine_layout") or [] if str(p).strip()]
        done: QuestState = {"refine_layout": [], "layout_missed": [], "layout_not_redrawn": []}
        self._restore_layout_backup()
        from .figure_labels import MARK as _LABEL_MARK
        from .plot_style import RECORDS_DIRNAME, write_boot

        records_dir = self.fi_dir / RECORDS_DIRNAME
        old_records = {str(k): v for k, v in (state.get("figure_records") or {}).items() if isinstance(v, dict)}
        means = {f for f, r in old_records.items() if r.get("replicate_mean")}

        def about(note: str) -> str | None:
            # The figure one of FI's own notes is about (``figures/<name>: ...``); None for a person's note.
            if _LABEL_MARK not in note or not note.startswith("figures/"):
                return None
            return note[len("figures/"):].split(":", 1)[0].strip()

        # A figure drawn as the mean of the seeds: the model sees only the saved files, never each seed's numbers, so it
        # could only draw other numbers under the figure's caption; FI renames it from its own numbers instead.
        mean_notes = [n for n in notes if about(n) in means]
        notes = [n for n in notes if n not in mean_notes]
        # A PDF saved again with nothing changed has the same bytes only with a fixed creation date; without it, a
        # redraw that changed nothing would look like one that did. The recorder writes each figure's record again; a
        # redrawn figure is no seed's, so the seeds' own lines (kept for the mean of the seeds) are left as they are.
        env: dict[str, str] = {**scrub_secrets(), "SOURCE_DATE_EPOCH": "0"}
        recorded = False
        try:
            env = {**env, "FI_FIGURE_RECORDS": str(records_dir), "FI_REPLOT": "1", "PYTHONPATH": os.pathsep.join(
                p for p in (str(write_boot(self.fi_dir, self.config.output.paper_style)), os.environ.get("PYTHONPATH", ""))
                if p)}
            recorded = True
        except Exception as exc:  # styling must never stop a redraw
            self._log.warning("[replot_layout] plot-style bootstrap skipped: %s", exc)
        if notes:
            done = {**done, **await self._replot_by_model(
                notes, env, recorded=recorded, records_dir=records_dir, old_records=old_records, means=means,
                renamed={f for f in (about(n) for n in notes) if f})}
        if mean_notes:
            await self._rename_mean_figures(state, sorted({str(about(n)) for n in mean_notes}), env)
        fresh = _read_figure_records(records_dir, list(state.get("figures") or []))
        records: dict[str, Any] = {}
        for name in [str(f) for f in state.get("figures") or []]:
            record = fresh.get(name) or old_records.get(name)
            if not isinstance(record, dict):
                continue
            # Which seeds a figure shows is the engine's to say, not the record's: a redraw changes its text, not that.
            record = {**record, **{k: v for k, v in (old_records.get(name) or {}).items()
                                   if k in ("replicate_mean", "single_seed", "declared")}}
            records[name] = record
        if records or old_records:
            done["figure_records"] = records
        return done

    async def _replot_by_model(
        self, notes: list[str], env: dict[str, str], *, recorded: bool, records_dir: Path,
        old_records: dict[str, Any], means: set[str], renamed: set[str],
    ) -> QuestState:
        """The model's redraw of the figures ``notes`` ask for (:meth:`_node_replot_layout`). ``renamed`` are the
        figures FI's own notes ask to rename: each must draw the numbers its record held."""
        done: QuestState = {"refine_layout": [], "layout_missed": [], "layout_not_redrawn": []}
        fig_dir = self.quest_root / "figures"
        figures = sorted(f.name for f in fig_dir.iterdir() if f.is_file() and f.suffix.lower() in _FIGURE_SUFFIXES) \
            if fig_dir.is_dir() else []
        data = sorted(
            str(f.relative_to(self.quest_root)).replace("\\", "/")
            for root in (self.quest_root / "data" / "results", self.quest_root / "data")
            if root.is_dir() for f in root.rglob("*")
            if f.is_file() and f.suffix.lower() in self._RUN_DATA_SUFFIXES and "auto_collected" not in f.parts
            and "literature" not in f.parts
        )
        data = list(dict.fromkeys(data))
        from .figure_labels import MARK as _LABEL_MARK

        def persons(found: list[str]) -> list[str]:
            # FI's own requests (figure text that shows a code name, ``_node_write``) are not a reader's: one the
            # redraw could not carry out is said in run.log, and the paper is not written again for it.
            return [n for n in found if _LABEL_MARK not in n]

        if not figures or not data:
            self._log.warning(
                "[replot_layout] nothing to redraw from (%s); the figures stay as they are",
                "no figure" if not figures else "no saved numbers",
            )
            return {**done, "layout_missed": persons(notes)}
        from .data_shape import describe_data_files

        experiment = self.quest_root / "code" / "experiment.py"
        # What is inside each saved file (array keys, columns, JSON keys), so the script uses the real names and does
        # not guess them; a guessed key was what failed a real redraw.
        data_block = await asyncio.to_thread(describe_data_files, self.quest_root, data[:200])
        experiment_code = experiment.read_text(encoding="utf-8")[:12000] if experiment.is_file() else "(not saved)"
        script = self.quest_root / "code" / "replot_layout.py"
        # The script is model-written and can write anywhere: the figures, their records and the saved numbers are put
        # back whenever it fails, and any saved number it changed is put back even when it does not.
        backup = self.fi_dir / "layout_backup"
        shutil.rmtree(backup, ignore_errors=True)
        (backup / "figures").mkdir(parents=True)
        (backup / "records").mkdir(parents=True)
        big: list[str] = []
        for f in figures:
            shutil.copy2(fig_dir / f, backup / "figures" / f)
        stems = sorted({Path(f).stem for f in figures})
        for stem in stems:
            if (records_dir / f"{stem}.json").is_file():
                shutil.copy2(records_dir / f"{stem}.json", backup / "records" / f"{stem}.json")
        from .plot_style import PAPER_DIRNAME

        if (fig_dir / PAPER_DIRNAME).is_dir():  # the figures' paper copies, put back with them
            shutil.copytree(fig_dir / PAPER_DIRNAME, backup / "figures" / PAPER_DIRNAME)
        for d in data:
            if (self.quest_root / d).stat().st_size > self._RUN_DATA_MAX_BYTES:
                big.append(d)
                continue
            (backup / d).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.quest_root / d, backup / d)
        # Written last: a backup without it was cut short and is not restored from.
        (backup / "listed.json").write_text(
            json.dumps({"figures": figures, "data": data, "big": big, "stems": stems}), encoding="utf-8")

        def digest(path: Path) -> str:
            if not path.is_file():
                return ""
            with path.open("rb") as fh:  # in chunks: a saved-number file can be larger than memory allows
                return hashlib.file_digest(fh, "sha256").hexdigest()

        def put_back_records(names: list[str]) -> None:
            # A record goes with its figure: the one the figure had, or none when it had none.
            for stem in sorted({Path(n).stem for n in names}):
                saved = backup / "records" / f"{stem}.json"
                if saved.is_file():
                    records_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(saved, records_dir / f"{stem}.json")
                else:
                    (records_dir / f"{stem}.json").unlink(missing_ok=True)

        def other_numbers(name: str) -> bool:
            # A figure FI asked to rename must draw what its record held: only its text was to change.
            old = _record_numbers(old_records.get(name) or _read_json_or_none_path(
                backup / "records" / f"{Path(name).stem}.json"))
            if not old or not recorded:
                return False  # nothing recorded to compare with, or nothing could record the redraw
            path = records_dir / f"{Path(name).stem}.json"
            new = _read_json_or_none_path(path) if digest(path) != before_rec.get(Path(name).stem, "") else None
            return not _same_record_numbers(old, _record_numbers(new))

        before_fig = {f: digest(fig_dir / f) for f in figures}
        before_rec = {stem: digest(records_dir / f"{stem}.json") for stem in stems}
        before_big = {d: digest(self.quest_root / d) for d in big}
        listed = [f"- {f}" + (" (drawn by FI as the mean of the seeds, from each seed's numbers: leave it as it is)"
                              if f in means else "") for f in figures]
        # Two tries: a script that fails (or runs and changes no figure) is shown its error and written once more. The
        # figures and saved numbers are put back after a failed try, so the second starts from the same files.
        failure_block = ""
        failed, drawn, not_drawn = True, [], []
        for attempt in (1, 2):
            prompt = self._prompts["replot_layout"].substitute(
                notes_block="\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1)),
                figures_block="\n".join(listed),
                data_block=data_block,
                failure_block=failure_block,
                experiment_code=experiment_code,
            )
            try:
                reply = await self._chat(prompt, node="replot_layout")
            except BaseException:
                # Nothing has run on this try, and a failed earlier try was put back (all but a data file too large to
                # back up, which is reported where it changed): there is nothing left to restore.
                shutil.rmtree(backup, ignore_errors=True)
                raise
            code, _deps = _parse_implement_response(reply)
            if not code.strip():
                self._log.warning("[replot_layout] no script came back (try %d of 2)", attempt)
                failure_block = ("\n## Your previous reply had no script\nReply with exactly one fenced Python block.\n")
                continue
            script.write_text(code.rstrip() + "\n", encoding="utf-8")
            try:
                result = await self.executor.execute(
                    [str(self.executor.python_path(self.quest_root)), str(script)],
                    cwd=self.quest_root, timeout_s=min(self.config.execution.timeout_s, 300), env=env,
                )
            except BaseException:
                self._restore_layout_backup()
                raise
            # ``REPLOTTED: fig2.png`` and ``REPLOTTED: figures/fig2.png`` name the same figure.
            drawn = [Path(ln.split(":", 1)[1].strip().replace("\\", "/")).name
                     for ln in (result.stdout or "").splitlines() if ln.startswith("REPLOTTED:")]
            not_drawn = [ln.split(":", 1)[1].strip() for ln in (result.stdout or "").splitlines()
                         if ln.startswith("NOT_DRAWN:") and ln.split(":", 1)[1].strip()]
            timed_out = bool(getattr(result, "timed_out", False))
            failed = result.returncode != 0 or timed_out
            stderr_tail = (result.stderr or "")[-1500:].strip()
            typed: list[Any] = []
            other: list[str] = []
            if failed:
                what = "ran too long and was stopped" if timed_out else f"stopped with an error (exit code {result.returncode})"
                self._log.warning("[replot_layout] the redraw failed (try %d of 2, %s); the figures are as they were: %s",
                                  attempt, "it ran too long" if timed_out else f"rc={result.returncode}",
                                  stderr_tail[-300:])
            else:
                # Checked as the run's own figure scripts are: a figure drawn from numbers typed into the script, or one
                # FI asked only to rename that now draws other numbers, is not what the run computed.
                typed = _figdata.typed_series(script.read_text(encoding="utf-8", errors="replace"), script.name)
                other = [] if typed else [
                    f for f in figures if f in renamed and f not in means and digest(fig_dir / f) != before_fig[f]
                    and other_numbers(f)]
                if typed:
                    failed, what = True, "drew numbers typed into the script instead of reading them from the saved files"
                elif other:
                    failed, what = True, "drew other numbers than the run saved in " + ", ".join(other)
                if failed:
                    self._log.warning("[replot_layout] the redraw %s (try %d of 2); the figures are put back: %s", what,
                                      attempt, "; ".join(t.says() for t in typed[:4]) or ", ".join(other))
            for f in figures:
                changed = digest(fig_dir / f) != before_fig[f]
                # A figure drawn as the mean of the seeds is the engine's: the model cannot draw it from the seeds.
                if failed or (changed and (f not in drawn or f in means)):
                    if not failed:
                        self._log.warning("[replot_layout] %s was changed %s; it is put back", f,
                                          "though FI draws it" if f in means else "without being named")
                    shutil.copy2(backup / "figures" / f, fig_dir / f)
                    _put_back_paper_copies(backup / "figures", fig_dir, f)
                    if failed or not any(g != f and Path(g).stem == Path(f).stem and g in drawn for g in figures):
                        put_back_records([f])
            for d in data:
                if d in before_big:
                    if digest(self.quest_root / d) != before_big[d]:
                        self._log.warning("[replot_layout] the script changed %s, which is too large to have been backed up", d)
                    continue
                if digest(self.quest_root / d) != digest(backup / d):
                    self._log.warning("[replot_layout] the script changed the saved numbers in %s; they are put back", d)
                    shutil.copy2(backup / d, self.quest_root / d)
            # A file the script added: a new figure stays only when it is named and the script worked; a new data file
            # never.
            for f in self._files_added_since(figures, data, fig_dir):
                path = self.quest_root / f
                if failed or f.startswith("data/") or Path(f).name not in drawn:
                    self._log.warning("[replot_layout] %s was added by the script; it is removed", f)
                    path.unlink(missing_ok=True)
                    if f.startswith("figures/"):
                        # Its paper copy goes with it (or is the earlier one again, for a file name the run had), and so
                        # does its record.
                        _put_back_paper_copies(backup / "figures", fig_dir, f)
                        if failed or not any(Path(g).stem == Path(f).stem and g in drawn for g in figures):
                            put_back_records([f])
            if failed:
                pass
            elif (len(not_drawn) < len(notes) and all(digest(fig_dir / f) == before_fig[f] for f in figures)
                  and not self._files_added_since(figures, data, fig_dir)):
                # It ran, but drew nothing new, and not every note was one it said it cannot draw: a request was not
                # carried out. (A limit: figures cannot be matched to notes, so any changed figure counts as done.)
                failed = True
                put_back_records(figures)  # saved again unchanged, perhaps, which wrote the records again
                what = "ran without an error but changed no figure"
                self._log.warning("[replot_layout] the redraw %s (try %d of 2)", what, attempt)
                failure_block = (
                    f"\n## Your previous script failed\nIt {what}. Write the script again: save each figure the notes "
                    "are about over its file in `figures/`, and print `REPLOTTED: <file name>` for it; a note that is "
                    "not about how a figure is drawn gets a `NOT_DRAWN:` line instead.\n"
                )
                continue
            else:
                break
            failure_block = (
                f"\n## Your previous script failed\nIt {what}."
                + (f" The last lines of its error output:\n```\n{stderr_tail}\n```\n"
                   if stderr_tail and not typed and not other else "\n")
                + ("These plotting calls draw lists of numbers typed into the script:\n"
                   + "\n".join(f"- {t.says()}" for t in typed[:12])
                   + "\nRead every number a figure draws from the saved files listed above.\n" if typed else "")
                + ("Only the text of these figures was to change: draw each from the saved file that holds its "
                   "numbers, as they are.\n" if other else "")
                + "Write the script again. Use only the file names and the keys, columns and field names listed above; "
                  "do not guess a name.\n"
            )
        answered, loose = self._layout_answers(not_drawn, notes)
        if failed:
            done["layout_missed"] = persons(notes)
            # A note the script said is about the text (NOT_DRAWN) goes to the writer, not onto the "not applied" line.
            declared = {n.casefold() for n in answered + loose}
            done["layout_not_redrawn"] = persons([n for n in notes if n.casefold() not in declared] or notes)
            self._log.warning(
                "[replot_layout] your figure request was NOT applied: two tries at redrawing did not work, so the "
                "figures are as they were (%s)", "; ".join(n[:120] for n in notes))
        else:
            # What the writer is told the reader asked: the person's own notes the redraw could not draw, in their
            # words. FI's own notes never reach it; a line that names no note counts only when FI sent none.
            asked = persons(notes)
            ours = len(asked) < len(notes)
            told = persons(answered) + [ln for ln in loose if not ours or ln.casefold() in {n.casefold() for n in asked}]
            done["layout_missed"] = list(dict.fromkeys(told)) if asked else []
            self._log.info("[replot_layout] redrew %d figure(s) from the saved numbers, the experiment was not run again: %s",
                           len(drawn), ", ".join(drawn) or "-")
        shutil.rmtree(backup, ignore_errors=True)
        return done

    @staticmethod
    def _layout_answers(lines: list[str], notes: list[str]) -> tuple[list[str], list[str]]:
        """``(notes, loose)``: the notes the redraw's ``NOT_DRAWN: <number>: ...`` lines name by number, as they were
        written, and the lines that name none."""
        answered: list[str] = []
        loose: list[str] = []
        for line in lines:
            match = re.match(r"\s*(\d+)\s*[:.)]\s*", line)
            if match and 1 <= int(match.group(1)) <= len(notes):
                answered.append(notes[int(match.group(1)) - 1])
            else:
                loose.append(line)
        return answered, loose

    async def _rename_mean_figures(self, state: QuestState, figures: list[str], env: dict[str, str]) -> list[str]:
        """FI's renames of a figure drawn as the mean of the seeds: each code name the plan has a label for is written as
        the plan names it in ``code/replot_figures.json``, and ``code/replot_figures.py`` (FI's own, from each seed's
        numbers) draws the figures again. A code name the plan has no label for is left as it is, and said in
        run.log. Returns the figures drawn again."""
        from .plan import plan_labels

        code_dir = self.quest_root / "code"
        plan_file, script = code_dir / "replot_figures.json", code_dir / "replot_figures.py"
        labels = plan_labels((state.get("design") or {}).get("protocol"))
        renames: dict[str, dict[str, str]] = {}
        left: list[str] = []
        for found in _figure_label_found(state, code_dir, root=self.quest_root):
            if found.get("figure") not in figures:
                continue
            for name in found.get("code_names") or []:
                if labels.get(name):
                    renames.setdefault(found["figure"], {})[name] = labels[name]
                else:
                    left.append(f'{found["figure"]}: "{name}"')
        if left:
            self._log.warning(
                "[replot_layout] the mean of the seeds keeps a code name the plan gives no label for (FI draws such "
                "figures itself and renames only to the plan's own label): %s", "; ".join(dict.fromkeys(left)))
        plan = _read_json_or_none_path(plan_file)
        if not renames:
            return []
        if not isinstance(plan, dict) or not script.is_file():
            self._log.warning("[replot_layout] %s cannot be renamed: FI's redraw of the mean of the seeds is not saved",
                              ", ".join(sorted(renames)))
            return []
        before = plan_file.read_text(encoding="utf-8")
        for figure in plan.get("figures") or []:
            mapping = renames.get(str(figure.get("file"))) if isinstance(figure, dict) else None
            if not mapping:
                continue
            figure["suptitle"] = _rename_code_names(figure.get("suptitle"), mapping)
            for panel in figure.get("axes") or []:
                for key in ("title", "xlabel", "ylabel"):
                    panel[key] = _rename_code_names(panel.get(key), mapping)
                for line in panel.get("lines") or []:
                    line["label"] = _rename_code_names(line.get("label"), mapping)
        plan_file.write_text(json.dumps(plan), encoding="utf-8")
        try:
            result = await self.executor.execute(
                [str(self.executor.python_path(self.quest_root)), str(script)],
                cwd=self.quest_root, timeout_s=min(self.config.execution.timeout_s, 300), env=env,
            )
        except BaseException:
            plan_file.write_text(before, encoding="utf-8")
            raise
        drawn = [ln.split(":", 1)[1].strip() for ln in (result.stdout or "").splitlines() if ln.startswith("REPLOTTED:")]
        if result.returncode != 0:
            self._log.warning("[replot_layout] renaming the mean of the seeds rc=%d: %s", result.returncode,
                              (result.stderr or "")[-300:])
        self._log.info("[replot_layout] renamed %s as the plan names it, drawn again as the mean of the seeds",
                       ", ".join(f for f in drawn if f in renames) or "nothing")
        return [f for f in drawn if f in renames]

    def _files_added_since(self, figures: list[str], data: list[str], fig_dir: Path) -> list[str]:
        """Figure and saved-number files now on disk that were not there when the list was taken (relative to the quest)."""
        now_figures = [f"figures/{f.name}" for f in fig_dir.iterdir() if f.is_file() and f.suffix.lower() in _FIGURE_SUFFIXES] \
            if fig_dir.is_dir() else []
        now_data = [
            str(f.relative_to(self.quest_root)).replace("\\", "/")
            for root in (self.quest_root / "data" / "results", self.quest_root / "data")
            if root.is_dir() for f in root.rglob("*")
            if f.is_file() and f.suffix.lower() in self._RUN_DATA_SUFFIXES and "auto_collected" not in f.parts
            and "literature" not in f.parts
        ]
        known = {f"figures/{f}" for f in figures} | set(data)
        return sorted(f for f in dict.fromkeys(now_figures + now_data) if f not in known)

    def _restore_layout_backup(self) -> None:
        """Put back what a redraw that was cut short (the process was killed, or the run raised) left changed: the
        figures, their records and saved numbers as they were, the files it added removed, and its script removed. A
        backup with no list was cut short while it was being made, and nothing was changed yet."""
        (self.quest_root / "code" / "replot_layout.py").unlink(missing_ok=True)
        backup = self.fi_dir / "layout_backup"
        if not backup.is_dir():
            return
        listed = _read_json_or_none_path(backup / "listed.json")
        if isinstance(listed, dict):
            figures = [str(f) for f in listed.get("figures") or []]
            data = [str(d) for d in listed.get("data") or []]
            fig_dir = self.quest_root / "figures"
            from .plot_style import RECORDS_DIRNAME

            records_dir = self.fi_dir / RECORDS_DIRNAME
            stems = {str(s) for s in listed.get("stems") or []}
            for f in self._files_added_since(figures, data, fig_dir):
                (self.quest_root / f).unlink(missing_ok=True)
                if f.startswith("figures/") and Path(f).stem not in stems:
                    (records_dir / f"{Path(f).stem}.json").unlink(missing_ok=True)
            for stem in stems:
                if (backup / "records" / f"{stem}.json").is_file():
                    records_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup / "records" / f"{stem}.json", records_dir / f"{stem}.json")
                else:
                    (records_dir / f"{stem}.json").unlink(missing_ok=True)
            for f in figures:
                if (backup / "figures" / f).is_file():
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup / "figures" / f, fig_dir / f)
            from .plot_style import PAPER_DIRNAME

            shutil.rmtree(fig_dir / PAPER_DIRNAME, ignore_errors=True)
            if (backup / "figures" / PAPER_DIRNAME).is_dir():
                shutil.copytree(backup / "figures" / PAPER_DIRNAME, fig_dir / PAPER_DIRNAME)
            for d in data:
                if (backup / d).is_file():
                    (self.quest_root / d).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup / d, self.quest_root / d)
            self._log.warning("[replot_layout] an earlier redraw was cut short; the figures and saved numbers are put back")
        shutil.rmtree(backup, ignore_errors=True)

    async def _replot_replicate_figures(
        self, figures: list[str], records_dir: Path, seeds: list[int], *,
        python: Any, env: dict[str, str] | None, assertions: list[Any],
    ) -> dict[str, int]:
        """Draw each line figure again as the mean over ``seeds``, shaded with
        its 95% confidence interval, from the lines the plot-style recorder
        kept at every seed. The engine computes the numbers and
        ``code/replot_figures.py`` draws them in the quest's Python, under the
        house style, over the same file. Returns each figure drawn, with its
        number of seeds. Error bars count as lines: each series is drawn at its
        mean without them. A figure that holds anything else, or whose lines
        differ between seeds in label or x, is not drawn, and neither is a
        figure the redraw fails on."""

        def recorded(path: Path) -> Any:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None

        plans = []
        for name in figures:
            runs = [recorded(records_dir / f"{Path(name).stem}.seed{seed}.json") for seed in seeds]
            plan = _replicate_line_figure(name, runs, assertions)
            if plan is not None:
                plans.append(plan)
        if not plans:
            return {}
        code_dir = self.quest_root / "code"
        script = code_dir / "replot_figures.py"
        try:
            code_dir.mkdir(parents=True, exist_ok=True)
            script.write_text(
                (Path(__file__).parent / "replot_figures.py").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (code_dir / "replot_figures.json").write_text(
                json.dumps({"figures": plans}), encoding="utf-8",
            )
        except OSError as e:
            self._log.warning("[execute] could not write the figure redraw: %s", e)
            return {}
        result = await self.executor.execute(
            [str(python), str(script)],
            cwd=self.quest_root,
            timeout_s=min(self.config.execution.timeout_s, 300),
            env=env,
        )
        drawn = {
            line.split(":", 1)[1].strip()
            for line in (result.stdout or "").splitlines() if line.startswith("REPLOTTED:")
        }
        if result.returncode != 0:
            self._log.warning(
                "[execute] redrawing figures as the mean of the seeds rc=%d: %s",
                result.returncode, (result.stderr or "")[-300:],
            )
        replotted = {plan["file"]: len(seeds) for plan in plans if plan["file"] in drawn}
        if replotted:
            self._log.info(
                "[execute] drew %d figure(s) as the mean of %d seeds with 95%% CI bands: %s",
                len(replotted), len(seeds), ", ".join(sorted(replotted)),
            )
        return replotted

    def _manifest_repair_gave_up(self, state: QuestState, history: list[dict[str, Any]] | None = None) -> QuestState | None:
        """When the repair gives up, or none is left, on a run the run-manifest check sent back
        (any difference under ``engine.run_manifest_check: block``; under ``warn`` numbers FI's trials never produced): the
        difference it found still stands, so the quest must stop and name
        it, as when the repairs run out, and not go on as a run that merely produced no results (which went back to
        design, then wrote a paper from none, the named difference lost). The repairs count as spent and the same
        scripts run once more, so the stop happens in the run step and a resume after fixing them runs them again: the
        check finds the difference and the run stops there, with the reason. On the trial contract FI's trials are
        reused (the simulation is unchanged); the older contract, or a record changed on disk, runs them again.
        ``None`` when the failure was not a run-manifest difference."""
        stop = self._trial_checks_stop(state)  # a fault FI's checks of the trials found stands: the quest stops (no paper)
        if stop is not None:
            return stop
        # run_manifest_failures counts repairs the check sent: under block any difference, under warn only numbers FI's
        # trials never produced.
        if self.config.engine.run_manifest_check == "off" or not int(state.get("run_manifest_failures") or 0):
            return None
        self._log.warning(
            "[execute_reflect] no repair of the run-manifest difference (it gave up, or none is left); the run is "
            "checked once more and stops with it"
        )
        patch: QuestState = {
            "run_manifest_failures": _manifest_repair_budget(
                int(self.config.engine.run_manifest_repair_attempts), bool(state.get("run_manifest_analysis_only"))),
            "exec_patch_pending": True,
            # a repair of the analysis spends no crash repair (it has the budget of its own above)
            "exec_reflect_iter": int(state.get("exec_reflect_iter") or 0) + (0 if state.get("run_manifest_analysis_only") else 1),
        }
        if history is not None:
            patch["exec_reflect_history"] = history  # the repair's own reason for giving up, kept
        return patch

    def _removed_by_fi(self, violation: Any) -> bool:
        """Whether ``violation`` is a bounded quantity FI's own record took out of the results (``needs/RESULT_CHECK.json``,
        the typed-number record): it is gone on purpose, so a repair asking for it again would undo FI's check."""
        if getattr(violation, "kind", "") != "missing":
            return False
        from core import plausibility
        from core import result_checks as _result_checks

        gone = set(_result_checks.left_out_by_record(self.quest_root))
        path = str(getattr(violation, "path", ""))
        return bool(gone) and (path in gone or any(seg in gone for seg in plausibility._segments(path)))

    def _fis_body_of_problems(self, record: dict[str, Any], findings: Any, implausible: Any) -> str:
        """The function of derive.py FI's record of a run that ended cleanly lays its problems to: one whose own call raised the
        numeric warnings found (FI caught them per function), else the one that made a derived quantity a declared bound
        flags. ``""``: none, the simulation's to repair."""
        bodies = record.get("bodies") if isinstance(record.get("bodies"), dict) else {}
        wanted = {(w.kind, w.text) for w in findings or []}
        for name, row in bodies.items():
            said = (row or {}).get("warnings") if isinstance(row, dict) else None
            if said and wanted & {(w.kind, w.text) for w in _numeric.scan("\n".join(map(str, said)))}:
                return str(name)
        derived = record.get("derived") if isinstance(record.get("derived"), dict) else {}
        for v in implausible or []:
            for path in (getattr(v, "path", ""), *(getattr(v, "paths", ()) or ())):
                for part in reversed(re.split(r"[.\[\]/]", str(path or ""))):
                    if part in derived:
                        body = _analysis_frame.made_by(record, quantity=part)
                        if body:
                            return body
        return ""

    def _fis_analysis_route(self, state: QuestState, exec_result: dict[str, Any], rc: int,
                            overlaps: list[str], findings: Any = (), implausible: Any = ()) -> dict[str, Any]:
        """Where the repair of a run goes when FI wrote its analysis (core/analysis_frame.py), read from FI's own record of the
        run (``data/results/derived.json``), never from the words of a traceback: ``{"body": name}`` for a function of the
        study's derive.py (``""``: a statement of derive.py outside the functions), ``{"body": None}`` for the simulation, or
        ``{"stop": patch}``. Who caused a failure is what was running: whatever stopped the analysis while a function of the
        study's ran is that function's. FI's experiment.py is never sent to the model: a fault of FI's own (exit code 2, a stop
        in one of FI's own steps) ends the quest with the reason said plainly, and no model call. A check after a clean run
        (``findings``: its numeric warnings; ``implausible``: a declared bound broken) goes to the function FI's record lays it
        to, else to the simulation; a check that sent the run back is never a stop."""
        design = state.get("design")
        self._routed_failure = ""  # the reason FI's record gave for the function it routes to (read by the repair prompt)
        record = _analysis_frame.read_record(self.quest_root)
        fresh = bool(state.get("result_json")) and not record.get("from_earlier_run")
        if rc == 0:
            if not overlaps:
                # a result that runs but is wrong (all zero, out of bounds, numerics): the function of derive.py FI's record
                # lays it to, else the simulation
                return {"body": (self._fis_body_of_problems(record, findings, implausible) if fresh else "") or None}
            files = {Path(str(f.get("file") or "")).name: f"figure_{i}"
                     for i, f in (_analysis_frame.plan_of(design).get("figures") or {}).items() if isinstance(f, dict)}
            for line in overlaps:
                body = files.get(Path(line.split(":", 1)[0]).name)
                if body:
                    return {"body": body}
            # A figure FI drew itself: there is no function of the study's to redraw it.
            self._log.warning("[execute_reflect] figure_overlap: the overlapping figure is drawn by FI, not by a function of "
                              "derive.py; keeping it as drawn")
            return {"stop": {"figure_overlap_repaired": True}}
        stderr = str(exec_result.get("stderr_tail") or "")
        if "failed_body" in exec_result:
            return {"body": str(exec_result.get("failed_body") or "")}  # FI's check named the function itself
        # The analysis printed its RESULT_JSON line, so FI's record is this run's (the line itself is not kept in the tail)
        printed = bool(state.get("result_json")) or "RESULT_JSON:" in str(exec_result.get("stdout_tail") or "")
        sent_back = bool(exec_result.get("directive") or exec_result.get("result_check_sent_back")
                         or (exec_result.get("failed_script") == _analysis_frame.EXPERIMENT_FILE
                             and rc != _analysis_frame.FI_FAULT_EXIT))
        # The step read when the run ended (its whole stderr, then the file of what was running, if that is the run's own);
        # the file is read here only for a record that has none (an older one).
        step = str(exec_result.get("analysis_step") or "") or _analysis_frame.running_step(
            stderr, None if "analysis_step" in exec_result else self.quest_root)
        found = _analysis_frame.failure(record, stderr, rc, printed or sent_back, step=step,
                                        timed_out=bool(exec_result.get("timed_out")))
        if found["kind"] == "body":
            self._routed_failure = found["why"]
            return {"body": found["body"]}
        if found["kind"] == "module":
            return {"body": ""}
        if sent_back:
            # A check of FI's sent a run back that FI's record shows ended cleanly, naming no function of the study's: it
            # goes where it always went (the simulation), never to a stop.
            return {"body": None}
        why = found["why"]
        if found["formula"]:
            text = f"FI's analysis cannot compute a formula of the plan: {why}"
            # ``--from design`` takes the design written in plan.md again (it is not moved aside) and replaces the frozen
            # protocol, so it needs a person's approval (``--approve-as``).
            advice = ("Correct that formula in the design block of plan.md, then run the quest again from the design step: "
                      f"`python launch.py --resume {self.quest_id} --from design --approve-as <your name>` (plan.md is "
                      "read again as you left it; the frozen experiment plan, the code and the run are made again from it, "
                      "and the old ones are kept in .fi/previous/).")
            tried = ["ran the study's trials", "worked out the plan's formulas from FI's record of the trials",
                     "a formula is not one FI can compute, so no function of the study's was asked to repair it"]
        else:
            text = f"FI's own analysis stopped (a fault of FI's, not of the study's code): {why}"
            advice = ("This is a fault in FI: report it with the quest's .fi/run.log, and once it is fixed run the quest "
                      f"again from the run step (`python launch.py --resume {self.quest_id} --from run`).")
            tried = ["ran the study's trials", "ran FI's analysis on FI's record of them",
                     "the fault is in FI's own analysis, so it was not sent to the model to repair"]
        stop = self._stop_before_run(text, "cannot_repair", tried=tried, advice=advice, ran=True)
        stop["exec_result"]["fis_fault"] = True  # the stop report says no repair was tried, and why
        return {"stop": stop}

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
        resumed = self._resume_after_numeric_stop()
        if resumed is not None:
            return resumed
        exec_result = state.get("exec_result") or {}
        if any(exec_result.get(k) for k in _STOPPED_BEFORE_THE_STUDY) or exec_result.get("flat_objective"):
            # the study would not fit the time allowed (core/run_estimate.py), or the search's objective did not move
            # (core/optimise.py): there is no script to repair
            return {}
        rc = exec_result.get("returncode", 0)
        # ``_node_execute`` stores ``result_json or {}``, so a script that
        # exits 0 WITHOUT a RESULT_JSON marker lands as an empty dict — which
        # ``is not None`` wrongly counted as "parsed", skipping repair. Treat
        # an empty result_json as no usable result so execute_reflect retries.
        has_result_json = bool(state.get("result_json"))
        findings = self._numeric_findings(state)
        degenerate = bool(
            rc == 0
            and has_result_json
            and self.config.engine.degenerate_run_guard
            and _is_degenerate_result(state.get("result_json") or {})
        )
        implausible = [v for v in _assertion_violations(self._in_force(state), self._walk_memo())
                       if not self._removed_by_fi(v)]
        # A legend drawn over its data, or a figure title over a panel title, is
        # sent back for a redraw once, and only when nothing else is wrong with
        # the run: a repair of a crash, an all-zero result or a broken bound
        # regenerates the figures anyway, and they are measured again after it.
        overlaps = (
            self._figure_overlaps_to_repair(state)
            if rc == 0 and has_result_json and not degenerate and not implausible and not findings
            else []
        )

        # Clean success (and not a degenerate all-zero run) → no-op.
        if rc == 0 and has_result_json and not degenerate and implausible:
            for v in implausible:
                self._log.warning("[execute_reflect] implausible: %s", v.describe())
        if rc == 0 and has_result_json and not degenerate and not implausible and not findings and not overlaps:
            left = _figure_overlap_findings({**state, "figure_overlap_repaired": False})
            if left:
                self._log.warning(
                    "[execute_reflect] figure_overlap left as drawn (%s): %s",
                    "the one redraw has been made" if state.get("figure_overlap_repaired")
                    else "no repair attempt to spare", "; ".join(left[:3]),
                )
            self._log.info("[execute_reflect] script succeeded; skipping repair")
            return {}

        iters = state.get("exec_reflect_iter", 0)
        # The repair of an analysis the run record sent back has a budget of its own (``run_manifest_failures``, at most
        # three): it neither waits for a crash repair to be left nor spends one.
        manifest_turn = (rc != 0 and bool(state.get("run_manifest_failures")) and bool(state.get("run_manifest_analysis_only"))
                         and self.config.engine.run_manifest_check != "off")
        # A run that ran to the end and only warned has a budget of its own too (exec_reflect_max_iterations repairs for
        # its warnings, counted from the repairs the history records as made for them): repairs spent on crashes never
        # leave the warnings with none, and a stop for them says how many repairs were made for them.
        numeric_turn = self._numeric_turn(state)
        numeric_made = _numeric.repairs_for_warnings(state.get("exec_reflect_history"))
        # Every repair is numbered in one sequence, whichever budget it is on.
        # The repair of a fault the gate found in the analysis has a budget of its own too (``gate_repairs``, counted when it
        # was sent): it neither waits for a crash repair to be left nor spends one.
        gate_turn = rc != 0 and bool(exec_result.get("gate_repair"))
        gate_made = int(state.get("gate_repairs") or 0)
        attempt_no = iters + numeric_made + gate_made + (0 if gate_turn else 1)
        bump = 0 if manifest_turn or numeric_turn or gate_turn else 1
        if numeric_turn and numeric_made >= self.config.engine.exec_reflect_max_iterations:
            stop = self._manifest_repair_gave_up(state) or self._result_check_gave_up(state)
            if stop is not None:
                return stop
            # Numeric warnings the repairs made for them did not remove: not something to write a paper over.
            research = getattr(self.config, "rigor_profile", "default") == "research"
            return self._stop_before_run(
                "the run's numerics warned: " + "; ".join(w.describe() for w in findings),
                "cannot_repair",
                tried=["ran the experiment and read the numerical warnings it raised",
                       *[_one_line(w.describe(), 300) for w in findings[:3]],
                       f"sent the script back to be repaired {numeric_made} time(s) for these warnings; they stood"
                       if numeric_made else "no repair is allowed (engine.exec_reflect_max_iterations is 0)"],
                # A setting the person may choose (not a code edit); a research quest holds it fixed.
                advice=None if research else (
                    f"To go on with the warnings noted in the evidence instead, set `engine.numeric_warnings: warn` in "
                    f"the quest's config.yaml, approve it with `python launch.py --update {self.quest_id}`, and run it "
                    f"again from the run step (`--resume {self.quest_id} --from run`)."))
        # A fault of FI's own analysis (core/analysis_frame.py) stops plainly whatever repairs are left: it asks the model for
        # nothing, so the repair budget has no say in it, and it is never taken on to the analysis.
        if (rc != 0 and self._split_on(state) and (self.quest_root / "code" / _split_run.SIMULATE_NAME).is_file()
                and exec_result.get("failed_script") not in (_split_run.SIMULATE_NAME, _trial_runner.SUBMIT_NAME)
                and _analysis_frame.is_fis_file(self.quest_root / "code" / _analysis_frame.EXPERIMENT_FILE)):
            fis_stop = self._fis_analysis_route(state, exec_result, rc, []).get("stop")
            if fis_stop is not None:
                return fis_stop
        if iters >= self.config.engine.exec_reflect_max_iterations and not manifest_turn and not numeric_turn and not gate_turn:
            # A run the run-manifest check sent back, with no repair left (earlier crashes spent them, or none are
            # allowed): the difference stands, and the quest stops and names it, never going on with no results.
            stop = self._manifest_repair_gave_up(state) or self._result_check_gave_up(state)
            if stop is not None:
                return stop
            # Proceed regardless; analyze owns the authoritative degenerate
            # flag against the final result (covers max_iterations=0 too).
            self._log.warning(
                "[execute_reflect] %s after %d attempt(s); proceeding to analyze",
                "result still degenerate (all metrics ~0)" if degenerate
                else "iterations exhausted",
                iters,
            )
            return {}

        await self._shadow("repair", state, taken="repaired the script")
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
        elif rc == 0 and has_result_json and implausible:
            # Say WHICH declared bound broke. Without this the model saw rc=0,
            # a parsed RESULT_JSON and "the script just failed", had nothing
            # to repair, and invented a code fault -- then capped the diverging
            # value at the bound on the next pass so the gate would take it.
            # The repair has to be told that an honest null beats a plausible
            # number.
            clamped = any(getattr(v, "kind", "") == "clamped" for v in implausible)
            self._log.warning(
                "[execute_reflect] rc=0 but %d value(s) break the design's "
                "declared bounds or are capped by the script%s — attempting repair (iter %d)",
                len(implausible),
                " (at least one is a capped number)" if clamped else "",
                iters + 1,
            )
            rj_preview = json.dumps(state.get("result_json") or {}, indent=2)[:1500]
            stdout_for_prompt = (
                "IMPLAUSIBLE RESULT: the script exited 0 and printed RESULT_JSON, "
                "but these values break the bounds the DESIGN declared for them, or are a number the script "
                "capped rather than computed:\n"
                + "\n".join(f"- {v.describe()}" for v in implausible[:10])
                + "\n\nFind the cause before changing anything. If it is a bug "
                "(wrong units, a factor of two, a sign error, a mis-set "
                "parameter), fix it. If the value is genuinely what the method "
                "produces here (a scheme that diverges at this step size, an "
                "unstable fit), that is a finding, not a bug: emit null for that "
                "value and say why under \"why_missing\" (for example \"why_missing\": {\"<name>\": \"diverged\"}). "
                "NEVER clamp, cap or clip a result to the bound or replace it "
                "with any constant: a capped value is detected and rejected, and "
                "it would state something false."
                + ("\n\nA value above is capped by the script itself (at its bound, or at a constant the "
                   "script caps that very quantity at, inside the range); remove that cap." if clamped else "")
                + "\n\nRESULT_JSON was:\n"
                f"{rj_preview}\n\nOriginal stdout tail:\n"
                + exec_result.get("stdout_tail", "")[:1000]
            )
            returncode_for_prompt = "0 (ran, but values break the declared bounds)"
            result_json_note = "yes (rejected — see stdout)"
        elif numeric_turn:
            self._log.warning(
                "[execute_reflect] rc=0 but the run's numerics warned (%d) — attempting repair (iter %d): %s",
                len(findings), iters + 1, "; ".join(w.describe() for w in findings[:4]),
            )
            stdout_for_prompt = _numeric.directive(findings) + "\n\nOriginal stdout tail:\n" + exec_result.get("stdout_tail", "")[:1000]
            returncode_for_prompt = "0 (ran, but its numerics warned)"
            result_json_note = "yes (stands only if the warnings are resolved)"
        elif overlaps:
            self._log.warning(
                "[execute_reflect] figure_overlap: rc=0 and the results stand, but %d overlap(s) "
                "were measured on the saved canvas -- asking for one redraw (iter %d): %s",
                len(overlaps), iters + 1, "; ".join(overlaps[:3]),
            )
            stdout_for_prompt = _figure_overlap_directive(overlaps)
            returncode_for_prompt = "0 (ran; the figures overlap)"
            result_json_note = "yes (stands: change only the figures' layout)"
        else:
            stdout_for_prompt = exec_result.get("stdout_tail", "")[:2000]
            returncode_for_prompt = str(rc)
            result_json_note = "yes" if has_result_json else "no"

        # Two scripts (execution.split_analysis): a failure in the simulation rewrites
        # simulate.py; everything else (a crash in the analysis, a result that runs but is
        # wrong, a figure that overlaps) rewrites experiment.py against the raw files that
        # are already on disk.
        simulate_path = self.quest_root / "code" / _split_run.SIMULATE_NAME
        split = self._split_on(state) and simulate_path.is_file()
        repair_simulation = split and exec_result.get("failed_script") == _split_run.SIMULATE_NAME
        repair_submit = split and exec_result.get("failed_script") == _trial_runner.SUBMIT_NAME
        script_code = state.get("code") or ""
        analysis_on_disk = self.quest_root / "code" / "experiment.py"
        # The analysis FI writes (core/analysis_frame.py): experiment.py is FI's and is never sent to the model. What failed is
        # read from FI's own record of the run: a function of the study's derive.py (repaired alone, through its body), a run
        # that needs the simulation repaired, or a fault of FI's own (said plainly, and the quest stops).
        repair_body, failed_body = False, ""
        if split and not (repair_simulation or repair_submit) and _analysis_frame.is_fis_file(analysis_on_disk):
            routed = self._fis_analysis_route(state, exec_result, rc, overlaps, findings, implausible)
            if "stop" in routed:
                return routed["stop"]
            repair_body, failed_body = routed["body"] is not None, routed["body"] or ""
            repair_simulation = not repair_body
        if analysis_on_disk.is_file() and not (repair_simulation or repair_submit):
            # The file is what runs; the state can be a version behind it (a repair of a second model's findings, a redrawn
            # figure), and a repair of the stale text would undo it.
            script_code = analysis_on_disk.read_text(encoding="utf-8")
        if repair_body:
            script_code = (self.quest_root / "code" / _analysis_frame.DERIVE_FILE).read_text(encoding="utf-8")
        if split:
            if repair_body:
                split_note = _skeleton.reflect_derive(failed_body, compared=bool(_in_run.of_study(self._protocol_block(state))),
                                                       frame_version=_skeleton.quest_frame_version(self.quest_root / "code"))
            elif repair_submit:
                script_code = (self.quest_root / "code" / _trial_runner.SUBMIT_NAME).read_text(encoding="utf-8")
                split_note = (
                    "[FI NOTE] The script below is code/submit.py, the driver that submits FI's job array to the cluster "
                    "and reports it pending or done (the cluster section of the design above). It failed: fix it, and "
                    "return submit.py only.\n"
                )
            elif repair_simulation:
                script_code = simulate_path.read_text(encoding="utf-8")
                # The crash may be in the model's package simulate.py calls: shown, and fixable there.
                framed = _skeleton.has_frame(script_code)
                split_note = ((_skeleton.REFLECT_SIMULATE if framed else _SPLIT_REFLECT_SIMULATE)
                              + ((_skeleton.REFLECT_SEARCH if framed else _SEARCH_REFLECT_SIMULATE)
                                 if _optim.has_block(state.get("design")) else "")
                              + ("" if framed else _with_newline(_skeleton.level_rule(self._protocol_block(state), script_code)))
                              + self._package_repair_note(state))
            elif _optim.has_block(state.get("design")):
                split_note = _SEARCH_REFLECT_ANALYSIS
            else:
                raw_root = _split_run.raw_root_of(self.quest_root, self.config.execution.raw_dir)
                split_note = _SPLIT_REFLECT_ANALYSIS.format(
                    listing=_split_run.listing(_split_run.raw_dir_for(raw_root, 0)) or "(none)",
                )
            stdout_for_prompt = split_note + "\n" + stdout_for_prompt

        # The one round the redraw gets, spent whatever the model answers.
        spent: QuestState = {"figure_overlap_repaired": True} if overlaps else {}
        # The failed function's own failure, whole, from FI's record, ahead of the stderr (cut to its last 2000 characters: with
        # several figures failing, the cut can hold only the later ones' text); nothing is added when the cut keeps it.
        kept_stderr = str(exec_result.get("stderr_tail", ""))
        if self._agent_mode() or not exec_result.get("directive"):
            kept_stderr = kept_stderr[-2000:]
        own_failure_block = ""
        if repair_body and failed_body:
            own = _analysis_frame.own_failure(_analysis_frame.read_record(self.quest_root), failed_body, getattr(self, "_routed_failure", ""))
            if own and own not in kept_stderr:
                own_failure_block = self._prompts["execute_reflect_failure"].substitute(failure=own) + "\n\n"
        if self._agent_mode():
            # The coding agent fixes its own code (core/agent_build.py): one round, with what the run said.
            return await self._reflect_by_agent(state, iters, bump, returncode_for_prompt, stdout_for_prompt,
                                                exec_result, spent, own_failure=own_failure_block,
                                                tag=_numeric.REPAIR_FOR if numeric_turn else "", attempt=attempt_no)
        prompt = self._prompts["execute_reflect"].substitute(
            topic_block=self._topic_block(),
            own_failure_block=own_failure_block,
            # Whole, always: the reply is the full corrected script, and a model shown only the first 8000 characters
            # has to make up the rest. A live kimi-k3 quest's 15,706-character analysis was repaired three times from
            # its first half; one repair came back with a helper cut off mid-name, and the run never recovered.
            previous_code=script_code,
            returncode=returncode_for_prompt,
            stdout_tail=stdout_for_prompt,
            stderr_tail=exec_result.get("packages_note", "") + kept_stderr,
            duration_s=f"{exec_result.get('duration_s', 0):.2f}",
            figures_count=str(len(state.get("figures") or [])),
            result_json_present=result_json_note,
            reflect_history_block=history_block,
            design_block=self._code_design_block(state),
            clarify_block=_format_clarify(state),
        )
        if repair_simulation:
            prompt = _skeleton.framed_prompt(script_code, prompt)  # what may change, first (nothing when there is no frame)
        elif repair_body:
            prompt = _skeleton.derive_prompt(script_code, prompt, failed_body)  # the function of derive.py, never experiment.py
        try:
            text = await self._chat(prompt, node="execute_reflect")
        except Exception as exc:  # noqa: BLE001
            if not overlaps:
                raise
            # A redraw is a nicety on a run that worked: failing to ask for it
            # never stops the quest.
            self._log.warning(
                "[execute_reflect] the figure redraw call failed (%r); keeping the figures as drawn", exc,
            )
            return spent
        parsed = _repair_reply_json(text, node="execute_reflect")

        # A run whose figures merely overlap has a result that stands, so a
        # refusal, an empty answer or a redraw that is not Python leaves the
        # script and its figures as they are: no give-up sentinel (which would
        # switch off every later repair) and no repair attempt spent.
        give_up = (parsed.get("give_up_reason") or "").strip()
        if give_up:
            self._log.warning("[execute_reflect] LLM gave up: %s", give_up[:200])
            if overlaps:
                return spent
            history.append({
                **({"for": _numeric.REPAIR_FOR} if numeric_turn else {}),
                "iter": attempt_no,
                "returncode": rc,
                "stderr_tail": exec_result.get("stderr_tail", "")[-400:],
                "patch_summary": f"(gave up: {give_up[:120]})",
            })
            return self._manifest_repair_gave_up(state, history) or await self._result_check_gave_up_with_figures(
                state, history) or {
                "exec_give_up_reason": give_up,
                "exec_reflect_iter": iters + bump,
                "exec_reflect_history": history,
                **await self._figures_after_give_up(state),
            }

        new_code = parsed.get("code") or ""
        script_name = (_trial_runner.SUBMIT_NAME if repair_submit
                       else _split_run.SIMULATE_NAME if repair_simulation
                       else _analysis_frame.DERIVE_FILE if repair_body else "experiment.py")
        # The reply's functions are put into the script and the rest of it stays as it was (core/script_reply.py).
        split_why = ""
        if new_code.strip() or (isinstance(parsed.get("functions"), str) and parsed["functions"].strip()):
            new_code, split_why = self._script_from_reply("execute_reflect", script_code, new_code, parsed, script_name)
        # A package file that changes a parameter list the outline fixed is not used; the repair is still spent and the
        # history says why, so the next repair sees it (a refused package-only fix is not a repair that gave up).
        refused = self._package_refused(state, parsed) if repair_simulation else []
        if not new_code.strip() and repair_simulation and (self._package_repair_files(state, parsed) or refused):
            # The fix is in the model's package alone: simulate.py stays as it is.
            new_code, split_why = script_code, ""
        if not new_code.strip() and not split_why:
            self._log.warning(
                "[execute_reflect] LLM returned no `code` field; proceeding without repair"
            )
            if overlaps:
                return spent
            return self._manifest_repair_gave_up(state) or await self._result_check_gave_up_with_figures(state) or {
                "exec_give_up_reason": "(LLM produced no patched code)",
                "exec_reflect_iter": iters + bump,
                **await self._figures_after_give_up(state),
            }
        if overlaps:
            try:
                ast.parse(new_code)
            except (SyntaxError, ValueError):
                self._log.warning(
                    "[execute_reflect] the figure redraw is not a Python script; keeping the "
                    "script as written, with its figures as drawn",
                )
                await self._keep_refused_repair(
                    "execute_reflect", script_name, new_code, "the figure redraw is not a Python script")
                return spent

        patch_summary = parsed.get("patch_summary") or "(no summary)"
        # The repaired script must keep what FI calls it by (a stand-in word parses as Python and defines nothing): a
        # reply that does not is refused with the script kept, and the reason goes back with the next try (the history).
        broken = split_why or _trial_runner.repair_breaks_contract(script_code, new_code, script_name)
        if broken:
            self._log.warning("[execute_reflect] the repair of %s is not used: %s; keeping it as written", script_name, broken)
            await self._keep_refused_repair("execute_reflect", script_name, new_code, broken, {
                "returncode": rc, "stderr_tail": str(exec_result.get("stderr_tail", ""))[-1500:],
                "result_json": exec_result.get("result_json")})
            history.append({
                **({"for": _numeric.REPAIR_FOR} if numeric_turn else {}),
                "iter": attempt_no, "returncode": rc, "stderr_tail": exec_result.get("stderr_tail", "")[-400:],
                "patch_summary": (f"[the repair was not used and {script_name} is as it was: {broken}. Give the "
                                  f"functions to change, each one whole.] " + patch_summary)[:400],
            })
            return {**spent, "exec_reflect_iter": iters + bump, "exec_reflect_history": history}
        if refused:
            patch_summary = ("[the package change was not used: " + "; ".join(refused) + "] " + patch_summary)
        history.append({
            **({"for": _numeric.REPAIR_FOR} if numeric_turn else {}),
            "iter": attempt_no,
            "returncode": rc,
            "stderr_tail": exec_result.get("stderr_tail", "")[-400:],
            "patch_summary": patch_summary[:400 if refused else 200],
        })
        self._log.info(
            "[execute_reflect] iter=%d patch=%s",
            iters + 1, patch_summary[:120],
        )

        # Write the patched code to disk so the next `execute` picks it
        # up. We mirror the implement node's behavior.
        code_path = self.quest_root / "code" / script_name
        await self._gate_baseline(state)
        code_path.parent.mkdir(parents=True, exist_ok=True)
        parent_sha = _attempts._file_sha(code_path)
        package_before = self._package_snapshot() if repair_simulation else {}
        code_path.write_text(new_code, encoding="utf-8")
        package_fixed: dict[str, str] = {}
        if not repair_simulation and not repair_submit and (fault := await self._gate_refusal(state)):
            # experiment.py alone does not reach the simulation's checks below: the analysis gate is its check.
            code_path.write_text(script_code, encoding="utf-8")
            self._log.warning("[execute_reflect] the repair of %s is not used: %s; keeping it as written", script_name, fault)
            await self._keep_refused_repair("execute_reflect", script_name, new_code, fault)
            history[-1]["patch_summary"] = (f"[the repair was not used and {script_name} is as it was: {fault}] "
                                            + str(history[-1].get("patch_summary") or ""))[:400]
            return {**spent, "exec_reflect_iter": iters + bump, "exec_reflect_history": history}
        if repair_simulation:
            code_dir = self.quest_root / "code"
            package_fixed = {rel: _attempts._sha((code_dir / rel).read_bytes())
                             for rel in self._apply_package_repair(state, parsed, node="execute_reflect")
                             if (code_dir / rel).is_file()}
            # The function FI calls for the study must still give the study's results (FI calls the repaired code once).
            fault = await self._simulation_repair_fault(state, new_code)
            if fault:
                code_path.write_text(script_code, encoding="utf-8")
                self._restore_package(package_before)
                self._log.warning("[execute_reflect] the repair of %s is not used: %s; keeping it as written",
                                  script_name, fault)
                await self._keep_refused_repair("execute_reflect", script_name, new_code, fault)
                history[-1]["patch_summary"] = (f"[the repair was not used and {script_name} is as it was: {fault}] "
                                                + str(history[-1].get("patch_summary") or ""))[:400]
                return {**spent, "exec_reflect_iter": iters + bump, "exec_reflect_history": history}
        self._gate_commit(state)  # kept: the analysis gate's baseline
        if getattr(self, "_code_runs", None) is not None:  # the commit of this repair says why (from this record)
            _tail = [ln.strip() for ln in str(exec_result.get("stderr_tail", "")).splitlines() if ln.strip()]
            _fault = str(exec_result.get("directive") or "") or (_tail[-1] if _tail else "") or (
                str(overlaps[0]) if overlaps else "")
            self._code_runs.describe(_changelog.detail(
                stage="execute_reflect", attempt=attempt_no, fault=_fault,
                functions=_changelog.changed_functions(script_code, new_code), patch=patch_summary))
        self._last_ledger_id = self._record(_attempts.LEDGER, lambda: {
            "kind": "repair", "script": code_path.name, "attempt": attempt_no,
            "parent_sha256": parent_sha, "sha256": _attempts._sha(new_code.encode("utf-8")),
            "summary": patch_summary[:300],
            # A fix in the model's package: the files it rewrote, by their new hash.
            **({"package_files": package_fixed} if package_fixed else {}),
        }) or getattr(self, "_last_ledger_id", None)

        patch: QuestState = {
            **spent,
            "exec_reflect_iter": iters + bump,
            "exec_reflect_history": history,
            "exec_patch_pending": True,
        }
        if not repair_simulation and not repair_submit and not repair_body:
            # ``code`` is the script that prints RESULT_JSON, which is the analysis when
            # there are two (FI's own experiment.py, unchanged, when FI writes the analysis).
            patch["code"] = new_code
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
            _skip_top = {"literature", "auto_collected", "results"}
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
            # Collected pages and files are material to mine, never instructions (core/source_text.py).
            sources=_source_text.fence(sources_text[:12000], "web pages and data files the quest collected"),
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
            # Captions are retrieved text: material to judge, never instructions (core/source_text.py).
            f"Topic: {topic[:400]}\n\nCandidate illustrative figures:\n"
            f"{_source_text.fence(lines, 'figure captions from papers and Wikimedia Commons')}\n\n"
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
        # Survey mode is a text-only path (no experiment, no data plots), so
        # illustrative images ARE the paper's only visuals — force web_figures
        # on for survey even when ``knowledge.fetch_web_figures`` is off (its
        # default), still honouring ``web_figures_max``. Other modes keep the
        # explicit opt-in.
        survey = bool(state.get("survey_mode_resolved"))
        if (not k.fetch_web_figures and not survey) or k.web_figures_max <= 0:
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
