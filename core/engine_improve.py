"""Methods of :class:`core.engine.Engine` for improving and speeding up the simulation.

``Engine`` inherits :class:`ImproveStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, Path, QuestState, _FRESH_SCRIPT, _ModelAnswerProblem, _REFERENCE_FORMS, _SEED_REPAIR_DIRECTIVE,
    _TRIAL_ORACLE_NOTE, _accepted, _attempts, _audit_log, _changelog, _code_project, _coerce_dep_list, _criteria,
    _declare_scope, _equation_tests, _format_clarify, _format_reflect_history, _forms, _frozen, _gate_receipt,
    _hidden, _improve, _in_run, _mc_error, _one_line, _optimise_search, _oracle, _oracle_challenge, _oracle_triage,
    _oracle_with_reader_flags, _pairs, _parse_implement_response, _parse_json_lenient, _phased, _plan, _plan_scope,
    _protocol, _refine, _repair_reply_json, _replicate_env, _review, _script_has_random_source,
    _script_reads_replicate_seed, _skeleton, _split_run, _stop_power, _trial_runner, _unseeded_rng_calls,
    _unseeded_rng_directive, _with_newline, asyncio, hashlib, json, re, shutil, unseeded_rng_calls,
)


class ImproveStageMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    async def _node_improve(self, state: QuestState) -> QuestState:
        """After a run with a result: change the simulation one step at a time against the protocol's checks of
        correctness (core/criteria.py), keeping a version only when no check got worse beyond its own tolerance
        (core/improve.py). A passthrough when there is nothing to do; the log says why."""
        record = _improve.load(self.quest_root)
        # The last full run's row: a round's own row (with its ``improve`` block) is never the starting point.
        runs = [r for r in _criteria.history(self.quest_root) if not r.get("improve")]
        last = runs[-1] if runs else {}
        unfinished = bool(record) and (record.get("blocked") or not record.get("stopped"))
        if unfinished and not state.get("improve_rerun") and record.get("baseline_n") != last.get("n"):
            # A stop or a loop cut short over code that has run again since (a rerun from an earlier step): it is not
            # this code's, and nothing it kept is put back.
            record.update(blocked=False, stopped=record.get("stopped") or "set aside: the code ran again from an earlier "
                                                                           "step before the loop finished")
            self._improve_save(record)
            _improve.clear_own_dir(self.quest_root, "raw", fi_real=getattr(self, "_fi_real", None))  # that code's trial record
            self._log.info("[improve] an earlier loop's record belongs to code that has run again since; set aside")
        if record.get("blocked"):
            return await self._improve_resume_after_block(state, record)
        if state.get("improve_rerun"):
            return await self._improve_after_rerun(state, record)
        spent = 0
        if record.get("started") and not record.get("stopped"):
            # A loop cut short (the process stopped mid-round): the version the last full run used and FI's record of
            # its trials go back before anything else, so a half-finished round can never stand in for them. Its rounds
            # stay spent.
            spent = await self._improve_put_back_first(record)
        done: QuestState = {"improve_rerun": False}
        if spent > int(state.get("improve_rounds_used") or 0):
            state = {**state, "improve_rounds_used": spent}
            done["improve_rounds_used"] = spent
        why = self._improve_skip(state)
        protocol = _frozen.protocol_of(self.quest_root) or self._draft_protocol(state)
        # A known-answer check FI could not confirm (its expected value may be the wrong one) is never optimised against:
        # a change that moved the simulation towards it would bend correct code to a wrong number.
        unconfirmed = self._unconfirmed_checks()
        dropped: set[str] = set()
        if unconfirmed and isinstance(protocol, dict) and isinstance(protocol.get("criteria"), list):
            dropped = {str(c.get("name") or "") for c in protocol["criteria"]
                       if isinstance(c, dict) and str(c.get("oracle") or "") in unconfirmed}
            if dropped:
                self._log.info("[improve] not optimising against the unconfirmed check(s) %s",
                               ", ".join(repr(n) for n in sorted(unconfirmed)))
                protocol = {**protocol, "criteria": [c for c in protocol["criteria"]
                                                     if not (isinstance(c, dict) and str(c.get("name") or "") in dropped)]}
        if why is None and not (isinstance(protocol, dict) and _criteria.countable(protocol)):
            why = ("every check of correctness it has rests on a known-answer check FI could not confirm" if dropped
                   else "the plan has no check of correctness FI measures itself (`criteria` in the protocol)")
        # (A criterion's row in the last run's record carries its name, not its check's.)
        baseline = [r for r in last.get("criteria") or [] if isinstance(r, dict) and str(r.get("name") or "") not in dropped]
        if why is None and not _improve.counted(baseline):
            why = "no check of correctness was measured by FI in this run"
        if why is None and _improve.all_met(baseline):
            why = "every check of correctness is already met"
        if why is None and (noise := _improve.only_noise(baseline)):
            why = noise
        if why is not None:
            self._log.info("[improve] no change to the simulation: %s", why)
            return done
        with self._hold_code_runs():  # each round commits the version it ran, with its values
            return await self._improve_loop(state, protocol, baseline, baseline_n=last.get("n"), earlier=record)

    def _hold_code_runs(self) -> Any:
        """While a step commits each version it tries with its own words, the commit before each run waits
        (code_project.CodeRuns). An engine built without one (a test's stand-in) holds nothing."""
        import contextlib

        runs = getattr(self, "_code_runs", None)
        return runs.held() if runs is not None else contextlib.nullcontext()

    def _improve_undo_unfinished(self) -> None:
        """Before a rerun from a step moves the loop's record aside: a loop cut short left a round's version in code/,
        which the rerun would otherwise run as the study's code. The version the last full run used goes back first."""
        record = _improve.load(self.quest_root)
        if not record.get("started") or record.get("stopped") or record.get("blocked"):
            return
        if record.get("baseline_n") != self._improve_last_run_n():
            return  # a loop over code that has run again since: nothing of it is put back
        original = _improve.load_snapshot(self.quest_root, "original", fi_real=getattr(self, "_fi_real", None))
        if original is None:
            return
        _improve.restore(self.quest_root, original)
        if record.get("raw_saved"):
            self._improve_put_back_raw()
        record.update(stopped=_improve.STOP_CUT_SHORT, ended=_improve._now())
        self._improve_save(record)
        _code_project.record_change(self.quest_root, "improve: the loop was cut short; back to the version the last "
                                    "full run used", category="changed",
                                    results=_changelog.put_back("the version the last full run used"),
                                    undo_pending="undone before a run finished (the improve loop was cut short)",
                                    log=self._log)
        self._log.warning("[improve] a loop cut short left a changed version in code/; the version the last full run "
                          "used is back before the rerun")

    def _improve_last_run_n(self) -> Any:
        runs = [r for r in _criteria.history(self.quest_root) if not r.get("improve")]
        return runs[-1].get("n") if runs else None

    def _improve_save(self, record: dict[str, Any]) -> None:
        if not _improve.save(self.quest_root, record):
            self._log.warning("[improve] could not write %s; the loop's record on disk is out of date", _improve.RECORD)

    def _improve_fresh_run(self) -> QuestState:
        """What a run of a changed version starts with: the repair counters of a new script, and no cached trials (a
        kept change to a helper module leaves simulate.py's text, and so the key of the trials already run, unchanged)."""
        (self.quest_root / _trial_runner.RUN_RECORD).unlink(missing_ok=True)
        return dict(_FRESH_SCRIPT)

    def _improve_raw_files(self) -> list[Path]:
        root = self.quest_root
        return [root / _trial_runner.RAW_DIRNAME / _trial_runner.LEDGER_NAME,
                root / _trial_runner.RAW_DIRNAME / _trial_runner.SUMMARY_NAME, root / _trial_runner.RUN_RECORD]

    def _improve_save_raw(self) -> None:
        """FI's record of the last full run's trials, copied to ``.fi/improve/raw/`` before a round runs trials of its
        own (on disk, so a loop cut short can still put it back)."""
        _improve.clear_own_dir(self.quest_root, "raw", fi_real=getattr(self, "_fi_real", None))
        folder = _improve.own_dir(self.quest_root, "raw", fi_real=getattr(self, "_fi_real", None))
        if folder is None:
            raise OSError(f"the loop's copy of the trial record ({self.quest_root / _improve.SNAPSHOTS / 'raw'}) is "
                          "reached through a link that could not be removed")
        folder.mkdir(parents=True, exist_ok=True)
        kept: dict[str, bool] = {}
        for i, path in enumerate(self._improve_raw_files()):
            kept[str(i)] = path.is_file()
            if path.is_file():
                shutil.copyfile(path, folder / f"{i}.bin")
        (folder / "kept.json").write_text(json.dumps(kept), encoding="utf-8")

    def _improve_put_back_raw(self, saved: dict[Path, bytes | None] | None = None) -> None:
        """FI's record of the last full run's trials back in place: from ``saved`` (the bytes held in memory while the
        loop ran, which no round's run can reach) or, for a loop cut short, from the copy on disk."""
        if saved is None:
            folder = _improve.own_dir(self.quest_root, "raw", clear_links=False, fi_real=getattr(self, "_fi_real", None))
            if folder is None:
                return  # reached through a link: nothing is read from it
            try:
                kept = json.loads((folder / "kept.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            saved = {}
            for i, path in enumerate(self._improve_raw_files()):
                try:
                    saved[path] = (folder / f"{i}.bin").read_bytes() if kept.get(str(i)) else None
                except OSError:
                    return
        for path, data in saved.items():
            try:
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
            except OSError:
                pass
        _improve.clear_own_dir(self.quest_root, "raw", fi_real=getattr(self, "_fi_real", None))

    async def _improve_put_back_first(self, record: dict[str, Any]) -> int:
        """Undo a loop cut short; returns the rounds it had spent in this quest."""
        original = _improve.load_snapshot(self.quest_root, "original", fi_real=getattr(self, "_fi_real", None))
        if original is not None:
            _improve.restore(self.quest_root, original)
        if record.get("raw_saved"):
            self._improve_put_back_raw()
        spent = int(record.get("rounds_used_before") or 0) + len(record.get("rounds") or [])
        record.update(stopped=_improve.STOP_CUT_SHORT, ended=_improve._now(), rerun=False, best_round=0,
                      rounds_used=spent)
        self._improve_save(record)
        await asyncio.to_thread(
            _code_project.record_change, self.quest_root,
            "improve: the loop was cut short; back to the version the last full run used", category="changed",
            results=_changelog.put_back("the version the last full run used"),
            undo_pending="undone before a run finished (the improve loop was cut short)", log=self._log)
        self._log.warning("[improve] an earlier improve loop was cut short: the version the last full run used, and FI's "
                          "record of its trials, are back before the loop starts again (%d round(s) stay spent)", spent)
        return spent

    async def _improve_measure(self, protocol: dict[str, Any], oracles: list[dict[str, Any]],
                               trial_names: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
        """Each criterion of the version in ``code/``, computed by FI: the known-answer cases the criteria name, run by
        the engine (the check's own record, needs/ORACLE_CHECK.json, is not touched), and every trial of the protocol
        when a criterion is about the trials. Never the script's own report."""
        root = self.quest_root
        py = self.executor.python_path(root)
        stride = max(1, int(self.config.engine.replicate_seed_stride))
        env = _replicate_env(self.__dict__.get("_last_exec_env"), 0, stride)
        if _phased.enabled(self.config):
            env = _phased.seed_env(root, env)
        # Each round's version is imported fresh: no compiled copy of an earlier version is read or left behind.
        env = {**env, "PYTHONDONTWRITEBYTECODE": "1"}
        _improve.forget_bytecode(root)
        simulate = f"code/{_split_run.SIMULATE_NAME}"
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
        judged: list[dict[str, Any]] = []
        series: dict[str, dict[str, list[float]]] = {}
        problems: list[str] = []
        why_missing: dict[str, str] = {}
        if oracles:
            try:
                checks, found, _timed_out = await _trial_runner.measure_oracles(
                    self.executor, py, root, simulate, oracles,
                    timeout_s=max(30, int(self.config.execution.timeout_s * self.config.engine.pilot_timeout_frac)),
                    env=env, thresholds=thresholds, case_env=dict(env), in_each_run=_in_run.of_study(protocol),
                    mean_names=_mc_error.mean_names(protocol), paired=_mc_error.paired(protocol, self.quest_root),
                    proportion_names=_mc_error.proportion_names(protocol),
                    repeated=_mc_error.repeats_trials(protocol), protocol=protocol, log=self._log,
                )
                problems += found
                judged = _oracle_with_reader_flags(
                    root, _oracle.judged(oracles, {"checks": checks, "engine_measured": True}))
            except Exception as e:  # noqa: BLE001 -- a round that cannot run is a round that measured nothing
                problems.append(f"the known-answer cases could not be run: {e!r}"[:300])
                why_missing["oracle"] = "the known-answer cases could not be run"
        if trial_names:
            grid = _optimise_search.study_grid(protocol)
            entries = _trial_runner.entries(root / simulate)
            try:
                run = await _trial_runner.run_trials(
                    self.executor, py, root, simulate, grid,
                    runs_per_setting=int(protocol.get("runs_per_setting") or 1),
                    base_seed=int(env.get("FI_REPLICATE_SEED") or 0), deterministic="run_trial" not in entries,
                    timeout_s=self.config.execution.timeout_s, env=env, thresholds=thresholds,
                    paired=_mc_error.paired(protocol, self.quest_root),
                    in_each_run=_in_run.of_study(protocol), log=self._log,
                    at_once=_trial_runner.settings_at_once(self.config.execution.settings_at_once),
                )
                series = _improve.series_of(run, trial_names)
                if run.ok_trials == 0:
                    problems.append("no trial ran successfully")
            except Exception as e:  # noqa: BLE001
                problems.append(f"the trials could not be run: {e!r}"[:300])
                why_missing["trials"] = "the trials could not be run"
        rows = _criteria.evaluate(_criteria.declared(protocol), judged=judged, series=series, why_missing=why_missing)
        return rows, problems

    async def _improve_loop(self, state: QuestState, protocol: dict[str, Any], baseline: list[dict[str, Any]], *,
                            baseline_n: Any = None, earlier: dict[str, Any] | None = None) -> QuestState:
        root = self.quest_root
        await self._gate_baseline(state)
        limit = int(self.config.engine.improve_rounds)
        used = int(state.get("improve_rounds_used") or 0)
        research = getattr(self.config, "rigor_profile", "default") == "research"
        countable = _criteria.countable(protocol)
        oracle_names = {c["oracle"] for c in countable if c.get("oracle")}
        oracles = [o for o in _oracle.declared(protocol) if str(o["name"]).strip() in oracle_names]
        trial_names = {c["trials"] for c in countable if c.get("trials")}
        # What an edit must not write in (the values the checks expect, the criteria's targets), what it must not change
        # the computation of (the numbers the checks read), and what it must not test for (the checks' own settings).
        expected, measures, settings = _improve.guards(protocol, oracles, countable, trial_names)
        original = _improve.snapshot(root)
        _improve.save_snapshot(root, "original", original, fi_real=getattr(self, "_fi_real", None))
        _improve.save_snapshot(root, "best", original, fi_real=getattr(self, "_fi_real", None))
        best: dict[str, Any] = {"round": 0, "rows": baseline, "files": dict(original)}
        record: dict[str, Any] = {
            "started": _improve._now(), "limit": limit, "rounds_used_before": used, "rounds": [], "best_round": 0,
            "research": research, "baseline": _improve.values_line(baseline), "baseline_n": baseline_n,
            # The loop's earlier passes in this quest (a redesign, a loop cut short), for the paper's note.
            "earlier": ([*(earlier.get("earlier") or []), {k: v for k, v in earlier.items() if k != "earlier"}]
                        if earlier and earlier.get("stopped") else []),
            # One hash per result: enough to say later whether the study's results changed, never their values.
            "headline_before": _improve.headline_digest(state.get("result_json")),
        }
        self._improve_save(record)
        tried = {_improve.fingerprint(original)}
        # FI's record of the first run's trials: held in memory (what is put back) and on disk (for a loop cut short).
        raw_saved: dict[Path, bytes | None] | None = None
        if trial_names:
            raw_saved = {p: (p.read_bytes() if p.is_file() else None) for p in self._improve_raw_files()}
            self._improve_save_raw()
            record["raw_saved"] = True
            self._improve_save(record)
        stopped, calls = _improve.STOP_BUDGET, 0
        self._log.info(
            "[improve] %s; changing the simulation one step at a time, keeping a change only when no check gets worse "
            "beyond its own tolerance (%d round(s) left of engine.improve_rounds: %d)",
            _improve.values_line(baseline), limit - used, limit,
        )
        try:
            while used < limit:
                used += 1
                self._progress(f"[improve] Changing the simulation to meet its checks of correctness (round {used} of "
                               f"{limit}).")
                files = _improve.snapshot(root)
                allowed = _improve.editable(root)
                prompt = self._prompts["improve"].substitute(
                    topic_block=self._topic_block(),
                    criteria_block=_improve.criteria_block(countable, protocol),
                    values_block="\n".join(f"- {part}" for part in _improve.values_line(best["rows"]).split("; ")),
                    history_block=_improve.history_block(record["rounds"]),
                    editable=", ".join(allowed) or "(none)",
                    code_block=_improve.code_block(files, allowed),
                    rounds_left=f"This is round {used} of {limit}.",
                )
                prompt = _skeleton.repair_note(files.get(_split_run.SIMULATE_NAME, "")) + prompt
                entry: dict[str, Any] = {"round": used, "at": _improve._now()}
                record["rounds"].append(entry)
                self._improve_save(record)  # counted before the model is asked: a stop now still spends it
                try:
                    reply = await self._chat(prompt, node="improve")
                    calls += 1
                except _ModelAnswerProblem:
                    raise
                except Exception as e:  # noqa: BLE001 -- no answer is no change
                    entry.update(outcome="not asked", reason=_improve.one_line(repr(e), 200))
                    self._log.warning("[improve] round %d: the model could not be asked for a change: %r", used, e)
                    stopped = _improve.STOP_NO_MODEL
                    break
                edit, why = _improve.parse_edit(_parse_json_lenient(reply, node="improve", want=_improve.EDIT_KEYS))
                new_text = None
                if edit is not None:
                    new_text, why = _improve.check_edit(files, edit, allowed, tried=tried, expected=expected,
                                                        measures=measures, settings=settings)
                if edit is None or new_text is None:
                    entry.update(outcome="refused", reason=why, why=edit.why if edit else "")
                    self._log.warning("[improve] round %d: the change was refused before it ran: %s", used, why)
                    self._audit("check_result", check="improve_round", status="refused", summary=why[:300])
                    await asyncio.to_thread(
                        _code_project.record_note, root,
                        f"improve round {used} refused before it ran ({why}); nothing was changed", log=self._log)
                    self._improve_save(record)
                    continue
                name = _improve.edited_path(edit)  # `simulate.py`, or `<package>/model.py` for the model's package
                candidate = {**files, name: new_text}
                tried.add(_improve.fingerprint(candidate))
                entry.update(why=edit.why, file=name, diff=_improve.diff_text(edit), ran=True)
                _improve.restore(root, candidate)
                saved = _improve.guard_bytes(root)
                tree = _improve.tree_bytes(root)  # code/ as the round starts, git's settings and hooks included
                before = _improve.guard_hashes(root)
                rows, problems = await self._improve_measure(protocol, oracles, trial_names)
                touched = _improve.changed(before, _improve.guard_hashes(root))
                if touched:
                    # The run wrote into what FI judges by (the protocol, a record, FI's own code) or into its own
                    # source: nothing it measured is believed, and no further round is run. The code as the run left it
                    # is committed first, as the record of what was tried.
                    shown = ", ".join(touched[:8]) + (f" and {len(touched) - 8} more" if len(touched) > 8 else "")
                    # Whatever the run wrote into code/ (git's settings and hooks, links included) goes before any
                    # commit; a history the run replaced is not committed into at all.
                    history_ok = _improve.put_back_tree(root, tree)
                    if history_ok:
                        await asyncio.to_thread(
                            _code_project.record_change, root,
                            f"improve round {used} aborted ({edit.why}): its run changed {shown}, which FI judges the "
                            "code by (the change it ran is committed here; what the run wrote is not)",
                            category="changed",
                            results=_changelog.not_measured("this round's run changed what FI judges the code by, so "
                                                            "nothing it measured is believed; this version was not kept"),
                            log=self._log)
                    _improve.put_back(root, saved)
                    _improve.restore(root, best["files"])
                    # The loop's own copies, from memory: the run may have changed them too.
                    try:
                        _improve.save_snapshot(root, "original", original, fi_real=getattr(self, "_fi_real", None))
                        _improve.save_snapshot(root, "best", best["files"], fi_real=getattr(self, "_fi_real", None))
                    except OSError as e:  # the loop stops here all the same; a resume then never uses those copies
                        self._log.warning("[improve] could not write the loop's copies back: %s", e)
                    entry.update(outcome="aborted", reason="its run changed " + shown)
                    own = [t for t in touched if t.startswith("FI's own")]
                    self._log.warning(
                        "[improve] round %d aborted: its run changed %s, which FI judges the code by; the quest's records "
                        "and the version kept so far are back, and no more changes are tried%s", used, shown,
                        (f". FI's own files ({', '.join(own)}) were changed on this machine and cannot be put back by "
                         "FI: reinstall FI before trusting any quest") if own else "")
                    self._audit("check_result", check="improve_round", status="fail",
                                summary=f"round {used} aborted: its run changed {shown}"[:300])
                    if history_ok:
                        back = f"round {best['round']}'s version" if best["round"] else "the first version"
                        await asyncio.to_thread(
                            _code_project.record_change, root, f"improve round {used} aborted: back to {back}",
                            category="changed", results=_changelog.put_back(back, ran_in_full=not best["round"]),
                            log=self._log)
                    else:
                        self._log.warning("[improve] the run replaced code/'s git history (code/.git); FI makes no commit "
                                          "into it")
                    stopped = _improve.STOP_TAMPERED
                    break
                comparison = _improve.compare(best["rows"], rows)
                verdict = _improve.verdict(comparison)
                values = _improve.values_line(rows, comparison)
                kept = bool(verdict["best"])
                study_fault = ""
                if kept:
                    # A version that meets the checks of correctness is kept only if the function FI calls for the study still
                    # returns the study's results (a checks-only simulation can satisfy every check).
                    study_fault = await self._simulation_repair_fault(
                        state, (root / "code" / _split_run.SIMULATE_NAME).read_text(encoding="utf-8"))
                    if study_fault:
                        kept = False
                        self._log.warning("[improve] round %d is not kept: %s", used, study_fault)
                entry.update(values=values, better=verdict["better"], worse=verdict["worse"], broken=verdict["broken"])
                if problems:
                    entry["problems"] = problems[:5]
                # The commit of the version that ran, named with its values. A version kept waits for the full run after
                # the loop to say whether the study's results changed; one not kept never runs in full.
                await asyncio.to_thread(
                    _code_project.record_change, root,
                    f"improve round {used}: {edit.why} | {values} | "
                    + ("kept as the best version so far" if kept else "not kept"), category="changed",
                    results=None if kept else _changelog.round_not_kept(best["rows"], rows), log=self._log)
                commit, _dirty = await asyncio.to_thread(_code_project.head, root)
                frozen = _frozen.load(root)
                _criteria.record(
                    root, run=_frozen.run_id(root), code_commit=commit, results=rows,
                    protocol_version=int(frozen.get("version", 1) or 1) if frozen else None,
                    protocol_sha256=str(frozen.get("sha256")) if frozen else None,
                    improve={"round": used, "kept": kept, "compared_with_round": best["round"],
                             "better": verdict["better"], "worse": verdict["worse"], "broken": verdict["broken"]},
                )
                if kept:
                    self._gate_commit(state)
                    best = {"round": used, "rows": rows, "files": candidate}
                    _improve.save_snapshot(root, "best", candidate, fi_real=getattr(self, "_fi_real", None))
                    record["best_round"] = used
                    entry["outcome"] = "kept"
                    self._log.info("[improve] round %d kept (%s): %s", used, edit.why, values)
                    self._audit("check_result", check="improve_round", status="pass",
                                summary=f"round {used} kept: {values}"[:300])
                else:
                    back = f"round {best['round']}'s version" if best["round"] else "the first version"
                    if study_fault:
                        reason = study_fault
                    elif verdict["broken"]:
                        reason = ("the change stopped the simulation from running its checks"
                                  + (f" ({_improve.one_line('; '.join(problems), 200)})" if problems else ""))
                    elif verdict["worse"]:
                        reason = f"it made {', '.join(verdict['worse'])} worse by more than the tolerance"
                    else:
                        reason = "it made no check better by more than its tolerance"
                    entry.update(outcome="not kept", reason=reason)
                    _improve.restore(root, best["files"])
                    await asyncio.to_thread(
                        _code_project.record_change, root,
                        f"improve round {used} not kept ({reason}): back to {back}", category="changed",
                        results=_changelog.put_back(back, ran_in_full=not best["round"]), log=self._log)
                    if verdict["worse"]:
                        self._log.warning(
                            "[improve] round %d made %s worse by more than the tolerance (%s); %s is kept%s", used,
                            ", ".join(verdict["worse"]), values, back,
                            ", and the quest stops for you because it is set up for research" if research else
                            " (a warning only: rigor_profile: research would stop the quest here)")
                        self._audit("check_result", check="improve_regression", status="fail" if research else "warn",
                                    summary=f"round {used} made {', '.join(verdict['worse'])} worse: {values}"[:300])
                    else:
                        self._log.info("[improve] round %d not kept: %s (%s)", used, reason, values)
                        self._audit("check_result", check="improve_round", status="not_kept",
                                    summary=f"round {used}: {reason}"[:300])
                self._improve_save(record)
                if verdict["worse"] and research:
                    self._improve_stop_for_regression(record, entry, best["round"], used)
                if kept and _improve.all_met(best["rows"]):
                    stopped = _improve.STOP_ALL_MET
                    break
                if not verdict["better"]:
                    stopped = _improve.STOP_PLATEAU
                    break
        finally:
            if raw_saved is not None:  # FI's record of the trials of the run the paper may still use
                self._improve_put_back_raw(raw_saved)
        _improve.restore(root, best["files"])
        rerun = best["round"] > 0
        record.update(stopped=stopped, ended=_improve._now(), rounds_used=used, model_calls=calls, rerun=rerun)
        if not rerun:
            record["result_n"] = baseline_n  # the results the paper is written from are the first run's
        self._improve_save(record)
        kept_text = f"the version from round {best['round']}" if rerun else "the first version"
        self._log.info(
            "[improve] stopped: %s. Kept %s (%s). %d of engine.improve_rounds: %d round(s) spent in this quest; %d model "
            "call(s) in this pass (FI has no spending limit to count them against; each one is in .fi/model_calls.jsonl "
            "under the step `improve`).", stopped, kept_text, _improve.values_line(best["rows"]), used, limit, calls)
        self._progress(f"[improve] Stopped: {stopped}. Kept {kept_text}"
                       + ("; running it once more in full." if rerun else "."))
        self._audit("check_result", check="improve_loop", status="ok",
                    summary=f"stopped: {stopped}; kept {kept_text}"[:300])
        return {**(self._improve_fresh_run() if rerun else {}), "improve_rounds_used": used, "improve_rerun": rerun,
                "improve_fell_back": False,
                "improve_summary": {"stopped": stopped, "best_round": best["round"], "rounds_used": used,
                                    "model_calls": calls, "summary": _improve.summary(record)}}

    def _improve_stop_for_regression(self, record: dict[str, Any], entry: dict[str, Any], best_round: int,
                                     used: int) -> None:
        """Under ``rigor_profile: research``, a change that made a check worse stops the quest for a person. The version
        kept so far is already back in code/; a resume goes on with it and makes no more changes in this pass."""
        back = f"round {best_round}'s version" if best_round else "the first version"
        record.update(blocked=True, stopped=_improve.STOP_REGRESSION, rounds_used=used, ended=_improve._now())
        self._improve_save(record)
        self._pause_for_contract(
            kind="improve",
            headline="a change to the simulation made a check of correctness worse",
            steps=[
                f"Round {entry['round']} changed the simulation ({entry.get('why')}) and made "
                f"{', '.join(entry.get('worse') or [])} worse by more than its own tolerance: {entry.get('values')}.",
                f"FI put back {back}: it is what is in code/ now. code/CHANGELOG.md and .fi/criteria_history.jsonl hold "
                "every round's values.",
                "Resume to go on with that version; FI makes no more changes to the simulation in this pass. If a check "
                "itself is wrong, the protocol is frozen: start a new quest whose plan states the right one.",
            ],
            problems=[f"round {entry['round']}: {', '.join(entry.get('worse') or [])} worse"],
        )

    async def _improve_resume_after_block(self, state: QuestState, record: dict[str, Any]) -> QuestState:
        """A resume after the stop for a check that got worse: no more changes; the version in code/ goes on (run once
        more in full when it is not the one the last full run used)."""
        original = _improve.load_snapshot(self.quest_root, "original", fi_real=getattr(self, "_fi_real", None))
        now = _improve.snapshot(self.quest_root)
        if original is not None and not any("/" in k for k in original) and any("/" in k for k in now):
            # A copy saved before the package was part of it cannot tell whether the package was edited during the
            # stop: the version in code/ is run once more in full rather than risk results that are not its own.
            rerun = True
        else:
            rerun = original is not None and now != original
        record.update(blocked=False, resumed=_improve._now(), rerun=rerun)
        if rerun:
            record["kept_from"] = ("round " + str(record["best_round"]) if record.get("best_round")
                                   else "edited by hand during the stop")
        else:
            record["result_n"] = record.get("baseline_n")
        self._improve_save(record)
        self._log.warning("[improve] resumed after the stop for a check that got worse: no more changes are made%s",
                          "; the version kept runs once more in full" if rerun else "")
        return {**(self._improve_fresh_run() if rerun else {}),
                "improve_rounds_used": int(record.get("rounds_used") or state.get("improve_rounds_used") or 0),
                "improve_rerun": rerun,
                "improve_summary": {"stopped": record.get("stopped"), "best_round": record.get("best_round"),
                                    "rounds_used": record.get("rounds_used"), "summary": _improve.summary(record)}}

    async def _improve_after_rerun(self, state: QuestState, record: dict[str, Any]) -> QuestState:
        """Back from the kept version's full run. If it produced no result, the first version is put back and run once
        more. Otherwise, whether the study's results changed is written in code/CHANGELOG.md, by name only: they were
        never what chose the version."""
        root = self.quest_root
        result = state.get("exec_result") or {}
        failed = self._run_has_no_result(state)
        if failed and not (record.get("fell_back") or state.get("improve_fell_back")):
            original = _improve.load_snapshot(root, "original", fi_real=getattr(self, "_fi_real", None))
            if original is not None:
                _improve.restore(root, original)
                record.update(fell_back=record.get("kept_from") or f"round {record.get('best_round')}", best_round=0)
                self._improve_save(record)
                self._log.warning("[improve] the kept version produced no result in its full run; the first version is "
                                  "back and runs once more")
                await asyncio.to_thread(
                    _code_project.record_change, root,
                    "improve: the kept version produced no result in its full run; the first version is back",
                    category="changed", results=_changelog.put_back("the first version, which the last full run used"),
                    undo_pending="undone: this version's full run produced no result, so the first version was put "
                                 "back", log=self._log)
                return {**self._improve_fresh_run(), "improve_rerun": True, "improve_fell_back": True}
        before = record.get("headline_before") if isinstance(record.get("headline_before"), dict) else {}
        after = _improve.headline_digest(state.get("result_json"))
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        record["headline_changed"] = changed
        record["result_n"] = self._improve_last_run_n()  # the results the paper is written from are this run's
        self._improve_save(record)
        if changed and not record.get("fell_back"):
            shown = ", ".join(changed[:12]) + (f" and {len(changed) - 12} more" if len(changed) > 12 else "")
            await asyncio.to_thread(
                _code_project.record_note, root,
                f"improve: the study's results changed after the kept change ({shown}). Reported here only as changed; "
                "they were not used to choose the version", log=self._log)
        self._log.info("[improve] the %s full run is done; the study's results %s (they did not choose the version)",
                       "first version's" if record.get("fell_back") else "kept version's",
                       f"changed: {', '.join(changed[:12])}" if changed else "did not change")
        return {"improve_rerun": False, "improve_fell_back": False}

    def _oracle_record_clear(self) -> None:
        """Remove a record this run did not write: when the gate does not run, an earlier run's verdicts must not reach the
        analysis or count as numbers this run computed."""
        try:
            (self.quest_root / "needs" / "ORACLE_CHECK.json").unlink(missing_ok=True)
        except OSError:
            pass

    def _oracle_record_read(self) -> dict[str, Any] | None:
        """``needs/ORACLE_CHECK.json`` as the last oracle gate wrote it, or ``None``."""
        try:
            record = json.loads((self.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    async def _rejudge_oracles_on_delivered_run(self, state: QuestState, seed_path: Path) -> list[str]:
        """After the main run: each known-answer check the gate's short side run could not decide for lack of time is judged
        again on the delivered run's own trials (``raw/trials.json``), when its case is a setting of that run and the code
        is the code that measured it (core/oracle_check.py::rejudge_on_delivered). The names judged again; the record is
        rewritten when one is, or when one is still not decided on those trials (its words then name them)."""
        record = self._oracle_record_read()
        if not record or record.get("status") != _accepted.WENT_ON:
            return []
        protocol = self._protocol_block(state)
        if not protocol:
            return []
        path = self.quest_root / _trial_runner.RAW_DIRNAME / _trial_runner.SUMMARY_NAME

        def read() -> Any:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None

        trials = await asyncio.to_thread(read)
        new, names, reworded = _oracle.rejudge_on_delivered(record, trials, protocol,
                                                            self._measuring_code_version(seed_path))
        if new is None:
            return []
        self._oracle_record(new)
        for n in names:  # no longer set aside: the delivered run decided it
            self._oracle_undecided.discard(n)
            self._oracle_set_aside.pop(n, None)
        if names:
            self._log.info("[oracle] judged again on the delivered run's own trials: %s", ", ".join(names))
        aside = new.get("set_aside") if isinstance(new.get("set_aside"), dict) else {}
        for n in reworded:  # still not decided: it stays set aside, and its reason now names the delivered trials
            if n in self._oracle_set_aside and n in aside:
                self._oracle_set_aside[n] = aside[n]
        if reworded:
            self._log.info("[oracle] still not decided on the delivered run's own trials: %s", ", ".join(reworded))
        return names

    def _judged_checks(self) -> list[dict[str, Any]]:
        """The engine's verdicts on the plan's checks of the last gate, with the checks that a second model's reading of the
        code flags marked (``core/code_review.py:flagged_checks``): the one list the note to the analysis and the criteria read."""
        return _oracle_with_reader_flags(self.quest_root, _oracle.last_judged(self._oracle_record_read()))

    def _oracle_record(self, payload: dict[str, Any]) -> None:
        try:
            path = self.quest_root / "needs" / "ORACLE_CHECK.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
            # A quest that went on although a check failed: the trace says who chose it (or that it was automatic).
            self._audit_check("oracle", path, status=str(payload.get("status")), problems=payload.get("problems"),
                              summary=str(payload.get("summary") or ""))
        except OSError:
            pass  # a record that cannot be written must never stop a quest

    def _restore_oracle_disputes(self, oracles: list[dict[str, Any]]) -> None:
        """Read back what the last run of the oracle gate found disputed (``needs/ORACLE_CHECK.json``): its proposals,
        and the checks whose expected value is disputed, for each check still exactly as it was then (its fingerprint:
        expected value, tolerance, case, measure, statement). A resume after a stop therefore never rewrites the script
        towards a value already disputed, and a check a person changed (or whose proposal they accepted) starts afresh."""
        prior = self._oracle_record_read() or {}
        prints = prior.get("fingerprints") if isinstance(prior.get("fingerprints"), dict) else {}
        now = {str(o.get("name") or "").strip(): _oracle_triage.fingerprint(o) for o in oracles}
        disputed = {str(n).strip() for n in prior.get("disputed") or []}
        for proposal in prior.get("proposed_changes") or []:
            name = str((proposal or {}).get("name") or "").strip() if isinstance(proposal, dict) else ""
            if not name or name not in now or prints.get(name) != now[name]:
                continue
            # Only the dispute comes back; a confirmation is never taken from a file (the next look confirms afresh).
            self._oracle_proposals[name] = {k: v for k, v in proposal.items() if k != "confirmed_by"}
            if name in disputed:
                self._oracle_disputed.add(name)
                self._log.info("[oracle] kept from the last run: the expected value of %r is disputed (proposed %s); the "
                               "script is not rewritten for it", name, proposal.get("expected"))

    async def _look_at_failing_checks(
        self, state: QuestState, py: Any, seed_path: Path, oracles: list[dict[str, Any]],
        run_oracles: list[dict[str, Any]], record: dict[str, Any], looked: set[str], *, protocol: dict[str, Any],
        timeout: int, case_env: dict[str, str],
    ) -> tuple[int, int]:
        """FI's own look at each check failing in ``record`` (one run of the gate) and not looked at yet, before a repair
        is spent on it (core/oracle_triage.py): its expected value worked out again by another model, never shown the
        measured value; then, when FI runs its case itself (the trial contract), the case at a smaller step (a
        deterministic check) or with more seeds (a random one). What each found goes into ``record["triage"]``; a
        recomputed value that disputes the plan's becomes a proposal, and a tolerance found tighter than the method's
        error or the noise sets the check aside from the repairs. Returns ``(model calls, extra runs)``."""
        by_name = {str(o["name"]).strip(): o for o in oracles}
        cases = {str(o["name"]).strip(): o for o in run_oracles}
        trial = bool(getattr(self, "_trial_mode", False))
        rel = seed_path.relative_to(self.quest_root).as_posix()
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
        calls = runs = 0
        entries: list[dict[str, Any]] = []
        for j in record.get("judged") or []:
            name = str(j.get("name") or "").strip()
            if name in self._oracle_undecided and not j.get("undecided"):
                # Set aside while its trials had not decided it; now they have (a later measurement adds to the stored
                # trials): a failure is a fault like any other again, and is looked at and repaired as one.
                self._oracle_undecided.discard(name)
                self._oracle_set_aside.pop(name, None)
                looked.discard(name)
            value = _oracle_triage.num(j.get("value"))
            if ((j.get("passed_by_engine") is not False and not j.get("undecided")) or value is None or name in looked
                    or name not in by_name or name in self._oracle_disputed or name in self._oracle_set_aside):
                continue
            looked.add(name)
            oracle = by_name[name]
            if j.get("undecided"):
                # The trials FI could run did not decide it (core/mc_error.py): not a fault to rewrite the script for,
                # and not a pass. It is set aside (still held by the repair guard: a repair aimed at another check may
                # not bend it) and goes on unconfirmed, said as not decided.
                found = _oracle_triage.trials_entry(oracle, j)
                entries.append(found)
                self._oracle_set_aside[name] = found["cause"]["text"]
                self._oracle_undecided.add(name)
                self._log.warning("[oracle] %s", found["cause"]["text"])
                continue
            slip = _oracle_triage.plan_slip(oracle)
            if slip is not None:
                # The plan's own working does not give the expected value it writes: the check is what is wrong, so
                # the script is not rewritten towards it (it still fails; the card says why in plain words). The
                # arithmetic alone never rewrites the check (``_may_correct``): it is set aside and goes on unconfirmed
                # unless another model, never shown the measured value, gets the same value.
                found = _oracle_triage.arithmetic_entry(oracle, slip)
                entries.append(found)
                self._oracle_set_aside[name] = found["cause"]["evidence"]
                # The value the plan's own working gives: a candidate for the correction (``_correct_expected_values``),
                # for a check whose number is the quantity itself (never a violation formula, see ``correctable``).
                if _oracle_triage.correctable(oracle):
                    self._oracle_proposals.setdefault(name, _oracle_triage.arithmetic_proposal(oracle, slip))
                self._log.warning("[oracle] %s", found["tried"])
            arith = self._oracle_proposals.get(name)
            if (arith or {}).get("source") == "arithmetic":
                arith.pop("confirmed_by", None)  # confirmed afresh by this look, or not at all
            # With a slip in the plan's working, the recheck never sees that working (it could repeat the misread step).
            shown = _oracle_triage.blind(oracle) if slip is not None else oracle
            computed = _oracle_triage.formula_value(oracle, protocol) if slip is None else None
            if computed is not None:
                # FI computed the expected value itself from the plan's formula: a model's arithmetic cannot dispute it.
                entry, called = _oracle_triage.formula_entry(oracle, computed, value), False
                self._log.info("[oracle] the expected value of %r is FI's own computation from the plan's formula (%s); no "
                               "model is asked to work it out again", name, _oracle.fmt_digits(computed))
            else:
                entry, called = await self._recompute_expected(state, shown, value)
            if slip is not None and called and not _oracle_triage.shows_working(shown):
                # Blind to the plan's working AND answered now (a kept answer is a file anyone could write): only
                # such a recheck can confirm the plan's arithmetic.
                entry = {**entry, "blind": True, "fingerprint": _oracle_triage.fingerprint(oracle)}
            calls += int(called)
            entries.append(entry)
            if (arith or {}).get("source") == "arithmetic" and (second := self._second_source(oracle, arith, entry)):
                # Two workings independent of each other give the same value: only then may FI correct the check.
                arith["confirmed_by"] = second
                self._log.warning("[oracle] another model (%s), never shown the measured value, gets %s for %r too: "
                                  "the plan's own arithmetic is confirmed", second["model"],
                                  _oracle.fmt_digits(second["value"]), name)
            if entry["verdict"] == "disputed" and (self._oracle_proposals.get(name) or {}).get("source") != "arithmetic":
                # (A value FI worked out from the plan's own arithmetic stays: it is deterministic, a model's is not.)
                proposal = {**_oracle_triage.recompute_proposal(oracle, entry),
                            **({"how_slip": True} if entry.get("how_slip") else {})}
                self._oracle_proposals[name] = proposal
                # Any disputing recheck keeps the script from being rewritten towards a value it disputes: a repair
                # aimed at a check that may be wrong can fit correct code to it, and that is the worse failure. A
                # recheck that is not independent (the plan's own model, or working that does not add up) never
                # corrects the check and never counts as evidence: the check goes on unconfirmed, the result exploratory.
                self._oracle_disputed.add(name)
                self._log.warning(
                    "[oracle] the expected value of %r is disputed: worked out again without the measured value it is %s "
                    "(the plan says %s, measured %s); the script is not rewritten for it%s", name,
                    _oracle.fmt_digits(entry["recomputed"]), _oracle.fmt_digits(entry["expected"]),
                    _oracle.fmt_digits(value),
                    " (one recheck alone never corrects a check: it goes on unconfirmed)"
                    if self._independent_value(proposal) else
                    " (the recheck is not independent, so the check is not corrected either: it goes on unconfirmed)")
            own = _oracle.case_of(cases.get(name) or oracle)
            if not trial or own is None or j.get("measured_by") != "engine":
                continue  # FI runs a case itself only under the trial contract, for a check it measured
            if (j.get("mc") or {}).get("read_as") == "mean":
                continue  # already decided on many trials, with their Monte Carlo error: more seeds would add nothing
            case, measure = own
            # Per check: which function FI calls depends on the check (a one-trial check, or a number of the setting).
            try:
                random_trials = _trial_runner.oracle_entry_for(seed_path, oracle) == "run_trial" and (
                    "run_trial" in _trial_runner.entries(seed_path))
            except (ValueError, OSError):
                random_trials = False

            async def at(cell: dict[str, Any], trial_no: int = 0) -> float | None:
                try:
                    values, _why = await _trial_runner.run_case(
                        self.executor, py, self.quest_root, rel, cell=cell, timeout_s=timeout, env=case_env,
                        thresholds=thresholds, trial=trial_no, oracle=oracle, in_each_run=_in_run.of_study(protocol),
                        paired=_mc_error.paired(protocol, self.quest_root), log=self._log)
                except Exception as e:  # noqa: BLE001 -- a look that could not be had never stops a quest
                    self._log.warning("[oracle] an extra run of %r's case could not be made: %r", name, e)
                    return None
                return _oracle_triage.value_of(values, measure)

            found: dict[str, Any] | None = None
            if random_trials:
                more = [await at(case, t) for t in range(1, _oracle_triage.EXTRA_SEEDS + 1)]
                runs += _oracle_triage.EXTRA_SEEDS
                result = _oracle_triage.seeds_verdict(oracle, [value, *[v for v in more if v is not None]])
                if result is not None:
                    found = _oracle_triage.seeds_entry(oracle, result)
            elif _oracle.kind_of(oracle) != "convergence_rate" and (step := _oracle_triage.step_of(case)) is not None:
                key, size, counts = step
                order = _oracle_triage.num(oracle.get("order"))
                levels = 2  # three values give the order the scheme really shows
                smaller = [_oracle_triage.refined(case, key, counts, k) for k in range(1, levels + 1)]
                values = [value]
                for cell in smaller:
                    values.append(await at(cell))
                    runs += 1
                    if values[-1] is None:
                        break
                if None not in values:
                    result = _oracle_triage.step_verdict(oracle, values, order=order)
                    if result is not None:
                        found = _oracle_triage.step_entry(oracle, key, [size, *[float(c[key]) for c in smaller]], result)
            if found is None:
                continue
            entries.append(found)
            if found["points_to"] == "tolerance" and found.get("cause"):
                self._oracle_set_aside[name] = found["cause"]["text"]
                if found["kind"] == "seeds":
                    self._oracle_noisy.add(name)
            self._log.warning("[oracle] %s %s", found["tried"], (found.get("cause") or {}).get("text") or "")
        if entries:
            record.setdefault("triage", []).extend(entries)
        return calls, runs

    async def _answer_challenges(self, state: QuestState, oracles: list[dict[str, Any]], attempt: dict[str, Any]) -> None:
        """The plan's model answers each new question the code put to the plan's working, and a third model judges when
        it stands by it (core/oracle_challenge.py); neither is shown the measured value, and FI computes every formula
        itself. Each exchange is asked once per version of the check and the question, kept in ``.fi/oracle_triage.json``
        so a resume never asks again; what it decided goes into the attempt's ``triage`` and run.log."""
        declared = {str(o.get("name") or "").strip(): o for o in oracles}
        kept = _oracle_triage.read(self.fi_dir)
        record = kept.get(_oracle_challenge.RECORD_KEY) if isinstance(kept.get(_oracle_challenge.RECORD_KEY), dict) else {}
        topic = str(state.get("topic") or self.config.topic or "")
        frozen = _frozen.load(self.quest_root) is not None
        judge = self._model_for_node(_review.NODE) or self.config.provider.model or "the provider's default model"
        writers = {self._model_for_node(n) or self.config.provider.model or "" for n in _review.WRITER_NODES}
        changed = False
        for name, challenge in list(self._oracle_challenges.items()):
            oracle = declared.get(name)
            if oracle is None or challenge.get("outcome"):
                continue
            fp = _oracle_triage.fingerprint(oracle)
            key = fp + "|" + hashlib.sha256(challenge["formula"].encode("utf-8")).hexdigest()[:16]
            saved = record.get(key) if isinstance(record.get(key), dict) else None
            if saved is not None:
                answer, verdict = saved.get("answer") or {}, saved.get("verdict")
            else:
                self._progress("Asking the plan's model to answer the code's objection to a known answer")
                try:
                    reply = await self._chat(_oracle_challenge.plan_prompt(
                        self._prompts["oracle_challenge_plan"], topic=topic, oracle=oracle, challenge=challenge),
                        node=_oracle_challenge.PLAN_NODE)
                    answer = _oracle_challenge.parse_plan(_parse_json_lenient(reply, node=_oracle_challenge.PLAN_NODE))
                except Exception as e:  # noqa: BLE001 -- an answer that could not be had never stops a quest
                    answer = _oracle_challenge.parse_plan(None)
                    answer["why"] = f"the plan's model could not be asked ({str(e)[:160] or type(e).__name__})"
                    answer["error"] = True
                verdict = None
                if _oracle_challenge.settle(oracle, challenge, answer, None)["outcome"] != "corrected":
                    self._progress("Asking a third model to judge between the two workings")
                    try:
                        reply = await self._chat(_oracle_challenge.judge_prompt(
                            self._prompts["oracle_challenge_judge"], topic=topic, oracle=oracle, challenge=challenge,
                            answer=answer), node=_oracle_challenge.JUDGE_NODE)
                        verdict = _oracle_challenge.parse_judge(_parse_json_lenient(reply,
                                                                                    node=_oracle_challenge.JUDGE_NODE))
                    except Exception as e:  # noqa: BLE001
                        verdict = _oracle_challenge.parse_judge(None)
                        verdict["why"] = f"the judge could not be asked ({str(e)[:160] or type(e).__name__})"
                        verdict["error"] = True
                    # Whether the judge is a model that may have written the plan: said with its verdict, never hidden.
                    verdict["model"] = judge
                    verdict["same_model"] = ("" in writers or any(_review.same_model(judge, w) for w in writers))
                if not (answer.get("error") or (verdict or {}).get("error")):  # a failed call is asked again on resume
                    record[key] = {"name": name, "answer": answer, "verdict": verdict}
                    changed = True
            result = _oracle_challenge.settle(oracle, challenge, answer, verdict)
            if result["outcome"] == "unsettled":
                # The last referee is not a model's word: the number computed numerically from the definitions.
                numeric = record[key].get("numeric") if isinstance(record.get(key), dict) else None
                if not isinstance(numeric, dict):
                    numeric = await self._numeric_reference(state, oracle)
                    if key in record and numeric.get("value") is not None:  # a script that gave no value is tried again
                        record[key] = {**record[key], "numeric": numeric}
                        changed = True
                result = _oracle_challenge.settle_numeric(oracle, challenge, verdict, numeric.get("value"),
                                                          str(numeric.get("how") or ""), result)
            if result["outcome"] in ("corrected", "both") and frozen:
                result = {"outcome": "unsettled", "by": "", "text": result["text"] + "; the protocol is frozen, so FI "
                          "does not correct the check: it goes on unconfirmed"}
            if result["outcome"] == "corrected" and (verdict or {}).get("same_model") and result["by"] == "the judge":
                result["text"] += " (the judge is the model that wrote the plan, so this counts for less)"
            challenge["outcome"] = result
            proposal = self._oracle_proposals.get(name) or {}
            if result["outcome"] in ("corrected", "both"):
                formula = challenge["formula"] if result["outcome"] == "corrected" else result["formula"]
                self._oracle_proposals[name] = {**proposal, "source": "challenge", "expected": result["expected"],
                                                "expected_formula": formula, "reason": result["text"],
                                                "confirmed_by": {"source": "challenge", "by": result["by"],
                                                                 "value": result["expected"], "fingerprint": fp}}
                self._log.warning("[oracle] '%s': %s; FI corrects the check's expected value to %s", name, result["text"],
                                  _oracle.fmt_digits(result["expected"]))
                if result["outcome"] == "both":
                    # The code was wrong too: once the check holds the corrected value, the code is repaired towards it.
                    self._oracle_disputed.discard(name)
                    self._oracle_challenge_hints[name] = result["text"]
            else:
                self._oracle_disputed.discard(name)
                self._oracle_proposals.pop(name, None)
                if result["outcome"] == "code":
                    self._oracle_challenge_hints[name] = result["text"]
                    self._log.warning("[oracle] '%s': %s; the code is repaired once more, told why", name, result["text"])
                else:
                    self._oracle_set_aside[name] = result["text"]
                    self._log.warning("[oracle] %s; it goes on unconfirmed", result["text"])
            attempt.setdefault("triage", []).append({
                "kind": "challenge", "check": name, "outcome": result["outcome"], "tried": result["text"],
                **({"points_to": "check"} if result["outcome"] == "corrected" else {}),
                "code": {k: challenge[k] for k in ("step", "formula", "value", "reason") if k in challenge},
                "plan": answer, **({"judge": verdict} if verdict is not None else {})})
        if changed:
            _oracle_triage.write(self.fi_dir, {**_oracle_triage.read(self.fi_dir), _oracle_challenge.RECORD_KEY: record})

    async def _referee_failing_checks(self, state: QuestState, oracles: list[dict[str, Any]],
                                      judged: list[dict[str, Any]], protocol: Any, attempt: dict[str, Any]) -> bool:
        """Each check still failing when FI is about to go on (no repair left, or none to spend) whose expected value is
        the plan's (worked out or typed; not a published value) and that nothing has objected to yet: the number computed
        numerically from the definitions
        (:meth:`_numeric_reference`) decides which side is wrong (core/oracle_challenge.py::settle_referee). Once per check
        per run; each number kept in ``.fi/oracle_triage.json`` by the check's version, so a resume never asks again. A
        correction is only proposed here (``source`` ``numeric``) and made by :meth:`_correct_expected_values`, never once
        the protocol is frozen. Returns whether anything was proposed or a repair was told what to fix."""
        declared = {str(o.get("name") or "").strip(): o for o in oracles}
        kept = _oracle_triage.read(self.fi_dir)
        key = _oracle_challenge.REFEREE_KEY
        record = kept.get(key) if isinstance(kept.get(key), dict) else {}
        frozen = _frozen.load(self.quest_root) is not None
        done = self._corrections_read()
        # A check a challenge already put to the referee (possibly in a run before a resume): one number per check.
        exchanges = kept.get(_oracle_challenge.RECORD_KEY) if isinstance(kept.get(_oracle_challenge.RECORD_KEY), dict) else {}
        challenged = {str(e.get("name") or "") for e in exchanges.values() if isinstance(e, dict)}
        # The simulation's code and the script must not be one model's reading of the definitions: the same misreading
        # in both would agree, so with the same model the referee never corrects a check (it can still find the code
        # wrong, which only sends it to a repair).
        referee_model = self._model_for_node(_oracle_challenge.NUMERIC_NODE) or self.config.provider.model or ""
        coders = {self._model_for_node(n) or self.config.provider.model or ""
                  for n in ("implement", "implement_outline", "implement_seed", "implement_oracle", "execute_reflect")}
        same_coder = "" in coders or not referee_model or any(_review.same_model(referee_model, c) for c in coders)
        acted = changed = False
        for j in judged:
            name = str(j.get("name") or "").strip()
            measured = _oracle_triage.num(j.get("value"))
            oracle = declared.get(name)
            if (j.get("passed_by_engine") is not False or measured is None or oracle is None or name in done
                    or name in self._oracle_challenges or name in self._oracle_refereed or name in self._oracle_noisy
                    or name in self._oracle_undecided or name in self._oracle_set_aside or name in challenged):
                continue  # a check set aside for its method's error is not the script's, nor one already put to it
            plan_value = _oracle_challenge.referee_expected(oracle)
            if plan_value is None:
                continue  # a published value, or a check FI cannot judge
            self._oracle_refereed.add(name)
            fp = _oracle_triage.fingerprint(oracle)
            # Kept by what the script is shown (the model's equations, the check's words, case and measure), so a
            # plan changed since never reuses an old number.
            shown = _oracle_challenge.numeric_prompt(
                self._prompts["oracle_numeric_reference"], topic=str(state.get("topic") or self.config.topic or ""),
                oracle=oracle, protocol=self._protocol_block(state))
            cache = fp + "|" + hashlib.sha256(shown.encode("utf-8")).hexdigest()[:16]
            numeric = record.get(cache) if isinstance(record.get(cache), dict) else None
            if numeric is None:
                numeric = await self._numeric_reference(state, oracle)
                if numeric.get("value") is not None:  # a script that gave no value is tried again on resume
                    record[cache] = {"name": name, **numeric}
                    changed = True
            result = _oracle_challenge.settle_referee(oracle, measured, plan_value, numeric.get("value"),
                                                      str(numeric.get("how") or ""))
            if result["outcome"] == "corrected" and frozen:
                result = {"outcome": "unsettled", "by": "", "text": result["text"] + "; the protocol is frozen, so FI "
                          "does not correct the check: it goes on unconfirmed"}
            if result["outcome"] == "corrected" and same_coder:
                result = {"outcome": "unsettled", "by": "", "text": result["text"] + "; the script was written by the "
                          "model that wrote the simulation, so the two are not independent and FI does not correct "
                          "the check: it goes on unconfirmed"}
            if result["outcome"] == "corrected":
                if name in self._oracle_proposals:  # put back if the correction cannot be made
                    self._oracle_kept_proposals[name] = self._oracle_proposals[name]
                self._oracle_proposals[name] = {
                    "name": name, "source": "numeric", "expected": result["expected"], "reason": result["text"],
                    "confirmed_by": {"source": "numeric", "by": result["by"], "value": result["expected"],
                                     "model": numeric.get("model"), "script": numeric.get("script"), "fingerprint": fp}}
                acted = True
            elif result["outcome"] == "code":
                self._oracle_disputed.discard(name)
                self._oracle_proposals.pop(name, None)
                self._oracle_set_aside.pop(name, None)
                self._oracle_challenge_hints[name] = result["text"]
                acted = True
            self._log.warning("[oracle] '%s': %s", name, result["text"])
            attempt.setdefault("triage", []).append({
                "kind": "referee", "check": name, "outcome": result["outcome"], "tried": result["text"],
                **({"points_to": "check"} if result["outcome"] == "corrected" else
                   {"points_to": "script"} if result["outcome"] == "code" else {}),
                "numeric": {k: numeric.get(k) for k in ("value", "how", "model", "script") if k in numeric}})
        if changed:
            _oracle_triage.write(self.fi_dir, {**_oracle_triage.read(self.fi_dir), key: record})
        return acted

    async def _numeric_reference(self, state: QuestState, oracle: dict[str, Any]) -> dict[str, Any]:
        """The check's number computed numerically from the definitions by a short script the model named for
        ``oracle_review`` writes (never shown either working or the measurement) and FI runs in the quest's environment,
        bounded in time: ``{"value", "how", "model", "script"}`` (``value`` None when it gave none)."""
        model = self._model_for_node(_oracle_challenge.NUMERIC_NODE) or self.config.provider.model or "the provider's default model"
        self._progress("Computing a known answer numerically, from the definitions")
        try:
            reply = await self._chat(_oracle_challenge.numeric_prompt(
                self._prompts["oracle_numeric_reference"], topic=str(state.get("topic") or self.config.topic or ""),
                oracle=oracle, protocol=self._protocol_block(state)), node=_oracle_challenge.NUMERIC_NODE)
        except Exception as e:  # noqa: BLE001 -- a referee that could not be had leaves the check unconfirmed
            return {"value": None, "how": f"it could not be asked ({str(e)[:160] or type(e).__name__})", "model": model,
                    "error": True}
        script, why = _oracle_challenge.numeric_script(reply)
        if not script:
            return {"value": None, "how": why, "model": model}
        folder = self.fi_dir / "numeric_reference"
        folder.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(oracle.get("name") or "check"))[:60] or "check"
        path = folder / f"{safe}_{_oracle_triage.fingerprint(oracle)[:8]}.py"
        path.write_text(script, encoding="utf-8")
        code, out, err = await self._run_in_env([self._rel_to_quest(path)], timeout_s=_oracle_challenge.numeric_timeout(self.config.execution.timeout_s),
                                                apart=True)
        value, how = _oracle_challenge.numeric_value(out)
        if value is None and code != 0:
            how = f"the script failed ({(err or '').strip().splitlines()[-1][:200] if (err or '').strip() else code})"
        self._log.info("[oracle] the numerical computation of %r from the definitions: %s", str(oracle.get("name")),
                       f"{_oracle.fmt_digits(value)} ({how})" if value is not None else how)
        return {"value": value, "how": how, "model": model, "script": self._rel_to_quest(path)}

    async def _recompute_expected(self, state: QuestState, oracle: dict[str, Any], value: float | None,
                                  ) -> tuple[dict[str, Any], bool]:
        """A failing check's expected value worked out again (prompt ``oracle_recompute``) by the model named for
        ``oracle_review`` (else the main model, which the record and the card say), from the check's statement, case and
        reference: never the measured value. One call per version of a check, kept in ``.fi/oracle_triage.json``, so a
        resume never asks again. ``(the triage entry, whether a call was made)``."""
        kept = _oracle_triage.read(self.fi_dir)
        fp = _oracle_triage.recompute_key(oracle)
        answers = kept.get("recompute") if isinstance(kept.get("recompute"), dict) else {}
        named = self._model_for_node(_review.NODE)
        planner = self._model_for_node("plan") or self.config.provider.model or ""
        # Every model that may have written the checks' derivations (the plan, its rewrites, the design).
        writers = {self._model_for_node(n) or self.config.provider.model or "" for n in _review.WRITER_NODES}
        called = False
        if isinstance(answers.get(fp), dict):
            answer = answers[fp]
        else:
            prompt = _oracle_triage.recompute_prompt(
                self._prompts["oracle_recompute"], topic=str(state.get("topic") or self.config.topic or ""),
                oracle=oracle)
            parsed: Any = None
            error = ""
            try:
                reply = await self._chat(prompt, node=_oracle_triage.NODE)
                parsed = _parse_json_lenient(reply, node=_oracle_triage.NODE)
            except Exception as e:  # noqa: BLE001 -- a look that could not be had never stops a quest
                error = str(e)[:200] or type(e).__name__
                self._log.warning("[oracle] the expected value of %r could not be worked out again: %s",
                                  str(oracle.get("name")), error)
            called = True
            recomputed, how = _oracle_triage.parse_recompute(parsed)
            answer = {"name": str(oracle.get("name") or "").strip(), "recomputed": recomputed, "how": how,
                      "model": named or planner or "the provider's default model",
                      **({"error": error} if error else {})}
            if not error:  # a call that got no answer is asked again on the next run (once per run, at most)
                # Kept with whether the plan's own model answered as it was when asked (today's settings can differ).
                asked_same = (not named or "" in writers or any(_review.same_model(named, w) for w in writers))
                _oracle_triage.write(self.fi_dir, {**kept, "recompute": {**answers, fp: {**answer,
                                                                                         "same_model": asked_same}}})
        # Whether the plan's own model answered: worked out from the model that ACTUALLY answered (a kept answer may come
        # from an earlier run with other settings), never from today's settings alone. An unknown model -- the answer's
        # or a writer's (the provider's default) -- counts as the same model: fail closed. A kept "same" stays same.
        who = str(answer.get("model") or "").strip()
        unknown = not who or who == "the provider's default model"
        same = (bool(answer.get("same_model")) or unknown or "" in writers
                or any(_review.same_model(who, w) for w in writers))
        answer = {**answer, "same_model": same}
        recomputed = _oracle_triage.num(answer.get("recomputed"))
        verdict = _oracle_triage.recompute_verdict(oracle, recomputed, value)
        entry = _oracle_triage.recompute_entry(oracle, verdict, recomputed, value, str(answer.get("how") or ""),
                                               model=str(answer.get("model") or ""),
                                               same_model=bool(answer.get("same_model")))
        return entry, called

    async def _declare_oracles(self, incomplete: list[str] | None = None) -> bool:
        """Ask for the plan to be rewritten with at least one oracle in its protocol (or, with ``incomplete``, with the numbers
        the declared ones lack); ``False`` when it cannot be."""
        if incomplete:
            request = (
                "These declared oracles cannot be judged by the engine: " + " ".join(incomplete) + " Give each of them a numeric "
                "`expected` (worked out from the closed form, the limit or the invariant; 0 for an invariant's worst violation or for the difference between two implementations) and "
                "a numeric `tolerance` (plus `tolerance_mode: relative` only when the expected value is not 0), a `kind` "
                "(one of special_case, invariant, symmetry, second_implementation, convergence_rate, published_value) "
                f"and a `reference` saying where the expected value comes from: {_REFERENCE_FORMS} Also give "
                f"`expected_formula`, {_forms.EXPECTED_FORMULA_LANGUAGE} (not for an invariant, a symmetry or a second "
                "implementation). Change nothing else."
            )
        else:
            request = (
            "The protocol declares no oracle. Add `oracles` to the protocol block of the design: at least one check "
            "that does not rely on the script's own numbers being right (a closed form the simulation must reproduce, a "
            "limiting case with a known answer, a conservation law or invariant every run must satisfy, a small case whose "
            "exact answer can be computed another way, or a second independent implementation). Each entry has a `name`, a "
            "`kind` (one of special_case, invariant, symmetry, second_implementation, convergence_rate, published_value), "
            "a `check` saying what the script measures on which small case, a NUMERIC `expected` (the value the "
            "measurement must agree with, worked out here from the closed form, the limit or the invariant; 0 for an "
            "invariant's worst violation), a NUMERIC `tolerance` (how far from `expected` still agrees), optionally a "
            "`tolerance_mode` (`absolute`, the default, or `relative`) and a `reference` saying where the expected value comes "
            f"from: {_REFERENCE_FORMS}" " For every check that is not an invariant, a symmetry or a second implementation "
            f"give `expected_formula` too, {_forms.EXPECTED_FORMULA_LANGUAGE}." " Also give each a `case` (the settings of one small, fast run of the simulation, e.g. {\"dt\": 0.1}) and a "
            "`measure` (how the number is computed from what that run returns: one returned name, or a formula of them "
            "such as abs(P_out - P_in) / P_in; for an invariant, a symmetry or a second implementation it is the worst "
            "violation, expecting 0), so the engine can run the simulation on it itself (for a "
            "random simulation that run is one trial, so a case suits a check one trial shows exactly); when the "
            "check claims an order of accuracy, give `order` too. The engine judges the measurement against these numbers. "
            "Change nothing else."
            )
        before = self._planned_oracles()
        try:
            await self.revise_plan(
                request, by="engine", scope=_declare_scope(bool(incomplete)),
                note="the checks against known answers the engine asked for" + (" (their numbers)" if incomplete else ""))
        except (FileNotFoundError, ValueError) as e:
            self._log.warning("[oracle] the plan could not be rewritten with an oracle: %s", e)
            return False
        self._put_back_moved_expected(before)
        # An oracle this rewrite added, or whose numbers it changed, is the engine's, not what a person read at the plan
        # stop: it is recorded, so the freeze never says a person approved it unless they were shown it.
        after = self._planned_oracles()
        changed = [name for name, oracle in after.items() if before.get(name) != oracle]
        if changed:
            self._note_engine_change(changed)
        return True

    def _note_engine_change(self, changed: list[str], removed: list[str] | None = None, reason: str = "") -> None:
        """Record checks the engine added, changed or removed in plan.md after it was written, merged with what is
        recorded and not yet shown, so the stop before the freeze (``_hold_added_oracles``) names all of them and the
        freeze record never says a person approved one they were not shown."""
        earlier = self._oracles_added_read() or {}
        unseen = not earlier.get("shown")
        carried = [str(n) for n in earlier.get("oracles") or []] if unseen else []
        gone = [str(n) for n in earlier.get("removed") or []] if unseen else []
        reasons = [str(earlier["reason"])] if unseen and earlier.get("reason") else []
        merged = " ".join([*reasons, reason]).strip()
        # ``fitted`` (checks the test run was fitted to) is never dropped by a later change: losing it would let such a
        # check count as independently validated (``_fitted_to_test_run``).
        fitted = sorted({str(n) for n in earlier.get("fitted") or []})
        disagreements = [d for d in earlier.get("formula_disagreements") or [] if isinstance(d, dict)]
        self._oracles_added_write({
            **({"formula_disagreements": disagreements} if disagreements else {}),
            "oracles": list(dict.fromkeys([*carried, *changed])), "removed": list(dict.fromkeys([*gone, *(removed or [])])),
            "shown": False, "at": _frozen.now(), **({"reason": merged} if merged else {}),
            **({"fitted": fitted} if fitted else {}),
        })

    def _planned_oracles(self) -> dict[str, dict[str, Any]]:
        """The oracles of the protocol in ``plan.md``, by name."""
        planned, _why = _plan.load_design(self.quest_root)
        protocol = planned.get("protocol") if isinstance(planned, dict) else None
        return {str(o["name"]): o for o in _oracle.declared(protocol if isinstance(protocol, dict) else None)}

    def _planned_equations(self) -> dict[str, str]:
        """The formulas of the model's equations in ``plan.md``, by id."""
        planned, _why = _plan.load_design(self.quest_root)
        protocol = planned.get("protocol") if isinstance(planned, dict) else None
        return {str(e["id"]).strip(): " ".join(str(e.get("formula") or "").split())
                for e in _review.equations_of(protocol if isinstance(protocol, dict) else None)}

    def _oracles_added_path(self) -> Path:
        return self.fi_dir / "oracles_added.json"

    def _oracles_added_read(self) -> dict[str, Any] | None:
        """What the oracle gate added to the plan's protocol after the plan was written (``None`` when it added nothing)."""
        try:
            record = json.loads(self._oracles_added_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    def _oracles_added_write(self, record: dict[str, Any]) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            self._oracles_added_path().write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[oracle] couldn't record the oracles added to the plan: %r", e)

    def _hold_added_oracles(self) -> None:
        """With ``pauses.plan: ask``, say (once per addition, without stopping) which checks against known answers the
        engine added to the plan, changed or removed after the person read it. Judging a check's expected value or
        tolerance is not something a person is asked to do: FI's own look checks them when they run (another model
        works each expected value out again without seeing the result, core/oracle_triage.py), and a check that then
        fails is repaired, corrected or gone on with marked unconfirmed. The record stays ``shown: false``, so the freeze
        record never says a person approved a check they were not shown. A no-op once the protocol is frozen, or when
        the plan stop is off."""
        if self.config.pauses.plan != "ask" or _frozen.load(self.quest_root) is not None:
            return
        added = self._oracles_added_read()
        if added is None or added.get("shown") or added.get("told") or not (added.get("oracles") or added.get("removed")):
            return
        # Only the added checks still in the plan: one the person renamed or removed at an earlier stop they have seen.
        planned = self._planned_oracles()
        still = [str(n) for n in added.get("oracles") or [] if str(n) in planned]
        removed = [str(n) for n in added.get("removed") or [] if str(n) not in planned]
        self._oracles_added_write({**added, "oracles": still, "removed": removed, "told": True})
        if still or removed:
            named = ", ".join([*(f"'{n}'" for n in still), *(f"'{n}' (removed)" for n in removed)])
            self._log.info("[plan] after the plan was read, FI added or changed these checks against known answers: "
                           "%s%s; not stopping for them (FI checks their expected values itself when they run)",
                           named, f" ({added['reason']})" if added.get("reason") else "")
            print(f"[FI] FI added or changed {len(still) + len(removed)} check(s) against known answers after you read "
                  f"the plan ({named}). Nothing to do: FI checks their expected values itself when they run; plan.md "
                  "lists them.")

    # --- the checks against known answers, before anything runs (core/oracle_forms.py) ---------------------------------

    def _oracle_guidance_path(self) -> Path:
        return self.fi_dir / "oracle_guidance.json"

    async def _guide_oracles(self, state: QuestState) -> None:
        """Plan time, once per quest, before anything runs: hold each check against a known answer in ``plan.md`` to its
        kind's numeric form. A rewrite that cannot change the check's verdict is made and said in plan.md; a check that
        needs the plan's own judgement goes back to the plan once, with a precise request (``plan_revise``). A no-op for
        a quest that runs no experiment, with the checks off, once the protocol is frozen, or when it was done before."""
        if (not self._runs_code(state) or self.config.engine.oracle_check == "off"
                or _frozen.load(self.quest_root) is not None or self._oracle_guidance_path().is_file()):
            return
        path = _plan.plan_path(self.quest_root)
        if not path.is_file():
            return
        record: dict[str, Any] = {"at": _frozen.now()}
        try:
            record["forms"] = await self._hold_oracle_forms(path, fi_runs=self._split_on(state), state=state)
        except _ModelAnswerProblem:
            raise
        except Exception as e:  # noqa: BLE001 -- a check of the plan must never stop a quest by crashing
            self._log.warning("[oracle] the checks in the plan could not be looked at: %r", e)
            record["error"] = repr(e)[:300]
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            self._oracle_guidance_path().write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[oracle] couldn't record the look at the plan's checks: %r", e)

    def _write_plan_section(self, path: Path, lines: list[str], note: str) -> None:
        """Add ``lines`` to plan.md's section on the checks (prose, never read back) and keep that version."""
        text = _plan.add_to_section(path.read_text(encoding="utf-8"), _forms.HEADING, lines)
        path.write_text(text, encoding="utf-8")
        _plan.record_version(self.quest_root, text, by="engine", note=note)

    async def _revise_checks_only(self, request: str, *, equations: bool | str = False, isolation: bool = False) -> str:
        """Ask the plan (``plan_revise``) for ``request`` and keep only what it changed in the checks
        (``protocol.oracles``): anything else it changed in the design block is put back, so a request about the checks
        can never move the grid, the thresholds or the criteria. ``""`` when the plan was rewritten, else why not (a
        failed call is reported, never a crash). With ``equations`` (a reader of another model objected to an equation
        of the model), the equations of ``protocol.model`` are also taken from the rewrite, none removed; and then no
        check keeps a looser tolerance than it had (the equation changes, the bar never does)."""
        path = _plan.plan_path(self.quest_root)
        before_text = path.read_text(encoding="utf-8")
        # Only the checks (written whole: a request may replace one), the criteria that read them and, when asked, the
        # model's equations are asked for; the tolerance guards below still hold what comes back.
        scope = _plan_scope.PartScope(
            parts=("oracles", "criteria"), inside={"model": ("equations",)} if equations else {})
        try:
            await self.revise_plan(request, by="engine", scope=scope, note="the plan's checks changed on FI's request")
        except _ModelAnswerProblem:
            raise
        except Exception as e:  # noqa: BLE001 -- a request that got no usable answer leaves the plan as it was
            return f"the plan could not be rewritten ({str(e)[:200] or type(e).__name__})"
        after_text = path.read_text(encoding="utf-8")
        before, after = _plan.raw_design_block(before_text), _plan.raw_design_block(after_text)
        if before is None or after is None:
            return ""

        def part(block: dict[str, Any], key: str) -> list[Any]:
            protocol = block.get("protocol") if isinstance(block.get("protocol"), dict) else {}
            items = protocol.get(key)
            return items if isinstance(items, list) else []

        def named(items: list[Any]) -> dict[str, Any]:
            return {str(o.get("name") if isinstance(o, dict) else o).strip(): o for o in items}

        old_checks, new_checks = named(part(before, "oracles")), named(part(after, "oracles"))
        # The checks that changed (added, removed, renamed or edited): a criterion that reads one of them may change
        # with it (its number changed meaning); every other criterion stays as it was.
        touched = {n.lower() for n in {*old_checks, *new_checks} if old_checks.get(n) != new_checks.get(n)}

        def reads_touched(c: Any) -> bool:
            return isinstance(c, dict) and str(c.get("oracle") or "").strip().lower() in touched

        def key(c: Any) -> str:
            return str(c.get("name") if isinstance(c, dict) else c).strip().lower()

        # Criteria matched by their own name, in the rewrite's order: one that read, or now reads, a changed check is
        # taken from the rewrite (added, changed or dropped with it); every other one is the plan's as it was.
        old_criteria = {key(c): c for c in part(before, "criteria")}
        criteria: list[Any] = []
        for c in part(after, "criteria"):
            old = old_criteria.get(key(c))
            if reads_touched(c) or reads_touched(old):
                criteria.append(c)
            elif old is not None:
                criteria.append(old)
        seen = {key(c) for c in criteria}
        criteria += [c for k, c in old_criteria.items() if k not in seen and not reads_touched(c)]

        def keep(block: dict[str, Any]) -> dict[str, Any]:
            protocol = dict(block.get("protocol") or {})
            protocol["oracles"] = part(after, "oracles")
            if isolation and not equations:
                # A case or a measure was objected to: no tolerance loosens and no `expected` is written by hand.
                protocol["oracles"] = _review.never_looser(part(before, "oracles"), protocol["oracles"], keep_expected=True)
            if equations:
                protocol["oracles"] = _review.never_looser(part(before, "oracles"), protocol["oracles"],
                                                           keep_expected=isolation)
                model_before = protocol.get("model")
                model_after = (after.get("protocol") or {}).get("model") if isinstance(after.get("protocol"), dict) else None
                if isinstance(model_before, dict) and isinstance(model_after, dict):
                    # ``"example"``: only the worked examples may change; an equation itself only when a reader objected.
                    protocol["model"] = {**model_before, "equations": _review.keep_every_equation(
                        model_before.get("equations"), model_after.get("equations"), only_example=equations == "example")}
            if criteria or "criteria" in protocol:
                protocol["criteria"] = criteria
            return {**block, "protocol": protocol}

        # The plan as it was, with only the checks (and the criteria reading them) from the rewrite: nothing else in
        # the design, and no prose the rewrite dropped (FI's own sentences about the checks among it), is lost.
        kept = _plan.edit_design_block(before_text, keep)
        if kept is None:
            return ""
        kept = _plan.refresh_model_section(kept)
        if _plan.raw_design_block(kept) != after or _plan.parse(kept).design is None:
            self._log.warning("[oracle] the plan's rewrite changed more than the checks; only the checks were kept")
        if _plan.parse(kept).design is None:
            path.write_text(before_text, encoding="utf-8")
            # Recorded, so the plan put back is not later taken for a person's edit.
            _plan.record_version(self.quest_root, before_text, by="engine",
                                 note="the plan's rewrite of its checks could not be read; put back as it was")
            return "the plan's rewrite of the checks could not be read"
        path.write_text(kept, encoding="utf-8")
        _plan.record_version(self.quest_root, kept, by="engine", note="the plan's changes to its checks")
        self._put_back_moved_expected(old_checks)
        return ""

    def _put_back_moved_expected(self, before: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """After the plan was rewritten for its checks: a check whose expected value the rewrite moved away from its own
        formula, when the earlier number agreed with that formula, gets the earlier number back (:func:`core.oracle_forms.moved_expected`;
        pure computation, no model, no measured value). Only before the protocol is frozen. Said in ``run.log`` and in plan.md,
        and recorded as the engine's change."""
        if _frozen.load(self.quest_root) is not None:
            return []
        try:
            path = _plan.plan_path(self.quest_root)
            text = path.read_text(encoding="utf-8")
            design, _why = _plan.load_design(self.quest_root)
            protocol = design.get("protocol") if isinstance(design, dict) else None
            if not isinstance(protocol, dict):
                return []
            moved = _forms.moved_expected(list(before.values()), _oracle.declared(protocol),
                                          _forms.fixed_settings_of_design(design))
            if not moved:
                return []
            wanted = {m["name"]: m["now"] for m in moved}

            def change(block: dict[str, Any]) -> dict[str, Any] | None:
                prot = block.get("protocol")
                if not isinstance(prot, dict) or not isinstance(prot.get("oracles"), list):
                    return None
                items = [{**o, "expected": wanted[str(o.get("name") or "").strip()]}
                         if isinstance(o, dict) and str(o.get("name") or "").strip() in wanted else o for o in prot["oracles"]]
                return {**block, "protocol": {**prot, "oracles": items}}

            edited = _plan.edit_design_block(text, change)
            if edited is None or _plan.parse(edited).design is None:
                return []
            new_text = _plan.refresh_model_section(edited)
            path.write_text(new_text, encoding="utf-8")
            _plan.record_version(self.quest_root, new_text, by="engine",
                                 note="expected values the rewrite moved away from their own formula, put back")
            lines = [f"the rewrite of the plan moved the expected value of {m['name']!r} from {m['now']:.10g} to {m['was']:.10g}, "
                     f"but its own formula `{m['formula']}` computes {m['value']:.10g}; the check keeps {m['now']:.10g}, "
                     "which agrees with the formula" for m in moved]
            for line in lines:
                self._log.warning("[oracle] %s", line)
            self._write_plan_section(path, ["", *[f"- {line[0].upper()}{line[1:]}." for line in lines]],
                                     note="expected values put back after a rewrite of the plan")
            return moved
        except Exception as exc:  # noqa: BLE001 -- a convenience guard, never a reason to stop
            self._log.warning("[oracle] could not compare the rewritten checks with their own formulas (%r)", exc)
            return []

    async def _hidden_check(self, state: QuestState, python: Path | str, env: dict[str, str] | None) -> None:
        """Under ``rigor_profile: research``, after a run: FI's own check at a setting the code never saw
        (core/hidden_check.py), kept in ``needs/HIDDEN_CHECK.json``. Never stops the quest; a check that fails there
        is a gap in the evidence."""
        protocol = self._protocol_block(state)
        if protocol is None or self.config.engine.oracle_check == "off" or not _hidden.candidates(protocol):
            return
        timeout = max(30, int(self.config.execution.timeout_s * self.config.engine.pilot_timeout_frac))
        written = self._hidden_check_written()
        try:
            record = await self._await_with_heartbeat(_hidden.run(
                self.executor, python, self.quest_root, protocol, timeout_s=timeout, env=env,
                engine_callable=bool(getattr(self, "_trial_mode", False)),
                trusted_sha=str(written[-1].get("sha256") or "") if written else ""),
                label="checking a setting the code never saw")
            sha = _hidden.write(self.quest_root, record)
        except Exception as e:  # noqa: BLE001 -- the run itself succeeded; this check must not end the quest
            self._log.warning("[oracle] the check at a setting the code never saw could not be made: %r", e)
            return
        # The record's hash goes into the trace: a record the simulation's own code writes is not FI's. A record reused
        # as FI wrote it is not a new run, so it adds no event.
        if not (written and sha == written[-1].get("sha256")):
            self._audit("hidden_check", sha256=sha, status=record.get("status"), code_sha256=record.get("code_sha256"),
                        checks_key=record.get("checks_key"))
        for gap in _hidden.evidence_gaps(self.quest_root, protocol, self._hidden_check_written()):
            self._log.warning("[oracle] %s", gap)
        if record.get("status") == "passed":
            self._log.info("[oracle] %d check(s) also passed at a setting the code never saw", len(record["cases"]))

    async def _refinement_check(self, state: QuestState, python: Path | str, env: dict[str, str] | None) -> None:
        """After a run, a measurement's numerical settings (``protocol.numerical_settings``): FI runs the settings where each
        headline result was lowest and highest again at the finer values and judges convergence (core/refinement_check.py),
        kept in ``needs/REFINEMENT_CHECK.json``. Never stops the quest; what is not converged is a gap in the evidence."""
        protocol = self._protocol_block(state)
        if not _refine.settings(protocol):
            return
        record = _refine.read(self.quest_root)
        if record and _hidden.same_code(record.get("code_sha256"), self.quest_root) \
                and record.get("key") == _refine.key(protocol):
            return  # already run on this code and these settings (a resume)
        timeout = max(60, int(self.config.execution.timeout_s * self.config.engine.pilot_timeout_frac))
        names = self._headline_result_names(state, protocol)
        block = protocol.get("optimisation") if isinstance(protocol.get("optimisation"), dict) else None
        inside = (_optimise_search.in_each_run(block) if block is not None else _in_run.of(protocol)) or None
        try:
            record = await self._await_with_heartbeat(_refine.run(
                self.executor, python, self.quest_root, protocol, deterministic="run_trial" not in _trial_runner.entries(self.quest_root / "code" / "simulate.py"),
                timeout_s=timeout, env=env, thresholds=protocol.get("thresholds") or {},
                in_each_run=inside, names=names),
                label="running the headline results again at finer numerical settings")
            _refine.write(self.quest_root, record)
        except Exception as e:  # noqa: BLE001 -- the run itself succeeded; this check must not end the quest
            self._log.warning("[execute] the headline results could not be run again at finer numerical settings: %r", e)
            return
        for line in _refine.lines(record):
            self._log.info("[execute] %s", line)

    def _independence_gaps(self, protocol: dict[str, Any] | None) -> list[str]:
        """Under ``rigor_profile: research``: why the checks are not shown independent of the model that wrote them (a
        usable second reading by another model, core/oracle_review.py; the checks at a setting the code never saw,
        core/hidden_check.py). Empty outside research, with no checks, or with the checks off (its own gap)."""
        if (self.config.rigor_profile != "research" or self.config.engine.oracle_check == "off"
                or not _oracle.declared(protocol)):
            return []
        calls = _attempts.read(self.fi_dir, _attempts.MODEL_CALLS)
        return [*_review.independence_gaps(self._oracle_review_read(), calls,
                                           configured=bool(self._model_for_node(_review.NODE)), protocol=protocol),
                *_hidden.evidence_gaps(self.quest_root, protocol, self._hidden_check_written(),
                                       trace_ok=self._trace_checks_out())]

    def _hidden_check_written(self) -> list[dict[str, Any]]:
        """The trace's ``hidden_check`` events, oldest first: the hash of each record FI wrote, its status and code.
        None from a trace whose hash chain does not check out (the simulation's code could have added one)."""
        if not self._trace_checks_out():
            return []
        try:
            return [e for e in _audit_log.read(self.audit.path) if e.get("kind") == "hidden_check"]
        except Exception:  # noqa: BLE001 -- an unreadable trace names no record, which is a gap, never a pass
            return []

    def _trace_checks_out(self) -> bool:
        try:
            return bool(_audit_log.verify(self.audit.path).ok)
        except Exception:  # noqa: BLE001 -- a trace that cannot be read does not check out
            return False

    def _oracle_review_path(self) -> Path:
        return self.fi_dir / "oracle_review.json"

    def _oracle_review_read(self) -> dict[str, Any]:
        """What the second reading of the checks left on disk (its answer, and how far the look got), or ``{}``."""
        try:
            record = json.loads(self._oracle_review_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return record if isinstance(record, dict) else {}

    def _oracle_review_write(self, record: dict[str, Any]) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            self._oracle_review_path().write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[oracle] couldn't record the second reading of the checks: %r", e)

    async def _review_oracles(self, state: QuestState) -> dict[str, Any]:
        """A second opinion on the plan's checks (core/oracle_review.py): one call, node ``oracle_review``, on the
        checks as plan.md now states them. Returns its record (``findings`` for the plan, ``lines`` for plan.md, the
        verdicts), kept in ``.fi/oracle_review.json`` as soon as it is had, so a resume never asks again; ``{}`` when
        the plan declares no check. The reviewer is the model named for ``oracle_review`` in ``provider.node_models``
        (the model that actually answered is the one named); when it is, or may be, the planner's own model, plan.md
        and run.log say so (under ``rigor_profile: research``, with how to name another). A second opinion that could
        not be had (a failed call, an answer the provider withheld or cut off, one that names none of the checks) is
        said, and never stops the quest."""
        kept = self._oracle_review_read()
        if kept.get("lines"):
            return kept
        design, _why = _plan.load_design(self.quest_root)
        protocol = design.get("protocol") if isinstance(design, dict) else None
        if not _oracle.declared(protocol if isinstance(protocol, dict) else None):
            return {}
        model_text, oracles_text, partial = _review.prompt_parts(protocol)
        review, error = None, ""
        if oracles_text.lstrip().startswith("[]"):
            error = "the plan's checks are too long to show to a reader"  # nothing to judge: no call spent on it
        else:
            prompt = self._prompts["oracle_review"].substitute(
                topic=" ".join(str(state.get("topic") or self.config.topic or "").split())[:1500],
                model=model_text, oracles=oracles_text)
            try:
                reply = await self._chat(prompt, node=_review.NODE)
                review = _review.parse(_parse_json_lenient(reply, node=_review.NODE), protocol, partial=partial)
                if review is not None and review.said_nothing:
                    # No problem named and nothing said to be checked is no review: asked once more, then not reviewed.
                    self._log.warning("[oracle] the second reader of the checks named no problem and did not say what "
                                      "it checked; asking once more")
                    reply = await self._chat(prompt + _pairs.REASK, node=_review.NODE)
                    review = _review.parse(_parse_json_lenient(reply, node=_review.NODE), protocol, partial=partial)
                    if review is not None and review.said_nothing:
                        review, error = None, _pairs.NOT_A_REVIEW
                if review is None and not error:
                    error = "it named none of the plan's checks"
            except _ModelAnswerProblem as e:
                error = f"the provider withheld or cut off its answer ({str(e)[:160] or type(e).__name__})"
            except Exception as e:  # noqa: BLE001 -- a second opinion that could not be had never stops a quest
                error = f"the call failed ({str(e)[:160] or type(e).__name__})"
        # Who read the checks: the model that answered when the connection named it, else the one asked for; the planner
        # likewise. Two names are compared only when they are the same kind of name (both named by their connections,
        # or both as asked for): a dated id a connection reports is not compared with an alias in the config.
        answered, planned = self._chat_provenance(_review.NODE), self._chat_provenance("plan")
        reviewer_asked = self._model_for_node(_review.NODE) or self.config.provider.model or ""
        planner_asked = self._model_for_node("plan") or self.config.provider.model or ""
        reviewer = str(answered.get("model") if answered.get("reported") else reviewer_asked or "")
        planner = str(planned.get("model") if planned.get("reported") else planner_asked or "")
        if answered.get("reported") and planned.get("reported"):
            same: bool | None = _review.same_model(answered.get("model"), planned.get("model"))
        elif reviewer_asked and planner_asked:
            same = _review.same_model(reviewer_asked, planner_asked)
        elif not self._model_for_node(_review.NODE) and not self._model_for_node("plan"):
            same = True  # both on the provider's own default
        else:
            same = None
        research = self.config.rigor_profile == "research"
        below = "; under research this keeps the result below independently validated" if research else ""
        if same is not False and not error:
            (self._log.warning if research else self._log.info)(
                "[oracle] the checks were read a second time by %s (%s)%s",
                "the model that wrote the plan" if same else "what may be the model that wrote the plan",
                planner or "the provider's default model",
                f"{below}; for a second opinion from another model set provider.node_models.oracle_review"
                if research else "")
        found = _review.findings(review) if review is not None else []
        # The model's equations, read in the same call: a reader that is shown to be another model than the planner may
        # object to one (its objection goes into the one request to the plan); the planner's own reading is never
        # counted as a second reading of them, and the record says so.
        eq_record = _review.equation_record(review, protocol, same_model=same, reviewer=reviewer or "the reviewer",
                                            planner=planner or "the planner")
        eq_found = _review.equation_findings(review) if review is not None and same is False else []
        if eq_record is not None and not error:
            if eq_record["read_by_other_model"]:
                self._log.info("[oracle] the model's %d equation(s) were read by another model than the one that wrote "
                               "the plan; %s", eq_record["total"],
                               f"it objects to {len(eq_found)}" if eq_found else "it has no objection")
            else:
                self._log.info("[oracle] the model's %d equation(s) were not read by another model (%s)",
                               eq_record["total"], eq_record["why"])
        # The checks' cases, judged in the same call: objections count only from another model than the planner.
        iso_record = _review.isolation_record(review, same_model=same, reviewer=reviewer or "the reviewer",
                                              planner=planner or "the planner") if review is not None else None
        iso_found = _review.isolation_findings(review) if review is not None and same is False else []
        if iso_record is not None and not error:
            self._log.info("[oracle] the checks' cases %s", (
                f"were judged by another model than the one that wrote the plan; it objects to {len(iso_found)}"
                if iso_record["read_by_other_model"] else f"were not judged by another model ({iso_record['why']})"))
        found = [*found, *eq_found, *iso_found]
        for f in found:
            self._log.info("[oracle] the second reader of the checks: %s", f)
        if error:
            # Never a stop: the quest goes on, and under research the evidence level keeps the gap.
            self._log.warning("[oracle] the second reading of the checks could not be used: %s%s", error, below)
        default = "the provider's default model"
        record = {
            "reviewer": reviewer or default, "planner": planner or default, "same_model": same,
            "reported": bool(answered.get("reported")), "findings": found,
            # The line of .fi/model_calls.jsonl that holds this reading: the evidence level reads who answered from there.
            # Always present (None when the call's line was not written): only a record from before it was kept is
            # matched to the one reading on record.
            "call_id": answered.get("call_id") if not error else None,
            # Which checks it read, as they were shown (what decides each verdict): a check added or changed later was
            # not read (core/oracle_review.py::independence_gaps).
            **({"read": _review.fingerprints(protocol, shown_only=True)} if review is not None else {}),
            "lines": _review.plan_lines(review, reviewer=reviewer or default, planner=planner or default,
                                        same_model=same, research=research,
                                        reported=bool(answered.get("reported")), error=error,
                                        sent=not oracles_text.lstrip().startswith("[]"), equations=eq_record,
                                        isolation=iso_record),
            **({"isolation": iso_record} if iso_record is not None else {}),
            **({"isolation_objections": iso_found} if iso_found else {}),
            **({"equations": eq_record} if eq_record is not None else {}),
            **({"equation_objections": eq_found} if eq_found else {}),
            **({"verdicts": review.checks, "add": review.add, "summary": review.summary} if review is not None else {}),
            **({"error": error} if error else {}),
        }
        self._oracle_review_write({**self._oracle_review_read(), **record})  # keeps how far the look got
        return record

    async def _hold_oracle_forms(self, path: Path, *, fi_runs: bool = True,
                                 state: QuestState | None = None) -> dict[str, Any]:
        """Hold plan.md's checks to their kinds' forms and have them read by a second model (see
        :meth:`_guide_oracles`); both go back to the plan in ONE request. Each part is done at most once, a resume
        included (``.fi/oracle_review.json`` records how far it got). What was rewritten, found, asked and left."""
        text, rewrites, requests = _forms.apply_to_plan(path.read_text(encoding="utf-8"), fi_runs=fi_runs)
        too_tight = _forms.precision_gaps(text)  # checks whose tolerance is below the precision of their expected value
        left: list[str] = []
        for r in rewrites:
            self._log.info("[oracle] %s", r)
        progress = self._oracle_review_read()
        if rewrites:
            # The reviewer reads the checks as FI rewrote them, and the rewrite is recorded as FI's own version at once
            # (a run stopped during the review must not leave it looking like a person's edit), with its sentences.
            path.write_text(text, encoding="utf-8")
            _plan.record_version(self.quest_root, text, by="engine",
                                 note="checks against known answers written in their kind's numeric form")
            self._oracle_review_write({**progress, "rewrites": rewrites})
        rewrites = rewrites or [str(r) for r in progress.get("rewrites") or []]
        review = await self._review_oracles(state or {})
        findings = [str(f) for f in review.get("findings") or []]
        progress = self._oracle_review_read()
        # FI's own computation of each expected value, from the formula the plan gives (core/oracle_forms.py): what the
        # plan must still say goes into the SAME request as the rest; nothing is asked on a resume that already asked.
        formula_asks: list[dict[str, Any]] = []
        example_asks: list[dict[str, Any]] = []
        fixed_now: dict[str, float] = {}
        if not progress.get("asked") and not self._simulation_ran():
            design_now, _why = _plan.load_design(self.quest_root)
            protocol_now = design_now.get("protocol") if isinstance(design_now, dict) else None
            fixed_now = _forms.fixed_settings_of_design(design_now)
            formula_asks = [f for f in _forms.formula_findings(protocol_now if isinstance(protocol_now, dict) else None,
                                                               fixed_now)
                            if f["state"] != "agrees"]
            # A worked example for each equation the code computes as a function (core/equation_tests.py): one that is
            # missing or cannot be used goes into the SAME request, never guessed.
            if fi_runs:
                example_asks = [r for r in _equation_tests.example_rows(
                    protocol_now if isinstance(protocol_now, dict) else None)
                    if r["state"] in ("missing", "unusable", "unprotected")]
        if (rewrites or requests or review.get("lines") or formula_asks or example_asks) and not progress.get("written"):
            # What FI did and what the reviewer said are written before the plan is asked anything, so a failed request
            # never leaves them unexplained.
            self._write_plan_section(path, [
                "> What FI did to the checks before anything ran. Each kind of check has one numeric form: the worst "
                "violation, expecting 0, for an invariant, a symmetry or a second implementation; the quantity itself "
                "for the others.", "",
                *[f"- {r}" for r in rewrites], *[str(line) for line in review.get("lines") or []]],
                note="the checks against known answers looked at")
            progress = {**progress, "written": True}
            self._oracle_review_write(progress)
        if progress.get("asked") and not progress.get("answered"):
            # The request went out in a run that stopped before its answer was read: it is not made again, and the plan
            # says so.
            # Only the checks the recorded ask was about are told of, and only those the plan left tighter than their
            # digits (their tolerance is not changed).
            widened = self._note_tight_tolerances(path, list(progress.get("precision_asked") or []),
                                                  added=progress.get("precision_added_asked") or [])
            requests = _forms.apply_to_plan(path.read_text(encoding="utf-8"), fi_runs=fi_runs, precision_asks=False)[2]
            self._write_plan_section(path, [
                "", "- FI asked the plan once to change its checks, and the quest stopped before an answer to it could "
                "be used; it was not asked again. Read the checks in the block below before the run.",
                *[f"- {w}." for w in widened],
                *[f"- Still not in its kind's form (read it before the run): {r}" for r in requests]],
                note="the checks against known answers: a request whose answer was not read")
            self._oracle_review_write({**progress, "answered": True})
            self._log.warning("[oracle] the request to the plan about its checks was not answered before the quest "
                              "stopped; it is not made again, and a tolerance it was asked about that is still tighter "
                              "than its expected value's digits is said so in the plan (it is not changed)")
        elif (requests or findings or formula_asks or example_asks) and not progress.get("asked"):
            for r in requests:
                self._log.warning("[oracle] asking the plan to change a check: %s", r)
            for f in formula_asks:
                reason = {"missing": "it gave none", "unusable": f"its formula could not be computed: {f.get('why')}",
                          "differs": "its formula and its expected value disagree",
                          "ambiguous": "its formula can be read two ways",
                          "arithmetic": f"it states {f.get('expression')} = {f.get('stated')}, which comes to "
                                        f"{f.get('value'):.10g}" if f["state"] == "arithmetic" else ""}[f["state"]]
                self._log.info("[oracle] asking the plan about the expected value of %r (%s)", f["name"], reason)
            # Recorded before the request: it is made at most once, even if this run stops while it is out.
            self._oracle_review_write({**progress, "asked": True, **({"precision_asked": too_tight} if too_tight else {})})
            before = self._planned_oracles()
            equations_before = self._planned_equations()
            for r in example_asks:
                self._log.info("[plan] asking the plan for a worked example of equation %s (%s)", r["id"], r["why"])
            must = [p for p in (_forms.request(requests, last=not (findings or formula_asks or example_asks)) if requests else "",
                                _forms.formula_request(formula_asks, last=not (findings or example_asks), fixed=fixed_now),
                                _equation_tests.request(example_asks, last=not findings)) if p]
            parts = [*must, _review.request(findings, equations=bool(review.get("equation_objections")),
                                            isolation=bool(review.get("isolation_objections")))
                     if findings else ""]
            if len(must) + bool(findings) > 1:
                parts.insert(0, "Several things about the checks against known answers, below. Those that say what to "
                                "give or change must be done" + ("; the last is a reader's findings, to follow where "
                                                                 "they are right." if findings else "."))
            failed = await self._revise_checks_only(
                "\n\n".join(p for p in parts if p),
                equations=True if review.get("equation_objections") else ("example" if example_asks else False),
                isolation=bool(review.get("isolation_objections")))
            if failed:
                self._log.warning("[oracle] %s", failed)
            after = self._planned_oracles()
            changes = _forms.describe_changes(before, after)
            equations_after = self._planned_equations()
            equation_changes = [f"equation {eid} was changed: it was `{was}`, it is now `{equations_after[eid]}`"
                                for eid, was in equations_before.items()
                                if eid in equations_after and equations_after[eid] != was]
            for line in equation_changes:
                self._log.warning("[oracle] after the reader's objection, the plan's %s", line)
            still_untested: list[str] = []
            if example_asks:
                design_after, _why = _plan.load_design(self.quest_root)
                protocol_after = design_after.get("protocol") if isinstance(design_after, dict) else None
                still_untested = _equation_tests.untested_notes(
                    [r for r in _equation_tests.example_rows(protocol_after if isinstance(protocol_after, dict) else None)
                     if r["state"] in ("missing", "unusable", "unprotected")], {})
                for note in still_untested:
                    self._log.warning("[plan] after the plan was asked once, %s", note)
            if len(after) < len(before):
                self._log.warning("[oracle] the plan has fewer checks against known answers after it was asked (%d, was %d)",
                                  len(after), len(before))
            text, again, left = _forms.apply_to_plan(path.read_text(encoding="utf-8"), fi_runs=fi_runs, precision_asks=False)
            if again:
                path.write_text(text, encoding="utf-8")
            # A tolerance the plan, asked once, left tighter than its expected value's own digits stays as written and
            # is said so (never widened). A check the reply ADDED that is too tight was not asked about: it is asked
            # about once, then said so the same way.
            widened = self._note_tight_tolerances(path, too_tight)
            widened += await self._precision_of_added_checks(path, too_tight)
            if widened:
                again = [*again, *widened]
            for r in left:
                self._log.warning("[oracle] still not in its kind's form after the plan was asked once: %s", r)
            self._write_plan_section(path, [
                "",
                *[f"- FI asked the plan to change a check: {r}" for r in requests],
                *(["- FI asked the plan, once, to look at what the second reader found."] if findings else []),
                *([f"- {failed[0].upper()}{failed[1:]}; nothing was changed for it."] if failed else
                  [f"- What changed: {c}." for c in changes] or ["- The plan did not change the checks."]),
                *[f"- {c[0].upper()}{c[1:]}." for c in equation_changes],
                *[f"- After the plan was asked once for a worked example, {n}." for n in still_untested],
                *[f"- {r}" for r in again],
                *[f"- Still not in its kind's form (read it before the run): {r}" for r in left],
            ], note="the checks against known answers looked at")
            self._oracle_review_write({**self._oracle_review_read(), "answered": True})
        formulas = self._apply_expected_formulas(path)
        end = self._oracle_review_read()
        if end.get("verdicts") and "after_look" not in end and (not end.get("asked") or end.get("answered")):
            # The checks as this one look left them, after what the reader found was applied: a later change to one is
            # a change the reader never read.
            design_now, _why = _plan.load_design(self.quest_root)
            now = design_now.get("protocol") if isinstance(design_now, dict) else None
            self._oracle_review_write({**end, "after_look": _review.fingerprints(now if isinstance(now, dict) else None)})
        return {"rewritten": rewrites, "asked": requests, "left": left,
                **({"expected_formula": {"asked": [f["name"] for f in formula_asks], "corrected": formulas}}
                   if formula_asks or formulas else {}),
                **({"review": {k: v for k, v in review.items() if k != "lines"}} if review else {})}

    async def _precision_of_added_checks(self, path: Path, asked: list[str]) -> list[str]:
        """A check the reply to the ask added that is tighter than its expected value's digits was never asked about: it is
        asked about once, for that check only, and said so (its tolerance is not changed) when that does not fix it. The sentences."""
        known = {str(n).strip().lower() for n in asked}
        added = [n for n in _forms.precision_gaps(path.read_text(encoding="utf-8")) if n.strip().lower() not in known]
        record = self._oracle_review_read()
        if not added or record.get("precision_added_asked"):
            return []
        self._oracle_review_write({**record, "precision_added_asked": added})
        self._log.warning("[oracle] the plan added a check whose tolerance is tighter than its expected value's digits; "
                          "asking about it once: %s", ", ".join(repr(n) for n in added))
        failed = await self._revise_checks_only(_forms.request(
            _forms.precision_requests(path.read_text(encoding="utf-8"), added)))
        if failed:
            self._log.warning("[oracle] %s", failed)
        return self._note_tight_tolerances(path, added, added=added)

    def _note_tight_tolerances(self, path: Path, names: list[str] | None = None, added: list[str] | None = None) -> list[str]:
        """Say, for each named check (the ones the plan was asked about) whose tolerance is still smaller than half a unit
        of the last digit of its expected value (:func:`core.oracle_forms.tight_tolerance_notes`), that FI kept it as the
        plan wrote it: in the log and in ``oracle_review.json`` (``tolerance_kept``, which the paper's limitations carry).
        A tolerance is never widened to the digits the plan typed. The sentences, or ``[]`` when there are none."""
        told = _forms.tight_tolerance_notes(path.read_text(encoding="utf-8"), names, added)
        if not told:
            return []
        for sentence in told:
            self._log.warning("[oracle] %s", sentence)
        record = self._oracle_review_read()
        self._oracle_review_write({**record, "tolerance_kept": [*record.get("tolerance_kept", []), *told]})
        return told

    def _simulation_ran(self) -> bool:
        """Whether the audit trace shows the experiment step ever started (the trace outlives a ``--from plan`` rerun,
        which moves the frozen protocol and the run's records aside): after that, a measured value may have been seen,
        so FI's own computation of expected values is not made (the rules for a check that failed apply)."""
        try:
            return any(e.get("kind") == "node_started" and e.get("node") == "execute"
                       for e in _audit_log.read(self.audit.path))
        except Exception:  # noqa: BLE001 -- a trace that cannot be read says nothing ran; the freeze guard still holds
            return False

    def _apply_expected_formulas(self, path: Path) -> list[dict[str, Any]]:
        """Before the protocol is frozen and before anything ran (the callers' guard), once per quest (``formulas`` in
        ``.fi/oracle_review.json``): a check whose ``expected_formula`` still gives another number than its ``expected``
        (by more than the typed number's written precision or the check's tolerance, see ``oracle_forms.disagrees``), takes the formula's value (the plan's own working, computed exactly;
        a number written from memory is the weaker of the two). The tolerance, mode, case and measure are never
        touched and no measured value is used. Recorded as the engine's change, never a person's. A check whose
        formula is still missing or unusable is left exactly as it was."""
        progress = self._oracle_review_read()
        if "formulas" in progress or _frozen.load(self.quest_root) is not None or self._simulation_ran():
            return [dict(c) for c in progress.get("formulas") or []]
        text = path.read_text(encoding="utf-8")
        refused: list[dict[str, Any]] = []
        new_text, corrections = _forms.apply_formulas(text, refused)
        design, _why = _plan.load_design(self.quest_root)
        protocol = design.get("protocol") if isinstance(design, dict) else None
        left = []
        for f in _forms.formula_findings(protocol if isinstance(protocol, dict) else None,
                                         _forms.fixed_settings_of_design(design)):
            if f["state"] == "arithmetic":
                self._log.warning("[oracle] the plan still states %s = %s for %r, which FI computes as %.10g",
                                  f["expression"], f["stated"], f["name"], f["value"])
                left.append(f"The plan states {f['expression']} = {f['stated']} for {f['name']!r}, which FI computes "
                            f"as {f['value']:.10g}; read the check before trusting it.")
            if f["state"] in ("missing", "unusable", "ambiguous"):
                why = ("the plan gave no formula for it" if f["state"] == "missing"
                       else f"its formula can be read two ways, so FI did not use it: {f.get('formula')}"
                       if f["state"] == "ambiguous" else f"its formula cannot be computed: {f.get('why')}")
                self._log.info("[oracle] FI could not compute the expected value of %r itself (%s); the check is "
                               "used as the plan wrote it", f["name"], why)
                left.append(f"FI could not compute the expected value of {f['name']!r} itself ({why}); the check is "
                            "used as the plan wrote it.")
        if refused:
            # An edit that would make plan.md unreadable is not made: the check keeps the number the plan wrote, and the
            # disagreement is recorded as unconfirmed (.fi/oracles_added.json), so the evidence and the paper disclose it.
            for r in refused:
                line = (f"the check {r['name']!r} was not changed: its formula gives {r['formula_value']:.10g} but the plan "
                        f"wrote {r['typed']:.10g}, and the plan with that change would no longer read ({_one_line(r['why'], 300)}); "
                        "the number the plan wrote is kept and the check counts as unconfirmed")
                self._log.warning("[oracle] %s", line)
                left.append(line[0].upper() + line[1:] + ".")
            earlier = self._oracles_added_read() or {}
            known = {str(d.get("name")): d for d in earlier.get("formula_disagreements") or [] if isinstance(d, dict)}
            known.update({str(r["name"]): {k: r[k] for k in ("name", "typed", "formula_value", "formula", "why")} for r in refused})
            self._oracles_added_write({**earlier, "formula_disagreements": list(known.values())})
            if not corrections:
                self._write_plan_section(path, ["", *[f"- {line}" for line in left]],
                                         note="expected values FI did not change")
        if left and not corrections and not refused:
            self._write_plan_section(path, ["", *[f"- {line}" for line in left]],
                                     note="expected values FI could not compute itself")
        if corrections:
            names = [c["name"] for c in corrections]
            lines = [f"before anything ran, FI computed the expected value of {c['name']!r} from the plan's own formula: "
                     f"{c['now']:.10g} (the plan had written {c['was']:.10g}); the check uses {c['now']:.10g}"
                     for c in corrections]
            for line in lines:
                self._log.warning("[oracle] %s", line)
            self._note_engine_change(names, reason=(
                "Before anything ran, FI computed each of these expected values itself from the plan's own formula and "
                "it differed from the number the plan wrote (by more than that number's own digits), so the check uses the "
                "formula's value (worked out before any run): "
                + "; ".join(f"{c['name']!r} {c['was']:.10g} -> {c['now']:.10g}" for c in corrections) + "."))
            path.write_text(new_text, encoding="utf-8")
            _plan.record_version(self.quest_root, new_text, by="engine",
                                 note="expected values computed by FI from the plan's own formulas")
            self._write_plan_section(path, ["", *[f"- {line[0].upper()}{line[1:]}." for line in lines],
                                            *[f"- {line}" for line in left]],
                                     note="expected values computed by FI from the plan's own formulas")
        self._oracle_review_write({**self._oracle_review_read(), "formulas": corrections})
        return corrections

    def _dry_run_path(self) -> Path:
        return self.fi_dir / "oracle_dry_run.json"

    async def _revise_after_dry_run(self, oracles: list[dict[str, Any]], record: dict[str, Any]) -> bool:
        """The oracle gate's first measurement, before the protocol is frozen, is a test run of the checks: a number whose
        size says the plan and the simulation mean different things by a check (:func:`core.oracle_forms.mismatch`) sends
        ONE targeted request to the plan (never a change to an expected value made here), and the checks it changes or
        removes are recorded as the engine's (the freeze says nobody approved them). A changed check that then passes
        on the test run's own numbers was fitted to what it was meant to check: it is recorded
        (``.fi/oracles_added.json`` ``fitted``, ``needs/ORACLE_CHECK.json`` ``fitted_to_test_run``), the evidence keeps
        a gap for it, and the improve step never optimises against it. ``True`` when the plan's checks changed and the
        gate should measure again. Asked at most once per quest."""
        if self._dry_run_path().is_file():
            return False
        found = _forms.mismatches(oracles, record.get("checks"))
        if not found:
            return False
        record["test_run_mismatches"] = found
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)
            self._dry_run_path().write_text(json.dumps({"mismatches": found, "at": _frozen.now()}, indent=2) + "\n",
                                            encoding="utf-8")
        except OSError as e:
            self._log.warning("[oracle] couldn't record the test run of the checks: %r", e)
        for why in found:
            self._log.warning("[oracle] a test run of the checks before the study: %s", why)
        print(f"[FI] A test run of the checks before the study: {len(found)} of them may mean something different in the "
              "plan and in the simulation; asking the plan to look at them once (see plan.md and .fi/run.log)")
        before = self._planned_oracles()
        # The names the simulation returned on the test run: a rewrite of a check may read only these and the plan's own
        # declared names (:func:`core.contract.unknown_measure_names`, applied where the rewrite is read).
        self._test_run_returned = {str(k) for c in record.get("checks") or [] if isinstance(c, dict)
                                   and isinstance(c.get("returned"), dict) for k in c["returned"]}
        try:
            failed = await self._revise_checks_only(_forms.dry_run_request(found))
        finally:
            self._test_run_returned = None
        after = self._planned_oracles()
        changes = _forms.describe_changes(before, after)
        changed = [name for name, oracle in after.items() if before.get(name) != oracle]
        removed = [name for name in before if name not in after]
        # Whether a changed check now passes on the test run's own numbers: a change that simply makes the measured value
        # pass is said as exactly that, so a person checks its reason, not its result.
        returned = {str(c.get("name") or "").strip(): c.get("returned") for c in record.get("checks") or []
                    if isinstance(c, dict) and c.get("measured_by") == "engine"}
        fits = [n for n in changed
                if _forms.passes_on(after[n], returned.get(n), (before.get(n) or {}).get("case")) is True]
        # Fitted: a change to the number a check is judged by (its expected value or tolerance) that makes the test run
        # pass. A change of form only (the measure, a unit, a sign) is what the test run is for, and is not fitted.
        fitted = [n for n in fits if any((before.get(n) or {}).get(k) != after[n].get(k)
                                         for k in ("expected", "tolerance", "tolerance_mode"))]
        if fitted:
            self._log.warning("[oracle] after the test run, %s pass on the test run's own numbers with a changed expected "
                              "value or tolerance: they count as unconfirmed", ", ".join(repr(n) for n in fitted))
        path = _plan.plan_path(self.quest_root)
        try:
            self._write_plan_section(path, [
                "", "A test run of the checks before the study, on each check's own case:", "",
                *[f"- {why[0].upper()}{why[1:]}." for why in found],
                *([f"- The plan was asked once to look at their definitions, and {failed}; nothing was changed."]
                  if failed else
                  [f"- The plan was asked once to look at their definitions. What changed: {c}." for c in changes]
                  or ["- The plan was asked once to look at their definitions, and kept them as they were."]),
                *[f"- With this change the check {n!r} passes on the test run's own numbers: check the reason for the "
                  "change, not the result." for n in fits],
            ], note="a test run of the checks before the study")
        except OSError as e:
            self._log.warning("[oracle] couldn't write the test run of the checks into plan.md: %r", e)
        for c in changes:
            self._log.warning("[oracle] after the test run the plan changed: %s", c)
        if failed:
            self._log.warning("[oracle] after the test run, %s", failed)
        if not changed and not removed:
            self._log.info("[oracle] after the test run the plan kept its checks as they were")
            return False
        # The engine changed the plan after it was written: a person reads that before the freeze, or the freeze record
        # says nobody approved it (as for a check the gate adds).
        self._note_engine_change(changed, removed, reason=(
            "A test run of the checks before the study measured numbers that did not fit their definitions, and the "
            "plan was asked once to look at them: " + "; ".join(changes) + "."
            + (f" With the change, {', '.join(repr(n) for n in fits)} pass on the test run's own numbers: check the "
               "reason for the change, not the result." if fits else "")))
        if fitted:
            record = self._oracles_added_read() or {}
            self._oracles_added_write({**record, "fitted": sorted({*record.get("fitted", []), *fitted})})
        return True

    async def _repair_script_for_oracle(
        self, state: QuestState, path: Path, oracles: list[dict[str, Any]], found: list[str], stderr_tail: str = "",
        *, disputed: list[str] | None = None, passing: list[str] | None = None,
    ) -> tuple[str | None, bool, str]:
        """ONE repair of the script for its oracle checks: ``(new code or None, the repair call itself failed, what came of
        it)``; the last is ``applied``, ``set_aside_disputed``, ``no_code``/``not_usable`` or ``call_failed``, and goes in
        the record. Kept only if it parses and now mentions FI_ORACLE. The oracle run's own stderr goes with it: "printed no
        ORACLE_JSON line" says only that the script failed, and a repair asked without the traceback rewrote the oracle
        branch while the crash was elsewhere (a module-level read, a library call) and stayed.

        ``disputed``: the checks an earlier repair already called wrong, left out of the request. A repair that calls
        another check wrong, one not passing now (``passing``), has its code set aside: that code cannot be split per
        check, and code written by a repair that believes the check is wrong is code bent towards (or around) it. A
        proposal about a check that passes is recorded, and its fix for the rest is still used."""
        code = path.read_text(encoding="utf-8")
        self._oracle_patch_summary = ""
        already = {str(n).strip().lower() for n in [*(disputed or []), *(passing or [])]}
        prompt = self._prompts["execute_reflect"].substitute(
            topic_block=self._topic_block(), own_failure_block="",
            previous_code=code,
            returncode="(the oracle run did not pass)",
            # The contract of the layout this quest has: a two-script quest's simulation is a function FI calls, with
            # no FI_ORACLE run and no ORACLE_JSON line (saying both contracts sent repairs to write a branch nothing reads).
            stdout_tail=_oracle.directive(oracles, found, disputed=disputed, trial=bool(getattr(self, "_trial_mode", False)),
                                          frame=path.name == _split_run.SIMULATE_NAME and _skeleton.has_frame(code))
            + ((_skeleton.ORACLE_NOTE if path.name == _split_run.SIMULATE_NAME and _skeleton.has_frame(code)
                else _TRIAL_ORACLE_NOTE) if getattr(self, "_trial_mode", False) else "")
            # What each run is given and returns when the plan compares values inside each run (the check's case names
            # one of them): without it a repair guesses a form and the case reads the wrong value.
            + _with_newline(_skeleton.level_rule(self._protocol_block(state), code))
            # A check against a known answer tests the equations, which may be in the model's package.
            + (self._package_repair_note(state) if path.name == _split_run.SIMULATE_NAME else ""),
            stderr_tail=stderr_tail,
            duration_s="0.00",
            figures_count="0",
            result_json_present="no (not run yet)",
            reflect_history_block=_format_reflect_history([]),
            design_block=self._code_design_block(state),
            clarify_block=_format_clarify(state),
        )
        prompt = _skeleton.framed_prompt(code if path.name == _split_run.SIMULATE_NAME else "", prompt)
        try:
            text = await self._chat(prompt, node="implement_oracle")
        except Exception as exc:  # noqa: BLE001 -- a repair is best-effort
            self._log.warning("[oracle] the repair call failed (%r); keeping %s as written", exc, path.name)
            return None, True, "call_failed"
        parsed = _repair_reply_json(text, node="implement_oracle")
        self._oracle_patch_summary = " ".join(str(parsed.get("patch_summary") or "").split())[:300]
        newly_disputed = fi_value_disputed = False
        protocol_now = self._protocol_block(state)
        for proposal in _oracle.proposals(parsed.get("oracle_change"), oracles):
            own = next((o for o in oracles if str(o.get("name") or "").strip() == proposal["name"]), None)
            if own is not None and _oracle_triage.formula_value(own, protocol_now if isinstance(protocol_now, dict) else None) is not None:
                # FI computed this check's expected value from the plan's own formula: still only the plan's working. The
                # repair may question it -- naming the step and giving a formula FI computes -- and the plan's model must
                # answer (core/oracle_challenge.py); the repair's code is not used either way.
                fi_value_disputed = True
                challenge, why_not = _oracle_challenge.challenge_of(
                    own, proposal, self._oracle_measured.get(proposal["name"]),
                    constants=_forms.fixed_settings(protocol_now if isinstance(protocol_now, dict) else None))
                if challenge is None or proposal["name"] in self._oracle_challenges:
                    self._log.warning("[oracle] the repair proposes expected %s for '%s', whose expected value FI computed "
                                      "from the plan's own formula; the proposal is not taken (%s)", proposal["expected"],
                                      proposal["name"], why_not or "the plan has answered this check's question already")
                    continue
                self._oracle_challenges[proposal["name"]] = challenge
                self._oracle_proposals[proposal["name"]] = {**proposal, "expected": challenge["value"],
                                                            "source": "challenge"}
                self._oracle_disputed.add(proposal["name"])
                self._log.warning("[oracle] the code questions the plan's working for '%s' (the step: %s; its formula gives "
                                  "%s): the plan's model is asked to answer", proposal["name"], challenge["step"][:200],
                                  _oracle.fmt_digits(challenge["value"]))
                continue
            self._oracle_proposals[proposal["name"]] = proposal
            if proposal["name"].strip().lower() not in {str(n).strip().lower() for n in passing or []}:
                self._oracle_disputed.add(proposal["name"])
            newly_disputed = newly_disputed or proposal["name"].strip().lower() not in already
            self._log.info(
                "[oracle] the repair says the check '%s' is what is wrong and proposes expected %s, tolerance %s (%s): %s",
                proposal["name"], proposal["expected"], proposal["tolerance"], proposal["tolerance_mode"], proposal["reason"][:200],
            )
        if newly_disputed:
            self._log.info(
                "[oracle] kept %s as it was: a repair that says a check is wrong is not used to rewrite the script", path.name,
            )
            return None, False, "set_aside_disputed"
        if fi_value_disputed:
            # Written against an expected value FI computed itself: a spent repair, its code not used (the check still fails).
            self._log.info("[oracle] kept %s as it was: the repair was written against a value FI computed from the plan's "
                           "own formula", path.name)
            return None, False, "disputed_fi_value"
        new_code = parsed.get("code")
        gives_functions = isinstance(parsed.get("functions"), str) and bool(parsed["functions"].strip())
        if not (isinstance(new_code, str) and new_code.strip()) and not gives_functions:
            package_was = self._package_snapshot() if path.name == _split_run.SIMULATE_NAME else {}
            if (path.name == _split_run.SIMULATE_NAME and self._package_repair_files(state, parsed)
                    and self._apply_package_repair(state, parsed, node="oracle")):
                # The fault was in the model's package: that is fixed, simulate.py stays as it is. The study's entry must
                # still return the study's results with it.
                fault = await self._simulation_kept_or_back(state, path, code, package_was, "oracle")
                if fault:
                    await self._keep_refused_repair("oracle", path.name, code, fault)
                    return None, False, "not_usable"
                self._log.info("[oracle] %s is kept as it is: the repair fixed the model's package (%s)", path.name,
                               str(parsed.get("patch_summary") or "no summary")[:160])
                return code, False, "applied"
            if parsed and not parsed.get("code"):
                return None, False, "no_code"  # an answer with nothing to change
            new_code, _deps = _parse_implement_response(text)
        # One rule decides whether a repaired script still keeps FI's contract (what FI calls it by, its result line, the
        # oracle marker or cases): `repair_breaks_contract`; its reason is logged.
        self._keep_repair_answer("oracle", text)
        new_code, split_why = self._script_from_reply("oracle", code, new_code, parsed, path.name)
        broken = split_why or _trial_runner.repair_breaks_contract(code, new_code, path.name, oracles=oracles,
                                                                   trial=bool(getattr(self, "_trial_mode", False)))
        if broken:
            self._log.warning("[oracle] the repair of %s is not usable (%s); keeping it as written", path.name, broken)
            await self._keep_refused_repair("oracle", path.name, new_code, broken)
            return None, False, "not_usable"
        package_before = self._package_snapshot() if path.name == _split_run.SIMULATE_NAME else {}
        path.write_text(new_code, encoding="utf-8")
        if path.name == _split_run.SIMULATE_NAME:
            self._apply_package_repair(state, parsed, node="oracle")
            # run_trial is ONE trial, and the function FI calls for the study must still give the study's results: FI calls
            # the repaired code (a run_trial that returns run_cell's mean of many trials made every trial as long as a whole
            # cell; one that sends every setting to the check branch left the whole search with no objective).
            fault = await self._simulation_repair_fault(state, new_code)
            if fault:
                path.write_text(code, encoding="utf-8")
                self._restore_package(package_before)
                self._log.warning("[oracle] the repair of %s is not used: %s; keeping it as written", path.name, fault)
                await self._keep_refused_repair("oracle", path.name, new_code, fault)
                return None, False, "not_usable"
        elif fault := await self._gate_refusal(state):  # the analysis script: the analysis gate is its check
            path.write_text(code, encoding="utf-8")
            self._log.warning("[oracle] the repair of %s is not used: %s; keeping it as written", path.name, fault)
            await self._keep_refused_repair("oracle", path.name, new_code, fault)
            return None, False, "not_usable"
        self._gate_commit(state)
        self._log.info(
            "[oracle] rewrote %s (%d bytes): %s", path.name, len(new_code),
            str(parsed.get("patch_summary") or "no summary")[:160],
        )
        return new_code, False, "applied"

    def _stop_for_protocol(self, mismatches: list[Any]) -> None:
        """The script still contradicts the plan's protocol after the repairs: the difference goes to the record of problems
        FI could not repair (:meth:`_record_cannot_repair`) with what it rests on, which decides whether the run step
        ends in the honest stop. Nobody is asked to edit it."""
        differences = [m.message() for m in mismatches]
        self._record_cannot_repair(
            "the experiment's script differs from the protocol the plan fixed: " + "; ".join(differences),
            ["compared the script with the protocol the plan fixed", *[_one_line(d, 300) for d in differences[:3]],
             f"asked the model to repair the script up to {int(self.config.engine.protocol_repair_attempts)} time(s); "
             "the difference stood"], source=_protocol.SOURCE, check="protocol")

    def _adopt_scripts_fixed_by_hand(self, state: QuestState) -> QuestState | None:
        """After a protocol stop, keep the scripts on disk when they now agree with the plan (a person edited them, or
        the plan), instead of writing them again. ``None`` when the quest was not stopped for the protocol, or they
        still differ (the model then writes the script again, told what differed)."""
        marker = self.fi_dir / "protocol_stop.json"
        if not marker.is_file():
            return None
        protocol = self._protocol_of(state)
        scripts = self._scripts_on_disk()
        if protocol is None or "experiment.py" not in scripts or _protocol.check(protocol, scripts):
            marker.unlink(missing_ok=True)  # the script is written again: a later pass must not adopt the old one
            return None
        try:
            deps = _coerce_dep_list(json.loads(marker.read_text(encoding="utf-8")).get("deps"))
            marker.unlink(missing_ok=True)
        except (OSError, ValueError):
            deps = []
        deps = sorted({*deps, *_coerce_dep_list((state.get("design") or {}).get("dependencies"))})
        self._log.info("[protocol] the script on disk now agrees with the plan's protocol; keeping it (not written again)")
        self._protocol_record({"status": "ok", "adopted": "the script on disk after the stop", "protocol": protocol})
        return {"code": scripts["experiment.py"], "deps": deps}

    async def _repair_ignored_replicate_seed(
        self, state: QuestState, code_path: Path, code: str, deps: list[str],
    ) -> tuple[str, list[str]]:
        """Ask for ONE repair of a script whose randomness the seed cannot reach.

        Two shapes, one call. A script that never reads ``FI_REPLICATE_SEED``
        makes every replicate run the same run: ``execute`` finds the results
        identical, publishes no aggregate, and the paper can report one
        measurement with no interval (four of six codex quests did exactly
        that). A script that reads it and still builds a generator from OS
        entropy has the opposite failure and hides it better: the replicates
        differ, so an aggregate is published, but the seed reached none of the
        randomness and no run can be reproduced. The instruction not to write a
        seed of its own is in the implement prompts; a model that ignores it
        either way is asked again, once, here.

        The request reuses the ``execute_reflect`` template (the directive
        stands where a traceback would) and is applied the way its patches
        are, but it is not a repair iteration: the node's
        ``exec_reflect_*`` budget is untouched. Never a call when replication
        is off, for a background job (no replicate runs are made), or when the
        script already names the variable.

        A repair is kept only if it parses as Python and the script would no
        longer be sent back -- it reads the seed AND builds no generator
        without one. A model that rewrote the script and still missed the
        target, or returned half of it, would leave a script worse than the one
        it had, so the original stays and the fallbacks in ``execute`` (the
        single-measurement one, or the warning that the runs cannot be
        reproduced) handle it. Failure never blocks the quest: one warning,
        then on.
        """
        reads_seed = _script_reads_replicate_seed(code_path)
        unseeded = _unseeded_rng_calls(code_path)
        if (
            self.config.execution.background_jobs
            or max(1, int(self.config.engine.execute_replicates)) <= 1
            or state.get("no_simulation_resolved")
            or state.get("survey_mode_resolved")
            or (reads_seed and not unseeded)
            or (not reads_seed and not _script_has_random_source(code_path))
        ):
            return code, deps
        if reads_seed:
            self._log.info(
                "[implement] %s reads FI_REPLICATE_SEED but builds a generator with no "
                "seed (%s), so none of its %d replicate runs could be reproduced; "
                "asking for one repair",
                code_path.name,
                "; ".join(f"line {line}: {expr}" for line, expr in unseeded[:6]),
                self.config.engine.execute_replicates,
            )
        else:
            self._log.info(
                "[implement] %s never reads FI_REPLICATE_SEED, so its %d replicate runs "
                "would repeat one run; asking for one repair",
                code_path.name, self.config.engine.execute_replicates,
            )
        prompt = self._prompts["execute_reflect"].substitute(
            topic_block=self._topic_block(), own_failure_block="",
            # Whole: a repair that returns the script must not lose the tail of it.
            previous_code=code,
            returncode="(not run yet)",
            stdout_tail=_unseeded_rng_directive(unseeded) if reads_seed else _SEED_REPAIR_DIRECTIVE,
            stderr_tail="",
            duration_s="0.00",
            figures_count="0",
            result_json_present="no (not run yet)",
            reflect_history_block=_format_reflect_history([]),
            design_block=self._code_design_block(state),
            clarify_block=_format_clarify(state),
        )
        why = ""
        prompt = _skeleton.framed_prompt(code if code_path.name == _split_run.SIMULATE_NAME else "", prompt)
        try:
            text = await self._chat(prompt, node="implement_seed")
        except Exception as exc:  # noqa: BLE001 -- a repair is best-effort
            text, why = "", f"the repair call failed ({exc!r})"
        parsed = _repair_reply_json(text, node="implement_seed")
        new_code, new_deps = parsed.get("code"), _coerce_dep_list(parsed.get("deps"))
        if not (isinstance(new_code, str) and new_code.strip()):
            new_code, new_deps = _parse_implement_response(text)  # fence drift
        if not why:
            self._keep_repair_answer("seed", text)
            new_code, split_why = self._script_from_reply("implement", code, new_code, parsed, code_path.name)
            broken = split_why or _trial_runner.repair_breaks_contract(code, new_code, code_path.name)
            if broken:
                why = f"the model returned no usable script ({broken})"
            elif "FI_REPLICATE_SEED" not in new_code:  # the test _script_reads_replicate_seed applies
                why = "the repaired script still does not read it"
            elif unseeded_rng_calls(new_code):
                why = "the repaired script still builds a generator without a seed"
        if why:
            await self._keep_refused_repair("seed", code_path.name, new_code, why)
            if reads_seed:
                self._log.warning(
                    "[implement] %s: keeping the script as written. Its replicates will "
                    "differ, so the quest keeps its aggregate, but the randomness is "
                    "drawn from OS entropy and a rerun will not reproduce these numbers",
                    why,
                )
            else:
                self._log.warning(
                    "[implement] %s: keeping the script as written. Its replicates will "
                    "repeat one run, so the quest will be reported as a single "
                    "measurement with no confidence interval", why,
                )
            if self.config.rigor_profile == "research":
                # This repair failing used to let a research run finish on results nobody could reproduce (or a single
                # measurement reported as if it were replicated). The quest ends in the honest stop at the run step; the
                # script is not run, and nobody is asked to fix it by hand.
                self._record_cannot_repair(
                    "the script's randomness cannot be traced to FI_REPLICATE_SEED, which a research run requires: " + why,
                    ["read the script for the seed FI gives each repeat",
                     "asked the model for one repair: " + _one_line(why, 300)], source=_stop_power.READ, check="seed")
            return code, deps
        code_path.write_text(new_code, encoding="utf-8")
        self._log.info(
            "[implement] the script now seeds every generator it builds from "
            "FI_REPLICATE_SEED (%s); rewrote %s (%d bytes)",
            str(parsed.get("patch_summary") or "no summary")[:120], code_path, len(new_code),
        )
        return new_code, sorted({*deps, *new_deps})

    async def _gate_pass(self, state: QuestState, py: Any, exec_env: Any, seed_path: Path) -> tuple[
            QuestState | None, str | None, bool]:
        """The checks made on the code before it runs, in order: typed-in results, the cheap static faults, the equations'
        labels, the code review, one small test per equation, the known-answer (oracle) checks, each repair of one read again by
        the checks before it. ``(a stop for the quest or None, the script as the oracle gate repaired it or None, whether FI
        rewrote the code)``. Run once before the sizing and the freeze, and again by :meth:`_verify_gated_code` when the code
        was changed afterwards."""
        await self._gate_baseline(state)  # the analysis gate's baseline: the code as it is (nothing with the switch off)
        await self._results_from_computation(state)
        # Cheap faults a person finds in a minute, read before anything runs (core/static_checks.py).
        if (static_stop := await self._static_gate(state)) is not None:
            return static_stop, None, False
        # A function of derive.py that is still FI's stub, for a result the plan asks for (core/analysis_frame.py).
        if (stub := self._fis_stub_sendback(state)) is not None:
            return stub, None, False
        # Where the simulation implements each equation the plan's model computes the data with (`# E1`): read before
        # anything runs, so a stop for it spends nothing and a resume reads the script a person labelled.
        stop = self._cannot_repair_stop() or self._code_problems_stop(state)
        if stop is not None:
            return stop, None, False
        # One small test per equation of the model, from the plan's own worked example, before anything else runs: a
        # function that computes another equation than the plan's is found here, not after the study.
        # FI runs the simulation on a small case first: a crash goes to the repair of a failed run, and the code is read
        # with what the run gave (core/small_run.py).
        if (small := await self._small_run_gate(state, py, exec_env)) is not None:
            return small, None, False
        if (gated := await self._gate_first(state)) is not None:  # the analysis on the recorded results (switch off: None)
            return gated, None, False
        await self._code_review_gate(state)
        failed_equations = await self._equation_gate(state, py, exec_env)
        if failed_equations is not None:
            return failed_equations, None, False
        # A second, independently written version of the model's key functions, compared with the first on many inputs
        # (core/second_impl.py). It never stops the quest; a repair of a function it makes is read again just below, and the
        # receipt and the final reading of the code see the package it left.
        second_rewrote = await self._second_version_gate(state, py, exec_env)
        if second_rewrote:
            # A function was repaired: every equation test is run again on the package it left (a repair that fails one is
            # already undone inside the check; this also covers a repair kept before a restart).
            failed_equations = await self._equation_gate(state, py, exec_env)
            if failed_equations is not None:
                return failed_equations, None, False
            if (stop := self._code_problems_stop(state)) is not None:
                return stop, None, False
        # A repair of one equation's function is a new text of the simulation: read again for typed-in results (the same
        # text is never read twice), before the oracle gate tests it.
        await self._results_from_computation(state)
        if (static_stop := await self._static_gate(state)) is not None:
            return static_stop, None, False
        before_gate = seed_path.read_text(encoding="utf-8") if seed_path.is_file() else ""
        package_before_gate = self._package_snapshot()
        oracle_code = await self._oracle_gate(state, py, exec_env, seed_path)
        # FI's own repair: of either script or of the model's package (a script found changed on disk below may be a
        # person's edit during a stop, so it is not counted here).
        gate_rewrote = (seed_path.is_file() and seed_path.read_text(encoding="utf-8") != before_gate) or (
            self._package_snapshot() != package_before_gate)
        gate_repaired = oracle_code is not None or gate_rewrote or second_rewrote
        if gate_rewrote:
            # An oracle repair rewrote the simulation or the model's package: its labels and layout are read again.
            # ... and the oracle gate's own repair is read for typed-in results too; if that rewrote the simulation again,
            # the gate tests the text that will run.
            oracle_code, unchecked = await self._typed_results_stable(state, py, exec_env, seed_path, oracle_code)
            if unchecked is not None:
                return unchecked, None, False
            stop = self._code_problems_stop(state)
            if stop is not None:
                return stop, None, False
        return None, oracle_code, gate_repaired

    async def _verify_gated_code(self, state: QuestState, py: Any, exec_env: Any, seed_path: Path) -> tuple[
            QuestState | None, str | None, bool]:
        """Before the run: the code is the version the checks were made for (core/gate_receipt.py). Only the drawing changed:
        the receipt is renewed. The computation changed (a rewrite after the checks, a person's edit): the checks are made
        again on this version, and the run starts only on a version they passed. ``(a stop, the repaired script or None,
        whether the checks rewrote the code)``."""
        said = _gate_receipt.verdict(self.quest_root)
        if said in (_gate_receipt.SAME, _gate_receipt.DRAWING_ONLY):
            if said == _gate_receipt.DRAWING_ONLY:
                self._log.info("[execute] only the drawing of the figures changed since the checks: the checks stand")
            return None, None, False
        self._log.warning(
            "[execute] the code that would run is not the version the checks were made for (%s): making the checks again on it "
            "before it runs", "no record of the checks" if said == _gate_receipt.NONE else "its computation changed")
        stop, repaired, rewrote = await self._gate_pass(state, py, exec_env, seed_path)
        if stop is not None:
            return stop, None, rewrote
        _gate_receipt.write(self.quest_root)
        return None, repaired, rewrote

    def _first_run_env(self, exec_env: dict[str, str] | None, *, record: bool = True) -> dict[str, str]:
        """The environment the study's first run is given: ``exec_env`` with the first replicate's seed and index, and the
        phased stage's seed. The check that runs the study in shared processes is given exactly this, so that its first run
        can stand for the run (``GroupedRun.problem``); ``record=False`` works it out without counting a run."""
        env = _replicate_env(exec_env, 0, max(1, int(self.config.engine.replicate_seed_stride)))
        if _phased.enabled(self.config):
            env = _phased.seed_env(self.quest_root, env, record=record)
        return env
