"""Two checks made on the simulation before and while it runs, as methods the engine mixes in.

* **A reported result is computed** (``core/typed_results.py``): the simulation is read, never run, and a returned value that
  is a number typed into the code is sent back to be computed (or removed), at most twice; one that is still typed in
  afterwards is left out of the results the paper may use, and the paper says so.
* **The run fits the time it is allowed** (``core/run_estimate.py``): a few real trials are timed before the freeze and
  added up; a study that would not finish is shrunk by the plan's model (fewer runs per setting, fewer or coarser
  settings, with the measured numbers in the request; nothing that decides whether a result is right may change), at most
  twice; one that still would not fit stops the quest plainly with no paper.
* **The simulation measures what the protocol names** (the same trials, ``RunChecksMixin._pilot_gate``): what the timed trials
  returned is read before the full run; a metric that is not a number in any pilot trial at any pilot setting (after the
  settings with few values were tried in every combination), one that never moves while others do, or an output that is the
  same everywhere goes to the existing repair of the simulation with the exact finding, and what the repairs leave ends in
  the honest stop in minutes instead of after the study.

They live here and not in ``engine.py`` so the node only calls them. Every method uses the engine's own attributes
(``quest_root``, ``fi_dir``, ``config``, ``_log``, ``_chat`` ...).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from . import frozen_protocol as _frozen
from . import oracle_forms as _forms
from . import plan as _plan
from . import run_estimate as _estimate
from . import split_run as _split_run
from . import trial_runner as _trial_runner
from . import typed_results as _typed

#: How many times, in all, a simulation that returns typed-in numbers is sent back.
TYPED_RESULT_REPAIRS = 2
TYPED_RECORD = "TYPED_RESULTS_CHECK.json"
NO_DIFFERENCE_RECORD = "NO_DIFFERENCE_OBSERVED.json"
_COUNTER = "typed_results.json"


def _one(text: str, limit: int = 400) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit - 3] + "..."


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class RunChecksMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    # ---- a reported result is computed --------------------------------------------------------------------------------

    def _typed_result_exempt(self, state: Any) -> set[str]:
        """Names the plan fixes (its fixed settings, the grid's settings, the thresholds): echoed back, not results."""
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        design = state.get("design") if isinstance(state.get("design"), dict) else {}
        names = set(_forms.fixed_settings_of_design({**design, "protocol": protocol}))
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        names |= {str(k) for k in grid}
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
        return names | {str(k) for k in thresholds}

    def _typed_results_in_simulation(self, state: Any) -> tuple[Path, list[_typed.TypedResult]]:
        path = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
        if not path.is_file():
            return path, []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return path, []
        return path, _typed.typed_results(text, path.name, exempt=self._typed_result_exempt(state))

    def _typed_counter(self, text: str) -> int:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record = _read_json(self.fi_dir / _COUNTER)  # type: ignore[attr-defined]
        return int(record.get("used") or 0) if record.get("sha") == sha else 0

    def _save_typed_counter(self, text: str, used: int) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
            (self.fi_dir / _COUNTER).write_text(  # type: ignore[attr-defined]
                json.dumps({"sha": hashlib.sha256(text.encode("utf-8")).hexdigest(), "used": used}), encoding="utf-8")
        except OSError:
            pass

    async def _results_from_computation(self, state: Any) -> bool:
        """Send a simulation that returns numbers typed into its code back to compute them, then leave out what is still
        typed in. ``True`` when the simulation was rewritten. The left-out quantities are this pass's alone: the record of
        an earlier pass goes first."""
        record_path = self.quest_root / "needs" / TYPED_RECORD  # type: ignore[attr-defined]
        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
        try:
            sha = hashlib.sha256(simulate.read_bytes()).hexdigest() if simulate.is_file() else ""
        except OSError:
            sha = ""
        # Asked again for every later text of the script (a gate's repair, a repair of the run, a new round), never for
        # one that was read already: its answer (the left-out quantities and their record) stands.
        if sha and getattr(self, "_typed_checked_sha", None) == sha:
            return False
        record_path.unlink(missing_ok=True)
        self._typed_result_keys: set[str] = set()
        path, found = self._typed_results_in_simulation(state)
        if not found:
            self._typed_checked_sha = sha
            return False
        rewrote = False
        text = path.read_text(encoding="utf-8")
        used = self._typed_counter(text)
        while found and used < TYPED_RESULT_REPAIRS:
            used += 1
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] %s returns numbers typed into the code (%s); asking for them to be computed (%d of %d)",
                path.name, "; ".join(f.says() for f in found[:4]), used, TYPED_RESULT_REPAIRS)
            rewrote = await self._repair_typed_results(state, path, found) or rewrote
            text = path.read_text(encoding="utf-8")
            self._save_typed_counter(text, used)
            path, found = self._typed_results_in_simulation(state)
        self._typed_checked_sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        if not found:
            return rewrote
        keys = sorted({f.key for f in found})
        self._typed_result_keys = set(keys)
        note = _typed.plain_note(keys, found)
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(_typed.record(keys, found), indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", TYPED_RECORD, exc)  # type: ignore[attr-defined]
        self._log.warning(  # type: ignore[attr-defined]
            "[execute] %s The paper does not use %s and says so in its limitations.", note,
            "it" if len(keys) == 1 else "them")
        return rewrote

    async def _repair_typed_results(self, state: Any, path: Path, found: list[_typed.TypedResult]) -> bool:
        """ONE repair of the simulation: kept only if it parses, keeps what FI reads from it, and has fewer typed-in
        results than before."""
        import ast

        from . import engine as _e

        code = path.read_text(encoding="utf-8")
        prompt = self._prompts["execute_reflect"].substitute(  # type: ignore[attr-defined]
            topic_block=self._topic_block(),
            previous_code=code, returncode="(not run yet)", stdout_tail=_typed.directive(found), stderr_tail="",
            duration_s="0.00", figures_count="0", result_json_present="no (not run yet)",
            reflect_history_block=_e._format_reflect_history([]),
            design_block=json.dumps(state.get("design") or {}, indent=2), clarify_block=_e._format_clarify(state),
        )
        try:
            text = await self._chat(prompt, node="execute_reflect")  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 -- a repair is best-effort
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the call to compute the typed-in results failed (%r); keeping %s as written", exc, path.name)
            return False
        parsed: dict[str, Any] = {}
        if _e._strip_outer_fence(text).lstrip().startswith("{"):
            parsed = _e._parse_json_lenient(text, node="execute_reflect") or {}
        new_code = parsed.get("code")
        if not (isinstance(new_code, str) and new_code.strip()):
            new_code, _deps = _e._parse_implement_response(text)
        try:
            ast.parse(new_code)
            usable = bool(new_code.strip()) and all(mark in new_code for mark in _typed.KEEP_MARKS if mark in code)
        except (SyntaxError, ValueError):
            usable = False
        if not usable:
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the rewrite of %s is not the same simulation with its results computed; keeping it as written",
                path.name)
            return False
        left = _typed.typed_results(new_code, path.name, exempt=self._typed_result_exempt(state))
        if len(left) >= len(found):
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the rewritten %s still returns %d number(s) typed into the code; keeping it as written",
                path.name, len(left))
            return False
        path.write_text(new_code, encoding="utf-8")
        self._log.info(  # type: ignore[attr-defined]
            "[execute] rewrote %s so its results are computed (%d typed-in left): %s", path.name, len(left),
            str(parsed.get("patch_summary") or "no summary")[:120])
        return True

    #: The most times the oracle gate and the typed-results check hand the simulation to each other in one pass.
    TYPED_GATE_ROUNDS = 3

    async def _typed_results_stable(self, state: Any, py: Any, env: Any, seed_path: Path,
                                    oracle_code: str | None) -> tuple[str | None, dict[str, Any] | None]:
        """After an oracle repair rewrote the simulation: read the text it left for typed-in results, and when that repair
        rewrote it again run the oracle gate on the result, and read what that left, until the text is one the check has
        read (each step is bounded by its own budget per text). ``(the repaired code for the state, None)``; a text still
        unread when the rounds are spent is not run: ``(code, the quest's stop)``."""
        for _ in range(self.TYPED_GATE_ROUNDS):
            if not await self._results_from_computation(state):
                return oracle_code, None  # the text is read: nothing to rewrite, or nothing it could rewrite
            again = await self._oracle_gate(state, py, env, seed_path)  # type: ignore[attr-defined]
            oracle_code = again if again is not None else seed_path.read_text(encoding="utf-8")
            self._check_equation_labels(state)  # type: ignore[attr-defined]
            self._check_code_layout(state)  # type: ignore[attr-defined]
        if self._typed_text_unread(seed_path):
            text = ("the simulation kept being rewritten by the known-answer checks' repair and by the repair of its typed-in "
                    "results, and the text FI would run was not checked for typed-in results")
            return oracle_code, self._stop_before_run(text, "not_run")
        return oracle_code, None

    def _typed_text_unread(self, seed_path: Path) -> bool:
        try:
            sha = hashlib.sha256(seed_path.read_bytes()).hexdigest() if seed_path.is_file() else ""
        except OSError:
            return True
        return bool(sha) and getattr(self, "_typed_checked_sha", None) != sha

    def _left_out_quantities(self) -> set[str]:
        """The quantities this pass's analysis and results must not use (still typed into the code after the repairs)."""
        typed = getattr(self, "_typed_result_keys", None)
        if typed is None:
            # A fresh process (a resume at a later node): the pass that left them out wrote its record, which is what stands.
            typed = {str(k) for k in _read_json(self.quest_root / "needs" / TYPED_RECORD).get("removed_quantities") or []}  # type: ignore[attr-defined]
        return set(typed) | self._result_check_keys()  # type: ignore[attr-defined]

    def _without_typed_results(self, result_json: Any) -> Any:
        """``result_json`` without the left-out quantities."""
        keys = self._left_out_quantities()
        return _typed.strip_keys(result_json, keys) if keys and result_json else result_json

    def _lines_without_left_out(self, lines: list[str]) -> tuple[list[str], int, list[str]]:
        """``lines`` without those that name a left-out quantity: the kept lines, how many were taken out, and the names."""
        keys = sorted(self._left_out_quantities())
        patterns = [re.compile(r"(?<![A-Za-z0-9_])" + re.escape(k) + r"(?![A-Za-z0-9_])") for k in keys]
        kept = [line for line in lines if not any(p.search(line) for p in patterns)]
        return kept, len(lines) - len(kept), keys

    def _taken_out_note(self, dropped: int, keys: list[str], where: str) -> str:
        """The plain note that says how many lines of ``where`` were taken out, and why."""
        names = ", ".join(f"`{k}`" for k in keys)
        return (f"[FI NOTE] {dropped} line(s) of the run's {where} that name {names} were taken out: "
                f"{'that quantity was' if len(keys) == 1 else 'those quantities were'} left out of the results "
                "(typed into the code, or failed a check of what ran) and must not be used, derived from or renamed.")

    def _stdout_for_models(self, stdout: str) -> str:
        """What a model may read of a run's output: the output without its ``RESULT_JSON`` line (the verified results reach a
        model once, as FI's accepted projection of them, with the left-out quantities taken out) and without any line that
        names a left-out quantity, with a plain note saying what was taken out and why. A bare left-out value is replaced
        too. The raw output stays in the records (``.fi/last_run_stdout.txt``)."""
        text = stdout or ""
        lines = text.splitlines()
        body = [line for line in lines if not line.lstrip().startswith("RESULT_JSON:")]
        kept, dropped, keys = self._lines_without_left_out(body)
        notes: list[str] = []
        if len(body) != len(lines):
            notes.append("[FI NOTE] The run's RESULT_JSON line is not repeated here: the results block is FI's accepted "
                         "projection of the results, and the only place to read them.")
        if dropped:
            notes.append(self._taken_out_note(dropped, keys, "output"))
        out = "\n".join([*kept, *notes]) if notes else text
        return self._withhold_left_out_values(out)

    def _stderr_for_models(self, stderr: str) -> str:
        """What a model may read of a run's error output: every line except those that name a left-out quantity (the
        traceback and the rest stay readable), with a plain note saying how many were taken out. A bare left-out value is
        replaced too."""
        text = stderr or ""
        lines = text.splitlines()
        # FI's own messages in the error output (why a run was refused: "FI ran no trial successfully: ...", "FI's run gave
        # a result that is not usable: ...") name a quantity to say what to repair; they are FI's diagnosis, not a value the
        # script printed, so they stay. Only what the script itself printed is filtered (and any bare value, below).
        kept: list[str] = []
        dropped, keys = 0, []
        for line in lines:
            if line.lstrip().startswith(("FI ", "FI's ", "[FI")):
                kept.append(line)
                continue
            ok, gone, keys = self._lines_without_left_out([line])
            kept.extend(ok)
            dropped += gone
        if not dropped:
            return self._withhold_left_out_values(text)
        out = "\n".join([*kept, self._taken_out_note(dropped, keys, "error output")])
        return self._withhold_left_out_values(out)

    def _keep_raw_stdout(self, stdout: str) -> None:
        """The run's output exactly as it was, for the records (no model reads it)."""
        try:
            path = self.fi_dir / "last_run_stdout.txt"  # type: ignore[attr-defined]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((stdout or "")[-200000:], encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not keep the run's raw output (%r)", exc)  # type: ignore[attr-defined]

    def _note_typed_values(self, raw_result_json: Any) -> None:
        """The values of the quantities left out for being typed into the code, kept in this pass's record so the writer's
        text can be checked for quoting one (core/record_claims.py). Nothing is added when none was left out."""
        from . import result_checks as _checks

        keys = self._typed_result_keys if getattr(self, "_typed_result_keys", None) is not None else set()
        if not keys or not raw_result_json:
            return
        path = self.quest_root / "needs" / TYPED_RECORD  # type: ignore[attr-defined]
        record = _read_json(path)
        if not record:
            return
        record["removed_values"] = _checks._values_left_out(raw_result_json, set(keys))
        try:
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", TYPED_RECORD, exc)  # type: ignore[attr-defined]

    # ---- a result that did not change is recorded as observed ----------------------------------------------------------

    def _record_no_difference(self, note: str | None) -> None:
        """Every setting gave the same numbers and the code does read the settings (core/flat_output.py): the observation is
        kept in ``needs/NO_DIFFERENCE_OBSERVED.json`` (this run's alone) for the analysis and the writer. A run without one
        removes an earlier run's record."""
        path = self.quest_root / "needs" / NO_DIFFERENCE_RECORD  # type: ignore[attr-defined]
        if not note:
            path.unlink(missing_ok=True)
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"note": note}, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", NO_DIFFERENCE_RECORD, exc)  # type: ignore[attr-defined]

    def _no_difference_note(self) -> str:
        """For the analysis and the writer: a study whose numbers did not change with its settings, said as an observation
        (never as proof of no effect, and never as a broken simulation). Empty when there is none."""
        record = _read_json(self.quest_root / "needs" / NO_DIFFERENCE_RECORD)  # type: ignore[attr-defined]
        note = record.get("note") if isinstance(record, dict) else None
        if not isinstance(note, str) or not note.strip():
            return ""
        return (f"Observed: {note} Report it as a result that did not change in the settings tried; do not call the "
                "simulation broken, and do not claim that the setting can have no effect in general.")

    # ---- the note for the writer ---------------------------------------------------------------------------------------

    def _withhold_left_out_values(self, text: str) -> str:
        """``text`` (a note a model reads) with any number written precisely enough to be recognised that equals a value of a
        quantity left out of the results replaced by plain words: a note that says why a quantity is out never repeats it."""
        from .number_provenance import _carries_enough_precision

        values: list[float] = []
        for name in (TYPED_RECORD, "RESULT_CHECK.json"):
            removed = _read_json(self.quest_root / "needs" / name).get("removed_values")  # type: ignore[attr-defined]
            if isinstance(removed, dict):
                values += [float(v) for vs in removed.values() if isinstance(vs, list) for v in vs
                           if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not values:
            return text

        def swap(m: re.Match[str]) -> str:
            token = m.group(0)
            if not _carries_enough_precision(token):
                return token
            number = float(token)
            return "(a value that is not given)" if any(abs(number - v) <= 1e-9 * max(1.0, abs(v)) for v in values) else token

        return re.sub(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w])", swap, text)

    def _run_checks_note(self) -> str:
        """For the writer: quantities left out because they were typed into the code, and a run FI made smaller to fit the
        time allowed, so the limitations say so."""
        parts: list[str] = []
        record = _read_json(self.quest_root / "needs" / TYPED_RECORD)  # type: ignore[attr-defined]
        note = record.get("note")
        if isinstance(note, str) and note.strip():
            names = ", ".join(f"`{k}`" for k in record.get("removed_quantities") or [])
            parts.append(f"{self._withhold_left_out_values(note)} Do not report, describe or quote any value for {names or 'them'}: say plainly, in the "
                         "limitations, that this quantity was not computed by the simulation and is left out.")
        if checked := self._result_checks_note():  # type: ignore[attr-defined]  # core/result_checks.py
            parts.append(self._withhold_left_out_values(checked))
        if static := self._static_checks_note():  # type: ignore[attr-defined]  # core/static_checks.py
            parts.append(static)
        if observed := self._no_difference_note():
            parts.append(observed)
        sized = _read_json(self.quest_root / _estimate.RECORD)  # type: ignore[attr-defined]
        reduced = sized.get("reduced")
        if isinstance(reduced, dict) and reduced.get("note"):
            parts.append(f"{reduced['note']} Say so plainly in the limitations, with the numbers.")
        return "\n\n".join(parts)

    # ---- the run fits the time it is allowed ---------------------------------------------------------------------------

    def _grid_and_runs(self, state: Any) -> tuple[dict[str, list[Any]], int, dict[str, Any]]:
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        grid = {str(k): list(v) for k, v in grid.items() if isinstance(v, list) and v}
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
        return grid, max(1, int(protocol.get("runs_per_setting") or 1)), thresholds

    def _sizing_key(self, runner: Any, grid: dict[str, list[Any]], runs: int, limit_s: float) -> str:
        return _trial_runner._run_key(runner.simulate, {"grid": grid}, runs, 0, runner.deterministic) + f"|{int(limit_s)}"

    async def _size_the_run(self, state: Any, runner: Any, python: Any, env: Any) -> dict[str, Any] | None:
        """Before the protocol is frozen and the study starts: time a few real trials, add them up, and compare the total with
        ``execution.timeout_s``. Fits (or cannot be timed): go on, saying so in one line. Does not fit: ask the plan's model to
        make the run smaller (at most :data:`core.run_estimate.ASKS` times, each followed by timing the new plan), and when
        it still would not fit return the quest's stop (no paper). ``None`` to go on.

        Only the trial contract is timed, and only before the first freeze: a frozen protocol cannot be changed, a search for
        the best design or a cluster job has its own limits, and a resume of a run that already fits is not timed again."""
        self._sized_protocol: dict[str, Any] | None = None
        self._pilot_spent = 0.0  # seconds the pilot's extra trials took in this sizing: they count against the same budget
        if (not isinstance(runner, _trial_runner.TrialsRunner) or self.config.execution.background_jobs  # type: ignore[attr-defined]
                or _frozen.load(self.quest_root) is not None or int(state.get("iteration", 0) or 0) > 0):  # type: ignore[attr-defined]
            return None
        limit_s = float(self.config.execution.timeout_s)  # type: ignore[attr-defined]
        budget_s = max(limit_s * float(self.config.engine.pilot_timeout_frac), 30.0)  # type: ignore[attr-defined]
        cached: dict[str, _estimate.Probe] = {}
        prior = _estimate.read_record(self.quest_root)  # type: ignore[attr-defined]
        # A sizing that did not finish (a request was made, or the run did not fit) goes on counting; one that fitted is over.
        unfinished = prior.get("fits") is False
        asks = int(prior.get("asks") or 0) if unfinished else 0
        reduced = prior.get("reduced") if unfinished or prior.get("fits") else None
        for _round in range(_estimate.ASKS + 1):
            grid, runs, thresholds = self._grid_and_runs(state)
            key = self._sizing_key(runner, grid, runs, limit_s)
            record = _estimate.read_record(self.quest_root)  # type: ignore[attr-defined]
            if record.get("key") == key and record.get("fits"):
                self._log.info("[execute] %s (timed before; nothing in the experiment has changed)",  # type: ignore[attr-defined]
                               record.get("says") or "the experiment fits")
                return None
            if record.get("key") == key and record.get("stopped"):
                return self._stop_too_long(str(record["stopped"]))
            probes, why = await _estimate.measure(
                self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(), grid,  # type: ignore[attr-defined]
                runs=runs, deterministic=runner.deterministic, limit_s=limit_s,
                budget_s=max(0.0, budget_s - self._pilot_spent), env=env,
                thresholds=thresholds, cached=cached)
            if probes is None:
                # A setting that fails when timed is a setting the simulation fails at: the same repair as a failed run, naming
                # it (the error only; the repair is never given an expected value).
                self._log.warning("[execute] the simulation failed while FI timed the experiment %s", why)  # type: ignore[attr-defined]
                return self._sizing_failed(why)
            for p in probes:
                cached[_estimate.probe_key(p.cell, runs)] = p
            if (broken := await self._pilot_gate(state, runner, python, env, grid, probes, thresholds, limit_s, budget_s)) is not None:
                return broken
            est = _estimate.estimate(grid, runs, runner.deterministic, probes)
            if est is None:
                return None
            line = _estimate.says(est, limit_s)
            base = {"key": key, "estimate_s": est.seconds * _estimate.SAFETY, "limit_s": limit_s, "says": line,
                    "reduced": reduced, "at": time.time(), "timed": est.timed, "cells": est.cells, "trials": est.trials}
            if not _estimate.known_too_long(est, limit_s):
                self._log.info("[execute] %s", line)  # type: ignore[attr-defined]
                print(f"[FI] {line}")
                _estimate.write_record(self.quest_root, {**base, "fits": True, "asks": 0})  # type: ignore[attr-defined]
                return None
            self._log.warning("[execute] %s, so it would not finish", line)  # type: ignore[attr-defined]
            design, sha = self._design_from_plan()  # type: ignore[attr-defined]
            can_ask = design is not None and bool(sha) and self._client is not None  # type: ignore[attr-defined]
            if asks >= _estimate.ASKS or not can_ask:
                text = self._too_long_text(est, limit_s, asks, can_ask)
                _estimate.write_record(self.quest_root, {**base, "fits": False, "asks": asks, "stopped": text})  # type: ignore[attr-defined]
                return self._stop_too_long(text)
            asks += 1
            # Counted BEFORE the request, so a quest killed during it never asks more often than it may.
            _estimate.write_record(self.quest_root, {**base, "fits": False, "asks": asks})  # type: ignore[attr-defined]
            changed, reduced_now, made = await self._ask_to_make_the_run_smaller(
                state, est, limit_s, runs, runner.deterministic, _estimate.ASKS - (asks - 1))
            asks = asks - 1 + max(1, made)
            if changed:
                reduced = reduced_now
                self._sized_protocol = {**(self._sized_protocol or {}), **changed}
                _estimate.write_record(  # type: ignore[attr-defined]
                    self.quest_root, {**base, "fits": False, "asks": asks, "reduced": reduced})
                await self._settle_plan_sources(state, protocol=self._draft_protocol(state))  # type: ignore[attr-defined]
        return None

    async def _pilot_gate(self, state: Any, runner: Any, python: Any, env: Any, grid: dict[str, list[Any]],
                          probes: list[_estimate.Probe], thresholds: dict[str, Any], limit_s: float,
                          budget_s: float) -> dict[str, Any] | None:
        """Read what the timed trials returned (:func:`core.run_estimate.pilot`). Nothing wrong: one line in run.log and the
        record in ``.fi/pilot_check.json`` (which QUALITY.md shows). Something wrong: a failed run for the existing repair of the
        simulation, with the exact finding (at most ``engine.exec_reflect_max_iterations`` repairs, counted in the record
        BEFORE the run goes back); what the repairs leave ends in the honest stop. Nobody is asked to debug."""
        from . import degenerate_results as _degenerate
        from . import hidden_check as _hidden

        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        kinds = {str(m.get("id")): str(m.get("kind") or "") for m in protocol.get("metrics") or []
                 if isinstance(m, dict) and str(m.get("id") or "").strip()}
        total = sum(len(p.trials) for p in probes)
        if not total:
            return None  # nothing came back to read (a timed trial ran past its allowance)
        _grid, runs, _t = self._grid_and_runs(state)
        total_trials = _estimate.trial_count(grid, runs, runner.deterministic)
        allowance = max(60.0, 3.0 * limit_s / max(1, total_trials))
        record = _estimate.read_pilot_record(self.quest_root)  # type: ignore[attr-defined]
        repairs = int(record.get("repairs") or 0)
        # Once per version of the code: resizing the run does not change what the simulation does, so a reading already made
        # of this code (and these metrics) is reused, and its extra trials are not run again.
        code_key = _hidden.code_sha(self.quest_root) + "|" + ",".join(f"{k}:{v}" for k, v in sorted(kinds.items()))  # type: ignore[attr-defined]
        if code_key[0] != "|" and record.get("code_key") == code_key and record.get("outcome") == "passed":
            self._log.info("[execute] the pilot has read this version of the simulation already: nothing was wrong")  # type: ignore[attr-defined]
            return None
        if code_key[0] != "|" and record.get("code_key") == code_key and record.get("problem_items"):
            result = _estimate.Pilot(
                trials=int(record.get("trials") or 0), settings=int(record.get("settings") or 0),
                checked=[str(x) for x in record.get("metrics_read") or []],
                problems=[_estimate.PilotProblem(str(d.get("metric")), str(d.get("kind")), str(d.get("says")))
                          for d in record["problem_items"] if isinstance(d, dict)],
                unsure=[str(x) for x in record.get("not_ruled_out") or []])
        else:
            # The extra trials come out of the sizing's own budget (at most half of it in all, across the rounds).
            left = max(0.0, budget_s / 2.0 - self._pilot_spent)
            began = time.monotonic()
            result = await _estimate.pilot(
                self.executor, python, self.quest_root,  # type: ignore[attr-defined]
                runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
                grid, probes, kinds, deterministic=runner.deterministic, budget_s=left,
                allowance_s=allowance, env=env, thresholds=thresholds, exempt=_degenerate.expected_constants(protocol))
            self._pilot_spent += time.monotonic() - began
        base = {"trials": result.trials, "settings": result.settings, "metrics_read": result.checked,
                "not_ruled_out": result.unsure, "at": time.time(), "trial_count_read": total, "code_key": code_key,
                "problem_items": [{"metric": p.metric, "kind": p.kind, "says": p.says} for p in result.problems]}
        if result.unsure:
            self._log.info("[execute] the pilot could not try every combination of the small settings, so it says nothing of "  # type: ignore[attr-defined]
                           "%s", ", ".join(f"`{n}`" for n in result.unsure))
        if not result.problems:
            self._log.info("[execute] the pilot read %d trial(s) at %d setting(s): %d metric(s) of the protocol checked, nothing "  # type: ignore[attr-defined]
                           "wrong%s", result.trials, result.settings, len(result.checked),
                           f" (after {repairs} repair(s))" if repairs else "")
            _estimate.write_pilot_record(self.quest_root, {**base, "outcome": "passed", "repairs": repairs, "problems": []})  # type: ignore[attr-defined]
            return None
        says = [p.says for p in result.problems]
        iters = int(state.get("exec_reflect_iter") or 0)
        for line in says:
            self._log.warning("[execute] the pilot found: %s", line)  # type: ignore[attr-defined]
        if iters < self.config.engine.exec_reflect_max_iterations:  # type: ignore[attr-defined]
            # Counted BEFORE the run goes back: a quest killed here never repairs more often than it may.
            _estimate.write_pilot_record(  # type: ignore[attr-defined]
                self.quest_root, {**base, "outcome": "repairing", "repairs": repairs + 1, "problems": says})
            print("[FI] a small trial run before the study shows the simulation does not measure what the protocol names; "
                  "sending it back to be repaired")
            return self._code_review_failed_run(  # type: ignore[attr-defined]
                _estimate.directive(result), script=_split_run.SIMULATE_NAME)
        _estimate.write_pilot_record(  # type: ignore[attr-defined]
            self.quest_root, {**base, "outcome": "stopped", "repairs": repairs, "problems": says})
        return self._stop_before_run(  # type: ignore[attr-defined]
            "a small trial run before the study (" + f"{result.trials} trial(s) at {result.settings} setting(s)) showed: "
            + "; ".join(says),
            "cannot_repair",
            tried=[f"ran {result.trials} trial(s) of the simulation at {result.settings} setting(s) before the full study "
                   "and read what they returned",
                   *[_one(s) for s in says[:3]],
                   f"sent the simulation back to be repaired {iters} time(s) with this named; it still showed it"])

    def _design_after_sizing(self, state: Any) -> dict[str, Any] | None:
        """The state's design with the protocol parts the plan's model made smaller, or ``None`` when none were."""
        sized = getattr(self, "_sized_protocol", None)
        if not sized:
            return None
        design = dict(state.get("design") or {})
        design["protocol"] = {**(design.get("protocol") if isinstance(design.get("protocol"), dict) else {}), **sized}
        return design

    async def _ask_to_make_the_run_smaller(self, state: Any, est: _estimate.Estimate, limit_s: float, runs: int,
                                           deterministic: bool, remaining: int) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
        """One request to the plan's model (the bounded one every part of the plan uses): ``({"grid": ..., "runs_per_setting":
        ...}, the note, requests made)`` as the plan now has them when it was made smaller, else ``({}, None, n)``. Only those two parts of the
        answer are kept; a different set of settings, or a run that is not smaller, is not used."""
        design, sha = self._design_from_plan()  # type: ignore[attr-defined]
        if design is None or not sha or self._client is None:  # type: ignore[attr-defined]
            return {}, None, 0
        before_grid, before_runs, _t = self._grid_and_runs(state)
        before_protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        words = "at least " if est.at_least else "about "
        asked = {"n": 0}

        def lacks(now: Any) -> list[str]:
            # The run is still too big until it has become smaller; the number of requests is bounded by the helper.
            protocol = now.get("protocol") if isinstance(now, dict) and isinstance(now.get("protocol"), dict) else {}
            grid = {str(k): list(v) for k, v in (protocol.get("grid") or {}).items() if isinstance(v, list) and v}
            now_runs = max(1, int(protocol.get("runs_per_setting") or before_runs))
            smaller = (_estimate.trial_count(grid, now_runs, deterministic)
                       < _estimate.trial_count(before_grid, before_runs, deterministic))
            if smaller:
                return []
            return [f"a smaller run (the experiment would take {words}"
                    f"{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}, more than the "
                    f"{_estimate.plain_duration(limit_s)} allowed)"]

        def request(_now: Any, _text: str) -> str:
            asked["n"] += 1
            return _estimate.request(est, limit_s, runs, deterministic,
                                     _estimate.rules(before_protocol, before_runs, deterministic, est.spreads))

        def keep(before: str, revised: str) -> str:
            kept = self._keep_only_the_parts(before, revised, ["grid", "runs_per_setting"])  # type: ignore[attr-defined]
            if kept is None:
                raise ValueError("the revised plan has no smaller run in a form that can be read; plan.md is unchanged")
            block = _plan.raw_design_block(kept)
            protocol = block.get("protocol") if isinstance(block, dict) and isinstance(block.get("protocol"), dict) else {}
            grid = {str(k): list(v) for k, v in (protocol.get("grid") or {}).items() if isinstance(v, list) and v}
            new_runs = max(1, int(protocol.get("runs_per_setting") or before_runs))
            problems = _estimate.smaller_run_problems(
                before_protocol, before_grid, before_runs, grid, new_runs, deterministic, est.spreads)
            if problems:
                raise ValueError("the revised plan is not used: " + "; ".join(problems) + "; plan.md is unchanged")
            return kept

        await self._ask_plan_to_complete(  # type: ignore[attr-defined]
            design, sha, lacks=lacks, request=request, keep=keep, file="run_size_asked.json", what="run size",
            short="a smaller run", note="a smaller run, so the experiment fits the time allowed, written by the plan's model",
            ask_limit=remaining + self._search_record(sha, "run_size_asked.json")[0])  # type: ignore[attr-defined]
        grid, new_runs, _t = self._grid_and_runs(state)
        if (grid, new_runs) == (before_grid, before_runs):
            return {}, None, asked["n"]
        note = (f"FI made the experiment smaller so that it fits the time allowed ({_estimate.plain_duration(limit_s)}): it "
                f"would have taken {words}{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}. The plan's model changed "
                f"it from {_estimate.cell_count(before_grid)} setting(s) with {before_runs} run(s) each to "
                f"{_estimate.cell_count(grid)} setting(s) with {new_runs} run(s) each; the checks, thresholds and "
                "tolerances are unchanged.")
        self._log.warning("[execute] %s", note)  # type: ignore[attr-defined]
        print(f"[FI] {note}")
        return ({"grid": grid, "runs_per_setting": new_runs},
                {"note": note, "from": {"grid": before_grid, "runs_per_setting": before_runs},
                 "to": {"grid": grid, "runs_per_setting": new_runs}}, asked["n"])

    @staticmethod
    def _too_long_text(est: _estimate.Estimate, limit_s: float, asks: int, can_ask: bool) -> str:
        said = (f"the experiment would take {'at least ' if est.at_least else 'about '}"
                f"{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}, more than the "
                f"{_estimate.plain_duration(limit_s)} allowed")
        if asks:
            return f"{said}; FI asked the plan {asks} time{'s' if asks != 1 else ''} to make it smaller"
        return said if can_ask else f"{said}; FI had no way to ask the plan to make it smaller"

    def _sizing_failed(self, why: str) -> dict[str, Any]:
        """A failed run for the existing repair of the simulation: it failed while the experiment was timed, at the setting
        named in ``why``."""
        text = f"FI ran a trial of the simulation to time the experiment before the study, and it failed {why}"
        return {
            "exec_result": {
                "returncode": 1, "duration_s": 0.0, "timed_out": False, "stdout_tail": "", "stderr_tail": text[-2000:],
                "packages_note": getattr(self, "_packages_note", ""), "failed_script": _split_run.SIMULATE_NAME,
                "flat_output": "", "numeric_warnings": [],
            },
            "figures": [], "figure_records": {}, "result_json": {}, "exec_patch_pending": False,
            "result_json_replicates": [], "result_json_deterministic": False, "result_json_trials": False,
            "result_json_replicate_seed_ignored": False, "result_json_no_random_source": False,
        }

    def _stop_too_long(self, text: str) -> dict[str, Any]:
        """The quest's plain stop: nothing was run for the study, it would not fit, and no paper is written."""
        return self._stop_before_run(text, "too_long")

    def _stop_before_run(self, text: str, kind: str, tried: list[str] | None = None,
                         advice: str | None = None) -> dict[str, Any]:
        """The quest's plain stop before the study (``too_long``: it would not fit the time allowed; ``not_run``: a text of
        the simulation could not be checked), with no paper."""
        self._log.warning("[execute] %s; stopping, the study was not run", text)  # type: ignore[attr-defined]
        return {
            "exec_result": {
                "returncode": 1, "duration_s": 0.0, "timed_out": False, "stdout_tail": "", "stderr_tail": text[-2000:],
                "packages_note": getattr(self, "_packages_note", ""), "failed_script": "", "flat_output": "",
                "numeric_warnings": [], kind: text, **({f"{kind}_tried": tried} if tried else {}),
            **({f"{kind}_advice": advice} if advice else {}),
            },
            "exec_give_up_reason": text,
            "figures": [], "figure_records": {}, "result_json": {}, "exec_patch_pending": False,
            "result_json_replicates": [], "result_json_deterministic": False, "result_json_trials": False,
            "result_json_replicate_seed_ignored": False, "result_json_no_random_source": False,
        }
