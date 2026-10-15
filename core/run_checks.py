"""Two checks made on the simulation before and while it runs, as methods the engine mixes in.

* **A reported result is computed** (``core/typed_results.py``): the simulation is read, never run, and a returned value that
  is a number typed into the code is sent back to be computed (or removed), at most twice; one that is still typed in
  afterwards is left out of the results the paper may use, and the paper says so.
* **The run fits the time it is allowed** (``core/run_estimate.py``): a few real trials are timed before the freeze and
  added up; a study that would not finish gets fewer runs per setting from FI itself (no model is asked; every setting
  is kept, and the runs never go below the fewest the plan allows); one that still would not fit stops the quest
  plainly with no paper.
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
from . import in_run as _in_run
from . import oracle_forms as _forms
from . import plan as _plan
from . import probes as _probes
from . import probes_run as _probes_run
from . import run_estimate as _estimate
from . import split_run as _split_run
from . import trial_runner as _trial_runner
from . import typed_results as _typed

#: How many times, in all, a simulation that returns typed-in numbers is sent back.
TYPED_RESULT_REPAIRS = 2
TYPED_RECORD = "TYPED_RESULTS_CHECK.json"
NO_DIFFERENCE_RECORD = "NO_DIFFERENCE_OBSERVED.json"
_COUNTER = "typed_results.json"
#: How many crashes inside the code, shown by a finding's test during the code review, go to the crash repair in a quest.
REVIEW_CRASHES = 2

#: FI's own behaviour checks (core/probes.py) are RECORD ONLY for now: a core check that does not hold is written down (the
#: record, needs/PROBES.json, QUALITY.md, a sentence in the paper's limitations) and nothing else happens. Turning on
#: ``REPAIR_ENABLED`` sends the simulation back once per allowed repair with the finding; turning on ``STOP_ENABLED`` stops the
#: quest when the repairs are used up. Both stay off until the false-alarm rate has been measured on good quests.
STOP_ENABLED = False
REPAIR_ENABLED = False
#: The share of the run's time limit the checks may spend running cells, and its ceiling and floor in seconds.
PROBE_BUDGET_FRAC = 0.10
PROBE_BUDGET_MAX_S = 900.0
PROBE_BUDGET_MIN_S = 30.0
#: Calls to the reader in a quest, at most (one per version of the plan; a redesign is a new version).
PROBE_MAX_ASKS = 3
PROBE_NODE = "oracle_review.probes"
_PROBE_PLAN_CHARS = 12000


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
        """Names the plan fixes (its fixed settings, the grid's settings, what each run compares, the thresholds): echoed back,
        not results."""
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        design = state.get("design") if isinstance(state.get("design"), dict) else {}
        names = set(_forms.fixed_settings_of_design({**design, "protocol": protocol}))
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        names |= {str(k) for k in grid} | {str(k) for k in _in_run.of(protocol) or {}}
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
        if probed := self._probe_note():
            parts.append(probed)
        sized = _read_json(self.quest_root / _estimate.RECORD)  # type: ignore[attr-defined]
        reduced = sized.get("reduced")
        if isinstance(reduced, dict) and reduced.get("note"):
            parts.append(f"{reduced['note']} Say this in the paper with the numbers (it is among FI's facts about this run, "
                         "given below), and write no result for what was left out.")
        return "\n\n".join(parts)

    # ---- the run fits the time it is allowed ---------------------------------------------------------------------------

    def _grid_and_runs(self, state: Any) -> tuple[dict[str, list[Any]], int, dict[str, Any]]:
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        grid = {str(k): list(v) for k, v in grid.items() if isinstance(v, list) and v}
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
        return grid, max(1, int(protocol.get("runs_per_setting") or 1)), thresholds

    def _sizing_key(self, runner: Any, grid: dict[str, list[Any]], runs: int, limit_s: float,
                    inside: dict[str, list[Any]] | None = None) -> str:
        what = {"grid": grid, **({"in_each_run": inside} if inside else {})}
        return _trial_runner._run_key(runner.simulate, what, runs, 0, runner.deterministic) + f"|{int(limit_s)}"

    async def _size_the_run(self, state: Any, runner: Any, python: Any, env: Any) -> dict[str, Any] | None:
        """Before the protocol is frozen and the study starts: time a few real trials, add them up, and compare the total with
        ``execution.timeout_s``. Fits (or cannot be timed): go on, saying so in one line. Does not fit: FI lowers the runs per
        setting itself (no model is asked, every setting is kept, never below the fewest runs the plan states), times the
        shorter run again, and when it still would not fit returns the quest's stop (no paper). ``None`` to go on.

        Only the trial contract is timed, and only before the first freeze: a frozen protocol cannot be changed, a search for
        the best design or a cluster job has its own limits, and a resume of a run that already fits is not timed again."""
        self._sized_protocol: dict[str, Any] | None = None
        self._pilot_spent = 0.0  # seconds the pilot's extra trials took in this sizing: they count against the same budget
        # A run_trial that returns the mean of many trials (it calls run_cell) is a fault of the code, not of the plan's size:
        # it goes back to the code step, never to the plan to shrink the grid (which dropped settings the topic required).
        if isinstance(runner, _trial_runner.TrialsRunner):
            try:
                aggregate = _trial_runner.trial_aggregates(runner.simulate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                aggregate = ""
            if aggregate:
                self._log.warning("[execute] %s; sending simulate.py back to be repaired before the experiment is timed",  # type: ignore[attr-defined]
                                  aggregate)
                return self._sizing_failed(aggregate, text=f"FI read simulate.py before timing the experiment: {aggregate}. "
                                                           "Make run_trial compute one trial (one random draw of the setting) "
                                                           "on its own; keep any mean over trials in run_cell, which may "
                                                           "call run_trial, never the other way round.")
        from . import optimise as _optimise

        if isinstance(runner, _optimise.OptimisationRunner):
            return await self._size_the_scan(state, runner, python, env)
        if (not isinstance(runner, _trial_runner.TrialsRunner) or self.config.execution.background_jobs  # type: ignore[attr-defined]
                or _frozen.load(self.quest_root) is not None or int(state.get("iteration", 0) or 0) > 0):  # type: ignore[attr-defined]
            return None
        limit_s = float(self.config.execution.timeout_s)  # type: ignore[attr-defined]
        budget_s = max(limit_s * float(self.config.engine.pilot_timeout_frac), 30.0)  # type: ignore[attr-defined]
        cached: dict[str, _estimate.Probe] = {}
        prior = _estimate.read_record(self.quest_root)  # type: ignore[attr-defined]
        # The note of a cut made before a resume is kept, whether the sizing finished or not.
        reduced = prior.get("reduced") if prior.get("fits") is not None else None
        # FI lowers the runs once; the lowered run is timed again, and a resume after the cut does not cut a second time.
        cut_done = bool(reduced) and prior.get("fits") is False
        for _round in range(2):
            grid, runs, thresholds = self._grid_and_runs(state)
            inside = _in_run.of(self._protocol_block(state) or {})  # type: ignore[attr-defined]
            key = self._sizing_key(runner, grid, runs, limit_s, inside)
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
                thresholds=thresholds, cached=cached, in_each_run=inside)
            if probes is None:
                # A setting that fails when timed is a setting the simulation fails at: the same repair as a failed run, naming
                # it (the error only; the repair is never given an expected value).
                self._log.warning("[execute] the simulation failed while FI timed the experiment %s", why)  # type: ignore[attr-defined]
                return self._sizing_failed(why)
            for p in probes:
                cached[_estimate.probe_key(p.cell, runs)] = p
            if (broken := await self._pilot_gate(state, runner, python, env, grid, probes, thresholds, limit_s, budget_s)) is not None:
                return broken
            if (broken := await self._probe_gate(state, runner, python, env, grid, probes, thresholds, limit_s)) is not None:
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
                _estimate.write_record(self.quest_root, {**base, "fits": True})  # type: ignore[attr-defined]
                return None
            self._log.warning("[execute] %s, so it would not finish", line)  # type: ignore[attr-defined]
            # Fewer runs per setting is the only cut FI makes: every setting the study compares is kept, and the runs never go
            # below the fewest the plan states. The shorter run is timed again before it is used. No model is asked.
            fewer = None if cut_done else _estimate.runs_that_fit(
                est, limit_s, runs, runner.deterministic, self._protocol_block(state) or {}, est.spreads)  # type: ignore[attr-defined]
            note = self._lower_the_runs(state, runs, fewer, est, limit_s, grid) if fewer is not None else None
            if note is None:
                text = self._too_long_text(est, limit_s)
                _estimate.write_record(self.quest_root, {**base, "fits": False, "stopped": text})  # type: ignore[attr-defined]
                return self._stop_too_long(text)
            cut_done = True
            reduced = note
            self._sized_protocol = {**(self._sized_protocol or {}), "runs_per_setting": fewer}
            _estimate.write_record(self.quest_root, {**base, "fits": False, "reduced": reduced})  # type: ignore[attr-defined]
            await self._settle_plan_sources(state, protocol=self._draft_protocol(state))  # type: ignore[attr-defined]
        return None

    async def _size_the_scan(self, state: Any, runner: Any, python: Any, env: Any) -> dict[str, Any] | None:
        """The coarse scan of a search for the best design runs every setting of the plan's grid, so its size is counted
        (the product of the axes' lengths, nothing listed) and its time added up BEFORE anything lists or runs the settings.
        More settings than FI lists: the quest's plain stop. Otherwise the time of one setting is the larger of the plan's own
        estimate and one real evaluation of the baseline (a plan's claim alone is not trusted); over ``execution.timeout_s``:
        the same stop, as for a measurement (FI does not run a smaller scan than the plan's). ``None`` to go on.

        Not skipped for a frozen protocol or a background job: the search runs here either way, sizing never changes the
        protocol, and the check is arithmetic and at most one timed evaluation, so a resume may repeat it."""
        from . import optimisation_plan as _optimisation_plan
        from . import optimise as _optimise
        from . import optimise_search as _optimise_search

        raw = _optimise.block_of(self._protocol_block(state))  # type: ignore[attr-defined]
        block, _why = _optimisation_plan.normalize(raw) if raw is not None else (None, "")
        if block is None:
            return None  # the run reports a block it cannot read
        scan = _optimisation_plan.evaluations(block)[1]
        if scan <= 1:
            return None
        try:
            _optimise_search.check_listable(_optimise_search.scan_grid(block))
        except _optimise_search.TooManySettings as exc:
            return self._stop_too_long(str(exc))
        except (KeyError, TypeError, ValueError):
            pass  # a block the scan cannot be built from is reported by the run
        deterministic = "run_trial" not in _trial_runner.entries(runner.simulate)
        runs = 1 if deterministic else max(1, int(block.get("runs_per_evaluation") or 1))
        limit_s = float(self.config.execution.timeout_s)  # type: ignore[attr-defined]
        claimed = (block.get("evaluation_budget") or {}).get("seconds_per_evaluation")
        per_setting = float(claimed) * runs if isinstance(claimed, (int, float)) and not isinstance(claimed, bool)             and claimed > 0 else None
        at_least = False
        baseline = _optimise.complete_case(self._protocol_block(state), {})  # type: ignore[attr-defined]
        # The same allowance a measurement's timed trial gets (core/run_estimate.py::measure), for this many trials.
        allowance = max(60.0, 3.0 * limit_s / max(1, scan * runs))
        probe, why = await _estimate._time_one(
            self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
            baseline, runs=runs, deterministic=deterministic, timeout_s=int(max(5, min(allowance, limit_s))), env=env,
            thresholds=(self._protocol_block(state) or {}).get("thresholds"))  # type: ignore[attr-defined]
        if probe is None:
            # A baseline that cannot be timed fails in the real run, where it is repaired; the plan's claim, when it gave one,
            # still counts, and the count above has already kept the listing within what memory holds.
            self._log.warning("[execute] FI could not time one evaluation of the search's baseline (%s)", why)  # type: ignore[attr-defined]
        elif per_setting is None or probe.seconds > per_setting:
            per_setting, at_least = probe.seconds, probe.at_least
        if per_setting is None:
            return None
        est = _estimate.Estimate(seconds=per_setting * scan, at_least=at_least, complete=True, cells=scan, trials=scan * runs)
        if not _estimate.known_too_long(est, limit_s):
            return None
        text = self._too_long_text(est, limit_s, cut=False)
        self._log.warning("[execute] the search's coarse scan: %s", text)  # type: ignore[attr-defined]
        return self._stop_too_long(text)

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
                allowance_s=allowance, env=env, thresholds=thresholds, exempt=_degenerate.expected_constants(protocol),
                in_each_run=_in_run.of(protocol))
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

    # ---- FI's own behaviour checks (core/probes.py): recorded, not yet acting ---------------------------------------------

    def _probe_text_of_plan(self) -> str:
        """The prose of ``plan.md`` without the design block (the reader gets the settings and metric names separately)."""
        try:
            text = _plan.plan_path(self.quest_root).read_text(encoding="utf-8")  # type: ignore[attr-defined]
        except OSError:
            return ""
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        return text.strip()[:_PROBE_PLAN_CHARS]

    def _probe_save(self, **fields: Any) -> None:
        """Merge ``fields`` into ``.fi/probes.json`` (what was asked is written BEFORE the call, so a resume never asks again)."""
        record = _probes_run.load(self.quest_root) or {}  # type: ignore[attr-defined]
        record.update(fields)
        path = self.quest_root / _probes_run.RECORD  # type: ignore[attr-defined]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        except OSError as e:
            self._log.warning("[execute] couldn't record FI's own behaviour check: %r", e)  # type: ignore[attr-defined]

    async def _probe_relations(self, plan_text: str, grid: dict[str, list[Any]], shown: list[str],
                               accepted: list[str]) -> tuple[list[_probes.Relation], list[dict[str, str]]]:
        """ONE call (node ``oracle_review.probes``): the relations the plan states, read strictly. Metric names are the ones the
        simulation returns (``accepted``); a reply naming anything else is dropped."""
        settings = "\n".join(f"- `{k}`: " + ", ".join(str(v) for v in vals[:12]) + (", ..." if len(vals) > 12 else "")
                             for k, vals in grid.items())
        prompt = self._prompts["probe_relations"].substitute(  # type: ignore[attr-defined]
            plan_text=plan_text, grid_settings=settings, metrics_list="\n".join(shown))
        reply = await self._chat(prompt, node=PROBE_NODE, temperature=0.0)  # type: ignore[attr-defined]
        return _probes.parse(reply, {"grid": grid, "metrics": [{"id": n} for n in accepted]})

    async def _probe_gate(self, state: Any, runner: Any, python: Any, env: Any, grid: dict[str, list[Any]],
                          probes: list[_estimate.Probe], thresholds: dict[str, Any], limit_s: float) -> dict[str, Any] | None:
        """After the pilot passed: FI tests the relations the plan states between settings and results (does the result move
        with the setting the plan says it does?) by running a few cells of its own, within about a tenth of the run's time
        limit, and writes down what it found. RECORD ONLY (:data:`STOP_ENABLED`, :data:`REPAIR_ENABLED`): nothing is sent back
        and nothing stops. Never raises; ``None`` to go on."""
        try:
            return await self._probe_gate_inner(state, runner, python, env, grid, probes, thresholds, limit_s)
        except Exception as e:  # noqa: BLE001 -- a check that cannot be made never stops a quest
            self._log.warning("[execute] FI's own behaviour check could not be made (%s); nothing is changed",  # type: ignore[attr-defined]
                              str(e)[:200] or type(e).__name__)
            return None

    async def _probe_gate_inner(self, state: Any, runner: Any, python: Any, env: Any, grid: dict[str, list[Any]],
                                probes: list[_estimate.Probe], thresholds: dict[str, Any], limit_s: float) -> dict[str, Any] | None:
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        _grid, protocol_ids = _probes._protocol_parts({**protocol, "grid": grid})
        returned = sorted({str(k) for p in probes for t in p.trials for k in t})
        sha = _probes_run.code_sha(self.quest_root)  # type: ignore[attr-defined]
        if not grid or not protocol_ids or not returned or not sha:
            return None   # nothing to vary, nothing to read, or nothing came back: no check, no noise
        alias, shown = _probes_run.metric_names(sorted(protocol_ids), returned)
        accepted = sorted(set(returned) | set(alias))
        plan_text = self._probe_text_of_plan()
        key = hashlib.sha256(json.dumps([plan_text, grid, shown], sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
        kept = _probes_run.load(self.quest_root) or {}  # type: ignore[attr-defined]
        asked = dict(kept.get("asked")) if isinstance(kept.get("asked"), dict) else {}
        had = sha in (kept.get("results") or {})
        relations: list[_probes.Relation] = []
        dropped: list[dict[str, str]] = list(kept.get("dropped") or [])
        if asked.get("key") == key:
            relations = [r for r in (_probes.from_dict(d) for d in kept.get("relations") or []) if r is not None]
        elif plan_text and self._client is not None and int(asked.get("calls") or 0) < PROBE_MAX_ASKS:  # type: ignore[attr-defined]
            asked = {"key": key, "calls": int(asked.get("calls") or 0) + 1}
            self._probe_save(asked=asked, relations=[], dropped=[])   # counted BEFORE the call: a resume never asks again
            try:
                relations, dropped = await self._probe_relations(plan_text, grid, shown, accepted)
            except Exception as e:  # noqa: BLE001 -- a reading that could not be had never stops a quest
                dropped = [{"id": "reader", "reason": f"the call failed ({str(e)[:120] or type(e).__name__})"}]
                self._log.warning("[execute] FI could not read the relations the plan states: %s", dropped[0]["reason"])  # type: ignore[attr-defined]
            self._probe_save(relations=[r.to_dict() for r in relations], dropped=dropped)
            if not had:
                self._log.info("[execute] FI read %d relation(s) between settings and results from the plan to check "  # type: ignore[attr-defined]
                               "(%d not usable)", len(relations), len(dropped))
        budget = min(PROBE_BUDGET_MAX_S, max(PROBE_BUDGET_MIN_S, float(limit_s) * PROBE_BUDGET_FRAC))
        mapped = {alias.get(m, m) for m in protocol_ids}
        out = await _probes_run.run(
            self.executor, python, self.quest_root, {**protocol, "grid": grid}, relations,  # type: ignore[attr-defined]
            reported=mapped, budget_s=budget, timeout_s=int(min(_probes_run.DEFAULT_TIMEOUT_S, max(PROBE_BUDGET_MIN_S, budget))),
            env=env, metric_alias=alias, asked=asked, dropped=dropped)
        results = {k: v for k, v in ((out.get("results") or {}).get(sha) or {}).items() if k != "saturation"}
        count = {w: sum(1 for v in results.values() if v.get("verdict") == w) for w in ("held", "broken", "unconfirmed")}
        broken = {k: v for k, v in results.items() if v.get("verdict") == "broken" and v.get("strength", "core") == "core"}
        if not had:
            self._log.info("[execute] FI's own behaviour checks (recorded, not yet acting): %d held, %d could not be checked, %d did "  # type: ignore[attr-defined]
                           "not hold", count["held"], count["unconfirmed"], count["broken"])
            for rid, v in broken.items():
                self._log.warning("[execute] FI's own check %s did not hold: %s",  # type: ignore[attr-defined]
                                  _probes.describe(next((r for r in relations if r.id == rid), None), rid), v.get("says"))
        repair_on = bool(getattr(self, "probe_repair_enabled", REPAIR_ENABLED))   # an instance can differ: no process-global switch
        stop_on = bool(getattr(self, "probe_stop_enabled", STOP_ENABLED))
        if not broken or not (repair_on or stop_on):
            return None
        says = [str(v.get("says")) for v in broken.values()]
        iters = int(state.get("exec_reflect_iter") or 0)
        if repair_on and iters < self.config.engine.exec_reflect_max_iterations:  # type: ignore[attr-defined]
            self._probe_save(repairs=int(out.get("repairs") or 0) + 1)   # counted BEFORE the run goes back
            return self._code_review_failed_run(  # type: ignore[attr-defined]
                "FI ran the simulation on a few settings of its own and compared what came back with what the plan says "
                "must happen. It did not hold: " + " ".join(says)
                + " Repair simulate.py so that it does what the plan says; keep every setting and every metric as they are.",
                script=_split_run.SIMULATE_NAME)
        if stop_on:
            return self._stop_before_run(  # type: ignore[attr-defined]
                "FI's own check of the simulation did not hold: " + "; ".join(_one(x, 300) for x in says[:3]), "cannot_repair",
                tried=["ran the simulation on a few settings of its own and compared the results with what the plan says",
                       *[_one(x) for x in says[:3]]])
        return None

    def _probe_note(self) -> str:
        """For the writer: a core check of FI's own that did not hold, in plain words, to be said in the limitations."""
        record = _probes_run.load(self.quest_root)  # type: ignore[attr-defined]
        if not record:
            return ""
        sha = _probes_run.code_sha(self.quest_root)  # type: ignore[attr-defined]
        by_id = {str(r.get("id")): _probes.from_dict(r) for r in record.get("relations") or [] if isinstance(r, dict)}
        lines = []
        for rid, v in ((record.get("results") or {}).get(sha) or {}).items():
            if rid == "saturation" or v.get("verdict") != "broken" or v.get("strength", "core") != "core":
                continue
            lines.append(f"FI's own check {_probes.describe(by_id.get(rid), rid)} did not hold: {_one(str(v.get('says') or ''), 300)}")
        if not lines:
            return ""
        return "\n".join(lines) + "\nSay so plainly in the limitations, with the numbers: this was recorded, and the study was run as written."

    def _design_after_sizing(self, state: Any) -> dict[str, Any] | None:
        """The state's design with the runs per setting FI lowered, or ``None`` when none were."""
        sized = getattr(self, "_sized_protocol", None)
        if not sized:
            return None
        design = dict(state.get("design") or {})
        design["protocol"] = {**(design.get("protocol") if isinstance(design.get("protocol"), dict) else {}), **sized}
        return design

    def _lower_the_runs(self, state: Any, runs: int, fewer: int, est: _estimate.Estimate, limit_s: float,
                        grid: dict[str, list[Any]]) -> dict[str, Any] | None:
        """FI's own smaller run: the same settings with ``fewer`` runs each, written into plan.md as a version of the engine
        (no request to the plan's model). The record of it, or ``None`` when plan.md could not be changed."""
        path = _plan.plan_path(self.quest_root)  # type: ignore[attr-defined]
        try:
            text = path.read_text(encoding="utf-8")
            edited = _plan.edit_design_block(
                text, lambda d: {**d, "protocol": {**(d.get("protocol") or {}), "runs_per_setting": int(fewer)}})
            if edited is None or _plan.parse(edited).design is None:
                return None
            edited = _plan.refresh_model_section(edited)
            _plan.note_edit(self.quest_root, text)  # type: ignore[attr-defined]
            path.write_text(edited, encoding="utf-8")
            _plan.record_version(self.quest_root, edited, by="engine",  # type: ignore[attr-defined]
                                 note="fewer runs per setting, so the experiment fits the time allowed, set by FI")
        except OSError as e:
            self._log.warning("[execute] FI could not lower the runs per setting in plan.md: %r", e)  # type: ignore[attr-defined]
            return None
        words = "at least " if est.at_least else "about "
        note = (f"FI made the experiment smaller so that it fits the time allowed ({_estimate.plain_duration(limit_s)}): it "
                f"would have taken {words}{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}. FI lowered the runs per "
                f"setting from {runs} to {fewer} and kept all {_estimate.cell_count(grid)} setting(s); the checks, thresholds "
                "and tolerances are unchanged.")
        self._log.warning("[execute] %s", note)  # type: ignore[attr-defined]
        print(f"[FI] {note}")
        return {"note": note, "from": {"grid": grid, "runs_per_setting": runs},
                "to": {"grid": grid, "runs_per_setting": fewer}}

    @staticmethod
    def _too_long_text(est: _estimate.Estimate, limit_s: float, cut: bool = True) -> str:
        return (f"the experiment's {est.cells:,} settings would take {'at least ' if est.at_least else 'about '}"
                f"{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}, more than the "
                f"{_estimate.plain_duration(limit_s)} allowed; FI does not run a smaller study than the plan's settings"
                + (", and they cannot fit even with the fewest runs per setting the plan allows" if cut else ""))

    def _sizing_failed(self, why: str, text: str | None = None) -> dict[str, Any]:
        """A failed run for the existing repair of the simulation: it failed while the experiment was timed, at the setting
        named in ``why``."""
        text = text or f"FI ran a trial of the simulation to time the experiment before the study, and it failed {why}"
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

    async def _small_run(self, state: Any, py: Any, exec_env: Any) -> Any:
        """FI's small run of the simulation on the code as it is (core/small_run.py), or ``None`` when the simulation does
        not follow the trial contract (``run_trial`` / ``run_cell`` in simulate.py). A run of the same code and settings is
        not made again: its record (``.fi/small_run.json``) is read back."""
        from . import small_run as _small

        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
        if not getattr(self, "_trial_mode", False) or not simulate.is_file():
            return None
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        from . import optimisation_plan as _optim
        from . import optimise as _optimise

        # A search for the best design has no grid: FI's search runner evaluates the baseline first, at its own settings.
        if _optim.has_block(state.get("design")) or _optimise.block_of(protocol) is not None:
            return None
        self._small_run_ctx = (state, py, exec_env)
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
        deterministic = "run_trial" not in set(getattr(self, "_trial_entries", set()) or ())
        key = _small.key_of(simulate, grid, deterministic, thresholds)
        record = _small.read_record(self.quest_root)  # type: ignore[attr-defined]
        if record.get("key") == key:
            self._small_run_note = str(record.get("for_reader") or "")
            return record
        paired = any(isinstance(m, dict) and m.get("paired") for m in protocol.get("metrics") or [])
        inside = protocol.get("in_each_run") if isinstance(protocol.get("in_each_run"), dict) else None
        got = await _small.run(
            self.executor, py, self.quest_root, simulate.relative_to(self.quest_root).as_posix(), grid,  # type: ignore[attr-defined]
            deterministic=deterministic, timeout_s=_small.budget(self.config.execution.timeout_s),  # type: ignore[attr-defined]
            env=dict(exec_env or {}), thresholds=thresholds, paired=paired, in_each_run=inside)
        _small.write_record(self.quest_root, key, got)  # type: ignore[attr-defined]
        self._small_run_note = got.for_reader()
        self._log.info("[execute] FI's small run before the code is read: %s", got.says())  # type: ignore[attr-defined]
        return _small.read_record(self.quest_root)  # type: ignore[attr-defined]

    async def _small_run_gate(self, state: Any, py: Any, exec_env: Any) -> dict[str, Any] | None:
        """Before the code is read: FI runs the simulation on a small case (core/small_run.py). A crash (it cannot be loaded,
        or most trials of a setting fail) goes to the repair of a failed run with the place and the traceback, while a
        repair is left; then the code is read with what the run gave. ``None`` to go on."""
        from . import small_run as _small

        record = await self._small_run(state, py, exec_env)
        if not record or not record.get("crashed"):
            return None
        if int(state.get("exec_reflect_iter") or 0) >= self.config.engine.exec_reflect_max_iterations:  # type: ignore[attr-defined]
            self._log.warning("[execute] FI's small run still fails (%s) and no repair is left; the study's own run "  # type: ignore[attr-defined]
                              "and its checks decide", record.get("says"))
            return None
        settings = [_small.Setting(str(s.get("key") or ""), int(s.get("planned") or 0), int(s.get("ok") or 0),
                                   int(s.get("failed") or 0), bool(s.get("timed_out")),
                                   [(str(r), int(n)) for r, n in s.get("reasons") or []])
                    for s in record.get("settings") or [] if isinstance(s, dict)]
        tb = ""
        try:
            for path in sorted((self.quest_root / _small.DIR).glob("s*/cell*.out.jsonl")):  # type: ignore[attr-defined]
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line) if line.strip().startswith("{") else {}
                    if isinstance(row, dict) and row.get("traceback"):
                        tb = str(row["traceback"])
                        break
                if tb:
                    break
        except (OSError, ValueError):
            tb = ""
        for s in settings:
            if s.fault and not s.traceback:
                s.traceback = tb
                break
        message = _small.SmallRun(settings, str(record.get("load_error") or "")).directive()
        self._log.warning("[execute] FI's small run of the simulation failed before the code is read (%s); sending it "  # type: ignore[attr-defined]
                          "to be repaired", record.get("says"))
        return self._code_review_failed_run(message, script=_split_run.SIMULATE_NAME)  # type: ignore[attr-defined]

    async def _small_run_again(self) -> str:
        """For a repair of a reader's finding: ``""`` when FI's small run still runs on the code as it is now, else why not.
        ``""`` when there is no small run to make."""
        ctx = getattr(self, "_small_run_ctx", None)
        if not ctx:
            return ""
        record = await self._small_run(*ctx)
        return str(record.get("says") or "it failed") if record and record.get("crashed") else ""

    async def _review_crash_run(self, state: Any) -> dict[str, Any] | None:
        """A test written for a reader's finding that crashed inside the code (not in the test) showed a crash, not the
        finding: it goes to the crash repair like a failed run, with the place and the call that crashed, while a repair is
        left and at most :data:`REVIEW_CRASHES` times a quest. The test is run again first: a crash a later repair removed
        is not sent. ``None`` when there is nothing to send."""
        from . import code_findings as _code_findings
        from . import code_review as _code_review
        from . import finding_repair as _finding_repair

        record = _code_review.load(self.quest_root)  # type: ignore[attr-defined]
        if not record or int(record.get("crashes_sent") or 0) >= REVIEW_CRASHES:
            return None
        if int(state.get("exec_reflect_iter") or 0) >= self.config.engine.exec_reflect_max_iterations:  # type: ignore[attr-defined]
            return None
        for issue in _code_findings.issues_of(record).values():
            crash = issue.get("crash") if isinstance(issue, dict) else None
            if not isinstance(crash, dict) or crash.get("sent"):
                continue
            crash["sent"] = True
            rel = str(crash.get("test") or "")
            _rc, text, _err = await self._run_in_env(  # type: ignore[attr-defined]
                ["-c", _finding_repair.harness_source("code", [rel])], timeout_s=180, env=self._finding_test_env())  # type: ignore[attr-defined]
            now = _finding_repair.parse_runs(text, [rel]).get(rel)
            if now is None or not now.crash_in_program:
                crash["gone"] = True
                _code_review.save(self.quest_root, record)  # type: ignore[attr-defined]
                continue
            record["crashes_sent"] = int(record.get("crashes_sent") or 0) + 1
            _code_review.save(self.quest_root, record)  # type: ignore[attr-defined]
            try:
                test_text = (self.quest_root / rel).read_text(encoding="utf-8")  # type: ignore[attr-defined]
            except OSError:
                test_text = ""
            at = now.at or str(crash.get("at") or "")
            self._log.warning("[code_review] a test of finding %s crashed inside the code at %s (%s); sending the code to "  # type: ignore[attr-defined]
                              "the crash repair", issue.get("id"), at, _one(now.detail))
            message = (f"FI called the code the way the test below does, and it crashed inside the code at {at}: "
                       f"{now.detail}\n\nThe call (a test written to check a reader's finding about the code):\n```python\n"
                       f"{test_text[:1500]}\n```\n\nRepair the code so that this call runs to the end and returns its values. "
                       "Keep everything else as it is.")
            script = ("experiment.py" if at.startswith("code/experiment.py") or not self._split_on(state)  # type: ignore[attr-defined]
                      else _split_run.SIMULATE_NAME)
            return self._code_review_failed_run(message, script=script)  # type: ignore[attr-defined]
        return None

    def _trial_checks_stop(self, state: Any) -> dict[str, Any] | None:
        """When no repair of a fault FI's checks of the trials found is coming (they are spent, or the repair gave up): failed
        trials in some settings only run the study once more with the repairs counted as spent, so those settings are left
        out (FI's trials are reused, the simulation being unchanged); any other fault stops the quest with its plain record
        (``cannot_repair``, no paper). ``None`` when the run was not sent back by those checks (core/trial_checks.py)."""
        result = state.get("exec_result") or {}
        fault = result.get("trial_checks") if isinstance(result, dict) else None
        if not isinstance(fault, dict) or not fault.get("say") or result.get("returncode", 0) == 0:
            return None
        if fault.get("partial"):
            self._log.warning(  # type: ignore[attr-defined]
                "[execute_reflect] no repair of the settings where most trials failed is coming: the study is read once "
                "more and those settings are left out")
            return {"exec_patch_pending": True, "exec_give_up_reason": "",
                    "exec_reflect_iter": max(int(state.get("exec_reflect_iter") or 0),
                                             int(self.config.engine.exec_reflect_max_iterations))}  # type: ignore[attr-defined]
        repairs = int(state.get("exec_reflect_iter") or 0)
        return self._stop_before_run(
            str(fault["say"]), "cannot_repair",
            tried=[*[str(t) for t in fault.get("tried") or []],
                   f"sent the simulation back to be repaired with these numbers ({repairs} repair"
                   f"{'s' if repairs != 1 else ''}); the fault stood"])

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
