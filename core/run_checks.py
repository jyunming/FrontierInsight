"""Two checks made on the simulation before and while it runs, as methods the engine mixes in.

* **A reported result is computed** (``core/typed_results.py``): the simulation is read, never run, and a returned value that
  is a number typed into the code is sent back to be computed (or removed), at most twice; one that is still typed in
  afterwards is left out of the results the paper may use, and the paper says so.
* **The run fits the time it is allowed** (``core/run_estimate.py``): a few real trials are timed before the freeze and
  added up against the topic's own time limit when it states one (``execution.timeout_s`` stays FI's hard stop) or else
  ``execution.timeout_s``; a study that would not finish gets fewer runs per setting from FI itself (no model is asked;
  every setting is kept; the runs go down to the fewest the plan allows, and below it only for the topic's own limit,
  said in the paper with the precision those runs give); one that would not fit FI's own limit even so stops the quest
  plainly with no paper, and one that cannot meet the topic's limit runs at the plan's fewest and says it was over.
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
from . import optimise_search as _optimise_search
from . import oracle_forms as _forms
from . import plan as _plan
from . import run_estimate as _estimate
from .result_checks import TYPED_RECORD
from . import split_run as _split_run
from . import trial_runner as _trial_runner
from . import typed_results as _typed

#: How many times, in all, a simulation that returns typed-in numbers is sent back.
TYPED_RESULT_REPAIRS = 2
#: In ``.fi/``: the settings FI took out of the grid because the simulation never reads them.
NO_DIFFERENCE_RECORD = "NO_DIFFERENCE_OBSERVED.json"
_COUNTER = "typed_results.json"
#: How many crashes inside the code, shown by a finding's test during the code review, go to the crash repair in a quest.
REVIEW_CRASHES = 2

# A number is written precisely enough to be recognised as a value (so that a note never repeats a left-out one): three
# significant digits, or two written to four decimals ("0.0068"). Below that a number is a round figure, a setting or a label.
_MIN_SIG_DIGITS = 3
_LOW_PRECISION_SIG_DIGITS = 2
_LOW_PRECISION_MIN_DECIMALS = 4


def _carries_enough_precision(token: str) -> bool:
    """Is this number written precisely enough to be recognised as a value?"""
    t = token.lstrip("-+").replace(",", "")
    decimals = len(t.split(".", 1)[1]) if "." in t else 0
    if "." in t:
        whole, frac = t.split(".", 1)
        whole = whole.lstrip("0")
        sig = len(whole) + len(frac) if whole else len(frac.lstrip("0")) or len(frac)
    else:
        sig = len(t.strip("0")) or 1
    return sig >= _MIN_SIG_DIGITS or (sig >= _LOW_PRECISION_SIG_DIGITS and decimals >= _LOW_PRECISION_MIN_DECIMALS)


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
            from . import result_checks as _checks

            return set(_checks.left_out_by_record(self.quest_root))  # type: ignore[attr-defined]
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
            parts.append(f"{reduced['note']} Say this in the paper with the numbers (it is among FI's facts about this run, "
                         "given below), and write no result for what was left out.")
        over = sized.get("over_topic_limit")
        if isinstance(over, dict) and over.get("note"):
            parts.append(f"{over['note']} Say this in the paper (it is among FI's facts about this run, given below).")
        return "\n\n".join(parts)

    # ---- a setting of the grid the code never reads ----------------------------------------------------------------------

    def _check_unread_settings(self, state: Any, simulate: Path) -> dict[str, Any] | None:
        """After the checks of the code before the run (the static gate is clean) and before the run is timed and the
        protocol frozen: a setting of ``protocol.grid`` (two values or more) that FI's reading of the code
        (:func:`core.unread_settings.unread_axes`) finds no use of. FI never takes a setting out of the grid on its own
        reading: a reading of code can be wrong, and a setting taken out is a study that was not run.

        * The study compares the setting (a factor the topic compares, as the plan's own topic map or a name match places
          it; a key of a check's case, of ``in_each_run``, or named by a metric or a comparison) and the simulation never
          reads it: a fault of the code. The simulation goes back to be repaired (a failed run, at most
          ``engine.exec_reflect_max_iterations`` times), naming the setting; after that the grid is kept as written.
        * Otherwise: one plain line in plan.md and run.log says FI's reading finds no use of it, and every setting is kept
          and run.

        Nothing is said when the reading is not certain or the code reads a name the plan does not give (a misspelt read
        is a fault of the code: core/grid_keys.py sends it back), after the freeze, or for a redesign. No model is asked.
        Returns the failed run for the repair, or ``None`` to go on."""
        from . import engine as _e
        from . import grid_keys as _grid_keys
        from . import unread_settings as _unread

        if _frozen.load(self.quest_root) is not None or int(state.get("iteration", 0) or 0) > 0:  # type: ignore[attr-defined]
            return None
        protocol = self._draft_protocol(state) or {}  # type: ignore[attr-defined]
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        if len(grid) < 1:
            return None
        files = {rel: text for rel, text in _grid_keys.code_files(simulate.parent).items()
                 if rel not in ("run.py", _trial_runner.SUBMIT_NAME)}
        misread = _grid_keys.problems(files, protocol, state.get("design"))
        if misread:
            self._log.info("[execute] the code reads a name the plan does not give (%s); left to the check of the code",  # type: ignore[attr-defined]
                           ", ".join(f"`{k}`" for r in misread for k in r.keys) or "a fixed number of runs")
            return None
        # A setting the study compares and the simulation itself never reads is the simulation's fault, even when the
        # analysis reads it (then it groups results that differ in nothing but their seeds).
        swept = [str(a) for a, v in grid.items() if isinstance(v, list) and len(v) >= 2]
        simulated, _why = _unread.setting_reads(files, analysis=False)
        if simulated is not None and any(a in simulated for a in swept):
            compared = self._settings_the_study_compares(state, protocol, [a for a in swept if a not in simulated])
            if compared:
                return self._unread_compared_setting(state, grid, compared, simulate)
        unread, why = _unread.unread_axes(files, protocol)
        if not unread:
            if why:
                self._log.info("[execute] FI could not tell whether the code reads every setting of the grid (%s); "  # type: ignore[attr-defined]
                               "every setting is kept and run", why)
            return None
        compared = self._settings_the_study_compares(state, protocol, unread)
        if compared:
            return self._unread_compared_setting(state, grid, compared, simulate)
        line = (f"FI's reading of the code finds no use of {', '.join(f'`{k}`' for k in unread)}; every setting is kept "
                "and run.")
        self._log.warning("[execute] %s", line)  # type: ignore[attr-defined]
        path = _plan.plan_path(self.quest_root)  # type: ignore[attr-defined]
        try:
            text = path.read_text(encoding="utf-8")
            if line not in text:
                edited = _plan.add_to_section(text, _e._PART_HEADING, ["", f"- {line}"])
                _plan.note_edit(self.quest_root, text)  # type: ignore[attr-defined]
                path.write_text(edited, encoding="utf-8")
                _plan.record_version(self.quest_root, edited, by="engine",  # type: ignore[attr-defined]
                                     note="FI's reading of the code finds no use of a setting; every setting is kept")
        except OSError as e:
            self._log.warning("[execute] FI could not add that line to plan.md: %r", e)  # type: ignore[attr-defined]
        return None

    def _settings_the_study_compares(self, state: Any, protocol: dict[str, Any], axes: list[str]) -> dict[str, str]:
        """``{setting: why it is part of the study}`` for each of ``axes`` that the study compares, from FI's records: a
        factor the topic compares (:func:`core.topic_requirements.compared_axes`), a key of a check's case, of
        ``in_each_run``, or a name a metric or the plan's comparisons use."""
        from . import in_run as _in_run
        from . import oracle_check as _oracle
        from . import topic_requirements as _tr

        out: dict[str, str] = {}
        req = _read_json(self.fi_dir / _tr.FILE)  # type: ignore[attr-defined]
        if isinstance(req, dict) and isinstance(req.get("factors"), list):
            for axis, factor in _tr.compared_axes(req, state.get("design") or {}, protocol).items():
                out.setdefault(axis, f"the topic compares {factor}")
        for oracle in _oracle.declared(protocol):
            own = _oracle.case_of(oracle)
            for axis in (own[0] or {}) if own else {}:
                out.setdefault(str(axis), f"the check `{oracle.get('name')}` runs at a value of it")
        for axis in _in_run.of(protocol) or {}:
            out.setdefault(str(axis), "each run compares its values")

        def names(value: Any) -> set[str]:
            if isinstance(value, str):
                return {value}
            if isinstance(value, dict):
                return {n for v in value.values() for n in names(v)}
            if isinstance(value, list):
                return {n for v in value for n in names(v)}
            return set()

        for metric in protocol.get("metrics") or []:
            for axis in names({k: v for k, v in metric.items() if k not in ("id", "description", "kind")}
                              if isinstance(metric, dict) else {}):
                out.setdefault(axis, f"the metric `{metric.get('id')}` is given per value of it")
        design = state.get("design") if isinstance(state.get("design"), dict) else {}
        for axis in names(protocol.get("comparisons")) | names(design.get("comparisons")):
            out.setdefault(axis, "the plan compares its values")
        return {axis: why for axis, why in out.items() if axis in axes}

    def _unread_compared_setting(self, state: Any, grid: dict[str, Any], compared: dict[str, str],
                                 simulate: Path) -> dict[str, Any] | None:
        """A setting the study compares and the code never reads: a fault of the code. Sent back to be repaired a bounded
        number of times; after that the grid is kept as written (the run then shows the setting changed nothing)."""
        said = "; ".join(f"`{axis}` ({why})" for axis, why in compared.items())
        if int(state.get("exec_reflect_iter") or 0) >= self.config.engine.exec_reflect_max_iterations:  # type: ignore[attr-defined]
            self._log.warning("[execute] the code still never reads %s; the repairs are used up, so FI keeps the grid as the "  # type: ignore[attr-defined]
                              "plan wrote it", said)
            return None
        self._log.warning("[execute] nothing in the code reads %s, a setting the study compares: sending the simulation back "  # type: ignore[attr-defined]
                          "to be repaired, not taking the setting out of the plan", said)
        print("[FI] the code never reads a setting the study compares; sending the simulation back to be repaired")
        values = "; ".join(f"`{axis}` takes {', '.join(str(v) for v in grid[axis][:6])}" for axis in compared)
        return self._code_review_failed_run(  # type: ignore[attr-defined]
            f"FI read the code before the run: nothing in it reads {said}. The plan's grid gives each run its own value "
            f"({values}), and the study compares the results across those values, so the simulation must use the value it is "
            "given (`cell[\"<name>\"]`, the name exactly as the grid spells it) where the plan's model uses it. Do not fix "
            "the value inside the code.", script=simulate.name)

    # ---- the run fits the time it is allowed ---------------------------------------------------------------------------

    def _grid_and_runs(self, state: Any) -> tuple[dict[str, list[Any]], int, dict[str, Any]]:
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        whole = _optimise_search.study_grid(protocol)
        grid = _optimise_search.with_axes(whole, {str(k): list(v) for k, v in whole.items() if isinstance(v, list) and v})
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
        return grid, max(1, int(protocol.get("runs_per_setting") or 1)), thresholds

    def _sizing_key(self, runner: Any, grid: dict[str, list[Any]], runs: int, limit_s: float,
                    inside: dict[str, list[Any]] | None = None) -> str:
        what = {"grid": grid, **({"sub_studies": grid.sub_studies} if getattr(grid, "sub_studies", None) else {}),
                **({"in_each_run": inside} if inside else {})}
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
        # The time the run is sized to: the topic's own limit when it states one (never more than FI's), else FI's
        # ``execution.timeout_s``, which stays the hard stop of a run in either case.
        hard_s = float(self.config.execution.timeout_s)  # type: ignore[attr-defined]
        topic_s = self._topic_limit_s(state)
        by_topic = topic_s is not None and topic_s < hard_s
        limit_s = float(topic_s) if by_topic else hard_s  # type: ignore[arg-type]
        budget_s = max(hard_s * float(self.config.engine.pilot_timeout_frac), 30.0)  # type: ignore[attr-defined]
        cached: dict[str, _estimate.Probe] = {}
        prior = _estimate.read_record(self.quest_root)  # type: ignore[attr-defined]
        # The note of a cut made before a resume is kept, whether the sizing finished or not.
        reduced = prior.get("reduced") if prior.get("fits") is not None else None
        # FI lowers the runs once; the lowered run is timed again, and a resume after the cut does not cut a second time.
        cut_done = bool(reduced) and prior.get("fits") is False
        speed_done = False
        for _round in range(3):
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
            # The checks of the pilot's trials get the time they always had (FI's own limit, not the topic's).
            if (broken := await self._pilot_gate(state, runner, python, env, grid, probes, thresholds, hard_s, budget_s)) is not None:
                return broken
            est = _estimate.estimate(grid, runs, runner.deterministic, probes)
            if est is None:
                return None
            line = _estimate.says(est, limit_s, topic=by_topic)
            base = {"key": key, "estimate_s": est.seconds * _estimate.SAFETY, "limit_s": limit_s, "says": line,
                    "reduced": reduced, "at": time.time(), "timed": est.timed, "cells": est.cells, "trials": est.trials,
                    "limit": "topic" if by_topic else "execution.timeout_s", "topic_limit_s": topic_s, "hard_limit_s": hard_s}
            # The smaller run FI chose is timed again: it goes on unless it is over the limit even without the margin
            # (the margin was spent choosing it; a second timing of the same run differs by about as much).
            if not _estimate.known_too_long(est, limit_s, timed_again=cut_done):
                if _estimate.known_too_long(est, limit_s):
                    line += (" (timed again with the fewer runs; without FI's margin of one half it is "
                             f"{_estimate.plain_duration(est.seconds)}, within the limit)")
                    base["says"] = line
                self._log.info("[execute] %s", line)  # type: ignore[attr-defined]
                print(f"[FI] {line}")
                self._say_headroom(est, hard_s)
                _estimate.write_record(self.quest_root, {**base, "fits": True})  # type: ignore[attr-defined]
                return None
            self._log.warning("[execute] %s, so it would not finish within it", line)  # type: ignore[attr-defined]
            if (not speed_done and not cut_done and int(self.config.engine.improve_rounds or 0) > 0  # type: ignore[attr-defined]
                    and isinstance(runner, _trial_runner.TrialsRunner)):
                # Before the runs are cut or the quest stops: try to make the simulation itself faster, same numbers.
                speed_done = True
                sped = await self._speed_up(state, runner, python, env, grid, runs, thresholds, inside, est, limit_s,
                                            budget_s=max(0.0, budget_s - self._pilot_spent), probes=probes)
                if sped.get("kept"):
                    cached.clear()
                    # the slower of two timings of the faster simulation is the one the next decision is made on
                    for p in sped.get("probes") or []:
                        cached[_estimate.probe_key(p.cell, runs)] = p
                    continue
            protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
            if cut_done:
                # The shorter run, timed again, is still over: over the topic's limit but within FI's own, it runs and the
                # paper says it was over; over FI's own, the quest stops (below).
                if by_topic and not _estimate.known_too_long(est, hard_s):
                    over = (f"Timed again with the fewer runs, the experiment is expected to take {self._padded(est)}, over the "
                            f"topic's limit of {_estimate.plain_duration(limit_s)}; FI runs it, and the paper reports the "
                            "measured run time against that limit.")
                    self._log.warning("[execute] %s", over)  # type: ignore[attr-defined]
                    if isinstance(reduced, dict):
                        reduced = {**reduced, "note": f"{reduced.get('note', '')} {over}".strip(),
                                   "over_topic_limit": "cut_still_over"}
                    self._say_headroom(est, hard_s)
                    _estimate.write_record(self.quest_root, {**base, "reduced": reduced, "fits": False})  # type: ignore[attr-defined]
                    return None
                fit = None
            else:
                # Fewer runs per setting is the only cut FI makes: every setting the study compares is kept. The runs go down
                # to the fewest the plan states, or, for the topic's own limit, below it (said in the paper). The shorter run
                # is timed again before it is used. No model is asked.
                fit = _estimate.fit_runs(grid, runs, runner.deterministic, probes, protocol, limit_s=limit_s,
                                         topic_limit=by_topic, spreads=est.spreads)
            over_topic = ""
            if fit is None and by_topic and not cut_done:
                # Not even the fewest runs meet the topic's limit: going below the plan's minimum would not meet it either,
                # so the plan's minimum stands and the run is nearest the limit there, over it (said in the paper). Only a
                # run over FI's own limit stops.
                floor = _estimate.runs_floor(protocol, runs, est.spreads) if _estimate.floor_is_stated(protocol) else runs
                floor = min(runs, max(floor, _estimate.runs_checks_need(protocol, grid)))  # what the plan's checks need
                least = _estimate.fewest_for_topic(protocol, grid, runs)
                fewest = _estimate.seconds_at(grid, least, runs, runner.deterministic, probes) or 0.0
                at_floor = _estimate.seconds_at(grid, floor, runs, runner.deterministic, probes) or est.seconds
                if at_floor * _estimate.SAFETY <= hard_s:
                    over_topic = (f"The topic's limit of {_estimate.plain_duration(limit_s)} cannot be met by fewer runs per "
                                  f"setting: even with {least} runs per setting the {est.cells:,} settings would take about "
                                  f"{_estimate.plain_duration(fewest * _estimate.SAFETY)} with FI's margin (each setting runs "
                                  "in a process of its own).")
                    if floor < runs:
                        fit = _estimate.Fit(runs=floor, seconds=at_floor, plan_floor=floor, below_plan=False)
                    else:
                        note = (f"{over_topic} FI ran the plan's {runs} runs per setting and all {est.cells:,} settings; the "
                                f"experiment is expected to take {self._padded(est)}, over the topic's limit.")
                        self._log.warning("[execute] %s", note)  # type: ignore[attr-defined]
                        print(f"[FI] {note}")
                        self._say_headroom(est, hard_s)
                        _estimate.write_record(self.quest_root, {**base, "fits": False, "over_topic_limit": {"note": note}})  # type: ignore[attr-defined]
                        return None
            if fit is None:
                text = self._too_long_text(est, hard_s)
                _estimate.write_record(self.quest_root, {**base, "fits": False, "stopped": text})  # type: ignore[attr-defined]
                return self._stop_too_long(text)
            note = self._lower_the_runs(state, runs, fit, est, limit_s, grid, protocol, topic=by_topic, over_topic=over_topic)
            if note is None:
                text = self._too_long_text(est, hard_s)
                _estimate.write_record(self.quest_root, {**base, "fits": False, "stopped": text})  # type: ignore[attr-defined]
                return self._stop_too_long(text)
            cut_done = True
            reduced = note
            self._sized_protocol = {**(self._sized_protocol or {}), "runs_per_setting": fit.runs}
            _estimate.write_record(self.quest_root, {**base, "fits": False, "reduced": reduced})  # type: ignore[attr-defined]
            await self._settle_plan_sources(state, protocol=self._draft_protocol(state))  # type: ignore[attr-defined]
            if over_topic:
                return None  # sized to the plan's minimum, knowingly over the topic's limit: nothing to gain by timing again
        return None

    def _topic_limit_s(self, state: Any) -> float | None:
        """The one run-time limit the topic states (in seconds), read as the paper's run record reads it, or ``None``."""
        from . import disclosure as _disclosure

        limit = _disclosure.topic_limit(self.quest_root, str(state.get("topic") or ""))  # type: ignore[attr-defined]
        return float(limit) if isinstance(limit, (int, float)) and limit > 0 else None

    @staticmethod
    def _padded(est: _estimate.Estimate) -> str:
        return ("at least " if est.at_least else "about ") + _estimate.plain_duration(est.seconds * _estimate.SAFETY)

    def _say_headroom(self, est: _estimate.Estimate, hard_s: float) -> None:
        """One line when FI's hard stop (``execution.timeout_s``) is close to what the run is expected to take: a run slower
        than estimated would be stopped before it ends."""
        if est.seconds * _estimate.SAFETY * 2 > hard_s:
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] FI stops a run that takes longer than execution.timeout_s (%s); this one is expected to take %s, "
                "so a run much slower than the estimate would be stopped before it ends",
                _estimate.plain_duration(hard_s), self._padded(est))

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

    def _design_after_sizing(self, state: Any) -> dict[str, Any] | None:
        """The state's design with the runs per setting FI lowered, or ``None`` when none were."""
        sized = getattr(self, "_sized_protocol", None)
        if not sized:
            return None
        design = dict(state.get("design") or {})
        design["protocol"] = {**(design.get("protocol") if isinstance(design.get("protocol"), dict) else {}), **sized}
        return design

    def _lower_the_runs(self, state: Any, runs: int, fit: _estimate.Fit, est: _estimate.Estimate, limit_s: float,
                        grid: dict[str, list[Any]], protocol: dict[str, Any], *, topic: bool = False,
                        over_topic: str = "") -> dict[str, Any] | None:
        """FI's own smaller run: the same settings with ``fit.runs`` runs each, written into plan.md as a version of the engine
        (no request to the plan's model). The record of it (the plain note the paper states, the precision those runs give,
        and whether they are fewer than the plan's minimum), or ``None`` when plan.md could not be changed. ``topic``: the
        limit is the one the topic states; ``over_topic``: why that limit cannot be met even so (the run then has the plan's
        minimum and is over it)."""
        fewer = int(fit.runs)
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
        whose = "the topic's limit" if topic else "the time allowed"
        cells = _estimate.cell_count(grid)
        if over_topic:
            note = (f"{over_topic} FI lowered the runs per setting from {runs} to the plan's minimum of {fewer} and kept all "
                    f"{cells} setting(s); the experiment is expected to take about "
                    f"{_estimate.plain_duration(fit.seconds * _estimate.SAFETY)}, over the topic's limit; the checks, "
                    "thresholds and tolerances are unchanged.")
        else:
            note = (f"FI made the experiment smaller so that it fits {whose} ({_estimate.plain_duration(limit_s)}): it would "
                    f"have taken {self._padded(est)}. FI lowered the runs per setting from {runs} to {fewer} and kept all "
                    f"{cells} setting(s); the checks, thresholds and tolerances are unchanged.")
        if fit.below_plan:
            note += (f" That is fewer than the {fit.plan_floor} runs per setting the plan allows at the least ("
                     + ("its minimum and its precision target" if fit.precision_known else "its stated minimum")
                     + "): the topic's time limit asked for it, so the plan's minimum number of runs is a part of the study "
                     "not fully done.")
        elif not _estimate.floor_is_stated(protocol) and topic and not over_topic:
            note += (" The plan states no fewest number of runs per setting; for the topic's own limit FI goes down to "
                     f"{_estimate.fewest_for_topic(protocol, grid, runs)} at the least.")
        precision = _estimate.expected_precision(protocol, fewer, est.spreads) if fit.precision_known else None
        if precision is not None:
            note += (f" With {fewer} runs per setting, the 95% interval of `{precision['metric']}` is about "
                     f"±{precision['half_width']:g} ({precision['how']})"
                     + (f"; the plan's target is ±{precision['target']:g}" if precision.get("target") else "") + ".")
        elif not fit.precision_known:
            metric, _kind = _estimate.precision_metric(protocol)
            note += (f" The precision {fewer} runs per setting give for `{metric}` was not known before the run: the timed "
                     "trials did not show its spread, so the 95% interval is the one the run itself gives.")
        self._log.warning("[execute] %s", note)  # type: ignore[attr-defined]
        print(f"[FI] {note}")
        return {"note": note, "from": {"grid": grid, "runs_per_setting": runs},
                "to": {"grid": grid, "runs_per_setting": fewer}, "limit": "topic" if topic else "execution.timeout_s",
                "below_plan_minimum": bool(fit.below_plan), "plan_minimum": fit.plan_floor, "precision": precision,
                "precision_known": bool(fit.precision_known),
                # why the run is over the topic's limit: sized to the plan's minimum (no fewer runs meet it), or, set when the
                # cut is timed again, still over it after the cut (_size_the_run)
                "over_topic_limit": "at_plan_minimum" if over_topic else False}

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
        grid = _optimise_search.study_grid(protocol) if isinstance(protocol.get("grid"), dict) else {}  # knows its sub-studies
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

    def _gate_crash_repair(self, issue: dict[str, Any], run: Any, rel: str, record: dict[str, Any],
                           state: Any = None) -> dict[str, Any] | None:
        """At the evidence gate, a shown fault's test crashed inside the program: the crash is recorded on the issue (unsent)
        and, while the budget of the crash repair before the run is left (:data:`REVIEW_CRASHES` a quest, and the run's own
        repairs), the gate's verdict sends the quest back to the run step (``repair_crash``), where
        :meth:`_review_crash_run` gives it to the repair of a failed run. ``None`` when the budget is spent: it is only
        recorded, and nothing stops."""
        issue["crash"] = {"at": run.at, "detail": str(run.detail or "")[:300], "test": rel, "at_gate": True}
        left = int(record.get("crashes_sent") or 0) < REVIEW_CRASHES and (
            int((state or {}).get("exec_reflect_iter") or 0) < self.config.engine.exec_reflect_max_iterations)  # type: ignore[attr-defined]
        if not left:
            issue["crash"]["sent"] = True
            issue["crash"]["not_sent"] = "the repairs of a crash are spent"
            return None
        why = (f"the test of {issue.get('id')} crashed inside the code at {run.at or 'an unknown line'} on the code that "
               f"ran ({_one(str(run.detail or ''), 160)}): a crash, not a shown fault; the code goes back to the crash repair")
        self._log.warning("[evidence_gate] %s", why)  # type: ignore[attr-defined]
        return {"verdict": "insufficient", "rationale": why, "gaps": [why], "repair_crash": True}

    async def _ask_finding_test_again(self, issue: dict[str, Any], error: Any, record: dict[str, Any],
                                      state: Any = None) -> Any:
        """At the evidence gate, a shown issue's test errors in its own code: the code model is asked once more for it,
        shown the error, as an assertion about values (:func:`core.finding_repair.ask_again`). Its ``TestRun``, or ``None``
        (asked before, no test came back, or the model could not be asked: an error here never stops anything)."""
        from . import code_review as _code_review
        from . import finding_repair as _finding_repair
        from . import review_pairs as _pairs

        if not isinstance(issue, dict) or issue.get("asked_again"):
            return None
        protocol = self._protocol_block(state) if state is not None else None  # type: ignore[attr-defined]
        protocol = protocol if isinstance(protocol, dict) else None
        values = _pairs.plan_values(protocol)

        async def ask(prompt: str, who: str, kind: str) -> str:
            try:
                return await self._chat(prompt, node="implement.finding_test")  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - a request that cannot be made shows nothing and stops nothing
                self._log.warning("[evidence_gate] the code model could not be asked again for the test of %s: %s",  # type: ignore[attr-defined]
                                  issue.get("id"), _one(str(exc)))
                return ""

        async def run_tests(paths: list[str]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for path in paths:
                if (run := await self._rerun_finding_test(path)) is not None:  # type: ignore[attr-defined]
                    out[path] = run
            return out

        async def snapshot() -> dict[str, str]:
            return {}

        code = self.quest_root / "code"  # type: ignore[attr-defined]
        env = _finding_repair.Env(
            root=self.quest_root, code=code, record=record,  # type: ignore[attr-defined]
            save=lambda: _code_review.save(self.quest_root, record),  # type: ignore[attr-defined]
            ask=ask, run_tests=run_tests, snapshot=snapshot, static=dict, keeps=lambda *a: "",
            prompts=self._prompts,  # type: ignore[attr-defined]
            topic=_pairs.scrub(self._topic_block(), values) or "(not stated)",  # type: ignore[attr-defined]
            equations=_code_review.equations_view(protocol, values), scrub=lambda text: _pairs.scrub(text, values),
            log=self._log, package=(self._package_in_use(state) or "") if state is not None else "")  # type: ignore[attr-defined]
        again = await _finding_repair.ask_again(env, issue, _finding_repair.target_file(issue, code), error)
        if again is not None:
            self._log.warning("[evidence_gate] the test of %s errored in its own code; FI asked once more for a test that "  # type: ignore[attr-defined]
                              "asserts on the values, and it %s", issue.get("id"),
                              {"fail": "fails on the code that ran", "pass": "passes on the code that ran"}.get(
                                  again.status, "errors too: it shows nothing"))
        return again

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

    async def _speed_up(self, state: Any, runner: Any, python: Any, env: Any, grid: dict[str, list[Any]], runs: int,
                        thresholds: dict[str, Any], inside: dict[str, list[Any]], est: Any, limit_s: float,
                        budget_s: float = 600.0, probes: list[Any] | None = None) -> dict[str, Any]:
        """The study would not fit the time allowed: up to ``engine.improve_rounds`` edits that make the simulation itself
        faster without changing a number (core/improve_speed.py). FI runs a fixed sample of trials, the checks' own cases
        included, before and after each edit; an edit is kept only when every number is the same and the sample takes
        clearly less time, and it passes every guard the correctness loop has. The study is then timed twice and the
        slower timing stands (``probes``), so the noise of one timing cannot make it fit. ``{"kept": bool, "rounds": [...],
        "calls": n, "speedup": x, "probes": [...] | None}``."""
        import asyncio

        from . import changelog as _changelog
        from . import code_project as _code_project
        from . import criteria as _criteria
        from . import improve as _improve
        from . import improve_speed as _speed
        from . import oracle_check as _oracle

        root = self.quest_root  # type: ignore[attr-defined]
        log = self._log  # type: ignore[attr-defined]
        limit = int(self.config.engine.improve_rounds or 0)  # type: ignore[attr-defined]
        module = runner.simulate.relative_to(root).as_posix()
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        # The guards of the correctness loop, over every check of the plan (not only the ones a criterion names): a speed-up
        # may not write in a value a check expects, change how a check's number is worked out (so a second implementation
        # stays a second implementation), or recognise the setting a check runs on.
        oracles = [o for o in _oracle.declared(protocol)]
        countable = _criteria.countable(protocol)
        trial_names = {c["trials"] for c in countable if c.get("trials")}
        expected, measures, settings = _improve.guards(protocol, oracles, countable, trial_names)
        cells = _speed.sample_cells(grid, oracles)
        for cell in cells:
            for key, value in cell.items():
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    settings.setdefault(str(key), []).append(value)
        timeout_s = max(60, int(self.config.execution.timeout_s))  # type: ignore[attr-defined]
        timeouts: dict[str, int] | None = None
        if probes is not None:
            # The sample is held to what the timing measured: a setting it did not see finish is left out, each gets a
            # few times its measured cost, and every sample of the step fits in the time left for timing.
            cells, timeouts, why = _speed.bounded_sample(cells, probes, runs=runs, deterministic=runner.deterministic,
                                                         budget_s=budget_s, samples=2 + limit)
            if not cells:
                log.info("[improve] The simulation is not sped up: %s.", why)
                return {"goal": "speed", "limit_s": limit_s, "rounds": [], "kept": False, "calls": 0, "probes": None,
                        "stopped": why}
            if why:
                log.info("[improve] %s; a speed-up is not tried on them.", why[0].upper() + why[1:])

        async def sample() -> Any:
            return await _speed.run_sample(self.executor, python, root, module, cells,  # type: ignore[attr-defined]
                                           deterministic=runner.deterministic, thresholds=thresholds, in_each_run=inside,
                                           env=env, timeout_s=timeout_s, log=log, timeouts=timeouts)

        async def time_study() -> list[Any] | None:
            probes, _why = await _estimate.measure(
                self.executor, python, root, module, grid, runs=runs, deterministic=runner.deterministic,  # type: ignore[attr-defined]
                limit_s=limit_s, budget_s=budget_s, env=env, thresholds=thresholds, cached=None, in_each_run=inside)
            return probes

        padded = est.seconds * _estimate.SAFETY
        record: dict[str, Any] = {"goal": "speed", "limit_s": limit_s, "estimate_s": padded, "rounds": [], "kept": False,
                                  "calls": 0, "probes": None}
        first = await sample()
        if first.failed or first.seconds <= 0:
            record["stopped"] = f"the sample did not run ({first.failed or 'no time measured'})"
            return record
        # The unchanged code is run a second time: a number that is not the same in both runs is not fixed by the code (a
        # measure of its own run time, say), so it cannot show that an edit changed a result and is left out of every comparison.
        again = await sample()
        if again.failed:
            record["stopped"] = f"the sample did not run a second time ({again.failed})"
            return record
        noisy = _speed.varying(first, again)
        if noisy:
            names = sorted(noisy)
            shown = [f"{k or 'its only setting'} trial {t} {n}" for k, t, n in names[:3]]
            record["varying"] = shown
            which = sorted({n.strip("/") for _k, _t, n in noisy})
            log.info("[improve] %d number(s) of the sample change from one run of the unchanged code to the next (%s); a "
                     "speed-up is judged on the other numbers.", len(noisy),
                     ", ".join(which[:3]) + (f" and {len(which) - 3} more" if len(which) > 3 else ""))
            if len(noisy) >= _speed.count_numbers(first):
                record["stopped"] = "every number of the sample differs from one run of the unchanged code to the next, so an edit cannot be judged"
                log.info("[improve] The simulation is not sped up: %s.", record["stopped"])
                return record
        needed = padded / max(limit_s, 1e-9)
        target = needed * _speed.MARGIN
        original = _improve.snapshot(root)
        best = {"files": original, "sample": first}
        tried = {_improve.fingerprint(original)}
        block = _speed.speed_block(padded, limit_s, first, cells, runs)
        log.info("[improve] The study would take about %s, more than the %s allowed. Trying to make the simulation itself "
                 "faster without changing any result (up to %d change(s)).", _estimate.plain_duration(padded),
                 _estimate.plain_duration(limit_s), limit)
        self._progress("[improve] The study is too slow for the time allowed; trying to make the simulation faster "  # type: ignore[attr-defined]
                       "without changing any result.")
        used = 0
        while used < limit:
            used += 1
            files = _improve.snapshot(root)
            allowed = _improve.editable(root)
            prompt = self._prompts["improve_speed"].substitute(  # type: ignore[attr-defined]
                topic_block=self._topic_block(), speed_block=block,  # type: ignore[attr-defined]
                history_block=_improve.history_block(record["rounds"]), editable=", ".join(allowed) or "(none)",
                code_block=_improve.code_block(files, allowed), rounds_left=f"This is round {used} of {limit}.")
            entry: dict[str, Any] = {"round": used}
            record["rounds"].append(entry)
            try:
                reply = await self._chat(prompt, node="improve")  # type: ignore[attr-defined]
                record["calls"] += 1
            except Exception as e:  # noqa: BLE001 -- no answer is no change
                entry.update(outcome="not asked", reason=repr(e)[:200])
                break
            from .engine import _parse_json_lenient

            edit, why = _improve.parse_edit(_parse_json_lenient(reply, node="improve"))
            new_text = None
            if edit is not None:
                new_text, why = _improve.check_edit(files, edit, allowed, tried=tried, expected=expected,
                                                    measures=measures, settings=settings)
            if edit is None or new_text is None:
                entry.update(outcome="refused", reason=why, why=edit.why if edit else "")
                log.info("[improve] Speed-up %d refused before it ran: %s", used, why)
                continue
            name = _improve.edited_path(edit)
            candidate = {**files, name: new_text}
            tried.add(_improve.fingerprint(candidate))
            entry.update(why=edit.why, file=name, diff=_improve.diff_text(edit))
            _improve.restore(root, candidate)
            saved = _improve.guard_bytes(root)
            before_guard = _improve.guard_hashes(root)
            after = await sample()
            touched = _improve.changed(before_guard, _improve.guard_hashes(root))
            if touched:
                _improve.put_back(root, saved)
                _improve.restore(root, best["files"])
                entry.update(outcome="aborted", reason="its run changed " + ", ".join(touched[:4]))
                log.warning("[improve] Speed-up %d stopped: its test run changed %s, which FI judges the code by; the "
                            "earlier version is back.", used, ", ".join(touched[:4]))
                break
            if after.failed:
                entry.update(outcome="not kept", reason=f"the sample did not run ({after.failed})")
            elif (bad := _speed.disagreements(first, after, ignore=noisy)):
                entry.update(outcome="not kept", reason=f"{len(bad)} number(s) of the sample changed, first: {bad[0]}")
            elif not _speed.fast_enough(best["sample"].seconds, after.seconds):
                entry.update(outcome="not kept", reason=f"the sample took {after.seconds:.3f} s against "
                                                         f"{best['sample'].seconds:.3f} s: not clearly faster")
            else:
                entry.update(outcome="kept", reason=f"the sample took {after.seconds:.3f} s against "
                                                    f"{best['sample'].seconds:.3f} s, with the same numbers")
                best = {"files": candidate, "sample": after}
                record["kept"] = True
                await asyncio.to_thread(
                    _code_project.record_change, root, f"speed-up {used}: {edit.why}", category="changed",
                    results=_changelog.results(
                        "no: every number of a fixed sample of trials (the checks' own cases included) is the same, "
                        f"and the sample runs {first.seconds / max(best['sample'].seconds, 1e-9):.1f} times faster."),
                    log=log)
            if entry["outcome"] != "kept":
                _improve.restore(root, best["files"])
            log.info("[improve] Speed-up %d %s: %s", used, "kept" if entry["outcome"] == "kept" else "not kept",
                     entry.get("reason"))
            if record["kept"] and first.seconds / max(best["sample"].seconds, 1e-9) >= target:
                break
        _improve.restore(root, best["files"])
        record["sample_s_before"] = first.seconds
        record["sample_s_after"] = best["sample"].seconds
        record["speedup"] = first.seconds / max(best["sample"].seconds, 1e-9)
        record["needed"] = needed
        if record["kept"]:
            # Timed twice; the slower timing stands (one timing of the same study differs by a quarter or more).
            runs_timed = []
            for _ in range(2):
                probes = await time_study()
                if probes is None:
                    break
                e = _estimate.estimate(grid, runs, runner.deterministic, probes)
                if e is not None:
                    runs_timed.append((e.seconds, probes))
            if len(runs_timed) == 2:
                slower = max(runs_timed, key=lambda t: t[0])
                record["probes"] = slower[1]
                record["estimate_after_s"] = [round(t[0] * _estimate.SAFETY, 1) for t in runs_timed]
            log.info("[improve] The simulation is %.1f times faster with the same numbers on the sample%s.", record["speedup"],
                     f"; timed twice, the study would now take about {_estimate.plain_duration(max(record['estimate_after_s']))}"
                     if record.get("estimate_after_s") else "")
        else:
            log.info("[improve] No change made the simulation clearly faster with the same numbers; it is left as it was.")
        try:
            (root / ".fi").mkdir(parents=True, exist_ok=True)
            (root / ".fi" / "improve_speed.json").write_text(
                json.dumps({k: v for k, v in record.items() if k != "probes"}, indent=1, default=str), encoding="utf-8")
        except OSError:
            pass
        return record

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
