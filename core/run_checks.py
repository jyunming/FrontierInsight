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

import asyncio
import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import replace as _replace
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
from . import step_coverage as _step_coverage
from . import stop_power as _stop_power
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


def typed_result_exempt(protocol: dict[str, Any] | None, design: dict[str, Any] | None = None) -> set[str]:
    """Names the plan fixes (its fixed settings, the grid's settings, what each run compares, the thresholds): echoed back,
    not results. The one place that list is made (the engine's check and :mod:`core.code_checks` both read it)."""
    protocol = protocol if isinstance(protocol, dict) else {}
    design = design if isinstance(design, dict) else {}
    names = set(_forms.fixed_settings_of_design({**design, "protocol": protocol}))
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    names |= {str(k) for k in grid} | {str(k) for k in _in_run.of(protocol) or {}}
    thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
    return names | {str(k) for k in thresholds}


def settings_the_study_compares(fi_dir: Path, design: Any, protocol: dict[str, Any], axes: list[str]) -> dict[str, str]:
    """``{setting: why it is part of the study}`` for each of ``axes`` that the study compares, from FI's records: a
    factor the topic compares (:func:`core.topic_requirements.compared_axes`, read from ``fi_dir``), a key of a check's
    case, of ``in_each_run``, or a name a metric or the plan's comparisons use."""
    from . import in_run as _in_run
    from . import oracle_check as _oracle
    from . import topic_requirements as _tr

    design = design if isinstance(design, dict) else {}
    out: dict[str, str] = {}
    req = _read_json(Path(fi_dir) / _tr.FILE)
    if isinstance(req, dict) and isinstance(req.get("factors"), list):
        for axis, factor in _tr.compared_axes(req, design, protocol).items():
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
    for axis in names(protocol.get("comparisons")) | names(design.get("comparisons")):
        out.setdefault(axis, "the plan compares its values")
    return {axis: why for axis, why in out.items() if axis in axes}


def unread_findings(files: dict[str, str], protocol: dict[str, Any], design: Any, fi_dir: Path) -> tuple[
        list[Any], dict[str, str], list[str], str]:
    """FI's reading of which settings of ``protocol.grid`` the code never reads: ``(misread, compared, unread, why)``.
    ``misread``: reads of a name the plan does not give (a fault the static check owns: nothing else is said then).
    ``compared``: ``{setting: why}`` for a swept setting the code never reads that the study compares (a fault of the
    code; :func:`settings_the_study_compares`). ``unread``: the swept settings nothing reads (the compared ones among
    them), or ``[]`` with ``why`` when the reading is not certain. Nothing is run and no model is asked."""
    from . import grid_keys as _grid_keys
    from . import unread_settings as _unread

    misread = _grid_keys.problems(files, protocol, design)
    if misread:
        return misread, {}, [], ""
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    # A setting the study compares and the simulation itself never reads is the simulation's fault, even when the
    # analysis reads it (then it groups results that differ in nothing but their seeds).
    swept = [str(a) for a, v in grid.items() if isinstance(v, list) and len(v) >= 2]
    simulated, _why = _unread.setting_reads(files, analysis=False)
    if simulated is not None and any(a in simulated for a in swept):
        compared = settings_the_study_compares(fi_dir, design, protocol, [a for a in swept if a not in simulated])
        if compared:
            return [], compared, [], ""
    unread, why = _unread.unread_axes(files, protocol)
    return [], (settings_the_study_compares(fi_dir, design, protocol, unread) if unread else {}), unread, why


#: How many designs of a search's own space (besides its baseline) a speed-up is held to.
SEARCH_SAMPLE_DESIGNS = 3


def _search_rounds(block: dict[str, Any], search_n: int, at_once: int) -> tuple[int, int, int]:
    """How the search's ``search_n`` evaluations wait, from the search itself rehearsed on a stand-in objective
    (:func:`core.optimise_search.rehearse`, with the run's own ``batch``): ``(alone, batches, together)``: the evaluations
    asked for one at a time, the rounds of ``at_once`` the designs asked for together run in, and how many designs those
    are. Evaluations the rehearsal did not spend (a start that converged early on the stand-in) are counted alone, and a
    search that cannot be rehearsed (a library method) is counted one at a time."""
    widths = _optimise_search.rehearse(block, batch=at_once > 1) if at_once > 1 else None
    if not widths:
        return search_n, 0, 0
    alone = sum(w for w in widths if w == 1)
    together = sum(w for w in widths if w > 1)
    batches = sum(math.ceil(w / at_once) for w in widths if w > 1)
    return alone + max(0, search_n - alone - together), batches, together


def _search_sample(block: dict[str, Any], grid: Any) -> list[dict[str, Any]]:
    """The settings a search's speed-up is held to besides its coarse scan's: the baseline and
    :data:`SEARCH_SAMPLE_DESIGNS` designs spread over the search's space (fixed points of it, so the sample is the same on
    every run), each with the scan's other settings at their base values. A design the scan never computes (an array
    of free values, say) is where a search spends its time, so a change that alters a number there is caught too."""
    try:
        space = _optimise_search.Space(block["design_variables"])
        base = dict(next(iter(_estimate.probe_cells(grid)))[2])
        baseline = dict((block.get("baseline") or {}).get("values") or {})
    except (KeyError, TypeError, ValueError, StopIteration):
        return []
    out = [{**base, **baseline}] if baseline else []
    n = SEARCH_SAMPLE_DESIGNS
    for i in range(n):
        # points of the unit cube spread over it by the golden ratio (each coordinate its own offset), off its centre,
        # where a baseline often sits
        u = [(0.5 + 0.618034 * (i + 1) + 0.37 * k) % 1.0 for k in range(space.dims)]
        try:
            cell = {**base, **space.design(u)}
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if cell not in out:
            out.append(cell)
    return out


class RunChecksMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    # ---- a reported result is computed --------------------------------------------------------------------------------

    def _typed_result_exempt(self, state: Any) -> set[str]:
        """Names the plan fixes (its fixed settings, the grid's settings, what each run compares, the thresholds): echoed back,
        not results."""
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        design = state.get("design") if isinstance(state.get("design"), dict) else {}
        return typed_result_exempt(protocol, design)

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
        written: str | None = None  # the text a try of this pass wrote and no run has used yet
        text = path.read_text(encoding="utf-8")
        used = self._typed_counter(text)
        while found and used < TYPED_RESULT_REPAIRS:
            used += 1
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] %s returns numbers typed into the code (%s); asking for them to be computed (%d of %d)",
                path.name, "; ".join(f.says() for f in found[:4]), used, TYPED_RESULT_REPAIRS)
            earlier = path.read_text(encoding="utf-8") if written is not None else ""
            wrote = await self._repair_typed_results(state, path, found)
            rewrote = wrote or rewrote
            text = path.read_text(encoding="utf-8")
            if written is not None and wrote and earlier and earlier != text:
                # the version the first try wrote was replaced by this one before any run of it
                await self._keep_overwritten_try(path, used - 1, earlier, "typed-results repair replaced before it ran: "
                                                 "a later try wrote over it")
            written = text if wrote else written
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

    async def _keep_overwritten_try(self, path: Path, number: int, text: str, message: str) -> None:
        """Keep a version written into ``code/`` and overwritten before any run, on a side reference of the folder's git
        history. Best effort: never changes what the check does."""
        try:
            from . import code_project as _project

            code_dir = self.quest_root / "code"  # type: ignore[attr-defined]
            rel = path.relative_to(code_dir).as_posix()
            await asyncio.to_thread(_project.keep_refused, self.quest_root, f"typed-{path.stem}-{number}",  # type: ignore[attr-defined]
                                    {rel: text}, message, log=self._log)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 -- the record is a convenience; the check goes on whatever happens to it
            pass

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
        package_was = self._package_snapshot()  # type: ignore[attr-defined]
        path.write_text(new_code, encoding="utf-8")
        if await self._simulation_kept_or_back(state, path, code, package_was, "execute"):  # type: ignore[attr-defined]
            return False
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
        from . import code_review as _code_review

        now = _code_review.version(self.quest_root / "code")  # type: ignore[attr-defined]
        for p in self._cannot_repair_problems():  # type: ignore[attr-defined]  # core/engine.py
            if _stop_power.may_stop(p.get("source")):
                continue
            if p.get("code_version") and p["code_version"] != now:
                # The code was repaired again after this was found, and no check looked at it again.
                parts.append(f"FI's checks of an earlier version of the code found this: {p['text']}. The code was "
                             "repaired again after that and this was not checked again. Say it in the limitations as "
                             "not known to be fixed, not as a fact of the results.")
                continue
            parts.append(f"FI's checks of the code before the run found this, and the repairs did not remove it: {p['text']}. "
                         f"It is {_stop_power.source_words(p.get('source'))}, so the run went on. Say it in the "
                         "limitations as FI's reading of the code, not as a fact of the results.")
        if observed := self._no_difference_note():
            parts.append(observed)
        if uncomputed := self._step_coverage_note():
            parts.append(uncomputed)
        sized = _read_json(self.quest_root / _estimate.RECORD)  # type: ignore[attr-defined]
        reduced = sized.get("reduced")
        if isinstance(reduced, dict) and reduced.get("note"):
            parts.append(f"{reduced['note']} Say this in the paper with the numbers (it is among FI's facts about this run, "
                         "given below), and write no result for what was left out.")
        over = sized.get("over_topic_limit")
        if isinstance(over, dict) and over.get("note"):
            parts.append(f"{over['note']} Say this in the paper (it is among FI's facts about this run, given below).")
        if scope := self._scope_gap_note():
            parts.append(scope)
        return "\n\n".join(parts)

    def _scope_gap_note(self) -> str:
        """For the writer: each result or setting of the study the person chose that the frozen plan has no name for
        (core/scope_cards.py ``gaps``, a reading by names). It is told in the limitations and never stops the quest."""
        from . import scope_cards as _sc

        record = _sc.read_record(self.quest_root)  # type: ignore[attr-defined]
        card = record.get("chosen") if isinstance(record, dict) else None
        protocol = _frozen.protocol_of(self.quest_root)  # type: ignore[attr-defined]
        if not isinstance(card, dict) or not card or protocol is None:
            return ""
        design = {"protocol": protocol}
        names = [r for r in _sc._strings(card.get("main_results")) if _sc.gaps({"main_results": [r]}, design)]
        names += [s for s in _sc._strings(card.get("compared_settings"))
                  if _sc.gaps({"compared_settings": [s]}, design)]
        return "\n".join(f"The person chose a study that names {x}; the plan does not cover it and gives no reason. "
                         "Say so in the limitations." for x in names)

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

        if _frozen.load(self.quest_root) is not None or int(state.get("iteration", 0) or 0) > 0:  # type: ignore[attr-defined]
            return None
        protocol = self._draft_protocol(state) or {}  # type: ignore[attr-defined]
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        if len(grid) < 1:
            return None
        files = {rel: text for rel, text in _grid_keys.code_files(simulate.parent).items()
                 if rel not in ("run.py", _trial_runner.SUBMIT_NAME)}
        misread, compared, unread, why = unread_findings(files, protocol, state.get("design"), self.fi_dir)  # type: ignore[attr-defined]
        if misread:
            self._log.info("[execute] the code reads a name the plan does not give (%s); left to the check of the code",  # type: ignore[attr-defined]
                           ", ".join(f"`{k}`" for r in misread for k in r.keys) or "a fixed number of runs")
            return None
        if compared:
            return self._unread_compared_setting(state, grid, compared, simulate)
        if not unread:
            if why:
                self._log.info("[execute] FI could not tell whether the code reads every setting of the grid (%s); "  # type: ignore[attr-defined]
                               "every setting is kept and run", why)
            return None
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

    def _record_step_coverage(self, state: Any, simulate: Path) -> list[str]:
        """After the checks of the code and before the run: is each metric the simulation returns computed through the
        functions of the model's package (core/step_coverage.py)? A ``run_trial`` that returns the plan's true value plus noise
        with a matching standard error, and never calls the model, passes the failure-rate, calibration, seed and typed-result
        checks; only this question tells it from an honest one.

        RECORD ONLY, as the checks that read code without running it always were: the metrics that are not computed through the
        package are written to ``.fi/step_coverage.json``, become a gap in the evidence record (core/evidence.py) and a line in
        QUALITY.md, and are given to the writer for the limitations. The quest is neither stopped nor is the code sent back,
        because a reading of syntax can miss a call it should have followed. Nothing is said when the quest has no package or
        the simulation has no ``run_trial`` / ``run_cell``. Never raises; returns the warnings."""
        try:
            from . import code_layout as _code_layout
            from . import code_review as _code_review

            code_dir = self.quest_root / "code"  # type: ignore[attr-defined]
            sources = _code_layout.package_sources(code_dir)
            names = _step_coverage.package_function_names(sources)
            protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
            metrics = protocol.get("metrics") if isinstance(protocol.get("metrics"), list) else []
            path = self.fi_dir / _step_coverage.RECORD  # type: ignore[attr-defined]
            if not names or not metrics or not simulate.is_file():
                path.unlink(missing_ok=True)   # an earlier reading is not about this code
                return []
            warnings = _step_coverage.check(simulate.read_text(encoding="utf-8"), names, metrics, only_returned=True,
                                            package_modules=_step_coverage.package_module_names(sources))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"warnings": warnings, "package_functions": names,
                                        "code_version": _code_review.version(code_dir)}, indent=2) + "\n", encoding="utf-8")
            for w in warnings:
                self._log.warning("[execute] FI's step check: %s", w)  # type: ignore[attr-defined]
            return warnings
        except Exception as e:  # noqa: BLE001 -- a check that cannot be made never stops a quest
            self._log.warning("[execute] FI could not check which numbers are computed through the model (%s)",  # type: ignore[attr-defined]
                              str(e)[:200] or type(e).__name__)
            return []

    def _step_coverage_note(self) -> str:
        """For the writer: a metric the simulation returns without computing it through the model's functions."""
        warnings = [str(w) for w in _read_json(self.fi_dir / _step_coverage.RECORD).get("warnings") or []]  # type: ignore[attr-defined]
        if not warnings:
            return ""
        return ("FI's reading of the simulation found: " + "; ".join(_one(w, 300) for w in warnings[:4])
                + ".\nSay so plainly in the limitations: those numbers are not shown to come from the model's equations.")

    def _settings_the_study_compares(self, state: Any, protocol: dict[str, Any], axes: list[str]) -> dict[str, str]:
        """``{setting: why it is part of the study}`` for each of ``axes`` that the study compares, from FI's records
        (:func:`settings_the_study_compares`)."""
        return settings_the_study_compares(self.fi_dir, state.get("design"), protocol, axes)  # type: ignore[attr-defined]

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
        # How many settings run at once changes how long the run takes: a different number is timed again.
        at_once = int(getattr(runner, "at_once", 1) or 1)
        return (_trial_runner._run_key(runner.simulate, what, runs, 0, runner.deterministic) + f"|{int(limit_s)}"
                + (f"|{at_once}" if at_once > 1 else ""))

    async def _size_the_run(self, state: Any, runner: Any, python: Any, env: Any) -> dict[str, Any] | None:
        """Before the protocol is frozen and the study starts: time a few real trials, add them up, and compare the total with
        ``execution.timeout_s``. Fits (or cannot be timed): go on, saying so in one line. Does not fit: FI lowers the runs per
        setting itself (no model is asked, every setting is kept, never below the fewest runs the plan states), times the
        shorter run again, and when it still would not fit returns the quest's stop (no paper). ``None`` to go on.

        Only the trial contract is timed, and only before the first freeze: a frozen protocol cannot be changed, a search for
        the best design or a cluster job has its own limits, and a resume of a run that already fits is not timed again."""
        self._sized_protocol: dict[str, Any] | None = None
        self._pilot_spent = 0.0  # seconds the pilot's extra trials took in this sizing: they count against the same budget
        # Every sizing starts with no cut: what an earlier sizing set (the settings planned, the run's id) is decided again below.
        if isinstance(runner, _trial_runner.TrialsRunner):
            runner.first_cells = None
            runner.run_id = ""
        # A run_trial that returns the mean of many trials (it calls run_cell) is a fault of the code, not of the plan's size:
        # it goes back to the code step, never to the plan to shrink the grid (which dropped settings the topic required).
        if isinstance(runner, _trial_runner.TrialsRunner):
            try:
                text = runner.simulate.read_text(encoding="utf-8")
                aggregate = _trial_runner.trial_aggregates(text, self.quest_root)  # type: ignore[attr-defined]
                first = self._first_study_cell(self._protocol_block(state) or {}) if aggregate else None  # type: ignore[attr-defined]
                if aggregate and first is not None and await _trial_runner.settle_aggregate(
                        self.executor, python, self.quest_root, runner.simulate, cell=first[0], inside=first[1],  # type: ignore[attr-defined]
                        timeout_s=int(self.config.execution.timeout_s), env=env, log=self._log) is not False:  # type: ignore[attr-defined]
                    aggregate = ""  # two seeds gave the same numbers, or the call could not be measured (said in the log)
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
        if not isinstance(runner, _trial_runner.TrialsRunner) or self.config.execution.background_jobs:  # type: ignore[attr-defined]
            return None
        # A frozen protocol (or a redesign) cannot be cut, but its run is still timed and the estimate said before it starts:
        # a redesign's run took four hours with nothing said beforehand because only the first design was timed.
        if int(state.get("iteration", 0) or 0) > 0:  # type: ignore[attr-defined]
            await self._say_estimate_only(state, runner, python, env)
            return None
        if _frozen.load(self.quest_root) is not None:  # type: ignore[attr-defined]
            return None  # a resume of a run that was timed before it was frozen
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
        at_once = int(getattr(runner, "at_once", 1) or 1)
        slowdown: float | None = None  # measured once, only when the settings one after another would not fit
        # The timing with the most paid once: a later timing runs on the cache the earlier one filled (run_estimate.with_once).
        once_seen: _estimate.Estimate | None = None
        if float(prior.get("once_s") or 0.0) > 0:  # a resume times on the cache the earlier session filled
            once_seen = _estimate.Estimate(seconds=0.0, at_least=False, complete=True, cells=0, trials=0,
                                           once_s=float(prior["once_s"]), once_axes=list(prior.get("once_axes") or []))
        for _round in range(3):
            grid, runs, thresholds = self._grid_and_runs(state)
            inside = _in_run.of(self._protocol_block(state) or {})  # type: ignore[attr-defined]
            key = self._sizing_key(runner, grid, runs, limit_s, inside)
            record = _estimate.read_record(self.quest_root)  # type: ignore[attr-defined]
            if record.get("key") == key and record.get("fits"):
                self._log.info("[execute] %s (timed before; nothing in the experiment has changed)",  # type: ignore[attr-defined]
                               record.get("says") or "the experiment fits")
                return None
            if record.get("key") == key and record.get("truncated"):
                self._log.info("[execute] %s (timed before; nothing in the experiment has changed)",  # type: ignore[attr-defined]
                               record.get("says") or "the experiment is run up to the time limit")
                runner.cost = None  # type: ignore[attr-defined]  # the plan's order
                runner.run_id = str((record.get("truncated") or {}).get("run_id") or "")  # type: ignore[attr-defined]
                runner.first_cells = (record.get("truncated") or {}).get("planned_settings")  # type: ignore[attr-defined]
                return None
            if record.get("key") == key and record.get("stopped"):
                return self._stop_too_long(str(record["stopped"]), fit=record.get("fit"))
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
            if isinstance(runner, _trial_runner.TrialsRunner):
                runner.cost = _estimate.setting_cost(grid, probes)  # the run starts the costliest settings first
            est = _estimate.with_once(_estimate.estimate(grid, runs, runner.deterministic, probes), once_seen)
            if est is None:
                return None
            if est.once_s > (once_seen.once_s if once_seen is not None else 0.0):
                once_seen = est
            if at_once > 1 and _estimate.known_too_long(est, limit_s):
                if slowdown is None:
                    slowdown = await self._slowdown_at_once(runner, python, env, probes, runs, thresholds, inside,
                                                            at_once, budget_s)
                if slowdown is not None:
                    est = _estimate.with_once(_estimate.estimate(grid, runs, runner.deterministic, probes, at_once=at_once,
                                                                 slowdown=slowdown), once_seen)
            line = _estimate.says(est, limit_s, topic=by_topic)
            base = {"key": key, "estimate_s": est.seconds * _estimate.SAFETY, "limit_s": limit_s, "says": line,
                    "reduced": reduced, "at": time.time(), "timed": est.timed, "cells": est.cells, "trials": est.trials,
                    "limit": "topic" if by_topic else "execution.timeout_s", "topic_limit_s": topic_s, "hard_limit_s": hard_s,
                    "once_s": est.once_s, "once_axes": list(est.once_axes),
                    "at_once": est.at_once}
            # The smaller run FI chose is timed again: it goes on unless it is over the limit even without the margin
            # (the margin was spent choosing it; a second timing of the same run differs by about as much).
            # More settings than FI lists: the whole grid is never built. The settings the time allowed can reach are planned,
            # in the plan's order, and run up to the limit (a study that fits would still be cut at what FI lists).
            oversized = _optimise_search.grid_size(grid) > _optimise_search.MAX_LISTED_SETTINGS
            if not oversized and not _estimate.known_too_long(est, limit_s, timed_again=cut_done):
                if _estimate.known_too_long(est, limit_s):
                    line += (" (timed again with the fewer runs; without FI's margin of one half it is "
                             f"{_estimate.plain_duration(est.seconds)}, within the limit)")
                    base["says"] = line
                self._log.info("[execute] %s", line)  # type: ignore[attr-defined]
                print(f"[FI] {line}")
                self._say_headroom(est, hard_s)
                _estimate.write_record(self.quest_root, {**base, "fits": True})  # type: ignore[attr-defined]
                return None
            if oversized:
                fits_in = self._what_fits(est, hard_s)
                reach = int(fits_in["settings_that_fit"]) if fits_in else _optimise_search.MAX_LISTED_SETTINGS
                first = max(1, min(_optimise_search.MAX_LISTED_SETTINGS, reach))
                if self._run_up_to_the_limit(runner, est, hard_s, _estimate.cheapest_setting_s(probes), kind="grid",
                                             base=base, detail=self._too_long_text(est, hard_s), fit=fits_in,
                                             first_cells=first):
                    return None
                text = _optimise_search.grid_too_large(_optimise_search.study_grid(self._protocol_block(state) or {})) \
                    or self._too_long_text(est, hard_s)
                _estimate.write_record(self.quest_root, {**base, "fits": False, "stopped": text, "fit": fits_in})  # type: ignore[attr-defined]
                return self._stop_too_long(text, fit=fits_in)
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
                                         topic_limit=by_topic, spreads=est.spreads, at_once=est.at_once,
                                         slowdown=est.slowdown)
            over_topic = ""
            if fit is None and by_topic and not cut_done:
                # Not even the fewest runs meet the topic's limit: going below the plan's minimum would not meet it either,
                # so the plan's minimum stands and the run is nearest the limit there, over it (said in the paper). Only a
                # run over FI's own limit stops.
                floor = _estimate.runs_floor(protocol, runs, est.spreads) if _estimate.floor_is_stated(protocol) else runs
                floor = min(runs, max(floor, _estimate.runs_checks_need(protocol, grid)))  # what the plan's checks need
                least = _estimate.fewest_for_topic(protocol, grid, runs)
                fewest = _estimate.seconds_at(grid, least, runs, runner.deterministic, probes, at_once=est.at_once,
                                              slowdown=est.slowdown) or 0.0
                at_floor = _estimate.seconds_at(grid, floor, runs, runner.deterministic, probes, at_once=est.at_once,
                                                slowdown=est.slowdown) or est.seconds
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
                fits_in = self._what_fits(est, hard_s)
                if self._run_up_to_the_limit(runner, est, hard_s, _estimate.cheapest_setting_s(probes), kind="grid", base=base,
                                             detail=text, fit=fits_in):
                    return None
                _estimate.write_record(self.quest_root, {**base, "fits": False, "stopped": text, "fit": fits_in})  # type: ignore[attr-defined]
                return self._stop_too_long(text, fit=fits_in)
            note = self._lower_the_runs(state, runs, fit, est, limit_s, grid, protocol, topic=by_topic, over_topic=over_topic)
            if note is None:
                text = self._too_long_text(est, hard_s)
                fits_in = self._what_fits(est, hard_s)
                if self._run_up_to_the_limit(runner, est, hard_s, _estimate.cheapest_setting_s(probes), kind="grid", base=base,
                                             detail=text, fit=fits_in):
                    return None
                _estimate.write_record(self.quest_root, {**base, "fits": False, "stopped": text, "fit": fits_in})  # type: ignore[attr-defined]
                return self._stop_too_long(text, fit=fits_in)
            cut_done = True
            reduced = note
            self._sized_protocol = {**(self._sized_protocol or {}), "runs_per_setting": fit.runs}
            _estimate.write_record(self.quest_root, {**base, "fits": False, "reduced": reduced})  # type: ignore[attr-defined]
            await self._settle_plan_sources(state, protocol=self._draft_protocol(state))  # type: ignore[attr-defined]
            if over_topic:
                return None  # sized to the plan's minimum, knowingly over the topic's limit: nothing to gain by timing again
        return None

    async def _say_estimate_only(self, state: Any, runner: Any, python: Any, env: Any) -> None:
        """The estimate of a run whose protocol cannot be changed (frozen, or a redesign): time the same settings as for a first
        design, say the one plain line, and go on. Nothing is cut, stopped or recorded as sized, and a timing that fails or finds
        nothing says nothing (a failed timing never prints an estimate)."""
        try:
            hard_s = float(self.config.execution.timeout_s)  # type: ignore[attr-defined]
            topic_s = self._topic_limit_s(state)
            by_topic = topic_s is not None and topic_s < hard_s
            limit_s = float(topic_s) if by_topic else hard_s  # type: ignore[arg-type]
            budget_s = max(hard_s * float(self.config.engine.pilot_timeout_frac), 30.0)  # type: ignore[attr-defined]
            grid, runs, thresholds = self._grid_and_runs(state)
            inside = _in_run.of(self._protocol_block(state) or {})  # type: ignore[attr-defined]
            probes, why = await _estimate.measure(
                self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(), grid,  # type: ignore[attr-defined]
                runs=runs, deterministic=runner.deterministic, limit_s=limit_s, budget_s=budget_s, env=env,
                thresholds=thresholds, in_each_run=inside)
            if probes is None:
                self._log.info("[execute] the run was not timed before it started (%s)", why)  # type: ignore[attr-defined]
                return
            est = _estimate.estimate(grid, runs, runner.deterministic, probes)
            if est is None:
                return
            line = _estimate.says(est, limit_s, topic=by_topic)
            self._log.info("[execute] %s (this design is not cut: its plan is already fixed)", line)  # type: ignore[attr-defined]
            print(f"[FI] {line}")
            self._say_headroom(est, hard_s)
        except Exception as exc:  # noqa: BLE001 - a number said for information must never stop the run
            self._log.warning("[execute] the run could not be timed before it started (%r)", exc)  # type: ignore[attr-defined]

    async def _slowdown_at_once(self, runner: Any, python: Any, env: Any, probes: list[Any], runs: int,
                                thresholds: dict[str, Any], inside: Any, at_once: int, budget_s: float,
                                deterministic: bool | None = None) -> float | None:
        """How many times slower a setting runs when ``at_once`` run at the same time on this machine
        (:func:`core.run_estimate.slowdown_at_once`), or ``None`` when it could not be measured in the time left for timing;
        then the estimate counts the settings one after another, as before. It is measured on the costliest timed setting
        that finished and can be timed again in the time left: the costly settings are most of the run, and running beside
        others slows them the most (a short setting is mostly its process start, which hardly slows)."""
        left = budget_s - self._pilot_spent

        def timing_of(p: Any) -> float:
            # What timing the setting took: its timed trials, not the whole setting.
            if p.per_trial_s is not None and p.timed_trials > 0:
                return min(p.seconds, (p.fixed_s or 0.0) + p.timed_trials * p.per_trial_s)
            return p.seconds

        finished = [p for p in probes if not p.at_least and p.seconds > 0]
        # Alone once, then beside the others: at worst as slow as all of them one after another.
        fits = [p for p in finished if left >= timing_of(p) * (1 + at_once) + 10]
        base = max(fits, key=timing_of, default=None)
        if base is None:
            why = "no timed setting finished" if not finished else "no time was left for it"
            self._log.info("[execute] FI did not measure how much slower settings run at the same time (%s); the "  # type: ignore[attr-defined]
                           "estimate counts them one after another", why)
            return None
        timing = timing_of(base)
        started = time.monotonic()
        factor, why = await _estimate.slowdown_at_once(
            self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
            dict(base.cell), at_once=at_once, runs=runs,
            deterministic=runner.deterministic if deterministic is None else deterministic,
            timeout_s=int(max(30, 3 * timing + 10)), env=env, thresholds=thresholds, in_each_run=inside)
        self._pilot_spent += time.monotonic() - started
        if factor is None:
            self._log.info("[execute] FI could not measure how much slower settings run at the same time (%s); the "  # type: ignore[attr-defined]
                           "estimate counts them one after another", why or "no reason given")
            return None
        self._log.info("[execute] with %d settings running at the same time, each took %.1f times as long as alone (timed "  # type: ignore[attr-defined]
                       "on a setting that took %s alone)", at_once, factor, _estimate.plain_duration(timing))
        return factor

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

    async def _size_the_scan(self, state: Any, runner: Any, python: Any, env: Any,
                             speed_done: bool = False) -> dict[str, Any] | None:
        """The coarse scan of a search for the best design runs every setting of the plan's grid, so its size is counted
        (the product of the axes' lengths, nothing listed) and its time added up BEFORE anything lists or runs the settings.
        More settings than FI lists: the quest's plain stop. Otherwise the scan's time is the larger of the plan's own estimate
        and FI's timing of the scan's settings as a measurement's are timed (a plan's claim alone is not trusted), and the
        search's evaluations, which share the scan's time, are added at what one new design (the search's baseline) costs;
        over ``execution.timeout_s``: the same stop, as for a measurement (FI does not run a smaller scan than the plan's),
        with what fits for the plan. ``None`` to go on. Before that stop, the simulation is made faster with the same
        numbers as a measurement's is (:meth:`_speed_up`, once; ``speed_done``), on the scan's settings and a few designs of
        the search's own space, and timed again.

        Not skipped for a frozen protocol or a background job: the search runs here either way and sizing never changes the
        protocol; the timing is bounded by ``engine.pilot_timeout_frac`` of the time allowed, and a resume repeats it."""
        from . import optimisation_plan as _optimisation_plan
        from . import optimise as _optimise
        from . import optimise_search as _optimise_search

        self._clear_cut(runner)  # what an earlier sizing of this study decided is decided again below
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
        thresholds = (self._protocol_block(state) or {}).get("thresholds")  # type: ignore[attr-defined]
        # The scan's settings are timed as a measurement's are (core/run_estimate.py::measure): the base twice, each setting
        # at its ends (and quarter points), every part of the scan, so a cost that depends on the settings, or is paid once
        # per new value and reused, is counted as the scan will pay it; one evaluation of the baseline alone was not.
        est: _estimate.Estimate | None = None
        first_s = 0.0  # what the scan's first timed setting took the first time it ran
        probes: list[Any] | None = None
        at_once = int(getattr(runner, "at_once", 1) or 1)
        slowdown: float | None = None  # how much slower a setting runs beside ``at_once - 1`` others, once measured
        timing_started = time.monotonic()
        budget_s = max(limit_s * float(self.config.engine.pilot_timeout_frac), 30.0)  # type: ignore[attr-defined]
        try:
            grid = _optimise_search.scan_grid(block)
            inside = _optimise_search.in_each_run(block)
        except (KeyError, TypeError, ValueError):
            grid, inside = None, {}
        if grid is not None:
            probes, why = await _estimate.measure(
                self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
                grid, runs=runs, deterministic=deterministic, limit_s=limit_s, budget_s=budget_s, env=env,
                thresholds=thresholds, in_each_run=inside or None)
            if probes is None:
                # A setting that fails when timed is a setting the simulation fails at, as for a measurement: the same repair
                # as a failed run, before the protocol is frozen. Going on, the scan was sized from the plan's claim alone.
                self._log.warning("[execute] the simulation failed while FI timed the search's coarse scan %s", why)  # type: ignore[attr-defined]
                return self._sizing_failed(why)
            else:
                est = _estimate.estimate(grid, runs, deterministic, probes)
                if probes:
                    first_s = float(probes[0].first_s or probes[0].seconds)
                # The scan runs ``execution.settings_at_once`` settings at the same time (core/optimise.py::run_search), so
                # it is counted that way, with how much slower a setting runs beside the others on this machine, as a
                # measurement is (_size_the_run); one after another it was counted many times too long.
                if at_once > 1 and probes and _estimate.known_too_long(est, limit_s):
                    spent = time.monotonic() - timing_started
                    slowdown = await self._slowdown_at_once(runner, python, env, probes, runs, thresholds or {},
                                                            inside or None, at_once, max(0.0, budget_s - spent),
                                                            deterministic=deterministic)
                    if slowdown is not None:
                        est = _estimate.estimate(grid, runs, deterministic, probes, at_once=at_once, slowdown=slowdown)
        if per_setting is not None and (est is None or per_setting * scan > est.seconds):
            est = _estimate.Estimate(seconds=per_setting * scan, at_least=False, complete=True, cells=scan, trials=scan * runs)
        if est is None:
            return None
        searched = 0
        core_est: Any = None  # the scan and the search without the check, once the check was added
        check_s = 0.0
        checked = 0  # the evaluations of FI's check at finer settings, counted when they make the study too long
        whole: Any = None  # the scan with the search's evaluations, when one evaluation was timed
        alternatives_note = ""  # how many alternatives the search's evaluations are for, when the plan compares several
        alternatives = 1
        searching = 1  # how many of them the run searches
        per_alternative = 0
        probe_n = 0
        per_eval: float | None = None  # what one new design costs, once timed
        lower = False
        if not _estimate.known_too_long(est, limit_s):
            # The search shares the scan's time (one deadline for both, core/optimise.py): every evaluation is a design
            # the scan did not compute, so it costs what a setting costs the first time it runs: the search's own
            # baseline timed once here, never less than the scan's first timed setting took the first time (the baseline
            # may run on what the scan's timing computed) nor than the plan's own claim. Each evaluation is a whole
            # setting with its runs.
            search_n = int(_optimisation_plan.evaluations(block)[0] or 0)
            # A study that compares named alternatives searches each one with the whole budget (the alternatives are
            # counted from the plan's own scan, core/optimise_search.py::compared_levels), and asks a few designs of every
            # alternative after the first before it is searched (compare): evaluations outside the budget, counted here.
            alternatives = searching = _optimise_search.level_count(block)
            per_alternative = search_n // alternatives if search_n else 0
            probe_n = (alternatives - 1) * _optimise_search.PROBE_DESIGNS if search_n and alternatives > 1 else 0
            per_eval = max(per_setting or 0.0, first_s) or None
            left = budget_s - (time.monotonic() - timing_started)
            if search_n and left >= 5:
                baseline = _optimise.complete_case(self._protocol_block(state), {}, keep_inside=True)  # type: ignore[attr-defined]
                allowance = max(60.0, 3.0 * limit_s / max(1, (scan + search_n) * runs))
                probe, why = await _estimate._time_one(
                    self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
                    baseline, runs=runs, deterministic=deterministic, timeout_s=int(max(5, min(allowance, left))),
                    env=env, thresholds=thresholds, in_each_run=inside or None)
                if probe is None:
                    # The baseline is the first design the search evaluates: one it fails at is the same repair.
                    self._log.warning("[execute] the simulation failed while FI timed the search's baseline %s", why)  # type: ignore[attr-defined]
                    return self._sizing_failed(why)
                if per_eval is None or probe.seconds > per_eval:
                    per_eval, lower = probe.seconds, probe.at_least
            if search_n and per_eval:
                # The search asks for designs one at a time or several together (its starting points side by side, a
                # population, a simplex's corners), as the search itself shows when it is rehearsed (_search_rounds);
                # several together run ``at_once`` at a time, each as much slower as FI measured. Counted one after
                # another, a search that runs 8 at a time was counted several times too long.
                # One search is rehearsed (all alternatives search alike); what is counted is its rounds once for each
                # alternative the run searches, and the probes of every alternative after the first, asked together.
                alone1, batches1, together1 = await asyncio.to_thread(_search_rounds, block, per_alternative, at_once)

                def priced(searched_n: int) -> tuple[int, int, int, int]:
                    a, b, t = alone1 * searched_n, batches1 * searched_n, together1 * searched_n
                    if probe_n and at_once > 1:
                        b += (alternatives - 1) * math.ceil(_optimise_search.PROBE_DESIGNS / at_once)
                        t += probe_n
                    elif probe_n:
                        a += probe_n
                    return a, b, t, per_alternative * searched_n + probe_n

                # Every alternative is searched, until the simulation itself shows otherwise (below).
                alone, batches, together_n, search_n = priced(searching)
                if batches and slowdown is None and probes and _estimate.known_too_long(
                        _replace(est, seconds=est.seconds + per_eval * search_n), limit_s):
                    spent = time.monotonic() - timing_started
                    slowdown = await self._slowdown_at_once(runner, python, env, probes, runs, thresholds or {},
                                                            inside or None, at_once, max(0.0, budget_s - spent),
                                                            deterministic=deterministic)
                    if slowdown is not None and grid is not None:
                        est = _estimate.estimate(grid, runs, deterministic, probes, at_once=at_once,
                                                 slowdown=slowdown) or est

                def whole_study(a: int, b: int, t: int, n: int) -> _estimate.Estimate:
                    search_s = per_eval * (a + (b * slowdown if slowdown is not None else t))
                    return _estimate.Estimate(seconds=est.seconds + search_s, at_least=est.at_least or lower,
                                              complete=est.complete, cells=scan + n, trials=(scan + n) * runs,
                                              once_s=est.once_s, once_axes=list(est.once_axes))

                together = whole_study(alone, batches, together_n, search_n)
                if alternatives > 1 and _estimate.known_too_long(together, limit_s):
                    # Too long when every alternative is searched: the run searches only those that give results
                    # different from the first one's (compare), so FI asks the simulation, as the run will, before it
                    # says the study does not fit.
                    left = budget_s - (time.monotonic() - timing_started)
                    searching = await self._alternatives_that_differ(
                        state, block, runner, python, env, deterministic=deterministic, at_once=at_once,
                        thresholds=thresholds, left_s=left) or alternatives
                    if searching < alternatives:
                        alone, batches, together_n, search_n = priced(searching)
                        together = whole_study(alone, batches, together_n, search_n)
                if alternatives > 1:
                    alternatives_note = (
                        f" ({alternatives} alternatives the plan compares"
                        + (f", {searching} of them searched (the others give the first one's results)"
                           if searching < alternatives else ", each searched")
                        + f" with {per_alternative:,} evaluations, and {probe_n:,} evaluations that tell whether an "
                        "alternative gives the same results as the first)")
                    if searching < alternatives:
                        self._log.info(  # type: ignore[attr-defined]
                            "[execute] FI asked the simulation for %d designs at each of the %d alternatives the plan "
                            "compares; the others give the first one's results, so the search is counted for %d of "
                            "them, not %d", _optimise_search.PROBE_DESIGNS, alternatives, searching, alternatives)
                whole = together
                if _estimate.known_too_long(together, limit_s):
                    est, searched = together, search_n
            # FI's check of the best design at finer settings runs after the search and is held to the same time limit
            # (core/optimise.py): its evaluations are counted at what one design costs here, at the least (a finer setting
            # costs more, so this is a lower bound; a check of 72 runs at ~75 minutes each was not counted and ran far past
            # the limit).
            try:
                check_n = int(_optimisation_plan.budget(block).get("check") or 0)
            except (KeyError, TypeError, ValueError):
                check_n = 0
            unit = per_eval or first_s or None
            if check_n and unit:
                core_est = whole if whole is not None else est
                # The check runs its designs ``at_once`` at a time like the search, each as much slower as FI measured.
                share = (slowdown if slowdown is not None else 1.0) / at_once if at_once > 1 else 1.0
                check_s = unit / runs * check_n * share
                with_check = _replace(core_est, seconds=core_est.seconds + check_s, at_least=True)
                whole = with_check
                if _estimate.known_too_long(with_check, limit_s):
                    est, checked = with_check, check_n
                    if not searched and search_n:
                        searched = search_n
                    elif not searched:
                        searched = 0
            if not searched and not checked:
                # The line a person reads is the whole study: the scan alone said "about 6 minutes" of a search that ran
                # for hours.
                if whole is not None:
                    self._log.info("[execute] the search's coarse scan and its %s evaluations: %s (the coarse scan alone: "  # type: ignore[attr-defined]
                                   "%s)", f"{search_n:,}", _estimate.says(whole, limit_s),
                                   _estimate.plain_duration(est.seconds * _estimate.SAFETY))
                else:
                    self._log.info("[execute] the search's coarse scan: %s", _estimate.says(est, limit_s))  # type: ignore[attr-defined]
                return None
        if not speed_done and grid is not None and int(self.config.engine.improve_rounds or 0) > 0:  # type: ignore[attr-defined]
            # What FI runs is the same simulation, whatever the study is called: the speed-up a measurement gets is tried
            # here too, on the scan's settings and a few designs the search itself would try.
            sped = await self._speed_up(
                state, runner, python, env, grid, runs, thresholds or {}, inside or {}, est, limit_s,
                budget_s=max(0.0, budget_s - (time.monotonic() - timing_started)), probes=probes,
                deterministic=deterministic, extra_cells=_search_sample(block, grid))
            if sped.get("kept"):
                return await self._size_the_scan(state, runner, python, env, speed_done=True)
        text = self._too_long_text(est, limit_s, cut=False)
        if searched or checked:
            text = (f"the coarse scan's {scan:,} settings and the search's {searched:,} evaluations{alternatives_note}"
                    + (f" and FI's check of the best design at finer settings ({checked:,} runs)" if checked else "")
                    + f" would take "
                    f"{'at least ' if est.at_least else 'about '}{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}"
                    f", more than the {_estimate.plain_duration(limit_s)} allowed (one new design takes about "
                    f"{'at least ' if lower else ''}{_estimate.plain_duration(per_eval or 0.0)}); FI does not run a smaller "
                    "study than the plan's")
        # The check's time comes out of the room for settings first: it is not a cost of each setting.
        fits_in = (self._what_fits(core_est, max(1.0, limit_s - check_s * _estimate.SAFETY), search=True)
                   if checked and core_est is not None else self._what_fits(est, limit_s, search=True))
        if fits_in is not None:
            # FI's own check of the best design at finer settings runs after the search and is not in its budget: the plan
            # is told what it costs, so the scan and the search are sized to the time left.
            try:
                fits_in["check"] = int(_optimisation_plan.budget(block).get("check") or 0)
            except (KeyError, TypeError, ValueError):
                pass
            if alternatives > 1:
                # Where the search's evaluations come from, in FI's own numbers: the plan lowers them by comparing fewer
                # alternatives or by giving each a smaller budget.
                first_level = _optimise_search.compared_levels(block)[0]
                fits_in["alternatives"] = alternatives
                fits_in["alternatives_searched"] = searching
                fits_in["alternative_conditions"] = [str(k) for k in first_level]
                fits_in["per_alternative"] = per_alternative
                fits_in["probes"] = probe_n
        # Not stopped while at least one design fits: the scan runs in the plan's order, then the search, then FI's check (the
        # time the check needs is kept for it), up to the time limit, and the paper says what ran and what did not.
        try:
            planned_search = int(_optimisation_plan.evaluations(block)[0] or 0)
            planned_check = int(_optimisation_plan.budget(block).get("check") or 0)
        except (KeyError, TypeError, ValueError):
            planned_search = planned_check = 0
        one_s = per_eval or first_s or _estimate.cheapest_setting_s(probes)
        if self._run_up_to_the_limit(
                runner, est, limit_s, one_s, kind="search", reserve_s=check_s * _estimate.SAFETY,
                planned={"scan": scan, "evaluations": planned_search, "check_runs": planned_check}, detail=text,
                fit=fits_in):
            return None
        self._log.warning("[execute] the search's coarse scan: %s", text)  # type: ignore[attr-defined]
        return self._stop_too_long(text, fit=fits_in)

    async def _alternatives_that_differ(self, state: Any, block: dict[str, Any], runner: Any, python: Any, env: Any, *,
                                         deterministic: bool, at_once: int, thresholds: Any, left_s: float) -> int | None:
        """How many of the alternatives a study compares the run would search, by asking the simulation what the run asks
        (:func:`core.optimise_search.compare`): the same few designs (:func:`core.optimise_search.sizing_designs`) at every
        alternative, one of them once more at the first to see which results change by themselves, and the same rule for
        "gives the first one's results". ``None`` when it could not be asked in the time left or something did not answer
        (every alternative is then counted as searched)."""
        from time import monotonic as _clock  # the clock core/optimise.py judges its deadline by

        from . import optimise as _optimise

        if left_s < 15:
            return None
        levels = _optimise_search.compared_levels(block)
        try:
            designs = _optimise_search.sizing_designs(block)
            names = _optimise_search.compared_names(block, _optimise.watched_results(self._protocol_block(state) or {}))  # type: ignore[attr-defined]
            cells = [_optimise_search.cell_of(_optimise_search.level_block(block, lv), d) for lv in levels for d in designs]
        except (KeyError, TypeError, ValueError):
            return None
        cells.append(dict(cells[0]))  # the first alternative's first design again
        try:
            got = await _optimise.evaluate_many(
                self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
                "run_cell" if deterministic else "run_trial", cells, at_once=max(1, at_once), runs=1, base_seed=0,
                deadline=_clock() + left_s, env=env, thresholds=thresholds,
                in_each_run=_optimise_search.in_each_run(block) or None)
        except Exception as exc:  # noqa: BLE001 -- the count stays the plan's whole count
            self._log.warning("[execute] FI could not ask the simulation which alternatives give different results (%r); "  # type: ignore[attr-defined]
                              "every alternative is counted as searched", exc)
            return None
        values = [a.get("values") if isinstance(a, dict) and isinstance(a.get("values"), dict) else None
                  for a, _err, _loaded in got]
        n = len(designs)
        replies = [values[i * n:(i + 1) * n] for i in range(len(levels))]
        return _optimise_search.alternatives_searched(block, levels, replies, values[-1], names)

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
                in_each_run=_in_run.of_study(protocol))
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
        at_once = f" with {est.at_once} running at the same time" if est.at_once > 1 else ""
        return (f"the experiment's {est.cells:,} settings would take {'at least ' if est.at_least else 'about '}"
                f"{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}{at_once}, more than the "
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

        # A search for the best design runs on its coarse scan's settings, with the search's baseline design and every
        # value it compares inside each run, as the search calls the simulation: a fault there would otherwise be met
        # first by the known-answer checks, on their own few repairs.
        block = _optimise.block_of(protocol)
        search = block is not None or _optim.has_block(state.get("design"))
        if search and block is None:
            return None
        self._small_run_ctx = (state, py, exec_env)
        if search:
            try:
                grid = _optimise_search.scan_grid(block)
            except (KeyError, TypeError, ValueError):
                return None  # a block the scan cannot be built from is reported by the run
        else:
            grid = _optimise_search.study_grid(protocol) if isinstance(protocol.get("grid"), dict) else {}  # knows its sub-studies
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
        deterministic = "run_trial" not in set(getattr(self, "_trial_entries", set()) or ())
        inside = (_optimise_search.in_each_run(block) or None if search
                  else protocol.get("in_each_run") if isinstance(protocol.get("in_each_run"), dict) else None)
        key = _small.key_of(simulate, grid, deterministic, thresholds, inside)
        record = _small.read_record(self.quest_root)  # type: ignore[attr-defined]
        if record.get("key") == key:
            self._small_run_note = str(record.get("for_reader") or "")
            return record
        paired = any(isinstance(m, dict) and m.get("paired") for m in protocol.get("metrics") or [])
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
        if record and record.get("crashed"):
            return str(record.get("says") or "it failed")
        # The repaired code must also still give the study's results from the function FI calls for the study.
        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
        try:
            return await self._simulation_repair_fault(ctx[0], simulate.read_text(encoding="utf-8")) if simulate.is_file() else ""  # type: ignore[attr-defined]
        except OSError:
            return ""

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

    async def _speed_up(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """:meth:`_speed_up_rounds`, which commits each version it keeps with its own words (code_project.CodeRuns)."""
        with self._hold_code_runs():  # type: ignore[attr-defined]
            return await self._speed_up_rounds(*args, **kwargs)

    async def _speed_up_rounds(self, state: Any, runner: Any, python: Any, env: Any, grid: dict[str, list[Any]], runs: int,
                               thresholds: dict[str, Any], inside: dict[str, list[Any]], est: Any, limit_s: float,
                               budget_s: float = 600.0, probes: list[Any] | None = None,
                               deterministic: bool | None = None,
                               extra_cells: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """The study would not fit the time allowed: up to ``engine.improve_rounds`` edits that make the simulation itself
        faster without changing a number (core/improve_speed.py). FI runs a fixed sample of trials, the checks' own cases
        included, before and after each edit; an edit is kept only when every number is the same and the sample takes
        clearly less time, and it passes every guard the correctness loop has. The study is then timed twice and the
        slower timing stands (``probes``), so the noise of one timing cannot make it fit. ``{"kept": bool, "rounds": [...],
        "calls": n, "speedup": x, "probes": [...] | None}``. ``deterministic``: whether a setting is one call with no
        randomness (default: the runner's own); ``extra_cells``: settings the sample holds the change to as well (a
        search's designs outside its coarse scan)."""
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
        det = bool(runner.deterministic if deterministic is None else deterministic)
        cells = _speed.sample_cells(grid, oracles)
        cells += [dict(c) for c in extra_cells or [] if c not in cells]
        for cell in cells:
            for key, value in cell.items():
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    settings.setdefault(str(key), []).append(value)
        timeout_s = max(60, int(self.config.execution.timeout_s))  # type: ignore[attr-defined]
        timeouts: dict[str, int] | None = None
        if probes is not None:
            # The sample is held to what the timing measured: a setting it did not see finish is left out, each gets a
            # few times its measured cost, and every sample of the step fits in the time left for timing.
            cells, timeouts, why = _speed.bounded_sample(cells, probes, runs=runs, deterministic=det,
                                                         budget_s=budget_s, samples=2 + limit)
            if not cells:
                log.info("[improve] The simulation is not sped up: %s.", why)
                return {"goal": "speed", "limit_s": limit_s, "rounds": [], "kept": False, "calls": 0, "probes": None,
                        "stopped": why}
            if why:
                log.info("[improve] %s; a speed-up is not tried on them.", why[0].upper() + why[1:])

        async def sample() -> Any:
            return await _speed.run_sample(self.executor, python, root, module, cells,  # type: ignore[attr-defined]
                                           deterministic=det, thresholds=thresholds, in_each_run=inside,
                                           env=env, timeout_s=timeout_s, log=log, timeouts=timeouts)

        async def time_study() -> list[Any] | None:
            probes, _why = await _estimate.measure(
                self.executor, python, root, module, grid, runs=runs, deterministic=det,  # type: ignore[attr-defined]
                limit_s=limit_s, budget_s=budget_s, env=env, thresholds=thresholds, cached=None, in_each_run=inside)
            return probes

        padded = est.seconds * _estimate.SAFETY
        record: dict[str, Any] = {"goal": "speed", "limit_s": limit_s, "estimate_s": padded, "rounds": [], "kept": False,
                                  "calls": 0, "probes": None}
        def stop(why: str) -> dict[str, Any]:
            # Every way the step ends without trying a change says why, in run.log and the record.
            record["stopped"] = why
            log.info("[improve] The simulation is not sped up: %s.", why)
            self._save_speed_record(record)
            return record

        first = await sample()
        if first.failed or first.seconds <= 0:
            return stop(f"the sample did not run ({first.failed or 'no time measured'})")
        # The unchanged code is run a second time: a number that is not the same in both runs is not fixed by the code (a
        # measure of its own run time, say), so it cannot show that an edit changed a result and is left out of every comparison.
        again = await sample()
        if again.failed:
            return stop(f"the sample did not run a second time ({again.failed})")
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
                return stop("every number of the sample differs from one run of the unchanged code to the next, so an "
                            "edit cannot be judged")
        needed = padded / max(limit_s, 1e-9)
        target = needed * _speed.MARGIN
        original = _improve.snapshot(root)
        best = {"files": original, "sample": first}
        tried = {_improve.fingerprint(original)}
        # Where the time goes, measured on the costliest setting the timing saw (the base when none was timed).
        timed = {_trial_runner.cell_key(p.cell): float(p.seconds) for p in probes or [] if getattr(p, "cell", None)}
        costliest = max(cells, key=lambda c: timed.get(_trial_runner.cell_key(c), 0.0))
        profile = await _speed.profile_text(self.executor, python, root, module, costliest,  # type: ignore[attr-defined]
                                            deterministic=det, thresholds=thresholds, in_each_run=inside, env=env,
                                            timeout_s=max(60, int(min(timeout_s, budget_s or timeout_s))), log=log)
        record["profiled"] = bool(profile)
        block = _speed.speed_block(padded, limit_s, first, cells, runs, profile=profile)
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

            edit, why = _improve.parse_edit(_parse_json_lenient(reply, node="improve", want=_improve.EDIT_KEYS))
            new_text = None
            if edit is not None:
                new_text, why = _improve.check_edit(files, edit, allowed, tried=tried, expected=expected,
                                                    measures=measures, settings=settings)
            if edit is None or new_text is None:
                # The edit is kept with its refusal, so whether a guard refused it rightly can be judged afterwards.
                entry.update(outcome="refused", reason=why, why=edit.why if edit else "",
                             **({"file": edit.file, "diff": _improve.diff_text(edit)} if edit is not None
                                else {"reply": str(reply or "")[:8000]}))
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
                e = _estimate.estimate(grid, runs, det, probes)
                if e is not None:
                    runs_timed.append((e.seconds, probes))
            if len(runs_timed) == 2:
                slower = max(runs_timed, key=lambda t: t[0])
                record["probes"] = slower[1]
                record["estimate_after_s"] = [round(t[0] * _estimate.SAFETY, 1) for t in runs_timed]
            # For a search this timing is of its coarse scan only: the whole study, with the search's evaluations, is
            # timed again right after (_size_the_scan), and says its own estimate.
            what = "the search's coarse scan" if extra_cells else "the study"
            log.info("[improve] The simulation is %.1f times faster with the same numbers on the sample%s.", record["speedup"],
                     f"; timed twice, {what} would now take about "
                     f"{_estimate.plain_duration(max(record['estimate_after_s']))}"
                     if record.get("estimate_after_s") else "")
        else:
            log.info("[improve] No change made the simulation clearly faster with the same numbers; it is left as it was.")
        self._save_speed_record(record)
        return record

    def _save_speed_record(self, record: dict[str, Any]) -> None:
        root = self.quest_root  # type: ignore[attr-defined]
        try:
            (root / ".fi").mkdir(parents=True, exist_ok=True)
            (root / ".fi" / "improve_speed.json").write_text(
                json.dumps({k: v for k, v in record.items() if k != "probes"}, indent=1, default=str), encoding="utf-8")
        except OSError:
            pass

    def _clear_cut(self, runner: Any) -> None:
        """A search is sized again at every run: the cut an earlier sizing recorded (and the time it kept for FI's check) is
        taken away first, so a study that now fits is never described as cut."""
        runner.check_reserve_s = 0.0
        runner.run_id = ""
        if hasattr(runner, "first_cells"):
            runner.first_cells = None
        record = _estimate.read_record(self.quest_root)  # type: ignore[attr-defined]
        if record.pop("truncated", None) is not None:
            _estimate.write_record(self.quest_root, record)  # type: ignore[attr-defined]

    def _run_up_to_the_limit(self, runner: Any, est: _estimate.Estimate, limit_s: float, one_s: float | None, *, kind: str,
                             base: dict[str, Any] | None = None, planned: dict[str, Any] | None = None,
                             reserve_s: float = 0.0, detail: str = "", fit: dict[str, Any] | None = None,
                             first_cells: int | None = None) -> bool:
        """A study FI's own timing says does not fit the time allowed is run anyway, in the plan's order, up to the limit,
        unless not even one setting (or one design) fits: ``True`` to go on. FI does not choose which settings matter, so
        the grid runs in the plan's order (not costliest first), and a search runs its baseline, its coarse scan, its
        search and then FI's check, as it always does; the check keeps the time it needs (``reserve_s``, never more than
        half the limit) so the scan and the search cannot take all of it. The cut is kept in ``.fi/run_estimate.json``
        (``truncated``); the paper and the quality report say what ran from FI's records (:func:`core.disclosure.cut_short_text`)."""
        if not _estimate.one_fits(one_s, limit_s):
            return False
        reserve = min(max(0.0, reserve_s), limit_s / 2.0)
        cut = {"kind": kind, "estimate_s": est.seconds * _estimate.SAFETY, "at_least": bool(est.at_least),
               "limit_s": float(limit_s), "settings": int(est.cells), "one_s": one_s, "order": "plan",
               "run_id": uuid.uuid4().hex[:12], "check_reserve_s": round(reserve, 1), **(planned or {}), "detail": detail.split("; FI does not run")[0],
               **({"planned_settings": int(first_cells)} if first_cells else {}), **({"fit": fit} if fit else {})}
        what = "the search for the best design" if kind == "search" else "the experiment"
        line = (f"The plan has {int(est.cells):,} settings, more than the {_optimise_search.MAX_LISTED_SETTINGS:,} FI lists at most; FI estimates they "
                f"would take {'at least ' if est.at_least else 'about '}{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}"
                f" (the limit is {_estimate.plain_duration(limit_s)}); it will plan the first {first_cells:,} in the plan's order "
                "and run them until the time limit, and the paper will say what was and was not done") if first_cells else (
                f"FI estimates {what} would take {'at least ' if est.at_least else 'about '}"
                f"{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}, more than the "
                f"{_estimate.plain_duration(limit_s)} allowed; it will run the plan's settings in the plan's order until "
                "the time limit, and the paper will say what was and was not done")
        self._log.warning("[execute] %s", line)  # type: ignore[attr-defined]
        print(f"[FI] {line}")
        runner.run_id = cut["run_id"]
        if first_cells:
            runner.first_cells = int(first_cells)
        if kind == "grid":
            runner.cost = None
        else:
            runner.check_reserve_s = reserve
        record = {k: v for k, v in (base or {}).items() if k not in ("stopped", "fit")}
        record.setdefault("limit_s", float(limit_s))
        record.update(fits=False, truncated=cut, says=line)
        _estimate.write_record(self.quest_root, record)  # type: ignore[attr-defined]
        return True

    def _stop_too_long(self, text: str, fit: dict[str, Any] | None = None) -> dict[str, Any]:
        """The quest's plain stop: nothing was run for the study, it would not fit, and no paper is written. ``fit``
        (:meth:`_what_fits`): what FI measured, which lets the evidence gate ask the plan once for a study that fits."""
        out = self._stop_before_run(text, "too_long")
        if fit:
            out["exec_result"]["too_long_fit"] = dict(fit)
        return out

    @staticmethod
    def _what_fits(est: _estimate.Estimate, limit_s: float, *, search: bool = False) -> dict[str, Any] | None:
        """What FI measured about a study too long to run: the seconds one setting takes (as the run would take them,
        with the settings that run at the same time), how many settings fit the time allowed with FI's margin, and the
        settings the plan has. ``None`` when nothing was measured."""
        if est.cells <= 0 or est.seconds <= 0:
            return None
        # What is paid once per new value is not a cost of every setting: it is named apart, so the plan can take fewer of
        # those values, and the settings that fit are counted from what is left of the time.
        once = min(float(est.once_s) * max(1.0, float(est.slowdown or 1.0)), est.seconds * 0.999)
        per = (est.seconds - once) / est.cells
        room = max(per, limit_s / _estimate.SAFETY - once)
        out = {"seconds_per_setting": round(per, 3), "at_least": bool(est.at_least), "settings": int(est.cells),
               "settings_that_fit": max(1, int(room / per)), "limit_s": float(limit_s), "search": bool(search)}
        if once > 0:
            out.update(once_s=round(once, 1), once_axes=list(est.once_axes))
        return out

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
