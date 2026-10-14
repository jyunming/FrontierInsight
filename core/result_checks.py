"""Cheap checks made on a run's results before the paper, as methods the engine mixes in (no model reads anything here).

After the run has finished and FI's own checks of it passed, the results are compared with what ran:

* what the analysis reports for each setting must be what ran at that setting (:mod:`core.reported_settings`);
* the figures and the reported quantities must show something: no empty panel, no series at one x, no quantity that is
  the same at every setting or stuck at a bound (:mod:`core.degenerate_results`);
* a quantity the plan names that a second model found the code computes differently, and that the repairs did not change,
  is not reported under the plan's name: it is left out of the results and the paper says so
  (:meth:`ResultChecksMixin._code_reader_findings`; nothing is sent back for it, the repairs were already spent).

A fault is sent back like a failed run, in plain words and with no expected value in the request: the analysis is repaired
(or the simulation, when the fault is in what it returns), and the study's trials are not run again unless the simulation
changed. At most :data:`REPAIRS` times per design iteration (and never more than the repairs the quest has left); what is still
there afterwards is left out of the results the paper may use, and the paper says so in its limitations. Nobody is asked to
edit code, and the quest does not stop for it.

They live here and not in ``engine.py`` so the node only calls :meth:`ResultChecksMixin._results_match_run`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import code_review as _code_review
from . import degenerate_results as _degenerate
from . import grid_keys as _grid_keys
from . import reported_settings as _reported

#: Why a quantity a failing test showed wrong is left out, as the must-fix hit for a paper that quotes it says it.
SHOWN_FAULT_WHY = "a test FI ran on the code that produced the results shows the code computes it wrongly"

#: How many times, in all, the results of one design iteration are sent back for the checks of this module.
REPAIRS = 2
_COUNTER = "result_checks.json"
RECORD = "RESULT_CHECK.json"
ANALYSIS = "analysis"
SIMULATION = "simulation"


@dataclass
class Found:
    """What one check found in the results."""

    kind: str
    fix: str  # ANALYSIS or SIMULATION: the script a repair is sent to
    directive: str
    note: str
    say: str
    keys: list[str] = field(default_factory=list)  # to take out of the results
    figures: list[str] = field(default_factory=list)  # to take out of the figures
    quantity: str = ""  # the plan's name of the quantity the keys belong to, when the check is about one


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _values_left_out(result_json: Any, keys: set[str]) -> dict[str, list[float]]:
    """``{key: its numbers}`` for each left-out key of ``result_json`` (at any depth), without a number that a quantity that
    stays also holds: the paper is checked for quoting the first, and a value both hold is not one that can be told apart."""
    out: dict[str, list[float]] = {}
    kept: set[float] = set()

    def walk(node: Any, inside: str | None) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, inside or (str(k) if str(k) in keys else None))
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v, inside)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if inside is not None:
                out.setdefault(inside, []).append(float(node))
            else:
                kept.add(float(node))

    walk(result_json, None)
    return {k: sorted({v for v in vals if v not in kept}) for k, vals in out.items() if any(v not in kept for v in vals)}


class ResultChecksMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    def _result_findings(self, state: Any, result_json: Any) -> list[Found]:
        """What the checks of this module find in the run's results. A check that cannot be made finds nothing."""
        found: list[Found] = []
        cells = _reported.read_record(self.quest_root)  # type: ignore[attr-defined]
        try:
            mismatches = _reported.problems(cells, _reported.reported_of(self.quest_root, result_json))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 -- a check that cannot be made flags nothing
            self._log.warning("[execute] could not compare the reported settings with the run (%r)", exc)  # type: ignore[attr-defined]
            mismatches = []
        if mismatches:
            found.append(Found("reported_settings", ANALYSIS, _reported.directive(mismatches, cells),
                               _reported.plain_note(mismatches), _reported.writer_note(mismatches),
                               keys=_reported.keys_to_leave_out(mismatches)))
        try:
            found.extend(self._degenerate_findings(state, result_json, cells))
        except Exception as exc:  # noqa: BLE001 -- a check that cannot be made flags nothing
            self._log.warning("[execute] could not check the run's results for empty figures and constants (%r)", exc)  # type: ignore[attr-defined]
        return found

    def _saved_figures(self) -> list[str]:
        from . import engine as _e

        folder = self.quest_root / "figures"  # type: ignore[attr-defined]
        return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _e._FIGURE_SUFFIXES) if folder.is_dir() else []

    def _no_figures_standing(self, state: Any) -> list[Found]:
        try:
            return self._no_figures_findings(state)
        except Exception as exc:  # noqa: BLE001 -- a check that cannot be made flags nothing
            self._log.warning("[execute] could not check whether the planned figures were drawn (%r)", exc)  # type: ignore[attr-defined]
            return []

    def _no_figures_findings(self, state: Any) -> list[Found]:
        """The plan has figures and the run, which finished, saved none: nothing was drawn, so the paper has no figure. Not
        sent back (the results stand); the writer is told, and the limitations say so."""
        planned = (state.get("design") or {}).get("figures_planned")
        names = [Path(str(f)).name for f in planned if str(f).strip()] if isinstance(planned, list) else []
        if not names or self._saved_figures():
            return []
        listed = ", ".join(f"`{n}`" for n in names[:6])
        directive = (f"FI looked at what the run produced: the plan has figures ({listed}) and the run saved none. Draw each from "
                     "the saved results and save it to the figures folder. If there is nothing valid to draw because the results "
                     "are not valid, find out why and fix that first. Change nothing else.")
        note = f"the plan has figures ({listed}) and the run saved none"
        say = (f"{note[0].upper()}{note[1:]}, so the paper has no figure. Do not describe, cite or caption a figure, and do not "
               "write \"as shown in Figure N\": say plainly, in the limitations, that no figure was drawn.")
        return [Found("no_figures", ANALYSIS, directive, note, say)]

    def _figure_gap_findings(self, dropped: set[str]) -> list[Found]:
        """Where a figure that stays has no point at a setting (:func:`core.degenerate_results.figure_gaps`): not repaired and
        nothing is left out, the writer is told which settings the figure does not show."""
        try:
            gaps = [g for g in _degenerate.figure_gaps(self.fi_dir / "figure_records", self._saved_figures())  # type: ignore[attr-defined]
                    if g.figure not in dropped]
        except Exception as exc:  # noqa: BLE001 -- a check that cannot be made flags nothing
            self._log.warning("[execute] could not check the figures for settings they do not show (%r)", exc)  # type: ignore[attr-defined]
            return []
        if not gaps:
            return []
        return [Found("figure_gaps", ANALYSIS, "", "a figure has no point at some settings (" + "; ".join(g.says() for g in gaps[:3]) + ")",
                      _degenerate.gap_note(gaps))]

    def _code_reader_findings(self, result_json: Any) -> list[Found]:
        """The quantities the plan names that the code computes differently (a second model's finding on the version that ran,
        left standing by the repairs) and that the run's results hold under a name that begins with the plan's. They are left
        out: the paper may not present a number under a definition the code does not compute. A finding about a quantity the
        results do not hold has nothing to leave out (the writer's note about requirements not met covers it)."""
        try:
            differing = _code_review.differing_quantities(self.quest_root, self.quest_root / "code")  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 -- a check that cannot be made flags nothing
            self._log.warning("[execute] could not read the code reader's findings against the results (%r)", exc)  # type: ignore[attr-defined]
            return []
        found: list[Found] = []
        for q in differing:
            # Only "computed differently": a quantity called missing that the results hold is one the reader got wrong
            # (:func:`core.code_review.resolve_headline` says so for the headline; the writer's note covers the rest).
            if q["status"] != _code_review.DIFFERENT:
                continue
            keys = sorted(_code_review.named_in_results(q["name"], result_json))
            if not keys:
                continue
            what = "computed differently from the plan"
            why = q["note"].rstrip(".") or what
            note = f"the plan's {q['kind']} `{q['name']}` is {what} in the code ({why})"
            say = (f"A second model compared the code with the plan and found that `{q['name']}` is {what} ({why}); the "
                   f"repairs did not remove it. FI took {', '.join('`' + k + '`' for k in keys[:6])} out of the results. "
                   f"Do NOT report, describe or quote any value for `{q['name']}`, and do not call another number by "
                   f"the plan's name for it. Say plainly, in the limitations, that `{q['name']}` is left out of this paper "
                   "because the code does not compute it as the plan defines it.")
            found.append(Found("code_reader", ANALYSIS, "", note, say, keys=keys, quantity=q["name"]))
        return found

    def _degenerate_findings(self, state: Any, result_json: Any, cells: list[dict[str, Any]]) -> list[Found]:
        from . import engine as _e

        reported: set[str] = set()
        for _where, value in _reported.reported_of(self.quest_root, result_json):  # type: ignore[attr-defined]
            reported |= _degenerate.keys_in(value)
        folder = self.quest_root / "figures"  # type: ignore[attr-defined]
        figures = sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _e._FIGURE_SUFFIXES) if folder.is_dir() else []
        drawn = _degenerate.figure_problems(self.fi_dir / "figure_records", figures)  # type: ignore[attr-defined]
        protocol = self._protocol_block(state)  # type: ignore[attr-defined]
        quantities = _degenerate.quantity_problems(
            cells, reported, self._typed_result_exempt(state) | _degenerate.expected_constants(protocol))  # type: ignore[attr-defined]
        marks = _degenerate.accepted_marks(_grid_keys.code_files(self.quest_root / "code"), [q.name for q in quantities],  # type: ignore[attr-defined]
                                           self.quest_root, self.quest_root / "code")  # type: ignore[attr-defined]
        self._result_marked = []
        for q in quantities:
            if q.name in marks:
                mark, how = marks[q.name]
                said = (f"`{q.name}` is {'the same at every setting' if q.kind == 'constant' else 'at a bound for most settings'} "
                        f"by design ({mark.reason}; `{mark.name}`, {mark.script} line {mark.line}")
                self._result_marked.append(said + ("; confirmed by a second model)" if how == _degenerate.CONFIRMED else
                                                   "; declared by the code's own writer and not independently confirmed)"))
        quantities = [q for q in quantities if q.name not in marks]
        if not (drawn or quantities):
            return []
        return [Found("degenerate_results", SIMULATION if quantities else ANALYSIS, _degenerate.directive(drawn, quantities),
                      _degenerate.plain_note(drawn, quantities), _degenerate.writer_note(drawn, quantities),
                      keys=[q.name for q in quantities], figures=sorted({p.figure for p in drawn}))]

    def _drop_figures(self, names: list[str]) -> None:
        """Take exactly the named figure files out of ``figures/``; a figure's record (shared by its formats) goes with the last
        of them."""
        folder = self.quest_root / "figures"  # type: ignore[attr-defined]
        for name in names:
            stem = Path(name).stem
            targets = [folder / Path(name).name]
            others = [p for p in (folder.iterdir() if folder.is_dir() else [])
                      if p.is_file() and p.stem == stem and p.name != Path(name).name and p.suffix.lower() != ".json"]
            if not others:
                targets += [folder / f"{stem}.json", self.fi_dir / "figure_records" / f"{stem}.json",  # type: ignore[attr-defined]
                            self.fi_dir / "figure_records" / f"{stem}.seed0.json"]  # type: ignore[attr-defined]
            for target in targets:
                try:
                    target.unlink(missing_ok=True)
                except OSError as exc:
                    self._log.warning("[execute] could not remove %s (%r)", target.name, exc)  # type: ignore[attr-defined]

    def _result_counter(self, state: Any) -> int:
        record = _read_json(self.fi_dir / _COUNTER)  # type: ignore[attr-defined]
        iteration = int(state.get("iteration", 0) or 0)
        return int(record.get("used") or 0) if record.get("iteration") == iteration else 0

    def _save_result_counter(self, state: Any, used: int) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
            (self.fi_dir / _COUNTER).write_text(  # type: ignore[attr-defined]
                json.dumps({"iteration": int(state.get("iteration", 0) or 0), "used": used}), encoding="utf-8")
        except OSError:
            pass

    async def _results_match_run(self, state: Any, result: Any, failed_script: str | None, split: bool) -> tuple[Any, str | None]:
        """After a run that finished: send the results back like a failed run when they do not match what ran (at most
        :data:`REPAIRS` times, and while the quest has repairs left), else leave out what is still wrong and say so.
        ``(the result to go on with, the script to repair)``."""
        from . import engine as _e
        from . import split_run as _split_run

        record_path = self.quest_root / "needs" / RECORD  # type: ignore[attr-defined]
        record_path.unlink(missing_ok=True)  # an earlier pass's record is not this run's
        if result.returncode != 0:
            return result, failed_script
        result_json = _e._extract_result_json(result.stdout or "")
        self._result_marked = []
        found = self._result_findings(state, result_json) if getattr(self, "_trial_mode", False) else []
        # What a second model found the code computes otherwise cannot be repaired here (its repairs were spent before the run).
        # A run that drew none of the planned figures still has its results: they stand, and the writer is told there is
        # no figure. Sent back like a failed run, a repair that gave up would have thrown the results away and stopped the
        # quest with nothing to write up (a false stop).
        standing = [*self._code_reader_findings(result_json), *self._figure_gap_findings(set()),
                    *self._no_figures_standing(state)]
        if not found and not standing:
            if self._result_marked:
                self._write_marked(record_path)
            return result, failed_script
        used = self._result_counter(state)
        repair_left = (int(state.get("exec_reflect_iter", 0) or 0) < self.config.engine.exec_reflect_max_iterations  # type: ignore[attr-defined]
                       and not state.get("exec_give_up_reason"))
        if found and used < REPAIRS and repair_left:
            self._save_result_counter(state, used + 1)
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the results do not match what ran (%s); sending the %s back (%d of %d)",
                "; ".join(f.note for f in found), "simulation" if any(f.fix == SIMULATION for f in found) else "analysis",
                used + 1, REPAIRS)
            to_fix = (_split_run.SIMULATE_NAME if any(f.fix == SIMULATION for f in found) else _split_run.ANALYSIS_NAME) if split else None
            kept = "\n".join(line for line in (result.stdout or "").splitlines() if not line.startswith("RESULT_JSON:"))
            why = "\n\n".join(f.directive for f in found[:2])
            result = _e.ExecutionResult(1, kept, (result.stderr or "") + "\n[FI] " + why, result.duration_s, result.timed_out)
            self._clear_stale_figures()  # type: ignore[attr-defined]
            return result, to_fix
        if found:  # a figure that is taken out has no gap to report
            standing = [*self._code_reader_findings(result_json),
                        *self._figure_gap_findings({n for f in found for n in f.figures}), *self._no_figures_standing(state)]
        self._leave_out_results([*found, *standing], record_path, result_json)
        return result, failed_script

    def _leave_out_results(self, found: list[Found], record_path: Path, result_json: Any = None) -> None:
        """What is still wrong after the repairs is left out of the results the paper may use: written to
        ``needs/RESULT_CHECK.json`` (read for the writer's note and for the quantities the analysis no longer sees)."""
        self._drop_figures(sorted({n for f in found for n in f.figures}))
        keys = sorted({k for f in found for k in f.keys})
        record = {"removed_quantities": keys, "removed_figures": sorted({n for f in found for n in f.figures}),
                  "removed_values": _values_left_out(result_json, set(keys)),
                  "checks": [{"kind": f.kind, "note": f.note, **({"quantity": f.quantity, "keys": f.keys} if f.quantity else {})}
                             for f in found],
                  "note": "; ".join(f.note for f in found),
                  "say": "\n\n".join([*(f.say for f in found), *self._marked_say()])}
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", RECORD, exc)  # type: ignore[attr-defined]
        self._log.warning(  # type: ignore[attr-defined]
            "[execute] %s. The paper is told and says so in its limitations.", record["note"])

    def _leave_out_shown_faults(self, left_out: list[tuple[str, list[str], str]], not_done: list[tuple[str, str]],
                                result_json: Any) -> None:
        """At the evidence gate: the quantities a failing test showed the code computes wrongly (``(quantity, keys, how)``)
        are left out of the results the paper may use, and the plan's quantities the code does not compute
        (``(quantity, how)``) are named, by adding them to ``needs/RESULT_CHECK.json`` (what the run's checks left out
        stays). The same issues found again add nothing."""
        path = self.quest_root / "needs" / RECORD  # type: ignore[attr-defined]
        record = _read_json(path)
        found: list[Found] = []
        for quantity, keys, how in left_out:
            note = f"a test FI ran on the code that produced the results shows a fault in how it computes `{quantity}` ({how})"
            say = (f"A test FI ran on the code that produced the results fails because of a fault in how `{quantity}` is "
                   f"computed ({how}). FI took {', '.join('`' + k + '`' for k in keys[:6])} out of the results. Do NOT report, "
                   f"describe or quote any value for `{quantity}`, and do not call another number by its name. Say plainly, in "
                   f"the limitations, that `{quantity}` is left out of this paper because a test showed the code computes it "
                   "wrongly.")
            found.append(Found("shown_fault", ANALYSIS, "", note, say, keys=list(keys), quantity=quantity))
        for quantity, how in not_done:
            note = f"the code does not compute the plan's `{quantity}` ({how})"
            say = (f"The code does not compute `{quantity}`, which the plan asks for ({how}). Do not report a value for it or "
                   f"call another number by its name. Say plainly, in the limitations, that `{quantity}` was not computed.")
            found.append(Found("not_computed", ANALYSIS, "", note, say, quantity=quantity))
        checks = [c for c in record.get("checks") or [] if isinstance(c, dict)]
        seen = {(c.get("kind"), c.get("quantity")) for c in checks}
        new = [f for f in found if (f.kind, f.quantity) not in seen]
        if not new:
            return
        keys = sorted({*(str(k) for k in record.get("removed_quantities") or []), *(k for f in new for k in f.keys)})
        why = record.get("removed_why") if isinstance(record.get("removed_why"), dict) else {}
        for f in new:
            for k in f.keys:
                why.setdefault(k, SHOWN_FAULT_WHY)
        record.update({
            "removed_quantities": keys, "removed_figures": list(record.get("removed_figures") or []),
            "removed_values": _values_left_out(result_json, set(keys)), "removed_why": why,
            "checks": [*checks, *({"kind": f.kind, "note": f.note, "quantity": f.quantity, "keys": f.keys} for f in new)],
            "note": "; ".join(x for x in [str(record.get("note") or ""), *(f.note for f in new)] if x),
            "say": "\n\n".join(x for x in [str(record.get("say") or ""), *(f.say for f in new)] if x)})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[evidence_gate] could not write needs/%s (%r)", RECORD, exc)  # type: ignore[attr-defined]

    def _marked_say(self) -> list[str]:
        marked = getattr(self, "_result_marked", None) or []
        if not marked:
            return []
        loose = [m for m in marked if "not independently confirmed" in m]
        return ["The code marks " + "; ".join(marked) + ". Keep these values, and say plainly, in the limitations, that "
                "they are constant by design and not measured"
                + (", and that the constant was declared by the code's own writer and not independently confirmed." if loose else ".")]

    def _write_marked(self, record_path: Path) -> None:
        """A quantity the code marks ``# constant by design`` is accepted and noted in the limitations, never dropped."""
        record = {"removed_quantities": [], "removed_figures": [], "checks": [], "note": "; ".join(self._result_marked),
                  "say": "\n\n".join(self._marked_say())}
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", RECORD, exc)  # type: ignore[attr-defined]

    def _result_check_keys(self) -> set[str]:
        """The names the last run's results check took out of the results."""
        record = _read_json(self.quest_root / "needs" / RECORD)  # type: ignore[attr-defined]
        return {str(k) for k in record.get("removed_quantities") or []}

    def _result_checks_note(self) -> str:
        """For the writer: what the results check left out, so the limitations say so."""
        record = _read_json(self.quest_root / "needs" / RECORD)  # type: ignore[attr-defined]
        say = record.get("say")
        return say if isinstance(say, str) and say.strip() else ""
