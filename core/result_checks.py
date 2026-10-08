"""Cheap checks made on a run's results before the paper, as methods the engine mixes in (no model reads anything here).

After the run has finished and FI's own checks of it passed, the results are compared with what ran:

* what the analysis reports for each setting must be what ran at that setting (:mod:`core.reported_settings`);
* the figures and the reported quantities must show something: no empty panel, no series at one x, no quantity that is
  the same at every setting or stuck at a bound (:mod:`core.degenerate_results`);

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

from . import degenerate_results as _degenerate
from . import grid_keys as _grid_keys
from . import reported_settings as _reported

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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


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
        if result.returncode != 0 or not getattr(self, "_trial_mode", False):
            return result, failed_script
        self._result_marked = []
        found = self._result_findings(state, _e._extract_result_json(result.stdout or ""))
        if not found:
            if self._result_marked:
                self._write_marked(record_path)
            return result, failed_script
        used = self._result_counter(state)
        repair_left = (int(state.get("exec_reflect_iter", 0) or 0) < self.config.engine.exec_reflect_max_iterations  # type: ignore[attr-defined]
                       and not state.get("exec_give_up_reason"))
        if used < REPAIRS and repair_left:
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
        self._leave_out_results(found, record_path)
        return result, failed_script

    def _leave_out_results(self, found: list[Found], record_path: Path) -> None:
        """What is still wrong after the repairs is left out of the results the paper may use: written to
        ``needs/RESULT_CHECK.json`` (read for the writer's note and for the quantities the analysis no longer sees)."""
        self._drop_figures(sorted({n for f in found for n in f.figures}))
        keys = sorted({k for f in found for k in f.keys})
        record = {"removed_quantities": keys, "removed_figures": sorted({n for f in found for n in f.figures}),
                  "checks": [{"kind": f.kind, "note": f.note} for f in found], "note": "; ".join(f.note for f in found),
                  "say": "\n\n".join([*(f.say for f in found), *self._marked_say()])}
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", RECORD, exc)  # type: ignore[attr-defined]
        self._log.warning(  # type: ignore[attr-defined]
            "[execute] %s. The paper does not use them and says so in its limitations.", record["note"])

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
