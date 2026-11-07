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

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import code_findings as _code_findings
from . import code_reach as _code_reach
from . import code_review as _code_review
from .contract import figure_file
from . import degenerate_results as _degenerate
from . import reported_settings as _reported

#: Why a quantity a failing test is about is left out, as the must-fix hit for a paper that quotes it says it. FI holds no record
#: that tells a wrong value from a missing one: only the study run again with a repair says more, and only when it changed numbers.
SHOWN_FAULT_WHY = ("a test FI ran on the code that produced the results failed; FI did not establish whether the value is wrong "
                   "or missing")
SHOWN_FAULT_CHANGED_WHY = ("a test FI ran on the code that produced the results failed, and a repair that passes the test "
                           "changes the reported number")
SHOWN_FAULT_FIGURE_WHY = ("a test FI ran on the code that produced the results failed; FI did not establish whether the figure "
                          "is wrong or missing")

#: How many times, in all, the results of one design iteration are sent back for the checks of this module.
REPAIRS = 2
_COUNTER = "result_checks.json"
RECORD = "RESULT_CHECK.json"
TYPED_RECORD = "TYPED_RESULTS_CHECK.json"
ANALYSIS = "analysis"
SIMULATION = "simulation"


def _mark_dropped_figures(quest_root: Path, names: list[str], reasons: dict[str, str], log: Any) -> None:
    """The figures' entries in FI's record of the run (``data/results/derived.json``) become ``dropped``, with the reason FI
    took each out (``reasons``, else a plain default). Nothing else in the record changes; a record that is not there is left."""
    from . import analysis_frame as _analysis

    path = quest_root / _analysis.DERIVED_PATH
    if not names or not path.is_file():
        return
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("[execute] could not mark the dropped figures in the record (%r)", exc)
        return
    entries = record.get("figures") if isinstance(record, dict) else None
    if not isinstance(entries, list):
        return
    changed = False
    for name in names:
        file_name = Path(name).name
        why = reasons.get(name) or reasons.get(file_name) or "FI took it out of the results"
        for entry in entries:
            if isinstance(entry, dict) and Path(str(entry.get("file") or "")).name == file_name:
                entry.update({"status": "dropped", "why": why,
                              "said": f"Figure `{entry.get('id')}` was left out of the results: {why}."})
                changed = True
    if changed:
        from . import analysis_figures as _figures

        # The record's own sentence about the figures, built as the run builds it (so it counts the dropped one as not drawn).
        record["figures_said"] = _figures.said(entries)
        try:
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError as exc:
            log.warning("[execute] could not write the dropped figures to the record (%r)", exc)


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
    contested: str = ""  # "twice" (the plan defines the name two ways) or "plan_problem", when FI holds no passing check of the value
    claimed: list[str] = field(default_factory=list)  # the keys of a second model's claim that stay in the results until a measurement moves a number


#: ``Found.kind`` of a declared result that equals an earlier declared one in every trial.
COPY_KIND = "copy_of_result"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def removed_figure_stems(quest_root: Path | str) -> set[str]:
    """The stems of the figures FI took out of the paper's figures: ``removed_figures`` of ``needs/RESULT_CHECK.json``, the
    one record of it. A drawing saved in several formats is one figure, so a stem names all its files."""
    record = _read_json(Path(quest_root) / "needs" / RECORD)
    return {Path(str(n)).stem for n in record.get("removed_figures") or [] if str(n).strip()}


def _keep_measured(record: dict[str, Any], kept: list[dict[str, Any]]) -> None:
    """Change ``record`` (``needs/RESULT_CHECK.json``) in place for the checks of a second model's finding that no measurement
    showed to move a number (``kept``, each with its ``measured`` answer: ``unchanged``, or ``not_shown`` when no test showed
    the claim): their quantities are no longer taken out of the results. Only the keys that really return to the results are
    said to (``kept_differences``, the writer's note and the Limitations item are made for those alone): a key another check
    still takes out stays out, with that check's own words, and the kept check says nothing about it."""
    if not kept:
        return
    gone = {id(c) for c in kept}
    left = [c for c in record.get("checks") or [] if id(c) not in gone]
    held = {str(k) for c in left for k in c.get("keys") or []}
    back = {str(k) for c in kept for k in c.get("keys") or []} - held
    keys = [k for k in record.get("removed_quantities") or [] if str(k) not in back]
    names = {str(c["quantity"]) for c in kept}
    paragraphs = [p for p in str(record.get("say") or "").split("\n\n")
                  if p.strip() and not any(f"found that `{n}` is " in p for n in names)]
    returned = [{"quantity": c["quantity"], "keys": [k for k in c.get("keys") or [] if str(k) in back],
                 "note": c.get("note") or "", "measured": c.get("measured")} for c in kept]
    returned = [d for d in returned if d["keys"]]
    for d in returned:
        why = f"A second model found that `{d['quantity']}` is computed differently from the plan ({str(d['note']).rstrip('.')}). "
        if (d["measured"] or {}).get("status") == "unchanged":
            paragraphs.append(why + "FI observed this: a test written for the claim failed on the code that produced the results, "
                              "it passed on a repair, and the numbers the study reported with that repair did not move beyond their "
                              f"allowance. The value of `{d['quantity']}` stays in the results. Report it, and say plainly, in the "
                              "limitations, that a second model claims the method differs from the plan's definition and that these "
                              "three things were observed; do not say that the plan's definition was run.")
        else:
            paragraphs.append(why + "FI asked for a test that shows it and none did, so the claim is not shown and the value stays in "
                              f"the results. Report it, and say plainly, in the limitations, that a second model claims the code "
                              f"computes `{d['quantity']}` differently from the plan's definition and that FI could not show it.")
    record.update({"checks": left, "removed_quantities": keys,
                   "removed_values": {k: v for k, v in (record.get("removed_values") or {}).items() if str(k) in keys},
                   "kept_differences": [*(record.get("kept_differences") or []), *returned],
                   "note": "; ".join(str(c.get("note")) for c in left if c.get("note")), "say": "\n\n".join(paragraphs)})


def left_out_by_record(quest_root: Path | str) -> dict[str, str]:
    """The quantities FI left out of the results, each with the check's own note: what the results check
    (``needs/RESULT_CHECK.json``) and the typed-number check (``needs/TYPED_RESULTS_CHECK.json``) wrote as
    ``removed_quantities``. The one place that names them; the analysis (core/run_checks.py) and the paper's values
    (core/paper_values.py) both read it."""
    out: dict[str, str] = {}
    for name in (RECORD, TYPED_RECORD):
        record = _read_json(Path(quest_root) / "needs" / name)
        note = str(record.get("note") or "")
        for k in record.get("removed_quantities") or []:
            out.setdefault(str(k), note)
    return out


def _derived_numbers(derived: Any) -> dict[str, list[float]]:
    """``{id: its numbers}`` for each quantity FI's record of its analysis (``data/results/derived.json``) worked out: its value
    for each setting and its one value for the study. These live nowhere else: the printed results do not hold them."""
    items = derived.get("derived") if isinstance(derived, dict) and isinstance(derived.get("derived"), dict) else {}
    out: dict[str, list[float]] = {}
    for ident, item in items.items():
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        by_setting = item.get("by_setting") if isinstance(item.get("by_setting"), dict) else {}
        for v in [*by_setting.values(), item.get("value")]:
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(float(v)) != float("inf"):
                out.setdefault(str(ident), []).append(float(v))
    return out


def _values_left_out(result_json: Any, keys: set[str], derived: Any = None) -> dict[str, list[float]]:
    """``{key: its numbers}`` for each left-out key of ``result_json`` (at any depth), without a number that a quantity that
    stays also holds: the paper is checked for quoting the first, and a value both hold is not one that can be told apart.
    ``derived``: FI's record of its analysis; a left-out quantity that only that record holds (worked out by
    ``derive_<id>``, never printed) has its numbers listed from it too, or the check of the paper would have nothing to
    look for."""
    out: dict[str, list[float]] = {}
    kept: set[float] = set()
    for ident, vals in _derived_numbers(derived).items():
        if ident in keys:
            out.setdefault(ident, []).extend(vals)
        else:
            kept.update(vals)

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
        try:
            found.extend(self._coverage_recount_findings(state, result_json))
        except Exception as exc:  # noqa: BLE001 -- a check that cannot be made flags nothing
            self._log.warning("[execute] could not recount the coverages the analysis reports (%r)", exc)  # type: ignore[attr-defined]
        return found

    def _coverage_recount_findings(self, state: Any, result_json: Any) -> list[Found]:
        """A coverage of a declared known value that the analysis reports and FI, counting the same trials against the
        plan's true value, does not get (:func:`core.trial_checks.reported_coverage_mismatches`): sent back to the analysis."""
        from . import trial_checks as _trial_checks

        protocol = self._protocol_block(state)  # type: ignore[attr-defined]
        cells = _trial_checks.read_cells(self.quest_root)  # type: ignore[attr-defined]
        wrong = _trial_checks.reported_coverage_mismatches(
            cells, protocol, result_json, _trial_checks.read_thresholds(self.quest_root) or None)  # type: ignore[attr-defined]
        if not wrong:
            return []
        shown = "; ".join(f"`{w['id']}` at {w['where']}: reported {w['reported']:.4g}, FI counts {w['fi']:.4g} of "
                          f"{w['n']} trials" for w in wrong[:4])
        directive = ("FI counted, from the trials themselves, how often each trial's interval covers the true value the plan "
                     f"declares, and the analysis reports other coverages: {shown}. Compute each coverage as the share of "
                     "the trials whose interval (the two ends run_trial returns, or the estimate plus and minus the critical "
                     "value of the declared level times its standard error) contains the plan's true value, and change "
                     "nothing else.")
        note = f"the coverage the analysis reports is not FI's count of the same trials ({shown})"
        say = (f"The analysis reported coverages FI's own count of the trials does not give ({shown}); those numbers are "
               "left out: do not report them.")
        ids = sorted({w["id"] for w in wrong})
        return [Found("coverage_recount", ANALYSIS, directive, note, say, keys=ids, quantity=ids[0])]

    def _result_copy_findings(self) -> list[Found]:
        """A declared result that is the same number as an earlier declared one in every trial (FI's own comparison of the
        trials, :func:`core.flat_output.copied_results`): it is not reported as a quantity of its own, whatever the plan says it
        is (a half-width or an error written as a copy of the value). It is reported once, under the first. Nothing is repaired
        or stopped: the numbers stand under the first name, and the paper says the second was the same number."""
        from . import flat_output as _flat

        found: list[Found] = []
        for copy, (first, trials) in sorted(_flat.read_copies(self.quest_root).items()):  # type: ignore[attr-defined]
            note = f"`{copy}` was the same number as `{first}` in every one of {trials} trials"
            say = (f"{note[0].upper()}{note[1:]}, so it is not a second result: FI took `{copy}` out of the results. Do NOT "
                   f"report it as a quantity of its own or under its declared meaning; its numbers are those of `{first}`. "
                   f"Say plainly, in the limitations, that the study returned `{copy}` as a copy of `{first}` and reports it "
                   "once.")
            found.append(Found(COPY_KIND, ANALYSIS, "", note, say, keys=[copy], quantity=copy))
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
        names = [figure_file(f) for f in planned if str(f).strip()] if isinstance(planned, list) else []
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
        issues = _code_review.load(self.quest_root).get("issues") or {}  # type: ignore[attr-defined]
        never_run = _code_reach.never_run_in(self.quest_root)  # type: ignore[attr-defined]
        twice = _code_review.plan_defined_twice(self.quest_root)  # type: ignore[attr-defined]
        for q in differing:
            # Only "computed differently": a quantity called missing that the results hold is one the reader got wrong
            # (:func:`core.code_review.resolve_headline` says so for the headline; the writer's note covers the rest).
            if q["status"] != _code_review.DIFFERENT:
                continue
            # Not taken out when FI's own record contradicts the reader: a test written to show the fault passed on the code
            # (it was not shown), or the function it names is never run (core/code_reach.py). A finding no test was written
            # for still takes the quantity out: a real paper reported standard deviations under the plan's name for mean
            # absolute errors, and its reader, never tested, was right.
            issue = issues.get(q.get("fid")) if isinstance(issues, dict) else None
            if (isinstance(issue, dict) and issue.get("demonstrated") is False) or never_run(q.get("file_rel"), q.get("function")):
                continue
            # A name the plan defines two ways, or a finding marked as one about the plan, is not left out by the reader's word
            # alone but not kept by that mark either (the mark is a model's judgement): the value stays only when FI holds a
            # check it measured itself, that passed, and that reads this quantity.
            contested = ("twice" if _code_review.is_defined_twice(q["name"], twice) else
                         "plan_problem" if isinstance(issue, dict) and (issue.get("plan_conflict")
                                                                       or issue.get("outcome") == _code_findings.PLAN_PROBLEM) else "")
            if contested and _code_review.passing_checks(self.quest_root, q["name"]):  # type: ignore[attr-defined]
                continue
            keys = sorted(_code_review.named_in_results(q["name"], result_json))
            if not keys:
                continue
            what = "computed differently from the plan"
            why = q["note"].rstrip(".") or what
            if contested:
                why += ("; the plan defines it in two ways" if contested == "twice" else "; the finding is marked as one about the plan")
                why += ", and FI holds no passing check of its value"
            note = f"the plan's {q['kind']} `{q['name']}` is {what} in the code ({why})"
            # A second model's word alone never keeps a result out of the analysis (core/run_checks.py would strip it from the
            # input of derive.py): the evidence gate decides after a test shows the claim and a rerun moves a number
            # (engine._measure_reader_findings, _keep_measured). Until then the claim is recorded with its keys, which stay in.
            say = (f"A second model compared the code with the plan and found that `{q['name']}` is {what} ({why}). No test of "
                   f"FI's has shown it yet, so {', '.join('`' + k + '`' for k in keys[:6])} stays in the results for now.")
            found.append(Found("code_reader", ANALYSIS, "", note, say, quantity=q["name"], contested=contested, claimed=keys))
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
        found = _degenerate.quantity_problems(
            cells, reported, self._typed_result_exempt(state) | _degenerate.expected_constants(protocol, state.get("design")))  # type: ignore[attr-defined]
        quantities, kept = _degenerate.split_flat(found, protocol)
        self._result_kept = [_degenerate.kept_note(kept)] if kept else []
        if not (drawn or quantities):
            return []
        return [Found("degenerate_results", SIMULATION if quantities else ANALYSIS, _degenerate.directive(drawn, quantities),
                      _degenerate.plain_note(drawn, quantities), _degenerate.writer_note(drawn, quantities),
                      keys=[q.name for q in quantities], figures=sorted({p.figure for p in drawn}))]

    def _drop_figures(self, names: list[str], reasons: dict[str, str] | None = None) -> None:
        """Take exactly the named figure files out of ``figures/``; a figure's record (shared by its formats) goes with the last
        of them. The figure's entry in FI's record of the run (derived.json) is then ``dropped`` with its reason (``reasons``:
        figure file name -> why), so no reader takes it for a figure that was drawn."""
        folder = self.quest_root / "figures"  # type: ignore[attr-defined]
        gone: list[str] = []
        for name in names:
            stem = Path(name).stem
            targets = [folder / Path(name).name]
            others = [p for p in (folder.iterdir() if folder.is_dir() else [])
                      if p.is_file() and p.stem == stem and p.name != Path(name).name and p.suffix.lower() != ".json"]
            if not others:
                gone.append(name)
                targets += [folder / f"{stem}.json", self.fi_dir / "figure_records" / f"{stem}.json",  # type: ignore[attr-defined]
                            self.fi_dir / "figure_records" / f"{stem}.seed0.json"]  # type: ignore[attr-defined]
                # Its paper copy goes too: paper.pdf would otherwise show the figure through it.
                from .plot_style import remove_paper_copies

                remove_paper_copies(folder, name)
            for target in targets:
                try:
                    target.unlink(missing_ok=True)
                except OSError as exc:
                    self._log.warning("[execute] could not remove %s (%r)", target.name, exc)  # type: ignore[attr-defined]
        _mark_dropped_figures(self.quest_root, gone, reasons or {}, self._log)  # type: ignore[attr-defined]

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

    def _result_check_gave_up(self, state: Any, history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """When the repair gives up (or writes no code) on a run this check sent back: the run's results are not lost
        with it. The same code runs once more with this check's repairs counted as spent, so its results stand and what
        is still wrong is left out and said (:meth:`_leave_out_results`), as when the repairs run out. A reading of the
        results never ends a quest with nothing to write up. ``None`` when the last run was not sent back by this check."""
        if not (state.get("exec_result") or {}).get("result_check_sent_back"):
            return None
        self._save_result_counter(state, REPAIRS)
        self._log.warning(  # type: ignore[attr-defined]
            "[execute_reflect] no repair of what the results check found; the run is made once more and what is "
            "still wrong is left out of the results, as when its repairs run out")
        patch: dict[str, Any] = {
            "exec_patch_pending": True,
            "exec_reflect_iter": int(state.get("exec_reflect_iter") or 0) + 1,  # the repair asked counts, so this is bounded
        }
        if history is not None:
            patch["exec_reflect_history"] = history  # the repair's own reason for giving up, kept
        return patch

    async def _results_match_run(self, state: Any, result: Any, failed_script: str | None, split: bool) -> tuple[Any, str | None]:
        """After a run that finished: send the results back like a failed run when they do not match what ran (at most
        :data:`REPAIRS` times, and while the quest has repairs left), else leave out what is still wrong and say so.
        ``(the result to go on with, the script to repair)``."""
        from . import engine as _e
        from . import split_run as _split_run

        record_path = self.quest_root / "needs" / RECORD  # type: ignore[attr-defined]
        record_path.unlink(missing_ok=True)  # an earlier pass's record is not this run's
        self._analysis_sendback_body = ""  # the function of derive.py this run is sent back to, when FI wrote the analysis
        if not self._run_produced(result):  # type: ignore[attr-defined]
            return result, failed_script
        result_json = _e._extract_result_json(result.stdout or "")
        self._result_kept = []
        # Off the event loop: the comparison reads every trial value, and with --fleet a blocked loop stalls the other quests.
        found = (await asyncio.to_thread(self._result_findings, state, result_json)
                 if getattr(self, "_trial_mode", False) else [])
        # What a second model found the code computes otherwise cannot be repaired here (its repairs were spent before the run).
        # A run that drew none of the planned figures still has its results: they stand, and the writer is told there is
        # no figure. Sent back like a failed run, a repair that gave up would have thrown the results away and stopped the
        # quest with nothing to write up (a false stop).
        standing = [*self._code_reader_findings(result_json), *self._result_copy_findings(),
                    *self._figure_gap_findings(set()), *self._no_figures_standing(state)]
        if not found and not standing:
            if self._result_kept:
                self._write_kept(record_path)
            return result, failed_script
        used = self._result_counter(state)
        repair_left = (int(state.get("exec_reflect_iter", 0) or 0) < self.config.engine.exec_reflect_max_iterations  # type: ignore[attr-defined]
                       and not state.get("exec_give_up_reason"))
        body = ""
        if found and split and not any(f.fix == SIMULATION for f in found) and self._analysis_on():  # type: ignore[attr-defined]
            # FI wrote the analysis (core/analysis_frame.py): what is flagged goes to the function of derive.py FI's record
            # says made it (the figure, the derived quantity). What FI's own lines made has no function of the study's to
            # repair it: a fault of FI's, left out of the results and said, never a stop.
            body = self._fis_body_of(found)
            if not body:
                self._log.warning(  # type: ignore[attr-defined]
                    "[execute] the results check found %s in what FI's own analysis made, not a function of the study's: "
                    "a fault of FI's, left out of the results and said", "; ".join(f.note for f in found))
                repair_left = False
        if found and used < REPAIRS and repair_left:
            self._save_result_counter(state, used + 1)
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the results do not match what ran (%s); sending the %s back (%d of %d)",
                "; ".join(f.note for f in found), "simulation" if any(f.fix == SIMULATION for f in found) else "analysis",
                used + 1, REPAIRS)
            to_fix = (_split_run.SIMULATE_NAME if any(f.fix == SIMULATION for f in found) else _split_run.ANALYSIS_NAME) if split else None
            if body:
                from . import analysis_frame as _analysis

                to_fix, self._analysis_sendback_body = _analysis.DERIVE_FILE, body
            kept = "\n".join(line for line in (result.stdout or "").splitlines() if not line.startswith("RESULT_JSON:"))
            why = "\n\n".join(f.directive for f in found[:2])
            result = _e.ExecutionResult(1, kept, (result.stderr or "") + "\n[FI] " + why, result.duration_s, result.timed_out)
            self._clear_stale_figures()  # type: ignore[attr-defined]
            return result, to_fix
        if found:  # a figure that is taken out has no gap to report
            standing = [*self._code_reader_findings(result_json), *self._result_copy_findings(),
                        *self._figure_gap_findings({n for f in found for n in f.figures}), *self._no_figures_standing(state)]
        self._leave_out_results([*found, *standing], record_path, result_json)
        return result, failed_script

    def _fis_body_of(self, found: list[Found]) -> str:
        """The function of derive.py FI's record of the run (data/results/derived.json) says made what ``found`` flags: a
        figure by its file, a derived quantity by its id (also one of the keys taken out). ``""``: FI's own lines made it."""
        from . import analysis_frame as _analysis

        record = _analysis.read_record(self.quest_root)  # type: ignore[attr-defined]
        if record.get("from_earlier_run"):
            return ""
        for f in found:
            for name in f.figures:
                body = _analysis.made_by(record, figure=name)
                if body:
                    return body
            for ident in [f.quantity, *f.keys]:
                body = _analysis.made_by(record, quantity=str(ident).rsplit(".", 1)[-1]) if ident else ""
                if body:
                    return body
        return ""

    def _leave_out_results(self, found: list[Found], record_path: Path, result_json: Any = None) -> None:
        """What is still wrong after the repairs is left out of the results the paper may use: written to
        ``needs/RESULT_CHECK.json`` (read for the writer's note and for the quantities the analysis no longer sees)."""
        self._drop_figures(sorted({n for f in found for n in f.figures}), {n: f.note for f in found for n in f.figures})
        keys = sorted({k for f in found for k in f.keys})
        record = {"removed_quantities": keys, "removed_figures": sorted({n for f in found for n in f.figures}),
                  "removed_values": _values_left_out(result_json, set(keys), self._derived_record()),
                  "checks": [{"kind": f.kind, "note": f.note, **({"quantity": f.quantity, "keys": f.keys or f.claimed} if f.quantity else {}),
                              **({"contested": f.contested} if f.contested else {})}
                             for f in found],
                  "note": "; ".join(f.note for f in found),
                  "say": "\n\n".join([*(f.say for f in found), *self._kept_say()])}
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", RECORD, exc)  # type: ignore[attr-defined]
        self._log.warning(  # type: ignore[attr-defined]
            "[execute] %s. The paper is told and says so in its limitations.", record["note"])

    def _derived_record(self) -> dict[str, Any]:
        """FI's record of the last run of its analysis (``data/results/derived.json``), or ``{}``."""
        from . import analysis_frame as _analysis

        return _analysis.read_record(self.quest_root)  # type: ignore[attr-defined]

    def _leave_out_shown_faults(self, left_out: list[tuple[str, list[str], str]], not_done: list[tuple[str, str]],
                                result_json: Any, figures: list[tuple[str, str]] | None = None,
                                changed: dict[str, set[str]] | None = None) -> None:
        """At the evidence gate: the quantities a failing test is about (``(quantity, keys, how)``) are left out of the
        results the paper may use, the plan's quantities the code does not compute (``(quantity, how)``) are named, and the
        figures a failing test is about (``(file, how)``) are taken out of ``figures/``, by adding them to
        ``needs/RESULT_CHECK.json`` (what the run's checks left out stays). The words say only what FI's records show: that a
        repair which passes the test changes the reported number (``changed``: per quantity, the printed keys the rerun of that finding showed changed; every other key left out gets the second),
        else that the test failed and FI did not establish whether the value is wrong or missing. The same issues found again
        add nothing."""
        path = self.quest_root / "needs" / RECORD  # type: ignore[attr-defined]
        record = _read_json(path)
        found: list[Found] = []
        for quantity, keys, how in left_out:
            taken = ", ".join("`" + k + "`" for k in keys[:6])
            mine = (changed or {}).get(quantity, set())
            moved = [k for k in keys if k in mine]
            rest = [k for k in keys if k not in mine]
            said_moved = ", ".join("`" + k + "`" for k in moved[:6])
            said_rest = ", ".join("`" + k + "`" for k in rest[:6])
            if not rest:
                note = (f"a test FI ran on the code that produced the results failed on `{quantity}`, and a repair that passes "
                        f"the test changes the reported number ({how})")
                say = (f"A test FI ran on the code that produced the results failed on `{quantity}` ({how}). The study run again "
                       f"with a repair that passes the test reports other numbers, so the repair changes the reported number. "
                       f"FI took {taken} out of the results. Do NOT report, describe or quote any value for `{quantity}`, and do "
                       f"not call another number by its name. Say plainly, in the limitations, that `{quantity}` is left out of "
                       "this paper because a repair that passes a test of the code changes the reported number.")
            elif not moved:
                note = (f"a test FI ran on the code that produced the results failed on `{quantity}`; FI did not establish "
                        f"whether the value is wrong or missing ({how})")
                say = (f"A test FI ran on the code that produced the results failed on `{quantity}` ({how}). FI did not "
                       f"establish whether the value is wrong or missing. FI took {taken} out of the results. Do NOT report, "
                       f"describe or quote any value for `{quantity}`, and do not call another number by its name. Say plainly, "
                       f"in the limitations, that `{quantity}` is left out of this paper because a test of the code failed and "
                       "FI did not establish whether the value is wrong or missing.")
            else:  # the rerun showed some of what is left out changed, not all of it
                note = (f"a test FI ran on the code that produced the results failed on `{quantity}`; a repair that passes the "
                        f"test changes the reported number of {said_moved}, and FI did not establish whether the value of "
                        f"{said_rest} is wrong or missing ({how})")
                say = (f"A test FI ran on the code that produced the results failed on `{quantity}` ({how}). The study run again "
                       f"with a repair that passes the test reports other numbers for {said_moved}, so the repair changes the "
                       f"reported number of those. For {said_rest} FI did not establish whether the value is wrong or missing. "
                       f"FI took {taken} out of the results. Do NOT report, describe or quote any value for them, and do not "
                       "call another number by their names. Say plainly, in the limitations, that "
                       f"{said_moved} is left out because a repair that passes a test of the code changes the reported number, "
                       f"and {said_rest} because a test of the code failed and FI did not establish whether the value is wrong "
                       "or missing.")
            found.append(Found("shown_fault", ANALYSIS, "", note, say, keys=list(keys), quantity=quantity))
        for quantity, how in not_done:
            note = f"the code does not compute the plan's `{quantity}` ({how})"
            say = (f"The code does not compute `{quantity}`, which the plan asks for ({how}). Do not report a value for it or "
                   f"call another number by its name. Say plainly, in the limitations, that `{quantity}` was not computed.")
            found.append(Found("not_computed", ANALYSIS, "", note, say, quantity=quantity))
        for file, how in figures or []:
            note = (f"a test FI ran on the code that produced the results failed on the figure `{file}`; FI did not establish "
                    f"whether it is wrong or missing ({how})")
            say = (f"A test FI ran on the code that produced the results failed on the figure `{file}` ({how}). FI did not "
                   f"establish whether it is wrong or missing. FI took it out of the figures. Do NOT include it, refer to it or "
                   f"describe what it showed. Say plainly, in the limitations, that `{file}` is left out of this paper because a "
                   "test of the code failed and FI did not establish whether it is wrong or missing.")
            found.append(Found("shown_fault_figure", ANALYSIS, "", note, say, figures=[file], quantity=file))
        checks = [c for c in record.get("checks") or [] if isinstance(c, dict)]
        seen = {(c.get("kind"), c.get("quantity")) for c in checks}
        new = [f for f in found if (f.kind, f.quantity) not in seen]
        if not new:
            return
        gone_figures = sorted({n for f in new for n in f.figures})
        if gone_figures:
            self._drop_figures(gone_figures, {n: SHOWN_FAULT_FIGURE_WHY for n in gone_figures})
        keys = sorted({*(str(k) for k in record.get("removed_quantities") or []), *(k for f in new for k in f.keys)})
        why = record.get("removed_why") if isinstance(record.get("removed_why"), dict) else {}
        for f in new:
            mine = (changed or {}).get(f.quantity, set())
            for k in f.keys:
                if k in mine:
                    why[k] = SHOWN_FAULT_CHANGED_WHY  # only the finding whose own rerun moved the key says so
                else:
                    why.setdefault(k, SHOWN_FAULT_WHY)
        record.update({
            "removed_quantities": keys, "removed_figures": sorted({*(record.get("removed_figures") or []), *gone_figures}),
            "removed_values": _values_left_out(result_json, set(keys), self._derived_record()), "removed_why": why,
            "checks": [*checks, *({"kind": f.kind, "note": f.note, "quantity": f.quantity, "keys": f.keys,
                                  **({"contested": f.contested} if f.contested else {})} for f in new)],
            "note": "; ".join(x for x in [str(record.get("note") or ""), *(f.note for f in new)] if x),
            "say": "\n\n".join(x for x in [str(record.get("say") or ""), *(f.say for f in new)] if x)})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[evidence_gate] could not write needs/%s (%r)", RECORD, exc)  # type: ignore[attr-defined]

    def _kept_say(self) -> list[str]:
        """For the writer: the flat quantities the plan does not say are estimated from the data, kept and stated."""
        return list(getattr(self, "_result_kept", None) or [])

    def _write_kept(self, record_path: Path) -> None:
        """Flat quantities the plan does not name are kept in the results and stated in the limitations, never dropped."""
        record = {"removed_quantities": [], "removed_figures": [], "checks": [], "note": "; ".join(self._result_kept),
                  "say": "\n\n".join(self._kept_say())}
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
