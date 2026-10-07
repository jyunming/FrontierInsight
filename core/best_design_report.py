"""What a search for the best design hands over: the paper's *Best design found* section, one line for the person, and
the files to open.

The numbers come from FI's own record, never from the writer's memory: ``results/best_design.json`` (the search,
:mod:`core.optimise`) and ``needs/OPTIMUM_CHECK.json`` (the check at finer numerical settings, :mod:`core.optimum_check`).

* :func:`section` is the section the ENGINE puts into the paper after the writer has finished (:func:`mark_paper`,
  between ``<!-- fi:best-design -->`` markers, before the discussion): what was optimised, the baseline against the
  best design at every level of the numerical settings with the improvement and its numerical error, how the search
  ran, the check's verdict and each check in plain words, the target a person asked for and whether it was reached,
  and the limits of the result. Whatever the writer put between the markers is taken out first (:func:`without_block`).
  The writer is told the section is there (:func:`write_note`) and may discuss it, not restate its numbers.
* The numbers checks read the section like the rest of the paper: ``core/number_provenance.py`` is given the two
  records (:func:`records`), so every number in it traces to them. The claim check, which asks a model whether a
  sentence is grounded in the literature or the results, leaves the section out only when it is exactly the text FI
  writes (:func:`strip_for_checks`): it is FI's record, not a claim of the paper's.
* :func:`summary_line` is the ``[FI] best design:`` line the CLI prints and the VS Code chat shows; :func:`files` the
  files a person opens (``results/best_design.md``, the same text as a file of its own: :func:`write_readable`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import optimise as _optimise
from . import optimum_check as _check

BEGIN = "<!-- fi:best-design -->"
END = "<!-- /fi:best-design -->"
HEADING = "Best design found"
READABLE_PATH = Path("results") / "best_design.md"
PROGRESS_FIGURE = Path("figures") / "optimisation_progress.png"

#: The verdict of the check, as the one line says it.
_VERDICT_SHORT = {
    "verified": "it holds at finer numerical settings",
    "improvement_not_shown": "the improvement is not shown at finer numerical settings",
    "infeasible": "no design meets every limit at finer numerical settings",
    "not_local_optimum": "a nearby design is better at finer numerical settings (the search had not finished)",
    "unverified": "not checked at finer numerical settings",
}
_CHECK_LABELS = {"refinement": "Finer settings", "constraints": "Limits", "improvement": "Better than the baseline",
                 "starts": "Starting points", "neighbourhood": "Nearby designs", "budget": "Search budget"}
_STATUS = {"passed": "passed", "failed": "not passed", "not_checked": "not checked"}
_STOPPED = {
    "budget": "its evaluation budget ran out",
    "share": "each starting point converged or used its share of the budget",
    "converged": "every starting point converged",
    "finished": "every combination was evaluated",
    "time": "the time limit set for the run was reached",
    "not_run": "it did not run: the search before it had stopped (its time limit, or every evaluation failed)",
    "failures": "the first evaluations all failed",
    "library_error": "the library's method stopped with an error",
    "target": "a design reached the target asked for",
}


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt(value: Any) -> str:
    """A number as the paper prints it. A negative zero (``-0.0``, what ``0 - 0`` or ``-1 * 0`` gives, and what a tiny
    negative becomes once rounded) is printed as ``0``: no reader means anything by the sign of nothing."""
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        text = f"{value:.4g}"
        return "0" if float(text) == 0.0 else text
    return str(value)


#: The search methods FI has, as a scientist reads them. The paper names what the method does, never FI's identifier.
_METHOD_NAMES = {
    "bounded_local": "a local search inside the ranges, from several starting points",
    "global_then_local": "a search over the whole range first, then a local search from the best point found",
    "exhaustive": "every combination evaluated",
    "scan": "a coarse scan of the design space",
}
_LIBRARY_METHOD = re.compile(r"^(scipy|optuna):([A-Za-z][A-Za-z0-9_.-]*)$")


def method_name(method: Any) -> str:
    """The plain-language name of a search method id (``bounded_local``, ``scipy:differential_evolution``, ...), a
    neutral phrase for one this function does not know (an id is never printed in the paper)."""
    key = str(method or "").strip()
    if key in _METHOD_NAMES:
        return _METHOD_NAMES[key]
    library = _LIBRARY_METHOD.match(key)
    if library:
        where = "SciPy" if library.group(1) == "scipy" else "Optuna"
        return f"the {library.group(2)} method of {where}"
    return "a search method chosen for this study"


def _plain(text: Any) -> str:
    """``text`` with every method id in it replaced by its plain name."""
    out = str(text or "")
    for key in sorted(_METHOD_NAMES, key=len, reverse=True):
        out = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(key) + r"(?![A-Za-z0-9_])", _METHOD_NAMES[key], out)
    return _LIBRARY_METHOD.sub(lambda m: method_name(m.group(0)), out)


def _design(design: dict[str, Any] | None) -> str:
    return ", ".join(f"{k} = {_fmt(v)}" for k, v in (design or {}).items())


def _settings(settings: dict[str, Any] | None) -> str:
    return ", ".join(f"{k} = {_fmt(v)}" for k, v in (settings or {}).items())


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def records(quest_root: Path) -> dict[str, Any]:
    """``{"best_design": results/best_design.json, "optimum_check": needs/OPTIMUM_CHECK.json}`` as FI wrote them (each
    ``None`` when missing): what the paper's section is built from, and what the numbers checks trace it to."""
    root = Path(quest_root)
    return {"best_design": _read_json(root / _optimise.BEST_PATH), "optimum_check": _check.read(root)}


def _sign(best: dict[str, Any]) -> float:
    return 1.0 if str((best.get("objective") or {}).get("direction") or "minimise") == "minimise" else -1.0


def _variables_table(block: dict[str, Any] | None, baseline: dict[str, Any], chosen: dict[str, Any]) -> list[str]:
    variables = [v for v in ((block or {}).get("design_variables") or []) if isinstance(v, dict) and v.get("name")]
    names = [str(v["name"]) for v in variables] or list(dict.fromkeys([*baseline, *chosen]))
    by_name = {str(v["name"]): v for v in variables}
    rows = ["| Design variable | Range | Baseline | Best design |", "|---|---|---|---|"]
    for name in names:
        v = by_name.get(name) or {}
        unit = f" ({v['unit']})" if v.get("unit") else ""
        if isinstance(v.get("values"), list):
            allowed = "one of " + ", ".join(_fmt(x) for x in v["values"])
        elif _num(v.get("low")) and _num(v.get("high")):
            allowed = f"{_fmt(v['low'])} to {_fmt(v['high'])}"
        else:
            allowed = ""
        rows.append(f"| {name}{unit} | {allowed} | {_fmt(baseline.get(name, ''))} | {_fmt(chosen.get(name, ''))} |")
    return rows


def _objective_rows(best: dict[str, Any], check: dict[str, Any] | None, quantity: str, unit: str) -> list[str]:
    """Baseline against best design at the search's settings and at each finer level; the improvement at the search's
    and at the finest settings (with its numerical error, or its 95% interval with randomness)."""
    u = f" {unit}" if unit else ""
    search_best = best.get("best") or {}
    search_base = best.get("baseline") or {}
    imp = best.get("improvement") or {}
    chosen = (check or {}).get("best") if isinstance((check or {}).get("best"), dict) else None
    if chosen and chosen.get("replaces_search_best"):
        # The design reported is not the search's own best: its row is the reported design's, at the search's settings.
        search_best = {"objective": (chosen.get("objective") or {}).get("search")}
        imp = {"value": ((check or {}).get("improvement") or {}).get("search")}
    search_settings = best.get("search_settings") or {}
    rows = [f"| {quantity}{f' ({unit})' if unit else ''} | Baseline | Best design | Improvement over the baseline |",
            "|---|---|---|---|"]
    where = f" ({_settings(search_settings)})" if search_settings else ""
    rows.append(f"| at the search's settings{where} | {_fmt(search_base.get('objective', ''))} | "
                f"{_fmt(search_best.get('objective', ''))} | "
                f"{_fmt(imp['value']) + u if _num(imp.get('value')) else ''} |")
    if not check or not isinstance(check.get("best"), dict):
        return rows
    levels = (check.get("settings") or {}).get("levels") or []
    named = bool((check.get("settings") or {}).get("named"))
    b_vals = ((check.get("baseline") or {}).get("objective") or {}).get("check") or []
    c_vals = ((check["best"].get("objective") or {}).get("check")) or []
    cimp = check.get("improvement") or {}
    for i in range(max(len(b_vals), len(c_vals))):
        finest = i == max(len(b_vals), len(c_vals)) - 1
        label = ("at the same settings again" if not named else
                 f"at {'the finest' if finest else 'finer'} settings ({_settings(levels[i] if i < len(levels) else {})})")
        improvement = ""
        if finest and _num(cimp.get("value")):
            improvement = f"{_fmt(cimp['value'])}{u}"
            interval = cimp.get("interval") or {}
            if _num(interval.get("ci_lower")) and _num(interval.get("ci_upper")):
                improvement += f" (95% interval {_fmt(interval['ci_lower'])} to {_fmt(interval['ci_upper'])}{u})"
            elif _num(cimp.get("numerical_error")):
                improvement += f" ± {_fmt(cimp['numerical_error'])}{u} (numerical error of the two designs)"
        b = b_vals[i] if i < len(b_vals) else None
        c = c_vals[i] if i < len(c_vals) else None
        rows.append(f"| {label} | {_fmt(b) if _num(b) else 'no value'} | {_fmt(c) if _num(c) else 'no value'} | "
                    f"{improvement} |")
    return rows


def _limit_rows(check: dict[str, Any] | None, best: dict[str, Any]) -> list[str]:
    """Each limited quantity, baseline against best design, at the finest settings (else at the search's)."""
    chosen = (check or {}).get("best") if isinstance((check or {}).get("best"), dict) else None
    base = (check or {}).get("baseline") if chosen else None
    c_values = (chosen or {}).get("constraints") or ((best.get("best") or {}).get("constraints") or {})
    b_values = (base or {}).get("constraints") or ((best.get("baseline") or {}).get("constraints") or {})
    names = [n for n in dict.fromkeys([*c_values, *b_values])]
    if not names:
        return []
    where = "the finest settings" if chosen else "the search's settings"
    rows = [f"| Limited quantity (at {where}) | Baseline | Best design |", "|---|---|---|"]
    for name in names:
        bv, cv = b_values.get(name), c_values.get(name)
        limit = cv.get("limit") if isinstance(cv, dict) else None
        if limit:
            name = f"{name} (limit {limit})"
        if isinstance(cv, dict):
            cv = cv.get("value")
        if isinstance(bv, dict):
            bv = bv.get("value")
        rows.append(f"| {name} | {_fmt(bv) if _num(bv) else ''} | {_fmt(cv) if _num(cv) else ''} |")
    return rows


def section(quest_root: Path, block: dict[str, Any] | None = None, *, heading_level: int = 2) -> str:
    """The *Best design found* section (without its markers), or ``""`` when the quest has no record of a search.
    ``block`` (the plan's optimisation block) gives each design variable's range and unit."""
    rec = records(quest_root)
    best, check = rec["best_design"], rec["optimum_check"]
    if not isinstance(best, dict):
        return ""
    objective = best.get("objective") or {}
    quantity = str(objective.get("quantity") or "the objective")
    unit = str(objective.get("unit") or "")
    u = f" {unit}" if unit else ""
    sign = _sign(best)
    hashes = "#" * max(1, min(6, heading_level))
    lines = [f"{hashes} {HEADING}", "",
             "The numbers in this section are FI's own record of the search, every evaluation of it included, and of "
             "its check at finer numerical settings.", ""]
    what = f"**What was optimised.** {quantity}"
    if objective.get("meaning"):
        what += f" ({objective['meaning']})"
    what += f", made as {'low' if sign > 0 else 'high'} as possible"
    limits = [c for c in ((block or {}).get("constraints") or []) if isinstance(c, dict) and c.get("quantity")]
    if limits:
        what += "; limits: " + "; ".join(
            f"{c['quantity']} {c.get('limit')}{' ' + str(c['unit']) if c.get('unit') else ''}" for c in limits)
    lines += [what + ".", ""]
    chosen_check = (check or {}).get("best") if isinstance((check or {}).get("best"), dict) else None
    chosen = (chosen_check or {}).get("design") or (best.get("best") or {}).get("design") or {}
    baseline_design = (best.get("baseline") or {}).get("design") or {}
    if chosen or baseline_design:
        lines += [*_variables_table(block, baseline_design, chosen), ""]
    if best.get("best") is None:
        lines += [f"No design that meets every limit was found: {best.get('says')}", ""]
    else:
        lines += [*_objective_rows(best, check, quantity, unit), ""]
        limit_rows = _limit_rows(check, best)
        if limit_rows:
            lines += [*limit_rows, ""]
        if chosen_check and chosen_check.get("replaces_search_best"):
            lines += [f"The design reported is not the one the search chose ({_design((best.get('best') or {}).get('design'))}"
                      "): that one did not hold at the finest settings (see the check below).", ""]
    ev = best.get("evaluations") or {}
    method = best.get("method") or {}
    search = (f"**How the search ran.** Method: {method_name(method.get('used'))}"
              + (f" ({method_name(method.get('requested'))} was asked for, but {_plain(method.get('why'))})"
                 if method.get("why") else "")
              + f"; {ev.get('search')} of {ev.get('budget')} evaluations allowed were used")
    if _num(ev.get("starts")) and _num(ev.get("per_start")):
        search += f" (a budget of {ev['per_start']} evaluations for each of {ev['starts']} starting points"
        if _num(ev.get("added_budget")):
            search += f", and {ev['added_budget']} more asked for after the first result"
        search += ")"
    if ev.get("scan"):
        search += f", after a coarse scan of {ev['scan']} design(s) not counted in the budget"
    search += f"; the search stopped because {_STOPPED.get(str(ev.get('stopped_because')), ev.get('stopped_because'))}."
    lines += [search, ""]
    for r in best.get("continued") or []:
        toward = f", toward {quantity} = {_fmt(r['target'])}{u}" if _num(r.get("target")) else ""
        lines += [f"*Continued at the person's request (round {r.get('round')}):* {r.get('evaluations')} of "
                  f"{r.get('added')} more evaluations{toward}, from {_design(r.get('from'))}; stopped because "
                  f"{_STOPPED.get(str(r.get('stopped_because')), r.get('stopped_because'))}.", ""]
    if check:
        lines += [f"**Check at finer numerical settings.** {_plain(check.get('says'))}", ""]
        for name, label in _CHECK_LABELS.items():
            c = (check.get("checks") or {}).get(name)
            if isinstance(c, dict):
                lines.append(f"- {label}: {_STATUS.get(str(c.get('status')), c.get('status'))}: {_plain(c.get('says'))}.")
        if check.get("checks"):
            lines.append("")
    else:
        lines += ["**Check at finer numerical settings.** The best design has not been checked at finer numerical "
                  "settings: part of the improvement may be a numerical error.", ""]
    target = best.get("target") if isinstance(best.get("target"), dict) else None
    if target and _num(target.get("value")):
        asked = "the target asked for after the first result" if target.get("from") == "refine" else "the plan's target"
        relation = "at most" if sign > 0 else "at least"
        finest = (check or {}).get("best") if check else None
        f_values = [v for v in (((finest or {}).get("objective") or {}).get("check") or []) if _num(v)]
        reached = target.get("reached_at_finest_settings")
        if reached is True:
            outcome = f"reached at the finest settings ({quantity} = {_fmt(f_values[-1])}{u})" if f_values else "reached"
        elif reached is False:
            outcome = (f"not reached at the finest settings ({quantity} = {_fmt(f_values[-1])}{u})" if f_values
                       else "not reached")
        else:
            outcome = (f"{'reached' if target.get('reached') else 'not reached'} at the search's settings; not "
                       "confirmed at finer settings")
        lines += [f"**Target.** {asked[0].upper() + asked[1:]}: {quantity} {relation} {_fmt(target['value'])}{u}; "
                  f"{outcome}.", ""]
    limits_text = [*(check or {}).get("blind_spots", _check.BLIND_SPOTS)]
    if not (check or {}).get("passed"):
        limits_text.insert(0, "Not every check passed (see above), so this is the best design the search found, not "
                              "one shown to be the best there is.")
    lines += ["**Limits of this result.** " + " ".join(limits_text), ""]
    return "\n".join(lines).rstrip() + "\n"


# ---- the paper ---------------------------------------------------------------------------------------------------

_BLOCK = re.compile(re.escape(BEGIN) + r"[ \t]*\r?\n(.*?)\r?\n?[ \t]*" + re.escape(END), re.S)
_ANY_BLOCK = re.compile(r"\n*[ \t]*" + re.escape(BEGIN) + r".*?" + re.escape(END) + r"[ \t]*\n*", re.S)
_STRAY = re.compile(r"[ \t]*(?:" + re.escape(BEGIN) + "|" + re.escape(END) + r")[ \t]*\n?")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*)$")
_NUMBERED = r"^(?:\d+(?:\.\d+)*\.?\s+)?"
#: Where the section goes, in order: before the first of these that comes after the results (or the methods).
_BEFORE = [re.compile(_NUMBERED + w, re.I) for w in (
    r"discussion", r"conclusions?\b", r"limitations\b", r"references\b|further reading\b|acknowledg")]
_RESULTS = re.compile(_NUMBERED + r"(?:results|findings|methods?|methodology)\b", re.I)


def without_block(markdown: str) -> str:
    """``markdown`` with every block between the markers (and any marker left on its own) taken out."""
    def gap(m: re.Match[str]) -> str:
        return "" if m.start() == 0 else "\n" if m.end() == len(m.string) else "\n\n"

    text = markdown
    while True:
        before = text
        text = _ANY_BLOCK.sub(gap, text)
        text = _STRAY.sub("", text)
        if text == before:
            return text


def _headings(lines: list[str]) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    fenced = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        m = None if fenced else _HEADING.match(line)
        if m:
            out.append((i, len(m.group(1)), m.group(2).strip()))
    return out


def heading_level(markdown: str) -> int:
    """The level of the paper's sections: its second-level headings usually, the first level when every heading is
    one (a title and sections alike). The section uses the same."""
    levels = [lv for _i, lv, _t in _headings(markdown.split("\n"))]
    sections = [lv for lv in levels if lv > 1]
    if sections:
        return min(sections)
    return 1 if levels.count(1) > 1 else 2


def mark_paper(markdown: str, text: str) -> str:
    """``markdown`` with the section ``text`` (:func:`section`) once: before the discussion that follows the results
    (else the conclusions, the limitations, the references), else at the end. An empty ``text`` leaves no section."""
    markdown = without_block(markdown)
    if not text:
        return markdown
    lines = markdown.split("\n")
    headings = _headings(lines)
    level = heading_level(markdown)
    title = next((i for i, lv, _t in headings if lv == 1), None) if level == 1 else None
    sections = [(i, t) for i, lv, t in headings if lv <= level and i != title]
    after = next((i for i, t in sections if _RESULTS.match(t)), -1)
    at = len(lines)
    for pattern in _BEFORE:
        hit = next((i for i, t in sections if i > after and pattern.match(t)), None)
        if hit is not None:
            at = hit
            # The engine's references follow a rule line: the section goes before it.
            while at > 0 and not lines[at - 1].strip():
                at -= 1
            if at > 0 and lines[at - 1].strip() == "---":
                at -= 1
            break
    head = lines[:at]
    while head and not head[-1].strip():
        head.pop()
    tail = lines[at:]
    while tail and not tail[0].strip():
        tail.pop(0)
    block = [BEGIN, text.rstrip("\n"), END]
    return "\n".join([*head, *([""] if head else []), *block, *([""] if tail else []), *tail]) + (
        "" if tail else "\n")


def for_paper(quest_root: Path, block: dict[str, Any] | None, markdown: str) -> str:
    """The section as it goes into ``markdown`` (at the level of the paper's own sections)."""
    return section(quest_root, block, heading_level=heading_level(without_block(markdown)))


def strip_for_checks(text: str, expected: str) -> str:
    """``text`` with the section left out when it is exactly what FI writes (``expected``: :func:`section`), for the
    claim check; anything else between the markers stays and is checked."""
    if BEGIN not in text or not expected:
        return text
    want = expected.strip()
    return _BLOCK.sub(lambda m: " " if m.group(1).strip() == want else m.group(0), text)


def write_note(quest_root: Path) -> str:
    """What the writer is told: the section is FI's, with these numbers; discuss it, do not restate it."""
    rec = records(quest_root)
    best, check = rec["best_design"], rec["optimum_check"]
    if not isinstance(best, dict):
        return ""
    verdict = str((check or {}).get("verdict") or "unverified")
    passed = bool((check or {}).get("passed"))
    note = ("This study searched for the best design. FI puts a section titled \"" + HEADING + "\" into the paper "
            "itself after you finish, with the numbers from its own record: the objective, the baseline against the "
            "best design at each numerical setting, the improvement and its numerical error, how the search ran, the "
            "check at finer settings, any target asked for, and the limits. Do not write that table or restate its "
            "numbers in a section of your own; refer to the section and discuss what it means. Every number you do "
            "write about the best design, the baseline or the improvement must be one of FI's (the results' "
            "`fi_search`).")
    if verdict != "verified" or not passed:
        note += (" Not every check of the best design passed (" + _VERDICT_SHORT.get(verdict, verdict) + "): call it "
                 "\"the best design found\", never optimal or the best possible, and say what the check found.")
    target = best.get("target") if isinstance(best.get("target"), dict) else None
    if target and target.get("from") == "refine":
        reached = target.get("reached_at_finest_settings")
        note += (" The person running the study asked for a target after the first result; the section says whether it was reached ("
                 + ("reached" if reached else "not reached" if reached is False else "not confirmed") + "): say so "
                 "plainly in the abstract or the conclusions.")
    return note


# ---- the person ----------------------------------------------------------------------------------------------------


def summary_line(quest_root: Path) -> str:
    """One plain line: the best design, its improvement over the baseline, what the check says, the target. ``""``
    without a search."""
    rec = records(quest_root)
    best, check = rec["best_design"], rec["optimum_check"]
    if not isinstance(best, dict):
        return ""
    objective = best.get("objective") or {}
    quantity = str(objective.get("quantity") or "the objective")
    u = f" {objective['unit']}" if objective.get("unit") else ""
    chosen = (check or {}).get("best") if isinstance((check or {}).get("best"), dict) else None
    design = (chosen or {}).get("design") or (best.get("best") or {}).get("design")
    if not design:
        return f"no design that meets every limit was found ({_VERDICT_SHORT.get(str((check or {}).get('verdict')), 'not checked')})"
    imp = (check or {}).get("improvement") if chosen else None
    value = (imp or {}).get("value") if imp else (best.get("improvement") or {}).get("value")
    text = _design(design)
    if _num(value):
        text += (f" — {quantity} {_fmt(abs(value))}{u} {'better' if value > 0 else 'worse'} than the baseline"
                 if value else f" — {quantity} no better than the baseline")
        if imp and _num(imp.get("numerical_error")):
            text += f" (± {_fmt(imp['numerical_error'])}{u})"
    verdict = str((check or {}).get("verdict") or "unverified")
    text += f"; {_VERDICT_SHORT.get(verdict, verdict)}"
    target = best.get("target") if isinstance(best.get("target"), dict) else None
    if target and _num(target.get("value")):
        reached = target.get("reached_at_finest_settings")
        text += (f"; target {_fmt(target['value'])}{u}: "
                 + ("reached" if reached else "not reached" if reached is False else "not confirmed"))
    return text


def files(quest_root: Path) -> list[dict[str, str]]:
    """The files of a search for the best design a person opens, those that exist: ``[{"label", "path"}]`` (``path``
    relative to the quest folder)."""
    root = Path(quest_root)
    out = []
    for label, rel in (("The best design (readable)", READABLE_PATH), ("The best design (data)", _optimise.BEST_PATH),
                       ("Every evaluation of the search", _optimise.LEDGER_PATH),
                       ("The check at finer numerical settings", _check.CHECK_PATH),
                       ("The search's progress", PROGRESS_FIGURE)):
        if (root / rel).is_file():
            out.append({"label": label, "path": rel.as_posix()})
    return out


def write_readable(quest_root: Path, block: dict[str, Any] | None = None) -> Path | None:
    """``results/best_design.md``: the section as a file of its own, for a person to open. ``None`` without a search
    (or when it cannot be written: never a reason to stop)."""
    text = section(quest_root, block, heading_level=1)
    if not text:
        return None
    path = Path(quest_root) / READABLE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        return None
    return path
