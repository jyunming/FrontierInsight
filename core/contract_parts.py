"""The plan's model is asked, one part at a time, for what the study contract needs and the first plan call does not give.

The first plan call writes the settings and the list of results. Three more questions follow, each a plain one, each
checked against the ids FI already holds (:func:`core.contract.check_part`), and each asked again alone when its answer
cannot be used (at most :data:`MAX_ASKS` asks of a part in all):

* ``roles``: for every declared result, what kind of number it is and which settings it changes with;
* ``names``: for every name the plan relies on and does not declare (:func:`core.contract.undeclared_names`), whether it
  is a result, a reference value, a diagnostic, or computed afterwards from the results;
* ``comparisons``: which results the study compares between values of one setting;
* ``formulas``, only when FI works out the analysis itself (core/analysis_frame.py) and the plan has a formula FI cannot
  work out as written (:func:`core.contract.from_design` with ``fi_analysis``): a formula FI can work out for each, or that
  the study's own code computes it;
* ``answers``, only when FI is given the topic's main quantities (``topic_quantities``) and the plan's own map does not yet
  place them: which result (or equation) of the study answers each quantity, and at which setting when the topic asks
  for it at one. A quantity the plan cannot map stays unmapped and the paper's answers block says so.
* ``figures``, only when FI works out the analysis itself and the plan names a figure only by file name: what each figure
  shows, along which setting, in how many panels (or that it cannot be drawn from the study's numbers). FI draws only the
  figures the contract declares, so a figure named only by file name would be drawn by nobody.

When every part is usable FI writes ``form: contract`` into the protocol; the model never does. A part that stays
unusable is left out, named in ``protocol.parts_left_out`` (the contract reads that into its gaps), and the plan stays in
the older form; a formula still not workable is left out alone, with its gap, and the plan keeps its form; a figure still not declared is
left out alone, with its gap. Nothing here stops a quest.
"""

from __future__ import annotations

import copy
import json
import string
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import contract as _contract
from . import oracle_forms as _forms
from . import plan as _plan
from . import skeleton as _skeleton
from . import topic_requirements as _topic_req

#: The most times one part is asked: the first ask and two more.
MAX_ASKS = 3
PART_WORDS = {"roles": "what kind of number each result is, what it may change with and whether each run returns it",
              "names": "what the names the plan uses but does not declare are",
              "comparisons": "which results are compared between values of a setting",
              "formulas": "a formula FI can work out for each quantity the plan works out with one",
              "figures": "what each figure the plan names shows",
              "answers": "which result answers each quantity the topic asks for, and at which setting"}
#: The parts whose answer the plan's form does not wait on: a formula still not workable, or a figure still not declared,
#: is left out alone.
_OWN_PARTS = ("formulas", "figures", "answers")
_ROLE_HINT = {"reference": "its plan gives a formula, so it is a reference", "diagnostic": "a check of the plan measures it as a "
              "size of disagreement, so it is a diagnostic"}


@dataclass
class Outcome:
    design: Any
    parts: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # part -> traceback of a call that raised

    @property
    def form(self) -> str:
        return "contract" if self.parts and all(p["valid"] for k, p in self.parts.items() if k not in _OWN_PARTS) else "older"


def _short(text: Any, n: int = 1800) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n] + " ..."


def _settings_lines(p: dict[str, Any]) -> str:
    lines = []
    for key in ("grid", "in_each_run"):
        for name, values in (p.get(key) or {}).items() if isinstance(p.get(key), dict) else []:
            lines.append(f"- {name}: {json.dumps(values)}" + (" (every run goes through all of these together)" if key == "in_each_run" else ""))
    return "\n".join(lines) or "- (the study varies no setting)"


def _results_lines(p: dict[str, Any]) -> str:
    _by, violation = _contract.oracle_measures(p)
    lines = []
    for m in p.get("metrics") or []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        fixed = _contract.structural_role(m, violation)
        lines.append(f"- {m['id']}: {_short(m.get('estimand'), 200)}" + (f" [{_ROLE_HINT[fixed]}]" if fixed else ""))
    return "\n".join(lines)


def _names_lines(names: dict[str, str]) -> str:
    return "\n".join(f"- {n} (used in {where})" for n, where in names.items())


def _problems_text(problems: list[str], last: str) -> str:
    if not problems:
        return ""
    return ("\nYour last answer could not be used, for these reasons:\n" + "\n".join(f"- {x}" for x in problems[:8])
            + f"\n\nYour last answer was:\n{_short(last, 3000)}\n\nAnswer again, fixing only these.\n")


def _formulas_prompt(templates: dict[str, string.Template], design: dict[str, Any], faults: dict[str, str],
                     problems: list[str], last: str) -> str:
    from . import reference_values as _reference_values

    p = design.get("protocol") or {}
    rows = {str(d.get("id")): d for d in p.get("derived") or [] if isinstance(d, dict)}
    known = ([str(m.get("id")) for m in p.get("metrics") or [] if isinstance(m, dict) and m.get("id")]
             + [i for i in rows if i not in faults] + [str(k) for k in (p.get("grid") or {})]
             + sorted(_reference_values.analysis_inputs(p)))
    return templates["plan_formulas"].substitute(
        formulas="\n".join(f"- {i}: {rows.get(i, {}).get('formula')} ({why})" for i, why in faults.items()),
        known=", ".join(dict.fromkeys(known)) or "(nothing)", formula_language=_forms.ANALYSIS_FORMULA_LANGUAGE,
        method=_short(design.get("method")), again=_problems_text(problems, last))


def _formula_faults(design: dict[str, Any]) -> dict[str, str]:
    """The plan's derived quantities whose formula FI's own analysis cannot work out as written, each with why."""
    faults: dict[str, str] = {}
    _contract.from_design({**design, "protocol": {**design["protocol"], "form": "contract"}}, fi_analysis=True, left_out=faults)
    return faults


def _check_formulas(value: Any, asked: dict[str, str], design: dict[str, Any]) -> list[str]:
    """Why ``value``, the answer for ``formulas``, cannot be used; empty when it can. Each quantity asked about gets a
    ``formula``, or ``function: true`` with ``inputs`` (the reported numbers it is computed from)."""
    if not isinstance(value, dict):
        return ["the answer for `formulas` must be a mapping"]
    p = design.get("protocol") or {}
    declared = [str(m["id"]).strip() for m in p.get("metrics") or [] if isinstance(m, dict) and m.get("id")]
    out: list[str] = []
    for ident in asked:
        row = value.get(ident)
        if not isinstance(row, dict):
            out.append(f"`{ident}` is not answered")
        elif row.get("function") is True:
            inputs = row.get("inputs")
            if not isinstance(inputs, list) or not inputs or not all(i in declared for i in inputs):
                out.append(f"`{ident}`: a quantity the study's code computes needs `inputs`, the reported numbers it is "
                           f"computed from ({', '.join(declared)})")
        elif not (isinstance(row.get("formula"), str) and row["formula"].strip()):
            out.append(f"`{ident}`: give a `formula`, or `function`: true with its `inputs`")
    return out + [f"`{k}` was not asked about" for k in value if k not in asked]


def _apply_formulas(p: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """Each derived quantity asked about with the formula answered, or with none (the study's own function computes it from
    the inputs answered); the plan's own row is kept beside it, as the methodology audit judged it."""
    derived = []
    for d in p.get("derived") or []:
        row = answer.get(str(d.get("id"))) if isinstance(d, dict) else None
        if isinstance(row, dict):
            planned = {k: v for k, v in d.items() if k != "as_planned"}
            if row.get("function") is True:
                d = {**{k: v for k, v in planned.items() if k != "formula"}, "inputs": list(row["inputs"])}
            else:
                d = {**planned, "formula": row["formula"].strip()}
            d["as_planned"] = d.get("as_planned") or planned
        derived.append(d)
    return {**p, "derived": derived}


def _planned_figures(design: dict[str, Any]) -> list[Any]:
    p = design.get("protocol") or {}
    planned = p.get("figures") if "figures" in p else design.get("figures_planned")
    return planned if isinstance(planned, list) else []


def figure_faults(design: dict[str, Any]) -> dict[str, str]:
    """The figures the plan lists that nothing declares (named only by file name, or a declaration that no longer fits the
    plan's results and settings), each by file name with the plan's own words about it. A figure the plan's model answered
    ``drop`` for is not one."""
    p = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    c = _contract.from_design({**design, "protocol": {**p, "form": "contract"}})
    done = {_contract.figure_file(f.file) for f in c.figures if f.declared}
    said = p.get("figure_declarations") if isinstance(p.get("figure_declarations"), dict) else {}
    done |= {n for n, d in said.items() if isinstance(d, dict) and d.get("drop") is True}
    out: dict[str, str] = {}
    for f in _planned_figures(design):
        name = _contract.figure_file(f)
        if name and name not in done and name not in out:
            text = json.dumps(f, default=str) if isinstance(f, dict) else str(f).strip()
            out[name] = _short(text[len(name):].strip() if text.startswith(name) and text != name else text, 600)
    return out


def _per_trial(design: dict[str, Any]) -> list[str]:
    """The results every trial returns: a figure may use one as its ``x`` (each trial's pair of values is drawn)."""
    c = _contract.from_design({**design, "protocol": {**design["protocol"], "form": "contract"}}, fi_analysis=True)
    return c.per_trial_ids()


def _figure_holds(design: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """``(lines, settings, single)``: what a figure may show, one line each (a reported number or a quantity worked out from
    them: the analysis's table serves nothing else), the settings with more than one value, which a figure may use as ``x`` or
    ``series``, and the settings that have one."""
    c = _contract.from_design({**design, "protocol": {**design["protocol"], "form": "contract"}}, fi_analysis=True)
    lines = [f"- {r.id}: {_short(r.estimand or r.label, 160) or 'a reported number'}"
             + (f" [{r.unit}]" if r.unit else "") for r in c.results]
    lines += [f"- {d.id}: {_short(d.label, 160) or 'worked out from the reported numbers'}" for d in c.derived]
    values = {**c.settings.compared, **c.settings.in_each_run}
    return (lines, sorted(k for k in c.settings.varied() if len(values.get(k, ())) > 1),
            sorted(k for k in c.settings.varied() if len(values.get(k, ())) <= 1))


def _named_results(design: dict[str, Any]) -> dict[str, str]:
    """The numbers the plan itself says the study returns for each setting (its ``named_results`` answered ``result``) that no
    declared result or derived quantity carries: a figure may show one, and FI then declares it (:func:`_apply_figures`)."""
    p = design.get("protocol") or {}
    have = {str(m.get("id")) for key in ("metrics", "derived") for m in p.get(key) or [] if isinstance(m, dict)}
    return {str(n["id"]): _short(n.get("what") or n["id"], 160) for n in p.get("named_results") or []
            if isinstance(n, dict) and n.get("id") and n.get("answer") == "result" and str(n["id"]) not in have}


def _drawable(design: dict[str, Any]) -> list[str]:
    """What FI's own record offers a figure to show, along a setting that takes more than one value: the declared results and
    derived quantities and the numbers the plan names as returned. Empty when the study varies nothing or reports nothing; only
    then is a figure one nobody can draw. A model's word that a figure cannot be drawn is never what decides it."""
    lines, settings, _single = _figure_holds(design)
    return [line[2:].split(":")[0] for line in lines] + list(_named_results(design)) if settings else []


def _record_reads(design: dict[str, Any]) -> list[str]:
    """What FI's record of the trials holds, for the figures part: the frame's own reads, with this plan's facts (what each run
    compares inside it, the results that change with it, the results with a true value and an interval)."""
    from . import analysis_frame as _frame

    protocol = {**design["protocol"], "form": "contract"}
    c = _contract.from_design({**design, "protocol": protocol}, fi_analysis=True)
    compared = {k: list(v) for k, v in c.settings.in_each_run.items()}
    per_level = {r.id: [k for k in compared if k in (r.depends_on or ())] for r in c.results if set(r.depends_on or ()) & set(compared)}
    return _skeleton.table_reads(compared, per_level, [r.id for r in c.results if r.id in _frame._known_values(protocol, c)])


def _figures_prompt(templates: dict[str, string.Template], design: dict[str, Any], faults: dict[str, str],
                    problems: list[str], last: str) -> str:
    lines, settings, _single = _figure_holds(design)
    return templates["plan_figures"].substitute(
        figures="\n".join(f"- {name}: {text}" for name, text in faults.items()),
        shows="\n".join([*lines, *(f"- {j}: {w} (the plan names it as returned for each setting; FI declares it if a figure shows it)"
                                   for j, w in _named_results(design).items())]) or "- (nothing)",
        settings=", ".join(settings) or "(the study varies no setting)", max_panels=_contract.MAX_PANELS,
        method=_short(design.get("method")), again=_problems_text(problems, last),
        record="\n".join(_record_reads(design)), per_trial=", ".join(_per_trial(design)) or "(none)")


def _check_figures(value: Any, asked: dict[str, str], design: dict[str, Any]) -> list[str]:
    """Why ``value``, the answer for ``figures``, cannot be used; empty when it can. Each figure asked about gets what it
    shows (a reported number or a quantity worked out from them), its ``x`` setting, a ``series`` setting other than ``x`` (or
    none) and its panels, or ``drop`` with why. A setting must have more than one value to be drawn along."""
    if not isinstance(value, dict):
        return ["the answer for `figures` must be a mapping"]
    lines, settings, single = _figure_holds(design)
    per_trial = _per_trial(design)
    named = _named_results(design)
    ids = [line[2:].split(":")[0] for line in lines] + list(named)
    out: list[str] = []
    for name in asked:
        row = value.get(name)
        if not isinstance(row, dict):
            out.append(f"`{name}` is not answered")
            continue
        if row.get("drop") is True:
            if _drawable(design):
                out.append(f"`{name}`: FI does not drop a figure on a model's word; the study gives numbers to draw along "
                           f"{', '.join(settings)} ({', '.join(ids)}). Say which of them the figure shows, or the closest")
            continue
        shows = row.get("shows")
        if not isinstance(shows, list) or not shows or not all(isinstance(s, str) for s in shows):
            out.append(f"`{name}`: `shows` must list the numbers the figure shows ({', '.join(ids)}), or answer `drop`: true")
        elif [s for s in shows if s not in ids]:
            out.append(f"`{name}` shows {', '.join('`' + s + '`' for s in shows if s not in ids)}, which the study does not "
                       f"report or work out (it holds {', '.join(ids)})")
        for axis in ("x", "series"):
            used = row.get(axis)
            if axis == "x" and used in (None, ""):
                out.append(f"`{name}`: `x` is the setting the figure is drawn along ({', '.join(settings) or 'none'})")
            elif axis == "x" and used in per_trial and not [s for s in row.get("shows") or [] if s not in per_trial]:
                continue  # a result every trial returns, drawn against the results it is paired with
            elif used not in (None, "") and used not in settings:
                why = "has one value in the study" if used in single else "is not a setting of the study"
                out.append(f"`{name}`: `{axis}` is `{used}`, which {why} (the settings to use are {', '.join(settings) or 'none'})")
        if row.get("x") not in (None, "") and row.get("x") == row.get("series"):
            out.append(f"`{name}`: `series` must be another setting than `x`, or null")
        panels = row.get("panels", 1)
        if isinstance(panels, bool) or not isinstance(panels, int) or not 1 <= panels <= _contract.MAX_PANELS:
            out.append(f"`{name}`: `panels` must be a whole number from 1 to {_contract.MAX_PANELS}")
    return out + [f"`{k}` was not asked about" for k in value if k not in asked]


def _apply_figures(p: dict[str, Any], design: dict[str, Any], answer: dict[str, Any], asked: dict[str, str]) -> dict[str, Any]:
    """What the plan's model said about each figure asked, kept by file name in ``figure_declarations`` beside the plan's own
    list of figures, which is never changed (the contract joins the two). The earlier declarations of files the plan still
    lists are kept; a file no longer listed loses its."""
    listed = {_contract.figure_file(f) for f in _planned_figures({**design, "protocol": p})}
    old = p.get("figure_declarations") if isinstance(p.get("figure_declarations"), dict) else {}
    decls = {n: d for n, d in old.items() if n in listed and n not in asked and isinstance(d, dict)}
    for name in asked:
        row = answer[name]
        decls[name] = ({"drop": True, "why": _short(row.get("why"), 300)} if row.get("drop") is True else
                       {"shows": list(row["shows"]), "x": row.get("x") or None, "series": row.get("series") or None,
                        "panels": row.get("panels", 1)})
    # A number the plan names as returned for each setting, which a figure now shows, is declared as a result of the study,
    # computed afterwards by the study's own function (so no run is required to return it), along the settings the figure draws.
    named = _named_results({**design, "protocol": p})
    shown = [i for name in asked if answer[name].get("drop") is not True for i in answer[name]["shows"] if i in named]
    if not shown:
        return {**p, "figure_declarations": decls}
    metrics = list(p.get("metrics") or [])
    _lines, many, few = _figure_holds(design)
    for ident in dict.fromkeys(shown):
        along = list(dict.fromkeys(a for name in asked if ident in (answer[name].get("shows") or [])
                                   for a in (answer[name].get("x"), answer[name].get("series")) if a in (*many, *few)))
        metrics.append({"id": ident, "kind": "mean", "estimand": named[ident], "unit": "one setting", "role": "result",
                        "depends_on": along, "returned": "afterwards"})
    left = [n for n in p.get("named_results") or [] if not (isinstance(n, dict) and n.get("id") in shown)]
    return {**{k: v for k, v in p.items() if k != "named_results"}, "metrics": metrics, **({"named_results": left} if left else {}),
            "figure_declarations": decls}


#: What a question about a result may ask of it, a closed set FI works out itself from its keys: the result as it is (its mean in
#: each setting), its bias (the mean minus the true value the plan declares for it), or its standard deviation over the trials.
STATISTICS = ("value", "bias", "sd")


def _answers_prompt(templates: dict[str, string.Template], design: dict[str, Any], asked: dict[str, dict[str, str]],
                    problems: list[str], last: str) -> str:
    lines, _settings, _single = _figure_holds(design)
    declared = {str(m.get("id")): str(m["physical_unit"]).strip() or "1" for m in design["protocol"].get("metrics") or []
                if isinstance(m, dict) and m.get("physical_unit") is not None}
    lines = [f"{x} [{declared[x[2:].split(':')[0]]}]" if x[2:].split(":")[0] in declared else x for x in lines]
    quantities = "\n".join(f"- {i}: {q['name']}" + (f" (the topic: {_short(q['stated'], 200)})" if q["stated"] else "")
                           + (f" [the plan maps it to `{q['by']}`]" if q.get("by") else "") for i, q in asked.items())
    settings = "\n".join(f"- {a}: {json.dumps(v)}" for a, v in _topic_req.view(design, design["protocol"])["axes"].items())
    eqs = ", ".join(sorted(_contract._model_equation_ids(design["protocol"]))) or "(none)"
    return templates["plan_answers"].substitute(
        quantities=quantities, statistics=", ".join(STATISTICS), shows="\n".join(lines) or "- (nothing)", equations=eqs, settings=settings or "- (none)",
        method=_short(design.get("method")), again=_problems_text(problems, last))


def _worked_from(protocol: dict[str, Any], ident: str) -> list[str]:
    """The results a derived result of the plan is worked out from (``[]`` for anything else)."""
    return next((list(d.get("inputs") or []) for d in protocol.get("derived") or []
                 if isinstance(d, dict) and d.get("id") == ident and d.get("role", "result") == "result"), [])


def _refined_by(protocol: dict[str, Any], planned: str, answered: str) -> str:
    """The plan's own ``by`` (``planned``), unless the answer names a derived result worked out from the very result the
    plan mapped: a number computed from a result (a crossing, a rate, an order) is the quantity itself where the result is only
    what it was computed from. Any other answer leaves the plan's map as it is."""
    head, _s, name = planned.partition(":")
    new_head, _s, new_name = answered.partition(":")
    if head.strip() == "metric" and new_head.strip() == "metric" and name.strip() in _worked_from(protocol, new_name.strip()):
        return f"metric:{new_name.strip()}"
    return planned


def _unit_key(text: Any) -> str:
    """A unit as written, compared by its letters only: case and spaces do not matter, and a pure number (``""``) is ``1``."""
    return "".join(str(text if text is not None else "").lower().split()) or "1"


def _unit_of(design: dict[str, Any], result: str, stated: Any) -> Any:
    """The unit of the result ``metric:<id>`` / ``equation:<id>``: the plan's own ``physical_unit`` for it where the plan
    declares one, else what the answer states (``None`` when neither)."""
    head, _s, name = result.partition(":")
    declared = next((m.get("physical_unit") for m in design["protocol"].get("metrics") or []
                     if isinstance(m, dict) and head.strip() == "metric" and str(m.get("id")) == name.strip()
                     and m.get("physical_unit") is not None), None)
    return declared if declared is not None else stated


def _differ(asked_in: Any, got: Any) -> tuple[str, str] | None:
    """``(asked_in, got)`` as written when both are given and are not the same unit, else ``None``."""
    if asked_in is None or got is None or _unit_key(asked_in) == _unit_key(got):
        return None
    return str(asked_in).strip() or "1", str(got).strip() or "1"


def _same_unit(asked_in: Any, got: Any) -> bool:
    """Whether both units are given and are the same unit."""
    return asked_in is not None and got is not None and _unit_key(asked_in) == _unit_key(got)


def unit_gaps(row: Any, planned: str, design: dict[str, Any]) -> dict[str, tuple[str, str] | None]:
    """The units of one answer, ``{"chosen": ..., "planned": ...}``, each ``(the unit the question asks in, the unit of the
    result)`` when both are given and differ, else ``None``. ``chosen``: the result the answer names (``by``); ``planned``: the
    result the plan's own map names (``planned``, ``""`` for none). The question's unit is the answer's ``unit``; a result's
    is the plan's ``physical_unit`` for it where declared, else the answer's ``by_unit`` / ``plan_unit``."""
    out: dict[str, tuple[str, str] | None] = {"chosen": None, "planned": None}
    if not isinstance(row, dict):
        return out
    by = str(row.get("by") or "").strip()
    if by.partition(":")[0].strip() in ("metric", "equation"):
        out["chosen"] = _differ(row.get("unit"), _unit_of(design, by, row.get("by_unit")))
    if planned.partition(":")[0].strip() in ("metric", "equation"):
        stated = row.get("plan_unit")
        if stated is None and by in ("", planned):
            stated = row.get("by_unit")
        out["planned"] = _differ(row.get("unit"), _unit_of(design, planned, stated))
    return out


def _unit_reason(units: tuple[str, str], planned: bool) -> str:
    """Why a quantity is not answered: the result named for it is in another unit than the question asks in."""
    said = [("a pure number (no unit)" if _unit_key(u) == "1" else u) for u in units]
    return f"the result {'the plan names' if planned else 'named for it'} is in {said[1]}, the question asks for {said[0]}"


def _check_answers(value: Any, asked: dict[str, dict[str, str]], design: dict[str, Any],
                   conflicts: dict[str, str] | None = None) -> list[str]:
    """Why ``value``, the answer for ``answers``, cannot be used; empty when it can. Each quantity asked about gets ``by`` (a
    result or derived id as ``metric:<id>``, an equation as ``equation:<id>``, or ``none``; not needed where the plan already
    maps it) and, only if the topic asks for it at one setting, ``at``: a setting of the study with one of its values."""
    if not isinstance(value, dict):
        return ["the answer for `answers` must be a mapping"]
    lines, _settings, _single = _figure_holds(design)
    ids = [line[2:].split(":")[0] for line in lines]
    eqs = _contract._model_equation_ids(design["protocol"])
    axes = _topic_req.view(design, design["protocol"])["axes"]
    out: list[str] = []
    for ident, q in asked.items():
        row = value.get(ident)
        if not isinstance(row, dict):
            out.append(f"`{ident}` is not answered")
            continue
        by = str(row.get("by") or "").strip()
        head, _s, name = by.partition(":")
        if not q.get("by") and by != "none" and not ((head == "metric" and name in ids) or (head == "equation" and name in eqs)):
            out.append(f"`{ident}`: `by` must be `metric:<id>` with an id of {', '.join(ids)}, `equation:<id>` with an id of "
                       f"{', '.join(sorted(eqs)) or '(none)'}, or `none`")
        # The unit the question asks in against the unit of the result it will be answered by: the plan's map when the answer
        # keeps it, else the result the answer names. A change of the plan's map is taken only in _apply_answers, when the
        # plan's own result is in another unit.
        planned = q.get("by", "")
        kept = bool(planned) and by in ("", planned)
        gaps = unit_gaps(row, planned, design)
        units = gaps["planned"] if kept else gaps["chosen"] if not planned or gaps["planned"] else None
        if units:
            result = planned if kept else by
            if conflicts is None:
                out.append(f"`{ident}`: the question asks for a quantity in `{units[0]}`, but `{result}` is in `{units[1]}`; "
                           f"answer a result that is in `{units[0]}`, or `none`")
            else:
                conflicts[ident] = _unit_reason(units, kept)
        stat = row.get("statistic")
        if stat not in (None, "") and stat not in STATISTICS:
            out.append(f"`{ident}`: `statistic` must be one of {', '.join(STATISTICS)}")
        elif stat == "bias":
            from . import paper_values as _values

            target = (q.get("by") or by).partition(":")[2].strip()
            if _values.known_truth(design["protocol"], target) is None:
                out.append(f"`{ident}`: `bias` needs a result with a known true value that the plan fixes; `{target or by}` has none "
                           "(answer `value` or `sd`, or another result)")
        at = row.get("at")
        if at in (None, ""):
            continue
        if not isinstance(at, dict):
            out.append(f"`{ident}`: `at` must be a mapping of a setting to one of its values, or null")
            continue
        for axis, v in at.items():
            if axis not in axes:
                out.append(f"`{ident}`: `at` names `{axis}`, which is not a setting of the study ({', '.join(axes) or 'none'})")
            elif not any(v == x or str(v) == str(x) for x in axes[axis]):
                out.append(f"`{ident}`: `{axis}` has no value {v!r} (its values are {json.dumps(axes[axis])})")
    return out + [f"`{k}` was not asked about" for k in value if k not in asked]


def _apply_answers(p: dict[str, Any], design: dict[str, Any], answer: dict[str, Any],
                   asked: dict[str, dict[str, str]], conflicts: dict[str, str] | None = None) -> dict[str, Any]:
    """The plan's ``topic_map`` with an ``at`` on the entry of each quantity asked about (the plan's own ``by`` is kept), and
    an entry added where the plan had none and the model named a result; ``none`` adds nothing."""
    rows = [dict(m) for m in p.get("topic_map") or [] if isinstance(m, dict)]
    for ident, q in asked.items():
        row = answer[ident]
        at = row.get("at") if isinstance(row.get("at"), dict) else {}
        keys = {_topic_req._name_key(ident), _topic_req._name_key(q["name"])}
        mine = [m for m in rows if _topic_req._name_key(m.get("item")) in keys]
        by = str(row.get("by") or "").strip()
        stat = row.get("statistic") if row.get("statistic") in ("bias", "sd") else None
        planned = q.get("by", "")
        gone = unit_gaps(row, planned, design)["planned"] if planned and by not in ("", planned) else None
        for m in mine:
            m["at"] = at
            if stat:
                m["statistic"] = stat
            else:
                m.pop("statistic", None)  # the answer is the value: an earlier entry's bias or sd no longer holds
            m["by"] = _refined_by(design["protocol"], str(m.get("by") or ""), by)
            m.pop("unit_conflict", None)
            if ident in (conflicts or {}):
                m["unit_conflict"] = conflicts[ident]
            elif gone and by == "none":
                m["unit_conflict"] = _unit_reason(gone, True)  # the plan's result is in another unit than the question
            elif gone and _same_unit(row.get("unit"), _unit_of(design, by, row.get("by_unit"))):
                m["by"] = by  # ... and the answer names a result in the question's unit
            elif gone:  # ... and the result the answer names has no unit stated or declared: neither is taken
                m["unit_conflict"] = _unit_reason(gone, True)
        if not mine and by and by != "none":
            rows.append({"item": ident, "by": by, "at": at, **({"statistic": stat} if stat else {}),
                         **({"unit_conflict": conflicts[ident]} if ident in (conflicts or {}) else {})})
    return {**p, "topic_map": rows}


def _prompt(part: str, templates: dict[str, string.Template], design: dict[str, Any], names: dict[str, str],
            problems: list[str], last: str, derived: list[str]) -> str:
    p = design.get("protocol") or {}
    return templates[f"plan_{part}"].substitute(
        method=_short(design.get("method")), settings=_settings_lines(p), results=_results_lines(p),
        names=_names_lines(names), equations=", ".join(sorted(_contract._model_equation_ids(p))) or "(none)",
        derived=", ".join(derived) or "(none)", again=_problems_text(problems, last),
        formula_language=_forms.EXPECTED_FORMULA_LANGUAGE)


def _apply_roles(p: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """The model's role, what each result may change with ("may change with": a superset, read by no check) and whether
    each run returns it. A role the plan's structure gives (a reference formula, a violation-kind check) wins."""
    _by, violation = _contract.oracle_measures(p)
    metrics = []
    for m in p.get("metrics") or []:
        row = answer.get(str(m.get("id"))) if isinstance(m, dict) else None
        if isinstance(row, dict):
            role = _contract.structural_role(m, violation) or row["role"]
            m = {**m, "role": role, "depends_on": list(row["depends_on"]), "returned": row["returned"]}
        metrics.append(m)
    return {**p, "metrics": metrics}


def _apply_names(p: dict[str, Any], answer: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(kept, to_check)``. A name never becomes a result every run must return: a quantity computed afterwards, or a
    reference FI computes from a formula of the settings, is a ``derived`` entry; a name the model calls a result, or a
    reference by a model equation, is recorded in ``named_results`` (the contract's notes) and nothing requires it. ``to_check``
    is ``kept`` with each formula as a temporary metric, so the plan's own reading of a reference (it computes at every
    setting, it agrees with the checks) judges it."""
    derived, named, temp = list(p.get("derived") or []), list(p.get("named_results") or []), []
    for name, row in answer.items():
        what = _short(row.get("what") or name, 300)
        if row["answer"] == "derived":
            derived.append({"id": name, "inputs": list(row["inputs"]), "role": "result", "label": what, "asked": True})
        elif row["answer"] == "reference" and isinstance(row.get("formula"), str) and row["formula"].strip():
            formula = row["formula"].strip()
            derived.append({"id": name, "inputs": [], "role": "reference", "formula": formula, "label": what, "asked": True})
            temp.append({"id": name, "kind": "mean", "estimand": what, "unit": "one setting",
                         "reference": {"formula": formula, "kind": "value"}})
        elif row["answer"] in ("result", "reference"):
            named.append({"id": name, "answer": row["answer"], "what": what,
                          **({"equation": str(row["equation"]).strip()} if row.get("equation") else {})})
    kept = {**p, **({"derived": derived} if derived else {}), **({"named_results": named} if named else {})}
    return kept, {**kept, "metrics": [*(p.get("metrics") or []), *temp]}


def _apply_comparisons(p: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    """The plan's own comparisons stay; the model's are added when the plan has none for that result and setting."""
    rows = [c for c in p.get("comparisons") or [] if isinstance(c, dict)]
    have = {(str(c.get("result")), str(c.get("across"))) for c in rows}
    ids = {str(c.get("id")) for c in rows}
    for c in items:
        key = (str(c["result"]), str(c["across"]))
        if key in have:
            continue
        have.add(key)
        cid = f"{c['result']}_by_{c['across']}"
        rows.append({"id": cid if cid not in ids else f"{cid}_{len(rows) + 1}", "result": c["result"], "across": c["across"],
                     **({"values": list(c["values"])} if c.get("values") else {}), "how": c.get("how", "difference"),
                     "asked": True})
    return {**p, **({"comparisons": rows} if rows else {})}


def strip_fi_keys(design: Any) -> Any:
    """``design`` without the keys only FI writes (``form``, ``parts_left_out``, ``names_not_asked``, ``figure_declarations``, and the ``asked`` mark on a comparison or derived quantity): a model's own word for them is never
    believed."""
    if not isinstance(design, dict) or not isinstance(design.get("protocol"), dict):
        return design
    protocol = {k: v for k, v in design["protocol"].items() if k not in (*_FI_KEYS, "figure_declarations")}
    for key in _ASKED_ROWS:
        if isinstance(protocol.get(key), list):
            protocol[key] = [{k: v for k, v in r.items() if k not in ("asked", "as_planned")} if isinstance(r, dict) else r
                             for r in protocol[key]]
    return design if protocol == design["protocol"] else {**design, "protocol": protocol}


def without_answers(design: Any) -> Any:
    """``design`` without what FI wrote into it from the answers to its questions after the plan: each metric's
    ``role``, ``depends_on`` and ``returned``, the comparisons and derived quantities marked ``asked``, the named
    results and the keys only FI writes. The methodology audit's verdict is on the rest (core/receipts.design_core)."""
    if not isinstance(design, dict) or not isinstance(design.get("protocol"), dict):
        return design
    protocol = {k: v for k, v in design["protocol"].items() if k not in (*_FI_KEYS, "named_results", "figure_declarations")}
    if isinstance(protocol.get("metrics"), list):
        protocol["metrics"] = [{k: v for k, v in m.items() if k not in _ANSWER_KEYS} if isinstance(m, dict) else m
                               for m in protocol["metrics"]]
    for key in _ASKED_ROWS:
        if isinstance(protocol.get(key), list):
            rows = [r["as_planned"] if isinstance(r, dict) and isinstance(r.get("as_planned"), dict) else r
                    for r in protocol[key]]
            rows = [r for r in rows if not (isinstance(r, dict) and r.get("asked"))]
            if rows:
                protocol[key] = rows
            else:
                del protocol[key]
    return {**design, "protocol": protocol}


_FI_KEYS = ("form", "parts_left_out", "names_not_asked")
_ANSWER_KEYS = ("role", "depends_on", "returned")
_ASKED_ROWS = ("comparisons", "derived")


def settle_form(protocol: dict[str, Any], planned: dict[str, Any] | None = None) -> dict[str, Any]:
    """``protocol`` of a redrafted design with ``form`` worked out again: ``contract`` only when every result it now has
    carries a validated role and the parts the plan left out (``planned``'s ``parts_left_out``, other than the roles) are
    none; otherwise ``parts_left_out`` names what is missing and the plan is the older form. Nothing is asked."""
    out = {k: v for k, v in protocol.items() if k not in _FI_KEYS}
    metrics = [m for m in out.get("metrics") or [] if isinstance(m, dict)]
    left = [w for w in (planned or {}).get("parts_left_out") or [] if w != PART_WORDS["roles"]]
    if not metrics or any(m.get("role") not in _contract.ROLES or m.get("returned") not in ("each_run", "afterwards")
                          for m in metrics):
        left.insert(0, PART_WORDS["roles"])
    if left:
        out["parts_left_out"] = left
    else:
        out["form"] = "contract"
    return out


async def ask_parts(design: Any, ask: Callable[[str], Awaitable[str]], parse: Callable[[str], Any],
                    templates: dict[str, string.Template], *, fi_analysis: bool = False,
                    only: tuple[str, ...] | None = None,
                    topic_quantities: dict[str, dict[str, str]] | None = None) -> Outcome:
    """Ask the parts of ``design`` (a normalized design with metrics) and return the design with what could be used.
    ``ask`` sends a prompt and returns the reply; ``parse`` reads a reply into a mapping or ``None``. A part whose call
    raises is left out alone (its traceback is in ``Outcome.errors``); the parts already usable are kept. ``fi_analysis``:
    FI works out this plan's analysis itself, so each formula is held to what FI can work out (the ``formulas`` part) and
    each figure is declared (the ``figures`` part): FI draws only the figures the contract declares. ``only``: ask just these
    parts of a design that already holds the others (a redesign's figures); its form and the declarations it holds are kept."""
    out = Outcome(design=design)
    if not isinstance(design, dict) or not isinstance(design.get("protocol"), dict) or not design["protocol"].get("metrics"):
        return out
    current = copy.deepcopy(design) if only else strip_fi_keys(copy.deepcopy(design))
    not_asked: list[str] = []
    names = {} if only else _contract.undeclared_names(current, not_asked)
    derived_ids: list[str] = []
    order = ["roles"] + (["names"] if names else []) + (["formulas"] if fi_analysis else []) + ["comparisons"]
    if fi_analysis:
        order.append("figures")  # last: a figure may show a quantity the parts before it declared
    if topic_quantities:
        order.append("answers")
    if only:
        order = [x for x in order if x in only]
    for part in order:
        if part == "figures" and "roles" in out.parts and not out.parts["roles"]["valid"]:
            continue  # the plan stays in the older form, which FI does not write the analysis for
        faults = (_formula_faults(current) if part == "formulas" else figure_faults(current) if part == "figures"
                  else (topic_quantities or {}) if part == "answers" else {})
        if part in ("formulas", "figures", "answers") and not faults:
            continue  # every formula is one FI can work out, every figure is declared: nothing to ask
        record: dict[str, Any] = {"asks": 0, "valid": False, "problems": []}
        out.parts[part] = record
        problems: list[str] = []
        last = ""
        try:
            while record["asks"] < MAX_ASKS and not record["valid"]:
                record["asks"] += 1
                reply = await ask(_formulas_prompt(templates, current, faults, problems, last) if part == "formulas" else
                                  _figures_prompt(templates, current, faults, problems, last) if part == "figures" else
                                  _answers_prompt(templates, current, faults, problems, last) if part == "answers" else
                                  _prompt(part, templates, current, names if part == "names" else {}, problems, last, derived_ids))
                last = reply or ""
                got = parse(reply) if reply else None
                if not isinstance(got, dict):
                    problems = ["the answer was not one JSON object"]
                    continue
                value = got.get({"roles": "results", "names": "names", "comparisons": "comparisons", "formulas": "formulas",
                                "figures": "figures", "answers": "answers"}[part])
                if part == "comparisons":
                    value = {"comparisons": value if value is not None else [], "_derived": derived_ids}
                conflicts: dict[str, str] = {}  # answers: units that still differ at the last ask (not answered, with why)
                problems = (_check_formulas(value, faults, current) if part == "formulas" else
                            _check_figures(value, faults, current) if part == "figures" else
                            _check_answers(value, faults, current, conflicts if record["asks"] >= MAX_ASKS else None)
                            if part == "answers" else
                            _contract.check_part(part, value, current))
                if problems:
                    continue
                p = current["protocol"]
                if part == "roles":
                    kept, checked = _apply_roles(p, value), None
                elif part == "formulas":
                    kept, checked = _apply_formulas(p, value), None
                elif part == "figures":
                    kept, checked = _apply_figures(p, current, value, faults), None
                elif part == "answers":
                    kept, checked = _apply_answers(p, current, value, faults, conflicts), None
                elif part == "names":
                    kept, checked = _apply_names(p, value)
                else:
                    kept, checked = _apply_comparisons(p, value["comparisons"]), None
                fixed, why = _plan.normalize_protocol(checked if checked is not None else kept)
                if fixed is None:
                    problems = [str(why)]
                    continue
                if checked is not None:
                    fixed, why = _plan.normalize_protocol(kept)  # the plan as kept, without the temporary metrics
                    if fixed is None:
                        problems = [str(why)]
                        continue
                before = set(_contract.from_design(current, fi_analysis=fi_analysis).gaps)
                gaps = [g for g in _contract.from_design({**current, "protocol": {**fixed, "form": "contract"}},
                                                         fi_analysis=fi_analysis).gaps
                        if g not in before and ("could not be read" in g or "refers to something" in g or "left out" in g)]
                if part == "formulas":  # a formula answered is held to what FI can work out, also one written as before
                    still = _formula_faults({**current, "protocol": fixed})
                    gaps = [*(f"`{i}`: its formula still cannot be worked out ({why})" for i, why in still.items() if i in faults),
                            *(g for g in gaps if not any(f"`{i}` is left out" in g for i in faults))]
                if part == "figures":  # a figure answered is declared, not only named by its file
                    still = figure_faults({**current, "protocol": fixed})
                    gaps = [*(f"`{n}` is still named only by its file name" for n in still if n in faults), *gaps]
                if gaps:
                    problems = gaps
                    continue
                current = {**current, "protocol": fixed}
                if part == "names":
                    derived_ids = [n for n, r in value.items()
                                   if r["answer"] == "derived" or (r["answer"] == "reference" and r.get("formula"))]
                record["valid"] = True
                record["answer"] = value if part != "comparisons" else value["comparisons"]
        except Exception:  # noqa: BLE001 -- one part is left out; the others are kept
            record["valid"] = False
            record["error"] = traceback.format_exc()
            out.errors[part] = record["error"]
            problems = ["the call could not be made: " + record["error"].strip().splitlines()[-1][:200]]
        record["problems"] = problems
    left = [PART_WORDS[k] for k, v in out.parts.items() if not v["valid"] and k not in _OWN_PARTS]
    still = _formula_faults(current) if "formulas" in out.parts and not out.parts["formulas"]["valid"] else {}
    if still:
        out.notes.append("The plan's model was asked for a formula FI can work out for " + ", ".join(f"`{i}`" for i in still)
                         + ", and did not give one; " + ("it is" if len(still) == 1 else "they are")
                         + " left out of the study (the plan's gaps say why).")
    if "figures" in out.parts:
        asked = figure_faults(current) if not out.parts["figures"]["valid"] else {}
        if asked:
            out.notes.append("The plan's model was asked what " + ", ".join(f"`{n}`" for n in asked)
                             + (" shows" if len(asked) == 1 else " show") + ", and did not say; "
                             + ("it is" if len(asked) == 1 else "they are") + " left out of the figures FI draws (the plan's "
                             "gaps say why).")
        dropped = {n: r.get("why") for n, r in (out.parts["figures"].get("answer") or {}).items() if r.get("drop") is True}
        if dropped:
            out.notes.append("The plan's model said " + ", ".join(f"`{n}`" + (f" ({_short(w, 160)})" if w else "")
                                                                   for n, w in dropped.items())
                             + " cannot be drawn from the study's numbers; "
                             + ("it is" if len(dropped) == 1 else "they are") + " left out of the figures FI draws.")
    if only:  # the parts already there, and the form, stay as they were
        out.design = current
        return out
    protocol = {k: v for k, v in current["protocol"].items() if k not in _FI_KEYS}
    if not_asked:
        protocol["names_not_asked"] = not_asked
        out.notes.append(f"The plan uses {len(not_asked)} more name(s) than FI asked about ({', '.join(not_asked)}).")
    if left:
        protocol["parts_left_out"] = left
        out.notes.append("The plan's model was asked, and did not give in a form that could be used: " + "; ".join(left)
                         + ". The plan stays in the older form.")
    else:
        protocol["form"] = "contract"
    out.design = {**current, "protocol": protocol}
    return out
