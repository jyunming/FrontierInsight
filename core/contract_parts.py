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

#: The most times one part is asked: the first ask and two more.
MAX_ASKS = 3
PART_WORDS = {"roles": "what kind of number each result is, what it may change with and whether each run returns it",
              "names": "what the names the plan uses but does not declare are",
              "comparisons": "which results are compared between values of a setting",
              "formulas": "a formula FI can work out for each quantity the plan works out with one",
              "figures": "what each figure the plan names shows"}
#: The parts whose answer the plan's form does not wait on: a formula still not workable, or a figure still not declared,
#: is left out alone.
_OWN_PARTS = ("formulas", "figures")
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


def _figure_holds(design: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """``(lines, settings, single)``: what a figure may show, one line each (a reported number or a quantity worked out from
    them: the analysis's table serves nothing else), the settings with more than one value, which a figure may use as ``x`` or
    ``series``, and the settings that have one."""
    c = _contract.from_design({**design, "protocol": {**design["protocol"], "form": "contract"}}, fi_analysis=True)
    lines = [f"- {r.id}: {_short(r.estimand or r.label, 160) or 'a reported number'}" for r in c.results]
    lines += [f"- {d.id}: {_short(d.label, 160) or 'worked out from the reported numbers'}" for d in c.derived]
    values = {**c.settings.compared, **c.settings.in_each_run}
    return (lines, sorted(k for k in c.settings.varied() if len(values.get(k, ())) > 1),
            sorted(k for k in c.settings.varied() if len(values.get(k, ())) <= 1))


def _figures_prompt(templates: dict[str, string.Template], design: dict[str, Any], faults: dict[str, str],
                    problems: list[str], last: str) -> str:
    lines, settings, _single = _figure_holds(design)
    return templates["plan_figures"].substitute(
        figures="\n".join(f"- {name}: {text}" for name, text in faults.items()), shows="\n".join(lines) or "- (nothing)",
        settings=", ".join(settings) or "(the study varies no setting)", max_panels=_contract.MAX_PANELS,
        method=_short(design.get("method")), again=_problems_text(problems, last),
        record="\n".join(_skeleton.table_reads()))


def _check_figures(value: Any, asked: dict[str, str], design: dict[str, Any]) -> list[str]:
    """Why ``value``, the answer for ``figures``, cannot be used; empty when it can. Each figure asked about gets what it
    shows (a reported number or a quantity worked out from them), its ``x`` setting, a ``series`` setting other than ``x`` (or
    none) and its panels, or ``drop`` with why. A setting must have more than one value to be drawn along."""
    if not isinstance(value, dict):
        return ["the answer for `figures` must be a mapping"]
    lines, settings, single = _figure_holds(design)
    ids = [line[2:].split(":")[0] for line in lines]
    out: list[str] = []
    for name in asked:
        row = value.get(name)
        if not isinstance(row, dict):
            out.append(f"`{name}` is not answered")
            continue
        if row.get("drop") is True:
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
    return {**p, "figure_declarations": decls}


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
                    only: tuple[str, ...] | None = None) -> Outcome:
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
    if only:
        order = [x for x in order if x in only]
    for part in order:
        if part == "figures" and "roles" in out.parts and not out.parts["roles"]["valid"]:
            continue  # the plan stays in the older form, which FI does not write the analysis for
        faults = _formula_faults(current) if part == "formulas" else figure_faults(current) if part == "figures" else {}
        if part in ("formulas", "figures") and not faults:
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
                                  _prompt(part, templates, current, names if part == "names" else {}, problems, last, derived_ids))
                last = reply or ""
                got = parse(reply) if reply else None
                if not isinstance(got, dict):
                    problems = ["the answer was not one JSON object"]
                    continue
                value = got.get({"roles": "results", "names": "names", "comparisons": "comparisons", "formulas": "formulas",
                                "figures": "figures"}[part])
                if part == "comparisons":
                    value = {"comparisons": value if value is not None else [], "_derived": derived_ids}
                problems = (_check_formulas(value, faults, current) if part == "formulas" else
                            _check_figures(value, faults, current) if part == "figures" else
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
