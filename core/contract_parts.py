"""The plan's model is asked, one part at a time, for what the study contract needs and the first plan call does not give.

The first plan call writes the settings and the list of results. Three more questions follow, each a plain one, each
checked against the ids FI already holds (:func:`core.contract.check_part`), and each asked again alone when its answer
cannot be used (at most :data:`MAX_ASKS` asks of a part in all):

* ``roles``: for every declared result, what kind of number it is and which settings it changes with;
* ``names``: for every name the plan relies on and does not declare (:func:`core.contract.undeclared_names`), whether it
  is a result, a reference value, a diagnostic, or computed afterwards from the results;
* ``comparisons``: which results the study compares between values of one setting.

When every part is usable FI writes ``form: contract`` into the protocol; the model never does. A part that stays
unusable is left out, named in ``protocol.parts_left_out`` (the contract reads that into its gaps), and the plan stays in
the older form. Nothing here stops a quest.
"""

from __future__ import annotations

import copy
import json
import string
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import contract as _contract
from . import oracle_forms as _forms
from . import plan as _plan

#: The most times one part is asked: the first ask and two more.
MAX_ASKS = 3
PART_WORDS = {"roles": "what kind of number each result is and which settings it changes with",
              "names": "what the names the plan uses but does not declare are",
              "comparisons": "which results are compared between values of a setting"}
_ROLE_HINT = {"reference": "its plan gives a formula, so it is a reference", "diagnostic": "a check of the plan measures it as a "
              "size of disagreement, so it is a diagnostic"}


@dataclass
class Outcome:
    design: Any
    parts: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def form(self) -> str:
        return "contract" if self.parts and all(p["valid"] for p in self.parts.values()) else "older"


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


def _prompt(part: str, templates: dict[str, string.Template], design: dict[str, Any], names: dict[str, str],
            problems: list[str], last: str, derived: list[str]) -> str:
    p = design.get("protocol") or {}
    return templates[f"plan_{part}"].substitute(
        method=_short(design.get("method")), settings=_settings_lines(p), results=_results_lines(p),
        names=_names_lines(names), equations=", ".join(sorted(_contract._model_equation_ids(p))) or "(none)",
        derived=", ".join(derived) or "(none)", again=_problems_text(problems, last),
        formula_language=_forms.EXPECTED_FORMULA_LANGUAGE)


def _apply_roles(p: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    _by, violation = _contract.oracle_measures(p)
    metrics = []
    for m in p.get("metrics") or []:
        row = answer.get(str(m.get("id"))) if isinstance(m, dict) else None
        if isinstance(row, dict):
            role = _contract.structural_role(m, violation) or row["role"]
            m = {**m, "role": role, "depends_on": list(row["depends_on"])}
        metrics.append(m)
    return {**p, "metrics": metrics}


def _apply_names(p: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    metrics, derived = list(p.get("metrics") or []), list(p.get("derived") or [])
    for name, row in answer.items():
        what = _short(row.get("what") or name, 300)
        unit = _short(row.get("unit") or "one run", 120)
        if row["answer"] in ("result", "reference"):
            entry: dict[str, Any] = {"id": name, "kind": "mean", "estimand": what, "unit": unit, "role": row["answer"],
                                     "depends_on": [s for s in row.get("depends_on") or [] if s in _contract.varied_settings(p)]}
            if row["answer"] == "reference":
                if isinstance(row.get("formula"), str) and row["formula"].strip():
                    entry["reference"] = {"formula": row["formula"].strip(), "kind": "value"}
                else:
                    entry["equation"] = str(row["equation"]).strip()
            metrics.append(entry)
        elif row["answer"] == "derived":
            derived.append({"id": name, "inputs": list(row["inputs"]), "role": "result", "label": what})
    return {**p, "metrics": metrics, **({"derived": derived} if derived else {})}


def _apply_comparisons(p: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [{"id": f"{c['result']}_by_{c['across']}", "result": c["result"], "across": c["across"],
             **({"values": list(c["values"])} if c.get("values") else {}), "how": c.get("how", "difference")} for c in items]
    return {**p, **({"comparisons": rows} if rows else {})}


async def ask_parts(design: Any, ask: Callable[[str], Awaitable[str]], parse: Callable[[str], Any],
                    templates: dict[str, string.Template]) -> Outcome:
    """Ask the parts of ``design`` (a normalized design with metrics) and return the design with what could be used.
    ``ask`` sends a prompt and returns the reply; ``parse`` reads a reply into a mapping or ``None``."""
    out = Outcome(design=design)
    if not isinstance(design, dict) or not isinstance(design.get("protocol"), dict) or not design["protocol"].get("metrics"):
        return out
    current = copy.deepcopy(design)
    names = _contract.undeclared_names(current)
    derived_ids: list[str] = []
    order = ["roles"] + (["names"] if names else []) + ["comparisons"]
    for part in order:
        record: dict[str, Any] = {"asks": 0, "valid": False, "problems": []}
        out.parts[part] = record
        problems: list[str] = []
        last = ""
        while record["asks"] < MAX_ASKS and not record["valid"]:
            record["asks"] += 1
            reply = await ask(_prompt(part, templates, current, names if part == "names" else {}, problems, last, derived_ids))
            last = reply or ""
            got = parse(reply) if reply else None
            if not isinstance(got, dict):
                problems = ["the answer was not one JSON object"]
                continue
            value = got.get({"roles": "results", "names": "names", "comparisons": "comparisons"}[part])
            if part == "comparisons":
                value = {"comparisons": value if value is not None else [], "_derived": derived_ids}
            problems = _contract.check_part(part, value, current)
            if problems:
                continue
            p = current["protocol"]
            trial = {"roles": _apply_roles, "names": _apply_names}.get(part, lambda q, v: _apply_comparisons(q, v["comparisons"]))(p, value)
            fixed, why = _plan.normalize_protocol(trial)
            if fixed is None:
                problems = [str(why)]
                continue
            before = set(_contract.from_design(current).gaps)
            gaps = [g for g in _contract.from_design({**current, "protocol": {**fixed, "form": "contract"}}).gaps
                    if g not in before and ("could not be read" in g or "refers to something" in g or "left out" in g)]
            if gaps:
                problems = gaps
                continue
            current = {**current, "protocol": fixed}
            if part == "names":
                derived_ids = [n for n, r in value.items() if r["answer"] == "derived"]
            record["valid"] = True
            record["answer"] = value if part != "comparisons" else value["comparisons"]
        record["problems"] = problems
    left = [PART_WORDS[k] for k, v in out.parts.items() if not v["valid"]]
    protocol = dict(current["protocol"])
    if left:
        protocol["parts_left_out"] = left
        out.notes.append("The plan's model was asked, and did not give in a form that could be used: " + "; ".join(left)
                         + ". The plan stays in the older form.")
    else:
        protocol["form"] = "contract"
    out.design = {**current, "protocol": protocol}
    return out
