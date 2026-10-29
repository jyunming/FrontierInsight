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
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import contract as _contract
from . import oracle_forms as _forms
from . import plan as _plan

#: The most times one part is asked: the first ask and two more.
MAX_ASKS = 3
PART_WORDS = {"roles": "what kind of number each result is, what it may change with and whether each run returns it",
              "names": "what the names the plan uses but does not declare are",
              "comparisons": "which results are compared between values of a setting"}
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
            derived.append({"id": name, "inputs": list(row["inputs"]), "role": "result", "label": what})
        elif row["answer"] == "reference" and isinstance(row.get("formula"), str) and row["formula"].strip():
            formula = row["formula"].strip()
            derived.append({"id": name, "inputs": [], "role": "reference", "formula": formula, "label": what})
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
                     **({"values": list(c["values"])} if c.get("values") else {}), "how": c.get("how", "difference")})
    return {**p, **({"comparisons": rows} if rows else {})}


def strip_fi_keys(design: Any) -> Any:
    """``design`` without the keys only FI writes (``form``, ``parts_left_out``, ``names_not_asked``): a model's own
    word for them is never believed."""
    if not isinstance(design, dict) or not isinstance(design.get("protocol"), dict):
        return design
    protocol = {k: v for k, v in design["protocol"].items() if k not in _FI_KEYS}
    return design if len(protocol) == len(design["protocol"]) else {**design, "protocol": protocol}


_FI_KEYS = ("form", "parts_left_out", "names_not_asked")


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
                    templates: dict[str, string.Template]) -> Outcome:
    """Ask the parts of ``design`` (a normalized design with metrics) and return the design with what could be used.
    ``ask`` sends a prompt and returns the reply; ``parse`` reads a reply into a mapping or ``None``. A part whose call
    raises is left out alone (its traceback is in ``Outcome.errors``); the parts already usable are kept."""
    out = Outcome(design=design)
    if not isinstance(design, dict) or not isinstance(design.get("protocol"), dict) or not design["protocol"].get("metrics"):
        return out
    current = strip_fi_keys(copy.deepcopy(design))
    not_asked: list[str] = []
    names = _contract.undeclared_names(current, not_asked)
    derived_ids: list[str] = []
    order = ["roles"] + (["names"] if names else []) + ["comparisons"]
    for part in order:
        record: dict[str, Any] = {"asks": 0, "valid": False, "problems": []}
        out.parts[part] = record
        problems: list[str] = []
        last = ""
        try:
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
                if part == "roles":
                    kept, checked = _apply_roles(p, value), None
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
                before = set(_contract.from_design(current).gaps)
                gaps = [g for g in _contract.from_design({**current, "protocol": {**fixed, "form": "contract"}}).gaps
                        if g not in before and ("could not be read" in g or "refers to something" in g or "left out" in g)]
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
    left = [PART_WORDS[k] for k, v in out.parts.items() if not v["valid"]]
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
