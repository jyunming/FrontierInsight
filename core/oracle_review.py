"""A second opinion on the plan's checks against known answers, asked of a model acting as a referee.

A check is only as good as its design: one that tests an identity the code satisfies whatever it does, or whose number
means something other than the plan thinks, passes a wrong simulation. The person asked for exactly this: consult a
model on whether each check is appropriate, or whether there is a better one. So at plan time, before anything runs
(:meth:`core.engine.Engine._guide_oracles`), one call (``agents/oracle_review.md``, node ``oracle_review``) reads the plan's
*model behind the numbers* and every check, and says per check whether it tests the model (**appropriate**), whether a
plausible bug would make it fail (**discriminating**), whether its number is **well defined** (units, representation,
the value at a finite step, a reachable tolerance), a **better** check if there is one, and which of the model's
equations no check tests. Its findings go back to the plan once, together with any check not in its kind's form
(:mod:`core.oracle_forms`), and ``plan.md`` says in plain words what the reviewer found and what changed.

Under ``rigor_profile: research`` the reviewer should be a different model from the one that wrote the plan: the
``provider.node_models`` entry ``oracle_review``, as the review panel's personas are given theirs. When none is set,
or it is the planner's own model, the review is still made and the plan says it was the same model's second look.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from . import oracle_check as _oracle

#: The node a model is named for in ``provider.node_models``.
NODE = "oracle_review"
#: How much of the plan the reviewer is shown (whole checks only: a check cut in half is not shown at all).
_MAX_MODEL_CHARS, _MAX_CHECK_CHARS = 8000, 12000
#: The keys a proposed check may carry (as the plan writes one).
_ADD_KEYS = ("name", "kind", "check", "expected", "tolerance", "tolerance_mode", "reference", "case", "measure")
#: What a reviewer writes for "nothing": not a finding.
_NOTHING = {"", "none", "n/a", "na", "-", "no", "nothing", "empty", "null", "not applicable", "none needed"}


def _yes(value: Any) -> bool | None:
    """``True``/``False`` for a yes or no, also one the reviewer qualified ("no, mostly"); ``None`` otherwise."""
    if isinstance(value, bool):
        return value
    word = str(value or "").lower()
    if (" ".join(word.split()).rstrip(".") in _NOTHING - {"no"}
            or re.match(r"\s*(no (verdict|opinion|answer|idea|judgement|judgment)|yes\s*/\s*no|no\s*/\s*yes|n\s*/\s*a\b"
                        r"|n\.a\.)", word)):
        return None  # "N/A", "none", "no verdict", "yes/no": no answer given
    match = re.match(r"\s*(yes|no|true|false|y|n)(?![/\w])", word)
    return None if match is None else match.group(1) in ("yes", "true", "y")


def _text(value: Any, limit: int = 400) -> str:
    """One line, no backtick, bounded; empty for a reviewer's way of saying nothing."""
    text = " ".join(str(value if value is not None else "").replace("`", "'").split())[:limit]
    return "" if text.lower().rstrip(".") in _NOTHING else text


@dataclass
class Review:
    """What the reviewer said, read strictly: only checks the plan declares, only yes/no it gave."""
    checks: list[dict[str, Any]] = field(default_factory=list)
    tested: list[str] = field(default_factory=list)
    untested: list[str] = field(default_factory=list)
    add: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    #: The reviewer was shown only part of the plan's checks (so what it says no check tests is not read).
    partial: bool = False


def prompt_parts(protocol: dict[str, Any] | None) -> tuple[str, str, bool]:
    """``(model, checks, partial)`` as the prompt shows them (JSON text): whole checks only, and ``partial`` when some
    did not fit (the text then says how many are not shown)."""
    protocol = protocol if isinstance(protocol, dict) else {}
    model = protocol.get("model")
    model_text = json.dumps(model if model is not None else "(the plan does not say what model produces the numbers)",
                            indent=2, default=str)
    if len(model_text) > _MAX_MODEL_CHARS:
        model_text = model_text[:_MAX_MODEL_CHARS] + "\n(the rest of the model is not shown)"
    shown: list[dict[str, Any]] = []
    checks = _oracle.declared(protocol)
    for oracle in checks:
        if len(json.dumps([*shown, oracle], indent=2, default=str)) > _MAX_CHECK_CHARS:
            break
        shown.append(oracle)
    text = json.dumps(shown, indent=2, default=str)
    partial = len(shown) < len(checks)
    if partial:
        text += f"\n({len(checks) - len(shown)} more checks are not shown)"
    return model_text, text, partial


def _add_item(item: Any, declared: set[str]) -> dict[str, Any] | None:
    """A proposed check with only the keys a plan's check has, text bounded and numbers numbers; ``None`` for one
    with no name or the name of a check the plan already has."""
    if not isinstance(item, dict):
        return None
    name = _text(item.get("name"), 80)
    if not name or name.lower() in declared:
        return None
    out: dict[str, Any] = {}
    if _oracle.kind_of(item) is None and item.get("kind") is not None:
        return None  # a kind that is none of the six
    if str(item.get("tolerance_mode") or "absolute").strip().lower() not in ("absolute", "relative"):
        return None
    for key in _ADD_KEYS:
        value = item.get(key)
        if value is None:
            continue
        if key in ("expected", "tolerance"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return None
            out[key] = value
        elif key == "case":
            if isinstance(value, dict):
                out[key] = {" ".join(str(k).replace("`", "'").split())[:60]: v for k, v in list(value.items())[:10]
                            if isinstance(v, (int, float, str)) and not isinstance(v, bool)}
        else:
            out[key] = _text(value, 400)
    out["name"] = name
    return out


def parse(reply: Any, protocol: dict[str, Any] | None, *, partial: bool = False) -> Review | None:
    """The reviewer's answer, or ``None`` when it cannot be read as one: a mapping whose ``checks`` name at least one
    check the plan declares. Anything about a check the plan does not declare is dropped, never guessed at."""
    if not isinstance(reply, dict) or not isinstance(reply.get("checks"), list):
        return None
    declared = {str(o["name"]).strip().lower(): str(o["name"]).strip() for o in _oracle.declared(protocol)}
    out = Review(partial=partial)
    for item in reply["checks"]:
        if not isinstance(item, dict):
            continue
        name = declared.get(str(item.get("name") or "").strip().lower())
        if name is None or any(c["name"] == name for c in out.checks):
            continue
        out.checks.append({
            "name": name, "appropriate": _yes(item.get("appropriate")), "why": _text(item.get("why")),
            "discriminating": _yes(item.get("discriminating")), "bug": _text(item.get("bug_it_would_catch")),
            "well_defined": _yes(item.get("well_defined")), "definition_note": _text(item.get("definition_note")),
            "better": _text(item.get("better")),
        })
    if not out.checks:
        return None
    model = protocol.get("model") if isinstance(protocol, dict) else None
    equations = model.get("equations") if isinstance(model, dict) else None
    ids = {str(e.get("id")).strip().upper(): str(e.get("id")).strip()
           for e in (equations if isinstance(equations, list) else []) if isinstance(e, dict) and e.get("id")}
    for key, target in (("equations_tested", out.tested), ("equations_not_tested", out.untested)):
        if key == "equations_not_tested" and partial:
            continue  # the reviewer did not see every check, so it cannot say what none of them tests
        for eid in reply.get(key) if isinstance(reply.get(key), list) else []:
            known = ids.get(str(eid).strip().upper())
            if known and known not in target:
                target.append(known)
    for item in reply.get("add") if isinstance(reply.get("add"), list) else []:
        kept = _add_item(item, set(declared) | {a["name"].lower() for a in out.add})
        if kept is not None and len(out.add) < 2:
            out.add.append(kept)
    out.summary = _text(reply.get("summary"), 600)
    return out


def findings(review: Review) -> list[str]:
    """One plain sentence per thing the plan should change: a check found not to test the model, to miss a plausible
    bug or to have an ill-defined number, an equation no check tests, a check to add. A better check alone is a
    suggestion, shown in plan.md, not a reason to rewrite the plan."""
    out: list[str] = []
    for c in review.checks:
        name = repr(c["name"])
        better = f" (a better check: {c['better']})" if c["better"] else ""
        if c["appropriate"] is False:
            out.append(f"the check {name} may not test the model: {c['why'] or 'no reason given'}{better}")
        if c["discriminating"] is False:
            out.append(f"the check {name} would not fail on a plausible bug: {c['bug'] or 'no bug named'}{better}")
        if c["well_defined"] is False:
            out.append(f"the number of the check {name} is not well defined: {c['definition_note'] or 'no reason given'}")
    if review.untested:
        out.append(f"no check tests equation{'s' if len(review.untested) > 1 else ''} {', '.join(review.untested)} "
                   "of the model behind the numbers")
    for item in review.add:
        out.append(f"a check to add: {json.dumps(item, default=str)}")
    return out


def request(found: list[str]) -> str:
    """The part of the one request to the plan that carries the reviewer's findings. The reviewer is a second reader,
    not the person: its findings are applied only when they are right."""
    return (
        "A second model read the plan's checks against known answers as a referee would (it is not the person; apply "
        "what it found only where it is right) and found:\n"
        + "\n".join(f"- {f[0].upper()}{f[1:]}." for f in found)
        + "\nFor each finding that is right, change the check: a better case, how its number is computed, its expected "
        "value worked out at the step it is compared at, a reachable tolerance, or one more check for an equation no "
        "check tests, with a `reference` saying how its expected value follows. Replace a check found trivial with a "
        "better one; never remove a check without putting one in its place. Keep each kind's numeric form. Change only "
        "the checks (and a criterion that reads a check you change, when its number changes meaning), and nothing else "
        "in the plan."
    )


def plan_lines(review: Review | None, *, reviewer: str, planner: str, same_model: bool | None, research: bool,
               reported: bool = True, error: str = "", sent: bool = True) -> list[str]:
    """What plan.md says about the second opinion, in plain words (never read back). ``same_model`` is ``None`` when
    it cannot be told whether the two were different models (a provider default FI cannot name)."""
    asked = "" if reported else " (the connection did not say which model answered)"
    if same_model is False:
        who = f"The checks were read by a second model ({reviewer}{asked}), not the one that wrote the plan ({planner})."
    elif same_model is None:
        who = (f"The checks were read by {reviewer}{asked}; the plan was written by {planner}, and FI cannot tell "
               "whether the two are different models.")
    else:
        who = (f"The checks were read again by the model that wrote the plan ({planner}{asked}): a second look, not a "
               "second opinion" + (". This quest is set up for research, where a different model should read them: set "
                                   "one under `provider: node_models: oracle_review: <another model your provider "
                                   "offers>`." if research else "."))
    if review is None and not sent:
        return ["", f"- No second reader looked at the checks ({error or 'nothing to show'}), so nothing was changed "
                "for it."]
    if review is None:
        return ["", f"- The checks were sent to {reviewer} for a second reading, but its answer could not be used "
                f"({error or 'it named none of the plan checks'}), so nothing was changed for it."]
    rows = ["", f"- {who}" + (f" Its summary: {review.summary}" if review.summary else "")]
    for c in review.checks:
        verdict = [
            {True: "tests the model", False: "may not test the model", None: "no verdict on what it tests"}[c["appropriate"]],
            {True: "would catch a plausible bug", False: "would miss a plausible bug",
             None: "no verdict on what it catches"}[c["discriminating"]],
            {True: "its number is well defined", False: "its number is not well defined",
             None: "no verdict on its number"}[c["well_defined"]],
        ]
        extra = "; ".join(x for x in (c["definition_note"], c["better"] and f"a better check: {c['better']}") if x)
        rows.append(f"  - **{c['name']}**: {', '.join(verdict)}." + (f" {extra}." if extra else ""))
    if review.tested or review.untested:
        rows.append(f"  - Equations the checks test: {', '.join(review.tested) or 'none named'}"
                    + ("." if review.partial else f"; not tested by any: {', '.join(review.untested) or 'none'}."))
    if review.partial:
        rows.append("  - The plan has more checks than the reader was shown; it did not judge the rest.")
    return rows
