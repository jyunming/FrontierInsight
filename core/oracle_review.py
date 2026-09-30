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
from dataclasses import dataclass, field
from typing import Any

from . import oracle_check as _oracle

#: The node a model is named for in ``provider.node_models``.
NODE = "oracle_review"


def _yes(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    word = str(value or "").strip().lower()
    if word in ("yes", "y", "true"):
        return True
    if word in ("no", "n", "false"):
        return False
    return None


def _text(value: Any, limit: int = 400) -> str:
    return " ".join(str(value or "").replace("`", "'").split())[:limit]


@dataclass
class Review:
    """What the reviewer said, read strictly: only checks the plan declares, only yes/no it gave."""
    checks: list[dict[str, Any]] = field(default_factory=list)
    tested: list[str] = field(default_factory=list)
    untested: list[str] = field(default_factory=list)
    add: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""


def prompt_parts(protocol: dict[str, Any] | None) -> tuple[str, str]:
    """``(model, oracles)`` as the prompt shows them (JSON text)."""
    protocol = protocol if isinstance(protocol, dict) else {}
    model = protocol.get("model")
    return (json.dumps(model if model is not None else "(the plan does not say what model produces the numbers)",
                       indent=2, default=str)[:8000],
            json.dumps(_oracle.declared(protocol), indent=2, default=str)[:12000])


def parse(reply: Any, protocol: dict[str, Any] | None) -> Review | None:
    """The reviewer's answer, or ``None`` when it cannot be read as one: a mapping whose ``checks`` name at least one
    check the plan declares. Anything about a check the plan does not declare is dropped, never guessed at."""
    if not isinstance(reply, dict) or not isinstance(reply.get("checks"), list):
        return None
    declared = {str(o["name"]).strip().lower(): str(o["name"]).strip() for o in _oracle.declared(protocol)}
    out = Review()
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
    ids = {str(e.get("id")).strip().upper(): str(e.get("id")).strip()
           for e in (model.get("equations") or [] if isinstance(model, dict) else []) if isinstance(e, dict) and e.get("id")}
    for key, target in (("equations_tested", out.tested), ("equations_not_tested", out.untested)):
        for eid in reply.get(key) if isinstance(reply.get(key), list) else []:
            known = ids.get(str(eid).strip().upper())
            if known and known not in target:
                target.append(known)
    for item in (reply.get("add") if isinstance(reply.get("add"), list) else [])[:2]:
        if isinstance(item, dict) and str(item.get("name") or "").strip():
            out.add.append(item)
    out.summary = _text(reply.get("summary"), 600)
    return out


def findings(review: Review) -> list[str]:
    """One plain sentence per thing the reviewer found that the plan should look at (empty when it found nothing)."""
    out: list[str] = []
    for c in review.checks:
        name = repr(c["name"])
        if c["appropriate"] is False:
            out.append(f"the check {name} may not test the model: {c['why'] or 'no reason given'}")
        if c["discriminating"] is False:
            out.append(f"the check {name} would not fail on a plausible bug: {c['bug'] or 'no bug named'}")
        if c["well_defined"] is False:
            out.append(f"the number of the check {name} is not well defined: {c['definition_note'] or 'no reason given'}")
        if c["better"]:
            out.append(f"for the check {name}, a better or additional check: {c['better']}")
    if review.untested:
        out.append(f"no check tests equation{'s' if len(review.untested) > 1 else ''} {', '.join(review.untested)} "
                   "of the model behind the numbers")
    for item in review.add:
        out.append(f"a check to add: {json.dumps(item, default=str)[:600]}")
    return out


def request(found: list[str]) -> str:
    """The part of the one request to the plan that carries the reviewer's findings."""
    return (
        "A second reader looked at the plan's checks against known answers and found:\n"
        + "\n".join(f"- {f[0].upper()}{f[1:]}." for f in found)
        + "\nFor each, change the check if the finding is right (a better case, how its number is computed, its expected "
        "value worked out at the step it is compared at, a reachable tolerance, or one more check for an equation no "
        "check tests, with a `reference` saying how its expected value follows), and leave it as it is if not. Keep each "
        "kind's numeric form. Change only the checks, and nothing else in the plan."
    )


def plan_lines(review: Review | None, *, reviewer: str, planner: str, same_model: bool, research: bool,
               error: str = "") -> list[str]:
    """What plan.md says about the second opinion, in plain words (never read back)."""
    who = (f"The checks were read by a second model ({reviewer}), not the one that wrote the plan ({planner})."
           if not same_model else
           f"The checks were read again by the model that wrote the plan ({planner}): a second look, not a second "
           "opinion" + (". This quest is set up for research, where a different model should read them: set one "
                        "under `provider: node_models: oracle_review: <another model your provider offers>`."
                        if research else "."))
    if review is None:
        return ["", f"- {who} Its answer could not be used ({error or 'it named none of the plan checks'}), so nothing "
                "was changed for it."]
    rows = ["", f"- {who}" + (f" Its summary: {review.summary}" if review.summary else "")]
    for c in review.checks:
        verdict = []
        verdict.append({True: "tests the model", False: "may not test the model", None: "no verdict on what it tests"}[c["appropriate"]])
        verdict.append({True: "would catch a plausible bug", False: "would miss a plausible bug",
                        None: "no verdict on what it catches"}[c["discriminating"]])
        verdict.append({True: "its number is well defined", False: "its number is not well defined",
                        None: "no verdict on its number"}[c["well_defined"]])
        extra = "; ".join(x for x in (c["definition_note"], c["better"] and f"better: {c['better']}") if x)
        rows.append(f"  - **{c['name']}**: {', '.join(verdict)}." + (f" {extra}." if extra else ""))
    if review.tested or review.untested:
        rows.append(f"  - Equations the checks test: {', '.join(review.tested) or 'none named'}; not tested by any: "
                    f"{', '.join(review.untested) or 'none'}.")
    return rows
