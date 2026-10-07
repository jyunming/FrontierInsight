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

Under research the review is also a condition of the evidence level ``independently_validated``
(:func:`independence_gaps`): it counts only when it gave a usable answer AND the quest's own record of its model calls
(``.fi/model_calls.jsonl``) shows that the model that answered it is not a model that wrote the plan's checks (the
models as the connection named them on each call, never the config's wish; the same model under another name, a dated
id or another provider counts as the same). Anything else is a plain gap. A review that could not be had still never
stops the quest.
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
#: The steps whose answers write the plan's checks: the plan, every rewrite of it (FI's own requests about the checks
#: included), the design step of a quest with no plan.md, and the design's own critique (which may amend the checks).
WRITER_NODES = ("plan", "plan_revise", "design", "design_self_critique")
#: How a person gives the review another model (the key that already names a step's model; nothing new).
HOW_TO_NAME_ANOTHER = ("set `oracle_review: <another model your provider offers>` under `provider: node_models:` in "
                       "the quest's config.yaml (or give it another provider's model under `provider: node_providers:`); it reads the checks when the plan is written (for this quest, do the "
                       "plan again: `--resume <quest_id> --from plan --approve-as <you>`)")
#: The gap's first words, the same whatever the reason.
NOT_REVIEWED = "the plan's checks were not reviewed by a second, different model"
#: How much of the plan the reviewer is shown (whole checks only: a check cut in half is not shown at all).
_MAX_MODEL_CHARS, _MAX_CHECK_CHARS = 8000, 12000
#: The keys a proposed check may carry (as the plan writes one).
_ADD_KEYS = ("name", "kind", "check", "expected", "expected_formula", "tolerance", "tolerance_mode", "reference", "case", "measure")
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
    #: What the reviewer said about each equation of the model (``protocol.model.equations``): ``{"id", "formula",
    #: "standard" (True / False / None), "correct", "why"}``, only for equations the plan lists. Whether these count is
    #: decided by who the reviewer was (:func:`equation_findings` is used only when it was another model).
    equations: list[dict[str, Any]] = field(default_factory=list)


def prompt_parts(protocol: dict[str, Any] | None) -> tuple[str, str, bool]:
    """``(model, checks, partial)`` as the prompt shows them (JSON text): whole checks only, and ``partial`` when some
    did not fit (the text then says how many are not shown)."""
    protocol = protocol if isinstance(protocol, dict) else {}
    model = protocol.get("model")
    model_text = json.dumps(model if model is not None else "(the plan does not say what model produces the numbers)",
                            indent=2, default=str)
    if len(model_text) > _MAX_MODEL_CHARS:
        model_text = model_text[:_MAX_MODEL_CHARS] + "\n(the rest of the model is not shown)"
    checks = _oracle.declared(protocol)
    shown = _shown(protocol)
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
            "isolates": _yes(item.get("isolates")), "isolation_note": _text(item.get("isolation_note")),
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
    out.equations = _equation_verdicts(reply.get("equations"), model)
    return out


def equations_of(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The equations of the plan's model that a reader is asked about: every one that has an id and a formula."""
    model = protocol.get("model") if isinstance(protocol, dict) else None
    view = _oracle.model_view(model)
    return [e for e in (view["equations"] if view else []) if isinstance(e, dict) and str(e.get("id") or "").strip()
            and str(e.get("formula") or "").strip()]


def _equation_verdicts(value: Any, model: Any) -> list[dict[str, Any]]:
    """The reviewer's per-equation verdicts, only for equations the plan lists (one verdict each, the first)."""
    view = _oracle.model_view(model)
    known = {str(e.get("id")).strip().upper(): e for e in (view["equations"] if view else [])
             if isinstance(e, dict) and str(e.get("id") or "").strip()}
    out: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        eq = known.get(str(item.get("id") or "").strip().upper())
        if eq is None or any(v["id"] == str(eq["id"]).strip() for v in out):
            continue
        standard = _yes(item.get("standard"))
        out.append({"id": str(eq["id"]).strip(), "formula": _text(eq.get("formula"), 300), "standard": standard,
                    "correct": _text(item.get("correct"), 400) if standard is False else "",
                    "why": _text(item.get("why"), 300)})
    return out


def equation_findings(review: Review) -> list[str]:
    """One plain sentence per equation the reader says is not the standard definition of the quantity it names (or a
    correct derivation), with the equation it gives instead. Used only when the reader is another model than the
    planner (:func:`equation_lines` says what is recorded otherwise)."""
    out: list[str] = []
    for e in review.equations:
        if e["standard"] is False:
            out.append(f"the equation {e['id']} ({e['formula']}) is not the standard definition of the quantity it "
                       f"names, or a correct derivation: the reader gives {e['correct'] or 'no replacement'} "
                       f"({e['why'] or 'no reason given'})")
    return out


def isolation_findings(review: Review) -> list[str]:
    """One plain sentence per check whose ``case`` the reader says does not isolate what its ``measure`` claims. Used
    only when the reader is another model than the planner (:func:`isolation_record` says what is recorded otherwise)."""
    return [f"the case of the check {c['name']!r} does not isolate what its measure claims: "
            f"{c['isolation_note'] or 'no reason given'}" for c in review.checks if c["isolates"] is False]


def isolation_record(review: Review | None, *, same_model: bool | None, reviewer: str, planner: str) -> dict[str, Any]:
    """Whether the checks' cases were judged by a model other than the planner: ``{"read_by_other_model", "why",
    "judged", "objections"}``. A reader that is the planner's own model, or that FI cannot tell apart from it, or that
    gave no verdict on a check, does not count; ``why`` says so."""
    judged = [c["name"] for c in (review.checks if review else []) if c["isolates"] is not None]
    if review is None:
        why = "the second reading gave no usable answer"
    elif same_model is not False:
        why = (f"the reader is the model that wrote the plan ({planner})" if same_model
               else f"FI cannot tell whether the reader ({reviewer}) is another model than the one that wrote the plan")
    elif len(judged) < len(review.checks):
        why = "the reader gave no verdict on every check's case"
    else:
        why = ""
    return {"read_by_other_model": not why, "why": why, "judged": judged,
            "objections": len(isolation_findings(review)) if review and same_model is False else 0}


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


#: Added to the request when the reader objected to an equation: the plan rewrites the equation and what depends on it.
EQUATION_REQUEST = (
    "\nWhere the reader objected to an equation: decide whether the equation it gives is the standard definition (or "
    "derive it again from the source); if it is, rewrite that equation in `protocol.model.equations` (its `formula`, "
    "`source` and `derivation`), and rewrite every check, `expected_formula`, `expected` or `example` that depends on "
    "it so they agree with the corrected equation. Never change a tolerance or a threshold, and never remove an "
    "equation. If the plan's equation is the right one, leave it and say why in its `derivation`.")


def never_looser(old: list[Any], new: list[Any], *, keep_expected: bool = False) -> list[Any]:
    """``new`` (the checks after a rewrite) with each check that also existed in ``old`` (matched by name) keeping its
    old ``tolerance`` and ``tolerance_mode`` when the rewrite made it larger or changed its mode: a rewrite of an
    equation may move an expected value, never loosen the bar a check is held to. ``keep_expected``: the rewrite was
    asked to change a check's case or measure only, so each check keeps the ``expected`` it had; a changed
    ``expected_formula`` still moves it, through FI's own computation before the run."""
    before = {str(o.get("name") or "").strip().lower(): o for o in old if isinstance(o, dict)}
    out: list[Any] = []
    for item in new:
        was = before.get(str(item.get("name") or "").strip().lower()) if isinstance(item, dict) else None
        if was is None or not isinstance(item, dict):
            out.append(item)
            continue
        old_mode = str(was.get("tolerance_mode") or "absolute").strip().lower()
        new_mode = str(item.get("tolerance_mode") or "absolute").strip().lower()
        try:
            looser = (float(item.get("tolerance")) > float(was.get("tolerance"))
                      if "tolerance" in item and "tolerance" in was else "tolerance" in item and "tolerance" not in was)
        except (TypeError, ValueError):
            looser = True
        if looser or new_mode != old_mode:
            item = {k: v for k, v in item.items() if k not in ("tolerance", "tolerance_mode")}
            for key in ("tolerance", "tolerance_mode"):
                if key in was:
                    item[key] = was[key]
        if keep_expected and "expected" in was and item.get("expected") != was["expected"]:
            item = {**item, "expected": was["expected"]}
        out.append(item)
    return out


def keep_every_equation(old: Any, new: Any, *, only_example: bool = False) -> list[Any]:
    """The equations after a rewrite that may correct one: every equation of ``old`` is still there (an id the rewrite
    dropped is put back as it was); one the rewrite added is kept. With ``only_example`` (the rewrite was asked for
    worked examples only) each equation is as it was except for its ``example``."""
    old_items = _oracle.equation_items(old)
    new_items = _oracle.equation_items(new)
    if only_example:
        fresh = {str(e.get("id")).strip().upper(): e for e in new_items}
        out = []
        for e in old_items:
            got = fresh.get(str(e.get("id")).strip().upper())
            out.append({**e, "example": got["example"]} if got is not None and "example" in got else e)
        return out
    have = {str(e.get("id")).strip().upper() for e in new_items}
    return [*new_items, *[e for e in old_items if str(e.get("id")).strip().upper() not in have]]


#: Added to the request when the reader objected to a check's case: the plan fixes the case or the measure.
ISOLATION_REQUEST = (
    "\nWhere the reader objected that a check's `case` does not isolate what its `measure` claims: decide whether it is "
    "right; if so change the check's `case` (or its `measure`) so the number is the one the check describes, in the "
    "regime where its condition holds everywhere, and rewrite its `expected_formula` for that case. Never loosen a "
    "tolerance, and do not write an `expected` by hand: FI computes it from the formula.")


def request(found: list[str], *, equations: bool = False, isolation: bool = False) -> str:
    """The part of the one request to the plan that carries the reviewer's findings. The reviewer is a second reader,
    not the person: its findings are applied only when they are right. ``equations``: some finding is about an
    equation of the model (the plan may then rewrite that equation too)."""
    return (
        "A second model read the plan's checks against known answers as a referee would (it is not the person; apply "
        "what it found only where it is right) and found:\n"
        + "\n".join(f"- {f[0].upper()}{f[1:]}." for f in found)
        + "\nFor each finding that is right, change the check: a better case, how its number is computed, its expected "
        "value worked out at the step it is compared at, a reachable tolerance, or one more check for an equation no "
        "check tests, with a `reference` saying how its expected value follows. Replace a check found trivial with a "
        "better one; never remove a check without putting one in its place. Keep each kind's numeric form. Change only "
        "the checks (and a criterion that reads a check you change, when its number changes meaning)"
        + (", and an equation of the model that a finding names" if equations else "")
        + ", and nothing else in the plan."
        + (EQUATION_REQUEST if equations else "")
        + (ISOLATION_REQUEST if isolation else "")
    )


def equation_record(review: Review | None, protocol: dict[str, Any] | None, *, same_model: bool | None,
                    reviewer: str, planner: str) -> dict[str, Any] | None:
    """What is recorded about the plan's equations (``None`` when the model lists none): how many there are, which the
    reader judged, and whether they were read by a model other than the planner (``read_by_other_model``). They count
    as read by another model only when the reader is shown to be another model AND it gave a verdict on each equation;
    otherwise ``why`` says plainly what is missing. The reader's objections are used (:func:`equation_findings`) only
    when they count."""
    listed = equations_of(protocol)
    if not listed:
        return None
    judged = [e["id"] for e in (review.equations if review is not None else []) if e["standard"] is not None]
    ids = [str(e["id"]).strip() for e in listed]
    unjudged = [i for i in ids if i not in judged]
    if review is None:
        why = "the second reading gave no usable answer"
    elif same_model is not False:
        why = (f"the reader is the model that wrote the plan ({planner})" if same_model
               else f"FI cannot tell whether the reader ({reviewer}) is another model than the one that wrote the plan")
    elif unjudged:
        why = f"the reader gave no verdict on {_names(unjudged)}"
    else:
        why = ""
    return {"total": len(ids), "judged": judged, "read_by_other_model": not why, "why": why,
            "verdicts": review.equations if review is not None else []}


def equation_lines(record: dict[str, Any] | None, *, reviewer: str, research: bool) -> list[str]:
    """What plan.md says about the equations of the model (prose, never read back)."""
    if not record:
        return []
    n = record["total"]
    plural = "equation" if n == 1 else "equations"
    if not record["read_by_other_model"]:
        below = (" Under research this keeps the result below *independently validated*." if research else "")
        return [f"- The model's {n} {plural} {'was' if n == 1 else 'were'} not read by another model "
                f"({record['why']}): nobody but the model that wrote them has checked that each is the standard "
                f"definition of its quantity.{below}"]
    rows = [f"- The model's {n} {plural} {'was' if n == 1 else 'were'} read by {reviewer}, not the model that wrote "
            "them:"]
    for v in record["verdicts"]:
        if v["standard"] is False:
            rows.append(f"  - **{v['id']}** is not the standard definition: {v['correct'] or 'no replacement given'}"
                        f"{' (' + v['why'] + ')' if v['why'] else ''}.")
        elif v["standard"]:
            rows.append(f"  - **{v['id']}** is the standard definition or a correct derivation.")
    return rows


def plan_lines(review: Review | None, *, reviewer: str, planner: str, same_model: bool | None, research: bool,
               reported: bool = True, error: str = "", sent: bool = True,
               equations: dict[str, Any] | None = None, isolation: dict[str, Any] | None = None) -> list[str]:
    """What plan.md says about the second opinion, in plain words (never read back). ``same_model`` is ``None`` when
    it cannot be told whether the two were different models (a provider default FI cannot name)."""
    return [*_check_lines(review, reviewer=reviewer, planner=planner, same_model=same_model, research=research,
                          reported=reported, error=error, sent=sent, isolation=isolation),
            *equation_lines(equations, reviewer=reviewer, research=research)]


def _check_lines(review: Review | None, *, reviewer: str, planner: str, same_model: bool | None, research: bool,
                 reported: bool, error: str, sent: bool, isolation: dict[str, Any] | None = None) -> list[str]:
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
    # Under research a second reading counts only from another model, with a usable answer (independence_gaps).
    below = (" Under research this keeps the result below *independently validated*."
             if research and (review is None or same_model is not False or not reported) else "")
    if review is None and not sent:
        return ["", f"- No second reader looked at the checks ({error or 'nothing to show'}), so nothing was changed "
                f"for it.{below}"]
    if review is None:
        return ["", f"- The checks were sent to {reviewer} for a second reading, but its answer could not be used "
                f"({error or 'it named none of the plan checks'}), so nothing was changed for it.{below}"]
    rows = ["", f"- {who}" + (f" Its summary: {review.summary}" if review.summary else "") + below]
    for c in review.checks:
        verdict = [
            {True: "tests the model", False: "may not test the model", None: "no verdict on what it tests"}[c["appropriate"]],
            {True: "would catch a plausible bug", False: "would miss a plausible bug",
             None: "no verdict on what it catches"}[c["discriminating"]],
            {True: "its number is well defined", False: "its number is not well defined",
             None: "no verdict on its number"}[c["well_defined"]],
        ]
        if c.get("isolates") is not None and not (isolation and not isolation.get("read_by_other_model")):
            verdict.append("its case isolates what it measures" if c["isolates"] else "its case may not isolate what it measures")
        extra = "; ".join(x for x in (c["definition_note"], c["isolates"] is False and c.get("isolation_note"),
                                      c["better"] and f"a better check: {c['better']}") if x)
        rows.append(f"  - **{c['name']}**: {', '.join(verdict)}." + (f" {extra}." if extra else ""))
    if review.tested or review.untested:
        rows.append(f"  - Equations the checks test: {', '.join(review.tested) or 'none named'}"
                    + ("." if review.partial else f"; not tested by any: {', '.join(review.untested) or 'none'}."))
    if review.partial:
        rows.append("  - The plan has more checks than the reader was shown; it did not judge the rest.")
    if isolation and not isolation.get("read_by_other_model"):
        rows.append(f"  - The checks' cases were not judged by another model ({isolation.get('why')}): nobody but the "
                    "model that wrote them has checked that each case isolates what its measure claims.")
    return rows


# --- whether the second reading counts as independent (rigor_profile: research) ------------------------------------

#: What a connection adds to a model's name without making it another model: a date or a version stamp, ``preview``,
#: ``latest``, a ``-v1``/``-v2:0`` revision, a ``-001`` release (``gpt-5.6-luna-2026-09`` is ``gpt-5.6-luna``;
#: ``claude-opus-4-5-20251101`` is ``claude-opus-4-5``; ``gemini-2.0-flash-001`` is ``gemini-2.0-flash``).
#: One such stamp at the end of a name, cut one at a time (a run of digits of two or more, ``v2``, ``v1:0``,
#: ``preview``, ``latest``): linear, whatever the name.
_STAMP_RE = re.compile(r"[-_.@](?:\d{2,}|v\d+(?::\d+)?|preview|latest)$")
#: A cloud's region and vendor before a model's name (``us.anthropic.claude-...``, ``anthropic.claude-...``).
_CLOUD_PREFIX_RE = re.compile(r"^(?:[a-z]{2}\.|apac\.|global\.|us-gov\.)?"
                              r"(?:anthropic|meta|amazon|mistral|cohere|ai21|openai|google|deepseek)\.")


def _unstamped(text: str) -> str:
    while True:
        cut = _STAMP_RE.sub("", text)
        if cut == text or not cut:
            return text
        text = cut
#: Connections that relay a model through a proxy of their own: the model they report may be the name FI asked for
#: echoed back, so it does not show which model answered.
PROXY_PROVIDERS = frozenset({"claude_code", "github_copilot_cli", "github_copilot_vscode"})
#: Which connections name the model that answered, in the gap's words.
_WHO_NAMES = ("an HTTP API, the claude command-line tool, and VS Code with a model picked (not Auto) name the model that "
              "answered")


def canonical_model(name: Any) -> str:
    """A model's name as one model: lower case, without a vendor or folder prefix (``openai/gpt-5``, ``models/gemini``,
    ``us.anthropic.``), without an Ollama ``:tag`` (``gemma4:27b`` is read as ``gemma4``: it may be the same weights),
    a date, a version or release stamp or ``-preview``/``-latest``, and with ``.`` and ``_`` read as ``-``. Two names
    that differ only in these are the same model. Errs towards "the same": two models taken for one only keep a gap."""
    text = str(name or "").strip().lower()[:200].rsplit("/", 1)[-1]
    text = _CLOUD_PREFIX_RE.sub("", text)
    text = _unstamped(text)  # a Bedrock revision (-v1:0) before the Ollama tag is cut
    text = text.split(":", 1)[0] or text
    return re.sub(r"[._]", "-", _unstamped(text))


def same_model(a: Any, b: Any) -> bool:
    """Whether two model names, as connections reported them, are one model (:func:`canonical_model`)."""
    first, second = canonical_model(a), canonical_model(b)
    return bool(first) and first == second


def _shown(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The checks the reader is shown (whole checks, as many as fit): :func:`prompt_parts`'s own cut."""
    shown: list[dict[str, Any]] = []
    for oracle in _oracle.declared(protocol):
        if len(json.dumps([*shown, oracle], indent=2, default=str)) > _MAX_CHECK_CHARS:
            break
        shown.append(oracle)
    return shown


def fingerprint(oracle: dict[str, Any]) -> str:
    """What decides a check's verdict, as one string: its name, expected value, tolerance and its mode, case and
    measure (not its wording, its kind or its reference, which a later step may fill in without changing what is
    judged)."""
    expected, limit, mode = _oracle.limit_of(oracle)
    return json.dumps({"name": str(oracle.get("name") or "").strip().lower(),
                       "expected": expected, "limit": limit, "mode": mode, "case": oracle.get("case"),
                       "measure": " ".join(str(oracle.get("measure") or "").split())}, sort_keys=True, default=str)


def fingerprints(protocol: dict[str, Any] | None, *, shown_only: bool = False) -> list[str]:
    """:func:`fingerprint` of each declared check (only those a reader is shown, with ``shown_only``)."""
    return [fingerprint(o) for o in (_shown(protocol) if shown_only else _oracle.declared(protocol))]


def _names(items: list[str]) -> str:
    return ", ".join(repr(n) for n in items[:5]) + (f" and {len(items) - 5} more" if len(items) > 5 else "")


def independence_gaps(record: dict[str, Any] | None, calls: list[dict[str, Any]], *, configured: bool,
                      protocol: dict[str, Any] | None = None) -> list[str]:
    """:func:`_check_gaps` (the checks), and the same sentence for the plan's equations: when the record says the model's
    equations were not read by another model (a record from before equations were read says nothing about them), the
    gap names them too, so the paper and the evidence level never imply that a different model checked them."""
    gaps = _check_gaps(record, calls, configured=configured, protocol=protocol)
    eq = record.get("equations") if isinstance(record, dict) else None
    if isinstance(eq, dict) and eq.get("total") and not eq.get("read_by_other_model"):
        n = int(eq["total"])
        tail = (f"the model's {n} equation{'s were' if n != 1 else ' was'} not read by another model either "
                f"({eq.get('why') or 'no reason recorded'})")
        if gaps:
            gaps[-1] = f"{gaps[-1]}; {tail}"
        else:
            gaps.append(f"the plan's equations were not read by a second, different model ({eq.get('why') or 'no reason recorded'})")
    return gaps


def _check_gaps(record: dict[str, Any] | None, calls: list[dict[str, Any]], *, configured: bool,
                protocol: dict[str, Any] | None = None) -> list[str]:
    """Under ``rigor_profile: research``: why the second reading of the plan's checks does not count, one plain sentence
    (empty when it counts). It counts only when

    * it gave a usable answer (``record``, ``.fi/oracle_review.json``);
    * it covers the checks ``protocol`` (the frozen one) declares: each was shown to the reader (``read``), has a
      verdict, and is the check as the reader read it or as the same look left it after applying what the reader found
      (``after_look``); a check the reader proposed itself (``add``) counts as read only with the numbers it gave it;
    * the quest's record of its model calls (``calls``, ``.fi/model_calls.jsonl``) names, for the call that gave it (the
      record's ``call_id``) and for every answered call of a step that writes the checks (:data:`WRITER_NODES`), the
      model that answered, through a connection that does not relay a proxy (:data:`PROXY_PROVIDERS`);
    * and the reading's model is none of the writers' (:func:`same_model`).

    ``configured``: ``provider.node_models.oracle_review`` is set; when it is not, the sentence says how to set it."""
    record = record if isinstance(record, dict) else {}
    how = f"; to have another model read them, {HOW_TO_NAME_ANOTHER}" if not configured else ""
    again = "; for this quest, do the plan again: `--resume <quest_id> --from plan --approve-as <you>`"
    if not record.get("lines") and not record.get("error") and not record.get("verdicts"):
        return [f"{NOT_REVIEWED}: no second reading of them is recorded{how}"]
    if record.get("error") or not record.get("verdicts"):
        why = str(record.get("error") or "it judged none of the checks")
        return [f"{NOT_REVIEWED}: the second reading gave no usable answer ({why[:200]}){how}"]
    gaps: list[str] = []
    if protocol is not None:
        judged = {str(v.get("name") or "").strip().lower() for v in record.get("verdicts") or [] if isinstance(v, dict)}
        # The reader's own proposals, with the numbers it gave them: one the plan took with other numbers was not read.
        proposed = {fingerprint(a) for a in record.get("add") or [] if isinstance(a, dict)}
        read = set(record.get("read") or []) | set(record.get("after_look") or [])
        shown: set[str] = set()
        for fp in record.get("read") or []:
            try:
                shown.add(str(json.loads(fp).get("name") or ""))
            except (TypeError, ValueError, AttributeError):
                continue
        unread = []
        for oracle in _oracle.declared(protocol):
            name = str(oracle["name"]).strip()
            if fingerprint(oracle) in proposed:
                continue  # the reader's own check, as it proposed it
            if name.lower() not in judged or name.lower() not in shown or fingerprint(oracle) not in read:
                unread.append(name)
        if unread:
            gaps.append(f"{NOT_REVIEWED}: the second reading did not judge {_names(unread)} as "
                        f"{'it stands' if len(unread) == 1 else 'they stand'} (added or changed after it, or not "
                        f"shown to it){again}")
    ok = [r for r in calls if isinstance(r, dict) and r.get("outcome") == "ok"]
    call_id = record.get("call_id")
    readings = [r for r in ok if r.get("node") == NODE]
    if call_id:
        reading = next((r for r in readings if r.get("call_id") == call_id), None)
    else:  # a record written before the call's id was kept: only when there is no other reading to mistake it for
        reading = readings[0] if "call_id" not in record and len(readings) == 1 else None
    writers = [r for r in ok if r.get("node") in WRITER_NODES]
    if reading is None or not writers:
        missing = "read" if reading is None else "wrote"
        return [*gaps, f"{NOT_REVIEWED}: the quest's record of its model calls does not show which model {missing} them"]

    def _named(row: dict[str, Any]) -> bool:
        return bool(row.get("reported") and row.get("served_model")) and row.get("provider") not in PROXY_PROVIDERS

    unnamed = "read" if not _named(reading) else ("wrote" if not all(_named(w) for w in writers) else "")
    if unnamed:
        return [*gaps, f"{NOT_REVIEWED}: the connection did not say which model {unnamed} them, so they are not shown "
                       f"to be two different models ({_WHO_NAMES}; a proxy such as claude_code or copilot-api does not)"]
    model = str(reading["served_model"])
    if any(same_model(w["served_model"], model) for w in writers):
        tail = how or "; the model set for `oracle_review` is one that wrote the plan, so name another"
        return [*gaps, f"{NOT_REVIEWED}: {model} wrote them and read them again{tail}"]
    return gaps

