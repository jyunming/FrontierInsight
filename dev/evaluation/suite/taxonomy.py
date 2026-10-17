"""A fixed taxonomy for why a run did not deliver a trustworthy paper, and the rules that sort a failure into it.

A failure is placed by ONE stage (where in the pipeline it stood: plan, checks, design, code, run, analysis, paper,
review, the stages of ``core/quality_report.py``) and ONE cause class:

====================  ============================================================================================
class                 meaning
====================  ============================================================================================
spec_plan_gap         the plan or specification was incomplete or unreadable, so there was nothing sound to run
model_capability      the model's code was still wrong after FI's repairs ran out
spec_gaming           the numbers were typed in or came from a surrogate, not computed by the study
gate_bypassed         a check failed or was left unresolved and the quest went on (or its repair budget was spent)
no_gate               a wrong result reached the paper and no check fired
integration_contract  the pieces disagree about what they hand each other (a manifest vs the data, a result field)
provider_infra        the model provider or the machine failed (timeout, quota, connection)
fi_bug                a defect in FI itself; set ONLY by a grader (a rule never infers it)
unclassified          the records do not say; never guessed
====================  ============================================================================================

The rules (first match wins; every one reads a record FI wrote, and quotes the line it read). Rule ids appear in
``round.json``:

R0  grader        a grader's explicit ``class`` in ``grades.json`` is taken as given (the only way to say fi_bug).
R1  infra         the end record (quest_failed.md or the stop card) names a provider or machine fault: "Kind of failure:
                  setup", ReadTimeout/ConnectTimeout, HTTP 429/5xx, rate limit, quota, connection refused/reset, or
                  "takes longer to answer" -> provider_infra, at the stage of the failing step.
R2  plan          the quest paused or stopped at the plan stage -> spec_plan_gap.
R3  manifest      a stop with ``.fi/run_manifest_stop.json`` (the run manifest disagrees with the analysis data)
                  -> integration_contract.
R4  repairs       an honest stop at the run stage whose card says the experiment "could not run to the end" or that
                  repairs ended -> model_capability.
R5  typed         a delivered paper while ``.fi/typed_results.json`` (numbers typed into the simulation) or
                  ``static_checks.json: typed_constants`` shows the check fired -> spec_gaming at the run stage. A
                  check that fired and was repaired is not proof; this is a candidate and is labelled "candidate".
R6  unfixed       a delivered paper with a problem the quality report lists as "not fixed" -> gate_bypassed at that stage.
R7  evidence      a delivered paper whose EVIDENCE.json says the evidence gate or the claim check did not pass, or a
                  known-answer check failed and FI went on -> gate_bypassed (stage paper/analysis).
R8  nogate        a delivered paper with a failed G1 in ``grades.json`` and none of R5-R7 -> no_gate.
(else)            unclassified.

Contributing flags (``also``) are listed for every run, whatever the primary class: a known-answer check failed and
FI went on; the final code was never read by the second model, or its repairs did not converge; a static check fired;
topic requirements were left uncovered.
"""

from __future__ import annotations

import re
from typing import Any, Callable

CLASSES: dict[str, str] = {
    "spec_plan_gap": "spec / plan gap",
    "model_capability": "model capability (code wrong after repairs)",
    "spec_gaming": "specification gaming (surrogate / typed-in numbers)",
    "gate_bypassed": "gate bypassed / budget spent",
    "no_gate": "no gate caught it",
    "integration_contract": "integration / contract",
    "provider_infra": "provider / infrastructure",
    "fi_bug": "FI bug (set by a grader)",
    "unclassified": "unclassified",
}
STAGES = ("plan", "checks", "design", "code", "run", "analysis", "paper", "review")

_INFRA = re.compile(r"Kind of failure:\*{0,2}\s*setup|ReadTimeout|ConnectTimeout|ConnectError|\b429\b|\b50[234]\b|rate.?limit|"
                    r"quota|connection (?:refused|reset|aborted)|takes longer to answer|timed out", re.I)
_REPAIRS = re.compile(r"could not run to the end|repairs? (?:ended|ran out)|repaired the script", re.I)
_EVIDENCE_GATE = re.compile(r"evidence gate did not pass|claim check did not pass|went on by itself although the known-answer "
                            r"check '[^']*' failed", re.I)
_KNOWN_ANSWER = re.compile(r"went on by itself although the known-answer check '([^']*)' failed", re.I)


def _line(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _hit(stage: str, cls: str, rule: str, evidence: str) -> dict[str, str]:
    return {"stage": stage, "class": cls, "rule": rule, "evidence": _line(evidence)}


def _failed_text(facts: dict[str, Any]) -> str:
    stuck = facts.get("stuck") or {}
    return " ".join(str(x) for x in (facts.get("failed_md") or "", stuck.get("problem") or "", stuck.get("say") or "",
                                    " ".join(map(str, stuck.get("tried") or []))))


def r0_grader(facts: dict[str, Any]) -> dict[str, str] | None:
    g = facts.get("grade") or {}
    cls = str(g.get("class") or "")
    if cls in CLASSES and cls != "unclassified":
        return _hit(str(g.get("stage") or facts.get("stage") or ""), cls, "R0 grader", f"grader: {g.get('class_note') or g.get('notes') or cls}")
    return None


def r1_infra(facts: dict[str, Any]) -> dict[str, str] | None:
    if facts["outcome"] not in ("crash", "honest_stop", "pause"):
        return None
    m = _INFRA.search(_failed_text(facts))
    if not m:
        return None
    return _hit(facts.get("stage") or "", "provider_infra", "R1 infra", f"end record says: {m.group(0)}")


def r2_plan(facts: dict[str, Any]) -> dict[str, str] | None:
    if facts["outcome"] not in ("pause", "honest_stop"):
        return None
    pause_kind = str((facts.get("pause") or {}).get("kind") or "")
    if facts.get("stage") == "plan" or (facts["outcome"] == "pause" and pause_kind == "plan"):
        why = (facts.get("pause") or {}).get("headline") or (facts.get("stuck") or {}).get("problem") or "stopped at the plan"
        return _hit("plan", "spec_plan_gap", "R2 plan", f"stopped at the plan: {why}")
    return None


def r3_manifest(facts: dict[str, Any]) -> dict[str, str] | None:
    if facts["outcome"] == "honest_stop" and facts.get("run_manifest_stop"):
        return _hit(facts.get("stage") or "analysis", "integration_contract", "R3 manifest",
                    f".fi/run_manifest_stop.json: {facts['run_manifest_stop']}")
    return None


def r4_repairs(facts: dict[str, Any]) -> dict[str, str] | None:
    if facts["outcome"] == "honest_stop" and facts.get("stage") == "run":
        m = _REPAIRS.search(_failed_text(facts))
        if m:
            return _hit("run", "model_capability", "R4 repairs", f"STUCK.json: {m.group(0)}")
    return None


def r5_typed(facts: dict[str, Any]) -> dict[str, str] | None:
    if facts["outcome"] != "paper":
        return None
    typed = int(facts.get("typed_results_used") or 0)
    const = int(((facts.get("static_fired") or {}).get("typed_constants")) or 0)
    if typed or const:
        which = "typed_results.json" if typed else "static_checks.json typed_constants"
        return _hit("run", "spec_gaming", "R5 typed", f"candidate: {which} shows the typed-number check fired and a paper was delivered")
    return None


def r6_unfixed(facts: dict[str, Any]) -> dict[str, str] | None:
    if facts["outcome"] == "paper" and facts.get("stage"):
        chain = facts.get("chain") or []
        return _hit(facts["stage"], "gate_bypassed", "R6 unfixed", chain[0] if chain else f"a {facts['stage']} problem was not fixed")
    return None


def r7_evidence(facts: dict[str, Any]) -> dict[str, str] | None:
    if facts["outcome"] != "paper":
        return None
    for gap in facts.get("evidence_gaps") or []:
        if _EVIDENCE_GATE.search(gap):
            stage = "analysis" if _KNOWN_ANSWER.search(gap) else "paper"
            return _hit(stage, "gate_bypassed", "R7 evidence", f"EVIDENCE.json: {gap}")
    return None


def r8_nogate(facts: dict[str, Any]) -> dict[str, str] | None:
    g = facts.get("grade") or {}
    if facts["outcome"] == "paper" and str(g.get("g1") or "").lower().startswith("fail"):
        return _hit("run", "no_gate", "R8 nogate", "G1 failed in grades.json and no check in the records fired")
    return None


RULES: tuple[Callable[[dict[str, Any]], dict[str, str] | None], ...] = (
    r0_grader, r1_infra, r2_plan, r3_manifest, r4_repairs, r5_typed, r6_unfixed, r7_evidence, r8_nogate)


def contributing(facts: dict[str, Any]) -> list[str]:
    """Flags worth knowing about whatever the primary class is, each with the record it comes from."""
    out: list[str] = []
    for gap in facts.get("evidence_gaps") or []:
        m = _KNOWN_ANSWER.search(gap)
        if m:
            out.append(f"known-answer check '{m.group(1)}' failed and FI went on (EVIDENCE.json)")
    code = facts.get("code") or {}
    if code.get("final_version_read") is False:
        why = f": {code['skipped']}" if code.get("skipped") else ""
        out.append(f"the final code was not read by a second model{why} (code_review.json)")
    if code.get("repairs_converged") is False:
        out.append("the code repairs did not converge (code_review.json)")
    for name, used in sorted((facts.get("static_fired") or {}).items()):
        out.append(f"static check '{name}' fired {used}x (static_checks.json)")
    if int(facts.get("typed_results_used") or 0):
        out.append("typed-results check fired (typed_results.json)")
    left = facts.get("coverage_left") or []
    if left:
        out.append(f"{len(left)} topic requirement(s) left uncovered (topic_covered.json)")
    return out


def classify(facts: dict[str, Any]) -> dict[str, Any] | None:
    """The failure of one run as ``{stage, class, rule, evidence, also}``, or ``None`` when the run has no failure.

    A run has a failure when it did not deliver a paper, or delivered one that is flawed (a failed gate in the grades, an
    unfixed problem or a failed evidence/claim check in the records)."""
    outcome = facts["outcome"]
    if outcome in ("running",):
        return None
    g = facts.get("grade") or {}
    clean = all(str(g.get(k) or "").lower().startswith("pass") for k in ("g1", "g2", "g3"))
    flawed = outcome == "paper" and (bool(g.get("class")) or (not clean and (
        bool(facts.get("stage")) or str(g.get("g1") or "").lower().startswith("fail") or bool(facts.get("evidence_flaw"))
        or bool(facts.get("typed_results_used")) or bool((facts.get("static_fired") or {}).get("typed_constants")))))
    if outcome == "paper" and not flawed:
        return None
    for rule in RULES:
        hit = rule(facts)
        if hit:
            return {**hit, "also": contributing(facts)}
    return {"stage": facts.get("stage") or "", "class": "unclassified", "rule": "none",
            "evidence": _line(facts.get("headline") or "no rule matched the records"), "also": contributing(facts)}
