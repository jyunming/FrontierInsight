"""How much of a quest's result has been checked against something other than itself.

A run used to end in one green state. The three stored SIR runs of one topic show why that misleads: each passed its number
audit, its statistics audit and its provenance audit, and one of them had swept a different grid than the topic asked for, so a
pass meant the paper copied the script's output faithfully and nothing more. The levels below are ordered, each needs the one
before it, and each says what it guarantees and what it does not (its *blind spots*), so a label never claims more than the
checks behind it prove:

* ``executed`` — the experiment ran to the end and printed results;
* ``internally_reconciled`` — and the audits that compare the paper with those results (numbers, statistics, provenance) found
  nothing: the paper says what the script printed;
* ``protocol_runtime_matched`` — and the protocol was frozen before the run and is intact, the script held to it, the run's own
  manifest says it did what the protocol fixed, and no warning from its own numerics was accepted;
* ``independently_validated`` — and the engine judged the script's oracle measurements against the protocol's expected values
  and tolerances;
* ``statistically_adequate`` — and every headline metric has a declared estimator matched to its data, the contrasts carry
  engine-computed p-values, and every precision target was reached;
* ``publication_ready`` — and the review accepted the paper with no must-fix finding and the protocol was not amended after the
  results were seen.

``gaps`` lists, one sentence each, what stands between the quest and the next level. This module reads the quest's records and
the final state; it needs no model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import frozen_protocol as _frozen
from . import run_manifest as _run_manifest

LEVELS = (
    "executed", "internally_reconciled", "protocol_runtime_matched", "independently_validated", "statistically_adequate",
    "publication_ready",
)

# What each level guarantees, what it does not, and which records back it.
INFO: dict[str, dict[str, Any]] = {
    "executed": {
        "assurance_claim": "The experiment ran to the end and printed results.",
        "known_blind_spots": ["Nothing says the results are right, or that the run did what the plan said."],
        "artifacts": ["code/experiment.py"],
    },
    "internally_reconciled": {
        "assurance_claim": "The number, statistics and provenance audits found the paper faithful to what the script printed.",
        "known_blind_spots": ["The script itself may be wrong: the audits compare the paper with the script's output, not with reality."],
        "artifacts": ["paper/numeric_audit.json", "paper/statistics_audit.json", "paper/provenance_audit.json"],
    },
    "protocol_runtime_matched": {
        "assurance_claim": (
            "The protocol was frozen before the first full run and is intact; the script held to it; the run's own manifest "
            "says it swept the fixed grid, ran the fixed number of trials in every setting and used the fixed thresholds; no "
            "numeric warning was accepted."
        ),
        "known_blind_spots": [
            "When the script keeps a per-trial ledger (trial_ledger.jsonl), the trial counts (realized_grid, "
            "attempted_per_cell, successful_per_cell, failed_trials) are derived by the engine from its real lines, not "
            "read from the script's own summary. Without a ledger, those fields are still the script's own statement, "
            "as before. Either way, a claimed trial count is also cross-checked against the real per-trial values a "
            "`kind: \"mean\"` metric reports (core/run_manifest.py's result_json check) — but a script that fabricates "
            "the ledger rows AND the metric's per-trial values together, consistently, is not caught by either "
            "mechanism; no check here re-executes the science itself.",
        ],
        "artifacts": ["needs/FROZEN_PROTOCOL.json", "needs/PROTOCOL_CHECK.json", "needs/RUN_MANIFEST_CHECK.json", "needs/NUMERIC_WARNINGS.json"],
    },
    "independently_validated": {
        "assurance_claim": "The engine judged the script's oracle measurements against the protocol's expected values and tolerances.",
        "known_blind_spots": [
            "The expected values come from the plan (the same model that wrote it): a wrong closed form is passed by a wrong "
            "simulator that agrees with it."
        ],
        "artifacts": ["needs/ORACLE_CHECK.json"],
    },
    "statistically_adequate": {
        "assurance_claim": (
            "Each headline metric has a declared estimator matched to its data, the contrasts carry engine-computed p-values "
            "adjusted within their family, and every precision target was reached."
        ),
        "known_blind_spots": ["The estimators' assumptions (independent trials or clusters, exchangeable pairs, enough clusters) are declared by the plan, not tested."],
        "artifacts": ["needs/FROZEN_PROTOCOL.json"],
    },
    "publication_ready": {
        "assurance_claim": "The review accepted the paper with no must-fix finding, and the protocol was not amended after the results were seen.",
        "known_blind_spots": ["The review is a model's opinion (and a person's decision when one was asked for); it does not re-run anything."],
        "artifacts": ["needs/DESIGN_HISTORY.json", "paper/review.json"],
    },
}

# The names an older record uses, and the level each one is read as: it lacks the runtime, engine-judged and statistical
# checks the newer levels stand for, so none of them is read as more than the audits' level.
LEGACY = {
    "internally_consistent": "internally_reconciled",
    "validated_against_oracle": "internally_reconciled",
    "publication_ready": "internally_reconciled",
}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _audit_gaps(paper_dir: Path) -> tuple[bool, list[str]]:
    """Whether the paper-vs-results audits all ran and passed, and what they found.

    Each of the three writes its report even when it finds nothing (a clean run leaves evidence the check ran), and marks
    itself ``skipped`` when there was nothing for it to check (a survey with no numeric results, say) — that is a legitimate
    outcome, not a gap. A report missing altogether means its check never ran (it crashed before it could write one): when
    none of the three ran, that is simply a quest that has not reached this stage yet, said once, plainly; when only SOME
    are missing, that is the suspicious pattern (one audit's clean report standing in for two that never ran), named for
    each one that is missing.
    """
    ran = present = 0
    gaps: list[str] = []
    missing: list[str] = []
    for name, label in (("numeric_audit", "the number check"), ("statistics_audit", "the statistics check"),
                        ("provenance_audit", "the provenance check")):
        report = _json(paper_dir / f"{name}.json")
        if not isinstance(report, dict):
            missing.append(label)
            continue
        present += 1
        if report.get("skipped"):
            continue
        ran += 1
        if report.get("ok") is False:
            findings = report.get("findings") or []
            gaps.append(f"{label} reported {len(findings)} finding(s)")
    if missing:
        if present == 0:
            gaps.append("the paper has not been checked against the results yet")
        else:
            gaps.extend(f"{label} did not run, or its report could not be read" for label in missing)
    return ran > 0 and not gaps and not missing, gaps


def assess(
    quest_root: Path, state: dict[str, Any], *, precision_missed: list[str] | None = None,
    settings: dict[str, str] | None = None, statistics_gaps: list[str] | None = None,
) -> dict[str, Any]:
    """The quest's evidence status, the ladder with what each level guarantees, and the gaps below the next level.

    ``state`` is the final (or paused) state; ``precision_missed`` names the probabilities whose pooled interval did not
    reach the protocol's target; ``statistics_gaps`` says what keeps the statistics from being adequate
    (:func:`core.metric_spec.coverage_gaps`); ``settings`` holds ``protocol_check``, ``oracle_check``, ``numeric_warnings``,
    ``run_manifest_check`` and ``rigor_profile`` as the run had them (a check that was turned off is a gap, not a pass)."""
    settings = settings or {}
    needs = quest_root / "needs"
    gaps: dict[str, list[str]] = {level: [] for level in LEVELS}

    # executed
    exec_result = state.get("exec_result") or {}
    executed = bool(state.get("result_json")) and exec_result.get("returncode", 0) == 0
    if not executed:
        gaps["executed"].append("the experiment has not produced results (it has not run, or it failed)")

    # internally reconciled
    audits_ok, audit_gaps = _audit_gaps(quest_root / "paper")
    reconciled = executed and audits_ok
    if executed and not audits_ok:
        gaps["internally_reconciled"] += audit_gaps or ["the paper has not been checked against the results yet"]

    # The protocol the gates held the run to is the frozen one, and never the design in the state, which a redesign can
    # leave without a protocol (a quest begun before the freeze existed has none frozen).
    frozen = _frozen.load(quest_root)
    if frozen is not None:
        protocol = frozen.get("protocol") if isinstance(frozen.get("protocol"), dict) else None
    else:
        design = state.get("design") or {}
        protocol = design.get("protocol") if isinstance(design.get("protocol"), dict) else None

    # protocol matched at runtime
    matched_gaps = gaps["protocol_runtime_matched"]
    if frozen is None and executed:
        matched_gaps.append("the protocol was not frozen before the run (the quest began before the freeze existed): nothing shows the run was held to the protocol it started with")
    if frozen is not None and frozen.get("problem"):
        matched_gaps.append(str(frozen["problem"]))
    if protocol is None:
        matched_gaps.append("the plan fixes no protocol, so nothing held the experiment to the grid, runs and thresholds it was meant to use")
    else:
        protocol_record = _json(needs / "PROTOCOL_CHECK.json")
        if settings.get("protocol_check") == "off":
            matched_gaps.append("the protocol check was turned off")
        elif not isinstance(protocol_record, dict) or protocol_record.get("status") != "ok":
            status = protocol_record.get("status") if isinstance(protocol_record, dict) else "not run"
            matched_gaps.append(f"the script was not shown to follow the protocol (protocol check: {status})")
    manifest_record: Any = None
    if settings.get("run_manifest_check") == "off":
        matched_gaps.append("the run manifest check was turned off")
    elif protocol is not None and _run_manifest.checkable(protocol):
        manifest_record = _json(needs / "RUN_MANIFEST_CHECK.json")
        status = manifest_record.get("status") if isinstance(manifest_record, dict) else None
        if status == "single_script":
            matched_gaps.append("the quest ran as one script, which writes no run manifest: nothing shows the run did what the protocol fixed (execution.split_analysis)")
        elif status == "not_applicable":
            pass  # a deterministic study runs as one script and has no per-trial outcomes to record
        elif status != "ok":
            matched_gaps.append(f"the run was not shown to have done what the protocol fixed (run manifest check: {status or 'not run'})")
    warnings = _json(needs / "NUMERIC_WARNINGS.json")
    last_warning_entry = warnings[-1] if isinstance(warnings, list) and warnings else None
    if isinstance(last_warning_entry, dict) and "error" in last_warning_entry:
        # The scanner crashed rather than running cleanly: this is never a pass, whatever engine.numeric_warnings says —
        # "off" and "warn" are choices about a check that ran; a crash means the run's numerics were never checked at all.
        matched_gaps.append("the numeric-warning scanner failed to run, so whether the run's numerics are sound is unknown (see needs/NUMERIC_WARNINGS.json)")
    elif state.get("numeric_warnings_accepted"):
        matched_gaps.append("the run's numeric warnings were accepted as they were (see needs/NUMERIC_WARNINGS.json)")
    elif settings.get("numeric_warnings") == "off":
        matched_gaps.append("the numeric-warning check was turned off")
    elif isinstance(warnings, list) and warnings and settings.get("numeric_warnings") == "warn":
        matched_gaps.append("the run's numerics warned and the check only recorded it (engine.numeric_warnings: warn)")
    environment = _json(needs / "ENVIRONMENT.json")
    if isinstance(environment, dict) and environment.get("isolated") is False:
        matched_gaps.append(
            "the run shared its Python environment with other quests (execution.shared_interpreter / "
            "system_site_packages); a package installed for a different quest could have affected this one (see needs/ENVIRONMENT.json)"
        )
    matched = reconciled and not matched_gaps

    # independently validated
    valid_gaps = gaps["independently_validated"]
    oracle_record = _json(needs / "ORACLE_CHECK.json")
    if settings.get("oracle_check") == "off":
        valid_gaps.append("the oracle check was turned off")
    elif protocol is not None and (not isinstance(oracle_record, dict) or oracle_record.get("status") != "ok"):
        status = oracle_record.get("status") if isinstance(oracle_record, dict) else "not run"
        valid_gaps.append(f"the script did not pass an independent oracle (oracle check: {status})")
    elif protocol is not None and oracle_record.get("judged_by") != "engine":
        valid_gaps.append("the oracle verdict is the script's own (this quest began before the engine judged oracles): it is not independent evidence")
    validated = matched and not valid_gaps

    # statistically adequate
    stat_gaps = gaps["statistically_adequate"]
    for name in precision_missed or []:
        stat_gaps.append(f"the target precision was not reached for {name}")
    for gap in statistics_gaps or []:
        stat_gaps.append(f"the statistics are not shown to be adequate: {gap}")
    if protocol is not None and int((manifest_record or {}).get("failed_trials") or 0) and not str(protocol.get("failure_policy") or "").strip():
        stat_gaps.append(
            f"{manifest_record['failed_trials']} trial(s) failed, and the protocol does not say how a failed trial is treated (failure_policy)"
        )
    adequate = validated and not stat_gaps

    # publication ready
    ready_gaps = gaps["publication_ready"]
    for amendment in _frozen.post_hoc(quest_root):
        ready_gaps.append(
            f"the protocol was amended after results were seen (amendment {amendment.get('n')}: "
            f"{'; '.join(amendment.get('changes') or ['no listed change'])}); the paper must say so, and the earlier run is archived"
        )
    review = state.get("review") or {}
    fi = quest_root / ".fi"
    if (fi / "human_review.json").is_file() and not (fi / "human_review_answer.json").is_file():
        # The review's verdict is the model's; the quest is waiting for a person to accept, reject or refine it.
        ready_gaps.append("the review is waiting for your decision (accept, reject or refine)")
    if not review:
        ready_gaps.append("the paper has not been reviewed yet")
    else:
        if review.get("verdict") != "accept":
            ready_gaps.append(f"the review verdict is {review.get('verdict') or 'not accept'}")
        if review.get("must_flag_hits"):
            ready_gaps.append(f"the review left {len(review['must_flag_hits'])} must-fix finding(s)")
    ready = adequate and not ready_gaps

    reached = [executed, reconciled, matched, validated, adequate, ready]
    status = "not_executed"
    for level, ok in zip(LEVELS, reached):
        if not ok:
            break
        status = level
    # Only the gaps that stand between the quest and its NEXT level are the gaps of the quest.
    next_level = LEVELS[LEVELS.index(status) + 1] if status in LEVELS and status != LEVELS[-1] else None
    if status == "not_executed":
        next_level = LEVELS[0]
    ladder = [
        {
            "level": level, "reached": ok, "assurance_claim": INFO[level]["assurance_claim"],
            "known_blind_spots": INFO[level]["known_blind_spots"],
            "evidence_artifacts": [a for a in INFO[level]["artifacts"] if (quest_root / a).exists()],
            "gaps": gaps[level],
        }
        for level, ok in zip(LEVELS, reached)
    ]
    return {
        "status": status,
        "levels": dict(zip(LEVELS, reached)),
        "next_level": next_level,
        "gaps": gaps.get(next_level, []) if next_level else [],
        "all_gaps": {k: v for k, v in gaps.items() if v},
        "ladder": ladder,
        "rigor_profile": settings.get("rigor_profile") or "default",
    }


def upgrade(record: Any) -> Any:
    """A record written under the older names, read under the current ones. A record that already has a ``ladder`` is returned
    as it is. The older levels stood for less than the newer ones (no runtime manifest, oracle verdicts the script wrote
    itself, no statistics check), so ``validated_against_oracle`` and ``publication_ready`` are read as
    ``internally_reconciled`` and the record says so."""
    if not isinstance(record, dict) or "ladder" in record or "levels" not in record:
        return record
    old = record.get("levels") or {}
    levels = {level: False for level in LEVELS}
    levels["executed"] = bool(old.get("executed"))
    levels["internally_reconciled"] = bool(old.get("internally_consistent"))
    status = LEGACY.get(str(record.get("status")), str(record.get("status") or "not_executed"))
    if status not in LEVELS:
        status = "executed" if levels["executed"] else "not_executed"
    if not levels["internally_reconciled"] and status != "not_executed":
        status = "executed"
    old_gaps = dict(record.get("all_gaps") or {})
    all_gaps = {
        "executed": old_gaps.get("executed", []),
        "internally_reconciled": old_gaps.get("internally_consistent", []),
    }
    upgraded = {
        **record,
        "status": status,
        "levels": levels,
        "next_level": "protocol_runtime_matched" if levels["internally_reconciled"] else (
            "internally_reconciled" if levels["executed"] else "executed"
        ),
        "all_gaps": {k: v for k, v in all_gaps.items() if v},
        "legacy": (
            "This record was written before the levels were renamed; it is read under the current names, and the checks the "
            "newer levels stand for (runtime manifest, engine-judged oracles, statistics) were not part of it."
        ),
    }
    if levels["internally_reconciled"]:
        upgraded["gaps"] = ["this quest's record predates the runtime, engine-judged and statistical checks"]
    return upgraded


_NOT_EXECUTED_CLAIM = "The experiment has not produced results (it has not run, or it failed)."


def summary_line(record: dict[str, Any], *, technical: bool = False) -> str:
    """One plain sentence for the CLI and the VSCode chat: what was shown, then what stands in the way of more — never the
    six level identifiers themselves (a scientist reading this after a run should not have to look up what
    ``internally_reconciled`` means). ``technical=True`` returns the old compact ``<level>; to reach <level>: <gap>`` form,
    for the record kept on disk and for anything reading a level name back out of this line."""
    record = upgrade(record)
    status = str(record.get("status") or "not_executed")
    gaps = record.get("gaps") or []
    if technical:
        tail = f"; to reach {record.get('next_level')}: {gaps[0]}" + (f" (+{len(gaps) - 1} more)" if len(gaps) > 1 else "") if gaps else ""
        return f"{status}{tail}"
    claim = INFO.get(status, {}).get("assurance_claim") or _NOT_EXECUTED_CLAIM
    tail = f" Next: {gaps[0]}" + (f" (+{len(gaps) - 1} more)" if len(gaps) > 1 else "") if gaps else ""
    return f"{claim}{tail}"
