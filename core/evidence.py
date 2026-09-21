"""How much of a quest's result has been checked against something other than itself.

A run used to end in one green state. The three stored SIR runs of one topic show why that misleads: each passed its number
audit, its statistics audit and its provenance audit, and one of them had swept a different grid than the topic asked for, so a
pass meant the paper copied the script's output faithfully and nothing more. The states below are ordered, each needs the one
before it, and each says what was actually checked:

* ``executed`` — the experiment ran to the end and printed results;
* ``internally_consistent`` — and the audits that compare the paper with those results (numbers, statistics, provenance) found
  nothing: the paper says what the script printed;
* ``validated_against_oracle`` — and the script was held to the plan's protocol (the grid, the runs, the thresholds it fixed),
  passed the oracles the protocol declares (something independent of its own numbers), and no warning from its own numerics
  was accepted: the numbers are evidence about the topic, not only about the script;
* ``publication_ready`` — and every precision target the protocol set was reached, and the review accepted the paper with no
  must-fix finding.

``gaps`` lists, one sentence each, what stands between the quest and the next level. This module reads the quest's records and
the final state; it needs no model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import frozen_protocol as _frozen
from . import run_manifest as _run_manifest

LEVELS = ("executed", "internally_consistent", "validated_against_oracle", "publication_ready")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _audit_gaps(paper_dir: Path) -> tuple[bool, list[str]]:
    """Whether the paper-vs-results audits that ran all passed, and what they found."""
    ran = 0
    gaps: list[str] = []
    for name, label in (("numeric_audit", "the number check"), ("statistics_audit", "the statistics check"),
                        ("provenance_audit", "the provenance check")):
        report = _json(paper_dir / f"{name}.json")
        if not isinstance(report, dict) or report.get("skipped"):
            continue
        ran += 1
        if report.get("ok") is False:
            findings = report.get("findings") or []
            gaps.append(f"{label} reported {len(findings)} finding(s)")
    return ran > 0 and not gaps, gaps


def assess(
    quest_root: Path, state: dict[str, Any], *, precision_missed: list[str] | None = None,
    settings: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The quest's evidence status and the gaps below the next level.

    ``state`` is the final (or paused) state; ``precision_missed`` names the probabilities whose pooled interval did not
    reach the protocol's target; ``settings`` holds ``protocol_check``, ``oracle_check`` and ``numeric_warnings`` as the run
    had them (a check that was turned off is a gap, not a pass)."""
    settings = settings or {}
    needs = quest_root / "needs"
    gaps: dict[str, list[str]] = {level: [] for level in LEVELS}

    # executed
    exec_result = state.get("exec_result") or {}
    executed = bool(state.get("result_json")) and exec_result.get("returncode", 0) == 0
    if not executed:
        gaps["executed"].append("the experiment has not produced results (it has not run, or it failed)")

    # internally consistent
    audits_ok, audit_gaps = _audit_gaps(quest_root / "paper")
    consistent = executed and audits_ok
    if executed and not audits_ok:
        gaps["internally_consistent"] += audit_gaps or ["the paper has not been checked against the results yet"]

    # validated against an oracle
    # The protocol the gates held the run to is the frozen one, and never the design in the state, which a redesign can
    # leave without a protocol (a quest begun before the freeze existed has none frozen).
    frozen = _frozen.load(quest_root)
    if frozen is not None:
        protocol = frozen.get("protocol") if isinstance(frozen.get("protocol"), dict) else None
    else:
        design = state.get("design") or {}
        protocol = design.get("protocol") if isinstance(design.get("protocol"), dict) else None
    validated = consistent
    problems = gaps["validated_against_oracle"]
    if frozen is None and executed:
        problems.append("the protocol was not frozen before the run (the quest began before the freeze existed): nothing shows the run was held to the protocol it started with")
    if frozen is not None and frozen.get("problem"):
        problems.append(str(frozen["problem"]))
    if protocol is None:
        problems.append("the plan fixes no protocol, so nothing held the experiment to the grid, runs and thresholds it was meant to use")
    else:
        protocol_record = _json(needs / "PROTOCOL_CHECK.json")
        if settings.get("protocol_check") == "off":
            problems.append("the protocol check was turned off")
        elif not isinstance(protocol_record, dict) or protocol_record.get("status") != "ok":
            status = protocol_record.get("status") if isinstance(protocol_record, dict) else "not run"
            problems.append(f"the script was not shown to follow the protocol (protocol check: {status})")
    if settings.get("run_manifest_check") == "off":
        problems.append("the run manifest check was turned off")
    elif protocol is not None and _run_manifest.checkable(protocol):
        manifest_record = _json(needs / "RUN_MANIFEST_CHECK.json")
        status = manifest_record.get("status") if isinstance(manifest_record, dict) else None
        if status == "single_script":
            problems.append("the quest ran as one script, which writes no run manifest: nothing shows the run did what the protocol fixed (execution.split_analysis)")
        elif status == "not_applicable":
            pass  # a deterministic study runs as one script and has no per-trial outcomes to record
        elif status != "ok":
            problems.append(f"the run was not shown to have done what the protocol fixed (run manifest check: {status or 'not run'})")
        elif int((manifest_record or {}).get("failed_trials") or 0) and not str(protocol.get("failure_policy") or "").strip():
            problems.append(
                f"{manifest_record['failed_trials']} trial(s) failed, and the protocol does not say how a failed trial is treated (failure_policy)"
            )
    oracle_record = _json(needs / "ORACLE_CHECK.json")
    if settings.get("oracle_check") == "off":
        problems.append("the oracle check was turned off")
    elif protocol is not None and (not isinstance(oracle_record, dict) or oracle_record.get("status") != "ok"):
        status = oracle_record.get("status") if isinstance(oracle_record, dict) else "not run"
        problems.append(f"the script did not pass an independent oracle (oracle check: {status})")
    warnings = _json(needs / "NUMERIC_WARNINGS.json")
    if state.get("numeric_warnings_accepted"):
        problems.append("the run's numeric warnings were accepted as they were (see needs/NUMERIC_WARNINGS.json)")
    elif settings.get("numeric_warnings") == "off":
        problems.append("the numeric-warning check was turned off")
    elif isinstance(warnings, list) and warnings and settings.get("numeric_warnings") == "warn":
        problems.append("the run's numerics warned and the check only recorded it (engine.numeric_warnings: warn)")
    validated = validated and not problems

    # publication ready
    ready_gaps = gaps["publication_ready"]
    for name in precision_missed or []:
        ready_gaps.append(f"the target precision was not reached for {name}")
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
    ready = validated and not ready_gaps

    reached = [executed, consistent, validated, ready]
    status = "not_executed"
    for level, ok in zip(LEVELS, reached):
        if not ok:
            break
        status = level
    # Only the gaps that stand between the quest and its NEXT level are the gaps of the quest.
    next_level = LEVELS[LEVELS.index(status) + 1] if status in LEVELS and status != LEVELS[-1] else None
    if status == "not_executed":
        next_level = LEVELS[0]
    return {
        "status": status,
        "levels": dict(zip(LEVELS, reached)),
        "next_level": next_level,
        "gaps": gaps.get(next_level, []) if next_level else [],
        "all_gaps": {k: v for k, v in gaps.items() if v},
    }


def summary_line(record: dict[str, Any]) -> str:
    """One line for the CLI and the VSCode chat."""
    status = str(record.get("status") or "not_executed")
    gaps = record.get("gaps") or []
    tail = f"; to reach {record.get('next_level')}: {gaps[0]}" + (f" (+{len(gaps) - 1} more)" if len(gaps) > 1 else "") if gaps else ""
    return f"{status}{tail}"
