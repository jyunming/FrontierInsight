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
* ``independently_validated`` — and the engine ran the simulation on each oracle's case and judged what it measured against
  the protocol's expected values and tolerances, and every one passed (a value the script reported itself never
  counts, on one script or two);
* ``statistically_adequate`` — and every headline metric has a declared estimator matched to its data, the contrasts carry
  engine-computed p-values, and every precision target was reached;
* ``publication_ready`` — and the review accepted the paper with no must-fix finding, a person accepted the result with an
  explicit yes (an automatic accept, "I did not check" or "partly" stays one level below, core/acceptance.py), the protocol was not amended after the
  results were seen, a design revised after the experiment had run was confirmed by a run on data or seeds never seen
  after its last change (core/disclosure.py), and the evidence gate, the design methodology audit and the claim check
  each left a receipt that says it passed (core/receipts.py; a missing one is a gap).

``gaps`` lists, one sentence each, what stands between the quest and the next level. This module reads the quest's records and
the final state; it needs no model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import acceptance as _acceptance
from . import disclosure as _disclosure
from . import frozen_protocol as _frozen
from . import optimise as _optimise
from . import optimum_check as _optimum_check
from . import oracle_check as _oracle_check
from . import receipts as _receipts
from . import rerun_from as _rerun_from
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
            "FI runs the trials and writes their record (core/trial_runner.py), so the counts are FI's own; what one "
            "trial computes is still the simulation's code, and nothing here checks that it computes the right thing "
            "(the oracle level is for that). A simulation on the older contract, which ran its own loop, is marked "
            "self_reported and does not reach this level.",
        ],
        "artifacts": ["needs/FROZEN_PROTOCOL.json", "needs/PROTOCOL_CHECK.json", "needs/RUN_MANIFEST_CHECK.json", "needs/NUMERIC_WARNINGS.json",
                      "raw/ledger.jsonl"],
    },
    "independently_validated": {
        "assurance_claim": (
            "The engine ran the simulation on each oracle's case and judged what it returned against the protocol's expected "
            "values and tolerances. A value the script reported itself (an oracle without a case, or a one-script quest's "
            "FI_ORACLE run) is judged too, but never counts toward this level, and a check that judged no value does not reach it. "
            "Under rigor_profile: research, a second model, shown by the record of the quest's model calls to be "
            "different from the one that wrote the checks, read them as they were frozen and gave a usable answer, and up "
            "to three invariant, symmetry or second-implementation checks for which FI could choose another setting "
            "also passed at a setting the code never saw."
        ),
        "known_blind_spots": [
            "The second model reads the checks; it does not work their expected values out again. A special or limiting "
            "case, a published value and a convergence rate are not run at a hidden setting (their expected value "
            "belongs to their own setting), nor is a check FI found no other setting for: a quest with only such checks "
            "reaches this level with no hidden setting at all.",
            "The expected values come from the plan (the same model that wrote it): a wrong closed form is passed by a wrong "
            "simulator that agrees with it. The engine checks that each one says where it comes from (a derivation with "
            "its steps, a source this quest retrieved, an equation of the plan's model, or a second implementation that "
            "shares no code), not that the derivation or the reading of the source is right. It checks that the "
            "simulation marks where it implements each equation the model computes the data with (a comment `# E1`), not "
            "that the code there computes it.",
            "A tolerance looser than the gap between the claimed method and a cruder one passes both; the record warns when the "
            "case names an order, and does not when it does not.",
        ],
        "artifacts": ["needs/ORACLE_CHECK.json", ".fi/oracle_review.json", "needs/HIDDEN_CHECK.json"],
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
        "assurance_claim": (
            "The review accepted the paper with no must-fix finding and a person accepted the result, the protocol was not "
            "amended after the results were seen, a design revised after the experiment had run was confirmed by a run on data or seeds never seen "
            "after its last change, and the evidence gate, the design methodology audit and the claim check each ran "
            "and passed, the claim check on the final draft (each left a record saying so; a check that is missing, was "
            "turned off or could not judge is not a pass)."
        ),
        "known_blind_spots": ["The review is a model's opinion and the accept a person's judgement; neither re-runs anything."],
        "artifacts": ["needs/DESIGN_HISTORY.json", "paper/review.json", "needs/receipts/evidence_gate.json",
                      "needs/receipts/design_audit.json", "needs/receipts/claim_check.json"],
    },
}

# What a search for the best design adds to a level (``study_type: find_best_design``): FI ran the search and wrote its
# record, and checked the best design at finer numerical settings (core/optimum_check.py). Each failed or unfinished part
# of that check is a gap at the level it bears on (core/optimum_check.py::evidence_gaps).
_OPTIMUM_INFO: dict[str, dict[str, Any]] = {
    "protocol_runtime_matched": {"artifacts": ["raw/optimisation_ledger.jsonl", "results/best_design.json"]},
    "independently_validated": {
        "known_blind_spots": [
            "For a search for the best design, FI evaluated the best design and the baseline again at finer numerical "
            "settings; that rules out a numerical error only. The improvement is an improvement within the plan's "
            "model: an error of the model itself is the same at every setting.",
        ],
        "artifacts": ["needs/OPTIMUM_CHECK.json"],
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


#: Steps a finished quest that wrote a paper has completed; a seal without them sealed an incomplete run.
_SEALED_STEPS = ("write", "review")


#: The quest's record files: what it tried, the choices along the way, and every model call it made. A quest that
#: never wrote to one still has it (an empty file), so the seal can name it.
SEALED_LEDGERS = (".fi/attempts.jsonl", ".fi/branch_ledger.jsonl", ".fi/model_calls.jsonl",
                  ".fi/shadow_recommendations.jsonl")
#: The record of the literature search queries (each entry with its digest): what the quest searched for.
SEALED_QUERIES = ".fi/literature_queries.json"
#: Files a finished quest's seal names by hash (quest-relative), every one of them required: what the evidence, the
#: record files and the search queries were when the quest sealed its trace. The paper is named by ``paper_sha256``.
SEALED_FILES = ("needs/EVIDENCE.json", *SEALED_LEDGERS, SEALED_QUERIES)
#: A seal that does not say which record files it names (written before the shadow record existed) is not required to
#: name it; a seal that says ``sealed_records: 2`` names all of :data:`SEALED_FILES`.
SEAL_RECORDS = 2
_SEALED_FILES_V1 = tuple(f for f in SEALED_FILES if f != ".fi/shadow_recommendations.jsonl")


def _required_files(seal: dict[str, Any]) -> tuple[str, ...]:
    return SEALED_FILES if (seal.get("sealed_records") or 1) >= SEAL_RECORDS else _SEALED_FILES_V1


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _trace_completeness_gaps(trace: Path, audit_log: Any, *, sealing: bool = False) -> list[str]:
    """What keeps a trace whose hash chain checks out from being the quest's complete record: no seal as its very last
    event (``quest_finalized``), events that could not be written, a seal that does not count the events before it,
    anything written after the seal, a seal without the paper's steps, or a file the seal names that changed since. A
    valid prefix alone (one ``quest_started`` line) proves only that what is there was not changed.

    ``sealing``: the finishing quest is assessing itself just before it writes the seal (which names this assessment's
    hash, so it cannot come first); only what can already be known counts: lost events so far and the paper's steps.
    The rest is checked when the record is read (:func:`verify_seal`)."""
    try:
        events = audit_log.read(trace)
    except Exception:  # noqa: BLE001
        return ["the quest's decision trace (.fi/audit.jsonl) could not be read"]
    if sealing:
        gaps = []
        lost = audit_log.lost_writes(trace)
        if lost:
            gaps.append(f"{lost} event(s) of the quest's decision trace could not be written")
        done = {e.get("node") for e in events if e.get("kind") == "node_completed"}
        missing = [s for s in _SEALED_STEPS if s not in done]
        if missing:
            gaps.append(f"the decision trace names no completed {', '.join(missing)} step")
        return gaps
    seals = [i for i, e in enumerate(events) if e.get("kind") == "quest_finalized"]
    if not seals:
        if (trace.parent / "pause.json").is_file():
            return ["the quest has not finished yet (it is paused): its decision trace is sealed when it finishes"]
        return ["the quest's decision trace has no final seal: the quest did not finish, or the end of its record was lost"]
    at = seals[-1]
    seal = events[at]
    gaps: list[str] = []
    if int(seal.get("write_errors") or 0) > 0:
        gaps.append(f"{seal['write_errors']} event(s) of the quest's decision trace could not be written")
    lost_now = audit_log.lost_writes(trace)
    if lost_now > int(seal.get("write_errors") or 0):
        gaps.append(f"{lost_now - int(seal.get('write_errors') or 0)} event(s) of the decision trace could not be "
                    "written after the quest was sealed")
    if int(seal.get("records_not_written") or 0) > 0:
        gaps.append(f"{seal['records_not_written']} of the quest's attempt record(s) could not be written")
    if int(seal.get("model_calls_not_written") or 0) > 0:
        gaps.append(f"{seal['model_calls_not_written']} of the quest's model call record(s) could not be written")
    calls = seal.get("model_calls") if isinstance(seal.get("model_calls"), dict) else None
    if calls is None:
        gaps.append("the decision trace's final seal says nothing about the quest's record of its model calls")
    else:
        # What the finishing quest found missing from its record of model calls (fewer lines than calls, a call not
        # in it, an answering model a connection should have named); lost lines are counted just above.
        gaps.extend(f"the record of the quest's model calls is incomplete: {g}" for g in calls.get("gaps") or []
                    if "could not be written" not in str(g))
    if seal.get("events_before") != at:
        gaps.append("the decision trace's final seal does not count the events before it (events are missing or were added)")
    # A person changing the title after the quest finished (core/quest_title.py) is recorded after the seal, with the
    # paper's hash before and after: those events are the one thing a seal may be followed by, and the paper is
    # checked against the hash the last of them left.
    renames = events[at + 1:] if len(audit_log.after_title_changes(events)) == at + 1 else []
    if any(e.get("kind") in ("node_started", "node_completed") for e in events[at + 1:]):
        gaps.append("steps ran after the decision trace's final seal (the quest was re-run and did not finish again)")
    elif at != len(events) - 1 and not renames:
        gaps.append(f"{len(events) - 1 - at} event(s) were written after the decision trace's final seal, which "
                    "therefore does not cover them")
    missing = [s for s in _SEALED_STEPS if s not in (seal.get("nodes_completed") or [])]
    if missing:
        gaps.append(f"the decision trace's final seal names no completed {', '.join(missing)} step")
    quest_root = trace.parent.parent
    files = seal.get("files") if isinstance(seal.get("files"), dict) else None
    if files is None:
        gaps.append("the decision trace's final seal names no hash of the evidence and the quest's records")
    else:
        for rel in _required_files(seal):
            if not files.get(rel):
                gaps.append(f"the decision trace's final seal names no hash of {rel}")
            elif files[rel] != _file_sha256(quest_root / rel):
                gaps.append(f"{rel} changed after the quest was sealed")
    paper = seal.get("paper_path")
    if not paper or not seal.get("paper_sha256"):
        gaps.append("the decision trace's final seal names no hash of the paper")
    else:
        expected = seal.get("paper_sha256")
        for rename in renames:
            if rename.get("paper_path") != paper or rename.get("paper_sha256_before") != expected:
                break
            expected = rename.get("paper_sha256_after") or expected
        if expected != _file_sha256(quest_root / str(paper)):
            gaps.append(f"the paper ({paper}) changed after the quest was sealed")
    return gaps

# What keeps a two-stage quest's result preliminary, by its stage (core/phased.py ``status``); a confirmed one has no gap.
_PHASED_GAPS = {
    "explore": ("the numbers come from the exploration stage, where the design could still be changed after its results "
                "were seen; they have not been confirmed on data or seeds exploration never saw"),
    "confirming": "the confirm run of the frozen design has not finished",
    "confirm_failed": "the confirm run of the frozen design produced no result, so nothing is confirmed",
    "confirm_reused": ("the frozen design was changed or run again after its confirm results were seen, so the numbers "
                       "are no longer from one untouched confirm run"),
    "compromised": ("the confirm run could not be kept apart from exploration (its held-back data, its new seeds or its "
                    "result), so nothing is confirmed"),
    "not_confirmable": ("no data was held back and the study gives the same numbers whatever its seeds, so a confirm "
                        "run could only repeat exploration's run; nothing is confirmed"),
}


def assess(
    quest_root: Path, state: dict[str, Any], *, precision_missed: list[str] | None = None,
    settings: dict[str, str] | None = None, statistics_gaps: list[str] | None = None,
    oracle_source_gaps: list[str] | None = None, equation_label_gaps: list[str] | None = None,
    independence_gaps: list[str] | None = None,
) -> dict[str, Any]:
    """The quest's evidence status, the ladder with what each level guarantees, and the gaps below the next level.

    ``state`` is the final (or paused) state; ``precision_missed`` names the probabilities whose pooled interval did not
    reach the protocol's target; ``statistics_gaps`` says what keeps the statistics from being adequate
    (:func:`core.metric_spec.coverage_gaps`); ``oracle_source_gaps`` names each oracle whose expected value has no
    source a reader can check (:func:`core.oracle_check.source_gaps`), a gap below ``independently_validated``; ``equation_label_gaps`` says which equations of the plan's model the
    simulation does not mark where it implements them (a comment ``# E1``), a gap at the same level: the oracles test the
    code against the model, and without the labels nobody can tell which code an equation's check tests;
    ``independence_gaps`` (``rigor_profile: research`` only) says why the checks are not shown to be independent of the
    model that wrote them: no usable second reading by a different model (:func:`core.oracle_review.independence_gaps`),
    or a check that failed or was not run at a setting the code never saw (:func:`core.hidden_check.evidence_gaps`), a
    gap at the same level; ``settings`` holds ``protocol_check``, ``oracle_check``, ``numeric_warnings``,
    ``run_manifest_check`` and ``rigor_profile`` as the run had them (a check that was turned off is a gap, not a pass), and
    ``evidence_gate`` / ``claim_check`` / ``design_audit`` as ``off`` or ``not_applicable`` when the run had them so."""
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

    # A search for the best design: FI's check of the best design at finer numerical settings (core/optimum_check.py)
    # holds each level up to the one its failed or unfinished parts keep the quest below.
    try:
        optimum_gaps = _optimum_check.evidence_gaps(quest_root, protocol) if protocol is not None else {}
    except Exception as e:  # noqa: BLE001 -- a record that cannot be read is a gap, never a pass
        optimum_gaps = {"independently_validated": [
            f"the check of the best design at finer numerical settings could not be read ({type(e).__name__})"]}

    # protocol matched at runtime
    matched_gaps = gaps["protocol_runtime_matched"]
    matched_gaps.extend(optimum_gaps.get("protocol_runtime_matched", []))
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
            pass  # a deterministic study run as one script (split_analysis: false) has no per-trial outcomes to record
        elif status == "self_reported":
            matched_gaps.append(
                "the trial record is the simulation's own statement (it ran its own loop): FI did not run the trials "
                "itself, so nothing independent shows it did what the protocol fixed"
            )
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
    if isinstance(environment, dict) and environment.get("packages_error"):
        # Not "no packages": which packages the run had is unknown, so it cannot be set up again as it ran.
        matched_gaps.append(
            "the packages the run had could not be listed, so its environment cannot be reproduced "
            f"({str(environment['packages_error'])[:160]}; see needs/ENVIRONMENT.json)"
        )
    elif isinstance(environment, dict) and environment.get("packages_source"):
        # Listed without pip (pip freeze failed): versions are known, but an editable or direct install among them may
        # not be installable again as it was.
        matched_gaps.append(
            "the packages the run had were listed without pip (pip freeze failed), so an editable or local install among "
            "them may not be installable again as it was (see needs/ENVIRONMENT.json)"
        )
    if isinstance(environment, dict) and environment.get("isolated") is False:
        matched_gaps.append(
            "the run shared its Python environment with other quests (execution.shared_interpreter / "
            "system_site_packages); a package installed for a different quest could have affected this one (see needs/ENVIRONMENT.json)"
        )
    matched = reconciled and not matched_gaps

    # independently validated
    valid_gaps = gaps["independently_validated"]
    oracle_record = _json(needs / "ORACLE_CHECK.json")
    corrected = oracle_record.get("corrected") if isinstance(oracle_record, dict) else None
    if protocol is not None and isinstance(corrected, dict):
        # A correction to another model's value: that value was taken because it lies near what was measured, so the
        # check it makes pass confirms nothing independent. (A correction from the plan's own arithmetic was not chosen
        # by the measurement, and is no gap.)
        valid_gaps.extend(f"the expected value of the known-answer check '{n}' was corrected to another model's value "
                          "that agrees with the measurement: nothing independent confirms it"
                          for n, c in corrected.items() if isinstance(c, dict) and c.get("source") == "recompute")
        # A correction from the plan's arithmetic alone (an earlier FI made those): the calculator may have misread
        # a correct derivation, and nothing else confirms it.
        valid_gaps.extend(f"the expected value of the known-answer check '{n}' was corrected from the plan's own "
                          "arithmetic alone: nothing independent confirms it"
                          for n, c in corrected.items()
                          if isinstance(c, dict) and c.get("source") == "arithmetic" and not c.get("confirmed_by"))
    fitted = oracle_record.get("fitted_to_test_run") if isinstance(oracle_record, dict) else None
    if protocol is not None and isinstance(fitted, list) and fitted:
        # The plan changed these checks after FI's test run measured them, and the change makes that measurement pass:
        # the value checked is the one the simulation produced, so it is no independent check of it.
        valid_gaps.extend(f"the known-answer check '{n}' was changed after a test run measured it, so that the "
                          "measured value passes: nothing independent confirms it" for n in fitted)
    if settings.get("oracle_check") == "off":
        valid_gaps.append("the oracle check was turned off")
    elif isinstance(oracle_record, dict) and oracle_record.get("status") == "went_on_failing":
        # A check failed and the quest went on: a person's named choice, or automatic for an exploration. Each is a gap
        # in its own plain words (core/accepted_checks.py), so the result never reaches this level or the ones above.
        from .accepted_checks import gap as _went_on_gap

        entries = [e for e in oracle_record.get("went_on") or []
                   if isinstance(e, dict) and (e.get("name") or e.get("unmeasured"))]
        valid_gaps.extend([_went_on_gap(e) for e in entries]
                          or ["a known-answer check failed and the quest went on (oracle check: went_on_failing)"])
    elif protocol is not None and (not isinstance(oracle_record, dict) or oracle_record.get("status") != "ok"):
        status = oracle_record.get("status") if isinstance(oracle_record, dict) else "not run"
        valid_gaps.append(f"the script did not pass an independent oracle (oracle check: {status})")
    elif protocol is not None and oracle_record.get("judged_by") != "engine":
        valid_gaps.append("the oracle verdict is the script's own (this quest began before the engine judged oracles): it is not independent evidence")
    elif protocol is not None and (scripted := _oracle_check.script_measured(_oracle_check.last_judged(oracle_record))):
        # One rule for every quest, one script or two: a value the script reported is its own word, never independent
        # evidence. Only the engine running the simulation on the oracle's case measures one.
        if oracle_record.get("contract") == "trial":
            valid_gaps.append(
                f"the value of {', '.join(scripted)} came from the script's own oracle(), not from the engine running the simulation "
                "on the oracle's case: a script that returns a closed form without simulating would pass (give the oracle a `case` and a "
                "`measure`; a check of a random simulation that needs many trials, such as a probability, cannot have one, "
                "and stays the script's own word)"
            )
        else:
            valid_gaps.append(
                f"the value of {', '.join(scripted)} was reported by the script itself, not measured by FI running the simulation: "
                "a script that prints the expected answer without simulating would pass. For FI to measure it, the simulation "
                "needs its own script that FI calls on the oracle's case itself (the default, `execution.split_analysis: "
                "auto`; this quest ran as one script because `split_analysis` is false, the code-writing step returned one "
                "script, or the code was written before every simulation got its own script; the simulation script "
                "defines `run_cell`, or `run_trial` for a study with randomness), and the "
                "oracle needs a `case` and a `measure`"
            )
    elif protocol is not None and (unpassed := _oracle_check.not_passed(_oracle_check.last_judged(oracle_record))):
        # The gate only writes "ok" when every oracle it judged passed; a record that says otherwise is not taken on its word.
        named = ", ".join(repr(n) for n in unpassed)
        valid_gaps.append(
            f"the oracle {named} has no measured value, or its value is outside the tolerance, although the oracle check "
            "recorded it as passed (see needs/ORACLE_CHECK.json)" if len(unpassed) == 1 else
            f"the oracles {named} have no measured value, or their values are outside the tolerance, although the oracle "
            "check recorded them as passed (see needs/ORACLE_CHECK.json)"
        )
    elif protocol is not None and not _oracle_check.engine_passed(_oracle_check.last_judged(oracle_record)):
        # "ok" with nothing judged checked nothing: at least one value FI measured by running the simulation must have passed.
        valid_gaps.append(
            "the oracle check judged no value: FI did not measure and pass any oracle by running the simulation, so nothing "
            "independent of the script was checked (see needs/ORACLE_CHECK.json)"
        )
    elif protocol is not None and (unmeasured := [
        str(o["name"]).strip() for o in _oracle_check.declared(protocol)
        if str(o["name"]).strip().lower() not in {
            n.strip().lower() for n in _oracle_check.engine_passed(_oracle_check.last_judged(oracle_record))
        }
    ]):
        # Each oracle the protocol declares must have been measured and passed, not just one of them (a record written
        # before an oracle was added to the plan does not cover it).
        one = len(unmeasured) == 1
        valid_gaps.append(
            f"the oracle check did not measure {', '.join(repr(n) for n in unmeasured)}: FI never ran the simulation on "
            f"{'its case' if one else 'their cases'} and compared the {'value' if one else 'values'} "
            "(run the experiment again so FI checks them)"
        )
    if protocol is not None:
        # An expected value nobody can trace (no reference, or a source this quest did not retrieve) is the plan's word
        # only: a check that agrees with it shows the script agrees with the plan, not that either is right.
        valid_gaps.extend(
            f"{g}, so the value the check expects cannot be checked by a reader" for g in oracle_source_gaps or []
        )
        # Which code implements which equation of the model: an oracle that rests on E1 checks the code the simulation
        # says is E1. Only the labels are read, not the mathematics behind them.
        valid_gaps.extend(equation_label_gaps or [])
        # Under research: a second, different model read the checks, and the checks held at a setting the code never saw.
        valid_gaps.extend(independence_gaps or [])
    valid_gaps.extend(optimum_gaps.get("independently_validated", []))
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
    stat_gaps.extend(optimum_gaps.get("statistically_adequate", []))
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
    pending = (fi / "human_review.json").is_file() and not (fi / "human_review_answer.json").is_file()
    if pending:
        # The review's verdict is the model's; the quest is waiting for a person to accept, reject or refine it.
        ready_gaps.append(_acceptance.WAITING_GAP)
    # Who accepted it: an accept no person made (auto_accept_on_pass, or no review pause at all) is marked so and keeps
    # the result one level below publication_ready. No record of a person counts as nobody (core/acceptance.py).
    # This quest's own paper, also when the folder was copied or moved after the paper was written.
    paper = _rerun_from.in_quest(state.get("paper_md"), quest_root) or quest_root / "paper" / "paper.md"
    paper_hash = _receipts.sha256(Path(str(paper)).read_bytes()) if Path(str(paper)).is_file() else ""
    accepted_by = _acceptance.accepted_by(state, pending=pending, paper_sha256=paper_hash)
    if accepted_by == "automatic":
        ready_gaps.append(_acceptance.NO_PERSON_GAP)
    elif accepted_by == "person":
        # Only a person's explicit "yes" (they reviewed the evidence and accept the claims and the limits listed) goes
        # on to publication_ready: "I did not check" and "partly" (with its note) each leave a gap.
        person_gap = _acceptance.review_gap(state.get("acceptance"))
        if person_gap:
            ready_gaps.append(person_gap)
    if not review:
        ready_gaps.append("the paper has not been reviewed yet")
    else:
        # A provider/parse failure makes the engine record verdict="accept" so the
        # quest still finishes and its outputs still render -- that is a flow
        # decision, not a review outcome, and must not read as one here. Panel
        # mode's per-persona receipts (review_panel) are the more precise
        # signal when they exist; single-reviewer mode falls back to the
        # review's own "status".
        review_panel = state.get("review_panel")
        if isinstance(review_panel, list) and review_panel:
            failed_roles = [
                str(p.get("persona")) for p in review_panel
                if isinstance(p, dict) and p.get("status") == "error"
            ]
            if failed_roles:
                ready_gaps.append(
                    f"the review panel got no real response from: {', '.join(failed_roles)} "
                    "(a fabricated accept, not a real review)"
                )
        elif review.get("status") == "unreviewed":
            ready_gaps.append("the review could not be completed (a provider/parse failure); its accept is not a real review outcome")
        if review.get("verdict") != "accept":
            ready_gaps.append(f"the review verdict is {review.get('verdict') or 'not accept'}")
        if review.get("must_flag_hits"):
            ready_gaps.append(f"the review left {len(review['must_flag_hits'])} must-fix finding(s)")
    # Under rigor_profile: research the panel knowingly ran on one model (engine.one_model_review).
    if settings.get("rigor_profile") == "research" and settings.get("one_model_review"):
        ready_gaps.append(
            "every reviewer used one model (engine.one_model_review: only one was available), so the review is one "
            "model's view"
        )
    # Reviewers set up on different models, but the connection did not say which model answered: unproven, not assumed.
    if settings.get("rigor_profile") == "research" and settings.get("review_identity_unverified"):
        ready_gaps.append(
            "the review's independence is unverified: the connection did not report which model answered the "
            f"{', '.join(map(str, settings['review_identity_unverified']))} reviewer(s), so they may all have been one model"
        )
    # A quest set up to explore: its result is preliminary whatever it passed (the person said so at the start).
    if settings.get("result_use") == "explore":
        ready_gaps.append(
            "this quest was set up to explore (what the result is for: explore, or not said), so its result is "
            "preliminary: run it again for research (`result_use: research`) to make it publication-ready"
        )
    # A quest run in two stages (engine.phased, core/phased.py): only the confirm run's result, on data or seeds the
    # exploration never saw, is more than preliminary.
    phased_gap = _PHASED_GAPS.get(str(settings.get("phased") or ""))
    if settings.get("phased") == "not_confirmable" and settings.get("phased_unconfirmable_why"):
        # A changed version of a study whose held-back rows an earlier version's confirm run already read.
        phased_gap = (f"this version of the study cannot be confirmed: {settings['phased_unconfirmable_why']}; the "
                      "numbers are exploratory")
    elif phased_gap and settings.get("phased_candidate"):
        # Only this version's own confirmation counts: an earlier version's (kept in .fi/confirmations.jsonl) does not.
        phased_gap += (f" (this is version {settings['phased_candidate']} of the study, changed after an earlier "
                       "version was confirmed; that confirmation does not count for it)")
    if settings.get("phased_record_gap"):
        ready_gaps.append(str(settings["phased_record_gap"]))
    if settings.get("phased") == "confirmed":
        # Held back is not unseen unless exploration could not read it (core/phased_isolation.py), and rows of one
        # subject or site split one by one are not unseen either (core/phased_data.py).
        if settings.get("phased_isolation") == "isolation_unverified":
            ready_gaps.append("the held-back rows were not shown to be out of exploration's reach ("
                              + str(settings.get("phased_isolation_why") or "no reason recorded")
                              + "), so the confirm run is not shown to be on data exploration never saw")
        if settings.get("phased_split_gap"):
            ready_gaps.append(str(settings["phased_split_gap"]))
    if phased_gap:
        ready_gaps.append(phased_gap)
    else:
        # A design revised after the experiment had run keeps the numbers exploratory unless a confirm run on data or
        # seeds never seen came after the last change: read from the records (.fi/phased.json), not the setting.
        revised = _disclosure.unconfirmed_gap(quest_root, no_simulation=bool(state.get("no_simulation_resolved")),
                                              survey=bool(state.get("survey_mode_resolved")))
        if revised:
            ready_gaps.append(revised)
    # Under rigor_profile: research the hash-chained trace is what the quest's decisions are audited from: missing, or
    # no longer checking out, the result cannot be shown to have come about as its records say.
    if settings.get("rigor_profile") == "research":
        from . import audit_log as _audit_log

        trace = quest_root / ".fi" / "audit.jsonl"
        if not trace.is_file():
            ready_gaps.append("the quest's decision trace (.fi/audit.jsonl) is missing")
        else:
            try:
                intact = _audit_log.verify(trace).ok
            except Exception:  # noqa: BLE001 -- a trace that cannot be read does not check out either
                intact = False
            if not intact:
                ready_gaps.append("the quest's decision trace (.fi/audit.jsonl) no longer checks out (its hash chain is broken)")
            else:
                ready_gaps.extend(_trace_completeness_gaps(trace, _audit_log, sealing=bool(settings.get("sealing"))))
    # Under any profile: a record file that held lines FI did not write when the quest started again (moved aside), or
    # that does not match FI's own note of how far it wrote it now (core/record_anchor.py). A valid hash chain alone
    # does not show this: a line chained onto the last one verifies.
    from . import audit_log as _trace_log
    from . import record_anchor as _record_anchor

    try:
        trace_events = _trace_log.read(quest_root / ".fi" / "audit.jsonl")
    except Exception:  # noqa: BLE001 -- an unreadable trace is reported above under research
        trace_events = []
    ready_gaps.extend(_record_anchor.evidence_gaps(quest_root / ".fi", trace_events))
    # The evidence gate, the design methodology audit and the claim check must each have run and judged. Their receipts
    # (core/receipts.py) are read: a missing, unreadable or malformed receipt is a gap, as is a check the person turned
    # off; only an explicit pass counts. It used to be the other way round (a gap only when a check reported a failure),
    # so three checks that never ran left a quest publication-ready.
    design_now = state.get("design")
    for check, name in _receipts.REQUIRED.items():
        setting = settings.get(check)
        if setting == "off":
            ready_gaps.append(f"{name} was turned off")
            continue
        if setting == "not_applicable":  # e.g. no design to audit when the data already exists (--analyze)
            continue
        status, receipt, problem = _receipts.read(quest_root, check)
        if problem or receipt is None:
            ready_gaps.append(f"{name} is not shown to have run: {problem} (needs/receipts/{check}.json)")
        elif status == "unknown":
            ready_gaps.append(f"{name} could not judge ({receipt.get('error') or 'no usable reply'})")
        elif status == "fail":
            ready_gaps.append(f"{name} did not pass: {receipt.get('detail') or 'see needs/receipts/' + check + '.json'}")
        elif status == "not_applicable":
            ready_gaps.append(f"{name} recorded itself as not applicable, which this quest does not allow")
        elif check == "claim_check" and not paper_hash:
            ready_gaps.append("there is no final paper for the claim check to cover")
        elif check == "claim_check" and (receipt.get("input_hashes") or {}).get("paper") != paper_hash:
            ready_gaps.append("the claim check did not run on the final draft (its record is for an earlier one)")
        elif (check == "design_audit" and isinstance(design_now, dict)
              and receipt.get("output_hash") != _receipts.sha256(_receipts.design_core(design_now))):
            ready_gaps.append("the design methodology audit judged a different design from the one that ran")
        elif check == "evidence_gate" and (stale := _receipts.stale_inputs(receipt, {
            k: v for k, v in _receipts.gate_inputs(state).items()
            # every input the receipt names, and always the analysis and the cross-check it judged; not the sources,
            # which the writing step trims to the ones the paper cites after the gate has judged them
            if k != "sources" and (k in (receipt.get("input_hashes") or {}) or k in ("analysis", "cross_check"))
        })):
            # The gate judged the analysis, the cross-check, the results... it was shown; any of them changed since is
            # evidence it did not judge.
            ready_gaps.append(
                "the evidence gate judged earlier inputs than the ones there are now (changed since: "
                + ", ".join(stale) + ")"
            )
        elif check == "evidence_gate" and (added := _receipts.unjudged_sources(receipt, _receipts.gate_inputs(state))):
            ready_gaps.append(f"{added} of the paper's sources were not among the ones the evidence gate judged")
    # What the quest's own state and records say about the same checks, as well: a receipt that reads "pass" while the
    # state says the check failed (a record left from an earlier pass) is not believed over the state.
    gate = state.get("evidence_assessment") or {}
    if gate.get("status") == "unknown":
        ready_gaps.append(f"the evidence gate could not be evaluated ({gate.get('failure') or 'no usable reply'})")
    elif gate.get("verdict") in ("insufficient", "broaden"):
        gaps_named = "; ".join(str(g) for g in gate.get("gaps") or []) or "none named"
        ready_gaps.append(f"the evidence gate judged the evidence {gate['verdict']} and the paper was written on it (gaps: {gaps_named})")
    critique_history = _json(needs / "DESIGN_CRITIQUE.json")
    critique_last = critique_history[-1] if isinstance(critique_history, list) and critique_history else None
    if isinstance(critique_last, dict) and critique_last.get("status") == "failed":
        ready_gaps.append(f"the design methodology audit did not complete: {critique_last.get('failure') or 'unknown reason'}")
    claim_failed = str(state.get("claim_check_failed") or "").strip()
    if claim_failed:
        ready_gaps.append(f"the claim check failed on the final draft: {claim_failed}")
    ready_gaps.extend(optimum_gaps.get("publication_ready", []))
    ready_gaps[:] = list(dict.fromkeys(ready_gaps))
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
    searched = _optimise.block_of(protocol) is not None
    ladder = [
        {
            "level": level, "reached": ok, "assurance_claim": INFO[level]["assurance_claim"],
            "known_blind_spots": INFO[level]["known_blind_spots"] + (
                _OPTIMUM_INFO.get(level, {}).get("known_blind_spots", []) if searched else []),
            "evidence_artifacts": [a for a in INFO[level]["artifacts"] + (
                _OPTIMUM_INFO.get(level, {}).get("artifacts", []) if searched else []) if (quest_root / a).exists()],
            "gaps": gaps[level],
        }
        for level, ok in zip(LEVELS, reached)
    ]
    record = {
        "status": status,
        "levels": dict(zip(LEVELS, reached)),
        "next_level": next_level,
        "gaps": gaps.get(next_level, []) if next_level else [],
        "all_gaps": {k: v for k, v in gaps.items() if v},
        "ladder": ladder,
        "rigor_profile": settings.get("rigor_profile") or "default",
    }
    kept = state.get("acceptance") if isinstance(state.get("acceptance"), dict) else {}
    if not accepted_by and not pending and kept.get("by") in ("person", "automatic"):
        # Accepted although the review did not accept it (its verdict is already the gap): still say who accepted.
        accepted_by = kept["by"]
    if accepted_by:
        # "automatic" is the mark of a result no person reviewed before it was accepted.
        record["accepted_by"] = accepted_by
        if kept.get("by") == accepted_by:
            record["acceptance"] = dict(kept)
        elif kept.get("by") == "person":  # a person accepted another version of the paper
            record["acceptance"] = {"by": "automatic", "via": "the paper changed after a person accepted it",
                                    "earlier": dict(kept)}
        else:
            record["acceptance"] = {"by": "automatic", "via": "none recorded (no review pause asked a person)"}
    if settings.get("phased"):
        record["phased"] = {"status": settings["phased"], "strategy": settings.get("phased_strategy") or "",
                            "why_no_data": settings.get("phased_why_no_data") or ""}
        if settings.get("phased_differs"):
            # A data quest's confirm numbers that differ from exploration's (core/phased_data.compare): not a gap.
            record["phased"]["confirm_differs"] = settings["phased_differs"]
        if settings.get("phased_candidate"):
            # Which version of the study this is, and what became of the earlier ones (core/confirmations.py).
            record["phased"]["version"] = int(settings["phased_candidate"])
            record["phased"]["earlier_versions"] = settings.get("phased_earlier") or ""
        if settings.get("phased_isolation"):
            # docker, encrypted+scanned, or isolation_unverified (core/phased_isolation.py).
            record["phased"]["isolation"] = settings["phased_isolation"]
    if settings.get("sealing") and settings.get("rigor_profile") == "research":
        # Written before the seal that names it: whoever reads it checks the seal (verify_seal).
        record["trace_seal"] = "pending"
    return record


def verify_seal(quest_root: Path, record: Any) -> Any:
    """The evidence record as it stands once the quest's seal is checked. Under ``rigor_profile: research`` a record is
    publication-ready only if the seal is the trace's last event, nothing was lost, and the evidence, the attempt
    records and the paper are what the seal names; otherwise it is read one level down, with the reasons. Whatever the
    record says about its own seal (``trace_seal``) is not believed: it is in the file the seal protects. A research
    record from before the seal named these files is read one level down too. Pure: it reads, never writes (not even
    to the trace it checks)."""
    if not isinstance(record, dict):
        return record
    from . import audit_log as _audit_log

    trace = quest_root / ".fi" / "audit.jsonl"
    if record.get("rigor_profile") != "research":
        # The record's own profile is in the file the seal protects: a seal that says research is believed over it.
        try:
            seals = [e for e in _audit_log.read(trace) if e.get("kind") == "quest_finalized"]
        except Exception:  # noqa: BLE001
            seals = []
        if not (seals and seals[-1].get("rigor_profile") == "research"):
            return record
    try:
        intact = trace.is_file() and _audit_log.verify(trace).ok
    except Exception:  # noqa: BLE001
        intact = False
    problems = (_trace_completeness_gaps(trace, _audit_log) if intact
                else ["the quest's decision trace (.fi/audit.jsonl) is missing or no longer checks out"])
    if not problems:
        return {**record, "trace_seal": "verified"}
    out = {**record, "trace_seal": "not_verified"}
    last = LEVELS[-1]
    all_gaps = {k: list(v) for k, v in (record.get("all_gaps") or {}).items()}
    all_gaps[last] = list(dict.fromkeys([*all_gaps.get(last, []), *problems]))
    out["all_gaps"] = all_gaps
    levels = dict(record.get("levels") or {})
    if levels.get(last):
        levels[last] = False
        out["levels"] = levels
        out["status"] = LEVELS[-2]
        out["next_level"] = last
    if out.get("next_level") == last:
        out["gaps"] = all_gaps[last]
    out["ladder"] = [
        {**step, "reached": bool(levels.get(step.get("level"), step.get("reached"))),
         **({"gaps": all_gaps[last]} if step.get("level") == last else {})}
        for step in record.get("ladder") or [] if isinstance(step, dict)
    ]
    return out


def read(quest_root: Path) -> dict[str, Any] | None:
    """``needs/EVIDENCE.json`` as every surface should show it: in the current names (:func:`upgrade`) and with its
    quest's seal checked (:func:`verify_seal`). ``None`` when there is none or it cannot be read."""
    record = _json(quest_root / "needs" / "EVIDENCE.json")
    return verify_seal(quest_root, upgrade(record)) if isinstance(record, dict) else None


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


def unassessed(error: str) -> dict[str, Any]:
    """The record kept when the assessment itself failed: no level is claimed, and the reason is the one gap. It replaces
    an earlier record, so a level worked out on a previous pass never outlives a failed assessment of this one."""
    gap = f"the evidence could not be assessed ({error})"
    return {
        "status": "not_executed",
        "assessment_failed": error,
        "levels": {level: False for level in LEVELS},
        "next_level": LEVELS[0],
        "gaps": [gap],
        "all_gaps": {LEVELS[0]: [gap]},
        "ladder": [],
    }


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
    if record.get("assessment_failed"):
        return f"The evidence could not be assessed ({record['assessment_failed']}); no level is claimed."
    claim = INFO.get(status, {}).get("assurance_claim") or _NOT_EXECUTED_CLAIM
    tail = f" Next: {gaps[0]}" + (f" (+{len(gaps) - 1} more)" if len(gaps) > 1 else "") if gaps else ""
    # The mark of an accept no person made, said on every surface even when another gap comes first.
    if record.get("accepted_by") == "automatic" and (not gaps or gaps[0] != _acceptance.NO_PERSON_GAP):
        tail += " Not reviewed by a person: it was accepted automatically."
    # A person's accept that was not a plain yes ("I did not check", "partly" and its note), likewise.
    person_mark = _acceptance.mark(record)
    if person_mark:
        tail += f" {person_mark}"
    return f"{claim}{tail}"
