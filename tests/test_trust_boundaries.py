"""The 2026-09-27 delta re-audit's trust boundaries (F2, F3, F4).

- F2: a result that is not accepted evidence (an exploration, evidence gaps) is written to Axon as preliminary, under
  its own kinds, and a later quest reads it as a reminder, never as a source to cite.
- F3: under ``rigor_profile: research`` a trace is complete only with a final seal that counts what came before it; a
  valid prefix alone (one ``quest_started`` line) is a gap.
- F4: each reviewer records who actually answered; reviewers set up on different models but answered by one pause the
  research quest, and going on anyway is an evidence gap. Calls made at the same time do not overwrite each other's
  record of who answered.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import audit_log, evidence
from core.engine import (
    Engine,
    QuestArtifacts,
    _format_lit,
    _is_citable,
    _preliminary_reminders,
)
from core.provider import LAST_CALL
from tests.test_evidence import ON, _quest, _state
from tests.test_knowledge import _BrainAddText, _calls_by_kind, _enabled_knowledge_with


# --- F2 -------------------------------------------------------------------------------------------------------------

def _writeback_engine(tmp_path: Path, captured: list, *, result_use: str) -> tuple:
    paper = tmp_path / "paper.md"
    paper.write_text("# Probe\n", encoding="utf-8")

    class _Knowledge:
        enabled = True

        def add_quest_artifacts(self, **kwargs):
            captured.append(kwargs)
            return True

    eng = object.__new__(Engine)
    eng.quest_id = "probe"
    eng.quest_root = tmp_path
    eng.knowledge = _Knowledge()
    eng.config = SimpleNamespace(
        knowledge=SimpleNamespace(write_back_quests=True, write_back_only_on_accept=True),
        provider=SimpleNamespace(name="fake", model="fake"), effective_result_use=result_use,
    )
    eng._log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    artifacts = QuestArtifacts(quest_id="probe", quest_root=tmp_path, paper_md=paper, paper_pdf=None,
                               figures_dir=None, bundle_manifest=None, raw_state={})
    return eng, artifacts


ACCEPT = {"review": {"verdict": "accept", "status": "ok", "must_flag_hits": []}, "analysis": {}, "design": {}}


@pytest.mark.parametrize("result_use,status,standing", [
    ("explore", "publication_ready", "preliminary"),   # an exploration is preliminary whatever it passed
    ("research", "protocol_runtime_matched", "preliminary"),  # a study with evidence gaps
    ("research", None, "preliminary"),                 # no evidence record at all
    ("research", "publication_ready", "accepted"),
    ("decision", "publication_ready", "accepted"),
])
def test_what_a_result_is_kept_as(tmp_path: Path, result_use: str, status: str | None, standing: str) -> None:
    captured: list = []
    eng, artifacts = _writeback_engine(tmp_path, captured, result_use=result_use)
    # A stale file from an earlier run is not read: only this run's evidence record counts.
    (tmp_path / "needs").mkdir(exist_ok=True)
    (tmp_path / "needs" / "EVIDENCE.json").write_text(json.dumps({"status": "publication_ready"}), encoding="utf-8")
    eng._write_back_knowledge(artifacts, ACCEPT, status)
    (call,) = captured
    meta = call["metadata"]
    assert meta["standing"] == standing
    assert meta["result_use"] == result_use and meta["evidence_status"] == (status or "unknown")


def test_a_preliminary_result_is_written_under_its_own_kinds_without_ref_spines(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# body", encoding="utf-8")
    refs = [{"title": "A real paper", "doi": "10.1/x", "authors": ["A"], "year": 2020}]
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    assert k.add_quest_artifacts(quest_id="q1", paper_md_path=paper, summary="found",
                                 metadata={"standing": "preliminary", "external_refs": refs, "topic": "t"})
    kinds = set(_calls_by_kind(brain))
    assert kinds == {"fi_preliminary_spine", "fi_preliminary_paper", "fi_preliminary_summary", "fi_topic_event"}
    topic_text, topic_meta = _calls_by_kind(brain)["fi_topic_event"]
    assert topic_meta["standing"] == "preliminary" and topic_text.startswith("PRELIMINARY")
    # An accepted result keeps the old kinds and its ref spines.
    brain2 = _BrainAddText()
    assert _enabled_knowledge_with(brain2).add_quest_artifacts(
        quest_id="q2", paper_md_path=paper, summary="found", metadata={"standing": "accepted", "external_refs": refs})
    assert "fi_quest_summary" in _calls_by_kind(brain2) and "fi_external_ref_spine" in _calls_by_kind(brain2)


def test_fi_memory_is_not_written_back_as_an_external_paper(tmp_path: Path) -> None:
    captured: list = []
    eng, artifacts = _writeback_engine(tmp_path, captured, result_use="research")
    state = {**ACCEPT, "literature": [
        {"metadata": {"kind": "fi_preliminary_spine", "title": "An earlier exploration"}, "content": "x"},
        {"metadata": {"kind": "fi_paper_spine", "title": "An earlier quest"}, "content": "x"},
        {"metadata": {"title": "A real paper", "doi": "10.1/x"}, "content": "abstract"},
        {"metadata": {"kind": "fi_local_paper", "title": "The person's own paper"}, "content": "x"},
        {"metadata": {"kind": "fi_external_ref_spine", "title": "A paper an earlier quest recorded"}, "content": "x"},
    ]}
    eng._write_back_knowledge(artifacts, state, "publication_ready")
    refs = captured[0]["metadata"]["external_refs"]
    assert [r["title"] for r in refs] == ["A real paper", "The person's own paper", "A paper an earlier quest recorded"]


def test_a_preliminary_result_is_not_counted_as_a_source_or_written_as_data(tmp_path: Path) -> None:
    from core.engine import _FI_INTERNAL_KINDS, PRELIMINARY_KINDS
    assert PRELIMINARY_KINDS <= _FI_INTERNAL_KINDS


def test_a_later_quest_reads_a_preliminary_result_as_a_reminder_never_a_source() -> None:
    prelim = SimpleNamespace(content="R0 above 2 gave outbreaks in 80% of runs.",
                             metadata={"kind": "fi_preliminary_summary", "title": "SIR scan", "quest_id": "q9",
                                       "authors": ["FI"], "year": 2026, "result_use": "explore",
                                       "evidence_status": "executed"})
    real = SimpleNamespace(content="A classic.", metadata={"title": "Kermack 1927", "authors": ["K"], "year": 1927})
    assert not _is_citable(prelim.metadata) and _is_citable(real.metadata)
    block = _format_lit([prelim, real], audience="internal")
    assert "[1]" in block and "Kermack" in block
    head, _, tail = block.partition("PRELIMINARY")
    assert "SIR scan" not in head and "SIR scan" in tail and "Never cite" in tail
    assert _preliminary_reminders([real]) == ""
    # Only preliminary memory: still no citable source, and the reminder is there.
    only = _format_lit([prelim])
    assert only.startswith("(no prior work") and "SIR scan" in only


# --- F3 -------------------------------------------------------------------------------------------------------------

RESEARCH = {**ON, "rigor_profile": "research"}


def _seal(log: audit_log.AuditLog, trace: Path, **over) -> None:
    events = audit_log.read(trace)
    log.append("quest_finalized", **{"events_before": len(events), "write_errors": 0,
                                    "nodes_completed": ["write", "review"], **over})


def _gaps(root: Path) -> list[str]:
    return [g for g in evidence.assess(root, _state(), settings=RESEARCH)["gaps"] if "trace" in g]


def test_a_valid_prefix_is_not_a_complete_trace(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    trace = root / ".fi" / "audit.jsonl"
    trace.unlink(missing_ok=True)
    log = audit_log.AuditLog(trace, root.name)
    log.append("quest_started")
    assert any("no final seal" in g for g in _gaps(root)), "one quest_started line used to pass"
    log.append("node_completed", node="write")
    log.append("node_completed", node="review")
    _seal(log, trace)
    assert _gaps(root) == []


@pytest.mark.parametrize("problem,expect", [
    ({"write_errors": 2}, "could not be written"),
    ({"events_before": 1}, "does not count"),
    ({"nodes_completed": ["write"]}, "no completed review"),
])
def test_a_seal_that_does_not_hold_up_is_a_gap(tmp_path: Path, problem: dict, expect: str) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    trace = root / ".fi" / "audit.jsonl"
    trace.unlink(missing_ok=True)
    log = audit_log.AuditLog(trace, root.name)
    log.append("quest_started")
    log.append("node_completed", node="write")
    log.append("node_completed", node="review")
    _seal(log, trace, **problem)
    assert any(expect in g for g in _gaps(root))


def test_steps_after_the_seal_are_a_gap(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    trace = root / ".fi" / "audit.jsonl"
    trace.unlink(missing_ok=True)
    log = audit_log.AuditLog(trace, root.name)
    log.append("node_completed", node="write")
    log.append("node_completed", node="review")
    _seal(log, trace)
    log.append("node_started", node="write")
    assert any("after the decision trace's final seal" in g for g in _gaps(root))
    # Outside research nothing changes.
    assert not any("seal" in g for g in evidence.assess(root, _state(), settings=ON)["gaps"])


def test_a_paused_quest_is_not_called_damaged(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    trace = root / ".fi" / "audit.jsonl"
    trace.unlink(missing_ok=True)
    audit_log.AuditLog(trace, root.name).append("quest_started")
    (root / ".fi" / "pause.json").write_text("{}", encoding="utf-8")
    assert any("has not finished yet" in g for g in _gaps(root))


def test_events_lost_before_a_pause_are_still_counted_by_the_seal(tmp_path: Path) -> None:
    trace = tmp_path / ".fi" / "audit.jsonl"
    trace.parent.mkdir(parents=True)
    audit_log._count_lost(trace)
    audit_log._count_lost(trace)
    assert audit_log.lost_writes(trace) == 2, "a later run of the quest reads the count left beside the trace"


def test_the_seal_describes_itself() -> None:
    line = audit_log.describe({"kind": "quest_finalized", "events_before": 42, "write_errors": 1})
    assert "42 events" in line and "1 could not be written" in line


# --- F4 -------------------------------------------------------------------------------------------------------------

def test_calls_at_the_same_time_keep_their_own_record_of_who_answered() -> None:
    async def call(model: str, delay: float) -> dict:
        LAST_CALL.set(None)
        LAST_CALL.set({"provider": "p", "model": model})
        await asyncio.sleep(delay)
        return LAST_CALL.get() or {}

    async def main() -> list[dict]:
        return await asyncio.gather(call("a", 0.05), call("b", 0.0))

    first, second = asyncio.run(main())
    assert first["model"] == "a" and second["model"] == "b"


def _research_engine(tmp_path: Path) -> Engine:
    from core.config import Config
    return Engine(Config.model_validate({
        "topic": "t", "rigor_profile": "research",
        "provider": {"name": "openai", "model": "m-main", "node_models": {"review_panel.statistician": "m-other"}},
        "knowledge": {"enabled": False}, "output": {"output_dir": str(tmp_path / "out")},
    }))


def test_reviewers_answered_by_one_model_are_found(tmp_path: Path) -> None:
    eng = _research_engine(tmp_path)
    ok = {"status": "ok", "actual_provider": "openai", "actual_model_reported": True}
    collapsed = [{**ok, "requested_model": "m-main", "actual_model": "same"},
                 {**ok, "requested_model": "m-other", "actual_model": "same"}]
    assert eng._review_models_collapsed(collapsed) == "openai/same"
    diverse = [{**ok, "requested_model": "m-main", "actual_model": "m-main"},
               {**ok, "requested_model": "m-other", "actual_model": "m-other"}]
    assert eng._review_models_collapsed(diverse) is None
    # A panel set up on one model is the pre-run check's business, not this one's.
    one = [{**ok, "requested_model": "m", "actual_model": "m"}, {**ok, "requested_model": "m", "actual_model": "m"}]
    assert eng._review_models_collapsed(one) is None
    # A transport that does not say which model served the call is unknown, not one model.
    unreported = [{**r, "actual_model_reported": False} for r in collapsed]
    assert eng._review_models_collapsed(unreported) is None


def test_one_model_review_goes_on_and_is_a_gap(tmp_path: Path) -> None:
    from core.config import Config
    eng = Engine(Config.model_validate({
        "topic": "t", "rigor_profile": "research", "provider": {"name": "openai", "model": "m"},
        "engine": {"one_model_review": True}, "knowledge": {"enabled": False},
        "output": {"output_dir": str(tmp_path / "out")},
    }))
    assert eng._review_models_stop() is None
    ok = {"status": "ok", "actual_provider": "openai", "actual_model_reported": True, "actual_model": "same"}
    assert eng._review_models_collapsed([{**ok, "requested_model": "a"}, {**ok, "requested_model": "b"}]) is None
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    gaps = evidence.assess(root, _state(), settings={**RESEARCH, "one_model_review": True})["gaps"]
    assert any("every reviewer used one model" in g for g in gaps)
    assert eng._one_model_panel() is True
    two = Engine(Config.model_validate({
        "topic": "t", "rigor_profile": "research",
        "provider": {"name": "openai", "model": "m", "node_models": {"review_panel.statistician": "m2"}},
        "engine": {"one_model_review": True}, "knowledge": {"enabled": False},
        "output": {"output_dir": str(tmp_path / "out2")},
    }))
    assert two._one_model_panel() is False, "the flag with a second model added is not a one-model review"


