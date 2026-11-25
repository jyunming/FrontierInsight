"""The saved search record the seal names (kept from the topic-search tests): ``.fi/literature_queries.json`` stays
a sealed file, so an edit after the quest is closed is a gap, and the trace watches it step by step. The commissioned
literature step still writes to it (the retraction lookup)."""
from __future__ import annotations

import json
from pathlib import Path

from core import audit_log, evidence
from tests.test_evidence import _quest, _state as _evidence_state

RESEARCH = {"protocol_check": "block", "oracle_check": "block", "numeric_warnings": "block",
            "run_manifest_check": "block", "rigor_profile": "research"}


def _sealed(tmp_path: Path) -> Path:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    (root / ".fi").mkdir(exist_ok=True)
    (root / evidence.SEALED_QUERIES).write_text(
        json.dumps({"schema": 2, "entries": [{"stage": "literature", "queries": ["q"], "digest": "d"}]}),
        encoding="utf-8")
    trace = root / ".fi" / "audit.jsonl"
    trace.unlink(missing_ok=True)
    log = audit_log.AuditLog(trace, root.name)
    log.append("quest_started")
    log.append("node_completed", node="write")
    log.append("node_completed", node="review")
    record = evidence.assess(root, _evidence_state(), settings={**RESEARCH, "sealing": True})
    (root / "needs" / "EVIDENCE.json").write_text(json.dumps(record), encoding="utf-8")
    for rel in evidence.SEALED_LEDGERS:
        (root / rel).touch()
    log.append("quest_finalized", events_before=len(audit_log.read(trace)), write_errors=0, records_not_written=0,
               model_calls={"lines": 0, "counts": {}, "gaps": []}, rigor_profile="research",
               nodes_completed=["review", "write"], paper_path="paper/paper.md",
               paper_sha256=evidence._file_sha256(root / "paper" / "paper.md"),
               files={rel: evidence._file_sha256(root / rel) for rel in evidence.SEALED_FILES})
    return root


def test_the_seal_names_the_query_record_and_an_edit_after_it_is_a_gap(tmp_path: Path) -> None:
    assert evidence.SEALED_QUERIES in evidence.SEALED_FILES
    root = _sealed(tmp_path)
    assert evidence.read(root)["trace_seal"] == "verified"
    (root / evidence.SEALED_QUERIES).write_text(
        json.dumps({"schema": 2, "entries": [{"stage": "literature", "queries": ["other"], "digest": "d"}]}),
        encoding="utf-8")
    read = evidence.read(root)
    assert read["trace_seal"] == "not_verified"
    assert any("literature_queries.json changed after the quest was sealed" in g
               for g in read["all_gaps"]["publication_ready"])


def test_a_seal_that_names_no_query_record_is_a_gap(tmp_path: Path) -> None:
    root = _sealed(tmp_path)
    trace = root / ".fi" / "audit.jsonl"
    events = audit_log.read(trace)
    lines = trace.read_text(encoding="utf-8").splitlines()
    trace.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    seal = events[-1]
    audit_log.AuditLog(trace, root.name).append(
        "quest_finalized", **{k: v for k, v in seal.items() if k not in audit_log._RESERVED | {"files"}},
        files={rel: evidence._file_sha256(root / rel) for rel in evidence.SEALED_FILES
               if rel != evidence.SEALED_QUERIES})
    assert any("names no hash of .fi/literature_queries.json" in g
               for g in evidence.read(root)["all_gaps"]["publication_ready"])


def test_the_record_is_watched_step_by_step_in_the_trace() -> None:
    from core.engine import Engine

    assert ".fi/literature_queries.json" in Engine._AUDIT_WATCHED

