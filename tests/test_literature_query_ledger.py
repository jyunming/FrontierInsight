"""The saved search queries are a record the seal names (the 2026-09-28 R2 re-audit, P2-01).

Each entry of ``.fi/literature_queries.json`` carries a digest over what was searched, why, and which model call
produced it; FI only adds entries. The digest has no key: it catches an accidental or uninformed edit (never reused;
FI derives again and says so); FI's own entry is also reused only when it names an answered call in the record of model
calls. The trace hashes the file after each step, and under ``rigor_profile: research`` the seal names it, so an edit
after the quest is closed is a gap. A person's own queries go in ``inputs/search_queries.txt`` and are recorded as
their revision.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from core import audit_log, evidence
from core.engine import _query_set_digest, _query_set_standing
from tests.test_evidence import _quest, _state as _evidence_state
from tests.test_literature_facets import FACETS, _engine, _state
from tests.test_literature_query_once import _counting, _sets, _unscored  # noqa: F401 -- the fixture is autouse

RESEARCH = {"protocol_check": "block", "oracle_check": "block", "numeric_warnings": "block",
            "run_manifest_check": "block", "rigor_profile": "research"}


def _write(eng, entries: list[dict]) -> None:
    (eng.fi_dir / "literature_queries.json").write_text(json.dumps({"schema": 2, "entries": entries}),
                                                        encoding="utf-8")


@pytest.mark.asyncio
async def test_every_entry_carries_a_digest_that_checks_out(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _counting(eng)
    await eng._node_literature(_state())
    (entry,) = _sets(eng)
    assert entry["digest"] == _query_set_digest(entry) and _query_set_standing(entry) == "verified"
    assert entry["revised_by"] == "fi" and entry["call_id"] and "model" in entry
    for field, value in (("queries", ["another query"]), ("model", "other-model"), ("call_id", "x"),
                         ("reason", "made up")):
        assert _query_set_standing({**entry, field: value}) == "edited", field


@pytest.mark.asyncio
async def test_queries_edited_after_the_first_run_are_not_silently_reused(tmp_path: Path, caplog) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    await eng._node_literature(_state())
    (entry,) = _sets(eng)
    tampered = {**entry, "queries": ["an unrelated query", "another"]}
    _write(eng, [tampered])
    with caplog.at_level(logging.WARNING):
        again = await eng._node_literature(_state())
    assert calls.count("literature_query") == 2, "derived again, not reused"
    assert again["literature_queries"] == FACETS
    assert again["literature_query_set"]["reason"] == "the saved queries for this pass were changed after FI wrote them"
    run_log = (eng.fi_dir / "run.log").read_text(encoding="utf-8") if (eng.fi_dir / "run.log").exists() else ""
    assert "changed after FI wrote them" in run_log or any("changed after FI wrote them" in r.getMessage()
                                                            for r in caplog.records)
    kept, new = _sets(eng)
    assert kept == tampered, "entries are only ever added: the edited one stays as it is"
    assert _query_set_standing(new) == "verified"
    third = await eng._node_literature(_state())
    assert third["literature_query_set"]["reused"] is True and calls.count("literature_query") == 2


@pytest.mark.asyncio
async def test_a_person_s_own_queries_are_used_and_recorded_as_their_revision(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    await eng._node_literature(_state())
    mine = eng.quest_root / "inputs" / "search_queries.txt"
    mine.parent.mkdir(parents=True, exist_ok=True)
    mine.write_text("# my own search\nlattice boltzmann cavity\n\nmulti relaxation time\n", encoding="utf-8")
    patch = await eng._node_literature(_state())
    assert patch["literature_queries"] == ["lattice boltzmann cavity", "multi relaxation time"]
    assert calls.count("literature_query") == 1, "no derivation for a person's queries"
    person = _sets(eng)[-1]
    assert person["revised_by"] == "person" and _query_set_standing(person) == "verified"
    assert person["reason"] == "your queries in inputs/search_queries.txt" and person["source_sha256"]
    # The same file again: the recorded revision is used, nothing new is added.
    again = await eng._node_literature(_state())
    assert again["literature_query_set"]["reused"] is True and len(_sets(eng)) == 2
    # A changed file is a new revision.
    mine.write_text("lattice boltzmann benchmark\n", encoding="utf-8")
    await eng._node_literature(_state())
    assert len(_sets(eng)) == 3 and _sets(eng)[-1]["queries"] == ["lattice boltzmann benchmark"]


@pytest.mark.asyncio
async def test_entries_from_before_the_digest_are_unverified_and_derived_once_more(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    await eng._node_literature(_state())
    (entry,) = _sets(eng)
    old = {k: v for k, v in entry.items() if k not in ("digest", "revised_by", "call_id")}
    _write(eng, [old])
    patch = await eng._node_literature(_state())
    assert calls.count("literature_query") == 2 and patch["literature_query_set"]["reason"].startswith(
        "the saved queries for this pass carry no digest")
    assert _query_set_standing(_sets(eng)[-1]) == "verified"
    await eng._node_literature(_state())
    assert calls.count("literature_query") == 2, "then the new one, with its digest, is reused"


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


def _call_ids(eng, node: str) -> set[str]:
    from core import attempt_records as ar

    return {r["call_id"] for r in ar.read(eng.fi_dir, ar.MODEL_CALLS) if r.get("node") == node and r.get("outcome") == "ok"}


@pytest.mark.asyncio
async def test_an_entry_names_its_real_line_in_the_record_of_model_calls(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _counting(eng)
    await eng._node_literature(_state())
    (entry,) = _sets(eng)
    assert entry["call_id"] and entry["call_id"] in _call_ids(eng, "literature_query")


@pytest.mark.asyncio
async def test_a_forged_entry_whose_digest_was_recomputed_is_not_reused(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    await eng._node_literature(_state())
    (entry,) = _sets(eng)
    forged = {k: v for k, v in entry.items() if k != "digest"}
    forged.update(queries=["planted query"], call_id="not-a-real-call")
    forged["digest"] = _query_set_digest(forged)
    assert _query_set_standing(forged) == "verified", "the digest alone cannot tell (it has no key)"
    _write(eng, [entry, forged])
    patch = await eng._node_literature(_state())
    # The newer forgery names no answered call: passed over; the genuine entry is the one reused.
    assert patch["literature_queries"] == FACETS and calls.count("literature_query") == 1
    _write(eng, [forged])  # only the forgery left: derived again rather than trusted
    patch = await eng._node_literature(_state())
    assert "planted query" not in patch["literature_queries"] and calls.count("literature_query") == 2
    assert "do not match an answered call in the record of model calls" in patch["literature_query_set"]["reason"]


@pytest.mark.asyncio
async def test_removing_your_own_queries_goes_back_to_fi_s(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    mine = eng.quest_root / "inputs" / "search_queries.txt"
    mine.parent.mkdir(parents=True, exist_ok=True)
    mine.write_text("my own query\n", encoding="utf-8")
    first = await eng._node_literature(_state())
    assert first["literature_queries"] == ["my own query"] and calls.count("literature_query") == 0
    events = audit_log.read(eng.audit.path)
    assert any(e.get("kind") == "check_result" and e.get("check") == "literature_queries"
               and e.get("status") == "person" and e.get("source_sha256") for e in events)
    mine.unlink()
    second = await eng._node_literature(_state())
    assert second["literature_queries"] == FACETS and calls.count("literature_query") == 1
    assert second["literature_query_set"]["revised_by"] == "fi"


def test_the_record_is_watched_step_by_step_in_the_trace() -> None:
    from core.engine import Engine

    assert ".fi/literature_queries.json" in Engine._AUDIT_WATCHED


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [RuntimeError("provider down"), "not json"])
async def test_a_failed_derivation_leaves_a_record_the_seal_can_name(tmp_path: Path, answer) -> None:
    eng = _engine(tmp_path)
    _counting(eng, answer=answer)
    await eng._node_literature(_state())
    (entry,) = _sets(eng)
    assert entry["queries"] == [] and "gave no query" in entry["reason"]
    assert ("(the call failed)" in entry["reason"]) == isinstance(answer, BaseException)
    # The entry points at the line of the record that call left, a failed one too.
    from core import attempt_records as ar

    rows = {r["call_id"]: r for r in ar.read(eng.fi_dir, ar.MODEL_CALLS)}
    assert entry["call_id"] in rows and rows[entry["call_id"]]["node"] == "literature_query"
    assert (rows[entry["call_id"]]["outcome"] == "ok") != isinstance(answer, BaseException)
    # What the seal would name exists (so a failed call is not a gap), and it names nothing to reuse.
    assert evidence._file_sha256(eng.quest_root / evidence.SEALED_QUERIES)


def _forge(entry: dict, **over) -> dict:
    forged = {k: v for k, v in entry.items() if k != "digest"}
    forged.update(over)
    forged["digest"] = _query_set_digest(forged)
    return forged


@pytest.mark.asyncio
async def test_a_copied_call_id_with_other_queries_is_not_reused(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    await eng._node_literature(_state())
    (entry,) = _sets(eng)
    # One call, two entries: neither is trusted (a call id backs one entry only).
    _write(eng, [entry, _forge(entry, queries=["planted query"])])
    patch = await eng._node_literature(_state())
    assert "planted query" not in patch["literature_queries"] and calls.count("literature_query") == 2
    assert "another entry also names" in patch["literature_query_set"]["reason"]


@pytest.mark.asyncio
async def test_an_entry_whose_prompt_is_not_the_calls_is_not_reused(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    await eng._node_literature(_state())
    (entry,) = _sets(eng)
    _write(eng, [_forge(entry, prompt_sha256="0" * 64)])
    patch = await eng._node_literature(_state())
    assert calls.count("literature_query") == 2 and patch["literature_queries"] == FACETS


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [RuntimeError("provider down"), "not json", None])
async def test_the_ideate_step_records_its_query_entry(tmp_path: Path, answer) -> None:
    eng = _engine(tmp_path)
    eng.knowledge.enabled = True
    _counting(eng, answer=answer)
    try:
        await eng._node_ideate({"topic": "t"})
    except Exception:  # noqa: BLE001 -- the ideas call after the seed search is not under test
        pass
    (entry,) = [e for e in _sets(eng) if e["stage"] == "ideate"]
    assert entry["revised_by"] == "fi" and _query_set_standing(entry) == "verified"
    from core import attempt_records as ar

    rows = {r["call_id"]: r for r in ar.read(eng.fi_dir, ar.MODEL_CALLS)}
    assert entry["call_id"] in rows and rows[entry["call_id"]]["node"] == "ideate_query"
    if answer is None:
        assert entry["queries"] == FACETS[:3] and rows[entry["call_id"]]["outcome"] == "ok"
    else:
        assert entry["queries"] == [] and "gave no query" in entry["reason"]
        assert ("(the call failed)" in entry["reason"]) == isinstance(answer, BaseException)
