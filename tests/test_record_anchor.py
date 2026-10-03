"""FI's note of how far it wrote the quest's record files (core/record_anchor.py): a line another program adds to the
trace or a record file (a simulation script runs as the same user) is told apart when the quest starts again, moved aside
and never used; a quest that only stopped (a pause, a crash, a torn last line) starts again with no alarm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import attempt_records as ar
from core import audit_log as al
from core import record_anchor as ra
from core.engine import Engine
from tests.test_frozen_protocol import _cfg, _fake


def _forge_trace_line(trace: Path, kind: str = "result_accepted", **fields: Any) -> dict[str, Any]:
    """What a script could write: a line with the next ``seq``, chained onto the last one, so the chain verifies."""
    last = al.read(trace)[-1]
    record = {"schema": last["schema"], "seq": last["seq"] + 1, "ts": last["ts"], "quest_id": last["quest_id"],
              "kind": kind, "provenance": al.DETERMINISTIC, "prev": last["hash"], **fields}
    record["hash"] = al.hash_of(last["hash"], record)
    with trace.open("ab") as fh:
        fh.write((al.canonical(record) + "\n").encode("utf-8"))
    return record


def _stopped_mid_line(path: Path, full: bytes, written: int) -> None:
    """What a kill while FI writes a line leaves: the note says the line is on its way, part of it is in the file."""
    con = ra._connect(path.parent)
    try:
        row = ra._rows(con).get(path.name) or ra.Row(ra.Head(), ra.Head(), False)
        body = full.rstrip(b"\n")
        nxt = ra.Head(row.now.size + len(body) + 1, row.now.lines + 1, ra._step(row.now.digest, body))
        ra._put(con, path.name, nxt, row.now, True)
    finally:
        con.close()
    with path.open("ab") as fh:
        fh.write(full[:written])


def _trace(tmp_path: Path, n: int = 5) -> al.AuditLog:
    log = al.AuditLog(tmp_path / ".fi" / "audit.jsonl", "q1")
    for i in range(n):
        log.append("check_result", check=f"c{i}", status="ok")
    return log


# --- the note and the check -----------------------------------------------------------------------------------------


def test_a_line_chained_onto_the_trace_verifies_but_is_told_apart_and_moved_aside(tmp_path: Path) -> None:
    log = _trace(tmp_path)
    fi = log.path.parent
    kept = ra.snapshot(fi)
    _forge_trace_line(log.path, check="oracle", status="pass")
    assert al.verify(log.path).ok, "the forged line chains: the hash chain alone cannot tell it apart"

    seen = ra.check(fi)  # what the evidence does: nothing changed
    assert len(seen) == 1 and seen[0].file == "audit.jsonl" and not seen[0].moved_to
    assert len(al.read(log.path)) == 6

    moved = ra.check(fi, kept, repair=True)
    assert len(moved) == 1 and moved[0].lines == 1 and moved[0].moved_to == "audit.outside_fi.jsonl"
    assert "changed outside FI" in moved[0].line() and "not used" in moved[0].line()
    assert len(al.read(log.path)) == 5 and al.verify(log.path).ok
    assert json.loads((fi / "audit.outside_fi.jsonl").read_text(encoding="utf-8"))["kind"] == "result_accepted"
    assert ra.check(fi) == [], "the note is settled to the file as it now is"

    # FI goes on from its own last line, and a second move never overwrites the first.
    again = al.AuditLog(log.path, "q1")
    again.append("quest_started", resumed=True)
    assert al.verify(log.path).ok and ra.check(fi) == []
    _forge_trace_line(log.path)
    assert ra.check(fi, repair=True)[0].moved_to == "audit.outside_fi.2.jsonl"


def test_a_line_added_to_a_record_file_is_found_even_while_fi_still_runs(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": "ok"})
    ar.append(fi, ar.MODEL_CALLS, {"node": "review", "outcome": "ok"})
    with (fi / ar.MODEL_CALLS).open("ab") as fh:
        fh.write(b'{"node": "review", "answered_by": "another-model", "outcome": "ok"}\n')
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": "ok"})  # FI's later lines do not hide it
    gaps = ra.evidence_gaps(fi)
    assert len(gaps) == 1 and "model_calls.jsonl" in gaps[0] and "changed outside FI" in gaps[0]


def test_an_edited_earlier_line_is_found_and_left_in_place(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    for i in range(3):
        ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": "crashed", "n": i})
    path = fi / ar.ATTEMPTS
    path.write_bytes(path.read_bytes().replace(b'"crashed"', b'"ok"', 1))
    found = ra.check(fi, repair=True)
    assert len(found) == 1 and not found[0].moved_to and "changed" in found[0].reason
    assert path.read_bytes().count(b"\n") == 3


def test_a_deleted_note_is_caught_by_the_copy_in_the_checkpoint(tmp_path: Path) -> None:
    log = _trace(tmp_path)
    fi = log.path.parent
    kept = ra.snapshot(fi)
    _forge_trace_line(log.path)
    (fi / ra.HEADS).unlink()
    found = ra.check(fi, kept, repair=True)
    assert len(found) == 1 and found[0].lines == 1 and found[0].moved_to, found
    assert len(al.read(log.path)) == 5
    assert any(ra.HEADS in g for g in ra.evidence_gaps(fi)), "the removed note is itself recorded"


def test_a_rewound_note_disagrees_with_the_checkpoint(tmp_path: Path) -> None:
    log = _trace(tmp_path, 2)
    fi = log.path.parent
    import shutil

    shutil.copy(fi / ra.HEADS, tmp_path / "old.sqlite")
    log.append("check_result", check="c9", status="ok")
    kept = ra.snapshot(fi)
    shutil.copy(tmp_path / "old.sqlite", fi / ra.HEADS)  # the note put back to an earlier point
    found = ra.check(fi, kept, repair=True)
    assert len(found) == 1 and "checkpoint" in found[0].reason


def test_a_plain_stop_raises_no_alarm_and_a_torn_last_line_is_still_repaired(tmp_path: Path) -> None:
    log = _trace(tmp_path)
    fi = log.path.parent
    kept = ra.snapshot(fi)
    _stopped_mid_line(log.path, b'{"schema":1,"seq":6,"kind":"node_started"}\n', 20)  # FI stopped half-way
    assert ra.check(fi, kept, repair=True) == []
    reopened = al.AuditLog(log.path, "q1")
    reopened.append("quest_started", resumed=True)
    kinds = [e["kind"] for e in al.read(log.path)]
    assert kinds[-2:] == ["audit_repair", "quest_started"] and al.verify(log.path).ok
    assert ra.check(fi) == [] and ra.moved_files(fi) == []


def test_a_line_fi_noted_but_never_wrote_is_no_alarm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = _trace(tmp_path, 2)
    fi = log.path.parent
    real_open = Path.open

    def failing_open(self: Path, mode: str = "r", *a: Any, **kw: Any):  # noqa: ANN202
        if self.name == "audit.jsonl" and "a" in mode:
            raise OSError("disk full")
        return real_open(self, mode, *a, **kw)

    monkeypatch.setattr(Path, "open", failing_open)
    assert log.append("check_result", check="lost", status="ok") is None
    monkeypatch.setattr(Path, "open", real_open)
    log.append("check_result", check="after", status="ok")
    assert ra.check(fi) == [] and al.verify(log.path).ok


def test_a_line_fi_could_not_note_is_still_fis_and_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    log = _trace(tmp_path, 2)
    fi = log.path.parent
    real = ra._open_note

    def locked(*a: Any, **kw: Any):  # noqa: ANN202
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ra, "_open_note", locked)
    log.append("check_result", check="unnoted", status="ok")
    monkeypatch.setattr(ra, "_open_note", real)
    log.append("check_result", check="after", status="ok")
    assert al.verify(log.path).ok and len(al.read(log.path)) == 4 and ra.moved_files(fi) == []
    assert ra.lost(fi) == 1 and any("could not keep its note" in g for g in ra.evidence_gaps(fi))


def test_a_removed_record_file_is_recorded(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "n": 0})
    (fi / ar.ATTEMPTS).unlink()
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "n": 1})
    assert ra.lost(fi) == 0 and [r["n"] for r in ar.read(fi, ar.ATTEMPTS)] == [1]
    assert any("removed" in g for g in ra.evidence_gaps(fi))


def test_a_quest_from_before_the_note_with_a_torn_line_raises_no_alarm(tmp_path: Path) -> None:
    log = _trace(tmp_path, 2)
    fi = log.path.parent
    (fi / ra.HEADS).unlink()  # as if written before FI kept the note (and no checkpoint names one)
    with log.path.open("ab") as fh:
        fh.write(b'{"schema":"fi.audit/v1","seq":3,"ki')
    (fi / ar.ATTEMPTS).write_bytes(b'{"kind": "run"}\n{"kind": "ru')
    assert ra.check_on_start(fi) == []
    again = al.AuditLog(log.path, "q1")
    again.append("quest_started", resumed=True)
    ar.append(fi, ar.ATTEMPTS, {"kind": "run"})
    assert [e["kind"] for e in al.read(log.path)][-2:] == ["audit_repair", "quest_started"]
    assert ra.moved_files(fi) == [] and ra.evidence_gaps(fi) == [] and len(ar.read(fi, ar.ATTEMPTS)) == 2


def test_files_outside_the_records_are_not_noted(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    ar.append(fi, "thinking.jsonl", {"x": 1})
    assert ra.snapshot(fi) == {}


def test_a_removed_last_line_of_fi_is_found(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    for outcome in ("ok", "ok", "oracle_failure"):
        ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": outcome})
    path = fi / ar.ATTEMPTS
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))  # the failure removed
    assert ra.check(fi) and ra.evidence_gaps(fi)
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": "ok"})  # FI going on does not hide it
    assert any("removed" in g for g in ra.evidence_gaps(fi)), ra.evidence_gaps(fi)


def test_a_forged_line_with_no_newline_is_not_taken_for_a_torn_one(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": "crashed"})
    with (fi / ar.ATTEMPTS).open("ab") as fh:
        fh.write(b'{"kind": "run", "outcome": "ok", "record_id": "forged"}')
    assert ra.check(fi) and ra.evidence_gaps(fi)
    moved = ra.check(fi, repair=True)
    assert moved and moved[0].moved_to and b"forged" not in (fi / ar.ATTEMPTS).read_bytes()


def test_a_part_written_record_line_of_fi_is_cut_off_with_no_alarm(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "n": 0})
    _stopped_mid_line(fi / ar.ATTEMPTS, b'{"kind": "run", "n": 1}\n', 10)
    assert ra.check(fi) == [] and ra.check_on_start(fi) == []
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "n": 2})
    assert [r["n"] for r in ar.read(fi, ar.ATTEMPTS)] == [0, 2] and ra.evidence_gaps(fi) == []


def test_a_line_added_while_fi_runs_is_moved_aside_before_fi_writes_on(tmp_path: Path) -> None:
    log = _trace(tmp_path, 3)
    fi = log.path.parent
    _forge_trace_line(log.path, check="oracle", status="pass")
    log.append("check_result", check="c9", status="ok")  # FI's next line: the forged one is set aside first
    assert al.verify(log.path).ok and len(al.read(log.path)) == 4
    assert ra.moved_files(fi) == ["audit.outside_fi.jsonl"] and ra.check(fi) == []
    assert ra.evidence_gaps(fi), "what was found while the quest ran is kept for the evidence"
    ar.append(fi, ar.MODEL_CALLS, {"node": "review"})
    with (fi / ar.MODEL_CALLS).open("ab") as fh:
        fh.write(b'{"node": "review", "answered_by": "another-model"}\n')
    ar.append(fi, ar.MODEL_CALLS, {"node": "review"})
    assert not [r for r in ar.read(fi, ar.MODEL_CALLS) if r.get("answered_by")]
    assert [f.file for f in ra.check_on_start(fi)] == ["audit.jsonl", ar.MODEL_CALLS]


def test_another_process_opening_the_trace_does_not_chain_onto_an_added_line(tmp_path: Path) -> None:
    log = _trace(tmp_path, 3)
    _forge_trace_line(log.path)
    other = al.AuditLog(log.path, "q1")  # e.g. a title change from the web page
    other.append("title_changed", old="a", new="b")
    assert al.verify(log.path).ok and [e["kind"] for e in al.read(log.path)][-1] == "title_changed"
    assert len(al.read(log.path)) == 4


# --- the engine -------------------------------------------------------------------------------------------------------


def _events(engine: Engine) -> list[dict[str, Any]]:
    return al.read(engine.quest_root / ".fi" / "audit.jsonl")


async def _paused_quest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Engine]:
    """A quest stopped at the plan pause: FI's process gone after its last write, as after a kill."""
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake([], second_design={}))
    cfg = _cfg(tmp_path)
    cfg.pauses.plan = "ask"
    first = Engine(cfg)
    await first.run()
    assert [e for e in _events(first) if e["kind"] == "node_paused"]
    return cfg, first


@pytest.mark.asyncio
async def test_a_forged_line_written_before_fi_stopped_is_moved_aside_on_resume_and_keeps_the_result_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, first = await _paused_quest(tmp_path, monkeypatch)
    fi = first.quest_root / ".fi"
    count = len(_events(first))
    _forge_trace_line(fi / "audit.jsonl", check="oracle", status="pass", summary="all checks hold")
    with (fi / ar.ATTEMPTS).open("ab") as fh:
        fh.write(b'{"kind": "run", "outcome": "ok", "record_id": "forged"}\n')
    assert al.verify(fi / "audit.jsonl").ok

    second = Engine(cfg, resume_quest_id=first.quest_id)
    await second.run()
    events = _events(second)
    assert al.verify(second.audit.path).ok, al.verify(second.audit.path).line()
    assert not [e for e in events if e.get("kind") == "result_accepted" and e.get("summary") == "all checks hold"]
    found = [e for e in events if e["kind"] == ra.EVENT]
    assert {e["file"] for e in found} == {"audit.jsonl", ar.ATTEMPTS}
    assert events[count]["kind"] == ra.EVENT, "recorded before this run writes anything else"
    assert set(ra.moved_files(fi)) == {"audit.outside_fi.jsonl", "attempts.outside_fi.jsonl"}
    assert b"forged" not in (fi / ar.ATTEMPTS).read_bytes()
    assert "changed outside FI" in (fi / "run.log").read_text(encoding="utf-8")
    evidence = json.loads((second.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert "changed outside FI" in json.dumps(evidence)
    assert evidence.get("level") != "publication_ready"


@pytest.mark.asyncio
async def test_a_resume_with_nothing_added_raises_no_alarm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, first = await _paused_quest(tmp_path, monkeypatch)
    fi = first.quest_root / ".fi"
    # FI's checkpoint holds its note, from the end of the last step before the pause.
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(str(fi / "state.sqlite")) as saver:
        last = await saver.aget_tuple({"configurable": {"thread_id": first.quest_id}})
    kept = last.checkpoint["channel_values"]["record_anchor"]
    assert kept["audit.jsonl"]["lines"] >= 1
    _stopped_mid_line(fi / "audit.jsonl", b'{"schema":1,"seq":999,"kind":"node_started"}\n', 15)  # a torn last line
    _stopped_mid_line(fi / ar.ATTEMPTS, b'{"kind":"quest","outcome":"crashed"}\n', 9)  # and one in a record file

    second = Engine(cfg, resume_quest_id=first.quest_id)
    await second.run()
    events = _events(second)
    assert al.verify(second.audit.path).ok
    assert not [e for e in events if e["kind"] == ra.EVENT] and ra.moved_files(fi) == []
    assert [e for e in events if e["kind"] == "audit_repair"]
    assert "changed outside FI" not in json.dumps(
        json.loads((second.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8")))
