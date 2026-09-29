"""Property tests for the audit trace's hash chain and the quest seal.

The example tests pin the cases someone thought of; these generate the trace and the edits, so a tamper the authors did
not think of still has to be caught. Each test is kept under about a second (few examples, no deadline, derandomized so a
CI failure reproduces) so they run in the fast tier; ``pytest --durations`` shows the actual time per test.

``hypothesis`` is a dev dependency: if it is missing these tests fail on import, they do not skip.

Not covered here: ``cost.jsonl``, ``trial_runner`` and ``web/server.py`` read their files with ``str.splitlines()``. That is
safe today because they write with ``ensure_ascii=True`` (U+0085/U+2028/U+2029 are escaped), so a record never holds a raw
line separator; do not switch those writers to ``ensure_ascii=False`` without changing the readers.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from core import attempt_records, audit_log, evidence

FAST = settings(max_examples=15, deadline=None, derandomize=True,
                suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])

# Any Unicode (CJK, emoji, control characters) plus the three characters str.splitlines() would split on and the
# readers must not. The tests cut the file with split("\n"), like the readers do.
_text = st.text(alphabet=st.one_of(st.characters(blacklist_categories=("Cs",)),
                                   st.sampled_from(["\x85", "\u2028", "\u2029", "\x00", "\x1f", "\r"])), max_size=20)
# Each seal example writes a dozen files, so it gets fewer examples to stay under a second.
SEAL = settings(FAST, max_examples=6)
_fields = st.dictionaries(st.sampled_from(["a", "b", "c", "note", "n"]), st.one_of(_text, st.integers(-5, 5), st.booleans()),
                          max_size=3)
_events = st.lists(st.tuples(st.sampled_from(["decision", "node_completed", "note"]), _fields), min_size=2, max_size=8)


class _Dir:
    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="fi_prop_"))
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _chain(root: Path, events) -> Path:
    trace = root / "audit.jsonl"
    log = audit_log.AuditLog(trace, "q")
    for kind, fields in events:
        log.append(kind, **fields)
    return trace


def _lines(trace: Path) -> list[str]:
    lines = trace.read_text(encoding="utf-8").split("\n")
    assert lines[-1] == "", "a trace ends with a newline"
    return lines[:-1]


def _write_lines(trace: Path, lines: list[str]) -> None:
    trace.write_text("\n".join(lines) + "\n", encoding="utf-8")


@FAST
@given(_events)
def test_a_chain_written_by_the_log_verifies(events) -> None:
    with _Dir() as root:
        v = audit_log.verify(_chain(root, events))
        assert v.ok and v.events == len(events)


@pytest.mark.parametrize("char", ["\x85", "\u2028", "\u2029"])
def test_a_line_separator_character_in_an_event_does_not_break_the_chain(char) -> None:
    with _Dir() as root:
        trace = _chain(root, [("note", {"a": "x" + char + "y"}), ("note", {})])
        v = audit_log.verify(trace)
        assert v.ok and v.events == 2 and len(audit_log.read(trace)) == 2


@pytest.mark.parametrize("char", ["\x85", "\u2028", "\u2029"])
def test_a_trace_holding_a_line_separator_character_resumes_and_keeps_verifying(char) -> None:
    with _Dir() as root:
        trace = _chain(root, [("note", {"a": "x" + char + "y"}), ("note", {})])
        log = audit_log.AuditLog(trace, "q")
        log.append("note", b="after")
        log.append("note", b=char)
        v = audit_log.verify(trace)
        assert v.ok and v.events == 4
        assert [e["seq"] for e in audit_log.read(trace)] == [1, 2, 3, 4]


@FAST
@given(_events, _events)
def test_appending_never_breaks_what_was_already_verified(first, more) -> None:
    with _Dir() as root:
        trace = _chain(root, first)
        before = _lines(trace)
        log = audit_log.AuditLog(trace, "q")
        for kind, fields in more:
            log.append(kind, **fields)
        assert _lines(trace)[: len(before)] == before, "an append rewrites nothing that was already there"
        assert audit_log.verify(trace).ok


@FAST
@given(_events, st.data())
def test_changing_any_value_in_any_event_is_detected(events, data) -> None:
    with _Dir() as root:
        trace = _chain(root, events)
        lines = _lines(trace)
        i = data.draw(st.integers(0, len(lines) - 1))
        ev = json.loads(lines[i])
        key = data.draw(st.sampled_from([k for k in ev if k not in ("prev", "hash")]))
        new = data.draw(st.one_of(_text, st.integers(), st.none(), st.booleans()).filter(lambda x: x != ev[key]))
        ev[key] = new
        lines[i] = json.dumps(ev, ensure_ascii=False)
        _write_lines(trace, lines)
        v = audit_log.verify(trace)
        assert not v.ok and v.bad_seq is not None


@FAST
@given(_events, st.data())
def test_deleting_any_line_but_the_last_is_detected(events, data) -> None:
    with _Dir() as root:
        trace = _chain(root, events)
        lines = _lines(trace)
        i = data.draw(st.integers(0, len(lines) - 2))
        del lines[i]
        _write_lines(trace, lines)
        assert not audit_log.verify(trace).ok


@FAST
@given(_events, st.data())
def test_swapping_two_different_lines_is_detected(events, data) -> None:
    with _Dir() as root:
        trace = _chain(root, events)
        lines = _lines(trace)
        i = data.draw(st.integers(0, len(lines) - 2))
        j = data.draw(st.integers(i + 1, len(lines) - 1))
        lines[i], lines[j] = lines[j], lines[i]
        _write_lines(trace, lines)
        assert not audit_log.verify(trace).ok


@FAST
@given(_events, st.data())
def test_a_line_copied_into_the_middle_is_detected(events, data) -> None:
    with _Dir() as root:
        trace = _chain(root, events)
        lines = _lines(trace)
        i = data.draw(st.integers(0, len(lines) - 1))
        lines.insert(i, lines[data.draw(st.integers(0, len(lines) - 1))])
        _write_lines(trace, lines)
        assert not audit_log.verify(trace).ok


@FAST
@given(_events, st.data())
def test_deleting_a_line_and_renumbering_the_later_ones_is_detected(events, data) -> None:
    with _Dir() as root:
        trace = _chain(root, events)
        lines = _lines(trace)
        i = data.draw(st.integers(0, len(lines) - 2))
        del lines[i]
        for k in range(i, len(lines)):
            ev = json.loads(lines[k])
            ev["seq"] = k + 1
            lines[k] = json.dumps(ev, ensure_ascii=False)
        _write_lines(trace, lines)
        assert not audit_log.verify(trace).ok


@FAST
@given(_events, st.data())
def test_editing_an_event_and_recomputing_only_its_own_hash_is_detected(events, data) -> None:
    with _Dir() as root:
        trace = _chain(root, events)
        lines = _lines(trace)
        i = data.draw(st.integers(0, len(lines) - 2))
        ev = json.loads(lines[i])
        ev["note"] = data.draw(_text.filter(lambda x: x != ev.get("note")))
        ev.pop("hash")
        ev["hash"] = audit_log.hash_of(ev["prev"], ev)
        lines[i] = json.dumps(ev, ensure_ascii=False)
        _write_lines(trace, lines)
        assert not audit_log.verify(trace).ok, "the next event's prev no longer matches"


@FAST
@given(_text)
def test_attempt_records_read_keeps_a_record_holding_a_line_separator(text) -> None:
    with _Dir() as root:
        notes = ["x\x85y", "x\u2028y", "x\u2029y", text]
        for name in (attempt_records.ATTEMPTS, attempt_records.MODEL_CALLS):
            for note in notes:
                assert attempt_records.append(root, name, {"kind": "k", "note": note}) is not None
            assert [r["note"] for r in attempt_records.read(root, name)] == notes


# ---- the seal -------------------------------------------------------------------------------------------------------

_RECORD = {"rigor_profile": "research", "status": "publication_ready",
           "levels": {lv: True for lv in evidence.LEVELS}, "all_gaps": {},
           "ladder": [{"level": lv, "reached": True} for lv in evidence.LEVELS]}


def _sealed(root: Path, files: tuple[str, ...] | None = None, *, sealed_records: int | None = evidence.SEAL_RECORDS,
            content: str = "x") -> Path:
    (root / ".fi").mkdir(exist_ok=True)
    (root / "needs").mkdir(exist_ok=True)
    (root / "paper").mkdir(exist_ok=True)
    (root / "paper" / "paper.md").write_text("# paper " + content, encoding="utf-8")
    named = files if files is not None else evidence.SEALED_FILES
    for rel in evidence.SEALED_FILES:
        (root / rel).write_text(content if rel.endswith(".json") is False else "{}", encoding="utf-8")
    trace = root / ".fi" / "audit.jsonl"
    log = audit_log.AuditLog(trace, "q")
    log.append("quest_started")
    log.append("node_completed", node="write")
    log.append("node_completed", node="review")
    extra = {"sealed_records": sealed_records} if sealed_records is not None else {}
    log.append("quest_finalized", events_before=len(audit_log.read(trace)), write_errors=0, records_not_written=0,
               model_calls={"lines": 0, "counts": {}, "gaps": []}, rigor_profile="research",
               nodes_completed=["review", "write"], paper_path="paper/paper.md",
               paper_sha256=evidence._file_sha256(root / "paper" / "paper.md"),
               files={rel: evidence._file_sha256(root / rel) for rel in named}, **extra)
    return trace


def _standing(root: Path) -> str:
    return evidence.verify_seal(root, dict(_RECORD))["trace_seal"]


def test_an_untouched_seal_verifies() -> None:
    with _Dir() as root:
        _sealed(root)
        assert _standing(root) == "verified"


@SEAL
@given(st.sampled_from(evidence.SEALED_FILES), st.text(min_size=1, max_size=10))
def test_any_change_to_a_sealed_file_breaks_the_seal(rel, tail) -> None:
    with _Dir() as root:
        _sealed(root)
        p = root / rel
        p.write_text(p.read_text(encoding="utf-8") + tail, encoding="utf-8")
        out = evidence.verify_seal(root, dict(_RECORD))
        assert out["trace_seal"] == "not_verified" and out["status"] != "publication_ready"
        assert any(rel in g for g in out["all_gaps"]["publication_ready"])


@SEAL
@given(st.sampled_from(evidence.SEALED_FILES))
def test_a_deleted_sealed_file_breaks_the_seal(rel) -> None:
    with _Dir() as root:
        _sealed(root)
        (root / rel).unlink()
        assert _standing(root) == "not_verified"


@SEAL
@given(st.text(min_size=1, max_size=10))
def test_a_changed_paper_breaks_the_seal(tail) -> None:
    with _Dir() as root:
        _sealed(root)
        p = root / "paper" / "paper.md"
        p.write_text(p.read_text(encoding="utf-8") + tail, encoding="utf-8")
        assert _standing(root) == "not_verified"


@SEAL
@given(_events)
def test_an_event_after_the_seal_breaks_it(events) -> None:
    with _Dir() as root:
        trace = _sealed(root)
        log = audit_log.AuditLog(trace, "q")
        kind, fields = events[0]
        log.append(kind, **fields)
        assert audit_log.verify(trace).ok, "the chain is still intact, only the seal is no longer last"
        assert _standing(root) == "not_verified"


@SEAL
@given(st.integers(1, 3))
def test_cutting_events_off_the_end_of_a_sealed_trace_breaks_it(n) -> None:
    with _Dir() as root:
        trace = _sealed(root)
        lines = _lines(trace)
        _write_lines(trace, lines[: len(lines) - n])
        assert audit_log.verify(trace).ok, "a shorter chain is still a valid chain; only the missing seal shows it"
        assert _standing(root) == "not_verified"


@SEAL
@given(st.data())
def test_a_forged_seal_with_a_wrong_hash_is_caught(data) -> None:
    with _Dir() as root:
        trace = _sealed(root)
        lines = _lines(trace)
        seal = json.loads(lines[-1])
        rel = data.draw(st.sampled_from(sorted(seal["files"])))
        seal["files"][rel] = data.draw(st.text(alphabet="0123456789abcdef", min_size=64, max_size=64))
        lines[-1] = json.dumps(seal)
        _write_lines(trace, lines)
        assert _standing(root) == "not_verified", "edited seal no longer matches its own hash in the chain"


def test_a_seal_from_before_the_shadow_record_still_verifies() -> None:
    old = tuple(f for f in evidence.SEALED_FILES if "shadow" not in f)
    with _Dir() as root:
        _sealed(root, files=old, sealed_records=None)
        assert _standing(root) == "verified"
    with _Dir() as root:
        _sealed(root, files=old, sealed_records=1)
        assert _standing(root) == "verified"


@SEAL
@given(st.sampled_from([f for f in evidence.SEALED_FILES if "shadow" not in f]), st.text(min_size=1, max_size=5))
def test_an_old_seal_still_notices_a_change_to_the_files_it_did_name(rel, tail) -> None:
    old = tuple(f for f in evidence.SEALED_FILES if "shadow" not in f)
    with _Dir() as root:
        _sealed(root, files=old, sealed_records=None)
        p = root / rel
        p.write_text(p.read_text(encoding="utf-8") + tail, encoding="utf-8")
        assert _standing(root) == "not_verified"


def test_a_current_seal_that_leaves_a_file_unnamed_is_not_verified() -> None:
    with _Dir() as root:
        _sealed(root, files=evidence.SEALED_FILES[:-1], sealed_records=evidence.SEAL_RECORDS)
        assert _standing(root) == "not_verified"


def test_the_record_cannot_vouch_for_its_own_seal() -> None:
    with _Dir() as root:
        _sealed(root)
        (root / "paper" / "paper.md").write_text("changed", encoding="utf-8")
        out = evidence.verify_seal(root, {**_RECORD, "trace_seal": "verified"})
        assert out["trace_seal"] == "not_verified"
