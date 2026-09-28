"""The 2026-09-28 re-audit's provenance findings (F-02..F-05), each a counterexample that used to pass.

- F-02: one model record made a quest's end context complete; now every call is a line in .fi/model_calls.jsonl and the
  quest's end is compared with it.
- F-03: a same-size edit with its modification time put back reused an old code hash; trust boundaries read again.
- F-04: a seal could leave the attempt records out; every record file is required, with a hash.
- F-05: the retry line had its own, narrower credential filter; it uses the audit trace's.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import attempt_records as ar
from core import audit_log, evidence
from core.config import Config
from core.engine import Engine
from core.provider import LAST_CALL, _retry_line, append_cost_row
from tests.test_evidence import ON, _quest, _state

RESEARCH = {**ON, "rigor_profile": "research"}


def _cfg() -> Config:
    return Config.model_validate({"topic": "t", "provider": {"name": "openai", "model": "m1"},
                                  "knowledge": {"enabled": False}})


# --- F-02: every model call is a line ------------------------------------------------------------------------------

class _Client:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.last_usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        self.last_model = "m1"

    async def chat(self, messages, **kw):  # noqa: ANN001
        if self.fail:
            raise TimeoutError("the model did not answer")
        LAST_CALL.set({"provider": "openai", "model": "m1-served", "reported": True})
        return "an answer"


def _engine(tmp_path: Path, client: _Client) -> Engine:
    eng = object.__new__(Engine)
    eng.quest_id = "q1"
    eng.quest_root = tmp_path
    eng.fi_dir = tmp_path / ".fi"
    eng._client = client
    eng._last_chat = {}
    eng._reports_model = True
    eng._log = logging.getLogger("test")
    eng.config = SimpleNamespace(provider=SimpleNamespace(name="openai", model="m1", node_models={}))
    return eng


def test_each_call_is_a_line_and_a_failed_call_too(tmp_path: Path) -> None:
    eng = _engine(tmp_path, _Client())
    asyncio.run(eng._chat("the prompt", node="design"))
    asyncio.run(eng._chat("another prompt", node="design"))
    rows = ar.read(eng.fi_dir, ar.MODEL_CALLS)
    assert [r["attempt"] for r in rows] == [1, 2] and all(r["node"] == "design" for r in rows)
    first = rows[0]
    assert first["served_model"] == "m1-served" and first["reported"] and first["outcome"] == "ok"
    assert first["prompt_sha256"] and first["response_sha256"] and first["quest_id"] == "q1"
    assert "the prompt" not in (eng.fi_dir / ar.MODEL_CALLS).read_text(encoding="utf-8"), "never the text"
    eng._client = _Client(fail=True)
    with pytest.raises(TimeoutError):
        asyncio.run(eng._chat("p", node="write"))
    assert ar.read(eng.fi_dir, ar.MODEL_CALLS)[-1]["outcome"] == "TimeoutError"


def test_a_generator_call_is_a_line(tmp_path: Path) -> None:
    fi = tmp_path / "q9" / ".fi"
    token = LAST_CALL.set({"provider": "openai", "model": "m-slides", "reported": True})
    try:
        append_cost_row(fi, node="slides", model="m-slides", usage=None,
                        messages=[{"role": "user", "content": "x"}], response="deck")
    finally:
        LAST_CALL.reset(token)  # the test's own context is shared with the tests after it
    (row,) = ar.read(fi, ar.MODEL_CALLS)
    assert row["node"] == "slides" and row["served_model"] == "m-slides" and row["quest_id"] == "q9"
    append_cost_row(fi, node="design", model="m", usage=None, ledger=False)
    assert len(ar.read(fi, ar.MODEL_CALLS)) == 1, "the engine writes its own line"


def _trace(root: Path, *nodes: str) -> None:
    log = audit_log.AuditLog(root / ".fi" / "audit.jsonl", root.name)
    for node in nodes:
        log.append("node_completed", node=node)


def _row(node: str, response: str, **over) -> dict:
    return ar.model_call_row(node=node, attempt=1, served={"provider": "openai", "model": "m", "reported": True},
                             requested_model="m", reports_model=True, messages="p", response=response, **over)


def test_one_model_record_no_longer_makes_the_quest_s_end_complete(tmp_path: Path) -> None:
    root = tmp_path
    _trace(root, "ideate", "design", "write", "review")
    fi = root / ".fi"
    ar.append_model_call(fi, "q", _row("ideate", "a"))
    models_used = {"ideate": {"response_hash": ar._text_sha("a")}}
    gaps, summary = ar.model_call_gaps(root, models_used)
    assert any("completed with no model call recorded: write" in g for g in gaps), gaps
    assert summary["lines"] == 1 and summary["sha256"]
    ctx = ar.context_fingerprint(_cfg(), root, {"survey_mode_resolved": True}, kind="quest_end", prompts={},
                                 models_used=models_used)
    assert not ctx["complete"] and ctx["model_calls"]["lines"] == 1


def test_the_call_record_s_other_gaps(tmp_path: Path) -> None:
    root = tmp_path
    _trace(root, "design", "write")
    fi = root / ".fi"
    ar.append_model_call(fi, "q", _row("design", "d"))
    ar.append_model_call(fi, "q", _row("write", "w"))
    assert ar.model_call_gaps(root, {"design": {"response_hash": ar._text_sha("d")}})[0] == []
    # the engine's last call of a step is not in the record
    gaps, _ = ar.model_call_gaps(root, {"review": {"response_hash": ar._text_sha("r")}})
    assert any("last call of review" in g for g in gaps)
    # a connection that names the model did not name it
    ar.append_model_call(fi, "q", ar.model_call_row(node="write", attempt=2, served={"provider": "openai"},
                                                    requested_model="m", reports_model=True, messages="p",
                                                    response="w2"))
    assert any("did not name it" in g for g in ar.model_call_gaps(root, {})[0])
    # lines that could not be written
    ar.count_lost(fi, ar.MODEL_CALLS_LOST)
    assert any("could not be written" in g for g in ar.model_call_gaps(root, {})[0])


# --- F-03: trust boundaries read the code again ----------------------------------------------------------------------

def test_a_same_size_edit_with_its_time_put_back_changes_the_hash_at_the_end(tmp_path: Path) -> None:
    script = tmp_path / "code" / "simulate.py"
    script.parent.mkdir()
    script.write_text("print('A')\n", encoding="utf-8")
    stat = script.stat()
    cache: dict = {}
    before = ar.context_fingerprint(_cfg(), tmp_path, {}, kind="in_progress", prompts={}, cache=cache)["code"]
    script.write_text("print('B')\n", encoding="utf-8")
    os.utime(script, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert ar.context_fingerprint(_cfg(), tmp_path, {}, kind="in_progress", prompts={}, cache=cache)["code"] == before, \
        "the cache is still used for a step in progress (the counterexample, where it is harmless)"
    for kind in ("after_run", "quest_end"):
        assert ar.context_fingerprint(_cfg(), tmp_path, {}, kind=kind, prompts={}, cache=cache)["code"] != before, kind


# --- F-04: the seal names every record ---------------------------------------------------------------------------

def _sealed(tmp_path: Path, drop: tuple[str, ...] = (), null: tuple[str, ...] = ()) -> Path:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    trace = root / ".fi" / "audit.jsonl"
    trace.unlink(missing_ok=True)
    log = audit_log.AuditLog(trace, root.name)
    log.append("node_completed", node="write")
    log.append("node_completed", node="review")
    record = evidence.assess(root, _state(), settings={**RESEARCH, "sealing": True})
    (root / "needs" / "EVIDENCE.json").write_text(json.dumps(record), encoding="utf-8")
    for rel in evidence.SEALED_LEDGERS:
        (root / rel).touch()
    files = {rel: (None if rel in null else evidence._file_sha256(root / rel))
             for rel in evidence.SEALED_FILES if rel not in drop}
    log.append("quest_finalized", events_before=len(audit_log.read(trace)), write_errors=0, records_not_written=0,
               nodes_completed=["review", "write"], files=files, paper_path="paper/paper.md",
               paper_sha256=evidence._file_sha256(root / "paper" / "paper.md"), rigor_profile="research")
    return root


def test_a_seal_that_leaves_out_the_attempt_records_is_not_verified(tmp_path: Path) -> None:
    assert evidence.read(_sealed(tmp_path / "ok"))["trace_seal"] == "verified"
    left_out = evidence.read(_sealed(tmp_path / "a", drop=(".fi/attempts.jsonl", ".fi/branch_ledger.jsonl")))
    assert left_out["trace_seal"] == "not_verified"
    assert any("names no hash of .fi/attempts.jsonl" in g for g in left_out["all_gaps"]["publication_ready"])
    nulled = evidence.read(_sealed(tmp_path / "b", null=(".fi/model_calls.jsonl",)))
    assert any("names no hash of .fi/model_calls.jsonl" in g for g in nulled["all_gaps"]["publication_ready"])


# --- F-05: one credential filter -----------------------------------------------------------------------------------

class _Outcome:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def exception(self) -> BaseException:
        return self._exc


@pytest.mark.parametrize("secret", [
    "ghp_ABCDEFGHIJKLMNOPQRSTUV",
    "AIzaSyA1234567890abcdefghijklmnop",
    "Bearer abcdefghijklmnop1234",
    "https://api.x/v1?key=abcdef123456",
    "sk-abcdefghijklmnopqrstu",
])
def test_the_retry_line_keeps_no_credential(secret: str, monkeypatch: pytest.MonkeyPatch) -> None:
    rs = SimpleNamespace(outcome=_Outcome(RuntimeError(f"failed with {secret}\nsecond line")), attempt_number=1,
                         next_action=SimpleNamespace(sleep=2))
    line = _retry_line("write", "HTTP", rs, 4)
    token = secret.split("=")[-1].split()[-1]
    assert token not in line and "[redacted]" in line and "\n" not in line, line


def test_the_retry_line_hides_a_secret_environment_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FI_SOMETHING_API_KEY", "plain-value-no-shape-9f3")
    rs = SimpleNamespace(outcome=_Outcome(RuntimeError("the server said plain-value-no-shape-9f3 is wrong")),
                         attempt_number=2, next_action=None)
    assert "plain-value-no-shape-9f3" not in _retry_line("design", "HTTP", rs, 4)
