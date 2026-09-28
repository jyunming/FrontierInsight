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


def _row(node: str, response: str, **over) -> dict:
    return ar.model_call_row(node=node, attempt=1, served={"provider": "openai", "model": "m", "reported": True},
                             requested_model="m", reports_model=True, messages="p", response=response, **over)


def test_one_model_record_no_longer_makes_the_quest_s_end_complete(tmp_path: Path) -> None:
    root = tmp_path
    fi = root / ".fi"
    ar.append_model_call(fi, "q", _row("ideate", "a"))
    models_used = {"ideate": {"response_hash": ar._text_sha("a")}}
    counts = {"ideate": 1, "design": 2, "write": 1}  # the engine's own tally: it made these calls
    gaps, summary = ar.model_call_gaps(root, models_used, counts)
    assert any("fewer lines than the calls made" in g and "design (0 of 2)" in g and "write (0 of 1)" in g
               for g in gaps), gaps
    assert summary["lines"] == 1 and summary["sha256"]
    ctx = ar.context_fingerprint(_cfg(), root, {"survey_mode_resolved": True, "model_call_counts": counts},
                                 kind="quest_end", prompts={}, models_used=models_used)
    assert not ctx["complete"] and ctx["model_calls"]["lines"] == 1
    # more lines than the tally (a step paused and run again) is not a gap
    assert ar.model_call_gaps(root, models_used, {"ideate": 1})[0] == []


def test_the_call_record_s_other_gaps(tmp_path: Path) -> None:
    root = tmp_path
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
    for rel in (*evidence.SEALED_LEDGERS, evidence.SEALED_QUERIES):
        (root / rel).touch()
    files = {rel: (None if rel in null else evidence._file_sha256(root / rel))
             for rel in evidence.SEALED_FILES if rel not in drop}
    log.append("quest_finalized", events_before=len(audit_log.read(trace)), write_errors=0, records_not_written=0,
               model_calls={"lines": 0, "counts": {}, "gaps": []},
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


# --- review follow-ups --------------------------------------------------------------------------------------------

def test_a_call_after_the_seal_goes_to_its_own_record_and_leaves_the_seal_intact(tmp_path: Path) -> None:
    root = _sealed(tmp_path)
    (root / ".fi" / ar.MODEL_CALLS_CLOSED).write_text("sealed\n", encoding="utf-8")  # as the engine does at the seal
    append_cost_row(root / ".fi", node="slides", model="m", usage=None, messages=[{"role": "user", "content": "x"}],
                    response="deck")
    assert evidence.read(root)["trace_seal"] == "verified"
    (row,) = ar.read(root / ".fi", ar.MODEL_CALLS_AFTER_SEAL)
    assert row["node"] == "slides" and row["attempt"] == 1


def test_the_seal_s_model_call_gaps_are_trace_gaps(tmp_path: Path) -> None:
    root = _sealed(tmp_path)
    trace = root / ".fi" / "audit.jsonl"
    lines = trace.read_text(encoding="utf-8").splitlines()
    seal = json.loads(lines[-1])
    trace.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    fields = {k: v for k, v in seal.items() if k not in audit_log._RESERVED}
    fields["model_calls"] = {"lines": 1, "counts": {"write": 3},
                             "gaps": ["the record of model calls has fewer lines than the calls made: write (1 of 3)"]}
    audit_log.AuditLog(trace, root.name).append("quest_finalized", **fields)
    read = evidence.read(root)
    assert read["trace_seal"] == "not_verified" and read["levels"]["publication_ready"] is False
    assert any("model calls is incomplete" in g for g in read["all_gaps"]["publication_ready"])


class _Retrying(_Client):
    """A client whose call fails once (a retry the provider made) before it answers."""

    async def chat(self, messages, **kw):  # noqa: ANN001
        from core.provider import _note_failed_attempt

        _note_failed_attempt("openai", "m1", TimeoutError("slow"))
        return await super().chat(messages, **kw)


def test_every_attempt_is_a_line_and_the_counts_are_the_engine_s_own(tmp_path: Path) -> None:
    eng = _engine(tmp_path, _Retrying())
    asyncio.run(eng._chat("p", node="design"))
    rows = ar.read(eng.fi_dir, ar.MODEL_CALLS)
    assert [r["outcome"] for r in rows] == ["TimeoutError", "ok"] and [r["attempt"] for r in rows] == [1, 2]
    assert eng._model_call_counts == {"design": 2}
    assert rows[0]["prompt_sha256"] == eng._last_chat["design"]["prompt_hash"], "one prompt hash, joinable"


def test_the_fallback_chain_notes_the_provider_it_passed_over(tmp_path: Path) -> None:
    from core.provider import FallbackLLMClient
    from tests.test_provider_fallback import _factory_for, _FakeClient

    eng = _engine(tmp_path, FallbackLLMClient(_FakeClient("primary", error=TimeoutError("down")),
                                              [("fallback", _factory_for(_FakeClient("fallback"), [0]))]))
    asyncio.run(eng._chat("p", node="write"))
    rows = ar.read(eng.fi_dir, ar.MODEL_CALLS)
    assert [(r["provider"], r["outcome"]) for r in rows] == [("primary", "TimeoutError"), ("fallback", "ok")]
    assert rows[1]["fallback"] is True


def test_a_failed_call_says_who_was_asked(tmp_path: Path) -> None:
    class _Failing:
        last_usage = None
        last_model = "m1"

        async def chat(self, messages, **kw):  # noqa: ANN001
            LAST_CALL.set({"provider": "openai", "model": "m1", "reported": False})
            raise RuntimeError("no")

    eng = _engine(tmp_path, _Failing())
    with pytest.raises(RuntimeError):
        asyncio.run(eng._chat("p", node="analyze"))
    (row,) = ar.read(eng.fi_dir, ar.MODEL_CALLS)
    assert row["provider"] == "openai" and row["served_model"] == "m1" and row["outcome"] == "RuntimeError"


def test_each_call_keeps_its_own_token_counts_when_they_run_at_once(tmp_path: Path) -> None:
    class _Concurrent:
        last_model = "m"
        last_usage = None

        async def chat(self, messages, **kw):  # noqa: ANN001
            mine = {"prompt_tokens": len(messages[0]["content"]), "completion_tokens": 1, "total_tokens": 0}
            self.last_usage = mine  # shared by both calls
            await asyncio.sleep(0.01 if "long" in messages[0]["content"] else 0)
            LAST_CALL.set({"provider": "openai", "model": "m", "reported": True, "usage": mine})
            return "ok"

    eng = _engine(tmp_path, _Concurrent())

    async def both():
        await asyncio.gather(eng._chat("long prompt", node="review_panel.a"), eng._chat("p", node="review_panel.b"))

    asyncio.run(both())
    by_node = {r["node"]: r["usage"]["prompt_tokens"] for r in ar.read(eng.fi_dir, ar.MODEL_CALLS)}
    assert by_node == {"review_panel.a": len("long prompt"), "review_panel.b": 1}


def test_a_stream_that_names_no_model_is_not_a_report() -> None:
    import httpx

    from core.config import ProviderConfig
    from core.provider import resolve_endpoint
    from tests.test_provider_kimi_streaming import KIMI, _chat, _chunk, _sse

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(_chunk("hi"), _chunk(finish="stop")))

    _text, _client, last = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert last["reported"] is False


def test_bearer_takes_a_token_and_leaves_the_word() -> None:
    from core.audit_log import redact_text

    assert redact_text("the bearer protocol and a Bearer token", whole=True) == "the bearer protocol and a Bearer token"
    for token in ("sk-abcdefghijkl", "abc123def", "abcdefghijklmnopqrst"):
        assert token not in redact_text(f"Authorization: Bearer {token}", whole=True), token


@pytest.mark.parametrize("text", ["a Bearer token.", "use Bearer tokens.", "the bearer protocol"])
def test_bearer_in_a_sentence_is_kept(text: str) -> None:
    from core.audit_log import redact_text

    assert redact_text(text, whole=True) == text


@pytest.mark.parametrize("token", ["eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig", "sk-proj-abcdefgh", "ya29.a0AfH6SM",
                                   "QUJDREVGR0hJSktMTU5PUFFS+/xyz="])
def test_bearer_takes_token_shaped_values(token: str) -> None:
    from core.audit_log import redact_text

    assert token not in redact_text(f"Authorization: Bearer {token}.", whole=True)


class _NamesThenFails:
    last_usage = None
    last_model = "primary-model"

    async def chat(self, messages, **kw):  # noqa: ANN001
        LAST_CALL.set({"provider": "primary", "model": "primary-model", "reported": True})
        raise TimeoutError("down")


class _NamesNothing:
    last_usage = None
    last_model = None

    async def chat(self, messages, **kw):  # noqa: ANN001
        return "fallback answer"


def test_a_fallback_answer_that_names_no_model_is_not_taken_for_the_primary_s(tmp_path: Path) -> None:
    from core.provider import FallbackLLMClient

    async def make():
        return _NamesNothing()

    eng = _engine(tmp_path, FallbackLLMClient(_NamesThenFails(), [("fallback", make)]))
    asyncio.run(eng._chat("p", node="write"))
    rows = ar.read(eng.fi_dir, ar.MODEL_CALLS)
    assert rows[-1]["provider"] == "fallback" and rows[-1]["served_model"] != "primary-model"
    assert rows[-1]["reported"] is False


def test_a_retry_inside_a_fallback_provider_is_noted_under_it(tmp_path: Path) -> None:
    from core.provider import FallbackLLMClient, _note_failed_attempt

    class _RetriesThenAnswers:
        last_usage = None
        last_model = "fb-model"
        last_provider = "some-other-name"

        async def chat(self, messages, **kw):  # noqa: ANN001
            _note_failed_attempt(self.last_provider, "fb-model", TimeoutError("slow"))
            return "ok"

    async def make():
        return _RetriesThenAnswers()

    eng = _engine(tmp_path, FallbackLLMClient(_NamesThenFails(), [("fallback", make)]))
    asyncio.run(eng._chat("p", node="write"))
    rows = ar.read(eng.fi_dir, ar.MODEL_CALLS)
    assert [(r["provider"], r["fallback"], r["outcome"]) for r in rows] == [
        ("primary", False, "TimeoutError"), ("fallback", True, "TimeoutError"), ("fallback", True, "ok")]
