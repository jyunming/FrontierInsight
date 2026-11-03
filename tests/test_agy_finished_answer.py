"""agy can finish a turn on a large prompt, then print nothing and not exit. FI used to wait the whole timeout (30 min,
four tries). It now reads the answer from agy's own per-call record once that has stood unchanged, and stops the call.
Nothing here calls the real agy: the process is faked."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import core.provider as provider
from core.provider import _CLI_SPECS, _read_agy_record


def _write_record(home: Path, steps: list[dict], conv: str = "c1") -> Path:
    logs = home / ".gemini" / "antigravity-cli" / "brain" / conv / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "transcript_full.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in steps) + "\n", encoding="utf-8")
    return path


USER = {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "content": "x" * 90000}


def _reply(content: str, **over) -> dict:
    step = {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "content": content,
            "input_tokens": 35698, "output_tokens": 25271, "cache_read_tokens": 0, "thinking": "..."}
    step.update(over)
    return step


def test_a_finished_reply_is_read_whole_with_its_token_counts(tmp_path: Path) -> None:
    answer = '```json\n{"items": [1, 2, 3]}\n```\n'
    _write_record(tmp_path, [USER, _reply(answer)])
    got = _read_agy_record(str(tmp_path))
    assert got is not None
    text, usage = got
    assert text == answer
    assert usage == {"prompt_tokens": 35698, "completion_tokens": 25271, "total_tokens": 60969,
                     "cached_input_tokens": 0, "estimated": False}


@pytest.mark.parametrize("last", [
    _reply("partial", status="RUNNING"),            # still being written
    {"step_index": 2, "source": "MODEL", "type": "RUN_COMMAND", "status": "DONE", "content": "ls"},  # a tool step follows
])
def test_an_unfinished_or_partial_record_gives_nothing(tmp_path: Path, last: dict) -> None:
    _write_record(tmp_path, [USER, _reply("an earlier reply"), last] if last.get("type") == "RUN_COMMAND" else [USER, last])
    assert _read_agy_record(str(tmp_path)) is None


@pytest.mark.parametrize("last", [
    _reply("cut", truncated_fields=["content"]),    # the record itself cut the reply: never a partial answer
    _reply("   "),                                  # nothing to return
])
def test_a_finished_but_unusable_reply_is_reported_not_taken(tmp_path: Path, last: dict) -> None:
    _write_record(tmp_path, [USER, last])
    with pytest.raises(provider._CliRecordIncomplete):
        _read_agy_record(str(tmp_path))


def test_a_half_written_line_or_a_missing_or_double_record_gives_nothing(tmp_path: Path) -> None:
    assert _read_agy_record(str(tmp_path / "nowhere")) is None
    path = _write_record(tmp_path, [USER, _reply("ok")])
    path.write_text(path.read_text(encoding="utf-8") + '{"step_index": 2, "source": "MO', encoding="utf-8")
    assert _read_agy_record(str(tmp_path)) is None
    path.write_text(json.dumps(USER) + "\n" + json.dumps(_reply("ok")) + "\n", encoding="utf-8")
    assert _read_agy_record(str(tmp_path)) is not None
    _write_record(tmp_path, [USER, _reply("other")], conv="c2")   # two conversations: do not guess which
    assert _read_agy_record(str(tmp_path)) is None


class _FakeProc:
    """A faked agy process with real stream readers: it prints what ``feed_out`` gives, and exits only when told."""

    def __init__(self) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdin = MagicMock()

        async def drain() -> None:
            return None

        self.stdin.drain = drain
        self.returncode = None
        self._exited = asyncio.Event()

    def exit(self, code: int = 0) -> None:
        self.returncode = code
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self._exited.set()

    async def wait(self) -> int:
        await self._exited.wait()
        return self.returncode


def _hung_proc() -> _FakeProc:
    """An agy that never prints and never exits."""
    return _FakeProc()


def _fast(monkeypatch, stable: float = 0.1) -> None:
    monkeypatch.setattr(provider, "_CLI_RECORD_POLL_S", 0.02)
    monkeypatch.setattr(provider, "_CLI_RECORD_STABLE_S", stable)

    async def fake_kill(*_a, **_k):
        return True

    monkeypatch.setattr(provider, "_kill_and_reap", fake_kill)


@pytest.mark.asyncio
async def test_a_silent_agy_with_a_finished_record_returns_the_answer_fast(tmp_path: Path, monkeypatch) -> None:
    """The original failure: record says DONE, stdout empty, no exit. Before: waits out ``timeout_s``."""
    monkeypatch.setattr(provider, "_CLI_RECORD_POLL_S", 0.02)
    monkeypatch.setattr(provider, "_CLI_RECORD_STABLE_S", 0.1)
    killed: list[bool] = []

    async def fake_kill(*_a, **_k):
        killed.append(True)
        return True

    monkeypatch.setattr(provider, "_kill_and_reap", fake_kill)
    _write_record(tmp_path, [USER, _reply("the whole answer\n")])
    usage: dict = {}
    spec = _CLI_SPECS["antigravity_cli"]
    out = await asyncio.wait_for(provider._collect_via_communicate(
        _hung_proc(), ["agy"], spec, b"{}", None, timeout_s=60.0, usage_out=usage,
        record=lambda: _read_agy_record(str(tmp_path)),
    ), timeout=10)
    assert out == "the whole answer"
    assert killed == [True]
    assert usage["prompt_tokens"] == 35698 and usage["estimated"] is False


@pytest.mark.asyncio
async def test_a_record_that_keeps_changing_is_not_taken(tmp_path: Path, monkeypatch) -> None:
    """A reply followed by more steps is not the end of the turn: only a record that stands still is taken, and with no
    record at all the call times out as before."""
    monkeypatch.setattr(provider, "_CLI_RECORD_POLL_S", 0.02)
    monkeypatch.setattr(provider, "_CLI_RECORD_STABLE_S", 0.2)

    async def no_kill(*_a, **_k):
        return True

    monkeypatch.setattr(provider, "_kill_and_reap", no_kill)
    spec = _CLI_SPECS["antigravity_cli"]
    n = {"i": 0}

    def changing():
        n["i"] += 1
        return f"reply {n['i']}", None

    with pytest.raises(provider._CliTransientError, match="exceeded"):
        await provider._collect_via_communicate(
            _hung_proc(), ["agy"], spec, b"{}", None, timeout_s=0.5, record=changing,
        )
    with pytest.raises(provider._CliTransientError, match="exceeded"):
        await provider._collect_via_communicate(
            _hung_proc(), ["agy"], spec, b"{}", None, timeout_s=0.5, record=lambda: None,
        )


@pytest.mark.asyncio
async def test_a_normal_call_uses_its_stdout_not_the_record(tmp_path: Path, monkeypatch) -> None:
    """Known positive: a call that prints its result and exits is answered from stdout, as before."""
    out = json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "pong\n",
                                                    "usage": {"input_tokens": 10, "output_tokens": 2}}}) + "\n"
    proc = _FakeProc()

    async def later() -> None:
        await asyncio.sleep(0.05)
        proc.stdout.feed_data(out.encode())
        proc.exit(0)

    task = asyncio.ensure_future(later())
    _write_record(tmp_path, [USER, _reply("something else")])
    usage: dict = {}
    got = await provider._collect_via_communicate(
        proc, ["agy"], _CLI_SPECS["antigravity_cli"], b"{}", None, timeout_s=30.0, usage_out=usage,
        record=lambda: _read_agy_record(str(tmp_path)),
    )
    await task
    assert got == "pong"
    assert usage["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_a_call_that_printed_something_is_never_answered_from_the_record(tmp_path: Path, monkeypatch) -> None:
    """Finding 1: agy printed an error (or a partial result) and has not exited while its record says DONE. The record
    must not replace what it printed: the call goes on to the normal path (here: waits, then times out)."""
    _fast(monkeypatch, stable=0.1)
    proc = _FakeProc()
    proc.stdout.feed_data(b'{"event": "error", "message": "quota"}\n')
    _write_record(tmp_path, [USER, _reply("the record answer")])
    with pytest.raises(provider._CliTransientError, match="exceeded"):
        await provider._collect_via_communicate(
            proc, ["agy"], _CLI_SPECS["antigravity_cli"], b"{}", None, timeout_s=0.6,
            record=lambda: _read_agy_record(str(tmp_path)),
        )


@pytest.mark.asyncio
async def test_a_reply_that_does_not_end_the_turn_is_not_taken(tmp_path: Path, monkeypatch) -> None:
    """Finding 2: a model reply after a tool step is an intermediate reply, not the answer: fail fast and plain."""
    _fast(monkeypatch, stable=0.1)
    tool = {"step_index": 1, "source": "MODEL", "type": "RUN_COMMAND", "status": "DONE", "content": "ls"}
    _write_record(tmp_path, [USER, tool, _reply("I will look at the files first.", step_index=2)])
    with pytest.raises(provider._CliTransientError, match="not shown to end the turn"):
        await asyncio.wait_for(provider._collect_via_communicate(
            _hung_proc(), ["agy"], _CLI_SPECS["antigravity_cli"], b"{}", None, timeout_s=60.0,
            record=lambda: _read_agy_record(str(tmp_path)),
        ), timeout=10)


def test_the_shape_of_a_finished_turn_is_two_steps(tmp_path: Path) -> None:
    _write_record(tmp_path, [USER, _reply("ok")])
    assert _read_agy_record(str(tmp_path)) is not None
    _write_record(tmp_path, [USER, _reply("ok"), _reply("and more", step_index=2)])
    with pytest.raises(provider._CliRecordIncomplete):
        _read_agy_record(str(tmp_path))
    _write_record(tmp_path, [USER, _reply("ok", tool_calls=[{"name": "x"}])])
    with pytest.raises(provider._CliRecordIncomplete):
        _read_agy_record(str(tmp_path))


@pytest.mark.asyncio
async def test_a_finished_but_cut_reply_fails_fast_not_after_the_full_timeout(tmp_path: Path, monkeypatch) -> None:
    """Finding 3: DONE with a cut content field used to read as 'not finished', so the wait ran to the full timeout."""
    _fast(monkeypatch, stable=0.1)
    _write_record(tmp_path, [USER, _reply("cut", truncated_fields=["content"])])
    with pytest.raises(provider._CliTransientError, match="cut its finished reply"):
        await asyncio.wait_for(provider._collect_via_communicate(
            _hung_proc(), ["agy"], _CLI_SPECS["antigravity_cli"], b"{}", None, timeout_s=60.0,
            record=lambda: _read_agy_record(str(tmp_path)),
        ), timeout=10)


def test_only_agy_reads_a_record() -> None:
    assert [n for n, s in _CLI_SPECS.items() if s.record_reader is not None] == ["antigravity_cli"]
