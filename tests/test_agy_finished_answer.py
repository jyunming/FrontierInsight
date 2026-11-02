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
    _reply("cut", truncated_fields=["content"]),    # the record itself cut the reply: never a partial answer
    _reply("   "),                                  # nothing to return
    {"step_index": 2, "source": "MODEL", "type": "RUN_COMMAND", "status": "DONE", "content": "ls"},  # a tool step follows
])
def test_an_unfinished_or_partial_record_gives_nothing(tmp_path: Path, last: dict) -> None:
    _write_record(tmp_path, [USER, _reply("an earlier reply"), last] if last.get("type") == "RUN_COMMAND" else [USER, last])
    assert _read_agy_record(str(tmp_path)) is None


def test_a_half_written_line_or_a_missing_or_double_record_gives_nothing(tmp_path: Path) -> None:
    assert _read_agy_record(str(tmp_path / "nowhere")) is None
    path = _write_record(tmp_path, [USER, _reply("ok")])
    path.write_text(path.read_text(encoding="utf-8") + '{"step_index": 2, "source": "MO', encoding="utf-8")
    assert _read_agy_record(str(tmp_path)) is None
    path.write_text(json.dumps(_reply("ok")) + "\n", encoding="utf-8")
    assert _read_agy_record(str(tmp_path)) is not None
    _write_record(tmp_path, [USER, _reply("other")], conv="c2")   # two conversations: do not guess which
    assert _read_agy_record(str(tmp_path)) is None


def _hung_proc():
    """An agy that never prints and never exits: communicate() waits forever."""
    proc = MagicMock()
    proc.returncode = None

    async def never(*_a, **_k):
        await asyncio.sleep(3600)

    proc.communicate = never
    return proc


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
    proc = MagicMock()
    proc.returncode = 0

    async def comm(*_a, **_k):
        await asyncio.sleep(0.05)
        return out.encode(), b""

    proc.communicate = comm
    _write_record(tmp_path, [USER, _reply("something else")])
    usage: dict = {}
    got = await provider._collect_via_communicate(
        proc, ["agy"], _CLI_SPECS["antigravity_cli"], b"{}", None, timeout_s=30.0, usage_out=usage,
        record=lambda: _read_agy_record(str(tmp_path)),
    )
    assert got == "pong"
    assert usage["prompt_tokens"] == 10


def test_only_agy_reads_a_record() -> None:
    assert [n for n, s in _CLI_SPECS.items() if s.record_reader is not None] == ["antigravity_cli"]
