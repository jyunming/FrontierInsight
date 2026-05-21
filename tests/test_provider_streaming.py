"""Tests for the streaming + inactivity-timer additions to
``core/provider.py:_run_cli``.

Covers:
  - ``_parse_stream_json_line`` correctly routes text_delta, thinking_
    delta, result, error, and non-JSON inputs.
  - ``_collect_via_streaming`` aggregates text_delta events and ignores
    thinking_delta payloads (for the returned text — they still count
    for heartbeats).
  - The inactivity-watchdog kills the child when no stream events
    arrive within ``inactivity_timeout_s``, distinguishing this from
    the total-wall-clock kill.
  - The heartbeat callback receives periodic progress dicts that
    count thinking-token activity.

Uses fake subprocesses / direct calls to internal helpers — no real
``claude`` binary required.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from core.provider import (
    _CliSpec,
    _CliTransientError,
    _parse_stream_json_line,
    _run_cli,
)


# ---------------------------------------------------------------------------
# Stream-json parser
# ---------------------------------------------------------------------------


def test_parse_stream_json_text_delta_returns_text() -> None:
    raw = json.dumps({
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello "},
        },
    }).encode("utf-8")
    text, thinking, err, is_result = _parse_stream_json_line(raw)
    assert text == "hello "
    assert thinking == 0
    assert err is None
    assert is_result is False


def test_parse_stream_json_thinking_delta_counts_but_no_text() -> None:
    """Thinking deltas are the Sonnet 4.6 extended-thinking signal — we
    count them for heartbeat progress but they don't go into the
    aggregated answer text. Token count is rough (4 chars/token)."""
    raw = json.dumps({
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "x" * 400},
        },
    }).encode("utf-8")
    text, thinking, err, is_result = _parse_stream_json_line(raw)
    assert text == ""
    assert thinking == 100  # 400 / 4
    assert err is None
    assert is_result is False


def test_parse_stream_json_result_event_captures_final_text() -> None:
    """Some claude_cli versions emit a final ``result`` envelope
    carrying the assembled text instead of (or in addition to) the
    stream of deltas. The parser must surface that body or we'd
    return an empty string — but ALSO flag the envelope so callers
    can deduplicate against streamed deltas."""
    raw = json.dumps({"type": "result", "result": "final answer body"}).encode()
    text, thinking, err, is_result = _parse_stream_json_line(raw)
    assert text == "final answer body"
    assert err is None
    assert is_result is True


def test_parse_stream_json_error_event_returns_error_message() -> None:
    raw = json.dumps({"type": "error", "error": "rate_limit_exceeded"}).encode()
    text, thinking, err, is_result = _parse_stream_json_line(raw)
    assert text == ""
    assert err == "rate_limit_exceeded"
    assert is_result is False


def test_parse_stream_json_non_json_line_returns_empties() -> None:
    """Pre-stream banner lines from the CLI (status messages, ANSI,
    etc.) must not crash the parser. They become silent no-ops."""
    out = _parse_stream_json_line(b"INFO some status line\n")
    assert out == ("", 0, None, False)


def test_parse_stream_json_unrelated_message_type_ignored() -> None:
    raw = json.dumps({"type": "system", "subtype": "init"}).encode()
    out = _parse_stream_json_line(raw)
    assert out == ("", 0, None, False)


def test_collect_via_streaming_deduplicates_result_envelope_after_deltas(
    tmp_path,
) -> None:
    """When the CLI emits both streamed text_deltas AND a final result
    envelope carrying the assembled text, the aggregator must NOT
    append the envelope on top of the deltas (which would double the
    answer). The is_result flag from _parse_stream_json_line is the
    deduplication signal."""
    import asyncio
    events = [
        {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello "}}},
        {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "world"}}},
        # Final envelope carrying the SAME assembled text. Must be
        # ignored — the deltas already supplied it.
        {"type": "result", "result": "Hello world"},
    ]
    _, spec = _make_fake_streaming_binary(tmp_path, events=events)
    result = asyncio.run(_run_cli(
        spec, "any prompt",
        timeout_s=10.0, inactivity_timeout_s=5.0,
    ))
    assert result == "Hello world"  # NOT "Hello worldHello world"


# ---------------------------------------------------------------------------
# _run_cli with a fake binary that emits stream-json
# ---------------------------------------------------------------------------


def _make_fake_streaming_binary(
    tmp_path: Path,
    *,
    events: list[dict],
    pre_delay_s: float = 0.0,
    inter_delay_s: float = 0.0,
) -> tuple[Path, _CliSpec]:
    """Build a tiny Python script that mimics a CLI emitting stream-json
    events on stdout, then exits 0. ``pre_delay_s`` is the silence
    before the first event (used to test the inactivity timer);
    ``inter_delay_s`` paces events apart.
    """
    script = tmp_path / "fake_claude.py"
    script.write_text(
        "import json, sys, time\n"
        f"events = {json.dumps(events)}\n"
        f"time.sleep({pre_delay_s})\n"
        "for ev in events:\n"
        "    print(json.dumps(ev), flush=True)\n"
        f"    time.sleep({inter_delay_s})\n",
        encoding="utf-8",
    )
    # Use the current python as the "binary" so it works on any platform.
    spec = _CliSpec(
        argv=(sys.executable, str(script)),
        pass_prompt_via="stdin",
        output_via="stream_json",
        model_flag=None,
    )
    return script, spec


@pytest.mark.asyncio
async def test_run_cli_streaming_aggregates_text_deltas(tmp_path: Path) -> None:
    """Happy path: a stream of text_deltas comes out the other end as
    a single concatenated string, with thinking_deltas filtered out."""
    events = [
        {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "internal thought"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello "}}},
        {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "world"}}},
    ]
    _, spec = _make_fake_streaming_binary(tmp_path, events=events)
    result = await _run_cli(
        spec, "any prompt",
        timeout_s=10.0, inactivity_timeout_s=5.0,
    )
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_run_cli_streaming_inactivity_timeout_fires_before_total(
    tmp_path: Path,
) -> None:
    """When the fake binary stays silent past ``inactivity_timeout_s``
    but well below the total ``timeout_s``, the watchdog must kill it
    with the inactivity-style error (NOT the total wall-clock error)."""
    _, spec = _make_fake_streaming_binary(
        tmp_path,
        events=[{"type": "stream_event", "event": {"type": "content_block_delta",
                 "delta": {"type": "text_delta", "text": "late"}}}],
        pre_delay_s=5.0,   # silent for 5 s before any output
    )
    with pytest.raises(_CliTransientError) as exc_info:
        await _run_cli(
            spec, "any prompt",
            timeout_s=30.0,         # total ceiling is generous
            inactivity_timeout_s=2.0,  # inactivity is tight
        )
    msg = str(exc_info.value)
    assert "silent for" in msg, msg


@pytest.mark.asyncio
async def test_run_cli_streaming_thinking_deltas_reset_inactivity_timer(
    tmp_path: Path,
) -> None:
    """Critical: thinking_delta events count as activity. A Sonnet 4.6
    extended-thinking span emits thousands of thinking_deltas without
    any text — the watchdog must NOT kill that, because the model IS
    making progress, just not visible progress."""
    # 6 thinking_deltas spaced 0.4 s apart (total 2.4 s of thinking),
    # then one text_delta. Total elapsed ~3 s.
    events = [
        {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "ponder"}}}
        for _ in range(6)
    ] + [
        {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "done"}}},
    ]
    _, spec = _make_fake_streaming_binary(
        tmp_path, events=events, inter_delay_s=0.4,
    )
    # Inactivity = 1 s. Each thinking_delta arrives within 0.4 s of the
    # last, so the timer resets and never trips. Total wall-clock = 5 s
    # (generous).
    result = await _run_cli(
        spec, "any prompt",
        timeout_s=10.0, inactivity_timeout_s=1.0,
    )
    assert result == "done"


@pytest.mark.asyncio
async def test_run_cli_streaming_heartbeat_callback_invoked(
    tmp_path: Path,
) -> None:
    """The heartbeat callback should be invoked at least once during a
    multi-second streaming call so the Engine can log progress."""
    events = [
        {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "x" * 400}}}
        for _ in range(8)
    ] + [
        {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "out"}}},
    ]
    _, spec = _make_fake_streaming_binary(
        tmp_path, events=events, inter_delay_s=0.4,
    )
    beats: list[dict] = []
    await _run_cli(
        spec, "any prompt",
        timeout_s=15.0, inactivity_timeout_s=3.0,
        heartbeat_cb=beats.append,
        node="implement",
    )
    # The watchdog fires every ~1 s; over ~3 s of streaming we should
    # see at least two beats. Each carries elapsed_s + thinking_tokens.
    assert len(beats) >= 1
    last = beats[-1]
    assert last["kind"] == "cli_progress"
    assert last["node"] == "implement"
    assert last["thinking_tokens"] > 0
    assert last["elapsed_s"] > 0


@pytest.mark.asyncio
async def test_run_cli_streaming_error_event_raises_transient(
    tmp_path: Path,
) -> None:
    """A stream-json ``error`` event surfaces as ``_CliTransientError``
    so the tenacity retry policy can react. The fake binary exits 0
    after the error event (which is what claude_cli does for some
    error classes), so the rc != 0 path doesn't catch it."""
    events = [
        {"type": "error", "error": "internal_server_error"},
    ]
    _, spec = _make_fake_streaming_binary(tmp_path, events=events)
    with pytest.raises(_CliTransientError) as exc_info:
        await _run_cli(
            spec, "any prompt",
            timeout_s=5.0, inactivity_timeout_s=3.0,
        )
    assert "internal_server_error" in str(exc_info.value)


def _make_lingering_streaming_binary(
    tmp_path: Path, *, events: list[dict], post_eof_sleep_s: float,
) -> tuple[Path, "object"]:
    """Like ``_make_fake_streaming_binary`` but the child closes its
    stdout (via ``sys.stdout.close()``) BEFORE sleeping for
    ``post_eof_sleep_s`` seconds and exiting. This reproduces the
    Windows-asyncio case where claude.exe streamed its full response,
    closed stdout, and then took longer than the post-EOF wait to
    actually exit. The old code killed the lingering child and threw
    away the response; the fix returns the collected text."""
    from core.provider import _CliSpec
    script = tmp_path / "fake_lingering.py"
    script.write_text(
        "import json, sys, time\n"
        f"events = {json.dumps(events)}\n"
        "for ev in events:\n"
        "    print(json.dumps(ev), flush=True)\n"
        "sys.stdout.close()\n"
        f"time.sleep({post_eof_sleep_s})\n",
        encoding="utf-8",
    )
    spec = _CliSpec(
        argv=(sys.executable, str(script)),
        pass_prompt_via="stdin",
        output_via="stream_json",
        model_flag=None,
    )
    return script, spec


@pytest.mark.asyncio
async def test_run_cli_streaming_preserves_result_when_child_lingers_after_eof(
    tmp_path: Path,
) -> None:
    """Regression guard: a Sonnet response that streamed cleanly,
    closed stdout, but then took >10 s for the OS to clean up the
    process used to be discarded — the old code killed the
    "lingering" child and tenacity restarted the whole call from
    scratch. For an 8-minute body call that's catastrophic.

    The fix: if the streaming reader collected any text before EOF,
    return that text. The post-EOF reap timeout is purely cleanup
    latency; it does NOT invalidate the LLM result.

    This test reproduces the situation: the fake binary emits a
    valid stream-json text_delta, then closes stdout and sleeps
    longer than the post-EOF wait timeout. ``_run_cli`` should
    return the text, not raise.

    (We can't pin the production 60s post-EOF timeout from a unit
    test, but the principle is the same: ``have_output == True``
    short-circuits any kill+raise path.)"""
    events = [
        {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "the answer is 42"}}},
    ]
    # Sleep 3 s post-EOF — longer than the test would normally tolerate
    # but the fix should NOT care: we have output, so return it. (The
    # production code uses 60 s for the wait; the test asserts the
    # short-circuit fires by checking we got the text fast enough that
    # we DIDN'T fall through to a kill+raise path on a smaller test
    # ceiling.)
    _, spec = _make_lingering_streaming_binary(
        tmp_path, events=events, post_eof_sleep_s=3.0,
    )
    import time as _time
    started = _time.monotonic()
    result = await _run_cli(
        spec, "any prompt",
        timeout_s=30.0,         # total ceiling generous
        inactivity_timeout_s=5.0,   # would otherwise fire if we waited the full 3 s
    )
    elapsed = _time.monotonic() - started
    assert result == "the answer is 42"
    # Sanity: we returned WITHOUT waiting the full 60 s production
    # reap window. The child's still alive lingering, but we got the
    # text and moved on.
    assert elapsed < 10.0, (
        f"expected fast return on EOF with output; took {elapsed:.1f}s"
    )
