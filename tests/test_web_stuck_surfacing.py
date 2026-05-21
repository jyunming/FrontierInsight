"""Tests for the stuck-quest surfacing added in Phase 1.

Covers:
  - ``_parse_log_timestamp`` correctly extracts engine log timestamps.
  - ``_node_progress_from_log`` returns the start, elapsed, and idle
    seconds for the most recent node tag in a tail.
  - ``_read_quest_failed_md`` extracts failing-node + what-broke lines
    from the engine's diagnostic file.
  - ``GET /api/quests/{id}`` returns these fields, and the dashboard
    UI can render the badge + banner from them.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.server import (
    _KNOWN_NODES,
    _node_progress_from_log,
    _parse_log_timestamp,
    _read_quest_failed_md,
    make_app,
)


def test_parse_log_timestamp_round_trip() -> None:
    line = "2026-05-21 00:12:56,798 [INFO] [implement] generating experiment code"
    ts = _parse_log_timestamp(line)
    assert ts is not None
    # ``2026-05-21 00:12:56`` parsed as local time. Sanity-check by
    # round-tripping through datetime.
    dt = datetime.fromtimestamp(ts)
    assert dt.year == 2026
    assert dt.month == 5
    assert dt.day == 21
    assert dt.hour == 0
    assert dt.minute == 12
    assert dt.second == 56


def test_parse_log_timestamp_returns_none_for_untimestamped_line() -> None:
    assert _parse_log_timestamp("    raise _CliTransientError(") is None
    assert _parse_log_timestamp("RESULT_JSON: {...}") is None


def test_node_progress_finds_start_and_idle_in_log() -> None:
    """Standard case: the log opens with ``[design]``, then ~3 s later
    moves into ``[implement]``, then 5 s later writes another line in
    the same node. ``node_progress`` should report ``implement`` as
    the current node, elapsed ~8 s from the implement-open, and idle
    ~0 s from "now" (now = last line + 0)."""
    lines = [
        "2026-05-21 00:00:00,000 [INFO] [design] iteration=0",
        "2026-05-21 00:00:03,000 [INFO] [implement] generating experiment code",
        "2026-05-21 00:00:08,000 [INFO] [implement] still waiting on LLM",
    ]
    # ``now`` chosen as 10 s after the implement open, so elapsed=10s.
    # Last activity is lines[2] at implement_open+5s, so idle =
    # now - last_activity = 10 - 5 = 5 s.
    now_ts = _parse_log_timestamp(lines[1])
    assert now_ts is not None
    now = now_ts + 10.0
    progress = _node_progress_from_log(lines, _KNOWN_NODES, now=now)
    assert progress["node_started_at"] == now_ts
    assert abs(progress["node_elapsed_s"] - 10.0) < 0.01
    assert abs(progress["node_idle_s"] - 5.0) < 0.01


def test_node_progress_caps_idle_at_elapsed() -> None:
    """A node that just started 0.5 s ago shouldn't report idle = 30 s
    even if the file's mtime is older. Cap idle <= elapsed."""
    lines = ["2026-05-21 00:00:00,000 [INFO] [implement] just started"]
    now_ts = _parse_log_timestamp(lines[0])
    assert now_ts is not None
    now = now_ts + 0.5
    progress = _node_progress_from_log(lines, _KNOWN_NODES, now=now)
    assert progress["node_elapsed_s"] is not None
    assert progress["node_idle_s"] is not None
    assert progress["node_idle_s"] <= progress["node_elapsed_s"]


def test_node_progress_unknown_node_returns_nulls() -> None:
    lines = ["2026-05-21 00:00:00,000 [INFO] just some preamble"]
    progress = _node_progress_from_log(lines, _KNOWN_NODES)
    assert progress["node_started_at"] is None
    assert progress["node_elapsed_s"] is None
    assert progress["node_idle_s"] is None


def test_read_quest_failed_md_extracts_node_and_exception(tmp_path: Path) -> None:
    quest_root = tmp_path / "qf"
    quest_root.mkdir()
    (quest_root / "quest_failed.md").write_text(
        "# Quest failed before producing a paper\n\n"
        "**Quest ID:** `1779268235-foo`\n"
        "**Topic:** something\n"
        "**Failing node:** `implement`\n\n"
        "## What broke\n\n"
        "```\n"
        "_CliTransientError: claude exceeded 300s wall-clock and was killed\n"
        "```\n",
        encoding="utf-8",
    )
    info = _read_quest_failed_md(quest_root)
    assert info is not None
    assert info["present"] is True
    assert info["failing_node"] == "implement"
    assert "_CliTransientError" in info["what_broke"]


def test_read_quest_failed_md_returns_none_when_absent(tmp_path: Path) -> None:
    assert _read_quest_failed_md(tmp_path) is None


# ---------------------------------------------------------------------------
# /api/quests/{id} integration — surfaces the new fields
# ---------------------------------------------------------------------------


def _mk_quest_with_log(root: Path, qid: str, *, lines: list[str]) -> Path:
    q = root / qid
    (q / ".fi").mkdir(parents=True)
    (q / "paper").mkdir()
    (q / "figures").mkdir()
    (q / ".fi" / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return q


def test_quest_detail_includes_node_elapsed_and_idle(tmp_path: Path) -> None:
    """End-to-end: writing a run.log with timestamped node tags must
    make the /api/quests/{id} endpoint surface node_elapsed_s and
    node_idle_s for the dashboard badge."""
    now = time.time()
    started = now - 15
    last_line = now - 3
    lines = [
        f"{datetime.fromtimestamp(started).isoformat(sep=' ', timespec='milliseconds').replace('.', ',')} [INFO] [implement] generating experiment code",
        f"{datetime.fromtimestamp(last_line).isoformat(sep=' ', timespec='milliseconds').replace('.', ',')} [INFO] [implement] still waiting on LLM",
    ]
    _mk_quest_with_log(tmp_path, "qe", lines=lines)
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/quests/qe")
    assert r.status_code == 200
    body = r.json()
    assert body["current_node"] == "implement"
    assert body["node_started_at"] is not None
    assert body["node_elapsed_s"] is not None
    # Elapsed is the gap between started and "now" inside the server.
    # It SHOULD be roughly 15 s (give 10s of slack for test scheduling).
    assert 5 < body["node_elapsed_s"] < 30
    # Idle is the gap between the last line and "now" inside the
    # server. Should be roughly 3 s.
    assert 0 < body["node_idle_s"] < 30


def test_quest_detail_surfaces_quest_failed_md(tmp_path: Path) -> None:
    """When the engine wrote quest_failed.md, the detail endpoint must
    expose it so the dashboard banner can render."""
    q = _mk_quest_with_log(
        tmp_path, "qf",
        lines=["2026-05-21 00:00:00,000 [INFO] [implement] generating experiment code"],
    )
    (q / "quest_failed.md").write_text(
        "**Failing node:** `implement`\n\n"
        "## What broke\n\n```\n_CliTransientError: died\n```\n",
        encoding="utf-8",
    )
    app = make_app(tmp_path)
    client = TestClient(app)
    body = client.get("/api/quests/qf").json()
    assert body["quest_failed"] is not None
    assert body["quest_failed"]["present"] is True
    assert body["quest_failed"]["failing_node"] == "implement"
    assert "_CliTransientError" in body["quest_failed"]["what_broke"]


def test_quest_detail_quest_failed_is_null_when_absent(tmp_path: Path) -> None:
    """The common case: no failure → null. Dashboard hides the banner."""
    _mk_quest_with_log(
        tmp_path, "qok",
        lines=["2026-05-21 00:00:00,000 [INFO] [write] starting paper draft"],
    )
    app = make_app(tmp_path)
    client = TestClient(app)
    body = client.get("/api/quests/qok").json()
    assert body["quest_failed"] is None
