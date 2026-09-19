"""Background jobs: the contract between a generated experiment script and FI
(core/job_watch.py) and the ``--watch`` loop that re-checks a paused quest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import job_watch as jw


def test_only_the_fi_job_key_marks_a_job() -> None:
    """A normal result with a field called ``pending`` is not a job."""
    assert jw.job_of(None) is None
    assert jw.job_of({"pending": True, "score": 1}) is None
    assert jw.job_of({"fi_job": "pending"}) is None
    job = jw.job_of({"fi_job": {"status": "Pending", "id": "42", "poll_s": 600}})
    assert job == {"status": "pending", "id": "42", "poll_s": 600}
    assert jw.job_of({"fi_job": {"id": "42"}})["status"] == "pending", "no status means pending"


def test_poll_seconds_takes_the_jobs_suggestion_within_limits() -> None:
    assert jw.poll_seconds({"poll_s": 900}) == 900
    assert jw.poll_seconds({"poll_s": 1}) == jw.MIN_POLL_S
    assert jw.poll_seconds({"poll_s": "soon"}) == jw.DEFAULT_POLL_S
    assert jw.poll_seconds(None) == jw.DEFAULT_POLL_S


def test_pending_file_remembers_when_the_wait_began(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    assert jw.read_pending(fi) is None
    first = jw.write_pending(fi, {"status": "pending", "id": "7"}, "code/experiment.py")
    second = jw.write_pending(fi, {"status": "pending", "id": "7", "note": "RUNNING"}, "code/experiment.py")
    assert second["submitted_at"] == first["submitted_at"]
    assert (first["checks"], second["checks"]) == (1, 2)
    assert jw.read_pending(fi)["job"]["note"] == "RUNNING"
    assert "id 7" in jw.describe(second) and "RUNNING" in jw.describe(second)
    assert jw.clear_pending(fi) is True and jw.read_pending(fi) is None
    assert jw.clear_pending(fi) is False


def _poller(states: list[str]):
    calls = {"n": 0}

    async def poll():
        state = states[min(calls["n"], len(states) - 1)]
        calls["n"] += 1
        return state, {"job": {"note": f"check {calls['n']}"}}

    return poll, calls


@pytest.fixture()
def no_sleep(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(jw.asyncio, "sleep", fake_sleep)
    return slept


@pytest.mark.asyncio
async def test_watch_checks_until_the_job_is_no_longer_pending(tmp_path: Path, no_sleep) -> None:
    fi = tmp_path / ".fi"
    jw.write_pending(fi, {"status": "pending", "poll_s": 60}, "code/experiment.py")
    poll, calls = _poller([jw.PENDING, jw.PENDING, jw.DONE])
    lines: list[str] = []

    outcome = await jw.watch(poll, fi_dir=fi, every_s=0, max_wait_s=0, out=lines.append)

    assert outcome == jw.DONE and calls["n"] == 3
    assert no_sleep == [60, 60], "waits the job's own poll_s between checks"
    assert len(lines) == 3, "every check is reported"
    assert "check 1: pending - check 1" in lines[0] and "check 3: done" in lines[2]


@pytest.mark.asyncio
async def test_a_failed_job_ends_the_watch_too(tmp_path: Path, no_sleep) -> None:
    poll, _ = _poller([jw.FAILED])
    outcome = await jw.watch(poll, fi_dir=tmp_path, every_s=30, max_wait_s=0, out=lambda s: None)
    assert outcome == jw.FAILED


@pytest.mark.asyncio
async def test_watch_stops_after_the_time_limit_and_says_the_quest_is_still_paused(
    tmp_path: Path, no_sleep, monkeypatch,
) -> None:
    clock = iter([0.0, 5000.0, 10000.0, 20000.0])
    monkeypatch.setattr(jw, "_now", lambda: next(clock))
    poll, _ = _poller([jw.PENDING])
    lines: list[str] = []

    outcome = await jw.watch(poll, fi_dir=tmp_path, every_s=60, max_wait_s=3600, out=lines.append)

    assert outcome == "timeout"
    assert "still paused" in lines[-1]
