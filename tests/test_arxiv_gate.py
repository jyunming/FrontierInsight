"""The arXiv queue: pacing, one connection, shared backoff, per-quest pause, cache.

arXiv asks for at most one request every three seconds over one connection and
answers 429 otherwise. FI used to open up to twenty arxiv.org connections at
once, answer a refusal with an immediate request to another arXiv URL, and
cache nothing. The maintainer chose: every arXiv connection queued; on 429 the
whole queue waits 1, 2, 4 minutes; still refused -> no more arXiv for that
quest; successful responses cached for 24 hours.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
import pytest

from core import arxiv_gate as gate
from core import source_failures as sf


class _Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def mono(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


@pytest.fixture
def clock(monkeypatch, tmp_path: Path) -> _Clock:
    c = _Clock()
    monkeypatch.setenv("FI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("FI_ARXIV_CACHE", raising=False)
    monkeypatch.setattr(gate, "_now", c.now)
    monkeypatch.setattr(gate, "_mono", c.mono)
    monkeypatch.setattr(gate, "_sleep", c.sleep)
    monkeypatch.setattr(gate, "MIN_INTERVAL_S", 3.0)
    gate._PAUSED_QUESTS.clear()
    yield c
    gate._PAUSED_QUESTS.clear()


def _resp(status: int, body: bytes = b"<html>paper text</html>", ctype: str = "text/html") -> httpx.Response:
    return httpx.Response(status, content=body, headers={"content-type": ctype},
                          request=httpx.Request("GET", "https://arxiv.org/x"))


def _in_quest(qid: str, fn, *a, **kw):
    token = sf.current_quest.set(qid)
    try:
        return fn(*a, **kw)
    finally:
        sf.current_quest.reset(token)
        sf.reset(qid)


# --- pacing -------------------------------------------------------------------

def test_requests_are_at_least_three_seconds_apart(clock: _Clock) -> None:
    sent: list[float] = []

    def send():
        sent.append(clock.t)
        return _resp(404)  # not cached, not a failure

    gate.request("https://arxiv.org/abs/2401.00001", send)
    gate.request("https://arxiv.org/abs/2401.00002", send)
    assert len(sent) == 2
    assert sent[1] - sent[0] >= 3.0


def test_one_connection_at_a_time(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(gate, "MIN_INTERVAL_S", 0.0)
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def send():
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.03)
        with lock:
            active["now"] -= 1
        return _resp(404)

    threads = [threading.Thread(target=gate.request, args=(f"https://arxiv.org/abs/2401.0000{i}", send))
               for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert active["max"] == 1


def test_non_arxiv_urls_are_not_queued(clock: _Clock) -> None:
    calls = []
    out = gate.request("https://eprints.example.ac.uk/1.pdf", lambda: calls.append(1) or _resp(200))
    assert calls == [1] and out.status_code == 200
    assert not (Path(gate._cache_root()) / "gate_state.json").exists()


# --- backoff and pause ----------------------------------------------------------

def test_rate_limit_waits_one_two_four_minutes_then_pauses_the_quest(clock: _Clock) -> None:
    calls = []

    def send():
        calls.append(clock.t)
        return _resp(429, b"Rate exceeded.")

    out = _in_quest("q-a", gate.request, "https://arxiv.org/pdf/2401.00001", send)
    assert out.status_code == 429
    assert len(calls) == 4, "first try plus three retries"
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert gaps[0] >= 60 and gaps[1] >= 120 and gaps[2] >= 240
    assert gaps[0] < 120 and gaps[1] < 240, "the schedule is 1, 2, 4 minutes, not longer"

    token = sf.current_quest.set("q-a")
    try:
        assert gate.request("https://arxiv.org/pdf/2401.00002", send) is None
        assert len(calls) == 4, "a paused quest sends nothing more to arXiv"
        snap = sf.snapshot("q-a")
    finally:
        sf.current_quest.reset(token)
        sf.reset("q-a")
    assert "arxiv_paused" in snap["by_source"]["arxiv"] or "paused_skip" in snap["by_source"]["arxiv"]


def test_the_pause_belongs_to_the_quest_that_hit_it(clock: _Clock) -> None:
    gate._PAUSED_QUESTS.add("q-a")
    calls = []
    out = _in_quest("q-b", gate.request, "https://arxiv.org/abs/2401.00003",
                    lambda: calls.append(1) or _resp(404))
    assert calls == [1] and out.status_code == 404
    gate.reset_quest("q-a")
    assert "q-a" not in gate._PAUSED_QUESTS


def test_a_success_resets_the_backoff(clock: _Clock) -> None:
    replies = iter([_resp(429), _resp(200)])
    out = gate.request("https://arxiv.org/html/2401.00004", lambda: next(replies))
    assert out.status_code == 200
    state = gate._read_state()
    assert state["consecutive_429"] == 0 and state["blocked_until"] == 0.0


def test_a_deadline_skips_instead_of_sleeping(clock: _Clock) -> None:
    """Enrichment has a 90 s budget; a 4-minute backoff must not park a
    worker thread that other sources share."""
    state = gate._read_state()
    state["blocked_until"] = clock.t + 240
    gate._write_state(state)
    calls = []
    token = gate.fetch_deadline.set(clock.t + 90)
    try:
        out = _in_quest("q-c", gate.request, "https://arxiv.org/pdf/2401.00005",
                        lambda: calls.append(1) or _resp(200))
    finally:
        gate.fetch_deadline.reset(token)
    assert out is None and calls == [] and clock.sleeps == []


def test_backoff_state_is_shared_through_the_state_file(clock: _Clock) -> None:
    """Another process's 429 (written to the shared state) holds this one too."""
    state = gate._read_state()
    state["blocked_until"] = clock.t + 60
    gate._write_state(state)
    sent = []
    gate.request("https://arxiv.org/abs/2401.00006", lambda: sent.append(clock.t) or _resp(404))
    assert sent and sent[0] >= 1_000_060


# --- cache ----------------------------------------------------------------------

def test_a_cached_response_needs_no_request(clock: _Clock) -> None:
    calls = []
    send = lambda: calls.append(1) or _resp(200, b"%PDF-1.7 real paper", "application/pdf")  # noqa: E731
    first = gate.request("https://arxiv.org/pdf/2401.01234v2", send)
    second = gate.request("https://arxiv.org/pdf/2401.01234", send)
    assert calls == [1], "same paper, version suffix ignored"
    assert second.content == first.content == b"%PDF-1.7 real paper"
    assert second.headers["content-type"] == "application/pdf"


def test_the_cache_expires_after_a_day(clock: _Clock) -> None:
    calls = []
    send = lambda: calls.append(1) or _resp(200)  # noqa: E731
    gate.request("https://arxiv.org/abs/2401.07777", send)
    clock.t += 24 * 3600 + 1
    gate.request("https://arxiv.org/abs/2401.07777", send)
    assert calls == [1, 1]


@pytest.mark.parametrize("reply", [
    _resp(429, b"Rate exceeded."),
    _resp(200, b"<html>Just a moment... checking your browser</html>"),
])
def test_refusals_and_challenge_pages_are_never_cached(clock: _Clock, reply) -> None:
    gate.cache_put("https://arxiv.org/abs/2401.08888", reply)
    assert gate.cache_get("https://arxiv.org/abs/2401.08888") is None


def test_cache_can_be_disabled(clock: _Clock, monkeypatch) -> None:
    monkeypatch.setenv("FI_ARXIV_CACHE", "0")
    gate.cache_put("https://arxiv.org/abs/2401.09999", _resp(200))
    assert gate.cache_get("https://arxiv.org/abs/2401.09999") is None


def test_eviction_keeps_the_cache_under_its_cap(clock: _Clock, monkeypatch) -> None:
    monkeypatch.setattr(gate, "MAX_CACHE_BYTES", 25)
    for i in range(4):
        gate.cache_put(f"https://arxiv.org/abs/2401.1000{i}", _resp(200, b"0123456789"))
    sizes = sum(p.stat().st_size for p in (gate._cache_root() / "responses").glob("*.bin"))
    assert sizes <= 25


@pytest.mark.parametrize("a,b", [
    ("http://www.arxiv.org/abs/2401.01234v3/", "https://arxiv.org/abs/2401.01234"),
    ("https://arxiv.org/pdf/2401.01234v2.pdf", "https://arxiv.org/pdf/2401.01234"),
    ("https://export.arxiv.org/abs/2401.01234", "https://export.arxiv.org/abs/2401.01234"),
])
def test_url_normalization(a: str, b: str) -> None:
    assert gate.normalize_url(a) == gate.normalize_url(b)


def test_arxiv_host_detection() -> None:
    assert gate.is_arxiv_url("https://arxiv.org/pdf/1")
    assert gate.is_arxiv_url("https://export.arxiv.org/abs/1")
    assert not gate.is_arxiv_url("https://notarxiv.org/abs/1")
    assert not gate.is_arxiv_url("https://api.openalex.org/works")
