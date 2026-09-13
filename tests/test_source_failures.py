"""A source that failed must be visible in the quest that lost it.

A real quest lost every arXiv and OpenAlex result (HTTP 429 and 400) and its
run.log said nothing: the adapters logged at INFO on a logger with no handler.
The only symptom was a bibliography built from whatever was left. These pin
the ledger, the per-quest isolation, and the two places it must surface — the
live run.log line and the end-of-run summary written on every exit path.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

import core.knowledge as kn
from core import source_failures as sf
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    ProviderConfig,
)
from core.engine import Engine, _close_quest_logger, _quest_logger

_QUESTS = ("q-a", "q-b", "q-log")


@pytest.fixture(autouse=True)
def _clean_ledgers():
    yield
    for q in _QUESTS:
        sf.reset(q)


def _in_quest(qid: str, fn, *args, **kwargs):
    token = sf.current_quest.set(qid)
    try:
        return fn(*args, **kwargs)
    finally:
        sf.current_quest.reset(token)


# --- the ledger ---------------------------------------------------------------

def test_outside_a_quest_nothing_is_ledgered() -> None:
    sf.record_failure("openalex", "http_429", status=429)
    assert sf.snapshot("q-a")["total"] == 0


def test_counts_group_by_source_and_kind() -> None:
    def work() -> None:
        for _ in range(3):
            sf.record_failure("openalex", "http_429", status=429)
        sf.record_failure("web_page", "timeout")

    _in_quest("q-a", work)
    snap = sf.snapshot("q-a")
    assert snap["total"] == 4
    assert snap["by_source"] == {"openalex": {"http_429": 3}, "web_page": {"timeout": 1}}
    assert sf.format_summary(snap) == "openalex 3 (http_429=3); web_page 1 (timeout=1)"


def test_a_clean_run_says_none() -> None:
    assert sf.format_summary(sf.snapshot("q-a")) == "none"


def test_not_found_is_an_answer_not_a_failure() -> None:
    """The open-access cascade asks PMC about papers PMC does not have.
    Counting each 404 would bury the real outages."""
    class _R:
        status_code = 404

    _in_quest("q-a", sf.record_response, "pmc", _R())
    assert sf.snapshot("q-a")["total"] == 0


def test_rate_limit_response_is_counted() -> None:
    class _R:
        status_code = 429
        url = "https://arxiv.org/pdf/2401.00001"

    _in_quest("q-a", sf.record_response, "arxiv", _R())
    assert sf.snapshot("q-a")["by_source"] == {"arxiv": {"http_429": 1}}


def test_test_doubles_without_an_int_status_are_ignored() -> None:
    _in_quest("q-a", sf.record_response, "x", MagicMock())
    assert sf.snapshot("q-a")["total"] == 0


def test_arxiv_hosts_are_attributed_to_arxiv() -> None:
    assert sf.source_for_url("https://arxiv.org/pdf/1", "oa_copy") == "arxiv"
    assert sf.source_for_url("https://export.arxiv.org/abs/1", "oa_copy") == "arxiv"
    assert sf.source_for_url("https://eprints.example.ac.uk/1.pdf", "oa_copy") == "oa_copy"


def test_credentials_are_redacted() -> None:
    text = "Client error '401' for url 'https://api.openalex.org/works?search=x&api_key=SECRET123'"
    assert "SECRET123" not in sf.redact(text)
    assert "email=***" in sf.redact("https://api.unpaywall.org/v2/10.1/x?email=me@example.com")


# --- a real recording site ------------------------------------------------------

def test_http_helper_records_a_rate_limit_against_its_source(monkeypatch) -> None:
    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, params=None, **_):
            return httpx.Response(429, request=httpx.Request("GET", url, params=params))

    monkeypatch.setattr("core.knowledge.httpx.Client", _Client)
    out = _in_quest(
        "q-a", kn._http_get_json, "https://api.openalex.org/works",
        {"search": "x", "api_key": "SECRET123"}, 5.0, source="openalex",
    )
    assert out is None
    snap = sf.snapshot("q-a")
    assert snap["by_source"] == {"openalex": {"http_429": 1}}
    assert "SECRET123" not in json.dumps(snap), "the key rode in the exception text"


def test_quests_sharing_a_process_keep_separate_ledgers() -> None:
    """--fleet runs several quests in one process; worker threads must still
    charge the quest that started them."""
    async def quest(qid: str, n: int) -> None:
        sf.current_quest.set(qid)
        for _ in range(n):
            await asyncio.to_thread(sf.record_failure, "crossref", "timeout")

    async def main() -> None:
        await asyncio.gather(
            asyncio.create_task(quest("q-a", 2)), asyncio.create_task(quest("q-b", 5)),
        )

    asyncio.run(main())
    assert sf.snapshot("q-a")["total"] == 2
    assert sf.snapshot("q-b")["total"] == 5


# --- where it surfaces -----------------------------------------------------------

def test_failure_line_lands_in_the_quests_run_log(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    _quest_logger("q-log", fi)
    try:
        _in_quest(
            "q-log", sf.record_failure, "arxiv", "http_429", status=429,
            url="https://export.arxiv.org/api/query",
        )
    finally:
        _close_quest_logger("q-log")
    text = (fi / "run.log").read_text(encoding="utf-8")
    assert "[source-failure] source=arxiv kind=http_429 status=429 host=export.arxiv.org" in text


def _engine(tmp_path: Path) -> Engine:
    cfg = Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=["paper_md"]),
    )
    return Engine(cfg)


def test_summary_is_written_even_when_the_run_fails(tmp_path: Path) -> None:
    eng = _engine(tmp_path)

    async def failing_setup(_root):
        sf.record_failure("openalex", "http_429", status=429)
        raise RuntimeError("setup failed")

    eng.executor.setup = failing_setup  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="setup failed"):
            asyncio.run(eng.run())
        data = json.loads((eng.fi_dir / "source_failures.json").read_text("utf-8"))
        assert data["by_source"] == {"openalex": {"http_429": 1}}
        assert data["summary"] == "openalex 1 (http_429=1)"
        log = (eng.fi_dir / "run.log").read_text("utf-8")
        assert "[source-failure] source=openalex kind=http_429" in log
        assert "[source-failures] openalex 1 (http_429=1)" in log
    finally:
        sf.reset(eng.quest_id)


def test_a_clean_run_writes_none_rather_than_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    try:
        _in_quest(eng.quest_id, eng._emit_source_failure_summary, started_at=0.0)
        data = json.loads((eng.fi_dir / "source_failures.json").read_text("utf-8"))
        assert data["total"] == 0 and data["summary"] == "none"
        assert "[source-failures] none" in (eng.fi_dir / "run.log").read_text("utf-8")
    finally:
        _close_quest_logger(eng.quest_id)
        sf.reset(eng.quest_id)
