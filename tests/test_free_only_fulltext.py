"""FI downloads only papers that are free to read; a paywalled paper is left for a person.

The full-text step used to GET whatever pdf_url or landing page a search hit carried, relying on the host's own
network access (a VPN or campus login) to get past a paywall. Now a hit is fetched from its own address only when it
is confirmed free (``is_open_access``). Anything else may be recovered only through the free-copy cascade (Unpaywall,
Europe PMC, Semantic Scholar, CORE); if that finds nothing the paper goes on ``needs/WANTED_PAPERS.md``. No network
and no model is used here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

import core.knowledge as kn
from core.knowledge import RetrievedDoc, is_open_access
from tests.test_literature_facets import _engine, _state

PUBLISHER_PDF = "https://publisher.example/paywalled.pdf"
PUBLISHER_PAGE = "https://publisher.example/article/123"
FREE_TEXT = "Free full text of the paper. " * 30


class _SpyClient:
    """Stands in for httpx.Client; records every URL asked for and answers with a real-looking PDF or a page."""

    urls: list[str] = []

    def __init__(self, *a, **k) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kw):
        _SpyClient.urls.append(url)
        raise AssertionError(f"unexpected GET {url}")


@pytest.fixture()
def spy(monkeypatch):
    _SpyClient.urls = []
    monkeypatch.setattr(kn.httpx, "Client", _SpyClient)
    return _SpyClient


def _paywalled() -> RetrievedDoc:
    return RetrievedDoc(content="abstract", metadata={
        "source": "crossref", "title": "A paywalled paper", "doi": "10.1109/x.1",
        "pdf_url": PUBLISHER_PDF, "url": PUBLISHER_PAGE,
    })


def test_a_paywalled_hit_is_never_requested_from_the_publisher(monkeypatch, spy) -> None:
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    assert kn._fetch_full_text(_paywalled(), timeout_s=5, max_kb=64) is None
    assert spy.urls == []


def test_a_paywalled_page_is_neither_fetched_nor_rendered(monkeypatch, spy) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    monkeypatch.setattr(kn, "_playwright_fetch_html", lambda url, **k: rendered.append(url) or "<html>x</html>")
    doc = RetrievedDoc(content="snippet", metadata={"url": "https://link.springer.com/article/10.1007/x"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=True) is None
    assert spy.urls == [] and rendered == []


def test_a_confirmed_free_hit_is_still_fetched(monkeypatch, spy) -> None:
    seen: list[str] = []
    monkeypatch.setattr(kn, "_fetch_pdf_bytes", lambda url, *, timeout_s: seen.append(url) or b"%PDF-1.4 x")
    monkeypatch.setattr(kn, "_pdf_bytes_to_text", lambda body, *, cap: FREE_TEXT)
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    doc = RetrievedDoc(content="abstract", metadata={"arxiv_id": "2401.00001", "pdf_url": "https://arxiv.org/pdf/2401.00001"})
    assert kn._fetch_full_text(doc, timeout_s=5, max_kb=64) == FREE_TEXT
    assert seen == ["https://arxiv.org/pdf/2401.00001"], "an arXiv paper is free, so its own address is still used"


@pytest.mark.parametrize("md,free", [
    ({"open_access": True}, True),
    ({"arxiv_id": "2401.1"}, True),
    ({"pmcid": "PMC1"}, True),
    ({"doi": "10.1101/2020.01.01.1"}, True),
    ({"url": "https://www.medrxiv.org/content/x"}, True),
    ({}, False),
    ({"open_access": None, "doi": "10.1016/j.x"}, False),
    ({"url": "https://ieeexplore.ieee.org/document/1"}, False),
    (None, False),
])
def test_only_confirmed_free_sources_count_as_free(md, free) -> None:
    assert is_open_access(md) is free


def test_an_unknown_hit_is_confirmed_through_the_free_copy_cascade_then_fetched(monkeypatch, spy) -> None:
    asked: list[str] = []

    def cascade(doc, *, timeout_s, cap):
        asked.append(doc.metadata["doi"])
        return FREE_TEXT  # Unpaywall found a free copy

    monkeypatch.setattr(kn, "_fetch_via_open_apis", cascade)
    assert kn._fetch_full_text(_paywalled(), timeout_s=5, max_kb=64) == FREE_TEXT
    assert asked == ["10.1109/x.1"] and spy.urls == []


def test_an_unpaywall_failure_leaves_the_paper_for_a_person(monkeypatch) -> None:
    import httpx

    asked: list[str] = []

    class _Down(_SpyClient):
        def get(self, url, **kw):
            asked.append(url)
            raise httpx.ConnectError("unpaywall is down")

    monkeypatch.setattr(kn.httpx, "Client", _Down)
    monkeypatch.setattr(kn, "_resolve_ids", lambda doc, timeout_s: {"doi": "10.1109/x.1", "pmcid": "", "arxiv_id": ""})
    assert kn._fetch_full_text(_paywalled(), timeout_s=5, max_kb=64) is None
    assert asked, "the free-copy lookup was tried"
    assert not [u for u in asked if "publisher.example" in u], asked


# --- the papers gate in the literature step -------------------------------------------------------------------

def _gate_engine(tmp_path: Path, monkeypatch, docs, *, pauses_papers: bool):
    eng = _engine(tmp_path)
    eng.config.pauses.papers = pauses_papers
    eng.config.knowledge.foundational_works = False

    async def chat(prompt, node=""):
        return '{"queries": ["a query"]}'

    eng._chat = chat

    async def hits(query, **kw):
        return list(docs)

    eng.knowledge.asearch = hits  # type: ignore[method-assign]

    async def route(*a, **k):
        return ["crossref"]

    eng.knowledge.choose_sources = route  # type: ignore[method-assign]

    async def no_fetch(found):
        return found

    eng.knowledge.fetch_full_text = no_fetch  # type: ignore[method-assign]
    paused: list[dict] = []
    monkeypatch.setattr(eng, "_pause_for_human", lambda **kw: paused.append(kw))
    lines: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    eng._log.addHandler(_H())
    eng._log.setLevel(logging.INFO)
    return eng, paused, lines


@pytest.mark.asyncio
async def test_an_unattended_run_logs_one_count_and_lists_the_papers(tmp_path: Path, monkeypatch) -> None:
    eng, paused, lines = _gate_engine(tmp_path, monkeypatch, [_paywalled()], pauses_papers=False)
    await eng._node_literature(_state())
    assert paused == []
    counted = [line for line in lines if "behind a paywall" in line]
    assert len(counted) == 1 and counted[0].startswith("[literature] 1 paper(s)")
    wanted = (eng.quest_root / "needs" / "WANTED_PAPERS.md").read_text(encoding="utf-8")
    assert "A paywalled paper" in wanted


@pytest.mark.asyncio
async def test_an_attended_run_pauses_for_the_paywalled_paper(tmp_path: Path, monkeypatch) -> None:
    eng, paused, lines = _gate_engine(tmp_path, monkeypatch, [_paywalled()], pauses_papers=True)
    await eng._node_literature(_state())
    assert len(paused) == 1 and paused[0]["kind"] == "papers"
    assert not [line for line in lines if "behind a paywall" in line]
    assert (eng.quest_root / "needs" / "WANTED_PAPERS.md").is_file()
