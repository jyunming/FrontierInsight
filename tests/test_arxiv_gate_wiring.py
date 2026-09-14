"""Every path that talks to arXiv goes through the queue.

A queue only paces what passes through it. The audit found arXiv contacted
from the preprint full-text route, Unpaywall's repository copies, Semantic
Scholar's open-access PDF, web-page enrichment (plus an immediate headless
browser retry after a refusal), publisher-PDF fetches and figure fetching.
These pin the routes, the deadline enrichment sets, and the rule that the
headless retry waits its turn instead of bypassing a paused queue.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import httpx

import core.figure_sources as fs
import core.knowledge as kn
from core import arxiv_gate as gate
from core import source_failures as sf
from core.knowledge import RetrievedDoc


def _spy_gate(monkeypatch) -> list[str]:
    seen: list[str] = []
    real = gate.request

    def spy(url, send):
        seen.append(url)
        return real(url, send)

    monkeypatch.setattr(kn._gate, "request", spy)
    return seen


class _Client:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def get(self, url, *a, **k):
        return httpx.Response(404, request=httpx.Request("GET", url))


def test_preprint_full_text_routes_through_the_queue(monkeypatch) -> None:
    seen = _spy_gate(monkeypatch)
    monkeypatch.setattr("core.knowledge.httpx.Client", _Client)
    kn._preprint_fulltext({"arxiv_id": "2401.00001", "doi": ""}, timeout_s=5, cap=1000)
    assert seen == ["https://arxiv.org/html/2401.00001", "https://arxiv.org/pdf/2401.00001"]


def test_publisher_pdf_and_landing_fetches_route_through_the_queue(monkeypatch) -> None:
    seen = _spy_gate(monkeypatch)
    monkeypatch.setattr("core.knowledge.httpx.Client", _Client)
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    doc = RetrievedDoc(content="", metadata={"pdf_url": "https://arxiv.org/pdf/2401.00002",
                                             "url": "https://arxiv.org/abs/2401.00002"})
    kn._fetch_full_text(doc, timeout_s=5, max_kb=10)
    assert "https://arxiv.org/pdf/2401.00002" in seen
    assert "https://arxiv.org/abs/2401.00002" in seen


def test_headless_retry_does_not_bypass_a_paused_queue(monkeypatch) -> None:
    rendered = []
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    monkeypatch.setattr(kn, "_playwright_fetch_html",
                        lambda url, **k: rendered.append(url) or "<html>x</html>")
    monkeypatch.setattr("core.knowledge.httpx.Client", _Client)
    token = sf.current_quest.set("q-paused")
    gate._PAUSED_QUESTS.add("q-paused")
    try:
        doc = RetrievedDoc(content="snippet", metadata={"url": "https://arxiv.org/abs/2401.00003"})
        kn._fetch_web_page_text(doc, timeout_s=5, max_kb=10, headless=True)
    finally:
        sf.current_quest.reset(token)
        sf.reset("q-paused")
    assert rendered == [], "a paused arXiv must not be re-asked through a browser"


def test_headless_still_renders_non_arxiv_pages(monkeypatch) -> None:
    rendered = []
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    monkeypatch.setattr(kn, "_is_academic_source", lambda url: False)
    monkeypatch.setattr(kn, "_playwright_fetch_html", lambda url, **k: rendered.append(url) or None)
    monkeypatch.setattr("core.knowledge.httpx.Client", _Client)
    doc = RetrievedDoc(content="snippet", metadata={"url": "https://www.iea.org/report"})
    kn._fetch_web_page_text(doc, timeout_s=5, max_kb=10, headless=True)
    assert rendered == ["https://www.iea.org/report"]


def test_enrichment_gives_arxiv_waits_its_budget_as_a_deadline() -> None:
    seen = []

    def fetch(doc, *, timeout_s, max_kb):
        seen.append(gate.fetch_deadline.get())
        return None

    docs = [RetrievedDoc(content="", metadata={"url": "https://arxiv.org/abs/2401.00004"})]
    asyncio.run(kn._enrich_with_full_text(docs, timeout_s=5, total_budget_s=90, max_kb=10, fetch_fn=fetch))
    assert seen and seen[0] is not None
    assert gate.fetch_deadline.get() is None, "the deadline must not leak past the batch"


def test_arxiv_figure_fetches_route_through_the_queue(monkeypatch, tmp_path) -> None:
    seen: list[str] = []
    real = gate.request
    monkeypatch.setattr(fs._gate, "request", lambda url, send: seen.append(url) or real(url, send))

    class _FigClient(_Client):
        def get(self, url, *a, **k):
            r = MagicMock()
            r.status_code = 404
            r.text = ""
            return r

    monkeypatch.setattr("core.figure_sources.httpx.Client", _FigClient)
    assert fs.fetch_arxiv_figure("2401.00005", "q", out_dir=tmp_path) is None
    assert seen == ["https://arxiv.org/abs/2401.00005"]
