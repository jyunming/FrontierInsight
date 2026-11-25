"""The knowledge layer's search options: given sources skip routing, web search and full text can be left out, and
DOAJ's rate limit is kept. (Kept from the topic-search tests; these options are still used by the cross-check, the
plan's criteria and the step passages.)"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import core.knowledge as kn
from core.config import KnowledgeConfig
from core.knowledge import Knowledge, RetrievedDoc


def _doc(title: str, doi: str) -> RetrievedDoc:
    return RetrievedDoc(content=f"{title}\n\nAn abstract.",
                        metadata={"title": title, "doi": doi, "source": "crossref"})


# --- the knowledge layer --------------------------------------------------------

def _knowledge(**over) -> Knowledge:
    cfg = KnowledgeConfig(**{"enabled": False, "source_routing": "manual",
                             "external_fallback": ["crossref"], "web_search": True, **over})
    k = Knowledge(cfg)
    k.cfg.enabled = True
    return k


def test_given_sources_are_used_without_routing_again(monkeypatch) -> None:
    async def must_not_route(**kw):
        raise AssertionError("routed again")

    got: dict = {}

    async def fake_router(query, top_k, sources, **kw):
        got["sources"] = sources
        return []

    monkeypatch.setattr(kn, "_route_sources_with_llm", must_not_route)
    monkeypatch.setattr(kn, "_route_external", fake_router)
    monkeypatch.setattr(kn, "_web_search", lambda *a, **k: [])
    k = _knowledge(source_routing="auto")
    asyncio.run(k.asearch("q", sources=["doaj"], chat_fn=AsyncMock()))
    assert got["sources"] == ["doaj"]


def test_web_false_skips_web_search(monkeypatch) -> None:
    ran: list = []

    async def fake_router(query, top_k, sources, **kw):
        return []

    monkeypatch.setattr(kn, "_web_search", lambda *a, **k: ran.append(1) or [])
    monkeypatch.setattr(kn, "_route_external", fake_router)
    k = _knowledge()
    asyncio.run(k.asearch("q", web=False))
    assert ran == []
    asyncio.run(k.asearch("q"))
    assert ran == [1]


def test_a_caller_can_defer_full_text(monkeypatch) -> None:
    enriched: list = []

    async def fake_enrich(docs, **kw):
        enriched.append(len(docs))
        return docs

    async def fake_router(query, top_k, sources, **kw):
        return [_doc("A paper about bilingual children", "10.1/p")]

    monkeypatch.setattr(kn, "_enrich_with_full_text", fake_enrich)
    monkeypatch.setattr(kn, "_route_external", fake_router)
    k = _knowledge(web_search=False, try_fetch_full_text=True)
    asyncio.run(k.asearch("q", fetch_full_text=False))
    assert enriched == []
    asyncio.run(k.asearch("q"))
    assert enriched == [1]


def test_choose_sources_spends_no_call_unless_routing_can_run(monkeypatch) -> None:
    calls: list = []

    async def router(**kw):
        calls.append(kw["topic"])
        return ["doaj"]

    monkeypatch.setattr(kn, "_route_sources_with_llm", router)
    chat = AsyncMock()
    k = _knowledge()
    assert asyncio.run(k.choose_sources("q", chat_fn=chat)) == ["crossref"], "manual routing"
    k.cfg.source_routing = "auto"
    assert asyncio.run(k.choose_sources("q")) == ["crossref"], "no chat function"
    k.cfg.enabled = False
    assert asyncio.run(k.choose_sources("q", chat_fn=chat)) == ["crossref"], "retrieval off"
    assert calls == []
    k.cfg.enabled = True
    assert asyncio.run(k.choose_sources("q", chat_fn=chat)) == ["doaj"]
    assert calls == ["q"]


def test_fetch_full_text_enriches_scholarly_records_in_place(monkeypatch) -> None:
    async def fake_enrich(docs, **kw):
        return [RetrievedDoc(content=d.content + " FULL",
                             metadata={**d.metadata, "fetched_full_text": True}) for d in docs]

    monkeypatch.setattr(kn, "_enrich_with_full_text", fake_enrich)
    page = RetrievedDoc(content="page", metadata={"source": "web_search", "url": "https://example.org"})
    paper = _doc("A paper about bilingual children", "10.1/p")
    done = RetrievedDoc(content="already", metadata={"source": "core", "fetched_full_text": True})
    k = _knowledge(try_fetch_full_text=True)
    out = asyncio.run(k.fetch_full_text([page, paper, done]))
    assert [d.content for d in out] == ["page", paper.content + " FULL", "already"]
    k.cfg.try_fetch_full_text = False
    assert asyncio.run(k.fetch_full_text([paper]))[0].content == paper.content


# --- DOAJ's rate limit --------------------------------------------------------

def test_doaj_requests_are_spaced_to_its_rate_limit(monkeypatch) -> None:
    clock = [100.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        clock[0] += s

    monkeypatch.setattr(kn, "_doaj_monotonic", lambda: clock[0])
    monkeypatch.setattr(kn, "_doaj_sleep", sleep)
    monkeypatch.setattr(kn, "_DOAJ_LAST_REQUEST", [0.0])
    kn._doaj_pace()
    kn._doaj_pace()
    assert slept == [pytest.approx(kn._DOAJ_MIN_INTERVAL_S)]


def test_every_doaj_request_waits_its_turn(monkeypatch) -> None:
    paced: list = []
    monkeypatch.setattr(kn, "_doaj_pace", lambda: paced.append(1))
    monkeypatch.setattr(kn, "_http_get_json", lambda *a, **k: {"results": []})
    kn._doaj_search("one two three four", 5)
    assert len(paced) == 2, "the query and its shorter retry each take a turn"
