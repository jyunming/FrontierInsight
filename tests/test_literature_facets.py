"""Three facet queries per literature pass, one routing decision, full text once.

One keyword query reaches only the work written in its own words. The
literature node asks for three facets of the topic -- the core subject, a
specific angle, the wider frame -- searches each, and interleaves the results.
The facets share one source-routing decision (they are phrasings of one
topic), only the first also runs web search, and legal full text is fetched
once for the sources that survive the relevance screen instead of per facet.
The query wording follows the kind of quest: methods and quantities for a
quest with an experiment, the names scholars write under for one without.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import core.knowledge as kn
import core.passages as pmod
from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    PausesConfig,
    ProviderConfig,
)
from core.engine import Engine
from core.knowledge import (
    WORK_SCOPE_PAPERS,
    WORK_SCOPE_PAPERS_AND_BOOKS,
    Knowledge,
    RetrievedDoc,
)

TOPIC = "Does speaking two languages improve executive function?"
FACETS = [
    "bilingual advantage executive function",
    "inhibitory control bilingual children",
    "cognitive effects of bilingualism",
]


@pytest.fixture(autouse=True)
def _unscored(monkeypatch):
    # No embedding scores: the relevance floor keeps every doc in order.
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: None)


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic=TOPIC, title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False, web_search=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
        pauses=PausesConfig(papers=False),
    ))
    eng.config.knowledge.enabled = True
    eng.config.knowledge.source_routing = "manual"
    return eng


def _doc(title: str, doi: str) -> RetrievedDoc:
    return RetrievedDoc(content=f"{title}\n\nAn abstract.",
                        metadata={"title": title, "doi": doi, "source": "crossref"})


async def _facet_chat(prompt, node=""):
    assert node == "literature_query", node
    return json.dumps({"queries": FACETS})


def _state(**over) -> dict:
    return {"topic": TOPIC, "chosen_idea": {"title": "Bilingual advantage"}, **over}


# --- the literature node --------------------------------------------------------

@pytest.mark.asyncio
async def test_each_facet_is_searched_with_one_routing_decision_and_web_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._chat = _facet_chat
    route = AsyncMock(return_value=["crossref", "doaj"])
    eng.knowledge.choose_sources = route  # type: ignore[method-assign]
    calls: list[tuple] = []

    async def fake_search(query, **kw):  # noqa: ANN001
        calls.append((query, kw.get("sources"), kw.get("web"), kw.get("fetch_full_text")))
        return []

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    patch = await eng._node_literature(_state())
    assert route.await_count == 1, "the facets share one routing decision"
    assert sorted(c[0] for c in calls) == sorted(FACETS)
    assert all(c[1] == ["crossref", "doaj"] for c in calls)
    assert {c[0]: c[2] for c in calls} == {FACETS[0]: True, FACETS[1]: False, FACETS[2]: False}
    assert all(c[3] is False for c in calls), "full text is fetched once, after the screen"
    assert patch["literature_queries"] == FACETS
    assert patch["literature_query"] == FACETS[0]


@pytest.mark.asyncio
async def test_facet_results_are_interleaved_and_a_work_found_twice_is_kept_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._chat = _facet_chat
    by_query = {
        FACETS[0]: [_doc("Bilingual advantage revisited", "10.1/a1"),
                    _doc("Executive function in adults", "10.1/a2")],
        FACETS[1]: [_doc("Inhibition in bilingual children", "10.1/b1"),
                    _doc("Bilingual advantage revisited", "10.1/a1")],
        FACETS[2]: [_doc("Cognitive effects of speaking two languages", "10.1/c1")],
    }

    async def fake_search(query, **kw):  # noqa: ANN001
        return by_query[query]

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    patch = await eng._node_literature(_state())
    assert [e["metadata"]["doi"] for e in patch["literature"]] == [
        "10.1/a1", "10.1/b1", "10.1/c1", "10.1/a2"]


@pytest.mark.asyncio
async def test_a_requery_never_repeats_a_facet_and_keeps_the_quest_kind(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    eng._chat = _facet_chat

    async def fake_search(query, **kw):  # noqa: ANN001
        return [_doc("An unrelated study of rivers", "10.1/x")]

    def floor(topic, docs, *, stats=None):  # noqa: ANN001
        if stats is not None:
            stats.update({"scored": True, "above_floor": 0, "best": 0.05})
        return docs

    seen: dict = {}

    async def propose(topic, tried, missed, **kw):  # noqa: ANN001
        seen["tried"], seen["scope"] = list(tried), kw.get("work_scope")
        return ""

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    monkeypatch.setattr(eng, "_filter_docs_by_relevance", floor)
    monkeypatch.setattr(eng, "_propose_literature_queries", propose)
    await eng._node_literature(_state(no_simulation_resolved=True))
    assert seen["tried"] == FACETS
    assert seen["scope"] == WORK_SCOPE_PAPERS_AND_BOOKS


@pytest.mark.asyncio
async def test_full_text_is_fetched_once_for_the_sources_that_were_kept(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._chat = _facet_chat
    dois = {q: f"10.1/f{i}" for i, q in enumerate(FACETS)}
    shared = _doc("A paper every facet finds on bilingualism", "10.1/shared")

    async def fake_search(query, **kw):  # noqa: ANN001
        return [_doc(f"A study on {query}", dois[query]), shared]

    fetched: list[list[str]] = []

    async def fake_fetch(docs):  # noqa: ANN001
        fetched.append([d.metadata["doi"] for d in docs])
        return docs

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    eng.knowledge.fetch_full_text = fake_fetch  # type: ignore[method-assign]
    patch = await eng._node_literature(_state())
    assert len(fetched) == 1
    assert fetched[0] == [e["metadata"]["doi"] for e in patch["literature"]]


@pytest.mark.asyncio
async def test_a_failed_derivation_searches_the_old_query_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path)

    async def down(prompt, node=""):
        raise RuntimeError("provider down")

    sent: list = []

    async def fake_search(query, **kw):  # noqa: ANN001
        sent.append((query, kw.get("web")))
        return []

    eng._chat = down
    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    patch = await eng._node_literature(_state())
    assert sent == [(f"Bilingual advantage {TOPIC}", True)]
    assert patch["literature_queries"] == [sent[0][0]]


# --- deriving the facets ------------------------------------------------------

@pytest.mark.parametrize("scope,present,absent", [
    (WORK_SCOPE_PAPERS, ["methods", "arXiv"], ["periods", "OpenAIRE"]),
    (WORK_SCOPE_PAPERS_AND_BOOKS, ["periods", "OpenAIRE"], ["arXiv"]),
])
@pytest.mark.asyncio
async def test_query_wording_follows_the_kind_of_quest(tmp_path: Path, scope, present, absent) -> None:
    eng = _engine(tmp_path)
    seen: dict = {}

    async def chat(prompt, node=""):
        seen["prompt"] = prompt
        return json.dumps({"queries": FACETS})

    eng._chat = chat
    await eng._derive_literature_queries("a topic", work_scope=scope)
    for word in present:
        assert word in seen["prompt"]
    for word in absent:
        assert word not in seen["prompt"]


@pytest.mark.asyncio
async def test_derivation_drops_repeats_and_sentences_and_accepts_one_query(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    replies = iter([
        json.dumps({"queries": ["a b c", "A  B c", " ".join(["w"] * 25), "d e f", "g h i", "j k l"]}),
        json.dumps({"query": "single facet query"}),
    ])

    async def chat(prompt, node=""):
        return next(replies)

    eng._chat = chat
    assert await eng._derive_literature_queries("t") == ["a b c", "d e f", "g h i"]
    assert await eng._derive_literature_queries("t") == ["single facet query"]


@pytest.mark.asyncio
async def test_the_requery_prompt_follows_the_kind_of_quest(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    prompts: list[str] = []

    async def chat(prompt, node=""):
        prompts.append(prompt)
        return '{"query": "something new"}'

    eng._chat = chat
    await eng._propose_literature_queries("t", ["q"], [], work_scope=WORK_SCOPE_PAPERS_AND_BOOKS)
    await eng._propose_literature_queries("t", ["q"], [])
    assert "periods" in prompts[0] and "periods" not in prompts[1]


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
