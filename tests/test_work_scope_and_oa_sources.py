"""Which scholarly record types a quest keeps, and the keyless open-access sources.

A scholarly index holds datasets, figure components, peer-review reports and
dictionary entries next to papers, and none of those is a source a claim can
cite. Which types a quest keeps follows its run mode: a quest with an
experiment keeps journal articles, conference papers and preprints; a quest
without one also keeps books and book chapters, because much humanities and
social-science scholarship is published there (a live Crossref check on a
popular-culture topic: 6 of its 10 on-topic hits were book chapters).

CORE answers searches without a key, so it no longer needs one to run.
OpenAIRE and DOAJ are keyless open-access sources that cover those fields.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import unquote

import pytest

import core.knowledge as kn
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


class _Fake:
    """Fake httpx.Client: records each GET; ``respond(url, params)`` is the JSON body."""

    calls: list[dict] = []

    def __init__(self, respond) -> None:
        self.respond = respond
        _Fake.calls = []

    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def get(self, url, params=None, headers=None, **_):
        _Fake.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        r = MagicMock()
        r.json.return_value = self.respond(url, dict(params or {}))
        r.raise_for_status = MagicMock()
        return r


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for var in ("OPENALEX_API_KEY", "CORE_API_KEY"):
        monkeypatch.setenv(var, "")


# --- type filters ---------------------------------------------------------------

@pytest.mark.parametrize("scope,books_kept", [
    (WORK_SCOPE_PAPERS, False),
    (WORK_SCOPE_PAPERS_AND_BOOKS, True),
])
def test_crossref_type_filter_follows_the_scope(monkeypatch, scope, books_kept) -> None:
    monkeypatch.setattr("core.knowledge.httpx.Client", _Fake(lambda u, p: {"message": {"items": []}}))
    kn._crossref_search("toys popular culture", 5, scope=scope)
    f = _Fake.calls[0]["params"]["filter"]
    types = {part.removeprefix("type:") for part in f.split(",")}
    assert {"journal-article", "proceedings-article", "posted-content"} <= types
    assert ({"book-chapter", "book"} <= types) is books_kept
    assert not types & {"component", "dataset", "peer-review", "reference-entry", "journal-issue"}
    assert "has-abstract" not in f, "a record without an abstract is still citable"


def test_crossref_records_the_work_type(monkeypatch) -> None:
    item = {"DOI": "10.1/x", "title": ["An island of toys"], "type": "book-chapter"}
    monkeypatch.setattr("core.knowledge.httpx.Client", _Fake(lambda u, p: {"message": {"items": [item]}}))
    (doc,) = kn._crossref_search("q", 5, scope=WORK_SCOPE_PAPERS_AND_BOOKS)
    assert doc.metadata["work_type"] == "book-chapter"


@pytest.mark.parametrize("scope,books_kept", [
    (WORK_SCOPE_PAPERS, False),
    (WORK_SCOPE_PAPERS_AND_BOOKS, True),
])
def test_openalex_type_filter_follows_the_scope(monkeypatch, scope, books_kept) -> None:
    work = {"id": "https://openalex.org/W1", "title": "Millennial monsters", "type": "article"}
    monkeypatch.setattr("core.knowledge.httpx.Client", _Fake(lambda u, p: {"results": [work]}))
    (doc,) = kn._openalex_search("q", 5, scope=scope)
    f = _Fake.calls[0]["params"]["filter"]
    assert f.startswith("type:")
    types = set(f.removeprefix("type:").split("|"))
    assert {"article", "preprint", "review"} <= types
    assert ({"book", "book-chapter"} <= types) is books_kept
    assert not types & {"dataset", "paratext", "peer-review", "reference-entry"}
    assert doc.metadata["work_type"] == "article"


def test_the_router_passes_the_scope_only_to_adapters_that_filter_by_type(monkeypatch) -> None:
    got: dict = {}
    scoped_names = ("openalex", "crossref", "openaire")

    def scoped(name):
        def fn(query, top_k, *, timeout_s=10.0, scope=WORK_SCOPE_PAPERS):
            got[name] = scope
            return []
        return fn

    def plain(query, top_k, *, timeout_s=10.0):
        got["pubmed"] = True
        return []

    for name in scoped_names:
        monkeypatch.setitem(kn._SOURCE_REGISTRY, name, scoped(name))
    monkeypatch.setitem(kn._SOURCE_REGISTRY, "pubmed", plain)
    asyncio.run(kn._route_external("q", 5, [*scoped_names, "pubmed"],
                                   work_scope=WORK_SCOPE_PAPERS_AND_BOOKS))
    assert got == {**{n: WORK_SCOPE_PAPERS_AND_BOOKS for n in scoped_names}, "pubmed": True}


def test_asearch_hands_the_scope_to_the_router(monkeypatch) -> None:
    seen: dict = {}

    async def fake_router(query, top_k, sources, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr("core.knowledge._route_external", fake_router)
    k = Knowledge(KnowledgeConfig(enabled=True, source_routing="manual",
                                  external_fallback=["crossref"], web_search=False))
    monkeypatch.setattr(k, "_search_axon", lambda q, top_k=0: [])
    asyncio.run(k.asearch("q", work_scope=WORK_SCOPE_PAPERS_AND_BOOKS))
    assert seen.get("work_scope") == WORK_SCOPE_PAPERS_AND_BOOKS
    seen.clear()
    asyncio.run(k.asearch("q"))
    assert seen.get("work_scope") == WORK_SCOPE_PAPERS, "papers only unless a caller says otherwise"


def _engine(tmp_path: Path) -> Engine:
    return Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
        # These tests are about which search runs, not about pausing for
        # paywalled PDFs; the fake DOI-bearing docs would otherwise trigger it.
        pauses=PausesConfig(papers=False),
    ))


@pytest.mark.parametrize("no_sim,expected", [
    (False, WORK_SCOPE_PAPERS),
    (True, WORK_SCOPE_PAPERS_AND_BOOKS),
])
@pytest.mark.asyncio
async def test_the_literature_node_keeps_books_only_for_a_quest_without_an_experiment(
    tmp_path: Path, no_sim: bool, expected: str,
) -> None:
    eng = _engine(tmp_path)
    scopes: list = []

    async def fake_search(query, **kw):  # noqa: ANN001
        scopes.append(kw.get("work_scope"))
        return [RetrievedDoc(content="a", metadata={"doi": "10.1/a", "title": "A study of toys"})]

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    await eng._node_literature({"topic": "t", "chosen_idea": {"title": "T"},
                                "no_simulation_resolved": no_sim})
    assert scopes and set(scopes) == {expected}


@pytest.mark.parametrize("no_sim,expected", [
    (False, WORK_SCOPE_PAPERS),
    (True, WORK_SCOPE_PAPERS_AND_BOOKS),
])
@pytest.mark.asyncio
async def test_cross_check_searches_with_the_same_scope(tmp_path: Path, no_sim: bool, expected: str) -> None:
    eng = _engine(tmp_path)
    scopes: list = []

    async def fake_search(query, **kw):  # noqa: ANN001
        scopes.append(kw.get("work_scope"))
        return []

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    await eng._node_cross_check({"analysis": {"key_findings": ["x rises with y"]},
                                 "no_simulation_resolved": no_sim})
    assert scopes == [expected]


@pytest.mark.parametrize("no_sim,expected", [
    (False, WORK_SCOPE_PAPERS),
    (True, WORK_SCOPE_PAPERS_AND_BOOKS),
])
@pytest.mark.asyncio
async def test_ideate_seeds_with_the_same_scope(tmp_path: Path, no_sim: bool, expected: str) -> None:
    eng = _engine(tmp_path)
    scopes: list = []

    class _Stop(Exception):
        pass

    async def fake_search(query, **kw):  # noqa: ANN001
        scopes.append(kw.get("work_scope"))
        raise _Stop

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    with pytest.raises(_Stop):
        await eng._node_ideate({"topic": "t", "no_simulation_resolved": no_sim})
    assert scopes == [expected]


@pytest.mark.asyncio
async def test_auto_collect_top_up_searches_with_the_same_scope(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    eng.knowledge.enabled = True
    scopes: list = []

    async def fake_search(query, **kw):  # noqa: ANN001
        scopes.append(kw.get("work_scope"))
        return []

    async def zero(*_a, **_k):
        return 0

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    monkeypatch.setattr(eng, "_run_dataset_adapters", zero)
    eng.config.engine.auto_collect_data = True
    await eng._node_auto_collect_data({"topic": "bilingualism", "design": {},
                                       "no_simulation_resolved": True})
    assert scopes == [WORK_SCOPE_PAPERS_AND_BOOKS]


# --- CORE without a key ---------------------------------------------------------

def test_core_searches_without_a_key(monkeypatch) -> None:
    work = {"title": "Games, toys, and pastimes", "abstract": "A history.", "doi": "10.1/core",
            "downloadUrl": "https://core.ac.uk/download/1.pdf", "yearPublished": 2007,
            "documentType": "research", "authors": [{"name": "M. Mikula"}]}
    monkeypatch.setattr("core.knowledge.httpx.Client", _Fake(lambda u, p: {"results": [work]}))
    (doc,) = kn._core_search("toys history", 5)
    assert _Fake.calls[0]["url"] == "https://api.core.ac.uk/v3/search/works/"
    assert "Authorization" not in _Fake.calls[0]["headers"]
    assert doc.metadata["doi"] == "10.1/core" and doc.metadata["year"] == 2007

    monkeypatch.setenv("CORE_API_KEY", "CORE-KEY")
    kn._core_search("toys history", 5)
    assert _Fake.calls[-1]["headers"]["Authorization"] == "Bearer CORE-KEY"


def test_core_tolerates_an_empty_fulltext_url_list(monkeypatch) -> None:
    work = {"title": "T", "sourceFulltextUrls": []}
    monkeypatch.setattr("core.knowledge.httpx.Client", _Fake(lambda u, p: {"results": [work]}))
    (doc,) = kn._core_search("q", 5)
    assert doc.metadata["url"] == ""


# --- OpenAIRE -------------------------------------------------------------------

def _openaire_record(title: str, instance_type: str, doi: str = "10.1/oa") -> dict:
    return {
        "mainTitle": title,
        "publicationDate": "2021-01-01",
        "publisher": "MIT Press",
        "container": {"name": "Neurobiology of Language"},
        "authors": [{"fullName": "E. Bialystok"}],
        "descriptions": ["<jats:p>Bilingual <jats:italic>advantage</jats:italic>.</jats:p>"],
        "pids": [{"scheme": "doi", "value": doi}],
        "instances": [{"type": instance_type, "urls": ["https://example.org/x"]}],
        "bestAccessRight": {"label": "OPEN"},
    }


def test_openaire_parses_a_publication(monkeypatch) -> None:
    body = {"results": [_openaire_record("Bilingualism and the brain", "Article")]}
    monkeypatch.setattr("core.knowledge.httpx.Client", _Fake(lambda u, p: body))
    (doc,) = kn._openaire_search("bilingualism executive function", 5)
    call = _Fake.calls[0]
    assert call["url"] == "https://api.openaire.eu/graph/v1/researchProducts"
    assert call["params"]["type"] == "publication"
    md = doc.metadata
    assert md["source"] == "openaire" and md["doi"] == "10.1/oa"
    assert md["year"] == 2021 and md["venue"] == "Neurobiology of Language"
    assert md["open_access"] is True and md["work_type"] == "article"
    assert "<" not in doc.content and "advantage" in doc.content


@pytest.mark.parametrize("scope,kept", [
    (WORK_SCOPE_PAPERS, {"A paper"}),
    (WORK_SCOPE_PAPERS_AND_BOOKS, {"A paper", "A chapter"}),
])
def test_openaire_keeps_types_by_scope(monkeypatch, scope, kept) -> None:
    body = {"results": [
        _openaire_record("A paper", "Article", doi="10.1/p"),
        _openaire_record("A chapter", "Part of book or chapter of book", doi="10.1/c"),
        _openaire_record("A dataset", "Dataset", doi="10.1/d"),
    ]}
    monkeypatch.setattr("core.knowledge.httpx.Client", _Fake(lambda u, p: body))
    docs = kn._openaire_search("q", 5, scope=scope)
    assert {d.metadata["title"] for d in docs} == kept


def test_openaire_retries_with_fewer_words_when_every_word_must_match(monkeypatch) -> None:
    hit = {"results": [_openaire_record("Toys", "Article")]}
    monkeypatch.setattr(
        "core.knowledge.httpx.Client",
        _Fake(lambda u, p: hit if len(p["search"].split()) <= 3 else {"results": []}),
    )
    docs = kn._openaire_search("action figures collectible toys popular culture", 5)
    assert [c["params"]["search"] for c in _Fake.calls] == [
        "action figures collectible toys popular culture", "action figures collectible"]
    assert len(docs) == 1


def test_openaire_does_not_retry_after_a_failed_request(monkeypatch) -> None:
    class _Down:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, *a, **k):
            _Down.n = getattr(_Down, "n", 0) + 1
            raise RuntimeError("down")

    monkeypatch.setattr("core.knowledge.httpx.Client", _Down)
    assert kn._openaire_search("one two three four five", 5) == []
    assert _Down.n == 1, "a network failure is not an empty result; do not spend a second call"


# --- DOAJ -----------------------------------------------------------------------

def _doaj_record(title: str) -> dict:
    return {"bibjson": {
        "title": title,
        "abstract": "Executive functions in early bilinguals.",
        "author": [{"name": "A. Author"}],
        "identifier": [{"type": "eissn", "id": "1234-5678"}, {"type": "doi", "id": "10.1/doaj"}],
        "journal": {"title": "Psychology", "publisher": "EKT"},
        "year": "2020",
        "link": [{"type": "fulltext", "url": "https://ejournals.example/x"}],
    }}


def test_doaj_parses_an_article_and_escapes_query_syntax(monkeypatch) -> None:
    body = {"results": [_doaj_record("Executive functions in bilinguals")]}
    monkeypatch.setattr("core.knowledge.httpx.Client", _Fake(lambda u, p: body))
    (doc,) = kn._doaj_search('bilingual: "executive" function/age -x', 5)
    url = _Fake.calls[0]["url"]
    assert url.startswith("https://doaj.org/api/search/articles/")
    q = unquote(url.rsplit("/", 1)[1])
    assert q == "bilingual executive function age x"
    md = doc.metadata
    assert md["source"] == "doaj" and md["doi"] == "10.1/doaj"
    assert md["url"] == "https://doi.org/10.1/doaj" and md["year"] == 2020
    assert md["venue"] == "Psychology" and md["open_access"] is True
    assert md["work_type"] == "journal-article"


def test_doaj_retries_with_fewer_words(monkeypatch) -> None:
    hit = {"results": [_doaj_record("Toys")]}

    def respond(url, params):
        words = unquote(url.rsplit("/", 1)[1]).split()
        return hit if len(words) <= 3 else {"results": []}

    monkeypatch.setattr("core.knowledge.httpx.Client", _Fake(respond))
    assert len(kn._doaj_search("sculpture popular culture film cartoon", 5)) == 1
    assert len(_Fake.calls) == 2


# --- catalog and router -------------------------------------------------------

def test_new_sources_are_searchable_and_described() -> None:
    names = {e["name"]: e for e in kn._SOURCE_CATALOG}
    for name in ("openaire", "doaj", "core"):
        assert name in kn._SOURCE_REGISTRY
        assert names[name]["has_search_adapter"] is True
    assert "required" not in names["core"]["access"].lower()


def test_router_prompt_points_humanities_at_the_open_access_sources() -> None:
    text = (Path(kn.__file__)).read_text(encoding="utf-8")
    assert "CORE, OpenAIRE, DOAJ" in text
