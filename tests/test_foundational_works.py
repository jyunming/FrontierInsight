"""Foundational works join a literature pass: suggested by the model and looked
up by title, or cited by several retrieved papers (``core/knowledge.py``), then
screened with the rest (``core/engine.py``)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import core.knowledge as kn
import core.passages as pmod
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig,
)
from core.engine import Engine
from core.knowledge import RetrievedDoc, _openalex_cited_by_retrieved, _openalex_title_lookup, _titles_match

VERLET = {
    "id": "https://openalex.org/W2323178299",
    "title": 'Computer "Experiments" on Classical Fluids. I. Thermodynamical Properties of Lennard-Jones Molecules',
    "publication_year": 1967, "type": "article", "doi": "https://doi.org/10.1103/physrev.159.98",
    "cited_by_count": 9428, "authorships": [{"author": {"display_name": "Loup Verlet"}}],
    "primary_location": {"source": {"display_name": "Physical Review"}},
}
DECOY = {
    "id": "https://openalex.org/W1", "title": "Computer experiments on classical fluids revisited",
    "publication_year": 2015, "type": "article", "authorships": [{"author": {"display_name": "A. Jones"}}],
}


@pytest.fixture(autouse=True)
def _unscored(monkeypatch):
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: None)


def _fake_http(monkeypatch, responses: list[dict]) -> list[dict]:
    monkeypatch.setattr(kn, "_OPENALEX_GAP_S", 0.0)
    calls: list[dict] = []

    def fake_get(url, params, timeout_s, *, source="", headers=None):  # noqa: ANN001
        calls.append(dict(params or {}))
        return responses[len(calls) - 1]

    monkeypatch.setattr(kn, "_http_get_json", fake_get)
    return calls


def test_titles_match_on_their_words() -> None:
    assert _titles_match("Computer experiments on classical fluids", VERLET["title"])
    assert _titles_match(
        "Geometric Numerical Integration",
        "Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations",
    )
    assert not _titles_match("Numerical Recipes", "Numerical Recipes in C")
    assert not _titles_match("Computer experiments on quantum fluids", VERLET["title"])


def test_a_suggested_work_is_kept_only_when_openalex_has_it(monkeypatch) -> None:
    calls = _fake_http(monkeypatch, [{"results": [DECOY, VERLET]}, {"results": [DECOY, VERLET]}])
    doc = _openalex_title_lookup({"title": "Computer experiments on classical fluids", "authors": "Verlet", "year": 1967})
    assert doc is not None and doc.metadata["doi"] == "10.1103/physrev.159.98"
    assert doc.metadata["foundational"] == kn.FOUNDATIONAL_SUGGESTED
    assert calls[0]["filter"].startswith("title.search:Computer experiments on classical fluids,type:")
    assert "book|book-chapter" in calls[0]["filter"] and calls[0]["sort"] == "cited_by_count:desc"
    # The title is there, but neither the year nor the author is this work's.
    assert _openalex_title_lookup({"title": "Computer experiments on classical fluids", "authors": "Smith", "year": 1990}) is None


def test_the_works_several_retrieved_papers_cite_are_added(monkeypatch) -> None:
    docs = [RetrievedDoc(content="", metadata={"source": "openalex", "url": f"https://openalex.org/W{i}"}) for i in (1, 2, 3)]
    docs.append(RetrievedDoc(content="", metadata={"source": "web_search", "url": "https://example.org/page"}))
    calls = _fake_http(monkeypatch, [
        {"results": [
            {"id": "https://openalex.org/W1", "referenced_works": [
                "https://openalex.org/W90", "https://openalex.org/W91", "https://openalex.org/W2"]},
            {"id": "https://openalex.org/W2", "referenced_works": ["https://openalex.org/W90", "https://openalex.org/W92"]},
            {"id": "https://openalex.org/W3", "referenced_works": [
                "https://openalex.org/W90", "https://openalex.org/W92", "https://openalex.org/W92"]},
        ]},
        {"results": [VERLET | {"id": "https://openalex.org/W90"},
                     {"id": "https://openalex.org/W92", "title": "A dataset", "type": "dataset"}]},
    ])
    found = _openalex_cited_by_retrieved(docs)
    assert calls[0]["filter"] == "openalex:W1|W2|W3" and calls[0]["select"] == "id,referenced_works"
    # W90 is cited by three papers and W92 by two; W91 by one, and W2 is already retrieved.
    assert calls[1]["filter"] == "openalex:W90|W92"
    assert [d.metadata["url"] for d in found] == ["https://openalex.org/W90"]  # a dataset is not a work to cite
    assert found[0].metadata["foundational"] == "cited by 3 of the retrieved papers"


@pytest.mark.asyncio
async def test_lookups_go_one_at_a_time_and_skip_what_was_retrieved(monkeypatch) -> None:
    """Fired together, three of five title lookups came back 429 from OpenAlex."""
    monkeypatch.setattr(kn, "_OPENALEX_GAP_S", 0.0)
    active: list[int] = []
    order: list[str] = []

    def lookup(work, **kw):  # noqa: ANN001
        active.append(1)
        assert len(active) == 1, "lookups overlap"
        order.append(work["title"])
        active.pop()
        return RetrievedDoc(content="", metadata={"title": work["title"], "url": f"https://openalex.org/{work['id']}"})

    def cited(docs, **kw):  # noqa: ANN001
        order.append("cited")
        return [RetrievedDoc(content="", metadata={"title": "Already here", "url": "https://openalex.org/W1"})]

    monkeypatch.setattr(kn, "_openalex_title_lookup", lookup)
    monkeypatch.setattr(kn, "_openalex_cited_by_retrieved", cited)
    k = kn.Knowledge(KnowledgeConfig(enabled=False))
    retrieved = [RetrievedDoc(content="", metadata={"title": "Already here", "url": "https://openalex.org/W1"})]
    found = await k.find_foundational_works(
        [{"title": "First", "id": "W7"}, "not a work", {"title": "Second", "id": "W8"}], retrieved)
    assert order == ["First", "Second", "cited"]
    assert [d.metadata["title"] for d in found] == ["First", "Second"]


@pytest.mark.asyncio
async def test_refused_lookups_are_logged(monkeypatch, caplog) -> None:
    """A lookup OpenAlex refuses looks like a work it does not hold: four real
    quests logged "0 new candidate(s)" after 13-15 refusals each."""
    import core.source_failures as sf

    monkeypatch.setattr(kn, "_OPENALEX_GAP_S", 0.0)

    def refused(work, **kw):  # noqa: ANN001
        sf.record_failure("openalex", "http_429", status=429, url="https://api.openalex.org/works")
        return None

    monkeypatch.setattr(kn, "_openalex_title_lookup", refused)
    monkeypatch.setattr(kn, "_openalex_cited_by_retrieved", lambda docs, **kw: [])
    k = kn.Knowledge(KnowledgeConfig(enabled=False))
    token = sf.current_quest.set("q-refused")
    try:
        # Refusals an earlier search in the quest met are not this lookup's.
        sf.record_failure("openalex", "http_429", status=429, count=3)
        with caplog.at_level("WARNING"):
            found = await k.find_foundational_works([{"title": "First work"}, {"title": "Second work"}], [])
    finally:
        sf.current_quest.reset(token)
        sf.reset("q-refused")
    assert found == []
    assert any("refused 2 request(s) with HTTP 429" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_plain_miss_is_not_reported_as_a_refusal(monkeypatch, caplog) -> None:
    import core.source_failures as sf

    monkeypatch.setattr(kn, "_OPENALEX_GAP_S", 0.0)
    monkeypatch.setattr(kn, "_openalex_title_lookup", lambda work, **kw: None)
    monkeypatch.setattr(kn, "_openalex_cited_by_retrieved", lambda docs, **kw: [])
    k = kn.Knowledge(KnowledgeConfig(enabled=False))
    token = sf.current_quest.set("q-miss")
    try:
        with caplog.at_level("WARNING"):
            assert await k.find_foundational_works([{"title": "First work"}], []) == []
    finally:
        sf.current_quest.reset(token)
        sf.reset("q-miss")
    assert not any("HTTP 429" in r.getMessage() for r in caplog.records)


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="Integrators for a damped oscillator", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False, web_search=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
        pauses=PausesConfig(papers=False),
    ))
    eng.config.knowledge.enabled = True
    eng.config.knowledge.source_routing = "manual"
    return eng


@pytest.mark.asyncio
async def test_foundational_works_are_suggested_looked_up_and_screened(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)

    async def chat(prompt, node=""):  # noqa: ANN001
        if node == "literature_query":
            return json.dumps({"queries": ["damped oscillator numerical integration"]})
        assert node == "literature_foundational" and "Integrators for a damped oscillator" in prompt
        return json.dumps({"works": [{"title": "Computer experiments on classical fluids", "authors": "Verlet",
                                      "year": 1967}, {"title": ""}]})

    eng._chat = chat  # type: ignore[method-assign]
    retrieved = RetrievedDoc(content="An abstract.", metadata={
        "title": "A class of symplectic integrators with adaptive time step for separable Hamiltonian systems",
        "doi": "10.1/s", "source": "openalex", "url": "https://openalex.org/W1"})

    async def fake_search(query, **kw):  # noqa: ANN001
        return [retrieved]

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    seen: dict = {}

    async def fake_find(suggestions, docs):  # noqa: ANN001
        seen["suggestions"], seen["docs"] = suggestions, docs
        return [
            RetrievedDoc(content="Verlet", metadata={
                "title": "Computer experiments on classical fluids", "doi": "10.1103/physrev.159.98",
                "source": "openalex", "work_type": "article", "foundational": kn.FOUNDATIONAL_SUGGESTED}),
            # The same work the search already found, under another DOI.
            RetrievedDoc(content="again", metadata={
                "title": "A Class of Symplectic Integrators with Adaptive Time Step for Separable Hamiltonian Systems.",
                "doi": "10.9/other", "source": "openalex", "foundational": "cited by 2 of the retrieved papers"}),
        ]

    eng.knowledge.find_foundational_works = fake_find  # type: ignore[method-assign]
    eng.knowledge.fetch_full_text = AsyncMock(side_effect=lambda docs: docs)  # type: ignore[method-assign]
    screened: dict = {}

    async def screen(topic, docs, **kw):  # noqa: ANN001
        screened["docs"] = docs
        return docs

    monkeypatch.setattr(eng, "_screen_literature", screen)
    patch = await eng._node_literature({"topic": "Integrators for a damped oscillator", "chosen_idea": {"title": "T"}})
    assert seen["suggestions"] == [{"title": "Computer experiments on classical fluids", "authors": "Verlet", "year": 1967}]
    assert [d.metadata["doi"] for d in seen["docs"]] == ["10.1/s"]
    assert [d.metadata["doi"] for d in screened["docs"]] == ["10.1/s", "10.1103/physrev.159.98"]
    assert "10.1103/physrev.159.98" in [e["metadata"]["doi"] for e in patch["literature"]]


@pytest.mark.asyncio
async def test_nothing_is_added_when_switched_off_and_a_failed_suggestion_still_snowballs(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.knowledge.find_foundational_works = AsyncMock(return_value=[])  # type: ignore[method-assign]
    eng.config.knowledge.foundational_works = False
    assert await eng._foundational_works("t", []) == []
    eng.knowledge.find_foundational_works.assert_not_awaited()

    eng.config.knowledge.foundational_works = True

    async def failing_chat(prompt, node=""):  # noqa: ANN001
        raise RuntimeError("provider down")

    eng._chat = failing_chat  # type: ignore[method-assign]
    assert await eng._foundational_works("t", []) == []
    eng.knowledge.find_foundational_works.assert_awaited_once_with([], [])


@pytest.mark.asyncio
async def test_the_screen_is_told_which_candidates_are_foundational(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    prompts: list[str] = []

    async def chat(prompt, node=""):  # noqa: ANN001
        prompts.append(prompt)
        return json.dumps({"grades": [{"i": 0, "grade": 2}, {"i": 1, "grade": 2}]})

    eng._chat = chat  # type: ignore[method-assign]
    docs = [
        RetrievedDoc(content="t", metadata={"title": "Geometric Numerical Integration", "work_type": "book",
                                            "year": 2006, "source": "openalex",
                                            "foundational": "cited by 5 of the retrieved papers"}),
        RetrievedDoc(content="t", metadata={"title": "Computer experiments on classical fluids", "work_type": "article",
                                            "year": 1967, "source": "openalex",
                                            "foundational": kn.FOUNDATIONAL_SUGGESTED}),
    ]
    await eng._screen_literature("Integrators for a damped oscillator", docs)
    assert ("[0] (foundational book) Geometric Numerical Integration — 2006, book, "
            "cited by 5 of the retrieved papers ::") in prompts[0]
    assert ("[1] (foundational paper) Computer experiments on classical fluids — 1967, article, "
            "suggested as a foundational work ::") in prompts[0]
    assert "Verlet's 1967 paper" in prompts[0]


# ---- up to eight are suggested and looked up, and the run log says what happened --

def _works(n: int) -> list[dict]:
    # Titles long enough that eight of them together pass 600 characters.
    return [
        {"title": f"An original paper on a method the research topic relies on, number {i}, in full",
         "authors": f"Author{i}", "year": 1900 + i}
        for i in range(1, n + 1)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [kn.WORK_SCOPE_PAPERS, kn.WORK_SCOPE_PAPERS_AND_BOOKS])
async def test_the_model_is_asked_for_eight_works_and_a_ninth_is_ignored(tmp_path: Path, scope: str) -> None:
    eng = _engine(tmp_path)
    prompts: list[str] = []

    async def chat(prompt, node=""):  # noqa: ANN001
        prompts.append(prompt)
        return json.dumps({"works": _works(9)})

    eng._chat = chat  # type: ignore[method-assign]
    suggested = await eng._suggest_foundational_works("Some research topic", work_scope=scope)
    assert "List up to EIGHT foundational works" in prompts[0] and "FIVE" not in prompts[0]
    assert [w["title"] for w in suggested] == [w["title"] for w in _works(8)]


@pytest.mark.asyncio
async def test_eight_suggested_works_are_looked_up_and_a_ninth_is_not(monkeypatch) -> None:
    monkeypatch.setattr(kn, "_OPENALEX_GAP_S", 0.0)
    looked_up: list[str] = []

    def lookup(work, **kw):  # noqa: ANN001
        looked_up.append(work["title"])
        return None

    monkeypatch.setattr(kn, "_openalex_title_lookup", lookup)
    monkeypatch.setattr(kn, "_openalex_cited_by_retrieved", lambda docs, **kw: [])
    await kn.Knowledge(KnowledgeConfig(enabled=False)).find_foundational_works(_works(9), [])
    assert kn.FOUNDATIONAL_MAX_SUGGESTED == 8
    assert looked_up == [w["title"] for w in _works(8)]


@pytest.mark.asyncio
async def test_the_run_log_lists_every_suggestion_and_what_became_of_each(tmp_path: Path) -> None:
    """A work the model never named looked exactly like one OpenAlex does not hold:
    the run log recorded only how many were suggested."""
    eng = _engine(tmp_path)
    suggestions = _works(6) + [
        {"title": "The outcome of a stochastic epidemic: a note on Bailey's paper", "authors": "Whittle", "year": 1955},
        {"title": "Stochastic epidemic models and their statistical analysis", "authors": ["Andersson", "Britton"], "year": 2000},
        {"title": "A ninth work the prompt never asked for", "authors": "Extra", "year": 1999},
    ]

    async def chat(prompt, node=""):  # noqa: ANN001
        return json.dumps({"works": suggestions})

    eng._chat = chat  # type: ignore[method-assign]
    whittle = RetrievedDoc(content="w", metadata={
        "title": "THE OUTCOME OF A STOCHASTIC EPIDEMIC - A NOTE ON BAILEY'S PAPER", "year": 1955,
        "foundational": kn.FOUNDATIONAL_SUGGESTED})
    cited = RetrievedDoc(content="c", metadata={
        "title": "Infectious diseases of humans", "year": 1991, "foundational": "cited by 7 of the retrieved papers"})
    already = RetrievedDoc(content="a", metadata={
        "title": "Stochastic epidemic models and their statistical analysis", "year": 2000, "doi": "10.1/andersson"})
    seen: dict = {}

    async def fake_find(suggested, docs):  # noqa: ANN001
        seen["suggested"] = suggested
        # The lookup dropped the work the search already returned.
        return [whittle, cited]

    eng.knowledge.find_foundational_works = fake_find  # type: ignore[method-assign]
    new = await eng._foundational_works("Some research topic", [already])

    assert [d.metadata["year"] for d in new] == [1955, 1991]
    assert len(seen["suggested"]) == 8, "the ninth is never looked up"
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    suggested_line = next(ln for ln in log.splitlines() if "foundational works suggested (8):" in ln)
    # Every suggestion, untruncated, as "title (year, authors)".
    assert len(suggested_line) > 700
    for w in suggestions[:8]:
        who = ", ".join(w["authors"]) if isinstance(w["authors"], list) else w["authors"]
        assert f"{w['title']} ({w['year']}, {who})" in suggested_line
    assert "A ninth work" not in log
    outcome = next(ln for ln in log.splitlines() if "found in OpenAlex and added" in ln)
    whittle_text = "The outcome of a stochastic epidemic: a note on Bailey's paper (1955, Whittle)"
    assert f"found in OpenAlex and added (1): {whittle_text} |" in outcome
    assert ("already among the search results (1): "
            "Stochastic epidemic models and their statistical analysis (2000, Andersson, Britton) |") in outcome
    assert "dropped, not found in OpenAlex (6): " + f"{_works(6)[0]['title']} (1901, Author1); " in outcome
    assert whittle_text not in outcome.split("dropped")[1]
