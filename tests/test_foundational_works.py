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
from core.engine import Engine, _foundational_outcomes
from core.knowledge import (
    RetrievedDoc, _content_stems, _openalex_author_year_lookup, _openalex_cited_by_retrieved,
    _openalex_title_lookup, _surname_among, _titles_match, foundational_work_text,
)

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


@pytest.fixture(autouse=True)
def _no_author_year_lookup(monkeypatch):
    """A title lookup that finds nothing is followed by an author + year one. Here that finds nothing too, unless a
    test puts the real one back, so no test reaches the network by accident."""
    monkeypatch.setattr(kn, "_openalex_author_year_lookup", lambda work, **kw: None)


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


# --- a title that matches nothing is looked up by author and year ----------------
#
# The shape three real quests met: the model names Whittle (1955) and gives the work an invented
# title. OpenAlex holds the real one as the first record of that surname and year that is about
# the subject; ahead of it by citations sit a paper on stationary processes by the same Whittle and
# a chemist's paper by another one. The fixtures below are the records OpenAlex returned.

WHITTLE_EPIDEMIC = {
    "id": "https://openalex.org/W1965", "title": "THE OUTCOME OF A STOCHASTIC EPIDEMIC—A NOTE ON BAILEY'S PAPER",
    "publication_year": 1955, "type": "article", "doi": "https://doi.org/10.1093/biomet/42.1-2.116",
    "cited_by_count": 257, "authorships": [{"author": {"display_name": "Peter Whittle"}}],
    "primary_location": {"source": {"display_name": "Biometrika"}},
}
WHITTLE_PLANE = {
    "id": "https://openalex.org/W2001", "title": "ON STATIONARY PROCESSES IN THE PLANE", "publication_year": 1954,
    "type": "article", "cited_by_count": 1502, "authorships": [{"author": {"display_name": "Peter Whittle"}}],
}
WHITTLE_CHEMIST = {
    "id": "https://openalex.org/W2002", "title": "Matrix Isolation Method for the Experimental Study of Unstable Species",
    "publication_year": 1954, "type": "article", "cited_by_count": 314,
    "authorships": [{"author": {"display_name": "E. Whittle"}}, {"author": {"display_name": "David A. Dows"}}],
}
INVENTED = {"title": "A stochastic model for the spread of an infectious disease", "authors": "Whittle", "year": 1955}
REAL_AUTHOR_YEAR_LOOKUP = _openalex_author_year_lookup


@pytest.fixture
def real_author_year_lookup(monkeypatch):
    monkeypatch.setattr(kn, "_openalex_author_year_lookup", REAL_AUTHOR_YEAR_LOOKUP)


def test_the_author_and_year_lookup_finds_the_work_behind_an_invented_title(monkeypatch) -> None:
    calls = _fake_http(monkeypatch, [{"results": [WHITTLE_PLANE, WHITTLE_CHEMIST, WHITTLE_EPIDEMIC]}])
    doc = _openalex_author_year_lookup(INVENTED)
    assert doc is not None and doc.metadata["doi"] == "10.1093/biomet/42.1-2.116"
    assert doc.metadata["foundational"] == kn.FOUNDATIONAL_BY_AUTHOR
    # One request: the surname, a year either side of 1955, most cited first, a page deep enough to
    # get past a common surname's famous work.
    assert len(calls) == 1
    assert calls[0]["filter"].startswith("raw_author_name.search:whittle,publication_year:1954-1956,type:")
    assert "book|book-chapter" in calls[0]["filter"]
    assert calls[0]["sort"] == "cited_by_count:desc" and calls[0]["per-page"] == "25"


def test_a_work_by_the_author_that_shares_no_subject_word_is_not_taken(monkeypatch) -> None:
    """Bailey (1975) is a name the model attached to a work that is not in the database: the
    author's other papers of those years are about something else."""
    _fake_http(monkeypatch, [{"results": [
        {"id": "https://openalex.org/W3", "title": "Communication of Innovations: A Cross-Cultural Approach.",
         "publication_year": 1974, "type": "article", "cited_by_count": 2821,
         "authorships": [{"author": {"display_name": "Fiona Bailey"}}]},
    ]}])
    assert _openalex_author_year_lookup(
        {"title": "A stochastic model for the spread of a disease", "authors": "Bailey", "year": 1975}) is None


def test_the_surname_must_be_a_whole_word_of_an_author_name(monkeypatch) -> None:
    assert _surname_among("ball", ["J. Timothy Ball", "Ian Woodrow"])
    assert not _surname_among("ball", ["Ballesteros", "Ian Woodrow"])
    epidemic = {"id": "https://openalex.org/W4", "title": "Stochastic epidemic models", "publication_year": 1986,
                "type": "article", "authorships": [{"author": {"display_name": "M. Ballesteros"}}]}
    _fake_http(monkeypatch, [{"results": [epidemic]}])
    assert _openalex_author_year_lookup(
        {"title": "The probability of a major outbreak in a stochastic epidemic model", "authors": "Ball", "year": 1986}) is None


def test_a_subject_word_is_matched_across_its_endings() -> None:
    assert _content_stems("The general theory of epidemics") & _content_stems(
        "A contribution to the mathematical theory of epidemics") == {"theory", "epidem"}
    # "Mikroskope" and "Mikroskops" are one word; the other words of the two titles differ.
    assert _content_stems("Über die Bildnisse der Mikroskope") & _content_stems(
        "Beiträge zur Theorie des Mikroskops und der mikroskopischen Wahrnehmung") == {"mikros"}
    # Nothing in common but words that name no subject: not evidence the record is the work meant.
    assert not _content_stems("A stochastic model for the spread of a disease") & _content_stems(
        "Perioperative outcomes after open and endovascular repair of an aneurysm")


def test_no_request_is_made_without_an_author_a_year_or_a_subject_word(monkeypatch) -> None:
    calls = _fake_http(monkeypatch, [])
    assert _openalex_author_year_lookup({"title": "A stochastic model", "authors": "", "year": 1955}) is None
    assert _openalex_author_year_lookup({"title": "A stochastic model", "authors": "Whittle"}) is None
    assert _openalex_author_year_lookup({"title": "The of the", "authors": "Whittle", "year": 1955}) is None
    assert calls == []


@pytest.mark.asyncio
async def test_a_title_that_matches_nothing_is_looked_up_again_by_author_and_year(
    monkeypatch, real_author_year_lookup,
) -> None:
    calls = _fake_http(monkeypatch, [{"results": []}, {"results": [WHITTLE_PLANE, WHITTLE_EPIDEMIC]}])
    monkeypatch.setattr(kn, "_openalex_cited_by_retrieved", lambda docs, **kw: [])
    found = await kn.Knowledge(KnowledgeConfig(enabled=False)).find_foundational_works([INVENTED], [])
    assert [c["filter"].split(":")[0] for c in calls] == ["title.search", "raw_author_name.search"]
    assert [d.metadata["title"] for d in found] == ["THE OUTCOME OF A STOCHASTIC EPIDEMIC—A NOTE ON BAILEY'S PAPER"]
    assert found[0].metadata["suggested_works"] == [foundational_work_text(INVENTED)]


@pytest.mark.asyncio
async def test_a_title_that_matched_costs_no_second_request(monkeypatch, real_author_year_lookup) -> None:
    calls = _fake_http(monkeypatch, [{"results": [VERLET]}])
    monkeypatch.setattr(kn, "_openalex_cited_by_retrieved", lambda docs, **kw: [])
    found = await kn.Knowledge(KnowledgeConfig(enabled=False)).find_foundational_works(
        [{"title": "Computer experiments on classical fluids", "authors": "Verlet", "year": 1967}], [])
    assert len(calls) == 1 and found[0].metadata["foundational"] == kn.FOUNDATIONAL_SUGGESTED
    assert "suggested_works" not in found[0].metadata


@pytest.mark.asyncio
async def test_two_suggestions_that_find_one_work_return_it_once_and_both_are_noted(
    monkeypatch, real_author_year_lookup,
) -> None:
    other = {"title": "Stochastic epidemic models and their normalization", "authors": "Whittle", "year": 1955}
    _fake_http(monkeypatch, [
        {"results": []}, {"results": [WHITTLE_EPIDEMIC]}, {"results": []}, {"results": [WHITTLE_EPIDEMIC]},
    ])
    monkeypatch.setattr(kn, "_openalex_cited_by_retrieved", lambda docs, **kw: [])
    found = await kn.Knowledge(KnowledgeConfig(enabled=False)).find_foundational_works([INVENTED, other], [])
    assert len(found) == 1
    assert found[0].metadata["suggested_works"] == sorted([foundational_work_text(INVENTED), foundational_work_text(other)])


@pytest.mark.asyncio
async def test_a_work_the_search_already_returned_is_annotated_not_returned_again(
    monkeypatch, real_author_year_lookup,
) -> None:
    _fake_http(monkeypatch, [{"results": []}, {"results": [WHITTLE_EPIDEMIC]}])
    monkeypatch.setattr(kn, "_openalex_cited_by_retrieved", lambda docs, **kw: [])
    retrieved = RetrievedDoc(content="w", metadata={
        "title": "The outcome of a stochastic epidemic", "doi": "10.1093/biomet/42.1-2.116", "source": "openalex"})
    found = await kn.Knowledge(KnowledgeConfig(enabled=False)).find_foundational_works([INVENTED], [retrieved])
    assert found == []
    assert retrieved.metadata["suggested_works"] == [foundational_work_text(INVENTED)]


@pytest.mark.asyncio
async def test_no_author_and_year_request_follows_a_refused_title_lookup(monkeypatch) -> None:
    """A budget OpenAlex has spent refuses the next request as well: asking again only counts a
    second refusal against the quest."""
    import core.source_failures as sf

    monkeypatch.setattr(kn, "_OPENALEX_GAP_S", 0.0)

    def refused(work, **kw):  # noqa: ANN001
        sf.record_failure("openalex", "http_429", status=429, url="https://api.openalex.org/works")
        return None

    def never(work, **kw):  # noqa: ANN001
        raise AssertionError("asked again after a refusal")

    monkeypatch.setattr(kn, "_openalex_title_lookup", refused)
    monkeypatch.setattr(kn, "_openalex_author_year_lookup", never)
    monkeypatch.setattr(kn, "_openalex_cited_by_retrieved", lambda docs, **kw: [])
    token = sf.current_quest.set("q-refused-title")
    try:
        assert await kn.Knowledge(KnowledgeConfig(enabled=False)).find_foundational_works([INVENTED], []) == []
    finally:
        sf.current_quest.reset(token)
        sf.reset("q-refused-title")


@pytest.mark.asyncio
async def test_a_failed_author_and_year_lookup_loses_only_its_own_work(monkeypatch) -> None:
    def boom(work, **kw):  # noqa: ANN001
        raise RuntimeError("network")

    monkeypatch.setattr(kn, "_OPENALEX_GAP_S", 0.0)
    monkeypatch.setattr(kn, "_openalex_title_lookup", lambda work, **kw: None)
    monkeypatch.setattr(kn, "_openalex_author_year_lookup", boom)
    monkeypatch.setattr(kn, "_openalex_cited_by_retrieved", lambda docs, **kw: [])
    assert await kn.Knowledge(KnowledgeConfig(enabled=False)).find_foundational_works([INVENTED], []) == []


def test_the_outcome_of_a_work_found_by_author_and_year_is_added_not_dropped() -> None:
    found_doc = RetrievedDoc(content="w", metadata={
        "title": "THE OUTCOME OF A STOCHASTIC EPIDEMIC—A NOTE ON BAILEY'S PAPER", "year": 1955,
        "foundational": kn.FOUNDATIONAL_BY_AUTHOR, "suggested_works": [foundational_work_text(INVENTED)]})
    held = RetrievedDoc(content="k", metadata={
        "title": "A contribution to the mathematical theory of epidemics", "year": 1927,
        "suggested_works": ["The general theory of epidemics (1927, Kermack)"]})
    matched = {"title": "Stochastic epidemic models and their statistical analysis", "authors": "Andersson", "year": 2000}
    by_title = RetrievedDoc(content="a", metadata={
        "title": "Stochastic epidemic models and their statistical analysis", "year": 2000,
        "foundational": kn.FOUNDATIONAL_SUGGESTED})
    kermack = {"title": "The general theory of epidemics", "authors": "Kermack", "year": 1927}
    invented = {"title": "A study nobody wrote", "authors": "Nobody", "year": 1901}
    added, already, dropped = _foundational_outcomes(
        [INVENTED, matched, kermack, invented], [found_doc, by_title], [found_doc, by_title], [held])
    assert added == [
        "A stochastic model for the spread of an infectious disease (1955, Whittle) -> "
        "THE OUTCOME OF A STOCHASTIC EPIDEMIC—A NOTE ON BAILEY'S PAPER, by author and year",
        "Stochastic epidemic models and their statistical analysis (2000, Andersson)",  # a title match reads as before
    ]
    assert already == [
        "The general theory of epidemics (1927, Kermack) -> "
        "A contribution to the mathematical theory of epidemics, by author and year",
    ]
    assert dropped == ["A study nobody wrote (1901, Nobody)"]


def test_two_suggestions_that_share_a_title_are_each_paired_with_their_own_record() -> None:
    """Whittle's 1955 note and Bailey's 1962 book were suggested under one title; a record was
    paired with the first suggestion that carried the title, so the log line named the wrong one."""
    same = "A stochastic model for the spread of infection"
    whittle = {"title": same, "authors": "Whittle", "year": 1955}
    bailey = {"title": same, "authors": "Bailey", "year": 1962}
    note = RetrievedDoc(content="w", metadata={
        "title": "THE OUTCOME OF A STOCHASTIC EPIDEMIC", "year": 1955, "foundational": kn.FOUNDATIONAL_BY_AUTHOR,
        "suggested_works": [foundational_work_text(whittle)]})
    book = RetrievedDoc(content="b", metadata={
        "title": "The mathematical theory of infectious diseases", "year": 1975,
        "foundational": kn.FOUNDATIONAL_BY_AUTHOR, "suggested_works": [foundational_work_text(bailey)]})
    added, already, dropped = _foundational_outcomes([bailey, whittle], [note, book], [note, book], [])
    assert added == [
        f"{same} (1962, Bailey) -> The mathematical theory of infectious diseases, by author and year",
        f"{same} (1955, Whittle) -> THE OUTCOME OF A STOCHASTIC EPIDEMIC, by author and year",
    ]
    assert already == [] and dropped == []
    # A suggestion no record answers is dropped even though another record answers its title's twin.
    added, already, dropped = _foundational_outcomes([bailey, whittle], [note], [note], [])
    assert added == [f"{same} (1955, Whittle) -> THE OUTCOME OF A STOCHASTIC EPIDEMIC, by author and year"]
    assert dropped == [f"{same} (1962, Bailey)"]


@pytest.mark.asyncio
async def test_the_run_log_says_which_record_a_suggestion_found_by_author_and_year_resolved_to(tmp_path: Path) -> None:
    eng = _engine(tmp_path)

    async def chat(prompt, node=""):  # noqa: ANN001
        return json.dumps({"works": [INVENTED]})

    eng._chat = chat  # type: ignore[method-assign]
    whittle = RetrievedDoc(content="w", metadata={
        "title": "THE OUTCOME OF A STOCHASTIC EPIDEMIC—A NOTE ON BAILEY'S PAPER", "year": 1955,
        "foundational": kn.FOUNDATIONAL_BY_AUTHOR, "suggested_works": [foundational_work_text(INVENTED)]})

    async def fake_find(suggested, docs):  # noqa: ANN001
        return [whittle]

    eng.knowledge.find_foundational_works = fake_find  # type: ignore[method-assign]
    assert await eng._foundational_works("Some research topic", []) == [whittle]
    outcome = next(ln for ln in (eng.fi_dir / "run.log").read_text(encoding="utf-8").splitlines()
                   if "found in OpenAlex and added" in ln)
    assert ("found in OpenAlex and added (1): A stochastic model for the spread of an infectious disease (1955, Whittle)"
            " -> THE OUTCOME OF A STOCHASTIC EPIDEMIC—A NOTE ON BAILEY'S PAPER, by author and year |") in outcome
    assert "dropped, not found in OpenAlex (0): -" in outcome
