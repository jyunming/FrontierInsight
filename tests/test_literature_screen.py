"""The literature screen: one batched LLM call grades every retrieved source 0-3.

The embedding relevance floor scores word overlap with the topic, so it keeps
anything that shares the search terms -- a journal's table of contents, a
paper on a different system that uses the same vocabulary. The screen asks
the question that matters: could the paper cite this source for a claim?
Scholarly records need a 2 to stay. Web pages supply quotable text rather
than citations, so they are dropped only at 0. The screen never starves the
quest (it keeps ``relevance_min_keep`` sources, best grades first) and fails
open: a failed or unreadable call keeps everything, and a source the model
did not grade is kept.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from core.engine import Engine, _temperature_for_node
from core.knowledge import WORK_SCOPE_PAPERS, WORK_SCOPE_PAPERS_AND_BOOKS, RetrievedDoc

TOPIC = "How did action figures shape children's play in the 1980s?"


def _engine(tmp_path: Path, *, min_keep: int = 0) -> Engine:
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
    eng.config.knowledge.relevance_min_keep = min_keep
    return eng


def _paper(title: str, doi: str, **md) -> RetrievedDoc:
    return RetrievedDoc(content=f"{title}\n\nAn abstract.", metadata={
        "title": title, "doi": doi, "source": "crossref", "venue": "J. Popular Culture",
        "year": 1999, "work_type": "journal-article", **md})


def _page(title: str, url: str) -> RetrievedDoc:
    return RetrievedDoc(content=f"{title} page text", metadata={
        "title": title, "url": url, "source": "web_search"})


def _grader(grades: dict[int, int], seen: dict | None = None):
    async def chat(prompt, node=""):
        if seen is not None:
            seen["prompt"], seen["node"] = prompt, node
        return json.dumps({"grades": [{"i": i, "grade": g} for i, g in grades.items()]})
    return chat


@pytest.mark.asyncio
async def test_papers_need_a_two_and_web_pages_are_dropped_only_at_zero(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    docs = [_paper("Action figures and play", "10.1/3"), _paper("Toys in the 1980s", "10.1/2"),
            _paper("Plastic polymer moulding", "10.1/1"),
            _page("A collector's blog on He-Man", "https://blog.example/he-man"),
            _page("Buy action figures now", "https://shop.example")]
    eng._chat = _grader({0: 3, 1: 2, 2: 1, 3: 1, 4: 0})
    kept = await eng._screen_literature(TOPIC, docs)
    assert [d.metadata["title"] for d in kept] == [
        "Action figures and play", "Toys in the 1980s", "A collector's blog on He-Man"]
    assert [d.metadata["screen_grade"] for d in kept] == [3, 2, 1]


@pytest.mark.asyncio
async def test_the_screen_never_starves_the_quest(tmp_path: Path) -> None:
    eng = _engine(tmp_path, min_keep=3)
    docs = [_paper(f"Paper {c} on toys and childhood", f"10.1/{c}") for c in "abcd"]
    eng._chat = _grader({0: 1, 1: 0, 2: 2, 3: 1})
    kept = await eng._screen_literature(TOPIC, docs)
    # The one that passed, then the best of the rest, in their original order.
    assert [d.metadata["doi"] for d in kept] == ["10.1/a", "10.1/c", "10.1/d"]


@pytest.mark.parametrize("reply", ["not json at all", '{"verdict": "fine"}', RuntimeError("provider down")])
@pytest.mark.asyncio
async def test_a_failed_or_unreadable_screen_keeps_everything(tmp_path: Path, reply) -> None:
    eng = _engine(tmp_path)
    docs = [_paper("Action figures and play", "10.1/a"), _paper("Plastic moulding", "10.1/b")]

    async def chat(prompt, node=""):
        if isinstance(reply, Exception):
            raise reply
        return reply

    eng._chat = chat
    kept = await eng._screen_literature(TOPIC, docs)
    assert [d.metadata["doi"] for d in kept] == ["10.1/a", "10.1/b"]


@pytest.mark.asyncio
async def test_a_source_the_model_did_not_grade_is_kept(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    docs = [_paper("Plastic moulding", "10.1/a"), _paper("Action figures and play", "10.1/b"),
            _paper("Toys in the 1980s", "10.1/c")]
    eng._chat = _grader({0: 0, 2: -1})  # index 1 missing, index 2 out of range
    kept = await eng._screen_literature(TOPIC, docs)
    assert [d.metadata["doi"] for d in kept] == ["10.1/b", "10.1/c"]


@pytest.mark.asyncio
async def test_a_mapping_reply_is_read_too(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    docs = [_paper("Plastic moulding", "10.1/a"), _paper("Action figures and play", "10.1/b")]

    async def chat(prompt, node=""):
        return '{"grades": {"0": 0, "1": 3}}'

    eng._chat = chat
    kept = await eng._screen_literature(TOPIC, docs)
    assert [d.metadata["doi"] for d in kept] == ["10.1/b"]


@pytest.mark.parametrize("scope,marker", [
    (WORK_SCOPE_PAPERS, "same system"),
    (WORK_SCOPE_PAPERS_AND_BOOKS, "books"),
])
@pytest.mark.asyncio
async def test_the_prompt_carries_the_topic_the_kind_and_each_candidate(tmp_path: Path, scope, marker) -> None:
    eng = _engine(tmp_path)
    seen: dict = {}
    docs = [_paper("Action figures and play", "10.1/a", work_type="book-chapter"),
            _page("A collector's blog", "https://blog.example")]
    eng._chat = _grader({0: 3, 1: 2}, seen)
    await eng._screen_literature(TOPIC, docs, work_scope=scope)
    prompt = seen["prompt"]
    assert seen["node"] == "literature_screen"
    assert TOPIC in prompt
    assert "[0] (paper) Action figures and play" in prompt and "book-chapter" in prompt
    assert "[1] (web page) A collector's blog" in prompt
    assert marker in prompt
    assert "$" not in prompt, "every template slot was filled"


@pytest.mark.asyncio
async def test_papers_the_user_supplied_are_never_screened(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    seen: dict = {}
    own = RetrievedDoc(content="My own PDF", metadata={"title": "My notes on toys", "source": "local_paper"})
    dropped_in = RetrievedDoc(content="A PDF", metadata={"title": "From inputs/papers", "source": "user_supplied"})
    docs = [own, _paper("Plastic moulding", "10.1/a"), dropped_in]
    eng._chat = _grader({0: 0, 1: 0, 2: 0}, seen)
    kept = await eng._screen_literature(TOPIC, docs)
    assert [d.metadata["title"] for d in kept] == ["My notes on toys", "From inputs/papers"]
    assert "My notes on toys" not in seen["prompt"]
    assert "[1] (paper) Plastic moulding" in seen["prompt"]

    async def must_not_call(prompt, node=""):
        raise AssertionError("nothing to grade")

    eng._chat = must_not_call
    assert await eng._screen_literature(TOPIC, [own, dropped_in]) == [own, dropped_in]


def test_the_screen_runs_at_temperature_zero() -> None:
    assert _temperature_for_node("literature_screen") == 0.0


@pytest.mark.asyncio
async def test_no_call_when_the_screen_is_off_or_there_is_nothing_to_grade(tmp_path: Path) -> None:
    eng = _engine(tmp_path)

    async def must_not_call(prompt, node=""):
        raise AssertionError("no screen call expected")

    eng._chat = must_not_call
    assert await eng._screen_literature(TOPIC, []) == []
    eng.config.knowledge.literature_screen = False
    docs = [_paper("Plastic moulding", "10.1/a")]
    assert await eng._screen_literature(TOPIC, docs) == docs


@pytest.mark.asyncio
async def test_the_literature_node_screens_after_the_floor_and_before_full_text(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: None)
    eng = _engine(tmp_path)

    async def no_derivation(prompt, node=""):
        raise RuntimeError("no model")

    retrieved = [_paper("Action figures and play", "10.1/a"), _paper("Plastic moulding", "10.1/b")]
    order: list[str] = []

    async def fake_search(query, **kw):  # noqa: ANN001
        return retrieved

    def floor(topic, docs, *, stats=None):  # noqa: ANN001
        order.append("floor")
        return docs

    async def screen(topic, docs, **kw):  # noqa: ANN001
        order.append("screen")
        assert kw.get("work_scope") == WORK_SCOPE_PAPERS_AND_BOOKS
        return docs[:1]

    fetched: list = []

    async def fetch(docs):  # noqa: ANN001
        order.append("fetch")
        fetched.extend(d.metadata["doi"] for d in docs)
        return docs

    eng._chat = no_derivation
    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    eng.knowledge.fetch_full_text = fetch  # type: ignore[method-assign]
    monkeypatch.setattr(eng, "_filter_docs_by_relevance", floor)
    monkeypatch.setattr(eng, "_screen_literature", screen)
    patch = await eng._node_literature({"topic": TOPIC, "chosen_idea": {"title": "T"},
                                        "no_simulation_resolved": True})
    assert order == ["floor", "screen", "fetch"]
    assert fetched == ["10.1/a"]
    assert [e["metadata"]["doi"] for e in patch["literature"]] == ["10.1/a"]
