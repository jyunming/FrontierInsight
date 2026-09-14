"""The same work reached under several identities is one source.

A real quest cited "Numerical Integration 4th-order Runge–Kutta Method" three
times: the same textbook appendix printed in three books, each with its own
DOI, so DOI-based dedup kept all three and they took three of eight reference
slots. OpenAlex also returns some arXiv preprints twice, once with a DOI and
once without, with a literal backslash-n in one title. A normalized title is
the only identity those copies share. Generic titles ("Introduction") must not
become identities, or different books' chapters would merge.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import core.knowledge as K
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    ProviderConfig,
)
from core.engine import Engine, build_references
from core.knowledge import RetrievedDoc, _doc_dedup_keys, _normalize_title

RK4 = "Numerical Integration 4th-order Runge–Kutta Method"


# --- normalization ------------------------------------------------------------

@pytest.mark.parametrize("variant", [
    "Numerical Integration 4th-order Runge-Kutta Method",      # hyphen
    "Numerical Integration 4th‑order Runge–Kutta Method",      # non-breaking hyphen, en dash
    "NUMERICAL INTEGRATION 4TH-ORDER RUNGE—KUTTA METHOD",      # case, em dash
    "Numerical Integration\\n 4th-order Runge–Kutta Method",   # OpenAlex literal backslash-n
    "Numerical Integration 4th-order Runge&ndash;Kutta Method", # HTML entity
])
def test_copies_of_one_title_normalize_identically(variant: str) -> None:
    assert _normalize_title(variant) == _normalize_title(RK4) != ""


@pytest.mark.parametrize("generic", [
    "Introduction", "Preface", "Book Review", "References", "Harmonic Oscillator", "",
])
def test_generic_or_short_titles_are_not_identities(generic: str) -> None:
    assert _normalize_title(generic) == ""


def test_doc_keys_carry_both_the_id_and_the_title() -> None:
    d = RetrievedDoc(content="", metadata={"doi": "10.1016/b978-1", "title": RK4})
    keys = _doc_dedup_keys(d)
    assert keys[0] == "doi:10.1016/b978-1"
    assert keys[1] == f"title:{_normalize_title(RK4)}"


# --- where it applies -------------------------------------------------------------

def _doc(doi: str, title: str, source: str = "crossref") -> RetrievedDoc:
    return RetrievedDoc(content=title, metadata={"doi": doi, "title": title, "source": source})


def test_external_merge_collapses_one_appendix_under_three_dois() -> None:
    def _a(query, top_k, *, timeout_s=10.0):
        return [_doc("10.1016/b978-1-78548-182-6.50012-9", RK4),
                _doc("10.1016/b978-1-78548-179-6.50010-5", "Numerical Integration 4th-order Runge-Kutta Method")]

    def _b(query, top_k, *, timeout_s=10.0):
        return [_doc("10.1016/b978-1-78548-186-4.50009-7", RK4.upper()),
                _doc("10.1000/other", "Symplectic integrators for the damped oscillator")]

    K._SOURCE_REGISTRY["_ta"] = _a  # type: ignore[assignment]
    K._SOURCE_REGISTRY["_tb"] = _b  # type: ignore[assignment]
    try:
        docs = asyncio.run(K._route_external("q", 10, ["_ta", "_tb"]))
    finally:
        K._SOURCE_REGISTRY.pop("_ta", None)
        K._SOURCE_REGISTRY.pop("_tb", None)
    titles = [d.metadata["title"] for d in docs]
    assert len(titles) == 2, titles
    assert titles[0] == RK4, "first source still wins a collision"


def test_external_merge_keeps_different_books_introductions_apart() -> None:
    def _a(query, top_k, *, timeout_s=10.0):
        return [_doc("10.1/a", "Introduction"), _doc("10.1/b", "Introduction")]

    K._SOURCE_REGISTRY["_tc"] = _a  # type: ignore[assignment]
    try:
        docs = asyncio.run(K._route_external("q", 10, ["_tc"]))
    finally:
        K._SOURCE_REGISTRY.pop("_tc", None)
    assert len(docs) == 2


def test_references_list_one_entry_per_work() -> None:
    lit = [{"content": "", "metadata": {"doi": d, "title": t, "source": "crossref"}}
           for d, t in (("10.1/x", RK4), ("10.1/y", RK4.replace("–", "-")),
                        ("10.1/z", "A different study of symplectic integration"))]
    refs = build_references(lit, audience="external")
    assert [r["n"] for r in refs] == [1, 2]
    assert refs[1]["title"].startswith("A different study")


@pytest.mark.asyncio
async def test_literature_node_merge_drops_a_same_title_copy(tmp_path: Path) -> None:
    cfg = Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
        pauses={"papers": False},  # about dedup, not the paywalled-paper pause
    )
    eng = Engine(cfg)

    async def fake_search(query, **kw):  # noqa: ANN001
        return [
            RetrievedDoc(content="a", metadata={"doi": "10.1/one", "title": RK4}),
            RetrievedDoc(content="b", metadata={"doi": "10.1/two", "title": RK4.lower()}),
            RetrievedDoc(content="c", metadata={"doi": "10.1/three", "title": "Energy drift in symplectic schemes"}),
        ]

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    patch = await eng._node_literature({"topic": "t", "chosen_idea": {"title": "T"}})
    dois = [e["metadata"]["doi"] for e in patch["literature"]]
    assert dois == ["10.1/one", "10.1/three"]
