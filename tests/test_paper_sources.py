"""The engine writes a paper's source lists (``core/engine.py:_finalize_paper_sources``).

The writer cites inline by the prior-work block's labels and writes no list.
The engine removes any References or Further reading section the writer put
in anyway, numbers the scholarly sources the text cites 1, 2, 3… in the order
it first cites them, rewrites the citations to match, lists exactly those
sources under ``## References`` and every web page under ``## Further
reading``, and reorders the quest's literature so the claim check, the
slides, the poster, the bib and the next write pass number the same way.

The validation quests showed why: the writer's own list named 8 sources the
text never cited while two cited numbers pointed at nothing in it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import (
    Engine,
    _finalize_paper_sources,
    _format_lit_from_state,
    _renumber_citations,
    _strip_source_lists,
    build_references,
    cited_references,
)
from core.numeric_oracle import extract_paper_numbers
from generation._keywords import keep_one_keywords_form

TOPIC = "How fast do stochastic epidemics die out?"


def _paper(title: str, doi: str) -> dict:
    return {"content": f"{title}. An abstract.", "metadata": {
        "source": "crossref", "title": title, "doi": doi,
        "authors": ["A. Author"], "year": 2020, "venue": "J. Tests"}}


def _web(title: str, url: str) -> dict:
    return {"content": f"{title} page text", "metadata": {
        "source": "web_search", "kind": "web_page", "title": title, "url": url,
        "site": url.split("/")[2]}}


LIT = [
    _paper("Alpha", "10.1/a"),                   # [1]
    _web("Page one", "https://one.example/p"),   # [W1]
    _paper("Beta", "10.1/b"),                    # [2]
    _paper("Gamma", "10.1/c"),                   # [3]
    _paper("Delta", "10.1/d"),                   # [4]
    _web("Page two", "https://two.example/p"),   # [W2]
]


def _section(md: str, heading: str) -> list[str]:
    body = md.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    return [line for line in body.splitlines() if line.strip()]


# --- numbering ----------------------------------------------------------------

def test_cited_papers_are_numbered_in_the_order_the_text_first_cites_them() -> None:
    md = (
        "# T\n\n## Introduction\nDelta showed it [4]. Beta and Alpha agree [2, 1]; a page says so [W2].\n"
        "Delta again [4].\n\n## References\n1. Alpha.\n2. Beta.\n3. Gamma.\n4. Delta.\n"
    )
    out, ordered, dropped = _finalize_paper_sources(md, LIT, "external")
    assert "Delta showed it [1]. Beta and Alpha agree [2, 3]; a page says so [W2].\nDelta again [1]." in out
    assert _section(out, "References") == [
        "1. A. Author (2020). Delta. J. Tests. DOI: 10.1/d",
        "2. A. Author (2020). Beta. J. Tests. DOI: 10.1/b",
        "3. A. Author (2020). Alpha. J. Tests. DOI: 10.1/a",
    ], "only the cited papers, in citation order; the writer's list is gone"
    assert _section(out, "Further reading") == [
        "- [W1] Page one. https://one.example/p",
        "- [W2] Page two. https://two.example/p",
    ]
    assert out.count("## References") == 1 and out.index("## References") < out.index("## Further reading")
    assert dropped == []
    # The literature now numbers the same way, so every later output agrees.
    assert [r["title"] for r in build_references(ordered)] == ["Delta", "Beta", "Alpha", "Gamma"]
    assert [r["n"] for r in cited_references(ordered, out)] == [1, 2, 3]
    assert "[1] A. Author (2020). Delta" in _format_lit_from_state({"literature": ordered})


def test_three_or_more_numbers_in_a_row_become_a_range() -> None:
    lit = [_paper(t, f"10.1/{t}") for t in ("a", "b", "c", "d", "e")]
    # [5] is cited first, so old 5 is new 1 and old 1–4 are new 2–5.
    out, _mapping, _dropped = _renumber_citations("See [5]. All of them [1, 2, 3, 4, 5]. Some [2-4].", 5)
    assert out == "See [1]. All of them [1–5]. Some [3–5]."
    assert _finalize_paper_sources("See [3].\n", lit, "external")[1][0]["metadata"]["title"] == "c"


def test_a_citation_of_a_source_that_does_not_exist_is_removed() -> None:
    out, _ordered, dropped = _finalize_paper_sources("# T\n\nClaim [2, 9]. Other [9]. End.\n", LIT, "external")
    assert "Claim [1]. Other. End." in out
    assert dropped == [9]


def test_brackets_that_are_not_citations_stay() -> None:
    body = (
        "Values lie in [0, 1]. The data span [1990–2020]. The code reads `x[2]`, the model "
        "$a_{[3]}$, and a [4](https://link.example) link. A real citation [3]."
    )
    out, mapping, dropped = _renumber_citations(body, 4)
    assert out == body.replace("citation [3]", "citation [1]")
    assert mapping == {3: 1} and dropped == []


def test_no_citations_means_no_references_section() -> None:
    md = "# T\n\nNo sources here.\n\n## References\n1. Invented (2020).\n"
    out, _ordered, _dropped = _finalize_paper_sources(md, LIT, "external")
    assert "## References" not in out and "Invented" not in out
    assert "## Further reading" in out, "web pages are listed whether or not they are cited"
    assert _finalize_paper_sources("# T\n\nNothing.\n", [], "external")[0] == "# T\n\nNothing.\n"


# --- the writer's lists go ------------------------------------------------------

def test_the_writers_lists_go_under_any_usual_heading_and_later_sections_stay() -> None:
    md = (
        "# T\n\nBody [1].\n\n## Bibliography\n1. Made up.\n\n### Primary\n2. More.\n\n"
        "## Appendix A\nKept.\n\n## Further Reading\n- an invented page\n"
    )
    assert _strip_source_lists(md) == "# T\n\nBody [1].\n\n## Appendix A\nKept.\n"
    assert _strip_source_lists("# T\n\n### Works Cited\n1. X.\n") == "# T\n"


# --- the write node ------------------------------------------------------------

def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic=TOPIC, title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=["paper_md"]),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    return eng


def test_the_write_node_writes_the_lists_and_returns_the_reordered_literature(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    seen: dict[str, str] = {}

    async def chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        seen["prompt"] = prompt
        return (
            "# Extinction Times\n\n<!-- Keywords: SIR, extinction -->\n\n## Abstract\nShort [3].\n\n"
            "**Keywords:** SIR, extinction\n\n## Results\nGamma [3] and Alpha [1].\n\n"
            "## References\n1. Somebody (1999). Not a source here.\n"
        )

    eng._chat = chat  # type: ignore[method-assign]
    out = asyncio.run(eng._node_write({"topic": TOPIC, "literature": LIT}))  # type: ignore[arg-type]
    text = Path(out["paper_md"]).read_text(encoding="utf-8")
    assert "Short [1]." in text and "Gamma [1] and Alpha [2]." in text
    assert _section(text, "References") == [
        "1. A. Author (2020). Gamma. J. Tests. DOI: 10.1/c",
        "2. A. Author (2020). Alpha. J. Tests. DOI: 10.1/a",
    ]
    assert "Somebody" not in text
    assert "<!-- Keywords" not in text and "**Keywords:** SIR, extinction" in text
    assert [r["title"] for r in build_references(out["literature"])] == ["Gamma", "Alpha", "Beta", "Delta"]
    # The prompt tells the writer the engine writes the lists.
    assert "Do not write a `## References` or a `## Further reading` section." in seen["prompt"]
    assert "End with `## References`" not in seen["prompt"]


# --- keywords -------------------------------------------------------------------

BOTH = "# T\n\n<!-- Keywords: a, b -->\n\n## Abstract\nText.\n\n**Keywords:** a, b\n\n## Introduction\nBody.\n"


def test_a_paper_that_shows_its_keywords_keeps_only_the_line() -> None:
    assert keep_one_keywords_form(BOTH, visible=True) == (
        "# T\n\n## Abstract\nText.\n\n**Keywords:** a, b\n\n## Introduction\nBody.\n"
    )


def test_a_paper_that_hides_its_keywords_keeps_only_the_comment() -> None:
    assert keep_one_keywords_form(BOTH, visible=False) == (
        "# T\n\n<!-- Keywords: a, b -->\n\n## Abstract\nText.\n\n## Introduction\nBody.\n"
    )


def test_one_keywords_form_is_left_alone() -> None:
    one = "# T\n<!-- Keywords: a -->\n\n## Summary\nKeywords: body text.\n"
    assert keep_one_keywords_form(one, visible=True) == one


# --- the numeric check ------------------------------------------------------------

def test_the_numeric_check_does_not_read_the_source_lists() -> None:
    md = (
        "# T\n\n## Results\nThe outbreak probability was 0.4213.\n\n## References\n\n"
        "1. van der Berg, K. (2016). A study of 12.75 outbreaks. J. X. DOI: 10.1/2.3456\n\n"
        "## Further reading\n\n- [W1] A page about 37.25 cases. https://x.example/a\n"
    )
    assert [value for value, _token, _context in extract_paper_numbers(md)] == [0.4213]
