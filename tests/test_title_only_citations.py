"""A source FI holds only the title of cannot back what a sentence says beyond that title.

A record with nothing but its title contains no statement, so a sentence that cites only such
records and says more than the title names is unsupported, whatever the claim check made of it.
On the 72 stored SIR papers 9 sentences were of that kind: the check had marked 6 unsupported and
had not listed the other 3 as claims at all; the graded D1 loss of one final draft was one of the
3. The review's advisory about foundational works still not cited carries the same mark the
writer's prior-work block has, so the reviewer does not ask for a title-only record to back a
finding.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import (
    Engine,
    _apply_title_only_rule,
    _foundational_review_block,
    _same_statement,
    _title_only_sentences,
)

AB_TITLE = "Stochastic Epidemic Models and Their Statistical Analysis"
LONG = "An abstract that says what the work shows, in about the space a real one takes. " * 3


def _source(title: str, text: str, *, authors: list[str], work_type: str = "article") -> tuple[dict[str, Any], str]:
    return {"title": title, "authors": authors, "work_type": work_type, "year": 2000}, text


SOURCES = {
    "1": _source("A contribution to the mathematical theory of epidemics", "A contribution. " + LONG,
                 authors=["W. O. Kermack", "A. G. McKendrick"]),
    "2": _source(AB_TITLE, AB_TITLE, authors=["Håkan Andersson", "Tom Britton"], work_type="book"),
    # A book with a description: a blurb the claim check can read, not a bare title.
    "3": _source("Mathematical Epidemiology of Infectious Diseases", "Mathematical Epidemiology of Infectious Diseases. "
                 + "This book describes the model building and analysis of infectious disease spread. " * 4,
                 authors=["Odo Diekmann"], work_type="book"),
    "4": _source("Extinction times of epidemic processes", "Extinction times of epidemic processes",
                 authors=["Ingemar Nåsell"]),
}

SPECIFIC = ("Random extinction of early infections is captured by branching process theory, whose outbreak "
            "probability equals one minus the inverse reproduction number [2].")
TWO_TITLES = ("In small populations extinction probabilities depend strongly on the initial number of "
              "infected individuals [2, 4].")
ATTRIBUTION = "Andersson and Britton wrote a book on stochastic epidemic models [2]."
FULL_TEXT = "The final size of an epidemic is set by the reproduction number and the threshold [1]."
MIXED = ("Branching process theory gives the outbreak probability of a stochastic epidemic exactly [1, 2].")
BLURB = "Epidemic thresholds for heterogeneous populations have been studied extensively [3]."
PAPER = (
    "# T\n\n## Introduction\n\n"
    f"{FULL_TEXT} {SPECIFIC} {ATTRIBUTION}\n\n"
    f"{BLURB} {TWO_TITLES} {MIXED}\n\n"
    "## References\n\n1. Kermack (1927). A contribution.\n"
)


def _flagged(paper: str = PAPER) -> dict[str, tuple[list[str], list[str]]]:
    return {s: (labels, words) for s, labels, words in _title_only_sentences(paper, SOURCES)}


# --- which sentences ---------------------------------------------------------------------

def test_a_sentence_that_cites_only_title_only_records_and_says_more_than_the_title_is_found() -> None:
    found = _flagged()
    assert set(found) == {SPECIFIC, TWO_TITLES}
    labels, words = found[SPECIFIC]
    assert labels == ["2"] and {"branch", "outbre", "probab"} <= set(words) and len(words) >= 4
    assert found[TWO_TITLES][0] == ["2", "4"]


def test_attribution_full_text_a_blurb_and_a_mixed_citation_are_not_found() -> None:
    found = _flagged()
    assert ATTRIBUTION not in found  # says who wrote a book on what the title names: one or two new words
    assert FULL_TEXT not in found  # a source with an abstract
    assert BLURB not in found  # a book's description is text the check can read
    assert MIXED not in found  # one of its two sources has text


def test_a_paper_with_no_title_only_record_or_no_citation_finds_nothing() -> None:
    assert _title_only_sentences(PAPER, {k: v for k, v in SOURCES.items() if k in ("1", "3")}) == []
    assert _title_only_sentences("# T\n\n## Introduction\n\nNo citation here at all, only words and more words.\n",
                                 SOURCES) == []
    # The reference list is not the paper's text.
    assert _title_only_sentences("# T\n\n## References\n\nA sentence about branching process theory [2].\n",
                                 SOURCES) == []


def test_a_sentence_is_found_once_however_often_it_is_written() -> None:
    twice = PAPER.replace("## References", f"{SPECIFIC}\n\n## References")
    assert [s for s, _l, _w in _title_only_sentences(twice, SOURCES)].count(SPECIFIC) == 1


def test_the_same_statement_needs_shared_words_not_just_a_short_title() -> None:
    claim = "Small populations may still experience minor outbreaks because of random fluctuations in contact"
    assert _same_statement(claim, claim + ".")
    assert not _same_statement(claim, "Outbreaks in Small Populations")  # a three-word title
    assert not _same_statement(claim, "Stochastic models are needed for small populations")


# --- what becomes of the claims -----------------------------------------------------------

def _claim(text: str, basis: str, cite: Any = None, evidence: str = "") -> dict[str, Any]:
    return {"claim": text, "basis": basis, "citation_index": cite, "quote": AB_TITLE if basis == "citation" else "",
            "evidence": evidence}


def test_a_claim_the_check_grounded_in_a_title_only_record_becomes_unsupported() -> None:
    claims = [_claim(FULL_TEXT, "citation", 1), _claim(SPECIFIC, "citation", 2, "the source is on this")]
    out, changed = _apply_title_only_rule(PAPER, SOURCES, claims)
    assert changed == 2  # the specific sentence, and the two-title sentence the check never listed
    grounded = {c["claim"]: c for c in out}
    assert grounded[FULL_TEXT]["basis"] == "citation"  # untouched
    flipped = grounded[SPECIFIC]
    assert flipped["basis"] == "unsupported" and flipped["quote"] == ""
    assert f"[2] is held as its title only ({AB_TITLE!r})" in flipped["evidence"]
    assert "branch" in flipped["evidence"]
    assert claims[1]["basis"] == "citation"  # the caller's claims are not changed in place


def test_a_sentence_the_check_never_listed_is_added_as_unsupported() -> None:
    out, changed = _apply_title_only_rule(PAPER, SOURCES, [_claim(FULL_TEXT, "citation", 1)])
    assert changed == 2
    added = [c for c in out if c["claim"] in (SPECIFIC, TWO_TITLES)]
    assert {c["claim"] for c in added} == {SPECIFIC, TWO_TITLES}
    assert all(c["basis"] == "unsupported" and c["quote"] == "" for c in added)
    assert {c["claim"]: c["citation_index"] for c in added} == {SPECIFIC: 2, TWO_TITLES: 2}
    assert "[2] is held as its title only" in added[0]["evidence"]


def test_an_unsupported_claim_and_one_the_run_grounds_are_left_alone() -> None:
    claims = [
        _claim(SPECIFIC, "unsupported", 2, "no text was provided for that source"),
        _claim(TWO_TITLES, "experiment", None, "the simulation's extinction probabilities"),
    ]
    out, changed = _apply_title_only_rule(PAPER, SOURCES, claims)
    assert changed == 0 and out == claims


def test_nothing_changes_without_a_title_only_record() -> None:
    claims = [_claim(FULL_TEXT, "citation", 1)]
    out, changed = _apply_title_only_rule(PAPER, {"1": SOURCES["1"], "3": SOURCES["3"]}, claims)
    assert changed == 0 and out is claims


# --- in the claim check -------------------------------------------------------------------

def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="stochastic epidemics", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, claim_grounding=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


def _literature() -> list[dict[str, Any]]:
    out = []
    for n in ("1", "2", "3", "4"):
        meta, text = SOURCES[n]
        out.append({"content": text, "metadata": {**meta, "doi": f"10.1/{n}", "source": "openalex", "venue": "V"}})
    return out


def test_the_claim_check_marks_a_title_only_citation_unsupported_and_says_so(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    paper = tmp_path / "paper" / "paper.md"
    paper.parent.mkdir(parents=True)
    paper.write_text(PAPER, encoding="utf-8")
    # The check grounds the specific sentence in the book by quoting its title: 25 characters or more,
    # and in the source's text, so nothing in the check itself refuses it.
    async def fake(prompt, *, node=None):  # noqa: ANN001, ARG001
        return json.dumps({"claims": [
            {"claim": FULL_TEXT, "basis": "citation", "citation_index": 1,
             "quote": "An abstract that says what the work shows", "evidence": "abstract"},
            {"claim": SPECIFIC, "basis": "citation", "citation_index": 2, "quote": AB_TITLE, "evidence": "the book"},
        ], "summary": "ok"})

    eng._chat = fake  # type: ignore[method-assign]
    logged: list[str] = []
    eng._log = type("L", (), {  # type: ignore[assignment]
        "info": lambda self, m, *a: logged.append(m % a if a else m),
        "warning": lambda self, m, *a: logged.append(m % a if a else m),
    })()
    out = asyncio.run(eng._node_claim_check({"topic": "t", "paper_md": str(paper), "literature": _literature()}))  # type: ignore[arg-type]
    grounding = out["claim_grounding"]
    assert SPECIFIC in grounding["unsupported"] and TWO_TITLES in grounding["unsupported"]
    assert FULL_TEXT not in grounding["unsupported"]
    assert grounding["total"] == 3 and grounding["grounded"] == 1
    assert any("2 claim(s) rest on a source held as its title only" in m for m in logged), logged
    ledger = json.loads((tmp_path / "paper" / "claims.json").read_text(encoding="utf-8"))
    assert ledger["unsupported"] == grounding["unsupported"]
    assert "held as its title only" in (tmp_path / "paper" / "CLAIMS.md").read_text(encoding="utf-8")


# --- the review's advisory ------------------------------------------------------------------

def _foundational(title: str, text: str, year: int, authors: list[str], **meta: Any) -> dict[str, Any]:
    return {"content": text, "metadata": {
        "title": title, "year": year, "authors": authors, "source": "openalex", "venue": "V",
        "doi": "10.1/" + title[:8].lower().replace(" ", "-"), "work_type": "book",
        "foundational": "cited by 6 of the retrieved papers", **meta,
    }}


def test_the_review_advisory_marks_a_title_only_record_and_asks_only_for_what_its_title_states() -> None:
    lit = [
        _foundational("A primer", LONG, 2017, ["L. Allen"], work_type="article"),
        _foundational(AB_TITLE, AB_TITLE, 2000, ["H. Andersson", "T. Britton"]),
    ]
    block = _foundational_review_block(lit, "# T\n\nCites [1].\n", "external")
    assert "## Foundational works this paper does not cite (advisory)" in block
    assert f"H. Andersson & T. Britton (2000). {AB_TITLE} [title only]" in block
    assert "ask for it only to say the work exists or for what its title states" in block
    assert "never to back a finding, a number or a mechanism" in block
    # The advice is still advisory: nothing is forced.
    assert "Do not add anything to `must_flag_hits`" in block


def test_the_review_advisory_has_no_caution_when_every_record_has_text() -> None:
    lit = [_foundational("A treatise on epidemics", LONG, 1950, ["M. Bartlett"], work_type="article")]
    block = _foundational_review_block(lit, "# T\n\nNothing cited.\n", "external")
    assert "[title only]" not in block and "short blurb only" not in block and "ask for it only to say" not in block
