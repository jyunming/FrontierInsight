"""The foundational works a literature pass retrieved and the literature screen
kept are handed to the writer with a request to cite the ones that bear on the
paper, and the review is told, as advice and never as a forced hit, which of them
the paper does not cite (``core/engine.py``: ``_foundational_write_block``,
``_foundational_review_block``; the write and review nodes).

On ten stored quests of one topic the paper left uncited works that had been
retrieved, kept by the screen and labelled foundational: Kermack and McKendrick
(1927) in one, Whittle (1955) in another, five of seven in a third."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import (
    Engine,
    _foundational_review_block,
    _foundational_sources,
    _foundational_write_block,
    _uncited_foundational_works,
)


def _lit(title: str, year: int, authors: list[str], **meta) -> dict:
    return {
        # An abstract of the length a real record has (the writer marks a shorter one as title-only).
        "content": f"{title}. " + "An abstract that says what the work shows, in about the space a real one takes. " * 2,
        "metadata": {
            "title": title, "year": year, "authors": authors, "source": "openalex",
            "venue": "A Venue", "doi": "10.1/" + re.sub(r"\W+", "-", title.lower())[:40],
            "work_type": "article", **meta,
        },
    }


PRIMER = _lit("A primer on stochastic epidemic models", 2017, ["Linda J. S. Allen"])
KERMACK = _lit(
    "A contribution to the mathematical theory of epidemics", 1927, ["W. O. Kermack", "A. G. McKendrick"],
    foundational="suggested as a foundational work",
)
THRESHOLDS = _lit("Extinction thresholds in deterministic and stochastic epidemic models", 2012, ["Linda J. S. Allen"])
WHITTLE = _lit(
    "The outcome of a stochastic epidemic - a note on Bailey's paper", 1955, ["P. Whittle"],
    foundational="cited by 7 of the retrieved papers",
)
ANDERSON_MAY = _lit(
    "Infectious diseases of humans", 1991, ["R. M. Anderson", "R. M. May"],
    foundational="cited by 10 of the retrieved papers", work_type="book",
)
# Labelled foundational, but a note from another quest an external reader cannot look up.
INTERNAL = _lit("An earlier quest's digest", 2026, ["FI"], foundational="cited by 3 of the retrieved papers",
                kind="fi_digest")
# A web page is Further reading, never a numbered reference.
WEB_PAGE = _lit("A page about SIR models", 2025, [], source="web_search", url="https://example.org/sir",
                foundational="cited by 2 of the retrieved papers")

# The writer labels the papers 1, 2, 3... in this order: PRIMER 1, KERMACK 2, THRESHOLDS 3, WHITTLE 4,
# ANDERSON_MAY 5. INTERNAL and WEB_PAGE are not numbered references.
LITERATURE = [PRIMER, KERMACK, THRESHOLDS, WHITTLE, ANDERSON_MAY, INTERNAL, WEB_PAGE]
NO_FOUNDATIONAL = [PRIMER, THRESHOLDS]


def _labels(works: list) -> list[str]:
    return [label for label, _meta in works]


def test_the_foundational_works_are_the_ones_in_the_citable_set() -> None:
    works = _foundational_sources(LITERATURE, "external")
    assert _labels(works) == ["2", "4", "5"]
    assert [m["title"] for _l, m in works] == [KERMACK["metadata"]["title"], WHITTLE["metadata"]["title"],
                                                ANDERSON_MAY["metadata"]["title"]]
    # An internal audience keeps the internal note, which is a citable entry there.
    assert "6" in _labels(_foundational_sources(LITERATURE, "internal"))


def test_the_write_block_names_the_foundational_works_by_the_writers_labels() -> None:
    block = _foundational_write_block(LITERATURE, "external")
    lines = [ln for ln in block.splitlines() if ln.startswith("- ")]
    assert len(lines) == 3
    assert lines[0] == "- [2] W. O. Kermack & A. G. McKendrick (1927). A contribution to the mathematical theory of epidemics"
    assert lines[1] == ("- [4] P. Whittle (1955). The outcome of a stochastic epidemic - a note on Bailey's paper "
                        "(cited by 7 of the retrieved papers)")
    # A book with no full text is a blurb, and the list says so, as the entry in the block does.
    assert lines[2] == ("- [5] R. M. Anderson & R. M. May (1991). Infectious diseases of humans [short blurb only] "
                        "(book, cited by 10 of the retrieved papers)")
    # The ask: cite what bears on the paper, and only that.
    assert "Cite each one that bears on this paper's claims" in block
    assert "the original paper for a method, model or relation the paper uses" in block
    assert "the standard textbook for the field" in block
    assert "A work that does not bear on the paper is not to be cited just to be cited." in block
    # Not the ordinary papers, the internal note or the web page.
    for left_out in ("primer", "Extinction thresholds", "digest", "SIR models"):
        assert left_out not in block


def test_the_write_block_is_empty_when_there_is_no_foundational_work() -> None:
    assert _foundational_write_block(NO_FOUNDATIONAL, "external") == ""
    assert _foundational_write_block([], "external") == ""
    # A foundational work that is not citable does not count either.
    assert _foundational_write_block([PRIMER, INTERNAL, WEB_PAGE], "external") == ""


def test_the_review_line_lists_the_foundational_works_the_paper_does_not_cite() -> None:
    # Cites [2] (Kermack) and [3]; not [4] Whittle or [5] Anderson & May. The References list
    # is not the text: the numbers in it are not citations.
    paper = "# T\n\nThe model of [2] and the results of [1, 3] hold.\n\n## References\n\n4. Whittle\n5. May\n"
    missing = _uncited_foundational_works(LITERATURE, paper, "external")
    assert _labels(missing) == ["4", "5"]
    block = _foundational_review_block(LITERATURE, paper, "external")
    assert block.startswith("\n\n## Foundational works this paper does not cite (advisory)\n")
    assert "- P. Whittle (1955). The outcome of a stochastic epidemic - a note on Bailey's paper (cited by 7" in block
    # A book with no full text carries the same mark as its entry in the writer's block, and the
    # advisory says what a marked entry may be asked for.
    assert "Infectious diseases of humans [short blurb only] (book, cited by 10 of the retrieved papers)" in block
    assert "ask for it only to say the work exists or for what its title states" in block
    assert "Kermack" not in block, "a cited work is not listed"
    # A reviewer reads a paper that cannot show it a label for a work it does not cite.
    assert "[4]" not in block and "[5]" not in block
    # Advisory: the reviewer is told not to force anything.
    assert "Advisory only" in block
    assert "Do not add anything to `must_flag_hits`" in block
    assert "do not choose `revise` because of this alone" in block


@pytest.mark.parametrize("paper", [
    "# T\n\nCites [2, 4] and [5].\n",       # every foundational work
    "# T\n\nCites [2-5].\n",                 # a range
])
def test_the_review_line_is_empty_when_every_foundational_work_is_cited(paper: str) -> None:
    assert _foundational_review_block(LITERATURE, paper, "external") == ""


def test_the_review_line_is_empty_when_there_is_no_foundational_work() -> None:
    assert _foundational_review_block(NO_FOUNDATIONAL, "# T\n\nNo citations.\n", "external") == ""


def test_the_review_line_is_empty_when_no_paper_was_read() -> None:
    # An unreadable draft cites nothing, and that is not the paper leaving the works out.
    assert _foundational_review_block(LITERATURE, "", "external") == ""
    assert _foundational_review_block(LITERATURE, "  \n", "external") == ""


# ---- through the nodes -------------------------------------------------------

def _engine(tmp_path: Path, *, panel: bool = False) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=False, review_panel=(["methodologist"] if panel else [])),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng._writing_skills_block = lambda state: ""  # type: ignore[method-assign,assignment]
    eng._resolve_write_persona = lambda state: ""  # type: ignore[method-assign,assignment]
    return eng


DRAFT = (
    "# A Study of Stochastic SIR Epidemics\n\n## Abstract\n\nWe study fade-out. **Keywords:** SIR, epidemic, "
    "fade-out, stochastic\n\n## Introduction\n\nThe deterministic model goes back to Kermack and McKendrick [2]. "
    "Fade-out has been analysed before [3] and summarised in a primer [1].\n\n## Results\n\nWe report seeds.\n"
)


def _write(eng: Engine, literature: list, draft: str = DRAFT) -> tuple[str, dict]:
    (eng.quest_root / "paper").mkdir(parents=True, exist_ok=True)
    prompts: list[str] = []

    async def fake_chat(prompt, *, node=None, temperature=None):  # noqa: ANN001
        prompts.append(prompt)
        return draft

    eng._chat = fake_chat  # type: ignore[assignment]
    patch = asyncio.run(eng._node_write({  # type: ignore[arg-type]
        "topic": "t", "title": "t", "iteration": 0, "literature": literature,
    }))
    return prompts[0], patch


def test_the_write_prompt_carries_the_block_and_its_labels_are_the_prior_work_blocks(tmp_path: Path) -> None:
    prompt, _patch = _write(_engine(tmp_path), LITERATURE)
    block = prompt.split("### Foundational works in the prior-work block", 1)[1].split("## Figures available", 1)[0]
    prior_work = prompt.split("## Prior work", 1)[1].split("### Foundational works", 1)[0]
    lines = [ln[2:] for ln in block.splitlines() if ln.startswith("- [")]
    assert len(lines) == 3
    for line in lines:
        # "[4] P. Whittle (1955). The outcome ..." is the header the prior-work block gives that entry.
        header = line.split(" (cited by", 1)[0].split(" (book", 1)[0]
        assert header in prior_work, header
    assert "[4] P. Whittle (1955)" in prior_work
    # The block sits under Prior work, after the entries, before the figures.
    assert prompt.index("## Prior work") < prompt.index("### Foundational works") < prompt.index("## Figures available")


def test_the_write_prompt_is_unchanged_when_there_is_no_foundational_work(tmp_path: Path) -> None:
    prompt, _patch = _write(_engine(tmp_path), NO_FOUNDATIONAL)
    assert "oundational" not in prompt
    # The placeholder leaves nothing behind: the prior-work block runs straight into the next section.
    assert "\n\n## Figures available" in prompt


def test_the_write_node_logs_how_many_foundational_works_the_draft_cites(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _prompt, patch = _write(eng, LITERATURE)
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    line = next(ln for ln in log.splitlines() if "foundational works in the prior-work block" in ln)
    # Kermack [2] is cited; Whittle and Anderson & May are not.
    assert "foundational works in the prior-work block: 3; the draft cites 1; not cited: " in line
    assert WHITTLE["metadata"]["title"] in line and ANDERSON_MAY["metadata"]["title"] in line
    assert KERMACK["metadata"]["title"] not in line.split("not cited: ", 1)[1]
    # And the literature the next pass sees is numbered as the paper is: Kermack was cited first.
    assert patch["literature"][0]["metadata"]["title"] == KERMACK["metadata"]["title"]


def _review(eng: Engine, tmp_path: Path, paper: str, literature: list, reply: dict) -> tuple[dict, list[str]]:
    prompts: list[str] = []

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001
        prompts.append(prompt)
        return json.dumps({"rationale": "fine"} if node == "review_moderator" else reply)

    eng._chat = fake_chat  # type: ignore[assignment]
    path = tmp_path / "paper.md"
    path.write_text(paper, encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(path), "literature": literature,
    }))
    return patch, prompts


ACCEPT = {"verdict": "accept", "score": 4, "suggestions": [], "must_flag_hits": []}
PAPER_CITING_KERMACK = "# T\n\n## Introduction\n\nThe model of Kermack and McKendrick [2] and a primer [1].\n"


@pytest.mark.parametrize("panel", [False, True])
def test_the_review_prompt_carries_the_advisory_line_and_forces_nothing(tmp_path: Path, panel: bool) -> None:
    eng = _engine(tmp_path, panel=panel)
    patch, prompts = _review(eng, tmp_path, PAPER_CITING_KERMACK, LITERATURE, ACCEPT)
    review_prompts = [p for p in prompts if "## Foundational works this paper does not cite (advisory)" in p]
    assert review_prompts, "the advisory line reached the reviewer"
    assert len(review_prompts) == len([p for p in prompts if "## Paper draft" in p])
    for p in review_prompts:
        assert "P. Whittle (1955)" in p and "Infectious diseases of humans" in p
        # Between the figure check and the paper.
        assert p.index("## Figures") < p.index("## Foundational works this paper") < p.index("## Paper draft")
    review = patch["review"]
    # No forced hit, no spent iteration, the reviewer's accept stands.
    assert review["must_flag_hits"] == []
    assert "iteration" not in patch
    assert review["verdict"] == "accept"
    assert eng._route_after_review({"review": review, "iteration": 0}) != "rewrite"  # type: ignore[arg-type]


def test_the_review_prompt_has_no_advisory_line_when_the_paper_cites_them_all(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _patch, prompts = _review(eng, tmp_path, "# T\n\nCites [2], [4] and [5].\n", LITERATURE, ACCEPT)
    assert "Foundational works this paper does not cite" not in prompts[0]
    assert "\n\n## Paper draft" in prompts[0]


def test_the_review_prompt_is_unchanged_without_foundational_works(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _patch, prompts = _review(eng, tmp_path, "# T\n\nCites [1].\n", NO_FOUNDATIONAL, ACCEPT)
    assert "oundational" not in prompts[0]
    assert "\n\n## Paper draft" in prompts[0]
