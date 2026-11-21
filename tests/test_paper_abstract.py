"""A scientific paper always has an Abstract: its structure is checked after every draft, and a missing one is written."""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from core import paper_sections as ps
from core.engine import Engine

NO_ABSTRACT = "# A Title\n\n<!-- Keywords: one, two -->\n\n## Introduction\n\nBody text.\n\n## Methods\n\nMore.\n"
ABSTRACT = " ".join(["We measured the thing and report its number."] * 8)


def test_a_paper_without_an_abstract_section_is_missing_one_whatever_it_says() -> None:
    assert ps.missing_abstract(NO_ABSTRACT)
    assert ps.missing_abstract("# T\n\n## Abstract\n\n**Keywords:** a, b\n\n## Introduction\n\nx\n"), "keywords are not a body"
    assert not ps.missing_abstract("# T\n\n## Abstract\n\nSomething was done.\n\n## Introduction\n\nx\n")


def test_an_abstract_goes_under_the_title_with_the_keywords_shown() -> None:
    out = ps.with_abstract(NO_ABSTRACT, "We did it.")
    assert out.index("## Abstract") < out.index("**Keywords:** one, two") < out.index("## Introduction")
    assert "<!--" not in out and not ps.missing_abstract(out)


def _engine(reply: str):
    eng = Engine.__new__(Engine)
    eng._log = logging.getLogger("t")
    eng.config = SimpleNamespace(topic="t")
    eng._prompts = {"write_abstract": __import__("string").Template("$topic $paper")}

    async def chat(prompt, **kw):
        if isinstance(reply, Exception):
            raise reply
        return reply
    eng._chat = chat
    return eng


def test_the_writer_is_asked_for_a_missing_abstract_once_and_it_goes_in() -> None:
    out = asyncio.run(_engine(ABSTRACT)._ensure_abstract({}, NO_ABSTRACT))
    assert not ps.missing_abstract(out) and ABSTRACT.split(".")[0] in out


def test_a_draft_that_has_one_is_not_touched_and_a_failed_ask_keeps_the_draft() -> None:
    has = "# T\n\n## Abstract\n\nSomething was done.\n\n## Introduction\n\nx\n"
    assert asyncio.run(_engine(RuntimeError("down"))._ensure_abstract({}, has)) == has
    assert asyncio.run(_engine(RuntimeError("down"))._ensure_abstract({}, NO_ABSTRACT)) == NO_ABSTRACT
    assert asyncio.run(_engine("too short")._ensure_abstract({}, NO_ABSTRACT)) == NO_ABSTRACT


# A draft whose writer put an Abstract heading with only a note to itself under it before the title, and the real
# Abstract under the title (the order the writer prompt asked for was ambiguous).
NOTE_FIRST = ("## Abstract\n\n<!-- a note the writer left -->\n\n# A Title\n\n## Abstract\n\n" + ABSTRACT
              + "\n\n**Keywords:** one, two\n\n## Introduction\n\nBody.\n")


def test_a_comment_is_not_a_body_and_a_section_ends_at_the_title() -> None:
    assert ps.missing_abstract("# T\n\n## Abstract\n\n<!-- to be written -->\n\n## Introduction\n\nx\n")
    assert ps.missing_abstract("## Abstract\n\n<!-- a\nnote -->\n\n# T\n\n## Introduction\n\nx\n"), "the title ends it"
    assert ps.abstract_of(NOTE_FIRST) == ABSTRACT, "the paper's own Abstract, not the note before the title"


def test_an_abstract_heading_with_no_body_is_taken_out_and_the_real_one_kept() -> None:
    out = ps.without_bodiless_abstract(NOTE_FIRST)
    assert out.startswith("# A Title\n\n## Abstract\n\n" + ABSTRACT) and out.count("## Abstract") == 1
    assert "<!--" not in out and out.endswith("## Introduction\n\nBody.\n")
    assert ps.without_bodiless_abstract(out) == out
    # A paper with one Abstract (or none) is returned as it is, byte for byte.
    for paper in (out, NO_ABSTRACT, "# T\n\n## Abstract\n\nSomething was done.\n\n## Introduction\n\nx\n"):
        assert ps.without_bodiless_abstract(paper) == paper
    # The keywords and FI's markers under an empty heading stay; only the heading and a note go.
    kept = ps.without_bodiless_abstract("# T\n\n## Abstract\n\n<!-- Keywords: a, b -->\n<!-- note -->\n\n## Introduction\n\nx\n")
    assert kept == "# T\n\n<!-- Keywords: a, b -->\n\n## Introduction\n\nx\n"


def test_the_abstract_step_leaves_one_abstract_without_asking_when_the_draft_has_a_real_one() -> None:
    out = asyncio.run(_engine(RuntimeError("must not be asked"))._ensure_abstract({}, NOTE_FIRST))
    assert out == ps.without_bodiless_abstract(NOTE_FIRST) and out.count("## Abstract") == 1
    # Only the empty heading and no real one: it goes, and the Abstract asked for goes under the title once.
    only_note = "## Abstract\n\n<!-- a note -->\n\n# A Title\n\n## Introduction\n\nBody.\n"
    out = asyncio.run(_engine(ABSTRACT)._ensure_abstract({}, only_note))
    assert out.startswith("# A Title\n\n## Abstract\n\n" + ABSTRACT.split(".")[0]) and out.count("## Abstract") == 1
