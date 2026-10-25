"""A round that answers a review or a person's notes writes only the sections it touches.

The paper writer used to get the review and one instruction, to write the whole paper again, and the earlier draft was not
in its prompt: text that was fine was lost and new faults appeared. Now FI decides which sections the round touches (a
check quotes or locates a passage in them, the request names them, or they cite a value of FI's record that changed), the
writer is shown the earlier draft read-only and writes only those sections, and FI puts them back by heading: every other
section stays as it was. A request FI cannot place in a section, and a first draft, still write the whole paper.

The chat client is a fake that records each call. No model runs.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from core import paper_scope
from tests.test_write_patch import CLAIM, DRAFT, _engine, _first_draft, _revise_state, _study

DISCUSSION_HIT = "single_point_eval: the Discussion does not say what limits the gap between the two curves"
NEW_DISCUSSION = "## Discussion\nThe gap closes as the population grows, because early extinction matters less [2].\n"


def _body(text: str) -> str:
    return text.split("## References")[0]


def _section(text: str, title: str) -> str:
    sections = paper_scope.split(_body(text))
    [s] = [s for s in sections if s.title == title]
    return paper_scope.text_of(_body(text), s)


def _revise(eng: Any, out: dict[str, Any], hits: list[str], **extra: Any) -> str:
    result = asyncio.run(eng._node_write(_revise_state(out, hits, **extra)))
    return Path(result["paper_md"]).read_text(encoding="utf-8")


def _info(logged: list[tuple[int, str]]) -> list[str]:
    return [m for lvl, m in logged if lvl == logging.INFO]


# --- a hit on one section: that section is written again, from the earlier draft -----------------------------------

def test_a_hit_on_one_section_asks_for_that_section_and_shows_the_old_draft(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, NEW_DISCUSSION])
    out1, before = _first_draft(eng)
    after = _revise(eng, out1, [DISCUSSION_HIT])

    assert client.nodes == ["write", "write"], "no call for edits and no second whole paper"
    prompt = client.calls[1][1]
    assert "<earlier_draft>" in prompt and CLAIM in prompt, "the earlier draft is in the prompt, as context"
    assert "1. `## Discussion`" in prompt and "2. `" not in prompt, "one section is asked for"
    assert "write only these sections" in prompt
    assert "<earlier_draft>" not in client.calls[0][1], "a first draft is written whole, with no earlier draft to show"
    for title in ("Abstract", "Introduction", "Results"):
        assert _section(after, title) == _section(before, title), f"{title} is as it was"
    assert "early extinction matters less" in _section(after, "Discussion")
    assert "A larger population would shrink" not in after
    assert any("writing 1 of 4 section(s) again (Discussion)" in m for m in _info(logged))


def test_a_reply_that_returns_a_section_not_asked_for_has_it_dropped(tmp_path: Path) -> None:
    reply = NEW_DISCUSSION + "\n## Results\nA different results section that nobody asked for.\n"
    eng, client, logged = _engine(tmp_path, [DRAFT, reply])
    out1, before = _first_draft(eng)
    after = _revise(eng, out1, [DISCUSSION_HIT])

    assert client.nodes == ["write", "write"], "dropped, not asked again"
    assert _section(after, "Results") == _section(before, "Results")
    assert "nobody asked for" not in after
    assert "early extinction matters less" in after
    assert any('also gave "Results"' in m and "not asked for" in m for m in _info(logged))


def test_a_reply_without_the_asked_section_writes_the_whole_paper(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, "## Results\nOnly another section.\n", DRAFT])
    out1, _before = _first_draft(eng)
    _revise(eng, out1, [DISCUSSION_HIT])
    assert client.nodes == ["write", "write", "write"]
    assert any("the sections could not be put back" in m and "Discussion" in m
               for lvl, m in logged if lvl == logging.WARNING)


def test_a_request_that_names_no_section_writes_the_whole_paper_and_says_so(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, DRAFT])
    out1, _before = _first_draft(eng)
    _revise(eng, out1, ["single_point_eval: only one sample size was tried"])
    assert client.nodes == ["write", "write"]
    assert "<earlier_draft>" not in client.calls[1][1]
    assert any("writing the whole paper: FI cannot place in a section" in m for m in _info(logged))


def test_a_first_draft_is_written_whole(tmp_path: Path) -> None:
    eng, client, _logged = _engine(tmp_path, [DRAFT])
    _first_draft(eng)
    assert client.nodes == ["write"]
    assert "write only these sections" not in client.calls[0][1] and "DESIGN-SENTINEL" in client.calls[0][1]


# --- a value of FI's record that changed: the sections that cite it ------------------------------------------------

def test_a_changed_value_of_fis_record_asks_for_the_sections_citing_it(tmp_path: Path) -> None:
    draft = DRAFT.replace("## Introduction", "## Methods\nEach setting ran for {{run_time}}.\n\n## Introduction", 1)
    eng, client, logged = _engine(tmp_path, [draft, "## Methods\nEach setting ran in the time FI measured.\n"])
    out1 = asyncio.run(eng._node_write(_study(exec_result={"duration_s": 12.0})))
    before = Path(out1["paper_md"]).read_text(encoding="utf-8")
    assert "{{run_time}}" not in before
    state = _revise_state(out1, [], exec_result={"duration_s": 95.0})
    state["review"] = {"verdict": "revise", "must_flag_hits": []}
    result = asyncio.run(eng._node_write(state))
    after = Path(result["paper_md"]).read_text(encoding="utf-8")

    assert client.nodes == ["write", "write"]
    assert "1. `## Methods`" in client.calls[1][1] and "2. `" not in client.calls[1][1]
    assert "value of FI's record" in client.calls[1][1]
    for title in ("Abstract", "Introduction", "Results", "Discussion"):
        assert _section(after, title) == _section(before, title)
    assert "in the time FI measured" in after


def test_changed_names_the_sections_whose_filled_values_differ() -> None:
    before = "# T\n\n## A\nOne 12 s.\n\n## B\nNo value.\n\n## C\nTwo 12 s.\n"
    after = before.replace("12 s", "95 s")
    assert paper_scope.changed(before, after) == {1, 3}
    assert paper_scope.changed(before, before) == set()
    assert paper_scope.changed(before, before.replace("## B", "## Z")) is None, "other headings: nothing to compare"


# --- a page limit: the longest sections, each cut by a share ---------------------------------------------------------

def test_a_page_limit_asks_for_the_longest_sections_with_a_length_to_cut_to(tmp_path: Path) -> None:
    filler = " ".join(["word"] * 400)
    long_draft = DRAFT.replace("## Discussion\n", f"## Discussion\n{filler} {filler[:800]}.\n\n").replace(
        "## Introduction\n", f"## Introduction\n{filler}.\n\n")
    reply = "## Discussion\nShort discussion [1].\n"
    eng, client, _ = _engine(tmp_path, [long_draft, reply])
    out1, before = _first_draft(eng)
    after = _revise(eng, out1, ["over_page_limit: the rendered PDF is 5 pages; the limit is 4. Cut about 100 words"])
    prompt = client.calls[1][1]
    assert client.nodes == ["write", "write"]
    assert "1. `## Discussion` (about " in prompt and "2. `" not in prompt, "the longest section gives the words"
    assert "over the page limit" in prompt
    assert _section(after, "Results") == _section(before, "Results")
    assert _section(after, "Abstract") == _section(before, "Abstract"), "the abstract is never cut"


# --- the module ------------------------------------------------------------------------------------------------------

def test_splice_keeps_every_other_byte_and_drops_unasked_sections() -> None:
    paper = "# T\n\n## A\nalpha\n\n### sub\nx\n\n## B\nbeta\n\n## C\ngamma\n"
    sections = paper_scope.split(paper)
    assert [s.title for s in sections] == ["", "A", "B", "C"]
    spliced = paper_scope.splice(paper, sections, {1}, "## A\nnew alpha\n\n### sub\ny\n\n## C\nsneaky\n")
    assert spliced.text == "# T\n\n## A\nnew alpha\n\n### sub\ny\n\n## B\nbeta\n\n## C\ngamma\n"
    assert spliced.written == ["A"] and spliced.ignored == ["C"]


def test_a_heading_inside_a_code_fence_is_not_a_section() -> None:
    paper = "# T\n\n## A\n```\n## not a heading\n```\ntext\n\n## B\nb\n"
    assert [s.title for s in paper_scope.split(paper)] == ["", "A", "B"]


def test_naming_reads_the_heading_as_written_or_a_quoted_passage() -> None:
    paper = "# T\n\n## Methods\nWe drew 300 runs for every setting of the grid.\n\n## Discussion\nIt is fine.\n"
    sections = paper_scope.split(paper)
    assert paper_scope.naming(paper, sections, "The Discussion is thin") == {2}
    assert paper_scope.naming(paper, sections, "the discussion is thin") == set(), "other words name nothing"
    assert paper_scope.naming(paper, sections, 'says "We drew 300 runs for every setting of the grid." only') == {1}


# --- a flagged caption that names no figure: the sections that hold a figure ---------------------------------------

def test_a_caption_flagged_without_a_figure_asks_for_the_sections_holding_a_figure(tmp_path: Path) -> None:
    reply = ("## Results\n![**Figure 1.** Final size against R0, the mean of 300 simulations.](figures/final_size.png)\n\n"
             "Figure 1 shows the final size rising with R0.\n")
    eng, client, logged = _engine(tmp_path, [DRAFT, reply])
    out1, before = _first_draft(eng)
    after = _revise(eng, out1, ["figure_caption"], claim_grounding={})
    assert client.nodes == ["write", "write"]
    assert "1. `## Results`" in client.calls[1][1] and "2. `" not in client.calls[1][1]
    for title in ("Abstract", "Introduction", "Discussion"):
        assert _section(after, title) == _section(before, title)
    assert "the mean of 300 simulations" in after
    assert not any("cannot place" in m for m in _info(logged))


def test_a_claim_the_check_paraphrased_is_placed_in_its_closest_section() -> None:
    paper = ("# T\n\n## Introduction\nDeterministic models predict a final size for every reproduction number [1]. "
             "Early extinction is certain below the threshold value of one.\n\n"
             "## Results\nThe mean over 300 runs was 0.583.\n")
    sections = paper_scope.split(paper)
    claim = "Extinction early on is certain when the threshold value is below one"
    assert paper_scope.placing(paper, sections, claim) == {1}
    assert paper_scope.placing(paper, sections, "A wholly different statement about quantum chromodynamics lattices") == set()
