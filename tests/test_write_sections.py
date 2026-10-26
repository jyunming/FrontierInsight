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
DISCUSSION_PLACE = [{"id": "4.2", "section": "Discussion", "passage": "A larger population would shrink the gap [1]."}]
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
    after = _revise(eng, out1, [DISCUSSION_HIT], review={"hit_places": {DISCUSSION_HIT: DISCUSSION_PLACE}})

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
    after = _revise(eng, out1, [DISCUSSION_HIT], review={"hit_places": {DISCUSSION_HIT: DISCUSSION_PLACE}})

    assert client.nodes == ["write", "write"], "dropped, not asked again"
    assert _section(after, "Results") == _section(before, "Results")
    assert "nobody asked for" not in after
    assert "early extinction matters less" in after
    assert any('also gave "Results"' in m and "not asked for" in m for m in _info(logged))


def test_a_reply_without_the_asked_section_writes_the_whole_paper(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, "## Results\nOnly another section.\n", DRAFT])
    out1, _before = _first_draft(eng)
    _revise(eng, out1, [DISCUSSION_HIT], review={"hit_places": {DISCUSSION_HIT: DISCUSSION_PLACE}})
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




def test_an_unsupported_claim_is_placed_by_the_id_the_check_gave(tmp_path: Path) -> None:
    grounding = {"claims": [{"claim": "something paraphrased", "basis": "unsupported", "evidence": "",
                             "where": [{"id": "1.3", "section": "Introduction", "passage": CLAIM}]}],
                 "unsupported": ["something paraphrased"]}
    eng, client, _ = _engine(tmp_path, [DRAFT, "garbage", "## Introduction\nRewritten introduction [1].\n"])
    out1, before = _first_draft(eng)
    after = _revise(eng, out1, ["unsupported_claim"], claim_grounding=grounding)
    assert client.nodes == ["write", "write.patch", "write"]
    assert "1. `## Introduction`" in client.calls[2][1] and "2. `" not in client.calls[2][1]
    for title in ("Abstract", "Results", "Discussion"):
        assert _section(after, title) == _section(before, title)


def test_a_claim_without_a_valid_id_writes_the_whole_paper(tmp_path: Path) -> None:
    old_record = {"claims": [{"claim": "Extinction is certain below R0 = 1.2.", "basis": "unsupported", "evidence": ""}],
                  "unsupported": ["Extinction is certain below R0 = 1.2."]}
    eng, client, logged = _engine(tmp_path, [DRAFT, "garbage", DRAFT])
    out1, _before = _first_draft(eng)
    _revise(eng, out1, ["unsupported_claim"], claim_grounding=old_record)
    assert client.nodes == ["write", "write.patch", "write"]
    assert "<earlier_draft>" not in client.calls[2][1]
    assert any("writing the whole paper: FI cannot place in a section" in m and "no valid id" in m for m in _info(logged))


def test_an_id_fi_never_issued_is_dropped() -> None:
    from core.engine import _read_hit_places

    marked, table = paper_scope.mark("# T\n\n## A\nOne sentence of the draft here.\n")
    assert "<<1.1>> One sentence" in marked and "## A <<section 1>>" in marked
    assert _read_hit_places([{"hit": "h", "ids": ["<<1.1>>", "9.9"]}], table) == {
        "h": [{"id": "1.1", "section": "A", "passage": "One sentence of the draft here."}]}
    assert _read_hit_places([{"hit": "h", "ids": ["9.9", "free words"]}], table) == {}


def test_a_passage_that_the_draft_no_longer_holds_cannot_be_placed() -> None:
    paper = "# T\n\n## A\nOne sentence here today.\n\n## B\nAnother one.\n"
    sections = paper_scope.split(paper)
    place = [{"id": "1.1", "section": "A", "passage": "One sentence here today."}]
    assert paper_scope.place(paper, sections, place) == {1}
    assert paper_scope.place(paper, sections, [{**place[0], "passage": "A sentence written since."}]) is None
    assert paper_scope.place(paper, sections, []) is None


def test_the_claim_check_and_the_review_carry_fis_ids(tmp_path: Path) -> None:
    import json

    from tests.test_claim_check import _engine as claim_engine

    eng = claim_engine(tmp_path)
    seen: list[str] = []

    async def fake_chat(prompt: str, **kw: Any) -> str:
        seen.append(prompt)
        return json.dumps({"claims": [
            {"claim": "c1", "basis": "unsupported", "evidence": "x", "where": ["<<1.2>>"]},
            {"claim": "c2", "basis": "unsupported", "evidence": "x", "where": ["7.7"]},
            {"claim": "c3", "basis": "unsupported", "evidence": "x"}], "summary": "s"})

    eng._chat = fake_chat  # type: ignore[assignment,method-assign]
    text = ("# P\n\n## Intro\nA first sentence about the study here. A second sentence that is a claim.\n\n"
            "## Results\nThe result is stated in this sentence of the results section.\n")
    path = tmp_path / "paper" / "paper.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    out = asyncio.run(eng._node_claim_check({"topic": "t", "paper_md": str(path)}))  # type: ignore[arg-type]
    assert "<<1.1>> A first sentence" in seen[0] and "<<2.1>> The result" in seen[0] and "## Intro <<section 1>>" in seen[0]
    c1, c2, c3 = out["claim_grounding"]["claims"][:3]
    assert c1["where"] == [{"id": "1.2", "section": "Intro", "passage": "A second sentence that is a claim."}]
    assert c2["where"] == [] and c3["where"] == []
    agents = Path(__file__).resolve().parent.parent / "agents"
    review_prompt = (agents / "review.md").read_text(encoding="utf-8")
    assert "must_flag_places" in review_prompt and "<<3.7>>" in review_prompt
    assert '"where"' in (agents / "claim_check.md").read_text(encoding="utf-8")


# --- a cited section id, a cited figure caption, a declared new section -------------------------------------------------

def test_a_cited_section_id_places_that_section() -> None:
    paper = "# T\n\n## A\nOne sentence here today.\n\n## B\nAnother one.\n"
    marked, table = paper_scope.mark(paper)
    assert "## B <<section 2>>" in marked and "section 2" in table
    for cited in ("<<section 2>>", "section 2", ["Section 2"]):
        places = paper_scope.read_places(cited, table)
        assert [p["id"] for p in places] == ["section 2"], cited
        assert paper_scope.place(paper, paper_scope.split(paper), places) == {2}
    assert paper_scope.read_places("<<section 9>>", table) == []


def test_a_figure_caption_hit_is_placed_by_the_cited_passage_not_by_every_figure_section(tmp_path: Path) -> None:
    hit = "figure_caption: the caption claims more than the figure shows"
    place = [{"id": "2.3", "section": "Introduction", "passage": CLAIM}]
    reply = "## Introduction\nRewritten introduction [1].\n"
    eng, client, logged = _engine(tmp_path, [DRAFT, reply])
    out1, before = _first_draft(eng)
    after = _revise(eng, out1, [hit], claim_grounding={}, review={"hit_places": {hit: place}})
    assert client.nodes == ["write", "write"]
    assert "1. `## Introduction`" in client.calls[1][1] and "2. `" not in client.calls[1][1]
    assert _section(after, "Results") == _section(before, "Results")
    assert not any("asking for every section that holds a figure" in m for m in _info(logged))


def test_a_declared_new_section_is_kept_right_after_the_section_it_came_with() -> None:
    paper = "# T\n\n## A\nalpha\n\n## B\nbeta\n\n## C\ngamma\n"
    sections = paper_scope.split(paper)
    reply = ("## A\nnew alpha\n\nNEW SECTION: Limitations -- the fix needs a place to state them\n"
             "## Limitations\nWhat limits it.\n\n## C\nsneaky\n")
    spliced = paper_scope.splice(paper, sections, {1}, reply)
    assert spliced.text == "# T\n\n## A\nnew alpha\n\n## Limitations\nWhat limits it.\n\n## B\nbeta\n\n## C\ngamma\n"
    assert spliced.added == ["Limitations"] and spliced.ignored == ["C"]
    undeclared = paper_scope.splice(paper, sections, {1}, "## A\nnew alpha\n\n## Limitations\nWhat limits it.\n")
    assert undeclared.text == "# T\n\n## A\nnew alpha\n\n## B\nbeta\n\n## C\ngamma\n"
    assert undeclared.ignored == ["Limitations"] and undeclared.added == []


# --- two figure_caption hits: each is placed on its own ------------------------------------------------------------------

FIGURE_DRAFT = (
    "# T\n\n## Abstract\nWe measure it.\n\n"
    "## Methods\nThe set-up is described in this sentence of the methods.\n\n"
    "![Methods figure caption.](figures/m.png)\n\n"
    "## Results\nThe results are described in this sentence.\n\n"
    "![Results figure caption.](figures/r.png)\n\n"
    "## Discussion\nThe discussion is described in this sentence.\n\n"
    "![Discussion figure caption.](figures/d.png)\n"
)
HIT_A = "figure_caption: the first caption claims more than the figure shows"
HIT_B = "figure_caption: the second caption claims more than the figure shows"
PLACE_A = [{"id": "3.1", "section": "Results", "passage": "The results are described in this sentence."}]
PLACE_B = [{"id": "4.1", "section": "Discussion", "passage": "The discussion is described in this sentence."}]


def _scope_for(tmp_path: Path, places: dict[str, Any]) -> tuple[dict[int, list[str]], str, list[str]]:
    eng, _client, logged = _engine(tmp_path, [FIGURE_DRAFT])
    path = tmp_path / "paper" / "paper.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FIGURE_DRAFT, encoding="utf-8")
    sections = paper_scope.split(FIGURE_DRAFT)
    state = _revise_state({"literature": [], "paper_md": str(path), "paper_basis": None}, [HIT_A, HIT_B],
                          review={"hit_places": places}, claim_grounding={})
    from core.engine import _paper_basis

    state["paper_basis"] = _paper_basis(state)
    _s, _end, asked, _targets, why = eng._scope_of_round(state, FIGURE_DRAFT, FIGURE_DRAFT, refine_round=False)
    return asked, why, _info(logged)


def test_a_cited_figure_hit_stays_cited_when_another_figure_hit_is_unplaced(tmp_path: Path) -> None:
    asked, why, logged = _scope_for(tmp_path, {HIT_A: PLACE_A})
    titles = {i: paper_scope.split(FIGURE_DRAFT)[i].title for i in asked}
    assert "Results" in titles.values(), "hit A's cited section is asked for"
    assert any("second caption" in m and "no cited passage" in m for m in logged), "the log names hit B"
    assert not any("first caption" in m and "no cited passage" in m for m in logged), "hit A is not the unplaced one"


def test_two_cited_figure_hits_touch_no_other_figure_section(tmp_path: Path) -> None:
    asked, why, logged = _scope_for(tmp_path, {HIT_A: PLACE_A, HIT_B: PLACE_B})
    titles = sorted(paper_scope.split(FIGURE_DRAFT)[i].title for i in asked)
    assert titles == ["Discussion", "Results"] and not why
    assert not any("no cited passage" in m for m in logged)
