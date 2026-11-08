"""FI adds its own list to the paper's Limitations from its records (core/paper_facts.py ``{{limitations}}``), keeping every
word the writer wrote, and the points of the last review go into it when no round is left to answer them
(core/disclosure.py ``last_review_points``). No model is called."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import disclosure, paper_facts, write_personas
from tests.test_plan_optimisation import _engine

ITEM = "The topic asks to vary temperature; FI's check did not find a setting of the study's plan that varies it."


def _quest(tmp_path: Path, *, item: bool = True) -> Path:
    (tmp_path / ".fi").mkdir(parents=True, exist_ok=True)
    if item:
        (tmp_path / ".fi" / "topic_covered.json").write_text(
            json.dumps({"left": [{"kind": "factor", "name": "temperature"}]}), encoding="utf-8")
    return tmp_path


def _filled(draft: str, root: Path, **kw: object) -> str:
    return paper_facts.render(paper_facts.place_blocks(draft, root, **kw), root).text  # type: ignore[arg-type]


def _kept(draft: str, out: str) -> bool:
    """Every non-blank line of ``draft`` is in ``out``, in the same order: nothing the writer wrote is lost."""
    lines = paper_facts.without_marks(out).splitlines()
    at = 0
    for want in (ln for ln in draft.splitlines() if ln.strip()):
        try:
            at = lines.index(want, at) + 1
        except ValueError:
            return False
    return True


def _one_block(out: str) -> bool:
    return out.count("<!-- fi:fact-limitations -->") == 1 and out.count(ITEM) == 1


def test_the_block_is_one_bullet_for_each_item_and_nothing_when_there_is_none(tmp_path: Path) -> None:
    assert paper_facts.limitations_text(_quest(tmp_path / "a", item=False)) == ""
    assert paper_facts.limitations_text(_quest(tmp_path / "b")) == f"- {ITEM}"


def test_the_block_goes_at_the_end_of_the_writers_own_limitations_and_its_words_are_kept(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    for heading in ("## Limitations", "### 5.2 Limitations", "## LIMITATIONS AND FUTURE WORK",
                    "## 6. Limitations and caveats"):
        draft = f"# T\n\n## Methods\nWe ran it.\n\n{heading}\nThe grid was coarse.\n\n## References\n1. A.\n"
        out = _filled(draft, root)
        assert _kept(draft, out) and _one_block(out), heading
        assert out.index("The grid was coarse.") < out.index(ITEM) < out.index("## References"), heading
        assert "## Limitations\n" not in out.replace(heading + "\n", ""), heading  # no second section of FI's own


def test_the_block_goes_before_the_first_sub_heading_of_the_section(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    draft = "# T\n\n## Limitations and future work\nOwn words.\n\n### Future work\nMore.\n\n## References\n1. A.\n"
    out = _filled(draft, root)
    assert _kept(draft, out) and _one_block(out)
    assert out.index("Own words.") < out.index(ITEM) < out.index("### Future work")


def test_a_results_sub_section_about_limitations_is_not_the_limitations_section(tmp_path: Path) -> None:
    """The review's case: a Results sub-section about the approximation's limitations holding a table, a figure and FI's own
    block: nothing in it is lost, and FI adds its own section."""
    root = _quest(tmp_path)
    keep = ("# T\n\n## Discussion and limitations\nWhat it means.\n\n### Future work\nMore studies.\n\n"
            "## Results\n\n### 2.1 Limitations of the approximation\nIt breaks down at large angles.\n\n"
            "| a | b |\n|---|---|\n| 1 | 2 |\n\n![**Figure 1.** A curve.](figures/a.png)\n\n")
    tail = "## Conclusion\nDone.\n\n## References\n1. A.\n"
    block = "<!-- fi:fact-results_table -->\nFI's table.\n<!-- /fi:fact-results_table -->\n\n"
    out = _filled(keep + block + tail, root)
    assert _kept(keep + tail, out) and _one_block(out)
    assert out.count("<!-- fi:fact-results_table -->") == 1  # FI's other block is still there, once
    # the Discussion and limitations section is the writer's Limitations: FI's list is in it, before its Future work
    assert out.index("What it means.") < out.index(ITEM) < out.index("### Future work")
    only = keep.replace("## Discussion and limitations", "## Discussion").replace("### Future work", "### Next")
    out = _filled(only + block + tail, root)
    assert _kept(only + tail, out) and _one_block(out)
    assert out.count("## Limitations") == 1 and out.index("## Results") < out.index("## Limitations") < out.index("## Conclusion")


def test_a_run_in_paragraph_is_the_writers_limitations_and_the_block_follows_it(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    draft = ("# T\n\n## Discussion\nIt agrees.\n\n**Limitations.** Our own limits here.\nMore of them.\n\n"
             "Next paragraph.\n\n## References\n1. A.\n")
    out = _filled(draft, root)
    assert _kept(draft, out) and _one_block(out) and "## Limitations" not in out
    assert out.index("More of them.") < out.index(ITEM) < out.index("Next paragraph.")


def test_a_placeholder_the_writer_put_stays_where_it_is(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    draft = "# T\n\n## Limitations\nFirst.\n\n{{limitations}}\n\nLast words.\n\n## References\n1. A.\n"
    out = _filled(draft, root)
    assert _kept(draft.replace("{{limitations}}\n\n", ""), out) and _one_block(out)
    assert out.index("First.") < out.index(ITEM) < out.index("Last words.")


def test_a_paper_with_no_limitations_gets_a_section_before_the_conclusion_else_the_references_else_the_end(
        tmp_path: Path) -> None:
    root = _quest(tmp_path)
    both = "# T\n\n## Methods\nM.\n\n## Conclusions\nC.\n\n## References\n1. A.\n"
    out = _filled(both, root)
    assert out.index("## Methods") < out.index("## Limitations") < out.index("## Conclusions") < out.index("## References")
    refs = "# T\n\n## Methods\nM.\n\n---\n\n## References\n1. A.\n"
    out = _filled(refs, root)
    assert out.index("## Limitations") < out.index("---") < out.index("## References")
    out = _filled("# T\n\nJust text.\n", root)
    assert out.endswith("<!-- /fi:fact-limitations -->\n") and out.index("Just text.") < out.index("## Limitations")


def test_a_format_that_forbids_the_heading_gets_a_run_in_paragraph_from_the_same_list_its_prompt_shows(
        tmp_path: Path) -> None:
    root = _quest(tmp_path)
    draft = "# A thought\n\nFirst argument.\n\n## What the noise tells us\nIt tells much.\n\n## Implications\nSo.\n"
    out = _filled(draft, root, limitations_heading=False)
    assert "## Limitations" not in out and _kept(draft, out) and _one_block(out)
    assert out.index("**Limitations.**") < out.index(ITEM)
    assert _filled(out, root, limitations_heading=False) == out  # a second pass adds nothing
    # one list, read by FI's placing and by the persona's prompt
    from core.engine import _load_persona_prefix

    for name in ("essay", "policy_brief"):
        assert write_personas.forbids_limitations(name)
        prompt = _load_persona_prefix(name, category="write")
        assert write_personas.TOKEN not in prompt and "`## Limitations`" in prompt
        assert all(f"`## {h}`" in prompt for h in write_personas.FORBIDDEN_HEADINGS[name])
    eng = _engine(tmp_path / "e", [])
    eng.config.output.paper_format = "essay"
    assert eng._limitations_heading({}) is False  # type: ignore[arg-type]
    eng.config.output.paper_format = "generic"
    assert eng._limitations_heading({}) is True  # type: ignore[arg-type]


def test_with_nothing_to_state_there_is_no_block_and_no_heading_and_an_old_one_goes(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    draft = "# T\n\n## Methods\nM.\n\n## References\n1. A.\n"
    assert paper_facts.place_blocks(draft, root) == draft and "ot recorded" not in _filled(draft, root)
    topic_covered = root / ".fi" / "topic_covered.json"
    # an item went away since the block was written: the block and the heading FI made for it go too
    topic_covered.write_text(json.dumps({"left": [{"kind": "factor", "name": "temperature"}]}))
    once = _filled(draft, root)
    assert ITEM in once and "## Limitations" in once
    topic_covered.unlink()
    assert _filled(once, root) == draft
    # the writer's own section stays when the block goes
    own = "# T\n\n## Limitations\nOwn words.\n\n## References\n1. A.\n"
    topic_covered.write_text(json.dumps({"left": [{"kind": "factor", "name": "temperature"}]}))
    with_block = _filled(own, root)
    topic_covered.unlink()
    assert _filled(with_block, root) == own
    assert "{{" not in _filled("# T\n\n## Limitations\n{{limitations}}\n\n## References\n1. A.\n", root)


def test_with_nothing_to_state_only_the_heading_fi_made_goes_never_a_writers_own(tmp_path: Path) -> None:
    """A writer's ``## Limitations`` holding only sub-sections (or a lead of its own) survives the no-items path; the heading
    FI added itself, marked as FI's, goes with its block."""
    root = _quest(tmp_path)
    topic_covered = root / ".fi" / "topic_covered.json"
    items = topic_covered.read_text(encoding="utf-8")
    only_subs = ("# T\n\n## Methods\nM.\n\n## Limitations\n\n### Scope\nOnly small angles.\n\n"
                 "### Assumptions\nNo drag.\n\n## References\n1. A.\n")
    with_block = _filled(only_subs, root)
    assert _one_block(with_block) and with_block.index("## Limitations") < with_block.index(ITEM) < with_block.index("### Scope")
    topic_covered.unlink()
    assert _filled(with_block, root) == only_subs  # the writer's heading and both sub-sections are all still there
    # the same for a run-in line the writer wrote, and for a writer's heading with a lead
    run_in = "# T\n\n## Methods\nM.\n\n**Limitations.** Own words.\n\n## References\n1. A.\n"
    topic_covered.write_text(items, encoding="utf-8")
    block = _filled(run_in, root)
    topic_covered.unlink()
    assert _filled(block, root) == run_in
    # FI's own heading is marked and goes with its block; the mark is not shown to a reader
    plain = "# T\n\n## Methods\nM.\n\n## References\n1. A.\n"
    topic_covered.write_text(items, encoding="utf-8")
    added = _filled(plain, root)
    assert paper_facts.HEADING_MARK in added and paper_facts.HEADING_MARK not in paper_facts.without_marks(added)
    topic_covered.unlink()
    assert _filled(added, root) == plain
    # a writer who typed under FI's heading while its block was there keeps the words and the heading
    topic_covered.write_text(items, encoding="utf-8")
    added = _filled(plain, root)
    typed = added.replace("<!-- /fi:fact-limitations -->", "<!-- /fi:fact-limitations -->\n\nThe writer's own sentence.")
    topic_covered.unlink()
    out = _filled(typed, root)
    assert "The writer's own sentence." in out and "## Limitations" in out and ITEM not in out
    assert paper_facts.HEADING_MARK not in out


def test_only_the_place_a_block_was_cut_is_tidied_not_the_rest_of_the_paper(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    topic_covered = root / ".fi" / "topic_covered.json"
    items = topic_covered.read_text(encoding="utf-8")
    draft = "# T\n\n## Methods\nM.\n\n```\na\n\n\n\nb\n```\n\n## Limitations\nOwn.\n\n## References\n1. A.\n"
    with_block = _filled(draft, root)
    topic_covered.unlink()
    assert _filled(with_block, root) == draft  # the three blank lines inside the code block stay
    topic_covered.write_text(items, encoding="utf-8")


def test_a_main_level_limitations_wins_over_a_deeper_one_that_comes_later(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    draft = ("# T\n\n## Limitations\nMain words.\n\n## Appendix\n\n### Limitations\nA note in the appendix.\n\n"
             "## References\n1. A.\n")
    out = _filled(draft, root)
    assert _kept(draft, out) and _one_block(out)
    assert out.index("Main words.") < out.index(ITEM) < out.index("## Appendix")
    # with no main-level one, a deeper heading that is just "Limitations" is used
    deeper = "# T\n\n## Discussion\nIt agrees.\n\n### Limitations\nOwn.\n\n## References\n1. A.\n"
    out = _filled(deeper, root)
    assert _kept(deeper, out) and _one_block(out) and out.count("imitations" + chr(10)) == 1
    assert out.index("Own.") < out.index(ITEM) < out.index("## References")


def test_a_final_record_of_another_round_is_cleared_and_cannot_come_back(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    old = _review(root, weaknesses=["Old point."])
    assert disclosure.mark_last_review_final(root, old) is True
    now = {"verdict": "revise", "weaknesses": ["New point."], "blocking": ""}
    assert disclosure.mark_last_review_final(root, now) is False  # the quest's review now is not the one on record
    assert json.loads((root / ".fi" / "last_review.json").read_text(encoding="utf-8"))["final"] is False
    assert disclosure.last_review_points(root) == []
    # and when the record is later brought back to the old review, it is not final any more
    assert disclosure.mark_last_review_final(root, old) is True  # a fresh mark of a matching record is what makes it final


def test_a_second_pass_adds_nothing_and_the_passage_editor_gets_the_placeholder(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    draft = "# T\n\n## Methods\nM.\n\n## Limitations\nOld.\n\n## References\n1. A.\n"
    once = _filled(draft, root)
    assert _filled(once, root) == once and _one_block(once)
    assert "{{limitations}}" in paper_facts.to_placeholders(once) and ITEM not in paper_facts.to_placeholders(once)
    (root / ".fi" / "topic_covered.json").write_text(json.dumps({"left": [{"kind": "factor", "name": "pressure"}]}))
    again = _filled(once, root)  # the records changed: the same block, its content from the records as they are now
    assert "pressure" in again and ITEM not in again and again.count("<!-- fi:fact-limitations -->") == 1


def test_the_writer_is_shown_the_list_and_may_write_a_section_about_the_methods_scope(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    section = paper_facts.writer_section(root, topic="t")
    assert f"- {ITEM}" in section and "see Limitations" in section and "{{not_done:" not in section
    assert "scope and assumptions" in section and "Do not restate" in section
    assert paper_facts.writer_section(_quest(tmp_path / "n", item=False), topic="t") == ""


# ---- the last review ----------------------------------------------------------------------------------------------------


def _review(root: Path, *, verdict: str = "revise", weaknesses: list[str] | None = None, blocking: str = "") -> dict:
    review = {"verdict": verdict, "weaknesses": weaknesses or [], "blocking": blocking}
    disclosure.write_last_review(root, review)
    return review


def test_the_last_reviews_points_are_stated_only_when_the_paper_goes_out_as_that_review_saw_it(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    review = _review(root, weaknesses=["The abstract calls a curve roughly flat; the paper's own numbers fall.",
                                       "A second point."], blocking="A reason.")
    assert disclosure.last_review_points(root) == []  # not final: a rewrite may still answer them
    assert not [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LAST_REVIEW]
    assert disclosure.mark_last_review_final(root, review) is True
    assert disclosure.last_review_points(root) == [
        "A reason.", "The abstract calls a curve roughly flat; the paper's own numbers fall.", "A second point."]
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LAST_REVIEW]
    assert item.sentence.startswith("The final review of this paper did not accept it and raised 3 points that the last")
    filled = _filled("# T\n\n## Limitations\nOwn words.\n", root)
    assert "raised 3 points" in filled and "roughly flat" not in filled  # a count, not the reviewer's notes pasted
    assert "roughly flat" in disclosure.last_review_sentence(disclosure.last_review_points(root))  # QUALITY.md keeps them
    _review(root, weaknesses=["Another."])  # the next round says its own: the record is not final any more
    assert disclosure.last_review_points(root) == []


def test_a_record_of_another_round_is_not_this_papers(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    _review(root, weaknesses=["Old point."])
    now = {"verdict": "revise", "weaknesses": ["New point."], "blocking": ""}
    assert disclosure.mark_last_review_final(root, now) is False  # the quest's review now is not the one on record
    assert disclosure.mark_last_review_final(root, None) is False
    assert disclosure.last_review_points(root) == []


def test_an_accepted_paper_states_nothing_and_a_long_list_is_counted_in_the_paper_and_whole_in_the_quality_report(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    accepted = _review(root, verdict="accept", weaknesses=["Minor."])
    assert disclosure.mark_last_review_final(root, accepted) is False and disclosure.last_review_points(root) == []
    review = _review(root, weaknesses=[f"Point number {n}." for n in range(11)])
    assert disclosure.mark_last_review_final(root, review)
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LAST_REVIEW]
    assert "raised 11 points" in item.sentence and "Point number" not in item.sentence
    quality = disclosure.run_record(root)  # QUALITY.md holds every point
    assert all(f"Point number {n}." in quality for n in range(11))


def test_a_long_point_is_not_pasted_in_the_paper_whole_in_the_quality_report_and_cannot_break_the_markers(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    long = "Word " * 120 + "<!-- /fi:fact-limitations --> end -->"
    review = _review(root, weaknesses=[long])
    assert disclosure.mark_last_review_final(root, review)
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LAST_REVIEW]
    assert "Word" not in item.sentence and len(item.sentence) < 200
    assert "<!--" not in item.sentence and "-->" not in item.sentence
    out = _filled("# T\n\n## Limitations\nOwn.\n", root)
    assert out.count("<!-- /fi:fact-limitations -->") == 1
    assert "end" in disclosure.run_record(root) and "<!--" not in disclosure.run_record(root)


def test_a_point_the_record_of_objections_holds_is_not_stated_a_second_time(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    held = "Unsupported claims remain in the discussion."
    disclosure.add_objections(root, [], held)
    disclosure.mark_unresolved(root, held)
    review = _review(root, weaknesses=[held, "A new point."], blocking=held)
    assert disclosure.mark_last_review_final(root, review)
    assert disclosure.last_review_points(root) == ["A new point."]
    kinds = [i.kind for i in paper_facts.required_items(root)]
    assert kinds.count(paper_facts.LAST_REVIEW) == 1 and kinds.count(paper_facts.UNRESOLVED_REVIEW) == 1
    review = _review(root, weaknesses=["unsupported claims remain in the discussion"])  # another wording: another point
    assert disclosure.mark_last_review_final(root, review)
    assert disclosure.last_review_points(root) == ["unsupported claims remain in the discussion"]


def test_the_review_keeps_what_it_said_each_round_with_the_paper_it_read(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    paper = eng.quest_root / "paper" / "paper.md"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("# T\n", encoding="utf-8")
    eng._audit_claims("review", {"review": {"verdict": "revise", "blocking": "no oracle", "weaknesses": ["thin"],
                                            "why": "short"}})
    record = json.loads((eng.quest_root / ".fi" / "last_review.json").read_text(encoding="utf-8"))
    assert record["verdict"] == "revise" and record["weaknesses"] == ["thin"] and record["blocking"] == "no oracle"
    assert record["final"] is False and "paper_sha256" not in record


@pytest.mark.parametrize("citation_next, verdict, same_round, stated", [
    ("done", "revise", True, True), ("done", "accept", True, False), ("human_feedback", "revise", True, False),
    ("done", "revise", False, False)])
async def test_the_last_step_states_the_last_reviews_points_only_for_a_paper_that_leaves_with_them_open(
        tmp_path: Path, citation_next: str, verdict: str, same_round: bool, stated: bool) -> None:
    eng = _engine(tmp_path, [])
    root = eng.quest_root
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    paper = root / "paper.md"
    paper.write_text("# T\n\n## Methods\nM.\n\n## Limitations\nOwn words.\n\n## References\n1. A.\n", encoding="utf-8")
    review = _review(root, verdict=verdict, weaknesses=["The abstract calls a curve roughly flat."])
    if not same_round:
        review = {**review, "weaknesses": ["Another point."]}  # the record is of a round that is not this paper's

    async def citations(state: dict) -> dict:
        return {"citation_next": citation_next}

    eng._check_citations = citations  # type: ignore[method-assign]
    await eng._node_citation_check({"paper_md": str(paper), "review": review})  # type: ignore[arg-type]
    text = paper.read_text(encoding="utf-8")
    assert ("did not accept it" in text) is stated and "roughly flat" not in text
    assert ("roughly flat" in disclosure.run_record(root)) is stated
    assert "Own words." in text
