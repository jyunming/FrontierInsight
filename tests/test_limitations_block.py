"""FI writes the paper's whole Limitations section from its records (core/paper_facts.py ``{{limitations}}``), and the points
of the last review go into it when no round is left to answer them (core/disclosure.py ``last_review_points``). No model
is called."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core import disclosure, paper_facts
from tests.test_plan_optimisation import _engine

ITEM = "The topic asks to vary temperature; FI's check did not find a setting of the study's plan that varies it."
NOTHING = paper_facts.NO_LIMITS_SENTENCE


def _quest(tmp_path: Path, *, item: bool = True) -> Path:
    (tmp_path / ".fi").mkdir(parents=True, exist_ok=True)
    if item:
        (tmp_path / ".fi" / "topic_covered.json").write_text(
            json.dumps({"left": [{"kind": "factor", "name": "temperature"}]}), encoding="utf-8")
    return tmp_path


def _filled(draft: str, root: Path) -> str:
    return paper_facts.render(paper_facts.place_blocks(draft, root), root).text


def _sections(text: str) -> list[str]:
    return re.findall(r"(?m)^#{1,6} .*limitations.*$", text, re.I)


def test_the_block_is_one_bullet_for_each_item_or_one_sentence_when_there_is_none(tmp_path: Path) -> None:
    empty = _quest(tmp_path / "a", item=False)
    assert paper_facts.limitations_text(empty) == NOTHING
    assert paper_facts.block_text(empty, "limitations") == NOTHING  # never empty: a paper always has the section
    root = _quest(tmp_path / "b")
    assert paper_facts.limitations_text(root) == f"- {ITEM}"


def test_the_writers_own_limitations_text_is_replaced_whatever_the_heading_looks_like(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    for heading in ("## Limitations", "### 5.2 Limitations", "## LIMITATIONS AND FUTURE WORK", "## 6. Study limitations"):
        sub = heading.split()[0] + "#"
        draft = (f"# T\n\n## Methods\nWe ran it.\n\n{heading}\nThe grid was coarse.\n\n{sub} Sub-point\nMore words.\n\n"
                 "## References\n1. A.\n")
        out = _filled(draft, root)
        assert "The grid was coarse" not in out and "More words" not in out, heading  # the writer's text is gone
        assert f"- {ITEM}" in out and heading in out and len(_sections(out)) == 1, heading
        assert out.index(heading) < out.index(ITEM) < out.index("## References")


def test_the_body_stops_at_the_next_heading_of_the_same_or_a_higher_level(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    draft = ("# T\n\n## Discussion\nD.\n\n### Limitations\nOld.\n\n#### Detail\nOld detail.\n\n### Source status\nS.\n\n"
             "## References\n1. A.\n")
    out = _filled(draft, root)
    assert "Old." not in out and "Old detail." not in out
    assert out.index(ITEM) < out.index("### Source status") and "S." in out


def test_a_paper_with_no_such_heading_gets_a_section_before_the_conclusion_else_the_references_else_the_end(
        tmp_path: Path) -> None:
    root = _quest(tmp_path)
    both = "# T\n\n## Methods\nM.\n\n## Conclusions\nC.\n\n## References\n1. A.\n"
    out = _filled(both, root)
    assert out.index("## Methods") < out.index("## Limitations") < out.index("## Conclusions") < out.index("## References")
    refs = "# T\n\n## Methods\nM.\n\n---\n\n## References\n1. A.\n"
    out = _filled(refs, root)
    assert out.index("## Limitations") < out.index("---") < out.index("## References")  # the rule stays before the sources
    plain = "# T\n\nJust text.\n"
    out = _filled(plain, root)
    assert out.endswith("<!-- /fi:fact-limitations -->\n") and out.index("Just text.") < out.index("## Limitations")


def test_a_second_pass_adds_nothing_and_a_copy_that_lost_its_markers_is_replaced(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    draft = "# T\n\n## Methods\nM.\n\n## Limitations\nOld.\n\n## References\n1. A.\n"
    once = _filled(draft, root)
    assert _filled(once, root) == once and once.count("<!-- fi:fact-limitations -->") == 1
    # the markers are gone (a copy through a model, or text a reader pasted back): FI's current content takes its place
    bare = paper_facts.without_marks(once).replace(ITEM, "An older wording of the same.")
    again = _filled(bare, root)
    assert again.count("<!-- fi:fact-limitations -->") == 1 and "An older wording" not in again and ITEM in again
    # a placeholder the writer typed is not a second block
    typed = "# T\n\n## Methods\nM.\n\n## Limitations\n{{limitations}}\n\n## References\n1. A.\n"
    out = _filled(typed, root)
    assert out.count("<!-- fi:fact-limitations -->") == 1 and out.count(ITEM) == 1
    # the passage editor gets the placeholder, and FI fills it again from the records as they are now
    assert "{{limitations}}" in paper_facts.to_placeholders(once) and ITEM not in paper_facts.to_placeholders(once)
    (root / ".fi" / "topic_covered.json").unlink()
    assert NOTHING in _filled(once, root) and ITEM not in _filled(once, root)


def test_the_writer_is_told_that_fi_writes_the_section_and_what_it_says(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    section = paper_facts.writer_section(root, topic="t")
    assert f"- {ITEM}" in section and "see Limitations" in section and "{{not_done:" not in section
    assert "Do not write a Limitations section" in section


# ---- the last review ----------------------------------------------------------------------------------------------------


def _review(root: Path, *, verdict: str = "revise", weaknesses: list[str] | None = None, blocking: str = "") -> None:
    disclosure.write_last_review(root, {"verdict": verdict, "weaknesses": weaknesses or [], "blocking": blocking})


def test_the_last_reviews_points_are_stated_only_when_the_paper_goes_out_as_that_review_saw_it(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    _review(root, weaknesses=["The abstract calls a curve roughly flat; the paper's own numbers fall.", "A second point."],
            blocking="A reason.")
    assert disclosure.last_review_points(root) == []  # not final: a rewrite may still answer them
    assert not [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LAST_REVIEW]
    assert disclosure.mark_last_review_final(root) is True
    assert disclosure.last_review_points(root) == [
        "A reason.", "The abstract calls a curve roughly flat; the paper's own numbers fall.", "A second point."]
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LAST_REVIEW]
    assert item.sentence.startswith("The final review raised these points, which no later round addressed: \"A reason.\"")
    assert "roughly flat" in _filled("# T\n\n## Limitations\nOld.\n", root)
    # the next round says its own: the record is not final any more
    _review(root, weaknesses=["Another."])
    assert disclosure.last_review_points(root) == []


def test_an_accepted_paper_states_nothing_and_a_long_list_is_cut_to_eight(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    _review(root, verdict="accept", weaknesses=["Minor."])
    assert disclosure.mark_last_review_final(root) is False and disclosure.last_review_points(root) == []
    _review(root, weaknesses=[f"Point number {n}." for n in range(11)])
    assert disclosure.mark_last_review_final(root)
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LAST_REVIEW]
    assert "Point number 7." in item.sentence and "Point number 8." not in item.sentence
    assert item.sentence.endswith("and 3 more, listed in QUALITY.md.")
    quality = disclosure.run_record(root)  # QUALITY.md holds every point
    assert all(f"Point number {n}." in quality for n in range(11))


def test_a_point_the_record_of_objections_holds_is_not_stated_a_second_time(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    held = "Unsupported claims remain in the discussion."
    disclosure.add_objections(root, [], held)
    disclosure.mark_unresolved(root, held)
    _review(root, weaknesses=[held, "A new point."], blocking=held)
    assert disclosure.mark_last_review_final(root)
    assert disclosure.last_review_points(root) == ["A new point."]
    kinds = [i.kind for i in paper_facts.required_items(root)]
    assert kinds.count(paper_facts.LAST_REVIEW) == 1 and kinds.count(paper_facts.UNRESOLVED_REVIEW) == 1
    # a different wording is a different point: nothing is matched by meaning
    _review(root, weaknesses=["unsupported claims remain in the discussion"])
    assert disclosure.mark_last_review_final(root)
    assert disclosure.last_review_points(root) == ["unsupported claims remain in the discussion"]


def test_the_review_keeps_what_it_said_each_round(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    eng._audit_claims("review", {"review": {"verdict": "revise", "blocking": "no oracle", "weaknesses": ["thin"],
                                            "why": "short"}})
    record = json.loads((eng.quest_root / ".fi" / "last_review.json").read_text(encoding="utf-8"))
    assert record == {"verdict": "revise", "weaknesses": ["thin"], "blocking": "no oracle", "final": False}


@pytest.mark.parametrize("citation_next, verdict, stated", [("done", "revise", True), ("done", "accept", False),
                                                            ("human_feedback", "revise", False)])
async def test_the_last_step_states_the_last_reviews_points_only_for_a_paper_that_leaves_with_them_open(
        tmp_path: Path, citation_next: str, verdict: str, stated: bool) -> None:
    eng = _engine(tmp_path, [])
    root = eng.quest_root
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    paper = root / "paper.md"
    paper.write_text("# T\n\n## Methods\nM.\n\n## References\n1. A.\n", encoding="utf-8")
    _review(root, verdict=verdict, weaknesses=["The abstract calls a curve roughly flat."])

    async def citations(state: dict) -> dict:
        return {"citation_next": citation_next}

    eng._check_citations = citations  # type: ignore[method-assign]
    await eng._node_citation_check({"paper_md": str(paper)})  # type: ignore[arg-type]
    text = paper.read_text(encoding="utf-8")
    assert ("roughly flat" in text) is stated
    assert ("roughly flat" in disclosure.run_record(root)) is stated
    assert text.count("<!-- fi:fact-limitations -->") <= 1
