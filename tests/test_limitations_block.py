"""FI writes no Limitations block: an older FI's block is taken out of a resumed paper with the heading FI put above it,
and every word the writer wrote stays. The points of the last review are kept in FI's record when no round is left to
answer them, whole, for the quality report (core/disclosure.py ``last_review_points``). No model is called."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import disclosure, paper_facts
from tests.test_plan_optimisation import _engine

ITEM = "The topic asks to vary temperature; FI's check did not find a setting of the study's plan that varies it."
OLD_BLOCK = f"<!-- fi:fact-limitations -->\n- {ITEM}\n<!-- /fi:fact-limitations -->"


def _quest(tmp_path: Path, *, item: bool = True) -> Path:
    (tmp_path / ".fi").mkdir(parents=True, exist_ok=True)
    if item:
        (tmp_path / ".fi" / "topic_covered.json").write_text(
            json.dumps({"left": [{"kind": "factor", "name": "temperature"}]}), encoding="utf-8")
    return tmp_path


def _filled(draft: str, root: Path) -> str:
    return paper_facts.render(paper_facts.place_blocks(draft, root), root).text


def test_fi_puts_no_limitations_block_in_a_paper_whatever_its_records_hold(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    assert any(i.sentence == ITEM for i in paper_facts.required_items(root)), "the record holds the item"
    for draft in ("# T\n\n## Methods\nM.\n\n## Limitations\nOwn words.\n\n## References\n1. A.\n",
                  "# T\n\n## Methods\nM.\n\n## References\n1. A.\n"):
        out = _filled(draft, root)
        assert ITEM not in out and "fi:fact-limitations" not in out
        assert out.count("## Limitations") == draft.count("## Limitations"), "no heading added"


def test_an_older_fis_block_goes_and_the_writers_own_section_stays(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    own = "# T\n\n## Limitations\nOwn words.\n\n## References\n1. A.\n"
    old = own.replace("Own words.\n", f"Own words.\n\n{OLD_BLOCK}\n")
    assert _filled(old, root) == own
    placeholder = own.replace("Own words.\n", "Own words.\n\n{{limitations}}\n")
    assert _filled(placeholder, root) == own, "the old placeholder on its own goes too"


def test_an_older_fis_heading_goes_with_its_block_unless_the_writer_typed_under_it(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    plain = "# T\n\n## Methods\nM.\n\n## References\n1. A.\n"
    added = plain.replace("## References", f"{paper_facts.HEADING_MARK}\n## Limitations\n\n{OLD_BLOCK}\n\n## References")
    assert _filled(added, root) == plain
    typed = added.replace("<!-- /fi:fact-limitations -->", "<!-- /fi:fact-limitations -->\n\nThe writer's own sentence.")
    out = _filled(typed, root)
    assert "The writer's own sentence." in out and "## Limitations" in out and ITEM not in out
    assert paper_facts.HEADING_MARK not in out


def test_a_limitations_section_that_held_only_the_old_block_loses_its_heading(tmp_path: Path) -> None:
    """An old block alone under the writer's Limitations heading, and no record of paragraphs that moved from there: the
    paper keeps no empty section heading. Known positive: a section with the writer's words keeps its heading."""
    root = _quest(tmp_path)
    plain = "# T\n\n## Methods\nM.\n\n## References\n1. A.\n"
    alone = plain.replace("## References", f"## Limitations\n\n{OLD_BLOCK}\n\n## References")
    assert _filled(alone, root) == plain
    own = plain.replace("## References", f"## Limitations\nOwn words.\n\n{OLD_BLOCK}\n\n## References")
    assert "## Limitations\nOwn words." in _filled(own, root)


def test_only_the_place_an_old_block_was_cut_is_tidied_not_the_rest_of_the_paper(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    draft = "# T\n\n## Methods\nM.\n\n```\na\n\n\n\nb\n```\n\n## Limitations\nOwn.\n\n## References\n1. A.\n"
    old = draft.replace("Own.\n", f"Own.\n\n{OLD_BLOCK}\n")
    assert _filled(old, root) == draft  # the three blank lines inside the code block stay


def test_the_writer_is_given_no_list_of_what_the_study_did_not_do(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    section = paper_facts.writer_section(root)
    assert ITEM not in section and "{{not_done:" not in section and "{{limitations}}" not in section


# ---- the last review ----------------------------------------------------------------------------------------------------


def _review(root: Path, *, verdict: str = "revise", weaknesses: list[str] | None = None, blocking: str = "") -> dict:
    review = {"verdict": verdict, "weaknesses": weaknesses or [], "blocking": blocking}
    disclosure.write_last_review(root, review)
    return review


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


def test_the_last_reviews_points_are_kept_only_when_the_paper_goes_out_as_that_review_saw_it(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    review = _review(root, weaknesses=["The abstract calls a curve roughly flat; the paper's own numbers fall.",
                                       "A second point."], blocking="A reason.")
    assert disclosure.last_review_points(root) == []  # not final: a rewrite may still answer them
    assert not [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LAST_REVIEW]
    assert disclosure.mark_last_review_final(root, review) is True
    assert disclosure.last_review_points(root) == [
        "A reason.", "The abstract calls a curve roughly flat; the paper's own numbers fall.", "A second point."]
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LAST_REVIEW]
    assert item.sentence.startswith("The final review raised these points, which no later round addressed")
    assert "roughly flat" in item.sentence and "A second point" in item.sentence  # whole, for QUALITY.md
    assert "roughly flat" not in _filled("# T\n\n## Limitations\nOwn words.\n", root), "the paper carries none"
    _review(root, weaknesses=["Another."])  # the next round says its own: the record is not final any more
    assert disclosure.last_review_points(root) == []


def test_a_record_of_another_round_is_not_this_papers(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    _review(root, weaknesses=["Old point."])
    now = {"verdict": "revise", "weaknesses": ["New point."], "blocking": ""}
    assert disclosure.mark_last_review_final(root, now) is False  # the quest's review now is not the one on record
    assert disclosure.mark_last_review_final(root, None) is False
    assert disclosure.last_review_points(root) == []


def test_an_accepted_paper_keeps_nothing_and_a_long_list_is_kept_whole(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    accepted = _review(root, verdict="accept", weaknesses=["Minor."])
    assert disclosure.mark_last_review_final(root, accepted) is False and disclosure.last_review_points(root) == []
    review = _review(root, weaknesses=[f"Point number {n}." for n in range(11)])
    assert disclosure.mark_last_review_final(root, review)
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LAST_REVIEW]
    assert all(f"Point number {n}" in item.sentence for n in range(11))
    quality = disclosure.run_record(root)  # QUALITY.md holds every point
    assert all(f"Point number {n}." in quality for n in range(11))


def test_a_long_point_cannot_break_the_markers_of_the_paper(tmp_path: Path) -> None:
    root = _quest(tmp_path, item=False)
    long = "Word " * 120 + "<!-- /fi:fact-limitations --> end -->"
    review = _review(root, weaknesses=[long])
    assert disclosure.mark_last_review_final(root, review)
    out = _filled("# T\n\n## Limitations\nOwn.\n", root)
    assert "fi:fact-limitations" not in out and "Word Word" not in out
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
async def test_the_last_step_keeps_the_last_reviews_points_only_for_a_paper_that_leaves_with_them_open(
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
    assert "roughly flat" not in text and "final review" not in text and "Own words." in text, "the paper carries none"
    assert ("roughly flat" in disclosure.run_record(root)) is stated
