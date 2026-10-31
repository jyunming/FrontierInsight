"""What the s3 papers got wrong in the items FI states itself (core/paper_facts.py): a precision line that read "is ±0.01,
wider than the plan's target of ±0.01" and ran one line per setting with the analysis's dotted path as the name, and a
page limit the paper missed without saying so. No model is called."""

from __future__ import annotations

import json
from pathlib import Path

from core import frozen_protocol, paper_facts

PROTOCOL = {"grid": {"m": [20, 200]}, "metrics": [
    {"id": "any_false_bh", "label": "P(any false discovery), Benjamini-Hochberg", "kind": "proportion"},
    {"id": "any_false_bonf", "label": "P(any false discovery), Bonferroni", "kind": "proportion"},
]}


def _quest(tmp_path: Path, missed: list[dict]) -> Path:
    (tmp_path / ".fi").mkdir()
    frozen_protocol.freeze(tmp_path, PROTOCOL, approved_by="test", source="test")
    (tmp_path / ".fi" / "precision_missed.json").write_text(json.dumps(missed), encoding="utf-8")
    return tmp_path


def _row(name: str, half_width: float) -> dict:
    return {"name": name, "half_width": half_width, "target": 0.01, "trials": 2000}


def test_a_precision_miss_is_one_sentence_per_metric_with_numbers_that_differ(tmp_path: Path) -> None:
    root = _quest(tmp_path, [_row("by_m.20.pi1_0.any_false_bh", 0.010018), _row("by_m.20.pi1_0.05.any_false_bh", 0.0218),
                             _row("by_m.200.pi1_0.any_false_bh", 0.0104), _row("by_m.20.pi1_0.any_false_bonf", 0.0123)])
    items = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.PRECISION]
    assert len(items) == 2  # was one per setting
    bh = next(i for i in items if "Benjamini" in i.sentence)
    assert "by_m" not in bh.sentence and "any_false_bh" not in bh.sentence  # the plan's own words, not the output's path
    assert "P(any false discovery), Benjamini-Hochberg" in bh.sentence and "in 3 of the settings" in bh.sentence
    assert "between ±0.01002 and ±0.0218" in bh.sentence and "target of ±0.01" in bh.sentence
    assert "from 2,000 trials each" in bh.sentence
    one = next(i for i in items if "Bonferroni" in i.sentence)
    assert "is ±0.0123 in one setting" in one.sentence


def test_a_name_that_is_no_metric_of_the_plan_stays_its_own_item_and_still_tells_the_numbers_apart(tmp_path: Path) -> None:
    root = _quest(tmp_path, [_row("coverage_lnA", 0.0100185)])
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.PRECISION]
    assert "`coverage_lnA`" in item.sentence and "±0.01002" in item.sentence and "target of ±0.01" in item.sentence


def test_a_paper_over_its_page_limit_after_the_rewrites_is_told_to_the_quality_report_not_the_paper(tmp_path: Path) -> None:
    root = _quest(tmp_path, [])
    assert paper_facts.page_limit_text(root) == ""
    paper_facts.write_page_record(root, 7, 4, final=False)  # a draft a rewrite will still shorten: nothing to state
    assert paper_facts.page_limit_text(root) == ""
    paper_facts.write_page_record(root, 4, 4, final=True)  # within the limit: nothing to state
    assert paper_facts.page_limit_text(root) == ""
    paper_facts.write_page_record(root, 7, 4, final=True)
    assert paper_facts.page_limit_text(root) == "This paper is 7 pages, over the 4-page limit set for it."
    paper = "# T\n\n## Methods\nM.\n\n## Discussion\n**Limitations.** Few.\n\n## References\n1. A.\n"
    out = paper_facts.render(paper_facts.place_blocks(paper, root), root).text
    assert "pages, over the" not in out and "page limit" not in out.lower()


async def test_the_page_overrun_is_recorded_and_the_paper_is_left_as_it_is(tmp_path: Path) -> None:
    from tests.test_plan_optimisation import _engine

    eng = _engine(tmp_path, [])
    root = eng.quest_root
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    paper = root / "paper.md"
    body = "# T\n\n## Methods\nM.\n\n## Discussion\n**Limitations.** Few.\n\n## References\n1. A.\n"
    paper.write_text(body, encoding="utf-8")
    assert await eng._state_page_overrun({}, paper, 4, 6) == 6
    assert paper.read_text(encoding="utf-8") == body
    assert json.loads((root / ".fi" / "page_limit.json").read_text()) == {"pages": 6, "limit": 4, "final": True}
    assert "6 pages, over the 4-page limit" in paper_facts.page_limit_text(root)


async def test_a_paper_that_leaves_the_review_over_its_page_limit_is_recorded_whatever_route_brought_it(
        tmp_path: Path, monkeypatch) -> None:
    """Round s4: both papers left the review with "shortening 2 of 2" still pending and other findings out of rounds, so
    the review never reached the step that states the overrun, and 6- and 7-page papers shipped silent. The last step
    before delivery measures the delivered paper itself."""
    import core.engine as engine_mod
    from tests.test_plan_optimisation import _engine

    eng = _engine(tmp_path, [])
    root = eng.quest_root
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    paper = root / "paper.md"
    body = "# T\n\n## Methods\nM.\n\n## Limitations\nFew.\n\n## References\n1. A.\n"
    paper.write_text(body, encoding="utf-8")
    monkeypatch.setattr(engine_mod, "resolve_page_limit", lambda config: 4)

    async def no_citations(state: dict) -> dict:
        return {"citation_next": "done"}

    pages = {"n": 7}

    async def measure(path: Path, state: dict) -> dict:
        return {"pages": pages["n"]}

    eng._check_citations = no_citations  # type: ignore[method-assign]
    eng._measure_draft_pages = measure  # type: ignore[method-assign]
    out = await eng._node_citation_check({"paper_md": str(paper)})  # type: ignore[arg-type]
    assert out == {"citation_next": "done"}
    assert paper.read_text(encoding="utf-8") == body  # the paper itself says nothing about its length
    assert json.loads((root / ".fi" / "page_limit.json").read_text()) == {"pages": 7, "limit": 4, "final": True}

    # A paper within the limit is left as it is.
    paper.write_text(body, encoding="utf-8")
    (root / ".fi" / "page_limit.json").unlink()
    pages["n"] = 4
    await eng._node_citation_check({"paper_md": str(paper)})  # type: ignore[arg-type]
    assert "limit set for it" not in paper.read_text(encoding="utf-8") and not (root / ".fi" / "page_limit.json").exists()


def test_the_quality_report_prints_the_page_limit_line_under_what_the_paper_does_not_carry() -> None:
    from core import quality_report

    lines = quality_report._paper_record_lines({"page_limit": "This paper is 6 pages, over the 4-page limit set for it."})
    text = "\n".join(lines)
    assert "### Page limit" in text and "This paper is 6 pages, over the 4-page limit set for it." in text
    assert quality_report._paper_record_lines({"page_limit": ""}) == []
