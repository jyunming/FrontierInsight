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


def test_a_paper_over_its_page_limit_after_the_rewrites_says_so_with_the_measured_pages(tmp_path: Path) -> None:
    root = _quest(tmp_path, [])
    assert not [i for i in paper_facts.required_items(root) if i.kind == paper_facts.PAGE_LIMIT]
    paper_facts.write_page_record(root, 7, 4, final=False)  # a draft a rewrite will still shorten: nothing to state
    assert not [i for i in paper_facts.required_items(root) if i.kind == paper_facts.PAGE_LIMIT]
    paper_facts.write_page_record(root, 4, 4, final=True)  # within the limit: nothing to state
    assert not [i for i in paper_facts.required_items(root) if i.kind == paper_facts.PAGE_LIMIT]
    paper_facts.write_page_record(root, 7, 4, final=True)
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.PAGE_LIMIT]
    assert "7 pages" in item.sentence and "limit of 4" in item.sentence
    paper = "# T\n\n## Methods\nM.\n\n## Discussion\n**Limitations.** Few.\n\n## References\n1. A.\n"
    out = paper_facts.render(paper_facts.place_items(paper, root), root).text
    assert "The paper is 7 pages long, over the topic's limit of 4" in out
    assert out.index("## Limitations") < out.index("The paper is 7 pages") < out.index("## References")


async def test_the_review_puts_the_page_overrun_in_the_paper_and_measures_the_paper_that_carries_it(tmp_path: Path) -> None:
    """The last review of a paper still over the limit: the record, the sentence in the Limitations (the paper's bold
    "Limitations." paragraph is no section), and the page count the sentence states is the final paper's."""
    from tests.test_plan_optimisation import _engine

    eng = _engine(tmp_path, [])
    root = eng.quest_root
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    paper = root / "paper.md"
    paper.write_text("# T\n\n## Methods\nM.\n\n## Discussion\n**Limitations.** Few.\n\n## References\n1. A.\n",
                     encoding="utf-8")
    seen: list[str] = []

    async def measure(path: Path, state: dict) -> dict:
        seen.append(Path(path).read_text(encoding="utf-8"))
        return {"pages": 7 if "pages long" in seen[-1] else 6}  # the sentence tips the paper onto one more page

    eng._measure_draft_pages = measure  # type: ignore[method-assign]
    pages = await eng._state_page_overrun({}, paper, 4, 6)
    text = paper.read_text(encoding="utf-8")
    assert pages == 7 and "The paper is 7 pages long, over the topic's limit of 4" in text
    assert json.loads((root / ".fi" / "page_limit.json").read_text()) == {"pages": 7, "limit": 4, "final": True}
    assert text.count("pages long") == 1  # re-stated, not added again
