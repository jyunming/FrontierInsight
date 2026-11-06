"""-381: an open code finding that names a result the paper prints is marked in the paper by a sentence FI writes from its
records; the quality report no longer calls such a finding missing from the paper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import code_review, paper_facts, quality_report

IDS = ["quantity_a", "quantity_b", "quantity_c"]


def _found(*flagged: dict[str, Any], **other: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"missing": [], "different": [], "conflicts": [], "unshown": [], "never_run": [],
                           "not_rechecked": [], "flagged": list(flagged)}
    out.update(other)
    return out


def _v(fid: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"fid": fid, "text": text, "severity": "high", "kind": "logic", **extra}


def _quest(tmp_path: Path, issues: dict[str, Any]) -> Path:
    (tmp_path / ".fi").mkdir(parents=True)
    (tmp_path / ".fi" / "code_review.json").write_text(json.dumps({"issues": issues}), encoding="utf-8")
    return tmp_path


def test_touched_results_names_the_declared_results_a_finding_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path, {})
    monkeypatch.setattr(code_review, "open_findings", lambda q, c: _found(
        _v("f1", "the spread of quantity_a and quantity_c uses n, not n - 1"),
        _v("f2", "a loop is slow", why="it also recomputes quantity_b each time"),
        _v("f3", "a variable name is unclear"),
        _v("f4", "quantity_ab is rounded")))
    got = code_review.touched_results(root, root / "code", IDS)
    assert got == {"f1": ["quantity_a", "quantity_c"], "f2": ["quantity_b"]}, "a finding naming no id marks nothing; a longer name is no match"


def test_touched_results_leaves_out_what_cannot_change_a_printed_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path, {"no": {"changes_result": False}, "yes": {"changes_result": True}})
    monkeypatch.setattr(code_review, "open_findings", lambda q, c: _found(
        _v("no", "quantity_a is wrong", changes_result=False),
        _v("yes", "quantity_a is wrong"),
        _v("never", "quantity_b is wrong", changes_result=True),
        _v("low", "quantity_c is wrong", severity="low"),
        never_run=[_v("nr", "quantity_a is wrong")], conflicts=[_v("pc", "quantity_a is wrong", plan_conflict=True)]))
    assert code_review.touched_results(root, root / "code", IDS) == {"yes": ["quantity_a"], "never": ["quantity_b"]}


def _printed_quest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _quest(tmp_path, {"f1": {"changes_result": True, "text": "quantity_a and quantity_b are scaled twice"}})
    monkeypatch.setattr(code_review, "open_findings", lambda q, c: _found(
        _v("f1", "quantity_a and quantity_b are scaled twice", changes_result=True),
        _v("f2", "an unrelated problem in the code", changes_result=True)))
    monkeypatch.setattr(paper_facts, "_declared_ids", lambda r: IDS)
    return root


PAPER = ("# P\n\n## Methods\n{{run_facts}}\n\n## Results\nThe value is {{v:quantity_a:mean_ci @ n=all}}. "
         "Another is {{v:quantity_c @ n=all}}.\n")


def test_fi_writes_a_sentence_for_a_touched_result_the_paper_prints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _printed_quest(tmp_path, monkeypatch)
    text = paper_facts.run_facts_text(root, paper_md=PAPER)
    assert text.startswith('**Run.** "quantity_a" comes from code'), "only a result the paper prints is named"
    assert "flagged" in text and "scaled twice" in text
    assert "quantity_a" not in paper_facts.run_facts_text(root), "without the paper's text nothing is marked"


def test_the_sentence_is_in_the_rendered_paper_and_the_finding_is_met(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _printed_quest(tmp_path, monkeypatch)
    out = paper_facts.render(PAPER, root).text
    assert "scaled twice" in out
    paper_facts.keep_check(root, [], out)
    unmet = [u["id"] for u in paper_facts.kept_check(root).get("unmet_items", []) if u["kind"] == paper_facts.CODE_FINDING]
    assert unmet == ["finding-f2"], "the finding that names no printed result stays unmet; the marked one is met"
    kept = paper_facts.kept_check(root)
    md = chr(10).join(quality_report._paper_record_lines({"paper_check": {"unmet_items": kept["unmet_items"]}}))
    lines = [ln for ln in md.splitlines() if ln.startswith("- Not in the paper's Limitations")]
    assert len(lines) == 1 and "`finding-f2`" in lines[0]


def test_a_finding_naming_a_result_the_paper_does_not_print_stays_unmet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _printed_quest(tmp_path, monkeypatch)
    paper = "# P\n\n## Methods\n{{run_facts}}\n\n## Results\nOnly {{v:quantity_c @ n=all}}.\n"
    out = paper_facts.render(paper, root).text
    paper_facts.keep_check(root, [], out)
    unmet = {u["id"] for u in paper_facts.kept_check(root).get("unmet_items", [])}
    assert unmet == {"finding-f1", "finding-f2"}
