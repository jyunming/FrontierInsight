"""A reason a model wrote is quoted whole in the paper's limitations or left out; FI never cuts it in the middle of a word
or a bracket (a delivered paper read "gain, drift and spread are hard-coded to )."). No model is called."""

from __future__ import annotations

import json
from pathlib import Path

from core import paper_facts

NOTE = ("the value is filled only for the first quantity, from FI's check record; the others are hard-coded to None, and "
        "the convergence study is never called, so the other headline numbers get no quadrature over the convergence "
        "settings of the run, which the plan asked for in every cell of the grid, for every source")


def _root(tmp_path: Path) -> Path:
    (tmp_path / ".fi").mkdir()
    (tmp_path / "needs").mkdir()
    return tmp_path


def _left_out(root: Path) -> str:
    [item] = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LEFT_OUT]
    return item.sentence


def test_a_long_note_is_whole_never_cut(tmp_path: Path) -> None:
    assert 200 < len(NOTE) < 400
    root = _root(tmp_path)
    (root / "needs" / "RESULT_CHECK.json").write_text(
        json.dumps({"checks": [{"quantity": "k_hat", "note": NOTE}]}), encoding="utf-8")
    sentence = _left_out(root)
    assert "hard-coded to None" in sentence and sentence.count("(") == sentence.count(")")
    assert NOTE in sentence


def test_a_note_too_long_to_quote_is_left_out_not_cut(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "needs" / "RESULT_CHECK.json").write_text(
        json.dumps({"checks": [{"quantity": "k_hat", "note": "x " * 400}]}), encoding="utf-8")
    assert _left_out(root).endswith("the value is wrong.")


def test_a_short_note_is_still_quoted(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "needs" / "RESULT_CHECK.json").write_text(
        json.dumps({"checks": [{"quantity": "k_hat", "note": "off by a factor"}]}), encoding="utf-8")
    assert "(off by a factor)" in _left_out(root)
