"""A reader sees the plan's label of a result or a setting, never its id, in what FI writes into the paper (the trend
sentences, the answers lines, the tables of every setting); an id the plan gives no label stays as it is, and every key, CSV
column and record keeps the id. Neutral made-up study: a score at several sizes. No model is called."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import paper_facts as pf
from core import paper_values as pv
from core import results_by_setting as rbs
from core import trend as tr  # noqa: F401 -- registers the trend: keys


def _metric(values: list[float]) -> dict:
    return {"values": values, "trials": list(range(len(values))), "count": len(values), "total": 0.0, "non_finite": 0}


def _cell(cell: dict, **metrics: list[float]) -> dict:
    key = ",".join(f"{k}={v}" for k, v in cell.items())
    return {"cell": cell, "key": key, "planned": 1, "ok": 1, "failed": 0, "metrics": {m: _metric(v) for m, v in metrics.items()}}


LABELLED = {"labels": {"noise_sd": "Noise level", "score_pct": "Score (percent)"},
            "metrics": [{"id": "score_pct", "label": "Score (percent)", "kind": "mean"}]}


def _rec(protocol: dict | None) -> pv.Records:
    cells = [_cell({"noise_sd": s, "group": g}, score_pct=[10.0 * s + d for d in (0.0, 0.1, 0.2)]) for s in (1, 2, 3) for g in "ab"]
    return pv.from_cells(cells, protocol)


def test_a_trend_sentence_names_the_result_and_the_setting_by_the_plans_labels() -> None:
    text = pv.resolve("trend:score_pct over noise_sd @ group=a", _rec(LABELLED)).text
    assert text.startswith("Score (percent) over Noise level at group = a: ")
    assert "score_pct" not in text and "noise_sd" not in text


def test_a_trend_sentence_keeps_an_id_the_plan_gives_no_label() -> None:
    text = pv.resolve("trend:score_pct over noise_sd @ group=a", _rec(None)).text
    assert text.startswith("score_pct over noise_sd at group = a: ")
    half = {"labels": {"noise_sd": "Noise level"}}
    assert pv.resolve("trend:score_pct over noise_sd @ group=a", _rec(half)).text.startswith("score_pct over Noise level at group = a: ")


def test_the_key_the_writer_types_is_still_the_id() -> None:
    value = pv.resolve("trend:score_pct over noise_sd @ group=a", _rec(LABELLED))
    assert value.filled
    assert not pv.resolve("trend:Score (percent) over Noise level @ group=a", _rec(LABELLED)).filled


def test_a_comparison_of_two_results_is_worded_with_their_labels() -> None:
    protocol = {"labels": {"a": "Alpha score", "b": "Beta score", "g": "Group"}}
    cells = [_cell({"g": g}, a=[1.0 * g, 1.1 * g], b=[2.0 * g, 2.2 * g]) for g in (1, 2, 3)]
    text = pv.resolve("comparison:a mean vs b mean @ g=each", pv.from_cells(cells, protocol)).text
    assert "the mean of Alpha score divided by the mean of Beta score" in text
    assert "(lowest at Group = " in text


# --- the tables of every setting ---------------------------------------------------------------------------------------


def _table_quest(tmp_path: Path, protocol: dict) -> Path:
    cells = [{"cell": {"m": m}, "key": f"m={m}", "planned": 3, "ok": 3, "failed": 0,
              "metrics": {"hit": {"values": [1.0, 0.0, 1.0]}, "other": {"values": [0.5, 0.6, 0.7]}}} for m in (1, 2)]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "simulate.py").write_text(
        "def run_trial(cell, i, seed):\n    return {'hit': float(cell['m'] == 1), 'other': 0.5}\n", encoding="utf-8")
    assert rbs.write(tmp_path, protocol) is not None
    return tmp_path


def _protocol(label: str | None, with_derived: bool = False) -> dict:
    hit = {"id": "hit", "estimand": "P(hit)", "kind": "proportion", "unit": "a run", "reference": {"formula": "p * 2"}}
    if label:
        hit["label"] = label
    protocol = {"grid": {"m": [1, 2]}, "thresholds": {"p": 0.25}, "metrics": [hit]}
    if label:
        protocol["labels"] = {"p": "Hit share"}
    if with_derived:
        protocol["metrics"].append({"id": "gone", "label": "A worked-out quantity", "estimand": "x", "kind": "mean", "unit": "r"})
    return protocol


def test_the_paper_table_is_titled_and_captioned_by_the_plans_label_and_the_files_keep_the_id(tmp_path: Path) -> None:
    root = _table_quest(tmp_path, _protocol("Hit rate"))
    paper = (root / rbs.PAPER_BLOCK).read_text(encoding="utf-8")
    assert "**Table (Hit rate).** Hit rate at each of the 2 settings" in paper
    assert "`hit`" not in paper and "Table (hit)" not in paper
    assert "Reference: (Hit share) * 2" in paper and "`p * 2`" not in paper
    files = (root / "paper" / "results_by_setting.md").read_text(encoding="utf-8")
    assert "**Table (hit).** `hit` at each of the 2 settings" in files and "Reference: `p * 2`" in files
    assert "hit," in (root / "paper" / "results_by_setting.csv").read_text(encoding="utf-8")


def test_a_table_of_a_result_with_no_label_still_shows_its_id(tmp_path: Path) -> None:
    root = _table_quest(tmp_path, _protocol(None))
    paper = (root / rbs.PAPER_BLOCK).read_text(encoding="utf-8")
    assert "**Table (hit).** `hit` at each of the 2 settings" in paper and "Reference: `p * 2`" in paper


def test_the_sentence_naming_tables_left_out_and_the_slice_use_the_labels(tmp_path: Path) -> None:
    root = _table_quest(tmp_path, _protocol("Hit rate"))
    built = rbs.from_quest(root, _protocol("Hit rate"))[1]
    both = _protocol("Hit rate")
    both["metrics"].append({"id": "other", "label": "Other rate", "estimand": "x", "kind": "mean", "unit": "r"})
    built = rbs.from_quest(root, both)[1]
    assert [t.value for t in built.tables] == ["hit", "other"]
    block = rbs.paper_block(built, cap=2)  # the first table fits, the second is named
    assert "the tables of Other rate are in" in block and "`other`" not in block


def test_the_note_on_quantities_with_no_table_of_their_own_uses_labels(tmp_path: Path) -> None:
    protocol = _protocol("Hit rate")
    protocol["metrics"].append({"id": "order_fit", "label": "Observed order", "estimand": "slope", "kind": "mean",
                                "unit": "a fit", "depends_on": ["m"], "returned": "afterwards"})
    root = _table_quest(tmp_path, protocol)
    paper = (root / rbs.PAPER_BLOCK).read_text(encoding="utf-8")
    files = (root / "paper" / "results_by_setting.md").read_text(encoding="utf-8")
    assert "no table of their own" in files and "`order_fit`" in files
    assert "no table of their own" in paper and "Observed order" in paper and "order_fit" not in paper


def test_plain_label_falls_back(tmp_path: Path) -> None:
    assert rbs.plain_label(None, "x") == "x"
    assert rbs.plain_label({"labels": {"x": "  An   x "}}, "x") == "An x"
    assert rbs.plain_label({"metrics": [{"id": "x", "label": "The x"}]}, "x") == "The x"
    assert rbs.plain_label({"labels": {}}, "x", "alias") == "alias"
    text, changed = rbs.plain_formula({"labels": {"m": "Tests"}}, "1 - 0.05/m")
    assert (text, changed) == ("1 - 0.05/Tests", True)
    assert rbs.plain_formula({"labels": {}}, "1 - 0.05/m") == ("1 - 0.05/m", False)


@pytest.mark.parametrize("name", ["score_pct"])
def test_pf_is_importable(name: str) -> None:
    assert pf.answers_text  # the answers lines are tested in test_answers_block.py


def test_a_formula_changes_only_whole_names_not_numbers_commands_or_longer_names() -> None:
    protocol = {"labels": {"e": "Energy", "p": "Hit share", "x": "Pos", "alpha": "Alpha", "q": "Rate"}}
    text, changed = rbs.plain_formula(protocol, r"1e-3 * p + x**2 - 2x + lpha_p + pq + sqrt(q)")
    assert changed
    assert text == r"1e-3 * (Hit share) + Pos**2 - 2x + lpha_p + pq + sqrt(Rate)"
    assert rbs.plain_formula({"labels": {"e": "Energy"}}, "1e-3") == ("1e-3", False)


def test_two_names_with_one_label_keep_their_ids() -> None:
    protocol = {"labels": {"a": "Score", "b": "score", "c": "Other"},
                "metrics": [{"id": "d", "label": "Other"}]}
    assert rbs.plain_label(protocol, "a") == "a" and rbs.plain_label(protocol, "b") == "b"
    assert rbs.plain_label(protocol, "c") == "c" and rbs.plain_label(protocol, "d") == "d"
    assert rbs.plain_label({"labels": {"a": "Score", "b": "Other"}}, "a") == "Score"
    rec = pv.from_cells([_cell({"a": 1, "b": 1}, m=[1.0])], protocol)
    assert rec.label("a") == "a"
