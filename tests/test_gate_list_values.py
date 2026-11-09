"""A setting whose value is a list (or a dict) is judged like any other setting (core/paper_values.py).

The analysis gate fills the plan's result quantities from FI's records; a mean over ``axis=all`` first groups the cells
by the axis's value. That grouping used the value itself as a dict key, which raised ``TypeError: unhashable type:
'list'`` for a value that is a list, and the gate then did not run at all.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import paper_values as pv

KEY = "{{v:score:mean @ pattern=all, alpha=all}}"


def _cell(setting: dict[str, Any], values: list[float]) -> dict[str, Any]:
    n = len(values)
    return {"cell": setting, "key": json.dumps(setting, sort_keys=True), "planned": n, "ok": n, "failed": 0,
            "metrics": {"score": {"values": values, "trials": list(range(n)), "count": n, "total": float(sum(values)),
                                  "non_finite": 0}}}


def _records(tmp_path: Path, cells: list[dict[str, Any]]) -> pv.Records:
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"schema": "fi.trials/v1", "thresholds": {}, "cells": cells}),
                                                  encoding="utf-8")
    rec = pv.load(tmp_path)
    assert rec is not None
    return rec


SAME = [1.0, 1.01, 0.99]
FAR = [5.0, 5.01, 4.99]


def test_list_valued_setting_pooled_mean_does_not_raise(tmp_path):
    rec = _records(tmp_path, [_cell({"pattern": [[0, 1], [1, 0]], "alpha": 1}, SAME),
                              _cell({"pattern": [[1, 1], [0, 0]], "alpha": 1}, SAME)])
    got = pv.resolve(KEY, rec)
    assert got.filled, got


def test_list_valued_setting_that_changes_the_result_is_refused_by_name(tmp_path):
    rec = _records(tmp_path, [_cell({"pattern": [[0, 1], [1, 0]], "alpha": 1}, SAME),
                              _cell({"pattern": [[1, 1], [0, 0]], "alpha": 1}, FAR)])
    got = pv.resolve(KEY, rec)
    assert not got.filled and "changes with pattern" in got.hit.reason


def test_list_valued_setting_nested_deeper(tmp_path):
    rec = _records(tmp_path, [_cell({"pattern": [[[0], [1]], [[1], [0]]], "alpha": 1}, SAME),
                              _cell({"pattern": [[[1], [1]], [[0], [0]]], "alpha": 1}, SAME)])
    assert pv.resolve(KEY, rec).filled


def test_list_value_in_a_grid_next_to_scalars(tmp_path):
    cells = [_cell({"pattern": p, "alpha": a}, SAME) for p in ([0, 1], [1, 0]) for a in (1, 2)]
    rec = _records(tmp_path, cells)
    assert pv.resolve(KEY, rec).filled
    assert pv.resolve("{{v:score:mean @ pattern=all, alpha=1}}", rec).filled


def test_dict_valued_setting(tmp_path):
    rec = _records(tmp_path, [_cell({"pattern": {"rows": 2, "fill": [0, 1]}, "alpha": 1}, SAME),
                              _cell({"pattern": {"rows": 3, "fill": [1, 1]}, "alpha": 1}, FAR)])
    got = pv.resolve(KEY, rec)
    assert not got.filled and "changes with pattern" in got.hit.reason


def test_equal_lists_are_one_level_even_when_written_with_other_numeric_types(tmp_path):
    # [1] and [1.0] are equal, so they are one level of the setting: nothing changes between levels, so nothing is refused
    rec = _records(tmp_path, [_cell({"pattern": [1], "alpha": 1}, SAME), _cell({"pattern": [1.0], "alpha": 1}, FAR)])
    got = pv.resolve(KEY, rec)
    assert got.filled, got
    rec2 = _records(tmp_path / "b", [_cell({"pattern": [[1, 0]], "alpha": 1}, SAME),
                                     _cell({"pattern": [[True, False]], "alpha": 1}, FAR)])
    assert pv.resolve(KEY, rec2).filled


def test_levels_counted_the_same_by_values_of_and_by_the_pooled_check(tmp_path):
    rec = _records(tmp_path, [_cell({"pattern": [1], "alpha": 1}, SAME), _cell({"pattern": [1.0], "alpha": 1}, SAME)])
    assert len(rec.values_of("score", "pattern")) == 1


def test_missing_words_names_settings_whose_value_is_a_list():
    words = pv._missing_words([([1],)], ["pattern"], [([1],), ([2],)])
    assert words == "pattern = [1]"
    both = pv._missing_words([([1], 1), ([2], 1)], ["pattern", "alpha"], [([1], 1), ([2], 1), ([3], 1)])
    assert "pattern = [1] or [2]" in both


def test_missing_words_equal_lists_are_one_value():
    assert pv._missing_words([([1],), ([1.0],)], ["pattern"], [([1],), ([2],)]) == "pattern = [1]"


class _Table:
    def __init__(self, cells):
        self.cells = cells

    def keys(self):
        return list(self.cells)

    def setting(self, k):
        return self.cells[k]

    def mean(self, rid, k):
        return 1.0 + len(k)


def test_default_figure_series_over_a_list_valued_setting():
    from core import analysis_figures as fig
    cells = {"a": {"x": 1, "pattern": [1, 2]}, "b": {"x": 2, "pattern": [1, 2]},
             "c": {"x": 1, "pattern": [3]}, "d": {"x": 2, "pattern": [3]}}
    lines = fig._default_lines({"x": "x", "series": "pattern", "shows": ["r"]}, _Table(cells), {})
    assert sorted(str(line[5]) for line in lines) == ["[1, 2]", "[3]"]


def test_one_form_for_the_values_of_a_setting():
    from core import run_manifest as rm
    assert rm.freeze([1]) == rm.freeze([1.0]) == rm.freeze([True])
    assert hash(rm.freeze({"a": [1, {"b": [2]}]})) == hash(rm.freeze({"a": [1.0, {"b": [2.0]}]}))
    assert rm.freeze([1]) != rm.freeze({"a": 1}) != rm.freeze([2])
    assert rm.distinct([[1], [1.0], {"a": [1]}, 5, 5.0, [2]]) == [[1], {"a": [1]}, 5, [2]]
    assert rm.freeze("x") == "x" and rm.freeze(None) is None
    assert fig_same_form([[1], [1.0], {"a": [1]}])


def fig_same_form(values):
    from core import analysis_figures as fig
    from core import run_manifest as rm
    return fig._distinct(values) == rm.distinct(values)


def test_scalar_settings_behave_as_before(tmp_path):
    same = _records(tmp_path, [_cell({"pattern": 1, "alpha": 1}, SAME), _cell({"pattern": 2, "alpha": 1}, SAME)])
    assert pv.resolve(KEY, same).filled
    far = _records(tmp_path / "far", [_cell({"pattern": 1, "alpha": 1}, SAME), _cell({"pattern": 2, "alpha": 1}, FAR)])
    got = pv.resolve(KEY, far)
    assert not got.filled and "changes with pattern" in got.hit.reason
