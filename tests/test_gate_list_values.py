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


def test_equal_lists_are_one_value(tmp_path):
    # the same list recorded in two cells is one level of the setting, so there is nothing that changes
    rec = _records(tmp_path, [_cell({"pattern": [0, 1], "alpha": 1}, SAME), _cell({"pattern": [0, 1], "alpha": 2}, FAR)])
    assert pv.resolve("{{v:score:mean @ pattern=all, alpha=1}}", rec).filled


def test_scalar_settings_behave_as_before(tmp_path):
    same = _records(tmp_path, [_cell({"pattern": 1, "alpha": 1}, SAME), _cell({"pattern": 2, "alpha": 1}, SAME)])
    assert pv.resolve(KEY, same).filled
    far = _records(tmp_path / "far", [_cell({"pattern": 1, "alpha": 1}, SAME), _cell({"pattern": 2, "alpha": 1}, FAR)])
    got = pv.resolve(KEY, far)
    assert not got.filled and "changes with pattern" in got.hit.reason
