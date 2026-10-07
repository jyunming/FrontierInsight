"""A number FI records for a search is never written as -0.0 (it printed as "-0.0" in the writer's input)."""
from __future__ import annotations

import json
from pathlib import Path

from core import optimise, optimum_check
from core import optimise_search as osearch


def test_plain_zero_rewrites_negative_zero_at_any_depth_and_nothing_else() -> None:
    out = osearch.plain_zero({"a": -0.0, "b": [1.5, -0.0, {"c": -0.0}], "d": (-0.0,), "e": -2.0, "f": 0, "g": "x", "h": None})
    assert json.dumps(out) == '{"a": 0.0, "b": [1.5, 0.0, {"c": 0.0}], "d": [0.0], "e": -2.0, "f": 0, "g": "x", "h": null}'


def test_the_improvement_the_writer_reads_is_not_negative_zero() -> None:
    record = {"best": {"design": {"x": 1.0}, "objective": 5.0}, "baseline": {"objective": 5.0},
              "improvement": {"value": -0.0}, "evaluations": {"search": 3}}
    stdout, _ = optimise.with_fi_record('RESULT_JSON: {"best_objective": 5.0}', record)
    assert '"improvement": 0.0' in stdout and "-0.0" not in stdout


def test_the_check_file_and_the_ledger_hold_no_negative_zero(tmp_path: Path) -> None:
    text = optimum_check._write(tmp_path, {"improvement": {"value": -0.0, "numerical_error": -0.0}}, "k")
    assert "-0.0" not in text
    assert "-0.0" not in (tmp_path / optimum_check.CHECK_PATH).read_text(encoding="utf-8")
    block = {"objective": {"quantity": "f"}}
    lines = osearch.ledger_lines({"method_used": "m", "method_requested": "m", "budget": 1, "starts": 1, "per_start": 1,
                                  "seed": 0, "rows": [{"objective": -0.0}]}, {**block, "numerical_settings": {}})
    assert not any("-0.0" in line for line in lines)
