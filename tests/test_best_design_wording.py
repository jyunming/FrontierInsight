"""The paper's *Best design found* section is written for the paper's reader: no file path, no FI identifier of a
search method, and no negative zero (``core/best_design_report.py``). Neutral made-up records."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core import best_design_report as report


def _best(method: str = "bounded_local", why: str = "", improvement: float = 0.5) -> dict:
    return {
        "objective": {"quantity": "yield", "unit": "kg", "direction": "maximise"},
        "baseline": {"design": {"a": 1.0}, "objective": 10.0},
        "best": {"design": {"a": 1.5}, "objective": 10.5},
        "improvement": {"value": improvement},
        "evaluations": {"search": 40, "budget": 100, "stopped_because": "converged"},
        "method": {"requested": method, "used": method, "why": why},
    }


def _section(monkeypatch, best: dict, check: dict | None = None) -> str:
    monkeypatch.setattr(report, "records", lambda _root: {"best_design": best, "optimum_check": check})
    return report.section(Path("."))


def test_the_section_names_no_file(monkeypatch) -> None:
    text = _section(monkeypatch, _best())
    assert "FI's own record of the search" in text
    for path in ("results/", "raw/", "needs/", ".json", ".jsonl"):
        assert path not in text


@pytest.mark.parametrize("method_id", ["bounded_local", "global_then_local", "exhaustive",
                                       "scipy:differential_evolution", "optuna:tpe", "something_new"])
def test_the_section_names_no_method_id(monkeypatch, method_id: str) -> None:
    text = _section(monkeypatch, _best(method=method_id))
    assert method_id not in text
    assert re.search(r"Method: \S", text)


def test_a_known_method_reads_as_what_it_does() -> None:
    assert report.method_name("bounded_local").startswith("a local search")
    assert "SciPy" in report.method_name("scipy:differential_evolution")
    assert report.method_name("") == report.method_name("never_seen_before")


def test_an_id_inside_the_reason_is_replaced(monkeypatch) -> None:
    text = _section(monkeypatch, _best(method="exhaustive", why="too many values; the built-in bounded_local search was used"))
    assert "bounded_local" not in text and "exhaustive" not in text
    assert "a local search inside the ranges" in text


def test_a_negative_zero_prints_as_zero(monkeypatch) -> None:
    assert report._fmt(-0.0) == "0"
    assert report._fmt(-0.00001) == "-1e-05"
    assert report._fmt(-0.5) == "-0.5"
    text = _section(monkeypatch, _best(improvement=-0.0))
    assert not re.search(r"(?<![\d.])-0(?:\.0+)?(?![\d.e])", text)


def test_the_one_line_for_a_zero_improvement(monkeypatch) -> None:
    monkeypatch.setattr(report, "records", lambda _root: {"best_design": _best(improvement=-0.0), "optimum_check": None})
    line = report.summary_line(Path("."))
    assert "-0" not in line and "no better than the baseline" in line
