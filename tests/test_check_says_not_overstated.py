"""The check at finer numerical settings does not say an improvement "holds" when FI's own record shows it is not more than
the numerical error of the two designs (the improvement passed the plan's threshold only). No model is called."""

from __future__ import annotations

import json
from pathlib import Path

from core import best_design_report, optimum_check

HOLDS = "The improvement over the baseline holds at finer numerical settings: 12 nm better than the baseline, more than 5 nm."


def _record(beyond: bool | None) -> dict:
    return {"verdict": "verified", "says": HOLDS, "checks": {},
            "improvement": {"value": 12.0, "numerical_error": 15.0, "beyond_numerical_error": beyond}}


def test_not_beyond_the_error_is_not_said_to_hold() -> None:
    says = optimum_check.says_of(_record(False))
    assert "holds" not in says
    assert "not shown to be more than their numerical error" in says
    assert says.endswith(": 12 nm better than the baseline, more than 5 nm.")


def test_beyond_the_error_still_holds() -> None:
    assert optimum_check.says_of(_record(True)) == HOLDS
    assert optimum_check.says_of(_record(None)) == HOLDS
    assert optimum_check.verdict_phrase("verified", {"beyond_numerical_error": True}) == optimum_check.VERDICTS["verified"]


def test_other_verdicts_are_untouched() -> None:
    rec = {"verdict": "improvement_not_shown", "says": "x", "improvement": {"beyond_numerical_error": False}}
    assert optimum_check.says_of(rec) == "x"
    assert optimum_check.verdict_phrase("unverified", {"beyond_numerical_error": False}) == optimum_check.VERDICTS["unverified"]


def test_the_one_line_summary_follows_the_record(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "needs").mkdir()
    best = {"objective": {"quantity": "q", "unit": "nm"}, "best": {"design": {"x": 1}}, "baseline": {"objective": 1},
            "improvement": {"value": 12.0}}
    (tmp_path / "results" / "best_design.json").write_text(json.dumps(best), encoding="utf-8")
    for beyond, expect in ((False, "not shown to be more than their numerical error"), (True, "it holds at finer")):
        rec = _record(beyond)
        rec["best"] = {"design": {"x": 1}}
        (tmp_path / "needs" / "OPTIMUM_CHECK.json").write_text(json.dumps(rec), encoding="utf-8")
        line = best_design_report.summary_line(tmp_path)
        assert expect in line
        assert ("holds" in line) is bool(beyond)
