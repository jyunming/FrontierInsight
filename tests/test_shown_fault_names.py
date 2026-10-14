"""A quantity the code reader called missing, which the run's results hold under another name, is a difference of names
and does not stop the quest (replay of suite r8 first_order_kinetics: the quest stopped without a paper on F-b5e6edb8,
"k_relative_bias is not reported", while the results held the same per-temperature values as
``k_relative_bias_by_temp``). The r7 EUV run, whose typed-in dose-to-size (a problem of logic) was a real fault, is
untouched."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from core import code_findings
from core import code_review

FIXTURES = Path(__file__).parent / "fixtures" / "shown_fault_names"


def _load(name: str, tmp_path: Path) -> tuple[Path, dict, object]:
    shutil.copytree(FIXTURES / name, tmp_path, dirs_exist_ok=True)
    return (tmp_path, code_review.load(tmp_path),
            json.loads((tmp_path / "result_json.json").read_text(encoding="utf-8")))


def test_r8_s_names_are_found_in_the_results(tmp_path: Path) -> None:
    _root, record, result = _load("kinetics_r8", tmp_path)
    issues = record["issues"]
    assert code_review.reported_under_another_name(record, issues["F-b5e6edb8"], result) == ["k_relative_bias_by_temp"]
    assert code_review.reported_under_another_name(record, issues["F-fe5d51ce"], result) == ["k_95_coverage_by_temp"]
    # not held under any name the check can tell: still a missing quantity
    assert code_review.reported_under_another_name(record, issues["F-650ad5f3"], result) == []
    # a problem of logic (a confidence interval set to null) is never a difference of names
    assert code_review.reported_under_another_name(record, issues["F-3af8cc05"], result) == []


def test_r7_s_shown_faults_are_untouched(tmp_path: Path) -> None:
    _root, record, result = _load("euv_r7", tmp_path)
    shown = code_findings.shown_open(record)
    assert "F-bed59bff" in {i["id"] for i in shown}  # the typed-in dose-to-size
    assert all(code_review.reported_under_another_name(record, i, result) == [] for i in shown)


def test_the_gate_does_not_stop_on_a_name_and_says_why(tmp_path: Path) -> None:
    from tests.test_disclosure import _engine

    root, _record, result = _load("kinetics_r8", tmp_path)
    eng = _engine(root)
    verdict = asyncio.run(eng._shown_fault_verdict(  # type: ignore[arg-type]
        {"exec_result": {"returncode": 0}, "result_json": result}))
    # It still stops on a fault the names do not explain (the finding tests are not in the fixture, so the first one
    # left is cited as "no longer on disk"), never on the two quantities held under another name.
    assert verdict is not None and verdict["stuck_reason"] == "shown_fault_open"
    assert "k_relative_bias" not in verdict["rationale"] and "k_95_coverage" not in verdict["rationale"]
    saved = code_review.load(root)["issues"]
    assert saved["F-b5e6edb8"]["changes_result"] is False
    assert saved["F-b5e6edb8"]["reported_as"] == ["k_relative_bias_by_temp"]
    assert saved["F-650ad5f3"]["changes_result"] is True
    assert {"F-b5e6edb8", "F-fe5d51ce"}.isdisjoint({i["id"] for i in code_findings.shown_open(code_review.load(root))})
