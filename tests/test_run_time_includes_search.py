"""The run time FI states counts the search for the best design and its check, from FI's own records of them, not only
the settings run before it (``core/paper_facts.py``: ``measured_run_time``, ``facts``)."""

from __future__ import annotations

import json
from pathlib import Path

from core import paper_facts


def _quest(root: Path, *, search_s: float | None, check_s: float | None) -> Path:
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "ledger.jsonl").write_text(
        "\n".join(json.dumps({"cell": i, "duration_s": 2.0}) for i in range(3)) + "\n", encoding="utf-8")
    if search_s is not None:
        (root / "results").mkdir()
        (root / "results" / "best_design.json").write_text(json.dumps({"time": {"seconds": search_s}}), encoding="utf-8")
    if check_s is not None:
        (root / "needs").mkdir()
        (root / "needs" / "OPTIMUM_CHECK.json").write_text(json.dumps({"time": {"seconds": check_s}}), encoding="utf-8")
    return root


def test_the_search_and_its_check_are_in_the_stated_time(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", search_s=5000.0, check_s=600.0)
    assert paper_facts.measured_run_time(root, None) == (None, 5606.0)
    said = paper_facts.facts(root)["run_time"].text
    assert "5,606" in said and "the search" in said and "trials alone" not in said


def test_a_study_without_a_search_keeps_the_trials_alone(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", search_s=None, check_s=None)
    assert paper_facts.measured_run_time(root, None) == (None, 6.0)
    assert "the trials alone" in paper_facts.facts(root)["run_time"].text


def test_a_record_that_holds_no_time_adds_nothing(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", search_s=None, check_s=None)
    (root / "results").mkdir()
    (root / "results" / "best_design.json").write_text("{not json", encoding="utf-8")
    assert paper_facts.measured_run_time(root, None) == (None, 6.0)
