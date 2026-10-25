"""The run time and the topic's time limit are judged against what a reader's run of the delivered code takes: its
settings run one after another. When FI ran them at the same time, the paper says both times and never "within the limit"
for a run that is over it one after another (``core/paper_facts.py``: ``facts``, ``core/disclosure.py``)."""

from __future__ import annotations

import json
from pathlib import Path

from core import disclosure, paper_facts, run_facts

TOPIC = "Estimate a rate. The whole study must run in under 60 seconds."


def _quest(root: Path, *, wall: float, total: float, at_once: int | None) -> Path:
    run_facts.record_trials(root, key="k", wall_s=wall, trials=100, cells=25, trial_sum_s=total, at_once=at_once)
    return root


def test_parallel_wall_under_the_limit_but_serial_over_it_is_not_within(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=35.4, total=112.0, at_once=16)
    known = paper_facts.facts(root, topic=TOPIC)
    text = known["run_time"].text
    assert "35.4" in text and "16 settings run at the same time" in text and "112" in text and "one after another" in text
    assert known["time_limit_met"].text == "over"
    sentence = disclosure._time_sentence(root, TOPIC, 35.4)
    assert "within" not in sentence and "over" in sentence and "112" in sentence and "35.4" in sentence


def test_parallel_and_serial_both_within_the_limit_says_within(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=12.0, total=40.0, at_once=4)
    known = paper_facts.facts(root, topic=TOPIC)
    assert known["time_limit_met"].text == "within" and "40" in known["run_time"].text


def test_a_run_of_one_setting_at_a_time_is_unchanged(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=45.0, total=44.0, at_once=None)
    known = paper_facts.facts(root, topic=TOPIC)
    assert known["run_time"].text == "45 s" and known["time_limit_met"].text == "within"
    assert "one after another" not in disclosure._time_sentence(root, TOPIC, 45.0)
    over = _quest(tmp_path / "r", wall=75.0, total=74.0, at_once=None)
    assert paper_facts.facts(over, topic=TOPIC)["time_limit_met"].text == "over"


def test_the_search_time_counts_in_the_serial_time(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=20.0, total=50.0, at_once=8)
    (root / "results").mkdir()
    (root / "results" / "best_design.json").write_text(json.dumps({"time": {"seconds": 30.0}}), encoding="utf-8")
    known = paper_facts.facts(root, topic=TOPIC)
    assert "80 s" in known["run_time"].text and known["time_limit_met"].text == "over"
