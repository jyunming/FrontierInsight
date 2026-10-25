"""The run time and the topic's time limit are judged against what a reader's run of the delivered code takes: its
settings run one after another. When FI ran them at the same time (it recorded how many), the paper says both times, the
one-after-another time is a lower bound, and a limit is never "within" on a lower bound
(``core/run_facts.py``: ``reader_run``, read by ``core/paper_facts.py`` and ``core/disclosure.py``)."""

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
    assert "35.4" in text and "16 settings run at the same time" in text and "at least 112" in text
    assert known["time_limit_met"].text == "over"
    sentence = disclosure._time_sentence(root, TOPIC, 35.4)
    assert "within" not in sentence and "over" in sentence and "112" in sentence and "35.4" in sentence


def test_a_lower_bound_under_the_limit_is_never_within(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=12.0, total=40.0, at_once=4)
    known = paper_facts.facts(root, topic=TOPIC)
    assert known["time_limit_met"].text == "not shown to be within" and "at least 40" in known["run_time"].text
    assert "not shown to be within" in disclosure._time_sentence(root, TOPIC, 12.0)


def test_overlap_is_what_was_recorded_not_a_ratio(tmp_path: Path) -> None:
    # two settings at once, wall 100 s, trials add to 105 s: a ratio would call it one at a time and say "within" 102 s
    topic = "The whole study must run in under 102 seconds."
    root = _quest(tmp_path / "q", wall=100.0, total=105.0, at_once=2)
    assert paper_facts.facts(root, topic=topic)["time_limit_met"].text == "over"
    assert "within" not in disclosure._time_sentence(root, topic, 100.0).replace("not shown to be within", "")


def test_a_record_without_at_once_never_claims_overlap(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=35.0, total=112.0, at_once=None)
    known = paper_facts.facts(root, topic=TOPIC)
    assert known["run_time"].text == "35 s" and known["time_limit_met"].text == "within"
    assert "one after another" not in disclosure._time_sentence(root, TOPIC, 35.0)


def test_a_run_of_one_setting_at_a_time_is_unchanged(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=45.0, total=44.0, at_once=1)
    known = paper_facts.facts(root, topic=TOPIC)
    assert known["run_time"].text == "45 s" and known["time_limit_met"].text == "within"
    over = _quest(tmp_path / "r", wall=75.0, total=74.0, at_once=None)
    assert paper_facts.facts(over, topic=TOPIC)["time_limit_met"].text == "over"


def test_paper_and_run_record_read_the_same_time_with_the_search(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=20.0, total=50.0, at_once=8)
    (root / "results").mkdir()
    (root / "results" / "best_design.json").write_text(json.dumps({"time": {"seconds": 30.0}}), encoding="utf-8")
    known = paper_facts.facts(root, topic=TOPIC)
    assert "at least 80 s" in known["run_time"].text and known["time_limit_met"].text == "over"
    assert "at least 80 s" in disclosure._time_sentence(root, TOPIC, 20.0)
