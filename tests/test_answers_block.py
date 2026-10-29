"""The block ``{{answers}}``: FI states, from its own records, the value of each quantity the topic asks for (core/paper_facts.py).
Neutral made-up study: a settling limit the analysis works out, a decay rate measured per trial, one quantity the plan maps to
nothing. No model is called."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import goal_coverage as gc
from core import paper_facts as pf
from core import paper_values as pv

PROTOCOL = {
    "metrics": [
        {"id": "decay_rate", "label": "Decay rate", "kind": "mean"},
        {"id": "settling_limit", "label": "Limit of the settling time (hours)", "kind": "mean"},
        {"id": "settling_gap", "label": "Gap of the settling time", "kind": "mean"},
    ],
    "topic_map": [
        {"item": "Q1", "by": "metric:decay_rate"},
        {"item": "Q2", "by": "metric:settling_limit"},
        {"item": "Q4", "by": "metric:missing_metric"},
    ],
}
REQ = {"quantities": [
    {"name": "decay rate", "stated": "report the decay rate", "main": True},
    {"name": "settling limit", "stated": "give the limit of the settling time", "main": True},
    {"name": "colour of the sample", "stated": "say which colour", "main": True},
    {"name": "mapped to nothing real", "stated": "x", "main": True},
    {"name": "a side quantity", "stated": "y", "main": False},
]}


def _quest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settling: float | None = 12.3456) -> Path:
    (tmp_path / "raw").mkdir(parents=True)
    cells = [{"cell": {"dose": d}, "key": f"dose={d}", "planned": 2, "ok": 2, "failed": 0,
              "metrics": {"decay_rate": {"values": [0.1 * d, 0.1 * d + 0.01], "trials": [0, 1], "count": 2, "total": 0.0, "non_finite": 0}}}
             for d in (1, 2, 3)]
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(REQ), encoding="utf-8")
    if settling is not None:
        pv.write_answers(tmp_path, {"settling_limit": settling}, PROTOCOL)
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: PROTOCOL)
    return tmp_path


def test_each_main_quantity_gets_its_value_or_is_said_as_not_answered(tmp_path, monkeypatch) -> None:
    text = pf.answers_text(_quest(tmp_path, monkeypatch))
    assert text.startswith("**Answers to the questions asked.**")
    assert "- decay rate (decay_rate): {{v:decay_rate @ dose=each | range}} (no single value: it differs with dose; this is the mean in each setting, lowest to highest" in text
    assert "- settling limit (settling_limit): {{v:settling_limit}} (one value worked out by the analysis; no interval" in text
    assert "- colour of the sample: not answered (the plan maps it to no result of the study)" in text
    assert "- mapped to nothing real: not answered" in text
    assert "side quantity" not in text  # only a quantity the topic marks main


def test_a_quantity_whose_value_is_not_recorded_is_not_answered_and_no_key_is_offered(tmp_path, monkeypatch) -> None:
    text = pf.answers_text(_quest(tmp_path, monkeypatch, settling=None))
    assert "- settling limit: not answered (FI holds no value for settling_limit)" in text
    assert "{{v:settling_limit" not in text


def test_the_block_is_placed_and_filled_and_the_asked_values_count_as_placed(tmp_path, monkeypatch) -> None:
    """Before: the writer typed the settling limit, no key placed it, and the review was told the paper never gives it."""
    root = _quest(tmp_path, monkeypatch)
    draft = "# Title\n\n## Results\n\nThe settling limit is 12.35 hours.\n\n## Discussion\n\nIt settles.\n"
    rec = pv.records_of(root, PROTOCOL)
    old = gc.unplaced_answers(REQ, PROTOCOL, rec.fillable(), pv.placed_metrics(draft))
    assert {a["metric"] for a in old} == {"decay_rate", "settling_limit"}  # the old check says the paper never gives them
    placed = pf.place_blocks(draft, root)
    assert placed.index("{{answers}}") < placed.index("## Discussion")
    out = pf.render(placed, root, records=rec)
    assert not out.unfilled and "<!-- fi:v {{v:settling_limit}} -->12.35<!-- /fi:v -->" in out.text
    assert gc.unplaced_answers(REQ, PROTOCOL, rec.fillable(), pv.placed_metrics(out.text)) == []
    assert pv.placed_metrics(out.text) >= {"settling_limit"}


def test_a_second_pass_changes_nothing_and_the_block_is_not_put_in_twice(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch)
    rec = pv.records_of(root, PROTOCOL)
    once = pf.render(pf.place_blocks("## Results\n\nText.\n\n## Discussion\n\nMore.\n", root), root, records=rec).text
    again = pf.render(pf.place_blocks(once, root), root, records=rec).text
    assert again == once and once.count("Answers to the questions asked") == 1


def test_a_topic_that_asks_for_no_main_quantity_has_no_block(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch)
    (root / ".fi" / "topic_requirements.json").write_text(json.dumps({"quantities": [{"name": "x", "main": False}]}), encoding="utf-8")
    assert pf.answers_text(root) == "" and "{{answers}}" not in pf.place_blocks("## Results\n\nText.\n", root)


def test_the_writer_is_told_about_the_block(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch)
    assert "{{answers}}" in pf.writer_section(root)


def test_a_map_the_name_does_not_confirm_is_not_an_answer(tmp_path, monkeypatch) -> None:
    """Before: a ``judge`` map (the metric exists but its name is unrelated to the quantity) put a wrong metric in under the
    quantity's name."""
    root = _quest(tmp_path, monkeypatch)
    wrong = {**PROTOCOL, "topic_map": PROTOCOL["topic_map"] + [{"item": "Q3", "by": "metric:decay_rate"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: wrong)
    text = pf.answers_text(root)
    assert "- colour of the sample: not answered (the plan's map of it to decay_rate is not confirmed" in text
    assert "colour of the sample (decay_rate)" not in text
    assert "- decay rate (decay_rate): {{v:decay_rate" in text  # an ok map is still answered


def test_a_varied_study_has_no_single_value_and_the_block_says_so(tmp_path, monkeypatch) -> None:
    text = pf.answers_text(_quest(tmp_path, monkeypatch))
    assert "no single value: it differs with dose" in text and "the tables give every setting" in text
