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
    assert "- decay rate (the plan's map: Decay rate): {{v:decay_rate @ dose=each | range}} (no single value: it differs with dose; this is the mean in each setting, lowest to highest" in text
    assert "- settling limit (the plan's map: Limit of the settling time (hours)): {{v:settling_limit}} (one value worked out by the analysis; no interval" in text
    assert "- colour of the sample: the plan links no result of the study to it here; see the Results" in text
    assert "- mapped to nothing real: the plan's map for it names no result the study returns" in text
    assert "not answered (the plan maps it to no result" not in text  # the run's results are not said missing
    assert "side quantity" not in text  # only a quantity the topic marks main


def test_a_quantity_whose_value_is_not_recorded_is_not_answered_and_no_key_is_offered(tmp_path, monkeypatch) -> None:
    text = pf.answers_text(_quest(tmp_path, monkeypatch, settling=None))
    assert '- settling limit: not answered (FI holds no value for "Limit of the settling time (hours)")' in text
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
    text = pf.writer_section(root)
    assert "{{answers}}" in text
    # No stand-in phrase to copy: the block has its own heading, and the writer is told not to refer to it.
    assert "see the answers" not in text and "refer to them as" not in text and "do not refer to the block" in text


def test_an_explicit_map_to_an_existing_metric_is_placed_and_named_whatever_the_name(tmp_path, monkeypatch) -> None:
    """The plan's explicit map is the fact even when the metric's name does not resemble the quantity's (``judge``): the line
    names the metric, so a wrong map is disclosed. A map to a missing metric is not answered."""
    root = _quest(tmp_path, monkeypatch)
    wrong = {**PROTOCOL, "topic_map": PROTOCOL["topic_map"] + [{"item": "Q3", "by": "metric:decay_rate"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: wrong)
    text = pf.answers_text(root)
    assert "- colour of the sample (the plan's map: Decay rate): {{v:decay_rate" in text
    assert "- mapped to nothing real: the plan's map for it names no result the study returns" in text


def test_a_varied_study_has_no_single_value_and_the_block_says_so(tmp_path, monkeypatch) -> None:
    text = pf.answers_text(_quest(tmp_path, monkeypatch))
    assert "no single value: it differs with dose" in text and "the tables give every setting" in text


def _in_run_quest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_total: bool = True) -> tuple[Path, dict]:
    """A study with a grid setting (dose) and a value compared inside each run (mode: a, b). ``rate`` is a number per mode
    (cells carry the mode), ``total`` is one number per run (its cells do not)."""
    def met(vs):
        return {"values": vs, "trials": list(range(len(vs))), "count": len(vs), "total": 0.0, "non_finite": 0}

    cells = []
    for d in (1, 2):
        for m in ("a", "b"):
            cells.append({"cell": {"dose": d, "mode": m}, "level": {"mode": m}, "key": f"dose={d},mode={m}", "planned": 2, "ok": 2,
                          "failed": 0, "metrics": {"rate": met([0.1 * d, 0.1 * d + 0.01])}})
        if with_total:
            cells.append({"cell": {"dose": d}, "key": f"dose={d}", "planned": 2, "ok": 2, "failed": 0,
                          "metrics": {"total": met([5.0 * d, 5.0 * d + 1.0])}})
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    req = {"quantities": [{"name": "rate per mode", "main": True}, {"name": "total", "main": True}]}
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(req), encoding="utf-8")
    protocol = {"metrics": [{"id": "rate", "kind": "mean"}, {"id": "total", "kind": "mean"}],
                "in_each_run": {"mode": ["a", "b"]},
                "topic_map": [{"item": "Q1", "by": "metric:rate"}, {"item": "Q2", "by": "metric:total"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol)
    return tmp_path, protocol


def test_a_quantity_with_one_number_per_run_is_answered_though_the_study_compares_values_inside_each_run(tmp_path, monkeypatch) -> None:
    """Before: the line looked the quantity up by every varied axis, the in-run one too, found no trial at ``mode = None`` and
    said "not answered" for a value FI holds. Now the in-run axis is read from the per-value records where the quantity has
    them (``=each``) and pooled whole where it is one number per run (``=all``)."""
    root, protocol = _in_run_quest(tmp_path, monkeypatch)
    text = pf.answers_text(root)
    assert "not answered" not in text and "None" not in text
    assert "{{v:total @ dose=each, mode=all | range}}" in text and "{{v:rate @ dose=each, mode=each | range}}" in text
    rec = pv.records_of(root, protocol)
    assert pv.resolve("{{v:total @ dose=each, mode=all | range}}", rec).filled
    assert pv.resolve("{{v:rate @ dose=each, mode=each | range}}", rec).filled


def test_a_quantity_that_is_in_no_record_is_still_not_answered_in_an_in_run_study(tmp_path, monkeypatch) -> None:
    """The known positive: with the per-run quantity's records gone, its line still says not answered."""
    root, _ = _in_run_quest(tmp_path, monkeypatch, with_total=False)
    text = pf.answers_text(root)
    assert "- total (the plan's map" not in text and "total: not answered" in text


# --- a bias or a spread asked of the estimate of a known true value --------------------------------------------------

ESTIMATE_PROTOCOL = {
    "constants": {"x_true": 10.0},
    "metrics": [
        {"id": "x_hat", "label": "Estimated x", "kind": "mean",
         "known_value": {"true_value": "x_true", "estimate": "x_hat", "standard_error": "x_se", "level": 0.95}},
        {"id": "y_hat", "label": "Estimated y", "kind": "mean",
         "known_value": {"true_value": "y_true", "estimate": "y_hat", "standard_error": "y_se", "level": 0.95}},
    ],
    "topic_map": [
        {"item": "Q1", "by": "metric:x_hat", "statistic": "bias"}, {"item": "Q2", "by": "metric:x_hat", "statistic": "sd"},
        {"item": "Q3", "by": "metric:x_hat"}, {"item": "Q4", "by": "metric:y_hat", "statistic": "bias"},
    ],
}
ESTIMATE_REQ = {"quantities": [
    {"name": "bias", "stated": "the bias", "main": True},
    {"name": "spread of the estimates", "stated": "the spread", "main": True},
    {"name": "x", "stated": "recover x", "main": True},
    {"name": "bias of y", "stated": "the bias of y", "main": True},
]}


def _estimate_quest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    def met(vs):
        return {"values": vs, "trials": list(range(len(vs))), "count": len(vs), "total": 0.0, "non_finite": 0}

    data = {1: [10.5, 11.5], 2: [9.0, 9.4]}  # settings with a mean of 11.0 and 9.2 around the true value 10
    cells = [{"cell": {"dose": d}, "key": f"dose={d}", "planned": 2, "ok": 2, "failed": 0,
              "metrics": {"x_hat": met(v), "y_hat": met(v)}} for d, v in data.items()]
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(ESTIMATE_REQ), encoding="utf-8")
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: ESTIMATE_PROTOCOL)
    return tmp_path


def test_a_bias_and_a_spread_of_an_estimate_are_worked_out_and_not_the_estimates_own_mean(tmp_path, monkeypatch) -> None:
    """Before: the questions 'bias' and 'spread' mapped to the estimate printed its mean under both words. Now the bias is the
    mean minus the true value and the spread the sd of the trials; the question that asks for the value keeps its line."""
    root = _estimate_quest(tmp_path, monkeypatch)
    text = pf.answers_text(root)
    assert "{{v:x_hat @ dose=each | diff ref | range}}" in text and "{{v:x_hat:sd @ dose=each | range}}" in text
    assert "- x (the plan's map: Estimated x): {{v:x_hat @ dose=each | range}} (no single value: it differs with dose; this is the mean in each setting, lowest to highest" in text
    rec = pv.records_of(root, ESTIMATE_PROTOCOL)
    filled = pv.fill_values(text, rec)[0]
    assert "-0.8000 to 1.000" in filled  # 9.2 - 10 and 11.0 - 10
    assert "0.2828 to 0.7071" in filled  # the sd of [9.0, 9.4] and of [10.5, 11.5]


def test_a_bias_without_one_true_value_is_not_answered(tmp_path, monkeypatch) -> None:
    root = _estimate_quest(tmp_path, monkeypatch)
    text = pf.answers_text(root)
    assert '- bias of y: not answered (the plan gives no one true value of "Estimated y" to measure the bias from)' in text
    assert "{{v:y_hat @ dose=each | diff ref" not in text


def test_the_true_value_of_a_result_without_a_reference_is_its_reference_value() -> None:
    assert pv.known_truth(ESTIMATE_PROTOCOL, "x_hat") == 10.0
    assert pv.known_truth(ESTIMATE_PROTOCOL, "y_hat") is None  # y_true is no fixed number of the plan
    assert pv.known_truth(ESTIMATE_PROTOCOL, "nothing") is None


def test_an_old_plan_with_no_statistic_prints_the_result_as_it_is(tmp_path, monkeypatch) -> None:
    """Old plans type no statistic: the question's words are never read, so 'bias' prints the result's value as before."""
    root = _estimate_quest(tmp_path, monkeypatch)
    old = {**ESTIMATE_PROTOCOL, "topic_map": [{k: v for k, v in e.items() if k != "statistic"} for e in ESTIMATE_PROTOCOL["topic_map"]]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: old)
    text = pf.answers_text(root)
    assert "- bias (the plan's map: Estimated x): {{v:x_hat @ dose=each | range}}" in text
    assert "diff ref" not in text and ":sd" not in text


def test_an_answer_line_names_the_setting_it_differs_with_by_the_plans_label(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch)
    labelled = {**PROTOCOL, "labels": {"dose": "Dose given (mg)"}}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: labelled)
    text = pf.answers_text(root)
    assert "it differs with Dose given (mg); this is the mean" in text
    assert "{{v:decay_rate @ dose=each | range}}" in text  # the key keeps the id
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: PROTOCOL)
    assert "it differs with dose; this is the mean" in pf.answers_text(root)

