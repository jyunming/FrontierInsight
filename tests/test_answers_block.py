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
    assert "- decay rate (measured as: Decay rate): {{v:decay_rate @ dose=each | range}} (no single value: it differs with dose; this is the mean in each setting, lowest to highest" in text
    assert "- settling limit (measured as: Limit of the settling time (hours)): {{v:settling_limit}} (one value worked out by the analysis; no interval" in text
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
    assert "- colour of the sample (measured as: Decay rate): {{v:decay_rate" in text
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
    assert "- x (measured as: Estimated x): {{v:x_hat @ dose=each | range}} (no single value: it differs with dose; this is the mean in each setting, lowest to highest" in text
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
    assert "- bias (measured as: Estimated x): {{v:x_hat @ dose=each | range}}" in text
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


def _compared_quest(tmp_path, monkeypatch, flat_values=True):
    """Two results over a step size ``dt`` and an angle ``ang``; the plan compares step sizes. ``flat`` has the same trial values at
    every step size; ``step`` does not."""
    def met(vs):
        return {"values": vs, "trials": list(range(len(vs))), "count": len(vs), "total": 0.0, "non_finite": 0}

    cells = []
    for dt in (1, 2, 3):
        for ang in (10, 20):
            flat = [0.1 * ang, 0.1 * ang + 0.01] if flat_values else [0.1 * ang * dt, 0.1 * ang * dt + 0.01]
            cells.append({"cell": {"dt": dt, "ang": ang}, "key": f"dt={dt},ang={ang}", "planned": 2, "ok": 2, "failed": 0,
                          "metrics": {"flat": met(flat), "step": met([0.5 * dt * ang, 0.5 * dt * ang + 0.1])}})
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    req = {"quantities": [{"name": "flat error", "main": True}, {"name": "step error", "main": True}]}
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(req), encoding="utf-8")
    protocol = {"metrics": [{"id": "flat", "label": "Flat error", "kind": "mean"}, {"id": "step", "label": "Step error", "kind": "mean"}],
                "comparisons": [{"id": "c1", "result": "flat", "across": "dt", "how": "ratio"},
                                {"id": "c2", "result": "step", "across": "dt", "how": "ratio"}],
                "topic_map": [{"item": "Q1", "by": "metric:flat"}, {"item": "Q2", "by": "metric:step"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol)
    return tmp_path


def test_a_result_the_compared_setting_does_not_enter_gets_one_line_not_one_for_each_value(tmp_path, monkeypatch) -> None:
    """Before: the plan compared step sizes, so the answers block wrote one line for each of the 3 step sizes for a result whose
    values are the same at all of them. Now FI sees that in its record and writes one line; the result that does change with the
    step size keeps one line for each value."""
    root = _compared_quest(tmp_path, monkeypatch)
    text = pf.answers_text(root)
    flat = [ln for ln in text.splitlines() if ln.startswith("- flat error")]
    step = [ln for ln in text.splitlines() if ln.startswith("- step error")]
    assert len(flat) == 1 and "(the same at every value of dt; read at dt = 1)" in flat[0] and "{{v:flat @ ang=each, dt=1 | range}}" in flat[0]
    assert "it differs with ang;" in flat[0]
    assert len(step) == 1 and all(f"{d}: {{{{v:step @ ang=each, dt={d} | range}}}}" in step[0] for d in (1, 2, 3))
    rec = pv.records_of(root, pf.plan_protocol(root))
    assert pv.resolve("{{v:flat @ ang=each, dt=1 | range}}", rec).filled


def test_a_result_that_changes_with_a_compared_setting_is_one_bullet_keeping_each_value_answer(tmp_path, monkeypatch) -> None:
    """Before: the compared step size (9 values) made one near-identical line for each value (36 lines for 5 questions). Now one
    bullet for the question that holds each value's own answer, paired with that value, nothing dropped."""
    root = _compared_quest(tmp_path, monkeypatch, flat_values=False)
    flat = [ln for ln in pf.answers_text(root).splitlines() if ln.startswith("- flat error")]
    assert len(flat) == 1 and "by dt: 1: {{v:flat @ ang=each, dt=1 | range}}; 2: {{v:flat @ ang=each, dt=2 | range}}; 3: " in flat[0]
    assert flat[0].count("it differs with ang") == 1  # a note every answer shares is said once


def test_one_line_for_each_question_when_a_compared_setting_has_many_number_values(tmp_path, monkeypatch) -> None:
    (tmp_path / "raw").mkdir(parents=True)
    def met(vs):
        return {"values": vs, "trials": list(range(len(vs))), "count": len(vs), "total": 0.0, "non_finite": 0}
    cells = [{"cell": {"step": s, "ang": a}, "key": f"step={s},ang={a}", "planned": 2, "ok": 2, "failed": 0,
              "metrics": {m: met([s * a + i, s * a + i + 0.1]) for i, m in enumerate(("m1", "m2", "m3"))}}
             for s in range(1, 10) for a in (10, 20)]
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    req = {"quantities": [{"name": f"quantity {i}", "main": True} for i in (1, 2, 3)]}
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(req), encoding="utf-8")
    protocol = {"metrics": [{"id": m, "label": f"Result {m}", "kind": "mean"} for m in ("m1", "m2", "m3")],
                "comparisons": [{"id": f"c{m}", "result": m, "across": "step", "how": "difference"} for m in ("m1", "m2", "m3")],
                "topic_map": [{"item": f"Q{i}", "by": f"metric:m{i}"} for i in (1, 2, 3)]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol)
    lines = [ln for ln in pf.answers_text(tmp_path).splitlines() if ln.startswith("- ")]
    assert len(lines) == 3 and all("the plan's map" not in ln and ln.count("{{v:") == 9 for ln in lines)


def test_alternatives_named_by_words_are_one_bullet_with_each_alternative(tmp_path, monkeypatch) -> None:
    (tmp_path / "raw").mkdir(parents=True)
    def met(vs):
        return {"values": vs, "trials": list(range(len(vs))), "count": len(vs), "total": 0.0, "non_finite": 0}
    cells = [{"cell": {"method": m, "ang": a}, "key": f"method={m},ang={a}", "planned": 2, "ok": 2, "failed": 0,
              "metrics": {"cov": met([0.1 * a * len(m), 0.1 * a * len(m) + 0.01])}}
             for m in ("plain", "careful", "robust") for a in (10, 20)]
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps({"quantities": [{"name": "coverage", "main": True}]}), encoding="utf-8")
    protocol = {"metrics": [{"id": "cov", "label": "Coverage", "kind": "mean"}],
                "comparisons": [{"id": "c", "result": "cov", "across": "method", "how": "difference"}],
                "topic_map": [{"item": "Q1", "by": "metric:cov"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol)
    lines = [ln for ln in pf.answers_text(tmp_path).splitlines() if ln.startswith("- coverage")]
    assert len(lines) == 1 and all(f"{m}: {{{{v:cov @ ang=each, method={m} | range}}}}" in lines[0] for m in ("plain", "careful", "robust"))


def test_numbers_that_code_distinct_methods_keep_each_methods_value(tmp_path, monkeypatch) -> None:
    """Review case 1: method = 1, 2, 3 are three methods, not a sweep: each keeps its own placeholder."""
    (tmp_path / "raw").mkdir(parents=True)
    def met(vs):
        return {"values": vs, "trials": list(range(len(vs))), "count": len(vs), "total": 0.0, "non_finite": 0}
    cells = [{"cell": {"method": m}, "key": f"method={m}", "planned": 2, "ok": 2, "failed": 0,
              "metrics": {"cov": met([0.1 * m, 0.1 * m + 0.01])}} for m in (1, 2, 3)]
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps({"quantities": [{"name": "coverage", "main": True}]}), encoding="utf-8")
    protocol = {"metrics": [{"id": "cov", "label": "Coverage", "kind": "mean"}],
                "comparisons": [{"id": "c", "result": "cov", "across": "method", "how": "difference"}],
                "topic_map": [{"item": "Q1", "by": "metric:cov"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol)
    lines = [ln for ln in pf.answers_text(tmp_path).splitlines() if ln.startswith("- coverage")]
    assert len(lines) == 1 and all(f"{m}: {{{{v:cov:mean_ci @ method={m}}}}}" in lines[0] for m in (1, 2, 3))
    assert "method=each" not in lines[0]


def test_a_per_level_answer_keeps_which_value_goes_with_which_level(tmp_path, monkeypatch) -> None:
    """Review case 2: a quantity read at each temperature (a crossing per level) says each level with its own value."""
    root = _compared_quest(tmp_path, monkeypatch, flat_values=False)
    line = next(ln for ln in pf.answers_text(root).splitlines() if ln.startswith("- step error"))
    rec = pv.records_of(root, pf.plan_protocol(root))
    pairs = {d: pv.resolve(f"{{{{v:step @ ang=each, dt={d} | range}}}}", rec) for d in (1, 2, 3)}
    assert all(p.filled for p in pairs.values()) and len({p.text for p in pairs.values()}) == 3
    assert all(f"{d}: {{{{v:step @ ang=each, dt={d} | range}}}}" in line for d in (1, 2, 3))


def test_a_difference_comparison_keeps_its_difference_form_for_each_value(tmp_path, monkeypatch) -> None:
    """Review case 3: a bias (the difference from the plan's true value) is stated at each value in its difference form."""
    cells = [{"cell": {"dt": dt, "ang": 1}, "key": f"dt={dt},ang=1", "planned": 2, "ok": 2, "failed": 0,
              "metrics": {"est": {"values": [dt * 1.0, dt * 1.0 + 0.2], "trials": [0, 1], "count": 2, "total": 0.0, "non_finite": 0}}} for dt in (1, 2, 3)]
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps({"quantities": [{"name": "bias of est", "main": True}]}), encoding="utf-8")
    protocol = {"constants": {"truth": 1.0},
                "metrics": [{"id": "est", "kind": "mean", "known_value": {"true_value": "truth", "estimate": "est", "standard_error": "se", "level": 0.95}}],
                "grid": {"dt": [1, 2, 3], "ang": [1]},
                "comparisons": [{"id": "c1", "result": "est", "across": "dt", "how": "difference"}],
                "topic_map": [{"item": "Q1", "by": "metric:est", "statistic": "bias"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol)
    lines = [ln for ln in pf.answers_text(tmp_path).splitlines() if ln.startswith("- bias of est")]
    assert len(lines) == 1 and all(f"{d}: {{{{v:est @ dt={d} | diff ref}}}}" in lines[0] for d in (1, 2, 3))


def test_same_across_needs_two_values_and_every_group_equal(tmp_path, monkeypatch) -> None:
    root = _compared_quest(tmp_path, monkeypatch)
    rec = pv.records_of(root, pf.plan_protocol(root))
    assert rec.same_across("flat", "dt") and not rec.same_across("step", "dt")
    assert not rec.same_across("flat", "ang")
    assert not rec.same_across("flat", "missing_axis")


def test_a_bias_line_is_not_collapsed_when_the_reference_changes_with_the_compared_setting(tmp_path, monkeypatch) -> None:
    """Same estimates at every step size, but the true value the bias is measured from depends on the step size: one line for
    each value, as before."""
    def met(vs):
        return {"values": vs, "trials": list(range(len(vs))), "count": len(vs), "total": 0.0, "non_finite": 0}

    cells = [{"cell": {"dt": dt, "ang": 1}, "key": f"dt={dt},ang=1", "planned": 2, "ok": 2, "failed": 0,
              "metrics": {"est": met([1.0, 1.2])}} for dt in (1, 2, 3)]
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    req = {"quantities": [{"name": "bias of est", "main": True}]}
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(req), encoding="utf-8")

    def protocol(truth):
        return {"metrics": [{"id": "est", "kind": "mean", "reference": {"formula": truth, "kind": "value"}}],
                "grid": {"dt": [1, 2, 3], "ang": [1]},
                "comparisons": [{"id": "c1", "result": "est", "across": "dt", "how": "difference"}],
                "topic_map": [{"item": "Q1", "by": "metric:est", "statistic": "bias"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol("dt"))
    lines = [ln for ln in pf.answers_text(tmp_path).splitlines() if ln.startswith("- bias of est")]
    assert len(lines) == 1 and all(f"{d}: not answered" in lines[0] or f"{d}: {{{{v:est" in lines[0] for d in (1, 2, 3))
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol("1.1"))
    assert len([ln for ln in pf.answers_text(tmp_path).splitlines() if ln.startswith("- bias of est")]) == 1


def test_same_across_counts_failed_and_non_finite_runs_and_needs_every_value() -> None:
    def cell(dt, ang, vals, failed=0, non_finite=0):
        return {"cell": {"dt": dt, "ang": ang}, "key": f"dt={dt},ang={ang}", "planned": 2, "ok": 2 - failed, "failed": failed,
                "metrics": {"m": {"values": vals, "trials": list(range(len(vals))), "count": len(vals), "total": 0.0,
                                  "non_finite": non_finite}}}
    same = pv.from_cells([cell(1, 1, [1.0, 2.0]), cell(2, 1, [1.0, 2.0])])
    assert same.same_across("m", "dt")
    nan = pv.from_cells([cell(1, 1, [1.0, 2.0]), cell(2, 1, [1.0, 2.0], non_finite=1)])
    assert not nan.same_across("m", "dt")
    failed = pv.from_cells([cell(1, 1, [1.0, 2.0]), cell(2, 1, [1.0, 2.0], failed=1)])
    assert not failed.same_across("m", "dt")
    # one setting is carried at both step sizes, another at one only: not the same everywhere
    partial = pv.from_cells([cell(1, 1, [1.0]), cell(2, 1, [1.0]), cell(2, 2, [5.0])])
    assert not partial.same_across("m", "dt")
    # the first value is the first one that carries the metric
    later = pv.from_cells([cell(2, 1, [1.0]), cell(3, 1, [1.0])])
    assert later.values_of("m", "dt") == [2, 3]


def _parts_quest(tmp_path, monkeypatch):
    """The plan runs two sub-studies: ``ang`` varies at the last step size only; ``dt`` varies at ``ang`` = 10 only. The plan
    compares step sizes."""
    def met(vs):
        return {"values": vs, "trials": list(range(len(vs))), "count": len(vs), "total": 0.0, "non_finite": 0}

    settings = [(3, ang) for ang in (10, 20, 30)] + [(dt, 10) for dt in (1, 2)]
    cells = [{"cell": {"dt": dt, "ang": ang}, "key": f"dt={dt},ang={ang}", "planned": 2, "ok": 2, "failed": 0,
              "metrics": {"step": met([0.5 * dt * ang, 0.5 * dt * ang + 0.1])}} for dt, ang in settings]
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    req = {"quantities": [{"name": "step error", "main": True}]}
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(req), encoding="utf-8")
    protocol = {"metrics": [{"id": "step", "label": "Step error", "kind": "mean"}],
                "comparisons": [{"id": "c1", "result": "step", "across": "dt", "how": "ratio"}],
                "topic_map": [{"item": "Q1", "by": "metric:step"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol)
    return tmp_path


def test_a_compared_value_with_no_trial_at_the_questions_other_settings_is_not_a_line_of_its_own(tmp_path, monkeypatch) -> None:
    """Before: one line for each of the 3 step sizes, two of them "not answered ... no trials at that combination", because the
    part that varies ``ang`` ran at the last step size only. Now one line, naming the value it was measured at."""
    root = _parts_quest(tmp_path, monkeypatch)
    text = pf.answers_text(root)
    lines = [ln for ln in text.splitlines() if ln.startswith("- step error")]
    assert len(lines) == 1 and "not answered" not in text
    assert "(measured at dt = 3 only)" in lines[0] and "{{v:step @ ang=each, dt=3 | range}}" in lines[0]
    rec = pv.records_of(root, pf.plan_protocol(root))
    assert pv.resolve("{{v:step @ ang=each, dt=3 | range}}", rec).filled


def _derived(root: Path, item: dict) -> None:
    (root / "data" / "results").mkdir(parents=True)
    (root / "data" / "results" / "derived.json").write_text(
        json.dumps({"schema": "fi.derived/v1", "derived": {"settling_limit": item}}), encoding="utf-8")


def test_a_derived_result_that_is_ok_is_answered_with_its_value(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch, settling=None)
    _derived(root, {"status": "ok", "why": "", "value": 12.3456, "role": "result", "by": "derive_settling_limit", "inputs": []})
    from core import analysis_frame as frame
    pv.write_answers(root, frame.study_values(frame.read_record(root)), PROTOCOL, ids=["settling_limit"])
    text = pf.answers_text(root)
    assert "- settling limit (measured as: Limit of the settling time (hours)): {{v:settling_limit}}" in text
    assert "holds no value" not in text


def test_a_result_a_check_took_out_is_said_as_left_out_with_the_checks_reason(tmp_path, monkeypatch) -> None:
    """Before: a quantity FI holds a value for but a check withheld read 'FI holds no value', as if nothing was computed."""
    root = _quest(tmp_path, monkeypatch)
    (root / "needs").mkdir()
    (root / "needs" / "RESULT_CHECK.json").write_text(json.dumps(
        {"removed_quantities": ["settling_limit"], "note": "the code computes it differently from the plan"}), encoding="utf-8")
    text = pf.answers_text(root)
    assert ('- settling limit: not answered (FI left "Limit of the settling time (hours)" out of the results: '
            "the code computes it differently from the plan)") in text
    assert "holds no value" not in text and "{{v:settling_limit" not in text


def test_a_derived_result_that_failed_is_not_answered_and_says_why(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch, settling=None)
    _derived(root, {"status": "failed", "why": "no trial reached the limit", "value": None, "role": "result"})
    text = pf.answers_text(root)
    assert '- settling limit: not answered (FI could not work out "Limit of the settling time (hours)": no trial reached the limit)' in text
    assert "{{v:settling_limit" not in text
