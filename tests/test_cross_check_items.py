"""What the literature check looks up comes from FI's typed record (core/cross_check_items.py), never from the analysis's text.

The fixtures are a made-up study of a widget scale (neutral names); the trial record, the plan and the topic's
requirements are written the way FI writes them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import cross_check_items as cci


def _cell(setting: dict[str, Any], **metrics: list[Any]) -> dict[str, Any]:
    return {"cell": setting, "key": ",".join(f"{k}={v}" for k, v in setting.items()), "planned": 4, "ok": 4, "failed": 0,
            "metrics": {k: {"values": v, "trials": list(range(len(v))), "count": len(v), "total": len(v), "non_finite": 0}
                        for k, v in metrics.items()}}


def _write(root: Path, cells: list[dict[str, Any]]) -> None:
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "trials.json").write_text(json.dumps({"schema": "x", "thresholds": {}, "cells": cells}), encoding="utf-8")


def _protocol(*metrics: dict[str, Any], topic_map: list[dict[str, Any]] | None = None,
              derived: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"grid": {"load": [1, 2]}, "metrics": list(metrics), "derived": derived or [], "topic_map": topic_map or [],
            "labels": {}}


def _metric(ident: str, label: str = "", role: str = "result", estimand: str = "") -> dict[str, Any]:
    return {"id": ident, "label": label or ident, "unit": "a trial", "role": role, "estimand": estimand or ident, "kind": "mean"}


def _req(*quantities: dict[str, Any]) -> dict[str, Any]:
    return {"factors": [], "quantities": list(quantities), "formulas": [], "comparisons": [], "searches": []}


def _q(name: str, main: bool = True) -> dict[str, Any]:
    return {"name": name, "stated": name, "main": main}


def _flat_record(root: Path, **values: float) -> None:
    """One setting, one number per result: a record a pooled value stands for."""
    _write(root, [_cell({"load": 1}, **{k: [v, v] for k, v in values.items()}),
                  _cell({"load": 2}, **{k: [v, v] for k, v in values.items()})])


def _items(root: Path, protocol: Any, req: Any, covered: Any = None, **kw: Any) -> list[cci.Item]:
    return cci.items(quest_root=root, protocol=protocol, requirements=req, covered=covered or {}, design={},
                     context="scale calibration", **kw)


# --- a confirmed map, an unconfirmed one --------------------------------------------------------------------------


def test_a_confirmed_quantity_is_looked_up_by_its_name_and_the_sentence_holds_the_recorded_value(tmp_path: Path) -> None:
    _flat_record(tmp_path, weight_err=2.5)
    protocol = _protocol(_metric("weight_err", "Weight error (g)", estimand="weight error of the scale"),
                         topic_map=[{"item": "Q1", "by": "metric:weight_err"}])

    (item,) = _items(tmp_path, protocol, _req(_q("weight error of the scale")))

    assert item.status == cci.TO_CHECK and item.result == "weight_err"
    assert item.quantity == "weight error of the scale"
    assert item.query == "weight error of the scale scale calibration"
    assert "Weight error (g)" in item.sentence and "2.5" in item.sentence


def test_a_value_changed_in_the_record_changes_the_sentence(tmp_path: Path) -> None:
    """Planted fault: the sentence is FI's reading of the record, so it moves with the record."""
    protocol = _protocol(_metric("weight_err", "Weight error (g)", estimand="weight error of the scale"),
                         topic_map=[{"item": "Q1", "by": "metric:weight_err"}])
    sentences = []
    for value in (2.5, 7.75):
        _flat_record(tmp_path, weight_err=value)
        sentences.append(_items(tmp_path, protocol, _req(_q("weight error of the scale")))[0].sentence)
    assert "2.5" in sentences[0] and "7.75" in sentences[1] and sentences[0] != sentences[1]


def test_a_map_the_reader_never_confirmed_is_checked_as_the_plans_result_only(tmp_path: Path) -> None:
    """Maintainer decision: the plan's own name is searched and stated; the person's quantity is never mentioned, and the
    entry names none (``quantity`` empty), so nothing counts it as a check of that quantity."""
    _flat_record(tmp_path, unrelated_metric=4.0)
    protocol = _protocol(_metric("unrelated_metric", "Pan drift (g)"),
                         topic_map=[{"item": "Q1", "by": "metric:unrelated_metric"}])
    req = _req(_q("how heavy a sample the scale can hold"))

    (item,) = _items(tmp_path, protocol, req, {"judged": {"h1": {"Q1": "missing"}}})

    assert item.status == cci.TO_CHECK and item.quantity is None
    assert "how heavy" not in item.query and "how heavy" not in item.sentence
    assert item.query.startswith("Pan drift (g)") and "Pan drift (g)" in item.sentence


def test_a_map_the_reader_settled_names_the_quantity(tmp_path: Path) -> None:
    _flat_record(tmp_path, unrelated_metric=4.0)
    protocol = _protocol(_metric("unrelated_metric", "Pan drift (g)"),
                         topic_map=[{"item": "Q1", "by": "metric:unrelated_metric"}])

    (item,) = _items(tmp_path, protocol, _req(_q("how heavy a sample the scale can hold")),
                     {"judged": {"h1": {"Q1": "covered"}}})

    assert item.quantity == "how heavy a sample the scale can hold" and item.query.startswith("how heavy a sample")


def test_quantities_on_one_result_give_one_item_naming_the_confirmed_ones(tmp_path: Path) -> None:
    _flat_record(tmp_path, weight_err=2.5)
    protocol = _protocol(_metric("weight_err", "Weight error (g)", estimand="weight error of the scale"),
                         topic_map=[{"item": "Q1", "by": "metric:weight_err"}, {"item": "Q2", "by": "metric:weight_err"}])

    items = _items(tmp_path, protocol, _req(_q("weight error of the scale"), _q("weight error")))

    assert len(items) == 1 and items[0].quantity == "weight error of the scale; weight error"
    assert items[0].query.startswith("weight error of the scale ")  # the first name only: no list in a query


def test_a_quantity_with_no_stated_result_or_a_setting_is_not_held_with_fis_reason(tmp_path: Path) -> None:
    _flat_record(tmp_path, weight_err=2.5)
    protocol = _protocol(_metric("weight_err"), topic_map=[{"item": "Q2", "by": "setting:load"}])

    items = _items(tmp_path, protocol, _req(_q("weight error"), _q("the load"), _q("a side remark", main=False)))

    assert [i.status for i in items] == [cci.NOT_HELD, cci.NOT_HELD]
    assert items[0].why_not == "the plan maps no result to it" and items[0].result == "Q1"
    assert items[1].why_not == "the plan maps it to a setting, an equation or a search, not to a result"
    assert items[1].query == "" and items[1].sentence == ""


# --- whether a value is held --------------------------------------------------------------------------------------


def test_a_result_the_record_does_not_hold_is_not_held(tmp_path: Path) -> None:
    _flat_record(tmp_path, other=1.0)
    protocol = _protocol(_metric("weight_err"), topic_map=[{"item": "Q1", "by": "metric:weight_err"}])

    (item,) = _items(tmp_path, protocol, _req(_q("weight error")))

    assert item.status == cci.NOT_HELD and "no value" in item.why_not


def test_a_result_fi_left_out_is_not_held_with_the_checks_own_note(tmp_path: Path) -> None:
    _flat_record(tmp_path, weight_err=2.5)
    (tmp_path / "needs").mkdir()
    (tmp_path / "needs" / "RESULT_CHECK.json").write_text(
        json.dumps({"note": "the check could not reproduce it", "removed_quantities": ["weight_err"]}), encoding="utf-8")
    protocol = _protocol(_metric("weight_err"), topic_map=[{"item": "Q1", "by": "metric:weight_err"}])

    (item,) = _items(tmp_path, protocol, _req(_q("weight error")))

    assert item.status == cci.NOT_HELD and item.why_not == "the check could not reproduce it"


def test_a_degenerate_run_holds_nothing(tmp_path: Path) -> None:
    _flat_record(tmp_path, weight_err=0.0)
    protocol = _protocol(_metric("weight_err"), topic_map=[{"item": "Q1", "by": "metric:weight_err"}])

    (item,) = _items(tmp_path, protocol, _req(_q("weight error")), degenerate=True)

    assert item.status == cci.NOT_HELD and "broken run" in item.why_not


def test_a_value_only_the_plan_named_answers_file_holds_is_held_by_its_exact_id(tmp_path: Path) -> None:
    """No trial record: the plan-named answers FI kept at execute are the record, matched on the exact id."""
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "analysis_answers.json").write_text(
        json.dumps({"scheduled": 1, "runs": [{"run": "0", "values": {"slope": 3.5, "slope_x": 9.0}}]}), encoding="utf-8")
    protocol = _protocol(_metric("slope", "Fitted slope"), topic_map=[{"item": "Q1", "by": "metric:slope"}])

    (item,) = _items(tmp_path, protocol, _req(_q("fitted slope")))

    assert item.status == cci.TO_CHECK and "3.5" in item.sentence and "9.0" not in item.sentence


def test_a_result_the_runs_disagree_on_is_not_held(tmp_path: Path) -> None:
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "analysis_answers.json").write_text(
        json.dumps({"scheduled": 2, "runs": [{"run": "0", "values": {"slope": 3.5}}, {"run": "1", "values": {"slope": 3.6}}]}),
        encoding="utf-8")
    protocol = _protocol(_metric("slope"), topic_map=[{"item": "Q1", "by": "metric:slope"}])

    (item,) = _items(tmp_path, protocol, _req(_q("slope")))

    assert item.status == cci.NOT_HELD and "different values" in item.why_not


def test_a_result_that_changes_with_a_setting_is_stated_as_a_range_with_no_direction(tmp_path: Path) -> None:
    _write(tmp_path, [_cell({"load": 1}, weight_err=[1.0, 1.0]), _cell({"load": 2}, weight_err=[3.0, 3.0])])
    protocol = _protocol(_metric("weight_err", "Weight error (g)", estimand="weight error of the scale"),
                         topic_map=[{"item": "Q1", "by": "metric:weight_err"}])

    (item,) = _items(tmp_path, protocol, _req(_q("weight error")))

    assert item.status == cci.TO_CHECK
    assert "from 1.000 to 3.000" in item.sentence and "no single value" in item.sentence
    assert "computed no comparison" in item.sentence
    assert not any(w in item.sentence for w in ("higher", "lower", "increases", "decreases"))


# --- the older plan, no requirements ------------------------------------------------------------------------------


def test_without_topic_requirements_the_declared_results_of_role_result_are_looked_up(tmp_path: Path) -> None:
    _flat_record(tmp_path, a=1.0, b=2.0, c=3.0)
    protocol = _protocol(_metric("a", "Result a"), _metric("b", "Check b", role="diagnostic"),
                         _metric("c", "Reference c", role="reference"),
                         derived=[{"id": "d", "inputs": ["a"], "role": "result", "label": "Derived d"}])

    items = _items(tmp_path, protocol, None)

    assert [(i.result, i.status) for i in items] == [("a", cci.TO_CHECK), ("d", cci.NOT_HELD)]
    assert items[0].quantity is None and items[0].query.startswith("Result a")


def test_at_most_ten_results_are_looked_up(tmp_path: Path) -> None:
    ids = [f"m{i}" for i in range(14)]
    _flat_record(tmp_path, **{i: 1.0 for i in ids})

    assert len(_items(tmp_path, _protocol(*[_metric(i) for i in ids]), None)) == cci.CAP


def test_no_protocol_gives_no_item(tmp_path: Path) -> None:
    assert _items(tmp_path, None, _req(_q("weight error"))) == []


def test_a_label_s_markup_is_not_sent_to_a_search(tmp_path: Path) -> None:
    _flat_record(tmp_path, a=1.0)
    (item,) = _items(tmp_path, _protocol(_metric("a", "Measured $G_x$ (units)")), None)
    assert "$" not in item.query and "\\" not in item.query and "G_x" in item.query


# --- the no-simulation path ---------------------------------------------------------------------------------------


def _data(measurements: dict[str, Any], missing: list[str] | None = None, **top: Any) -> list[cci.Item]:
    return cci.items(quest_root=Path("."), protocol=None, requirements=None, covered={}, design={}, context="ctx",
                     no_simulation=True,
                     result_json={"n_files": 7, "measurements": measurements, "missing_measurements": missing or [], **top})


def test_no_simulation_items_are_the_measured_variables_only() -> None:
    items = _data({"mean_height": 1.75, "colour": "blue", "weight": {"value": 70.5, "unit": "kg", "n_observations": 12,
                                                                     "source_file_ids": [3]}},
                  missing=["age"])

    by = {i.result: i for i in items}
    assert set(by) == {"mean_height", "colour", "weight", "age"}  # n_files is never an item
    assert by["mean_height"].status == cci.TO_CHECK and "1.75" in by["mean_height"].sentence
    assert by["colour"].sentence == "colour: blue."
    assert by["weight"].sentence == "weight: 70.50 kg." and "12" not in by["weight"].sentence
    assert by["age"].status == cci.NOT_HELD and by["age"].why_not == "the data holds no value of it"


def test_a_no_simulation_object_without_a_value_is_not_held() -> None:
    (item,) = _data({"weight": {"unit": "kg", "n_observations": 12}})
    assert item.status == cci.NOT_HELD


def test_the_analysis_text_is_not_an_input() -> None:
    """The source fix: the function has no parameter the analysis's findings could come through."""
    import inspect

    names = set(inspect.signature(cci.items).parameters)
    assert not names & {"analysis", "findings", "key_findings", "state"}


def test_requirements_that_name_no_quantity_still_look_up_the_plans_results(tmp_path: Path) -> None:
    _flat_record(tmp_path, weight_err=0.5)
    protocol = _protocol(_metric("weight_err", "Weight error"))
    req = {"factors": [{"name": "load"}], "quantities": [], "formulas": [], "comparisons": [], "searches": []}

    items = _items(tmp_path, protocol, req)

    assert [(i.result, i.status, i.quantity) for i in items] == [("weight_err", cci.TO_CHECK, None)]
