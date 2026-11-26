"""The outputs the plan states, checked from FI's own records (core/small_run.py ``planned_shapes`` / ``unheld_shapes``,
core/figure_text_check.py ``undrawn_declared_references``, core/quality_report.py ``_planned_output_problems`` and the
figure rows): a result the plan states is a curve or a map in each run, which FI's record of a trial cannot hold (numbers
only), is said and never sent back; after the run, a reference line the plan names for a figure that its record does not
show drawn, and a result worked out although a quantity the plan works it out from has no value, are one QUALITY.md line
each. Nothing here stops a run or changes a number. A neutral made-up study: a heated plate, its temperature over time."""

from __future__ import annotations

import asyncio
import json
import logging
import types
from pathlib import Path
from typing import Any

from core import figure_text_check as ftc
from core import frozen_protocol
from core import quality_report as qr
from core import small_run

PROTOCOL: dict[str, Any] = {
    "grid": {"power_w": [1.0, 2.0]},
    "metrics": [
        {"id": "peak_temp", "role": "result", "returned": "each_run", "depends_on": ["power_w"]},
        {"id": "temp_trace", "role": "result", "returned": "each_run", "depends_on": ["power_w"], "shape": "curve",
         "along": ["time"]},
        {"id": "surface_map", "role": "result", "returned": "each_run", "depends_on": ["power_w"], "shape": "map",
         "along": ["x", "y"]},
    ],
}


def _record(values: dict[str, list[Any]], *, whole: bool = True) -> dict[str, Any]:
    return {"crashed": False, "returned": sorted(values),
            "settings": [{"key": "power_w=1.0", "planned": 1, "ok": 1 if whole else 0, "failed": 0, "timed_out": False,
                          "values": values}]}


# --- before the run: the small run -------------------------------------------------------------------------------------


def test_the_shapes_the_plan_states_are_read_and_a_number_is_none() -> None:
    assert small_run.planned_shapes(PROTOCOL) == {"temp_trace": "a curve along `time`",
                                                  "surface_map": "a map over `x` and `y`"}
    assert small_run.planned_shapes({"metrics": [{"id": "a", "shape": "curve", "along": []}]}) == {}


def test_a_curve_returned_as_one_number_is_said_and_never_sent_back() -> None:
    faults = small_run.unheld_shapes(PROTOCOL, _record({"peak_temp": [3.0], "temp_trace": [2.5], "surface_map": [1.0]}))
    assert [(f.quantity, f.send, f.fact) for f in faults] == [("temp_trace", False, small_run.UNHELD),
                                                              ("surface_map", False, small_run.UNHELD)]
    assert "a curve along `time`" in faults[0].why and "not sent back" in faults[0].why
    assert small_run.unheld_shapes(PROTOCOL, _record({"peak_temp": [3.0]})) == [], "not returned: missing_outputs says it"
    assert small_run.unheld_shapes({"metrics": PROTOCOL["metrics"][:1]}, _record({"peak_temp": [3.0]})) == []


def test_a_curve_no_trial_returned_is_said_not_sent_back_and_a_number_still_is() -> None:
    """A repair asked to return a curve would make every trial fail (FI's record of a trial holds numbers only): the
    absent curve is a line. A plain number no trial returned is still sent back."""
    got = small_run.missing_outputs(PROTOCOL, _record({"surface_map": [1.0]}))
    assert [(f.quantity, f.send) for f in got] == [("peak_temp", True), ("temp_trace", False)]
    assert "FI keeps one number per run" in got[1].why


def _stand(tmp_path: Path) -> Any:
    from core.run_checks import RunChecksMixin

    class Stand(RunChecksMixin):
        def __init__(self) -> None:
            self.quest_root = tmp_path
            self.fi_dir = tmp_path / ".fi"
            self._log = logging.getLogger("test")
            self.config = types.SimpleNamespace(engine=types.SimpleNamespace(exec_reflect_max_iterations=3),
                                                execution=types.SimpleNamespace(sandbox="venv", timeout_s=60))
            self._small_run_note = ""

        def _protocol_block(self, state: Any) -> Any:
            return PROTOCOL

        def _code_review_failed_run(self, message: str, *, give_up: str = "", script: str | None = None,
                                    check: str = "") -> Any:
            return {"exec_result": {"returncode": 1, "stderr_tail": message, "failed_script": script,
                                    **({"check": check} if check else {})}}

    (tmp_path / ".fi").mkdir(exist_ok=True)
    (tmp_path / "plan.md").write_text("# Plan\n", encoding="utf-8")
    return Stand()


def test_the_small_run_says_a_curve_returned_as_one_number_and_the_quest_goes_on(tmp_path: Path) -> None:
    eng = _stand(tmp_path)
    rec = {"key": "k", **_record({"peak_temp": [3.0], "temp_trace": [2.5], "surface_map": [1.0]})}
    (tmp_path / small_run.RECORD).write_text(json.dumps(rec), encoding="utf-8")
    assert asyncio.run(eng._small_run_checks({}, rec)) is None, "never a send-back"
    plan = (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "`temp_trace` is a curve along `time` in each run" in plan and "`surface_map` is a map over `x` and `y`" in plan
    assert "temp_trace" in eng._small_run_note
    assert small_run.sent(small_run.read_record(tmp_path))["count"] == 0


# --- after the run: a figure's reference lines --------------------------------------------------------------------------


def _calls(*series: dict[str, Any], refs: tuple[dict[str, Any], ...] = (), raw: bool = False) -> dict[str, Any]:
    return {"version": 2, "panels": [{"panel": 1, "raw": raw, "series": list(series), "refs": list(refs)}]}


TRACE = {"id": "peak_temp@method:fine", "what": "peak_temp", "kind": "line"}


def test_a_reference_line_the_plan_names_and_the_record_does_not_draw_is_listed() -> None:
    assert ftc.undrawn_declared_references(_calls(TRACE), ["peak_temp_theory"], {}, ["peak_temp"]) == ["peak_temp_theory"]
    drawn = _calls(TRACE, {"id": "peak_temp_theory@method:fine", "what": "peak_temp_theory", "kind": "line"})
    assert ftc.undrawn_declared_references(drawn, ["peak_temp_theory"], {}, ["peak_temp"]) == []
    as_line = _calls(TRACE, refs=({"id": "peak_temp_theory", "source": "plan", "axis": "y"},))
    assert ftc.undrawn_declared_references(as_line, ["peak_temp_theory"], {}, ["peak_temp"]) == []
    assert ftc.undrawn_declared_references(_calls(TRACE, raw=True), ["peak_temp_theory"], {}, ["peak_temp"]) == [], \
        "a custom drawing says nothing FI can read"
    assert ftc.undrawn_declared_references({}, ["peak_temp_theory"], {}, ["peak_temp"]) == []


def test_a_shown_results_own_series_is_not_its_reference_line() -> None:
    """A result the figure shows whose plan gives a known value: its own series is not the line of that value; only a line
    FI's record binds to it (drawn from the plan under its id) is."""
    assert ftc.undrawn_declared_references(_calls(TRACE), ["peak_temp"], {}, ["peak_temp"]) == ["peak_temp"]
    bound = _calls(TRACE, refs=({"id": "peak_temp", "source": "plan", "axis": "y"},))
    assert ftc.undrawn_declared_references(bound, ["peak_temp"], {}, ["peak_temp"]) == []
    # a reference formula the plan gives the result is judged by missing_references, beside the result's series: one line
    protocol = {"metrics": [{"id": "peak_temp", "reference": {"formula": "2*power_w"}}]}
    assert ftc.missing_references(_calls(TRACE), protocol) != []
    assert ftc.undrawn_declared_references(_calls(TRACE), ["peak_temp"], protocol, ["peak_temp"]) == []


def _quest(tmp_path: Path, protocol: dict[str, Any], derived: dict[str, Any], figures: list[dict[str, Any]],
           not_computed: list[dict[str, Any]] | None = None) -> Path:
    root = tmp_path / "q"
    (root / "figures").mkdir(parents=True)
    frozen_protocol.freeze(root, protocol, approved_by="test", source="test")
    for f in figures:
        (root / "figures" / f["file"]).write_text("x", encoding="utf-8")
    path = root / "data" / "results" / "derived.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": "fi.derived/v1", "derived": derived, "figures": figures, "figures_said": "",
                                "not_computed": not_computed or []}), encoding="utf-8")
    return root


def _texts(root: Path) -> list[str]:
    return [p["text"] for p in qr._analysis_problems(qr._Quest(root))]


def test_quality_says_a_figure_draws_no_line_the_plan_names_for_it(tmp_path: Path) -> None:
    protocol = {"metrics": [{"id": "peak_temp", "role": "result", "returned": "each_run"},
                            {"id": "peak_temp_theory", "role": "reference", "returned": "afterwards"}],
                "figure_declarations": {"fig1.png": {"shows": ["peak_temp"], "x": "power_w", "references": ["peak_temp_theory"]},
                                        "fig2.png": {"shows": ["peak_temp"], "x": "power_w", "references": ["no_such_line"]}}}
    figures = [{"id": "fig1", "file": "fig1.png", "status": "ok"}, {"id": "fig2", "file": "fig2.png", "status": "ok"}]
    root = _quest(tmp_path, protocol, {"x": {"value": 1.0}}, figures)
    rec = root / ".fi" / "figure_records"
    rec.mkdir(parents=True)
    for stem in ("fig1", "fig2"):
        (rec / f"{stem}.calls.json").write_text(json.dumps(_calls(TRACE)), encoding="utf-8")
    texts = _texts(root)
    assert "Figure `fig1` draws no line of `peak_temp_theory`, a reference line the plan names for it." in texts
    assert not any("no_such_line" in t for t in texts), "a line the plan holds as no reference of the figure is left out"
    (rec / "fig1.calls.json").write_text(json.dumps(_calls(TRACE, refs=({"id": "peak_temp_theory", "source": "plan"},))),
                                         encoding="utf-8")
    assert not any("draws no line" in t for t in _texts(root))


# --- after the run: what the plan states of its results ----------------------------------------------------------------


def test_quality_says_a_result_worked_out_without_an_input_that_has_no_value(tmp_path: Path) -> None:
    protocol = {**PROTOCOL, "derived": [{"id": "heat_ratio", "inputs": ["peak_temp", "cool_time"], "role": "result"},
                                        {"id": "cool_time", "inputs": ["peak_temp"], "role": "result"},
                                        {"id": "lag", "inputs": ["cool_time"], "role": "result"}]}
    derived = {"heat_ratio": {"status": "ok", "by_setting": {"power_w=1.0": 0.5}},
               "cool_time": {"status": "ok", "value": None}, "lag": {"status": "ok", "value": None}}
    root = _quest(tmp_path, protocol, derived, [],
                  not_computed=[{"id": "cool_time", "status": "missing", "why": "never cools"},
                                {"id": "lag", "status": "missing", "why": "no cooling time"}])
    texts = _texts(root)
    assert ("`heat_ratio` has a value, though `cool_time`, which the plan works it out from, has none, so `heat_ratio` "
            "is worked out without it.") in texts
    assert not any(t.startswith("`lag` has a value") for t in texts), "a result with no value has its own line only"
    assert sum(t.startswith("cool_time was not computed") for t in texts) == 1
    shapes = [t for t in texts if t.startswith("The plan states")]
    assert shapes == ["The plan states `temp_trace` is a curve along `time` in each run; FI keeps one number per run for a "
                      "result, so the study's results hold no such curve.",
                      "The plan states `surface_map` is a map over `x` and `y` in each run; FI keeps one number per run for a "
                      "result, so the study's results hold no such map."]


def test_quality_says_nothing_of_a_plan_whose_results_are_numbers_with_values(tmp_path: Path) -> None:
    protocol = {"metrics": PROTOCOL["metrics"][:1], "derived": [{"id": "heat_ratio", "inputs": ["peak_temp"]}]}
    root = _quest(tmp_path, protocol, {"heat_ratio": {"status": "ok", "value": 0.5}}, [])
    assert _texts(root) == []


# --- what the outside review and the replay found ------------------------------------------------------------------------


def test_a_plan_with_no_derived_quantity_still_says_its_curve(tmp_path: Path) -> None:
    root = _quest(tmp_path, {"metrics": PROTOCOL["metrics"][1:2]}, {}, [])
    assert _texts(root) == ["The plan states `temp_trace` is a curve along `time` in each run; FI keeps one number per run "
                            "for a result, so the study's results hold no such curve."]


def test_a_figure_record_that_is_not_one_fi_wrote_names_nothing(tmp_path: Path) -> None:
    protocol = {"metrics": [{"id": "peak_temp", "role": "result", "returned": "each_run"},
                            {"id": "peak_temp_theory", "role": "reference", "returned": "afterwards"}],
                "figure_declarations": {"fig1.png": {"shows": ["peak_temp"], "x": "power_w", "references": ["peak_temp_theory"]}}}
    root = _quest(tmp_path, protocol, {}, [{"id": "fig1", "file": "fig1.png", "status": "ok"}])
    (root / ".fi" / "figure_records").mkdir(parents=True)
    (root / ".fi" / "figure_records" / "fig1.calls.json").write_text("[1]", encoding="utf-8")
    assert not any("draws no line" in t for t in _texts(root))
    assert ftc.undrawn_declared_references([1], ["peak_temp_theory"], {}, []) == []  # type: ignore[arg-type]
    (root / ".fi" / "figure_records" / "fig1.calls.json").write_text('{"panels": 1}', encoding="utf-8")
    assert not any("draws no line" in t for t in _texts(root)), "an object FI cannot read names nothing either"


def test_a_figure_not_written_yet_is_not_hidden_behind_an_earlier_figure_that_drew_nothing(tmp_path: Path) -> None:
    """FI's record names the first figure that failed; on three trials of a small run the first may have drawn nothing (too
    few trials to tell) while a later one raises NotImplementedError, a fault whatever the data: that one is sent back."""
    from types import SimpleNamespace

    from core import pipeline_gate as gate

    path = tmp_path / "data" / "results" / "derived.json"
    path.parent.mkdir(parents=True)
    figures = [{"id": "fig1", "status": "failed", "why": "`figure_fig1` drew nothing on panel 1."},
               {"id": "fig2", "status": "failed", "why": "`figure_fig2` stopped with NotImplementedError: the trace is not "
                                                         "among the results a trial may return (derive.py line 40)."}]
    path.write_text(json.dumps({"schema": "fi.derived/v1", "derived": {}, "figures": figures}), encoding="utf-8")
    run = SimpleNamespace(returncode=1, stdout='RESULT_JSON: {}\n', stderr="")
    got = gate._fis_analysis_result(tmp_path, run, {}, {}, [])
    assert got.kind == gate.FAULT and got.body == "figure_fig2" and got.exception == "NotImplementedError"
    assert got.detail.startswith("`figure_fig2` is not written yet")
    assert small_run.analysis_split(got).send
    # only figures that drew nothing: still "too few trials to tell", the first one named
    figures[1] = {"id": "fig2", "status": "failed", "why": "`figure_fig2` drew nothing on panel 1."}
    path.write_text(json.dumps({"schema": "fi.derived/v1", "derived": {}, "figures": figures}), encoding="utf-8")
    got = gate._fis_analysis_result(tmp_path, run, {}, {}, [])
    assert got.kind == gate.SMALL_DATA and got.body == "figure_fig1"
