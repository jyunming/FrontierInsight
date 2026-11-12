"""FI's small run compared with the frozen plan before the long run (core/small_run.py, core/in_run.py shape_problem,
RunChecksMixin._small_run_checks): a number the plan declares one per run that came back per value, a search's objective
or constraint its own judgement cannot read at the baseline design, a declared result no trial returned, and FI's analysis
on the small run's trials. An exact fault goes back to the code once per quest on a counter of its own; everything a small
sample can explain is a line; nothing here stops the quest."""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from core import in_run
from core import pipeline_gate as gate
from core import small_run

#: Neutral made-up study: a solver compared inside each run, a yield and a cost per run.
INSIDE = {"solver": ["fast", "exact"]}
PROTOCOL: dict[str, Any] = {
    "metrics": [
        {"id": "yield_rate", "depends_on": ["temperature"], "returned": "each_run", "role": "result"},
        {"id": "solve_error", "depends_on": ["temperature", "solver"], "returned": "each_run", "role": "result"},
        {"id": "wall_time", "depends_on": ["temperature"], "returned": "each_run", "role": "diagnostic"},
        {"id": "best_yield", "depends_on": ["temperature"], "returned": "afterwards", "role": "result"},
    ],
    "derived": [{"id": "yield_spread", "inputs": ["yield_rate", "temperature"], "role": "result"}],
    "optimisation": {
        "objective": {"quantity": "yield_rate", "direction": "maximise"},
        "constraints": [{"quantity": "pressure", "limit": "<= 5"}, {"quantity": "stress", "limit": "<= 2"}],
        "design_variables": [{"name": "temperature", "low": 1.0, "high": 9.0, "kind": "continuous"}],
        "baseline": {"values": {"temperature": 3.0}},
        "in_each_run": INSIDE,
    },
}
PER_VALUE = {"solver=fast": 1.0, "solver=exact": 2.0}


def _record(values: dict[str, list[Any]], *, baseline: dict[str, Any] | None = None, whole: bool = True) -> dict[str, Any]:
    return {"crashed": False, "returned": sorted(values),
            "settings": [{"key": "temperature=3", "planned": 1, "ok": 1 if whole else 0, "failed": 0, "timed_out": False,
                          "values": values}],
            **({"baseline": {"values": baseline, "reason": ""}} if baseline is not None else {})}


# --- the shape of a number per run --------------------------------------------------------------------------------------


def test_what_a_search_compares_inside_each_run_counts_for_the_numbers_per_run() -> None:
    """The plan of a search writes ``in_each_run`` in its block; reading only the top level put a number that varies over
    the solver among the numbers per run (old code: ``solve_error`` was listed)."""
    assert in_run.per_run_names(PROTOCOL) == {"yield_rate", "wall_time", "best_yield"}
    top = {"in_each_run": INSIDE, "metrics": PROTOCOL["metrics"]}
    assert in_run.per_run_names(top) == {"yield_rate", "wall_time", "best_yield"}


def test_a_number_per_run_returned_per_value_is_a_shape_problem() -> None:
    got = in_run.shape_problem(PROTOCOL, {"yield_rate": PER_VALUE, "solve_error": PER_VALUE, "wall_time": 0.1})
    assert list(got) == ["yield_rate"]
    assert "one number per value of `solver`" in got["yield_rate"] and "one number per run" in got["yield_rate"]
    assert in_run.shape_problem({"metrics": PROTOCOL["metrics"]}, {"yield_rate": PER_VALUE}) == {}, \
        "nothing is compared inside the runs: nothing to say"
    faults = small_run.shape_faults(PROTOCOL, _record({"yield_rate": [PER_VALUE], "solve_error": [PER_VALUE]}))
    assert [(f.kind, f.quantity, f.send) for f in faults] == [("shape", "yield_rate", True)]
    assert small_run.shape_faults(PROTOCOL, _record({"yield_rate": [1.5], "solve_error": [PER_VALUE]})) == []


# --- a declared result no trial returned --------------------------------------------------------------------------------


def test_a_declared_result_no_trial_returned_is_missing_output() -> None:
    got = small_run.missing_outputs(PROTOCOL, _record({"solve_error": [PER_VALUE], "pressure": [1.0]}))
    assert [(f.kind, f.quantity, f.send) for f in got] == [("missing_output", "yield_rate", True)]
    # a diagnostic, a result worked out afterwards and a setting a derived quantity reads are never missing
    ok = small_run.missing_outputs(PROTOCOL, _record({"yield_rate_values": [[1.0]], "solve_error": [PER_VALUE]}))
    assert ok == [], "a result returned under one of its forms is returned"
    # a small run that did not run every setting to the end makes no claim about what was never returned
    assert small_run.missing_outputs(PROTOCOL, _record({"solve_error": [PER_VALUE]}, whole=False)) == []


def test_a_derived_input_no_trial_returned_is_missing_output() -> None:
    protocol = {**PROTOCOL, "derived": [{"id": "spread", "inputs": ["peak_load", "best_yield", "temperature"]}]}
    got = small_run.missing_outputs(protocol, _record({"yield_rate": [1.0], "solve_error": [PER_VALUE]}))
    assert [(f.quantity, f.send) for f in got] == [("peak_load", True)]
    assert "`spread` is computed from `peak_load`" in got[0].why


# --- the search's own judgement of its baseline -------------------------------------------------------------------------


def _judged(baseline: dict[str, Any], returned: dict[str, list[Any]] | None = None) -> list[tuple[str, str, bool]]:
    rec = _record(returned if returned is not None else {k: [v] for k, v in baseline.items()}, baseline=baseline)
    return [(f.kind, f.quantity, f.send) for f in small_run.judge_baseline(PROTOCOL["optimisation"], rec)]


def test_the_objective_and_each_constraint_are_judged_on_their_own() -> None:
    assert _judged({"yield_rate": 1.0, "pressure": 1.0, "stress": 1.0}) == []
    # the objective returned per value; each constraint is still judged, and only the bad one is named
    assert _judged({"yield_rate": PER_VALUE, "pressure": 1.0, "stress": PER_VALUE}) == [
        ("objective", "yield_rate", True), ("constraint", "stress", True)]
    # a constraint that breaks its limit is a design, not a fault
    assert _judged({"yield_rate": 1.0, "pressure": 9.0, "stress": 1.0}) == []


def test_a_value_the_baseline_does_not_give_is_a_line_unless_no_design_gave_it() -> None:
    nan = float("nan")
    assert _judged({"yield_rate": nan, "pressure": 1.0, "stress": 1.0}) == [("objective", "yield_rate", False)]
    # absent at the baseline and in every trial of the small run: exact
    assert _judged({"pressure": 1.0, "stress": 1.0}) == [("objective", "yield_rate", True)]
    # absent at the baseline but returned at another design of the small run: a line only
    assert _judged({"pressure": 1.0, "stress": 1.0},
                   {"yield_rate": [2.0], "pressure": [1.0], "stress": [1.0]}) == [("objective", "yield_rate", False)]


# --- FI's analysis on the small run's trials ----------------------------------------------------------------------------


def test_the_analysis_is_split_by_evidence() -> None:
    hard = gate.GateResult(False, "derive", gate.FAULT, "code/derive.py, in derive_spread", "x", "NameError",
                           body="derive_spread")
    got = small_run.analysis_split(hard)
    assert got.send and got.script == "derive.py" and got.body == "derive_spread"
    maybe = gate.GateResult(False, "derive", gate.FAULT, "code/derive.py, in derive_spread", "x", "TypeError",
                            body="derive_spread")
    assert not small_run.analysis_split(maybe).send, "three trials can raise a TypeError: the study's record decides"
    assert not small_run.analysis_split(gate.GateResult(False, "facts", gate.SMALL_DATA, "", "no value")).send
    docker = small_run.analysis_split(gate.GateResult(False, "static", gate.NOT_CHECKED, "", "d"), docker=True)
    assert not docker.send and "not checked" in docker.why and "Docker" in docker.why
    assert small_run.analysis_split(gate.GateResult(True)) is None
    lines = small_run.no_value([{"id": "yield_spread", "status": "missing", "why": "the fit needs four points"}])
    assert [(f.send, f.quantity) for f in lines] == [(False, "no_value:yield_spread")]
    assert "the fit needs four points" in lines[0].why


def test_the_analysis_runs_on_the_small_runs_trials(tmp_path: Path) -> None:
    """Before the long run the quest has no raw/ yet: the gate runs on the small run's own record of its trials."""
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "experiment.py").write_text(textwrap.dedent("""
        import json, os
        cells = json.load(open(os.environ["FI_TRIALS"]))["cells"]
        print("RESULT_JSON: " + json.dumps({"n": len(cells)}))
    """), encoding="utf-8")
    raw = root / small_run.TRIALS_PATH
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"schema": "fi.trials/v1", "cells": [
        {"cell": {"temperature": 3}, "key": "temperature=3", "planned": 1, "ok": 1, "failed": 0,
         "metrics": {"yield_rate": {"values": [1.0], "trials": [0], "count": 1, "total": 1.0, "non_finite": 0}}}]}),
        encoding="utf-8")
    without = asyncio.run(gate.analysis_gate(root, root / "code", python=sys.executable))
    assert not without.ok and "no recorded raw" in without.detail
    got = asyncio.run(gate.analysis_gate(root, root / "code", python=sys.executable, raw_dir=raw.parent))
    assert got.ok, got.says()


def test_the_small_runs_trials_are_recorded_as_a_studys() -> None:
    run_ = small_run.SmallRun([small_run.Setting("temperature=3", 1, ok=1, cell={"temperature": 3},
                                                 levels=in_run.level_cells(INSIDE),
                                                 rows=[{"trial": 0, "status": "ok",
                                                        "values": {"yield_rate": 1.0, "solve_error": PER_VALUE}}])])
    cells = run_.trials()["cells"]
    assert [c["key"] for c in cells] == ["temperature=3", "temperature=3|solver=fast", "temperature=3|solver=exact"] or \
        len(cells) == 3
    assert cells[0]["metrics"]["yield_rate"]["values"] == [1.0]
    assert {tuple(c.get("level", {}).items()) for c in cells[1:]} == {(("solver", "fast"),), (("solver", "exact"),)}


# --- the one repair rule ------------------------------------------------------------------------------------------------


def test_the_ledger_of_send_backs_outlives_a_new_small_run(tmp_path: Path) -> None:
    run_ = small_run.SmallRun([small_run.Setting("a", 1, ok=1)])
    small_run.write_record(tmp_path, "before", run_)
    small_run.update_record(tmp_path, **{small_run.SENT: {"faults": ["shape:yield_rate"], "count": 1}})
    small_run.write_record(tmp_path, "after", run_)  # the repaired code: a new key
    rec = small_run.read_record(tmp_path)
    assert rec["key"] == "after" and small_run.sent(rec) == {"faults": ["shape:yield_rate"], "count": 1}


def _stand(tmp_path: Path, cap: int = 3) -> Any:
    import logging
    import types

    from core.run_checks import RunChecksMixin

    class Stand(RunChecksMixin):
        def __init__(self) -> None:
            self.quest_root = tmp_path
            self.fi_dir = tmp_path / ".fi"
            self._log = logging.getLogger("test")
            self.config = types.SimpleNamespace(engine=types.SimpleNamespace(exec_reflect_max_iterations=cap),
                                                execution=types.SimpleNamespace(sandbox="venv", timeout_s=60))
            self._small_run_note = ""

        def _protocol_block(self, state: Any) -> Any:
            return PROTOCOL

        def _code_review_failed_run(self, message: str, *, give_up: str = "", script: str | None = None) -> Any:
            return {"exec_result": {"returncode": 1, "stderr_tail": message, "failed_script": script}}

    (tmp_path / ".fi").mkdir(exist_ok=True)
    (tmp_path / "plan.md").write_text("# Plan\n", encoding="utf-8")
    return Stand()


def _write(tmp_path: Path, values: dict[str, list[Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    (tmp_path / ".fi").mkdir(exist_ok=True)
    rec = {"key": "k", **_record(values, baseline=baseline)}
    (tmp_path / small_run.RECORD).write_text(json.dumps(rec), encoding="utf-8")
    return rec


def test_a_fault_is_sent_back_once_then_said_and_the_quest_goes_on(tmp_path: Path) -> None:
    eng = _stand(tmp_path)
    rec = _write(tmp_path, {"yield_rate": [PER_VALUE], "solve_error": [PER_VALUE], "pressure": [1.0], "stress": [1.0]},
                 {"yield_rate": PER_VALUE, "pressure": 1.0, "stress": 1.0})
    # the crash repairs are spent: this send-back has its own counter and still goes
    sent = asyncio.run(eng._small_run_checks({"exec_reflect_iter": 99}, rec))
    assert sent is not None and sent["exec_result"]["failed_script"] == "simulate.py"
    assert sent["exec_result"]["gate_repair"] is True, "it spends no crash repair"
    assert "objective `yield_rate`" in sent["exec_result"]["stderr_tail"]
    assert sent["exec_result"]["stderr_tail"].count("yield_rate`") >= 1
    assert small_run.sent(small_run.read_record(tmp_path)) == {"faults": ["per_value:yield_rate"], "count": 1}
    # the repair kept the fault (any code version): never sent again, a line in plan.md and for the code's reader
    again = asyncio.run(eng._small_run_checks({}, small_run.read_record(tmp_path)))
    assert again is None
    assert "sent back to be repaired once" in (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "objective `yield_rate`" in eng._small_run_note


def test_no_send_back_is_left_with_the_cap_at_zero(tmp_path: Path) -> None:
    eng = _stand(tmp_path, cap=0)
    rec = _write(tmp_path, {"yield_rate": [PER_VALUE], "solve_error": [PER_VALUE], "pressure": [1.0], "stress": [1.0]},
                 {"yield_rate": PER_VALUE, "pressure": 1.0, "stress": 1.0})
    assert asyncio.run(eng._small_run_checks({}, rec)) is None
    assert "No send-back of the small run is left" in (tmp_path / "plan.md").read_text(encoding="utf-8")


def test_a_clean_small_run_says_nothing(tmp_path: Path) -> None:
    eng = _stand(tmp_path)
    rec = _write(tmp_path, {"yield_rate": [1.0], "solve_error": [PER_VALUE], "pressure": [1.0], "stress": [1.0]},
                 {"yield_rate": 1.0, "pressure": 1.0, "stress": 1.0})
    assert asyncio.run(eng._small_run_checks({}, rec)) is None
    assert eng._small_run_note == "" and (tmp_path / "plan.md").read_text(encoding="utf-8") == "# Plan\n"


@pytest.mark.parametrize("value", [float("nan")])
def test_an_objective_with_no_value_at_the_baseline_is_a_line_only(tmp_path: Path, value: float) -> None:
    eng = _stand(tmp_path)
    rec = _write(tmp_path, {"yield_rate": [value], "solve_error": [PER_VALUE], "pressure": [1.0], "stress": [1.0]},
                 {"yield_rate": value, "pressure": 1.0, "stress": 1.0})
    assert asyncio.run(eng._small_run_checks({}, rec)) is None
    assert "not a finite number" in (tmp_path / "plan.md").read_text(encoding="utf-8")


# --- what the first review found ----------------------------------------------------------------------------------------


def test_one_fault_found_by_two_checks_is_sent_back_once(tmp_path: Path) -> None:
    """The baseline's judgement sent back the objective returned per value; after the repair the baseline gives one number
    but another setting of the small run still returns it per value: the shape check names the same fault, not a new one."""
    eng = _stand(tmp_path)
    rec = _write(tmp_path, {"yield_rate": [PER_VALUE], "solve_error": [PER_VALUE], "pressure": [1.0], "stress": [1.0]},
                 {"yield_rate": PER_VALUE, "pressure": 1.0, "stress": 1.0})
    assert asyncio.run(eng._small_run_checks({}, rec)) is not None
    led = small_run.sent(small_run.read_record(tmp_path))
    assert led == {"faults": ["per_value:yield_rate"], "count": 1}
    rec = _write(tmp_path, {"yield_rate": [PER_VALUE], "solve_error": [PER_VALUE], "pressure": [1.0], "stress": [1.0]},
                 {"yield_rate": 1.0, "pressure": 1.0, "stress": 1.0})
    small_run.update_record(tmp_path, **{small_run.SENT: led})  # the repaired code's small run keeps the ledger
    assert asyncio.run(eng._small_run_checks({}, small_run.read_record(tmp_path))) is None
    assert "sent back to be repaired once" in (tmp_path / "plan.md").read_text(encoding="utf-8")


def test_a_failed_trial_leaves_no_claim_about_a_name_never_returned() -> None:
    rec = _record({"solve_error": [PER_VALUE]})
    rec["settings"][0].update(planned=3, ok=2, failed=1)
    assert small_run.missing_outputs(PROTOCOL, rec) == []


def test_the_small_run_is_made_again_for_another_baseline(tmp_path: Path) -> None:
    sim = tmp_path / "simulate.py"
    sim.write_text("def run_cell(cell):\n    return {'y': 1.0}\n", encoding="utf-8")
    one = small_run.key_of(sim, {"x": [1]}, True, None, None, {"x": 1.0})
    assert one != small_run.key_of(sim, {"x": [1]}, True, None, None, {"x": 2.0})
    assert small_run.key_of(sim, {"x": [1]}, True, None) == small_run.key_of(sim, {"x": [1]}, True, None, None, None)


def test_a_fault_of_derive_py_outside_its_functions_is_sent_to_derive_py(tmp_path: Path) -> None:
    """A fault of a statement of derive.py outside its functions names no function: the send-back still says derive.py
    (``failed_body`` ""), so the repair never looks for the function in a record of the analysis the quest has not made."""
    eng = _stand(tmp_path)
    rec = _write(tmp_path, {"yield_rate": [1.0], "solve_error": [PER_VALUE], "pressure": [1.0], "stress": [1.0]},
                 {"yield_rate": 1.0, "pressure": 1.0, "stress": 1.0})

    async def analysis(state: Any, protocol: Any, record: Any, *, search: bool) -> list[Any]:
        return [small_run.analysis_split(gate.GateResult(False, "derive", gate.FAULT, "code/derive.py", "x", "NameError"))]

    eng._small_run_analysis = analysis
    sent = asyncio.run(eng._small_run_checks({}, rec))
    assert sent is not None and sent["exec_result"]["failed_script"] == "derive.py"
    assert sent["exec_result"]["failed_body"] == ""
