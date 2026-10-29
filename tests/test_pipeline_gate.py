"""The analysis gate (core/pipeline_gate.py): a change is refused when the analysis, run from start to end on a small copy
of the quest's own raw, has a fault the code before the change did not have. Neutral made-up study: a "tank" whose
fill time is measured at a few pump rates."""

from __future__ import annotations

import json
from pathlib import Path

from core import pipeline_gate as gate

OK_SCRIPT = '''
import json, os

def summarise(cells):
    return {c["key"]: sum(c["metrics"]["fill_s"]["values"]) for c in cells}

def main():
    data = json.load(open(os.environ["FI_TRIALS"]))
    print("RESULT_JSON: " + json.dumps(summarise(data["cells"])))

if __name__ == "__main__":
    main()
'''


def make_quest(root: Path, trials: int = 5, settings: int = 3) -> Path:
    (root / "raw").mkdir(parents=True)
    cells = []
    for i in range(settings):
        vals = [10.0 + i + t * 0.1 for t in range(trials)]
        cells.append({"cell": {"rate": i + 1}, "key": f"rate={i + 1}", "planned": trials, "ok": trials, "failed": 0,
                      "metrics": {"fill_s": {"values": vals, "count": trials, "total": sum(vals), "non_finite": 0}}})
    (root / "raw" / "trials.json").write_text(json.dumps({"schema": "x", "cells": cells}), encoding="utf-8")
    return root


def make_code(root: Path, script: str) -> Path:
    code = root / "code"
    code.mkdir(parents=True)
    (code / "experiment.py").write_text(script, encoding="utf-8")
    return code


async def run_gate(tmp_path: Path, script: str, name: str = "q", **kw) -> gate.GateResult:
    quest = make_quest(tmp_path / name)
    code = make_code(tmp_path / f"{name}_code", script)
    return await gate.analysis_gate(quest, code, None, timeout_s=60, **kw)


async def test_a_script_that_runs_to_the_end_passes(tmp_path):
    got = await run_gate(tmp_path, OK_SCRIPT)
    assert got.ok and not got.kind


async def test_a_definition_after_the_main_guard_is_a_fault(tmp_path):
    script = OK_SCRIPT.replace("def summarise", "def unused").replace('summarise(data["cells"])', "later(data)")
    script += "\n\ndef later(data):\n    return {}\n"
    got = await run_gate(tmp_path, script)
    assert not got.ok and got.kind == gate.FAULT and got.stage == "static"
    assert got.late == ["experiment.py:later"] and "line" in got.where


def test_a_definition_after_the_guard_that_nothing_reaches_is_not_a_fault():
    source = OK_SCRIPT + "\n\ndef spare():\n    return 1\n"
    assert gate.late_definitions(source) == {}


def test_a_definition_after_the_guard_reached_through_a_function_is_found():
    source = OK_SCRIPT.replace("return {c", "return helper() or {c") + "\n\ndef helper():\n    return None\n"
    assert list(gate.late_definitions(source)) == ["helper"]


async def test_an_undefined_name_is_a_fault(tmp_path):
    got = await run_gate(tmp_path, OK_SCRIPT.replace("sum(", "total_of("))
    assert not got.ok and got.kind == gate.FAULT and got.stage == "derive"
    assert got.exception == "NameError" and "experiment.py" in got.where


async def test_an_error_from_too_little_data_is_small_data_and_is_not_refused(tmp_path):
    script = OK_SCRIPT.replace('print("RESULT_JSON', 'if len(data["cells"][0]["metrics"]["fill_s"]["values"]) < 4:\n'
                               '        raise ValueError("need at least 4 trials for the fit")\n    print("RESULT_JSON')
    got = await run_gate(tmp_path, script)
    assert not got.ok and got.kind == gate.SMALL_DATA and got.exception == "ValueError"
    verdict = gate.gate_verdict(gate.GateResult(True), got)
    assert not verdict.reject and verdict.disclose


async def test_a_run_past_its_time_is_time_and_is_not_refused(tmp_path):
    quest = make_quest(tmp_path / "q")
    code = make_code(tmp_path / "c", "import time\ntime.sleep(30)\n")
    got = await gate.analysis_gate(quest, code, None, timeout_s=2)
    assert got.kind == gate.TIME
    assert not gate.gate_verdict(gate.GateResult(True), got).reject


async def test_the_not_checked_record_of_a_search_reaches_the_analysis(tmp_path):
    script = ('import json, os\ncheck = json.load(open(os.environ["FI_OPTIMUM_CHECK"]))\n'
              'says = (check or {}).get("best", {}).get("says")\nprint("RESULT_JSON: {}")\n')
    got = await run_gate(tmp_path, script, search=True)
    assert not got.ok and got.kind == gate.FAULT and got.exception == "AttributeError"


def test_the_small_copy_keeps_three_trials_of_every_setting(tmp_path):
    quest = make_quest(tmp_path / "q", trials=8, settings=4)
    assert gate.small_raw(quest, tmp_path / "scratch")
    cells = json.loads((tmp_path / "scratch" / "raw" / "trials.json").read_text(encoding="utf-8"))["cells"]
    assert [c["key"] for c in cells] == [f"rate={i}" for i in (1, 2, 3, 4)]
    assert all(len(c["metrics"]["fill_s"]["values"]) == 3 and c["metrics"]["fill_s"]["count"] == 3 for c in cells)
    assert all(c["ok"] == 3 for c in cells)


def test_a_change_is_refused_when_it_passed_before_and_has_a_new_fault():
    after = gate.GateResult(False, "derive", gate.FAULT, "code/experiment.py line 9", "NameError: x", "NameError")
    assert gate.gate_verdict(gate.GateResult(True), after).reject


def test_a_change_is_not_refused_when_the_code_already_failed_before():
    before = gate.GateResult(False, "derive", gate.FAULT, "code/experiment.py line 5", "TypeError: y", "TypeError")
    after = gate.GateResult(False, "derive", gate.FAULT, "code/experiment.py line 9", "NameError: x", "NameError")
    assert not gate.gate_verdict(before, after).reject
    assert not gate.gate_verdict(None, after).reject


def test_a_definition_after_the_guard_that_the_code_before_did_not_have_is_refused_even_when_it_stopped_later():
    before = gate.GateResult(False, "derive", gate.FAULT, "", "AttributeError", "AttributeError")
    after = gate.GateResult(False, "static", gate.FAULT, "code/experiment.py line 9", "`f` is defined after", "NameError",
                            late=["experiment.py:f"])
    assert gate.gate_verdict(before, after).reject
    same = gate.GateResult(False, "static", gate.FAULT, "", "x", "NameError", late=["experiment.py:f"])
    assert not gate.gate_verdict(same, after).reject
